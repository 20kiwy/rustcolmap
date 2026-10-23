//! Patch-match stereo on CPU: per-view depth maps from 2..K support views.
//!
//! Cost model   : 7x7 census descriptor (48 bits), aggregated over a 5x5
//!                window, averaged over the support views (invalid warps
//!                count as full mismatch so off-screen planes are rejected).
//! Geometry     : slanted planes. Each pixel stores a unit normal n and a
//!                depth z at the pixel center; the supporting plane is
//!                n . X = D with D = z * (n . r_hat(pixel)), and the depth
//!                at any other pixel q is z_q = D / (n . r_hat(q)).
//! Init         : plane sweep over `num_hyp` fronto-parallel depths,
//!                uniformly spaced in inverse depth.
//! Refinement   : red-black propagation sweeps implemented as double-buffered
//!                passes (read `prev`, write `next` — race-free, deterministic,
//!                borrow-checker clean) + 3 random perturbations per pass
//!                with a decaying schedule. All randomness derives from a
//!                per-pixel xorshift seeded by (x, y, pass) — bit-reproducible.

use std::sync::Arc;

use rayon::prelude::*;

use super::{MvsOptions, ViewData};

#[derive(Clone, Copy, Default)]
struct Plane {
    nx: f32,
    ny: f32,
    nz: f32,
    z: f32,
    cost: f32,
}

pub struct DepthMap {
    /// Depth (z in the reference camera frame) per pixel; 0 = invalid.
    pub depth: Vec<f32>,
    /// Census hamming fraction at the accepted plane.
    pub cost: Vec<f32>,
}

const BITS: f32 = 48.0;
const WIN: i32 = 2; // 5x5 aggregation window

// ---------------------------------------------------------------------------
// Deterministic per-pixel RNG (xorshift64 + cheap Irwin-Hall gaussian)
// ---------------------------------------------------------------------------

#[inline]
fn pixel_rng(x: i32, y: i32, salt: u64) -> u64 {
    let mut s = (x as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ salt.wrapping_mul(0x165667B19E3779F9);
    s ^= s >> 30;
    s = s.wrapping_mul(0xBF58476D1CE4E5B9);
    s ^= s >> 27;
    s = s.wrapping_mul(0x94D049BB133111EB);
    s ^ (s >> 31)
}

#[inline]
fn next_u64(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

#[inline]
fn uniform(s: &mut u64) -> f32 {
    (next_u64(s) >> 40) as f32 / (1u64 << 24) as f32
}

/// Cheap gaussian-ish sample (Irwin-Hall(3), mean 0, sigma ~= 0.5).
#[inline]
fn gauss(s: &mut u64) -> f32 {
    uniform(s) + uniform(s) + uniform(s) - 1.5
}

// ---------------------------------------------------------------------------
// Per-reference-view workspace
// ---------------------------------------------------------------------------

struct SourceCtx {
    census: *const u64,
    w: i32,
    h: i32,
    f: f32,
    cx: f32,
    cy: f32,
    k: f32,
    /// R_s * R_r^T (row-major 3x3): ref-cam coords -> src-cam coords.
    rs: [f32; 9],
    /// t_s - RS * t_r.
    ts: [f32; 3],
}

// SAFETY: the raw pointer targets the census buffer of an `Arc<ViewData>`
// that outlives the whole stereo pass and is never mutated.
unsafe impl Send for SourceCtx {}
unsafe impl Sync for SourceCtx {}

struct RefWorkspace<'a> {
    refv: &'a ViewData,
    /// normalized rays (x, y; z = 1) per pixel
    rx: Vec<f32>,
    ry: Vec<f32>,
    sources: Vec<SourceCtx>,
    zmin: f32,
    zmax: f32,
}

impl<'a> RefWorkspace<'a> {
    fn new(refv: &'a ViewData, views: &[Arc<ViewData>], srcs: &[usize], zmin: f64, zmax: f64) -> Self {
        let n = (refv.w * refv.h) as usize;
        let mut rx = vec![0f32; n];
        let mut ry = vec![0f32; n];
        for y in 0..refv.h {
            for x in 0..refv.w {
                let i = (y * refv.w + x) as usize;
                let c = refv.cam.normalize(x as f64, y as f64);
                rx[i] = c[0] as f32;
                ry[i] = c[1] as f32;
            }
        }
        let rt = refv.pose;
        let sources = srcs
            .iter()
            .map(|&si| {
                let sv = &views[si];
                let rs64 = sv.pose.r * rt.r.transpose();
                let ts64 = sv.pose.t - rs64 * rt.t;
                SourceCtx {
                    census: sv.census.as_ptr(),
                    w: sv.w,
                    h: sv.h,
                    f: sv.cam.f as f32,
                    cx: sv.cam.cx as f32,
                    cy: sv.cam.cy as f32,
                    k: sv.cam.k as f32,
                    rs: [
                        rs64[(0, 0)] as f32, rs64[(0, 1)] as f32, rs64[(0, 2)] as f32,
                        rs64[(1, 0)] as f32, rs64[(1, 1)] as f32, rs64[(1, 2)] as f32,
                        rs64[(2, 0)] as f32, rs64[(2, 1)] as f32, rs64[(2, 2)] as f32,
                    ],
                    ts: [ts64[0] as f32, ts64[1] as f32, ts64[2] as f32],
                }
            })
            .collect();
        RefWorkspace { refv, rx, ry, sources, zmin: zmin as f32, zmax: zmax as f32 }
    }
}

// ---------------------------------------------------------------------------
// Cost evaluation
// ---------------------------------------------------------------------------

/// Cost of a slanted plane (n, d) at pixel (px, py), in [0, ~1].
/// `hint` enables early abort: bails out once the partial sum proves the
/// plane cannot beat the hint.
#[inline]
fn plane_cost(ws: &RefWorkspace, px: i32, py: i32, nx: f32, ny: f32, nz: f32, d: f32, hint: f32) -> f32 {
    let w = ws.refv.w;
    let h = ws.refv.h;
    let i = (py * w + px) as usize;
    let nsrc = ws.sources.len() as f32;
    let span = (WIN * 2 + 1) as f32;
    let total = span * span * nsrc;
    let mut sum = 0f32;
    let mut aborted = false;
    let ref_cen = &ws.refv.census;
    'src: for (si, s) in ws.sources.iter().enumerate() {
        let sw = s.w;
        let sh = s.h;
        let (f, cx, cy, k) = (s.f, s.cx, s.cy, s.k);
        let rs = &s.rs;
        let ts = &s.ts;
        for dy in -WIN..=WIN {
            let yy = py + dy;
            let ycl = yy.clamp(0, h - 1);
            for dx in -WIN..=WIN {
                let xx = px + dx;
                let xcl = xx.clamp(0, w - 1);
                let j = (ycl * w + xcl) as usize;
                let xo = ws.rx[j];
                let yo = ws.ry[j];
                let denom = nx * xo + ny * yo + nz;
                if denom < 0.05 {
                    sum += 1.0;
                    continue;
                }
                let z = d / denom;
                let xr = z * xo;
                let yr = z * yo;
                let xs = rs[0] * xr + rs[1] * yr + rs[2] * z + ts[0];
                let ys = rs[3] * xr + rs[4] * yr + rs[5] * z + ts[1];
                let zs = rs[6] * xr + rs[7] * yr + rs[8] * z + ts[2];
                if zs < 1e-3 {
                    sum += 1.0;
                    continue;
                }
                let xn = xs / zs;
                let yn = ys / zs;
                let dist = 1.0 + k * (xn * xn + yn * yn);
                let u = (f * xn * dist + cx) as i32;
                let v = (f * yn * dist + cy) as i32;
                if u < 0 || v < 0 || u >= sw || v >= sh {
                    sum += 1.0;
                    continue;
                }
                let csrc = unsafe { *s.census.add((v * sw + u) as usize) };
                let cref = ref_cen[j];
                sum += (cref ^ csrc).count_ones() as f32 / BITS;
            }
            // early abort: partial average clearly worse than hint?
            let rows_done = (dy + WIN + 1) as f32;
            if sum / (rows_done * span * (si + 1) as f32) > hint + 0.12 {
                aborted = true;
                break 'src;
            }
        }
    }
    if aborted {
        // must not return a partial (under-)estimate — report as clearly worse
        return 1e9;
    }
    sum / total
}

// ---------------------------------------------------------------------------
// Depth estimation for one reference view
// ---------------------------------------------------------------------------

pub fn estimate_depth(
    views: &[Arc<ViewData>],
    ref_idx: usize,
    srcs: &[usize],
    zmin: f64,
    zmax: f64,
    opts: &MvsOptions,
) -> DepthMap {
    let refv = &views[ref_idx];
    let ws = RefWorkspace::new(refv, views, srcs, zmin, zmax);
    let (w, h) = (refv.w, refv.h);
    let wu = w as usize;

    // ---- plane-sweep init: fronto-parallel hypotheses in inverse depth ----
    let inv_min = 1.0 / ws.zmax;
    let inv_max = 1.0 / ws.zmin;
    let nh = opts.num_hyp.max(2) as i32;
    let mut prev: Vec<Plane> = vec![Plane::default(); wu * h as usize];
    prev.par_chunks_mut(wu)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, p) in row.iter_mut().enumerate() {
                let mut best_c = f32::INFINITY;
                let mut best_d = 0f32;
                for hi in 0..nh {
                    let inv = inv_min + (inv_max - inv_min) * hi as f32 / (nh - 1) as f32;
                    let d = 1.0 / inv;
                    let c = plane_cost(&ws, x as i32, y as i32, 0.0, 0.0, 1.0, d, best_c + 0.02);
                    if c < best_c {
                        best_c = c;
                        best_d = d;
                    }
                }
                *p = Plane { nx: 0.0, ny: 0.0, nz: 1.0, z: best_d, cost: best_c.min(1.0) };
            }
        });

    // ---- double-buffered patch-match sweeps ----
    let n_passes = (opts.sweeps.max(1) * 2) as u64;
    let mut next = prev.clone();
    for pass in 0..n_passes {
        let t = pass as f32 / n_passes as f32; // 0..1 progress
        let sig_inv = 0.02 * 0.28f32.powf(t * 4.0); // inverse-depth perturbation (fraction of range)
        let sig_n = 0.30 * 0.42f32.powf(t * 4.0); // normal perturbation
        next.copy_from_slice(&prev); // every pixel is rewritten or carried over
        next.par_chunks_mut(wu)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as i32;
                for (x, out) in row.iter_mut().enumerate() {
                    let x = x as i32;
                    let i = (y * w + x) as usize;
                    let mut seed = pixel_rng(x, y, pass + 7);
                    let cur = &prev[i];
                    let mut best_c = cur.cost;
                    let mut bn = [cur.nx, cur.ny, cur.nz];
                    let mut bd = cur.z * (bn[0] * ws.rx[i] + bn[1] * ws.ry[i] + bn[2]);

                    // propagate from 4 neighbors (read-only on prev)
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let xx = x + dx;
                        let yy = y + dy;
                        if xx < 0 || yy < 0 || xx >= w || yy >= h {
                            continue;
                        }
                        let j = (yy * w + xx) as usize;
                        let nb = &prev[j];
                        let nbn = [nb.nx, nb.ny, nb.nz];
                        let a_nb = nbn[0] * ws.rx[j] + nbn[1] * ws.ry[j] + nbn[2];
                        if a_nb < 0.1 {
                            continue;
                        }
                        let d_plane = nb.z * a_nb;
                        let a_here = nbn[0] * ws.rx[i] + nbn[1] * ws.ry[i] + nbn[2];
                        if a_here < 0.1 {
                            continue;
                        }
                        let zcand = d_plane / a_here;
                        if !(ws.zmin * 0.5..=ws.zmax * 2.0).contains(&zcand) {
                            continue;
                        }
                        let c = plane_cost(&ws, x, y, nbn[0], nbn[1], nbn[2], d_plane, best_c + 0.02);
                        if c < best_c {
                            best_c = c;
                            bn = nbn;
                            bd = d_plane;
                        }
                    }

                    // random perturbations around the current plane
                    for _ in 0..3 {
                        let inv = 1.0 / cur.z.max(1e-6);
                        let inv_new = inv + sig_inv * (inv_max - inv_min) * gauss(&mut seed);
                        if inv_new <= 0.0 {
                            continue;
                        }
                        let zcand = 1.0 / inv_new;
                        if !(ws.zmin * 0.5..=ws.zmax * 2.0).contains(&zcand) {
                            continue;
                        }
                        let nnx = cur.nx + sig_n * gauss(&mut seed);
                        let nny = cur.ny + sig_n * gauss(&mut seed);
                        let nnz = cur.nz + sig_n * gauss(&mut seed);
                        let len = (nnx * nnx + nny * nny + nnz * nnz).sqrt();
                        if len < 1e-6 {
                            continue;
                        }
                        let (nnx, nny, nnz) = (nnx / len, nny / len, nnz / len);
                        let a_here = nnx * ws.rx[i] + nny * ws.ry[i] + nnz;
                        if a_here < 0.1 {
                            continue;
                        }
                        let d_plane = zcand * a_here;
                        let c = plane_cost(&ws, x, y, nnx, nny, nnz, d_plane, best_c + 0.02);
                        if c < best_c {
                            best_c = c;
                            bn = [nnx, nny, nnz];
                            bd = d_plane;
                        }
                    }

                    if best_c < cur.cost {
                        let a_here = bn[0] * ws.rx[i] + bn[1] * ws.ry[i] + bn[2];
                        if a_here > 0.05 {
                            let z_new = bd / a_here;
                            *out = Plane {
                                nx: bn[0],
                                ny: bn[1],
                                nz: bn[2],
                                z: z_new,
                                cost: best_c.min(1.0),
                            };
                        }
                    }
                }
            });
        std::mem::swap(&mut prev, &mut next);
    }

    // ---- validity: texture gate + cost threshold + depth clamp ----
    // A census descriptor with (almost) no set bits means the neighborhood
    // is perfectly flat (sky, overexposed walls): every depth hypothesis
    // "matches" trivially there, so such pixels carry no geometric signal
    // and produce streak artifacts in fusion. Gate them out.
    for (i, p) in prev.iter_mut().enumerate() {
        let tex = ws.refv.census[i].count_ones();
        let textured = tex >= 4 && tex <= 44;
        let ok = textured
            && p.cost < opts.cost_thr
            && p.z > ws.zmin * 0.5
            && p.z < ws.zmax * 2.0;
        if !ok {
            p.z = 0.0;
        }
    }

    DepthMap {
        depth: prev.iter().map(|p| p.z).collect(),
        cost: prev.iter().map(|p| p.cost).collect(),
    }
}
