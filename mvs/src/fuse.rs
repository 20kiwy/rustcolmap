//! Depth-map fusion: geometric consistency across views, voxel-hash
//! deduplication and colored point-cloud output (binary PLY).
//!
//! For every valid pixel of every reference view we reproject the 3D point
//! into the view's support views; a support view *confirms* the point when
//! its own (already computed) depth map holds the same surface within
//! `depth_diff` (relative). Points confirmed by >= `min_consistent` other
//! views are averaged and inserted into a voxel hash so overlapping depth
//! maps contribute one clean surface sample instead of duplicates.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use nalgebra::Vector3;
use rayon::prelude::*;

use super::stereo::DepthMap;
use super::{MvsOptions, ViewData};

struct FusedPoint {
    x: f64,
    y: f64,
    z: f64,
    r: u16,
    g: u16,
    b: u16,
    n: u32,
}

/// Bilinear depth sample with invalid-aware averaging (0 = no data).
#[inline]
fn sample_depth(depth: &[f32], w: i32, h: i32, u: f64, v: f64) -> f64 {
    let u0 = u.floor() as i32;
    let v0 = v.floor() as i32;
    if u0 < 0 || v0 < 0 || u0 + 1 >= w || v0 + 1 >= h {
        return 0.0;
    }
    let fx = (u - u0 as f64) as f32;
    let fy = (v - v0 as f64) as f32;
    let idx = |x: i32, y: i32| depth[(y * w + x) as usize];
    let (d00, d10, d01, d11) = (idx(u0, v0), idx(u0 + 1, v0), idx(u0, v0 + 1), idx(u0 + 1, v0 + 1));
    let mut acc = 0f32;
    let mut wsum = 0f32;
    for (d, wt) in [
        (d00, (1.0 - fx) * (1.0 - fy)),
        (d10, fx * (1.0 - fy)),
        (d01, (1.0 - fx) * fy),
        (d11, fx * fy),
    ] {
        if d > 0.0 {
            acc += d * wt;
            wsum += wt;
        }
    }
    if wsum > 1e-6 {
        (acc / wsum) as f64
    } else {
        0.0
    }
}

pub fn fuse_depth_maps(
    views: &[Arc<ViewData>],
    depth_maps: &[Option<DepthMap>],
    sources: &[Vec<usize>],
    sparse: &[Vector3<f64>],
    opts: &MvsOptions,
    log: &(dyn Fn(&str) + Sync),
) -> Vec<(f64, f64, f64, [u8; 3])> {
    // Voxel cell size from a ROBUST sparse bounding box (per-axis 2%/98%
    // percentiles so far-flung outlier points don't inflate the cell).
    let mut axis: Vec<Vec<f64>> = vec![Vec::with_capacity(sparse.len()); 3];
    for p in sparse {
        for a in 0..3 {
            axis[a].push(p[a]);
        }
    }
    let mut ext = [0.0f64; 3];
    for a in 0..3 {
        axis[a].sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        let lo = axis[a][((axis[a].len() as f64 * 0.02) as usize).min(axis[a].len() - 1)];
        let hi = axis[a][((axis[a].len() as f64 * 0.98) as usize).min(axis[a].len() - 1)];
        ext[a] = hi - lo;
    }
    let diag = (ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]).sqrt();
    let cell = (diag * opts.voxel_rel).max(1e-9);
    log(&format!(
        "mvs: fusion voxel cell {:.4} (robust sparse diag {:.2})",
        cell, diag
    ));

    // Per-view candidate points (parallel), then merge sequentially so the
    // accumulation order (and therefore the output) is deterministic.
    let per_view: Vec<Vec<[f64; 6]>> = (0..views.len())
        .into_par_iter()
        .map(|ri| {
            let dm = match &depth_maps[ri] {
                Some(d) => d,
                None => return Vec::new(),
            };
            let v = &views[ri];
            let (w, h) = (v.w, v.h);
            let mut out: Vec<[f64; 6]> = Vec::new();
            let r_inv = v.pose.r.transpose();
            for py in 0..h {
                for px in 0..w {
                    let i = (py * w + px) as usize;
                    let zr = dm.depth[i] as f64;
                    if zr <= 0.0 {
                        continue;
                    }
                    let nrm = v.cam.normalize(px as f64, py as f64);
                    let ray = Vector3::new(nrm[0], nrm[1], 1.0);
                    let x_r = ray * zr;
                    let x_w = r_inv * (x_r - v.pose.t);

                    // cross-view depth consistency
                    let mut consistent: Vec<Vector3<f64>> = Vec::new();
                    for &si in &sources[ri] {
                        let sv = &views[si];
                        let x_s = sv.pose.r * x_w + sv.pose.t;
                        if x_s.z <= 1e-6 {
                            continue;
                        }
                        let (us, vs) = match sv.pose.project(&sv.cam, &x_w) {
                            Some(p) => p,
                            None => continue,
                        };
                        let zs = sample_depth(&depth_maps[si].as_ref().unwrap().depth, sv.w, sv.h, us, vs);
                        if zs <= 0.0 {
                            continue;
                        }
                        let rel = ((x_s.z - zs) / x_s.z).abs();
                        if rel < opts.depth_diff {
                            let nrm_s = sv.cam.normalize(us, vs);
                            let ray_s = Vector3::new(nrm_s[0], nrm_s[1], 1.0);
                            let x_s2 = ray_s * zs;
                            let x_w2 = sv.pose.r.transpose() * (x_s2 - sv.pose.t);
                            consistent.push(x_w2);
                        }
                    }
                    if consistent.len() < opts.min_consistent {
                        continue;
                    }
                    // average ref + consistent estimates
                    let mut acc = x_w;
                    for p in &consistent {
                        acc += p;
                    }
                    let m = (consistent.len() + 1) as f64;
                    acc /= m;
                    let ci = i * 3;
                    out.push([
                        acc[0],
                        acc[1],
                        acc[2],
                        v.rgb[ci] as f64,
                        v.rgb[ci + 1] as f64,
                        v.rgb[ci + 2] as f64,
                    ]);
                }
            }
            out
        })
        .collect();

    let total_candidates: usize = per_view.iter().map(|v| v.len()).sum();
    log(&format!(
        "mvs: {} consistency-passed candidates from {} depth maps",
        total_candidates,
        depth_maps.iter().flatten().count()
    ));

    // ---- voxel hash dedup ----
    let mut vox: HashMap<(i64, i64, i64), FusedPoint> = HashMap::with_capacity(total_candidates / 2);
    for view_pts in &per_view {
        for p in view_pts {
            let key = (
                (p[0] / cell).floor() as i64,
                (p[1] / cell).floor() as i64,
                (p[2] / cell).floor() as i64,
            );
            let e = vox.entry(key).or_insert(FusedPoint {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                r: 0,
                g: 0,
                b: 0,
                n: 0,
            });
            e.x += p[0];
            e.y += p[1];
            e.z += p[2];
            e.r += p[3] as u16;
            e.g += p[4] as u16;
            e.b += p[5] as u16;
            e.n += 1;
        }
    }

    let mut pts: Vec<(f64, f64, f64, [u8; 3])> = Vec::with_capacity(vox.len());
    for e in vox.values() {
        let n = e.n as f64;
        pts.push((
            e.x / n,
            e.y / n,
            e.z / n,
            [(e.r as f64 / n).round() as u8, (e.g as f64 / n).round() as u8, (e.b as f64 / n).round() as u8],
        ));
    }
    pts
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// Binary little-endian PLY (float x/y/z + uchar red/green/blue).
pub fn write_ply_binary(path: &Path, pts: &[(f64, f64, f64, [u8; 3])]) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    let header = format!(
        "ply\nformat binary_little_endian 1.0\ncomment rustcolmap dense MVS fusion\n\
         element vertex {}\nproperty float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        pts.len()
    );
    w.write_all(header.as_bytes())?;
    for (x, y, z, c) in pts {
        w.write_all(&(*x as f32).to_le_bytes())?;
        w.write_all(&(*y as f32).to_le_bytes())?;
        w.write_all(&(*z as f32).to_le_bytes())?;
        w.write_all(&[c[0], c[1], c[2]])?;
    }
    w.flush()
}

/// Grayscale depth-map visualization (2nd/98th percentile normalized,
/// invalid = black) saved as `<dir>/<stem>_depth.png`.
pub fn save_depth_vis(dir: &Path, name: &str, depth: &[f32], w: i32, h: i32) -> std::io::Result<()> {
    let mut ds: Vec<f32> = depth.iter().copied().filter(|&z| z > 0.0).collect();
    ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = if ds.len() >= 32 {
        (ds[(ds.len() as f64 * 0.02) as usize], ds[(ds.len() as f64 * 0.98) as usize - 1])
    } else {
        (0.0, 1.0)
    };
    let range = (hi - lo).max(1e-9);
    let mut img = image::GrayImage::new(w as u32, h as u32);
    for (i, &z) in depth.iter().enumerate() {
        let val = if z <= 0.0 {
            0u8
        } else {
            let t = ((z - lo) / range).clamp(0.0, 1.0);
            (255.0 * (1.0 - t)) as u8 // near = bright
        };
        let (x, y) = ((i as i32 % w) as u32, (i as i32 / w) as u32);
        img.put_pixel(x, y, image::Luma([val]));
    }
    let stem = Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "view".into());
    img.save(dir.join(format!("{stem}_depth.png")))
        .map_err(|e| std::io::Error::other(e.to_string()))
}
