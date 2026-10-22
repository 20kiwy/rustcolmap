//! Descriptor matching (mutual-best + Lowe ratio, rayon-parallel) and
//! two-view geometric verification (H / F RANSAC, essential decomposition
//! with cheirality, reprojected inlier set).

use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::geom::{
    self, Cam, Pose, eight_point_f, estimate_h4, f_to_essential, h_sym_err, ransac, sampson,
    svd3, tri_angle_deg, triangulate,
};
use crate::sift::SiftFrame;

#[derive(Debug, Clone, Default)]
pub struct PairMatches {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
}

/// Match frame `a` against frame `b` with mutual-best check and ratio test.
/// Descriptors are L2-normalized; squared distance = 2 - 2*dot.
pub fn match_pair(fa: &SiftFrame, fb: &SiftFrame, ratio: f64) -> PairMatches {
    if fa.desc.is_empty() || fb.desc.is_empty() {
        return PairMatches::default();
    }
    // Forward pass: best + second best of b for each descriptor of a.
    let fwd: Vec<(u32, f32, f32)> = fa
        .desc
        .par_iter()
        .map(|da| {
            let mut best = (f32::INFINITY, u32::MAX);
            let mut second = f32::INFINITY;
            for (ib, db) in fb.desc.iter().enumerate() {
                let mut s = 0f32;
                for k in 0..128 {
                    let d = da[k] - db[k];
                    s += d * d;
                }
                if s < best.0 {
                    second = best.0;
                    best = (s, ib as u32);
                } else if s < second {
                    second = s;
                }
            }
            (best.1, best.0, second)
        })
        .collect();
    // Backward pass: best + second best of a for each descriptor of b
    // (used for the mutual check and a symmetric ratio test).
    let bwd: Vec<(u32, f32, f32)> = fb
        .desc
        .par_iter()
        .map(|db| {
            let mut best = (f32::INFINITY, u32::MAX);
            let mut second = f32::INFINITY;
            for (ia, da) in fa.desc.iter().enumerate() {
                let mut s = 0f32;
                for k in 0..128 {
                    let d = da[k] - db[k];
                    s += d * d;
                }
                if s < best.0 {
                    second = best.0;
                    best = (s, ia as u32);
                } else if s < second {
                    second = s;
                }
            }
            (best.1, best.0, second)
        })
        .collect();
    let mut out = PairMatches::default();
    let ratio = ratio as f32;
    for (ia, (ib, d1, d2)) in fwd.iter().enumerate() {
        if *ib == u32::MAX || !d2.is_finite() {
            continue;
        }
        if *d1 > 0.9 || *d1 > ratio * *d2 {
            continue;
        }
        let (jb, bd1, bd2) = bwd[*ib as usize];
        if jb != ia as u32 {
            continue;
        }
        // Symmetric ratio test: b's match must also be unambiguous.
        if !bd2.is_finite() || bd1 > ratio * bd2 {
            continue;
        }
        out.a.push(ia as u32);
        out.b.push(*ib);
    }
    out
}

/// Verified two-view geometry. `r`, `t` map camera-i coords to camera-j
/// coords: X_j = R X_i + t. `a`/`b` are the geometrically-inlier keypoint
/// indices into frames i and j.
pub struct TwoView {
    pub i: usize,
    pub j: usize,
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
    pub planar: bool,
    pub raw_matches: usize,
}

pub struct VerifyOptions {
    pub ratio: f64,
    /// Geometric inlier threshold in pixels.
    pub ransac_thr_px: f64,
    pub min_pair_inliers: usize,
    /// Minimum triangulation angle (degrees) for cheirality counting.
    pub min_tri_angle: f64,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        VerifyOptions {
            ratio: 0.8,
            ransac_thr_px: 4.0,
            min_pair_inliers: 30,
            min_tri_angle: 1.5,
        }
    }
}

fn norm_pts(cam: &Cam, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| cam.normalize(p[0], p[1])).collect()
}

/// Verify one matched pair, returning relative pose + inlier matches.
#[allow(clippy::too_many_arguments)]
pub fn verify_pair(
    i: usize,
    j: usize,
    fa: &SiftFrame,
    fb: &SiftFrame,
    cami: &Cam,
    camj: &Cam,
    pm: &PairMatches,
    opts: &VerifyOptions,
) -> Option<TwoView> {
    let n = pm.a.len();
    if n < 8 {
        return None;
    }
    let p1: Vec<[f64; 2]> = pm.a.iter().map(|&k| fa.pts[k as usize]).collect();
    let p2: Vec<[f64; 2]> = pm.b.iter().map(|&k| fb.pts[k as usize]).collect();
    let seed = 0x1234_5678u64 ^ (i as u64) ^ ((j as u64) << 16) ^ (n as u64);

    // --- Homography RANSAC (degeneracy / planarity test) ---
    let mut h_est = |idx: &[usize]| -> Option<Matrix3<f64>> {
        let q1: Vec<[f64; 2]> = idx.iter().map(|&k| p1[k]).collect();
        let q2: Vec<[f64; 2]> = idx.iter().map(|&k| p2[k]).collect();
        estimate_h4(&q1, &q2)
    };
    let h_res = |h: &Matrix3<f64>, k: usize| h_sym_err(h, p1[k], p2[k]);
    let h_res: Option<(Matrix3<f64>, Vec<usize>)> =
        ransac(n, 4, opts.ransac_thr_px, seed, &mut h_est, &h_res);
    let n_h = h_res.as_ref().map_or(0, |(_, v)| v.len());

    // --- Fundamental matrix RANSAC ---
    let mut f_est = |idx: &[usize]| -> Option<Matrix3<f64>> {
        let q1: Vec<[f64; 2]> = idx.iter().map(|&k| p1[k]).collect();
        let q2: Vec<[f64; 2]> = idx.iter().map(|&k| p2[k]).collect();
        eight_point_f(&q1, &q2)
    };
    let f_res = |f: &Matrix3<f64>, k: usize| sampson(f, p1[k], p2[k]);
    let (f_best, inl_f) = ransac(n, 8, opts.ransac_thr_px, seed ^ 0xdead, &mut f_est, &f_res)?;
    let n_f = inl_f.len();
    let dbg = std::env::var("SFM_DEBUG").is_ok();
    if dbg {
        eprintln!("[verify] pair ({i},{j}): n={n} H_inl={n_h} F_inl={n_f}");
    }
    if n_f < opts.min_pair_inliers {
        return None;
    }
    let planar = n_h > n_f && n_h as f64 >= 0.8 * n as f64;
    let _ = planar;

    let planar = n_h > 0 && n_h as f64 >= 0.5 * n_f as f64;

    // Candidate fundamental matrices. Planar-dominated scenes admit many F's
    // with high support; the true motion F is recovered by re-estimating on
    // the H-outlier matches (>= 2 planes + motion identify F uniquely), so
    // that candidate is ALWAYS built when enough outliers remain. The final
    // pose is chosen by cheirality scoring across all candidates.
    let mut candidates: Vec<Matrix3<f64>> = vec![f_best];
    {
        // Refine the global F on its inliers as a second candidate.
        let q1: Vec<[f64; 2]> = inl_f.iter().map(|&k| p1[k]).collect();
        let q2: Vec<[f64; 2]> = inl_f.iter().map(|&k| p2[k]).collect();
        if let Some(fr) = eight_point_f(&q1, &q2) {
            candidates.push(fr);
        }
        let hset: std::collections::HashSet<usize> =
            h_res.as_ref().map_or(Default::default(), |(_, v)| v.iter().copied().collect());
        let out_idx: Vec<usize> = (0..n).filter(|k| !hset.contains(k)).collect();
        if out_idx.len() >= 30 {
            let mut f2_est = |idx: &[usize]| -> Option<Matrix3<f64>> {
                let q1: Vec<[f64; 2]> = idx.iter().map(|&k| p1[k]).collect();
                let q2: Vec<[f64; 2]> = idx.iter().map(|&k| p2[k]).collect();
                eight_point_f(&q1, &q2)
            };
            let f2_res = |f: &Matrix3<f64>, k: usize| sampson(f, p1[k], p2[k]);
            if let Some((f2, inl2)) = ransac(
                out_idx.len(),
                8,
                opts.ransac_thr_px,
                seed ^ 0xbeef,
                &mut f2_est,
                &f2_res,
            ) {
                let gi: Vec<usize> = inl2.iter().map(|&k| out_idx[k]).collect();
                if gi.len() >= opts.min_pair_inliers {
                    let q1: Vec<[f64; 2]> = gi.iter().map(|&k| p1[k]).collect();
                    let q2: Vec<[f64; 2]> = gi.iter().map(|&k| p2[k]).collect();
                    if let Some(f2r) = eight_point_f(&q1, &q2) {
                        candidates.push(f2r);
                    }
                    candidates.push(f2);
                }
            }
        }
    }

    // Score every candidate by cheirality + reprojection, keep the best pose.
    let x1 = norm_pts(cami, &p1);
    let x2 = norm_pts(camj, &p2);
    let pose_i = Pose::identity();
    let thr2 = 2.0 * opts.ransac_thr_px;
    let mut best_pose: Option<(Pose, usize)> = None;
    for f_cand in &candidates {
        let k1 = cami.kmat();
        let k2 = camj.kmat();
        let e = match f_to_essential(f_cand, &k1, &k2) {
            Some(e) => e,
            None => continue,
        };
        for (r, t) in geom::decompose_e(&e) {
            let pose_j = Pose { r, t };
            let mut cnt = 0usize;
            for k in 0..n {
                if let Some(x) = triangulate(
                    &pose_i.r, &pose_i.t, &pose_j.r, &pose_j.t, x1[k], x2[k],
                ) {
                    let ang = tri_angle_deg(&pose_i, &pose_j, &x);
                    if ang < opts.min_tri_angle {
                        continue;
                    }
                    let e_i = pose_i.project(cami, &x);
                    let e_j = pose_j.project(camj, &x);
                    if let (Some(a), Some(b)) = (e_i, e_j) {
                        let d1 = (a.0 - p1[k][0]).hypot(a.1 - p1[k][1]);
                        let d2 = (b.0 - p2[k][0]).hypot(b.1 - p2[k][1]);
                        if d1 < thr2 && d2 < thr2 {
                            cnt += 1;
                        }
                    }
                }
            }
            if best_pose.as_ref().map_or(true, |(_, c)| cnt > *c) {
                best_pose = Some((pose_j, cnt));
            }
        }
    }
    let (pose_j, cnt) = best_pose?;
    if dbg {
        eprintln!("[verify] pair ({i},{j}): planar={planar} candidates={} best cheiral={}", candidates.len(), cnt);
    }
    if cnt < opts.min_pair_inliers {
        return None;
    }

    // Final inlier set: triangulation succeeds with angle + reprojection gates.
    let mut a = Vec::with_capacity(cnt);
    let mut b = Vec::with_capacity(cnt);
    for k in 0..n {
        if let Some(x) = triangulate(
            &pose_i.r, &pose_i.t, &pose_j.r, &pose_j.t, x1[k], x2[k],
        ) {
            let ang = tri_angle_deg(&pose_i, &pose_j, &x);
            if ang < opts.min_tri_angle {
                continue;
            }
            let e_i = pose_i.project(cami, &x);
            let e_j = pose_j.project(camj, &x);
            if let (Some(pa), Some(pb)) = (e_i, e_j) {
                let d1 = (pa.0 - p1[k][0]).hypot(pa.1 - p1[k][1]);
                let d2 = (pb.0 - p2[k][0]).hypot(pb.1 - p2[k][1]);
                if d1 < thr2 && d2 < thr2 {
                    a.push(pm.a[k]);
                    b.push(pm.b[k]);
                }
            }
        }
    }
    if a.len() < opts.min_pair_inliers {
        return None;
    }
    Some(TwoView {
        i,
        j,
        a,
        b,
        r: pose_j.r,
        t: pose_j.t,
        planar,
        raw_matches: n,
    })
}

/// PnP-RANSAC registration of one image against 2D-3D correspondences.
/// Returns (pose, inlier indices into `pixels`/`pts3d`).
pub fn pnp_ransac(
    pixels: &[[f64; 2]],
    pts3d: &[Vector3<f64>],
    cam: &Cam,
    thr_px: f64,
) -> Option<(Pose, Vec<usize>)> {
    let n = pixels.len();
    if n < 6 {
        return None;
    }
    let mut est = |idx: &[usize]| -> Option<Pose> {
        let q2: Vec<[f64; 2]> = idx.iter().map(|&k| pixels[k]).collect();
        let q3: Vec<Vector3<f64>> = idx.iter().map(|&k| pts3d[k]).collect();
        geom::pnp_dlt(&q2, &q3, cam)
    };
    let resid = |p: &Pose, k: usize| -> f64 {
        match p.project(cam, &pts3d[k]) {
            Some((u, v)) => (u - pixels[k][0]).hypot(v - pixels[k][1]),
            None => f64::INFINITY,
        }
    };
    let seed = 0xfeed_beefu64 ^ (n as u64);
    let (pose, inl) = ransac(n, 6, thr_px, seed, &mut est, &resid)?;
    // One refinement pass on all inliers.
    let q2: Vec<[f64; 2]> = inl.iter().map(|&k| pixels[k]).collect();
    let q3: Vec<Vector3<f64>> = inl.iter().map(|&k| pts3d[k]).collect();
    if let Some(p2) = geom::pnp_dlt(&q2, &q3, cam) {
        let mut inl2: Vec<usize> = (0..n)
            .filter(|&k| {
                p2.project(cam, &pts3d[k])
                    .map_or(false, |(u, v)| {
                        (u - pixels[k][0]).hypot(v - pixels[k][1]) < thr_px
                    })
            })
            .collect();
        if inl2.len() >= inl.len() {
            if inl2.len() > inl.len() {
                inl2 = (0..n)
                    .filter(|&k| {
                        p2.project(cam, &pts3d[k])
                            .map_or(false, |(u, v)| {
                                (u - pixels[k][0]).hypot(v - pixels[k][1]) < thr_px
                            })
                    })
                    .collect();
            }
            return Some((p2, inl2));
        }
    }
    Some((pose, inl))
}

/// Decomposed covariance-free re-check: recompute F from E for debugging.
#[allow(dead_code)]
fn e_to_f(e: &Matrix3<f64>, k1: &Matrix3<f64>, k2: &Matrix3<f64>) -> Matrix3<f64> {
    let inv1 = k1.try_inverse().unwrap_or(Matrix3::identity());
    let inv2 = k2.try_inverse().unwrap_or(Matrix3::identity());
    let mut f = inv2.transpose() * e * inv1;
    if let Some((u, sv, v)) = svd3(&f) {
        f = u * Matrix3::from_diagonal(&Vector3::new(sv[0], sv[1], 0.0)) * v.transpose();
    }
    f
}
