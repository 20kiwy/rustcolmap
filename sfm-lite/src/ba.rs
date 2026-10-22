//! Levenberg-Marquardt bundle adjustment with Schur-complement point
//! elimination, following COLMAP's formulation:
//! - SIMPLE_RADIAL camera model (f, cx, cy, k) refined per camera
//! - quaternion + translation pose parametrization (world->camera)
//! - Huber loss on reprojection error
//! - fixed-camera gauge via strong prior block (first registered camera)
//!
//! All matrices are small fixed-size blocks (10 camera params, 3 point
//! params); the reduced camera system is dense but tiny (#cams * 10).

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3};

use crate::geom::{quat_point_jac, quat_to_rot};

pub const CAM_DIM: usize = 11; // qw qx qy qz tx ty tz f cx cy k

#[derive(Debug, Clone)]
pub struct BaCam {
    pub q: [f64; 4],
    pub t: [f64; 3],
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BaObs {
    pub cam: usize,
    pub pt: usize,
    pub u: f64,
    pub v: f64,
}

/// Median positive depth over all observations; observations much closer
/// than this to a camera plane are numerically degenerate (J ~ f/z blows up)
/// and are excluded from the problem entirely.
fn depth_guard(cams: &[BaCam], pts: &[[f64; 3]], obs: &[BaObs]) -> f64 {
    let mut zs: Vec<f64> = obs
        .iter()
        .filter_map(|o| {
            let c = &cams[o.cam];
            let r = quat_to_rot(&c.q);
            let xc = r * Vector3::new(pts[o.pt][0], pts[o.pt][1], pts[o.pt][2])
                + Vector3::new(c.t[0], c.t[1], c.t[2]);
            if xc.z > 0.0 {
                Some(xc.z)
            } else {
                None
            }
        })
        .collect();
    if zs.is_empty() {
        return 1e-9;
    }
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (1e-3 * zs[zs.len() / 2]).max(1e-9)
}

#[inline]
fn project(c: &BaCam, xc: &Vector3<f64>) -> Option<(f64, f64)> {
    let z = xc.z;
    if z <= 1e-9 {
        return None;
    }
    let xn = xc.x / z;
    let yn = xc.y / z;
    let r2 = xn * xn + yn * yn;
    let d = 1.0 + c.k * r2;
    Some((c.f * xn * d + c.cx, c.f * yn * d + c.cy))
}

/// Total robustified cost (sum, not mean).
pub fn cost(cams: &[BaCam], pts: &[[f64; 3]], obs: &[BaObs], delta: f64) -> f64 {
    let z_guard = depth_guard(cams, pts, obs);
    let mut total = 0.0;
    for o in obs {
        let c = &cams[o.cam];
        let r = quat_to_rot(&c.q);
        let xc = r * Vector3::new(pts[o.pt][0], pts[o.pt][1], pts[o.pt][2])
            + Vector3::new(c.t[0], c.t[1], c.t[2]);
        if xc.z < z_guard {
            continue;
        }
        if let Some((u, v)) = project(c, &xc) {
            let e = ((u - o.u).hypot(v - o.v)) as f64;
            total += if e <= delta {
                0.5 * e * e
            } else {
                delta * e - 0.5 * delta * delta
            };
        } else {
            total += 0.5 * (delta * delta) * 4.0; // soft penalty for behind-camera
        }
    }
    total
}

/// Jacobian of pixel projection w.r.t. camera-center point coordinates
/// and all camera params. Returns (jc [2x10], jp [2x3]) and the projection.
#[allow(clippy::type_complexity)]
pub fn jacobians(
    c: &BaCam,
    x: &Vector3<f64>,
) -> Option<(SMatrix<f64, 2, CAM_DIM>, [[f64; 3]; 2], (f64, f64))> {
    let r = quat_to_rot(&c.q);
    let xc = r * *x + Vector3::new(c.t[0], c.t[1], c.t[2]);
    let z = xc.z;
    if z <= 1e-9 {
        return None;
    }
    let xn = xc.x / z;
    let yn = xc.y / z;
    let r2 = xn * xn + yn * yn;
    let d = 1.0 + c.k * r2;
    let u = c.f * xn * d + c.cx;
    let v = c.f * yn * d + c.cy;

    // d(proj)/d(xn, yn) (2x2)
    let dudx = 1.0 + c.k * (3.0 * xn * xn + yn * yn);
    let dudy = 2.0 * c.k * xn * yn;
    let dvdx = dudy;
    let dvdy = 1.0 + c.k * (xn * xn + 3.0 * yn * yn);
    // d(xn,yn)/d(xc) (2x3): rows [iz,0,-xn*iz] and [0,iz,-yn*iz]
    let iz = 1.0 / z;
    // dp/dxc = f * D * P  (2x2 * 2x3)
    let mut dp_dxc = [[0f64; 3]; 2];
    for j in 0..3 {
        let p0 = match j {
            0 => iz,
            1 => 0.0,
            _ => -xn * iz,
        };
        let p1 = match j {
            0 => 0.0,
            1 => iz,
            _ => -yn * iz,
        };
        dp_dxc[0][j] = c.f * (dudx * p0 + dudy * p1);
        dp_dxc[1][j] = c.f * (dvdx * p0 + dvdy * p1);
    }

    // dp/dX = dp/dxc * R
    let mut jp = [[0f64; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            jp[i][j] = dp_dxc[i][0] * r[(0, j)]
                + dp_dxc[i][1] * r[(1, j)]
                + dp_dxc[i][2] * r[(2, j)];
        }
    }

    let mut jc = SMatrix::<f64, 2, CAM_DIM>::zeros();
    // Quaternion block: dp/dxc * d(Rv)/dq
    let qjac = quat_point_jac(&c.q, x);
    for qi in 0..4 {
        jc[(0, qi)] = dp_dxc[0][0] * qjac[qi][0]
            + dp_dxc[0][1] * qjac[qi][1]
            + dp_dxc[0][2] * qjac[qi][2];
        jc[(1, qi)] = dp_dxc[1][0] * qjac[qi][0]
            + dp_dxc[1][1] * qjac[qi][1]
            + dp_dxc[1][2] * qjac[qi][2];
    }
    // Translation block = dp/dxc
    for j in 0..3 {
        jc[(0, 4 + j)] = dp_dxc[0][j];
        jc[(1, 4 + j)] = dp_dxc[1][j];
    }
    // Intrinsics: 7=f, 8=cx, 9=cy, 10=k
    jc[(0, 7)] = xn * d;
    jc[(1, 7)] = yn * d;
    jc[(0, 8)] = 1.0;
    jc[(1, 9)] = 1.0;
    jc[(0, 10)] = c.f * xn * r2;
    jc[(1, 10)] = c.f * yn * r2;
    Some((jc, jp, (u, v)))
}

pub fn bundle_adjust(
    cams: &mut Vec<BaCam>,
    pts: &mut Vec<[f64; 3]>,
    obs: &[BaObs],
    fixed_cams: &[usize],
    max_iters: usize,
    huber_delta: f64,
) -> f64 {
    let n_cams = cams.len();
    let n_pts = pts.len();
    if n_cams == 0 || n_pts == 0 || obs.is_empty() {
        return 0.0;
    }
    let debug = std::env::var("BA_DEBUG").is_ok();
    let mut lambda = 1e-4f64;
    let mut cur = cost(cams, pts, obs, huber_delta);

    for _iter in 0..max_iters {
        let z_guard = depth_guard(cams, pts, obs);

        // ---- Pass 1: per-observation weighted Jacobians + residuals ----
        let mut jcs: Vec<Option<SMatrix<f64, 2, CAM_DIM>>> = vec![None; obs.len()];
        let mut jps: Vec<[[f64; 3]; 2]> = vec![[ [0.0; 3]; 2 ]; obs.len()];
        let mut rws: Vec<[f64; 2]> = vec![[0.0; 2]; obs.len()];
        let mut cnorm = vec![[0.0f64; CAM_DIM]; n_cams];
        let mut pnorm = vec![[0.0f64; 3]; n_pts];
        for (oi, o) in obs.iter().enumerate() {
            let c = &cams[o.cam];
            let x = Vector3::new(pts[o.pt][0], pts[o.pt][1], pts[o.pt][2]);
            let xc = quat_to_rot(&c.q) * x + Vector3::new(c.t[0], c.t[1], c.t[2]);
            if xc.z < z_guard {
                continue;
            }
            let (jc, jp, uv) = match jacobians(c, &x) {
                Some(v) => v,
                None => continue,
            };
            let (u, v) = uv;
            if u.is_nan() || v.is_nan() {
                continue;
            }
            let ex = u - o.u;
            let ey = v - o.v;
            let e = ex.hypot(ey);
            let w = if e <= huber_delta { 1.0 } else { huber_delta / e };
            let sw = w.sqrt();
            let jcw = jc * sw;
            let jpw = [
                [jp[0][0] * sw, jp[0][1] * sw, jp[0][2] * sw],
                [jp[1][0] * sw, jp[1][1] * sw, jp[1][2] * sw],
            ];
            for k in 0..CAM_DIM {
                cnorm[o.cam][k] += jcw[(0, k)] * jcw[(0, k)] + jcw[(1, k)] * jcw[(1, k)];
            }
            for j in 0..3 {
                pnorm[o.pt][j] += jpw[0][j] * jpw[0][j] + jpw[1][j] * jpw[1][j];
            }
            jcs[oi] = Some(jcw);
            jps[oi] = jpw;
            rws[oi] = [ex * sw, ey * sw];
        }

        // Jacobi column scales (Ceres-style): all columns get unit norm so
        // the normal equations are well conditioned despite the f/z spread.
        let mut cscale = vec![[1.0f64; CAM_DIM]; n_cams];
        for c in 0..n_cams {
            for k in 0..CAM_DIM {
                cscale[c][k] = cnorm[c][k].max(1e-12).sqrt();
            }
        }
        let mut pscale = vec![[1.0f64; 3]; n_pts];
        for p in 0..n_pts {
            for j in 0..3 {
                pscale[p][j] = pnorm[p][j].max(1e-12).sqrt();
            }
        }

        // ---- Pass 2: scaled normal equations ----
        let mut hcc = vec![SMatrix::<f64, CAM_DIM, CAM_DIM>::zeros(); n_cams];
        let mut gc = vec![SVector::<f64, CAM_DIM>::zeros(); n_cams];
        let mut hp = vec![Matrix3::<f64>::zeros(); n_pts];
        let mut gp = vec![Vector3::<f64>::zeros(); n_pts];
        let mut pobs: Vec<Vec<(usize, SMatrix<f64, CAM_DIM, 3>)>> = vec![Vec::new(); n_pts];
        for (oi, o) in obs.iter().enumerate() {
            let jc = match &jcs[oi] {
                Some(m) => *m,
                None => continue,
            };
            let jp = &jps[oi];
            let r = rws[oi];
            let sc = &cscale[o.cam];
            let sp = &pscale[o.pt];
            // Scale columns of J.
            let mut jcs_ = jc;
            for k in 0..CAM_DIM {
                jcs_[(0, k)] /= sc[k];
                jcs_[(1, k)] /= sc[k];
            }
            let jps_ = SMatrix::<f64, 2, 3>::new(
                jp[0][0] / sp[0], jp[0][1] / sp[1], jp[0][2] / sp[2],
                jp[1][0] / sp[0], jp[1][1] / sp[1], jp[1][2] / sp[2],
            );
            let r2 = SVector::<f64, 2>::new(r[0], r[1]);
            hcc[o.cam] += jcs_.transpose() * jcs_;
            gc[o.cam] += jcs_.transpose() * r2;
            hp[o.pt] += jps_.transpose() * jps_;
            gp[o.pt] += jps_.transpose() * r2;
            pobs[o.pt].push((o.cam, jcs_.transpose() * jps_));
        }

        // ---- Schur complement (scaled space) ----
        let dim = n_cams * CAM_DIM;
        let mut h_schur = DMatrix::<f64>::zeros(dim, dim);
        let mut g_schur = DVector::<f64>::zeros(dim);
        let mut hp_inv = vec![Matrix3::<f64>::identity(); n_pts];
        for p in 0..n_pts {
            let mut h3 = hp[p];
            h3[(0, 0)] += 1e-9;
            h3[(1, 1)] += 1e-9;
            h3[(2, 2)] += 1e-9;
            hp_inv[p] = h3.try_inverse().unwrap_or(Matrix3::zeros());
        }
        let mut hcc_s = hcc.clone();
        let mut gc_s = gc.clone();
        for p in 0..n_pts {
            if pobs[p].is_empty() {
                continue;
            }
            let hi = hp_inv[p];
            let gpp = gp[p];
            for (c, b) in &pobs[p] {
                let bhi = b * hi;
                hcc_s[*c] -= bhi * b.transpose();
                gc_s[*c] -= bhi * gpp;
            }
        }
        for c in 0..n_cams {
            let off = c * CAM_DIM;
            for i in 0..CAM_DIM {
                g_schur[off + i] = gc_s[c][i];
                for j in 0..CAM_DIM {
                    h_schur[(off + i, off + j)] = hcc_s[c][(i, j)];
                }
            }
        }
        // LM damping + fixed-camera gauge.
        for i in 0..dim {
            h_schur[(i, i)] += lambda * (1.0 + h_schur[(i, i)]);
        }
        for &c in fixed_cams {
            let off = c * CAM_DIM;
            for i in 0..7 {
                h_schur[(off + i, off + i)] += 1e12;
            }
        }

        // ---- Solve: (H + lambda I) * delta = -g  (scaled space) ----
        let lu = h_schur.clone().lu();
        let rhs = &g_schur * -1.0;
        let dxt = match lu.solve(&rhs) {
            Some(v) => v,
            None => break,
        };
        // Unscale the step.
        let mut dx = DVector::<f64>::zeros(dim);
        for c in 0..n_cams {
            let off = c * CAM_DIM;
            for k in 0..CAM_DIM {
                dx[off + k] = dxt[off + k] / cscale[c][k];
            }
        }
        if debug {
            let mx = dx.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            let gmax = g_schur.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            eprintln!("[ba]   |dx|_inf={:.3e} |g|_inf={:.3e} lambda={:.0e}", mx, gmax, lambda);
        }

        // ---- Apply candidate step (points via scaled back-substitution) ----
        let mut new_cams = cams.clone();
        for c in 0..n_cams {
            let off = c * CAM_DIM;
            let q = [
                new_cams[c].q[0] + dx[off],
                new_cams[c].q[1] + dx[off + 1],
                new_cams[c].q[2] + dx[off + 2],
                new_cams[c].q[3] + dx[off + 3],
            ];
            let nq = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if nq > 1e-12 {
                new_cams[c].q = [q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq];
            }
            new_cams[c].t[0] += dx[off + 4];
            new_cams[c].t[1] += dx[off + 5];
            new_cams[c].t[2] += dx[off + 6];
            new_cams[c].f += dx[off + 7];
            new_cams[c].cx += dx[off + 8];
            new_cams[c].cy += dx[off + 9];
            new_cams[c].k += dx[off + 10];
            new_cams[c].f = new_cams[c].f.max(1.0);
            new_cams[c].k = new_cams[c].k.clamp(-0.5, 0.5);
        }
        let mut new_pts = pts.clone();
        for p in 0..n_pts {
            if hp_inv[p] == Matrix3::zeros() || pobs[p].is_empty() {
                continue;
            }
            let mut corr = gp[p];
            for (c, b) in &pobs[p] {
                let off = c * CAM_DIM;
                let dct = SVector::<f64, CAM_DIM>::from_column_slice(&[
                    dxt[off],
                    dxt[off + 1],
                    dxt[off + 2],
                    dxt[off + 3],
                    dxt[off + 4],
                    dxt[off + 5],
                    dxt[off + 6],
                    dxt[off + 7],
                    dxt[off + 8],
                    dxt[off + 9],
                    dxt[off + 10],
                ]);
                corr -= b.transpose() * dct;
            }
            let dpt = hp_inv[p] * corr;
            let sp = &pscale[p];
            new_pts[p][0] += dpt.x / sp[0];
            new_pts[p][1] += dpt.y / sp[1];
            new_pts[p][2] += dpt.z / sp[2];
        }
        let new_cost = cost(&new_cams, &new_pts, obs, huber_delta);
        if debug {
            eprintln!("[ba] iter {}: cost {:.1} -> {:.1}", _iter, cur, new_cost);
        }
        if new_cost < cur {
            let rel = (cur - new_cost) / (cur + 1e-12);
            *cams = new_cams;
            *pts = new_pts;
            cur = new_cost;
            lambda = (lambda * 0.5).max(1e-12);
            if rel < 1e-8 {
                break;
            }
        } else {
            lambda = (lambda * 10.0).min(1e10);
            if lambda > 1e9 {
                break;
            }
        }
    }
    if debug {
        eprintln!("[ba] end: cost {:.1}", cur);
    }
    cur
}

/// Mean reprojection error in pixels over all observations.
pub fn mean_reproj(cams: &[BaCam], pts: &[[f64; 3]], obs: &[BaObs]) -> f64 {
    if obs.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut cnt = 0;
    for o in obs {
        let c = &cams[o.cam];
        let r = quat_to_rot(&c.q);
        let xc = r * Vector3::new(pts[o.pt][0], pts[o.pt][1], pts[o.pt][2])
            + Vector3::new(c.t[0], c.t[1], c.t[2]);
        if let Some((u, v)) = project(c, &xc) {
            sum += (u - o.u).hypot(v - o.v);
            cnt += 1;
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum / cnt as f64
    }
}
