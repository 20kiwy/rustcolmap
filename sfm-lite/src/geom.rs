//! Geometric core: camera model, two-view geometry estimators (H / F / E),
//! RANSAC framework, DLT triangulation, PnP, Umeyama alignment and quaternion
//! utilities. Pure `nalgebra`, no external solver deps.

use nalgebra::{DMatrix, Matrix3, SVD, Vector3};

use crate::rng::Rng;

// ---------------------------------------------------------------------------
// Camera + pose
// ---------------------------------------------------------------------------

/// SIMPLE_RADIAL camera: x_d = x_n * (1 + k * r^2), u = f * x_d + cx.
#[derive(Debug, Clone, Copy)]
pub struct Cam {
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    pub k: f64,
}

impl Cam {
    pub fn new(f: f64, w: f64, h: f64) -> Self {
        Cam { f, cx: w * 0.5, cy: h * 0.5, k: 0.0 }
    }

    /// Pixel -> normalized (undistorted direction) coordinates.
    pub fn normalize(&self, u: f64, v: f64) -> [f64; 2] {
        let mut xn = (u - self.cx) / self.f;
        let mut yn = (v - self.cy) / self.f;
        // One Newton iteration to undo radial distortion.
        for _ in 0..2 {
            let r2 = xn * xn + yn * yn;
            let d = 1.0 + self.k * r2;
            if d.abs() < 1e-12 {
                break;
            }
            xn = (u - self.cx) / (self.f * d);
            yn = (v - self.cy) / (self.f * d);
        }
        [xn, yn]
    }

    pub fn kmat(&self) -> Matrix3<f64> {
        Matrix3::new(self.f, 0.0, self.cx, 0.0, self.f, self.cy, 0.0, 0.0, 1.0)
    }
}

/// Pose = world->camera: x_cam = R * x_world + t.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
}

impl Pose {
    pub fn identity() -> Self {
        Pose { r: Matrix3::identity(), t: Vector3::zeros() }
    }
    pub fn center(&self) -> Vector3<f64> {
        -self.r.transpose() * self.t
    }
    /// Project a world point (pixel coords). None if behind camera.
    pub fn project(&self, cam: &Cam, x: &Vector3<f64>) -> Option<(f64, f64)> {
        let xc = self.r * *x + self.t;
        let z = xc.z;
        if z <= 1e-9 {
            return None;
        }
        let xn = xc.x / z;
        let yn = xc.y / z;
        let r2 = xn * xn + yn * yn;
        let d = 1.0 + cam.k * r2;
        Some((cam.f * xn * d + cam.cx, cam.f * yn * d + cam.cy))
    }
}

// ---------------------------------------------------------------------------
// Quaternions (COLMAP convention: (qw, qx, qy, qz))
// ---------------------------------------------------------------------------

pub fn quat_to_rot(q: &[f64; 4]) -> Matrix3<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let n = w * w + x * x + y * y + z * z;
    let (w, x, y, z) = if n.abs() - 1.0 > 1e-12 {
        let s = n.sqrt();
        (w / s, x / s, y / s, z / s)
    } else {
        (w, x, y, z)
    };
    Matrix3::new(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    )
}

pub fn rot_to_quat(r: &Matrix3<f64>) -> [f64; 4] {
    let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let (w, x, y, z);
    if tr > 0.0 {
        let s = (1.0 + tr).sqrt() * 2.0;
        w = 0.25 * s;
        x = (r[(2, 1)] - r[(1, 2)]) / s;
        y = (r[(0, 2)] - r[(2, 0)]) / s;
        z = (r[(1, 0)] - r[(0, 1)]) / s;
    } else if r[(0, 0)] > r[(1, 1)] && r[(0, 0)] > r[(2, 2)] {
        let s = (1.0 + r[(0, 0)] - r[(1, 1)] - r[(2, 2)]).sqrt() * 2.0;
        w = (r[(2, 1)] - r[(1, 2)]) / s;
        x = 0.25 * s;
        y = (r[(0, 1)] + r[(1, 0)]) / s;
        z = (r[(0, 2)] + r[(2, 0)]) / s;
    } else if r[(1, 1)] > r[(2, 2)] {
        let s = (1.0 + r[(1, 1)] - r[(0, 0)] - r[(2, 2)]).sqrt() * 2.0;
        w = (r[(0, 2)] - r[(2, 0)]) / s;
        x = (r[(0, 1)] + r[(1, 0)]) / s;
        y = 0.25 * s;
        z = (r[(1, 2)] + r[(2, 1)]) / s;
    } else {
        let s = (1.0 + r[(2, 2)] - r[(0, 0)] - r[(1, 1)]).sqrt() * 2.0;
        w = (r[(1, 0)] - r[(0, 1)]) / s;
        x = (r[(0, 2)] + r[(2, 0)]) / s;
        y = (r[(1, 2)] + r[(2, 1)]) / s;
        z = 0.25 * s;
    }
    let mut q = [w, x, y, z];
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in q.iter_mut() {
        *v /= n;
    }
    q
}

/// d(R(q) v) / d q  -> 4 gradients (for q = (w,x,y,z)).
/// The rotation model is the standard homogeneous-degree-2 matrix form
/// (normalization inside quat_to_rot is a no-op), so the exact raw
/// derivatives of that form are used:
///   R v = (1 - 2|qv|^2) v + 2 (qv.v) qv + 2 w (qv x v)
pub fn quat_point_jac(q: &[f64; 4], v: &Vector3<f64>) -> [[f64; 3]; 4] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let (vx, vy, vz) = (v.x, v.y, v.z);
    let dot = x * vx + y * vy + z * vz;
    let exv = [0.0, -vz, vy];   // e_x x v
    let eyv = [vz, 0.0, -vx];   // e_y x v
    let ezv = [-vy, vx, 0.0];   // e_z x v
    let qxv = [
        y * vz - z * vy,
        z * vx - x * vz,
        x * vy - y * vx,
    ];
    // d/dw = 2 (qv x v)
    let dw = [2.0 * qxv[0], 2.0 * qxv[1], 2.0 * qxv[2]];
    // d/dx = -4x v + 2 vx qv + 2 dot e_x + 2 w (e_x x v)
    let dx = [
        -4.0 * x * vx + 2.0 * vx * x + 2.0 * dot + 2.0 * w * exv[0],
        -4.0 * x * vy + 2.0 * vx * y + 2.0 * w * exv[1],
        -4.0 * x * vz + 2.0 * vx * z + 2.0 * w * exv[2],
    ];
    let dy = [
        -4.0 * y * vx + 2.0 * vy * x + 2.0 * w * eyv[0],
        -4.0 * y * vy + 2.0 * vy * y + 2.0 * dot + 2.0 * w * eyv[1],
        -4.0 * y * vz + 2.0 * vy * z + 2.0 * w * eyv[2],
    ];
    let dz = [
        -4.0 * z * vx + 2.0 * vz * x + 2.0 * w * ezv[0],
        -4.0 * z * vy + 2.0 * vz * y + 2.0 * w * ezv[1],
        -4.0 * z * vz + 2.0 * vz * z + 2.0 * dot + 2.0 * w * ezv[2],
    ];
    [dw, dx, dy, dz]
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Hartley normalization: centroid at origin, mean distance sqrt(2).
pub fn normalize_points(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let dmean = pts
        .iter()
        .map(|p| (p[0] - cx).hypot(p[1] - cy))
        .sum::<f64>()
        / n;
    let s = if dmean > 1e-12 { (2f64).sqrt() / dmean } else { 1.0 };
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let out = pts.iter().map(|p| [s * (p[0] - cx), s * (p[1] - cy)]).collect();
    (t, out)
}

/// Sorted-descending 3x3 SVD -> (U, singular values, V).
pub fn svd3(m: &Matrix3<f64>) -> Option<(Matrix3<f64>, [f64; 3], Matrix3<f64>)> {
    let s = m.svd(true, true);
    let u = s.u?;
    let vt = s.v_t?;
    let sv = s.singular_values;
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| sv[b].partial_cmp(&sv[a]).unwrap());
    let u2 = Matrix3::from_columns(&[
        u.column(order[0]).into_owned(),
        u.column(order[1]).into_owned(),
        u.column(order[2]).into_owned(),
    ]);
    let v2t = Matrix3::from_rows(&[
        vt.row(order[0]).into_owned(),
        vt.row(order[1]).into_owned(),
        vt.row(order[2]).into_owned(),
    ]);
    Some((u2, [sv[order[0]], sv[order[1]], sv[order[2]]], v2t.transpose()))
}

/// Smallest right-singular vector of A, computed via eigendecomposition-free
/// SVD of A^T A (cheap for small, fixed column counts).
fn null_vector(a: &DMatrix<f64>) -> Option<Vec<f64>> {
    let cols = a.ncols();
    let ata = a.transpose() * a;
    let svd = SVD::new(ata, false, true);
    let vt = svd.v_t?;
    let k = vt.nrows() - 1; // smallest singular value is last
    Some((0..cols).map(|c| vt[(k, c)]).collect::<Vec<f64>>())
}

// ---------------------------------------------------------------------------
// Two-view estimators ( operate on raw pixel coordinates )
// ---------------------------------------------------------------------------

/// Normalized 8-point fundamental matrix from pixel correspondences.
pub fn eight_point_f(p1: &[[f64; 2]], p2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    let (t1, n1) = normalize_points(p1);
    let (t2, n2) = normalize_points(p2);
    let n = p1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let (u1, v1) = (n1[i][0], n1[i][1]);
        let (u2, v2) = (n2[i][0], n2[i][1]);
        a[(i, 0)] = u2 * u1;
        a[(i, 1)] = u2 * v1;
        a[(i, 2)] = u2;
        a[(i, 3)] = v2 * u1;
        a[(i, 4)] = v2 * v1;
        a[(i, 5)] = v2;
        a[(i, 6)] = u1;
        a[(i, 7)] = v1;
        a[(i, 8)] = 1.0;
    }
    let f = null_vector(&a)?;
    let mut e = Matrix3::new(
        f[0], f[1], f[2],
        f[3], f[4], f[5],
        f[6], f[7], f[8],
    );
    e = t2.transpose() * e * t1;
    // Rank-2 enforcement.
    let (u, sv, v) = svd3(&e)?;
    let d = Matrix3::from_diagonal(&Vector3::new(sv[0], sv[1], 0.0));
    e = u * d * v.transpose();
    if e.determinant() < 0.0 {
        e = -e;
    }
    Some(e)
}

pub fn sampson(f: &Matrix3<f64>, p1: [f64; 2], p2: [f64; 2]) -> f64 {
    let a = f * Vector3::new(p1[0], p1[1], 1.0);
    let b = f.transpose() * Vector3::new(p2[0], p2[1], 1.0);
    let num = (p2[0] * a.x + p2[1] * a.y + a.z).abs();
    let den = a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
    if den < 1e-15 {
        return f64::INFINITY;
    }
    num / den.sqrt()
}

/// 4-point DLT homography from pixel correspondences.
pub fn estimate_h4(p1: &[[f64; 2]], p2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    let (t1, n1) = normalize_points(p1);
    let (t2, n2) = normalize_points(p2);
    let n = p1.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let (u, v) = (n1[i][0], n1[i][1]);
        let (up, vp) = (n2[i][0], n2[i][1]);
        a[(2 * i, 0)] = -u;
        a[(2 * i, 1)] = -v;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = up * u;
        a[(2 * i, 7)] = up * v;
        a[(2 * i, 8)] = up;
        a[(2 * i + 1, 3)] = u;
        a[(2 * i + 1, 4)] = v;
        a[(2 * i + 1, 5)] = 1.0;
        a[(2 * i + 1, 6)] = -vp * u;
        a[(2 * i + 1, 7)] = -vp * v;
        a[(2 * i + 1, 8)] = -vp;
    }
    let h = null_vector(&a)?;
    let mut m = Matrix3::new(
        h[0], h[1], h[2],
        h[3], h[4], h[5],
        h[6], h[7], h[8],
    );
    if m[(2, 2)].abs() < 1e-12 {
        m[(2, 2)] = 1e-12;
    }
    let m = m / m[(2, 2)];
    // Map back: x2 ~ H_n x1 in normalized coords, x1n = T1 x1, x2 = T2^-1 x2n
    let t2inv = t2.try_inverse()?;
    Some(t2inv * m * t1)
}

/// Symmetric transfer error in pixels.
pub fn h_sym_err(h: &Matrix3<f64>, p1: [f64; 2], p2: [f64; 2]) -> f64 {
    let v1 = h * Vector3::new(p1[0], p1[1], 1.0);
    let hinv = match h.try_inverse() {
        Some(m) => m,
        None => return f64::INFINITY,
    };
    let v2 = hinv * Vector3::new(p2[0], p2[1], 1.0);
    if v1.z.abs() < 1e-12 || v2.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    let d1 = ((v1.x / v1.z - p2[0]).powi(2) + (v1.y / v1.z - p2[1]).powi(2)).sqrt();
    let d2 = ((v2.x / v2.z - p1[0]).powi(2) + (v2.y / v2.z - p1[1]).powi(2)).sqrt();
    d1 + d2
}

/// Essential matrix from F and intrinsics (E = K2^T F K1), forced onto the
/// essential manifold (rank 2, sigma1 == sigma2, sigma3 == 0).
pub fn f_to_essential(f: &Matrix3<f64>, k1: &Matrix3<f64>, k2: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let e = k2.transpose() * f * k1;
    let (u, sv, v) = svd3(&e)?;
    let s = 0.5 * (sv[0] + sv[1]);
    let mut e = u * Matrix3::from_diagonal(&Vector3::new(s, s, 0.0)) * v.transpose();
    if e.determinant() < 0.0 {
        e = -e;
    }
    Some(e)
}

/// All 4 (R, t) candidates from E, with proper rotations.
/// x2^T E x1 = 0 with x2 = R x1 + t mapping cam1 coords to cam2 coords.
pub fn decompose_e(e: &Matrix3<f64>) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
    let (mut u, _sv, mut v) = match svd3(e) {
        Some(x) => x,
        None => return Vec::new(),
    };
    if u.determinant() < 0.0 {
        u = -u;
    }
    if v.determinant() < 0.0 {
        v = -v;
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let wt = w.transpose();
    let r1 = u * w * v.transpose();
    let r2 = u * wt * v.transpose();
    let t1 = u.column(2).into_owned();
    let t2 = -t1;
    let mut out = Vec::with_capacity(4);
    for r in [r1, r2] {
        for t in [t1, t2] {
            if r.determinant() > 0.0 {
                out.push((r, t));
            }
        }
    }
    out
}

/// DLT triangulation with P1 = [R1|t1], P2 = [R2|t2] (normalized coords).
/// Keeps only points with positive depth in both cameras.
pub fn triangulate(
    r1: &Matrix3<f64>,
    t1: &Vector3<f64>,
    r2: &Matrix3<f64>,
    t2: &Vector3<f64>,
    x1: [f64; 2],
    x2: [f64; 2],
) -> Option<Vector3<f64>> {
    let rows1 = [
        [r1[(0, 0)], r1[(0, 1)], r1[(0, 2)], t1.x],
        [r1[(1, 0)], r1[(1, 1)], r1[(1, 2)], t1.y],
        [r1[(2, 0)], r1[(2, 1)], r1[(2, 2)], t1.z],
    ];
    let rows2 = [
        [r2[(0, 0)], r2[(0, 1)], r2[(0, 2)], t2.x],
        [r2[(1, 0)], r2[(1, 1)], r2[(1, 2)], t2.y],
        [r2[(2, 0)], r2[(2, 1)], r2[(2, 2)], t2.z],
    ];
    let (u1, v1) = (x1[0], x1[1]);
    let (u2, v2) = (x2[0], x2[1]);
    let mut a = DMatrix::<f64>::zeros(4, 4);
    for c in 0..4 {
        a[(0, c)] = u1 * rows1[2][c] - rows1[0][c];
        a[(1, c)] = v1 * rows1[2][c] - rows1[1][c];
        a[(2, c)] = u2 * rows2[2][c] - rows2[0][c];
        a[(3, c)] = v2 * rows2[2][c] - rows2[1][c];
    }
    let ata = a.transpose() * a;
    let svd = SVD::new(ata, false, true);
    let vt = svd.v_t?;
    let h = [vt[(3, 0)], vt[(3, 1)], vt[(3, 2)], vt[(3, 3)]];
    if h[3].abs() < 1e-12 {
        return None;
    }
    let p = Vector3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    let d1 = (r1 * p + t1).z;
    let d2 = (r2 * p + t2).z;
    if d1 > 1e-6 && d2 > 1e-6 {
        Some(p)
    } else {
        None
    }
}

/// Triangulation angle in degrees between the two camera rays toward point X.
pub fn tri_angle_deg(p1: &Pose, p2: &Pose, x: &Vector3<f64>) -> f64 {
    let c1 = p1.center();
    let c2 = p2.center();
    let v1 = x - c1;
    let v2 = x - c2;
    let n1 = v1.norm();
    let n2 = v2.norm();
    if n1 < 1e-12 || n2 < 1e-12 {
        return 0.0;
    }
    let c = (v1.dot(&v2) / (n1 * n2)).clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

// ---------------------------------------------------------------------------
// PnP (DLT, 6-point) - normalized image coords + 3D points
// ---------------------------------------------------------------------------

/// Estimate (R, t) with x_cam = R X + t from >= 6 normalized 2D-3D matches.
pub fn pnp_dlt(pixels: &[[f64; 2]], pts3d: &[Vector3<f64>], cam: &Cam) -> Option<Pose> {
    let n = pixels.len();
    if n < 6 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        let [u, v] = cam.normalize(pixels[i][0], pixels[i][1]);
        let x = pts3d[i];
        a[(2 * i, 0)] = x.x;
        a[(2 * i, 1)] = x.y;
        a[(2 * i, 2)] = x.z;
        a[(2 * i, 3)] = 1.0;
        a[(2 * i, 8)] = -u * x.x;
        a[(2 * i, 9)] = -u * x.y;
        a[(2 * i, 10)] = -u * x.z;
        a[(2 * i, 11)] = -u;
        a[(2 * i + 1, 4)] = x.x;
        a[(2 * i + 1, 5)] = x.y;
        a[(2 * i + 1, 6)] = x.z;
        a[(2 * i + 1, 7)] = 1.0;
        a[(2 * i + 1, 8)] = -v * x.x;
        a[(2 * i + 1, 9)] = -v * x.y;
        a[(2 * i + 1, 10)] = -v * x.z;
        a[(2 * i + 1, 11)] = -v;
    }
    let p = null_vector(&a)?;
    let m = Matrix3::new(p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]);
    let p4 = Vector3::new(p[3], p[7], p[11]);
    let (u, sv, v) = svd3(&m)?;
    let d = if (u * v.transpose()).determinant() < 0.0 { -1.0 } else { 1.0 };
    let r = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * v.transpose();
    let rho = (sv[0] + sv[1] + d * sv[2]) / 3.0;
    if rho.abs() < 1e-12 {
        return None;
    }
    let t = p4 / rho;
    let r = if r.determinant() > 0.0 { r } else { return None };
    Some(Pose { r, t })
}

// ---------------------------------------------------------------------------
// Umeyama similarity alignment (used by the self-test demo)
// ---------------------------------------------------------------------------

/// Find (s, R, t) minimizing sum || s*R*src_i + t - dst_i ||^2.
pub fn umeyama(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Option<(f64, Matrix3<f64>, Vector3<f64>)> {
    let n = src.len();
    if n < 3 {
        return None;
    }
    let mu_s = src.iter().fold(Vector3::zeros(), |a, v| a + v) / n as f64;
    let mu_d = dst.iter().fold(Vector3::zeros(), |a, v| a + v) / n as f64;
    let mut cov = Matrix3::zeros();
    let mut var = 0.0;
    for i in 0..n {
        let a = src[i] - mu_s;
        let b = dst[i] - mu_d;
        cov += b * a.transpose();
        var += a.norm_squared();
    }
    cov /= n as f64;
    var /= n as f64;
    let (u, sv, v) = svd3(&cov)?;
    let d = if (u * v.transpose()).determinant() < 0.0 { -1.0 } else { 1.0 };
    let r = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * v.transpose();
    let s = (sv[0] + sv[1] + d * sv[2]) / var.max(1e-12);
    let t = mu_d - s * r * mu_s;
    Some((s, r, t))
}

// ---------------------------------------------------------------------------
// Generic RANSAC
// ---------------------------------------------------------------------------

/// Generic adaptive RANSAC. `est` builds a model from `k` sampled indices,
/// `resid` scores one correspondence against the model (same units as `thr`).
pub fn ransac<T>(
    n: usize,
    k: usize,
    thr: f64,
    seed: u64,
    est: &mut dyn FnMut(&[usize]) -> Option<T>,
    resid: &dyn Fn(&T, usize) -> f64,
) -> Option<(T, Vec<usize>)> {
    if n < k {
        return None;
    }
    let mut rng = Rng::seeded(seed);
    let log_conf = (1.0 - 0.999f64).ln();
    let mut best: Option<(usize, T, Vec<bool>)> = None;
    let mut it = 0usize;
    let max_iters = 5000usize;
    while it < max_iters {
        // Adaptive termination.
        if let Some((cnt, _, _)) = &best {
            let w = *cnt as f64 / n as f64;
            let denom = (1.0 - w.powi(k as i32)).ln();
            if denom < -1e-12 {
                let need = (log_conf / denom).ceil() as usize;
                if it >= need.min(max_iters) {
                    break;
                }
            }
        }
        it += 1;
        let sample = rng.sample_k(n, k);
        let model = match est(&sample) {
            Some(m) => m,
            None => continue,
        };
        let mut inl = vec![false; n];
        let mut cnt = 0usize;
        for i in 0..n {
            if resid(&model, i) < thr {
                inl[i] = true;
                cnt += 1;
            }
        }
        if best.as_ref().map_or(true, |(c, _, _)| cnt > *c) {
            best = Some((cnt, model, inl));
        }
    }
    let (_, model, inl) = best?;
    Some((model, (0..n).filter(|&i| inl[i]).collect()))
}
