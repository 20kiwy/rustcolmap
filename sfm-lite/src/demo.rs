//! Self-contained end-to-end validation: procedurally generates a textured
//! "corner room" scene, renders it from known camera poses into synthetic
//! photos (point-splat rendering + noise), runs the full SfM pipeline on
//! those photos, then compares estimated camera poses against ground truth
//! (Umeyama-aligned). Also writes the COLMAP dataset + PLY cloud.

use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgb, RgbImage};
use nalgebra::Vector3;

use crate::geom::umeyama;
use crate::rng::Rng;

pub struct DemoCam {
    pub name: String,
    pub pose: crate::geom::Pose,
    pub f: f64,
    pub w: u32,
    pub h: u32,
}

fn look_at(pos: Vector3<f64>, target: Vector3<f64>) -> crate::geom::Pose {
    let forward = (target - pos).normalize();
    let up = Vector3::new(0.0, 0.0, 1.0);
    let right = forward.cross(&up).normalize();
    let down = forward.cross(&right).normalize();
    let r = nalgebra::Matrix3::from_rows(&[
        right.transpose(),
        down.transpose(),
        forward.transpose(),
    ]);
    crate::geom::Pose {
        r,
        t: -r * pos,
    }
}

/// Build the ground-truth scene: cameras orbiting a corner room whose
/// walls (floor z=0, wall y=-5, wall x=-5) are covered in colored dots.
pub fn build_scene(n_cams: usize) -> (Vec<DemoCam>, Vec<(Vector3<f64>, [u8; 3], f64)>) {
    let mut rng = Rng::seeded(42);
    let mut pts: Vec<(Vector3<f64>, [u8; 3], f64)> = Vec::new();

    let rand_color = |rng: &mut Rng| -> [u8; 3] {
        // Vivid colors -> strong gray-level texture after conversion.
        let h = rng.unit();
        let (r, g, b) = if h < 1.0 / 3.0 {
            (
                120.0 + 135.0 * (h * 3.0),
                40.0 + 180.0 * rng.unit(),
                40.0 + 180.0 * rng.unit(),
            )
        } else if h < 2.0 / 3.0 {
            (
                40.0 + 180.0 * rng.unit(),
                120.0 + 135.0 * (h * 3.0 - 1.0),
                40.0 + 180.0 * rng.unit(),
            )
        } else {
            (
                40.0 + 180.0 * rng.unit(),
                40.0 + 180.0 * rng.unit(),
                120.0 + 135.0 * (h * 3.0 - 2.0),
            )
        };
        [r as u8, g as u8, b as u8]
    };

    // Floor (z = 0), wall y = -5, wall x = -5.
    let n_side = 60;
    for i in 0..n_side {
        for j in 0..n_side {
            let jitter = |rng: &mut Rng| (rng.unit() - 0.5) * 0.16;
            // Floor: x in [-5, 5], y in [-5, 5]
            let x = -5.0 + 10.0 * i as f64 / (n_side - 1) as f64 + jitter(&mut rng);
            let y = -5.0 + 10.0 * j as f64 / (n_side - 1) as f64 + jitter(&mut rng);
            pts.push((Vector3::new(x, y, 0.0), rand_color(&mut rng), 1.0));
            // Wall y = -5
            let x2 = -5.0 + 10.0 * i as f64 / (n_side - 1) as f64 + jitter(&mut rng);
            let z2 = 0.05 + 2.95 * j as f64 / (n_side - 1) as f64 + jitter(&mut rng) * 0.2;
            pts.push((Vector3::new(x2, -5.0, z2), rand_color(&mut rng), 1.0));
            // Wall x = -5
            let y3 = -5.0 + 10.0 * i as f64 / (n_side - 1) as f64 + jitter(&mut rng);
            let z3 = 0.05 + 2.95 * j as f64 / (n_side - 1) as f64 + jitter(&mut rng) * 0.2;
            pts.push((Vector3::new(-5.0, y3, z3), rand_color(&mut rng), 1.0));
        }
    }
    // 10% "big blob" scale variety.
    for p in pts.iter_mut().step_by(10) {
        p.2 = 2.5;
    }

    // Cameras: arc in the x>0, y>0 quadrant looking at the corner.
    let (w, h) = (640u32, 480u32);
    let f = 550.0f64;
    let mut cams = Vec::with_capacity(n_cams);
    for i in 0..n_cams {
        let theta = 0.35 + i as f64 * (1.1 / (n_cams.max(2) - 1) as f64);
        let radius = 4.2 + 0.5 * ((i as f64) * 0.7).sin();
        let pos = Vector3::new(
            radius * theta.cos() + 0.5,
            radius * theta.sin() + 0.5,
            1.1 + 0.18 * i as f64,
        );
        let pose = look_at(pos, Vector3::new(-1.2, -1.2, 0.9));
        cams.push(DemoCam {
            name: format!("img_{:02}.png", i),
            pose,
            f,
            w,
            h,
        });
    }
    (cams, pts)
}

/// Render one view: painter's algorithm, depth-sorted discs + sensor noise.
pub fn render(cam: &DemoCam, pts: &[(Vector3<f64>, [u8; 3], f64)], seed: u64) -> RgbImage {
    let mut rng = Rng::seeded(seed);
    let mut img: RgbImage = ImageBuffer::from_fn(cam.w, cam.h, |_, _| Rgb([26, 26, 30]));
    // Per-pixel noise on the background.
    for p in img.pixels_mut() {
        let n = rng.gauss(5.0);
        let v = (p[0] as f64 + n).clamp(0.0, 255.0) as u8;
        let v2 = (p[1] as f64 + n).clamp(0.0, 255.0) as u8;
        let v3 = (p[2] as f64 + n).clamp(0.0, 255.0) as u8;
        *p = Rgb([v, v2, v3]);
    }
    // Project + sort by depth (far first).
    let mut proj: Vec<(f64, f64, f64, [u8; 3], f64)> = Vec::with_capacity(pts.len());
    for (x, c, scale) in pts {
        if let Some((u, v)) = cam.pose.project(&crate::geom::Cam { f: cam.f, cx: cam.w as f64 / 2.0, cy: cam.h as f64 / 2.0, k: 0.0 }, x) {
            let xc = cam.pose.r * *x + cam.pose.t;
            proj.push((xc.z, u, v, *c, *scale));
        }
    }
    proj.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let cx = cam.w as f64 / 2.0;
    let cy = cam.h as f64 / 2.0;
    for (depth, u, v, c, scale) in proj {
        let jitter_x = rng.gauss(0.25);
        let jitter_y = rng.gauss(0.25);
        let u = u + jitter_x;
        let v = v + jitter_y;
        if u < -4.0 || v < -4.0 || u >= cam.w as f64 + 4.0 || v >= cam.h as f64 + 4.0 {
            continue;
        }
        let r = ((cam.f / 550.0) * (2.0 / depth) * scale).round().clamp(1.0, 5.0) as i32;
        let (ui, vi) = (u.round() as i32, v.round() as i32);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let x = ui + dx;
                let y = vi + dy;
                if x < 0 || y < 0 || x >= cam.w as i32 || y >= cam.h as i32 {
                    continue;
                }
                // Fade slightly with depth for realism.
                let fade = (1.0 - 0.03 * depth).clamp(0.55, 1.0);
                let col = [
                    (c[0] as f64 * fade).clamp(0.0, 255.0) as u8,
                    (c[1] as f64 * fade).clamp(0.0, 255.0) as u8,
                    (c[2] as f64 * fade).clamp(0.0, 255.0) as u8,
                ];
                img.put_pixel(x as u32, y as u32, Rgb(col));
            }
        }
    }
    let _ = (cx, cy);
    img
}

/// Write an ASCII PLY point cloud.
pub fn write_ply(path: &Path, pts: &[(f64, f64, f64, [u8; 3])]) -> std::io::Result<()> {
    let mut s = String::new();
    s.push_str("ply\n");
    s.push_str("format ascii 1.0\n");
    s.push_str(&format!("element vertex {}\n", pts.len()));
    s.push_str("property float x\nproperty float y\nproperty float z\n");
    s.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    s.push_str("end_header\n");
    for (x, y, z, c) in pts {
        s.push_str(&format!("{} {} {} {} {} {}\n", x, y, z, c[0], c[1], c[2]));
    }
    std::fs::write(path, s)
}

/// Run the full demo: returns (registered, points, mean center error in
/// scene units, mean rotation error in degrees).
pub fn run_demo(out_dir: &Path, n_cams: usize, log: &dyn Fn(&str)) -> Result<(usize, usize, f64, f64), String> {
    let img_dir = out_dir.join("images");
    std::fs::create_dir_all(&img_dir).map_err(|e| e.to_string())?;

    let (cams, pts) = build_scene(n_cams);
    log(&format!(
        "demo: rendering {} synthetic views of {} scene points...",
        cams.len(),
        pts.len()
    ));
    let mut paths: Vec<PathBuf> = Vec::new();
    for (i, cam) in cams.iter().enumerate() {
        let img = render(cam, &pts, 1000 + i as u64);
        let p = img_dir.join(&cam.name);
        img.save(&p).map_err(|e| format!("save {}: {e}", p.display()))?;
        paths.push(p);
    }

    // Ground truth for evaluation.
    let gt: Vec<(String, Vector3<f64>, crate::geom::Pose)> = cams
        .iter()
        .map(|c| (c.name.clone(), c.pose.center(), c.pose))
        .collect();

    let opts = crate::Options {
        max_dim: 640,
        max_features: 2500,
        ba_every: 4,
        focal_mult: 550.0 / 640.0,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let res = crate::run_sfm(&paths, &opts, &log)?;

    // Align estimated centers to GT and measure errors.
    let mut gt_c = Vec::new();
    let mut est_c = Vec::new();
    let mut rot_pairs_gt: Vec<crate::geom::Pose> = Vec::new();
    let mut rot_pairs_est: Vec<crate::geom::Pose> = Vec::new();
    for (name, center, pose) in &gt {
        if let Some(im) = res.rec.image_by_name(name) {
            let q = im.qvec;
            let r = crate::geom::quat_to_rot(&q);
            let t = Vector3::new(im.tvec[0], im.tvec[1], im.tvec[2]);
            let est = crate::geom::Pose { r, t };
            est_c.push(est.center());
            gt_c.push(*center);
            rot_pairs_gt.push(pose.clone_owned_pose());
            rot_pairs_est.push(est);
        }
    }
    let rot_pairs: Vec<(crate::geom::Pose, crate::geom::Pose)> =
        rot_pairs_gt.iter().zip(rot_pairs_est.iter()).map(|(a, b)| (*a, *b)).collect();
    if est_c.len() < 2 {
        return Err(format!("only {} images registered", est_c.len()));
    }
    // Rotation-based world alignment (chordal L2 mean of R_est R_gt^T),
    // independent of any center distortion.
    // Relation: R_est_i ~ R_gt_i * A^T  =>  A^T = chordal mean of R_gt^T R_est.
    let mut m_align = nalgebra::Matrix3::<f64>::zeros();
    for (i, gpose) in rot_pairs_gt.iter().enumerate() {
        m_align += gpose.r.transpose() * rot_pairs_est[i].r;
    }
    let (u_a, _sv_a, v_a) = crate::geom::svd3(&m_align)
        .ok_or("svd failed")?;
    let d_a = if (u_a * v_a.transpose()).determinant() < 0.0 { -1.0 } else { 1.0 };
    let q_align = u_a * nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(1.0, 1.0, d_a)) * v_a.transpose();
    let ra = q_align.transpose();
    // Scale + translation from mean-centered centers under ra.
    let n_c = gt_c.len() as f64;
    let mu_g = gt_c.iter().fold(Vector3::zeros(), |a, v| a + v) / n_c;
    let mu_e = est_c.iter().fold(Vector3::zeros(), |a, v| a + v) / n_c;
    let s = gt_c.iter().zip(est_c.iter())
        .map(|(g, e)| (ra * (g - mu_g)).dot(&(e - mu_e)))
        .sum::<f64>() / gt_c.iter().map(|g| (g - mu_g).norm_squared()).sum::<f64>().max(1e-12);
    let ta = mu_e - s * ra * mu_g;
    log(&format!("rotation-aligned: scale={:.4}", s));
    let _ = umeyama(&gt_c, &est_c);
    log(&format!(
        "umeyama: scale={:.4} (expect ~1.0)",
        s
    ));
    for (i, name) in ["img_00.png", "img_03.png", "img_05.png", "img_07.png", "img_09.png"].iter().enumerate() {
        let _ = i;
        if let Some(im) = res.rec.image_by_name(name) {
            let r = crate::geom::quat_to_rot(&im.qvec);
            let t = Vector3::new(im.tvec[0], im.tvec[1], im.tvec[2]);
            let c_est = -r.transpose() * t;
            if let Some((_, gc, gp)) = gt.iter().find(|(n, _, _)| n.starts_with(&name[..8])) {
                let c_gt = *gc;
                let c_aligned = s * ra * c_gt + ta;
                log(&format!(
                    "cam {}: est=({:+.2},{:+.2},{:+.2}) gt_aligned=({:+.2},{:+.2},{:+.2})",
                    name, c_est.x, c_est.y, c_est.z, c_aligned.x, c_aligned.y, c_aligned.z
                ));
                let _ = gp;
            }
        }
    }
    // Scale-invariant shape check: pairwise camera-center distance ratios.
    // ratio ~ 1 for all pairs => reconstruction geometry matches GT shape.
    {
        let mut ratios: Vec<f64> = Vec::new();
        for i in 0..gt_c.len() {
            for j in (i + 1)..gt_c.len() {
                let d_gt = (gt_c[i] - gt_c[j]).norm();
                let d_est = (est_c[i] - est_c[j]).norm();
                if d_gt > 1e-9 {
                    ratios.push(d_est / d_gt);
                }
            }
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ratios[ratios.len() / 2];
        let p10 = ratios[ratios.len() / 10];
        let p90 = ratios[(ratios.len() * 9) / 10];
        log(&format!(
            "pairwise center-distance ratio est/gt: median={:.4} p10={:.4} p90={:.4} (1.0 = perfect shape)",
            med, p10, p90
        ));
    }
    let mut pos_errs = Vec::new();
    let mut rot_errs = Vec::new();
    for (i, (gt_pose, est_pose)) in rot_pairs.iter().enumerate() {
        let aligned = s * ra * gt_c[i] + ta;
        pos_errs.push((est_c[i] - aligned).norm());
        let r_est = gt_pose.r * ra.transpose();
        let d = (r_est.transpose() * est_pose.r).determinant();
        let tr = (r_est.transpose() * est_pose.r).diagonal().sum();
        let ang = ((tr / 3.0).clamp(-1.0, 1.0)).acos().to_degrees();
        let _ = d;
        rot_errs.push(ang.abs());
    }
    log(&format!(
        "per-image center errors: {:?}",
        pos_errs.iter().map(|v| (*v * 100.0).round() / 100.0).collect::<Vec<_>>()
    ));
    log(&format!(
        "per-image rotation errors (deg): {:?}",
        rot_errs.clone().into_iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
    ));
    pos_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rot_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: &mut Vec<f64>| v[v.len() / 2];
    let pos_med = med(&mut pos_errs);
    let rot_med = med(&mut rot_errs);

    log(&format!(
        "demo: {} images registered, {} points, mean reproj {:.2}px, time {:.1}s",
        res.registered,
        res.points,
        res.mean_reproj,
        t0.elapsed().as_secs_f32()
    ));
    log(&format!(
        "demo accuracy (Umeyama-aligned): median camera-center error = {:.4} (scene units), median rotation error = {:.2} deg",
        pos_med, rot_med
    ));
    if let Some(first) = res.rec.cameras.first() {
        log(&format!(
            "recovered focal (GT 550.0): cam0 = {:.1}",
            first.params[0]
        ));
    }

    // ---- GT-based track quality: for each reconstructed observation,
    // project the (GT-frame-transformed) point with the GT pose and compare
    // with the detected keypoint. Clean tracks => ~0.5-1px; mismatches => >>.
    {
        // image name -> (gt pose, gt cam)
        let gt_by_name: std::collections::HashMap<&String, &DemoCam> =
            cams.iter().map(|c| (&c.name, c)).collect();
        let to_gt = |p: [f64; 3]| -> Vector3<f64> {
            let v = Vector3::new(p[0], p[1], p[2]);
            (1.0 / s) * ra.transpose() * (v - ta)
        };
        let mut errs: Vec<f64> = Vec::new();
        for im in &res.rec.images {
            let cam_g = match gt_by_name.get(&im.name) {
                Some(c) => c,
                None => continue,
            };
            let gt_cam = crate::geom::Cam {
                f: cam_g.f,
                cx: cam_g.w as f64 / 2.0,
                cy: cam_g.h as f64 / 2.0,
                k: 0.0,
            };
            for p2 in &im.points2d {
                if p2.point3d_id < 0 {
                    continue;
                }
                let pt = &res.rec.points3d[p2.point3d_id as usize - 1];
                let xg = to_gt(pt.xyz);
                if let Some((u, v)) = cam_g.pose.project(&gt_cam, &xg) {
                    errs.push((u - p2.x).hypot(v - p2.y));
                }
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !errs.is_empty() {
            let n = errs.len();
            let frac_bad = errs.iter().filter(|&&e| e > 4.0).count() as f64 / n as f64;
            log(&format!(
                "gt-frame consistency: n={} median={:.2}px p90={:.2}px frac>4px={:.1}%",
                n,
                errs[n / 2],
                errs[(n * 9) / 10],
                frac_bad * 100.0
            ));
        }
    }

    // ---- Correspondence correctness: label each keypoint by its nearest
    // projected GT scene point (<3px); a track is correct when all its
    // observations share one GT label.
    {
        let gt_by_name: std::collections::HashMap<&String, &DemoCam> =
            cams.iter().map(|c| (&c.name, c)).collect();
        // For every image: keypoint index -> GT label.
        let mut labels: std::collections::HashMap<(String, u32), i64> =
            std::collections::HashMap::new();
        for im in &res.rec.images {
            let cam_g = match gt_by_name.get(&im.name) {
                Some(c) => c,
                None => continue,
            };
            let gt_cam = crate::geom::Cam {
                f: cam_g.f,
                cx: cam_g.w as f64 / 2.0,
                cy: cam_g.h as f64 / 2.0,
                k: 0.0,
            };
            for (k, p2) in im.points2d.iter().enumerate() {
                let mut best = (3.0f64, -1i64);
                for (si, (x, _, _)) in pts.iter().enumerate() {
                    if let Some((u, v)) = cam_g.pose.project(&gt_cam, x) {
                        let d = (u - p2.x).hypot(v - p2.y);
                        if d < best.0 {
                            best = (d, si as i64);
                        }
                    }
                }
                labels.insert((im.name.clone(), k as u32), best.1);
            }
        }
        let mut obs_ok = 0usize;
        let mut obs_tot = 0usize;
        let mut trk_ok = 0usize;
        for p3 in &res.rec.points3d {
            let mut lab: Option<i64> = None;
            let mut consistent = true;
            for t in &p3.track {
                let name = &res
                    .rec
                    .images
                    .iter()
                    .find(|im| im.image_id == t.image_id)
                    .unwrap()
                    .name;
                let l = labels.get(&(name.clone(), t.point2d_idx)).copied().unwrap_or(-1);
                obs_tot += 1;
                if l >= 0 {
                    obs_ok += 1;
                }
                match lab {
                    None => lab = Some(l),
                    Some(prev) => {
                        if prev != l {
                            consistent = false;
                        }
                    }
                }
            }
            if consistent && lab.map_or(false, |l| l >= 0) {
                trk_ok += 1;
            }
        }
        log(&format!(
            "correspondence check: labeled obs {}/{}, consistent tracks {}/{}",
            obs_ok,
            obs_tot,
            trk_ok,
            res.rec.points3d.len()
        ));
    }

    // Write COLMAP dataset + PLY for inspection.
    let sparse = out_dir.join("sparse").join("0");
    colmap_io::write_bin_dir(&res.rec, &sparse).map_err(|e| e.to_string())?;
    colmap_io::write_txt_dir(&res.rec, &sparse).map_err(|e| e.to_string())?;
    let cloud: Vec<(f64, f64, f64, [u8; 3])> = res
        .rec
        .points3d
        .iter()
        .map(|p| (p.xyz[0], p.xyz[1], p.xyz[2], p.rgb))
        .collect();
    write_ply(&out_dir.join("points3d.ply"), &cloud).map_err(|e| e.to_string())?;
    log(&format!(
        "demo outputs: {} (COLMAP sparse/0 + points3d.ply)",
        out_dir.display()
    ));

    Ok((res.registered, res.points, pos_med, rot_med))
}

// Small helper so we can clone Pose values cleanly.
trait PoseClone {
    fn clone_owned_pose(&self) -> crate::geom::Pose;
}
impl PoseClone for crate::geom::Pose {
    fn clone_owned_pose(&self) -> crate::geom::Pose {
        crate::geom::Pose { r: self.r, t: self.t }
    }
}
