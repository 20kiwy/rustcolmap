//! COLMAP-style incremental structure-from-motion:
//! 1. best initial pair (max verified inliers, non-planar, triangulation rate)
//! 2. iteratively register the image with the most already-reconstructed
//!    points visible (PnP-RANSAC)
//! 3. extend tracks, triangulate new points, filter outliers
//! 4. periodic global bundle adjustment, final BA + filtering

use std::collections::HashMap;

use nalgebra::{Matrix3, Vector3};

use crate::ba::{self, BaCam, BaObs};
use crate::geom::{Cam, Pose, tri_angle_deg, triangulate};
use crate::matching::{TwoView, pnp_ransac};
use crate::scene::{FrameInfo, Scene};

pub struct MapperOptions {
    /// Minimum verified inliers for an image to be registerable.
    pub abs_min_inliers: usize,
    pub reproj_thr: f64,
    pub min_tri_angle: f64,
    /// Run global BA after every N registrations (0 = only at the end).
    pub ba_every: usize,
    pub ba_iters: usize,
}

impl Default for MapperOptions {
    fn default() -> Self {
        MapperOptions {
            abs_min_inliers: 15,
            reproj_thr: 4.0,
            min_tri_angle: 1.5,
            ba_every: 4,
            ba_iters: 20,
        }
    }
}

pub struct MapStats {
    pub registered: usize,
    pub skipped: Vec<String>,
    pub mean_reproj: f64,
    pub ba_rounds: usize,
}

/// Canonical pair key (i < j).
#[inline]
fn pk(i: usize, j: usize) -> (usize, usize) {
    if i < j { (i, j) } else { (j, i) }
}

/// Look up verified matches between frames i and j from i's perspective,
/// plus relative pose X_j = R X_i + t (world = camera i).
fn pair_view<'a>(
    pairs: &'a HashMap<(usize, usize), TwoView>,
    i: usize,
    j: usize,
) -> Option<(&'a Vec<u32>, &'a Vec<u32>, Matrix3<f64>, Vector3<f64>)> {
    let key = pk(i, j);
    let tv = pairs.get(&key)?;
    if tv.i == i {
        Some((&tv.a, &tv.b, tv.r, tv.t))
    } else {
        Some((&tv.b, &tv.a, tv.r.transpose(), -tv.r.transpose() * tv.t))
    }
}

pub fn map2(
    frames: &[FrameInfo],
    cams: Vec<Cam>,
    pair_list: Vec<TwoView>,
    opts: &MapperOptions,
    log: &dyn Fn(&str),
) -> (Scene, MapStats) {
    let mut scene = Scene::new(frames.to_vec(), cams, opts.reproj_thr, opts.min_tri_angle);

    // Canonical pair map.
    let mut pairs: HashMap<(usize, usize), TwoView> = HashMap::new();
    for tv in pair_list {
        let key = pk(tv.i, tv.j);
        pairs.insert(key, tv);
    }
    // Sort pair keys by inlier count (desc) for initial-pair selection.
    // Tie-break by key so runs are deterministic (HashMap order is not).
    let mut order: Vec<(usize, usize)> = {
        let mut v: Vec<(usize, usize, usize)> = pairs
            .iter()
            .map(|(k, tv)| (k.0, k.1, if tv.i == k.0 { tv.a.len() } else { tv.b.len() }))
            .collect();
        v.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| (a.0, a.1).cmp(&(b.0, b.1))));
        v.into_iter().map(|(a, b, _)| (a, b)).collect()
    };
    // Prefer non-planar pairs for initialization (stable keeps inlier order).
    order.sort_by_key(|k| pairs.get(k).map(|tv| tv.planar).unwrap_or(false) as u8);

    // ---- 1. Initial pair ----
    let mut initialized = false;
    for &(i, j) in order.iter().take(50) {
        let tv = &pairs[&pk(i, j)];
        let (a, b, r, t) = if tv.i == i {
            (&tv.a, &tv.b, tv.r, tv.t)
        } else {
            (&tv.b, &tv.a, tv.r.transpose(), -tv.r.transpose() * tv.t)
        };
        let pose_i = Pose::identity();
        let pose_j = Pose { r, t };
        let cam_i = scene.cams[i];
        let cam_j = scene.cams[j];
        let mut made = 0usize;
        for (ka, kb) in a.iter().zip(b.iter()) {
            let x1 = cam_i.normalize(scene.frames[i].pts[*ka as usize][0], scene.frames[i].pts[*ka as usize][1]);
            let x2 = cam_j.normalize(scene.frames[j].pts[*kb as usize][0], scene.frames[j].pts[*kb as usize][1]);
            if let Some(x) = triangulate(&pose_i.r, &pose_i.t, &pose_j.r, &pose_j.t, x1, x2) {
                let ang = tri_angle_deg(&pose_i, &pose_j, &x);
                if ang < opts.min_tri_angle.max(2.0) {
                    continue;
                }
                let ok_i = pose_i.project(&cam_i, &x).map_or(false, |(u, v)| {
                    (u - scene.frames[i].pts[*ka as usize][0]).hypot(v - scene.frames[i].pts[*ka as usize][1]) <= opts.reproj_thr
                });
                let ok_j = pose_j.project(&cam_j, &x).map_or(false, |(u, v)| {
                    (u - scene.frames[j].pts[*kb as usize][0]).hypot(v - scene.frames[j].pts[*kb as usize][1]) <= opts.reproj_thr
                });
                if ok_i && ok_j {
                    made += 1;
                }
            }
        }
        let needed = (a.len() as f64 * 0.5).max(30.0) as usize;
        if made >= needed {
            scene.register(i, pose_i);
            scene.register(j, pose_j);
            // Create the initial points.
            for (ka, kb) in a.iter().zip(b.iter()) {
                let x1 = cam_i.normalize(scene.frames[i].pts[*ka as usize][0], scene.frames[i].pts[*ka as usize][1]);
                let x2 = cam_j.normalize(scene.frames[j].pts[*kb as usize][0], scene.frames[j].pts[*kb as usize][1]);
                if let Some(x) = triangulate(&pose_i.r, &pose_i.t, &pose_j.r, &pose_j.t, x1, x2) {
                    let ang = tri_angle_deg(&pose_i, &pose_j, &x);
                    if ang < opts.min_tri_angle.max(2.0) {
                        continue;
                    }
                    let ok_i = pose_i.project(&cam_i, &x).map_or(false, |(u, v)| {
                        (u - scene.frames[i].pts[*ka as usize][0]).hypot(v - scene.frames[i].pts[*ka as usize][1]) <= opts.reproj_thr
                    });
                    let ok_j = pose_j.project(&cam_j, &x).map_or(false, |(u, v)| {
                        (u - scene.frames[j].pts[*kb as usize][0]).hypot(v - scene.frames[j].pts[*kb as usize][1]) <= opts.reproj_thr
                    });
                    if ok_i && ok_j {
                        scene.create_point(i, *ka, j, *kb, x);
                    }
                }
            }
            initialized = true;
            log(&format!(
                "initial pair: {} <-> {} ({} matches, {} triangulated)",
                frames[i].name, frames[j].name, a.len(), made
            ));
            break;
        }
    }
    if !initialized {
        let stats = MapStats {
            registered: 0,
            skipped: frames.iter().map(|f| f.name.clone()).collect(),
            mean_reproj: 0.0,
            ba_rounds: 0,
        };
        return (scene, stats);
    }

    log(&format!(
        "pair quality (top 8): {:?}",
        {
            let mut v: Vec<(usize, usize, usize)> = pairs
                .iter()
                .map(|(k, tv)| {
                    let n = if tv.i == k.0 { tv.a.len() } else { tv.b.len() };
                    (k.0, k.1, n)
                })
                .collect();
            v.sort_by(|a, b| b.2.cmp(&a.2));
            v.into_iter().take(8).map(|(a, b, n)| (a, b, n)).collect::<Vec<_>>()
        }
    ));

    // ---- 2. Incremental registration ----
    let n_frames = frames.len();
    let mut attempted: Vec<usize> = Vec::new();
    let mut ba_rounds = 0usize;

    loop {
        // Next best view: most reconstructed points visible.
        let mut best: Option<(usize, usize, Vec<(u32, usize)>)> = None;
        for f in 0..n_frames {
            if scene.is_registered(f) || attempted.contains(&f) {
                continue;
            }
            let mut seen: HashMap<u32, usize> = HashMap::new();
            for &r in scene.registered.iter() {
                let (a, b, _, _) = match pair_view(&pairs, f, r) {
                    Some(v) => v,
                    None => continue,
                };
                for (kf, kr) in a.iter().zip(b.iter()) {
                    if let Some(pi) = scene.kp_point[r][*kr as usize] {
                        seen.entry(*kf).or_insert(pi as usize);
                    }
                }
            }
            let cnt = seen.len();
            if cnt > best.as_ref().map_or(0, |(_, c, _)| *c) {
                let corrs: Vec<(u32, usize)> = seen.into_iter().collect();
                best = Some((f, cnt, corrs));
            }
        }
        let (f, cnt, corrs) = match best {
            Some((f, c, corr)) if c >= opts.abs_min_inliers => (f, c, corr),
            _ => {
                let vis: Vec<usize> = (0..n_frames)
                    .filter(|&g| !scene.is_registered(g) && !attempted.contains(&g))
                    .map(|g| {
                        let mut n = 0;
                        for &r in scene.registered.iter() {
                            if let Some((a, b, _, _)) = pair_view(&pairs, g, r) {
                                for (kf, kr) in a.iter().zip(b.iter()) {
                                    if scene.kp_point[r][*kr as usize].is_some() {
                                        n += 1;
                                        break;
                                    }
                                }
                            }
                        }
                        n
                    })
                    .collect();
                log(&format!("registration stopped; visible-count of remaining frames: {:?}", vis));
                break;
            }
        };

        // PnP registration.
        let pixels: Vec<[f64; 2]> = corrs
            .iter()
            .map(|&(k, _)| scene.frames[f].pts[k as usize])
            .collect();
        let pts3d: Vec<Vector3<f64>> = corrs
            .iter()
            .map(|&(_, pi)| scene.points[pi].xyz)
            .collect();
        let cam = scene.cams[f];
        let reg = pnp_ransac(&pixels, &pts3d, &cam, opts.reproj_thr * 1.5);
        let (pose, inl) = match reg {
            Some((p, i)) => (p, i),
            None => {
                attempted.push(f);
                continue;
            }
        };
        if inl.len() < opts.abs_min_inliers.max(12) {
            attempted.push(f);
            continue;
        }
        scene.register(f, pose);

        // Extend existing tracks with the newly registered image.
        let others: Vec<usize> =
            scene.registered.iter().take(scene.registered.len() - 1).copied().collect();
        for &r in others.iter() {
            let (a, b, _, _) = match pair_view(&pairs, f, r) {
                Some(v) => v,
                None => continue,
            };
            for (_kf, kr) in a.iter().zip(b.iter()) {
                if let Some(pi) = scene.kp_point[r][*kr as usize] {
                    scene.add_observation(pi as usize, f, *kf);
                }
            }
        }

        // Triangulate brand-new points.
        for &r in others.iter() {
            let (a, b, _, _) = match pair_view(&pairs, f, r) {
                Some(v) => v,
                None => continue,
            };
            let pose_r = match scene.pose(r) {
                Some(p) => Pose { r: p.r, t: p.t },
                None => continue,
            };
            let pf: Pose = scene.pose(f).copied().unwrap();
            for (kf, kr) in a.iter().zip(b.iter()) {
                if scene.kp_point[f][*kf as usize].is_some()
                    || scene.kp_point[r][*kr as usize].is_some()
                {
                    continue;
                }
                let x1 = cam.normalize(scene.frames[f].pts[*kf as usize][0], scene.frames[f].pts[*kf as usize][1]);
                let x2 = scene.cams[r].normalize(scene.frames[r].pts[*kr as usize][0], scene.frames[r].pts[*kr as usize][1]);
                if let Some(x) = triangulate(&pf.r, &pf.t, &pose_r.r, &pose_r.t, x1, x2) {
                    let ang = tri_angle_deg(&pf, &pose_r, &x);
                    if ang < opts.min_tri_angle {
                        continue;
                    }
                    let ok_f = pf.project(&cam, &x).map_or(false, |(u, v)| {
                        (u - scene.frames[f].pts[*kf as usize][0]).hypot(v - scene.frames[f].pts[*kf as usize][1]) <= opts.reproj_thr
                    });
                    let ok_r = pose_r.project(&scene.cams[r], &x).map_or(false, |(u, v)| {
                        (u - scene.frames[r].pts[*kr as usize][0]).hypot(v - scene.frames[r].pts[*kr as usize][1]) <= opts.reproj_thr
                    });
                    if ok_f && ok_r {
                        scene.create_point(f, *kf, r, *kr, x);
                    }
                }
            }
        }

        // COLMAP-style refinement after every registration: BA -> filter
        // -> re-triangulate -> filter. Keeps drift in check as the model
        // grows and repairs early low-parallax triangulations.
        ba_global(&mut scene, opts.ba_iters);
        ba_rounds += 1;
        scene.filter_points();
        scene.retriangulate();
        scene.filter_points();
        log(&format!(
            "registered {} ({}/{} visible pts) | total points {}",
            frames[f].name,
            cnt,
            corrs.len(),
            scene.points.len()
        ));
        attempted.clear();
    }

    // ---- 3. Final BA + filter ----
    ba_global(&mut scene, opts.ba_iters.max(30));
    ba_rounds += 1;
    scene.filter_points();

    let stats = MapStats {
        registered: scene.registered.len(),
        skipped: (0..n_frames)
            .filter(|&f| !scene.is_registered(f))
            .map(|f| frames[f].name.clone())
            .collect(),
        mean_reproj: scene.last_mean_err,
        ba_rounds,
    };
    (scene, stats)
}

/// Global BA over the whole scene (first registered camera pose fixed).
pub fn ba_global(scene: &mut Scene, iters: usize) {
    let first_cam = scene.registered[0];
    let mut cams: Vec<BaCam> = Vec::with_capacity(scene.registered.len());
    let mut cam_of_frame: HashMap<usize, usize> = HashMap::new();
    for (n, &f) in scene.registered.iter().enumerate() {
        let (q, t) = match scene.pose(f) {
            Some(p) => (crate::geom::rot_to_quat(&p.r), [p.t.x, p.t.y, p.t.z]),
            None => continue,
        };
        let c = scene.cams[f];
        cams.push(BaCam { q, t, f: c.f, cx: c.cx, cy: c.cy, k: c.k });
        cam_of_frame.insert(f, n);
    }
    let mut pts: Vec<[f64; 3]> = scene.points.iter().map(|p| [p.xyz.x, p.xyz.y, p.xyz.z]).collect();
    let mut obs: Vec<BaObs> = Vec::new();
    for p in 0..pts.len() {
        for &(f, k) in &scene.points[p].track {
            if let Some(&c) = cam_of_frame.get(&f) {
                let uv = scene.frames[f].pts[k as usize];
                obs.push(BaObs { cam: c, pt: p, u: uv[0], v: uv[1] });
            }
        }
    }
    if obs.len() < 20 {
        return;
    }
    ba::bundle_adjust(&mut cams, &mut pts, &obs, &[0], iters, 1.5);
    // Write back.
    for (n, &f) in scene.registered.iter().enumerate() {
        let c = &cams[n];
        let q = {
            let mut qq = c.q;
            let nq = qq.iter().map(|v| v * v).sum::<f64>().sqrt();
            for v in qq.iter_mut() {
                *v /= nq;
            }
            qq
        };
        scene.poses[f] = Some(Pose {
            r: crate::geom::quat_to_rot(&q),
            t: Vector3::new(c.t[0], c.t[1], c.t[2]),
        });
        scene.cams[f] = Cam { f: c.f, cx: c.cx, cy: c.cy, k: c.k };
    }
    for (p, xyz) in pts.into_iter().enumerate() {
        scene.points[p].xyz = Vector3::new(xyz[0], xyz[1], xyz[2]);
    }
    let _ = first_cam;
}
