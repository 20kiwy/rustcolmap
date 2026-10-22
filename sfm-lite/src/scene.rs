//! In-memory reconstruction state shared by the incremental mapper:
//! per-frame keypoint data, per-frame cameras, poses, 3D points with tracks,
//! and conversion to the COLMAP `Reconstruction` model.

use std::collections::HashMap;

use nalgebra::{Matrix3, Vector3};

use colmap_io::{
    Camera as CCamera, Image as CImage, Point2D as CPoint2D, Point3D as CPoint3D,
    Reconstruction, TrackElement, SIMPLE_RADIAL,
};

use crate::geom::{Cam, Pose, rot_to_quat};

#[derive(Clone)]
pub struct FrameInfo {
    pub name: String,
    /// Downscaled (working) resolution.
    pub width: u32,
    pub height: u32,
    /// Original file resolution (output is exported at full resolution).
    pub orig_width: u32,
    pub orig_height: u32,
    pub pts: Vec<[f64; 2]>,
    pub colors: Vec<[u8; 3]>,
}

#[derive(Debug, Clone)]
pub struct Pt {
    pub xyz: Vector3<f64>,
    pub rgb: [u8; 3],
    pub err: f64,
    /// (frame idx, keypoint idx)
    pub track: Vec<(usize, u32)>,
}

pub struct Scene {
    pub frames: Vec<FrameInfo>,
    pub cams: Vec<Cam>,
    pub poses: Vec<Option<Pose>>,
    /// Registered frame indices in registration order.
    pub registered: Vec<usize>,
    pub points: Vec<Pt>,
    /// kp_point[f][k] = Some(point idx) when keypoint k of frame f is
    /// assigned to a 3D point.
    pub kp_point: Vec<Vec<Option<u32>>>,
    pub reproj_thr: f64,
    pub min_tri_angle: f64,
    /// Mean reprojection error of the last filtering pass (px).
    pub last_mean_err: f64,
}

impl Scene {
    pub fn new(frames: Vec<FrameInfo>, cams: Vec<Cam>, reproj_thr: f64, min_tri_angle: f64) -> Self {
        let kp_point = frames.iter().map(|f| vec![None; f.pts.len()]).collect();
        Scene {
            poses: vec![None; frames.len()],
            frames,
            cams,
            registered: Vec::new(),
            points: Vec::new(),
            kp_point,
            reproj_thr,
            min_tri_angle,
            last_mean_err: 0.0,
        }
    }

    pub fn is_registered(&self, f: usize) -> bool {
        self.poses[f].is_some()
    }

    pub fn pose(&self, f: usize) -> Option<&Pose> {
        self.poses[f].as_ref()
    }

    pub fn register(&mut self, f: usize, pose: Pose) {
        debug_assert!(!self.is_registered(f));
        self.poses[f] = Some(pose);
        self.registered.push(f);
    }

    /// Try to attach keypoint (f, k) to existing point p. Requires the
    /// reprojection to be inside the threshold.
    pub fn add_observation(&mut self, p: usize, f: usize, k: u32) -> bool {
        if self.kp_point[f][k as usize].is_some() {
            return false;
        }
        let xyz = self.points[p].xyz;
        let uv = self.frames[f].pts[k as usize];
        let pose = match self.poses[f] {
            Some(p) => p,
            None => return false,
        };
        let cam = self.cams[f];
        match pose.project(&cam, &xyz) {
            Some((u, v)) if (u - uv[0]).hypot(v - uv[1]) <= self.reproj_thr => {
                self.kp_point[f][k as usize] = Some(p as u32);
                self.points[p].track.push((f, k));
                true
            }
            _ => false,
        }
    }

    /// Create a new point triangulated from two views.
    pub fn create_point(&mut self, fa: usize, ka: u32, fb: usize, kb: u32, xyz: Vector3<f64>) -> Option<usize> {
        if self.kp_point[fa][ka as usize].is_some() || self.kp_point[fb][kb as usize].is_some() {
            return None;
        }
        let rgb = self.frames[fa].colors[ka as usize];
        let id = self.points.len();
        self.points.push(Pt {
            xyz,
            rgb,
            err: 0.0,
            track: vec![(fa, ka), (fb, kb)],
        });
        self.kp_point[fa][ka as usize] = Some(id as u32);
        self.kp_point[fb][kb as usize] = Some(id as u32);
        Some(id)
    }

    /// COLMAP-style filtering: drop observations with large reprojection
    /// error or negative depth, then delete points whose track got too short.
    /// Returns (removed observations, removed points).
    pub fn filter_points(&mut self) -> (usize, usize) {
        let thr = self.reproj_thr;
        // Scale-free near-plane guard: drop observations whose depth is
        // below 0.1% of the median scene depth (numerically degenerate).
        let mut depths: Vec<f64> = Vec::new();
        for (fi, pose_opt) in self.poses.iter().enumerate() {
            if let Some(pose) = pose_opt {
                for slot in self.kp_point[fi].iter().flatten() {
                    let p = &self.points[*slot as usize];
                    let xc = pose.r * p.xyz + pose.t;
                    if xc.z > 0.0 {
                        depths.push(xc.z);
                    }
                }
            }
        }
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let z_guard = if depths.is_empty() {
            0.0
        } else {
            1e-3 * depths[depths.len() / 2]
        };
        let mut removed_obs = 0usize;
        // Recompute errors, collect bad observations.
        let mut keep: Vec<Vec<(usize, u32)>> = Vec::with_capacity(self.points.len());
        let mut errs = Vec::with_capacity(self.points.len());
        for p in self.points.iter_mut() {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            let mut kept = Vec::with_capacity(p.track.len());
            for &(f, k) in &p.track {
                let pose = self.poses[f];
                let uv = self.frames[f].pts[k as usize];
                let cam = self.cams[f];
                let depth_ok = pose
                    .map_or(false, |po| (po.r * p.xyz + po.t).z >= z_guard);
                let ok = depth_ok
                    && pose
                        .and_then(|po| po.project(&cam, &p.xyz))
                        .map_or(false, |(u, v)| (u - uv[0]).hypot(v - uv[1]) <= thr);
                if ok {
                    if let Some(po) = pose {
                        if let Some((u, v)) = po.project(&cam, &p.xyz) {
                            sum += (u - uv[0]).hypot(v - uv[1]);
                            cnt += 1;
                        }
                    }
                    kept.push((f, k));
                } else {
                    removed_obs += 1;
                }
            }
            p.track = kept;
            p.err = if cnt > 0 { sum / cnt as f64 } else { f64::INFINITY };
            keep.push(p.track.clone());
            errs.push(p.err);
        }
        // Sync kp_point with kept tracks.
        for (pi, tr) in keep.iter().enumerate() {
            for &(f, k) in tr {
                self.kp_point[f][k as usize] = Some(pi as u32);
            }
        }
        // Delete points with < 2 observations.
        let mut dead = vec![false; self.points.len()];
        let mut removed_pts = 0usize;
        for (pi, p) in self.points.iter().enumerate() {
            if p.track.len() < 2 || !p.err.is_finite() {
                dead[pi] = true;
                removed_pts += 1;
            }
        }
        // Compact point list, remap indices.
        let mut remap: HashMap<usize, usize> = HashMap::new();
        let mut new_points = Vec::with_capacity(self.points.len());
        for (pi, p) in self.points.drain(..).enumerate() {
            if !dead[pi] {
                remap.insert(pi, new_points.len());
                new_points.push(p);
            }
        }
        for f in 0..self.kp_point.len() {
            for slot in self.kp_point[f].iter_mut() {
                if let Some(pi) = slot {
                    match remap.get(&(*pi as usize)) {
                        Some(&np) => *slot = Some(np as u32),
                        None => *slot = None,
                    }
                }
            }
        }
        self.points = new_points;
        let mean = if self.points.is_empty() {
            0.0
        } else {
            self.points.iter().map(|p| p.err).sum::<f64>() / self.points.len() as f64
        };
        self.last_mean_err = mean;
        (removed_obs, removed_pts)
    }

    /// Re-triangulate every point from the two track observations with the
    /// largest triangulation angle, using the current poses. Points whose
    /// re-triangulation fails the reproj gate keep their previous position.
    pub fn retriangulate(&mut self) {
        use crate::geom::triangulate as tri;
        for pi in 0..self.points.len() {
            let track = self.points[pi].track.clone();
            if track.len() < 2 {
                continue;
            }
            let mut best: Option<((usize, u32), (usize, u32), f64)> = None;
            for i in 0..track.len() {
                for j in (i + 1)..track.len() {
                    let (fa, ka) = track[i];
                    let (fb, kb) = track[j];
                    let (pa, pb) = match (self.poses[fa], self.poses[fb]) {
                        (Some(a), Some(b)) => (a, b),
                        _ => continue,
                    };
                    let x1 = self.cams[fa].normalize(self.frames[fa].pts[ka as usize][0], self.frames[fa].pts[ka as usize][1]);
                    let x2 = self.cams[fb].normalize(self.frames[fb].pts[kb as usize][0], self.frames[fb].pts[kb as usize][1]);
                    if let Some(x) = tri(&pa.r, &pa.t, &pb.r, &pb.t, x1, x2) {
                        let ang = crate::geom::tri_angle_deg(&pa, &pb, &x);
                        if ang > best.map_or(0.0, |(_, _, a)| a) {
                            best = Some(((fa, ka), (fb, kb), ang));
                        }
                    }
                }
            }
            if let Some(((fa, ka), (fb, kb), ang)) = best {
                if ang < self.min_tri_angle.max(2.0) {
                    continue;
                }
                let pa = self.poses[fa].unwrap();
                let pb = self.poses[fb].unwrap();
                let x1 = self.cams[fa].normalize(self.frames[fa].pts[ka as usize][0], self.frames[fa].pts[ka as usize][1]);
                let x2 = self.cams[fb].normalize(self.frames[fb].pts[kb as usize][0], self.frames[fb].pts[kb as usize][1]);
                if let Some(x) = tri(&pa.r, &pa.t, &pb.r, &pb.t, x1, x2) {
                    let ca = self.cams[fa];
                    let cb = self.cams[fb];
                    let ok_a = pa.project(&ca, &x).map_or(false, |(u, v)| {
                        let q = self.frames[fa].pts[ka as usize];
                        (u - q[0]).hypot(v - q[1]) <= self.reproj_thr
                    });
                    let ok_b = pb.project(&cb, &x).map_or(false, |(u, v)| {
                        let q = self.frames[fb].pts[kb as usize];
                        (u - q[0]).hypot(v - q[1]) <= self.reproj_thr
                    });
                    if ok_a && ok_b {
                        self.points[pi].xyz = x;
                    }
                }
            }
        }
    }

    /// Convert to the COLMAP model, rescaling camera parameters and 2D
    /// points from the working resolution to the original file resolution.
    pub fn to_reconstruction(&self) -> Reconstruction {
        let mut cameras: Vec<CCamera> = Vec::new();
        let mut images: Vec<CImage> = Vec::new();
        let mut points3d: Vec<CPoint3D> = Vec::new();

        let id_of: HashMap<usize, i32> = self
            .registered
            .iter()
            .enumerate()
            .map(|(n, &f)| (f, (n + 1) as i32))
            .collect();

        for (n, &f) in self.registered.iter().enumerate() {
            let frame = &self.frames[f];
            let cam = self.cams[f];
            let pose = self.poses[f].unwrap();
            let sx = frame.orig_width as f64 / frame.width as f64;
            let sy = frame.orig_height as f64 / frame.height as f64;
            let s = 0.5 * (sx + sy);
            let img_id = (n + 1) as i32;
            cameras.push(CCamera {
                camera_id: img_id,
                model_id: SIMPLE_RADIAL,
                width: frame.orig_width as u64,
                height: frame.orig_height as u64,
                params: vec![cam.f * s, cam.cx * sx, cam.cy * sy, cam.k],
            });
            let mut points2d: Vec<CPoint2D> = frame
                .pts
                .iter()
                .map(|p| CPoint2D {
                    x: p[0] * sx,
                    y: p[1] * sy,
                    point3d_id: -1,
                })
                .collect();
            for (k, slot) in self.kp_point[f].iter().enumerate() {
                if let Some(pi) = slot {
                    points2d[k].point3d_id = *pi as i64 + 1;
                }
            }
            images.push(CImage {
                image_id: img_id,
                qvec: rot_to_quat(&pose.r),
                tvec: [pose.t.x, pose.t.y, pose.t.z],
                camera_id: img_id,
                name: frame.name.clone(),
                points2d,
            });
        }

        for (pi, p) in self.points.iter().enumerate() {
            let sx = self.frames[p.track[0].0].orig_width as f64
                / self.frames[p.track[0].0].width as f64;
            points3d.push(CPoint3D {
                point3d_id: (pi + 1) as i64,
                xyz: [p.xyz.x, p.xyz.y, p.xyz.z],
                rgb: p.rgb,
                error: p.err * sx,
                track: p
                    .track
                    .iter()
                    .filter_map(|&(f, k)| {
                        id_of.get(&f).map(|&id| TrackElement {
                            image_id: id,
                            point2d_idx: k,
                        })
                    })
                    .collect(),
            });
        }

        let mut rec = Reconstruction { cameras, images, points3d };
        rec.sort_all();
        rec
    }
}

/// Unused import guard (Matrix3 kept for API stability).
#[allow(dead_code)]
type _M3 = Matrix3<f64>;
