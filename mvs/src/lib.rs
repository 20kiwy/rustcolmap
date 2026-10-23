//! mvs — dense multi-view stereo for the rustcolmap pipeline.
//!
//! This is the Rust-native counterpart of COLMAP's
//! `patch_match_stereo` + `stereo_fusion` stages: it takes the sparse
//! reconstruction (COLMAP `sparse/0` folder or an in-memory
//! `colmap_io::Reconstruction`) plus the source photos and produces
//!
//! * per-view depth maps via slanted-plane patch-match stereo
//!   (census cost, plane-sweep init, red-black propagation), then
//! * a fused, voxel-deduplicated DENSE colored point cloud written as
//!   binary PLY — the "full 3D" output that dense viewers and Gaussian
//!   Splatting initializers expect.
//!
//! Everything runs on CPU (std + rayon), no GPU required.

pub mod fuse;
pub mod stereo;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use nalgebra::Vector3;
use rayon::prelude::*;

use colmap_io::Reconstruction;
use sfm_lite::geom::{quat_to_rot, Cam, Pose};

// ---------------------------------------------------------------------------
// Options / stats
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MvsOptions {
    /// Longest image side used for stereo (everything is computed at this
    /// reduced resolution; colors are sampled from the same image).
    pub max_dim: u32,
    /// Number of source (support) views per reference view.
    pub num_sources: usize,
    /// Patch-match sweeps (each sweep = black + red propagation passes).
    pub sweeps: usize,
    /// Plane-sweep initial depth hypotheses per pixel.
    pub num_hyp: usize,
    /// Minimum number of OTHER views whose depth must agree for fusion.
    pub min_consistent: usize,
    /// Relative depth agreement threshold for cross-view consistency.
    pub depth_diff: f64,
    /// Max census hamming fraction for a pixel to be considered valid.
    pub cost_thr: f32,
    /// Voxel cell size relative to the sparse-bounding-box diagonal.
    pub voxel_rel: f64,
    /// Save per-view depth-map visualizations (grayscale PNGs).
    pub save_depth_vis: bool,
}

impl Default for MvsOptions {
    fn default() -> Self {
        MvsOptions {
            max_dim: 900,
            num_sources: 3,
            sweeps: 4,
            num_hyp: 48,
            min_consistent: 2,
            depth_diff: 0.03,
            cost_thr: 0.38,
            voxel_rel: 0.0015,
            save_depth_vis: false,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct MvsStats {
    pub views: usize,
    pub views_with_depth: usize,
    pub dense_points: usize,
    pub per_view_valid: Vec<usize>,
    pub elapsed_s: f32,
}

// ---------------------------------------------------------------------------
// View loading
// ---------------------------------------------------------------------------

/// One registered view, prepared at the stereo working resolution.
pub struct ViewData {
    pub name: String,
    pub image_id: i32,
    pub w: i32,
    pub h: i32,
    /// Camera intrinsics at the working resolution.
    pub cam: Cam,
    /// World->camera pose.
    pub pose: Pose,
    /// Grayscale in [0, 1], row-major, len = w*h.
    pub gray: Vec<f32>,
    /// RGB at the working resolution (for fused cloud colors).
    pub rgb: Vec<u8>,
    /// 7x7 census descriptor (48 bits in a u64), len = w*h.
    pub census: Vec<u64>,
}

fn cam_from_rec(c: &colmap_io::Camera, sx: f64, sy: f64) -> Option<Cam> {
    // Export model is SIMPLE_RADIAL [f, cx, cy, k]; other models are
    // approximated with k = 0 (MVS tolerates this well).
    let p = &c.params;
    let g = |i: usize| p.get(i).copied().unwrap_or(0.0);
    let (f, cx, cy, k) = match c.model_id {
        colmap_io::SIMPLE_PINHOLE => (g(0), g(1), g(2), 0.0),
        colmap_io::SIMPLE_RADIAL => (g(0), g(1), g(2), g(3)),
        colmap_io::RADIAL => (g(0), g(2), g(3), 0.0),
        colmap_io::PINHOLE | colmap_io::OPENCV => (0.5 * (g(0) + g(1)), g(2), g(3), 0.0),
        _ => (g(0), g(2), g(3), 0.0),
    };
    let s = 0.5 * (sx + sy);
    Some(Cam { f: f * s, cx: cx * sx, cy: cy * sy, k })
}

fn census_map(gray: &[f32], w: i32, h: i32) -> Vec<u64> {
    let mut out = vec![0u64; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let c = gray[(y * w + x) as usize];
            let mut bits = 0u64;
            for dy in -3i32..=3 {
                for dx in -3i32..=3 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let xx = (x + dx).clamp(0, w - 1) as usize;
                    let yy = (y + dy).clamp(0, h - 1) as usize;
                    let bit = ((dy + 3) * 7 + (dx + 3)) as u64;
                    if gray[yy * w as usize + xx] < c {
                        bits |= 1u64 << bit;
                    }
                }
            }
            out[(y * w + x) as usize] = bits;
        }
    }
    out
}

fn load_views(
    rec: &Reconstruction,
    image_dir: &Path,
    opts: &MvsOptions,
    log: &(dyn Fn(&str) + Sync),
) -> Result<Vec<std::sync::Arc<ViewData>>, String> {
    let mut views: Vec<(i32, String, Cam, Pose, u32, u32)> = Vec::new();
    for img in &rec.images {
        let cam_rec = rec
            .camera(img.camera_id)
            .ok_or_else(|| format!("image {} references missing camera {}", img.name, img.camera_id))?;
        let s = (opts.max_dim as f64 / cam_rec.width.max(cam_rec.height) as f64).min(1.0);
        let mw = ((cam_rec.width as f64 * s).round() as u32).max(16);
        let mh = ((cam_rec.height as f64 * s).round() as u32).max(16);
        let cam = cam_from_rec(
            cam_rec,
            mw as f64 / cam_rec.width as f64,
            mh as f64 / cam_rec.height as f64,
        )
        .ok_or_else(|| format!("unsupported camera model {} for {}", cam_rec.model_id, img.name))?;
        let r = quat_to_rot(&img.qvec);
        let t = Vector3::new(img.tvec[0], img.tvec[1], img.tvec[2]);
        views.push((
            img.image_id,
            img.name.clone(),
            cam,
            Pose { r, t },
            cam_rec.width as u32,
            cam_rec.height as u32,
        ));
    }
    if views.len() < 2 {
        return Err("need at least 2 registered views for MVS".into());
    }

    let loaded: Vec<std::sync::Arc<ViewData>> = views
        .par_iter()
        .map(|(image_id, name, cam, pose, ow, oh)| {
            let path = image_dir.join(name);
            let img = image::open(&path)
                .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
            let s = (opts.max_dim as f64 / (*ow).max(*oh) as f64).min(1.0);
            let mw = (((*ow as f64) * s).round() as u32).max(16);
            let mh = (((*oh as f64) * s).round() as u32).max(16);
            let small = if (mw, mh) != (*ow, *oh) {
                img.resize_exact(mw, mh, image::imageops::FilterType::Triangle)
            } else {
                img
            };
            let luma = small.to_luma8();
            let rgb = small.to_rgb8();
            let (mw, mh) = luma.dimensions();
            let gray: Vec<f32> = luma.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
            let rgbv: Vec<u8> = rgb.pixels().flat_map(|p| p.0).collect();
            let cen = census_map(&gray, mw as i32, mh as i32);
            Ok(std::sync::Arc::new(ViewData {
                name: name.clone(),
                image_id: *image_id,
                w: mw as i32,
                h: mh as i32,
                cam: *cam,
                pose: *pose,
                gray,
                rgb: rgbv,
                census: cen,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    log(&format!(
        "mvs: loaded {} views at working resolution (max_dim {})",
        loaded.len(),
        opts.max_dim
    ));
    Ok(loaded)
}

// ---------------------------------------------------------------------------
// Co-visibility + depth ranges
// ---------------------------------------------------------------------------

fn covisibility(rec: &Reconstruction, idx_of_image_id: &HashMap<i32, usize>, n: usize) -> Vec<Vec<u32>> {
    let mut cov = vec![vec![0u32; n]; n];
    for pt in &rec.points3d {
        let mut ids: Vec<usize> = pt
            .track
            .iter()
            .filter_map(|te| idx_of_image_id.get(&te.image_id).copied())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        for &a in &ids {
            for &b in &ids {
                if a != b {
                    cov[a][b] += 1;
                }
            }
        }
    }
    cov
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round().clamp(0.0, sorted.len() as f64 - 1.0) as usize;
    sorted[idx]
}

fn depth_range_for(rec: &Reconstruction, view: &ViewData, global: &[f64]) -> (f64, f64) {
    let mut ds: Vec<f64> = Vec::new();
    for pt in &rec.points3d {
        if pt.track.iter().any(|te| te.image_id == view.image_id) {
            let xc = view.pose.r * Vector3::new(pt.xyz[0], pt.xyz[1], pt.xyz[2]) + view.pose.t;
            if xc.z > 1e-6 {
                ds.push(xc.z);
            }
        }
    }
    let src: &[f64] = if ds.len() >= 8 {
        ds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        &ds
    } else {
        global
    };
    let (lo, hi) = if src.len() >= 8 {
        (percentile(src, 0.02), percentile(src, 0.98))
    } else {
        (0.0, 0.0)
    };
    if hi <= lo * 1.05 || lo <= 0.0 {
        // last-resort fallback: full global span
        let mut g2 = global.to_vec();
        g2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let (glo, ghi) = (percentile(&g2, 0.01), percentile(&g2, 0.99));
        return ((glo * 0.7).max(1e-3), ghi * 1.4);
    }
    (lo * 0.8, hi * 1.25)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the full dense stage: depth maps for every registered view +
/// geometric-consistency fusion, writing
///
/// * `<out_dir>/dense/dense.ply`          — fused dense colored point cloud
/// * `<out_dir>/dense/mvs_stats.json`     — run statistics
/// * `<out_dir>/dense/depth_vis/*.png`    — depth visualizations (optional)
pub fn run_mvs(
    rec: &Reconstruction,
    image_dir: &Path,
    out_dir: &Path,
    opts: &MvsOptions,
    log: &(dyn Fn(&str) + Sync),
) -> Result<MvsStats, String> {
    let t_all = Instant::now();

    // Global depth range (fallback + sanity).
    let mut global_depths: Vec<f64> = rec
        .points3d
        .iter()
        .map(|p| Vector3::new(p.xyz[0], p.xyz[1], p.xyz[2]).norm())
        .collect();
    global_depths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    if global_depths.len() < 8 {
        return Err(format!(
            "sparse reconstruction too small for MVS ({} points, need >= 8)",
            rec.points3d.len()
        ));
    }

    let views = load_views(rec, image_dir, opts, log)?;
    let idx_of_image_id: HashMap<i32, usize> = views
        .iter()
        .enumerate()
        .map(|(i, v)| (v.image_id, i))
        .collect();
    let cov = covisibility(rec, &idx_of_image_id, views.len());

    // Source selection: top-K co-visible views.
    let sources: Vec<Vec<usize>> = (0..views.len())
        .map(|i| {
            let mut order: Vec<(u32, usize)> = cov[i]
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, c)| c >= 10)
                .map(|(j, c)| (c, j))
                .collect();
            order.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            order
                .into_iter()
                .take(opts.num_sources)
                .map(|(_, j)| j)
                .collect()
        })
        .collect();
    let n_with_sources = sources.iter().filter(|s| !s.is_empty()).count();
    log(&format!(
        "mvs: {} views with >=1 source view (K = {})",
        n_with_sources, opts.num_sources
    ));

    // ---- Depth maps (parallel over reference views) ----
    let t_stereo = Instant::now();
    let depth_maps: Vec<Option<stereo::DepthMap>> = (0..views.len())
        .into_par_iter()
        .map(|ri| {
            let srcs = &sources[ri];
            if srcs.is_empty() {
                return None;
            }
            let (zmin, zmax) = depth_range_for(rec, &views[ri], &global_depths);
            let t0 = Instant::now();
            let dm = stereo::estimate_depth(&views, ri, srcs, zmin, zmax, opts);
            let valid = dm.depth.iter().filter(|&&z| z > 0.0).count();
            log(&format!(
                "mvs: [{}] depth {:?}-{:?} ({} srcs) valid {}/{} px in {:.1}s",
                views[ri].name,
                zmin,
                zmax,
                srcs.len(),
                valid,
                dm.depth.len(),
                t0.elapsed().as_secs_f32()
            ));
            Some(dm)
        })
        .collect();
    let stereo_s = t_stereo.elapsed().as_secs_f32();
    log(&format!(
        "mvs: stereo done in {:.1}s ({} depth maps)",
        stereo_s,
        depth_maps.iter().flatten().count()
    ));

    // ---- Optional depth visualizations ----
    if opts.save_depth_vis {
        let vis_dir = out_dir.join("dense").join("depth_vis");
        std::fs::create_dir_all(&vis_dir).map_err(|e| e.to_string())?;
        for (ri, dm) in depth_maps.iter().enumerate() {
            if let Some(dm) = dm {
                let _ = fuse::save_depth_vis(&vis_dir, &views[ri].name, &dm.depth, views[ri].w, views[ri].h);
            }
        }
        log("mvs: depth visualizations saved");
    }

    // ---- Fusion ----
    // Fusion may confirm a point against ANY co-visible view with a depth
    // map (not just the stereo sources) — this recovers coverage where one
    // or two source depth maps happen to be invalid at that pixel.
    let mut fusion_order: Vec<Vec<usize>> = (0..views.len())
        .map(|i| {
            let mut order: Vec<(u32, usize)> = cov[i]
                .iter()
                .copied()
                .enumerate()
                .filter(|&(j, c)| c >= 10 && depth_maps[j].is_some())
                .map(|(j, c)| (c, j))
                .collect();
            order.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            order.into_iter().map(|(_, j)| j).collect()
        })
        .collect();
    // keep stereo sources at the front (they were chosen for good baselines)
    for (i, s) in sources.iter().enumerate() {
        if let Some(order) = fusion_order.get_mut(i) {
            order.sort_by_key(|&j| {
                s.iter().position(|&x| x == j).unwrap_or(usize::MAX)
            });
        }
    }

    let t_fuse = Instant::now();
    let sparse: Vec<Vector3<f64>> = rec
        .points3d
        .iter()
        .map(|p| Vector3::new(p.xyz[0], p.xyz[1], p.xyz[2]))
        .collect();
    let pts = fuse::fuse_depth_maps(&views, &depth_maps, &fusion_order, &sparse, opts, log);
    let fuse_s = t_fuse.elapsed().as_secs_f32();

    let dense_dir = out_dir.join("dense");
    std::fs::create_dir_all(&dense_dir).map_err(|e| e.to_string())?;
    let ply_path = dense_dir.join("dense.ply");
    fuse::write_ply_binary(&ply_path, &pts).map_err(|e| format!("write {}: {e}", ply_path.display()))?;

    // Decimated JSON copy for web viewers (matches the dashboard format).
    // Outlier points (far above the robust radius) are dropped from the
    // display copy only — dense.ply stays complete.
    let cap = 400_000usize;
    if !pts.is_empty() {
        let cx: f64 = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
        let cy: f64 = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
        let cz: f64 = pts.iter().map(|p| p.2).sum::<f64>() / pts.len() as f64;
        let mut dists: Vec<f64> = pts
            .iter()
            .map(|p| {
                let dx = p.0 - cx;
                let dy = p.1 - cy;
                let dz = p.2 - cz;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let med = dists[dists.len() / 2];
        let p95 = dists[((dists.len() as f64 * 0.95) as usize).min(dists.len() - 1)];
        let r_keep = med + 6.0 * (p95 - med).abs().max(med.max(1e-9) * 0.5);
        let kept: Vec<&(f64, f64, f64, [u8; 3])> = pts
            .iter()
            .filter(|p| {
                let dx = p.0 - cx;
                let dy = p.1 - cy;
                let dz = p.2 - cz;
                (dx * dx + dy * dy + dz * dz).sqrt() <= r_keep
            })
            .collect();
        let stride = kept.len().div_ceil(cap);
        let web: Vec<&(f64, f64, f64, [u8; 3])> = kept.into_iter().step_by(stride).collect();
        let mut s = String::with_capacity(web.len() * 48);
        s.push_str("{\"n\":");
        s.push_str(&web.len().to_string());
        s.push_str(",\"pos\":[");
        for (i, (x, y, z, _)) in web.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{:.4},{:.4},{:.4}]", x, y, z));
        }
        s.push_str("],\"col\":[");
        for (i, (_, _, _, c)) in web.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{}, {},{}]", c[0], c[1], c[2]));
        }
        s.push_str("]}");
        let _ = std::fs::write(dense_dir.join("dense_web.json"), s);
    }

    let per_view_valid: Vec<usize> = depth_maps
        .iter()
        .map(|dm| {
            dm.as_ref()
                .map(|d| d.depth.iter().filter(|&&z| z > 0.0).count())
                .unwrap_or(0)
        })
        .collect();

    let stats = MvsStats {
        views: views.len(),
        views_with_depth: depth_maps.iter().flatten().count(),
        dense_points: pts.len(),
        per_view_valid,
        elapsed_s: t_all.elapsed().as_secs_f32(),
    };
    let stats_json = format!(
        "{{\"views\": {}, \"views_with_depth\": {}, \"dense_points\": {}, \"stereo_s\": {:.1}, \"fuse_s\": {:.1}, \"elapsed_s\": {:.1}}}",
        stats.views, stats.views_with_depth, stats.dense_points, stereo_s, fuse_s, stats.elapsed_s
    );
    let _ = std::fs::write(dense_dir.join("mvs_stats.json"), stats_json);

    log(&format!(
        "mvs: fused {} dense points -> {} in {:.1}s (total {:.1}s)",
        pts.len(),
        ply_path.display(),
        fuse_s,
        stats.elapsed_s
    ));
    Ok(stats)
}
