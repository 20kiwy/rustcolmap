//! sfm-lite v0.2 — a native-Rust incremental Structure-from-Motion engine.
//!
//! Pipeline (COLMAP's incremental mapper, distilled):
//!   SIFT (scale space, DoG, 128-d) -> mutual-best ratio matching ->
//!   H/F RANSAC two-view verification -> essential decomposition ->
//!   best-pair initialization -> next-best-view registration (PnP-RANSAC) ->
//!   track extension + triangulation -> outlier filtering ->
//!   Schur-complement Levenberg-Marquardt bundle adjustment ->
//!   COLMAP-format reconstruction (consumed directly by 3DGS trainers).

pub mod ba;
pub mod demo;
pub mod geom;
pub mod matching;
pub mod mapper;
pub mod rng;
pub mod scene;
pub mod sift;

use std::path::{Path, PathBuf};

use rayon::prelude::*;

pub use geom::{Cam, Pose};
pub use matching::TwoView;
pub use sift::{SiftFrame, SiftOptions};

/// Pipeline options (single struct so the CLI stays flat).
#[derive(Clone, Debug)]
pub struct Options {
    /// Longest image side for feature extraction (SIFT + cameras are
    /// exported back at the original resolution).
    pub max_dim: u32,
    /// Max SIFT keypoints per image (sorted by DoG response).
    pub max_features: usize,
    /// Lowe ratio threshold.
    pub ratio: f64,
    /// Geometric inlier threshold in pixels.
    pub ransac_thr_px: f64,
    /// Min verified inliers for a usable image pair.
    pub min_pair_inliers: usize,
    /// Min triangulation angle (degrees).
    pub min_tri_angle: f64,
    /// Reprojection outlier threshold (pixels).
    pub reproj_thr: f64,
    /// Global BA cadence (every N registrations; 0 = final only).
    pub ba_every: usize,
    /// Match only temporal neighbors (video-like capture).
    pub sequential: bool,
    /// Focal heuristic: focal = focal_mult * max(w, h) when EXIF is absent.
    pub focal_mult: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_dim: 1600,
            max_features: 2048,
            ratio: 0.75,
            ransac_thr_px: 4.0,
            min_pair_inliers: 30,
            min_tri_angle: 1.5,
            reproj_thr: 4.0,
            ba_every: 4,
            sequential: false,
            focal_mult: 1.2,
        }
    }
}

pub struct SfmResult {
    pub rec: colmap_io::Reconstruction,
    pub registered: usize,
    pub points: usize,
    pub skipped: Vec<String>,
    pub mean_reproj: f64,
    pub ba_rounds: usize,
}

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp")
    )
}

/// Full pipeline over a folder of photos.
pub fn run_sfm_dir(images_dir: &Path, opts: &Options, log: &dyn Fn(&str)) -> Result<SfmResult, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(images_dir)
        .map_err(|e| format!("cannot read {}: {e}", images_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    files.sort();
    run_sfm(&files, opts, log)
}

/// Full pipeline over an explicit, ordered list of image paths.
pub fn run_sfm(paths: &[PathBuf], opts: &Options, log: &dyn Fn(&str)) -> Result<SfmResult, String> {
    use std::time::Instant;
    let _t_all = Instant::now();
    if paths.len() < 2 {
        return Err("need at least 2 images".into());
    }

    // ---- Feature extraction (parallel) ----
    let t0 = Instant::now();
    let sift_opts = SiftOptions { max_dim: opts.max_dim, max_features: opts.max_features, ..Default::default() };
    let frames: Vec<(SiftFrame, u32, u32)> = paths
        .par_iter()
        .map(|p| {
            let img = image::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let orig = (img.width(), img.height());
            let name = p
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            Ok((sift::extract_sift(&img, &name, &sift_opts), orig.0, orig.1))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let total_kps: usize = frames.iter().map(|(f, _, _)| f.pts.len()).sum();
    log(&format!(
        "sift: {} keypoints over {} images in {:.1}s",
        total_kps,
        frames.len(),
        t0.elapsed().as_secs_f32()
    ));
    let frames: Vec<SiftFrame> = frames.into_iter().map(|(f, _, _)| f).collect();
    let orig_dims: Vec<(u32, u32)> = paths
        .par_iter()
        .map(|p| image::image_dimensions(p).unwrap_or((0, 0)))
        .collect();

    // ---- Cameras (one per image; refined later by BA) ----
    let cams: Vec<Cam> = frames
        .iter()
        .map(|f| {
            let mut c = Cam::new(opts.focal_mult * f.width.max(f.height) as f64, f.width as f64, f.height as f64);
            c.f = opts.focal_mult * f.width.max(f.height) as f64;
            c
        })
        .collect();

    // ---- Matching + two-view verification (parallel over pairs) ----
    let t1 = Instant::now();
    let mut job_pairs: Vec<(usize, usize)> = Vec::new();
    if opts.sequential {
        for i in 0..frames.len() {
            for k in 1..=3 {
                if i + k < frames.len() {
                    job_pairs.push((i, i + k));
                }
            }
        }
    } else {
        for i in 0..frames.len() {
            for j in (i + 1)..frames.len() {
                job_pairs.push((i, j));
            }
        }
    }
    let vopts = matching::VerifyOptions {
        ratio: opts.ratio,
        ransac_thr_px: opts.ransac_thr_px,
        min_pair_inliers: opts.min_pair_inliers,
        min_tri_angle: opts.min_tri_angle,
    };
    let verified: Vec<TwoView> = job_pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let pm = matching::match_pair(&frames[i], &frames[j], opts.ratio);
            matching::verify_pair(i, j, &frames[i], &frames[j], &cams[i], &cams[j], &pm, &vopts)
        })
        .collect();
    let tot_inl: usize = verified.iter().map(|tv| tv.a.len()).sum();
    log(&format!(
        "matching: {} pairs verified, {} geometric inliers in {:.1}s",
        verified.len(),
        tot_inl,
        t1.elapsed().as_secs_f32()
    ));
    if verified.is_empty() {
        return Err("no verified image pairs - images may have no overlap".into());
    }

    // ---- Incremental mapping ----
    let t2 = Instant::now();
    let frame_infos: Vec<scene::FrameInfo> = frames
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let (ow, oh) = if orig_dims[idx].0 > 0 {
                orig_dims[idx]
            } else {
                (f.width, f.height)
            };
            scene::FrameInfo {
                name: f.name.clone(),
                width: f.width,
                height: f.height,
                orig_width: ow,
                orig_height: oh,
                pts: f.pts.clone(),
                colors: f.colors.clone(),
            }
        })
        .collect();
    let mopts = mapper::MapperOptions {
        abs_min_inliers: opts.min_pair_inliers.min(20).max(10),
        reproj_thr: opts.reproj_thr,
        min_tri_angle: opts.min_tri_angle,
        ba_every: opts.ba_every,
        ba_iters: 40,
    };
    let (scene, stats) = mapper::map2(&frame_infos, cams, verified, &mopts, &log);
    if scene.registered.len() < 2 || scene.points.is_empty() {
        return Err(format!(
            "reconstruction degenerate: {} images registered, {} points",
            scene.registered.len(),
            scene.points.len()
        ));
    }
    log(&format!(
        "mapping: {} images, {} points in {:.1}s",
        scene.registered.len(),
        scene.points.len(),
        t2.elapsed().as_secs_f32()
    ));

    let rec = scene.to_reconstruction();
    Ok(SfmResult {
        registered: rec.images.len(),
        points: rec.points3d.len(),
        skipped: (0..frames.len())
            .filter(|&f| !scene.is_registered(f))
            .map(|f| frames[f].name.clone())
            .collect(),
        mean_reproj: scene.last_mean_err,
        ba_rounds: stats.ba_rounds,
        rec,
    })
}
