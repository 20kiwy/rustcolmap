//! gs-export — turn a folder of photos into a COLMAP-style dataset that
//! Gaussian Splatting trainers consume directly:
//!
//! ```text
//! <out>/
//! ├── images/          (copied from --images)
//! ├── points3d.ply     (optional, --ply)
//! └── sparse/0/
//!     ├── cameras.bin
//!     ├── images.bin
//!     └── points3D.bin  (+ .txt with --txt)
//! ```
//!
//! Usage examples:
//!   gs-export --images ./photos --out ./dataset
//!   gs-export --images ./photos --out ./dataset --sequential --txt --ply
//!   gs-export --demo ./demo_out          # synthetic self-test
//!   # then: python train.py -s ./dataset            (graphdeco 3DGS)
//!   # or:   ns-train splatfacto --data ./dataset    (nerfstudio/gsplat)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use colmap_io::{write_bin_dir, write_txt_dir};
use sfm_lite::{Options, run_sfm, run_sfm_dir};

const HELP: &str = "\
gs-export — Rust-native SfM (SIFT + incremental mapping + bundle adjustment)
            -> COLMAP sparse format for Gaussian Splatting

USAGE:
  gs-export --images <dir> --out <dir> [options]
  gs-export --demo <dir> [n_images]

OPTIONS:
  --images <dir>       input folder with photos (required)
  --out <dir>          output dataset folder (required)
  --max-features <n>   SIFT keypoints per image        [default: 2048]
  --max-dim <px>       downscale longest side for SfM  [default: 1600]
  --ratio <f>          Lowe ratio threshold            [default: 0.8]
  --min-angle <deg>    min triangulation angle         [default: 1.5]
  --ba-every <n>       global BA every N registrations [default: 4]
  --sequential         match temporal neighbors only (video-like sets)
  --txt                also write COLMAP .txt files
  --ply                also write points3d.ply
  --demo <dir> [n]     run synthetic end-to-end self-test (default 10 imgs)
  -h, --help           this help
";

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp")
    )
}

/// Natural sort so img2 < img10.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let mut ai = a.as_bytes();
    let mut bi = b.as_bytes();
    loop {
        if ai.is_empty() || bi.is_empty() {
            return ai.len().cmp(&bi.len());
        }
        let da = ai[0].is_ascii_digit();
        let db = bi[0].is_ascii_digit();
        if da && db {
            let na = ai.iter().take_while(|c| c.is_ascii_digit()).count();
            let nb = bi.iter().take_while(|c| c.is_ascii_digit()).count();
            let (sa, sb) = (&ai[..na], &bi[..nb]);
            let ta = sa.iter().position(|c| *c != b'0').unwrap_or(sa.len());
            let tb = sb.iter().position(|c| *c != b'0').unwrap_or(sb.len());
            let (va, vb) = (&sa[ta..], &sb[tb..]);
            let ord = va.len().cmp(&vb.len()).then_with(|| va.cmp(vb));
            if ord != Ordering::Equal {
                return ord;
            }
            ai = &ai[na..];
            bi = &bi[nb..];
        } else if da != db {
            return da.cmp(&db);
        } else {
            let na = ai.iter().take_while(|c| !c.is_ascii_digit()).count();
            let nb = bi.iter().take_while(|c| !c.is_ascii_digit()).count();
            let sa = std::str::from_utf8(&ai[..na]).unwrap_or("");
            let sb = std::str::from_utf8(&bi[..nb]).unwrap_or("");
            let ord = sa.cmp(sb);
            if ord != Ordering::Equal {
                return ord;
            }
            ai = &ai[na..];
            bi = &bi[nb..];
        }
    }
}

fn parse_num<T: std::str::FromStr, I: Iterator<Item = String>>(
    args: &mut I,
    what: &str,
) -> T {
    match args.next().and_then(|v| v.parse().ok()) {
        Some(v) => v,
        None => {
            eprintln!("--{what} needs a number");
            exit(2);
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut images: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut opts = Options::default();
    let mut also_txt = false;
    let mut also_ply = false;
    let mut demo: Option<(PathBuf, usize)> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--images" => images = Some(PathBuf::from(args.next().unwrap_or_default())),
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_default())),
            "--max-features" => opts.max_features = parse_num(&mut args, "max-features"),
            "--max-dim" => opts.max_dim = parse_num(&mut args, "max-dim"),
            "--ratio" => opts.ratio = parse_num(&mut args, "ratio"),
            "--min-angle" => opts.min_tri_angle = parse_num(&mut args, "min-angle"),
            "--ba-every" => opts.ba_every = parse_num(&mut args, "ba-every"),
            "--sequential" => opts.sequential = true,
            "--txt" => also_txt = true,
            "--ply" => also_ply = true,
            "--demo" => {
                let dir = PathBuf::from(args.next().unwrap_or_else(|| "demo_out".into()));
                let n = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(10usize);
                demo = Some((dir, n));
            }
            "-h" | "--help" => {
                println!("{HELP}");
                return;
            }
            other => {
                eprintln!("unknown argument: {other}\n{HELP}");
                exit(2);
            }
        }
    }

    let log = |s: &str| eprintln!("[sfm] {s}");

    // ---- Synthetic self-test mode ----
    if let Some((dir, n)) = demo {
        eprintln!("gs-export demo: synthetic corner-room scene, {n} cameras");
        match sfm_lite::demo::run_demo(&dir, n, &log) {
            Ok((registered, points, pos_err, rot_err)) => {
                let ok = registered >= (n - 1).max(2) && rot_err < 3.0 && pos_err < 0.3;
                println!(
                    "demo {}: registered={registered} points={points} median-center-err={pos_err:.4} median-rot-err={rot_err:.2}deg -> {}",
                    if ok { "PASS" } else { "CHECK" },
                    dir.display()
                );
                return;
            }
            Err(e) => {
                eprintln!("demo failed: {e}");
                exit(1);
            }
        }
    }

    // ---- Real photos mode ----
    let images = match images {
        Some(p) if p.is_dir() => p,
        _ => {
            eprintln!("--images must point to an existing folder\n{HELP}");
            exit(2);
        }
    };
    let out = match out {
        Some(p) => p,
        None => {
            eprintln!("--out is required\n{HELP}");
            exit(2);
        }
    };

    let read: std::io::Result<Vec<PathBuf>> = fs::read_dir(&images).map(|rd| {
        rd.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_image(p))
            .collect()
    });
    let mut files: Vec<PathBuf> = match read {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read {}: {e}", images.display());
            exit(1);
        }
    };
    files.sort_by(|a, b| {
        let na = a.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let nb = b.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        natural_cmp(&na, &nb)
    });
    if files.len() < 2 {
        eprintln!("need at least 2 images in {}", images.display());
        exit(1);
    }
    eprintln!(
        "sfm: {} images, features={}, max_dim={}, matching={} ...",
        files.len(),
        opts.max_features,
        opts.max_dim,
        if opts.sequential { "sequential" } else { "exhaustive" }
    );

    let t0 = Instant::now();
    let res = match run_sfm(&files, &opts, &log) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SfM failed: {e}");
            exit(1);
        }
    };
    eprintln!(
        "done: registered {} images, {} points, mean reproj {:.2}px, {} BA rounds in {:.1}s (skipped: {})",
        res.registered,
        res.points,
        res.mean_reproj,
        res.ba_rounds,
        t0.elapsed().as_secs_f32(),
        if res.skipped.is_empty() {
            "none".to_string()
        } else {
            res.skipped.join(", ")
        }
    );

    let sparse = out.join("sparse").join("0");
    if let Err(e) = write_bin_dir(&res.rec, &sparse) {
        eprintln!("failed writing sparse bin files: {e}");
        exit(1);
    }
    if also_txt {
        let _ = write_txt_dir(&res.rec, &sparse);
    }
    if also_ply {
        let cloud: Vec<(f64, f64, f64, [u8; 3])> = res
            .rec
            .points3d
            .iter()
            .map(|p| (p.xyz[0], p.xyz[1], p.xyz[2], p.rgb))
            .collect();
        let _ = sfm_lite::demo::write_ply(&out.join("points3d.ply"), &cloud);
    }

    let img_out = out.join("images");
    if let Err(e) = fs::create_dir_all(&img_out) {
        eprintln!("failed creating {}: {e}", img_out.display());
        exit(1);
    }
    for p in &files {
        if let Some(name) = p.file_name() {
            let _ = fs::copy(p, img_out.join(name));
        }
    }

    println!(
        "done: {} ({} images, {} points) -> feed this folder to your 3DGS trainer",
        out.display(),
        res.registered,
        res.points
    );
}
