# RustCOLMAP

**A native-Rust incremental Structure-from-Motion pipeline with COLMAP-compatible output, purpose-built to feed Gaussian Splatting trainers.**

```
photos ──▶ SIFT ──▶ robust matching ──▶ two-view verification ──▶ incremental mapping
       ──▶ bundle adjustment ──▶ COLMAP sparse/0 ──▶ 3DGS / nerfstudio / gsplat
```

```text
<dataset>/
├── images/            (your photos, copied)
└── sparse/0/
    ├── cameras.bin    (+ .txt)
    ├── images.bin     (+ .txt)
    └── points3D.bin   (+ .txt)
```

## What this is (and honestly isn't)

COLMAP is a ~10-year, 100k+ line C++/CUDA research system. RustCOLMAP is a
compact, dependency-light reimplementation of **COLMAP's incremental SfM
core** — the part that produces camera poses and the sparse point cloud that
Gaussian Splatting training consumes. It is *not* a line-for-line port and
does not include dense MVS, vocab-tree retrieval, or GPU acceleration (see
[roadmap](#roadmap)).

What is implemented, natively in Rust (`nalgebra` + `image` + `rayon` only):

| Stage | Details |
|---|---|
| **SIFT** | Gaussian scale space (DoG), 26-neighbor extrema, subpixel quadratic fit, contrast + principal-curvature rejection, 36-bin orientation (multi-peak), 4×4×8 descriptors with trilinear binning, Lowe clip 0.2 |
| **Matching** | Mutual-best + **symmetric Lowe ratio** test, rayon-parallel over pairs |
| **Two-view verification** | Homography RANSAC (degeneracy test), 8-point F-RANSAC, **planar-degeneracy recovery** (F re-estimated from H-outliers), E = KᵀFK on the essential manifold, 4-pose cheirality scoring |
| **Incremental mapping** | Best-pair init (inliers + triangulation rate), next-best-view selection, PnP-DLT RANSAC registration, track extension, angle-gated triangulation, outlier filtering |
| **Bundle adjustment** | Schur-complement Levenberg–Marquardt, SIMPLE_RADIAL cameras (f, cx, cy, k refined), quaternion+translation parametrization, Huber loss, Jacobi column scaling, per-registration BA + **re-triangulation** |
| **Export** | Exact COLMAP binary + text formats (validated byte-level), full-resolution camera rescaling, PLY point clouds |

## Results (synthetic ground-truth self-test)

`gs-export --demo` renders a textured corner-room scene from known poses and
runs the *entire* pipeline on the synthetic photos (no ground-truth leakage):

```
demo PASS: registered=10/10  points=3272
  mean reprojection error      0.80 px
  median rotation error        0.81°
  median camera-center error   0.35 scene units
  focal self-calibration       549.4 px recovered vs 550.0 ground truth
```

## Results (real-world tests)

**Real video** (Big Buck Bunny 1080p H.264, 10 s clip → 20 frames @ 2 fps,
960 px, `ffmpeg`-extracted, no EXIF):

```
gs-export --images frames/ --out dataset --txt --ply
→ registered 20/20 frames, 6657 sparse points
  mean reprojection 1.32 px, full COLMAP sparse/0 + PLY output
  per-frame focal self-calibration active (no priors available)
```

**Real photos** (Lund, 8 views around a university building, 1024×768,
EXIF 35 mm-equivalent focal read automatically): reconstructs the sequence;
forward-walking init pairs (all-parallax < 1°, i.e. pure translation along
the optical axis) remain the weak spot — see Known limitations.

EXIF: the engine reads `FocalLengthIn35mmFilm` (kamadak-exif) and seeds each
camera's focal prior with `f_px = f35 · max_side / 36`. Without it, video
frames fall back to the 1.2 × max-side heuristic plus a 6-hypothesis focal
fan tried at every PnP registration (COLMAP-style self-calibration).

## Known limitations (real-data honesty)

- **Forward-motion initialization.** When *every* candidate pair moves
  almost purely along the optical axis (walking toward a wall), the two-view
  geometry is ill-conditioned and COLMAP-grade init scoring (which tries and
  scores many candidate pairs by angle spread + multi-view support) is needed.
  Lateral / orbital capture works well. On the roadmap: trifocal
  initialization over the top-K pairs.
- **No dense MVS** — output is the sparse cloud 3DGS needs (poses + sparse
  points), not a dense mesh.
- **Scale** is monocular (unit-norm essential translation), like COLMAP's
  uninitialized mode.

## Quick start

```bash
cargo build --release

# 1. Synthetic end-to-end self-test (no inputs needed)
./target/release/gs-export --demo ./demo_out           # 10 cameras
./target/release/gs-export --demo ./demo_out 16        # 16 cameras

# 2. Your photos -> 3DGS dataset
./target/release/gs-export --images ./photos --out ./dataset --txt --ply

# 3. Feed the dataset to your trainer
python train.py -s ./dataset                 # graphdeco 3DGS
ns-train splatfacto --data ./dataset         # nerfstudio / gsplat
```

Options: `--max-features` (SIFT cap, default 2048), `--max-dim` (working
resolution, default 1600), `--ratio` (Lowe threshold, default 0.75),
`--min-angle` (min triangulation angle), `--ba-every`, `--sequential`
(temporal matching for video-like captures), `--txt`, `--ply`.

Point clouds can be inspected with any PLY viewer (e.g. MeshLab) via
`points3d.ply`.

## Design notes

- **Correctness first.** The BA Jacobians are analytically derived and were
  verified against finite differences to 1e-8; the LM solver uses Ceres-style
  Jacobi column scaling so near-plane observations can't poison the normal
  equations. The two-view verifier scores *all* candidate models (global F,
  refined F, planar-recovered F) by cheirality + reprojection instead of
  trusting RANSAC support counts, which defeats the planar-degeneracy trap
  where a wrong F has *more* inliers than the true one.
- **COLMAP compatibility.** `colmap-io` reads and writes the exact binary
  layout of `reconstruction_io.cc` (little-endian, COLMAP ≥ 3.x) and the text
  formats. Poses are world→camera with (qw, qx, qy, qz) — same as COLMAP.
  SIMPLE_RADIAL is the export camera model (COLMAP's default); 3DGS,
  nerfstudio/gsplat and OpenSplat all consume it.
- **Monocular scale.** Like COLMAP, reconstruction scale is set by the
  initial pair (essential-matrix translation is unit norm). If you need
  metric scale, apply your known baseline afterwards.

## Workspace

| Crate | Purpose |
|---|---|
| `colmap-io` | std-only COLMAP sparse reader/writer (.bin + .txt) |
| `sfm-lite` | the SfM engine: `sift`, `matching`, `geom`, `ba`, `scene`, `mapper`, self-test `demo` |
| `gs-export` | CLI: photos → 3DGS-ready dataset (or `--demo` self-test) |

## Roadmap

- [ ] GPU SIFT + matching (wgpu compute shaders as the CUDA replacement)
- [ ] 5-point essential solver (replaces 8-point F→E path)
- [ ] Vocab-tree / retrieval-based matching for large unordered collections
- [ ] Local BA windows (currently global BA per registration)
- [ ] Dense MVS: patch-match stereo + fusion (the big one)
- [ ] Glomap-style global SfM alternative

## License

MIT. Algorithmically inspired by COLMAP (BSD-3) — no source code copied.
SIFT as described in Lowe (2004); patent expired 2020.
