# RustCOLMAP

**A native-Rust incremental Structure-from-Motion + dense Multi-View Stereo pipeline with COLMAP-compatible output, purpose-built to feed Gaussian Splatting trainers.**

```
photos ──▶ SIFT ──▶ robust matching ──▶ two-view verification ──▶ incremental mapping
       ──▶ bundle adjustment ──▶ COLMAP sparse/0 ──▶ patch-match stereo ──▶ dense fusion
       ──▶ dense.ply + sparse/0 ──▶ 3DGS / nerfstudio / gsplat
```

```text
<dataset>/
├── images/            (your photos, copied)
├── dense/             (with --mvs)
│   ├── dense.ply      fused DENSE colored point cloud (binary PLY)
│   ├── dense_web.json decimated copy for web viewers
│   └── depth_vis/     per-view depth-map PNGs (--save-depth)
└── sparse/0/
    ├── cameras.bin    (+ .txt)
    ├── images.bin     (+ .txt)
    └── points3D.bin   (+ .txt)
```

## What this is (and honestly isn't)

COLMAP is a ~10-year, 100k+ line C++/CUDA research system. RustCOLMAP is a
compact, dependency-light reimplementation of **COLMAP's incremental SfM core
+ dense MVS stages** — camera poses, the sparse point cloud, and the fused
dense colored point cloud. It is *not* a line-for-line port and does not
include vocab-tree retrieval or GPU acceleration (see [roadmap](#roadmap)).

What is implemented, natively in Rust (`nalgebra` + `image` + `rayon` only):

| Stage | Details |
|---|---|
| **SIFT** | Gaussian scale space (DoG), 26-neighbor extrema, subpixel quadratic fit, contrast + principal-curvature rejection, 36-bin orientation (multi-peak), 4×4×8 descriptors with trilinear binning, Lowe clip 0.2 |
| **Matching** | Mutual-best + **symmetric Lowe ratio** test, rayon-parallel over pairs |
| **Two-view verification** | Homography RANSAC (degeneracy test), 8-point F-RANSAC, **planar-degeneracy recovery** (F re-estimated from H-outliers), E = KᵀFK on the essential manifold, 4-pose cheirality scoring |
| **Incremental mapping** | Best-pair init (inliers + triangulation rate), next-best-view selection, PnP-DLT RANSAC registration, track extension, angle-gated triangulation, outlier filtering |
| **Bundle adjustment** | Schur-complement Levenberg–Marquardt, SIMPLE_RADIAL cameras (f, cx, cy, k refined), quaternion+translation parametrization, Huber loss, Jacobi column scaling, per-registration BA + **re-triangulation** |
| **Export** | Exact COLMAP binary + text formats (validated byte-level), full-resolution camera rescaling, PLY point clouds |
| **Dense MVS** | Slanted-plane patch-match stereo per view: 7×7 census (48-bit) cost over a 5×5 window averaged across K support views, plane-sweep init in inverse depth, red-black propagation + decaying random refinement (deterministic per-pixel RNG), **texture gate** (flat census descriptors can't fake a match), geometric-consistency fusion across all co-visible views, voxel-hash dedup → binary PLY + depth-map PNGs. CPU only (rayon). |

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

**Dense MVS on the real data** (2 CPU cores, no GPU):

```
# add dense output to an existing reconstruction (skips SfM)
gs-export --sparse dataset/sparse/0 --images dataset/images \
          --out dataset --mvs --mvs-max-dim 700 --save-depth

Sceaux Castle (8 photos, 700 px):   51,239 fused dense points, ~2 min
Big Buck Bunny (20 cartoon frames, 560 px, loose gates):
                                     60,904 fused dense points
```

Depth maps show coherent structure (rooflines, chimneys, foliage patches);
flat sky and overexposed surfaces are texture-gated out by design. Cartoon
content (flat shading, foliage, big inter-frame baselines) is the worst case
for photometric stereo — real textured photos fuse much more completely.

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
- **Dense MVS is CPU-resolution-bounded.** Quality matches the COLMAP class
  at reduced working resolution (`--mvs-max-dim`); on 2 cores a full-HD
  dense run takes minutes. No slanted-plane normal fusion into meshes yet —
  output is the dense POINT cloud, meshing is a separate stage.
- **Scale** is monocular (unit-norm essential translation), like COLMAP's
  uninitialized mode.

## Quick start

```bash
cargo build --release

# 1. Synthetic end-to-end self-test (no inputs needed)
./target/release/gs-export --demo ./demo_out           # 10 cameras
./target/release/gs-export --demo ./demo_out 16        # 16 cameras

# 2. Your photos -> 3DGS dataset + DENSE point cloud
./target/release/gs-export --images ./photos --out ./dataset --txt --ply --mvs --save-depth

# or: add dense output to an existing sparse/0 (skips SfM)
./target/release/gs-export --sparse ./dataset/sparse/0 --images ./dataset/images \
                           --out ./dataset --mvs --mvs-max-dim 700

# 3. Feed the dataset to your trainer
python train.py -s ./dataset                 # graphdeco 3DGS
ns-train splatfacto --data ./dataset         # nerfstudio / gsplat
```

Options: `--max-features` (SIFT cap, default 2048), `--max-dim` (working
resolution, default 1600), `--ratio` (Lowe threshold, default 0.75),
`--min-angle` (min triangulation angle), `--ba-every`, `--sequential`
(temporal matching for video-like captures), `--txt`, `--ply`.

MVS options: `--mvs`, `--mvs-max-dim` (stereo resolution, default 900),
`--mvs-sources` (support views, default 3), `--mvs-cost-thr` (census gate,
default 0.38), `--mvs-depth-diff` (fusion agreement, default 0.03),
`--mvs-consistent` (min confirming views, default 2), `--save-depth`.

Point clouds can be inspected with any PLY viewer (e.g. MeshLab) via
`points3d.ply` (sparse) and `dense/dense.ply` (dense).

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
| `mvs` | dense stage: `stereo` (census patch-match depth maps), `fuse` (consistency fusion + PLY) |
| `gs-export` | CLI: photos → 3DGS-ready dataset + dense.ply (or `--demo` self-test) |

## Roadmap

- [ ] GPU SIFT + matching + patch-match (wgpu compute shaders as the CUDA replacement)
- [ ] 5-point essential solver (replaces 8-point F→E path)
- [ ] Vocab-tree / retrieval-based matching for large unordered collections
- [ ] Local BA windows (currently global BA per registration)
- [x] Dense MVS: patch-match stereo + consistency fusion (`mvs` crate, CPU)
- [ ] Dense meshing: normal estimation + Poisson-like surface reconstruction
- [ ] Glomap-style global SfM alternative

## License

MIT. Algorithmically inspired by COLMAP (BSD-3) — no source code copied.
SIFT as described in Lowe (2004); patent expired 2020.
