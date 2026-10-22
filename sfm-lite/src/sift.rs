//! SIFT feature extraction — native Rust implementation of the classic
//! scale-space pipeline (Lowe 2004 / OpenCV formulation):
//! Gaussian pyramid -> DoG -> 26-neighbor extrema -> subpixel quadratic fit
//! -> contrast & principal-curvature rejection -> orientation assignment
//! (36-bin, multi-peak) -> 4x4x8 gradient-orientation descriptor with
//! trilinear binning, lowe_clip 0.2.

use image::imageops::FilterType;
use image::DynamicImage;
use nalgebra::Matrix3;


pub const DESC_LEN: usize = 128;
const NSPO: usize = 3; // intervals per octave
const SIGMA0: f64 = 1.6;
const ASSUMED_BLUR: f64 = 0.5;
const EDGE_THR: f64 = 10.0;

#[derive(Clone, Debug)]
pub struct SiftOptions {
    pub max_dim: u32,
    pub max_features: usize,
    /// Contrast threshold on DoG values in [0,1] intensity units.
    pub contrast_thr: f64,
}

impl Default for SiftOptions {
    fn default() -> Self {
        SiftOptions { max_dim: 1600, max_features: 2048, contrast_thr: 0.0067 }
    }
}

pub struct SiftFrame {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub pts: Vec<[f64; 2]>,
    pub desc: Vec<[f32; DESC_LEN]>,
    pub colors: Vec<[u8; 3]>,
}

struct Octave {
    width: usize,
    height: usize,
    /// NSPO+3 blurred levels.
    levels: Vec<Vec<f32>>,
    /// DoG[i] = levels[i+1] - levels[i], len NSPO+2.
    dogs: Vec<Vec<f32>>,
}

fn gauss_blur(src: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    if sigma < 1e-4 {
        return src.to_vec();
    }
    let radius = (sigma * 3.0).ceil().max(1.0) as usize;
    let mut kernel = vec![0f64; 2 * radius + 1];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0f64;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f64 - radius as f64;
        *k = (-x * x / s2).exp();
        sum += *k;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    // Horizontal pass with clamped edges.
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let xx = (x + ki).saturating_sub(radius).min(w - 1);
                acc += kv as f32 * src[row + xx];
            }
            tmp[row + x] = acc;
        }
    }
    // Vertical pass.
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let yy = (y + ki).saturating_sub(radius).min(h - 1);
                acc += kv as f32 * tmp[yy * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

fn downsample2(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut out = vec![0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let x0 = (2 * x).min(w - 1);
            let y0 = (2 * y).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let s = src[y0 * w + x0] + src[y0 * w + x1] + src[y1 * w + x0] + src[y1 * w + x1];
            out[y * nw + x] = 0.25 * s;
        }
    }
    (out, nw, nh)
}

fn build_octave(base: Vec<f32>, w: usize, h: usize, first_sigma: f64) -> Octave {
    let mut levels = Vec::with_capacity(NSPO + 3);
    if first_sigma > 1e-4 {
        levels.push(gauss_blur(&base, w, h, first_sigma));
    } else {
        levels.push(base);
    }
    for l in 1..NSPO + 3 {
        let s_prev = SIGMA0 * 2f64.powi(((l - 1) as f64 / NSPO as f64) as i32);
        let s_cur = SIGMA0 * 2f64.powi((l as f64 / NSPO as f64) as i32);
        let inc = (s_cur * s_cur - s_prev * s_prev).max(1e-4).sqrt();
        let prev = levels.last().unwrap().clone();
        levels.push(gauss_blur(&prev, w, h, inc));
    }
    let mut dogs = Vec::with_capacity(NSPO + 2);
    for i in 0..NSPO + 2 {
        let a = &levels[i];
        let b = &levels[i + 1];
        let d: Vec<f32> = b.iter().zip(a.iter()).map(|(x, y)| x - y).collect();
        dogs.push(d);
    }
    Octave { width: w, height: h, levels, dogs }
}

/// Central-difference gradients of one level, with clamped borders.
fn gradients(img: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0f32; w * h];
    let mut gy = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            gx[i] = img[i + 1] - img[i - 1];
            gy[i] = img[i + w] - img[i - w];
        }
    }
    (gx, gy)
}

#[inline]
fn bilinear(img: &[f32], w: usize, h: usize, x: f64, y: f64) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let cl = |v: isize, m: isize| -> usize { v.clamp(0, m) as usize };
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let (xm, ym) = ((w - 1) as isize, (h - 1) as isize);
    let (x0, y0, x1, y1) = (cl(x0, xm), cl(y0, ym), cl(x1, xm), cl(y1, ym));
    let a = img[y0 * w + x0] * (1.0 - fx) + img[y0 * w + x1] * fx;
    let b = img[y1 * w + x0] * (1.0 - fx) + img[y1 * w + x1] * fx;
    a * (1.0 - fy) + b * fy
}

#[derive(Clone)]
struct RawKp {
    x: f64, // octave coords
    y: f64,
    octave: usize,
    level: usize, // DoG index, 1..=NSPO
    sigma: f64,   // in octave coords
    response: f64,
}

fn detect_extrema(oct: &Octave, octave: usize, thr: f64, out: &mut Vec<RawKp>) {
    let (w, h) = (oct.width, oct.height);
    let edge_lim = ((EDGE_THR + 1.0).powi(2)) / EDGE_THR;
    for level in 1..=NSPO {
        let d = &oct.dogs;
        let dog = &d[level];
        let below = &d[level - 1];
        let above = &d[level + 1];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                let v = dog[i];
                if (v.abs() as f64) < thr {
                    continue;
                }
                let mut is_max = true;
                let mut is_min = true;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        let j = ((y as isize + dy) as usize) * w + (x as isize + dx) as usize;
                        let nb = dog[j];
                        if nb >= v {
                            is_max = false;
                        }
                        if nb <= v {
                            is_min = false;
                        }
                        let nb_b = below[j];
                        if nb_b >= v {
                            is_max = false;
                        }
                        if nb_b <= v {
                            is_min = false;
                        }
                        let nb_a = above[j];
                        if nb_a >= v {
                            is_max = false;
                        }
                        if nb_a <= v {
                            is_min = false;
                        }
                    }
                }
                if !is_max && !is_min {
                    continue;
                }
                // Quadratic subpixel fit.
                let dx = 0.5 * (dog[i + 1] - dog[i - 1]);
                let dy = 0.5 * (dog[i + w] - dog[i - w]);
                let ds = 0.5 * (above[i] - below[i]);
                let dxx = dog[i + 1] + dog[i - 1] - 2.0 * v;
                let dyy = dog[i + w] + dog[i - w] - 2.0 * v;
                let dss = above[i] + below[i] - 2.0 * v;
                let dxy = 0.25 * (dog[i + w + 1] + dog[i - w - 1] - dog[i + w - 1] - dog[i - w + 1]);
                let dxs = 0.25 * (above[i + 1] + below[i - 1] - above[i - 1] - below[i + 1]);
                let dys = 0.25 * (above[i + w] + below[i - w] - above[i - w] - below[i + w]);
                let hmat = Matrix3::new(
                    dxx as f64, dxy as f64, dxs as f64,
                    dxy as f64, dyy as f64, dys as f64,
                    dxs as f64, dys as f64, dss as f64,
                );
                let hinv = match hmat.try_inverse() {
                    Some(m) => m,
                    None => continue,
                };
                let off = hinv * nalgebra::Vector3::new(-(dx as f64), -(dy as f64), -(ds as f64));
                if off.x.abs() > 1.0 || off.y.abs() > 1.0 || off.z.abs() > 1.0 {
                    continue;
                }
                let contrast =
                    v as f64 + 0.5 * (dx as f64 * off.x + dy as f64 * off.y + ds as f64 * off.z);
                if contrast.abs() < thr {
                    continue;
                }
                // Principal curvature rejection (2x2 Hessian of the DoG).
                let tr = (dxx + dyy) as f64;
                let det = (dxx * dyy - dxy * dxy) as f64;
                if det <= 0.0 || tr * tr / det > edge_lim {
                    continue;
                }
                out.push(RawKp {
                    x: x as f64 + off.x,
                    y: y as f64 + off.y,
                    octave,
                    level,
                    sigma: SIGMA0 * 2f64.powi((level as f64 / NSPO as f64) as i32),
                    response: contrast.abs(),
                });
            }
        }
    }
}

/// Orientation histogram (36 bins) with Gaussian weighting; returns
/// interpolated angles (radians) of the main peak + all peaks >= 0.8 * max.
fn orientations(grad: (&[f32], &[f32]), w: usize, h: usize, x: f64, y: f64, sigma: f64) -> Vec<f64> {
    let mut hist = [0f64; 36];
    let radius = ((3.0 * sigma).ceil().max(1.0)) as usize;
    let s2 = 2.0 * sigma * sigma;
    let x0 = x.round() as isize;
    let y0 = y.round() as isize;
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let sx = x0 + dx;
            let sy = y0 + dy;
            if sx < 1 || sy < 1 || sx >= w as isize - 1 || sy >= h as isize - 1 {
                continue;
            }
            let i = sy as usize * w + sx as usize;
            let mag = ((grad.0[i] * grad.0[i] + grad.1[i] * grad.1[i]) as f64).sqrt();
            let ang = (grad.1[i] as f64).atan2(grad.0[i] as f64);
            let wgt = (-(dx as f64).powi(2) - (dy as f64).powi(2)) / s2;
            let wgt = wgt.exp();
            let mut bin = ((ang + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 36.0)
                .floor() as usize;
            bin %= 36;
            hist[bin] += wgt * mag;
        }
    }
    // Smooth [1 1 1]/3, twice.
    for _ in 0..2 {
        let mut sm = [0f64; 36];
        for b in 0..36 {
            sm[b] = (hist[(b + 35) % 36] + 2.0 * hist[b] + hist[(b + 1) % 36]) / 4.0;
        }
        hist = sm;
    }
    let maxv = hist.iter().cloned().fold(0f64, f64::max);
    if maxv < 1e-12 {
        return vec![0.0];
    }
    let thresh = 0.8 * maxv;
    let mut peaks = Vec::new();
    for b in 0..36 {
        let prev = hist[(b + 35) % 36];
        let next = hist[(b + 1) % 36];
        if hist[b] > hist[(b + 35) % 36] && hist[b] > hist[(b + 1) % 36] && hist[b] >= thresh {
            // Parabolic interpolation over the peak.
            let denom = hist[b] * 2.0 - prev - next;
            let d = if denom.abs() > 1e-12 {
                0.5 * (prev - next) / denom
            } else {
                0.0
            };
            let theta = (b as f64 + d) / 36.0 * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
            let theta = theta.rem_euclid(2.0 * std::f64::consts::PI);
            peaks.push(theta);
            if peaks.len() >= 4 {
                break;
            }
        }
    }
    if peaks.is_empty() {
        peaks.push(0.0);
    }
    peaks
}

/// OpenCV-style descriptor: d=4 x d=4 cells, n=8 orientation bins,
/// cell size = 3 sigma, Gaussian window sigma = d/2 (in cell units),
/// trilinear binning, L2 -> clip 0.2 -> L2.
fn descriptor(
    grad: (&[f32], &[f32]),
    w: usize,
    h: usize,
    x: f64,
    y: f64,
    sigma: f64,
    theta: f64,
) -> Option<[f32; DESC_LEN]> {
    const D: usize = 4;
    const N: usize = 8;
    let cell = 3.0 * sigma;
    let radius = (1.5f64.sqrt() * cell * (D as f64 + 1.0) * 0.5).ceil() as usize;
    let (cos_t, sin_t) = (theta.cos(), theta.sin());
    let exp_scale = -1.0 / ((D * D) as f64 * 0.5);
    let mut hist = [[0f64; N]; D * D];
    let half = (D as f64) * 0.5;
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let sx = x + dx as f64;
            let sy = y + dy as f64;
            if sx < 1.0 || sy < 1.0 || sx >= w as f64 - 2.0 || sy >= h as f64 - 2.0 {
                continue;
            }
            let gx = bilinear(grad.0, w, h, sx, sy) as f64;
            let gy = bilinear(grad.1, w, h, sx, sy) as f64;
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < 1e-12 {
                continue;
            }
            // Rotate into keypoint frame.
            let rx = cos_t * dx as f64 + sin_t * dy as f64;
            let ry = -sin_t * dx as f64 + cos_t * dy as f64;
            let bx = rx / cell + half - 0.5;
            let by = ry / cell + half - 0.5;
            if bx < -1.0 || by < -1.0 || bx > D as f64 || by > D as f64 {
                continue;
            }
            let ang = gy.atan2(gx) - theta;
            let ob = ang.rem_euclid(2.0 * std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * N as f64;
            let wgt = ((bx - half + 0.5) * (bx - half + 0.5) + (by - half + 0.5) * (by - half + 0.5))
                * exp_scale;
            let wgt = wgt.exp() * mag;
            let (bx0, by0, ob0) = (bx.floor() as isize, by.floor() as isize, ob.floor() as isize);
            let (fx, fy, fo) = (bx - bx0 as f64, by - by0 as f64, ob - ob0 as f64);
            for iy in 0..2 {
                for ix in 0..2 {
                    let bx_i = bx0 + ix;
                    let by_i = by0 + iy;
                    if bx_i < 0 || by_i < 0 || bx_i >= D as isize || by_i >= D as isize {
                        continue;
                    }
                    let wy = if iy == 0 { 1.0 - fy } else { fy };
                    let wx = if ix == 0 { 1.0 - fx } else { fx };
                    for io in 0..2 {
                        let ob_i = (ob0 + io).rem_euclid(N as isize) as usize;
                        let wo = if io == 0 { 1.0 - fo } else { fo };
                        hist[(by_i as usize) * D + bx_i as usize][ob_i] +=
                            wgt * wx * wy * wo;
                    }
                }
            }
        }
    }
    let mut d = [0f32; DESC_LEN];
    for (k, cell_hist) in hist.iter().enumerate() {
        for (b, v) in cell_hist.iter().enumerate() {
            d[k * N + b] = *v as f32;
        }
    }
    let norm: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-6 {
        return None;
    }
    for v in d.iter_mut() {
        *v /= norm;
    }
    for v in d.iter_mut() {
        if *v > 0.2 {
            *v = 0.2;
        }
    }
    let norm2: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm2 < 1e-6 {
        return None;
    }
    for v in d.iter_mut() {
        *v /= norm2;
    }
    Some(d)
}

/// Full SIFT extraction on one image. Keypoint coordinates are in the
/// (possibly downscaled) input image frame; descriptor vectors are
/// L2-normalized f32.
pub fn extract_sift(img: &DynamicImage, name: &str, opts: &SiftOptions) -> SiftFrame {
    let (w0, h0) = (img.width(), img.height());
    let m = w0.max(h0);
    let img: DynamicImage = if m > opts.max_dim {
        let s = opts.max_dim as f64 / m as f64;
        let nw = ((w0 as f64 * s).round() as u32).max(1);
        let nh = ((h0 as f64 * s).round() as u32).max(1);
        img.resize_exact(nw, nh, FilterType::Triangle)
    } else {
        img.clone()
    };
    let rgb = img.to_rgb8();
    let gray = img.to_luma8();
    let w = gray.width() as usize;
    let h = gray.height() as usize;
    let base: Vec<f32> = gray.pixels().map(|p| p.0[0] as f32 / 255.0).collect();

    let mut n_oct = 1usize;
    while (w.max(h) >> n_oct) >= 40 && n_oct < 8 {
        n_oct += 1;
    }

    // Build all octaves, detecting extrema as we go.
    let mut raw: Vec<RawKp> = Vec::new();
    let mut octaves: Vec<Octave> = Vec::with_capacity(n_oct);
    let mut oct_base = base;
    let mut ow = w;
    let mut oh = h;
    for o in 0..n_oct {
        let first_sigma = if o == 0 {
            (SIGMA0 * SIGMA0 - ASSUMED_BLUR * ASSUMED_BLUR).max(1e-4).sqrt()
        } else {
            0.0
        };
        let oct = build_octave(oct_base.clone(), ow, oh, first_sigma);
        detect_extrema(&oct, o, opts.contrast_thr, &mut raw);
        let last = oct.levels[NSPO].clone();
        octaves.push(oct);
        let (nb, nw, nh) = downsample2(&last, ow, oh);
        oct_base = nb;
        ow = nw;
        oh = nh;
    }

    struct OutKp {
        resp: f64,
        x: f64,
        y: f64,
        d: [f32; DESC_LEN],
    }
    let mut outs: Vec<OutKp> = Vec::new();

    for (o, oct) in octaves.iter().enumerate() {
        let kps_o: Vec<RawKp> = raw.iter().filter(|k| k.octave == o).cloned().collect();
        if kps_o.is_empty() {
            continue;
        }
        let scale = 2f64.powi(o as i32);
        let mut grads: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None; NSPO + 2];
        for kp in &kps_o {
            let li = kp.level;
            if grads[li].is_none() {
                grads[li] = Some(gradients(&oct.levels[li], oct.width, oct.height));
            }
            let g = grads[li].as_ref().unwrap();
            let x0 = kp.x * scale;
            let y0 = kp.y * scale;
            // Keep an 8-px margin so descriptors have full support.
            if x0 < 8.0 || y0 < 8.0 || x0 >= w as f64 - 8.0 || y0 >= h as f64 - 8.0 {
                continue;
            }
            let oris = orientations(
                (&g.0[..], &g.1[..]),
                oct.width,
                oct.height,
                kp.x,
                kp.y,
                1.5 * kp.sigma,
            );
            for theta in oris {
                if let Some(d) = descriptor(
                    (&g.0[..], &g.1[..]),
                    oct.width,
                    oct.height,
                    kp.x,
                    kp.y,
                    kp.sigma,
                    theta,
                ) {
                    outs.push(OutKp { resp: kp.response, x: x0, y: y0, d });
                }
            }
        }
    }

    // Global response cap.
    outs.sort_by(|a, b| b.resp.partial_cmp(&a.resp).unwrap());
    outs.truncate(opts.max_features);

    let mut pts = Vec::with_capacity(outs.len());
    let mut desc = Vec::with_capacity(outs.len());
    let mut colors = Vec::with_capacity(outs.len());
    for kp in outs {
        let x = kp.x.clamp(0.0, (w - 1) as f64);
        let y = kp.y.clamp(0.0, (h - 1) as f64);
        let (px, py) = (x.round() as u32, y.round() as u32);
        let c = rgb.get_pixel(px.min(w as u32 - 1), py.min(h as u32 - 1)).0;
        pts.push([x, y]);
        desc.push(kp.d);
        colors.push([c[0], c[1], c[2]]);
    }

    SiftFrame {
        name: name.to_string(),
        width: w as u32,
        height: h as u32,
        pts,
        desc,
        colors,
    }
}
