//! colmap-io — std-only, native Rust reader/writer for COLMAP sparse
//! reconstruction files (`cameras`, `images`, `points3D`), in both `.bin`
//! and `.txt` flavors.
//!
//! The binary layout follows COLMAP's `src/colmap/scene/reconstruction_io.cc`
//! (little-endian, COLMAP >= 3.x). Files written by this crate are consumed
//! unchanged by Gaussian-Splatting pipelines that parse COLMAP output
//! (3D Gaussian Splatting, nerfstudio/gsplat, OpenSplat, ...).

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// Camera models (ids and parameter counts must match COLMAP exactly)
// ---------------------------------------------------------------------------

pub const SIMPLE_PINHOLE: i32 = 0;
pub const PINHOLE: i32 = 1;
pub const SIMPLE_RADIAL: i32 = 2;
pub const RADIAL: i32 = 3;
pub const OPENCV: i32 = 4;
pub const OPENCV_FISHEYE: i32 = 5;
pub const FULL_OPENCV: i32 = 6;
pub const FOV: i32 = 7;
pub const SIMPLE_RADIAL_FISHEYE: i32 = 8;
pub const RADIAL_FISHEYE: i32 = 9;
pub const THIN_PRISM_FISHEYE: i32 = 10;

pub struct ModelInfo {
    pub id: i32,
    pub name: &'static str,
    pub num_params: usize,
}

pub const MODELS: [ModelInfo; 11] = [
    ModelInfo { id: SIMPLE_PINHOLE, name: "SIMPLE_PINHOLE", num_params: 3 },
    ModelInfo { id: PINHOLE, name: "PINHOLE", num_params: 4 },
    ModelInfo { id: SIMPLE_RADIAL, name: "SIMPLE_RADIAL", num_params: 4 },
    ModelInfo { id: RADIAL, name: "RADIAL", num_params: 5 },
    ModelInfo { id: OPENCV, name: "OPENCV", num_params: 8 },
    ModelInfo { id: OPENCV_FISHEYE, name: "OPENCV_FISHEYE", num_params: 8 },
    ModelInfo { id: FULL_OPENCV, name: "FULL_OPENCV", num_params: 12 },
    ModelInfo { id: FOV, name: "FOV", num_params: 5 },
    ModelInfo { id: SIMPLE_RADIAL_FISHEYE, name: "SIMPLE_RADIAL_FISHEYE", num_params: 4 },
    ModelInfo { id: RADIAL_FISHEYE, name: "RADIAL_FISHEYE", num_params: 5 },
    ModelInfo { id: THIN_PRISM_FISHEYE, name: "THIN_PRISM_FISHEYE", num_params: 12 },
];

pub fn model_by_id(id: i32) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn model_by_name(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name.eq_ignore_ascii_case(name))
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Camera {
    pub camera_id: i32,
    pub model_id: i32,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub point3d_id: i64, // -1 = not triangulated
}

/// Pose is world->camera: x_cam = R(qvec) * x_world + tvec.
/// Quaternion stored as (qw, qx, qy, qz), matching COLMAP's convention.
#[derive(Debug, Clone)]
pub struct Image {
    pub image_id: i32,
    pub qvec: [f64; 4],
    pub tvec: [f64; 3],
    pub camera_id: i32,
    pub name: String,
    pub points2d: Vec<Point2D>,
}

#[derive(Debug, Clone, Copy)]
pub struct TrackElement {
    pub image_id: i32,
    pub point2d_idx: u32,
}

#[derive(Debug, Clone)]
pub struct Point3D {
    pub point3d_id: i64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,
    pub track: Vec<TrackElement>,
}

#[derive(Debug, Clone, Default)]
pub struct Reconstruction {
    pub cameras: Vec<Camera>,
    pub images: Vec<Image>,
    pub points3d: Vec<Point3D>,
}

impl Reconstruction {
    pub fn camera(&self, id: i32) -> Option<&Camera> {
        self.cameras.iter().find(|c| c.camera_id == id)
    }
    pub fn image_by_name(&self, name: &str) -> Option<&Image> {
        self.images.iter().find(|i| i.name == name)
    }
    pub fn sort_all(&mut self) {
        self.cameras.sort_by_key(|c| c.camera_id);
        self.images.sort_by_key(|i| i.image_id);
        self.points3d.sort_by_key(|p| p.point3d_id);
    }
}

// ---------------------------------------------------------------------------
// Little-endian binary helpers
// ---------------------------------------------------------------------------

struct BinReader<R: Read> {
    r: R,
}

impl<R: Read> BinReader<R> {
    fn new(r: R) -> Self {
        Self { r }
    }
    fn exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.r.read_exact(buf)
    }
    fn u8(&mut self) -> io::Result<u8> {
        let mut b = [0u8; 1];
        self.exact(&mut b)?;
        Ok(b[0])
    }
    fn i32(&mut self) -> io::Result<i32> {
        let mut b = [0u8; 4];
        self.exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn u64(&mut self) -> io::Result<u64> {
        let mut b = [0u8; 8];
        self.exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn i64(&mut self) -> io::Result<i64> {
        let mut b = [0u8; 8];
        self.exact(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }
    fn f64(&mut self) -> io::Result<f64> {
        let mut b = [0u8; 8];
        self.exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }
    fn f64s(&mut self, n: usize) -> io::Result<Vec<f64>> {
        (0..n).map(|_| self.f64()).collect()
    }
    fn string_u8len(&mut self) -> io::Result<String> {
        let n = self.u8()? as usize;
        let mut b = vec![0u8; n];
        self.exact(&mut b)?;
        String::from_utf8(b)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad utf8 name: {e}")))
    }
}

struct BinWriter<W: Write> {
    w: W,
}

impl<W: Write> BinWriter<W> {
    fn new(w: W) -> Self {
        Self { w }
    }
    fn put(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.w.write_all(bytes)
    }
    fn u8(&mut self, v: u8) -> io::Result<()> {
        self.put(&[v])
    }
    fn i32(&mut self, v: i32) -> io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn u64(&mut self, v: u64) -> io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn i64(&mut self, v: i64) -> io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn f64(&mut self, v: f64) -> io::Result<()> {
        self.put(&v.to_le_bytes())
    }
    fn string_u8len(&mut self, s: &str) -> io::Result<()> {
        let b = s.as_bytes();
        if b.len() > 255 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("image name longer than 255 bytes: {}", b.len()),
            ));
        }
        self.u8(b.len() as u8)?;
        self.put(b)
    }
}

fn open_bin(path: &Path) -> io::Result<BufReader<File>> {
    Ok(BufReader::new(File::open(path).map_err(|e| {
        io::Error::new(e.kind(), format!("{}: {e}", path.display()))
    })?))
}

fn create_bin(path: &Path) -> io::Result<BufWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            io::Error::new(e.kind(), format!("{}: {e}", parent.display()))
        })?;
    }
    Ok(BufWriter::new(File::create(path).map_err(|e| {
        io::Error::new(e.kind(), format!("{}: {e}", path.display()))
    })?))
}

fn model_or_err(id: i32) -> io::Result<&'static ModelInfo> {
    model_by_id(id).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("unknown camera model id {id}"))
    })
}

// ---------------------------------------------------------------------------
// Binary read
// ---------------------------------------------------------------------------

fn read_cameras_bin(path: &Path) -> io::Result<Vec<Camera>> {
    let mut r = BinReader::new(open_bin(path)?);
    let n = r.u64()?;
    let mut out = Vec::with_capacity(n.min(1 << 20) as usize);
    for _ in 0..n {
        let camera_id = r.i32()?;
        let model_id = r.i32()?;
        let width = r.u64()?;
        let height = r.u64()?;
        let model = model_or_err(model_id)?;
        let params = r.f64s(model.num_params)?;
        out.push(Camera { camera_id, model_id, width, height, params });
    }
    Ok(out)
}

fn read_images_bin(path: &Path) -> io::Result<Vec<Image>> {
    let mut r = BinReader::new(open_bin(path)?);
    let n = r.u64()?;
    let mut out = Vec::with_capacity(n.min(1 << 20) as usize);
    for _ in 0..n {
        let image_id = r.i32()?;
        let q = r.f64s(4)?;
        let t = r.f64s(3)?;
        let camera_id = r.i32()?;
        let name = r.string_u8len()?;
        let num_points = r.u64()?;
        let mut points2d = Vec::with_capacity(num_points.min(1 << 22) as usize);
        for _ in 0..num_points {
            let x = r.f64()?;
            let y = r.f64()?;
            let point3d_id = r.i64()?;
            points2d.push(Point2D { x, y, point3d_id });
        }
        out.push(Image {
            image_id,
            qvec: [q[0], q[1], q[2], q[3]],
            tvec: [t[0], t[1], t[2]],
            camera_id,
            name,
            points2d,
        });
    }
    Ok(out)
}

fn read_points3d_bin(path: &Path) -> io::Result<Vec<Point3D>> {
    let mut r = BinReader::new(open_bin(path)?);
    let n = r.u64()?;
    let mut out = Vec::with_capacity(n.min(1 << 24) as usize);
    for _ in 0..n {
        let point3d_id = r.i64()?;
        let x = r.f64()?;
        let y = r.f64()?;
        let z = r.f64()?;
        let rr = r.u8()?;
        let gg = r.u8()?;
        let bb = r.u8()?;
        let error = r.f64()?;
        let track_len = r.u64()?;
        let mut track = Vec::with_capacity(track_len.min(1 << 20) as usize);
        for _ in 0..track_len {
            let image_id = r.i32()?;
            let point2d_idx = r.i32()?;
            track.push(TrackElement { image_id, point2d_idx: point2d_idx as u32 });
        }
        out.push(Point3D { point3d_id, xyz: [x, y, z], rgb: [rr, gg, bb], error, track });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Binary write
// ---------------------------------------------------------------------------

fn write_cameras_bin(cameras: &[Camera], path: &Path) -> io::Result<()> {
    let mut w = BinWriter::new(create_bin(path)?);
    w.u64(cameras.len() as u64)?;
    for c in cameras {
        w.i32(c.camera_id)?;
        w.i32(c.model_id)?;
        w.u64(c.width)?;
        w.u64(c.height)?;
        for &p in &c.params {
            w.f64(p)?;
        }
    }
    w.w.flush()
}

fn write_images_bin(images: &[Image], path: &Path) -> io::Result<()> {
    let mut w = BinWriter::new(create_bin(path)?);
    w.u64(images.len() as u64)?;
    for i in images {
        w.i32(i.image_id)?;
        for &q in &i.qvec {
            w.f64(q)?;
        }
        for &t in &i.tvec {
            w.f64(t)?;
        }
        w.i32(i.camera_id)?;
        w.string_u8len(&i.name)?;
        w.u64(i.points2d.len() as u64)?;
        for p in &i.points2d {
            w.f64(p.x)?;
            w.f64(p.y)?;
            w.i64(p.point3d_id)?;
        }
    }
    w.w.flush()
}

fn write_points3d_bin(points: &[Point3D], path: &Path) -> io::Result<()> {
    let mut w = BinWriter::new(create_bin(path)?);
    w.u64(points.len() as u64)?;
    for p in points {
        w.i64(p.point3d_id)?;
        for &v in &p.xyz {
            w.f64(v)?;
        }
        w.u8(p.rgb[0])?;
        w.u8(p.rgb[1])?;
        w.u8(p.rgb[2])?;
        w.f64(p.error)?;
        w.u64(p.track.len() as u64)?;
        for t in &p.track {
            w.i32(t.image_id)?;
            w.i32(t.point2d_idx as i32)?;
        }
    }
    w.w.flush()
}

/// Write a reconstruction as COLMAP binary: `<dir>/cameras.bin`,
/// `<dir>/images.bin`, `<dir>/points3D.bin`.
pub fn write_bin_dir(rec: &Reconstruction, dir: &Path) -> io::Result<()> {
    write_cameras_bin(&rec.cameras, &dir.join("cameras.bin"))?;
    write_images_bin(&rec.images, &dir.join("images.bin"))?;
    write_points3d_bin(&rec.points3d, &dir.join("points3D.bin"))?;
    Ok(())
}

/// Read a COLMAP binary sparse directory.
pub fn read_bin_dir(dir: &Path) -> io::Result<Reconstruction> {
    Ok(Reconstruction {
        cameras: read_cameras_bin(&dir.join("cameras.bin"))?,
        images: read_images_bin(&dir.join("images.bin"))?,
        points3d: read_points3d_bin(&dir.join("points3D.bin"))?,
    })
}

// ---------------------------------------------------------------------------
// Text read/write (COLMAP cameras.txt / images.txt / points3D.txt)
// ---------------------------------------------------------------------------

fn lines_of(path: &Path) -> io::Result<impl Iterator<Item = String>> {
    let f = File::open(path).map_err(|e| {
        io::Error::new(e.kind(), format!("{}: {e}", path.display()))
    })?;
    Ok(BufReader::new(f).lines().map(|l| l.unwrap_or_default()))
}

fn txt_data_lines(path: &Path) -> io::Result<Vec<String>> {
    Ok(lines_of(path)?
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect())
}

fn parse_f64(s: &str) -> io::Result<f64> {
    s.parse::<f64>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad float `{s}`: {e}")))
}

fn parse_i64(s: &str) -> io::Result<i64> {
    s.parse::<i64>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad int `{s}`: {e}")))
}

pub fn write_txt_dir(rec: &Reconstruction, dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir).map_err(|e| {
        io::Error::new(e.kind(), format!("{}: {e}", dir.display()))
    })?;

    let model_names: Vec<&str> = MODELS.iter().map(|m| m.name).collect();

    // cameras.txt
    let mut s = String::new();
    s.push_str("# Camera list with one line of data per camera:\n");
    s.push_str("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
    s.push_str(&format!("# Supported models: {}\n", model_names.join(", ")));
    for c in &rec.cameras {
        let name = model_by_id(c.model_id).map(|m| m.name).unwrap_or("UNKNOWN");
        s.push_str(&format!(
            "{} {} {} {}",
            c.camera_id, name, c.width, c.height
        ));
        for p in &c.params {
            s.push_str(&format!(" {}", p));
        }
        s.push('\n');
    }
    fs::write(dir.join("cameras.txt"), s)?;

    // images.txt
    let mut s = String::new();
    s.push_str("# Image list with two lines of data per image:\n");
    s.push_str("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
    s.push_str("#   POINTS2D[] as (X, Y, POINT3D_ID)\n");
    for i in &rec.images {
        s.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {}\n",
            i.image_id, i.qvec[0], i.qvec[1], i.qvec[2], i.qvec[3],
            i.tvec[0], i.tvec[1], i.tvec[2], i.camera_id, i.name
        ));
        if i.points2d.is_empty() {
            s.push('\n');
        } else {
            for (k, p) in i.points2d.iter().enumerate() {
                if k > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{} {} {}", p.x, p.y, p.point3d_id));
            }
            s.push('\n');
        }
    }
    fs::write(dir.join("images.txt"), s)?;

    // points3D.txt
    let mut s = String::new();
    s.push_str("# 3D point list with one line of data per point:\n");
    s.push_str("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    for p in &rec.points3d {
        s.push_str(&format!(
            "{} {} {} {} {} {} {} {}",
            p.point3d_id, p.xyz[0], p.xyz[1], p.xyz[2], p.rgb[0], p.rgb[1], p.rgb[2], p.error
        ));
        for t in &p.track {
            s.push_str(&format!(" {} {}", t.image_id, t.point2d_idx));
        }
        s.push('\n');
    }
    fs::write(dir.join("points3D.txt"), s)?;

    Ok(())
}

pub fn read_txt_dir(dir: &Path) -> io::Result<Reconstruction> {
    // cameras.txt
    let mut cameras = Vec::new();
    for line in txt_data_lines(&dir.join("cameras.txt"))? {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 5 {
            continue;
        }
        let camera_id = parse_i64(t[0])? as i32;
        let model = model_by_name(t[1]).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("unknown model `{}`", t[1]))
        })?;
        let width = parse_i64(t[2])? as u64;
        let height = parse_i64(t[3])? as u64;
        let mut params = Vec::with_capacity(model.num_params);
        for tok in t.iter().take(4 + model.num_params).skip(4) {
            params.push(parse_f64(tok)?);
        }
        cameras.push(Camera { camera_id, model_id: model.id, width, height, params });
    }

    // images.txt (two lines per image)
    let data = txt_data_lines(&dir.join("images.txt"))?;
    let mut images = Vec::new();
    let mut it = data.into_iter();
    while let Some(l1) = it.next() {
        let l2 = it.next().unwrap_or_default();
        let t: Vec<&str> = l1.split_whitespace().collect();
        if t.len() < 10 {
            continue;
        }
        let image_id = parse_i64(t[0])? as i32;
        let qvec = [
            parse_f64(t[1])?, parse_f64(t[2])?, parse_f64(t[3])?, parse_f64(t[4])?,
        ];
        let tvec = [parse_f64(t[5])?, parse_f64(t[6])?, parse_f64(t[7])?];
        let camera_id = parse_i64(t[8])? as i32;
        let name = t[9..].join(" ");
        let mut points2d = Vec::new();
        for tok in l2.split_whitespace().collect::<Vec<_>>().chunks(3) {
            if tok.len() != 3 {
                continue;
            }
            points2d.push(Point2D {
                x: parse_f64(tok[0])?,
                y: parse_f64(tok[1])?,
                point3d_id: parse_i64(tok[2])?,
            });
        }
        images.push(Image { image_id, qvec, tvec, camera_id, name, points2d });
    }

    // points3D.txt
    let mut points3d = Vec::new();
    for line in txt_data_lines(&dir.join("points3D.txt"))? {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 8 {
            continue;
        }
        let point3d_id = parse_i64(t[0])?;
        let xyz = [parse_f64(t[1])?, parse_f64(t[2])?, parse_f64(t[3])?];
        let rgb = [
            parse_i64(t[4])? as u8,
            parse_i64(t[5])? as u8,
            parse_i64(t[6])? as u8,
        ];
        let error = parse_f64(t[7])?;
        let mut track = Vec::new();
        for tok in t[8..].chunks(2) {
            if tok.len() != 2 {
                continue;
            }
            track.push(TrackElement {
                image_id: parse_i64(tok[0])? as i32,
                point2d_idx: parse_i64(tok[1])? as u32,
            });
        }
        points3d.push(Point3D { point3d_id, xyz, rgb, error, track });
    }

    let mut rec = Reconstruction { cameras, images, points3d };
    rec.sort_all();
    Ok(rec)
}

/// Read whichever format is present: prefers `.bin`, falls back to `.txt`.
pub fn read_dir_any(dir: &Path) -> io::Result<Reconstruction> {
    if dir.join("cameras.bin").exists() {
        read_bin_dir(dir)
    } else {
        read_txt_dir(dir)
    }
}
