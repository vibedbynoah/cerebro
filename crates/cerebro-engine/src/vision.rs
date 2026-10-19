//! Vision encoder for Qwen3-VL style projectors (`clip.projector_type = qwen3vl_merger`, the
//! `mmproj` file beside a Qwen3.5 model): an image in, one row of the language model's width per
//! 2x2 patch block out, ready for `Session::feed_parts`.
//!
//! The graph follows the reference implementation (llama.cpp `tools/mtmd/models/qwen3vl.cpp`):
//! 16x16 patches embedded by two temporal kernels (one image is both frames, so the kernels are
//! summed), tokens ordered by 2x2 block, a learned 48x48 position grid interpolated to the image,
//! 12 pre-norm blocks with 2D rotary attention (rows on the first half of each head's frequencies,
//! columns on the second), a final norm, then four neighbouring tokens merged and projected.
//!
//! Everything runs in f32 on the processor. Measured on the 0.8B projector: 8-bit weights and
//! activations (the language model's VNNI path) left the output at cosine 0.992 from f32 and were
//! slower than the register-blocked f32 kernel here (0.80 s vs 0.66 s of matrix products for a
//! 768 x 768 image), while f32 matches the reference implementation to 0.3%.

use std::path::Path;

use crate::gguf::Gguf;
use crate::quant::{dequant_row, DType};
use crate::Error;

/// Shape of the encoder, from the projector's metadata.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub dim: usize,
    pub heads: usize,
    pub ff: usize,
    pub patch: usize,
    pub merge: usize,
    /// Output width (the language model's width).
    pub proj: usize,
    pub eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Merged tokens an image may become: images are resized into this budget.
    pub min_tokens: usize,
    pub max_tokens: usize,
}

/// A normalised image: `width * height` pixels, RGB interleaved, row by row; both sides multiples
/// of `patch * merge`.
pub struct ImageInput {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<f32>,
}

/// The encoder's output: `rows * cols` tokens of `dim` values, row by row.
#[derive(Debug, Clone)]
pub struct ImageEmbedding {
    pub cols: usize,
    pub rows: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

/// A weight matrix (out x in, row-major f32) and its bias.
pub(crate) struct Lin {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) w: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

struct Block {
    ln1: (Vec<f32>, Vec<f32>),
    qkv: Lin,
    out: Lin,
    ln2: (Vec<f32>, Vec<f32>),
    up: Lin,
    down: Lin,
}

pub struct VisionModel {
    pub cfg: VisionConfig,
    patch: Lin,
    /// Learned positions, `grid * grid` rows of `dim`, row-major over (y, x).
    pos: Vec<f32>,
    grid: usize,
    blocks: Vec<Block>,
    post: (Vec<f32>, Vec<f32>),
    mm0: Lin,
    mm2: Lin,
}

pub(crate) fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>, Error> {
    let t = g.tensor(name)?;
    if !matches!(t.dtype, DType::F32 | DType::F16 | DType::BF16 | DType::Q8_0) {
        return Err(Error::Unsupported(format!("vision tensor {name} in {:?}", t.dtype)));
    }
    let mut out = vec![0f32; t.elements() as usize];
    dequant_row(t.dtype, g.bytes(t), &mut out);
    Ok(out)
}

pub(crate) fn lin(g: &Gguf, name: &str) -> Result<Lin, Error> {
    let t = g.tensor(&format!("{name}.weight"))?;
    let (cols, rows) = (t.cols(), t.rows());
    if cols % 8 != 0 {
        return Err(Error::Unsupported(format!("vision matrix {name} of width {cols}")));
    }
    let w = f32s(g, &format!("{name}.weight"))?;
    let bias = if g.tensors.contains_key(&format!("{name}.bias")) { f32s(g, &format!("{name}.bias"))? } else { vec![0.0; rows] };
    Ok(Lin { rows, cols, w, bias })
}

pub(crate) fn norm(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<f32>), Error> {
    Ok((f32s(g, &format!("{name}.weight"))?, f32s(g, &format!("{name}.bias"))?))
}

fn arr3(g: &Gguf, key: &str, default: [f32; 3]) -> [f32; 3] {
    g.meta
        .get(key)
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_f32()).collect::<Vec<f32>>())
        .filter(|v| v.len() == 3)
        .map_or(default, |v| [v[0], v[1], v[2]])
}

impl VisionModel {
    /// Reads a projector file. Only Qwen3-VL style projectors without deepstack layers are
    /// supported for now.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let g = Gguf::open(path)?;
        let ptype = g.str("clip.projector_type").unwrap_or("");
        if ptype != "qwen3vl_merger" {
            return Err(Error::Unsupported(format!("This vision projector type is not supported yet: {ptype}")));
        }
        let deepstack = g
            .meta
            .get("clip.vision.is_deepstack_layers")
            .and_then(|v| v.as_array())
            .is_some_and(|a| a.iter().any(|x| x.as_u64().unwrap_or(0) != 0));
        if deepstack {
            return Err(Error::Unsupported("This vision projector uses deepstack layers, which are not supported yet".into()));
        }
        let u = |k: &str| g.u64(&format!("clip.vision.{k}")).map(|v| v as usize);
        let cfg = VisionConfig {
            dim: u("embedding_length")?,
            heads: u("attention.head_count")?,
            ff: u("feed_forward_length")?,
            patch: u("patch_size")?,
            merge: u("spatial_merge_size").unwrap_or(2),
            proj: u("projection_dim")?,
            eps: g.f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6),
            mean: arr3(&g, "clip.vision.image_mean", [0.5; 3]),
            std: arr3(&g, "clip.vision.image_std", [0.5; 3]),
            min_tokens: 8,
            // 768 x 768 pixels: enough detail for a chat image, and a few seconds to encode.
            max_tokens: 576,
        };
        if cfg.merge != 2 || cfg.dim % cfg.heads != 0 || (cfg.dim / cfg.heads) % 4 != 0 {
            return Err(Error::Unsupported(format!("vision shape (merge {}, width {}, heads {})", cfg.merge, cfg.dim, cfg.heads)));
        }
        // Patch embedding: two temporal kernels [kx, ky, channel, out]; a still image is both
        // frames, so their sum is the kernel.
        let k0 = f32s(&g, "v.patch_embd.weight")?;
        let k1 = if g.tensors.contains_key("v.patch_embd.weight.1") { f32s(&g, "v.patch_embd.weight.1")? } else { vec![0.0; k0.len()] };
        let kin = cfg.patch * cfg.patch * 3;
        if k0.len() != kin * cfg.dim || k1.len() != k0.len() {
            return Err(Error::Format("vision patch embedding has an unexpected shape".into()));
        }
        let sum: Vec<f32> = k0.iter().zip(&k1).map(|(a, b)| a + b).collect();
        let pb = if g.tensors.contains_key("v.patch_embd.bias") { f32s(&g, "v.patch_embd.bias")? } else { vec![0.0; cfg.dim] };
        let patch = Lin { rows: cfg.dim, cols: kin, w: sum, bias: pb };
        let pos = f32s(&g, "v.position_embd.weight")?;
        let grid = ((pos.len() / cfg.dim) as f64).sqrt() as usize;
        if grid * grid * cfg.dim != pos.len() {
            return Err(Error::Format("vision position grid is not square".into()));
        }
        let n_layer = u("block_count")?;
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("v.blk.{i}");
            blocks.push(Block {
                ln1: norm(&g, &format!("{p}.ln1"))?,
                qkv: lin(&g, &format!("{p}.attn_qkv"))?,
                out: lin(&g, &format!("{p}.attn_out"))?,
                ln2: norm(&g, &format!("{p}.ln2"))?,
                up: lin(&g, &format!("{p}.ffn_up"))?,
                down: lin(&g, &format!("{p}.ffn_down"))?,
            });
        }
        let post = norm(&g, "v.post_ln")?;
        let mm0 = lin(&g, "mm.0")?;
        let mm2 = lin(&g, "mm.2")?;
        if mm0.cols != cfg.dim * 4 || mm2.rows != cfg.proj {
            return Err(Error::Format("vision projection has an unexpected shape".into()));
        }
        Ok(VisionModel { cfg, patch, pos, grid, blocks, post, mm0, mm2 })
    }

    /// The size an image is resized to: the aspect ratio kept, both sides multiples of
    /// `patch * merge`, the pixel count within the token budget (the reference's "smart resize").
    pub fn target_size(&self, w: usize, h: usize) -> (usize, usize) {
        let a = self.cfg.patch * self.cfg.merge;
        target_size(w, h, a, self.cfg.min_tokens * a * a, self.cfg.max_tokens * a * a)
    }

    /// 8-bit RGB (interleaved, row by row) to the encoder's input: resized into the budget and
    /// normalised with the projector's mean and deviation.
    pub fn preprocess(&self, rgb8: &[u8], w: usize, h: usize) -> ImageInput {
        let (tw, th) = self.target_size(w.max(1), h.max(1));
        let src: Vec<f32> = rgb8.iter().map(|&v| v as f32 / 255.0).collect();
        let mut rgb = resize(&src, w.max(1), h.max(1), tw, th);
        for px in rgb.chunks_exact_mut(3) {
            for c in 0..3 {
                px[c] = (px[c] - self.cfg.mean[c]) / self.cfg.std[c];
            }
        }
        ImageInput { width: tw, height: th, rgb }
    }

    /// Encodes the reference debugging tool's synthetic images ("white", "black", "gray", "cb",
    /// "red", "green", "blue", "rainbow"), fed raw as that tool feeds them (not normalised).
    pub fn encode_debug(&self, kind: &str, side: usize) -> Result<ImageEmbedding, Error> {
        self.encode(&debug_image(kind, side)?)
    }

    /// Runs the encoder.
    pub fn encode(&self, img: &ImageInput) -> Result<ImageEmbedding, Error> {
        self.encode_traced(img, &mut |_, _| {})
    }

    /// `encode`, handing intermediate results to `trace` (name, tokens x width values): the
    /// reference's names `inp_pos_emb`, `layer_out.<i>`, `post_ln`, `out`.
    pub fn encode_traced(&self, img: &ImageInput, trace: &mut dyn FnMut(&str, &[f32])) -> Result<ImageEmbedding, Error> {
        let c = &self.cfg;
        let a = c.patch * c.merge;
        if img.width == 0 || img.height == 0 || img.width % a != 0 || img.height % a != 0 || img.rgb.len() != img.width * img.height * 3 {
            return Err(Error::Request(format!("the image must be a multiple of {a} pixels on each side")));
        }
        let (pw, ph) = (img.width / c.patch, img.height / c.patch);
        let n = pw * ph;
        let dim = c.dim;
        let hd = dim / c.heads;
        // Token t in 2x2-block order -> its patch (row, column).
        let at = |t: usize| -> (usize, usize) {
            let (blk, sub) = (t / 4, t % 4);
            let (yb, xb) = (blk / (pw / 2), blk % (pw / 2));
            (2 * yb + sub / 2, 2 * xb + sub % 2)
        };
        // Patches as rows of [channel][ky][kx], the kernel's layout.
        let kin = c.patch * c.patch * 3;
        let mut patches = vec![0f32; n * kin];
        for t in 0..n {
            let (py, px) = at(t);
            let row = &mut patches[t * kin..(t + 1) * kin];
            for ch in 0..3 {
                for ky in 0..c.patch {
                    for kx in 0..c.patch {
                        let (y, x) = (py * c.patch + ky, px * c.patch + kx);
                        row[(ch * c.patch + ky) * c.patch + kx] = img.rgb[(y * img.width + x) * 3 + ch];
                    }
                }
            }
        }
        let mut x = vec![0f32; n * dim];
        linear(&self.patch, &patches, n, &mut x, false);
        drop(patches);
        // Learned positions, interpolated to the patch grid (bilinear, corners aligned).
        for t in 0..n {
            let (py, px) = at(t);
            let row = &mut x[t * dim..(t + 1) * dim];
            add_position(&self.pos, self.grid, dim, pw, ph, px, py, row);
        }
        trace("inp_pos_emb", &x);
        // Rotary tables per token: rows on the first half of the frequencies, columns on the second.
        let half = hd / 2;
        let quarter = hd / 4;
        let mut cs = vec![(0f32, 0f32); n * half];
        for t in 0..n {
            let (py, px) = at(t);
            for i in 0..half {
                let p = if i < quarter { py } else { px } as f32;
                let theta = p * 10000f32.powf(-((i % quarter) as f32) / quarter as f32);
                let (s, co) = theta.sin_cos();
                cs[t * half + i] = (co, s);
            }
        }
        let mut h = vec![0f32; n * dim];
        let mut qkv = vec![0f32; n * 3 * dim];
        let mut att = vec![0f32; n * dim];
        let mut ff = vec![0f32; n * c.ff];
        // Section times (CEREBRO_VISION_PROF=1): norms+rope+gelu, matrix products, attention.
        let prof = std::env::var_os("CEREBRO_VISION_PROF").is_some();
        let mut pt = [0f64; 3];
        let mut tick = std::time::Instant::now();
        let mut lap = |k: usize, pt: &mut [f64; 3]| {
            if prof {
                pt[k] += tick.elapsed().as_secs_f64();
                tick = std::time::Instant::now();
            }
        };
        for (li, b) in self.blocks.iter().enumerate() {
            layer_norm_all(&x, &b.ln1.0, &b.ln1.1, c.eps, dim, &mut h);
            lap(0, &mut pt);
            linear(&b.qkv, &h, n, &mut qkv, false);
            lap(1, &mut pt);
            rope_all(&mut qkv, n, c.heads, hd, &cs);
            lap(0, &mut pt);
            attention(&qkv, n, c.heads, hd, &mut att);
            lap(2, &mut pt);
            linear(&b.out, &att, n, &mut x, true);
            lap(1, &mut pt);
            layer_norm_all(&x, &b.ln2.0, &b.ln2.1, c.eps, dim, &mut h);
            lap(0, &mut pt);
            linear(&b.up, &h, n, &mut ff, false);
            lap(1, &mut pt);
            gelu_all(&mut ff);
            lap(0, &mut pt);
            linear(&b.down, &ff, n, &mut x, true);
            lap(1, &mut pt);
            trace(&format!("layer_out.{li}"), &x);
        }
        if prof {
            eprintln!("vision {n} patches: elementwise {:.2} s, matrices {:.2} s, attention {:.2} s", pt[0], pt[1], pt[2]);
        }
        layer_norm_all(&x, &self.post.0, &self.post.1, c.eps, dim, &mut h);
        trace("post_ln", &h);
        // Four neighbours (one 2x2 block, consecutive in token order) become one row.
        let m = n / 4;
        let mut mid = vec![0f32; m * self.mm0.rows];
        linear(&self.mm0, &h, m, &mut mid, false);
        gelu_all(&mut mid);
        let mut out = vec![0f32; m * c.proj];
        linear(&self.mm2, &mid, m, &mut out, false);
        trace("out", &out);
        Ok(ImageEmbedding { cols: pw / 2, rows: ph / 2, dim: c.proj, data: out })
    }
}

/// The reference debugging tool's synthetic images, raw (not normalised), as it feeds them.
pub fn debug_image(kind: &str, side: usize) -> Result<ImageInput, Error> {
    let mut rgb = vec![0f32; side * side * 3];
    let (cx, cy) = (side as f32 / 2.0, side as f32 / 2.0);
    let max_dist = (cx * cx + cy * cy).sqrt();
    for y in 0..side {
        for x in 0..side {
            let p: [f32; 3] = match kind {
                "white" => [1.0; 3],
                "black" => [0.0; 3],
                "gray" => [0.5; 3],
                "cb" => [if (x + y) % 2 == 1 { 0.0 } else { 1.0 }; 3],
                "red" => [1.0, 0.0, 0.0],
                "green" => [0.0, 1.0, 0.0],
                "blue" => [0.0, 0.0, 1.0],
                "rainbow" => {
                    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                    let mut hue = dy.atan2(dx) / (2.0 * 3.141_592_65);
                    if hue < 0.0 {
                        hue += 1.0;
                    }
                    let sat = ((dx * dx + dy * dy).sqrt() / max_dist).min(1.0);
                    let h6 = hue * 6.0;
                    let i6 = h6 as i32;
                    let f = h6 - i6 as f32;
                    let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
                    match i6 % 6 {
                        0 => [1.0, t, p],
                        1 => [q, 1.0, p],
                        2 => [p, 1.0, t],
                        3 => [p, q, 1.0],
                        4 => [t, p, 1.0],
                        _ => [1.0, p, q],
                    }
                }
                _ => return Err(Error::Request(format!("unknown test image {kind}"))),
            };
            rgb[(y * side + x) * 3..(y * side + x) * 3 + 3].copy_from_slice(&p);
        }
    }
    Ok(ImageInput { width: side, height: side, rgb })
}

/// The reference's smart resize: round both sides to multiples of `a`, then scale into
/// [min_px, max_px] keeping the aspect ratio.
pub fn target_size(w: usize, h: usize, a: usize, min_px: usize, max_px: usize) -> (usize, usize) {
    let (wf, hf, af) = (w as f64, h as f64, a as f64);
    let round = |v: f64| ((v / af).round() as usize) * a;
    let floor = |v: f64| ((v / af).floor() as usize) * a;
    let ceil = |v: f64| ((v / af).ceil() as usize) * a;
    let (mut tw, mut th) = (round(wf).max(a), round(hf).max(a));
    if tw * th > max_px {
        let beta = (hf * wf / max_px as f64).sqrt();
        th = floor(hf / beta).max(a);
        tw = floor(wf / beta).max(a);
    } else if tw * th < min_px {
        let beta = (min_px as f64 / (hf * wf)).sqrt();
        th = ceil(hf * beta);
        tw = ceil(wf * beta);
    }
    (tw, th)
}

/// Separable resize of interleaved RGB with a triangle filter widened when shrinking (bilinear
/// with antialiasing), so a large photo is averaged rather than sampled.
pub fn resize(src: &[f32], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    if (w, h) == (tw, th) {
        return src.to_vec();
    }
    let weights = |n_in: usize, n_out: usize| -> Vec<(usize, Vec<f32>)> {
        let scale = n_in as f32 / n_out as f32;
        let support = scale.max(1.0);
        (0..n_out)
            .map(|o| {
                let center = (o as f32 + 0.5) * scale - 0.5;
                let lo = ((center - support).floor().max(0.0)) as usize;
                let hi = ((center + support).ceil() as usize).min(n_in - 1);
                let mut ws: Vec<f32> = (lo..=hi).map(|i| (1.0 - ((i as f32 - center) / support).abs()).max(0.0)).collect();
                let s: f32 = ws.iter().sum();
                if s > 0.0 {
                    ws.iter_mut().for_each(|v| *v /= s);
                } else {
                    ws = vec![0.0; hi - lo + 1];
                    ws[(center.round().max(lo as f32) as usize).min(hi) - lo] = 1.0;
                }
                (lo, ws)
            })
            .collect()
    };
    let wx = weights(w, tw);
    let wy = weights(h, th);
    let mut mid = vec![0f32; h * tw * 3];
    for y in 0..h {
        for (ox, (lo, ws)) in wx.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, wt) in ws.iter().enumerate() {
                let p = &src[(y * w + lo + k) * 3..(y * w + lo + k) * 3 + 3];
                for c in 0..3 {
                    acc[c] += wt * p[c];
                }
            }
            mid[(y * tw + ox) * 3..(y * tw + ox) * 3 + 3].copy_from_slice(&acc);
        }
    }
    let mut out = vec![0f32; th * tw * 3];
    for (oy, (lo, ws)) in wy.iter().enumerate() {
        for x in 0..tw {
            let mut acc = [0f32; 3];
            for (k, wt) in ws.iter().enumerate() {
                let p = &mid[((lo + k) * tw + x) * 3..((lo + k) * tw + x) * 3 + 3];
                for c in 0..3 {
                    acc[c] += wt * p[c];
                }
            }
            out[(oy * tw + x) * 3..(oy * tw + x) * 3 + 3].copy_from_slice(&acc);
        }
    }
    out
}

/// Adds the learned position of patch (px, py) of a pw x ph grid to `row`: the `grid` x `grid`
/// table sampled bilinearly with corners aligned (source = out * (in - 1) / (out - 1)).
#[allow(clippy::too_many_arguments)]
fn add_position(pos: &[f32], grid: usize, dim: usize, pw: usize, ph: usize, px: usize, py: usize, row: &mut [f32]) {
    let coord = |o: usize, n_out: usize| -> f32 {
        if n_out > 1 && grid > 1 {
            o as f32 * (grid - 1) as f32 / (n_out - 1) as f32
        } else {
            o as f32 * grid as f32 / n_out as f32
        }
    };
    let (sx, sy) = (coord(px, pw), coord(py, ph));
    let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
    let (x0, y0) = (x0.min(grid - 1), y0.min(grid - 1));
    let (x1, y1) = ((x0 + 1).min(grid - 1), (y0 + 1).min(grid - 1));
    let (dx, dy) = ((sx - x0 as f32).clamp(0.0, 1.0), (sy - y0 as f32).clamp(0.0, 1.0));
    let r = |y: usize, x: usize| &pos[(y * grid + x) * dim..(y * grid + x + 1) * dim];
    let (a, b, c, d) = (r(y0, x0), r(y0, x1), r(y1, x0), r(y1, x1));
    for i in 0..dim {
        row[i] += a[i] * (1.0 - dx) * (1.0 - dy) + b[i] * dx * (1.0 - dy) + c[i] * (1.0 - dx) * dy + d[i] * dx * dy;
    }
}

pub(crate) struct Ptr<T>(*mut T);
unsafe impl<T> Sync for Ptr<T> {}
unsafe impl<T> Send for Ptr<T> {}
impl<T> Ptr<T> {
    pub(crate) fn new(p: *mut T) -> Self {
        Ptr(p)
    }
    /// SAFETY: callers write disjoint indices from different tasks.
    pub(crate) unsafe fn at(&self, i: usize) -> *mut T {
        self.0.add(i)
    }
}

/// out (n x rows) = x (n x cols) W^T + bias, or `+=` when `add` (a residual). Tasks take a block
/// of rows for a block of tokens; inside, 3 rows x 4 tokens per pass over the inner dimension
/// (12 accumulators in registers).
pub(crate) fn linear(l: &Lin, x: &[f32], n: usize, out: &mut [f32], add: bool) {
    let o = Ptr(out.as_mut_ptr());
    let (rows, cols, w) = (l.rows, l.cols, &l.w);
    const RB: usize = 48;
    let tb = if cols > 1024 { 16 } else { 48 };
    let rtiles = rows.div_ceil(RB);
    crate::pool::global().run(rtiles * n.div_ceil(tb), &|task| {
        let (rt, tt) = (task % rtiles, task / rtiles);
        let (r0, r1) = (rt * RB, ((rt + 1) * RB).min(rows));
        let (t0, t1) = (tt * tb, ((tt + 1) * tb).min(n));
        let mut t = t0;
        while t < t1 {
            let k = (t1 - t).min(4);
            let xs: [&[f32]; 4] = std::array::from_fn(|j| &x[(t + j.min(k - 1)) * cols..(t + j.min(k - 1) + 1) * cols]);
            let mut r = r0;
            while r < r1 {
                let m = (r1 - r).min(3);
                let ws: [&[f32]; 3] = std::array::from_fn(|i| &w[(r + i.min(m - 1)) * cols..(r + i.min(m - 1) + 1) * cols]);
                let d = dot3x4(ws, xs);
                for (i, di) in d.iter().enumerate().take(m) {
                    for (j, v) in di.iter().enumerate().take(k) {
                        // SAFETY: (token, row) pairs are disjoint across tasks.
                        unsafe {
                            let p = o.at((t + j) * rows + r + i);
                            *p = if add { *p + v + l.bias[r + i] } else { v + l.bias[r + i] };
                        }
                    }
                }
                r += m;
            }
            t += k;
        }
    });
}

pub(crate) fn layer_norm_all(x: &[f32], w: &[f32], b: &[f32], eps: f32, dim: usize, out: &mut [f32]) {
    let n = x.len() / dim;
    let o = Ptr(out.as_mut_ptr());
    crate::pool::global().run(n.div_ceil(64), &|i| {
        for t in i * 64..((i + 1) * 64).min(n) {
            let xr = &x[t * dim..(t + 1) * dim];
            let mean = xr.iter().sum::<f32>() / dim as f32;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for j in 0..dim {
                // SAFETY: each task writes its own tokens.
                unsafe { *o.at(t * dim + j) = (xr[j] - mean) * inv * w[j] + b[j] };
            }
        }
    });
}

/// GELU, tanh approximation (the reference's `ggml_gelu`).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

pub(crate) fn gelu_all(x: &mut [f32]) {
    let n = x.len();
    let o = Ptr(x.as_mut_ptr());
    crate::pool::global().run(n.div_ceil(16384), &|i| {
        for j in i * 16384..((i + 1) * 16384).min(n) {
            // SAFETY: disjoint ranges per task.
            unsafe { *o.at(j) = gelu(*o.at(j)) };
        }
    });
}

/// Rotates every head of Q and K: pairs (i, i + hd/2) by the token's angle for frequency i.
fn rope_all(qkv: &mut [f32], n: usize, heads: usize, hd: usize, cs: &[(f32, f32)]) {
    let dim = heads * hd;
    let half = hd / 2;
    let o = Ptr(qkv.as_mut_ptr());
    crate::pool::global().run(n.div_ceil(64), &|i| {
        for t in i * 64..((i + 1) * 64).min(n) {
            let c = &cs[t * half..(t + 1) * half];
            for part in 0..2 {
                for h in 0..heads {
                    let base = t * 3 * dim + part * dim + h * hd;
                    for (k, &(co, s)) in c.iter().enumerate() {
                        // SAFETY: each task rotates its own tokens.
                        unsafe {
                            let (a, b) = (*o.at(base + k), *o.at(base + k + half));
                            *o.at(base + k) = a * co - b * s;
                            *o.at(base + k + half) = a * s + b * co;
                        }
                    }
                }
            }
        }
    });
}

/// Full (non-causal) attention per head: softmax(Q K^T / sqrt(hd)) V into `out` (n x dim).
/// Keys are transposed per head (hd x n, padded to 8) so four queries' scores come out of one
/// pass over them, eight keys at a time; values are copied per head so the weighted sum reads
/// them contiguously.
pub(crate) fn attention(qkv: &[f32], n: usize, heads: usize, hd: usize, out: &mut [f32]) {
    let dim = heads * hd;
    let n8 = n.next_multiple_of(8);
    let pool = crate::pool::global();
    let mut kt = vec![0f32; heads * hd * n8];
    let mut vh = vec![0f32; heads * n * hd];
    {
        let (k, v) = (Ptr(kt.as_mut_ptr()), Ptr(vh.as_mut_ptr()));
        pool.run(heads, &|h| {
            for j in 0..n {
                let src = j * 3 * dim + dim + h * hd;
                for d in 0..hd {
                    // SAFETY: each task fills its own head's slices.
                    unsafe {
                        *k.at((h * hd + d) * n8 + j) = qkv[src + d];
                        *v.at((h * n + j) * hd + d) = qkv[src + dim + d];
                    }
                }
            }
        });
    }
    let scale = 1.0 / (hd as f32).sqrt();
    const QB: usize = 4;
    let qblocks = n.div_ceil(QB);
    const PER_TASK: usize = 4;
    let o = Ptr(out.as_mut_ptr());
    pool.run(heads * qblocks.div_ceil(PER_TASK), &|task| {
        let h = task % heads;
        let qb0 = (task / heads) * PER_TASK;
        let kt = &kt[h * hd * n8..(h + 1) * hd * n8];
        let vh = &vh[h * n * hd..(h + 1) * n * hd];
        let mut s = vec![0f32; QB * n8];
        let mut q = vec![0f32; QB * hd];
        let mut acc = vec![0f32; QB * hd];
        for qb in qb0..(qb0 + PER_TASK).min(qblocks) {
            let (i0, i1) = (qb * QB, ((qb + 1) * QB).min(n));
            let nq = i1 - i0;
            q.iter_mut().for_each(|v| *v = 0.0);
            for qi in 0..nq {
                let src = &qkv[(i0 + qi) * 3 * dim + h * hd..(i0 + qi) * 3 * dim + (h + 1) * hd];
                for d in 0..hd {
                    q[qi * hd + d] = src[d] * scale;
                }
            }
            scores4(&q, kt, hd, n8, &mut s);
            for qi in 0..nq {
                let row = &mut s[qi * n8..qi * n8 + n];
                let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = crate::quant::exp_fast(*v - m);
                    sum += *v;
                }
                let inv = 1.0 / sum;
                row.iter_mut().for_each(|v| *v *= inv);
            }
            weighted4(&s, n8, vh, n, hd, &mut acc);
            for qi in 0..nq {
                for d in 0..hd {
                    // SAFETY: (token, head) slices are disjoint across tasks.
                    unsafe { *o.at((i0 + qi) * dim + h * hd + d) = acc[qi * hd + d] };
                }
            }
        }
    });
}

/// s[qi * n8 + j] = q_qi . k_j for four queries (q: 4 x hd) against transposed keys (hd x n8).
fn scores4(q: &[f32], kt: &[f32], hd: usize, n8: usize, s: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: AVX2/FMA checked; the slices hold 4 x hd, hd x n8 and 4 x n8 values.
            return unsafe { scores4_fma(q, kt, hd, n8, s) };
        }
    }
    for qi in 0..4 {
        for j in 0..n8 {
            s[qi * n8 + j] = (0..hd).map(|d| q[qi * hd + d] * kt[d * n8 + j]).sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scores4_fma(q: &[f32], kt: &[f32], hd: usize, n8: usize, s: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(q.len() >= 4 * hd && kt.len() >= hd * n8 && s.len() >= 4 * n8);
    let (qp, kp, sp) = (q.as_ptr(), kt.as_ptr(), s.as_mut_ptr());
    let mut j = 0;
    while j < n8 {
        let mut a = [_mm256_setzero_ps(); 4];
        for d in 0..hd {
            let k = _mm256_loadu_ps(kp.add(d * n8 + j));
            for (qi, ai) in a.iter_mut().enumerate() {
                *ai = _mm256_fmadd_ps(_mm256_set1_ps(*qp.add(qi * hd + d)), k, *ai);
            }
        }
        for (qi, ai) in a.iter().enumerate() {
            _mm256_storeu_ps(sp.add(qi * n8 + j), *ai);
        }
        j += 8;
    }
}

/// acc[qi] = sum_j p[qi * n8 + j] v_j for four queries (v: n x hd, row-major).
fn weighted4(p: &[f32], n8: usize, v: &[f32], n: usize, hd: usize, acc: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if hd % 16 == 0 && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: AVX2/FMA checked; hd a multiple of 16; sizes as documented.
            return unsafe { weighted4_fma(p, n8, v, n, hd, acc) };
        }
    }
    acc.iter_mut().for_each(|x| *x = 0.0);
    for j in 0..n {
        for qi in 0..4 {
            let w = p[qi * n8 + j];
            for d in 0..hd {
                acc[qi * hd + d] += w * v[j * hd + d];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted4_fma(p: &[f32], n8: usize, v: &[f32], n: usize, hd: usize, acc: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(p.len() >= 4 * n8 && v.len() >= n * hd && acc.len() >= 4 * hd);
    let (pp, vp, ap) = (p.as_ptr(), v.as_ptr(), acc.as_mut_ptr());
    let mut dc = 0;
    while dc < hd {
        let mut a = [[_mm256_setzero_ps(); 2]; 4];
        for j in 0..n {
            let v0 = _mm256_loadu_ps(vp.add(j * hd + dc));
            let v1 = _mm256_loadu_ps(vp.add(j * hd + dc + 8));
            for (qi, ai) in a.iter_mut().enumerate() {
                let w = _mm256_set1_ps(*pp.add(qi * n8 + j));
                ai[0] = _mm256_fmadd_ps(w, v0, ai[0]);
                ai[1] = _mm256_fmadd_ps(w, v1, ai[1]);
            }
        }
        for (qi, ai) in a.iter().enumerate() {
            _mm256_storeu_ps(ap.add(qi * hd + dc), ai[0]);
            _mm256_storeu_ps(ap.add(qi * hd + dc + 8), ai[1]);
        }
        dc += 16;
    }
}

/// The 3 x 4 dot products of rows `w` with tokens `x` (all the same length).
fn dot3x4(w: [&[f32]; 3], x: [&[f32]; 4]) -> [[f32; 4]; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if w[0].len() % 8 == 0 && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: AVX2/FMA checked; every slice is w[0].len() long, a multiple of 8.
            return unsafe { dot3x4_fma(w, x) };
        }
    }
    w.map(|wi| x.map(|xj| wi.iter().zip(xj).map(|(a, b)| a * b).sum()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot3x4_fma(w: [&[f32]; 3], x: [&[f32]; 4]) -> [[f32; 4]; 3] {
    use std::arch::x86_64::*;
    let n = w[0].len();
    assert!(w.iter().all(|r| r.len() >= n) && x.iter().all(|r| r.len() >= n));
    let mut a = [[_mm256_setzero_ps(); 4]; 3];
    let (w0, w1, w2) = (w[0].as_ptr(), w[1].as_ptr(), w[2].as_ptr());
    let xp: [*const f32; 4] = x.map(|r| r.as_ptr());
    let mut i = 0;
    while i < n {
        let wv = [_mm256_loadu_ps(w0.add(i)), _mm256_loadu_ps(w1.add(i)), _mm256_loadu_ps(w2.add(i))];
        for j in 0..4 {
            let xv = _mm256_loadu_ps(xp[j].add(i));
            for r in 0..3 {
                a[r][j] = _mm256_fmadd_ps(wv[r], xv, a[r][j]);
            }
        }
        i += 8;
    }
    let hs = |v: __m256| {
        let hi = _mm256_extractf128_ps(v, 1);
        let q = _mm_add_ps(hi, _mm256_castps256_ps128(v));
        let q = _mm_add_ps(q, _mm_movehl_ps(q, q));
        _mm_cvtss_f32(_mm_add_ss(q, _mm_shuffle_ps(q, q, 1)))
    };
    a.map(|r| r.map(hs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_the_reference_rules() {
        // 640 x 480 is already a multiple of 32 and inside the 576-token budget: unchanged.
        assert_eq!(target_size(640, 480, 32, 8 * 1024, 576 * 1024), (640, 480));
        // A 4000 x 3000 photo scales down to fit 589,824 pixels.
        let (w, h) = target_size(4000, 3000, 32, 8 * 1024, 576 * 1024);
        assert!(w * h <= 576 * 1024 && w % 32 == 0 && h % 32 == 0, "{w}x{h}");
        assert_eq!((w, h), (864, 640));
        // A tiny image grows to the minimum.
        let (w, h) = target_size(20, 20, 32, 8 * 1024, 576 * 1024);
        assert!(w * h >= 8 * 1024, "{w}x{h}");
    }

    #[test]
    fn block_order_puts_each_2x2_together() {
        // The same mapping as `encode`'s `at`, on a 4 x 2 patch grid.
        let pw = 4;
        let at = |t: usize| {
            let (blk, sub) = (t / 4, t % 4);
            let (yb, xb) = (blk / (pw / 2), blk % (pw / 2));
            (2 * yb + sub / 2, 2 * xb + sub % 2)
        };
        let order: Vec<(usize, usize)> = (0..8).map(at).collect();
        assert_eq!(order, vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
    }

    #[test]
    fn positions_interpolate_with_corners_aligned() {
        // A 2 x 2 grid of 1-wide positions: values 0, 1 (top), 2, 3 (bottom).
        let pos = vec![0.0, 1.0, 2.0, 3.0];
        let v = |px, py| {
            let mut r = [0f32];
            add_position(&pos, 2, 1, 3, 3, px, py, &mut r);
            r[0]
        };
        assert_eq!(v(0, 0), 0.0);
        assert_eq!(v(2, 0), 1.0);
        assert_eq!(v(0, 2), 2.0);
        assert_eq!(v(2, 2), 3.0);
        assert_eq!(v(1, 1), 1.5);
    }

    #[test]
    fn gelu_matches_known_values() {
        assert!((gelu(0.0)).abs() < 1e-7);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158_808).abs() < 1e-5);
        assert!((gelu(3.0) - 2.996_363).abs() < 1e-5);
    }

    #[test]
    fn resize_keeps_flat_images_flat_and_averages_when_shrinking() {
        let src = vec![0.25f32; 10 * 6 * 3];
        let out = resize(&src, 10, 6, 4, 3);
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-6));
        // Alternating columns shrink to their average.
        let mut stripes = vec![0f32; 8 * 2 * 3];
        for y in 0..2 {
            for x in 0..8 {
                for c in 0..3 {
                    stripes[(y * 8 + x) * 3 + c] = (x % 2) as f32;
                }
            }
        }
        let out = resize(&stripes, 8, 2, 2, 2);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 0.15), "{out:?}");
    }

    #[test]
    fn linear_matches_a_plain_product() {
        let (rows, cols, n) = (17, 64, 5);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let x: Vec<f32> = (0..n * cols).map(|i| ((i * 13 % 29) as f32 - 14.0) / 14.0).collect();
        let bias: Vec<f32> = (0..rows).map(|r| r as f32 * 0.1).collect();
        let l = Lin { rows, cols, w: w.clone(), bias: bias.clone() };
        let mut out = vec![0f32; n * rows];
        linear(&l, &x, n, &mut out, false);
        for t in 0..n {
            for r in 0..rows {
                let want: f32 = (0..cols).map(|k| w[r * cols + k] * x[t * cols + k]).sum::<f32>() + bias[r];
                let got = out[t * rows + r];
                assert!((got - want).abs() < 1e-4 * (1.0 + want.abs()), "t{t} r{r}: {got} vs {want}");
            }
        }
    }
}
