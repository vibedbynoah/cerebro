//! Image generation with Stable Diffusion 2.x-family models in the diffusers layout (SD-Turbo):
//! a prompt in, an RGB image out, on the graphics card (NVIDIA with tensor cores, `diffusion_gpu.rs`)
//! or on the processor.
//!
//! Follows the diffusers reference (`StableDiffusionPipeline` with `EulerDiscreteScheduler`):
//! the CLIP tokenizer and OpenCLIP text encoder, the UNet (ResNet blocks with GroupNorm and SiLU,
//! linear-projection transformer blocks with self- and cross-attention and GEGLU), Euler steps
//! over "trailing" timesteps for an epsilon-predicting model, and the VAE decoder. Activations
//! are rows of channels (pixel-major), so the vision encoder's matrix and attention kernels apply.
//! Weights are read from safetensors (f16 or f32) and kept in f32.

use std::collections::HashMap;
use std::path::Path;

use crate::vision::{attention, linear, Lin, Ptr};
use crate::Error;

#[cfg(feature = "cuda")]
#[path = "diffusion_gpu.rs"]
mod gpu;

// ---- safetensors ---------------------------------------------------------------------------------

/// A safetensors file, memory-mapped; tensors read to f32 on request.
pub struct SafeTensors {
    map: memmap2::Mmap,
    data0: usize,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let f = std::fs::File::open(path)?;
        // SAFETY: the file is opened read-only and not modified while mapped.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() < 8 {
            return Err(Error::Format("safetensors file too short".into()));
        }
        let n = u64::from_le_bytes(map[0..8].try_into().expect("8 bytes")) as usize;
        if 8 + n > map.len() {
            return Err(Error::Format("safetensors header runs past the file".into()));
        }
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + n]).map_err(|e| Error::Format(format!("safetensors header: {e}")))?;
        let mut tensors = HashMap::new();
        for (k, v) in header.as_object().ok_or_else(|| Error::Format("safetensors header is not an object".into()))? {
            if k == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap_or("").to_owned();
            let shape: Vec<usize> = v["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or_default();
            let off = v["data_offsets"].as_array().ok_or_else(|| Error::Format(format!("{k}: no offsets")))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
            if 8 + n + b > map.len() || a > b {
                return Err(Error::Format(format!("{k}: data outside the file")));
            }
            tensors.insert(k.clone(), (dtype, shape, a, b));
        }
        Ok(SafeTensors { map, data0: 8 + n, tensors })
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        self.tensors.get(name).map(|t| t.1.as_slice()).ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>, Error> {
        let (dtype, _, a, b) = self.tensors.get(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))?;
        let bytes = &self.map[self.data0 + a..self.data0 + b];
        Ok(match dtype.as_str() {
            "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => bytes.chunks_exact(2).map(|c| crate::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            d => return Err(Error::Unsupported(format!("{name}: tensor type {d}"))),
        })
    }

    /// A linear layer `name.weight` [out, in] (+ `name.bias`).
    fn lin(&self, name: &str) -> Result<Lin, Error> {
        let s = self.shape(&format!("{name}.weight"))?.to_vec();
        let (rows, cols) = (s[0], s.iter().skip(1).product());
        let w = self.f32(&format!("{name}.weight"))?;
        let bias = if self.has(&format!("{name}.bias")) { self.f32(&format!("{name}.bias"))? } else { vec![0.0; rows] };
        Ok(Lin { rows, cols, w, bias })
    }

    fn norm(&self, name: &str) -> Result<(Vec<f32>, Vec<f32>), Error> {
        Ok((self.f32(&format!("{name}.weight"))?, self.f32(&format!("{name}.bias"))?))
    }
}

// ---- elementwise and normalisation ---------------------------------------------------------------

fn par_rows(n: usize, f: &(dyn Fn(usize) + Sync)) {
    crate::pool::global().run(n.div_ceil(64), &|i| {
        for r in i * 64..((i + 1) * 64).min(n) {
            f(r);
        }
    });
}

fn silu_all(x: &mut [f32]) {
    let n = x.len();
    let p = Ptr::new(x.as_mut_ptr());
    crate::pool::global().run(n.div_ceil(65536), &|i| {
        for j in i * 65536..((i + 1) * 65536).min(n) {
            // SAFETY: disjoint ranges per task.
            unsafe {
                let v = *p.at(j);
                *p.at(j) = v / (1.0 + (-v).exp());
            }
        }
    });
}

/// LayerNorm of each row of `dim`.
fn layer_norm(x: &[f32], dim: usize, w: &(Vec<f32>, Vec<f32>), eps: f32) -> Vec<f32> {
    let n = x.len() / dim;
    let mut out = vec![0f32; x.len()];
    let o = Ptr::new(out.as_mut_ptr());
    par_rows(n, &|r| {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..dim {
            // SAFETY: each row is written by one task.
            unsafe { *o.at(r * dim + j) = (row[j] - mean) * inv * w.0[j] + w.1[j] };
        }
    });
    out
}

/// GroupNorm over [n pixels][c channels] in `groups` groups of consecutive channels.
fn group_norm(x: &[f32], c: usize, groups: usize, w: &(Vec<f32>, Vec<f32>), eps: f32) -> Vec<f32> {
    let n = x.len() / c;
    let gs = c / groups;
    let mut stats = vec![(0f64, 0f64); groups];
    {
        let sp = Ptr::new(stats.as_mut_ptr());
        crate::pool::global().run(groups, &|g| {
            let (mut s, mut s2) = (0f64, 0f64);
            for p in 0..n {
                for v in &x[p * c + g * gs..p * c + (g + 1) * gs] {
                    s += *v as f64;
                    s2 += (*v as f64) * (*v as f64);
                }
            }
            let cnt = (n * gs) as f64;
            let mean = s / cnt;
            // SAFETY: one task per group.
            unsafe { *sp.at(g) = (mean, (s2 / cnt - mean * mean).max(0.0)) };
        });
    }
    let mut out = vec![0f32; x.len()];
    let o = Ptr::new(out.as_mut_ptr());
    par_rows(n, &|p| {
        for ch in 0..c {
            let (mean, var) = stats[ch / gs];
            let v = ((x[p * c + ch] as f64 - mean) / (var + eps as f64).sqrt()) as f32;
            // SAFETY: each pixel row is written by one task.
            unsafe { *o.at(p * c + ch) = v * w.0[ch] + w.1[ch] };
        }
    });
    out
}

// ---- convolution ---------------------------------------------------------------------------------

/// A 2-D convolution (weights [oc][ic][kh][kw] as rows).
struct Conv {
    lin: Lin,
    ic: usize,
    k: usize,
    stride: usize,
    /// Padding only below and to the right (the VAE encoder's downsampling: diffusers pads
    /// (0, 1, 0, 1) and convolves without padding).
    one_sided: bool,
}

impl Conv {
    fn load(st: &SafeTensors, name: &str, stride: usize) -> Result<Self, Error> {
        let s = st.shape(&format!("{name}.weight"))?.to_vec();
        Ok(Conv { lin: st.lin(name)?, ic: s[1], k: s[2], stride, one_sided: false })
    }

    /// x [h][w][ic] -> ([oh][ow][oc], oh, ow); padding k/2.
    fn run(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let (k, s, ic) = (self.k, self.stride, self.ic);
        let pad = if self.one_sided { 0 } else { k / 2 };
        let extra = if self.one_sided { 1 } else { 2 * pad };
        let (oh, ow) = ((h + extra - k) / s + 1, (w + extra - k) / s + 1);
        let oc = self.lin.rows;
        let mut out = vec![0f32; oh * ow * oc];
        if k == 1 && s == 1 {
            linear(&self.lin, x, h * w, &mut out, false);
            return (out, oh, ow);
        }
        let cols = ic * k * k;
        // Rows of output pixels per chunk, so the patch matrix stays near 64 MB.
        let rows_per = ((16_000_000 / (cols * ow).max(1)).max(1)).min(oh);
        let mut y0 = 0;
        while y0 < oh {
            let ny = rows_per.min(oh - y0);
            let n = ny * ow;
            let mut patches = vec![0f32; n * cols];
            {
                let pp = Ptr::new(patches.as_mut_ptr());
                par_rows(n, &|p| {
                    let (yy, xx) = (y0 + p / ow, p % ow);
                    // SAFETY: each patch row is written by one task.
                    let row = unsafe { std::slice::from_raw_parts_mut(pp.at(p * cols), cols) };
                    for ky in 0..k {
                        let iy = (yy * s + ky) as isize - pad as isize;
                        if iy < 0 || iy as usize >= h {
                            continue;
                        }
                        for kx in 0..k {
                            let ix = (xx * s + kx) as isize - pad as isize;
                            if ix < 0 || ix as usize >= w {
                                continue;
                            }
                            let src = &x[(iy as usize * w + ix as usize) * ic..(iy as usize * w + ix as usize + 1) * ic];
                            for c in 0..ic {
                                row[(c * k + ky) * k + kx] = src[c];
                            }
                        }
                    }
                });
            }
            linear(&self.lin, &patches, n, &mut out[y0 * ow * oc..(y0 + ny) * ow * oc], false);
            y0 += ny;
        }
        (out, oh, ow)
    }
}

fn upsample2(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; 4 * h * w * c];
    let o = Ptr::new(out.as_mut_ptr());
    par_rows(2 * h, &|y| {
        for xx in 0..2 * w {
            let src = &x[((y / 2) * w + xx / 2) * c..((y / 2) * w + xx / 2 + 1) * c];
            for (j, v) in src.iter().enumerate() {
                // SAFETY: each output row is written by one task.
                unsafe { *o.at((y * 2 * w + xx) * c + j) = *v };
            }
        }
    });
    out
}

// ---- building blocks -----------------------------------------------------------------------------

struct Resnet {
    n1: (Vec<f32>, Vec<f32>),
    c1: Conv,
    temb: Option<Lin>,
    n2: (Vec<f32>, Vec<f32>),
    c2: Conv,
    skip: Option<Conv>,
    eps: f32,
}

impl Resnet {
    fn load(st: &SafeTensors, p: &str, eps: f32) -> Result<Self, Error> {
        Ok(Resnet {
            n1: st.norm(&format!("{p}.norm1"))?,
            c1: Conv::load(st, &format!("{p}.conv1"), 1)?,
            temb: if st.has(&format!("{p}.time_emb_proj.weight")) { Some(st.lin(&format!("{p}.time_emb_proj"))?) } else { None },
            n2: st.norm(&format!("{p}.norm2"))?,
            c2: Conv::load(st, &format!("{p}.conv2"), 1)?,
            skip: if st.has(&format!("{p}.conv_shortcut.weight")) { Some(Conv::load(st, &format!("{p}.conv_shortcut"), 1)?) } else { None },
            eps,
        })
    }

    fn run(&self, x: &[f32], h: usize, w: usize, c: usize, temb: Option<&[f32]>) -> Vec<f32> {
        let mut t = group_norm(x, c, 32, &self.n1, self.eps);
        silu_all(&mut t);
        let (mut t, _, _) = self.c1.run(&t, h, w);
        let oc = self.c1.lin.rows;
        if let (Some(l), Some(e)) = (&self.temb, temb) {
            let mut se = e.to_vec();
            silu_all(&mut se);
            let mut add = vec![0f32; oc];
            linear(l, &se, 1, &mut add, false);
            for px in t.chunks_exact_mut(oc) {
                px.iter_mut().zip(&add).for_each(|(a, b)| *a += b);
            }
        }
        let mut t = group_norm(&t, oc, 32, &self.n2, self.eps);
        silu_all(&mut t);
        let (mut t, _, _) = self.c2.run(&t, h, w);
        match &self.skip {
            Some(s) => {
                let (sk, _, _) = s.run(x, h, w);
                t.iter_mut().zip(&sk).for_each(|(a, b)| *a += b);
            }
            None => t.iter_mut().zip(x).for_each(|(a, b)| *a += b),
        }
        t
    }
}

/// Attention of `nq` queries against `nk` keys (one head layout per row: heads x hd), into out.
fn cross_attention(q: &[f32], k: &[f32], v: &[f32], nq: usize, nk: usize, heads: usize, hd: usize, out: &mut [f32]) {
    let dim = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let o = Ptr::new(out.as_mut_ptr());
    par_rows(nq, &|i| {
        let mut s = vec![0f32; nk];
        for h in 0..heads {
            let qh = &q[i * dim + h * hd..i * dim + (h + 1) * hd];
            for j in 0..nk {
                s[j] = qh.iter().zip(&k[j * dim + h * hd..j * dim + (h + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let m = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for v in s.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            for d in 0..hd {
                let mut acc = 0.0;
                for j in 0..nk {
                    acc += s[j] * v[j * dim + h * hd + d];
                }
                // SAFETY: each query row is written by one task.
                unsafe { *o.at(i * dim + h * hd + d) = acc / sum };
            }
        }
    });
}

struct Attn {
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
}

impl Attn {
    fn load(st: &SafeTensors, p: &str) -> Result<Self, Error> {
        Ok(Attn { q: st.lin(&format!("{p}.to_q"))?, k: st.lin(&format!("{p}.to_k"))?, v: st.lin(&format!("{p}.to_v"))?, out: st.lin(&format!("{p}.to_out.0"))? })
    }

    /// Self-attention over `n` rows (context None) or cross-attention to `ctx` rows.
    fn run(&self, x: &[f32], n: usize, ctx: Option<(&[f32], usize)>, heads: usize) -> Vec<f32> {
        let dim = self.q.rows;
        let hd = dim / heads;
        let mut att = vec![0f32; n * dim];
        match ctx {
            None => {
                // Packed [q | k | v] per row, the layout the vision attention kernel takes.
                let mut qkv = vec![0f32; n * 3 * dim];
                let (mut q, mut k, mut v) = (vec![0f32; n * dim], vec![0f32; n * dim], vec![0f32; n * dim]);
                linear(&self.q, x, n, &mut q, false);
                linear(&self.k, x, n, &mut k, false);
                linear(&self.v, x, n, &mut v, false);
                for r in 0..n {
                    qkv[r * 3 * dim..r * 3 * dim + dim].copy_from_slice(&q[r * dim..(r + 1) * dim]);
                    qkv[r * 3 * dim + dim..r * 3 * dim + 2 * dim].copy_from_slice(&k[r * dim..(r + 1) * dim]);
                    qkv[r * 3 * dim + 2 * dim..(r + 1) * 3 * dim].copy_from_slice(&v[r * dim..(r + 1) * dim]);
                }
                attention(&qkv, n, heads, hd, &mut att);
            }
            Some((c, nc)) => {
                let mut q = vec![0f32; n * dim];
                linear(&self.q, x, n, &mut q, false);
                let (mut k, mut v) = (vec![0f32; nc * dim], vec![0f32; nc * dim]);
                linear(&self.k, c, nc, &mut k, false);
                linear(&self.v, c, nc, &mut v, false);
                cross_attention(&q, &k, &v, n, nc, heads, hd, &mut att);
            }
        }
        let mut out = vec![0f32; n * self.out.rows];
        linear(&self.out, &att, n, &mut out, false);
        out
    }
}

struct Transformer {
    norm: (Vec<f32>, Vec<f32>),
    proj_in: Lin,
    n1: (Vec<f32>, Vec<f32>),
    a1: Attn,
    n2: (Vec<f32>, Vec<f32>),
    a2: Attn,
    n3: (Vec<f32>, Vec<f32>),
    ff1: Lin,
    ff2: Lin,
    proj_out: Lin,
}

impl Transformer {
    fn load(st: &SafeTensors, p: &str) -> Result<Self, Error> {
        let b = format!("{p}.transformer_blocks.0");
        Ok(Transformer {
            norm: st.norm(&format!("{p}.norm"))?,
            proj_in: st.lin(&format!("{p}.proj_in"))?,
            n1: st.norm(&format!("{b}.norm1"))?,
            a1: Attn::load(st, &format!("{b}.attn1"))?,
            n2: st.norm(&format!("{b}.norm2"))?,
            a2: Attn::load(st, &format!("{b}.attn2"))?,
            n3: st.norm(&format!("{b}.norm3"))?,
            ff1: st.lin(&format!("{b}.ff.net.0.proj"))?,
            ff2: st.lin(&format!("{b}.ff.net.2"))?,
            proj_out: st.lin(&format!("{p}.proj_out"))?,
        })
    }

    fn run(&self, x: &[f32], n: usize, c: usize, ctx: &[f32], nc: usize) -> Vec<f32> {
        let heads = c / 64;
        let t = group_norm(x, c, 32, &self.norm, 1e-6);
        let mut h = vec![0f32; n * c];
        linear(&self.proj_in, &t, n, &mut h, false);
        let a = self.a1.run(&layer_norm(&h, c, &self.n1, 1e-5), n, None, heads);
        h.iter_mut().zip(&a).for_each(|(p, q)| *p += q);
        let a = self.a2.run(&layer_norm(&h, c, &self.n2, 1e-5), n, Some((ctx, nc)), heads);
        h.iter_mut().zip(&a).for_each(|(p, q)| *p += q);
        // GEGLU: the projection's first half is the value, the second the GELU gate.
        let t = layer_norm(&h, c, &self.n3, 1e-5);
        let inner = self.ff1.rows / 2;
        let mut g = vec![0f32; n * 2 * inner];
        linear(&self.ff1, &t, n, &mut g, false);
        let mut gg = vec![0f32; n * inner];
        {
            let o = Ptr::new(gg.as_mut_ptr());
            par_rows(n, &|r| {
                for j in 0..inner {
                    let (a, b) = (g[r * 2 * inner + j], g[r * 2 * inner + inner + j]);
                    // SAFETY: each row is written by one task.
                    unsafe { *o.at(r * inner + j) = a * crate::audio::gelu_erf(b) };
                }
            });
        }
        let mut f = vec![0f32; n * c];
        linear(&self.ff2, &gg, n, &mut f, false);
        h.iter_mut().zip(&f).for_each(|(p, q)| *p += q);
        let mut out = vec![0f32; n * c];
        linear(&self.proj_out, &h, n, &mut out, false);
        out.iter_mut().zip(x).for_each(|(p, q)| *p += q);
        out
    }
}

// ---- text encoder --------------------------------------------------------------------------------

/// CLIP's byte-level BPE tokenizer (lower case, words end with `</w>`), padded to 77.
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    bos: u32,
    eos: u32,
    pad: u32,
    byte_char: Vec<char>,
    pattern: fancy_regex::Regex,
}

impl ClipTokenizer {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let vocab: HashMap<String, u32> = serde_json::from_slice(&std::fs::read(dir.join("vocab.json"))?).map_err(|e| Error::Format(format!("vocab.json: {e}")))?;
        let merges = std::fs::read_to_string(dir.join("merges.txt"))?;
        let mut ranks = HashMap::new();
        for (i, l) in merges.lines().skip(1).filter(|l| !l.is_empty()).enumerate() {
            if let Some((a, b)) = l.split_once(' ') {
                ranks.insert((a.to_owned(), b.to_owned()), i);
            }
        }
        let special: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("special_tokens_map.json"))?).map_err(|e| Error::Format(format!("special_tokens_map.json: {e}")))?;
        let tok = |k: &str| special[k]["content"].as_str().or(special[k].as_str()).and_then(|s| vocab.get(s).copied());
        // GPT-2's byte to printable character map.
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut byte_char = vec![' '; 256];
        for (b, c) in bs.iter().zip(&cs) {
            byte_char[*b as usize] = char::from_u32(*c).unwrap_or(' ');
        }
        Ok(ClipTokenizer {
            bos: tok("bos_token").unwrap_or(49406),
            eos: tok("eos_token").unwrap_or(49407),
            pad: tok("pad_token").unwrap_or(49407),
            vocab,
            ranks,
            byte_char,
            pattern: fancy_regex::Regex::new(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+").map_err(|e| Error::Format(e.to_string()))?,
        })
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let mut parts: Vec<String> = chars.iter().enumerate().map(|(i, c)| if i + 1 == chars.len() { format!("{c}</w>") } else { c.to_string() }).collect();
        loop {
            let best = (0..parts.len().saturating_sub(1)).filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|r| (*r, i))).min();
            let Some((_, i)) = best else { break };
            let (a, b) = (parts[i].clone(), parts[i + 1].clone());
            // Merge every occurrence of this pair, left to right.
            let mut out = Vec::with_capacity(parts.len());
            let mut j = 0;
            while j < parts.len() {
                if j + 1 < parts.len() && parts[j] == a && parts[j + 1] == b {
                    out.push(format!("{a}{b}"));
                    j += 2;
                } else {
                    out.push(parts[j].clone());
                    j += 1;
                }
            }
            parts = out;
        }
        parts
    }

    /// 77 ids: start, the prompt (cut to fit), end, then padding.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut ids = vec![self.bos];
        for m in self.pattern.find_iter(&clean).flatten() {
            let word: String = m.as_str().bytes().map(|b| self.byte_char[b as usize]).collect();
            for p in self.bpe(&word) {
                if let Some(&id) = self.vocab.get(&p) {
                    ids.push(id);
                }
            }
        }
        ids.truncate(76);
        ids.push(self.eos);
        while ids.len() < 77 {
            ids.push(self.pad);
        }
        ids
    }
}

struct ClipLayer {
    ln1: (Vec<f32>, Vec<f32>),
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: (Vec<f32>, Vec<f32>),
    fc1: Lin,
    fc2: Lin,
}

pub struct TextEncoder {
    tok_embd: Vec<f32>,
    pos_embd: Vec<f32>,
    dim: usize,
    heads: usize,
    layers: Vec<ClipLayer>,
    final_ln: (Vec<f32>, Vec<f32>),
}

impl TextEncoder {
    pub fn load(path: &Path, heads: usize) -> Result<Self, Error> {
        let st = SafeTensors::open(path)?;
        let p = "text_model";
        let dim = st.shape(&format!("{p}.embeddings.token_embedding.weight"))?[1];
        let mut layers = Vec::new();
        for i in 0.. {
            let l = format!("{p}.encoder.layers.{i}");
            if !st.has(&format!("{l}.layer_norm1.weight")) {
                break;
            }
            layers.push(ClipLayer {
                ln1: st.norm(&format!("{l}.layer_norm1"))?,
                q: st.lin(&format!("{l}.self_attn.q_proj"))?,
                k: st.lin(&format!("{l}.self_attn.k_proj"))?,
                v: st.lin(&format!("{l}.self_attn.v_proj"))?,
                o: st.lin(&format!("{l}.self_attn.out_proj"))?,
                ln2: st.norm(&format!("{l}.layer_norm2"))?,
                fc1: st.lin(&format!("{l}.mlp.fc1"))?,
                fc2: st.lin(&format!("{l}.mlp.fc2"))?,
            });
        }
        Ok(TextEncoder {
            tok_embd: st.f32(&format!("{p}.embeddings.token_embedding.weight"))?,
            pos_embd: st.f32(&format!("{p}.embeddings.position_embedding.weight"))?,
            dim,
            heads,
            layers,
            final_ln: st.norm(&format!("{p}.final_layer_norm"))?,
        })
    }

    /// The prompt's 77 x dim embeddings (the last layer, final norm applied).
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        let (n, d) = (ids.len(), self.dim);
        let hd = d / self.heads;
        let mut x = vec![0f32; n * d];
        for (t, &id) in ids.iter().enumerate() {
            for j in 0..d {
                x[t * d + j] = self.tok_embd[id as usize * d + j] + self.pos_embd[t * d + j];
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        for l in &self.layers {
            let h = layer_norm(&x, d, &l.ln1, 1e-5);
            let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
            linear(&l.q, &h, n, &mut q, false);
            linear(&l.k, &h, n, &mut k, false);
            linear(&l.v, &h, n, &mut v, false);
            let mut att = vec![0f32; n * d];
            // Causal attention: token i sees tokens 0..=i.
            for i in 0..n {
                for hh in 0..self.heads {
                    let qh = &q[i * d + hh * hd..i * d + (hh + 1) * hd];
                    let s: Vec<f32> = (0..=i).map(|j| qh.iter().zip(&k[j * d + hh * hd..j * d + (hh + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                    let m = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
                    let sum: f32 = e.iter().sum();
                    for (j, w) in e.iter().enumerate() {
                        for dd in 0..hd {
                            att[i * d + hh * hd + dd] += w / sum * v[j * d + hh * hd + dd];
                        }
                    }
                }
            }
            linear(&l.o, &att, n, &mut x, true);
            let h = layer_norm(&x, d, &l.ln2, 1e-5);
            let mut f = vec![0f32; n * l.fc1.rows];
            linear(&l.fc1, &h, n, &mut f, false);
            f.iter_mut().for_each(|v| *v = crate::audio::gelu_erf(*v));
            linear(&l.fc2, &f, n, &mut x, true);
        }
        layer_norm(&x, d, &self.final_ln, 1e-5)
    }
}

// ---- UNet ----------------------------------------------------------------------------------------

struct DownBlock {
    resnets: Vec<Resnet>,
    attns: Vec<Transformer>,
    down: Option<Conv>,
}

struct UpBlock {
    resnets: Vec<Resnet>,
    attns: Vec<Transformer>,
    up: Option<Conv>,
}

pub struct UNet {
    t1: Lin,
    t2: Lin,
    conv_in: Conv,
    down: Vec<DownBlock>,
    mid: (Resnet, Transformer, Resnet),
    up: Vec<UpBlock>,
    norm_out: (Vec<f32>, Vec<f32>),
    conv_out: Conv,
    base: usize,
}

impl UNet {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let st = SafeTensors::open(path)?;
        let eps = 1e-5;
        let mut down = Vec::new();
        for i in 0.. {
            let p = format!("down_blocks.{i}");
            if !st.has(&format!("{p}.resnets.0.norm1.weight")) {
                break;
            }
            let resnets = (0..2).map(|j| Resnet::load(&st, &format!("{p}.resnets.{j}"), eps)).collect::<Result<Vec<_>, _>>()?;
            let attns = (0..2).filter(|j| st.has(&format!("{p}.attentions.{j}.norm.weight"))).map(|j| Transformer::load(&st, &format!("{p}.attentions.{j}"))).collect::<Result<Vec<_>, _>>()?;
            let down_conv = if st.has(&format!("{p}.downsamplers.0.conv.weight")) { Some(Conv::load(&st, &format!("{p}.downsamplers.0.conv"), 2)?) } else { None };
            down.push(DownBlock { resnets, attns, down: down_conv });
        }
        let mut up = Vec::new();
        for i in 0.. {
            let p = format!("up_blocks.{i}");
            if !st.has(&format!("{p}.resnets.0.norm1.weight")) {
                break;
            }
            let resnets = (0..3).map(|j| Resnet::load(&st, &format!("{p}.resnets.{j}"), eps)).collect::<Result<Vec<_>, _>>()?;
            let attns = (0..3).filter(|j| st.has(&format!("{p}.attentions.{j}.norm.weight"))).map(|j| Transformer::load(&st, &format!("{p}.attentions.{j}"))).collect::<Result<Vec<_>, _>>()?;
            let up_conv = if st.has(&format!("{p}.upsamplers.0.conv.weight")) { Some(Conv::load(&st, &format!("{p}.upsamplers.0.conv"), 1)?) } else { None };
            up.push(UpBlock { resnets, attns, up: up_conv });
        }
        let conv_in = Conv::load(&st, "conv_in", 1)?;
        Ok(UNet {
            t1: st.lin("time_embedding.linear_1")?,
            t2: st.lin("time_embedding.linear_2")?,
            base: conv_in.lin.rows,
            conv_in,
            down,
            mid: (Resnet::load(&st, "mid_block.resnets.0", eps)?, Transformer::load(&st, "mid_block.attentions.0")?, Resnet::load(&st, "mid_block.resnets.1", eps)?),
            up,
            norm_out: st.norm("conv_norm_out")?,
            conv_out: Conv::load(&st, "conv_out", 1)?,
        })
    }

    /// The noise prediction for latents x [h][w][4] at timestep t, given the prompt context.
    pub fn run(&self, x: &[f32], h: usize, w: usize, t: f32, ctx: &[f32], nc: usize) -> Vec<f32> {
        // Sinusoidal time embedding (cos first), then two linears with SiLU.
        let half = self.base / 2;
        let mut te = vec![0f32; self.base];
        for i in 0..half {
            let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            te[i] = (t * f).cos();
            te[half + i] = (t * f).sin();
        }
        let mut e1 = vec![0f32; self.t1.rows];
        linear(&self.t1, &te, 1, &mut e1, false);
        silu_all(&mut e1);
        let mut emb = vec![0f32; self.t2.rows];
        linear(&self.t2, &e1, 1, &mut emb, false);

        let (mut x, mut hh, mut ww) = self.conv_in.run(x, h, w);
        let mut c = self.base;
        let mut skips: Vec<(Vec<f32>, usize)> = vec![(x.clone(), c)];
        for b in &self.down {
            for (j, r) in b.resnets.iter().enumerate() {
                x = r.run(&x, hh, ww, c, Some(&emb));
                c = r.c1.lin.rows;
                if let Some(a) = b.attns.get(j) {
                    x = a.run(&x, hh * ww, c, ctx, nc);
                }
                skips.push((x.clone(), c));
            }
            if let Some(d) = &b.down {
                let (y, oh, ow) = d.run(&x, hh, ww);
                x = y;
                (hh, ww) = (oh, ow);
                skips.push((x.clone(), c));
            }
        }
        x = self.mid.0.run(&x, hh, ww, c, Some(&emb));
        x = self.mid.1.run(&x, hh * ww, c, ctx, nc);
        x = self.mid.2.run(&x, hh, ww, c, Some(&emb));
        for b in &self.up {
            for (j, r) in b.resnets.iter().enumerate() {
                let (s, sc) = skips.pop().expect("skip connection");
                // Concatenate the skip's channels after x's, per pixel.
                let n = hh * ww;
                let mut cat = vec![0f32; n * (c + sc)];
                for p in 0..n {
                    cat[p * (c + sc)..p * (c + sc) + c].copy_from_slice(&x[p * c..(p + 1) * c]);
                    cat[p * (c + sc) + c..(p + 1) * (c + sc)].copy_from_slice(&s[p * sc..(p + 1) * sc]);
                }
                x = r.run(&cat, hh, ww, c + sc, Some(&emb));
                c = r.c1.lin.rows;
                if let Some(a) = b.attns.get(j) {
                    x = a.run(&x, hh * ww, c, ctx, nc);
                }
            }
            if let Some(u) = &b.up {
                let up = upsample2(&x, hh, ww, c);
                (hh, ww) = (hh * 2, ww * 2);
                let (y, _, _) = u.run(&up, hh, ww);
                x = y;
            }
        }
        let mut t = group_norm(&x, c, 32, &self.norm_out, 1e-5);
        silu_all(&mut t);
        self.conv_out.run(&t, hh, ww).0
    }
}

// ---- VAE decoder ---------------------------------------------------------------------------------

pub struct VaeDecoder {
    post_quant: Option<Conv>,
    conv_in: Conv,
    mid: (Resnet, (Vec<f32>, Vec<f32>), Attn, Resnet),
    up: Vec<(Vec<Resnet>, Option<Conv>)>,
    norm_out: (Vec<f32>, Vec<f32>),
    conv_out: Conv,
    pub scaling: f32,
}

impl VaeDecoder {
    pub fn load(path: &Path, scaling: f32) -> Result<Self, Error> {
        let st = SafeTensors::open(path)?;
        let eps = 1e-6;
        let p = "decoder";
        let mut up = Vec::new();
        for i in 0.. {
            let b = format!("{p}.up_blocks.{i}");
            if !st.has(&format!("{b}.resnets.0.norm1.weight")) {
                break;
            }
            let resnets = (0..3).map(|j| Resnet::load(&st, &format!("{b}.resnets.{j}"), eps)).collect::<Result<Vec<_>, _>>()?;
            let u = if st.has(&format!("{b}.upsamplers.0.conv.weight")) { Some(Conv::load(&st, &format!("{b}.upsamplers.0.conv"), 1)?) } else { None };
            up.push((resnets, u));
        }
        let a = format!("{p}.mid_block.attentions.0");
        Ok(VaeDecoder {
            post_quant: if st.has("post_quant_conv.weight") { Some(Conv::load(&st, "post_quant_conv", 1)?) } else { None },
            conv_in: Conv::load(&st, &format!("{p}.conv_in"), 1)?,
            mid: (
                Resnet::load(&st, &format!("{p}.mid_block.resnets.0"), eps)?,
                st.norm(&format!("{a}.group_norm"))?,
                Attn { q: st.lin(&format!("{a}.to_q"))?, k: st.lin(&format!("{a}.to_k"))?, v: st.lin(&format!("{a}.to_v"))?, out: st.lin(&format!("{a}.to_out.0"))? },
                Resnet::load(&st, &format!("{p}.mid_block.resnets.1"), eps)?,
            ),
            up,
            norm_out: st.norm(&format!("{p}.conv_norm_out"))?,
            conv_out: Conv::load(&st, &format!("{p}.conv_out"), 1)?,
            scaling,
        })
    }

    /// Latents [h][w][4] (already divided by the scaling factor) to an image [8h][8w][3] in -1..1.
    pub fn run(&self, z: &[f32], h: usize, w: usize) -> Vec<f32> {
        let z = match &self.post_quant {
            Some(c) => c.run(z, h, w).0,
            None => z.to_vec(),
        };
        let (mut x, mut hh, mut ww) = self.conv_in.run(&z, h, w);
        let mut c = self.conv_in.lin.rows;
        x = self.mid.0.run(&x, hh, ww, c, None);
        {
            let n = hh * ww;
            let t = group_norm(&x, c, 32, &self.mid.1, 1e-6);
            let a = self.mid.2.run(&t, n, None, 1);
            x.iter_mut().zip(&a).for_each(|(p, q)| *p += q);
        }
        x = self.mid.3.run(&x, hh, ww, c, None);
        for (resnets, u) in &self.up {
            for r in resnets {
                x = r.run(&x, hh, ww, c, None);
                c = r.c1.lin.rows;
            }
            if let Some(u) = u {
                let up = upsample2(&x, hh, ww, c);
                (hh, ww) = (hh * 2, ww * 2);
                x = u.run(&up, hh, ww).0;
            }
        }
        let mut t = group_norm(&x, c, 32, &self.norm_out, 1e-6);
        silu_all(&mut t);
        self.conv_out.run(&t, hh, ww).0
    }
}

// ---- VAE encoder (image to latents, for editing an image) -----------------------------------------

/// The VAE's encoder half: an image [H][W][3] in -1..1 to latents [H/8][W/8][4] (the mean of the
/// latent distribution, times the scaling factor), as diffusers' AutoencoderKL.encode does.
pub struct VaeEncoder {
    conv_in: Conv,
    down: Vec<(Vec<Resnet>, Option<Conv>)>,
    mid: (Resnet, (Vec<f32>, Vec<f32>), Attn, Resnet),
    norm_out: (Vec<f32>, Vec<f32>),
    conv_out: Conv,
    quant: Option<Conv>,
    scaling: f32,
}

impl VaeEncoder {
    pub fn load(path: &Path, scaling: f32) -> Result<Self, Error> {
        let st = SafeTensors::open(path)?;
        let eps = 1e-6;
        let p = "encoder";
        let mut down = Vec::new();
        for i in 0.. {
            let b = format!("{p}.down_blocks.{i}");
            if !st.has(&format!("{b}.resnets.0.norm1.weight")) {
                break;
            }
            let mut resnets = Vec::new();
            for j in 0.. {
                if !st.has(&format!("{b}.resnets.{j}.norm1.weight")) {
                    break;
                }
                resnets.push(Resnet::load(&st, &format!("{b}.resnets.{j}"), eps)?);
            }
            let d = if st.has(&format!("{b}.downsamplers.0.conv.weight")) {
                let mut c = Conv::load(&st, &format!("{b}.downsamplers.0.conv"), 2)?;
                c.one_sided = true;
                Some(c)
            } else {
                None
            };
            down.push((resnets, d));
        }
        if down.is_empty() {
            return Err(Error::Unsupported("this image model's VAE has no encoder, so it cannot edit images".into()));
        }
        let a = format!("{p}.mid_block.attentions.0");
        Ok(VaeEncoder {
            conv_in: Conv::load(&st, &format!("{p}.conv_in"), 1)?,
            down,
            mid: (
                Resnet::load(&st, &format!("{p}.mid_block.resnets.0"), eps)?,
                st.norm(&format!("{a}.group_norm"))?,
                Attn { q: st.lin(&format!("{a}.to_q"))?, k: st.lin(&format!("{a}.to_k"))?, v: st.lin(&format!("{a}.to_v"))?, out: st.lin(&format!("{a}.to_out.0"))? },
                Resnet::load(&st, &format!("{p}.mid_block.resnets.1"), eps)?,
            ),
            norm_out: st.norm(&format!("{p}.conv_norm_out"))?,
            conv_out: Conv::load(&st, &format!("{p}.conv_out"), 1)?,
            quant: if st.has("quant_conv.weight") { Some(Conv::load(&st, "quant_conv", 1)?) } else { None },
            scaling,
        })
    }

    /// Image [h][w][3] in -1..1 (h, w multiples of 8) to scaled latents [h/8][w/8][4].
    pub fn run(&self, img: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (mut x, mut hh, mut ww) = self.conv_in.run(img, h, w);
        let mut c = self.conv_in.lin.rows;
        for (resnets, d) in &self.down {
            for r in resnets {
                x = r.run(&x, hh, ww, c, None);
                c = r.c1.lin.rows;
            }
            if let Some(d) = d {
                let (y, nh, nw) = d.run(&x, hh, ww);
                (x, hh, ww) = (y, nh, nw);
            }
        }
        x = self.mid.0.run(&x, hh, ww, c, None);
        {
            let t = group_norm(&x, c, 32, &self.mid.1, 1e-6);
            let a = self.mid.2.run(&t, hh * ww, None, 1);
            x.iter_mut().zip(&a).for_each(|(p, q)| *p += q);
        }
        x = self.mid.3.run(&x, hh, ww, c, None);
        let mut t = group_norm(&x, c, 32, &self.norm_out, 1e-6);
        silu_all(&mut t);
        let mut m = self.conv_out.run(&t, hh, ww).0;
        if let Some(q) = &self.quant {
            m = q.run(&m, hh, ww).0;
        }
        // Moments are [mean (4) | log variance (4)] per pixel: the mean, scaled.
        let oc = m.len() / (hh * ww);
        (0..hh * ww).flat_map(|i| (0..4).map(move |k| (i, k))).map(|(i, k)| m[i * oc + k] * self.scaling).collect()
    }
}

// ---- scheduler and pipeline ----------------------------------------------------------------------

/// The Euler scheduler with "trailing" spacing (scaled-linear betas 0.00085..0.012 over 1000):
/// the (timestep, sigma) of each of `steps` steps, then the final sigma 0.
pub fn euler_schedule(steps: usize) -> Vec<(f32, f32)> {
    let (b0, b1) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let mut ac = 1.0f64;
    let sigmas: Vec<f64> = (0..1000)
        .map(|i| {
            let b = b0 + (b1 - b0) * i as f64 / 999.0;
            ac *= 1.0 - b * b;
            ((1.0 - ac) / ac).sqrt()
        })
        .collect();
    let mut out: Vec<(f32, f32)> = (0..steps)
        .map(|i| {
            let t = (1000.0 - i as f64 * 1000.0 / steps as f64).round() - 1.0;
            (t as f32, sigmas[t as usize] as f32)
        })
        .collect();
    out.push((0.0, 0.0));
    out
}

/// A Stable Diffusion model folder (diffusers layout), loaded onto the graphics card when there
/// is one this backend runs on, otherwise for the processor.
pub struct Diffusion {
    pub tokenizer: ClipTokenizer,
    /// The VAE weights, for the encoder an edit loads on first use.
    vae_path: std::path::PathBuf,
    encoder: std::sync::OnceLock<Result<VaeEncoder, String>>,
    /// The VAE's latent scaling factor.
    pub scaling: f32,
    cpu: Option<(TextEncoder, UNet, VaeDecoder)>,
    #[cfg(feature = "cuda")]
    gpu: Option<(gpu::GText, gpu::GUNet, gpu::GVae)>,
}

impl Diffusion {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let pick = |sub: &str, base: &str| -> std::path::PathBuf {
            let f16 = dir.join(sub).join(format!("{base}.fp16.safetensors"));
            if f16.exists() { f16 } else { dir.join(sub).join(format!("{base}.safetensors")) }
        };
        let te_cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("text_encoder").join("config.json"))?).map_err(|e| Error::Format(format!("text encoder config: {e}")))?;
        let vae_cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("vae").join("config.json"))?).map_err(|e| Error::Format(format!("vae config: {e}")))?;
        let unet_cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("unet").join("config.json"))?).map_err(|e| Error::Format(format!("unet config: {e}")))?;
        if unet_cfg["use_linear_projection"].as_bool() != Some(true) || te_cfg["hidden_act"].as_str() != Some("gelu") {
            return Err(Error::Unsupported("this image model's layout (only Stable Diffusion 2.x-style models such as SD-Turbo for now)".into()));
        }
        let heads = te_cfg["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        let scaling = vae_cfg["scaling_factor"].as_f64().unwrap_or(0.18215) as f32;
        let tokenizer = ClipTokenizer::load(&dir.join("tokenizer"))?;
        let text = || TextEncoder::load(&pick("text_encoder", "model"), heads);
        let unet = || UNet::load(&pick("unet", "diffusion_pytorch_model"));
        let vae = || VaeDecoder::load(&pick("vae", "diffusion_pytorch_model"), scaling);
        #[cfg(feature = "cuda")]
        if let Some(g) = crate::gtensor::Gx::get() {
            // One part at a time, so the processor copy of each is freed once it is on the card.
            let on_card = (|| Ok::<_, Error>((gpu::GText::new(g, text()?)?, gpu::GUNet::new(g, unet()?)?, gpu::GVae::new(g, &vae()?)?)))();
            match on_card {
                Ok(parts) => {
                    // Slice sums in f16: 13% faster on GeForce cards, and the image matches the f32
                    // reference to cosine 1.000000 (`sdcheck`). CEREBRO_HALF_ACC=0 turns it off.
                    g.set_half_accumulation(std::env::var("CEREBRO_HALF_ACC").map_or(true, |v| v != "0"));
                    return Ok(Diffusion { tokenizer, scaling, vae_path: pick("vae", "diffusion_pytorch_model"), encoder: std::sync::OnceLock::new(), cpu: None, gpu: Some(parts) });
                }
                Err(e) => eprintln!("image model on the graphics card: {e}; using the processor"),
            }
        }
        Ok(Diffusion {
            tokenizer,
            scaling,
            vae_path: pick("vae", "diffusion_pytorch_model"),
            encoder: std::sync::OnceLock::new(),
            cpu: Some((text()?, unet()?, vae()?)),
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    /// Whether the model runs on the graphics card.
    pub fn on_gpu(&self) -> bool {
        self.cpu.is_none()
    }

    #[cfg(feature = "cuda")]
    fn card(&self) -> Option<(&'static crate::gtensor::Gx, &(gpu::GText, gpu::GUNet, gpu::GVae))> {
        Some((crate::gtensor::Gx::get()?, self.gpu.as_ref()?))
    }

    /// The prompt's embeddings (ids.len() x dim).
    pub fn encode_text(&self, ids: &[u32]) -> Result<Vec<f32>, Error> {
        #[cfg(feature = "cuda")]
        if let Some((g, p)) = self.card() {
            return g.down(&p.0.encode(g, ids)?);
        }
        Ok(self.cpu.as_ref().expect("a processor model").0.encode(ids))
    }

    /// The UNet's noise prediction for latents x [h][w][4] at timestep t.
    pub fn unet(&self, x: &[f32], h: usize, w: usize, t: f32, ctx: &[f32], nc: usize) -> Result<Vec<f32>, Error> {
        #[cfg(feature = "cuda")]
        if let Some((g, p)) = self.card() {
            return p.1.run(g, x, h, w, t, &g.up(ctx)?, nc);
        }
        Ok(self.cpu.as_ref().expect("a processor model").1.run(x, h, w, t, ctx, nc))
    }

    /// The VAE decoder: latents [h][w][4] (divided by `scaling`) to [8h][8w][3] in -1..1.
    pub fn vae(&self, z: &[f32], h: usize, w: usize) -> Result<Vec<f32>, Error> {
        #[cfg(feature = "cuda")]
        if let Some((g, p)) = self.card() {
            return p.2.run(g, z, h, w);
        }
        Ok(self.cpu.as_ref().expect("a processor model").2.run(z, h, w))
    }

    /// An image for `prompt`: `size` x `size` pixels (a multiple of 64), `steps` Euler steps from
    /// noise drawn with `seed`. RGB, row by row, 0..255.
    pub fn generate(&self, prompt: &str, size: usize, steps: usize, seed: u64) -> Result<Vec<u8>, Error> {
        if size % 64 != 0 || size == 0 {
            return Err(Error::Request("the image size must be a multiple of 64".into()));
        }
        let ids = self.tokenizer.encode(prompt);
        let ctx = self.encode_text(&ids)?;
        let lh = size / 8;
        let mut rng = crate::tts::Rng::new(seed);
        // Standard normal noise (Box-Muller), [h][w][4].
        let mut lat: Vec<f32> = (0..lh * lh * 4)
            .map(|_| {
                let (u1, u2) = (rng.next_f32().max(1e-7), rng.next_f32());
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        let sched = euler_schedule(steps.max(1));
        let init = (sched[0].1 * sched[0].1 + 1.0).sqrt();
        lat.iter_mut().for_each(|v| *v *= init);
        self.finish(lat, lh, &ctx, ids.len(), &sched)
    }

    /// An edit of `image` (RGB, row by row, `size` x `size`, 0..255) following `prompt`: the image
    /// is encoded, noised to `strength` (0 keeps it, 1 starts from noise) and denoised over the last
    /// `steps` steps of a schedule stretched to steps / strength, as diffusers' img2img does.
    pub fn edit(&self, prompt: &str, image: &[u8], size: usize, steps: usize, strength: f32, seed: u64) -> Result<Vec<u8>, Error> {
        if size % 64 != 0 || size == 0 || image.len() != size * size * 3 {
            return Err(Error::Request("the image must be size x size RGB with size a multiple of 64".into()));
        }
        let enc = self.encoder.get_or_init(|| VaeEncoder::load(&self.vae_path, self.scaling).map_err(|e| e.to_string()));
        let enc = enc.as_ref().map_err(|e| Error::Unsupported(e.clone()))?;
        let x: Vec<f32> = image.iter().map(|&v| v as f32 / 127.5 - 1.0).collect();
        let z = enc.run(&x, size, size);
        let ids = self.tokenizer.encode(prompt);
        let ctx = self.encode_text(&ids)?;
        let lh = size / 8;
        let strength = strength.clamp(0.05, 1.0);
        let steps = steps.max(1);
        let total = ((steps as f32 / strength).round() as usize).max(steps);
        let full = euler_schedule(total);
        let sched = &full[total - steps..];
        let mut rng = crate::tts::Rng::new(seed);
        let sigma = sched[0].1;
        let lat: Vec<f32> = z
            .iter()
            .map(|v| {
                let (u1, u2) = (rng.next_f32().max(1e-7), rng.next_f32());
                v + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        self.finish(lat, lh, &ctx, ids.len(), sched)
    }

    /// Denoises latents (pixel-major [h][w][4], already scaled) over `sched` and decodes them.
    pub fn finish(&self, mut lat: Vec<f32>, lh: usize, ctx: &[f32], nc: usize, sched: &[(f32, f32)]) -> Result<Vec<u8>, Error> {
        for i in 0..sched.len() - 1 {
            let (t, sigma) = sched[i];
            let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
            let input: Vec<f32> = lat.iter().map(|v| v * scale).collect();
            let eps = self.unet(&input, lh, lh, t, ctx, nc)?;
            let dt = sched[i + 1].1 - sigma;
            lat.iter_mut().zip(&eps).for_each(|(x, e)| *x += e * dt);
        }
        let z: Vec<f32> = lat.iter().map(|v| v / self.scaling).collect();
        let img = self.vae(&z, lh, lh)?;
        Ok(img.iter().map(|v| (((v / 2.0 + 0.5).clamp(0.0, 1.0)) * 255.0).round() as u8).collect())
    }
}

/// [c][h][w] (planar, as the reference stores tensors) to [h][w][c].
pub fn planar_to_rows(x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for ch in 0..c {
        for p in 0..hw {
            out[p * c + ch] = x[ch * hw + p];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_euler_schedule_matches_the_reference_shape() {
        let s = euler_schedule(1);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].0, 999.0);
        // diffusers: sigma at t=999 for the scaled-linear schedule is about 14.6146.
        assert!((s[0].1 - 14.6146).abs() < 0.01, "{}", s[0].1);
        assert_eq!(s[1], (0.0, 0.0));
        let s4 = euler_schedule(4);
        assert_eq!(s4.iter().map(|x| x.0).collect::<Vec<_>>(), vec![999.0, 749.0, 499.0, 249.0, 0.0]);
    }

    #[test]
    fn group_norm_normalises_each_group() {
        // 2 pixels, 4 channels, 2 groups: group 0 = channels 0,1.
        let x = vec![1.0, 3.0, 10.0, 10.0, 1.0, 3.0, 20.0, 20.0];
        let w = (vec![1.0; 4], vec![0.0; 4]);
        let y = group_norm(&x, 4, 2, &w, 0.0);
        assert!((y[0] + 1.0).abs() < 1e-5 && (y[1] - 1.0).abs() < 1e-5);
        assert!((y[2] + 1.0).abs() < 1e-5 && (y[6] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_3x3_convolution_with_padding() {
        // One channel, identity kernel at the centre plus 1 at the top-left: y = x + x[y-1][x-1].
        let mut w = vec![0.0; 9];
        w[4] = 1.0;
        w[0] = 1.0;
        let conv = Conv { lin: Lin { rows: 1, cols: 9, w, bias: vec![0.0] }, ic: 1, k: 3, stride: 1 };
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (y, h, ww) = conv.run(&x, 2, 2);
        assert_eq!((h, ww), (2, 2));
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0 + 1.0]);
        let conv2 = Conv { lin: Lin { rows: 1, cols: 9, w: vec![1.0; 9], bias: vec![0.0] }, ic: 1, k: 3, stride: 2 };
        let (y, h, _) = conv2.run(&[1.0; 16], 4, 4);
        assert_eq!(h, 2);
        assert_eq!(y[0], 4.0); // the top-left output sees a 2x2 corner
    }
}
