//! Text to speech with Qwen3-TTS: text in, 24 kHz mono samples out.
//!
//! Three parts, following the reference implementation (llama.cpp `tools/mtmd/models/
//! qwen3tts-gen.cpp` and the Qwen3-TTS pipeline in `mtmd-helper-gen.cpp`):
//! - the talker, a qwen3tts language model run by `model::Session`, fed sums of text and codec
//!   embeddings and sampling the first codebook of each 80 ms frame from its codec-only head;
//! - the code predictor (in the projector file), a small transformer that, from the talker's
//!   final hidden state, samples the other 15 codebooks of the frame;
//! - code2wav (also in the projector), which turns the frames' codes into audio: codebook sums,
//!   a causal convolution, a sliding-window transformer, two 2x upsampling ConvNeXt stages and a
//!   DAC decoder (Snake activations, transposed convolutions; 1,920 samples per frame).
//! The reference decodes audio in streaming batches with carried state; here the whole utterance
//! is decoded at once, which is the same computation with the state starting at zero.

use std::path::Path;

use crate::gguf::Gguf;
use crate::model::{Model, Session};
use crate::quant::{gemv_q8_0_x8, repack_q8_0_x8, DType, QAct};
use crate::vision::{f32s, linear, Lin, Ptr};
use crate::Error;

/// Samples per second of the audio.
pub const SAMPLE_RATE: u32 = 24_000;
/// Frames per sliding attention window in code2wav.
const SWA: usize = 72;

/// A weight matrix: f32, or the file's own Q8_0 interleaved for the engine's 8-bit kernels. The
/// code predictor runs 16 small steps per 80 ms frame; read as f32 its weights moved four times the
/// bytes and took 141 ms a frame (measured), so the file's 8-bit form is kept.
enum W {
    F(Lin),
    Q { rows: usize, cols: usize, packed: Vec<u8>, bias: Vec<f32> },
}

impl W {
    fn rows(&self) -> usize {
        match self {
            W::F(l) => l.rows,
            W::Q { rows, .. } => *rows,
        }
    }

    /// out (n x rows) = x (n x cols) W^T + bias.
    fn mul(&self, x: &[f32], n: usize, out: &mut [f32], _add: bool) {
        match self {
            W::F(l) => linear(l, x, n, out, false),
            W::Q { rows, cols, packed, bias } => {
                let (rows, cols) = (*rows, *cols);
                let nb = cols / 32;
                let groups = rows / 8;
                let mut qa = QAct::new(cols);
                for t in 0..n {
                    qa.quantize(&x[t * cols..(t + 1) * cols]);
                    let o = Ptr::new(out[t * rows..(t + 1) * rows].as_mut_ptr());
                    let qa = &qa;
                    crate::pool::global().run(groups.div_ceil(16), &|i| {
                        for gi in i * 16..((i + 1) * 16).min(groups) {
                            let r = gemv_q8_0_x8(packed, nb, gi, qa);
                            for (k, v) in r.iter().enumerate() {
                                // SAFETY: each task writes its own row group.
                                unsafe { *o.at(gi * 8 + k) = v + bias[gi * 8 + k] };
                            }
                        }
                    });
                }
            }
        }
    }
}

/// A matrix (and its bias, if any) from a tensor (ggml layout: input dimension first); `slice`
/// picks one matrix of a stacked [cols, rows, n] tensor.
fn mat_part(g: &Gguf, name: &str, slice: Option<usize>) -> Result<W, Error> {
    let t = g.tensor(name)?;
    let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let bias = match name.strip_suffix(".weight") {
        Some(base) if g.tensors.contains_key(&format!("{base}.bias")) => f32s(g, &format!("{base}.bias"))?,
        _ => vec![0.0; rows],
    };
    let i = slice.unwrap_or(0);
    if t.dtype == DType::Q8_0 && rows % 8 == 0 && cols % 32 == 0 {
        let per = rows * cols / 32 * 34;
        let bytes = &g.bytes(t)[i * per..(i + 1) * per];
        return Ok(W::Q { rows, cols, packed: repack_q8_0_x8(bytes, rows, cols), bias });
    }
    let all = f32s(g, name)?;
    Ok(W::F(Lin { rows, cols, w: all[i * rows * cols..(i + 1) * rows * cols].to_vec(), bias }))
}

fn mat(g: &Gguf, name: &str) -> Result<W, Error> {
    mat_part(g, name, None)
}

fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for (v, g) in x.iter_mut().zip(w) {
        *v *= s * g;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// NeoX rotary on one head (pairs i, i + d/2).
fn rope(x: &mut [f32], pos: usize, theta: f32) {
    let half = x.len() / 2;
    for i in 0..half {
        let f = pos as f32 * theta.powf(-2.0 * i as f32 / x.len() as f32);
        let (s, c) = f.sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

/// A small deterministic random source (xorshift64*).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Samples an index from logits: temperature, top-k, top-p, min-p, then a draw.
pub fn sample(logits: &[f32], temp: f32, top_k: usize, top_p: f32, min_p: f32, rng: &mut Rng) -> usize {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 {
        idx.truncate(top_k);
    }
    let t = temp.max(1e-5);
    let m = logits[idx[0]];
    let mut p: Vec<f32> = idx.iter().map(|&i| ((logits[i] - m) / t).exp()).collect();
    let s: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= s);
    let mut keep = p.len();
    if top_p < 1.0 {
        let mut c = 0.0;
        for (i, v) in p.iter().enumerate() {
            c += v;
            if c >= top_p {
                keep = i + 1;
                break;
            }
        }
    }
    if min_p > 0.0 {
        let floor = p[0] * min_p;
        keep = keep.min(p.iter().take_while(|&&v| v >= floor).count().max(1));
    }
    let s: f32 = p[..keep].iter().sum();
    let mut r = rng.next_f32() * s;
    for i in 0..keep {
        r -= p[i];
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[keep - 1]
}

struct Layer {
    ln1: Vec<f32>,
    q: W,
    k: W,
    v: W,
    o: W,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ln2: Vec<f32>,
    gate: W,
    up: W,
    down: W,
    ls1: Option<Vec<f32>>,
    ls2: Option<Vec<f32>>,
}

fn layer(g: &Gguf, p: &str) -> Result<Layer, Error> {
    let opt = |n: &str| -> Result<Option<Vec<f32>>, Error> { if g.tensors.contains_key(n) { f32s(g, n).map(Some) } else { Ok(None) } };
    Ok(Layer {
        ln1: f32s(g, &format!("{p}.ln1.weight"))?,
        q: mat(g, &format!("{p}.attn_q.weight"))?,
        k: mat(g, &format!("{p}.attn_k.weight"))?,
        v: mat(g, &format!("{p}.attn_v.weight"))?,
        o: mat(g, &format!("{p}.attn_out.weight"))?,
        q_norm: opt(&format!("{p}.attn_q_norm.weight"))?,
        k_norm: opt(&format!("{p}.attn_k_norm.weight"))?,
        ln2: f32s(g, &format!("{p}.ln2.weight"))?,
        gate: mat(g, &format!("{p}.ffn_gate.weight"))?,
        up: mat(g, &format!("{p}.ffn_up.weight"))?,
        down: mat(g, &format!("{p}.ffn_down.weight"))?,
        ls1: opt(&format!("{p}.ls1.weight"))?,
        ls2: opt(&format!("{p}.ls2.weight"))?,
    })
}

/// A run of tokens through pre-norm layers (RMSNorm, attention with rotary and optional per-head
/// q/k norms, SwiGLU, optional layer scales). `kv`: per layer, the keys and values so far
/// (appended to); queries attend causally to at most `window` positions back.
#[allow(clippy::too_many_arguments)]
fn layers_forward(layers: &[Layer], x: &mut [f32], n: usize, pos0: usize, heads: usize, kv_heads: usize, eps: f32, theta: f32, window: usize, kv: &mut [(Vec<f32>, Vec<f32>)]) {
    let dim = x.len() / n;
    for (li, l) in layers.iter().enumerate() {
        let hd = l.q.rows() / heads;
        let kvd = kv_heads * hd;
        let mut h = x.to_vec();
        for t in 0..n {
            rmsnorm(&mut h[t * dim..(t + 1) * dim], &l.ln1, eps);
        }
        let (mut q, mut k, mut v) = (vec![0f32; n * l.q.rows()], vec![0f32; n * kvd], vec![0f32; n * kvd]);
        l.q.mul(&h, n, &mut q, false);
        l.k.mul(&h, n, &mut k, false);
        l.v.mul(&h, n, &mut v, false);
        for t in 0..n {
            for hh in 0..heads {
                let s = &mut q[t * l.q.rows() + hh * hd..t * l.q.rows() + (hh + 1) * hd];
                if let Some(w) = &l.q_norm {
                    rmsnorm(s, w, eps);
                }
                rope(s, pos0 + t, theta);
            }
            for hh in 0..kv_heads {
                let s = &mut k[t * kvd + hh * hd..t * kvd + (hh + 1) * hd];
                if let Some(w) = &l.k_norm {
                    rmsnorm(s, w, eps);
                }
                rope(s, pos0 + t, theta);
            }
        }
        let (kc, vc) = &mut kv[li];
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let total = kc.len() / kvd;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = heads / kv_heads;
        let mut att = vec![0f32; n * l.q.rows()];
        for t in 0..n {
            let me = total - n + t;
            let first = (me + 1).saturating_sub(window);
            for hh in 0..heads {
                let qh = &q[t * l.q.rows() + hh * hd..t * l.q.rows() + (hh + 1) * hd];
                let kh = hh / group;
                let scores: Vec<f32> = (first..=me).map(|j| qh.iter().zip(&kc[j * kvd + kh * hd..j * kvd + (kh + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = e.iter().sum();
                let out = &mut att[t * l.q.rows() + hh * hd..t * l.q.rows() + (hh + 1) * hd];
                for (jj, w) in e.iter().enumerate() {
                    let j = first + jj;
                    let vrow = &vc[j * kvd + kh * hd..j * kvd + (kh + 1) * hd];
                    for d in 0..hd {
                        out[d] += w / sum * vrow[d];
                    }
                }
            }
        }
        let mut o = vec![0f32; n * dim];
        l.o.mul(&att, n, &mut o, false);
        for t in 0..n {
            for d in 0..dim {
                let s = l.ls1.as_ref().map_or(1.0, |w| w[d]);
                x[t * dim + d] += o[t * dim + d] * s;
            }
        }
        let mut h2 = x.to_vec();
        for t in 0..n {
            rmsnorm(&mut h2[t * dim..(t + 1) * dim], &l.ln2, eps);
        }
        let ff = l.gate.rows();
        let (mut gt, mut ut) = (vec![0f32; n * ff], vec![0f32; n * ff]);
        l.gate.mul(&h2, n, &mut gt, false);
        l.up.mul(&h2, n, &mut ut, false);
        for (a, b) in gt.iter_mut().zip(&ut) {
            *a = silu(*a) * b;
        }
        let mut dn = vec![0f32; n * dim];
        l.down.mul(&gt, n, &mut dn, false);
        for t in 0..n {
            for d in 0..dim {
                let s = l.ls2.as_ref().map_or(1.0, |w| w[d]);
                x[t * dim + d] += dn[t * dim + d] * s;
            }
        }
    }
}

/// Causal 1-D convolution over x [T][IC] (time-major) with weight [OC][IC][K] (ggml [K, IC, OC]),
/// dilation `d`, left zero padding; the output is [T][OC]. Time runs in chunks to bound memory.
fn conv1d(x: &[f32], t_len: usize, ic: usize, w: &Lin, k: usize, d: usize) -> Vec<f32> {
    let oc = w.rows;
    let mut out = vec![0f32; t_len * oc];
    let cols = ic * k;
    let chunk = (16_000_000 / cols.max(1)).clamp(64, 8192);
    let mut t0 = 0;
    while t0 < t_len {
        let n = chunk.min(t_len - t0);
        let mut patches = vec![0f32; n * cols];
        {
            let pp = Ptr::new(patches.as_mut_ptr());
            crate::pool::global().run(n.div_ceil(64), &|i| {
                for t in i * 64..((i + 1) * 64).min(n) {
                    // SAFETY: each task fills its own rows.
                    let row = unsafe { std::slice::from_raw_parts_mut(pp.at(t * cols), cols) };
                    for kk in 0..k {
                        let src = (t0 + t) as isize - ((k - 1 - kk) * d) as isize;
                        if src < 0 {
                            continue;
                        }
                        let xs = &x[src as usize * ic..(src as usize + 1) * ic];
                        for c in 0..ic {
                            row[c * k + kk] = xs[c];
                        }
                    }
                }
            });
        }
        linear(w, &patches, n, &mut out[t0 * oc..(t0 + n) * oc], false);
        t0 += n;
    }
    out
}

/// Causal transposed convolution, stride `s`, kernel `k`: x [T][IC] -> [T*s][OC]; the overlap past
/// the end is dropped. `w2` rows are (oc, k) and columns ic.
fn conv_transpose(x: &[f32], t_len: usize, w2: &Lin, oc: usize, k: usize, s: usize, bias: &[f32]) -> Vec<f32> {
    let mut col = vec![0f32; t_len * oc * k];
    linear(w2, x, t_len, &mut col, false);
    let out_len = t_len * s;
    let mut y = vec![0f32; out_len * oc];
    for t in 0..t_len {
        for kk in 0..k {
            let o = t * s + kk;
            if o >= out_len {
                continue;
            }
            let dst = &mut y[o * oc..(o + 1) * oc];
            for c in 0..oc {
                dst[c] += col[t * oc * k + c * k + kk];
            }
        }
    }
    for t in 0..out_len {
        for c in 0..oc {
            y[t * oc + c] += bias[c];
        }
    }
    y
}

/// A transposed-convolution weight ggml [K, OC, IC] as a matrix with rows (oc, k), columns ic.
fn transpose_weight(g: &Gguf, name: &str) -> Result<(Lin, usize, usize, Vec<f32>), Error> {
    let t = g.tensor(&format!("{name}.weight"))?;
    let (k, oc, ic) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
    let w = f32s(g, &format!("{name}.weight"))?;
    let mut w2 = vec![0f32; oc * k * ic];
    for i in 0..ic {
        for o in 0..oc {
            for kk in 0..k {
                w2[(o * k + kk) * ic + i] = w[(i * oc + o) * k + kk];
            }
        }
    }
    let bias = f32s(g, &format!("{name}.bias"))?;
    Ok((Lin { rows: oc * k, cols: ic, w: w2, bias: vec![0.0; oc * k] }, oc, k, bias))
}

fn snake(x: &mut [f32], c: usize, alpha: &[f32], beta: &[f32]) {
    let n = x.len() / c;
    let pp = Ptr::new(x.as_mut_ptr());
    crate::pool::global().run(n.div_ceil(256), &|i| {
        for t in i * 256..((i + 1) * 256).min(n) {
            // SAFETY: each task updates its own rows.
            let row = unsafe { std::slice::from_raw_parts_mut(pp.at(t * c), c) };
            for j in 0..c {
                let s = (row[j] * alpha[j]).sin();
                row[j] += s * s * beta[j];
            }
        }
    });
}

struct ConvW {
    w: Lin,
    k: usize,
    ic: usize,
}

fn conv_w(g: &Gguf, name: &str) -> Result<ConvW, Error> {
    let t = g.tensor(&format!("{name}.weight"))?;
    let (k, ic, oc) = (t.dims[0] as usize, t.dims[1] as usize, t.dims[2] as usize);
    let w = f32s(g, &format!("{name}.weight"))?;
    let bias = if g.tensors.contains_key(&format!("{name}.bias")) { f32s(g, &format!("{name}.bias"))? } else { vec![0.0; oc] };
    Ok(ConvW { w: Lin { rows: oc, cols: ic * k, w, bias }, k, ic })
}

struct Upsample {
    conv: (Lin, usize, usize, Vec<f32>),
    dw: Vec<f32>,
    dw_b: Vec<f32>,
    norm: (Vec<f32>, Vec<f32>),
    pw1: W,
    pw2: W,
    gamma: Vec<f32>,
}

struct Res {
    a1: (Vec<f32>, Vec<f32>),
    conv1: ConvW,
    a2: (Vec<f32>, Vec<f32>),
    conv2: ConvW,
}

struct Dac {
    snake: (Vec<f32>, Vec<f32>),
    up: (Lin, usize, usize, Vec<f32>),
    stride: usize,
    res: Vec<Res>,
}

/// The projector half of Qwen3-TTS: code predictor and code2wav.
pub struct TtsDecoder {
    // code predictor
    cp_layers: Vec<Layer>,
    cp_heads: usize,
    cp_kv_heads: usize,
    cp_eps: f32,
    cp_proj_in: W,
    cp_norm: Vec<f32>,
    cp_heads_w: Vec<W>,
    /// Per acoustic codebook g (1..=15): its table, vocab rows of the talker's width.
    cp_embd: Vec<Vec<f32>>,
    code_vocab: usize,
    /// codec_0 embeddings (the first codebook), talker width.
    out_embd: Vec<f32>,
    talker_dim: usize,
    // code2wav
    q_first: Vec<f32>,
    q_first_out: W,
    q_rest: Vec<Vec<f32>>,
    q_rest_out: W,
    q_dim: usize,
    pre_conv: ConvW,
    tfm_in: W,
    tfm: Vec<Layer>,
    tfm_norm: Vec<f32>,
    tfm_out: W,
    up: Vec<Upsample>,
    dac_entry: ConvW,
    dac: Vec<Dac>,
    post_snake: (Vec<f32>, Vec<f32>),
    post_conv: ConvW,
}

impl TtsDecoder {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let g = Gguf::open(path)?;
        if g.str("clip.gen.audio.projector_type") != Some("qwen3tts_gen") {
            return Err(Error::Unsupported("This speech projector type is not supported yet (Qwen3-TTS only)".into()));
        }
        let u = |k: &str| g.u64(&format!("clip.gen.audio.{k}")).map(|v| v as usize);
        let n_layer = u("block_count")?;
        let cp_layers = (0..n_layer).map(|i| layer(&g, &format!("a.gen.code.blk.{i}"))).collect::<Result<Vec<_>, _>>()?;
        let head_t = g.tensor("a.gen.code.head.weight")?;
        let (hd_in, vocab, n_ac) = (head_t.dims[0] as usize, head_t.dims[1] as usize, head_t.dims[2] as usize);
        let _ = hd_in;
        let cp_heads_w = (0..n_ac).map(|i| mat_part(&g, "a.gen.code.head.weight", Some(i))).collect::<Result<Vec<_>, _>>()?;
        let embd_t = g.tensor("a.gen.code.embd.weight")?;
        let talker_dim = embd_t.dims[0] as usize;
        let embd_all = f32s(&g, "a.gen.code.embd.weight")?;
        let per = talker_dim * embd_t.dims[1] as usize;
        let cp_embd = (0..n_ac).map(|i| embd_all[i * per..(i + 1) * per].to_vec()).collect();
        let q_rest_t = g.tensor("a.gen.wav.quant.rest.codebook.weight")?;
        let q_dim = q_rest_t.dims[0] as usize;
        let q_all = f32s(&g, "a.gen.wav.quant.rest.codebook.weight")?;
        let qper = q_dim * q_rest_t.dims[1] as usize;
        let q_rest = (0..q_rest_t.dims[2] as usize).map(|i| q_all[i * qper..(i + 1) * qper].to_vec()).collect();
        let mut up = Vec::new();
        for i in 0.. {
            let p = format!("a.gen.wav.up.blk.{i}");
            if !g.tensors.contains_key(&format!("{p}.conv.weight")) {
                break;
            }
            up.push(Upsample {
                conv: transpose_weight(&g, &format!("{p}.conv"))?,
                dw: f32s(&g, &format!("{p}.dwconv.weight"))?,
                dw_b: f32s(&g, &format!("{p}.dwconv.bias"))?,
                norm: (f32s(&g, &format!("{p}.norm.weight"))?, f32s(&g, &format!("{p}.norm.bias"))?),
                pw1: mat(&g, &format!("{p}.pw1.weight"))?,
                pw2: mat(&g, &format!("{p}.pw2.weight"))?,
                gamma: f32s(&g, &format!("{p}.gamma"))?,
            });
        }
        let mut dac = Vec::new();
        for i in 0.. {
            let p = format!("a.gen.wav.dac.blk.{i}");
            if !g.tensors.contains_key(&format!("{p}.conv.weight")) {
                break;
            }
            let upw = transpose_weight(&g, &format!("{p}.conv"))?;
            let stride = upw.2 / 2;
            let res = (0..3)
                .map(|j| -> Result<Res, Error> {
                    let r = format!("{p}.res.{j}");
                    Ok(Res {
                        a1: (f32s(&g, &format!("{r}.act1.alpha"))?, f32s(&g, &format!("{r}.act1.beta"))?),
                        conv1: conv_w(&g, &format!("{r}.conv1"))?,
                        a2: (f32s(&g, &format!("{r}.act2.alpha"))?, f32s(&g, &format!("{r}.act2.beta"))?),
                        conv2: conv_w(&g, &format!("{r}.conv2"))?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            dac.push(Dac { snake: (f32s(&g, &format!("{p}.snake.alpha"))?, f32s(&g, &format!("{p}.snake.beta"))?), up: upw, stride, res });
        }
        Ok(TtsDecoder {
            cp_layers,
            cp_heads: u("attention.head_count")?,
            cp_kv_heads: u("attention.head_count_kv")?,
            cp_eps: g.f32("clip.gen.audio.attention.layer_norm_epsilon").unwrap_or(1e-6),
            cp_proj_in: mat(&g, "a.gen.code.proj_in.weight")?,
            cp_norm: f32s(&g, "a.gen.code.output_norm.weight")?,
            cp_heads_w,
            cp_embd,
            code_vocab: vocab,
            out_embd: f32s(&g, "a.gen.code.out_embd.weight")?,
            talker_dim,
            q_first: f32s(&g, "a.gen.wav.quant.first.codebook.weight")?,
            q_first_out: mat(&g, "a.gen.wav.quant.first.out_proj.weight")?,
            q_rest,
            q_rest_out: mat(&g, "a.gen.wav.quant.rest.out_proj.weight")?,
            q_dim,
            pre_conv: conv_w(&g, "a.gen.wav.pre_conv")?,
            tfm_in: mat(&g, "a.gen.wav.tfm.in_proj.weight")?,
            tfm: (0..8).map(|i| layer(&g, &format!("a.gen.wav.tfm.blk.{i}"))).collect::<Result<Vec<_>, _>>()?,
            tfm_norm: f32s(&g, "a.gen.wav.tfm.output_norm.weight")?,
            tfm_out: mat(&g, "a.gen.wav.tfm.out_proj.weight")?,
            up,
            dac_entry: conv_w(&g, "a.gen.wav.dac.entry")?,
            dac,
            post_snake: (f32s(&g, "a.gen.wav.dac.post_snake.alpha")?, f32s(&g, "a.gen.wav.dac.post_snake.beta")?),
            post_conv: conv_w(&g, "a.gen.wav.dac.post_conv")?,
        })
    }

    fn row<'a>(table: &'a [f32], i: usize, dim: usize) -> &'a [f32] {
        &table[i * dim..(i + 1) * dim]
    }

    /// One frame's 16 codes from the talker's hidden state and its first code; also the summed
    /// embedding of all 16 (the talker's next input, before the text overlay).
    pub fn predict_frame(&self, h_state: &[f32], code0: usize, top_k: usize, top_p: f32, rng: &mut Rng) -> (Vec<usize>, Vec<f32>) {
        let td = self.talker_dim;
        let n_ac = self.cp_heads_w.len();
        let mut kv: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); self.cp_layers.len()];
        let dim = self.cp_proj_in.rows();
        let project = |v: &[f32]| {
            let mut o = vec![0f32; dim];
            self.cp_proj_in.mul(v, 1, &mut o, false);
            o
        };
        // Position 0 seeds the cache with the talker's state; position 1 is the first code.
        let mut x = project(h_state);
        layers_forward(&self.cp_layers, &mut x, 1, 0, self.cp_heads, self.cp_kv_heads, self.cp_eps, 1_000_000.0, usize::MAX, &mut kv);
        let code0_embd = Self::row(&self.out_embd, code0, td).to_vec();
        let mut sum = code0_embd.clone();
        let mut codes = vec![code0];
        let mut input = code0_embd;
        for gi in 0..n_ac {
            let mut x = project(&input);
            layers_forward(&self.cp_layers, &mut x, 1, gi + 1, self.cp_heads, self.cp_kv_heads, self.cp_eps, 1_000_000.0, usize::MAX, &mut kv);
            rmsnorm(&mut x, &self.cp_norm, self.cp_eps);
            let mut logits = vec![0f32; self.code_vocab];
            self.cp_heads_w[gi].mul(&x, 1, &mut logits, false);
            let c = sample(&logits, 1.0, top_k, top_p, 0.0, rng);
            codes.push(c);
            let e = Self::row(&self.cp_embd[gi], c, td);
            for (s, v) in sum.iter_mut().zip(e) {
                *s += v;
            }
            input = e.to_vec();
        }
        (codes, sum)
    }

    /// Audio (24 kHz, -1..1) for frames of 16 codes each.
    pub fn decode(&self, frames: &[Vec<usize>]) -> Vec<f32> {
        let n = frames.len();
        if n == 0 {
            return Vec::new();
        }
        let qd = self.q_dim;
        // 1. Codebooks: the first through its projection, the other 15 summed then projected.
        let (mut sem, mut ac) = (vec![0f32; n * qd], vec![0f32; n * qd]);
        for (t, f) in frames.iter().enumerate() {
            sem[t * qd..(t + 1) * qd].copy_from_slice(Self::row(&self.q_first, f[0], qd));
            for (gi, table) in self.q_rest.iter().enumerate() {
                let r = Self::row(table, f[gi + 1], qd);
                for d in 0..qd {
                    ac[t * qd + d] += r[d];
                }
            }
        }
        let hd = self.q_first_out.rows();
        let (mut h, mut h2) = (vec![0f32; n * hd], vec![0f32; n * hd]);
        self.q_first_out.mul(&sem, n, &mut h, false);
        self.q_rest_out.mul(&ac, n, &mut h2, false);
        h.iter_mut().zip(&h2).for_each(|(a, b)| *a += b);
        // 2. Causal convolution, 3. the sliding-window transformer.
        let x = conv1d(&h, n, self.pre_conv.ic, &self.pre_conv.w, self.pre_conv.k, 1);
        let mut t = vec![0f32; n * self.tfm_in.rows()];
        self.tfm_in.mul(&x, n, &mut t, false);
        let mut kv: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); self.tfm.len()];
        layers_forward(&self.tfm, &mut t, n, 0, 16, 16, 1e-5, 10_000.0, SWA, &mut kv);
        let td = self.tfm_in.rows();
        for r in t.chunks_exact_mut(td) {
            rmsnorm(r, &self.tfm_norm, 1e-5);
        }
        let mut x = vec![0f32; n * self.tfm_out.rows()];
        self.tfm_out.mul(&t, n, &mut x, false);
        let mut c = self.tfm_out.rows();
        let mut len = n;
        // 4. Two 2x upsampling stages, each a transposed convolution and a ConvNeXt block.
        for u in &self.up {
            let (w2, oc, k, b) = &u.conv;
            x = conv_transpose(&x, len, w2, *oc, *k, 2, b);
            len *= 2;
            c = *oc;
            let res = x.clone();
            // Depthwise causal convolution (kernel 7), LayerNorm, pointwise MLP, layer scale.
            let kdw = u.dw.len() / c;
            let mut y = vec![0f32; len * c];
            for ti in 0..len {
                for ch in 0..c {
                    let mut s = u.dw_b[ch];
                    for kk in 0..kdw {
                        let src = ti as isize - (kdw - 1 - kk) as isize;
                        if src >= 0 {
                            s += u.dw[ch * kdw + kk] * x[src as usize * c + ch];
                        }
                    }
                    y[ti * c + ch] = s;
                }
            }
            for row in y.chunks_exact_mut(c) {
                let mean = row.iter().sum::<f32>() / c as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
                let inv = 1.0 / (var + 1e-6).sqrt();
                for i in 0..c {
                    row[i] = (row[i] - mean) * inv * u.norm.0[i] + u.norm.1[i];
                }
            }
            let mut mid = vec![0f32; len * u.pw1.rows()];
            u.pw1.mul(&y, len, &mut mid, false);
            mid.iter_mut().for_each(|v| *v = crate::vision::gelu(*v));
            let mut o = vec![0f32; len * c];
            u.pw2.mul(&mid, len, &mut o, false);
            for ti in 0..len {
                for ch in 0..c {
                    x[ti * c + ch] = res[ti * c + ch] + o[ti * c + ch] * u.gamma[ch];
                }
            }
        }
        // 5. The DAC decoder.
        x = conv1d(&x, len, c, &self.dac_entry.w, self.dac_entry.k, 1);
        c = self.dac_entry.w.rows;
        for blk in &self.dac {
            snake(&mut x, c, &blk.snake.0, &blk.snake.1);
            let (w2, oc, k, b) = &blk.up;
            x = conv_transpose(&x, len, w2, *oc, *k, blk.stride, b);
            len *= blk.stride;
            c = *oc;
            for (r, dil) in blk.res.iter().zip([1usize, 3, 9]) {
                let mut h = x.clone();
                snake(&mut h, c, &r.a1.0, &r.a1.1);
                let mut h = conv1d(&h, len, c, &r.conv1.w, r.conv1.k, dil);
                snake(&mut h, c, &r.a2.0, &r.a2.1);
                let h = conv1d(&h, len, c, &r.conv2.w, r.conv2.k, 1);
                x.iter_mut().zip(&h).for_each(|(a, b)| *a += b);
            }
        }
        snake(&mut x, c, &self.post_snake.0, &self.post_snake.1);
        let y = conv1d(&x, len, c, &self.post_conv.w, self.post_conv.k, 1);
        y.into_iter().map(|v| v.clamp(-1.0, 1.0)).collect()
    }
}

/// What a speech synthesis produced.
pub struct Speech {
    pub samples: Vec<f32>,
    pub frames: usize,
}

/// Token ids the talker's prompt is built from.
struct Specials {
    codec_0: u32,
    codec_bos: u32,
    codec_eos: u32,
    codec_pad: u32,
    think: u32,
    think_bos: u32,
    think_eos: u32,
    tts_pad: u32,
    tts_bos: u32,
    tts_eos: u32,
}

/// Speaks `text` with the talker `m` (a qwen3tts model, on a processor session) and `dec`.
/// `lang` is a codec language name ("english", "chinese", ...). Sampling as the model card sets
/// it (temperature 0.9, top-k 50) with a fixed seed, so a sentence sounds the same each time.
pub fn speak(m: &Model, tok: &crate::tokenizer::Tokenizer, dec: &TtsDecoder, text: &str, lang: &str, seed: u64, max_frames: usize) -> Result<Speech, Error> {
    let find = |s: &str| tok.token_id(s).ok_or_else(|| Error::Format(format!("the speech model has no token {s}")));
    let sp = Specials {
        codec_0: find("<|codec_0|>")?,
        codec_bos: find("<|codec_bos|>")?,
        codec_eos: find("<|codec_eos_token|>")?,
        codec_pad: find("<|codec_pad|>")?,
        think: find("<|codec_think|>")?,
        think_bos: find("<|codec_think_bos|>")?,
        think_eos: find("<|codec_think_eos|>")?,
        tts_pad: find("<tts_pad>")?,
        tts_bos: find("<tts_text_bos>")?,
        tts_eos: find("<tts_text_eod>")?,
    };
    let c_lang = find(&format!("<|codec_language_{}|>", lang.to_lowercase())).or_else(|_| find("<|codec_language_english|>"))?;
    let dim = m.cfg.dim;
    let row = |t: u32| {
        let mut v = vec![0f32; dim];
        m.embedding(t, &mut v);
        v
    };
    let sum = |a: u32, b: u32| {
        let (mut x, y) = (row(a), row(b));
        x.iter_mut().zip(&y).for_each(|(p, q)| *p += q);
        x
    };
    // The reference's prompt: the role tokens, the codec preamble over tts padding, then the text
    // over codec padding, the end of text, and the codec start.
    let ids = tok.encode(&format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"));
    if ids.len() < 8 {
        return Err(Error::Request("nothing to say".into()));
    }
    let mut prompt: Vec<Vec<f32>> = ids[..3].iter().map(|&t| row(t)).collect();
    for c in [sp.think, sp.think_bos, c_lang, sp.think_eos] {
        prompt.push(sum(sp.tts_pad, c));
    }
    prompt.push(sum(sp.tts_bos, sp.codec_pad));
    for &t in &ids[3..ids.len() - 5] {
        prompt.push(sum(t, sp.codec_pad));
    }
    prompt.push(sum(sp.tts_eos, sp.codec_pad));
    prompt.push(sum(sp.tts_pad, sp.codec_bos));
    let mut s = Session::new(m, prompt.len() + max_frames + 8);
    for r in &prompt {
        s.step_embedding(m, r)?;
    }
    let overlay = row(sp.tts_pad);
    let mut rng = Rng::new(seed);
    let mut frames = Vec::new();
    // Section times (CEREBRO_TTS_PROF=1): talker, code predictor, audio decoder.
    let prof = std::env::var_os("CEREBRO_TTS_PROF").is_some();
    let (mut t_talk, mut t_pred) = (0f64, 0f64);
    for _ in 0..max_frames {
        let n_out = s.logits.len().min(dec.out_embd.len() / dec.talker_dim);
        let code = sample(&s.logits[..n_out], 0.9, 50, 1.0, 0.05, &mut rng);
        let token = sp.codec_0 + code as u32;
        if token == sp.codec_eos || code >= dec.code_vocab {
            break;
        }
        let t0 = std::time::Instant::now();
        let (codes, embd) = dec.predict_frame(&s.hidden, code, 50, 1.0, &mut rng);
        t_pred += t0.elapsed().as_secs_f64();
        frames.push(codes);
        let next: Vec<f32> = embd.iter().zip(&overlay).map(|(a, b)| a + b).collect();
        let t0 = std::time::Instant::now();
        s.step_embedding(m, &next)?;
        t_talk += t0.elapsed().as_secs_f64();
    }
    let t0 = std::time::Instant::now();
    let samples = dec.decode(&frames);
    if prof {
        eprintln!("tts {} frames: talker {t_talk:.2} s, code predictor {t_pred:.2} s, audio decoder {:.2} s", frames.len(), t0.elapsed().as_secs_f64());
    }
    Ok(Speech { samples, frames: frames.len() })
}

/// 16-bit PCM WAV bytes for mono samples.
pub fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = samples.len() * 2;
    let mut b = Vec::with_capacity(44 + data);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data as u32).to_le_bytes());
    for v in samples {
        b.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_respects_top_k_and_min_p() {
        let logits = vec![5.0, 4.0, -10.0, 3.9, 0.0];
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let i = sample(&logits, 1.0, 2, 1.0, 0.0, &mut rng);
            assert!(i == 0 || i == 1);
        }
        for _ in 0..200 {
            // min-p 0.5 keeps only tokens with at least half the top probability.
            let i = sample(&logits, 1.0, 0, 1.0, 0.5, &mut rng);
            assert!(i == 0, "{i}");
        }
    }

    #[test]
    fn a_causal_convolution_only_looks_back() {
        // One channel in and out, kernel 2: y[t] = 1 * x[t-1] + 10 * x[t].
        let w = Lin { rows: 1, cols: 2, w: vec![1.0, 10.0], bias: vec![0.0] };
        let y = conv1d(&[1.0, 2.0, 3.0], 3, 1, &w, 2, 1);
        assert_eq!(y, vec![10.0, 21.0, 32.0]);
        // Dilation 2: y[t] = x[t-2] + 10 x[t].
        let y = conv1d(&[1.0, 2.0, 3.0], 3, 1, &w, 2, 2);
        assert_eq!(y, vec![10.0, 20.0, 31.0]);
    }

    #[test]
    fn a_transposed_convolution_overlaps_and_adds() {
        // One channel, kernel 3, stride 2: each input spreads [1, 2, 3] times itself.
        let w2 = Lin { rows: 3, cols: 1, w: vec![1.0, 2.0, 3.0], bias: vec![0.0; 3] };
        let y = conv_transpose(&[1.0, 10.0], 2, &w2, 1, 3, 2, &[0.0]);
        // t=0 -> positions 0,1,2; t=1 -> 2,3,(4 dropped).
        assert_eq!(y, vec![1.0, 2.0, 3.0 + 10.0, 20.0]);
    }

    #[test]
    fn wav_bytes_have_a_valid_header() {
        let b = wav_bytes(&[0.0, 0.5, -0.5], 24_000);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(b.len(), 44 + 6);
        let (s, rate) = crate::audio::read_wav(&b).expect("reads back");
        assert_eq!(rate, 24_000);
        assert!((s[1] - 0.5).abs() < 1e-3);
    }
}
