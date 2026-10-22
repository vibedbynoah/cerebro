//! Qwen4 experimental (`qwen4exp`, Qwen3.8-Flash-Next): the parts the other architectures lack.
//!
//! - Hyper-connections: the residual is `hc` parallel streams (hc x dim) instead of one. Before each
//!   token mixer and each MoE, a mixer normalises every stream, gates it through a low-rank
//!   projection and averages the streams into the block's input; the block's output is added back
//!   to every stream with its own weight, 2 * sigmoid(inject / hc). The last mixer is the output
//!   norm. (Reference: llama.cpp `src/models/qwen4exp.cpp`, build_hc_mix / build_hc_combine.)
//! - PLE n-gram hash embeddings: one layer adds a gated lookup into a large table indexed by
//!   hashes of the last 2..n tokens, followed by a causal depthwise convolution dilated by n.
//!
//! Both are building blocks other architectures use too (DeepSeek-V4 has hyper-connections); the
//! token mixers (Gated DeltaNet, gated attention) and the experts (routed and shared) are the
//! engine's generic ones in model.rs.

use crate::gguf::{Gguf, Value};
use crate::model::Mat;
use crate::Error;
use crate::quant::{dequant_row, QAct};

/// One hyper-connection mixer.
pub struct Mix {
    /// Per-stream RMSNorm weight over the whole hc x dim layout (the converter folded 1 + w).
    pub norm: Vec<f32>,
    /// hc x dim -> low rank, and back.
    pub down: Mat,
    pub up: Mat,
    /// hc rows of hc x dim: the weights each stream gets the block's output with. Empty for the
    /// output mixer.
    pub inject: Vec<f32>,
}

pub struct Ple {
    pub layer: usize,
    /// emb (heads x head_dim = dim) -> hc x dim key, and -> dim value.
    pub key: Mat,
    pub value: Mat,
    pub norm_key: Vec<f32>,
    pub norm_query: Vec<f32>,
    pub norm_conv: Vec<f32>,
    /// Depthwise taps, channel-major: channel c's taps at [c * kernel .. (c + 1) * kernel], oldest first.
    pub conv: Vec<f32>,
    pub kernel: usize,
    /// The hash table: rows of head_dim values.
    pub table: Mat,
    pub ngram: usize,
    pub per_ngram: usize,
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u64>,
    pub sizes: Vec<u64>,
    pub eos: u32,
    pub image: u32,
}

impl Ple {
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.per_ngram
    }

    /// Positions of history the convolution reads.
    pub fn hist(&self) -> usize {
        (self.kernel - 1) * self.ngram
    }

    /// Table rows for a token given its predecessors (`prev[s - 1]` is s tokens back; None before
    /// the start). An EOS in the window cuts everything at or before it; the token's own EOS does not.
    pub fn rows(&self, tok: u32, prev: &[Option<u32>], out: &mut Vec<usize>) {
        out.clear();
        let mut ctx = vec![tok as u64; self.ngram];
        let mut cut = false;
        for s in 1..self.ngram {
            let t = if cut { None } else { prev.get(s - 1).copied().flatten() };
            cut = cut || t.is_none_or(|t| t == self.eos);
            ctx[s] = if cut { self.eos as u64 } else { t.unwrap_or(self.eos) as u64 };
        }
        for n in 2..=self.ngram {
            let mut mixed = ctx[0].wrapping_mul(self.multipliers[0]);
            for j in 1..n {
                mixed ^= ctx[j].wrapping_mul(self.multipliers[j]);
            }
            let base = (n - 2) * self.per_ngram;
            for g in 0..self.per_ngram {
                let h = base + g;
                out.push((mixed % self.sizes[h] + self.offsets[h]) as usize);
            }
        }
    }
}

pub struct HcModel {
    pub hc: usize,
    pub rank: usize,
    pub attn: Vec<Mix>,
    pub ffn: Vec<Mix>,
    pub head: Mix,
    pub ple: Option<Ple>,
}

fn f32s(g: &Gguf, name: &str) -> Result<Vec<f32>, Error> {
    let t = g.tensor(name)?;
    let mut out = vec![0f32; t.elements() as usize];
    dequant_row(t.dtype, g.bytes(t), &mut out);
    Ok(out)
}

fn u64s(g: &Gguf, key: &str) -> Result<Vec<u64>, Error> {
    let a = g.meta.get(key).and_then(Value::as_array).ok_or_else(|| Error::Format(format!("missing metadata {key}")))?;
    a.iter().map(|v| v.as_u64().ok_or_else(|| Error::Format(format!("{key}: not an integer")))).collect()
}

impl HcModel {
    pub fn load(g: &Gguf, arch: &str, n_layer: usize, dim: usize) -> Result<Self, Error> {
        let k = |s: &str| format!("{arch}.{s}");
        let hc = g.u64(&k("hyper_connection.count"))? as usize;
        let rank = g.u64(&k("hyper_connection.low_rank"))? as usize;
        if hc < 2 || rank == 0 {
            return Err(Error::Format("hyper-connection count must be at least 2".into()));
        }
        let mat = |n: &str| g.tensor(n).map(Mat::of);
        let mix = |p: &str, inject: bool| -> Result<Mix, Error> {
            let m = Mix { norm: f32s(g, &format!("{p}_norm.weight"))?, down: mat(&format!("{p}_down.weight"))?, up: mat(&format!("{p}_up.weight"))?, inject: if inject { f32s(g, &format!("{p}_inject.weight"))? } else { Vec::new() } };
            if m.norm.len() != hc * dim || m.down.cols != hc * dim || m.down.rows != rank || m.up.cols != rank || m.up.rows != hc * dim || (inject && m.inject.len() != hc * hc * dim) {
                return Err(Error::Format(format!("{p}: unexpected hyper-connection shapes")));
            }
            Ok(m)
        };
        let mut attn = Vec::with_capacity(n_layer);
        let mut ffn = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            attn.push(mix(&format!("blk.{i}.hc_attn"), true)?);
            ffn.push(mix(&format!("blk.{i}.hc_ffn"), true)?);
        }
        let head = mix("output_hc", false)?;
        let ple = match g.meta.get(&k("ple.layers")).and_then(Value::as_array) {
            Some(a) if !a.is_empty() => {
                if a.len() != 1 {
                    return Err(Error::Unsupported("more than one PLE layer".into()));
                }
                let layer = a[0].as_u64().ok_or_else(|| Error::Format("ple.layers".into()))? as usize;
                if layer >= n_layer {
                    return Err(Error::Format("PLE layer out of range".into()));
                }
                let ngram = g.u64(&k("ple.ngram_size"))? as usize;
                let per_ngram = g.u64(&k("ple.heads_per_ngram"))? as usize;
                let kernel = g.u64(&k("ple.conv_kernel"))? as usize;
                let eos = g.u64(&k("ple.eos_token_id"))? as u32;
                let image = g.u64(&k("ple.image_token_id")).map(|v| v as u32).unwrap_or(eos);
                let heads = (ngram.max(2) - 1) * per_ngram;
                let multipliers = u64s(g, &k("ple.layer_multipliers"))?;
                let offsets = u64s(g, &k("ple.head_offsets"))?;
                let sizes = u64s(g, &k("ple.head_vocab_sizes"))?;
                if !(2..=8).contains(&ngram) || kernel == 0 || multipliers.len() < ngram || offsets.len() < heads || sizes.len() < heads || sizes.iter().take(heads).any(|&s| s == 0) {
                    return Err(Error::Format("PLE metadata is incomplete".into()));
                }
                let table = mat("per_layer_token_embd.weight")?;
                if table.cols * heads != dim || (0..heads).any(|h| (offsets[h] + sizes[h]) as usize > table.rows) {
                    return Err(Error::Format("PLE table does not match its heads".into()));
                }
                let p = format!("blk.{layer}");
                let raw = f32s(g, &format!("{p}.ple_conv1d.weight"))?;
                if raw.len() != kernel * hc * dim {
                    return Err(Error::Format("PLE convolution has the wrong size".into()));
                }
                Some(Ple {
                    layer,
                    key: mat(&format!("{p}.ple_key.weight"))?,
                    value: mat(&format!("{p}.ple_value.weight"))?,
                    norm_key: f32s(g, &format!("{p}.ple_norm_key.weight"))?,
                    norm_query: f32s(g, &format!("{p}.ple_norm_query.weight"))?,
                    norm_conv: f32s(g, &format!("{p}.ple_norm_conv.weight"))?,
                    conv: raw,
                    kernel,
                    table,
                    ngram,
                    per_ngram,
                    multipliers,
                    offsets,
                    sizes,
                    eos,
                    image,
                })
            }
            _ => None,
        };
        Ok(Self { hc, rank, attn, ffn, head, ple })
    }

    /// Bytes of the mixers and PLE projections every token reads: what the placement keeps closest.
    pub fn dense_bytes(&self) -> u64 {
        let b = |m: &Mat| (m.rows * m.row_bytes) as u64;
        let mixes: u64 = self.attn.iter().chain(&self.ffn).chain(std::iter::once(&self.head)).map(|m| b(&m.down) + b(&m.up)).sum();
        mixes + self.ple.as_ref().map_or(0, |p| b(&p.key) + b(&p.value))
    }
}

/// A session's hyper-connection state and scratch.
pub struct HcState {
    /// The residual streams, hc x dim.
    pub xh: Vec<f32>,
    xn: Vec<f32>,
    gate: Vec<f32>,
    lo: Vec<f32>,
    pub inject: Vec<f32>,
    qa_wide: QAct,
    qa_rank: QAct,
    // PLE: the last `hist` normalised inputs of the convolution, oldest first, and scratch.
    ple_hist: Vec<f32>,
    emb: Vec<f32>,
    qa_emb: QAct,
    pk: Vec<f32>,
    pv: Vec<f32>,
    rows: Vec<usize>,
    /// The token being computed (set before the layers run); u32::MAX for an image embedding.
    pub token: u32,
}

impl HcState {
    pub fn new(h: &HcModel, dim: usize) -> Self {
        let wide = h.hc * dim;
        let hist = h.ple.as_ref().map_or(0, |p| p.hist());
        Self {
            xh: vec![0.0; wide],
            xn: vec![0.0; wide],
            gate: vec![0.0; wide],
            lo: vec![0.0; h.rank],
            inject: vec![0.0; h.hc],
            qa_wide: QAct::new(wide),
            qa_rank: QAct::new(h.rank.next_multiple_of(32)),
            ple_hist: vec![0.0; hist * wide],
            emb: vec![0.0; dim],
            qa_emb: QAct::new(dim),
            pk: vec![0.0; wide],
            pv: vec![0.0; dim],
            rows: Vec::new(),
            token: 0,
        }
    }

    /// The streams start as copies of the token's embedding.
    pub fn start(&mut self, x: &[f32]) {
        for s in self.xh.chunks_exact_mut(x.len()) {
            s.copy_from_slice(x);
        }
    }

    /// RMSNorm of each dim-sized stream of `x` into `out`, times the hc x dim weight `w`.
    fn grouped_norm(x: &[f32], w: &[f32], dim: usize, eps: f32, out: &mut [f32]) {
        for (c, s) in x.chunks_exact(dim).enumerate() {
            let r = 1.0 / (s.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
            let o = &mut out[c * dim..(c + 1) * dim];
            let ww = &w[c * dim..(c + 1) * dim];
            for i in 0..dim {
                o[i] = s[i] * r * ww[i];
            }
        }
    }

    /// The mixer `mx` over the streams: the block's input into `out` (dim), and (unless it is the
    /// output mixer) the streams' injection weights into `self.inject`.
    pub fn mix(&mut self, m: &crate::model::Model, mx: &Mix, hc: usize, eps: f32, out: &mut [f32]) {
        let dim = out.len();
        Self::grouped_norm(&self.xh, &mx.norm, dim, eps, &mut self.xn);
        self.qa_wide.quantize(&self.xn);
        m.matvec_pub(&mx.down, &self.xn, &self.qa_wide, &mut self.lo);
        let s = 1.0 / hc as f32;
        for v in self.lo.iter_mut() {
            let x = *v * s;
            *v = x / (1.0 + (-x).exp());
        }
        self.qa_rank.quantize(&self.lo);
        m.matvec_pub(&mx.up, &self.lo, &self.qa_rank, &mut self.gate);
        out.fill(0.0);
        for c in 0..hc {
            for i in 0..dim {
                let j = c * dim + i;
                out[i] += self.xn[j] / (1.0 + (-self.gate[j]).exp());
            }
        }
        for v in out.iter_mut() {
            *v *= s;
        }
        if !mx.inject.is_empty() {
            let wide = hc * dim;
            for c in 0..hc {
                self.inject[c] = crate::quant::dot_f32(&mx.inject[c * wide..(c + 1) * wide], &self.xn);
            }
        }
    }

    /// Adds the block's output to every stream with its injection weight.
    pub fn combine(&mut self, block: &[f32], hc: usize) {
        let dim = block.len();
        for c in 0..hc {
            let w = 2.0 / (1.0 + (-self.inject[c] / hc as f32).exp());
            let s = &mut self.xh[c * dim..(c + 1) * dim];
            for i in 0..dim {
                s[i] += block[i] * w;
            }
        }
    }

    /// The PLE layer: adds the gated hash embedding and its convolution to the streams.
    /// `prev[s - 1]` is the token s positions back (None before the start).
    pub fn ple(&mut self, m: &crate::model::Model, p: &Ple, hc: usize, dim: usize, eps: f32, prev: &[Option<u32>]) {
        let tok = if self.token == u32::MAX { p.image } else { self.token };
        let mut rows = std::mem::take(&mut self.rows);
        p.rows(tok, prev, &mut rows);
        let hd = p.table.cols;
        let g = m.gguf();
        for (h, &r) in rows.iter().enumerate() {
            let b = g.bytes_at(p.table.offset + r * p.table.row_bytes, p.table.row_bytes);
            dequant_row(p.table.dtype, b, &mut self.emb[h * hd..(h + 1) * hd]);
        }
        self.rows = rows;
        self.qa_emb.quantize(&self.emb);
        m.matvec_many_pub(&mut [(&p.key, &mut self.pk[..]), (&p.value, &mut self.pv[..])], &self.emb, &self.qa_emb);
        let wide = hc * dim;
        // key and query, each grouped-normed; the gate per stream is a signed square root of
        // their scaled dot product, through a sigmoid.
        let mut key = vec![0f32; wide];
        Self::grouped_norm(&self.pk, &p.norm_key, dim, eps, &mut key);
        Self::grouped_norm(&self.xh, &p.norm_query, dim, eps, &mut self.xn);
        let mut gated = vec![0f32; wide];
        for c in 0..hc {
            let s = crate::quant::dot_f32(&key[c * dim..(c + 1) * dim], &self.xn[c * dim..(c + 1) * dim]) / (dim as f32).sqrt();
            let gate = 1.0 / (1.0 + (-(s.signum() * s.abs().max(1e-6).sqrt())).exp());
            for i in 0..dim {
                gated[c * dim + i] = self.pv[i] * gate;
            }
        }
        // Causal depthwise convolution over the normalised gated values, dilated by the n-gram size.
        let mut norm = vec![0f32; wide];
        Self::grouped_norm(&gated, &p.norm_conv, dim, eps, &mut norm);
        let hist = p.hist();
        let kern = p.kernel;
        for ch in 0..wide {
            let w = &p.conv[ch * kern..(ch + 1) * kern];
            // Tap k reads (kern - 1 - k) * ngram positions back; the last tap is this position.
            let mut y = w[kern - 1] * norm[ch];
            for k in 0..kern - 1 {
                let back = (kern - 1 - k) * p.ngram;
                y += w[k] * self.ple_hist[(hist - back) * wide + ch];
            }
            let y = y / (1.0 + (-y).exp());
            self.xh[ch] += gated[ch] + y;
        }
        if hist > 0 {
            self.ple_hist.copy_within(wide.., 0);
            self.ple_hist[(hist - 1) * wide..].copy_from_slice(&norm);
        }
    }

    /// Forgets the convolution history (a new conversation).
    pub fn reset(&mut self) {
        self.ple_hist.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ple(ngram: usize, per: usize) -> Ple {
        let none = Mat { dtype: crate::quant::DType::F32, rows: 0, cols: 0, offset: 0, row_bytes: 0 };
        Ple {
            layer: 0,
            key: none.clone(),
            value: none.clone(),
            norm_key: vec![],
            norm_query: vec![],
            norm_conv: vec![],
            conv: vec![],
            kernel: 4,
            table: none,
            ngram,
            per_ngram: per,
            multipliers: vec![3, 5, 7],
            offsets: vec![0, 100, 200, 300],
            sizes: vec![97, 89, 83, 79],
            eos: 9,
            image: 9,
        }
    }

    #[test]
    fn ngram_rows_hash_the_window_and_eos_cuts_it() {
        let p = ple(3, 2);
        let mut r = Vec::new();
        // token 4, then 2 and 1 positions back: bigram heads hash (4*3 ^ 2*5), trigram (.. ^ 1*7).
        p.rows(4, &[Some(2), Some(1)], &mut r);
        let bi = (4u64 * 3) ^ (2 * 5);
        let tri = bi ^ 7;
        assert_eq!(r, vec![(bi % 97) as usize, (bi % 89 + 100) as usize, (tri % 83 + 200) as usize, (tri % 79 + 300) as usize]);
        // A missing or EOS predecessor reads as EOS from there on.
        p.rows(4, &[Some(9), Some(1)], &mut r);
        let bi = (4u64 * 3) ^ (9 * 5);
        let tri = bi ^ (9 * 7);
        assert_eq!(r[0], (bi % 97) as usize);
        assert_eq!(r[2], (tri % 83 + 200) as usize);
        let mut r2 = Vec::new();
        p.rows(4, &[None, None], &mut r2);
        assert_eq!(r, r2);
    }
}
