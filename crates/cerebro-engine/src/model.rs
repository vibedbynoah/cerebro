//! Transformer graph for the Qwen3 family (and Llama-style dense models): RMSNorm, grouped-query
//! attention with per-head Q/K norms and NeoX RoPE, SwiGLU feed-forward, either dense or a mixture
//! of experts (qwen3moe: softmax router, top-k experts, renormalised weights). CPU backend: every matrix
//! stays quantised; matrix-vector products run in parallel over output rows.
//!
//! Conversation cache: a `Session` keeps the tokens whose keys and values are in its KV cache. A new
//! request reuses the longest common prefix and computes only the tokens after it.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;


use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{dequant_row, dot_k_q8, dot_q8_0_q8, dot_row, DType, QAct};
use crate::Error;

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub n_layer: usize,
    pub dim: usize,
    pub ffn: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub ctx_train: usize,
    pub vocab: usize,
    pub qk_norm: bool,
    /// Mixture of experts: experts per layer, experts used per token, each expert's hidden size.
    /// Zero for dense models.
    pub n_expert: usize,
    pub n_used: usize,
    pub expert_ffn: usize,
    /// Rotary dimensions per head (Qwen3.8 rotates 64 of 256); head_dim when the whole head rotates.
    pub rope_dim: usize,
    /// The query projection also produces a per-head output gate (Qwen3.8 full-attention layers).
    pub attn_gate: bool,
    /// Gated DeltaNet linear-attention layers (Qwen3.8 hybrid); None for pure attention models.
    pub ssm: Option<Ssm>,
}

/// Shape of the Gated DeltaNet layers: `nk` key heads of `dk`, `nv` value heads of `dv`, a causal
/// depthwise convolution of `conv` taps over the q|k|v channels.
#[derive(Clone, Copy, Debug)]
pub struct Ssm {
    pub conv: usize,
    pub nk: usize,
    pub dk: usize,
    pub nv: usize,
    pub dv: usize,
}

impl Ssm {
    /// Channels through the convolution: q and k (nk x dk each) and v (nv x dv).
    pub fn channels(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
}

#[derive(Clone)]
pub struct Mat {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub offset: usize,
    pub row_bytes: usize,
}

impl Mat {
    fn of(t: &TensorInfo) -> Self {
        Self { dtype: t.dtype, rows: t.rows(), cols: t.cols(), offset: t.offset, row_bytes: t.dtype.row_bytes(t.cols()) }
    }

    /// Part `i` of `n` equal row ranges: one expert of a stacked expert tensor.
    pub fn part(&self, i: usize, n: usize) -> Mat {
        let rows = self.rows / n;
        Mat { rows, offset: self.offset + i * rows * self.row_bytes, ..self.clone() }
    }
}

/// The feed-forward block of a layer.
pub enum Ffn {
    Dense { gate: Mat, up: Mat, down: Mat },
    Moe(Moe),
}

/// A mixture-of-experts feed-forward: `n_expert` SwiGLU experts stacked in three tensors, of which the
/// router picks `n_used` per token.
pub struct Moe {
    /// Router weights, n_expert rows of dim.
    pub router: Vec<f32>,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

/// A Gated DeltaNet layer's weights (Qwen3.8 linear attention).
pub struct Delta {
    /// x -> q|k|v (before the convolution).
    pub qkv: Mat,
    /// x -> z, the output gate of the gated RMSNorm.
    pub z: Mat,
    /// x -> per value head decay input (alpha) and write strength (beta).
    pub alpha: Mat,
    pub beta: Mat,
    /// -exp(A_log) per value head, and the time-step bias.
    pub a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Convolution taps, channel-major: channel c's taps at [c * conv .. c * conv + conv], oldest first.
    pub conv: Vec<f32>,
    /// Gated RMSNorm weight, dv values shared by every value head.
    pub norm: Vec<f32>,
    pub out: Mat,
}

pub struct Layer {
    /// Some for a Gated DeltaNet layer: then q/k/v/o are empty and unused.
    pub delta: Option<Delta>,
    pub attn_norm: Vec<f32>,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub o: Mat,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn: Ffn,
}

pub struct Model {
    pub cfg: Config,
    pub path: std::path::PathBuf,
    gguf: Arc<Gguf>,
    embed: Mat,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    output: Mat,
    /// RoPE inverse frequencies, head_dim/2.
    inv_freq: Vec<f32>,
}

fn vec_f32(g: &Gguf, name: &str) -> Result<Vec<f32>, Error> {
    let t = g.tensor(name)?;
    let mut out = vec![0f32; t.elements() as usize];
    dequant_row(t.dtype, g.bytes(t), &mut out);
    Ok(out)
}

impl Model {
    /// Whether any attention or output matrix is Q8_0 (four-row dots only apply to those).
    pub fn uses_q8_0(&self) -> bool {
        self.output.dtype == DType::Q8_0 || self.layers.iter().any(|l| [&l.q, &l.k, &l.v, &l.o].iter().any(|m| m.dtype == DType::Q8_0))
    }

    pub fn load(path: &Path) -> Result<Self, Error> {
        let g = Arc::new(Gguf::open(path)?);

        let arch = g.str("general.architecture").unwrap_or("").to_owned();
        if !matches!(arch.as_str(), "qwen3" | "qwen3moe" | "qwen2" | "llama" | "qwen35") {
            return Err(Error::Unsupported(format!("architecture {arch}")));
        }
        let k = |s: &str| format!("{arch}.{s}");
        let dim = g.u64(&k("embedding_length"))? as usize;
        let n_head = g.u64(&k("attention.head_count"))? as usize;
        let head_dim = g.u64(&k("attention.key_length")).map(|v| v as usize).unwrap_or(dim / n_head);
        let embed_t = g.tensor("token_embd.weight")?;
        let cfg = Config {
            n_layer: g.u64(&k("block_count"))? as usize,
            dim,
            ffn: g.u64(&k("feed_forward_length"))? as usize,
            n_head,
            n_kv: g.u64(&k("attention.head_count_kv")).map(|v| v as usize).unwrap_or(n_head),
            head_dim,
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
            eps: g.f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            ctx_train: g.u64(&k("context_length")).unwrap_or(4096) as usize,
            vocab: embed_t.rows(),
            // Any layer: in Qwen3.5/3.8 layer 0 is a DeltaNet layer, without attention norms.
            qk_norm: g.tensors.keys().any(|k| k.starts_with("blk.") && k.ends_with(".attn_q_norm.weight")),
            n_expert: g.u64(&k("expert_count")).unwrap_or(0) as usize,
            n_used: g.u64(&k("expert_used_count")).unwrap_or(0) as usize,
            expert_ffn: g.u64(&k("expert_feed_forward_length")).unwrap_or(0) as usize,
            rope_dim: g.u64(&k("rope.dimension_count")).map_or(head_dim, |v| v as usize).min(head_dim),
            attn_gate: arch == "qwen35",
            ssm: if g.u64(&k("ssm.state_size")).is_ok() {
                let nv = g.u64(&k("ssm.time_step_rank"))? as usize;
                Some(Ssm {
                    conv: g.u64(&k("ssm.conv_kernel"))? as usize,
                    nk: g.u64(&k("ssm.group_count"))? as usize,
                    dk: g.u64(&k("ssm.state_size"))? as usize,
                    nv,
                    dv: g.u64(&k("ssm.inner_size"))? as usize / nv.max(1),
                })
            } else {
                None
            },
            arch,
        };
        if cfg.n_head % cfg.n_kv != 0 || head_dim % 2 != 0 {
            return Err(Error::Format("inconsistent head counts".into()));
        }
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let none = Mat { dtype: DType::F32, rows: 0, cols: 0, offset: 0, row_bytes: 0 };
        for i in 0..cfg.n_layer {
            let t = |s: &str| g.tensor(&format!("blk.{i}.{s}.weight")).map(Mat::of);
            let nv = |s: &str| vec_f32(&g, &format!("blk.{i}.{s}.weight"));
            let raw = |s: &str| vec_f32(&g, &format!("blk.{i}.{s}"));
            // Qwen3.8 layers are Gated DeltaNet unless they carry full-attention weights.
            let delta = match cfg.ssm {
                Some(ssm) if !g.tensors.contains_key(&format!("blk.{i}.attn_q.weight")) => {
                    let conv = raw("ssm_conv1d.weight")?;
                    if conv.len() != ssm.channels() * ssm.conv {
                        return Err(Error::Format(format!("layer {i}: convolution has {} taps, expected {}", conv.len(), ssm.channels() * ssm.conv)));
                    }
                    Some(Delta {
                        qkv: t("attn_qkv")?,
                        z: t("attn_gate")?,
                        alpha: t("ssm_alpha")?,
                        beta: t("ssm_beta")?,
                        a: raw("ssm_a")?,
                        dt_bias: raw("ssm_dt.bias")?,
                        conv,
                        norm: nv("ssm_norm")?,
                        out: t("ssm_out")?,
                    })
                }
                _ => None,
            };
            let attn = |s: &str| if delta.is_some() { Ok(none.clone()) } else { t(s) };
            let has_qk_norm = delta.is_none() && g.tensors.contains_key(&format!("blk.{i}.attn_q_norm.weight"));
            // Qwen3.8 names the pre-FFN norm post_attention_norm.
            let ffn_norm = if g.tensors.contains_key(&format!("blk.{i}.ffn_norm.weight")) { nv("ffn_norm")? } else { nv("post_attention_norm")? };
            layers.push(Layer {
                attn_norm: nv("attn_norm")?,
                q: attn("attn_q")?,
                k: attn("attn_k")?,
                v: attn("attn_v")?,
                o: attn("attn_output")?,
                q_norm: if has_qk_norm { Some(nv("attn_q_norm")?) } else { None },
                k_norm: if has_qk_norm { Some(nv("attn_k_norm")?) } else { None },
                delta,
                ffn_norm,
                ffn: if cfg.n_expert > 0 {
                    Ffn::Moe(Moe { router: nv("ffn_gate_inp")?, gate: t("ffn_gate_exps")?, up: t("ffn_up_exps")?, down: t("ffn_down_exps")? })
                } else {
                    Ffn::Dense { gate: t("ffn_gate")?, up: t("ffn_up")?, down: t("ffn_down")? }
                },
            });
        }
        let output = g.tensors.get("output.weight").map(Mat::of).unwrap_or_else(|| Mat::of(embed_t));
        let rd = cfg.rope_dim;
        let inv_freq = (0..rd / 2).map(|i| 1.0 / cfg.rope_base.powf(2.0 * i as f32 / rd as f32)).collect();
        Ok(Self { path: path.to_path_buf(), embed: Mat::of(embed_t), out_norm: vec_f32(&g, "output_norm.weight")?, layers, output, inv_freq, cfg, gguf: g })
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn output_mat(&self) -> &Mat {
        &self.output
    }

    pub fn out_norm(&self) -> &[f32] {
        &self.out_norm
    }

    /// RoPE inverse frequencies (rope_dim / 2).
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// True when the output projection reuses the token embedding table.
    pub fn tied(&self) -> bool {
        self.output.offset == self.embed.offset
    }

    pub fn embed_mat(&self) -> &Mat {
        &self.embed
    }

    pub fn embed_into(&self, token: u32, out: &mut [f32]) {
        self.embed_row(token, out);
    }

    fn row<'a>(&'a self, m: &Mat, r: usize) -> &'a [u8] {
        let g: &Gguf = &self.gguf;
        let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
        &all[r * m.row_bytes..(r + 1) * m.row_bytes]
    }

    /// out = M x, parallel over output rows. Integer kernels use `qa`, the activations quantised once
    /// for every matrix that reads them.
    fn matvec(&self, m: &Mat, x: &[f32], qa: &QAct, out: &mut [f32]) {
        self.matvec_multi(&mut [(m, x, qa, out)]);
    }

    /// Several matrix-vector products that read the same input, in one parallel region: one dispatch
    /// instead of one per matrix (dispatch, not arithmetic, dominated small models).
    fn matvec_many(&self, jobs: &mut [(&Mat, &mut [f32])], x: &[f32], qa: &QAct) {
        let mut j: Vec<(&Mat, &[f32], &QAct, &mut [f32])> = jobs.iter_mut().map(|(m, o)| (*m, x, qa, &mut **o)).collect();
        self.matvec_multi(&mut j);
    }

    /// Matrix-vector products with their own inputs (e.g. one per expert), in one parallel region.
    fn matvec_multi(&self, jobs: &mut [(&Mat, &[f32], &QAct, &mut [f32])]) {
        let g: &Gguf = &self.gguf;
        let total: usize = jobs.iter().map(|j| j.0.rows).sum();
        let pool = crate::pool::global();
        // Small items keep the slowest core (an E-core) from holding up the end of each job.
        let div = crate::switches::get(crate::switches::CPU_CHUNK_DIV).max(1) as usize;
        let chunk = (total / (pool.threads() * div)).clamp(4, 512);
        // Each work item is (job index, first row, row count); outputs are disjoint slices.
        let mut items = Vec::with_capacity(total / chunk + jobs.len());
        for (ji, j) in jobs.iter().enumerate() {
            let mut r = 0;
            while r < j.0.rows {
                let n = chunk.min(j.0.rows - r);
                items.push((ji, r, n));
                r += n;
            }
        }
        struct Out(*mut f32);
        // SAFETY: every (job, row range) pair is written by exactly one work item.
        unsafe impl Sync for Out {}
        unsafe impl Send for Out {}
        let outs: Vec<Out> = jobs.iter_mut().map(|j| Out(j.3.as_mut_ptr())).collect();
        let ins: Vec<(&Mat, &[f32], &QAct)> = jobs.iter().map(|j| (j.0, j.1, j.2)).collect();
        let rows4 = crate::switches::enabled(crate::switches::CPU_ROWS4);
        pool.run(items.len(), &|ii| {
            let (ji, r0, n) = items[ii];
            let (m, x, qa) = ins[ji];
            let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
            let rb = m.row_bytes;
            let mut r = r0;
            // Q8_0 decode: four rows per call share the activation loads (bit-identical results).
            if rows4 && m.dtype == DType::Q8_0 {
                while r + 4 <= r0 + n {
                    let row = |k: usize| &all[(r + k) * rb..(r + k + 1) * rb];
                    let Some(v) = crate::quant::dot_q8_0_q8_rows4([row(0), row(1), row(2), row(3)], qa) else { break };
                    for (k, v) in v.into_iter().enumerate() {
                        // SAFETY: r + k < m.rows == the job's output length; ranges never overlap.
                        unsafe { *outs[ji].0.add(r + k) = v };
                    }
                    r += 4;
                }
            }
            for r in r..r0 + n {
                let row = &all[r * rb..(r + 1) * rb];
                let v = dot_any(m.dtype, row, x, qa);
                // SAFETY: r < m.rows == the job's output length; ranges never overlap.
                unsafe { *outs[ji].0.add(r) = v };
            }
        });
    }

    /// CPU experts: out = sum of weight * expert(h) over `experts` (expert index, weight) of layer `li`.
    /// `qa` is `h` quantised. The experts' matrices run in two parallel regions (gate|up, then down).
    pub fn moe_cpu(&self, li: usize, h: &[f32], qa: &QAct, experts: &[(usize, f32)], s: &mut MoeScratch, out: &mut [f32]) {
        out.fill(0.0);
        let Ffn::Moe(e) = &self.layers[li].ffn else { return };
        if experts.is_empty() {
            return;
        }
        let (ne, eff, dim) = (self.cfg.n_expert, self.cfg.expert_ffn, self.cfg.dim);
        let parts: Vec<(Mat, Mat, Mat)> = experts.iter().map(|&(ex, _)| (e.gate.part(ex, ne), e.up.part(ex, ne), e.down.part(ex, ne))).collect();
        {
            let mut jobs: Vec<(&Mat, &[f32], &QAct, &mut [f32])> = Vec::with_capacity(2 * parts.len());
            for ((pg, pu, _), (g, u)) in parts.iter().zip(s.g.chunks_mut(eff).zip(s.u.chunks_mut(eff))) {
                jobs.push((pg, h, qa, g));
                jobs.push((pu, h, qa, u));
            }
            self.matvec_multi(&mut jobs);
        }
        for (k, q) in s.qa.iter_mut().enumerate().take(parts.len()) {
            let (g, u) = (&mut s.g[k * eff..(k + 1) * eff], &s.u[k * eff..(k + 1) * eff]);
            for i in 0..eff {
                g[i] = g[i] / (1.0 + crate::quant::exp_fast(-g[i])) * u[i];
            }
            q.quantize(g);
        }
        {
            let mut jobs: Vec<(&Mat, &[f32], &QAct, &mut [f32])> = Vec::with_capacity(parts.len());
            for (k, ((_, _, pd), o)) in parts.iter().zip(s.eout.chunks_mut(dim)).enumerate() {
                jobs.push((pd, &s.g[k * eff..(k + 1) * eff], &s.qa[k], o));
            }
            self.matvec_multi(&mut jobs);
        }
        for (k, &(_, w)) in experts.iter().enumerate() {
            let o = &s.eout[k * dim..(k + 1) * dim];
            for i in 0..dim {
                out[i] += w * o[i];
            }
        }
    }

    /// CPU experts for a batch of tokens (prompt processing): `groups` lists, per expert of layer
    /// `li`, the (token, weight) pairs routed to it; `h` holds the T normed inputs (T x dim). Adds the
    /// weighted expert outputs into `out` (T x dim). Each weight row is read once and applied to all
    /// of its expert's tokens.
    pub fn moe_cpu_batch(&self, li: usize, groups: &[(usize, Vec<(usize, f32)>)], h: &[f32], out: &mut [f32]) {
        let Ffn::Moe(e) = &self.layers[li].ffn else { return };
        let (ne, eff, dim) = (self.cfg.n_expert, self.cfg.expert_ffn, self.cfg.dim);
        let g: &Gguf = &self.gguf;
        let t_n = h.len() / dim;
        let mut qa: Vec<QAct> = (0..t_n).map(|_| QAct::new(dim)).collect();
        for (t, q) in qa.iter_mut().enumerate() {
            q.quantize(&h[t * dim..(t + 1) * dim]);
        }
        // Pair p = one (expert, token); pairs of a group are contiguous starting at start[gi].
        let mut start = Vec::with_capacity(groups.len());
        let mut pairs: Vec<(usize, f32)> = Vec::new();
        for (_, toks) in groups {
            start.push(pairs.len());
            pairs.extend_from_slice(toks);
        }
        let np = pairs.len();
        if np == 0 {
            return;
        }
        let mut gbuf = vec![0f32; np * eff];
        let mut ubuf = vec![0f32; np * eff];
        let mut ebuf = vec![0f32; np * dim];
        struct P(*mut f32);
        // SAFETY: every (pair, row) element is written by exactly one work item.
        unsafe impl Sync for P {}
        impl P {
            fn at(&self, i: usize) -> *mut f32 {
                self.0.wrapping_add(i)
            }
        }
        let pool = crate::pool::global();
        let chunk = 32;
        // Gate and up rows for every expert's tokens.
        {
            let (gp, up) = (P(gbuf.as_mut_ptr()), P(ubuf.as_mut_ptr()));
            let mut items = Vec::new();
            for gi in 0..groups.len() {
                for r0 in (0..eff).step_by(chunk) {
                    items.push((gi, r0, 0u8));
                    items.push((gi, r0, 1u8));
                }
            }
            pool.run(items.len(), &|ii| {
                let (gi, r0, which) = items[ii];
                let ex = groups[gi].0;
                let m = if which == 0 { e.gate.part(ex, ne) } else { e.up.part(ex, ne) };
                let dst = if which == 0 { &gp } else { &up };
                let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
                for r in r0..(r0 + chunk).min(eff) {
                    let row = &all[r * m.row_bytes..(r + 1) * m.row_bytes];
                    for (k, &(t, _)) in groups[gi].1.iter().enumerate() {
                        let p = start[gi] + k;
                        let v = dot_any(m.dtype, row, &h[t * dim..(t + 1) * dim], &qa[t]);
                        // SAFETY: (p, r) is unique to this item.
                        unsafe { *dst.at(p * eff + r) = v };
                    }
                }
            });
        }
        let mut aq: Vec<QAct> = (0..np).map(|_| QAct::new(eff)).collect();
        for (p, q) in aq.iter_mut().enumerate() {
            let (gr, ur) = (&mut gbuf[p * eff..(p + 1) * eff], &ubuf[p * eff..(p + 1) * eff]);
            for i in 0..eff {
                gr[i] = gr[i] / (1.0 + crate::quant::exp_fast(-gr[i])) * ur[i];
            }
            q.quantize(gr);
        }
        // Down rows.
        {
            let ep = P(ebuf.as_mut_ptr());
            let mut items = Vec::new();
            for gi in 0..groups.len() {
                for r0 in (0..dim).step_by(chunk * 2) {
                    items.push((gi, r0));
                }
            }
            pool.run(items.len(), &|ii| {
                let (gi, r0) = items[ii];
                let m = e.down.part(groups[gi].0, ne);
                let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
                for r in r0..(r0 + chunk * 2).min(dim) {
                    let row = &all[r * m.row_bytes..(r + 1) * m.row_bytes];
                    for k in 0..groups[gi].1.len() {
                        let p = start[gi] + k;
                        let v = dot_any(m.dtype, row, &gbuf[p * eff..(p + 1) * eff], &aq[p]);
                        // SAFETY: (p, r) is unique to this item.
                        unsafe { *ep.at(p * dim + r) = v };
                    }
                }
            });
        }
        for (p, &(t, w)) in pairs.iter().enumerate() {
            let o = &mut out[t * dim..(t + 1) * dim];
            for (a, b) in o.iter_mut().zip(&ebuf[p * dim..(p + 1) * dim]) {
                *a += w * b;
            }
        }
    }

    fn embed_row(&self, token: u32, out: &mut [f32]) {
        dequant_row(self.embed.dtype, self.row(&self.embed, token as usize), out);
    }

    /// out[t * rows + r] = M x_t for T tokens (`xs` token-major, `qas` the same rows quantised). Each
    /// weight row is read once and dotted against every token while it is in L1, so weights cross the
    /// memory bus once per batch instead of once per token.
    fn matmul_batch(&self, m: &Mat, xs: &[f32], qas: &[QAct], out: &mut [f32]) {
        let g: &Gguf = &self.gguf;
        let (cols, rows, rb) = (m.cols, m.rows, m.row_bytes);
        assert!(out.len() >= qas.len() * rows && xs.len() >= qas.len() * cols);
        let pool = crate::pool::global();
        let chunk = (rows / (pool.threads() * 4)).clamp(4, 256);
        let o = SyncPtr(out.as_mut_ptr());
        pool.run(rows.div_ceil(chunk), &|i| {
            let all = g.bytes_at(m.offset, rows * rb);
            for r in i * chunk..((i + 1) * chunk).min(rows) {
                let row = &all[r * rb..(r + 1) * rb];
                let mut t0 = 0;
                if m.dtype == DType::Q8_0 {
                    while t0 + 4 <= qas.len() {
                        let v = crate::quant::dot_q8_0_q8_x4(row, [&qas[t0], &qas[t0 + 1], &qas[t0 + 2], &qas[t0 + 3]]);
                        for (k, v) in v.into_iter().enumerate() {
                            // SAFETY: as below.
                            unsafe { *o.at((t0 + k) * rows + r) = v };
                        }
                        t0 += 4;
                    }
                }
                for (t, qa) in qas.iter().enumerate().skip(t0) {
                    let v = dot_any(m.dtype, row, &xs[t * cols..(t + 1) * cols], qa);
                    // SAFETY: (t, r) is written by exactly this item; out holds qas.len() * rows values.
                    unsafe { *o.at(t * rows + r) = v };
                }
            }
        });
    }
}

/// One weight row times the activations: integer kernels for Q8_0 and the K-quants, float otherwise.
#[inline]
pub(crate) fn dot_any(dt: DType, row: &[u8], x: &[f32], qa: &QAct) -> f32 {
    match dt {
        DType::Q8_0 => dot_q8_0_q8(row, qa),
        DType::Q4K | DType::Q5K | DType::Q6K => dot_k_q8(dt, row, qa),
        _ => dot_row(dt, row, x),
    }
}

/// Scratch for CPU expert computation.
pub struct MoeScratch {
    g: Vec<f32>,
    u: Vec<f32>,
    qa: Vec<QAct>,
    eout: Vec<f32>,
}

impl MoeScratch {
    pub fn new(c: &Config) -> Self {
        Self {
            g: vec![0.0; c.n_used * c.expert_ffn],
            u: vec![0.0; c.n_used * c.expert_ffn],
            qa: (0..c.n_used).map(|_| QAct::new(c.expert_ffn)).collect(),
            eout: vec![0.0; c.n_used * c.dim],
        }
    }
}

/// Positions per CPU attention work item.
const ATT_SPAN: usize = 256;

/// Tokens per batched CPU prefill step: a chunk's activations stay in L2 while each weight row is
/// reused across them.
const CPU_CHUNK: usize = 32;

/// A raw output pointer shared by parallel work items that write disjoint elements.
struct SyncPtr(*mut f32);
// SAFETY: callers guarantee every element is written by at most one item.
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}
impl SyncPtr {
    fn at(&self, i: usize) -> *mut f32 {
        self.0.wrapping_add(i)
    }
}

/// Router: softmax over every expert's score, keep the top `k`, renormalise their weights to sum to 1
/// (Qwen3 MoE). Writes (expert, weight) pairs into `chosen`.
pub(crate) fn route(router: &[f32], h: &[f32], k: usize, scores: &mut [f32], chosen: &mut Vec<(usize, f32)>) {
    let dim = h.len();
    // Scores in parallel (the router is n_expert x dim f32: 1 MB a layer for Qwen3-30B-A3B, which
    // single-threaded took 12 ms of every token).
    struct P(*mut f32);
    // SAFETY: each expert's score is written by one item.
    unsafe impl Sync for P {}
    impl P {
        fn at(&self, i: usize) -> *mut f32 {
            self.0.wrapping_add(i)
        }
    }
    let sp = P(scores.as_mut_ptr());
    let ne = scores.len();
    let per = 8;
    crate::pool::global().run(ne.div_ceil(per), &|i| {
        for e in i * per..((i + 1) * per).min(ne) {
            // SAFETY: e < ne, unique to this item.
            unsafe { *sp.at(e) = crate::quant::dot_f32(&router[e * dim..(e + 1) * dim], h) };
        }
    });
    let mx = scores.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    for s in scores.iter_mut() {
        *s = crate::quant::exp_fast(*s - mx);
    }
    chosen.clear();
    for _ in 0..k {
        let (mut best, mut bv) = (0, f32::NEG_INFINITY);
        for (e, &s) in scores.iter().enumerate() {
            if s > bv && !chosen.iter().any(|&(c, _)| c == e) {
                best = e;
                bv = s;
            }
        }
        chosen.push((best, bv));
    }
    let sum: f32 = chosen.iter().map(|c| c.1).sum();
    for c in chosen.iter_mut() {
        c.1 /= sum;
    }
}

#[inline]
fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * s * w[i];
    }
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        x[i] *= s * w[i];
    }
}

/// NeoX RoPE with the position's (cos, sin) table, computed once per token for every head and layer.
const KV_MAGIC: &[u8; 8] = b"CBKV\x01\0\0\0";

/// Identifies the model a saved cache belongs to: its shape and its file (name and size).
pub fn model_fingerprint(m: &Model) -> u64 {
    let c = &m.cfg;
    let name = m.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for v in [c.n_layer, c.dim, c.n_head, c.n_kv, c.head_dim, c.vocab, c.ffn, m.gguf().len()] {
        mix(v as u64);
    }
    for b in name.bytes() {
        mix(b as u64);
    }
    h
}

fn read_kv_header(m: &Model, r: &mut dyn std::io::Read) -> Result<(usize, usize), Error> {
    let mut head = [0u8; 40];
    r.read_exact(&mut head).map_err(|e| Error::Format(format!("not a saved cache: {e}")))?;
    let num = |i: usize| u64::from_le_bytes(head[8 + i * 8..16 + i * 8].try_into().expect("8 bytes"));
    if &head[..8] != KV_MAGIC || num(0) != model_fingerprint(m) || num(3) != m.cfg.n_layer as u64 || num(2) != (m.cfg.n_kv * m.cfg.head_dim) as u64 {
        return Err(Error::Format("the saved cache belongs to another model".into()));
    }
    Ok((num(1) as usize, num(2) as usize))
}

/// f32 -> IEEE half bits, rounding to nearest even.
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// The file ranges of one layer's weight matrices.
fn layer_ranges(l: &Layer) -> Vec<(usize, usize)> {
    let r = |m: &Mat| (m.offset, m.rows * m.row_bytes);
    let mut v: Vec<(usize, usize)> = match &l.delta {
        Some(d) => [&d.qkv, &d.z, &d.alpha, &d.beta, &d.out].into_iter().map(r).collect(),
        None => [&l.q, &l.k, &l.v, &l.o].into_iter().map(r).collect(),
    };
    match &l.ffn {
        Ffn::Dense { gate, up, down } => v.extend([gate, up, down].into_iter().map(r)),
        Ffn::Moe(e) => v.extend([&e.gate, &e.up, &e.down].into_iter().map(r)),
    }
    v.retain(|x| x.1 > 0);
    v
}

fn rope(x: &mut [f32], cs: &[(f32, f32)]) {
    let half = cs.len();
    for (i, &(c, s)) in cs.iter().enumerate() {
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

/// How far the current prompt read has got, shared with readers on other threads (the Monitor):
/// `base` tokens came from the conversation cache, `pos` are in the cache now, since `started`.
#[derive(Debug, Default)]
pub struct Progress {
    pub base: std::sync::atomic::AtomicUsize,
    pub pos: std::sync::atomic::AtomicUsize,
    started: std::sync::Mutex<Option<Instant>>,
}

impl Progress {
    fn begin(&self, base: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.base.store(base, Relaxed);
        self.pos.store(base, Relaxed);
        *self.started.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    pub fn set(&self, pos: usize) {
        self.pos.store(pos, std::sync::atomic::Ordering::Relaxed);
    }

    /// (tokens in the cache, tokens reused, tokens per second read so far).
    pub fn read(&self) -> (usize, usize, Option<f64>) {
        use std::sync::atomic::Ordering::Relaxed;
        let (pos, base) = (self.pos.load(Relaxed), self.base.load(Relaxed));
        let secs = self.started.lock().unwrap_or_else(|e| e.into_inner()).map(|t| t.elapsed().as_secs_f64());
        (pos, base, secs.filter(|&s| s > 0.05 && pos > base).map(|s| (pos - base) as f64 / s))
    }
}

/// Timing and cache figures for the last `Session::feed` / generation, read by the server and Monitor.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub prefill_ms: f64,
    pub generated: usize,
    pub decode_ms: f64,
}

impl Stats {
    pub fn prefill_tps(&self) -> f64 {
        let n = self.prompt_tokens - self.reused_tokens;
        if self.prefill_ms > 0.0 { n as f64 / (self.prefill_ms / 1000.0) } else { 0.0 }
    }
    pub fn decode_tps(&self) -> f64 {
        if self.decode_ms > 0.0 { self.generated as f64 / (self.decode_ms / 1000.0) } else { 0.0 }
    }
}

/// One conversation's state: KV cache plus the tokens it holds.
pub struct Session {
    pub max_ctx: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub tokens: Vec<u32>,
    // Scratch buffers, reused every step.
    x: Vec<f32>,
    h: Vec<f32>,
    q: Vec<f32>,
    kb: Vec<f32>,
    vb: Vec<f32>,
    att: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    tmp: Vec<f32>,
    qa_dim: QAct,
    qa_att: QAct,
    qa_ffn: QAct,
    /// MoE scratch: router scores, chosen experts and weights, per-expert activations and outputs.
    router: Vec<f32>,
    chosen: Vec<(usize, f32)>,
    moe: MoeScratch,
    /// How often each expert of each layer was chosen (layer * n_expert + expert): the profile
    /// behind expert placement, and Monitor's expert telemetry.
    pub expert_counts: Vec<u32>,
    /// Attention partials: per (head, span) the weighted values, max and sum.
    att_part: Vec<f32>,
    /// Gated DeltaNet state per layer (empty for attention layers): the last conv-1 convolution
    /// inputs, and the recurrent state (nv x dk x dv).
    conv_hist: Vec<Vec<f32>>,
    ssm_state: Vec<Vec<f32>>,
    /// Gated DeltaNet scratch: q|k|v, z, alpha, beta, and the heads' output.
    dmix: Vec<f32>,
    dz: Vec<f32>,
    dalpha: Vec<f32>,
    dbeta: Vec<f32>,
    dout: Vec<f32>,
    qa_dout: QAct,
    /// Query projection output with the per-head gate (Qwen3.8), and the gates.
    qg: Vec<f32>,
    gate: Vec<f32>,
    /// Seconds per phase of CPU decoding: [q/k/v, rope+cache, attention, o, router, ffn/experts, output].
    pub prof: [f64; 7],
    pub logits: Vec<f32>,
    pub stats: Stats,
    /// Live prompt-reading progress (Monitor).
    pub progress: Arc<Progress>,
    /// When set, every step runs on the GPU and the CPU buffers above stay idle.
    #[cfg(feature = "cuda")]
    pub gpu: Option<crate::gpu::GpuState>,
    /// When set, every step runs on the portable (Vulkan / DX12 / Metal) GPU backend.
    #[cfg(feature = "portable")]
    pub portable: Option<crate::portable::PortableState>,
}

impl Session {
    /// A CPU session. The weights are touched into memory up front when they fit in available RAM
    /// (see `Gguf::touch`); `CEREBRO_PREFAULT=0` turns that off.
    pub fn new(m: &Model, max_ctx: usize) -> Self {
        let g = m.gguf();
        if crate::switches::enabled(crate::switches::PREFAULT) {
            // As much of the file as available RAM holds (all of it when it fits).
            let room = (crate::gguf::available_ram() as f64 * 0.9) as usize;
            let t = Instant::now();
            g.touch(&[(0, g.len().min(room))]);
            if std::env::var_os("CEREBRO_PROFILE").is_some() {
                eprintln!("touched {:.1} of {:.1} GB in {:.2}s", g.len().min(room) as f64 / 1e9, g.len() as f64 / 1e9, t.elapsed().as_secs_f64());
            }
        }
        Self::empty(m, max_ctx)
    }

    fn empty(m: &Model, max_ctx: usize) -> Self {
        Self::empty_from(m, max_ctx, 0)
    }

    /// A session whose CPU runs layers `from`.. (a GPU runs the ones before): only those keep a KV
    /// cache or recurrent state here.
    fn empty_from(m: &Model, max_ctx: usize, from: usize) -> Self {
        let c = &m.cfg;
        let kv = c.n_kv * c.head_dim;
        // Only attention layers keep a KV cache; DeltaNet layers keep a fixed-size state instead.
        let mine = |i: usize| i >= from;
        let attn = |i: usize| m.layers[i].delta.is_none();
        let ssm = c.ssm.unwrap_or(Ssm { conv: 1, nk: 0, dk: 0, nv: 0, dv: 0 });
        Self {
            max_ctx,
            k: (0..c.n_layer).map(|i| if mine(i) && attn(i) { vec![0f32; max_ctx * kv] } else { Vec::new() }).collect(),
            v: (0..c.n_layer).map(|i| if mine(i) && attn(i) { vec![0f32; max_ctx * kv] } else { Vec::new() }).collect(),
            conv_hist: (0..c.n_layer).map(|i| if !mine(i) || attn(i) { Vec::new() } else { vec![0f32; (ssm.conv - 1) * ssm.channels()] }).collect(),
            ssm_state: (0..c.n_layer).map(|i| if !mine(i) || attn(i) { Vec::new() } else { vec![0f32; ssm.nv * ssm.dk * ssm.dv] }).collect(),
            dmix: vec![0.0; ssm.channels()],
            dz: vec![0.0; ssm.nv * ssm.dv],
            dalpha: vec![0.0; ssm.nv],
            dbeta: vec![0.0; ssm.nv],
            dout: vec![0.0; ssm.nv * ssm.dv],
            qa_dout: QAct::new((ssm.nv * ssm.dv).max(32)),
            qg: vec![0.0; if c.attn_gate { 2 * c.n_head * c.head_dim } else { 0 }],
            gate: vec![0.0; if c.attn_gate { c.n_head * c.head_dim } else { 0 }],
            tokens: Vec::new(),
            x: vec![0.0; c.dim],
            h: vec![0.0; c.dim],
            q: vec![0.0; c.n_head * c.head_dim],
            kb: vec![0.0; kv],
            vb: vec![0.0; kv],
            att: vec![0.0; c.n_head * c.head_dim],
            g: vec![0.0; c.ffn],
            u: vec![0.0; c.ffn],
            tmp: vec![0.0; c.dim.max(c.n_head * c.head_dim).max(ssm.nv * ssm.dv)],
            qa_dim: QAct::new(c.dim),
            qa_att: QAct::new(c.n_head * c.head_dim),
            qa_ffn: QAct::new(c.ffn),
            router: vec![0.0; c.n_expert],
            chosen: Vec::with_capacity(c.n_used),
            moe: MoeScratch::new(c),
            expert_counts: vec![0; c.n_layer * c.n_expert],
            prof: [0.0; 7],
            att_part: Vec::new(),
            logits: vec![0.0; c.vocab],
            stats: Stats::default(),
            progress: Arc::default(),
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "portable")]
            portable: None,
        }
    }

    /// Makes the cache hold exactly `prompt`, reusing the longest prefix already present
    /// (conversation cache), and leaves the logits for the next token. Returns tokens reused.
    pub fn feed(&mut self, m: &Model, prompt: &[u32]) -> Result<usize, Error> {
        if prompt.is_empty() {
            return Err(Error::Request("empty prompt".into()));
        }
        if prompt.len() >= self.max_ctx {
            return Err(Error::Request(format!("prompt of {} tokens exceeds the {}-token context", prompt.len(), self.max_ctx)));
        }
        let mut common = self.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        // The last prompt token must be computed to produce fresh logits.
        if common == prompt.len() {
            common -= 1;
        }
        // A recurrent (DeltaNet) state holds everything up to the last token and cannot rewind:
        // reuse only an exact continuation, otherwise start over.
        if m.cfg.ssm.is_some() && common < self.tokens.len() {
            common = 0;
            self.reset_recurrent();
            #[cfg(feature = "cuda")]
            if let Some(g) = self.gpu.as_mut() {
                g.reset_recurrent()?;
            }
        }
        self.tokens.truncate(common);
        let t0 = Instant::now();
        self.progress.begin(common);
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.progress = Some(self.progress.clone());
        }
        #[cfg(feature = "cuda")]
        if let Some(from) = self.gpu.as_ref().filter(|g| g.is_split()).map(|g| g.gpu_layers()) {
            // Split: each chunk through the GPU's layers, then the same chunk through the CPU's.
            let mut xs = Vec::new();
            // The CPU's batched path covers plain dense layers; hybrids go token by token.
            let batched = m.cfg.ssm.is_none() && !m.cfg.attn_gate;
            let dim = m.cfg.dim;
            for c in prompt[common..].chunks(crate::gpu::GpuState::SPLIT_CHUNK) {
                let pos = self.tokens.len();
                self.gpu.as_mut().expect("gpu").prefill_split(m, c, pos, &mut xs)?;
                if batched {
                    self.step_batch_from(m, c, Some(std::mem::take(&mut xs)), from);
                    self.progress.set(self.tokens.len());
                } else {
                    for (i, &tok) in c.iter().enumerate() {
                        self.x.copy_from_slice(&xs[i * dim..(i + 1) * dim]);
                        self.layers_from(m, from, pos + i);
                        self.tokens.push(tok);
                        self.progress.set(self.tokens.len());
                    }
                    self.head(m);
                }
            }
            self.stats = Stats { prompt_tokens: prompt.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
            return Ok(common);
        }
        #[cfg(feature = "portable")]
        if let Some(p) = self.portable.as_mut() {
            p.run(m, &prompt[common..], common, &mut self.logits)?;
            self.tokens.extend_from_slice(&prompt[common..]);
            self.progress.set(self.tokens.len());
            self.stats = Stats { prompt_tokens: prompt.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
            return Ok(common);
        }
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            // Batched prefill on the GPU: every weight read once per chunk of tokens.
            g.prefill(m, &prompt[common..], common, Some(&mut self.logits))?;
            self.tokens.extend_from_slice(&prompt[common..]);
            self.stats = Stats { prompt_tokens: prompt.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
            return Ok(common);
        }
        let dense = m.layers.iter().all(|l| matches!(l.ffn, Ffn::Dense { .. })) && m.cfg.ssm.is_none() && !m.cfg.attn_gate;
        if dense && crate::switches::enabled(crate::switches::CPU_BATCH_PREFILL) {
            for c in prompt[common..].chunks(CPU_CHUNK) {
                self.step_batch(m, c);
                self.progress.set(self.tokens.len());
            }
        } else {
            for &tok in &prompt[common..] {
                self.step(m, tok);
                self.progress.set(self.tokens.len());
            }
        }
        self.stats = Stats { prompt_tokens: prompt.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
        Ok(common)
    }

    /// Appends one token: runs the full forward pass and writes logits.
    /// A session whose whole forward pass runs on the first CUDA GPU.
    #[cfg(feature = "cuda")]
    pub fn new_gpu(m: &Model, max_ctx: usize) -> Result<Self, Error> {
        Self::new_gpu_with(m, max_ctx, None)
    }

    /// `new_gpu` with the GPU layer count forced (Some(0) is refused: use `new` for the processor).
    #[cfg(feature = "cuda")]
    pub fn new_gpu_with(m: &Model, max_ctx: usize, force_layers: Option<usize>) -> Result<Self, Error> {
        if !crate::switches::enabled(crate::switches::GPU_ENABLED) {
            return Err(Error::Unsupported("the graphics card is switched off (Use the graphics card)".into()));
        }
        if force_layers == Some(0) {
            return Err(Error::Unsupported("no layers on the graphics card (chosen when loading)".into()));
        }
        let g = crate::gpu::GpuState::load_with(m, max_ctx, force_layers)?;
        if g.is_split() && crate::switches::enabled(crate::switches::PREFAULT) {
            // The CPU's share is read into RAM now (as a CPU session does at start), not by page
            // faults during the first prompt; as much as available RAM holds.
            let mut ranges: Vec<(usize, usize)> = m.layers[g.gpu_layers()..].iter().flat_map(layer_ranges).collect();
            ranges.push((m.output.offset, m.output.rows * m.output.row_bytes));
            ranges.push((m.embed.offset, m.embed.rows * m.embed.row_bytes));
            let mut room = (crate::gguf::available_ram() as f64 * 0.9) as usize;
            ranges.retain_mut(|r| {
                let keep = r.1.min(room);
                room -= keep;
                r.1 = keep;
                keep > 0
            });
            m.gguf().touch(&ranges);
        }
        // A split model keeps the KV cache of the CPU's layers here; a whole-GPU model keeps none.
        let mut s = Self::empty_from(m, max_ctx, g.gpu_layers());
        s.max_ctx = max_ctx;
        s.gpu = Some(g);
        Ok(s)
    }

    /// A session on the portable GPU backend (Vulkan / DX12 / Metal): the whole model on the GPU.
    #[cfg(feature = "portable")]
    pub fn new_portable(m: &Model, max_ctx: usize) -> Result<Self, Error> {
        if !crate::switches::enabled(crate::switches::GPU_ENABLED) {
            return Err(Error::Unsupported("the graphics card is switched off (Use the graphics card)".into()));
        }
        let p = crate::portable::PortableState::load(m, max_ctx)?;
        let mut s = Self::empty_from(m, max_ctx, m.cfg.n_layer);
        s.max_ctx = max_ctx;
        s.portable = Some(p);
        Ok(s)
    }

    /// Appends `toks` (dense models): the same math as `step` for a chunk of tokens, with every
    /// matrix applied to the whole chunk at once (`Model::matmul_batch`). Writes logits for the last.
    fn step_batch(&mut self, m: &Model, toks: &[u32]) {
        self.step_batch_from(m, toks, None, 0);
    }

    /// `step_batch` from layer `from`, on activations `x0` (tokens x width) when a GPU ran the
    /// layers before; embeddings otherwise.
    fn step_batch_from(&mut self, m: &Model, toks: &[u32], x0: Option<Vec<f32>>, from: usize) {
        let c = &m.cfg;
        let (nt, dim, hd, ffn) = (toks.len(), c.dim, c.head_dim, c.ffn);
        let (qd, kvd) = (c.n_head * hd, c.n_kv * hd);
        let group = c.n_head / c.n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos0 = self.tokens.len();
        let embed = x0.is_none();
        let mut x = x0.unwrap_or_else(|| vec![0f32; nt * dim]);
        let mut h = vec![0f32; nt * dim];
        let mut q = vec![0f32; nt * qd];
        let mut kb = vec![0f32; nt * kvd];
        let mut vb = vec![0f32; nt * kvd];
        let mut att = vec![0f32; nt * qd];
        let mut tmp = vec![0f32; nt * dim];
        let mut gt = vec![0f32; nt * ffn];
        let mut ut = vec![0f32; nt * ffn];
        let mut qa_dim: Vec<QAct> = (0..nt).map(|_| QAct::new(dim)).collect();
        let mut qa_att: Vec<QAct> = (0..nt).map(|_| QAct::new(qd)).collect();
        let mut qa_ffn: Vec<QAct> = (0..nt).map(|_| QAct::new(ffn)).collect();
        if embed {
            for (t, &tok) in toks.iter().enumerate() {
                m.embed_row(tok, &mut x[t * dim..(t + 1) * dim]);
            }
        }
        let cs: Vec<Vec<(f32, f32)>> = (0..nt)
            .map(|t| m.inv_freq.iter().map(|f| ((pos0 + t) as f32 * f).sin_cos()).map(|(s, c)| (c, s)).collect())
            .collect();
        let norm_quant = |x: &[f32], w: &[f32], h: &mut [f32], qa: &mut [QAct]| {
            for t in 0..nt {
                rmsnorm(&x[t * dim..(t + 1) * dim], w, c.eps, &mut h[t * dim..(t + 1) * dim]);
                qa[t].quantize(&h[t * dim..(t + 1) * dim]);
            }
        };
        for (li, l) in m.layers.iter().enumerate().skip(from) {
            norm_quant(&x, &l.attn_norm, &mut h, &mut qa_dim);
            m.matmul_batch(&l.q, &h, &qa_dim, &mut q);
            m.matmul_batch(&l.k, &h, &qa_dim, &mut kb);
            m.matmul_batch(&l.v, &h, &qa_dim, &mut vb);
            for t in 0..nt {
                for hh in 0..c.n_head {
                    let qh = &mut q[t * qd + hh * hd..t * qd + (hh + 1) * hd];
                    if let Some(w) = &l.q_norm {
                        rmsnorm_inplace(qh, w, c.eps);
                    }
                    rope(qh, &cs[t]);
                }
                for kh in 0..c.n_kv {
                    let kk = &mut kb[t * kvd + kh * hd..t * kvd + (kh + 1) * hd];
                    if let Some(w) = &l.k_norm {
                        rmsnorm_inplace(kk, w, c.eps);
                    }
                    rope(kk, &cs[t]);
                }
                let p = pos0 + t;
                self.k[li][p * kvd..(p + 1) * kvd].copy_from_slice(&kb[t * kvd..(t + 1) * kvd]);
                self.v[li][p * kvd..(p + 1) * kvd].copy_from_slice(&vb[t * kvd..(t + 1) * kvd]);
            }
            // Causal attention, one work item per (token, query head).
            let (kc, vc, qr) = (&self.k[li], &self.v[li], &q);
            let out = SyncPtr(att.as_mut_ptr());
            crate::pool::global().run(nt * c.n_head, &|item| {
                let (t, hh) = (item / c.n_head, item % c.n_head);
                let kvh = hh / group;
                let len = pos0 + t + 1;
                let qv = &qr[t * qd + hh * hd..t * qd + (hh + 1) * hd];
                thread_local!(static SC: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) });
                SC.with(|sc| {
                    let mut sc = sc.borrow_mut();
                    sc.resize(len, 0.0);
                    let mut mx = f32::NEG_INFINITY;
                    for p in 0..len {
                        let v = crate::quant::dot_f32(qv, &kc[p * kvd + kvh * hd..p * kvd + (kvh + 1) * hd]) * scale;
                        sc[p] = v;
                        mx = mx.max(v);
                    }
                    let mut sum = 0.0;
                    for v in sc.iter_mut() {
                        *v = crate::quant::exp_fast(*v - mx);
                        sum += *v;
                    }
                    // SAFETY: (t, hh) owns this head slice of att.
                    let o = unsafe { std::slice::from_raw_parts_mut(out.at(t * qd + hh * hd), hd) };
                    o.fill(0.0);
                    for p in 0..len {
                        crate::quant::axpy(o, sc[p], &vc[p * kvd + kvh * hd..p * kvd + (kvh + 1) * hd]);
                    }
                    // Normalise last, in the same order as `step`, so both paths give identical results.
                    for v in o.iter_mut() {
                        *v /= sum;
                    }
                });
            });
            for t in 0..nt {
                qa_att[t].quantize(&att[t * qd..(t + 1) * qd]);
            }
            m.matmul_batch(&l.o, &att, &qa_att, &mut tmp);
            for i in 0..nt * dim {
                x[i] += tmp[i];
            }
            norm_quant(&x, &l.ffn_norm, &mut h, &mut qa_dim);
            let Ffn::Dense { gate, up, down } = &l.ffn else { unreachable!("step_batch needs a dense model") };
            m.matmul_batch(gate, &h, &qa_dim, &mut gt);
            m.matmul_batch(up, &h, &qa_dim, &mut ut);
            for i in 0..nt * ffn {
                let g = gt[i];
                gt[i] = g / (1.0 + crate::quant::exp_fast(-g)) * ut[i];
            }
            for t in 0..nt {
                qa_ffn[t].quantize(&gt[t * ffn..(t + 1) * ffn]);
            }
            m.matmul_batch(down, &gt, &qa_ffn, &mut tmp);
            for i in 0..nt * dim {
                x[i] += tmp[i];
            }
        }
        let last = &mut x[(nt - 1) * dim..nt * dim];
        rmsnorm_inplace(last, &m.out_norm, c.eps);
        self.qa_dim.quantize(last);
        m.matvec(&m.output, last, &self.qa_dim, &mut self.logits);
        self.tokens.extend_from_slice(toks);
    }

    pub fn step(&mut self, m: &Model, token: u32) {
        #[cfg(feature = "portable")]
        if let Some(p) = self.portable.as_mut() {
            let pos = self.tokens.len();
            if let Err(e) = p.run(m, &[token], pos, &mut self.logits) {
                panic!("GPU step failed: {e}");
            }
            self.tokens.push(token);
            return;
        }
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            let pos = self.tokens.len();
            if g.is_split() {
                // The GPU's layers, then the CPU's layers and the output head here.
                let from = g.gpu_layers();
                if let Err(e) = g.step_split(m, token, pos, &mut self.x) {
                    panic!("GPU step failed: {e}");
                }
                self.layers_from(m, from, pos);
                self.head(m);
                self.tokens.push(token);
                return;
            }
            if let Err(e) = g.step(m, token, pos, Some(&mut self.logits)) {
                panic!("GPU step failed: {e}");
            }
            self.tokens.push(token);
            return;
        }
        let pos = self.tokens.len();
        m.embed_row(token, &mut self.x);
        self.layers_from(m, 0, pos);
        self.head(m);
        self.tokens.push(token);
    }

    /// Layers `from`.. for the token at `pos`, on the activation in `self.x`.
    fn layers_from(&mut self, m: &Model, from: usize, pos: usize) {
        self.layers_range(m, from, m.cfg.n_layer, pos);
    }

    /// Checking aid: the residual stream after the first `upto` layers for every prompt token, on
    /// the CPU (a fresh session). Compared against the GPU's in the split tests.
    pub fn residual_after(&mut self, m: &Model, prompt: &[u32], upto: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(prompt.len() * m.cfg.dim);
        for (i, &tok) in prompt.iter().enumerate() {
            m.embed_row(tok, &mut self.x);
            self.layers_range(m, 0, upto, i);
            out.extend_from_slice(&self.x);
            self.tokens.push(tok);
        }
        out
    }

    fn layers_range(&mut self, m: &Model, from: usize, to: usize, pos: usize) {
        let c = &m.cfg;
        let hd = c.head_dim;
        let kvd = c.n_kv * hd;
        let group = c.n_head / c.n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let cs: Vec<(f32, f32)> = m.inv_freq.iter().map(|f| (pos as f32 * f).sin_cos()).map(|(s, c)| (c, s)).collect();
        for (li, l) in m.layers.iter().enumerate().take(to).skip(from) {
            rmsnorm(&self.x, &l.attn_norm, c.eps, &mut self.h);
            self.qa_dim.quantize(&self.h);
            if let Some(d) = &l.delta {
                self.delta_step(m, li, d);
                for i in 0..c.dim {
                    self.x[i] += self.tmp[i];
                }
                self.ffn_step(m, li, l);
                continue;
            }
            let mut tp = Instant::now();
            let mut lap = |i: usize, prof: &mut [f64; 7]| {
                let n = Instant::now();
                prof[i] += (n - tp).as_secs_f64();
                tp = n;
            };
            if c.attn_gate {
                // Each head's projection is [query | gate] (Qwen3.8).
                m.matvec_many(&mut [(&l.q, &mut self.qg[..]), (&l.k, &mut self.kb[..]), (&l.v, &mut self.vb[..])], &self.h, &self.qa_dim);
                for hh in 0..c.n_head {
                    self.q[hh * hd..(hh + 1) * hd].copy_from_slice(&self.qg[2 * hh * hd..(2 * hh + 1) * hd]);
                    self.gate[hh * hd..(hh + 1) * hd].copy_from_slice(&self.qg[(2 * hh + 1) * hd..(2 * hh + 2) * hd]);
                }
            } else {
                m.matvec_many(&mut [(&l.q, &mut self.q[..]), (&l.k, &mut self.kb[..]), (&l.v, &mut self.vb[..])], &self.h, &self.qa_dim);
            }
            lap(0, &mut self.prof);
            let rd = c.rope_dim;
            for hh in 0..c.n_head {
                let qh = &mut self.q[hh * hd..(hh + 1) * hd];
                if let Some(w) = &l.q_norm {
                    rmsnorm_inplace(qh, w, c.eps);
                }
                rope(&mut qh[..rd], &cs);
            }
            for kh in 0..c.n_kv {
                let kk = &mut self.kb[kh * hd..(kh + 1) * hd];
                if let Some(w) = &l.k_norm {
                    rmsnorm_inplace(kk, w, c.eps);
                }
                rope(&mut kk[..rd], &cs);
            }
            self.k[li][pos * kvd..(pos + 1) * kvd].copy_from_slice(&self.kb);
            self.v[li][pos * kvd..(pos + 1) * kvd].copy_from_slice(&self.vb);
            // Attention, one work item per (KV head, span of up to 256 positions): every query head
            // of the group is computed from one read of each key and value (GQA shares them), and
            // long contexts split across threads; partial softmaxes are merged per head after.
            let (kc, vc, q) = (&self.k[li], &self.v[li], &self.q);
            let len = pos + 1;
            let spans = len.div_ceil(ATT_SPAN);
            let part = hd + 2;
            struct Ptr(*mut f32);
            // SAFETY: each item writes only its own (head, span) partial.
            unsafe impl Sync for Ptr {}
            impl Ptr {
                fn at(&self, i: usize) -> *mut f32 {
                    self.0.wrapping_add(i)
                }
            }
            if self.att_part.len() < c.n_head * spans * part {
                self.att_part.resize(c.n_head * spans * part, 0.0);
            }
            let parts = Ptr(self.att_part.as_mut_ptr());
            // Split each group's heads across items too when there are fewer (KV head, span) pairs
            // than threads (short contexts would otherwise use n_kv threads).
            let hsplit = (crate::pool::global().threads() / (c.n_kv * spans)).clamp(1, group);
            let hsplit = (1..=hsplit).rev().find(|d| group % d == 0).unwrap_or(1);
            let gper = group / hsplit;
            lap(1, &mut self.prof);
            crate::pool::global().run(c.n_kv * spans * hsplit, &|item| {
                let (hs, item) = (item % hsplit, item / hsplit);
                let (kvh, sp) = (item / spans, item % spans);
                let gr = hs * gper..(hs + 1) * gper;
                let (t0, t1) = (sp * ATT_SPAN, ((sp + 1) * ATT_SPAN).min(len));
                let n = t1 - t0;
                thread_local!(static SC: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) });
                SC.with(|sc| {
                    let mut sc = sc.borrow_mut();
                    sc.resize(group * ATT_SPAN, 0.0);
                    // Scores for every head of the group, key-major so each key is read once.
                    for t in t0..t1 {
                        let kt = &kc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd];
                        for g in gr.clone() {
                            let hh = kvh * group + g;
                            sc[g * ATT_SPAN + (t - t0)] = crate::quant::dot_f32(&q[hh * hd..(hh + 1) * hd], kt) * scale;
                        }
                    }
                    for g in gr.clone() {
                        let row = &mut sc[g * ATT_SPAN..g * ATT_SPAN + n];
                        let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                        let mut l = 0.0;
                        for v in row.iter_mut() {
                            *v = crate::quant::exp_fast(*v - mx);
                            l += *v;
                        }
                        let hh = kvh * group + g;
                        // SAFETY: (hh, sp) is unique to this item; the buffer holds n_head * spans parts.
                        let o = unsafe { std::slice::from_raw_parts_mut(parts.at((hh * spans + sp) * part), part) };
                        o[hd] = mx;
                        o[hd + 1] = l;
                        o[..hd].fill(0.0);
                    }
                    for t in t0..t1 {
                        let vt = &vc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd];
                        for g in gr.clone() {
                            let hh = kvh * group + g;
                            let p = sc[g * ATT_SPAN + (t - t0)];
                            // SAFETY: as above.
                            let o = unsafe { std::slice::from_raw_parts_mut(parts.at((hh * spans + sp) * part), hd) };
                            crate::quant::axpy(o, p, vt);
                        }
                    }
                });
            });
            for hh in 0..c.n_head {
                let ps = &self.att_part[hh * spans * part..(hh + 1) * spans * part];
                let m_all = (0..spans).fold(f32::NEG_INFINITY, |m, s| m.max(ps[s * part + hd]));
                let out = &mut self.att[hh * hd..(hh + 1) * hd];
                out.fill(0.0);
                let mut l_all = 0.0;
                for s in 0..spans {
                    let w = crate::quant::exp_fast(ps[s * part + hd] - m_all);
                    l_all += w * ps[s * part + hd + 1];
                    for i in 0..hd {
                        out[i] += w * ps[s * part + i];
                    }
                }
                for v in out.iter_mut() {
                    *v /= l_all;
                }
            }
            if c.attn_gate {
                for (a, g) in self.att.iter_mut().zip(&self.gate) {
                    *a *= 1.0 / (1.0 + crate::quant::exp_fast(-g));
                }
            }
            self.qa_att.quantize(&self.att);
            lap(2, &mut self.prof);
            m.matvec(&l.o, &self.att, &self.qa_att, &mut self.tmp[..c.dim]);
            lap(3, &mut self.prof);
            for i in 0..c.dim {
                self.x[i] += self.tmp[i];
            }
            self.ffn_step(m, li, l);
        }
    }

    /// The output head: final norm and logits.
    fn head(&mut self, m: &Model) {
        let to = Instant::now();
        rmsnorm_inplace(&mut self.x, &m.out_norm, m.cfg.eps);
        self.qa_dim.quantize(&self.x);
        m.matvec(&m.output, &self.x, &self.qa_dim, &mut self.logits);
        self.prof[6] += to.elapsed().as_secs_f64();
    }

    /// Writes the cache (tokens, every attention layer's K and V as f16, every DeltaNet layer's
    /// state) so a later session on the same model can continue without re-reading the prompt.
    pub fn save_kv(&self, m: &Model, w: &mut dyn std::io::Write) -> Result<(), Error> {
        let c = &m.cfg;
        let n = self.tokens.len();
        let kvd = c.n_kv * c.head_dim;
        let io = |e: std::io::Error| Error::Request(format!("could not write the cache: {e}"));
        #[cfg(feature = "portable")]
        if self.portable.is_some() {
            return Err(Error::Unsupported("saving the cache is not available on the portable GPU backend yet".into()));
        }
        w.write_all(KV_MAGIC).map_err(io)?;
        for v in [model_fingerprint(m), n as u64, kvd as u64, c.n_layer as u64] {
            w.write_all(&v.to_le_bytes()).map_err(io)?;
        }
        let bytes32 = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let bytes16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        w.write_all(&self.tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()).map_err(io)?;
        #[cfg(feature = "cuda")]
        let gpu_layers = self.gpu.as_ref().map_or(0, |g| g.gpu_layers());
        #[cfg(not(feature = "cuda"))]
        let gpu_layers = 0;
        for li in 0..c.n_layer {
            let attn = m.layers[li].delta.is_none();
            if li < gpu_layers {
                #[cfg(feature = "cuda")]
                {
                    let g = self.gpu.as_ref().expect("gpu");
                    if attn {
                        let (k, v) = g.kv_get(li, n, kvd)?.ok_or_else(|| Error::Format("layer kind mismatch".into()))?;
                        w.write_all(&[0]).map_err(io)?;
                        w.write_all(&bytes16(&k)).map_err(io)?;
                        w.write_all(&bytes16(&v)).map_err(io)?;
                    } else {
                        let (h, st) = g.rec_get(li)?.ok_or_else(|| Error::Format("layer kind mismatch".into()))?;
                        w.write_all(&[1]).map_err(io)?;
                        w.write_all(&(h.len() as u64).to_le_bytes()).map_err(io)?;
                        w.write_all(&bytes32(&h)).map_err(io)?;
                        w.write_all(&(st.len() as u64).to_le_bytes()).map_err(io)?;
                        w.write_all(&bytes32(&st)).map_err(io)?;
                    }
                }
            } else if attn {
                let k: Vec<u16> = self.k[li][..n * kvd].iter().map(|&x| f32_to_f16(x)).collect();
                let v: Vec<u16> = self.v[li][..n * kvd].iter().map(|&x| f32_to_f16(x)).collect();
                w.write_all(&[0]).map_err(io)?;
                w.write_all(&bytes16(&k)).map_err(io)?;
                w.write_all(&bytes16(&v)).map_err(io)?;
            } else {
                w.write_all(&[1]).map_err(io)?;
                w.write_all(&(self.conv_hist[li].len() as u64).to_le_bytes()).map_err(io)?;
                w.write_all(&bytes32(&self.conv_hist[li])).map_err(io)?;
                w.write_all(&(self.ssm_state[li].len() as u64).to_le_bytes()).map_err(io)?;
                w.write_all(&bytes32(&self.ssm_state[li])).map_err(io)?;
            }
        }
        Ok(())
    }

    /// The tokens a saved cache holds, read from its header only (to pick the best match cheaply).
    pub fn peek_kv(m: &Model, r: &mut dyn std::io::Read) -> Result<Vec<u32>, Error> {
        let (n, _) = read_kv_header(m, r)?;
        let mut buf = vec![0u8; n * 4];
        r.read_exact(&mut buf).map_err(|e| Error::Format(format!("saved cache cut short: {e}")))?;
        Ok(buf.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    /// Restores a cache written by `save_kv` for this model; returns the tokens it holds. The next
    /// `feed` continues from them.
    pub fn load_kv(&mut self, m: &Model, r: &mut dyn std::io::Read) -> Result<usize, Error> {
        let c = &m.cfg;
        let (n, kvd) = read_kv_header(m, r)?;
        if n == 0 || n >= self.max_ctx {
            return Err(Error::Format("the saved cache does not fit this context".into()));
        }
        let rd = |r: &mut dyn std::io::Read, len: usize| -> Result<Vec<u8>, Error> {
            let mut b = vec![0u8; len];
            r.read_exact(&mut b).map_err(|e| Error::Format(format!("saved cache cut short: {e}")))?;
            Ok(b)
        };
        let u16s = |b: Vec<u8>| b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect::<Vec<u16>>();
        let f32s = |b: Vec<u8>| b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect::<Vec<f32>>();
        let tokens: Vec<u32> = rd(r, n * 4)?.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        #[cfg(feature = "cuda")]
        let gpu_layers = self.gpu.as_ref().map_or(0, |g| g.gpu_layers());
        #[cfg(not(feature = "cuda"))]
        let gpu_layers = 0;
        for li in 0..c.n_layer {
            let tag = rd(r, 1)?[0];
            let attn = m.layers[li].delta.is_none();
            if (tag == 0) != attn {
                return Err(Error::Format("the saved cache does not match this model's layers".into()));
            }
            if attn {
                let k = u16s(rd(r, n * kvd * 2)?);
                let v = u16s(rd(r, n * kvd * 2)?);
                if li < gpu_layers {
                    #[cfg(feature = "cuda")]
                    self.gpu.as_mut().expect("gpu").kv_set(li, &k, &v)?;
                } else {
                    for (dst, src) in self.k[li][..n * kvd].iter_mut().zip(&k) {
                        *dst = crate::quant::f16_to_f32(*src);
                    }
                    for (dst, src) in self.v[li][..n * kvd].iter_mut().zip(&v) {
                        *dst = crate::quant::f16_to_f32(*src);
                    }
                }
            } else {
                let hl = u64::from_le_bytes(rd(r, 8)?.try_into().expect("8 bytes")) as usize;
                let h = f32s(rd(r, hl * 4)?);
                let sl = u64::from_le_bytes(rd(r, 8)?.try_into().expect("8 bytes")) as usize;
                let st = f32s(rd(r, sl * 4)?);
                if li < gpu_layers {
                    #[cfg(feature = "cuda")]
                    self.gpu.as_mut().expect("gpu").rec_set(li, &h, &st)?;
                } else {
                    if h.len() != self.conv_hist[li].len() || st.len() != self.ssm_state[li].len() {
                        return Err(Error::Format("saved state does not fit this model".into()));
                    }
                    self.conv_hist[li].copy_from_slice(&h);
                    self.ssm_state[li].copy_from_slice(&st);
                }
            }
        }
        self.tokens = tokens;
        Ok(n)
    }

    /// Clears every DeltaNet state (a new sequence).
    fn reset_recurrent(&mut self) {
        for v in self.conv_hist.iter_mut().chain(self.ssm_state.iter_mut()) {
            v.fill(0.0);
        }
    }

    /// The feed-forward half of a layer: norm, SwiGLU or experts, residual add.
    fn ffn_step(&mut self, m: &Model, li: usize, l: &Layer) {
        let c = &m.cfg;
        let mut tp = Instant::now();
        let mut lap = |i: usize, prof: &mut [f64; 7]| {
            let n = Instant::now();
            prof[i] += (n - tp).as_secs_f64();
            tp = n;
        };
        {
            rmsnorm(&self.x, &l.ffn_norm, c.eps, &mut self.h);
            self.qa_dim.quantize(&self.h);
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    m.matvec_many(&mut [(gate, &mut self.g[..c.ffn]), (up, &mut self.u[..c.ffn])], &self.h, &self.qa_dim);
                    for i in 0..c.ffn {
                        let gv = self.g[i];
                        self.g[i] = gv / (1.0 + crate::quant::exp_fast(-gv)) * self.u[i];
                    }
                    self.qa_ffn.quantize(&self.g[..c.ffn]);
                    m.matvec(down, &self.g[..c.ffn], &self.qa_ffn, &mut self.tmp[..c.dim]);
                    for i in 0..c.dim {
                        self.x[i] += self.tmp[i];
                    }
                }
                Ffn::Moe(e) => {
                    route(&e.router, &self.h, c.n_used, &mut self.router, &mut self.chosen);
                    lap(4, &mut self.prof);
                    for &(ex, _) in &self.chosen {
                        self.expert_counts[li * c.n_expert + ex] += 1;
                    }
                    m.moe_cpu(li, &self.h, &self.qa_dim, &self.chosen, &mut self.moe, &mut self.tmp[..c.dim]);
                    for i in 0..c.dim {
                        self.x[i] += self.tmp[i];
                    }
                }
            }
            lap(5, &mut self.prof);
        }
    }

    /// One token through a Gated DeltaNet layer (Qwen3.8 / Qwen3-Next linear attention), from the
    /// normed input in `h` / `qa_dim` to the layer output in `tmp[..dim]`. Per value head, with
    /// L2-normalised q and k from the causal convolution:
    ///   S <- S * exp(a * softplus(alpha + dt_bias));  S <- S + k (beta * (v - S^T k))^T;  o = S^T q
    /// then o <- rmsnorm(o) * w * silu(z), and the output projection.
    fn delta_step(&mut self, m: &Model, li: usize, d: &Delta) {
        let c = &m.cfg;
        let Some(ss) = c.ssm else { return };
        let (nk, dk, nv, dv, kc) = (ss.nk, ss.dk, ss.nv, ss.dv, ss.conv);
        let ch = ss.channels();
        m.matvec_many(
            &mut [(&d.qkv, &mut self.dmix[..]), (&d.z, &mut self.dz[..]), (&d.alpha, &mut self.dalpha[..]), (&d.beta, &mut self.dbeta[..])],
            &self.h,
            &self.qa_dim,
        );
        // Causal depthwise convolution over the last `kc` inputs, then SiLU. hist holds the kc-1
        // previous inputs, oldest first.
        let hist = &mut self.conv_hist[li];
        for ci in 0..ch {
            let w = &d.conv[ci * kc..(ci + 1) * kc];
            let x = self.dmix[ci];
            let mut y = w[kc - 1] * x;
            for j in 0..kc - 1 {
                y += w[j] * hist[j * ch + ci];
            }
            for j in 0..kc.saturating_sub(2) {
                hist[j * ch + ci] = hist[(j + 1) * ch + ci];
            }
            if kc > 1 {
                hist[(kc - 2) * ch + ci] = x;
            }
            self.dmix[ci] = y / (1.0 + crate::quant::exp_fast(-y));
        }
        // L2-normalise each q and k head; q also takes the 1/sqrt(dk) scale.
        let qscale = 1.0 / (dk as f32).sqrt();
        for hh in 0..2 * nk {
            let v = &mut self.dmix[hh * dk..(hh + 1) * dk];
            let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
            let s = if hh < nk { inv * qscale } else { inv };
            v.iter_mut().for_each(|x| *x *= s);
        }
        let (mix, z, alpha, beta) = (&self.dmix, &self.dz, &self.dalpha, &self.dbeta);
        struct Ptr(*mut f32);
        // SAFETY: each value head writes only its own state block and output slice.
        unsafe impl Sync for Ptr {}
        impl Ptr {
            fn at(&self, i: usize) -> *mut f32 {
                self.0.wrapping_add(i)
            }
        }
        let state = Ptr(self.ssm_state[li].as_mut_ptr());
        let out = Ptr(self.dout.as_mut_ptr());
        crate::pool::global().run(nv, &|h| {
            // With more value heads than key heads, value head h reads key head h % nk (the GGUF
            // orders value heads so the key heads tile). Verified on Qwen3.5-4B (32 value / 16 key
            // heads): identical greedy text to the reference over 48 tokens; h / ratio is garbage.
            let kh = h % nk;
            let q = &mix[kh * dk..(kh + 1) * dk];
            let k = &mix[(nk + kh) * dk..(nk + kh + 1) * dk];
            let v = &mix[2 * nk * dk + h * dv..2 * nk * dk + (h + 1) * dv];
            let x = alpha[h] + d.dt_bias[h];
            let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            let decay = (d.a[h] * softplus).exp();
            let b = 1.0 / (1.0 + (-beta[h]).exp());
            // SAFETY: disjoint per head (see Ptr).
            let s = unsafe { std::slice::from_raw_parts_mut(state.at(h * dk * dv), dk * dv) };
            let o = unsafe { std::slice::from_raw_parts_mut(out.at(h * dv), dv) };
            // S is dk rows of dv. kv = S^T k after the decay; delta = beta (v - kv).
            let mut kv = vec![0f32; dv];
            for i in 0..dk {
                let row = &mut s[i * dv..(i + 1) * dv];
                for j in 0..dv {
                    row[j] *= decay;
                    kv[j] += row[j] * k[i];
                }
            }
            for j in 0..dv {
                kv[j] = (v[j] - kv[j]) * b;
            }
            o.fill(0.0);
            for i in 0..dk {
                let row = &mut s[i * dv..(i + 1) * dv];
                for j in 0..dv {
                    row[j] += k[i] * kv[j];
                    o[j] += row[j] * q[i];
                }
            }
            // Gated RMSNorm: rmsnorm(o) * w * silu(z).
            let zz = &z[h * dv..(h + 1) * dv];
            let ms = o.iter().map(|x| x * x).sum::<f32>() / dv as f32;
            let r = 1.0 / (ms + c.eps).sqrt();
            for j in 0..dv {
                let zj = zz[j];
                o[j] = o[j] * r * d.norm[j] * (zj / (1.0 + (-zj).exp()));
            }
        });
        self.qa_dout.quantize(&self.dout);
        m.matvec(&d.out, &self.dout, &self.qa_dout, &mut self.tmp[..c.dim]);
    }
}
