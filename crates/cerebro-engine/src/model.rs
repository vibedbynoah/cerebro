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

pub struct Layer {
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
    pub fn load(path: &Path) -> Result<Self, Error> {
        let g = Arc::new(Gguf::open(path)?);

        let arch = g.str("general.architecture").unwrap_or("").to_owned();
        if !matches!(arch.as_str(), "qwen3" | "qwen3moe" | "qwen2" | "llama") {
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
            qk_norm: g.tensors.contains_key("blk.0.attn_q_norm.weight"),
            n_expert: g.u64(&k("expert_count")).unwrap_or(0) as usize,
            n_used: g.u64(&k("expert_used_count")).unwrap_or(0) as usize,
            expert_ffn: g.u64(&k("expert_feed_forward_length")).unwrap_or(0) as usize,
            arch,
        };
        if cfg.n_head % cfg.n_kv != 0 || head_dim % 2 != 0 {
            return Err(Error::Format("inconsistent head counts".into()));
        }
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let t = |s: &str| g.tensor(&format!("blk.{i}.{s}.weight")).map(Mat::of);
            let nv = |s: &str| vec_f32(&g, &format!("blk.{i}.{s}.weight"));
            layers.push(Layer {
                attn_norm: nv("attn_norm")?,
                q: t("attn_q")?,
                k: t("attn_k")?,
                v: t("attn_v")?,
                o: t("attn_output")?,
                q_norm: if cfg.qk_norm { Some(nv("attn_q_norm")?) } else { None },
                k_norm: if cfg.qk_norm { Some(nv("attn_k_norm")?) } else { None },
                ffn_norm: nv("ffn_norm")?,
                ffn: if cfg.n_expert > 0 {
                    Ffn::Moe(Moe { router: nv("ffn_gate_inp")?, gate: t("ffn_gate_exps")?, up: t("ffn_up_exps")?, down: t("ffn_down_exps")? })
                } else {
                    Ffn::Dense { gate: t("ffn_gate")?, up: t("ffn_up")?, down: t("ffn_down")? }
                },
            });
        }
        let output = g.tensors.get("output.weight").map(Mat::of).unwrap_or_else(|| Mat::of(embed_t));
        let inv_freq = (0..head_dim / 2).map(|i| 1.0 / cfg.rope_base.powf(2.0 * i as f32 / head_dim as f32)).collect();
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
        static DIV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let div = *DIV.get_or_init(|| crate::switches::get(crate::switches::CPU_CHUNK_DIV).max(1) as usize);
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
        pool.run(items.len(), &|ii| {
            let (ji, r0, n) = items[ii];
            let (m, x, qa) = ins[ji];
            let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
            let rb = m.row_bytes;
            for r in r0..r0 + n {
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
fn rope(x: &mut [f32], cs: &[(f32, f32)]) {
    let half = cs.len();
    for (i, &(c, s)) in cs.iter().enumerate() {
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
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
    /// Seconds per phase of CPU decoding: [q/k/v, rope+cache, attention, o, router, ffn/experts, output].
    pub prof: [f64; 7],
    pub logits: Vec<f32>,
    pub stats: Stats,
    /// When set, every step runs on the GPU and the CPU buffers above stay idle.
    #[cfg(feature = "cuda")]
    pub gpu: Option<crate::gpu::GpuState>,
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
        let c = &m.cfg;
        let kv = c.n_kv * c.head_dim;
        Self {
            max_ctx,
            k: (0..c.n_layer).map(|_| vec![0f32; max_ctx * kv]).collect(),
            v: (0..c.n_layer).map(|_| vec![0f32; max_ctx * kv]).collect(),
            tokens: Vec::new(),
            x: vec![0.0; c.dim],
            h: vec![0.0; c.dim],
            q: vec![0.0; c.n_head * c.head_dim],
            kb: vec![0.0; kv],
            vb: vec![0.0; kv],
            att: vec![0.0; c.n_head * c.head_dim],
            g: vec![0.0; c.ffn],
            u: vec![0.0; c.ffn],
            tmp: vec![0.0; c.dim.max(c.n_head * c.head_dim)],
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
            #[cfg(feature = "cuda")]
            gpu: None,
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
        self.tokens.truncate(common);
        let t0 = Instant::now();
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            // Batched prefill on the GPU: every weight read once per chunk of tokens.
            g.prefill(m, &prompt[common..], common, Some(&mut self.logits))?;
            self.tokens.extend_from_slice(&prompt[common..]);
            self.stats = Stats { prompt_tokens: prompt.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
            return Ok(common);
        }
        for &tok in &prompt[common..] {
            self.step(m, tok);
        }
        self.stats = Stats { prompt_tokens: prompt.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
        Ok(common)
    }

    /// Appends one token: runs the full forward pass and writes logits.
    /// A session whose whole forward pass runs on the first CUDA GPU.
    #[cfg(feature = "cuda")]
    pub fn new_gpu(m: &Model, max_ctx: usize) -> Result<Self, Error> {
        let mut s = Self::empty(m, 1);
        s.max_ctx = max_ctx;
        s.gpu = Some(crate::gpu::GpuState::load(m, max_ctx)?);
        Ok(s)
    }

    pub fn step(&mut self, m: &Model, token: u32) {
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            let pos = self.tokens.len();
            if let Err(e) = g.step(m, token, pos, Some(&mut self.logits)) {
                panic!("GPU step failed: {e}");
            }
            self.tokens.push(token);
            return;
        }
        let c = &m.cfg;
        let pos = self.tokens.len();
        let hd = c.head_dim;
        let kvd = c.n_kv * hd;
        let group = c.n_head / c.n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        m.embed_row(token, &mut self.x);
        let cs: Vec<(f32, f32)> = m.inv_freq.iter().map(|f| (pos as f32 * f).sin_cos()).map(|(s, c)| (c, s)).collect();
        for (li, l) in m.layers.iter().enumerate() {
            rmsnorm(&self.x, &l.attn_norm, c.eps, &mut self.h);
            self.qa_dim.quantize(&self.h);
            let mut tp = Instant::now();
            let mut lap = |i: usize, prof: &mut [f64; 7]| {
                let n = Instant::now();
                prof[i] += (n - tp).as_secs_f64();
                tp = n;
            };
            m.matvec_many(&mut [(&l.q, &mut self.q[..]), (&l.k, &mut self.kb[..]), (&l.v, &mut self.vb[..])], &self.h, &self.qa_dim);
            lap(0, &mut self.prof);
            for hh in 0..c.n_head {
                let qh = &mut self.q[hh * hd..(hh + 1) * hd];
                if let Some(w) = &l.q_norm {
                    rmsnorm_inplace(qh, w, c.eps);
                }
                rope(qh, &cs);
            }
            for kh in 0..c.n_kv {
                let kk = &mut self.kb[kh * hd..(kh + 1) * hd];
                if let Some(w) = &l.k_norm {
                    rmsnorm_inplace(kk, w, c.eps);
                }
                rope(kk, &cs);
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
            self.qa_att.quantize(&self.att);
            lap(2, &mut self.prof);
            m.matvec(&l.o, &self.att, &self.qa_att, &mut self.tmp[..c.dim]);
            lap(3, &mut self.prof);
            for i in 0..c.dim {
                self.x[i] += self.tmp[i];
            }
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
        let to = Instant::now();
        rmsnorm_inplace(&mut self.x, &m.out_norm, c.eps);
        self.qa_dim.quantize(&self.x);
        m.matvec(&m.output, &self.x, &self.qa_dim, &mut self.logits);
        self.prof[6] += to.elapsed().as_secs_f64();
        self.tokens.push(token);
    }
}
