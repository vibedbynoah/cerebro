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
        let div = *DIV.get_or_init(|| std::env::var("CEREBRO_CHUNK_DIV").ok().and_then(|v| v.parse().ok()).unwrap_or(4));
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
                g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
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

/// Router: softmax over every expert's score, keep the top `k`, renormalise their weights to sum to 1
/// (Qwen3 MoE). Writes (expert, weight) pairs into `chosen`.
pub(crate) fn route(router: &[f32], h: &[f32], k: usize, scores: &mut [f32], chosen: &mut Vec<(usize, f32)>) {
    let dim = h.len();
    let mut mx = f32::NEG_INFINITY;
    for (e, s) in scores.iter_mut().enumerate() {
        *s = router[e * dim..(e + 1) * dim].iter().zip(h).map(|(a, b)| a * b).sum();
        mx = mx.max(*s);
    }
    for s in scores.iter_mut() {
        *s = (*s - mx).exp();
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

fn rope(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = inv_freq.len();
    for i in 0..half {
        let a = pos as f32 * inv_freq[i];
        let (s, c) = a.sin_cos();
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
        if std::env::var("CEREBRO_PREFAULT").as_deref() != Ok("0") && (g.len() as f64) < crate::gguf::available_ram() as f64 * 0.9 {
            g.touch(&[(0, g.len())]);
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
        for (li, l) in m.layers.iter().enumerate() {
            rmsnorm(&self.x, &l.attn_norm, c.eps, &mut self.h);
            self.qa_dim.quantize(&self.h);
            m.matvec_many(&mut [(&l.q, &mut self.q[..]), (&l.k, &mut self.kb[..]), (&l.v, &mut self.vb[..])], &self.h, &self.qa_dim);
            for hh in 0..c.n_head {
                let qh = &mut self.q[hh * hd..(hh + 1) * hd];
                if let Some(w) = &l.q_norm {
                    rmsnorm_inplace(qh, w, c.eps);
                }
                rope(qh, pos, &m.inv_freq);
            }
            for kh in 0..c.n_kv {
                let kk = &mut self.kb[kh * hd..(kh + 1) * hd];
                if let Some(w) = &l.k_norm {
                    rmsnorm_inplace(kk, w, c.eps);
                }
                rope(kk, pos, &m.inv_freq);
            }
            self.k[li][pos * kvd..(pos + 1) * kvd].copy_from_slice(&self.kb);
            self.v[li][pos * kvd..(pos + 1) * kvd].copy_from_slice(&self.vb);
            let (kc, vc, q) = (&self.k[li], &self.v[li], &self.q);
            struct Ptr(*mut f32);
            // SAFETY: each head writes only its own hd-long slice of att.
            unsafe impl Sync for Ptr {}
            impl Ptr {
                fn at(&self, i: usize) -> *mut f32 {
                    self.0.wrapping_add(i)
                }
            }
            let att = Ptr(self.att.as_mut_ptr());
            crate::pool::global().run(c.n_head, &|hh| {
                let out = unsafe { std::slice::from_raw_parts_mut(att.at(hh * hd), hd) };
                let qh = &q[hh * hd..(hh + 1) * hd];
                let kvh = hh / group;
                let mut scores = vec![0f32; pos + 1];
                let mut mx = f32::NEG_INFINITY;
                for (t, s) in scores.iter_mut().enumerate() {
                    let kt = &kc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd];
                    *s = qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f32>() * scale;
                    mx = mx.max(*s);
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                out.fill(0.0);
                for (t, s) in scores.iter().enumerate() {
                    let p = s / sum;
                    let vt = &vc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd];
                    for i in 0..hd {
                        out[i] += p * vt[i];
                    }
                }
            });
            self.qa_att.quantize(&self.att);
            m.matvec(&l.o, &self.att, &self.qa_att, &mut self.tmp[..c.dim]);
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
                        self.g[i] = gv / (1.0 + (-gv).exp()) * self.u[i];
                    }
                    self.qa_ffn.quantize(&self.g[..c.ffn]);
                    m.matvec(down, &self.g[..c.ffn], &self.qa_ffn, &mut self.tmp[..c.dim]);
                    for i in 0..c.dim {
                        self.x[i] += self.tmp[i];
                    }
                }
                Ffn::Moe(e) => {
                    route(&e.router, &self.h, c.n_used, &mut self.router, &mut self.chosen);
                    for &(ex, _) in &self.chosen {
                        self.expert_counts[li * c.n_expert + ex] += 1;
                    }
                    m.moe_cpu(li, &self.h, &self.qa_dim, &self.chosen, &mut self.moe, &mut self.tmp[..c.dim]);
                    for i in 0..c.dim {
                        self.x[i] += self.tmp[i];
                    }
                }
            }
        }
        rmsnorm_inplace(&mut self.x, &m.out_norm, c.eps);
        self.qa_dim.quantize(&self.x);
        m.matvec(&m.output, &self.x, &self.qa_dim, &mut self.logits);
        self.tokens.push(token);
    }
}
