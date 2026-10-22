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
    /// Interleaved multi-axis rotary positions (Qwen3-VL / Qwen3.5): frequency sections for the
    /// time, height and width position of a token. Text tokens use one position on every axis, so
    /// only image tokens see a difference. None for plain rotary models.
    pub rope_sections: Option<[usize; 4]>,
    /// Gated DeltaNet linear-attention layers (Qwen3.8 hybrid); None for pure attention models.
    pub ssm: Option<Ssm>,
    /// The DeltaNet output gate is sigmoid(z) instead of silu(z) (qwen4exp).
    pub delta_sigmoid: bool,
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
    pub(crate) fn of(t: &TensorInfo) -> Self {
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
    /// A shared expert every token uses next to the routed ones (Qwen3-Next, Qwen3.5-MoE,
    /// qwen4exp, DeepSeek), when the model has one.
    pub shared: Option<SharedExpert>,
}

pub struct SharedExpert {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
    /// dim values: the shared expert's output is scaled by sigmoid(gate_inp . x); None = unscaled.
    pub gate_inp: Option<Vec<f32>>,
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
    /// Q8_0 matrices re-laid for the processor (`quant::repack_q8_0_x8`), by file offset.
    packs: std::sync::RwLock<std::collections::HashMap<usize, Arc<Vec<u8>>>>,
    /// The drive tier (stream.rs): each layer's unit, and the output head's, once it is on.
    stream_map: std::sync::OnceLock<(Vec<Option<usize>>, Option<usize>)>,
    /// Hyper-connections and n-gram embeddings (hc.rs), for models that have them.
    pub hc: Option<crate::hc::HcModel>,
    /// GPU assist (gpu.rs Offload): matrices the GPU multiplies for a processor session.
    #[cfg(feature = "cuda")]
    offload: std::sync::OnceLock<std::sync::Mutex<crate::gpu::Offload>>,
    /// Prompt batches of the processor's weights on the GPU (gpu.rs BatchGpu).
    #[cfg(feature = "cuda")]
    batch_gpu: std::sync::OnceLock<Option<std::sync::Mutex<crate::gpu::BatchGpu>>>,
}

fn vec_f32(g: &Gguf, name: &str) -> Result<Vec<f32>, Error> {
    let t = g.tensor(name)?;
    let mut out = vec![0f32; t.elements() as usize];
    dequant_row(t.dtype, g.bytes(t), &mut out);
    Ok(out)
}

/// The architectures this engine runs (`general.architecture`): the app lists models by it.
pub const ARCHITECTURES: &[&str] = &["qwen3", "qwen3moe", "qwen2", "llama", "qwen35", "qwen35moe", "qwen3vl", "qwen3tts", "qwen4exp"];

impl Model {
    /// Whether any attention or output matrix is Q8_0 (four-row dots only apply to those).
    pub fn uses_q8_0(&self) -> bool {
        self.output.dtype == DType::Q8_0 || self.layers.iter().any(|l| [&l.q, &l.k, &l.v, &l.o].iter().any(|m| m.dtype == DType::Q8_0))
    }

    pub fn load(path: &Path) -> Result<Self, Error> {
        let g = Arc::new(Gguf::open(path)?);
        let arch_name = g.str("general.architecture").unwrap_or("");
        if !ARCHITECTURES.contains(&arch_name) {
            return Err(Error::Unsupported(format!("architecture {arch_name}")));
        }

        let arch = g.str("general.architecture").unwrap_or("").to_owned();
        // qwen3vl: Qwen3 with multi-axis rotary positions (the language model of Qwen3-VL and
        // Qwen3-ASR); deepstack (image features added into early layers) is not run yet.
        // qwen3tts: the Qwen3-TTS talker, a qwen3vl model whose output head covers only the codec
        // tokens (logit i is the token `<|codec_0|>` + i; see tts.rs).
        if g.u64(&format!("{arch}.n_deepstack_layers")).unwrap_or(0) > 0 {
            return Err(Error::Unsupported("deepstack layers (Qwen3-VL image features in early layers)".into()));
        }
        let k = |s: &str| format!("{arch}.{s}");
        let dim = g.u64(&k("embedding_length"))? as usize;
        let n_head = g.u64(&k("attention.head_count"))? as usize;
        let head_dim = g.u64(&k("attention.key_length")).map(|v| v as usize).unwrap_or(dim / n_head);
        let embed_t = g.tensor("token_embd.weight")?;
        // Gated attention (a per-head output gate packed after each query head) shows in the
        // query projection's size: twice the heads' width.
        let has_hc = g.u64(&k("hyper_connection.count")).is_ok();
        let attn_gate = g.tensors.iter().any(|(k, t)| k.starts_with("blk.") && k.ends_with(".attn_q.weight") && t.rows() == 2 * n_head * head_dim);
        let cfg = Config {
            n_layer: g.u64(&k("block_count"))? as usize,
            dim,
            // Mixture-of-experts files may give only the experts' size.
            ffn: g.u64(&k("feed_forward_length")).map(|v| v as usize).unwrap_or(0),
            n_head,
            n_kv: g.u64(&k("attention.head_count_kv")).map(|v| v as usize).unwrap_or(n_head),
            head_dim,
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
            eps: g.f32(&k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            ctx_train: g.u64(&k("context_length")).unwrap_or(4096) as usize,
            // Logits: as many as the output head has rows (usually the embedding table's).
            vocab: g.tensors.get("output.weight").map_or(embed_t.rows(), |t| t.rows()),
            // Any layer: in Qwen3.5/3.8 layer 0 is a DeltaNet layer, without attention norms.
            qk_norm: g.tensors.keys().any(|k| k.starts_with("blk.") && k.ends_with(".attn_q_norm.weight")),
            n_expert: g.u64(&k("expert_count")).unwrap_or(0) as usize,
            n_used: g.u64(&k("expert_used_count")).unwrap_or(0) as usize,
            expert_ffn: g.u64(&k("expert_feed_forward_length")).unwrap_or(0) as usize,
            rope_dim: g.u64(&k("rope.dimension_count")).map_or(head_dim, |v| v as usize).min(head_dim),
            attn_gate,
            rope_sections: g.meta.get(&k("rope.dimension_sections")).and_then(|v| v.as_array()).and_then(|a| {
                let v: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
                (v.len() >= 3 && v.iter().sum::<usize>() > 0).then(|| [v[0], v[1], v[2], v.get(3).copied().unwrap_or(0)])
            }),
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
            delta_sigmoid: arch == "qwen4exp",
            arch,
        };
        // Hyper-connection models replace the per-block norms with mixers (hc.rs).
        let hc = if has_hc { Some(crate::hc::HcModel::load(&g, &cfg.arch, cfg.n_layer, cfg.dim)?) } else { None };
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
            // Qwen3.8 names the pre-FFN norm post_attention_norm; hyper-connection models have none.
            let ffn_norm = if g.tensors.contains_key(&format!("blk.{i}.ffn_norm.weight")) {
                nv("ffn_norm")?
            } else if hc.is_some() {
                Vec::new()
            } else {
                nv("post_attention_norm")?
            };
            let shared = |s: &str| g.tensors.contains_key(&format!("blk.{i}.{s}.weight"));
            layers.push(Layer {
                attn_norm: if hc.is_some() && !shared("attn_norm") { Vec::new() } else { nv("attn_norm")? },
                q: attn("attn_q")?,
                k: attn("attn_k")?,
                v: attn("attn_v")?,
                o: attn("attn_output")?,
                q_norm: if has_qk_norm { Some(nv("attn_q_norm")?) } else { None },
                k_norm: if has_qk_norm { Some(nv("attn_k_norm")?) } else { None },
                delta,
                ffn_norm,
                ffn: if cfg.n_expert > 0 {
                    Ffn::Moe(Moe {
                        router: nv("ffn_gate_inp")?,
                        gate: t("ffn_gate_exps")?,
                        up: t("ffn_up_exps")?,
                        down: t("ffn_down_exps")?,
                        shared: if shared("ffn_up_shexp") {
                            Some(SharedExpert {
                                gate: t("ffn_gate_shexp")?,
                                up: t("ffn_up_shexp")?,
                                down: t("ffn_down_shexp")?,
                                gate_inp: if shared("ffn_gate_inp_shexp") { Some(nv("ffn_gate_inp_shexp")?) } else { None },
                            })
                        } else {
                            None
                        },
                    })
                } else {
                    Ffn::Dense { gate: t("ffn_gate")?, up: t("ffn_up")?, down: t("ffn_down")? }
                },
            });
        }
        let output = g.tensors.get("output.weight").map(Mat::of).unwrap_or_else(|| Mat::of(embed_t));
        let rd = cfg.rope_dim;
        let inv_freq = (0..rd / 2).map(|i| 1.0 / cfg.rope_base.powf(2.0 * i as f32 / rd as f32)).collect();
        // Hyper-connection models norm the output in their last mixer.
        let out_norm = if hc.is_some() && !g.tensors.contains_key("output_norm.weight") { Vec::new() } else { vec_f32(&g, "output_norm.weight")? };
        let m = Self { path: path.to_path_buf(), embed: Mat::of(embed_t), out_norm, layers, output, inv_freq, cfg, gguf: g, packs: Default::default(), stream_map: Default::default(), hc, #[cfg(feature = "cuda")] offload: Default::default(), #[cfg(feature = "cuda")] batch_gpu: Default::default() };
        let n = crate::switches::get(crate::switches::STREAM_LAYERS).max(0) as usize;
        if n > 0 {
            m.stream_from_drive(n);
        }
        Ok(m)
    }

    /// Weight bytes of each layer and of the output head, as the processor reads them.
    fn layer_bytes(&self) -> (Vec<u64>, u64) {
        let g = &self.gguf;
        let mut per = vec![0u64; self.cfg.n_layer];
        for t in g.tensors.values() {
            if let Some(i) = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok()) {
                if let Some(p) = per.get_mut(i) {
                    *p += t.size as u64;
                }
            }
        }
        (per, (self.output.rows * self.output.row_bytes) as u64)
    }

    /// The automatic drive tier, decided when a session starts (the GPU's share is known then):
    /// when the processor's share does not fit in available RAM, the fewest trailing layers that
    /// make it fit are read from the drive (placement::drive_layers). Once per model; an explicit
    /// "Layers read from the drive" number was applied at load instead.
    pub fn auto_drive_tier(&self, gpu_layers: usize, max_ctx: usize) {
        if self.stream_map.get().is_some() || self.cfg.n_expert > 0 || !crate::switches::enabled(crate::switches::DRIVE_TIER) || crate::switches::get(crate::switches::STREAM_LAYERS) > 0 {
            return;
        }
        let c = &self.cfg;
        let (per, head) = self.layer_bytes();
        let kv: u64 = self.layers.iter().skip(gpu_layers).filter(|l| l.delta.is_none()).count() as u64 * 2 * max_ctx as u64 * (c.n_kv * c.head_dim) as u64 * 2;
        let budget = crate::gguf::available_ram().saturating_sub(crate::placement::RAM_MARGIN).saturating_sub(kv);
        let k = crate::placement::drive_layers(&per, head, gpu_layers, budget);
        if k > 0 && self.stream_from_drive(k) {
            let streamed: u64 = per[c.n_layer - k..].iter().sum();
            eprintln!("drive tier: {k} of {} layers ({:.1} GiB) read from the drive on every pass; {:.1} GiB of RAM available", c.n_layer - gpu_layers, streamed as f64 / (1u64 << 30) as f64, budget as f64 / (1u64 << 30) as f64);
        }
    }

    /// Reads the last `n` layers (and the output head too when `n` covers every layer) from the
    /// drive on every pass instead of keeping them in RAM (stream.rs). Not for mixtures of experts,
    /// which already read only their active experts. Leaves everything in RAM when a layer's
    /// tensors are not contiguous in the file or the file cannot be opened unbuffered.
    fn stream_from_drive(&self, n: usize) -> bool {
        let c = &self.cfg;
        if c.n_expert > 0 || n == 0 || self.stream_map.get().is_some() {
            return false;
        }
        let g = self.gguf.clone();
        let first = c.n_layer.saturating_sub(n);
        let range_of = |pick: &dyn Fn(&str) -> bool| -> Option<(usize, usize)> {
            let (mut lo, mut hi) = (usize::MAX, 0);
            for t in g.tensors.values().filter(|t| pick(&t.name)) {
                lo = lo.min(t.offset);
                hi = hi.max(t.offset + t.size);
            }
            // Nothing else may lie inside: the whole range is served from the buffer.
            (lo < hi && !g.tensors.values().any(|t| !pick(&t.name) && t.offset < hi && t.offset + t.size > lo)).then_some((lo, hi))
        };
        let mut units = Vec::new();
        let mut of = vec![None; c.n_layer];
        for (li, slot) in of.iter_mut().enumerate().skip(first) {
            let p = format!("blk.{li}.");
            let Some(r) = range_of(&|name: &str| name.starts_with(&p)) else {
                eprintln!("drive tier off: layer {li}'s tensors are not contiguous in the file");
                return false;
            };
            *slot = Some(units.len());
            units.push(r);
        }
        let mut out = None;
        if first == 0 {
            let name = if self.tied() { "token_embd.weight" } else { "output.weight" };
            if let Some(r) = range_of(&|t: &str| t == name) {
                out = Some(units.len());
                units.push(r);
            }
        }
        match g.set_stream(&self.path, &units) {
            Ok(()) => self.stream_map.set((of, out)).is_ok(),
            Err(e) => {
                eprintln!("drive tier off: {e}");
                false
            }
        }
    }

    /// The RAM expert cache for a mixture-of-experts model whose experts do not fit in available
    /// RAM next to the weights read every token (experts.rs). Decided once per model when a session
    /// starts; a model that fits keeps its experts memory-mapped.
    pub fn auto_expert_cache(&self, max_ctx: usize) {
        let c = &self.cfg;
        let g = &self.gguf;
        if c.n_expert == 0 || g.experts().is_some() || !crate::switches::enabled(crate::switches::EXPERT_RAM_CACHE) {
            return;
        }
        let mut experts = Vec::with_capacity(c.n_layer * c.n_expert);
        for l in &self.layers {
            let Ffn::Moe(e) = &l.ffn else { return };
            for ex in 0..c.n_expert {
                let r = |m: &Mat| {
                    let p = m.part(ex, c.n_expert);
                    (p.offset, p.rows * p.row_bytes)
                };
                experts.push(crate::experts::ExpertRanges { r: [r(&e.gate), r(&e.up), r(&e.down)] });
            }
        }
        let expert_bytes: usize = experts.iter().map(|e| e.bytes()).sum();
        // Read every token besides the experts: everything except the experts and the tables only
        // looked up a row at a time (token and per-layer embeddings). With GPU assist the matrices
        // on the graphics card are not read from RAM at all.
        #[cfg(feature = "cuda")]
        let on_gpu: usize = self.offload_stats().map_or(0, |o| o.2);
        #[cfg(not(feature = "cuda"))]
        let on_gpu = 0usize;
        let per_token: usize = g.tensors.values().filter(|t| !t.name.contains("_exps") && !t.name.contains("token_embd")).map(|t| t.size).sum::<usize>().saturating_sub(on_gpu);
        let kv: usize = self.layers.iter().filter(|l| l.delta.is_none()).count() * 2 * max_ctx * c.n_kv * c.head_dim * 2;
        let budget = (crate::gguf::available_ram() as usize).saturating_sub(crate::placement::RAM_MARGIN as usize + per_token + kv);
        // CEREBRO_EXPERT_CACHE_MB pins the cache size (benchmarks: experts mostly from the drive).
        let forced = std::env::var("CEREBRO_EXPERT_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()).map(|mb| mb << 20);
        let budget = forced.unwrap_or(budget);
        if forced.is_none() && (expert_bytes <= budget || budget < 256 << 20) {
            return;
        }
        // The use counts saved with the model (the GPU expert cache's profile format).
        let profile = std::fs::read_to_string(self.expert_profile_path()).ok().map(|t| t.split_whitespace().filter_map(|x| x.parse::<f32>().ok()).collect::<Vec<f32>>());
        // 16 handles per file: each carries one request at a time, and a drive reaches its speed
        // with many requests in flight (4 measured 0.39 GB/s on a 2.1 GB/s drive).
        match crate::experts::ExpertCache::with_profile(experts, &g.files(), budget, 16, profile) {
            Ok(cache) => {
                let st = cache.stats();
                eprintln!("expert cache: {} of {} experts in memory ({:.1} of {:.1} GiB); the rest read from the drive as used", st.slots, c.n_layer * c.n_expert, (st.slots * st.slot_bytes) as f64 / (1u64 << 30) as f64, expert_bytes as f64 / (1u64 << 30) as f64);
                if g.set_experts(cache) {
                    // Read the experts earlier conversations used most, beside the first message.
                    let gg = g.clone();
                    let _ = std::thread::Builder::new().name("cerebro-experts-warm".into()).spawn(move || {
                        if let Some(c) = gg.experts() {
                            let t = std::time::Instant::now();
                            let n = c.warm();
                            if n > 0 {
                                eprintln!("expert cache: {n} experts from the saved profile read in {:.1} s", t.elapsed().as_secs_f64());
                            }
                        }
                    });
                }
            }
            Err(e) => eprintln!("expert cache off: {e}"),
        }
    }

    /// Where a MoE model's expert use counts are kept: next to the model file.
    fn expert_profile_path(&self) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}.experts", self.path.display()))
    }

    /// Saves the RAM expert cache's use counts next to the model (MoE models larger than memory),
    /// so the next load starts with the experts conversations use.
    pub fn save_expert_profile(&self) {
        if let Some(c) = self.gguf.experts() {
            let text: Vec<String> = c.profile().iter().map(|v| (v.round() as u64).to_string()).collect();
            let _ = std::fs::write(self.expert_profile_path(), text.join(" "));
        }
    }

    /// Before the processor reads layer `li`: makes sure the drive tier has it.
    #[inline]
    fn stream_enter(&self, li: usize) {
        if let (Some((of, _)), Some(s)) = (self.stream_map.get(), self.gguf.stream()) {
            if let Some(Some(u)) = of.get(li) {
                s.enter(*u);
            }
        }
    }

    #[inline]
    fn stream_enter_output(&self) {
        if let (Some((_, Some(u))), Some(s)) = (self.stream_map.get(), self.gguf.stream()) {
            s.enter(*u);
        }
    }

    /// Layers read from the drive on every pass (0 when the drive tier is off).
    pub fn streamed_layers(&self) -> usize {
        self.stream_map.get().map_or(0, |(of, _)| of.iter().filter(|u| u.is_some()).count())
    }

    fn streamed(&self, m: &Mat) -> bool {
        self.stream_map.get().is_some_and(|(of, out)| {
            let li = self.layers.iter().position(|l| [&l.q, &l.k, &l.v, &l.o].iter().any(|x| x.offset == m.offset) || matches!(&l.ffn, Ffn::Dense { gate, up, down } if [gate, up, down].iter().any(|x| x.offset == m.offset)));
            li.is_some_and(|li| of[li].is_some()) || (out.is_some() && m.offset == self.output.offset)
        })
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    /// Re-lays the processor's Q8_0 matrices (layers `from`.. and the output head) for the
    /// interleaved kernels, once per model, and releases their file pages from the working set.
    pub fn prepare_cpu(&self, from: usize) {
        if !crate::switches::enabled(crate::switches::CPU_REPACK) {
            return;
        }
        let mut mats: Vec<&Mat> = Vec::new();
        for l in &self.layers[from.min(self.layers.len())..] {
            if l.delta.is_none() {
                mats.extend([&l.q, &l.k, &l.v, &l.o]);
            }
            if let Ffn::Dense { gate, up, down } = &l.ffn {
                mats.extend([gate, up, down]);
            }
        }
        mats.push(&self.output);
        let todo: Vec<&Mat> = {
            let have = self.packs.read().unwrap_or_else(|e| e.into_inner());
            mats.into_iter().filter(|m| packable(m) && !have.contains_key(&m.offset) && !self.streamed(m)).collect()
        };
        use rayon::prelude::*;
        let made: Vec<(usize, Arc<Vec<u8>>)> = todo
            .par_iter()
            .map(|m| {
                let src = self.gguf.bytes_at(m.offset, m.rows * m.row_bytes);
                let p = if m.dtype == DType::Q4K { crate::quant::repack_q4k_x8(src, m.rows, m.cols) } else { crate::quant::repack_q8_0_x8(src, m.rows, m.cols) };
                (m.offset, Arc::new(p))
            })
            .collect();
        let tied = self.tied();
        for m in &todo {
            // The embedding table is still read by token; a tied output head shares it.
            if !(tied && m.offset == self.embed.offset) {
                self.gguf.release(m.offset, m.rows * m.row_bytes);
            }
        }
        self.packs.write().unwrap_or_else(|e| e.into_inner()).extend(made);
    }

    /// The file ranges the processor still reads directly for layers `from`.. (everything but the
    /// interleaved copies), plus the output head and embedding table: what a prefault should touch.
    fn direct_ranges(&self, from: usize) -> Vec<(usize, usize)> {
        let packs = self.packs.read().unwrap_or_else(|e| e.into_inner());
        let (of, out) = self.stream_map.get().map_or((&[][..], None), |(of, out)| (&of[..], *out));
        let mut v: Vec<(usize, usize)> = self.layers[from.min(self.layers.len())..]
            .iter()
            .enumerate()
            .filter(|(i, _)| !matches!(of.get(from + i), Some(Some(_))))
            .flat_map(|(_, l)| {
                let mut r = layer_ranges(l);
                if let (Some(_), Ffn::Moe(e)) = (self.gguf.experts(), &l.ffn) {
                    let skip = [e.gate.offset, e.up.offset, e.down.offset];
                    r.retain(|x| !skip.contains(&x.0));
                }
                r
            })
            .filter(|r| !packs.contains_key(&r.0))
            .collect();
        if !packs.contains_key(&self.output.offset) && out.is_none() {
            v.push((self.output.offset, self.output.rows * self.output.row_bytes));
        }
        v.push((self.embed.offset, self.embed.rows * self.embed.row_bytes));
        v
    }

    /// The interleaved copy of `m`, if `prepare_cpu` made one.
    fn pack(&self, m: &Mat) -> Option<Arc<Vec<u8>>> {
        if !packable(m) {
            return None;
        }
        self.packs.read().ok()?.get(&m.offset).cloned()
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

    /// The rotary (cos, sin) pairs for a token at position `p` = (time, height, width). With
    /// interleaved sections (Qwen3-VL style), frequency j follows the height position when
    /// j mod 3 == 1, the width position when j mod 3 == 2 (within each section's span) and the time
    /// position otherwise; a text token has all three equal, which is plain RoPE.
    pub fn rope_cs(&self, p: [usize; 3]) -> Vec<(f32, f32)> {
        self.inv_freq
            .iter()
            .enumerate()
            .map(|(j, f)| {
                let pos = match self.cfg.rope_sections {
                    Some(sec) => {
                        let total = (sec[0] + sec[1] + sec[2] + sec[3]).max(1);
                        let sector = j % total;
                        if sector % 3 == 1 && sector < 3 * sec[1] {
                            p[1]
                        } else if sector % 3 == 2 && sector < 3 * sec[2] {
                            p[2]
                        } else if sector % 3 == 0 && sector < 3 * sec[0] {
                            p[0]
                        } else {
                            0
                        }
                    }
                    None => p[0],
                };
                let (sn, cs) = (pos as f32 * f).sin_cos();
                (cs, sn)
            })
            .collect()
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

    /// The weight matrices a token reads besides the experts and the row-lookup tables, layer by
    /// layer and then the output head: what GPU assist puts on the graphics card.
    pub fn dense_mats(&self) -> Vec<&Mat> {
        let mut v: Vec<&Mat> = Vec::new();
        for (li, l) in self.layers.iter().enumerate() {
            match &l.delta {
                Some(d) => v.extend([&d.qkv, &d.z, &d.alpha, &d.beta, &d.out]),
                None => v.extend([&l.q, &l.k, &l.v, &l.o]),
            }
            match &l.ffn {
                Ffn::Dense { gate, up, down } => v.extend([gate, up, down]),
                Ffn::Moe(e) => {
                    if let Some(s) = &e.shared {
                        v.extend([&s.gate, &s.up, &s.down]);
                    }
                }
            }
            if let Some(h) = &self.hc {
                v.extend([&h.attn[li].down, &h.attn[li].up, &h.ffn[li].down, &h.ffn[li].up]);
                if let Some(p) = h.ple.as_ref().filter(|p| p.layer == li) {
                    v.extend([&p.key, &p.value]);
                }
            }
        }
        if let Some(h) = &self.hc {
            v.extend([&h.head.down, &h.head.up]);
        }
        v.push(&self.output);
        v.retain(|m| m.rows > 0);
        v
    }

    /// Sends prompt batches of the processor's weights to the GPU from now on (when there is a CUDA
    /// GPU with tensor cores and "Use the graphics card" is on). Returns whether it is on.
    #[cfg(feature = "cuda")]
    pub fn enable_batch_gpu(&self) -> bool {
        self.batch_gpu
            .get_or_init(|| {
                if !crate::switches::enabled(crate::switches::GPU_ENABLED) {
                    return None;
                }
                crate::gpu::BatchGpu::new().map_err(|e| eprintln!("prompt batches stay on the processor: {e}")).ok().map(std::sync::Mutex::new)
            })
            .is_some()
    }

    /// Turns GPU assist on for this model (once): the matrices of `dense_mats` that fit in VRAM.
    #[cfg(feature = "cuda")]
    pub fn enable_offload(&self) -> Result<(usize, usize, usize), Error> {
        if self.offload.get().is_none() {
            let o = crate::gpu::Offload::new(self)?;
            let _ = self.offload.set(std::sync::Mutex::new(o));
        }
        let o = self.offload.get().expect("set").lock().unwrap_or_else(|e| e.into_inner());
        Ok((o.placed, o.candidates, o.bytes))
    }

    /// GPU assist's counters: matrices placed, candidates, VRAM bytes, products computed.
    #[cfg(feature = "cuda")]
    pub fn offload_stats(&self) -> Option<(usize, usize, usize, u64)> {
        self.offload.get().map(|o| {
            let o = o.lock().unwrap_or_else(|e| e.into_inner());
            (o.placed, o.candidates, o.bytes, o.calls)
        })
    }

    /// `matvec` for the building blocks in other modules (hc.rs).
    pub(crate) fn matvec_pub(&self, m: &Mat, x: &[f32], qa: &QAct, out: &mut [f32]) {
        self.matvec(m, x, qa, out);
    }

    pub(crate) fn matvec_many_pub(&self, jobs: &mut [(&Mat, &mut [f32])], x: &[f32], qa: &QAct) {
        self.matvec_many(jobs, x, qa);
    }

    /// Several matrix-vector products that read the same input, in one parallel region: one dispatch
    /// instead of one per matrix (dispatch, not arithmetic, dominated small models).
    fn matvec_many(&self, jobs: &mut [(&Mat, &mut [f32])], x: &[f32], qa: &QAct) {
        let mut j: Vec<(&Mat, &[f32], &QAct, &mut [f32])> = jobs.iter_mut().map(|(m, o)| (*m, x, qa, &mut **o)).collect();
        self.matvec_multi(&mut j);
    }

    /// Matrix-vector products with their own inputs (e.g. one per expert), in one parallel region.
    fn matvec_multi(&self, jobs: &mut [(&Mat, &[f32], &QAct, &mut [f32])]) {
        #[cfg(feature = "cuda")]
        if let Some(off) = self.offload.get() {
            let mut o = off.lock().unwrap_or_else(|e| e.into_inner());
            if !jobs.is_empty() && jobs.iter().all(|j| o.has(j.0) && j.2.q.len() >= j.0.cols) {
                for j in jobs.iter_mut() {
                    if let Err(e) = o.run(j.0, j.2, j.3) {
                        panic!("GPU assist failed: {e}");
                    }
                }
                return;
            }
        }
        let g: &Gguf = &self.gguf;
        let total: usize = jobs.iter().map(|j| j.0.rows).sum();
        let pool = crate::pool::global();
        // Small items keep the slowest core (an E-core) from holding up the end of each job.
        let div = crate::switches::get(crate::switches::CPU_CHUNK_DIV).max(1) as usize;
        // A multiple of eight rows: the interleaved kernels work on eight-row groups.
        let chunk = (total / (pool.threads() * div)).clamp(8, 512).next_multiple_of(8);
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
        let packs: Vec<Option<Arc<Vec<u8>>>> = jobs.iter().map(|j| self.pack(j.0)).collect();
        pool.run(items.len(), &|ii| {
            let (ji, r0, n) = items[ii];
            let (m, x, qa) = ins[ji];
            let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
            let rb = m.row_bytes;
            let mut r = r0;
            if let Some(p) = &packs[ji] {
                while r % 8 == 0 && r + 8 <= r0 + n {
                    let v = gemv_packed(m, p, r / 8, qa);
                    for (k, v) in v.into_iter().enumerate() {
                        // SAFETY: r + k < m.rows == the job's output length; ranges never overlap.
                        unsafe { *outs[ji].0.add(r + k) = v };
                    }
                    r += 8;
                }
            }
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
        if let Some(c) = self.gguf.experts() {
            // Read the next layer's likely experts in the same batch: its router on this input.
            let ahead: Vec<usize> = match self.layers.get(li + 1).map(|l| &l.ffn) {
                Some(Ffn::Moe(next)) => {
                    let mut scores = vec![0f32; ne];
                    let mut chosen = Vec::with_capacity(experts.len());
                    route(&next.router, h, experts.len(), &mut scores, &mut chosen);
                    chosen.iter().map(|&(ex, _)| (li + 1) * ne + ex).collect()
                }
                _ => Vec::new(),
            };
            c.ensure_with(&experts.iter().map(|&(ex, _)| li * ne + ex).collect::<Vec<_>>(), &ahead);
        }
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
        if let Some(c) = g.experts() {
            c.ensure(&groups.iter().map(|(ex, _)| li * ne + ex).collect::<Vec<_>>());
        }
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

    /// Token `token`'s embedding row (input side), into `out` (`cfg.dim` values).
    pub fn embedding(&self, token: u32, out: &mut [f32]) {
        self.embed_row(token, out);
    }

    fn embed_row(&self, token: u32, out: &mut [f32]) {
        dequant_row(self.embed.dtype, self.row(&self.embed, token as usize), out);
    }

    /// out[t * rows + r] = M x_t for T tokens (`xs` token-major, `qas` the same rows quantised). Each
    /// weight row is read once and dotted against every token while it is in L1, so weights cross the
    /// memory bus once per batch instead of once per token.
    fn matmul_batch(&self, m: &Mat, xs: &[f32], qas: &[QAct], out: &mut [f32]) {
        let g: &Gguf = &self.gguf;
        #[cfg(feature = "cuda")]
        if qas.len() >= crate::gpu::BATCH_GPU_MIN_TOKENS && crate::gpu::BatchGpu::takes(m) {
            if let Some(Some(b)) = self.batch_gpu.get() {
                let mut b = b.lock().unwrap_or_else(|e| e.into_inner());
                match b.run(m, g.bytes_at(m.offset, m.rows * m.row_bytes), qas, out) {
                    Ok(()) => return,
                    Err(e) => {
                        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            eprintln!("a prompt batch stayed on the processor ({}x{} {:?}): {e}", m.rows, m.cols, m.dtype);
                        }
                    }
                }
            }
        }
        let (cols, rows, rb) = (m.cols, m.rows, m.row_bytes);
        assert!(out.len() >= qas.len() * rows && xs.len() >= qas.len() * cols);
        let pool = crate::pool::global();
        let chunk = (rows / (pool.threads() * 4)).clamp(8, 256).next_multiple_of(8);
        let o = SyncPtr(out.as_mut_ptr());
        let pack = self.pack(m);
        pool.run(rows.div_ceil(chunk), &|i| {
            let all = g.bytes_at(m.offset, rows * rb);
            let (r0, r1) = (i * chunk, ((i + 1) * chunk).min(rows));
            let mut r = r0;
            if let Some(p) = &pack {
                while r % 8 == 0 && r + 8 <= r1 {
                    let mut t0 = 0;
                    while t0 + 4 <= qas.len() {
                        let v = gemm_packed(m, p, r / 8, [&qas[t0], &qas[t0 + 1], &qas[t0 + 2], &qas[t0 + 3]]);
                        for (t, row) in v.iter().enumerate() {
                            for (k, &x) in row.iter().enumerate() {
                                // SAFETY: (t0 + t, r + k) is written by exactly this item.
                                unsafe { *o.at((t0 + t) * rows + r + k) = x };
                            }
                        }
                        t0 += 4;
                    }
                    for (t, qa) in qas.iter().enumerate().skip(t0) {
                        for (k, &x) in gemv_packed(m, p, r / 8, qa).iter().enumerate() {
                            // SAFETY: as above.
                            unsafe { *o.at(t * rows + r + k) = x };
                        }
                    }
                    r += 8;
                }
            }
            // Q8_0: tiles of two rows by four tokens; what does not fill a tile goes below.
            if m.dtype == DType::Q8_0 && qas.len() >= 4 {
                while r + 2 <= r1 {
                    let pair = [&all[r * rb..(r + 1) * rb], &all[(r + 1) * rb..(r + 2) * rb]];
                    let mut t0 = 0;
                    while t0 + 4 <= qas.len() {
                        let Some(v) = crate::quant::gemm_q8_0_2x4(pair, [&qas[t0], &qas[t0 + 1], &qas[t0 + 2], &qas[t0 + 3]]) else { break };
                        for (k, row) in v.iter().enumerate() {
                            for (t, &x) in row.iter().enumerate() {
                                // SAFETY: (t0 + t, r + k) is written by exactly this item.
                                unsafe { *o.at((t0 + t) * rows + r + k) = x };
                            }
                        }
                        t0 += 4;
                    }
                    if t0 == 0 {
                        break;
                    }
                    for (t, qa) in qas.iter().enumerate().skip(t0) {
                        for k in 0..2 {
                            let v = dot_any(m.dtype, pair[k], &xs[t * cols..(t + 1) * cols], qa);
                            // SAFETY: as above.
                            unsafe { *o.at(t * rows + r + k) = v };
                        }
                    }
                    r += 2;
                }
            }
            for r in r..r1 {
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
        DType::IQ4NL => crate::quant::dot_iq4nl_q8(row, qa),
        DType::IQ2S => crate::quant::dot_iq2s_k8(row, qa).unwrap_or_else(|| dot_row(dt, row, x)),
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

/// Prompt tokens per layer-major chunk (step_layer_major).
const LAYER_MAJOR_CHUNK: usize = 256;

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

/// The hidden size of the model's shared experts (0 without any).
fn shared_ffn(m: &Model) -> usize {
    m.layers.iter().find_map(|l| match &l.ffn {
        Ffn::Moe(e) => e.shared.as_ref().map(|s| s.gate.rows),
        _ => None,
    }).unwrap_or(0)
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


/// Whether `m` has an interleaved processor layout (Q8_0 and Q4_K, rows in eights).
fn packable(m: &Mat) -> bool {
    m.rows % 8 == 0 && match m.dtype {
        DType::Q8_0 => m.cols % 32 == 0,
        DType::Q4K => m.cols % 256 == 0,
        _ => false,
    }
}

/// Eight rows' outputs of an interleaved matrix for one token.
fn gemv_packed(m: &Mat, p: &[u8], g: usize, qa: &QAct) -> [f32; 8] {
    if m.dtype == DType::Q4K {
        crate::quant::gemv_q4k_x8(p, m.cols / 256, g, qa)
    } else {
        crate::quant::gemv_q8_0_x8(p, m.cols / 32, g, qa)
    }
}

/// Eight rows' outputs of an interleaved matrix for four tokens.
fn gemm_packed(m: &Mat, p: &[u8], g: usize, qa: [&QAct; 4]) -> [[f32; 8]; 4] {
    if m.dtype == DType::Q4K {
        crate::quant::gemm_q4k_x8_4(p, m.cols / 256, g, qa)
    } else {
        crate::quant::gemm_q8_0_x8_4(p, m.cols / 32, g, qa)
    }
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

/// How many rotary positions all of `images` saved: text after them lags its slots by this.
fn rope_shift_of(images: &[ImageSpan]) -> usize {
    images.iter().filter(|im| !im.linear).map(|im| im.nx * im.ny - im.nx.max(im.ny)).sum()
}

/// `Session::rope_pos` for a list of image spans (sorted by slot).
fn rope_pos_in(images: &[ImageSpan], slot: usize) -> [usize; 3] {
    let mut saved = 0;
    for im in images {
        let end = im.slot0 + im.nx * im.ny;
        if slot < im.slot0 {
            break;
        }
        if im.linear {
            // Audio: one position per token on every axis, like text.
            continue;
        }
        if slot < end {
            let i = slot - im.slot0;
            let p0 = im.slot0 - saved;
            return [p0, p0 + i / im.nx, p0 + i % im.nx];
        }
        saved += im.nx * im.ny - im.nx.max(im.ny);
    }
    let p = slot - saved;
    [p, p, p]
}

/// An image (or audio) in a session's cache: `nx * ny` slots from `slot0`, laid out row by row.
/// Audio is `linear`: its tokens take consecutive 1-D positions, as text does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    pub slot0: usize,
    pub nx: usize,
    pub ny: usize,
    pub linear: bool,
}

/// Image encoder output as a language model takes it: `rows * cols` tokens of `dim` values.
#[derive(Debug, Clone, Copy)]
pub struct ImageTokens<'a> {
    pub cols: usize,
    pub rows: usize,
    pub dim: usize,
    pub data: &'a [f32],
}

/// One piece of a prompt with images (`Session::feed_parts`).
#[derive(Debug, Clone, Copy)]
pub enum Part<'a> {
    Text(&'a [u32]),
    Image(ImageTokens<'a>),
    /// Audio encoder output (`audio::AudioEmbedding`): `cols` tokens in a row, `rows` = 1.
    Audio(ImageTokens<'a>),
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
    /// The processor's conversation cache, f16 (as on the GPU): half the bytes attention reads.
    k: Vec<Vec<u16>>,
    v: Vec<Vec<u16>>,
    pub tokens: Vec<u32>,
    /// Image spans in the cache, in slot order: `nx * ny` slots each, whose tokens have 2D rotary
    /// positions; text after an image continues `max(nx, ny)` positions on, not `nx * ny`.
    pub images: Vec<ImageSpan>,
    /// Layer ranges run by workers on other computers (cluster.rs); processor sessions only.
    pub remote: Option<crate::cluster::Remote>,
    /// Hyper-connection streams and n-gram state (hc.rs), for models that have them.
    hcs: Option<crate::hc::HcState>,
    /// Shared-expert scratch: gate, up, quantised activation, output.
    shg: Vec<f32>,
    shu: Vec<f32>,
    qa_sh: QAct,
    sh_out: Vec<f32>,
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
    pub prof: [f64; 10],
    pub logits: Vec<f32>,
    /// The last position's hidden state after the final norm (the input to the output head):
    /// what a TTS talker hands its code predictor. Processor sessions only.
    pub hidden: Vec<f32>,
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
        m.auto_drive_tier(0, max_ctx);
        m.auto_expert_cache(max_ctx);
        m.prepare_cpu(0);
        if crate::switches::enabled(crate::switches::PREFAULT) {
            // As much of what the processor reads from the file as available RAM holds (interleaved
            // copies are in memory already).
            let mut room = (crate::gguf::available_ram() as f64 * 0.9) as usize;
            let t = Instant::now();
            let mut ranges = m.direct_ranges(0);
            ranges.retain_mut(|r| {
                let keep = r.1.min(room);
                room -= keep;
                r.1 = keep;
                keep > 0
            });
            g.touch(&ranges);
            if std::env::var_os("CEREBRO_PROFILE").is_some() {
                let touched: usize = ranges.iter().map(|r| r.1).sum();
                eprintln!("touched {:.1} of {:.1} GB in {:.2}s", touched as f64 / 1e9, g.len() as f64 / 1e9, t.elapsed().as_secs_f64());
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
            k: (0..c.n_layer).map(|i| if mine(i) && attn(i) { vec![0u16; max_ctx * kv] } else { Vec::new() }).collect(),
            v: (0..c.n_layer).map(|i| if mine(i) && attn(i) { vec![0u16; max_ctx * kv] } else { Vec::new() }).collect(),
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
            prof: [0.0; 10],
            att_part: Vec::new(),
            logits: vec![0.0; c.vocab],
            hidden: vec![0.0; c.dim],
            stats: Stats::default(),
            progress: Arc::default(),
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "portable")]
            portable: None,
            images: Vec::new(),
            remote: None,
            hcs: m.hc.as_ref().map(|h| crate::hc::HcState::new(h, c.dim)),
            shg: vec![0.0; shared_ffn(m)],
            shu: vec![0.0; shared_ffn(m)],
            qa_sh: QAct::new(shared_ffn(m).max(32)),
            sh_out: vec![0.0; if shared_ffn(m) > 0 { c.dim } else { 0 }],
        }
    }

    /// Makes the cache hold exactly `prompt`, reusing the longest prefix already present
    /// (conversation cache), and leaves the logits for the next token. Returns tokens reused.
    pub fn feed(&mut self, m: &Model, prompt: &[u32]) -> Result<usize, Error> {
        crate::pool::for_reading();
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
        // An image the common prefix cuts into is read again whole.
        if let Some(im) = self.images.iter().find(|im| im.slot0 < common && common < im.slot0 + im.nx * im.ny) {
            common = im.slot0;
            if m.cfg.ssm.is_some() {
                common = 0;
                self.reset_recurrent();
            }
        }
        self.images.retain(|im| im.slot0 + im.nx * im.ny <= common);
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.rope_shift = rope_shift_of(&self.images) as i32;
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
        if let Some(from) = self.portable.as_ref().filter(|p| p.is_split()).map(|p| p.gpu_layers()) {
            // Split: each chunk through the GPU's layers, then the same chunk through the processor's.
            for c in prompt[common..].chunks(32) {
                let mut xs = Vec::new();
                self.portable.as_mut().expect("portable").run_split(m, c, self.tokens.len(), &mut xs)?;
                self.step_batch_from(m, c, Some(xs), from);
                self.progress.set(self.tokens.len());
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
        if dense && self.remote.is_none() && crate::switches::enabled(crate::switches::CPU_BATCH_PREFILL) {
            for c in prompt[common..].chunks(CPU_CHUNK) {
                self.step_batch(m, c);
                self.progress.set(self.tokens.len());
            }
        } else if self.processor_only() && crate::switches::enabled(crate::switches::CPU_BATCH_PREFILL) {
            for c in prompt[common..].chunks(LAYER_MAJOR_CHUNK) {
                self.step_layer_major(m, c);
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

    /// Whether every layer runs on this computer's processor (no GPU session, no workers).
    fn processor_only(&self) -> bool {
        #[cfg(feature = "cuda")]
        if self.gpu.is_some() {
            return false;
        }
        #[cfg(feature = "portable")]
        if self.portable.is_some() {
            return false;
        }
        self.remote.is_none()
    }

    /// A prompt chunk through the model layer by layer: every token through layer 0, then every
    /// token through layer 1, and so on, with the per-token kernels. The results are the same as
    /// one token at a time (each layer's cache and recurrent state still fill in token order), but
    /// a layer's weights and experts are fetched once for the whole chunk: for a mixture of
    /// experts larger than memory that is one expert load per chunk instead of one per token.
    /// For the layer types the batched path does not cover (recurrent, gated attention, experts,
    /// hyper-connections).
    fn step_layer_major(&mut self, m: &Model, toks: &[u32]) {
        let dim = m.cfg.dim;
        let n = toks.len();
        let pos0 = self.tokens.len();
        let wide = self.hcs.as_ref().map_or(0, |s| s.xh.len());
        let mut xs = vec![0f32; n * dim];
        let mut xhs = vec![0f32; n * wide];
        for (t, &tok) in toks.iter().enumerate() {
            m.embed_row(tok, &mut xs[t * dim..(t + 1) * dim]);
        }
        // The chunk's tokens go in first, so n-gram lookups see the chunk's earlier tokens.
        self.tokens.extend_from_slice(toks);
        for li in 0..m.cfg.n_layer {
            for t in 0..n {
                self.x.copy_from_slice(&xs[t * dim..(t + 1) * dim]);
                if let Some(s) = self.hcs.as_mut() {
                    // Layer 0 starts the streams from the embedding (layers_range).
                    if li > 0 {
                        s.xh.copy_from_slice(&xhs[t * wide..(t + 1) * wide]);
                    }
                    s.token = toks[t];
                }
                self.layers_range(m, li, li + 1, pos0 + t);
                xs[t * dim..(t + 1) * dim].copy_from_slice(&self.x);
                if let Some(s) = self.hcs.as_ref() {
                    xhs[t * wide..(t + 1) * wide].copy_from_slice(&s.xh);
                }
            }
        }
        // Logits for the last token.
        self.x.copy_from_slice(&xs[(n - 1) * dim..]);
        if let Some(s) = self.hcs.as_mut() {
            s.xh.copy_from_slice(&xhs[(n - 1) * wide..]);
        }
        self.head(m);
    }

    /// Appends one token: runs the full forward pass and writes logits.
    /// A processor session with GPU assist: the CPU runs the forward pass and the first CUDA GPU
    /// multiplies the weight matrices read every token (gpu.rs Offload). For architectures the
    /// whole-GPU path does not run yet; works for every architecture the processor runs.
    #[cfg(feature = "cuda")]
    pub fn new_assisted(m: &Model, max_ctx: usize) -> Result<Self, Error> {
        if !crate::switches::enabled(crate::switches::GPU_ENABLED) {
            return Err(Error::Unsupported("the graphics card is switched off (Use the graphics card)".into()));
        }
        // The graphics card's share first, so the RAM left for experts (Session::new) counts it.
        let (placed, cands, bytes) = m.enable_offload()?;
        eprintln!("GPU assist: {placed} of {cands} weight matrices on the graphics card ({:.2} GiB)", bytes as f64 / (1u64 << 30) as f64);
        m.enable_batch_gpu();
        Ok(Self::new(m, max_ctx))
    }

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
        if m.hc.is_some() || m.layers.iter().any(|l| matches!(&l.ffn, Ffn::Moe(e) if e.shared.is_some())) {
            return Err(Error::Unsupported("this model's layer types run on the processor for now".into()));
        }
        let g = crate::gpu::GpuState::load_with(m, max_ctx, force_layers)?;
        if g.is_split() {
            m.auto_drive_tier(g.gpu_layers(), max_ctx);
            m.prepare_cpu(g.gpu_layers());
            // The processor's layers read prompts with the GPU's help.
            m.enable_batch_gpu();
        }
        if g.is_split() && crate::switches::enabled(crate::switches::PREFAULT) {
            // The CPU's share is read into RAM now (as a CPU session does at start), not by page
            // faults during the first prompt; as much as available RAM holds.
            let mut ranges = m.direct_ranges(g.gpu_layers());
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
        if !crate::switches::enabled(crate::switches::PORTABLE_GPU) {
            return Err(Error::Unsupported("other graphics cards are switched off (Turbo > Other graphics cards)".into()));
        }
        if m.hc.is_some() || m.layers.iter().any(|l| matches!(&l.ffn, Ffn::Moe(e) if e.shared.is_some())) {
            return Err(Error::Unsupported("this model's layer types run on the processor for now".into()));
        }
        let p = crate::portable::PortableState::load(m, max_ctx)?;
        let from = p.gpu_layers();
        if p.is_split() {
            m.auto_drive_tier(from, max_ctx);
            m.prepare_cpu(from);
        }
        if p.is_split() && crate::switches::enabled(crate::switches::PREFAULT) {
            // The processor's share is read into RAM now, as for a CUDA split.
            m.gguf().touch(&m.direct_ranges(from));
        }
        // A split keeps the KV cache of the processor's layers here; a whole-GPU model keeps none.
        let mut s = Self::empty_from(m, max_ctx, from);
        s.max_ctx = max_ctx;
        s.portable = Some(p);
        Ok(s)
    }

    /// Like `feed`, for a prompt with images: `parts` in order, text as token ids and each image
    /// as its encoder output (`vision::ImageEmbedding`, one row of `dim` values per image token,
    /// `cols` across). Image tokens sit in the cache as pseudo-tokens derived from the image's
    /// values, so the conversation cache reuses an image only when it is the same image. Runs on
    /// the processor or wholly on a CUDA card. Returns the reused prefix length.
    pub fn feed_parts(&mut self, m: &Model, parts: &[Part<'_>]) -> Result<usize, Error> {
        if !parts.iter().any(|p| matches!(p, Part::Image(_) | Part::Audio(_))) {
            let ids: Vec<u32> = parts.iter().flat_map(|p| if let Part::Text(t) = p { t.to_vec() } else { Vec::new() }).collect();
            return self.feed(m, &ids);
        }
        #[cfg(feature = "portable")]
        if self.portable.is_some() {
            return Err(Error::Unsupported("images are not read on this graphics card yet: load the model on the processor or an NVIDIA card to use them".into()));
        }
        #[cfg(feature = "cuda")]
        if self.gpu.as_ref().is_some_and(|g| g.is_split()) {
            return Err(Error::Unsupported("images are not read when a model is split between the graphics card and the processor yet: load it whole on one of them".into()));
        }
        // The full slot sequence, with image tokens as pseudo-ids.
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut linear: Vec<usize> = Vec::new();
        for p in parts {
            match p {
                Part::Text(t) => ids.extend_from_slice(t),
                Part::Image(e) | Part::Audio(e) => {
                    if e.dim != m.cfg.dim {
                        return Err(Error::Request(format!("the encoder gives {} values per token; this model takes {}", e.dim, m.cfg.dim)));
                    }
                    let h = e.data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x0100_0000_01b3));
                    if matches!(p, Part::Audio(_)) {
                        linear.push(ids.len());
                    }
                    spans.push((ids.len(), e.cols, e.rows, *e));
                    for i in 0..e.cols * e.rows {
                        ids.push(0x8000_0000 | ((h as u32 ^ (i as u32).wrapping_mul(0x9e37_79b9)) & 0x7fff_ffff));
                    }
                }
            }
        }
        if ids.len() >= self.max_ctx {
            return Err(Error::Request(format!("prompt of {} tokens exceeds the {}-token context", ids.len(), self.max_ctx)));
        }
        let mut common = self.tokens.iter().zip(&ids).take_while(|(a, b)| a == b).count();
        if common == ids.len() {
            common -= 1;
        }
        let mut reset = m.cfg.ssm.is_some() && common < self.tokens.len();
        if let Some(&(s0, _, _, _)) = spans.iter().find(|(s0, nx, ny, _)| *s0 < common && common < s0 + nx * ny) {
            common = s0;
            reset |= m.cfg.ssm.is_some();
        }
        if reset {
            common = 0;
            self.reset_recurrent();
            #[cfg(feature = "cuda")]
            if let Some(g) = self.gpu.as_mut() {
                g.reset_recurrent()?;
            }
        }
        self.tokens.truncate(common);
        self.images.retain(|im| im.slot0 + im.nx * im.ny <= common);
        let t0 = Instant::now();
        self.progress.begin(common);
        for &(s0, nx, ny, _) in &spans {
            if s0 >= common || s0 + nx * ny <= common {
                if s0 >= common {
                    self.images.push(ImageSpan { slot0: s0, nx, ny, linear: linear.contains(&s0) });
                }
            }
        }
        self.images.sort_by_key(|im| im.slot0);
        #[cfg(feature = "cuda")]
        if self.gpu.is_some() {
            let rope: Vec<i32> = (common..ids.len()).flat_map(|sl| self.rope_pos(sl)).map(|v| v as i32).collect();
            let rows: Vec<(usize, Vec<f32>)> = spans
                .iter()
                .flat_map(|&(s0, nx, ny, e)| (0..nx * ny).filter(move |i| s0 + i >= common).map(move |i| (s0 + i, e.data[i * e.dim..(i + 1) * e.dim].to_vec())))
                .collect();
            let shift = rope_shift_of(&self.images) as i32;
            let g = self.gpu.as_mut().expect("gpu");
            g.progress = Some(self.progress.clone());
            g.image = Some(crate::gpu::ImagePrefill { from: common, rope, rows });
            let r = g.prefill(m, &ids[common..], common, Some(&mut self.logits));
            g.image = None;
            g.rope_shift = shift;
            r?;
            self.tokens.extend_from_slice(&ids[common..]);
            self.progress.set(self.tokens.len());
            self.stats = Stats { prompt_tokens: ids.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
            return Ok(common);
        }
        crate::pool::for_reading();
        for slot in common..ids.len() {
            let id = ids[slot];
            match spans.iter().find(|(s0, nx, ny, _)| *s0 <= slot && slot < s0 + nx * ny) {
                Some(&(s0, _, _, e)) => {
                    let i = slot - s0;
                    self.x.copy_from_slice(&e.data[i * e.dim..(i + 1) * e.dim]);
                }
                None => m.embed_row(id, &mut self.x),
            }
            if let Some(s) = self.hcs.as_mut() {
                s.token = if id & 0x8000_0000 != 0 { u32::MAX } else { id };
            }
            self.layers_from(m, 0, slot);
            self.tokens.push(id);
            self.progress.set(self.tokens.len());
        }
        self.head(m);
        self.stats = Stats { prompt_tokens: ids.len(), reused_tokens: common, prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, ..Default::default() };
        Ok(common)
    }

    /// Appends `toks` (dense models): the same math as `step` for a chunk of tokens, with every
    /// matrix applied to the whole chunk at once (`Model::matmul_batch`). Writes logits for the last.
    fn step_batch(&mut self, m: &Model, toks: &[u32]) {
        self.step_batch_from(m, toks, None, 0);
    }

    /// `step_batch` from layer `from`, on activations `x0` (tokens x width) when a GPU ran the
    /// layers before; embeddings otherwise.
    fn step_batch_from(&mut self, m: &Model, toks: &[u32], x0: Option<Vec<f32>>, from: usize) {
        // Section times for this chunk (the same indices as `prof`), added to `prof` at the end.
        let mut pp = [0f64; 7];
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
        let cs: Vec<Vec<(f32, f32)>> = (0..nt).map(|t| m.rope_cs(self.rope_pos(pos0 + t))).collect();
        let norm_quant = |x: &[f32], w: &[f32], h: &mut [f32], qa: &mut [QAct]| {
            for t in 0..nt {
                rmsnorm(&x[t * dim..(t + 1) * dim], w, c.eps, &mut h[t * dim..(t + 1) * dim]);
                qa[t].quantize(&h[t * dim..(t + 1) * dim]);
            }
        };
        for (li, l) in m.layers.iter().enumerate().skip(from) {
            m.stream_enter(li);
            let mut tp = Instant::now();
            let mut lap = |k: usize, pp: &mut [f64; 7]| {
                let n = Instant::now();
                pp[k] += (n - tp).as_secs_f64();
                tp = n;
            };
            norm_quant(&x, &l.attn_norm, &mut h, &mut qa_dim);
            lap(1, &mut pp);
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
                crate::quant::store_f16(&mut self.k[li][p * kvd..(p + 1) * kvd], &kb[t * kvd..(t + 1) * kvd]);
                crate::quant::store_f16(&mut self.v[li][p * kvd..(p + 1) * kvd], &vb[t * kvd..(t + 1) * kvd]);
            }
            lap(0, &mut pp);
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
                    crate::quant::scores_f16(qv, &kc[kvh * hd..], kvd, &mut sc[..len]);
                    let mut mx = f32::NEG_INFINITY;
                    for v in sc.iter_mut() {
                        *v *= scale;
                        mx = mx.max(*v);
                    }
                    let mut sum = 0.0;
                    for v in sc.iter_mut() {
                        *v = crate::quant::exp_fast(*v - mx);
                        sum += *v;
                    }
                    // SAFETY: (t, hh) owns this head slice of att.
                    let o = unsafe { std::slice::from_raw_parts_mut(out.at(t * qd + hh * hd), hd) };
                    o.fill(0.0);
                    crate::quant::weighted_f16(&sc[..len], &vc[kvh * hd..], kvd, o);
                    // Normalise last, in the same order as `step`, so both paths give identical results.
                    for v in o.iter_mut() {
                        *v /= sum;
                    }
                });
            });
            for t in 0..nt {
                qa_att[t].quantize(&att[t * qd..(t + 1) * qd]);
            }
            lap(2, &mut pp);
            m.matmul_batch(&l.o, &att, &qa_att, &mut tmp);
            lap(3, &mut pp);
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
            lap(5, &mut pp);
            for i in 0..nt * dim {
                x[i] += tmp[i];
            }
        }
        let last = &mut x[(nt - 1) * dim..nt * dim];
        rmsnorm_inplace(last, &m.out_norm, c.eps);
        self.qa_dim.quantize(last);
        m.matvec(&m.output, last, &self.qa_dim, &mut self.logits);
        self.tokens.extend_from_slice(toks);
        for (a, b) in self.prof.iter_mut().zip(pp) {
            *a += b;
        }
    }

    /// Forgets every position, keeping the session's buffers (and its graphics-card copy of the
    /// model) for the next prompt.
    pub fn clear(&mut self) -> Result<(), Error> {
        self.tokens.clear();
        self.images.clear();
        self.reset_recurrent();
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            g.reset_recurrent()?;
            g.rope_shift = 0;
        }
        Ok(())
    }

    /// Appends one position whose input is `row` (a raw embedding, `cfg.dim` values) instead of a
    /// token's: a TTS talker's inputs are sums of text and codec embeddings. 1-D positions; the
    /// cache holds a pseudo-token for it. Processor sessions, and whole-model CUDA sessions (where
    /// the row goes in as the image path's rows do).
    pub fn step_embedding(&mut self, m: &Model, row: &[f32]) -> Result<(), Error> {
        if row.len() != m.cfg.dim {
            return Err(Error::Request(format!("an embedding of {} values for a model {} wide", row.len(), m.cfg.dim)));
        }
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            if g.is_split() {
                return Err(Error::Unsupported("embedding inputs on a model split between the graphics card and the processor".into()));
            }
            let pos = self.tokens.len();
            if pos + 1 >= self.max_ctx {
                return Err(Error::Request(format!("the {}-token context is full", self.max_ctx)));
            }
            let id = 0x8000_0000 | (pos as u32 & 0x7fff_ffff);
            g.image = Some(crate::gpu::ImagePrefill { from: pos, rope: Vec::new(), rows: vec![(pos, row.to_vec())] });
            g.hidden = Some(std::mem::take(&mut self.hidden));
            let r = g.prefill(m, &[id], pos, Some(&mut self.logits));
            g.image = None;
            self.hidden = g.hidden.take().unwrap_or_default();
            r?;
            self.tokens.push(id);
            return Ok(());
        }
        #[cfg(feature = "portable")]
        if self.portable.is_some() {
            return Err(Error::Unsupported("embedding inputs run on the processor for now".into()));
        }
        let pos = self.tokens.len();
        if pos + 1 >= self.max_ctx {
            return Err(Error::Request(format!("the {}-token context is full", self.max_ctx)));
        }
        crate::pool::for_writing(pos);
        self.x.copy_from_slice(row);
        self.layers_from(m, 0, pos);
        self.tokens.push(0x8000_0000 | (pos as u32 & 0x7fff_ffff));
        self.head(m);
        Ok(())
    }

    pub fn step(&mut self, m: &Model, token: u32) {
        crate::pool::for_writing(self.tokens.len());
        #[cfg(feature = "portable")]
        if let Some(p) = self.portable.as_mut() {
            let pos = self.tokens.len();
            if p.is_split() {
                // The GPU's layers, then the processor's layers and the output head here.
                let from = p.gpu_layers();
                let mut xs = Vec::new();
                if let Err(e) = p.run_split(m, &[token], pos, &mut xs) {
                    panic!("GPU step failed: {e}");
                }
                self.x.copy_from_slice(&xs);
                self.layers_from(m, from, pos);
                self.head(m);
                self.tokens.push(token);
                return;
            }
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
        if let Some(s) = self.hcs.as_mut() {
            s.token = token;
        }
        self.layers_from(m, 0, pos);
        self.head(m);
        self.tokens.push(token);
    }

    /// The rotary position of cache slot `slot`: inside an image span its 2D grid position, else the
    /// text position (the slot minus what earlier images saved).
    pub fn rope_pos(&self, slot: usize) -> [usize; 3] {
        rope_pos_in(&self.images, slot)
    }

    /// Layers `from`.. for the token at `pos`, on the activation in `self.x`. Ranges a worker on
    /// another computer runs (cluster.rs) go over the link.
    fn layers_from(&mut self, m: &Model, from: usize, pos: usize) {
        let Some(mut remote) = self.remote.take() else {
            return self.layers_range(m, from, m.cfg.n_layer, pos);
        };
        let mut li = from;
        let n = m.cfg.n_layer;
        while li < n {
            let next = remote.ranges.iter().map(|r| r.lo).filter(|&lo| lo >= li).min().unwrap_or(n);
            if next > li {
                self.layers_range(m, li, next, pos);
                li = next;
                continue;
            }
            let r = remote.starting_at(li).expect("a range starts here");
            if let Err(e) = r.forward(pos, &mut self.x) {
                self.remote = Some(remote);
                panic!("cluster step failed: {e}");
            }
            li = r.hi;
        }
        self.remote = Some(remote);
    }

    /// A worker's part (cluster.rs): layers lo..hi for consecutive positions from pos0, on `rows`
    /// (one activation per position), in place.
    pub fn forward_range(&mut self, m: &Model, lo: usize, hi: usize, pos0: usize, rows: &mut [f32]) {
        let dim = m.cfg.dim;
        for (i, row) in rows.chunks_exact_mut(dim).enumerate() {
            self.x.copy_from_slice(row);
            self.layers_range(m, lo, hi, pos0 + i);
            row.copy_from_slice(&self.x);
            if self.tokens.len() <= pos0 + i {
                self.tokens.push(0x8000_0000 | ((pos0 + i) as u32 & 0x7fff_ffff));
            }
        }
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
        let cs: Vec<(f32, f32)> = m.rope_cs(self.rope_pos(pos));
        if from == 0 {
            if let Some(s) = self.hcs.as_mut() {
                s.start(&self.x);
            }
        }
        for (li, l) in m.layers.iter().enumerate().take(to).skip(from) {
            m.stream_enter(li);
            self.block_in(m, li, l, pos, true);
            self.qa_dim.quantize(&self.h);
            if let Some(d) = &l.delta {
                let t = Instant::now();
                self.delta_step(m, li, d);
                self.prof[8] += t.elapsed().as_secs_f64();
                self.block_out(m);
                self.ffn_step(m, li, l);
                continue;
            }
            let mut tp = Instant::now();
            let mut lap = |i: usize, prof: &mut [f64; 10]| {
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
            crate::quant::store_f16(&mut self.k[li][pos * kvd..(pos + 1) * kvd], &self.kb);
            crate::quant::store_f16(&mut self.v[li][pos * kvd..(pos + 1) * kvd], &self.vb);
            // Attention, one work item per (KV head, span of up to 256 positions): every query head
            // of the group is computed from one read of each key and value (GQA shares them), and
            // long contexts split across threads; partial softmaxes are merged per head after.
            let (kc, vc, q) = (&self.k[li], &self.v[li], &self.q);
            let len = pos + 1;
            // Span length: about two work items per thread (16 items on 12 threads left most of them
            // idle for the second round at 512 positions), at most ATT_SPAN, at least 32.
            // Attention is arithmetic more than memory: it runs on the performance cores' threads.
            let reading = (crate::switches::get(crate::switches::CPU_THREADS).max(1) as usize).min(crate::pool::global().threads());
            let threads = reading;
            let span = (len * c.n_kv).div_ceil(2 * threads).next_multiple_of(16).clamp(32, ATT_SPAN);
            let spans = len.div_ceil(span);
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
            let hsplit = (threads / (c.n_kv * spans)).clamp(1, group);
            let hsplit = (1..=hsplit).rev().find(|d| group % d == 0).unwrap_or(1);
            let gper = group / hsplit;
            lap(1, &mut self.prof);
            crate::pool::global().run_limited(reading, c.n_kv * spans * hsplit, &|item| {
                let (hs, item) = (item % hsplit, item / hsplit);
                let (kvh, sp) = (item / spans, item % spans);
                let gr = hs * gper..(hs + 1) * gper;
                let (t0, t1) = (sp * span, ((sp + 1) * span).min(len));
                let n = t1 - t0;
                thread_local!(static SC: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) });
                SC.with(|sc| {
                    let mut sc = sc.borrow_mut();
                    sc.resize(group * ATT_SPAN, 0.0);
                    // Scores for every head of the group, key-major so each key is read once.
                    let kbase = &kc[t0 * kvd + kvh * hd..];
                    for g in gr.clone() {
                        let hh = kvh * group + g;
                        let row = &mut sc[g * ATT_SPAN..g * ATT_SPAN + n];
                        crate::quant::scores_f16(&q[hh * hd..(hh + 1) * hd], kbase, kvd, row);
                        for v in row.iter_mut() {
                            *v *= scale;
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
                    let vbase = &vc[t0 * kvd + kvh * hd..];
                    for g in gr.clone() {
                        let hh = kvh * group + g;
                        // SAFETY: as above.
                        let o = unsafe { std::slice::from_raw_parts_mut(parts.at((hh * spans + sp) * part), hd) };
                        crate::quant::weighted_f16(&sc[g * ATT_SPAN..g * ATT_SPAN + n], vbase, kvd, o);
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
            self.block_out(m);
            self.ffn_step(m, li, l);
        }
    }

    /// A block's input into `h`: the RMSNorm of the residual, or for hyper-connection models
    /// (after the n-gram embedding on its layer) the mixer over the streams. `attn` picks the
    /// token mixer's input, else the feed-forward's.
    fn block_in(&mut self, m: &Model, li: usize, l: &Layer, pos: usize, attn: bool) {
        let c = &m.cfg;
        match (m.hc.as_ref(), self.hcs.take()) {
            (Some(hm), Some(mut s)) => {
                if attn {
                    if let Some(p) = hm.ple.as_ref().filter(|p| p.layer == li) {
                        let t = Instant::now();
                        // Tokens 1..n-1 positions back; image slots hash as the image token.
                        let prev: Vec<Option<u32>> = (1..p.ngram).map(|k| pos.checked_sub(k).and_then(|i| self.tokens.get(i)).map(|&t| if t & 0x8000_0000 != 0 { p.image } else { t })).collect();
                        s.ple(m, p, hm.hc, c.dim, c.eps, &prev);
                        self.prof[9] += t.elapsed().as_secs_f64();
                    }
                }
                let t = Instant::now();
                s.mix(m, if attn { &hm.attn[li] } else { &hm.ffn[li] }, hm.hc, c.eps, &mut self.h);
                self.prof[7] += t.elapsed().as_secs_f64();
                self.hcs = Some(s);
            }
            (_, s) => {
                self.hcs = s;
                rmsnorm(&self.x, if attn { &l.attn_norm } else { &l.ffn_norm }, c.eps, &mut self.h);
            }
        }
    }

    /// Adds a block's output (`tmp`) to the residual (every stream, for hyper-connection models).
    fn block_out(&mut self, m: &Model) {
        let dim = m.cfg.dim;
        match (m.hc.as_ref(), self.hcs.as_mut()) {
            (Some(hm), Some(s)) => s.combine(&self.tmp[..dim], hm.hc),
            _ => {
                for i in 0..dim {
                    self.x[i] += self.tmp[i];
                }
            }
        }
    }

    /// Seconds spent per part of the forward pass since the session started (processor path).
    pub fn profile(&self) -> Vec<(&'static str, f64)> {
        let names = ["attention projections", "attention setup", "attention", "attention output", "expert routing", "feed-forward and experts", "output head", "hyper-connection mixers", "linear attention (DeltaNet)", "n-gram embeddings"];
        names.iter().zip(self.prof.iter()).map(|(n, &s)| (*n, s)).filter(|x| x.1 > 0.0).collect()
    }

    /// The output head: final norm and logits.
    fn head(&mut self, m: &Model) {
        let to = Instant::now();
        match (m.hc.as_ref(), self.hcs.take()) {
            (Some(hm), Some(mut s)) => {
                s.mix(m, &hm.head, hm.hc, m.cfg.eps, &mut self.x);
                self.hcs = Some(s);
            }
            (_, s) => {
                self.hcs = s;
                rmsnorm_inplace(&mut self.x, &m.out_norm, m.cfg.eps);
            }
        }
        self.hidden.copy_from_slice(&self.x);
        self.qa_dim.quantize(&self.x);
        m.stream_enter_output();
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
        if !self.images.is_empty() {
            return Err(Error::Unsupported("saving a chat with images is not available yet".into()));
        }
        #[cfg(feature = "portable")]
        if self.portable.is_some() {
            return Err(Error::Unsupported("saving the cache is not available on the portable GPU backend yet".into()));
        }
        if m.hc.is_some() {
            return Err(Error::Unsupported("saving the cache of a hyper-connection model is not available yet".into()));
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
                let (k, v) = (&self.k[li][..n * kvd], &self.v[li][..n * kvd]);
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
    /// `feed` continues from them. If the file fails partway, the session is left empty (no
    /// tokens, recurrent state cleared), never half old and half restored.
    pub fn load_kv(&mut self, m: &Model, r: &mut dyn std::io::Read) -> Result<usize, Error> {
        #[cfg(feature = "portable")]
        if self.portable.is_some() {
            return Err(Error::Unsupported("restoring a cache is not available on the portable GPU backend yet".into()));
        }
        // A file for another model or context is refused before anything is touched.
        let (n, kvd) = read_kv_header(m, r)?;
        if n == 0 || n >= self.max_ctx {
            return Err(Error::Format("the saved cache does not fit this context".into()));
        }
        let res = self.load_kv_body(m, r, n, kvd);
        if res.is_err() {
            self.tokens.clear();
            self.images.clear();
            #[cfg(feature = "cuda")]
            if let Some(g) = self.gpu.as_mut() {
                g.rope_shift = 0;
            }
            self.reset_recurrent();
            #[cfg(feature = "cuda")]
            if let Some(g) = self.gpu.as_mut() {
                let _ = g.reset_recurrent();
            }
        }
        res
    }

    fn load_kv_body(&mut self, m: &Model, r: &mut dyn std::io::Read, n: usize, kvd: usize) -> Result<usize, Error> {
        let c = &m.cfg;
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
                    self.k[li][..n * kvd].copy_from_slice(&k);
                    self.v[li][..n * kvd].copy_from_slice(&v);
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
        if let Some(s) = self.hcs.as_mut() {
            s.reset();
        }
    }

    /// The feed-forward half of a layer: norm, SwiGLU or experts, residual add.
    fn ffn_step(&mut self, m: &Model, li: usize, l: &Layer) {
        let c = &m.cfg;
        let mut tp = Instant::now();
        let mut lap = |i: usize, prof: &mut [f64; 10]| {
            let n = Instant::now();
            prof[i] += (n - tp).as_secs_f64();
            tp = n;
        };
        {
            self.block_in(m, li, l, 0, false);
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
                    self.block_out(m);
                }
                Ffn::Moe(e) => {
                    route(&e.router, &self.h, c.n_used, &mut self.router, &mut self.chosen);
                    lap(4, &mut self.prof);
                    for &(ex, _) in &self.chosen {
                        self.expert_counts[li * c.n_expert + ex] += 1;
                    }
                    m.moe_cpu(li, &self.h, &self.qa_dim, &self.chosen, &mut self.moe, &mut self.tmp[..c.dim]);
                    if let Some(sh) = &e.shared {
                        self.shared_expert(m, sh);
                        for i in 0..c.dim {
                            self.tmp[i] += self.sh_out[i];
                        }
                    }
                    self.block_out(m);
                }
            }
            lap(5, &mut self.prof);
        }
    }

    /// The shared expert on the block input `h`, into `sh_out`: SwiGLU, scaled by its gate.
    fn shared_expert(&mut self, m: &Model, sh: &SharedExpert) {
        m.matvec_many(&mut [(&sh.gate, &mut self.shg[..]), (&sh.up, &mut self.shu[..])], &self.h, &self.qa_dim);
        for i in 0..self.shg.len() {
            let g = self.shg[i];
            self.shg[i] = g / (1.0 + crate::quant::exp_fast(-g)) * self.shu[i];
        }
        self.qa_sh.quantize(&self.shg);
        m.matvec(&sh.down, &self.shg, &self.qa_sh, &mut self.sh_out);
        if let Some(w) = &sh.gate_inp {
            let gate = 1.0 / (1.0 + (-crate::quant::dot_f32(w, &self.h)).exp());
            self.sh_out.iter_mut().for_each(|v| *v *= gate);
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
                let gate = if c.delta_sigmoid { 1.0 / (1.0 + (-zj).exp()) } else { zj / (1.0 + (-zj).exp()) };
                o[j] = o[j] * r * d.norm[j] * gate;
            }
        });
        self.qa_dout.quantize(&self.dout);
        m.matvec(&d.out, &self.dout, &self.qa_dout, &mut self.tmp[..c.dim]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_slots_get_grid_positions_and_text_after_skips_ahead() {
        // Text 0..3, a 4x2 image at slots 3..11, then text.
        let ims = [ImageSpan { slot0: 3, nx: 4, ny: 2, linear: false }];
        assert_eq!(rope_pos_in(&ims, 2), [2, 2, 2]);
        assert_eq!(rope_pos_in(&ims, 3), [3, 3, 3]);
        assert_eq!(rope_pos_in(&ims, 4), [3, 3, 4]);
        assert_eq!(rope_pos_in(&ims, 7), [3, 4, 3]);
        assert_eq!(rope_pos_in(&ims, 10), [3, 4, 6]);
        // The image used max(4, 2) = 4 positions: text resumes at 3 + 4.
        assert_eq!(rope_pos_in(&ims, 11), [7, 7, 7]);
        let two = [ImageSpan { slot0: 3, nx: 4, ny: 2, linear: false }, ImageSpan { slot0: 12, nx: 1, ny: 3, linear: false }];
        assert_eq!(rope_pos_in(&two, 12), [8, 8, 8]);
        assert_eq!(rope_pos_in(&two, 14), [8, 10, 8]);
        assert_eq!(rope_pos_in(&two, 15), [11, 11, 11]);
    }
}
