//! Gemma 4 (Google, 2026; architecture "gemma4", e.g. Gemma 4 E4B): what it adds to a standard
//! decoder, as llama.cpp's reference reads it (src/models/gemma4.cpp):
//!
//! - Two layer kinds: sliding-window layers (attention over the last `swa` positions, 256-wide
//!   heads, rotary base `base_swa`) and global layers (every position, 512-wide heads, rotary base
//!   `base`, with frequency factors `rope_freqs`: factors of 1e30 leave those pairs unrotated).
//! - q and k are RMS-normed per head with weights, v without; attention is not scaled by 1/sqrt(d).
//! - Sandwich norms: the attention and feed-forward outputs are normed again before the residual.
//! - A GELU-gated feed-forward.
//! - Per-layer inputs: each token has a `ple`-wide vector per layer, from its own table plus a
//!   projection of the token embedding; each layer gates its output into that vector, projects it
//!   back to the model width, norms it and adds it.
//! - The last `n_layer - kv_layers` layers keep no KV cache: they attend over the cache of the
//!   last earlier layer of their kind.
//! - A per-layer output scale, tied embeddings (scaled by sqrt(dim) on the way in) and logits
//!   soft-capped at `softcap`.
//!
//! The standard weights (q/k/v/o, norms, gate/up/down) load into `model::Layer`; this holds the rest.

use crate::gguf::Gguf;
use crate::model::Mat;
use crate::Error;

/// One layer's Gemma 4 specifics.
pub struct G4Layer {
    /// Sliding-window (true) or global attention.
    pub swa: bool,
    /// Head width (q, k and v alike).
    pub hd: usize,
    /// The layer whose KV cache this layer attends over (itself unless it shares).
    pub kv_from: usize,
    pub post_attn_norm: Vec<f32>,
    pub post_ffn_norm: Vec<f32>,
    /// Per-layer input: dim -> ple gate, ple -> dim projection, and its norm.
    pub inp_gate: Mat,
    pub proj: Mat,
    pub post_norm: Vec<f32>,
    pub out_scale: f32,
}

pub struct G4Model {
    pub layers: Vec<G4Layer>,
    /// Per-layer token table (n_layer * ple per token) and the projection of the token embedding
    /// into the same space, with the norm applied to each layer's slice.
    pub ple_embd: Mat,
    pub ple_proj: Mat,
    pub ple_proj_norm: Vec<f32>,
    pub ple: usize,
    /// Rotary frequency factors of the global layers (head width / 2).
    pub rope_freqs: Vec<f32>,
    pub base: f32,
    pub base_swa: f32,
    pub swa: usize,
    pub softcap: f32,
    /// Layers 0..kv_layers keep a KV cache.
    pub kv_layers: usize,
    pub n_kv: usize,
    pub n_head: usize,
}

impl G4Model {
    pub fn load(g: &Gguf, n_layer: usize, eps_check: f32) -> Result<Self, Error> {
        let _ = eps_check;
        let k = |s: &str| format!("gemma4.{s}");
        let f32v = |name: &str| crate::model::vec_f32(g, name);
        let mat = |name: &str| g.tensor(name).map(Mat::of);
        let pattern: Vec<bool> = g
            .meta
            .get(&k("attention.sliding_window_pattern"))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_bool().unwrap_or(false)).collect())
            .ok_or_else(|| Error::Format("gemma4: no sliding_window_pattern".into()))?;
        if pattern.len() != n_layer {
            return Err(Error::Format(format!("gemma4: {} pattern entries for {n_layer} layers", pattern.len())));
        }
        let hd_full = g.u64(&k("attention.key_length"))? as usize;
        let hd_swa = g.u64(&k("attention.key_length_swa"))? as usize;
        let shared = g.u64(&k("attention.shared_kv_layers")).unwrap_or(0) as usize;
        let kv_layers = n_layer - shared.min(n_layer);
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let swa = pattern[i];
            // A sharing layer reads the last cache-keeping layer of its kind (llama.cpp:
            // n_layer_kv_from_start - (swa ? 2 : 1)).
            let kv_from = if i < kv_layers { i } else { kv_layers - if swa { 2 } else { 1 } };
            let out_scale = f32v(&format!("blk.{i}.layer_output_scale.weight")).ok().and_then(|v| v.first().copied()).unwrap_or(1.0);
            layers.push(G4Layer {
                swa,
                hd: if swa { hd_swa } else { hd_full },
                kv_from,
                post_attn_norm: f32v(&format!("blk.{i}.post_attention_norm.weight"))?,
                post_ffn_norm: f32v(&format!("blk.{i}.post_ffw_norm.weight"))?,
                inp_gate: mat(&format!("blk.{i}.inp_gate.weight"))?,
                proj: mat(&format!("blk.{i}.proj.weight"))?,
                post_norm: f32v(&format!("blk.{i}.post_norm.weight"))?,
                out_scale,
            });
        }
        Ok(Self {
            layers,
            ple_embd: mat("per_layer_token_embd.weight")?,
            ple_proj: mat("per_layer_model_proj.weight")?,
            ple_proj_norm: f32v("per_layer_proj_norm.weight")?,
            ple: g.u64(&k("embedding_length_per_layer_input"))? as usize,
            rope_freqs: f32v("rope_freqs.weight").unwrap_or_else(|_| vec![1.0; hd_full / 2]),
            base: g.f32(&k("rope.freq_base")).unwrap_or(1e6),
            base_swa: g.f32(&k("rope.freq_base_swa")).unwrap_or(1e4),
            swa: g.u64(&k("attention.sliding_window")).unwrap_or(512) as usize,
            softcap: g.f32(&k("final_logit_softcapping")).unwrap_or(0.0),
            kv_layers,
            n_kv: g.u64(&k("attention.head_count_kv"))? as usize,
            n_head: g.u64(&k("attention.head_count"))? as usize,
        })
    }
}

/// tanh-approximated GELU (ggml_gelu).
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
}

/// RMS norm of `x` in place, with weights `w` (empty: none).
pub fn rms(x: &mut [f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ss + eps).sqrt();
    if w.is_empty() {
        x.iter_mut().for_each(|v| *v *= s);
    } else {
        x.iter_mut().zip(w).for_each(|(v, w)| *v *= s * w);
    }
}

/// Rotate-halves rotary embedding of one head (`x.len()` = head width, all of it rotated):
/// frequency i = base^(-2i/d) / factor_i.
pub fn rope(x: &mut [f32], pos: usize, base: f32, factors: Option<&[f32]>) {
    let d = x.len();
    let half = d / 2;
    for i in 0..half {
        let mut f = base.powf(-2.0 * i as f32 / d as f32);
        if let Some(ff) = factors {
            f /= ff[i];
        }
        let (s, c) = (pos as f32 * f).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}
