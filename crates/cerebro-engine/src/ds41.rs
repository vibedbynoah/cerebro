//! DeepSeek-V4.1 (`deepseek41`) on the processor, one token at a time.
//!
//! - Four hyper-connected copies of the residual stream; before each sublayer a small matrix of
//!   the normed copies gives a pre-mix (collapse to one input), a post-mix (spread the output) and a
//!   Sinkhorn-normalised 4 x 4 copy mix. "Single pass": a layer's attention input collapses with the
//!   previous layer's FFN pre-mix, its FFN input with its own attention pre-mix.
//! - Attention: 64 query heads over one shared 512-wide key that is also the value (low-rank Q,
//!   grouped low-rank output, RoPE on the last 64 dims and undone on the output, a learned sink per
//!   head). Every layer sees a 128-token window; layers with a compress ratio (1 or 2) also see
//!   latent rows, one per `ratio` tokens, written by a source layer's compressor and picked per
//!   query by a lightning indexer (top 512, optionally within 2048 candidate blocks of 8).
//! - Engram on two layers: hashed 2-, 3- and 4-gram tables (48 x 16M rows of 256) gated into the
//!   copies.
//! - 384 experts, 6 used, sqrt(softplus) scores with a selection bias, clamped SwiGLU, one shared.
//!
//! Reference: llama.cpp b11982 src/models/deepseek41.cpp and deepseek4.cpp, llama-graph.cpp
//! (build_moe_ffn), llama-kv-cache-dsv4.cpp (which compressed rows a query sees).

use crate::gguf::Gguf;
use crate::model::{vec_f32, Mat, Model};
use crate::quant::{dequant_row, QAct};
use crate::Error;

/// Hyperparameters (GGUF `deepseek41.*`).
pub struct Hp {
    pub n_layer: usize,
    pub dim: usize,
    pub n_head: usize,
    pub hd: usize,
    pub n_rot: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub n_swa: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub w_scale: f32,
    pub w_norm: bool,
    pub clamp_exp: Vec<f32>,
    pub clamp_sh: Vec<f32>,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub top_k: usize,
    pub ratio: Vec<usize>,
    pub kv_src: Vec<Option<usize>>,
    pub topk_src: Vec<Option<usize>>,
    pub cand_src: Option<usize>,
    pub cand_blocks: usize,
    pub cand_bs: usize,
    pub hc: usize,
    pub sinkhorn: usize,
    pub hc_eps: f32,
    pub eps: f32,
    pub rope_base: f32,
    pub comp_base: f32,
    pub yarn_factor: f32,
    pub yarn_orig: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub ngram: usize,
    pub e_heads: usize,
    pub e_dim: usize,
    pub e_mult: Vec<u64>,
    pub e_off: Vec<u64>,
    pub e_prime: Vec<u64>,
    pub e_pad: i64,
}

impl Hp {
    /// Columns (hash heads) per Engram layer: (ngram - 1) shifts x heads per n-gram.
    fn e_cols(&self) -> usize {
        (self.ngram - 1) * self.e_heads
    }
}

/// The mix weights of one hyper-connection site.
struct Mix {
    /// (2 + hc) * hc rows of hc * dim.
    w: Vec<f32>,
    base: Vec<f32>,
    scale: [f32; 3],
}

struct Comp {
    wkv: Mat,
    wgate: Option<Mat>,
    norm: Vec<f32>,
    idx_k: Mat,
    idx_k_norm: Vec<f32>,
}

struct Indexer {
    q_b: Mat,
    proj: Mat,
}

struct Engram {
    kv: Mat,
    /// hc rows of dim.
    gate: Vec<f32>,
    module: usize,
}

struct Layer {
    attn_norm: Vec<f32>,
    sinks: Vec<f32>,
    q_a: Mat,
    q_a_norm: Vec<f32>,
    q_b: Mat,
    kv: Mat,
    kv_norm: Vec<f32>,
    wo_a: Mat,
    wo_b: Mat,
    mix_attn: Mix,
    mix_ffn: Mix,
    comp: Option<Comp>,
    idx: Option<Indexer>,
    engram: Option<Engram>,
    router: Mat,
    probs_b: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_e: Mat,
    up_e: Mat,
    down_e: Mat,
    gate_sh: Mat,
    up_sh: Mat,
    down_sh: Mat,
}

pub struct Ds41Model {
    pub hp: Hp,
    layers: Vec<Layer>,
    embed: Mat,
    output: Mat,
    out_norm: Vec<f32>,
    /// The Engram tables in layer order; hash row ids run across them back to back.
    tables: Vec<Mat>,
    /// Token id -> Engram id (-1: none).
    token_map: Vec<i64>,
}

fn arr<'a>(g: &'a Gguf, key: &str) -> Result<&'a [crate::gguf::Value], Error> {
    g.meta.get(key).and_then(|v| v.as_array()).ok_or_else(|| Error::Format(format!("missing {key}")))
}

fn arr_u64(g: &Gguf, key: &str) -> Result<Vec<u64>, Error> {
    Ok(arr(g, key)?.iter().map(|v| v.as_u64().unwrap_or(0)).collect())
}

fn arr_f32(g: &Gguf, key: &str) -> Result<Vec<f32>, Error> {
    Ok(arr(g, key)?.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
}

impl Ds41Model {
    pub fn load(g: &Gguf) -> Result<Self, Error> {
        let k = |s: &str| format!("deepseek41.{s}");
        let u = |s: &str| g.u64(&k(s)).map(|v| v as usize);
        let n_layer = u("block_count")?;
        let ratio_all = arr_u64(g, &k("attention.compress_ratios"))?;
        if ratio_all.len() < n_layer {
            return Err(Error::Format("compress_ratios shorter than block_count".into()));
        }
        let ratio: Vec<usize> = ratio_all[..n_layer].iter().map(|&v| v as usize).collect();
        let kv_layers: Vec<usize> = arr_u64(g, &k("attention.kv_source_layers"))?.into_iter().map(|v| v as usize).collect();
        let idx_layers: Vec<usize> = arr_u64(g, &k("attention.indexer.source_layers"))?.into_iter().map(|v| v as usize).collect();
        // Each compressed layer reads the KV rows and top-k of the latest source layer at or before it.
        let (mut kv_src, mut topk_src) = (vec![None; n_layer], vec![None; n_layer]);
        let (mut last_kv, mut last_idx) = (None, None);
        for il in 0..n_layer {
            if kv_layers.contains(&il) {
                last_kv = Some(il);
            }
            if idx_layers.contains(&il) {
                last_idx = Some(il);
            }
            if ratio[il] == 0 {
                continue;
            }
            match (last_kv, last_idx) {
                (Some(kv), Some(ix)) if ratio[kv] == ratio[il] && ratio[ix] == ratio[il] && ratio[il] <= 2 => {
                    kv_src[il] = Some(kv);
                    topk_src[il] = Some(ix);
                }
                _ => return Err(Error::Format(format!("layer {il}: compress ratio {} without a matching kv/index source", ratio[il]))),
            }
        }
        let cand_src = g.u64(&k("attention.indexer.candidate_source_layer")).ok().map(|v| v as usize);
        let e_layers: Vec<usize> = arr_u64(g, &k("ple.layers")).unwrap_or_default().into_iter().map(|v| v as usize).collect();
        let hp = Hp {
            n_layer,
            dim: u("embedding_length")?,
            n_head: u("attention.head_count")?,
            hd: u("attention.key_length")?,
            n_rot: u("rope.dimension_count")?,
            o_groups: u("attention.output_group_count")?,
            o_lora: u("attention.output_lora_rank")?,
            n_swa: u("attention.sliding_window")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            w_scale: g.f32(&k("expert_weights_scale")).unwrap_or(1.0),
            w_norm: g.meta.get(&k("expert_weights_norm")).and_then(|v| v.as_bool()).unwrap_or(false),
            clamp_exp: arr_f32(g, &k("swiglu_clamp_exp")).unwrap_or_else(|_| vec![0.0; n_layer]),
            clamp_sh: arr_f32(g, &k("swiglu_clamp_shexp")).or_else(|_| arr_f32(g, &k("swiglu_clamp_exp"))).unwrap_or_else(|_| vec![0.0; n_layer]),
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            top_k: u("attention.indexer.top_k")?,
            ratio,
            kv_src,
            topk_src,
            cand_src,
            cand_blocks: u("attention.indexer.candidate_top_k_blocks").unwrap_or(0),
            cand_bs: u("attention.indexer.candidate_block_size").unwrap_or(1).max(1),
            hc: u("hyper_connection.count")?,
            sinkhorn: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: g.f32(&k("hyper_connection.epsilon"))?,
            eps: g.f32(&k("attention.layer_norm_rms_epsilon"))?,
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
            comp_base: g.f32(&k("attention.compress_rope_freq_base"))?,
            yarn_factor: g.f32(&k("rope.scaling.factor")).unwrap_or(1.0),
            yarn_orig: g.u64(&k("rope.scaling.original_context_length")).unwrap_or(0) as f32,
            beta_fast: g.f32(&k("rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
            beta_slow: g.f32(&k("rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
            ngram: u("ple.ngram_size").unwrap_or(0),
            e_heads: u("ple.heads_per_ngram").unwrap_or(0),
            e_dim: u("embedding_length_per_layer_input").unwrap_or(0),
            e_mult: arr_u64(g, &k("ple.layer_multipliers")).unwrap_or_default(),
            e_off: arr_u64(g, &k("ple.head_offsets")).unwrap_or_default(),
            e_prime: arr_u64(g, &k("ple.head_vocab_sizes")).unwrap_or_default(),
            e_pad: 0,
        };
        if hp.hc != 4 || hp.n_head % hp.o_groups != 0 || hp.n_rot > hp.hd || hp.ratio.iter().any(|&r| r > 2) {
            return Err(Error::Unsupported("this DeepSeek-V4.1 shape".into()));
        }
        let t = |s: &str| g.tensor(s).map(Mat::of);
        let tl = |i: usize, s: &str| t(&format!("blk.{i}.{s}"));
        let vl = |i: usize, s: &str| vec_f32(g, &format!("blk.{i}.{s}"));
        let has = |i: usize, s: &str| g.tensors.contains_key(&format!("blk.{i}.{s}"));
        let mix = |i: usize, site: &str| -> Result<Mix, Error> {
            let sc = vl(i, &format!("hc_{site}_scale.weight"))?;
            Ok(Mix { w: vl(i, &format!("hc_{site}_fn.weight"))?, base: vl(i, &format!("hc_{site}_base.weight"))?, scale: [sc[0], sc[1], sc[2]] })
        };
        let mut layers = Vec::with_capacity(n_layer);
        let mut tables = Vec::new();
        for i in 0..n_layer {
            let comp = if hp.kv_src[i] == Some(i) {
                Some(Comp {
                    wkv: tl(i, "attn_compressor_kv.weight")?,
                    wgate: if hp.ratio[i] > 1 { Some(tl(i, "attn_compressor_gate.weight")?) } else { None },
                    norm: vl(i, "attn_compressor_norm.weight")?,
                    idx_k: tl(i, "indexer.attn_k.weight")?,
                    idx_k_norm: vl(i, "indexer.k_norm.weight")?,
                })
            } else {
                None
            };
            let idx = if hp.topk_src[i] == Some(i) { Some(Indexer { q_b: tl(i, "indexer.attn_q_b.weight")?, proj: tl(i, "indexer.proj.weight")? }) } else { None };
            let engram = if e_layers.contains(&i) && has(i, "engram_kv.weight") {
                tables.push(tl(i, "engram_embd.weight")?);
                Some(Engram { kv: tl(i, "engram_kv.weight")?, gate: vl(i, "engram_gate.weight")?, module: tables.len() - 1 })
            } else {
                None
            };
            layers.push(Layer {
                attn_norm: vl(i, "attn_norm.weight")?,
                sinks: vl(i, "attn_sinks.weight")?,
                q_a: tl(i, "attn_q_a.weight")?,
                q_a_norm: vl(i, "attn_q_a_norm.weight")?,
                q_b: tl(i, "attn_q_b.weight")?,
                kv: tl(i, "attn_kv.weight")?,
                kv_norm: vl(i, "attn_kv_a_norm.weight")?,
                wo_a: tl(i, "attn_output_a.weight")?,
                wo_b: tl(i, "attn_output_b.weight")?,
                mix_attn: mix(i, "attn")?,
                mix_ffn: mix(i, "ffn")?,
                comp,
                idx,
                engram,
                router: tl(i, "ffn_gate_inp.weight")?,
                probs_b: vl(i, "exp_probs_b.bias")?,
                ffn_norm: vl(i, "ffn_norm.weight")?,
                gate_e: tl(i, "ffn_gate_exps.weight")?,
                up_e: tl(i, "ffn_up_exps.weight")?,
                down_e: tl(i, "ffn_down_exps.weight")?,
                gate_sh: tl(i, "ffn_gate_shexp.weight")?,
                up_sh: tl(i, "ffn_up_shexp.weight")?,
                down_sh: tl(i, "ffn_down_shexp.weight")?,
            });
        }
        let token_map: Vec<i64> = match g.meta.get("deepseek41.engram.token_map").and_then(|v| v.as_array()) {
            Some(a) => a.iter().map(|v| v.as_u64().map_or(-1, |x| x as i64)).collect(),
            None => Vec::new(),
        };
        let mut hp = hp;
        if let Ok(pad) = g.u64("deepseek41.engram.pad_token_id") {
            hp.e_pad = token_map.get(pad as usize).copied().unwrap_or(0);
        }
        // Engram metadata must agree with the tables (else the layers run without it).
        let n_e = tables.len();
        let cols = if hp.ngram >= 2 { hp.e_cols() } else { 0 };
        if n_e > 0 && (cols == 0 || hp.e_mult.len() != n_e * hp.ngram || hp.e_off.len() != n_e * cols || hp.e_prime.len() != n_e * cols || token_map.is_empty()) {
            return Err(Error::Format("DeepSeek-V4.1 Engram metadata is inconsistent".into()));
        }
        Ok(Self {
            hp,
            layers,
            embed: t("token_embd.weight")?,
            output: t("output.weight")?,
            out_norm: vec_f32(g, "output_norm.weight")?,
            tables,
            token_map,
        })
    }

    /// The weights a token reads besides the experts (what is worth keeping in RAM).
    pub fn dense_ranges(&self) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        let mut push = |m: &Mat| v.push((m.offset, m.rows * m.row_bytes));
        for l in &self.layers {
            for m in [&l.q_a, &l.q_b, &l.kv, &l.wo_a, &l.wo_b, &l.router, &l.gate_sh, &l.up_sh, &l.down_sh] {
                push(m);
            }
            if let Some(c) = &l.comp {
                push(&c.wkv);
                push(&c.idx_k);
                if let Some(w) = &c.wgate {
                    push(w);
                }
            }
            if let Some(ix) = &l.idx {
                push(&ix.q_b);
                push(&ix.proj);
            }
            if let Some(e) = &l.engram {
                push(&e.kv);
            }
        }
        push(&self.output);
        v
    }
}

/// Per-session state: the windows, the compressed rows and the compressors' unfinished blocks.
pub struct State {
    /// Per layer: the last n_swa keys (hd each), slot pos % n_swa.
    win: Vec<Vec<f32>>,
    /// Per KV source layer: compressed rows of hd + idx_dim (latent, then index key).
    rows: Vec<Vec<f32>>,
    /// Per KV source layer: the current block's kv and score rows (ratio x hd each).
    pend_kv: Vec<Vec<f32>>,
    pend_sc: Vec<Vec<f32>>,
}

impl State {
    pub fn new(m: &Ds41Model) -> Self {
        let h = &m.hp;
        Self {
            win: vec![vec![0.0; h.n_swa * h.hd]; h.n_layer],
            rows: vec![Vec::new(); h.n_layer],
            pend_kv: (0..h.n_layer).map(|i| vec![0.0; h.ratio[i].max(1) * h.hd]).collect(),
            pend_sc: (0..h.n_layer).map(|i| vec![0.0; h.ratio[i].max(1) * h.hd]).collect(),
        }
    }

    pub fn reset(&mut self) {
        for r in &mut self.rows {
            r.clear();
        }
    }
}

/// x / sqrt(mean(x^2) + eps), times w when given.
fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let s = 1.0 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    match w {
        Some(w) => x.iter().zip(w).map(|(&v, &w)| v * s * w).collect(),
        None => x.iter().map(|&v| v * s).collect(),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One matrix times x.
fn mv(m: &Model, w: &Mat, x: &[f32]) -> Vec<f32> {
    let mut qa = QAct::new(x.len());
    qa.quantize(x);
    let mut out = vec![0f32; w.rows];
    m.matvec_pub(w, x, &qa, &mut out);
    out
}

/// RoPE cosines and sines for `n_rot` dims at position `p` (ggml's rope_yarn, NORM pairs): the
/// window-only layers use plain RoPE, the compressed ones YaRN with the compress base, whose
/// magnitude correction DeepSeek's attention factor cancels.
fn rope_cs(h: &Hp, compressed: bool, p: usize) -> Vec<(f32, f32)> {
    let n = h.n_rot;
    let (base, fscale, ext) = if compressed { (h.comp_base, 1.0 / h.yarn_factor, 1.0f32) } else { (h.rope_base, 1.0, 0.0) };
    let attn_factor = if ext == 0.0 { 1.0 } else { 1.0 / (1.0 + 0.1 * (1.0 / fscale).ln()) };
    let corr = |beta: f32| n as f32 * (h.yarn_orig / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
    let (lo, hi) = if compressed { ((corr(h.beta_fast)).floor().max(0.0), (corr(h.beta_slow)).ceil().min(n as f32 - 1.0)) } else { (0.0, 0.0) };
    let theta_scale = base.powf(-2.0 / n as f32);
    (0..n / 2)
        .map(|i| {
            let i0 = 2 * i;
            let extrap = p as f32 * theta_scale.powi(i as i32);
            let interp = fscale * extrap;
            let (mut theta, mut mscale) = (interp, attn_factor);
            if ext != 0.0 {
                let y = (i0 as f32 / 2.0 - lo) / (hi - lo).max(0.001);
                let ramp = (1.0 - y.clamp(0.0, 1.0)) * ext;
                theta = interp * (1.0 - ramp) + extrap * ramp;
                mscale *= 1.0 + 0.1 * (1.0 / fscale).ln();
            }
            (theta.cos() * mscale, theta.sin() * mscale)
        })
        .collect()
}

/// Rotates consecutive pairs of `x` (NORM mode); `inverse` undoes it.
fn rope(x: &mut [f32], cs: &[(f32, f32)], inverse: bool) {
    for (i, &(c, s)) in cs.iter().enumerate() {
        let s = if inverse { -s } else { s };
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// (pre, post, comb) of one hyper-connection site from the copies `hs` (hc x dim).
fn mixes(h: &Hp, mix: &Mix, hs: &[f32]) -> ([f32; 4], [f32; 4], [f32; 16]) {
    let flat = rms(hs, None, h.eps);
    let n = h.hc * h.dim;
    let mut m = [0f32; 24];
    for (k, v) in m.iter_mut().enumerate() {
        *v = dot(&mix.w[k * n..(k + 1) * n], &flat);
    }
    let mut pre = [0f32; 4];
    let mut post = [0f32; 4];
    for i in 0..4 {
        pre[i] = sigmoid(m[i] * mix.scale[0] + mix.base[i]) + h.hc_eps;
        post[i] = 2.0 * sigmoid(m[4 + i] * mix.scale[1] + mix.base[4 + i]);
    }
    // comb[dst + 4 src]: softmax over dst for each src, then Sinkhorn (columns: sum over src per dst).
    let mut c = [0f32; 16];
    for i in 0..16 {
        c[i] = m[8 + i] * mix.scale[2] + mix.base[8 + i];
    }
    for s in 0..4 {
        let mx = (0..4).map(|d| c[d + 4 * s]).fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = (0..4).map(|d| (c[d + 4 * s] - mx).exp()).sum();
        for d in 0..4 {
            c[d + 4 * s] = (c[d + 4 * s] - mx).exp() / sum + h.hc_eps;
        }
    }
    let norm_cols = |c: &mut [f32; 16]| {
        for d in 0..4 {
            let sum: f32 = (0..4).map(|s| c[d + 4 * s]).sum::<f32>() + h.hc_eps;
            for s in 0..4 {
                c[d + 4 * s] /= sum;
            }
        }
    };
    let norm_rows = |c: &mut [f32; 16]| {
        for s in 0..4 {
            let sum: f32 = (0..4).map(|d| c[d + 4 * s]).sum::<f32>() + h.hc_eps;
            for d in 0..4 {
                c[d + 4 * s] /= sum;
            }
        }
    };
    norm_cols(&mut c);
    for _ in 1..h.sinkhorn {
        norm_rows(&mut c);
        norm_cols(&mut c);
    }
    (pre, post, c)
}

/// sum_h hs[h] * w[h].
fn collapse(hs: &[f32], w: &[f32; 4], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    for (h, &wh) in w.iter().enumerate() {
        for (o, &v) in out.iter_mut().zip(&hs[h * dim..(h + 1) * dim]) {
            *o += v * wh;
        }
    }
    out
}

/// New copies: x * post[d] + sum_s hs[s] * comb[d + 4 s].
fn spread(x: &[f32], hs: &[f32], post: &[f32; 4], comb: &[f32; 16], dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; 4 * dim];
    for d in 0..4 {
        let o = &mut out[d * dim..(d + 1) * dim];
        for i in 0..dim {
            let mut v = x[i] * post[d];
            for s in 0..4 {
                v += hs[s * dim + i] * comb[d + 4 * s];
            }
            o[i] = v;
        }
    }
    out
}

/// sqrt(softplus(x)).
fn sqrt_softplus(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
    sp.sqrt()
}

/// Indices of the `k` largest values (ties: lower index first).
fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(v.len());
    if k < v.len() {
        idx.select_nth_unstable_by(k, |&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        idx.truncate(k);
    }
    idx
}

impl Ds41Model {
    /// The Engram rows of `token` at `pos` (with up to ngram - 1 previous tokens): e_cols rows of
    /// e_dim, back to back.
    fn engram_rows(&self, g: &Gguf, module: usize, prev: &[u32], token: u32, pos: usize) -> Vec<f32> {
        let h = &self.hp;
        let (n_gram, cols) = (h.ngram, h.e_cols());
        let cid = |t: u32| self.token_map.get(t as usize).copied().unwrap_or(-1);
        // Shift 0 is the token itself; once a shift is blocked (sequence start or an unmapped token)
        // every longer one is too.
        let mut toks = vec![0u64; n_gram];
        let mut c = cid(token);
        let mut blocked = c < 0;
        toks[0] = if blocked { h.e_pad as u64 } else { c as u64 };
        for s in 1..n_gram {
            c = if pos < s || prev.len() < s { -1 } else { cid(prev[prev.len() - s]) };
            blocked = blocked || c < 0;
            toks[s] = if blocked { h.e_pad as u64 } else { c as u64 };
        }
        let mult = &h.e_mult[module * n_gram..(module + 1) * n_gram];
        let primes = &h.e_prime[module * cols..(module + 1) * cols];
        let offs = &h.e_off[module * cols..(module + 1) * cols];
        let mut out = vec![0f32; cols * h.e_dim];
        let mut rolling = toks[0].wrapping_mul(mult[0]);
        for o in 1..n_gram {
            rolling ^= toks[o].wrapping_mul(mult[o]);
            for hh in 0..h.e_heads {
                let col = (o - 1) * h.e_heads + hh;
                let mut id = (rolling % primes[col] + offs[col]) as usize;
                // The id runs across the tables in layer order.
                let mut t = 0;
                while t + 1 < self.tables.len() && id >= self.tables[t].rows {
                    id -= self.tables[t].rows;
                    t += 1;
                }
                let tab = &self.tables[t];
                let id = id.min(tab.rows - 1);
                let bytes = g.bytes_at(tab.offset + id * tab.row_bytes, tab.row_bytes);
                dequant_row(tab.dtype, bytes, &mut out[col * h.e_dim..(col + 1) * h.e_dim]);
            }
        }
        out
    }
}

/// One token at `pos` (`prev`: the tokens before it): leaves the logits in `logits`.
pub fn step(model: &Model, d: &Ds41Model, st: &mut State, prev: &[u32], token: u32, pos: usize, logits: &mut [f32]) {
    let h = &d.hp;
    let (dim, hc, hd, nh) = (h.dim, h.hc, h.hd, h.n_head);
    let g = model.gguf();
    let mut x = vec![0f32; dim];
    dequant_row(d.embed.dtype, g.bytes_at(d.embed.offset + token as usize * d.embed.row_bytes, d.embed.row_bytes), &mut x);
    let mut hs: Vec<f32> = (0..hc).flat_map(|_| x.iter().copied()).collect();
    let mut pre_prev: Option<[f32; 4]> = None;
    // Per-token selections: each index source's top-k, and the candidate blocks.
    let mut topk_of: Vec<Option<Vec<usize>>> = vec![None; h.n_layer];
    let mut cand: Option<Vec<bool>> = None;
    let n_nope = hd - h.n_rot;
    let cs_raw = rope_cs(h, false, pos);
    let cs_comp = rope_cs(h, true, pos);
    for (il, l) in d.layers.iter().enumerate() {
        if let Some(e) = &l.engram {
            let rows = d.engram_rows(g, e.module, prev, token, pos);
            let kv = mv(model, &e.kv, &rows);
            let value = &kv[hc * dim..(hc + 1) * dim];
            for c in 0..hc {
                let xs = rms(&hs[c * dim..(c + 1) * dim], None, h.eps);
                let ks = rms(&kv[c * dim..(c + 1) * dim], None, h.eps);
                let gw = &e.gate[c * dim..(c + 1) * dim];
                let dt: f32 = xs.iter().zip(&ks).zip(gw).map(|((a, b), w)| a * b * w).sum::<f32>() / (dim as f32).sqrt();
                let gate = sigmoid(dt.abs().max(1e-6).sqrt() * if dt < 0.0 { -1.0 } else { 1.0 });
                for (o, &v) in hs[c * dim..(c + 1) * dim].iter_mut().zip(value) {
                    *o += v * gate;
                }
            }
        }
        // Attention.
        let (a_pre, a_post, a_comb) = mixes(h, &l.mix_attn, &hs);
        let cur = match pre_prev {
            Some(w) => collapse(&hs, &w, dim),
            None => hs[..dim].to_vec(),
        };
        let cur = rms(&cur, Some(&l.attn_norm), h.eps);
        let ratio = h.ratio[il];
        let cs = if ratio != 0 { &cs_comp } else { &cs_raw };
        let qr = rms(&mv(model, &l.q_a, &cur), Some(&l.q_a_norm), h.eps);
        let mut q = mv(model, &l.q_b, &qr);
        for hh in 0..nh {
            rope(&mut q[hh * hd + n_nope..(hh + 1) * hd], cs, false);
        }
        let mut kvv = rms(&mv(model, &l.kv, &cur), Some(&l.kv_norm), h.eps);
        rope(&mut kvv[n_nope..], cs, false);
        let slot = pos % h.n_swa;
        st.win[il][slot * hd..(slot + 1) * hd].copy_from_slice(&kvv);
        // The compressor: one latent row per `ratio` tokens, softmax-weighted per dimension.
        if let Some(c) = &l.comp {
            let skv = mv(model, &c.wkv, &cur);
            let ssc = match &c.wgate {
                Some(w) => mv(model, w, &cur),
                None => vec![0f32; hd],
            };
            let j = pos % ratio;
            st.pend_kv[il][j * hd..(j + 1) * hd].copy_from_slice(&skv);
            st.pend_sc[il][j * hd..(j + 1) * hd].copy_from_slice(&ssc);
            if (pos + 1) % ratio == 0 {
                let mut comp = vec![0f32; hd];
                for (dd, o) in comp.iter_mut().enumerate() {
                    let mx = (0..ratio).map(|t| st.pend_sc[il][t * hd + dd]).fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    let mut acc = 0f32;
                    for t in 0..ratio {
                        let w = (st.pend_sc[il][t * hd + dd] - mx).exp();
                        sum += w;
                        acc += w * st.pend_kv[il][t * hd + dd];
                    }
                    *o = acc / sum;
                }
                let mut comp = rms(&comp, Some(&c.norm), h.eps);
                let mut ik = rms(&mv(model, &c.idx_k, &comp), Some(&c.idx_k_norm), h.eps);
                // A latent stands for the first token of its block.
                let start = pos + 1 - ratio;
                let cw = rope_cs(h, true, start);
                let id = h.idx_dim;
                rope(&mut ik[id - h.n_rot..], &cw, false);
                rope(&mut comp[n_nope..], &cw, false);
                st.rows[il].extend_from_slice(&comp);
                st.rows[il].extend_from_slice(&ik);
            }
        }
        // The keys: the window, then the selected compressed rows.
        let first = (pos + 1).saturating_sub(h.n_swa);
        let mut keys: Vec<&[f32]> = (first..=pos).map(|p| &st.win[il][(p % h.n_swa) * hd..(p % h.n_swa + 1) * hd]).collect();
        let row_w = hd + h.idx_dim;
        let n_vis = if ratio != 0 { (pos + 1) / ratio } else { 0 };
        if n_vis > 0 {
            let src = h.kv_src[il].expect("compressed layer");
            let tsrc = h.topk_src[il].expect("compressed layer");
            if tsrc == il {
                let ix = l.idx.as_ref().expect("index source");
                let rows = &st.rows[src];
                let n_blk = n_vis.div_ceil(h.cand_bs);
                let pick_blk = h.cand_src == Some(il) && h.cand_blocks > 0 && n_blk > h.cand_blocks;
                let use_blk = h.cand_src.is_some_and(|c| c < il) && cand.is_some();
                let sel: Vec<usize> = if n_vis <= h.top_k && !pick_blk {
                    (0..n_vis).collect()
                } else {
                    let mut iq = mv(model, &ix.q_b, &qr);
                    let id = h.idx_dim;
                    for hh in 0..h.idx_heads {
                        rope(&mut iq[hh * id + id - h.n_rot..(hh + 1) * id], cs, false);
                    }
                    let scale = 1.0 / ((id * h.idx_heads) as f32).sqrt();
                    let iw: Vec<f32> = mv(model, &ix.proj, &cur).into_iter().map(|v| v * scale).collect();
                    let mut score: Vec<f32> = (0..n_vis)
                        .map(|j| {
                            let k = &rows[j * row_w + hd..(j + 1) * row_w];
                            (0..h.idx_heads).map(|hh| dot(&iq[hh * id..(hh + 1) * id], k).max(0.0) * iw[hh]).sum()
                        })
                        .collect();
                    if pick_blk {
                        // Best row per block, the newest block pinned, keep the top blocks.
                        let mut bmax: Vec<f32> = (0..n_blk).map(|b| score[b * h.cand_bs..((b + 1) * h.cand_bs).min(n_vis)].iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
                        bmax[(n_vis - 1) / h.cand_bs] += 1e30;
                        let mut keep = vec![false; n_blk];
                        for b in top_k(&bmax, h.cand_blocks) {
                            keep[b] = true;
                        }
                        cand = Some(keep);
                    } else if use_blk {
                        let keep = cand.as_ref().expect("checked");
                        for (j, s) in score.iter_mut().enumerate() {
                            if !keep.get(j / h.cand_bs).copied().unwrap_or(false) {
                                *s = f32::NEG_INFINITY;
                            }
                        }
                    }
                    let mut t = top_k(&score, h.top_k);
                    t.sort_unstable();
                    t
                };
                topk_of[il] = Some(sel);
            }
            let rows = &st.rows[src];
            for &j in topk_of[tsrc].as_deref().unwrap_or(&[]) {
                if j < n_vis {
                    keys.push(&rows[j * row_w..j * row_w + hd]);
                }
            }
        }
        // Attention with sinks; the value is the key.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; nh * hd];
        {
            use rayon::prelude::*;
            out.par_chunks_mut(hd).enumerate().for_each(|(hh, o)| {
                let qh = &q[hh * hd..(hh + 1) * hd];
                let s: Vec<f32> = keys.iter().map(|k| dot(qh, k) * scale).collect();
                let sink = l.sinks[hh];
                let mx = s.iter().copied().fold(sink, f32::max);
                let w: Vec<f32> = s.iter().map(|&v| (v - mx).exp()).collect();
                let denom = w.iter().sum::<f32>() + (sink - mx).exp();
                for (k, &wk) in keys.iter().zip(&w) {
                    let a = wk / denom;
                    for (o, &kv) in o.iter_mut().zip(k.iter()) {
                        *o += a * kv;
                    }
                }
                rope(&mut o[n_nope..], cs, true);
            });
        }
        // Grouped low-rank output.
        let gw = nh / h.o_groups * hd;
        let mut oa = vec![0f32; h.o_groups * h.o_lora];
        for gi in 0..h.o_groups {
            let part = l.wo_a.part(gi, h.o_groups);
            let r = mv(model, &part, &out[gi * gw..(gi + 1) * gw]);
            oa[gi * h.o_lora..(gi + 1) * h.o_lora].copy_from_slice(&r);
        }
        let attn_out = mv(model, &l.wo_b, &oa);
        hs = spread(&attn_out, &hs, &a_post, &a_comb, dim);
        // Feed-forward.
        let (f_pre, f_post, f_comb) = mixes(h, &l.mix_ffn, &hs);
        let cur = rms(&collapse(&hs, &a_pre, dim), Some(&l.ffn_norm), h.eps);
        pre_prev = Some(f_pre);
        let ffn_out = moe(model, h, l, il, &cur);
        hs = spread(&ffn_out, &hs, &f_post, &f_comb, dim);
    }
    let cur = rms(&collapse(&hs, &pre_prev.expect("at least one layer"), dim), Some(&d.out_norm), h.eps);
    let lg = mv(model, &d.output, &cur);
    logits[..lg.len()].copy_from_slice(&lg);
}

/// The experts and the shared expert on `cur`.
fn moe(model: &Model, h: &Hp, l: &Layer, il: usize, cur: &[f32]) -> Vec<f32> {
    let dim = h.dim;
    let logits = mv(model, &l.router, cur);
    let probs: Vec<f32> = logits.iter().map(|&v| sqrt_softplus(v)).collect();
    let biased: Vec<f32> = probs.iter().zip(&l.probs_b).map(|(p, b)| p + b).collect();
    let sel = top_k(&biased, h.n_used);
    let mut w: Vec<f32> = sel.iter().map(|&e| probs[e]).collect();
    if h.w_norm && h.n_used > 1 {
        let sum: f32 = w.iter().sum::<f32>() + 1e-20;
        w.iter_mut().for_each(|v| *v /= sum);
    }
    w.iter_mut().for_each(|v| *v *= h.w_scale);
    let ne = h.n_expert;
    let gates: Vec<Mat> = sel.iter().map(|&e| l.gate_e.part(e, ne)).chain([l.gate_sh.clone()]).collect();
    let ups: Vec<Mat> = sel.iter().map(|&e| l.up_e.part(e, ne)).chain([l.up_sh.clone()]).collect();
    let downs: Vec<Mat> = sel.iter().map(|&e| l.down_e.part(e, ne)).chain([l.down_sh.clone()]).collect();
    let n = gates.len();
    let ff = gates[0].rows;
    let mut gbuf = vec![0f32; n * ff];
    let mut ubuf = vec![0f32; n * ff];
    let mut qa = QAct::new(cur.len());
    qa.quantize(cur);
    {
        let mut jobs: Vec<(&Mat, &mut [f32])> = Vec::with_capacity(2 * n);
        for (i, (gc, uc)) in gbuf.chunks_mut(ff).zip(ubuf.chunks_mut(ff)).enumerate() {
            jobs.push((&gates[i], gc));
            jobs.push((&ups[i], uc));
        }
        model.matvec_many_pub(&mut jobs, cur, &qa);
    }
    let acts: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let lim = if i + 1 == n { h.clamp_sh[il] } else { h.clamp_exp[il] };
            (0..ff)
                .map(|j| {
                    let (mut g, mut u) = (gbuf[i * ff + j], ubuf[i * ff + j]);
                    if lim > 1e-6 {
                        g = g.min(lim);
                        u = u.clamp(-lim, lim);
                    }
                    g / (1.0 + (-g).exp()) * u
                })
                .collect()
        })
        .collect();
    let qas: Vec<QAct> = acts
        .iter()
        .map(|a| {
            let mut q = QAct::new(a.len());
            q.quantize(a);
            q
        })
        .collect();
    let mut outs = vec![0f32; n * dim];
    {
        let mut jobs: Vec<(&Mat, &[f32], &QAct, &mut [f32])> = Vec::with_capacity(n);
        for (i, oc) in outs.chunks_mut(dim).enumerate() {
            jobs.push((&downs[i], &acts[i], &qas[i], oc));
        }
        model.matvec_multi_pub(&mut jobs);
    }
    let mut y = vec![0f32; dim];
    for i in 0..n {
        let wi = if i + 1 == n { 1.0 } else { w[i] };
        for (o, &v) in y.iter_mut().zip(&outs[i * dim..(i + 1) * dim]) {
            *o += wi * v;
        }
    }
    y
}
