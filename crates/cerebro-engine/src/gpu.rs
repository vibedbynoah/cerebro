//! CUDA backend. Kernels are Cerebro's own (kernels.cu), compiled to PTX at build time and embedded,
//! so users need only the NVIDIA driver. Weights are repacked on upload into an aligned layout (int8
//! values in one array, one f32 scale per 32 values in another); Q/K/V and gate/up are concatenated
//! into one matrix each. The token id and position live in device memory, so a whole decode step is
//! captured once as a CUDA graph and replayed per token: no per-kernel launch cost, no logits copy for
//! greedy decoding (the next token is picked on the GPU).
//!
//! Dense models larger than the free VRAM are split by layer (placement.rs plans it): the first N
//! layers run here in the file's own K-quant format, their output activation is copied to the host,
//! and the CPU runs the remaining layers and the output head. A model that fits runs entirely here,
//! in the int8 layout when that fits too (fastest prompt reading), else in its own format.
//!
//! Mixture-of-experts models run hybrid: attention, router, embeddings and output on the GPU; the
//! experts live in RAM and the hottest ones are also cached in VRAM in their native K-quant format.
//! Per layer the GPU picks the experts, cached ones run on the GPU while the CPU computes the rest
//! at the same time, and the results are summed on the GPU. Expert use is counted, the cache is
//! rebalanced towards the most-used experts as generation runs, and the counts are saved next to the
//! model so the next session starts with the right experts resident.

use crate::switches;
use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::gguf::Gguf;
use crate::model::{Ffn, Mat, Model, MoeScratch};
use crate::quant::{dequant_row, f16_to_f32, DType, QAct};
use crate::Error;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
/// Batches smaller than this use the dp4a kernel (an MMA tile of 64 tokens would be mostly padding).
const MMA_MIN_T: usize = 8;
const PTX_MMA: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels_mma.ptx"));

/// A weight matrix on the GPU: repacked to int8 (`Q8`), or kept in its K-quant format as row-stacked
/// parts (`K`; fused Q/K/V can mix formats, e.g. Q4_K and Q6_K).
enum GMat {
    Q8 { qs: CudaSlice<i8>, ds: CudaSlice<f32>, rows: usize, cols: usize },
    K { parts: Vec<KPart>, cols: usize },
}

struct KPart {
    w: CudaSlice<u8>,
    dtype: i32,
    bb: i32,
    rows: usize,
}

impl GMat {
    fn q8(&self) -> (&CudaSlice<i8>, &CudaSlice<f32>, usize, usize) {
        match self {
            GMat::Q8 { qs, ds, rows, cols } => (qs, ds, *rows, *cols),
            GMat::K { .. } => panic!("batched kernels need the int8 layout"),
        }
    }
}

struct GLayer {
    attn_norm: CudaSlice<f32>,
    qkv: GMat,
    o: GMat,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    ffn_norm: CudaSlice<f32>,
    /// Dense feed-forward (None for MoE layers).
    gu: Option<GMat>,
    down: Option<GMat>,
    /// MoE router weights (n_expert x dim, f32).
    router: Option<CudaSlice<f32>>,
    /// f16 KV cache, max_ctx positions (one element for DeltaNet layers, which keep a state instead).
    kc: CudaSlice<u16>,
    vc: CudaSlice<u16>,
    /// Gated DeltaNet layer (Qwen3.5/3.8): then `qkv` is the fused [q|k|v | z | alpha | beta]
    /// projection and `o` the output projection.
    delta: Option<GDelta>,
}

/// A Gated DeltaNet layer's small tensors and recurrent state on the GPU.
struct GDelta {
    a: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    conv: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    /// The last kc-1 convolution inputs per channel, and the nv x dk x dv state.
    hist: CudaSlice<f32>,
    state: CudaSlice<f32>,
}

struct Funcs {
    /// Tensor-core flash prefill attention for head size 256 (None below sm_80 or switched off).
    attn_mma256: Option<CudaFunction>,
    delta_conv: CudaFunction,
    delta_l2: CudaFunction,
    delta_rule: CudaFunction,
    delta_gnorm: CudaFunction,
    unpack_qgate: CudaFunction,
    gate_mul: CudaFunction,
    matvec: CudaFunction,
    rmsnorm_quant: CudaFunction,
    quant: CudaFunction,
    qkv_post: CudaFunction,
    attn_partial: CudaFunction,
    attn_combine: CudaFunction,
    /// Grouped-query decode attention (one block per KV head and split); None when switched off.
    attn_gqa: Option<CudaFunction>,
    silu_mul_quant: CudaFunction,
    embed: CudaFunction,
    argmax: CudaFunction,
    embed_b: CudaFunction,
    norm_b: CudaFunction,
    matmul_b: CudaFunction,
    /// Tensor-core prefill matmul (sm_80+); None on older GPUs or with CEREBRO_NO_MMA set.
    matmul_mma: Option<CudaFunction>,
    /// Tensor-core flash attention for prefill, head size 128 (sm_80+).
    attn_mma: Option<CudaFunction>,
    silu_b: CudaFunction,
    attn_pre: CudaFunction,
    moe_matvec: CudaFunction,
    router_topk: CudaFunction,
    moe_silu: CudaFunction,
    moe_combine: CudaFunction,
    matvec_k: CudaFunction,
    kmatmul: CudaFunction,
    router_b: CudaFunction,
    scatter_add: CudaFunction,
    add_inplace: CudaFunction,
    attn_tiled: CudaFunction,
}

/// GPU-side block size of a K-quant format (Q6_K padded to 224 bytes for aligned loads) and its GGML
/// type id, or None if experts in this format cannot be cached on the GPU.
fn gpu_block(dt: DType) -> Option<(usize, i32)> {
    match dt {
        DType::Q4K => Some((144, 12)),
        DType::Q5K => Some((176, 13)),
        DType::Q6K => Some((224, 14)),
        _ => None,
    }
}

fn to_gpu_layout(dt: DType, src: &[u8]) -> Vec<u8> {
    if dt != DType::Q6K {
        return src.to_vec();
    }
    let mut out = vec![0u8; src.len() / 210 * 224];
    for (i, b) in src.chunks_exact(210).enumerate() {
        out[i * 224..i * 224 + 210].copy_from_slice(b);
    }
    out
}

/// One matrix kind (gate, up or down) of the stacked experts.
#[derive(Clone, Copy)]
struct ExpMat {
    dtype: i32,
    bb: i32,
    rows: usize,
    cols: usize,
    /// Bytes of one expert on the GPU.
    bytes: usize,
}

/// VRAM expert cache and per-layer MoE scratch for hybrid decoding.
struct MoeGpu {
    ne: usize,
    k: usize,
    eff: usize,
    /// Per layer: the gate, up and down formats (files mix formats across layers), or None when that
    /// layer's experts cannot be cached on the GPU.
    kinds: Vec<Option<[ExpMat; 3]>>,
    pool: CudaSlice<u8>,
    slot_bytes: usize,
    /// slot_of[layer * ne + expert] = cache slot; owner[slot] = layer * ne + expert.
    slot_of: Vec<Option<u32>>,
    owner: Vec<usize>,
    /// Use counts per (layer, expert), loaded from and saved to the model's profile.
    counts: Vec<f32>,
    profile: std::path::PathBuf,
    pub hits: u64,
    pub lookups: u64,
    /// Seconds spent waiting for the GPU's routing result, and computing CPU experts.
    pub wait_s: f64,
    pub cpu_s: f64,
    steps: u64,
    /// Device copy of slot_of (-1 = not cached), read by the router kernel.
    slot_tab: CudaSlice<i32>,
    // Device scratch. sel = [ids k | weights k | slots k].
    h: CudaSlice<f32>,
    sel: CudaSlice<i32>,
    gu: CudaSlice<f32>,
    axq: CudaSlice<i8>,
    axd: CudaSlice<f32>,
    eo: CudaSlice<f32>,
    cpu: CudaSlice<f32>,
    // Host scratch; pinned so the per-layer copies are asynchronous.
    sel_h: PinnedHostSlice<i32>,
    h_pin: PinnedHostSlice<f32>,
    cpu_pin: PinnedHostSlice<f32>,
    h_h: Vec<f32>,
    // Prompt-processing scratch (CHUNK tokens).
    bh: CudaSlice<f32>,
    bsel: CudaSlice<i32>,
    bidx: CudaSlice<i32>,
    bw: CudaSlice<f32>,
    bgu: CudaSlice<f32>,
    baxq: CudaSlice<i8>,
    baxd: CudaSlice<f32>,
    beo: CudaSlice<f32>,
    bcpu: CudaSlice<f32>,
    /// Streaming of uncached experts for prompt processing: a second stream copies expert weights
    /// from pinned host slots into a small VRAM ring while the compute stream works on the previous
    /// ones (the GPU computes a batch of tokens far faster than the CPU can).
    copy: Arc<CudaStream>,
    ring_dev: CudaSlice<u8>,
    ring_host: Vec<PinnedHostSlice<u8>>,
    ring_done: Vec<Option<cudarc::driver::CudaEvent>>,
    qa: QAct,
    scratch: MoeScratch,
    cpu_out: Vec<f32>,
}

impl MoeGpu {
    fn expert_bytes<'a>(model: &'a Model, li: usize, e: usize) -> [(DType, &'a [u8]); 3] {
        let Ffn::Moe(m) = &model.layers()[li].ffn else { unreachable!() };
        let ne = model.cfg.n_expert;
        let g = model.gguf();
        let one = |mat: &Mat| {
            let p = mat.part(e, ne);
            (p.dtype, g.bytes_at(p.offset, p.rows * p.row_bytes))
        };
        [one(&m.gate), one(&m.up), one(&m.down)]
    }

    /// The file byte ranges of one expert (gate, up, down).
    fn expert_ranges(model: &Model, li: usize, e: usize) -> [(usize, usize); 3] {
        let Ffn::Moe(m) = &model.layers()[li].ffn else { unreachable!() };
        let ne = model.cfg.n_expert;
        let r = |mat: &Mat| {
            let p = mat.part(e, ne);
            (p.offset, p.rows * p.row_bytes)
        };
        [r(&m.gate), r(&m.up), r(&m.down)]
    }

    fn upload(&mut self, stream: &Arc<CudaStream>, model: &Model, slot: usize, key: usize) -> Result<(), Error> {
        let (li, e) = (key / self.ne, key % self.ne);
        if let Some(&old) = self.owner.get(slot) {
            if old != usize::MAX {
                self.slot_of[old] = None;
                // The evicted expert is computed on the CPU from now on: map it in.
                model.gguf().touch(&Self::expert_ranges(model, old / self.ne, old % self.ne));
                stream.memcpy_htod(&[-1i32], &mut self.slot_tab.slice_mut(old..old + 1)).map_err(gerr)?;
            }
        }
        let mut off = slot * self.slot_bytes;
        for (dt, src) in Self::expert_bytes(model, li, e) {
            let bytes = to_gpu_layout(dt, src);
            let mut view = self.pool.slice_mut(off..off + bytes.len());
            stream.memcpy_htod(&bytes, &mut view).map_err(gerr)?;
            off += bytes.len();
        }
        self.owner[slot] = key;
        self.slot_of[key] = Some(slot as u32);
        // Served from VRAM now: its RAM copy can make way for experts the CPU reads.
        for (o, l) in Self::expert_ranges(model, li, e) {
            model.gguf().release(o, l);
        }
        stream.memcpy_htod(&[slot as i32], &mut self.slot_tab.slice_mut(key..key + 1)).map_err(gerr)?;
        Ok(())
    }

    /// Swaps up to `max` cold cached experts for hotter uncached ones.
    fn rebalance(&mut self, stream: &Arc<CudaStream>, model: &Model, max: usize) -> Result<usize, Error> {
        if self.owner.is_empty() {
            return Ok(0);
        }
        let mut cold: Vec<usize> = (0..self.owner.len()).collect();
        let ne = self.ne;
        cold.sort_by(|&a, &b| self.counts[self.owner[a]].total_cmp(&self.counts[self.owner[b]]));
        let mut hot: Vec<usize> = (0..self.slot_of.len()).filter(|&k| self.slot_of[k].is_none() && self.kinds[k / ne].is_some()).collect();
        hot.sort_by(|&a, &b| self.counts[b].total_cmp(&self.counts[a]));
        let mut n = 0;
        for (slot, key) in cold.into_iter().zip(hot) {
            if n >= max || self.counts[key] <= self.counts[self.owner[slot]] + 1.0 {
                break;
            }
            self.upload(stream, model, slot, key)?;
            n += 1;
        }
        Ok(n)
    }

    fn save_profile(&self) {
        let text: Vec<String> = self.counts.iter().map(|c| (c.round() as u64).to_string()).collect();
        let _ = std::fs::write(&self.profile, text.join(" "));
    }
}

/// VRAM ring slots for streaming uncached experts during prompt processing.
const RING: usize = 6;
/// Uncached experts with at least this many tokens in a prompt chunk are streamed to the GPU; fewer
/// are computed on the CPU (a 3 MB copy costs about as much as four CPU expert-token products).
const STREAM_MIN_TOKENS: usize = 4;

/// Most positions one speculative verify pass checks (the token plus its drafts).
pub const SPEC_MAX: usize = 16;

/// Prefill scratch: up to `CHUNK` tokens per batched pass.
const CHUNK: usize = 512;
/// Positions per split-K attention block.
const SPLIT: usize = 256;

struct Batch {
    x: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    att: CudaSlice<f32>,
    gu: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    toks: CudaSlice<i32>,
    /// [slot of the chunk's first token, rotary lag] (see qkv_post).
    pos0: CudaSlice<i32>,
    /// Per-token rotary (time, height, width) for a chunk with image tokens.
    rtab: CudaSlice<i32>,
    /// Gated attention: the projection's [q|gate] rows before they are split, and the gates.
    qg: CudaSlice<f32>,
    gate: CudaSlice<f32>,
}

/// Gated DeltaNet sizes: key heads, key/value head size, value heads, convolution taps.
#[derive(Clone, Copy)]
struct DeltaShape {
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
    kc: usize,
}

impl DeltaShape {
    /// Channels through the convolution (q, k and v).
    fn ch(&self) -> usize {
        2 * self.nk * self.dk + self.nv * self.dv
    }
    /// Width of the fused projection's output: q|k|v, z, alpha, beta.
    fn row(&self) -> usize {
        self.ch() + self.nv * self.dv + 2 * self.nv
    }
}

struct Shape {
    dim: usize,
    ffn: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    vocab: usize,
    eps: f32,
    base: f32,
    /// Dimensions RoPE rotates (Qwen3.8: 64 of 256).
    rd: usize,
    /// Attention heads carry an output gate (Qwen3.8).
    gate: bool,
    delta: Option<DeltaShape>,
    /// Multi-axis rotary sections, one byte each (0 = plain RoPE); see `rope_axis` in kernels.cu.
    sec: i32,
}

/// An image prompt being read on the GPU (`Session::feed_parts`): rotary (time, height, width)
/// positions for slots `from..`, and the image tokens' input rows by slot (they replace the
/// embedding lookup).
pub struct ImagePrefill {
    pub from: usize,
    pub rope: Vec<i32>,
    pub rows: Vec<(usize, Vec<f32>)>,
}

impl ImagePrefill {
    /// The rotary table for slots p0..p0 + t, when it covers them.
    fn table(&self, p0: usize, t: usize) -> Option<&[i32]> {
        let a = p0.checked_sub(self.from)? * 3;
        self.rope.get(a..a + 3 * t)
    }
}

/// Image slots carry pseudo ids (high bit set) that are not in the vocabulary: look up token 0
/// for them; their rows are replaced afterwards.
fn real_id(t: u32) -> u32 {
    if t & 0x8000_0000 != 0 { 0 } else { t }
}

/// The model's weights, KV cache and scratch on one GPU, plus the captured decode graph.
pub struct GpuState {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f: Funcs,
    sh: Shape,
    layers: Vec<GLayer>,
    out_norm: CudaSlice<f32>,
    /// The output head; None when the model is split (the CPU computes the logits).
    output: Option<GMat>,
    embed: Option<GMat>,
    /// MoE models look embeddings up on the CPU (the table stays out of VRAM); the row is uploaded.
    host_embed: Option<Vec<f32>>,
    x: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    att: CudaSlice<f32>,
    gu: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    /// Gated attention (decode): the [q|gate] projection and the gates.
    qg: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    tok: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    next: CudaSlice<i32>,
    pm: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    pacc: CudaSlice<f32>,
    splits: usize,
    /// Splits for attn_decode_gqa: enough blocks to fill the GPU, independent of the context size.
    gqa_splits: usize,
    b: Batch,
    moe: Option<MoeGpu>,
    graph: Option<CudaGraph>,
    pub max_ctx: usize,
    pub vram_bytes: usize,
    /// Where the weights went and why (also shown to the user).
    pub plan: crate::placement::Plan,
    /// Checking aid: batched prefill stops after this many layers (None: all of them).
    pub debug_stop: Option<usize>,
    /// Prompt-reading progress, advanced after each finished chunk (set by the session).
    pub progress: Option<Arc<crate::model::Progress>>,
    /// How far rotary positions lag cache slots for text after images (set by the session).
    pub rope_shift: i32,
    /// The image prompt being read, during `Session::feed_parts`.
    pub image: Option<ImagePrefill>,
    /// When set, a prefill also returns the last position's final-normed hidden state here (a
    /// speech talker's code predictor reads it).
    pub hidden: Option<Vec<f32>>,
    hidden_dev: Option<CudaSlice<f32>>,
    /// Logits of every position of a verify pass (SPEC_MAX x vocab); None where speculation
    /// does not apply (split, mixture of experts, recurrent layers).
    spec: Option<CudaSlice<f32>>,
    /// Split models: the activation after the last GPU layer, copied for the CPU.
    host_x: Option<PinnedHostSlice<f32>>,
}

fn gerr<E: std::fmt::Debug>(e: E) -> Error {
    Error::Unsupported(format!("CUDA: {e:?}"))
}

/// Repacks one or more matrices with the same column count, stacked by rows, into int8 values plus one
/// f32 scale per 32. Q8_0 keeps its exact values; other formats are dequantised and requantised.
fn repack(g: &Gguf, mats: &[&Mat]) -> (Vec<i8>, Vec<f32>, usize, usize) {
    let cols = mats[0].cols;
    let rows: usize = mats.iter().map(|m| m.rows).sum();
    let nb = cols / 32;
    let mut qs = vec![0i8; rows * cols];
    let mut ds = vec![0f32; rows * nb];
    let mut row_f = vec![0f32; cols];
    let mut r0 = 0;
    for m in mats {
        let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
        for r in 0..m.rows {
            let row = &all[r * m.row_bytes..(r + 1) * m.row_bytes];
            let (oq, od) = ((r0 + r) * cols, (r0 + r) * nb);
            if m.dtype == DType::Q8_0 {
                for b in 0..nb {
                    let blk = &row[b * 34..b * 34 + 34];
                    ds[od + b] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for i in 0..32 {
                        qs[oq + b * 32 + i] = blk[2 + i] as i8;
                    }
                }
            } else {
                dequant_row(m.dtype, row, &mut row_f);
                for b in 0..nb {
                    let xb = &row_f[b * 32..b * 32 + 32];
                    let amax = xb.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                    ds[od + b] = d;
                    for i in 0..32 {
                        qs[oq + b * 32 + i] = (xb[i] * id).round() as i8;
                    }
                }
            }
        }
        r0 += m.rows;
    }
    (qs, ds, rows, cols)
}

impl GpuState {
    pub fn load(model: &Model, max_ctx: usize) -> Result<Self, Error> {
        Self::load_with(model, max_ctx, None)
    }

    /// `load` with the number of GPU layers forced (None: the switch, or as many as fit).
    pub fn load_with(model: &Model, max_ctx: usize, force_layers: Option<usize>) -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gerr)?;
        // One stream, so the library's cross-stream event tracking is unnecessary (and it would
        // otherwise record events inside graph capture).
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(gerr)?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let lf = |n: &str| module.load_function(n).map_err(gerr);
        let cc8 = ctx.compute_capability().map_err(gerr)?.0 >= 8;
        let (mm_on, at_on) = (cc8 && switches::enabled(switches::GPU_MMA_MATMUL), cc8 && switches::enabled(switches::GPU_MMA_ATTENTION));
        let (matmul_mma, attn_mma, attn_mma256) = if mm_on || at_on {
            let m = ctx.load_module(Ptx::from_src(PTX_MMA)).map_err(gerr)?;
            let f = |on: bool, n: &str| if on { m.load_function(n).map(Some).map_err(gerr) } else { Ok(None) };
            (f(mm_on, "matmul_q8_mma")?, f(at_on, "attn_prefill_mma")?, f(at_on, "attn_prefill_mma256")?)
        } else {
            (None, None, None)
        };
        let f = Funcs {
            attn_mma256,
            delta_conv: lf("delta_conv")?,
            delta_l2: lf("delta_l2")?,
            delta_rule: lf("delta_rule")?,
            delta_gnorm: lf("delta_gnorm")?,
            unpack_qgate: lf("unpack_qgate")?,
            gate_mul: lf("gate_mul")?,
            matmul_mma,
            attn_mma,
            matvec: lf("matvec_q8")?,
            rmsnorm_quant: lf("rmsnorm_quant")?,
            quant: lf("quantize_q8")?,
            qkv_post: lf("qkv_post")?,
            attn_partial: lf("attn_partial")?,
            attn_combine: lf("attn_combine")?,
            attn_gqa: None,
            silu_mul_quant: lf("silu_mul_quant")?,
            embed: lf("embed")?,
            argmax: lf("argmax_next")?,
            embed_b: lf("embed_batch")?,
            norm_b: lf("rmsnorm_quant_batch")?,
            matmul_b: lf("matmul_q8_batch")?,
            silu_b: lf("silu_mul_quant_batch")?,
            attn_pre: lf("attn_prefill")?,
            moe_matvec: lf("moe_matvec")?,
            router_topk: lf("router_topk")?,
            moe_silu: lf("moe_silu_quant")?,
            moe_combine: lf("moe_combine")?,
            matvec_k: lf("matvec_k")?,
            kmatmul: lf("kmatmul")?,
            router_b: lf("router_topk_batch")?,
            scatter_add: lf("scatter_add")?,
            add_inplace: lf("add_inplace")?,
            attn_tiled: lf("attn_prefill_tiled")?,
        };
        let c = &model.cfg;
        // The grouped decode kernel is compiled per head size (64, 128, 256).
        let mut f = f;
        if switches::enabled(switches::GPU_DECODE_GQA) && matches!(c.head_dim, 64 | 128 | 256) && c.n_kv > 0 && c.n_head / c.n_kv <= 8 {
            // Smallest compiled group size that holds this model's group (registers follow it).
            let g = [1, 2, 4, 8].into_iter().find(|&g| g >= c.n_head / c.n_kv).unwrap_or(8);
            f.attn_gqa = Some(lf(&format!("attn_decode_gqa_{}_{g}", c.head_dim))?);
        }
        // Qwen3.8 hybrids (Gated DeltaNet + gated attention) run on the CPU until their GPU kernels land.
        if (c.ssm.is_some() || c.attn_gate) && c.n_expert > 0 {
            return Err(Error::Unsupported("hybrid mixture-of-experts layers on the GPU".into()));
        }
        if let Some(ss) = c.ssm {
            if ss.dv % 32 != 0 || ss.dk > 256 || ss.dk % 32 != 0 || ss.conv > 9 || ss.dv > 1024 {
                return Err(Error::Unsupported("this DeltaNet layer size on the GPU".into()));
            }
        }
        if c.head_dim > 256 || c.head_dim % 64 != 0 || c.dim > 8192 {
            return Err(Error::Unsupported(format!("head size {} / width {} on the GPU", c.head_dim, c.dim)));
        }
        let g = model.gguf();
        let mut vram = 0usize;
        let kvd = c.n_kv * c.head_dim;
        // Where everything goes (placement.rs). MoE models keep their K-quant weights native on
        // the GPU (about half the bytes of int8), leaving VRAM to the expert cache.
        let (free, _) = ctx.mem_get_info().map_err(gerr)?;
        let plan = plan_for(model, max_ctx, free as u64, force_layers)?;
        if plan.layout == crate::placement::Layout::Cpu {
            return Err(Error::Unsupported(plan.notes.join(" ")));
        }
        let native = c.n_expert > 0 || plan.layout != crate::placement::Layout::GpuInt8;
        let n_gpu = if c.n_expert > 0 { c.n_layer } else { plan.gpu_layers };
        let split = n_gpu < c.n_layer;
        let mut up = |mats: &[&Mat]| -> Result<GMat, Error> {
            if native && mats.iter().all(|m| gpu_block(m.dtype).is_some() && m.cols % 256 == 0) {
                let mut parts = Vec::with_capacity(mats.len());
                for m in mats {
                    let (bb, dtype) = gpu_block(m.dtype).unwrap();
                    let bytes = to_gpu_layout(m.dtype, g.bytes_at(m.offset, m.rows * m.row_bytes));
                    vram += bytes.len();
                    parts.push(KPart { w: stream.clone_htod(&bytes).map_err(gerr)?, dtype, bb: bb as i32, rows: m.rows });
                }
                return Ok(GMat::K { parts, cols: mats[0].cols });
            }
            let (qs, ds, rows, cols) = repack(g, mats);
            vram += qs.len() + ds.len() * 4;
            Ok(GMat::Q8 { qs: stream.clone_htod(&qs).map_err(gerr)?, ds: stream.clone_htod(&ds).map_err(gerr)?, rows, cols })
        };
        let mut layers = Vec::with_capacity(n_gpu);
        for l in &model.layers()[..n_gpu] {
            let (gu, down, router) = match &l.ffn {
                Ffn::Dense { gate, up: upm, down } => (Some(up(&[gate, upm])?), Some(up(&[down])?), None),
                Ffn::Moe(m) => (None, None, Some(stream.clone_htod(&m.router).map_err(gerr)?)),
            };
            let (qkv, o, delta) = match (&l.delta, c.ssm) {
                (Some(d), Some(ss)) => (
                    up(&[&d.qkv, &d.z, &d.alpha, &d.beta])?,
                    up(&[&d.out])?,
                    Some(GDelta {
                        a: stream.clone_htod(&d.a).map_err(gerr)?,
                        dt_bias: stream.clone_htod(&d.dt_bias).map_err(gerr)?,
                        conv: stream.clone_htod(&d.conv).map_err(gerr)?,
                        norm: stream.clone_htod(&d.norm).map_err(gerr)?,
                        hist: stream.alloc_zeros::<f32>(((ss.conv - 1) * ss.channels()).max(1)).map_err(gerr)?,
                        state: stream.alloc_zeros::<f32>(ss.nv * ss.dk * ss.dv).map_err(gerr)?,
                    }),
                ),
                _ => (up(&[&l.q, &l.k, &l.v])?, up(&[&l.o])?, None),
            };
            let attn = delta.is_none();
            layers.push(GLayer {
                attn_norm: stream.clone_htod(&l.attn_norm).map_err(gerr)?,
                qkv,
                o,
                delta,
                q_norm: l.q_norm.as_deref().map(|v| stream.clone_htod(v)).transpose().map_err(gerr)?,
                k_norm: l.k_norm.as_deref().map(|v| stream.clone_htod(v)).transpose().map_err(gerr)?,
                ffn_norm: stream.clone_htod(&l.ffn_norm).map_err(gerr)?,
                gu,
                down,
                router,
                kc: stream.alloc_zeros::<u16>(if attn { max_ctx * kvd } else { 1 }).map_err(gerr)?,
                vc: stream.alloc_zeros::<u16>(if attn { max_ctx * kvd } else { 1 }).map_err(gerr)?,
            });
        }
        let output = if split { None } else { Some(up(&[model.output_mat()])?) };
        // Tied embeddings reuse the output matrix on the GPU; otherwise upload the table too. In the
        // native layout the row is looked up on the CPU and uploaded (the table stays out of VRAM).
        let host_embed = native.then(|| vec![0f32; c.dim]);
        let embed = if model.tied() || native { None } else { Some(up(&[model.embed_mat()])?) };
        let attn_layers = model.layers()[..n_gpu].iter().filter(|l| l.delta.is_none()).count();
        vram += attn_layers * 2 * max_ctx * kvd * 2;
        let dsh = c.ssm.map(|ss| DeltaShape { nk: ss.nk, dk: ss.dk, nv: ss.nv, dv: ss.dv, kc: ss.conv });
        if let Some(d) = dsh {
            vram += (n_gpu - attn_layers) * (d.nv * d.dk * d.dv + (d.kc - 1) * d.ch()) * 4;
        }
        let z = |n: usize| stream.alloc_zeros::<f32>(n.max(1)).map_err(gerr);
        let qd = c.n_head * c.head_dim;
        let dout = dsh.map_or(0, |d| d.nv * d.dv);
        let widest = c.dim.max(c.ffn).max(qd).max(dout);
        let qkv_w = (qd + 2 * kvd).max(dsh.map_or(0, |d| d.row()));
        let qg_w = if c.attn_gate { 2 * qd + 2 * kvd } else { 0 };
        let splits = max_ctx.div_ceil(SPLIT);
        // About eight blocks per SM for the grouped decode kernel; it sizes each split from the live
        // position, so short contexts leave most splits empty and long ones keep every SM busy.
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap_or(40).max(1) as usize;
        let gqa_splits = (8 * sms).div_ceil(c.n_kv.max(1)).clamp(1, 128);
        let row = qd + 2 * kvd;
        let _ = row;
        let b = Batch {
            qg: z(CHUNK * qg_w)?,
            gate: z(if c.attn_gate { CHUNK * qd } else { 0 })?,
            x: z(CHUNK * c.dim)?,
            qkv: z(CHUNK * qkv_w)?,
            att: z(CHUNK * qd.max(dout))?,
            gu: z(CHUNK * 2 * c.ffn)?,
            xq: stream.alloc_zeros::<i8>(CHUNK * widest).map_err(gerr)?,
            xd: z(CHUNK * widest / 32)?,
            toks: stream.alloc_zeros::<i32>(CHUNK).map_err(gerr)?,
            pos0: stream.alloc_zeros::<i32>(2).map_err(gerr)?,
            rtab: stream.alloc_zeros::<i32>(3 * CHUNK).map_err(gerr)?,
        };
        // Everything uploaded so far is read from VRAM only; free its RAM for the CPU-side weights.
        // A split keeps the CPU's layers and the head (and the embeddings, looked up on the CPU).
        {
            let rel = |m: &Mat| g.release(m.offset, m.rows * m.row_bytes);
            for l in &model.layers()[..n_gpu] {
                match &l.delta {
                    Some(d) => {
                        for m in [&d.qkv, &d.z, &d.alpha, &d.beta, &d.out] {
                            rel(m);
                        }
                    }
                    None => {
                        for m in [&l.q, &l.k, &l.v, &l.o] {
                            rel(m);
                        }
                    }
                }
                if let Ffn::Dense { gate, up, down } = &l.ffn {
                    for m in [gate, up, down] {
                        rel(m);
                    }
                }
            }
            if !split {
                let tied_and_host = model.tied() && native;
                if !tied_and_host {
                    rel(model.output_mat());
                }
            }
            if !native {
                rel(model.embed_mat());
            }
        }
        let moe = if c.n_expert > 0 { Some(Self::load_moe(&ctx, &stream, model)?) } else { None };
        if let Some(m) = &moe {
            vram += m.owner.len() * m.slot_bytes;
        }
        let sh = Shape { dim: c.dim, ffn: c.ffn, n_head: c.n_head, n_kv: c.n_kv, hd: c.head_dim, vocab: c.vocab, eps: c.eps, base: c.rope_base, rd: c.rope_dim, gate: c.attn_gate, delta: dsh, sec: c.rope_sections.map_or(0, |s| s.iter().enumerate().map(|(i, &v)| (v.min(255) as i32) << (8 * i)).fold(0, |a, b| a | b)) };
        let ctx_pin = ctx.clone();
        let spec = if split || c.n_expert > 0 || c.ssm.is_some() { None } else { Some(z(SPEC_MAX * c.vocab)?) };
        Ok(Self {
            out_norm: stream.clone_htod(model.out_norm()).map_err(gerr)?,
            output,
            embed,
            host_embed,
            x: z(c.dim)?,
            qkv: z(qkv_w)?,
            att: z(qd.max(dout))?,
            gu: z(2 * c.ffn)?,
            xq: stream.alloc_zeros::<i8>(widest).map_err(gerr)?,
            xd: z(widest / 32)?,
            qg: z(qg_w)?,
            gate: z(if c.attn_gate { qd } else { 0 })?,
            logits: z(c.vocab)?,
            tok: stream.alloc_zeros::<i32>(1).map_err(gerr)?,
            pos: stream.alloc_zeros::<i32>(2).map_err(gerr)?,
            next: stream.alloc_zeros::<i32>(1).map_err(gerr)?,
            pm: z(c.n_head * splits.max(gqa_splits))?,
            pl: z(c.n_head * splits.max(gqa_splits))?,
            pacc: z(c.n_head * splits.max(gqa_splits) * c.head_dim)?,
            splits,
            gqa_splits,
            b,
            moe,
            graph: None,
            layers,
            f,
            sh,
            stream,
            _ctx: ctx,
            max_ctx,
            vram_bytes: vram,
            plan,
            debug_stop: None,
            progress: None,
            rope_shift: 0,
            image: None,
            hidden: None,
            hidden_dev: None,
            spec,
            // SAFETY: a plain host buffer, read only after the stream is synchronized.
            host_x: if split { Some(unsafe { ctx_pin.alloc_pinned::<f32>(c.dim) }.map_err(gerr)?) } else { None },
        })
    }

    /// A DeltaNet layer after its fused projection (rows of `ld` floats in `row`, T tokens): the
    /// convolution, q/k normalisation, the delta rule and the gated norm, into `out` (T x nv*dv).
    fn delta_core(s: &CudaStream, f: &Funcs, sh: &Shape, dl: &mut GDelta, row: &mut CudaSlice<f32>, out: &mut CudaSlice<f32>, t: usize) -> Result<(), Error> {
        let d = sh.delta.expect("delta shape");
        let (ld, ch, nk, dk, nv, dv, kc, ti) = (d.row() as i32, d.ch() as i32, d.nk as i32, d.dk as i32, d.nv as i32, d.dv as i32, d.kc as i32, t as i32);
        let conv_cfg = LaunchConfig { grid_dim: ((d.ch() as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.delta_conv).arg(&mut *row).arg(&ld).arg(&dl.conv).arg(&mut dl.hist).arg(&ch).arg(&kc).arg(&ti).launch(conv_cfg) }.map_err(gerr)?;
        let qscale = 1.0f32 / (d.dk as f32).sqrt();
        let l2_cfg = LaunchConfig { grid_dim: (2 * d.nk as u32, t as u32, 1), block_dim: (d.dk as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.delta_l2).arg(&mut *row).arg(&ld).arg(&nk).arg(&dk).arg(&qscale).launch(l2_cfg) }.map_err(gerr)?;
        let ldo = (d.nv * d.dv) as i32;
        let rule_cfg = LaunchConfig { grid_dim: (d.nv as u32, (d.dv / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: ((d.dk + 4) * 32 * 4) as u32 };
        unsafe { s.launch_builder(&f.delta_rule).arg(&*row).arg(&ld).arg(&ch).arg(&nk).arg(&dk).arg(&nv).arg(&dv).arg(&dl.a).arg(&dl.dt_bias).arg(&mut dl.state).arg(&mut *out).arg(&ldo).arg(&ti).launch(rule_cfg) }.map_err(gerr)?;
        let zoff = ch;
        let gn_cfg = LaunchConfig { grid_dim: (d.nv as u32, t as u32, 1), block_dim: (d.dv as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.delta_gnorm).arg(&mut *out).arg(&ldo).arg(&*row).arg(&ld).arg(&zoff).arg(&dl.norm).arg(&dv).arg(&sh.eps).launch(gn_cfg) }.map_err(gerr)?;
        Ok(())
    }

    /// Gated attention: the [q|gate] projection rows (`src`) into [q|k|v] rows and the gates.
    fn unpack_gate(s: &CudaStream, f: &Funcs, sh: &Shape, src: &CudaSlice<f32>, qkv: &mut CudaSlice<f32>, gate: &mut CudaSlice<f32>, t: usize) -> Result<(), Error> {
        let (nh, nkv, hd) = (sh.n_head as i32, sh.n_kv as i32, sh.hd as i32);
        let cfg = LaunchConfig { grid_dim: ((sh.n_head + 2 * sh.n_kv) as u32, t as u32, 1), block_dim: (sh.hd as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.unpack_qgate).arg(src).arg(qkv).arg(gate).arg(&nh).arg(&nkv).arg(&hd).launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    fn apply_gate(s: &CudaStream, f: &Funcs, att: &mut CudaSlice<f32>, gate: &CudaSlice<f32>, n: usize) -> Result<(), Error> {
        let ni = n as i32;
        let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.gate_mul).arg(att).arg(gate).arg(&ni).launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    /// Clears every DeltaNet state (a new sequence).
    pub fn reset_recurrent(&mut self) -> Result<(), Error> {
        for l in &mut self.layers {
            if let Some(d) = l.delta.as_mut() {
                self.stream.memset_zeros(&mut d.hist).map_err(gerr)?;
                self.stream.memset_zeros(&mut d.state).map_err(gerr)?;
            }
        }
        Ok(())
    }

    /// The layers the GPU runs (all of them unless split).
    pub fn gpu_layers(&self) -> usize {
        self.layers.len()
    }

    /// The CPU runs the layers from here on and the output head.
    pub fn is_split(&self) -> bool {
        self.output.is_none()
    }

}

/// Placement for `model` with `vram_free` bytes of GPU memory (0: none), the RAM available now and
/// the model's drive. `force_layers` overrides the GPU layer count (None: the switch, else as many
/// as fit). Models whose layers have no GPU kernels yet plan as processor-only.
pub fn plan_for(model: &Model, max_ctx: usize, vram_free: u64, force_layers: Option<usize>) -> Result<crate::placement::Plan, Error> {
    {
        use crate::placement::{drive_of, plan_dense, plan_moe, LayerCost, Machine, ModelCost};
        let c = &model.cfg;
        let (_, ram) = crate::hw::memory();
        let hybrid = c.ssm.is_some() || c.attn_gate;
        let gpu_kernels = !(hybrid && c.n_expert > 0) && c.head_dim <= 256 && c.head_dim % 64 == 0 && c.dim <= 8192;
        let mach = Machine::new(if gpu_kernels { vram_free } else { 0 }, ram, drive_of(&model.path));
        let kvd = (c.n_kv * c.head_dim) as u64;
        let native_bytes = |m: &Mat| -> Option<u64> {
            let (bb, _) = gpu_block(m.dtype)?;
            (m.cols % 256 == 0).then_some((m.rows * m.cols / 256 * bb) as u64)
        };
        let int8_bytes = |m: &Mat| (m.rows * m.cols + m.rows * m.cols / 32 * 4) as u64;
        let qd = c.n_head * c.head_dim;
        let widest = c.dim.max(c.ffn).max(qd);
        let scratch = (CHUNK * (c.dim + qd + 2 * kvd as usize + qd + 2 * c.ffn) * 4 + CHUNK * widest * 2 + (SPEC_MAX + 1) * c.vocab * 4 + 64 * (c.n_head * c.head_dim) * 4 * 4) as u64;
        if c.n_expert > 0 {
            let mut dense = scratch + c.n_layer as u64 * 2 * max_ctx as u64 * kvd * 2;
            let mut experts = 0u64;
            for l in model.layers() {
                dense += [&l.q, &l.k, &l.v, &l.o].iter().map(|m| native_bytes(m).unwrap_or(int8_bytes(m))).sum::<u64>();
                if let Ffn::Moe(m) = &l.ffn {
                    experts += [&m.gate, &m.up, &m.down].iter().map(|x| (x.rows * x.row_bytes) as u64).sum::<u64>();
                }
            }
            let out = model.output_mat();
            dense += native_bytes(out).unwrap_or(int8_bytes(out));
            return Ok(plan_moe(dense, experts, c.n_used as f64 / c.n_expert.max(1) as f64, &mach));
        }
        if !gpu_kernels {
            // Processor only: the layers' share of the file, evenly (DeltaNet layers have their own
            // tensors, so the generic matrix walk below does not apply).
            let file = model.gguf().len() as u64;
            let head = (model.output_mat().rows * model.output_mat().row_bytes) as u64;
            let per = file.saturating_sub(head) / c.n_layer.max(1) as u64;
            let kv_cpu = 2 * max_ctx as u64 * kvd * 4;
            let layers = (0..c.n_layer).map(|i| LayerCost { int8: per, native: per, kv_gpu: 0, kv_cpu: if model.layers()[i].delta.is_none() { kv_cpu } else { 0 }, native_ok: false }).collect();
            let mut p = plan_dense(&ModelCost { layers, head_int8: head, head_native: head, scratch: 0, file }, &mach, None);
            if vram_free > 0 {
                p.notes[0] = "This model's layer type does not run on the graphics card yet: everything runs on the processor.".to_owned();
            }
            return Ok(p);
        }
        let mut layers = Vec::with_capacity(c.n_layer);
        for l in model.layers() {
            let Ffn::Dense { gate, up, down } = &l.ffn else { return Err(Error::Unsupported("mixed dense and expert layers".into())) };
            let mats: Vec<&Mat> = match &l.delta {
                Some(d) => vec![&d.qkv, &d.z, &d.alpha, &d.beta, &d.out, gate, up, down],
                None => vec![&l.q, &l.k, &l.v, &l.o, gate, up, down],
            };
            let attn = l.delta.is_none();
            let state = c.ssm.map_or(0, |ss| if attn { 0 } else { (ss.nv * ss.dk * ss.dv + (ss.conv - 1) * ss.channels()) as u64 * 4 });
            let nat: Option<u64> = mats.iter().map(|m| native_bytes(m)).sum();
            layers.push(LayerCost {
                int8: mats.iter().map(|m| int8_bytes(m)).sum(),
                native: nat.unwrap_or_else(|| mats.iter().map(|m| (m.rows * m.row_bytes) as u64).sum()),
                kv_gpu: if attn { 2 * max_ctx as u64 * kvd * 2 } else { state },
                kv_cpu: if attn { 2 * max_ctx as u64 * kvd * 4 } else { state },
                native_ok: nat.is_some(),
            });
        }
        let out = model.output_mat();
        let mut head_int8 = int8_bytes(out);
        let mut head_native = native_bytes(out).unwrap_or(head_int8);
        if !model.tied() {
            head_int8 += int8_bytes(model.embed_mat());
        }
        // In the native layout the embeddings are looked up on the CPU: nothing extra on the GPU.
        let _ = &mut head_native;
        let cost = ModelCost { layers, head_int8, head_native, scratch, file: model.gguf().len() as u64 };
        let force = force_layers.or(match crate::switches::get(crate::switches::GPU_LAYERS) {
            n if n > 0 => Some(n as usize),
            _ => None,
        });
        Ok(plan_dense(&cost, &mach, force))
    }
}

impl GpuState {

    /// Sets up the VRAM expert cache in the memory left after the dense weights and KV cache (minus a
    /// safety margin; `CEREBRO_EXPERT_VRAM_MB` caps it for placement experiments) and fills it from the
    /// saved expert profile, or evenly across layers when there is none.
    fn load_moe(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, model: &Model) -> Result<MoeGpu, Error> {
        let c = &model.cfg;
        let em = |mat: &Mat| -> Option<ExpMat> {
            let (bb, dtype) = gpu_block(mat.dtype)?;
            let rows = mat.rows / c.n_expert;
            (mat.cols % 256 == 0).then_some(ExpMat { dtype, bb: bb as i32, rows, cols: mat.cols, bytes: rows * mat.cols / 256 * bb })
        };
        let kinds: Vec<Option<[ExpMat; 3]>> = model
            .layers()
            .iter()
            .map(|l| match &l.ffn {
                Ffn::Moe(m) => Some([em(&m.gate)?, em(&m.up)?, em(&m.down)?]),
                Ffn::Dense { .. } => None,
            })
            .collect();
        let cacheable = kinds.iter().filter(|k| k.is_some()).count() * c.n_expert;
        let total = c.n_layer * c.n_expert;
        let (free, _) = ctx.mem_get_info().map_err(gerr)?;
        let mut budget = free.saturating_sub(600 << 20);
        let mb = switches::get(switches::EXPERT_VRAM_MB) as usize;
        if mb > 0 {
            budget = budget.min(mb << 20);
        }
        let slot_bytes = kinds.iter().flatten().map(|k| k.iter().map(|m| m.bytes).sum::<usize>()).max().unwrap_or(0).div_ceil(256).max(1) * 256;
        let budget = budget.saturating_sub(RING * slot_bytes);
        let n_slots = (budget / slot_bytes).min(cacheable);
        let pool = stream.alloc_zeros::<u8>((n_slots * slot_bytes).max(256)).map_err(gerr)?;
        let profile = std::path::PathBuf::from(format!("{}.experts", model.path.display()));
        let mut counts = vec![0f32; total];
        if let Ok(t) = std::fs::read_to_string(&profile) {
            let v: Vec<f32> = t.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if v.len() == total {
                counts = v;
            }
        }
        let (k, eff, dim) = (c.n_used, c.expert_ffn, c.dim);
        let z = |n: usize| stream.alloc_zeros::<f32>(n).map_err(gerr);
        let mut mg = MoeGpu {
            ne: c.n_expert,
            k,
            eff,
            kinds,
            pool,
            slot_bytes,
            slot_of: vec![None; total],
            owner: vec![usize::MAX; n_slots],
            counts,
            profile,
            hits: 0,
            lookups: 0,
            wait_s: 0.0,
            cpu_s: 0.0,
            steps: 0,
            slot_tab: stream.clone_htod(&vec![-1i32; total]).map_err(gerr)?,
            h: z(dim)?,
            sel: stream.alloc_zeros::<i32>(3 * k).map_err(gerr)?,
            gu: z(2 * k * eff)?,
            axq: stream.alloc_zeros::<i8>(k * eff).map_err(gerr)?,
            axd: z(k * eff / 32)?,
            eo: z(k * dim)?,
            cpu: z(dim)?,
            // SAFETY: plain host buffers, read only after their copies complete (as_slice syncs).
            sel_h: unsafe { ctx.alloc_pinned::<i32>(3 * k) }.map_err(gerr)?,
            h_pin: unsafe { ctx.alloc_pinned::<f32>(dim) }.map_err(gerr)?,
            cpu_pin: unsafe { ctx.alloc_pinned::<f32>(dim) }.map_err(gerr)?,
            h_h: vec![0.0; dim],
            bh: z(CHUNK * dim)?,
            bsel: stream.alloc_zeros::<i32>(CHUNK * 3 * k).map_err(gerr)?,
            bidx: stream.alloc_zeros::<i32>(CHUNK * k).map_err(gerr)?,
            bw: z(CHUNK * k)?,
            bgu: z(2 * CHUNK * eff)?,
            baxq: stream.alloc_zeros::<i8>(CHUNK * eff).map_err(gerr)?,
            baxd: z(CHUNK * eff / 32)?,
            beo: z(CHUNK * dim)?,
            bcpu: z(CHUNK * dim)?,
            copy: ctx.new_stream().map_err(gerr)?,
            ring_dev: stream.alloc_zeros::<u8>(RING * slot_bytes).map_err(gerr)?,
            ring_host: (0..RING).map(|_| unsafe { ctx.alloc_pinned::<u8>(slot_bytes) }).collect::<Result<Vec<_>, _>>().map_err(gerr)?,
            ring_done: (0..RING).map(|_| None).collect(),
            qa: QAct::new(dim),
            scratch: MoeScratch::new(c),
            cpu_out: vec![0.0; dim],
        };
        // Initial fill: most-used experts first; with no profile, the same number per layer.
        let mut order: Vec<usize> = (0..total).map(|i| (i % c.n_layer) * c.n_expert + i / c.n_layer).filter(|&k| mg.kinds[k / c.n_expert].is_some()).collect();
        order.sort_by(|&a, &b| mg.counts[b].total_cmp(&mg.counts[a]));
        for (slot, &key) in order.iter().take(n_slots).enumerate() {
            mg.upload(stream, model, slot, key)?;
        }
        // Map the experts the CPU will compute into memory before the first token, most used first,
        // as many as available RAM holds (CEREBRO_PREFAULT=0 skips this); ask the OS to read ahead
        // the rest.
        let mut cpu_side: Vec<usize> = (0..total).filter(|&k| mg.slot_of[k].is_none()).collect();
        cpu_side.sort_by(|&a, &b| mg.counts[b].total_cmp(&mg.counts[a]));
        let mut room = if !switches::enabled(switches::PREFAULT) { 0 } else { (crate::gguf::available_ram() as f64 * 0.85) as usize };
        let (mut touch, mut ahead) = (Vec::new(), Vec::new());
        for k in cpu_side {
            let r = MoeGpu::expert_ranges(model, k / c.n_expert, k % c.n_expert);
            let bytes: usize = r.iter().map(|x| x.1).sum();
            if bytes <= room {
                room -= bytes;
                touch.extend(r);
            } else {
                ahead.extend(r);
            }
        }
        model.gguf().touch(&touch);
        model.gguf().prefetch(&ahead);
        Ok(mg)
    }

    /// (experts cached in VRAM, experts in the model, share of expert uses served from VRAM so far).
    pub fn expert_cache(&self) -> Option<(usize, usize, f64)> {
        self.moe.as_ref().map(|m| (m.owner.len(), m.slot_of.len(), if m.lookups > 0 { m.hits as f64 / m.lookups as f64 } else { 0.0 }))
    }

    /// Self-test of the GPU expert kernels: runs the gate and down matrices of the expert in cache slot
    /// 0 on the GPU and on the CPU with the same quantised inputs; returns the largest relative error
    /// of each. None when no expert is cached.
    pub fn check_expert_kernels(&mut self, model: &Model) -> Result<Option<(f32, f32)>, Error> {
        let s = &self.stream;
        let f = &self.f;
        let Some(mo) = self.moe.as_mut() else { return Ok(None) };
        let Some(&key) = mo.owner.first().filter(|&&k| k != usize::MAX) else { return Ok(None) };
        let (li, e, ne) = (key / mo.ne, key % mo.ne, mo.ne);
        let Ffn::Moe(m) = &model.layers()[li].ffn else { return Ok(None) };
        let g = model.gguf();
        let cpu_rows = |mat: &Mat, x: &[f32]| -> Vec<f32> {
            let p = mat.part(e, ne);
            let mut qa = QAct::new(x.len());
            qa.quantize(x);
            let all = g.bytes_at(p.offset, p.rows * p.row_bytes);
            (0..p.rows).map(|r| crate::model::dot_any(p.dtype, &all[r * p.row_bytes..(r + 1) * p.row_bytes], x, &qa)).collect()
        };
        let err = |a: &[f32], b: &[f32]| {
            let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
        };
        let slot = mo.owner.iter().position(|&k| k == key).unwrap() as i32;
        let run = |mo: &mut MoeGpu, x: &[f32], off: i64, em: ExpMat| -> Result<Vec<f32>, Error> {
            let n = x.len() as i32;
            let xd = s.clone_htod(x).map_err(gerr)?;
            let cfg = LaunchConfig { grid_dim: ((x.len() as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.quant).arg(&xd).arg(&mut mo.axq).arg(&mut mo.axd).arg(&n).launch(cfg) }.map_err(gerr)?;
            // Only choice 0 is "cached" (the slot under test); the rest are skipped.
            let mut sel = vec![-1i32; 3 * mo.k];
            sel[2 * mo.k] = slot;
            s.memcpy_htod(&sel, &mut mo.sel).map_err(gerr)?;
            let (rows, cols, zero, sb) = (em.rows as i32, em.cols as i32, 0i32, mo.slot_bytes as i64);
            let cfg = LaunchConfig { grid_dim: ((em.rows as u32).div_ceil(8), mo.k as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.moe_matvec).arg(&mo.pool).arg(&sb).arg(&off).arg(&mo.sel).arg(&em.dtype).arg(&em.bb).arg(&mo.axq).arg(&mo.axd).arg(&zero).arg(&mut mo.eo).arg(&rows).arg(&cols).launch(cfg) }.map_err(gerr)?;
            let mut out = vec![0f32; mo.eo.len()];
            s.memcpy_dtoh(&mo.eo, &mut out).map_err(gerr)?;
            out.truncate(em.rows);
            Ok(out)
        };
        let [gate, up, down] = mo.kinds[li].expect("cached");
        let xh: Vec<f32> = (0..gate.cols).map(|i| ((i as f32) * 0.37).sin()).collect();
        let gg = run(mo, &xh, 0, gate)?;
        let gc = cpu_rows(&m.gate, &xh);
        let xa: Vec<f32> = (0..down.cols).map(|i| ((i as f32) * 0.11).cos()).collect();
        let dg = run(mo, &xa, (gate.bytes + up.bytes) as i64, down)?;
        let dc = cpu_rows(&m.down, &xa);
        Ok(Some((err(&gg, &gc), err(&dg, &dc))))
    }

    /// (seconds waiting for the GPU's routing, seconds computing CPU experts) so far.
    pub fn moe_times(&self) -> Option<(f64, f64)> {
        self.moe.as_ref().map(|m| (m.wait_s, m.cpu_s))
    }

    /// Saves the expert-use profile next to the model (MoE models only).
    pub fn save_expert_profile(&self) {
        if let Some(m) = &self.moe {
            m.save_profile();
        }
    }

    /// Causal attention for a batch of t tokens at positions *pos0.. : the tiled kernel for head size
    /// 128 (K/V tiles shared by all query heads of a KV head), the simple one otherwise.
    #[allow(clippy::too_many_arguments)]
    fn attn_batch(s: &CudaStream, f: &Funcs, sh: &Shape, qkv: &CudaSlice<f32>, kc: &CudaSlice<u16>, vc: &CudaSlice<u16>, att: &mut CudaSlice<f32>, pos0: &CudaSlice<i32>, t: usize) -> Result<(), Error> {
        let (n_head, n_kv, hd) = (sh.n_head as i32, sh.n_kv as i32, sh.hd as i32);
        let group = (sh.n_head / sh.n_kv) as i32;
        let scale = 1.0f32 / (sh.hd as f32).sqrt();
        if let (Some(k), 256) = (&f.attn_mma256, sh.hd) {
            let ti = t as i32;
            let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, t.div_ceil(64) as u32, 2), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(k).arg(qkv).arg(kc).arg(vc).arg(att).arg(&n_head).arg(&n_kv).arg(pos0).arg(&group).arg(&scale).arg(&ti).launch(cfg) }.map_err(gerr)?;
        } else if let (Some(k), 128) = (&f.attn_mma, sh.hd) {
            let ti = t as i32;
            let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, t.div_ceil(64) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(k).arg(qkv).arg(kc).arg(vc).arg(att).arg(&n_head).arg(&n_kv).arg(pos0).arg(&group).arg(&scale).arg(&ti).launch(cfg) }.map_err(gerr)?;
        } else if sh.hd == 128 && sh.n_head / sh.n_kv <= 64 {
            let qt = (64 / (sh.n_head / sh.n_kv)).max(1);
            let (ti, qti) = (t as i32, qt as i32);
            let cfg = LaunchConfig { grid_dim: (sh.n_kv as u32, t.div_ceil(qt) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.attn_tiled).arg(qkv).arg(kc).arg(vc).arg(att).arg(&n_head).arg(&n_kv).arg(pos0).arg(&group).arg(&scale).arg(&ti).arg(&qti).launch(cfg) }.map_err(gerr)?;
        } else {
            let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, t as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.attn_pre).arg(qkv).arg(kc).arg(vc).arg(att).arg(&n_head).arg(&n_kv).arg(&hd).arg(pos0).arg(&group).arg(&scale).launch(cfg) }.map_err(gerr)?;
        }
        Ok(())
    }

    fn matvec(s: &CudaStream, f: &Funcs, m: &GMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, add: bool) -> Result<(), Error> {
        let addi = add as i32;
        let cfg = |rows: usize| LaunchConfig { grid_dim: ((rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        match m {
            GMat::Q8 { qs, ds, rows, cols } => {
                let (r, c) = (*rows as i32, *cols as i32);
                unsafe { s.launch_builder(&f.matvec).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&r).arg(&c).arg(&addi).launch(cfg(*rows)) }.map_err(gerr)?;
            }
            GMat::K { parts, cols } => {
                let c = *cols as i32;
                let mut r0 = 0;
                for p in parts {
                    let mut yv = y.slice_mut(r0..r0 + p.rows);
                    let r = p.rows as i32;
                    unsafe { s.launch_builder(&f.matvec_k).arg(&p.w).arg(&p.dtype).arg(&p.bb).arg(xq).arg(xd).arg(&mut yv).arg(&r).arg(&c).arg(&addi).launch(cfg(p.rows)) }.map_err(gerr)?;
                    r0 += p.rows;
                }
            }
        }
        Ok(())
    }

    /// Enqueues one full decode step reading the token and position from device memory.
    fn enqueue(&mut self, model: &Model) -> Result<(), Error> {
        let sh = &self.sh;
        let s = &self.stream;
        let f = &self.f;
        let blocks = |n: usize| LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (dim, ffn, hd) = (sh.dim as i32, sh.ffn as i32, sh.hd as i32);
        let (n_head, n_kv) = (sh.n_head as i32, sh.n_kv as i32);
        let (eps, base) = (sh.eps, sh.base);
        let null_f: u64 = 0;
        if let Some(row) = &self.host_embed {
            s.memcpy_htod(row, &mut self.x).map_err(gerr)?;
        } else {
            let (qs, ds, _, _) = self.embed.as_ref().or(self.output.as_ref()).expect("embedding table").q8();
            unsafe { s.launch_builder(&f.embed).arg(qs).arg(ds).arg(&self.tok).arg(&mut self.x).arg(&dim).launch(blocks(sh.dim)) }.map_err(gerr)?;
        }
        let group = (sh.n_head / sh.n_kv) as i32;
        let scale = 1.0f32 / (sh.hd as f32).sqrt();
        for (li, l) in self.layers.iter_mut().enumerate() {
            unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&l.attn_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
            if let Some(dl) = l.delta.as_mut() {
                // Gated DeltaNet: fused projection, the recurrent core, output projection.
                Self::matvec(s, f, &l.qkv, &self.xq, &self.xd, &mut self.qkv, false)?;
                Self::delta_core(s, f, sh, dl, &mut self.qkv, &mut self.att, 1)?;
                let n = sh.delta.map_or(0, |d| d.nv * d.dv) as i32;
                unsafe { s.launch_builder(&f.quant).arg(&self.att).arg(&mut self.xq).arg(&mut self.xd).arg(&n).launch(blocks(n as usize)) }.map_err(gerr)?;
                Self::matvec(s, f, &l.o, &self.xq, &self.xd, &mut self.x, true)?;
            } else {
            if sh.gate {
                Self::matvec(s, f, &l.qkv, &self.xq, &self.xd, &mut self.qg, false)?;
                Self::unpack_gate(s, f, sh, &self.qg, &mut self.qkv, &mut self.gate, 1)?;
            } else {
                Self::matvec(s, f, &l.qkv, &self.xq, &self.xd, &mut self.qkv, false)?;
            }
            {
                let qn = l.q_norm.as_ref().unwrap_or(&l.attn_norm);
                let kn = l.k_norm.as_ref().unwrap_or(&l.attn_norm);
                let cfg = LaunchConfig { grid_dim: ((sh.n_head + sh.n_kv) as u32, 1, 1), block_dim: ((sh.hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    s.launch_builder(&f.qkv_post)
                        .arg(&mut self.qkv)
                        .arg(qn)
                        .arg(kn)
                        .arg(&(l.q_norm.is_some() as i32))
                        .arg(&mut l.kc)
                        .arg(&mut l.vc)
                        .arg(&n_head)
                        .arg(&n_kv)
                        .arg(&hd)
                        .arg(&self.pos)
                        .arg(&base)
                        .arg(&eps)
                        .arg(&(sh.rd as i32))
                        .arg(&0u64)
                        .arg(&sh.sec)
                        .launch(cfg)
                }
                .map_err(gerr)?;
            }
            let gqa = f.attn_gqa.as_ref();
            if let Some(k) = gqa {
                let cfg = LaunchConfig { grid_dim: (sh.n_kv as u32, self.gqa_splits as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    s.launch_builder(k)
                        .arg(&self.qkv)
                        .arg(&l.kc)
                        .arg(&l.vc)
                        .arg(&mut self.pm)
                        .arg(&mut self.pl)
                        .arg(&mut self.pacc)
                        .arg(&n_kv)
                        .arg(&self.pos)
                        .arg(&group)
                        .arg(&scale)
                        .launch(cfg)
                }
                .map_err(gerr)?;
                let splits = self.gqa_splits as i32;
                let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, 1, 1), block_dim: (sh.hd as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&f.attn_combine).arg(&self.pm).arg(&self.pl).arg(&self.pacc).arg(&mut self.att).arg(&hd).arg(&splits).launch(cfg) }.map_err(gerr)?;
            } else {
                let chunk = SPLIT as i32;
                let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, self.splits as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    s.launch_builder(&f.attn_partial)
                        .arg(&self.qkv)
                        .arg(&l.kc)
                        .arg(&l.vc)
                        .arg(&mut self.pm)
                        .arg(&mut self.pl)
                        .arg(&mut self.pacc)
                        .arg(&n_kv)
                        .arg(&hd)
                        .arg(&self.pos)
                        .arg(&group)
                        .arg(&scale)
                        .arg(&chunk)
                        .launch(cfg)
                }
                .map_err(gerr)?;
                let splits = self.splits as i32;
                let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, 1, 1), block_dim: (sh.hd as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&f.attn_combine).arg(&self.pm).arg(&self.pl).arg(&self.pacc).arg(&mut self.att).arg(&hd).arg(&splits).launch(cfg) }.map_err(gerr)?;
            }
            let qd = (sh.n_head * sh.hd) as i32;
            if sh.gate {
                Self::apply_gate(s, f, &mut self.att, &self.gate, sh.n_head * sh.hd)?;
            }
            unsafe { s.launch_builder(&f.quant).arg(&self.att).arg(&mut self.xq).arg(&mut self.xd).arg(&qd).launch(blocks(sh.n_head * sh.hd)) }.map_err(gerr)?;
            Self::matvec(s, f, &l.o, &self.xq, &self.xd, &mut self.x, true)?;
            }
            if let (Some(gu), Some(down)) = (&l.gu, &l.down) {
                unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&l.ffn_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
                Self::matvec(s, f, gu, &self.xq, &self.xd, &mut self.gu, false)?;
                unsafe { s.launch_builder(&f.silu_mul_quant).arg(&self.gu).arg(&mut self.xq).arg(&mut self.xd).arg(&ffn).launch(blocks(sh.ffn)) }.map_err(gerr)?;
                Self::matvec(s, f, down, &self.xq, &self.xd, &mut self.x, true)?;
                continue;
            }
            // Mixture of experts: route on the GPU, then cached experts on the GPU and the rest on the
            // CPU at the same time.
            let mo = self.moe.as_mut().expect("MoE state");
            let router = l.router.as_ref().expect("router");
            unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&l.ffn_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&mut mo.h).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
            let (ne, k) = (mo.ne as i32, mo.k as i32);
            {
                let tab = mo.slot_tab.slice(li * mo.ne..(li + 1) * mo.ne);
                unsafe { s.launch_builder(&f.router_topk).arg(router).arg(&mo.h).arg(&dim).arg(&ne).arg(&k).arg(&tab).arg(&mut mo.sel).launch(one) }.map_err(gerr)?;
            }
            // Asynchronous copies of the choice and the normed input for the CPU; the cached experts
            // are queued straight away, since the GPU needs nothing from the host to run them.
            s.memcpy_dtoh(&mo.sel, &mut mo.sel_h).map_err(gerr)?;
            s.memcpy_dtoh(&mo.h, &mut mo.h_pin).map_err(gerr)?;
            // Event tracking is off (single stream), so the copies are waited on explicitly.
            let copied = s.record_event(None).map_err(gerr)?;
            if let Some([kg, ku, kd]) = mo.kinds[li] {
                let eff = mo.eff;
                let sb = mo.slot_bytes as i64;
                let kk = mo.k;
                let launch_mv = |off: i64, em: &ExpMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, xstride: i32, y: &mut cudarc::driver::CudaViewMut<f32>, pool: &CudaSlice<u8>, sel: &CudaSlice<i32>| -> Result<(), Error> {
                    let (rows, cols) = (em.rows as i32, em.cols as i32);
                    let cfg = LaunchConfig { grid_dim: ((em.rows as u32).div_ceil(8), kk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(&f.moe_matvec).arg(pool).arg(&sb).arg(&off).arg(sel).arg(&em.dtype).arg(&em.bb).arg(xq).arg(xd).arg(&xstride).arg(y).arg(&rows).arg(&cols).launch(cfg) }.map_err(gerr)?;
                    Ok(())
                };
                launch_mv(0, &kg, &self.xq, &self.xd, 0, &mut mo.gu.slice_mut(0..kk * eff), &mo.pool, &mo.sel)?;
                launch_mv(kg.bytes as i64, &ku, &self.xq, &self.xd, 0, &mut mo.gu.slice_mut(kk * eff..2 * kk * eff), &mo.pool, &mo.sel)?;
                let (effi, ki) = (eff as i32, kk as i32);
                unsafe { s.launch_builder(&f.moe_silu).arg(&mo.gu).arg(&mut mo.axq).arg(&mut mo.axd).arg(&effi).arg(&ki).launch(blocks(kk * eff)) }.map_err(gerr)?;
                launch_mv((kg.bytes + ku.bytes) as i64, &kd, &mo.axq, &mo.axd, effi, &mut mo.eo.slice_mut(0..kk * sh.dim), &mo.pool, &mo.sel)?;
            }
            // CPU experts while the GPU runs its share. The cpu_pin upload of the previous layer is
            // complete too: it was queued before this layer's copies.
            let tw = std::time::Instant::now();
            copied.synchronize().map_err(gerr)?;
            mo.wait_s += tw.elapsed().as_secs_f64();
            let tc = std::time::Instant::now();
            let sel = mo.sel_h.as_slice().map_err(gerr)?.to_vec();
            let mut on_cpu: Vec<(usize, f32)> = Vec::with_capacity(mo.k);
            for t in 0..mo.k {
                let (e, w, slot) = (sel[t] as usize, f32::from_bits(sel[mo.k + t] as u32), sel[2 * mo.k + t]);
                mo.counts[li * mo.ne + e] += 1.0;
                mo.lookups += 1;
                if slot >= 0 {
                    mo.hits += 1;
                } else {
                    on_cpu.push((e, w));
                }
            }
            if on_cpu.is_empty() {
                mo.cpu_out.fill(0.0);
            } else {
                mo.h_h.copy_from_slice(mo.h_pin.as_slice().map_err(gerr)?);
                mo.qa.quantize(&mo.h_h);
                model.moe_cpu(li, &mo.h_h, &mo.qa, &on_cpu, &mut mo.scratch, &mut mo.cpu_out);
            }
            mo.cpu_s += tc.elapsed().as_secs_f64();
            mo.cpu_pin.as_mut_slice().map_err(gerr)?.copy_from_slice(&mo.cpu_out);
            s.memcpy_htod(&mo.cpu_pin, &mut mo.cpu).map_err(gerr)?;
            unsafe { s.launch_builder(&f.moe_combine).arg(&mut self.x).arg(&mo.eo).arg(&mo.sel).arg(&k).arg(&mo.cpu).arg(&dim).launch(blocks(sh.dim)) }.map_err(gerr)?;
        }
        let Some(output) = &self.output else {
            // Split: the CPU takes the activation from here.
            let hx = self.host_x.as_mut().expect("split host buffer");
            s.memcpy_dtoh(&self.x, hx).map_err(gerr)?;
            return Ok(());
        };
        unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        Self::matvec(s, f, output, &self.xq, &self.xd, &mut self.logits, false)?;
        let vocab = sh.vocab as i32;
        unsafe { s.launch_builder(&f.argmax).arg(&self.logits).arg(&vocab).arg(&mut self.next).arg(&mut self.tok).arg(&mut self.pos).launch(one) }.map_err(gerr)?;
        Ok(())
    }

    /// Captures the decode step as a graph the first time; replays it afterwards.
    fn run(&mut self, model: &Model) -> Result<(), Error> {
        if let Some(m) = self.moe.as_mut() {
            // Hybrid decoding syncs with the CPU every layer, so it is not captured as a graph.
            m.steps += 1;
            if m.steps % 8 == 0 {
                m.rebalance(&self.stream, model, 8)?;
            }
            return self.enqueue(model);
        }
        if self.graph.is_none() {
            self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).map_err(gerr)?;
            let r = self.enqueue(model);
            let g = self.stream.end_capture(CUgraphInstantiate_flags(0)).map_err(gerr)?;
            r?;
            self.graph = g;
        }
        match &self.graph {
            Some(g) => g.launch().map_err(gerr),
            None => self.enqueue(model),
        }
    }

    /// Runs `token` at `pos`. Copies the logits into `out` when given (sampling); returns the greedy
    /// next token either way.
    pub fn step(&mut self, model: &Model, token: u32, pos: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        if pos >= self.max_ctx {
            return Err(Error::Request("context full".into()));
        }
        self.stream.memcpy_htod(&[token as i32], &mut self.tok).map_err(gerr)?;
        self.stream.memcpy_htod(&[pos as i32, self.rope_shift], &mut self.pos).map_err(gerr)?;
        if let Some(row) = self.host_embed.as_mut() {
            model.embed_into(token, row);
        }
        self.run(model)?;
        if let Some(o) = out {
            self.stream.memcpy_dtoh(&self.logits, o).map_err(gerr)?;
        }
        let mut next = [0i32];
        self.stream.memcpy_dtoh(&self.next, &mut next).map_err(gerr)?;
        Ok(next[0] as u32)
    }

    /// Split models: runs the GPU's layers for `token` at `pos` and leaves the activation in `x`
    /// for the CPU's layers.
    pub fn step_split(&mut self, model: &Model, token: u32, pos: usize, x: &mut [f32]) -> Result<(), Error> {
        if pos >= self.max_ctx {
            return Err(Error::Request("context full".into()));
        }
        self.stream.memcpy_htod(&[token as i32], &mut self.tok).map_err(gerr)?;
        self.stream.memcpy_htod(&[pos as i32, self.rope_shift], &mut self.pos).map_err(gerr)?;
        if let Some(row) = self.host_embed.as_mut() {
            model.embed_into(token, row);
        }
        self.run(model)?;
        self.stream.synchronize().map_err(gerr)?;
        x.copy_from_slice(self.host_x.as_ref().expect("split host buffer").as_slice().map_err(gerr)?);
        Ok(())
    }

    /// Split models: batched GPU layers over one chunk of up to CHUNK tokens at pos0..; their
    /// activations (tokens x width) go to `xs` for the CPU's layers.
    pub fn prefill_split(&mut self, model: &Model, toks: &[u32], pos0: usize, xs: &mut Vec<f32>) -> Result<(), Error> {
        if toks.is_empty() || toks.len() > CHUNK || pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("prefill outside the context".into()));
        }
        self.prefill_chunk(model, toks, pos0, None)?;
        let n = toks.len() * self.sh.dim;
        xs.resize(n, 0.0);
        self.stream.memcpy_dtoh(&self.b.x.slice(0..n), xs.as_mut_slice()).map_err(gerr)?;
        Ok(())
    }

    /// Checking aid: the batched residual stream of the last prefill chunk (tokens x width).
    pub fn debug_batch_x(&self, t: usize) -> Result<Vec<f32>, Error> {
        let n = t * self.sh.dim;
        let mut v = vec![0f32; n];
        self.stream.memcpy_dtoh(&self.b.x.slice(0..n), v.as_mut_slice()).map_err(gerr)?;
        Ok(v)
    }

    /// After a prefill chunk: wait for it (so the count is finished work, not queued work) and
    /// publish how many prompt tokens are in the cache. Only when someone is watching.
    fn chunk_done(&mut self, pos: usize) -> Result<(), Error> {
        if let Some(p) = &self.progress {
            self.stream.synchronize().map_err(gerr)?;
            p.set(pos);
        }
        Ok(())
    }

    /// Saved-cache access: an attention layer's K and V (f16 bits) for positions 0..n, or None for
    /// a DeltaNet layer.
    pub fn kv_get(&self, li: usize, n: usize, kvd: usize) -> Result<Option<(Vec<u16>, Vec<u16>)>, Error> {
        let l = &self.layers[li];
        if l.delta.is_some() {
            return Ok(None);
        }
        let mut k = vec![0u16; n * kvd];
        let mut v = vec![0u16; n * kvd];
        self.stream.memcpy_dtoh(&l.kc.slice(0..n * kvd), k.as_mut_slice()).map_err(gerr)?;
        self.stream.memcpy_dtoh(&l.vc.slice(0..n * kvd), v.as_mut_slice()).map_err(gerr)?;
        Ok(Some((k, v)))
    }

    pub fn kv_set(&mut self, li: usize, k: &[u16], v: &[u16]) -> Result<(), Error> {
        let l = &mut self.layers[li];
        self.stream.memcpy_htod(k, &mut l.kc.slice_mut(0..k.len())).map_err(gerr)?;
        self.stream.memcpy_htod(v, &mut l.vc.slice_mut(0..v.len())).map_err(gerr)?;
        Ok(())
    }

    /// A DeltaNet layer's convolution history and recurrent state.
    pub fn rec_get(&self, li: usize) -> Result<Option<(Vec<f32>, Vec<f32>)>, Error> {
        let Some(dl) = self.layers[li].delta.as_ref() else { return Ok(None) };
        Ok(Some((self.stream.clone_dtoh(&dl.hist).map_err(gerr)?, self.stream.clone_dtoh(&dl.state).map_err(gerr)?)))
    }

    pub fn rec_set(&mut self, li: usize, hist: &[f32], state: &[f32]) -> Result<(), Error> {
        let Some(dl) = self.layers[li].delta.as_mut() else { return Err(Error::Format("not a DeltaNet layer".into())) };
        if hist.len() != dl.hist.len() || state.len() != dl.state.len() {
            return Err(Error::Format("saved state does not fit this model".into()));
        }
        self.stream.memcpy_htod(hist, &mut dl.hist).map_err(gerr)?;
        self.stream.memcpy_htod(state, &mut dl.state).map_err(gerr)?;
        Ok(())
    }

    /// Speculation can verify drafts on this model (whole GPU, attention layers only).
    pub fn can_verify(&self) -> bool {
        self.spec.is_some()
    }

    /// One batched forward over `toks` at pos0.. (a token and its drafts, at most SPEC_MAX) that
    /// writes the KV cache and returns the logits of every position (tokens x vocab) in `logits`.
    pub fn verify(&mut self, model: &Model, toks: &[u32], pos0: usize, logits: &mut Vec<f32>) -> Result<(), Error> {
        if !self.can_verify() || toks.is_empty() || toks.len() > SPEC_MAX || pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("verify outside its limits".into()));
        }
        self.prefill_chunk_with(model, toks, pos0, None, Some(logits)).map(|_| ())
    }

    /// Tokens per split prefill chunk.
    pub const SPLIT_CHUNK: usize = CHUNK;

    /// Batched forward over `toks` at positions pos0.. (prefill, or verifying a draft): every weight is
    /// read once per chunk of up to CHUNK tokens. Writes the KV cache for all of them and the logits of
    /// the last one into `out` when given; returns the greedy next token.
    pub fn prefill(&mut self, model: &Model, toks: &[u32], pos0: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        if toks.is_empty() || pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("prefill outside the context".into()));
        }
        if let Some(mo) = self.moe.as_mut() {
            // Older history counts half as much once a new prompt arrives: the prompt's own routing
            // is the best predictor of the experts its answer will use.
            for c in mo.counts.iter_mut() {
                *c *= 0.5;
            }
            let mut out = out;
            let mut last = 0u32;
            for (ci, chunk) in toks.chunks(CHUNK).enumerate() {
                let is_last = (ci + 1) * CHUNK >= toks.len();
                last = self.prefill_moe_chunk(model, chunk, pos0 + ci * CHUNK, if is_last { out.as_deref_mut() } else { None })?;
                self.chunk_done(pos0 + ci * CHUNK + chunk.len())?;
            }
            // Prompt-aware placement: move the experts this prompt used most into VRAM before
            // generation starts (bounded so a short prompt does not pay for a large reshuffle).
            let mo = self.moe.as_mut().expect("MoE state");
            let max = (mo.owner.len() / 4).min(toks.len() * 8);
            mo.rebalance(&self.stream, model, max)?;
            return Ok(last);
        }
        let mut last = 0u32;
        let mut out = out;
        for (ci, chunk) in toks.chunks(CHUNK).enumerate() {
            let p0 = pos0 + ci * CHUNK;
            let is_last = (ci + 1) * CHUNK >= toks.len();
            last = self.prefill_chunk(model, chunk, p0, if is_last { out.as_deref_mut() } else { None })?;
            self.chunk_done(p0 + chunk.len())?;
        }
        Ok(last)
    }

    /// Batched hybrid forward for MoE models: attention for all tokens on the GPU; per layer the
    /// router runs for every token, tokens are grouped by expert, cached experts run on the GPU over
    /// their tokens (weights read once per tile of tokens) while the CPU runs the rest, likewise once
    /// per expert.
    fn prefill_moe_chunk(&mut self, model: &Model, toks: &[u32], pos0: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        let t = toks.len();
        let sh = &self.sh;
        let s = &self.stream;
        let f = &self.f;
        let (dim, hd) = (sh.dim as i32, sh.hd as i32);
        let (n_head, n_kv) = (sh.n_head as i32, sh.n_kv as i32);
        let (eps, base) = (sh.eps, sh.base);
        let ti = t as i32;
        let null_f: u64 = 0;
        let null_i: u64 = 0;
        let b = &mut self.b;
        // Embeddings on the CPU.
        let mut emb = vec![0f32; t * sh.dim];
        for (i, &tok) in toks.iter().enumerate() {
            model.embed_into(real_id(tok), &mut emb[i * sh.dim..(i + 1) * sh.dim]);
        }
        s.memcpy_htod(&emb, &mut b.x.slice_mut(0..t * sh.dim)).map_err(gerr)?;
        s.memcpy_htod(&[pos0 as i32, self.rope_shift], &mut b.pos0).map_err(gerr)?;
        let tab = self.image.as_ref().and_then(|im| im.table(pos0, t));
        if let Some(tb) = tab {
            s.memcpy_htod(tb, &mut b.rtab.slice_mut(0..tb.len())).map_err(gerr)?;
        }
        if let Some(im) = &self.image {
            for (slot, row) in im.rows.iter().filter(|(sl, _)| (pos0..pos0 + t).contains(sl)) {
                let o = (slot - pos0) * sh.dim;
                s.memcpy_htod(row, &mut b.x.slice_mut(o..o + sh.dim)).map_err(gerr)?;
            }
        }
        let use_tab = tab.is_some();
        let norm_cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let kmm = |w: &cudarc::driver::CudaView<u8>, dtype: i32, bb: i32, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, idx: Option<&cudarc::driver::CudaView<i32>>, n: usize, y: &mut cudarc::driver::CudaViewMut<f32>, ystride: usize, rows: usize, cols: usize, add: bool| -> Result<(), Error> {
            let (ni, ys, r, c, a) = (n as i32, ystride as i32, rows as i32, cols as i32, add as i32);
            let cfg = LaunchConfig { grid_dim: ((rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut lb = s.launch_builder(&f.kmatmul);
            lb.arg(w).arg(&dtype).arg(&bb).arg(xq).arg(xd);
            match idx {
                Some(v) => lb.arg(v),
                None => lb.arg(&null_i),
            };
            unsafe { lb.arg(&ni).arg(y).arg(&ys).arg(&r).arg(&c).arg(&a).launch(cfg) }.map_err(gerr)?;
            Ok(())
        };
        // Dense matrix (K parts stacked by rows) over all T tokens.
        let dense = |m: &GMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, ystride: usize, add: bool| -> Result<(), Error> {
            let GMat::K { parts, cols } = m else { return Err(Error::Unsupported("int8 weights in the MoE prefill".into())) };
            let mut r0 = 0;
            for p in parts {
                let mut yv = y.slice_mut(r0..);
                kmm(&p.w.slice(0..), p.dtype, p.bb, xq, xd, None, t, &mut yv, ystride, p.rows, *cols, add)?;
                r0 += p.rows;
            }
            Ok(())
        };
        let qd = sh.n_head * sh.hd;
        let row = qd + 2 * sh.n_kv * sh.hd;
        for li in 0..self.layers.len() {
            let l = &mut self.layers[li];
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.attn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&null_f).launch(norm_cfg) }.map_err(gerr)?;
            dense(&l.qkv, &b.xq, &b.xd, &mut b.qkv, row, false)?;
            {
                let qn = l.q_norm.as_ref().unwrap_or(&l.attn_norm);
                let kn = l.k_norm.as_ref().unwrap_or(&l.attn_norm);
                let cfg = LaunchConfig { grid_dim: ((sh.n_head + sh.n_kv) as u32, t as u32, 1), block_dim: ((sh.hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                let no_tab = 0u64;
                let rd = sh.rd as i32;
                let has_norm = l.q_norm.is_some() as i32;
                let mut lb = s.launch_builder(&f.qkv_post);
                lb.arg(&mut b.qkv).arg(qn).arg(kn).arg(&has_norm).arg(&mut l.kc).arg(&mut l.vc);
                lb.arg(&n_head).arg(&n_kv).arg(&hd).arg(&b.pos0).arg(&base).arg(&eps).arg(&rd);
                if use_tab {
                    lb.arg(&b.rtab);
                } else {
                    lb.arg(&no_tab);
                }
                lb.arg(&sh.sec);
                unsafe { lb.launch(cfg) }
                .map_err(gerr)?;
                Self::attn_batch(s, f, sh, &b.qkv, &l.kc, &l.vc, &mut b.att, &b.pos0, t)?;
            }
            let qn_total = (t * qd) as i32;
            unsafe { s.launch_builder(&f.quant).arg(&b.att).arg(&mut b.xq).arg(&mut b.xd).arg(&qn_total).launch(LaunchConfig { grid_dim: ((qn_total as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            dense(&l.o, &b.xq, &b.xd, &mut b.x, sh.dim, true)?;
            // Router for every token.
            let mo = self.moe.as_mut().expect("MoE state");
            let router = l.router.as_ref().expect("router");
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.ffn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&mut mo.bh).launch(norm_cfg) }.map_err(gerr)?;
            let (ne, k) = (mo.ne as i32, mo.k as i32);
            {
                let tab = mo.slot_tab.slice(li * mo.ne..(li + 1) * mo.ne);
                let cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&f.router_b).arg(router).arg(&mo.bh).arg(&dim).arg(&ne).arg(&k).arg(&tab).arg(&mut mo.bsel).launch(cfg) }.map_err(gerr)?;
            }
            let mut sel = vec![0i32; t * 3 * mo.k];
            s.memcpy_dtoh(&mo.bsel.slice(0..t * 3 * mo.k), &mut sel).map_err(gerr)?;
            // Group the tokens by expert.
            let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); mo.ne];
            let mut slot_of_e: Vec<i32> = vec![-1; mo.ne];
            for tt in 0..t {
                let st = &sel[tt * 3 * mo.k..(tt + 1) * 3 * mo.k];
                for j in 0..mo.k {
                    let e = st[j] as usize;
                    by_expert[e].push((tt, f32::from_bits(st[mo.k + j] as u32)));
                    slot_of_e[e] = st[2 * mo.k + j];
                    mo.counts[li * mo.ne + e] += 1.0;
                }
            }
            let mut gpu_groups: Vec<(usize, i32)> = Vec::new();
            let mut stream_groups: Vec<usize> = Vec::new();
            let mut cpu_groups: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
            let can_stream = mo.kinds[li].is_some();
            for e in 0..mo.ne {
                if by_expert[e].is_empty() {
                    continue;
                }
                if slot_of_e[e] >= 0 {
                    gpu_groups.push((e, slot_of_e[e]));
                } else if can_stream && by_expert[e].len() >= STREAM_MIN_TOKENS {
                    stream_groups.push(e);
                } else {
                    cpu_groups.push((e, std::mem::take(&mut by_expert[e])));
                }
            }
            // Cached experts on the GPU, one expert at a time over its tokens.
            if let (Some([kg, ku, kd]), false) = (mo.kinds[li], gpu_groups.is_empty() && stream_groups.is_empty()) {
                let mut idx: Vec<i32> = Vec::new();
                let mut wv: Vec<f32> = Vec::new();
                // (offset, tokens, cache slot or -1 - ring index for streamed experts, expert)
                let mut spans = Vec::new();
                let streamed = stream_groups.iter().enumerate().map(|(i, &e)| (e, -1 - i as i32));
                for (e, slot) in gpu_groups.iter().copied().chain(streamed) {
                    spans.push((idx.len(), by_expert[e].len(), slot, e));
                    for &(tt, w) in &by_expert[e] {
                        idx.push(tt as i32);
                        wv.push(w);
                    }
                }
                s.memcpy_htod(&idx, &mut mo.bidx.slice_mut(0..idx.len())).map_err(gerr)?;
                s.memcpy_htod(&wv, &mut mo.bw.slice_mut(0..wv.len())).map_err(gerr)?;
                let eff = mo.eff;
                let sb = mo.slot_bytes;
                for &(o, n, slot, e) in &spans {
                    // Streamed expert: copy it into the next ring slot (host side, then DMA on the
                    // copy stream once the slot's previous expert has been computed).
                    let ring = (slot < 0).then(|| (-1 - slot) as usize % RING);
                    if let Some(r) = ring {
                        {
                            // Parallel copy into the pinned slot (one thread copies ~10 GB/s; the
                            // memory bus allows several times that).
                            let host = mo.ring_host[r].as_mut_slice().map_err(gerr)?;
                            struct H(*mut u8);
                            // SAFETY: items write disjoint byte ranges of the slot.
                            unsafe impl Sync for H {}
                            impl H {
                                fn at(&self, i: usize) -> *mut u8 {
                                    self.0.wrapping_add(i)
                                }
                            }
                            let dst = H(host.as_mut_ptr());
                            let parts = MoeGpu::expert_bytes(model, li, e);
                            let mut items = Vec::new();
                            let mut off = 0;
                            for (pi, (dt, src)) in parts.iter().enumerate() {
                                // Q6_K is re-laid out block by block; other formats copy in 64 KB pieces.
                                let (sbb, dbb) = if *dt == DType::Q6K { (210, 224) } else { (65536, 65536) };
                                let nblk = src.len().div_ceil(sbb);
                                let per = nblk.div_ceil(8).max(1);
                                for b0 in (0..nblk).step_by(per) {
                                    items.push((pi, off, b0, (b0 + per).min(nblk), sbb, dbb));
                                }
                                off += if *dt == DType::Q6K { nblk * dbb } else { src.len() };
                            }
                            crate::pool::global().run(items.len(), &|ii| {
                                let (pi, off, b0, b1, sbb, dbb) = items[ii];
                                let src = parts[pi].1;
                                for bi in b0..b1 {
                                    let n = sbb.min(src.len() - bi * sbb);
                                    // SAFETY: block bi of part pi maps to its own destination range.
                                    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(bi * sbb), dst.at(off + bi * dbb), n) };
                                }
                            });
                        }
                        if let Some(ev) = &mo.ring_done[r] {
                            mo.copy.wait(ev).map_err(gerr)?;
                        }
                        mo.copy.memcpy_htod(&mo.ring_host[r], &mut mo.ring_dev.slice_mut(r * sb..(r + 1) * sb)).map_err(gerr)?;
                        let arrived = mo.copy.record_event(None).map_err(gerr)?;
                        s.wait(&arrived).map_err(gerr)?;
                    }
                    let (pool, base) = match ring {
                        Some(r) => (&mo.ring_dev, r * sb),
                        None => (&mo.pool, slot as usize * sb),
                    };
                    let iv = mo.bidx.slice(o..o + n);
                    let gw = pool.slice(base..base + kg.bytes);
                    let uw = pool.slice(base + kg.bytes..base + kg.bytes + ku.bytes);
                    let dw = pool.slice(base + kg.bytes + ku.bytes..base + kg.bytes + ku.bytes + kd.bytes);
                    kmm(&gw, kg.dtype, kg.bb, &b.xq, &b.xd, Some(&iv), n, &mut mo.bgu.slice_mut(0..n * eff), eff, eff, kg.cols, false)?;
                    kmm(&uw, ku.dtype, ku.bb, &b.xq, &b.xd, Some(&iv), n, &mut mo.bgu.slice_mut(n * eff..2 * n * eff), eff, eff, ku.cols, false)?;
                    let (effi, ni) = (eff as i32, n as i32);
                    let cfg = LaunchConfig { grid_dim: (((n * eff) as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(&f.moe_silu).arg(&mo.bgu).arg(&mut mo.baxq).arg(&mut mo.baxd).arg(&effi).arg(&ni).launch(cfg) }.map_err(gerr)?;
                    kmm(&dw, kd.dtype, kd.bb, &mo.baxq, &mo.baxd, None, n, &mut mo.beo.slice_mut(0..n * sh.dim), sh.dim, sh.dim, kd.cols, false)?;
                    let wvv = mo.bw.slice(o..o + n);
                    let cfg = LaunchConfig { grid_dim: ((sh.dim as u32).div_ceil(256), n as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(&f.scatter_add).arg(&mut b.x).arg(&mo.beo).arg(&iv).arg(&wvv).arg(&ni).arg(&dim).launch(cfg) }.map_err(gerr)?;
                    if let Some(r) = ring {
                        mo.ring_done[r] = Some(s.record_event(None).map_err(gerr)?);
                    }
                }
            }
            // The rest on the CPU at the same time.
            if !cpu_groups.is_empty() {
                let mut h = vec![0f32; t * sh.dim];
                s.memcpy_dtoh(&mo.bh.slice(0..t * sh.dim), &mut h).map_err(gerr)?;
                let mut acc = vec![0f32; t * sh.dim];
                model.moe_cpu_batch(li, &cpu_groups, &h, &mut acc);
                s.memcpy_htod(&acc, &mut mo.bcpu.slice_mut(0..t * sh.dim)).map_err(gerr)?;
                let n = (t * sh.dim) as i32;
                unsafe { s.launch_builder(&f.add_inplace).arg(&mut b.x).arg(&mo.bcpu).arg(&n).launch(LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            }
            let _ = ti;
        }
        // Logits for the last token.
        {
            let last = b.x.slice((t - 1) * sh.dim..t * sh.dim);
            let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&last).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        }
        Self::matvec(s, f, self.output.as_ref().expect("output head"), &self.xq, &self.xd, &mut self.logits, false)?;
        let vocab = sh.vocab as i32;
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.argmax).arg(&self.logits).arg(&vocab).arg(&mut self.next).arg(&mut self.tok).arg(&mut self.pos).launch(one) }.map_err(gerr)?;
        if let Some(o) = out {
            s.memcpy_dtoh(&self.logits, o).map_err(gerr)?;
        }
        let mut next = [0i32];
        s.memcpy_dtoh(&self.next, &mut next).map_err(gerr)?;
        Ok(next[0] as u32)
    }

    fn prefill_chunk(&mut self, model: &Model, toks: &[u32], pos0: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        self.prefill_chunk_with(model, toks, pos0, out, None)
    }

    /// `prefill_chunk`, optionally returning the logits of every position (speculative verify).
    fn prefill_chunk_with(&mut self, model: &Model, toks: &[u32], pos0: usize, out: Option<&mut [f32]>, all: Option<&mut Vec<f32>>) -> Result<u32, Error> {
        let t = toks.len();
        let sh = &self.sh;
        let s = &self.stream;
        let f = &self.f;
        let ids: Vec<i32> = toks.iter().map(|&v| real_id(v) as i32).collect();
        s.memcpy_htod(&ids, &mut self.b.toks).map_err(gerr)?;
        s.memcpy_htod(&[pos0 as i32, self.rope_shift], &mut self.b.pos0).map_err(gerr)?;
        let tab = self.image.as_ref().and_then(|im| im.table(pos0, t));
        if let Some(tb) = tab {
            s.memcpy_htod(tb, &mut self.b.rtab.slice_mut(0..tb.len())).map_err(gerr)?;
        }
        let use_tab = tab.is_some();
        let (dim, ffn, hd) = (sh.dim as i32, sh.ffn as i32, sh.hd as i32);
        let (n_head, n_kv) = (sh.n_head as i32, sh.n_kv as i32);
        let (eps, base) = (sh.eps, sh.base);
        let ti = t as i32;
        let rows_cfg = |rows: usize| LaunchConfig { grid_dim: ((rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let b = &mut self.b;
        if self.host_embed.is_some() {
            // Native layout: embeddings looked up on the CPU and uploaded.
            let mut emb = vec![0f32; t * sh.dim];
            for (i, &tok) in toks.iter().enumerate() {
                model.embed_into(real_id(tok), &mut emb[i * sh.dim..(i + 1) * sh.dim]);
            }
            s.memcpy_htod(&emb, &mut b.x.slice_mut(0..t * sh.dim)).map_err(gerr)?;
        } else {
            let (qs, ds, _, _) = self.embed.as_ref().or(self.output.as_ref()).expect("embedding table").q8();
            let cfg = LaunchConfig { grid_dim: ((sh.dim as u32).div_ceil(256), t as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.embed_b).arg(qs).arg(ds).arg(&b.toks).arg(&mut b.x).arg(&dim).launch(cfg) }.map_err(gerr)?;
        }
        if let Some(im) = &self.image {
            for (slot, row) in im.rows.iter().filter(|(sl, _)| (pos0..pos0 + t).contains(sl)) {
                let o = (slot - pos0) * sh.dim;
                s.memcpy_htod(row, &mut b.x.slice_mut(o..o + sh.dim)).map_err(gerr)?;
            }
        }
        let norm_cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let null_f: u64 = 0;
        let null_i: u64 = 0;
        let mm = |m: &GMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, add: bool| -> Result<(), Error> {
            if let GMat::K { parts, cols } = m {
                // The file's own format: one kmatmul per stacked part, rows offset into each token's row.
                let ystride: usize = parts.iter().map(|p| p.rows).sum();
                let mut r0 = 0;
                for p in parts {
                    let (ni, ys, r, c, a) = (t as i32, ystride as i32, p.rows as i32, *cols as i32, add as i32);
                    let cfg = LaunchConfig { grid_dim: ((p.rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let mut yv = y.slice_mut(r0..);
                    unsafe { s.launch_builder(&f.kmatmul).arg(&p.w).arg(&p.dtype).arg(&p.bb).arg(xq).arg(xd).arg(&null_i).arg(&ni).arg(&mut yv).arg(&ys).arg(&r).arg(&c).arg(&a).launch(cfg) }.map_err(gerr)?;
                    r0 += p.rows;
                }
                return Ok(());
            }
            let (qs, ds, mrows, mcols) = m.q8();
            let (rows, cols, add) = (mrows as i32, mcols as i32, add as i32);
            match &f.matmul_mma {
                Some(k) if t >= MMA_MIN_T => {
                    let cfg = LaunchConfig { grid_dim: ((mrows as u32).div_ceil(64), (t as u32).div_ceil(64), 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(k).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&rows).arg(&cols).arg(&ti).arg(&add).launch(cfg) }.map_err(gerr)?;
                }
                _ => unsafe { s.launch_builder(&f.matmul_b).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&rows).arg(&cols).arg(&ti).arg(&add).launch(rows_cfg(mrows)) }.map_err(gerr).map(|_| ())?,
            }
            Ok(())
        };
        let stop = self.debug_stop.unwrap_or(usize::MAX);
        for l in self.layers.iter_mut().take(stop) {
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.attn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&null_f).launch(norm_cfg) }.map_err(gerr)?;
            if let Some(dl) = l.delta.as_mut() {
                mm(&l.qkv, &b.xq, &b.xd, &mut b.qkv, false)?;
                Self::delta_core(s, f, sh, dl, &mut b.qkv, &mut b.att, t)?;
                let n = (t * sh.delta.map_or(0, |d| d.nv * d.dv)) as i32;
                unsafe { s.launch_builder(&f.quant).arg(&b.att).arg(&mut b.xq).arg(&mut b.xd).arg(&n).launch(LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
                mm(&l.o, &b.xq, &b.xd, &mut b.x, true)?;
            } else {
            if sh.gate {
                mm(&l.qkv, &b.xq, &b.xd, &mut b.qg, false)?;
                Self::unpack_gate(s, f, sh, &b.qg, &mut b.qkv, &mut b.gate, t)?;
            } else {
                mm(&l.qkv, &b.xq, &b.xd, &mut b.qkv, false)?;
            }
            {
                let qn = l.q_norm.as_ref().unwrap_or(&l.attn_norm);
                let kn = l.k_norm.as_ref().unwrap_or(&l.attn_norm);
                let cfg = LaunchConfig { grid_dim: ((sh.n_head + sh.n_kv) as u32, t as u32, 1), block_dim: ((sh.hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                let no_tab = 0u64;
                let rd = sh.rd as i32;
                let has_norm = l.q_norm.is_some() as i32;
                let mut lb = s.launch_builder(&f.qkv_post);
                lb.arg(&mut b.qkv).arg(qn).arg(kn).arg(&has_norm).arg(&mut l.kc).arg(&mut l.vc);
                lb.arg(&n_head).arg(&n_kv).arg(&hd).arg(&b.pos0).arg(&base).arg(&eps).arg(&rd);
                if use_tab {
                    lb.arg(&b.rtab);
                } else {
                    lb.arg(&no_tab);
                }
                lb.arg(&sh.sec);
                unsafe { lb.launch(cfg) }
                .map_err(gerr)?;
            }
            {
                Self::attn_batch(s, f, sh, &b.qkv, &l.kc, &l.vc, &mut b.att, &b.pos0, t)?;
            }
            let qn_total = (t * sh.n_head * sh.hd) as i32;
            if sh.gate {
                Self::apply_gate(s, f, &mut b.att, &b.gate, t * sh.n_head * sh.hd)?;
            }
            unsafe { s.launch_builder(&f.quant).arg(&b.att).arg(&mut b.xq).arg(&mut b.xd).arg(&qn_total).launch(LaunchConfig { grid_dim: ((qn_total as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            mm(&l.o, &b.xq, &b.xd, &mut b.x, true)?;
            }
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.ffn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&null_f).launch(norm_cfg) }.map_err(gerr)?;
            mm(l.gu.as_ref().expect("dense layer"), &b.xq, &b.xd, &mut b.gu, false)?;
            let n = (t * sh.ffn) as u32;
            unsafe { s.launch_builder(&f.silu_b).arg(&b.gu).arg(&mut b.xq).arg(&mut b.xd).arg(&ffn).arg(&ti).launch(LaunchConfig { grid_dim: (n.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            mm(l.down.as_ref().expect("dense layer"), &b.xq, &b.xd, &mut b.x, true)?;
        }
        let Some(output) = &self.output else {
            // Split: the caller copies the activations to the CPU.
            return Ok(0);
        };
        if let (Some(dst), Some(spec)) = (all, self.spec.as_mut()) {
            // Every position's logits: final norm over all rows, the head as one batched matmul.
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&self.out_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&null_f).launch(norm_cfg) }.map_err(gerr)?;
            mm(output, &b.xq, &b.xd, spec, false)?;
            dst.resize(t * sh.vocab, 0.0);
            s.memcpy_dtoh(&spec.slice(0..t * sh.vocab), dst.as_mut_slice()).map_err(gerr)?;
            return Ok(0);
        }
        // Logits for the last token only, through the decode buffers (and the normed state when asked).
        {
            let last = b.x.slice((t - 1) * sh.dim..t * sh.dim);
            let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            if self.hidden.is_some() && self.hidden_dev.is_none() {
                self.hidden_dev = Some(s.alloc_zeros::<f32>(sh.dim).map_err(gerr)?);
            }
            let mut lb = s.launch_builder(&f.rmsnorm_quant);
            lb.arg(&last).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd);
            match (self.hidden.is_some(), self.hidden_dev.as_mut()) {
                (true, Some(h)) => lb.arg(h),
                _ => lb.arg(&null_f),
            };
            unsafe { lb.arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        }
        if let (Some(h), Some(hd)) = (self.hidden.as_mut(), self.hidden_dev.as_ref()) {
            h.resize(sh.dim, 0.0);
            s.memcpy_dtoh(hd, h.as_mut_slice()).map_err(gerr)?;
        }
        Self::matvec(s, f, output, &self.xq, &self.xd, &mut self.logits, false)?;
        let vocab = sh.vocab as i32;
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        // argmax_next also bumps pos/tok, which decode overwrites before it runs, so this is harmless.
        unsafe { s.launch_builder(&f.argmax).arg(&self.logits).arg(&vocab).arg(&mut self.next).arg(&mut self.tok).arg(&mut self.pos).launch(one) }.map_err(gerr)?;
        if let Some(o) = out {
            s.memcpy_dtoh(&self.logits, o).map_err(gerr)?;
        }
        let mut next = [0i32];
        s.memcpy_dtoh(&self.next, &mut next).map_err(gerr)?;
        Ok(next[0] as u32)
    }

    /// Greedy decoding entirely on the GPU: each replay feeds its own pick into the next, with only a
    /// 4-byte read per token so generation can stop at an end-of-turn token.
    /// Returns the number of decode steps run; step i fed `first` (i = 0) or the token emitted before
    /// it. `emit` returning false stops generation.
    pub fn generate_greedy(&mut self, model: &Model, first: u32, pos: usize, max_new: usize, stop: &[u32], mut emit: impl FnMut(u32) -> bool) -> Result<usize, Error> {
        // A split model's later layers and head run on the processor: Session::step does both.
        if self.is_split() {
            return Err(Error::Unsupported("generate_greedy runs whole-GPU models only; a split model steps through Session::step".into()));
        }
        if self.moe.is_some() {
            let mut t = first;
            let mut n = 0;
            while n < max_new && pos + n < self.max_ctx {
                t = self.step(model, t, pos + n, None)?;
                n += 1;
                if stop.contains(&t) || !emit(t) {
                    break;
                }
            }
            return Ok(n);
        }
        self.stream.memcpy_htod(&[first as i32], &mut self.tok).map_err(gerr)?;
        self.stream.memcpy_htod(&[pos as i32, self.rope_shift], &mut self.pos).map_err(gerr)?;
        let mut n = 0;
        let mut next = [0i32];
        while n < max_new && pos + n < self.max_ctx {
            self.run(model)?;
            self.stream.memcpy_dtoh(&self.next, &mut next).map_err(gerr)?;
            n += 1;
            let t = next[0] as u32;
            if stop.contains(&t) || !emit(t) {
                break;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32 -> IEEE half bits (truncating; the reference decodes the same bits, so it stays exact).
    fn f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let man = b & 0x7f_ffff;
        if exp <= 0 {
            if exp < -10 {
                return sign;
            }
            return sign | (((man | 0x80_0000) >> (14 - exp)) as u16);
        }
        sign | ((exp as u16) << 10) | ((man >> 13) as u16)
    }

    fn f16_value(h: u16) -> f64 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let man = (h & 0x3ff) as f64;
        sign * if exp == 0 { man * 2f64.powi(-24) } else { (1.0 + man / 1024.0) * 2f64.powi(exp - 15) }
    }

    /// The grouped decode kernel against an f64 reference and against attn_partial, on random
    /// queries and an f16 cache, at lengths that hit one tile, a partial tile and many splits.
    #[test]
    fn grouped_decode_attention_matches_the_reference() {
        let Ok(ctx) = CudaContext::new(0) else { return eprintln!("no CUDA device; skipped") };
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX)).unwrap();
        let (n_head, n_kv, hd) = (16usize, 8usize, 128usize);
        let group = n_head / n_kv;
        let gqa = module.load_function("attn_decode_gqa_128_2").unwrap();
        let old = module.load_function("attn_partial").unwrap();
        let combine = module.load_function("attn_combine").unwrap();
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        for len in [1usize, 16, 31, 33, 300, 2048, 4100] {
            let max_ctx = 4352;
            let q: Vec<f32> = (0..n_head * hd).map(|_| rnd()).collect();
            let kc: Vec<u16> = (0..max_ctx * n_kv * hd).map(|_| f16_bits(rnd())).collect();
            let vc: Vec<u16> = (0..max_ctx * n_kv * hd).map(|_| f16_bits(rnd())).collect();
            let scale = 1.0 / (hd as f32).sqrt();
            // f64 reference.
            let mut want = vec![0f64; n_head * hd];
            for h in 0..n_head {
                let kvh = h / group;
                let s: Vec<f64> = (0..len).map(|t| (0..hd).map(|d| q[h * hd + d] as f64 * f16_value(kc[(t * n_kv + kvh) * hd + d])).sum::<f64>() * scale as f64).collect();
                let mx = s.iter().cloned().fold(f64::MIN, f64::max);
                let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in 0..hd {
                    want[h * hd + d] = (0..len).map(|t| w[t] * f16_value(vc[(t * n_kv + kvh) * hd + d])).sum::<f64>() / z;
                }
            }
            let dq = stream.clone_htod(&q).unwrap();
            let dk = stream.clone_htod(&kc).unwrap();
            let dv = stream.clone_htod(&vc).unwrap();
            let dpos = stream.clone_htod(&[(len - 1) as i32]).unwrap();
            let run = |f: &CudaFunction, grid: (u32, u32), splits: i32, gqa_kernel: bool| -> Vec<f32> {
                let mut pm = stream.alloc_zeros::<f32>(n_head * splits as usize).unwrap();
                let mut pl = stream.alloc_zeros::<f32>(n_head * splits as usize).unwrap();
                let mut pacc = stream.alloc_zeros::<f32>(n_head * splits as usize * hd).unwrap();
                let mut out = stream.alloc_zeros::<f32>(n_head * hd).unwrap();
                let (n_kv_i, hd_i, group_i) = (n_kv as i32, hd as i32, group as i32);
                let cfg = LaunchConfig { grid_dim: (grid.0, grid.1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                let chunk = SPLIT as i32;
                unsafe {
                    let mut b = stream.launch_builder(f);
                    b.arg(&dq).arg(&dk).arg(&dv).arg(&mut pm).arg(&mut pl).arg(&mut pacc).arg(&n_kv_i);
                    if gqa_kernel {
                        b.arg(&dpos).arg(&group_i).arg(&scale);
                    } else {
                        b.arg(&hd_i).arg(&dpos).arg(&group_i).arg(&scale).arg(&chunk);
                    }
                    b.launch(cfg).unwrap();
                    let cfg = LaunchConfig { grid_dim: (n_head as u32, 1, 1), block_dim: (hd as u32, 1, 1), shared_mem_bytes: 0 };
                    stream.launch_builder(&combine).arg(&pm).arg(&pl).arg(&pacc).arg(&mut out).arg(&hd_i).arg(&splits).launch(cfg).unwrap();
                }
                stream.clone_dtoh(&out).unwrap()
            };
            let gqa_splits = 46i32;
            let new = run(&gqa, (n_kv as u32, gqa_splits as u32), gqa_splits, true);
            let old_splits = max_ctx.div_ceil(SPLIT) as i32;
            let old_out = run(&old, (n_head as u32, old_splits as u32), old_splits, false);
            let err = |got: &[f32]| got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0f64, f64::max);
            let (e_new, e_old) = (err(&new), err(&old_out));
            eprintln!("len {len:>5}: max |error| grouped {e_new:.2e}, previous kernel {e_old:.2e}");
            assert!(e_new < 1e-4, "grouped kernel error {e_new} at len {len}");
            assert!(e_new <= e_old * 4.0 + 1e-6, "grouped kernel is much less accurate than before at len {len}");
        }
    }
}
