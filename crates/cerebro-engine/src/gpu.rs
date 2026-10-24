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
use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PinnedHostSlice, PushKernelArg};
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
    /// Q8_0 blocks into int8 values and f32 scales (streamed split layers).
    q80_split: CudaFunction,
    /// Q6_K blocks to the GPU's 224-byte stride (streamed split layers read straight from the file).
    q6k_widen: CudaFunction,
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
    /// The same GEMM with 128 x 64 blocks, for big matrices and long prompts.
    matmul_mma_big: Option<CudaFunction>,
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
    k_to_q8: CudaFunction,
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
        // i-quants and Q2_K as stored (kernels.cu iq_vals); IQ4_NL's 18-byte blocks eight at a time.
        DType::Q2K => Some((84, 10)),
        DType::IQ2XXS => Some((66, 16)),
        DType::IQ3XXS => Some((98, 18)),
        DType::IQ1S => Some((50, 19)),
        DType::IQ4NL => Some((144, 20)),
        DType::IQ3S => Some((110, 21)),
        DType::IQ2S => Some((82, 22)),
        DType::IQ4XS => Some((136, 23)),
        DType::IQ1M => Some((56, 29)),
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
    /// Split models: the output head, run after the processor's layers (`head_split`).
    split_head: Option<GMat>,
    /// Split models: what the processor's layers stream through for prompt batches (made on first
    /// use; see `prefill_streamed`).
    pipe: Option<Pipe>,
}

/// Split prefill: the processor's layers pass through two GPU slots. The host copies a layer into
/// a pinned buffer and a copy stream sends it to one slot while the other slot's layer computes.
struct Pipe {
    slots: [Option<GLayer>; 2],
    /// Q8_0 bytes per slot, split into int8 and scales on the compute stream.
    raw: [Option<CudaSlice<u8>>; 2],
    /// (group, byte offset in `raw`, first row in the group, rows, cols) per Q8_0 matrix in a slot.
    splits: [Vec<(usize, usize, usize, usize, usize)>; 2],
    /// Three host buffers, so filling layer j+2 never waits for layer j's upload.
    host: [Option<PinnedHostSlice<u8>>; 3],
    copy: Arc<CudaStream>,
    /// Per streamed layer of the chunk: its upload is done (its host buffer is free, its compute
    /// may start).
    copied: Vec<Option<CudaEvent>>,
    /// The slot's layer is done computing (the slot may be overwritten).
    freed: [Option<CudaEvent>; 2],
    /// The new KV rows of every streamed layer, back on the host (layers x 2 x tokens x kvd).
    kv_back: Option<PinnedHostSlice<u16>>,
    /// The processor layers' weights in the model file's memory, page-locked for the GPU to read
    /// directly (no host copy): page-aligned ranges registered, or None when that failed.
    direct: Option<Vec<usize>>,
    /// Q6_K bytes as stored, widened on the GPU (direct mode).
    raw6: [Option<CudaSlice<u8>>; 2],
    /// A slot's DeltaNet buffers while it holds an attention layer.
    gdelta: [Option<GDelta>; 2],
    /// The convolution history and state of every streamed DeltaNet layer, back on the host.
    rec_back: Option<PinnedHostSlice<f32>>,
}

/// f32 values as their bytes.
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain floats viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        for &p in self.direct.iter().flatten() {
            // SAFETY: pointers registered by `register_direct`, unregistered once.
            unsafe { cudarc::driver::sys::cuMemHostUnregister(p as *mut std::ffi::c_void) };
        }
    }
}

/// Page-locks the file memory of layers `from..` so the GPU copies it directly. None (with nothing
/// left registered) when the driver refuses; CEREBRO_NO_DIRECT_STREAM skips it.
fn register_direct(model: &Model, from: usize) -> Option<Vec<usize>> {
    if std::env::var_os("CEREBRO_NO_DIRECT_STREAM").is_some() {
        return None;
    }
    let g = model.gguf();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for l in &model.layers()[from..] {
        let Ffn::Dense { gate, up, down } = &l.ffn else { return None };
        let mix: Vec<&Mat> = match &l.delta {
            Some(d) => vec![&d.qkv, &d.z, &d.alpha, &d.beta, &d.out],
            None => vec![&l.q, &l.k, &l.v, &l.o],
        };
        for m in mix.into_iter().chain([gate, up, down]) {
            let b = g.bytes_at(m.offset, m.rows * m.row_bytes);
            let a = b.as_ptr() as usize;
            spans.push((a & !4095, (a + b.len() + 4095) & !4095));
        }
    }
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in spans {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    // Page-locked memory cannot be paged out: only when it leaves the computer its reserve.
    // Otherwise the layers are copied through the pinned slots instead (a little slower).
    let locked: usize = merged.iter().map(|(a, b)| b - a).sum();
    if locked > crate::gguf::prefetch_room() {
        eprintln!("split prefill: {:.1} GB of the processor's layers would not leave enough free memory page-locked; copying them instead", locked as f64 / 1e9);
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut done = Vec::new();
    for &(a, b) in &merged {
        // SAFETY: a mapped, readable range of the model file, unregistered in Pipe::drop.
        let r = unsafe { cudarc::driver::sys::cuMemHostRegister_v2(a as *mut std::ffi::c_void, b - a, cudarc::driver::sys::CU_MEMHOSTREGISTER_READ_ONLY) };
        if r != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            eprintln!("split prefill: the GPU cannot read the model file's memory directly ({r:?}); copying it instead");
            for &p in &done {
                unsafe { cudarc::driver::sys::cuMemHostUnregister(p as *mut std::ffi::c_void) };
            }
            return None;
        }
        done.push(a);
    }
    if std::env::var_os("CEREBRO_PIPE_LOG").is_some() {
        eprintln!("pipe: {} ranges, {:.2} GB page-locked in {:.0} ms", merged.len(), merged.iter().map(|(a, b)| b - a).sum::<usize>() as f64 / 1e9, t0.elapsed().as_secs_f64() * 1e3);
    }
    Some(done)
}

/// Copies `jobs` (source bytes, destination offset, whether to widen Q6_K blocks to the GPU's
/// 224-byte layout) into `dst` on several threads.
fn fill_host(dst: &mut [u8], jobs: &[(&[u8], usize, bool)]) {
    const PIECE: usize = 4 << 20;
    // Pieces of whole Q6_K blocks (210 bytes in, 224 out) or plain bytes.
    let mut pieces = Vec::new();
    for (ji, &(src, _, q6)) in jobs.iter().enumerate() {
        let step = if q6 { PIECE / 210 * 210 } else { PIECE };
        let mut a = 0;
        while a < src.len() {
            let b = (a + step).min(src.len());
            pieces.push((ji, a, b));
            a = b;
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let base = dst.as_mut_ptr() as usize;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(ji, a, b)) = pieces.get(i) else { break };
                let (src, off, q6) = jobs[ji];
                // SAFETY: pieces write disjoint ranges of `dst`, which outlives the scope.
                unsafe {
                    let d = (base + off) as *mut u8;
                    if q6 {
                        for (k, blk) in src[a..b].chunks_exact(210).enumerate() {
                            std::ptr::copy_nonoverlapping(blk.as_ptr(), d.add((a / 210 + k) * 224), 210);
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(src[a..b].as_ptr(), d.add(a), b - a);
                    }
                }
            });
        }
    });
}

/// Bytes of `m` in the GPU's layout.
fn gpu_len(m: &Mat) -> usize {
    let n = m.rows * m.row_bytes;
    if m.dtype == DType::Q6K { n / 210 * 224 } else { n }
}

/// The processor's layers of a split, streamed through the GPU for one prompt chunk: layers
/// `from..` read their weights from `model` and their f16 KV rows from `k`/`v`.
struct StreamIn<'a> {
    /// Pinned buffers and slots for this chunk, taken from the state for its duration.
    pipe: Option<Pipe>,
    /// The processor's DeltaNet convolution histories and states (empty for attention layers).
    hist: &'a mut [Vec<f32>],
    state: &'a mut [Vec<f32>],
    from: usize,
    k: &'a mut [Vec<u16>],
    v: &'a mut [Vec<u16>],
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
        let (matmul_mma, attn_mma, attn_mma256, matmul_mma_big) = if mm_on || at_on {
            let m = ctx.load_module(Ptx::from_src(PTX_MMA)).map_err(gerr)?;
            let f = |on: bool, n: &str| if on { m.load_function(n).map(Some).map_err(gerr) } else { Ok(None) };
            let big = if std::env::var_os("CEREBRO_NO_BIG_GEMM").is_some() { None } else { f(mm_on, "matmul_q8_mma_big")? };
            (f(mm_on, "matmul_q8_mma")?, f(at_on, "attn_prefill_mma")?, f(at_on, "attn_prefill_mma256")?, big)
        } else {
            (None, None, None, None)
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
            matmul_mma_big,
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
            k_to_q8: lf("k_to_q8")?,
            q80_split: lf("q80_split")?,
            q6k_widen: lf("q6k_widen")?,
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
        let split_head = if split && split_head_on(model) { Some(up(&[model.output_mat()])?) } else { None };
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
            split_head,
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
            pipe: None,
        })
    }

    /// Split models with the head on the GPU: the final norm and the head over `x` (the residual after
    /// the processor's layers) into `logits`. Ok(false) when the head stays on the processor.
    pub fn head_split(&mut self, x: &[f32], logits: &mut [f32]) -> Result<bool, Error> {
        let Some(head) = self.split_head.as_ref() else { return Ok(false) };
        let s = &self.stream;
        let f = &self.f;
        s.memcpy_htod(x, &mut self.x).map_err(gerr)?;
        let (dim, eps) = (self.sh.dim as i32, self.sh.eps);
        let null_f: u64 = 0;
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let mut lb = s.launch_builder(&f.rmsnorm_quant);
        lb.arg(&self.x).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f);
        unsafe { lb.arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        Self::matvec(s, f, head, &self.xq, &self.xd, &mut self.logits, false)?;
        s.memcpy_dtoh(&self.logits, &mut logits[..self.sh.vocab]).map_err(gerr)?;
        Ok(true)
    }

    /// Whether `prefill_streamed` can run this model's processor layers (`from..`): attention or
    /// Gated DeltaNet with a dense feed-forward, in formats the GPU converts.
    pub fn can_stream(model: &Model, from: usize) -> bool {
        let ok = |ms: &[&Mat]| ms.iter().all(|m| m.cols % 256 == 0 && gpu_block(m.dtype).is_some()) || ms.iter().all(|m| m.cols % 32 == 0 && m.dtype == DType::Q8_0);
        model.hc.is_none()
            && model.layers()[from..].iter().all(|l| {
                let mix = match &l.delta {
                    Some(d) => ok(&[&d.qkv, &d.z, &d.alpha, &d.beta]) && ok(&[&d.out]),
                    None => ok(&[&l.q, &l.k, &l.v]) && ok(&[&l.o]),
                };
                mix && match &l.ffn {
                    Ffn::Dense { gate, up, down } => ok(&[gate, up]) && ok(&[down]),
                    Ffn::Moe(_) => false,
                }
            })
    }

    /// Sends processor layer `li` (with its KV rows before `pos0`) to slot `si`: the host fills the
    /// slot's pinned buffer once its last upload is done, and the copy stream uploads it once the
    /// slot's last layer is done computing.
    #[allow(clippy::too_many_arguments)]
    fn pipe_issue(p: &mut Pipe, ctx: &Arc<CudaContext>, f: &Funcs, sh: &Shape, max_ctx: usize, model: &Model, li: usize, j: usize, pos0: usize, k: &[u16], v: &[u16], hist: &[f32], state: &[f32]) -> Result<(), Error> {
        let (si, hb) = (j % 2, j % 3);
        let direct = p.direct.is_some();
        let g = model.gguf();
        let l = &model.layers()[li];
        let Ffn::Dense { gate, up, down } = &l.ffn else { return Err(Error::Unsupported("streamed MoE layer".into())) };
        let (g0, g1): (Vec<&Mat>, Vec<&Mat>) = match &l.delta {
            Some(d) => (vec![&d.qkv, &d.z, &d.alpha, &d.beta], vec![&d.out]),
            None => (vec![&l.q, &l.k, &l.v], vec![&l.o]),
        };
        let (g2, g3) = ([gate, up], [down]);
        let groups: [&[&Mat]; 4] = [&g0, &g1, &g2, &g3];
        let kvd = sh.n_kv * sh.hd;
        let nkv = if l.delta.is_none() { pos0 * kvd } else { 0 };
        // Small vectors: the norms, a DeltaNet layer's own tensors and its recurrent state.
        let none: &[f32] = &[];
        let dv = |pick: fn(&crate::model::Delta) -> &[f32]| l.delta.as_ref().map_or(none, pick);
        let norms: Vec<&[u8]> = [
            &l.attn_norm[..],
            &l.ffn_norm[..],
            l.q_norm.as_deref().unwrap_or(none),
            l.k_norm.as_deref().unwrap_or(none),
            dv(|d| &d.a),
            dv(|d| &d.dt_bias),
            dv(|d| &d.conv),
            dv(|d| &d.norm),
            if l.delta.is_some() { hist } else { none },
            if l.delta.is_some() { state } else { none },
        ]
        .into_iter()
        .map(f32_bytes)
        .collect();
        // Host layout: every matrix, then the norms, then the earlier K and V rows.
        let mut jobs: Vec<(&[u8], usize, bool)> = Vec::new();
        let mut off = 0usize;
        let mut mat_off = Vec::new();
        for grp in &groups {
            for m in grp.iter() {
                mat_off.push(off);
                if !direct {
                    jobs.push((g.bytes_at(m.offset, m.rows * m.row_bytes), off, m.dtype == DType::Q6K));
                    off += gpu_len(m).next_multiple_of(256);
                }
            }
        }
        let norm_off: Vec<usize> = norms
            .iter()
            .map(|n| {
                let o = off;
                off += n.len().next_multiple_of(256);
                o
            })
            .collect();
        for (i, n) in norms.iter().enumerate() {
            if !n.is_empty() {
                jobs.push((*n, norm_off[i], false));
            }
        }
        // SAFETY: f16 bits viewed as bytes.
        let kb = unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, nkv * 2) };
        let vb = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, nkv * 2) };
        let (k_off, v_off) = (off, off + (nkv * 2).next_multiple_of(256));
        if nkv > 0 {
            jobs.push((kb, k_off, false));
            jobs.push((vb, v_off, false));
        }
        let total = v_off + (nkv * 2).next_multiple_of(256);
        let tw = std::time::Instant::now();
        if let Some(e) = j.checked_sub(3).and_then(|i| p.copied.get(i)).and_then(|e| e.as_ref()) {
            e.synchronize().map_err(gerr)?;
        }
        let waited = tw.elapsed().as_secs_f64();
        if p.host[hb].as_ref().is_none_or(|h| h.len() < total) {
            p.host[hb] = None;
            // SAFETY: page-locked host memory, written here and read by the copy stream.
            p.host[hb] = Some(unsafe { ctx.alloc_pinned::<u8>(total + (64 << 20)) }.map_err(gerr)?);
        }
        let host = p.host[hb].as_mut().expect("made above").as_mut_slice().map_err(gerr)?;
        let tf = std::time::Instant::now();
        // Measuring aid: CEREBRO_PIPE_SKIP=fill|copy leaves out that stage after the first layers
        // (wrong results; for timing the rest of the pipeline only).
        let skip = std::env::var("CEREBRO_PIPE_SKIP").ok().filter(|_| j >= 3);
        if skip.is_none() {
            fill_host(host, &jobs);
        }
        if std::env::var_os("CEREBRO_PIPE_LOG").is_some() {
            let s = tf.elapsed().as_secs_f64();
            eprintln!("pipe: layer {li} slot {si}: waited {:.1} ms, filled {:.0} MB in {:.1} ms ({:.1} GB/s)", waited * 1e3, total as f64 / 1e6, s * 1e3, total as f64 / s / 1e9);
        }
        let c = p.copy.clone();
        // The slot's buffers, made or grown on the copy stream (ordered before the uploads).
        let z = |n: usize| c.alloc_zeros::<f32>(n.max(1)).map_err(gerr);
        if p.slots[si].is_none() {
            p.slots[si] = Some(GLayer {
                attn_norm: z(l.attn_norm.len())?,
                qkv: GMat::K { parts: Vec::new(), cols: 0 },
                o: GMat::K { parts: Vec::new(), cols: 0 },
                q_norm: l.q_norm.as_ref().map(|v| z(v.len())).transpose()?,
                k_norm: l.k_norm.as_ref().map(|v| z(v.len())).transpose()?,
                ffn_norm: z(l.ffn_norm.len())?,
                gu: None,
                down: None,
                router: None,
                kc: c.alloc_zeros::<u16>(max_ctx * kvd).map_err(gerr)?,
                vc: c.alloc_zeros::<u16>(max_ctx * kvd).map_err(gerr)?,
                delta: None,
            });
        }
        {
            // Norms the first layer lacked, and the DeltaNet buffers in or out of the slot.
            let sl = p.slots[si].as_mut().expect("made above");
            if l.q_norm.is_some() && sl.q_norm.is_none() {
                sl.q_norm = Some(z(l.q_norm.as_ref().map_or(0, |v| v.len()))?);
            }
            if l.k_norm.is_some() && sl.k_norm.is_none() {
                sl.k_norm = Some(z(l.k_norm.as_ref().map_or(0, |v| v.len()))?);
            }
            match &l.delta {
                Some(d) => {
                    if sl.delta.is_none() {
                        sl.delta = p.gdelta[si].take();
                    }
                    if sl.delta.is_none() {
                        let ss = model.cfg.ssm.expect("DeltaNet layers have a state shape");
                        sl.delta = Some(GDelta {
                            a: z(d.a.len())?,
                            dt_bias: z(d.dt_bias.len())?,
                            conv: z(d.conv.len())?,
                            norm: z(d.norm.len())?,
                            hist: z((ss.conv - 1) * ss.channels())?,
                            state: z(ss.nv * ss.dk * ss.dv)?,
                        });
                    }
                }
                None => {
                    if sl.delta.is_some() {
                        p.gdelta[si] = sl.delta.take();
                    }
                }
            }
        }
        if let Some(e) = p.freed[si].take() {
            c.wait(&e).map_err(gerr)?;
        }
        let host = p.host[hb].as_ref().expect("made above").as_slice().map_err(gerr)?;
        let cs = c.cu_stream();
        // SAFETY: async copies from page-locked memory that stays untouched until `copied` fires.
        let no_copy = skip.as_deref() == Some("copy");
        let up = |dst: u64, a: usize, n: usize| if no_copy { Ok(()) } else { unsafe { cudarc::driver::result::memcpy_htod_async(dst, &host[a..a + n], cs) }.map_err(gerr) };
        // SAFETY: async copies from the page-locked model file memory (register_direct).
        let up_file = |dst: u64, src: &[u8]| if no_copy { Ok(()) } else { unsafe { cudarc::driver::result::memcpy_htod_async(dst, src, cs) }.map_err(gerr) };
        let q6_max = groups.iter().flat_map(|g| g.iter()).filter(|m| m.dtype == DType::Q6K).map(|m| m.rows * m.row_bytes).max().unwrap_or(0);
        if direct && q6_max > 0 && p.raw6[si].as_ref().is_none_or(|r| r.len() < q6_max) {
            p.raw6[si] = Some(c.alloc_zeros::<u8>(q6_max).map_err(gerr)?);
        }
        let sl = p.slots[si].as_mut().expect("made above");
        p.splits[si].clear();
        let q8_bytes: usize = groups.iter().flat_map(|g| g.iter()).filter(|m| m.dtype == DType::Q8_0).map(|m| m.rows * m.row_bytes).sum();
        if q8_bytes > 0 && p.raw[si].as_ref().is_none_or(|r| r.len() < q8_bytes) {
            p.raw[si] = Some(c.alloc_zeros::<u8>(q8_bytes).map_err(gerr)?);
        }
        let mut raw_off = 0usize;
        let mut mi = 0usize;
        for (gi, grp) in groups.iter().enumerate() {
            let slot_mat: &mut Option<GMat> = match gi {
                0 => &mut None,
                1 => &mut None,
                2 => &mut sl.gu,
                _ => &mut sl.down,
            };
            // qkv and o are not Options in GLayer: go through a temporary.
            let mut tmp = match gi {
                0 => Some(std::mem::replace(&mut sl.qkv, GMat::K { parts: Vec::new(), cols: 0 })),
                1 => Some(std::mem::replace(&mut sl.o, GMat::K { parts: Vec::new(), cols: 0 })),
                _ => slot_mat.take(),
            };
            let cols = grp[0].cols;
            if grp[0].dtype == DType::Q8_0 {
                let rows: usize = grp.iter().map(|m| m.rows).sum();
                if !matches!(&tmp, Some(GMat::Q8 { qs, .. }) if qs.len() >= rows * cols) {
                    tmp = Some(GMat::Q8 { qs: c.alloc_zeros::<i8>(rows * cols).map_err(gerr)?, ds: c.alloc_zeros::<f32>(rows * cols / 32).map_err(gerr)?, rows, cols });
                }
                if let Some(GMat::Q8 { rows: r, cols: cc, .. }) = tmp.as_mut() {
                    (*r, *cc) = (rows, cols);
                }
                let raw = p.raw[si].as_ref().expect("made above");
                let mut r0 = 0;
                for m in grp.iter() {
                    let n = m.rows * m.row_bytes;
                    if direct {
                        up_file(raw.device_ptr(&c).0 + raw_off as u64, g.bytes_at(m.offset, n))?;
                    } else {
                        up(raw.device_ptr(&c).0 + raw_off as u64, mat_off[mi], n)?;
                    }
                    p.splits[si].push((gi, raw_off, r0, m.rows, cols));
                    raw_off += n;
                    r0 += m.rows;
                    mi += 1;
                }
            } else {
                if !matches!(&tmp, Some(GMat::K { .. })) {
                    tmp = Some(GMat::K { parts: Vec::new(), cols });
                }
                let Some(GMat::K { parts, cols: dc }) = tmp.as_mut() else { unreachable!() };
                *dc = cols;
                for (i, m) in grp.iter().enumerate() {
                    let (bb, dtype) = gpu_block(m.dtype).expect("can_stream checked the format");
                    let n = gpu_len(m);
                    if parts.len() <= i || parts[i].w.len() < n {
                        let part = KPart { w: c.alloc_zeros::<u8>(n).map_err(gerr)?, dtype, bb: bb as i32, rows: m.rows };
                        if parts.len() <= i {
                            parts.push(part);
                        } else {
                            parts[i] = part;
                        }
                    }
                    let pt = &mut parts[i];
                    (pt.dtype, pt.bb, pt.rows) = (dtype, bb as i32, m.rows);
                    if !direct {
                        up(pt.w.device_ptr(&c).0, mat_off[mi], n)?;
                    } else if m.dtype == DType::Q6K {
                        let src = g.bytes_at(m.offset, m.rows * m.row_bytes);
                        let r6 = p.raw6[si].as_ref().expect("made above");
                        up_file(r6.device_ptr(&c).0, src)?;
                        let nb = src.len() as i64;
                        let cfg = LaunchConfig { grid_dim: ((nb as u64).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                        if !no_copy {
                            unsafe { c.launch_builder(&f.q6k_widen).arg(r6).arg(&mut pt.w).arg(&nb).launch(cfg) }.map_err(gerr)?;
                        }
                    } else {
                        up_file(pt.w.device_ptr(&c).0, g.bytes_at(m.offset, n))?;
                    }
                    mi += 1;
                }
            }
            match gi {
                0 => sl.qkv = tmp.expect("set above"),
                1 => sl.o = tmp.expect("set above"),
                2 => sl.gu = tmp,
                _ => sl.down = tmp,
            }
        }
        let fl = |x: &CudaSlice<f32>| x.device_ptr(&c).0;
        up(fl(&sl.attn_norm), norm_off[0], norms[0].len())?;
        up(fl(&sl.ffn_norm), norm_off[1], norms[1].len())?;
        if let (Some(qn), false) = (sl.q_norm.as_ref(), norms[2].is_empty()) {
            up(fl(qn), norm_off[2], norms[2].len())?;
        }
        if let (Some(kn), false) = (sl.k_norm.as_ref(), norms[3].is_empty()) {
            up(fl(kn), norm_off[3], norms[3].len())?;
        }
        if let Some(gd) = sl.delta.as_ref() {
            for (i, buf) in [&gd.a, &gd.dt_bias, &gd.conv, &gd.norm, &gd.hist, &gd.state].into_iter().enumerate() {
                if !norms[4 + i].is_empty() {
                    up(fl(buf), norm_off[4 + i], norms[4 + i].len())?;
                }
            }
        }
        if nkv > 0 {
            up(sl.kc.device_ptr(&c).0, k_off, nkv * 2)?;
            up(sl.vc.device_ptr(&c).0, v_off, nkv * 2)?;
        }
        if p.copied.len() <= j {
            p.copied.resize_with(j + 1, || None);
        }
        p.copied[j] = Some(c.record_event(None).map_err(gerr)?);
        // WDDM batches submissions: push the copies to the GPU now.
        unsafe { cudarc::driver::sys::cuStreamQuery(cs) };
        Ok(())
    }

    /// Split models, one prompt chunk at pos0.. through every layer on the GPU: the resident layers,
    /// then the processor's layers (from `from`) streamed through a slot, their new KV rows written
    /// back to `k`/`v`. The last position's residual goes to `last` for the processor's head.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_streamed(&mut self, model: &Model, toks: &[u32], pos0: usize, from: usize, k: &mut [Vec<u16>], v: &mut [Vec<u16>], hist: &mut [Vec<f32>], state: &mut [Vec<f32>], last: &mut [f32]) -> Result<(), Error> {
        let t = toks.len();
        if t == 0 || t > CHUNK || pos0 + t > self.max_ctx || from != self.layers.len() {
            return Err(Error::Request("prefill outside the context".into()));
        }
        if self.pipe.is_none() {
            let copy = self.stream.context().new_stream().map_err(gerr)?;
            self.pipe = Some(Pipe { slots: [None, None], raw: [None, None], splits: [Vec::new(), Vec::new()], host: [None, None, None], copy, copied: Vec::new(), freed: [None, None], kv_back: None, direct: register_direct(model, from), raw6: [None, None], gdelta: [None, None], rec_back: None });
        }
        // The copy path (no page-locked file memory) gives wrong output on hybrid DeltaNet layers
        // (Qwen3.8 27B split answered "festivity"; with the processor running its layers it is
        // right): those models take the processor path until it is fixed.
        if self.pipe.as_ref().is_some_and(|p| p.direct.is_none()) && model.layers()[from..].iter().any(|l| l.delta.is_some()) {
            return Err(Error::Unsupported("streamed split prefill needs page-locked memory for hybrid layers".into()));
        }
        let pipe = self.pipe.take();
        let mut sin = StreamIn { pipe, hist, state, from, k, v };
        let t0 = std::time::Instant::now();
        let r = self.prefill_chunk_streamed(model, toks, pos0, Some(&mut sin));
        self.pipe = sin.pipe.take();
        r?;
        if std::env::var_os("CEREBRO_PIPE_LOG").is_some() {
            eprintln!("pipe: chunk of {t} at {pos0}: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let d = self.sh.dim;
        self.stream.memcpy_dtoh(&self.b.x.slice((t - 1) * d..t * d), last).map_err(gerr)?;
        Ok(())
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
/// Whether a split of `model` puts its output head on the GPU (switch "split_head"; not for
/// hyper-connection models, whose head mixes streams on the CPU).
fn split_head_on(model: &Model) -> bool {
    model.hc.is_none() && switches::enabled(switches::SPLIT_HEAD)
}

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
            let mut p = plan_dense(&ModelCost { layers, head_int8: head, head_native: head, scratch: 0, split_head: 0, file }, &mach, None);
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
        let split_head = if split_head_on(model) { native_bytes(out).unwrap_or(int8_bytes(out)) } else { 0 };
        let cost = ModelCost { layers, head_int8, head_native, scratch, split_head, file: model.gguf().len() as u64 };
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
        let mut room = if !switches::enabled(switches::PREFAULT) { 0 } else { crate::gguf::prefetch_room() };
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
        // With the RAM expert cache the processor's experts come from there, not the OS file cache.
        if model.gguf().experts().is_none() {
            model.gguf().touch(&touch);
            model.gguf().prefetch(&ahead);
        }
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

    fn prefill_chunk_streamed(&mut self, model: &Model, toks: &[u32], pos0: usize, sin: Option<&mut StreamIn<'_>>) -> Result<u32, Error> {
        self.prefill_chunk_full(model, toks, pos0, None, None, sin)
    }

    /// `prefill_chunk`, optionally returning the logits of every position (speculative verify).
    fn prefill_chunk_with(&mut self, model: &Model, toks: &[u32], pos0: usize, out: Option<&mut [f32]>, all: Option<&mut Vec<f32>>) -> Result<u32, Error> {
        self.prefill_chunk_full(model, toks, pos0, out, all, None)
    }

    fn prefill_chunk_full(&mut self, model: &Model, toks: &[u32], pos0: usize, out: Option<&mut [f32]>, all: Option<&mut Vec<f32>>, mut sin: Option<&mut StreamIn<'_>>) -> Result<u32, Error> {
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
        // Native-format weights are converted to int8 for a prompt batch so they take the
        // tensor-core GEMM too (k_to_q8): scratch for the largest such matrix, made on first use.
        let k8: std::cell::RefCell<Option<(CudaSlice<i8>, CudaSlice<f32>)>> = std::cell::RefCell::new(None);
        // CEREBRO_GPU_PROFILE: time the conversions and GEMMs of this chunk (synchronising after
        // each, so only for measuring); the rest of the chunk is attention, norms and the like.
        let profiling = std::env::var_os("CEREBRO_GPU_PROFILE").is_some();
        let prof = std::cell::Cell::new([0f64; 3]);
        let chunk_t0 = std::time::Instant::now();
        let lap = |i: usize, t: std::time::Instant| -> Result<(), Error> {
            if profiling {
                s.synchronize().map_err(gerr)?;
                let mut p = prof.get();
                p[i] += t.elapsed().as_secs_f64();
                prof.set(p);
            }
            Ok(())
        };
        let mm = |m: &GMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, add: bool| -> Result<(), Error> {
            if let (GMat::K { parts, cols }, Some(k), true) = (m, &f.matmul_mma, t >= MMA_MIN_T) {
                let rows: usize = parts.iter().map(|p| p.rows).sum();
                let n = rows * cols;
                let mut sc = k8.borrow_mut();
                if sc.as_ref().is_none_or(|(q, _)| q.len() < n) {
                    *sc = Some((s.alloc_zeros::<i8>(n).map_err(gerr)?, s.alloc_zeros::<f32>(n / 32).map_err(gerr)?));
                }
                let (qs, ds) = sc.as_mut().expect("scratch");
                if profiling {
                    s.synchronize().map_err(gerr)?;
                }
                let t_conv = std::time::Instant::now();
                let mut r0 = 0;
                for p in parts {
                    let (r, c) = (p.rows as i32, *cols as i32);
                    let subs = (p.rows * cols / 32) as u32;
                    let cfg = LaunchConfig { grid_dim: (subs.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let mut qv = qs.slice_mut(r0 * cols..(r0 + p.rows) * cols);
                    let mut dv = ds.slice_mut(r0 * cols / 32..(r0 + p.rows) * cols / 32);
                    unsafe { s.launch_builder(&f.k_to_q8).arg(&p.w).arg(&p.dtype).arg(&p.bb).arg(&r).arg(&c).arg(&mut qv).arg(&mut dv).launch(cfg) }.map_err(gerr)?;
                    r0 += p.rows;
                }
                lap(0, t_conv)?;
                let t_mm = std::time::Instant::now();
                let (rows_i, cols_i, add_i) = (rows as i32, *cols as i32, add as i32);
                let (k, bm) = match &f.matmul_mma_big {
                    Some(b) if rows >= 1024 && t >= 128 => (b, 128u32),
                    _ => (k, 64u32),
                };
                let cfg = LaunchConfig { grid_dim: ((t as u32).div_ceil(64), (rows as u32).div_ceil(bm), 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(k).arg(&*qs).arg(&*ds).arg(xq).arg(xd).arg(y).arg(&rows_i).arg(&cols_i).arg(&ti).arg(&add_i).launch(cfg) }.map_err(gerr)?;
                lap(1, t_mm)?;
                return Ok(());
            }
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
                    let (k, bm) = match &f.matmul_mma_big {
                        Some(b) if mrows >= 1024 && t >= 128 => (b, 128u32),
                        _ => (k, 64u32),
                    };
                    let cfg = LaunchConfig { grid_dim: ((t as u32).div_ceil(64), (mrows as u32).div_ceil(bm), 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(k).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&rows).arg(&cols).arg(&ti).arg(&add).launch(cfg) }.map_err(gerr)?;
                }
                _ => unsafe { s.launch_builder(&f.matmul_b).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&rows).arg(&cols).arg(&ti).arg(&add).launch(rows_cfg(mrows)) }.map_err(gerr).map(|_| ())?,
            }
            Ok(())
        };
        let stop = self.debug_stop.unwrap_or(usize::MAX);
        let n_res = self.layers.len().min(stop);
        let n_streamed = sin.as_ref().map_or(0, |si| model.layers().len() - si.from);
        let kvd = sh.n_kv * sh.hd;
        let ctx = self.stream.context().clone();
        let max_ctx = self.max_ctx;
        let issue = |sin: &mut Option<&mut StreamIn<'_>>, j: usize| -> Result<(), Error> {
            let si = sin.as_mut().expect("streamed layers");
            let li = si.from + j;
            let p = si.pipe.as_mut().expect("pipe");
            Self::pipe_issue(p, &ctx, f, sh, max_ctx, model, li, j, pos0, &si.k[li], &si.v[li], &si.hist[li], &si.state[li])
        };
        if n_streamed > 0 {
            let back = n_streamed * 2 * t * kvd;
            let p = sin.as_mut().expect("streamed layers").pipe.as_mut().expect("pipe");
            p.copied.clear();
            if let Some(ss) = model.cfg.ssm {
                let rec = n_streamed * ((ss.conv - 1) * ss.channels() + ss.nv * ss.dk * ss.dv);
                if p.rec_back.as_ref().is_none_or(|h| h.len() < rec) {
                    p.rec_back = None;
                    // SAFETY: page-locked host memory written by the GPU and read after a synchronize.
                    p.rec_back = Some(unsafe { ctx.alloc_pinned_with_flags::<f32>(rec, 0) }.map_err(gerr)?);
                }
            }
            if p.kv_back.as_ref().is_none_or(|h| h.len() < back) {
                p.kv_back = None;
                // SAFETY: page-locked host memory written by the GPU and read after a synchronize.
                p.kv_back = Some(unsafe { ctx.alloc_pinned_with_flags::<u16>(back, 0) }.map_err(gerr)?);
            }
            if n_res == 0 {
                for j in 0..n_streamed.min(2) {
                    issue(&mut sin, j)?;
                }
            }
        }
        for idx in 0..n_res + n_streamed {
            let l: &mut GLayer = if idx < n_res {
                &mut self.layers[idx]
            } else {
                let j = idx - n_res;
                let p = sin.as_mut().expect("streamed layers").pipe.as_mut().expect("pipe");
                s.wait(p.copied[j].as_ref().expect("issued")).map_err(gerr)?;
                let sl = p.slots[j % 2].as_mut().expect("issued");
                for &(gi, raw_off, r0, rows, cols) in &p.splits[j % 2] {
                    let raw = p.raw[j % 2].as_ref().expect("issued");
                    let m = match gi {
                        0 => &mut sl.qkv,
                        1 => &mut sl.o,
                        2 => sl.gu.as_mut().expect("issued"),
                        _ => sl.down.as_mut().expect("issued"),
                    };
                    let GMat::Q8 { qs, ds, .. } = m else { unreachable!() };
                    let nb = (rows * cols / 32) as i32;
                    let rv = raw.slice(raw_off..raw_off + rows * cols / 32 * 34);
                    let mut qv = qs.slice_mut(r0 * cols..(r0 + rows) * cols);
                    let mut dv = ds.slice_mut(r0 * cols / 32..(r0 + rows) * cols / 32);
                    let cfg = LaunchConfig { grid_dim: ((nb as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(&f.q80_split).arg(&rv).arg(&nb).arg(&mut qv).arg(&mut dv).launch(cfg) }.map_err(gerr)?;
                }
                sl
            };
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
            if idx >= n_res {
                // The new KV rows to the pinned host rows (copied to the processor's cache at the end).
                let j = idx - n_res;
                let n = t * kvd;
                let rec = l.delta.as_ref().map(|gd| (gd.hist.device_ptr(s).0, gd.hist.len(), gd.state.device_ptr(s).0, gd.state.len()));
                let (kp, vp) = (l.kc.device_ptr(s).0 + (pos0 * kvd * 2) as u64, l.vc.device_ptr(s).0 + (pos0 * kvd * 2) as u64);
                let p = sin.as_mut().expect("streamed layers").pipe.as_mut().expect("pipe");
                if let Some((hp, hl, sp, sl_)) = rec {
                    // The DeltaNet layer's history and state after the chunk.
                    let back = p.rec_back.as_mut().expect("made above").as_mut_slice().map_err(gerr)?;
                    let o = j * (hl + sl_);
                    // SAFETY: async copies into page-locked memory read only after the stream synchronizes.
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_async(&mut back[o..o + hl], hp, s.cu_stream()).map_err(gerr)?;
                        cudarc::driver::result::memcpy_dtoh_async(&mut back[o + hl..o + hl + sl_], sp, s.cu_stream()).map_err(gerr)?;
                    }
                } else {
                    let back = p.kv_back.as_mut().expect("made above").as_mut_slice().map_err(gerr)?;
                    // SAFETY: async copies into page-locked memory read only after the stream synchronizes.
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_async(&mut back[2 * j * n..(2 * j + 1) * n], kp, s.cu_stream()).map_err(gerr)?;
                        cudarc::driver::result::memcpy_dtoh_async(&mut back[(2 * j + 1) * n..(2 * j + 2) * n], vp, s.cu_stream()).map_err(gerr)?;
                    }
                }
                p.freed[j % 2] = Some(s.record_event(None).map_err(gerr)?);
                // Push this layer's kernels to the GPU before the host fills the next upload.
                unsafe { cudarc::driver::sys::cuStreamQuery(s.cu_stream()) };
                if j + 2 < n_streamed {
                    issue(&mut sin, j + 2)?;
                }
            } else if idx == 0 && n_streamed > 0 {
                // The first two processor layers upload while the GPU works through its own.
                for j in 0..n_streamed.min(2) {
                    issue(&mut sin, j)?;
                }
            }
        }
        if n_streamed > 0 {
            s.synchronize().map_err(gerr)?;
            let si = sin.as_mut().expect("streamed layers");
            let p = si.pipe.as_ref().expect("pipe");
            let back = p.kv_back.as_ref().expect("made above").as_slice().map_err(gerr)?;
            let rec = p.rec_back.as_ref().map(|r| r.as_slice()).transpose().map_err(gerr)?;
            let n = t * kvd;
            for j in 0..n_streamed {
                let li = si.from + j;
                if model.layers()[li].delta.is_some() {
                    let (hl, sl_) = (si.hist[li].len(), si.state[li].len());
                    let r = rec.expect("DeltaNet layers have a state buffer");
                    let o = j * (hl + sl_);
                    si.hist[li].copy_from_slice(&r[o..o + hl]);
                    si.state[li].copy_from_slice(&r[o + hl..o + hl + sl_]);
                    continue;
                }
                let (a, z) = (pos0 * kvd, (pos0 + t) * kvd);
                si.k[li][a..z].copy_from_slice(&back[2 * j * n..(2 * j + 1) * n]);
                si.v[li][a..z].copy_from_slice(&back[(2 * j + 1) * n..(2 * j + 2) * n]);
            }
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
        if profiling {
            let p = prof.get();
            let total = chunk_t0.elapsed().as_secs_f64();
            eprintln!("gpu prefill chunk of {t}: {:.1} ms; native-to-int8 {:.1} ms, GEMM {:.1} ms, the rest {:.1} ms", total * 1e3, p[0] * 1e3, p[1] * 1e3, (total - p[0] - p[1]) * 1e3);
        }
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

    /// k_to_q8 decodes every format kept native on the GPU (i-quants, Q2_K) like the CPU's reference
    /// decoder: random blocks with sane scales, int8 output dequantised and compared value by value.
    #[test]
    fn native_formats_decode_like_the_reference() {
        let Ok(ctx) = CudaContext::new(0) else { return eprintln!("no CUDA device; skipped") };
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX)).unwrap();
        let f = module.load_function("k_to_q8").unwrap();
        let (rows, cols) = (4usize, 512usize);
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut byte = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        };
        // (format, bytes per 256 on the GPU, bytes per file block, file block values, f16 scale offsets)
        let cases: [(DType, &[usize]); 9] = [
            (DType::Q2K, &[80, 82]),
            (DType::IQ2XXS, &[0]),
            (DType::IQ3XXS, &[0]),
            (DType::IQ1S, &[0]),
            (DType::IQ4NL, &[0]),
            (DType::IQ3S, &[0]),
            (DType::IQ2S, &[0]),
            (DType::IQ4XS, &[0]),
            (DType::IQ1M, &[]),
        ];
        for (dt, scales) in cases {
            let (bb, code) = gpu_block(dt).unwrap();
            let (fbb, fbe) = (dt.block_bytes(), dt.block_elems());
            let mut w: Vec<u8> = (0..rows * cols / fbe * fbb).map(|_| byte()).collect();
            for blk in w.chunks_exact_mut(fbb) {
                for &o in scales {
                    blk[o..o + 2].copy_from_slice(&0x2400u16.to_le_bytes());
                }
                if dt == DType::IQ1M {
                    for (o, nib) in [(48usize, 0u8), (50, 0), (52, 4), (54, 2)] {
                        blk[o + 1] = (blk[o + 1] & 0x0f) | (nib << 4);
                    }
                }
            }
            let mut want = vec![0f32; cols];
            let mut got_all = Vec::new();
            let dw = stream.clone_htod(&w).unwrap();
            let mut qs = stream.alloc_zeros::<i8>(rows * cols).unwrap();
            let mut ds = stream.alloc_zeros::<f32>(rows * cols / 32).unwrap();
            let (code_i, bb_i, r_i, c_i) = (code, bb as i32, rows as i32, cols as i32);
            let cfg = LaunchConfig { grid_dim: (((rows * cols / 32) as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { stream.launch_builder(&f).arg(&dw).arg(&code_i).arg(&bb_i).arg(&r_i).arg(&c_i).arg(&mut qs).arg(&mut ds).launch(cfg) }.unwrap();
            let q = stream.clone_dtoh(&qs).unwrap();
            let d = stream.clone_dtoh(&ds).unwrap();
            let row_bytes = cols / fbe * fbb;
            for r in 0..rows {
                crate::quant::dequant_row(dt, &w[r * row_bytes..(r + 1) * row_bytes], &mut want);
                let amax = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
                let got: Vec<f32> = (0..cols).map(|i| q[r * cols + i] as f32 * d[(r * cols + i) / 32]).collect();
                let bad = want.iter().zip(&got).position(|(a, b)| (a - b).abs() > amax / 60.0);
                assert!(bad.is_none(), "{dt:?} row {r}: value {} is {} on the GPU, {} on the CPU", bad.unwrap(), got[bad.unwrap()], want[bad.unwrap()]);
                got_all.push(got);
            }
            let _ = got_all;
            // The decode kernel (kmatmul, one token) against the reference: dequantised rows times
            // the same int8 activations.
            let km = module.load_function("kmatmul").unwrap();
            let xs: Vec<i8> = (0..cols).map(|_| byte() as i8).collect();
            let xd: Vec<f32> = (0..cols / 32).map(|i| 0.01 + 0.001 * i as f32).collect();
            let (dxq, dxd) = (stream.clone_htod(&xs).unwrap(), stream.clone_htod(&xd).unwrap());
            let mut y = stream.alloc_zeros::<f32>(rows).unwrap();
            let null_i = stream.alloc_zeros::<i32>(1).unwrap();
            let (one, ys, add) = (1i32, rows as i32, 0i32);
            let null_ptr: Option<&CudaSlice<i32>> = None;
            let _ = null_ptr;
            let cfg = LaunchConfig { grid_dim: ((rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { stream.launch_builder(&km).arg(&dw).arg(&code_i).arg(&bb_i).arg(&dxq).arg(&dxd).arg(&0u64).arg(&one).arg(&mut y).arg(&ys).arg(&r_i).arg(&c_i).arg(&add).launch(cfg) }.unwrap();
            let _ = null_i;
            let yv = stream.clone_dtoh(&y).unwrap();
            for r in 0..rows {
                crate::quant::dequant_row(dt, &w[r * row_bytes..(r + 1) * row_bytes], &mut want);
                let exp: f32 = (0..cols).map(|i| want[i] * xs[i] as f32 * xd[i / 32]).sum();
                let mag: f32 = (0..cols).map(|i| (want[i] * xs[i] as f32 * xd[i / 32]).abs()).sum::<f32>().max(1e-6);
                assert!((yv[r] - exp).abs() <= 1e-3 * mag, "{dt:?} row {r}: decode dot {} on the GPU, {} reference", yv[r], exp);
            }
        }
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

/// GPU assist: the processor runs the forward pass and the graphics card multiplies the weight
/// matrices read every token. For any architecture, including ones without a full GPU path yet:
/// whatever the CPU computes stays on the CPU (attention, recurrent layers, routing, experts from
/// the RAM cache) and every matrix-vector product whose matrix is in VRAM runs here instead. The
/// matrices are uploaded once, K-quants in their own layout and everything else as int8; one that
/// does not fit stays on the processor. The activations are the processor's own int8 quantisation,
/// so the GPU computes the same products the CPU kernels would.
pub struct Offload {
    stream: Arc<CudaStream>,
    matvec: CudaFunction,
    matvec_k: CudaFunction,
    mats: std::collections::HashMap<usize, GMat>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    y: CudaSlice<f32>,
    /// VRAM the matrices take, and how many of the candidates are there.
    pub bytes: usize,
    pub placed: usize,
    pub candidates: usize,
    pub calls: u64,
}

/// The smallest matrix GPU assist takes (rows x cols weights).
const OFFLOAD_MIN_WEIGHTS: usize = 8 << 20;

impl Offload {
    pub fn new(model: &Model) -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gerr)?;
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(gerr)?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let matvec = module.load_function("matvec_q8").map_err(gerr)?;
        let matvec_k = module.load_function("matvec_k").map_err(gerr)?;
        let (free, _) = ctx.mem_get_info().map_err(gerr)?;
        let mut budget = (free as usize).saturating_sub(crate::placement::VRAM_MARGIN as usize);
        let g = model.gguf();
        let mut mats = std::collections::HashMap::new();
        let (mut bytes, mut maxc, mut maxr) = (0usize, 32usize, 1usize);
        let cands = model.dense_mats();
        let candidates = cands.len();
        for m in cands {
            // A product pays a fixed cost for the round trip (copy in, launch, copy out, wait);
            // below about 8M weights the processor finishes first (Qwen3.6-35B-A3B: 4.6 tok/s on
            // the CPU alone, 2.4 with every small matrix sent to the GPU).
            if mats.contains_key(&m.offset) || m.rows == 0 || m.cols % 32 != 0 || m.rows * m.cols < OFFLOAD_MIN_WEIGHTS {
                continue;
            }
            let gm = match gpu_block(m.dtype) {
                Some((bb, dtype)) if m.cols % 256 == 0 => {
                    let need = m.rows * m.row_bytes / 210 * 224 + m.rows * m.row_bytes;
                    if need > budget {
                        continue;
                    }
                    let w = to_gpu_layout(m.dtype, g.bytes_at(m.offset, m.rows * m.row_bytes));
                    budget -= w.len();
                    bytes += w.len();
                    GMat::K { parts: vec![KPart { w: stream.clone_htod(&w).map_err(gerr)?, dtype, bb: bb as i32, rows: m.rows }], cols: m.cols }
                }
                _ => {
                    let need = m.rows * m.cols + m.rows * m.cols / 32 * 4;
                    if need > budget {
                        continue;
                    }
                    let (qs, ds, rows, cols) = repack(g, &[m]);
                    budget -= need;
                    bytes += need;
                    GMat::Q8 { qs: stream.clone_htod(&qs).map_err(gerr)?, ds: stream.clone_htod(&ds).map_err(gerr)?, rows, cols }
                }
            };
            maxc = maxc.max(m.cols);
            maxr = maxr.max(m.rows);
            mats.insert(m.offset, gm);
        }
        let placed = mats.len();
        Ok(Self {
            xq: stream.alloc_zeros::<i8>(maxc).map_err(gerr)?,
            xd: stream.alloc_zeros::<f32>(maxc / 32).map_err(gerr)?,
            y: stream.alloc_zeros::<f32>(maxr).map_err(gerr)?,
            stream,
            matvec,
            matvec_k,
            mats,
            bytes,
            placed,
            candidates,
            calls: 0,
        })
    }

    pub fn has(&self, m: &Mat) -> bool {
        self.mats.contains_key(&m.offset)
    }

    /// out = M x on the GPU, x given as the processor's int8 activations.
    pub fn run(&mut self, m: &Mat, qa: &QAct, out: &mut [f32]) -> Result<(), Error> {
        let Some(gm) = self.mats.get(&m.offset) else { return Err(Error::Unsupported("matrix not on the GPU".into())) };
        let (rows, cols) = (m.rows, m.cols);
        let s = &self.stream;
        s.memcpy_htod(&qa.q[..cols], &mut self.xq.slice_mut(0..cols)).map_err(gerr)?;
        s.memcpy_htod(&qa.d[..cols / 32], &mut self.xd.slice_mut(0..cols / 32)).map_err(gerr)?;
        let cfg = |r: usize| LaunchConfig { grid_dim: ((r as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (c, add) = (cols as i32, 0i32);
        match gm {
            GMat::Q8 { qs, ds, .. } => {
                let r = rows as i32;
                unsafe { s.launch_builder(&self.matvec).arg(qs).arg(ds).arg(&self.xq).arg(&self.xd).arg(&mut self.y).arg(&r).arg(&c).arg(&add).launch(cfg(rows)) }.map_err(gerr)?;
            }
            GMat::K { parts, .. } => {
                let mut r0 = 0;
                for p in parts {
                    let mut yv = self.y.slice_mut(r0..r0 + p.rows);
                    let r = p.rows as i32;
                    unsafe { s.launch_builder(&self.matvec_k).arg(&p.w).arg(&p.dtype).arg(&p.bb).arg(&self.xq).arg(&self.xd).arg(&mut yv).arg(&r).arg(&c).arg(&add).launch(cfg(p.rows)) }.map_err(gerr)?;
                    r0 += p.rows;
                }
            }
        }
        s.memcpy_dtoh(&self.y.slice(0..rows), &mut out[..rows]).map_err(gerr)?;
        s.synchronize().map_err(gerr)?;
        self.calls += 1;
        Ok(())
    }
}

/// Prompt batches on the GPU for weights that live in RAM: the processor's batched matmuls (a split
/// model's processor layers, GPU assist, a processor session on a computer with a GPU) send each
/// matrix to the GPU in its file format, convert it to int8 there and run the tensor-core GEMM over
/// the whole chunk of tokens. Copying a layer over PCIe takes milliseconds; multiplying it by a
/// few hundred tokens on the processor takes far longer. Formats without a GPU converter (the
/// i-quants) stay on the processor.
pub struct BatchGpu {
    stream: Arc<CudaStream>,
    k_to_q8: CudaFunction,
    q80_split: CudaFunction,
    gemm: CudaFunction,
    w: CudaSlice<u8>,
    qs: CudaSlice<i8>,
    ds: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    y: CudaSlice<f32>,
    pub calls: u64,
}

/// The fewest tokens for which a batch goes to the GPU.
pub const BATCH_GPU_MIN_TOKENS: usize = 32;

impl BatchGpu {
    pub fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gerr)?;
        unsafe { ctx.disable_event_tracking() };
        if ctx.compute_capability().map_err(gerr)?.0 < 8 {
            return Err(Error::Unsupported("prompt batches on the GPU need tensor cores (sm_80 or newer)".into()));
        }
        let stream = ctx.new_stream().map_err(gerr)?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let mma = ctx.load_module(Ptx::from_src(PTX_MMA)).map_err(gerr)?;
        let z8 = |n| stream.alloc_zeros::<i8>(n).map_err(gerr);
        let zf = |n| stream.alloc_zeros::<f32>(n).map_err(gerr);
        Ok(Self {
            k_to_q8: module.load_function("k_to_q8").map_err(gerr)?,
            q80_split: module.load_function("q80_split").map_err(gerr)?,
            gemm: mma.load_function("matmul_q8_mma").map_err(gerr)?,
            w: stream.alloc_zeros::<u8>(1).map_err(gerr)?,
            qs: z8(1)?,
            ds: zf(1)?,
            xq: z8(1)?,
            xd: zf(1)?,
            y: zf(1)?,
            stream,
            calls: 0,
        })
    }

    /// Whether this matrix's format converts on the GPU.
    pub fn takes(m: &Mat) -> bool {
        m.cols % 32 == 0 && (m.dtype == DType::Q8_0 || (gpu_block(m.dtype).is_some() && m.cols % 256 == 0))
    }

    /// out[t * rows + r] = M x_t for the T tokens of `qas` (the processor's int8 activations);
    /// `bytes` is the matrix as stored in the file.
    pub fn run(&mut self, m: &Mat, bytes: &[u8], qas: &[QAct], out: &mut [f32]) -> Result<(), Error> {
        let log = std::env::var_os("CEREBRO_BATCH_GPU_LOG").is_some();
        let t0 = std::time::Instant::now();
        let (rows, cols, t) = (m.rows, m.cols, qas.len());
        let s = self.stream.clone();
        let grow8 = |b: &mut CudaSlice<i8>, n: usize| -> Result<(), Error> {
            if b.len() < n {
                *b = s.alloc_zeros::<i8>(n).map_err(gerr)?;
            }
            Ok(())
        };
        let growf = |b: &mut CudaSlice<f32>, n: usize| -> Result<(), Error> {
            if b.len() < n {
                *b = s.alloc_zeros::<f32>(n).map_err(gerr)?;
            }
            Ok(())
        };
        grow8(&mut self.qs, rows * cols)?;
        growf(&mut self.ds, rows * cols / 32)?;
        grow8(&mut self.xq, t * cols)?;
        growf(&mut self.xd, t * cols / 32)?;
        growf(&mut self.y, t * rows)?;
        // The weights, converted on the GPU.
        let wb: std::borrow::Cow<'_, [u8]> = if m.dtype == DType::Q8_0 { std::borrow::Cow::Borrowed(bytes) } else { std::borrow::Cow::Owned(to_gpu_layout(m.dtype, bytes)) };
        if self.w.len() < wb.len() {
            self.w = s.alloc_zeros::<u8>(wb.len()).map_err(gerr)?;
        }
        s.memcpy_htod(&wb[..], &mut self.w.slice_mut(0..wb.len())).map_err(gerr)?;
        if m.dtype == DType::Q8_0 {
            let nb = (rows * cols / 32) as i32;
            let cfg = LaunchConfig { grid_dim: ((nb as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&self.q80_split).arg(&self.w).arg(&nb).arg(&mut self.qs).arg(&mut self.ds).launch(cfg) }.map_err(gerr)?;
        } else {
            let (bb, dtype) = gpu_block(m.dtype).expect("takes() checked the format");
            let (bb, r, c) = (bb as i32, rows as i32, cols as i32);
            let subs = (rows * cols / 32) as u32;
            let cfg = LaunchConfig { grid_dim: (subs.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&self.k_to_q8).arg(&self.w).arg(&dtype).arg(&bb).arg(&r).arg(&c).arg(&mut self.qs).arg(&mut self.ds).launch(cfg) }.map_err(gerr)?;
        }
        // The activations, token-major.
        let mut hq = Vec::with_capacity(t * cols);
        let mut hd = Vec::with_capacity(t * cols / 32);
        for qa in qas {
            hq.extend_from_slice(&qa.q[..cols]);
            hd.extend_from_slice(&qa.d[..cols / 32]);
        }
        s.memcpy_htod(&hq, &mut self.xq.slice_mut(0..t * cols)).map_err(gerr)?;
        s.memcpy_htod(&hd, &mut self.xd.slice_mut(0..t * cols / 32)).map_err(gerr)?;
        let (ri, ci, ti, add) = (rows as i32, cols as i32, t as i32, 0i32);
        let cfg = LaunchConfig { grid_dim: ((t as u32).div_ceil(64), (rows as u32).div_ceil(64), 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&self.gemm).arg(&self.qs).arg(&self.ds).arg(&self.xq).arg(&self.xd).arg(&mut self.y).arg(&ri).arg(&ci).arg(&ti).arg(&add).launch(cfg) }.map_err(gerr)?;
        s.memcpy_dtoh(&self.y.slice(0..t * rows), &mut out[..t * rows]).map_err(gerr)?;
        s.synchronize().map_err(gerr)?;
        self.calls += 1;
        if log {
            eprintln!("batch gpu: {rows}x{cols} {:?} x {t} tokens: {:.2} ms", m.dtype, t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }
}
