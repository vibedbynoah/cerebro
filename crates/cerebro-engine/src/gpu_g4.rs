//! Gemma 4 on an NVIDIA card (gemma4.rs has the architecture): every layer, its KV cache and the
//! tied output head in VRAM; each token's embedding and per-layer table rows are looked up on the
//! processor (the per-layer table is 2.3 GB and only one row of it is read per token) and uploaded.
//! K-quant matrices stay in their own format (matvec_k); the BF16 embedding projection is int8.

use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::gpu::{gerr, gpu_block, repack, to_gpu_layout, PTX, PTX_MMA};
use crate::model::{Ffn, Mat, Model};
use crate::Error;

/// A weight matrix on the GPU: the file's own K-quant blocks, or int8 rows with a scale per 32.
enum W {
    K { w: CudaSlice<u8>, dtype: i32, bb: i32, rows: usize, cols: usize },
    Q8 { qs: CudaSlice<i8>, ds: CudaSlice<f32>, rows: usize, cols: usize },
}

impl W {
    fn rows(&self) -> usize {
        match self {
            W::K { rows, .. } | W::Q8 { rows, .. } => *rows,
        }
    }
}

struct L {
    swa: bool,
    hd: usize,
    kv_from: usize,
    own_kv: bool,
    attn_norm: CudaSlice<f32>,
    q: W,
    k: Option<W>,
    v: Option<W>,
    o: W,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    post_attn: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    gate: W,
    up: W,
    down: W,
    post_ffn: CudaSlice<f32>,
    inp_gate: W,
    proj: W,
    post_norm: CudaSlice<f32>,
    out_scale: f32,
    /// f16 K and V, max_ctx positions of n_kv * hd (empty for a layer that shares another's).
    kc: CudaSlice<u16>,
    vc: CudaSlice<u16>,
}

struct F {
    matvec: CudaFunction,
    matvec_k: CudaFunction,
    matvec_k3: CudaFunction,
    rmsnorm_quant: CudaFunction,
    quant: CudaFunction,
    qkv: CudaFunction,
    attn: CudaFunction,
    combine: CudaFunction,
    gelu: CudaFunction,
    rms_add_q: CudaFunction,
    ple: CudaFunction,
    softcap: CudaFunction,
    // Prompt batches.
    k_to_q8: CudaFunction,
    norm_b: CudaFunction,
    qkv_b: CudaFunction,
    attn_b: CudaFunction,
    attn_t: CudaFunction,
    rms_add_b: CudaFunction,
    ple_b: CudaFunction,
    gelu_ple_b: CudaFunction,
    /// Tensor-core GEMMs (sm_80+): 64x64, 128x64 and 128x128 blocks, and Q4_K straight from
    /// its blocks.
    mma: Option<[CudaFunction; 7]>,
}

/// Prompt-batch buffers, CH tokens.
struct B {
    x: CudaSlice<f32>,
    tok: CudaSlice<f32>,
    pp: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    att: CudaSlice<f32>,
    t: CudaSlice<f32>,
    g: CudaSlice<f32>,
    u: CudaSlice<f32>,
    pg: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    /// int8 copy of one K-quant matrix for the tensor cores.
    k8q: CudaSlice<i8>,
    k8d: CudaSlice<f32>,
}

/// Tokens per prompt batch.
pub const CH: usize = 512;

pub struct G4Gpu {
    _ctx: Arc<CudaContext>,
    s: Arc<CudaStream>,
    f: F,
    layers: Vec<L>,
    ple_proj: W,
    ple_norm: CudaSlice<f32>,
    out_norm: CudaSlice<f32>,
    output: W,
    /// Rotary frequencies of the sliding and the global layers (factors applied).
    freq_swa: CudaSlice<f32>,
    freq_full: CudaSlice<f32>,
    // Per-token buffers.
    x: CudaSlice<f32>,
    tok: CudaSlice<f32>,
    pp: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kb: CudaSlice<f32>,
    vb: CudaSlice<f32>,
    att: CudaSlice<f32>,
    t: CudaSlice<f32>,
    g: CudaSlice<f32>,
    u: CudaSlice<f32>,
    pg: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    pm: CudaSlice<f32>,
    plm: CudaSlice<f32>,
    pacc: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    pos: CudaSlice<i32>,
    splits: usize,
    max_ctx: usize,
    pub vram_bytes: usize,
    /// The token's kernels, recorded once and replayed (about 840 launches a token otherwise).
    graph: Option<CudaGraph>,
    /// Page-locked landing place for each step's logits: a copy into pageable memory is
    /// synchronous and staged under Windows' driver model (262K logits, 1 MB, every token).
    host_logits: PinnedF32,
    /// The same for a step's inputs: embedding, per-layer rows, then the position's bits.
    host_in: PinnedF32,
    b: Option<B>,
}

/// Decode attention: a fixed grid of SPLITS splits per head (so a captured graph replays at any
/// length), each taking an equal share of the positions in view, at least CHUNK. With one split
/// per 256 positions, a short conversation ran as 8 blocks (one per head) on a 46-SM card.
const SPLITS: usize = 32;
const CHUNK: usize = 16;

impl G4Gpu {
    pub fn load(model: &Model, max_ctx: usize) -> Result<Self, Error> {
        let g4 = model.g4.as_ref().ok_or_else(|| Error::Unsupported("not a Gemma 4 model".into()))?;
        let c = &model.cfg;
        let ctx = CudaContext::new(0).map_err(gerr)?;
        unsafe { ctx.disable_event_tracking() };
        let s = ctx.new_stream().map_err(gerr)?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let lf = |n: &str| module.load_function(n).map_err(gerr);
        let f = F {
            matvec: lf("matvec_q8")?,
            matvec_k: lf("matvec_k")?,
            matvec_k3: lf("matvec_k3")?,
            rmsnorm_quant: lf("rmsnorm_quant")?,
            quant: lf("quantize_q8")?,
            qkv: lf("g4_qkv")?,
            attn: lf("g4_attn_partial")?,
            combine: lf("g4_attn_combine")?,
            gelu: lf("g4_gelu_mul_quant")?,
            rms_add_q: lf("g4_rms_add_quant")?,
            ple: lf("g4_ple_prep")?,
            softcap: lf("g4_softcap")?,
            k_to_q8: lf("k_to_q8")?,
            norm_b: lf("rmsnorm_quant_batch")?,
            qkv_b: lf("g4_qkv_b")?,
            attn_b: lf("g4_attn_b")?,
            attn_t: lf("g4_attn_t")?,
            rms_add_b: lf("g4_rms_add_b")?,
            ple_b: lf("g4_ple_prep_b")?,
            gelu_ple_b: lf("g4_gelu_ple_b")?,
            mma: if ctx.compute_capability().map_err(gerr)?.0 >= 8 {
                let m = ctx.load_module(Ptx::from_src(PTX_MMA)).map_err(gerr)?;
                Some([m.load_function("matmul_q8_mma").map_err(gerr)?, m.load_function("matmul_q8_mma_big").map_err(gerr)?, m.load_function("matmul_q8_mma_x").map_err(gerr)?, m.load_function("mmq_q4k").map_err(gerr)?, m.load_function("mmq_q4k_x").map_err(gerr)?, m.load_function("attn_prefill_mma256w").map_err(gerr)?, m.load_function("attn_prefill_mma512w").map_err(gerr)?])
            } else {
                None
            },
        };
        let g = model.gguf();
        let mut vram = 0usize;
        let mut up = |m: &Mat| -> Result<W, Error> {
            if let (Some((bb, dtype)), true) = (gpu_block(m.dtype), m.cols % 256 == 0) {
                let bytes = to_gpu_layout(m.dtype, g.bytes_at(m.offset, m.rows * m.row_bytes));
                vram += bytes.len();
                return Ok(W::K { w: s.clone_htod(&bytes).map_err(gerr)?, dtype, bb: bb as i32, rows: m.rows, cols: m.cols });
            }
            let (qs, ds, rows, cols) = repack(g, &[m]);
            vram += qs.len() + ds.len() * 4;
            Ok(W::Q8 { qs: s.clone_htod(&qs).map_err(gerr)?, ds: s.clone_htod(&ds).map_err(gerr)?, rows, cols })
        };
        let v = |x: &[f32]| s.clone_htod(x).map_err(gerr);
        let mut kv_bytes = 0usize;
        let mut layers = Vec::with_capacity(c.n_layer);
        for (li, (l, gl)) in model.layers().iter().zip(&g4.layers).enumerate() {
            crate::check_stop()?;
            let Ffn::Dense { gate, up: upm, down } = &l.ffn else { return Err(Error::Unsupported("gemma4 expert layers".into())) };
            let own = li < g4.kv_layers;
            let kd = g4.n_kv * gl.hd;
            layers.push(L {
                swa: gl.swa,
                hd: gl.hd,
                kv_from: gl.kv_from,
                own_kv: own,
                attn_norm: v(&l.attn_norm)?,
                q: up(&l.q)?,
                k: if own { Some(up(&l.k)?) } else { None },
                v: if own { Some(up(&l.v)?) } else { None },
                o: up(&l.o)?,
                q_norm: v(l.q_norm.as_deref().unwrap_or(&[1.0]))?,
                k_norm: v(l.k_norm.as_deref().unwrap_or(&[1.0]))?,
                post_attn: v(&gl.post_attn_norm)?,
                ffn_norm: v(&l.ffn_norm)?,
                gate: up(gate)?,
                up: up(upm)?,
                down: up(down)?,
                post_ffn: v(&gl.post_ffn_norm)?,
                inp_gate: up(&gl.inp_gate)?,
                proj: up(&gl.proj)?,
                post_norm: v(&gl.post_norm)?,
                out_scale: gl.out_scale,
                kc: s.alloc_zeros::<u16>(if own { max_ctx * kd } else { 1 }).map_err(gerr)?,
                vc: s.alloc_zeros::<u16>(if own { max_ctx * kd } else { 1 }).map_err(gerr)?,
            });
            if own {
                kv_bytes += 2 * max_ctx * kd * 2;
            }
        }
        let freq = |hd: usize, base: f32, factors: Option<&[f32]>| -> Vec<f32> {
            (0..hd / 2).map(|i| base.powf(-2.0 * i as f32 / hd as f32) / factors.map_or(1.0, |f| f[i])).collect()
        };
        let hd_swa = g4.layers.iter().find(|l| l.swa).map_or(256, |l| l.hd);
        let hd_full = g4.layers.iter().find(|l| !l.swa).map_or(512, |l| l.hd);
        let hd_max = hd_swa.max(hd_full);
        let ff = model.layers().iter().map(|l| if let Ffn::Dense { gate, .. } = &l.ffn { gate.rows } else { 0 }).max().unwrap_or(c.ffn);
        let widest = c.dim.max(g4.n_head * hd_max).max(ff);
        // Half the splits for contexts up to 8K: each split then does twice the keys, and the 256
        // blocks of 32 splits were mostly overhead at short contexts (E4B, 656-token context: 79.1
        // -> 81.4 tok/s writing). Longer contexts keep 32 to spread the keys over the GPU.
        let cap = if max_ctx <= 8192 { SPLITS / 2 } else { SPLITS };
        let splits = cap.min(max_ctx.div_ceil(CHUNK)).max(1);
        let z = |n: usize| s.alloc_zeros::<f32>(n.max(1)).map_err(gerr);
        let output = up(model.output_mat())?;
        let ple_proj = up(&g4.ple_proj)?;
        drop(up);
        // The largest K-quant matrix, for the int8 copy prompt batches multiply.
        let k8 = layers.iter().flat_map(|l| [&l.q, &l.o, &l.gate, &l.up, &l.down, &l.inp_gate, &l.proj].into_iter().chain(l.k.iter()).chain(l.v.iter())).filter_map(|w| match w {
            W::K { rows, cols, .. } => Some(rows * cols),
            W::Q8 { .. } => None,
        }).max().unwrap_or(32);
        let nlp = c.n_layer * g4.ple;
        let b = if f.mma.is_some() {
            Some(B {
                x: z(CH * c.dim)?,
                tok: z(CH * nlp)?,
                pp: z(CH * nlp)?,
                pl: z(CH * nlp)?,
                q: z(CH * g4.n_head * hd_max)?,
                k: z(CH * g4.n_kv * hd_max)?,
                v: z(CH * g4.n_kv * hd_max)?,
                att: z(CH * g4.n_head * hd_max)?,
                t: z(CH * c.dim)?,
                g: z(CH * ff)?,
                u: z(CH * ff)?,
                pg: z(CH * g4.ple)?,
                xq: s.alloc_zeros::<i8>(CH * widest).map_err(gerr)?,
                xd: z(CH * widest / 32)?,
                k8q: s.alloc_zeros::<i8>(k8).map_err(gerr)?,
                k8d: z(k8 / 32)?,
            })
        } else {
            None
        };
        let vram = vram + kv_bytes;
        Ok(Self {
            freq_swa: v(&freq(hd_swa, g4.base_swa, None))?,
            freq_full: v(&freq(hd_full, g4.base, Some(&g4.rope_freqs[..hd_full / 2])))?,
            ple_norm: v(&g4.ple_proj_norm)?,
            out_norm: v(model.out_norm())?,
            x: z(c.dim)?,
            tok: z(c.n_layer * g4.ple)?,
            pp: z(c.n_layer * g4.ple)?,
            pl: z(c.n_layer * g4.ple)?,
            q: z(g4.n_head * hd_max)?,
            kb: z(g4.n_kv * hd_max)?,
            vb: z(g4.n_kv * hd_max)?,
            att: z(g4.n_head * hd_max)?,
            t: z(c.dim)?,
            g: z(ff)?,
            u: z(ff)?,
            pg: z(g4.ple)?,
            xq: s.alloc_zeros::<i8>(widest).map_err(gerr)?,
            xd: z(widest / 32)?,
            pm: z(g4.n_head * splits)?,
            plm: z(g4.n_head * splits)?,
            pacc: z(g4.n_head * splits * hd_max)?,
            logits: z(c.vocab)?,
            pos: s.alloc_zeros::<i32>(1).map_err(gerr)?,
            splits,
            max_ctx,
            vram_bytes: vram,
            graph: None,
            host_logits: PinnedF32::new(c.vocab)?,
            host_in: PinnedF32::new(c.dim + c.n_layer * g4.ple + 1)?,
            b,
            output,
            ple_proj,
            layers,
            f,
            s,
            _ctx: ctx,
        })
    }

    /// Whether prompts can be read in batches here (tensor cores).
    pub fn batches(&self) -> bool {
        self.b.is_some()
    }

    /// y[t * rows + r] = W x_t for `t` tokens, on the tensor cores: a K-quant matrix is first
    /// converted to int8 rows (k_to_q8) in `k8`.
    fn gemm(s: &CudaStream, f: &F, k8: (&mut CudaSlice<i8>, &mut CudaSlice<f32>), m: &W, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, t: usize) -> Result<(), Error> {
        let mma = f.mma.as_ref().expect("batches need tensor cores");
        let (rows, cols) = match m {
            W::K { rows, cols, .. } | W::Q8 { rows, cols, .. } => (*rows, *cols),
        };
        let (k8q, k8d) = k8;
        // Q4_K straight from its blocks (no int8 copy) only with CEREBRO_MMQ set: converting to int8
        // for the int8 tensor-core GEMM read prompts faster (E4B Q4_K_M, 512 tokens: 2692 -> 2821).
        if let (W::K { w, dtype: 12, .. }, Some(_)) = (m, std::env::var_os("CEREBRO_MMQ")) {
            let (ri, ci, ti, add) = (rows as i32, cols as i32, t as i32, 0i32);
            // 128 tokens a block once a prompt is long enough to fill it.
            let (k, bn, th) = if t >= 128 { (&mma[4], 128u32, 256u32) } else { (&mma[3], 64, 128) };
            let cfg = LaunchConfig { grid_dim: ((t as u32).div_ceil(bn), (rows as u32).div_ceil(128), 1), block_dim: (th, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(k).arg(w).arg(xq).arg(xd).arg(y).arg(&ri).arg(&ci).arg(&ti).arg(&add).arg(&ri).launch(cfg) }.map_err(gerr)?;
            return Ok(());
        }
        if let W::K { w, dtype, bb, .. } = m {
            let (r, c) = (rows as i32, cols as i32);
            let subs = (rows * cols / 32) as u32;
            let cfg = LaunchConfig { grid_dim: (subs.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.k_to_q8).arg(w).arg(dtype).arg(bb).arg(&r).arg(&c).arg(&mut *k8q).arg(&mut *k8d).launch(cfg) }.map_err(gerr)?;
        }
        let (qs, ds): (&CudaSlice<i8>, &CudaSlice<f32>) = match m {
            W::K { .. } => (&*k8q, &*k8d),
            W::Q8 { qs, ds, .. } => (qs, ds),
        };
        let (k, bm, bn, th) = if rows >= 1024 && t >= 256 {
            (&mma[2], 128u32, 128u32, 256u32)
        } else if rows >= 1024 && t >= 128 {
            (&mma[1], 128, 64, 128)
        } else {
            (&mma[0], 64, 64, 128)
        };
        let cfg = LaunchConfig { grid_dim: ((t as u32).div_ceil(bn), (rows as u32).div_ceil(bm), 1), block_dim: (th, 1, 1), shared_mem_bytes: 0 };
        let (ri, ci, ti, add) = (rows as i32, cols as i32, t as i32, 0i32);
        unsafe { s.launch_builder(k).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&ri).arg(&ci).arg(&ti).arg(&add).arg(&ri).launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    /// A prompt chunk of `n` tokens (at most CH) at positions pos0..: `xs` their scaled embeddings,
    /// `toks` their per-layer rows, token-major. Fills their KV cache and leaves the last token's
    /// soft-capped logits in `logits`.
    pub fn prefill(&mut self, model: &Model, xs: &[f32], toks: &[f32], n: usize, pos0: usize, logits: &mut [f32]) -> Result<(), Error> {
        if pos0 + n > self.max_ctx || n > CH || n == 0 {
            return Err(Error::Request("prompt chunk outside the context".into()));
        }
        let g4 = model.g4.as_ref().expect("gemma4");
        let c = &model.cfg;
        let s = self.s.clone();
        let f = &self.f;
        let b = self.b.as_mut().ok_or_else(|| Error::Unsupported("prompt batches need tensor cores".into()))?;
        let (dim, eps, ple, nl) = (c.dim as i32, c.eps, g4.ple as i32, c.n_layer as i32);
        let (nh, nkv, ti) = (g4.n_head as i32, g4.n_kv as i32, n as i32);
        let null_f: u64 = 0;
        let blocks = |m: usize| LaunchConfig { grid_dim: ((m as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let rows_cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        s.memcpy_htod(&xs[..n * c.dim], &mut b.x.slice_mut(0..n * c.dim)).map_err(gerr)?;
        s.memcpy_htod(&toks[..n * c.n_layer * g4.ple], &mut b.tok.slice_mut(0..n * c.n_layer * g4.ple)).map_err(gerr)?;
        s.memcpy_htod(&[pos0 as i32], &mut self.pos).map_err(gerr)?;
        // Per-layer inputs.
        let nd = (n * c.dim) as i32;
        unsafe { s.launch_builder(&f.quant).arg(&b.x).arg(&mut b.xq).arg(&mut b.xd).arg(&nd).launch(blocks(n * c.dim)) }.map_err(gerr)?;
        Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &self.ple_proj, &b.xq, &b.xd, &mut b.pp, n)?;
        {
            let (ps, ts) = (1.0 / (c.dim as f32).sqrt(), (g4.ple as f32).sqrt());
            let cfg = LaunchConfig { grid_dim: (c.n_layer as u32, n as u32, 1), block_dim: (g4.ple as u32, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.ple_b).arg(&b.pp).arg(&b.tok).arg(&self.ple_norm).arg(&mut b.pl).arg(&ple).arg(&nl).arg(&ps).arg(&ts).arg(&eps).launch(cfg) }.map_err(gerr)?;
        }
        let unit = 1.0f32;
        for li in 0..self.layers.len() {
            let (hd, own, swa, kv_from) = {
                let l = &self.layers[li];
                (l.hd, l.own_kv, l.swa, l.kv_from)
            };
            let hdi = hd as i32;
            {
                let l = &self.layers[li];
                unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.attn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&null_f).launch(rows_cfg) }.map_err(gerr)?;
                Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.q, &b.xq, &b.xd, &mut b.q, n)?;
                if own {
                    Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), l.k.as_ref().expect("k"), &b.xq, &b.xd, &mut b.k, n)?;
                    Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), l.v.as_ref().expect("v"), &b.xq, &b.xd, &mut b.v, n)?;
                }
            }
            {
                let freq = if swa { &self.freq_swa } else { &self.freq_full };
                let grid = if own { g4.n_head + 2 * g4.n_kv } else { g4.n_head } as u32;
                let cfg = LaunchConfig { grid_dim: (grid, n as u32, 1), block_dim: ((hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                let l = &mut self.layers[li];
                unsafe { s.launch_builder(&f.qkv_b).arg(&mut b.q).arg(&b.k).arg(&b.v).arg(&l.q_norm).arg(&l.k_norm).arg(&mut l.kc).arg(&mut l.vc).arg(freq).arg(&self.pos).arg(&nh).arg(&nkv).arg(&hdi).arg(&eps).launch(cfg) }.map_err(gerr)?;
            }
            {
                let src = &self.layers[kv_from];
                let window = if swa { g4.swa as i32 } else { 0 };
                let cfg = LaunchConfig { grid_dim: (g4.n_head as u32, n as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                let group = (g4.n_head / g4.n_kv) as i32;
                if let (Some(mma), 256 | 512, None) = (f.mma.as_ref(), hd, std::env::var_os("CEREBRO_G4_NO_MMA_ATTN")) {
                    // Tensor-core flash attention (256- or 512-wide heads, sliding or global, scale
                    // 1), each block 128 output dimensions.
                    let cfg = LaunchConfig { grid_dim: (g4.n_head as u32, (n as u32).div_ceil(64), (hd / 128) as u32), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                    let (qrow, one) = ((g4.n_head * hd) as i32, 1.0f32);
                    let k = if hd == 256 { &mma[5] } else { &mma[6] };
                    unsafe { s.launch_builder(k).arg(&b.q).arg(&src.kc).arg(&src.vc).arg(&mut b.att).arg(&nh).arg(&nkv).arg(&self.pos).arg(&group).arg(&one).arg(&ti).arg(&qrow).arg(&window).launch(cfg) }.map_err(gerr)?;
                } else if (g4.n_head / g4.n_kv) * 8 <= 32 && std::env::var_os("CEREBRO_G4_UNTILED").is_none() {
                    // Tiled: queries of one KV head share its K/V tiles (needs group * 8 <= 32 queries).
                    let cfg = LaunchConfig { grid_dim: (g4.n_kv as u32, (n as u32).div_ceil(8), 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    unsafe { s.launch_builder(&f.attn_t).arg(&b.q).arg(&src.kc).arg(&src.vc).arg(&mut b.att).arg(&nh).arg(&nkv).arg(&hdi).arg(&self.pos).arg(&window).arg(&ti).launch(cfg) }.map_err(gerr)?;
                } else {
                    unsafe { s.launch_builder(&f.attn_b).arg(&b.q).arg(&src.kc).arg(&src.vc).arg(&mut b.att).arg(&nh).arg(&nkv).arg(&hdi).arg(&self.pos).arg(&window).launch(cfg) }.map_err(gerr)?;
                }
            }
            let l = &self.layers[li];
            let qn = (n * g4.n_head * hd) as i32;
            unsafe { s.launch_builder(&f.quant).arg(&b.att).arg(&mut b.xq).arg(&mut b.xd).arg(&qn).launch(blocks(qn as usize)) }.map_err(gerr)?;
            Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.o, &b.xq, &b.xd, &mut b.t, n)?;
            unsafe { s.launch_builder(&f.rms_add_b).arg(&mut b.x).arg(&b.t).arg(&l.post_attn).arg(&eps).arg(&dim).arg(&unit).launch(rows_cfg) }.map_err(gerr)?;
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.ffn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).arg(&null_f).launch(rows_cfg) }.map_err(gerr)?;
            Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.gate, &b.xq, &b.xd, &mut b.g, n)?;
            Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.up, &b.xq, &b.xd, &mut b.u, n)?;
            let fn_ = (n * l.gate.rows()) as i32;
            unsafe { s.launch_builder(&f.gelu).arg(&b.g).arg(&b.u).arg(&mut b.xq).arg(&mut b.xd).arg(&fn_).launch(blocks(fn_ as usize)) }.map_err(gerr)?;
            Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.down, &b.xq, &b.xd, &mut b.t, n)?;
            unsafe { s.launch_builder(&f.rms_add_b).arg(&mut b.x).arg(&b.t).arg(&l.post_ffn).arg(&eps).arg(&dim).arg(&unit).launch(rows_cfg) }.map_err(gerr)?;
            unsafe { s.launch_builder(&f.quant).arg(&b.x).arg(&mut b.xq).arg(&mut b.xd).arg(&nd).launch(blocks(n * c.dim)) }.map_err(gerr)?;
            Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.inp_gate, &b.xq, &b.xd, &mut b.pg, n)?;
            let lii = li as i32;
            unsafe { s.launch_builder(&f.gelu_ple_b).arg(&b.pg).arg(&b.pl).arg(&mut b.xq).arg(&mut b.xd).arg(&ple).arg(&nl).arg(&lii).arg(&ti).launch(blocks(n * g4.ple)) }.map_err(gerr)?;
            Self::gemm(&s, f, (&mut b.k8q, &mut b.k8d), &l.proj, &b.xq, &b.xd, &mut b.t, n)?;
            let sc = l.out_scale;
            unsafe { s.launch_builder(&f.rms_add_b).arg(&mut b.x).arg(&b.t).arg(&l.post_norm).arg(&eps).arg(&dim).arg(&sc).launch(rows_cfg) }.map_err(gerr)?;
        }
        // The last token's head.
        s.memcpy_dtod(&b.x.slice((n - 1) * c.dim..n * c.dim), &mut self.x).map_err(gerr)?;
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        Self::mv(&s, f, &self.output, &self.xq, &self.xd, &mut self.logits)?;
        if g4.softcap > 0.0 {
            let v = c.vocab as i32;
            unsafe { s.launch_builder(&f.softcap).arg(&mut self.logits).arg(&v).arg(&g4.softcap).launch(blocks(c.vocab)) }.map_err(gerr)?;
        }
        s.memcpy_dtoh(&self.logits, &mut logits[..c.vocab]).map_err(gerr)?;
        s.synchronize().map_err(gerr)?;
        Ok(())
    }

    fn mv(s: &CudaStream, f: &F, m: &W, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>) -> Result<(), Error> {
        let add = 0i32;
        let cfg = LaunchConfig { grid_dim: ((m.rows() as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        match m {
            W::K { w, dtype, bb, rows, cols } => {
                let (r, c) = (*rows as i32, *cols as i32);
                unsafe { s.launch_builder(&f.matvec_k).arg(w).arg(dtype).arg(bb).arg(xq).arg(xd).arg(y).arg(&r).arg(&c).arg(&add).launch(cfg) }.map_err(gerr)?;
            }
            W::Q8 { qs, ds, rows, cols } => {
                let (r, c) = (*rows as i32, *cols as i32);
                unsafe { s.launch_builder(&f.matvec).arg(qs).arg(ds).arg(xq).arg(xd).arg(y).arg(&r).arg(&c).arg(&add).launch(cfg) }.map_err(gerr)?;
            }
        }
        Ok(())
    }

    /// Two or three matvecs on the same activations, in one launch when all are K-quant (each `y`
    /// gets its matrix's rows), else one by one.
    fn mv_many(s: &CudaStream, f: &F, ms: &mut [(&W, &mut CudaSlice<f32>)], xq: &CudaSlice<i8>, xd: &CudaSlice<f32>) -> Result<(), Error> {
        if !(2..=3).contains(&ms.len()) || !ms.iter().all(|(m, _)| matches!(m, W::K { .. })) {
            for (m, y) in ms.iter_mut() {
                Self::mv(s, f, m, xq, xd, y)?;
            }
            return Ok(());
        }
        let first: &W = ms[0].0;
        let W::K { w: w_any, dtype: dt_any, bb: bb_any, cols, .. } = first else { unreachable!("checked") };
        let cols = *cols as i32;
        let rows: Vec<i32> = ms.iter().map(|(m, _)| m.rows() as i32).collect();
        let blocks: u32 = rows.iter().map(|&r| (r as u32).div_ceil(8)).sum();
        let (zero, n) = (0i32, ms.len());
        let mut lb = s.launch_builder(&f.matvec_k3);
        for (i, (m, y)) in ms.iter_mut().enumerate() {
            let W::K { w, dtype, bb, .. } = *m else { unreachable!("checked") };
            lb.arg(w).arg(dtype).arg(bb).arg(&mut **y).arg(&rows[i]);
        }
        if n == 2 {
            // No third matrix: a segment of 0 rows, which no block reaches (xd stands in for y).
            lb.arg(w_any).arg(dt_any).arg(bb_any).arg(xd).arg(&zero);
        }
        lb.arg(xq).arg(xd).arg(&cols);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    /// One token at `pos`: `x` its scaled embedding, `tok` its per-layer table rows (scaled on the
    /// GPU). Leaves the soft-capped logits in `logits`.
    pub fn step(&mut self, model: &Model, x: &[f32], tok: &[f32], pos: usize, logits: &mut [f32]) -> Result<(), Error> {
        if pos >= self.max_ctx {
            return Err(Error::Request("context full".into()));
        }
        let s = self.s.clone();
        {
            use cudarc::driver::DevicePtrMut;
            let (nx, nt) = (x.len(), tok.len());
            let h = self.host_in.as_mut_slice();
            h[..nx].copy_from_slice(x);
            h[nx..nx + nt].copy_from_slice(tok);
            h[nx + nt] = f32::from_bits(pos as i32 as u32);
            let cs = s.cu_stream();
            // SAFETY: page-locked source ranges sized above; the step synchronises the stream
            // before the buffer is written again.
            unsafe {
                let (d, _g) = self.x.device_ptr_mut(&s);
                cudarc::driver::result::memcpy_htod_async(d, &h[..nx], cs).map_err(gerr)?;
                let (d, _g) = self.tok.device_ptr_mut(&s);
                cudarc::driver::result::memcpy_htod_async(d, &h[nx..nx + nt], cs).map_err(gerr)?;
                let (d, _g) = self.pos.device_ptr_mut(&s);
                cudarc::driver::result::memcpy_htod_async(d, &h[nx + nt..nx + nt + 1], cs).map_err(gerr)?;
            }
        }
        if self.graph.is_none() && std::env::var_os("CEREBRO_NO_GRAPH").is_none() {
            s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).map_err(gerr)?;
            let r = self.enqueue(model);
            let g = s.end_capture(CUgraphInstantiate_flags(0)).map_err(gerr)?;
            r?;
            self.graph = g;
        }
        match &self.graph {
            Some(g) => g.launch().map_err(gerr)?,
            None => self.enqueue(model)?,
        }
        let vocab = model.cfg.vocab;
        {
            use cudarc::driver::DevicePtr;
            let (src, _guard) = self.logits.device_ptr(&s);
            // SAFETY: host_logits holds vocab floats; the stream is synchronised before they are read.
            unsafe { cudarc::driver::result::memcpy_dtoh_async(self.host_logits.as_mut_slice(), src, s.cu_stream()) }.map_err(gerr)?;
        }
        s.synchronize().map_err(gerr)?;
        logits[..vocab].copy_from_slice(&self.host_logits.as_mut_slice()[..vocab]);
        Ok(())
    }

    /// The token's kernels, reading the embedding, per-layer rows and position from device memory.
    fn enqueue(&mut self, model: &Model) -> Result<(), Error> {
        let g4 = model.g4.as_ref().expect("gemma4");
        let c = &model.cfg;
        let (s, f) = (&self.s, &self.f);
        let (dim, eps, ple) = (c.dim as i32, c.eps, g4.ple as i32);
        let (nh, nkv) = (g4.n_head as i32, g4.n_kv as i32);
        let null_f: u64 = 0;
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let blocks = |n: usize| LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        // Per-layer inputs.
        unsafe { s.launch_builder(&f.quant).arg(&self.x).arg(&mut self.xq).arg(&mut self.xd).arg(&dim).launch(blocks(c.dim)) }.map_err(gerr)?;
        Self::mv(s, f, &self.ple_proj, &self.xq, &self.xd, &mut self.pp)?;
        {
            let (ps, ts) = (1.0 / (c.dim as f32).sqrt(), (g4.ple as f32).sqrt());
            let cfg = LaunchConfig { grid_dim: (c.n_layer as u32, 1, 1), block_dim: (g4.ple as u32, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.ple).arg(&self.pp).arg(&self.tok).arg(&self.ple_norm).arg(&mut self.pl).arg(&ple).arg(&ps).arg(&ts).arg(&eps).launch(cfg) }.map_err(gerr)?;
        }
        let chunk = CHUNK as i32;
        let splits = self.splits;
        for li in 0..self.layers.len() {
            let (hd, own, swa, kv_from) = { let l = &self.layers[li]; (l.hd, l.own_kv, l.swa, l.kv_from) };
            let hdi = hd as i32;
            {
                let l = &self.layers[li];
                // Layers after the first got this input from the previous layer's last kernel.
                if li == 0 {
                    unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&l.attn_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
                }
                if own {
                    Self::mv_many(s, f, &mut [(&l.q, &mut self.q), (l.k.as_ref().expect("k"), &mut self.kb), (l.v.as_ref().expect("v"), &mut self.vb)], &self.xq, &self.xd)?;
                } else {
                    Self::mv(s, f, &l.q, &self.xq, &self.xd, &mut self.q)?;
                }
            }
            {
                let freq = if swa { &self.freq_swa } else { &self.freq_full };
                let grid = if own { g4.n_head + 2 * g4.n_kv } else { g4.n_head } as u32;
                let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: ((hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                let l = &mut self.layers[li];
                unsafe { s.launch_builder(&f.qkv).arg(&mut self.q).arg(&self.kb).arg(&self.vb).arg(&l.q_norm).arg(&l.k_norm).arg(&mut l.kc).arg(&mut l.vc).arg(freq).arg(&self.pos).arg(&nh).arg(&nkv).arg(&hdi).arg(&eps).launch(cfg) }.map_err(gerr)?;
            }
            {
                let src = &self.layers[kv_from];
                let window = if swa { g4.swa as i32 } else { 0 };
                let cfg = LaunchConfig { grid_dim: (g4.n_head as u32, splits as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&f.attn).arg(&self.q).arg(&src.kc).arg(&src.vc).arg(&mut self.pm).arg(&mut self.plm).arg(&mut self.pacc).arg(&nh).arg(&nkv).arg(&hdi).arg(&self.pos).arg(&window).arg(&chunk).launch(cfg) }.map_err(gerr)?;
                let sp = splits as i32;
                let cfg = LaunchConfig { grid_dim: (g4.n_head as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&f.combine).arg(&self.pm).arg(&self.plm).arg(&self.pacc).arg(&mut self.att).arg(&hdi).arg(&sp).launch(cfg) }.map_err(gerr)?;
            }
            let l = &self.layers[li];
            let qd = (g4.n_head * hd) as i32;
            unsafe { s.launch_builder(&f.quant).arg(&self.att).arg(&mut self.xq).arg(&mut self.xd).arg(&qd).launch(blocks(qd as usize)) }.map_err(gerr)?;
            Self::mv(s, f, &l.o, &self.xq, &self.xd, &mut self.t)?;
            let unit = 1.0f32;
            // The attention block's residual, then the feed-forward's normed input.
            unsafe { s.launch_builder(&f.rms_add_q).arg(&mut self.x).arg(&self.t).arg(&l.post_attn).arg(&eps).arg(&dim).arg(&unit).arg(&l.ffn_norm).arg(&mut self.xq).arg(&mut self.xd).launch(one) }.map_err(gerr)?;
            Self::mv_many(s, f, &mut [(&l.gate, &mut self.g), (&l.up, &mut self.u)], &self.xq, &self.xd)?;
            let ff = l.gate.rows() as i32;
            unsafe { s.launch_builder(&f.gelu).arg(&self.g).arg(&self.u).arg(&mut self.xq).arg(&mut self.xd).arg(&ff).launch(blocks(ff as usize)) }.map_err(gerr)?;
            Self::mv(s, f, &l.down, &self.xq, &self.xd, &mut self.t)?;
            // The feed-forward's residual, then x itself quantised for the per-layer input gate.
            unsafe { s.launch_builder(&f.rms_add_q).arg(&mut self.x).arg(&self.t).arg(&l.post_ffn).arg(&eps).arg(&dim).arg(&unit).arg(&null_f).arg(&mut self.xq).arg(&mut self.xd).launch(one) }.map_err(gerr)?;
            Self::mv(s, f, &l.inp_gate, &self.xq, &self.xd, &mut self.pg)?;
            {
                let pl = self.pl.slice(li * g4.ple..(li + 1) * g4.ple);
                unsafe { s.launch_builder(&f.gelu).arg(&self.pg).arg(&pl).arg(&mut self.xq).arg(&mut self.xd).arg(&ple).launch(blocks(g4.ple)) }.map_err(gerr)?;
            }
            Self::mv(s, f, &l.proj, &self.xq, &self.xd, &mut self.t)?;
            let sc = l.out_scale;
            // The layer's residual, then the next layer's normed attention input (the output
            // norm's after the last layer).
            let next = self.layers.get(li + 1).map_or(&self.out_norm, |n| &n.attn_norm);
            let l = &self.layers[li];
            unsafe { s.launch_builder(&f.rms_add_q).arg(&mut self.x).arg(&self.t).arg(&l.post_norm).arg(&eps).arg(&dim).arg(&sc).arg(next).arg(&mut self.xq).arg(&mut self.xd).launch(one) }.map_err(gerr)?;
        }
        Self::mv(s, f, &self.output, &self.xq, &self.xd, &mut self.logits)?;
        if g4.softcap > 0.0 {
            let n = c.vocab as i32;
            unsafe { s.launch_builder(&f.softcap).arg(&mut self.logits).arg(&n).arg(&g4.softcap).launch(blocks(c.vocab)) }.map_err(gerr)?;
        }
        Ok(())
    }
}

/// Page-locked host floats (cuMemHostAlloc), freed on drop. Plain pointer and length, so copies
/// into it carry none of the event bookkeeping of cudarc's pinned slices.
struct PinnedF32 {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: owned memory, used by one session at a time.
unsafe impl Send for PinnedF32 {}

impl PinnedF32 {
    fn new(len: usize) -> Result<Self, Error> {
        // SAFETY: a fresh allocation of len floats.
        let ptr = unsafe { cudarc::driver::result::malloc_host(len * 4, 0) }.map_err(gerr)? as *mut f32;
        Ok(Self { ptr, len })
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: ptr holds len floats for the life of self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedF32 {
    fn drop(&mut self) {
        // SAFETY: allocated by malloc_host and freed once.
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void) };
    }
}
