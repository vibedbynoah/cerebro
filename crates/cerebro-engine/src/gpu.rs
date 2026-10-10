//! CUDA backend. Kernels are Cerebro's own (kernels.cu), compiled to PTX at build time and embedded,
//! so users need only the NVIDIA driver. Weights are repacked on upload into an aligned layout (int8
//! values in one array, one f32 scale per 32 values in another); Q/K/V and gate/up are concatenated
//! into one matrix each. The token id and position live in device memory, so a whole decode step is
//! captured once as a CUDA graph and replayed per token: no per-kernel launch cost, no logits copy for
//! greedy decoding (the next token is picked on the GPU).

use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::gguf::Gguf;
use crate::model::{Mat, Model};
use crate::quant::{dequant_row, f16_to_f32, DType};
use crate::Error;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

struct GMat {
    qs: CudaSlice<i8>,
    ds: CudaSlice<f32>,
    rows: usize,
    cols: usize,
}

struct GLayer {
    attn_norm: CudaSlice<f32>,
    qkv: GMat,
    o: GMat,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    ffn_norm: CudaSlice<f32>,
    gu: GMat,
    down: GMat,
    /// f16 KV cache, max_ctx positions.
    kc: CudaSlice<u16>,
    vc: CudaSlice<u16>,
}

struct Funcs {
    matvec: CudaFunction,
    rmsnorm_quant: CudaFunction,
    quant: CudaFunction,
    qkv_post: CudaFunction,
    attn_partial: CudaFunction,
    attn_combine: CudaFunction,
    silu_mul_quant: CudaFunction,
    embed: CudaFunction,
    argmax: CudaFunction,
    embed_b: CudaFunction,
    norm_b: CudaFunction,
    matmul_b: CudaFunction,
    silu_b: CudaFunction,
    attn_pre: CudaFunction,
}

/// Prefill scratch: up to `CHUNK` tokens per batched pass.
const CHUNK: usize = 256;
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
    pos0: CudaSlice<i32>,
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
    qk_norm: bool,
}

/// The model's weights, KV cache and scratch on one GPU, plus the captured decode graph.
pub struct GpuState {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f: Funcs,
    sh: Shape,
    layers: Vec<GLayer>,
    out_norm: CudaSlice<f32>,
    output: GMat,
    embed: Option<GMat>,
    x: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    att: CudaSlice<f32>,
    gu: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    tok: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    next: CudaSlice<i32>,
    pm: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    pacc: CudaSlice<f32>,
    splits: usize,
    b: Batch,
    graph: Option<CudaGraph>,
    pub max_ctx: usize,
    pub vram_bytes: usize,
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
        let ctx = CudaContext::new(0).map_err(gerr)?;
        // One stream, so the library's cross-stream event tracking is unnecessary (and it would
        // otherwise record events inside graph capture).
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(gerr)?;
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let lf = |n: &str| module.load_function(n).map_err(gerr);
        let f = Funcs {
            matvec: lf("matvec_q8")?,
            rmsnorm_quant: lf("rmsnorm_quant")?,
            quant: lf("quantize_q8")?,
            qkv_post: lf("qkv_post")?,
            attn_partial: lf("attn_partial")?,
            attn_combine: lf("attn_combine")?,
            silu_mul_quant: lf("silu_mul_quant")?,
            embed: lf("embed")?,
            argmax: lf("argmax_next")?,
            embed_b: lf("embed_batch")?,
            norm_b: lf("rmsnorm_quant_batch")?,
            matmul_b: lf("matmul_q8_batch")?,
            silu_b: lf("silu_mul_quant_batch")?,
            attn_pre: lf("attn_prefill")?,
        };
        let c = &model.cfg;
        if c.head_dim > 256 || c.head_dim % 64 != 0 || c.dim > 8192 {
            return Err(Error::Unsupported(format!("head size {} / width {} on the GPU", c.head_dim, c.dim)));
        }
        let g = model.gguf();
        let mut vram = 0usize;
        let mut up = |mats: &[&Mat]| -> Result<GMat, Error> {
            let (qs, ds, rows, cols) = repack(g, mats);
            vram += qs.len() + ds.len() * 4;
            Ok(GMat { qs: stream.clone_htod(&qs).map_err(gerr)?, ds: stream.clone_htod(&ds).map_err(gerr)?, rows, cols })
        };
        let kvd = c.n_kv * c.head_dim;
        let mut layers = Vec::with_capacity(c.n_layer);
        for l in model.layers() {
            layers.push(GLayer {
                attn_norm: stream.clone_htod(&l.attn_norm).map_err(gerr)?,
                qkv: up(&[&l.q, &l.k, &l.v])?,
                o: up(&[&l.o])?,
                q_norm: l.q_norm.as_deref().map(|v| stream.clone_htod(v)).transpose().map_err(gerr)?,
                k_norm: l.k_norm.as_deref().map(|v| stream.clone_htod(v)).transpose().map_err(gerr)?,
                ffn_norm: stream.clone_htod(&l.ffn_norm).map_err(gerr)?,
                gu: up(&[&l.gate, &l.up])?,
                down: up(&[&l.down])?,
                kc: stream.alloc_zeros::<u16>(max_ctx * kvd).map_err(gerr)?,
                vc: stream.alloc_zeros::<u16>(max_ctx * kvd).map_err(gerr)?,
            });
        }
        let output = up(&[model.output_mat()])?;
        // Tied embeddings reuse the output matrix on the GPU; otherwise upload the table too.
        let embed = if model.tied() { None } else { Some(up(&[model.embed_mat()])?) };
        vram += c.n_layer * 2 * max_ctx * kvd * 2;
        let z = |n: usize| stream.alloc_zeros::<f32>(n).map_err(gerr);
        let qd = c.n_head * c.head_dim;
        let widest = c.dim.max(c.ffn).max(qd);
        let splits = max_ctx.div_ceil(SPLIT);
        let row = qd + 2 * kvd;
        let b = Batch {
            x: z(CHUNK * c.dim)?,
            qkv: z(CHUNK * row)?,
            att: z(CHUNK * qd)?,
            gu: z(CHUNK * 2 * c.ffn)?,
            xq: stream.alloc_zeros::<i8>(CHUNK * widest).map_err(gerr)?,
            xd: z(CHUNK * widest / 32)?,
            toks: stream.alloc_zeros::<i32>(CHUNK).map_err(gerr)?,
            pos0: stream.alloc_zeros::<i32>(1).map_err(gerr)?,
        };
        let sh = Shape { dim: c.dim, ffn: c.ffn, n_head: c.n_head, n_kv: c.n_kv, hd: c.head_dim, vocab: c.vocab, eps: c.eps, base: c.rope_base, qk_norm: c.qk_norm };
        Ok(Self {
            out_norm: stream.clone_htod(model.out_norm()).map_err(gerr)?,
            output,
            embed,
            x: z(c.dim)?,
            qkv: z(qd + 2 * kvd)?,
            att: z(qd)?,
            gu: z(2 * c.ffn)?,
            xq: stream.alloc_zeros::<i8>(widest).map_err(gerr)?,
            xd: z(widest / 32)?,
            logits: z(c.vocab)?,
            tok: stream.alloc_zeros::<i32>(1).map_err(gerr)?,
            pos: stream.alloc_zeros::<i32>(1).map_err(gerr)?,
            next: stream.alloc_zeros::<i32>(1).map_err(gerr)?,
            pm: z(c.n_head * splits)?,
            pl: z(c.n_head * splits)?,
            pacc: z(c.n_head * splits * c.head_dim)?,
            splits,
            b,
            graph: None,
            layers,
            f,
            sh,
            stream,
            _ctx: ctx,
            max_ctx,
            vram_bytes: vram,
        })
    }

    fn matvec(s: &CudaStream, f: &CudaFunction, m: &GMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, add: bool) -> Result<(), Error> {
        let (rows, cols, add) = (m.rows as i32, m.cols as i32, add as i32);
        let cfg = LaunchConfig { grid_dim: ((m.rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(f).arg(&m.qs).arg(&m.ds).arg(xq).arg(xd).arg(y).arg(&rows).arg(&cols).arg(&add).launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    /// Enqueues one full decode step reading the token and position from device memory.
    fn enqueue(&mut self) -> Result<(), Error> {
        let sh = &self.sh;
        let s = &self.stream;
        let f = &self.f;
        let blocks = |n: usize| LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (dim, ffn, hd) = (sh.dim as i32, sh.ffn as i32, sh.hd as i32);
        let (n_head, n_kv) = (sh.n_head as i32, sh.n_kv as i32);
        let (eps, base) = (sh.eps, sh.base);
        let null_f: u64 = 0;
        {
            let e = self.embed.as_ref().unwrap_or(&self.output);
            unsafe { s.launch_builder(&f.embed).arg(&e.qs).arg(&e.ds).arg(&self.tok).arg(&mut self.x).arg(&dim).launch(blocks(sh.dim)) }.map_err(gerr)?;
        }
        let has_norm = sh.qk_norm as i32;
        let group = (sh.n_head / sh.n_kv) as i32;
        let scale = 1.0f32 / (sh.hd as f32).sqrt();
        for l in &mut self.layers {
            unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&l.attn_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
            Self::matvec(s, &f.matvec, &l.qkv, &self.xq, &self.xd, &mut self.qkv, false)?;
            {
                let qn = l.q_norm.as_ref().unwrap_or(&l.attn_norm);
                let kn = l.k_norm.as_ref().unwrap_or(&l.attn_norm);
                let cfg = LaunchConfig { grid_dim: ((sh.n_head + sh.n_kv) as u32, 1, 1), block_dim: ((sh.hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    s.launch_builder(&f.qkv_post)
                        .arg(&mut self.qkv)
                        .arg(qn)
                        .arg(kn)
                        .arg(&has_norm)
                        .arg(&mut l.kc)
                        .arg(&mut l.vc)
                        .arg(&n_head)
                        .arg(&n_kv)
                        .arg(&hd)
                        .arg(&self.pos)
                        .arg(&base)
                        .arg(&eps)
                        .launch(cfg)
                }
                .map_err(gerr)?;
            }
            {
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
            unsafe { s.launch_builder(&f.quant).arg(&self.att).arg(&mut self.xq).arg(&mut self.xd).arg(&qd).launch(blocks(sh.n_head * sh.hd)) }.map_err(gerr)?;
            Self::matvec(s, &f.matvec, &l.o, &self.xq, &self.xd, &mut self.x, true)?;
            unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&l.ffn_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
            Self::matvec(s, &f.matvec, &l.gu, &self.xq, &self.xd, &mut self.gu, false)?;
            unsafe { s.launch_builder(&f.silu_mul_quant).arg(&self.gu).arg(&mut self.xq).arg(&mut self.xd).arg(&ffn).launch(blocks(sh.ffn)) }.map_err(gerr)?;
            Self::matvec(s, &f.matvec, &l.down, &self.xq, &self.xd, &mut self.x, true)?;
        }
        unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&self.x).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        Self::matvec(s, &f.matvec, &self.output, &self.xq, &self.xd, &mut self.logits, false)?;
        let vocab = sh.vocab as i32;
        unsafe { s.launch_builder(&f.argmax).arg(&self.logits).arg(&vocab).arg(&mut self.next).arg(&mut self.tok).arg(&mut self.pos).launch(one) }.map_err(gerr)?;
        Ok(())
    }

    /// Captures the decode step as a graph the first time; replays it afterwards.
    fn run(&mut self) -> Result<(), Error> {
        if self.graph.is_none() {
            self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).map_err(gerr)?;
            let r = self.enqueue();
            let g = self.stream.end_capture(CUgraphInstantiate_flags(0)).map_err(gerr)?;
            r?;
            self.graph = g;
        }
        match &self.graph {
            Some(g) => g.launch().map_err(gerr),
            None => self.enqueue(),
        }
    }

    /// Runs `token` at `pos`. Copies the logits into `out` when given (sampling); returns the greedy
    /// next token either way.
    pub fn step(&mut self, token: u32, pos: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        if pos >= self.max_ctx {
            return Err(Error::Request("context full".into()));
        }
        self.stream.memcpy_htod(&[token as i32], &mut self.tok).map_err(gerr)?;
        self.stream.memcpy_htod(&[pos as i32], &mut self.pos).map_err(gerr)?;
        self.run()?;
        if let Some(o) = out {
            self.stream.memcpy_dtoh(&self.logits, o).map_err(gerr)?;
        }
        let mut next = [0i32];
        self.stream.memcpy_dtoh(&self.next, &mut next).map_err(gerr)?;
        Ok(next[0] as u32)
    }

    /// Batched forward over `toks` at positions pos0.. (prefill, or verifying a draft): every weight is
    /// read once per chunk of up to CHUNK tokens. Writes the KV cache for all of them and the logits of
    /// the last one into `out` when given; returns the greedy next token.
    pub fn prefill(&mut self, toks: &[u32], pos0: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        if toks.is_empty() || pos0 + toks.len() > self.max_ctx {
            return Err(Error::Request("prefill outside the context".into()));
        }
        let mut last = 0u32;
        let mut out = out;
        for (ci, chunk) in toks.chunks(CHUNK).enumerate() {
            let p0 = pos0 + ci * CHUNK;
            let is_last = (ci + 1) * CHUNK >= toks.len();
            last = self.prefill_chunk(chunk, p0, if is_last { out.as_deref_mut() } else { None })?;
        }
        Ok(last)
    }

    fn prefill_chunk(&mut self, toks: &[u32], pos0: usize, out: Option<&mut [f32]>) -> Result<u32, Error> {
        let t = toks.len();
        let sh = &self.sh;
        let s = &self.stream;
        let f = &self.f;
        let ids: Vec<i32> = toks.iter().map(|&v| v as i32).collect();
        s.memcpy_htod(&ids, &mut self.b.toks).map_err(gerr)?;
        s.memcpy_htod(&[pos0 as i32], &mut self.b.pos0).map_err(gerr)?;
        let (dim, ffn, hd) = (sh.dim as i32, sh.ffn as i32, sh.hd as i32);
        let (n_head, n_kv) = (sh.n_head as i32, sh.n_kv as i32);
        let (eps, base) = (sh.eps, sh.base);
        let ti = t as i32;
        let has_norm = sh.qk_norm as i32;
        let group = (sh.n_head / sh.n_kv) as i32;
        let scale = 1.0f32 / (sh.hd as f32).sqrt();
        let rows_cfg = |rows: usize| LaunchConfig { grid_dim: ((rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let b = &mut self.b;
        {
            let e = self.embed.as_ref().unwrap_or(&self.output);
            let cfg = LaunchConfig { grid_dim: ((sh.dim as u32).div_ceil(256), t as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { s.launch_builder(&f.embed_b).arg(&e.qs).arg(&e.ds).arg(&b.toks).arg(&mut b.x).arg(&dim).launch(cfg) }.map_err(gerr)?;
        }
        let norm_cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let mm = |m: &GMat, xq: &CudaSlice<i8>, xd: &CudaSlice<f32>, y: &mut CudaSlice<f32>, add: bool| -> Result<(), Error> {
            let (rows, cols, add) = (m.rows as i32, m.cols as i32, add as i32);
            unsafe { s.launch_builder(&f.matmul_b).arg(&m.qs).arg(&m.ds).arg(xq).arg(xd).arg(y).arg(&rows).arg(&cols).arg(&ti).arg(&add).launch(rows_cfg(m.rows)) }.map_err(gerr)?;
            Ok(())
        };
        for l in &mut self.layers {
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.attn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).launch(norm_cfg) }.map_err(gerr)?;
            mm(&l.qkv, &b.xq, &b.xd, &mut b.qkv, false)?;
            {
                let qn = l.q_norm.as_ref().unwrap_or(&l.attn_norm);
                let kn = l.k_norm.as_ref().unwrap_or(&l.attn_norm);
                let cfg = LaunchConfig { grid_dim: ((sh.n_head + sh.n_kv) as u32, t as u32, 1), block_dim: ((sh.hd / 2) as u32, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    s.launch_builder(&f.qkv_post)
                        .arg(&mut b.qkv)
                        .arg(qn)
                        .arg(kn)
                        .arg(&has_norm)
                        .arg(&mut l.kc)
                        .arg(&mut l.vc)
                        .arg(&n_head)
                        .arg(&n_kv)
                        .arg(&hd)
                        .arg(&b.pos0)
                        .arg(&base)
                        .arg(&eps)
                        .launch(cfg)
                }
                .map_err(gerr)?;
            }
            {
                let cfg = LaunchConfig { grid_dim: (sh.n_head as u32, t as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&f.attn_pre).arg(&b.qkv).arg(&l.kc).arg(&l.vc).arg(&mut b.att).arg(&n_head).arg(&n_kv).arg(&hd).arg(&b.pos0).arg(&group).arg(&scale).launch(cfg) }.map_err(gerr)?;
            }
            let qn_total = (t * sh.n_head * sh.hd) as i32;
            unsafe { s.launch_builder(&f.quant).arg(&b.att).arg(&mut b.xq).arg(&mut b.xd).arg(&qn_total).launch(LaunchConfig { grid_dim: ((qn_total as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            mm(&l.o, &b.xq, &b.xd, &mut b.x, true)?;
            unsafe { s.launch_builder(&f.norm_b).arg(&b.x).arg(&l.ffn_norm).arg(&mut b.xq).arg(&mut b.xd).arg(&dim).arg(&eps).launch(norm_cfg) }.map_err(gerr)?;
            mm(&l.gu, &b.xq, &b.xd, &mut b.gu, false)?;
            let n = (t * sh.ffn) as u32;
            unsafe { s.launch_builder(&f.silu_b).arg(&b.gu).arg(&mut b.xq).arg(&mut b.xd).arg(&ffn).arg(&ti).launch(LaunchConfig { grid_dim: (n.div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            mm(&l.down, &b.xq, &b.xd, &mut b.x, true)?;
        }
        // Logits for the last token only, through the decode buffers.
        {
            let last = b.x.slice((t - 1) * sh.dim..t * sh.dim);
            let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            let null_f: u64 = 0;
            unsafe { s.launch_builder(&f.rmsnorm_quant).arg(&last).arg(&self.out_norm).arg(&mut self.xq).arg(&mut self.xd).arg(&null_f).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        }
        Self::matvec(s, &f.matvec, &self.output, &self.xq, &self.xd, &mut self.logits, false)?;
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
    pub fn generate_greedy(&mut self, first: u32, pos: usize, max_new: usize, stop: &[u32], mut emit: impl FnMut(u32)) -> Result<usize, Error> {
        self.stream.memcpy_htod(&[first as i32], &mut self.tok).map_err(gerr)?;
        self.stream.memcpy_htod(&[pos as i32], &mut self.pos).map_err(gerr)?;
        let mut n = 0;
        let mut next = [0i32];
        while n < max_new && pos + n < self.max_ctx {
            self.run()?;
            self.stream.memcpy_dtoh(&self.next, &mut next).map_err(gerr)?;
            n += 1;
            let t = next[0] as u32;
            if stop.contains(&t) {
                break;
            }
            emit(t);
        }
        Ok(n)
    }
}
