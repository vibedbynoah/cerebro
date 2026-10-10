//! CUDA backend. Kernels are Cerebro's own, written below in CUDA C and compiled at load time for the
//! detected GPU architecture. Weights are repacked on upload into an aligned layout (int8 values in
//! one array, one f32 scale per 32 values in another) so each thread loads 16 bytes at once and uses
//! the integer dot-product instruction; activations are quantised to int8 on the GPU per step.
//! The whole forward pass stays on the GPU; only the token id goes in and the logits come out.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::gguf::Gguf;
use crate::model::{Mat, Model};
use crate::quant::{dequant_row, DType};
use crate::Error;

/// Kernels compiled to PTX at build time (build.rs, kernels.cu); the driver finishes compiling them
/// for the installed GPU, so users need only the NVIDIA driver.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

struct GMat {
    qs: CudaSlice<i8>,
    ds: CudaSlice<f32>,
    rows: usize,
    cols: usize,
}

struct GLayer {
    attn_norm: CudaSlice<f32>,
    q: GMat,
    k: GMat,
    v: GMat,
    o: GMat,
    q_norm: Option<CudaSlice<f32>>,
    k_norm: Option<CudaSlice<f32>>,
    ffn_norm: CudaSlice<f32>,
    gate: GMat,
    up: GMat,
    down: GMat,
    kc: CudaSlice<f32>,
    vc: CudaSlice<f32>,
}

struct Funcs {
    matvec: CudaFunction,
    quant: CudaFunction,
    rmsnorm: CudaFunction,
    norm_rope: CudaFunction,
    copy: CudaFunction,
    attention: CudaFunction,
    silu_mul: CudaFunction,
    add: CudaFunction,
}

/// The model's weights, KV cache and scratch on one GPU.
pub struct GpuState {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f: Funcs,
    layers: Vec<GLayer>,
    out_norm: CudaSlice<f32>,
    output: GMat,
    x: CudaSlice<f32>,
    h: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kb: CudaSlice<f32>,
    vb: CudaSlice<f32>,
    att: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    g: CudaSlice<f32>,
    u: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    pub max_ctx: usize,
    pub vram_bytes: usize,
}

fn gerr<E: std::fmt::Debug>(e: E) -> Error {
    Error::Unsupported(format!("CUDA: {e:?}"))
}

/// Repacks a weight matrix into int8 values plus one f32 scale per 32 values. Q8_0 keeps its exact
/// values; other formats are dequantised and requantised per 32 (no loss for Q8_0).
fn repack(g: &Gguf, m: &Mat) -> (Vec<i8>, Vec<f32>) {
    let nb = m.cols / 32;
    let mut qs = vec![0i8; m.rows * m.cols];
    let mut ds = vec![0f32; m.rows * nb];
    let all = g.bytes_at(m.offset, m.rows * m.row_bytes);
    let mut row_f = vec![0f32; m.cols];
    for r in 0..m.rows {
        let row = &all[r * m.row_bytes..(r + 1) * m.row_bytes];
        if m.dtype == DType::Q8_0 {
            for b in 0..nb {
                let blk = &row[b * 34..b * 34 + 34];
                ds[r * nb + b] = crate::quant::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..32 {
                    qs[r * m.cols + b * 32 + i] = blk[2 + i] as i8;
                }
            }
        } else {
            dequant_row(m.dtype, row, &mut row_f);
            for b in 0..nb {
                let xb = &row_f[b * 32..b * 32 + 32];
                let amax = xb.iter().fold(0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                ds[r * nb + b] = d;
                for i in 0..32 {
                    qs[r * m.cols + b * 32 + i] = (xb[i] * id).round() as i8;
                }
            }
        }
    }
    (qs, ds)
}

impl GpuState {
    pub fn load(model: &Model, max_ctx: usize) -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gerr)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let lf = |n: &str| module.load_function(n).map_err(gerr);
        let f = Funcs {
            matvec: lf("matvec_q8")?,
            quant: lf("quantize_q8")?,
            rmsnorm: lf("rmsnorm")?,
            norm_rope: lf("norm_rope")?,
            copy: lf("copy_f32")?,
            attention: lf("attention")?,
            silu_mul: lf("silu_mul")?,
            add: lf("add_inplace")?,
        };
        let c = &model.cfg;
        if c.head_dim > 256 || c.head_dim % 64 != 0 {
            return Err(Error::Unsupported(format!("head size {} on the GPU", c.head_dim)));
        }
        let g = model.gguf();
        let mut vram = 0usize;
        let mut up_mat = |m: &Mat| -> Result<GMat, Error> {
            let (qs, ds) = repack(g, m);
            vram += qs.len() + ds.len() * 4;
            Ok(GMat { qs: stream.clone_htod(&qs).map_err(gerr)?, ds: stream.clone_htod(&ds).map_err(gerr)?, rows: m.rows, cols: m.cols })
        };
        let up_vec = |v: &[f32]| stream.clone_htod(v).map_err(gerr);
        let kvd = c.n_kv * c.head_dim;
        let mut layers = Vec::with_capacity(c.n_layer);
        for l in model.layers() {
            layers.push(GLayer {
                attn_norm: up_vec(&l.attn_norm)?,
                q: up_mat(&l.q)?,
                k: up_mat(&l.k)?,
                v: up_mat(&l.v)?,
                o: up_mat(&l.o)?,
                q_norm: l.q_norm.as_deref().map(up_vec).transpose()?,
                k_norm: l.k_norm.as_deref().map(up_vec).transpose()?,
                ffn_norm: up_vec(&l.ffn_norm)?,
                gate: up_mat(&l.gate)?,
                up: up_mat(&l.up)?,
                down: up_mat(&l.down)?,
                kc: stream.alloc_zeros::<f32>(max_ctx * kvd).map_err(gerr)?,
                vc: stream.alloc_zeros::<f32>(max_ctx * kvd).map_err(gerr)?,
            });
        }
        let output = up_mat(model.output_mat())?;
        vram += c.n_layer * 2 * max_ctx * kvd * 4;
        let z = |n: usize| stream.alloc_zeros::<f32>(n).map_err(gerr);
        let qd = c.n_head * c.head_dim;
        let widest = c.dim.max(c.ffn).max(qd);
        Ok(Self {
            out_norm: up_vec(model.out_norm())?,
            output,
            x: z(c.dim)?,
            h: z(c.dim)?,
            q: z(qd)?,
            kb: z(kvd)?,
            vb: z(kvd)?,
            att: z(qd)?,
            tmp: z(c.dim)?,
            g: z(c.ffn)?,
            u: z(c.ffn)?,
            xq: stream.alloc_zeros::<i8>(widest).map_err(gerr)?,
            xd: z(widest / 32)?,
            logits: z(c.vocab)?,
            layers,
            f,
            stream,
            _ctx: ctx,
            max_ctx,
            vram_bytes: vram,
        })
    }

    fn mv(&mut self, which: MatRef, src: Src, dst: Dst) -> Result<(), Error> {
        let n = match src {
            Src::H => self.h.len(),
            Src::Att => self.att.len(),
            Src::G => self.g.len(),
            Src::X => self.x.len(),
        };
        let (xq, xd) = (&mut self.xq, &mut self.xd);
        let s = &self.stream;
        {
            let x = match src {
                Src::H => &self.h,
                Src::Att => &self.att,
                Src::G => &self.g,
                Src::X => &self.x,
            };
            let warps = n / 32;
            let cfg = LaunchConfig { grid_dim: ((warps as u32 * 32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let ni = n as i32;
            unsafe { s.launch_builder(&self.f.quant).arg(x).arg(&mut *xq).arg(&mut *xd).arg(&ni).launch(cfg) }.map_err(gerr)?;
        }
        let m = match which {
            MatRef::L(li, k) => {
                let l = &self.layers[li];
                match k {
                    0 => &l.q,
                    1 => &l.k,
                    2 => &l.v,
                    3 => &l.o,
                    4 => &l.gate,
                    5 => &l.up,
                    _ => &l.down,
                }
            }
            MatRef::Out => &self.output,
        };
        let y = match dst {
            Dst::Q => &mut self.q,
            Dst::K => &mut self.kb,
            Dst::V => &mut self.vb,
            Dst::Tmp => &mut self.tmp,
            Dst::G => &mut self.g,
            Dst::U => &mut self.u,
            Dst::Logits => &mut self.logits,
        };
        let (rows, cols) = (m.rows as i32, m.cols as i32);
        let cfg = LaunchConfig { grid_dim: ((m.rows as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { s.launch_builder(&self.f.matvec).arg(&m.qs).arg(&m.ds).arg(&self.xq).arg(&self.xd).arg(y).arg(&rows).arg(&cols).launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    /// Runs one token at position `pos` and copies the logits into `out`.
    pub fn step(&mut self, model: &Model, token: u32, pos: usize, out: &mut [f32]) -> Result<(), Error> {
        let c = model.cfg.clone();
        let mut emb = vec![0f32; c.dim];
        model.embed_into(token, &mut emb);
        self.stream.memcpy_htod(&emb, &mut self.x).map_err(gerr)?;
        let hd = c.head_dim as i32;
        let kvd = c.n_kv * c.head_dim;
        let eps = c.eps;
        let base = c.rope_base;
        let posi = pos as i32;
        let block = |n: usize| LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let one = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let dim = c.dim as i32;
        for li in 0..c.n_layer {
            {
                let l = &self.layers[li];
                unsafe { self.stream.launch_builder(&self.f.rmsnorm).arg(&self.x).arg(&l.attn_norm).arg(&mut self.h).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
            }
            self.mv(MatRef::L(li, 0), Src::H, Dst::Q)?;
            self.mv(MatRef::L(li, 1), Src::H, Dst::K)?;
            self.mv(MatRef::L(li, 2), Src::H, Dst::V)?;
            let rope_cfg = |heads: usize| LaunchConfig { grid_dim: (heads as u32, 1, 1), block_dim: ((c.head_dim / 2) as u32, 1, 1), shared_mem_bytes: 0 };
            let s = &self.stream;
            let l = &mut self.layers[li];
            let has_q = l.q_norm.is_some() as i32;
            let null: u64 = 0;
            {
                let qn = l.q_norm.as_ref().unwrap_or(&l.attn_norm);
                unsafe { s.launch_builder(&self.f.norm_rope).arg(&mut self.q).arg(qn).arg(&has_q).arg(&hd).arg(&posi).arg(&base).arg(&eps).arg(&null).launch(rope_cfg(c.n_head)) }.map_err(gerr)?;
            }
            {
                // Rotate K straight into its cache slot.
                let kn = l.k_norm.as_ref().unwrap_or(&l.attn_norm);
                let mut slot = l.kc.slice_mut(pos * kvd..(pos + 1) * kvd);
                unsafe { s.launch_builder(&self.f.norm_rope).arg(&mut self.kb).arg(kn).arg(&has_q).arg(&hd).arg(&posi).arg(&base).arg(&eps).arg(&mut slot).launch(rope_cfg(c.n_kv)) }.map_err(gerr)?;
            }
            {
                let mut slot = l.vc.slice_mut(pos * kvd..(pos + 1) * kvd);
                let n = kvd as i32;
                unsafe { s.launch_builder(&self.f.copy).arg(&self.vb).arg(&mut slot).arg(&n).launch(block(kvd)) }.map_err(gerr)?;
            }
            {
                let len = (pos + 1) as i32;
                let group = (c.n_head / c.n_kv) as i32;
                let scale = 1.0f32 / (c.head_dim as f32).sqrt();
                let n_kv = c.n_kv as i32;
                let cfg = LaunchConfig { grid_dim: (c.n_head as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                unsafe { s.launch_builder(&self.f.attention).arg(&self.q).arg(&l.kc).arg(&l.vc).arg(&mut self.att).arg(&n_kv).arg(&hd).arg(&len).arg(&group).arg(&scale).launch(cfg) }.map_err(gerr)?;
            }
            self.mv(MatRef::L(li, 3), Src::Att, Dst::Tmp)?;
            unsafe { self.stream.launch_builder(&self.f.add).arg(&mut self.x).arg(&self.tmp).arg(&dim).launch(block(c.dim)) }.map_err(gerr)?;
            {
                let l = &self.layers[li];
                unsafe { self.stream.launch_builder(&self.f.rmsnorm).arg(&self.x).arg(&l.ffn_norm).arg(&mut self.h).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
            }
            self.mv(MatRef::L(li, 4), Src::H, Dst::G)?;
            self.mv(MatRef::L(li, 5), Src::H, Dst::U)?;
            let ffn = c.ffn as i32;
            unsafe { self.stream.launch_builder(&self.f.silu_mul).arg(&mut self.g).arg(&self.u).arg(&ffn).launch(block(c.ffn)) }.map_err(gerr)?;
            self.mv(MatRef::L(li, 6), Src::G, Dst::Tmp)?;
            unsafe { self.stream.launch_builder(&self.f.add).arg(&mut self.x).arg(&self.tmp).arg(&dim).launch(block(c.dim)) }.map_err(gerr)?;
        }
        unsafe { self.stream.launch_builder(&self.f.rmsnorm).arg(&self.x).arg(&self.out_norm).arg(&mut self.h).arg(&dim).arg(&eps).launch(one) }.map_err(gerr)?;
        self.mv(MatRef::Out, Src::H, Dst::Logits)?;
        self.stream.memcpy_dtoh(&self.logits, out).map_err(gerr)?;
        self.stream.synchronize().map_err(gerr)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum MatRef {
    L(usize, u8),
    Out,
}
#[derive(Clone, Copy)]
enum Src {
    H,
    Att,
    G,
    #[allow(dead_code)]
    X,
}
#[derive(Clone, Copy)]
enum Dst {
    Q,
    K,
    V,
    Tmp,
    G,
    U,
    Logits,
}
