//! The GPU tensor backend for everything that is not a language model (image encoders, speech
//! encoders, the speech-synthesis decoder, image generation): f16 tensor-core matrix products with
//! f32 accumulation (`tensor.cu`), and the normalisations, activations and layout changes around
//! them. Activations stay on the card as f32 rows of channels; weights are uploaded once as f16.
//! NVIDIA, compute capability 8.0+ (Ampere and newer); callers fall back to the processor otherwise.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::vision::Lin;
use crate::Error;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tensor.ptx"));

fn gerr<E: std::fmt::Debug>(e: E) -> Error {
    Error::Unsupported(format!("GPU: {e:?}"))
}

/// Device activations: f32, row-major.
pub type DBuf = CudaSlice<f32>;

/// A weight matrix on the card: f16 rows padded to `ld` (a multiple of 8), and an f32 bias.
pub struct DW {
    pub rows: usize,
    pub cols: usize,
    ld: usize,
    w: CudaSlice<u16>,
    bias: Option<CudaSlice<f32>>,
}

/// A k x k convolution on the card (weights rows over (channel, ky, kx)).
pub struct DConv {
    pub w: DW,
    pub ic: usize,
    pub k: usize,
    pub stride: usize,
    /// Weights reordered (ky, kx, channel) for the implicit-GEMM kernel (ic % 8 == 0, k > 1).
    implicit: Option<CudaSlice<u16>>,
}

struct Funcs {
    gemm: CudaFunction,
    conv: CudaFunction,
    conv_wide: CudaFunction,
    conv_h: CudaFunction,
    conv_wide_h: CudaFunction,
    flash64: CudaFunction,
    group_partial: CudaFunction,
    group_merge: CudaFunction,
    to_f16: CudaFunction,
    transpose: CudaFunction,
    im2col: CudaFunction,
    upsample2: CudaFunction,
    concat: CudaFunction,
    act: CudaFunction,
    geglu: CudaFunction,
    add_rows: CudaFunction,
    row_norm: CudaFunction,
    group_apply: CudaFunction,
    softmax: CudaFunction,
}

pub struct Gx {
    _ctx: Arc<CudaContext>,
    /// Streaming multiprocessors (for choosing tile sizes).
    sms: usize,
    /// Convolutions and linears sum each 32-value slice in f16 (faster on GeForce cards); set per
    /// model by its caller after checking the model tolerates it.
    half_acc: std::sync::atomic::AtomicBool,
    pub s: Arc<CudaStream>,
    f: Funcs,
    /// CEREBRO_GX_PROF: time per operation (synchronising after each), printed by `report`.
    prof: Option<std::sync::Mutex<std::collections::BTreeMap<&'static str, (u32, f64)>>>,
}

/// Times one operation when profiling.
struct Span<'a>(&'a Gx, &'static str, std::time::Instant);

impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Some(p) = &self.0.prof {
            let _ = self.0.s.synchronize();
            let mut m = p.lock().unwrap_or_else(|e| e.into_inner());
            let e = m.entry(self.1).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += self.2.elapsed().as_secs_f64();
        }
    }
}

/// Activations for `act`.
#[derive(Clone, Copy)]
pub enum Act {
    Silu = 0,
    GeluTanh = 1,
    GeluErf = 2,
}

fn grid(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

static GX: OnceLock<Option<Gx>> = OnceLock::new();

impl Gx {
    /// The card, when there is an NVIDIA GPU with tensor cores this backend runs on (and the
    /// "use the graphics card" switch is on).
    pub fn get() -> Option<&'static Gx> {
        GX.get_or_init(|| {
            if !crate::switches::enabled(crate::switches::GPU_ENABLED) || std::env::var_os("CEREBRO_TENSOR_CPU").is_some() {
                return None;
            }
            Gx::new().map_err(|e| eprintln!("GPU tensor backend: {e}")).ok()
        })
        .as_ref()
    }

    fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(gerr)?;
        if ctx.compute_capability().map_err(gerr)?.0 < 8 {
            return Err(Error::Unsupported("tensor cores of compute capability 8.0+ are needed".into()));
        }
        let s = ctx.new_stream().map_err(gerr)?;
        // Keep freed memory in the pool between images instead of returning it to the driver at
        // every synchronisation (allocations are stream-ordered).
        unsafe {
            use cudarc::driver::sys;
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            if sys::cuDeviceGetDefaultMemPool(&mut pool, 0) == sys::CUresult::CUDA_SUCCESS {
                let mut max = u64::MAX;
                sys::cuMemPoolSetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&mut max as *mut u64).cast());
            }
        }
        let m = ctx.load_module(Ptx::from_src(PTX)).map_err(gerr)?;
        let lf = |n: &str| m.load_function(n).map_err(gerr);
        let f = Funcs {
            gemm: lf("gemm_f16")?,
            conv: lf("conv_f16")?,
            conv_wide: lf("conv_f16_wide")?,
            conv_h: lf("conv_f16_h")?,
            conv_wide_h: lf("conv_f16_wide_h")?,
            flash64: lf("flash_attn64")?,
            group_partial: lf("group_partial")?,
            group_merge: lf("group_merge")?,
            to_f16: lf("to_f16")?,
            transpose: lf("transpose_f16")?,
            im2col: lf("im2col_f16")?,
            upsample2: lf("upsample2")?,
            concat: lf("concat_rows")?,
            act: lf("act")?,
            geglu: lf("geglu")?,
            add_rows: lf("add_rows")?,
            row_norm: lf("row_norm")?,
            group_apply: lf("group_apply")?,
            softmax: lf("softmax_f16")?,
        };
        let prof = std::env::var_os("CEREBRO_GX_PROF").map(|_| std::sync::Mutex::new(Default::default()));
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap_or(40).max(1) as usize;
        Ok(Gx { _ctx: ctx, sms, s, f, prof, half_acc: std::sync::atomic::AtomicBool::new(false) })
    }

    pub fn up(&self, x: &[f32]) -> Result<DBuf, Error> {
        self.s.clone_htod(x).map_err(gerr)
    }

    pub fn down(&self, x: &DBuf) -> Result<Vec<f32>, Error> {
        self.s.clone_dtoh(x).map_err(gerr)
    }

    pub fn zeros(&self, n: usize) -> Result<DBuf, Error> {
        self.s.alloc_zeros::<f32>(n.max(1)).map_err(gerr)
    }

    /// An output buffer the next operation overwrites in full (not cleared first).
    pub fn buf(&self, n: usize) -> Result<DBuf, Error> {
        self.raw::<f32>(n)
    }

    fn raw<T: cudarc::driver::DeviceRepr>(&self, n: usize) -> Result<CudaSlice<T>, Error> {
        // SAFETY: every caller's next kernel writes all n values before any is read.
        unsafe { self.s.alloc::<T>(n.max(1)) }.map_err(gerr)
    }

    /// Uploads a weight matrix as f16 (rows padded to a multiple of 8 columns).
    pub(crate) fn weights(&self, l: &Lin) -> Result<DW, Error> {
        let ld = pad8(l.cols);
        let mut h = vec![0u16; l.rows * ld];
        for r in 0..l.rows {
            for c in 0..l.cols {
                h[r * ld + c] = crate::quant::to_f16(l.w[r * l.cols + c]);
            }
        }
        let has_bias = l.bias.iter().any(|v| *v != 0.0);
        Ok(DW { rows: l.rows, cols: l.cols, ld, w: self.s.clone_htod(&h).map_err(gerr)?, bias: if has_bias { Some(self.s.clone_htod(&l.bias).map_err(gerr)?) } else { None } })
    }

    pub(crate) fn conv(&self, l: &Lin, ic: usize, k: usize, stride: usize) -> Result<DConv, Error> {
        let implicit = if ic % 8 == 0 && k > 1 {
            let kk = k * k;
            let mut h = vec![0u16; l.rows * ic * kk];
            for o in 0..l.rows {
                for ch in 0..ic {
                    for t in 0..kk {
                        h[o * ic * kk + t * ic + ch] = crate::quant::to_f16(l.w[o * ic * kk + ch * kk + t]);
                    }
                }
            }
            Some(self.s.clone_htod(&h).map_err(gerr)?)
        } else {
            None
        };
        Ok(DConv { w: self.weights(l)?, ic, k, stride, implicit })
    }

    /// f16 copy of `rows` x `cols` f32 values (row stride `ldx`) with rows padded to `ld`.
    fn f16(&self, x: &CudaView<f32>, ldx: usize, rows: usize, cols: usize, ld: usize) -> Result<CudaSlice<u16>, Error> {
        let mut y = self.raw::<u16>((rows * ld).max(1))?;
        let (ldx_i, ld_i, rows_i, cols_i) = (ldx as i32, ld as i32, rows as i32, cols as i32);
        let mut b = self.s.launch_builder(&self.f.to_f16);
        b.arg(x).arg(&ldx_i).arg(&mut y).arg(&ld_i).arg(&rows_i).arg(&cols_i);
        unsafe { b.launch(grid(rows * ld)) }.map_err(gerr)?;
        Ok(y)
    }

    /// Y (T x ldy) = alpha * B A^T (+ bias) (+ Y): A rows x K (stride lda), B T x K (stride ldb), f16.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, a: &CudaView<u16>, lda: usize, b: &CudaView<u16>, ldb: usize, y: &mut CudaViewMut<f32>, ldy: usize, rows: usize, t: usize, k: usize, bias: Option<&CudaSlice<f32>>, add: bool, alpha: f32) -> Result<(), Error> {
        if rows == 0 || t == 0 {
            return Ok(());
        }
        let cfg = LaunchConfig { grid_dim: ((rows as u32).div_ceil(128), (t as u32).div_ceil(64), 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (lda, ldb, ldy, rows_i, t_i, k, add) = (lda as i32, ldb as i32, ldy as i32, rows as i32, t as i32, k as i32, add as i32);
        let null: u64 = 0;
        let mut lb = self.s.launch_builder(&self.f.gemm);
        lb.arg(a).arg(&lda).arg(b).arg(&ldb).arg(y).arg(&ldy).arg(&rows_i).arg(&t_i).arg(&k);
        match bias {
            Some(bb) => lb.arg(bb),
            None => lb.arg(&null),
        };
        lb.arg(&add).arg(&alpha);
        unsafe { lb.launch(cfg) }.map(|_| ()).map_err(gerr)
    }

    /// The implicit-GEMM kernel: out = conv(x [h][w][c], weights wt with rows of k*k*c) (+ out).
    #[allow(clippy::too_many_arguments)]
    fn conv_launch(&self, wt: &CudaSlice<u16>, ldw: usize, bias: Option<&CudaSlice<f32>>, x: &DBuf, out: &mut DBuf, oc: usize, h: usize, w: usize, c: usize, k: usize, s: usize, oh: usize, ow: usize, add: bool) -> Result<(), Error> {
        // Wide pixel tiles once there are enough tiles to fill the card twice over.
        let wide = (oc.div_ceil(128) * (oh * ow).div_ceil(128)) >= 2 * self.sms && std::env::var_os("CEREBRO_CONV_NARROW").is_none();
        let bn = if wide { 128 } else { 64 };
        let cfg = LaunchConfig { grid_dim: ((oc as u32).div_ceil(128), ((oh * ow) as u32).div_ceil(bn), 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ldw, oci, hi, wi, ci, ki, si, ohi, owi, addi) = (ldw as i32, oc as i32, h as i32, w as i32, c as i32, k as i32, s as i32, oh as i32, ow as i32, add as i32);
        let null: u64 = 0;
        let h16 = self.half_acc.load(std::sync::atomic::Ordering::Relaxed);
        let f = match (wide, h16) {
            (true, true) => &self.f.conv_wide_h,
            (true, false) => &self.f.conv_wide,
            (false, true) => &self.f.conv_h,
            (false, false) => &self.f.conv,
        };
        let mut b = self.s.launch_builder(f);
        b.arg(wt).arg(&ldw).arg(x).arg(out).arg(&oci).arg(&hi).arg(&wi).arg(&ci).arg(&ki).arg(&si).arg(&ohi).arg(&owi);
        match bias {
            Some(bb) => b.arg(bb),
            None => b.arg(&null),
        };
        b.arg(&addi);
        unsafe { b.launch(cfg) }.map(|_| ()).map_err(gerr)
    }

    /// out (n x w.rows) = x (n x w.cols) W^T + bias (+ out when `add`).
    pub fn linear(&self, x: &DBuf, n: usize, w: &DW, out: &mut DBuf, add: bool) -> Result<(), Error> {
        let _sp = self.span("linear");
        if w.cols % 8 == 0 {
            // Reads x as f32 directly: a 1x1 convolution over n pixels.
            return self.conv_launch(&w.w, w.ld, w.bias.as_ref(), x, out, w.rows, 1, n, w.cols, 1, 1, 1, n, add);
        }
        let xh = self.f16(&x.slice(..), w.cols, n, w.cols, w.ld)?;
        self.gemm(&w.w.slice(..), w.ld, &xh.slice(..), w.ld, &mut out.slice_mut(..), w.rows, w.rows, n, w.ld, w.bias.as_ref(), add, 1.0)
    }

    /// x [h][w][c] -> ([oh][ow][oc], oh, ow), padding k/2.
    pub fn conv2d(&self, x: &DBuf, h: usize, w: usize, cv: &DConv) -> Result<(DBuf, usize, usize), Error> {
        let _sp = self.span("conv2d");
        let (k, s, ic) = (cv.k, cv.stride, cv.ic);
        let pad = k / 2;
        let (oh, ow) = ((h + 2 * pad - k) / s + 1, (w + 2 * pad - k) / s + 1);
        let oc = cv.w.rows;
        let mut out = self.buf(oh * ow * oc)?;
        if k == 1 && s == 1 {
            self.linear(x, h * w, &cv.w, &mut out, false)?;
            return Ok((out, oh, ow));
        }
        if let Some(wi) = &cv.implicit {
            self.conv_launch(wi, ic * k * k, cv.w.bias.as_ref(), x, &mut out, oc, h, w, ic, k, s, oh, ow, false)?;
            return Ok((out, oh, ow));
        }
        let ld = cv.w.ld;
        let total = oh * ow;
        // Patch rows per pass so the f16 patch matrix stays near 256 MB.
        let per = ((128_000_000 / ld).max(64)).min(total);
        let mut p0 = 0;
        while p0 < total {
            let n = per.min(total - p0);
            let mut patches = self.raw::<u16>(n * ld)?;
            let (hi, wi, ci, ki, si, owi, p0i, ni, ldi) = (h as i32, w as i32, ic as i32, k as i32, s as i32, ow as i32, p0 as i32, n as i32, ld as i32);
            let mut b = self.s.launch_builder(&self.f.im2col);
            b.arg(x).arg(&hi).arg(&wi).arg(&ci).arg(&ki).arg(&si).arg(&owi).arg(&p0i).arg(&ni).arg(&mut patches).arg(&ldi);
            unsafe { b.launch(grid(n * ld)) }.map_err(gerr)?;
            self.gemm(&cv.w.w.slice(..), ld, &patches.slice(..), ld, &mut out.slice_mut(p0 * oc..(p0 + n) * oc), oc, oc, n, ld, cv.w.bias.as_ref(), false, 1.0)?;
            p0 += n;
        }
        Ok((out, oh, ow))
    }

    pub fn act(&self, x: &mut DBuf, n: usize, op: Act) -> Result<(), Error> {
        let _sp = self.span("act");
        let opi = op as i32;
        let nn = n as u64;
        let mut b = self.s.launch_builder(&self.f.act);
        b.arg(x).arg(&nn).arg(&opi);
        unsafe { b.launch(grid(n)) }.map(|_| ()).map_err(gerr)
    }

    /// y[r][j] = x[r][j] * gelu_erf(x[r][inner + j]).
    pub fn geglu(&self, x: &DBuf, n: usize, inner: usize) -> Result<DBuf, Error> {
        let _sp = self.span("geglu");
        let mut y = self.buf(n * inner)?;
        let (ni, ii) = (n as i32, inner as i32);
        let mut b = self.s.launch_builder(&self.f.geglu);
        b.arg(x).arg(&mut y).arg(&ni).arg(&ii);
        unsafe { b.launch(grid(n * inner)) }.map_err(gerr)?;
        Ok(y)
    }

    /// x += other (elementwise, n values).
    pub fn add(&self, x: &mut DBuf, other: &DBuf, n: usize) -> Result<(), Error> {
        let _sp = self.span("add");
        let null: u64 = 0;
        let (nn, c) = (n as u64, 1i32);
        let mut b = self.s.launch_builder(&self.f.add_rows);
        b.arg(x).arg(&null).arg(other).arg(&nn).arg(&c);
        unsafe { b.launch(grid(n)) }.map(|_| ()).map_err(gerr)
    }

    /// x[r][c] += v[c] for n values in rows of c.
    pub fn add_vec(&self, x: &mut DBuf, v: &DBuf, n: usize, c: usize) -> Result<(), Error> {
        let _sp = self.span("add_vec");
        let null: u64 = 0;
        let (nn, ci) = (n as u64, c as i32);
        let mut b = self.s.launch_builder(&self.f.add_rows);
        b.arg(x).arg(v).arg(&null).arg(&nn).arg(&ci);
        unsafe { b.launch(grid(n)) }.map(|_| ()).map_err(gerr)
    }

    /// LayerNorm (or RMSNorm when `rms`) of each row of `dim`; `b` may be None.
    pub fn row_norm(&self, x: &DBuf, rows: usize, dim: usize, w: &DBuf, b: Option<&DBuf>, eps: f32, rms: bool) -> Result<DBuf, Error> {
        let _sp = self.span("row_norm");
        let mut y = self.buf(rows * dim)?;
        let (di, ri) = (dim as i32, rms as i32);
        let null: u64 = 0;
        let mut lb = self.s.launch_builder(&self.f.row_norm);
        lb.arg(x).arg(&mut y).arg(w);
        match b {
            Some(bb) => lb.arg(bb),
            None => lb.arg(&null),
        };
        lb.arg(&di).arg(&eps).arg(&ri);
        unsafe { lb.launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
        Ok(y)
    }

    /// GroupNorm over n pixels of c channels in `groups` groups, then SiLU when `silu`.
    pub fn group_norm(&self, x: &DBuf, n: usize, c: usize, groups: usize, w: &DBuf, b: &DBuf, eps: f32, silu: bool) -> Result<DBuf, Error> {
        let _sp = self.span("group_norm");
        let mut stats = self.raw::<f32>(2 * groups)?;
        let (ni, ci, gi) = (n as i32, c as i32, groups as i32);
        let parts = n.div_ceil(64);
        let mut part = self.raw::<f32>(parts * c * 2)?;
        let mut lb = self.s.launch_builder(&self.f.group_partial);
        lb.arg(x).arg(&mut part).arg(&ni).arg(&ci).arg(&gi);
        unsafe { lb.launch(LaunchConfig { grid_dim: (parts as u32, 1, 1), block_dim: ((c as u32).min(256), 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
        let pi = parts as i32;
        let mut lb = self.s.launch_builder(&self.f.group_merge);
        lb.arg(x).arg(&part).arg(&mut stats).arg(&pi).arg(&ni).arg(&ci).arg(&gi).arg(&eps);
        unsafe { lb.launch(LaunchConfig { grid_dim: (groups as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
        let mut y = self.buf(n * c)?;
        let (nn, si) = ((n * c) as u64, silu as i32);
        let mut lb = self.s.launch_builder(&self.f.group_apply);
        lb.arg(x).arg(&mut y).arg(&stats).arg(w).arg(b).arg(&nn).arg(&ci).arg(&gi).arg(&si);
        unsafe { lb.launch(grid(n * c)) }.map_err(gerr)?;
        Ok(y)
    }

    pub fn upsample2(&self, x: &DBuf, h: usize, w: usize, c: usize) -> Result<DBuf, Error> {
        let _sp = self.span("upsample2");
        let mut y = self.buf(4 * h * w * c)?;
        let (hi, wi, ci) = (h as i32, w as i32, c as i32);
        let mut b = self.s.launch_builder(&self.f.upsample2);
        b.arg(x).arg(&mut y).arg(&hi).arg(&wi).arg(&ci);
        unsafe { b.launch(grid(4 * h * w * c)) }.map_err(gerr)?;
        Ok(y)
    }

    /// [n][ca + cb] from [n][ca] and [n][cb].
    pub fn concat(&self, a: &DBuf, ca: usize, b: &DBuf, cb: usize, n: usize) -> Result<DBuf, Error> {
        let _sp = self.span("concat");
        let mut y = self.buf(n * (ca + cb))?;
        let (cai, cbi, ni) = (ca as i32, cb as i32, n as i32);
        let mut lb = self.s.launch_builder(&self.f.concat);
        lb.arg(a).arg(&cai).arg(b).arg(&cbi).arg(&mut y).arg(&ni);
        unsafe { lb.launch(grid(n * (ca + cb))) }.map_err(gerr)?;
        Ok(y)
    }

    /// Multi-head attention: q rows (nq, stride ldq), k and v rows (nk, strides ldk, ldv), each
    /// head at column h*hd of its buffer; out (nq x heads*hd). Query i sees keys 0..=i when
    /// `causal`. Per head: S = Q K^T and O = softmax(S) V as tensor-core matrix products.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, q: &CudaView<f32>, ldq: usize, k: &CudaView<f32>, ldk: usize, v: &CudaView<f32>, ldv: usize, nq: usize, nk: usize, heads: usize, hd: usize, causal: bool, out: &mut DBuf) -> Result<(), Error> {
        let _sp = self.span("attention");
        if hd == 64 && ldq % 4 == 0 && ldk % 4 == 0 && ldv % 4 == 0 {
            let cfg = LaunchConfig { grid_dim: ((nq as u32).div_ceil(64), heads as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let (ldqi, ldki, ldvi, ldoi, nqi, nki, ci) = (ldq as i32, ldk as i32, ldv as i32, (heads * hd) as i32, nq as i32, nk as i32, causal as i32);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut b = self.s.launch_builder(&self.f.flash64);
            b.arg(q).arg(&ldqi).arg(k).arg(&ldki).arg(v).arg(&ldvi).arg(out).arg(&ldoi).arg(&nqi).arg(&nki).arg(&scale).arg(&ci);
            return unsafe { b.launch(cfg) }.map(|_| ()).map_err(gerr);
        }
        let nkp = pad8(nk);
        let hdp = pad8(hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut s = self.buf(nq * nkp)?;
        let mut p = self.raw::<u16>(nq * nkp)?;
        let mut vt = self.raw::<u16>(hd * nkp)?;
        for h in 0..heads {
            let qh = self.f16(&q.slice(h * hd..), ldq, nq, hd, hdp)?;
            let kh = self.f16(&k.slice(h * hd..), ldk, nk, hd, hdp)?;
            self.gemm(&kh.slice(..), hdp, &qh.slice(..), hdp, &mut s.slice_mut(..), nkp, nk, nq, hdp, None, false, 1.0)?;
            let (ni, ldi) = (nk as i32, nkp as i32);
            let mut b = self.s.launch_builder(&self.f.softmax);
            let ci = causal as i32;
            b.arg(&s).arg(&mut p).arg(&ni).arg(&ldi).arg(&scale).arg(&ci);
            unsafe { b.launch(LaunchConfig { grid_dim: (nq as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            let (ldvi, ldti, nki, hdi) = (ldv as i32, nkp as i32, nk as i32, hd as i32);
            let vh = v.slice(h * hd..);
            let mut b = self.s.launch_builder(&self.f.transpose);
            b.arg(&vh).arg(&ldvi).arg(&mut vt).arg(&ldti).arg(&nki).arg(&hdi);
            unsafe { b.launch(LaunchConfig { grid_dim: ((hd as u32).div_ceil(32), (nkp as u32).div_ceil(32), 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 }) }.map_err(gerr)?;
            let ldo = heads * hd;
            self.gemm(&vt.slice(..), nkp, &p.slice(..), nkp, &mut out.slice_mut(h * hd..), ldo, hd, nq, nkp, None, false, 1.0)?;
        }
        Ok(())
    }

    pub fn copy(&self, x: &DBuf) -> Result<DBuf, Error> {
        let _sp = self.span("copy");
        self.s.clone_dtod(x).map_err(gerr)
    }

    fn span(&self, name: &'static str) -> Span<'_> {
        if self.prof.is_some() {
            let _ = self.s.synchronize();
        }
        Span(self, name, std::time::Instant::now())
    }

    /// The profile so far (CEREBRO_GX_PROF), slowest first; clears it.
    pub fn report(&self) -> String {
        let Some(p) = &self.prof else { return String::new() };
        let mut m = p.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<_> = m.iter().map(|(k, (n, t))| (*k, *n, *t)).collect();
        m.clear();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v.iter().map(|(k, n, t)| format!("{k:>12} {n:5} x {:8.2} ms
", t * 1e3)).collect()
    }

    /// Whether convolutions and linears accumulate each 32-value slice in f16 (see `half_acc`).
    pub fn set_half_accumulation(&self, on: bool) {
        self.half_acc.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn sync(&self) -> Result<(), Error> {
        self.s.synchronize().map_err(gerr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_lin(rows: usize, cols: usize, seed: u32) -> Lin {
        let w: Vec<f32> = (0..rows * cols).map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 1000.0 - 0.5).collect();
        let bias = (0..rows).map(|r| r as f32 * 0.01).collect();
        Lin { rows, cols, w, bias }
    }

    #[test]
    fn the_gemm_matches_the_processor() {
        let Some(g) = Gx::get() else { return eprintln!("no tensor-core GPU; skipped") };
        for (rows, cols, n) in [(17, 40, 5), (130, 72, 70), (300, 1024, 129)] {
            let l = cpu_lin(rows, cols, 3);
            let x: Vec<f32> = (0..n * cols).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();
            let w = g.weights(&l).expect("upload");
            let dx = g.up(&x).expect("up");
            let mut dy = g.zeros(n * rows).expect("zeros");
            g.linear(&dx, n, &w, &mut dy, false).expect("linear");
            let y = g.down(&dy).expect("down");
            let mut want = vec![0f32; n * rows];
            crate::vision::linear(&l, &x, n, &mut want, false);
            let err = y.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs() / (1.0 + b.abs())));
            assert!(err < 2e-2, "{rows}x{cols} n{n}: {err}");
        }
    }

    #[test]
    fn causal_flash_attention_matches_a_plain_softmax() {
        let Some(g) = Gx::get() else { return eprintln!("no tensor-core GPU; skipped") };
        let (n, heads, hd) = (77, 2, 64);
        let dim = heads * hd;
        let x: Vec<f32> = (0..n * 3 * dim).map(|i| (((i * 37) % 101) as f32 / 101.0 - 0.5) * 2.0).collect();
        let mut want = vec![0f32; n * dim];
        for h in 0..heads {
            for i in 0..n {
                let qi = &x[i * 3 * dim + h * hd..i * 3 * dim + (h + 1) * hd];
                let s: Vec<f32> = (0..=i).map(|j| qi.iter().zip(&x[j * 3 * dim + dim + h * hd..]).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()).collect();
                let m = s.iter().fold(f32::MIN, |a, &b| a.max(b));
                let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..hd {
                    want[i * dim + h * hd + d] = (0..=i).map(|j| e[j] / sum * x[j * 3 * dim + 2 * dim + h * hd + d]).sum();
                }
            }
        }
        let d = g.up(&x).expect("up");
        let mut out = g.zeros(n * dim).expect("zeros");
        g.attention(&d.slice(..), 3 * dim, &d.slice(dim..), 3 * dim, &d.slice(2 * dim..), 3 * dim, n, n, heads, hd, true, &mut out).expect("attention");
        let got = g.down(&out).expect("down");
        let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-2, "{err}");
    }

    #[test]
    fn attention_matches_the_processor() {
        let Some(g) = Gx::get() else { return eprintln!("no tensor-core GPU; skipped") };
        let (n, heads, hd) = (37, 3, 64);
        let dim = heads * hd;
        let qkv: Vec<f32> = (0..n * 3 * dim).map(|i| (((i * 31) % 97) as f32 / 97.0 - 0.5) * 2.0).collect();
        let mut want = vec![0f32; n * dim];
        crate::vision::attention(&qkv, n, heads, hd, &mut want);
        let d = g.up(&qkv).expect("up");
        let mut out = g.zeros(n * dim).expect("zeros");
        g.attention(&d.slice(..), 3 * dim, &d.slice(dim..), 3 * dim, &d.slice(2 * dim..), 3 * dim, n, n, heads, hd, false, &mut out).expect("attention");
        let got = g.down(&out).expect("down");
        let err = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-2, "{err}");
    }
}
