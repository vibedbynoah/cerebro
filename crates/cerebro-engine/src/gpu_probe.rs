//! What this graphics card's tensor cores actually do, measured: overclocks, power limits and
//! unlock mods (such as GPU Unlock for GeForce Ampere, which lifts the FP16-into-FP32 rate to the
//! professional cards') change these numbers, so they are measured on the card as it runs, not
//! looked up from a table.
use crate::gpu::{gerr, PTX_MMA};
use crate::Error;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::time::Instant;

/// Dense tensor-core throughput in trillions of operations per second (a multiply-add counts two).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TensorRates {
    /// FP16 inputs, FP32 accumulation: what attention and FP16/BF16 matrix work use.
    pub f16_f32acc: f64,
    /// FP16 inputs, FP16 accumulation (never rate-limited on GeForce).
    pub f16_f16acc: f64,
    /// INT8 inputs, INT32 accumulation: what Cerebro's prompt matrix multiplies use.
    pub int8: f64,
    /// INT4 inputs (both sides 4-bit), INT32 accumulation.
    pub int4: f64,
    /// SM clock while measuring, in MHz (the rates alone cannot tell an overclock from a boost,
    /// so this comes from the driver).
    pub sm_clock_mhz: Option<u32>,
}

impl TensorRates {
    /// FP16-into-FP32 runs at (about) the FP16-into-FP16 rate: the professional-card rate, i.e.
    /// a GeForce card with the issue-rate limit lifted. Stock GeForce Ampere shows about 0.5.
    pub fn fp32_accumulate_unlocked(&self) -> bool {
        self.f16_f32acc >= 0.8 * self.f16_f16acc
    }
}

/// Runs each kind of MMA on every SM for about 50 ms (after a warm-up) and times it.
pub fn tensor_rates() -> Result<TensorRates, Error> {
    let ctx = CudaContext::new(0).map_err(gerr)?;
    let stream = ctx.default_stream();
    let m = ctx.load_module(Ptx::from_src(PTX_MMA)).map_err(gerr)?;
    let f = m.load_function("tensor_probe").map_err(gerr)?;
    let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).map_err(gerr)? as u32;
    let mut out = stream.alloc_zeros::<f32>((sms * 4 * 256) as usize).map_err(gerr)?;
    let cfg = LaunchConfig { grid_dim: (sms * 4, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let mut rate = |kind: i32| -> Result<f64, Error> {
        // MMA shape: m16n8k16 (f16), m16n8k32 (int8) or m16n8k64 (int4); 8 per round per warp.
        let ops_per_mma = 2.0 * 16.0 * 8.0 * match kind { 2 => 32.0, 3 => 64.0, _ => 16.0 };
        let mut iters = 256i32;
        loop {
            unsafe { stream.launch_builder(&f).arg(&kind).arg(&iters).arg(&mut out).launch(cfg) }.map_err(gerr)?;
            stream.synchronize().map_err(gerr)?;
            let t = Instant::now();
            unsafe { stream.launch_builder(&f).arg(&kind).arg(&iters).arg(&mut out).launch(cfg) }.map_err(gerr)?;
            stream.synchronize().map_err(gerr)?;
            let s = t.elapsed().as_secs_f64();
            if s > 0.05 || iters > 1 << 22 {
                let warps = (sms * 4 * 8) as f64;
                return Ok(warps * iters as f64 * 8.0 * ops_per_mma / s / 1e12);
            }
            iters *= 4;
        }
    };
    let (a, b, c, d) = (rate(0)?, rate(1)?, rate(2)?, rate(3)?);
    Ok(TensorRates { f16_f32acc: a, f16_f16acc: b, int8: c, int4: d, sm_clock_mhz: sm_clock() })
}

fn sm_clock() -> Option<u32> {
    let o = std::process::Command::new("nvidia-smi").args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]).output().ok()?;
    String::from_utf8_lossy(&o.stdout).lines().next()?.trim().parse().ok()
}
