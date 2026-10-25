//! Measured tensor-core rates of this card (shows whether an FP16 unlock mod or an overclock is in
//! effect). cargo run --release --example tensorprobe
fn main() {
    let r = cerebro_engine::gpu_probe::tensor_rates().expect("CUDA");
    println!("FP16 -> FP32 accumulate: {:6.1} TFLOPS", r.f16_f32acc);
    println!("FP16 -> FP16 accumulate: {:6.1} TFLOPS", r.f16_f16acc);
    println!("INT8 -> INT32:           {:6.1} TOPS", r.int8);
    println!("INT4 -> INT32:           {:6.1} TOPS", r.int4);
    println!("SM clock: {}", r.sm_clock_mhz.map_or("unknown".into(), |m| format!("{m} MHz")));
    println!("FP32-accumulate rate: {}", if r.fp32_accumulate_unlocked() { "unlocked (full rate)" } else { "stock GeForce (half rate)" });
}
