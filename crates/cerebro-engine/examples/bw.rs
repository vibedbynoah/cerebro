use cerebro_engine::gguf::Gguf;
use cerebro_engine::quant::{dot_q8_0_q8, QAct};
use rayon::prelude::*;
use std::time::Instant;
fn main() {
    let p = std::env::args().nth(1).expect("path");
    let g = Gguf::open(std::path::Path::new(&p)).expect("open");
    let t = g.tensors.get("token_embd.weight").unwrap().clone();
    let (rows, cols) = (t.rows(), t.cols());
    let rb = cols / 32 * 34;
    let bytes = g.bytes(&t);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut qa = QAct::new(cols);
    qa.quantize(&x);
    let mut out = vec![0f32; rows];
    for chunk in [64usize, 256, 1024, 4096] {
        for _ in 0..2 { out.par_chunks_mut(chunk).enumerate().for_each(|(ci, ch)| for (j, o) in ch.iter_mut().enumerate() { let r = ci*chunk+j; *o = dot_q8_0_q8(&bytes[r*rb..(r+1)*rb], &qa); }); }
        let n = 20;
        let t0 = Instant::now();
        for _ in 0..n { out.par_chunks_mut(chunk).enumerate().for_each(|(ci, ch)| for (j, o) in ch.iter_mut().enumerate() { let r = ci*chunk+j; *o = dot_q8_0_q8(&bytes[r*rb..(r+1)*rb], &qa); }); }
        let s = t0.elapsed().as_secs_f64() / n as f64;
        println!("chunk {chunk}: {:.2} ms, {:.1} GB/s ({} threads)", s * 1e3, bytes.len() as f64 / s / 1e9, rayon::current_num_threads());
    }
    // single thread
    let t0 = Instant::now();
    for r in 0..rows { out[r] = dot_q8_0_q8(&bytes[r*rb..(r+1)*rb], &qa); }
    let s = t0.elapsed().as_secs_f64();
    println!("1 thread: {:.2} ms, {:.1} GB/s", s * 1e3, bytes.len() as f64 / s / 1e9);
}
