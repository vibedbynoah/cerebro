//! Kernel bandwidth: one weight tensor times int8 activations, all threads and one thread.
//! cargo run --release --example bw -- model.gguf [tensor name]
use cerebro_engine::gguf::Gguf;
use cerebro_engine::quant::{dot_k_q8, dot_q8_0_q8, DType, QAct};
use rayon::prelude::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1])).expect("open");
    let name = a.get(2).cloned().unwrap_or_else(|| "token_embd.weight".into());
    let t = g.tensors.get(&name).expect("tensor").clone();
    let (rows, cols, dt) = (t.rows(), t.cols(), t.dtype);
    let rb = dt.row_bytes(cols);
    let bytes = g.bytes(&t);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut qa = QAct::new(cols);
    qa.quantize(&x);
    let dot = |r: usize| if dt == DType::Q8_0 { dot_q8_0_q8(&bytes[r * rb..(r + 1) * rb], &qa) } else { dot_k_q8(dt, &bytes[r * rb..(r + 1) * rb], &qa) };
    let mut out = vec![0f32; rows];
    println!("{name}: {dt:?} {rows}x{cols}, {:.1} MB", bytes.len() as f64 / 1e6);
    for chunk in [64usize, 256, 1024] {
        for _ in 0..2 {
            out.par_chunks_mut(chunk).enumerate().for_each(|(ci, ch)| for (j, o) in ch.iter_mut().enumerate() { *o = dot(ci * chunk + j); });
        }
        let n = 10;
        let t0 = Instant::now();
        for _ in 0..n {
            out.par_chunks_mut(chunk).enumerate().for_each(|(ci, ch)| for (j, o) in ch.iter_mut().enumerate() { *o = dot(ci * chunk + j); });
        }
        let s = t0.elapsed().as_secs_f64() / n as f64;
        println!("chunk {chunk}: {:.2} ms, {:.1} GB/s ({} threads)", s * 1e3, bytes.len() as f64 / s / 1e9, rayon::current_num_threads());
    }
    let t0 = Instant::now();
    for r in 0..rows { out[r] = dot(r); }
    let s = t0.elapsed().as_secs_f64();
    println!("1 thread: {:.2} ms, {:.1} GB/s", s * 1e3, bytes.len() as f64 / s / 1e9);
}
