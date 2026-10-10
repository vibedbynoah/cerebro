//! CPU expert throughput: Model::moe_cpu on `k` experts of every layer in turn, as in decoding.
//! cargo run --release --example moebench -- model.gguf [k]
use cerebro_engine::model::{Ffn, Model, MoeScratch};
use cerebro_engine::quant::QAct;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Model::load(std::path::Path::new(&a[1])).expect("load");
    let k: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    let c = &m.cfg;
    let h: Vec<f32> = (0..c.dim).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut qa = QAct::new(c.dim);
    qa.quantize(&h);
    let mut s = MoeScratch::new(c);
    let mut out = vec![0f32; c.dim];
    let Ffn::Moe(e) = &m.layers()[0].ffn else { panic!("not MoE") };
    let per_expert = (e.gate.rows * e.gate.row_bytes + e.up.rows * e.up.row_bytes + e.down.rows * e.down.row_bytes) / c.n_expert;
    let gap_us: u64 = std::env::var("MOEBENCH_GAP_US").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for round in 0..3 {
        let t = Instant::now();
        let mut n = 0;
        for step in 0..20 {
            for li in 0..c.n_layer {
                let ex: Vec<(usize, f32)> = (0..k).map(|j| ((step * 7 + li * 13 + j * 17) % c.n_expert, 1.0 / k as f32)).collect();
                m.moe_cpu(li, &h, &qa, &ex, &mut s, &mut out);
                n += k;
                if gap_us > 0 {
                    // Stand-in for the GPU's share of a layer: the main thread waits, workers idle.
                    let t = Instant::now();
                    while t.elapsed().as_micros() < gap_us as u128 {
                        std::hint::spin_loop();
                    }
                }
            }
        }
        let secs = t.elapsed().as_secs_f64() - (20 * c.n_layer) as f64 * gap_us as f64 * 1e-6;
        println!("round {round}: {:.1} us per layer call, {:.1} GB/s ({} threads)", secs / (20 * c.n_layer) as f64 * 1e6, (n * per_expert) as f64 / secs / 1e9, cerebro_engine::pool::global().threads());
    }
}
