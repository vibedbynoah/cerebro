//! Batched CPU prefill vs token-by-token: max logit difference and argmax agreement after each prompt.
//! cargo run --release --example batchcheck -- MODEL.gguf

use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::argmax;
use cerebro_engine::switches;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let model = Model::load(std::path::Path::new(&path)).expect("load");
    for n in [7usize, 33, 64, 100, 300] {
        let prompt: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let mut logits = Vec::new();
        let modes: Vec<i64> = std::env::var("MODES").map(|v| v.split(",").map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 0]);
        for &on in &modes {
            switches::set(switches::CPU_BATCH_PREFILL, on).unwrap();
            let mut s = Session::new(&model, 1024);
            s.feed(&model, &prompt).expect("feed");
            logits.push(s.logits.clone());
        }
        let d = logits[0].iter().zip(&logits[1]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let mx = logits[1].iter().fold(0f32, |m, v| m.max(v.abs()));
        println!("{n:4} tokens: max |diff| {d:.4} (max |logit| {mx:.1}), argmax {} vs {}", argmax(&logits[0]), argmax(&logits[1]));
    }
}
