//! Greedy text from a raw prompt (no chat template), for checking a new architecture against a
//! reference implementation token for token.
//! cargo run --release --example greedy -- MODEL.gguf "The capital of France is" [tokens] [cpu|gpu]

use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::argmax;
use cerebro_engine::tokenizer::Tokenizer;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("model path");
    let prompt = a.next().unwrap_or_else(|| "The capital of France is".into());
    let n: usize = a.next().and_then(|v| v.parse().ok()).unwrap_or(32);
    let gpu = a.next().as_deref() == Some("gpu");
    let model = Model::load(std::path::Path::new(&path)).expect("load");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ids = tok.encode(&prompt);
    #[cfg(feature = "cuda")]
    let mut s = if gpu { Session::new_gpu(&model, 4096).expect("gpu session") } else { Session::new(&model, 4096) };
    #[cfg(not(feature = "cuda"))]
    let mut s = {
        let _ = gpu;
        Session::new(&model, 4096)
    };
    let t = std::time::Instant::now();
    s.feed(&model, &ids).expect("feed");
    let prefill = t.elapsed().as_secs_f64();
    // TOP=1: the five most likely next tokens after the prompt, as log-probabilities.
    if std::env::var_os("TOP").is_some() {
        let mx = s.logits.iter().fold(f32::MIN, |m, &v| m.max(v));
        let lse = mx + s.logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
        let mut idx: Vec<usize> = (0..s.logits.len()).collect();
        idx.sort_by(|&a, &b| s.logits[b].total_cmp(&s.logits[a]));
        for &i in &idx[..5] {
            println!("top {:>6} {:?} logprob {:.4}", i, tok.decode(&[i as u32]), s.logits[i] - lse);
        }
    }
    let mut out = Vec::new();
    let t = std::time::Instant::now();
    for _ in 0..n {
        let next = argmax(&s.logits) as u32;
        out.push(next);
        s.step(&model, next);
    }
    let decode = t.elapsed().as_secs_f64();
    println!("prompt tokens {:?}", ids);
    println!("output tokens {:?}", out);
    println!("{}{}", prompt, tok.decode(&out));
    eprintln!("prefill {:.1} tok/s, decode {:.1} tok/s", ids.len() as f64 / prefill, n as f64 / decode);
}
