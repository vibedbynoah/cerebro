//! Perplexity on a text file, scored like llama.cpp's `llama-perplexity`: the text is cut into
//! chunks of `ctx` tokens, each chunk is read from a fresh cache, and only the second half of each
//! chunk is scored (the first half is context). Comparing the number with the reference on the same
//! file and settings checks a model port without the noise of free generation.
//! cargo run --release --example ppl -- MODEL.gguf TEXT.txt [ctx=256] [chunks=2] [cpu|gpu]

use cerebro_engine::model::{Model, Session};
use cerebro_engine::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Model::load(std::path::Path::new(&a[1])).expect("load");
    let text = std::fs::read_to_string(&a[2]).expect("text");
    let ctx: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(256);
    let chunks: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(2);
    let gpu = a.get(5).map(String::as_str) == Some("gpu");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ids = tok.encode(&text);
    let chunks = chunks.min(ids.len() / ctx);
    #[cfg(feature = "cuda")]
    let mut s = if gpu { Session::new_gpu(&model, ctx + 8).expect("gpu session") } else { Session::new(&model, ctx + 8) };
    #[cfg(not(feature = "cuda"))]
    let mut s = {
        let _ = gpu;
        Session::new(&model, ctx + 8)
    };
    let (mut nll, mut count) = (0f64, 0usize);
    let t = std::time::Instant::now();
    for c in 0..chunks {
        s.clear().expect("clear");
        let chunk = &ids[c * ctx..(c + 1) * ctx];
        for (j, &id) in chunk.iter().enumerate() {
            s.step(&model, id);
            // The logits after token j predict token j + 1; score the second half.
            if j >= ctx / 2 && j + 1 < ctx {
                let l = &s.logits;
                let mx = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                let lse = mx as f64 + l.iter().map(|&v| ((v - mx) as f64).exp()).sum::<f64>().ln();
                nll += lse - l[chunk[j + 1] as usize] as f64;
                count += 1;
            }
        }
        println!("[{}] {:.4}", c + 1, (nll / count as f64).exp());
    }
    println!("ppl {:.4} over {count} tokens, {chunks} chunks of {ctx}, {:.0} s", (nll / count.max(1) as f64).exp(), t.elapsed().as_secs_f64());
}
