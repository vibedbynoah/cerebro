//! Where CPU decode time goes at a given context depth: per-token milliseconds by layer section.
//! cargo run --release --example prof -- MODEL.gguf DEPTH [TOKENS]
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::argmax;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Model::load(std::path::Path::new(&a[1])).expect("model");
    let depth: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(512);
    let n: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(64);
    let tok = cerebro_engine::tokenizer::Tokenizer::from_gguf(m.gguf()).expect("tokenizer");
    // PROF_TEXT=ocean: the bench example's prompt text instead.
    let line = if std::env::var("PROF_TEXT").as_deref() == Ok("ocean") {
        "The ocean covers most of the planet and holds most of its water. Currents move heat from the equator toward the poles, shaping weather on every continent. "
    } else {
        "The lighthouse keeper climbed the stairs at dusk and lit the great lamp. "
    };
    let text = line.repeat(depth / 8 + 1);
    let mut ids = tok.encode(&text);
    ids.truncate(depth.max(1));
    let mut s = Session::new(&m, depth + n + 8);
    // PROF_REUSE=1: read and write once first, clear, then time (as the bench example does).
    if std::env::var_os("PROF_REUSE").is_some() {
        s.feed(&m, &ids).expect("feed");
        for _ in 0..16 {
            let t = argmax(&s.logits);
            s.step(&m, t);
        }
        s.tokens.clear();
        // PROF_GAP=seconds: pause before the timed read (boost budget vs session state).
        if let Some(r) = std::env::var("PROF_GAP").ok().and_then(|v| v.parse::<u64>().ok()) {
            std::thread::sleep(std::time::Duration::from_secs(r));
        }
    }
    s.prof = [0.0; 7];
    let tf = std::time::Instant::now();
    s.feed(&m, &ids).expect("feed");
    let feed = tf.elapsed().as_secs_f64();
    let names = ["qkv matmul", "norm+rope+kv", "attention", "o matmul", "router", "ffn", "head"];
    println!("prompt {} tokens: {:.1} tok/s ({:.2} s)", ids.len(), ids.len() as f64 / feed, feed);
    for (i, name) in names.iter().enumerate() {
        if s.prof[i] > 0.0 {
            println!("  {:14} {:6.3} s ({:4.1}%)", name, s.prof[i], 100.0 * s.prof[i] / feed);
        }
    }
    // PROF_REST=seconds: let the processor cool after reading the prompt (heat vs context).
    if let Some(r) = std::env::var("PROF_REST").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::sleep(std::time::Duration::from_secs(r));
    }
    s.prof = [0.0; 7];
    let mut t = argmax(&s.logits);
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        s.step(&m, t);
        t = argmax(&s.logits);
    }
    let total = t0.elapsed().as_secs_f64();
    let names = ["qkv matvec", "norm+rope+kv", "attention", "o matvec", "router", "ffn", "head"];
    println!("depth {depth}: {:.1} tok/s, {:.2} ms/token", n as f64 / total, total * 1e3 / n as f64);
    for (i, name) in names.iter().enumerate() {
        println!("  {:14} {:6.3} ms", name, s.prof[i] * 1e3 / n as f64);
    }
}
