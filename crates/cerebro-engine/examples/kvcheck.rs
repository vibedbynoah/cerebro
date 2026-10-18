//! Saved-cache round trip: a session restored from a saved prompt cache, fed a longer prompt, must
//! reuse the saved tokens and produce the same greedy text as a fresh session.
//! cargo run --release --example kvcheck -- MODEL.gguf [cpu|gpu|split:N]
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::argmax;
use cerebro_engine::tokenizer::Tokenizer;

fn session(m: &Model, mode: &str) -> Session {
    match mode {
        "cpu" => Session::new(m, 4096),
        "gpu" => Session::new_gpu(m, 4096).expect("gpu"),
        s => Session::new_gpu_with(m, 4096, Some(s.trim_start_matches("split:").parse().expect("split:N"))).expect("split"),
    }
}

fn greedy(m: &Model, s: &mut Session, n: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..n {
        let t = argmax(&s.logits);
        out.push(t);
        s.step(m, t);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Model::load(std::path::Path::new(&a[1])).expect("model");
    let mode = a.get(2).map_or("gpu", String::as_str);
    let tok = Tokenizer::from_gguf(m.gguf()).expect("tokenizer");
    let p1 = tok.encode("The history of the lighthouse begins in ancient Egypt, where the Pharos of Alexandria stood for centuries.");
    let mut p2 = p1.clone();
    p2.extend(tok.encode(" Describe how it was built."));
    let mut fresh = session(&m, mode);
    fresh.feed(&m, &p2).expect("feed");
    let want = greedy(&m, &mut fresh, 24);

    let mut first = session(&m, mode);
    first.feed(&m, &p1).expect("feed");
    let mut buf = Vec::new();
    first.save_kv(&m, &mut buf).expect("save");
    drop(first);
    let mut restored = session(&m, mode);
    let n = restored.load_kv(&m, &mut buf.as_slice()).expect("load");
    let reused = restored.feed(&m, &p2).expect("feed");
    let got = greedy(&m, &mut restored, 24);
    // Control: the same two-step reading (p1, then the rest) without saving: isolates batch-shape
    // numerics from the save/restore itself.
    let mut two = session(&m, mode);
    two.feed(&m, &p1).expect("feed");
    two.feed(&m, &p2).expect("feed");
    let ctl = greedy(&m, &mut two, 24);
    println!("{mode}: two-step reading without a save gives the restored text: {}", ctl == got);
    println!("{mode}: saved {n} tokens ({:.1} MB), reused {reused}, same text: {}", buf.len() as f64 / 1e6, got == want);
    if got != want {
        println!("  want {want:?}\n  got  {got:?}");
    }
}
