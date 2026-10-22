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
    let mode = a.next();
    let gpu = mode.as_deref() == Some("gpu");
    // "assist": the processor runs the model and the GPU multiplies the matrices (GPU assist).
    let assist = mode.as_deref() == Some("assist");
    // "portable": the Vulkan / DX12 / Metal backend (CEREBRO_PORTABLE_GPU / _BACKEND pick the device).
    let portable = mode.as_deref() == Some("portable");
    let model = Model::load(std::path::Path::new(&path)).expect("load");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ids = tok.encode(&prompt);
    #[cfg(feature = "cuda")]
    let mut s = if gpu {
        Session::new_gpu(&model, 4096).expect("gpu session")
    } else if assist {
        Session::new_assisted(&model, 4096).expect("assisted session")
    } else {
        Session::new(&model, 4096)
    };
    #[cfg(feature = "cuda")]
    if let Some(g) = s.gpu.as_ref() {
        let p = &g.plan;
        eprintln!("placement: {:?}, {}/{} GPU layers, {:.2} GiB VRAM, estimate {:.1} tok/s", p.layout, p.gpu_layers, p.total_layers, p.vram_bytes as f64 / (1u64 << 30) as f64, p.est_decode_tps);
        for n in &p.notes {
            eprintln!("  {n}");
        }
    }
    #[cfg(not(feature = "cuda"))]
    let mut s = {
        let _ = gpu;
        Session::new(&model, 4096)
    };
    #[cfg(feature = "portable")]
    if portable {
        s = Session::new_portable(&model, 4096).expect("portable session");
        let i = &s.portable.as_ref().expect("portable").info;
        let p = s.portable.as_ref().expect("portable");
        eprintln!("portable GPU: {} ({}, {}), {:.2} GiB on the GPU; kernels: {}", i.name, i.backend, i.kind, p.bytes as f64 / (1u64 << 30) as f64, p.kernels);
    }
    #[cfg(not(feature = "portable"))]
    let _ = portable;
    let t = std::time::Instant::now();
    // PROGRESS=1: sample the live prompt-reading progress every 100 ms while the prompt is read.
    let watch = std::env::var_os("PROGRESS").map(|_| {
        let p = s.progress.clone();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d = done.clone();
        (done, std::thread::spawn(move || {
            while !d.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let (pos, base, tps) = p.read();
                eprintln!("progress: {pos} in cache ({base} reused), {}", tps.map_or("-".into(), |t| format!("{t:.0} tok/s")));
            }
        }))
    });
    // A cache warming from the saved profile competes for the drive: time the steady state.
    while model.gguf().experts().is_some_and(|c| !c.warmed()) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    s.feed(&model, &ids).expect("feed");
    if let Some((done, h)) = watch {
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = h.join();
    }
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
    if std::env::var_os("PROFILE").is_some() {
        let total: f64 = s.profile().iter().map(|x| x.1).sum();
        for (name, secs) in s.profile() {
            eprintln!("  {name:<28} {secs:7.2} s  {:4.1}%", 100.0 * secs / total.max(1e-9));
        }
        #[cfg(feature = "cuda")]
        if let Some((placed, cands, bytes, calls)) = model.offload_stats() {
            eprintln!("GPU assist: {placed}/{cands} matrices, {:.2} GiB, {calls} products", bytes as f64 / (1u64 << 30) as f64);
        }
    }
    model.save_expert_profile();
    if let Some(c) = model.gguf().experts() {
        let x = c.stats();
        eprintln!(
            "expert cache: {} slots, {} hits, {} misses ({:.0}% hit), {:.2} GB read from the drive in {:.2} s",
            x.slots,
            x.hits,
            x.misses,
            100.0 * x.hits as f64 / (x.hits + x.misses).max(1) as f64,
            x.bytes_read as f64 / 1e9,
            x.read_s
        );
    }
    if let Some(st) = model.gguf().stream() {
        let x = st.stats();
        eprintln!(
            "drive tier: {} layers, {:.2} GB read ({:.2} GB/s while reading), waited {:.2} s, {} read errors, {} MB of buffers",
            model.streamed_layers(),
            x.bytes_read as f64 / 1e9,
            x.bytes_read as f64 / 1e9 / x.read_s.max(1e-9),
            x.wait_s,
            x.errors,
            st.buffer_bytes() >> 20
        );
    }
}
