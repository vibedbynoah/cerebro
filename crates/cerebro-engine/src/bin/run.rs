//! `cerebro-run --model <gguf> --prompt "..." [--n 128] [--temp 0.7] [--ctx 4096] [--think] [--threads N]`
//! Loads a model with Cerebro Engine, generates a reply, and prints prefill and decode speed.
//! `--again "..."` sends a second turn to show the conversation cache reusing the first.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use cerebro_engine::chatml;
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::{Sampler, SamplerConfig};
use cerebro_engine::tokenizer::Tokenizer;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn die(msg: String) -> ! {
    eprintln!("{msg}");
    std::process::exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(arg(&args, "--model").unwrap_or_else(|| die("usage: cerebro-run --model <file.gguf> --prompt \"...\"".into())));
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| "Explain what a mixture-of-experts model is in two sentences.".into());
    let n: usize = arg(&args, "--n").and_then(|v| v.parse().ok()).unwrap_or(128);
    let temp: f32 = arg(&args, "--temp").and_then(|v| v.parse().ok()).unwrap_or(0.7);
    let ctx: usize = arg(&args, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let think = args.iter().any(|a| a == "--think");
    if let Some(t) = arg(&args, "--threads").and_then(|v| v.parse::<usize>().ok()) {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
    }

    let t0 = Instant::now();
    let model = Model::load(&model_path).unwrap_or_else(|e| die(format!("load failed: {e}")));
    let tok = Tokenizer::from_gguf(model.gguf()).unwrap_or_else(|e| die(format!("tokenizer: {e}")));
    let c = &model.cfg;
    eprintln!(
        "loaded {} ({} layers, dim {}, {} heads / {} kv, vocab {}) in {:.2}s, {} threads",
        c.arch,
        c.n_layer,
        c.dim,
        c.n_head,
        c.n_kv,
        c.vocab,
        t0.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    let use_gpu = !args.iter().any(|a| a == "--cpu");
    let mut session = if use_gpu {
        let t = Instant::now();
        match Session::new_gpu(&model, ctx) {
            Ok(s) => {
                eprintln!("GPU: weights and KV cache on the GPU ({:.0} MB) in {:.2}s", s.gpu.as_ref().map_or(0, |g| g.vram_bytes) as f64 / 1e6, t.elapsed().as_secs_f64());
                s
            }
            Err(e) => {
                eprintln!("GPU unavailable ({e}); running on the CPU");
                Session::new(&model, ctx)
            }
        }
    } else {
        Session::new(&model, ctx)
    };
    let mut sampler = Sampler::new(SamplerConfig { temperature: temp, ..Default::default() });

    let mut history: Vec<(String, String)> = Vec::new();
    let mut prompts = vec![prompt];
    if let Some(again) = arg(&args, "--again") {
        prompts.push(again);
    }
    for p in prompts {
        let turns: Vec<(&str, &str)> = history.iter().flat_map(|(u, a)| [("user", u.as_str()), ("assistant", a.as_str())]).chain([("user", p.as_str())]).collect();
        // --raw: the prompt is already in the model's own chat format (Gemma 4: <|turn>user ...).
        let ids = if args.iter().any(|a| a == "--raw") { tok.encode(&p) } else { tok.encode(&chatml(None, &turns, think)) };
        let reused = session.feed(&model, &ids).unwrap_or_else(|e| die(e.to_string()));
        let mut out = Vec::new();
        let td = Instant::now();
        let mut printed = 0usize;
        // Greedy on the GPU: the whole loop stays on the device (graph replay, pick on the GPU).
        #[cfg(feature = "cuda")]
        if temp <= 0.0 {
            if let Some(g) = session.gpu.as_mut() {
                let first = cerebro_engine::sample::argmax(&session.logits);
                if !tok.eot.contains(&first) {
                    out.push(first);
                    let pos = session.tokens.len();
                    g.generate_greedy(&model, first, pos, n.saturating_sub(1), &tok.eot, |t| {
                        out.push(t);
                        true
                    }).unwrap_or_else(|e| die(e.to_string()));
                }
                let secs = td.elapsed().as_secs_f64();
                println!("{}", tok.decode(&out));
                eprintln!(
                    "prompt {} tokens ({} reused), prefill {:.1} tok/s; generated {} tokens at {:.1} tok/s (GPU greedy)",
                    ids.len(),
                    reused,
                    session.stats.prefill_tps(),
                    out.len(),
                    out.len() as f64 / secs
                );
                history.push((p, tok.decode(&out)));
                continue;
            }
        }
        for _ in 0..n {
            let next = sampler.sample(&session.logits, &session.tokens);
            if tok.eot.contains(&next) {
                break;
            }
            out.push(next);
            let bytes = tok.decode_bytes(&out);
            if let Ok(s) = std::str::from_utf8(&bytes[printed..]) {
                print!("{s}");
                std::io::stdout().flush().ok();
                printed = bytes.len();
            }
            session.step(&model, next);
        }
        let secs = td.elapsed().as_secs_f64();
        println!();
        eprintln!(
            "prompt {} tokens ({} reused from the conversation cache), prefill {:.1} tok/s; generated {} tokens at {:.1} tok/s",
            ids.len(),
            reused,
            session.stats.prefill_tps(),
            out.len(),
            out.len() as f64 / secs
        );
        history.push((p, tok.decode(&out)));
    }
    if std::env::var_os("CEREBRO_PROFILE").is_some() {
        let names = ["q/k/v", "rope+cache", "attention", "o", "router", "ffn/experts", "output"];
        let parts: Vec<String> = names.iter().zip(session.prof).map(|(n, t)| format!("{n} {t:.2}s")).collect();
        eprintln!("CPU phases: {}", parts.join(", "));
    }
    #[cfg(feature = "cuda")]
    if let Some(g) = session.gpu.as_ref() {
        if let Some((cached, total, hit)) = g.expert_cache() {
            eprintln!("experts: {cached} of {total} cached in VRAM, {:.0}% of expert uses served from VRAM", hit * 100.0);
            if let Some((w, c)) = g.moe_times() {
                eprintln!("time waiting for GPU routing {w:.2}s, CPU experts {c:.2}s");
            }
            g.save_expert_profile();
        }
    }
}
