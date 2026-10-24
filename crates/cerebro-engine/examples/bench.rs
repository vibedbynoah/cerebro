//! Engine benchmark harness. For each backend and prompt length: one warm-up run, then `--reps` timed
//! runs (median reported) of prefill and greedy decode, plus a correctness check: the GPU's greedy
//! tokens are compared with the CPU reference. Appends one JSON line per configuration to `--out`.
//!
//! cargo run --release --example bench -- --model M.gguf --out runs.jsonl --machine "i9-12900H + RTX 3070 Ti"
//!   [--reps 5] [--decode 256] [--lengths 32,512,2048] [--backends cpu,gpu,portable] [--commit HASH]
//!   [--gpu-frac F] [--drive-frac F] [--label NAME]
//! --gpu-frac / --drive-frac place that fraction of the model's layers on the graphics card / read
//! them from the drive (the "Layers on the graphics card" / "Layers read from the drive" switches),
//! so one placement description works for any model. `portable` is the Vulkan / DX12 / Metal backend
//! (CEREBRO_PORTABLE_GPU picks the device, e.g. an integrated GPU by name).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::argmax;
use cerebro_engine::tokenizer::Tokenizer;

fn arg(a: &[String], n: &str) -> Option<String> {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

const TEXT: &str = "The ocean covers most of the planet and holds most of its water. Currents move heat from the equator toward the poles, shaping weather on every continent. ";

fn prompt_of(tok: &Tokenizer, len: usize) -> Vec<u32> {
    let mut s = String::from("<|im_start|>user\nSummarise the following text.\n");
    let mut ids = tok.encode(&s);
    while ids.len() < len.saturating_sub(8) {
        s.push_str(TEXT);
        ids = tok.encode(&s);
    }
    s.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let mut ids = tok.encode(&s);
    ids.truncate(len.max(8));
    ids
}

/// Greedy decode of `n` tokens after `prompt`; returns (prefill s, decode s, tokens).
fn run(model: &Model, sess: &mut Session, prompt: &[u32], n: usize, gpu: bool) -> (f64, f64, Vec<u32>) {
    // A fresh conversation each run, recurrent (DeltaNet) state included.
    sess.clear().expect("clear");
    let t0 = Instant::now();
    sess.feed(model, prompt).expect("feed");
    let pre = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut out = Vec::with_capacity(n);
    let first = argmax(&sess.logits);
    out.push(first);
    #[cfg(feature = "cuda")]
    if gpu && sess.gpu.as_ref().is_some_and(|g| !g.is_split()) {
        let pos = sess.tokens.len();
        let g = sess.gpu.as_mut().expect("gpu");
        g.generate_greedy(model, first, pos, n - 1, &[], |t| {
                        out.push(t);
                        true
                    }).expect("decode");
        return (pre, t1.elapsed().as_secs_f64(), out);
    }
    let _ = gpu;
    for _ in 1..n {
        let last = *out.last().unwrap();
        sess.step(model, last);
        out.push(argmax(&sess.logits));
    }
    (pre, t1.elapsed().as_secs_f64(), out)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(arg(&a, "--model").expect("--model"));
    // --print-layers: the model's layer count, for scripts that place fractions of it elsewhere.
    if a.iter().any(|x| x == "--print-layers" || x == "--print-experts") {
        let g = cerebro_engine::gguf::Gguf::open(&model_path).expect("open");
        let arch = g.str("general.architecture").unwrap_or("").to_owned();
        let key = if a.iter().any(|x| x == "--print-experts") { "expert_count" } else { "block_count" };
        println!("{}", g.u64(&format!("{arch}.{key}")).unwrap_or(0));
        return;
    }
    let out_path = PathBuf::from(arg(&a, "--out").unwrap_or_else(|| "bench.jsonl".into()));
    let machine = arg(&a, "--machine").unwrap_or_else(|| "unknown".into());
    let commit = arg(&a, "--commit").unwrap_or_default();
    // A name for the placement being measured (GPU only, CPU + drive, ...).
    let label = arg(&a, "--label").unwrap_or_default();
    let reps: usize = arg(&a, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let decode: usize = arg(&a, "--decode").and_then(|v| v.parse().ok()).unwrap_or(256);
    let lengths: Vec<usize> = arg(&a, "--lengths").unwrap_or_else(|| "32,512,2048".into()).split(',').filter_map(|v| v.parse().ok()).collect();
    let backends: Vec<String> = arg(&a, "--backends").unwrap_or_else(|| "cpu,gpu".into()).split(',').map(String::from).collect();
    // Fractions of the model's layers -> the layer-count switches, before the model loads.
    let frac = |n: &str| arg(&a, n).and_then(|v| v.parse::<f64>().ok()).filter(|f| (0.0..=1.0).contains(f));
    if frac("--gpu-frac").is_some() || frac("--drive-frac").is_some() {
        let g = cerebro_engine::gguf::Gguf::open(&model_path).expect("open");
        let arch = g.str("general.architecture").unwrap_or("").to_owned();
        let n = g.u64(&format!("{arch}.block_count")).expect("block count") as f64;
        if let Some(f) = frac("--gpu-frac") {
            std::env::set_var("CEREBRO_GPU_LAYERS", ((f * n).round() as u64).max(1).to_string());
        }
        if let Some(f) = frac("--drive-frac") {
            std::env::set_var("CEREBRO_STREAM_LAYERS", ((f * n).round() as u64).max(1).to_string());
        }
    }

    let model = Model::load(&model_path).expect("load");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ctx = lengths.iter().max().copied().unwrap_or(512) + decode + 16;
    // Weight bytes one token reads: everything, except that an MoE layer reads only n_used of its
    // n_expert experts.
    let bytes_per_token: usize = model
        .gguf()
        .tensors
        .values()
        .map(|t| if t.name.contains("_exps") && model.cfg.n_expert > 0 { t.size * model.cfg.n_used / model.cfg.n_expert } else { t.size })
        .sum();
    let mut reference: Vec<Vec<u32>> = Vec::new();
    let mut f = OpenOptions::new().create(true).append(true).open(&out_path).expect("open out");

    for be in &backends {
        let gpu = be == "gpu";
        #[cfg(feature = "portable")]
        let portable = be == "portable";
        #[cfg(not(feature = "portable"))]
        let portable = false;
        let mut sess = if portable {
            #[cfg(feature = "portable")]
            {
                Session::new_portable(&model, ctx).expect("portable session")
            }
            #[cfg(not(feature = "portable"))]
            {
                panic!("built without the portable backend")
            }
        } else if gpu {
            // As the app does: the whole-GPU path, else the processor with GPU assist (architectures
            // the whole-GPU path does not run yet, such as Flash-Next).
            #[cfg(feature = "cuda")]
            {
                Session::new_gpu(&model, ctx).or_else(|_| Session::new_assisted(&model, ctx)).expect("gpu session")
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("built without CUDA")
            }
        } else {
            Session::new(&model, ctx)
        };
        while model.gguf().experts().is_some_and(|c| !c.warmed()) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        for (li, &len) in lengths.iter().enumerate() {
            let p = prompt_of(&tok, len);
            let _ = run(&model, &mut sess, &p, 16.min(decode), gpu); // warm-up
            let (mut pre, mut dec) = (Vec::new(), Vec::new());
            let mut toks = Vec::new();
            for _ in 0..reps {
                let (a, b, t) = run(&model, &mut sess, &p, decode, gpu);
                pre.push(p.len() as f64 / a);
                dec.push((decode as f64) / b);
                toks = t;
            }
            let agree = if be == "cpu" {
                reference.push(toks.clone());
                1.0
            } else if let Some(r) = reference.get(li) {
                let n = r.len().min(toks.len()).min(64);
                let same = r.iter().zip(&toks).take(n).take_while(|(x, y)| x == y).count();
                same as f64 / n as f64
            } else {
                f64::NAN
            };
            let dtps = median(dec);
            #[cfg(feature = "cuda")]
            let cache = sess.gpu.as_ref().and_then(|g| g.expert_cache());
            #[cfg(not(feature = "cuda"))]
            let cache: Option<(usize, usize, f64)> = None;
            #[cfg(feature = "cuda")]
            let gpu_layers = sess.gpu.as_ref().map(|g| g.gpu_layers()).unwrap_or(0);
            #[cfg(not(feature = "cuda"))]
            let gpu_layers = 0usize;
            #[cfg(feature = "portable")]
            let (gpu_layers, device) = match sess.portable.as_ref() {
                Some(p) => (p.gpu_layers(), format!("{} ({}, {})", p.info.name, p.info.backend, p.info.kind)),
                None => (gpu_layers, String::new()),
            };
            #[cfg(not(feature = "portable"))]
            let device = String::new();
            let drive = model.gguf().stream().map(|s| s.stats());
            let line = serde_json::json!({
                "label": label,
                "device": device,
                "gpu_layers": gpu_layers,
                "drive_layers": model.streamed_layers(),
                "total_layers": model.cfg.n_layer,
                "drive_gb_read": drive.map(|d| d.bytes_read as f64 / 1e9),
                "drive_wait_s": drive.map(|d| d.wait_s),
                "drive_errors": drive.map(|d| d.errors),
                "commit": commit,
                "machine": machine,
                "model": model_path.file_name().unwrap().to_string_lossy(),
                "backend": be,
                "prompt_tokens": p.len(),
                "decode_tokens": decode,
                "reps": reps,
                "prefill_tps_median": median(pre),
                "decode_tps_median": dtps,
                "weight_bytes_per_token": bytes_per_token,
                "effective_gb_s": bytes_per_token as f64 * dtps / 1e9,
                "greedy_prefix_match_vs_cpu_64": agree,
                "threads": cerebro_engine::pool::global().threads(),
                "ctx": ctx,
                "experts_in_vram": cache.map(|c| c.0),
                "experts_total": cache.map(|c| c.1),
                "expert_vram_hit_rate": cache.map(|c| c.2),
            });
            println!("{line}");
            writeln!(f, "{line}").expect("write");
        }
    }
}
