//! Engine benchmark harness. For each backend and prompt length: one warm-up run, then `--reps` timed
//! runs (median reported) of prefill and greedy decode, plus a correctness check: the GPU's greedy
//! tokens are compared with the CPU reference. Appends one JSON line per configuration to `--out`.
//!
//! cargo run --release --example bench -- --model M.gguf --out runs.jsonl --machine "i9-12900H + RTX 3070 Ti"
//!   [--reps 5] [--decode 256] [--lengths 32,512,2048] [--backends cpu,gpu] [--commit HASH]

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
    sess.tokens.clear();
    let t0 = Instant::now();
    sess.feed(model, prompt).expect("feed");
    let pre = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut out = Vec::with_capacity(n);
    let first = argmax(&sess.logits);
    out.push(first);
    #[cfg(feature = "cuda")]
    if gpu {
        let pos = sess.tokens.len();
        let g = sess.gpu.as_mut().expect("gpu");
        g.generate_greedy(first, pos, n - 1, &[], |t| out.push(t)).expect("decode");
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
    let out_path = PathBuf::from(arg(&a, "--out").unwrap_or_else(|| "bench.jsonl".into()));
    let machine = arg(&a, "--machine").unwrap_or_else(|| "unknown".into());
    let commit = arg(&a, "--commit").unwrap_or_default();
    let reps: usize = arg(&a, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let decode: usize = arg(&a, "--decode").and_then(|v| v.parse().ok()).unwrap_or(256);
    let lengths: Vec<usize> = arg(&a, "--lengths").unwrap_or_else(|| "32,512,2048".into()).split(',').filter_map(|v| v.parse().ok()).collect();
    let backends: Vec<String> = arg(&a, "--backends").unwrap_or_else(|| "cpu,gpu".into()).split(',').map(String::from).collect();

    let model = Model::load(&model_path).expect("load");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ctx = lengths.iter().max().copied().unwrap_or(512) + decode + 16;
    let bytes_per_token: usize = model.gguf().tensors.values().map(|t| t.size).sum();
    let mut reference: Vec<Vec<u32>> = Vec::new();
    let mut f = OpenOptions::new().create(true).append(true).open(&out_path).expect("open out");

    for be in &backends {
        let gpu = be == "gpu";
        let mut sess = if gpu {
            #[cfg(feature = "cuda")]
            {
                Session::new_gpu(&model, ctx).expect("gpu session")
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("built without CUDA")
            }
        } else {
            Session::new(&model, ctx)
        };
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
            let line = serde_json::json!({
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
                "threads": rayon::current_num_threads(),
            });
            println!("{line}");
            writeln!(f, "{line}").expect("write");
        }
    }
}
