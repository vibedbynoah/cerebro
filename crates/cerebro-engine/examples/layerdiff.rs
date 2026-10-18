//! GPU vs CPU residual after the first N layers of a prompt (split-mode GPU prefill against the CPU):
//! cargo run --release --example layerdiff -- MODEL.gguf "prompt" N1,N2,...
use cerebro_engine::model::{Model, Session};
use cerebro_engine::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Model::load(std::path::Path::new(&a[1])).expect("model");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ids = tok.encode(&a[2]);
    let dim = model.cfg.dim;
    for n in a[3].split(',').map(|v| v.parse::<usize>().unwrap()) {
        let mut cpu = Session::new(&model, 4096);
        let want = cpu.residual_after(&model, &ids, n);
        // "full": the whole-GPU session, its batched prefill stopped after n layers; else a split.
        let full = std::env::var_os("FULL").is_some();
        let mut s = Session::new_gpu_with(&model, 4096, if full { None } else { Some(n) }).expect("gpu");
        let g = s.gpu.as_mut().expect("gpu state");
        g.debug_stop = full.then_some(n);
        let mut xs = Vec::new();
        if full {
            let mut logits = vec![0f32; model.cfg.vocab];
            g.prefill(&model, &ids, 0, Some(&mut logits)).expect("prefill");
            xs = g.debug_batch_x(ids.len()).expect("x");
        } else {
            g.prefill_split(&model, &ids, 0, &mut xs).expect("prefill");
        }
        let mut worst = 0f32;
        let mut per = Vec::new();
        for t in 0..ids.len() {
            let (w, x) = (&want[t * dim..(t + 1) * dim], &xs[t * dim..(t + 1) * dim]);
            let norm = w.iter().map(|v| v * v).sum::<f32>().sqrt();
            let err = w.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            worst = worst.max(err / norm.max(1e-6));
            per.push(format!("{:.3}", err / norm.max(1e-6)));
        }
        println!("layers {n:>3}: worst relative error over {} tokens {worst:.5}  per token [{}]", ids.len(), per.join(" "));
        if std::env::var_os("DUMP").is_some() {
            let t = 1;
            let (w, x) = (&want[t * dim..(t + 1) * dim], &xs[t * dim..(t + 1) * dim]);
            let mut idx: Vec<usize> = (0..dim).collect();
            idx.sort_by(|&a, &b| (w[b] - x[b]).abs().partial_cmp(&(w[a] - x[a]).abs()).unwrap());
            for &i in idx.iter().take(6) {
                println!("   dim {i}: cpu {:.4} gpu {:.4}", w[i], x[i]);
            }
            let big: Vec<usize> = (0..dim).filter(|&i| w[i].abs() > 20.0).collect();
            println!("   cpu dims with |v| > 20: {:?}", big.iter().map(|&i| (i, w[i] as i32, x[i] as i32)).collect::<Vec<_>>());
        }
    }
}
