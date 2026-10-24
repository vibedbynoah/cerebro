//! Split prompt reading with and without streaming the processor's layers through the GPU: compares
//! every processor-side cache the two leave behind and the logits, layer by layer.
//! cargo run --release --example streamdiff -- MODEL.gguf GPU_LAYERS "prompt"
use cerebro_engine::model::{Model, Session};
use cerebro_engine::tokenizer::Tokenizer;

fn rel_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() / n
}

fn h(x: u16) -> f32 {
    half_to_f32(x)
}

fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    s * if e == 0 { m * 2f32.powi(-24) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Model::load(std::path::Path::new(&a[1])).expect("model");
    let n: usize = a[2].parse().expect("gpu layers");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let ids = tok.encode(&a[3]);
    let kvd = model.cfg.n_kv * model.cfg.head_dim;
    let run = |stream: bool| {
        cerebro_engine::switches::set(cerebro_engine::switches::SPLIT_STREAM, stream as i64).expect("switch");
        let mut s = Session::new_gpu_with(&model, 2048, Some(n)).expect("gpu");
        s.feed(&model, &ids).expect("feed");
        (s.debug_caches(), s.logits.clone())
    };
    let ((k0, v0, h0, s0), l0) = run(false);
    let ((k1, v1, h1, s1), l1) = run(true);
    println!("{} tokens, layers {}..{} on the processor", ids.len(), n, model.cfg.n_layer);
    for li in n..model.cfg.n_layer {
        let t = ids.len() * kvd;
        let kd = if k0[li].len() >= t { rel_f32(&k0[li][..t].iter().map(|&x| h(x)).collect::<Vec<_>>(), &k1[li][..t].iter().map(|&x| h(x)).collect::<Vec<_>>()) } else { -1.0 };
        let vd = if v0[li].len() >= t { rel_f32(&v0[li][..t].iter().map(|&x| h(x)).collect::<Vec<_>>(), &v1[li][..t].iter().map(|&x| h(x)).collect::<Vec<_>>()) } else { -1.0 };
        let hd = if !h0[li].is_empty() { rel_f32(&h0[li], &h1[li]) } else { -1.0 };
        let sd = if !s0[li].is_empty() { rel_f32(&s0[li], &s1[li]) } else { -1.0 };
        println!("layer {li:>2}: k {kd:>8.4} v {vd:>8.4} hist {hd:>8.4} state {sd:>8.4}");
    }
    println!("logits relative difference {:.4}", rel_f32(&l0, &l1));
}
