use cerebro_engine::gguf::Gguf;
fn main() {
    let p = std::env::args().nth(1).expect("path");
    let g = Gguf::open(std::path::Path::new(&p)).expect("open");
    let mut counts = std::collections::BTreeMap::new();
    for t in g.tensors.values() {
        *counts.entry(format!("{:?}", t.dtype)).or_insert(0usize) += t.size;
    }
    for (k, v) in counts { println!("{k}: {:.1} MB", v as f64 / 1e6); }
    for n in ["token_embd.weight", "output.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight"] {
        if let Some(t) = g.tensors.get(n) { println!("{n}: {:?} {:?}", t.dtype, t.dims); }
    }
}
