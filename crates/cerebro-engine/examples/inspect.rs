//! Prints a GGUF's scalar metadata and the tensors of layer 0 (shapes and formats).
use cerebro_engine::gguf::Gguf;
fn main() {
    let p = std::env::args().nth(1).expect("path");
    let g = Gguf::open(std::path::Path::new(&p)).expect("open");
    let mut keys: Vec<_> = g.meta.keys().filter(|k| !k.starts_with("tokenizer.ggml.")).collect();
    keys.sort();
    for k in keys {
        let v = &g.meta[k];
        let s = v.as_u64().map(|x| x.to_string()).or_else(|| v.as_f32().map(|x| x.to_string())).or_else(|| v.as_str().map(|x| x.chars().take(60).collect())).unwrap_or_else(|| "[..]".into());
        println!("{k} = {s}");
    }
    let mut ts: Vec<_> = g.tensors.values().filter(|t| t.name.starts_with("blk.0.") || !t.name.starts_with("blk.")).collect();
    ts.sort_by(|a, b| a.name.cmp(&b.name));
    for t in ts { println!("{} {:?} {:?} {:.1} MB", t.name, t.dtype, t.dims, t.size as f64 / 1e6); }
}
