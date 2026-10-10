//! Formats of each layer's expert tensors.
use cerebro_engine::gguf::Gguf;
fn main() {
    let p = std::env::args().nth(1).expect("path");
    let g = Gguf::open(std::path::Path::new(&p)).expect("open");
    for i in 0.. {
        let f = |s: &str| g.tensors.get(&format!("blk.{i}.{s}.weight")).map(|t| format!("{:?}", t.dtype));
        let Some(gate) = f("ffn_gate_exps") else { break };
        println!("{i}: gate {gate} up {} down {}", f("ffn_up_exps").unwrap(), f("ffn_down_exps").unwrap());
    }
}
