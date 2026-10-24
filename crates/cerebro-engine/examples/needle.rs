//! Long-context check from the command line (longctx.rs):
//! `needle <model.gguf> <tokens>[,<tokens>...] [--depths 0.1,0.5,0.9] [--no-yarn] [--cpu] [--trained N]`
//! A length past the model's trained context is run with YaRN unless --no-yarn.

use cerebro_engine::longctx;
use cerebro_engine::model::{Model, Session};
use cerebro_engine::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("model path");
    let lengths: Vec<usize> = args.get(2).expect("lengths").split(',').map(|v| v.parse().expect("length")).collect();
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let depths: Vec<f32> = flag("--depths").unwrap_or_else(|| "0.1,0.5,0.9".into()).split(',').map(|v| v.parse().expect("depth")).collect();
    let yarn = !args.iter().any(|a| a == "--no-yarn");
    let cpu = args.iter().any(|a| a == "--cpu");
    let mut model = Model::load(std::path::Path::new(path)).expect("load");
    // --trained N: treat the model as trained for N tokens (exercises YaRN at a large factor on a
    // card too small for a real 4x context).
    if let Some(t) = flag("--trained").and_then(|v| v.parse().ok()) {
        model.cfg.ctx_train = t;
    }
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let max = *lengths.iter().max().expect("a length") + 64;
    let factor = if yarn { model.extend_context(max) } else { 1.0 };
    println!("{}: trained for {} tokens; checking up to {max} (YaRN factor {factor:.2})", model.cfg.arch, model.cfg.ctx_train);
    let mut s = if cpu { Session::new(&model, max) } else { Session::new_gpu(&model, max).unwrap_or_else(|e| { eprintln!("GPU: {e}; processor"); Session::new(&model, max) }) };
    for &n in &lengths {
        for &d in &depths {
            let nd = longctx::build(&tok, n, d, (n as u64) * 31 + (d * 100.0) as u64);
            let p = longctx::probe(&model, &mut s, &tok, &nd).expect("probe");
            println!("{:>7} tokens, depth {:>3.0}%: {} ({:.1} s) reply {:?} want {}", p.tokens, p.depth * 100.0, if p.found { "found" } else { "MISSED" }, p.seconds, p.reply, nd.passcode);
        }
    }
}
