//! Speculation on and off for one prompt, through `chat::generate` (chat template applied):
//! cargo run --release --example spec -- MODEL.gguf "message" [tokens] [temperature]
use cerebro_engine::chat::{generate, GenConfig};
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = Model::load(std::path::Path::new(&a[1])).expect("model");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let n: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(256);
    let temp: f32 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let text = cerebro_engine::chatml(None, &[("user", a[2].as_str())], false);
    let ids = tok.encode(&text);
    for draft in [0i64, 8] {
        cerebro_engine::switches::set(cerebro_engine::switches::SPEC_LOOKUP, draft).expect("switch");
        let mut s = Session::new_gpu(&model, 8192).expect("gpu");
        let cfg = GenConfig { sampler: SamplerConfig { temperature: temp, top_k: 20, top_p: 0.8, min_p: 0.05, repeat_penalty: 1.0, seed: 42 }, max_tokens: n };
        let mut outtxt = String::new();
        let st = generate(&model, &mut s, &tok, &ids, &cfg, |p| {
            outtxt.push_str(p);
            true
        })
        .expect("generate");
        println!("draft {draft}: {} tokens, decode {:.1} tok/s, text hash {:x}", st.output_tokens, st.decode_tps, outtxt.bytes().fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64)));
    }
}
