//! Asks a vision model about a photo, end to end: decode, encode with the projector, feed text and
//! image together, write greedily.
//!
//! Usage: visionchat <model.gguf> <mmproj.gguf> <image.jpg|png> ["question"] [gpu]

use std::path::Path;
use std::time::Instant;

use cerebro_engine::chat::{generate_parts, GenConfig};
use cerebro_engine::model::{ImageTokens, Model, Part, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;
use cerebro_engine::vision::VisionModel;

fn main() {
    let mut a = std::env::args().skip(1);
    let model_path = a.next().expect("model");
    let proj = a.next().expect("mmproj");
    let img_path = a.next().expect("image");
    let question = a.next().unwrap_or_else(|| "Describe this image in two sentences.".into());
    let gpu = a.next().as_deref() == Some("gpu");
    let model = Model::load(Path::new(&model_path)).expect("load model");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let vm = VisionModel::load(Path::new(&proj)).expect("load projector");
    let img = image::open(&img_path).expect("decode image").to_rgb8();
    let (w, h) = img.dimensions();
    let t = Instant::now();
    let input = vm.preprocess(img.as_raw(), w as usize, h as usize);
    let emb = vm.encode(&input).expect("encode");
    println!("image {w}x{h} -> {}x{} pixels -> {}x{} tokens, encoded in {:.2} s", input.width, input.height, emb.cols, emb.rows, t.elapsed().as_secs_f64());
    let mut s = if gpu { Session::new_gpu(&model, 4096).expect("gpu session") } else { Session::new(&model, 4096) };
    let before = tok.encode("<|im_start|>user\n<|vision_start|>");
    let after = tok.encode(&format!("<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    let parts = [Part::Text(&before), Part::Image(ImageTokens { cols: emb.cols, rows: emb.rows, dim: emb.dim, data: &emb.data }), Part::Text(&after)];
    let cfg = GenConfig { sampler: SamplerConfig { temperature: 0.0, ..Default::default() }, max_tokens: 80 };
    let mut out = String::new();
    let g = generate_parts(&model, &mut s, &tok, &parts, &cfg, |p| {
        out.push_str(p);
        true
    })
    .expect("generate");
    println!("{}", out.trim());
    println!("prompt {} tokens at {:.0} tok/s, {} written at {:.1} tok/s", g.prompt_tokens, g.prefill_tps, g.output_tokens, g.decode_tps);
}
