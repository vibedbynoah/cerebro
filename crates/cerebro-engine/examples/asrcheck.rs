//! Transcribes WAV files with a speech model (Qwen3-ASR: language model + audio projector) and,
//! when a file has a `<name>.ref.txt` beside it, checks the text against it.
//!
//! Usage: asrcheck <model.gguf> <mmproj.gguf> <a.wav> [b.wav ...] [gpu]

use std::path::Path;
use std::time::Instant;

use cerebro_engine::audio::{read_wav, resample, AudioModel, SAMPLE_RATE};
use cerebro_engine::chat::{generate_parts, GenConfig};
use cerebro_engine::model::{ImageTokens, Model, Part, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.last().filter(|a| *a == "gpu" || *a == "cpu").cloned();
    let gpu = mode.as_deref() == Some("gpu");
    let (model_path, proj) = (&args[0], &args[1]);
    let files = &args[2..args.len() - mode.is_some() as usize];
    let model = Model::load(Path::new(model_path)).expect("load model");
    let tok = Tokenizer::from_gguf(model.gguf()).expect("tokenizer");
    let am = AudioModel::load(Path::new(proj)).expect("load projector");
    let mut s = if gpu { Session::new_gpu(&model, 4096).expect("gpu") } else { Session::new(&model, 4096) };
    let mut failed = false;
    for f in files {
        let (x, rate) = read_wav(&std::fs::read(f).expect("read wav")).expect("wav");
        let x = resample(&x, rate, SAMPLE_RATE);
        let t = Instant::now();
        let e = am.encode(&x).expect("encode");
        let enc_s = t.elapsed().as_secs_f64();
        let before = tok.encode("<|im_start|>user\n<|audio_start|>");
        let after = tok.encode("<|audio_end|>Transcribe.<|im_end|>\n<|im_start|>assistant\n");
        let parts = [Part::Text(&before), Part::Audio(ImageTokens { cols: e.tokens, rows: 1, dim: e.dim, data: &e.data }), Part::Text(&after)];
        let cfg = GenConfig { sampler: SamplerConfig { temperature: 0.0, ..Default::default() }, max_tokens: 160 };
        let mut out = String::new();
        let g = generate_parts(&model, &mut s, &tok, &parts, &cfg, |p| {
            out.push_str(p);
            true
        })
        .expect("generate");
        let text = out.trim();
        println!("{f}: {:.1} s of audio, {} tokens, encoded in {:.2} s, {} written at {:.0} tok/s", x.len() as f64 / SAMPLE_RATE as f64, e.tokens, enc_s, g.output_tokens, g.decode_tps);
        println!("  {text}");
        let reference = Path::new(f).with_extension("ref.txt");
        if let Ok(want) = std::fs::read_to_string(&reference) {
            let ok = want.trim() == text;
            failed |= !ok;
            println!("  {}", if ok { "matches the reference" } else { "DIFFERS from the reference" });
            if !ok {
                println!("  reference: {}", want.trim());
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
