//! Speaks a sentence with Qwen3-TTS, writes the WAV, and (when a speech-to-text model is given)
//! transcribes it back as an objective check that the audio says the words.
//!
//! Usage: ttsrun <talker.gguf> <tts-mmproj.gguf> "<text>" <out.wav> [<asr.gguf> <asr-mmproj.gguf>]

use std::path::Path;
use std::time::Instant;

use cerebro_engine::audio::{resample, AudioModel, SAMPLE_RATE as ASR_RATE};
use cerebro_engine::chat::{generate_parts, GenConfig};
use cerebro_engine::model::{ImageTokens, Model, Part, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;
use cerebro_engine::tts::{speak, wav_bytes, TtsDecoder, SAMPLE_RATE};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let t = Instant::now();
    let m = Model::load(Path::new(&a[0])).expect("talker");
    let tok = Tokenizer::from_gguf(m.gguf()).expect("tokenizer");
    let dec = TtsDecoder::load(Path::new(&a[1])).expect("decoder");
    println!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let sp = speak(&m, &tok, &dec, &a[2], "english", 42, 600).expect("speak");
    let secs = sp.samples.len() as f64 / SAMPLE_RATE as f64;
    println!("{} frames, {:.2} s of audio in {:.1} s", sp.frames, secs, t.elapsed().as_secs_f64());
    std::fs::write(&a[3], wav_bytes(&sp.samples, SAMPLE_RATE)).expect("write wav");
    let peak = sp.samples.iter().fold(0f32, |p, v| p.max(v.abs()));
    let rms = (sp.samples.iter().map(|v| v * v).sum::<f32>() / sp.samples.len().max(1) as f32).sqrt();
    println!("peak {peak:.3}, rms {rms:.4}");
    if a.len() >= 6 {
        let asr = Model::load(Path::new(&a[4])).expect("asr");
        let atok = Tokenizer::from_gguf(asr.gguf()).expect("asr tokenizer");
        let am = AudioModel::load(Path::new(&a[5])).expect("asr projector");
        let x = resample(&sp.samples, SAMPLE_RATE, ASR_RATE);
        let e = am.encode(&x).expect("encode");
        let mut s = Session::new(&asr, 4096);
        let (b, c) = (atok.encode("<|im_start|>user\n<|audio_start|>"), atok.encode("<|audio_end|>Transcribe.<|im_end|>\n<|im_start|>assistant\n"));
        let parts = [Part::Text(&b), Part::Audio(ImageTokens { cols: e.tokens, rows: 1, dim: e.dim, data: &e.data }), Part::Text(&c)];
        let mut out = String::new();
        generate_parts(&asr, &mut s, &atok, &parts, &GenConfig { sampler: SamplerConfig { temperature: 0.0, ..Default::default() }, max_tokens: 200 }, |p| {
            out.push_str(p);
            true
        })
        .expect("transcribe");
        println!("heard: {}", out.trim());
    }
}
