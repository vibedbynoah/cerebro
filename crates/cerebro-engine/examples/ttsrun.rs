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
use cerebro_engine::tts::{speak_in, talker_session, wav_bytes, TtsDecoder, SAMPLE_RATE};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let t = Instant::now();
    let m = Model::load(Path::new(&a[0])).expect("talker");
    let tok = Tokenizer::from_gguf(m.gguf()).expect("tokenizer");
    let dec = TtsDecoder::load(Path::new(&a[1])).expect("decoder");
    dec.prepare();
    let mut session = talker_session(&m, 4096);
    println!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    // Twice: the first sentence after loading, then a warm one in the same session.
    let mut sp = None;
    for run in 1..=2 {
        let t = Instant::now();
        let s = speak_in(&m, &mut session, &tok, &dec, &a[2], "english", 42, 600).expect("speak");
        let secs = s.samples.len() as f64 / SAMPLE_RATE as f64;
        println!("run {run}: {} frames, {:.2} s of audio in {:.2} s", s.frames, secs, t.elapsed().as_secs_f64());
        sp = Some(s);
    }
    let sp = sp.expect("speech");
    std::fs::write(&a[3], wav_bytes(&sp.samples, SAMPLE_RATE)).expect("write wav");
    let peak = sp.samples.iter().fold(0f32, |p, v| p.max(v.abs()));
    let rms = (sp.samples.iter().map(|v| v * v).sum::<f32>() / sp.samples.len().max(1) as f32).sqrt();
    println!("peak {peak:.3}, rms {rms:.4}");
    // The audio decoder on the graphics card against the processor, on the same codes.
    if std::env::var_os("CEREBRO_TTS_COMPARE").is_some() {
        let t = Instant::now();
        let g = dec.decode(&sp.codes);
        let tg = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let c = dec.decode_cpu(&sp.codes);
        let tc = t.elapsed().as_secs_f64();
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (x, y) in g.iter().zip(&c) {
            ab += (*x as f64) * (*y as f64);
            aa += (*x as f64) * (*x as f64);
            bb += (*y as f64) * (*y as f64);
        }
        println!("decoder: graphics card {:.3} s, processor {:.3} s, cosine {:.6}", tg, tc, ab / (aa.sqrt() * bb.sqrt()).max(1e-30));
    }
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
