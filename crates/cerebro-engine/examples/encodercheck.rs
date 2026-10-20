//! Compares an encoder on the graphics card with the same encoder on the processor, and times both.
//! Usage: encodercheck vision <mmproj.gguf> <image>
//!        encodercheck audio <mmproj.gguf> <speech.wav>

use std::path::Path;
use std::time::Instant;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64) * (*x as f64);
        bb += (*y as f64) * (*y as f64);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (kind, model, input) = (a[1].as_str(), Path::new(&a[2]), Path::new(&a[3]));
    let time = |f: &mut dyn FnMut() -> Vec<f32>| {
        let mut out = f();
        let t = Instant::now();
        for _ in 0..3 {
            out = f();
        }
        (out, t.elapsed().as_secs_f64() / 3.0 * 1e3)
    };
    let (gpu, cpu, tg, tc) = match kind {
        "vision" => {
            let m = cerebro_engine::vision::VisionModel::load(model).expect("load");
            let img = image::open(input).expect("image").to_rgb8();
            let inp = m.preprocess(img.as_raw(), img.width() as usize, img.height() as usize);
            println!("{}x{} pixels", inp.width, inp.height);
            let (g, tg) = time(&mut || m.encode(&inp).expect("encode").data);
            let (c, tc) = time(&mut || m.encode_traced(&inp, &mut |_, _| {}).expect("encode").data);
            (g, c, tg, tc)
        }
        "audio" => {
            let m = cerebro_engine::audio::AudioModel::load(model).expect("load");
            let (x, rate) = cerebro_engine::audio::read_wav(&std::fs::read(input).expect("read wav")).expect("wav");
            let wav = cerebro_engine::audio::resample(&x, rate, cerebro_engine::audio::SAMPLE_RATE);
            let (g, tg) = time(&mut || m.encode(&wav).expect("encode").data);
            let (c, tc) = time(&mut || m.encode_cpu(&wav).expect("encode").data);
            (g, c, tg, tc)
        }
        k => panic!("unknown encoder {k}"),
    };
    println!("graphics card {tg:.1} ms, processor {tc:.1} ms, cosine {:.6} over {} values", cosine(&gpu, &cpu), gpu.len());
}
