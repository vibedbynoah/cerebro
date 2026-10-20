//! Checks the diffusion code against the reference dumps of `.ref/sd/dump.py` stage by stage
//! (tokens, text embeddings, the UNet's noise prediction, the VAE's image), then generates an
//! image from scratch and writes it.
//!
//! Usage: sdcheck <model folder> <dumps folder> <out.png> ["prompt"]

use std::path::Path;
use std::time::Instant;

use cerebro_engine::diffusion::{planar_to_rows, Diffusion};

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).expect("dump").chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// (max abs error, max abs reference value, cosine similarity).
fn compare(a: &[f32], b: &[f32]) -> (f32, f32, f64) {
    let err = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let mx = b.iter().fold(0f32, |m, y| m.max(y.abs()));
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (err, mx, dot / (na * nb))
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, dumps, out) = (Path::new(&a[0]), Path::new(&a[1]), &a[2]);
    let t = Instant::now();
    let d = Diffusion::load(model).expect("load");
    println!("loaded in {:.1} s ({})", t.elapsed().as_secs_f64(), if d.on_gpu() { "graphics card" } else { "processor" });
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dumps.join("meta.json")).expect("meta")).expect("json");
    let mut ok = true;
    // Tokens.
    let want: Vec<u32> = std::fs::read(dumps.join("ids.bin")).expect("ids").chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
    let ids = d.tokenizer.encode(meta["prompt"].as_str().unwrap_or(""));
    let same = ids == want;
    ok &= same;
    println!("tokens: {}", if same { "identical".to_owned() } else { format!("DIFFER\n  ours {:?}\n  want {:?}", &ids[..12], &want[..12]) });
    // Text encoder.
    let t = Instant::now();
    let ctx = d.encode_text(&want).expect("text");
    let (e, m, c) = compare(&ctx, &read_f32(&dumps.join("prompt_embeds.bin")));
    println!("text embeddings: max error {e:.4} (values to {m:.2}), cosine {c:.6}, {:.2} s", t.elapsed().as_secs_f64());
    ok &= c > 0.9999;
    // UNet, on the dumped input.
    let size = meta["size"].as_u64().unwrap_or(512) as usize;
    let lh = size / 8;
    let input = planar_to_rows(&read_f32(&dumps.join("unet_in.bin")), 4, lh * lh);
    let t = Instant::now();
    let ctx_ref = read_f32(&dumps.join("prompt_embeds.bin"));
    let eps = d.unet(&input, lh, lh, meta["timestep"].as_f64().unwrap_or(999.0) as f32, &ctx_ref, 77).expect("unet");
    let (e, m, c) = compare(&eps, &planar_to_rows(&read_f32(&dumps.join("noise_pred.bin")), 4, lh * lh));
    println!("unet: max error {e:.4} (values to {m:.2}), cosine {c:.6}, {:.2} s", t.elapsed().as_secs_f64());
    ok &= c > 0.999;
    // VAE, on the dumped latents.
    let z: Vec<f32> = planar_to_rows(&read_f32(&dumps.join("latents_out.bin")), 4, lh * lh).iter().map(|v| v / d.scaling).collect();
    let t = Instant::now();
    let img = d.vae(&z, lh, lh).expect("vae");
    let img01: Vec<f32> = img.iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
    let (e, m, c) = compare(&img01, &read_f32(&dumps.join("image.bin")));
    println!("vae: max error {e:.4} (values to {m:.2}), cosine {c:.6}, {:.2} s", t.elapsed().as_secs_f64());
    ok &= c > 0.999;
    // From scratch.
    let prompt = a.get(3).cloned().unwrap_or_else(|| meta["prompt"].as_str().unwrap_or("").to_owned());
    let mut rgb = Vec::new();
    for run in 0..3 {
        let t = Instant::now();
        rgb = d.generate(&prompt, size, 1, 7).expect("generate");
        println!("generated {size}x{size} in {:.3} s (run {})", t.elapsed().as_secs_f64(), run + 1);
        if let Some(g) = cerebro_engine::gtensor::Gx::get() {
            print!("{}", g.report());
        }
    }
    // Warm time of each stage (each synchronised by its download).
    let lh = size / 8;
    let ids = d.tokenizer.encode(&prompt);
    let lat = vec![0.1f32; lh * lh * 4];
    for _ in 0..2 {
        let t = Instant::now();
        let ctx = d.encode_text(&ids).expect("text");
        let t1 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = d.unet(&lat, lh, lh, 999.0, &ctx, 77).expect("unet");
        let t2 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = d.vae(&lat, lh, lh).expect("vae");
        println!("stages: text {:.1} ms, unet {:.1} ms, vae {:.1} ms", t1 * 1e3, t2 * 1e3, t.elapsed().as_secs_f64() * 1e3);
    }
    image::RgbImage::from_raw(size as u32, size as u32, rgb).expect("image").save(out).expect("save");
    if !ok {
        std::process::exit(1);
    }
}
