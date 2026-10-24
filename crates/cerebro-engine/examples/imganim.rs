//! Animation check: writes the frames of a short clip as PNGs.
//! cargo run --release --example imganim -- MODEL_DIR OUT_DIR "prompt" ["end prompt"] [frames]
use cerebro_engine::diffusion::Diffusion;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = Diffusion::load(std::path::Path::new(&a[1])).expect("model");
    let out = std::path::Path::new(&a[2]);
    let frames: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(12);
    let t = std::time::Instant::now();
    let f = d.animate(&a[3], a.get(4).map(String::as_str), 512, frames, 0.35, 11, |_| {}).expect("animate");
    eprintln!("{} frames in {:.1} s", f.len(), t.elapsed().as_secs_f64());
    for (i, img) in f.iter().enumerate() {
        image::save_buffer(out.join(format!("frame{i:02}.png")), img, 512, 512, image::ColorType::Rgb8).unwrap();
    }
}
