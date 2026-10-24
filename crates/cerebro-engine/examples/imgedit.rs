//! Image editing check: makes a picture, then edits it, and saves both as PNG.
//! cargo run --release --example imgedit -- MODEL_DIR OUT_DIR "first prompt" "edit prompt" [strength]
use cerebro_engine::diffusion::Diffusion;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = Diffusion::load(std::path::Path::new(&a[1])).expect("model");
    let out = std::path::Path::new(&a[2]);
    let strength: f32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.6);
    let t = std::time::Instant::now();
    let img = d.generate(&a[3], 512, 1, 42).expect("generate");
    eprintln!("generated in {:.1} s", t.elapsed().as_secs_f64());
    image::save_buffer(out.join("first.png"), &img, 512, 512, image::ColorType::Rgb8).unwrap();
    let t = std::time::Instant::now();
    let ed = d.edit(&a[4], &img, 512, 2, strength, 7).expect("edit");
    eprintln!("edited in {:.1} s", t.elapsed().as_secs_f64());
    image::save_buffer(out.join("edited.png"), &ed, 512, 512, image::ColorType::Rgb8).unwrap();
}
