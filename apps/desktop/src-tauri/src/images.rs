//! Image Studio on Cerebro Engine: image models (Stable Diffusion 2.x-family folders in the
//! diffusers layout, such as SD-Turbo) found in the model folders, generation on their own thread
//! (a model is loaded on first use and kept), and the results saved as PNG files under the app's
//! data folder so the gallery survives a restart.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;

use base64::Engine as _;
use cerebro_engine::diffusion::Diffusion;
use serde::Serialize;

use crate::engine::{LibraryModel, ModelArch};

/// Whether `dir` is a diffusers pipeline folder this engine runs (model_index.json naming a
/// Stable Diffusion pipeline, with a UNet using linear projections: the 2.x family).
pub fn is_image_model(dir: &Path) -> bool {
    let Ok(t) = std::fs::read_to_string(dir.join("model_index.json")) else { return false };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else { return false };
    v["_class_name"].as_str() == Some("StableDiffusionPipeline")
}

fn folder_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            total += if p.is_dir() { folder_size(&p) } else { e.metadata().map_or(0, |m| m.len()) };
        }
    }
    total
}

/// The library entry for an image model folder.
pub fn describe(dir: &Path, source: &'static str, id: String) -> Option<LibraryModel> {
    if !is_image_model(dir) {
        return None;
    }
    let unet: serde_json::Value = std::fs::read_to_string(dir.join("unet").join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())?;
    let supported = unet["use_linear_projection"].as_bool() == Some(true);
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let fp16 = dir.join("unet").join("diffusion_pytorch_model.fp16.safetensors").exists();
    Some(LibraryModel {
        id,
        name: name.clone(),
        quant: if fp16 { "FP16".into() } else { "FP32".into() },
        params: String::new(),
        architecture: "dense",
        size_gi_b: folder_size(dir) as f64 / (1u64 << 30) as f64,
        context_max: 0,
        tune: "untuned",
        provenance: "measured",
        kind: "image",
        modalities: vec!["image-gen"],
        family: "Stable Diffusion".into(),
        file: name,
        path: dir.to_string_lossy().into_owned(),
        source,
        sha256: String::new(),
        arch: ModelArch { layers: 0, kv_heads: 0, head_dim: 0, embedding: 0, experts: None, active_experts: None },
        tuned_params: None,
        tuned_at: None,
        tune_result: None,
        projector: None,
        supported,
        arch_name: "stable-diffusion".into(),
    })
}

/// One generated image.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Generated {
    /// The PNG as a data: URL, for showing at once.
    pub data_url: String,
    /// Where the PNG was saved.
    pub path: String,
    pub seed: u64,
    pub seconds: f64,
    pub width: u32,
    pub height: u32,
}

/// A request: prompt, side length (multiple of 64), Euler steps, seed.
pub struct Request {
    pub prompt: String,
    pub size: usize,
    pub steps: usize,
    pub seed: u64,
}

enum Job {
    Generate(LibraryModel, Request, Sender<Result<(Vec<u8>, f64), String>>),
}

/// Image generation on its own thread.
pub struct Images {
    tx: Mutex<Sender<Job>>,
    out_dir: Mutex<Option<PathBuf>>,
}

impl Default for Images {
    fn default() -> Self {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("cerebro-images".into())
            .spawn(move || {
                let mut loaded: Option<(String, Diffusion)> = None;
                while let Ok(Job::Generate(m, r, reply)) = rx.recv() {
                    let res = (|| -> Result<(Vec<u8>, f64), String> {
                        if loaded.as_ref().is_none_or(|l| l.0 != m.id) {
                            loaded = None;
                            let d = Diffusion::load(Path::new(&m.path)).map_err(|e| format!("Could not load the image model: {e}"))?;
                            loaded = Some((m.id.clone(), d));
                        }
                        let d = &loaded.as_ref().ok_or("The image model is not loaded.")?.1;
                        let t = std::time::Instant::now();
                        let rgb = d.generate(&r.prompt, r.size, r.steps, r.seed).map_err(|e| format!("The image could not be made: {e}"))?;
                        Ok((rgb, t.elapsed().as_secs_f64()))
                    })();
                    let _ = reply.send(res);
                }
            })
            .expect("image thread");
        Images { tx: Mutex::new(tx), out_dir: Mutex::new(None) }
    }
}

impl Images {
    /// Where images are saved (`<app data>\images`).
    pub fn set_out_dir(&self, dir: PathBuf) {
        *self.out_dir.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir);
    }

    pub fn generate(&self, m: &LibraryModel, r: Request) -> Result<Generated, String> {
        if r.prompt.trim().is_empty() {
            return Err("Describe the picture first.".to_owned());
        }
        let (size, seed) = (r.size, r.seed);
        let (reply, rx) = channel();
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Job::Generate(m.clone(), r, reply)).map_err(|_| "The image engine stopped.".to_owned())?;
        let (rgb, seconds) = rx.recv().map_err(|_| "The image engine stopped.".to_owned())??;
        let img = image::RgbImage::from_raw(size as u32, size as u32, rgb).ok_or("The image had the wrong size.")?;
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(|e| format!("The image could not be encoded: {e}."))?;
        let dir = self.out_dir.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&dir).map_err(|e| format!("The images folder could not be made: {e}."))?;
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
        let path = dir.join(format!("cerebro-{stamp}-{seed}.png"));
        std::fs::write(&path, &png).map_err(|e| format!("The image could not be saved: {e}."))?;
        Ok(Generated {
            data_url: format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png)),
            path: path.to_string_lossy().into_owned(),
            seed,
            seconds,
            width: size as u32,
            height: size as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_folders_are_recognised() {
        let d = std::env::temp_dir().join(format!("cerebro-img-{}", std::process::id()));
        std::fs::create_dir_all(d.join("unet")).unwrap();
        assert!(!is_image_model(&d));
        std::fs::write(d.join("model_index.json"), r#"{"_class_name": "StableDiffusionPipeline"}"#).unwrap();
        std::fs::write(d.join("unet").join("config.json"), r#"{"use_linear_projection": true}"#).unwrap();
        let m = describe(&d, "other", "mtest".into()).expect("an image model");
        assert_eq!((m.kind, m.supported), ("image", true));
        std::fs::write(d.join("model_index.json"), r#"{"_class_name": "FluxPipeline"}"#).unwrap();
        assert!(!is_image_model(&d));
        std::fs::remove_dir_all(&d).ok();
    }

    /// A real image through the app path: needs the SD-Turbo folder.
    /// `cargo test --release -p cerebro-desktop makes_an_image -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn makes_an_image() {
        let dir = Path::new(r"C:\models\dev\sd-turbo");
        let m = describe(dir, "other", "msd".into()).expect("model");
        let img = Images::default();
        img.set_out_dir(std::env::temp_dir().join("cerebro-images-test"));
        let g = img.generate(&m, Request { prompt: "a red apple on a wooden table, photo".into(), size: 512, steps: 1, seed: 7 }).expect("generate");
        println!("{} in {:.1} s", g.path, g.seconds);
        assert!(g.data_url.starts_with("data:image/png;base64,") && std::path::Path::new(&g.path).exists());
    }
}
