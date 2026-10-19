//! Images in chat: finding a model's vision projector, reading the images the Chat screen sends
//! (data: URLs), and splitting a formatted prompt at the places images go.
//!
//! The prompt is formatted as text with one `MARK` character where each image's tokens belong
//! (between the model's vision start and end tokens); `split_prompt` cuts it there so text and
//! encoded images can be fed in order (`cerebro_engine::model::Part`).

use std::path::Path;

use base64::Engine as _;
use cerebro_engine::gguf::Gguf;
use serde::Serialize;

/// Stands in for an image inside a formatted prompt. A private-use character: never produced by
/// the tokenizer's normal text, and stripped from everything the user or a tool wrote.
pub const MARK: char = '\u{E000}';

/// Largest image accepted, decoded: bigger ones are refused before decoding to keep memory bounded.
const MAX_DATA_BYTES: usize = 20 * 1024 * 1024;

/// The vision projector paired with a model (an `mmproj*.gguf` beside it whose output width is the
/// model's width).
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Projector {
    pub file: String,
    pub size_gi_b: f64,
    pub path: String,
    /// "vision" (images), "audio" (speech in) or "tts" (speech out).
    pub kind: &'static str,
}

/// The projector for the model at `model_path` with hidden width `embedding`: among the
/// `mmproj*.gguf` files in its folder, one whose `clip.vision.projection_dim` equals the width
/// (a projector made for another size of the family cannot feed this one), preferring a file
/// whose name mentions the model's base name.
pub fn find_projector(model_path: &Path, embedding: u64, basename: &str) -> Option<Projector> {
    let dir = model_path.parent()?;
    let mut found: Vec<(bool, Projector)> = Vec::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let lower = name.to_lowercase();
        if !lower.starts_with("mmproj") || !lower.ends_with(".gguf") {
            continue;
        }
        let Ok(g) = Gguf::open(&p) else { continue };
        // A speech decoder (text to speech) first: its file also carries a speaker encoder that
        // looks like an audio input.
        let kind = if g.u64("clip.gen.audio.projection_dim").ok() == Some(embedding) {
            "tts"
        } else if g.u64("clip.vision.projection_dim").ok() == Some(embedding) {
            "vision"
        } else if g.u64("clip.audio.projection_dim").ok() == Some(embedding) {
            "audio"
        } else {
            continue;
        };
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        let named = !basename.is_empty() && lower.contains(&basename.to_lowercase());
        found.push((named, Projector { file: name, size_gi_b: size as f64 / (1u64 << 30) as f64, path: p.to_string_lossy().into_owned(), kind }));
    }
    found.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.file.cmp(&b.1.file)));
    found.into_iter().next().map(|(_, p)| p)
}

/// An image decoded to 8-bit RGB, row by row.
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// Reads a `data:image/...;base64,...` URL (PNG, JPEG, WebP or GIF; a GIF's first frame).
pub fn decode_data_url(url: &str) -> Result<Rgb, String> {
    let (head, body) = url.split_once(',').ok_or("An attached image is not a data URL.")?;
    if !head.starts_with("data:image/") || !head.ends_with(";base64") {
        return Err("An attached image is not a base64 image data URL.".to_owned());
    }
    if body.len() / 4 * 3 > MAX_DATA_BYTES {
        return Err("An attached image is over 20 MB.".to_owned());
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(body.trim()).map_err(|_| "An attached image's data is damaged.".to_owned())?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("An attached image could not be read: {e}."))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Err("An attached image is empty.".to_owned());
    }
    Ok(Rgb { width: w as usize, height: h as usize, pixels: rgb.into_raw() })
}

/// A stable key for an image's data (FNV-1a), so a chat's images are encoded once, not every turn.
pub fn key(data: &str) -> u64 {
    data.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// The text that holds an image's place in a user turn: the model's vision start and end tokens
/// around a `MARK`.
pub fn placeholder() -> String {
    format!("<|vision_start|>{MARK}<|vision_end|>")
}

/// Cuts a formatted prompt at its first `images` marks: `images + 1` text pieces, with the text
/// between image i and i + 1 at index i + 1. Marks beyond `images` are dropped from the text.
pub fn split_prompt(text: &str, images: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::with_capacity(images + 1);
    let mut rest = text;
    for _ in 0..images {
        match rest.split_once(MARK) {
            Some((a, b)) => {
                pieces.push(a.to_owned());
                rest = b;
            }
            None => break,
        }
    }
    pieces.push(rest.replace(MARK, ""));
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_puts_text_between_images_in_order() {
        let t = format!("a{MARK}b{MARK}c");
        assert_eq!(split_prompt(&t, 2), vec!["a", "b", "c"]);
        assert_eq!(split_prompt("plain", 0), vec!["plain"]);
    }

    #[test]
    fn extra_marks_are_dropped_and_missing_ones_shorten_the_list() {
        let t = format!("a{MARK}b{MARK}c");
        assert_eq!(split_prompt(&t, 1), vec!["a", "bc"]);
        assert_eq!(split_prompt("no marks", 2), vec!["no marks"]);
    }

    #[test]
    fn data_urls_decode_to_rgb() {
        // A 2x1 PNG: red, then blue.
        let mut png = Vec::new();
        let img = image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 0, 0, 255]).expect("image");
        image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).expect("png");
        let url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png));
        let rgb = decode_data_url(&url).expect("decode");
        assert_eq!((rgb.width, rgb.height), (2, 1));
        assert_eq!(rgb.pixels, vec![255, 0, 0, 0, 0, 255]);
    }

    #[test]
    fn bad_data_urls_are_refused_in_plain_words() {
        assert!(decode_data_url("https://example.com/x.png").is_err());
        assert!(decode_data_url("data:text/plain;base64,aGk=").is_err());
        assert!(decode_data_url("data:image/png;base64,!!!").is_err());
    }

    #[test]
    fn keys_differ_for_different_images() {
        assert_ne!(key("data:image/png;base64,AAAA"), key("data:image/png;base64,AAAB"));
        assert_eq!(key("x"), key("x"));
    }

    #[test]
    fn the_projector_beside_the_model_is_found_by_width() {
        let p = Path::new(r"C:\models\dev\qwen35\Qwen3.5-0.8B-Q8_0.gguf");
        if !p.exists() {
            return;
        }
        let small = find_projector(p, 1024, "Qwen3.5-0.8B").expect("projector for 0.8B");
        assert_eq!(small.file, "mmproj-F16.gguf");
        // The 4B model is 2560 wide: the 0.8B projector does not fit it.
        assert!(find_projector(p, 2560, "Qwen3.5-4B").is_none());
    }
}
