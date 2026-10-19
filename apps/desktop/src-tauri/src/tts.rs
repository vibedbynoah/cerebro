//! Text to speech: a speech model (Qwen3-TTS: a talker with its decoder projector beside it) on
//! its own thread, and playback through the default output device on another (an audio stream
//! stays on the thread that opened it). Reading a chat reply aloud goes sentence by sentence, so
//! the first sentence plays while the next is being made.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use cerebro_engine::model::Model;
use cerebro_engine::tokenizer::Tokenizer;
use cerebro_engine::tts::{speak, TtsDecoder, SAMPLE_RATE};

use crate::engine::LibraryModel;

/// Longest speech made in one piece, frames (80 ms each): about 48 s.
const MAX_FRAMES: usize = 600;

/// The output device: a queue of samples (at the device's rate) the stream drains.
pub struct Player {
    queue: Arc<Mutex<VecDeque<f32>>>,
    rate: Arc<AtomicU64>,
    started: Mutex<bool>,
}

impl Default for Player {
    fn default() -> Self {
        Player { queue: Arc::default(), rate: Arc::new(AtomicU64::new(0)), started: Mutex::new(false) }
    }
}

impl Player {
    /// Opens the default output device once (on its own thread, kept for the app's life).
    fn ensure(&self) -> Result<u32, String> {
        let mut started = self.started.lock().unwrap_or_else(|e| e.into_inner());
        if *started {
            return Ok(self.rate.load(Ordering::SeqCst) as u32);
        }
        let (tx, rx) = channel::<Result<u32, String>>();
        let (queue, rate) = (self.queue.clone(), self.rate.clone());
        std::thread::Builder::new()
            .name("cerebro-speaker".into())
            .spawn(move || {
                use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
                let opened = (|| -> Result<(cpal::Stream, u32), String> {
                    let device = cpal::default_host().default_output_device().ok_or("No speaker or headphones were found.")?;
                    let cfg = device.default_output_config().map_err(|e| format!("The speaker could not be opened: {e}."))?;
                    let (channels, sr) = (cfg.channels() as usize, cfg.sample_rate().0);
                    let q = queue.clone();
                    let err = |e| eprintln!("Speaker: {e}");
                    let s = match cfg.sample_format() {
                        cpal::SampleFormat::F32 => device.build_output_stream(
                            &cfg.config(),
                            move |d: &mut [f32], _: &_| {
                                let mut q = q.lock().unwrap_or_else(|e| e.into_inner());
                                for frame in d.chunks_mut(channels.max(1)) {
                                    let v = q.pop_front().unwrap_or(0.0);
                                    frame.iter_mut().for_each(|o| *o = v);
                                }
                            },
                            err,
                            None,
                        ),
                        cpal::SampleFormat::I16 => device.build_output_stream(
                            &cfg.config(),
                            move |d: &mut [i16], _: &_| {
                                let mut q = q.lock().unwrap_or_else(|e| e.into_inner());
                                for frame in d.chunks_mut(channels.max(1)) {
                                    let v = (q.pop_front().unwrap_or(0.0).clamp(-1.0, 1.0) * 32767.0) as i16;
                                    frame.iter_mut().for_each(|o| *o = v);
                                }
                            },
                            err,
                            None,
                        ),
                        f => return Err(format!("The speaker's sample format ({f:?}) is not supported.")),
                    }
                    .map_err(|e| format!("The speaker could not be opened: {e}."))?;
                    s.play().map_err(|e| format!("The speaker could not start: {e}."))?;
                    Ok((s, sr))
                })();
                match opened {
                    Ok((stream, sr)) => {
                        rate.store(sr as u64, Ordering::SeqCst);
                        let _ = tx.send(Ok(sr));
                        // The stream lives as long as this thread, which parks for the app's life.
                        let _keep = stream;
                        loop {
                            std::thread::park();
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                    }
                }
            })
            .map_err(|_| "The speaker thread could not start.".to_owned())?;
        let sr = rx.recv().map_err(|_| "The speaker could not be opened.".to_owned())??;
        *started = true;
        Ok(sr)
    }

    /// Queues 24 kHz samples; returns their length in seconds.
    pub fn play(&self, samples: &[f32]) -> Result<f64, String> {
        let sr = self.ensure()?;
        let x = cerebro_engine::audio::resample(samples, SAMPLE_RATE, sr);
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).extend(x);
        Ok(samples.len() as f64 / SAMPLE_RATE as f64)
    }

    /// Seconds of audio still queued.
    pub fn remaining(&self) -> f64 {
        let sr = self.rate.load(Ordering::SeqCst).max(1) as f64;
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).len() as f64 / sr
    }

    pub fn stop(&self) {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

struct Loaded {
    id: String,
    model: Model,
    tok: Tokenizer,
    dec: TtsDecoder,
}

enum Job {
    Speak(LibraryModel, String, Sender<Result<Vec<f32>, String>>),
}

/// The speech model on its own thread, loaded on first use.
pub struct Tts {
    tx: Mutex<Sender<Job>>,
    loaded_id: Arc<Mutex<Option<String>>>,
    /// Set by `stop`: reading aloud ends after the sentence being made.
    pub stopping: Arc<AtomicBool>,
}

impl Default for Tts {
    fn default() -> Self {
        let (tx, rx) = channel::<Job>();
        let loaded_id: Arc<Mutex<Option<String>>> = Arc::default();
        let id = loaded_id.clone();
        std::thread::Builder::new()
            .name("cerebro-tts".into())
            .spawn(move || {
                let mut loaded: Option<Loaded> = None;
                while let Ok(Job::Speak(m, text, reply)) = rx.recv() {
                    let r = (|| -> Result<Vec<f32>, String> {
                        if loaded.as_ref().is_none_or(|l| l.id != m.id) {
                            loaded = None;
                            *id.lock().unwrap_or_else(|e| e.into_inner()) = None;
                            let proj = m.projector.as_ref().filter(|p| p.kind == "tts").ok_or("This model has no speech decoder beside it, so it cannot speak.")?;
                            let model = Model::load(Path::new(&m.path)).map_err(|e| format!("Could not load the speech model: {e}"))?;
                            let tok = Tokenizer::from_gguf(model.gguf()).map_err(|e| format!("Could not read the speech model's tokenizer: {e}"))?;
                            let dec = TtsDecoder::load(Path::new(&proj.path)).map_err(|e| format!("Could not load the speech decoder: {e}"))?;
                            loaded = Some(Loaded { id: m.id.clone(), model, tok, dec });
                            *id.lock().unwrap_or_else(|e| e.into_inner()) = Some(m.id.clone());
                        }
                        let l = loaded.as_ref().ok_or("The speech model is not loaded.")?;
                        let sp = speak(&l.model, &l.tok, &l.dec, &text, "english", 42, MAX_FRAMES).map_err(|e| format!("Speech failed: {e}"))?;
                        Ok(sp.samples)
                    })();
                    let _ = reply.send(r);
                }
            })
            .expect("speech synthesis thread");
        Tts { tx: Mutex::new(tx), loaded_id, stopping: Arc::default() }
    }
}

impl Tts {
    /// 24 kHz samples of `text` spoken by `m`.
    pub fn synthesize(&self, m: &LibraryModel, text: &str) -> Result<Vec<f32>, String> {
        let (reply, rx) = channel();
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Job::Speak(m.clone(), text.to_owned(), reply)).map_err(|_| "The speech engine stopped.".to_owned())?;
        rx.recv().map_err(|_| "The speech engine stopped.".to_owned())?
    }

    pub fn loaded_id(&self) -> Option<String> {
        self.loaded_id.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Text split into speakable sentences (at most ~240 characters each), markdown marks removed.
pub fn sentences(text: &str) -> Vec<String> {
    let clean: String = text.chars().filter(|c| !matches!(c, '*' | '#' | '`' | '_' | '>' | '|')).collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in clean.split_whitespace() {
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
        let end = word.ends_with(['.', '!', '?', ':', ';']);
        if (end && cur.len() >= 20) || cur.len() >= 240 {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// The speech model to use: the loaded one, else the first speech-synthesis model in the library.
pub fn default_model(tts: &Tts, library: &[LibraryModel]) -> Option<LibraryModel> {
    let loaded = tts.loaded_id();
    library.iter().find(|m| Some(&m.id) == loaded.as_ref()).or_else(|| library.iter().find(|m| m.kind == "tts" && m.supported)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replies_are_read_sentence_by_sentence_without_markdown() {
        let s = sentences("**Paris** is the capital of France. It has about two million people! Ok.");
        assert_eq!(s, vec!["Paris is the capital of France.", "It has about two million people!", "Ok."]);
        let long = "word ".repeat(100);
        assert!(sentences(&long).iter().all(|p| p.len() <= 245));
        assert!(sentences("   ").is_empty());
    }

    /// The default speaker opens and drains queued audio; the audio is silence (test runs make no
    /// sound). `cargo test --release -p cerebro-desktop the_speaker_plays -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn the_speaker_plays_silence() {
        let p = Player::default();
        let secs = p.play(&vec![0.0; SAMPLE_RATE as usize / 2]).expect("speaker");
        assert!((secs - 0.5).abs() < 1e-6);
        let t = std::time::Instant::now();
        while p.remaining() > 0.0 && t.elapsed().as_secs() < 5 {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        println!("drained in {:.2} s", t.elapsed().as_secs_f64());
        assert_eq!(p.remaining(), 0.0);
    }

    /// A real sentence through the app's speech path (no playback): needs the Qwen3-TTS files.
    /// `cargo test --release -p cerebro-desktop speaks_a_sentence -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn speaks_a_sentence() {
        let m = crate::engine::scan_library_one(Path::new(r"C:\models\dev\tts\Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf")).expect("model");
        assert_eq!(m.kind, "tts");
        let tts = Tts::default();
        let x = tts.synthesize(&m, "Hello from Cerebro.").expect("speak");
        println!("{:.2} s", x.len() as f64 / SAMPLE_RATE as f64);
        assert!(x.len() > SAMPLE_RATE as usize / 2);
    }
}
