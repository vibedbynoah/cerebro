//! Speech to text: the microphone (push-to-talk in Chat, the model test in Models) and a speech
//! model (Qwen3-ASR: a language model with an audio projector beside it) kept loaded on the
//! processor, apart from the chat model, so dictation never unloads or slows the conversation.
//!
//! Capture runs on its own thread (an audio stream must stay on the thread that opened it): the
//! default input device, mixed to mono, at most MAX_SECONDS, while a recording is active.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use cerebro_engine::audio::{resample, AudioModel, SAMPLE_RATE};
use cerebro_engine::chat::{generate_parts, GenConfig};
use cerebro_engine::model::{ImageTokens, Model, Part, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;

use crate::engine::LibraryModel;

/// Longest recording kept, seconds.
pub const MAX_SECONDS: usize = 120;

enum Cmd {
    Start(Sender<Result<(), String>>),
    Stop(Sender<(Vec<f32>, u32)>),
}

/// The microphone: a capture thread started on first use.
pub struct Mic {
    tx: Mutex<Option<Sender<Cmd>>>,
    /// Recent peak level 0..1 (f32 bits), for the recording waveform.
    level: Arc<AtomicU32>,
    recording: Arc<AtomicBool>,
}

impl Default for Mic {
    fn default() -> Self {
        Mic { tx: Mutex::new(None), level: Arc::new(AtomicU32::new(0)), recording: Arc::new(AtomicBool::new(false)) }
    }
}

impl Mic {
    fn sender(&self) -> Sender<Cmd> {
        let mut g = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = g.as_ref() {
            return tx.clone();
        }
        let (tx, rx) = channel::<Cmd>();
        let (level, recording) = (self.level.clone(), self.recording.clone());
        std::thread::Builder::new()
            .name("cerebro-mic".into())
            .spawn(move || {
                use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
                let buf: Arc<Mutex<Vec<f32>>> = Arc::default();
                let mut stream: Option<cpal::Stream> = None;
                let mut rate = SAMPLE_RATE;
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Cmd::Start(reply) => {
                            buf.lock().unwrap_or_else(|e| e.into_inner()).clear();
                            let opened = (|| -> Result<(cpal::Stream, u32), String> {
                                let device = cpal::default_host().default_input_device().ok_or("No microphone was found.")?;
                                let cfg = device.default_input_config().map_err(|e| format!("The microphone could not be opened: {e}."))?;
                                let (channels, sr) = (cfg.channels() as usize, cfg.sample_rate().0);
                                let cap = sr as usize * MAX_SECONDS;
                                let (b, lv, rec) = (buf.clone(), level.clone(), recording.clone());
                                let push = move |data: &[f32]| {
                                    if !rec.load(Ordering::Relaxed) {
                                        return;
                                    }
                                    let mut g = b.lock().unwrap_or_else(|e| e.into_inner());
                                    let mut peak = 0f32;
                                    for frame in data.chunks(channels.max(1)) {
                                        let v = frame.iter().sum::<f32>() / frame.len() as f32;
                                        peak = peak.max(v.abs());
                                        if g.len() < cap {
                                            g.push(v);
                                        }
                                    }
                                    lv.store(peak.min(1.0).to_bits(), Ordering::Relaxed);
                                };
                                let err = |e| eprintln!("Microphone: {e}");
                                let s = match cfg.sample_format() {
                                    cpal::SampleFormat::F32 => device.build_input_stream(&cfg.config(), move |d: &[f32], _: &_| push(d), err, None),
                                    cpal::SampleFormat::I16 => {
                                        let mut tmp = Vec::new();
                                        device.build_input_stream(
                                            &cfg.config(),
                                            move |d: &[i16], _: &_| {
                                                tmp.clear();
                                                tmp.extend(d.iter().map(|&v| v as f32 / 32768.0));
                                                push(&tmp)
                                            },
                                            err,
                                            None,
                                        )
                                    }
                                    f => return Err(format!("The microphone's sample format ({f:?}) is not supported.")),
                                }
                                .map_err(|e| format!("The microphone could not be opened: {e}."))?;
                                s.play().map_err(|e| format!("The microphone could not start: {e}."))?;
                                Ok((s, sr))
                            })();
                            match opened {
                                Ok((s, sr)) => {
                                    recording.store(true, Ordering::SeqCst);
                                    stream = Some(s);
                                    rate = sr;
                                    let _ = reply.send(Ok(()));
                                }
                                Err(e) => {
                                    let _ = reply.send(Err(e));
                                }
                            }
                        }
                        Cmd::Stop(reply) => {
                            recording.store(false, Ordering::SeqCst);
                            stream = None;
                            level.store(0, Ordering::Relaxed);
                            let samples = std::mem::take(&mut *buf.lock().unwrap_or_else(|e| e.into_inner()));
                            let _ = reply.send((samples, rate));
                        }
                    }
                }
                drop(stream);
            })
            .expect("microphone thread");
        *g = Some(tx.clone());
        tx
    }

    /// Starts recording from the default microphone.
    pub fn start(&self) -> Result<(), String> {
        let (reply, rx) = channel();
        self.sender().send(Cmd::Start(reply)).map_err(|_| "The microphone stopped.".to_owned())?;
        rx.recv().map_err(|_| "The microphone stopped.".to_owned())?
    }

    /// Stops recording; the samples at 16 kHz mono.
    pub fn stop(&self) -> Result<Vec<f32>, String> {
        if !self.recording.load(Ordering::SeqCst) {
            return Err("Nothing is being recorded.".to_owned());
        }
        let (reply, rx) = channel();
        self.sender().send(Cmd::Stop(reply)).map_err(|_| "The microphone stopped.".to_owned())?;
        let (x, rate) = rx.recv().map_err(|_| "The microphone stopped.".to_owned())?;
        Ok(resample(&x, rate, SAMPLE_RATE))
    }

    /// The recent peak level, 0..1.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.level.load(Ordering::Relaxed))
    }

    pub fn recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }
}

struct Loaded {
    id: String,
    model: Model,
    tok: Tokenizer,
    session: Session,
    audio: AudioModel,
}

enum Job {
    Load(LibraryModel, Sender<Result<(), String>>),
    Transcribe(LibraryModel, Vec<f32>, Sender<Result<String, String>>),
}

/// The speech model, on its own thread (a session stays on the thread that made it), loaded on
/// the processor on first use or when the user loads one.
pub struct Stt {
    tx: Mutex<Sender<Job>>,
    loaded_id: Arc<Mutex<Option<String>>>,
}

impl Default for Stt {
    fn default() -> Self {
        let (tx, rx) = channel::<Job>();
        let loaded_id: Arc<Mutex<Option<String>>> = Arc::default();
        let id = loaded_id.clone();
        std::thread::Builder::new()
            .name("cerebro-stt".into())
            .spawn(move || {
                let mut loaded: Option<Loaded> = None;
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Load(m, reply) => {
                            let _ = reply.send(load(&mut loaded, &m, &id));
                        }
                        Job::Transcribe(m, samples, reply) => {
                            let r = load(&mut loaded, &m, &id).and_then(|_| transcribe(loaded.as_mut().expect("loaded"), &samples));
                            let _ = reply.send(r);
                        }
                    }
                }
            })
            .expect("speech thread");
        Stt { tx: Mutex::new(tx), loaded_id }
    }
}

/// Loads `m` (a speech model with its audio projector) unless it is the loaded one.
fn load(loaded: &mut Option<Loaded>, m: &LibraryModel, id: &Mutex<Option<String>>) -> Result<(), String> {
    if loaded.as_ref().is_some_and(|l| l.id == m.id) {
        return Ok(());
    }
    let proj = m.projector.as_ref().filter(|p| p.kind == "audio").ok_or("This model has no audio projector beside it, so it cannot transcribe.")?;
    *loaded = None;
    *id.lock().unwrap_or_else(|e| e.into_inner()) = None;
    let model = Model::load(Path::new(&m.path)).map_err(|e| format!("Could not load the speech model: {e}"))?;
    let tok = Tokenizer::from_gguf(model.gguf()).map_err(|e| format!("Could not read the speech model's tokenizer: {e}"))?;
    let audio = AudioModel::load(Path::new(&proj.path)).map_err(|e| format!("Could not load the audio projector: {e}"))?;
    // The processor: the graphics card belongs to the chat model.
    let session = Session::new(&model, 4096);
    *loaded = Some(Loaded { id: m.id.clone(), model, tok, session, audio });
    *id.lock().unwrap_or_else(|e| e.into_inner()) = Some(m.id.clone());
    Ok(())
}

fn transcribe(l: &mut Loaded, samples: &[f32]) -> Result<String, String> {
    let e = l.audio.encode(samples).map_err(|e| format!("Could not read the recording: {e}"))?;
    let before = l.tok.encode("<|im_start|>user
<|audio_start|>");
    let after = l.tok.encode("<|audio_end|>Transcribe.<|im_end|>
<|im_start|>assistant
");
    let parts = [Part::Text(&before), Part::Audio(ImageTokens { cols: e.tokens, rows: 1, dim: e.dim, data: &e.data }), Part::Text(&after)];
    let cfg = GenConfig { sampler: SamplerConfig { temperature: 0.0, ..Default::default() }, max_tokens: 512 };
    let mut out = String::new();
    generate_parts(&l.model, &mut l.session, &l.tok, &parts, &cfg, |p| {
        out.push_str(p);
        true
    })
    .map_err(|e| format!("Transcription failed: {e}"))?;
    Ok(asr_text(&out))
}

impl Stt {
    fn send<T>(&self, job: impl FnOnce(Sender<T>) -> Job) -> Result<T, String> {
        let (reply, rx) = channel();
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(job(reply)).map_err(|_| "The speech engine stopped.".to_owned())?;
        rx.recv().map_err(|_| "The speech engine stopped.".to_owned())
    }

    /// Loads `m` (a speech model with its audio projector) unless it is the loaded one.
    pub fn load(&self, m: &LibraryModel) -> Result<(), String> {
        self.send(|r| Job::Load(m.clone(), r))?
    }

    pub fn loaded_id(&self) -> Option<String> {
        self.loaded_id.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Transcribes 16 kHz mono samples with `m` (loading it if needed).
    pub fn transcribe(&self, m: &LibraryModel, samples: &[f32]) -> Result<String, String> {
        if samples.len() < SAMPLE_RATE as usize / 4 {
            return Err("The recording is too short to transcribe.".to_owned());
        }
        self.send(|r| Job::Transcribe(m.clone(), samples.to_vec(), r))?
    }
}

/// The transcript from the model's answer ("language English<asr_text>Hello." -> "Hello.").
pub fn asr_text(out: &str) -> String {
    let t = out.split_once("<asr_text>").map_or(out, |(_, t)| t);
    t.replace("</asr_text>", "").trim().to_owned()
}

/// The speech model to use: the loaded one, else the first speech model in the library.
pub fn default_model(stt: &Stt, library: &[LibraryModel]) -> Option<LibraryModel> {
    let loaded = stt.loaded_id();
    library
        .iter()
        .find(|m| Some(&m.id) == loaded.as_ref())
        .or_else(|| library.iter().find(|m| m.kind == "stt" && m.supported))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transcript_is_taken_from_the_answer() {
        assert_eq!(asr_text("language English<asr_text>The quick brown fox."), "The quick brown fox.");
        assert_eq!(asr_text("  Plain text.  "), "Plain text.");
        assert_eq!(asr_text("language None<asr_text></asr_text>"), "");
    }

    /// The default microphone opens and delivers about a second of 16 kHz audio (nothing is kept).
    /// `cargo test --release -p cerebro-desktop the_microphone_records -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn the_microphone_records() {
        let mic = Mic::default();
        mic.start().expect("microphone");
        std::thread::sleep(std::time::Duration::from_millis(1000));
        let level = mic.level();
        let x = mic.stop().expect("stop");
        println!("{} samples, level {level:.3}", x.len());
        assert!(x.len() > 12_000 && x.len() < 20_000, "{}", x.len());
        assert!(!mic.recording());
    }

    /// A real recording through the speech model: needs the files under .ref/audio.
    /// `cargo test --release -p cerebro-desktop transcribes_a_recording -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn transcribes_a_recording() {
        let model = crate::engine::scan_library_one(Path::new(r"C:\models\dev\asr\Qwen3-ASR-0.6B-Q8_0.gguf")).expect("model");
        assert_eq!(model.kind, "stt");
        let wav = std::fs::read(r"C:\Users\noahm\Projects\cerebro-engine\.ref\audio\speech1.wav").expect("recording");
        let (x, rate) = cerebro_engine::audio::read_wav(&wav).expect("wav");
        let x = resample(&x, rate, SAMPLE_RATE);
        let stt = Stt::default();
        let t = stt.transcribe(&model, &x).expect("transcribe");
        println!("{t}");
        assert_eq!(t, "The quick brown fox jumps over the lazy dog. Cerebro runs language models on this computer.");
    }
}
