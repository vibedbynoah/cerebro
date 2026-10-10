//! The inference engine inside the app: Cerebro Engine (our own; no external server or process).
//!
//! One worker thread owns the loaded model, its session (KV cache, GPU state) and tokenizer; IPC
//! commands send it jobs and read a shared status that the Monitor and status bar poll. Model
//! files are found by scanning Cerebro's, LM Studio's and the user's model folders; nothing is
//! moved or modified.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cerebro_engine::chat::{generate, GenConfig};
use cerebro_engine::gguf::Gguf;
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

const GIB: f64 = (1u64 << 30) as f64;

// ---- Library ------------------------------------------------------------------------------------

/// One model file. Mirrors `LibraryModel` in ui/src/service/types.ts.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LibraryModel {
    pub id: String,
    pub name: String,
    pub quant: String,
    pub params: String,
    pub architecture: &'static str,
    pub size_gi_b: f64,
    pub context_max: u64,
    pub tune: &'static str,
    pub provenance: &'static str,
    pub kind: &'static str,
    pub modalities: Vec<&'static str>,
    pub family: String,
    pub file: String,
    pub path: String,
    pub source: &'static str,
    pub sha256: String,
    pub arch: ModelArch,
    pub tuned_params: Option<()>,
    pub tuned_at: Option<String>,
    pub tune_result: Option<()>,
    pub projector: Option<()>,
    /// Whether Cerebro Engine can run this file's architecture today.
    pub supported: bool,
    pub arch_name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelArch {
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub embedding: u64,
    pub experts: Option<u64>,
    pub active_experts: Option<u64>,
}

/// The folders scanned for GGUF files, with the source label the UI shows.
fn model_dirs() -> Vec<(PathBuf, &'static str)> {
    let mut v = Vec::new();
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        v.push((PathBuf::from(local).join("Cerebro").join("models"), "cerebro"));
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        v.push((PathBuf::from(&home).join(".lmstudio").join("models"), "lmstudio"));
        v.push((PathBuf::from(&home).join(".cache").join("lm-studio").join("models"), "lmstudio"));
    }
    v.push((PathBuf::from("C:\\models"), "other"));
    v
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if depth < 6 {
                walk(&p, depth + 1, out);
            }
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) {
            let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            // Vision projectors are paired with their model, not listed; only the first part of a
            // split model is opened (the reader follows the rest).
            let split_tail = name.contains("-of-") && !name.contains("-00001-of-");
            if !name.starts_with("mmproj") && !split_tail {
                out.push(p);
            }
        }
    }
}

/// GGUF `general.file_type` to the quant label people use.
fn file_type_name(t: u64) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        7 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        _ => "other",
    }
}

const SUPPORTED_ARCHS: &[&str] = &["qwen3", "qwen3moe", "qwen2", "llama"];

/// A short, stable id from the path (FNV-1a).
fn id_of(path: &Path) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.to_string_lossy().to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("m{h:016x}")
}

fn describe(path: &Path, source: &'static str) -> Option<LibraryModel> {
    let size = std::fs::metadata(path).ok()?.len();
    let g = Gguf::open(path).ok()?;
    let arch = g.str("general.architecture").unwrap_or("unknown").to_owned();
    let k = |s: &str| format!("{arch}.{s}");
    let u = |s: &str| g.u64(&k(s)).ok();
    let n_head = u("attention.head_count").unwrap_or(0);
    let dim = u("embedding_length").unwrap_or(0);
    let experts = u("expert_count").filter(|&e| e > 0);
    let name = g.str("general.name").map(str::to_owned).unwrap_or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    let params = g.str("general.size_label").map(str::to_owned).unwrap_or_default();
    let quant = g.u64("general.file_type").map(file_type_name).unwrap_or("other").to_owned();
    let family = g.str("general.basename").map(str::to_owned).unwrap_or_else(|| arch.clone());
    Some(LibraryModel {
        id: id_of(path),
        name,
        quant,
        params,
        architecture: if experts.is_some() { "moe" } else { "dense" },
        size_gi_b: size as f64 / GIB,
        context_max: u("context_length").unwrap_or(0),
        tune: "untuned",
        provenance: "measured",
        kind: "text",
        modalities: vec!["text"],
        family,
        file: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        path: path.to_string_lossy().into_owned(),
        source,
        // Not a full-file hash (that would read many GB on every scan): identifies the file by
        // path, size and modification time for keying per-file settings.
        sha256: {
            let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            format!("{}{:016x}{:016x}", id_of(path), size, mtime)
        },
        arch: ModelArch {
            layers: u("block_count").unwrap_or(0),
            kv_heads: u("attention.head_count_kv").unwrap_or(n_head),
            head_dim: u("attention.key_length").unwrap_or(if n_head > 0 { dim / n_head } else { 0 }),
            embedding: dim,
            experts,
            active_experts: u("expert_used_count").filter(|_| experts.is_some()),
        },
        tuned_params: None,
        tuned_at: None,
        tune_result: None,
        projector: None,
        supported: SUPPORTED_ARCHS.contains(&arch.as_str()),
        arch_name: arch,
    })
}

pub fn scan_library() -> Vec<LibraryModel> {
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (dir, source) in model_dirs() {
        files.clear();
        walk(&dir, 0, &mut files);
        for f in &files {
            let key = f.to_string_lossy().to_lowercase();
            if seen.insert(key) {
                if let Some(m) = describe(f, source) {
                    out.push(m);
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

// ---- Hardware -------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_gi_b: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuInfo {
    pub name: String,
    pub cores: usize,
    pub threads: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    pub provenance: &'static str,
    pub cpu: CpuInfo,
    pub gpu: Option<GpuInfo>,
    pub ram_total_gi_b: f64,
}

pub fn hardware() -> HardwareProfile {
    let m = cerebro_engine::hw::machine();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    HardwareProfile {
        provenance: "measured",
        cpu: CpuInfo { name: m.cpu_name, cores: physical_cores().unwrap_or(cores), threads: m.cpu_threads },
        gpu: m.gpus.first().map(|g| GpuInfo { name: g.name.clone(), vram_total_gi_b: g.vram_total as f64 / GIB }),
        ram_total_gi_b: m.ram_total as f64 / GIB,
    }
}

fn physical_cores() -> Option<usize> {
    cerebro_probe::cpu::probe().ok().map(|t| t.physical_cores() as usize)
}

// ---- Engine worker --------------------------------------------------------------------------------

/// What the UI sends to start a reply. Mirrors `ChatRequest` (fields the engine acts on).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRequest {
    pub model_id: String,
    pub system_prompt: String,
    pub turns: Vec<ChatTurn>,
    pub sampler: ChatSampler,
    pub native: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSampler {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
}

/// Streamed to the UI. Mirrors `ChatEvent`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ChatEvent {
    Start {
        #[serde(rename = "ttftMs")]
        ttft_ms: f64,
    },
    Thinking {
        text: String,
    },
    Token {
        text: String,
    },
    Done {
        stats: ChatStats,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatStats {
    pub output_tokens: usize,
    pub thinking_tokens: usize,
    pub tok_per_sec: f64,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub extra_ms: f64,
    pub stopped: bool,
}

/// One finished or running request, for the Monitor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestRecord {
    pub id: String,
    pub started_at: u64,
    pub model: String,
    pub state: &'static str,
    pub wall_s: Option<f64>,
    pub first_token_s: Option<f64>,
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub output_tokens: usize,
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub input: String,
    pub output: String,
}

/// Shared status read by polling commands.
#[derive(Default)]
pub struct Status {
    pub loaded: Option<LoadedInfo>,
    pub activity: &'static str,
    pub last_decode_tps: Option<f64>,
    pub live_decode_tps: Option<f64>,
    pub live_generated: usize,
    pub live_prompt: usize,
    pub live_reused: usize,
    pub requests: VecDeque<RequestRecord>,
    pub totals: Totals,
    /// One sample per second: (decode tok/s, prefill tok/s, expert hit %, VRAM GiB, RAM GiB).
    pub history: VecDeque<[Option<f64>; 5]>,
}

#[derive(Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Totals {
    pub since: u64,
    pub requests: u64,
    pub errors: u64,
    pub prompt_tokens: u64,
    pub reused_tokens: u64,
    pub prompt_ms: f64,
    pub output_tokens: u64,
    pub decode_ms: f64,
}

#[derive(Clone)]
pub struct LoadedInfo {
    pub model: LibraryModel,
    pub ctx: usize,
    pub on_gpu: bool,
    pub vram_bytes: usize,
    /// (cached experts, total experts, hit rate 0..1).
    pub experts: Option<(usize, usize, f64)>,
    pub context_used: usize,
    pub last_prefill_tps: Option<f64>,
}

enum Job {
    Load { model: LibraryModel, ctx: usize, reply: Sender<Result<(), String>> },
    Unload { reply: Sender<()> },
    Chat { req: ChatRequest, events: Channel<ChatEvent>, reply: Sender<()> },
}

/// Cheap to clone: every field is shared.
#[derive(Clone)]
pub struct Engine {
    tx: Arc<Mutex<Sender<Job>>>,
    pub status: Arc<Mutex<Status>>,
    pub cancel: Arc<AtomicBool>,
    pub library: Arc<Mutex<Vec<LibraryModel>>>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

struct Loaded {
    model: Model,
    session: Session,
    tok: Tokenizer,
    info: LibraryModel,
}

impl Engine {
    pub fn start() -> Self {
        let (tx, rx) = channel::<Job>();
        let status = Arc::new(Mutex::new(Status { activity: "idle", totals: Totals { since: now_ms(), ..Default::default() }, ..Default::default() }));
        let cancel = Arc::new(AtomicBool::new(false));
        let st = status.clone();
        let cn = cancel.clone();
        std::thread::Builder::new()
            .name("cerebro-engine".into())
            .spawn(move || {
                let mut loaded: Option<Loaded> = None;
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Load { model, ctx, reply } => {
                            if let Some(l) = loaded.take() {
                                save_profile(&l);
                            }
                            set(&st, |s| {
                                s.loaded = None;
                                s.activity = "loading";
                            });
                            let r = load(&model, ctx);
                            let msg = match r {
                                Ok(l) => {
                                    let info = loaded_info(&l, ctx);
                                    loaded = Some(l);
                                    set(&st, |s| s.loaded = Some(info));
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            };
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(msg);
                        }
                        Job::Unload { reply } => {
                            if let Some(l) = loaded.take() {
                                save_profile(&l);
                            }
                            set(&st, |s| s.loaded = None);
                            let _ = reply.send(());
                        }
                        Job::Chat { req, events, reply } => {
                            cn.store(false, Ordering::SeqCst);
                            match loaded.as_mut() {
                                Some(l) if l.info.id == req.model_id => run_chat(l, &req, &events, &st, &cn),
                                _ => {
                                    let _ = events.send(ChatEvent::Error { message: "Load this model first (Models screen).".into() });
                                }
                            }
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(());
                        }
                    }
                }
            })
            .expect("engine thread");
        // Once a second: a history sample for the Monitor.
        let st = status.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let (total, avail) = cerebro_engine::hw::memory();
            let mut s = st.lock().unwrap_or_else(|e| e.into_inner());
            let gen = s.activity == "generating";
            let sample = [
                if gen { s.live_decode_tps } else { None },
                s.loaded.as_ref().and_then(|l| l.last_prefill_tps),
                s.loaded.as_ref().and_then(|l| l.experts).map(|e| e.2 * 100.0),
                s.loaded.as_ref().map(|l| l.vram_bytes as f64 / GIB),
                Some((total - avail) as f64 / GIB),
            ];
            s.history.push_back(sample);
            while s.history.len() > 24 * 3600 {
                s.history.pop_front();
            }
        });
        Self { tx: Arc::new(Mutex::new(tx)), status, cancel, library: Arc::new(Mutex::new(Vec::new())) }
    }

    /// A handle that can move into a blocking task.
    pub fn handle(&self) -> Self {
        self.clone()
    }

    fn send(&self, job: Job) -> Result<(), String> {
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(job).map_err(|_| "The engine has stopped.".to_owned())
    }

    pub fn load(&self, model: LibraryModel, ctx: usize) -> Result<(), String> {
        let (reply, rx) = channel();
        self.send(Job::Load { model, ctx, reply })?;
        rx.recv().map_err(|_| "The engine stopped while loading.".to_owned())?
    }

    pub fn unload(&self) -> Result<(), String> {
        let (reply, rx) = channel();
        self.send(Job::Unload { reply })?;
        rx.recv().map_err(|_| "The engine stopped while unloading.".to_owned())
    }

    pub fn chat(&self, req: ChatRequest, events: Channel<ChatEvent>) -> Result<(), String> {
        let (reply, rx) = channel();
        self.send(Job::Chat { req, events, reply })?;
        rx.recv().map_err(|_| "The engine stopped during the reply.".to_owned())
    }
}

fn set(st: &Arc<Mutex<Status>>, f: impl FnOnce(&mut Status)) {
    let mut s = st.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut s);
}

fn load(m: &LibraryModel, ctx: usize) -> Result<Loaded, String> {
    if !m.supported {
        return Err(format!("Cerebro Engine does not run the {} architecture yet.", m.arch_name));
    }
    let model = Model::load(Path::new(&m.path)).map_err(|e| format!("Could not load the model: {e}"))?;
    let tok = Tokenizer::from_gguf(model.gguf()).map_err(|e| format!("Could not read the tokenizer: {e}"))?;
    let ctx = ctx.clamp(512, model.cfg.ctx_train.max(512));
    let session = match Session::new_gpu(&model, ctx) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GPU unavailable for this model ({e}); running on the CPU");
            Session::new(&model, ctx)
        }
    };
    Ok(Loaded { model, session, tok, info: m.clone() })
}

fn loaded_info(l: &Loaded, ctx: usize) -> LoadedInfo {
    let g = l.session.gpu.as_ref();
    LoadedInfo {
        model: l.info.clone(),
        ctx,
        on_gpu: g.is_some(),
        vram_bytes: g.map_or(0, |g| g.vram_bytes),
        experts: g.and_then(|g| g.expert_cache()),
        context_used: l.session.tokens.len(),
        last_prefill_tps: None,
    }
}

fn save_profile(l: &Loaded) {
    if let Some(g) = l.session.gpu.as_ref() {
        g.save_expert_profile();
    }
}

/// ChatML prompt for Qwen-family models; `think` leaves reasoning on.
fn prompt_text(req: &ChatRequest, think: bool) -> String {
    let sys = (!req.system_prompt.trim().is_empty()).then_some(req.system_prompt.as_str());
    let turns: Vec<(&str, &str)> = req.turns.iter().map(|t| (if t.role == "assistant" { "assistant" } else { "user" }, t.text.as_str())).collect();
    cerebro_engine::chatml(sys, &turns, think)
}

fn run_chat(l: &mut Loaded, req: &ChatRequest, events: &Channel<ChatEvent>, st: &Arc<Mutex<Status>>, cancel: &Arc<AtomicBool>) {
    let think = req.native.is_some();
    let text = prompt_text(req, think);
    let ids = l.tok.encode(&text);
    let id = format!("r{}", now_ms());
    let started = now_ms();
    let input = req.turns.last().map(|t| t.text.clone()).unwrap_or_default();
    set(st, |s| {
        s.activity = "reading";
        s.live_prompt = ids.len();
        s.live_generated = 0;
        s.live_decode_tps = None;
        s.requests.push_front(RequestRecord {
            id: id.clone(),
            started_at: started,
            model: l.info.name.clone(),
            state: "reading",
            wall_s: None,
            first_token_s: None,
            prompt_tokens: ids.len(),
            reused_tokens: 0,
            output_tokens: 0,
            decode_tps: None,
            prefill_tps: None,
            input: input.chars().take(256 * 1024).collect(),
            output: String::new(),
        });
        while s.requests.len() > 100 {
            s.requests.pop_back();
        }
    });
    let cfg = GenConfig {
        sampler: SamplerConfig { temperature: req.sampler.temperature, top_k: req.sampler.top_k, top_p: req.sampler.top_p, min_p: req.sampler.min_p, repeat_penalty: req.sampler.repeat_penalty, seed: now_ms() },
        max_tokens: req.sampler.max_tokens.max(1),
    };
    // Reasoning (Qwen3: <think> ... </think>) streams as `thinking`, the rest as `token`.
    let mut in_think = think;
    let mut pending = String::new();
    let mut thinking_tokens = 0usize;
    let mut answer = String::new();
    let mut started_ev = false;
    let t0 = Instant::now();
    let mut n = 0usize;
    let result = generate(&l.model, &mut l.session, &l.tok, &ids, &cfg, |piece| {
        if !started_ev {
            started_ev = true;
            let _ = events.send(ChatEvent::Start { ttft_ms: t0.elapsed().as_secs_f64() * 1000.0 });
            set(st, |s| s.activity = "generating");
        }
        n += 1;
        pending.push_str(piece);
        loop {
            if in_think {
                if let Some(i) = pending.find("</think>") {
                    let t = pending[..i].replace("<think>", "");
                    if !t.is_empty() {
                        let _ = events.send(ChatEvent::Thinking { text: t });
                    }
                    pending = pending[i + 8..].trim_start_matches('\n').to_owned();
                    in_think = false;
                    continue;
                }
                // Hold back a tail that could be the start of "</think>".
                let hold = (1..8).rev().find(|&h| h <= pending.len() && pending.is_char_boundary(pending.len() - h) && "</think>".starts_with(&pending[pending.len() - h..])).unwrap_or(0);
                let cut = pending.len() - hold;
                if cut > 0 {
                    let t = pending[..cut].replace("<think>", "");
                    if !t.is_empty() {
                        let _ = events.send(ChatEvent::Thinking { text: t });
                    }
                    thinking_tokens += 1;
                    pending = pending[cut..].to_owned();
                }
            } else if !pending.is_empty() {
                answer.push_str(&pending);
                let _ = events.send(ChatEvent::Token { text: std::mem::take(&mut pending) });
            }
            break;
        }
        let secs = t0.elapsed().as_secs_f64();
        set(st, |s| {
            s.live_generated = n;
            s.live_decode_tps = Some(n as f64 / secs.max(1e-3));
        });
        !cancel.load(Ordering::SeqCst)
    });
    if !pending.is_empty() {
        let ev = if in_think { ChatEvent::Thinking { text: pending.clone() } } else { ChatEvent::Token { text: pending.clone() } };
        answer.push_str(&pending);
        let _ = events.send(ev);
    }
    match result {
        Ok(g) => {
            let _ = events.send(ChatEvent::Done {
                stats: ChatStats { output_tokens: g.output_tokens, thinking_tokens, tok_per_sec: g.decode_tps, ttft_ms: g.ttft_ms, total_ms: g.total_ms, extra_ms: 0.0, stopped: g.stopped },
            });
            let info = loaded_info(l, l.session.max_ctx);
            set(st, |s| {
                s.last_decode_tps = Some(g.decode_tps);
                s.live_decode_tps = None;
                s.loaded = Some(LoadedInfo { last_prefill_tps: Some(g.prefill_tps), ..info });
                s.totals.requests += 1;
                s.totals.prompt_tokens += g.prompt_tokens as u64;
                s.totals.reused_tokens += g.reused_tokens as u64;
                s.totals.output_tokens += g.output_tokens as u64;
                s.totals.decode_ms += g.total_ms - g.ttft_ms;
                s.totals.prompt_ms += g.ttft_ms;
                if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
                    r.state = if g.stopped { "cancelled" } else { "completed" };
                    r.wall_s = Some(g.total_ms / 1000.0);
                    r.first_token_s = Some(g.ttft_ms / 1000.0);
                    r.reused_tokens = g.reused_tokens;
                    r.output_tokens = g.output_tokens;
                    r.decode_tps = Some(g.decode_tps);
                    r.prefill_tps = Some(g.prefill_tps);
                    r.output = answer.chars().take(256 * 1024).collect();
                }
            });
        }
        Err(e) => {
            let _ = events.send(ChatEvent::Error { message: format!("Generation failed: {e}") });
            set(st, |s| {
                s.totals.errors += 1;
                if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
                    r.state = "error";
                }
            });
        }
    }
}
