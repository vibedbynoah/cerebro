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
    /// MCP tools the model may call this turn, with the user's rule for each; None = no tools.
    #[serde(default)]
    pub tools: Option<ToolsRequest>,
    /// Orchestrate mode (GVS5H, orchestrate.rs): several instances of the model and a shared ledger.
    #[serde(default)]
    pub orchestrate: Option<OrchestrateRequest>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrchestrateRequest {
    pub workers: usize,
    pub rounds: usize,
}

/// The user's rules (Tools screen), resolved by the UI to allow / ask / deny per "server/tool".
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsRequest {
    #[serde(default)]
    pub rules: std::collections::HashMap<String, String>,
    pub default_rule: String,
    /// Offline mode is on (Settings): built-in tools that touch the network must refuse.
    #[serde(default)]
    pub offline: bool,
    /// The user turned off "Keep web fetches off this network" (Settings > Protections).
    #[serde(default)]
    pub allow_local_fetch: bool,
}

impl ToolsRequest {
    fn rule(&self, server: &str, tool: &str) -> &str {
        self.rules.get(&format!("{server}/{tool}")).map_or(self.default_rule.as_str(), String::as_str)
    }
}

/// What a tool call needs to know about the user's settings for this turn.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolContext {
    /// Offline mode is on (Settings): the tool must not touch the network.
    pub offline: bool,
    /// A fetch may reach loopback, private and link-local addresses.
    pub allow_local_fetch: bool,
}

/// Where tool calls go: the MCP client and the built-in tools (commands.rs). Blocking; called from
/// the engine thread.
pub trait ToolHost: Send + Sync {
    fn tools(&self) -> Vec<crate::toolcall::ToolSpec>;
    /// The tool's result as text for the model, or an error message.
    fn call(&self, server: &str, tool: &str, arguments: serde_json::Value, ctx: ToolContext) -> Result<String, String>;
}

/// Most tool rounds in one reply; a model that keeps calling tools past this is stopped.
const MAX_TOOL_ROUNDS: usize = 8;
/// A tool result longer than this is cut before it goes back to the model.
const MAX_TOOL_RESULT_CHARS: usize = 16_000;
/// How long an approval waits for the user before it counts as Deny.
const APPROVAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15 * 60);

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

/// Where a reply's events go: the window's IPC channel, or a paired phone (remote.rs).
pub type ChatSink = Arc<dyn Fn(ChatEvent) + Send + Sync>;

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
    /// Orchestrate mode: a role started (`step`), or a ledger file changed (`file` + `content`).
    Ledger {
        step: Option<String>,
        round: usize,
        file: Option<&'static str>,
        content: Option<String>,
        /// The ledger folder on disk (first event only).
        dir: Option<String>,
    },
    /// A tool call and each change of its state: approval, running, done, denied, error.
    Tool {
        id: String,
        server: String,
        name: String,
        arguments: serde_json::Value,
        state: &'static str,
        result: Option<String>,
        error: Option<String>,
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
    Chat { req: ChatRequest, events: ChatSink, reply: Sender<()> },
}

/// Cheap to clone: every field is shared.
#[derive(Clone)]
pub struct Engine {
    tx: Arc<Mutex<Sender<Job>>>,
    pub status: Arc<Mutex<Status>>,
    pub cancel: Arc<AtomicBool>,
    pub library: Arc<Mutex<Vec<LibraryModel>>>,
    /// Tool calls waiting for the user, by call id; `decide` answers them.
    approvals: Arc<Mutex<std::collections::HashMap<String, Sender<bool>>>>,
    tool_host: Arc<Mutex<Option<Arc<dyn ToolHost>>>>,
    /// Where Orchestrate mode keeps its ledgers (the app's data folder).
    ledger_root: Arc<Mutex<Option<PathBuf>>>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

struct Loaded {
    model: Model,
    session: Session,
    tok: Tokenizer,
    info: LibraryModel,
    /// How this model writes tool calls, read from its chat template.
    dialect: crate::toolcall::Dialect,
}

impl Engine {
    pub fn start() -> Self {
        let (tx, rx) = channel::<Job>();
        let status = Arc::new(Mutex::new(Status { activity: "idle", totals: Totals { since: now_ms(), ..Default::default() }, ..Default::default() }));
        let cancel = Arc::new(AtomicBool::new(false));
        let approvals: Arc<Mutex<std::collections::HashMap<String, Sender<bool>>>> = Arc::default();
        let tool_host: Arc<Mutex<Option<Arc<dyn ToolHost>>>> = Arc::default();
        let st = status.clone();
        let cn = cancel.clone();
        let ledger_root: Arc<Mutex<Option<PathBuf>>> = Arc::default();
        let (ap, th, lr) = (approvals.clone(), tool_host.clone(), ledger_root.clone());
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
                                Some(l) if l.info.id == req.model_id && req.orchestrate.is_some() => {
                                    let root = lr.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(std::env::temp_dir);
                                    run_orchestrate(l, &req, &events, &st, &cn, &root);
                                }
                                Some(l) if l.info.id == req.model_id => {
                                    let host = th.lock().unwrap_or_else(|e| e.into_inner()).clone();
                                    run_chat(l, &req, &events, &st, &cn, Tooling { host, approvals: &ap });
                                }
                                _ => {
                                    events(ChatEvent::Error { message: "Load this model first (Models screen).".into() });
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
        Self { tx: Arc::new(Mutex::new(tx)), status, cancel, library: Arc::new(Mutex::new(Vec::new())), approvals, tool_host, ledger_root }
    }

    /// A handle that can move into a blocking task.
    pub fn handle(&self) -> Self {
        self.clone()
    }

    /// Where chat tool calls go (set once at start-up, to the MCP client).
    pub fn set_tool_host(&self, host: Arc<dyn ToolHost>) {
        *self.tool_host.lock().unwrap_or_else(|e| e.into_inner()) = Some(host);
    }

    /// Where Orchestrate ledgers are written (set once at start-up).
    pub fn set_ledger_root(&self, dir: PathBuf) {
        *self.ledger_root.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir);
    }

    /// The user's answer to a tool call waiting for approval. False if nothing was waiting.
    pub fn decide(&self, id: &str, allow: bool) -> bool {
        let tx = self.approvals.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
        tx.is_some_and(|tx| tx.send(allow).is_ok())
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

    pub fn chat(&self, req: ChatRequest, events: ChatSink) -> Result<(), String> {
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
    let dialect = crate::toolcall::dialect_from_template(model.gguf().str("tokenizer.chat_template").unwrap_or(""));
    Ok(Loaded { model, session, tok, info: m.clone(), dialect })
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

/// What run_chat needs to run tools: the MCP host (if any) and the approvals waiting for the user.
struct Tooling<'a> {
    host: Option<Arc<dyn ToolHost>>,
    approvals: &'a Arc<Mutex<std::collections::HashMap<String, Sender<bool>>>>,
}

/// One round's output: what the user saw, the raw text (tool markup included) for the history, and
/// the calls the model made.
struct Round {
    raw: String,
    calls: Vec<crate::toolcall::ToolCall>,
    bad: Vec<(String, String)>,
}

fn run_chat(l: &mut Loaded, req: &ChatRequest, events: &ChatSink, st: &Arc<Mutex<Status>>, cancel: &Arc<AtomicBool>, tooling: Tooling<'_>) {
    use crate::toolcall::{self, Piece, Scanner};
    let think = req.native.is_some();
    let ctx = ToolContext { offline: req.tools.as_ref().is_some_and(|t| t.offline), allow_local_fetch: req.tools.as_ref().is_some_and(|t| t.allow_local_fetch) };
    // Tools the model may call: the host's list, names made unique across servers ("server__tool").
    let tools: Vec<(String, toolcall::ToolSpec)> = match (&req.tools, &tooling.host) {
        (Some(_), Some(h)) => {
            let specs = h.tools();
            let mut seen = std::collections::HashMap::<String, usize>::new();
            for t in &specs {
                *seen.entry(t.name.clone()).or_default() += 1;
            }
            specs
                .into_iter()
                .map(|t| {
                    let exposed = if seen.get(&t.name).copied().unwrap_or(0) > 1 { format!("{}__{}", t.server, t.name) } else { t.name.clone() };
                    (exposed.clone(), toolcall::ToolSpec { name: exposed, ..t })
                })
                .collect()
        }
        _ => Vec::new(),
    };
    let specs: Vec<toolcall::ToolSpec> = tools.iter().map(|(_, t)| t.clone()).collect();
    let system = {
        let mut sys = req.system_prompt.trim().to_owned();
        if !specs.is_empty() {
            if !sys.is_empty() {
                sys.push_str("\n\n");
            }
            sys.push_str(&toolcall::tools_prompt(l.dialect, &specs));
        }
        sys
    };
    let mut convo: Vec<(String, String)> = req.turns.iter().map(|t| (if t.role == "assistant" { "assistant" } else { "user" }.to_owned(), t.text.clone())).collect();

    let id = format!("r{}", now_ms());
    let started = now_ms();
    let input = req.turns.last().map(|t| t.text.clone()).unwrap_or_default();
    set(st, |s| {
        s.activity = "reading";
        s.live_generated = 0;
        s.live_decode_tps = None;
        s.requests.push_front(RequestRecord {
            id: id.clone(),
            started_at: started,
            model: l.info.name.clone(),
            state: "reading",
            wall_s: None,
            first_token_s: None,
            prompt_tokens: 0,
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

    let t0 = Instant::now();
    let mut started_ev = false;
    let mut answer = String::new();
    let (mut thinking_tokens, mut out_tokens, mut prompt_tokens, mut reused_tokens) = (0usize, 0usize, 0usize, 0usize);
    let (mut ttft_ms, mut decode_ms, mut prefill_tps, mut stopped) = (0.0f64, 0.0f64, 0.0f64, false);
    let mut failed: Option<String> = None;
    let mut n = 0usize;

    for round in 0..=MAX_TOOL_ROUNDS {
        let turns: Vec<(&str, &str)> = convo.iter().map(|(r, t)| (r.as_str(), t.as_str())).collect();
        let text = cerebro_engine::chatml((!system.is_empty()).then_some(system.as_str()), &turns, think);
        let ids = l.tok.encode(&text);
        set(st, |s| s.live_prompt = ids.len());
        // Reasoning (Qwen3: <think> ... </think>) streams as `thinking`; the answer goes through the
        // tool-call scanner, so tool markup never reaches the screen.
        let mut in_think = think;
        let mut pending = String::new();
        let mut scanner = Scanner::new(l.dialect, &specs);
        let mut out = Round { raw: String::new(), calls: Vec::new(), bad: Vec::new() };
        let emit = |pieces: Vec<Piece>, out: &mut Round, answer: &mut String| {
            for p in pieces {
                match p {
                    Piece::Text(t) => {
                        answer.push_str(&t);
                        events(ChatEvent::Token { text: t });
                    }
                    Piece::Call(c) => out.calls.push(c),
                    Piece::BadCall { raw, error } => out.bad.push((raw, error)),
                }
            }
        };
        let result = generate(&l.model, &mut l.session, &l.tok, &ids, &cfg, |piece| {
            if !started_ev {
                started_ev = true;
                events(ChatEvent::Start { ttft_ms: t0.elapsed().as_secs_f64() * 1000.0 });
            }
            set(st, |s| s.activity = "generating");
            n += 1;
            pending.push_str(piece);
            loop {
                if in_think {
                    if let Some(i) = pending.find("</think>") {
                        let t = pending[..i].replace("<think>", "");
                        if !t.is_empty() {
                            events(ChatEvent::Thinking { text: t });
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
                            events(ChatEvent::Thinking { text: t });
                        }
                        thinking_tokens += 1;
                        pending = pending[cut..].to_owned();
                    }
                } else if !pending.is_empty() {
                    let t = std::mem::take(&mut pending);
                    out.raw.push_str(&t);
                    emit(scanner.feed(&t), &mut out, &mut answer);
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
            if in_think {
                events(ChatEvent::Thinking { text: std::mem::take(&mut pending) });
            } else {
                let t = std::mem::take(&mut pending);
                out.raw.push_str(&t);
                emit(scanner.feed(&t), &mut out, &mut answer);
            }
        }
        emit(scanner.finish(), &mut out, &mut answer);
        match result {
            Ok(g) => {
                if round == 0 {
                    ttft_ms = g.ttft_ms;
                }
                out_tokens += g.output_tokens;
                prompt_tokens += g.prompt_tokens;
                reused_tokens += g.reused_tokens;
                decode_ms += g.total_ms - g.ttft_ms;
                prefill_tps = g.prefill_tps;
                stopped = g.stopped;
            }
            Err(e) => {
                failed = Some(format!("Generation failed: {e}"));
                break;
            }
        }
        if stopped || cancel.load(Ordering::SeqCst) || (out.calls.is_empty() && out.bad.is_empty()) || specs.is_empty() {
            break;
        }
        if round == MAX_TOOL_ROUNDS {
            events(ChatEvent::Token { text: format!("\n\n(Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls.)") });
            break;
        }
        // Run this round's calls, then hand the results back as the next user turn.
        convo.push(("assistant".to_owned(), out.raw.trim().to_owned()));
        let mut responses = Vec::new();
        for (raw, error) in &out.bad {
            responses.push(toolcall::tool_response(&format!("Error: the tool call could not be read ({error}). Call it again with valid JSON.")));
            let _ = raw;
        }
        for call in &out.calls {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let call_id = format!("t{}-{}", now_ms(), responses.len());
            let Some((server, tool)) = tools.iter().find(|(exposed, _)| *exposed == call.name).map(|(_, t)| (t.server.clone(), original_name(&t.server, &call.name))) else {
                events(ChatEvent::Tool { id: call_id, server: String::new(), name: call.name.clone(), arguments: call.arguments.clone(), state: "error", result: None, error: Some("No such tool.".into()) });
                responses.push(toolcall::tool_response(&format!("Error: there is no tool named {}.", call.name)));
                continue;
            };
            let tool_ev = |state: &'static str, result: Option<String>, error: Option<String>| ChatEvent::Tool {
                id: call_id.clone(),
                server: server.clone(),
                name: tool.clone(),
                arguments: call.arguments.clone(),
                state,
                result,
                error,
            };
            let rule = req.tools.as_ref().map_or("deny", |t| t.rule(&server, &tool));
            let allowed = match rule {
                "allow" => true,
                "ask" => {
                    let (tx, rx) = channel::<bool>();
                    tooling.approvals.lock().unwrap_or_else(|e| e.into_inner()).insert(call_id.clone(), tx);
                    events(tool_ev("approval", None, None));
                    set(st, |s| s.activity = "waiting");
                    let deadline = Instant::now() + APPROVAL_TIMEOUT;
                    let decision = loop {
                        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(d) => break d,
                            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if !cancel.load(Ordering::SeqCst) && Instant::now() < deadline => continue,
                            Err(_) => break false,
                        }
                    };
                    tooling.approvals.lock().unwrap_or_else(|e| e.into_inner()).remove(&call_id);
                    decision
                }
                _ => false,
            };
            if !allowed {
                events(tool_ev("denied", None, None));
                responses.push(toolcall::tool_response("The user did not allow this tool call. Do not call it again; answer without it."));
                continue;
            }
            events(tool_ev("running", None, None));
            set(st, |s| s.activity = "tool");
            let Some(host) = tooling.host.as_ref() else { break };
            match host.call(&server, &tool, call.arguments.clone(), ctx) {
                Ok(text) => {
                    let text = toolcall::truncate_result(&text, MAX_TOOL_RESULT_CHARS);
                    events(tool_ev("done", Some(text.clone()), None));
                    responses.push(toolcall::tool_response(&text));
                }
                Err(e) => {
                    events(tool_ev("error", None, Some(e.clone())));
                    responses.push(toolcall::tool_response(&format!("Error: {e}")));
                }
            }
        }
        if cancel.load(Ordering::SeqCst) {
            stopped = true;
            break;
        }
        convo.push(("user".to_owned(), responses.join("\n")));
        // Text before the call and text after it are separate paragraphs; no blank lead-in when the
        // model went straight to the tool.
        if !answer.trim().is_empty() && !answer.ends_with("\n\n") {
            answer.push_str("\n\n");
            events(ChatEvent::Token { text: "\n\n".into() });
        }
    }

    if let Some(message) = failed {
        events(ChatEvent::Error { message });
        set(st, |s| {
            s.totals.errors += 1;
            if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
                r.state = "error";
            }
        });
        return;
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = if decode_ms > 0.0 { out_tokens as f64 / (decode_ms / 1000.0) } else { 0.0 };
    events(ChatEvent::Done {
        stats: ChatStats { output_tokens: out_tokens, thinking_tokens, tok_per_sec: decode_tps, ttft_ms, total_ms, extra_ms: 0.0, stopped },
    });
    let info = loaded_info(l, l.session.max_ctx);
    set(st, |s| {
        s.last_decode_tps = Some(decode_tps);
        s.live_decode_tps = None;
        s.loaded = Some(LoadedInfo { last_prefill_tps: Some(prefill_tps), ..info });
        s.totals.requests += 1;
        s.totals.prompt_tokens += prompt_tokens as u64;
        s.totals.reused_tokens += reused_tokens as u64;
        s.totals.output_tokens += out_tokens as u64;
        s.totals.decode_ms += decode_ms;
        s.totals.prompt_ms += ttft_ms;
        if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
            r.state = if stopped { "cancelled" } else { "completed" };
            r.wall_s = Some(total_ms / 1000.0);
            r.first_token_s = Some(ttft_ms / 1000.0);
            r.prompt_tokens = prompt_tokens;
            r.reused_tokens = reused_tokens;
            r.output_tokens = out_tokens;
            r.decode_tps = Some(decode_tps);
            r.prefill_tps = Some(prefill_tps);
            r.output = answer.chars().take(256 * 1024).collect();
        }
    });
}

/// The engine as an orchestration participant: every call is a fresh context on the loaded model.
/// Each role's output streams to the reply's reasoning, headed by the role, so the discussion is
/// visible as it happens.
struct EngineLlm<'a> {
    l: &'a mut Loaded,
    cfg: GenConfig,
    think: bool,
    events: &'a ChatSink,
    st: &'a Arc<Mutex<Status>>,
    cancel: &'a Arc<AtomicBool>,
    t0: Instant,
    started: bool,
    tokens: usize,
    decode_ms: f64,
    prompt_tokens: usize,
    reused_tokens: usize,
    ttft_ms: Option<f64>,
}

impl crate::orchestrate::Llm for EngineLlm<'_> {
    fn call(&mut self, role: &str, system: &str, user: &str, temperature: f32) -> Result<crate::orchestrate::Reply, String> {
        let text = cerebro_engine::chatml(Some(system), &[("user", user)], self.think);
        let ids = self.l.tok.encode(&text);
        self.cfg.sampler.temperature = temperature;
        self.cfg.sampler.seed = now_ms();
        let cfg = &self.cfg;
        (self.events)(ChatEvent::Thinking { text: format!("\n\n**{role}**\n") });
        set(self.st, |s| {
            s.activity = "reading";
            s.live_prompt = ids.len();
        });
        let mut out = String::new();
        let (events, st, cancel, t0) = (self.events, self.st, self.cancel, self.t0);
        let mut started = self.started;
        let mut n = 0usize;
        let r = generate(&self.l.model, &mut self.l.session, &self.l.tok, &ids, cfg, |piece| {
            if !started {
                started = true;
                events(ChatEvent::Start { ttft_ms: t0.elapsed().as_secs_f64() * 1000.0 });
            }
            n += 1;
            out.push_str(piece);
            events(ChatEvent::Thinking { text: piece.to_owned() });
            let secs = t0.elapsed().as_secs_f64();
            set(st, |s| {
                s.activity = "generating";
                s.live_generated += 1;
                s.live_decode_tps = Some(s.live_generated as f64 / secs.max(1e-3));
            });
            !cancel.load(Ordering::SeqCst)
        });
        self.started = started;
        let g = r.map_err(|e| format!("Generation failed: {e}"))?;
        self.ttft_ms.get_or_insert(g.ttft_ms);
        self.tokens += g.output_tokens;
        self.decode_ms += g.total_ms - g.ttft_ms;
        self.prompt_tokens += g.prompt_tokens;
        self.reused_tokens += g.reused_tokens;
        if g.stopped && self.cancel.load(Ordering::SeqCst) {
            return Err("Stopped.".into());
        }
        // A reply that used all of max_tokens was cut off.
        Ok(crate::orchestrate::Reply { text: crate::orchestrate::strip_think(&out), cut_off: g.output_tokens >= cfg.max_tokens })
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// Orchestrate mode: GVS5H over the loaded model; the final result is the reply.
fn run_orchestrate(l: &mut Loaded, req: &ChatRequest, events: &ChatSink, st: &Arc<Mutex<Status>>, cancel: &Arc<AtomicBool>, root: &Path) {
    let Some(o) = req.orchestrate else { return };
    // The goal is the last user message; earlier turns are context.
    let goal = req.turns.iter().rev().find(|t| t.role != "assistant").map(|t| t.text.clone()).unwrap_or_default();
    let mut context = req.turns[..req.turns.len().saturating_sub(1)].iter().map(|t| format!("{}: {}", t.role, t.text)).collect::<Vec<_>>().join("\n\n");
    if !req.system_prompt.trim().is_empty() {
        context = format!("Instructions: {}\n\n{context}", req.system_prompt.trim());
    }
    let id = format!("r{}", now_ms());
    let dir = root.join("orchestrate").join(&id);
    let ledger = match crate::orchestrate::Ledger::create(&dir, &goal) {
        Ok(x) => x,
        Err(e) => return events(ChatEvent::Error { message: e }),
    };
    events(ChatEvent::Ledger { step: None, round: 0, file: None, content: None, dir: Some(ledger.dir().to_string_lossy().into_owned()) });
    set(st, |s| {
        s.live_generated = 0;
        s.live_decode_tps = None;
        s.requests.push_front(RequestRecord {
            id: id.clone(),
            started_at: now_ms(),
            model: l.info.name.clone(),
            state: "generating",
            wall_s: None,
            first_token_s: None,
            prompt_tokens: 0,
            reused_tokens: 0,
            output_tokens: 0,
            decode_tps: None,
            prefill_tps: None,
            input: goal.chars().take(256 * 1024).collect(),
            output: String::new(),
        });
        while s.requests.len() > 100 {
            s.requests.pop_back();
        }
    });
    let cfg = GenConfig {
        sampler: SamplerConfig { temperature: 0.2, top_k: req.sampler.top_k, top_p: req.sampler.top_p, min_p: req.sampler.min_p, repeat_penalty: req.sampler.repeat_penalty, seed: now_ms() },
        max_tokens: req.sampler.max_tokens.max(256),
    };
    let t0 = Instant::now();
    let mut llm = EngineLlm { l, cfg, think: req.native.is_some(), events, st, cancel, t0, started: false, tokens: 0, decode_ms: 0.0, prompt_tokens: 0, reused_tokens: 0, ttft_ms: None };
    let opts = crate::orchestrate::Options { workers: o.workers, rounds: o.rounds };
    let result = crate::orchestrate::run(&mut llm, &ledger, &goal, &context, opts, &mut |p| match p {
        crate::orchestrate::Progress::Step { role, round } => events(ChatEvent::Ledger { step: Some(role.to_owned()), round, file: None, content: None, dir: None }),
        crate::orchestrate::Progress::File { name, content } => events(ChatEvent::Ledger { step: None, round: 0, file: Some(name), content: Some(content.to_owned()), dir: None }),
    });
    let stopped = cancel.load(Ordering::SeqCst);
    let (tokens, decode_ms, prompt_tokens, reused_tokens, ttft) = (llm.tokens, llm.decode_ms, llm.prompt_tokens, llm.reused_tokens, llm.ttft_ms.unwrap_or(0.0));
    let answer = match result {
        Ok(text) => text,
        Err(_) if stopped => std::fs::read_to_string(ledger.dir().join("result.md")).unwrap_or_default(),
        Err(e) => {
            events(ChatEvent::Error { message: e });
            set(st, |s| {
                s.totals.errors += 1;
                if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
                    r.state = "error";
                }
            });
            return;
        }
    };
    for chunk in answer.split_inclusive(' ') {
        events(ChatEvent::Token { text: chunk.to_owned() });
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tps = if decode_ms > 0.0 { tokens as f64 / (decode_ms / 1000.0) } else { 0.0 };
    events(ChatEvent::Done { stats: ChatStats { output_tokens: tokens, thinking_tokens: 0, tok_per_sec: tps, ttft_ms: ttft, total_ms, extra_ms: 0.0, stopped } });
    set(st, |s| {
        s.last_decode_tps = Some(tps);
        s.live_decode_tps = None;
        s.totals.requests += 1;
        s.totals.prompt_tokens += prompt_tokens as u64;
        s.totals.reused_tokens += reused_tokens as u64;
        s.totals.output_tokens += tokens as u64;
        s.totals.decode_ms += decode_ms;
        if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
            r.state = if stopped { "cancelled" } else { "completed" };
            r.wall_s = Some(total_ms / 1000.0);
            r.first_token_s = Some(ttft / 1000.0);
            r.prompt_tokens = prompt_tokens;
            r.reused_tokens = reused_tokens;
            r.output_tokens = tokens;
            r.decode_tps = Some(tps);
            r.output = answer.chars().take(256 * 1024).collect();
        }
    });
}

/// The tool's own name from the name the model saw ("server__tool" when names collided).
fn original_name(server: &str, exposed: &str) -> String {
    exposed.strip_prefix(&format!("{server}__")).unwrap_or(exposed).to_owned()
}

#[cfg(test)]
mod tool_loop_tests {
    use super::*;

    struct Weather;
    impl ToolHost for Weather {
        fn tools(&self) -> Vec<crate::toolcall::ToolSpec> {
            vec![crate::toolcall::ToolSpec {
                server: "weather".into(),
                name: "get_weather".into(),
                description: "Current weather for a city.".into(),
                input_schema: serde_json::json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }),
            }]
        }
        fn call(&self, _server: &str, tool: &str, args: serde_json::Value, _ctx: ToolContext) -> Result<String, String> {
            assert_eq!(tool, "get_weather");
            Ok(format!("{}: sunny, 21 degrees Celsius, wind 9 km/h", args["city"].as_str().unwrap_or("?")))
        }
    }

    /// The real engine and a real model through the whole loop: the model asks for the tool, the
    /// "ask" rule waits for approval, the result goes back, the answer uses it. Needs the model at
    /// $CEREBRO_TEST_MODEL (default C:\models\dev\Qwen3-0.6B-Q8_0.gguf).
    /// `cargo test -p cerebro-desktop tool_loop -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn a_real_model_calls_a_tool_and_uses_the_result() {
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into());
        let Some(model) = describe(Path::new(&path), "test") else { panic!("model not readable: {path}") };
        let engine = Engine::start();
        engine.set_tool_host(Arc::new(Weather));
        engine.load(model.clone(), 4096).unwrap();
        let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
        let ev = events.clone();
        let decider = engine.handle();
        let sink: ChatSink = Arc::new(move |e: ChatEvent| {
            if let ChatEvent::Tool { id, state: "approval", .. } = &e {
                // The user clicks "Allow once" in Chat.
                let (d, id) = (decider.clone(), id.clone());
                std::thread::spawn(move || assert!(d.decide(&id, true)));
            }
            ev.lock().unwrap().push(e);
        });
        let req = ChatRequest {
            model_id: model.id.clone(),
            system_prompt: String::new(),
            turns: vec![ChatTurn { role: "user".into(), text: "What is the weather in Paris right now? Use the tool.".into() }],
            sampler: ChatSampler { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 1, repeat_penalty: 1.0, max_tokens: 400 },
            native: None,
            tools: Some(ToolsRequest { rules: Default::default(), default_rule: "ask".into(), offline: false, allow_local_fetch: false }),
            orchestrate: None,
        };
        engine.chat(req, sink).unwrap();
        let events = events.lock().unwrap();
        let states: Vec<&str> = events.iter().filter_map(|e| if let ChatEvent::Tool { state, .. } = e { Some(*state) } else { None }).collect();
        let answer: String = events.iter().filter_map(|e| if let ChatEvent::Token { text } = e { Some(text.as_str()) } else { None }).collect();
        eprintln!("tool states: {states:?}\nanswer: {answer}");
        assert_eq!(states, ["approval", "running", "done"]);
        assert!(!answer.contains("<tool_call>"), "tool markup never reaches the screen");
        assert!(answer.contains("21"), "the answer uses the tool result");
        assert!(!answer.starts_with(char::is_whitespace), "no blank lead-in before the answer");
        assert!(matches!(events.last(), Some(ChatEvent::Done { .. })));
    }

    /// Orchestrate mode on the real engine: plan, ideate, manage, workers and finalize all run on
    /// the loaded model, the ledger lands on disk, and the final result is the reply.
    /// `cargo test -p cerebro-desktop orchestrate_on -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn orchestrate_on_a_real_model() {
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into());
        let model = describe(Path::new(&path), "test").expect("model");
        let engine = Engine::start();
        let root = std::env::temp_dir().join(format!("cerebro-orch-{}", std::process::id()));
        engine.set_ledger_root(root.clone());
        engine.load(model.clone(), 8192).unwrap();
        let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
        let ev = events.clone();
        let sink: ChatSink = Arc::new(move |e: ChatEvent| ev.lock().unwrap().push(e));
        let req = ChatRequest {
            model_id: model.id.clone(),
            system_prompt: String::new(),
            turns: vec![ChatTurn { role: "user".into(), text: "What is 17 multiplied by 23? Show the working briefly, then the number.".into() }],
            sampler: ChatSampler { temperature: 0.2, top_p: 0.8, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens: 600 },
            native: None,
            tools: None,
            orchestrate: Some(OrchestrateRequest { workers: 2, rounds: 2 }),
        };
        let t = Instant::now();
        engine.chat(req, sink).unwrap();
        let events = events.lock().unwrap();
        let steps: Vec<String> = events.iter().filter_map(|e| if let ChatEvent::Ledger { step: Some(s), .. } = e { Some(s.clone()) } else { None }).collect();
        let files: Vec<&str> = events.iter().filter_map(|e| if let ChatEvent::Ledger { file: Some(f), .. } = e { Some(*f) } else { None }).collect();
        let answer: String = events.iter().filter_map(|e| if let ChatEvent::Token { text } = e { Some(text.as_str()) } else { None }).collect();
        eprintln!("{:.1} s, steps {steps:?}
files {files:?}
answer: {answer}", t.elapsed().as_secs_f64());
        assert_eq!(steps.first().map(String::as_str), Some("plan"));
        assert_eq!(steps.last().map(String::as_str), Some("finalize"));
        assert!(files.contains(&"result.md") && files.contains(&"plan.md"));
        assert!(!answer.trim().is_empty());
        assert!(matches!(events.last(), Some(ChatEvent::Done { .. })));
        let _ = std::fs::remove_dir_all(root);
    }
}
