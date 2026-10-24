//! The inference engine inside the app: Cerebro Engine (our own; no external server or process).
//!
//! One worker thread owns the loaded model, its session (KV cache, GPU state) and tokenizer; IPC
//! commands send it jobs and read a shared status that the Monitor and status bar poll. Model
//! files are found by scanning Cerebro's, another app's and the user's model folders; nothing is
//! moved or modified.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cerebro_engine::chat::{generate, generate_parts, GenConfig};
use cerebro_engine::gguf::Gguf;
use cerebro_engine::model::{ImageTokens, Model, Part, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;
use cerebro_engine::tune;
use serde::{Deserialize, Serialize};

use crate::kv_store::{self, KvStore};

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
    /// The longest context Cerebro runs it at: YARN_MAX times the trained length (YaRN past it).
    pub context_extended: u64,
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
    pub tune_result: Option<serde_json::Value>,
    /// The vision projector paired with this model, when one sits beside it (vision_input.rs).
    pub projector: Option<crate::vision_input::Projector>,
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

/// The folders scanned for models, with the source label the UI shows and how deep to look: every
/// place a model is commonly kept, so nothing has to be set up. Cerebro's own folder; LM Studio's;
/// the Hugging Face cache; Jan, GPT4All and text-generation-webui; the user's Downloads, Desktop
/// and Documents (shallow); and a models / LLM / AI folder at the root of any drive.
fn model_dirs() -> Vec<(PathBuf, &'static str, usize)> {
    let mut v = Vec::new();
    let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    let roaming = std::env::var_os("APPDATA").map(PathBuf::from);
    if let Some(l) = &local {
        v.push((l.join("Cerebro").join("models"), "cerebro", 6));
    }
    if let Some(home) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
        v.push((home.join(".lmstudio").join("models"), "lmstudio", 6));
        v.push((home.join(".cache").join("lm-studio").join("models"), "lmstudio", 6));
        v.push((home.join(".cache").join("huggingface").join("hub"), "other", 5));
        v.push((home.join("jan").join("models"), "other", 4));
        v.push((home.join("text-generation-webui").join("models"), "other", 3));
        for d in ["Downloads", "Desktop", "Documents"] {
            v.push((home.join(d), "other", 2));
        }
    }
    if let Some(r) = &roaming {
        v.push((r.join("Jan").join("data").join("models"), "other", 4));
    }
    if let Some(l) = &local {
        v.push((l.join("nomic.ai").join("GPT4All"), "other", 2));
    }
    v.push((PathBuf::from("C:\\models"), "other", 6));
    // models / LLM / AI folders at the root of every drive.
    for letter in b'C'..=b'Z' {
        let root = PathBuf::from(format!("{}:\\", letter as char));
        let Ok(rd) = std::fs::read_dir(&root) else { continue };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_lowercase();
            if e.path().is_dir() && matches!(name.as_str(), "models" | "model" | "llm" | "llms" | "ai" | "gguf" | "ggufs" | "lm studio" | "lmstudio") && !(letter == b'C' && name == "models") {
                v.push((e.path(), "other", 6));
            }
        }
    }
    v
}

/// Ollama's models: its files have no extension, so they are found through its own manifests
/// (each names the blob that holds the model's GGUF).
fn ollama_models() -> Vec<PathBuf> {
    let root = std::env::var_os("OLLAMA_MODELS").map(PathBuf::from).or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".ollama").join("models")));
    let Some(root) = root else { return Vec::new() };
    let mut manifests = Vec::new();
    fn files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if depth < 5 {
                    files(&p, depth + 1, out);
                }
            } else {
                out.push(p);
            }
        }
    }
    files(&root.join("manifests"), 0, &mut manifests);
    let mut out = Vec::new();
    for m in manifests {
        let Some(v) = std::fs::read_to_string(&m).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) else { continue };
        for layer in v["layers"].as_array().into_iter().flatten() {
            if layer["mediaType"].as_str() == Some("application/vnd.ollama.image.model") {
                if let Some(d) = layer["digest"].as_str() {
                    let blob = root.join("blobs").join(d.replace(':', "-"));
                    if blob.is_file() && !out.contains(&blob) {
                        out.push(blob);
                    }
                }
            }
        }
    }
    out
}

fn walk(dir: &Path, depth: usize, max: usize, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            let name = e.file_name().to_string_lossy().to_lowercase();
            // An image model is a folder (diffusers layout): listed whole, not searched inside.
            if crate::images::is_image_model(&p) {
                out.push(p);
            } else if depth < max && !matches!(name.as_str(), "node_modules" | ".git" | "$recycle.bin" | "windows" | "appdata") {
                walk(&p, depth + 1, max, out);
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

/// Architectures Cerebro Engine runs. qwen35 is the Qwen3.5/3.8 hybrid (Gated DeltaNet + gated attention).
const SUPPORTED_ARCHS: &[&str] = cerebro_engine::model::ARCHITECTURES;

/// A short, stable id from the path (FNV-1a).
fn id_of(path: &Path) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.to_string_lossy().to_lowercase().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("m{h:016x}")
}

/// The parts of a split model: the file name without its "-00001-of-00003" tail, and how many parts
/// it has. None for a file that is not one part of a split model.
fn split_parts(path: &Path) -> Option<(String, u64)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF"))?;
    let (head, count) = stem.rsplit_once("-of-")?;
    if count.len() != 5 || !count.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (prefix, index) = head.rsplit_once('-')?;
    if index.len() != 5 || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((prefix.to_owned(), count.parse().ok()?))
}

/// The size of a model file on disk. A split model is one model in the library, so all its parts
/// are added up: the number shown is what the model really takes on the drive.
fn file_size(path: &Path) -> u64 {
    let own = std::fs::metadata(path).map_or(0, |m| m.len());
    let Some((prefix, count)) = split_parts(path) else { return own };
    let Some(dir) = path.parent() else { return own };
    let Ok(entries) = std::fs::read_dir(dir) else { return own };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let p = entry.path();
        if p == path {
            total += own;
        } else if split_parts(&p).is_some_and(|(pre, n)| pre == prefix && n == count) {
            total += entry.metadata().map_or(0, |m| m.len());
        }
    }
    total
}

/// A model's name as people write it, from the GGUF's name or its file name: no quant label
/// ("Qwen3.8-27B-IQ1_M" -> "Qwen3.8 27B"), no publisher prefix a re-uploader added
/// ("Qwen_Qwen3.6-35B-A3B" -> "Qwen3.6 35B A3B"), spaces between words and the usual capitals
/// ("sd-turbo" -> "SD Turbo", "awq" -> "AWQ", "0.6b" -> "0.6B").
pub fn display_name(raw: &str) -> String {
    let is_quant = |t: &str| {
        let u = t.to_ascii_uppercase();
        let u = u.strip_prefix("UD_").or_else(|| u.strip_prefix("UD")).unwrap_or(&u).to_owned();
        if matches!(u.as_str(), "F16" | "F32" | "BF16" | "FP16" | "FP32" | "GGUF" | "UD" | "") {
            return true;
        }
        let b = u.strip_prefix('I').unwrap_or(&u);
        b.len() >= 2 && b.starts_with('Q') && b.as_bytes()[1].is_ascii_digit() && b[2..].chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let mut tokens: Vec<String> = raw.split(|c: char| c == '-' || c == ' ').filter(|t| !t.is_empty() && !is_quant(t)).map(str::to_owned).collect();
    // "Owner_Model..." where the model name repeats the owner: keep the model name.
    if let Some(first) = tokens.first_mut() {
        if let Some((owner, rest)) = first.split_once('_') {
            if !rest.is_empty() && rest.to_ascii_lowercase().starts_with(&owner.to_ascii_lowercase()) {
                *first = rest.to_owned();
            }
        }
    }
    let words: Vec<String> = tokens
        .iter()
        .flat_map(|t| t.split('_').filter(|w| !w.is_empty()).map(str::to_owned).collect::<Vec<_>>())
        .map(|w| {
            let l = w.to_ascii_lowercase();
            let size_like = l.chars().next().is_some_and(|c| c.is_ascii_digit()) || (l.len() <= 5 && l.starts_with('a') && l.ends_with('b') && l[1..l.len() - 1].chars().all(|c| c.is_ascii_digit() || c == '.') && l.len() > 2);
            match l.as_str() {
                "awq" | "gptq" | "sd" | "sdxl" | "tts" | "asr" | "vl" | "it" | "mlx" => l.to_ascii_uppercase(),
                "moe" => "MoE".to_owned(),
                _ if size_like => {
                    // Sizes: 0.6b -> 0.6B, 35b -> 35B, a3b -> A3B, 12hz -> 12Hz.
                    if let Some(n) = l.strip_suffix("hz") {
                        format!("{n}Hz")
                    } else {
                        l.to_ascii_uppercase()
                    }
                }
                _ if w.chars().all(|c| c.is_ascii_lowercase()) => {
                    let mut c = w.chars();
                    c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
                }
                _ => w.clone(),
            }
        })
        .collect();
    let out = words.join(" ");
    if out.is_empty() {
        raw.to_owned()
    } else {
        out
    }
}

fn describe(path: &Path, source: &'static str) -> Option<LibraryModel> {
    // All of a split model's files together.
    let size = file_size(path);
    let g = Gguf::open(path).ok()?;
    let arch = g.str("general.architecture").unwrap_or("unknown").to_owned();
    let k = |s: &str| format!("{arch}.{s}");
    let u = |s: &str| g.u64(&k(s)).ok();
    let n_head = u("attention.head_count").unwrap_or(0);
    let dim = u("embedding_length").unwrap_or(0);
    let experts = u("expert_count").filter(|&e| e > 0);
    let name = display_name(&g.str("general.name").map(str::to_owned).unwrap_or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()));
    // A metadata-edited copy (gguf_edit.rs) is its own model; its name says so.
    let name = if crate::gguf_edit::is_copy(path) { format!("{name} (edited)") } else { name };
    let params = g.str("general.size_label").map(str::to_owned).unwrap_or_default();
    let quant = g.u64("general.file_type").map(file_type_name).unwrap_or("other").to_owned();
    let family = g.str("general.basename").map(str::to_owned).unwrap_or_else(|| arch.clone());
    let projector = crate::vision_input::find_projector(path, dim, g.str("general.basename").unwrap_or(""));
    let kept = crate::tune_screen::kept(&id_of(path));
    Some(LibraryModel {
        id: id_of(path),
        name,
        quant,
        params,
        architecture: if experts.is_some() { "moe" } else { "dense" },
        size_gi_b: size as f64 / GIB,
        context_max: u("context_length").unwrap_or(0),
        context_extended: u("context_length").unwrap_or(0) * YARN_MAX as u64,
        tune: if kept.is_some() { "tuned" } else { "untuned" },
        provenance: "measured",
        kind: match projector.as_ref().map(|p| p.kind) {
            Some("audio") => "stt",
            Some("tts") => "tts",
            Some(_) => "vision",
            None => "text",
        },
        modalities: match projector.as_ref().map(|p| p.kind) {
            Some("audio") => vec!["audio-in"],
            Some("tts") => vec!["audio-out"],
            Some(_) => vec!["text", "vision"],
            None => vec!["text"],
        },
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
        tuned_at: kept.as_ref().map(|k| k.tuned_at.clone()),
        tune_result: kept.as_ref().map(|k| k.result.clone()),
        projector,
        supported: SUPPORTED_ARCHS.contains(&arch.as_str()),
        arch_name: arch,
    })
}

/// One file as the library lists it (tests and tools).
#[allow(dead_code)]
pub fn scan_library_one(path: &Path) -> Option<LibraryModel> {
    describe(path, "test")
}

pub fn scan_library() -> Vec<LibraryModel> {
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut dirs = model_dirs();
    dirs.push((PathBuf::new(), "ollama", 0));
    for (dir, source, max) in dirs {
        files.clear();
        if source == "ollama" {
            files = ollama_models();
        } else {
            walk(&dir, 0, max, &mut files);
        }
        for f in &files {
            let key = f.to_string_lossy().to_lowercase();
            if seen.insert(key) {
                let m = if f.is_dir() { crate::images::describe(f, source, id_of(f)) } else { describe(f, source) };
                if let Some(m) = m {
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
    /// Review passes after the reply (effort levels): 1 rereads and corrects it in the same
    /// conversation, 2 also checks it in a fresh context. 0 = none.
    #[serde(default)]
    pub passes: u8,
    /// Settings > Protections "Keep long chats ready": save this conversation's cache to disk after
    /// a slow, long reply and restore it when the same chat returns. Absent means on.
    #[serde(default = "yes")]
    pub keep_cache: bool,
    /// Tools defined by a local API client (local_api.rs). The model sees them like any tool, but a
    /// call is handed back to the client (`Tool` event, state "requested") instead of being run, and
    /// the reply ends there. Never set by the window.
    #[serde(skip)]
    pub client_tools: Option<Vec<crate::toolcall::ToolSpec>>,
}

/// `keep_cache`'s serde default: saving is on unless the caller turns it off.
fn yes() -> bool {
    true
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
    /// What the built-in tools may touch: "read-only", "workspace" (default) or "full".
    #[serde(default)]
    pub access: Access,
    /// Settings > Protections "Hide secrets from tools" is off.
    #[serde(default)]
    pub show_secrets: bool,
    /// Settings > Protections "Read before write" is off.
    #[serde(default)]
    pub blind_writes: bool,
    /// Plan mode (Chat): investigate read-only, then hand a plan over with exit_plan_mode.
    #[serde(default)]
    pub plan_mode: bool,
}

/// How far the built-in tools reach. Commands are contained (Job Object, time limit) but not
/// confined to the folders by the operating system, which is why read-only refuses them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Access {
    /// Reading inside the workspace folders only: no writes, edits or commands.
    ReadOnly,
    /// Reading and writing inside the workspace folders; commands run in the first folder.
    #[default]
    Workspace,
    /// Any path this Windows account can reach.
    Full,
}

impl Access {
    /// One factual sentence for the system prompt, so the model knows its bounds before it tries.
    pub fn statement(self) -> &'static str {
        match self {
            Access::ReadOnly => "Tool access is read-only: you can read files in the workspace folders, but you cannot change files or run commands.",
            Access::Workspace => "Tool access is workspace: you can read and change files only inside the workspace folders; commands run in the first workspace folder.",
            Access::Full => "Tool access is full: you can read and change files anywhere this user can.",
        }
    }
}

impl ToolsRequest {
    fn rule(&self, server: &str, tool: &str) -> &str {
        self.rules.get(&format!("{server}/{tool}")).map_or(self.default_rule.as_str(), String::as_str)
    }
}

/// What a tool call needs to know about the user's settings for this turn.
#[derive(Debug, Clone, Copy)]
pub struct ToolContext {
    /// Offline mode is on (Settings): the tool must not touch the network.
    pub offline: bool,
    /// A fetch may reach loopback, private and link-local addresses.
    pub allow_local_fetch: bool,
    pub access: Access,
    /// Variables that look like secrets are removed from a command's environment.
    pub scrub_env: bool,
    /// A file must have been read (and be unchanged since) before it is edited or replaced.
    pub read_before_write: bool,
}

impl Default for ToolContext {
    fn default() -> Self {
        ToolContext { offline: false, allow_local_fetch: false, access: Access::Workspace, scrub_env: true, read_before_write: true }
    }
}

impl ToolContext {
    pub fn from_request(t: Option<&ToolsRequest>) -> Self {
        match t {
            Some(t) => ToolContext { offline: t.offline, allow_local_fetch: t.allow_local_fetch, access: t.access, scrub_env: !t.show_secrets, read_before_write: !t.blind_writes },
            None => ToolContext::default(),
        }
    }
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
    /// What the Chat screen attached; images carry their data (a data: URL) so a vision model can
    /// see them, other files only their names.
    #[serde(default)]
    pub attachments: Vec<TurnAttachment>,
}

/// An attachment as the engine uses it: only an image's data matters (names and types are for the
/// screen).
#[derive(Debug, Clone, Deserialize)]
pub struct TurnAttachment {
    #[serde(default)]
    pub data: Option<String>,
}

impl ChatTurn {
    /// The images attached to this turn, as data: URLs.
    pub(crate) fn images(&self) -> impl Iterator<Item = &str> {
        self.attachments.iter().filter_map(|a| a.data.as_deref()).filter(|d| d.starts_with("data:image/"))
    }
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
    /// A review pass finished (`pass` 1 or 2) and what it found.
    Check {
        pass: u8,
        text: String,
    },
    /// A review pass corrected the reply: `text` replaces everything streamed so far.
    Revise {
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
    pub prompt_tokens: usize,
    /// The reply stopped because it reached max_tokens.
    pub length: bool,
    pub output_tokens: usize,
    pub thinking_tokens: usize,
    pub tok_per_sec: f64,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub extra_ms: f64,
    pub stopped: bool,
}

/// Where a tuning run's events go: the window's IPC channel.
pub type TuneSink = Arc<dyn Fn(TuneMachineEvent) + Send + Sync>;

/// Streamed to the UI while tuning. Mirrors `MachineTuneEvent` in ui/src/service/types.ts: a trial's
/// fields sit beside `type`, exactly as the engine's `Trial` serializes.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TuneMachineEvent {
    Trial(tune::Trial),
    Done { outcome: tune::Outcome },
    Error { message: String },
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
    /// Why the request failed, when its state is "error".
    pub error: Option<String>,
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
    /// The loaded session's prompt-reading progress (live prompt tokens read and speed).
    pub progress: Option<Arc<cerebro_engine::model::Progress>>,
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
    /// How the model writes tool calls (for replaying a client's tool history, local_api.rs).
    pub dialect: crate::toolcall::Dialect,
    /// Where the weights went: GPU layers, RAM, drive, and why (cerebro_engine::placement).
    pub placement: Option<cerebro_engine::placement::Plan>,
    /// Layers running on the GPU (all of them, some, or none).
    pub gpu_layers: usize,
}

enum Job {
    Load { model: LibraryModel, ctx: usize, gpu_layers: Option<usize>, reply: Sender<Result<(), String>> },
    /// Load with some layers on other computers (cluster.rs); the rest on this processor.
    LoadCluster { model: LibraryModel, ctx: usize, local_share: f64, workers: Vec<crate::cluster::WorkerChoice>, reply: Sender<Result<Vec<crate::cluster::LinkReport>, String>> },
    Unload { reply: Sender<()> },
    /// The long-context check (engine longctx.rs) on the loaded model at its full context.
    ContextCheck { reply: Sender<Result<serde_json::Value, String>> },
    Chat { req: ChatRequest, events: ChatSink, reply: Sender<()> },
    Tune { events: TuneSink, reply: Sender<Result<tune::Outcome, String>> },
    /// The Models screen's image test: one image, one question, on the loaded vision model.
    Describe { id: String, data: String, reply: Sender<Result<String, String>> },
    /// The Tune screen's run (tune_screen.rs): measured on the engine thread, which owns the model.
    TuneModel { hub: Arc<crate::tune_screen::TuneHub> },
    /// The Benchmark lab's run (bench.rs): one model, three depths and twenty questions.
    Benchmark { model: LibraryModel, placement: String, events: crate::bench::BenchSink, reply: Sender<Result<crate::bench::BenchRun, String>> },
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
    /// Questions and plans waiting for the user, by call id; `answer` replies.
    answers: Answers,
    tool_host: Arc<Mutex<Option<Arc<dyn ToolHost>>>>,
    /// Where Orchestrate mode keeps its ledgers (the app's data folder).
    ledger_root: Arc<Mutex<Option<PathBuf>>>,
    /// Saved conversation caches ("Keep long chats ready"), shared with the worker thread and the
    /// Settings commands; None when the app data folder cannot be found.
    kv_store: Option<Arc<Mutex<KvStore>>>,
}

type Answers = Arc<Mutex<std::collections::HashMap<String, Sender<serde_json::Value>>>>;

/// Today's date as YYYY-MM-DD (UTC).
pub fn today_iso() -> String {
    let days = (SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) / 86_400) as i64;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
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
    placement: Option<cerebro_engine::placement::Plan>,
    /// The number of graphics-card layers the load forced (None: as many as fit). Kept so a later
    /// run can measure the same placement the user chose.
    gpu_layers: Option<usize>,
    /// The vision encoder, loaded the first time an image arrives (it takes RAM a text chat does
    /// not need).
    vision: Option<cerebro_engine::vision::VisionModel>,
    /// llama.cpp running this model (llama_backend.rs), when it serves the placement; replies come
    /// from it and `session` is a placeholder.
    llama: Option<crate::llama_backend::Server>,
    /// Encoded images of recent chats by `vision_input::key`, so a chat's images are encoded once.
    images: std::collections::HashMap<u64, Arc<cerebro_engine::vision::ImageEmbedding>>,
}

/// Encoded images kept for reuse (a few chats' worth).
const MAX_CACHED_IMAGES: usize = 16;

/// The image test: what the model says about one image (greedy, short).
fn describe_image(l: &mut Loaded, data: &str) -> Result<String, String> {
    let pics = vec![data.to_owned()];
    let embeds = encode_images(l, &pics)?;
    let e = embeds.first().ok_or("The image could not be read.")?;
    let before = l.tok.encode("<|im_start|>user\n<|vision_start|>");
    let after = l.tok.encode("<|vision_end|>Describe this image in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let parts = [Part::Text(&before), Part::Image(ImageTokens { cols: e.cols, rows: e.rows, dim: e.dim, data: &e.data }), Part::Text(&after)];
    let cfg = GenConfig { sampler: SamplerConfig { temperature: 0.0, top_k: 1, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, seed: 0 }, max_tokens: 160 };
    let mut out = String::new();
    generate_parts(&l.model, &mut l.session, &l.tok, &parts, &cfg, |p| {
        out.push_str(p);
        true
    })
    .map_err(|e| format!("The model could not read the image: {e}"))?;
    Ok(out.trim().to_owned())
}

/// Encodes `pics` (data: URLs) for the loaded model, reusing earlier encodings of the same image.
fn encode_images(l: &mut Loaded, pics: &[String]) -> Result<Vec<Arc<cerebro_engine::vision::ImageEmbedding>>, String> {
    if pics.is_empty() {
        return Ok(Vec::new());
    }
    let Some(proj) = l.info.projector.clone().filter(|p| p.kind == "vision") else {
        return Err("This model cannot see images: load one with a vision projector (an mmproj file) beside it.".to_owned());
    };
    if l.vision.is_none() {
        let v = cerebro_engine::vision::VisionModel::load(Path::new(&proj.path)).map_err(|e| format!("Could not load the vision projector {}: {e}", proj.file))?;
        l.vision = Some(v);
    }
    let v = l.vision.as_ref().expect("vision encoder");
    let mut out = Vec::with_capacity(pics.len());
    for p in pics {
        let k = crate::vision_input::key(p);
        if let Some(e) = l.images.get(&k) {
            out.push(e.clone());
            continue;
        }
        let rgb = crate::vision_input::decode_data_url(p)?;
        let input = v.preprocess(&rgb.pixels, rgb.width, rgb.height);
        let emb = Arc::new(v.encode(&input).map_err(|e| format!("Could not read an attached image: {e}"))?);
        if l.images.len() >= MAX_CACHED_IMAGES {
            l.images.clear();
        }
        l.images.insert(k, emb.clone());
        out.push(emb);
    }
    Ok(out)
}

impl Engine {
    pub fn start() -> Self {
        let (tx, rx) = channel::<Job>();
        let status = Arc::new(Mutex::new(Status { activity: "idle", totals: Totals { since: now_ms(), ..Default::default() }, ..Default::default() }));
        let cancel = Arc::new(AtomicBool::new(false));
        let approvals: Arc<Mutex<std::collections::HashMap<String, Sender<bool>>>> = Arc::default();
        let answers: Answers = Arc::default();
        let tool_host: Arc<Mutex<Option<Arc<dyn ToolHost>>>> = Arc::default();
        let st = status.clone();
        let cn = cancel.clone();
        let ledger_root: Arc<Mutex<Option<PathBuf>>> = Arc::default();
        // Saved conversation caches: one folder under the app's data, at most 8 GiB (the oldest go
        // first). Without a LOCALAPPDATA there is no store and saving is quietly off.
        let kv_store = std::env::var_os("LOCALAPPDATA").map(|local| Arc::new(Mutex::new(KvStore::new(PathBuf::from(local).join("Cerebro").join("kv"), 8u64 << 30))));
        let (ap, th, lr, an, kvs) = (approvals.clone(), tool_host.clone(), ledger_root.clone(), answers.clone(), kv_store.clone());
        std::thread::Builder::new()
            .name("cerebro-engine".into())
            .spawn(move || {
                let mut loaded: Option<Loaded> = None;
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Load { model, ctx, gpu_layers, reply } => {
                            // Automatic settings on a model the Tune screen tuned: its kept placement
                            // and switches.
                            let gpu_layers = if gpu_layers.is_none() {
                                crate::tune_screen::apply_kept_switches(&model.id);
                                crate::tune_screen::kept(&model.id).and_then(|k| k.gpu_layers)
                            } else {
                                gpu_layers
                            };
                            if let Some(l) = loaded.take() {
                                save_profile(&l);
                            }
                            set(&st, |s| {
                                s.loaded = None;
                                s.activity = "loading";
                            });
                            let r = load(&model, ctx, gpu_layers);
                            let msg = match r {
                                Ok(l) => {
                                    let info = loaded_info(&l, ctx);
                                    let progress = l.session.progress.clone();
                                    loaded = Some(l);
                                    set(&st, |s| {
                                        s.loaded = Some(info);
                                        s.progress = Some(progress);
                                    });
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            };
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(msg);
                        }
                        Job::LoadCluster { model, ctx, local_share, workers, reply } => {
                            if let Some(l) = loaded.take() {
                                save_profile(&l);
                            }
                            set(&st, |s| {
                                s.loaded = None;
                                s.activity = "loading";
                            });
                            let r = load_cluster(&model, ctx, local_share, &workers);
                            let msg = match r {
                                Ok((l, links)) => {
                                    let info = loaded_info(&l, ctx);
                                    let progress = l.session.progress.clone();
                                    loaded = Some(l);
                                    set(&st, |s| {
                                        s.loaded = Some(info);
                                        s.progress = Some(progress);
                                    });
                                    Ok(links)
                                }
                                Err(e) => Err(e),
                            };
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(msg);
                        }
                        Job::ContextCheck { reply } => {
                            let r = match loaded.as_mut() {
                                None => Err("Load a model first.".to_owned()),
                                Some(l) if l.llama.is_some() => Err("The long-context check runs on Cerebro's engine; this model is running on the llama.cpp backend.".to_owned()),
                                Some(l) => {
                                    set(&st, |s| s.activity = "reading");
                                    let r = context_check(l);
                                    set(&st, |s| s.activity = "idle");
                                    r
                                }
                            };
                            let _ = reply.send(r);
                        }
                        Job::Unload { reply } => {
                            if let Some(l) = loaded.take() {
                                save_profile(&l);
                            }
                            set(&st, |s| {
                                s.loaded = None;
                                s.progress = None;
                            });
                            let _ = reply.send(());
                        }
                        Job::Chat { req, events, reply } => {
                            cn.store(false, Ordering::SeqCst);
                            match loaded.as_mut() {
                                Some(l) if l.info.id == req.model_id && l.llama.is_some() => {
                                    crate::llama_backend::reply(l.llama.as_ref().expect("checked"), &req, &events);
                                }
                                Some(l) if l.info.id == req.model_id && req.orchestrate.is_some() => {
                                    let root = lr.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(std::env::temp_dir);
                                    run_orchestrate(l, &req, &events, &st, &cn, &root);
                                }
                                Some(l) if l.info.id == req.model_id => {
                                    let host = th.lock().unwrap_or_else(|e| e.into_inner()).clone();
                                    let spill_dir = lr.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|d| d.join("spills"));
                                    run_chat(l, &req, &events, &st, &cn, Tooling { host, approvals: &ap, answers: &an, spill_dir }, kvs.as_ref());
                                }
                                _ => {
                                    events(ChatEvent::Error { message: "Load this model first (Models screen).".into() });
                                }
                            }
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(());
                        }
                        Job::Describe { id, data, reply } => {
                            let r = match loaded.as_mut() {
                                Some(l) if l.info.id == id => describe_image(l, &data),
                                _ => Err("Load this model first, then test it.".to_owned()),
                            };
                            let _ = reply.send(r);
                        }
                        Job::TuneModel { hub } => {
                            let Some(spec) = hub.spec() else { continue };
                            set(&st, |s| s.activity = "tuning");
                            // The model's weights: the loaded ones when it is this file, else read now.
                            // Its session goes first, so every candidate has the whole card.
                            let prev_ctx = loaded.as_ref().filter(|l| l.info.id == spec.model.id).map(|l| l.session.max_ctx);
                            let weights = match loaded.take() {
                                Some(l) if l.info.id == spec.model.id => {
                                    save_profile(&l);
                                    let Loaded { model, tok, .. } = l;
                                    Ok((model, tok))
                                }
                                other => {
                                    if let Some(l) = other {
                                        save_profile(&l);
                                    }
                                    Model::load(Path::new(&spec.model.path))
                                        .map_err(|e| format!("Could not load the model: {e}"))
                                        .and_then(|m| Tokenizer::from_gguf(m.gguf()).map(|t| (m, t)).map_err(|e| format!("Could not read the tokenizer: {e}")))
                                }
                            };
                            set(&st, |s| {
                                s.loaded = None;
                                s.progress = None;
                            });
                            match weights {
                                Ok((model, tok)) => {
                                    let place_fn = |m: &Model, ctx: usize, layers: Option<usize>| -> Result<(Session, usize, usize, bool), String> {
                                        let (s, _) = place(m, ctx, layers);
                                        let on = s.gpu.as_ref().map_or(0, |g| g.gpu_layers()) + s.portable.as_ref().map_or(0, |p| p.gpu_layers());
                                        // A forced count the planner could not honour is not the candidate it
                                        // claims to be: measuring it would compare the baseline with itself.
                                        if let Some(want) = layers.filter(|&w| w > 0 && w != on) {
                                            return Err(format!("Only {on} of the requested {want} layers fit on the graphics card at this context."));
                                        }
                                        let cuda = s.gpu.is_some();
                                        if std::env::var_os("CEREBRO_TUNE_DEBUG").is_some() {
                                            eprintln!("place {layers:?}: {on} layers on a card (cuda {cuda}), spec {}, threads {}, free {:.2} GiB", cerebro_engine::switches::get(cerebro_engine::switches::SPEC_LOOKUP), cerebro_engine::switches::get(cerebro_engine::switches::CPU_DECODE_THREADS), cerebro_engine::hw::gpus().first().map_or(0.0, |g| g.vram_free as f64 / (1u64 << 30) as f64));
                                        }
                                        Ok((s, on, m.cfg.n_layer, cuda))
                                    };
                                    let done = crate::tune_screen::run_job(&hub, &model, &tok, &place_fn);
                                    // The tuned model stays loaded, with the winner's placement (or as it
                                    // was when the run stopped early): it is the model the user is about
                                    // to use. A previously loaded different model is not reloaded.
                                    let layers = done.as_ref().map_or(None, |d| d.gpu_layers);
                                    let ctx = prev_ctx.unwrap_or(8192);
                                    let l = assemble(model, tok, &spec.model, ctx, layers);
                                    let info = loaded_info(&l, l.session.max_ctx);
                                    let progress = l.session.progress.clone();
                                    loaded = Some(l);
                                    set(&st, |s| {
                                        s.loaded = Some(info);
                                        s.progress = Some(progress);
                                    });
                                }
                                Err(e) => hub.emit(serde_json::json!({ "kind": "failed", "reason": e })),
                            }
                            set(&st, |s| s.activity = "idle");
                        }
                        Job::Benchmark { model, placement, events, reply } => {
                            cn.store(false, Ordering::SeqCst);
                            set(&st, |s| s.activity = "benchmark");
                            // Benchmarks want the deepest window the model can hold; a model already
                            // loaded at least that large is reused, anything else is loaded now.
                            let wanted = crate::bench::session_ctx(model.context_max);
                            let layers = model.arch.layers as usize;
                            // The placement the user chose. "auto" keeps the placement the model is
                            // loaded with, so the benchmark measures what the user picked.
                            let loaded_forced = loaded.as_ref().filter(|l| l.info.id == model.id).and_then(|l| l.gpu_layers);
                            let forced = crate::bench::forced_layers(&placement, layers, loaded_forced);
                            let reuse = loaded.as_ref().is_some_and(|l| l.info.id == model.id && l.session.max_ctx >= wanted && l.gpu_layers == forced);
                            let result = if reuse {
                                let l = loaded.as_mut().expect("checked above");
                                let plan = crate::bench::placement_of(&placement, l.placement.as_ref(), gpu_layers_of(l), layers);
                                Ok(crate::bench::run(&l.model, &mut l.session, &l.tok, &l.info, plan, &events, &cn))
                            } else {
                                if let Some(l) = loaded.take() {
                                    save_profile(&l);
                                }
                                set(&st, |s| {
                                    s.loaded = None;
                                    s.progress = None;
                                });
                                match load(&model, wanted, forced) {
                                    Ok(mut l) => {
                                        let plan = crate::bench::placement_of(&placement, l.placement.as_ref(), gpu_layers_of(&l), layers);
                                        let run = crate::bench::run(&l.model, &mut l.session, &l.tok, &l.info, plan, &events, &cn);
                                        loaded = Some(l);
                                        Ok(run)
                                    }
                                    Err(e) => Err(e),
                                }
                            };
                            if let Some(l) = loaded.as_mut() {
                                let info = loaded_info(l, l.session.max_ctx);
                                let progress = l.session.progress.clone();
                                save_profile(l);
                                set(&st, |s| {
                                    s.loaded = Some(info);
                                    s.progress = Some(progress);
                                });
                            }
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(result);
                        }
                        Job::Tune { events, reply } => {
                            cn.store(false, Ordering::SeqCst);
                            let msg = match loaded.as_mut() {
                                Some(l) => {
                                    set(&st, |s| s.activity = "tuning");
                                    let result = run_tune(l, &events, &cn);
                                    // Tuning replaces the session's cache with its own prompt; the
                                    // status reflects the model after the user's cache was restored.
                                    let info = loaded_info(l, l.session.max_ctx);
                                    set(&st, |s| s.loaded = Some(info));
                                    result
                                }
                                None => Err("Load a model first (Models screen).".to_owned()),
                            };
                            set(&st, |s| s.activity = "idle");
                            let _ = reply.send(msg);
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
                // Live while a prompt is being read, else the last prompt's figure.
                s.progress.as_ref().filter(|_| s.activity == "reading").and_then(|p| p.read().2).or_else(|| s.loaded.as_ref().and_then(|l| l.last_prefill_tps)),
                s.loaded.as_ref().and_then(|l| l.experts).map(|e| e.2 * 100.0),
                s.loaded.as_ref().map(|l| l.vram_bytes as f64 / GIB),
                Some((total - avail) as f64 / GIB),
            ];
            s.history.push_back(sample);
            while s.history.len() > 24 * 3600 {
                s.history.pop_front();
            }
        });
        Self { tx: Arc::new(Mutex::new(tx)), status, cancel, library: Arc::new(Mutex::new(Vec::new())), approvals, answers, tool_host, ledger_root, kv_store }
    }

    /// A handle that can move into a blocking task.
    pub fn handle(&self) -> Self {
        self.clone()
    }

    /// The folder of saved conversation caches, shared with the worker thread (None when the app
    /// data folder is unknown, and then saving and restoring are off).
    pub fn kv_store(&self) -> Option<Arc<Mutex<KvStore>>> {
        self.kv_store.clone()
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

    /// The user's answer to a question or plan waiting in the chat. False if nothing was waiting.
    pub fn answer(&self, id: &str, value: serde_json::Value) -> bool {
        let tx = self.answers.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
        tx.is_some_and(|tx| tx.send(value).is_ok())
    }

    fn send(&self, job: Job) -> Result<(), String> {
        self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(job).map_err(|_| "The engine has stopped.".to_owned())
    }

    /// Loads `model`; `gpu_layers` forces how many layers run on the GPU (None: as many as fit,
    /// Some(0): the processor only).
    pub fn load(&self, model: LibraryModel, ctx: usize, gpu_layers: Option<usize>) -> Result<(), String> {
        let (reply, rx) = channel();
        self.send(Job::Load { model, ctx, gpu_layers, reply })?;
        rx.recv().map_err(|_| "The engine stopped while loading.".to_owned())?
    }

    /// Loads `model` with layers on the paired workers; returns how each link measured.
    pub fn load_cluster(&self, model: LibraryModel, ctx: usize, local_share: f64, workers: Vec<crate::cluster::WorkerChoice>) -> Result<Vec<crate::cluster::LinkReport>, String> {
        let (reply, rx) = channel();
        self.send(Job::LoadCluster { model, ctx, local_share, workers, reply })?;
        rx.recv().map_err(|_| "The engine stopped while loading.".to_owned())?
    }

    /// Runs the long-context check on the loaded model (minutes at long contexts).
    pub fn context_check(&self) -> Result<serde_json::Value, String> {
        let (reply, rx) = channel();
        self.send(Job::ContextCheck { reply })?;
        rx.recv().map_err(|_| "The engine stopped during the check.".to_owned())?
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

    /// What the engine is doing now ("idle", "loading", "generating", "reading", "tuning" or
    /// "benchmark"), for the commands that must not overlap a benchmark.
    pub fn activity(&self) -> &'static str {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).activity
    }

    /// Runs the Benchmark lab on `model` in the placement the user chose ("auto", "gpu", "split"
    /// or "cpu"), streaming each point through `events`; resolves with the finished run. The model
    /// is left loaded afterwards.
    pub fn bench(&self, model: LibraryModel, placement: String, events: crate::bench::BenchSink) -> Result<crate::bench::BenchRun, String> {
        let (reply, rx) = channel();
        self.send(Job::Benchmark { model, placement, events, reply })?;
        rx.recv().map_err(|_| "The engine stopped during the benchmark.".to_owned())?
    }

    /// The image test on the loaded vision model `id`.
    pub fn describe(&self, id: String, data: String) -> Result<String, String> {
        let (reply, rx) = channel();
        self.send(Job::Describe { id, data, reply })?;
        rx.recv().map_err(|_| "The engine stopped during the test.".to_owned())?
    }

    /// Queues the Tune screen's current run (or its resume) on the engine thread; returns at once.
    pub fn tune_model(&self, hub: Arc<crate::tune_screen::TuneHub>) -> Result<(), String> {
        self.send(Job::TuneModel { hub })
    }

    /// Times the processor settings on the loaded model for this machine, streaming every finished
    /// trial to `events`; resolves with the outcome (or the reason it stopped).
    pub fn tune_machine(&self, events: TuneSink) -> Result<tune::Outcome, String> {
        let (reply, rx) = channel();
        self.send(Job::Tune { events, reply })?;
        rx.recv().map_err(|_| "The engine stopped during tuning.".to_owned())?
    }
}

fn set(st: &Arc<Mutex<Status>>, f: impl FnOnce(&mut Status)) {
    let mut s = st.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut s);
}

fn load(m: &LibraryModel, ctx: usize, gpu_layers: Option<usize>) -> Result<Loaded, String> {
    if !m.supported {
        return Err(format!("Cerebro Engine does not run the {} architecture yet.", m.arch_name));
    }
    let model = Model::load(Path::new(&m.path)).map_err(|e| format!("Could not load the model: {e}"))?;
    let tok = Tokenizer::from_gguf(model.gguf()).map_err(|e| format!("Could not read the tokenizer: {e}"))?;
    match with_llama(model, tok, m, ctx, gpu_layers) {
        Ok(l) => Ok(l),
        Err((model, tok)) => Ok(assemble(model, tok, m, ctx, gpu_layers)),
    }
}

/// A split model on llama.cpp when that is switched on, the architecture is one llama.cpp runs and
/// it can start (installing the release first if needed); None leaves the model to Cerebro's engine.
fn with_llama(model: Model, tok: Tokenizer, m: &LibraryModel, ctx: usize, gpu_layers: Option<usize>) -> Result<Loaded, (Model, Tokenizer)> {
    if !cerebro_engine::switches::enabled(cerebro_engine::switches::LLAMA_SPLITS) || gpu_layers == Some(0) || !crate::llama_backend::runs(&m.arch_name) {
        return Err((model, tok));
    }
    let ctx = ctx.clamp(512, model.cfg.ctx_train.max(512));
    let nvidia = cerebro_engine::hw::gpus();
    let (build, plan, layers) = if !nvidia.is_empty() {
        // NVIDIA: Cerebro's engine, whole or split. Its splits now read and write faster than
        // llama.cpp's on the same card (sweep C, Qwen3-8B half on an RTX 3070 Ti: 1136 vs 947 tok/s
        // reading, 16.7 vs 12.8 writing), and only its own session has Thinking, Tune, Benchmark,
        // saved chat caches and review passes.
        return Err((model, tok));
    } else {
        // AMD / Intel graphics the portable backend would use: llama.cpp's Vulkan build.
        let integrated_ok = cerebro_engine::switches::enabled(cerebro_engine::switches::PORTABLE_INTEGRATED);
        let adapters = cerebro_engine::portable::adapters();
        let discrete = adapters.iter().any(|a| a.kind.contains("Discrete"));
        let usable = cerebro_engine::switches::enabled(cerebro_engine::switches::PORTABLE_GPU) && (discrete || (integrated_ok && adapters.iter().any(|a| a.kind.contains("Integrated"))));
        if !usable {
            return Err((model, tok));
        }
        let mut p = match cerebro_engine::gpu::plan_for(&model, ctx, 0, None) {
            Ok(p) => p,
            Err(_) => return Err((model, tok)),
        };
        // Integrated graphics hold their layers in system RAM: as many as the RAM free now holds
        // (2 GiB kept for the processor's share and the desktop); the rest run on the processor,
        // which reads from the drive when RAM runs short. A card of its own fills its own memory.
        let n = p.total_layers;
        let layers = if discrete {
            n
        } else {
            let per = model.gguf().len() as u64 / n.max(1) as u64;
            let room = cerebro_engine::gguf::available_ram().saturating_sub(cerebro_engine::placement::RAM_MARGIN + (2 << 30));
            (room / per.max(1)) as usize
        }
        .min(n);
        if layers == 0 {
            return Err((model, tok));
        }
        p.gpu_layers = layers;
        (crate::llama_backend::Build::Vulkan, p, if layers == n { 999 } else { layers })
    };
    let dir = match crate::llama_backend::install(build, |done, total| {
        if done % (64 << 20) < (1 << 20) {
            eprintln!("llama.cpp download: {} of {} MB", done >> 20, total >> 20);
        }
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("llama.cpp is not available, so Cerebro's engine runs this split: {e}");
            return Err((model, tok));
        }
    };
    match crate::llama_backend::Server::start(&dir, Path::new(&m.path), layers, ctx) {
        Ok(server) => {
            let mut p = plan.clone();
            p.notes.insert(0, match build {
                crate::llama_backend::Build::Cuda => format!("llama.cpp runs this split ({} of {} layers on the graphics card): it reads prompts faster than Cerebro's engine does for splits today.", plan.gpu_layers, plan.total_layers),
                crate::llama_backend::Build::Vulkan if plan.gpu_layers < plan.total_layers => format!("llama.cpp's Vulkan build runs {} of {} layers on your graphics, as many as the free memory holds; the processor runs the rest.", plan.gpu_layers, plan.total_layers),
                crate::llama_backend::Build::Vulkan => "llama.cpp's Vulkan build runs this model on your graphics card: faster than Cerebro's engine on AMD and Intel graphics today.".to_owned(),
            });
            let dialect = crate::toolcall::dialect_from_template(model.gguf().str("tokenizer.chat_template").unwrap_or(""));
            // The model stays open for its tokenizer and metadata; its weights are not read here.
            let session = Session::placeholder(&model);
            Ok(Loaded { model, session, tok, info: m.clone(), dialect, placement: Some(p), gpu_layers, vision: None, llama: Some(server), images: Default::default() })
        }
        Err(e) => {
            eprintln!("llama.cpp could not start, so Cerebro's engine runs this split: {e}");
            Err((model, tok))
        }
    }
}

/// A model across computers: a processor session here for the first layers, each worker's range
/// over its link (cerebro_engine::cluster). Every computer needs the same model file.
fn load_cluster(m: &LibraryModel, ctx: usize, local_share: f64, workers: &[crate::cluster::WorkerChoice]) -> Result<(Loaded, Vec<crate::cluster::LinkReport>), String> {
    if !m.supported {
        return Err(format!("Cerebro Engine does not run the {} architecture yet.", m.arch_name));
    }
    let model = Model::load(Path::new(&m.path)).map_err(|e| format!("Could not load the model: {e}"))?;
    if model.cfg.ssm.is_some() {
        return Err("Models with recurrent layers cannot be shared across computers yet.".to_owned());
    }
    let tok = Tokenizer::from_gguf(model.gguf()).map_err(|e| format!("Could not read the tokenizer: {e}"))?;
    let ctx = ctx.clamp(512, model.cfg.ctx_train.max(512));
    let (remote, links) = crate::cluster::connect(Path::new(&m.path), model.cfg.n_layer, ctx, local_share, workers)?;
    let mut session = Session::new(&model, ctx);
    session.remote = Some(remote);
    let dialect = crate::toolcall::dialect_from_template(model.gguf().str("tokenizer.chat_template").unwrap_or(""));
    Ok((Loaded { model, session, tok, info: m.clone(), dialect, placement: None, gpu_layers: None, vision: None, llama: None, images: Default::default() }, links))
}

/// How far past its trained length a model's context may be stretched with YaRN (Qwen's own
/// long-context setting uses 4).
pub const YARN_MAX: usize = 4;

/// The long-context check on a loaded model: a passcode hidden at the start, middle and end of
/// documents a quarter, half and nine tenths of its context long (engine longctx.rs). The
/// conversation cache is cleared afterwards.
fn context_check(l: &mut Loaded) -> Result<serde_json::Value, String> {
    let max = l.session.max_ctx;
    let mut lengths: Vec<usize> = [max / 4, max / 2, max * 9 / 10].into_iter().filter(|&n| n >= 1024).collect();
    lengths.dedup();
    if lengths.is_empty() {
        return Err("The context is too short to check (load it with at least 2,048 tokens).".to_owned());
    }
    let mut probes = Vec::new();
    for (k, &n) in lengths.iter().enumerate() {
        for (j, &d) in [0.1f32, 0.5, 0.9].iter().enumerate() {
            let needle = cerebro_engine::longctx::build(&l.tok, n, d, (k * 3 + j) as u64 + 7);
            let p = cerebro_engine::longctx::probe(&l.model, &mut l.session, &l.tok, &needle).map_err(|e| format!("The check failed: {e}"))?;
            probes.push(serde_json::json!({ "tokens": p.tokens, "depth": p.depth, "found": p.found, "reply": p.reply.chars().take(120).collect::<String>(), "seconds": p.seconds }));
        }
    }
    let _ = l.session.clear();
    Ok(serde_json::json!({
        "context": max,
        "trained": l.model.cfg.ctx_train,
        "factor": l.model.ctx_extension.map_or(1.0, |e| e.0),
        "probes": probes,
    }))
}

/// A loaded model from its weights and tokenizer: the session placed as `place` decides.
fn assemble(mut model: Model, tok: Tokenizer, m: &LibraryModel, ctx: usize, gpu_layers: Option<usize>) -> Loaded {
    // Past the trained length the model runs stretched with YaRN (engine Model::extend_context).
    let ctx = ctx.clamp(512, (model.cfg.ctx_train * YARN_MAX).max(512));
    let factor = model.extend_context(ctx);
    if factor > 1.0 {
        eprintln!("context: {ctx} tokens, {factor:.2}x the trained {} with YaRN", model.cfg.ctx_train);
    }
    let (session, placement) = place(&model, ctx, gpu_layers);
    let dialect = crate::toolcall::dialect_from_template(model.gguf().str("tokenizer.chat_template").unwrap_or(""));
    Loaded { model, session, tok, info: m.clone(), dialect, placement, gpu_layers, vision: None, llama: None, images: Default::default() }
}

/// Where a session for `model` runs: an NVIDIA card (with `gpu_layers` forced, or as many as fit),
/// else the processor with the NVIDIA card multiplying its weights (GPU assist, for layer types the
/// card does not run whole yet), else another graphics card through the portable backend, else the
/// processor; with the plan.
fn place(model: &Model, ctx: usize, gpu_layers: Option<usize>) -> (Session, Option<cerebro_engine::placement::Plan>) {
    match Session::new_gpu_with(model, ctx, gpu_layers) {
        Ok(s) => {
            let p = s.gpu.as_ref().map(|g| g.plan.clone());
            (s, p)
        }
        Err(e) if gpu_layers != Some(0) && matches!(e, cerebro_engine::Error::Unsupported(ref m) if m.contains("layer types")) => {
            match Session::new_assisted(model, ctx) {
                Ok(s) => {
                    let mut p = cerebro_engine::gpu::plan_for(model, ctx, 0, None).ok();
                    if let (Some(p), Some((placed, cands, bytes, _))) = (p.as_mut(), model.offload_stats()) {
                        p.notes.insert(0, format!(
                            "The processor runs this model's layers and the graphics card multiplies its weights: {placed} of {cands} weight matrices ({:.1} GiB) are on the card.",
                            bytes as f64 / (1u64 << 30) as f64
                        ));
                    }
                    (s, p)
                }
                Err(ae) => {
                    eprintln!("GPU assist: {ae}");
                    (Session::new(model, ctx), cerebro_engine::gpu::plan_for(model, ctx, 0, None).ok())
                }
            }
        }
        Err(e) => {
            // No NVIDIA GPU took it: another graphics card through the portable backend (Vulkan /
            // DX12 / Metal) when one is there and the model is a kind it runs, else the processor.
            let portable = if gpu_layers == Some(0) { None } else { Session::new_portable(model, ctx).map_err(|pe| eprintln!("Other graphics cards: {pe}")).ok() };
            match portable {
                Some(s) => {
                    let p = s.portable.as_ref().map(|g| g.plan(model));
                    (s, p)
                }
                None => {
                    eprintln!("Running on the processor: {e}");
                    // The processor-only plan still says what RAM and the drive will do.
                    let mut p = cerebro_engine::gpu::plan_for(model, ctx, 0, None).ok();
                    if let (Some(p), Some(0)) = (p.as_mut(), gpu_layers) {
                        p.notes[0] = "No layers on the graphics card (chosen when loading): everything runs on the processor.".to_owned();
                    }
                    (Session::new(model, ctx), p)
                }
            }
        }
    }
}

/// The layers running on the graphics card, whichever backend holds the model.
fn gpu_layers_of(l: &Loaded) -> usize {
    l.session.gpu.as_ref().map_or(0, |g| g.gpu_layers()) + l.session.portable.as_ref().map_or(0, |p| p.gpu_layers())
}

fn loaded_info(l: &Loaded, ctx: usize) -> LoadedInfo {
    let g = l.session.gpu.as_ref();
    let portable = l.session.portable.as_ref();
    LoadedInfo {
        model: l.info.clone(),
        ctx,
        on_gpu: g.is_some() || portable.is_some() || l.model.offload_stats().is_some(),
        vram_bytes: g.map_or(0, |g| g.vram_bytes) + portable.map_or(0, |p| p.bytes as usize) + l.model.offload_stats().map_or(0, |o| o.2),
        experts: g.and_then(|g| g.expert_cache()),
        context_used: l.session.tokens.len(),
        last_prefill_tps: None,
        dialect: l.dialect,
        placement: l.placement.clone(),
        gpu_layers: g.map_or(0, |g| g.gpu_layers()) + portable.map_or(0, |p| p.gpu_layers()),
    }
}

fn save_profile(l: &Loaded) {
    if let Some(g) = l.session.gpu.as_ref() {
        g.save_expert_profile();
    }
    {
        l.model.save_expert_profile();
    }
}

/// What run_chat needs to run tools: the MCP host (if any), the approvals waiting for the user, and
/// where long results are kept whole.
struct Tooling<'a> {
    host: Option<Arc<dyn ToolHost>>,
    approvals: &'a Arc<Mutex<std::collections::HashMap<String, Sender<bool>>>>,
    answers: &'a Answers,
    spill_dir: Option<PathBuf>,
}

/// Waits for the user's answer to call `id` (registered by the caller) until it arrives, the reply
/// is stopped, or the approval timeout passes. None means no answer.
fn wait_answer(answers: &Answers, id: &str, rx: std::sync::mpsc::Receiver<serde_json::Value>, cancel: &AtomicBool) -> Option<serde_json::Value> {
    let deadline = Instant::now() + APPROVAL_TIMEOUT;
    let out = loop {
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(v) => break Some(v),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if !cancel.load(Ordering::SeqCst) && Instant::now() < deadline => continue,
            Err(_) => break None,
        }
    };
    answers.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
    out
}

/// A tool result longer than `max` characters: the whole text goes to `<spill>/<call>.txt` and the
/// model gets its start and end plus the file's path (which read_file may open). Without a spill
/// folder, or if writing fails, the result is cut instead; the call itself never fails over this.
pub fn spill_or_truncate(text: &str, max: usize, spill: Option<&Path>, call_id: &str) -> String {
    let total = text.chars().count();
    if total <= max {
        return text.to_owned();
    }
    let Some(dir) = spill else { return crate::toolcall::truncate_result(text, max) };
    let path = dir.join(format!("{}.txt", call_id.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_")));
    if std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, text)).is_err() {
        return crate::toolcall::truncate_result(text, max);
    }
    let head_n = max * 3 / 4;
    let tail_n = max - head_n;
    let head: String = text.chars().take(head_n).collect();
    let tail: String = text.chars().skip(total - tail_n).collect();
    format!("{head}\n[... {} characters omitted; full output: {}]\n{tail}", total - head_n - tail_n, path.display())
}

/// One round's output: what the user saw, the raw text (tool markup included) for the history, and
/// the calls the model made.
struct Round {
    raw: String,
    calls: Vec<crate::toolcall::ToolCall>,
    bad: Vec<(String, String)>,
}

/// The first round of a chat: restore the best saved cache for this model when it continues `ids`
/// further than the session's own tokens already do (a cache that diverges is no help on a
/// recurrent model, and `best` only returns a prefix of `ids` at all).
fn restore_cache(l: &mut Loaded, store: &Arc<Mutex<KvStore>>, ids: &[u32]) {
    let fingerprint = cerebro_engine::model::model_fingerprint(&l.model);
    let path = {
        let mut kv = store.lock().unwrap_or_else(|e| e.into_inner());
        kv.scan(&l.model);
        let own = kv_store::common_prefix(&l.session.tokens, ids);
        match kv.best(fingerprint, ids) {
            Some((path, len)) if len > own + kv_store::MIN_GAIN => path,
            _ => return,
        }
    };
    let mut file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(error) => return eprintln!("A saved chat cache could not be opened: {error}"),
    };
    if let Err(error) = l.session.load_kv(&l.model, &mut file) {
        eprintln!("A saved chat cache could not be restored: {error}");
    }
}

/// Keeps this conversation's cache on disk for next time: only a long chat whose history would take
/// more than a moment to read again is worth a file (`tokens / the last prefill speed`).
fn save_cache(l: &mut Loaded, store: &Arc<Mutex<KvStore>>) {
    // A chat with images is not saved yet (the engine refuses: image positions are not stored).
    if !l.session.images.is_empty() {
        return;
    }
    let tokens = l.session.tokens.len();
    if tokens < kv_store::MIN_SAVE_TOKENS || !kv_store::worth_saving(tokens, l.session.stats.prefill_tps()) {
        return;
    }
    let mut bytes = Vec::new();
    if let Err(error) = l.session.save_kv(&l.model, &mut bytes) {
        return eprintln!("A chat cache could not be written: {error}");
    }
    let fingerprint = cerebro_engine::model::model_fingerprint(&l.model);
    let mut kv = store.lock().unwrap_or_else(|e| e.into_inner());
    kv.save(fingerprint, &l.session.tokens, bytes);
}

/// Measures this machine's processor settings on the loaded model (Turbo screen) and streams every
/// finished trial to `events`. The session's conversation cache is saved first and put back after,
/// so the user's chat does not have to be read again; a session that cannot save simply starts from
/// the measurement prompt.
fn run_tune(l: &mut Loaded, events: &TuneSink, cancel: &Arc<AtomicBool>) -> Result<tune::Outcome, String> {
    if l.llama.is_some() {
        return Err("This model runs on llama.cpp's Vulkan build on this graphics card, so Cerebro's processor settings do not change its speed: there is nothing to tune.".to_owned());
    }
    let mut saved = Vec::new();
    let saved_ok = match l.session.save_kv(&l.model, &mut saved) {
        Ok(()) => true,
        Err(error) => {
            eprintln!("A chat cache could not be saved before tuning: {error}");
            false
        }
    };
    let sink = events.clone();
    let mut on_trial = |t: &tune::Trial| sink(TuneMachineEvent::Trial(t.clone()));
    let result = tune::tune(&l.model, &mut l.session, cancel, &mut on_trial);
    if saved_ok {
        if let Err(error) = l.session.load_kv(&l.model, &mut saved.as_slice()) {
            eprintln!("The chat cache could not be restored after tuning: {error}");
        }
    }
    match result {
        Ok(outcome) => {
            events(TuneMachineEvent::Done { outcome: outcome.clone() });
            Ok(outcome)
        }
        Err(error) => {
            let message = match error {
                cerebro_engine::Error::Request(detail) if detail == "tuning was cancelled" => "Tuning was stopped.".to_owned(),
                other => other.to_string(),
            };
            events(TuneMachineEvent::Error { message: message.clone() });
            Err(message)
        }
    }
}

/// Whether the model thinks, and for how many pieces: the UI's `native` parameters
/// ({reasoningEffort, thinkingBudget}); absent or a budget of 0 means thinking off, and a budget
/// the UI sends as null (no limit) means as long as the reply allows.
fn thinking(native: Option<&serde_json::Value>) -> (bool, usize) {
    let Some(n) = native else { return (false, 0) };
    match n.get("thinkingBudget") {
        Some(serde_json::Value::Number(b)) => {
            let b = b.as_f64().unwrap_or(0.0);
            if b <= 0.0 {
                (false, 0)
            } else {
                (true, if b >= usize::MAX as f64 { usize::MAX } else { b as usize })
            }
        }
        _ => (true, usize::MAX),
    }
}

/// Tokens of the reply kept for the answer when thinking has a budget (ui/src/lib/chat/prompt.ts).
const ANSWER_RESERVE: usize = 1024;

const REVIEW_SELF: &str = "Check your answer above for mistakes in facts, numbers, logic or code, and for any part of my question it does not answer. Recompute every number. Reply with one word only: CORRECT if it has no mistake and answers everything, WRONG otherwise.";
const REVIEW_SYSTEM: &str = "You check answers carefully. Recompute numbers, retrace logic and code, and test each claim.";
const REVIEW_FRESH: &str = "Is the proposed answer correct and complete for the question? Recompute every number. Reply with one word only: CORRECT or WRONG.";
/// After a WRONG verdict: the correction, in the same conversation as the verdict.
const REVIEW_FIX: &str = "Write the full corrected answer to the original question, for the person who asked it. Do not mention the check or the earlier answer.";

/// A reply without any <think> block.
fn strip_think(s: &str) -> String {
    let mut t = s.to_owned();
    while let Some(a) = t.find("<think>") {
        match t[a..].find("</think>") {
            Some(b) => t.replace_range(a..a + b + 8, ""),
            None => t.truncate(a),
        }
    }
    t.trim().to_owned()
}

/// A review's verdict: Some(true) for CORRECT, Some(false) for WRONG, None when it gave neither
/// (treated as nothing found, so an unclear review never replaces the answer).
fn review_verdict(v: &str) -> Option<bool> {
    let w = v.trim().trim_matches(|c: char| !c.is_ascii_alphabetic()).to_ascii_uppercase();
    let first = w.split(|c: char| !c.is_ascii_alphabetic()).find(|s| !s.is_empty()).unwrap_or("");
    match first {
        "CORRECT" => Some(true),
        "WRONG" | "INCORRECT" => Some(false),
        _ => None,
    }
}

fn run_chat(l: &mut Loaded, req: &ChatRequest, events: &ChatSink, st: &Arc<Mutex<Status>>, cancel: &Arc<AtomicBool>, tooling: Tooling<'_>, kv: Option<&Arc<Mutex<KvStore>>>) {
    use crate::toolcall::{self, Piece, Scanner};
    let (think, think_budget) = thinking(req.native.as_ref());
    // The model's own chat format (Gemma 4 and Llama 3 are not ChatML); their thinking is not
    // rendered yet, so they answer directly.
    let fmt = cerebro_engine::ChatFormat::of(l.model.gguf());
    let think = think && fmt.think_tags();
    let think_budget = think_budget.min(req.sampler.max_tokens.saturating_sub(ANSWER_RESERVE).max(ANSWER_RESERVE / 4));
    let plan_mode = req.client_tools.is_none() && req.tools.as_ref().is_some_and(|t| t.plan_mode);
    let mut ctx = ToolContext::from_request(req.tools.as_ref());
    if plan_mode {
        // Nothing changes until the plan is approved.
        ctx.access = Access::ReadOnly;
    }
    // Tools the model may call: the host's list, names made unique across servers ("server__tool").
    let tools: Vec<(String, toolcall::ToolSpec)> = match (&req.client_tools, &req.tools, &tooling.host) {
        (Some(ct), _, _) => ct.iter().map(|t| (t.name.clone(), t.clone())).collect(),
        (None, Some(tr), host) if host.is_none() => crate::agent_tools::specs(crate::builtin_tools::SERVER, tr.plan_mode).into_iter().map(|t| (t.name.clone(), t)).collect(),
        (None, Some(tr), Some(h)) => {
            let plan_mode = tr.plan_mode;
            let specs = h.tools();
            let mut seen = std::collections::HashMap::<String, usize>::new();
            for t in &specs {
                *seen.entry(t.name.clone()).or_default() += 1;
            }
            specs
                .into_iter()
                .chain(crate::agent_tools::specs(crate::builtin_tools::SERVER, plan_mode))
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
            if req.client_tools.is_none() {
                sys.push_str("\n\n");
                sys.push_str(ctx.access.statement());
            }
            if plan_mode {
                sys.push_str("\n\n");
                sys.push_str(crate::agent_tools::PLAN_PROMPT);
            }
        }
        sys
    };
    // Images go before their turn's text, as the model's own template places them; `pics` lists
    // them in prompt order. MARK is stripped from everything else so only placeholders carry it.
    let mark = crate::vision_input::MARK;
    let mut pics: Vec<String> = Vec::new();
    let mut convo: Vec<(String, String)> = Vec::new();
    for t in &req.turns {
        let role = if t.role == "assistant" { "assistant" } else { "user" };
        let mut text = t.text.replace(mark, "");
        if role == "user" {
            let mut pre = String::new();
            for d in t.images() {
                pics.push(d.to_owned());
                pre.push_str(&crate::vision_input::placeholder());
            }
            text = pre + &text;
        }
        convo.push((role.to_owned(), text));
    }
    let system = system.replace(mark, "");

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
            input,
            output: String::new(),
            error: None,
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
    // When the first and the latest piece arrived, and how many: the writing speed is the pieces
    // after the first over that time, so a two-word reply is not charged the step that ends it.
    let mut piece_t: Option<(std::time::Instant, std::time::Instant, usize)> = None;
    let embeds = match encode_images(l, &pics) {
        Ok(v) => v,
        Err(e) => {
            failed = Some(e);
            Vec::new()
        }
    };

    for round in 0..=MAX_TOOL_ROUNDS {
        if failed.is_some() {
            break;
        }
        let turns: Vec<(&str, &str)> = convo.iter().map(|(r, t)| (r.as_str(), t.as_str())).collect();
        let text = fmt.render((!system.is_empty()).then_some(system.as_str()), &turns, think);
        // Text pieces around the images (one piece when there are none). Tool results added in
        // later rounds may contain the mark: only the first `embeds.len()` count.
        let pieces: Vec<Vec<u32>> = crate::vision_input::split_prompt(&text, embeds.len()).iter().map(|p| l.tok.encode(p)).collect();
        let ids: Vec<u32> = pieces.concat();
        let image_tokens: usize = embeds.iter().take(pieces.len() - 1).map(|e| e.cols * e.rows).sum();
        set(st, |s| s.live_prompt = ids.len() + image_tokens);
        // Returning to a long chat: start from its saved cache (Settings > Protections) instead of
        // reading the history again. Restored before `generate`, so its reused count is the feed's.
        if round == 0 && req.keep_cache && embeds.is_empty() {
            if let Some(store) = kv {
                restore_cache(l, store, &ids);
            }
        }
        // Reasoning (Qwen3: <think> ... </think>) streams as `thinking`; the answer goes through the
        // tool-call scanner, so tool markup never reaches the screen.
        let mut in_think = think;
        let mut think_pieces = 0usize;
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
        // The thinking budget: once it is spent the generation stops, "</think>" is appended and it
        // continues with the answer (`cont`); the cache makes the continuation free to read.
        let mut cont: Option<Vec<u32>> = None;
        let mut first_ttft: Option<f64> = None;
        let result = loop {
        let mut budget_hit = false;
        let on_piece = |piece: &str| {
            if in_think {
                think_pieces += 1;
                if think_pieces > think_budget {
                    let t = std::mem::take(&mut pending).replace("<think>", "");
                    if !t.is_empty() {
                        events(ChatEvent::Thinking { text: t });
                    }
                    in_think = false;
                    budget_hit = true;
                    return false;
                }
            }
            if !started_ev {
                started_ev = true;
                events(ChatEvent::Start { ttft_ms: t0.elapsed().as_secs_f64() * 1000.0 });
            }
            set(st, |s| s.activity = "generating");
            n += 1;
            let now = std::time::Instant::now();
            piece_t = Some(piece_t.map_or((now, now, 1), |(a, _, k)| (a, now, k + 1)));
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
        };
        let r = if let Some(c) = &cont {
            generate(&l.model, &mut l.session, &l.tok, c, &cfg, on_piece)
        } else if embeds.is_empty() {
            generate(&l.model, &mut l.session, &l.tok, &ids, &cfg, on_piece)
        } else {
            let mut parts: Vec<Part<'_>> = Vec::with_capacity(pieces.len() * 2);
            for (i, p) in pieces.iter().enumerate() {
                if i > 0 {
                    let e = &embeds[i - 1];
                    parts.push(Part::Image(ImageTokens { cols: e.cols, rows: e.rows, dim: e.dim, data: &e.data }));
                }
                parts.push(Part::Text(p));
            }
            generate_parts(&l.model, &mut l.session, &l.tok, &parts, &cfg, on_piece)
        };
        match &r {
            Ok(g) if budget_hit && !cancel.load(Ordering::SeqCst) => {
                first_ttft.get_or_insert(g.ttft_ms);
                out_tokens += g.output_tokens;
                prompt_tokens += g.prompt_tokens;
                reused_tokens += g.reused_tokens;
                decode_ms += g.total_ms - g.ttft_ms;
                let mut c = l.session.tokens.clone();
                c.extend(l.tok.encode("\n</think>\n\n"));
                cont = Some(c);
            }
            _ => break r,
        }
        };
        thinking_tokens += think_pieces;
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
        // A reply that thought but wrote no answer (seen on the owner's machine, not reproduced
        // yet): answer in a second pass with thinking off, the path that always answers, so a
        // thinking reply never ends empty. The log keeps what the model wrote, for the root cause.
        if think && thinking_tokens > 0 && answer.trim().is_empty() && out.calls.is_empty() && out.bad.is_empty() && result.is_ok() && !cancel.load(Ordering::SeqCst) {
            eprintln!("chat: thinking ({thinking_tokens} pieces) ended without an answer; answering with thinking off. Last output: {:?}", out.raw.chars().rev().take(200).collect::<String>().chars().rev().collect::<String>());
            let plain = l.tok.encode(&fmt.render((!system.is_empty()).then_some(system.as_str()), &turns, false));
            let again = generate(&l.model, &mut l.session, &l.tok, &plain, &cfg, |piece| {
                answer.push_str(piece);
                out.raw.push_str(piece);
                events(ChatEvent::Token { text: piece.to_owned() });
                !cancel.load(Ordering::SeqCst)
            });
            if let Ok(g) = &again {
                out_tokens += g.output_tokens;
            }
        }
        match result {
            Ok(g) => {
                if round == 0 {
                    ttft_ms = first_ttft.unwrap_or(g.ttft_ms);
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
        if req.client_tools.is_some() {
            // The client runs its own tools: hand the calls over and end the reply. A call that could
            // not be read goes back as the text the model wrote, so the client sees what happened.
            for (raw, _) in &out.bad {
                answer.push_str(raw);
                events(ChatEvent::Token { text: raw.clone() });
            }
            for (i, call) in out.calls.iter().enumerate() {
                events(ChatEvent::Tool { id: format!("call_{}_{i}", now_ms()), server: String::new(), name: call.name.clone(), arguments: call.arguments.clone(), state: "requested", result: None, error: None });
            }
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
            if server == crate::builtin_tools::SERVER && crate::agent_tools::is_agent_tool(&tool) {
                use crate::agent_tools as at;
                let reply: Result<String, String> = match tool.as_str() {
                    at::TODO => at::todo(&call.arguments),
                    at::ASK | at::EXIT_PLAN => {
                        let shown = if tool == at::ASK { at::questions(&call.arguments) } else { at::plan(&call.arguments).map(|p| serde_json::json!({ "plan": p })) };
                        match shown {
                            Err(e) => Err(e),
                            Ok(args) => {
                                let (tx, rx) = channel::<serde_json::Value>();
                                tooling.answers.lock().unwrap_or_else(|e| e.into_inner()).insert(call_id.clone(), tx);
                                let state = if tool == at::ASK { "question" } else { "plan" };
                                events(ChatEvent::Tool { id: call_id.clone(), server: server.clone(), name: tool.clone(), arguments: args, state, result: None, error: None });
                                set(st, |s| s.activity = "waiting");
                                match wait_answer(tooling.answers, &call_id, rx, cancel) {
                                    None => Err("The user did not answer. Continue without the answer, or say what you need.".to_owned()),
                                    Some(a) if tool == at::ASK => at::answers_text(&a).ok_or_else(|| "The answer could not be read.".to_owned()),
                                    Some(a) => Ok(at::plan_reply(&a)),
                                }
                            }
                        }
                    }
                    _ => Err("Unknown tool.".to_owned()),
                };
                match reply {
                    Ok(text) => {
                        events(tool_ev("done", Some(text.clone()), None));
                        responses.push(toolcall::tool_response(&text));
                    }
                    Err(e) => {
                        events(tool_ev("error", None, Some(e.clone())));
                        responses.push(toolcall::tool_response(&format!("Error: {e}")));
                    }
                }
                continue;
            }
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
                    let spill = tooling.spill_dir.as_deref();
                    let text = spill_or_truncate(&text, MAX_TOOL_RESULT_CHARS, spill, &call_id);
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
        events(ChatEvent::Error { message: message.clone() });
        set(st, |s| {
            s.totals.errors += 1;
            if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
                r.state = "error";
                r.error = Some(message);
            }
        });
        return;
    }
    // Review passes (effort levels): the reply reread in this conversation, then checked again in a
    // fresh context; a pass that finds a problem replaces the reply with its correction.
    let mut extra_ms = 0.0;
    let tools_used = convo.len() > req.turns.len();
    if req.passes > 0 && fmt == cerebro_engine::ChatFormat::ChatMl && !stopped && !cancel.load(Ordering::SeqCst) && !answer.trim().is_empty() && !tools_used && pics.is_empty() {
        let tp = Instant::now();
        set(st, |s| s.activity = "checking");
        let question = req.turns.last().map(|t| t.text.as_str()).unwrap_or("");
        let rcfg = GenConfig { sampler: SamplerConfig { temperature: 0.2, top_k: 20, top_p: 0.9, min_p: 0.05, repeat_penalty: 1.0, seed: now_ms() }, max_tokens: cfg.max_tokens };
        for pass in 1..=req.passes.min(2) {
            let text = if pass == 1 {
                let mut turns: Vec<(&str, &str)> = convo.iter().map(|(r, t)| (r.as_str(), t.as_str())).collect();
                turns.push(("assistant", answer.trim()));
                turns.push(("user", REVIEW_SELF));
                cerebro_engine::chatml((!system.is_empty()).then_some(system.as_str()), &turns, false)
            } else {
                let ask = format!("Question:\n{question}\n\nProposed answer:\n{}\n\n{REVIEW_FRESH}", answer.trim());
                cerebro_engine::chatml(Some(REVIEW_SYSTEM), &[("user", ask.as_str())], false)
            };
            let vcfg = GenConfig { sampler: rcfg.sampler.clone(), max_tokens: 8 };
            let ids = l.tok.encode(&text);
            let mut verdict = String::new();
            let r = generate(&l.model, &mut l.session, &l.tok, &ids, &vcfg, |p| {
                verdict.push_str(p);
                !cancel.load(Ordering::SeqCst)
            });
            if r.is_err() || cancel.load(Ordering::SeqCst) {
                break;
            }
            if review_verdict(&strip_think(&verdict)) != Some(false) {
                events(ChatEvent::Check { pass, text: if pass == 1 { "Reread the answer: no error found.".into() } else { "Checked again in a fresh context: no error found.".into() } });
                continue;
            }
            // WRONG: the correction, continuing the same conversation (the cache holds it).
            let fix = format!("{text}{}<|im_end|>\n<|im_start|>user\n{REVIEW_FIX}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", strip_think(&verdict));
            let ids = l.tok.encode(&fix);
            let mut reply = String::new();
            let r = generate(&l.model, &mut l.session, &l.tok, &ids, &rcfg, |p| {
                reply.push_str(p);
                !cancel.load(Ordering::SeqCst)
            });
            if r.is_err() || cancel.load(Ordering::SeqCst) {
                break;
            }
            let fixed = strip_think(&reply);
            if fixed.is_empty() {
                continue;
            }
            answer = fixed;
            events(ChatEvent::Revise { text: answer.clone() });
            events(ChatEvent::Check { pass, text: if pass == 1 { "Reread the answer, found a problem and corrected it.".into() } else { "Checked in a fresh context, found a problem and corrected it.".into() } });
        }
        extra_ms = tp.elapsed().as_secs_f64() * 1000.0;
    }
    // Keep this conversation's cache for next time (Settings > Protections, "Keep long chats
    // ready"); the serialising happens here, the file itself is written on a background thread.
    if req.keep_cache {
        if let Some(store) = kv {
            save_cache(l, store);
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = match piece_t {
        Some((a, b, k)) if k >= 2 && b > a => (k - 1) as f64 / (b - a).as_secs_f64(),
        _ if decode_ms > 0.0 => out_tokens as f64 / (decode_ms / 1000.0),
        _ => 0.0,
    };
    let length = !stopped && out_tokens >= cfg.max_tokens;
    events(ChatEvent::Done {
        stats: ChatStats { prompt_tokens, length, output_tokens: out_tokens, thinking_tokens, tok_per_sec: decode_tps, ttft_ms, total_ms, extra_ms, stopped },
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
            r.output = answer;
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
        let fmt = cerebro_engine::ChatFormat::of(self.l.model.gguf());
        let text = fmt.render(Some(system), &[("user", user)], self.think && fmt.think_tags());
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
            input: goal.clone(),
            output: String::new(),
            error: None,
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
    let mut llm = EngineLlm { l, cfg, think: thinking(req.native.as_ref()).0, events, st, cancel, t0, started: false, tokens: 0, decode_ms: 0.0, prompt_tokens: 0, reused_tokens: 0, ttft_ms: None };
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
            events(ChatEvent::Error { message: e.clone() });
            set(st, |s| {
                s.totals.errors += 1;
                if let Some(r) = s.requests.iter_mut().find(|r| r.id == id) {
                    r.state = "error";
                    r.error = Some(e);
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
    events(ChatEvent::Done { stats: ChatStats { prompt_tokens: 0, length: false, output_tokens: tokens, thinking_tokens: 0, tok_per_sec: tps, ttft_ms: ttft, total_ms, extra_ms: 0.0, stopped } });
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
            r.output = answer;
        }
    });
}

/// The tool's own name from the name the model saw ("server__tool" when names collided).
fn original_name(server: &str, exposed: &str) -> String {
    exposed.strip_prefix(&format!("{server}__")).unwrap_or(exposed).to_owned()
}

#[cfg(test)]
mod spill_tests {
    use super::spill_or_truncate;

    #[test]
    fn long_results_are_kept_whole_in_a_file() {
        let dir = std::env::temp_dir().join(format!("cerebro-spill-{}", std::process::id()));
        let text: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let out = spill_or_truncate(&text, 1000, Some(&dir), "t1-0");
        assert!(out.starts_with("line 0\n"), "head kept");
        assert!(out.trim_end().ends_with("line 4999"), "tail kept");
        let path = dir.join("t1-0.txt");
        assert!(out.contains(&path.display().to_string()));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "the whole result is on disk");
        assert_eq!(spill_or_truncate("short", 1000, Some(&dir), "t2"), "short");
        assert!(spill_or_truncate(&text, 1000, None, "t3").contains("[... truncated"), "no folder: cut instead");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod split_model_tests {
    use super::{file_size, split_parts};

    /// A split model is one model in the library, and its size is the size of all its parts.
    #[test]
    fn a_split_model_counts_every_part() {
        let dir = std::env::temp_dir().join(format!("cerebro-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("Model-00001-of-00003.gguf");
        let b = dir.join("Model-00002-of-00003.gguf");
        let c = dir.join("Model-00003-of-00003.gguf");
        let other = dir.join("Model-00001-of-00002.gguf");
        let plain = dir.join("plain.gguf");
        std::fs::write(&a, vec![0u8; 10]).unwrap();
        std::fs::write(&b, vec![0u8; 20]).unwrap();
        std::fs::write(&c, vec![0u8; 30]).unwrap();
        std::fs::write(&other, vec![0u8; 40]).unwrap();
        std::fs::write(&plain, vec![0u8; 50]).unwrap();
        // Every part of the same model reports the same total.
        assert_eq!(file_size(&a), 60);
        assert_eq!(file_size(&b), 60);
        assert_eq!(file_size(&c), 60);
        // A different split, and a file that is not split, are counted on their own.
        assert_eq!(file_size(&other), 40);
        assert_eq!(file_size(&plain), 50);
        assert_eq!(split_parts(&a), Some(("Model".to_owned(), 3)));
        assert_eq!(split_parts(&plain), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod tune_event_tests {
    use super::*;
    use cerebro_engine::tune::{Outcome, Setting, Trial};

    #[test]
    fn tune_events_match_the_ui_shape() {
        let trial = Trial {
            phase: "threads",
            label: "12 threads".into(),
            settings: vec![Setting { id: "cpu_threads", value: 12 }],
            tps: 48.3,
            reps: vec![48.3, 48.2, 48.9],
        };
        let v = serde_json::to_value(TuneMachineEvent::Trial(trial.clone())).expect("trial json");
        assert_eq!(v["type"], "trial");
        assert_eq!(v["label"], "12 threads");
        assert_eq!(v["tps"], 48.3);
        assert_eq!(v["settings"][0]["id"], "cpu_threads");
        assert_eq!(v["reps"][0], 48.3);
        let outcome = Outcome {
            machine: "12th Gen Intel(R) Core(TM) i9-12900H".into(),
            model: "0123456789abcdef".into(),
            gpu_layers: 0,
            settings: vec![Setting { id: "cpu_threads", value: 12 }],
            start: vec![Setting { id: "cpu_threads", value: 16 }],
            start_tps: 47.2,
            tuned_tps: 48.2,
            gain: 0.022,
            kept: true,
            busy: Some(0.06),
            note: "As fast with 4 fewer threads, so it runs cooler.".into(),
            trials: vec![trial],
            seconds: 52.0,
        };
        let v = serde_json::to_value(TuneMachineEvent::Done { outcome }).expect("done json");
        assert_eq!(v["type"], "done");
        assert_eq!(v["outcome"]["tunedTps"], 48.2);
        assert_eq!(v["outcome"]["startTps"], 47.2);
        assert_eq!(v["outcome"]["gpuLayers"], 0);
        let v = serde_json::to_value(TuneMachineEvent::Error { message: "Tuning was stopped.".into() }).expect("error json");
        assert_eq!(v, serde_json::json!({ "type": "error", "message": "Tuning was stopped." }));
    }
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

    /// Tune for this machine on a real model on the processor: trials stream out, an outcome comes
    /// back, and the user's conversation cache is the same afterwards (saved before, restored after).
    /// `cargo test -p cerebro-desktop tune_on_a_real_model -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn tune_on_a_real_model_keeps_the_conversation() {
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into());
        let model = describe(Path::new(&path), "test").expect("model");
        cerebro_engine::switches::set(cerebro_engine::switches::GPU_ENABLED, 0).unwrap();
        let mut l = load(&model, 4096, None).expect("load");
        let ids = l.tok.encode("A conversation the user had before tuning, long enough to matter a little.");
        l.session.feed(&l.model, &ids).unwrap();
        let before = l.session.tokens.clone();
        let trials: Arc<Mutex<Vec<String>>> = Arc::default();
        let tr = trials.clone();
        let sink: TuneSink = Arc::new(move |e| {
            if let TuneMachineEvent::Trial(t) = e {
                tr.lock().unwrap().push(format!("{} {:.1}", t.label, t.tps));
            }
        });
        let outcome = run_tune(&mut l, &sink, &Arc::new(AtomicBool::new(false))).expect("tune");
        println!("{} trials; {}", trials.lock().unwrap().len(), outcome.note);
        assert!(!outcome.trials.is_empty() && trials.lock().unwrap().len() == outcome.trials.len());
        assert_eq!(l.session.tokens, before, "the conversation cache was not restored");
        cerebro_engine::switches::reset();
    }

    /// A machine without a usable NVIDIA GPU (run with CUDA_VISIBLE_DEVICES=-1 to hide CUDA): the
    /// load falls through to the portable backend, reports where the model went, and answers.
    #[test]
    #[ignore]
    fn without_cuda_a_real_model_loads_on_another_graphics_card() {
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into());
        let model = describe(Path::new(&path), "test").expect("model");
        let l = load(&model, 4096, None).expect("load");
        assert!(l.session.gpu.is_none(), "CUDA was not hidden: run with CUDA_VISIBLE_DEVICES=-1");
        let p = l.session.portable.as_ref().expect("on the portable backend");
        let plan = l.placement.as_ref().expect("placement");
        println!("{} ({}): {:?}", p.info.name, p.info.backend, plan.notes);
        assert!(plan.notes[0].starts_with("All 28 layers on the "));
        let info = loaded_info(&l, 4096);
        assert!(info.on_gpu && info.gpu_layers == 28 && info.vram_bytes > 0);
        let engine = Engine::start();
        engine.load(model.clone(), 4096, None).unwrap();
        let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
        let ev = events.clone();
        let sink: ChatSink = Arc::new(move |e: ChatEvent| ev.lock().unwrap().push(e));
        let req = ChatRequest {
            model_id: model.id.clone(),
            system_prompt: String::new(),
            turns: vec![ChatTurn { role: "user".into(), text: "What is the capital of France? One word.".into(), attachments: Vec::new() }],
            sampler: ChatSampler { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 1, repeat_penalty: 1.0, max_tokens: 400 },
            native: None,
            tools: None,
            orchestrate: None,
            passes: 0,
            keep_cache: false,
            client_tools: None,
        };
        engine.chat(req, sink).unwrap();
        let text: String = events.lock().unwrap().iter().filter_map(|e| if let ChatEvent::Token { text } = e { Some(text.clone()) } else { None }).collect();
        println!("reply: {text:?}");
        assert!(text.contains("Paris"), "{text}");
    }

    /// A vision model sees a photo attached in Chat: the projector is found beside the model, the
    /// data URL decoded and encoded, and the reply is about the picture (a newspaper front page about
    /// the Moon landing). `cargo test -p cerebro-desktop sees_an_attached -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn a_vision_model_sees_an_attached_image() {
        use base64::Engine as _;
        let model = describe(Path::new(r"C:\models\dev\qwen35\Qwen3.5-0.8B-Q8_0.gguf"), "test").expect("model");
        assert_eq!(model.kind, "vision", "the projector beside the model was not found");
        let jpeg = std::fs::read(r"C:\Users\noahm\Projects\cerebro-engine\.ref\test-1.jpeg").expect("test photo");
        let data = format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(jpeg));
        let engine = Engine::start();
        engine.load(model.clone(), 4096, None).unwrap();
        let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
        let ev = events.clone();
        let sink: ChatSink = Arc::new(move |e: ChatEvent| ev.lock().unwrap().push(e));
        let req = ChatRequest {
            model_id: model.id.clone(),
            system_prompt: String::new(),
            turns: vec![ChatTurn { role: "user".into(), text: "What is this a picture of? One sentence.".into(), attachments: vec![TurnAttachment { data: Some(data) }] }],
            sampler: ChatSampler { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 1, repeat_penalty: 1.0, max_tokens: 80 },
            native: None,
            tools: None,
            orchestrate: None,
            passes: 0,
            keep_cache: false,
            client_tools: None,
        };
        engine.chat(req, sink).unwrap();
        let evs = events.lock().unwrap();
        let text: String = evs.iter().filter_map(|e| if let ChatEvent::Token { text } = e { Some(text.clone()) } else { None }).collect();
        println!("reply: {text:?}");
        let lower = text.to_lowercase();
        assert!(lower.contains("newspaper") || lower.contains("times") || lower.contains("moon"), "{text}");
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
        engine.load(model.clone(), 4096, None).unwrap();
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
            turns: vec![ChatTurn { role: "user".into(), text: "What is the weather in Paris right now? Use the tool.".into(), attachments: Vec::new() }],
            sampler: ChatSampler { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 1, repeat_penalty: 1.0, max_tokens: 400 },
            native: None,
            tools: Some(ToolsRequest { rules: Default::default(), default_rule: "ask".into(), offline: false, allow_local_fetch: false, access: Access::Workspace, show_secrets: false, blind_writes: false, plan_mode: false }),
            orchestrate: None,
            passes: 0,
            keep_cache: false,
            client_tools: None,
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

    /// The Speed <-> Accuracy slider's five levels on a real model (ui/src/lib/chat/modes.ts and
    /// prompt.ts, built here the way the window builds them): time, tokens and whether each answer
    /// is right, per question. Measures; asserts only that every level answered.
    /// `CEREBRO_TEST_MODEL=... cargo test --release -p cerebro-desktop slider_levels -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn slider_levels_on_a_real_model() {
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\qwen35\Qwen3.5-4B-Q8_0.gguf".into());
        let Some(model) = describe(Path::new(&path), "test") else { panic!("model not readable: {path}") };
        let engine = Engine::start();
        engine.load(model.clone(), 8192, None).unwrap();
        let fastest = "Answer directly and briefly. No preamble, no restating the question.\n\nAnswer first, in as few words as the question needs. Give details only when asked.";
        let fast = "You are a helpful assistant on this computer. Be concise and correct.\n\nAnswer first, in as few words as the question needs. Give details only when asked.";
        let balanced = "You are a helpful assistant running locally on this computer.\n\nAnswer first, in as few words as the question needs. Give details only when asked.";
        let accurate = std::env::var("CEREBRO_ACCURATE_PROMPT").unwrap_or_else(|_| "You are a careful assistant running locally on this computer. Correctness matters more than speed.\n\nBefore answering, work out exactly what is being asked. For math, logic and code, work step by step and recompute anything you are not sure of.\n\nAnswer clearly with the context a careful reader needs, without padding.\n\nAssume every part of your answer is wrong until you have checked it.".into());
        let sampler = |t: f32, k: usize, p: f32, minp: f32, rp: f32, max: usize| ChatSampler { temperature: t, top_p: p, min_p: minp, top_k: k, repeat_penalty: rp, max_tokens: max };
        // (label, system prompt, thinking budget (None: off), passes, sampler, orchestrate)
        type Level = (&'static str, String, Option<f64>, u8, ChatSampler, Option<OrchestrateRequest>);
        let levels: Vec<Level> = vec![
            ("1 Fastest", fastest.into(), None, 0, sampler(0.0, 1, 1.0, 0.0, 1.0, 1024), None),
            ("2 Fast", fast.into(), None, 0, sampler(0.5, 40, 0.95, 0.05, 1.05, 2048), None),
            ("3 Balanced", balanced.into(), Some(1024.0), 0, sampler(0.7, 40, 0.95, 0.05, 1.05, 4096), None),
            ("4 Accurate", accurate.clone(), Some(16384.0), 1, sampler(0.4, 40, 0.95, 0.05, 1.05, 8192), None),
            ("5 Maximum", accurate, Some(16384.0), 2, sampler(0.4, 40, 0.95, 0.05, 1.05, 8192), std::env::var_os("CEREBRO_SLIDER_ORCH").map(|_| OrchestrateRequest { workers: 5, rounds: 3 })),
        ];
        // (question, every one of these must appear in the answer, lower-cased)
        let questions: [(&str, &[&str]); 8] = [
            ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", &["0.05"]),
            ("How many times does the letter r appear in the word strawberry?", &["3"]),
            ("Is 391 a prime number? Answer yes or no, then give its factors if it has any.", &["17", "23"]),
            ("What is 17 x 24 + 13 x 19?", &["655"]),
            ("If today is Wednesday, what day of the week will it be 100 days from now?", &["friday"]),
            ("How many positive divisors does 360 have?", &["24"]),
            ("I have 3 apples. I eat 2, buy 5 more, then give half of what I have to a friend. How many apples do I have now?", &["3"]),
            ("What is the sum of the digits of 2 to the power of 15?", &["26"]),
        ];
        let only = std::env::var("CEREBRO_SLIDER_LEVELS").unwrap_or_default();
        for (label, sys, budget, passes, smp, orch) in levels {
            if !only.is_empty() && !only.split(',').any(|n| label.starts_with(n.trim())) {
                continue;
            }
            let mut line = format!("{label:<11}");
            let (mut right, mut secs, mut toks, mut think) = (0, 0.0, 0usize, 0usize);
            for (q, ok) in questions {
                let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
                let ev = events.clone();
                let sink: ChatSink = Arc::new(move |e: ChatEvent| ev.lock().unwrap().push(e));
                let req = ChatRequest {
                    model_id: model.id.clone(),
                    system_prompt: sys.clone(),
                    turns: vec![ChatTurn { role: "user".into(), text: q.into(), attachments: Vec::new() }],
                    sampler: smp.clone(),
                    native: budget.map(|b| serde_json::json!({ "reasoningEffort": "high", "thinkingBudget": b })),
                    tools: None,
                    orchestrate: orch,
                    passes,
                    keep_cache: false,
                    client_tools: None,
                };
                let t = Instant::now();
                engine.chat(req, sink).unwrap();
                secs += t.elapsed().as_secs_f64();
                let events = events.lock().unwrap();
                let mut answer = String::new();
                for e in events.iter() {
                    match e {
                        ChatEvent::Token { text } => answer.push_str(text),
                        ChatEvent::Revise { text } => answer = text.clone(),
                        ChatEvent::Done { stats } => {
                            toks += stats.output_tokens;
                            think += stats.thinking_tokens;
                        }
                        _ => {}
                    }
                }
                let low = answer.to_lowercase();
                let good = ok.iter().all(|w| low.contains(w)) && !(q.contains("prime") && low.trim_start().starts_with("yes"));
                right += good as usize;
                assert!(!answer.trim().is_empty() || orch.is_some(), "{label}: no answer to {q}");
                eprintln!("  [{label}] {}: {}", if good { "right" } else { "WRONG" }, answer.replace('\n', " ").chars().take(160).collect::<String>());
            }
            line.push_str(&format!(" {right}/{} right, {secs:6.1} s, {toks:5} tokens out, {think:5} thinking", questions.len()));
            println!("{line}");
        }
    }

    /// Cluster through the app: this computer shares a model (cluster.rs), the engine loads the same
    /// model with its second half on that worker over localhost, and a greedy reply is word for word
    /// the reply of the model loaded normally on the processor.
    /// `cargo test --release -p cerebro-desktop cluster_through_the_app -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cluster_through_the_app_matches_a_local_load() {
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into());
        let model = describe(Path::new(&path), "test").expect("model");
        cerebro_engine::switches::set(cerebro_engine::switches::GPU_ENABLED, 0).unwrap();
        let ask = |engine: &Engine| -> String {
            let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
            let ev = events.clone();
            let sink: ChatSink = Arc::new(move |e: ChatEvent| ev.lock().unwrap().push(e));
            let req = ChatRequest {
                model_id: model.id.clone(),
                system_prompt: String::new(),
                turns: vec![ChatTurn { role: "user".into(), text: "Name three planets and one fact about each.".into(), attachments: Vec::new() }],
                sampler: ChatSampler { temperature: 0.0, top_p: 1.0, min_p: 0.0, top_k: 1, repeat_penalty: 1.0, max_tokens: 80 },
                native: None,
                tools: None,
                orchestrate: None,
                passes: 0,
                keep_cache: false,
                client_tools: None,
            };
            engine.chat(req, sink).unwrap();
            let events = events.lock().unwrap();
            events.iter().filter_map(|e| if let ChatEvent::Token { text } = e { Some(text.as_str()) } else { None }).collect()
        };
        let local = Engine::start();
        local.load(model.clone(), 2048, Some(0)).unwrap();
        let want = ask(&local);
        local.unload().unwrap();

        let view = crate::cluster::share_start(Path::new(&path), &model.name).expect("share");
        let code = view.code.clone().unwrap();
        let addr = format!("127.0.0.1:{}", view.port.unwrap());
        let engine = Engine::start();
        let links = engine.load_cluster(model.clone(), 2048, 0.5, vec![crate::cluster::WorkerChoice { addr: addr.clone(), code: code.clone(), share: 0.5 }]).expect("cluster load");
        println!("links: {:?}", links.iter().map(|l| (l.lo, l.hi, l.rtt_ms, l.mb_per_s)).collect::<Vec<_>>());
        let got = ask(&engine);
        println!("local:   {want}\ncluster: {got}");
        assert_eq!(got, want, "the reply across computers must be the reply on one");
        // A wrong code is refused before anything runs.
        let wrong = engine.load_cluster(model.clone(), 2048, 0.5, vec![crate::cluster::WorkerChoice { addr, code: "000000".into(), share: 0.5 }]);
        assert!(wrong.unwrap_err().contains("not paired"));
        crate::cluster::share_stop();
        cerebro_engine::switches::reset();
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
        engine.load(model.clone(), 8192, None).unwrap();
        let events: Arc<Mutex<Vec<ChatEvent>>> = Arc::default();
        let ev = events.clone();
        let sink: ChatSink = Arc::new(move |e: ChatEvent| ev.lock().unwrap().push(e));
        let req = ChatRequest {
            model_id: model.id.clone(),
            system_prompt: String::new(),
            turns: vec![ChatTurn { role: "user".into(), text: "What is 17 multiplied by 23? Show the working briefly, then the number.".into(), attachments: Vec::new() }],
            sampler: ChatSampler { temperature: 0.2, top_p: 0.8, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens: 600 },
            native: None,
            tools: None,
            orchestrate: Some(OrchestrateRequest { workers: 2, rounds: 2 }),
            passes: 0,
            keep_cache: false,
            client_tools: None,
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

#[cfg(test)]
mod name_tests {
    #[test]
    fn model_names_read_like_people_write_them() {
        assert_eq!(super::display_name("Qwen_Qwen3.6-35B-A3B-Q4_K_M"), "Qwen3.6 35B A3B");
        assert_eq!(super::display_name("Qwen3.8-27B-IQ1_M"), "Qwen3.8 27B");
        assert_eq!(super::display_name("Qwen3-0.6B"), "Qwen3 0.6B");
        assert_eq!(super::display_name("sd-turbo"), "SD Turbo");
        assert_eq!(super::display_name("Qwen3 8B Awq Compatible Instruct"), "Qwen3 8B AWQ Compatible Instruct");
        assert_eq!(super::display_name("Qwen3-TTS-12Hz-1.7B-Base-Q8_0"), "Qwen3 TTS 12Hz 1.7B Base");
        assert_eq!(super::display_name("Qwen3.8-27B-UD-Q4_K_XL"), "Qwen3.8 27B");
        assert_eq!(super::display_name("Gemma 4 E4B"), "Gemma 4 E4B");
        assert_eq!(super::display_name("Qwen3 0.6B Instruct"), "Qwen3 0.6B Instruct");
    }
}
