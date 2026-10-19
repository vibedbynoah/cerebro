//! The Tune screen on Cerebro Engine: measures the fastest safe settings for one model on this
//! machine and streams the run as the events the screen already understands (ui/src/service/
//! types.ts `TuneEventBody`, rebuilt by ui/src/lib/tune.ts).
//!
//! Phases, as on the screen: probe (free video memory, the file's integrity), screen (each
//! candidate: three ~2 s bursts of greedy writing, the first discarded as warm-up), measure (the
//! three numbers: prompt reading, writing with an empty context, writing at the profile's depth),
//! quality (next-token distributions against the baseline's, KL divergence), soak (sustained
//! writing with free video memory watched), decide. Every number is measured on this machine;
//! nothing is estimated. The engine's automatic placement is the baseline every candidate must
//! beat by more than the noise margin.
//!
//! A cancelled run keeps every finished step; resuming re-runs the interrupted one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cerebro_engine::chat::{generate, GenConfig};
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::switches;
use cerebro_engine::tokenizer::Tokenizer;
use serde_json::{json, Value};
use tauri::ipc::Channel;

/// A candidate must beat the automatic settings by more than this share to replace them (the
/// screen's NOISE_MARGIN): repeated runs of one configuration differ by about this much.
pub const NOISE_MARGIN: f64 = 0.03;
/// Screening drops a candidate this much slower than the fastest so far (the baseline always goes on).
pub const DISCARD_SLOWER: f64 = 0.12;
/// KL divergence from the baseline's next-token distributions a candidate may reach.
pub const QUALITY_GATE: f64 = 0.01;
/// A soak passes unless free video memory falls faster than this over its final third, MiB/min
/// (the reading's own noise is about this size).
pub const SOAK_SLOPE_LIMIT: f64 = -1.0;
/// Seconds of one soak (CEREBRO_TUNE_SOAK_S overrides it, for tests).
pub const SOAK_SECONDS: u64 = 120;
/// Prompt reading is measured on at most this many tokens, to bound the run's length.
pub const PREFILL_CAP: usize = 2048;
/// Most candidates in one run.
pub const MAX_CANDIDATES: usize = 8;
const SAMPLER: &str = "temperature 0 (greedy), set explicitly";

/// What the model will mostly be used for: sets the depth and what "fastest" means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Chat,
    Coding,
    LongDocs,
}

impl Profile {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "chat" => Some(Profile::Chat),
            "coding" => Some(Profile::Coding),
            "long-docs" => Some(Profile::LongDocs),
            _ => None,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Profile::Chat => "chat",
            Profile::Coding => "coding",
            Profile::LongDocs => "long-docs",
        }
    }
    fn depth(self) -> usize {
        match self {
            Profile::Chat => 4096,
            Profile::Coding => 16384,
            Profile::LongDocs => 32768,
        }
    }
}

/// The profile's working depth, capped at what the model holds (lib/tune.ts `tuneDepth`).
pub fn tune_depth(profile: Profile, context_max: usize) -> usize {
    profile.depth().min(context_max.max(512))
}

/// The three numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meas {
    pub prefill: f64,
    pub decode0: f64,
    pub decode_depth: f64,
}

/// The single number a profile ranks by (lib/tune.ts `objectiveScore`).
pub fn objective(profile: Profile, m: &Meas, depth: usize) -> f64 {
    match profile {
        Profile::LongDocs => 1.0 / (depth as f64 / m.prefill.max(1e-9) + 500.0 / m.decode_depth.max(1e-9)),
        _ => m.decode_depth,
    }
}

// ---- Candidates ----------------------------------------------------------------------------------

/// One configuration to try: plain-word settings, and what it changes in the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Cand {
    pub id: String,
    pub label: String,
    pub baseline: bool,
    pub settings: Vec<(String, String)>,
    /// Layers on the graphics card; None: the planner's automatic choice.
    pub gpu_layers: Option<usize>,
    /// Engine switches set while it runs (and kept when it wins).
    pub switches: Vec<(&'static str, i64)>,
}

/// What planning needs to know about the automatic placement and the machine.
#[derive(Debug, Clone)]
pub struct PlanInput {
    pub total_layers: usize,
    /// Layers the automatic placement put on a graphics card (0: processor only).
    pub auto_gpu_layers: usize,
    /// The automatic placement is on a CUDA card (layer counts can be forced there).
    pub cuda: bool,
    /// Room in video memory for one more layer at this depth.
    pub room_for_one_more: bool,
    pub moe: bool,
    /// Video memory the expert cache would take automatically, MB (MoE on a card only).
    pub expert_auto_mb: Option<i64>,
    /// SPEC_LOOKUP now.
    pub spec_now: i64,
    pub perf_threads: usize,
    pub hybrid_cpu: bool,
}

/// The candidates for one run: the automatic placement first, then the changes that can matter
/// on this machine and model (at most MAX_CANDIDATES).
pub fn plan(p: &PlanInput) -> Vec<Cand> {
    let l = p.total_layers;
    let where_ = |n: usize| if n == 0 { "processor only".to_owned() } else if n >= l { format!("all {l} on the graphics card") } else { format!("{n} of {l} on the graphics card") };
    let mut out = vec![Cand {
        id: "c0".into(),
        label: "Cerebro Engine automatic".into(),
        baseline: true,
        settings: vec![("Placement".into(), "automatic".into()), ("GPU layers".into(), where_(p.auto_gpu_layers))],
        gpu_layers: None,
        switches: Vec::new(),
    }];
    let mut push = |label: String, settings: Vec<(String, String)>, gpu_layers: Option<usize>, sw: Vec<(&'static str, i64)>| {
        let id = format!("c{}", out.len());
        out.push(Cand { id, label, baseline: false, settings, gpu_layers, switches: sw });
    };
    let split = p.auto_gpu_layers > 0 && p.auto_gpu_layers < l;
    if split && p.cuda {
        for k in [2usize, 4] {
            if p.auto_gpu_layers > k {
                let n = p.auto_gpu_layers - k;
                push(format!("{} fewer layers on the graphics card", if k == 2 { "Two" } else { "Four" }), vec![("GPU layers".into(), where_(n))], Some(n), Vec::new());
            }
        }
        if p.room_for_one_more {
            let n = p.auto_gpu_layers + 1;
            push("One more layer on the graphics card".into(), vec![("GPU layers".into(), where_(n))], Some(n), Vec::new());
        }
    }
    if p.moe {
        if let Some(mb) = p.expert_auto_mb.filter(|&mb| mb >= 1024) {
            for (frac, words) in [(2, "half"), (4, "a quarter")] {
                let v = mb / frac;
                push(format!("Expert cache at {words} of the automatic size"), vec![("Expert cache".into(), format!("{} MB of video memory", v))], None, vec![(switches::EXPERT_VRAM_MB, v)]);
            }
        }
    }
    if p.spec_now == 0 {
        push("Prompt-lookup speculation on".into(), vec![("Speculation".into(), "prompt lookup, up to 8 tokens".into())], None, vec![(switches::SPEC_LOOKUP, 8)]);
    } else {
        push("Speculation off".into(), vec![("Speculation".into(), "off".into())], None, vec![(switches::SPEC_LOOKUP, 0)]);
    }
    if p.hybrid_cpu && p.auto_gpu_layers < l && p.perf_threads > 0 {
        push(format!("Writing threads on the {} performance cores", p.perf_threads), vec![("Threads".into(), format!("{}, performance cores", p.perf_threads))], None, vec![(switches::CPU_DECODE_THREADS, p.perf_threads as i64)]);
    }
    out.truncate(MAX_CANDIDATES);
    out
}

/// Screening's rule: the baseline always advances; others advance unless more than
/// DISCARD_SLOWER slower than the fastest so far. Returns (advance, reason).
pub fn screen_outcome(baseline: bool, tps: f64, best_so_far: f64) -> (bool, String) {
    if baseline {
        return (true, "Reference: every candidate is compared with it.".into());
    }
    let pct = |a: f64, b: f64| ((a / b - 1.0).abs() * 100.0).round() as i64;
    if tps >= best_so_far * (1.0 - DISCARD_SLOWER) {
        let reason = if tps > best_so_far {
            "Fastest so far.".to_owned()
        } else if pct(tps, best_so_far) == 0 {
            "As fast as the fastest so far; measured in full.".to_owned()
        } else {
            format!("Within {}% of the fastest; measured in full.", pct(tps, best_so_far))
        };
        (true, reason)
    } else {
        (false, format!("{}% slower than the fastest so far.", pct(tps, best_so_far)))
    }
}

/// Probabilities from logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut p: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= s);
    p
}

/// KL(p || q) in nats, skipping zero-probability terms of p.
pub fn kl(p: &[f32], q: &[f32]) -> f64 {
    p.iter()
        .zip(q)
        .filter(|(a, _)| **a > 0.0)
        .map(|(&a, &b)| a as f64 * ((a as f64) / (b as f64).max(1e-30)).ln())
        .sum::<f64>()
        .max(0.0)
}

/// Least-squares slope of free video memory over the final third of a soak, MiB per minute
/// (lib/tune.ts `finalThirdSlope`): samples are (seconds, GiB). None until two samples are there.
pub fn final_third_slope(samples: &[(f64, f64)], soak_s: f64) -> Option<f64> {
    let tail: Vec<&(f64, f64)> = samples.iter().filter(|s| s.0 >= soak_s * 2.0 / 3.0).collect();
    if tail.len() < 2 {
        return None;
    }
    let n = tail.len() as f64;
    let mt = tail.iter().map(|s| s.0).sum::<f64>() / n;
    let my = tail.iter().map(|s| s.1).sum::<f64>() / n;
    let num: f64 = tail.iter().map(|s| (s.0 - mt) * (s.1 - my)).sum();
    let den: f64 = tail.iter().map(|s| (s.0 - mt).powi(2)).sum();
    (den > 0.0).then(|| num / den * 1024.0 * 60.0)
}

/// The winner's explanation, in plain words, from what was measured.
pub fn reasons(profile: Profile, depth: usize, winner: &Cand, base: &Meas, won: &Meas, auto_layers: usize, total_layers: usize, rejected: &[String], best_other: Option<(&Cand, &Meas)>) -> Vec<String> {
    let mut out = Vec::new();
    let score = |m: &Meas| objective(profile, m, depth);
    let what = match profile {
        Profile::LongDocs => format!("the time to read a {}-token document and write a 500-token answer", depth),
        _ => format!("writing speed with {} tokens in view", depth),
    };
    if winner.baseline {
        out.push(if auto_layers >= total_layers {
            format!("The automatic placement already put all {total_layers} layers on the graphics card.")
        } else if auto_layers == 0 {
            "The automatic placement runs this model on the processor.".to_owned()
        } else {
            format!("The automatic placement put {auto_layers} of {total_layers} layers on the graphics card.")
        });
        if let Some((c, m)) = best_other {
            let gain = score(m) / score(base) - 1.0;
            if gain <= NOISE_MARGIN {
                out.push(format!(
                    "The best other candidate, \"{}\", measured {}{}% on {what}: inside the {}% noise margin, so changing settings would add risk for no measurable gain.",
                    c.label,
                    if gain >= 0.0 { "+" } else { "" },
                    (gain * 100.0).round() as i64,
                    (NOISE_MARGIN * 100.0).round() as i64
                ));
            }
        }
        if !rejected.is_empty() {
            out.push(format!("Faster candidates were rejected: {}.", rejected.join("; ")));
        }
        out.push("Cerebro keeps the engine's automatic settings rather than change something that did not help.".into());
        return out;
    }
    let gain = score(won) / score(base) - 1.0;
    out.push(format!("\"{}\" measured {}% better on {what} than the automatic settings ({:.1} vs {:.1} tok/s writing at depth).", winner.label, (gain * 100.0).round() as i64, won.decode_depth, base.decode_depth));
    if let Some(n) = winner.gpu_layers {
        out.push(format!("It places {n} of {total_layers} layers on the graphics card instead of {auto_layers}."));
    }
    for (name, value) in &winner.settings {
        if name != "GPU layers" {
            out.push(format!("{name}: {value}."));
        }
    }
    if winner.switches.iter().any(|(k, v)| *k == switches::SPEC_LOOKUP && *v > 0) {
        out.push("Prompt-lookup speculation is exact: the model checks every guessed token, so the text is the same.".into());
    }
    if !rejected.is_empty() {
        out.push(format!("Faster on paper but rejected: {}.", rejected.join("; ")));
    }
    out
}

// ---- Prompts -------------------------------------------------------------------------------------

/// A long, varied document for the profile (deterministic): prose for chat and long documents,
/// code for coding (where repeated structure is what speculation can copy).
pub fn document(profile: Profile, approx_words: usize) -> String {
    const NOUNS: &[&str] = &["river", "ledger", "engine", "harbor", "garden", "signal", "archive", "bridge", "market", "lantern", "valley", "council", "orchard", "compass", "furnace", "meadow"];
    const VERBS: &[&str] = &["measures", "carries", "records", "shapes", "guards", "follows", "repairs", "divides", "gathers", "explains", "crosses", "balances"];
    const ADJ: &[&str] = &["quiet", "northern", "patient", "copper", "early", "narrow", "careful", "distant", "steady", "bright", "hollow", "modest"];
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15 ^ profile.id().len() as u64;
    let mut next = |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    let mut s = String::new();
    let mut words = 0;
    let mut i = 0;
    while words < approx_words {
        i += 1;
        if profile == Profile::Coding {
            let (a, b, c) = (NOUNS[next(NOUNS.len())], NOUNS[next(NOUNS.len())], VERBS[next(VERBS.len())]);
            s.push_str(&format!(
                "/// {c} the {a} for record {i}.\nfn {c}_{a}_{i}(input: &[{b}], limit: usize) -> Option<usize> {{\n    let total = input.iter().take(limit).map(|x| x.weight * {}).sum::<usize>();\n    if total > {} {{ Some(total - {}) }} else {{ None }}\n}}\n\n",
                next(9) + 1,
                next(900) + 100,
                next(90) + 10
            ));
            words += 40;
        } else {
            let sentence = format!(
                "In section {i}, the {} {} {} the {} {} while the {} {} waits by the {}. ",
                ADJ[next(ADJ.len())],
                NOUNS[next(NOUNS.len())],
                VERBS[next(VERBS.len())],
                ADJ[next(ADJ.len())],
                NOUNS[next(NOUNS.len())],
                ADJ[next(ADJ.len())],
                NOUNS[next(NOUNS.len())],
                NOUNS[next(NOUNS.len())]
            );
            words += sentence.split_whitespace().count();
            s.push_str(&sentence);
            if i % 6 == 0 {
                s.push_str("\n\n");
            }
        }
    }
    s
}

/// The fixed prompts of the quality gate.
const QUALITY_PROMPTS: [&str; 8] = [
    "The capital of France is",
    "Write a haiku about the sea.",
    "def fibonacci(n):",
    "Explain why the sky is blue in one sentence.",
    "List three prime numbers greater than 100:",
    "Translate to German: The weather is nice today.",
    "SELECT name FROM users WHERE",
    "Once upon a time, in a small village,",
];

fn chat_ids(tok: &Tokenizer, user: &str) -> Vec<u32> {
    tok.encode(&cerebro_engine::chatml(None, &[("user", user)], false))
}

// ---- The event hub -------------------------------------------------------------------------------

/// What a run was started with.
#[derive(Clone)]
pub struct Spec {
    pub model: crate::engine::LibraryModel,
    pub profile: Profile,
    pub integrity: bool,
}

/// The finished steps of a run, so a resume can skip them.
#[derive(Default)]
pub struct Memory {
    pub probed: bool,
    pub plan: Vec<Cand>,
    pub auto_layers: usize,
    pub total_layers: usize,
    pub screened: HashMap<String, Option<f64>>,
    pub advanced: Vec<String>,
    pub measured: HashMap<String, Meas>,
    pub dists: HashMap<String, Vec<Vec<f32>>>,
    pub quality_done: bool,
    pub passed_quality: Vec<String>,
    pub rejected: Vec<String>,
    pub soaked: HashMap<String, bool>,
    pub winner: Option<String>,
    /// The file's SHA-256 from the probe (when the integrity check is on).
    pub sha256: Option<String>,
}

struct HubState {
    seq: u64,
    log: Vec<Value>,
    channels: Vec<Channel<Value>>,
    run: Option<Value>,
    status: &'static str,
    spec: Option<Spec>,
    memory: Memory,
    /// Run-clock seconds before the current running stretch; the stretch's start.
    clock: f64,
    since: Option<Instant>,
    runs: u64,
}

/// One tune at a time: its event log (for replay), live listeners, and its finished steps.
pub struct TuneHub {
    st: Mutex<HubState>,
    pub cancel: Arc<AtomicBool>,
    results: Mutex<Option<PathBuf>>,
}

impl Default for TuneHub {
    fn default() -> Self {
        TuneHub {
            st: Mutex::new(HubState { seq: 0, log: Vec::new(), channels: Vec::new(), run: None, status: "idle", spec: None, memory: Memory::default(), clock: 0.0, since: None, runs: 0 }),
            cancel: Arc::new(AtomicBool::new(false)),
            results: Mutex::new(None),
        }
    }
}

impl TuneHub {
    fn lock(&self) -> std::sync::MutexGuard<'_, HubState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Where finished tunes are kept (`tune-results.json` in the app's config folder).
    pub fn set_results_path(&self, p: PathBuf) {
        *self.results.lock().unwrap_or_else(|e| e.into_inner()) = Some(p.clone());
        let _ = RESULTS_PATH.set(p);
    }

    pub fn running(&self) -> bool {
        self.lock().status == "running"
    }

    /// Adds `body` to the log (with its sequence number and run clock) and sends it live.
    pub fn emit(&self, body: Value) {
        let mut s = self.lock();
        s.seq += 1;
        let at = s.clock + s.since.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let mut ev = body;
        if let Some(o) = ev.as_object_mut() {
            o.insert("seq".into(), json!(s.seq));
            o.insert("at".into(), json!((at * 10.0).round() / 10.0));
        }
        match ev["kind"].as_str() {
            Some("winner") => s.status = "done",
            Some("failed") => s.status = "failed",
            _ => {}
        }
        s.log.push(ev.clone());
        let batch = json!({ "events": [ev], "replay": false });
        s.channels.retain(|c| c.send(batch.clone()).is_ok());
    }

    /// A new listener: the whole log now (replay), then every new event.
    pub fn subscribe(&self, ch: Channel<Value>) -> Result<(), String> {
        let mut s = self.lock();
        ch.send(json!({ "events": s.log, "replay": true })).map_err(|_| "The Tune screen could not be updated.".to_owned())?;
        s.channels.push(ch);
        Ok(())
    }

    /// Starts a new run (refused while one runs); returns its run info.
    pub fn begin(&self, spec: Spec, run: Value) -> Result<Value, String> {
        let mut s = self.lock();
        if s.status == "running" {
            return Err("A tune is already running. Cancel it first.".to_owned());
        }
        s.runs += 1;
        let mut run = run;
        run["runId"] = json!(format!("tune-{}", s.runs));
        s.run = Some(run.clone());
        s.spec = Some(spec);
        s.memory = Memory::default();
        s.log.clear();
        s.status = "running";
        s.clock = 0.0;
        s.since = Some(Instant::now());
        self.cancel.store(false, Ordering::SeqCst);
        drop(s);
        self.emit(json!({ "kind": "started", "run": run }));
        Ok(run)
    }

    fn check_run(&self, run_id: &str) -> Result<(), String> {
        let s = self.lock();
        match s.run.as_ref().and_then(|r| r["runId"].as_str()) {
            Some(id) if id == run_id => Ok(()),
            _ => Err("That tune is not the current one.".to_owned()),
        }
    }

    pub fn request_cancel(&self, run_id: &str) -> Result<(), String> {
        self.check_run(run_id)?;
        if !self.running() {
            return Err("This tune is not running.".to_owned());
        }
        self.cancel.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Marks a cancelled run as running again; the caller queues the job.
    pub fn resume(&self, run_id: &str) -> Result<(), String> {
        self.check_run(run_id)?;
        {
            let mut s = self.lock();
            if s.status != "cancelled" {
                return Err("Only a cancelled tune can be resumed.".to_owned());
            }
            s.status = "running";
            s.since = Some(Instant::now());
        }
        self.cancel.store(false, Ordering::SeqCst);
        self.emit(json!({ "kind": "resumed" }));
        Ok(())
    }

    fn stopped(&self) {
        self.emit(json!({ "kind": "cancelled" }));
        let mut s = self.lock();
        s.status = "cancelled";
        if let Some(t) = s.since.take() {
            s.clock += t.elapsed().as_secs_f64();
        }
    }

    pub fn spec(&self) -> Option<Spec> {
        self.lock().spec.clone()
    }

    /// The current run's events (tests read the whole run back).
    #[cfg(test)]
    pub fn events(&self) -> Vec<Value> {
        self.lock().log.clone()
    }
}

// ---- Kept results --------------------------------------------------------------------------------

static RESULTS_PATH: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// A model's kept tune: what to load it with, and what was measured.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Kept {
    pub gpu_layers: Option<usize>,
    pub switches: Vec<(String, i64)>,
    pub tuned_at: String,
    pub result: Value,
    /// SHA-256 of the file when it was tuned (the next tune's integrity check compares with it).
    pub sha256: Option<String>,
}

fn read_kept() -> HashMap<String, Kept> {
    RESULTS_PATH.get().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

/// The kept tune of model `id`, if any.
pub fn kept(id: &str) -> Option<Kept> {
    read_kept().remove(id)
}

fn keep(id: &str, k: Kept) -> Result<(), String> {
    let Some(p) = RESULTS_PATH.get() else { return Ok(()) };
    let mut all = read_kept();
    all.insert(id.to_owned(), k);
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(p, serde_json::to_string_pretty(&all).unwrap_or_default()).map_err(|e| format!("The tune finished but could not be kept: {e}."))
}

/// Sets the kept switches of model `id` (when it is loaded with automatic settings).
pub fn apply_kept_switches(id: &str) {
    if let Some(k) = kept(id) {
        for (sw, v) in &k.switches {
            let _ = switches::set(sw, *v);
        }
    }
}

// ---- The run -------------------------------------------------------------------------------------

/// Why a run stopped early.
pub enum Stop {
    Cancelled,
    Failed(String),
}

/// Builds a session for `model` at `ctx` with `gpu_layers` (None: automatic), like loading does,
/// returning it with (layers on a graphics card, total layers, on CUDA).
pub type Place<'a> = dyn Fn(&Model, usize, Option<usize>) -> Result<(Session, usize, usize, bool), String> + 'a;

/// What the engine thread needs back: the placement to load the model with afterwards.
pub struct Done {
    pub gpu_layers: Option<usize>,
}

fn soak_seconds() -> u64 {
    std::env::var("CEREBRO_TUNE_SOAK_S").ok().and_then(|v| v.parse().ok()).unwrap_or(SOAK_SECONDS)
}

/// Switches set for one candidate and put back after.
struct SwitchGuard(Vec<(&'static str, i64)>);

impl SwitchGuard {
    fn set(sw: &[(&'static str, i64)]) -> Self {
        let old = sw.iter().map(|(k, _)| (*k, switches::get(k))).collect();
        for (k, v) in sw {
            let _ = switches::set(k, *v);
        }
        SwitchGuard(old)
    }
}

impl Drop for SwitchGuard {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            let _ = switches::set(k, *v);
        }
    }
}

fn vram_free_gib() -> f64 {
    cerebro_engine::hw::gpus().first().map_or(0.0, |g| g.vram_free as f64 / (1u64 << 30) as f64)
}

/// Fewest tokens a measurement writes, so slow models are not judged on a handful of tokens.
const MIN_TOKENS: usize = 64;

/// Greedy writing for `secs` of writing time (and at least MIN_TOKENS tokens when measuring,
/// within four times `secs`), calling `tick(tokens so far, writing seconds so
/// far)` about every half second; starts again from the prompt when the model ends its answer
/// early. Reading the prompt is not counted: each call's clock starts at its first written token,
/// and the tokens are those after it. Returns (tokens, writing seconds).
fn write_for(m: &Model, s: &mut Session, tok: &Tokenizer, prompt: &[u32], secs: f64, min_tokens: usize, cancel: &AtomicBool, mut tick: impl FnMut(usize, f64)) -> Result<(usize, f64), Stop> {
    let enough = |n: usize, e: f64| e >= secs * 4.0 || (e >= secs && n >= min_tokens);
    let cfg = GenConfig { sampler: SamplerConfig { temperature: 0.0, ..Default::default() }, max_tokens: 100_000 };
    let mut n = 0usize;
    let mut done_s = 0.0;
    let mut last_tick = 0.0;
    while !enough(n, done_s) {
        let mut first: Option<Instant> = None;
        let mut call_s = 0.0;
        let g = generate(m, s, tok, prompt, &cfg, |_| {
            let Some(f) = first else {
                first = Some(Instant::now());
                return !cancel.load(Ordering::SeqCst);
            };
            n += 1;
            call_s = f.elapsed().as_secs_f64();
            let e = done_s + call_s;
            if e - last_tick >= 0.5 {
                tick(n, e);
                last_tick = e;
            }
            !enough(n, e) && !cancel.load(Ordering::SeqCst)
        })
        .map_err(|e| Stop::Failed(format!("Writing failed: {e}")))?;
        done_s += call_s;
        if cancel.load(Ordering::SeqCst) {
            return Err(Stop::Cancelled);
        }
        if g.output_tokens <= 1 {
            break;
        }
    }
    Ok((n, done_s))
}

/// Runs (or resumes) the hub's current run on `model`, emitting events; returns the placement
/// to load the model with afterwards (the winner's), or why it stopped.
pub fn run(hub: &TuneHub, model: &Model, tok: &Tokenizer, place: &Place<'_>) -> Result<Done, Stop> {
    let spec = hub.spec().ok_or_else(|| Stop::Failed("No tune to run.".into()))?;
    let depth = tune_depth(spec.profile, spec.model.context_max as usize);
    let ctx = depth + 1024;
    let cancel = hub.cancel.clone();
    let cancelled = || cancel.load(Ordering::SeqCst);
    let mem = |f: &mut dyn FnMut(&mut Memory)| f(&mut hub.lock().memory);

    // ---- Probe ----
    if !hub.lock().memory.probed {
        hub.emit(json!({ "kind": "phase", "phase": "probe" }));
        let free = vram_free_gib();
        let integrity = if spec.integrity {
            let path = Path::new(&spec.model.path);
            let sha = crate::dedup::sha256_of(path).map_err(|e| Stop::Failed(format!("The model file could not be read in full: {e}.")))?;
            if let Some(old) = kept(&spec.model.id).and_then(|k| k.sha256) {
                if old != sha {
                    return Err(Stop::Failed("The model file changed since it was last tuned (its SHA-256 differs). Tune again only if you replaced it on purpose.".into()));
                }
            }
            mem(&mut |m| m.sha256 = Some(sha.clone()));
            "verified"
        } else {
            "off"
        };
        if cancelled() {
            return Err(Stop::Cancelled);
        }
        hub.emit(json!({ "kind": "probe", "vramFreeGiB": (free * 100.0).round() / 100.0, "integrity": integrity }));
        // The automatic placement decides the candidates.
        let (s, auto, total, cuda) = place(model, ctx, None).map_err(Stop::Failed)?;
        let plan_vram = s.gpu.as_ref().map_or(0, |g| g.plan.vram_bytes);
        drop(s);
        let cores = cerebro_engine::hw::cores();
        let free_after = vram_free_gib();
        let per_layer = if auto > 0 { plan_vram as f64 / auto as f64 / (1u64 << 30) as f64 } else { 0.0 };
        let input = PlanInput {
            total_layers: total,
            auto_gpu_layers: auto,
            cuda,
            room_for_one_more: per_layer > 0.0 && free_after - per_layer > 0.6,
            moe: model.cfg.n_expert > 0,
            expert_auto_mb: (model.cfg.n_expert > 0 && auto > 0).then(|| ((free * 1024.0) as i64 - 600).max(0)),
            spec_now: switches::get(switches::SPEC_LOOKUP),
            perf_threads: cores.performance_logical,
            hybrid_cpu: cores.hybrid(),
        };
        let cands = plan(&input);
        mem(&mut |m| {
            m.plan = cands.clone();
            m.auto_layers = auto;
            m.total_layers = total;
            m.probed = true;
        });
        {
            let mut s = hub.lock();
            if let Some(r) = s.run.as_mut() {
                r["plannedCandidates"] = json!(cands.len());
            }
        }
        hub.emit(json!({ "kind": "phase", "phase": "screen" }));
        for c in &cands {
            hub.emit(json!({ "kind": "candidate", "candidate": cand_json(c) }));
        }
    }
    let cands = hub.lock().memory.plan.clone();
    let (auto, total) = {
        let m = &hub.lock().memory;
        (m.auto_layers, m.total_layers)
    };

    // ---- Screen ----
    let screen_prompt = chat_ids(tok, &match spec.profile {
        Profile::Coding => format!("Here is some code:\n\n{}\nWrite three more functions in the same style.", document(Profile::Coding, 300)),
        _ => "Write a long, detailed essay about the history of bridges, with many examples.".to_owned(),
    });
    for c in &cands {
        if hub.lock().memory.screened.contains_key(&c.id) {
            continue;
        }
        hub.emit(json!({ "kind": "screen-start", "candidateId": c.id }));
        let guard = SwitchGuard::set(&c.switches);
        let built = place(model, ctx, c.gpu_layers);
        let mut s = match built {
            Ok((s, ..)) => s,
            Err(e) => {
                drop(guard);
                hub.emit(json!({ "kind": "screened", "candidateId": c.id, "tps": null, "outcome": "fail", "reason": e }));
                mem(&mut |m| {
                    m.screened.insert(c.id.clone(), None);
                });
                continue;
            }
        };
        // Warm-up, then three bursts; the first is discarded (the screen says so).
        write_for(model, &mut s, tok, &screen_prompt, 0.5, 0, &cancel, |_, _| {})?;
        let mut kept_tps = Vec::new();
        let c0 = Instant::now();
        for rep in 1..=3 {
            let mut prev = (0usize, 0.0f64);
            let id = c.id.clone();
            let (n, secs) = write_for(model, &mut s, tok, &screen_prompt, 2.0, 0, &cancel, |n, e| {
                let tps = (n - prev.0) as f64 / (e - prev.1).max(1e-3);
                prev = (n, e);
                hub.emit(json!({ "kind": "sample", "candidateId": id, "rep": rep, "t": ((c0.elapsed().as_secs_f64()) * 100.0).round() / 100.0, "tps": (tps * 100.0).round() / 100.0 }));
            })?;
            if rep > 1 && secs > 0.0 {
                kept_tps.push(n as f64 / secs);
            }
        }
        drop(s);
        drop(guard);
        let tps = kept_tps.iter().sum::<f64>() / kept_tps.len().max(1) as f64;
        let best = {
            let m = &hub.lock().memory;
            m.screened.values().flatten().fold(0.0f64, |a, &b| a.max(b))
        };
        let (advance, reason) = screen_outcome(c.baseline, tps, best);
        hub.emit(json!({ "kind": "screened", "candidateId": c.id, "tps": (tps * 10.0).round() / 10.0, "outcome": if advance { "advance" } else { "discard" }, "reason": reason }));
        mem(&mut |m| {
            m.screened.insert(c.id.clone(), Some(tps));
            if advance {
                m.advanced.push(c.id.clone());
            }
        });
    }

    // ---- Measure (and the quality gate's distributions, while each session is up) ----
    if hub.lock().memory.measured.is_empty() {
        hub.emit(json!({ "kind": "phase", "phase": "measure" }));
    }
    let advanced = hub.lock().memory.advanced.clone();
    let doc = document(spec.profile, depth);
    let doc_ids_full = tok.encode(&doc);
    for id in &advanced {
        if hub.lock().memory.measured.contains_key(id) {
            continue;
        }
        let c = cands.iter().find(|c| &c.id == id).expect("planned candidate");
        let _guard = SwitchGuard::set(&c.switches);
        let (mut s, ..) = place(model, ctx, c.gpu_layers).map_err(Stop::Failed)?;
        // Prompt reading: a fresh document prefix, capped.
        let n_pre = PREFILL_CAP.min(depth).min(doc_ids_full.len());
        s.feed(model, &doc_ids_full[..n_pre]).map_err(|e| Stop::Failed(format!("Reading failed: {e}")))?;
        let prefill = s.stats.prefill_tps();
        if cancelled() {
            return Err(Stop::Cancelled);
        }
        // Writing with an empty context.
        let short = chat_ids(tok, "Tell me about the ocean.");
        let (n0, s0) = write_for(model, &mut s, tok, &short, 3.0, MIN_TOKENS, &cancel, |_, _| {})?;
        // Writing at depth: the document fills the context up to `depth`, then a question.
        let tail = chat_ids(tok, "Summarise the document above in detail.");
        let fill = depth.saturating_sub(tail.len() + 64).min(doc_ids_full.len());
        let mut deep: Vec<u32> = doc_ids_full[..fill].to_vec();
        deep.extend_from_slice(&tail);
        let (nd, sd) = write_for(model, &mut s, tok, &deep, 3.0, MIN_TOKENS, &cancel, |_, _| {})?;
        // The quality gate's next-token distributions.
        let mut dists = Vec::new();
        for p in QUALITY_PROMPTS {
            s.feed(model, &chat_ids(tok, p)).map_err(|e| Stop::Failed(format!("Reading failed: {e}")))?;
            dists.push(softmax(&s.logits));
        }
        drop(s);
        let m3 = Meas { prefill, decode0: n0 as f64 / s0.max(1e-9), decode_depth: nd as f64 / sd.max(1e-9) };
        hub.emit(json!({ "kind": "measured", "candidateId": id, "prefillTps": m3.prefill.round(), "decodeTps0": (m3.decode0 * 10.0).round() / 10.0, "decodeTpsDepth": (m3.decode_depth * 10.0).round() / 10.0 }));
        mem(&mut |m| {
            m.measured.insert(id.clone(), m3);
            m.dists.insert(id.clone(), dists.clone());
        });
    }

    let (measured, base_id) = {
        let m = &hub.lock().memory;
        (m.measured.clone(), cands[0].id.clone())
    };
    let base = *measured.get(&base_id).ok_or_else(|| Stop::Failed("The automatic settings could not be measured.".into()))?;
    let score = |m: &Meas| objective(spec.profile, m, depth);
    let mut ranked: Vec<&Cand> = cands.iter().filter(|c| measured.contains_key(&c.id)).collect();
    ranked.sort_by(|a, b| score(&measured[&b.id]).partial_cmp(&score(&measured[&a.id])).unwrap_or(std::cmp::Ordering::Equal));
    let contenders: Vec<&Cand> = ranked.iter().copied().filter(|c| !c.baseline && score(&measured[&c.id]) > score(&base) * (1.0 + NOISE_MARGIN)).collect();

    // ---- Quality ----
    if !hub.lock().memory.quality_done {
        hub.emit(json!({ "kind": "phase", "phase": "quality" }));
        let dists = hub.lock().memory.dists.clone();
        let mut passed = Vec::new();
        let mut rejected = Vec::new();
        for c in contenders.iter().take(4) {
            let (Some(b), Some(d)) = (dists.get(&base_id), dists.get(&c.id)) else { continue };
            let v = b.iter().zip(d).map(|(p, q)| kl(p, q)).sum::<f64>() / b.len().max(1) as f64;
            let ok = v <= QUALITY_GATE;
            hub.emit(json!({
                "kind": "quality", "candidateId": c.id, "metric": "KL divergence", "value": (v * 1e5).round() / 1e5,
                "gate": QUALITY_GATE, "passed": ok,
                "reference": "next-token probabilities from the engine's automatic settings, on 8 fixed prompts",
            }));
            if ok {
                passed.push(c.id.clone());
            } else {
                rejected.push(format!("\"{}\" failed the quality gate", c.label));
            }
        }
        mem(&mut |m| {
            m.passed_quality = passed.clone();
            m.rejected.extend(rejected.clone());
            m.quality_done = true;
        });
    }

    // ---- Soak ----
    let soak_s = soak_seconds();
    let soak = |c: &Cand| -> Result<bool, Stop> {
        hub.emit(json!({ "kind": "soak-start", "candidateId": c.id }));
        let _guard = SwitchGuard::set(&c.switches);
        let (mut s, ..) = place(model, ctx, c.gpu_layers).map_err(Stop::Failed)?;
        let prompt = chat_ids(tok, "Write a long, detailed essay about the history of bridges, with many examples.");
        let mut samples: Vec<(f64, f64)> = Vec::new();
        let t0 = Instant::now();
        let mut next_at = 4.0;
        let mut prev = (0usize, 0.0f64);
        let id = c.id.clone();
        write_for(model, &mut s, tok, &prompt, soak_s as f64, 0, &cancel, |n, e| {
            if e >= next_at {
                let tps = (n - prev.0) as f64 / (e - prev.1).max(1e-3);
                prev = (n, e);
                let free = vram_free_gib();
                let t = t0.elapsed().as_secs_f64().round();
                samples.push((t, free));
                hub.emit(json!({ "kind": "soak-sample", "candidateId": id, "t": t, "vramFreeGiB": (free * 1000.0).round() / 1000.0, "tps": (tps * 10.0).round() / 10.0 }));
                next_at += 4.0;
            }
        })?;
        drop(s);
        let slope = final_third_slope(&samples, soak_s as f64).unwrap_or(0.0);
        let passed = slope >= SOAK_SLOPE_LIMIT;
        let on_gpu = cerebro_engine::hw::gpus().first().is_some();
        let reason = if !on_gpu {
            "No graphics card: writing speed held through the soak.".to_owned()
        } else if passed {
            "Free video memory held steady over the final third.".to_owned()
        } else {
            format!("Free video memory kept falling over the final third ({:.1} MiB a minute).", slope)
        };
        hub.emit(json!({ "kind": "soak-end", "candidateId": c.id, "passed": passed, "slopeMiBPerMin": (slope * 10.0).round() / 10.0, "reason": reason }));
        Ok(passed)
    };
    if hub.lock().memory.soaked.is_empty() {
        hub.emit(json!({ "kind": "phase", "phase": "soak" }));
    }
    let passed_quality = hub.lock().memory.passed_quality.clone();
    let mut winner: Option<&Cand> = None;
    for id in passed_quality.iter().take(3) {
        let c = cands.iter().find(|c| &c.id == id).expect("planned");
        let done = hub.lock().memory.soaked.get(id).copied();
        let ok = match done {
            Some(v) => v,
            None => {
                let v = soak(c)?;
                mem(&mut |m| {
                    m.soaked.insert(id.clone(), v);
                    if !v {
                        m.rejected.push(format!("\"{}\" failed the soak", c.label));
                    }
                });
                v
            }
        };
        if ok {
            winner = Some(c);
            break;
        }
    }
    let winner = match winner {
        Some(w) => w,
        None => {
            let b = &cands[0];
            if !hub.lock().memory.soaked.contains_key(&b.id) {
                let v = soak(b)?;
                mem(&mut |m| {
                    m.soaked.insert(b.id.clone(), v);
                });
            }
            b
        }
    };

    // ---- Decide ----
    hub.emit(json!({ "kind": "phase", "phase": "decide" }));
    let rejected = hub.lock().memory.rejected.clone();
    let won = measured[&winner.id];
    let best_other = ranked.iter().find(|c| !c.baseline).map(|c| (*c, &measured[&c.id]));
    let why = reasons(spec.profile, depth, winner, &base, &won, auto, total, &rejected, best_other);
    // Kept before the winner event, so the library shows it as soon as the screen does.
    let gpu = cerebro_engine::hw::gpus().first().map(|g| g.name.clone()).filter(|_| auto > 0 || winner.gpu_layers.is_some_and(|n| n > 0)).unwrap_or_else(|| "CPU only".into());
    let today = crate::engine::today_iso();
    let result = json!({
        "prefillTps": won.prefill.round(), "decodeTps0": (won.decode0 * 10.0).round() / 10.0, "decodeTpsDepth": (won.decode_depth * 10.0).round() / 10.0,
        "depth": depth, "build": build(), "gpu": gpu, "sampler": SAMPLER, "measuredAt": today, "provenance": "measured",
    });
    let sha = hub.lock().memory.sha256.clone();
    let kept = Kept { gpu_layers: winner.gpu_layers, switches: winner.switches.iter().map(|(k, v)| (k.to_string(), *v)).collect(), tuned_at: today, result, sha256: sha };
    if let Err(e) = keep(&spec.model.id, kept) {
        eprintln!("{e}");
    }
    for (k, v) in &winner.switches {
        let _ = switches::set(k, *v);
    }
    hub.emit(json!({ "kind": "winner", "candidateId": winner.id, "baselineId": base_id, "reasons": why }));
    mem(&mut |m| m.winner = Some(winner.id.clone()));
    Ok(Done { gpu_layers: winner.gpu_layers })
}

/// Runs the job and reports how it ended on the hub.
pub fn run_job(hub: &TuneHub, model: &Model, tok: &Tokenizer, place: &Place<'_>) -> Option<Done> {
    match run(hub, model, tok, place) {
        Ok(d) => Some(d),
        Err(Stop::Cancelled) => {
            hub.stopped();
            None
        }
        Err(Stop::Failed(reason)) => {
            hub.emit(json!({ "kind": "failed", "reason": reason }));
            None
        }
    }
}

fn cand_json(c: &Cand) -> Value {
    json!({
        "id": c.id, "label": c.label, "baseline": c.baseline,
        "settings": c.settings.iter().map(|(n, v)| json!({ "name": n, "value": v, "flag": null })).collect::<Vec<_>>(),
    })
}

/// "Cerebro Engine <version>".
pub fn build() -> String {
    format!("Cerebro Engine {}", cerebro_engine::VERSION)
}

/// The run info for `spec` on this machine (the hub fills the run id).
pub fn run_info(spec: &Spec) -> Value {
    let hw = crate::engine::hardware();
    let m = &spec.model;
    json!({
        "runId": "",
        "modelId": m.id, "modelName": m.name, "quant": m.quant, "architecture": m.architecture, "sizeGiB": m.size_gi_b,
        "profile": spec.profile.id(), "depth": tune_depth(spec.profile, m.context_max as usize), "sampler": SAMPLER,
        "build": build(), "backendCommit": cerebro_engine::COMMIT,
        "gpu": hw.gpu.as_ref().map_or("none".to_owned(), |g| g.name.clone()),
        "vramTotalGiB": hw.gpu.as_ref().map_or(0.0, |g| (g.vram_total_gi_b * 10.0).round() / 10.0),
        "cpu": format!("{} ({} cores, {} threads)", hw.cpu.name, hw.cpu.cores, hw.cpu.threads),
        "ramTotalGiB": (hw.ram_total_gi_b * 10.0).round() / 10.0,
        "soakSeconds": soak_seconds(),
        "plannedCandidates": 0, "plannedSoaks": 1,
        "integrityCheck": spec.integrity, "provenance": "measured",
    })
}

/// A whole tune on a real model through the engine thread: the event order the screen expects and
/// every measurement positive. Short soak via CEREBRO_TUNE_SOAK_S (default here 12 s).
/// `cargo test --release -p cerebro-desktop whole_tune -- --ignored --nocapture`
#[cfg(test)]
mod real_tests {
    use super::*;

    #[test]
    #[ignore]
    fn a_whole_tune_of_a_real_model_runs_in_order() {
        if std::env::var_os("CEREBRO_TUNE_SOAK_S").is_none() {
            std::env::set_var("CEREBRO_TUNE_SOAK_S", "12");
        }
        let path = std::env::var("CEREBRO_TEST_MODEL").unwrap_or_else(|_| r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf".into());
        let model = crate::engine::scan_library_one(Path::new(&path)).expect("model");
        let engine = crate::engine::Engine::start();
        let hub = Arc::new(TuneHub::default());
        let spec = Spec { model, profile: Profile::Chat, integrity: true };
        hub.begin(spec.clone(), run_info(&spec)).expect("begin");
        engine.tune_model(hub.clone()).expect("queue");
        let t0 = Instant::now();
        while hub.running() && t0.elapsed().as_secs() < 3600 {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let ev = hub.events();
        let kinds: Vec<&str> = ev.iter().filter_map(|e| e["kind"].as_str()).filter(|k| *k != "sample" && *k != "soak-sample").collect();
        println!("{kinds:?}");
        for e in &ev {
            if matches!(e["kind"].as_str(), Some("measured" | "screened" | "quality" | "soak-end" | "winner" | "probe" | "failed")) {
                println!("{e}");
            }
        }
        assert_eq!(kinds.first(), Some(&"started"));
        assert_eq!(kinds.last(), Some(&"winner"), "the run did not finish: {kinds:?}");
        let order = ["probe", "screen", "measure", "quality", "soak", "decide"];
        let phases: Vec<&str> = ev.iter().filter(|e| e["kind"] == "phase").filter_map(|e| e["phase"].as_str()).collect();
        assert_eq!(phases, order);
        for e in ev.iter().filter(|e| e["kind"] == "measured") {
            for k in ["prefillTps", "decodeTps0", "decodeTpsDepth"] {
                assert!(e[k].as_f64().is_some_and(|v| v > 0.0), "{k} in {e}");
            }
        }
        let seqs: Vec<u64> = ev.iter().filter_map(|e| e["seq"].as_u64()).collect();
        assert!(seqs.windows(2).all(|w| w[1] == w[0] + 1));
        assert!(ev.iter().any(|e| e["kind"] == "sample"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> PlanInput {
        PlanInput { total_layers: 40, auto_gpu_layers: 28, cuda: true, room_for_one_more: true, moe: false, expert_auto_mb: None, spec_now: 0, perf_threads: 12, hybrid_cpu: true }
    }

    #[test]
    fn a_split_model_gets_placement_neighbours_threads_and_speculation() {
        let c = plan(&input());
        let labels: Vec<&str> = c.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, ["Cerebro Engine automatic", "Two fewer layers on the graphics card", "Four fewer layers on the graphics card", "One more layer on the graphics card", "Prompt-lookup speculation on", "Writing threads on the 12 performance cores"]);
        assert!(c[0].baseline && c[0].gpu_layers.is_none());
        assert_eq!((c[1].gpu_layers, c[2].gpu_layers, c[3].gpu_layers), (Some(26), Some(24), Some(29)));
        assert_eq!(c[0].settings[1].1, "28 of 40 on the graphics card");
    }

    #[test]
    fn a_model_wholly_on_the_card_only_tries_speculation() {
        let c = plan(&PlanInput { auto_gpu_layers: 40, ..input() });
        assert_eq!(c.len(), 2);
        assert_eq!(c[1].switches, vec![(switches::SPEC_LOOKUP, 8)]);
        let off = plan(&PlanInput { auto_gpu_layers: 40, spec_now: 6, ..input() });
        assert_eq!(off[1].label, "Speculation off");
    }

    #[test]
    fn moe_models_try_smaller_expert_caches() {
        let c = plan(&PlanInput { moe: true, expert_auto_mb: Some(6000), auto_gpu_layers: 40, ..input() });
        let caps: Vec<i64> = c.iter().flat_map(|c| c.switches.iter().filter(|(k, _)| *k == switches::EXPERT_VRAM_MB).map(|(_, v)| *v)).collect();
        assert_eq!(caps, vec![3000, 1500]);
    }

    #[test]
    fn never_more_than_the_cap() {
        let c = plan(&PlanInput { moe: true, expert_auto_mb: Some(8000), ..input() });
        assert!(c.len() <= MAX_CANDIDATES);
    }

    #[test]
    fn screening_keeps_the_baseline_and_drops_slow_ones() {
        assert!(screen_outcome(true, 1.0, 100.0).0);
        assert_eq!(screen_outcome(false, 120.0, 100.0), (true, "Fastest so far.".into()));
        assert!(screen_outcome(false, 89.0, 100.0).0);
        let (adv, why) = screen_outcome(false, 80.0, 100.0);
        assert!(!adv && why == "20% slower than the fastest so far.");
    }

    #[test]
    fn kl_is_zero_for_equal_and_positive_otherwise() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!(kl(&p, &p).abs() < 1e-9);
        let q = softmax(&[1.0, 2.0, 2.0]);
        let v = kl(&p, &q);
        assert!(v > 0.0 && v < 1.0);
        // Hand-computed: p = [0.5, 0.5], q = [0.25, 0.75] -> 0.5 ln 2 + 0.5 ln(2/3).
        let want = 0.5 * 2f64.ln() + 0.5 * (2.0f64 / 3.0).ln();
        assert!((kl(&[0.5, 0.5], &[0.25, 0.75]) - want).abs() < 1e-6);
    }

    #[test]
    fn slope_uses_only_the_final_third_in_mib_per_minute() {
        // Flat for two thirds, then falling 1 GiB per 60 s = -1024 MiB/min.
        let s: Vec<(f64, f64)> = (0..=30).map(|i| (i as f64 * 4.0, if i * 4 < 80 { 5.0 } else { 5.0 - (i as f64 * 4.0 - 80.0) / 60.0 })).collect();
        let v = final_third_slope(&s, 120.0).expect("slope");
        assert!((v + 1024.0).abs() < 1.0, "{v}");
        assert_eq!(final_third_slope(&[(1.0, 1.0)], 120.0), None);
        let flat: Vec<(f64, f64)> = (0..30).map(|i| (i as f64 * 4.0, 3.0)).collect();
        assert_eq!(final_third_slope(&flat, 120.0), Some(0.0));
    }

    #[test]
    fn long_documents_rank_on_reading_and_writing_together() {
        let fast_reader = Meas { prefill: 4000.0, decode0: 50.0, decode_depth: 40.0 };
        let fast_writer = Meas { prefill: 1000.0, decode0: 60.0, decode_depth: 48.0 };
        assert!(objective(Profile::LongDocs, &fast_reader, 32768) > objective(Profile::LongDocs, &fast_writer, 32768));
        assert!(objective(Profile::Chat, &fast_writer, 4096) > objective(Profile::Chat, &fast_reader, 4096));
    }

    #[test]
    fn reasons_name_what_changed_and_by_how_much() {
        let c = plan(&input());
        let base = Meas { prefill: 900.0, decode0: 10.0, decode_depth: 8.0 };
        let won = Meas { prefill: 950.0, decode0: 11.0, decode_depth: 9.0 };
        let r = reasons(Profile::Chat, 4096, &c[1], &base, &won, 28, 40, &[], None);
        assert!(r[0].contains("13% better") && r[0].contains("9.0 vs 8.0"), "{r:?}");
        assert!(r[1].contains("26 of 40"), "{r:?}");
        let keep = reasons(Profile::Chat, 4096, &c[0], &base, &base, 40, 40, &[], Some((&c[4], &Meas { decode_depth: 8.1, ..base })));
        assert!(keep[0].contains("all 40 layers") && keep[1].contains("+1%") && keep.last().is_some_and(|l| l.contains("rather than change")), "{keep:?}");
    }

    #[test]
    fn documents_are_deterministic_and_long_enough() {
        let a = document(Profile::Chat, 2000);
        assert_eq!(a, document(Profile::Chat, 2000));
        assert!(a.split_whitespace().count() >= 2000);
        assert!(document(Profile::Coding, 400).contains("fn "));
    }

    #[test]
    fn depth_is_capped_by_the_model() {
        assert_eq!(tune_depth(Profile::LongDocs, 8192), 8192);
        assert_eq!(tune_depth(Profile::Chat, 40960), 4096);
        assert_eq!(Profile::parse("long-docs"), Some(Profile::LongDocs));
        assert_eq!(Profile::parse("x"), None);
    }
}
