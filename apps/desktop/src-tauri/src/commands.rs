//! The app's own IPC commands. Each is granted to the main window only (capabilities/main-window.json)
//! and allowlisted in the isolation hook. None accepts an argument from the webview: each acts on
//! the window that called it, which Tauri resolves on the Rust side, so there is nothing to forge
//! or validate beyond the caller's identity.

use serde::Serialize;
use tauri::{Runtime, WebviewWindow};

use crate::MAIN_WINDOW;

/// What the UI may learn about the app. Mirrors `AppInfo` in ui/src/service/types.ts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppInfo {
    /// Product name from tauri.conf.json.
    pub name: String,
    /// App version from tauri.conf.json.
    pub version: String,
    /// Always "desktop" here; the browser preview reports "web".
    pub mode: &'static str,
}

/// Errors cross the IPC boundary as one fixed sentence: no paths, OS messages or internals.
fn fail<E: std::fmt::Display>(action: &'static str) -> impl FnOnce(E) -> String {
    move |error| {
        eprintln!("window command could not {action}: {error}");
        format!("The window could not {action}.")
    }
}

/// Defence in depth behind the capability file: only the main window may drive window state.
fn require_main<R: Runtime>(window: &WebviewWindow<R>) -> Result<(), String> {
    if window.label() == MAIN_WINDOW {
        Ok(())
    } else {
        Err("This window may not use window controls.".to_owned())
    }
}

/// Minimizes the calling window.
#[tauri::command]
pub fn window_minimize<R: Runtime>(window: WebviewWindow<R>) -> Result<(), String> {
    require_main(&window)?;
    window.minimize().map_err(fail("minimize"))
}

/// Maximizes the calling window, or restores it if it is maximized.
#[tauri::command]
pub fn window_toggle_maximize<R: Runtime>(window: WebviewWindow<R>) -> Result<(), String> {
    require_main(&window)?;
    if window.is_maximized().map_err(fail("read its size"))? {
        window.unmaximize().map_err(fail("restore"))
    } else {
        window.maximize().map_err(fail("maximize"))
    }
}

/// Whether the calling window is maximized, so the title bar can show Maximize or Restore.
#[tauri::command]
pub fn window_is_maximized<R: Runtime>(window: WebviewWindow<R>) -> Result<bool, String> {
    require_main(&window)?;
    window.is_maximized().map_err(fail("read its size"))
}

/// Closes the calling window (the app exits when its last window closes).
#[tauri::command]
pub fn window_close<R: Runtime>(window: WebviewWindow<R>) -> Result<(), String> {
    require_main(&window)?;
    window.close().map_err(fail("close"))
}

/// Name and version of the running app.
#[tauri::command]
pub fn app_info<R: Runtime>(app: tauri::AppHandle<R>) -> AppInfo {
    let info = app.package_info();
    AppInfo { name: info.name.clone(), version: info.version.to_string(), mode: "desktop" }
}

// ---- Engine commands. Arguments from the webview are validated here (and by type in the isolation
// hook); errors are plain sentences without paths or internals. ----

use crate::engine::{self, ChatEvent, ChatRequest, Engine, HardwareProfile, LibraryModel};
use tauri::ipc::Channel;
use tauri::State;

/// Processor, graphics card and memory, read from the OS and the GPU driver.
#[tauri::command]
pub async fn hw_detect() -> HardwareProfile {
    tauri::async_runtime::spawn_blocking(engine::hardware).await.unwrap_or_else(|_| engine::hardware())
}

/// Scans the model folders (Cerebro's, LM Studio's, C:\models) and remembers the result.
#[tauri::command]
pub async fn library_list(engine: State<'_, Engine>) -> Result<Vec<LibraryModel>, String> {
    let list = tauri::async_runtime::spawn_blocking(engine::scan_library).await.map_err(|_| "The model scan failed.".to_owned())?;
    *engine.library.lock().unwrap_or_else(|e| e.into_inner()) = list.clone();
    Ok(list)
}

fn find_model(engine: &Engine, id: &str) -> Result<LibraryModel, String> {
    if id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err("That is not a model id.".to_owned());
    }
    engine.library.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|m| m.id == id).cloned().ok_or_else(|| "That model is not in the library.".to_owned())
}

/// Loads a library model into the engine (replacing the one loaded). `context` in tokens.
#[tauri::command]
pub async fn model_load(engine: State<'_, Engine>, id: String, context: Option<u32>, gpu_layers: Option<u32>) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &id)?;
    let ctx = context.unwrap_or(8192).clamp(512, 1 << 20) as usize;
    // Every layer (or more) means "as many as fit"; fewer forces a split; 0 the processor only.
    let force = gpu_layers.map(|n| n as usize).filter(|&n| n < model.arch.layers as usize);
    // The load runs on the engine thread; a blocking task waits for it.
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.load(model, ctx, force)).await.map_err(|_| "The load failed.".to_owned())??;
    Ok(loaded_json(&engine))
}

/// Where a model would go if loaded now: GPU layers, RAM, the drive, and an estimated speed. The
/// model loaded now counts as freed (loading replaces it).
#[tauri::command]
pub async fn placement_preview(engine: State<'_, Engine>, id: String, context: Option<u32>, gpu_layers: Option<u32>) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &id)?;
    let ctx = context.unwrap_or(8192).clamp(512, 1 << 20) as usize;
    let force = gpu_layers.map(|n| n as usize).filter(|&n| n < model.arch.layers as usize);
    let held = engine.status.lock().unwrap_or_else(|e| e.into_inner()).loaded.as_ref().map_or(0, |l| l.vram_bytes as u64);
    tauri::async_runtime::spawn_blocking(move || {
        let m = cerebro_engine::model::Model::load(std::path::Path::new(&model.path)).map_err(|e| format!("Could not read the model: {e}"))?;
        let vram = if force == Some(0) || !cerebro_engine::switches::enabled(cerebro_engine::switches::GPU_ENABLED) {
            0
        } else {
            cerebro_engine::hw::gpus().first().map_or(0, |g| g.vram_free + held)
        };
        let plan = cerebro_engine::gpu::plan_for(&m, ctx.min(m.cfg.ctx_train.max(512)), vram, force.filter(|&n| n > 0)).map_err(|e| e.to_string())?;
        serde_json::to_value(plan).map_err(|e| e.to_string())
    })
    .await
    .map_err(|_| "The preview failed.".to_owned())?
}

#[tauri::command]
pub async fn model_unload(engine: State<'_, Engine>) -> Result<serde_json::Value, String> {
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.unload()).await.map_err(|_| "The unload failed.".to_owned())??;
    Ok(loaded_json(&engine))
}

#[tauri::command]
pub fn models_loaded(engine: State<'_, Engine>) -> serde_json::Value {
    loaded_json(&engine)
}

fn loaded_json(engine: &Engine) -> serde_json::Value {
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    match &s.loaded {
        Some(l) => serde_json::json!([{ "id": l.model.id, "slot": "llm", "params": null, "tuned": false, "placement": l.placement }]),
        None => serde_json::json!([]),
    }
}

/// Streams a reply through `on_event`; resolves when it is finished or stopped.
#[tauri::command]
pub async fn chat_send(engine: State<'_, Engine>, request: ChatRequest, on_event: Channel<ChatEvent>) -> Result<(), String> {
    if request.turns.is_empty() || request.turns.len() > 10_000 {
        return Err("A chat needs at least one message.".to_owned());
    }
    find_model(&engine, &request.model_id)?;
    let eng = engine.inner().handle();
    let sink: crate::engine::ChatSink = std::sync::Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    tauri::async_runtime::spawn_blocking(move || eng.chat(request, sink)).await.map_err(|_| "The reply failed.".to_owned())?
}

/// Stops the reply being written.
#[tauri::command]
pub fn chat_stop(engine: State<'_, Engine>) {
    engine.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// The status bar's sample.
// `async` runs these off the UI thread: a poll never stalls the window, even while NVML is slow.
#[tauri::command(async)]
pub fn live_stats(engine: State<'_, Engine>, tele: State<'_, Telemetry>) -> serde_json::Value {
    let (live, _) = tele.tick();
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = live.gpus.first();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    serde_json::json!({
        "provenance": "measured",
        "model": s.loaded.as_ref().map(|l| l.model.name.clone()),
        "tokensPerSecond": s.live_decode_tps.or(s.last_decode_tps),
        "vramUsedGiB": gpu.and_then(|g| g.vram_used).map_or(0.0, gib),
        "vramTotalGiB": gpu.and_then(|g| g.vram_total).map_or(0.0, gib),
        "backend": { "state": if s.loaded.is_some() { "running" } else { "stopped" }, "address": null, "build": concat!("cerebro-engine ", env!("CARGO_PKG_VERSION")) },
    })
}

/// The Monitor's snapshot (camelCase, matching ui/src/service/monitor-types.ts). Values the engine
/// cannot read yet are null and show as a dash.
#[tauri::command(async)]
pub fn metrics(engine: State<'_, Engine>, tele: State<'_, Telemetry>) -> serde_json::Value {
    let (live, th) = tele.tick();
    let gp = live.gpus.first().cloned().unwrap_or_default();
    let (ram_total, ram_available) = cerebro_engine::hw::memory();
    let (cpu_name, cpu_threads) = machine_fixed();
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let model = s.loaded.as_ref().map(|l| {
        serde_json::json!({
            "id": l.model.id, "name": l.model.name, "quant": l.model.quant, "params": l.model.params,
            "architecture": l.model.architecture, "contextMax": l.ctx, "contextUsed": l.context_used,
            "kvType": if l.on_gpu { "f16" } else { "f32" }, "layers": l.model.arch.layers,
            "gpuLayers": if l.on_gpu && l.experts.is_some() { l.model.arch.layers as usize } else { l.gpu_layers }, "expertLayersOnCpu": 0,
            "placement": l.placement,
        })
    });
    let moe = s.loaded.as_ref().and_then(|l| {
        let (cached, total, hit) = l.experts?;
        Some(serde_json::json!({
            "expertsPerToken": l.model.arch.active_experts.unwrap_or(0), "expertsTotal": total,
            "cacheSlots": cached, "cacheHitPct": hit * 100.0, "channel": "stable",
        }))
    });
    let n = s.history.len().min(60);
    let tail: Vec<[Option<f64>; 5]> = s.history.iter().skip(s.history.len() - n).copied().collect();
    let col = |i: usize| -> Vec<Option<f64>> {
        let mut v: Vec<Option<f64>> = vec![None; 60 - n];
        v.extend(tail.iter().map(|x| x[i]));
        v
    };
    let nulls: Vec<Option<f64>> = vec![None; 60];
    let reqs: Vec<serde_json::Value> = s
        .requests
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id, "startedAt": r.started_at, "method": "POST", "path": "/chat", "model": r.model,
                "state": r.state, "stream": true, "httpStatus": null, "wallS": r.wall_s, "loadS": null,
                "queueS": null, "firstTokenS": r.first_token_s, "promptTokens": r.prompt_tokens,
                "reusedTokens": r.reused_tokens, "outputTokens": r.output_tokens, "reasoningTokens": 0,
                "decodeTps": r.decode_tps, "prefillTps": r.prefill_tps, "responseFormat": "text",
                "session": "chat", "tool": null,
            })
        })
        .collect();
    let activity = s.activity;
    let running = s.requests.front().filter(|_| activity != "idle").map(|r| r.id.clone());
    serde_json::json!({
        "provenance": "measured",
        "at": now,
        "live": {
            "state": activity, "queued": 0, "requestId": running,
            "promptRead": s.live_prompt, "promptTotal": s.live_prompt, "promptReused": s.live_reused,
            "generated": s.live_generated, "maxTokens": 0,
            "decodeTps": if activity == "generating" { s.live_decode_tps } else { None },
            "prefillTps": null, "firstTokenS": null,
            "phase": if activity == "generating" { Some("answer") } else { None },
        },
        "hardware": {
            "gpuUtil": gp.util_pct,
            "vramUsedGiB": gp.vram_used.map(gib),
            "vramTotalGiB": gp.vram_total.map_or(0.0, gib),
            "vram": null, "gpuTempC": gp.temp_c, "gpuPowerW": gp.power_w, "gpuPowerLimitW": gp.power_limit_w,
            "pcieRxMBs": gp.pcie_rx_mbs, "pcieTxMBs": gp.pcie_tx_mbs, "pcieGen": gp.pcie_gen, "pcieGenMax": gp.pcie_gen_max, "pcieWidth": gp.pcie_width,
            "cpu": live.cpu_pct,
            "ramUsedGiB": gib(ram_total.saturating_sub(ram_available)), "ramTotalGiB": gib(ram_total),
            "ramPinnedGiB": null, "diskReadMBs": live.disk_read_mbs, "diskWriteMBs": live.disk_write_mbs, "throttle": gp.throttle_reasons.iter().filter(|r| **r != "idle").collect::<Vec<_>>(),
            "gpus": live.gpus.iter().map(|g| serde_json::json!({
                "index": g.index, "name": g.name, "util": g.util_pct, "memUsedGiB": g.vram_used.map(gib),
                "memTotalGiB": g.vram_total.map(gib), "tempC": g.temp_c,
                "powerW": g.power_w, "powerLimitW": g.power_limit_w, "pcieRxMBs": g.pcie_rx_mbs,
            })).collect::<Vec<_>>(),
        },
        "hardwareStatic": {
            "gpuName": gp.name, "gpuCount": live.gpus.len(), "driver": gp.driver,
            "cpuName": cpu_name, "cores": cpu_threads, "threads": cpu_threads,
            "ramTotalGiB": gib(ram_total), "gpuThrottleTempC": null,
        },
        "engine": {
            "version": env!("CARGO_PKG_VERSION"), "backend": "cerebro", "build": "cerebro-engine",
            "address": "in-process", "model": model, "moe": moe, "speculation": null, "baseline": null,
        },
        "history": {
            "stepS": 1, "end": now,
            "series": {
                "decodeTps": col(0), "prefillTps": col(1), "gpuUtil": th[0], "vramUsedGiB": col(3),
                "gpuTempC": th[1], "gpuPowerW": th[2], "pcieRxMBs": th[3], "cpu": th[4],
                "ramUsedGiB": col(4), "diskReadMBs": th[5], "expertHitPct": col(2), "draftAcceptPct": nulls,
            },
        },
        "totals": s.totals,
        "requests": reqs,
        "requestsKept": 100,
        "conversationCache": {
            "enabled": false, "slots": 1, "parked": 0, "budgetGiB": 0, "usedGiB": 0,
            "requests": s.totals.requests, "requestsReused": 0, "promptTokens": s.totals.prompt_tokens,
            "reusedTokens": s.totals.reused_tokens, "parks": 0, "restores": 0, "evictions": 0, "lastEvent": null,
        },
    })
}

// ---- Model Hub (network: huggingface.co only) ----

use crate::hub::{self, DownloadEvent, Downloads, HubFile, HubFileMeta, HubRepo};

/// Searches Hugging Face for repositories with GGUF files.
#[tauri::command]
pub async fn hub_search(query: String) -> Result<Vec<HubRepo>, String> {
    tauri::async_runtime::spawn_blocking(move || hub::search(&query)).await.map_err(|_| "The search failed.".to_owned())?
}

/// The GGUF files of one repository, with sizes.
#[tauri::command]
pub async fn hub_files(repo: String) -> Result<Vec<HubFile>, String> {
    tauri::async_runtime::spawn_blocking(move || hub::files(&repo)).await.map_err(|_| "Listing the files failed.".to_owned())?
}

/// The architecture numbers and real size of one file's GGUF header (metadata only), so the Hub can
/// size the KV cache and memory for it before anything is downloaded.
#[tauri::command]
pub async fn hub_file_meta(repo: String, file: String) -> Result<HubFileMeta, String> {
    tauri::async_runtime::spawn_blocking(move || hub::file_meta(&repo, &file)).await.map_err(|_| "Reading the file header failed.".to_owned())?
}

/// Downloads one model file (all its parts) into Cerebro's model folder, streaming progress.
#[tauri::command]
pub async fn hub_download(downloads: State<'_, Downloads>, repo: String, file: String, parts: Vec<String>, total: u64, on_event: Channel<DownloadEvent>) -> Result<String, String> {
    let key = format!("{repo}/{file}");
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let mut map = downloads.cancel.lock().unwrap_or_else(|e| e.into_inner());
        if map.contains_key(&key) {
            return Err("That file is already downloading.".into());
        }
        map.insert(key.clone(), flag.clone());
    }
    let ev = on_event.clone();
    let r = tauri::async_runtime::spawn_blocking(move || hub::download(&repo, &file, &parts, total, &ev, &flag)).await.map_err(|_| "The download failed.".to_owned())?;
    downloads.cancel.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
    if let Err(e) = &r {
        let _ = on_event.send(DownloadEvent::Error { message: e.clone() });
    }
    r
}

/// Stops a running download (the partial file is kept and resumed next time).
#[tauri::command]
pub fn hub_cancel(downloads: State<'_, Downloads>, repo: String, file: String) {
    if let Some(f) = downloads.cancel.lock().unwrap_or_else(|e| e.into_inner()).get(&format!("{repo}/{file}")) {
        f.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Engine switches (Turbo screen): every optimisation with its explanation, channel and value.
#[tauri::command]
pub fn switches_list() -> Vec<cerebro_engine::switches::State> {
    cerebro_engine::switches::catalogue()
}

/// Sets one switch and saves the user's choices; returns the stored (clamped) value.
#[tauri::command]
pub fn switch_set<R: Runtime>(app: tauri::AppHandle<R>, id: String, value: i64) -> Result<i64, String> {
    let v = cerebro_engine::switches::set(&id, value)?;
    save_switches(&app);
    Ok(v)
}

/// Back to the start-up values.
#[tauri::command]
pub fn switches_reset<R: Runtime>(app: tauri::AppHandle<R>) -> Vec<cerebro_engine::switches::State> {
    cerebro_engine::switches::reset();
    save_switches(&app);
    cerebro_engine::switches::catalogue()
}

fn switches_file<R: Runtime>(app: &tauri::AppHandle<R>) -> Option<std::path::PathBuf> {
    use tauri::Manager;
    app.path().app_config_dir().ok().map(|d| d.join("switches.json"))
}

/// Writes the switches that differ from their defaults to the app's config folder.
fn save_switches<R: Runtime>(app: &tauri::AppHandle<R>) {
    let changed: serde_json::Map<String, serde_json::Value> = cerebro_engine::switches::catalogue()
        .into_iter()
        .filter(|s| s.value != s.default)
        .map(|s| (s.info.id.to_string(), s.value.into()))
        .collect();
    if let Some(p) = switches_file(app) {
        let _ = std::fs::create_dir_all(p.parent().unwrap_or(&p));
        let _ = std::fs::write(p, serde_json::to_vec_pretty(&changed).unwrap_or_default());
    }
}

/// Applies saved switches at start-up (before any model loads or the worker pool starts).
pub fn load_switches<R: Runtime>(app: &tauri::AppHandle<R>) {
    let Some(p) = switches_file(app) else { return };
    let Ok(bytes) = std::fs::read(p) else { return };
    let Ok(map) = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&bytes) else { return };
    for (id, v) in map {
        if let Some(v) = v.as_i64() {
            let _ = cerebro_engine::switches::set(&id, v);
        }
    }
}

/// Live GPU/CPU telemetry (NVML + OS), sampled on each Monitor poll, with the last 60 samples of the
/// series the Monitor charts: GPU load, temperature, power, PCIe receive and CPU load.
#[derive(Default)]
pub struct Telemetry {
    inner: std::sync::Mutex<TelemetryState>,
}

#[derive(Default)]
struct TelemetryState {
    sampler: cerebro_engine::telemetry::Sampler,
    /// One row per second: GPU load, temperature, power, PCIe receive, CPU load, disk read.
    history: std::collections::VecDeque<[Option<f64>; 6]>,
    /// The latest reading; callers inside FRESH share it instead of sampling again.
    last: Option<(std::time::Instant, cerebro_engine::telemetry::Sample)>,
}

/// The UI polls at 3 Hz from more than one place (status bar, Monitor); within this window they all
/// get the same reading, so the hardware is asked at most ~3 times a second.
const FRESH: std::time::Duration = std::time::Duration::from_millis(250);

impl Telemetry {
    fn tick(&self) -> (cerebro_engine::telemetry::Sample, [Vec<Option<f64>>; 6]) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *g;
        let sample = match &st.last {
            Some((at, s)) if at.elapsed() < FRESH => s.clone(),
            _ => {
                let s = st.sampler.sample();
                if s.new_second {
                    let gp = s.gpus.first();
                    st.history.push_back([gp.and_then(|x| x.util_pct), gp.and_then(|x| x.temp_c), gp.and_then(|x| x.power_w), gp.and_then(|x| x.pcie_rx_mbs), s.cpu_pct, s.disk_read_mbs]);
                    while st.history.len() > 60 {
                        st.history.pop_front();
                    }
                }
                st.last = Some((std::time::Instant::now(), s.clone()));
                s
            }
        };
        let hist = &st.history;
        let col = |i: usize| -> Vec<Option<f64>> {
            let mut v = vec![None; 60 - hist.len()];
            v.extend(hist.iter().map(|x| x[i]));
            v
        };
        (sample, [col(0), col(1), col(2), col(3), col(4), col(5)])
    }
}

/// CPU name and thread count never change while the app runs; the registry is read once.
fn machine_fixed() -> (String, usize) {
    static FIXED: std::sync::OnceLock<(String, usize)> = std::sync::OnceLock::new();
    FIXED.get_or_init(|| (cerebro_engine::hw::cpu_name(), std::thread::available_parallelism().map_or(1, |n| n.get()))).clone()
}

// ---- Phone access (remote.rs) ----

use tauri::Manager as _;

/// The engine as a paired phone sees it: status, stop, and (when allowed) a reply.
pub struct PhoneBackend(pub tauri::AppHandle);

impl crate::remote::Backend for PhoneBackend {
    fn status(&self) -> serde_json::Value {
        let tele = self.0.state::<Telemetry>();
        let (live, _) = tele.tick();
        let engine = self.0.state::<Engine>();
        let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let (ram_total, ram_available) = cerebro_engine::hw::memory();
        let gp = live.gpus.first();
        let last = s.requests.front();
        serde_json::json!({
            "at": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            "engine": { "state": s.activity },
            "model": s.loaded.as_ref().map(|l| serde_json::json!({ "name": l.model.name, "quant": l.model.quant, "contextUsed": l.context_used, "contextMax": l.ctx })),
            "decodeTps": if s.activity == "generating" { s.live_decode_tps } else { None },
            "lastDecodeTps": s.last_decode_tps,
            "prefillTps": last.and_then(|r| r.prefill_tps),
            "gpu": gp.map(|g| serde_json::json!({
                "name": g.name, "utilPct": g.util_pct, "tempC": g.temp_c, "powerW": g.power_w,
                "vramUsedGiB": g.vram_used.map(gib), "vramTotalGiB": g.vram_total.map(gib),
            })),
            "cpuPct": live.cpu_pct,
            "ramUsedGiB": gib(ram_total.saturating_sub(ram_available)),
            "ramTotalGiB": gib(ram_total),
            "requests": s.requests.iter().take(5).map(|r| serde_json::json!({
                "id": r.id, "state": r.state, "startedAt": r.started_at, "outputTokens": r.output_tokens, "decodeTps": r.decode_tps,
            })).collect::<Vec<_>>(),
        })
    }

    fn stop(&self) {
        self.0.state::<Engine>().cancel.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn chat(&self, turns: Vec<(String, String)>, sink: Box<dyn Fn(crate::remote::PhoneEvent) + Send + Sync>) -> Result<(), String> {
        use crate::remote::PhoneEvent;
        let engine = self.0.state::<Engine>();
        let model_id = engine.status.lock().unwrap_or_else(|e| e.into_inner()).loaded.as_ref().map(|l| l.model.id.clone()).ok_or("Load a model on the computer first.")?;
        let request = ChatRequest {
            model_id,
            system_prompt: String::new(),
            turns: turns.into_iter().map(|(role, text)| crate::engine::ChatTurn { role, text }).collect(),
            // Qwen's recommended non-thinking sampler; min_p keeps long replies from looping.
            sampler: crate::engine::ChatSampler { temperature: 0.7, top_p: 0.8, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens: 4096 },
            native: None,
            // A phone never drives tools: approvals belong at the computer.
            tools: None,
            orchestrate: None,
            client_tools: None,
        };
        let out: crate::engine::ChatSink = std::sync::Arc::new(move |e| match e {
            ChatEvent::Token { text } => sink(PhoneEvent::Token(text)),
            ChatEvent::Thinking { text } => sink(PhoneEvent::Thinking(text)),
            ChatEvent::Done { stats } => sink(PhoneEvent::Done { tokens: stats.output_tokens as u64, tps: stats.tok_per_sec }),
            ChatEvent::Error { message } => sink(PhoneEvent::Error(message)),
            ChatEvent::Start { .. } | ChatEvent::Tool { .. } | ChatEvent::Ledger { .. } => {}
        });
        engine.handle().chat(request, out)
    }
}

pub fn remote_file<R: Runtime>(app: &tauri::AppHandle<R>) -> Option<std::path::PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("phone-access.json"))
}

type RemoteState<'a> = State<'a, std::sync::Arc<crate::remote::Remote>>;

#[tauri::command(async)]
pub fn remote_get(remote: RemoteState<'_>) -> serde_json::Value {
    remote.view()
}

#[tauri::command(async)]
pub fn remote_set_password(remote: RemoteState<'_>, password: String) -> Result<serde_json::Value, String> {
    remote.set_password(&password)?;
    Ok(remote.view())
}

#[tauri::command(async)]
pub fn remote_enable(remote: RemoteState<'_>, on: bool) -> Result<serde_json::Value, String> {
    remote.set_enabled(on)?;
    Ok(remote.view())
}

#[tauri::command(async)]
pub fn remote_options(remote: RemoteState<'_>, allow_chat: bool, allow_tailscale: bool, address: Option<String>, session_days: u32) -> Result<serde_json::Value, String> {
    remote.set_options(allow_chat, allow_tailscale, address, session_days)?;
    Ok(remote.view())
}

#[tauri::command(async)]
pub fn remote_pair(remote: RemoteState<'_>) -> Result<serde_json::Value, String> {
    remote.pair()
}

#[tauri::command(async)]
pub fn remote_revoke(remote: RemoteState<'_>, id: String) -> serde_json::Value {
    remote.revoke(&id);
    remote.view()
}

// ---- Tools in Chat (engine.rs ToolHost over the MCP client) ----

/// Chat's tool calls go to the MCP servers on the Tools screen.
pub struct McpToolHost(pub tauri::AppHandle);

impl crate::engine::ToolHost for McpToolHost {
    fn tools(&self) -> Vec<crate::toolcall::ToolSpec> {
        let mcp = self.0.state::<crate::mcp::Mcp>();
        let list = mcp.tools_json();
        list.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| {
                        Some(crate::toolcall::ToolSpec {
                            server: t["server"].as_str()?.to_owned(),
                            name: t["name"].as_str()?.to_owned(),
                            description: t["description"].as_str().unwrap_or("").to_owned(),
                            input_schema: if t["input_schema"].is_object() { t["input_schema"].clone() } else { serde_json::json!({ "type": "object", "properties": {} }) },
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn call(&self, server: &str, tool: &str, arguments: serde_json::Value, _ctx: crate::engine::ToolContext) -> Result<String, String> {
        let mcp = self.0.state::<crate::mcp::Mcp>();
        let v = mcp.call_tool(server, tool, arguments)?;
        Ok(mcp_result_text(&v))
    }
}

/// An MCP tools/call result as text for the model: text parts joined, other parts named, and an
/// error result marked as one.
pub fn mcp_result_text(v: &serde_json::Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    for c in v["content"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        match c["type"].as_str() {
            Some("text") => parts.push(c["text"].as_str().unwrap_or("").to_owned()),
            Some("resource") => parts.push(c["resource"]["text"].as_str().map_or_else(|| format!("[resource {}]", c["resource"]["uri"].as_str().unwrap_or("")), str::to_owned)),
            Some(kind) => parts.push(format!("[{kind} content not shown]")),
            None => {}
        }
    }
    if parts.is_empty() {
        if let Some(sc) = v.get("structuredContent") {
            parts.push(sc.to_string());
        }
    }
    let text = parts.join("\n");
    if v["isError"].as_bool() == Some(true) {
        format!("Error: {text}")
    } else {
        text
    }
}

/// The user's answer to a tool call waiting in Chat.
#[tauri::command(async)]
pub fn chat_tool_decide(engine: State<'_, Engine>, id: String, allow: bool) -> bool {
    engine.decide(&id, allow)
}

// ---- Built-in tools (builtin_tools.rs) ----

type FoldersState<'a> = State<'a, Arc<crate::builtin_tools::WorkspaceFolders>>;

/// The built-in tools' specs, in the same JSON shape as `mcp_tools`, for the Tools screen.
#[tauri::command(async)]
pub fn builtin_tools() -> serde_json::Value {
    crate::builtin_tools::tools_json()
}

/// The workspace folders the built-in tools may touch, as display strings.
#[tauri::command(async)]
pub fn builtin_folders_get(folders: FoldersState<'_>) -> Vec<String> {
    folders.list()
}

/// Replaces the workspace folders: validates, saves, then swaps them in. Returns the saved list.
#[tauri::command(async)]
pub fn builtin_folders_set<R: Runtime>(app: tauri::AppHandle<R>, workspace: FoldersState<'_>, folders: Vec<String>) -> Result<Vec<String>, String> {
    let checked = crate::builtin_tools::WorkspaceFolders::checked(&folders)?;
    crate::builtin_tools::save_folders(&app, &checked)?;
    workspace.replace(checked);
    Ok(workspace.list())
}

// ---- Cerebro as an MCP server (mcp_server.rs) ----

use std::sync::{Arc, Mutex};

/// The engine as other apps see it through MCP.
pub struct McpEngineBackend(pub tauri::AppHandle);

impl McpEngineBackend {
    fn library(&self) -> Vec<LibraryModel> {
        let engine = self.0.state::<Engine>();
        let mut lib = engine.library.lock().unwrap_or_else(|e| e.into_inner());
        if lib.is_empty() {
            *lib = crate::engine::scan_library();
        }
        lib.clone()
    }
}

impl crate::mcp_server::McpBackend for McpEngineBackend {
    fn generate(&self, system: &str, turns: Vec<(String, String)>, max_tokens: usize, temperature: f32) -> Result<(String, u64, f64, String), String> {
        let engine = self.0.state::<Engine>();
        let (model_id, name) = engine.status.lock().unwrap_or_else(|e| e.into_inner()).loaded.as_ref().map(|l| (l.model.id.clone(), l.model.name.clone())).ok_or("No model is loaded in Cerebro. Call load_model first.")?;
        let request = ChatRequest {
            model_id,
            system_prompt: system.to_owned(),
            turns: turns.into_iter().map(|(role, text)| crate::engine::ChatTurn { role, text }).collect(),
            sampler: crate::engine::ChatSampler { temperature, top_p: 0.8, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens },
            native: None,
            tools: None,
            orchestrate: None,
            client_tools: None,
        };
        let out: Arc<Mutex<(String, u64, f64, Option<String>)>> = Arc::default();
        let o = out.clone();
        let sink: crate::engine::ChatSink = Arc::new(move |e| {
            let mut g = o.lock().unwrap_or_else(|e| e.into_inner());
            match e {
                ChatEvent::Token { text } => g.0.push_str(&text),
                ChatEvent::Done { stats } => {
                    g.1 = stats.output_tokens as u64;
                    g.2 = stats.tok_per_sec;
                }
                ChatEvent::Error { message } => g.3 = Some(message),
                _ => {}
            }
        });
        engine.handle().chat(request, sink)?;
        let g = out.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = &g.3 {
            return Err(e.clone());
        }
        Ok((g.0.trim().to_owned(), g.1, g.2, name))
    }

    fn models(&self) -> serde_json::Value {
        let loaded = self.0.state::<Engine>().status.lock().unwrap_or_else(|e| e.into_inner()).loaded.as_ref().map(|l| l.model.id.clone());
        serde_json::Value::Array(
            self.library()
                .iter()
                .map(|m| serde_json::json!({ "id": m.id, "name": m.name, "quant": m.quant, "params": m.params, "sizeGiB": m.size_gi_b, "contextMax": m.context_max, "supported": m.supported, "loaded": loaded.as_deref() == Some(m.id.as_str()) }))
                .collect(),
        )
    }

    fn load(&self, id: &str, context: Option<usize>) -> Result<String, String> {
        self.library();
        let engine = self.0.state::<Engine>();
        let model = find_model(&engine, id)?;
        let name = model.name.clone();
        engine.handle().load(model, context.unwrap_or(8192), None)?;
        Ok(format!("Loaded {name}."))
    }

    fn status(&self) -> serde_json::Value {
        crate::remote::Backend::status(&PhoneBackend(self.0.clone()))
    }
}

type McpServerState<'a> = State<'a, Arc<crate::mcp_server::McpServer>>;

#[tauri::command(async)]
pub fn mcp_server_get(server: McpServerState<'_>) -> serde_json::Value {
    server.view()
}

#[tauri::command(async)]
pub fn mcp_server_enable(server: McpServerState<'_>, on: bool) -> Result<serde_json::Value, String> {
    server.set_enabled(on)?;
    Ok(server.view())
}

#[tauri::command(async)]
pub fn mcp_server_rotate_token(server: McpServerState<'_>) -> Result<serde_json::Value, String> {
    server.rotate_token()?;
    Ok(server.view())
}

pub fn mcp_server_file<R: Runtime>(app: &tauri::AppHandle<R>) -> Option<std::path::PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("mcp-server.json"))
}

// ---- Local API (local_api.rs) over the engine -----------------------------------------------------

/// The local API's view of the engine.
pub struct ApiEngineBackend(pub Engine);

/// The app's conversation as chat turns in the loaded model's own words: an assistant turn that
/// called tools carries the calls as the model writes them, and consecutive tool results become one
/// user turn of `<tool_response>` blocks, the way the model was trained to read them.
pub fn api_turns(messages: &[crate::local_api::Msg], dialect: crate::toolcall::Dialect) -> Vec<crate::engine::ChatTurn> {
    use crate::local_api::Msg;
    let mut out: Vec<crate::engine::ChatTurn> = Vec::new();
    let mut tool_block: Vec<String> = Vec::new();
    let flush = |out: &mut Vec<crate::engine::ChatTurn>, block: &mut Vec<String>| {
        if !block.is_empty() {
            out.push(crate::engine::ChatTurn { role: "user".into(), text: std::mem::take(block).join("\n") });
        }
    };
    for m in messages {
        match m {
            Msg::Tool(t) => tool_block.push(crate::toolcall::tool_response(t)),
            Msg::User(t) => {
                flush(&mut out, &mut tool_block);
                out.push(crate::engine::ChatTurn { role: "user".into(), text: t.clone() });
            }
            Msg::Assistant { text, calls } => {
                flush(&mut out, &mut tool_block);
                let mut parts: Vec<String> = Vec::new();
                if !text.trim().is_empty() {
                    parts.push(text.trim().to_owned());
                }
                parts.extend(calls.iter().map(|c| crate::toolcall::render_call(dialect, c)));
                out.push(crate::engine::ChatTurn { role: "assistant".into(), text: parts.join("\n") });
            }
        }
    }
    flush(&mut out, &mut tool_block);
    out
}

impl crate::local_api::ApiBackend for ApiEngineBackend {
    fn models(&self) -> Vec<crate::local_api::ApiModel> {
        let engine = &self.0;
        let loaded = engine.status.lock().unwrap_or_else(|e| e.into_inner()).loaded.as_ref().map(|l| l.model.id.clone());
        let mut lib = engine.library.lock().unwrap_or_else(|e| e.into_inner());
        if lib.is_empty() {
            *lib = crate::engine::scan_library();
        }
        lib.iter().filter(|m| m.supported).map(|m| crate::local_api::ApiModel { id: m.id.clone(), name: m.name.clone(), loaded: loaded.as_deref() == Some(m.id.as_str()) }).collect()
    }

    fn chat(&self, req: crate::local_api::ApiRequest, on: &mut dyn FnMut(crate::local_api::ApiEvent) -> bool) -> Result<crate::local_api::ApiDone, crate::local_api::ApiError> {
        use crate::local_api::{ApiDone, ApiError, ApiEvent};
        let engine = self.0.handle();
        let unavailable = |m: String| ApiError { status: 503, message: m };
        if let Some(id) = &req.load {
            let model = find_model(&engine, id).map_err(unavailable)?;
            let ctx = engine.status.lock().unwrap_or_else(|e| e.into_inner()).loaded.as_ref().map_or(16_384, |l| l.ctx);
            engine.load(model, ctx, None).map_err(unavailable)?;
        }
        let (model_id, name, dialect) = engine
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .loaded
            .as_ref()
            .map(|l| (l.model.id.clone(), l.model.id.clone(), l.dialect))
            .ok_or_else(|| unavailable("No model is loaded in Cerebro. Load one on the Models screen, or name one from /v1/models.".into()))?;
        let request = ChatRequest {
            model_id,
            system_prompt: req.system.clone(),
            turns: api_turns(&req.messages, dialect),
            sampler: crate::engine::ChatSampler { temperature: req.temperature, top_p: req.top_p, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens: req.max_tokens },
            native: req.think.then(|| serde_json::json!({})),
            tools: None,
            orchestrate: None,
            client_tools: (!req.tools.is_empty()).then(|| req.tools.clone()),
        };
        // The engine writes into a channel from its own thread; this thread hands each piece to the
        // connection and asks the engine to stop when the app has gone or a stop sequence was hit.
        let (tx, rx) = std::sync::mpsc::channel::<ChatEvent>();
        let sink: crate::engine::ChatSink = Arc::new(move |e| {
            let _ = tx.send(e);
        });
        let runner = engine.handle();
        let job = std::thread::spawn(move || runner.chat(request, sink));
        let mut done: Option<crate::engine::ChatStats> = None;
        let mut error: Option<String> = None;
        let mut stopping = false;
        for e in rx {
            let keep = match e {
                ChatEvent::Token { text } => stopping || on(ApiEvent::Text(text)),
                ChatEvent::Thinking { text } => stopping || on(ApiEvent::Thinking(text)),
                ChatEvent::Tool { name, arguments, state: "requested", .. } => stopping || on(ApiEvent::Call { name, arguments }),
                ChatEvent::Done { stats } => {
                    done = Some(stats);
                    true
                }
                ChatEvent::Error { message } => {
                    error = Some(message);
                    true
                }
                _ => true,
            };
            if !keep && !stopping {
                stopping = true;
                engine.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        job.join().map_err(|_| ApiError { status: 500, message: "The engine stopped during the reply.".into() })?.map_err(|m| ApiError { status: 500, message: m })?;
        if let Some(m) = error {
            return Err(ApiError { status: 500, message: m });
        }
        let s = done.ok_or_else(|| ApiError { status: 500, message: "The reply ended without finishing.".into() })?;
        Ok(ApiDone { model: name, prompt_tokens: s.prompt_tokens, output_tokens: s.output_tokens, length: s.length })
    }
}

/// The user's answer to a question (ask_user) or plan (exit_plan_mode) waiting in the chat.
#[tauri::command]
pub fn chat_tool_answer(engine: State<'_, Engine>, id: String, answer: serde_json::Value) -> bool {
    if id.len() > 64 || !answer.is_object() || answer.to_string().len() > 64 * 1024 {
        return false;
    }
    engine.answer(&id, answer)
}

type LocalApiState<'a> = State<'a, Arc<crate::local_api::LocalApi>>;

#[tauri::command(async)]
pub fn local_api_get(api: LocalApiState<'_>) -> serde_json::Value {
    api.view()
}

#[tauri::command(async)]
pub fn local_api_enable(api: LocalApiState<'_>, on: bool) -> Result<serde_json::Value, String> {
    api.set_enabled(on)?;
    Ok(api.view())
}

#[tauri::command(async)]
pub fn local_api_options(api: LocalApiState<'_>, require_key: bool, allow_lan: bool, allow_model_switch: bool) -> Result<serde_json::Value, String> {
    api.set_options(require_key, allow_lan, allow_model_switch)?;
    Ok(api.view())
}

#[tauri::command(async)]
pub fn local_api_rotate_key(api: LocalApiState<'_>) -> Result<serde_json::Value, String> {
    api.rotate_key()?;
    Ok(api.view())
}

pub fn local_api_file<R: Runtime>(app: &tauri::AppHandle<R>) -> Option<std::path::PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("local-api.json"))
}

#[cfg(test)]
mod api_turn_tests {
    use super::api_turns;
    use crate::local_api::Msg;
    use crate::toolcall::{Dialect, ToolCall};

    #[test]
    fn tool_history_becomes_the_models_own_turns() {
        let call = ToolCall { name: "w".into(), arguments: serde_json::json!({ "city": "Oslo" }) };
        let turns = api_turns(
            &[Msg::User("weather?".into()), Msg::Assistant { text: "Checking.".into(), calls: vec![call.clone(), call] }, Msg::Tool("12 C".into()), Msg::Tool("13 C".into()), Msg::User("thanks".into())],
            Dialect::Hermes,
        );
        let roles: Vec<&str> = turns.iter().map(|t| t.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "user", "user"]);
        assert!(turns[1].text.starts_with("Checking.\n<tool_call>\n{"));
        assert_eq!(turns[1].text.matches("<tool_call>").count(), 2);
        assert_eq!(turns[2].text, "<tool_response>\n12 C\n</tool_response>\n<tool_response>\n13 C\n</tool_response>");
    }
}

#[cfg(test)]
mod api_real_tests {
    /// The local API with a real model, driven by the official OpenAI and Anthropic Python SDKs
    /// (tests-py/api_sdks.py). Needs CEREBRO_SDK_PYTHON (a Python with `openai` and `anthropic`)
    /// and Qwen3-0.6B-Q8_0.gguf in a library folder.
    #[test]
    #[ignore = "needs a model file and a Python with the SDKs"]
    fn local_api_with_real_sdks() {
        use std::sync::Arc;
        let py = std::env::var("CEREBRO_SDK_PYTHON").expect("CEREBRO_SDK_PYTHON");
        let engine = crate::engine::Engine::start();
        let lib = crate::engine::scan_library();
        let m = lib.iter().find(|m| m.path.ends_with("Qwen3-0.6B-Q8_0.gguf")).expect("Qwen3-0.6B-Q8_0.gguf in the library").clone();
        *engine.library.lock().unwrap() = lib.clone();
        engine.load(m, 8192, None).unwrap();
        let api = crate::local_api::LocalApi::new(crate::local_api::Settings::default(), Arc::new(super::ApiEngineBackend(engine.handle())), Box::new(|_| {}));
        api.set_enabled(true).unwrap();
        let v = api.view();
        let out = std::process::Command::new(py)
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests-py/api_sdks.py"))
            .arg(v["url"].as_str().unwrap())
            .arg(v["key"].as_str().unwrap())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        println!("{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success() && stdout.contains("ALL PASS"), "{stdout}");
        api.set_enabled(false).unwrap();
    }
}
