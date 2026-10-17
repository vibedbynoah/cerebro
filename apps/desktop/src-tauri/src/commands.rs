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
pub async fn model_load(engine: State<'_, Engine>, id: String, context: Option<u32>) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &id)?;
    let ctx = context.unwrap_or(8192).clamp(512, 1 << 20) as usize;
    // The load runs on the engine thread; a blocking task waits for it.
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.load(model, ctx)).await.map_err(|_| "The load failed.".to_owned())??;
    Ok(loaded_json(&engine))
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
        Some(l) => serde_json::json!([{ "id": l.model.id, "slot": "llm", "params": null, "tuned": false }]),
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
    tauri::async_runtime::spawn_blocking(move || eng.chat(request, on_event)).await.map_err(|_| "The reply failed.".to_owned())?
}

/// Stops the reply being written.
#[tauri::command]
pub fn chat_stop(engine: State<'_, Engine>) {
    engine.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// The status bar's sample.
#[tauri::command]
pub fn live_stats(engine: State<'_, Engine>) -> serde_json::Value {
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    let gpu = cerebro_engine::hw::gpus();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    serde_json::json!({
        "provenance": "measured",
        "model": s.loaded.as_ref().map(|l| l.model.name.clone()),
        "tokensPerSecond": s.live_decode_tps.or(s.last_decode_tps),
        "vramUsedGiB": gpu.first().map_or(0.0, |g| gib(g.vram_total - g.vram_free)),
        "vramTotalGiB": gpu.first().map_or(0.0, |g| gib(g.vram_total)),
        "backend": { "state": if s.loaded.is_some() { "running" } else { "stopped" }, "address": null, "build": concat!("cerebro-engine ", env!("CARGO_PKG_VERSION")) },
    })
}

/// The Monitor's snapshot (camelCase, matching ui/src/service/monitor-types.ts). Values the engine
/// cannot read yet are null and show as a dash.
#[tauri::command]
pub fn metrics(engine: State<'_, Engine>, tele: State<'_, Telemetry>) -> serde_json::Value {
    let (live, th) = tele.tick();
    let gp = live.gpus.first().cloned().unwrap_or_default();
    let hw = cerebro_engine::hw::machine();
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let g0 = hw.gpus.first();
    let model = s.loaded.as_ref().map(|l| {
        serde_json::json!({
            "id": l.model.id, "name": l.model.name, "quant": l.model.quant, "params": l.model.params,
            "architecture": l.model.architecture, "contextMax": l.ctx, "contextUsed": l.context_used,
            "kvType": if l.on_gpu { "f16" } else { "f32" }, "layers": l.model.arch.layers,
            "gpuLayers": if l.on_gpu { l.model.arch.layers } else { 0 }, "expertLayersOnCpu": 0,
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
            "vramUsedGiB": g0.map(|g| gib(g.vram_total - g.vram_free)),
            "vramTotalGiB": g0.map_or(0.0, |g| gib(g.vram_total)),
            "vram": null, "gpuTempC": gp.temp_c, "gpuPowerW": gp.power_w, "gpuPowerLimitW": gp.power_limit_w,
            "pcieRxMBs": gp.pcie_rx_mbs, "pcieTxMBs": gp.pcie_tx_mbs, "pcieGen": gp.pcie_gen, "pcieGenMax": gp.pcie_gen_max, "pcieWidth": gp.pcie_width,
            "cpu": live.cpu_pct,
            "ramUsedGiB": gib(hw.ram_total - hw.ram_available), "ramTotalGiB": gib(hw.ram_total),
            "ramPinnedGiB": null, "diskReadMBs": null, "diskWriteMBs": null, "throttle": gp.throttle_reasons.iter().filter(|r| **r != "idle").collect::<Vec<_>>(),
            "gpus": hw.gpus.iter().enumerate().map(|(i, g)| serde_json::json!({
                "index": i, "name": g.name, "util": live.gpus.get(i).and_then(|t| t.util_pct), "memUsedGiB": gib(g.vram_total - g.vram_free),
                "memTotalGiB": gib(g.vram_total), "tempC": live.gpus.get(i).and_then(|t| t.temp_c),
                "powerW": live.gpus.get(i).and_then(|t| t.power_w), "powerLimitW": live.gpus.get(i).and_then(|t| t.power_limit_w),
                "pcieRxMBs": live.gpus.get(i).and_then(|t| t.pcie_rx_mbs),
            })).collect::<Vec<_>>(),
        },
        "hardwareStatic": {
            "gpuName": g0.map(|g| g.name.clone()), "gpuCount": hw.gpus.len(), "driver": gp.driver,
            "cpuName": hw.cpu_name, "cores": hw.cpu_threads, "threads": hw.cpu_threads,
            "ramTotalGiB": gib(hw.ram_total), "gpuThrottleTempC": null,
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
                "ramUsedGiB": col(4), "diskReadMBs": nulls, "expertHitPct": col(2), "draftAcceptPct": nulls,
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

use crate::hub::{self, DownloadEvent, Downloads, HubFile, HubRepo};

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
    inner: std::sync::Mutex<(cerebro_engine::telemetry::Sampler, std::collections::VecDeque<[Option<f64>; 5]>)>,
}

impl Telemetry {
    fn tick(&self) -> (cerebro_engine::telemetry::Sample, [Vec<Option<f64>>; 5]) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (sampler, hist) = &mut *g;
        let s = sampler.sample();
        let gp = s.gpus.first();
        hist.push_back([gp.and_then(|x| x.util_pct), gp.and_then(|x| x.temp_c), gp.and_then(|x| x.power_w), gp.and_then(|x| x.pcie_rx_mbs), s.cpu_pct]);
        while hist.len() > 60 {
            hist.pop_front();
        }
        let col = |i: usize| -> Vec<Option<f64>> {
            let mut v = vec![None; 60 - hist.len()];
            v.extend(hist.iter().map(|x| x[i]));
            v
        };
        (s, [col(0), col(1), col(2), col(3), col(4)])
    }
}
