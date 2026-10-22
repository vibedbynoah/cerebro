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

use crate::engine::{self, ChatEvent, ChatRequest, Engine, HardwareProfile, LibraryModel, RequestRecord};
use tauri::ipc::Channel;
use tauri::State;

/// Processor, graphics card and memory, read from the OS and the GPU driver.
#[tauri::command]
pub async fn hw_detect() -> HardwareProfile {
    tauri::async_runtime::spawn_blocking(engine::hardware).await.unwrap_or_else(|_| engine::hardware())
}

/// Scans the model folders (Cerebro's, another app's, C:\models) and remembers the result.
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
pub async fn model_load(engine: State<'_, Engine>, stt: SttState<'_>, id: String, context: Option<u32>, gpu_layers: Option<u32>) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &id)?;
    if model.kind != "stt" && engine.activity() == "benchmark" {
        return Err("The Benchmark lab is measuring this computer. Wait for it to finish, or stop it there.".to_owned());
    }
    // A speech model loads into its own engine beside the chat model (stt.rs).
    if model.kind == "stt" {
        let s = std::sync::Arc::clone(stt.inner());
        tauri::async_runtime::spawn_blocking(move || s.load(&model)).await.map_err(|_| "The load failed.".to_owned())??;
        return Ok(loaded_json(&engine, &stt));
    }
    let ctx = context.unwrap_or(8192).clamp(512, 1 << 20) as usize;
    // Every layer (or more) means "as many as fit"; fewer forces a split; 0 the processor only.
    let force = gpu_layers.map(|n| n as usize).filter(|&n| n < model.arch.layers as usize);
    // The load runs on the engine thread; a blocking task waits for it.
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.load(model, ctx, force)).await.map_err(|_| "The load failed.".to_owned())??;
    Ok(loaded_json(&engine, &stt))
}

/// Tools > Browse servers, "Official registry": a live search of registry.modelcontextprotocol.io.
#[tauri::command]
pub async fn mcp_registry_search(query: String) -> Result<Vec<crate::mcp_registry::Entry>, String> {
    tauri::async_runtime::spawn_blocking(move || crate::mcp_registry::search(&query)).await.map_err(|_| "The search failed.".to_owned())?
}

/// Cluster: whether this computer is shared as a worker, its pairing code and addresses.
#[tauri::command]
pub fn cluster_share_get() -> crate::cluster::ShareView {
    crate::cluster::share_view()
}

/// Shares this computer: other Cerebro apps holding the pairing code may run layers of this model here.
#[tauri::command]
pub async fn cluster_share_start(engine: State<'_, Engine>, id: String) -> Result<crate::cluster::ShareView, String> {
    let model = find_model(&engine, &id)?;
    if model.kind != "text" && model.kind != "vision" {
        return Err("Only language models can be shared across computers.".to_owned());
    }
    tauri::async_runtime::spawn_blocking(move || crate::cluster::share_start(std::path::Path::new(&model.path), &model.name)).await.map_err(|_| "Sharing failed.".to_owned())?
}

#[tauri::command]
pub fn cluster_share_stop() -> crate::cluster::ShareView {
    crate::cluster::share_stop();
    crate::cluster::share_view()
}

/// Cerebro workers announcing themselves on the local network.
#[tauri::command]
pub async fn cluster_discover() -> Result<Vec<cerebro_engine::cluster::Announced>, String> {
    tauri::async_runtime::spawn_blocking(crate::cluster::discover).await.map_err(|_| "Looking for computers failed.".to_owned())
}

/// Loads a model across this computer and the chosen workers; returns each link's measurements.
#[tauri::command]
pub async fn cluster_load(engine: State<'_, Engine>, stt: SttState<'_>, id: String, context: Option<u32>, local_share: f64, workers: Vec<crate::cluster::WorkerChoice>) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &id)?;
    let ctx = context.unwrap_or(8192).clamp(512, 1 << 20) as usize;
    let eng = engine.inner().handle();
    let links = tauri::async_runtime::spawn_blocking(move || eng.load_cluster(model, ctx, local_share, workers)).await.map_err(|_| "The load failed.".to_owned())??;
    Ok(serde_json::json!({ "loaded": loaded_json(&engine, &stt), "links": links }))
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
pub async fn model_unload(engine: State<'_, Engine>, stt: SttState<'_>) -> Result<serde_json::Value, String> {
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.unload()).await.map_err(|_| "The unload failed.".to_owned())??;
    Ok(loaded_json(&engine, &stt))
}

#[tauri::command]
pub fn models_loaded(engine: State<'_, Engine>, stt: SttState<'_>) -> serde_json::Value {
    loaded_json(&engine, &stt)
}

type SttState<'a> = State<'a, std::sync::Arc<crate::stt::Stt>>;
/// The speech-synthesis model shown as loaded (it loads on first use; see tts.rs).
static TTS_LOADED: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
type MicState<'a> = State<'a, std::sync::Arc<crate::stt::Mic>>;

fn loaded_json(engine: &Engine, stt: &crate::stt::Stt) -> serde_json::Value {
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    if let Some(l) = &s.loaded {
        out.push(serde_json::json!({ "id": l.model.id, "slot": "llm", "params": null, "tuned": false, "placement": l.placement }));
    }
    if let Some(id) = stt.loaded_id() {
        out.push(serde_json::json!({ "id": id, "slot": "stt", "params": null, "tuned": false, "placement": null }));
    }
    if let Some(id) = TTS_LOADED.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        out.push(serde_json::json!({ "id": id, "slot": "tts", "params": null, "tuned": false, "placement": null }));
    }
    serde_json::Value::Array(out)
}

/// Starts recording from the default microphone (push-to-talk, or a speech model's test).
#[tauri::command]
pub async fn mic_start(mic: MicState<'_>) -> Result<(), String> {
    let mic = std::sync::Arc::clone(mic.inner());
    tauri::async_runtime::spawn_blocking(move || mic.start()).await.map_err(|_| "The microphone could not start.".to_owned())?
}

/// The microphone's recent peak level, 0..1 (0 when not recording), for the waveform.
#[tauri::command]
pub fn mic_level(mic: MicState<'_>) -> f32 {
    if mic.recording() { mic.level() } else { 0.0 }
}

/// Stops the recording and transcribes it with `model_id`, or the speech model in use (the
/// loaded one, else the first in the library).
#[tauri::command]
pub async fn stt_transcribe(engine: State<'_, Engine>, mic: MicState<'_>, stt: SttState<'_>, model_id: Option<String>) -> Result<String, String> {
    let samples = mic.stop()?;
    let model = match model_id {
        Some(id) => find_model(&engine, &id)?,
        None => {
            let library = engine.library.lock().unwrap_or_else(|e| e.into_inner()).clone();
            crate::stt::default_model(&stt, &library).ok_or("No speech-to-text model is in the library. Download one (for example Qwen3-ASR) from Model Hub.")?
        }
    };
    if model.kind != "stt" {
        return Err("That model does not transcribe speech.".to_owned());
    }
    let s = std::sync::Arc::clone(stt.inner());
    tauri::async_runtime::spawn_blocking(move || s.transcribe(&model, &samples)).await.map_err(|_| "Transcription failed.".to_owned())?
}

/// The bytes of an audio file, from the base64 the window sends. A file that is too large, or not
/// base64 at all, is refused with a plain sentence.
fn audio_bytes(data: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    if data.len() > 40_000_000 {
        return Err("That audio file is too large. Keep it under 30 MB.".to_owned());
    }
    base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .map_err(|_| "That audio file could not be read.".to_owned())
}

/// Transcribes an audio file (WAV) with `model_id`, or the speech model in use (the loaded one,
/// else the first in the library). The bytes arrive base64-encoded, so a recording can be turned
/// into notes without writing it to disk first.
#[tauri::command]
pub async fn stt_transcribe_file(engine: State<'_, Engine>, stt: SttState<'_>, data: String, model_id: Option<String>) -> Result<String, String> {
    let model = match model_id {
        Some(id) => find_model(&engine, &id)?,
        None => {
            let library = engine.library.lock().unwrap_or_else(|e| e.into_inner()).clone();
            crate::stt::default_model(&stt, &library).ok_or("No speech-to-text model is in the library. Download one (for example Qwen3-ASR) from Model Hub.")?
        }
    };
    if model.kind != "stt" {
        return Err("That model does not transcribe speech.".to_owned());
    }
    let bytes = audio_bytes(&data)?;
    let (samples, rate) = cerebro_engine::audio::read_wav(&bytes).map_err(|e| format!("That audio file could not be read: {e}"))?;
    let samples = cerebro_engine::audio::resample(&samples, rate, cerebro_engine::audio::SAMPLE_RATE);
    let s = std::sync::Arc::clone(stt.inner());
    tauri::async_runtime::spawn_blocking(move || s.transcribe(&model, &samples)).await.map_err(|_| "Transcription failed.".to_owned())?
}

type TtsState<'a> = State<'a, std::sync::Arc<crate::tts::Tts>>;type PlayerState<'a> = State<'a, std::sync::Arc<crate::tts::Player>>;

/// The speech-synthesis model `model_id`, or the one in use (loaded, else the first in the library).
fn tts_model(engine: &Engine, tts: &crate::tts::Tts, model_id: Option<String>) -> Result<LibraryModel, String> {
    let m = match model_id {
        Some(id) => find_model(engine, &id)?,
        None => {
            let library = engine.library.lock().unwrap_or_else(|e| e.into_inner()).clone();
            crate::tts::default_model(tts, &library).ok_or("No text-to-speech model is in the library. Download one (for example Qwen3-TTS) from Model Hub.")?
        }
    };
    if m.kind != "tts" {
        return Err("That model does not speak.".to_owned());
    }
    *TTS_LOADED.lock().unwrap_or_else(|e| e.into_inner()) = Some(m.id.clone());
    Ok(m)
}

/// The Models screen's speech test: speaks `text` with `model_id` and starts playing it; resolves
/// with its length in seconds as playback begins.
#[tauri::command]
pub async fn tts_sample(engine: State<'_, Engine>, tts: TtsState<'_>, player: PlayerState<'_>, model_id: String, text: String) -> Result<f64, String> {
    if text.trim().is_empty() || text.chars().count() > 280 {
        return Err("Use a sentence of 1 to 280 characters.".to_owned());
    }
    let m = tts_model(&engine, &tts, Some(model_id))?;
    let (t, p) = (std::sync::Arc::clone(tts.inner()), std::sync::Arc::clone(player.inner()));
    tauri::async_runtime::spawn_blocking(move || {
        p.stop();
        let x = t.synthesize(&m, &text)?;
        p.play(&x)
    })
    .await
    .map_err(|_| "Speech failed.".to_owned())?
}

/// Reads `text` aloud with the speech model in use, sentence by sentence (each plays while the
/// next is made); resolves when the reading has finished or was stopped.
#[tauri::command]
pub async fn tts_read(engine: State<'_, Engine>, tts: TtsState<'_>, player: PlayerState<'_>, text: String) -> Result<(), String> {
    let m = tts_model(&engine, &tts, None)?;
    let (t, p) = (std::sync::Arc::clone(tts.inner()), std::sync::Arc::clone(player.inner()));
    tauri::async_runtime::spawn_blocking(move || {
        p.stop();
        t.stopping.store(false, std::sync::atomic::Ordering::SeqCst);
        for s in crate::tts::sentences(&text).into_iter().take(40) {
            if t.stopping.load(std::sync::atomic::Ordering::SeqCst) {
                return Ok(());
            }
            let x = t.synthesize(&m, &s)?;
            if t.stopping.load(std::sync::atomic::Ordering::SeqCst) {
                return Ok(());
            }
            p.play(&x)?;
        }
        while p.remaining() > 0.0 && !t.stopping.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        Ok(())
    })
    .await
    .map_err(|_| "Reading aloud failed.".to_owned())?
}

/// Stops speech: the queued audio is dropped and a reading ends after the sentence being made.
#[tauri::command]
pub fn tts_stop(tts: TtsState<'_>, player: PlayerState<'_>) {
    tts.stopping.store(true, std::sync::atomic::Ordering::SeqCst);
    player.stop();
}

type ImagesState<'a> = State<'a, std::sync::Arc<crate::images::Images>>;

/// Image Studio: an image for `prompt` from image model `model_id`, `size` pixels square (a
/// multiple of 64, at most 1024), `steps` denoising steps (1 to 8), `seed` (random when absent).
/// Saved as a PNG under the app's data; resolves with it as a data: URL and its path.
#[tauri::command]
pub async fn image_generate(engine: State<'_, Engine>, images: ImagesState<'_>, model_id: String, prompt: String, size: u32, steps: u32, seed: Option<u64>) -> Result<crate::images::Generated, String> {
    let m = find_model(&engine, &model_id)?;
    if m.kind != "image" || !m.supported {
        return Err("That model does not make images (Stable Diffusion 2.x-family models such as SD-Turbo do).".to_owned());
    }
    if size % 64 != 0 || !(256..=1024).contains(&size) || !(1..=8).contains(&steps) || prompt.chars().count() > 2000 {
        return Err("Use a size of 256 to 1,024 pixels in steps of 64, 1 to 8 steps, and a prompt under 2,000 characters.".to_owned());
    }
    let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64 & 0xffff_ffff));
    let img = std::sync::Arc::clone(images.inner());
    tauri::async_runtime::spawn_blocking(move || img.generate(&m, crate::images::Request { prompt, size: size as usize, steps: steps as usize, seed }))
        .await
        .map_err(|_| "The image could not be made.".to_owned())?
}

/// Opens File Explorer at a generated image (only files in Cerebro's images folder).
#[tauri::command]
pub fn image_reveal<R: Runtime>(app: tauri::AppHandle<R>, path: String) -> Result<bool, String> {
    use tauri::Manager;
    let dir = app.path().app_data_dir().map_err(|_| "The images folder is unknown.".to_owned())?.join("images");
    let p = std::path::Path::new(&path);
    let inside = p.canonicalize().ok().zip(dir.canonicalize().ok()).is_some_and(|(f, d)| f.starts_with(d));
    if !inside {
        return Err("That file is not one of Cerebro's images.".to_owned());
    }
    Ok(crate::dedup::reveal(p))
}

/// The Models screen's image test: the loaded vision model describes one image (a data: URL).
#[tauri::command]
pub async fn model_describe_image(engine: State<'_, Engine>, id: String, data: String) -> Result<String, String> {
    if !data.starts_with("data:image/") || data.len() > 28_000_000 {
        return Err("Use a PNG, JPEG, WebP or GIF image under 20 MB.".to_owned());
    }
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.describe(id, data)).await.map_err(|_| "The test failed.".to_owned())?
}

// ---- Disk space (dedup.rs): the same file kept in more than one folder. Hashing takes seconds,
// so every one of these runs on a blocking thread. ----

type DiskState<'a> = State<'a, std::sync::Arc<crate::dedup::Disk>>;

/// Every model file stored more than once: grouped by exact byte size first, then SHA-256. The
/// first scan reads each file once; an unchanged library is rescanned without reading anything.
#[tauri::command]
pub async fn disk_scan(disk: DiskState<'_>, engine: State<'_, Engine>) -> Result<crate::dedup::DiskReport, String> {
    let disk = std::sync::Arc::clone(disk.inner());
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut library = engine.library.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if library.is_empty() {
            // The Models screen lists the library before the disk panel opens; if nothing has yet,
            // scan the folders now (and remember it, like library_list does).
            library = engine::scan_library();
            *engine.library.lock().unwrap_or_else(|e| e.into_inner()) = library.clone();
        }
        disk.scan(&library)
    })
    .await
    .map_err(|_| "The disk scan failed.".to_owned())
}

/// Keeps the copy at `keep_path` and replaces the group's other copies on its drive with hard
/// links to it, after checking every copy's hash from disk again.
#[tauri::command]
pub async fn disk_deduplicate(disk: DiskState<'_>, group_id: String, keep_path: String) -> Result<crate::dedup::DedupeResult, String> {
    let disk = std::sync::Arc::clone(disk.inner());
    tauri::async_runtime::spawn_blocking(move || disk.deduplicate(&group_id, &keep_path))
        .await
        .map_err(|_| "The deduplicate failed.".to_owned())?
}

/// Undoes a deduplicate: copies the bytes back so each path is its own file again.
#[tauri::command]
pub async fn disk_unlink(disk: DiskState<'_>, group_id: String) -> Result<crate::dedup::DuplicateGroup, String> {
    let disk = std::sync::Arc::clone(disk.inner());
    tauri::async_runtime::spawn_blocking(move || disk.unlink(&group_id)).await.map_err(|_| "The undo failed.".to_owned())?
}

/// Opens File Explorer at the model's file, with the file selected. False where that is not
/// possible (the UI then shows the path instead).
#[tauri::command]
pub fn model_reveal(engine: State<'_, Engine>, id: String) -> Result<bool, String> {
    let model = find_model(&engine, &id)?;
    Ok(crate::dedup::reveal(std::path::Path::new(&model.path)))
}

/// A model file's metadata for the Models screen: every entry (arrays summarised), the chat
/// template, and the edited copy when there is one.
#[tauri::command]
pub async fn model_detail(engine: State<'_, Engine>, id: String) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &id)?;
    tauri::async_runtime::spawn_blocking(move || {
        let path = std::path::Path::new(&model.path);
        let (entries, template) = crate::gguf_edit::detail(path)?;
        Ok(serde_json::json!({
            "id": model.id,
            "gguf": entries,
            "chatTemplate": template,
            "quirks": [],
            "author": { "rope": null, "sampler": null },
            "copy": crate::gguf_edit::existing_copy(path),
        }))
    })
    .await
    .map_err(|_| "The model's details could not be read.".to_owned())?
}

/// Writes (or rewrites) a metadata-edited copy of a model file beside it; the original is never
/// written. Copying gigabytes takes a while, so it runs off the command thread.
#[tauri::command]
pub async fn gguf_copy_write(engine: State<'_, Engine>, id: String, edits: Vec<crate::gguf_edit::Edit>) -> Result<crate::gguf_edit::Copy, String> {
    let model = find_model(&engine, &id)?;
    if crate::gguf_edit::is_copy(std::path::Path::new(&model.path)) {
        return Err("This is already an edited copy: edit the original instead.".to_owned());
    }
    tauri::async_runtime::spawn_blocking(move || crate::gguf_edit::write_copy(std::path::Path::new(&model.path), &edits))
        .await
        .map_err(|_| "The copy could not be written.".to_owned())?
}

/// Deletes a model's edited copy and its edit list.
#[tauri::command]
pub async fn gguf_copy_revert(engine: State<'_, Engine>, id: String) -> Result<(), String> {
    let model = find_model(&engine, &id)?;
    tauri::async_runtime::spawn_blocking(move || crate::gguf_edit::revert_copy(std::path::Path::new(&model.path)))
        .await
        .map_err(|_| "The copy could not be deleted.".to_owned())?
}

type TuneHubState<'a> = State<'a, std::sync::Arc<crate::tune_screen::TuneHub>>;

/// Starts the Tune screen's run for a library model and workload profile; the measuring happens on
/// the engine thread and streams to `tune_subscribe` listeners. Resolves with the run's info.
#[tauri::command]
pub fn tune_start(engine: State<'_, Engine>, hub: TuneHubState<'_>, model_id: String, profile: String, integrity_check: bool) -> Result<serde_json::Value, String> {
    let model = find_model(&engine, &model_id)?;
    if !model.supported {
        return Err(format!("Cerebro Engine does not run the {} architecture yet.", model.arch_name));
    }
    if model.kind != "text" && model.kind != "vision" {
        return Err("Only text and vision models can be tuned.".to_owned());
    }
    let profile = crate::tune_screen::Profile::parse(&profile).ok_or("That is not a workload profile.")?;
    let spec = crate::tune_screen::Spec { model, profile, integrity: integrity_check };
    let run = hub.begin(spec.clone(), crate::tune_screen::run_info(&spec))?;
    engine.tune_model(std::sync::Arc::clone(hub.inner()))?;
    Ok(run)
}

/// The run so far (one replay batch), then every new event, through `on_event`.
#[tauri::command]
pub fn tune_subscribe(hub: TuneHubState<'_>, on_event: Channel<serde_json::Value>) -> Result<(), String> {
    hub.subscribe(on_event)
}

/// Stops the running tune at the end of the step it is in (finished steps are kept).
#[tauri::command]
pub fn tune_cancel(hub: TuneHubState<'_>, run_id: String) -> Result<(), String> {
    hub.request_cancel(&run_id)
}

/// Continues a cancelled tune: finished steps are kept, the interrupted one runs again.
#[tauri::command]
pub fn tune_resume(engine: State<'_, Engine>, hub: TuneHubState<'_>, run_id: String) -> Result<(), String> {
    hub.resume(&run_id)?;
    engine.tune_model(std::sync::Arc::clone(hub.inner()))
}

/// Streams a reply through `on_event`; resolves when it is finished or stopped.
#[tauri::command]
pub async fn chat_send(engine: State<'_, Engine>, hub: TuneHubState<'_>, request: ChatRequest, on_event: Channel<ChatEvent>) -> Result<(), String> {
    if request.turns.is_empty() || request.turns.len() > 10_000 {
        return Err("A chat needs at least one message.".to_owned());
    }
    if hub.running() {
        return Err("A tune is measuring the model on the Tune screen. Wait for it to finish, or cancel it there.".to_owned());
    }
    if engine.activity() == "benchmark" {
        return Err("The Benchmark lab is measuring this computer. Wait for it to finish, or stop it there.".to_owned());
    }
    let sink: crate::engine::ChatSink = std::sync::Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    if let Some(rest) = request.model_id.strip_prefix(crate::cloud::PREFIX) {
        let (provider, model) = rest.split_once('/').map(|(a, b)| (a.to_owned(), b.to_owned())).ok_or_else(|| "That cloud model id is not valid.".to_owned())?;
        return tauri::async_runtime::spawn_blocking(move || cloud_reply(&provider, &model, &request, &sink)).await.map_err(|_| "The reply failed.".to_owned());
    }
    find_model(&engine, &request.model_id)?;
    let eng = engine.inner().handle();
    tauri::async_runtime::spawn_blocking(move || eng.chat(request, sink)).await.map_err(|_| "The reply failed.".to_owned())?
}

/// A reply from a cloud model (cloud.rs), streamed as the same events a local reply sends.
fn cloud_reply(provider: &str, model: &str, req: &ChatRequest, events: &crate::engine::ChatSink) {
    let t0 = std::time::Instant::now();
    let turns: Vec<crate::cloud::Turn<'_>> = req
        .turns
        .iter()
        .map(|t| crate::cloud::Turn { role: if t.role == "assistant" { "assistant" } else { "user" }, text: t.text.as_str(), images: t.images().collect() })
        .collect();
    let settings = crate::cloud::Settings { temperature: req.sampler.temperature, max_tokens: req.sampler.max_tokens.max(1) };
    let mut started: Option<f64> = None;
    let r = crate::cloud::chat(provider, model, req.system_prompt.trim(), &turns, &settings, |piece| {
        if started.is_none() {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            started = Some(ms);
            events(ChatEvent::Start { ttft_ms: ms });
        }
        match piece {
            crate::cloud::Piece::Thinking(t) => events(ChatEvent::Thinking { text: t.to_owned() }),
            crate::cloud::Piece::Text(t) => events(ChatEvent::Token { text: t.to_owned() }),
        }
        true
    });
    match r {
        Ok(o) => {
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let ttft_ms = started.unwrap_or(total_ms);
            let gen_s = ((total_ms - ttft_ms) / 1000.0).max(1e-3);
            events(ChatEvent::Done {
                stats: crate::engine::ChatStats {
                    prompt_tokens: 0,
                    length: false,
                    output_tokens: o.output_tokens,
                    thinking_tokens: o.thinking_pieces,
                    tok_per_sec: o.output_tokens as f64 / gen_s,
                    ttft_ms,
                    total_ms,
                    extra_ms: 0.0,
                    stopped: o.stopped,
                },
            });
        }
        Err(message) => events(ChatEvent::Error { message }),
    }
}

/// The cloud providers Cerebro knows (official API addresses and where to get a key).
#[tauri::command]
pub fn cloud_presets() -> Vec<crate::cloud::Preset> {
    crate::cloud::PRESETS.to_vec()
}

/// The providers the user added (never their keys).
#[tauri::command]
pub fn cloud_providers() -> Vec<crate::cloud::Provider> {
    crate::cloud::providers()
}

/// Adds or updates a provider; `api_key`, when given, goes to the Windows Credential Manager.
#[tauri::command]
pub fn cloud_provider_save(provider: crate::cloud::Provider, api_key: Option<String>) -> Result<Vec<crate::cloud::Provider>, String> {
    crate::cloud::save(provider, api_key)
}

#[tauri::command]
pub fn cloud_provider_remove(id: String) -> Result<Vec<crate::cloud::Provider>, String> {
    crate::cloud::remove(&id)
}

/// The models a provider offers with this key (or the stored one): the "Test" button.
#[tauri::command]
pub async fn cloud_models_list(provider: crate::cloud::Provider, api_key: Option<String>) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || crate::cloud::list_models(&provider, api_key.as_deref())).await.map_err(|_| "The check failed.".to_owned())?
}

/// Stops the reply being written.
#[tauri::command]
pub fn chat_stop(engine: State<'_, Engine>) {
    crate::cloud::cancel();
    engine.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Times this machine's processor settings on the loaded model (Turbo screen), streaming each
/// finished trial through `on_event`. Kept settings are saved (they survive a restart) and the
/// outcome is written to machine-tune.json so the screen can show it on the next visit.
/// Stopping uses `chat_stop`, the same switch the stop button in Chat sets.
#[tauri::command]
pub async fn tune_machine_start<R: Runtime>(app: tauri::AppHandle<R>, engine: State<'_, Engine>, on_event: Channel<serde_json::Value>) -> Result<serde_json::Value, String> {
    let eng = engine.inner().handle();
    let sink: crate::engine::TuneSink = std::sync::Arc::new(move |event| {
        if let Ok(value) = serde_json::to_value(&event) {
            let _ = on_event.send(value);
        }
    });
    let outcome = tauri::async_runtime::spawn_blocking(move || eng.tune_machine(sink)).await.map_err(|_| "Tuning failed.".to_owned())??;
    save_switches(&app);
    let json = serde_json::to_value(&outcome).map_err(|_| "The tuning result could not be read.".to_owned())?;
    if let Some(path) = machine_tune_file(&app) {
        let _ = std::fs::create_dir_all(path.parent().unwrap_or(&path));
        let _ = std::fs::write(path, serde_json::to_vec_pretty(&json).unwrap_or_default());
    }
    Ok(json)
}

/// The last finished tune's result (Turbo screen), or null before the first one.
#[tauri::command]
pub fn tune_machine_last<R: Runtime>(app: tauri::AppHandle<R>) -> serde_json::Value {
    machine_tune_file(&app)
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null)
}

fn machine_tune_file<R: Runtime>(app: &tauri::AppHandle<R>) -> Option<std::path::PathBuf> {
    use tauri::Manager;
    app.path().app_config_dir().ok().map(|d| d.join("machine-tune.json"))
}

// ---- Benchmark lab (bench.rs). A run loads the model it needs (or reuses it when already loaded
// at a large enough window), streams every finished point, and is kept in bench-history.json. ----

fn bench_history_file<R: Runtime>(app: &tauri::AppHandle<R>) -> Option<std::path::PathBuf> {
    use tauri::Manager;
    app.path().app_config_dir().ok().map(|d| d.join("bench-history.json"))
}

/// Adds `run` to the front of the saved history, replacing a run with the same id, at most
/// `bench::MAX_HISTORY` entries. A write failure leaves the screen's run intact.
fn save_bench_run<R: Runtime>(app: &tauri::AppHandle<R>, run: &crate::bench::BenchRun) {
    let Some(path) = bench_history_file(app) else { return };
    let mut history: Vec<crate::bench::BenchRun> = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    history.retain(|r| r.id != run.id);
    history.insert(0, run.clone());
    history.truncate(crate::bench::MAX_HISTORY);
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(&path));
    if let Ok(bytes) = serde_json::to_vec_pretty(&history) {
        let _ = std::fs::write(path, bytes);
    }
}

/// Runs the Benchmark lab on one library model, streaming the progress through `on_event`.
/// `placement` says where the model runs: "auto" (the placement it is loaded with), "gpu" (every
/// layer on the graphics card), "split" (half on the card) or "cpu" (the processor only).
/// Stopping uses `bench_stop`, which sets the same switch as Chat's stop button.
#[tauri::command]
pub async fn bench_run<R: Runtime>(app: tauri::AppHandle<R>, engine: State<'_, Engine>, hub: TuneHubState<'_>, model_id: String, placement: Option<String>, on_event: Channel<crate::bench::BenchEvent>) -> Result<crate::bench::BenchRun, String> {
    let model = find_model(&engine, &model_id)?;
    if model.kind != "text" && model.kind != "vision" {
        return Err("Only text and vision models can be benchmarked.".to_owned());
    }
    if !model.supported {
        return Err(format!("Cerebro Engine does not run the {} architecture yet.", model.arch_name));
    }
    let placement = placement.unwrap_or_else(|| "auto".to_owned());
    if !crate::bench::PLACEMENTS.contains(&placement.as_str()) {
        return Err("That placement is not one of the Benchmark lab's.".to_owned());
    }
    if hub.running() {
        return Err("A tune is measuring the model on the Tune screen. Wait for it to finish, or cancel it there.".to_owned());
    }
    if engine.activity() != "idle" {
        return Err("The engine is busy with another request. Wait for it to finish, then run the benchmark.".to_owned());
    }
    let eng = engine.inner().handle();
    let sink: crate::bench::BenchSink = std::sync::Arc::new(move |event| {
        let _ = on_event.send(event);
    });
    let run = tauri::async_runtime::spawn_blocking(move || eng.bench(model, placement, sink)).await.map_err(|_| "The benchmark failed.".to_owned())??;
    save_bench_run(&app, &run);
    Ok(run)
}

/// Every saved benchmark run, newest first. Empty before the first run.
#[tauri::command]
pub fn bench_history<R: Runtime>(app: tauri::AppHandle<R>) -> Vec<crate::bench::BenchRun> {
    bench_history_file(&app).and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// Stops the benchmark after the step it is in: finished points and questions are kept.
#[tauri::command]
pub fn bench_stop(engine: State<'_, Engine>) {
    engine.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// How much the saved-chat folder holds ("Keep long chats ready"): cache files and their total size.
#[tauri::command(async)]
pub fn kv_store_usage(engine: State<'_, Engine>) -> serde_json::Value {
    let (files, bytes) = engine.kv_store().map_or((0, 0), |kv| kv.lock().unwrap_or_else(|e| e.into_inner()).usage());
    serde_json::json!({ "files": files, "bytes": bytes })
}

/// Deletes every saved chat cache; returns what the folder held ({ files, bytes }).
#[tauri::command(async)]
pub fn kv_store_clear(engine: State<'_, Engine>) -> serde_json::Value {
    let (files, bytes) = engine.kv_store().map_or((0, 0), |kv| {
        let mut store = kv.lock().unwrap_or_else(|e| e.into_inner());
        let held = store.usage();
        store.clear();
        held
    });
    serde_json::json!({ "files": files, "bytes": bytes })
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
    let reqs: Vec<serde_json::Value> = s.requests.iter().map(request_summary_json).collect();
    let activity = s.activity;
    let running = s.requests.front().filter(|_| activity != "idle").map(|r| r.id.clone());
    // Prompt reading, live: tokens in the cache, tokens reused, tokens per second so far.
    let read = s.progress.as_ref().filter(|_| activity == "reading").map(|p| p.read());
    serde_json::json!({
        "provenance": "measured",
        "at": now,
        "live": {
            "state": activity, "queued": 0, "requestId": running,
            "promptRead": read.map_or(s.live_prompt, |r| r.0.min(s.live_prompt)), "promptTotal": s.live_prompt, "promptReused": read.map_or(s.live_reused, |r| r.1),
            "generated": s.live_generated, "maxTokens": 0,
            "decodeTps": if activity == "generating" { s.live_decode_tps } else { None },
            "prefillTps": read.and_then(|r| r.2), "firstTokenS": null,
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

/// One request as the Monitor's list shows it (mirrors `RequestSummary`); the snapshot and the
/// request inspector send the same fields, so the mapping lives in one place.
fn request_summary_json(r: &RequestRecord) -> serde_json::Value {
    serde_json::json!({
        "id": r.id, "startedAt": r.started_at, "method": "POST", "path": "/chat", "model": r.model,
        "state": r.state, "stream": true, "httpStatus": null, "wallS": r.wall_s, "loadS": null,
        "queueS": null, "firstTokenS": r.first_token_s, "promptTokens": r.prompt_tokens,
        "reusedTokens": r.reused_tokens, "outputTokens": r.output_tokens, "reasoningTokens": 0,
        "decodeTps": r.decode_tps, "prefillTps": r.prefill_tps, "responseFormat": "text",
        "session": "chat", "tool": null,
    })
}

/// How many characters of one capture field the inspector receives.
const CAPTURE_CAP: usize = 256 * 1024;

/// A capture field cut at [`CAPTURE_CAP`] characters; the flag says whether anything was cut.
fn cut_capture(text: &str) -> (String, bool) {
    if let Some((at, _)) = text.char_indices().nth(CAPTURE_CAP) {
        (text[..at].to_owned(), true)
    } else {
        (text.to_owned(), false)
    }
}

/// One request's capture: the summary the list shows, plus the bodies the engine kept (cut at
/// [`CAPTURE_CAP`] characters each) and the error when the request failed.
#[tauri::command(async)]
pub fn request_detail(engine: State<'_, Engine>, id: String) -> Result<serde_json::Value, String> {
    if id.is_empty() || id.len() > 64 {
        return Err("That is not a request id.".to_owned());
    }
    let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
    let Some(r) = s.requests.iter().find(|r| r.id == id) else {
        return Err("That request is no longer kept (the Monitor keeps the last 100).".to_owned());
    };
    let (input, input_cut) = cut_capture(&r.input);
    let (output, output_cut) = cut_capture(&r.output);
    let error = if r.state == "error" { r.error.clone() } else { None };
    let mut detail = request_summary_json(r);
    if let Some(o) = detail.as_object_mut() {
        o.insert("input".into(), input.into());
        o.insert("output".into(), output.clone().into());
        o.insert("reasoning".into(), String::new().into());
        o.insert("response".into(), output.into());
        o.insert("truncated".into(), serde_json::json!({ "input": input_cut, "output": output_cut, "reasoning": false, "response": output_cut }));
        o.insert("error".into(), serde_json::json!(error));
    }
    Ok(detail)
}

/// One column of a stored per-second series as its own series, oldest first.
fn column<const N: usize>(history: &std::collections::VecDeque<[Option<f64>; N]>, i: usize) -> Vec<Option<f64>> {
    history.iter().map(|row| row[i]).collect()
}

/// The seconds of history and the seconds per point of each named range.
fn range_shape(range: &str) -> Result<(usize, usize), String> {
    match range {
        "1m" => Ok((1, 60)),
        "15m" => Ok((5, 180)),
        "1h" => Ok((20, 180)),
        "24h" => Ok((480, 180)),
        _ => Err("That is not a history range.".to_owned()),
    }
}

/// One series' step means, oldest first. `samples` holds one sample per second with the newest
/// sample being now, so the point at `j` means the samples `(points - 1 - j) * step_s` seconds
/// back. A step with no sample inside it is null (the app may be younger than the range); null
/// samples inside a step are skipped, never counted as zero.
fn downsample(samples: &[Option<f64>], step_s: usize, points: usize) -> Vec<Option<f64>> {
    let mut out = Vec::with_capacity(points);
    for j in 0..points {
        let newest_age = (points - 1 - j) * step_s;
        let end = samples.len() as isize - 1 - newest_age as isize;
        let start = end - step_s as isize + 1;
        let mut sum = 0.0;
        let mut n = 0usize;
        for i in start.max(0)..=end {
            if let Some(v) = samples.get(i as usize).copied().flatten() {
                sum += v;
                n += 1;
            }
        }
        out.push((n > 0).then(|| sum / n as f64));
    }
    out
}

/// The Monitor's history for one range of the series the live snapshot charts, each point the mean
/// of its step's seconds, oldest first, ending now. Missing time (the app is younger than the
/// range) is null.
#[tauri::command(async)]
pub fn metrics_history(engine: State<'_, Engine>, tele: State<'_, Telemetry>, range: String) -> Result<serde_json::Value, String> {
    let (step_s, points) = range_shape(&range)?;
    let (decode, prefill, expert, vram, ram) = {
        let s = engine.status.lock().unwrap_or_else(|e| e.into_inner());
        (column(&s.history, 0), column(&s.history, 1), column(&s.history, 2), column(&s.history, 3), column(&s.history, 4))
    };
    let [gpu_util, gpu_temp, gpu_power, pcie_rx, cpu, disk_read] = tele.columns();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    Ok(serde_json::json!({
        "stepS": step_s,
        "end": now,
        "series": {
            "decodeTps": downsample(&decode, step_s, points),
            "prefillTps": downsample(&prefill, step_s, points),
            "gpuUtil": downsample(&gpu_util, step_s, points),
            "vramUsedGiB": downsample(&vram, step_s, points),
            "gpuTempC": downsample(&gpu_temp, step_s, points),
            "gpuPowerW": downsample(&gpu_power, step_s, points),
            "pcieRxMBs": downsample(&pcie_rx, step_s, points),
            "cpu": downsample(&cpu, step_s, points),
            "ramUsedGiB": downsample(&ram, step_s, points),
            "diskReadMBs": downsample(&disk_read, step_s, points),
            "expertHitPct": downsample(&expert, step_s, points),
            "draftAcceptPct": vec![None::<f64>; points],
        },
    }))
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

/// Live GPU/CPU telemetry (NVML + OS), sampled on each Monitor poll. The series the Monitor charts
/// (GPU load, temperature, power, PCIe receive, CPU load, disk read) keeps one sample per second
/// for 24 hours; the live snapshot reads only the newest 60 of them.
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

/// Telemetry history kept: 24 hours at one sample per second.
const TELEMETRY_KEPT: usize = 24 * 3600;

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
                    while st.history.len() > TELEMETRY_KEPT {
                        st.history.pop_front();
                    }
                }
                st.last = Some((std::time::Instant::now(), s.clone()));
                s
            }
        };
        let hist = &st.history;
        // Only the newest 60 samples go to the live snapshot; the rest wait for history queries.
        let n = hist.len().min(60);
        let col = |i: usize| -> Vec<Option<f64>> {
            let mut v = vec![None; 60 - n];
            v.extend(hist.iter().skip(hist.len() - n).map(|x| x[i]));
            v
        };
        (sample, [col(0), col(1), col(2), col(3), col(4), col(5)])
    }

    /// The stored per-second series (up to 24 h), one column per value, oldest first.
    fn columns(&self) -> [Vec<Option<f64>>; 6] {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: [Vec<Option<f64>>; 6] = Default::default();
        for row in &g.history {
            for (i, v) in row.iter().enumerate() {
                out[i].push(*v);
            }
        }
        out
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
            turns: turns.into_iter().map(|(role, text)| crate::engine::ChatTurn { role, text, attachments: Vec::new() }).collect(),
            // Qwen's recommended non-thinking sampler; min_p keeps long replies from looping.
            sampler: crate::engine::ChatSampler { temperature: 0.7, top_p: 0.8, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens: 4096 },
            native: None,
            // A phone never drives tools: approvals belong at the computer.
            tools: None,
            orchestrate: None,
            passes: 0,
            // The user's own conversation: worth a cache like a window chat.
            keep_cache: true,
            client_tools: None,
        };
        let out: crate::engine::ChatSink = std::sync::Arc::new(move |e| match e {
            ChatEvent::Token { text } => sink(PhoneEvent::Token(text)),
            ChatEvent::Thinking { text } => sink(PhoneEvent::Thinking(text)),
            ChatEvent::Done { stats } => sink(PhoneEvent::Done { tokens: stats.output_tokens as u64, tps: stats.tok_per_sec }),
            ChatEvent::Error { message } => sink(PhoneEvent::Error(message)),
            ChatEvent::Start { .. } | ChatEvent::Tool { .. } | ChatEvent::Ledger { .. } | ChatEvent::Check { .. } | ChatEvent::Revise { .. } => {}
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
            turns: turns.into_iter().map(|(role, text)| crate::engine::ChatTurn { role, text, attachments: Vec::new() }).collect(),
            sampler: crate::engine::ChatSampler { temperature, top_p: 0.8, min_p: 0.05, top_k: 20, repeat_penalty: 1.0, max_tokens },
            native: None,
            tools: None,
            orchestrate: None,
            passes: 0,
            // Another app's own request: its conversations are not the user's window chats.
            keep_cache: false,
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
            out.push(crate::engine::ChatTurn { role: "user".into(), text: std::mem::take(block).join("\n"), attachments: Vec::new() });
        }
    };
    for m in messages {
        match m {
            Msg::Tool(t) => tool_block.push(crate::toolcall::tool_response(t)),
            Msg::User(t) => {
                flush(&mut out, &mut tool_block);
                out.push(crate::engine::ChatTurn { role: "user".into(), text: t.clone(), attachments: Vec::new() });
            }
            Msg::Assistant { text, calls } => {
                flush(&mut out, &mut tool_block);
                let mut parts: Vec<String> = Vec::new();
                if !text.trim().is_empty() {
                    parts.push(text.trim().to_owned());
                }
                parts.extend(calls.iter().map(|c| crate::toolcall::render_call(dialect, c)));
                out.push(crate::engine::ChatTurn { role: "assistant".into(), text: parts.join("\n"), attachments: Vec::new() });
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
            passes: 0,
            // An API client manages its own context: no caches accumulate from its calls.
            keep_cache: false,
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
mod audio_tests {
    use super::audio_bytes;

    #[test]
    fn the_audio_bytes_are_read_from_base64() {
        assert_eq!(audio_bytes("aGk=").expect("base64"), b"hi");
        assert_eq!(audio_bytes(" aGk= ").expect("base64 with spaces around it"), b"hi");
        assert!(audio_bytes("!!!").is_err());
    }

    #[test]
    fn a_file_that_is_too_large_is_refused() {
        let big = "A".repeat(40_000_001);
        assert!(audio_bytes(&big).is_err());
    }
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

#[cfg(test)]
mod history_tests {
    use super::{downsample, range_shape};

    /// A history of `n` seconds, one sample per second, oldest first.
    fn samples(n: usize, f: impl Fn(usize) -> Option<f64>) -> Vec<Option<f64>> {
        (0..n).map(f).collect()
    }

    #[test]
    fn an_empty_history_is_all_gaps() {
        assert_eq!(downsample(&[], 1, 60), vec![None; 60]);
        assert_eq!(downsample(&[], 480, 180), vec![None; 180]);
    }

    #[test]
    fn a_step_of_only_nulls_is_null() {
        assert_eq!(downsample(&[None; 10], 1, 10), vec![None; 10]);
        // The newest step is all null while the older one holds a sample.
        assert_eq!(downsample(&[Some(4.0), None, None], 2, 2), vec![Some(4.0), None]);
    }

    #[test]
    fn time_before_the_app_started_is_null() {
        // 30 seconds of data read as a full minute: the oldest 30 points are gaps.
        let out = downsample(&samples(30, |_| Some(1.0)), 1, 60);
        assert!(out[..30].iter().all(|v| v.is_none()));
        assert!(out[30..].iter().all(|v| *v == Some(1.0)));
    }

    #[test]
    fn each_point_is_the_mean_of_its_step() {
        let out = downsample(&samples(10, |i| Some(i as f64 + 1.0)), 5, 2);
        assert_eq!(out, vec![Some(3.0), Some(8.0)]);
        // Nulls are skipped inside a step, never counted as zero.
        assert_eq!(downsample(&[Some(1.0), None, None, Some(3.0)], 4, 1), vec![Some(2.0)]);
    }

    #[test]
    fn the_newest_samples_fall_in_the_last_point() {
        // 120 seconds read as two minute points: 1..=60 and 61..=120.
        let out = downsample(&samples(120, |i| Some(i as f64 + 1.0)), 60, 2);
        assert_eq!(out, vec![Some(30.5), Some(90.5)]);
    }

    #[test]
    fn every_range_names_its_step_and_point_count() {
        assert_eq!(range_shape("1m"), Ok((1, 60)));
        assert_eq!(range_shape("15m"), Ok((5, 180)));
        assert_eq!(range_shape("1h"), Ok((20, 180)));
        assert_eq!(range_shape("24h"), Ok((480, 180)));
        assert!(range_shape("2h").is_err());
        for range in ["1m", "15m", "1h", "24h"] {
            let (step_s, points) = range_shape(range).expect("a known range");
            assert_eq!(downsample(&samples(30, |_| Some(1.0)), step_s, points).len(), points);
        }
    }
}
