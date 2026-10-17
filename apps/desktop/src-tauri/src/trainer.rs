//! Train mode: fine-tune a model on this computer with Unsloth, export it to GGUF, and add it to the
//! model library.
//!
//! Unsloth is Python, so training runs in a sidecar: a Python environment that `uv` builds under
//! %APPDATA%\Cerebro\trainer, running OUR script (trainer/train.py, bundled in the binary). The only
//! programs started are the pinned uv.exe (checked against a SHA-256 compiled into this file) and the
//! environment's python.exe running that script; nothing the user or a model types is executed. The
//! script prints one JSON event per line, which is parsed strictly and streamed to the Train screen.
//! Inference never uses this environment: trained models run on Cerebro Engine like any other GGUF.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::ipc::Channel;

/// uv, pinned: version and the SHA-256 of its Windows x86_64 release zip.
const UV_VERSION: &str = "0.12.22";
const UV_SHA256: &str = "ea1397797a0ca15f63516dd0f49c2dde9776db9be5861cab152ebe8ad199894d";
const PYTHON: &str = "3.12";
/// PyTorch with CUDA 12.8 kernels (Turing through Blackwell).
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
const TRAIN_PY: &str = include_str!("../trainer/train.py");
/// GGUF export tools, pinned to one upstream release: the quantizer from its Windows CPU build and
/// the HF-to-GGUF converter from its source. Both archives are checked against these hashes. Only
/// the files export needs are unpacked (the source tree has paths past Windows' 260-character limit).
const LLAMA_TAG: &str = "b11361";
const LLAMA_BIN_SHA256: &str = "0e0a61a1da9832724a292ce371361ba53bfea03c19fa72ddcfd22a4d846fc596";
const LLAMA_SRC_SHA256: &str = "56be04bc7012360843c5885742332fd6820ddd8d71c82f124b996504ec39e7bb";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
/// Lines of the script's stderr kept for the "Details" disclosure and error reports.
const STDERR_TAIL: usize = 200;

/// Exports the user can pick; anything else is refused before the job starts.
pub const QUANTS: [&str; 4] = ["q4_k_m", "q5_k_m", "q8_0", "f16"];

fn root() -> Result<PathBuf, String> {
    std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("Cerebro").join("trainer")).ok_or_else(|| "APPDATA is not set.".to_owned())
}

fn uv_exe(root: &Path) -> PathBuf {
    root.join("uv").join("uv.exe")
}

fn venv_python(root: &Path) -> PathBuf {
    root.join("venv").join("Scripts").join("python.exe")
}

fn tools_dir(root: &Path) -> PathBuf {
    root.join("llama.cpp")
}

fn system32() -> Result<PathBuf, String> {
    std::env::var_os("SystemRoot").map(|r| PathBuf::from(r).join("System32")).ok_or_else(|| "SystemRoot is not set.".to_owned())
}

/// Downloads `url` and returns its bytes only if they hash to `sha256`.
fn fetch_checked(url: &str, sha256: &str, what: &str) -> Result<Vec<u8>, String> {
    let bytes = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .and_then(|c| c.get(url).send())
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.bytes())
        .map_err(|e| format!("Could not download {what}: {e}"))?;
    if sha256_hex(&bytes) != sha256 {
        return Err(format!("The {what} download did not match its published checksum; nothing was installed."));
    }
    Ok(bytes.to_vec())
}

/// Unpacks a zip (or the named members of it) with Windows' own tar.exe, which reads zip.
fn untar(archive: &Path, dest: &Path, extra: &[&str], events: &Channel<TrainEvent>) -> Result<(), String> {
    let mut c = Command::new(system32()?.join("tar.exe"));
    c.arg("-xf").arg(archive).arg("-C").arg(dest).args(extra);
    run_streamed(c, events).map(|_| ())
}

fn tools_ready(dir: &Path) -> bool {
    dir.join("llama-quantize.exe").is_file() && dir.join("convert_hf_to_gguf.py").is_file() && std::fs::read_to_string(dir.join(".tag")).is_ok_and(|t| t.trim() == LLAMA_TAG)
}

/// The quantizer and converter that Unsloth's GGUF export calls, in a folder it is pointed at, so
/// it never downloads, builds or installs anything itself.
fn ensure_tools(root: &Path, events: &Channel<TrainEvent>) -> Result<(), String> {
    let dir = tools_dir(root);
    if tools_ready(&dir) {
        return Ok(());
    }
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let _ = events.send(TrainEvent::Setup { text: format!("Downloading GGUF export tools ({LLAMA_TAG})") });
    let bin = fetch_checked(&format!("https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"), LLAMA_BIN_SHA256, "quantizer")?;
    let src = fetch_checked(&format!("https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/{LLAMA_TAG}"), LLAMA_SRC_SHA256, "converter")?;
    let (bin_zip, src_zip) = (root.join("tools-bin.zip"), root.join("tools-src.zip"));
    std::fs::write(&bin_zip, bin).map_err(|e| e.to_string())?;
    std::fs::write(&src_zip, src).map_err(|e| e.to_string())?;
    let members = ["convert_hf_to_gguf.py", "conversion", "gguf-py"].map(|m| format!("llama.cpp-{LLAMA_TAG}/{m}"));
    let r = untar(&bin_zip, &dir, &["llama-quantize.exe", "*.dll", "LICENSE*"], events).and_then(|_| {
        let mut args = vec!["--strip-components=1"];
        args.extend(members.iter().map(String::as_str));
        untar(&src_zip, &dir, &args, events)
    });
    let _ = std::fs::remove_file(&bin_zip);
    let _ = std::fs::remove_file(&src_zip);
    r?;
    std::fs::write(dir.join(".tag"), LLAMA_TAG).map_err(|e| e.to_string())
}

/// The environment the sidecar runs in: the venv first on PATH (nothing it starts reaches another
/// Python), no user site-packages, no automatic system installs, export tools pinned.
fn sidecar_env(cmd: &mut Command, root: &Path) {
    let venv = root.join("venv");
    let mut path = std::ffi::OsString::from(venv.join("Scripts"));
    if let Ok(sys) = system32() {
        path.push(";");
        path.push(&sys);
        if let Some(win) = sys.parent() {
            path.push(";");
            path.push(win);
        }
    }
    // Unsloth writes compiled kernels under the working directory unless told otherwise; keep them
    // in the trainer folder (shared by jobs, so later runs start faster), never where the app started.
    cmd.current_dir(root)
        .env("UNSLOTH_COMPILE_LOCATION", root.join("compiled_cache"))
        .env("PATH", path)
        .env("VIRTUAL_ENV", &venv)
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("UNSLOTH_AUTO_INSTALL", "0")
        .env("UNSLOTH_LLAMA_CPP_PATH", tools_dir(root))
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME");
}

/// Written after a successful setup: what was installed, for the status line.
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Installed {
    pub uv: String,
    pub python: String,
    pub torch: String,
    pub unsloth: String,
    pub cuda: bool,
}

/// One training job, as the Train screen sends it. Validated by [`TrainJob::check`].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TrainJob {
    /// Display name; also the folder the trained model lands in.
    pub name: String,
    /// Hugging Face model id, e.g. unsloth/Qwen3-0.6B.
    pub model: String,
    /// A local dataset file (.jsonl/.json/.csv/.parquet) or a Hugging Face dataset id.
    pub dataset_path: Option<String>,
    pub dataset_hf: Option<String>,
    pub format: String,
    pub method: String,
    pub rank: u32,
    pub alpha: u32,
    pub lr: f64,
    pub epochs: f64,
    pub max_steps: u32,
    pub max_seq_len: u32,
    pub batch: u32,
    pub grad_accum: u32,
    pub export: Vec<String>,
}

fn hf_id(s: &str) -> bool {
    let mut parts = s.split('/');
    let ok = |p: Option<&str>| p.is_some_and(|p| !p.is_empty() && p.len() <= 96 && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) && p != "." && p != "..");
    ok(parts.next()) && ok(parts.next()) && parts.next().is_none()
}

impl TrainJob {
    /// Every field in range; the dataset is one existing file of a known kind or a Hugging Face id.
    pub fn check(&self) -> Result<(), String> {
        let name_ok = !self.name.trim().is_empty() && self.name.len() <= 64 && self.name.chars().all(|c| c.is_alphanumeric() || " -_.".contains(c));
        if !name_ok {
            return Err("Give the model a name of letters, digits, spaces, - _ or . (up to 64).".into());
        }
        if !hf_id(&self.model) {
            return Err("The base model must be a Hugging Face id like unsloth/Qwen3-0.6B.".into());
        }
        match (&self.dataset_path, &self.dataset_hf) {
            (Some(p), None) => {
                let path = Path::new(p);
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                if !matches!(ext.as_str(), "jsonl" | "json" | "csv" | "parquet") || !path.is_file() {
                    return Err("The dataset must be an existing .jsonl, .json, .csv or .parquet file.".into());
                }
            }
            (None, Some(h)) if hf_id(h) => {}
            _ => return Err("Pick one dataset: a file, or a Hugging Face dataset id.".into()),
        }
        if !matches!(self.format.as_str(), "auto" | "alpaca" | "chatml" | "sharegpt") || !matches!(self.method.as_str(), "qlora" | "lora") {
            return Err("Unknown dataset format or training method.".into());
        }
        let in_range = (1..=256).contains(&self.rank)
            && (1..=512).contains(&self.alpha)
            && self.lr > 0.0
            && self.lr <= 0.01
            && self.epochs > 0.0
            && self.epochs <= 50.0
            && self.max_steps <= 100_000
            && (128..=32_768).contains(&self.max_seq_len)
            && (1..=64).contains(&self.batch)
            && (1..=256).contains(&self.grad_accum);
        if !in_range {
            return Err("A training setting is out of range.".into());
        }
        if self.export.len() > QUANTS.len() || !self.export.iter().all(|q| QUANTS.contains(&q.as_str())) {
            return Err("Unknown export format.".into());
        }
        Ok(())
    }

    fn folder_name(&self) -> String {
        self.name.trim().replace(' ', "-")
    }
}

/// What the Train screen hears while setting up or training.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum TrainEvent {
    /// A line of setup progress (uv, pip).
    Setup { text: String },
    Status { stage: String, detail: Option<String> },
    Log { step: u64, max_steps: u64, loss: Option<f64>, eval_loss: Option<f64>, lr: Option<f64>, grad_norm: Option<f64>, epoch: Option<f64> },
    Export { quant: String, path: String },
    /// Finished: the GGUF files now in the model library, and where the adapter is.
    Done { models: Vec<String>, adapter: Option<String>, steps: u64, seconds: f64, stopped: bool },
    Error { message: String },
}

/// One stdout line of train.py as an event; anything else is ignored (never trusted as an event).
pub fn parse_event(line: &str) -> Option<TrainEvent> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    let s = |k: &str, max: usize| v[k].as_str().map(|x| x.chars().take(max).collect::<String>());
    let f = |k: &str| v[k].as_f64().filter(|x| x.is_finite());
    Some(match v["type"].as_str()? {
        "status" => TrainEvent::Status { stage: s("stage", 32)?, detail: s("detail", 300) },
        "log" => TrainEvent::Log { step: v["step"].as_u64()?, max_steps: v["max_steps"].as_u64().unwrap_or(0), loss: f("loss"), eval_loss: f("eval_loss"), lr: f("lr"), grad_norm: f("grad_norm"), epoch: f("epoch") },
        "export" => TrainEvent::Export { quant: s("quant", 16)?, path: s("path", 1000)? },
        "done" => TrainEvent::Done {
            models: v["gguf"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).take(16).collect()).unwrap_or_default(),
            adapter: s("adapter", 1000),
            steps: v["steps"].as_u64().unwrap_or(0),
            seconds: f("seconds").unwrap_or(0.0),
            stopped: v["stopped"].as_bool().unwrap_or(false),
        },
        "error" => TrainEvent::Error { message: s("message", 2000)? },
        _ => return None,
    })
}

#[derive(Default)]
pub struct Trainer {
    /// The running job's folder (its stop file lives there) and whether setup is running.
    running: Mutex<Option<PathBuf>>,
    setting_up: AtomicBool,
    tail: Arc<Mutex<std::collections::VecDeque<String>>>,
}

fn installed(root: &Path) -> Option<Installed> {
    let bytes = std::fs::read(root.join("installed.json")).ok()?;
    let i: Installed = serde_json::from_slice(&bytes).ok()?;
    venv_python(root).is_file().then_some(i)
}

/// Runs a setup step, streaming its output lines (uv reports on stderr).
fn run_streamed(mut cmd: Command, events: &Channel<TrainEvent>) -> Result<String, String> {
    use std::os::windows::process::CommandExt;
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null()).creation_flags(CREATE_NO_WINDOW).spawn().map_err(|e| format!("Could not start setup: {e}"))?;
    let stderr = child.stderr.take();
    let ev = events.clone();
    let err_thread = std::thread::spawn(move || {
        let mut last = String::new();
        if let Some(s) = stderr {
            for line in BufReader::new(s).lines().map_while(Result::ok) {
                let line = line.trim().to_owned();
                if !line.is_empty() {
                    let _ = ev.send(TrainEvent::Setup { text: line.chars().take(300).collect() });
                    last = line;
                }
            }
        }
        last
    });
    let mut out = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut out);
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    let last = err_thread.join().unwrap_or_default();
    if !status.success() {
        return Err(format!("Setup step failed: {last}"));
    }
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Downloads uv.exe (if missing), checks its hash, unpacks it.
fn ensure_uv(root: &Path, events: &Channel<TrainEvent>) -> Result<(), String> {
    let exe = uv_exe(root);
    if exe.is_file() {
        return Ok(());
    }
    let _ = events.send(TrainEvent::Setup { text: format!("Downloading uv {UV_VERSION}") });
    let bytes = fetch_checked(&format!("https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"), UV_SHA256, "uv")?;
    let dir = root.join("uv");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let archive = dir.join("uv.zip");
    std::fs::write(&archive, &bytes).map_err(|e| e.to_string())?;
    let r = untar(&archive, &dir, &[], events);
    let _ = std::fs::remove_file(&archive);
    r?;
    exe.is_file().then_some(()).ok_or_else(|| "uv.exe was not in the archive.".to_owned())
}

impl Trainer {
    pub fn status(&self) -> Value {
        let root = root().ok();
        let inst = root.as_deref().and_then(installed);
        json!({
            "installed": inst,
            "settingUp": self.setting_up.load(Ordering::SeqCst),
            "running": self.running.lock().unwrap_or_else(|e| e.into_inner()).is_some(),
            "root": root.map(|r| r.to_string_lossy().into_owned()),
            "stderrTail": self.tail.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect::<Vec<_>>(),
        })
    }

    /// Builds the Python environment: uv, Python 3.12, PyTorch (CUDA), Unsloth. Several GB.
    pub fn setup(&self, events: &Channel<TrainEvent>) -> Result<Installed, String> {
        if self.setting_up.swap(true, Ordering::SeqCst) {
            return Err("Setup is already running.".into());
        }
        let r = (|| {
            let root = root()?;
            std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
            ensure_uv(&root, events)?;
            let uv = uv_exe(&root);
            let py = venv_python(&root);
            if !py.is_file() {
                let mut c = Command::new(&uv);
                sidecar_env(&mut c, &root);
                c.args(["venv", "--python", PYTHON]).arg(root.join("venv"));
                run_streamed(c, events)?;
            }
            let _ = events.send(TrainEvent::Setup { text: "Installing PyTorch with CUDA (about 3 GB)".into() });
            let mut c = Command::new(&uv);
            sidecar_env(&mut c, &root);
            c.args(["pip", "install", "--python"]).arg(&py).args(["torch", "torchvision", "--index-url", TORCH_INDEX]);
            run_streamed(c, events)?;
            let _ = events.send(TrainEvent::Setup { text: "Installing Unsloth".into() });
            let mut c = Command::new(&uv);
            sidecar_env(&mut c, &root);
            c.args(["pip", "install", "--python"]).arg(&py).arg("unsloth");
            run_streamed(c, events)?;
            ensure_tools(&root, events)?;
            let mut c = Command::new(&py);
            sidecar_env(&mut c, &root);
            c.args(["-c", "import json,sys,torch,unsloth; print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'unsloth': unsloth.__version__, 'cuda': torch.cuda.is_available()}))"]);
            let out = run_streamed(c, events)?;
            let v: Value = serde_json::from_str(out.lines().last().unwrap_or("{}")).map_err(|_| "Could not read the installed versions.".to_owned())?;
            let inst = Installed {
                uv: UV_VERSION.into(),
                python: v["python"].as_str().unwrap_or("").into(),
                torch: v["torch"].as_str().unwrap_or("").into(),
                unsloth: v["unsloth"].as_str().unwrap_or("").into(),
                cuda: v["cuda"].as_bool().unwrap_or(false),
            };
            std::fs::write(root.join("installed.json"), serde_json::to_vec_pretty(&inst).unwrap_or_default()).map_err(|e| e.to_string())?;
            Ok(inst)
        })();
        self.setting_up.store(false, Ordering::SeqCst);
        r
    }

    /// Runs one job to the end, streaming events. Blocking. The exported GGUF files are copied into
    /// Cerebro's model folder so the Models screen lists them.
    pub fn run(&self, job: &TrainJob, events: &Channel<TrainEvent>) -> Result<(), String> {
        job.check()?;
        let root = root()?;
        if installed(&root).is_none() || !tools_ready(&tools_dir(&root)) {
            return Err("Set up training first.".into());
        }
        let id = format!("{}-{}", job.folder_name(), crate::remote::now_ms());
        let dir = root.join("jobs").join(&id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        {
            let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
            if running.is_some() {
                return Err("A training job is already running.".into());
            }
            *running = Some(dir.clone());
        }
        let r = self.run_in(job, &root, &dir, events);
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = None;
        r
    }

    fn run_in(&self, job: &TrainJob, root: &Path, dir: &Path, events: &Channel<TrainEvent>) -> Result<(), String> {
        use std::os::windows::process::CommandExt;
        let script = dir.join("train.py");
        std::fs::write(&script, TRAIN_PY).map_err(|e| e.to_string())?;
        let spec = json!({
            "model": job.model,
            "dataset": match (&job.dataset_path, &job.dataset_hf) { (Some(p), _) => json!({ "path": p }), (_, Some(h)) => json!({ "hf": h }), _ => Value::Null },
            "format": job.format, "method": job.method, "rank": job.rank, "alpha": job.alpha, "lr": job.lr,
            "epochs": job.epochs, "max_steps": job.max_steps, "max_seq_len": job.max_seq_len, "batch": job.batch,
            "grad_accum": job.grad_accum, "export": job.export,
            "output_dir": dir.join("out"), "stop_file": dir.join("STOP"),
        });
        let job_file = dir.join("job.json");
        std::fs::write(&job_file, serde_json::to_vec_pretty(&spec).unwrap_or_default()).map_err(|e| e.to_string())?;
        self.tail.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let mut cmd = Command::new(venv_python(root));
        sidecar_env(&mut cmd, root);
        let mut child = cmd
            .arg(&script)
            .arg(&job_file)
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("Could not start training: {e}"))?;
        let tail = self.tail.clone();
        let stderr = child.stderr.take();
        let err_thread = std::thread::spawn(move || {
            if let Some(s) = stderr {
                for line in BufReader::new(s).lines().map_while(Result::ok) {
                    let mut t = tail.lock().unwrap_or_else(|e| e.into_inner());
                    t.push_back(line.chars().take(400).collect());
                    while t.len() > STDERR_TAIL {
                        t.pop_front();
                    }
                }
            }
        });
        let mut done: Option<TrainEvent> = None;
        let mut error: Option<String> = None;
        if let Some(out) = child.stdout.take() {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                match parse_event(&line) {
                    Some(TrainEvent::Done { models, adapter, steps, seconds, stopped }) => {
                        done = Some(TrainEvent::Done { models, adapter, steps, seconds, stopped });
                    }
                    Some(TrainEvent::Error { message }) => error = Some(message),
                    Some(ev) => {
                        let _ = events.send(ev);
                    }
                    None => {}
                }
            }
        }
        let status = child.wait().map_err(|e| e.to_string())?;
        let _ = err_thread.join();
        if let Some(message) = error {
            return Err(message);
        }
        let Some(TrainEvent::Done { models, adapter, steps, seconds, stopped }) = done else {
            let last = self.tail.lock().unwrap_or_else(|e| e.into_inner()).back().cloned().unwrap_or_default();
            return Err(format!("Training ended without finishing (exit {}). {last}", status.code().unwrap_or(-1)));
        };
        // Into the model library: %LOCALAPPDATA%\Cerebro\models\trained\<name>\.
        let mut library = Vec::new();
        if !models.is_empty() {
            let dest = std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join("Cerebro").join("models").join("trained").join(job.folder_name())).ok_or("LOCALAPPDATA is not set.")?;
            std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
            for m in &models {
                let src = Path::new(m);
                // Only files the job wrote, inside its own folder.
                if !src.starts_with(dir) || src.extension().and_then(|e| e.to_str()) != Some("gguf") {
                    continue;
                }
                let target = dest.join(src.file_name().unwrap_or_default());
                // Moved, not copied: a model file is gigabytes. Copy only across volumes.
                if std::fs::rename(src, &target).is_err() {
                    std::fs::copy(src, &target).map_err(|e| format!("Could not add the model to the library: {e}"))?;
                    let _ = std::fs::remove_file(src);
                }
                library.push(target.to_string_lossy().into_owned());
            }
        }
        let _ = events.send(TrainEvent::Done { models: library, adapter, steps, seconds, stopped });
        Ok(())
    }

    /// Asks the running job to stop at its next step (it still saves what it learned).
    pub fn cancel(&self) -> bool {
        let running = self.running.lock().unwrap_or_else(|e| e.into_inner());
        running.as_ref().is_some_and(|dir| std::fs::File::create(dir.join("STOP")).and_then(|mut f| f.write_all(b"stop")).is_ok())
    }
}

#[tauri::command(async)]
pub fn trainer_status(trainer: tauri::State<'_, Trainer>) -> Value {
    trainer.status()
}

#[tauri::command(async)]
pub fn trainer_setup(trainer: tauri::State<'_, Trainer>, on_event: Channel<TrainEvent>) -> Result<Installed, String> {
    trainer.setup(&on_event)
}

#[tauri::command(async)]
pub fn trainer_start(trainer: tauri::State<'_, Trainer>, job: TrainJob, on_event: Channel<TrainEvent>) -> Result<(), String> {
    trainer.run(&job, &on_event)
}

#[tauri::command(async)]
pub fn trainer_cancel(trainer: tauri::State<'_, Trainer>) -> bool {
    trainer.cancel()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job() -> TrainJob {
        let ds = std::env::temp_dir().join("cerebro-train-test.jsonl");
        std::fs::write(&ds, "{\"instruction\":\"hi\",\"output\":\"hello\"}\n").unwrap();
        TrainJob {
            name: "My model 1".into(),
            model: "unsloth/Qwen3-0.6B".into(),
            dataset_path: Some(ds.to_string_lossy().into_owned()),
            dataset_hf: None,
            format: "auto".into(),
            method: "qlora".into(),
            rank: 16,
            alpha: 16,
            lr: 2e-4,
            epochs: 1.0,
            max_steps: 30,
            max_seq_len: 2048,
            batch: 2,
            grad_accum: 4,
            export: vec!["q4_k_m".into()],
        }
    }

    #[test]
    fn job_validation() {
        assert!(job().check().is_ok());
        let bad = |f: &dyn Fn(&mut TrainJob)| {
            let mut j = job();
            f(&mut j);
            j.check().is_err()
        };
        assert!(bad(&|j| j.model = "../evil".into()));
        assert!(bad(&|j| j.model = "a/b/c".into()));
        assert!(bad(&|j| j.name = "x\\..\\y".into()));
        assert!(bad(&|j| j.dataset_path = Some("C:\\nope\\missing.jsonl".into())));
        assert!(bad(&|j| j.dataset_hf = Some("tatsu-lab/alpaca".into())), "both dataset kinds at once");
        assert!(bad(&|j| j.export = vec!["q2_k; rm -rf".into()]));
        assert!(bad(&|j| j.lr = 1.0));
        assert!(bad(&|j| j.rank = 0));
        let mut hf = job();
        hf.dataset_path = None;
        hf.dataset_hf = Some("tatsu-lab/alpaca".into());
        assert!(hf.check().is_ok());
    }

    /// A real fine-tune through the installed environment (run with --ignored after setup): five
    /// steps of Qwen3-0.6B, exported to Q4_K_M and moved into the model library.
    #[test]
    #[ignore = "needs the training environment, a CUDA GPU and the network"]
    fn real_five_step_run_lands_a_gguf_in_the_library() {
        let dir = std::env::temp_dir().join("cerebro-train-itest");
        std::fs::create_dir_all(&dir).unwrap();
        let ds = dir.join("data.jsonl");
        let rows: String = (1..=32).map(|i| format!("{{\"instruction\":\"Code for {i}?\",\"output\":\"CEREBRO-{}\"}}\n", i * 7)).collect();
        std::fs::write(&ds, rows).unwrap();
        let mut j = job();
        j.name = "itest model".into();
        j.dataset_path = Some(ds.to_string_lossy().into_owned());
        j.method = "lora".into();
        j.max_steps = 5;
        j.batch = 4;
        j.grad_accum = 1;
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let s2 = seen.clone();
        let ch: Channel<TrainEvent> = Channel::new(move |body| {
            if let tauri::ipc::InvokeResponseBody::Json(t) = body {
                s2.lock().unwrap().push(t);
            }
            Ok(())
        });
        let t = Trainer::default();
        let inst = t.setup(&ch).unwrap_or_else(|e| panic!("setup: {e}"));
        assert!(inst.cuda && !inst.unsloth.is_empty(), "{:?}", serde_json::to_string(&inst));
        seen.lock().unwrap().clear();
        t.run(&j, &ch).unwrap_or_else(|e| panic!("{e}\n{:?}", t.status()["stderrTail"]));
        let seen = seen.lock().unwrap();
        assert!(seen.iter().filter(|e| e.contains("\"type\":\"log\"")).count() >= 5, "{seen:?}");
        let done: Value = serde_json::from_str(seen.last().unwrap()).unwrap();
        assert_eq!(done["type"], "done");
        let model = PathBuf::from(done["models"][0].as_str().unwrap());
        assert!(model.is_file() && model.to_string_lossy().contains("trained"), "{model:?}");
        assert!(std::fs::metadata(&model).unwrap().len() > 100 << 20);
        let _ = std::fs::remove_dir_all(model.parent().unwrap());
    }

    #[test]
    fn events_reach_the_ui_in_camel_case() {
        let v = serde_json::to_value(TrainEvent::Log { step: 2, max_steps: 9, loss: Some(1.5), eval_loss: None, lr: None, grad_norm: Some(0.5), epoch: None }).unwrap();
        assert_eq!(v["type"], "log");
        assert_eq!(v["maxSteps"], 9);
        assert_eq!(v["gradNorm"], 0.5);
        let v = serde_json::to_value(TrainEvent::Setup { text: "x".into() }).unwrap();
        assert_eq!(v, serde_json::json!({ "type": "setup", "text": "x" }));
    }

    #[test]
    fn event_lines_are_parsed_strictly() {
        assert!(matches!(parse_event(r#"{"type":"log","step":3,"max_steps":30,"loss":1.25,"lr":0.0002}"#), Some(TrainEvent::Log { step: 3, loss: Some(l), .. }) if l == 1.25));
        assert!(matches!(parse_event(r#"{"type":"status","stage":"training","detail":"10 examples"}"#), Some(TrainEvent::Status { .. })));
        assert!(matches!(parse_event(r#"{"type":"done","gguf":["a.gguf"],"steps":30,"seconds":12.5}"#), Some(TrainEvent::Done { steps: 30, .. })));
        assert!(parse_event("Unsloth: loading model...").is_none());
        assert!(parse_event(r#"{"type":"run","cmd":"evil"}"#).is_none());
        assert!(parse_event(r#"{"type":"log"}"#).is_none(), "a log line needs a step");
        assert!(matches!(parse_event(r#"{"type":"log","step":1,"loss":NaN}"#), None | Some(TrainEvent::Log { loss: None, .. })));
    }
}
