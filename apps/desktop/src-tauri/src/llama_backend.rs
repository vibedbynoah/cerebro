//! llama.cpp as a second backend, for the placements where it is measured faster than Cerebro's
//! own engine (today: a model split between the graphics card and the processor). The app runs
//! the official llama.cpp release's `llama-server` on 127.0.0.1 with the same layer split the
//! planner chose, and streams replies over its OpenAI-compatible API with the same client as the
//! cloud models (cloud.rs). Nothing listens beyond this computer.
//!
//! The release is not shipped in the installer (the CUDA build is about 650 MB): the first split
//! that needs it downloads the official release files from GitHub, checks each against the
//! SHA-256 GitHub publishes for it, and unpacks them into Cerebro's own folder. A folder that
//! already holds the release can be named with CEREBRO_LLAMA_DIR. Offline mode blocks the
//! download; the model then runs on Cerebro's engine.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::engine::{ChatEvent, ChatRequest, ChatSink, ChatStats};

/// The llama.cpp release Cerebro uses, and its Windows builds: (file, SHA-256, bytes).
pub const TAG: &str = "b10751";
const CUDA: &[(&str, &str, u64)] = &[
    ("llama-b10751-bin-win-cuda-12.4-x64.zip", "0d735a42f54627dcdd595da787d5fae068bd7f4b9867d14f6dd37dcb2ebd9077", 254_269_280),
    ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391_443_627),
];
const VULKAN: &[(&str, &str, u64)] = &[("llama-b10751-bin-win-vulkan-x64.zip", "535abda5e9c40280ec88aa3b414e93639cb0275391651ce7c5bc4db86b6e1455", 35_183_789)];

/// Which build: CUDA for NVIDIA cards, Vulkan for every other graphics card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    /// NVIDIA cards (CUDA 12.4 build plus its runtime).
    Cuda,
    /// Every other graphics card.
    Vulkan,
}

impl Build {
    fn files(self) -> &'static [(&'static str, &'static str, u64)] {
        match self {
            Build::Cuda => CUDA,
            Build::Vulkan => VULKAN,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Build::Cuda => "cuda-12.4",
            Build::Vulkan => "vulkan",
        }
    }

    /// The size of the download, all files together.
    pub fn download_bytes(self) -> u64 {
        self.files().iter().map(|f| f.2).sum()
    }
}

/// Architectures this llama.cpp release runs (`general.architecture`).
pub fn runs(arch: &str) -> bool {
    matches!(arch, "qwen3" | "qwen3moe" | "qwen2" | "llama" | "qwen35" | "qwen35moe" | "gemma3" | "gemma4" | "mistral3" | "phi3" | "gpt-oss")
}

fn folder(build: Build) -> Option<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")?;
    Some(PathBuf::from(local).join("Cerebro").join("llama.cpp").join(format!("{TAG}-{}", build.name())))
}

/// Offline mode, as the app's setting says (`network_offline_set`): no download starts while it is on.
static OFFLINE: AtomicBool = AtomicBool::new(false);

/// Follows the app's Offline mode setting.
pub fn set_offline(on: bool) {
    OFFLINE.store(on, Ordering::SeqCst);
}

/// Whether Offline mode is on.
pub fn offline() -> bool {
    OFFLINE.load(Ordering::SeqCst)
}

/// The folder holding `llama-server.exe`, when it is there.
pub fn installed(build: Build) -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("CEREBRO_LLAMA_DIR").map(PathBuf::from).filter(|d| d.join("llama-server.exe").is_file()) {
        return Some(d);
    }
    folder(build).filter(|d| d.join("llama-server.exe").is_file())
}

/// Downloads and unpacks the release (see the module notes). `progress(done, total)` as it goes.
/// Refused in Offline mode; a copy already installed is used either way.
pub fn install(build: Build, progress: impl FnMut(u64, u64)) -> Result<PathBuf, String> {
    if let Some(d) = installed(build) {
        return Ok(d);
    }
    if offline() {
        return Err("Offline mode is on, so llama.cpp was not downloaded.".to_owned());
    }
    let dir = folder(build).ok_or("There is no local app data folder.")?;
    install_from(&format!("https://github.com/ggml-org/llama.cpp/releases/download/{TAG}"), build.files(), &dir, progress)?;
    installed(build).ok_or_else(|| "The llama.cpp files did not contain llama-server.exe.".to_owned())
}

/// Downloads `files` (name, SHA-256, bytes) from `base` into `dir`, checks each and unpacks it.
/// A file that does not match its checksum is deleted and nothing of it is unpacked.
fn install_from(base: &str, files: &[(&str, &str, u64)], dir: &Path, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    let total: u64 = files.iter().map(|f| f.2).sum();
    let client = reqwest::blocking::Client::builder().timeout(None).build().map_err(|e| e.to_string())?;
    let mut done = 0u64;
    for &(name, sha, size) in files {
        let url = format!("{base}/{name}");
        let part = dir.join(format!("{name}.part"));
        let mut r = client.get(&url).send().map_err(|e| format!("Could not download {name}: {e}"))?;
        if !r.status().is_success() {
            return Err(format!("Could not download {name}: HTTP {}", r.status()));
        }
        let mut f = std::fs::File::create(&part).map_err(|e| e.to_string())?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = r.read(&mut buf).map_err(|e| format!("The download of {name} broke off: {e}"))?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
            f.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            got += n as u64;
            progress(done + got, total);
        }
        drop(f);
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if hex != sha || got != size {
            let _ = std::fs::remove_file(&part);
            return Err(format!("{name} did not match the checksum llama.cpp published; nothing was installed."));
        }
        // Windows' own tar reads zip files.
        let mut tar = Command::new(r"C:\Windows\System32\tar.exe");
        tar.args(["-xf"]).arg(&part).arg("-C").arg(dir).stdout(Stdio::null()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            tar.creation_flags(0x0800_0000);
        }
        let out = tar.output().map_err(|e| format!("Could not unpack {name}: {e}"))?;
        let _ = std::fs::remove_file(&part);
        if !out.status.success() {
            return Err(format!("Could not unpack {name}: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        done += size;
    }
    Ok(())
}

/// A running `llama-server` with one model loaded.
pub struct Server {
    child: Child,
    /// The local port it listens on.
    pub port: u16,
    /// Layers on the graphics card.
    pub gpu_layers: usize,
    log: std::sync::Arc<std::sync::Mutex<String>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    /// Starts `llama-server` on a free local port with `gpu_layers` layers on the graphics card and
    /// waits until the model is loaded.
    pub fn start(dir: &Path, model: &Path, gpu_layers: usize, ctx: usize) -> Result<Server, String> {
        let port = std::net::TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map(|a| a.port()).map_err(|e| e.to_string())?;
        let mut cmd = Command::new(dir.join("llama-server.exe"));
        cmd.arg("-m").arg(model);
        cmd.args(["-ngl", &gpu_layers.to_string(), "-c", &ctx.to_string(), "-fa", "on", "--host", "127.0.0.1", "--port", &port.to_string(), "--no-webui", "--jinja"]);
        cmd.stdout(Stdio::null()).stderr(Stdio::piped()).stdin(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        let mut child = cmd.spawn().map_err(|e| format!("Could not start llama.cpp: {e}"))?;
        // Keep the last lines of its log for error messages.
        let log = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        if let Some(err) = child.stderr.take() {
            let l = log.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    let mut s = l.lock().unwrap_or_else(|e| e.into_inner());
                    s.push_str(&line);
                    s.push('\n');
                    if s.len() > 16_000 {
                        let cut = s.len() - 8_000;
                        s.drain(..cut);
                    }
                }
            });
        }
        let mut server = Server { child, port, gpu_layers, log };
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        loop {
            if let Ok(Some(status)) = server.child.try_wait() {
                return Err(format!("llama.cpp stopped while loading the model ({status}): {}", server.log_tail()));
            }
            if client.get(format!("http://127.0.0.1:{port}/health")).send().is_ok_and(|r| r.status().is_success()) {
                return Ok(server);
            }
            if t0.elapsed() > Duration::from_secs(900) {
                return Err(format!("llama.cpp did not finish loading in 15 minutes: {}", server.log_tail()));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn log_tail(&self) -> String {
        let s = self.log.lock().unwrap_or_else(|e| e.into_inner());
        s.lines().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | ")
    }

    /// Its OpenAI-compatible address.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }
}

/// A reply from the model llama.cpp is running, streamed as the same events a Cerebro reply sends.
pub fn reply(server: &Server, req: &ChatRequest, events: &ChatSink) {
    let t0 = Instant::now();
    let turns: Vec<crate::cloud::Turn<'_>> = req
        .turns
        .iter()
        .map(|t| crate::cloud::Turn { role: if t.role == "assistant" { "assistant" } else { "user" }, text: t.text.as_str(), images: t.images().collect() })
        .collect();
    let settings = crate::cloud::Settings { temperature: req.sampler.temperature, max_tokens: req.sampler.max_tokens.max(1) };
    let mut started: Option<f64> = None;
    let r = crate::cloud::chat_at(&server.base_url(), None, "llama.cpp", "local", req.system_prompt.trim(), &turns, &settings, |piece| {
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
                stats: ChatStats {
                    prompt_tokens: 0,
                    length: false,
                    output_tokens: o.output_tokens,
                    thinking_tokens: o.thinking_pieces,
                    tok_per_sec: (o.output_tokens + o.thinking_pieces) as f64 / gen_s,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Serves `body` for every request on a free local port; returns the base address.
    fn serve(body: Vec<u8>) -> String {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { break };
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf);
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&body);
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    /// A zip holding one file `llama-server.exe`, made with Windows' tar, and its SHA-256.
    fn test_zip(dir: &Path) -> (Vec<u8>, String) {
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("llama-server.exe"), b"not a real program").unwrap();
        let zip = dir.join("t.zip");
        let ok = Command::new(r"C:\Windows\System32\tar.exe").arg("-a").arg("-cf").arg(&zip).arg("-C").arg(&src).arg("llama-server.exe").status().unwrap().success();
        assert!(ok, "tar could not make the test zip");
        let bytes = std::fs::read(&zip).unwrap();
        let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        (bytes, hex)
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cerebro-llama-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[cfg(windows)]
    #[test]
    fn a_checked_download_is_unpacked_with_progress() {
        let work = temp("ok");
        let (zip, sha) = test_zip(&work);
        let base = serve(zip.clone());
        let dest = work.join("dest");
        let mut last = (0, 0);
        install_from(&base, &[("t.zip", &sha, zip.len() as u64)], &dest, |d, t| last = (d, t)).unwrap();
        assert!(dest.join("llama-server.exe").is_file());
        assert!(!dest.join("t.zip.part").exists());
        assert_eq!(last, (zip.len() as u64, zip.len() as u64));
        let _ = std::fs::remove_dir_all(&work);
    }

    #[cfg(windows)]
    #[test]
    fn a_checksum_mismatch_installs_nothing() {
        let work = temp("bad");
        let (zip, _) = test_zip(&work);
        let base = serve(zip.clone());
        let dest = work.join("dest");
        let wrong = "0".repeat(64);
        let e = install_from(&base, &[("t.zip", &wrong, zip.len() as u64)], &dest, |_, _| {}).unwrap_err();
        assert!(e.contains("checksum"), "{e}");
        assert!(!dest.join("llama-server.exe").exists());
        assert!(!dest.join("t.zip.part").exists());
        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn offline_mode_refuses_a_download() {
        set_offline(true);
        // CEREBRO_LLAMA_DIR could point at an installed copy; this test needs none.
        if installed(Build::Vulkan).is_none() {
            let e = install(Build::Vulkan, |_, _| {}).unwrap_err();
            assert!(e.contains("Offline"), "{e}");
        }
        set_offline(false);
    }

    #[test]
    fn builds_list_the_published_files_and_sizes() {
        assert_eq!(Build::Cuda.files().len(), 2);
        assert!(Build::Cuda.download_bytes() > 600_000_000);
        assert!(Build::Vulkan.files().iter().all(|f| f.1.len() == 64 && f.0.contains(TAG)));
        assert!(runs("qwen3") && runs("qwen35moe") && !runs("qwen4exp"));
    }
}
