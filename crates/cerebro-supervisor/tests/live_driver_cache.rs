//! What pinning the GPU driver caches (`SpawnOptions::cache_dir`) buys on a GPU whose kernels the
//! driver has to compile, measured on a real `llama-server`.
//!
//! The official CUDA build carries native code for this machine's GPU, so nothing is compiled at
//! load time here. `CUDA_FORCE_PTX_JIT=1` makes the driver ignore that native code and JIT the PTX
//! instead, which is what every launch on a GPU newer than the build's targets does. The child's
//! environment is built with [`cerebro_probe::child_env::build_child_env`], the one definition the
//! supervised spawn and the probe queries use, so the blocks measured are the product's own; only
//! `CUDA_FORCE_PTX_JIT` is added on top (it is not an allowlisted name, so the spawn itself would
//! drop it). Nothing here mutates this process's environment.
//!
//! One test in its own binary: it loads the tiny model up to seven times and takes about a minute.

#![cfg(windows)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// `CREATE_NO_WINDOW`: no console window opens for the server.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A scratch directory unique to this test process, removed on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cerebro-supervisor-driver-cache-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Kills and reaps the server on drop, so a failed assertion cannot leave `llama-server` running.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A loopback port the OS just reported free. The gap before the server binds it is harmless here:
/// a lost race fails the launch loudly rather than measuring the wrong thing.
fn free_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .and_then(|l| l.local_addr())
        .expect("bind an ephemeral loopback port")
        .port()
}

/// Whether `GET /health` on `port` answers 200 right now.
fn healthy(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(200)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response.starts_with("HTTP/1.1 200")
}

/// Launch `llama-server` with the tiny model fully offloaded under exactly `env` (or the inherited
/// environment plus `extra` when `env` is `None`), wait for `/health` to answer 200, stop it, and
/// return the load-to-ready time.
fn load_to_ready(
    backend_dir: &Path,
    model: &Path,
    env: Option<&BTreeMap<OsString, OsString>>,
    extra: &[(&str, &str)],
) -> Duration {
    let port = free_port();
    let mut command = Command::new(backend_dir.join("llama-server.exe"));
    command
        .arg("-m")
        .arg(model)
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-ngl",
            "999",
        ])
        .current_dir(backend_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);
    if let Some(env) = env {
        command.env_clear().envs(env);
    }
    command.envs(extra.iter().copied());

    let started = Instant::now();
    let mut server = ServerGuard(command.spawn().expect("spawn llama-server"));
    while !healthy(port) {
        if let Ok(Some(status)) = server.0.try_wait() {
            panic!("llama-server exited before it was ready: {status}");
        }
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "llama-server was not ready within 180 s"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let elapsed = started.elapsed();
    drop(server);
    elapsed
}

/// Total size of the regular files under `dir`, recursively.
fn tree_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(t) if t.is_dir() => tree_bytes(&entry.path()),
            Ok(_) => entry.metadata().map_or(0, |m| m.len()),
            Err(_) => 0,
        })
        .sum()
}

#[test]
#[ignore = "needs an NVIDIA GPU, the official llama.cpp CUDA build (CEREBRO_LLAMACPP_OFFICIAL) and \
            the tiny GGUF (CEREBRO_TINY_GGUF); loads the model seven times with PTX JIT forced, \
            about a minute"]
fn a_pinned_driver_cache_removes_the_per_launch_jit_cost_the_closed_environment_would_add() {
    let backend_dir =
        cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
    let system32 = cerebro_probe::child_env::system_directory().expect("resolve System32");
    let scratch = ScratchDir::new();
    let temp = scratch.0.join("tmp");
    let cache = scratch.0.join("driver-cache");
    std::fs::create_dir_all(&temp).expect("create temp dir");
    std::fs::create_dir_all(&cache).expect("create cache dir");
    let jit = [("CUDA_FORCE_PTX_JIT", "1")];

    let block = |cache_dir: Option<&Path>| {
        cerebro_probe::child_env::build_child_env(
            std::env::vars_os(),
            &system32,
            &backend_dir,
            Some(&temp),
            None,
            cache_dir,
        )
    };
    let dropped = block(None);
    let pinned = block(Some(&cache));

    // The closed block without a cache: the driver has nowhere to keep what it compiled.
    let dropped_1 = load_to_ready(&backend_dir, &model, Some(&dropped), &jit);
    let dropped_2 = load_to_ready(&backend_dir, &model, Some(&dropped), &jit);
    // The closed block with the pinned cache, as every supervised launch now gets it.
    let pinned_cold = load_to_ready(&backend_dir, &model, Some(&pinned), &jit);
    let pinned_warm_1 = load_to_ready(&backend_dir, &model, Some(&pinned), &jit);
    let pinned_warm_2 = load_to_ready(&backend_dir, &model, Some(&pinned), &jit);
    // llama-server run directly, the baseline Cerebro must not be slower than. The first launch
    // warms the user's own ComputeCache in case it has never seen this build's PTX.
    let _ = load_to_ready(&backend_dir, &model, None, &jit);
    let direct_warm = load_to_ready(&backend_dir, &model, None, &jit);

    let cuda_cache_bytes = tree_bytes(&cache.join("cuda"));
    eprintln!(
        "load-to-ready, PTX JIT forced: dropped {dropped_1:.2?} / {dropped_2:.2?}; pinned cold \
         {pinned_cold:.2?}, warm {pinned_warm_1:.2?} / {pinned_warm_2:.2?}; direct warm \
         {direct_warm:.2?}; CUDA cache {cuda_cache_bytes} bytes"
    );

    assert!(
        cuda_cache_bytes > 0,
        "the driver must have written its compiled kernels under the pinned CUDA_CACHE_PATH"
    );
    let pinned_warm = pinned_warm_1.min(pinned_warm_2);
    assert!(
        pinned_warm * 3 < dropped_2,
        "without a cache every launch recompiles, so a warm pinned launch ({pinned_warm:?}) must be \
         far faster than a repeat launch with the caches dropped ({dropped_2:?})"
    );
    assert!(
        pinned_warm <= direct_warm + direct_warm / 2 + Duration::from_millis(500),
        "a warm pinned launch ({pinned_warm:?}) must not be slower than llama-server run directly \
         with a warm cache ({direct_warm:?}), beyond launch-to-launch noise"
    );
}
