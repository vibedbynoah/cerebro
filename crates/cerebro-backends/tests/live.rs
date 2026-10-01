//! Machine-dependent end-to-end tests: the live hardware probe through the matrix and selection,
//! and the CUDA 12 backend's own argv and health check against the official build with the tiny
//! model. `#[ignore]`d; run with `cargo test -p cerebro-backends -- --ignored`.
//!
//! The llama.cpp fixtures come from `CEREBRO_LLAMACPP_OFFICIAL` and `CEREBRO_TINY_GGUF`. The
//! built-in default for the first (`C:/bench/llamacpp-official`) was emptied on 2026-09-24; on the
//! development machine set `CEREBRO_LLAMACPP_OFFICIAL=C:\llama-cuda` (an official b10603 CUDA build).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use cerebro_backends::{
    BackendId, BackendRegistry, CapabilityMatrix, Feature, LaunchRequest, LlamaCppBuild,
    Location, ModelFormat, SelectionPolicy, Verification, VERIFIED_FEATURES,
};
use cerebro_flags::value::FlashAttn;
use cerebro_flags::Flag;

/// The live development machine selects the CUDA 12 build first, keeps the CPU baseline, and its
/// matrix agrees with the captured reference profile on every feature.
#[test]
#[ignore = "requires the development machine's hardware (RTX 3070 Ti Laptop + Iris Xe)"]
fn live_profile_selects_cuda_and_matches_the_captured_reference() {
    let live = cerebro_probe::hardware::probe();
    let reference = cerebro_probe::hardware::profiles::rtx3070ti_laptop().unwrap();
    let registry = BackendRegistry::standard();

    let sel = registry.select(&live, ModelFormat::Gguf, &SelectionPolicy::default());
    assert_eq!(sel.first(), Some(BackendId::LlamaCpp(LlamaCppBuild::Cuda12)), "{sel:?}");
    assert!(sel.ids().contains(&BackendId::LlamaCpp(LlamaCppBuild::Cpu)));

    let none = BTreeMap::new();
    let live_m = CapabilityMatrix::evaluate(&live, &registry, &none);
    let ref_m = CapabilityMatrix::evaluate(&reference, &registry, &none);
    for f in Feature::ALL {
        let (l, r) = (live_m.get(*f).unwrap(), ref_m.get(*f).unwrap());
        assert_eq!(
            l.availability.is_available(),
            r.availability.is_available(),
            "{f}: live {:?} vs reference {:?}",
            l.summary(),
            r.summary()
        );
        assert_eq!(l.verified_via, r.verified_via, "{f}");
    }
}

/// Kills and reaps the server on every exit path, panics included, and removes its scratch folder.
struct Server {
    child: Child,
    port: u16,
    key: &'static str,
    scratch: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.scratch);
    }
}

/// Launch the official CUDA 12 build with this crate's argv, the tiny model, `extra` flags and its
/// output written to `<scratch>/server.log`, and wait (90 s at most) until its health check says
/// ready.
fn launch_cuda12(tag: &str, extra: Vec<Flag>) -> Server {
    let dir = cerebro_probe::fixtures::official_backend_dir().unwrap();
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap();
    let registry = BackendRegistry::standard();
    let backend = registry.get(BackendId::LlamaCpp(LlamaCppBuild::Cuda12)).unwrap();
    assert_eq!(backend.verification(), Verification::Verified);
    backend.check_install(&dir).unwrap();

    let scratch = std::env::temp_dir().join(format!("cerebro-backends-live-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let key = "cerebro-live-test-key";
    let key_file = scratch.join("key.txt");
    std::fs::write(&key_file, format!("{key}\n")).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();

    let mut request = LaunchRequest::new(Location::Native(dir), model, port);
    request.api_key_file = Some(key_file);
    request.context = Some(512);
    request.llama_flags = vec![Flag::GpuLayers(99)];
    request.llama_flags.extend(extra);
    let cmd = backend.build_argv(&request).unwrap();

    // Both streams into one log: llama.cpp's logger writes to either depending on the build.
    let log = std::fs::File::create(scratch.join("server.log")).unwrap();
    let log_out = log.try_clone().unwrap();
    let mut command = Command::new(&cmd.program);
    command.args(&cmd.args).stdin(Stdio::null()).stdout(Stdio::from(log_out)).stderr(Stdio::from(log));
    // Test harness only, after Cerebro's argv: debug verbosity, so the log states which attention
    // path the context was built with (this build prints it only at `-lv 4`).
    command.args(["-lv", "4"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut server = Server { child: command.spawn().unwrap(), port, key, scratch };

    let health = backend.health_check();
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut last = None;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child.try_wait() {
            panic!("llama-server exited early with {status}: {}", server_log(&server));
        }
        last = http(port, "GET", health.path, None, "").map(|(s, _)| s);
        if last == Some(health.ready_status) {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    assert_eq!(last, Some(health.ready_status), "never became ready: {}", server_log(&server));
    server
}

fn server_log(server: &Server) -> String {
    // Lossy: the log carries token text from the model, which need not be UTF-8.
    std::fs::read(server.scratch.join("server.log"))
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

/// One HTTP/1.1 request; returns the status and the whole response (the server closes after it).
fn http(port: u16, method: &str, path: &str, bearer: Option<&str>, body: &str) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok()?;
    let auth = bearer.map_or_else(String::new, |k| format!("Authorization: Bearer {k}\r\n"));
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n{auth}\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, text))
}

/// The CUDA 12 backend's argv starts the official build with the tiny model, and its health check
/// reports ready: what `Verification::Verified` for this build rests on.
#[test]
#[ignore = "requires the official llama.cpp CUDA 12 build (CEREBRO_LLAMACPP_OFFICIAL, e.g. C:\\llama-cuda; the default is empty) and the tiny GGUF (CEREBRO_TINY_GGUF)"]
fn live_cuda12_argv_launches_and_passes_its_health_check() {
    let server = launch_cuda12("health", Vec::new());
    // The completion API is behind the key file the argv named: without the key, 401.
    assert_eq!(http(server.port, "GET", "/v1/completions", None, "").map(|(s, _)| s), Some(401));
}

/// The row `(llama.cpp CUDA 12, llama.cpp flash attention)` in `VERIFIED_FEATURES` rests on this:
/// `-fa on` through the allowlist, the server log reports flash attention enabled, and an
/// authenticated completion generates tokens.
#[test]
#[ignore = "requires the official llama.cpp CUDA 12 build (CEREBRO_LLAMACPP_OFFICIAL, e.g. C:\\llama-cuda; the default is empty) and the tiny GGUF (CEREBRO_TINY_GGUF)"]
fn live_cuda12_flash_attention_serves_a_completion() {
    let cuda12 = BackendId::LlamaCpp(LlamaCppBuild::Cuda12);
    assert!(VERIFIED_FEATURES.contains(&(cuda12, Feature::LlamaFlashAttention)));
    let server = launch_cuda12("fa", vec![Flag::FlashAttn(FlashAttn::On)]);

    let body = r#"{"prompt":"The capital of France is","n_predict":8,"temperature":0}"#;
    let (status, response) =
        http(server.port, "POST", "/completion", Some(server.key), body).expect("completion request");
    assert_eq!(status, 200, "{response}");
    let predicted = response
        .split("\"tokens_predicted\":")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|n| n.parse::<u32>().ok());
    assert!(predicted.is_some_and(|n| n > 0), "{response}");

    let log = server_log(&server);
    let fa_line = log
        .lines()
        .find(|l| l.contains("flash_attn") && l.contains('='))
        .unwrap_or_else(|| panic!("no flash_attn line in the server log:\n{log}"));
    println!("server log: {fa_line}");
    assert!(
        (fa_line.contains("enabled") || fa_line.trim_end().ends_with("= 1")) && !fa_line.contains("disabled"),
        "{fa_line}"
    );
}
