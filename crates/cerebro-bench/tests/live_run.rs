//! Live test of `cerebro_bench::run`: the real, preflighted, job-contained `llama-bench` on the
//! tiny model, CPU only (`-dev none`), three kept repetitions.
//!
//! `#[ignore]`d: it needs the official llama.cpp build (`CEREBRO_LLAMACPP_OFFICIAL`, default
//! `C:/bench/llamacpp-official`; on the development machine now `C:\llama-cuda`) and the tiny GGUF
//! (`CEREBRO_TINY_GGUF`). Run explicitly:
//! `cargo test -p cerebro-bench --test live_run -- --ignored --test-threads=1` (one at a time, so
//! the runs do not measure each other).
//!
//! The backend manifest is pinned from the directory's own hashes at test start (the test proves
//! the run, not the build's provenance). The folder-permission audit is switched off, as a user
//! would: the official build at `C:\llama-cuda` inherits `Authenticated Users: Modify` from `C:\`
//! and fails it. Elevation refusal is switched off only when the test itself runs elevated. Both
//! must then appear once in the run's preflight warnings.

#![cfg(windows)]

use std::io::Read as _;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use cerebro_bench::run::{
    BenchRequest, BenchRun, BenchRunner, RunError, RunPlan, DEFAULT_TOOL_NAME,
    SERVER_DEFAULT_FIT_TARGET_MIB,
};
use cerebro_flags::value::DeviceList;
use cerebro_flags::Flag;
use cerebro_supervisor::{
    BenchLaunchSpec, PreflightWarning, Protection, Protections, SpawnOptions,
};
use sha2::{Digest, Sha256};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A scratch directory unique to this test process, removed on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cerebro-bench-live-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        for sub in ["tmp", "cache", "log"] {
            std::fs::create_dir_all(dir.join(sub)).expect("create scratch dirs");
        }
        Self(dir)
    }

    fn spawn_options(&self) -> SpawnOptions {
        SpawnOptions {
            temp_dir: self.0.join("tmp"),
            cache_dir: self.0.join("cache"),
            cuda_visible_devices: None,
            log_path: self.0.join("log").join("llama-bench.log"),
            log_max_bytes: 4 * 1024 * 1024,
        }
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `tasklist` output for `filter`, windowless.
fn tasklist(filter: &str) -> String {
    let mut child = Command::new("tasklist")
        .args(["/FI", filter, "/NH"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .expect("run tasklist");
    let mut out = String::new();
    child
        .stdout
        .take()
        .expect("piped stdout")
        .read_to_string(&mut out)
        .expect("read tasklist output");
    child.wait().expect("reap tasklist");
    out
}

fn assert_no_llama_bench_left(pid: Option<u32>) {
    if let Some(pid) = pid {
        let by_pid = tasklist(&format!("PID eq {pid}"));
        assert!(
            !by_pid.to_ascii_lowercase().contains("llama-bench"),
            "llama-bench pid {pid} is still running: {by_pid}"
        );
    }
    let by_name = tasklist("IMAGENAME eq llama-bench.exe");
    assert!(
        !by_name.to_ascii_lowercase().contains("llama-bench.exe"),
        "a llama-bench.exe process remains: {by_name}"
    );
}

fn request(backend_dir: &Path, model: &Path, scratch: &ScratchDir, plan: RunPlan) -> BenchRequest {
    let (manifest, _) =
        cerebro_probe::backend::hash_directory(backend_dir).expect("hash the backend directory");
    let elevated = cerebro_probe::token::is_elevated().expect("query elevation");
    BenchRequest {
        launch: BenchLaunchSpec {
            backend_dir: backend_dir.to_path_buf(),
            tool_name: DEFAULT_TOOL_NAME.to_string(),
            manifest,
            model_path: model.to_path_buf(),
            allow_unsafe_dacl: None,
            protections: Protections {
                audit_folder_permissions: false,
                refuse_elevation: !elevated,
                ..Protections::default()
            },
        },
        spawn: scratch.spawn_options(),
        // Four threads: on this hybrid laptop CPU more threads measured no faster and much hotter.
        config: vec![Flag::Device(DeviceList::None), Flag::Threads(4)],
        plan,
    }
}

#[test]
#[ignore = "runs the official llama-bench (CEREBRO_LLAMACPP_OFFICIAL) on the tiny GGUF \
            (CEREBRO_TINY_GGUF), CPU only; about 10-20 s"]
fn a_real_cpu_run_returns_a_complete_keyed_triple_and_leaves_no_process() {
    let backend_dir =
        cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
    let scratch = ScratchDir::new("ok");
    let plan = RunPlan::new(3, 32, 8, [0, 128])
        .expect("valid plan")
        .with_timeout(Duration::from_secs(300))
        .expect("valid timeout");
    let req = request(&backend_dir, &model, &scratch, plan);
    let runner = BenchRunner::new().expect("probe CPU and GPU");

    let run = runner.run(&req).unwrap_or_else(|e| panic!("live run failed: {e}"));
    assert_no_llama_bench_left(Some(run.pid));

    // The argv is exactly the Bench-dialect protocol, with the preflighted model path.
    let model_arg = run.argv.get(1).cloned().expect("-m value");
    assert_eq!(
        run.argv,
        [
            "-m", &model_arg, "-dev", "none", "-t", "4", "-p", "32", "-n", "8", "-d", "0,128",
            "-r", "4", "-o", "json"
        ]
    );
    assert!(Path::new(&model_arg).is_absolute());

    // A complete pp / tg / tg-at-depth triple over three kept repetitions.
    assert_eq!(run.records.len(), 4);
    assert_eq!(run.triple.deepest_depth.get(), 128);
    for summary in [run.triple.pp_tok_s, run.triple.tg_tok_s_depth0, run.triple.tg_tok_s_deepest] {
        assert_eq!(summary.kept, 3);
        assert!(summary.min > 0.0 && summary.min <= summary.median && summary.median <= summary.max);
        assert!(summary.ci95_low <= summary.ci95_high);
    }
    assert_eq!(run.config.devices(), Some("none"));

    // Provenance: the model digest is the file's real SHA-256, the backend digest covers verified
    // hashes, and the key converts to the store's.
    let bytes = std::fs::read(&model).expect("read the tiny model");
    let expected: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    let p = &run.provenance;
    assert_eq!(p.model_sha256, expected);
    assert_eq!(p.model_size_bytes, bytes.len() as u64);
    assert!(p.backend_integrity_verified);
    assert_eq!(p.backend_manifest_digest.len(), 64);
    assert!(p.backend_build_number > 0 && !p.backend_commit.is_empty());
    assert_eq!(p.free_vram_mib_at_start.len(), p.gpus.len());
    assert!(p.power.is_some(), "GetSystemPowerStatus is always available on Windows");
    assert!(p.cpu.logical_processors >= p.cpu.physical_cores && p.cpu.physical_cores > 0);
    assert!(p.cpu.brand.as_deref().is_some_and(|b| !b.is_empty()), "{:?}", p.cpu.brand);
    // The official Windows build ships one ggml-cpu DLL per CPU generation and logs which it chose;
    // the build's own `--version` names its compiler.
    assert!(p.backend_cpu_variant.as_deref().is_some_and(|v| !v.is_empty()), "no ggml-cpu variant");
    assert!(p.backend_compiler.as_deref().is_some_and(|c| !c.is_empty()), "no compiler");
    assert_eq!(p.gpu_utilization_pct_at_start.len(), p.gpus.len());
    assert!(p.other_llama_processes_at_start.is_some(), "the process snapshot must work");
    if let Some(others) = p.other_llama_processes_at_start.as_ref().filter(|o| !o.is_empty()) {
        eprintln!("live run: contended by {others:?}; throughput ratios are not meaningful");
    }
    let key = p.store_key().expect("a well-formed store key");
    assert_eq!(key.backend_cpu_variant(), p.backend_cpu_variant.as_deref());
    assert_eq!(key.backend_compiler(), p.backend_compiler.as_deref());

    // Every protection turned off is recorded once, and nothing else.
    let mut disabled: Vec<Protection> = run
        .preflight_warnings
        .iter()
        .filter_map(|w| match w {
            PreflightWarning::ProtectionDisabled(p) => Some(p.protection()),
            _ => None,
        })
        .collect();
    disabled.sort_by_key(|p| p.key());
    let mut expected_off = vec![Protection::FolderPermissionAudit];
    if !req.launch.protections.refuse_elevation {
        expected_off.push(Protection::ElevationRefusal);
    }
    expected_off.sort_by_key(|p| p.key());
    assert_eq!(disabled, expected_off);
    assert!(run.spawn_warnings.is_empty(), "{:?}", run.spawn_warnings);

    // The digest is cached for the next candidate configuration of this model.
    assert_eq!(runner.model_hashes().len(), 1);
    let started = std::time::Instant::now();
    let (cached, _) = runner.model_hashes().sha256(Path::new(&model_arg)).expect("cached digest");
    assert_eq!(cached, expected);
    assert_eq!(runner.model_hashes().len(), 1, "a cache hit adds no entry");
    let cached_lookup = started.elapsed();

    // Integrity verification off: the run hashes the backend directory itself, which must key the
    // same binary preflight verified, and records the switched-off protection exactly once.
    let mut off = req.clone();
    off.launch.protections.verify_backend_integrity = false;
    let run_off = runner.run(&off).unwrap_or_else(|e| panic!("integrity-off run failed: {e}"));
    assert_no_llama_bench_left(Some(run_off.pid));
    assert_eq!(
        run_off.provenance.backend_manifest_digest, p.backend_manifest_digest,
        "hashing the directory ourselves must key the same binary preflight verified"
    );
    assert!(!run_off.provenance.backend_integrity_verified);
    assert_eq!(disabled_count(&run_off, Protection::BackendIntegrityVerification), 1);
    assert_eq!(disabled_count(&run, Protection::BackendIntegrityVerification), 0);
    assert_eq!(run_off.provenance.backend_compiler, p.backend_compiler, "cached per build");
    eprintln!(
        "live run: wall {:?}; pp {:.1} tg {:.1} tg@128 {:.1} tok/s; cached digest lookup {:?}",
        run.wall_time,
        run.triple.pp_tok_s.median,
        run.triple.tg_tok_s_depth0.median,
        run.triple.tg_tok_s_deepest.median,
        cached_lookup
    );
}

fn disabled_count(run: &BenchRun, protection: Protection) -> usize {
    run.preflight_warnings
        .iter()
        .filter(|w| matches!(w, PreflightWarning::ProtectionDisabled(p) if p.protection() == protection))
        .count()
}

#[test]
#[ignore = "runs the official llama-bench (CEREBRO_LLAMACPP_OFFICIAL) on the tiny GGUF \
            (CEREBRO_TINY_GGUF) as the llama-server-default baseline, on the GPU if present"]
fn the_server_default_baseline_runs_with_llama_servers_fit_margin() {
    let backend_dir =
        cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
    let scratch = ScratchDir::new("baseline");
    let plan = RunPlan::new(3, 32, 8, [0, 128])
        .expect("valid plan")
        .with_timeout(Duration::from_secs(300))
        .expect("valid timeout");
    let template = request(&backend_dir, &model, &scratch, plan);
    let req = BenchRequest::server_default_baseline(template.launch, template.spawn, template.plan);
    let runner = BenchRunner::new().expect("probe CPU and GPU");

    let run = runner.run(&req).unwrap_or_else(|e| panic!("baseline run failed: {e}"));
    assert_no_llama_bench_left(Some(run.pid));
    let model_arg = run.argv.get(1).cloned().expect("-m value");
    assert_eq!(
        run.argv,
        [
            "-m", &model_arg, "-fitt", "1024", "-p", "32", "-n", "8", "-d", "0,128", "-r", "4",
            "-o", "json"
        ]
    );
    // llama-bench itself confirms it fitted with llama-server's margin, not its own default (off).
    let fit_target = run
        .config
        .extra()
        .iter()
        .find(|(k, _)| k == "fit_target")
        .map(|(_, v)| v.clone());
    assert_eq!(fit_target, Some(SERVER_DEFAULT_FIT_TARGET_MIB.to_string()));
    assert_eq!(run.records.len(), 4);
    eprintln!(
        "baseline run: wall {:?}; ngl {:?}; pp {:.1} tg {:.1} tg@128 {:.1} tok/s",
        run.wall_time,
        run.config.n_gpu_layers(),
        run.triple.pp_tok_s.median,
        run.triple.tg_tok_s_depth0.median,
        run.triple.tg_tok_s_deepest.median
    );
}

#[test]
#[ignore = "runs the official llama-bench (CEREBRO_LLAMACPP_OFFICIAL) on the tiny GGUF \
            (CEREBRO_TINY_GGUF) and kills it at a 300 ms timeout"]
fn a_real_run_past_its_timeout_is_killed_with_its_tree() {
    let backend_dir =
        cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
    let scratch = ScratchDir::new("timeout");
    // Far more work than 300 ms allows: backend load alone takes longer.
    let plan = RunPlan::new(20, 512, 256, [0, 4096])
        .expect("valid plan")
        .with_timeout(Duration::from_millis(300))
        .expect("valid timeout");
    let req = request(&backend_dir, &model, &scratch, plan);
    let runner = BenchRunner::new().expect("probe CPU and GPU");

    match runner.run(&req) {
        Err(RunError::TimedOut {
            timeout,
            confirmed_gone,
            ..
        }) => {
            assert_eq!(timeout, Duration::from_millis(300));
            assert!(confirmed_gone, "the OS must confirm the killed process gone");
        }
        other => panic!("expected TimedOut, got {other:?}"),
    }
    assert_no_llama_bench_left(None);
}
