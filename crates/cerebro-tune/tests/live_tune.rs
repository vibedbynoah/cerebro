//! Live tests of `cerebro_tune::tune`: the real, preflighted llama-bench on the tiny model, with
//! this machine's probed CPU and GPU, persisted to a real store.
//!
//! `#[ignore]`d: they need an official llama.cpp build (`CEREBRO_LLAMACPP_OFFICIAL`, default
//! `C:/bench/llamacpp-official`) and the tiny GGUF (`CEREBRO_TINY_GGUF`). Run explicitly, one at a
//! time so the runs do not measure each other:
//! `cargo test -p cerebro-tune --test live_tune -- --ignored --test-threads=1`.
//!
//! As in `cerebro-bench`'s live test, the backend manifest is pinned from the directory's own
//! hashes, the folder-permission audit is switched off (a build directory directly under a drive
//! root typically inherits `Authenticated Users: Modify` from it), and elevation refusal is
//! switched off only when the test itself runs elevated.
//!
//! If another llama.cpp process is running (or the GPU is busy) the tune still completes, but the
//! contention check sets those measurements aside and the outcome carries its warning; the test
//! prints it.

#![cfg(windows)]

use std::io::Read as _;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use cerebro_bench::run::{BenchRunner, DEFAULT_TOOL_NAME};
use cerebro_store::{SamplerProfile, Store};
use cerebro_supervisor::{BenchLaunchSpec, Protections, SpawnOptions};
use cerebro_tune::{
    tune, CancelToken, LlamaBench, MachineFacts, ModelFacts, ScreenSettings, TuneEvent,
    TuneRequest, Verdict, Workload,
};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cerebro-tune-live-{tag}-{}", std::process::id()));
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

    fn store(&self) -> Store {
        Store::new(self.0.join("results.jsonl"))
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_no_llama_bench_left() {
    let mut child = Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq llama-bench.exe", "/NH"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .expect("run tasklist");
    let mut out = String::new();
    child.stdout.take().expect("piped").read_to_string(&mut out).expect("read tasklist");
    child.wait().expect("reap tasklist");
    assert!(!out.to_ascii_lowercase().contains("llama-bench.exe"), "a llama-bench.exe remains: {out}");
}

fn launch(backend_dir: &Path, model: &Path) -> BenchLaunchSpec {
    let (manifest, _) = cerebro_probe::backend::hash_directory(backend_dir).expect("hash the backend");
    let elevated = cerebro_probe::token::is_elevated().expect("query elevation");
    BenchLaunchSpec {
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
    }
}

/// Small enough that six candidates finish in well under a minute on the tiny model.
fn screen() -> ScreenSettings {
    ScreenSettings {
        kept_repetitions: 3,
        prompt_tokens: 64,
        gen_tokens: 16,
        depth_override: Some(256),
        timeout: Duration::from_secs(300),
        delay_secs: 0,
        allow_contended: false,
        recheck_baseline: true,
    }
}

fn fixtures() -> (PathBuf, PathBuf) {
    let backend = cerebro_probe::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
    let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
    (backend, model)
}

#[test]
#[ignore = "runs the official llama-bench (CEREBRO_LLAMACPP_OFFICIAL) on the tiny GGUF \
            (CEREBRO_TINY_GGUF) for every candidate; about 30-90 s"]
fn a_real_tune_measures_persists_and_never_recommends_anything_slower_than_the_baseline() {
    let (backend, model_path) = fixtures();
    let model = ModelFacts::from_gguf(&model_path).expect("read the tiny model's header");
    let file_bytes = std::fs::metadata(&model_path).expect("stat the model").len();
    assert!(model.block_count > 0 && model.blocks.len() == model.block_count as usize, "{model:?}");
    assert!(!model.is_moe());
    let weights = model.full_offload_bytes() + model.input_bytes;
    assert!(weights <= file_bytes && weights > file_bytes / 2, "{weights} of {file_bytes} bytes");

    let scratch = ScratchDir::new("ok");
    let runner = BenchRunner::new().expect("probe the machine");
    let bench = LlamaBench::new(&runner, launch(&backend, &model_path), scratch.spawn_options());
    let sampler = SamplerProfile::new(0.8, 40, 0.95, 0.05).expect("llama.cpp's sampler defaults");
    let request = TuneRequest {
        workload: Workload::Chat,
        model,
        machine: MachineFacts::probe(),
        screen: screen(),
        sampler,
    };
    let mut events = Vec::new();
    let outcome = tune(&request, &bench, &scratch.store(), &CancelToken::new(), &mut |e| {
        events.push(e.clone())
    })
    .unwrap_or_else(|e| panic!("tune failed: {e}"));
    assert_no_llama_bench_left();

    for f in &outcome.failed {
        eprintln!("candidate {} ({}) failed: {}", f.index, f.config, f.reason);
    }
    for m in &outcome.measured {
        let t = &m.measurement.triple;
        eprintln!(
            "candidate {} ({}): pp {:.0}, tg@0 {:.1}, tg@{} {:.1} [{:.1}, {:.1}] tok/s, {:?} threads",
            m.index,
            m.config,
            t.pp_tok_s.median,
            t.tg_tok_s_depth0.median,
            t.deepest_depth,
            t.tg_tok_s_deepest.median,
            t.tg_tok_s_deepest.ci95_low,
            t.tg_tok_s_deepest.ci95_high,
            m.measurement.threads
        );
    }
    if let Some(r) = &outcome.baseline_recheck {
        let t = &r.measurement.triple;
        eprintln!("baseline re-measured: pp {:.0}, tg@{} {:.1} tok/s", t.pp_tok_s.median, t.deepest_depth, t.tg_tok_s_deepest.median);
    }
    for m in outcome.measured.iter().chain(&outcome.baseline_recheck) {
        if let Some(c) = m.contention {
            eprintln!("candidate {} contended: {}", m.index, c.describe());
        }
    }
    eprintln!("excluded: {:?}", outcome.excluded);
    eprintln!("warnings: {:?}", outcome.warnings);
    eprintln!("verdict: {:?}, speedup {:?}", outcome.verdict, outcome.speedup());

    let baseline = outcome.baseline().expect("the baseline is measured first");
    assert_eq!(baseline.index, 0);
    assert_eq!(outcome.depth, 256);
    assert!(outcome.measured.len() >= 3, "baseline plus at least the KV and thread variants");
    let best = outcome.recommended().expect("a completed tune recommends something");
    let scored = |m: &cerebro_tune::MeasuredCandidate| cerebro_tune::Scored {
        tg_at_depth: m.measurement.triple.tg_tok_s_deepest,
        pp: m.measurement.triple.pp_tok_s,
        tg_depth0: m.measurement.triple.tg_tok_s_depth0,
        threads: m.measurement.threads,
    };
    match outcome.verdict {
        Verdict::Improved { .. } => {
            assert!(
                cerebro_tune::eligible(&scored(best), &scored(baseline)),
                "an improvement must be demonstrably faster than the baseline and not slower at \
                 prefill or depth-0 decode"
            );
            assert!(best.contention.is_none() && baseline.contention.is_none());
            let again = outcome.baseline_recheck.as_ref().expect("a winner is re-checked");
            assert!(cerebro_tune::eligible(&scored(best), &scored(again)));
        }
        Verdict::BaselineWins => assert_eq!(best.index, 0),
        Verdict::Cancelled => panic!("not cancelled"),
    }

    let rechecked = usize::from(outcome.baseline_recheck.is_some());
    let records = scratch.store().load_verified().expect("load the store").records;
    assert_eq!(records.len(), outcome.measured.len() + rechecked, "every measurement is persisted");
    assert!(records.iter().all(|r| *r.sampler() == sampler));
    assert!(records.iter().all(|r| r.measurement().deepest_depth().get() == 256));
    let started = events.iter().filter(|e| matches!(e, TuneEvent::CandidateStarted { .. })).count();
    assert_eq!(started, outcome.measured.len() + outcome.failed.len() + rechecked);
    let finished = events.iter().filter(|e| matches!(e, TuneEvent::CandidateFinished { .. })).count();
    assert_eq!(finished, started, "every started candidate finishes");
    assert!(matches!(events.last(), Some(TuneEvent::Winner { .. })));
}

#[test]
#[ignore = "runs the official llama-bench (CEREBRO_LLAMACPP_OFFICIAL) on the tiny GGUF \
            (CEREBRO_TINY_GGUF) and cancels it mid-run; about 5-15 s"]
fn cancelling_a_real_tune_kills_the_running_llama_bench() {
    let (backend, model_path) = fixtures();
    let scratch = ScratchDir::new("cancel");
    let runner = BenchRunner::new().expect("probe the machine");
    let bench = LlamaBench::new(&runner, launch(&backend, &model_path), scratch.spawn_options());
    // A run long enough to be cancelled in the middle of: many repetitions at a deeper depth.
    let mut settings = screen();
    settings.kept_repetitions = 200;
    settings.depth_override = Some(2048);
    let request = TuneRequest {
        workload: Workload::Chat,
        model: ModelFacts::from_gguf(&model_path).expect("read the tiny model's header"),
        machine: MachineFacts::probe(),
        screen: settings,
        sampler: SamplerProfile::new(0.8, 40, 0.95, 0.05).expect("sampler"),
    };
    let token = CancelToken::new();
    let canceller = token.clone();
    let mut events = Vec::new();
    let outcome = std::thread::scope(|s| {
        s.spawn(move || {
            std::thread::sleep(Duration::from_secs(3));
            canceller.cancel();
        });
        tune(&request, &bench, &scratch.store(), &token, &mut |e| events.push(e.clone()))
    })
    .unwrap_or_else(|e| panic!("tune failed: {e}"));
    assert_eq!(outcome.verdict, Verdict::Cancelled);
    assert!(outcome.recommended().is_none());
    assert!(outcome.measured.is_empty(), "the baseline run was still going");
    assert!(matches!(events.last(), Some(TuneEvent::Cancelled { measured: 0 })));
    assert_no_llama_bench_left();
}
