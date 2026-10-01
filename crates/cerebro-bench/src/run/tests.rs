//! Failure classification with a fake spawner, and output interpretation against recorded
//! llama-bench JSON. The live counterpart is `tests/live_run.rs`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::*;

/// Real `llama-bench -o json` output (b10751, CPU, `-p 64 -n 32 -d 0,128 -r 4`).
const FIXTURE: &str = include_str!("../../tests/fixtures/cpu_tiny_model.json");
/// The `-m` value that fixture was produced with.
const FIXTURE_MODEL: &str = "models/bastion-control-0.5b-f16.gguf";

fn fixture_plan() -> RunPlan {
    RunPlan::new(3, 64, 32, [0, 128]).expect("the plan the fixture was produced with")
}

/// How the fake child behaves.
struct Script {
    /// `None`: never exits on its own.
    exit: Option<Option<i32>>,
    /// Polls answered "still running" before `exit` is reported.
    running_polls: u32,
    stdout: Result<CapturedStdout, SpawnError>,
    lines: Vec<String>,
    output_closes: bool,
    /// The post-kill poll fails instead of confirming the exit.
    kill_fails: bool,
}

impl Script {
    fn exits(code: i32, stdout: &str) -> Self {
        Self {
            exit: Some(Some(code)),
            running_polls: 2,
            stdout: Ok(CapturedStdout {
                bytes: stdout.as_bytes().to_vec(),
                overflowed: false,
                read_error: None,
            }),
            lines: vec![
                "stderr: load_backend: loaded CPU backend".to_string(),
                "stderr: main: done".to_string(),
            ],
            output_closes: true,
            kill_fails: false,
        }
    }
}

/// What the test can inspect after `execute` dropped the fake.
#[derive(Default)]
struct Observed {
    killed: AtomicBool,
    dropped: AtomicBool,
    polls: AtomicU32,
}

struct FakeProcess {
    script: Mutex<Script>,
    observed: Arc<Observed>,
}

impl Drop for FakeProcess {
    fn drop(&mut self) {
        self.observed.dropped.store(true, Ordering::SeqCst);
    }
}

impl Process for FakeProcess {
    fn pid(&self) -> u32 {
        4242
    }
    fn try_wait(&self) -> Result<Option<Option<i32>>, SpawnError> {
        let polls = self.observed.polls.fetch_add(1, Ordering::SeqCst);
        let script = self.script.lock().unwrap();
        if self.observed.killed.load(Ordering::SeqCst) {
            return Ok(Some(Some(1)));
        }
        Ok(script.exit.filter(|_| polls >= script.running_polls))
    }
    fn kill_and_wait(&self, _timeout: Duration) -> Result<bool, SpawnError> {
        self.observed.killed.store(true, Ordering::SeqCst);
        if self.script.lock().unwrap().kill_fails {
            return Err(SpawnError::Wait {
                cause: "access denied".to_string(),
            });
        }
        Ok(true)
    }
    fn wait_output_closed(&self, _timeout: Duration) -> bool {
        self.script.lock().unwrap().output_closes
    }
    fn take_stdout(&self, _timeout: Duration) -> Result<CapturedStdout, SpawnError> {
        std::mem::replace(
            &mut self.script.lock().unwrap().stdout,
            Err(SpawnError::StdoutCaptureAlreadyTaken),
        )
    }
    fn recent_lines(&self) -> Vec<String> {
        self.script.lock().unwrap().lines.clone()
    }
    fn protection_warnings(&self) -> Vec<ProtectionWarning> {
        Vec::new()
    }
}

/// Run `execute` against a fake following `script`; the result and what the fake observed.
fn run_fake(script: Script, plan: &RunPlan) -> (Result<Finished, RunError>, Arc<Observed>) {
    let observed = Arc::new(Observed::default());
    let cell = Mutex::new(Some(script));
    let obs = Arc::clone(&observed);
    let spawn = move || -> Result<Box<dyn Process>, SpawnError> {
        let script = cell.lock().unwrap().take().expect("spawned once");
        Ok(Box::new(FakeProcess {
            script: Mutex::new(script),
            observed: Arc::clone(&obs),
        }))
    };
    let result = execute(&spawn, plan);
    (result, observed)
}

fn tail() -> Vec<String> {
    vec![
        "load_backend: loaded CPU backend".to_string(),
        "main: done".to_string(),
    ]
}

#[test]
fn a_clean_exit_with_the_fixture_document_yields_the_full_triple() {
    let (result, observed) = run_fake(Script::exits(0, FIXTURE), &fixture_plan());
    let finished = result.expect("exit 0 with a document");
    assert_eq!(finished.pid, 4242);
    assert_eq!(finished.stderr_tail, tail());
    assert!(observed.dropped.load(Ordering::SeqCst), "the child is released before returning");
    assert!(!observed.killed.load(Ordering::SeqCst));
    assert!(observed.polls.load(Ordering::SeqCst) >= 3, "it waited for the exit");

    let (config, triple, records) =
        interpret(&finished.stdout, &finished.stderr_tail, &fixture_plan(), FIXTURE_MODEL)
            .expect("the fixture is exactly what the plan asks for");
    assert_eq!(records.len(), 4);
    assert_eq!(config.devices(), Some("none"));
    assert_eq!(triple.deepest_depth.get(), 128);
    for summary in [triple.pp_tok_s, triple.tg_tok_s_depth0, triple.tg_tok_s_deepest] {
        assert_eq!(summary.kept, 3, "-r 4 keeps 3 after discarding the first");
    }
    // prefill@0 samples 42.1838, [41.2307, 38.2606, 42.3633]: the discarded first rep must not
    // be the median.
    assert_eq!(triple.pp_tok_s.median, 41.2307);
}

#[test]
fn a_non_zero_exit_is_an_infrastructure_error_with_only_stderr_in_its_tail() {
    let mut script = Script::exits(1, "");
    script.lines = vec![
        "stderr: llama_model_load: error loading model".to_string(),
        "stdout: this is not stderr".to_string(),
        "cerebro: crash log write failed, in-memory only from here: disk full".to_string(),
        "stderr: main: error: failed to load model".to_string(),
    ];
    let (result, _) = run_fake(script, &fixture_plan());
    let err = result.expect_err("exit 1 is never a result");
    assert!(err.is_infrastructure());
    match &err {
        RunError::BackendExited { code, stderr_tail } => {
            assert_eq!(*code, Some(1));
            assert_eq!(
                stderr_tail,
                &[
                    "llama_model_load: error loading model",
                    "cerebro: crash log write failed, in-memory only from here: disk full",
                    "main: error: failed to load model",
                ]
            );
        }
        other => panic!("expected BackendExited, got {other:?}"),
    }
    assert!(err.to_string().contains("failed to load model"), "{err}");
}

#[test]
fn an_exit_without_a_code_is_still_a_failure() {
    let mut script = Script::exits(0, FIXTURE);
    script.exit = Some(None);
    let (result, _) = run_fake(script, &fixture_plan());
    assert!(matches!(result, Err(RunError::BackendExited { code: None, .. })));
}

#[test]
fn a_clean_exit_with_no_output_is_missing_json_not_an_empty_result() {
    for stdout in ["", "  \r\n\t"] {
        let (result, _) = run_fake(Script::exits(0, stdout), &fixture_plan());
        match result {
            Err(RunError::MissingJson { stderr_tail }) => assert_eq!(stderr_tail, tail()),
            other => panic!("{stdout:?}: expected MissingJson, got {other:?}"),
        }
    }
    // An empty array parses, but is no more a result than no output.
    let err = interpret("[]", &tail(), &fixture_plan(), FIXTURE_MODEL).expect_err("no records");
    assert!(matches!(err, RunError::MissingJson { .. }), "{err:?}");
}

#[test]
fn a_run_past_its_timeout_is_killed_and_reported() {
    let mut script = Script::exits(0, FIXTURE);
    script.exit = None;
    let plan = fixture_plan().with_timeout(Duration::from_millis(50)).unwrap();
    let started = Instant::now();
    let (result, observed) = run_fake(script, &plan);
    assert!(started.elapsed() >= Duration::from_millis(50));
    assert!(observed.killed.load(Ordering::SeqCst), "the tree must be killed");
    assert!(observed.dropped.load(Ordering::SeqCst));
    match result {
        Err(RunError::TimedOut {
            timeout,
            confirmed_gone,
            stderr_tail,
        }) => {
            assert_eq!(timeout, Duration::from_millis(50));
            assert!(confirmed_gone);
            assert_eq!(stderr_tail, tail());
        }
        other => panic!("expected TimedOut, got {other:?}"),
    }
}

#[test]
fn a_kill_that_cannot_be_confirmed_is_still_a_timeout_and_says_why() {
    let mut script = Script::exits(0, FIXTURE);
    script.exit = None;
    script.kill_fails = true;
    let plan = fixture_plan().with_timeout(Duration::from_millis(20)).unwrap();
    let (result, observed) = run_fake(script, &plan);
    assert!(observed.killed.load(Ordering::SeqCst));
    match result {
        Err(RunError::TimedOut {
            confirmed_gone,
            stderr_tail,
            ..
        }) => {
            assert!(!confirmed_gone);
            assert_eq!(
                stderr_tail.last().map(String::as_str),
                Some("cerebro: could not confirm the kill: could not poll child exit status: access denied")
            );
        }
        other => panic!("expected TimedOut, got {other:?}"),
    }
}

#[test]
fn unreadable_truncated_or_undrained_output_is_never_parsed() {
    let mut overflow = Script::exits(0, "[{");
    overflow.stdout = Ok(CapturedStdout {
        bytes: b"[{".to_vec(),
        overflowed: true,
        read_error: None,
    });
    let (result, _) = run_fake(overflow, &fixture_plan());
    assert!(matches!(result, Err(RunError::OutputTooLarge { .. })), "{result:?}");

    let mut broken = Script::exits(0, FIXTURE);
    broken.stdout = Ok(CapturedStdout {
        bytes: FIXTURE.as_bytes().to_vec(),
        overflowed: false,
        read_error: Some("pipe broke".to_string()),
    });
    let (result, _) = run_fake(broken, &fixture_plan());
    assert!(
        matches!(&result, Err(RunError::OutputUnreadable { cause, .. }) if cause == "pipe broke"),
        "{result:?}"
    );

    let mut open = Script::exits(0, FIXTURE);
    open.output_closes = false;
    let (result, _) = run_fake(open, &fixture_plan());
    assert!(matches!(result, Err(RunError::OutputUnreadable { .. })), "{result:?}");

    let mut binary = Script::exits(0, "");
    binary.stdout = Ok(CapturedStdout {
        bytes: vec![0xff, 0xfe, b'['],
        overflowed: false,
        read_error: None,
    });
    let (result, _) = run_fake(binary, &fixture_plan());
    assert!(matches!(result, Err(RunError::OutputUnreadable { .. })), "{result:?}");

    let mut taken = Script::exits(0, FIXTURE);
    taken.stdout = Err(SpawnError::StdoutNotCaptured);
    let (result, _) = run_fake(taken, &fixture_plan());
    assert!(matches!(result, Err(RunError::OutputUnreadable { .. })), "{result:?}");
}

#[test]
fn a_failed_spawn_is_reported_as_such() {
    let spawn = || -> Result<Box<dyn Process>, SpawnError> {
        Err(SpawnError::Spawn {
            program: "llama-bench.exe".to_string(),
            cause: "access denied".to_string(),
        })
    };
    let err = execute(&spawn, &fixture_plan()).expect_err("no process, no result");
    assert!(matches!(err, RunError::Spawn(_)));
    assert!(err.is_infrastructure());
}

/// The fixture document with `edit` applied to its records.
fn edited(edit: impl FnOnce(&mut Vec<Value>)) -> String {
    let mut records: Vec<Value> = serde_json::from_str(FIXTURE).unwrap();
    edit(&mut records);
    serde_json::to_string(&records).unwrap()
}

fn unexpected(doc: &str, plan: &RunPlan) -> String {
    match interpret(doc, &tail(), plan, FIXTURE_MODEL) {
        Err(RunError::UnexpectedRecords { detail, stderr_tail }) => {
            assert_eq!(stderr_tail, tail());
            detail
        }
        other => panic!("expected UnexpectedRecords, got {other:?}"),
    }
}

#[test]
fn records_that_do_not_match_the_request_are_refused() {
    // Asked for 4 kept (5 run); the document has 4 run.
    let plan = RunPlan::new(4, 64, 32, [0, 128]).unwrap();
    assert!(unexpected(FIXTURE, &plan).contains("4 repetitions, the run asked for 5"));

    // A document from some other run's model.
    let err = interpret(FIXTURE, &tail(), &fixture_plan(), r"C:\models\other.gguf")
        .expect_err("wrong model");
    assert!(err.to_string().contains("not the model this run passed"), "{err}");

    // The depth-128 pair is missing.
    let doc = edited(|r| r.truncate(2));
    assert!(unexpected(&doc, &fixture_plan()).contains("2 records, expected 4"));

    // Asked for depth 256; the document stops at 128.
    let plan = RunPlan::new(3, 64, 32, [0, 256]).unwrap();
    assert!(unexpected(FIXTURE, &plan).contains("(p=64, n=0, d=128), which this run did not ask for"));

    // Right count, right deepest depth, wrong prefill size: not the run that was asked for.
    let doc = edited(|r| r[0]["n_prompt"] = Value::from(512));
    assert_eq!(
        unexpected(&doc, &fixture_plan()),
        "record 0 is test (p=512, n=0, d=0), which this run did not ask for"
    );
    let doc = edited(|r| r[1]["n_gen"] = Value::from(16));
    assert!(unexpected(&doc, &fixture_plan()).contains("(p=0, n=16, d=0)"));

    // Depths [0, 128, 512] asked; [0, 300, 512] returned: same count, same deepest depth.
    let plan = RunPlan::new(3, 64, 32, [0, 128, 512]).unwrap();
    let doc = edited(|r| {
        let mut deep = r[2..4].to_vec();
        for rec in &mut deep {
            rec["n_depth"] = Value::from(512);
        }
        for rec in &mut r[2..4] {
            rec["n_depth"] = Value::from(300);
        }
        r.extend(deep);
    });
    assert_eq!(
        unexpected(&doc, &plan),
        "record 2 is test (p=64, n=0, d=300), which this run did not ask for"
    );

    // The deep decode test relabelled depth 0: the count is right, but decode@0 appears twice.
    let doc = edited(|r| r[3]["n_depth"] = Value::from(0));
    assert_eq!(unexpected(&doc, &fixture_plan()), "record 3 repeats test (p=0, n=32, d=0)");

    // Two configurations (the thread count differs between the records).
    let doc = edited(|r| {
        for rec in r.iter_mut().skip(2) {
            rec["n_threads"] = Value::from(6);
        }
    });
    assert!(unexpected(&doc, &fixture_plan()).contains("2 configurations"));
}

#[test]
fn malformed_json_and_an_incomplete_triple_carry_the_stderr_tail() {
    let err = interpret(&FIXTURE[..FIXTURE.len() / 2], &tail(), &fixture_plan(), FIXTURE_MODEL)
        .expect_err("truncated JSON");
    assert!(matches!(&err, RunError::MalformedJson { .. }), "{err:?}");
    assert_eq!(err.stderr_tail(), tail().as_slice());

    // Every shape is right, but a sample is one no run produces: refused at the parse.
    let doc = edited(|r| r[3]["samples_ts"][2] = Value::from(-1.0));
    let err = interpret(&doc, &tail(), &fixture_plan(), FIXTURE_MODEL).expect_err("bad sample");
    assert!(matches!(&err, RunError::MalformedJson { .. }), "{err:?}");
    assert_eq!(err.stderr_tail(), tail().as_slice());

    // With the exact-shape check ahead of it, an incomplete triple can no longer come out of a
    // document that parses; the variant stays as a backstop, and is still a launched-run failure.
    let err = RunError::IncompleteTriple {
        source: crate::error::TripleError {
            problems: vec![crate::error::TripleProblem::NoDepthSweep],
        },
        stderr_tail: tail(),
    };
    assert!(err.is_infrastructure());
    assert_eq!(err.stderr_tail(), tail().as_slice());
}

#[test]
fn the_stderr_tail_keeps_the_last_lines_without_tags() {
    let lines: Vec<String> = (0..100).map(|i| format!("stderr: line {i}")).collect();
    let tail = stderr_tail(&lines);
    assert_eq!(tail.len(), STDERR_TAIL_LINES);
    assert_eq!(tail.first().map(String::as_str), Some("line 60"));
    assert_eq!(tail.last().map(String::as_str), Some("line 99"));
}

// ---- `BenchRunner::run_with`: the real preflight, hashing and keying around a fake child ----

/// A scratch backend directory holding a fake `llama-bench.exe`, and a GGUF-magic model file.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "cerebro-bench-runwith-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("backend")).unwrap();
        std::fs::write(dir.join("backend").join("llama-bench.exe"), b"fake bench").unwrap();
        std::fs::write(dir.join("backend").join("ggml-cpu-x64.dll"), b"fake cpu").unwrap();
        let mut model = b"GGUF".to_vec();
        model.extend_from_slice(&[7; 64]);
        std::fs::write(dir.join("model.gguf"), model).unwrap();
        Self { dir }
    }

    fn request(&self, verify_backend_integrity: bool) -> BenchRequest {
        let backend_dir = self.dir.join("backend");
        let (manifest, _) = cerebro_probe::backend::hash_directory(&backend_dir).unwrap();
        let elevated = cerebro_probe::token::is_elevated().unwrap();
        BenchRequest {
            launch: BenchLaunchSpec {
                backend_dir,
                tool_name: DEFAULT_TOOL_NAME.to_string(),
                manifest,
                model_path: self.dir.join("model.gguf"),
                allow_unsafe_dacl: None,
                protections: cerebro_supervisor::Protections {
                    // A temp directory's inherited ACL is not what this test is about.
                    audit_folder_permissions: false,
                    refuse_elevation: !elevated,
                    verify_backend_integrity,
                    ..cerebro_supervisor::Protections::default()
                },
            },
            spawn: SpawnOptions {
                temp_dir: self.dir.clone(),
                cache_dir: self.dir.clone(),
                cuda_visible_devices: None,
                log_path: self.dir.join("bench.log"),
                log_max_bytes: 1024,
            },
            config: vec![Flag::Device(cerebro_flags::value::DeviceList::None)],
            plan: fixture_plan(),
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn test_runner() -> BenchRunner {
    BenchRunner::from_summary(
        CpuSummary {
            brand: Some("Test CPU".into()),
            logical_processors: 4,
            physical_cores: 4,
            performance_cores: 4,
            efficiency_cores: 0,
            numa_nodes: 1,
            hybrid: false,
        },
        GpuInventory::default(),
    )
}

/// The fixture document as llama-bench would print it for `argv`'s `-m`.
fn fixture_for(argv: &[String]) -> String {
    let model = argv.get(1).cloned().unwrap_or_default();
    edited(|r| {
        for rec in r.iter_mut() {
            rec["model_filename"] = Value::from(model.clone());
        }
    })
}

/// Counts how many fake children are alive at once.
struct Tracked {
    inner: Box<dyn Process>,
    active: Arc<AtomicU32>,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Process for Tracked {
    fn pid(&self) -> u32 {
        self.inner.pid()
    }
    fn try_wait(&self) -> Result<Option<Option<i32>>, SpawnError> {
        self.inner.try_wait()
    }
    fn kill_and_wait(&self, timeout: Duration) -> Result<bool, SpawnError> {
        self.inner.kill_and_wait(timeout)
    }
    fn wait_output_closed(&self, timeout: Duration) -> bool {
        self.inner.wait_output_closed(timeout)
    }
    fn take_stdout(&self, timeout: Duration) -> Result<CapturedStdout, SpawnError> {
        self.inner.take_stdout(timeout)
    }
    fn recent_lines(&self) -> Vec<String> {
        self.inner.recent_lines()
    }
    fn protection_warnings(&self) -> Vec<ProtectionWarning> {
        self.inner.protection_warnings()
    }
}

fn fake_child(argv: &[String], running_polls: u32) -> Box<dyn Process> {
    let mut script = Script::exits(0, &fixture_for(argv));
    script.running_polls = running_polls;
    script.lines.insert(
        0,
        r"stderr: load_backend: loaded CPU backend from C:\llama\ggml-cpu-alderlake.dll".to_string(),
    );
    Box::new(FakeProcess {
        script: Mutex::new(script),
        observed: Arc::new(Observed::default()),
    })
}

#[cfg(windows)]
#[test]
fn a_run_keys_its_result_and_pins_the_model_while_the_child_runs() {
    let scratch = Scratch::new("keyed");
    let runner = test_runner();
    let write_refused = Mutex::new(None);
    let run = runner
        .run_with(&scratch.request(true), &|pre, argv, _| {
            // While llama-bench would be reading it, nobody may write the hashed model.
            let write = std::fs::OpenOptions::new().write(true).open(pre.model_path());
            *write_refused.lock().unwrap() = Some(write.err().and_then(|e| e.raw_os_error()));
            Ok(fake_child(argv, 2))
        })
        .expect("a clean fake run");
    const ERROR_SHARING_VIOLATION: i32 = 32;
    assert_eq!(*write_refused.lock().unwrap(), Some(Some(ERROR_SHARING_VIOLATION)));
    std::fs::OpenOptions::new()
        .write(true)
        .open(scratch.dir.join("model.gguf"))
        .expect("the pin is released once the run returns");

    let p = &run.provenance;
    let model = std::fs::read(scratch.dir.join("model.gguf")).unwrap();
    assert_eq!(p.model_sha256, provenance_hex(&model));
    assert_eq!(p.model_size_bytes, 68);
    assert!(p.backend_integrity_verified);
    assert_eq!(p.backend_cpu_variant.as_deref(), Some("alderlake"));
    assert_eq!(p.backend_compiler, None, "no llama-server.exe here to ask");
    assert_eq!((p.backend_build_number, p.backend_commit.as_str()), (10751, "3466812d1"));
    assert!(p.free_vram_mib_at_start.is_empty() && p.gpu_utilization_pct_at_start.is_empty());
    assert!(p.other_llama_processes_at_start.is_some(), "the snapshot works on Windows");
    assert_eq!(p.cpu.brand.as_deref(), Some("Test CPU"));
    assert_eq!(p.store_key().unwrap().backend_cpu_variant(), Some("alderlake"));
    assert_eq!(run.argv[2..4], ["-dev", "none"]);
}

#[cfg(windows)]
#[test]
fn hashing_the_backend_ourselves_keys_the_same_binary_preflight_verified() {
    let scratch = Scratch::new("integrity");
    let runner = test_runner();
    let spawn = |_: &BenchPreflighted, argv: &[String], _: &SpawnOptions| Ok(fake_child(argv, 0));
    let on = runner.run_with(&scratch.request(true), &spawn).expect("integrity on");
    let off = runner.run_with(&scratch.request(false), &spawn).expect("integrity off");
    assert_eq!(off.provenance.backend_manifest_digest, on.provenance.backend_manifest_digest);
    assert!(on.provenance.backend_integrity_verified);
    assert!(!off.provenance.backend_integrity_verified);
    let integrity_warnings = |run: &BenchRun| {
        run.preflight_warnings
            .iter()
            .filter(|w| {
                matches!(w, PreflightWarning::ProtectionDisabled(p)
                    if p.protection() == cerebro_supervisor::Protection::BackendIntegrityVerification)
            })
            .count()
    };
    assert_eq!(integrity_warnings(&on), 0);
    assert_eq!(integrity_warnings(&off), 1);
}

#[cfg(windows)]
#[test]
fn two_threads_never_measure_at_the_same_time_on_one_runner() {
    let scratch = Scratch::new("serial");
    let runner = test_runner();
    let request = scratch.request(true);
    let active = Arc::new(AtomicU32::new(0));
    let peak = AtomicU32::new(0);
    let start = std::sync::Barrier::new(2);
    let spawn = |_: &BenchPreflighted, argv: &[String], _: &SpawnOptions| {
        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        // About 100 ms of "running" per child, far longer than the gap between the two starts.
        Ok(Box::new(Tracked {
            inner: fake_child(argv, 10),
            active: Arc::clone(&active),
        }) as Box<dyn Process>)
    };
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                start.wait();
                for _ in 0..2 {
                    runner.run_with(&request, &spawn).expect("fake run");
                }
            });
        }
    });
    assert_eq!(peak.load(Ordering::SeqCst), 1, "two llama-bench runs overlapped");
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[test]
fn the_baseline_request_carries_llama_servers_fit_margin_and_nothing_else() {
    let scratch = Scratch::new("baseline");
    let template = scratch.request(true);
    let baseline =
        BenchRequest::server_default_baseline(template.launch, template.spawn, template.plan);
    assert_eq!(
        baseline.config,
        [Flag::FitTarget(NonZeroU32::new(SERVER_DEFAULT_FIT_TARGET_MIB).unwrap())]
    );
    assert_eq!(SERVER_DEFAULT_FIT_TARGET_MIB, 1024);
}

fn provenance_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}
