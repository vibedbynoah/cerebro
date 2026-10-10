//! Driving `llama-bench` safely and returning parsed, provenance-keyed results.
//!
//! [`BenchRunner::run`] is the one I/O edge of this crate. One runner measures one configuration
//! at a time (two concurrent llama-bench runs would corrupt each other's numbers); for each it:
//!
//! 1. preflights the run with the user's [`cerebro_supervisor::Protections`]
//!    ([`cerebro_supervisor::preflight_bench`]: elevation, backend directory audit and integrity,
//!    model file, and that the tool is `llama-bench.exe`), exactly the checks a served launch gets;
//! 2. builds the argv with the closed flag allowlist in the Bench dialect ([`build_argv`]):
//!    always `-o json`, `-r` kept repetitions + 1 (the first is discarded by [`crate::stats`]),
//!    optional `--delay`, and the depth list;
//! 3. keys the run ([`RunProvenance`]): backend build and commit, backend manifest digest, model
//!    SHA-256 (cached by file version, and the hashed handle held deny-write until llama-bench
//!    exits), CPU brand and topology, GPU models, the power state, other llama.cpp processes, and
//!    free VRAM and GPU utilization immediately before launch;
//! 4. spawns through the supervisor ([`cerebro_supervisor::spawn_bench_preflighted`]): job-contained,
//!    closed environment, no window, logged output, stdout captured whole;
//! 5. waits with a timeout that kills the whole process tree, samples free VRAM again, reads the
//!    selected `ggml-cpu-*` variant from stderr and the compiler from the build's `--version`
//!    (once per build), and turns the output into records, one configuration's [`Triple`], or a
//!    typed [`RunError`].
//!
//! The baseline every tuned configuration must beat is
//! [`BenchRequest::server_default_baseline`]: llama-bench fitting the model the way llama-server
//! does by default. An empty configuration is llama-bench's own defaults, which differ.
//!
//! A non-zero exit, a timeout, missing or malformed JSON, and records that do not match the
//! request are infrastructure errors carrying the tail of llama-bench's stderr
//! ([`RunError::is_infrastructure`]); none of them is ever returned as an empty or partial result.

mod error;
mod plan;
mod provenance;

use std::num::NonZeroU32;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cerebro_flags::Flag;
use cerebro_probe::{CpuTopology, GpuInventory};
use cerebro_supervisor::{
    BenchLaunchSpec, BenchPreflighted, CapturedStdout, ChildHandle, PreflightWarning,
    ProtectionWarning, SpawnError, SpawnOptions,
};

pub use error::{PlanError, ProvenanceError, RunError};
pub use plan::{build_argv, RunPlan, DEFAULT_SEED, DEFAULT_TIMEOUT, MAX_DEPTHS, MAX_TIMEOUT};
pub use provenance::{
    backend_manifest_digest, cpu_variant, gpu_summaries, machine_profile_id, CpuSummary,
    GpuSummary, ModelHashCache, PinnedModel, RunProvenance,
};
use provenance::CompilerCache;

use std::collections::BTreeSet;

use crate::parse::{parse_records, BenchRecord};
use crate::stats::{assemble_triples, ConfigKey, Triple};

/// The benchmark tool's file name in an official llama.cpp build: the only tool a bench preflight
/// admits ([`cerebro_supervisor::BENCH_TOOL_NAME`]).
pub const DEFAULT_TOOL_NAME: &str = cerebro_supervisor::BENCH_TOOL_NAME;

/// llama-server's default fit margin per device, in MiB: `llama-server --help` reads
/// `-fitt, --fit-target ... default: 1024` beside `--fit ... default: 'on'`. llama-bench's own
/// default is fit off, so the baseline states this explicitly. If llama.cpp changes its default,
/// this is the one line to update.
pub const SERVER_DEFAULT_FIT_TARGET_MIB: u32 = 1024;

/// The sibling tool asked for the build's `--version` (llama-bench has none).
const VERSION_TOOL_NAME: &str = "llama-server.exe";

/// Image names whose presence at launch marks a run as probably contended.
const LLAMA_IMAGE_NAMES: &[&str] = &["llama-server.exe", "llama-bench.exe", "llama-cli.exe"];

/// How many of llama-bench's last stderr lines an error carries.
pub const STDERR_TAIL_LINES: usize = 40;

/// How often the child is polled for exit. A run lasts seconds to hours; 10 ms adds at most that
/// much to its wall time and costs nothing measurable.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long a killed child gets to be confirmed gone.
const KILL_GRACE: Duration = Duration::from_secs(10);

/// How long after exit both output pipes get to reach end-of-file. They close with the process;
/// this only bounds a pipe held open by something else.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// One benchmark run: what to launch, how, and what to measure.
#[derive(Debug, Clone)]
pub struct BenchRequest {
    /// Backend directory, tool ([`DEFAULT_TOOL_NAME`]), manifest, model and the user's
    /// protections, preflighted before anything runs.
    pub launch: BenchLaunchSpec,
    /// Temp, driver-cache and log locations for the child.
    pub spawn: SpawnOptions,
    /// The configuration under test, Bench-dialect flags only, without `-m` (the run supplies the
    /// preflighted model path). Empty measures *llama-bench's* defaults, which are not
    /// llama-server's: fit off and `-ngl -1`, so a model that does not fit VRAM fails or spills.
    /// The baseline to beat is [`BenchRequest::server_default_baseline`].
    pub config: Vec<Flag>,
    /// Repetitions, test sizes, depths, delay and timeout.
    pub plan: RunPlan,
}

/// The configuration [`BenchRequest::server_default_baseline`] measures: `-fitt 1024` and nothing
/// else. llama-bench then sizes `-ngl`, the tensor split and the MoE expert placement to the
/// device memory free at load, keeping llama-server's default margin, exactly as an untuned
/// llama-server launch would. (`-fitc` is left out: llama-bench fixes the context to the test size,
/// so the server's 4096 minimum context has nothing to act on.)
pub(crate) fn server_default_baseline_config() -> Vec<Flag> {
    // 1024 is a non-zero literal; the `None` arm cannot be taken and still builds a valid plan.
    let margin = NonZeroU32::new(SERVER_DEFAULT_FIT_TARGET_MIB).unwrap_or(NonZeroU32::MIN);
    vec![Flag::FitTarget(margin)]
}

impl BenchRequest {
    /// The baseline every tuned configuration must beat before Cerebro may recommend it:
    /// llama-bench fitting the model to device memory exactly as llama-server does by default
    /// (`-fitt` [`SERVER_DEFAULT_FIT_TARGET_MIB`]).
    ///
    /// An empty `config` is not this. llama-bench's fit is off by default and its `-ngl -1`
    /// offloads every layer, so a model that does not fit VRAM (the large-MoE case) either fails
    /// to load or spills to system memory and runs far slower than llama-server's automatic
    /// placement: a straw man every tuned configuration would beat.
    ///
    /// Results land under a different [`ConfigKey`] from tuned runs (llama-bench records
    /// `fit_target`), so compare baseline and tuned triples by value, not by key equality.
    pub fn server_default_baseline(
        launch: BenchLaunchSpec,
        spawn: SpawnOptions,
        plan: RunPlan,
    ) -> Self {
        Self {
            launch,
            spawn,
            config: server_default_baseline_config(),
            plan,
        }
    }
}

/// A completed run: one configuration's measurement triple, the records behind it, and
/// everything it is keyed to.
#[derive(Debug, Clone)]
pub struct BenchRun {
    /// What the result is keyed to.
    pub provenance: RunProvenance,
    /// The exact argv llama-bench was given (without the program path).
    pub argv: Vec<String>,
    /// The configuration as llama-bench reported it.
    pub config: ConfigKey,
    /// Prefill@0, decode@0 and decode at the deepest depth, each summarized over the kept
    /// repetitions.
    pub triple: Triple,
    /// Every record llama-bench returned, one per (test, depth).
    pub records: Vec<BenchRecord>,
    /// Preflight's warnings: every protection turned off for this run, recorded once.
    pub preflight_warnings: Vec<PreflightWarning>,
    /// The spawn's own protection warnings (environment allowlist, job containment).
    pub spawn_warnings: Vec<ProtectionWarning>,
    /// The llama-bench process id, for correlating with system logs.
    pub pid: u32,
    /// Launch to exit.
    pub wall_time: Duration,
}

/// Launches one preflighted llama-bench: the supervisor's spawn in production, a fake in tests.
type Spawner<'a> =
    dyn Fn(&BenchPreflighted, &[String], &SpawnOptions) -> Result<Box<dyn Process>, SpawnError> + 'a;

/// Runs benchmarks on this machine, holding what does not change between runs: CPU brand and
/// topology, the GPU inventory, the model digest cache and the per-build compiler cache. It
/// measures one configuration at a time.
#[derive(Debug)]
pub struct BenchRunner {
    cpu: CpuSummary,
    gpus: GpuInventory,
    gpu_summaries: Vec<GpuSummary>,
    machine_profile_id: String,
    model_hashes: ModelHashCache,
    compilers: CompilerCache,
    /// One measurement at a time: two concurrent llama-bench runs contend for the same cores,
    /// memory bandwidth and VRAM, corrupting both results and both free-VRAM keys.
    run_lock: Mutex<()>,
}

impl BenchRunner {
    /// Probe CPU brand and topology and the GPU inventory once.
    pub fn new() -> Result<Self, RunError> {
        let cpu = cerebro_probe::cpu::probe().map_err(ProvenanceError::Cpu)?;
        let gpus = cerebro_probe::gpu::probe().map_err(ProvenanceError::Gpu)?;
        Ok(Self::from_parts(&cpu, cerebro_probe::cpu::brand_string(), gpus))
    }

    /// A runner from facts the caller already probed (`cpu_brand` from
    /// `cerebro_probe::cpu::brand_string`).
    pub fn from_parts(cpu: &CpuTopology, cpu_brand: Option<String>, gpus: GpuInventory) -> Self {
        Self::from_summary(CpuSummary::new(cpu, cpu_brand), gpus)
    }

    fn from_summary(cpu: CpuSummary, gpus: GpuInventory) -> Self {
        let gpu_summaries = gpu_summaries(&gpus);
        let machine_profile_id = machine_profile_id(&cpu, &gpu_summaries);
        Self {
            cpu,
            gpus,
            gpu_summaries,
            machine_profile_id,
            model_hashes: ModelHashCache::new(),
            compilers: CompilerCache::default(),
            run_lock: Mutex::new(()),
        }
    }

    /// The model digest cache this runner fills. Usable from any thread while a run is in progress.
    pub fn model_hashes(&self) -> &ModelHashCache {
        &self.model_hashes
    }

    /// Run one benchmark; see this module's docs for the sequence. A call made while another
    /// thread's run on this runner is in progress waits for it to finish first.
    pub fn run(&self, request: &BenchRequest) -> Result<BenchRun, RunError> {
        self.run_with(request, &|preflighted, argv, options| {
            let child = cerebro_supervisor::spawn_bench_preflighted(preflighted, argv, options)?;
            Ok(Box::new(child) as Box<dyn Process>)
        })
    }

    /// Held from preflight to the last sample of a run.
    fn lock_measurement(&self) -> MutexGuard<'_, ()> {
        // The guarded value is `()`: a panic in another run leaves nothing inconsistent behind.
        self.run_lock.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn run_with(&self, request: &BenchRequest, spawner: &Spawner<'_>) -> Result<BenchRun, RunError> {
        let _one_at_a_time = self.lock_measurement();
        let preflighted = cerebro_supervisor::preflight_bench(&request.launch)?;
        let argv = build_argv(preflighted.model_path(), &request.config, &request.plan)?;

        // Held until llama-bench has exited: nothing can rewrite or replace the hashed file.
        let model = self.model_hashes.sha256_pinned(preflighted.model_path())?;
        let integrity_verified = !preflighted.backend_hashes().is_empty();
        let backend_digest = if integrity_verified {
            backend_manifest_digest(preflighted.backend_hashes())
        } else {
            // Integrity verification is off, so preflight hashed nothing; the key still needs the
            // binary's identity, so hash what is on disk (0.6 s warm for the official CUDA build).
            let (hashes, _) = cerebro_probe::backend::hash_directory(preflighted.backend_dir())
                .map_err(ProvenanceError::BackendUnhashable)?;
            backend_manifest_digest(&hashes)
        };
        let power = cerebro_probe::power::probe().ok();
        let other_llama_processes_at_start =
            cerebro_probe::process::running_with_image_names(LLAMA_IMAGE_NAMES).ok();
        // Last before launch: free VRAM is the most perishable fact in the key.
        let gpu_start = self.gpus.sample().map_err(ProvenanceError::Gpu)?;

        let started = Instant::now();
        let spawn = || spawner(&preflighted, &argv, &request.spawn);
        let finished = execute(&spawn, &request.plan)?;
        let wall_time = started.elapsed();
        // llama-bench has exited, so nothing reads the file any more: release the pin.
        let (model_sha256, model_size_bytes) = model.release();
        let free_vram_mib_after = self.gpus.sample_free_vram().ok();

        // The exact `-m` value; `build_argv` already refused a non-UTF-8 path, so this cannot fail,
        // but a failure is still an error rather than an empty string compared against records.
        let model_arg = preflighted
            .model_path()
            .to_str()
            .ok_or_else(|| RunError::ModelPathNotUtf8(preflighted.model_path().to_path_buf()))?;
        let (config, triple, records) = interpret(
            &finished.stdout,
            &finished.stderr_tail,
            &request.plan,
            model_arg,
        )?;
        // `interpret` refuses an empty record list; should that ever change, the run fails rather
        // than keying its result to a defaulted build 0 and an empty commit.
        let Some(first) = records.first() else {
            return Err(RunError::MissingJson {
                stderr_tail: finished.stderr_tail,
            });
        };
        let (backend_build_number, backend_commit) = (first.build_number, first.build_commit.clone());
        let backend_compiler = self.compilers.compiler(&backend_digest, backend_build_number, || {
            let env = preflighted.protections().probe_settings().environment_allowlist;
            match cerebro_probe::backend::query_version(preflighted.backend_dir(), VERSION_TOOL_NAME, env)
            {
                Ok((Some(build), _, Some(compiler))) => Some((build, compiler)),
                _ => None,
            }
        });

        Ok(BenchRun {
            provenance: RunProvenance {
                machine_profile_id: self.machine_profile_id.clone(),
                backend_build_number,
                backend_commit,
                backend_compiler,
                backend_cpu_variant: finished.cpu_variant,
                backend_manifest_digest: backend_digest,
                backend_integrity_verified: integrity_verified,
                model_sha256,
                model_size_bytes,
                free_vram_mib_at_start: gpu_start.iter().map(|g| g.free_mib).collect(),
                free_vram_mib_after,
                gpu_utilization_pct_at_start: gpu_start.iter().map(|g| g.utilization_pct).collect(),
                other_llama_processes_at_start,
                cpu: self.cpu.clone(),
                gpus: self.gpu_summaries.clone(),
                power,
            },
            argv,
            config,
            triple,
            records,
            preflight_warnings: preflighted.warnings().to_vec(),
            spawn_warnings: finished.spawn_warnings,
            pid: finished.pid,
            wall_time,
        })
    }
}

/// The parts of a running child [`execute`] drives; implemented by the supervisor's
/// [`ChildHandle`] and by the tests' fake.
trait Process {
    fn pid(&self) -> u32;
    /// `Some(exit code)` once exited (`Some(None)`: exited without a code).
    fn try_wait(&self) -> Result<Option<Option<i32>>, SpawnError>;
    /// Kill the tree; whether the OS confirmed it gone within `timeout`.
    fn kill_and_wait(&self, timeout: Duration) -> Result<bool, SpawnError>;
    /// Whether both output pipes reached end-of-file within `timeout`.
    fn wait_output_closed(&self, timeout: Duration) -> bool;
    fn take_stdout(&self, timeout: Duration) -> Result<CapturedStdout, SpawnError>;
    fn recent_lines(&self) -> Vec<String>;
    fn protection_warnings(&self) -> Vec<ProtectionWarning>;
}

impl Process for ChildHandle {
    fn pid(&self) -> u32 {
        ChildHandle::pid(self)
    }
    fn try_wait(&self) -> Result<Option<Option<i32>>, SpawnError> {
        Ok(ChildHandle::try_wait(self)?.map(|status| status.code()))
    }
    fn kill_and_wait(&self, timeout: Duration) -> Result<bool, SpawnError> {
        Ok(ChildHandle::kill_and_wait(self, timeout)?.is_some())
    }
    fn wait_output_closed(&self, timeout: Duration) -> bool {
        ChildHandle::wait_output_closed(self, timeout)
    }
    fn take_stdout(&self, timeout: Duration) -> Result<CapturedStdout, SpawnError> {
        self.wait_stdout_capture(timeout)
    }
    fn recent_lines(&self) -> Vec<String> {
        ChildHandle::recent_lines(self)
    }
    fn protection_warnings(&self) -> Vec<ProtectionWarning> {
        ChildHandle::protection_warnings(self).to_vec()
    }
}

/// A child that exited 0 with a complete, non-empty stdout document.
#[derive(Debug)]
struct Finished {
    pid: u32,
    stdout: String,
    stderr_tail: Vec<String>,
    /// From the whole stderr ring, not just the tail: the `load_backend` line comes first.
    cpu_variant: Option<String>,
    spawn_warnings: Vec<ProtectionWarning>,
}

/// Spawn, wait under the plan's timeout, and classify how the process ended. Everything short of
/// "exited 0 with a complete UTF-8 document" is a typed infrastructure error with the stderr tail.
/// The process is dropped before this returns, which closes its job and so ends its whole tree.
fn execute(
    spawn: &dyn Fn() -> Result<Box<dyn Process>, SpawnError>,
    plan: &RunPlan,
) -> Result<Finished, RunError> {
    let process = spawn()?;
    let deadline = Instant::now().checked_add(plan.timeout());
    let code = loop {
        if let Some(code) = process.try_wait()? {
            break code;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            // A failed poll after the kill means the exit could not be confirmed, which is what
            // `confirmed_gone: false` reports; the poll error itself goes into the tail rather
            // than replacing the timeout, the fact the caller acts on. Dropping `process` on
            // return still closes its job, so the kernel ends the tree either way.
            let (confirmed_gone, kill_note) = match process.kill_and_wait(KILL_GRACE) {
                Ok(gone) => (gone, None),
                Err(e) => (false, Some(format!("cerebro: could not confirm the kill: {e}"))),
            };
            process.wait_output_closed(DRAIN_TIMEOUT);
            let mut stderr_tail = stderr_tail(&process.recent_lines());
            stderr_tail.extend(kill_note);
            return Err(RunError::TimedOut {
                timeout: plan.timeout(),
                confirmed_gone,
                stderr_tail,
            });
        }
        std::thread::sleep(POLL_INTERVAL);
    };

    let drained = process.wait_output_closed(DRAIN_TIMEOUT);
    let recent = process.recent_lines();
    let stderr_tail = stderr_tail(&recent);
    if code != Some(0) {
        return Err(RunError::BackendExited { code, stderr_tail });
    }
    if !drained {
        return Err(RunError::OutputUnreadable {
            cause: format!("output pipes still open {DRAIN_TIMEOUT:?} after llama-bench exited"),
            stderr_tail,
        });
    }
    let captured = match process.take_stdout(Duration::ZERO) {
        Ok(c) => c,
        Err(e) => {
            return Err(RunError::OutputUnreadable {
                cause: e.to_string(),
                stderr_tail,
            })
        }
    };
    if let Some(cause) = captured.read_error {
        return Err(RunError::OutputUnreadable { cause, stderr_tail });
    }
    if captured.overflowed {
        return Err(RunError::OutputTooLarge {
            limit: cerebro_supervisor::MAX_STDOUT_CAPTURE_BYTES,
            stderr_tail,
        });
    }
    let stdout = match String::from_utf8(captured.bytes) {
        Ok(s) => s,
        Err(e) => {
            return Err(RunError::OutputUnreadable {
                cause: format!("stdout is not UTF-8: {e}"),
                stderr_tail,
            })
        }
    };
    if stdout.trim().is_empty() {
        return Err(RunError::MissingJson { stderr_tail });
    }
    Ok(Finished {
        pid: process.pid(),
        stdout,
        stderr_tail,
        cpu_variant: cpu_variant(&recent),
        spawn_warnings: process.protection_warnings(),
    })
}

/// The last [`STDERR_TAIL_LINES`] stderr lines (and Cerebro's own markers) from the child's
/// recent output, without their stream tags. Stdout is not in the ring for a capturing spawn;
/// any `stdout:` line that is (a fake, a future change) is left out, never shown as stderr.
fn stderr_tail(recent: &[String]) -> Vec<String> {
    let lines: Vec<String> = recent
        .iter()
        .filter_map(|line| {
            line.strip_prefix("stderr: ")
                .map(str::to_string)
                .or_else(|| line.starts_with("cerebro: ").then(|| line.clone()))
        })
        .collect();
    let skip = lines.len().saturating_sub(STDERR_TAIL_LINES);
    lines.into_iter().skip(skip).collect()
}

/// Parse the document and check it is exactly what the plan asked for: exactly one record per
/// (test, depth) of the plan, i.e. the record shapes `(n_prompt, n_gen, n_depth)` are exactly
/// `{(p, 0, d), (0, n, d) for each depth d}`; every record with the requested repetitions and this
/// run's model; one configuration; and a complete triple measured at the deepest requested depth.
fn interpret(
    stdout: &str,
    stderr_tail: &[String],
    plan: &RunPlan,
    model_arg: &str,
) -> Result<(ConfigKey, Triple, Vec<BenchRecord>), RunError> {
    let unexpected = |detail: String| RunError::UnexpectedRecords {
        detail,
        stderr_tail: stderr_tail.to_vec(),
    };
    let records = parse_records(stdout).map_err(|source| RunError::MalformedJson {
        source,
        stderr_tail: stderr_tail.to_vec(),
    })?;
    if records.is_empty() {
        return Err(RunError::MissingJson {
            stderr_tail: stderr_tail.to_vec(),
        });
    }
    if records.len() != plan.expected_records() {
        return Err(unexpected(format!(
            "{} records, expected {} (a prefill and a decode test at each of {} depths)",
            records.len(),
            plan.expected_records(),
            plan.depths().len()
        )));
    }
    let expected: BTreeSet<(u64, u64, u64)> = plan
        .depths()
        .iter()
        .flat_map(|&d| {
            let d = u64::from(d);
            [
                (u64::from(plan.prompt_tokens()), 0, d),
                (0, u64::from(plan.gen_tokens()), d),
            ]
        })
        .collect();
    let mut seen = BTreeSet::new();
    let reps = usize::try_from(plan.total_repetitions()).unwrap_or(usize::MAX);
    for (index, record) in records.iter().enumerate() {
        let shape = (record.n_prompt, record.n_gen, record.n_depth);
        if !expected.contains(&shape) {
            return Err(unexpected(format!(
                "record {index} is test (p={}, n={}, d={}), which this run did not ask for",
                shape.0, shape.1, shape.2
            )));
        }
        if !seen.insert(shape) {
            return Err(unexpected(format!(
                "record {index} repeats test (p={}, n={}, d={})",
                shape.0, shape.1, shape.2
            )));
        }
        if record.samples_ts.len() != reps {
            return Err(unexpected(format!(
                "record {index} carries {} repetitions, the run asked for {reps}",
                record.samples_ts.len()
            )));
        }
        if record.model_filename != model_arg {
            return Err(unexpected(format!(
                "record {index} measured {:?}, not the model this run passed",
                record.model_filename
            )));
        }
    }
    let triples =
        assemble_triples(&records, plan.seed()).map_err(|e| unexpected(e.to_string()))?;
    if triples.len() != 1 {
        return Err(unexpected(format!(
            "{} configurations in the output, the run measured one",
            triples.len()
        )));
    }
    let Some((config, result)) = triples.into_iter().next() else {
        return Err(unexpected("no configuration in the output".to_string()));
    };
    let triple = result.map_err(|source| RunError::IncompleteTriple {
        source,
        stderr_tail: stderr_tail.to_vec(),
    })?;
    if triple.deepest_depth.get() != u64::from(plan.deepest_depth()) {
        return Err(unexpected(format!(
            "deepest decode measured at depth {}, the run asked for {}",
            triple.deepest_depth,
            plan.deepest_depth()
        )));
    }
    Ok((config, triple, records))
}

#[cfg(test)]
mod tests;
