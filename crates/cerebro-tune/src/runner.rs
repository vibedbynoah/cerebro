//! How a candidate gets measured: the [`CandidateRunner`] seam, its production implementation
//! over `cerebro-bench` ([`LlamaBench`]), and cancellation.
//!
//! The tuner only ever sees a [`Measurement`] or a [`MeasureError`], so its selection, staging,
//! persistence and cancellation logic is tested with a fake runner, and the one real runner is a
//! thin conversion over [`BenchRunner::run_cancellable`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cerebro_bench::run::{BenchRequest, BenchRunner, RunPlan};
use cerebro_bench::{BenchRun, RunError, TestKind, Triple};
use cerebro_flags::Flag;
use cerebro_probe::PowerState;
use cerebro_store::ProvenanceKey;
use cerebro_supervisor::{BenchLaunchSpec, SpawnOptions};

use crate::error::MeasureError;

/// Cancels a tune from any thread. Clones share one flag.
///
/// With [`LlamaBench`], a cancel is honoured at every stage of a run: before preflight, between
/// 4 MiB chunks of a first model hash, immediately before the spawn (so nothing is launched after
/// a cancel that arrives before it), and at every 10 ms poll while llama-bench runs, which kills
/// its whole process tree. No further candidate starts after a cancel.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that has not been cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether [`CancelToken::cancel`] has been called on this token or a clone.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// The underlying flag, for [`BenchRunner::run_cancellable`].
    pub fn flag(&self) -> &AtomicBool {
        &self.0
    }
}

/// One test's kept samples (the first repetition already discarded).
#[derive(Debug, Clone, PartialEq)]
pub struct TestSamples {
    /// Prefill or decode.
    pub kind: TestKind,
    /// Context depth the test ran at.
    pub depth: u64,
    /// Tokens per second, one per kept repetition.
    pub kept_tok_s: Vec<f64>,
}

/// The conditions one measurement was taken under. Two measurements are comparable only when
/// their identities (model, build, machine) are equal; the contention signals say whether this
/// one was taken on a machine busy with something else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conditions {
    /// SHA-256 of the model file measured.
    pub model_sha256: String,
    /// The llama.cpp build's manifest digest.
    pub backend_manifest_digest: String,
    /// The machine profile (CPU topology and GPU models).
    pub machine_profile_id: String,
    /// Mains or battery and battery saver at launch; `None` if unavailable.
    pub power: Option<PowerState>,
    /// Other llama-server / llama-bench / llama-cli processes running at launch; `None` if the
    /// process snapshot failed.
    pub other_llama_processes: Option<usize>,
    /// The highest GPU utilization of any device at launch, in percent; `None` without a GPU
    /// that reports it.
    pub gpu_utilization_pct: Option<u32>,
}

/// One candidate's completed measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// What it is keyed to: the exact binary, model and machine state.
    pub provenance: ProvenanceKey,
    /// The conditions it was taken under, for the tuner's comparability checks.
    pub conditions: Conditions,
    /// The exact llama-bench argv (without the program path).
    pub argv: Vec<String>,
    /// Prefill@0, decode@0 and decode at the workload's depth.
    pub triple: Triple,
    /// Generation threads as llama-bench reported them.
    pub threads: Option<u32>,
    /// Every test's kept samples, for progress display.
    pub samples: Vec<TestSamples>,
    /// Protections switched off for this run, one line each (the neutral recorded warning).
    pub warnings: Vec<String>,
}

/// Measures one candidate configuration.
pub trait CandidateRunner {
    /// Measure `flags` (Bench dialect, without `-m`) under `plan`, stopping early when `cancel`
    /// is set.
    ///
    /// # Errors
    /// [`MeasureError::Cancelled`] after a cancel (saying whether the kill was confirmed);
    /// [`MeasureError::CandidateFailed`] when this configuration launched but gave no trustworthy
    /// result; [`MeasureError::Run`] or [`MeasureError::Key`] for a refusal that would stop every
    /// candidate.
    fn measure(
        &self,
        flags: &[Flag],
        plan: &RunPlan,
        cancel: &CancelToken,
    ) -> Result<Measurement, MeasureError>;
}

/// The production runner: preflighted, job-contained llama-bench through `cerebro-bench`.
#[derive(Debug)]
pub struct LlamaBench<'a> {
    runner: &'a BenchRunner,
    launch: BenchLaunchSpec,
    spawn: SpawnOptions,
}

impl<'a> LlamaBench<'a> {
    /// Measure with `runner`, launching per `launch` (backend, manifest, model, the user's
    /// protections) with the child locations in `spawn`.
    pub fn new(runner: &'a BenchRunner, launch: BenchLaunchSpec, spawn: SpawnOptions) -> Self {
        Self {
            runner,
            launch,
            spawn,
        }
    }
}

impl CandidateRunner for LlamaBench<'_> {
    fn measure(
        &self,
        flags: &[Flag],
        plan: &RunPlan,
        cancel: &CancelToken,
    ) -> Result<Measurement, MeasureError> {
        let request = BenchRequest {
            launch: self.launch.clone(),
            spawn: self.spawn.clone(),
            config: flags.to_vec(),
            plan: plan.clone(),
        };
        let run = self
            .runner
            .run_cancellable(&request, cancel.flag())
            .map_err(classify)?;
        measurement(run)
    }
}

/// A cancel is a cancel; an infrastructure error belongs to this candidate; anything else
/// (preflight, provenance, a refused configuration) would stop every candidate. The bench error
/// is kept whole either way.
fn classify(e: RunError) -> MeasureError {
    match e {
        RunError::Cancelled { confirmed_gone, .. } => MeasureError::Cancelled { confirmed_gone },
        e if e.is_infrastructure() => MeasureError::CandidateFailed(e),
        e => MeasureError::Run(e),
    }
}

/// Convert a completed bench run.
fn measurement(run: BenchRun) -> Result<Measurement, MeasureError> {
    let provenance = run.provenance.store_key()?;
    let p = &run.provenance;
    let conditions = Conditions {
        model_sha256: p.model_sha256.clone(),
        backend_manifest_digest: p.backend_manifest_digest.clone(),
        machine_profile_id: p.machine_profile_id.clone(),
        power: p.power,
        other_llama_processes: p.other_llama_processes_at_start.as_ref().map(Vec::len),
        gpu_utilization_pct: p.gpu_utilization_pct_at_start.iter().flatten().copied().max(),
    };
    let samples = run
        .records
        .iter()
        .map(|r| TestSamples {
            kind: if r.n_gen > 0 { TestKind::Decode } else { TestKind::Prefill },
            depth: r.n_depth,
            kept_tok_s: r.samples_ts.iter().skip(1).copied().collect(),
        })
        .collect();
    let warnings = run
        .preflight_warnings
        .iter()
        .map(ToString::to_string)
        .chain(run.spawn_warnings.iter().map(ToString::to_string))
        .collect();
    Ok(Measurement {
        provenance,
        conditions,
        argv: run.argv,
        triple: run.triple,
        threads: run.config.n_threads(),
        samples,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cancel_is_shared_by_clones() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(!clone.is_cancelled());
        token.cancel();
        assert!(clone.is_cancelled());
        assert!(clone.flag().load(Ordering::Relaxed));
    }

    #[test]
    fn bench_errors_are_classified_by_who_they_stop() {
        let cancelled = RunError::Cancelled {
            confirmed_gone: true,
            stderr_tail: Vec::new(),
        };
        assert!(matches!(classify(cancelled), MeasureError::Cancelled { confirmed_gone: true }));
        let unconfirmed = RunError::Cancelled {
            confirmed_gone: false,
            stderr_tail: Vec::new(),
        };
        assert!(matches!(classify(unconfirmed), MeasureError::Cancelled { confirmed_gone: false }));
        let oom = RunError::BackendExited {
            code: Some(1),
            stderr_tail: vec!["cudaMalloc failed: out of memory".into()],
        };
        match classify(oom) {
            MeasureError::CandidateFailed(e @ RunError::BackendExited { code: Some(1), .. }) => {
                assert!(e.to_string().contains("out of memory"), "{e}");
            }
            other => panic!("expected a candidate failure, got {other:?}"),
        }
        let orphan = RunError::TimedOut {
            timeout: std::time::Duration::from_secs(1),
            confirmed_gone: false,
            stderr_tail: Vec::new(),
        };
        assert!(matches!(
            classify(orphan),
            MeasureError::CandidateFailed(RunError::TimedOut { confirmed_gone: false, .. })
        ));
        let refused = RunError::Plan(cerebro_bench::run::PlanError::NoDepthZero);
        assert!(matches!(classify(refused), MeasureError::Run(_)));
    }
}
