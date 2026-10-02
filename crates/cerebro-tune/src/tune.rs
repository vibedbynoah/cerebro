//! The tune itself: screen the candidates stage by stage, persist every measurement, stream
//! progress, and choose the winner.
//!
//! # Comparability
//!
//! A verdict compares measurements, so it is only as good as the claim that they were taken
//! under the same conditions. Every [`Measurement`] carries its [`Conditions`], and they are
//! checked rather than assumed:
//!
//! - A measurement keyed to a different model file, llama.cpp build or machine than the
//!   baseline's stops the tune ([`TuneError::ConditionsChanged`]): nothing measured from there on
//!   would be comparable.
//! - A measurement taken under contention ([`Contention`]: another llama.cpp process running at
//!   launch, a GPU at least [`CONTENDED_GPU_UTILIZATION_PCT`] busy, or a power source or battery
//!   saver state different from the baseline's) is still persisted and reported, but is used
//!   neither to choose nor to pick a stage's leader. A contended baseline means nothing is
//!   compared and llama.cpp's automatic settings are recommended. The outcome carries one neutral
//!   warning. [`ScreenSettings::allow_contended`] switches this off (the warning then says so).
//! - Free VRAM is not compared: it legitimately varies between runs, and it is recorded in every
//!   record's key.
//!
//! # Drift check
//!
//! When a candidate wins, the baseline is measured once more at the end
//! ([`ScreenSettings::recheck_baseline`], on by default; it costs one run, and only when there is
//! a recommendation to defend). The winner must be eligible against the re-measured baseline
//! too. If the two baseline measurements disagree (their decode-at-depth intervals do not
//! overlap) the machine drifted during the tune (heat, power, background load), and the winner
//! stands only if it stays demonstrably faster than both even when scaled down by the ratio the
//! baseline moved ([`crate::select::faster_despite_drift`]); then the outcome carries one neutral
//! warning naming both baseline figures, otherwise the baseline is recommended. Withdrawing on
//! any disagreement would be wrong the other way: three-sample intervals of two identical runs
//! miss each other about one time in ten, which would throw away a measured 6x improvement for
//! noise. A withdrawal's reason is recorded in [`TuneOutcome::excluded`].
//!
//! # Events
//!
//! Every [`TuneEvent::CandidateStarted`] is followed by exactly one
//! [`TuneEvent::CandidateFinished`], on every path including errors and cancels; a measured
//! candidate's [`TuneEvent::Sample`]s come between the two, after the measurement is persisted.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cerebro_bench::run::{RunPlan, DEFAULT_TIMEOUT};
use cerebro_bench::{RunError, Summary, TestKind, Triple};
use cerebro_flags::{Flag, ValueError};
use cerebro_store::{MeasurementSummary, MeasurementTriple, Record, SamplerProfile, Store};

use crate::candidate::{CandidateConfig, MachineFacts, Planner, Workload};
use crate::error::{Identity, MeasureError, TuneError};
use crate::gguf::ModelFacts;
use crate::runner::{CancelToken, CandidateRunner, Conditions, Measurement};
use crate::select::{
    agree_at_depth, broken_guard, choose, eligible, faster_at_depth, faster_despite_drift, Choice,
    Guard, Scored,
};

/// GPU utilization at launch, in percent, from which a measurement counts as contended: something
/// else is computing on the GPU it is measured on.
pub const CONTENDED_GPU_UTILIZATION_PCT: u32 = 50;

/// How each candidate is screened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenSettings {
    /// Repetitions kept per test (llama-bench runs one more and the first is discarded).
    pub kept_repetitions: u32,
    /// `-p`: prompt tokens per prefill test.
    pub prompt_tokens: u32,
    /// `-n`: tokens generated per decode test.
    pub gen_tokens: u32,
    /// Measure decode at this depth instead of the workload's (still clamped to the model's
    /// context).
    pub depth_override: Option<u32>,
    /// Per-candidate timeout; a candidate past it is killed and recorded as failed.
    pub timeout: Duration,
    /// `--delay` seconds before each test, for a thermally limited machine.
    pub delay_secs: u32,
    /// Compare measurements taken under contention too (another llama.cpp process, a busy GPU, a
    /// power source change). Off by default; on, the outcome records one neutral warning.
    pub allow_contended: bool,
    /// Measure the baseline again at the end when a candidate won, and recommend the baseline if
    /// the winner does not beat the second measurement, or the two baseline measurements
    /// disagree by enough to account for the winner's margin. On by default.
    pub recheck_baseline: bool,
}

impl Default for ScreenSettings {
    /// Three kept repetitions of `-p 512 -n 128`, the workload's depth, the bench default
    /// timeout, no delay, contended measurements not compared, the baseline re-checked.
    fn default() -> Self {
        Self {
            kept_repetitions: 3,
            prompt_tokens: 512,
            gen_tokens: 128,
            depth_override: None,
            timeout: DEFAULT_TIMEOUT,
            delay_secs: 0,
            allow_contended: false,
            recheck_baseline: true,
        }
    }
}

/// Everything one tune needs besides the runner, the store and the callbacks.
#[derive(Debug, Clone)]
pub struct TuneRequest {
    /// What the model is for: sets the depth the winner is chosen at.
    pub workload: Workload,
    /// The model's shape and weight layout.
    pub model: ModelFacts,
    /// Free VRAM and CPU topology.
    pub machine: MachineFacts,
    /// How each candidate is screened.
    pub screen: ScreenSettings,
    /// The sampler profile the configuration will serve with, recorded with every measurement.
    pub sampler: SamplerProfile,
}

/// Which stage a candidate belongs to (see [`crate::candidate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// The baseline and the weight placements.
    Placement,
    /// KV cache type variants of the placement leader.
    KvCache,
    /// Thread variants of the leader so far.
    Threads,
    /// The baseline measured again at the end, when a candidate won (the drift check).
    Recheck,
}

/// Why a measurement was not taken under conditions comparable with the baseline's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Contention {
    /// This many other llama-server / llama-bench / llama-cli processes were running at launch.
    OtherLlamaProcesses(usize),
    /// A GPU was this busy (percent) at launch.
    GpuBusy(u32),
    /// The power source or battery saver state differed from the baseline's.
    PowerChanged,
}

impl Contention {
    /// A short neutral description, for warnings and progress.
    pub fn describe(&self) -> String {
        match self {
            Contention::OtherLlamaProcesses(n) => format!("{n} other llama.cpp process(es) running"),
            Contention::GpuBusy(pct) => format!("a GPU {pct}% busy at launch"),
            Contention::PowerChanged => "the power source differed from the baseline's".to_string(),
        }
    }
}

/// Why a measured candidate was not chosen although it was not simply slower, or why a winner
/// was withdrawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exclusion {
    /// Taken under contention, so not compared. On the baseline (index 0): no candidate was
    /// compared at all.
    Contended(Contention),
    /// Faster decode at depth, but demonstrably slower at a guarded metric ("faster decode,
    /// slower prefill: not chosen").
    Guard(Guard),
    /// The winner, withdrawn because the baseline re-measured at the end disagreed with the
    /// first baseline measurement by enough to account for the winner's margin: the machine
    /// drifted during the tune.
    BaselineDrifted,
    /// The winner, withdrawn because it was not eligible against the re-measured baseline.
    NotFasterThanRecheck,
    /// The winner, withdrawn because the baseline re-measure failed or was contended, so the
    /// improvement could not be confirmed.
    Unconfirmed,
}

/// One exclusion, by start-order index (as in [`TuneEvent`]s and [`MeasuredCandidate::index`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Excluded {
    /// The candidate's start-order index.
    pub index: usize,
    /// Why.
    pub reason: Exclusion,
}

/// What kind of failure a candidate had, for display without matching on message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureKind {
    /// Killed at the per-candidate timeout.
    TimedOut,
    /// llama-bench exited unsuccessfully (an out-of-memory allocation, a crash, a rejected flag).
    Exited {
        /// The exit code, when the OS reported one.
        code: Option<i32>,
    },
    /// It exited cleanly but its output was missing, unreadable or did not match the request.
    BadOutput,
    /// It could not be launched.
    Spawn,
    /// Anything else.
    Other,
}

impl FailureKind {
    /// Classify a bench error.
    pub fn of(e: &RunError) -> Self {
        match e {
            RunError::TimedOut { .. } => FailureKind::TimedOut,
            RunError::BackendExited { code, .. } => FailureKind::Exited { code: *code },
            RunError::OutputUnreadable { .. }
            | RunError::OutputTooLarge { .. }
            | RunError::MissingJson { .. }
            | RunError::MalformedJson { .. }
            | RunError::UnexpectedRecords { .. }
            | RunError::IncompleteTriple { .. } => FailureKind::BadOutput,
            RunError::Spawn(_) => FailureKind::Spawn,
            _ => FailureKind::Other,
        }
    }
}

/// A candidate that produced a measurement (persisted before it is reported).
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredCandidate {
    /// Order in which candidates started, from 0 (the baseline).
    pub index: usize,
    /// Its stage.
    pub stage: Stage,
    /// What was measured.
    pub config: CandidateConfig,
    /// The measurement.
    pub measurement: Measurement,
    /// Whether it was taken under contention (see the module docs).
    pub contention: Option<Contention>,
    /// Position of its record in the store file when it was appended
    /// ([`cerebro_store::AppendOutcome::index`]); a rotation of the file starts a new numbering.
    pub record_index: usize,
}

/// A candidate that launched but gave no trustworthy result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedCandidate {
    /// Order in which candidates started.
    pub index: usize,
    /// Its stage.
    pub stage: Stage,
    /// What was attempted.
    pub config: CandidateConfig,
    /// What kind of failure.
    pub kind: FailureKind,
    /// Why it failed (with llama-bench's stderr tail).
    pub reason: String,
}

/// How the tune ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Candidate `winner` (an index into [`TuneOutcome::measured`], which differs from its
    /// start-order [`MeasuredCandidate::index`] once a candidate has failed) is demonstrably
    /// faster than llama.cpp's automatic settings at the workload's depth and not demonstrably
    /// slower at prefill or depth-0 decode.
    Improved {
        /// Index into [`TuneOutcome::measured`].
        winner: usize,
    },
    /// Nothing beat llama.cpp's automatic settings: they are the recommendation.
    BaselineWins,
    /// Cancelled before finishing: nothing is recommended. Measurements taken are persisted.
    Cancelled,
}

/// A finished (or cancelled) tune.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneOutcome {
    /// The workload tuned for.
    pub workload: Workload,
    /// The depth decode was judged at.
    pub depth: u32,
    /// Every measured candidate in start order; the first is the baseline when it was measured.
    /// The end-of-tune baseline re-measure is not here but in [`TuneOutcome::baseline_recheck`].
    pub measured: Vec<MeasuredCandidate>,
    /// Every candidate that failed.
    pub failed: Vec<FailedCandidate>,
    /// The verdict.
    pub verdict: Verdict,
    /// Candidates set aside for a reason other than being slower, and withdrawn winners.
    pub excluded: Vec<Excluded>,
    /// The baseline measured again at the end, when a candidate won and
    /// [`ScreenSettings::recheck_baseline`] is on.
    pub baseline_recheck: Option<MeasuredCandidate>,
    /// Each distinct protection or contention warning, once.
    pub warnings: Vec<String>,
}

impl TuneOutcome {
    /// The baseline's measurement, when it was taken.
    pub fn baseline(&self) -> Option<&MeasuredCandidate> {
        self.measured.first().filter(|m| m.config.is_baseline())
    }

    /// The configuration to use: the winner, the baseline when nothing beat it, `None` when
    /// cancelled. Its [`MeasuredCandidate::record_index`] locates its record in the store; serve
    /// it with [`TuneOutcome::recommended_server_flags`].
    pub fn recommended(&self) -> Option<&MeasuredCandidate> {
        match self.verdict {
            Verdict::Improved { winner } => self.measured.get(winner),
            Verdict::BaselineWins => self.baseline(),
            Verdict::Cancelled => None,
        }
    }

    /// The llama-server flags for [`TuneOutcome::recommended`] (Server dialect, without `-m`, the
    /// context size, the network or the key, which the serving launch adds): the configuration
    /// served exactly as it was measured, with the thread count llama-bench reported, so nothing
    /// the tuner varied is left to a llama-server default. `None` when cancelled.
    ///
    /// # Errors
    /// As [`CandidateConfig::server_flags`].
    pub fn recommended_server_flags(&self) -> Result<Option<Vec<Flag>>, ValueError> {
        self.recommended()
            .map(|m| m.config.server_flags(m.measurement.threads))
            .transpose()
    }

    /// The recommendation's median decode tok/s at depth over the baseline's.
    pub fn speedup(&self) -> Option<f64> {
        let base = self.baseline()?.measurement.triple.tg_tok_s_deepest.median;
        let best = self.recommended()?.measurement.triple.tg_tok_s_deepest.median;
        (base > 0.0).then(|| best / base)
    }
}

/// How one candidate ended, for [`TuneEvent::CandidateFinished`].
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateResult {
    /// Measured and persisted.
    Measured {
        /// Prefill at depth 0.
        pp: Summary,
        /// Decode at depth 0.
        tg_depth0: Summary,
        /// Decode at the workload's depth: what the winner is chosen by.
        tg_at_depth: Summary,
        /// Generation threads llama-bench used.
        threads: Option<u32>,
        /// Whether it was taken under contention, and so is not compared.
        contention: Option<Contention>,
    },
    /// Launched but gave no trustworthy result, or its result could not be used (the tune may
    /// stop: see the error it returns).
    Failed {
        /// Why.
        reason: String,
    },
    /// Cancelled while waiting or running; its process, if any, was confirmed gone.
    Cancelled,
}

/// Progress, streamed to the caller's callback as it happens.
///
/// llama-bench reports a run's samples when the run ends, so a candidate's [`TuneEvent::Sample`]
/// events arrive together just before its [`TuneEvent::CandidateFinished`].
#[derive(Debug, Clone, PartialEq)]
pub enum TuneEvent {
    /// A stage with `candidates` candidates is starting.
    StageStarted {
        /// The stage.
        stage: Stage,
        /// How many candidates it will run (fewer if cancelled).
        candidates: usize,
    },
    /// A candidate is starting.
    CandidateStarted {
        /// Its start-order index.
        index: usize,
        /// Its stage.
        stage: Stage,
        /// Its configuration (`Display` gives a short description).
        config: CandidateConfig,
    },
    /// One test's kept samples.
    Sample {
        /// The candidate's index.
        index: usize,
        /// Prefill or decode.
        kind: TestKind,
        /// Context depth.
        depth: u64,
        /// Tokens per second, one per kept repetition.
        kept_tok_s: Vec<f64>,
    },
    /// A candidate finished.
    CandidateFinished {
        /// Its index.
        index: usize,
        /// How it ended.
        result: CandidateResult,
    },
    /// The tune chose: `index` is the winner's start-order index (the baseline's, 0, when it won).
    Winner {
        /// Start-order index of the recommendation.
        index: usize,
        /// Its configuration.
        config: CandidateConfig,
        /// Whether it beat the baseline or is the baseline.
        verdict: Verdict,
        /// Its median decode tok/s at depth.
        tg_at_depth: f64,
        /// The baseline's median decode tok/s at depth.
        baseline_tg_at_depth: f64,
    },
    /// The tune was cancelled after `measured` measurements.
    Cancelled {
        /// Candidates measured (and persisted) before the cancel.
        measured: usize,
    },
}

/// The decode depth for `request`: the workload's (or the override), clamped so the deepest test
/// fits the model's training context.
///
/// # Errors
/// [`TuneError::ContextTooShort`] when no depth above zero fits; [`TuneError::ZeroDepth`] for a
/// zero override.
pub fn resolve_depth(request: &TuneRequest) -> Result<u32, TuneError> {
    let s = &request.screen;
    let target = s.depth_override.unwrap_or(request.workload.target_depth());
    let depth = match request.model.context_length {
        Some(context) => {
            let room = context.saturating_sub(s.prompt_tokens.max(s.gen_tokens));
            if room == 0 {
                return Err(TuneError::ContextTooShort {
                    context,
                    prompt: s.prompt_tokens,
                    generated: s.gen_tokens,
                });
            }
            target.min(room)
        }
        None => target,
    };
    if depth == 0 {
        return Err(TuneError::ZeroDepth);
    }
    Ok(depth)
}

/// Run a tune: measure the baseline and every stage's candidates with `runner`, append each
/// measurement to `store` as it completes, report progress to `on_event`, and choose.
///
/// Cancelling `cancel` kills the running candidate and returns an outcome with
/// [`Verdict::Cancelled`] holding what was measured.
///
/// # Errors
/// [`TuneError`]: an invalid plan, a model context too short, a baseline that could not be
/// measured, a refusal that stops every candidate, a measurement keyed to a different model,
/// build or machine, a killed llama-bench not confirmed gone, or a measurement that could not be
/// persisted.
pub fn tune(
    request: &TuneRequest,
    runner: &dyn CandidateRunner,
    store: &Store,
    cancel: &CancelToken,
    on_event: &mut dyn FnMut(&TuneEvent),
) -> Result<TuneOutcome, TuneError> {
    let depth = resolve_depth(request)?;
    let s = &request.screen;
    let plan = RunPlan::new(s.kept_repetitions, s.prompt_tokens, s.gen_tokens, [0, depth])?
        .with_timeout(s.timeout)?
        .with_delay_secs(s.delay_secs)?;
    let kv_tokens = u64::from(depth) + u64::from(s.prompt_tokens.max(s.gen_tokens));
    let planner = Planner::new(&request.model, &request.machine, kv_tokens);

    let mut t = Tune {
        request,
        runner,
        store,
        cancel,
        on_event,
        plan,
        tried: Vec::new(),
        outcome: TuneOutcome {
            workload: request.workload,
            depth,
            measured: Vec::new(),
            failed: Vec::new(),
            verdict: Verdict::Cancelled,
            excluded: Vec::new(),
            baseline_recheck: None,
            warnings: Vec::new(),
        },
    };

    let mut placement = vec![CandidateConfig::BASELINE];
    placement.extend(planner.placements());
    if !t.stage(Stage::Placement, placement)? {
        return Ok(t.cancelled());
    }
    let leader = t.leader();
    if !t.stage(Stage::KvCache, planner.kv_variants(&leader))? {
        return Ok(t.cancelled());
    }
    let leader = t.leader();
    if !t.stage(Stage::Threads, planner.thread_variants(&leader))? {
        return Ok(t.cancelled());
    }
    t.finish()
}

struct Tune<'a> {
    request: &'a TuneRequest,
    runner: &'a dyn CandidateRunner,
    store: &'a Store,
    cancel: &'a CancelToken,
    on_event: &'a mut dyn FnMut(&TuneEvent),
    plan: RunPlan,
    /// Every configuration started, so a stage never repeats one.
    tried: Vec<CandidateConfig>,
    outcome: TuneOutcome,
}

/// How one started candidate ended when the tune goes on: measured, or failed.
type Ran = Result<MeasuredCandidate, FailedCandidate>;

/// What the end-of-tune baseline re-measure decided.
enum Recheck {
    Confirmed,
    Rejected(Exclusion),
    Cancelled,
}

impl Tune<'_> {
    /// Run a stage's untried candidates. `Ok(false)`: cancelled.
    fn stage(&mut self, stage: Stage, candidates: Vec<CandidateConfig>) -> Result<bool, TuneError> {
        let fresh: Vec<CandidateConfig> =
            candidates.into_iter().filter(|c| !self.tried.contains(c)).collect();
        if fresh.is_empty() {
            return Ok(true);
        }
        (self.on_event)(&TuneEvent::StageStarted {
            stage,
            candidates: fresh.len(),
        });
        for config in fresh {
            match self.run_one(stage, config)? {
                None => return Ok(false),
                Some(Ok(m)) => self.outcome.measured.push(m),
                Some(Err(f)) => self.outcome.failed.push(f),
            }
        }
        Ok(true)
    }

    /// Start, measure and record one candidate: `Ok(Some(Ok(_)))` measured, `Ok(Some(Err(_)))`
    /// failed with the tune going on, `Ok(None)` cancelled. Every path after
    /// [`TuneEvent::CandidateStarted`] emits exactly one [`TuneEvent::CandidateFinished`].
    fn run_one(&mut self, stage: Stage, config: CandidateConfig) -> Result<Option<Ran>, TuneError> {
        if self.cancel.is_cancelled() {
            return Ok(None);
        }
        let flags = config.bench_flags()?;
        let index = self.tried.len();
        self.tried.push(config);
        (self.on_event)(&TuneEvent::CandidateStarted {
            index,
            stage,
            config,
        });
        match self.runner.measure(&flags, &self.plan, self.cancel) {
            Ok(measurement) => match self.record(index, stage, config, measurement) {
                Ok(measured) => Ok(Some(Ok(measured))),
                Err(e) => Err(self.finished_with_error(index, e)),
            },
            Err(MeasureError::Cancelled {
                confirmed_gone: true,
            }) => {
                (self.on_event)(&TuneEvent::CandidateFinished {
                    index,
                    result: CandidateResult::Cancelled,
                });
                Ok(None)
            }
            Err(MeasureError::Cancelled {
                confirmed_gone: false,
            }) => Err(self.finished_with_error(
                index,
                TuneError::ProcessNotConfirmedGone(
                    "cancelled, and the OS did not confirm the kill".to_string(),
                ),
            )),
            Err(MeasureError::CandidateFailed(e)) => {
                let reason = e.to_string();
                if matches!(
                    e,
                    RunError::TimedOut {
                        confirmed_gone: false,
                        ..
                    }
                ) {
                    return Err(
                        self.finished_with_error(index, TuneError::ProcessNotConfirmedGone(reason))
                    );
                }
                (self.on_event)(&TuneEvent::CandidateFinished {
                    index,
                    result: CandidateResult::Failed {
                        reason: reason.clone(),
                    },
                });
                // The first baseline is what everything is compared with; the re-measure at the
                // end failing only leaves the winner unconfirmed.
                if config.is_baseline() && self.outcome.measured.is_empty() {
                    return Err(TuneError::BaselineFailed(e));
                }
                Ok(Some(Err(FailedCandidate {
                    index,
                    stage,
                    config,
                    kind: FailureKind::of(&e),
                    reason,
                })))
            }
            Err(other) => Err(self.finished_with_error(index, TuneError::Measure(other))),
        }
    }

    /// Report candidate `index` as failed with `e`'s message, and hand `e` back.
    fn finished_with_error(&mut self, index: usize, e: TuneError) -> TuneError {
        (self.on_event)(&TuneEvent::CandidateFinished {
            index,
            result: CandidateResult::Failed {
                reason: e.to_string(),
            },
        });
        e
    }

    /// Check a completed measurement's identity against the baseline's, persist it, then report
    /// its samples and its finish.
    fn record(
        &mut self,
        index: usize,
        stage: Stage,
        config: CandidateConfig,
        measurement: Measurement,
    ) -> Result<MeasuredCandidate, TuneError> {
        let baseline = self.outcome.measured.first().map(|b| &b.measurement.conditions);
        if let Some(changed) = baseline.and_then(|b| identity_change(b, &measurement.conditions)) {
            return Err(TuneError::ConditionsChanged(changed));
        }
        let contention = contention(&measurement.conditions, baseline);
        let record_index = self.persist(&measurement)?;
        for test in &measurement.samples {
            (self.on_event)(&TuneEvent::Sample {
                index,
                kind: test.kind,
                depth: test.depth,
                kept_tok_s: test.kept_tok_s.clone(),
            });
        }
        for w in &measurement.warnings {
            if !self.outcome.warnings.contains(w) {
                self.outcome.warnings.push(w.clone());
            }
        }
        let triple = measurement.triple;
        (self.on_event)(&TuneEvent::CandidateFinished {
            index,
            result: CandidateResult::Measured {
                pp: triple.pp_tok_s,
                tg_depth0: triple.tg_tok_s_depth0,
                tg_at_depth: triple.tg_tok_s_deepest,
                threads: measurement.threads,
                contention,
            },
        });
        Ok(MeasuredCandidate {
            index,
            stage,
            config,
            measurement,
            contention,
            record_index,
        })
    }

    /// Append `m` to the store; its record's position.
    fn persist(&self, m: &Measurement) -> Result<usize, TuneError> {
        let record = Record::new(
            m.provenance.clone(),
            self.request.sampler,
            m.argv.clone(),
            store_triple(&m.triple)?,
            None,
            None,
            now_unix_ms(),
        )?;
        Ok(self.store.append(&record)?.index)
    }

    /// Whether `m` may be compared.
    fn comparable(&self, m: &MeasuredCandidate) -> bool {
        self.request.screen.allow_contended || m.contention.is_none()
    }

    /// The selection over everything measured so far: the measured index of the choice, or
    /// `None` for the baseline. Contended measurements are left out (see the module docs).
    fn choice(&self) -> Option<usize> {
        let (baseline, rest) = self.outcome.measured.split_first()?;
        if !self.comparable(baseline) {
            return None;
        }
        // `rest` starts after the baseline at `measured[0]`, hence `i + 1`.
        let usable: Vec<(usize, Scored)> = rest
            .iter()
            .enumerate()
            .filter(|(_, m)| self.comparable(m))
            .map(|(i, m)| (i + 1, scored(m)))
            .collect();
        let scores: Vec<Scored> = usable.iter().map(|&(_, s)| s).collect();
        match choose(&scored(baseline), &scores) {
            Choice::Baseline => None,
            Choice::Candidate(i) => usable.get(i).map(|&(measured, _)| measured),
        }
    }

    /// The configuration the next stage varies.
    fn leader(&self) -> CandidateConfig {
        self.choice()
            .and_then(|i| self.outcome.measured.get(i))
            .map_or(CandidateConfig::BASELINE, |m| m.config)
    }

    /// Record every candidate set aside for contention, and every one that decoded faster but
    /// broke a guard.
    fn record_exclusions(&mut self) {
        let Some((baseline, rest)) = self.outcome.measured.split_first() else {
            return;
        };
        let mut excluded = Vec::new();
        if !self.comparable(baseline) {
            if let Some(c) = baseline.contention {
                excluded.push(Excluded {
                    index: baseline.index,
                    reason: Exclusion::Contended(c),
                });
            }
        } else {
            let base = scored(baseline);
            for m in rest {
                let reason = match m.contention {
                    Some(c) if !self.comparable(m) => Some(Exclusion::Contended(c)),
                    _ => {
                        let s = scored(m);
                        broken_guard(&s, &base)
                            .filter(|_| faster_at_depth(&s, &base))
                            .map(Exclusion::Guard)
                    }
                };
                if let Some(reason) = reason {
                    excluded.push(Excluded {
                        index: m.index,
                        reason,
                    });
                }
            }
        }
        self.outcome.excluded.extend(excluded);
    }

    /// Measure the baseline again and decide whether the winner (`measured[winner]`) stands.
    fn recheck(&mut self, winner: usize) -> Result<Recheck, TuneError> {
        (self.on_event)(&TuneEvent::StageStarted {
            stage: Stage::Recheck,
            candidates: 1,
        });
        let again = match self.run_one(Stage::Recheck, CandidateConfig::BASELINE)? {
            None => return Ok(Recheck::Cancelled),
            Some(Err(f)) => {
                self.outcome.failed.push(f);
                return Ok(Recheck::Rejected(Exclusion::Unconfirmed));
            }
            Some(Ok(m)) => m,
        };
        let decision = match (self.outcome.measured.first(), self.outcome.measured.get(winner)) {
            _ if !self.comparable(&again) => Recheck::Rejected(Exclusion::Unconfirmed),
            (Some(first), Some(best)) => {
                let (first, second, best) = (scored(first), scored(&again), scored(best));
                if !eligible(&best, &second) {
                    Recheck::Rejected(Exclusion::NotFasterThanRecheck)
                } else if agree_at_depth(&first, &second) {
                    Recheck::Confirmed
                } else if faster_despite_drift(&best, &first, &second) {
                    self.outcome.warnings.push(format!(
                        "llama.cpp's automatic settings decoded {:.1} tok/s at the start of the \
                         tune and {:.1} at the end; the recommendation is faster than both by \
                         more than that change",
                        first.tg_at_depth.median, second.tg_at_depth.median
                    ));
                    Recheck::Confirmed
                } else {
                    Recheck::Rejected(Exclusion::BaselineDrifted)
                }
            }
            // Unreachable: a winner implies a baseline and the winner itself.
            _ => Recheck::Rejected(Exclusion::Unconfirmed),
        };
        self.outcome.baseline_recheck = Some(again);
        Ok(decision)
    }

    /// The one neutral warning about contention: what was set aside, or that the check is off.
    fn contention_warning(&self) -> Option<String> {
        let all = self.outcome.measured.iter().chain(&self.outcome.baseline_recheck);
        let mut reasons: Vec<String> = Vec::new();
        let mut contended = 0usize;
        for c in all.filter_map(|m| m.contention) {
            contended += 1;
            let text = c.describe();
            if !reasons.contains(&text) {
                reasons.push(text);
            }
        }
        if self.request.screen.allow_contended {
            return Some(
                "measurements taken under contention are compared (allow_contended is on)"
                    .to_string(),
            );
        }
        (contended > 0).then(|| {
            format!(
                "{contended} measurement(s) taken under contention ({}) were not compared; \
                 allow_contended compares them",
                reasons.join("; ")
            )
        })
    }

    fn cancelled(mut self) -> TuneOutcome {
        self.outcome.verdict = Verdict::Cancelled;
        let measured = self.outcome.measured.len();
        (self.on_event)(&TuneEvent::Cancelled { measured });
        self.outcome
    }

    fn finish(mut self) -> Result<TuneOutcome, TuneError> {
        self.record_exclusions();
        let mut winner = self.choice();
        if let Some(w) = winner.filter(|_| self.request.screen.recheck_baseline) {
            match self.recheck(w)? {
                Recheck::Cancelled => return Ok(self.cancelled()),
                Recheck::Confirmed => {}
                Recheck::Rejected(reason) => {
                    if let Some(m) = self.outcome.measured.get(w) {
                        self.outcome.excluded.push(Excluded {
                            index: m.index,
                            reason,
                        });
                    }
                    winner = None;
                }
            }
        }
        if let Some(w) = self.contention_warning() {
            self.outcome.warnings.push(w);
        }
        self.outcome.verdict = match winner {
            Some(winner) => Verdict::Improved { winner },
            None => Verdict::BaselineWins,
        };
        if let (Some(base), Some(best)) = (self.outcome.baseline(), self.outcome.recommended()) {
            let event = TuneEvent::Winner {
                index: best.index,
                config: best.config,
                verdict: self.outcome.verdict,
                tg_at_depth: best.measurement.triple.tg_tok_s_deepest.median,
                baseline_tg_at_depth: base.measurement.triple.tg_tok_s_deepest.median,
            };
            (self.on_event)(&event);
        }
        Ok(self.outcome)
    }
}

/// What selection reads from a measured candidate.
fn scored(m: &MeasuredCandidate) -> Scored {
    let t = &m.measurement.triple;
    Scored {
        tg_at_depth: t.tg_tok_s_deepest,
        pp: t.pp_tok_s,
        tg_depth0: t.tg_tok_s_depth0,
        threads: m.measurement.threads,
    }
}

/// The first identity in which `c` differs from the baseline's `b`.
fn identity_change(b: &Conditions, c: &Conditions) -> Option<Identity> {
    if b.model_sha256 != c.model_sha256 {
        Some(Identity::Model)
    } else if b.backend_manifest_digest != c.backend_manifest_digest {
        Some(Identity::Backend)
    } else if b.machine_profile_id != c.machine_profile_id {
        Some(Identity::Machine)
    } else {
        None
    }
}

/// Whether `c` was taken under contention: another llama.cpp process, a busy GPU, or (against
/// the baseline's `baseline`) a different power source or battery saver state. A power state
/// either side could not read is not a change.
fn contention(c: &Conditions, baseline: Option<&Conditions>) -> Option<Contention> {
    if let Some(n) = c.other_llama_processes.filter(|&n| n > 0) {
        return Some(Contention::OtherLlamaProcesses(n));
    }
    if let Some(pct) = c
        .gpu_utilization_pct
        .filter(|&pct| pct >= CONTENDED_GPU_UTILIZATION_PCT)
    {
        return Some(Contention::GpuBusy(pct));
    }
    let power = |p: &Conditions| p.power.map(|s| (s.source, s.battery_saver));
    match (baseline.and_then(power), power(c)) {
        (Some(b), Some(now)) if b != now => Some(Contention::PowerChanged),
        _ => None,
    }
}

fn store_summary(s: &Summary) -> Result<MeasurementSummary, cerebro_store::RecordError> {
    MeasurementSummary::new(s.kept, s.median, s.min, s.max, s.iqr, s.ci95_low, s.ci95_high)
}

fn store_triple(t: &Triple) -> Result<MeasurementTriple, cerebro_store::RecordError> {
    MeasurementTriple::new(
        store_summary(&t.pp_tok_s)?,
        store_summary(&t.tg_tok_s_depth0)?,
        store_summary(&t.tg_tok_s_deepest)?,
        t.deepest_depth.get(),
    )
}

/// Wall-clock milliseconds for the record's timestamp (0 for a clock set before 1970).
fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

// Windows only: these tests persist through `cerebro_store::Store::append`, which refuses to
// write off Windows (`StoreError::UnsupportedPlatform`: its single-writer share-mode lock has no
// portable equivalent).
#[cfg(all(test, windows))]
mod tests;
