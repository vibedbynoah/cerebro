//! Typed failures for a `llama-bench` run.
//!
//! Three families, distinguished because a caller acts on them differently:
//!
//! - **Refusals before anything runs** ([`RunError::Plan`], [`RunError::Preflight`],
//!   [`RunError::Argv`], [`RunError::ModelPathNotUtf8`]): the request itself cannot be run as asked.
//!   Retrying the same request changes nothing.
//! - **Provenance failures** ([`RunError::Provenance`]): the run could not be keyed to the exact
//!   binary, model and machine state it would measure, so it is not started (or, for the digest,
//!   not reported). A number without its key is not comparable to anything.
//! - **Infrastructure errors** (everything [`RunError::is_infrastructure`] is true for): the tool
//!   was launched but did not produce a complete, trustworthy result: a non-zero exit, a timeout,
//!   no JSON, malformed JSON, records that do not match what was asked for. Each carries the tail
//!   of the tool's stderr. None of them is ever turned into an empty or partial result.

use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

use crate::error::{ParseError, TripleError};

/// A [`super::RunPlan`] that cannot be run as asked.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PlanError {
    /// Fewer kept repetitions than [`crate::stats::MIN_KEPT_SAMPLES`]; the statistics refuse
    /// fewer, so the run would be wasted.
    #[error("{kept} kept repetitions requested; at least {min} are needed for a summary")]
    TooFewRepetitions {
        /// Kept repetitions requested.
        kept: u32,
        /// The minimum.
        min: u32,
    },
    /// More repetitions (kept plus the discarded first) than [`crate::stats::MAX_REPETITIONS`].
    #[error("{total} repetitions requested; the limit is {max}")]
    TooManyRepetitions {
        /// Kept repetitions plus the discarded first one.
        total: u64,
        /// The limit.
        max: u64,
    },
    /// `-p 0` would measure no prefill, leaving the triple without its first member.
    #[error("prompt tokens must be at least 1")]
    ZeroPromptTokens,
    /// `-n 0` would measure no decode, leaving the triple without its decode members.
    #[error("generated tokens must be at least 1")]
    ZeroGenTokens,
    /// The depth list lacks 0, so there is no prefill@0 or decode@0.
    #[error("the depth list must include 0 (prefill and decode at depth 0)")]
    NoDepthZero,
    /// The depth list has no depth above 0, so decode at working depth is never measured
    /// (`docs/CLAIMS.md`: a depth-0-only number overstates real use).
    #[error("the depth list must include a depth above 0 (decode at working depth)")]
    NoDepthSweep,
    /// More distinct depths than [`super::MAX_DEPTHS`].
    #[error("{count} depths requested; the limit is {max}")]
    TooManyDepths {
        /// Distinct depths requested.
        count: usize,
        /// The limit.
        max: usize,
    },
    /// The `--delay` before each test, summed over the plan's tests, is not below the timeout, so
    /// the run could only end in a timeout.
    #[error("delays totalling {delay_total:?} leave no time inside the {timeout:?} timeout")]
    DelayExceedsTimeout {
        /// The delay times the number of tests.
        delay_total: Duration,
        /// The plan's timeout.
        timeout: Duration,
    },
    /// A timeout of zero, or above [`super::MAX_TIMEOUT`].
    #[error("timeout {got:?} must be above zero and at most {max:?}")]
    TimeoutOutOfRange {
        /// The timeout requested.
        got: Duration,
        /// The largest accepted.
        max: Duration,
    },
}

/// The run could not be keyed to what it measures.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProvenanceError {
    /// The model file could not be opened, read or stat'd for hashing.
    #[error("model file {path} could not be hashed: {cause}")]
    ModelUnreadable {
        /// The model path.
        path: PathBuf,
        /// The OS error, as text.
        cause: String,
    },
    /// The model file's size or modification time changed while it was being hashed, so the
    /// digest describes no single version of it.
    #[error("model file {path} changed while it was being hashed")]
    ModelChangedWhileHashing {
        /// The model path.
        path: PathBuf,
    },
    /// Hashing the backend directory failed (only needed when integrity verification is off and
    /// preflight therefore hashed nothing).
    #[error("backend directory could not be hashed: {0}")]
    BackendUnhashable(cerebro_probe::ProbeError),
    /// CPU topology could not be established.
    #[error("CPU topology could not be probed: {0}")]
    Cpu(cerebro_probe::ProbeError),
    /// The GPU inventory could not be read, or free VRAM could not be sampled.
    #[error("GPU state could not be probed: {0}")]
    Gpu(cerebro_probe::ProbeError),
}

/// Why a `llama-bench` run produced no result.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RunError {
    /// The run plan is invalid.
    #[error(transparent)]
    Plan(#[from] PlanError),

    /// Preflight refused the run (elevated, unsafe or unverified backend, bad model file, ...).
    /// See [`cerebro_supervisor::PreflightError::governing_protection`] for the setting that
    /// governs it, when one does.
    #[error("preflight refused the benchmark run: {0}")]
    Preflight(#[from] cerebro_supervisor::PreflightError),

    /// A configuration flag is not valid for llama-bench, duplicates another, or breaks a hard
    /// cross-flag rule. `-m` is supplied by the run itself, so a configuration carrying
    /// `Flag::Model` is refused here as a duplicate.
    #[error("benchmark configuration refused: {0}")]
    Argv(#[from] cerebro_flags::EmitError),

    /// The preflighted model path is not valid UTF-8, so it cannot be placed in the argv the flag
    /// allowlist builds.
    #[error("model path {0:?} is not valid UTF-8")]
    ModelPathNotUtf8(PathBuf),

    /// The run could not be keyed to the binary, model and machine state it measures.
    #[error(transparent)]
    Provenance(#[from] ProvenanceError),

    /// The supervised spawn failed, or polling the child failed.
    #[error("llama-bench could not be run: {0}")]
    Spawn(#[from] cerebro_supervisor::SpawnError),

    /// llama-bench did not finish within the plan's timeout; its whole process tree was killed.
    #[error(
        "llama-bench did not finish within {timeout:?} and was killed (exit confirmed: \
         {confirmed_gone}){}",
        tail_suffix(stderr_tail)
    )]
    TimedOut {
        /// The plan's timeout.
        timeout: Duration,
        /// Whether the OS confirmed the process gone after the kill.
        confirmed_gone: bool,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// llama-bench exited with a non-zero status (a model that failed to load, an out-of-memory
    /// allocation, a rejected flag, ...).
    #[error("llama-bench exited with {}{}", exit_text(*code), tail_suffix(stderr_tail))]
    BackendExited {
        /// The exit code, when the OS reported one.
        code: Option<i32>,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// llama-bench's output could not be read completely: a pipe error, a pipe still open after
    /// exit, or output that is not UTF-8.
    #[error("llama-bench output could not be read: {cause}{}", tail_suffix(stderr_tail))]
    OutputUnreadable {
        /// What went wrong.
        cause: String,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// llama-bench wrote more than [`cerebro_supervisor::MAX_STDOUT_CAPTURE_BYTES`] to stdout;
    /// a truncated document is never parsed.
    #[error("llama-bench wrote more than {limit} bytes of output{}", tail_suffix(stderr_tail))]
    OutputTooLarge {
        /// The capture cap.
        limit: usize,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// llama-bench exited successfully but wrote no JSON records.
    #[error("llama-bench exited successfully but produced no JSON results{}", tail_suffix(stderr_tail))]
    MissingJson {
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// llama-bench's stdout is not a valid `-o json` document.
    #[error("llama-bench output is not valid benchmark JSON: {source}{}", tail_suffix(stderr_tail))]
    MalformedJson {
        /// The parse failure.
        source: ParseError,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// The records parse but do not match what this run asked for (wrong repetition count, wrong
    /// model, wrong number of records or configurations, missing depth).
    #[error("llama-bench results do not match the run: {detail}{}", tail_suffix(stderr_tail))]
    UnexpectedRecords {
        /// What did not match.
        detail: String,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },

    /// The records do not yield the pp / tg / tg-at-depth triple the plan guarantees.
    #[error("{source}{}", tail_suffix(stderr_tail))]
    IncompleteTriple {
        /// Which members are missing and why.
        source: TripleError,
        /// The last stderr lines.
        stderr_tail: Vec<String>,
    },
}

impl RunError {
    /// Whether this is an infrastructure error: llama-bench was launched (or its launch failed)
    /// and no complete, trustworthy result came back. False for refusals of the request itself
    /// and for provenance failures.
    pub fn is_infrastructure(&self) -> bool {
        matches!(
            self,
            Self::Spawn(_)
                | Self::TimedOut { .. }
                | Self::BackendExited { .. }
                | Self::OutputUnreadable { .. }
                | Self::OutputTooLarge { .. }
                | Self::MissingJson { .. }
                | Self::MalformedJson { .. }
                | Self::UnexpectedRecords { .. }
                | Self::IncompleteTriple { .. }
        )
    }

    /// The last lines llama-bench wrote to stderr, for the variants that ran it; empty otherwise.
    pub fn stderr_tail(&self) -> &[String] {
        match self {
            Self::TimedOut { stderr_tail, .. }
            | Self::BackendExited { stderr_tail, .. }
            | Self::OutputUnreadable { stderr_tail, .. }
            | Self::OutputTooLarge { stderr_tail, .. }
            | Self::MissingJson { stderr_tail }
            | Self::MalformedJson { stderr_tail, .. }
            | Self::UnexpectedRecords { stderr_tail, .. }
            | Self::IncompleteTriple { stderr_tail, .. } => stderr_tail,
            _ => &[],
        }
    }
}

fn exit_text(code: Option<i32>) -> String {
    match code {
        // Windows reports NTSTATUS crash codes (0xC0000005 ...) as negative i32s; hex reads better.
        Some(c) if c < 0 => format!("status {c:#010x}"),
        Some(c) => format!("exit code {c}"),
        None => "no exit code".to_string(),
    }
}

fn tail_suffix(tail: &[String]) -> String {
    if tail.is_empty() {
        "; stderr was empty".to_string()
    } else {
        format!("; last stderr lines: {}", tail.join(" | "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_carries_the_exit_code_and_the_stderr_tail() {
        let err = RunError::BackendExited {
            code: Some(1),
            stderr_tail: vec!["error: failed to load model".into(), "main: exiting".into()],
        };
        assert_eq!(
            err.to_string(),
            "llama-bench exited with exit code 1; last stderr lines: error: failed to load \
             model | main: exiting"
        );
        let crash = RunError::BackendExited {
            code: Some(0xC000_0005_u32 as i32),
            stderr_tail: Vec::new(),
        };
        assert_eq!(
            crash.to_string(),
            "llama-bench exited with status 0xc0000005; stderr was empty"
        );
    }

    #[test]
    fn only_launched_runs_are_infrastructure_errors() {
        assert!(RunError::MissingJson { stderr_tail: Vec::new() }.is_infrastructure());
        assert!(!RunError::Plan(PlanError::NoDepthZero).is_infrastructure());
        assert!(!RunError::ModelPathNotUtf8(PathBuf::from("x")).is_infrastructure());
        assert!(RunError::Plan(PlanError::NoDepthZero).stderr_tail().is_empty());
    }
}
