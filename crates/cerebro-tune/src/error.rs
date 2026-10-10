//! Typed failures for reading a model, measuring a candidate and running a tune.

use thiserror::Error;

/// A GGUF header that could not be read into [`crate::ModelFacts`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GgufError {
    /// The file could not be opened or read.
    #[error("model file could not be read: {0}")]
    Io(String),
    /// The file does not start with the GGUF magic.
    #[error("not a GGUF file")]
    NotGguf,
    /// A GGUF version other than 2 or 3.
    #[error("GGUF version {0} is not supported (2 and 3 are)")]
    UnsupportedVersion(u32),
    /// The file ends inside its header.
    #[error("the GGUF header is truncated")]
    Truncated,
    /// A count or length beyond what any real model carries.
    #[error("GGUF {what} {value} exceeds the limit of {limit}")]
    TooLarge {
        /// Which count or length.
        what: &'static str,
        /// The value found.
        value: u64,
        /// The limit.
        limit: u64,
    },
    /// A metadata value type code the GGUF specification does not define.
    #[error("unknown GGUF metadata value type {0}")]
    BadValueType(u32),
    /// Arrays nested deeper than any real model uses.
    #[error("GGUF metadata arrays are nested too deeply")]
    TooDeep,
    /// A key or string that is not UTF-8.
    #[error("a GGUF string is not valid UTF-8")]
    NotUtf8,
    /// A metadata key the tuner needs is absent.
    #[error("GGUF metadata key {0} is missing")]
    MissingKey(String),
    /// A tensor description that cannot be right (dimensions, offsets, alignment).
    #[error("malformed GGUF tensor table: {0}")]
    BadTensor(String),
    /// One part of a split model; its sizes would describe only that part.
    #[error("split GGUF models ({count} parts) are not supported by the tuner yet")]
    SplitModel {
        /// `split.count`.
        count: u64,
    },
}

/// Why one candidate produced no measurement.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MeasureError {
    /// The tune was cancelled while this candidate was waiting or running; its process tree, if
    /// any, was killed. `confirmed_gone` false means the OS did not confirm the kill, so a
    /// llama-bench may still be running (possible when job containment is switched off).
    #[error("cancelled (exit confirmed: {confirmed_gone})")]
    Cancelled {
        /// Whether the OS confirmed the process gone (true when none had started).
        confirmed_gone: bool,
    },
    /// The candidate itself failed: it launched but gave no trustworthy result (out of memory,
    /// a crash, a timeout, output that does not match the request). The bench error is kept
    /// whole so the tuner can tell a timeout whose kill was not confirmed from an ordinary
    /// failure; other candidates may still run.
    #[error(transparent)]
    CandidateFailed(cerebro_bench::RunError),
    /// A refusal that would stop every candidate alike (preflight, provenance, a configuration
    /// the flag allowlist refuses): the tune stops.
    #[error(transparent)]
    Run(cerebro_bench::RunError),
    /// The measurement could not be converted to the store's provenance key.
    #[error("the measurement could not be keyed for the store: {0}")]
    Key(#[from] cerebro_store::RecordError),
}

/// Which identity changed between the baseline and a later measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Identity {
    /// The model file's SHA-256: the file was replaced mid-tune.
    Model,
    /// The backend manifest digest: the llama.cpp build changed mid-tune.
    Backend,
    /// The machine profile (CPU topology and GPU models).
    Machine,
}

impl Identity {
    /// Short lowercase name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            Identity::Model => "model file",
            Identity::Backend => "llama.cpp build",
            Identity::Machine => "machine profile",
        }
    }
}

/// Why a tune produced no outcome.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TuneError {
    /// The screening plan is invalid (repetitions, sizes, timeout).
    #[error("invalid screening plan: {0}")]
    Plan(#[from] cerebro_bench::run::PlanError),
    /// The model's context is too short for the screening sizes at any depth above zero.
    #[error(
        "the model's {context}-token context leaves no room for a decode test at depth beyond \
         {prompt} prompt + {generated} generated tokens"
    )]
    ContextTooShort {
        /// The model's training context.
        context: u32,
        /// Prompt tokens per prefill test.
        prompt: u32,
        /// Tokens generated per decode test.
        generated: u32,
    },
    /// A depth override of zero: decode at working depth would not be measured.
    #[error("the decode depth must be above zero")]
    ZeroDepth,
    /// The baseline (llama.cpp's own automatic settings) could not be measured, so no candidate
    /// can be shown not to be slower than it, and nothing is recommended.
    #[error("the llama.cpp automatic-settings baseline could not be measured: {0}")]
    BaselineFailed(#[source] cerebro_bench::RunError),
    /// A measurement was keyed to a different model, llama.cpp build or machine than the
    /// baseline's, so the two are not comparable and nothing measured from here on would be.
    #[error("the {} changed during the tune; measurements are no longer comparable", .0.name())]
    ConditionsChanged(Identity),
    /// A llama-bench that was killed (timeout or cancel) was not confirmed gone, so it may still
    /// be running; the tune stopped rather than measure another candidate beside it.
    #[error("a llama-bench process may still be running, so the tune stopped: {0}")]
    ProcessNotConfirmedGone(String),
    /// A candidate's flags could not be built (an internal error: the generated values are
    /// fixed and valid).
    #[error("candidate flags could not be built: {0}")]
    Flags(#[from] cerebro_flags::ValueError),
    /// A refusal that stops every candidate alike (see [`MeasureError::Run`]).
    #[error(transparent)]
    Measure(MeasureError),
    /// A measured candidate could not be turned into a store record.
    #[error("a measurement could not be recorded: {0}")]
    Record(#[from] cerebro_store::RecordError),
    /// A measured candidate could not be persisted. The tune stops: a result that is not on disk
    /// cannot be recommended later.
    #[error("a measurement could not be persisted: {0}")]
    Store(#[from] cerebro_store::StoreError),
}
