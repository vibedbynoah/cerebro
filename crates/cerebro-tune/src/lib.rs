//! Cerebro's autotuner, v1: measure a few llama.cpp configurations for a model and a workload,
//! and recommend one only if it is demonstrably faster than llama.cpp's own automatic settings.
//!
//! Tuning a 30B MoE that does not fit VRAM measured about 6x faster than llama.cpp's defaults on
//! the development laptop, while for a small model that fits, the defaults beat hand-tuned flags.
//! So nothing here assumes a configuration is better: every candidate is measured, and the rules
//! in [`select`] never pick one whose confidence interval is not entirely above the baseline's, or
//! one demonstrably slower than the baseline at prefill or at depth-0 decode.
//!
//! - [`gguf`] reads the model's shape and per-block weight bytes from its header ([`ModelFacts`]).
//! - [`candidate`] generates each stage's candidates ([`Planner`]): llama.cpp's automatic settings
//!   (the `--fit` baseline, always measured first), full GPU offload when the fit estimate allows,
//!   `-ncmoe` steps for a MoE or dense FFN offload steps for a dense model that does not fit,
//!   q8_0 versus f16 KV with flash attention, and thread / P-core mask variants.
//! - [`runner`] measures a candidate ([`CandidateRunner`]; [`LlamaBench`] drives `cerebro-bench`'s
//!   preflighted llama-bench), cancellably ([`CancelToken`]), with the [`Conditions`] it ran under.
//! - [`select`] picks the winner by decode tok/s at the workload's depth, guarded by prefill and
//!   depth-0 decode, with a bootstrap-CI tie rule preferring fewer threads.
//! - [`tune()`] runs the stages, appends every measurement to `cerebro-store` with the sampler
//!   profile as it completes, checks that measurements are comparable (same model, build and
//!   machine; contention set aside), re-measures the baseline when a candidate won, and streams
//!   [`TuneEvent`]s to a callback for the CLI and the app. The recommendation is served with
//!   [`CandidateConfig::server_flags`].
//!
//! Screening only: candidates are measured with short llama-bench runs (first repetition
//! discarded, three numbers per configuration). The sustained soak that `docs/CLAIMS.md` requires
//! before quoting a throughput figure is not part of v1, so records carry no soak outcome.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod candidate;
pub mod error;
pub mod gguf;
pub mod runner;
pub mod select;
pub mod tune;

pub use candidate::{
    CandidateConfig, KvCache, MachineFacts, Placement, Planner, ThreadPlan, Vram, Workload,
    SERVER_DEFAULT_FIT_CTX,
};
pub use error::{GgufError, Identity, MeasureError, TuneError};
pub use gguf::{BlockBytes, ModelFacts};
pub use runner::{CancelToken, CandidateRunner, Conditions, LlamaBench, Measurement, TestSamples};
pub use select::{choose, eligible, Choice, Guard, Scored};
pub use tune::{
    resolve_depth, tune, CandidateResult, Contention, Excluded, Exclusion, FailedCandidate,
    FailureKind, MeasuredCandidate, ScreenSettings, Stage, TuneEvent, TuneOutcome, TuneRequest,
    Verdict, CONTENDED_GPU_UTILIZATION_PCT,
};
