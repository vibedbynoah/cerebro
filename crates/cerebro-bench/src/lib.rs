//! Parsing and nonparametric statistics for llama-bench measurement records.
//!
//! Cerebro's search over Axes 2-4 (`docs/ARCHITECTURE.md` section 4) is driven by `llama-bench`,
//! and the whole search is only as trustworthy as the arithmetic done on its output. This crate is
//! that arithmetic, split into the two concerns section 6 of the architecture doc treats as
//! load-bearing:
//!
//! - [`parse`] turns `llama-bench -o json` text into typed [`parse::BenchRecord`]s, tolerating
//!   fields it does not know about but refusing a document whose sample arrays are structurally
//!   broken.
//! - [`stats`] turns a record's raw samples into a [`stats::Summary`] — always discarding the
//!   first repetition, computing median/min/max/IQR and a seeded percentile-bootstrap confidence
//!   interval for the median — and groups records by configuration into a [`stats::Triple`] of
//!   prefill@0, decode@0 and decode@deepest-depth, or a typed [`error::TripleError`] naming
//!   exactly what is missing.
//!
//! Both modules are pure: no file I/O, no process spawning, nothing that depends on wall-clock
//! time or ambient state. Every function's output is determined entirely by its arguments, which
//! is what makes the exact-number tests in each module possible at all.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod parse;
pub mod stats;

pub use error::{ClassifyError, ParseError, StatsError, TripleError, TripleMember, TripleProblem};
pub use parse::{parse_records, BenchRecord, MAX_RECORDS};
pub use stats::{
    assemble_triples, classify, summarize, summarize_with_iterations, Classified, ConfigKey,
    Summary, TestKind, Triple, DEFAULT_BOOTSTRAP_ITERATIONS, MAX_BOOTSTRAP_ITERATIONS,
    MAX_REPETITIONS, MIN_KEPT_SAMPLES, NON_CONFIG_EXTRA_FIELDS,
};
