//! Durable, tamper-evident storage for Cerebro's measurement results and chosen configurations.
//!
//! One [`store::Store`] is an append-only JSON Lines file for a single machine profile, one line
//! per measured configuration. `docs/ARCHITECTURE.md` sections 6 and 8 are why this crate exists
//! rather than a plain `serde_json::to_writer` per record:
//!
//! - **Provenance is part of every record** ([`record::ProvenanceKey`]): a self-built llama.cpp
//!   fork measured 3-5x slower than the official release of the same commit, so a throughput
//!   number that does not name the exact binary and model it came from is not safe to compare
//!   against another.
//! - **The sampler profile is required, not optional** ([`record::SamplerProfile`]): Axis 0 of the
//!   search space conditions Axis 1 (speculation), so a speculative throughput number measured
//!   under one sampler setting does not transfer to another.
//! - **Every record is valid by construction, on disk or in memory** ([`record`] module docs): a
//!   results file is a hostile input, so loading routes every record through the same validating
//!   constructors a caller building one directly would use.
//! - **Every record carries a hash chain link** ([`store`] module docs): loading detects the first
//!   broken link rather than silently accepting or skipping corruption, though this is explicitly
//!   *not* a defense against a write-capable attacker -- `docs/CLAIMS.md` forbids shipping a
//!   control as security it is not.
//! - **A crash mid-append leaves a detectable, excludable torn tail**, never a file that silently
//!   loses or misattributes an earlier record.
//! - **Two appenders racing fail one of them, never fork the chain** ([`store`] module docs'
//!   concurrency section).
//!
//! [`query`] answers the two questions Cerebro's tuner and its results UI actually ask: which
//! configuration to recommend, and whether a previously-recorded recommendation is still trustworthy
//! for the machine as it stands today.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod line;

pub mod error;
pub mod query;
pub mod record;
pub mod store;

pub use error::{RecordError, StoreError};
pub use query::{best_config, check_staleness, QualityPolicy, Staleness};
pub use record::{
    MeasurementSummary, MeasurementTriple, ProvenanceKey, QualityGateKind, QualityGateOutcome, Record, SamplerProfile, Sha256Hex,
    SoakOutcome, SCHEMA_VERSION,
};
pub use store::{AppendOutcome, LoadOutcome, Store, TornReason, TornTail, MAX_FILE_BYTES, MAX_RECORDS, MAX_RECORD_BYTES};
