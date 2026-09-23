//! The measurement record schema.
//!
//! Every type here is deliberately self-contained: this crate defines its own `ProvenanceKey` and
//! `MeasurementTriple` rather than depending on `cerebro-probe`/`cerebro-bench` for them, because
//! those crates' types describe *live* probe/parse results (not all fields serializable, shapes
//! free to evolve with the tools they wrap) while this schema is a durable on-disk format that
//! must keep reading its own old records. A caller who already has a `cerebro_bench::stats::Triple`
//! or a `cerebro_probe::backend::BackendIdentity` converts the handful of relevant fields across
//! at the call site; that conversion is cheap and keeps the two crates decoupled from this one's
//! storage format.
//!
//! # Every value here is valid by construction, including one read back from disk
//!
//! Every field of every type in this module is private. The only way to produce one is through
//! its validating constructor (`::new`, or the narrower helpers below), and the only way to
//! deserialize one is [`Record::from_wire`], which calls those same constructors on plain,
//! `deny_unknown_fields`-checked `*Wire` structs. A results file is a hostile input
//! (`docs/ARCHITECTURE.md` section 8: "a poisoned config or results file steers it into emitting
//! dangerous flags"), so a record loaded from disk is validated exactly as strictly as one built
//! in memory by a caller -- there is no second, looser path.

use std::fmt;
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::error::RecordError;

/// Current schema version. Bumped whenever a change to [`Record`]'s shape would not round-trip
/// through an old reader; carried on every record so [`crate::store::Store::load_verified`] can
/// refuse a version it does not understand instead of silently misreading it.
///
/// Bumped `1 -> 2` when [`ProvenanceKey::machine_profile_id`] became a required field: a v1 record
/// has no machine identity at all, so there is no sane default to backfill, and silently accepting
/// it as "machine unknown" is exactly the kind of guess `docs/ARCHITECTURE.md` section 6 forbids.
pub const SCHEMA_VERSION: u32 = 2;

/// Truncate an untrusted echoed value to a bounded number of characters for an error message.
///
/// A results file is a hostile input, and several fields this crate validates (a hash, an argv
/// element) are echoed into `Debug`/`Display` on rejection; without a cap, a single oversized
/// value could inflate an error message (and anything that logs it) without bound.
pub(crate) fn cap_echo(value: &str) -> String {
    value.chars().take(80).collect()
}

/// A validated lowercase 64-hex SHA-256 digest.
///
/// A bare `String` field for a hash lets a caller compare a digest computed with a different
/// hex-case convention, or a truncated/corrupted value, and get a silent `false` out of `==`
/// instead of a construction-time refusal. Every hash-shaped field in this crate is one of these.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Sha256Hex(String);

impl Sha256Hex {
    /// Parse and validate a digest, naming `field` in the error for a caller with several hash
    /// fields in scope (e.g. [`ProvenanceKey::new`]).
    ///
    /// # Errors
    /// [`RecordError::InvalidHash`] if `value` is not exactly 64 lowercase hex characters.
    pub fn parse(field: &'static str, value: impl Into<String>) -> Result<Self, RecordError> {
        let value = value.into();
        let is_valid = value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if is_valid {
            Ok(Self(value))
        } else {
            Err(RecordError::InvalidHash { field, value: cap_echo(&value) })
        }
    }

    /// The digest as lowercase hex.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Sha256Hex {
    type Error = RecordError;

    /// Generic entry point for `serde`'s `try_from` machinery, which cannot pass a field name.
    /// Callers that know which field they are validating (every constructor in this module) use
    /// [`Sha256Hex::parse`] instead, so the field name in a validation error is meaningful.
    fn try_from(value: String) -> Result<Self, RecordError> {
        Self::parse("sha256", value)
    }
}

impl From<Sha256Hex> for String {
    fn from(hash: Sha256Hex) -> Self {
        hash.0
    }
}

impl fmt::Display for Sha256Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identity of the machine, backend build, and model that a measurement was taken under.
///
/// `docs/ARCHITECTURE.md` section 6, "Provenance is part of the result key": a self-built
/// llama.cpp fork measured 3-5x slower than the official release of the same commit, so binary
/// identity is a dimension of the result, not a footnote. This struct carries the exact tuple that
/// section names -- backend build number, commit, compiler, selected `ggml-cpu-*` variant, backend
/// SHA-256, model SHA-256, and free VRAM at start -- plus a machine profile identity (below).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceKey {
    machine_profile_id: Sha256Hex,
    model_sha256: Sha256Hex,
    backend_build_number: Option<u32>,
    backend_commit: Option<String>,
    backend_compiler: Option<String>,
    backend_cpu_variant: Option<String>,
    backend_manifest_digest: Sha256Hex,
    free_vram_mib_at_start: Option<u64>,
}

impl ProvenanceKey {
    /// Build a provenance key, validating that every hash field is a well-formed SHA-256 digest.
    ///
    /// `machine_profile_id` is a SHA-256 the *caller* computes over this machine's stable identity
    /// (e.g. `cerebro_probe`'s CPU topology plus accelerator inventory) -- this crate depends on
    /// nothing from `cerebro-probe` (see the module docs), so it only carries the digest, never
    /// computes it. Without this field, a results file copied between machines, restored from a
    /// backup onto different hardware, or concatenated from two profiles would verify its hash
    /// chain cleanly while mixing measurements from unrelated machines; see
    /// [`crate::query::Staleness::MachineChanged`].
    ///
    /// # Errors
    /// [`RecordError::InvalidHash`] if `machine_profile_id`, `model_sha256`, or
    /// `backend_manifest_digest` is not exactly 64 lowercase hex characters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        machine_profile_id: impl Into<String>,
        model_sha256: impl Into<String>,
        backend_manifest_digest: impl Into<String>,
        backend_build_number: Option<u32>,
        backend_commit: Option<String>,
        backend_compiler: Option<String>,
        backend_cpu_variant: Option<String>,
        free_vram_mib_at_start: Option<u64>,
    ) -> Result<Self, RecordError> {
        Ok(Self {
            machine_profile_id: Sha256Hex::parse("machine_profile_id", machine_profile_id)?,
            model_sha256: Sha256Hex::parse("model_sha256", model_sha256)?,
            backend_manifest_digest: Sha256Hex::parse("backend_manifest_digest", backend_manifest_digest)?,
            backend_build_number,
            backend_commit,
            backend_compiler,
            backend_cpu_variant,
            free_vram_mib_at_start,
        })
    }

    /// SHA-256 of this machine's stable identity (CPU topology plus accelerator inventory), as the
    /// caller computed it. The results file is per machine profile; this is what lets
    /// [`crate::store::Store::append`] refuse a record copied in from a different machine, and lets
    /// [`crate::query::check_staleness`] report a restored or concatenated file instead of silently
    /// trusting it.
    pub fn machine_profile_id(&self) -> &Sha256Hex {
        &self.machine_profile_id
    }

    /// SHA-256 hex digest of the model file that was loaded.
    pub fn model_sha256(&self) -> &Sha256Hex {
        &self.model_sha256
    }

    /// A single SHA-256 hex digest summarizing every hashed file in the backend directory.
    pub fn backend_manifest_digest(&self) -> &Sha256Hex {
        &self.backend_manifest_digest
    }

    /// Upstream build number of the llama.cpp backend, when known.
    pub fn backend_build_number(&self) -> Option<u32> {
        self.backend_build_number
    }

    /// Short commit hash of the llama.cpp backend, when known.
    pub fn backend_commit(&self) -> Option<&str> {
        self.backend_commit.as_deref()
    }

    /// Compiler string the backend was built with, when known.
    pub fn backend_compiler(&self) -> Option<&str> {
        self.backend_compiler.as_deref()
    }

    /// Which `ggml-cpu-*` variant the loader would select on this machine, when known.
    pub fn backend_cpu_variant(&self) -> Option<&str> {
        self.backend_cpu_variant.as_deref()
    }

    /// Free VRAM in MiB measured immediately before the run that produced this record started.
    ///
    /// `docs/ARCHITECTURE.md` section 6: free VRAM swung from 431 MiB to 7285 MiB within minutes
    /// on an otherwise idle machine, so a configuration's viability depends on the headroom it was
    /// actually measured against.
    pub fn free_vram_mib_at_start(&self) -> Option<u64> {
        self.free_vram_mib_at_start
    }
}

/// Sampler parameters a measurement was taken under.
///
/// All four fields are required, never optional: `docs/ARCHITECTURE.md` Axis 0 states that
/// speculative-decoding throughput is a direct function of how peaked the sampler's output
/// distribution is, so "a speculative throughput number measured at one sampler setting does not
/// transfer to another." A record with no sampler profile would be a number nobody can safely
/// reuse.
///
/// # Equality is by quantized value, not by bit pattern
///
/// [`SamplerProfile`] is a lookup key ([`crate::query::best_config`] matches records by it), and
/// exact `f64` equality is the wrong relation for a key: a caller that arrives at the same logical
/// sampler setting by a different arithmetic route (`0.7 + 0.2` is `0.8999999999999999`, not
/// `0.9` -- confirmed on this build, unlike the more commonly cited but not-actually-off
/// `0.85 + 0.05`) would otherwise compare unequal to a record measured at the "same" value, and
/// `best_config` would silently return `None` -- which reads as "this configuration was never
/// measured," not as a float-comparison bug. [`PartialEq`] is implemented by hand over each field
/// quantized to `1e-6`, which is far finer than any difference llama.cpp's sampler can act on and
/// so can never merge two genuinely different profiles. This affects comparison only: the fields
/// stay exactly as measured for [`Serialize`] and for [`crate::line::link_hash`], so on-disk bytes
/// and the hash chain are unaffected.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SamplerProfile {
    temp: f64,
    top_k: u32,
    top_p: f64,
    min_p: f64,
}

impl SamplerProfile {
    /// Build a sampler profile, validating that every field is a value llama.cpp could actually
    /// accept.
    ///
    /// # Errors
    /// - [`RecordError::NonFiniteOrNegative`] if `temp` is not finite or is negative.
    /// - [`RecordError::OutOfUnitRange`] if `top_p` or `min_p` is not finite or not in
    ///   `0.0..=1.0`.
    pub fn new(temp: f64, top_k: u32, top_p: f64, min_p: f64) -> Result<Self, RecordError> {
        if !(temp.is_finite() && temp >= 0.0) {
            return Err(RecordError::NonFiniteOrNegative { field: "temp", value: temp });
        }
        if !(top_p.is_finite() && (0.0..=1.0).contains(&top_p)) {
            return Err(RecordError::OutOfUnitRange { field: "top_p", value: top_p });
        }
        if !(min_p.is_finite() && (0.0..=1.0).contains(&min_p)) {
            return Err(RecordError::OutOfUnitRange { field: "min_p", value: min_p });
        }
        Ok(Self { temp, top_k, top_p, min_p })
    }

    /// `--temp`.
    pub fn temp(&self) -> f64 {
        self.temp
    }

    /// `--top-k`.
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    /// `--top-p`.
    pub fn top_p(&self) -> f64 {
        self.top_p
    }

    /// `--min-p`.
    pub fn min_p(&self) -> f64 {
        self.min_p
    }

    /// The lookup key equality is actually computed over: each field quantized to `1e-6`.
    ///
    /// See [`SamplerProfile`]'s doc comment for why bit-exact `f64` equality is the wrong relation
    /// for a key. `top_k` is an exact integer already and needs no quantization.
    fn key(&self) -> (i64, u32, i64, i64) {
        let quantize = |v: f64| (v * 1_000_000.0).round() as i64;
        (quantize(self.temp), self.top_k, quantize(self.top_p), quantize(self.min_p))
    }
}

impl PartialEq for SamplerProfile {
    /// Quantized-value equality, not bit-exact `f64` equality -- see [`SamplerProfile`]'s doc
    /// comment.
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

/// A robust summary of one repetition set, mirroring `cerebro_bench::stats::Summary`'s fields so a
/// caller can convert one into the other field-by-field.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MeasurementSummary {
    kept: usize,
    median: f64,
    min: f64,
    max: f64,
    iqr: f64,
    ci95_low: f64,
    ci95_high: f64,
}

impl MeasurementSummary {
    /// Build a measurement summary, validating that it could have come from a real sample set:
    /// every numeric field finite and non-negative, at least one sample kept, and
    /// `min <= ci95_low <= median <= ci95_high <= max`.
    ///
    /// # Errors
    /// - [`RecordError::ZeroKeptSamples`] if `kept` is `0`.
    /// - [`RecordError::NonFiniteOrNegative`] if any numeric field is not finite or negative.
    /// - [`RecordError::SummaryOutOfOrder`] if the ordering above does not hold.
    #[allow(clippy::too_many_arguments)]
    pub fn new(kept: usize, median: f64, min: f64, max: f64, iqr: f64, ci95_low: f64, ci95_high: f64) -> Result<Self, RecordError> {
        if kept == 0 {
            return Err(RecordError::ZeroKeptSamples);
        }
        for (field, value) in [
            ("median", median),
            ("min", min),
            ("max", max),
            ("iqr", iqr),
            ("ci95_low", ci95_low),
            ("ci95_high", ci95_high),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(RecordError::NonFiniteOrNegative { field, value });
            }
        }
        if !(min <= ci95_low && ci95_low <= median && median <= ci95_high && ci95_high <= max) {
            return Err(RecordError::SummaryOutOfOrder { min, ci95_low, median, ci95_high, max });
        }
        Ok(Self { kept, median, min, max, iqr, ci95_low, ci95_high })
    }

    /// Number of samples the summary was computed from. Always greater than 0.
    pub fn kept(&self) -> usize {
        self.kept
    }

    /// Median of the kept samples.
    pub fn median(&self) -> f64 {
        self.median
    }

    /// Minimum of the kept samples.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Maximum of the kept samples.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Interquartile range of the kept samples.
    pub fn iqr(&self) -> f64 {
        self.iqr
    }

    /// Lower bound of the 95% confidence interval for the median.
    pub fn ci95_low(&self) -> f64 {
        self.ci95_low
    }

    /// Upper bound of the 95% confidence interval for the median.
    pub fn ci95_high(&self) -> f64 {
        self.ci95_high
    }
}

/// The three-number measurement triple a configuration must produce, per
/// `docs/ARCHITECTURE.md` section 6: "Never report a single number."
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MeasurementTriple {
    pp_tok_s: MeasurementSummary,
    tg_tok_s_depth0: MeasurementSummary,
    tg_tok_s_deepest: MeasurementSummary,
    deepest_depth: NonZeroU64,
}

impl MeasurementTriple {
    /// Build a measurement triple, validating that `deepest_depth` is a real context-decay
    /// measurement and not depth 0 again.
    ///
    /// The wire-facing parameter stays a plain `u64` (a hostile record carrying a JSON `0` must
    /// surface as [`RecordError::ZeroDeepestDepth`] through [`Record::from_wire`], not as a
    /// structural [`crate::error::StoreError::MalformedLine`]); the domain type carries the
    /// invariant as [`NonZeroU64`], the same way `cerebro_bench::stats::Triple::deepest_depth`
    /// already does for the equivalent live-probe field, so [`crate::query::best_config`]'s
    /// depth comparison cannot silently compare against a meaningless zero.
    ///
    /// # Errors
    /// [`RecordError::ZeroDeepestDepth`] if `deepest_depth` is `0`.
    pub fn new(
        pp_tok_s: MeasurementSummary,
        tg_tok_s_depth0: MeasurementSummary,
        tg_tok_s_deepest: MeasurementSummary,
        deepest_depth: u64,
    ) -> Result<Self, RecordError> {
        let deepest_depth = NonZeroU64::new(deepest_depth).ok_or(RecordError::ZeroDeepestDepth)?;
        Ok(Self { pp_tok_s, tg_tok_s_depth0, tg_tok_s_deepest, deepest_depth })
    }

    /// Prefill throughput at depth 0.
    pub fn pp_tok_s(&self) -> MeasurementSummary {
        self.pp_tok_s
    }

    /// Decode throughput at depth 0.
    pub fn tg_tok_s_depth0(&self) -> MeasurementSummary {
        self.tg_tok_s_depth0
    }

    /// Decode throughput at the deepest depth measured (the target working context depth).
    pub fn tg_tok_s_deepest(&self) -> MeasurementSummary {
        self.tg_tok_s_deepest
    }

    /// The `n_depth` value `tg_tok_s_deepest` was measured at. Always greater than 0.
    pub fn deepest_depth(&self) -> NonZeroU64 {
        self.deepest_depth
    }
}

/// Which lossless/quality check a [`QualityGateOutcome`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityGateKind {
    /// `llama-results` prefill-logit NMSE against a reference (the primary, lossless gate).
    Nmse,
    /// `llama-perplexity --kl-divergence` against a reference-logits file (the secondary gate).
    KlDivergence,
}

/// Result of running a configuration's quality gate.
///
/// `docs/ARCHITECTURE.md` section 7: throughput without a correctness gate is how you ship a fast,
/// broken product, so every record that reached measurement should carry one of these -- but a
/// record's [`Record::quality_gate`] is `Option` because a screening-tier record may not have run
/// the gate yet, and `crate::query::best_config` treats that absence the same as an explicit
/// failure: not eligible to be recommended.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualityGateOutcome {
    gate: QualityGateKind,
    value: f64,
    threshold: f64,
}

impl QualityGateOutcome {
    /// Build a quality gate outcome. `passed` is *not* a parameter: it is always
    /// [`value <= threshold`](QualityGateOutcome::passed), derived rather than stored, so a
    /// hand-edited or corrupted record can never claim a `passed` flag that contradicts its own
    /// `value`/`threshold` (both gates in `docs/ARCHITECTURE.md` section 7 -- NMSE and
    /// KL-divergence -- are "lower is better" checks against a fixed threshold).
    ///
    /// **This is provenance, not policy.** `threshold` is recorded exactly as the run was judged
    /// against *at the time it was measured* -- it is read straight out of a results file, which
    /// `docs/ARCHITECTURE.md` section 8 names as adversary number one. A record carrying
    /// `{value: 1e-2, threshold: 1e9}` makes `passed()` derive to `true` with a threshold nobody
    /// ever ran the gate against; deriving the flag only stops the narrower forgery of `passed`
    /// disagreeing with its own `value`/`threshold`, it does **not** stop the threshold itself from
    /// being moved. The trusted gate and tolerance (`docs/ARCHITECTURE.md` section 7 fixes both
    /// project-wide) are policy the caller holds, never data a results file supplies, so
    /// eligibility decisions such as [`crate::query::best_config`] take a
    /// [`crate::query::QualityPolicy`] and never call [`QualityGateOutcome::passed`] to decide what
    /// gets recommended. `passed()` remains here for display -- e.g. an operator-facing table
    /// showing what a record's own numbers say -- not for that decision.
    ///
    /// # Errors
    /// [`RecordError::NonFiniteOrNegative`] if `value` or `threshold` is not finite or negative.
    pub fn new(gate: QualityGateKind, value: f64, threshold: f64) -> Result<Self, RecordError> {
        if !(value.is_finite() && value >= 0.0) {
            return Err(RecordError::NonFiniteOrNegative { field: "quality_gate.value", value });
        }
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(RecordError::NonFiniteOrNegative { field: "quality_gate.threshold", value: threshold });
        }
        Ok(Self { gate, value, threshold })
    }

    /// Which check produced this outcome.
    pub fn gate(&self) -> QualityGateKind {
        self.gate
    }

    /// The measured value (e.g. NMSE or KL-divergence).
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The pass/fail threshold the value was compared against.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether `value` cleared `threshold`. Derived, never stored: see [`QualityGateOutcome::new`].
    pub fn passed(&self) -> bool {
        self.value <= self.threshold
    }
}

/// Result of a sustained soak test, when one was run.
///
/// `docs/ARCHITECTURE.md` section 6: "VRAM trend is a pass criterion. A soak passes only if free
/// VRAM has a non-negative trend over its final third." `vram_trend_non_negative` records that
/// check directly rather than making a reader re-derive it from raw samples this crate does not
/// store.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SoakOutcome {
    passed: bool,
    vram_trend_non_negative: bool,
    duration_s: u64,
    notes: Option<String>,
}

impl SoakOutcome {
    /// Build a soak outcome, refusing the one combination `docs/ARCHITECTURE.md` section 6 makes
    /// impossible: passing despite a negative VRAM trend.
    ///
    /// # Errors
    /// [`RecordError::InconsistentSoakOutcome`] if `passed` is `true` while
    /// `vram_trend_non_negative` is `false`.
    pub fn new(passed: bool, vram_trend_non_negative: bool, duration_s: u64, notes: Option<String>) -> Result<Self, RecordError> {
        if passed && !vram_trend_non_negative {
            return Err(RecordError::InconsistentSoakOutcome);
        }
        Ok(Self { passed, vram_trend_non_negative, duration_s, notes })
    }

    /// Whether the soak passed overall.
    pub fn passed(&self) -> bool {
        self.passed
    }

    /// Whether free VRAM had a non-negative trend over the soak's final third.
    pub fn vram_trend_non_negative(&self) -> bool {
        self.vram_trend_non_negative
    }

    /// How long the soak ran, in seconds.
    pub fn duration_s(&self) -> u64 {
        self.duration_s
    }

    /// Free-form operator notes (e.g. what failed, or what was observed).
    pub fn notes(&self) -> Option<&str> {
        self.notes.as_deref()
    }
}

/// One measured configuration: everything needed to reproduce it, judge it, and rank it against
/// others, plus the caller-supplied timestamp it was recorded at.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Record {
    schema_version: u32,
    provenance: ProvenanceKey,
    sampler: SamplerProfile,
    argv: Vec<String>,
    measurement: MeasurementTriple,
    soak: Option<SoakOutcome>,
    quality_gate: Option<QualityGateOutcome>,
    recorded_at_unix_ms: u64,
}

/// `argv` must be non-empty (a record names the command line it measured) and free of embedded
/// NUL bytes (which cannot appear in anything that was actually spawned). Other control
/// characters, such as an embedded newline in a multi-line prompt value, are legitimate argv
/// content and are accepted -- see [`Record::measured_argv_untrusted`] for why that is safe.
fn validate_argv(argv: &[String]) -> Result<(), RecordError> {
    if argv.is_empty() {
        return Err(RecordError::EmptyArgv);
    }
    if let Some(position) = argv.iter().position(|a| a.contains('\0')) {
        return Err(RecordError::NulByteInArgv { position });
    }
    Ok(())
}

impl Record {
    /// Build a record, stamping it with the current [`SCHEMA_VERSION`].
    ///
    /// # Errors
    /// [`RecordError::EmptyArgv`] if `argv` is empty, or [`RecordError::NulByteInArgv`] if any
    /// element contains an embedded NUL byte.
    pub fn new(
        provenance: ProvenanceKey,
        sampler: SamplerProfile,
        argv: Vec<String>,
        measurement: MeasurementTriple,
        soak: Option<SoakOutcome>,
        quality_gate: Option<QualityGateOutcome>,
        recorded_at_unix_ms: u64,
    ) -> Result<Self, RecordError> {
        validate_argv(&argv)?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            provenance,
            sampler,
            argv,
            measurement,
            soak,
            quality_gate,
            recorded_at_unix_ms,
        })
    }

    /// Schema version this record was written under. Always [`SCHEMA_VERSION`] for a record built
    /// through [`Record::new`] or read back through [`Record::from_wire`].
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Machine/backend/model identity this measurement was taken under.
    pub fn provenance(&self) -> &ProvenanceKey {
        &self.provenance
    }

    /// Sampler profile this measurement was taken under.
    pub fn sampler(&self) -> &SamplerProfile {
        &self.sampler
    }

    /// The exact argv that was measured, in spawn order, **as read from disk: untrusted**.
    ///
    /// A results file is a hostile input (`docs/ARCHITECTURE.md` section 8), and the hash chain
    /// documented on [`crate::store`] does not authenticate it -- only detects naive corruption.
    /// **Never pass this to a process spawn.** It exists as descriptive provenance (what was
    /// measured) and for equality comparison against an argv rebuilt from typed flags. A caller
    /// that wants to reproduce a recommended configuration must rebuild it through
    /// `cerebro_flags::ArgvBuilder`'s closed allowlist and may compare the result against this
    /// value, but must not spawn this value directly.
    pub fn measured_argv_untrusted(&self) -> &[String] {
        &self.argv
    }

    /// The measurement triple, with confidence intervals.
    pub fn measurement(&self) -> &MeasurementTriple {
        &self.measurement
    }

    /// Sustained soak-test outcome, if one was run for this configuration.
    pub fn soak(&self) -> Option<&SoakOutcome> {
        self.soak.as_ref()
    }

    /// Quality gate outcome, if the gate was run for this configuration.
    pub fn quality_gate(&self) -> Option<&QualityGateOutcome> {
        self.quality_gate.as_ref()
    }

    /// Wall-clock time this record was created, as Unix milliseconds supplied by the caller.
    ///
    /// This crate never reads the system clock itself -- the same "no ambient state" rule
    /// `cerebro-bench` applies to its bootstrap seed -- so a record's timestamp is exactly what the
    /// caller passed in and reproducible in tests without mocking time.
    pub fn recorded_at_unix_ms(&self) -> u64 {
        self.recorded_at_unix_ms
    }

    /// Build a record from its plain, structurally-checked wire representation, routing every
    /// field through the same validating constructor a caller building one in memory would use.
    ///
    /// This is the *only* path [`crate::line::decode_line`] uses to turn a parsed line into a
    /// [`Record`]; there is no second, looser deserialization path for data read from disk.
    ///
    /// # Errors
    /// The first [`RecordError`] encountered, in field order.
    pub(crate) fn from_wire(wire: RecordWire) -> Result<Self, RecordError> {
        let p = wire.provenance;
        let provenance = ProvenanceKey::new(
            p.machine_profile_id,
            p.model_sha256,
            p.backend_manifest_digest,
            p.backend_build_number,
            p.backend_commit,
            p.backend_compiler,
            p.backend_cpu_variant,
            p.free_vram_mib_at_start,
        )?;

        let s = wire.sampler;
        let sampler = SamplerProfile::new(s.temp, s.top_k, s.top_p, s.min_p)?;

        let m = wire.measurement;
        let measurement = MeasurementTriple::new(
            MeasurementSummary::from_wire(m.pp_tok_s)?,
            MeasurementSummary::from_wire(m.tg_tok_s_depth0)?,
            MeasurementSummary::from_wire(m.tg_tok_s_deepest)?,
            m.deepest_depth,
        )?;

        let soak = wire
            .soak
            .map(|s| SoakOutcome::new(s.passed, s.vram_trend_non_negative, s.duration_s, s.notes))
            .transpose()?;
        let quality_gate = wire.quality_gate.map(|q| QualityGateOutcome::new(q.gate, q.value, q.threshold)).transpose()?;

        Record::new(provenance, sampler, wire.argv, measurement, soak, quality_gate, wire.recorded_at_unix_ms)
    }
}

impl MeasurementSummary {
    /// Build from a plain wire struct, routing through [`MeasurementSummary::new`].
    pub(crate) fn from_wire(wire: MeasurementSummaryWire) -> Result<Self, RecordError> {
        Self::new(wire.kept, wire.median, wire.min, wire.max, wire.iqr, wire.ci95_low, wire.ci95_high)
    }
}

// --- Plain, structurally-checked wire representations -----------------------------------------
//
// Every struct below mirrors a domain type field-for-field with unvalidated plain types, and
// derives `Deserialize` with `deny_unknown_fields` so an extra or misspelled key is a structural
// parse failure rather than a silently-ignored field. None of these are `pub`: the only way to
// turn one into its domain type is [`Record::from_wire`] (and, transitively, the `from_wire`
// helpers beside it), which is the sole place semantic validation happens for data read from disk.

/// Wire form of [`ProvenanceKey`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProvenanceWire {
    machine_profile_id: String,
    model_sha256: String,
    backend_build_number: Option<u32>,
    backend_commit: Option<String>,
    backend_compiler: Option<String>,
    backend_cpu_variant: Option<String>,
    backend_manifest_digest: String,
    free_vram_mib_at_start: Option<u64>,
}

/// Wire form of [`SamplerProfile`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SamplerWire {
    temp: f64,
    top_k: u32,
    top_p: f64,
    min_p: f64,
}

/// Wire form of [`MeasurementSummary`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MeasurementSummaryWire {
    kept: usize,
    median: f64,
    min: f64,
    max: f64,
    iqr: f64,
    ci95_low: f64,
    ci95_high: f64,
}

/// Wire form of [`MeasurementTriple`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MeasurementTripleWire {
    pp_tok_s: MeasurementSummaryWire,
    tg_tok_s_depth0: MeasurementSummaryWire,
    tg_tok_s_deepest: MeasurementSummaryWire,
    deepest_depth: u64,
}

/// Wire form of [`SoakOutcome`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SoakOutcomeWire {
    passed: bool,
    vram_trend_non_negative: bool,
    duration_s: u64,
    notes: Option<String>,
}

/// Wire form of [`QualityGateOutcome`]. Deliberately has no `passed` field: accepting one from
/// disk would let a hostile or corrupted record claim a `passed` flag inconsistent with its own
/// `value`/`threshold`, which is exactly what [`QualityGateOutcome::passed`] being derived
/// prevents. A record carrying an on-disk `passed` key is rejected as an unknown field.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualityGateOutcomeWire {
    gate: QualityGateKind,
    value: f64,
    threshold: f64,
}

/// Wire form of [`Record`]. [`crate::line::decode_line`] checks `schema_version` with a separate,
/// permissive peek at the raw JSON *before* parsing into this struct -- an old or new schema's
/// shape may not satisfy this strict, `deny_unknown_fields` struct at all -- then parses into this
/// struct and validates it field-by-field through [`Record::from_wire`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecordWire {
    pub(crate) schema_version: u32,
    provenance: ProvenanceWire,
    sampler: SamplerWire,
    argv: Vec<String>,
    measurement: MeasurementTripleWire,
    soak: Option<SoakOutcomeWire>,
    quality_gate: Option<QualityGateOutcomeWire>,
    recorded_at_unix_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ProvenanceKey {
        ProvenanceKey::new("c".repeat(64), "a".repeat(64), "b".repeat(64), Some(10751), None, None, None, Some(7113)).unwrap()
    }

    fn sampler() -> SamplerProfile {
        SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap()
    }

    fn summary(median: f64) -> MeasurementSummary {
        MeasurementSummary::new(3, median, median - 1.0, median + 1.0, 1.0, median - 0.5, median + 0.5).unwrap()
    }

    fn triple() -> MeasurementTriple {
        MeasurementTriple::new(summary(40.0), summary(3.0), summary(2.5), 128).unwrap()
    }

    #[test]
    fn provenance_rejects_wrong_length_hash() {
        let err = ProvenanceKey::new("c".repeat(64), "deadbeef", "b".repeat(64), None, None, None, None, None).unwrap_err();
        assert_eq!(err, RecordError::InvalidHash { field: "model_sha256", value: "deadbeef".into() });
    }

    #[test]
    fn provenance_rejects_uppercase_hash() {
        let bad = "A".repeat(64);
        let err = ProvenanceKey::new("c".repeat(64), bad.clone(), "b".repeat(64), None, None, None, None, None).unwrap_err();
        assert_eq!(err, RecordError::InvalidHash { field: "model_sha256", value: bad });
    }

    #[test]
    fn provenance_rejects_wrong_length_machine_hash() {
        let err = ProvenanceKey::new("short", "a".repeat(64), "b".repeat(64), None, None, None, None, None).unwrap_err();
        assert_eq!(err, RecordError::InvalidHash { field: "machine_profile_id", value: "short".into() });
    }

    #[test]
    fn provenance_accepts_real_sha256_hex() {
        // A real digest, not just 64 arbitrary hex characters, exercises the same code path a
        // caller hashing an actual model file would hit.
        use sha2::{Digest, Sha256};
        let digest = format!("{:x}", Sha256::digest(b"model bytes"));
        assert!(ProvenanceKey::new("c".repeat(64), digest, "b".repeat(64), None, None, None, None, None).is_ok());
    }

    #[test]
    fn invalid_hash_error_caps_the_echoed_value() {
        let huge = "z".repeat(500);
        let err = ProvenanceKey::new("c".repeat(64), huge, "b".repeat(64), None, None, None, None, None).unwrap_err();
        match err {
            RecordError::InvalidHash { value, .. } => assert_eq!(value.len(), 80),
            other => panic!("expected InvalidHash, got {other:?}"),
        }
    }

    #[test]
    fn sha256_hex_try_from_string_validates() {
        assert!(Sha256Hex::try_from("a".repeat(64)).is_ok());
        assert!(Sha256Hex::try_from("A".repeat(64)).is_err(), "uppercase must not be accepted at the type level");
        assert!(Sha256Hex::try_from("short".to_string()).is_err());
    }

    #[test]
    fn sampler_rejects_negative_temp() {
        assert_eq!(
            SamplerProfile::new(-0.1, 40, 0.9, 0.05).unwrap_err(),
            RecordError::NonFiniteOrNegative { field: "temp", value: -0.1 }
        );
    }

    #[test]
    fn sampler_rejects_top_p_above_one() {
        assert_eq!(
            SamplerProfile::new(0.7, 40, 1.5, 0.05).unwrap_err(),
            RecordError::OutOfUnitRange { field: "top_p", value: 1.5 }
        );
    }

    #[test]
    fn sampler_rejects_nan_min_p() {
        let err = SamplerProfile::new(0.7, 40, 0.9, f64::NAN).unwrap_err();
        assert!(matches!(err, RecordError::OutOfUnitRange { field: "min_p", .. }));
    }

    #[test]
    fn sampler_equality_survives_caller_side_float_arithmetic() {
        let recorded = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let computed = SamplerProfile::new(0.7, 40, 0.7 + 0.2, 0.05).unwrap();
        // Without this, the test would be a tautology on a future compiler/platform that happened
        // to make the arithmetic exact. (`0.85 + 0.05` is the more commonly cited example, but it
        // rounds back to exactly `0.9` on this build -- measured, not assumed, per CLAIMS.md.)
        assert_ne!(0.7_f64 + 0.2, 0.9, "the fixture must really differ in bits for this test to mean anything");
        assert_eq!(recorded, computed, "the same logical sampler setting, reached by different arithmetic, must compare equal");
        assert_ne!(
            recorded,
            SamplerProfile::new(0.7, 40, 0.91, 0.05).unwrap(),
            "a genuinely different top_p must still compare unequal"
        );
    }

    #[test]
    fn sampler_accepts_boundary_values() {
        // 0.0 and 1.0 are both legal proportions; an off-by-one in the range check would reject
        // one of these two.
        assert!(SamplerProfile::new(0.0, 1, 0.0, 0.0).is_ok());
        assert!(SamplerProfile::new(1.0, 1, 1.0, 1.0).is_ok());
    }

    #[test]
    fn summary_rejects_zero_kept() {
        assert_eq!(MeasurementSummary::new(0, 1.0, 0.5, 1.5, 1.0, 0.5, 1.5).unwrap_err(), RecordError::ZeroKeptSamples);
    }

    #[test]
    fn summary_rejects_non_finite_field() {
        let err = MeasurementSummary::new(3, f64::NAN, 0.0, 2.0, 1.0, 0.5, 1.5).unwrap_err();
        assert!(matches!(err, RecordError::NonFiniteOrNegative { field: "median", .. }));
    }

    #[test]
    fn summary_rejects_out_of_order_values() {
        // median (5.0) is below min (10.0): cannot have come from a real sample set.
        let err = MeasurementSummary::new(3, 5.0, 10.0, 20.0, 1.0, 12.0, 18.0).unwrap_err();
        assert!(matches!(err, RecordError::SummaryOutOfOrder { .. }));
    }

    #[test]
    fn summary_accepts_degenerate_single_point() {
        // min == ci95_low == median == ci95_high == max is legal: a single repeated sample.
        assert!(MeasurementSummary::new(1, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0).is_ok());
    }

    #[test]
    fn triple_rejects_zero_deepest_depth() {
        assert_eq!(
            MeasurementTriple::new(summary(40.0), summary(3.0), summary(2.5), 0).unwrap_err(),
            RecordError::ZeroDeepestDepth
        );
    }

    #[test]
    fn quality_gate_passed_is_derived_from_value_and_threshold() {
        let passing = QualityGateOutcome::new(QualityGateKind::Nmse, 1e-7, 3.3e-3).unwrap();
        assert!(passing.passed());
        let failing = QualityGateOutcome::new(QualityGateKind::Nmse, 1e-2, 3.3e-3).unwrap();
        assert!(!failing.passed());
        // Exactly at the threshold passes: the check is `<=`.
        let boundary = QualityGateOutcome::new(QualityGateKind::Nmse, 3.3e-3, 3.3e-3).unwrap();
        assert!(boundary.passed());
    }

    #[test]
    fn quality_gate_rejects_non_finite_value() {
        let err = QualityGateOutcome::new(QualityGateKind::Nmse, f64::NAN, 3.3e-3).unwrap_err();
        assert!(matches!(err, RecordError::NonFiniteOrNegative { field: "quality_gate.value", .. }));
    }

    #[test]
    fn soak_rejects_passed_with_negative_vram_trend() {
        assert_eq!(SoakOutcome::new(true, false, 3600, None).unwrap_err(), RecordError::InconsistentSoakOutcome);
    }

    #[test]
    fn soak_accepts_failed_with_negative_vram_trend() {
        // A soak can fail for reasons other than VRAM (a crash, a hang); only the reverse
        // combination (passed=true with a negative trend) is impossible.
        assert!(SoakOutcome::new(false, false, 120, Some("crashed at 90s".into())).is_ok());
    }

    #[test]
    fn record_rejects_empty_argv() {
        let err = Record::new(provenance(), sampler(), vec![], triple(), None, None, 0).unwrap_err();
        assert_eq!(err, RecordError::EmptyArgv);
    }

    #[test]
    fn record_rejects_nul_byte_in_argv() {
        let err = Record::new(provenance(), sampler(), vec!["ok".into(), "bad\0arg".into()], triple(), None, None, 0).unwrap_err();
        assert_eq!(err, RecordError::NulByteInArgv { position: 1 });
    }

    #[test]
    fn record_accepts_embedded_newline_in_argv() {
        // A newline is a legitimate character inside a measured argument value (a multi-line
        // system prompt or grammar), and this argv is never spawned directly -- see
        // `measured_argv_untrusted`'s docs -- so only NUL is rejected, not every control char.
        let r = Record::new(provenance(), sampler(), vec!["--grammar".into(), "a\nb".into()], triple(), None, None, 0).unwrap();
        assert_eq!(r.measured_argv_untrusted()[1], "a\nb");
    }

    #[test]
    fn record_stamps_current_schema_version() {
        let r = Record::new(provenance(), sampler(), vec!["llama-server".into()], triple(), None, None, 1).unwrap();
        assert_eq!(r.schema_version(), SCHEMA_VERSION);
    }

    #[test]
    fn from_wire_rejects_invalid_nested_fields() {
        let wire = RecordWire {
            schema_version: SCHEMA_VERSION,
            provenance: ProvenanceWire {
                machine_profile_id: "c".repeat(64),
                model_sha256: "xyz".into(),
                backend_build_number: None,
                backend_commit: None,
                backend_compiler: None,
                backend_cpu_variant: None,
                backend_manifest_digest: "b".repeat(64),
                free_vram_mib_at_start: None,
            },
            sampler: SamplerWire { temp: 0.7, top_k: 40, top_p: 0.9, min_p: 0.05 },
            argv: vec!["llama-server".into()],
            measurement: MeasurementTripleWire {
                pp_tok_s: MeasurementSummaryWire { kept: 3, median: 40.0, min: 39.0, max: 41.0, iqr: 1.0, ci95_low: 39.5, ci95_high: 40.5 },
                tg_tok_s_depth0: MeasurementSummaryWire { kept: 3, median: 3.0, min: 2.0, max: 4.0, iqr: 1.0, ci95_low: 2.5, ci95_high: 3.5 },
                tg_tok_s_deepest: MeasurementSummaryWire { kept: 3, median: 2.5, min: 2.0, max: 3.0, iqr: 1.0, ci95_low: 2.2, ci95_high: 2.8 },
                deepest_depth: 128,
            },
            soak: None,
            quality_gate: None,
            recorded_at_unix_ms: 0,
        };
        let err = Record::from_wire(wire).unwrap_err();
        assert_eq!(err, RecordError::InvalidHash { field: "model_sha256", value: "xyz".into() });
    }
}
