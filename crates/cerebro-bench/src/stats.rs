//! Nonparametric statistics over llama-bench raw per-repetition samples.
//!
//! `docs/ARCHITECTURE.md` section 6 sets three binding rules this module encodes as types rather
//! than as caller discipline:
//!
//! 1. **The first repetition is always discarded**, unconditionally — measured on this machine at
//!    124 t/s (rep 1) against 173 t/s (rep 2) with warmup already on. [`summarize`] takes the full
//!    raw sample slice and drops the first element itself; there is no entry point that skips it.
//! 2. **A configuration is never characterized by one number.** [`assemble_triples`] produces
//!    `pp_tok_s`@depth0, `tg_tok_s`@depth0 and `tg_tok_s`@deepest-depth together, or refuses with a
//!    [`TripleError`] naming exactly which member is missing — never a partial [`Triple`].
//! 3. **Confidence intervals come from a seeded, local RNG**, not a global one. [`summarize`] takes
//!    the seed as a parameter; nothing here reads from `rand::thread_rng` or any other ambient
//!    source, so the same input always reproduces the same interval.

use std::collections::BTreeMap;
use std::num::NonZeroU64;

use crate::error::{ClassifyError, StatsError, TripleError, TripleMember, TripleProblem};
use crate::parse::BenchRecord;

/// Number of bootstrap resamples [`summarize`] uses. Exposed so a caller who wants a cheaper or
/// more precise interval can call [`summarize_with_iterations`] directly instead.
pub const DEFAULT_BOOTSTRAP_ITERATIONS: usize = 2000;

/// The fewest kept samples (after discarding the first repetition) a summary can be computed from.
///
/// Below this, min/max/median are numerically defined but statistically meaningless, and a
/// bootstrap over them would report false precision. Kept as a named constant, and `pub`, because
/// it appears in both the check and the [`StatsError::InsufficientSamples`] error it raises, and a
/// caller deciding how many repetitions to request from llama-bench needs to know it too.
pub const MIN_KEPT_SAMPLES: usize = 3;

/// Most repetitions [`summarize_with_iterations`] (and, by extension, [`crate::parse::parse_records`])
/// accepts for a single record.
///
/// `docs/ARCHITECTURE.md` section 6 runs `-r 7` to `-r 9`; a record anywhere near this limit is
/// corrupt rather than a legitimately large sweep, and the bootstrap's cost is `iterations * kept *
/// log(kept)` per triple member, so an unbounded repetition count is an unbounded hang, not just an
/// unbounded allocation.
pub const MAX_REPETITIONS: usize = 1_000;

/// Upper bound on the number of bootstrap resamples [`summarize_with_iterations`] accepts.
///
/// Beyond this the percentile estimate has already converged and only CPU time and memory keep
/// growing; without a cap, an absurd caller-supplied count (say, `usize::MAX / 4`) would make
/// `Vec::with_capacity` panic on allocation failure or overflow, which this crate's "no panic
/// outside tests" rule forbids.
pub const MAX_BOOTSTRAP_ITERATIONS: usize = 100_000;

/// A robust summary of one repetition set, after unconditionally discarding the first repetition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Number of samples the summary was computed from (raw count minus the discarded first rep).
    pub kept: usize,
    /// Median of the kept samples.
    pub median: f64,
    /// Minimum of the kept samples.
    pub min: f64,
    /// Maximum of the kept samples.
    pub max: f64,
    /// Interquartile range: `Q3 - Q1`, where `Q1`/`Q3` are the medians of the lower and upper
    /// halves of the sorted kept samples, with the overall median excluded from both halves when
    /// `kept` is odd (the Moore-McCabe convention; Tukey's hinges would instead include the
    /// overall median in both halves, giving a narrower range).
    pub iqr: f64,
    /// Lower bound of the 95% percentile-bootstrap confidence interval for the median.
    pub ci95_low: f64,
    /// Upper bound of the 95% percentile-bootstrap confidence interval for the median.
    pub ci95_high: f64,
}

/// Which of the three throughput numbers in a [`Triple`] a given member is.
///
/// `deepest_depth` on [`Triple`] records `n_depth` for [`TripleMember::DecodeDeepest`]; the other
/// two members are pinned to depth 0 by definition and need no depth field of their own.
///
/// Both `tg_tok_s_*` members are filled only from *pure* decode records (`n_prompt == 0`). A
/// combined `-pg` record classifies as decode per [`classify`] (the brief: `n_gen > 0`), but its
/// `samples_ts` is `(n_prompt + n_gen) / time` — combined throughput, not decode throughput — so
/// it is excluded from eligibility for these members even though it carries the right label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triple {
    /// Prefill throughput at depth 0.
    pub pp_tok_s: Summary,
    /// Decode throughput at depth 0.
    pub tg_tok_s_depth0: Summary,
    /// Decode throughput at the deepest depth measured for this configuration.
    pub tg_tok_s_deepest: Summary,
    /// The `n_depth` value `tg_tok_s_deepest` was measured at. Always greater than 0: a
    /// configuration whose only decode depth is 0 has no context-decay member to report, and
    /// [`assemble_triples`] refuses it with [`TripleProblem::NoDepthSweep`] rather than build a
    /// `Triple` whose "deepest" member is silently identical to `tg_tok_s_depth0`.
    pub deepest_depth: NonZeroU64,
}

/// Whether a record is a prefill test, a decode test, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestKind {
    /// `n_prompt > 0 && n_gen == 0`.
    Prefill,
    /// `n_gen > 0`, regardless of `n_prompt` (a combined pp+tg test still measures decode).
    Decode,
}

/// A record's test shape: what it measures, and at which context depth.
///
/// [`classify`] returns this instead of a bare [`TestKind`] so that grouping code (`assemble_triples`
/// and any future caller) never needs to go back to the source [`BenchRecord`] just to learn the
/// depth or token counts next to a classification — the two travel together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classified {
    /// Prefill or decode.
    pub kind: TestKind,
    /// `n_depth` the test ran at.
    pub depth: u64,
    /// `n_prompt` of the record (nonzero for prefill, and for a combined `-pg` decode test).
    pub n_prompt: u64,
    /// `n_gen` of the record (nonzero for every decode test).
    pub n_gen: u64,
}

/// Classify a record as prefill or decode, attaching its depth and token counts.
///
/// `index` is the record's position in whatever list the caller is classifying; it is carried
/// through to [`ClassifyError::NeitherPrefillNorDecode`] unchanged; `classify` has no list position
/// of its own to discover, so every error it returns already names the caller's real position —
/// there is no placeholder for a caller to discard and replace.
///
/// # Errors
/// Returns [`ClassifyError::NeitherPrefillNorDecode`] for a record with `n_prompt == 0 &&
/// n_gen == 0`, which llama-bench never emits for a real test but which a hand-edited or
/// corrupted results file could contain.
pub fn classify(index: usize, record: &BenchRecord) -> Result<Classified, ClassifyError> {
    let kind = if record.n_gen > 0 {
        TestKind::Decode
    } else if record.n_prompt > 0 {
        TestKind::Prefill
    } else {
        return Err(ClassifyError::NeitherPrefillNorDecode { index });
    };
    Ok(Classified { kind, depth: record.n_depth, n_prompt: record.n_prompt, n_gen: record.n_gen })
}

/// A small, local, deterministic PRNG (splitmix64) for the percentile bootstrap.
///
/// This is deliberately not `rand::thread_rng()` or any other ambient/global source: the whole
/// point of taking a seed as an argument is that the same seed and the same samples always
/// reproduce the same confidence interval, which a global generator cannot promise. splitmix64 is
/// Sebastiano Vigna's public-domain generator; it is not cryptographically secure, which is fine
/// here — this RNG never sees untrusted input, only drives array-index selection.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniformly-ish distributed index in `0..len`. `len` is always the kept-sample count here
    /// (at least [`MIN_KEPT_SAMPLES`]), so the modulo bias this introduces is negligible and never
    /// worth a full rejection-sampling implementation for a benchmarking tool.
    /// # Panics
    /// Never at runtime as this module calls it: `len` is always `sorted_kept.len()` in
    /// [`bootstrap_median_ci`], which is only ever called after `summarize_with_iterations` has
    /// already enforced `kept.len() >= MIN_KEPT_SAMPLES` (a compile-time-adjacent invariant, not a
    /// runtime assertion — there is no code path left that could call this with `len == 0`).
    fn index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }
}

/// Median of an already-sorted, nonempty slice.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// `(Q1, Q3)` of an already-sorted, nonempty slice, Moore-McCabe convention: the median of the
/// lower half and the median of the upper half, excluding the overall median from both halves
/// when the length is odd. (This is *not* Tukey's hinges, which would include the overall median
/// in both halves and so give a narrower range for odd-length input.)
fn quartiles(sorted: &[f64]) -> (f64, f64) {
    let n = sorted.len();
    let mid = n / 2;
    let (lower, upper) = if n % 2 == 0 {
        (&sorted[..mid], &sorted[mid..])
    } else {
        (&sorted[..mid], &sorted[mid + 1..])
    };
    (median_sorted(lower), median_sorted(upper))
}

/// Linearly-interpolated percentile of an already-sorted, nonempty slice (numpy's default
/// `"linear"` method), so an independent reimplementation used only to pin test expectations is a
/// two-line function rather than a second copy of this one.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}

/// Percentile-bootstrap 95% confidence interval for the median of `sorted_kept`.
///
/// Resamples are drawn, with replacement, by index into `sorted_kept` (not the original unsorted
/// order) — an arbitrary but fixed convention, documented here because it is exactly the kind of
/// detail that silently changes which value a given RNG draw resolves to and would otherwise make
/// this function impossible to reproduce independently for a test.
fn bootstrap_median_ci(sorted_kept: &[f64], seed: u64, iterations: usize) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let n = sorted_kept.len();
    let mut boot_medians: Vec<f64> = Vec::with_capacity(iterations);
    let mut resample: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..iterations {
        resample.clear();
        for _ in 0..n {
            resample.push(sorted_kept[rng.index(n)]);
        }
        resample.sort_by(f64::total_cmp);
        boot_medians.push(median_sorted(&resample));
    }
    boot_medians.sort_by(f64::total_cmp);
    (percentile_sorted(&boot_medians, 2.5), percentile_sorted(&boot_medians, 97.5))
}

/// Summarize raw per-repetition throughput samples, discarding the first repetition and using
/// [`DEFAULT_BOOTSTRAP_ITERATIONS`] resamples for the confidence interval.
///
/// # Errors
/// [`StatsError::InsufficientSamples`] if fewer than [`MIN_KEPT_SAMPLES`] samples remain after
/// discarding the first repetition.
pub fn summarize(raw_samples: &[f64], seed: u64) -> Result<Summary, StatsError> {
    summarize_with_iterations(raw_samples, seed, DEFAULT_BOOTSTRAP_ITERATIONS)
}

/// As [`summarize`], but with an explicit bootstrap iteration count.
///
/// This function is a trust boundary in its own right, not just a downstream consumer of
/// [`crate::parse::parse_records`]'s checks: it is `pub` and takes a bare `&[f64]`, so it must be
/// safe to call directly on samples that never passed through `parse_records` at all (built in
/// code, deserialized some other way, or mutated after parsing).
///
/// # Errors
/// - [`StatsError::BootstrapIterationsOutOfRange`] if `iterations` is `0` or exceeds
///   [`MAX_BOOTSTRAP_ITERATIONS`].
/// - [`StatsError::TooManySamples`] if `raw_samples.len()` exceeds [`MAX_REPETITIONS`].
/// - [`StatsError::InvalidSample`] if any raw sample (including the one about to be discarded) is
///   not a positive, finite number.
/// - [`StatsError::InsufficientSamples`] if fewer than [`MIN_KEPT_SAMPLES`] samples remain after
///   discarding the first repetition.
pub fn summarize_with_iterations(
    raw_samples: &[f64],
    seed: u64,
    iterations: usize,
) -> Result<Summary, StatsError> {
    if iterations == 0 || iterations > MAX_BOOTSTRAP_ITERATIONS {
        return Err(StatsError::BootstrapIterationsOutOfRange {
            got: iterations,
            max: MAX_BOOTSTRAP_ITERATIONS,
        });
    }

    if raw_samples.len() > MAX_REPETITIONS {
        return Err(StatsError::TooManySamples { got: raw_samples.len(), max: MAX_REPETITIONS });
    }

    // Validated before the discard: a NaN/zero/negative/infinite value in the rep about to be
    // dropped still means the record it came from is corrupt, not merely that the discarded rep
    // was itself unusable.
    if let Some((position, value)) =
        raw_samples.iter().enumerate().find(|(_, v)| !(v.is_finite() && **v > 0.0))
    {
        return Err(StatsError::InvalidSample { position, value: value.to_string() });
    }

    // The first repetition is always discarded, unconditionally, regardless of how many samples
    // were supplied — this is the one rule in this module a caller cannot opt out of.
    let raw_len = raw_samples.len();
    let kept: Vec<f64> = raw_samples.iter().copied().skip(1).collect();
    if kept.len() < MIN_KEPT_SAMPLES {
        return Err(StatsError::InsufficientSamples {
            raw: raw_len,
            kept: kept.len(),
            required: MIN_KEPT_SAMPLES,
        });
    }

    let mut sorted = kept;
    sorted.sort_by(f64::total_cmp);

    let median = median_sorted(&sorted);
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let (q1, q3) = quartiles(&sorted);
    let (ci95_low, ci95_high) = bootstrap_median_ci(&sorted, seed, iterations);

    Ok(Summary { kept: sorted.len(), median, min, max, iqr: q3 - q1, ci95_low, ci95_high })
}

/// Fields llama-bench echoes on every record that describe the *result* of a test (or vary between
/// otherwise-identical runs, like a wall-clock timestamp) rather than the configuration under
/// test. These are excluded from [`ConfigKey::extra`] specifically so they never split two records
/// of the same configuration into two configurations — every record in a single `-r 4` sweep has
/// its own `test_time`, and each test shape has its own `avg_ns`/`avg_ts`/`stddev_ns`/`stddev_ts`.
///
/// `pub` so a caller reading [`ConfigKey::extra`] can see, without guessing, exactly which fields
/// were deliberately left out of the configuration identity and why.
pub const NON_CONFIG_EXTRA_FIELDS: &[&str] =
    &["test_time", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts"];

/// Grouping key for "everything llama-bench echoes about how it was invoked", excluding the test
/// dimensions (`n_prompt`/`n_gen`/`n_depth`), the raw samples, and the result-only fields in
/// [`NON_CONFIG_EXTRA_FIELDS`].
///
/// Two records with equal keys are the same configuration measured at different test shapes —
/// exactly the records [`assemble_triples`] must combine into one [`Triple`]. [`ConfigKey::extra`]
/// is what makes this identity complete by construction: every field llama-bench echoes that this
/// struct does not have a named field for still ends up in the key (as canonical JSON text, since
/// `serde_json::Value` is not itself `Ord`), so a field this crate has never heard of can only
/// ever split two configurations apart, never silently merge them.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConfigKey {
    build_commit: String,
    build_number: u32,
    backends: Option<String>,
    model_filename: String,
    model_type: Option<String>,
    model_size: Option<u64>,
    n_params: Option<u64>,
    n_gpu_layers: Option<i64>,
    n_cpu_moe: Option<i64>,
    type_k: Option<String>,
    type_v: Option<String>,
    flash_attn: Option<i64>,
    n_threads: Option<u32>,
    cpu_mask: Option<String>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    devices: Option<String>,
    /// `(field name, canonical JSON text of its value)`, sorted by field name (inherited from
    /// iterating a `BTreeMap`), for every field on the source record's [`BenchRecord::extra`] that
    /// is not in [`NON_CONFIG_EXTRA_FIELDS`].
    extra: Vec<(String, String)>,
}

impl From<&BenchRecord> for ConfigKey {
    fn from(r: &BenchRecord) -> Self {
        Self {
            build_commit: r.build_commit.clone(),
            build_number: r.build_number,
            backends: r.backends.clone(),
            model_filename: r.model_filename.clone(),
            model_type: r.model_type.clone(),
            model_size: r.model_size,
            n_params: r.n_params,
            n_gpu_layers: r.n_gpu_layers,
            n_cpu_moe: r.n_cpu_moe,
            type_k: r.type_k.clone(),
            type_v: r.type_v.clone(),
            flash_attn: r.flash_attn,
            n_threads: r.n_threads,
            cpu_mask: r.cpu_mask.clone(),
            n_batch: r.n_batch,
            n_ubatch: r.n_ubatch,
            devices: r.devices.clone(),
            extra: r
                .extra
                .iter()
                .filter(|(k, _)| !NON_CONFIG_EXTRA_FIELDS.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect(),
        }
    }
}

impl ConfigKey {
    /// Short commit hash of the llama.cpp build that produced this configuration's records.
    pub fn build_commit(&self) -> &str {
        &self.build_commit
    }
    /// Upstream build number of the llama.cpp build.
    pub fn build_number(&self) -> u32 {
        self.build_number
    }
    /// Comma-separated backend names the binary was compiled with.
    pub fn backends(&self) -> Option<&str> {
        self.backends.as_deref()
    }
    /// Path to the model file, exactly as passed to `-m`.
    pub fn model_filename(&self) -> &str {
        &self.model_filename
    }
    /// Human-readable model type string llama-bench prints.
    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }
    /// Model file size in bytes.
    pub fn model_size(&self) -> Option<u64> {
        self.model_size
    }
    /// Model parameter count.
    pub fn n_params(&self) -> Option<u64> {
        self.n_params
    }
    /// `-ngl` value this configuration was measured with.
    pub fn n_gpu_layers(&self) -> Option<i64> {
        self.n_gpu_layers
    }
    /// `-ncmoe` value this configuration was measured with.
    pub fn n_cpu_moe(&self) -> Option<i64> {
        self.n_cpu_moe
    }
    /// `-ctk` value this configuration was measured with.
    pub fn type_k(&self) -> Option<&str> {
        self.type_k.as_deref()
    }
    /// `-ctv` value this configuration was measured with.
    pub fn type_v(&self) -> Option<&str> {
        self.type_v.as_deref()
    }
    /// `-fa` value this configuration was measured with.
    pub fn flash_attn(&self) -> Option<i64> {
        self.flash_attn
    }
    /// `-t` value this configuration was measured with.
    pub fn n_threads(&self) -> Option<u32> {
        self.n_threads
    }
    /// `-C`/`--cpu-mask` value this configuration was measured with.
    pub fn cpu_mask(&self) -> Option<&str> {
        self.cpu_mask.as_deref()
    }
    /// `-b` value this configuration was measured with.
    pub fn n_batch(&self) -> Option<u32> {
        self.n_batch
    }
    /// `-ub` value this configuration was measured with.
    pub fn n_ubatch(&self) -> Option<u32> {
        self.n_ubatch
    }
    /// `-dev` value this configuration was measured with.
    pub fn devices(&self) -> Option<&str> {
        self.devices.as_deref()
    }
    /// Configuration fields llama-bench echoed that this crate does not model by name, as
    /// `(field name, canonical JSON text of its value)` pairs sorted by field name. Excludes the
    /// result-only fields in [`NON_CONFIG_EXTRA_FIELDS`] (`test_time`, `avg_ns`, ...).
    pub fn extra(&self) -> &[(String, String)] {
        &self.extra
    }
}

/// Per-configuration bucket of every record that could plausibly fill one of the three triple
/// members, collected without yet deciding which (if any) is *the* one for that member.
///
/// Deferring that decision to [`pick`] after all records are grouped is what makes triple
/// assembly independent of input order: a last-wins `Option::replace` can never silently discard a
/// competing record, because there is no single slot to overwrite until `pick` runs.
#[derive(Default)]
struct Bucket<'a> {
    /// Every prefill (`n_prompt > 0 && n_gen == 0`) record at `n_depth == 0`.
    prefill_depth0: Vec<&'a BenchRecord>,
    /// Every *pure* decode record (`n_gen > 0 && n_prompt == 0`), at any depth.
    ///
    /// A combined `-pg` record (`n_prompt > 0 && n_gen > 0`) still classifies as
    /// [`TestKind::Decode`] per the brief, but its `samples_ts` is `(n_prompt + n_gen) / time` —
    /// combined throughput, not decode throughput — so it must never compete for a `tg_tok_s`
    /// triple member. Excluding it here, at collection time, means it is not merely "not picked":
    /// it never becomes a candidate in the first place, so it cannot manufacture a false
    /// [`TripleProblem::AmbiguousTest`] against a real decode record either.
    decode: Vec<&'a BenchRecord>,
}

/// Resolve one triple member from its candidate records.
///
/// Exhaustive over the candidate count: zero is [`TripleProblem::NoSuchTest`], exactly one is the
/// answer, and two or more is [`TripleProblem::AmbiguousTest`] naming the first two competitors —
/// never a silent pick of either. This is what removes all dependence on the order records
/// appeared in the input.
fn pick<'a>(candidates: &[&'a BenchRecord], member: TripleMember) -> Result<&'a BenchRecord, TripleProblem> {
    match candidates {
        [] => Err(TripleProblem::NoSuchTest(member)),
        [one] => Ok(*one),
        [a, b, ..] => Err(TripleProblem::AmbiguousTest {
            member,
            first: (a.n_prompt, a.n_gen),
            second: (b.n_prompt, b.n_gen),
        }),
    }
}

/// Summarize the record `pick` already resolved for one triple member.
fn resolve_member(
    picked: Result<&BenchRecord, TripleProblem>,
    member: TripleMember,
    seed: u64,
) -> Result<Summary, TripleProblem> {
    let record = picked?;
    summarize(&record.samples_ts, seed).map_err(|source| TripleProblem::Unsummarizable(member, source))
}

/// Resolve the [`TripleMember::DecodeDeepest`] member from a configuration's pure decode records.
///
/// Only a record with `n_depth > 0` may fill this member. `docs/ARCHITECTURE.md` section 6: the
/// member exists precisely because "a depth-0-only number systematically overstates real use", so
/// a configuration whose only decode depth is 0 (no `-d` sweep, or one whose deeper record was
/// dropped) must never yield a `Triple` whose "deepest" number is quietly the depth-0 number again
/// — it yields [`TripleProblem::NoDepthSweep`] instead.
fn resolve_deepest(decode: &[&BenchRecord], seed: u64) -> Result<(Summary, NonZeroU64), TripleProblem> {
    // Computing the max directly over `NonZeroU64` (rather than over raw `u64` filtered to `> 0`)
    // means there is no `d > 0` fact to re-establish with an `unwrap`/`expect` a few lines down —
    // the zero-depth records are simply never candidates for this `max` at all.
    let deepest_depth: Option<NonZeroU64> =
        decode.iter().filter_map(|r| NonZeroU64::new(r.n_depth)).max();

    let depth = match deepest_depth {
        Some(d) => d,
        None if decode.is_empty() => return Err(TripleProblem::NoSuchTest(TripleMember::DecodeDeepest)),
        None => return Err(TripleProblem::NoDepthSweep),
    };

    let deepest: Vec<&BenchRecord> = decode.iter().filter(|r| r.n_depth == depth.get()).copied().collect();
    let record = pick(&deepest, TripleMember::DecodeDeepest)?;
    let summary = summarize(&record.samples_ts, seed)
        .map_err(|source| TripleProblem::Unsummarizable(TripleMember::DecodeDeepest, source))?;
    Ok((summary, depth))
}

/// Group records by configuration and assemble each configuration's measurement triple.
///
/// `docs/ARCHITECTURE.md` section 6: every configuration is characterized by prefill tok/s at
/// depth 0, decode tok/s at depth 0, and decode tok/s at the deepest depth actually measured for
/// it — never fewer numbers, and never a partial triple presented as if it were complete. A
/// configuration that cannot produce all three yields `Err(TripleError)` naming every missing or
/// unsummarizable member, not a `Triple` with a member silently defaulted.
///
/// The returned map is a `BTreeMap` (rather than a `HashMap`) purely so iteration order — and
/// therefore anything printed from it — is deterministic without the caller having to sort.
///
/// # Errors
/// Returns [`ClassifyError`] if any record is neither a prefill nor a decode test; a results file
/// containing such a record is treated as untrustworthy as a whole rather than silently ignoring
/// the one bad record and estimating from the rest.
pub fn assemble_triples(
    records: &[BenchRecord],
    seed: u64,
) -> Result<BTreeMap<ConfigKey, Result<Triple, TripleError>>, ClassifyError> {
    let mut buckets: BTreeMap<ConfigKey, Bucket<'_>> = BTreeMap::new();

    for (index, record) in records.iter().enumerate() {
        let classified = classify(index, record)?;
        let bucket = buckets.entry(ConfigKey::from(record)).or_default();
        match classified.kind {
            TestKind::Prefill => {
                if classified.depth == 0 {
                    bucket.prefill_depth0.push(record);
                }
            }
            TestKind::Decode => {
                // Only a pure decode record may stand in for "decode tok/s" — see the doc on
                // `Bucket::decode` for why a combined `-pg` record is excluded here rather than
                // merely never picked.
                if classified.n_prompt == 0 {
                    bucket.decode.push(record);
                }
            }
        }
    }

    let mut out = BTreeMap::new();
    for (key, bucket) in buckets {
        let depth0: Vec<&BenchRecord> =
            bucket.decode.iter().filter(|r| r.n_depth == 0).copied().collect();

        let pp = resolve_member(
            pick(&bucket.prefill_depth0, TripleMember::PrefillDepth0),
            TripleMember::PrefillDepth0,
            seed,
        );
        let tg0 = resolve_member(pick(&depth0, TripleMember::DecodeDepth0), TripleMember::DecodeDepth0, seed);
        let tgd = resolve_deepest(&bucket.decode, seed);

        let result = match (pp, tg0, tgd) {
            (Ok(pp), Ok(tg0), Ok((tgd, deepest_depth))) => {
                Ok(Triple { pp_tok_s: pp, tg_tok_s_depth0: tg0, tg_tok_s_deepest: tgd, deepest_depth })
            }
            (pp, tg0, tgd) => {
                let mut problems = Vec::with_capacity(3);
                if let Err(p) = pp {
                    problems.push(p);
                }
                if let Err(p) = tg0 {
                    problems.push(p);
                }
                if let Err(p) = tgd {
                    problems.push(p);
                }
                Err(TripleError { problems })
            }
        };
        out.insert(key, result);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_records;

    /// Builds a minimal, valid `BenchRecord` for tests that only care about a few fields.
    fn record(n_prompt: u64, n_gen: u64, n_depth: u64, samples_ts: &[f64]) -> BenchRecord {
        let samples_ns: Vec<u64> = (0..samples_ts.len() as u64).collect();
        BenchRecord {
            build_commit: "deadbeef".into(),
            build_number: 1,
            backends: None,
            model_filename: "m.gguf".into(),
            model_type: None,
            model_size: None,
            n_params: None,
            n_gpu_layers: None,
            n_cpu_moe: None,
            type_k: None,
            type_v: None,
            flash_attn: None,
            n_threads: None,
            cpu_mask: None,
            n_batch: None,
            n_ubatch: None,
            devices: None,
            n_prompt,
            n_gen,
            n_depth,
            samples_ns,
            samples_ts: samples_ts.to_vec(),
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn classifies_prefill_and_decode() {
        assert_eq!(classify(0, &record(64, 0, 0, &[1.0, 2.0])).map(|c| c.kind), Ok(TestKind::Prefill));
        assert_eq!(classify(0, &record(0, 32, 0, &[1.0, 2.0])).map(|c| c.kind), Ok(TestKind::Decode));
        // A combined pp+tg test is still a decode test per the classification rule.
        assert_eq!(classify(0, &record(64, 32, 0, &[1.0, 2.0])).map(|c| c.kind), Ok(TestKind::Decode));
        assert_eq!(
            classify(0, &record(0, 0, 0, &[1.0, 2.0])),
            Err(ClassifyError::NeitherPrefillNorDecode { index: 0 })
        );
    }

    #[test]
    fn classify_carries_depth_and_index() {
        // `classify` returns depth and token counts alongside the classification, so grouping
        // code never has to go back to the source `BenchRecord` to learn them.
        assert_eq!(
            classify(5, &record(0, 32, 128, &[1.0, 2.0])),
            Ok(Classified { kind: TestKind::Decode, depth: 128, n_prompt: 0, n_gen: 32 })
        );
        assert_eq!(
            classify(5, &record(0, 0, 0, &[1.0, 2.0])),
            Err(ClassifyError::NeitherPrefillNorDecode { index: 5 })
        );
    }

    #[test]
    fn summarize_rejects_fewer_than_three_kept_samples() {
        // 4 raw -> 3 kept is the minimum that succeeds.
        assert!(summarize(&[1.0, 2.0, 3.0, 4.0], 0).is_ok());
        // 3 raw -> 2 kept must be refused, not silently summarized from 2 points.
        assert_eq!(
            summarize(&[1.0, 2.0, 3.0], 0),
            Err(StatsError::InsufficientSamples { raw: 3, kept: 2, required: 3 })
        );
        assert_eq!(
            summarize(&[], 0),
            Err(StatsError::InsufficientSamples { raw: 0, kept: 0, required: 3 })
        );
    }

    #[test]
    fn summarize_rejects_zero_bootstrap_iterations() {
        assert_eq!(
            summarize_with_iterations(&[1.0, 2.0, 3.0, 4.0], 0, 0),
            Err(StatsError::BootstrapIterationsOutOfRange { got: 0, max: MAX_BOOTSTRAP_ITERATIONS })
        );
    }

    #[test]
    fn rejects_out_of_range_bootstrap_iterations() {
        // A caller-supplied count above the cap must be a typed error, never a panic from
        // `Vec::with_capacity` and never unbounded CPU time.
        let got = MAX_BOOTSTRAP_ITERATIONS + 1;
        assert_eq!(
            summarize_with_iterations(&[1.0, 2.0, 3.0, 4.0], 0, got),
            Err(StatsError::BootstrapIterationsOutOfRange { got, max: MAX_BOOTSTRAP_ITERATIONS })
        );
    }

    #[test]
    fn summarize_rejects_nan_inf_zero_negative() {
        // Checked on the raw slice, before the first-repetition discard: a NaN/zero/negative/
        // infinite value in rep 1 still means the record it came from is corrupt.
        assert_eq!(
            summarize(&[1.0, 2.0, 3.0, f64::NAN], 0),
            Err(StatsError::InvalidSample { position: 3, value: f64::NAN.to_string() })
        );
        assert_eq!(
            summarize(&[f64::INFINITY, 2.0, 3.0, 4.0], 0),
            Err(StatsError::InvalidSample { position: 0, value: f64::INFINITY.to_string() })
        );
        assert_eq!(
            summarize(&[1.0, 0.0, 3.0, 4.0], 0),
            Err(StatsError::InvalidSample { position: 1, value: 0.0f64.to_string() })
        );
        assert_eq!(
            summarize(&[1.0, 2.0, -3.0, 4.0], 0),
            Err(StatsError::InvalidSample { position: 2, value: (-3.0f64).to_string() })
        );
    }

    #[test]
    fn summarize_rejects_more_than_max_repetitions() {
        let raw = vec![1.0; MAX_REPETITIONS + 1];
        assert_eq!(
            summarize(&raw, 0),
            Err(StatsError::TooManySamples { got: MAX_REPETITIONS + 1, max: MAX_REPETITIONS })
        );
    }

    #[test]
    fn rng_matches_published_splitmix64_reference() {
        // Sebastiano Vigna's splitmix64.c reference implementation (public domain,
        // http://prng.di.unimi.it/splitmix64.c), seed x = 0: its first output. Pinning this
        // catches a typo'd constant that no amount of "the CI numbers still look plausible"
        // review would catch, since a wrong constant still produces *a* pseudo-random stream.
        let mut r = Rng::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn single_iteration_bootstrap_is_hand_traceable() {
        // raw = [99, 30, 10, 20] -> discard rep1 (99.0) -> kept sorted = [10.0, 20.0, 30.0], n = 3.
        //
        // `Rng::new(0)`'s draws, computed from the same splitmix64 constants pinned by
        // `rng_matches_published_splitmix64_reference` above (and cross-checked independently in
        // `tools/bootstrap_cross_check.py`):
        //   draw 0 = 0xE220_A839_7B1D_CDAF % 3 = 1 -> sorted_kept[1] = 20.0
        //   draw 1 = 0x6E78_9E6A_A1B9_65F4 % 3 = 0 -> sorted_kept[0] = 10.0
        //   draw 2 = 0x06C4_5D18_8009_454F % 3 = 1 -> sorted_kept[1] = 20.0
        // resample = [20.0, 10.0, 20.0] -> sorted = [10.0, 20.0, 20.0] -> median = 20.0.
        // A single bootstrap median means the 2.5th and 97.5th percentiles of that one-element
        // set are both exactly that value.
        let s = summarize_with_iterations(&[99.0, 30.0, 10.0, 20.0], 0, 1).unwrap();
        assert_eq!(s.ci95_low, 20.0);
        assert_eq!(s.ci95_high, 20.0);
    }

    // The two cases below pin exact numbers. median/min/max/iqr were computed by hand; ci95_low
    // and ci95_high were computed by an independent Python reimplementation of the exact same
    // splitmix64 + resample + linear-interpolated-percentile algorithm, committed at
    // `tools/bootstrap_cross_check.py` so these numbers are reproducible by anyone with only this
    // repo and a Python interpreter, not just by re-running the Rust code against itself.

    #[test]
    fn pins_exact_stats_even_kept_count() {
        // raw = [10, 20, 24, 28, 32] -> discard 10.0 -> kept sorted = [20, 24, 28, 32].
        // median = (24+28)/2 = 26; min=20; max=32.
        // Moore-McCabe quartiles: lower=[20,24] -> 22, upper=[28,32] -> 30, iqr = 30-22 = 8.
        let s = summarize_with_iterations(&[10.0, 20.0, 24.0, 28.0, 32.0], 7, 8).unwrap();
        assert_eq!(s.kept, 4);
        assert_eq!(s.median, 26.0);
        assert_eq!(s.min, 20.0);
        assert_eq!(s.max, 32.0);
        assert_eq!(s.iqr, 8.0);
        assert_eq!(s.ci95_low, 22.35);
        assert_eq!(s.ci95_high, 29.65);
    }

    #[test]
    fn pins_exact_stats_odd_kept_count() {
        // raw = [5, 10, 12, 14, 16, 18] -> discard 5.0 -> kept sorted = [10, 12, 14, 16, 18].
        // median = 14 (middle element); min=10; max=18.
        // Moore-McCabe quartiles: lower=[10,12] -> 11, upper=[16,18] -> 17, iqr = 17-11 = 6.
        let s = summarize_with_iterations(&[5.0, 10.0, 12.0, 14.0, 16.0, 18.0], 42, 20).unwrap();
        assert_eq!(s.kept, 5);
        assert_eq!(s.median, 14.0);
        assert_eq!(s.min, 10.0);
        assert_eq!(s.max, 18.0);
        assert_eq!(s.iqr, 6.0);
        assert_eq!(s.ci95_low, 10.95);
        assert_eq!(s.ci95_high, 18.0);
    }

    #[test]
    fn discards_first_repetition_by_position_not_by_value() {
        // In the two pinned tests above, rep1 also happens to be the minimum, so a sort-then-
        // skip(1) bug (discard the smallest value, not the first repetition) would pass them
        // unchanged. These two cases break that tie: rep1 is not the min here.

        // rep1 = 100.0 is the MAXIMUM of the raw set; a sort-then-skip bug would instead drop
        // 20.0 (the true minimum) and keep 100.0, producing a very different median/max.
        // kept (by position) = [20, 24, 28, 32] -> median 26, min 20, max 32.
        // Moore-McCabe quartiles: lower=[20,24]->22, upper=[28,32]->30, iqr=8.
        let s = summarize_with_iterations(&[100.0, 20.0, 24.0, 28.0, 32.0], 7, 8).unwrap();
        assert_eq!((s.kept, s.min, s.max, s.median, s.iqr), (4, 20.0, 32.0, 26.0, 8.0));

        // rep1 = 20.0 sits in the middle of the raw set's range, neither min nor max.
        // kept (by position) = [10, 30, 40] -> median 30, min 10, max 40.
        let s = summarize_with_iterations(&[20.0, 10.0, 30.0, 40.0], 1, 4).unwrap();
        assert_eq!((s.min, s.median, s.max), (10.0, 30.0, 40.0));
    }

    #[test]
    fn iqr_uses_exclusive_halves_for_odd_kept() {
        // raw = [99, 2, 10, 12, 14, 16] (6 raw) -> discard rep1 (99.0) -> kept sorted =
        // [2, 10, 12, 14, 16] (5 elements, odd), overall median = 12.
        // Moore-McCabe (exclusive): lower=[2,10]->6, upper=[14,16]->15, iqr = 15-6 = 9.
        // Tukey's hinges (inclusive of the median in both halves) would instead give
        // lower=[2,10,12]->10, upper=[12,14,16]->14, iqr = 4 — a different, wrong answer here.
        let s = summarize_with_iterations(&[99.0, 2.0, 10.0, 12.0, 14.0, 16.0], 3, 8).unwrap();
        assert_eq!(s.kept, 5);
        assert_eq!(s.median, 12.0);
        assert_eq!(s.iqr, 9.0, "Moore-McCabe exclusive halves, not Tukey's inclusive hinges");
    }

    #[test]
    fn bootstrap_ci_of_constant_samples_is_degenerate() {
        // Every kept sample resamples to the same value, so every bootstrap median is 5.0 and the
        // interval must collapse to a point, for any seed.
        for seed in [0, 1, 42, u64::MAX] {
            let s = summarize(&[1.0, 5.0, 5.0, 5.0, 5.0], seed).unwrap();
            assert_eq!((s.ci95_low, s.ci95_high), (5.0, 5.0));
        }
    }

    #[test]
    fn same_seed_and_samples_reproduce_the_same_interval() {
        // The point of a seeded local RNG rather than a global one: determinism is checkable, with
        // no dependency on call order or any other ambient state.
        let a = summarize(&[1.0, 5.0, 3.0, 8.0, 2.0, 9.0], 123).unwrap();
        let b = summarize(&[1.0, 5.0, 3.0, 8.0, 2.0, 9.0], 123).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_can_produce_different_intervals() {
        // At a large iteration count the percentile estimate converges and can legitimately look
        // seed-independent for a given sample set, so this checks sensitivity with a small
        // iteration count instead — the property under test is "the seed parameter is load-bearing
        // on the resample draws", not "every seed disagrees with every other seed at n=2000".
        let samples = [1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 6.0, 10.0];
        let intervals: Vec<(f64, f64)> = (0..8)
            .map(|seed| {
                let s = summarize_with_iterations(&samples, seed, 5).unwrap();
                (s.ci95_low, s.ci95_high)
            })
            .collect();
        assert!(
            intervals.windows(2).any(|w| w[0] != w[1]),
            "seed must actually influence the bootstrap: got {intervals:?}"
        );
    }

    #[test]
    fn assembles_a_complete_triple_from_the_real_fixture() {
        let json = include_str!("../tests/fixtures/cpu_tiny_model.json");
        let records = parse_records(json).unwrap();
        let triples = assemble_triples(&records, 1).unwrap();
        assert_eq!(triples.len(), 1, "the fixture is a single configuration");

        let (_, result) = triples.into_iter().next().unwrap();
        let triple = result.expect("the fixture has all three triple members");
        assert_eq!(triple.deepest_depth, NonZeroU64::new(128).unwrap());
        for s in [triple.pp_tok_s, triple.tg_tok_s_depth0, triple.tg_tok_s_deepest] {
            assert_eq!(s.kept, 3);
            assert!(s.min <= s.median && s.median <= s.max);
            assert!(s.ci95_low <= s.ci95_high);
        }
    }

    #[test]
    fn real_fixture_triple_pins_medians() {
        // Values below are read straight off the committed fixture, discarding each record's
        // first sample per the warmup rule and sorting by hand. This is the exact-number
        // counterpart to the sanity-bounds check above: it fails if the wrong record were ever
        // picked for a triple member (e.g. pp@128 mistaken for pp@0), which "min <= median <= max"
        // alone cannot catch since the candidate values are of similar magnitude.
        let json = include_str!("../tests/fixtures/cpu_tiny_model.json");
        let records = parse_records(json).unwrap();
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        let triple = result.unwrap();

        // pp@0 raw samples_ts = [42.1838, 41.2307, 38.2606, 42.3633]; discard rep1 (42.1838) ->
        // kept sorted = [38.2606, 41.2307, 42.3633].
        assert_eq!(triple.pp_tok_s.min, 38.2606);
        assert_eq!(triple.pp_tok_s.median, 41.2307);
        assert_eq!(triple.pp_tok_s.max, 42.3633);

        // tg@0 raw samples_ts = [3.49733, 2.57026, 3.04891, 2.95694]; discard rep1 (3.49733) ->
        // kept sorted = [2.57026, 2.95694, 3.04891].
        assert_eq!(triple.tg_tok_s_depth0.min, 2.57026);
        assert_eq!(triple.tg_tok_s_depth0.median, 2.95694);
        assert_eq!(triple.tg_tok_s_depth0.max, 3.04891);

        // tg@128 (deepest) raw samples_ts = [4.38805, 3.48984, 5.93664, 5.66639]; discard rep1
        // (4.38805) -> kept sorted = [3.48984, 5.66639, 5.93664].
        assert_eq!(triple.tg_tok_s_deepest.min, 3.48984);
        assert_eq!(triple.tg_tok_s_deepest.median, 5.66639);
        assert_eq!(triple.tg_tok_s_deepest.max, 5.93664);
    }

    #[test]
    fn result_only_fields_do_not_split_configurations() {
        // Every record in the fixture has its own test_time/avg_ns/stddev_ns/avg_ts/stddev_ts,
        // yet they must all fall under one ConfigKey — those fields describe the result of a
        // test, not the configuration under test.
        let json = include_str!("../tests/fixtures/cpu_tiny_model.json");
        let records = parse_records(json).unwrap();
        let triples = assemble_triples(&records, 1).unwrap();
        assert_eq!(
            triples.len(),
            1,
            "records differing only in test_time/avg_ns/stddev_ns/avg_ts/stddev_ts must share a \
             configuration"
        );
        let (key, _) = triples.into_iter().next().unwrap();
        let extra_keys: Vec<&str> = key.extra().iter().map(|(k, _)| k.as_str()).collect();
        for result_only in ["test_time", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts"] {
            assert!(!extra_keys.contains(&result_only), "{result_only} must not be in extra()");
        }
        assert!(
            !extra_keys.contains(&"model_n_params"),
            "model_n_params is consumed by the renamed n_params field, not left in extra"
        );
        assert!(
            extra_keys.contains(&"tensor_buft_overrides"),
            "a genuinely unmodeled configuration field must still be part of the key"
        );
    }

    #[test]
    fn missing_triple_member_is_a_typed_error_not_a_partial_triple() {
        // Only a decode-depth0 record exists: no prefill, and no -d sweep was measured at all.
        // decode@0 can fill DecodeDepth0, but it must never also be accepted as DecodeDeepest --
        // that would make a depth-0-only run look like a complete context-decay measurement.
        let records = vec![record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0])];
        let triples = assemble_triples(&records, 1).unwrap();
        assert_eq!(triples.len(), 1);
        let (_, result) = triples.into_iter().next().unwrap();
        let err = result.expect_err("prefill is absent and no -d sweep was measured");
        assert_eq!(
            err.problems,
            vec![TripleProblem::NoSuchTest(TripleMember::PrefillDepth0), TripleProblem::NoDepthSweep],
        );
    }

    #[test]
    fn triple_member_with_too_few_samples_is_unsummarizable_not_missing() {
        // A record for every member exists, but the prefill one has too few raw samples.
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0]), // 2 raw -> 1 kept, below MIN_KEPT_SAMPLES
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 128, &[1.0, 2.0, 3.0, 4.0]),
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        let err = result.expect_err("prefill cannot be summarized from 1 kept sample");
        assert_eq!(
            err.problems,
            vec![TripleProblem::Unsummarizable(
                TripleMember::PrefillDepth0,
                StatsError::InsufficientSamples { raw: 2, kept: 1, required: 3 }
            )]
        );
    }

    #[test]
    fn deepest_decode_picks_the_greatest_depth_seen() {
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 64, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 256, &[1.0, 2.0, 3.0, 4.0]),
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        assert_eq!(result.unwrap().deepest_depth, NonZeroU64::new(256).unwrap());
    }

    #[test]
    fn deepest_ignores_depth0_when_a_deeper_decode_exists() {
        // tg@0 and tg@256 both present: the deepest member must come only from the depth>0
        // record, never fall back to depth 0 just because it also happens to exist.
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 0, &[10.0, 1.0, 2.0, 3.0]), // tg@0: kept sorted [1,2,3] -> median 2.0
            record(0, 32, 256, &[10.0, 28.0, 30.0, 32.0]), // tg@256: kept sorted [28,30,32] -> median 30.0
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        let triple = result.expect("a real -d sweep yields a complete triple");
        assert_eq!(triple.deepest_depth, NonZeroU64::new(256).unwrap());
        assert_eq!(triple.tg_tok_s_deepest.median, 30.0);
        assert_eq!(triple.tg_tok_s_depth0.median, 2.0);
    }

    #[test]
    fn depth0_only_decode_is_no_depth_sweep_not_complete() {
        // pp@0 + tg@0 only, no -d sweep at all: the triple must never look complete by quietly
        // reusing the depth-0 record as its own "deepest" member.
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        let err = result.expect_err("no -d sweep was measured, so no context-decay member exists");
        assert_eq!(err.problems, vec![TripleProblem::NoDepthSweep]);
    }

    #[test]
    fn distinct_configurations_are_never_merged() {
        let mut a = record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
        a.n_threads = Some(6);
        let mut b = record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
        b.n_threads = Some(12);
        let triples = assemble_triples(&[a, b], 1).unwrap();
        assert_eq!(triples.len(), 2, "differing n_threads must not collapse into one configuration");
    }

    #[test]
    fn unknown_config_field_splits_configurations() {
        // Two otherwise-identical prefill records whose only difference is a field this crate has
        // no named field for at all (the -ot MoE axis). Merging them under one key would be
        // exactly the "confidently wrong number" this crate exists to prevent.
        let mut a = record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
        a.extra.insert("tensor_buft_overrides".into(), serde_json::Value::String("none".into()));
        let mut b = record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
        b.extra.insert("tensor_buft_overrides".into(), serde_json::Value::String("exps=CPU".into()));
        let triples = assemble_triples(&[a, b], 1).unwrap();
        assert_eq!(
            triples.len(),
            2,
            "an unmodeled configuration field that differs must split, never merge, configurations"
        );
    }

    #[test]
    fn classify_error_names_the_record_index() {
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 0, 0, &[1.0, 2.0, 3.0, 4.0]), // neither prefill nor decode: index 1
        ];
        assert_eq!(
            assemble_triples(&records, 1),
            Err(ClassifyError::NeitherPrefillNorDecode { index: 1 })
        );
    }

    #[test]
    fn sweep_of_two_prompt_sizes_is_ambiguous_not_last_wins() {
        // `-p 64,512` at depth 0 produces two prefill@0 records for one configuration. Neither a
        // first-wins nor a last-wins pick is correct here: both are real measurements of the same
        // configuration at different test shapes, and silently keeping one throws the other away
        // without a word.
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),  // pp@64
            record(512, 0, 0, &[1.0, 2.0, 3.0, 4.0]), // pp@512
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),  // tg@0
            record(0, 32, 128, &[1.0, 2.0, 3.0, 4.0]), // tg@128
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        assert_eq!(triples.len(), 1);
        let (_, result) = triples.into_iter().next().unwrap();
        let err = result.expect_err("two competing prefill@0 records must never be resolved silently");
        assert_eq!(
            err.problems,
            vec![TripleProblem::AmbiguousTest {
                member: TripleMember::PrefillDepth0,
                first: (64, 0),
                second: (512, 0),
            }]
        );
    }

    #[test]
    fn two_gen_sizes_at_depth0_never_mix_records_across_members() {
        // `-n 32,128` at depth 0 with no deeper test: both tg32@0 and tg128@0 compete for
        // DecodeDepth0. Neither can stand in for DecodeDeepest either -- with no record at any
        // depth > 0, this is a NoDepthSweep, not a second ambiguity to resolve from the same pair.
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),  // pp@0 (unambiguous)
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),  // tg32@0
            record(0, 128, 0, &[1.0, 2.0, 3.0, 4.0]), // tg128@0
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        let err = result.expect_err("two n_gen sizes at depth 0, with no sweep, must never resolve silently");
        assert_eq!(
            err.problems,
            vec![
                TripleProblem::AmbiguousTest {
                    member: TripleMember::DecodeDepth0,
                    first: (0, 32),
                    second: (0, 128),
                },
                TripleProblem::NoDepthSweep,
            ]
        );
    }

    #[test]
    fn assemble_triples_surfaces_invalid_sample_from_code_built_record() {
        // `BenchRecord`'s fields are all pub, so a record built directly in code (rather than
        // through `parse_records`) can carry a NaN sample. `assemble_triples` must still refuse
        // it, not silently compute a confidently wrong median.
        let records = vec![
            record(64, 0, 0, &[1.0, 2.0, f64::NAN, 4.0]),
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 128, &[1.0, 2.0, 3.0, 4.0]),
        ];
        let triples = assemble_triples(&records, 1).unwrap();
        let (_, result) = triples.into_iter().next().unwrap();
        let err = result.expect_err("a NaN sample must fail summarization even when built in code");
        assert_eq!(
            err.problems,
            vec![TripleProblem::Unsummarizable(
                TripleMember::PrefillDepth0,
                StatsError::InvalidSample { position: 2, value: f64::NAN.to_string() }
            )]
        );
    }

    #[test]
    fn pg_record_is_not_used_as_decode_throughput() {
        // A combined `-pg 64,32` record classifies as Decode (n_gen > 0) per the brief, but its
        // samples_ts is (n_prompt+n_gen)/time -- combined throughput, not decode tok/s -- and must
        // never be picked for a tg_tok_s member, even when it is the only other decode-shaped
        // record around. A real depth>0 decode record is included too so the triple is complete
        // (otherwise this would report NoDepthSweep instead of exercising the pg exclusion).
        let pp = record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]);
        let tg0 = record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]); // kept sorted [2,3,4] -> median 3.0
        let tgd = record(0, 32, 128, &[1.0, 2.0, 3.0, 4.0]); // kept sorted [2,3,4] -> median 3.0
        let pg = record(64, 32, 0, &[100.0, 200.0, 300.0, 400.0]); // must be excluded, not merged
        let triples = assemble_triples(&[pp, tg0, tgd, pg], 1).unwrap();
        assert_eq!(triples.len(), 1);
        let (_, result) = triples.into_iter().next().unwrap();
        let triple = result.expect("the pg record must not create a spurious ambiguity either");
        assert_eq!(triple.tg_tok_s_depth0.median, 3.0);
    }

    #[test]
    fn a_malformed_record_fails_the_whole_batch_even_before_valid_ones() {
        // The bad record is first here (index 0), with a record for every triple member
        // following it. If classification short-circuited only on the *last* bad record, or if
        // a partial map of already-assembled configurations could leak out, this would show it.
        let records = vec![
            record(0, 0, 0, &[1.0, 2.0, 3.0, 4.0]), // bad, index 0: neither prefill nor decode
            record(64, 0, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 0, &[1.0, 2.0, 3.0, 4.0]),
            record(0, 32, 128, &[1.0, 2.0, 3.0, 4.0]),
        ];
        assert_eq!(
            assemble_triples(&records, 1),
            Err(ClassifyError::NeitherPrefillNorDecode { index: 0 })
        );
    }
}
