//! Typed failures for JSON parsing and statistical summarization.
//!
//! Every variant here exists because the alternative is a confidently wrong number: a truncated
//! sample array silently averaged, a discarded-warmup rule silently skipped, or an incomplete
//! measurement triple silently presented as if it were complete. `docs/ARCHITECTURE.md` section 6
//! is the reason this crate refuses rather than guesses in each of these cases.

use thiserror::Error;

/// Something wrong with a llama-bench `-o json` document that `parse` cannot recover from.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ParseError {
    /// The document is not valid JSON, or its shape does not match a llama-bench record array
    /// (wrong top-level type, a required field missing, or a field present with the wrong type).
    ///
    /// `serde_json::Error` itself is not `Clone`/`Eq`, so its message is captured as text — the
    /// same convention `cerebro-probe::ProbeError` uses for foreign OS errors.
    #[error("llama-bench JSON did not parse: {detail}")]
    Malformed {
        /// The underlying `serde_json` error message.
        detail: String,
    },

    /// A record's `samples_ns` and `samples_ts` arrays have different lengths.
    ///
    /// llama-bench always emits one timing pair per repetition; a mismatch means the document was
    /// hand-edited or truncated, and trusting either array here would misalign which duration goes
    /// with which throughput sample.
    #[error(
        "record {index}: samples_ns has {ns_len} entries but samples_ts has {ts_len}; \
         llama-bench emits one of each per repetition"
    )]
    SampleLengthMismatch {
        /// Position of the offending record in the top-level array.
        index: usize,
        /// Length of `samples_ns`.
        ns_len: usize,
        /// Length of `samples_ts`.
        ts_len: usize,
    },

    /// A record's `samples_ns`/`samples_ts` arrays are both present but empty.
    ///
    /// This means llama-bench was asked to run zero repetitions. `stats::summarize` would reject
    /// it anyway (there is no first repetition to discard), but rejecting it here gives a clearer
    /// error at the parse boundary instead of a confusing one two layers downstream.
    #[error("record {index}: samples_ns/samples_ts is empty; llama-bench recorded zero repetitions")]
    EmptySamples {
        /// Position of the offending record in the top-level array.
        index: usize,
    },

    /// A record's `samples_ts` contains a value that cannot be a real throughput measurement.
    ///
    /// llama-bench never emits zero, negative, or non-finite throughput; a value like that means
    /// the document was hand-edited or corrupted upstream of this crate, and letting it reach
    /// [`crate::stats::summarize`] would silently compute a confidently wrong median from data
    /// that was never physically possible in the first place.
    #[error("record {index}: samples_ts[{sample}] = {value} is not a positive finite throughput")]
    InvalidSample {
        /// Position of the offending record in the top-level array.
        index: usize,
        /// Position of the offending value within that record's `samples_ts`.
        sample: usize,
        /// The offending value, rendered as text (an `f64` field here would not be `Eq`).
        value: String,
    },

    /// A record's `samples_ns`/`samples_ts` carries more repetitions than
    /// [`crate::stats::MAX_REPETITIONS`].
    ///
    /// `docs/ARCHITECTURE.md` section 6 runs `-r 7` to `-r 9`; a record anywhere near this limit is
    /// corrupt (hand-edited or truncated-then-duplicated), and letting it through would let the
    /// bootstrap in [`crate::stats::summarize_with_iterations`] multiply an unbounded repetition
    /// count against its own iteration count, hanging the harness.
    #[error("record {index}: {len} repetitions exceeds the limit of {max}")]
    TooManyRepetitions {
        /// Position of the offending record in the top-level array.
        index: usize,
        /// Number of repetitions the record actually carries.
        len: usize,
        /// The largest repetition count [`crate::parse::parse_records`] accepts.
        max: usize,
    },

    /// The top-level document carries more records than [`crate::parse::MAX_RECORDS`].
    ///
    /// Bounds the other factor a corrupt or tampered results file could inflate: repetitions per
    /// record are capped by [`Self::TooManyRepetitions`], and this caps how many records multiply
    /// that cost.
    #[error("document has {len} records, limit is {max}")]
    TooManyRecords {
        /// Number of records the document actually carries.
        len: usize,
        /// The largest record count [`crate::parse::parse_records`] accepts.
        max: usize,
    },
}

/// Something wrong with classifying a [`crate::parse::BenchRecord`] as prefill or decode.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ClassifyError {
    /// A record has `n_prompt == 0` and `n_gen == 0`, so it is neither a prefill test
    /// (`n_prompt > 0 && n_gen == 0`) nor a decode test (`n_gen > 0`).
    #[error(
        "record {index}: n_prompt=0 and n_gen=0, which is neither a prefill test nor a decode test"
    )]
    NeitherPrefillNorDecode {
        /// Position of the offending record. [`crate::stats::classify`] takes this as a parameter
        /// (it has no list position of its own to discover), so the value here is always the
        /// caller's real position — there is no placeholder to substitute downstream.
        index: usize,
    },
}

/// A raw sample set could not be summarized.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum StatsError {
    /// Fewer than the required number of samples remained after discarding the first repetition.
    ///
    /// `docs/ARCHITECTURE.md` section 6: rep 1 measured 124 t/s against rep 2's 173 t/s with
    /// warmup already enabled, so the first repetition is always dropped before anything is
    /// computed from it — there is no way to opt out, only a way to have too few samples left.
    #[error(
        "need at least {required} kept samples after discarding the first repetition, \
         got {kept} kept from {raw} raw sample(s)"
    )]
    InsufficientSamples {
        /// Number of raw samples supplied, before discarding.
        raw: usize,
        /// Number remaining after discarding the first repetition.
        kept: usize,
        /// Minimum number of kept samples a summary requires.
        required: usize,
    },

    /// `summarize_with_iterations` was asked for a bootstrap iteration count of zero (which cannot
    /// produce a percentile interval) or one large enough to risk a `Vec` capacity overflow and
    /// unbounded CPU time for no statistical benefit beyond [`crate::stats::MAX_BOOTSTRAP_ITERATIONS`].
    #[error("bootstrap iterations must be in 1..={max}, got {got}")]
    BootstrapIterationsOutOfRange {
        /// The iteration count that was requested.
        got: usize,
        /// The largest iteration count [`crate::stats::summarize_with_iterations`] accepts.
        max: usize,
    },

    /// More raw samples were supplied than [`crate::stats::MAX_REPETITIONS`].
    ///
    /// `summarize`/`summarize_with_iterations` are public and reachable without going through
    /// [`crate::parse::parse_records`] at all (a caller can build a `Vec<f64>` directly), so this
    /// boundary validates its own input rather than trusting that parsing already checked it.
    #[error("{got} raw samples exceeds the limit of {max}")]
    TooManySamples {
        /// Number of raw samples supplied.
        got: usize,
        /// The largest raw sample count this module accepts.
        max: usize,
    },

    /// A raw sample is not a positive finite throughput (llama-bench never emits zero, negative,
    /// `NaN`, or infinite throughput).
    ///
    /// Checked on every raw sample, including the repetition that gets discarded next: a `NaN`
    /// warmup rep still means the record it came from is corrupt, and this function is a trust
    /// boundary of its own, not just a downstream consumer of [`crate::parse::parse_records`]'s
    /// check.
    #[error("raw sample {position} = {value} is not a positive finite throughput")]
    InvalidSample {
        /// Position of the offending value within the raw sample slice (before discarding).
        position: usize,
        /// The offending value, rendered as text (an `f64` field here would not be `Eq`).
        value: String,
    },
}

/// One member of the three-number measurement triple a configuration must produce.
///
/// See `docs/ARCHITECTURE.md` section 6: "Never report a single number." Every configuration is
/// characterized by prefill throughput at depth 0, decode throughput at depth 0, and decode
/// throughput at the deepest depth actually measured for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TripleMember {
    /// Prefill (`n_prompt > 0`, `n_gen == 0`) throughput at `n_depth == 0`.
    PrefillDepth0,
    /// Decode (`n_gen > 0`) throughput at `n_depth == 0`.
    DecodeDepth0,
    /// Decode throughput at the deepest `n_depth` present for this configuration.
    DecodeDeepest,
}

impl std::fmt::Display for TripleMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PrefillDepth0 => "prefill@depth0",
            Self::DecodeDepth0 => "decode@depth0",
            Self::DecodeDeepest => "decode@deepest-depth",
        })
    }
}

/// Why one member of a configuration's measurement triple could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TripleProblem {
    /// No record for this configuration matches the member's test shape at all.
    NoSuchTest(TripleMember),
    /// Two records both match this member's test shape (same configuration, same test dimensions
    /// relevant to this member). Picking either would be a guess, so [`crate::stats`] never picks:
    /// this is reported instead of a last-wins overwrite. Carries `(n_prompt, n_gen)` of each of
    /// the first two competing records, in the order they appeared in the input.
    AmbiguousTest {
        /// Which triple member the two records collide on.
        member: TripleMember,
        /// `(n_prompt, n_gen)` of the first competing record.
        first: (u64, u64),
        /// `(n_prompt, n_gen)` of the second competing record.
        second: (u64, u64),
    },
    /// A matching record exists, but its samples could not be summarized.
    Unsummarizable(TripleMember, StatsError),
    /// This configuration has pure decode records, but every one of them is at depth 0.
    ///
    /// `docs/ARCHITECTURE.md` section 6: the third triple member exists because "a depth-0-only
    /// number systematically overstates real use", so a run with no `-d` sweep (or one whose only
    /// deeper decode record was dropped) must never be reported as if `tg_tok_s_deepest` were a
    /// real context-decay measurement — it would just be `tg_tok_s_depth0` again under a different
    /// name.
    NoDepthSweep,
}

impl std::fmt::Display for TripleProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchTest(member) => write!(f, "{member}: no matching test record"),
            Self::AmbiguousTest { member, first, second } => write!(
                f,
                "{member}: two records compete for this slot: (n_prompt={}, n_gen={}) and \
                 (n_prompt={}, n_gen={})",
                first.0, first.1, second.0, second.1
            ),
            Self::Unsummarizable(member, source) => write!(f, "{member}: {source}"),
            Self::NoDepthSweep => write!(
                f,
                "{}: every decode record is at depth 0; no -d sweep was measured",
                TripleMember::DecodeDeepest
            ),
        }
    }
}

/// A configuration did not yield a complete measurement triple.
///
/// `docs/ARCHITECTURE.md` section 6 is explicit that a configuration missing any member of the
/// triple must never be presented as if the triple were complete, so `stats::assemble_triples`
/// returns this instead of a `Triple` with a member silently left out or defaulted to zero.
///
/// `Display` is implemented by hand (rather than via `#[error("...")]`) so the message joins each
/// problem's own `Display` output instead of falling back to `{problems:?}` Rust `Debug` syntax.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub struct TripleError {
    /// Every member that could not be produced, in [`TripleMember`] order.
    pub problems: Vec<TripleProblem>,
}

impl std::fmt::Display for TripleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "configuration's measurement triple is incomplete: ")?;
        for (i, problem) in self.problems.iter().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            write!(f, "{problem}")?;
        }
        Ok(())
    }
}
