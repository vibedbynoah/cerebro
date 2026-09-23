//! Pure queries over a slice of loaded [`Record`]s.
//!
//! Deliberately free of I/O: a caller runs [`crate::store::Store::load_verified`] once and then
//! asks these functions as many questions as it wants, which also makes them trivial to test with
//! hand-built records rather than real files.

use std::num::NonZeroU64;

use crate::error::RecordError;
use crate::record::{QualityGateKind, Record, SamplerProfile, Sha256Hex};

/// Whether a previously-recorded configuration is still trustworthy for the machine as it stands
/// today.
///
/// `docs/ARCHITECTURE.md` section 6: binary identity and model identity are both dimensions of a
/// result, not footnotes, so a change to either invalidates a stored recommendation even though
/// the flags themselves are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Staleness {
    /// The machine profile, model hash, and backend manifest digest all still match.
    Fresh,
    /// The model file has changed (different SHA-256) since this record was measured.
    ModelChanged,
    /// The backend build has changed (different manifest digest) since this record was measured.
    BackendChanged,
    /// Both the model and the backend have changed.
    BothChanged,
    /// The machine profile itself does not match -- this record was measured on different
    /// hardware. Takes precedence over the model/backend checks: a results file copied between
    /// machines, restored from a backup onto different hardware, or concatenated from two
    /// profiles can still have matching model and backend digests, but the throughput numbers
    /// were never measured on the machine asking about them.
    MachineChanged,
}

/// Compare a record's provenance against the machine, model, and backend identity in use right
/// now.
///
/// Every digest is the validated [`Sha256Hex`] type, not a bare `&str`: a caller comparing a
/// digest produced by a different hex-case convention, or a truncated value, gets a
/// construction-time refusal from [`Sha256Hex::parse`] instead of a silent, wrong "changed"
/// verdict from this function.
pub fn check_staleness(
    record: &Record,
    current_machine_profile_id: &Sha256Hex,
    current_model_sha256: &Sha256Hex,
    current_backend_manifest_digest: &Sha256Hex,
) -> Staleness {
    if record.provenance().machine_profile_id() != current_machine_profile_id {
        // Checked first, and returned unconditionally: a record from another machine is not
        // "fresh with caveats" just because its model/backend digests happen to also match.
        return Staleness::MachineChanged;
    }
    let model_changed = record.provenance().model_sha256() != current_model_sha256;
    let backend_changed = record.provenance().backend_manifest_digest() != current_backend_manifest_digest;
    match (model_changed, backend_changed) {
        (false, false) => Staleness::Fresh,
        (true, false) => Staleness::ModelChanged,
        (false, true) => Staleness::BackendChanged,
        (true, true) => Staleness::BothChanged,
    }
}

/// The quality gate and tolerance a caller trusts, used to judge a record's eligibility for
/// [`best_config`].
///
/// `docs/ARCHITECTURE.md` section 7 fixes the gate kind and its tolerance band project-wide (NMSE
/// bracketed to `(8.454e-07, 3.324e-03]`); that is policy the caller holds, never data a results
/// file supplies. A record's own [`crate::record::QualityGateOutcome::threshold`] is recorded
/// provenance -- what the run was judged against at the time it was measured -- and must never by
/// itself decide whether that record gets recommended: a poisoned or hand-edited record can move
/// its own threshold to make any value pass. See [`crate::record::QualityGateOutcome::new`]'s docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityPolicy {
    gate: QualityGateKind,
    max_value: f64,
}

impl QualityPolicy {
    /// Build a quality policy.
    ///
    /// # Errors
    /// [`RecordError::NonFiniteOrNegative`] if `max_value` is not finite or is negative --
    /// otherwise a `NaN` policy would make every comparison in [`best_config`] false and silently
    /// return `None`, the same hole [`SamplerProfile::new`] and
    /// [`crate::record::MeasurementSummary::new`] already close for their own fields.
    pub fn new(gate: QualityGateKind, max_value: f64) -> Result<Self, RecordError> {
        if !(max_value.is_finite() && max_value >= 0.0) {
            return Err(RecordError::NonFiniteOrNegative { field: "quality_policy.max_value", value: max_value });
        }
        Ok(Self { gate, max_value })
    }

    /// Which check a record must have run to be eligible.
    pub fn gate(&self) -> QualityGateKind {
        self.gate
    }

    /// The largest measured value that still counts as a pass under this policy.
    pub fn max_value(&self) -> f64 {
        self.max_value
    }
}

/// Find the best configuration for `(machine_profile_id, model_sha256, backend_manifest_digest,
/// sampler)` at a specific working context depth, by median decode tok/s, considering only
/// records whose quality gate passed and whose soak (if any) did not fail.
///
/// **`machine_profile_id` is filtered first, ahead of every other predicate** -- the same
/// precedence [`check_staleness`] gives [`Staleness::MachineChanged`]. A record measured on other
/// hardware is not a weaker recommendation to fall back to; it is not a recommendation at all, no
/// matter how it compares to the alternatives on model, backend, sampler, or throughput. This is
/// enforced here, not merely documented as a caller precondition: every other identity check in
/// this crate ([`Store::append`](crate::store::Store::append)'s machine-profile check,
/// [`check_staleness`]) is enforced the same way.
///
/// Two further eligibility rules follow directly from `docs/ARCHITECTURE.md` section 6 and are not
/// optional:
///
/// - **`working_depth` narrows to records measured at that exact `deepest_depth`.** Decode
///   throughput decays with context depth (measured: 35.08 -> 31.49 t/s between d0 and d64 on a
///   small model), so a median measured at depth 128 is not comparable to one measured at depth
///   32768 -- ranking across depths would pick "the config tested at the shallowest depth," not
///   the fastest one at the depth that matters.
/// - **A record whose soak *failed* is excluded, even if its median is higher.** "VRAM trend is a
///   pass criterion... this catches the configuration that runs for hours and then dies." A
///   record with no soak at all is still eligible: tier 1 (this query's natural use, screening)
///   runs before the soak.
///
/// A record is eligible only if it ran the gate `policy` names and its measured value clears
/// `policy`'s tolerance -- **never** by asking the record's own
/// [`crate::record::QualityGateOutcome::passed`], which is self-certified from a `threshold` field
/// a hostile results file also controls (see [`QualityPolicy`]'s docs). A record with no quality
/// gate outcome at all is excluded, the same as one that ran the wrong gate or failed: "no gate was
/// run" is not evidence that the configuration is correct.
///
/// A record whose soak *failed* is excluded even if its median is higher: `docs/ARCHITECTURE.md`
/// section 6, "VRAM trend is a pass criterion... this catches the configuration that runs for hours
/// and then dies." Unlike the quality gate, judging `soak.passed()` directly is safe even though
/// it is also a stored, file-controlled bool: it only ever *excludes* a record, so a hostile file
/// can at worst hide a good configuration by flipping this to `false`, never promote a bad one by
/// flipping it to `true` (a record with no soak at all is already eligible, and the gate itself is
/// the only thing standing between an unqualified configuration and a recommendation). A record
/// with no soak at all is still eligible: tier 1 (this query's natural use, screening) runs before
/// the soak.
///
/// Ties (equal median) resolve to the first-encountered record in `records`, so the result is
/// deterministic for a given input order rather than depending on `f64` comparison direction.
///
/// `sampler` is matched by [`SamplerProfile`]'s quantized-value [`PartialEq`], not bit-exact `f64`
/// equality: a caller that recomputes the same logical sampler setting by different arithmetic
/// (`0.7 + 0.2` rather than a literal `0.9`) still matches a record measured at "the same" value.
/// See [`SamplerProfile`]'s doc comment.
#[allow(clippy::too_many_arguments)]
pub fn best_config<'a>(
    records: &'a [Record],
    machine_profile_id: &Sha256Hex,
    model_sha256: &Sha256Hex,
    backend_manifest_digest: &Sha256Hex,
    sampler: &SamplerProfile,
    working_depth: NonZeroU64,
    policy: QualityPolicy,
) -> Option<&'a Record> {
    records
        .iter()
        // First, and unconditionally: see this function's doc comment for why machine identity
        // takes precedence over every other predicate, including a candidate's throughput.
        .filter(|r| r.provenance().machine_profile_id() == machine_profile_id)
        .filter(|r| r.provenance().model_sha256() == model_sha256)
        .filter(|r| r.provenance().backend_manifest_digest() == backend_manifest_digest)
        .filter(|r| r.sampler() == sampler)
        .filter(|r| r.measurement().deepest_depth() == working_depth)
        .filter(|r| match r.quality_gate() {
            // `outcome.passed()` is self-certifying: it derives from a `threshold` field the
            // results file also supplies, so a hostile record can move its own threshold to make
            // any value pass. Judge the measured value against the caller's policy instead.
            Some(outcome) => outcome.gate() == policy.gate() && outcome.value() <= policy.max_value(),
            None => false,
        })
        .filter(|r| r.soak().is_none_or(|s| s.passed()))
        .fold(None, |best: Option<&Record>, candidate| match best {
            Some(current) if candidate.measurement().tg_tok_s_deepest().median() <= current.measurement().tg_tok_s_deepest().median() => {
                Some(current)
            }
            _ => Some(candidate),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{MeasurementSummary, MeasurementTriple, ProvenanceKey, QualityGateOutcome, SoakOutcome};

    fn hash(byte: char) -> Sha256Hex {
        Sha256Hex::try_from(byte.to_string().repeat(64)).unwrap()
    }

    /// Every test record in this module was measured on the same fixture machine unless a test is
    /// specifically about machine identity.
    fn machine() -> String {
        "c".repeat(64)
    }

    fn summary(median: f64) -> MeasurementSummary {
        MeasurementSummary::new(3, median, median - 1.0, median + 1.0, 1.0, median - 0.5, median + 0.5).unwrap()
    }

    fn gate(passed: bool) -> QualityGateOutcome {
        QualityGateOutcome::new(QualityGateKind::Nmse, if passed { 1e-7 } else { 1e-2 }, 3.3e-3).unwrap()
    }

    /// The trusted policy every "genuine pass" fixture in this module is measured against:
    /// `docs/ARCHITECTURE.md` section 7's NMSE tolerance band.
    fn policy() -> QualityPolicy {
        QualityPolicy::new(QualityGateKind::Nmse, 3.324e-3).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn make(
        model: &str,
        backend: &str,
        sampler: SamplerProfile,
        decode_median: f64,
        depth: u64,
        quality: Option<QualityGateOutcome>,
        soak: Option<SoakOutcome>,
    ) -> Record {
        make_on(&machine(), model, backend, sampler, decode_median, depth, quality, soak)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_on(
        machine: &str,
        model: &str,
        backend: &str,
        sampler: SamplerProfile,
        decode_median: f64,
        depth: u64,
        quality: Option<QualityGateOutcome>,
        soak: Option<SoakOutcome>,
    ) -> Record {
        let provenance = ProvenanceKey::new(machine, model, backend, None, None, None, None, None).unwrap();
        let triple = MeasurementTriple::new(summary(40.0), summary(3.0), summary(decode_median), depth).unwrap();
        Record::new(provenance, sampler, vec!["llama-server".into()], triple, soak, quality, 0).unwrap()
    }

    fn sampler() -> SamplerProfile {
        SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap()
    }

    fn other_sampler() -> SamplerProfile {
        SamplerProfile::new(0.0, 1, 1.0, 0.0).unwrap()
    }

    /// `best_config`'s `working_depth` is `NonZeroU64`; every fixture depth in this module is a
    /// small nonzero literal, so this never panics.
    fn depth(n: u64) -> NonZeroU64 {
        NonZeroU64::new(n).unwrap()
    }

    #[test]
    fn best_config_picks_the_highest_median_among_passing_records() {
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![
            make(&model, &backend, sampler(), 10.0, 128, Some(gate(true)), None),
            make(&model, &backend, sampler(), 50.0, 128, Some(gate(true)), None),
            make(&model, &backend, sampler(), 30.0, 128, Some(gate(true)), None),
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 50.0);
    }

    #[test]
    fn best_config_ignores_records_whose_quality_gate_failed_or_is_absent() {
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![
            make(&model, &backend, sampler(), 90.0, 128, Some(gate(false)), None), // fastest, but FAILED
            make(&model, &backend, sampler(), 80.0, 128, None, None),              // fast, but never gated
            make(&model, &backend, sampler(), 20.0, 128, Some(gate(true)), None),  // the only eligible one
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 20.0, "must skip the faster failing/ungated records");
    }

    #[test]
    fn best_config_is_scoped_to_model_backend_and_sampler() {
        let model = "a".repeat(64);
        let other_model = "d".repeat(64);
        let backend = "b".repeat(64);
        let other_backend = "e".repeat(64);
        let records = vec![
            make(&other_model, &backend, sampler(), 999.0, 128, Some(gate(true)), None), // wrong model
            make(&model, &other_backend, sampler(), 999.0, 128, Some(gate(true)), None), // wrong backend
            make(&model, &backend, other_sampler(), 999.0, 128, Some(gate(true)), None), // wrong sampler
            make(&model, &backend, sampler(), 15.0, 128, Some(gate(true)), None),        // the real match
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 15.0);
    }

    #[test]
    fn best_config_matches_a_sampler_the_caller_recomputed() {
        // The end-to-end form of the float-equality bug: without the quantized-key `PartialEq`,
        // this returns `None` today, indistinguishable from "this configuration was never
        // measured."
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let recorded_sampler = SamplerProfile::new(0.7, 40, 0.9, 0.05).unwrap();
        let recomputed_sampler = SamplerProfile::new(0.7, 40, 0.7 + 0.2, 0.05).unwrap();
        assert_ne!(0.7_f64 + 0.2, 0.9, "the fixture must really differ in bits");
        let records = vec![make(&model, &backend, recorded_sampler, 40.0, 128, Some(gate(true)), None)];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &recomputed_sampler, depth(128), policy());
        assert!(best.is_some(), "a sampler recomputed by different arithmetic must still match the recorded one");
    }

    #[test]
    fn best_config_returns_none_when_nothing_qualifies() {
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![make(&model, &backend, sampler(), 50.0, 128, Some(gate(false)), None)];
        assert!(best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).is_none());
    }

    #[test]
    fn best_config_ignores_records_from_another_machine_profile() {
        // A slice holding records from two machines (loaded from two profiles' files, or a
        // restored backup concatenated onto a live log): a fast record measured on hardware other
        // than the one asking must never be recommended over a slower one measured on the actual
        // asking machine, no matter how the two compare on throughput alone.
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![
            make_on(&"9".repeat(64), &model, &backend, sampler(), 999.0, 128, Some(gate(true)), None),
            make_on(&"c".repeat(64), &model, &backend, sampler(), 20.0, 128, Some(gate(true)), None),
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(best.provenance().machine_profile_id(), &hash('c'), "must not return the faster record from a different machine");
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 20.0);
    }

    #[test]
    fn best_config_ignores_other_depths() {
        // A depth-128 median of 90 and a depth-32768 median of 20 are not comparable (decode
        // throughput decays with context); querying working_depth=32768 must return the 20 t/s
        // record, not the faster-but-shallower one.
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![
            make(&model, &backend, sampler(), 90.0, 128, Some(gate(true)), None),
            make(&model, &backend, sampler(), 20.0, 32768, Some(gate(true)), None),
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(32768), policy()).unwrap();
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 20.0);
        assert_eq!(best.measurement().deepest_depth(), depth(32768));
    }

    #[test]
    fn best_config_excludes_failed_soak_but_allows_absent_soak() {
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![
            make(&model, &backend, sampler(), 50.0, 128, Some(gate(true)), Some(SoakOutcome::new(false, false, 60, None).unwrap())), // failed soak: dies hours later
            make(&model, &backend, sampler(), 40.0, 128, Some(gate(true)), None), // no soak run yet: tier 1 only, still eligible
            make(&model, &backend, sampler(), 30.0, 128, Some(gate(true)), Some(SoakOutcome::new(true, true, 3600, None).unwrap())), // passed soak
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 40.0);
    }

    #[test]
    fn best_config_tie_returns_first() {
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let records = vec![
            make(&model, &backend, sampler(), 40.0, 128, Some(gate(true)), None),
            make(&model, &backend, sampler(), 40.0, 128, Some(gate(true)), None),
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert!(std::ptr::eq(best, &records[0]));
    }

    #[test]
    fn best_config_rejects_a_record_that_moved_its_own_threshold() {
        // A poisoned record: value=1e-2 (a real FAIL by any honest NMSE tolerance) paired with a
        // threshold of 1e9, so `QualityGateOutcome::passed()` derives to `true`. `best_config` must
        // judge the measured value against the caller's policy, not the record's own claim.
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let poisoned = QualityGateOutcome::new(QualityGateKind::Nmse, 1e-2, 1e9).unwrap();
        assert!(poisoned.passed(), "the poisoned record's own derived flag must say it passed");
        let genuine = QualityGateOutcome::new(QualityGateKind::Nmse, 1e-7, 3.3e-3).unwrap();
        let records = vec![
            make(&model, &backend, sampler(), 90.0, 128, Some(poisoned), None), // faster, but only "passes" by its own moved goalpost
            make(&model, &backend, sampler(), 20.0, 128, Some(genuine), None),  // the only one that clears the caller's policy
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(
            best.measurement().tg_tok_s_deepest().median(),
            20.0,
            "must not recommend a record whose own threshold was moved to make it pass"
        );
    }

    #[test]
    fn best_config_rejects_a_record_gated_by_the_wrong_check() {
        // A KL-divergence outcome whose number would clear an NMSE policy's tolerance if compared
        // blindly must still be excluded: the policy names which CHECK was required, not just a
        // number.
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let wrong_gate = QualityGateOutcome::new(QualityGateKind::KlDivergence, 1e-7, 3.3e-3).unwrap();
        let right_gate = QualityGateOutcome::new(QualityGateKind::Nmse, 1e-7, 3.3e-3).unwrap();
        let records = vec![
            make(&model, &backend, sampler(), 90.0, 128, Some(wrong_gate), None),
            make(&model, &backend, sampler(), 20.0, 128, Some(right_gate), None),
        ];
        let best = best_config(&records, &hash('c'), &hash('a'), &hash('b'), &sampler(), depth(128), policy()).unwrap();
        assert_eq!(best.measurement().tg_tok_s_deepest().median(), 20.0, "must not accept a passing number from the wrong gate kind");
    }

    #[test]
    fn quality_policy_rejects_non_finite_max_value() {
        assert!(matches!(
            QualityPolicy::new(QualityGateKind::Nmse, f64::NAN).unwrap_err(),
            RecordError::NonFiniteOrNegative { field: "quality_policy.max_value", .. }
        ));
        assert!(matches!(
            QualityPolicy::new(QualityGateKind::Nmse, -1.0).unwrap_err(),
            RecordError::NonFiniteOrNegative { field: "quality_policy.max_value", .. }
        ));
    }

    #[test]
    fn staleness_reports_which_side_changed() {
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let r = make(&model, &backend, sampler(), 40.0, 128, Some(gate(true)), None);

        assert_eq!(check_staleness(&r, &hash('c'), &hash('a'), &hash('b')), Staleness::Fresh);
        assert_eq!(check_staleness(&r, &hash('c'), &hash('f'), &hash('b')), Staleness::ModelChanged);
        assert_eq!(check_staleness(&r, &hash('c'), &hash('a'), &hash('f')), Staleness::BackendChanged);
        assert_eq!(check_staleness(&r, &hash('c'), &hash('f'), &hash('f')), Staleness::BothChanged);
    }

    #[test]
    fn staleness_reports_a_changed_machine() {
        // A results file restored onto different hardware: model and backend digests still match
        // (the same GGUF and the same backend build could genuinely be present on both machines),
        // but the throughput numbers were never measured on the machine asking about them.
        let model = "a".repeat(64);
        let backend = "b".repeat(64);
        let r = make(&model, &backend, sampler(), 40.0, 128, Some(gate(true)), None);

        assert_eq!(check_staleness(&r, &hash('9'), &hash('a'), &hash('b')), Staleness::MachineChanged);
        // Takes precedence even when the model or backend has ALSO changed.
        assert_eq!(check_staleness(&r, &hash('9'), &hash('f'), &hash('f')), Staleness::MachineChanged);
    }

    #[test]
    fn staleness_rejects_malformed_digest_at_type_level() {
        assert!(Sha256Hex::try_from("A".repeat(64)).is_err(), "uppercase must be rejected before it can reach a comparison");
        assert!(Sha256Hex::try_from("not-hex".to_string()).is_err());
    }
}
