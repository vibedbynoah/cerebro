//! Cross-flag validation.
//!
//! Individual values are validated at construction ([`crate::value`]), but several failure modes
//! live in *combinations* of individually-valid flags. Those are checked here, before a
//! configuration is spawned or — worse — before it consumes a multi-hour soak test.
//!
//! Three kinds of rule live here, and the distinction matters:
//!
//! - **Inert rules.** Combinations where a flag silently does nothing, which would cause the tuner
//!   to attribute a measurement to a setting that was never active. Hard errors, because a
//!   mislabelled measurement is worse than a missing one.
//! - **Cannot-win rules.** Combinations that are legal and active but structurally unable to be
//!   optimal. Warnings, because the tuner should prune them from the search space rather than
//!   refuse them outright — a user may still want to measure one deliberately.

use crate::emit::Flag;
use crate::value::{CacheType, FlashAttn, SpecType};

/// How seriously a violated rule should be taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// The configuration is legal and active but cannot be optimal; prune it from the search.
    Warning,
    /// The configuration will crash, or will silently not do what it says.
    Error,
}

/// One violated cross-flag rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// How seriously to take it.
    pub severity: Severity,
    /// Stable identifier for the rule, usable as a test anchor and a log key.
    pub rule: &'static str,
    /// What is wrong with this particular configuration.
    pub detail: String,
    /// What to do about it.
    pub remedy: String,
}

impl Violation {
    fn error(rule: &'static str, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            rule,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }

    fn warn(rule: &'static str, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            rule,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

/// A read-only view over the flags in a configuration, so rules can ask questions without
/// repeatedly scanning the slice.
struct View<'a> {
    flags: &'a [Flag],
}

impl<'a> View<'a> {
    fn spec_type(&self) -> Option<SpecType> {
        self.flags.iter().find_map(|f| match f {
            Flag::SpecType(s) => Some(*s),
            _ => None,
        })
    }

    fn flash_attn(&self) -> Option<FlashAttn> {
        self.flags.iter().find_map(|f| match f {
            Flag::FlashAttn(v) => Some(*v),
            _ => None,
        })
    }

    fn kv_types(&self) -> Vec<(&'static str, CacheType)> {
        self.flags
            .iter()
            .filter_map(|f| match f {
                Flag::CacheTypeK(t) => Some(("-ctk", *t)),
                Flag::CacheTypeV(t) => Some(("-ctv", *t)),
                _ => None,
            })
            .collect()
    }

    fn has(&self, pred: impl Fn(&Flag) -> bool) -> bool {
        self.flags.iter().any(pred)
    }

}

/// Whether a KV cache type is quantized, and therefore depends on flash attention.
fn is_quantized(t: CacheType) -> bool {
    !matches!(t, CacheType::F32 | CacheType::F16 | CacheType::Bf16)
}

/// Check every cross-flag rule against a configuration.
///
/// Returns all violations rather than the first, so a caller fixing a generated config sees the
/// whole picture in one pass.
pub fn validate(flags: &[Flag]) -> Vec<Violation> {
    let v = View { flags };
    let mut out = Vec::new();

    let spec = v.spec_type().unwrap_or(SpecType::None);

    // ---- INVESTIGATED AND DISPROVEN: grammar plus backend sampling ----
    //
    // Do not re-add a rule here. A previous revision blocked grammar-constrained output combined
    // with backend sampling as a hard error, on the strength of a source reading that reported
    //   GGML_ASSERT(!gsmpl->grmr && "using grammar in combination with backend sampling is not
    //                                supported")
    // and the fact that --spec-draft-backend-sampling defaults to ENABLED.
    //
    // That rule was wrong. Measured against b10751 with a 0.5B model, six cases across both
    // backends all completed with rc=0 and correct grammar-constrained output:
    //
    //   CPU  (-dev none)          --grammar + --spec-type ngram-simple   (draft sampling default)
    //   CPU  (-dev none)          --grammar + -bs
    //   CPU  (-dev none)          -bs alone
    //   CUDA (-dev CUDA0 -ngl 999) --grammar + --spec-type ngram-simple  (draft sampling default)
    //   CUDA (-dev CUDA0 -ngl 999) --grammar + -bs
    //   CUDA (-dev CUDA0 -ngl 999) --grammar + -bs + --spec-type ngram-simple
    //
    // The assert is not reachable through the documented flag surface on either backend in this
    // build. Blocking the combination would have denied structured output to exactly the users most
    // likely to want it (JSON-mode extraction) for no reason.
    //
    // The lesson is recorded in docs/CLAIMS.md: read the source to form a hypothesis, run the
    // binary to make a claim. Regression coverage lives in the tests below.

    // ---- Inert rule: draft-model flags without a draft-model strategy ----
    //
    // The five ngram-* strategies use no draft model, so every draft-placement flag is silently
    // ignored under them. Recording a measurement as "draft pinned to E-cores" when the draft
    // apparatus was never engaged would poison the tuner's model of the machine.
    if !spec.accepts_draft_model() {
        let draft_flags: Vec<&'static str> = flags
            .iter()
            .filter(|f| {
                matches!(
                    f,
                    Flag::DraftModel(_)
                        | Flag::SpecDraftDevice(_)
                        | Flag::SpecDraftGpuLayers(_)
                        | Flag::SpecDraftCpuMask(_)
                        | Flag::SpecDraftCpuStrict(_)
                        | Flag::SpecDraftCacheTypeK(_)
                        | Flag::SpecDraftCacheTypeV(_)
                )
            })
            .map(|f| f.name())
            .collect();
        if !draft_flags.is_empty() {
            out.push(Violation::error(
                "draft_flags_without_draft_strategy",
                format!(
                    "draft-placement flags {} are set, but --spec-type {} uses no draft model",
                    draft_flags.join(" "),
                    spec
                ),
                "remove the draft-placement flags, or choose a draft-* strategy",
            ));
        }
    }

    // ---- Inert rule: ngram tuning without an ngram strategy ----
    if !spec.is_ngram() {
        let ngram_flags: Vec<&'static str> = flags
            .iter()
            .filter(|f| {
                matches!(
                    f,
                    Flag::SpecNgramSimpleSizeN(_)
                        | Flag::SpecNgramSimpleSizeM(_)
                        | Flag::SpecNgramSimpleMinHits(_)
                )
            })
            .map(|f| f.name())
            .collect();
        if !ngram_flags.is_empty() {
            out.push(Violation::error(
                "ngram_params_without_ngram_strategy",
                format!(
                    "n-gram parameters {} are set under --spec-type {}",
                    ngram_flags.join(" "),
                    spec
                ),
                "choose an ngram-* strategy, or remove the n-gram parameters",
            ));
        }
    }

    // ---- Inert rule: a draft-requiring strategy with no draft model ----
    if spec.requires_draft_model() && !v.has(|f| matches!(f, Flag::DraftModel(_))) {
        out.push(Violation::error(
            "draft_strategy_without_draft_model",
            format!("--spec-type {spec} requires a draft model but no -md was given"),
            "supply -md, or switch to an ngram-* strategy which needs no draft model",
        ));
    }

    // ---- Cannot-win rule: quantized KV without flash attention ----
    //
    // Quantized KV with flash attention off forces per-step dequantization and is frequently
    // slower than plain f16. Generating this cross-product wastes screening time on configurations
    // that are structurally unable to be optimal.
    if v.flash_attn() == Some(FlashAttn::Off) {
        for (flag, t) in v.kv_types() {
            if is_quantized(t) {
                out.push(Violation::warn(
                    "quantized_kv_without_flash_attn",
                    format!("{flag} {t} is quantized but -fa is off"),
                    "pair quantized KV types with -fa on, or use f16/bf16 KV when flash attention is off",
                ));
            }
        }
    }

    // ---- Inert rule: synthetic acceptance rates outside benchmarking ----
    //
    // --spec-synth-* are documented "benchmarking only": they fabricate acceptance behavior to find
    // this machine's break-even rate. A production config carrying them would serve users synthetic
    // speculation rather than real speculation.
    let synth: Vec<&'static str> = flags
        .iter()
        .filter(|f| matches!(f, Flag::SpecSynthLen(_) | Flag::SpecSynthRates(_)))
        .map(|f| f.name())
        .collect();
    if !synth.is_empty() {
        out.push(Violation::warn(
            "synthetic_acceptance_outside_benchmark",
            format!("{} present; these fabricate acceptance behavior", synth.join(" ")),
            "keep --spec-synth-* confined to break-even calibration runs; never emit them into a served configuration",
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{CpuMask, DeviceList};

    fn rules(v: &[Violation]) -> Vec<&'static str> {
        v.iter().map(|x| x.rule).collect()
    }

    // Regression coverage for the rule that was investigated and disproven. These combinations
    // were measured working on both the CPU and CUDA backends of b10751; if a future revision
    // re-adds a constraint blocking them, these fail.
    #[test]
    fn grammar_with_speculation_is_not_blocked() {
        let flags = vec![
            Flag::SpecType(SpecType::NgramSimple),
            Flag::JsonSchema(r#"{"type":"object"}"#.into()),
        ];
        assert!(validate(&flags).is_empty(), "{:?}", validate(&flags));
    }

    #[test]
    fn grammar_with_explicit_backend_sampling_is_not_blocked() {
        let flags = vec![
            Flag::BackendSampling,
            Flag::Grammar("root ::= \"a\"".into()),
        ];
        assert!(validate(&flags).is_empty(), "{:?}", validate(&flags));
    }

    #[test]
    fn grammar_with_both_sampling_paths_and_speculation_is_not_blocked() {
        let flags = vec![
            Flag::BackendSampling,
            Flag::SpecType(SpecType::NgramSimple),
            Flag::Grammar("root ::= \"a\"".into()),
        ];
        assert!(validate(&flags).is_empty(), "{:?}", validate(&flags));
    }

    #[test]
    fn a_warning_only_config_still_builds() {
        use crate::emit::{ArgvBuilder, Dialect};
        let mut b = ArgvBuilder::new(Dialect::Server);
        b.push(Flag::Model(r"C:\m.gguf".into())).unwrap();
        b.push(Flag::CacheTypeK(CacheType::Q8_0)).unwrap();
        b.push(Flag::FlashAttn(FlashAttn::Off)).unwrap();
        assert!(b.build().is_ok(), "warnings must not block emission");
    }

    #[test]
    fn draft_placement_under_ngram_is_inert_and_refused() {
        // The exact mislabelling risk: an ngram run recorded as "draft on E-cores".
        let flags = vec![
            Flag::SpecType(SpecType::NgramMapK),
            Flag::SpecDraftCpuMask(CpuMask::new(0xFF000).unwrap()),
            Flag::SpecDraftDevice(DeviceList::None),
        ];
        assert!(rules(&validate(&flags)).contains(&"draft_flags_without_draft_strategy"));
    }

    #[test]
    fn draft_strategy_without_a_draft_model_is_refused() {
        let flags = vec![Flag::SpecType(SpecType::DraftSimple)];
        assert!(rules(&validate(&flags)).contains(&"draft_strategy_without_draft_model"));
    }

    #[test]
    fn ngram_params_under_a_draft_strategy_are_refused() {
        let flags = vec![
            Flag::SpecType(SpecType::DraftSimple),
            Flag::DraftModel(r"C:\d.gguf".into()),
            Flag::SpecNgramSimpleSizeM(4),
        ];
        assert!(rules(&validate(&flags)).contains(&"ngram_params_without_ngram_strategy"));
    }

    #[test]
    fn quantized_kv_without_flash_attn_warns_but_does_not_error() {
        let flags = vec![
            Flag::CacheTypeK(CacheType::Q8_0),
            Flag::FlashAttn(FlashAttn::Off),
        ];
        let v = validate(&flags);
        assert!(rules(&v).contains(&"quantized_kv_without_flash_attn"));
        assert!(v.iter().all(|x| x.severity == Severity::Warning));
    }

    #[test]
    fn unquantized_kv_without_flash_attn_is_clean() {
        let flags = vec![
            Flag::CacheTypeK(CacheType::F16),
            Flag::FlashAttn(FlashAttn::Off),
        ];
        assert!(validate(&flags).is_empty());
    }

    #[test]
    fn a_correct_ngram_configuration_is_clean() {
        // The zero-cost speculation path: no draft model, no extra VRAM, no RPC worker.
        let flags = vec![
            Flag::SpecType(SpecType::NgramSimple),
            Flag::SpecNgramSimpleSizeN(8),
            Flag::SpecNgramSimpleSizeM(4),
            Flag::SpecNgramSimpleMinHits(1),
            Flag::CacheTypeK(CacheType::Q8_0),
            Flag::FlashAttn(FlashAttn::On),
            Flag::Temp(0.0),
        ];
        assert!(validate(&flags).is_empty(), "{:?}", validate(&flags));
    }
}
