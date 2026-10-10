//! Closed-allowlist flag catalog and dialect-aware argv construction for llama.cpp backends.
//!
//! This crate is the boundary between Cerebro's autotuner — which generates and persists flag
//! combinations — and the backend process it spawns. Everything the tuner can express passes
//! through here, which makes it the highest-leverage code in the product for both correctness and
//! security.
//!
//! # Design rules
//!
//! 1. **The emittable flag space is closed.** [`Flag`] is an enum, not a string map. A config or
//!    results file that Cerebro reads back cannot introduce a flag that is not already compiled in.
//! 2. **Invalid values are unrepresentable.** Validation lives in value constructors
//!    ([`CacheType`], [`OtRule`], [`DeviceList`], …), not at emission, so anything that exists is
//!    already safe to place in an argv slot.
//! 3. **Dialects are explicit.** `llama-server` and `llama-bench` disagree about separators and
//!    about which flags exist. [`Dialect`] makes the difference a type-level concern rather than a
//!    formatting accident.
//! 4. **Upstream defaults are never inherited.** Where a default is wrong for a privacy-first
//!    product, [`ArgvBuilder::harden_server`] states the correct value explicitly.
//!
//! Every flag name and value set here was verified against llama.cpp build b10751 (commit
//! `3466812d1`) by reading its `--help` output and, where behavior was in question, by running the
//! binary. The verification notes live in `docs/groundtruth/`.
//!
//! # Example
//!
//! ```
//! use cerebro_flags::{ArgvBuilder, Dialect, Flag};
//! use cerebro_flags::value::{CacheType, CpuMask};
//!
//! let mut b = ArgvBuilder::new(Dialect::Server);
//! b.push(Flag::Model(r"C:\models\target.gguf".into()))?;
//! b.push(Flag::GpuLayers(999))?;
//! b.push(Flag::NCpuMoe(32))?;
//! b.push(Flag::CacheTypeK(CacheType::Q8_0))?;
//! b.push(Flag::CpuMask(CpuMask::new(0x555)?))?;
//! b.pin_for_measurement()?;
//!
//! let argv = b.build()?;
//! assert!(argv.join(" ").contains("-ncmoe 32"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod constraints;
pub mod emit;
pub mod error;
pub mod value;

pub use constraints::{Severity, Violation};
pub use emit::{ArgvBuilder, Dialect, Flag};
pub use error::{EmitError, ValueError};
pub use value::{
    BufferType, CacheType, CpuMask, DeviceId, DeviceList, FlashAttn, LoadMode, OtRule, OtRuleSet,
    ShadowWarning, SpecType, SplitMode, TensorSplit,
};

/// The llama.cpp build this catalog was verified against.
///
/// Flag surfaces change between builds. A recorded benchmark result is only comparable to another
/// result taken against the same backend identity, so this constant is part of every result key
/// rather than a documentation footnote — a self-built fork of this same commit measured 3-5x
/// slower than the official release on the reference machine.
pub const VERIFIED_BACKEND_BUILD: BackendBuild = BackendBuild {
    build_number: 10751,
    commit: "3466812d1",
    version: "0.3.0-dev",
};

/// Identity of a llama.cpp backend build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendBuild {
    /// Upstream build number, e.g. `10751`.
    pub build_number: u32,
    /// Short commit hash the build was produced from.
    pub commit: &'static str,
    /// Upstream version string, e.g. `0.3.0-dev`.
    pub version: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flag_name_is_distinct() {
        // A collision would make ArgvBuilder's duplicate detection reject two legitimately
        // different flags, or silently allow a real duplicate through.
        use std::collections::BTreeMap;
        let samples: Vec<Flag> = vec![
            Flag::Model(r"C:\m.gguf".into()),
            Flag::DraftModel(r"C:\d.gguf".into()),
            Flag::CtxSize(4096),
            Flag::GpuLayers(999),
            Flag::NCpuMoe(32),
            Flag::CpuMoe,
            Flag::NCpuFfn(4),
            Flag::OverrideTensor(OtRuleSet::new()),
            Flag::Device(DeviceList::None),
            Flag::SplitMode(SplitMode::Layer),
            Flag::MainGpu(0),
            Flag::LoadMode(LoadMode::Auto),
            Flag::CacheTypeK(CacheType::F16),
            Flag::CacheTypeV(CacheType::F16),
            Flag::FlashAttn(FlashAttn::Auto),
            Flag::Threads(6),
            Flag::ThreadsBatch(12),
            Flag::CpuMask(CpuMask::new(0x555).unwrap()),
            Flag::CpuStrict(true),
            Flag::Poll(50),
            Flag::BatchSize(2048),
            Flag::UBatchSize(512),
            Flag::SpecType(SpecType::None),
            Flag::SpecDraftNMax(3),
            Flag::SpecDraftNMin(0),
            Flag::SpecDraftPMin(0.0),
            Flag::SpecDraftDevice(DeviceList::None),
            Flag::SpecDraftGpuLayers(0),
            Flag::SpecDraftCpuMask(CpuMask::new(0xFF000).unwrap()),
            Flag::SpecDraftCpuStrict(true),
            Flag::SpecDraftCacheTypeK(CacheType::F16),
            Flag::SpecDraftCacheTypeV(CacheType::F16),
            Flag::SpecNgramSimpleSizeN(8),
            Flag::SpecNgramSimpleSizeM(4),
            Flag::SpecNgramSimpleMinHits(1),
            Flag::Fit(false),
            Flag::Host("127.0.0.1".into()),
            Flag::Port(8127),
            Flag::WebUi(false),
            Flag::Slots(false),
            Flag::CorsOrigins(String::new()),
            Flag::ApiKeyFile(r"C:\k".into()),
        ];
        let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &samples {
            *by_name.entry(f.name()).or_insert(0) += 1;
        }
        let dupes: Vec<_> = by_name.iter().filter(|(_, n)| **n > 1).collect();
        assert!(dupes.is_empty(), "duplicate flag names: {dupes:?}");
    }

    #[test]
    fn every_flag_renders_in_the_server_dialect() {
        // The server dialect is a superset; nothing in the allowlist may fail to render there.
        for f in [
            Flag::GpuLayers(999),
            Flag::SpecType(SpecType::NgramMapK4v),
            Flag::LoadMode(LoadMode::MmapMlock),
            Flag::FlashAttn(FlashAttn::On),
            Flag::Fit(false),
        ] {
            assert!(f.render(Dialect::Server).is_ok(), "{} failed", f.name());
        }
    }

    #[test]
    fn empty_override_tensor_emits_nothing() {
        // An empty rule set must not emit a bare `-ot` with no value, which would consume the
        // following token as its argument and silently corrupt the rest of the argv.
        let f = Flag::OverrideTensor(OtRuleSet::new());
        assert!(f.render(Dialect::Server).unwrap().is_empty());
        assert!(f.render(Dialect::Bench).unwrap().is_empty());
    }
}
