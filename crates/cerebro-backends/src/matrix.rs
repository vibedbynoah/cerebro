//! The capability matrix: detected hardware mapped to every feature, with the reason each
//! unavailable one is unavailable.
//!
//! A feature is provided by one or more backends, each possibly with extra conditions of its own
//! ([`PROVISIONS`]): FlashAttention 3 comes from vLLM or SGLang *and* needs a Hopper GPU. A feature
//! is **available** when at least one provider's backend requirements and the feature's extra
//! conditions all hold on this machine and Cerebro has a launch recipe for that provider.
//! Otherwise it is **unavailable** with every provider's reasons, or **unknown** when no provider
//! is ruled out but one could not be checked. The matrix answers "does this hardware support it",
//! not "is it installed": installing a backend is a separate, explicit step.
//!
//! Three independent facts decide what a row says: the hardware supports it ([`Availability`]),
//! Cerebro can launch the provider ([`crate::backend::LaunchRecipe`]; a missing recipe is a
//! [`Reason::NotLaunchable`] of its own, so a user whose hardware matches sees exactly that), and
//! Cerebro has run the feature on that provider ([`VERIFIED_FEATURES`], per (backend, feature)
//! pair, never per backend: a backend that launches has not thereby run every feature it models).
//!
//! Every feature is switchable ([`Override`]): `Auto` follows detection, `ForceOn` turns it on
//! anyway for a user who has hardware the probe could not see (one neutral warning, recorded),
//! and `ForceOff` turns it off.

use std::collections::BTreeMap;

use cerebro_probe::hardware::{AmdArch, GpuVendor, HardwareProfile, Isa, NpuArch, NvidiaArch};

use crate::backend::{BackendId, LaunchRecipe, LlamaCppBuild, Verification};
use crate::external::NVIDIA_AMPERE_PLUS;
use crate::feature::{Feature, Override};
use crate::llamacpp::NVIDIA_TURING_PLUS;
use crate::registry::BackendRegistry;
use crate::requirement::{check_all, Check, Reason, Requirement};

/// Which backends a provision row covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Providers {
    /// Every llama.cpp build.
    AllLlamaCpp,
    /// These backends.
    Only(&'static [BackendId]),
}

impl Providers {
    fn ids(self) -> Vec<BackendId> {
        match self {
            Self::AllLlamaCpp => LlamaCppBuild::ALL.iter().map(|b| BackendId::LlamaCpp(*b)).collect(),
            Self::Only(list) => list.to_vec(),
        }
    }
}

/// One row: `feature` is provided by `providers` when `extra` also holds.
#[derive(Debug, Clone, Copy)]
pub struct Provision {
    /// The feature.
    pub feature: Feature,
    /// Who provides it.
    pub providers: Providers,
    /// Conditions beyond the backend's own requirements.
    pub extra: &'static [Requirement],
}

const CUDA12: BackendId = BackendId::LlamaCpp(LlamaCppBuild::Cuda12);
const CUDA13: BackendId = BackendId::LlamaCpp(LlamaCppBuild::Cuda13);
const HIP: BackendId = BackendId::LlamaCpp(LlamaCppBuild::Hip);
const SYCL: BackendId = BackendId::LlamaCpp(LlamaCppBuild::Sycl);
const VULKAN: BackendId = BackendId::LlamaCpp(LlamaCppBuild::Vulkan);
const DATACENTER: &[BackendId] = &[BackendId::Vllm, BackendId::Sglang];

const ADA_PLUS: &[NvidiaArch] = &[
    NvidiaArch::Ada,
    NvidiaArch::Hopper,
    NvidiaArch::BlackwellDatacenter,
    NvidiaArch::BlackwellClient,
];
const BLACKWELL: &[NvidiaArch] = &[NvidiaArch::BlackwellDatacenter, NvidiaArch::BlackwellClient];

const fn gpus(vendors: &'static [GpuVendor], min: usize) -> Requirement {
    Requirement::Gpus {
        vendors,
        min,
        discrete_only: true,
    }
}

/// The feature table. Order within a feature is the order providers are reported in.
pub const PROVISIONS: &[Provision] = &[
    // ---- quantization ----
    Provision { feature: Feature::GgufQuants, providers: Providers::AllLlamaCpp, extra: &[] },
    Provision {
        feature: Feature::Fp8,
        providers: Providers::Only(DATACENTER),
        extra: &[Requirement::AnyOf(&[
            Requirement::NvidiaArch(ADA_PLUS),
            Requirement::AmdArch(&[AmdArch::Cdna3]),
        ])],
    },
    Provision {
        feature: Feature::Nvfp4,
        providers: Providers::Only(DATACENTER),
        extra: &[Requirement::NvidiaArch(BLACKWELL)],
    },
    Provision {
        feature: Feature::AwqGptq,
        providers: Providers::Only(DATACENTER),
        extra: &[Requirement::NvidiaArch(NVIDIA_TURING_PLUS)],
    },
    Provision { feature: Feature::Exl3, providers: Providers::Only(&[BackendId::ExLlamaV3]), extra: &[] },
    Provision { feature: Feature::MlxQuant, providers: Providers::Only(&[BackendId::Mlx]), extra: &[] },
    Provision {
        feature: Feature::OpenVinoInt4,
        providers: Providers::Only(&[BackendId::OpenVino]),
        extra: &[],
    },
    // ---- attention ----
    Provision { feature: Feature::LlamaFlashAttention, providers: Providers::AllLlamaCpp, extra: &[] },
    Provision {
        feature: Feature::FlashAttention2,
        providers: Providers::Only(&[BackendId::Vllm, BackendId::Sglang, BackendId::ExLlamaV3]),
        extra: &[Requirement::NvidiaArch(NVIDIA_AMPERE_PLUS)],
    },
    Provision {
        feature: Feature::FlashAttention3,
        providers: Providers::Only(DATACENTER),
        extra: &[Requirement::NvidiaArch(&[NvidiaArch::Hopper])],
    },
    // ---- multi-GPU ----
    Provision {
        feature: Feature::LayerSplit,
        providers: Providers::Only(&[CUDA12, CUDA13]),
        extra: &[gpus(&[GpuVendor::Nvidia], 2)],
    },
    Provision { feature: Feature::LayerSplit, providers: Providers::Only(&[HIP]), extra: &[gpus(&[GpuVendor::Amd], 2)] },
    Provision { feature: Feature::LayerSplit, providers: Providers::Only(&[SYCL]), extra: &[gpus(&[GpuVendor::Intel], 2)] },
    Provision { feature: Feature::LayerSplit, providers: Providers::Only(&[VULKAN]), extra: &[gpus(&[], 2)] },
    Provision {
        feature: Feature::RowSplit,
        providers: Providers::Only(&[CUDA12, CUDA13]),
        extra: &[gpus(&[GpuVendor::Nvidia], 2)],
    },
    Provision { feature: Feature::RowSplit, providers: Providers::Only(&[HIP]), extra: &[gpus(&[GpuVendor::Amd], 2)] },
    // One process group shards across GPUs of one vendor (one CUDA or one ROCm group), so an
    // NVIDIA card and an AMD card do not make two tensor-parallel GPUs.
    Provision {
        feature: Feature::TensorParallel,
        providers: Providers::Only(DATACENTER),
        extra: &[Requirement::AnyOf(&[gpus(&[GpuVendor::Nvidia], 2), gpus(&[GpuVendor::Amd], 2)])],
    },
    Provision {
        feature: Feature::NvlinkPeerToPeer,
        providers: Providers::Only(&[CUDA12, CUDA13, BackendId::Vllm, BackendId::Sglang]),
        extra: &[Requirement::NvlinkAllPairs],
    },
    // ---- speculative decoding ----
    Provision { feature: Feature::SpecNgram, providers: Providers::AllLlamaCpp, extra: &[] },
    Provision { feature: Feature::SpecNgram, providers: Providers::Only(&[BackendId::Vllm]), extra: &[] },
    Provision { feature: Feature::SpecDraftModel, providers: Providers::AllLlamaCpp, extra: &[] },
    Provision { feature: Feature::SpecDraftModel, providers: Providers::Only(DATACENTER), extra: &[] },
    Provision { feature: Feature::SpecEagle3, providers: Providers::AllLlamaCpp, extra: &[] },
    Provision { feature: Feature::SpecEagle3, providers: Providers::Only(DATACENTER), extra: &[] },
    Provision { feature: Feature::SpecMtp, providers: Providers::AllLlamaCpp, extra: &[] },
    Provision { feature: Feature::SpecMtp, providers: Providers::Only(DATACENTER), extra: &[] },
    // ---- NPU offload ----
    Provision {
        feature: Feature::NpuOffloadIntel,
        providers: Providers::Only(&[BackendId::OpenVino]),
        extra: &[Requirement::Npu(&[NpuArch::IntelNpu3, NpuArch::IntelNpu4])],
    },
    Provision {
        feature: Feature::NpuOffloadAmd,
        providers: Providers::Only(&[BackendId::RyzenAi]),
        extra: &[],
    },
    Provision {
        feature: Feature::NpuOffloadQualcomm,
        providers: Providers::Only(&[BackendId::Qnn]),
        extra: &[],
    },
    // ---- CPU kernels (every llama.cpp build carries the CPU backend) ----
    Provision {
        feature: Feature::CpuAvx512Kernels,
        providers: Providers::AllLlamaCpp,
        extra: &[Requirement::Isa(Isa::Avx512F)],
    },
    Provision {
        feature: Feature::CpuAmxKernels,
        providers: Providers::AllLlamaCpp,
        extra: &[Requirement::Isa(Isa::AmxInt8)],
    },
    Provision {
        feature: Feature::CpuArmI8mmKernels,
        providers: Providers::AllLlamaCpp,
        extra: &[Requirement::Isa(Isa::I8mm)],
    },
];

/// (backend, feature) pairs a live test has run end to end with this crate's argv. A row is added
/// only in the same change as the `#[ignore]` live test that exercises it, named in a comment on
/// the row; everything else reads "supported by your hardware; not yet verified", never as working
/// (docs/CLAIMS.md: run the binary to make the claim).
///
/// Not listed, on purpose: `GgufQuants` (the only model run is an f16 GGUF; no K- or I-quant has
/// run), multi-GPU splits and NVLink (no multi-GPU machine), speculative decoding, the CPU kernel
/// families, and every non-llama.cpp backend.
pub const VERIFIED_FEATURES: &[(BackendId, Feature)] = &[
    // tests/live.rs::live_cuda12_flash_attention_serves_a_completion: `-fa on` with the tiny f16
    // GGUF on the official CUDA 12 build; the server log reports flash attention enabled and an
    // authenticated completion returns text.
    (CUDA12, Feature::LlamaFlashAttention),
];

/// Whether `feature` has been run on backend `id`: the backend launches (verified) and the pair is
/// in [`VERIFIED_FEATURES`].
fn feature_verified(registry: &BackendRegistry, id: BackendId, feature: Feature) -> bool {
    registry
        .get(id)
        .is_some_and(|b| b.verification() == Verification::Verified)
        && VERIFIED_FEATURES.contains(&(id, feature))
}

/// Features a backend can provide (before their extra conditions).
pub fn features_of(backend: BackendId) -> Vec<Feature> {
    let mut out: Vec<Feature> = Vec::new();
    for p in PROVISIONS {
        if p.providers.ids().contains(&backend) && !out.contains(&p.feature) {
            out.push(p.feature);
        }
    }
    out
}

/// Whether a feature can run on this hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// Supported, by these backends.
    Available {
        /// Backends that can provide it here.
        via: Vec<BackendId>,
    },
    /// Not supported: every provider is ruled out, with its reasons. Empty when the registry
    /// holds no backend that provides the feature at all.
    Unavailable {
        /// Each provider and why it cannot provide the feature here.
        blocked: Vec<(BackendId, Vec<Reason>)>,
    },
    /// No provider is ruled out, but none could be confirmed.
    Unknown {
        /// Each unconfirmed provider and what could not be checked.
        unchecked: Vec<(BackendId, Vec<Reason>)>,
    },
}

impl Availability {
    /// Whether available.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }

    /// The single most actionable reason: the first reason of the provider with the fewest.
    pub fn main_reason(&self) -> Option<&Reason> {
        let list = match self {
            Self::Available { .. } => return None,
            Self::Unavailable { blocked } => blocked,
            Self::Unknown { unchecked } => unchecked,
        };
        list.iter()
            .min_by_key(|(_, reasons)| reasons.len())
            .and_then(|(_, reasons)| reasons.first())
    }
}

/// One feature's row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureState {
    /// The feature.
    pub feature: Feature,
    /// What the hardware supports.
    pub availability: Availability,
    /// The user's setting.
    pub setting: Override,
    /// Whether Cerebro has run this feature on at least one backend that provides it here
    /// ([`VERIFIED_FEATURES`]). An available feature that is not verified is shown as supported by
    /// the hardware, never as working (docs/CLAIMS.md: run the binary to make the claim).
    pub verified: bool,
    /// The providers available here on which Cerebro has run the feature, in provider order.
    pub verified_via: Vec<BackendId>,
}

/// Capitalize the first letter of a reason for a sentence start.
fn sentence(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

impl FeatureState {
    /// Whether the feature is on: the user's override, or detection under `Auto`.
    pub fn enabled(&self) -> bool {
        match self.setting {
            Override::ForceOn => true,
            Override::ForceOff => false,
            Override::Auto => self.availability.is_available(),
        }
    }

    /// The line the app shows: "Available on your hardware (llama.cpp (CUDA 12))" (only for
    /// providers Cerebro has run it on), "Supported by your hardware (...); not yet verified by
    /// Cerebro", "Needs an NVIDIA Hopper GPU (found Ampere)", "Could not check for ...".
    pub fn summary(&self) -> String {
        match &self.availability {
            Availability::Available { via } => {
                let names = |ids: &mut dyn Iterator<Item = &BackendId>| -> String {
                    ids.map(|b| b.name()).collect::<Vec<_>>().join(", ")
                };
                let mut unverified = via.iter().filter(|b| !self.verified_via.contains(b)).peekable();
                if self.verified_via.is_empty() {
                    format!("Supported by your hardware ({}); not yet verified by Cerebro", names(&mut unverified))
                } else if unverified.peek().is_none() {
                    format!("Available on your hardware ({})", names(&mut self.verified_via.iter()))
                } else {
                    format!(
                        "Available on your hardware ({}); also supported by your hardware, not yet \
                         verified by Cerebro: {}",
                        names(&mut self.verified_via.iter()),
                        names(&mut unverified)
                    )
                }
            }
            // No reason at all: the registry holds no backend that provides it.
            other => other.main_reason().map_or_else(
                || "Not provided by any backend in this build".to_string(),
                |r| sentence(&r.to_string()),
            ),
        }
    }

    /// The one neutral warning recorded when the user forced on a feature detection did not
    /// confirm; `None` otherwise (never repeated, never for `Auto` or `ForceOff`).
    pub fn warning(&self) -> Option<String> {
        if self.setting != Override::ForceOn || self.availability.is_available() {
            return None;
        }
        let why = self
            .availability
            .main_reason()
            .map_or_else(String::new, |r| format!(": {r}"));
        Some(format!("{} is on by your choice; detection did not confirm support{why}", self.feature))
    }
}

/// Every feature evaluated against one machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityMatrix {
    states: Vec<FeatureState>,
}

impl CapabilityMatrix {
    /// Evaluate every feature in [`Feature::ALL`] for `hw`, using the backends in `registry`, with
    /// the user's overrides (missing means `Auto`).
    pub fn evaluate(
        hw: &HardwareProfile,
        registry: &BackendRegistry,
        overrides: &BTreeMap<Feature, Override>,
    ) -> Self {
        let states = Feature::ALL
            .iter()
            .map(|&feature| {
                let availability = availability(feature, hw, registry);
                let verified_via: Vec<BackendId> = match &availability {
                    Availability::Available { via } => via
                        .iter()
                        .copied()
                        .filter(|id| feature_verified(registry, *id, feature))
                        .collect(),
                    _ => Vec::new(),
                };
                FeatureState {
                    feature,
                    availability,
                    setting: overrides.get(&feature).copied().unwrap_or_default(),
                    verified: !verified_via.is_empty(),
                    verified_via,
                }
            })
            .collect();
        Self { states }
    }

    /// One feature's row.
    pub fn get(&self, feature: Feature) -> Option<&FeatureState> {
        self.states.iter().find(|s| s.feature == feature)
    }

    /// Every row, in [`Feature::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = &FeatureState> {
        self.states.iter()
    }

    /// The warnings to record, one per forced-on feature detection did not confirm.
    pub fn warnings(&self) -> Vec<String> {
        self.states.iter().filter_map(FeatureState::warning).collect()
    }
}

fn availability(feature: Feature, hw: &HardwareProfile, registry: &BackendRegistry) -> Availability {
    let mut via = Vec::new();
    let mut blocked = Vec::new();
    let mut unchecked = Vec::new();
    for provision in PROVISIONS.iter().filter(|p| p.feature == feature) {
        for id in provision.providers.ids() {
            let Some(backend) = registry.get(id) else { continue };
            // Feature conditions first: "needs a Hopper GPU" says more than "needs WSL".
            let extra = check_all(provision.extra, hw);
            let base = backend.support(hw);
            let mut unmet: Vec<Reason> = Vec::new();
            let mut unknown: Vec<Reason> = Vec::new();
            for check in [extra, base] {
                match check {
                    Check::Met => {}
                    Check::Unmet(r) => unmet.extend(r),
                    Check::Unknown(r) => unknown.extend(r),
                }
            }
            // No launch recipe: blocked whatever the hardware. Established hardware gaps come
            // first (more actionable for a user without the hardware), then the recipe (decisive
            // over anything that merely could not be checked); alone, it tells a user whose
            // hardware matches that only Cerebro is missing.
            if let LaunchRecipe::Missing(detail) = backend.launch_recipe() {
                let mut reasons = unmet;
                reasons.push(Reason::NotLaunchable { backend: id, detail });
                reasons.extend(unknown);
                blocked.push((id, reasons));
                continue;
            }
            if !unmet.is_empty() {
                blocked.push((id, unmet));
            } else if !unknown.is_empty() {
                unchecked.push((id, unknown));
            } else if !via.contains(&id) {
                via.push(id);
            }
        }
    }
    if !via.is_empty() {
        Availability::Available { via }
    } else if !unchecked.is_empty() {
        Availability::Unknown { unchecked }
    } else {
        Availability::Unavailable { blocked }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_probe::hardware::profiles;

    fn matrix(hw: &HardwareProfile) -> CapabilityMatrix {
        CapabilityMatrix::evaluate(hw, &BackendRegistry::standard(), &BTreeMap::new())
    }

    fn available(m: &CapabilityMatrix, f: Feature) -> bool {
        m.get(f).unwrap().availability.is_available()
    }

    #[test]
    fn every_feature_has_a_provider_and_a_row() {
        for f in Feature::ALL {
            assert!(PROVISIONS.iter().any(|p| p.feature == *f), "{f} has no provision");
        }
        let m = matrix(&profiles::rtx3070ti_laptop().unwrap());
        assert_eq!(m.iter().count(), Feature::ALL.len());
    }

    #[test]
    fn the_development_laptop() {
        let m = matrix(&profiles::rtx3070ti_laptop().unwrap());
        for f in [Feature::GgufQuants, Feature::LlamaFlashAttention, Feature::SpecNgram, Feature::SpecEagle3] {
            assert!(available(&m, f), "{f}: {}", m.get(f).unwrap().summary());
        }
        // GGUF quants: supported, but no quantized GGUF has run, so not claimed as working.
        let ga = m.get(Feature::GgufQuants).unwrap().summary();
        assert!(
            ga.starts_with("Supported by your hardware (llama.cpp (CUDA 12), llama.cpp (CUDA 13), llama.cpp (Vulkan)"),
            "{ga}"
        );
        assert!(ga.ends_with("; not yet verified by Cerebro"), "{ga}");
        // Flash attention has run on CUDA 12 only: that provider reads as working, the rest not.
        let fa = m.get(Feature::LlamaFlashAttention).unwrap();
        assert_eq!(fa.verified_via, [CUDA12]);
        assert!(
            fa.summary().starts_with(
                "Available on your hardware (llama.cpp (CUDA 12)); also supported by your hardware, \
                 not yet verified by Cerebro: llama.cpp (CUDA 13)"
            ),
            "{}",
            fa.summary()
        );
        assert_eq!(
            m.get(Feature::FlashAttention3).unwrap().summary(),
            "Needs an NVIDIA Hopper GPU (found Ampere)"
        );
        assert_eq!(
            m.get(Feature::Fp8).unwrap().summary(),
            "Needs one of: an NVIDIA Ada Lovelace, Hopper, Blackwell (datacenter) or Blackwell GPU; an AMD CDNA 3 GPU"
        );
        assert_eq!(
            m.get(Feature::LayerSplit).unwrap().summary(),
            "Needs 2 or more discrete NVIDIA GPUs (found 1)"
        );
        assert!(!available(&m, Feature::CpuAvx512Kernels));
        assert!(!available(&m, Feature::NpuOffloadIntel));
        assert!(!m.get(Feature::GgufQuants).unwrap().verified);
        // FlashAttention 2 hardware is there (Ampere) but vLLM/SGLang need WSL with a distro
        // and ExLlamaV3 is not ruled out: available via ExLlamaV3.
        assert_eq!(
            m.get(Feature::FlashAttention2).unwrap().availability,
            Availability::Available { via: vec![BackendId::ExLlamaV3] }
        );
    }

    #[test]
    fn hopper_server_gets_fa3_fp8_and_amx_but_not_cuda13() {
        let m = matrix(&profiles::h100_server().unwrap());
        for f in [Feature::FlashAttention3, Feature::Fp8, Feature::CpuAmxKernels, Feature::CpuAvx512Kernels, Feature::AwqGptq] {
            assert!(available(&m, f), "{f}: {}", m.get(f).unwrap().summary());
        }
        assert!(!available(&m, Feature::Nvfp4));
        assert!(!available(&m, Feature::TensorParallel));
        let fa3 = &m.get(Feature::FlashAttention3).unwrap().availability;
        assert_eq!(fa3, &Availability::Available { via: vec![BackendId::Vllm, BackendId::Sglang] });
    }

    #[test]
    fn rtx4090_gets_fp8_through_wsl_but_not_fa3() {
        let m = matrix(&profiles::rtx4090_desktop().unwrap());
        assert!(available(&m, Feature::Fp8));
        assert_eq!(
            m.get(Feature::FlashAttention3).unwrap().summary(),
            "Needs an NVIDIA Hopper GPU (found Ada Lovelace)"
        );
        assert!(available(&m, Feature::CpuAvx512Kernels));
        assert!(!available(&m, Feature::CpuAmxKernels));
    }

    #[test]
    fn dual_3090s_split_tensors_and_use_nvlink() {
        let m = matrix(&profiles::dual_rtx3090_nvlink().unwrap());
        for f in [Feature::LayerSplit, Feature::RowSplit, Feature::TensorParallel, Feature::NvlinkPeerToPeer] {
            assert!(available(&m, f), "{f}: {}", m.get(f).unwrap().summary());
        }
        assert!(!available(&m, Feature::Fp8));
    }

    /// The CUDA 12 build launches (verified), but that does not make every feature it models
    /// working: none of these has been run, so each reads "not yet verified".
    #[test]
    fn only_listed_pairs_read_as_working() {
        let m = matrix(&profiles::dual_rtx3090_nvlink().unwrap());
        for f in [
            Feature::LayerSplit,
            Feature::RowSplit,
            Feature::NvlinkPeerToPeer,
            Feature::SpecEagle3,
            Feature::SpecMtp,
            Feature::GgufQuants,
        ] {
            let state = m.get(f).unwrap();
            assert!(state.availability.is_available(), "{f}");
            assert!(!state.verified && state.verified_via.is_empty(), "{f}");
            assert!(state.summary().contains("not yet verified"), "{f}: {}", state.summary());
            assert!(!state.summary().starts_with("Available"), "{f}: {}", state.summary());
        }
        // Every listed pair names a backend that itself launches (verified).
        let reg = BackendRegistry::standard();
        for (id, f) in VERIFIED_FEATURES {
            assert_eq!(reg.get(*id).unwrap().verification(), Verification::Verified, "{id} {f}");
            assert!(features_of(*id).contains(f), "{id} does not provide {f}");
        }
    }

    /// One NVIDIA and one AMD discrete GPU are two GPUs, but not two of one vendor: no tensor
    /// parallelism (vLLM/SGLang shard within one CUDA or one ROCm group).
    #[test]
    fn mixed_vendor_gpus_are_not_tensor_parallel() {
        let mut hw = profiles::rtx4090_desktop().unwrap();
        let amd = profiles::rx7900xtx().unwrap().gpus.known().unwrap()[0].clone();
        if let cerebro_probe::hardware::Detected::Known(gpus) = &mut hw.gpus {
            gpus.push(amd);
        }
        let m = matrix(&hw);
        let tp = m.get(Feature::TensorParallel).unwrap();
        assert!(!tp.availability.is_available(), "{}", tp.summary());
        // Two of one vendor still is.
        assert!(available(&matrix(&profiles::dual_rtx3090_nvlink().unwrap()), Feature::TensorParallel));
    }

    /// QNN matches the Snapdragon's hardware but has no launch recipe: never "available", and the
    /// reason names what is missing rather than the hardware.
    #[test]
    fn qnn_is_modelled_but_never_offered() {
        let sd = profiles::snapdragon_x_elite().unwrap();
        let m = matrix(&sd);
        let npu = m.get(Feature::NpuOffloadQualcomm).unwrap();
        assert!(!npu.availability.is_available());
        assert!(!npu.enabled());
        assert_eq!(
            npu.summary(),
            "Cerebro cannot launch Qualcomm QNN yet: the QNN SDK has no OpenAI-compatible server; \
             a serving shim is not built yet"
        );
        let Availability::Unavailable { blocked } = &npu.availability else { panic!() };
        assert_eq!(blocked[0].1.len(), 1, "the hardware matched: only the recipe is missing");
        // Elsewhere the hardware reason comes first.
        let laptop = matrix(&profiles::rtx3070ti_laptop().unwrap());
        assert!(laptop.get(Feature::NpuOffloadQualcomm).unwrap().summary().starts_with("Needs"));
    }

    #[test]
    fn amd_intel_snapdragon_and_npu_machines() {
        let amd = matrix(&profiles::rx7900xtx().unwrap());
        let gguf = &amd.get(Feature::GgufQuants).unwrap().availability;
        let Availability::Available { via } = gguf else { panic!("{gguf:?}") };
        assert!(via.contains(&HIP) && via.contains(&VULKAN), "{via:?}");
        assert!(!via.contains(&CUDA12));
        assert!(!available(&amd, Feature::TensorParallel));

        let arc = matrix(&profiles::arc_a770().unwrap());
        let Availability::Available { via } = &arc.get(Feature::GgufQuants).unwrap().availability else { panic!() };
        assert!(via.contains(&SYCL), "{via:?}");
        // One discrete Arc plus an iGPU: no multi-GPU split.
        assert!(!available(&arc, Feature::LayerSplit));

        let sd = matrix(&profiles::snapdragon_x_elite().unwrap());
        // QNN matches the hardware but cannot be launched (see qnn_is_modelled_but_never_offered).
        assert!(!available(&sd, Feature::NpuOffloadQualcomm));
        // No CUDA 12 build on Snapdragon: nothing that provides features there is verified.
        assert!(sd.iter().all(|s| !s.verified));
        assert!(available(&sd, Feature::CpuArmI8mmKernels));
        assert!(!available(&sd, Feature::NpuOffloadIntel));

        let mtl = matrix(&profiles::core_ultra_155h().unwrap());
        assert!(available(&mtl, Feature::NpuOffloadIntel));
        let strix = matrix(&profiles::ryzen_ai_hx370().unwrap());
        assert!(available(&strix, Feature::NpuOffloadAmd));
        assert_eq!(
            mtl.get(Feature::NpuOffloadAmd).unwrap().summary(),
            "Needs an AMD XDNA 2 NPU (found Intel AI Boost 3720)"
        );

        let mac = matrix(&profiles::apple_m3_max().unwrap());
        assert!(available(&mac, Feature::MlxQuant));
        assert!(!available(&mac, Feature::Exl3));
    }

    #[test]
    fn nothing_detected_is_unknown_not_unavailable() {
        let m = matrix(&HardwareProfile::unknown("detection failed"));
        for state in m.iter() {
            // The one exception: a feature whose every provider Cerebro cannot launch is
            // unavailable whatever the hardware, and says why.
            if state.feature == Feature::NpuOffloadQualcomm {
                assert!(!state.availability.is_available());
                assert!(state.summary().starts_with("Cerebro cannot launch Qualcomm QNN yet"), "{}", state.summary());
                continue;
            }
            assert!(
                matches!(state.availability, Availability::Unknown { .. }),
                "{}: {:?}",
                state.feature,
                state.availability
            );
            assert!(state.summary().starts_with("Could not check"), "{}", state.summary());
        }
    }

    #[test]
    fn overrides_switch_features_and_warn_once_when_forcing_past_detection() {
        let hw = profiles::rtx3070ti_laptop().unwrap();
        let overrides = BTreeMap::from([
            (Feature::FlashAttention3, Override::ForceOn),
            (Feature::GgufQuants, Override::ForceOn),
            (Feature::SpecNgram, Override::ForceOff),
        ]);
        let m = CapabilityMatrix::evaluate(&hw, &BackendRegistry::standard(), &overrides);
        let fa3 = m.get(Feature::FlashAttention3).unwrap();
        assert!(fa3.enabled());
        assert!(!m.get(Feature::SpecNgram).unwrap().enabled());
        assert!(m.get(Feature::LlamaFlashAttention).unwrap().enabled());
        assert!(!m.get(Feature::Fp8).unwrap().enabled());
        // One warning: FA3 forced past detection. Forcing an available feature warns nothing.
        let warnings = m.warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0],
            "FlashAttention 3 is on by your choice; detection did not confirm support: needs an NVIDIA Hopper GPU (found Ampere)"
        );
    }

    #[test]
    fn a_feature_no_registered_backend_provides_says_so() {
        use crate::llamacpp::LlamaCpp;
        let only_cpu = BackendRegistry::from_backends(vec![Box::new(LlamaCpp::new(LlamaCppBuild::Cpu))]);
        let hw = profiles::h100_server().unwrap();
        let m = CapabilityMatrix::evaluate(&hw, &only_cpu, &BTreeMap::new());
        let fa3 = m.get(Feature::FlashAttention3).unwrap();
        assert_eq!(fa3.availability, Availability::Unavailable { blocked: vec![] });
        assert_eq!(fa3.summary(), "Not provided by any backend in this build");
        // The CPU build alone is unverified, so even GGUF reads as supported, not working.
        assert!(m.get(Feature::GgufQuants).unwrap().summary().starts_with("Supported by your hardware"));
    }

    #[test]
    fn features_of_lists_each_feature_once() {
        let cuda = features_of(CUDA12);
        assert!(cuda.contains(&Feature::LayerSplit) && cuda.contains(&Feature::RowSplit));
        let mut dedup = cuda.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), cuda.len());
        assert!(features_of(BackendId::Qnn) == vec![Feature::NpuOffloadQualcomm]);
        assert!(feature_verified(&BackendRegistry::standard(), CUDA12, Feature::LlamaFlashAttention));
        assert!(!feature_verified(&BackendRegistry::standard(), CUDA13, Feature::LlamaFlashAttention));
    }
}
