//! The backend registry and backend selection.
//!
//! Selection produces an ordered **candidate list**, not a verdict. The first supported candidate
//! is the one to try first; the tuner then measures the candidates, because which build is fastest
//! is a measurement on this machine, never an assumption (a CUDA build and a Vulkan build trade
//! places by model and GPU, and a CPU build can beat an integrated GPU sharing its memory bus).
//!
//! **The baseline rule.** llama.cpp's own automatic settings are each build run with no tuning
//! flags (`--fit` and every other default left to upstream): on a GPU machine that is the GPU
//! build with defaults, not the CPU build. The tuner measures every candidate build that way as
//! that build's own baseline and never stores a tuned configuration that loses to it (owner rule:
//! never recommend anything slower than llama.cpp's automatic settings; measured, tuned flags have
//! both won by ~6x and lost to defaults on this machine). The CPU build is always among the GGUF
//! candidates when supported, as a fallback that runs everywhere, not as that baseline.
//!
//! A backend Cerebro has no launch recipe for ([`crate::backend::LaunchRecipe::Missing`]) is never
//! a candidate, whatever the hardware or the user's switches: it would fail every time.

use std::collections::BTreeSet;

use cerebro_probe::hardware::{GpuArch, GpuKind, GpuVendor, HardwareProfile, NvidiaArch};

use crate::backend::{Backend, BackendId, LaunchRecipe, LlamaCppBuild, ModelFormat};
use crate::external::{ExLlamaV3, Mlx, OpenVino, Qnn, RyzenAi, Sglang, Vllm};
use crate::llamacpp::LlamaCpp;
use crate::requirement::{Check, Reason};

/// Every backend Cerebro knows, behind the one [`Backend`] trait.
#[derive(Debug)]
pub struct BackendRegistry {
    backends: Vec<Box<dyn Backend>>,
}

impl BackendRegistry {
    /// The standard registry: every llama.cpp build plus every optional backend.
    pub fn standard() -> Self {
        let mut backends: Vec<Box<dyn Backend>> = LlamaCppBuild::ALL
            .iter()
            .map(|b| Box::new(LlamaCpp::new(*b)) as Box<dyn Backend>)
            .collect();
        backends.push(Box::new(Vllm));
        backends.push(Box::new(Sglang));
        backends.push(Box::new(ExLlamaV3));
        backends.push(Box::new(Mlx));
        backends.push(Box::new(OpenVino));
        backends.push(Box::new(RyzenAi));
        backends.push(Box::new(Qnn));
        Self { backends }
    }

    /// A registry of exactly these backends.
    pub fn from_backends(backends: Vec<Box<dyn Backend>>) -> Self {
        Self { backends }
    }

    /// One backend by id.
    pub fn get(&self, id: BackendId) -> Option<&dyn Backend> {
        self.backends.iter().find(|b| b.id() == id).map(|b| b.as_ref())
    }

    /// Every backend, in registry order.
    pub fn iter(&self) -> impl Iterator<Item = &dyn Backend> {
        self.backends.iter().map(|b| b.as_ref())
    }

    /// Choose backends for a model format on this hardware, honouring the user's policy.
    pub fn select(
        &self,
        hw: &HardwareProfile,
        format: ModelFormat,
        policy: &SelectionPolicy,
    ) -> Selection {
        let order = preference_order(hw, format);
        let mut supported = Vec::new();
        let mut unverifiable = Vec::new();
        let mut forced = Vec::new();
        let mut rejected = Vec::new();
        for id in order {
            let Some(backend) = self.get(id) else { continue };
            if !backend.model_formats().contains(&format) {
                continue;
            }
            if policy.disabled.contains(&id) {
                rejected.push(Rejection {
                    backend: id,
                    cause: RejectionCause::DisabledByUser,
                });
                continue;
            }
            if let LaunchRecipe::Missing(detail) = backend.launch_recipe() {
                rejected.push(Rejection {
                    backend: id,
                    cause: RejectionCause::NoLaunchRecipe(detail),
                });
                continue;
            }
            match backend.support(hw) {
                Check::Met => supported.push(Candidate {
                    backend: id,
                    status: CandidateStatus::Supported,
                }),
                Check::Unknown(reasons) => unverifiable.push(Candidate {
                    backend: id,
                    status: CandidateStatus::Unverifiable(reasons),
                }),
                Check::Unmet(reasons) if policy.force_on.contains(&id) => forced.push(Candidate {
                    backend: id,
                    status: CandidateStatus::ForcedByUser(reasons),
                }),
                Check::Unmet(reasons) => rejected.push(Rejection {
                    backend: id,
                    cause: RejectionCause::Hardware(reasons),
                }),
            }
        }
        let mut candidates = supported;
        candidates.extend(unverifiable);
        candidates.extend(forced);
        Selection {
            format,
            candidates,
            rejected,
        }
    }
}

/// The user's backend switches. Empty (the default) follows detection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SelectionPolicy {
    /// Offer these even when detection rules them out (the user has hardware the probe missed).
    pub force_on: BTreeSet<BackendId>,
    /// Never offer these.
    pub disabled: BTreeSet<BackendId>,
}

/// Why a candidate is on the list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateStatus {
    /// Every requirement is met.
    Supported,
    /// Not ruled out, but some requirement could not be checked.
    Unverifiable(Vec<Reason>),
    /// Ruled out by detection and offered because the user forced it on.
    ForcedByUser(Vec<Reason>),
}

/// One backend to try, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Which backend.
    pub backend: BackendId,
    /// Why it is offered.
    pub status: CandidateStatus,
}

/// Why a backend is not offered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectionCause {
    /// This hardware does not meet its requirements.
    Hardware(Vec<Reason>),
    /// The user turned it off.
    DisabledByUser,
    /// Cerebro cannot launch it yet (the hardware may well match); forcing it on cannot help.
    NoLaunchRecipe(&'static str),
}

/// A backend that serves the format but is not offered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    /// Which backend.
    pub backend: BackendId,
    /// Why.
    pub cause: RejectionCause,
}

/// The outcome of [`BackendRegistry::select`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The format selected for.
    pub format: ModelFormat,
    /// Backends to try, supported first, then unverifiable, then user-forced.
    pub candidates: Vec<Candidate>,
    /// Backends for this format that are not offered, with why.
    pub rejected: Vec<Rejection>,
}

impl Selection {
    /// The backend to try first, if any.
    pub fn first(&self) -> Option<BackendId> {
        self.candidates.first().map(|c| c.backend)
    }

    /// The candidate ids in order.
    pub fn ids(&self) -> Vec<BackendId> {
        self.candidates.iter().map(|c| c.backend).collect()
    }

    /// One neutral warning per backend offered only because the user forced it on.
    pub fn warnings(&self) -> Vec<String> {
        self.candidates
            .iter()
            .filter_map(|c| match &c.status {
                CandidateStatus::ForcedByUser(reasons) => Some(format!(
                    "{} is offered by your choice; detection did not confirm support: {}",
                    c.backend,
                    reasons
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                )),
                _ => None,
            })
            .collect()
    }
}

/// The order backends are tried in for `format` on `hw`.
fn preference_order(hw: &HardwareProfile, format: ModelFormat) -> Vec<BackendId> {
    use LlamaCppBuild as B;
    match format {
        ModelFormat::Gguf => {
            let gpus = hw.gpus.known().map(Vec::as_slice).unwrap_or_default();
            // Blackwell needs native sm_120/sm_100 code to avoid a PTX JIT on first load (measured
            // 20 s vs 1.3 s with JIT forced on Ampere, docs/RESTART.md); CUDA 13 builds carry it.
            let blackwell = gpus.iter().any(|g| {
                matches!(
                    g.arch.known(),
                    Some(GpuArch::Nvidia(NvidiaArch::BlackwellClient | NvidiaArch::BlackwellDatacenter))
                )
            });
            let mut order = if blackwell {
                vec![B::Cuda13, B::Cuda12]
            } else {
                vec![B::Cuda12, B::Cuda13]
            };
            order.extend([B::Hip, B::Sycl, B::Metal, B::OpenClAdreno, B::Vulkan]);
            // Only integrated GPUs (Apple's excepted): the CPU build first, as the baseline the
            // shared-memory GPU must beat.
            let integrated_only = !gpus.is_empty()
                && gpus
                    .iter()
                    .all(|g| g.kind == GpuKind::Integrated && g.vendor != GpuVendor::Apple);
            if integrated_only || gpus.is_empty() {
                order.insert(0, B::Cpu);
            } else {
                order.push(B::Cpu);
            }
            order.into_iter().map(BackendId::LlamaCpp).collect()
        }
        ModelFormat::Safetensors => vec![BackendId::Vllm, BackendId::Sglang],
        ModelFormat::Exl3 => vec![BackendId::ExLlamaV3],
        ModelFormat::Mlx => vec![BackendId::Mlx],
        ModelFormat::OpenVinoIr => vec![BackendId::OpenVino],
        ModelFormat::OnnxGenAi => vec![BackendId::RyzenAi],
        ModelFormat::QnnContext => vec![BackendId::Qnn],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_probe::hardware::{profiles, Detected};

    const fn l(b: LlamaCppBuild) -> BackendId {
        BackendId::LlamaCpp(b)
    }

    fn gguf(hw: &HardwareProfile) -> Selection {
        BackendRegistry::standard().select(hw, ModelFormat::Gguf, &SelectionPolicy::default())
    }

    #[test]
    fn the_standard_registry_has_every_backend_once() {
        let reg = BackendRegistry::standard();
        let ids: Vec<BackendId> = reg.iter().map(|b| b.id()).collect();
        assert_eq!(ids.len(), LlamaCppBuild::ALL.len() + 7);
        let unique: BTreeSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
        assert!(reg.get(BackendId::Vllm).is_some());
    }

    #[test]
    fn gguf_selection_per_hardware_profile() {
        use LlamaCppBuild::*;
        let cases: Vec<(&str, HardwareProfile, Vec<BackendId>)> = vec![
            ("laptop", profiles::rtx3070ti_laptop().unwrap(), vec![l(Cuda12), l(Cuda13), l(Sycl), l(Vulkan), l(Cpu)]),
            ("4090", profiles::rtx4090_desktop().unwrap(), vec![l(Cuda12), l(Cuda13), l(Vulkan), l(Cpu)]),
            // Driver 570 is below CUDA 13's floor; no Vulkan loader on the server.
            ("h100", profiles::h100_server().unwrap(), vec![l(Cuda12), l(Cpu)]),
            ("2x3090", profiles::dual_rtx3090_nvlink().unwrap(), vec![l(Cuda12), l(Cuda13), l(Vulkan), l(Cpu)]),
            ("7900xtx", profiles::rx7900xtx().unwrap(), vec![l(Hip), l(Vulkan), l(Cpu)]),
            ("a770", profiles::arc_a770().unwrap(), vec![l(Sycl), l(Vulkan), l(Cpu)]),
            ("cpu-only", profiles::cpu_only().unwrap(), vec![l(Cpu)]),
            ("snapdragon", profiles::snapdragon_x_elite().unwrap(), vec![l(Cpu), l(OpenClAdreno), l(Vulkan)]),
            ("155h", profiles::core_ultra_155h().unwrap(), vec![l(Cpu), l(Sycl), l(Vulkan)]),
            ("hx370", profiles::ryzen_ai_hx370().unwrap(), vec![l(Cpu), l(Hip), l(Vulkan)]),
            ("m3", profiles::apple_m3_max().unwrap(), vec![l(Metal), l(Cpu)]),
        ];
        for (name, hw, expected) in cases {
            let sel = gguf(&hw);
            assert_eq!(sel.ids(), expected, "{name}: {:?}", sel.rejected);
            assert!(sel.candidates.iter().all(|c| c.status == CandidateStatus::Supported), "{name}");
        }
    }

    #[test]
    fn blackwell_prefers_the_cuda13_build() {
        let mut hw = profiles::rtx4090_desktop().unwrap();
        if let Detected::Known(gpus) = &mut hw.gpus {
            gpus[0].arch = Detected::Known(GpuArch::Nvidia(NvidiaArch::BlackwellClient));
        }
        assert_eq!(gguf(&hw).first(), Some(l(LlamaCppBuild::Cuda13)));
    }

    #[test]
    fn rejected_backends_carry_reasons_and_unknown_hardware_still_gets_candidates() {
        let sel = gguf(&profiles::rx7900xtx().unwrap());
        let cuda = sel.rejected.iter().find(|r| r.backend == l(LlamaCppBuild::Cuda12)).unwrap();
        let RejectionCause::Hardware(reasons) = &cuda.cause else { panic!() };
        assert!(reasons[0].to_string().contains("NVIDIA"), "{}", reasons[0]);

        let sel = gguf(&HardwareProfile::unknown("no detectors"));
        assert!(!sel.candidates.is_empty());
        assert_eq!(sel.first(), Some(l(LlamaCppBuild::Cpu)));
        assert!(sel.candidates.iter().all(|c| matches!(c.status, CandidateStatus::Unverifiable(_))));
    }

    #[test]
    fn the_user_can_force_on_or_disable_any_backend() {
        let hw = profiles::rtx3070ti_laptop().unwrap();
        let policy = SelectionPolicy {
            force_on: BTreeSet::from([l(LlamaCppBuild::Hip)]),
            disabled: BTreeSet::from([l(LlamaCppBuild::Cuda12)]),
        };
        let sel = BackendRegistry::standard().select(&hw, ModelFormat::Gguf, &policy);
        assert_eq!(sel.first(), Some(l(LlamaCppBuild::Cuda13)));
        assert_eq!(sel.ids().last(), Some(&l(LlamaCppBuild::Hip)));
        assert!(sel.rejected.iter().any(|r| r.backend == l(LlamaCppBuild::Cuda12) && r.cause == RejectionCause::DisabledByUser));
        let warnings = sel.warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].starts_with("llama.cpp (ROCm/HIP) is offered by your choice"), "{}", warnings[0]);
    }

    #[test]
    fn non_gguf_formats_route_to_their_runtimes() {
        let reg = BackendRegistry::standard();
        let none = SelectionPolicy::default();
        let st = reg.select(&profiles::h100_server().unwrap(), ModelFormat::Safetensors, &none);
        assert_eq!(st.ids(), [BackendId::Vllm, BackendId::Sglang]);
        let st = reg.select(&profiles::rtx3070ti_laptop().unwrap(), ModelFormat::Safetensors, &none);
        assert!(st.candidates.is_empty());
        assert_eq!(st.rejected.len(), 2);
        let mlx = reg.select(&profiles::apple_m3_max().unwrap(), ModelFormat::Mlx, &none);
        assert_eq!(mlx.ids(), [BackendId::Mlx]);
        let ov = reg.select(&profiles::core_ultra_155h().unwrap(), ModelFormat::OpenVinoIr, &none);
        assert_eq!(ov.ids(), [BackendId::OpenVino]);
        // QNN matches the Snapdragon but has no launch recipe: never a candidate, even forced on.
        let qnn = reg.select(&profiles::snapdragon_x_elite().unwrap(), ModelFormat::QnnContext, &none);
        assert!(qnn.candidates.is_empty(), "{:?}", qnn.candidates);
        assert!(matches!(qnn.rejected[0].cause, RejectionCause::NoLaunchRecipe(d) if d.contains("QNN SDK")));
        let forced = SelectionPolicy { force_on: BTreeSet::from([BackendId::Qnn]), ..SelectionPolicy::default() };
        let qnn = reg.select(&profiles::snapdragon_x_elite().unwrap(), ModelFormat::QnnContext, &forced);
        assert!(qnn.candidates.is_empty());
        let exl = reg.select(&profiles::rtx3070ti_laptop().unwrap(), ModelFormat::Exl3, &none);
        assert_eq!(exl.ids(), [BackendId::ExLlamaV3]);
    }
}
