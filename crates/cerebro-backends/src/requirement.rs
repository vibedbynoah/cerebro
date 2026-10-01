//! Hardware and OS requirements, checked against a [`HardwareProfile`] with three outcomes.
//!
//! A requirement is **met**, **unmet** (the profile establishes that it is not satisfied), or
//! **unknown** (a fact it depends on was not detected). Unknown is never folded into either of the
//! others: "could not check" must not read as "available" (a feature that then fails) nor as
//! "unavailable" (a feature hidden from a user who has the hardware). Every unmet or unknown
//! outcome carries a [`Reason`] whose text the app shows as is: "needs an NVIDIA Hopper GPU".

use std::fmt;

use cerebro_probe::hardware::{
    AmdArch, CpuArch, Detected, GpuArch, GpuDevice, GpuKind, GpuVendor, HardwareProfile,
    IntelArch, Isa, NpuArch, NvidiaArch, OsFamily,
};

use crate::backend::BackendId;

/// One condition a backend or feature needs. Built from `'static` tables, so it is `Copy` and a
/// table row reads like the documentation it encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// The OS is one of these.
    Os(&'static [OsFamily]),
    /// Native Linux, or Windows with a general-purpose WSL 2 distribution that can start.
    LinuxOrWsl2,
    /// The machine's native CPU architecture is one of these.
    NativeArch(&'static [CpuArch]),
    /// The CPU has (and the OS enabled) this extension.
    Isa(Isa),
    /// At least `min` GPUs from `vendors` (empty = any vendor), counting only discrete ones when
    /// `discrete_only`.
    Gpus {
        /// Vendors that count; empty means every vendor.
        vendors: &'static [GpuVendor],
        /// How many are needed.
        min: usize,
        /// Count only discrete GPUs.
        discrete_only: bool,
    },
    /// An NVIDIA GPU of one of these architectures.
    NvidiaArch(&'static [NvidiaArch]),
    /// The NVIDIA driver is at least this `(major, minor)` version on each OS (the CUDA runtime
    /// floors differ: CUDA 12 needs 527.41 on Windows but 525.60 on Linux). Vacuously met without
    /// an NVIDIA GPU: it is always paired with an NVIDIA GPU requirement, which reports that absence
    /// once. On any other OS, or an unknown one, it is undetected.
    NvidiaDriverAtLeast {
        /// Floor on Windows.
        windows: (u32, u32),
        /// Floor on Linux.
        linux: (u32, u32),
    },
    /// An AMD GPU of one of these architectures.
    AmdArch(&'static [AmdArch]),
    /// An Intel GPU of one of these architectures.
    IntelArch(&'static [IntelArch]),
    /// An NPU of one of these generations.
    Npu(&'static [NpuArch]),
    /// Two or more NVIDIA GPUs with every pair joined by NVLink.
    NvlinkAllPairs,
    /// The Vulkan loader is installed.
    VulkanLoader,
    /// AMD's HIP runtime is installed.
    HipRuntime,
    /// At least one of these.
    AnyOf(&'static [Requirement]),
}

/// Why a requirement is not met, or could not be checked. `Display` is the user-facing text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// The profile establishes the requirement is not met.
    Unmet {
        /// What is needed.
        requirement: Requirement,
        /// What was found instead, when that helps (`found 1`, `found Ampere`).
        found: Option<String>,
    },
    /// A fact the requirement depends on was not detected.
    Undetected {
        /// What could not be checked.
        requirement: Requirement,
        /// Why (from the probe).
        detail: String,
    },
    /// Not about the hardware: Cerebro has no launch recipe for this backend
    /// ([`crate::backend::LaunchRecipe::Missing`]). When it is a provider's only reason, the
    /// hardware matches and only Cerebro is missing.
    NotLaunchable {
        /// The backend.
        backend: BackendId,
        /// What is missing.
        detail: &'static str,
    },
}

impl Reason {
    /// The hardware requirement this reason is about; `None` for [`Reason::NotLaunchable`], which
    /// is about Cerebro, not the machine.
    pub fn requirement(&self) -> Option<Requirement> {
        match self {
            Self::Unmet { requirement, .. } | Self::Undetected { requirement, .. } => Some(*requirement),
            Self::NotLaunchable { .. } => None,
        }
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unmet {
                requirement,
                found: Some(found),
            } => write!(f, "needs {requirement} (found {found})"),
            Self::Unmet { requirement, .. } => write!(f, "needs {requirement}"),
            Self::Undetected {
                requirement,
                detail,
            } => write!(f, "could not check for {requirement}: {detail}"),
            Self::NotLaunchable { backend, detail } => {
                write!(f, "Cerebro cannot launch {backend} yet: {detail}")
            }
        }
    }
}

/// The outcome of checking requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    /// Every requirement is met.
    Met,
    /// At least one is established as unmet (unknown ones are then omitted: the unmet one already
    /// decides).
    Unmet(Vec<Reason>),
    /// None is unmet but at least one could not be checked.
    Unknown(Vec<Reason>),
}

impl Check {
    /// Whether every requirement is met.
    pub fn is_met(&self) -> bool {
        matches!(self, Self::Met)
    }

    /// The reasons, empty when met.
    pub fn reasons(&self) -> &[Reason] {
        match self {
            Self::Met => &[],
            Self::Unmet(r) | Self::Unknown(r) => r,
        }
    }
}

/// Join display names as `A, B or C`.
fn join_or(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => (*one).to_string(),
        [init @ .., last] => format!("{} or {}", init.join(", "), last),
    }
}

/// `527.41`, or `580` for a `.0` floor.
fn version_text((major, minor): (u32, u32)) -> String {
    if minor == 0 {
        major.to_string()
    } else {
        format!("{major}.{minor:02}")
    }
}

/// `(major, minor)` of an NVIDIA driver version (`616.92`, `570.124.06`); a missing minor is 0.
fn driver_version(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = match parts.next() {
        Some(m) => m.parse().ok()?,
        None => 0,
    };
    Some((major, minor))
}

/// `a`/`an` for a following word.
fn article(word: &str) -> &'static str {
    match word.chars().next() {
        Some('A' | 'E' | 'I' | 'O' | 'U' | 'a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    }
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Os(list) => {
                let names: Vec<&str> = list.iter().map(|o| o.name()).collect();
                f.write_str(&join_or(&names))
            }
            Self::LinuxOrWsl2 => f.write_str("Linux, or Windows with a WSL 2 Linux distribution"),
            Self::NativeArch(list) => {
                let names: Vec<&str> = list.iter().map(|a| a.name()).collect();
                let text = join_or(&names);
                write!(f, "{} {text} CPU", article(&text))
            }
            Self::Isa(isa) => write!(f, "a CPU with {isa}"),
            Self::Gpus {
                vendors,
                min,
                discrete_only,
            } => {
                let vendor = if vendors.is_empty() {
                    String::new()
                } else {
                    let names: Vec<&str> = vendors.iter().map(|v| v.name()).collect();
                    format!("{} ", join_or(&names))
                };
                let discrete = if *discrete_only { "discrete " } else { "" };
                if *min <= 1 {
                    let words = format!("{discrete}{vendor}GPU");
                    write!(f, "{} {words}", article(&words))
                } else {
                    write!(f, "{min} or more {discrete}{vendor}GPUs")
                }
            }
            Self::NvidiaArch(list) => {
                let names: Vec<&str> = list.iter().map(|a| a.name()).collect();
                write!(f, "an NVIDIA {} GPU", join_or(&names))
            }
            Self::NvidiaDriverAtLeast { windows, linux } if windows == linux => {
                write!(f, "NVIDIA driver {} or newer", version_text(*windows))
            }
            Self::NvidiaDriverAtLeast { windows, linux } => write!(
                f,
                "NVIDIA driver {} or newer on Windows ({} on Linux)",
                version_text(*windows),
                version_text(*linux)
            ),
            Self::AmdArch(list) => {
                let names: Vec<&str> = list.iter().map(|a| a.name()).collect();
                write!(f, "an AMD {} GPU", join_or(&names))
            }
            Self::IntelArch(list) => {
                let names: Vec<&str> = list.iter().map(|a| a.name()).collect();
                write!(f, "an Intel {} GPU", join_or(&names))
            }
            Self::Npu(list) => {
                let names: Vec<&str> = list.iter().map(|a| a.name()).collect();
                let text = join_or(&names);
                write!(f, "{} {text} NPU", article(&text))
            }
            Self::NvlinkAllPairs => f.write_str("NVIDIA GPUs joined by NVLink"),
            Self::VulkanLoader => f.write_str("the Vulkan runtime (installed with the GPU driver)"),
            Self::HipRuntime => {
                f.write_str("AMD's HIP runtime (installed with the AMD Software: Adrenalin driver)")
            }
            Self::AnyOf(list) => {
                let parts: Vec<String> = list.iter().map(ToString::to_string).collect();
                write!(f, "one of: {}", parts.join("; "))
            }
        }
    }
}

/// One requirement's outcome.
enum Single {
    Met,
    Unmet(Reason),
    Unknown(Reason),
}

fn unmet(requirement: Requirement, found: Option<String>) -> Single {
    Single::Unmet(Reason::Unmet { requirement, found })
}

fn undetected(requirement: Requirement, detail: &str) -> Single {
    Single::Unknown(Reason::Undetected {
        requirement,
        detail: detail.to_string(),
    })
}

/// Check one fact that may be unknown.
fn on<T>(requirement: Requirement, fact: &Detected<T>, test: impl FnOnce(&T) -> Single) -> Single {
    match fact {
        Detected::Known(v) => test(v),
        Detected::Unknown { reason } => undetected(requirement, reason),
    }
}

/// Architecture match over the GPUs: met if one GPU's known architecture matches, unknown if none
/// matches but some GPU of `vendor` has an unknown architecture, unmet otherwise.
fn gpu_arch(
    requirement: Requirement,
    hw: &HardwareProfile,
    vendor: GpuVendor,
    matches: impl Fn(&GpuArch) -> bool,
    describe: impl Fn(&GpuArch) -> Option<String>,
) -> Single {
    on(requirement, &hw.gpus, |gpus| {
        let of_vendor: Vec<&GpuDevice> = gpus.iter().filter(|g| g.vendor == vendor).collect();
        if of_vendor
            .iter()
            .any(|g| g.arch.known().is_some_and(&matches))
        {
            return Single::Met;
        }
        if let Some(g) = of_vendor.iter().find(|g| g.arch.known().is_none()) {
            return undetected(requirement, g.arch.reason().unwrap_or("architecture unknown"));
        }
        let found: Vec<String> = of_vendor
            .iter()
            .filter_map(|g| g.arch.known().and_then(&describe))
            .collect();
        let found = if found.is_empty() {
            format!("no {} GPU", vendor.name())
        } else {
            found.join(", ")
        };
        unmet(requirement, Some(found))
    })
}

fn check_one(requirement: Requirement, hw: &HardwareProfile) -> Single {
    match requirement {
        Requirement::Os(list) => on(requirement, &hw.os, |os| {
            if list.contains(&os.family) {
                Single::Met
            } else {
                unmet(requirement, Some(os.family.name().to_string()))
            }
        }),
        Requirement::LinuxOrWsl2 => on(requirement, &hw.os, |os| match os.family {
            OsFamily::Linux => Single::Met,
            OsFamily::MacOs => unmet(requirement, Some("macOS".into())),
            OsFamily::Windows => on(requirement, &hw.runtimes.wsl, |wsl| {
                if wsl.usable_wsl2().next().is_none() {
                    let found = if wsl.installed {
                        "WSL without a Linux distribution"
                    } else {
                        "WSL not installed"
                    };
                    return unmet(requirement, Some(found.into()));
                }
                // A registered distribution does not prove WSL 2 can start (no hypervisor).
                on(requirement, &wsl.wsl2_can_start, |can| {
                    if *can {
                        Single::Met
                    } else {
                        unmet(
                            requirement,
                            Some("a WSL 2 distribution that cannot start: no hypervisor is running".into()),
                        )
                    }
                })
            }),
        }),
        Requirement::NativeArch(list) => on(requirement, &hw.cpu, |cpu| {
            if list.contains(&cpu.arch) {
                Single::Met
            } else {
                unmet(requirement, Some(cpu.arch.name().to_string()))
            }
        }),
        Requirement::Isa(isa) => on(requirement, &hw.cpu, |cpu| {
            on(requirement, &cpu.isa, |set| {
                if set.has(isa) {
                    Single::Met
                } else {
                    unmet(requirement, None)
                }
            })
        }),
        Requirement::Gpus {
            vendors,
            min,
            discrete_only,
        } => on(requirement, &hw.gpus, |gpus| {
            let counted = gpus
                .iter()
                .filter(|g| vendors.is_empty() || vendors.contains(&g.vendor));
            let (mut sure, mut maybe) = (0usize, 0usize);
            for g in counted {
                match (discrete_only, g.kind) {
                    (false, _) | (true, GpuKind::Discrete) => sure += 1,
                    (true, GpuKind::Unknown) => maybe += 1,
                    (true, GpuKind::Integrated) => {}
                }
            }
            if sure >= min {
                Single::Met
            } else if sure + maybe >= min {
                undetected(requirement, "whether some GPUs are discrete is unknown")
            } else {
                unmet(requirement, Some(sure.to_string()))
            }
        }),
        Requirement::NvidiaArch(list) => gpu_arch(
            requirement,
            hw,
            GpuVendor::Nvidia,
            |a| matches!(a, GpuArch::Nvidia(n) if list.contains(n)),
            |a| match a {
                GpuArch::Nvidia(n) => Some(n.name().to_string()),
                _ => None,
            },
        ),
        Requirement::AmdArch(list) => gpu_arch(
            requirement,
            hw,
            GpuVendor::Amd,
            |a| matches!(a, GpuArch::Amd { family, .. } if list.contains(family)),
            |a| match a {
                GpuArch::Amd { family, .. } => Some(family.name().to_string()),
                _ => None,
            },
        ),
        Requirement::IntelArch(list) => gpu_arch(
            requirement,
            hw,
            GpuVendor::Intel,
            |a| matches!(a, GpuArch::Intel(i) if list.contains(i)),
            |a| match a {
                GpuArch::Intel(i) => Some(i.name().to_string()),
                _ => None,
            },
        ),
        Requirement::NvidiaDriverAtLeast { windows, linux } => on(requirement, &hw.gpus, |gpus| {
            let nvidia: Vec<&GpuDevice> =
                gpus.iter().filter(|g| g.vendor == GpuVendor::Nvidia).collect();
            if nvidia.is_empty() {
                return Single::Met;
            }
            on(requirement, &hw.os, |os| {
                let min = match os.family {
                    OsFamily::Windows => windows,
                    OsFamily::Linux => linux,
                    OsFamily::MacOs => {
                        return undetected(requirement, "no NVIDIA driver floor applies on macOS")
                    }
                };
                let version = nvidia
                    .iter()
                    .filter_map(|g| g.driver_version.as_deref())
                    .find_map(|v| driver_version(v).map(|parsed| (parsed, v)));
                match version {
                    Some((found, _)) if found >= min => Single::Met,
                    Some((_, text)) => unmet(requirement, Some(text.to_string())),
                    None => undetected(requirement, "NVIDIA driver version not reported"),
                }
            })
        }),
        Requirement::Npu(list) => on(requirement, &hw.npus, |npus| {
            if npus
                .iter()
                .any(|n| n.arch.known().is_some_and(|a| list.contains(a)))
            {
                return Single::Met;
            }
            if let Some(n) = npus.iter().find(|n| n.arch.known().is_none()) {
                return undetected(requirement, n.arch.reason().unwrap_or("NPU generation unknown"));
            }
            let found: Vec<&str> = npus
                .iter()
                .filter_map(|n| n.arch.known().map(|a| a.name()))
                .collect();
            unmet(
                requirement,
                Some(if found.is_empty() {
                    "no NPU".to_string()
                } else {
                    found.join(", ")
                }),
            )
        }),
        Requirement::NvlinkAllPairs => {
            let nvidia = hw.gpus_of(GpuVendor::Nvidia).count();
            if hw.gpus.known().is_some() && nvidia < 2 {
                return unmet(requirement, Some(format!("{nvidia} NVIDIA GPU")));
            }
            on(requirement, &hw.topology, |topo| {
                if topo.all_pairs_nvlink() {
                    Single::Met
                } else {
                    unmet(requirement, Some("PCIe only".into()))
                }
            })
        }
        Requirement::VulkanLoader => on(requirement, &hw.runtimes.vulkan_loader, |present| {
            if *present {
                Single::Met
            } else {
                unmet(requirement, None)
            }
        }),
        Requirement::HipRuntime => on(requirement, &hw.runtimes.hip_runtime, |present| {
            if *present {
                Single::Met
            } else {
                unmet(requirement, None)
            }
        }),
        Requirement::AnyOf(list) => {
            let mut unknown = None;
            for r in list {
                match check_one(*r, hw) {
                    Single::Met => return Single::Met,
                    Single::Unknown(reason) => unknown = unknown.or(Some(reason)),
                    Single::Unmet(_) => {}
                }
            }
            match unknown {
                Some(reason) => Single::Unknown(reason),
                None => unmet(requirement, None),
            }
        }
    }
}

/// Check every requirement (all must hold).
pub fn check_all(requirements: &[Requirement], hw: &HardwareProfile) -> Check {
    let mut unmet = Vec::new();
    let mut unknown = Vec::new();
    for r in requirements {
        match check_one(*r, hw) {
            Single::Met => {}
            Single::Unmet(reason) => unmet.push(reason),
            Single::Unknown(reason) => unknown.push(reason),
        }
    }
    if !unmet.is_empty() {
        Check::Unmet(unmet)
    } else if !unknown.is_empty() {
        Check::Unknown(unknown)
    } else {
        Check::Met
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_probe::hardware::profiles;

    const HOPPER: Requirement = Requirement::NvidiaArch(&[NvidiaArch::Hopper]);

    #[test]
    fn reasons_read_as_user_facing_text() {
        assert_eq!(
            Reason::Unmet { requirement: HOPPER, found: Some("Ada Lovelace".into()) }.to_string(),
            "needs an NVIDIA Hopper GPU (found Ada Lovelace)"
        );
        let two = Requirement::Gpus { vendors: &[GpuVendor::Nvidia], min: 2, discrete_only: true };
        assert_eq!(two.to_string(), "2 or more discrete NVIDIA GPUs");
        let one = Requirement::Gpus { vendors: &[], min: 1, discrete_only: false };
        assert_eq!(one.to_string(), "a GPU");
        let amd = Requirement::Gpus { vendors: &[GpuVendor::Amd], min: 1, discrete_only: false };
        assert_eq!(amd.to_string(), "an AMD GPU");
        assert_eq!(Requirement::NativeArch(&[CpuArch::Aarch64]).to_string(), "an Arm64 CPU");
        assert_eq!(
            Requirement::Npu(&[NpuArch::IntelNpu3, NpuArch::IntelNpu4]).to_string(),
            "an Intel AI Boost 3720 or Intel AI Boost 4 NPU"
        );
        assert_eq!(
            Requirement::Os(&[OsFamily::Windows, OsFamily::Linux, OsFamily::MacOs]).to_string(),
            "Windows, Linux or macOS"
        );
    }

    #[test]
    fn architecture_gates_distinguish_unmet_from_unknown() {
        let h100 = profiles::h100_server().unwrap();
        assert_eq!(check_all(&[HOPPER], &h100), Check::Met);

        let laptop = profiles::rtx3070ti_laptop().unwrap();
        let Check::Unmet(reasons) = check_all(&[HOPPER], &laptop) else { panic!() };
        assert_eq!(reasons[0].to_string(), "needs an NVIDIA Hopper GPU (found Ampere)");

        let amd = profiles::rx7900xtx().unwrap();
        let Check::Unmet(reasons) = check_all(&[HOPPER], &amd) else { panic!() };
        assert_eq!(reasons[0].to_string(), "needs an NVIDIA Hopper GPU (found no NVIDIA GPU)");

        // An NVIDIA GPU whose architecture is unknown cannot be ruled in or out.
        let mut unknown = laptop.clone();
        if let Detected::Known(gpus) = &mut unknown.gpus {
            gpus[0].arch = Detected::unknown("compute capability 13.0 is newer than Cerebro's table");
        }
        assert!(matches!(check_all(&[HOPPER], &unknown), Check::Unknown(_)));

        let nothing = HardwareProfile::unknown("no detectors");
        assert!(matches!(check_all(&[HOPPER], &nothing), Check::Unknown(_)));
    }

    #[test]
    fn unmet_beats_unknown_and_all_reasons_are_kept() {
        let mut p = profiles::rtx3070ti_laptop().unwrap();
        p.runtimes.vulkan_loader = Detected::unknown("not probed");
        let check = check_all(&[HOPPER, Requirement::VulkanLoader, Requirement::LinuxOrWsl2], &p);
        let Check::Unmet(reasons) = check else { panic!("{check:?}") };
        assert_eq!(reasons.len(), 2, "{reasons:?}");
        assert_eq!(
            reasons[1].to_string(),
            "needs Linux, or Windows with a WSL 2 Linux distribution (found WSL without a Linux distribution)"
        );
    }

    #[test]
    fn wsl_counts_only_with_a_general_purpose_wsl2_distribution() {
        let req = [Requirement::LinuxOrWsl2];
        assert!(check_all(&req, &profiles::rtx4090_desktop().unwrap()).is_met());
        assert!(check_all(&req, &profiles::h100_server().unwrap()).is_met());
        // Docker Desktop's distributions only: not usable.
        assert!(!check_all(&req, &profiles::rtx3070ti_laptop().unwrap()).is_met());
        assert!(!check_all(&req, &profiles::apple_m3_max().unwrap()).is_met());
    }

    /// A registered Ubuntu is not enough: with no hypervisor WSL 2 cannot start (unmet), and when
    /// that was not established the answer is "could not check", never yes.
    #[test]
    fn a_registered_distribution_counts_only_if_wsl2_can_start() {
        let req = [Requirement::LinuxOrWsl2];
        let with_start = |fact: Detected<bool>| {
            let mut p = profiles::rtx4090_desktop().unwrap();
            if let Detected::Known(wsl) = &mut p.runtimes.wsl {
                wsl.wsl2_can_start = fact;
            }
            p
        };
        let Check::Unmet(r) = check_all(&req, &with_start(Detected::Known(false))) else { panic!() };
        assert!(r[0].to_string().contains("no hypervisor"), "{}", r[0]);
        assert!(matches!(check_all(&req, &with_start(Detected::unknown("not checked"))), Check::Unknown(_)));
        // Not installed at all reads as such, not as "without a distribution".
        let mut none = profiles::rx7900xtx().unwrap();
        none.runtimes.wsl = Detected::Known(cerebro_probe::hardware::WslInfo::default());
        let Check::Unmet(r) = check_all(&req, &none) else { panic!() };
        assert!(r[0].to_string().ends_with("(found WSL not installed)"), "{}", r[0]);
    }

    #[test]
    fn not_launchable_is_about_cerebro_not_the_hardware() {
        let r = Reason::NotLaunchable { backend: BackendId::Qnn, detail: "no server" };
        assert_eq!(r.to_string(), "Cerebro cannot launch Qualcomm QNN yet: no server");
        assert_eq!(r.requirement(), None);
        assert_eq!(
            Reason::Unmet { requirement: HOPPER, found: None }.requirement(),
            Some(HOPPER)
        );
    }

    #[test]
    fn gpu_counts_respect_vendor_and_discreteness() {
        let two_nv = [Requirement::Gpus { vendors: &[GpuVendor::Nvidia], min: 2, discrete_only: true }];
        assert!(check_all(&two_nv, &profiles::dual_rtx3090_nvlink().unwrap()).is_met());
        let Check::Unmet(r) = check_all(&two_nv, &profiles::rtx4090_desktop().unwrap()) else { panic!() };
        assert!(r[0].to_string().ends_with("(found 1)"), "{}", r[0]);

        // The laptop has two GPUs, but one is an iGPU.
        let two_any = [Requirement::Gpus { vendors: &[], min: 2, discrete_only: true }];
        assert!(!check_all(&two_any, &profiles::rtx3070ti_laptop().unwrap()).is_met());
        let two_any_incl = [Requirement::Gpus { vendors: &[], min: 2, discrete_only: false }];
        assert!(check_all(&two_any_incl, &profiles::rtx3070ti_laptop().unwrap()).is_met());
        // The 4090 box's iGPU has unknown kind: whether it makes two discrete GPUs is unknown.
        assert!(matches!(
            check_all(&two_any, &profiles::rtx4090_desktop().unwrap()),
            Check::Unknown(_)
        ));
    }

    #[test]
    fn nvlink_and_driver_gates() {
        assert!(check_all(&[Requirement::NvlinkAllPairs], &profiles::dual_rtx3090_nvlink().unwrap()).is_met());
        let Check::Unmet(r) = check_all(&[Requirement::NvlinkAllPairs], &profiles::rtx4090_desktop().unwrap()) else { panic!() };
        assert_eq!(r[0].to_string(), "needs NVIDIA GPUs joined by NVLink (found 1 NVIDIA GPU)");

        let cuda13 = [Requirement::NvidiaDriverAtLeast { windows: (580, 0), linux: (580, 0) }];
        assert!(check_all(&cuda13, &profiles::rtx3070ti_laptop().unwrap()).is_met());
        let Check::Unmet(r) = check_all(&cuda13, &profiles::h100_server().unwrap()) else { panic!() };
        assert_eq!(r[0].to_string(), "needs NVIDIA driver 580 or newer (found 570.124.06)");
        // No NVIDIA GPU: the paired GPU requirement reports that, not this one.
        assert!(check_all(&cuda13, &profiles::rx7900xtx().unwrap()).is_met());
    }

    /// CUDA 12's floor is 527.41 on Windows but 525.60 on Linux: a Linux 525.60.13 driver meets
    /// it, the same driver number on Windows does not.
    #[test]
    fn driver_floors_depend_on_the_os() {
        let cuda12 = [Requirement::NvidiaDriverAtLeast { windows: (527, 41), linux: (525, 60) }];
        let with_driver = |mut p: HardwareProfile, v: &str| {
            if let Detected::Known(gpus) = &mut p.gpus {
                gpus[0].driver_version = Some(v.into());
            }
            p
        };
        let linux = with_driver(profiles::h100_server().unwrap(), "525.60.13");
        assert!(check_all(&cuda12, &linux).is_met());
        let linux_old = with_driver(profiles::h100_server().unwrap(), "525.53");
        assert!(!check_all(&cuda12, &linux_old).is_met());
        let windows = with_driver(profiles::rtx4090_desktop().unwrap(), "526.98");
        let Check::Unmet(r) = check_all(&cuda12, &windows) else { panic!() };
        assert_eq!(
            r[0].to_string(),
            "needs NVIDIA driver 527.41 or newer on Windows (525.60 on Linux) (found 526.98)"
        );
        assert!(check_all(&cuda12, &with_driver(profiles::rtx4090_desktop().unwrap(), "527.41")).is_met());
        let mut no_os = profiles::rtx4090_desktop().unwrap();
        no_os.os = Detected::unknown("not probed");
        assert!(matches!(check_all(&cuda12, &no_os), Check::Unknown(_)));
    }

    #[test]
    fn any_of_is_met_by_one_and_unknown_only_without_a_met_one() {
        const FP8: Requirement = Requirement::AnyOf(&[
            Requirement::NvidiaArch(&[NvidiaArch::Ada, NvidiaArch::Hopper]),
            Requirement::AmdArch(&[AmdArch::Cdna3]),
        ]);
        assert!(check_all(&[FP8], &profiles::rtx4090_desktop().unwrap()).is_met());
        assert!(!check_all(&[FP8], &profiles::rx7900xtx().unwrap()).is_met());
        assert!(matches!(
            check_all(&[FP8], &HardwareProfile::unknown("x")),
            Check::Unknown(_)
        ));
    }

    #[test]
    fn isa_and_npu_gates() {
        let amx = [Requirement::Isa(Isa::AmxInt8)];
        assert!(check_all(&amx, &profiles::h100_server().unwrap()).is_met());
        assert!(!check_all(&amx, &profiles::rtx4090_desktop().unwrap()).is_met());
        let xdna2 = [Requirement::Npu(&[NpuArch::AmdXdna2])];
        assert!(check_all(&xdna2, &profiles::ryzen_ai_hx370().unwrap()).is_met());
        let Check::Unmet(r) = check_all(&xdna2, &profiles::core_ultra_155h().unwrap()) else { panic!() };
        assert_eq!(r[0].to_string(), "needs an AMD XDNA 2 NPU (found Intel AI Boost 3720)");
    }
}
