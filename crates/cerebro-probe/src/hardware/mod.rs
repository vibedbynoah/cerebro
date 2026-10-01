//! The full hardware picture: OS, CPU and its instruction sets, memory, every GPU and NPU, multi-GPU
//! topology, and the runtime facts that gate inference backends.
//!
//! This is what the capability matrix in `cerebro-backends` reads to decide which features a user
//! can switch on. Cerebro ships to machines it has never seen, so two rules hold throughout:
//!
//! * **Every detector fails soft.** A missing vendor tool, an unreadable registry key or an
//!   unfamiliar device yields [`Detected::Unknown`] with the reason, never an error that hides the
//!   rest of the picture and never a panic. [`probe`] always returns a profile.
//! * **Nothing is guessed.** A GPU whose PCI ID is not in a table keeps its vendor and name, and
//!   its architecture is `Unknown`. A feature gated on that architecture then reads "could not
//!   determine" rather than a confident yes or no.
//!
//! The physical-inventory caveat in [`crate::gpu`] applies here too: what is present is not what a
//! particular backend build can address. That second set comes from the build's own
//! `--list-devices` ([`crate::backend`]).

pub mod accel;
#[cfg(windows)]
mod devnode;
pub mod isa;
pub mod linux;
pub mod macos;
pub mod nvidia;
#[cfg(any(test, feature = "test-support"))]
pub mod profiles;
#[cfg(windows)]
mod registry;
pub mod system;

use std::collections::BTreeSet;

pub use accel::{
    AdapterRecord, AmdArch, GpuArch, GpuDevice, GpuKind, GpuVendor, IntelArch, NpuArch, NpuDevice,
    NpuVendor, NvidiaArch, PciId,
};
pub use isa::{Isa, IsaSet};
pub use nvidia::{ComputeCapability, GpuLink, GpuTopology, LinkKind};
pub use system::{
    CpuArch, CpuInfo, MemoryInfo, OsFamily, OsInfo, RuntimeFacts, WslDistro, WslInfo,
};

/// A fact that was either established or could not be.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Detected<T> {
    /// Established.
    Known(T),
    /// Not established, with why (shown to the user and recorded).
    Unknown {
        /// Why the fact could not be established.
        reason: String,
    },
}

impl<T> Detected<T> {
    /// An unknown fact with its reason.
    pub fn unknown(reason: impl Into<String>) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }

    /// The value, when known.
    pub fn known(&self) -> Option<&T> {
        match self {
            Self::Known(v) => Some(v),
            Self::Unknown { .. } => None,
        }
    }

    /// The reason, when unknown.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Known(_) => None,
            Self::Unknown { reason } => Some(reason),
        }
    }
}

/// Everything Cerebro knows about the machine's hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HardwareProfile {
    /// Operating system.
    pub os: Detected<OsInfo>,
    /// CPU.
    pub cpu: Detected<CpuInfo>,
    /// System memory.
    pub memory: Detected<MemoryInfo>,
    /// Every GPU, NVIDIA first in `nvidia-smi` order, then the rest in registry order. `Known` and
    /// empty means the probe looked and found none.
    pub gpus: Detected<Vec<GpuDevice>>,
    /// NVIDIA GPU-to-GPU topology. `Known` and empty with fewer than two NVIDIA GPUs.
    pub topology: Detected<GpuTopology>,
    /// Every NPU.
    pub npus: Detected<Vec<NpuDevice>>,
    /// Runtime facts (WSL, Vulkan loader, HIP runtime).
    pub runtimes: RuntimeFacts,
}

impl HardwareProfile {
    /// A profile in which nothing is known: what a platform without detectors reports, and the
    /// baseline for checking that every consumer handles `Unknown`.
    pub fn unknown(reason: &str) -> Self {
        Self {
            os: Detected::unknown(reason),
            cpu: Detected::unknown(reason),
            memory: Detected::unknown(reason),
            gpus: Detected::unknown(reason),
            topology: Detected::unknown(reason),
            npus: Detected::unknown(reason),
            runtimes: RuntimeFacts {
                wsl: Detected::unknown(reason),
                vulkan_loader: Detected::unknown(reason),
                hip_runtime: Detected::unknown(reason),
            },
        }
    }

    /// GPUs from one vendor (empty when the GPU list is unknown).
    pub fn gpus_of(&self, vendor: GpuVendor) -> impl Iterator<Item = &GpuDevice> {
        self.gpus
            .known()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter(move |g| g.vendor == vendor)
    }
}

/// Assemble a GPU list from the two sources: `nvidia-smi` (authoritative for NVIDIA, with compute
/// capability) and the display-class registry entries (everything else). When `nvidia` is `Some`,
/// registry NVIDIA entries are dropped as duplicates; when it is `None` (no tool, or it failed),
/// they are kept with an unknown architecture so the GPU is at least listed.
pub fn merge_gpus(nvidia: Option<Vec<GpuDevice>>, registry: &[AdapterRecord]) -> Vec<GpuDevice> {
    let have_smi = nvidia.is_some();
    let mut gpus = nvidia.unwrap_or_default();
    gpus.extend(
        registry
            .iter()
            .filter_map(accel::classify_display_adapter)
            .filter(|g| !(have_smi && g.vendor == GpuVendor::Nvidia)),
    );
    gpus
}

/// Keep the class-key records whose `NNNN` instance belongs to a device present now. Records are
/// `(instance name, record)` pairs as read from `Class\{guid}\NNNN`; a key whose device was
/// removed, disconnected or disabled in firmware is dropped. Pure, so the rule is tested on
/// recorded data.
pub fn keep_present(records: Vec<(String, AdapterRecord)>, present: &BTreeSet<String>) -> Vec<AdapterRecord> {
    records
        .into_iter()
        .filter(|(key, _)| present.contains(key))
        .map(|(_, record)| record)
        .collect()
}

/// The topology when `nvidia-smi` is not installed. With fewer than two NVIDIA GPUs among the
/// display records there are no links to relate (`Known`, empty). With two or more, the GPUs exist
/// but their links were never examined (an older non-DCH driver keeps `nvidia-smi` outside the
/// system directory), so the topology is `Unknown`, never "PCIe only". Unreadable display records
/// leave it `Unknown`.
pub fn topology_without_smi(display: Option<&[AdapterRecord]>) -> Detected<GpuTopology> {
    let Some(records) = display else {
        return Detected::unknown("display adapters unreadable and nvidia-smi not found");
    };
    let nvidia = records
        .iter()
        .filter_map(accel::classify_display_adapter)
        .filter(|g| g.vendor == GpuVendor::Nvidia)
        .count();
    if nvidia >= 2 {
        Detected::unknown(format!(
            "nvidia-smi not found at its fixed path; links between the {nvidia} NVIDIA GPUs are \
             unknown"
        ))
    } else {
        Detected::Known(GpuTopology::default())
    }
}

/// NVIDIA devices from `nvidia-smi`: `Ok(None)` when the tool is not installed at its fixed path
/// (no NVIDIA driver, or one that keeps it elsewhere), `Err` when it is installed but its answer
/// could not be used.
#[cfg(any(windows, target_os = "linux"))]
fn nvidia_devices() -> Result<Option<Vec<GpuDevice>>, String> {
    let Some(tool) = crate::gpu::locate_nvidia_smi() else {
        return Ok(None);
    };
    let out = crate::gpu::run_tool(&tool, &[nvidia::DEVICE_QUERY, "--format=csv,noheader,nounits"])
        .map_err(|e| e.to_string())?;
    nvidia::parse_device_query(&out)
        .map(Some)
        .map_err(|e| e.to_string())
}

/// Topology among the NVIDIA devices: empty for fewer than two, `topo -m` otherwise, and
/// [`topology_without_smi`] when the tool is missing.
#[cfg(any(windows, target_os = "linux"))]
fn nvidia_topology(
    nvidia: &Result<Option<Vec<GpuDevice>>, String>,
    display: Option<&[AdapterRecord]>,
) -> Detected<GpuTopology> {
    let devices = match nvidia {
        Ok(Some(d)) => d,
        Ok(None) => return topology_without_smi(display),
        Err(reason) => return Detected::unknown(reason.clone()),
    };
    if devices.len() < 2 {
        return Detected::Known(GpuTopology::default());
    }
    let Some(tool) = crate::gpu::locate_nvidia_smi() else {
        return Detected::unknown("nvidia-smi disappeared between queries");
    };
    match crate::gpu::run_tool(&tool, &["topo", "-m"]) {
        Ok(out) => match nvidia::parse_topology(&out) {
            Ok(t) => Detected::Known(t),
            Err(e) => Detected::unknown(e.to_string()),
        },
        Err(e) => Detected::unknown(e.to_string()),
    }
}

/// Probe the machine. Never fails: every part that cannot be established is `Unknown`.
///
/// Runs `nvidia-smi` (when present in the system directory) once for the device list and, with two
/// or more NVIDIA GPUs, once more for the topology; each is bounded by
/// [`crate::gpu::TOOL_TIMEOUT`], windowless, and killed on timeout. Everything else is registry
/// reads, configuration-manager queries (which class instances are present), CPUID and a few Win32
/// calls; nothing else is launched and nothing is written.
#[cfg(windows)]
pub fn probe() -> HardwareProfile {
    let os = system::live::os();
    let cpu = live_cpu(system::live::native_arch());
    let memory = system::live::memory();
    let display = live::adapters(live::DISPLAY_CLASS);
    let nvidia = nvidia_devices();
    let topology = nvidia_topology(&nvidia, display.as_deref().ok());
    let gpus = match display {
        // A failed nvidia-smi passes `None`, so registry NVIDIA entries are kept, unclassified.
        Ok(records) => Detected::Known(merge_gpus(nvidia.ok().flatten(), &records)),
        // Without the registry the non-NVIDIA GPUs are not established, so neither is the list.
        Err(reason) => Detected::unknown(reason),
    };
    let npus = match live::adapters(live::COMPUTE_ACCELERATOR_CLASS) {
        Ok(records) => Detected::Known(
            records
                .iter()
                .map(accel::classify_compute_accelerator)
                .collect(),
        ),
        Err(reason) => Detected::unknown(reason),
    };
    HardwareProfile {
        os,
        cpu,
        memory,
        gpus,
        topology,
        npus,
        runtimes: RuntimeFacts {
            wsl: system::live::wsl(),
            vulkan_loader: system::live::vulkan_loader(),
            hip_runtime: system::live::hip_runtime(),
        },
    }
}

/// Probe the machine. Linux: OS and memory from `/etc/os-release` and `/proc/meminfo`, the CPU
/// from CPUID, NVIDIA GPUs from `/usr/bin/nvidia-smi`, every other GPU from `/sys/class/drm`, NPUs
/// from `/sys/class/accel`, and the Vulkan loader and HIP runtime from their standard install
/// paths ([`linux::assemble`], tested on recorded directory trees).
#[cfg(target_os = "linux")]
pub fn probe() -> HardwareProfile {
    let mut profile = linux::assemble(std::path::Path::new("/"), nvidia_devices(), nvidia_topology);
    profile.cpu = live_cpu(CpuArch::of_this_build());
    profile
}

/// Probe the machine. macOS: OS, CPU, memory and the Apple GPU from `sysctl`
/// ([`macos::assemble`], tested on recorded values); each key is one windowless, bounded
/// `/usr/sbin/sysctl -n` query.
#[cfg(target_os = "macos")]
pub fn probe() -> HardwareProfile {
    let sysctl = |key: &str| -> Option<String> {
        crate::gpu::run_tool(std::path::Path::new("/usr/sbin/sysctl"), &["-n", key])
            .ok()
            .map(|out| out.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    macos::assemble(sysctl, CpuArch::of_this_build(), isa::detect)
}

/// Probe the machine: no detectors exist for this platform, so everything is `Unknown`.
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub fn probe() -> HardwareProfile {
    HardwareProfile::unknown("hardware detection is not implemented on this platform")
}

/// The CPU part of a live probe, given the machine's native architecture.
#[cfg(any(windows, target_os = "linux"))]
fn live_cpu(native: Option<CpuArch>) -> Detected<CpuInfo> {
    let Some(arch) = native else {
        return Detected::unknown("native CPU architecture not established");
    };
    let isa = if Some(arch) != CpuArch::of_this_build() {
        Detected::unknown(format!(
            "this {} build runs under emulation on {}; CPUID describes the emulator",
            CpuArch::of_this_build().map_or("unknown", CpuArch::name),
            arch.name()
        ))
    } else {
        let set = isa::detect();
        if set.iter().next().is_none() {
            Detected::unknown("no instruction-set detector for this architecture")
        } else {
            Detected::Known(set)
        }
    };
    let topology = crate::cpu::probe().ok();
    Detected::Known(CpuInfo {
        arch,
        brand: crate::cpu::brand_string(),
        isa,
        physical_cores: topology.as_ref().map(crate::cpu::CpuTopology::physical_cores),
        logical_processors: topology.as_ref().map(crate::cpu::CpuTopology::logical_processors),
    })
}

#[cfg(windows)]
mod live {
    use super::registry::{Key, HKEY_LOCAL_MACHINE};
    use super::*;

    /// Display adapters.
    pub(super) const DISPLAY_CLASS: &str = "{4d36e968-e325-11ce-bfc1-08002be10318}";
    /// Compute accelerators (where Windows files NPUs).
    pub(super) const COMPUTE_ACCELERATOR_CLASS: &str = "{f01a9d53-3ff6-48d2-9f97-c8a7004be10c}";

    /// Every `NNNN` entry under a device class whose device is present now (see [`keep_present`]).
    /// A class key that does not exist (no device of that class was ever installed) is an empty
    /// list. If the configuration manager cannot say which devices are present, the list is an
    /// error (reported as `Unknown`), never the unfiltered class keys, which may name hardware
    /// that has left the machine.
    pub(super) fn adapters(class: &str) -> Result<Vec<AdapterRecord>, String> {
        let path = format!(r"SYSTEM\CurrentControlSet\Control\Class\{class}");
        let key = match Key::open(HKEY_LOCAL_MACHINE, &path) {
            Ok(k) => k,
            Err(2) => return Ok(Vec::new()),
            Err(code) => return Err(format!("device class {class} unreadable ({code})")),
        };
        let present = super::devnode::present_driver_keys(class)?;
        let names = key
            .subkey_names()
            .map_err(|code| format!("device class {class} unreadable ({code})"))?;
        let mut records = Vec::new();
        for name in names {
            // Instance keys are four decimal digits; `Configuration`/`Properties` are not devices
            // (and `Properties` denies read to non-administrators). An absent device's key is not
            // even opened.
            if name.len() != 4 || !name.bytes().all(|b| b.is_ascii_digit()) || !present.contains(&name) {
                continue;
            }
            let Ok(sub) = key.subkey(&name) else { continue };
            let text = |v: &str| sub.string(v).ok().flatten();
            let Some(hardware_id) = text("MatchingDeviceId") else {
                continue;
            };
            let memory_bytes = sub
                .integer("HardwareInformation.qwMemorySize")
                .ok()
                .flatten()
                .or_else(|| sub.integer("HardwareInformation.MemorySize").ok().flatten());
            records.push((
                name,
                AdapterRecord {
                    description: text("DriverDesc").unwrap_or_default(),
                    hardware_id,
                    provider: text("ProviderName"),
                    driver_version: text("DriverVersion"),
                    memory_bytes,
                },
            ));
        }
        Ok(keep_present(records, &present))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The live device-class read sees exactly the present display adapters: every record it
    /// returns is classifiable or a software adapter, and the class read does not fail.
    #[cfg(windows)]
    #[test]
    fn live_display_adapters_are_read_through_the_present_filter() {
        let records = live::adapters(live::DISPLAY_CLASS).unwrap();
        assert!(!records.is_empty());
        assert!(live::adapters(live::COMPUTE_ACCELERATOR_CLASS).is_ok());
    }

    #[test]
    fn merge_prefers_nvidia_smi_and_keeps_registry_nvidia_only_without_it() {
        let registry = [
            AdapterRecord {
                description: "Intel(R) Iris(R) Xe Graphics".into(),
                hardware_id: "PCI\\VEN_8086&DEV_46A6&SUBSYS_13231043".into(),
                ..AdapterRecord::default()
            },
            AdapterRecord {
                description: "NVIDIA GeForce RTX 3070 Ti Laptop GPU".into(),
                hardware_id: "pci\\ven_10de&dev_24a0&subsys_13231043".into(),
                memory_bytes: Some(8_589_934_592),
                ..AdapterRecord::default()
            },
        ];
        let smi = nvidia::parse_device_query(
            "0, NVIDIA GeForce RTX 3070 Ti Laptop GPU, 00000000:01:00.0, 8192, 8.6, 616.92\n",
        )
        .unwrap();
        let merged = merge_gpus(Some(smi), &registry);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].vendor, GpuVendor::Nvidia);
        assert!(merged[0].arch.known().is_some());
        assert_eq!(merged[1].vendor, GpuVendor::Intel);

        let without = merge_gpus(None, &registry);
        assert_eq!(without.len(), 2);
        let nv = without.iter().find(|g| g.vendor == GpuVendor::Nvidia).unwrap();
        assert!(nv.arch.known().is_none());
        assert_eq!(nv.vram_mib, Some(8192));
    }

    fn adapter(description: &str, hardware_id: &str) -> AdapterRecord {
        AdapterRecord {
            description: description.into(),
            hardware_id: hardware_id.into(),
            ..AdapterRecord::default()
        }
    }

    /// A user who moved from an RX 6800 to an RTX card still has the RX 6800's class key (0000);
    /// only the present card's key (0001) survives, so no phantom AMD GPU is reported.
    #[test]
    fn class_keys_of_absent_devices_are_dropped() {
        let records = vec![
            ("0000".to_string(), adapter("AMD Radeon RX 6800", "PCI\\VEN_1002&DEV_73BF")),
            ("0001".to_string(), adapter("NVIDIA GeForce RTX 4070", "PCI\\VEN_10DE&DEV_2786")),
        ];
        let present = BTreeSet::from(["0001".to_string()]);
        let kept = keep_present(records, &present);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].description, "NVIDIA GeForce RTX 4070");
        let gpus = merge_gpus(None, &kept);
        assert!(gpus.iter().all(|g| g.vendor != GpuVendor::Amd), "{gpus:?}");
        assert!(keep_present(vec![("0000".into(), adapter("x", "y"))], &BTreeSet::new()).is_empty());
    }

    /// Two NVIDIA adapters and no nvidia-smi: their links were never examined, so the topology is
    /// unknown rather than "PCIe only". One adapter, or none, has no links to relate.
    #[test]
    fn missing_nvidia_smi_with_two_nvidia_gpus_leaves_the_topology_unknown() {
        let two = [
            adapter("NVIDIA GeForce RTX 3090", "PCI\\VEN_10DE&DEV_2204"),
            adapter("NVIDIA GeForce RTX 3090", "PCI\\VEN_10DE&DEV_2204"),
            adapter("Intel(R) UHD Graphics 770", "PCI\\VEN_8086&DEV_A780"),
        ];
        let topo = topology_without_smi(Some(&two));
        assert!(topo.reason().is_some_and(|r| r.contains("2 NVIDIA GPUs")), "{topo:?}");
        assert_eq!(topology_without_smi(Some(&two[1..])), Detected::Known(GpuTopology::default()));
        assert!(topology_without_smi(None).known().is_none());
    }

    #[test]
    fn an_unknown_profile_has_no_gpus_to_iterate() {
        let p = HardwareProfile::unknown("test");
        assert_eq!(p.gpus_of(GpuVendor::Nvidia).count(), 0);
        assert_eq!(p.os.reason(), Some("test"));
    }

    /// The live probe on the development machine: an RTX 3070 Ti Laptop (Ampere) through
    /// nvidia-smi, the Iris Xe from the registry, no NPU, Windows 11 on x86-64 with AVX2 and no
    /// AVX-512.
    #[cfg(windows)]
    #[test]
    #[ignore = "requires the development machine's hardware (RTX 3070 Ti Laptop + Iris Xe)"]
    fn live_probe_sees_the_development_machine() {
        let p = probe();
        let os = p.os.known().unwrap();
        assert_eq!(os.family, OsFamily::Windows);
        assert!(os.name.starts_with("Windows 11"), "{}", os.name);
        let cpu = p.cpu.known().unwrap();
        assert_eq!(cpu.arch, CpuArch::X86_64);
        let isa = cpu.isa.known().unwrap();
        assert!(isa.has(Isa::Avx2));
        assert!(!isa.has(Isa::Avx512F));
        assert!(p.memory.known().unwrap().total_mib > 8 * 1024);
        let gpus = p.gpus.known().unwrap();
        let nv: Vec<_> = p.gpus_of(GpuVendor::Nvidia).collect();
        assert_eq!(nv.len(), 1, "{gpus:?}");
        assert_eq!(nv[0].arch, Detected::Known(GpuArch::Nvidia(NvidiaArch::Ampere)));
        let intel: Vec<_> = p.gpus_of(GpuVendor::Intel).collect();
        assert_eq!(intel.len(), 1, "{gpus:?}");
        assert_eq!(intel[0].arch, Detected::Known(GpuArch::Intel(IntelArch::XeLp)));
        assert_eq!(intel[0].kind, GpuKind::Integrated);
        assert_eq!(p.topology, Detected::Known(GpuTopology::default()));
        assert_eq!(p.npus, Detected::Known(Vec::new()));
        assert_eq!(p.runtimes.vulkan_loader, Detected::Known(true));
    }
}
