//! macOS detection as a pure function over `sysctl` values.
//!
//! [`assemble`] takes a lookup (`key -> value`) so it is tested on recorded values from any build
//! platform; [`crate::hardware::probe`] on macOS passes one that runs `/usr/sbin/sysctl -n <key>`.
//! It establishes the OS, the CPU (architecture, brand, core counts, instruction sets on a native
//! build), memory and, on Apple Silicon, the GPU, so the MLX and Metal rows of the capability
//! matrix are decided rather than "could not check".

use super::accel::{GpuArch, GpuDevice, GpuKind, GpuVendor};
use super::isa::IsaSet;
use super::nvidia::GpuTopology;
use super::system::{CpuArch, CpuInfo, MemoryInfo, OsFamily, OsInfo, RuntimeFacts};
use super::{Detected, HardwareProfile};

/// The macOS hardware profile from `sysctl` values.
///
/// Keys read: `kern.osproductversion`, `hw.optional.arm64` (`1` on Apple Silicon, also for an
/// x86-64 build under Rosetta), `machdep.cpu.brand_string`, `hw.physicalcpu`, `hw.logicalcpu`,
/// `hw.memsize`, `hw.pagesize`, `vm.page_free_count`. `this_build` is the architecture this
/// process was compiled for; instruction sets are read with `detect_isa` only when it matches the
/// machine (under Rosetta they would describe the translator).
pub fn assemble(
    sysctl: impl Fn(&str) -> Option<String>,
    this_build: Option<CpuArch>,
    detect_isa: impl FnOnce() -> IsaSet,
) -> HardwareProfile {
    let number = |key: &str| sysctl(key).and_then(|v| v.trim().parse::<u64>().ok());

    let os = match sysctl("kern.osproductversion") {
        Some(version) => Detected::Known(OsInfo {
            family: OsFamily::MacOs,
            name: format!("macOS {version}"),
            version: Some(version),
            build: None,
        }),
        None => Detected::unknown("sysctl kern.osproductversion unavailable"),
    };

    // `hw.optional.arm64` is absent on Intel Macs, so absence with a readable OS means x86-64.
    let arch = match (number("hw.optional.arm64"), &os) {
        (Some(1), _) => Some(CpuArch::Aarch64),
        (Some(0) | None, Detected::Known(_)) => Some(CpuArch::X86_64),
        _ => None,
    };

    let cpu = match arch {
        Some(arch) => {
            let isa = if Some(arch) == this_build {
                Detected::Known(detect_isa())
            } else {
                Detected::unknown(format!(
                    "this {} build runs translated on {}; feature flags describe the translator",
                    this_build.map_or("unknown", CpuArch::name),
                    arch.name()
                ))
            };
            Detected::Known(CpuInfo {
                arch,
                brand: sysctl("machdep.cpu.brand_string"),
                isa,
                physical_cores: number("hw.physicalcpu").and_then(|n| u32::try_from(n).ok()),
                logical_processors: number("hw.logicalcpu").and_then(|n| u32::try_from(n).ok()),
            })
        }
        None => Detected::unknown("CPU architecture not established from sysctl"),
    };

    let memory = match number("hw.memsize") {
        Some(total) if total > 0 => {
            // Free pages only: a lower bound on what is available (inactive and purgeable pages
            // can also be reclaimed), never an overstatement.
            let available = number("vm.page_free_count")
                .zip(number("hw.pagesize"))
                .map_or(0, |(pages, size)| pages.saturating_mul(size));
            Detected::Known(MemoryInfo {
                total_mib: total / (1024 * 1024),
                available_mib: available.min(total) / (1024 * 1024),
            })
        }
        _ => Detected::unknown("sysctl hw.memsize unavailable"),
    };

    let gpus = match arch {
        // Every Apple Silicon Mac has exactly one GPU, on the SoC, sharing unified memory.
        Some(CpuArch::Aarch64) => Detected::Known(vec![GpuDevice {
            vendor: GpuVendor::Apple,
            name: sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon GPU".into()),
            kind: GpuKind::Integrated,
            arch: Detected::Known(GpuArch::Apple),
            pci: None,
            vram_mib: None,
            driver_version: None,
            compute_capability: None,
            nvidia_index: None,
        }]),
        Some(CpuArch::X86_64) => Detected::unknown("GPU detection on Intel Macs is not implemented"),
        None => Detected::unknown("CPU architecture not established from sysctl"),
    };

    HardwareProfile {
        os,
        cpu,
        memory,
        gpus,
        topology: Detected::Known(GpuTopology::default()),
        // The Apple Neural Engine exists on every Apple Silicon Mac but is not modelled; "no NPU"
        // would be false.
        npus: Detected::unknown("the Apple Neural Engine is not modelled"),
        runtimes: RuntimeFacts {
            wsl: Detected::unknown("WSL does not apply on macOS"),
            vulkan_loader: Detected::unknown("Vulkan on macOS (MoltenVK) is not detected"),
            // ROCm does not exist for macOS.
            hip_runtime: Detected::Known(false),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::isa::Isa;
    use std::collections::BTreeMap;

    /// `sysctl -n` values of an M3 Max MacBook Pro (reconstructed from Apple's documented keys).
    fn m3_max() -> BTreeMap<&'static str, &'static str> {
        BTreeMap::from([
            ("kern.osproductversion", "15.6"),
            ("hw.optional.arm64", "1"),
            ("machdep.cpu.brand_string", "Apple M3 Max"),
            ("hw.physicalcpu", "16"),
            ("hw.logicalcpu", "16"),
            ("hw.memsize", "68719476736"),
            ("hw.pagesize", "16384"),
            ("vm.page_free_count", "262144"),
        ])
    }

    fn lookup<'a>(map: &'a BTreeMap<&'static str, &'static str>) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| map.get(k).map(|v| (*v).to_string())
    }

    #[test]
    fn apple_silicon_gets_os_cpu_memory_and_its_gpu() {
        let map = m3_max();
        let p = assemble(lookup(&map), Some(CpuArch::Aarch64), || {
            IsaSet::from_slice(&[Isa::Neon, Isa::DotProd, Isa::I8mm])
        });
        let os = p.os.known().unwrap();
        assert_eq!((os.family, os.name.as_str()), (OsFamily::MacOs, "macOS 15.6"));
        let cpu = p.cpu.known().unwrap();
        assert_eq!(cpu.arch, CpuArch::Aarch64);
        assert!(cpu.isa.known().unwrap().has(Isa::I8mm));
        assert_eq!(cpu.physical_cores, Some(16));
        let mem = p.memory.known().unwrap();
        assert_eq!((mem.total_mib, mem.available_mib), (65_536, 4_096));
        let gpus = p.gpus.known().unwrap();
        assert_eq!((gpus[0].vendor, gpus[0].kind), (GpuVendor::Apple, GpuKind::Integrated));
        assert!(p.npus.known().is_none());
    }

    #[test]
    fn a_translated_build_does_not_trust_its_feature_flags() {
        let map = m3_max();
        let p = assemble(lookup(&map), Some(CpuArch::X86_64), || unreachable!("must not be read"));
        let cpu = p.cpu.known().unwrap();
        assert_eq!(cpu.arch, CpuArch::Aarch64);
        assert!(cpu.isa.known().is_none());
    }

    #[test]
    fn intel_macs_and_missing_values_stay_unknown_where_unestablished() {
        let mut map = m3_max();
        map.remove("hw.optional.arm64");
        let p = assemble(lookup(&map), Some(CpuArch::X86_64), IsaSet::default);
        assert_eq!(p.cpu.known().unwrap().arch, CpuArch::X86_64);
        assert!(p.gpus.known().is_none());
        let none = assemble(|_| None, Some(CpuArch::Aarch64), IsaSet::default);
        assert!(none.os.known().is_none() && none.cpu.known().is_none() && none.memory.known().is_none());
    }
}
