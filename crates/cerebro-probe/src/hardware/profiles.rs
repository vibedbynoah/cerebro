//! Reference hardware profiles for tests: the machines Cerebro must serve but the development
//! laptop is not.
//!
//! Each profile is assembled the way the live probe assembles one: vendor-tool text goes through
//! [`nvidia::parse_device_query`] and [`nvidia::parse_topology`], registry entries through
//! [`merge_gpus`] and [`accel::classify_compute_accelerator`], x86 CPUID register dumps through
//! [`isa::decode_x86`], and the Linux server's `/etc/os-release` and `/proc/meminfo` text through
//! [`parse_os_release`] and [`parse_meminfo`]. A parser regression therefore shows up as a wrong
//! profile, not only as a failed parser test. Arm instruction sets are listed directly: the live
//! Arm detector is the standard library's (no register dump to decode), and it is checked on real
//! hardware by `isa`'s ignored Windows-on-Arm live test.
//!
//! Provenance, stated so nobody mistakes a reconstruction for a capture:
//!
//! * [`rtx3070ti_laptop`] was **captured** on the development machine on 2026-10-01
//!   (`nvidia-smi` query and `topo -m`, the display-class registry entries, CPU, OS, memory, WSL).
//! * Every other profile is **reconstructed** from the vendor tools' documented output formats and
//!   the public PCI IDs of those parts. They exercise the classification and gating rules; they are
//!   not recordings, and a capture from a real machine should replace one whenever available.
//!
//! Test support only (`cfg(test)` or the `test-support` feature).

use super::accel::{self, AdapterRecord, GpuArch, GpuDevice, GpuKind, GpuVendor};
use super::isa::{self, CpuidRegs, Isa, IsaSet};
use super::nvidia;
use super::system::{
    parse_meminfo, parse_os_release, CpuArch, CpuInfo, MemoryInfo, OsFamily, OsInfo, RuntimeFacts,
    WslDistro, WslInfo,
};
use super::{merge_gpus, Detected, GpuTopology, HardwareProfile};
use crate::error::ProbeError;

/// `nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap,driver_version`,
/// captured on the development machine.
pub const SMI_RTX3070TI_LAPTOP: &str =
    "0, NVIDIA GeForce RTX 3070 Ti Laptop GPU, 00000000:01:00.0, 8192, 8.6, 616.92\n";
/// Reconstructed: one RTX 4090.
pub const SMI_RTX4090: &str = "0, NVIDIA GeForce RTX 4090, 00000000:01:00.0, 24564, 8.9, 581.57\n";
/// Reconstructed: one H100 SXM on a Linux server, driver 570 (below CUDA 13's minimum).
pub const SMI_H100: &str =
    "0, NVIDIA H100 80GB HBM3, 00000000:18:00.0, 81559, 9.0, 570.124.06\n";
/// Reconstructed: two RTX 3090s.
pub const SMI_DUAL_RTX3090: &str = "0, NVIDIA GeForce RTX 3090, 00000000:01:00.0, 24576, 8.6, 581.57\n1, NVIDIA GeForce RTX 3090, 00000000:02:00.0, 24576, 8.6, 581.57\n";
/// Reconstructed: `nvidia-smi topo -m` for two RTX 3090s joined by an NVLink bridge (4 links).
pub const TOPO_DUAL_RTX3090_NVLINK: &str = "\t\u{1b}[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\u{1b}[0m\nGPU0\t X \tNV4\t0-19\t0\t\tN/A\nGPU1\tNV4\t X \t0-19\t0\t\tN/A\n\nLegend:\n\n  X    = Self\n  NV#  = Connection traversing a bonded set of # NVLinks\n";

fn adapter(description: &str, hardware_id: &str, provider: &str, memory_bytes: Option<u64>) -> AdapterRecord {
    AdapterRecord {
        description: description.into(),
        hardware_id: hardware_id.into(),
        provider: Some(provider.into()),
        driver_version: None,
        memory_bytes,
    }
}

fn windows(name: &str, version: &str, build: u32) -> Detected<OsInfo> {
    Detected::Known(OsInfo {
        family: OsFamily::Windows,
        name: name.into(),
        version: Some(version.into()),
        build: Some(build),
    })
}

/// An x86-64 CPU whose instruction sets are decoded from a CPUID register dump.
fn x86(brand: &str, regs: &CpuidRegs, cores: u32, threads: u32) -> Detected<CpuInfo> {
    cpu_with(CpuArch::X86_64, brand, isa::decode_x86(regs), cores, threads)
}

/// An Arm64 CPU with its instruction sets listed (see the module docs).
fn arm(brand: &str, isa: &[Isa], cores: u32, threads: u32) -> Detected<CpuInfo> {
    cpu_with(CpuArch::Aarch64, brand, IsaSet::from_slice(isa), cores, threads)
}

fn cpu_with(arch: CpuArch, brand: &str, isa: IsaSet, cores: u32, threads: u32) -> Detected<CpuInfo> {
    Detected::Known(CpuInfo {
        arch,
        brand: Some(brand.into()),
        isa: Detected::Known(isa),
        physical_cores: Some(cores),
        logical_processors: Some(threads),
    })
}

fn memory(total_mib: u64) -> Detected<MemoryInfo> {
    Detected::Known(MemoryInfo {
        total_mib,
        available_mib: total_mib / 2,
    })
}

fn runtimes(wsl: Detected<WslInfo>, vulkan: bool, hip: bool) -> RuntimeFacts {
    RuntimeFacts {
        wsl,
        vulkan_loader: Detected::Known(vulkan),
        hip_runtime: Detected::Known(hip),
    }
}

/// WSL installed with these distributions, on a machine where WSL 2 starts (the reconstructed
/// profiles model working setups; the live probe leaves the start fact unknown).
fn wsl(distros: &[(&str, u32)]) -> Detected<WslInfo> {
    Detected::Known(WslInfo {
        installed: true,
        wsl2_can_start: Detected::Known(true),
        distributions: distros
            .iter()
            .map(|(name, version)| WslDistro {
                name: (*name).into(),
                version: Some(*version),
            })
            .collect(),
    })
}

fn no_wsl() -> Detected<WslInfo> {
    Detected::Known(WslInfo::default())
}

fn npus(records: &[AdapterRecord]) -> Detected<Vec<super::NpuDevice>> {
    Detected::Known(records.iter().map(accel::classify_compute_accelerator).collect())
}

fn no_topology() -> Detected<GpuTopology> {
    Detected::Known(GpuTopology::default())
}

/// `CPUID.1:ECX` bits SSE4.2 (20), FMA (12), AVX (28), F16C (29) and OSXSAVE (27).
const LEAF1_AVX_FAMILY: u32 = (1 << 20) | (1 << 12) | (1 << 28) | (1 << 29) | (1 << 27);
/// `CPUID.(7,0):EBX` bits AVX2 (5), AVX-512 F (16), BW (30), VL (31).
const LEAF7_AVX2: u32 = 1 << 5;
const LEAF7_AVX512_FBWVL: u32 = (1 << 16) | (1 << 30) | (1 << 31);
/// `XCR0`: x87, SSE and AVX state; plus opmask/ZMM state; plus AMX tile state.
const XCR0_AVX: u64 = 0b111;
const XCR0_AVX512: u64 = 0b1110_0111;
const XCR0_AMX: u64 = 0b1110_0111 | (0b11 << 17);

/// Captured on the development machine's i9-12900H (Alder Lake) on 2026-10-01 with
/// `isa::read_cpuid`: AVX2 and AVX-VNNI, AVX-512 fused off, OS saving AVX state only. Raptor Lake
/// and Meteor Lake report the same bits for every extension Cerebro decodes, so the 13600K and 155H
/// profiles reuse it (reconstructed, not captured on those parts).
const ALDER_LAKE_CAPTURED: CpuidRegs = CpuidRegs {
    leaf1_ecx: 0xfffa_f38b,
    leaf7_ebx: 0x239c_27a9,
    leaf7_ecx: 0x1840_27a4,
    leaf7_edx: 0xbc18_c410,
    leaf7_1_eax: 0x0040_0810,
    xcr0: 0x7,
};
/// Reconstructed from the SDM bits: AVX2 class (Comet Lake, Zen 3), no AVX-512, no AVX-VNNI.
const AVX2_CLASS: CpuidRegs = CpuidRegs {
    leaf1_ecx: LEAF1_AVX_FAMILY,
    leaf7_ebx: LEAF7_AVX2,
    leaf7_ecx: 0,
    leaf7_edx: 0,
    leaf7_1_eax: 0,
    xcr0: XCR0_AVX,
};
/// Reconstructed: Zen 4 / Zen 5 (AVX-512 F/BW/VL, VNNI (leaf 7 ECX 11), BF16 (leaf 7.1 EAX 5)).
const ZEN4_ZEN5: CpuidRegs = CpuidRegs {
    leaf1_ecx: LEAF1_AVX_FAMILY,
    leaf7_ebx: LEAF7_AVX2 | LEAF7_AVX512_FBWVL,
    leaf7_ecx: 1 << 11,
    leaf7_edx: 0,
    leaf7_1_eax: 1 << 5,
    xcr0: XCR0_AVX512,
};
/// Reconstructed: Sapphire Rapids, adding AMX-BF16/TILE/INT8 (leaf 7 EDX 22/24/25) and AVX-VNNI,
/// with the OS saving AMX tile state.
const SAPPHIRE_RAPIDS: CpuidRegs = CpuidRegs {
    leaf1_ecx: LEAF1_AVX_FAMILY,
    leaf7_ebx: LEAF7_AVX2 | LEAF7_AVX512_FBWVL,
    leaf7_ecx: 1 << 11,
    leaf7_edx: (1 << 22) | (1 << 24) | (1 << 25),
    leaf7_1_eax: (1 << 4) | (1 << 5),
    xcr0: XCR0_AMX,
};
const ARM_V8_6: &[Isa] = &[Isa::Neon, Isa::DotProd, Isa::I8mm];

/// Reconstructed `/etc/os-release` of the H100 server (Ubuntu 22.04.5).
pub const OS_RELEASE_UBUNTU_2204: &str = "PRETTY_NAME=\"Ubuntu 22.04.5 LTS\"\nNAME=\"Ubuntu\"\nVERSION_ID=\"22.04\"\nVERSION=\"22.04.5 LTS (Jammy Jellyfish)\"\nID=ubuntu\nID_LIKE=debian\n";
/// Reconstructed `/proc/meminfo` head of the H100 server (2 TiB class).
pub const MEMINFO_H100: &str = "MemTotal:       2113323008 kB\nMemFree:        1985000000 kB\nMemAvailable:   2013323008 kB\nBuffers:          123456 kB\n";

/// Captured: ASUS laptop, i9-12900H, RTX 3070 Ti Laptop (8 GB, Ampere) + Iris Xe, Windows 11 25H2,
/// WSL with only Docker Desktop's distributions.
pub fn rtx3070ti_laptop() -> Result<HardwareProfile, ProbeError> {
    let registry = [
        AdapterRecord {
            description: "Intel(R) Iris(R) Xe Graphics".into(),
            hardware_id: "PCI\\VEN_8086&DEV_46A6&SUBSYS_13231043".into(),
            provider: Some("Intel Corporation".into()),
            driver_version: Some("30.0.101.1660".into()),
            memory_bytes: None,
        },
        AdapterRecord {
            description: "NVIDIA GeForce RTX 3070 Ti Laptop GPU".into(),
            hardware_id: "pci\\ven_10de&dev_24a0&subsys_13231043".into(),
            provider: Some("NVIDIA".into()),
            driver_version: Some("32.0.16.1692".into()),
            memory_bytes: Some(8_589_934_592),
        },
    ];
    Ok(HardwareProfile {
        os: windows("Windows 11 25H2", "10.0.26200.9550", 26200),
        cpu: x86("12th Gen Intel(R) Core(TM) i9-12900H", &ALDER_LAKE_CAPTURED, 14, 20),
        memory: memory(32_458),
        gpus: Detected::Known(merge_gpus(
            Some(nvidia::parse_device_query(SMI_RTX3070TI_LAPTOP)?),
            &registry,
        )),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(wsl(&[("docker-desktop", 2)]), true, false),
    })
}

/// Reconstructed: desktop, Ryzen 9 7950X (whose Raphael iGPU is not in the AMD table), RTX 4090,
/// Windows 11 24H2, WSL with Ubuntu.
pub fn rtx4090_desktop() -> Result<HardwareProfile, ProbeError> {
    let registry = [
        adapter("NVIDIA GeForce RTX 4090", "PCI\\VEN_10DE&DEV_2684", "NVIDIA", Some(25_757_220_864)),
        adapter("AMD Radeon(TM) Graphics", "PCI\\VEN_1002&DEV_164E", "Advanced Micro Devices, Inc.", Some(536_870_912)),
    ];
    Ok(HardwareProfile {
        os: windows("Windows 11 24H2", "10.0.26100.6584", 26100),
        cpu: x86("AMD Ryzen 9 7950X 16-Core Processor", &ZEN4_ZEN5, 16, 32),
        memory: memory(65_536),
        gpus: Detected::Known(merge_gpus(Some(nvidia::parse_device_query(SMI_RTX4090)?), &registry)),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(wsl(&[("Ubuntu-24.04", 2)]), true, false),
    })
}

/// Reconstructed: Linux server, Xeon Platinum 8480+ (AMX), one H100 SXM, driver 570.
pub fn h100_server() -> Result<HardwareProfile, ProbeError> {
    Ok(HardwareProfile {
        os: parse_os_release(OS_RELEASE_UBUNTU_2204)
            .map_or_else(|| Detected::unknown("os-release fixture did not parse"), Detected::Known),
        cpu: x86("Intel(R) Xeon(R) Platinum 8480+", &SAPPHIRE_RAPIDS, 56, 112),
        memory: parse_meminfo(MEMINFO_H100)
            .map_or_else(|| Detected::unknown("meminfo fixture did not parse"), Detected::Known),
        gpus: Detected::Known(merge_gpus(Some(nvidia::parse_device_query(SMI_H100)?), &[])),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(Detected::unknown("WSL does not apply on Linux"), false, false),
    })
}

/// Reconstructed: workstation, i9-10900K, two RTX 3090s on an NVLink bridge, Windows 11 23H2,
/// WSL with Ubuntu.
pub fn dual_rtx3090_nvlink() -> Result<HardwareProfile, ProbeError> {
    Ok(HardwareProfile {
        os: windows("Windows 11 23H2", "10.0.22631.5909", 22631),
        cpu: x86("Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz", &AVX2_CLASS, 10, 20),
        memory: memory(131_072),
        gpus: Detected::Known(merge_gpus(Some(nvidia::parse_device_query(SMI_DUAL_RTX3090)?), &[])),
        topology: Detected::Known(nvidia::parse_topology(TOPO_DUAL_RTX3090_NVLINK)?),
        npus: npus(&[]),
        runtimes: runtimes(wsl(&[("Ubuntu-22.04", 2)]), true, false),
    })
}

/// Reconstructed: desktop, Ryzen 7 7800X3D, Radeon RX 7900 XTX (RDNA 3, gfx1100) with the HIP
/// runtime from the Adrenalin driver, Windows 11 24H2, no WSL, no NVIDIA driver.
pub fn rx7900xtx() -> Result<HardwareProfile, ProbeError> {
    let registry = [adapter(
        "AMD Radeon RX 7900 XTX",
        "PCI\\VEN_1002&DEV_744C&SUBSYS_471E1849&REV_C8",
        "Advanced Micro Devices, Inc.",
        Some(25_769_803_776),
    )];
    Ok(HardwareProfile {
        os: windows("Windows 11 24H2", "10.0.26100.6584", 26100),
        cpu: x86("AMD Ryzen 7 7800X3D 8-Core Processor", &ZEN4_ZEN5, 8, 16),
        memory: memory(32_768),
        gpus: Detected::Known(merge_gpus(None, &registry)),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(no_wsl(), true, true),
    })
}

/// Reconstructed: desktop, i5-13600K (UHD 770 iGPU), Arc A770 16 GB, Windows 11 24H2.
pub fn arc_a770() -> Result<HardwareProfile, ProbeError> {
    let registry = [
        adapter("Intel(R) Arc(TM) A770 Graphics", "PCI\\VEN_8086&DEV_56A0", "Intel Corporation", Some(17_179_869_184)),
        adapter("Intel(R) UHD Graphics 770", "PCI\\VEN_8086&DEV_A780", "Intel Corporation", Some(134_217_728)),
    ];
    Ok(HardwareProfile {
        os: windows("Windows 11 24H2", "10.0.26100.6584", 26100),
        cpu: x86("13th Gen Intel(R) Core(TM) i5-13600K", &ALDER_LAKE_CAPTURED, 14, 20),
        memory: memory(32_768),
        gpus: Detected::Known(merge_gpus(None, &registry)),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(no_wsl(), true, false),
    })
}

/// Reconstructed: desktop, Ryzen 5 5600 (AVX2, no AVX-512) with only Microsoft's basic display
/// adapter, Windows 10 22H2.
pub fn cpu_only() -> Result<HardwareProfile, ProbeError> {
    let registry = [adapter(
        "Microsoft Basic Display Adapter",
        "ROOT\\BasicDisplay",
        "Microsoft",
        None,
    )];
    Ok(HardwareProfile {
        os: windows("Windows 10 22H2", "10.0.19045.6332", 19045),
        cpu: x86("AMD Ryzen 5 5600 6-Core Processor", &AVX2_CLASS, 6, 12),
        memory: memory(16_384),
        gpus: Detected::Known(merge_gpus(None, &registry)),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(no_wsl(), false, false),
    })
}

/// Reconstructed: Snapdragon X Elite laptop (Arm64), Adreno X1-85, Hexagon NPU, Windows 11 24H2.
pub fn snapdragon_x_elite() -> Result<HardwareProfile, ProbeError> {
    let registry = [adapter(
        "Qualcomm(R) Adreno(TM) X1-85 GPU",
        "ACPI\\VEN_QCOM&DEV_0C36",
        "Qualcomm Technologies Inc.",
        None,
    )];
    let npu = [adapter(
        "Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Hexagon(TM) NPU",
        "ACPI\\VEN_QCOM&DEV_0D0A",
        "Qualcomm Technologies Inc.",
        None,
    )];
    Ok(HardwareProfile {
        os: windows("Windows 11 24H2", "10.0.26100.6584", 26100),
        cpu: arm("Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU", ARM_V8_6, 12, 12),
        memory: memory(32_768),
        gpus: Detected::Known(merge_gpus(None, &registry)),
        topology: no_topology(),
        npus: npus(&npu),
        runtimes: runtimes(no_wsl(), true, false),
    })
}

/// Reconstructed: Core Ultra 7 155H laptop (Meteor Lake): Arc iGPU and Intel NPU, no dGPU.
pub fn core_ultra_155h() -> Result<HardwareProfile, ProbeError> {
    let registry = [adapter("Intel(R) Arc(TM) Graphics", "PCI\\VEN_8086&DEV_7D55", "Intel Corporation", Some(134_217_728))];
    let npu = [adapter("Intel(R) AI Boost", "PCI\\VEN_8086&DEV_7D1D", "Intel Corporation", None)];
    Ok(HardwareProfile {
        os: windows("Windows 11 23H2", "10.0.22631.5909", 22631),
        cpu: x86("Intel(R) Core(TM) Ultra 7 155H", &ALDER_LAKE_CAPTURED, 16, 22),
        memory: memory(32_768),
        gpus: Detected::Known(merge_gpus(None, &registry)),
        topology: no_topology(),
        npus: npus(&npu),
        runtimes: runtimes(no_wsl(), true, false),
    })
}

/// Reconstructed: Ryzen AI 9 HX 370 laptop (Strix Point): Radeon 890M iGPU and XDNA 2 NPU.
pub fn ryzen_ai_hx370() -> Result<HardwareProfile, ProbeError> {
    let registry = [adapter("AMD Radeon(TM) 890M Graphics", "PCI\\VEN_1002&DEV_150E", "Advanced Micro Devices, Inc.", Some(536_870_912))];
    let npu = [adapter("NPU Compute Accelerator Device", "PCI\\VEN_1022&DEV_17F0", "Advanced Micro Devices, Inc.", None)];
    Ok(HardwareProfile {
        os: windows("Windows 11 24H2", "10.0.26100.6584", 26100),
        cpu: x86("AMD Ryzen AI 9 HX 370 w/ Radeon 890M", &ZEN4_ZEN5, 12, 24),
        memory: memory(32_768),
        gpus: Detected::Known(merge_gpus(None, &registry)),
        topology: no_topology(),
        npus: npus(&npu),
        runtimes: runtimes(no_wsl(), true, true),
    })
}

/// Reconstructed: MacBook Pro, Apple M3 Max, macOS 15 (for the future macOS build).
pub fn apple_m3_max() -> Result<HardwareProfile, ProbeError> {
    Ok(HardwareProfile {
        os: Detected::Known(OsInfo {
            family: OsFamily::MacOs,
            name: "macOS 15.6".into(),
            version: Some("15.6".into()),
            build: None,
        }),
        cpu: arm("Apple M3 Max", ARM_V8_6, 16, 16),
        memory: memory(65_536),
        gpus: Detected::Known(vec![GpuDevice {
            vendor: GpuVendor::Apple,
            name: "Apple M3 Max".into(),
            kind: GpuKind::Integrated,
            arch: Detected::Known(GpuArch::Apple),
            pci: None,
            vram_mib: None,
            driver_version: None,
            compute_capability: None,
            nvidia_index: None,
        }]),
        topology: no_topology(),
        npus: npus(&[]),
        runtimes: runtimes(Detected::unknown("WSL does not apply on macOS"), false, false),
    })
}

/// Every reference profile with a short name, for table-driven tests.
pub fn all() -> Result<Vec<(&'static str, HardwareProfile)>, ProbeError> {
    Ok(vec![
        ("rtx3070ti-laptop", rtx3070ti_laptop()?),
        ("rtx4090", rtx4090_desktop()?),
        ("h100", h100_server()?),
        ("2x-rtx3090-nvlink", dual_rtx3090_nvlink()?),
        ("rx7900xtx", rx7900xtx()?),
        ("arc-a770", arc_a770()?),
        ("cpu-only", cpu_only()?),
        ("snapdragon-x-elite", snapdragon_x_elite()?),
        ("core-ultra-155h", core_ultra_155h()?),
        ("ryzen-ai-hx370", ryzen_ai_hx370()?),
        ("apple-m3-max", apple_m3_max()?),
    ])
}

#[cfg(test)]
mod tests {
    use super::super::{AmdArch, IntelArch, LinkKind, NpuArch, NvidiaArch};
    use super::*;

    fn archs(p: &HardwareProfile) -> Vec<Option<GpuArch>> {
        p.gpus.known().unwrap().iter().map(|g| g.arch.known().cloned()).collect()
    }

    #[test]
    fn every_profile_assembles() {
        assert_eq!(all().unwrap().len(), 11);
    }

    fn isa_of(p: &HardwareProfile) -> IsaSet {
        p.cpu.known().unwrap().isa.known().unwrap().clone()
    }

    /// The x86 profiles' instruction sets come out of the decoder, so a decoder regression shows
    /// here as a wrong profile.
    #[test]
    fn x86_profiles_decode_their_cpuid_dumps() {
        let laptop = isa_of(&rtx3070ti_laptop().unwrap());
        let expected = [Isa::Sse42, Isa::Avx, Isa::Avx2, Isa::Fma, Isa::F16c, Isa::AvxVnni];
        assert_eq!(laptop, IsaSet::from_slice(&expected), "captured Alder Lake");
        let zen4 = isa_of(&rtx4090_desktop().unwrap());
        assert!(zen4.has(Isa::Avx512F) && zen4.has(Isa::Avx512Bf16) && !zen4.has(Isa::AmxInt8));
        let spr = isa_of(&h100_server().unwrap());
        assert!(spr.has(Isa::AmxInt8) && spr.has(Isa::AmxBf16) && spr.has(Isa::AvxVnni));
        let comet = isa_of(&dual_rtx3090_nvlink().unwrap());
        assert!(comet.has(Isa::Avx2) && !comet.has(Isa::AvxVnni) && !comet.has(Isa::Avx512F));
    }

    #[test]
    fn the_linux_server_reads_its_os_and_memory_through_the_parsers() {
        let p = h100_server().unwrap();
        let os = p.os.known().unwrap();
        assert_eq!((os.family, os.name.as_str()), (OsFamily::Linux, "Ubuntu 22.04.5 LTS"));
        assert_eq!(os.version.as_deref(), Some("22.04"));
        assert_eq!(p.memory.known().unwrap().total_mib, 2_063_792);
    }

    #[test]
    fn nvidia_profiles_classify_from_compute_capability() {
        assert_eq!(
            archs(&rtx3070ti_laptop().unwrap()),
            [Some(GpuArch::Nvidia(NvidiaArch::Ampere)), Some(GpuArch::Intel(IntelArch::XeLp))]
        );
        let p = rtx4090_desktop().unwrap();
        // The Raphael iGPU (164E) is not in the table: listed, architecture unknown.
        assert_eq!(archs(&p), [Some(GpuArch::Nvidia(NvidiaArch::Ada)), None]);
        assert_eq!(archs(&h100_server().unwrap()), [Some(GpuArch::Nvidia(NvidiaArch::Hopper))]);
        let dual = dual_rtx3090_nvlink().unwrap();
        assert_eq!(dual.gpus_of(GpuVendor::Nvidia).count(), 2);
        let topo = dual.topology.known().unwrap();
        assert_eq!(topo.link(0, 1), Some(LinkKind::NvLink(4)));
        assert!(topo.all_pairs_nvlink());
    }

    #[test]
    fn non_nvidia_profiles_classify_from_pci_ids() {
        let amd = rx7900xtx().unwrap();
        assert_eq!(
            archs(&amd),
            [Some(GpuArch::Amd { family: AmdArch::Rdna3, gfx: "gfx1100".into() })]
        );
        assert_eq!(amd.gpus.known().unwrap()[0].vram_mib, Some(24_576));
        let arc = arc_a770().unwrap();
        let gpus = arc.gpus.known().unwrap();
        assert_eq!(gpus[0].kind, GpuKind::Discrete);
        assert_eq!(gpus[1].kind, GpuKind::Integrated);
        assert!(cpu_only().unwrap().gpus.known().unwrap().is_empty());
        let sd = snapdragon_x_elite().unwrap();
        assert_eq!(archs(&sd), [Some(GpuArch::Adreno)]);
        assert_eq!(
            sd.npus.known().unwrap()[0].arch,
            Detected::Known(NpuArch::QualcommHexagon)
        );
        assert_eq!(
            core_ultra_155h().unwrap().npus.known().unwrap()[0].arch,
            Detected::Known(NpuArch::IntelNpu3)
        );
        assert_eq!(
            ryzen_ai_hx370().unwrap().npus.known().unwrap()[0].arch,
            Detected::Known(NpuArch::AmdXdna2)
        );
    }
}
