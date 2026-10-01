//! GPUs and NPUs: what they are, classified from the identifiers the OS and vendor tools report.
//!
//! Classification is table-driven and deliberately conservative. A device whose PCI ID is not in a
//! table keeps its vendor and name but gets [`Detected::Unknown`] for its architecture: a guessed
//! architecture would unlock (or hide) features the device does not (or does) have. The tables
//! cover the generations llama.cpp's official GPU builds and the optional runtimes target; adding a
//! row is the whole job of supporting a new part.
//!
//! NVIDIA devices are never classified here by PCI ID: their architecture comes from the compute
//! capability `nvidia-smi` reports (see [`crate::hardware::nvidia`]), which is authoritative and
//! needs no table that goes stale with every product launch.

use std::fmt;

use super::Detected;

/// A PCI vendor/device pair, e.g. `10DE:24A0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PciId {
    /// PCI vendor ID (`0x10DE` NVIDIA, `0x1002` AMD, `0x8086` Intel, `0x1022` AMD system devices).
    pub vendor: u16,
    /// PCI device ID.
    pub device: u16,
}

impl fmt::Display for PciId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04X}:{:04X}", self.vendor, self.device)
    }
}

/// PCI vendor ID of NVIDIA.
pub const PCI_VENDOR_NVIDIA: u16 = 0x10DE;
/// PCI vendor ID of AMD/ATI graphics.
pub const PCI_VENDOR_AMD: u16 = 0x1002;
/// PCI vendor ID of AMD system devices (NPUs are enumerated under it).
pub const PCI_VENDOR_AMD_SYSTEM: u16 = 0x1022;
/// PCI vendor ID of Intel.
pub const PCI_VENDOR_INTEL: u16 = 0x8086;
/// PCI vendor ID of Qualcomm.
pub const PCI_VENDOR_QUALCOMM: u16 = 0x17CB;
/// PCI vendor ID Microsoft uses for its software display adapters (Basic Display, RDP).
pub const PCI_VENDOR_MICROSOFT: u16 = 0x1414;

/// A device's hardware ID as Windows reports it in `MatchingDeviceId`, parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareId {
    /// `PCI\VEN_xxxx&DEV_yyyy...`.
    Pci(PciId),
    /// `ACPI\VEN_xxxx&DEV_yyyy` or `ACPI\xxxxyyyy` (SoC devices on Arm laptops), upper-cased.
    Acpi {
        /// ACPI vendor, e.g. `QCOM`.
        vendor: String,
        /// ACPI device part, e.g. `0D0A`, when the ID carries one.
        device: Option<String>,
    },
    /// Anything else (software adapters, virtual devices).
    Other,
}

/// Parse a Windows hardware ID. Case-insensitive; trailing `&SUBSYS_...&REV_..` parts are ignored.
pub fn parse_hardware_id(raw: &str) -> HardwareId {
    let lower = raw.trim().to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("pci\\") {
        let mut vendor = None;
        let mut device = None;
        for part in rest.split('&') {
            if let Some(v) = part.strip_prefix("ven_") {
                vendor = parse_hex16(v);
            } else if let Some(d) = part.strip_prefix("dev_") {
                device = parse_hex16(d);
            }
        }
        return match (vendor, device) {
            (Some(vendor), Some(device)) => HardwareId::Pci(PciId { vendor, device }),
            _ => HardwareId::Other,
        };
    }
    if let Some(rest) = lower.strip_prefix("acpi\\") {
        let (vendor, device) = match rest.strip_prefix("ven_") {
            Some(v) => {
                let mut parts = v.split('&');
                let vendor = parts.next().unwrap_or_default().to_string();
                let device = parts.find_map(|p| p.strip_prefix("dev_")).map(str::to_string);
                (vendor, device)
            }
            // `ACPI\QCOM0D0A`-style: the vendor is the leading four letters, the device the rest.
            None => {
                let vendor: String = rest.chars().take_while(char::is_ascii_alphabetic).take(4).collect();
                let device = rest
                    .get(vendor.len()..)
                    .map(|d| d.split(['&', '\\']).next().unwrap_or_default().to_string());
                (vendor, device)
            }
        };
        if vendor.is_empty() {
            return HardwareId::Other;
        }
        return HardwareId::Acpi {
            vendor: vendor.to_ascii_uppercase(),
            device: device.filter(|d| !d.is_empty()).map(|d| d.to_ascii_uppercase()),
        };
    }
    HardwareId::Other
}

fn parse_hex16(text: &str) -> Option<u16> {
    (text.len() == 4)
        .then(|| u16::from_str_radix(text, 16).ok())
        .flatten()
}

/// Who made a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GpuVendor {
    /// NVIDIA (CUDA).
    Nvidia,
    /// AMD Radeon / Instinct (ROCm/HIP, Vulkan).
    Amd,
    /// Intel Arc and integrated graphics (SYCL, Vulkan).
    Intel,
    /// Qualcomm Adreno (Snapdragon).
    Qualcomm,
    /// Apple Silicon GPU (Metal, MLX).
    Apple,
}

impl GpuVendor {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Qualcomm => "Qualcomm",
            Self::Apple => "Apple",
        }
    }
}

/// Whether a GPU has its own memory or shares system RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GpuKind {
    /// Own VRAM on a separate board or package.
    Discrete,
    /// Shares system memory with the CPU (iGPU, SoC GPU, Apple unified memory).
    Integrated,
    /// Not established from the identifiers available.
    Unknown,
}

/// NVIDIA GPU architecture, from compute capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NvidiaArch {
    /// Compute capability 5.x.
    Maxwell,
    /// Compute capability 6.x.
    Pascal,
    /// Compute capability 7.0 / 7.2.
    Volta,
    /// Compute capability 7.5.
    Turing,
    /// Compute capability 8.0 / 8.6 / 8.7 (A100, RTX 30, Jetson Orin).
    Ampere,
    /// Compute capability 8.9 (RTX 40, L4, L40): first with FP8 tensor cores.
    Ada,
    /// Compute capability 9.0 (H100, H200, GH200): FlashAttention 3's target.
    Hopper,
    /// Compute capability 10.x / 11.0 (B200, GB200, B300, Thor): datacenter Blackwell.
    BlackwellDatacenter,
    /// Compute capability 12.x (RTX 50, RTX PRO Blackwell, DGX Spark): client Blackwell.
    BlackwellClient,
}

impl NvidiaArch {
    /// Display name, e.g. `Hopper`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Maxwell => "Maxwell",
            Self::Pascal => "Pascal",
            Self::Volta => "Volta",
            Self::Turing => "Turing",
            Self::Ampere => "Ampere",
            Self::Ada => "Ada Lovelace",
            Self::Hopper => "Hopper",
            Self::BlackwellDatacenter => "Blackwell (datacenter)",
            Self::BlackwellClient => "Blackwell",
        }
    }
}

/// AMD GPU architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AmdArch {
    /// RDNA 2 (RX 6000, gfx103x).
    Rdna2,
    /// RDNA 3 (RX 7000, Radeon 780M, gfx110x).
    Rdna3,
    /// RDNA 3.5 (Strix Point / Strix Halo iGPUs, gfx115x).
    Rdna35,
    /// RDNA 4 (RX 9000, gfx120x).
    Rdna4,
    /// CDNA 2 (Instinct MI200, gfx90a).
    Cdna2,
    /// CDNA 3 (Instinct MI300, gfx942): FP8 matrix cores.
    Cdna3,
}

impl AmdArch {
    /// Display name, e.g. `RDNA 3`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rdna2 => "RDNA 2",
            Self::Rdna3 => "RDNA 3",
            Self::Rdna35 => "RDNA 3.5",
            Self::Rdna4 => "RDNA 4",
            Self::Cdna2 => "CDNA 2",
            Self::Cdna3 => "CDNA 3",
        }
    }
}

/// Intel GPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IntelArch {
    /// Xe-LP integrated graphics (Iris Xe / UHD on 11th-14th gen Core).
    XeLp,
    /// Xe-LPG integrated graphics (Arc graphics on Core Ultra Series 1 / 200H).
    XeLpg,
    /// Xe2-LPG integrated graphics (Arc 130V/140V on Lunar Lake).
    Xe2Lpg,
    /// Arc A-series discrete (Alchemist, DG2).
    Alchemist,
    /// Arc B-series discrete (Battlemage).
    Battlemage,
}

impl IntelArch {
    /// Display name, e.g. `Arc Alchemist`.
    pub fn name(self) -> &'static str {
        match self {
            Self::XeLp => "Xe-LP",
            Self::XeLpg => "Xe-LPG",
            Self::Xe2Lpg => "Xe2-LPG",
            Self::Alchemist => "Arc Alchemist",
            Self::Battlemage => "Arc Battlemage",
        }
    }
}

/// A GPU's architecture, per vendor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GpuArch {
    /// NVIDIA, from compute capability.
    Nvidia(NvidiaArch),
    /// AMD, with the LLVM target (`gfx1100`) ROCm/HIP builds are compiled for.
    Amd {
        /// Architecture family.
        family: AmdArch,
        /// LLVM `gfx` target of this die.
        gfx: String,
    },
    /// Intel.
    Intel(IntelArch),
    /// Qualcomm Adreno (X1 series on Snapdragon X).
    Adreno,
    /// Apple Silicon GPU.
    Apple,
}

/// One GPU, physically present.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuDevice {
    /// Manufacturer.
    pub vendor: GpuVendor,
    /// Name as the driver reports it.
    pub name: String,
    /// Discrete or integrated.
    pub kind: GpuKind,
    /// Architecture; `Unknown` when the identifiers do not establish it.
    pub arch: Detected<GpuArch>,
    /// PCI identity, when the device is on PCI (SoC GPUs are ACPI devices and have none).
    pub pci: Option<PciId>,
    /// Dedicated memory in MiB. `None` for integrated GPUs (their driver-reported carve-out is
    /// not usable model memory) and when not reported.
    pub vram_mib: Option<u64>,
    /// Driver version as reported.
    pub driver_version: Option<String>,
    /// CUDA compute capability (NVIDIA only, from `nvidia-smi`).
    pub compute_capability: Option<super::nvidia::ComputeCapability>,
    /// `nvidia-smi` index, which the topology matrix refers to (NVIDIA only).
    pub nvidia_index: Option<u32>,
}

/// Who made an NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NpuVendor {
    /// Intel AI Boost (Core Ultra).
    Intel,
    /// AMD XDNA (Ryzen AI).
    Amd,
    /// Qualcomm Hexagon (Snapdragon).
    Qualcomm,
}

impl NpuVendor {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Intel => "Intel",
            Self::Amd => "AMD",
            Self::Qualcomm => "Qualcomm",
        }
    }
}

/// NPU generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NpuArch {
    /// Intel NPU 3720 (Meteor Lake, Arrow Lake).
    IntelNpu3,
    /// Intel NPU 4 (Lunar Lake).
    IntelNpu4,
    /// AMD XDNA (Phoenix, Hawk Point: Ryzen 7040/8040).
    AmdXdna,
    /// AMD XDNA 2 (Strix Point, Strix Halo, Krackan: Ryzen AI 300 and newer).
    AmdXdna2,
    /// Qualcomm Hexagon NPU (Snapdragon X).
    QualcommHexagon,
}

impl NpuArch {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::IntelNpu3 => "Intel AI Boost 3720",
            Self::IntelNpu4 => "Intel AI Boost 4",
            Self::AmdXdna => "AMD XDNA",
            Self::AmdXdna2 => "AMD XDNA 2",
            Self::QualcommHexagon => "Qualcomm Hexagon",
        }
    }
}

/// One NPU, physically present.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NpuDevice {
    /// Manufacturer, when established.
    pub vendor: Detected<NpuVendor>,
    /// Name as the driver reports it.
    pub name: String,
    /// Generation; `Unknown` when the identifiers do not establish it.
    pub arch: Detected<NpuArch>,
    /// PCI identity, when on PCI.
    pub pci: Option<PciId>,
    /// Driver version as reported.
    pub driver_version: Option<String>,
}

/// One device-class registry entry, as read (or as recorded for a fixture).
///
/// The probe reads these from `HKLM\SYSTEM\CurrentControlSet\Control\Class\{guid}\NNNN`; keeping
/// the raw record separate from classification lets every classification rule be tested against
/// recorded entries from machines Cerebro has never run on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AdapterRecord {
    /// `DriverDesc`, e.g. `AMD Radeon RX 7900 XTX`.
    pub description: String,
    /// `MatchingDeviceId`, e.g. `pci\ven_1002&dev_744c`.
    pub hardware_id: String,
    /// `ProviderName`, e.g. `Advanced Micro Devices, Inc.`.
    pub provider: Option<String>,
    /// `DriverVersion`.
    pub driver_version: Option<String>,
    /// `HardwareInformation.qwMemorySize` (or `MemorySize`) in bytes.
    pub memory_bytes: Option<u64>,
}

/// AMD GPU device-ID table: (device ID, family, gfx target, kind).
const AMD_GPUS: &[(u16, AmdArch, &str, GpuKind)] = &[
    (0x73A5, AmdArch::Rdna2, "gfx1030", GpuKind::Discrete), // RX 6950 XT
    (0x73AF, AmdArch::Rdna2, "gfx1030", GpuKind::Discrete), // RX 6900 XT
    (0x73BF, AmdArch::Rdna2, "gfx1030", GpuKind::Discrete), // RX 6800 / 6800 XT / 6900 XT
    (0x73DF, AmdArch::Rdna2, "gfx1031", GpuKind::Discrete), // RX 6700 XT / 6750 XT
    (0x73FF, AmdArch::Rdna2, "gfx1032", GpuKind::Discrete), // RX 6600 / 6650 XT
    (0x7448, AmdArch::Rdna3, "gfx1100", GpuKind::Discrete), // Radeon PRO W7900
    (0x744C, AmdArch::Rdna3, "gfx1100", GpuKind::Discrete), // RX 7900 XTX / XT / GRE
    (0x747E, AmdArch::Rdna3, "gfx1101", GpuKind::Discrete), // RX 7800 XT / 7700 XT
    (0x7480, AmdArch::Rdna3, "gfx1102", GpuKind::Discrete), // RX 7600 / 7600 XT
    (0x15BF, AmdArch::Rdna3, "gfx1103", GpuKind::Integrated), // Radeon 780M (Phoenix/Hawk Point)
    (0x150E, AmdArch::Rdna35, "gfx1150", GpuKind::Integrated), // Radeon 890M (Strix Point)
    (0x1586, AmdArch::Rdna35, "gfx1151", GpuKind::Integrated), // Radeon 8060S (Strix Halo)
    (0x7590, AmdArch::Rdna4, "gfx1200", GpuKind::Discrete), // RX 9060 XT
    (0x7550, AmdArch::Rdna4, "gfx1201", GpuKind::Discrete), // RX 9070 / 9070 XT
    (0x740C, AmdArch::Cdna2, "gfx90a", GpuKind::Discrete),  // Instinct MI250X / MI250
    (0x740F, AmdArch::Cdna2, "gfx90a", GpuKind::Discrete),  // Instinct MI210
    (0x74A1, AmdArch::Cdna3, "gfx942", GpuKind::Discrete),  // Instinct MI300X
];

/// Intel GPU device-ID ranges: (first, last, arch, kind).
const INTEL_GPUS: &[(u16, u16, IntelArch, GpuKind)] = &[
    (0x5690, 0x56C1, IntelArch::Alchemist, GpuKind::Discrete), // Arc A-series, Flex 140/170
    (0xE202, 0xE212, IntelArch::Battlemage, GpuKind::Discrete), // Arc B580/B570, Pro B50/B60
    (0x4600, 0x46FF, IntelArch::XeLp, GpuKind::Integrated),    // Alder Lake
    (0x9A00, 0x9AFF, IntelArch::XeLp, GpuKind::Integrated),    // Tiger Lake
    (0xA700, 0xA7FF, IntelArch::XeLp, GpuKind::Integrated),    // Raptor Lake
    (0x7D00, 0x7DFF, IntelArch::XeLpg, GpuKind::Integrated),   // Meteor Lake, Arrow Lake
    (0x6400, 0x64FF, IntelArch::Xe2Lpg, GpuKind::Integrated),  // Lunar Lake
];

/// Qualcomm NPU ACPI device table: (ACPI device part, arch). Every Qualcomm NPU not listed (8cx
/// Gen 3, Snapdragon X2, ...) keeps an unknown generation: the vendor alone does not establish
/// which Hexagon generation, and which QNN support, a part has.
const QCOM_NPUS: &[(&str, NpuArch)] = &[
    // Snapdragon X Elite / X Plus Hexagon NPU. Reconstructed (the Snapdragon reference profile),
    // not captured; replace with a capture from real hardware.
    ("0D0A", NpuArch::QualcommHexagon),
];

/// NPU PCI-ID table: (vendor, device, arch).
const NPUS: &[(u16, u16, NpuArch)] = &[
    (PCI_VENDOR_INTEL, 0x7D1D, NpuArch::IntelNpu3), // Meteor Lake
    (PCI_VENDOR_INTEL, 0xAD1D, NpuArch::IntelNpu3), // Arrow Lake
    (PCI_VENDOR_INTEL, 0x643E, NpuArch::IntelNpu4), // Lunar Lake
    (PCI_VENDOR_AMD_SYSTEM, 0x1502, NpuArch::AmdXdna), // Phoenix / Hawk Point
    (PCI_VENDOR_AMD_SYSTEM, 0x17F0, NpuArch::AmdXdna2), // Strix Point / Strix Halo / Krackan
];

fn provider_mentions(record: &AdapterRecord, needle: &str) -> bool {
    let needle = needle.to_ascii_lowercase();
    record
        .provider
        .as_deref()
        .is_some_and(|p| p.to_ascii_lowercase().contains(&needle))
}

/// Classify one display-class registry entry as a GPU. `None` for software and virtual adapters
/// (Microsoft Basic Display, remote display, anything with no hardware vendor Cerebro can use).
pub fn classify_display_adapter(record: &AdapterRecord) -> Option<GpuDevice> {
    let id = parse_hardware_id(&record.hardware_id);
    let (vendor, pci) = match &id {
        HardwareId::Pci(pci) => {
            let vendor = match pci.vendor {
                PCI_VENDOR_NVIDIA => GpuVendor::Nvidia,
                PCI_VENDOR_AMD => GpuVendor::Amd,
                PCI_VENDOR_INTEL => GpuVendor::Intel,
                PCI_VENDOR_QUALCOMM => GpuVendor::Qualcomm,
                _ => return None,
            };
            (vendor, Some(*pci))
        }
        HardwareId::Acpi { vendor, .. } if vendor == "QCOM" => (GpuVendor::Qualcomm, None),
        HardwareId::Acpi { .. } if provider_mentions(record, "qualcomm") => {
            (GpuVendor::Qualcomm, None)
        }
        _ => return None,
    };

    let (kind, arch) = match (vendor, pci) {
        (GpuVendor::Amd, Some(pci)) => match AMD_GPUS.iter().find(|(d, ..)| *d == pci.device) {
            Some((_, family, gfx, kind)) => (
                *kind,
                Detected::Known(GpuArch::Amd {
                    family: *family,
                    gfx: (*gfx).to_string(),
                }),
            ),
            None => (
                GpuKind::Unknown,
                Detected::unknown(format!("AMD device {pci} is not in Cerebro's table")),
            ),
        },
        (GpuVendor::Intel, Some(pci)) => {
            match INTEL_GPUS
                .iter()
                .find(|(lo, hi, ..)| (*lo..=*hi).contains(&pci.device))
            {
                Some((_, _, arch, kind)) => (*kind, Detected::Known(GpuArch::Intel(*arch))),
                None => (
                    GpuKind::Unknown,
                    Detected::unknown(format!("Intel device {pci} is not in Cerebro's table")),
                ),
            }
        }
        (GpuVendor::Qualcomm, _) => (GpuKind::Integrated, Detected::Known(GpuArch::Adreno)),
        (GpuVendor::Nvidia, _) => (
            GpuKind::Unknown,
            Detected::unknown("nvidia-smi did not report this GPU, so its compute capability is unknown"),
        ),
        _ => (GpuKind::Unknown, Detected::unknown("no PCI identity")),
    };

    let vram_mib = match kind {
        GpuKind::Integrated => None,
        _ => record.memory_bytes.filter(|b| *b > 0).map(|b| b / (1024 * 1024)),
    };
    Some(GpuDevice {
        vendor,
        name: record.description.clone(),
        kind,
        arch,
        pci,
        vram_mib,
        driver_version: record.driver_version.clone(),
        compute_capability: None,
        nvidia_index: None,
    })
}

/// Classify one compute-accelerator-class registry entry as an NPU. Every entry in that class is
/// an accelerator by the OS's own classification, so it is always returned; vendor and generation
/// are `Unknown` unless the identifiers establish them.
pub fn classify_compute_accelerator(record: &AdapterRecord) -> NpuDevice {
    let id = parse_hardware_id(&record.hardware_id);
    let pci = match &id {
        HardwareId::Pci(pci) => Some(*pci),
        _ => None,
    };
    let vendor = match &id {
        HardwareId::Pci(p) if p.vendor == PCI_VENDOR_INTEL => Detected::Known(NpuVendor::Intel),
        HardwareId::Pci(p) if p.vendor == PCI_VENDOR_AMD_SYSTEM || p.vendor == PCI_VENDOR_AMD => {
            Detected::Known(NpuVendor::Amd)
        }
        HardwareId::Pci(p) if p.vendor == PCI_VENDOR_QUALCOMM => {
            Detected::Known(NpuVendor::Qualcomm)
        }
        HardwareId::Acpi { vendor, .. } if vendor == "QCOM" => Detected::Known(NpuVendor::Qualcomm),
        _ if provider_mentions(record, "qualcomm") => Detected::Known(NpuVendor::Qualcomm),
        _ => Detected::unknown(format!(
            "accelerator vendor not established from {:?}",
            record.hardware_id
        )),
    };
    let arch = match (&vendor, pci) {
        (_, Some(pci)) => match NPUS
            .iter()
            .find(|(v, d, _)| *v == pci.vendor && *d == pci.device)
        {
            Some((_, _, arch)) => Detected::Known(*arch),
            None => Detected::unknown(format!("NPU {pci} is not in Cerebro's table")),
        },
        (Detected::Known(NpuVendor::Qualcomm), None) => {
            let device = match &id {
                HardwareId::Acpi { device, .. } => device.as_deref(),
                _ => None,
            };
            match device.and_then(|d| QCOM_NPUS.iter().find(|(id, _)| id.eq_ignore_ascii_case(d))) {
                Some((_, arch)) => Detected::Known(*arch),
                None => Detected::unknown(format!(
                    "Qualcomm NPU {:?} is not in Cerebro's table",
                    record.hardware_id
                )),
            }
        }
        _ => Detected::unknown("no PCI identity"),
    };
    NpuDevice {
        vendor,
        name: record.description.clone(),
        arch,
        pci,
        driver_version: record.driver_version.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(description: &str, hardware_id: &str) -> AdapterRecord {
        AdapterRecord {
            description: description.into(),
            hardware_id: hardware_id.into(),
            ..AdapterRecord::default()
        }
    }

    #[test]
    fn hardware_ids_parse_case_insensitively_and_ignore_suffixes() {
        assert_eq!(
            parse_hardware_id("PCI\\VEN_8086&DEV_46A6&SUBSYS_13231043"),
            HardwareId::Pci(PciId { vendor: 0x8086, device: 0x46A6 })
        );
        assert_eq!(
            parse_hardware_id("pci\\ven_10de&dev_24a0&subsys_13231043"),
            HardwareId::Pci(PciId { vendor: 0x10DE, device: 0x24A0 })
        );
        assert_eq!(
            parse_hardware_id("ACPI\\VEN_QCOM&DEV_0C36"),
            HardwareId::Acpi { vendor: "QCOM".into(), device: Some("0C36".into()) }
        );
        assert_eq!(
            parse_hardware_id("ACPI\\QCOM0d0a"),
            HardwareId::Acpi { vendor: "QCOM".into(), device: Some("0D0A".into()) }
        );
        assert_eq!(parse_hardware_id("ACPI\\VEN_QCOM"), HardwareId::Acpi { vendor: "QCOM".into(), device: None });
        // Truncated or non-hex IDs are not guessed at.
        assert_eq!(parse_hardware_id("PCI\\VEN_10D&DEV_24A0"), HardwareId::Other);
        assert_eq!(parse_hardware_id("PCI\\VEN_10DE"), HardwareId::Other);
        assert_eq!(parse_hardware_id("ROOT\\BasicDisplay"), HardwareId::Other);
        assert_eq!(parse_hardware_id(""), HardwareId::Other);
    }

    #[test]
    fn software_adapters_are_not_gpus() {
        assert!(classify_display_adapter(&record("Microsoft Basic Display Adapter", "ROOT\\BasicDisplay")).is_none());
        assert!(classify_display_adapter(&record("Microsoft Remote Display Adapter", "PCI\\VEN_1414&DEV_008E")).is_none());
        assert!(classify_display_adapter(&record("Unknown", "PCI\\VEN_1234&DEV_1111")).is_none());
    }

    #[test]
    fn amd_parts_classify_from_the_table_and_unknown_parts_stay_unknown() {
        let mut xtx = record("AMD Radeon RX 7900 XTX", "PCI\\VEN_1002&DEV_744C&SUBSYS_471E1849&REV_C8");
        xtx.memory_bytes = Some(25_769_803_776);
        let gpu = classify_display_adapter(&xtx).unwrap();
        assert_eq!(gpu.vendor, GpuVendor::Amd);
        assert_eq!(gpu.kind, GpuKind::Discrete);
        assert_eq!(
            gpu.arch,
            Detected::Known(GpuArch::Amd { family: AmdArch::Rdna3, gfx: "gfx1100".into() })
        );
        assert_eq!(gpu.vram_mib, Some(24_576));

        let igpu = classify_display_adapter(&record("AMD Radeon 780M Graphics", "PCI\\VEN_1002&DEV_15BF")).unwrap();
        assert_eq!(igpu.kind, GpuKind::Integrated);
        assert_eq!(igpu.vram_mib, None, "an iGPU carve-out is not model memory");

        let future = classify_display_adapter(&record("AMD Radeon RX 9999", "PCI\\VEN_1002&DEV_7FFF")).unwrap();
        assert_eq!(future.vendor, GpuVendor::Amd);
        assert_eq!(future.kind, GpuKind::Unknown);
        assert!(future.arch.known().is_none());
    }

    #[test]
    fn intel_discrete_and_integrated_are_told_apart_by_id_not_name() {
        // Meteor Lake's iGPU is *named* "Intel(R) Arc(TM) Graphics" but is integrated.
        let mtl = classify_display_adapter(&record("Intel(R) Arc(TM) Graphics", "PCI\\VEN_8086&DEV_7D55")).unwrap();
        assert_eq!(mtl.kind, GpuKind::Integrated);
        assert_eq!(mtl.arch, Detected::Known(GpuArch::Intel(IntelArch::XeLpg)));

        let a770 = classify_display_adapter(&record("Intel(R) Arc(TM) A770 Graphics", "PCI\\VEN_8086&DEV_56A0")).unwrap();
        assert_eq!(a770.kind, GpuKind::Discrete);
        assert_eq!(a770.arch, Detected::Known(GpuArch::Intel(IntelArch::Alchemist)));

        let b580 = classify_display_adapter(&record("Intel(R) Arc(TM) B580 Graphics", "PCI\\VEN_8086&DEV_E20B")).unwrap();
        assert_eq!(b580.arch, Detected::Known(GpuArch::Intel(IntelArch::Battlemage)));

        // DG1 (Iris Xe MAX) is not in the table: unknown, not "integrated because Intel".
        let dg1 = classify_display_adapter(&record("Intel(R) Iris(R) Xe MAX Graphics", "PCI\\VEN_8086&DEV_4905")).unwrap();
        assert_eq!(dg1.kind, GpuKind::Unknown);
        assert!(dg1.arch.known().is_none());
    }

    #[test]
    fn an_nvidia_adapter_seen_only_in_the_registry_has_no_architecture() {
        let gpu = classify_display_adapter(&record("NVIDIA GeForce RTX 3070 Ti Laptop GPU", "pci\\ven_10de&dev_24a0")).unwrap();
        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert!(gpu.arch.known().is_none(), "PCI IDs are never mapped to an NVIDIA architecture");
    }

    #[test]
    fn npus_classify_by_pci_id_or_qualcomm_identity() {
        let intel = classify_compute_accelerator(&record("Intel(R) AI Boost", "PCI\\VEN_8086&DEV_7D1D"));
        assert_eq!(intel.vendor, Detected::Known(NpuVendor::Intel));
        assert_eq!(intel.arch, Detected::Known(NpuArch::IntelNpu3));

        let amd = classify_compute_accelerator(&record("NPU Compute Accelerator Device", "PCI\\VEN_1022&DEV_17F0"));
        assert_eq!(amd.arch, Detected::Known(NpuArch::AmdXdna2));

        let qcom = classify_compute_accelerator(&record("Qualcomm(R) Hexagon(TM) NPU", "ACPI\\VEN_QCOM&DEV_0D0A"));
        assert_eq!(qcom.arch, Detected::Known(NpuArch::QualcommHexagon));
        assert_eq!(
            classify_compute_accelerator(&record("Qualcomm NPU", "ACPI\\QCOM0D0A")).arch,
            Detected::Known(NpuArch::QualcommHexagon)
        );
        // Another Qualcomm NPU (8cx Gen 3, X2, ...) is a Qualcomm NPU of unknown generation.
        let other = classify_compute_accelerator(&record("Qualcomm(R) Hexagon(TM) NPU", "ACPI\\VEN_QCOM&DEV_0C1B"));
        assert_eq!(other.vendor, Detected::Known(NpuVendor::Qualcomm));
        assert!(other.arch.known().is_none(), "{:?}", other.arch);

        let unknown = classify_compute_accelerator(&record("Some Accelerator", "PCI\\VEN_ABCD&DEV_0001"));
        assert!(unknown.vendor.known().is_none());
        assert!(unknown.arch.known().is_none());

        let new_intel = classify_compute_accelerator(&record("Intel(R) AI Boost", "PCI\\VEN_8086&DEV_FFFF"));
        assert_eq!(new_intel.vendor, Detected::Known(NpuVendor::Intel));
        assert!(new_intel.arch.known().is_none());
    }
}
