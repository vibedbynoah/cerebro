//! Linux detectors as pure functions over a filesystem root.
//!
//! Every reader takes the root it reads under (`/` live, a recorded directory tree in tests), so
//! each rule is tested on fixtures from machines Cerebro has never run on, from any build platform.
//! [`crate::hardware::probe`] on Linux is a thin shell that passes `/` and the `nvidia-smi` answer.
//!
//! Sources: `/sys/class/drm/card<N>/device/{vendor,device}` for GPUs (shaped like the Windows
//! registry records, so [`classify_display_adapter`](super::accel::classify_display_adapter) and
//! [`merge_gpus`](super::merge_gpus) apply unchanged), `/sys/class/accel/accel<N>/device` for NPUs
//! (the kernel's compute-accelerator class: Intel `intel_vpu`, AMD `amdxdna`, Qualcomm `qaic`),
//! `/etc/os-release` and `/proc/meminfo`, and the standard install paths of the Vulkan loader and
//! ROCm's HIP runtime. Nothing is executed here.

use std::io::ErrorKind;
use std::path::Path;

use super::accel::{self, AdapterRecord, GpuDevice};
use super::nvidia::GpuTopology;
use super::system::{self, RuntimeFacts};
use super::{merge_gpus, Detected, HardwareProfile};

/// Library directories the Vulkan loader (`libvulkan.so.1`) is installed into by the major
/// distributions (Debian/Ubuntu multiarch, Fedora/RHEL `lib64`, Arch `lib`).
const LIB_DIRS: &[&str] = &[
    "usr/lib/x86_64-linux-gnu",
    "usr/lib/aarch64-linux-gnu",
    "usr/lib64",
    "usr/lib",
];

/// One PCI function's identity from a sysfs `device` directory: `vendor` and `device` files
/// (`0x1002`), the bound driver's name, and amdgpu's VRAM size. `None` for a device that is not on
/// PCI (no `vendor` file: `simpledrm`, virtual devices).
fn pci_record(device_dir: &Path) -> Option<AdapterRecord> {
    let read = |file: &str| -> Option<String> {
        let text = std::fs::read_to_string(device_dir.join(file)).ok()?;
        let hex = text.trim().trim_start_matches("0x").trim_start_matches("0X");
        (hex.len() == 4 && hex.bytes().all(|b| b.is_ascii_hexdigit())).then(|| hex.to_ascii_uppercase())
    };
    let vendor = read("vendor")?;
    let device = read("device")?;
    let driver = std::fs::read_link(device_dir.join("driver"))
        .ok()
        .and_then(|target| target.file_name().map(|n| n.to_string_lossy().into_owned()));
    let memory_bytes = std::fs::read_to_string(device_dir.join("mem_info_vram_total"))
        .ok()
        .and_then(|t| t.trim().parse::<u64>().ok());
    Some(AdapterRecord {
        description: format!("PCI device {vendor}:{device}"),
        hardware_id: format!("PCI\\VEN_{vendor}&DEV_{device}"),
        provider: driver,
        driver_version: None,
        memory_bytes,
    })
}

/// `<prefix><N>` (digits only after the prefix): `card0` but not `card0-HDMI-A-1` (a connector)
/// or `renderD128` (a second node of the same device).
fn is_numbered(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix)
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Records for every `<prefix><N>` entry under a sysfs class directory. A missing class directory
/// means the kernel registered no such device (`Ok`, empty); any other read failure is `Err`.
fn class_records(class_dir: &Path, prefix: &str) -> Result<Vec<AdapterRecord>, String> {
    let entries = match std::fs::read_dir(class_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", class_dir.display())),
    };
    let mut named: Vec<(String, AdapterRecord)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !is_numbered(&name, prefix) {
                return None;
            }
            pci_record(&entry.path().join("device")).map(|r| (name, r))
        })
        .collect();
    // Directory order is arbitrary; card numbers give a stable order.
    named.sort_by_key(|(name, _)| name.trim_start_matches(prefix).parse::<u32>().unwrap_or(u32::MAX));
    Ok(named.into_iter().map(|(_, r)| r).collect())
}

/// GPU records from `<sys_class_drm>/card<N>/device`, in card order.
pub fn drm_records(sys_class_drm: &Path) -> Result<Vec<AdapterRecord>, String> {
    class_records(sys_class_drm, "card")
}

/// NPU records from `<sys_class_accel>/accel<N>/device`, in device order.
pub fn accel_records(sys_class_accel: &Path) -> Result<Vec<AdapterRecord>, String> {
    class_records(sys_class_accel, "accel")
}

/// Whether `libvulkan.so.1` is in one of the standard library directories under `root`.
pub fn vulkan_loader(root: &Path) -> bool {
    LIB_DIRS
        .iter()
        .any(|dir| root.join(dir).join("libvulkan.so.1").is_file())
}

/// Whether ROCm's HIP runtime (`libamdhip64.so*`) is under `<root>/opt/rocm/lib` or a standard
/// library directory.
pub fn hip_runtime(root: &Path) -> bool {
    std::iter::once("opt/rocm/lib")
        .chain(LIB_DIRS.iter().copied())
        .filter_map(|dir| std::fs::read_dir(root.join(dir)).ok())
        .flat_map(|entries| entries.flatten())
        .any(|e| e.file_name().to_string_lossy().starts_with("libamdhip64.so"))
}

/// The hardware profile of the Linux system under `root`, except the CPU (CPUID describes the
/// running machine, not a recorded tree; the caller fills it in).
///
/// `nvidia` is the `nvidia-smi` answer (`Ok(None)`: not installed). `topology` relates the NVIDIA
/// GPUs given that answer and the DRM records, as the Windows probe does with registry records.
pub fn assemble(
    root: &Path,
    nvidia: Result<Option<Vec<GpuDevice>>, String>,
    topology: impl FnOnce(&Result<Option<Vec<GpuDevice>>, String>, Option<&[AdapterRecord]>) -> Detected<GpuTopology>,
) -> HardwareProfile {
    let os = std::fs::read_to_string(root.join("etc/os-release"))
        .ok()
        .and_then(|t| system::parse_os_release(&t))
        .map_or_else(|| Detected::unknown("/etc/os-release unreadable"), Detected::Known);
    let memory = std::fs::read_to_string(root.join("proc/meminfo"))
        .ok()
        .and_then(|t| system::parse_meminfo(&t))
        .map_or_else(|| Detected::unknown("/proc/meminfo unreadable"), Detected::Known);
    let drm = drm_records(&root.join("sys/class/drm"));
    let topology = topology(&nvidia, drm.as_deref().ok());
    let gpus = match drm {
        Ok(records) => Detected::Known(merge_gpus(nvidia.ok().flatten(), &records)),
        Err(reason) => Detected::unknown(reason),
    };
    let npus = match accel_records(&root.join("sys/class/accel")) {
        Ok(records) => Detected::Known(records.iter().map(accel::classify_compute_accelerator).collect()),
        Err(reason) => Detected::unknown(reason),
    };
    HardwareProfile {
        os,
        cpu: Detected::unknown("filled in by the caller"),
        memory,
        gpus,
        topology,
        npus,
        runtimes: RuntimeFacts {
            wsl: Detected::unknown("WSL does not apply on Linux"),
            vulkan_loader: Detected::Known(vulkan_loader(root)),
            hip_runtime: Detected::Known(hip_runtime(root)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::accel::{AmdArch, GpuArch, GpuKind, GpuVendor, NpuArch};
    use std::path::PathBuf;

    /// A scratch directory tree, removed on drop.
    struct Tree(PathBuf);

    impl Tree {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("cerebro-linux-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, rel: &str, text: &str) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Recorded shape of a desktop with a Radeon RX 7900 XTX on amdgpu (sysfs layout of
    /// `/sys/class/drm` and `/proc`, values reconstructed from the part's public PCI ID).
    fn rx7900xtx_tree(tag: &str) -> Tree {
        let t = Tree::new(tag);
        t.file("sys/class/drm/card1/device/vendor", "0x1002\n");
        t.file("sys/class/drm/card1/device/device", "0x744c\n");
        t.file("sys/class/drm/card1/device/mem_info_vram_total", "25753026560\n");
        // A connector and a render node of the same card: not devices of their own.
        t.file("sys/class/drm/card1-DP-1/status", "connected\n");
        t.file("sys/class/drm/renderD128/device/vendor", "0x1002\n");
        t.file("sys/class/drm/renderD128/device/device", "0x744c\n");
        // simpledrm's boot framebuffer: no PCI identity.
        t.file("sys/class/drm/card0/device/modalias", "platform:simple-framebuffer\n");
        t.file("etc/os-release", "PRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\nVERSION_ID=\"24.04\"\n");
        t.file("proc/meminfo", "MemTotal:       65536000 kB\nMemAvailable:   60000000 kB\n");
        t.file("usr/lib/x86_64-linux-gnu/libvulkan.so.1", "");
        t.file("opt/rocm/lib/libamdhip64.so.6", "");
        t
    }

    #[test]
    fn drm_cards_become_gpus_and_connectors_are_skipped() {
        let t = rx7900xtx_tree("drm");
        let records = drm_records(&t.0.join("sys/class/drm")).unwrap();
        assert_eq!(records.len(), 1, "{records:?}");
        assert_eq!(records[0].hardware_id, "PCI\\VEN_1002&DEV_744C");
        let gpu = accel::classify_display_adapter(&records[0]).unwrap();
        assert_eq!(gpu.vendor, GpuVendor::Amd);
        assert_eq!(gpu.kind, GpuKind::Discrete);
        assert_eq!(gpu.arch, Detected::Known(GpuArch::Amd { family: AmdArch::Rdna3, gfx: "gfx1100".into() }));
        assert_eq!(gpu.vram_mib, Some(24_560));
    }

    #[test]
    fn the_assembled_profile_reads_os_memory_gpus_and_runtimes() {
        let t = rx7900xtx_tree("assemble");
        let p = assemble(&t.0, Ok(None), |nv, d| {
            assert_eq!(nv, &Ok(None));
            crate::hardware::topology_without_smi(d)
        });
        assert_eq!(p.os.known().unwrap().name, "Ubuntu 24.04.1 LTS");
        assert_eq!(p.memory.known().unwrap().total_mib, 64_000);
        assert_eq!(p.gpus.known().unwrap().len(), 1);
        assert_eq!(p.topology, Detected::Known(GpuTopology::default()));
        // No accel class directory: no NPU driver registered, so no usable NPU.
        assert_eq!(p.npus, Detected::Known(Vec::new()));
        assert_eq!(p.runtimes.vulkan_loader, Detected::Known(true));
        assert_eq!(p.runtimes.hip_runtime, Detected::Known(true));
    }

    #[test]
    fn nvidia_smi_is_authoritative_and_npus_come_from_accel() {
        let t = Tree::new("h100");
        t.file("sys/class/drm/card0/device/vendor", "0x10de\n");
        t.file("sys/class/drm/card0/device/device", "0x2330\n");
        t.file("sys/class/accel/accel0/device/vendor", "0x8086\n");
        t.file("sys/class/accel/accel0/device/device", "0x7d1d\n");
        let smi = crate::hardware::nvidia::parse_device_query(
            "0, NVIDIA H100 80GB HBM3, 00000000:18:00.0, 81559, 9.0, 570.124.06\n",
        )
        .unwrap();
        let p = assemble(&t.0, Ok(Some(smi)), |_, _| Detected::Known(GpuTopology::default()));
        let gpus = p.gpus.known().unwrap();
        assert_eq!(gpus.len(), 1, "the DRM NVIDIA card is the nvidia-smi one: {gpus:?}");
        assert!(gpus[0].compute_capability.is_some());
        assert_eq!(p.npus.known().unwrap()[0].arch, Detected::Known(NpuArch::IntelNpu3));
        assert_eq!(p.runtimes.vulkan_loader, Detected::Known(false));
        assert!(p.os.known().is_none());
    }

    #[test]
    fn two_nvidia_cards_without_nvidia_smi_have_unknown_links() {
        let t = Tree::new("2x3090");
        for (card, dev) in [("card0", "0x2204"), ("card1", "0x2204")] {
            t.file(&format!("sys/class/drm/{card}/device/vendor"), "0x10de\n");
            t.file(&format!("sys/class/drm/{card}/device/device"), &format!("{dev}\n"));
        }
        let p = assemble(&t.0, Ok(None), |_, d| crate::hardware::topology_without_smi(d));
        assert!(p.topology.known().is_none(), "{:?}", p.topology);
        assert_eq!(p.gpus.known().unwrap().len(), 2);
    }
}
