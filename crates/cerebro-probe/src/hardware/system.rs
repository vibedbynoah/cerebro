//! Operating system, CPU identity, memory, and the runtime facts (WSL, Vulkan loader, HIP runtime)
//! that decide which inference backends can run at all.

use super::isa::IsaSet;
use super::Detected;

/// Operating system family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OsFamily {
    /// Windows 10 or 11.
    Windows,
    /// Linux (native; WSL is reported separately on Windows).
    Linux,
    /// macOS.
    MacOs,
}

impl OsFamily {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Windows => "Windows",
            Self::Linux => "Linux",
            Self::MacOs => "macOS",
        }
    }
}

/// The running operating system.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OsInfo {
    /// Family.
    pub family: OsFamily,
    /// Human-readable name, e.g. `Windows 11 25H2` or `Ubuntu 24.04.1 LTS`.
    pub name: String,
    /// Version string, e.g. `10.0.26200.6584`, when established.
    pub version: Option<String>,
    /// Windows build number (`26200`), when on Windows.
    pub build: Option<u32>,
}

/// Instruction-set architecture of the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CpuArch {
    /// x86-64 (Intel, AMD).
    X86_64,
    /// AArch64 (Snapdragon X, Apple Silicon, Ampere, Graviton).
    Aarch64,
}

impl CpuArch {
    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86-64",
            Self::Aarch64 => "Arm64",
        }
    }

    /// The architecture this process was compiled for, if it is one Cerebro models.
    pub fn of_this_build() -> Option<Self> {
        if cfg!(target_arch = "x86_64") {
            Some(Self::X86_64)
        } else if cfg!(target_arch = "aarch64") {
            Some(Self::Aarch64)
        } else {
            None
        }
    }
}

/// The CPU.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CpuInfo {
    /// Native architecture of the machine (not of this process: an x64 build under emulation on
    /// Snapdragon reports `Aarch64` here).
    pub arch: CpuArch,
    /// Brand string, e.g. `12th Gen Intel(R) Core(TM) i9-12900H`.
    pub brand: Option<String>,
    /// Usable instruction-set extensions. `Unknown` under emulation (CPUID then describes the
    /// emulator, not the silicon) and where no detector exists.
    pub isa: Detected<IsaSet>,
    /// Physical cores, when the topology probe succeeded.
    pub physical_cores: Option<u32>,
    /// Logical processors, when the topology probe succeeded.
    pub logical_processors: Option<u32>,
}

/// System memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryInfo {
    /// Installed physical memory visible to the OS, in MiB.
    pub total_mib: u64,
    /// Available at sampling time, in MiB. Perishable.
    pub available_mib: u64,
}

/// One installed WSL distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WslDistro {
    /// Distribution name, e.g. `Ubuntu-24.04`.
    pub name: String,
    /// WSL version (1 or 2), when recorded.
    pub version: Option<u32>,
}

/// Windows Subsystem for Linux state.
///
/// Three separate facts: WSL is installed, a distribution is registered, and a WSL 2 distribution
/// can start. The last needs the Windows hypervisor (Virtual Machine Platform plus firmware
/// virtualization), which a registered distribution does not prove.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WslInfo {
    /// Whether WSL itself is installed: the Store package (`%ProgramFiles%\WSL\wsl.exe`) or the
    /// inbox optional component (`System32\lxss\LxssManager.dll`). `System32\wsl.exe` does not
    /// count: current Windows ships it as a stub that offers to install WSL.
    pub installed: bool,
    /// Registered distributions for this user.
    pub distributions: Vec<WslDistro>,
    /// Whether a WSL 2 distribution can start. `Known(false)` when no hypervisor is running;
    /// otherwise `Unknown`, because a running hypervisor (Hyper-V, VBS) does not prove the Virtual
    /// Machine Platform component is enabled.
    pub wsl2_can_start: Detected<bool>,
}

impl Default for WslInfo {
    /// Not installed, nothing registered, start capability not checked.
    fn default() -> Self {
        Self {
            installed: false,
            distributions: Vec::new(),
            wsl2_can_start: Detected::unknown("not probed"),
        }
    }
}

/// Whether a WSL 2 distribution can start, from the CPUID hypervisor-present bit
/// (`CPUID.1:ECX[31]`; `None` where there is no CPUID). WSL 2 runs on the Windows hypervisor, so
/// with no hypervisor running it cannot start; a running one is necessary, not sufficient.
pub fn wsl2_start_fact(hypervisor_present: Option<bool>) -> Detected<bool> {
    match hypervisor_present {
        Some(false) => Detected::Known(false),
        Some(true) => Detected::unknown(
            "a hypervisor is running, but whether the Virtual Machine Platform component is \
             enabled is not checked",
        ),
        None => Detected::unknown("whether the Windows hypervisor is running is not checked on this CPU"),
    }
}

impl WslInfo {
    /// WSL 2 distributions a Linux-only backend can run in: Docker Desktop's internal
    /// distributions are excluded, because they are not general-purpose Linux environments.
    pub fn usable_wsl2(&self) -> impl Iterator<Item = &WslDistro> {
        self.distributions.iter().filter(|d| {
            d.version == Some(2) && !d.name.to_ascii_lowercase().starts_with("docker-desktop")
        })
    }
}

/// Software facts that gate backends independently of the silicon.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeFacts {
    /// WSL (Windows only; `Unknown` elsewhere).
    pub wsl: Detected<WslInfo>,
    /// Whether the Vulkan loader (`vulkan-1.dll` / `libvulkan.so.1`) is installed.
    pub vulkan_loader: Detected<bool>,
    /// Whether AMD's HIP runtime (`amdhip64*.dll`, shipped with the Adrenalin driver) is installed.
    pub hip_runtime: Detected<bool>,
}

impl Default for RuntimeFacts {
    fn default() -> Self {
        Self {
            wsl: Detected::unknown("not probed"),
            vulkan_loader: Detected::unknown("not probed"),
            hip_runtime: Detected::unknown("not probed"),
        }
    }
}

/// Parse `/etc/os-release` into an [`OsInfo`]. `None` when it has neither `PRETTY_NAME` nor `NAME`.
pub fn parse_os_release(text: &str) -> Option<OsInfo> {
    let mut pretty = None;
    let mut name = None;
    let mut version = None;
    for line in text.lines() {
        let Some((key, value)) = line.trim().split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'').to_string();
        match key {
            "PRETTY_NAME" => pretty = Some(value),
            "NAME" => name = Some(value),
            "VERSION_ID" => version = Some(value),
            _ => {}
        }
    }
    Some(OsInfo {
        family: OsFamily::Linux,
        name: pretty.or(name).filter(|n| !n.is_empty())?,
        version,
        build: None,
    })
}

/// Parse `/proc/meminfo`'s `MemTotal` and `MemAvailable` (kB). `None` if either is missing.
pub fn parse_meminfo(text: &str) -> Option<MemoryInfo> {
    let field = |key: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(key))?;
        let rest = line.get(key.len()..)?.trim().trim_end_matches("kB").trim();
        rest.parse::<u64>().ok()
    };
    Some(MemoryInfo {
        total_mib: field("MemTotal:")? / 1024,
        available_mib: field("MemAvailable:")? / 1024,
    })
}

/// Windows' name for a build: Windows 11 starts at build 22000 (Microsoft's own boundary; both
/// report major version 10).
pub fn windows_name(build: Option<u32>, display_version: Option<&str>) -> String {
    let base = match build {
        Some(b) if b >= 22000 => "Windows 11",
        Some(_) => "Windows 10",
        None => "Windows",
    };
    match display_version {
        Some(v) if !v.is_empty() => format!("{base} {v}"),
        _ => base.to_string(),
    }
}

#[cfg(windows)]
pub(crate) mod live {
    //! Windows detectors. Each returns `Detected::Unknown` with the reason instead of failing.

    use super::super::registry::{Key, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    use super::*;

    /// OS identity from the `CurrentVersion` key.
    pub(crate) fn os() -> Detected<OsInfo> {
        let key = match Key::open(HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
        {
            Ok(k) => k,
            Err(code) => return Detected::unknown(format!("CurrentVersion key unreadable ({code})")),
        };
        let build = key
            .string("CurrentBuildNumber")
            .ok()
            .flatten()
            .and_then(|b| b.parse::<u32>().ok());
        let major = key.integer("CurrentMajorVersionNumber").ok().flatten();
        let minor = key.integer("CurrentMinorVersionNumber").ok().flatten();
        let ubr = key.integer("UBR").ok().flatten();
        let display = key.string("DisplayVersion").ok().flatten();
        let version = match (major, minor, build) {
            (Some(ma), Some(mi), Some(b)) => Some(match ubr {
                Some(u) => format!("{ma}.{mi}.{b}.{u}"),
                None => format!("{ma}.{mi}.{b}"),
            }),
            _ => None,
        };
        Detected::Known(OsInfo {
            family: OsFamily::Windows,
            name: windows_name(build, display.as_deref()),
            version,
            build,
        })
    }

    /// The machine's native architecture, which differs from this process's under emulation.
    pub(crate) fn native_arch() -> Option<CpuArch> {
        use windows_sys::Win32::System::SystemInformation::{
            IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_ARM64,
        };
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, IsWow64Process2};
        let mut process: u16 = 0;
        let mut native: u16 = 0;
        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that needs no closing; both out
        // pointers are live locals the call writes one `u16` into each.
        let ok = unsafe { IsWow64Process2(GetCurrentProcess(), &mut process, &mut native) };
        if ok == 0 {
            return None;
        }
        match native {
            IMAGE_FILE_MACHINE_AMD64 => Some(CpuArch::X86_64),
            IMAGE_FILE_MACHINE_ARM64 => Some(CpuArch::Aarch64),
            _ => None,
        }
    }

    /// Physical memory from `GlobalMemoryStatusEx`.
    pub(crate) fn memory() -> Detected<MemoryInfo> {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: MEMORYSTATUSEX is plain old data; all-zero is a valid value to overwrite.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `status` is a live, correctly sized struct with `dwLength` set as the API
        // requires; the call writes only within it.
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok == 0 || status.ullTotalPhys == 0 {
            return Detected::unknown("GlobalMemoryStatusEx failed");
        }
        Detected::Known(MemoryInfo {
            total_mib: status.ullTotalPhys / (1024 * 1024),
            available_mib: status.ullAvailPhys / (1024 * 1024),
        })
    }

    /// `%ProgramFiles%` from the registry (`ProgramFilesDir`), never from the environment, which
    /// the parent process sets.
    fn program_files_dir() -> Option<std::path::PathBuf> {
        Key::open(HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion")
            .ok()?
            .string("ProgramFilesDir")
            .ok()
            .flatten()
            .map(std::path::PathBuf::from)
    }

    /// The CPUID hypervisor-present bit, on x86-64.
    fn hypervisor_present() -> Option<bool> {
        #[cfg(target_arch = "x86_64")]
        {
            Some(super::super::isa::read_cpuid().leaf1_ecx & (1 << 31) != 0)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }

    /// WSL install, distributions (from this user's `Lxss` key) and whether WSL 2 can start;
    /// read-only, launches nothing.
    pub(crate) fn wsl() -> Detected<WslInfo> {
        let system = match crate::child_env::system_directory() {
            Ok(dir) => dir,
            Err(e) => return Detected::unknown(format!("system directory unknown: {e}")),
        };
        let store = program_files_dir().is_some_and(|p| p.join("WSL").join("wsl.exe").is_file());
        let inbox = system.join("lxss").join("LxssManager.dll").is_file();
        let mut info = WslInfo {
            installed: store || inbox,
            distributions: Vec::new(),
            wsl2_can_start: wsl2_start_fact(hypervisor_present()),
        };
        let lxss = match Key::open(HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Lxss") {
            Ok(k) => k,
            // No key: WSL has never registered a distribution for this user.
            Err(2) => return Detected::Known(info),
            Err(code) => return Detected::unknown(format!("Lxss key unreadable ({code})")),
        };
        let names = match lxss.subkey_names() {
            Ok(n) => n,
            Err(code) => return Detected::unknown(format!("Lxss key unreadable ({code})")),
        };
        for sub in names {
            let Ok(key) = lxss.subkey(&sub) else { continue };
            let Some(name) = key.string("DistributionName").ok().flatten() else {
                continue;
            };
            let version = key
                .integer("Version")
                .ok()
                .flatten()
                .and_then(|v| u32::try_from(v).ok());
            info.distributions.push(WslDistro { name, version });
        }
        Detected::Known(info)
    }

    /// Whether the Vulkan loader is in the system directory.
    pub(crate) fn vulkan_loader() -> Detected<bool> {
        match crate::child_env::system_directory() {
            Ok(dir) => Detected::Known(dir.join("vulkan-1.dll").is_file()),
            Err(e) => Detected::unknown(format!("system directory unknown: {e}")),
        }
    }

    /// Whether any `amdhip64*.dll` is in the system directory (the name carries the HIP major,
    /// `amdhip64_6.dll`, so the directory is scanned by prefix rather than for one fixed name).
    pub(crate) fn hip_runtime() -> Detected<bool> {
        let dir = match crate::child_env::system_directory() {
            Ok(d) => d,
            Err(e) => return Detected::unknown(format!("system directory unknown: {e}")),
        };
        match std::fs::read_dir(&dir) {
            Ok(entries) => Detected::Known(entries.flatten().any(|e| {
                let name = e.file_name().to_string_lossy().to_ascii_lowercase();
                name.starts_with("amdhip64") && name.ends_with(".dll")
            })),
            Err(e) => Detected::unknown(format!("system directory unreadable: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_release_prefers_pretty_name() {
        let ubuntu = "PRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\nNAME=\"Ubuntu\"\nVERSION_ID=\"24.04\"\nID=ubuntu\n";
        let os = parse_os_release(ubuntu).unwrap();
        assert_eq!(os.family, OsFamily::Linux);
        assert_eq!(os.name, "Ubuntu 24.04.1 LTS");
        assert_eq!(os.version.as_deref(), Some("24.04"));
        let minimal = "NAME=Alpine\n";
        assert_eq!(parse_os_release(minimal).unwrap().name, "Alpine");
        assert!(parse_os_release("ID=x\n").is_none());
        assert!(parse_os_release("").is_none());
    }

    #[test]
    fn meminfo_needs_both_fields() {
        let text = "MemTotal:       2113323008 kB\nMemFree:        100 kB\nMemAvailable:   2013323008 kB\n";
        let mem = parse_meminfo(text).unwrap();
        assert_eq!(mem.total_mib, 2_063_792);
        assert_eq!(mem.available_mib, 1_966_135);
        assert!(parse_meminfo("MemTotal: 1024 kB\n").is_none());
        assert!(parse_meminfo("MemTotal: lots kB\nMemAvailable: 1 kB\n").is_none());
    }

    #[test]
    fn windows_11_starts_at_build_22000() {
        assert_eq!(windows_name(Some(26200), Some("25H2")), "Windows 11 25H2");
        assert_eq!(windows_name(Some(22000), None), "Windows 11");
        assert_eq!(windows_name(Some(19045), Some("22H2")), "Windows 10 22H2");
        assert_eq!(windows_name(None, None), "Windows");
    }

    #[test]
    fn docker_desktop_and_wsl1_distributions_are_not_usable() {
        let info = WslInfo {
            installed: true,
            wsl2_can_start: Detected::Known(true),
            distributions: vec![
                WslDistro { name: "docker-desktop".into(), version: Some(2) },
                WslDistro { name: "docker-desktop-data".into(), version: Some(2) },
                WslDistro { name: "Legacy".into(), version: Some(1) },
                WslDistro { name: "Ubuntu-24.04".into(), version: Some(2) },
            ],
        };
        let usable: Vec<&str> = info.usable_wsl2().map(|d| d.name.as_str()).collect();
        assert_eq!(usable, ["Ubuntu-24.04"]);
    }

    #[test]
    fn wsl2_start_is_ruled_out_only_without_a_hypervisor() {
        assert_eq!(wsl2_start_fact(Some(false)), Detected::Known(false));
        assert!(wsl2_start_fact(Some(true)).known().is_none());
        assert!(wsl2_start_fact(None).known().is_none());
    }

    /// The development machine has WSL from the Store (`Program Files\WSL`) and Hyper-V running,
    /// so WSL is installed and whether WSL 2 can start is left unknown, never assumed.
    #[cfg(windows)]
    #[test]
    #[ignore = "requires the development machine (Store WSL installed, hypervisor running)"]
    fn live_wsl_is_detected_from_the_real_install_not_the_inbox_stub() {
        let wsl = live::wsl();
        let wsl = wsl.known().unwrap();
        assert!(wsl.installed);
        assert!(wsl.wsl2_can_start.known().is_none(), "{:?}", wsl.wsl2_can_start);
    }

    #[cfg(windows)]
    #[test]
    fn live_windows_detectors_answer_without_failing() {
        let os = live::os();
        let os = os.known().unwrap();
        assert_eq!(os.family, OsFamily::Windows);
        assert!(os.build.is_some());
        assert!(live::native_arch().is_some());
        assert!(live::memory().known().unwrap().total_mib > 1024);
        assert!(live::wsl().known().is_some());
        assert!(live::vulkan_loader().known().is_some());
        assert!(live::hip_runtime().known().is_some());
    }
}
