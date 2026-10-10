//! Physical accelerator inventory — what silicon exists in this machine.
//!
//! # This is deliberately not the list of devices you may pass to llama.cpp
//!
//! Cerebro keeps two separate tiers, and conflating them is the specific bug class this module
//! exists to prevent:
//!
//! * **Physical inventory** (this module) — what the machine contains, from the vendor tools and
//!   the OS. On the development machine that includes an Intel Iris Xe iGPU.
//! * **Addressable set** ([`crate::backend`]) — what one particular llama.cpp binary can target,
//!   from that binary's own `--list-devices`. On the development machine that is exactly `CUDA0`,
//!   because the shipping build has no Vulkan backend and therefore cannot see the Iris Xe at all.
//!
//! These sets differ in practice, and an earlier design round assumed they did not — which made a
//! whole iGPU offload plan look viable against hardware the backend is blind to. The two are
//! distinct Rust types so that a device string from this tier cannot be passed where the other is
//! required.
//!
//! # Free VRAM is a measurement precondition, not a static fact
//!
//! Free VRAM on this machine was observed swinging from 431 MiB to 7285 MiB within minutes, purely
//! from other desktop processes opening and closing. A configuration tuned once against a quiet
//! machine will exhaust VRAM the moment the user opens a browser, and on an 8 GB card CUDA responds
//! to exhaustion by silently spilling to host memory over PCIe — throughput collapses by roughly an
//! order of magnitude with no error raised. So [`GpuInventory::sample_free_vram`] is cheap and is
//! meant to be called repeatedly: before a run as a gate, and during a soak to detect a leak.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::error::ProbeError;

/// One physically present NVIDIA device.
///
/// Field units are fixed at construction so no caller has to guess: memory is mebibytes, matching
/// what the vendor tool reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalGpu {
    name: String,
    total_mib: u64,
    free_mib: u64,
    compute_capability: String,
    driver_version: String,
}

impl PhysicalGpu {
    /// Marketing name, e.g. `NVIDIA GeForce RTX 3070 Ti Laptop GPU`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total device memory in MiB.
    pub fn total_mib(&self) -> u64 {
        self.total_mib
    }

    /// Free device memory in MiB *at the moment it was sampled*.
    ///
    /// Treat this as perishable. See the module docs for why it must be re-sampled rather than
    /// cached.
    pub fn free_mib(&self) -> u64 {
        self.free_mib
    }

    /// CUDA compute capability as reported, e.g. `8.6`.
    ///
    /// Kept as a string rather than a parsed pair because it is used for capability gating against
    /// documented thresholds, and inventing a numeric ordering for it invites comparisons that do
    /// not mean what they look like.
    pub fn compute_capability(&self) -> &str {
        &self.compute_capability
    }

    /// Driver version string.
    pub fn driver_version(&self) -> &str {
        &self.driver_version
    }
}

/// One device's perishable state at one instant: what a benchmark keys to and how busy it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSample {
    /// Free device memory in MiB.
    pub free_mib: u64,
    /// GPU utilization in percent over the vendor tool's last sample period; `None` where the
    /// device reports `[N/A]` / `[Not Supported]`. Anything well above zero before a benchmark
    /// starts means another process is using the GPU, so the measurement would be contended.
    pub utilization_pct: Option<u32>,
}

/// Everything physically present, as distinct from anything addressable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuInventory {
    nvidia: Vec<PhysicalGpu>,
    nvidia_smi: Option<PathBuf>,
}

impl GpuInventory {
    /// NVIDIA devices found, in the vendor tool's enumeration order.
    ///
    /// Empty is a normal, successful result: a machine with no NVIDIA hardware, or one where the
    /// vendor tool is not installed, is not an error condition.
    pub fn nvidia(&self) -> &[PhysicalGpu] {
        &self.nvidia
    }

    /// Whether any accelerator was found at all.
    pub fn is_empty(&self) -> bool {
        self.nvidia.is_empty()
    }

    /// Re-sample free VRAM for every known device, cheaply.
    ///
    /// Returns free MiB per device in the same order as [`GpuInventory::nvidia`]. This is the call a
    /// benchmark gate and a soak-test watchdog both use; it deliberately does not rebuild the whole
    /// inventory, because the expensive fields do not change and the cheap one is the entire point.
    pub fn sample_free_vram(&self) -> Result<Vec<u64>, ProbeError> {
        let Some(tool) = self.nvidia_smi.as_ref() else {
            return Ok(Vec::new());
        };
        let out = run_tool(tool, &["--query-gpu=memory.free", "--format=csv,noheader,nounits"])?;
        let mut vals = Vec::new();
        for line in out.lines().map(str::trim).filter(|l| !l.is_empty()) {
            vals.push(parse_u64_field(line, "memory.free")?);
        }
        Ok(vals)
    }

    /// Re-sample free VRAM and utilization for every known device in one vendor-tool call, in the
    /// same order as [`GpuInventory::nvidia`]: the pre-benchmark sample, which keys the run and
    /// records whether the GPU was already busy.
    pub fn sample(&self) -> Result<Vec<GpuSample>, ProbeError> {
        let Some(tool) = self.nvidia_smi.as_ref() else {
            return Ok(Vec::new());
        };
        let out = run_tool(
            tool,
            &["--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        )?;
        parse_sample_csv(&out)
    }
}

/// Parse `--query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits` output. Free
/// memory must be a number; utilization may be `[N/A]`-style text, which becomes `None`.
pub fn parse_sample_csv(text: &str) -> Result<Vec<GpuSample>, ProbeError> {
    let mut samples = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let [free, utilization] = fields.as_slice() else {
            return Err(ProbeError::ToolOutputMalformed {
                tool: "nvidia-smi".into(),
                detail: format!("expected 2 comma-separated fields, got {}", fields.len()),
            });
        };
        let utilization_pct = if utilization.starts_with('[') {
            None
        } else {
            let pct = parse_u64_field(utilization, "utilization.gpu")?;
            let pct = u32::try_from(pct).ok().filter(|p| *p <= 100).ok_or_else(|| {
                ProbeError::ToolOutputMalformed {
                    tool: "nvidia-smi".into(),
                    detail: format!("utilization.gpu was {pct}, above 100 percent"),
                }
            })?;
            Some(pct)
        };
        samples.push(GpuSample {
            free_mib: parse_u64_field(free, "memory.free")?,
            utilization_pct,
        });
    }
    Ok(samples)
}

/// Probe the machine for physically present accelerators.
///
/// Never fails merely because no accelerator is present or no vendor tool is installed — those are
/// legitimate machines and yield an empty inventory. It fails only when a tool *was* found, *did*
/// run, and produced output that cannot be trusted, because silently accepting garbage here would
/// let every downstream VRAM calculation be wrong.
pub fn probe() -> Result<GpuInventory, ProbeError> {
    let Some(tool) = locate_nvidia_smi() else {
        return Ok(GpuInventory::default());
    };
    let out = run_tool(
        &tool,
        &[
            "--query-gpu=name,memory.total,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let nvidia = parse_nvidia_smi_csv(&out)?;
    Ok(GpuInventory {
        nvidia,
        nvidia_smi: Some(tool),
    })
}

/// Locate `nvidia-smi` at a known-good absolute path.
///
/// Deliberately does NOT search `PATH`. Resolving a tool name through `PATH` means whatever
/// directory happens to come first decides which binary Cerebro executes, and Cerebro runs on
/// machines whose `PATH` contains directories writable by other principals — the development
/// machine's own llama.cpp directory is modify-writable by a non-owner SID. The driver installs this
/// tool into the protected system directory, so that is the only place it is honored. That directory
/// comes from the kernel (`GetSystemDirectoryW`), never from `%SystemRoot%`, which the parent
/// process sets and could point at a folder of its choosing.
#[cfg(windows)]
pub(crate) fn locate_nvidia_smi() -> Option<PathBuf> {
    let candidate = crate::child_env::system_directory().ok()?.join("nvidia-smi.exe");
    candidate.is_file().then_some(candidate)
}

/// Linux: the driver packages install `nvidia-smi` into `/usr/bin` (root-owned); that fixed path is
/// the only one honoured, for the same reason `PATH` is not searched on Windows.
#[cfg(target_os = "linux")]
pub(crate) fn locate_nvidia_smi() -> Option<PathBuf> {
    let candidate = Path::new("/usr/bin/nvidia-smi");
    candidate.is_file().then(|| candidate.to_path_buf())
}

/// No NVIDIA driver ships `nvidia-smi` on any other platform Cerebro builds for.
#[cfg(not(any(windows, target_os = "linux")))]
pub(crate) fn locate_nvidia_smi() -> Option<PathBuf> {
    None
}

/// How long one vendor-tool query may take before the tool is killed. `nvidia-smi` answers in tens
/// of milliseconds; it is known to hang during a driver timeout-recovery or when a GPU drops off the
/// bus, and a benchmark samples it outside the run's own timeout, so the hang must be bounded here.
pub const TOOL_TIMEOUT: Duration = Duration::from_secs(10);

/// Run a vendor tool with explicit arguments and capture stdout, within [`TOOL_TIMEOUT`].
///
/// No shell is involved: arguments go to the process directly, so quoting and metacharacters carry
/// no meaning and cannot be used to inject a second command.
pub(crate) fn run_tool(tool: &Path, args: &[&str]) -> Result<String, ProbeError> {
    run_tool_within(tool, args, TOOL_TIMEOUT)
}

/// [`run_tool`] with an explicit deadline. No window, stdin closed, stderr discarded, stdout kept
/// up to the probe's query cap; past `timeout` the tool is killed and reaped and
/// [`ProbeError::ToolTimedOut`] returned.
fn run_tool_within(tool: &Path, args: &[&str], timeout: Duration) -> Result<String, ProbeError> {
    let mut command = Command::new(tool);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // Sampled before and after every benchmark run: from a GUI process without a console, a
    // console tool would otherwise flash a window each time.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let child = command.spawn().map_err(|e| ProbeError::ToolSpawn {
        tool: tool.display().to_string(),
        cause: e.to_string(),
    })?;
    let output = crate::backend::collect_bounded_streams(child, tool, timeout)?;

    if !output.status.success() {
        return Err(ProbeError::ToolFailed {
            tool: tool.display().to_string(),
            status: output.status.code().unwrap_or(-1),
        });
    }

    String::from_utf8(output.stdout).map_err(|_| ProbeError::ToolOutputNotUtf8 {
        tool: tool.display().to_string(),
    })
}

/// Parse `nvidia-smi --format=csv,noheader,nounits` output into devices.
///
/// Split out as a pure function so the parser is covered by fixture tests on machines with no
/// NVIDIA hardware. This matters more than it looks: a parser bug here produces a *plausible* VRAM
/// figure, and every offload plan built on it would be quietly wrong rather than visibly broken.
pub fn parse_nvidia_smi_csv(text: &str) -> Result<Vec<PhysicalGpu>, ProbeError> {
    let mut gpus = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != 5 {
            return Err(ProbeError::ToolOutputMalformed {
                tool: "nvidia-smi".into(),
                detail: format!("expected 5 comma-separated fields, got {}", fields.len()),
            });
        }
        // `[N/A]` and `[Not Supported]` appear in real output on laptop parts for some fields;
        // they are legitimate values to carry through as text but never as a number.
        gpus.push(PhysicalGpu {
            name: fields[0].to_string(),
            total_mib: parse_u64_field(fields[1], "memory.total")?,
            free_mib: parse_u64_field(fields[2], "memory.free")?,
            compute_capability: fields[3].to_string(),
            driver_version: fields[4].to_string(),
        });
    }
    Ok(gpus)
}

fn parse_u64_field(raw: &str, field: &str) -> Result<u64, ProbeError> {
    raw.trim()
        .parse::<u64>()
        .map_err(|_| ProbeError::ToolOutputMalformed {
            tool: "nvidia-smi".into(),
            detail: format!("field {field} was {raw:?}, which is not a non-negative integer"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `nvidia-smi` is only ever looked for in the kernel-reported system directory, so a parent
    /// that sets `%SystemRoot%` elsewhere cannot choose the executable.
    #[cfg(windows)]
    #[test]
    fn nvidia_smi_is_located_only_in_the_kernel_system_directory() {
        let system = crate::child_env::system_directory().unwrap();
        match locate_nvidia_smi() {
            Some(found) => assert_eq!(found, system.join("nvidia-smi.exe")),
            None => assert!(!system.join("nvidia-smi.exe").is_file()),
        }
    }

    /// Real captured output from the development machine.
    const REAL: &str =
        "NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192, 7498, 8.6, 610.74\n";

    #[test]
    fn parses_real_captured_output() {
        let gpus = parse_nvidia_smi_csv(REAL).unwrap();
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name(), "NVIDIA GeForce RTX 3070 Ti Laptop GPU");
        assert_eq!(gpus[0].total_mib(), 8192);
        assert_eq!(gpus[0].free_mib(), 7498);
        assert_eq!(gpus[0].compute_capability(), "8.6");
        assert_eq!(gpus[0].driver_version(), "610.74");
    }

    #[test]
    fn parses_multiple_devices_preserving_order() {
        let two = "GPU A, 8192, 100, 8.6, 610.74\nGPU B, 24576, 24000, 8.9, 610.74\n";
        let gpus = parse_nvidia_smi_csv(two).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name(), "GPU A");
        assert_eq!(gpus[1].total_mib(), 24576);
    }

    #[test]
    fn empty_output_is_a_successful_empty_inventory() {
        // A machine with the tool installed but no supported device. Not an error.
        assert!(parse_nvidia_smi_csv("").unwrap().is_empty());
        assert!(parse_nvidia_smi_csv("\n  \n").unwrap().is_empty());
    }

    #[test]
    fn wrong_field_count_is_rejected_rather_than_guessed() {
        // Vendor tools change their output. Guessing which field is which would produce a
        // plausible-looking VRAM number that is silently wrong.
        let e = parse_nvidia_smi_csv("only, three, fields").unwrap_err();
        assert!(matches!(e, ProbeError::ToolOutputMalformed { .. }));
    }

    #[test]
    fn non_numeric_memory_is_rejected() {
        // `[N/A]` is real nvidia-smi output for unsupported queries on laptop parts. Accepting it
        // as zero would make an 8 GB card look like it had no memory at all.
        let e = parse_nvidia_smi_csv("GPU, [N/A], 7498, 8.6, 610.74").unwrap_err();
        assert!(matches!(e, ProbeError::ToolOutputMalformed { .. }));
    }

    #[test]
    fn inventory_without_a_tool_samples_nothing_rather_than_failing() {
        let empty = GpuInventory::default();
        assert!(empty.is_empty());
        assert!(empty.sample_free_vram().unwrap().is_empty());
    }

    #[test]
    fn a_sample_parses_free_memory_and_optional_utilization() {
        let samples = parse_sample_csv("7285, 3\n431, [N/A]\n").unwrap();
        assert_eq!(
            samples,
            [
                GpuSample { free_mib: 7285, utilization_pct: Some(3) },
                GpuSample { free_mib: 431, utilization_pct: None },
            ]
        );
        assert!(parse_sample_csv("").unwrap().is_empty());
        for bad in ["7285", "[N/A], 3", "7285, 101", "7285, 3, 9"] {
            assert!(
                matches!(parse_sample_csv(bad), Err(ProbeError::ToolOutputMalformed { .. })),
                "{bad:?}"
            );
        }
    }

    /// A tool that hangs is killed at the deadline, reported as a timeout, and not left running.
    /// `ping -n 30` to loopback stands in for a hung `nvidia-smi`: always in System32, windowless
    /// with `CREATE_NO_WINDOW`, and busy for about 29 seconds unless killed.
    #[cfg(windows)]
    #[test]
    fn a_hung_tool_is_killed_at_the_deadline() {
        let ping = crate::child_env::system_directory().unwrap().join("PING.EXE");
        let timeout = Duration::from_millis(300);
        let started = std::time::Instant::now();
        let err = run_tool_within(&ping, &["-n", "30", "127.0.0.1"], timeout).unwrap_err();
        let elapsed = started.elapsed();
        assert!(
            matches!(&err, ProbeError::ToolTimedOut { after, .. } if *after == timeout),
            "{err:?}"
        );
        assert!(elapsed >= timeout && elapsed < Duration::from_secs(5), "{elapsed:?}");
    }

    #[cfg(windows)]
    #[test]
    fn a_tool_that_fails_is_reported_with_its_status() {
        let ping = crate::child_env::system_directory().unwrap().join("PING.EXE");
        // `-n` without its value: ping prints usage and exits non-zero at once.
        let err = run_tool_within(&ping, &["-n"], Duration::from_secs(10)).unwrap_err();
        assert!(matches!(err, ProbeError::ToolFailed { .. }), "{err:?}");
    }

    #[test]
    #[ignore = "requires the development machine's NVIDIA driver and hardware"]
    fn live_probe_finds_the_development_gpu() {
        let inv = probe().unwrap();
        assert_eq!(inv.nvidia().len(), 1);
        assert_eq!(inv.nvidia()[0].total_mib(), 8192);
        // Free VRAM is perishable; assert only that a re-sample is shaped correctly.
        assert_eq!(inv.sample_free_vram().unwrap().len(), 1);
        assert_eq!(inv.sample().unwrap().len(), 1);
    }
}
