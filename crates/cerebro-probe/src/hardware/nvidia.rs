//! NVIDIA specifics: compute capability, architecture, and multi-GPU topology from `nvidia-smi`.
//!
//! Compute capability is the one authoritative architecture signal NVIDIA exposes, so it is the
//! only one used: architecture comes from an explicit `(major, minor)` table and a capability not
//! in the table (a GPU newer than this build) is `Unknown`, never rounded to its nearest
//! neighbour. A Blackwell part reported as `12.0` and a future `13.0` part must not both read as
//! "Blackwell or newer" until someone has checked what `13.0` actually supports.

use std::fmt;

use super::accel::{GpuArch, GpuDevice, GpuKind, GpuVendor, NvidiaArch};
use super::Detected;
use crate::error::ProbeError;

/// A CUDA compute capability, e.g. `8.6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComputeCapability {
    /// Major version.
    pub major: u8,
    /// Minor version.
    pub minor: u8,
}

impl ComputeCapability {
    /// Parse `nvidia-smi`'s `compute_cap` text (`8.6`, `12.0`). `None` for `[N/A]` and anything
    /// that is not exactly `major.minor`.
    pub fn parse(text: &str) -> Option<Self> {
        let (major, minor) = text.trim().split_once('.')?;
        Some(Self {
            major: major.parse().ok()?,
            minor: minor.parse().ok()?,
        })
    }

    /// The architecture this capability belongs to, or `None` when it is not in the table.
    pub fn arch(self) -> Option<NvidiaArch> {
        let arch = match (self.major, self.minor) {
            (5, 0 | 2 | 3) => NvidiaArch::Maxwell,
            (6, 0..=2) => NvidiaArch::Pascal,
            (7, 0 | 2) => NvidiaArch::Volta,
            (7, 5) => NvidiaArch::Turing,
            (8, 0 | 6 | 7) => NvidiaArch::Ampere,
            (8, 9) => NvidiaArch::Ada,
            (9, 0) => NvidiaArch::Hopper,
            (10, 0 | 1 | 3) | (11, 0) => NvidiaArch::BlackwellDatacenter,
            (12, 0 | 1) => NvidiaArch::BlackwellClient,
            _ => return None,
        };
        Some(arch)
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// The `--query-gpu` field list [`parse_device_query`] expects, in order.
pub const DEVICE_QUERY: &str =
    "--query-gpu=index,name,pci.bus_id,memory.total,compute_cap,driver_version";

fn malformed(detail: String) -> ProbeError {
    ProbeError::ToolOutputMalformed {
        tool: "nvidia-smi".into(),
        detail,
    }
}

/// Parse `nvidia-smi` [`DEVICE_QUERY`] `--format=csv,noheader,nounits` output into GPUs.
///
/// A wrong field count or a non-numeric index or memory size is an error (the format changed and
/// every field may be shifted). A `[N/A]` compute capability is carried as an unknown
/// architecture, not an error: older drivers report it that way and the rest of the row is good.
pub fn parse_device_query(text: &str) -> Result<Vec<GpuDevice>, ProbeError> {
    let mut gpus = Vec::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let [index, name, _bus_id, total, cc, driver] = fields.as_slice() else {
            return Err(malformed(format!(
                "expected 6 comma-separated fields, got {}",
                fields.len()
            )));
        };
        let index = index
            .parse::<u32>()
            .map_err(|_| malformed(format!("index {index:?} is not a number")))?;
        // `[N/A]` / `[Not Supported]`: unified-memory parts (DGX Spark) report no dedicated pool.
        let total_mib = if total.starts_with('[') {
            None
        } else {
            Some(total.parse::<u64>().map_err(|_| {
                malformed(format!("memory.total {total:?} is not a number"))
            })?)
        };
        let compute_capability = ComputeCapability::parse(cc);
        let arch = match compute_capability {
            Some(cc) => match cc.arch() {
                Some(arch) => Detected::Known(GpuArch::Nvidia(arch)),
                None => Detected::unknown(format!(
                    "compute capability {cc} is newer than Cerebro's table"
                )),
            },
            None => Detected::unknown(format!("nvidia-smi reported compute_cap as {cc:?}")),
        };
        gpus.push(GpuDevice {
            vendor: GpuVendor::Nvidia,
            name: (*name).to_string(),
            // A device with its own reported memory pool is discrete; one without (a unified-memory
            // SoC) is not established either way.
            kind: if total_mib.is_some() { GpuKind::Discrete } else { GpuKind::Unknown },
            arch,
            // `pci.bus_id` is a bus location, not a vendor/device ID.
            pci: None,
            vram_mib: total_mib,
            driver_version: Some((*driver).to_string()).filter(|d| !d.starts_with('[')),
            compute_capability,
            nvidia_index: Some(index),
        });
    }
    Ok(gpus)
}

/// The NVIDIA driver's major version (`616` from `616.92`), used to gate CUDA runtime majors.
pub fn driver_major(version: &str) -> Option<u32> {
    version.trim().split('.').next()?.parse().ok()
}

/// How two GPUs are connected, from the `nvidia-smi topo -m` legend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinkKind {
    /// `NV#`: a bonded set of this many NVLinks.
    NvLink(u8),
    /// `PIX`: at most one PCIe bridge.
    PcieSingleBridge,
    /// `PXB`: several PCIe bridges, no host bridge.
    PcieMultiBridge,
    /// `PHB`: through a PCIe host bridge (typically the CPU).
    PcieHostBridge,
    /// `NODE`: across host bridges within one NUMA node.
    NumaNode,
    /// `SYS`: across the SMP interconnect between NUMA nodes.
    System,
}

impl LinkKind {
    fn parse(cell: &str) -> Option<Self> {
        let cell = cell.trim();
        Some(match cell {
            "PIX" => Self::PcieSingleBridge,
            "PXB" => Self::PcieMultiBridge,
            "PHB" => Self::PcieHostBridge,
            "NODE" => Self::NumaNode,
            "SYS" => Self::System,
            _ => Self::NvLink(cell.strip_prefix("NV")?.parse().ok().filter(|n| *n > 0)?),
        })
    }
}

/// One GPU-to-GPU connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuLink {
    /// `nvidia-smi` index of the first GPU (the smaller one).
    pub a: u32,
    /// `nvidia-smi` index of the second GPU.
    pub b: u32,
    /// How they are connected.
    pub kind: LinkKind,
}

/// The GPU-to-GPU connection matrix. Empty for zero or one GPU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GpuTopology {
    /// One entry per unordered GPU pair.
    pub links: Vec<GpuLink>,
}

impl GpuTopology {
    /// The connection between two GPUs, in either order.
    pub fn link(&self, a: u32, b: u32) -> Option<LinkKind> {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        self.links
            .iter()
            .find(|l| l.a == a && l.b == b)
            .map(|l| l.kind)
    }

    /// Whether there is at least one pair and every pair is NVLink-connected: the condition under
    /// which tensor parallelism avoids PCIe for its all-reduce.
    pub fn all_pairs_nvlink(&self) -> bool {
        !self.links.is_empty()
            && self
                .links
                .iter()
                .all(|l| matches!(l.kind, LinkKind::NvLink(_)))
    }
}

/// Remove ANSI escape sequences: `nvidia-smi topo -m` underlines its header row even when stdout
/// is a pipe (captured on the development machine).
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                // CSI: parameters and intermediates until a final byte in '@'..='~'.
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Parse `nvidia-smi topo -m` output. Rows and columns named `GPU<n>` form the matrix; `NIC<n>`
/// columns, affinity columns and the legend are ignored. An unrecognised cell, a GPU row with too
/// few cells, or a matrix that is not symmetric is an error rather than a partial topology.
pub fn parse_topology(text: &str) -> Result<GpuTopology, ProbeError> {
    let text = strip_ansi(text);
    let mut lines = text.lines();
    let header = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| malformed("topo -m output is empty".into()))?;
    let columns: Vec<&str> = header
        .split('\t')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();
    // The matrix columns are the leading GPU/NIC columns; affinity columns follow.
    let matrix_cols: Vec<&str> = columns
        .iter()
        .copied()
        .take_while(|c| gpu_ordinal(c).is_some() || c.starts_with("NIC"))
        .collect();

    let mut cells: Vec<(u32, u32, LinkKind)> = Vec::new();
    let mut rows = 0usize;
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        let Some(label) = fields.first().map(|f| f.trim()) else {
            continue;
        };
        let Some(row) = gpu_ordinal(label) else {
            continue;
        };
        rows += 1;
        for (j, col) in matrix_cols.iter().enumerate() {
            let Some(other) = gpu_ordinal(col) else {
                continue;
            };
            let cell = fields
                .get(j + 1)
                .map(|c| c.trim())
                .ok_or_else(|| malformed(format!("row {label} has no cell for {col}")))?;
            if other == row {
                if cell != "X" {
                    return Err(malformed(format!("diagonal cell {label}/{col} is {cell:?}")));
                }
                continue;
            }
            let kind = LinkKind::parse(cell)
                .ok_or_else(|| malformed(format!("unrecognised link {cell:?} at {label}/{col}")))?;
            cells.push((row, other, kind));
        }
    }
    let gpu_cols = matrix_cols.iter().filter(|c| gpu_ordinal(c).is_some()).count();
    if rows != gpu_cols {
        return Err(malformed(format!(
            "{gpu_cols} GPU columns but {rows} GPU rows"
        )));
    }

    let mut links: Vec<GpuLink> = Vec::new();
    for &(a, b, kind) in &cells {
        let mirrored = cells
            .iter()
            .find(|&&(x, y, _)| x == b && y == a)
            .map(|&(_, _, k)| k);
        if mirrored != Some(kind) {
            return Err(malformed(format!("GPU{a}/GPU{b} link is not symmetric")));
        }
        if a < b {
            links.push(GpuLink { a, b, kind });
        }
    }
    Ok(GpuTopology { links })
}

fn gpu_ordinal(label: &str) -> Option<u32> {
    label.trim().strip_prefix("GPU")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_capability_maps_through_an_explicit_table() {
        let arch = |s: &str| ComputeCapability::parse(s).and_then(ComputeCapability::arch);
        assert_eq!(arch("8.6"), Some(NvidiaArch::Ampere));
        assert_eq!(arch("8.9"), Some(NvidiaArch::Ada));
        assert_eq!(arch("9.0"), Some(NvidiaArch::Hopper));
        assert_eq!(arch("10.0"), Some(NvidiaArch::BlackwellDatacenter));
        assert_eq!(arch("12.0"), Some(NvidiaArch::BlackwellClient));
        assert_eq!(arch("7.5"), Some(NvidiaArch::Turing));
        assert_eq!(arch("6.1"), Some(NvidiaArch::Pascal));
        // Not in the table: unknown, not the nearest neighbour.
        assert_eq!(arch("8.8"), None);
        assert_eq!(arch("13.0"), None);
        assert_eq!(ComputeCapability::parse("[N/A]"), None);
        assert_eq!(ComputeCapability::parse("8"), None);
        assert_eq!(ComputeCapability::parse("8.6.1"), None);
    }

    #[test]
    fn the_device_query_parses_real_and_recorded_output() {
        // Captured on the development machine, 2026-10-01.
        let gpus = parse_device_query(
            "0, NVIDIA GeForce RTX 3070 Ti Laptop GPU, 00000000:01:00.0, 8192, 8.6, 616.92\n",
        )
        .unwrap();
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].arch, Detected::Known(GpuArch::Nvidia(NvidiaArch::Ampere)));
        assert_eq!(gpus[0].vram_mib, Some(8192));
        assert_eq!(gpus[0].nvidia_index, Some(0));
        assert_eq!(gpus[0].driver_version.as_deref(), Some("616.92"));
        assert_eq!(gpus[0].kind, GpuKind::Discrete);

        let old = parse_device_query("0, Tesla K80, 00000000:04:00.0, 11441, [N/A], 470.82\n").unwrap();
        assert!(old[0].arch.known().is_none());
        assert_eq!(old[0].compute_capability, None);

        let future = parse_device_query("0, NVIDIA Next, 00000000:01:00.0, 98304, 13.0, 700.10\n").unwrap();
        assert!(future[0].arch.known().is_none());
        assert_eq!(future[0].compute_capability, Some(ComputeCapability { major: 13, minor: 0 }));

        for bad in [
            "0, GPU, 00000000:01:00.0, 8192, 8.6",
            "x, GPU, 00000000:01:00.0, 8192, 8.6, 616.92",
            "0, GPU, 00000000:01:00.0, 8 GB, 8.6, 616.92",
        ] {
            assert!(parse_device_query(bad).is_err(), "{bad:?}");
        }
        assert!(parse_device_query("\n").unwrap().is_empty());

        // A unified-memory part reports no dedicated pool: memory and kind are unknown.
        let spark =
            parse_device_query("0, NVIDIA GB10, 0000000F:01:00.0, [N/A], 12.1, 580.95.05\n").unwrap();
        assert_eq!(spark[0].vram_mib, None);
        assert_eq!(spark[0].kind, GpuKind::Unknown);
        assert_eq!(spark[0].arch, Detected::Known(GpuArch::Nvidia(NvidiaArch::BlackwellClient)));
    }

    #[test]
    fn driver_major_reads_the_leading_component() {
        assert_eq!(driver_major("616.92"), Some(616));
        assert_eq!(driver_major("580.65.06"), Some(580));
        assert_eq!(driver_major("[N/A]"), None);
    }

    /// Captured on the development machine (single GPU, with the underline escape it really
    /// emits into a pipe).
    const TOPO_SINGLE: &str = "\t\u{1b}[4mGPU0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\u{1b}[0m\nGPU0\t X \t\t\t\tN/A\n\nLegend:\n\n  X    = Self\n  NV#  = Connection traversing a bonded set of # NVLinks\n";

    #[test]
    fn a_single_gpu_topology_has_no_links() {
        let topo = parse_topology(TOPO_SINGLE).unwrap();
        assert!(topo.links.is_empty());
        assert!(!topo.all_pairs_nvlink());
    }

    #[test]
    fn nvlink_and_pcie_matrices_parse_with_nic_columns_ignored() {
        let nvlink = "\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\nGPU0\t X \tNV4\t0-23\t0\t\tN/A\nGPU1\tNV4\t X \t0-23\t0\t\tN/A\n";
        let topo = parse_topology(nvlink).unwrap();
        assert_eq!(topo.links, vec![GpuLink { a: 0, b: 1, kind: LinkKind::NvLink(4) }]);
        assert!(topo.all_pairs_nvlink());
        assert_eq!(topo.link(1, 0), Some(LinkKind::NvLink(4)));

        let mixed = "\tGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\nGPU0\t X \tNV2\tSYS\tPXB\t0-15\nGPU1\tNV2\t X \tSYS\tPXB\t0-15\nGPU2\tSYS\tSYS\t X \tSYS\t16-31\nNIC0\tPXB\tPXB\tSYS\t X \t\n\nNIC Legend:\n\n  NIC0: mlx5_0\n";
        let topo = parse_topology(mixed).unwrap();
        assert_eq!(topo.links.len(), 3);
        assert_eq!(topo.link(0, 2), Some(LinkKind::System));
        assert!(!topo.all_pairs_nvlink());
    }

    #[test]
    fn malformed_topologies_are_errors_not_partial_answers() {
        let asym = "\tGPU0\tGPU1\nGPU0\t X \tNV4\nGPU1\tPHB\t X \n";
        let unknown_cell = "\tGPU0\tGPU1\nGPU0\t X \tNV?\nGPU1\tNV?\t X \n";
        let missing_row = "\tGPU0\tGPU1\nGPU0\t X \tNV4\n";
        let short_row = "\tGPU0\tGPU1\nGPU0\t X \nGPU1\tNV4\t X \n";
        let bad_diag = "\tGPU0\nGPU0\tNV4\n";
        for bad in [asym, unknown_cell, missing_row, short_row, bad_diag, ""] {
            assert!(parse_topology(bad).is_err(), "{bad:?}");
        }
    }
}
