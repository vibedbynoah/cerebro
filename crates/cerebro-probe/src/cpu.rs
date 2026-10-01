//! Runtime CPU topology: core counts, NUMA nodes, and the affinity masks the tuner pins with.
//!
//! Everything here is *measured on the machine Cerebro is running on*. Nothing is hardcoded, and
//! nothing may be: a mask baked in from the development box would pin inference threads onto
//! processors that do not exist on someone else's CPU, and Windows silently accepts that.
//!
//! Two facts drive the shape of this module.
//!
//! 1. On a hybrid CPU, the two obvious performance-core masks are **not** interchangeable.
//!    [`CpuTopology::p_core_mask_all_logical`] uses every logical processor on the fastest cores;
//!    [`CpuTopology::p_core_mask_one_per_core`] uses one thread per physical core. For a
//!    memory-bandwidth-bound decode loop the SMT sibling often contends for the same load/store
//!    ports and the narrower mask wins — but which wins is a property of the machine and the model,
//!    so both are handed to the tuner as distinct candidates and neither is guessed at.
//! 2. Efficiency class is a `u8`, not a boolean. Some parts report more than two classes, so the
//!    performance cores are "every core in the highest class present", never "class 1".
//!
//! The kernel-facing half is one call to `GetLogicalProcessorInformationEx(RelationProcessorCore)`,
//! which returns a packed sequence of variable-length records. The walker over that buffer is
//! [`parse_processor_cores`], deliberately public and pure so it can be tested against synthetic
//! bytes on any machine — including CI, which is not hybrid.

use std::num::NonZeroU64;

use crate::error::ProbeError;

/// `RelationProcessorCore`, the only relationship this module asks for or parses.
const RELATION_PROCESSOR_CORE: u32 = 0;

/// Field offsets within one `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` record.
///
/// These are load-bearing and easy to get subtly wrong. An earlier revision read the affinity mask
/// at offset 16 — the offset it would have if `Reserved` were absent — and got `0x0` for every
/// core. The call still succeeded, so nothing failed loudly; see [`ProbeError::EmptyCoreMask`].
mod offset {
    /// `DWORD Relationship`
    pub const RELATIONSHIP: usize = 0;
    /// `DWORD Size` — the cursor advances by exactly this, records are variable-length.
    pub const SIZE: usize = 4;
    /// `BYTE Flags` (bit 0 = this core has more than one logical processor).
    pub const FLAGS: usize = 8;
    /// `BYTE EfficiencyClass` — higher is faster.
    pub const EFFICIENCY_CLASS: usize = 9;
    /// `WORD GroupCount`, after `BYTE Reserved[20]`.
    pub const GROUP_COUNT: usize = 30;
    /// `GROUP_AFFINITY GroupMask[]`.
    pub const GROUP_MASK: usize = 32;
    /// `KAFFINITY Mask` within a `GROUP_AFFINITY`.
    pub const AFFINITY_MASK: usize = 0;
    /// `WORD Group` within a `GROUP_AFFINITY`.
    pub const AFFINITY_GROUP: usize = 8;
}

/// `sizeof(GROUP_AFFINITY)` on x64: `KAFFINITY` (8) + `WORD Group` (2) + `WORD Reserved[3]` (6).
const GROUP_AFFINITY_SIZE: usize = 16;

/// Smallest record that can carry its own `Relationship` and `Size`. Anything shorter cannot
/// advance the cursor safely.
const MIN_RECORD_SIZE: usize = 8;

/// `LTP_PC_SMT`: the core exposes more than one logical processor.
const LTP_PC_SMT: u8 = 0x1;

/// One physical core, exactly as the kernel described it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalCore {
    /// Scheduler efficiency class. Higher is faster. A non-hybrid CPU reports `0` for every core,
    /// which is why "is hybrid" must be derived from the *set* of classes seen, not from this value.
    pub efficiency_class: u8,
    /// Whether the core exposes more than one logical processor (`LTP_PC_SMT`).
    pub smt: bool,
    /// Affinity mask of this core's logical processors, within [`PhysicalCore::group`].
    pub mask: u64,
    /// Processor group this core belongs to.
    pub group: u16,
}

impl PhysicalCore {
    /// Number of logical processors on this core.
    pub fn logical_processors(&self) -> u32 {
        self.mask.count_ones()
    }

    /// The lowest-numbered logical processor on this core.
    ///
    /// This is the representative thread for the one-per-physical-core mask. Returning `None` for
    /// an empty mask keeps the caller from inventing processor 0 out of garbage data.
    pub fn first_logical_processor(&self) -> Option<NonZeroU64> {
        NonZeroU64::new(self.mask & self.mask.wrapping_neg())
    }
}

/// Measured CPU topology for the machine this process is running on.
///
/// Constructed only by [`probe`] or [`CpuTopology::from_cores`], both of which reject any layout
/// whose masks would be unusable. By the time a value of this type exists: the performance mask is
/// non-zero, the one-per-core mask is a subset of it, and the efficiency mask (when present) is
/// disjoint from both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTopology {
    logical_processors: u32,
    physical_cores: u32,
    numa_nodes: u32,
    performance_class: u8,
    performance_cores: u32,
    efficiency_cores: u32,
    p_core_mask_all_logical: NonZeroU64,
    p_core_mask_one_per_core: NonZeroU64,
    e_core_mask: Option<NonZeroU64>,
}

impl CpuTopology {
    /// Derive a topology from parsed core records and a NUMA node count.
    ///
    /// Split out from [`probe`] so the whole derivation — class grouping, mask construction, every
    /// invariant below — is exercised from a synthetic buffer on machines that are not hybrid.
    ///
    /// Rejects: an empty core list, a core with an empty mask (the wrong-offset signature),
    /// overlapping cores, more than one processor group, and a zero NUMA count.
    pub fn from_cores(cores: &[PhysicalCore], numa_nodes: u32) -> Result<Self, ProbeError> {
        if cores.is_empty() {
            return Err(ProbeError::NoCores);
        }
        if numa_nodes == 0 {
            return Err(ProbeError::ZeroNumaNodes);
        }

        let mut seen: u64 = 0;
        for core in cores {
            if core.group != 0 {
                return Err(ProbeError::MultipleProcessorGroups { group: core.group });
            }
            if core.mask == 0 {
                return Err(ProbeError::EmptyCoreMask {
                    efficiency_class: core.efficiency_class,
                });
            }
            let overlap = seen & core.mask;
            if overlap != 0 {
                return Err(ProbeError::OverlappingCores { overlap });
            }
            seen |= core.mask;
        }

        // Highest class present is the performance class. Never "class 1": parts exist that report
        // three classes, and a non-hybrid part reports only class 0.
        let performance_class = cores.iter().map(|c| c.efficiency_class).max().unwrap_or(0);
        let hybrid = cores
            .iter()
            .any(|c| c.efficiency_class != performance_class);

        let mut p_all: u64 = 0;
        let mut p_one: u64 = 0;
        let mut e_all: u64 = 0;
        let mut performance_cores = 0u32;
        let mut efficiency_cores = 0u32;

        for core in cores {
            if core.efficiency_class == performance_class {
                performance_cores += 1;
                p_all |= core.mask;
                // One representative thread per physical core. `first_logical_processor` cannot be
                // None here — empty masks were rejected above — but the error path is kept rather
                // than unwrapped so no public entry point can panic.
                let first = core
                    .first_logical_processor()
                    .ok_or(ProbeError::EmptyCoreMask {
                        efficiency_class: core.efficiency_class,
                    })?;
                p_one |= first.get();
            } else {
                efficiency_cores += 1;
                e_all |= core.mask;
            }
        }

        let p_core_mask_all_logical = NonZeroU64::new(p_all).ok_or(ProbeError::EmptyCoreMask {
            efficiency_class: performance_class,
        })?;
        let p_core_mask_one_per_core = NonZeroU64::new(p_one).ok_or(ProbeError::EmptyCoreMask {
            efficiency_class: performance_class,
        })?;

        // `Option<NonZeroU64>` rather than `u64`: on a non-hybrid CPU the answer is "there is no
        // such mask", and `Some(0)` — a mask pinning threads to no processor — is made
        // unrepresentable rather than documented against.
        let e_core_mask = if hybrid { NonZeroU64::new(e_all) } else { None };

        Ok(Self {
            logical_processors: seen.count_ones(),
            physical_cores: cores.len() as u32,
            numa_nodes,
            performance_class,
            performance_cores,
            efficiency_cores,
            p_core_mask_all_logical,
            p_core_mask_one_per_core,
            e_core_mask,
        })
    }

    /// Total logical processors (hardware threads) visible to this process.
    pub fn logical_processors(&self) -> u32 {
        self.logical_processors
    }

    /// Total physical cores.
    pub fn physical_cores(&self) -> u32 {
        self.physical_cores
    }

    /// NUMA node count, at least 1.
    ///
    /// Reported so the tuner can prune: with a single node the `--numa` axis cannot change
    /// throughput, and benchmarking its settings burns soak-test minutes to measure noise. See
    /// [`CpuTopology::numa_axis_can_move`].
    pub fn numa_nodes(&self) -> u32 {
        self.numa_nodes
    }

    /// Whether the `--numa` axis is worth searching on this machine.
    pub fn numa_axis_can_move(&self) -> bool {
        self.numa_nodes > 1
    }

    /// Whether more than one efficiency class is present.
    pub fn is_hybrid(&self) -> bool {
        self.e_core_mask.is_some()
    }

    /// The efficiency class treated as "performance" — the highest class present.
    pub fn performance_class(&self) -> u8 {
        self.performance_class
    }

    /// Physical cores in the performance class.
    pub fn performance_cores(&self) -> u32 {
        self.performance_cores
    }

    /// Physical cores below the performance class. `0` on a non-hybrid CPU.
    pub fn efficiency_cores(&self) -> u32 {
        self.efficiency_cores
    }

    /// Every logical processor on the performance cores, SMT siblings included.
    ///
    /// Candidate A for the tuner. On the development i9-12900H this is `0xFFF` (12 threads).
    pub fn p_core_mask_all_logical(&self) -> NonZeroU64 {
        self.p_core_mask_all_logical
    }

    /// Exactly one logical processor per physical performance core.
    ///
    /// Candidate B, and a strict subset of [`CpuTopology::p_core_mask_all_logical`]. It is offered
    /// separately because for a bandwidth-bound workload the SMT sibling can cost throughput rather
    /// than add it — the two are not equivalent and the tuner must measure both. On the development
    /// i9-12900H this is `0x555` (6 threads).
    pub fn p_core_mask_one_per_core(&self) -> NonZeroU64 {
        self.p_core_mask_one_per_core
    }

    /// Every logical processor *not* in the performance class.
    ///
    /// `None` on a non-hybrid CPU — where there are no efficiency cores, there is no mask, not an
    /// empty one.
    pub fn e_core_mask(&self) -> Option<NonZeroU64> {
        self.e_core_mask
    }
}

/// Walk a `GetLogicalProcessorInformationEx` buffer and collect every processor-core record.
///
/// Public and pure so the record walker is testable from synthetic bytes; the hazards it guards are
/// impossible to reproduce on demand with a real CPU.
///
/// The buffer is length-prefixed data walked with a cursor, so it is treated as untrusted shape
/// even though the kernel wrote it: `Size` drives the cursor, and a zero, undersized, or
/// past-the-end `Size` is an error rather than an infinite loop or an out-of-bounds read. Records
/// whose `Relationship` is not `RelationProcessorCore` are skipped by `Size`, not assumed absent.
pub fn parse_processor_cores(buffer: &[u8]) -> Result<Vec<PhysicalCore>, ProbeError> {
    let mut cores = Vec::new();
    let mut cursor = 0usize;

    while cursor < buffer.len() {
        let remaining = buffer.len() - cursor;
        let header =
            buffer
                .get(cursor..cursor + MIN_RECORD_SIZE)
                .ok_or(ProbeError::TruncatedRecord {
                    offset: cursor,
                    size: MIN_RECORD_SIZE,
                    remaining,
                })?;

        let relationship = read_u32(header, offset::RELATIONSHIP)?;
        let size = read_u32(header, offset::SIZE)? as usize;

        // Guard the cursor before it moves. `size == 0` would spin forever on the same bytes.
        if size < MIN_RECORD_SIZE {
            return Err(ProbeError::ImpossibleRecordSize {
                offset: cursor,
                size,
                minimum: MIN_RECORD_SIZE,
            });
        }
        let record =
            buffer
                .get(cursor..cursor.saturating_add(size))
                .ok_or(ProbeError::TruncatedRecord {
                    offset: cursor,
                    size,
                    remaining,
                })?;

        if relationship == RELATION_PROCESSOR_CORE {
            parse_core_record(record, cursor, &mut cores)?;
        }

        cursor += size;
    }

    Ok(cores)
}

/// Decode one `RelationProcessorCore` record, appending one [`PhysicalCore`] per group entry.
fn parse_core_record(
    record: &[u8],
    offset_in_buffer: usize,
    out: &mut Vec<PhysicalCore>,
) -> Result<(), ProbeError> {
    let flags = *record
        .get(offset::FLAGS)
        .ok_or(ProbeError::ImpossibleRecordSize {
            offset: offset_in_buffer,
            size: record.len(),
            minimum: offset::GROUP_MASK + GROUP_AFFINITY_SIZE,
        })?;
    let efficiency_class =
        *record
            .get(offset::EFFICIENCY_CLASS)
            .ok_or(ProbeError::ImpossibleRecordSize {
                offset: offset_in_buffer,
                size: record.len(),
                minimum: offset::GROUP_MASK + GROUP_AFFINITY_SIZE,
            })?;
    let group_count =
        read_u16(record, offset::GROUP_COUNT).map_err(|_| ProbeError::ImpossibleRecordSize {
            offset: offset_in_buffer,
            size: record.len(),
            minimum: offset::GROUP_MASK + GROUP_AFFINITY_SIZE,
        })? as usize;

    // `GroupCount` is a count of trailing variable-length entries: check it against the record's
    // own declared length before indexing, or a bad count reads past the record into the next one.
    let needed = offset::GROUP_MASK
        .checked_add(group_count.saturating_mul(GROUP_AFFINITY_SIZE))
        .ok_or(ProbeError::GroupCountOverrun {
            offset: offset_in_buffer,
            group_count,
            size: record.len(),
        })?;
    if group_count == 0 || needed > record.len() {
        return Err(ProbeError::GroupCountOverrun {
            offset: offset_in_buffer,
            group_count,
            size: record.len(),
        });
    }

    for i in 0..group_count {
        let base = offset::GROUP_MASK + i * GROUP_AFFINITY_SIZE;
        let mask = read_u64(record, base + offset::AFFINITY_MASK)?;
        let group = read_u16(record, base + offset::AFFINITY_GROUP)?;
        out.push(PhysicalCore {
            efficiency_class,
            smt: flags & LTP_PC_SMT != 0,
            mask,
            group,
        });
    }

    Ok(())
}

/// Read a little-endian `u16` at `at`, or report the record as truncated. Slice indexing is never
/// used directly anywhere in this module; every read goes through one of these three.
fn read_u16(bytes: &[u8], at: usize) -> Result<u16, ProbeError> {
    let raw: [u8; 2] = bytes
        .get(at..at.saturating_add(2))
        .and_then(|s| s.try_into().ok())
        .ok_or(ProbeError::TruncatedRecord {
            offset: at,
            size: 2,
            remaining: bytes.len().saturating_sub(at),
        })?;
    Ok(u16::from_le_bytes(raw))
}

/// Read a little-endian `u32` at `at`, or report the record as truncated.
fn read_u32(bytes: &[u8], at: usize) -> Result<u32, ProbeError> {
    let raw: [u8; 4] = bytes
        .get(at..at.saturating_add(4))
        .and_then(|s| s.try_into().ok())
        .ok_or(ProbeError::TruncatedRecord {
            offset: at,
            size: 4,
            remaining: bytes.len().saturating_sub(at),
        })?;
    Ok(u32::from_le_bytes(raw))
}

/// Read a little-endian `u64` at `at`, or report the record as truncated.
fn read_u64(bytes: &[u8], at: usize) -> Result<u64, ProbeError> {
    let raw: [u8; 8] = bytes
        .get(at..at.saturating_add(8))
        .and_then(|s| s.try_into().ok())
        .ok_or(ProbeError::TruncatedRecord {
            offset: at,
            size: 8,
            remaining: bytes.len().saturating_sub(at),
        })?;
    Ok(u64::from_le_bytes(raw))
}

/// Probe this machine's CPU topology.
///
/// Every value is read from the running kernel. Nothing about the development machine's layout is
/// baked in, because Cerebro ships to other people's CPUs.
#[cfg(windows)]
pub fn probe() -> Result<CpuTopology, ProbeError> {
    let buffer = query_processor_cores()?;
    let cores = parse_processor_cores(&buffer)?;
    let numa_nodes = numa_node_count()?;
    CpuTopology::from_cores(&cores, numa_nodes)
}

/// Probe this machine's CPU topology.
///
/// Always [`ProbeError::UnsupportedPlatform`] off Windows: efficiency class and NUMA topology have
/// no portable API, and a guessed topology is worse than an admitted absence.
#[cfg(not(windows))]
pub fn probe() -> Result<CpuTopology, ProbeError> {
    Err(ProbeError::UnsupportedPlatform)
}

/// The CPU's brand string, e.g. `12th Gen Intel(R) Core(TM) i9-12900H`, from CPUID leaves
/// `0x8000_0002..=0x8000_0004`: the same text llama.cpp records as `cpu_info`.
///
/// Topology alone does not identify a CPU: an i7-12700H and an i9-12900H are both 6P+8E/20T but
/// clock differently, so a machine identity built only from core counts would merge them. `None`
/// on a CPU that does not implement the brand leaves, off x86, or when the string is empty.
pub fn brand_string() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return None;
        }
        let mut bytes = Vec::with_capacity(48);
        for leaf in 0x8000_0002_u32..=0x8000_0004 {
            let r = __cpuid(leaf);
            for word in [r.eax, r.ebx, r.ecx, r.edx] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        brand_from_cpuid_bytes(&bytes)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// The brand string from CPUID's 48 raw bytes: up to the first NUL, surrounding spaces trimmed
/// (Intel pads on the left, AMD on the right), `None` if empty or not ASCII.
fn brand_from_cpuid_bytes(bytes: &[u8]) -> Option<String> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let text = std::str::from_utf8(bytes.get(..end)?).ok()?.trim();
    (!text.is_empty() && text.is_ascii()).then(|| text.to_string())
}

/// Ask the kernel for the `RelationProcessorCore` records, using the documented two-call pattern.
///
/// The required length is asked for, never guessed: record size varies with `GroupCount` and with
/// the Windows build, so any fixed buffer constant is a truncation waiting for a bigger machine.
#[cfg(windows)]
fn query_processor_cores() -> Result<Vec<u8>, ProbeError> {
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_INSUFFICIENT_BUFFER};
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    const CALL: &str = "GetLogicalProcessorInformationEx";
    /// Sizing and filling race against CPU hot-add. One retry is normal; unbounded retries would
    /// hang the tuner, so the loop is bounded and the exhaustion is a typed error.
    const MAX_ATTEMPTS: u32 = 4;

    let mut length: u32 = 0;
    // SAFETY: the documented sizing call. A null buffer pointer is explicitly allowed when the
    // length it points at is 0; the API then writes the required byte count into `length` and fails
    // with ERROR_INSUFFICIENT_BUFFER. `length` is a live local for the duration of the call.
    let sized = unsafe {
        GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut length)
    };
    if sized != 0 {
        // Succeeded with a zero-byte buffer: the kernel is telling us there is nothing to report.
        return Err(ProbeError::NoCores);
    }
    // SAFETY: reading the calling thread's last-error value immediately after the failed call above.
    let code = unsafe { GetLastError() };
    if code != ERROR_INSUFFICIENT_BUFFER {
        return Err(ProbeError::Win32 { call: CALL, code });
    }

    for _ in 0..MAX_ATTEMPTS {
        if length == 0 {
            return Err(ProbeError::NoCores);
        }
        // Allocated as u64s so the buffer is 8-byte aligned for the kernel's writes; it is only
        // ever read back as bytes.
        let words = (length as usize).div_ceil(std::mem::size_of::<u64>());
        let mut storage: Vec<u64> = vec![0; words];
        let capacity_bytes = words * std::mem::size_of::<u64>();
        let mut written = length;

        // SAFETY: `storage` owns `capacity_bytes >= written` writable, 8-byte-aligned bytes, and
        // `written` tells the API exactly how many it may use. The pointer cast only reinterprets
        // that allocation as the record type the API fills; it is never dereferenced as that type
        // on this side.
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                storage
                    .as_mut_ptr()
                    .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
                &mut written,
            )
        };

        if ok != 0 {
            let written = (written as usize).min(capacity_bytes);
            // SAFETY: `storage` is live, initialized (zeroed at allocation, then partly overwritten
            // by the kernel), and `written` is clamped to its byte capacity. `u8` has no alignment
            // requirement, so viewing a `[u64]` allocation as bytes is sound.
            let bytes =
                unsafe { std::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), written) };
            return Ok(bytes.to_vec());
        }

        // SAFETY: reading the calling thread's last-error value immediately after the failed call.
        let code = unsafe { GetLastError() };
        if code != ERROR_INSUFFICIENT_BUFFER {
            return Err(ProbeError::Win32 { call: CALL, code });
        }
        // A core was hot-added between sizing and filling; `written` now holds the new requirement.
        length = written;
    }

    Err(ProbeError::BufferGrowthLoop {
        attempts: MAX_ATTEMPTS,
    })
}

/// NUMA node count via `GetNumaHighestNodeNumber`.
///
/// The API reports the highest node *number*, so the count is that plus one; a single-node machine
/// answers 0 and must be reported as 1 node, not 0.
#[cfg(windows)]
fn numa_node_count() -> Result<u32, ProbeError> {
    use windows_sys::Win32::Foundation::GetLastError;
    // Lives under `Threading`, not `SystemInformation`, despite being a topology query.
    use windows_sys::Win32::System::Threading::GetNumaHighestNodeNumber;

    let mut highest: u32 = 0;
    // SAFETY: the API writes one ULONG through the pointer; `highest` is a live, initialized local
    // of exactly that type for the duration of the call.
    let ok = unsafe { GetNumaHighestNodeNumber(&mut highest) };
    if ok == 0 {
        // SAFETY: reading the calling thread's last-error value immediately after the failed call.
        let code = unsafe { GetLastError() };
        return Err(ProbeError::Win32 {
            call: "GetNumaHighestNodeNumber",
            code,
        });
    }
    Ok(highest.saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_brand_string_is_trimmed_at_its_nul_and_padding() {
        let mut raw = b"       Intel(R) Core(TM) i9-12900H".to_vec();
        raw.resize(48, 0);
        assert_eq!(
            brand_from_cpuid_bytes(&raw).as_deref(),
            Some("Intel(R) Core(TM) i9-12900H")
        );
        assert_eq!(brand_from_cpuid_bytes(&[0; 48]), None);
        assert_eq!(brand_from_cpuid_bytes(b"   "), None);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn this_machine_reports_a_brand_string() {
        // Every x86-64 CPU shipped since 2003 implements the brand leaves.
        let brand = brand_string().expect("an x86-64 CPU has a brand string");
        assert!(!brand.is_empty() && brand.trim() == brand, "{brand:?}");
    }

    /// Build one `RelationProcessorCore` record. `trailing` pads the record beyond its group
    /// entries so that tests exercise a `Size` that is not a constant.
    fn core_record(
        efficiency_class: u8,
        smt: bool,
        groups: &[(u64, u16)],
        trailing: usize,
    ) -> Vec<u8> {
        let size = offset::GROUP_MASK + groups.len() * GROUP_AFFINITY_SIZE + trailing;
        let mut r = vec![0u8; size];
        r[offset::RELATIONSHIP..offset::RELATIONSHIP + 4]
            .copy_from_slice(&RELATION_PROCESSOR_CORE.to_le_bytes());
        r[offset::SIZE..offset::SIZE + 4].copy_from_slice(&(size as u32).to_le_bytes());
        r[offset::FLAGS] = if smt { LTP_PC_SMT } else { 0 };
        r[offset::EFFICIENCY_CLASS] = efficiency_class;
        r[offset::GROUP_COUNT..offset::GROUP_COUNT + 2]
            .copy_from_slice(&(groups.len() as u16).to_le_bytes());
        for (i, (mask, group)) in groups.iter().enumerate() {
            let base = offset::GROUP_MASK + i * GROUP_AFFINITY_SIZE;
            r[base + offset::AFFINITY_MASK..base + offset::AFFINITY_MASK + 8]
                .copy_from_slice(&mask.to_le_bytes());
            r[base + offset::AFFINITY_GROUP..base + offset::AFFINITY_GROUP + 2]
                .copy_from_slice(&group.to_le_bytes());
        }
        r
    }

    /// A record of some other relationship, to be skipped by `Size`.
    fn other_record(relationship: u32, size: usize) -> Vec<u8> {
        let mut r = vec![0u8; size];
        r[0..4].copy_from_slice(&relationship.to_le_bytes());
        r[4..8].copy_from_slice(&(size as u32).to_le_bytes());
        r
    }

    /// The development i9-12900H layout, synthesized: 6 SMT performance cores (class 1) on logical
    /// processors 0-11, then 8 non-SMT efficiency cores (class 0) on 12-19.
    fn hybrid_buffer() -> Vec<u8> {
        let mut buf = Vec::new();
        for i in 0..6u32 {
            buf.extend(core_record(1, true, &[(0b11u64 << (i * 2), 0)], 0));
        }
        for i in 0..8u32 {
            // Padded: efficiency-core records are deliberately a different length here, so a walker
            // that advanced by a constant instead of by `Size` would desynchronize immediately.
            buf.extend(core_record(0, false, &[(1u64 << (12 + i), 0)], 16));
        }
        buf
    }

    #[test]
    fn walks_variable_length_records_and_reads_the_mask_at_the_right_offset() {
        // The highest-value test in the component. It pins two hazards at once:
        //  * the affinity mask lives at record offset 32, not 16 — reading 16 yields 0x0 for every
        //    core while the kernel call still reports success, so only exact mask values catch it;
        //  * the cursor must advance by the record's own `Size`, which differs between the two core
        //    kinds in this buffer.
        let cores =
            parse_processor_cores(&hybrid_buffer()).expect("synthetic buffer is well formed");
        assert_eq!(cores.len(), 14);

        let topo = CpuTopology::from_cores(&cores, 1).expect("layout is valid");
        assert_eq!(topo.physical_cores(), 14);
        assert_eq!(topo.logical_processors(), 20);
        assert_eq!(topo.performance_class(), 1);
        assert_eq!(topo.performance_cores(), 6);
        assert_eq!(topo.efficiency_cores(), 8);
        assert_eq!(topo.p_core_mask_all_logical().get(), 0xFFF);
        assert_eq!(topo.p_core_mask_one_per_core().get(), 0x555);
        assert_eq!(topo.e_core_mask().map(NonZeroU64::get), Some(0xFF000));
        assert!(topo.is_hybrid());
        assert!(!topo.numa_axis_can_move());
    }

    #[test]
    fn unrelated_records_are_skipped_by_size_not_assumed_absent() {
        // The query filters by relationship, but the walker must not depend on that filter holding.
        let mut buf = other_record(1, 40);
        buf.extend(core_record(0, true, &[(0b11, 0)], 0));
        buf.extend(other_record(3, 24));
        buf.extend(core_record(0, true, &[(0b1100, 0)], 8));

        let cores = parse_processor_cores(&buf).expect("mixed buffer parses");
        assert_eq!(cores.len(), 2);
        assert_eq!(cores[0].mask, 0b11);
        assert_eq!(cores[1].mask, 0b1100);
    }

    #[test]
    fn zero_size_is_rejected_before_the_cursor_moves() {
        // A `Size` of 0 would re-parse the same bytes forever and hang the tuner at startup.
        let mut buf = core_record(0, false, &[(0b1, 0)], 0);
        let start = buf.len();
        buf.extend(core_record(0, false, &[(0b10, 0)], 0));
        buf[start + offset::SIZE..start + offset::SIZE + 4].copy_from_slice(&0u32.to_le_bytes());

        assert!(matches!(
            parse_processor_cores(&buf),
            Err(ProbeError::ImpossibleRecordSize { size: 0, .. })
        ));
    }

    #[test]
    fn size_past_the_end_of_the_buffer_is_rejected() {
        // Trusting `Size` here would read past the length the kernel actually filled.
        let mut buf = core_record(0, false, &[(0b1, 0)], 0);
        let declared = (buf.len() + 64) as u32;
        buf[offset::SIZE..offset::SIZE + 4].copy_from_slice(&declared.to_le_bytes());

        assert!(matches!(
            parse_processor_cores(&buf),
            Err(ProbeError::TruncatedRecord { offset: 0, .. })
        ));
    }

    #[test]
    fn a_record_header_split_by_truncation_is_rejected() {
        // Fewer than 8 trailing bytes cannot even hold Relationship + Size.
        let mut buf = core_record(0, false, &[(0b1, 0)], 0);
        buf.extend_from_slice(&[0u8; 5]);
        assert!(matches!(
            parse_processor_cores(&buf),
            Err(ProbeError::TruncatedRecord { .. })
        ));
    }

    #[test]
    fn group_count_beyond_the_record_is_rejected() {
        // A `GroupCount` larger than the record's own `Size` would walk into the next record's
        // bytes and manufacture cores that do not exist.
        let mut buf = core_record(0, true, &[(0b11, 0)], 0);
        buf[offset::GROUP_COUNT..offset::GROUP_COUNT + 2].copy_from_slice(&9u16.to_le_bytes());
        assert!(matches!(
            parse_processor_cores(&buf),
            Err(ProbeError::GroupCountOverrun { group_count: 9, .. })
        ));

        // Zero is equally malformed: a core belongs to at least one group.
        let mut buf = core_record(0, true, &[(0b11, 0)], 0);
        buf[offset::GROUP_COUNT..offset::GROUP_COUNT + 2].copy_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            parse_processor_cores(&buf),
            Err(ProbeError::GroupCountOverrun { group_count: 0, .. })
        ));
    }

    #[test]
    fn an_all_zero_mask_is_an_error_not_a_topology() {
        // This is exactly what reading the mask from the wrong offset produces: a successful call
        // and a set of cores that pin to nothing. It must never reach the tuner as a usable mask.
        let cores = vec![PhysicalCore {
            efficiency_class: 0,
            smt: false,
            mask: 0,
            group: 0,
        }];
        assert!(matches!(
            CpuTopology::from_cores(&cores, 1),
            Err(ProbeError::EmptyCoreMask { .. })
        ));
    }

    #[test]
    fn a_non_hybrid_cpu_has_no_e_core_mask_rather_than_an_empty_one() {
        // `Some(0)` would be accepted by `--cpu-mask` and pin inference to no processor at all.
        let buf: Vec<u8> = (0..8u32)
            .flat_map(|i| core_record(0, true, &[(0b11u64 << (i * 2), 0)], 0))
            .collect();
        let cores = parse_processor_cores(&buf).expect("uniform buffer parses");
        let topo = CpuTopology::from_cores(&cores, 1).expect("layout is valid");

        assert!(!topo.is_hybrid());
        assert_eq!(topo.e_core_mask(), None);
        assert_eq!(topo.performance_cores(), 8);
        assert_eq!(topo.efficiency_cores(), 0);
        assert_eq!(topo.p_core_mask_all_logical().get(), 0xFFFF);
        assert_eq!(topo.p_core_mask_one_per_core().get(), 0x5555);
    }

    #[test]
    fn more_than_two_efficiency_classes_resolve_to_the_highest() {
        // Assuming "class 1 means performance" would pick the middle tier on a part that reports
        // three classes, pinning inference to cores that are not the fastest available.
        let mut buf = core_record(2, true, &[(0b11, 0)], 0);
        buf.extend(core_record(1, false, &[(0b100, 0)], 0));
        buf.extend(core_record(0, false, &[(0b1000, 0)], 0));

        let cores = parse_processor_cores(&buf).expect("three-class buffer parses");
        let topo = CpuTopology::from_cores(&cores, 2).expect("layout is valid");

        assert_eq!(topo.performance_class(), 2);
        assert_eq!(topo.performance_cores(), 1);
        assert_eq!(topo.efficiency_cores(), 2);
        assert_eq!(topo.p_core_mask_all_logical().get(), 0b11);
        assert_eq!(topo.p_core_mask_one_per_core().get(), 0b01);
        // Everything below the top class is an efficiency core, not just class 0.
        assert_eq!(topo.e_core_mask().map(NonZeroU64::get), Some(0b1100));
        assert!(topo.numa_axis_can_move());
    }

    #[test]
    fn multi_group_machines_are_refused_rather_than_silently_halved() {
        let cores = vec![
            PhysicalCore {
                efficiency_class: 0,
                smt: true,
                mask: 0b11,
                group: 0,
            },
            PhysicalCore {
                efficiency_class: 0,
                smt: true,
                mask: 0b11,
                group: 1,
            },
        ];
        assert!(matches!(
            CpuTopology::from_cores(&cores, 1),
            Err(ProbeError::MultipleProcessorGroups { group: 1 })
        ));
    }

    #[test]
    fn overlapping_and_degenerate_inputs_are_refused() {
        let overlapping = vec![
            PhysicalCore {
                efficiency_class: 0,
                smt: true,
                mask: 0b011,
                group: 0,
            },
            PhysicalCore {
                efficiency_class: 0,
                smt: true,
                mask: 0b110,
                group: 0,
            },
        ];
        assert!(matches!(
            CpuTopology::from_cores(&overlapping, 1),
            Err(ProbeError::OverlappingCores { overlap: 0b010 })
        ));
        assert!(matches!(
            CpuTopology::from_cores(&[], 1),
            Err(ProbeError::NoCores)
        ));
        // A machine always has at least one NUMA node; 0 would mean the count was never read.
        let ok = vec![PhysicalCore {
            efficiency_class: 0,
            smt: false,
            mask: 0b1,
            group: 0,
        }];
        assert!(matches!(
            CpuTopology::from_cores(&ok, 0),
            Err(ProbeError::ZeroNumaNodes)
        ));
        assert!(matches!(parse_processor_cores(&[]), Ok(v) if v.is_empty()));
    }

    /// Invariants that must hold on every machine this ever runs on, hybrid or not, CI included.
    #[cfg(windows)]
    #[test]
    fn live_topology_satisfies_the_invariants_the_tuner_relies_on() {
        let topo = probe().expect("topology probe must succeed on Windows");

        assert!(topo.physical_cores() >= 1);
        assert!(topo.logical_processors() >= topo.physical_cores());
        assert!(topo.numa_nodes() >= 1);
        assert_eq!(topo.numa_axis_can_move(), topo.numa_nodes() > 1);

        let p_all = topo.p_core_mask_all_logical().get();
        let p_one = topo.p_core_mask_one_per_core().get();

        // One representative thread per physical performance core, so: a subset of the full mask,
        // with exactly one bit per performance core.
        assert_eq!(
            p_one & p_all,
            p_one,
            "one-per-core must be a subset of all-logical"
        );
        assert_eq!(p_one.count_ones(), topo.performance_cores());
        assert!(p_all.count_ones() >= p_one.count_ones());

        match topo.e_core_mask() {
            Some(e) => {
                assert!(topo.is_hybrid());
                assert_eq!(p_all & e.get(), 0, "P and E masks must be disjoint");
                assert!(topo.efficiency_cores() >= 1);
                assert_eq!(
                    (p_all | e.get()).count_ones(),
                    topo.logical_processors(),
                    "every logical processor belongs to exactly one of the two masks"
                );
            }
            None => {
                assert!(!topo.is_hybrid());
                assert_eq!(topo.efficiency_cores(), 0);
                assert_eq!(p_all.count_ones(), topo.logical_processors());
            }
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "asserts the i9-12900H development machine's exact layout; meaningless on any other CPU"]
    fn live_topology_matches_the_development_machine() {
        let topo = probe().expect("topology probe must succeed on Windows");
        assert_eq!(topo.physical_cores(), 14);
        assert_eq!(topo.logical_processors(), 20);
        assert_eq!(topo.numa_nodes(), 1);
        assert_eq!(topo.performance_class(), 1);
        assert_eq!(topo.performance_cores(), 6);
        assert_eq!(topo.efficiency_cores(), 8);
        assert_eq!(topo.p_core_mask_all_logical().get(), 0xFFF);
        assert_eq!(topo.p_core_mask_one_per_core().get(), 0x555);
        assert_eq!(topo.e_core_mask().map(NonZeroU64::get), Some(0xFF000));
    }
}
