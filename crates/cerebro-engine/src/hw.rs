//! The machine, as the engine sees it: processor name and threads, memory, and the NVIDIA GPUs the
//! CUDA backend can use. Read from the OS and the driver, never assumed.

#[derive(Debug, Clone, serde::Serialize)]
pub struct Gpu {
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
}

/// The processor's cores. On hybrid CPUs (Intel 12th gen and later, some ARM) "performance" are the
/// cores of the highest efficiency class; on others every core counts as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cores {
    pub physical: usize,
    pub logical: usize,
    pub performance_physical: usize,
    pub performance_logical: usize,
}

impl Cores {
    pub fn hybrid(&self) -> bool {
        self.performance_physical < self.physical
    }
}

/// Core layout from the OS (Windows: GetLogicalProcessorInformationEx); a flat layout of the
/// logical CPU count elsewhere or on failure.
pub fn cores() -> Cores {
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let flat = Cores { physical: logical, logical, performance_physical: logical, performance_logical: logical };
    #[cfg(windows)]
    {
        let mut len = 0u32;
        // SAFETY: a size query with a null buffer.
        unsafe { win::GetLogicalProcessorInformationEx(0, std::ptr::null_mut(), &mut len) };
        if len == 0 {
            return flat;
        }
        let mut buf = vec![0u8; len as usize];
        // SAFETY: buf holds len bytes; RelationProcessorCore (0).
        if unsafe { win::GetLogicalProcessorInformationEx(0, buf.as_mut_ptr(), &mut len) } == 0 {
            return flat;
        }
        let mut list = Vec::new();
        let mut off = 0usize;
        // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship u32, Size u32, then PROCESSOR_RELATIONSHIP:
        // Flags u8, EfficiencyClass u8, Reserved[20], GroupCount u16, GROUP_AFFINITY[GroupCount] (16 bytes each, mask first).
        while off + 32 <= len as usize {
            let size = u32::from_le_bytes(buf[off + 4..off + 8].try_into().expect("4 bytes")) as usize;
            if size == 0 || off + size > len as usize {
                break;
            }
            let class = buf[off + 9];
            let groups = u16::from_le_bytes([buf[off + 30], buf[off + 31]]) as usize;
            let mut threads = 0usize;
            for g in 0..groups {
                let at = off + 32 + g * 16;
                if at + 8 <= off + size {
                    threads += u64::from_le_bytes(buf[at..at + 8].try_into().expect("8 bytes")).count_ones() as usize;
                }
            }
            list.push((class, threads.max(1)));
            off += size;
        }
        return layout(&list).unwrap_or(flat);
    }
    #[allow(unreachable_code)]
    flat
}

/// Cores from (efficiency class, logical threads) per physical core.
fn layout(cores: &[(u8, usize)]) -> Option<Cores> {
    let top = cores.iter().map(|c| c.0).max()?;
    let perf: Vec<_> = cores.iter().filter(|c| c.0 == top).collect();
    Some(Cores {
        physical: cores.len(),
        logical: cores.iter().map(|c| c.1).sum(),
        performance_physical: perf.len(),
        performance_logical: perf.iter().map(|c| c.1).sum(),
    })
}

/// How busy the processor is, 0..1, measured over `window` (Windows: GetSystemTimes); None elsewhere.
pub fn cpu_busy(window: std::time::Duration) -> Option<f64> {
    let a = crate::telemetry::current_cpu_times()?;
    std::thread::sleep(window);
    let b = crate::telemetry::current_cpu_times()?;
    crate::telemetry::cpu_delta(a, b).map(|pct| pct / 100.0)
}

/// A short key for the machine a tuning result belongs to: processor, GPUs and core layout.
pub fn machine_key() -> String {
    let c = cores();
    let gpus: Vec<String> = gpus().into_iter().map(|g| format!("{} {} MiB", g.name, g.vram_total >> 20)).collect();
    format!("{} | {}P{}L | {}", cpu_name(), c.performance_physical, c.logical, if gpus.is_empty() { "no GPU".into() } else { gpus.join(", ") })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Machine {
    pub cpu_name: String,
    pub cpu_threads: usize,
    pub ram_total: u64,
    pub ram_available: u64,
    pub gpus: Vec<Gpu>,
}

/// Physical memory (total, available to applications), bytes.
pub fn memory() -> (u64, u64) {
    #[cfg(windows)]
    {
        let mut st = win::MemoryStatusEx { length: std::mem::size_of::<win::MemoryStatusEx>() as u32, ..Default::default() };
        // SAFETY: st is a properly sized MEMORYSTATUSEX with its length set.
        if unsafe { win::GlobalMemoryStatusEx(&mut st) } != 0 {
            return (st.total_phys, st.avail_phys);
        }
    }
    (0, 0)
}

/// The processor's marketing name from the registry (Windows), or "Unknown processor".
pub fn cpu_name() -> String {
    #[cfg(windows)]
    {
        let key: Vec<u16> = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\0".encode_utf16().collect();
        let val: Vec<u16> = "ProcessorNameString\0".encode_utf16().collect();
        let mut buf = [0u16; 256];
        let mut len = (buf.len() * 2) as u32;
        // SAFETY: valid NUL-terminated names; buf/len describe a writable buffer; RRF_RT_REG_SZ.
        let rc = unsafe { win::RegGetValueW(win::HKEY_LOCAL_MACHINE, key.as_ptr(), val.as_ptr(), 0x0000_0002, std::ptr::null_mut(), buf.as_mut_ptr().cast(), &mut len) };
        if rc == 0 {
            let n = (len as usize / 2).saturating_sub(1).min(buf.len());
            return String::from_utf16_lossy(&buf[..n]).trim().to_owned();
        }
    }
    "Unknown processor".to_owned()
}

/// NVIDIA GPUs through the CUDA driver (empty without one or without the `cuda` feature).
pub fn gpus() -> Vec<Gpu> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        let n = CudaContext::device_count().unwrap_or(0);
        let mut out = Vec::new();
        for i in 0..n.max(0) as usize {
            if let Ok(ctx) = CudaContext::new(i) {
                let name = ctx.name().unwrap_or_else(|_| "NVIDIA GPU".into());
                let (free, total) = ctx.mem_get_info().unwrap_or((0, 0));
                out.push(Gpu { name, vram_total: total as u64, vram_free: free as u64 });
            }
        }
        return out;
    }
    #[allow(unreachable_code)]
    Vec::new()
}

pub fn machine() -> Machine {
    let (ram_total, ram_available) = memory();
    Machine { cpu_name: cpu_name(), cpu_threads: std::thread::available_parallelism().map_or(1, |n| n.get()), ram_total, ram_available, gpus: gpus() }
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    pub const HKEY_LOCAL_MACHINE: isize = 0x8000_0002u32 as i32 as isize;

    #[repr(C)]
    #[derive(Default)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub memory_load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GlobalMemoryStatusEx(st: *mut MemoryStatusEx) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetLogicalProcessorInformationEx(relation: u32, buf: *mut u8, len: *mut u32) -> i32;
    }

    #[link(name = "advapi32")]
    extern "system" {
        pub fn RegGetValueW(hkey: isize, subkey: *const u16, value: *const u16, flags: u32, ty: *mut u32, data: *mut c_void, len: *mut u32) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_and_flat_layouts() {
        // i9-12900H: 6 P-cores with two threads (class 1), 8 E-cores with one (class 0).
        let mut h: Vec<(u8, usize)> = vec![(1, 2); 6];
        h.extend(vec![(0, 1); 8]);
        let c = layout(&h).unwrap();
        assert_eq!((c.physical, c.logical, c.performance_physical, c.performance_logical), (14, 20, 6, 12));
        assert!(c.hybrid());
        let f = layout(&vec![(0, 2); 8]).unwrap();
        assert_eq!((f.physical, f.logical, f.performance_physical, f.performance_logical), (8, 16, 8, 16));
        assert!(!f.hybrid());
        assert!(layout(&[]).is_none());
        let live = cores();
        assert!(live.logical >= live.physical && live.physical >= live.performance_physical && live.performance_physical >= 1);
    }
}
