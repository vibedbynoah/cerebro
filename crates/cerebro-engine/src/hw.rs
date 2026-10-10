//! The machine, as the engine sees it: processor name and threads, memory, and the NVIDIA GPUs the
//! CUDA backend can use. Read from the OS and the driver, never assumed.

#[derive(Debug, Clone, serde::Serialize)]
pub struct Gpu {
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
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

    #[link(name = "advapi32")]
    extern "system" {
        pub fn RegGetValueW(hkey: isize, subkey: *const u16, value: *const u16, flags: u32, ty: *mut u32, data: *mut c_void, len: *mut u32) -> i32;
    }
}
