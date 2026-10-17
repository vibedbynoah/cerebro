//! Live hardware telemetry for the Monitor: CPU load and, per NVIDIA GPU, utilization, memory use,
//! temperature, power, clocks and PCIe throughput — sampled about once per second, like nvidia-smi
//! plus Task Manager. Read from the OS and the driver, never assumed: anything the machine will not
//! tell us comes back as `None`, never as an error and never as a panic.

#[cfg(windows)]
use std::ffi::c_void;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct GpuSample {
    pub index: u32,
    pub util_pct: Option<f64>,
    pub mem_util_pct: Option<f64>,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub sm_clock_mhz: Option<f64>,
    pub pcie_rx_mbs: Option<f64>,
    pub pcie_tx_mbs: Option<f64>,
    pub pcie_gen: Option<u32>,
    pub pcie_gen_max: Option<u32>,
    pub pcie_width: Option<u32>,
    pub throttle_reasons: Vec<&'static str>,
    pub driver: Option<String>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Sample {
    pub cpu_pct: Option<f64>,
    pub gpus: Vec<GpuSample>,
}

/// Holds the resolved NVML entry points (loaded once) and the previous CPU-time reading.
pub struct Sampler {
    #[cfg(windows)]
    nvml: Option<Nvml>,
    cpu: CpuTimes,
}

impl Sampler {
    /// Loads NVML if present and prepares the first CPU delta. Cheap and non-failing.
    pub fn new() -> Self {
        Sampler {
            #[cfg(windows)]
            nvml: Nvml::load(),
            cpu: CpuTimes::default(),
        }
    }

    /// One reading. Never panics; unreadable values are `None`.
    pub fn sample(&mut self) -> Sample {
        let cpu_pct = self.cpu.sample();
        #[cfg(windows)]
        let gpus = gpu_samples(self.nvml.as_ref());
        #[cfg(not(windows))]
        let gpus = Vec::new();
        Sample { cpu_pct, gpus }
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct CpuTimes {
    prev: Option<(u64, u64, u64)>,
}

impl CpuTimes {
    fn sample(&mut self) -> Option<f64> {
        let now = current_cpu_times()?;
        let prev = self.prev.replace(now);
        prev.and_then(|prev| cpu_delta(prev, now))
    }
}

fn cpu_delta(prev: (u64, u64, u64), now: (u64, u64, u64)) -> Option<f64> {
    let idle = now.0.wrapping_sub(prev.0) as f64;
    // GetSystemTimes' kernel time already includes idle time, so total = kernel + user.
    let total = (now.1.wrapping_sub(prev.1) + now.2.wrapping_sub(prev.2)) as f64;
    (total > 0.0).then(|| ((total - idle) / total * 100.0).clamp(0.0, 100.0))
}

#[cfg(windows)]
fn current_cpu_times() -> Option<(u64, u64, u64)> {
    let mut idle = win::FileTime::default();
    let mut kernel = win::FileTime::default();
    let mut user = win::FileTime::default();
    // SAFETY: three valid, writable FILETIME out-params.
    let ok = unsafe { win::GetSystemTimes(&mut idle, &mut kernel, &mut user) } != 0;
    ok.then(|| (idle.as_u64(), kernel.as_u64(), user.as_u64()))
}

#[cfg(not(windows))]
fn current_cpu_times() -> Option<(u64, u64, u64)> {
    None
}

// ---------------------------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------------------------

/// NVML bitmask from `nvmlDeviceGetCurrentClocksThrottleReasons`, in label form.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn throttle_labels(bits: u64) -> Vec<&'static str> {
    #[rustfmt::skip]
    const TABLE: [(u64, &str); 8] = [
        (0x1,   "idle"),
        (0x2,   "app clocks"),
        (0x4,   "power cap"),
        (0x8,   "hw slowdown"),
        (0x20,  "thermal"),
        (0x40,  "thermal"),
        (0x80,  "power brake"),
        (0x100, "display clock"),
    ];
    let mut v: Vec<&'static str> = TABLE.iter().filter(|(mask, _)| bits & mask != 0).map(|(_, label)| *label).collect();
    v.dedup();
    v
}

#[cfg(windows)]
fn gpu_samples(nvml: Option<&Nvml>) -> Vec<GpuSample> {
    let Some(nvml) = nvml else { return Vec::new() };
    let mut count = 0u32;
    if nvml.get_count(&mut count) != 0 {
        return Vec::new();
    }
    let driver = nvml.driver_version();
    let mut out = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut dev = std::ptr::null_mut();
        if nvml.get_handle(index, &mut dev) != 0 || dev.is_null() {
            continue;
        }
        let mut g = GpuSample { index, driver: driver.clone(), ..Default::default() };

        let mut util = Util::default();
        if nvml.get_utilization(dev, &mut util) == 0 {
            g.util_pct = Some(util.gpu as f64);
            g.mem_util_pct = Some(util.memory as f64);
        }
        let mut temp = 0u32;
        if nvml.get_temperature(dev, 0, &mut temp) == 0 {
            g.temp_c = Some(temp as f64);
        }
        let mut mw = 0u32;
        if nvml.get_power_usage(dev, &mut mw) == 0 {
            g.power_w = Some(mw as f64 / 1000.0);
        }
        if nvml.get_power_limit(dev, &mut mw) == 0 {
            g.power_limit_w = Some(mw as f64 / 1000.0);
        }
        let mut mhz = 0u32;
        if nvml.get_clock(dev, 1, &mut mhz) == 0 {
            g.sm_clock_mhz = Some(mhz as f64);
        }
        // PCIe counters: 0 = TX, 1 = RX, both in KB/s. Each call costs ~20 ms.
        let mut kbs = 0u32;
        if nvml.get_pcie_throughput(dev, 0, &mut kbs) == 0 {
            g.pcie_tx_mbs = Some(kbs as f64 / 1000.0);
        }
        if nvml.get_pcie_throughput(dev, 1, &mut kbs) == 0 {
            g.pcie_rx_mbs = Some(kbs as f64 / 1000.0);
        }
        let mut link = 0u32;
        if nvml.get_pcie_gen(dev, &mut link) == 0 {
            g.pcie_gen = Some(link);
        }
        if nvml.get_pcie_gen_max(dev, &mut link) == 0 {
            g.pcie_gen_max = Some(link);
        }
        if nvml.get_pcie_width(dev, &mut link) == 0 {
            g.pcie_width = Some(link);
        }
        let mut reasons = 0u64;
        if nvml.get_throttle_reasons(dev, &mut reasons) == 0 {
            g.throttle_reasons = throttle_labels(reasons);
        }
        out.push(g);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Windows FFI
// ---------------------------------------------------------------------------------------------

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;

    #[repr(C)]
    #[derive(Default)]
    pub struct FileTime {
        pub low: u32,
        pub high: u32,
    }

    impl FileTime {
        pub fn as_u64(&self) -> u64 {
            ((self.high as u64) << 32) | self.low as u64
        }
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn LoadLibraryW(name: *const u16) -> Handle;
        pub fn GetProcAddress(module: Handle, name: *const u8) -> *mut c_void;
        pub fn FreeLibrary(module: Handle) -> i32;
        pub fn GetSystemTimes(idle: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct Util {
    gpu: u32,
    memory: u32,
}

/// Resolved `nvml*.dll` entry points, one per function we use. All take/return the C types NVML
/// documents; a non-zero return means the value was not available.
#[cfg(windows)]
struct Nvml {
    module: win::Handle,
    get_count: unsafe extern "system" fn(*mut u32) -> i32,
    get_handle: unsafe extern "system" fn(u32, *mut *mut c_void) -> i32,
    get_utilization: unsafe extern "system" fn(*mut c_void, *mut Util) -> i32,
    get_temperature: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
    get_power_usage: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    get_power_limit: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    get_clock: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
    get_pcie_throughput: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
    get_pcie_gen: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    get_pcie_gen_max: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    get_pcie_width: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    get_throttle_reasons: unsafe extern "system" fn(*mut c_void, *mut u64) -> i32,
    get_driver_version: unsafe extern "system" fn(*mut u8, u32) -> i32,
}

#[cfg(windows)]
impl Nvml {
    fn load() -> Option<Nvml> {
        for path in nvml_library_paths() {
            let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: wide is a NUL-terminated UTF-16 path.
            let module = unsafe { win::LoadLibraryW(wide.as_ptr()) };
            if module.is_null() {
                continue;
            }
            match unsafe { Self::resolve(module) } {
                Some(nvml) => return Some(nvml),
                None => {
                    // SAFETY: module is a live handle we loaded and are not keeping.
                    unsafe { win::FreeLibrary(module) };
                }
            }
        }
        None
    }

    /// # Safety
    /// `module` must be a live HMODULE from `LoadLibraryW`.
    unsafe fn resolve(module: win::Handle) -> Option<Nvml> {
        let sym = move |name: &str| unsafe { get_symbol(module, name) };
        // Call nvmlInit_v2 before constructing Nvml so a failure leaves the module fully owned by
        // the caller, which frees it once.
        let init: unsafe extern "system" fn() -> i32 = std::mem::transmute_copy(&sym("nvmlInit_v2")?);
        if unsafe { init() } != 0 {
            return None;
        }
        let nvml = Nvml {
            module,
            get_count: std::mem::transmute_copy(&sym("nvmlDeviceGetCount_v2")?),
            get_handle: std::mem::transmute_copy(&sym("nvmlDeviceGetHandleByIndex_v2")?),
            get_utilization: std::mem::transmute_copy(&sym("nvmlDeviceGetUtilizationRates")?),
            get_temperature: std::mem::transmute_copy(&sym("nvmlDeviceGetTemperature")?),
            get_power_usage: std::mem::transmute_copy(&sym("nvmlDeviceGetPowerUsage")?),
            get_power_limit: std::mem::transmute_copy(&sym("nvmlDeviceGetEnforcedPowerLimit")?),
            get_clock: std::mem::transmute_copy(&sym("nvmlDeviceGetClockInfo")?),
            get_pcie_throughput: std::mem::transmute_copy(&sym("nvmlDeviceGetPcieThroughput")?),
            get_pcie_gen: std::mem::transmute_copy(&sym("nvmlDeviceGetCurrPcieLinkGeneration")?),
            get_pcie_gen_max: std::mem::transmute_copy(&sym("nvmlDeviceGetMaxPcieLinkGeneration")?),
            get_pcie_width: std::mem::transmute_copy(&sym("nvmlDeviceGetCurrPcieLinkWidth")?),
            get_throttle_reasons: std::mem::transmute_copy(&sym("nvmlDeviceGetCurrentClocksThrottleReasons")?),
            get_driver_version: std::mem::transmute_copy(&sym("nvmlSystemGetDriverVersion")?),
        };
        Some(nvml)
    }

    fn get_count(&self, out: *mut u32) -> i32 {
        // SAFETY: the stored pointer is the nvmlDeviceGetCount_v2 we resolved; out is valid.
        unsafe { (self.get_count)(out) }
    }

    fn get_handle(&self, index: u32, out: *mut *mut c_void) -> i32 {
        // SAFETY: stored entry point; out points to a writable device handle slot.
        unsafe { (self.get_handle)(index, out) }
    }

    fn get_utilization(&self, dev: *mut c_void, out: *mut Util) -> i32 {
        // SAFETY: stored entry point; dev is a live handle; out is a writable Util.
        unsafe { (self.get_utilization)(dev, out) }
    }

    fn get_temperature(&self, dev: *mut c_void, sensor: u32, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_temperature)(dev, sensor, out) }
    }

    fn get_power_usage(&self, dev: *mut c_void, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_power_usage)(dev, out) }
    }

    fn get_power_limit(&self, dev: *mut c_void, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_power_limit)(dev, out) }
    }

    fn get_clock(&self, dev: *mut c_void, kind: u32, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_clock)(dev, kind, out) }
    }

    fn get_pcie_throughput(&self, dev: *mut c_void, counter: u32, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_pcie_throughput)(dev, counter, out) }
    }

    fn get_pcie_gen(&self, dev: *mut c_void, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_pcie_gen)(dev, out) }
    }

    fn get_pcie_gen_max(&self, dev: *mut c_void, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_pcie_gen_max)(dev, out) }
    }

    fn get_pcie_width(&self, dev: *mut c_void, out: *mut u32) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_pcie_width)(dev, out) }
    }

    fn get_throttle_reasons(&self, dev: *mut c_void, out: *mut u64) -> i32 {
        // SAFETY: stored entry point; dev is live; out is writable.
        unsafe { (self.get_throttle_reasons)(dev, out) }
    }

    fn driver_version(&self) -> Option<String> {
        let mut buf = [0u8; 80];
        // SAFETY: buf is a valid writable buffer of the length we pass.
        if unsafe { (self.get_driver_version)(buf.as_mut_ptr(), buf.len() as u32) } != 0 {
            return None;
        }
        let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..n]).into_owned())
    }
}

// SAFETY: NVML is thread-safe; the handle and entry points are plain pointers into the loaded DLL.
#[cfg(windows)]
unsafe impl Send for Nvml {}
#[cfg(windows)]
unsafe impl Sync for Nvml {}

#[cfg(windows)]
impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: module is the handle we loaded and have not freed yet.
        unsafe { win::FreeLibrary(self.module) };
    }
}

/// The usual places `nvml.dll` hides, in the order we try them: PATH, the system directory, then
/// the NVSMI folder.
#[cfg(windows)]
fn nvml_library_paths() -> Vec<String> {
    let mut paths = vec!["nvml.dll".to_owned(), "C:\\Windows\\System32\\nvml.dll".to_owned()];
    if let Ok(program_files) = std::env::var("ProgramFiles") {
        paths.push(format!("{program_files}\\NVIDIA Corporation\\NVSMI\\nvml.dll"));
    }
    paths
}

/// # Safety
/// `module` must be a live HMODULE.
#[cfg(windows)]
unsafe fn get_symbol(module: win::Handle, name: &str) -> Option<*mut c_void> {
    let mut bytes = name.as_bytes().to_vec();
    bytes.push(0);
    // SAFETY: module is live and bytes is NUL-terminated.
    let ptr = unsafe { win::GetProcAddress(module, bytes.as_ptr()) };
    (!ptr.is_null()).then_some(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_does_not_panic() {
        let mut sampler = Sampler::new();
        let _ = sampler.sample();
        let _ = sampler.sample();
    }

    #[test]
    fn throttle_bitmask_becomes_labels() {
        assert_eq!(throttle_labels(0), Vec::<&'static str>::new());
        assert_eq!(throttle_labels(0x4), vec!["power cap"]);
        assert_eq!(throttle_labels(0x1 | 0x40), vec!["idle", "thermal"]);
        assert_eq!(throttle_labels(0x20 | 0x40), vec!["thermal"]);
    }
}
