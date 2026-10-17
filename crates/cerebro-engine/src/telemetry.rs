//! Live hardware telemetry for the Monitor: CPU load and, per NVIDIA GPU, utilization, memory use,
//! temperature, power, clocks and PCIe throughput, like nvidia-smi plus Task Manager. Instant readings
//! (load, temperature, power, clocks, VRAM) are fresh on every call, so the UI can poll at 3 Hz; rates
//! (CPU %, PCIe and disk MB/s) are measured over a 1-second window and refresh once a second. Read from the OS and the driver, never assumed: anything the machine will not
//! tell us comes back as `None`, never as an error and never as a panic.

#[cfg(windows)]
use std::ffi::c_void;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct GpuSample {
    pub index: u32,
    pub name: Option<String>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
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
    /// System-wide physical disk throughput over the last second (all disks).
    pub disk_read_mbs: Option<f64>,
    pub disk_write_mbs: Option<f64>,
    pub gpus: Vec<GpuSample>,
    /// True when this call closed a new 1-second rate window; history appends exactly then.
    pub new_second: bool,
}

/// Rates are re-measured when at least this long has passed (slack for timer jitter at 3 Hz polls).
const RATE_WINDOW: std::time::Duration = std::time::Duration::from_millis(950);

/// Holds the resolved NVML entry points (loaded once) and the previous CPU-time reading.
pub struct Sampler {
    #[cfg(windows)]
    nvml: Option<Nvml>,
    #[cfg(windows)]
    disk: Option<Disk>,
    cpu: CpuTimes,
    /// The last rate window: when it closed and what it measured.
    rates_at: Option<std::time::Instant>,
    rates: Rates,
    /// Facts that do not change while the app runs, read once per GPU: (name, driver, max PCIe gen).
    fixed: Vec<Fixed>,
}

type Fixed = (Option<String>, Option<String>, Option<u32>);

#[derive(Clone, Default)]
struct Rates {
    cpu_pct: Option<f64>,
    disk_read_mbs: Option<f64>,
    disk_write_mbs: Option<f64>,
    /// Per GPU: PCIe (rx, tx) MB/s.
    pcie: Vec<(Option<f64>, Option<f64>)>,
}

impl Sampler {
    /// Loads NVML if present and prepares the first CPU delta. Cheap and non-failing.
    pub fn new() -> Self {
        Sampler {
            #[cfg(windows)]
            nvml: Nvml::load(),
            #[cfg(windows)]
            disk: Disk::open(),
            cpu: CpuTimes::default(),
            rates_at: None,
            rates: Rates::default(),
            fixed: Vec::new(),
        }
    }

    /// One reading. Never panics; unreadable values are `None`. Cheap enough for 3 calls a second:
    /// the slow PCIe counters (~20 ms each) only run when a rate window closes.
    pub fn sample(&mut self) -> Sample {
        let now = std::time::Instant::now();
        let new_second = self.rates_at.is_none_or(|t| now.duration_since(t) >= RATE_WINDOW);
        #[cfg(windows)]
        let mut gpus = gpu_samples(self.nvml.as_ref(), &mut self.fixed);
        #[cfg(not(windows))]
        let mut gpus: Vec<GpuSample> = Vec::new();
        if new_second {
            self.rates_at = Some(now);
            #[cfg(windows)]
            let (disk_read_mbs, disk_write_mbs) = self.disk.as_mut().map_or((None, None), |d| d.rates());
            #[cfg(not(windows))]
            let (disk_read_mbs, disk_write_mbs) = (None, None);
            #[cfg(windows)]
            let pcie = pcie_rates(self.nvml.as_ref(), gpus.len());
            #[cfg(not(windows))]
            let pcie = Vec::new();
            // The first window has no previous reading, so CPU load starts on the second one.
            let cpu_pct = self.cpu.sample();
            self.rates = Rates { cpu_pct, disk_read_mbs, disk_write_mbs, pcie };
        }
        for (g, (rx, tx)) in gpus.iter_mut().zip(&self.rates.pcie) {
            g.pcie_rx_mbs = *rx;
            g.pcie_tx_mbs = *tx;
        }
        Sample { cpu_pct: self.rates.cpu_pct, disk_read_mbs: self.rates.disk_read_mbs, disk_write_mbs: self.rates.disk_write_mbs, gpus, new_second }
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

/// Instant readings for every GPU; `fixed` caches the facts that never change.
#[cfg(windows)]
fn gpu_samples(nvml: Option<&Nvml>, fixed: &mut Vec<Fixed>) -> Vec<GpuSample> {
    let Some(nvml) = nvml else { return Vec::new() };
    let mut count = 0u32;
    if nvml.get_count(&mut count) != 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut dev = std::ptr::null_mut();
        if nvml.get_handle(index, &mut dev) != 0 || dev.is_null() {
            continue;
        }
        let i = index as usize;
        if fixed.len() <= i {
            fixed.resize(i + 1, (None, None, None));
        }
        if fixed[i].0.is_none() {
            let mut gen_max = 0u32;
            let gen_max = (nvml.get_pcie_gen_max(dev, &mut gen_max) == 0).then_some(gen_max);
            fixed[i] = (nvml.name(dev), nvml.driver_version(), gen_max);
        }
        let (name, driver, gen_max) = fixed[i].clone();
        let mut g = GpuSample { index, name, driver, pcie_gen_max: gen_max, ..Default::default() };
        let mut mem = MemInfo::default();
        if nvml.get_memory(dev, &mut mem) == 0 {
            g.vram_used = Some(mem.used);
            g.vram_total = Some(mem.total);
        }

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
        let mut link = 0u32;
        if nvml.get_pcie_gen(dev, &mut link) == 0 {
            g.pcie_gen = Some(link);
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

/// PCIe (rx, tx) MB/s per GPU. NVML counters: 0 = TX, 1 = RX, in KB/s; each call costs ~20 ms,
/// which is why this runs once per rate window, not on every poll.
#[cfg(windows)]
fn pcie_rates(nvml: Option<&Nvml>, count: usize) -> Vec<(Option<f64>, Option<f64>)> {
    let Some(nvml) = nvml else { return Vec::new() };
    (0..count as u32)
        .map(|index| {
            let mut dev = std::ptr::null_mut();
            if nvml.get_handle(index, &mut dev) != 0 || dev.is_null() {
                return (None, None);
            }
            let mut kbs = 0u32;
            let tx = (nvml.get_pcie_throughput(dev, 0, &mut kbs) == 0).then(|| kbs as f64 / 1000.0);
            let rx = (nvml.get_pcie_throughput(dev, 1, &mut kbs) == 0).then(|| kbs as f64 / 1000.0);
            (rx, tx)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Disk (PDH): system-wide physical disk bytes/s, the same counter Task Manager reads.
// ---------------------------------------------------------------------------------------------

#[cfg(windows)]
struct Disk {
    query: win::Handle,
    read: win::Handle,
    write: win::Handle,
    primed: bool,
}

#[cfg(windows)]
impl Disk {
    fn open() -> Option<Disk> {
        let mut query = std::ptr::null_mut();
        // SAFETY: valid out-pointer; a NULL data source means live data.
        if unsafe { win::PdhOpenQueryW(std::ptr::null(), 0, &mut query) } != 0 {
            return None;
        }
        let add = |path: &str| -> Option<win::Handle> {
            let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
            let mut counter = std::ptr::null_mut();
            // SAFETY: query is open; wide is NUL-terminated; counter is a valid out-pointer.
            (unsafe { win::PdhAddEnglishCounterW(query, wide.as_ptr(), 0, &mut counter) } == 0).then_some(counter)
        };
        let (Some(read), Some(write)) = (add(r"\PhysicalDisk(_Total)\Disk Read Bytes/sec"), add(r"\PhysicalDisk(_Total)\Disk Write Bytes/sec")) else {
            // SAFETY: query is open and owned here.
            unsafe { win::PdhCloseQuery(query) };
            return None;
        };
        // SAFETY: open query. The first collection only primes the rate counters.
        let primed = unsafe { win::PdhCollectQueryData(query) } == 0;
        Some(Disk { query, read, write, primed })
    }

    /// (read, write) MB/s since the previous call.
    fn rates(&mut self) -> (Option<f64>, Option<f64>) {
        // SAFETY: open query.
        let ok = unsafe { win::PdhCollectQueryData(self.query) } == 0;
        let primed = std::mem::replace(&mut self.primed, ok);
        if !ok || !primed {
            return (None, None);
        }
        let value = |c: win::Handle| -> Option<f64> {
            let mut v = win::PdhFmtCounterValue::default();
            // SAFETY: c belongs to this query; v is a writable PDH_FMT_COUNTERVALUE.
            let rc = unsafe { win::PdhGetFormattedCounterValue(c, win::PDH_FMT_DOUBLE, std::ptr::null_mut(), &mut v) };
            (rc == 0 && v.status == 0).then(|| v.value / 1e6)
        };
        (value(self.read), value(self.write))
    }
}

// SAFETY: the PDH query is only touched through &mut self, so one thread at a time.
#[cfg(windows)]
unsafe impl Send for Disk {}

#[cfg(windows)]
impl Drop for Disk {
    fn drop(&mut self) {
        // SAFETY: query is the handle we opened.
        unsafe { win::PdhCloseQuery(self.query) };
    }
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

    pub const PDH_FMT_DOUBLE: u32 = 0x0000_0200;

    /// PDH_FMT_COUNTERVALUE read through the double arm of its union.
    #[repr(C)]
    #[derive(Default)]
    pub struct PdhFmtCounterValue {
        pub status: u32,
        pub value: f64,
    }

    #[link(name = "pdh")]
    extern "system" {
        pub fn PdhOpenQueryW(source: *const u16, user: usize, query: *mut Handle) -> u32;
        pub fn PdhAddEnglishCounterW(query: Handle, path: *const u16, user: usize, counter: *mut Handle) -> u32;
        pub fn PdhCollectQueryData(query: Handle) -> u32;
        pub fn PdhGetFormattedCounterValue(counter: Handle, format: u32, kind: *mut u32, value: *mut PdhFmtCounterValue) -> u32;
        pub fn PdhCloseQuery(query: Handle) -> u32;
    }
}

/// nvmlMemory_t, in bytes.
#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct MemInfo {
    total: u64,
    free: u64,
    used: u64,
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
    get_memory: unsafe extern "system" fn(*mut c_void, *mut MemInfo) -> i32,
    get_name: unsafe extern "system" fn(*mut c_void, *mut u8, u32) -> i32,
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
            get_memory: std::mem::transmute_copy(&sym("nvmlDeviceGetMemoryInfo")?),
            get_name: std::mem::transmute_copy(&sym("nvmlDeviceGetName")?),
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

    fn get_memory(&self, dev: *mut c_void, out: *mut MemInfo) -> i32 {
        // SAFETY: stored entry point; dev is live; out is a writable nvmlMemory_t.
        unsafe { (self.get_memory)(dev, out) }
    }

    fn name(&self, dev: *mut c_void) -> Option<String> {
        let mut buf = [0u8; 96];
        // SAFETY: stored entry point; dev is live; buf is writable for the length we pass.
        if unsafe { (self.get_name)(dev, buf.as_mut_ptr(), buf.len() as u32) } != 0 {
            return None;
        }
        let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..n]).into_owned())
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
    fn rates_close_once_per_second() {
        let mut sampler = Sampler::new();
        assert!(sampler.sample().new_second, "the first call opens a window");
        assert!(!sampler.sample().new_second, "a call right after reuses it");
        std::thread::sleep(RATE_WINDOW);
        let s = sampler.sample();
        assert!(s.new_second);
        #[cfg(windows)]
        assert!(s.cpu_pct.is_some(), "a full window measures CPU load");
    }

    #[test]
    fn throttle_bitmask_becomes_labels() {
        assert_eq!(throttle_labels(0), Vec::<&'static str>::new());
        assert_eq!(throttle_labels(0x4), vec!["power cap"]);
        assert_eq!(throttle_labels(0x1 | 0x40), vec!["idle", "thermal"]);
        assert_eq!(throttle_labels(0x20 | 0x40), vec!["thermal"]);
    }
}
