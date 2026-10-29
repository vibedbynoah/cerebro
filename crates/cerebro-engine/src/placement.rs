//! Where a model's weights go: GPU memory, system RAM, or read from the drive as they are needed.
//!
//! Cerebro keeps every byte in the fastest tier it fits in. For a dense model, in order:
//! 1. every layer on the GPU in the int8 layout (fastest prompt reading, about 1.13 bytes/weight);
//! 2. every layer on the GPU in the file's own K-quant format (about half the bytes, decode at full
//!    speed, prompt reading somewhat slower);
//! 3. split: as many leading layers as fit on the GPU in their own format, the rest on the CPU.
//!
//! Weights on the CPU are memory-mapped from the file: whatever does not fit in RAM is read from the
//! drive again for each token, which is why the drive matters (NVMe, SATA SSD, hard disk, USB or SD
//! card). The plan says how many bytes each tier serves per token and estimates decode speed from
//! each tier's bandwidth, so the user sees the consequence before loading.
//!
//! Mixture-of-experts models already run hybrid (gpu.rs): attention on the GPU, experts in RAM with
//! the most-used ones cached in VRAM. Their plan accounts the same tiers.

use serde::Serialize;

/// The kind of drive a model file is on, as Windows reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Drive {
    Nvme,
    SataSsd,
    HardDisk,
    /// USB flash drives and memory cards (SD, microSD, MMC).
    Removable,
    Unknown,
}

impl Drive {
    /// Sustained read speed for page faults over a memory-mapped file, bytes/s. Conservative: page
    /// faults read in smaller pieces than a sequential copy.
    pub fn read_bps(self) -> f64 {
        match self {
            Drive::Nvme => 2.0e9,
            Drive::SataSsd => 4.0e8,
            Drive::HardDisk => 8.0e7,
            Drive::Removable => 3.0e7,
            Drive::Unknown => 2.0e8,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Drive::Nvme => "NVMe SSD",
            Drive::SataSsd => "SATA SSD",
            Drive::HardDisk => "hard disk",
            Drive::Removable => "USB drive or memory card",
            Drive::Unknown => "drive",
        }
    }
}

/// What the machine offers, in bytes and bytes/s.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Machine {
    /// Free VRAM on the GPU the engine uses (0 without one).
    pub vram_free: u64,
    /// RAM available to the engine.
    pub ram_available: u64,
    pub drive: Drive,
    /// Effective read bandwidth of decode on each tier (measured values when known).
    pub gpu_bps: f64,
    pub ram_bps: f64,
    /// Layers that do not fit in RAM may be read from the drive on every pass (stream.rs).
    pub drive_tier: bool,
}

impl Machine {
    /// Bandwidths measured on this project's reference laptop (i9-12900H + 3070 Ti 8 GB): the GPU
    /// decodes at ~300-390 GB/s effective; on Qwen3-14B Q4_K_M the CPU read 28 GB/s alone (3.2 tok/s)
    /// and ~40 GB/s as the tail of a 31/40 split (11.8 tok/s measured, 9.2 estimated). Callers with
    /// their own measurements override.
    pub fn new(vram_free: u64, ram_available: u64, drive: Drive) -> Self {
        Machine { vram_free, ram_available, drive, gpu_bps: 3.0e11, ram_bps: 3.0e10, drive_tier: true }
    }
}

/// One layer's cost on each side.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LayerCost {
    /// Weights in the int8 layout (GPU, fastest prefill).
    pub int8: u64,
    /// Weights in the file's own format (GPU native layout, and the CPU's mapped bytes).
    pub native: u64,
    /// KV cache on the GPU (f16) for the whole context.
    pub kv_gpu: u64,
    /// KV cache on the CPU (f32) for the whole context.
    pub kv_cpu: u64,
    /// The native layout is possible on the GPU for every matrix of the layer.
    pub native_ok: bool,
}

/// A dense model's costs.
#[derive(Debug, Clone, Serialize)]
pub struct ModelCost {
    pub layers: Vec<LayerCost>,
    /// Output head (and embeddings, when they are separate), int8 and native.
    pub head_int8: u64,
    pub head_native: u64,
    /// Scratch the GPU needs for decoding and batched prompt reading.
    pub scratch: u64,
    /// VRAM the output head takes when a split puts it on the GPU (0: a split keeps it on the CPU).
    pub split_head: u64,
    /// The whole file (the CPU maps all of it).
    pub file: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Layout {
    /// Every layer on the GPU in the int8 layout.
    GpuInt8,
    /// Every layer on the GPU in the file's own format.
    GpuNative,
    /// The first `gpu_layers` on the GPU (own format), the rest and the output head on the CPU.
    Split,
    /// Nothing on the GPU.
    Cpu,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    pub layout: Layout,
    pub gpu_layers: usize,
    pub total_layers: usize,
    /// Bytes on the GPU (weights, KV, scratch).
    pub vram_bytes: u64,
    /// Weights the CPU reads per token, and how many of them stay in RAM / come from the drive.
    pub cpu_bytes: u64,
    pub ram_resident: u64,
    pub drive_bytes_per_token: u64,
    /// Trailing processor layers read from the drive on every pass so the rest fits in RAM.
    pub drive_layers: usize,
    /// Estimated decode speed (tokens/s): an estimate from tier bandwidths, not a measurement.
    pub est_decode_tps: f64,
    pub drive: Drive,
    /// Plain sentences for the user, most important first.
    pub notes: Vec<String>,
    /// A split puts the output head on the GPU (`ModelCost::split_head` > 0 and it pays).
    pub head_on_gpu: bool,
}

/// RAM the drive tier's ring of buffers takes for layers of `biggest` bytes (stream.rs).
pub fn drive_ring_bytes(biggest: u64) -> u64 {
    let n = ((512u64 << 20) / biggest.max(1)).clamp(2, 8);
    n * biggest
}

/// How many trailing processor layers to read from the drive on every pass so that the rest of
/// the processor's share, plus the drive tier's buffers, fits in `ram_budget`: 0 when everything
/// fits. `layers` is each layer's weight bytes, `head` the output head's; layers before
/// `gpu_layers` are on the GPU. Streaming every processor layer of a model with nothing on the GPU
/// streams the head too. When even that does not fit, every processor layer is streamed (RAM then
/// holds only the buffers and whatever the OS keeps).
pub fn drive_layers(layers: &[u64], head: u64, gpu_layers: usize, ram_budget: u64) -> usize {
    let cpu = &layers[gpu_layers.min(layers.len())..];
    let total: u64 = cpu.iter().sum::<u64>() + head;
    if total <= ram_budget || cpu.is_empty() {
        return 0;
    }
    let ring = drive_ring_bytes(cpu.iter().copied().max().unwrap_or(0).max(if gpu_layers == 0 { head } else { 0 }));
    let mut resident = total;
    for k in 1..=cpu.len() {
        resident -= cpu[cpu.len() - k];
        let r = if k == cpu.len() && gpu_layers == 0 { resident - head } else { resident };
        if r + ring <= ram_budget {
            return k;
        }
    }
    cpu.len()
}

/// VRAM left free for the driver, CUDA graphs and fragmentation (the buffers a model needs are
/// counted in its plan, including the prompt conversion scratch).
pub const VRAM_MARGIN: u64 = 256 << 20;
/// RAM left for the OS and the app when planning (a thirty-second of 32 GB; `gguf::prefetch_room`
/// keeps a thirty-second of whatever is installed).
pub const RAM_MARGIN: u64 = 1 << 30;

/// VRAM a load leaves free on a card of `total` bytes: a sixteenth (512 MB on 8 GB), never less
/// than `VRAM_MARGIN`. Everything else holds weights, experts and the cache; what does not fit
/// stays in RAM, and past RAM on the drive.
pub fn vram_reserve(total: u64) -> u64 {
    (total / 16).max(VRAM_MARGIN)
}

/// How far a dense split may plan past free VRAM on Windows, where the driver pages idle GPU
/// allocations to shared system memory instead of failing them ("sysmem fallback"). The buffers
/// only prompt reading uses (the streamed-layer slots, conversion scratch) sit idle while a reply
/// is written, so they are what gets paged, and the layers read every token stay in VRAM.
/// The prompt slots grow with the layer, so the allowance shrinks with it: 3/32 of the card times
/// (167 MB / layer bytes)^3. Measured on an 8 GB RTX 3070 Ti, 512-token prompt:
/// - Qwen2.5-14B Q4_K_M (167 MB layers): 768 MB put 44 layers on the card instead of 40 (19.3
///   instead of 15.5 tok/s writing, 923 instead of 869 reading); 1 GB put 46 there but reading fell
///   to 436 as the slots themselves were paged; 1.5 GB thrashed.
/// - Qwen3.8-27B Q4_K_M (245 MB layers): 256 MB gave 29 layers instead of 28 (4.57 instead of 4.06
///   writing, 357 instead of 349 reading); 512 MB dropped reading to 111.
/// `CEREBRO_SPILL_MB` sets it outright.
pub fn spill_allowance(total: u64, layer_bytes: u64) -> u64 {
    if let Some(mb) = std::env::var("CEREBRO_SPILL_MB").ok().and_then(|v| v.parse::<u64>().ok()) {
        return mb << 20;
    }
    if !cfg!(windows) || SPILL_OFF.load(std::sync::atomic::Ordering::Relaxed) || layer_bytes == 0 {
        return 0;
    }
    let r = (167e6 / layer_bytes as f64).min(1.0);
    (total as f64 * 3.0 / 32.0 * r * r * r) as u64
}

/// Set when a load with the spill allowance failed (a driver with sysmem fallback turned off), so
/// the retry plans within VRAM.
pub static SPILL_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The plan for a dense model. `force_gpu_layers` (the "GPU layers" switch) overrides the choice,
/// clamped to what fits.
/// What a split's GPU head costs per token in synchronisation and transfers, in seconds.
const HEAD_TRIP_S: f64 = 1.5e-3;

pub fn plan_dense(cost: &ModelCost, m: &Machine, force_gpu_layers: Option<usize>) -> Plan {
    let n = cost.layers.len();
    let budget = m.vram_free.saturating_sub(VRAM_MARGIN);
    let sum = |f: &dyn Fn(&LayerCost) -> u64, r: std::ops::Range<usize>| -> u64 { cost.layers[r].iter().map(f).sum() };
    let all_native_ok = cost.layers.iter().all(|l| l.native_ok);
    let full_int8 = sum(&|l| l.int8 + l.kv_gpu, 0..n) + cost.head_int8 + cost.scratch;
    let full_native = sum(&|l| (if l.native_ok { l.native } else { l.int8 }) + l.kv_gpu, 0..n) + cost.head_native + cost.scratch;
    // In a split the GPU holds layers, their KV and scratch, and the head when `split_head` says so.
    let layer_gpu = |l: &LayerCost| (if l.native_ok { l.native } else { l.int8 }) + l.kv_gpu;
    let fit = |reserve: u64| {
        let mut used = cost.scratch + reserve;
        let mut k = 0;
        for l in &cost.layers {
            if used + layer_gpu(l) > budget {
                break;
            }
            used += layer_gpu(l);
            k += 1;
        }
        k
    };
    // The head on the GPU only when the card then holds more of what every token reads, by more
    // than the extra round trip it costs per token (the residual up after the processor's layers,
    // the logits back: about 1.5 ms under WDDM, charged as that much RAM reading). Qwen2.5-14B
    // Q4_K_M on 8 GB, 512 + 64 tokens: head + 37 layers (6.37 GiB of weights) wrote 16.57 tok/s,
    // 41 layers (6.36 GiB) with the head on the processor 16.96.
    let weights = |k: usize| sum(&|l| if l.native_ok { l.native } else { l.int8 }, 0..k);
    let (fit_head, fit_plain) = (fit(cost.split_head), fit(0));
    let trip = (m.ram_bps * HEAD_TRIP_S) as u64;
    let head_pays = cost.split_head > 0 && weights(fit_head) + cost.split_head > weights(fit_plain) + trip;
    let fit_layers = if head_pays { fit_head } else { fit_plain };
    if std::env::var_os("CEREBRO_PLAN_DEBUG").is_some() {
        let g = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "plan: budget {:.2} GiB, full int8 {:.2}, full native {:.2} (all native {all_native_ok}), layers native {:.2}, kv {:.2}, head int8 {:.2} native {:.2}, scratch {:.2}, split head {:.2}, fit {fit_layers}",
            g(budget), g(full_int8), g(full_native), g(sum(&|l| if l.native_ok { l.native } else { l.int8 }, 0..n)), g(sum(&|l| l.kv_gpu, 0..n)), g(cost.head_int8), g(cost.head_native), g(cost.scratch), g(cost.split_head)
        );
    }
    let (layout, gpu_layers) = match force_gpu_layers {
        Some(k) if k < n => {
            let k = k.min(fit_plain);
            (if k == 0 { Layout::Cpu } else { Layout::Split }, k)
        }
        _ if m.vram_free == 0 => (Layout::Cpu, 0),
        _ if full_int8 <= budget => (Layout::GpuInt8, n),
        _ if all_native_ok && full_native <= budget => (Layout::GpuNative, n),
        _ if fit_layers == 0 => (Layout::Cpu, 0),
        _ => (Layout::Split, fit_layers.min(n.saturating_sub(1))),
    };
    // What the CPU reads per token: its layers and, unless the GPU holds it, the head. A forced
    // layer count keeps the head on the GPU while both still fit.
    let head_on_gpu = layout == Layout::Split
        && cost.split_head > 0
        && match force_gpu_layers {
            Some(k) if k < n => gpu_layers <= fit_head,
            _ => head_pays,
        };
    let vram_bytes = match layout {
        Layout::GpuInt8 => full_int8,
        Layout::GpuNative => full_native,
        Layout::Split => sum(&layer_gpu, 0..gpu_layers) + cost.scratch + if head_on_gpu { cost.split_head } else { 0 },
        Layout::Cpu => 0,
    };
    let cpu_weights = match layout {
        Layout::GpuInt8 | Layout::GpuNative => 0,
        _ => sum(&|l| l.native, gpu_layers..n) + if head_on_gpu { 0 } else { cost.head_native },
    };
    let cpu_kv = match layout {
        Layout::GpuInt8 | Layout::GpuNative => 0,
        _ => sum(&|l| l.kv_cpu, gpu_layers..n),
    };
    let ram_budget = m.ram_available.saturating_sub(RAM_MARGIN).saturating_sub(cpu_kv);
    // With the drive tier, the last layers that do not fit are read in large unbuffered requests
    // at the drive's full speed; without it, the OS pages the overflow in 4 KB at a time.
    let drive_layers = if m.drive_tier && cpu_weights > ram_budget { drive_layers(&cost.layers.iter().map(|l| l.native).collect::<Vec<_>>(), cost.head_native, gpu_layers, ram_budget) } else { 0 };
    let (ram_resident, from_drive) = if drive_layers > 0 {
        let streamed = sum(&|l| l.native, n - drive_layers..n) + if drive_layers == n { cost.head_native } else { 0 };
        (cpu_weights - streamed, streamed)
    } else {
        let r = cpu_weights.min(ram_budget);
        (r, cpu_weights - r)
    };
    let gpu_read = match layout {
        Layout::GpuInt8 | Layout::GpuNative => vram_bytes.saturating_sub(cost.scratch).saturating_sub(sum(&|l| l.kv_gpu, 0..n)),
        Layout::Split => sum(&|l| if l.native_ok { l.native } else { l.int8 }, 0..gpu_layers),
        Layout::Cpu => 0,
    };
    let secs = gpu_read as f64 / m.gpu_bps + ram_resident as f64 / m.ram_bps + from_drive as f64 / m.drive.read_bps();
    let est = if secs > 0.0 { 1.0 / secs } else { 0.0 };
    let mut notes = Vec::new();
    match layout {
        Layout::GpuInt8 => notes.push(format!("Every layer fits on the GPU ({:.1} GiB).", gib(vram_bytes))),
        Layout::GpuNative => notes.push(format!("Every layer fits on the GPU in the file's own format ({:.1} GiB); prompt reading is a little slower than with more VRAM.", gib(vram_bytes))),
        Layout::Split => notes.push(format!("{gpu_layers} of {n} layers on the GPU ({:.1} GiB), the rest on the processor ({:.1} GiB).", gib(vram_bytes), gib(cpu_weights))),
        Layout::Cpu if m.vram_free == 0 => notes.push("No supported GPU: everything runs on the processor.".to_owned()),
        Layout::Cpu => notes.push("Not even one layer fits in the free GPU memory: everything runs on the processor.".to_owned()),
    }
    if drive_layers > 0 {
        notes.push(format!(
            "{drive_layers} of the processor's layers ({:.1} GiB) do not fit in RAM and are read from the {} for every token at its full speed (about {:.1} s per token from the drive alone); {:.1} GiB stays in RAM. Freeing RAM, a smaller quant or a shorter context is faster.",
            gib(from_drive),
            m.drive.label(),
            from_drive as f64 / m.drive.read_bps(),
            gib(ram_resident)
        ));
    } else if from_drive > 0 {
        let per_tok = from_drive as f64 / m.drive.read_bps();
        notes.push(format!(
            "{:.1} GiB does not fit in RAM and is read from the {} for every token (about {:.1} s per token from the drive alone). Freeing RAM, a smaller quant or a shorter context helps most.",
            gib(from_drive),
            m.drive.label(),
            per_tok
        ));
    } else if cpu_weights > 0 {
        notes.push(format!("The processor's share stays in RAM ({:.1} GiB); the {} is read only while loading.", gib(ram_resident), m.drive.label()));
    }
    Plan { layout, gpu_layers, total_layers: n, vram_bytes, cpu_bytes: cpu_weights, ram_resident, drive_bytes_per_token: from_drive, drive_layers, est_decode_tps: est, drive: m.drive, notes, head_on_gpu }
}

/// The plan for a mixture-of-experts model: attention, router and head on the GPU, experts in RAM,
/// as many experts as fit cached in VRAM. `expert_bytes` is all experts; `dense_gpu` the rest.
pub fn plan_moe(dense_gpu: u64, expert_bytes: u64, active_fraction: f64, m: &Machine) -> Plan {
    let budget = m.vram_free.saturating_sub(VRAM_MARGIN);
    let gpu_ok = m.vram_free > 0 && dense_gpu <= budget;
    if std::env::var_os("CEREBRO_PLAN_DEBUG").is_some() {
        eprintln!("plan (experts): budget {:.2} GiB, shared layers {:.2} GiB, experts {:.2} GiB", gib(budget), gib(dense_gpu), gib(expert_bytes));
    }
    let cache = if gpu_ok { budget - dense_gpu } else { 0 }.min(expert_bytes);
    let ram_budget = m.ram_available.saturating_sub(RAM_MARGIN);
    let ram_resident = expert_bytes.min(ram_budget);
    let from_drive = expert_bytes - ram_resident;
    // Per token only the active experts are read; cached ones from VRAM (assume the cache serves
    // its share of uses), the rest from RAM or the drive in proportion.
    let active = expert_bytes as f64 * active_fraction;
    let hit = if expert_bytes > 0 { cache as f64 / expert_bytes as f64 } else { 0.0 };
    let miss = active * (1.0 - hit);
    let drive_share = if expert_bytes > 0 { from_drive as f64 / expert_bytes as f64 } else { 0.0 };
    let secs = (if gpu_ok { dense_gpu as f64 } else { 0.0 }) / m.gpu_bps + active * hit / m.gpu_bps + miss * (1.0 - drive_share) / m.ram_bps + miss * drive_share / m.drive.read_bps() + if gpu_ok { 0.0 } else { dense_gpu as f64 / m.ram_bps };
    let mut notes = vec![if gpu_ok {
        format!("Attention on the GPU; {:.1} of {:.1} GiB of experts cached in VRAM, the rest in RAM.", gib(cache), gib(expert_bytes))
    } else {
        "The model's shared layers do not fit on the GPU: everything runs on the processor.".to_owned()
    }];
    if from_drive > 0 {
        notes.push(format!("{:.1} GiB of experts does not fit in RAM and is read from the {} when those experts are used.", gib(from_drive), m.drive.label()));
    }
    Plan {
        layout: if gpu_ok { Layout::Split } else { Layout::Cpu },
        gpu_layers: 0,
        total_layers: 0,
        vram_bytes: if gpu_ok { dense_gpu + cache } else { 0 },
        cpu_bytes: expert_bytes,
        ram_resident,
        drive_bytes_per_token: (miss * drive_share) as u64,
        drive_layers: 0,
        est_decode_tps: if secs > 0.0 { 1.0 / secs } else { 0.0 },
        drive: m.drive,
        notes,
        head_on_gpu: false,
    }
}

/// The drive a file is on (Windows: the volume's storage bus and seek penalty).
pub fn drive_of(path: &std::path::Path) -> Drive {
    #[cfg(windows)]
    {
        win::drive_of(path).unwrap_or(Drive::Unknown)
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Drive::Unknown
    }
}

#[cfg(windows)]
mod win {
    use super::Drive;
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    const GENERIC_NONE: u32 = 0;
    const FILE_SHARE_READ_WRITE: u32 = 0x1 | 0x2;
    const OPEN_EXISTING: u32 = 3;
    const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D_1400;
    const INVALID: isize = -1;

    #[repr(C)]
    struct Query {
        property_id: u32,
        query_type: u32,
        extra: [u8; 4],
    }

    #[repr(C)]
    #[derive(Default)]
    struct SeekPenalty {
        version: u32,
        size: u32,
        incurs_seek_penalty: u8,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(name: *const u16, access: u32, share: u32, sec: *mut c_void, disp: u32, flags: u32, template: isize) -> isize;
        fn DeviceIoControl(h: isize, code: u32, inb: *const c_void, inlen: u32, outb: *mut c_void, outlen: u32, ret: *mut u32, ov: *mut c_void) -> i32;
        fn CloseHandle(h: isize) -> i32;
        fn GetVolumePathNameW(file: *const u16, vol: *mut u16, len: u32) -> i32;
    }

    pub fn drive_of(path: &std::path::Path) -> Option<Drive> {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let mut vol = [0u16; 260];
        // SAFETY: NUL-terminated input; the output buffer length is passed.
        if unsafe { GetVolumePathNameW(wide.as_ptr(), vol.as_mut_ptr(), vol.len() as u32) } == 0 {
            return None;
        }
        let v = String::from_utf16_lossy(&vol[..vol.iter().position(|&c| c == 0)?]);
        // "C:\" -> "\\.\C:" (a volume handle answers the storage queries for its disk).
        let letter = v.trim_end_matches('\\');
        if letter.len() != 2 || !letter.ends_with(':') {
            return None;
        }
        let dev: Vec<u16> = format!(r"\\.\{letter}").encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: valid NUL-terminated device name; no access rights are needed for these queries.
        let h = unsafe { CreateFileW(dev.as_ptr(), GENERIC_NONE, FILE_SHARE_READ_WRITE, std::ptr::null_mut(), OPEN_EXISTING, 0, 0) };
        if h == INVALID || h == 0 {
            return None;
        }
        let query = |id: u32, out: &mut [u8]| -> bool {
            let q = Query { property_id: id, query_type: 0, extra: [0; 4] };
            let mut ret = 0u32;
            // SAFETY: q and out are valid for the given sizes; synchronous call.
            unsafe { DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, (&q as *const Query).cast(), std::mem::size_of::<Query>() as u32, out.as_mut_ptr().cast(), out.len() as u32, &mut ret, std::ptr::null_mut()) != 0 && ret >= 8 }
        };
        // StorageDeviceProperty (0): STORAGE_DEVICE_DESCRIPTOR; BusType is the u32 at offset 28,
        // RemovableMedia the byte at offset 10.
        let mut desc = [0u8; 1024];
        let bus = query(0, &mut desc).then(|| u32::from_le_bytes([desc[28], desc[29], desc[30], desc[31]]));
        let removable = desc[10] != 0;
        // StorageDeviceSeekPenaltyProperty (7).
        let mut sp = [0u8; std::mem::size_of::<SeekPenalty>()];
        let seek = query(7, &mut sp).then(|| sp[8] != 0);
        // SAFETY: h is a handle this function opened.
        unsafe { CloseHandle(h) };
        // BusType: 7 USB, 12 SD, 13 MMC, 17 NVMe, 11 SATA, 3 ATA, 1 SCSI, 10 SAS, 8 RAID.
        Some(match (bus, seek) {
            (Some(7 | 12 | 13), _) => Drive::Removable,
            _ if removable => Drive::Removable,
            (Some(17), _) => Drive::Nvme,
            (_, Some(true)) => Drive::HardDisk,
            (Some(11 | 3 | 1 | 10 | 8), Some(false)) => Drive::SataSsd,
            (_, Some(false)) => Drive::SataSsd,
            _ => Drive::Unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    /// A 64-layer dense model: 0.25 GB/layer native, 0.47 int8, KV 32 MB/layer on the GPU.
    fn model27b() -> ModelCost {
        ModelCost {
            layers: (0..64).map(|_| LayerCost { int8: 470 << 20, native: 250 << 20, kv_gpu: 32 << 20, kv_cpu: 64 << 20, native_ok: true }).collect(),
            head_int8: 1200 << 20,
            head_native: 640 << 20,
            scratch: 200 << 20,
            split_head: 0,
            file: 16 * GB,
        }
    }

    #[test]
    fn small_models_go_fully_on_the_gpu_in_the_fast_layout() {
        let mut m = model27b();
        m.layers.truncate(8);
        let p = plan_dense(&m, &Machine::new(8 * GB, 16 * GB, Drive::Nvme), None);
        assert_eq!((p.layout, p.gpu_layers), (Layout::GpuInt8, 8));
        assert_eq!(p.drive_bytes_per_token, 0);
    }

    #[test]
    fn native_format_when_int8_does_not_fit_but_the_file_does() {
        // 20 layers: 11.2 GiB as int8, 6.3 GiB in the file's format, 7.4 GiB usable.
        let mut m = model27b();
        m.layers.truncate(20);
        let p = plan_dense(&m, &Machine::new(8 * GB, 16 * GB, Drive::Nvme), None);
        assert_eq!(p.layout, Layout::GpuNative);
    }

    #[test]
    fn split_on_an_8_gb_laptop_with_everything_else_in_ram() {
        let p = plan_dense(&model27b(), &Machine::new(7 * GB + GB / 2, 20 * GB, Drive::Nvme), None);
        assert_eq!(p.layout, Layout::Split);
        assert!(p.gpu_layers > 20 && p.gpu_layers < 30, "{}", p.gpu_layers);
        assert_eq!(p.drive_bytes_per_token, 0);
        assert!(p.vram_bytes <= 7 * GB + GB / 2 - VRAM_MARGIN);
        assert!(p.notes[0].contains("of 64 layers on the GPU"));
        // More GPU layers than fit are clamped; zero means the processor only.
        assert!(plan_dense(&model27b(), &Machine::new(7 * GB + GB / 2, 20 * GB, Drive::Nvme), Some(60)).gpu_layers <= p.gpu_layers);
        assert_eq!(plan_dense(&model27b(), &Machine::new(7 * GB + GB / 2, 20 * GB, Drive::Nvme), Some(0)).layout, Layout::Cpu);
    }

    #[test]
    fn what_ram_cannot_hold_comes_from_the_drive_and_slow_drives_say_so() {
        let ssd = plan_dense(&model27b(), &Machine::new(0, 8 * GB, Drive::Nvme), None);
        assert_eq!(ssd.layout, Layout::Cpu);
        assert!(ssd.drive_bytes_per_token > 8 * GB);
        let sd = plan_dense(&model27b(), &Machine::new(0, 8 * GB, Drive::Removable), None);
        assert!(sd.est_decode_tps < ssd.est_decode_tps / 10.0);
        assert!(sd.notes[1].contains("USB drive or memory card"));
        assert!(sd.notes[1].contains("for every token"));
    }

    #[test]
    fn more_vram_is_never_slower() {
        let m = model27b();
        let mut last = 0.0;
        for vram in [0, 2, 4, 6, 8, 12, 16, 24, 32] {
            let p = plan_dense(&m, &Machine::new(vram * GB, 24 * GB, Drive::SataSsd), None);
            assert!(p.est_decode_tps >= last * 0.999, "{vram} GB: {} < {last}", p.est_decode_tps);
            last = p.est_decode_tps;
        }
    }

    #[test]
    fn layers_without_a_native_gpu_kernel_use_int8() {
        let mut m = model27b();
        for l in m.layers.iter_mut() {
            l.native_ok = false;
        }
        let p = plan_dense(&m, &Machine::new(7 * GB + GB / 2, 20 * GB, Drive::Nvme), None);
        assert_eq!(p.layout, Layout::Split);
        assert!(p.gpu_layers < 16, "int8 layers are bigger: {}", p.gpu_layers);
    }

    #[test]
    fn moe_plans_cache_experts_and_count_the_drive() {
        let p = plan_moe(2 * GB, 16 * GB, 8.0 / 128.0, &Machine::new(8 * GB, 20 * GB, Drive::Nvme));
        assert_eq!(p.layout, Layout::Split);
        assert!(p.vram_bytes > 2 * GB);
        let tight = plan_moe(2 * GB, 60 * GB, 8.0 / 128.0, &Machine::new(8 * GB, 32 * GB, Drive::HardDisk));
        assert!(tight.notes.iter().any(|n| n.contains("hard disk")));
    }

    #[test]
    fn this_machines_drive_is_classified() {
        let d = drive_of(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        assert_ne!(d, Drive::Unknown, "the system drive answers the storage query");
    }
}
