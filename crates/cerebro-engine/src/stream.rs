//! The drive tier: layers read from the model file on every pass instead of being kept in RAM.
//!
//! A memory-mapped layer that does not fit in RAM is paged in 4 KB at a time on demand, and Windows
//! keeps whatever it read in its file cache, so how much really comes from the drive depends on
//! everything else running. Streaming makes it explicit: each streamed layer (and optionally the
//! output head) is read with unbuffered I/O (`FILE_FLAG_NO_BUFFERING`, so nothing is cached) in large
//! parallel chunks into a ring of aligned buffers (2 to 8, about 512 MB). While the processor computes
//! one layer the next ones are read into the other buffers, so the drive and the processor work at
//! the same time. RAM use is the ring, whatever the model's size.
//!
//! The weights the processor reads all go through `Gguf::bytes_at`; it serves offsets inside a
//! buffered unit from the buffer and everything else from the mapping, so results are identical with
//! streaming on or off (only where the bytes come from changes).

use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Sector alignment for unbuffered reads (offsets, lengths and buffer addresses).
const ALIGN: usize = 4096;
/// One read request, MB; several are in flight at once (NVMe drives need a deep queue). 16 x 4 MB
/// reached the drive's own sequential speed on a Micron 3400 (2.08 of 2.10 GB/s); 8 x 2 MB gave 1.58.
/// CEREBRO_DRIVE_CHUNK_MB overrides it.
fn chunk() -> usize {
    std::env::var("CEREBRO_DRIVE_CHUNK_MB").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| (1..=64).contains(&v)).unwrap_or(4) << 20
}
/// Reader handles, each issuing one request at a time. CEREBRO_DRIVE_READERS overrides it.
fn readers() -> usize {
    std::env::var("CEREBRO_DRIVE_READERS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| (1..=64).contains(&v)).unwrap_or(16)
}
const NONE: usize = usize::MAX;
/// RAM for the ring of buffers; at least two buffers, at most eight.
const RING_BYTES: usize = 512 << 20;
const RING_MAX: usize = 8;

struct Buf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl Buf {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len.max(ALIGN).next_multiple_of(ALIGN), ALIGN).expect("buffer layout");
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory for the drive buffers");
        Self { ptr, layout }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

struct Slot {
    buf: Buf,
    /// The unit whose bytes the buffer holds; NONE while empty or being refilled.
    held: AtomicUsize,
}

/// Counters for the Monitor and the benchmarks.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct StreamStats {
    pub bytes_read: u64,
    /// Time spent reading, summed over reads (they overlap computing).
    pub read_s: f64,
    /// Time the processor waited for a layer that was not read yet.
    pub wait_s: f64,
    /// Reads that failed and fell back to the mapping.
    pub errors: u64,
}

struct Inner {
    /// Aligned [lo, hi) file ranges, one per unit.
    units: Vec<(usize, usize)>,
    slots: Vec<Slot>,
    files: Vec<File>,
    file_len: usize,
    chunk: usize,
    /// Unit being read into each slot by the reader thread, NONE when idle.
    loading: Mutex<Vec<usize>>,
    done: Condvar,
    bytes: AtomicU64,
    read_ns: AtomicU64,
    wait_ns: AtomicU64,
    errors: AtomicU64,
    failed: AtomicBool,
}

// SAFETY: the buffers are written only by the reader while their slot's `held` is NONE and its
// `loading` entry names the unit; `lookup` hands out slices only for a slot whose `held` names a
// unit, and the schedule (enter) never refills the slot the processor is reading.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct Stream {
    inner: Arc<Inner>,
    tx: std::sync::mpsc::Sender<(usize, usize)>,
    /// The whole streamed region, for a quick test in `lookup`.
    span: (usize, usize),
}

fn open_unbuffered(path: &std::path::Path) -> std::io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        o.custom_flags(FILE_FLAG_NO_BUFFERING);
    }
    o.open(path)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, off)
}

impl Inner {
    /// Reads unit `u` into slot `s` with every reader handle in parallel.
    fn load(&self, u: usize, s: usize) {
        let t = Instant::now();
        let (lo, hi) = self.units[u];
        let n = hi - lo;
        let dst = self.slots[s].buf.ptr as usize;
        let chunk = self.chunk;
        let chunks = n.div_ceil(chunk);
        let next = AtomicUsize::new(0);
        let ok = AtomicBool::new(true);
        std::thread::scope(|sc| {
            for f in self.files.iter().take(chunks.max(1)) {
                let (next, ok) = (&next, &ok);
                sc.spawn(move || loop {
                    let c = next.fetch_add(1, Ordering::Relaxed);
                    if c >= chunks {
                        break;
                    }
                    let off = c * chunk;
                    let len = chunk.min(n - off);
                    // SAFETY: [off, off + len) lies inside this slot's buffer (n <= its size) and
                    // each chunk is written by one thread.
                    let buf = unsafe { std::slice::from_raw_parts_mut((dst + off) as *mut u8, len) };
                    let mut got = 0;
                    while got < len {
                        let at = lo + off + got;
                        if at >= self.file_len {
                            break;
                        }
                        match read_at(f, &mut buf[got..], at as u64) {
                            Ok(0) => break,
                            Ok(k) => got += k,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                            Err(_) => {
                                ok.store(false, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                });
            }
        });
        self.bytes.fetch_add(n as u64, Ordering::Relaxed);
        self.read_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if ok.load(Ordering::Relaxed) {
            self.slots[s].held.store(u, Ordering::Release);
        } else {
            // The mapping serves the bytes instead: slower, still correct.
            self.errors.fetch_add(1, Ordering::Relaxed);
            self.failed.store(true, Ordering::Relaxed);
        }
    }
}

impl Stream {
    /// `units`: file byte ranges, in the order the processor reads them each pass. Fails when the
    /// file cannot be opened for unbuffered reading.
    pub fn new(path: &std::path::Path, file_len: usize, units: &[(usize, usize)]) -> std::io::Result<Self> {
        let units: Vec<(usize, usize)> = units.iter().map(|&(lo, hi)| (lo / ALIGN * ALIGN, hi.next_multiple_of(ALIGN))).collect();
        let biggest = units.iter().map(|&(lo, hi)| hi - lo).max().unwrap_or(ALIGN);
        let files = (0..readers()).map(|_| open_unbuffered(path)).collect::<std::io::Result<Vec<_>>>()?;
        let span = (units.iter().map(|u| u.0).min().unwrap_or(0), units.iter().map(|u| u.1).max().unwrap_or(0));
        // Never more buffers than units: a unit is never held twice.
        let n = (RING_BYTES / biggest).clamp(2, RING_MAX).min(units.len().max(1));
        let inner = Arc::new(Inner {
            units,
            slots: (0..n).map(|_| Slot { buf: Buf::new(biggest), held: AtomicUsize::new(NONE) }).collect(),
            files,
            file_len,
            chunk: chunk(),
            loading: Mutex::new(vec![NONE; n]),
            done: Condvar::new(),
            bytes: AtomicU64::new(0),
            read_ns: AtomicU64::new(0),
            wait_ns: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            failed: AtomicBool::new(false),
        });
        let (tx, rx) = std::sync::mpsc::channel::<(usize, usize)>();
        let w = inner.clone();
        std::thread::Builder::new()
            .name("cerebro-drive".into())
            .spawn(move || {
                // Ends when the Stream (and its sender) is dropped.
                while let Ok((u, s)) = rx.recv() {
                    w.load(u, s);
                    let mut l = w.loading.lock().unwrap_or_else(|e| e.into_inner());
                    l[s] = NONE;
                    w.done.notify_all();
                }
            })
            .map_err(std::io::Error::other)?;
        Ok(Self { inner, tx, span })
    }

    pub fn units(&self) -> usize {
        self.inner.units.len()
    }

    /// RAM the ring of buffers takes.
    pub fn buffer_bytes(&self) -> usize {
        self.inner.slots.len() * self.inner.slots[0].buf.layout.size()
    }

    pub fn stats(&self) -> StreamStats {
        let i = &self.inner;
        StreamStats {
            bytes_read: i.bytes.load(Ordering::Relaxed),
            read_s: i.read_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            wait_s: i.wait_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            errors: i.errors.load(Ordering::Relaxed),
        }
    }

    /// The bytes at [off, off + len) when a buffer holds them.
    #[inline]
    pub fn lookup(&self, off: usize, len: usize) -> Option<&[u8]> {
        if off < self.span.0 || off >= self.span.1 {
            return None;
        }
        for s in &self.inner.slots {
            let u = s.held.load(Ordering::Acquire);
            if u == NONE {
                continue;
            }
            let (lo, hi) = self.inner.units[u];
            if off >= lo && off + len <= hi {
                // SAFETY: inside the buffer (hi - lo <= its size); the slot is not refilled while
                // the processor reads this unit (see `enter`).
                return Some(unsafe { std::slice::from_raw_parts(s.buf.ptr.add(off - lo), len) });
            }
        }
        None
    }

    /// Called before the processor reads unit `u`: waits until a buffer holds it (reading it now
    /// if nobody asked for it), then queues the units after it into the other buffers.
    pub fn enter(&self, u: usize) {
        let i = &self.inner;
        if i.failed.load(Ordering::Relaxed) {
            return;
        }
        let ns = i.slots.len();
        let held = |k: usize| i.slots[k].held.load(Ordering::Acquire);
        let t = Instant::now();
        let mut s = (0..ns).find(|&k| held(k) == u);
        if s.is_none() {
            let mut l = i.loading.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(k) = (0..ns).find(|&k| l[k] == u) {
                while l[k] == u {
                    l = i.done.wait(l).unwrap_or_else(|e| e.into_inner());
                }
                s = Some(k);
            } else {
                // Nobody is reading it: take a buffer the reader is not filling.
                let k = loop {
                    if let Some(k) = (0..ns).find(|&k| l[k] == NONE) {
                        break k;
                    }
                    l = i.done.wait(l).unwrap_or_else(|e| e.into_inner());
                };
                l[k] = u;
                i.slots[k].held.store(NONE, Ordering::Release);
                drop(l);
                i.load(u, k);
                let mut l = i.loading.lock().unwrap_or_else(|e| e.into_inner());
                l[k] = NONE;
                i.done.notify_all();
                s = Some(k);
            }
        }
        i.wait_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let Some(s) = s else { return };
        // Read ahead: the next ns - 1 units, each into a buffer holding none of the wanted ones.
        let nu = i.units.len();
        let want: Vec<usize> = (1..ns).map(|d| (u + d) % nu).filter(|&x| x != u).collect();
        let mut l = i.loading.lock().unwrap_or_else(|e| e.into_inner());
        for &w in &want {
            if (0..ns).any(|k| held(k) == w || l[k] == w) {
                continue;
            }
            let free = (0..ns).find(|&k| k != s && l[k] == NONE && !want.contains(&held(k)));
            let Some(k) = free else { break };
            l[k] = w;
            i.slots[k].held.store(NONE, Ordering::Release);
            if self.tx.send((w, k)).is_err() {
                l[k] = NONE;
                break;
            }
        }
    }
}
