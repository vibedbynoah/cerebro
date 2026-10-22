//! The RAM tier of a mixture-of-experts model larger than memory: an expert cache Cerebro manages
//! itself, for every MoE architecture.
//!
//! Without it the experts are memory-mapped, and an expert that is not in the OS file cache is read
//! 4 KB per page fault, one fault at a time: tens of MB/s from a drive that reads GB/s (measured on
//! Qwen3.8-Flash-Next: 20-60 MB/s of hard faults during decode). Here each expert (its gate, up
//! and down rows together) lives in one slot of a fixed arena sized to the RAM the model can use.
//! A miss reads the expert's three ranges with unbuffered requests in parallel, at the drive's
//! sequential speed. Slots are kept by use: each use adds one to the expert's count and the counts
//! halve every few hundred tokens, so the experts the current topic uses stay and old ones go.
//!
//! The processor reads weights through `Gguf::bytes_at`, which asks the cache first: a cached
//! expert's bytes come from its slot, anything else from the mapping. Results are identical either
//! way; only where the bytes come from changes. The GPU's own VRAM expert cache (gpu.rs) sits above
//! this one.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const ALIGN: usize = 4096;
const NONE: u32 = u32::MAX;
/// Counts halve after this many expert uses per layer (about this many tokens).
const DECAY_EVERY: u64 = 256;

/// Where one expert's bytes are in the model: three (offset, len) ranges in the combined offset space.
#[derive(Clone, Copy, Debug)]
pub struct ExpertRanges {
    pub r: [(usize, usize); 3],
}

impl ExpertRanges {
    pub fn bytes(&self) -> usize {
        self.r.iter().map(|x| x.1).sum()
    }
}

/// One of the model's files: where it starts in the combined offset space, and its handles.
pub struct FilePart {
    pub base: usize,
    pub len: usize,
    pub files: Vec<File>,
}

#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub struct ExpertStats {
    pub slots: usize,
    pub slot_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub read_s: f64,
}

struct Book {
    /// Slot -> expert key (NONE when free).
    owner: Vec<u32>,
    /// Expert key -> decayed use count.
    count: Vec<f32>,
    uses: u64,
}

pub struct ExpertCache {
    experts: Vec<ExpertRanges>,
    /// Offset of each of an expert's ranges -> (expert key, which range).
    by_offset: HashMap<usize, (u32, u8)>,
    /// Expert key -> slot (NONE when not cached).
    slot_of: Vec<AtomicU32>,
    arena: *mut u8,
    layout: std::alloc::Layout,
    slot_bytes: usize,
    n_slots: usize,
    parts: Vec<FilePart>,
    book: Mutex<Book>,
    hits: AtomicU64,
    misses: AtomicU64,
    bytes: AtomicU64,
    read_ns: AtomicU64,
}

// SAFETY: a slot is written only by `ensure` while it is owned by no cached expert (its key's
// slot_of entry was cleared first) and no reader holds it: the processor reads an expert only after
// `ensure` returned for it, and the slots `ensure` evicts are never ones the current call keeps.
unsafe impl Send for ExpertCache {}
unsafe impl Sync for ExpertCache {}

fn open_unbuffered(path: &std::path::Path) -> std::io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        o.custom_flags(0x2000_0000); // FILE_FLAG_NO_BUFFERING
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

impl ExpertCache {
    /// `experts[key]` for key = layer * n_expert + expert; `files` the model's files in order with
    /// their combined-space starts; `budget` the RAM the arena may take.
    pub fn new(experts: Vec<ExpertRanges>, files: &[(std::path::PathBuf, usize, usize)], budget: usize, readers: usize) -> std::io::Result<Self> {
        let slot_bytes = experts.iter().map(|e| e.bytes()).max().unwrap_or(0).next_multiple_of(64).max(64);
        let n_slots = (budget / slot_bytes).min(experts.len()).max(1);
        let layout = std::alloc::Layout::from_size_align(n_slots * slot_bytes, ALIGN).map_err(std::io::Error::other)?;
        // SAFETY: non-zero size.
        let arena = unsafe { std::alloc::alloc(layout) };
        if arena.is_null() {
            return Err(std::io::Error::other("not enough memory for the expert cache"));
        }
        let parts = files
            .iter()
            .map(|(p, base, len)| Ok(FilePart { base: *base, len: *len, files: (0..readers.max(1)).map(|_| open_unbuffered(p)).collect::<std::io::Result<Vec<_>>>()? }))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut by_offset = HashMap::with_capacity(experts.len() * 3);
        for (k, e) in experts.iter().enumerate() {
            for (w, r) in e.r.iter().enumerate() {
                by_offset.insert(r.0, (k as u32, w as u8));
            }
        }
        Ok(Self {
            slot_of: (0..experts.len()).map(|_| AtomicU32::new(NONE)).collect(),
            book: Mutex::new(Book { owner: vec![NONE; n_slots], count: vec![0.0; experts.len()], uses: 0 }),
            experts,
            by_offset,
            arena,
            layout,
            slot_bytes,
            n_slots,
            parts,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            read_ns: AtomicU64::new(0),
        })
    }

    pub fn stats(&self) -> ExpertStats {
        ExpertStats {
            slots: self.n_slots,
            slot_bytes: self.slot_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bytes_read: self.bytes.load(Ordering::Relaxed),
            read_s: self.read_ns.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    fn slot_ptr(&self, slot: u32) -> *mut u8 {
        // SAFETY: slot < n_slots, so the slot lies inside the arena.
        unsafe { self.arena.add(slot as usize * self.slot_bytes) }
    }

    /// The bytes at [off, off + len) when they are a cached expert's range (or a prefix of one).
    #[inline]
    pub fn lookup(&self, off: usize, len: usize) -> Option<&[u8]> {
        let &(k, w) = self.by_offset.get(&off)?;
        let slot = self.slot_of[k as usize].load(Ordering::Acquire);
        if slot == NONE {
            return None;
        }
        let e = &self.experts[k as usize];
        if len > e.r[w as usize].1 {
            return None;
        }
        let within: usize = e.r[..w as usize].iter().map(|x| x.1).sum();
        // SAFETY: within + len <= the expert's bytes <= slot_bytes.
        Some(unsafe { std::slice::from_raw_parts(self.slot_ptr(slot).add(within), len) })
    }

    /// Makes every expert in `keys` cached before the processor reads them: counts the use, and
    /// reads the missing ones from the drive (in parallel) into the least-used slots that are not
    /// among `keys`. When `keys` holds more experts than there are slots, only the first ones are
    /// cached and the rest are read from the mapping.
    pub fn ensure(&self, keys: &[usize]) {
        let mut book = self.book.lock().unwrap_or_else(|e| e.into_inner());
        book.uses += 1;
        if book.uses % (DECAY_EVERY * keys.len().max(1) as u64) == 0 {
            book.count.iter_mut().for_each(|c| *c *= 0.5);
        }
        for &k in keys {
            book.count[k] += 1.0;
        }
        let missing: Vec<usize> = keys.iter().copied().filter(|&k| self.slot_of[k].load(Ordering::Acquire) == NONE).collect();
        self.hits.fetch_add((keys.len() - missing.len()) as u64, Ordering::Relaxed);
        if missing.is_empty() {
            return;
        }
        self.misses.fetch_add(missing.len() as u64, Ordering::Relaxed);
        // Victims: free slots first, then the least-used experts not wanted now.
        let keep: std::collections::HashSet<usize> = keys.iter().copied().collect();
        let mut cand: Vec<(f32, u32)> = (0..self.n_slots as u32)
            .filter(|&s| {
                let o = book.owner[s as usize];
                o == NONE || !keep.contains(&(o as usize))
            })
            .map(|s| {
                let o = book.owner[s as usize];
                (if o == NONE { f32::NEG_INFINITY } else { book.count[o as usize] }, s)
            })
            .collect();
        let take = missing.len().min(cand.len());
        if take == 0 {
            return;
        }
        cand.select_nth_unstable_by(take - 1, |a, b| a.0.total_cmp(&b.0));
        let mut jobs = Vec::with_capacity(take);
        for (i, &k) in missing.iter().take(take).enumerate() {
            let s = cand[i].1;
            let old = book.owner[s as usize];
            if old != NONE {
                self.slot_of[old as usize].store(NONE, Ordering::Release);
            }
            book.owner[s as usize] = k as u32;
            jobs.push((k, s));
        }
        drop(book);
        let t = Instant::now();
        let ok: Vec<bool> = {
            use rayon::prelude::*;
            jobs.par_iter().enumerate().map(|(i, &(k, s))| self.read_expert(k, s, i)).collect()
        };
        self.read_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut book = self.book.lock().unwrap_or_else(|e| e.into_inner());
        for (&(k, s), ok) in jobs.iter().zip(ok) {
            if ok {
                self.slot_of[k].store(s, Ordering::Release);
            } else {
                // The mapping serves it instead.
                book.owner[s as usize] = NONE;
            }
        }
    }

    /// Reads expert `k`'s three ranges into slot `s`. `lane` picks the reader handles.
    fn read_expert(&self, k: usize, s: u32, lane: usize) -> bool {
        let e = self.experts[k];
        let dst = self.slot_ptr(s) as usize;
        let mut at = 0usize;
        let mut buf: Vec<u8> = Vec::new();
        for &(off, len) in &e.r {
            let Some(p) = self.parts.iter().rev().find(|p| p.base <= off) else { return false };
            let local = off - p.base;
            // Unbuffered reads need sector-aligned offsets and lengths: read the aligned span into
            // an aligned scratch buffer and copy the exact bytes.
            let lo = local / ALIGN * ALIGN;
            let hi = (local + len).next_multiple_of(ALIGN).min(p.len.next_multiple_of(ALIGN));
            let n = hi - lo;
            buf.resize(n + ALIGN, 0);
            let pad = buf.as_ptr().align_offset(ALIGN);
            let f = &p.files[lane % p.files.len()];
            let mut got = 0;
            while got < n {
                match read_at(f, &mut buf[pad + got..pad + n], (lo + got) as u64) {
                    Ok(0) => break,
                    Ok(r) => got += r,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => return false,
                }
                if lo + got >= p.len {
                    break;
                }
            }
            if got < local - lo + len {
                return false;
            }
            // SAFETY: at + len <= the expert's bytes <= slot_bytes; the slot is owned by this read.
            unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr().add(pad + local - lo), (dst + at) as *mut u8, len) };
            at += len;
            self.bytes.fetch_add(n as u64, Ordering::Relaxed);
        }
        true
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.arena, self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_experts_read_back_exactly_and_the_least_used_go_first() {
        let dir = std::env::temp_dir().join(format!("cerebro-experts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.bin");
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        // Four experts with unaligned ranges.
        let experts: Vec<ExpertRanges> = (0..4).map(|k| ExpertRanges { r: [(1000 + k * 60_000, 9_001), (21_000 + k * 60_000, 8_999), (41_000 + k * 60_000, 10_000)] }).collect();
        let c = ExpertCache::new(experts.clone(), &[(path.clone(), 0, data.len())], 2 * 28_100, 2).unwrap();
        assert_eq!(c.stats().slots, 2);
        assert!(c.lookup(1000, 9001).is_none());
        c.ensure(&[0, 1]);
        for k in 0..2 {
            for &(o, l) in &experts[k].r {
                assert_eq!(c.lookup(o, l).unwrap(), &data[o..o + l]);
            }
        }
        // Expert 0 used again; 2 then evicts 1, the least used.
        c.ensure(&[0]);
        c.ensure(&[2]);
        assert!(c.lookup(experts[0].r[0].0, 10).is_some());
        assert!(c.lookup(experts[1].r[0].0, 10).is_none());
        assert_eq!(c.lookup(experts[2].r[2].0, 10_000).unwrap(), &data[experts[2].r[2].0..experts[2].r[2].0 + 10_000]);
        let st = c.stats();
        assert_eq!((st.hits, st.misses), (1, 3));
        drop(c);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
