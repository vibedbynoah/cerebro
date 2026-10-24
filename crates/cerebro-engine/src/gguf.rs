//! GGUF reader. The file is memory-mapped; the header, metadata and tensor table are parsed with every
//! length, count, offset and size checked against the file, so a hostile file returns an error
//! instead of reading out of bounds.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::quant::DType;
use crate::Error;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const MAX_STRING: u64 = 1 << 24;
const MAX_ARRAY: u64 = 1 << 26;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KV: u64 = 1 << 20;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            Value::U8(v) => Some(v != 0),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions, innermost first (GGUF order): ne[0] is the row length.
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// Absolute byte offset in the file.
    pub offset: usize,
    pub size: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn rows(&self) -> usize {
        (self.elements() / self.dims[0]) as usize
    }
    pub fn cols(&self) -> usize {
        self.dims[0] as usize
    }
}

/// One file of a model: a whole model, or one part of a split model ("-00001-of-00003.gguf").
struct Part {
    /// Where this file starts in the model's combined offset space.
    base: usize,
    map: Mmap,
    path: std::path::PathBuf,
}

pub struct Gguf {
    /// The model's files in order. Tensor offsets are in one combined space: part i's bytes start
    /// at `parts[i].base` (the sum of the earlier parts' lengths); a tensor never spans two parts.
    parts: Vec<Part>,
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub alignment: usize,
    /// Layers read from the drive on every pass (stream.rs), when switched on.
    stream: std::sync::OnceLock<crate::stream::Stream>,
    /// The RAM expert cache of a mixture-of-experts model larger than memory (experts.rs).
    experts: std::sync::OnceLock<crate::experts::ExpertCache>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or_else(|| Error::Format("offset overflow".into()))?;
        let s = self.buf.get(self.pos..end).ok_or_else(|| Error::Format("unexpected end of file".into()))?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, Error> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return Err(Error::Format(format!("string of {n} bytes")));
        }
        let b = self.take(n as usize)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::Format("string is not UTF-8".into()))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, Error> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(Error::Format("nested arrays too deep".into()));
                }
                let et = self.u32()?;
                let n = self.u64()?;
                if n > MAX_ARRAY {
                    return Err(Error::Format(format!("array of {n} items")));
                }
                let mut v = Vec::with_capacity((n as usize).min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => return Err(Error::Format(format!("unknown metadata type {t}"))),
        })
    }
}

impl Gguf {
    /// Opens a model file. Any part of a split model ("name-00002-of-00003.gguf") opens the
    /// whole model: every part is mapped, the first part's metadata applies, and the tensors of
    /// all parts are listed together.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let files = split_parts(path).unwrap_or_else(|| vec![path.to_path_buf()]);
        let mut parts = Vec::with_capacity(files.len());
        let mut first: Option<(u32, HashMap<String, Value>, usize)> = None;
        let mut tensors = HashMap::new();
        let mut base = 0usize;
        for f in &files {
            let file = File::open(f).map_err(|e| Error::Format(format!("{}: {e}", f.display())))?;
            // SAFETY: the file is opened read-only and the map is never written; a concurrent writer
            // could change bytes under us, which every read below tolerates because sizes are re-checked.
            let map = unsafe { Mmap::map(&file)? };
            let (version, meta, ts, alignment) = Self::parse(&map)?;
            for (name, mut t) in ts {
                t.offset += base;
                if tensors.insert(name.clone(), t).is_some() {
                    return Err(Error::Format(format!("tensor {name} is in two parts of the model")));
                }
            }
            if first.is_none() {
                first = Some((version, meta, alignment));
            }
            let len = map.len();
            parts.push(Part { base, map, path: f.clone() });
            base = base.checked_add(len).ok_or_else(|| Error::Format("model too large".into()))?;
        }
        let (version, meta, alignment) = first.ok_or_else(|| Error::Format("no model file".into()))?;
        if let Some(n) = meta.get("split.count").and_then(Value::as_u64) {
            if n as usize != parts.len() {
                return Err(Error::Format(format!("this model has {n} parts but {} were found next to it", parts.len())));
            }
        }
        Ok(Self { parts, version, meta, tensors, alignment, stream: Default::default(), experts: Default::default() })
    }

    /// The part holding combined offset `offset`, and the offset inside it.
    #[inline]
    fn locate(&self, offset: usize) -> (&Part, usize) {
        let i = match self.parts.len() {
            1 => 0,
            _ => self.parts.partition_point(|p| p.base <= offset).saturating_sub(1),
        };
        let p = &self.parts[i];
        (p, offset - p.base)
    }

    /// The number of files the model is stored in.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    fn parse(mmap: &[u8]) -> Result<(u32, HashMap<String, Value>, HashMap<String, TensorInfo>, usize), Error> {
        let mut r = Reader { buf: mmap, pos: 0 };
        if r.u32()? != MAGIC {
            return Err(Error::Format("not a GGUF file".into()));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Format(format!("unsupported GGUF version {version}")));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
            return Err(Error::Format("implausible tensor or metadata count".into()));
        }
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let v = r.value(ty, 0)?;
            meta.insert(key, v);
        }
        let alignment = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 1 << 16 {
            return Err(Error::Format("bad alignment".into()));
        }
        let mut infos = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()?;
            if nd == 0 || nd > 4 {
                return Err(Error::Format(format!("tensor {name} has {nd} dimensions")));
            }
            let mut dims = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                let d = r.u64()?;
                if d == 0 || d > 1 << 32 {
                    return Err(Error::Format(format!("tensor {name} has a dimension of {d}")));
                }
                dims.push(d);
            }
            let dtype = DType::from_ggml(r.u32()?).ok_or_else(|| Error::Format(format!("tensor {name} has an unsupported type")))?;
            let rel = r.u64()?;
            infos.push((name, dims, dtype, rel));
        }
        let data_start = r.pos.div_ceil(alignment) * alignment;
        let mut tensors = HashMap::with_capacity(infos.len());
        for (name, dims, dtype, rel) in infos {
            let elements: u64 = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).ok_or_else(|| Error::Format("element count overflow".into()))?;
            if dims[0] % dtype.block_elems() as u64 != 0 {
                return Err(Error::Format(format!("tensor {name}: row length {} is not a multiple of the block size", dims[0])));
            }
            let size = (elements / dtype.block_elems() as u64)
                .checked_mul(dtype.block_bytes() as u64)
                .ok_or_else(|| Error::Format("size overflow".into()))? as usize;
            let offset = data_start.checked_add(rel as usize).ok_or_else(|| Error::Format("offset overflow".into()))?;
            if offset % alignment != 0 || offset.checked_add(size).is_none_or(|e| e > mmap.len()) {
                return Err(Error::Format(format!("tensor {name} lies outside the file")));
            }
            tensors.insert(name.clone(), TensorInfo { name, dims, dtype, offset, size });
        }
        Ok((version, meta, tensors, alignment))
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let (p, o) = self.locate(t.offset);
        &p.map[o..o + t.size]
    }

    /// A byte range already validated when the tensor table was parsed.
    /// Reads one byte of every page in these ranges, in parallel, so they are mapped into this
    /// process before decoding needs them. A first touch costs a page fault per 4 KB even when the
    /// file is in the OS cache; done lazily on the CPU expert path it measured 30.3 tok/s against
    /// 36.2 after touching up front (Qwen3-Coder-30B-A3B hybrid, i9-12900H + RTX 3070 Ti).
    pub fn touch(&self, ranges: &[(usize, usize)]) {
        use rayon::prelude::*;
        // Large ranges are split so every thread takes part.
        const PIECE: usize = 32 << 20;
        let pieces: Vec<(usize, usize)> = ranges.iter().flat_map(|&(o, l)| (0..l.div_ceil(PIECE)).map(move |i| (o + i * PIECE, PIECE.min(l - i * PIECE)))).collect();
        pieces.par_iter().for_each(|&(o, l)| {
            let (p, o) = self.locate(o);
            let m: &[u8] = &p.map;
            let end = (o + l).min(m.len());
            let mut acc = 0u8;
            for i in (o..end).step_by(4096) {
                // SAFETY: i is inside the mapping; the volatile read keeps the touch from being elided.
                acc ^= unsafe { std::ptr::read_volatile(m.as_ptr().add(i)) };
            }
            std::hint::black_box(acc);
        });
    }

    /// Bytes in all of the model's files together.
    pub fn len(&self) -> usize {
        self.parts.last().map_or(0, |p| p.base + p.map.len())
    }

    /// Tells the OS these file bytes will not be read soon (e.g. experts now served from VRAM): their
    /// pages leave this process's working set and become the first candidates for eviction, so RAM
    /// goes to the weights the CPU still reads. The file stays mapped; a later read pages them back.
    pub fn release(&self, offset: usize, len: usize) {
        #[cfg(windows)]
        {
            let (start, end) = self.page_range(offset, len);
            // SAFETY: the range lies inside our mapping. VirtualUnlock on pages that are not locked
            // fails with ERROR_NOT_LOCKED but still removes them from the working set (documented).
            unsafe { win::VirtualUnlock(start as *mut _, end - start) };
        }
        let _ = (offset, len);
    }

    /// Asks the OS to read these file ranges into memory ahead of use (asynchronous, best effort):
    /// experts the CPU will compute, so a decode step does not stall on a page fault from disk.
    pub fn prefetch(&self, ranges: &[(usize, usize)]) {
        #[cfg(windows)]
        {
            let entries: Vec<win::MemoryRange> = ranges
                .iter()
                .map(|&(o, l)| {
                    let (a, b) = self.page_range(o, l);
                    win::MemoryRange { va: a as *mut _, bytes: b - a }
                })
                .collect();
            if !entries.is_empty() {
                // SAFETY: every range lies inside our mapping; the call only hints the memory manager.
                unsafe { win::PrefetchVirtualMemory(win::GetCurrentProcess(), entries.len(), entries.as_ptr(), 0) };
            }
        }
        let _ = ranges;
    }

    /// The mapped pages covering a range (which lies inside one part).
    #[cfg(windows)]
    fn page_range(&self, offset: usize, len: usize) -> (usize, usize) {
        let (p, o) = self.locate(offset);
        let base = p.map.as_ptr() as usize;
        let start = (base + o) & !4095;
        let end = (base + o + len + 4095) & !4095;
        (start, end.min(base + p.map.len()).max(start))
    }

    #[inline]
    pub fn bytes_at(&self, offset: usize, len: usize) -> &[u8] {
        if let Some(b) = self.experts.get().and_then(|c| c.lookup(offset, len)) {
            return b;
        }
        if let Some(b) = self.stream.get().and_then(|s| s.lookup(offset, len)) {
            return b;
        }
        let (p, o) = self.locate(offset);
        &p.map[o..o + len]
    }

    /// Serves these file ranges from the drive on every pass (stream.rs) and drops them from RAM.
    /// Once per model; single-file models only (a split model's parts are separate files).
    pub fn set_stream(&self, path: &Path, units: &[(usize, usize)]) -> std::io::Result<()> {
        if self.parts.len() != 1 {
            return Err(std::io::Error::other("reading layers from the drive is not available for a model stored in several files yet"));
        }
        let s = crate::stream::Stream::new(path, self.len(), units)?;
        for &(lo, hi) in units {
            self.release(lo, hi - lo);
        }
        self.stream.set(s).map_err(|_| std::io::Error::other("the drive tier is already set"))
    }

    pub fn stream(&self) -> Option<&crate::stream::Stream> {
        self.stream.get()
    }

    /// The model's files with where each starts in the combined offset space and its length.
    pub fn files(&self) -> Vec<(std::path::PathBuf, usize, usize)> {
        self.parts.iter().map(|p| (p.path.clone(), p.base, p.map.len())).collect()
    }

    pub fn set_experts(&self, c: crate::experts::ExpertCache) -> bool {
        self.experts.set(c).is_ok()
    }

    pub fn experts(&self) -> Option<&crate::experts::ExpertCache> {
        self.experts.get()
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, Error> {
        self.tensors.get(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    }

    pub fn u64(&self, key: &str) -> Result<u64, Error> {
        self.meta.get(key).and_then(Value::as_u64).ok_or_else(|| Error::Format(format!("missing metadata {key}")))
    }

    pub fn f32(&self, key: &str) -> Result<f32, Error> {
        self.meta.get(key).and_then(Value::as_f32).ok_or_else(|| Error::Format(format!("missing metadata {key}")))
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.meta.get(key).and_then(Value::as_str)
    }
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct MemoryRange {
        pub va: *mut c_void,
        pub bytes: usize,
    }

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
        pub fn VirtualUnlock(addr: *mut c_void, size: usize) -> i32;
        pub fn PrefetchVirtualMemory(process: *mut c_void, n: usize, entries: *const MemoryRange, flags: u32) -> i32;
        pub fn GetCurrentProcess() -> *mut c_void;
    }
}

/// Physical memory currently available to applications, in bytes (u64::MAX if unknown).
pub fn available_ram() -> u64 {
    #[cfg(windows)]
    {
        let mut st = win::MemoryStatusEx { length: std::mem::size_of::<win::MemoryStatusEx>() as u32, ..Default::default() };
        // SAFETY: st is a properly sized MEMORYSTATUSEX with its length set.
        if unsafe { win::GlobalMemoryStatusEx(&mut st) } != 0 {
            return st.avail_phys;
        }
    }
    u64::MAX
}

/// Physical memory installed, in bytes (0 if unknown).
pub fn total_ram() -> u64 {
    #[cfg(windows)]
    {
        let mut st = win::MemoryStatusEx { length: std::mem::size_of::<win::MemoryStatusEx>() as u32, ..Default::default() };
        // SAFETY: st is a properly sized MEMORYSTATUSEX with its length set.
        if unsafe { win::GlobalMemoryStatusEx(&mut st) } != 0 {
            return st.total_phys;
        }
    }
    0
}

/// How much of the model the engine may read into RAM ahead of time or keep in its expert cache:
/// what is available now, less a reserve the rest of the computer keeps (the larger of 2.5 GiB and
/// a twelfth of the installed memory). Reading ahead into the last free gigabytes made a 32 GB
/// laptop run critically low while a 20 GB model loaded; a wider reserve (a sixth) starved
/// Flash-Next's expert cache. Past this, layers stay on the drive and are read as they are used.
pub fn prefetch_room() -> usize {
    let reserve = (5u64 << 29).max(total_ram() / 12);
    available_ram().saturating_sub(reserve) as usize
}

/// The files of a split model, in order, when `path` is one part of one ("name-00002-of-00003.gguf"
/// gives name-00001.. to name-00003..). None for an ordinary file.
pub fn split_parts(path: &Path) -> Option<Vec<std::path::PathBuf>> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, _index) = head.rsplit_once('-')?;
    if count.len() != 5 || !count.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: usize = count.parse().ok()?;
    if n < 2 {
        return None;
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    Some((1..=n).map(|i| dir.join(format!("{prefix}-{i:05}-of-{count}.gguf"))).collect())
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn split_names_list_every_part_and_plain_names_none() {
        let p = Path::new("C:/m/Model-UD-IQ3_XXS-00002-of-00003.gguf");
        let parts = split_parts(p).unwrap();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].ends_with("Model-UD-IQ3_XXS-00001-of-00003.gguf"));
        assert!(parts[2].ends_with("Model-UD-IQ3_XXS-00003-of-00003.gguf"));
        assert!(split_parts(Path::new("C:/m/Qwen3-8B-Q4_K_M.gguf")).is_none());
        assert!(split_parts(Path::new("C:/m/x-00001-of-00001.gguf")).is_none());
        assert!(split_parts(Path::new("C:/m/x-1-of-3.gguf")).is_none());
    }
}
