//! A model file's metadata, as the Models screen shows and edits it: the entries (`detail`), and
//! edited copies (`write_copy`, `revert_copy`).
//!
//! An edit never touches the original. The copy is `<stem>.edited.gguf` beside it: the header is
//! written again with the edited values (each keeps its stored type), the tensor table is copied
//! byte for byte (its offsets are relative to the data section, so they stay valid), the data
//! section starts at the next multiple of the file's alignment, and the tensor data is streamed
//! across unchanged. The copy is written under a temporary name and renamed when complete. Its edit
//! list is kept beside it as `<stem>.edited.json`, so the screen can show it after a restart.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Longest string value accepted (the editor's limit).
const STRING_MAX: usize = 65_536;
/// Most edits in one write.
const MAX_EDITS: usize = 64;
/// Bytes per read while copying the tensor data.
const COPY_CHUNK: usize = 4 << 20;

/// One metadata entry as the editor shows it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Entry {
    pub key: String,
    /// "string", "uint32", "int32", "float32", "bool" or "array" (wider integer and float types
    /// show as their 32-bit kind; the stored type is kept when written).
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub value: serde_json::Value,
    pub editable: bool,
}

/// One edit: a key and its new value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edit {
    pub key: String,
    pub value: serde_json::Value,
}

/// An edited copy: where it is and what it changes.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Copy {
    pub path: String,
    pub edits: Vec<Edit>,
}

/// A metadata value as stored.
#[derive(Debug, Clone, PartialEq)]
enum Val {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// Element type and count; the bytes are copied, never parsed further.
    Array(u32, u64),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// One key-value pair: its key, value, and the byte range of its type tag plus value in the file.
struct Kv {
    key: String,
    val: Val,
    raw: std::ops::Range<usize>,
}

/// A parsed header: version, tensor count, the pairs, the tensor table's byte range, alignment
/// and where the data section starts.
struct Header {
    version: u32,
    n_tensors: u64,
    kvs: Vec<Kv>,
    tensors: std::ops::Range<usize>,
    alignment: u64,
    data_start: u64,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("The file's header is cut short.")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            return Err("A string in the file's header is implausibly long.".to_owned());
        }
        String::from_utf8(self.take(n as usize)?.to_vec()).map_err(|_| "A string in the file's header is not text.".to_owned())
    }
    /// Skips one value of `t` (inside an array).
    fn skip(&mut self, t: u32) -> Result<(), String> {
        match t {
            0 | 1 | 7 => self.take(1).map(|_| ()),
            2 | 3 => self.take(2).map(|_| ()),
            4..=6 => self.take(4).map(|_| ()),
            10..=12 => self.take(8).map(|_| ()),
            8 => self.string().map(|_| ()),
            9 => {
                let (et, n) = (self.u32()?, self.u64()?);
                for _ in 0..n {
                    self.skip(et)?;
                }
                Ok(())
            }
            _ => Err(format!("The file's header has an unknown value type {t}.")),
        }
    }
    fn value(&mut self, t: u32) -> Result<Val, String> {
        Ok(match t {
            0 => Val::U8(self.take(1)?[0]),
            1 => Val::I8(self.take(1)?[0] as i8),
            2 => Val::U16(u16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes"))),
            3 => Val::I16(i16::from_le_bytes(self.take(2)?.try_into().expect("2 bytes"))),
            4 => Val::U32(self.u32()?),
            5 => Val::I32(self.u32()? as i32),
            6 => Val::F32(f32::from_bits(self.u32()?)),
            7 => Val::Bool(self.take(1)?[0] != 0),
            8 => Val::Str(self.string()?),
            9 => {
                let (et, n) = (self.u32()?, self.u64()?);
                if n > 1 << 26 {
                    return Err("An array in the file's header is implausibly long.".to_owned());
                }
                for _ in 0..n {
                    self.skip(et)?;
                }
                Val::Array(et, n)
            }
            10 => Val::U64(self.u64()?),
            11 => Val::I64(self.u64()? as i64),
            12 => Val::F64(f64::from_bits(self.u64()?)),
            _ => return Err(format!("The file's header has an unknown value type {t}.")),
        })
    }
}

/// Reads the header of a GGUF file (version 2 or 3).
fn read_header(path: &Path) -> Result<(Vec<u8>, Header), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("Could not open the model file: {e}."))?;
    // The header is read in growing pieces: it is usually a few MB (the vocabulary), never the data.
    let mut buf = Vec::new();
    let mut want = 1 << 20;
    loop {
        buf.resize(want, 0);
        f.seek(SeekFrom::Start(0)).map_err(|e| format!("Could not read the model file: {e}."))?;
        let mut got = 0;
        while got < want {
            match f.read(&mut buf[got..]).map_err(|e| format!("Could not read the model file: {e}."))? {
                0 => break,
                n => got += n,
            }
        }
        buf.truncate(got);
        match parse_header(&buf) {
            Ok(h) => return Ok((buf, h)),
            Err(e) if got == want && want < 1 << 30 => {
                let _ = e;
                want *= 4;
            }
            Err(e) => return Err(e),
        }
    }
}

fn parse_header(b: &[u8]) -> Result<Header, String> {
    let mut r = Reader { b, at: 0 };
    if r.take(4)? != b"GGUF" {
        return Err("This is not a GGUF file.".to_owned());
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("GGUF version {version} is not supported."));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_kv > 1 << 20 || n_tensors > 1 << 20 {
        return Err("The file's header has implausible counts.".to_owned());
    }
    let mut kvs = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = r.string()?;
        let start = r.at;
        let t = r.u32()?;
        let val = r.value(t)?;
        kvs.push(Kv { key, val, raw: start..r.at });
    }
    let t0 = r.at;
    for _ in 0..n_tensors {
        r.string()?;
        let nd = r.u32()?;
        if nd > 8 {
            return Err("A tensor in the file has too many dimensions.".to_owned());
        }
        r.take(nd as usize * 8 + 4 + 8)?;
    }
    let tensors = t0..r.at;
    let alignment = kvs.iter().find(|k| k.key == "general.alignment").and_then(|k| match k.val {
        Val::U32(v) => Some(v as u64),
        Val::U64(v) => Some(v),
        _ => None,
    });
    let alignment = alignment.filter(|&a| a > 0 && a.is_power_of_two()).unwrap_or(32);
    let data_start = (r.at as u64).div_ceil(alignment) * alignment;
    Ok(Header { version, n_tensors, kvs, tensors, alignment, data_start })
}

/// Keys whose change would make the file unreadable or the model wrong: architecture, layout and
/// shape. Everything else that is a single value can be edited (names, context length, rope and
/// norm constants, special token ids, the chat template).
fn structural(key: &str, arch: &str) -> bool {
    const GENERAL: &[&str] = &["general.architecture", "general.alignment", "general.file_type", "general.quantization_version", "tokenizer.ggml.model", "tokenizer.ggml.pre"];
    const SHAPE: &[&str] = &[
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_feed_forward_length",
        "expert_shared_count",
        "rope.dimension_count",
        "rope.dimension_sections",
        "vocab_size",
        "full_attention_interval",
    ];
    if GENERAL.contains(&key) {
        return true;
    }
    match key.strip_prefix(arch).and_then(|k| k.strip_prefix('.')) {
        Some(k) => SHAPE.contains(&k) || k.starts_with("ssm.") || k.starts_with("vision.") || k.starts_with("audio."),
        None => key.starts_with("clip."),
    }
}

fn kind(v: &Val) -> &'static str {
    match v {
        Val::Str(_) => "string",
        Val::Bool(_) => "bool",
        Val::F32(_) | Val::F64(_) => "float32",
        Val::I8(_) | Val::I16(_) | Val::I32(_) | Val::I64(_) => "int32",
        Val::U8(_) | Val::U16(_) | Val::U32(_) | Val::U64(_) => "uint32",
        Val::Array(..) => "array",
    }
}

fn json_of(v: &Val) -> serde_json::Value {
    use serde_json::json;
    match v {
        Val::U8(x) => json!(x),
        Val::I8(x) => json!(x),
        Val::U16(x) => json!(x),
        Val::I16(x) => json!(x),
        Val::U32(x) => json!(x),
        Val::I32(x) => json!(x),
        Val::U64(x) => json!(x),
        Val::I64(x) => json!(x),
        // The shortest decimal that reads back as the same f32 (1e-6, not 9.99999997e-7).
        Val::F32(x) => json!(format!("{x}").parse::<f64>().unwrap_or(*x as f64)),
        Val::F64(x) => json!(x),
        Val::Bool(x) => json!(x),
        Val::Str(s) => json!(s),
        Val::Array(t, n) => {
            let what = match t {
                8 => "strings",
                9 => "arrays",
                _ => "numbers",
            };
            json!(format!("{} {what}", group_digits(*n)))
        }
    }
}

fn group_digits(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A JSON value as the stored type, or a plain refusal.
fn coerce(stored: &Val, v: &serde_json::Value, key: &str) -> Result<Val, String> {
    let wrong = || format!("{key}: wrong type.");
    let int = |lo: i128, hi: i128| -> Result<i128, String> {
        let n = v.as_i64().map(i128::from).or_else(|| v.as_u64().map(i128::from)).ok_or_else(wrong)?;
        if n < lo || n > hi {
            return Err(format!("{key}: between {lo} and {hi}."));
        }
        Ok(n)
    };
    Ok(match stored {
        Val::U8(_) => Val::U8(int(0, u8::MAX as i128)? as u8),
        Val::I8(_) => Val::I8(int(i8::MIN as i128, i8::MAX as i128)? as i8),
        Val::U16(_) => Val::U16(int(0, u16::MAX as i128)? as u16),
        Val::I16(_) => Val::I16(int(i16::MIN as i128, i16::MAX as i128)? as i16),
        Val::U32(_) => Val::U32(int(0, u32::MAX as i128)? as u32),
        Val::I32(_) => Val::I32(int(i32::MIN as i128, i32::MAX as i128)? as i32),
        Val::U64(_) => Val::U64(int(0, u64::MAX as i128)? as u64),
        Val::I64(_) => Val::I64(int(i64::MIN as i128, i64::MAX as i128)? as i64),
        Val::F32(_) => {
            let x = v.as_f64().ok_or_else(wrong)?;
            if !x.is_finite() || x.abs() > f32::MAX as f64 {
                return Err(format!("{key}: out of range for a 32-bit float."));
            }
            Val::F32(x as f32)
        }
        Val::F64(_) => Val::F64(v.as_f64().filter(|x| x.is_finite()).ok_or_else(wrong)?),
        Val::Bool(_) => Val::Bool(v.as_bool().ok_or_else(wrong)?),
        Val::Str(_) => {
            let s = v.as_str().ok_or_else(wrong)?;
            if s.chars().count() > STRING_MAX || s.contains('\0') {
                return Err(format!("{key}: at most {STRING_MAX} characters, no null characters."));
            }
            Val::Str(s.to_owned())
        }
        Val::Array(..) => return Err(format!("{key} is an array; arrays are read-only here.")),
    })
}

/// A value's type tag and bytes.
fn encode(v: &Val) -> Vec<u8> {
    let mut o = Vec::new();
    let tag = |o: &mut Vec<u8>, t: u32| o.extend_from_slice(&t.to_le_bytes());
    match v {
        Val::U8(x) => {
            tag(&mut o, 0);
            o.push(*x);
        }
        Val::I8(x) => {
            tag(&mut o, 1);
            o.push(*x as u8);
        }
        Val::U16(x) => {
            tag(&mut o, 2);
            o.extend_from_slice(&x.to_le_bytes());
        }
        Val::I16(x) => {
            tag(&mut o, 3);
            o.extend_from_slice(&x.to_le_bytes());
        }
        Val::U32(x) => {
            tag(&mut o, 4);
            o.extend_from_slice(&x.to_le_bytes());
        }
        Val::I32(x) => {
            tag(&mut o, 5);
            o.extend_from_slice(&x.to_le_bytes());
        }
        Val::F32(x) => {
            tag(&mut o, 6);
            o.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        Val::Bool(x) => {
            tag(&mut o, 7);
            o.push(*x as u8);
        }
        Val::Str(s) => {
            tag(&mut o, 8);
            o.extend_from_slice(&(s.len() as u64).to_le_bytes());
            o.extend_from_slice(s.as_bytes());
        }
        Val::U64(x) => {
            tag(&mut o, 10);
            o.extend_from_slice(&x.to_le_bytes());
        }
        Val::I64(x) => {
            tag(&mut o, 11);
            o.extend_from_slice(&x.to_le_bytes());
        }
        Val::F64(x) => {
            tag(&mut o, 12);
            o.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        Val::Array(..) => unreachable!("arrays are copied as raw bytes"),
    }
    o
}

/// `<stem>.edited.gguf` and `<stem>.edited.json` beside `original`.
pub fn copy_paths(original: &Path) -> (PathBuf, PathBuf) {
    let stem = original.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let dir = original.parent().unwrap_or(Path::new("."));
    (dir.join(format!("{stem}.edited.gguf")), dir.join(format!("{stem}.edited.json")))
}

/// Whether `path` is an edited copy this module wrote.
pub fn is_copy(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().ends_with(".edited.gguf"))
}

/// The copy of `original`, when there is one.
pub fn existing_copy(original: &Path) -> Option<Copy> {
    let (gguf, json) = copy_paths(original);
    if !gguf.exists() {
        return None;
    }
    let edits: Vec<Edit> = std::fs::read_to_string(&json).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    Some(Copy { path: gguf.to_string_lossy().into_owned(), edits })
}

/// The file's metadata entries, sorted by key, and its chat template.
pub fn detail(path: &Path) -> Result<(Vec<Entry>, Option<String>), String> {
    let (_, h) = read_header(path)?;
    let arch = h.kvs.iter().find(|k| k.key == "general.architecture").and_then(|k| if let Val::Str(s) = &k.val { Some(s.clone()) } else { None }).unwrap_or_default();
    let mut entries: Vec<Entry> = h
        .kvs
        .iter()
        .map(|k| Entry { key: k.key.clone(), kind: kind(&k.val), value: json_of(&k.val), editable: !matches!(k.val, Val::Array(..)) && !structural(&k.key, &arch) })
        .collect();
    entries.sort_by(|a, b| a.key.cmp(&b.key));
    let template = h.kvs.iter().find(|k| k.key == "tokenizer.chat_template").and_then(|k| if let Val::Str(s) = &k.val { Some(s.clone()) } else { None });
    Ok((entries, template))
}

/// Writes (or rewrites) the edited copy of `original` with `edits` merged into the copy's earlier
/// ones: the newest value per key wins, and a value equal to the original's drops that edit.
pub fn write_copy(original: &Path, edits: &[Edit]) -> Result<Copy, String> {
    if edits.is_empty() {
        return Err("Nothing to write.".to_owned());
    }
    if edits.len() > MAX_EDITS {
        return Err("Too many edits at once.".to_owned());
    }
    let (head, h) = read_header(original)?;
    let arch = h.kvs.iter().find(|k| k.key == "general.architecture").and_then(|k| if let Val::Str(s) = &k.val { Some(s.clone()) } else { None }).unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    for e in edits {
        if !seen.insert(e.key.as_str()) {
            return Err(format!("{} is edited twice.", e.key));
        }
    }
    // Merge: earlier edits of the copy, then these; checked against the original's types.
    let mut merged: Vec<(String, Val)> = Vec::new();
    let prior = existing_copy(original).map(|c| c.edits).unwrap_or_default();
    for e in prior.iter().chain(edits) {
        let Some(kv) = h.kvs.iter().find(|k| k.key == e.key) else { return Err(format!("The file has no key {}.", e.key)) };
        if matches!(kv.val, Val::Array(..)) || structural(&kv.key, &arch) {
            return Err(format!("{} is structural and cannot be changed.", kv.key));
        }
        let v = coerce(&kv.val, &e.value, &e.key)?;
        merged.retain(|(k, _)| k != &e.key);
        if v != kv.val {
            merged.push((e.key.clone(), v));
        }
    }
    if merged.is_empty() {
        return Err("The copy would match the original; revert it instead.".to_owned());
    }
    let (dest, json) = copy_paths(original);
    let tmp = dest.with_extension(format!("gguf.part{}", std::process::id()));
    let result = (|| -> Result<(), String> {
        let io = |e: std::io::Error| format!("Could not write the copy: {e}.");
        // The header is a few MB at most (the vocabulary): built in memory, then the data streamed.
        let mut hb: Vec<u8> = Vec::with_capacity(h.tensors.end + 64);
        hb.extend_from_slice(b"GGUF");
        hb.extend_from_slice(&h.version.to_le_bytes());
        hb.extend_from_slice(&h.n_tensors.to_le_bytes());
        hb.extend_from_slice(&(h.kvs.len() as u64).to_le_bytes());
        for kv in &h.kvs {
            hb.extend_from_slice(&(kv.key.len() as u64).to_le_bytes());
            hb.extend_from_slice(kv.key.as_bytes());
            match merged.iter().find(|(k, _)| k == &kv.key) {
                Some((_, v)) => hb.extend_from_slice(&encode(v)),
                None => hb.extend_from_slice(&head[kv.raw.clone()]),
            }
        }
        hb.extend_from_slice(&head[h.tensors.clone()]);
        let n = hb.len() as u64;
        hb.resize((n.div_ceil(h.alignment) * h.alignment) as usize, 0);
        let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(io)?);
        out.write_all(&hb).map_err(io)?;
        // The tensor data, streamed.
        let mut src = std::fs::File::open(original).map_err(|e| format!("Could not read the model file: {e}."))?;
        src.seek(SeekFrom::Start(h.data_start)).map_err(|e| format!("Could not read the model file: {e}."))?;
        let mut chunk = vec![0u8; COPY_CHUNK];
        loop {
            let got = src.read(&mut chunk).map_err(|e| format!("Could not read the model file: {e}."))?;
            if got == 0 {
                break;
            }
            out.write_all(&chunk[..got]).map_err(io)?;
        }
        out.flush().map_err(io)?;
        let f = out.into_inner().map_err(|e| format!("Could not write the copy: {}.", e.error()))?;
        f.sync_all().map_err(io)?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    let _ = std::fs::remove_file(&dest);
    std::fs::rename(&tmp, &dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("Could not put the copy in place (is it loaded?): {e}.")
    })?;
    let list: Vec<Edit> = merged
        .iter()
        .map(|(k, v)| Edit { key: k.clone(), value: json_of(v) })
        .collect();
    std::fs::write(&json, serde_json::to_string_pretty(&list).unwrap_or_default()).map_err(|e| format!("The copy was written but its edit list was not: {e}."))?;
    Ok(Copy { path: dest.to_string_lossy().into_owned(), edits: list })
}

/// Deletes the edited copy of `original` and its edit list.
pub fn revert_copy(original: &Path) -> Result<(), String> {
    let (gguf, json) = copy_paths(original);
    if gguf.exists() {
        std::fs::remove_file(&gguf).map_err(|e| format!("Could not delete the copy (is it loaded?): {e}."))?;
    }
    if json.exists() {
        std::fs::remove_file(&json).map_err(|e| format!("Could not delete the copy's edit list: {e}."))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A tiny valid GGUF v3: a string, a u32, an f32, a bool, an array, alignment 32, two tensors.
    fn tiny(dir: &Path) -> PathBuf {
        let mut b = Vec::new();
        let s = |b: &mut Vec<u8>, t: &str| {
            b.extend_from_slice(&(t.len() as u64).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        };
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&6u64.to_le_bytes());
        s(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        s(&mut b, "llama");
        s(&mut b, "general.name");
        b.extend_from_slice(&8u32.to_le_bytes());
        s(&mut b, "Tiny");
        s(&mut b, "llama.context_length");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&2048u32.to_le_bytes());
        s(&mut b, "llama.rope.freq_base");
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&10000f32.to_bits().to_le_bytes());
        s(&mut b, "tokenizer.ggml.tokens");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        s(&mut b, "a");
        s(&mut b, "b");
        s(&mut b, "tokenizer.ggml.add_bos_token");
        b.extend_from_slice(&7u32.to_le_bytes());
        b.push(1);
        // Two f32 tensors of 4 and 8 values.
        for (name, n, off) in [("t0", 4u64, 0u64), ("t1", 8, 32)] {
            s(&mut b, name);
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&n.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&off.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        for i in 0..4 {
            b.extend_from_slice(&(i as f32).to_le_bytes());
        }
        b.extend_from_slice(&[0u8; 16]);
        for i in 0..8 {
            b.extend_from_slice(&(10.0 + i as f32).to_le_bytes());
        }
        let p = dir.join("tiny.gguf");
        std::fs::write(&p, b).expect("write");
        p
    }

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("cerebro-gguf-edit-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())));
        std::fs::create_dir_all(&d).expect("dir");
        d
    }

    fn hash(p: &Path) -> Vec<u8> {
        use sha2::Digest;
        sha2::Sha256::digest(std::fs::read(p).expect("read")).to_vec()
    }

    #[test]
    fn edits_land_in_the_copy_and_the_tensors_are_unchanged() {
        let d = tmpdir();
        let p = tiny(&d);
        let before = hash(&p);
        let c = write_copy(&p, &[Edit { key: "general.name".into(), value: json!("Renamed") }, Edit { key: "llama.context_length".into(), value: json!(8192) }]).expect("write");
        assert_eq!(hash(&p), before, "the original changed");
        let g = cerebro_engine::gguf::Gguf::open(Path::new(&c.path)).expect("the engine reads the copy");
        assert_eq!(g.str("general.name"), Some("Renamed"));
        assert_eq!(g.u64("llama.context_length").ok(), Some(8192));
        let o = cerebro_engine::gguf::Gguf::open(&p).expect("original");
        for name in ["t0", "t1"] {
            assert_eq!(g.bytes(g.tensor(name).expect("t")), o.bytes(o.tensor(name).expect("t")), "{name} differs");
        }
        let (_, h) = read_header(Path::new(&c.path)).expect("header");
        assert_eq!(h.data_start % 32, 0);
        assert_eq!(c.edits.len(), 2);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_value_equal_to_the_original_drops_the_edit() {
        let d = tmpdir();
        let p = tiny(&d);
        write_copy(&p, &[Edit { key: "general.name".into(), value: json!("A") }, Edit { key: "llama.context_length".into(), value: json!(4096) }]).expect("write");
        let c = write_copy(&p, &[Edit { key: "general.name".into(), value: json!("Tiny") }]).expect("rewrite");
        assert_eq!(c.edits, vec![Edit { key: "llama.context_length".into(), value: json!(4096) }]);
        let err = write_copy(&p, &[Edit { key: "llama.context_length".into(), value: json!(2048) }]).unwrap_err();
        assert!(err.contains("revert"), "{err}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn wrong_types_arrays_and_structural_keys_are_refused() {
        let d = tmpdir();
        let p = tiny(&d);
        assert!(write_copy(&p, &[Edit { key: "llama.context_length".into(), value: json!("big") }]).unwrap_err().contains("wrong type"));
        assert!(write_copy(&p, &[Edit { key: "llama.context_length".into(), value: json!(-1) }]).is_err());
        assert!(write_copy(&p, &[Edit { key: "tokenizer.ggml.tokens".into(), value: json!("x") }]).unwrap_err().contains("structural"));
        assert!(write_copy(&p, &[Edit { key: "general.architecture".into(), value: json!("qwen3") }]).unwrap_err().contains("structural"));
        assert!(write_copy(&p, &[Edit { key: "no.such".into(), value: json!(1) }]).unwrap_err().contains("no key"));
        assert!(!copy_paths(&p).0.exists(), "a refused write left a file");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn floats_and_bools_keep_their_types() {
        let d = tmpdir();
        let p = tiny(&d);
        let c = write_copy(&p, &[Edit { key: "llama.rope.freq_base".into(), value: json!(1e6) }, Edit { key: "tokenizer.ggml.add_bos_token".into(), value: json!(false) }]).expect("write");
        let g = cerebro_engine::gguf::Gguf::open(Path::new(&c.path)).expect("read");
        assert_eq!(g.f32("llama.rope.freq_base").ok(), Some(1e6));
        assert!(matches!(g.meta.get("tokenizer.ggml.add_bos_token"), Some(cerebro_engine::gguf::Value::Bool(false))));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn revert_removes_the_copy_and_its_list() {
        let d = tmpdir();
        let p = tiny(&d);
        write_copy(&p, &[Edit { key: "general.name".into(), value: json!("B") }]).expect("write");
        let (g, j) = copy_paths(&p);
        assert!(g.exists() && j.exists());
        assert!(existing_copy(&p).is_some());
        revert_copy(&p).expect("revert");
        assert!(!g.exists() && !j.exists());
        assert!(existing_copy(&p).is_none());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn detail_lists_entries_with_types_and_editability() {
        let d = tmpdir();
        let p = tiny(&d);
        let (entries, template) = detail(&p).expect("detail");
        assert!(template.is_none());
        let get = |k: &str| entries.iter().find(|e| e.key == k).expect(k).clone();
        assert_eq!(get("general.name").editable, true);
        assert_eq!(get("general.architecture").editable, false);
        assert_eq!(get("llama.context_length").kind, "uint32");
        assert_eq!(get("llama.rope.freq_base").value, json!(10000.0));
        let arr = get("tokenizer.ggml.tokens");
        assert_eq!((arr.kind, arr.editable, arr.value.clone()), ("array", false, json!("2 strings")));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn edits_accumulate_across_writes() {
        let d = tmpdir();
        let p = tiny(&d);
        write_copy(&p, &[Edit { key: "general.name".into(), value: json!("One") }]).expect("first");
        let c = write_copy(&p, &[Edit { key: "llama.context_length".into(), value: json!(16384) }]).expect("second");
        assert_eq!(c.edits.len(), 2);
        let g = cerebro_engine::gguf::Gguf::open(Path::new(&c.path)).expect("read");
        assert_eq!(g.str("general.name"), Some("One"));
        assert_eq!(g.u64("llama.context_length").ok(), Some(16384));
        std::fs::remove_dir_all(&d).ok();
    }

    /// A real model's copy (in a temp folder) loads in the engine and writes the same text.
    /// `cargo test -p cerebro-desktop real_copy -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn a_real_copy_loads_and_writes_the_same() {
        let src = Path::new(r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf");
        let d = tmpdir();
        let p = d.join("q.gguf");
        std::fs::copy(src, &p).expect("copy the test model");
        let c = write_copy(&p, &[Edit { key: "general.name".into(), value: json!("Edited Qwen") }, Edit { key: "qwen3.context_length".into(), value: json!(8192) }]).expect("write");
        let run = |path: &Path| {
            let m = cerebro_engine::model::Model::load(path).expect("load");
            let tok = cerebro_engine::tokenizer::Tokenizer::from_gguf(m.gguf()).expect("tokenizer");
            let mut s = cerebro_engine::model::Session::new(&m, 512);
            s.feed(&m, &tok.encode("The capital of France is")).expect("feed");
            let mut out = Vec::new();
            for _ in 0..8 {
                let t = cerebro_engine::sample::argmax(&s.logits);
                out.push(t);
                s.step(&m, t);
            }
            (m.cfg.ctx_train, tok.decode(&out))
        };
        let (a, b) = (run(&p), run(Path::new(&c.path)));
        println!("original {a:?}, copy {b:?}");
        assert_eq!(a.1, b.1);
        assert_eq!((a.0, b.0), (40960, 8192));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_real_model_header_round_trips() {
        let p = Path::new(r"C:\models\dev\Qwen3-0.6B-Q8_0.gguf");
        if !p.exists() {
            return;
        }
        let (entries, template) = detail(p).expect("detail");
        assert!(entries.iter().any(|e| e.key == "qwen3.block_count" && !e.editable));
        assert!(template.is_some_and(|t| t.contains("im_start")));
    }
}
