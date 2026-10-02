//! The model facts candidate generation needs, read from a GGUF file's header.
//!
//! The tuner sizes its candidates from three facts: how many transformer blocks the model has,
//! whether it is a mixture of experts, and how many bytes each block's expert and dense FFN
//! weights take. A GGUF header carries all three: the metadata section names the block and expert
//! counts and the attention shape (for the KV cache estimate), and the tensor-info section names
//! every tensor with its offset into the data section.
//!
//! Tensor sizes come from the offsets, not from a table of quantization block sizes: sorted by
//! offset, each tensor's size is the distance to the next one (the last runs to the end of the
//! file). That needs no knowledge of any `ggml_type`, so a quantization added upstream after this
//! was written still sizes correctly; the cost is up to `general.alignment - 1` bytes of padding
//! per tensor, which is noise against weights measured in megabytes.
//!
//! The file is untrusted input, so every count and length is bounded before anything is allocated
//! or iterated, nesting is capped, and a truncated header or a tensor offset past the end of the
//! file is an error, never a partial result. What is kept grows only with what the tuner reads:
//! tensor names are capped at llama.cpp's own limit, metadata keys longer than any the tuner reads
//! are skipped unread, and string values are kept only for the keys it reads
//! (`general.architecture`), so a hostile header of long strings costs reading time, not memory. Only the header is read; the tensor data is never
//! touched, so a file cut short inside its last tensor reads with that tensor smaller than it
//! should be (llama.cpp then refuses to load it, and the bench run fails).
//!
//! Split GGUFs (`split.count > 1`) are refused for now: the first part carries only some tensors,
//! so sizes read from it alone would understate the model.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::GgufError;

/// Most metadata entries a header may carry. Real models carry a few dozen; tokenizer vocabularies
/// are arrays inside one entry, not separate entries.
pub const MAX_METADATA_ENTRIES: u64 = 1 << 16;

/// Most tensors a header may describe. A 671B MoE has about 1,000 per split.
pub const MAX_TENSORS: u64 = 1 << 20;

/// Longest string (key, value or tensor name) accepted, in bytes. Chat templates are the longest
/// legitimate strings, at tens of kilobytes.
pub const MAX_STRING_BYTES: u64 = 1 << 24;

/// Most elements one metadata array may hold. The largest legitimate arrays are tokenizer
/// vocabularies and merges, a few hundred thousand entries.
pub const MAX_ARRAY_ELEMENTS: u64 = 1 << 24;

/// Most dimensions a tensor may have (`GGML_MAX_DIMS`).
const MAX_DIMS: u32 = 4;

/// Highest transformer block index accepted in a tensor name.
const MAX_BLOCK_INDEX: usize = 1 << 12;

/// Integer arrays up to this length are read for their largest element (per-layer head counts);
/// longer ones are skipped unread.
const SMALL_INT_ARRAY: u64 = 1 << 12;

/// Arrays of arrays are legal GGUF but unused by real models; deeper nesting is refused.
const MAX_ARRAY_DEPTH: u32 = 2;

/// Longest key kept for lookup; longer keys are skipped unread (never one the tuner reads).
const MAX_KEPT_KEY_BYTES: u64 = 256;

/// Longest tensor name accepted: llama.cpp refuses names of `GGML_MAX_NAME` (64) bytes or more
/// (`gguf.cpp`: "tensor name ... is too long"), so no model it loads has a longer one.
pub const MAX_TENSOR_NAME_BYTES: u64 = 63;

/// The only metadata keys whose string values are kept; every other string value is skipped.
const KEPT_TEXT_KEYS: &[&str] = &["general.architecture"];

/// The data-section alignment when `general.alignment` is absent (GGUF specification).
const DEFAULT_ALIGNMENT: u64 = 32;

/// Weight bytes inside one transformer block (`blk.N.*`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockBytes {
    /// Every tensor in the block.
    pub total: u64,
    /// Routed expert tensors (`*_exps*`): what `-ncmoe` keeps in host memory.
    pub experts: u64,
    /// Dense FFN tensors (`ffn_gate.`, `ffn_up.`, `ffn_down.`): what a dense FFN offload keeps in
    /// host memory.
    pub dense_ffn: u64,
}

/// What the tuner knows about a model: its shape and where its bytes are.
///
/// Built by [`ModelFacts::from_gguf`], or directly (every field is public) by a caller that knows
/// the model some other way, and by tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFacts {
    /// `general.architecture`, e.g. `qwen2`, `qwen3moe`.
    pub architecture: String,
    /// Transformer blocks (`<arch>.block_count`): the range of `-ncmoe` and FFN offload steps.
    pub block_count: u32,
    /// Routed experts per MoE layer (`<arch>.expert_count`); `None` or `Some(0 | 1)` for a dense
    /// model.
    pub expert_count: Option<u32>,
    /// Training context length (`<arch>.context_length`): the deepest depth worth measuring.
    pub context_length: Option<u32>,
    /// KV heads per layer (`<arch>.attention.head_count_kv`, the largest when given per layer;
    /// `head_count` when absent, as for multi-head attention).
    pub kv_heads: Option<u32>,
    /// Width of one K head (`<arch>.attention.key_length`, else embedding width / heads).
    pub key_length: Option<u32>,
    /// Width of one V head (`<arch>.attention.value_length`, else the K width).
    pub value_length: Option<u32>,
    /// `token_embd.*` bytes: llama.cpp keeps the input embedding in host memory whatever `-ngl`
    /// says, so it never counts against VRAM.
    pub input_bytes: u64,
    /// Every other tensor outside a block (output head, final norm): on the GPU under full
    /// offload.
    pub other_bytes: u64,
    /// Per-block bytes, indexed by block number.
    pub blocks: Vec<BlockBytes>,
}

impl ModelFacts {
    /// Read the facts from the GGUF file at `path` (header only).
    ///
    /// # Errors
    /// [`GgufError`] for an unreadable, truncated, oversized, split or malformed file.
    pub fn from_gguf(path: &Path) -> Result<Self, GgufError> {
        let file = File::open(path).map_err(|e| GgufError::Io(e.to_string()))?;
        Self::from_reader(file)
    }

    /// Read the facts from any seekable GGUF byte source.
    ///
    /// # Errors
    /// As [`ModelFacts::from_gguf`].
    pub fn from_reader<R: Read + Seek>(reader: R) -> Result<Self, GgufError> {
        Header::read(reader)?.into_facts()
    }

    /// Whether this is a mixture-of-experts model with routed expert tensors to offload.
    pub fn is_moe(&self) -> bool {
        self.expert_count.is_some_and(|n| n > 1) && self.blocks.iter().any(|b| b.experts > 0)
    }

    /// Bytes of weights on the GPU when every block is offloaded (`-ngl 999`): everything except
    /// the input embedding.
    pub fn full_offload_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .fold(self.other_bytes, |sum, b| sum.saturating_add(b.total))
    }

    /// Expert bytes in the first `n` blocks: what `-ncmoe n` moves to host memory.
    pub fn expert_bytes_of_first(&self, n: u32) -> u64 {
        self.first(n).map(|b| b.experts).fold(0, u64::saturating_add)
    }

    /// Dense FFN bytes in the first `n` blocks: what a dense FFN offload of `n` blocks moves to
    /// host memory.
    pub fn dense_ffn_bytes_of_first(&self, n: u32) -> u64 {
        self.first(n).map(|b| b.dense_ffn).fold(0, u64::saturating_add)
    }

    fn first(&self, n: u32) -> impl Iterator<Item = &BlockBytes> {
        self.blocks
            .iter()
            .take(usize::try_from(n).unwrap_or(usize::MAX))
    }

    /// KV cache bytes per token of context at `bytes_per_element` (2 for f16, 34/32 for q8_0,
    /// given as a numerator and denominator), across every block. `None` when the attention shape
    /// is unknown.
    pub fn kv_bytes_per_token(&self, numerator: u64, denominator: u64) -> Option<u64> {
        let heads = u64::from(self.kv_heads?);
        let k = u64::from(self.key_length?);
        let v = u64::from(self.value_length.or(self.key_length)?);
        let elements = u64::from(self.block_count)
            .checked_mul(heads)?
            .checked_mul(k.checked_add(v)?)?;
        elements.checked_mul(numerator)?.checked_div(denominator)
    }
}

/// One scalar metadata value, as far as the tuner cares.
#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    /// Any integer type, when non-negative.
    Unsigned(u64),
    Text(String),
    /// Something the tuner never reads (floats, bools, negative integers).
    Other,
}

/// GGUF metadata value type codes.
mod ty {
    pub const U8: u32 = 0;
    pub const I8: u32 = 1;
    pub const U16: u32 = 2;
    pub const I16: u32 = 3;
    pub const U32: u32 = 4;
    pub const I32: u32 = 5;
    pub const F32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const U64: u32 = 10;
    pub const I64: u32 = 11;
    pub const F64: u32 = 12;
}

/// The parsed parts of a header the facts are built from.
struct Header {
    scalars: BTreeMap<String, Scalar>,
    /// For integer arrays: the largest element (per-layer `head_count_kv`).
    array_max: BTreeMap<String, u64>,
    /// `(name, offset)` for every tensor.
    tensors: Vec<(String, u64)>,
    /// Bytes from the data section's start to the end of the file.
    data_len: u64,
    alignment: u64,
}

struct Cursor<R> {
    inner: BufReader<R>,
}

impl<R: Read + Seek> Cursor<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf).map_err(io_or_truncated)?;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        self.string_of(len)
    }

    /// The string body of `len` bytes (its length already read).
    fn string_of(&mut self, len: u64) -> Result<String, GgufError> {
        let len = bounded(len, MAX_STRING_BYTES, "string length")?;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).map_err(io_or_truncated)?;
        String::from_utf8(buf).map_err(|_| GgufError::NotUtf8)
    }

    fn skip(&mut self, bytes: u64) -> Result<(), GgufError> {
        // Callers bound every skip (strings and arrays) far below this; `seek_relative` takes i64.
        let bytes = i64::try_from(bytes).map_err(|_| GgufError::TooLarge {
            what: "skip",
            value: bytes,
            limit: i64::MAX as u64,
        })?;
        self.inner
            .seek_relative(bytes)
            .map_err(|e| GgufError::Io(e.to_string()))
    }

    fn skip_string(&mut self) -> Result<(), GgufError> {
        let len = self.u64()?;
        bounded(len, MAX_STRING_BYTES, "string length")?;
        self.skip(len)
    }

    fn position(&mut self) -> Result<u64, GgufError> {
        self.inner
            .stream_position()
            .map_err(|e| GgufError::Io(e.to_string()))
    }

    /// One scalar of type `ty` (not an array).
    fn scalar(&mut self, ty: u32) -> Result<Scalar, GgufError> {
        Ok(match ty {
            ty::U8 => Scalar::Unsigned(u64::from(self.bytes::<1>()?[0])),
            ty::I8 => signed(i64::from(i8::from_le_bytes(self.bytes()?))),
            ty::U16 => Scalar::Unsigned(u64::from(u16::from_le_bytes(self.bytes()?))),
            ty::I16 => signed(i64::from(i16::from_le_bytes(self.bytes()?))),
            ty::U32 => Scalar::Unsigned(u64::from(self.u32()?)),
            ty::I32 => signed(i64::from(i32::from_le_bytes(self.bytes()?))),
            ty::U64 => Scalar::Unsigned(self.u64()?),
            ty::I64 => signed(i64::from_le_bytes(self.bytes()?)),
            ty::F32 => {
                self.bytes::<4>()?;
                Scalar::Other
            }
            ty::F64 => {
                self.bytes::<8>()?;
                Scalar::Other
            }
            ty::BOOL => {
                self.bytes::<1>()?;
                Scalar::Other
            }
            ty::STRING => Scalar::Text(self.string()?),
            other => return Err(GgufError::BadValueType(other)),
        })
    }

    /// Read an array's element type and count, then skip its body. Returns the largest
    /// non-negative element of a short integer array, `None` otherwise.
    fn array(&mut self, depth: u32) -> Result<Option<u64>, GgufError> {
        if depth > MAX_ARRAY_DEPTH {
            return Err(GgufError::TooDeep);
        }
        let elem = self.u32()?;
        let count = self.u64()?;
        bounded(count, MAX_ARRAY_ELEMENTS, "array length")?;
        let fixed = match elem {
            ty::U8 | ty::I8 | ty::BOOL => Some(1),
            ty::U16 | ty::I16 => Some(2),
            ty::U32 | ty::I32 | ty::F32 => Some(4),
            ty::U64 | ty::I64 | ty::F64 => Some(8),
            ty::STRING | ty::ARRAY => None,
            other => return Err(GgufError::BadValueType(other)),
        };
        let integer = !matches!(elem, ty::F32 | ty::F64 | ty::BOOL | ty::STRING | ty::ARRAY);
        // Small integer arrays (per-layer head counts) are read for their maximum; everything
        // else is skipped without reading its elements.
        if integer && count <= SMALL_INT_ARRAY {
            let mut max = None;
            for _ in 0..count {
                if let Scalar::Unsigned(v) = self.scalar(elem)? {
                    max = Some(max.map_or(v, |m: u64| m.max(v)));
                }
            }
            return Ok(max);
        }
        match fixed {
            Some(size) => self.skip(count.saturating_mul(size))?,
            None if elem == ty::STRING => {
                for _ in 0..count {
                    self.skip_string()?;
                }
            }
            None => {
                for _ in 0..count {
                    self.array(depth + 1)?;
                }
            }
        }
        Ok(None)
    }
}

impl Header {
    fn read<R: Read + Seek>(mut reader: R) -> Result<Self, GgufError> {
        let file_len = reader
            .seek(SeekFrom::End(0))
            .and_then(|len| reader.seek(SeekFrom::Start(0)).map(|_| len))
            .map_err(|e| GgufError::Io(e.to_string()))?;
        let mut c = Cursor {
            inner: BufReader::new(reader),
        };
        if &c.bytes::<4>()? != b"GGUF" {
            return Err(GgufError::NotGguf);
        }
        let version = c.u32()?;
        // Version 1 used 32-bit counts and lengths; nothing current writes it.
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = c.u64()?;
        bounded(tensor_count, MAX_TENSORS, "tensor count")?;
        let kv_count = c.u64()?;
        bounded(kv_count, MAX_METADATA_ENTRIES, "metadata entry count")?;

        let mut scalars = BTreeMap::new();
        let mut array_max = BTreeMap::new();
        for _ in 0..kv_count {
            // A key longer than any the tuner reads is skipped without being allocated.
            let key_len = c.u64()?;
            let key = if key_len <= MAX_KEPT_KEY_BYTES {
                Some(c.string_of(key_len)?)
            } else {
                bounded(key_len, MAX_STRING_BYTES, "string length")?;
                c.skip(key_len)?;
                None
            };
            let value_type = c.u32()?;
            if value_type == ty::ARRAY {
                let max = c.array(1)?;
                if let (Some(key), Some(max)) = (key, max) {
                    array_max.insert(key, max);
                }
            } else if value_type == ty::STRING
                && !key.as_deref().is_some_and(|k| KEPT_TEXT_KEYS.contains(&k))
            {
                c.skip_string()?;
            } else {
                let value = c.scalar(value_type)?;
                if let Some(key) = key {
                    scalars.insert(key, value);
                }
            }
        }

        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name_len = c.u64()?;
            if name_len > MAX_TENSOR_NAME_BYTES {
                return Err(GgufError::BadTensor(format!(
                    "a tensor name of {name_len} bytes is longer than llama.cpp accepts \
                     ({MAX_TENSOR_NAME_BYTES})"
                )));
            }
            let name = c.string_of(name_len)?;
            let dims = c.u32()?;
            if dims == 0 || dims > MAX_DIMS {
                return Err(GgufError::BadTensor(format!("{name} has {dims} dimensions")));
            }
            for _ in 0..dims {
                c.u64()?;
            }
            let _ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push((name, offset));
        }

        let alignment = match scalars.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(Scalar::Unsigned(a)) if a.is_power_of_two() => *a,
            Some(_) => return Err(GgufError::BadTensor("general.alignment is not a power of two".into())),
        };
        let infos_end = c.position()?;
        let data_start = infos_end
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| GgufError::BadTensor("data section start overflows".into()))?;
        let data_len = file_len.checked_sub(data_start).ok_or(GgufError::Truncated)?;
        Ok(Self {
            scalars,
            array_max,
            tensors,
            data_len,
            alignment,
        })
    }

    fn unsigned(&self, key: &str) -> Option<u64> {
        match self.scalars.get(key) {
            Some(Scalar::Unsigned(v)) => Some(*v),
            _ => None,
        }
    }

    fn u32_key(&self, key: &str) -> Option<u32> {
        self.unsigned(key).and_then(|v| u32::try_from(v).ok())
    }

    fn into_facts(self) -> Result<ModelFacts, GgufError> {
        if let Some(count) = self.unsigned("split.count").filter(|&n| n > 1) {
            return Err(GgufError::SplitModel { count });
        }
        let architecture = match self.scalars.get("general.architecture") {
            Some(Scalar::Text(a)) => a.clone(),
            _ => return Err(GgufError::MissingKey("general.architecture".into())),
        };
        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let block_count = self
            .u32_key(&key("block_count"))
            .ok_or_else(|| GgufError::MissingKey(key("block_count")))?;
        // The planner searches offload steps over the block count; bound it like any count.
        if u64::from(block_count) > MAX_BLOCK_INDEX as u64 {
            return Err(GgufError::TooLarge {
                what: "block count",
                value: u64::from(block_count),
                limit: MAX_BLOCK_INDEX as u64,
            });
        }
        let heads = self.u32_key(&key("attention.head_count"));
        let kv_heads = self
            .u32_key(&key("attention.head_count_kv"))
            .or_else(|| {
                self.array_max
                    .get(&key("attention.head_count_kv"))
                    .and_then(|&v| u32::try_from(v).ok())
            })
            .or(heads)
            .filter(|&n| n > 0);
        let key_length = self.u32_key(&key("attention.key_length")).or_else(|| {
            let width = self.u32_key(&key("embedding_length"))?;
            width.checked_div(heads.filter(|&h| h > 0)?)
        });
        let value_length = self.u32_key(&key("attention.value_length")).or(key_length);

        let sizes = tensor_sizes(&self.tensors, self.data_len, self.alignment)?;
        let mut input_bytes = 0u64;
        let mut other_bytes = 0u64;
        let mut blocks = vec![BlockBytes::default(); usize::try_from(block_count).unwrap_or(0)];
        for ((name, _), size) in self.tensors.iter().zip(sizes) {
            match block_of(name) {
                Some((index, rest)) => {
                    if index >= MAX_BLOCK_INDEX {
                        return Err(GgufError::BadTensor(format!("{name}: block index too large")));
                    }
                    if index >= blocks.len() {
                        blocks.resize(index + 1, BlockBytes::default());
                    }
                    let block = &mut blocks[index];
                    block.total = block.total.saturating_add(size);
                    if rest.contains("_exps") {
                        block.experts = block.experts.saturating_add(size);
                    } else if ["ffn_gate.", "ffn_up.", "ffn_down."]
                        .iter()
                        .any(|p| rest.starts_with(p))
                    {
                        block.dense_ffn = block.dense_ffn.saturating_add(size);
                    }
                }
                None if name.starts_with("token_embd.") => {
                    input_bytes = input_bytes.saturating_add(size);
                }
                None => other_bytes = other_bytes.saturating_add(size),
            }
        }
        let expert_count = self.u32_key(&key("expert_count"));
        let context_length = self.u32_key(&key("context_length"));
        Ok(ModelFacts {
            architecture,
            block_count,
            expert_count,
            context_length,
            kv_heads,
            key_length,
            value_length,
            input_bytes,
            other_bytes,
            blocks,
        })
    }
}

/// Each tensor's size, in input order, from the distance to the next offset.
fn tensor_sizes(tensors: &[(String, u64)], data_len: u64, alignment: u64) -> Result<Vec<u64>, GgufError> {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].1);
    let mut sizes = vec![0u64; tensors.len()];
    for (pos, &i) in order.iter().enumerate() {
        let (name, offset) = &tensors[i];
        if offset % alignment != 0 {
            return Err(GgufError::BadTensor(format!("{name}: offset {offset} is not aligned to {alignment}")));
        }
        let end = match order.get(pos + 1) {
            Some(&next) => tensors[next].1,
            None => data_len,
        };
        sizes[i] = end
            .checked_sub(*offset)
            .ok_or_else(|| GgufError::BadTensor(format!("{name}: offset {offset} is past the end of the file")))?;
    }
    Ok(sizes)
}

/// `blk.<n>.<rest>` as `(n, rest)`.
fn block_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (index, rest) = rest.split_once('.')?;
    if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((index.parse().ok()?, rest))
}

fn signed(v: i64) -> Scalar {
    u64::try_from(v).map_or(Scalar::Other, Scalar::Unsigned)
}

fn bounded(value: u64, limit: u64, what: &'static str) -> Result<usize, GgufError> {
    if value > limit {
        return Err(GgufError::TooLarge { what, value, limit });
    }
    usize::try_from(value).map_err(|_| GgufError::TooLarge { what, value, limit })
}

fn io_or_truncated(e: std::io::Error) -> GgufError {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        GgufError::Truncated
    } else {
        GgufError::Io(e.to_string())
    }
}

/// Builds GGUF headers in memory for tests (this module's and the tuner's).
#[cfg(test)]
pub(crate) mod build {
    /// A metadata value to write.
    pub enum V<'a> {
        U32(u32),
        Str(&'a str),
        F32(f32),
        StrArray(&'a [&'a str]),
        I32Array(&'a [i32]),
    }

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A version-3 GGUF with `kvs` and `tensors` (`name`, `bytes`), the data section filled with
    /// zeros, tensors laid out in order at 32-byte alignment.
    pub fn gguf(kvs: &[(&str, V<'_>)], tensors: &[(&str, u64)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((kvs.len() as u64).to_le_bytes());
        for (key, value) in kvs {
            string(&mut out, key);
            match value {
                V::U32(v) => {
                    out.extend(4u32.to_le_bytes());
                    out.extend(v.to_le_bytes());
                }
                V::Str(s) => {
                    out.extend(8u32.to_le_bytes());
                    string(&mut out, s);
                }
                V::F32(v) => {
                    out.extend(6u32.to_le_bytes());
                    out.extend(v.to_le_bytes());
                }
                V::StrArray(items) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(8u32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    for s in *items {
                        string(&mut out, s);
                    }
                }
                V::I32Array(items) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(5u32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    for v in *items {
                        out.extend(v.to_le_bytes());
                    }
                }
            }
        }
        let mut offset = 0u64;
        for (name, bytes) in tensors {
            string(&mut out, name);
            out.extend(1u32.to_le_bytes());
            out.extend(bytes.to_le_bytes());
            out.extend(0u32.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offset += bytes.next_multiple_of(32);
        }
        let padded = (out.len() as u64).next_multiple_of(32) as usize;
        out.resize(padded, 0);
        out.resize(padded + offset as usize, 0);
        out
    }

    /// A small MoE-shaped model: `blocks` blocks, each with attention, a router and three expert
    /// tensors; `expert_bytes` per expert tensor.
    pub fn moe(blocks: u32, expert_bytes: u64) -> Vec<u8> {
        let names: Vec<(String, u64)> = (0..blocks)
            .flat_map(|i| {
                [
                    (format!("blk.{i}.attn_q.weight"), 1024),
                    (format!("blk.{i}.ffn_gate_inp.weight"), 256),
                    (format!("blk.{i}.ffn_gate_exps.weight"), expert_bytes),
                    (format!("blk.{i}.ffn_up_exps.weight"), expert_bytes),
                    (format!("blk.{i}.ffn_down_exps.weight"), expert_bytes),
                ]
            })
            .chain([("token_embd.weight".to_string(), 4096), ("output.weight".to_string(), 2048)])
            .collect();
        let tensors: Vec<(&str, u64)> = names.iter().map(|(n, b)| (n.as_str(), *b)).collect();
        gguf(
            &[
                ("general.architecture", V::Str("qwen3moe")),
                ("qwen3moe.block_count", V::U32(blocks)),
                ("qwen3moe.expert_count", V::U32(128)),
                ("qwen3moe.context_length", V::U32(40960)),
                ("qwen3moe.attention.head_count", V::U32(32)),
                ("qwen3moe.attention.head_count_kv", V::U32(4)),
                ("qwen3moe.attention.key_length", V::U32(128)),
                ("qwen3moe.attention.value_length", V::U32(128)),
            ],
            &tensors,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::build::{gguf, moe, V};
    use super::*;

    fn facts(bytes: Vec<u8>) -> Result<ModelFacts, GgufError> {
        ModelFacts::from_reader(std::io::Cursor::new(bytes))
    }

    #[test]
    fn a_moe_header_yields_per_block_expert_bytes_and_the_attention_shape() {
        let f = facts(moe(4, 1 << 20)).unwrap();
        assert_eq!(f.architecture, "qwen3moe");
        assert_eq!(f.block_count, 4);
        assert!(f.is_moe());
        assert_eq!(f.blocks.len(), 4);
        // Sizes come from offset gaps: 32-byte aligned tensors, so these are exact.
        assert_eq!(f.blocks[0], BlockBytes { total: 1024 + 256 + 3 * (1 << 20), experts: 3 << 20, dense_ffn: 0 });
        assert_eq!(f.input_bytes, 4096);
        assert_eq!(f.other_bytes, 2048);
        assert_eq!(f.expert_bytes_of_first(2), 6 << 20);
        assert_eq!(f.expert_bytes_of_first(99), 12 << 20, "past the last block counts every block");
        assert_eq!(f.full_offload_bytes(), 2048 + 4 * (1024 + 256 + (3 << 20)));
        // 4 blocks x 4 KV heads x (128 + 128) x 2 bytes (f16).
        assert_eq!(f.kv_bytes_per_token(2, 1), Some(4 * 4 * 256 * 2));
        assert_eq!(f.context_length, Some(40960));
    }

    #[test]
    fn a_dense_header_classifies_ffn_tensors_and_derives_the_head_width() {
        let bytes = gguf(
            &[
                ("general.architecture", V::Str("llama")),
                ("general.name", V::Str("tiny")),
                ("llama.block_count", V::U32(2)),
                ("llama.embedding_length", V::U32(4096)),
                ("llama.attention.head_count", V::U32(32)),
                ("llama.attention.head_count_kv", V::I32Array(&[8, 4])),
                ("llama.rope.freq_base", V::F32(10000.0)),
                ("tokenizer.ggml.tokens", V::StrArray(&["a", "bb", "ccc"])),
            ],
            &[
                ("token_embd.weight", 64),
                ("blk.0.attn_k.weight", 96),
                ("blk.0.ffn_up.weight", 320),
                ("blk.0.ffn_gate.weight", 320),
                ("blk.0.ffn_down.weight", 320),
                ("blk.1.ffn_up.weight", 320),
                ("blk.1.ffn_norm.weight", 32),
            ],
        );
        let f = facts(bytes).unwrap();
        assert!(!f.is_moe());
        assert_eq!(f.blocks[0], BlockBytes { total: 96 + 3 * 320, experts: 0, dense_ffn: 960 });
        // `ffn_norm` is not an FFN weight the offload moves.
        assert_eq!(f.blocks[1], BlockBytes { total: 352, experts: 0, dense_ffn: 320 });
        assert_eq!(f.kv_heads, Some(8), "the largest per-layer KV head count");
        assert_eq!(f.key_length, Some(128), "4096 / 32 heads");
        assert_eq!(f.value_length, Some(128));
        assert_eq!(f.dense_ffn_bytes_of_first(1), 960);
    }

    #[test]
    fn hostile_or_broken_headers_are_refused_not_half_read() {
        assert!(matches!(facts(b"GGML\0\0\0\0".to_vec()), Err(GgufError::NotGguf)));
        let mut v1 = b"GGUF".to_vec();
        v1.extend(1u32.to_le_bytes());
        assert!(matches!(facts(v1), Err(GgufError::UnsupportedVersion(1))));

        let good = moe(2, 64);
        for cut in [10, 30, 200, 700] {
            let err = facts(good[..cut].to_vec()).unwrap_err();
            assert!(matches!(err, GgufError::Truncated), "cut {cut}: {err:?}");
        }
        // Cut before the last tensor's offset (`output.weight`, 2048 bytes, last in the file).
        let err = facts(good[..good.len() - 2048 - 32].to_vec()).unwrap_err();
        assert!(matches!(err, GgufError::BadTensor(_)), "{err:?}");

        // A string length far beyond any real header is refused before allocating.
        let mut huge = b"GGUF".to_vec();
        huge.extend(3u32.to_le_bytes());
        huge.extend(0u64.to_le_bytes());
        huge.extend(1u64.to_le_bytes());
        huge.extend(u64::MAX.to_le_bytes());
        assert!(matches!(facts(huge), Err(GgufError::TooLarge { what: "string length", .. })));

        let split = gguf(
            &[
                ("general.architecture", V::Str("llama")),
                ("llama.block_count", V::U32(2)),
                ("split.count", V::U32(3)),
            ],
            &[],
        );
        assert!(matches!(facts(split), Err(GgufError::SplitModel { count: 3 })));

        let absurd = gguf(
            &[("general.architecture", V::Str("llama")), ("llama.block_count", V::U32(u32::MAX))],
            &[],
        );
        assert!(matches!(facts(absurd), Err(GgufError::TooLarge { what: "block count", .. })));

        let no_blocks = gguf(&[("general.architecture", V::Str("llama"))], &[]);
        assert!(matches!(facts(no_blocks), Err(GgufError::MissingKey(k)) if k == "llama.block_count"));
    }

    /// A v3 header prefix: magic, version, tensor count, metadata count.
    fn prefix(tensors: u64, kvs: u64) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(tensors.to_le_bytes());
        out.extend(kvs.to_le_bytes());
        out
    }

    fn put_string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    #[test]
    fn nesting_value_types_alignment_offsets_and_name_lengths_are_each_refused() {
        // Arrays nested three deep: array of array of array of u8.
        let mut deep = prefix(0, 1);
        put_string(&mut deep, "x.deep");
        deep.extend(9u32.to_le_bytes()); // ARRAY
        deep.extend(9u32.to_le_bytes()); // of ARRAY
        deep.extend(1u64.to_le_bytes());
        deep.extend(9u32.to_le_bytes()); // of ARRAY
        deep.extend(1u64.to_le_bytes());
        deep.extend(0u32.to_le_bytes()); // of U8
        deep.extend(1u64.to_le_bytes());
        deep.push(7);
        assert!(matches!(facts(deep), Err(GgufError::TooDeep)));

        // Two deep is legal GGUF and is read past.
        let mut two = prefix(0, 2);
        put_string(&mut two, "x.two");
        two.extend(9u32.to_le_bytes());
        two.extend(9u32.to_le_bytes());
        two.extend(1u64.to_le_bytes());
        two.extend(0u32.to_le_bytes());
        two.extend(1u64.to_le_bytes());
        two.push(7);
        put_string(&mut two, "general.architecture");
        two.extend(8u32.to_le_bytes());
        put_string(&mut two, "llama");
        // The (empty) data section starts at the next 32-byte boundary.
        two.resize(two.len().next_multiple_of(32), 0);
        assert!(matches!(facts(two), Err(GgufError::MissingKey(k)) if k == "llama.block_count"));

        // An undefined scalar type, and an undefined array element type.
        let mut bad = prefix(0, 1);
        put_string(&mut bad, "x.bad");
        bad.extend(13u32.to_le_bytes());
        assert!(matches!(facts(bad), Err(GgufError::BadValueType(13))));
        let mut bad_elem = prefix(0, 1);
        put_string(&mut bad_elem, "x.bad");
        bad_elem.extend(9u32.to_le_bytes());
        bad_elem.extend(99u32.to_le_bytes());
        bad_elem.extend(0u64.to_le_bytes());
        assert!(matches!(facts(bad_elem), Err(GgufError::BadValueType(99))));

        // An alignment of 0 or 3 is not a power of two.
        for alignment in [0, 3] {
            let bytes = gguf(
                &[
                    ("general.architecture", V::Str("llama")),
                    ("llama.block_count", V::U32(1)),
                    ("general.alignment", V::U32(alignment)),
                ],
                &[("blk.0.ffn_up.weight", 64)],
            );
            let err = facts(bytes).unwrap_err();
            assert!(matches!(&err, GgufError::BadTensor(m) if m.contains("power of two")), "{alignment}: {err:?}");
        }

        // A tensor offset that is not a multiple of the alignment. The builder writes offsets at
        // a fixed position after each name: patch the second tensor's offset from 64 to 65.
        let mut bytes = gguf(
            &[("general.architecture", V::Str("llama")), ("llama.block_count", V::U32(1))],
            &[("blk.0.ffn_up.weight", 64), ("blk.0.ffn_down.weight", 64)],
        );
        let name = b"blk.0.ffn_down.weight";
        let at = bytes.windows(name.len()).position(|w| w == name).unwrap() + name.len() + 4 + 8 + 4;
        assert_eq!(bytes[at..at + 8], 64u64.to_le_bytes());
        bytes[at..at + 8].copy_from_slice(&65u64.to_le_bytes());
        let err = facts(bytes).unwrap_err();
        assert!(matches!(&err, GgufError::BadTensor(m) if m.contains("not aligned")), "{err:?}");

        // Tensor names: 63 bytes is llama.cpp's longest; 64 is refused before it is read.
        let longest = format!("blk.0.{}", "x".repeat(57));
        assert_eq!(longest.len(), 63);
        let ok = gguf(
            &[("general.architecture", V::Str("llama")), ("llama.block_count", V::U32(1))],
            &[(longest.as_str(), 64)],
        );
        assert_eq!(facts(ok).unwrap().blocks[0].total, 64);
        let too_long = format!("{longest}x");
        let refused = gguf(
            &[("general.architecture", V::Str("llama")), ("llama.block_count", V::U32(1))],
            &[(too_long.as_str(), 64)],
        );
        let err = facts(refused).unwrap_err();
        assert!(matches!(&err, GgufError::BadTensor(m) if m.contains("64 bytes")), "{err:?}");
    }

    #[test]
    fn long_keys_and_unread_string_values_are_skipped_not_kept() {
        let long_key = "k".repeat(300);
        let template = "t".repeat(10_000);
        let bytes = gguf(
            &[
                (long_key.as_str(), V::U32(5)),
                ("tokenizer.chat_template", V::Str(&template)),
                ("general.architecture", V::Str("llama")),
                ("llama.block_count", V::U32(1)),
            ],
            &[("blk.0.ffn_up.weight", 64)],
        );
        let header = Header::read(std::io::Cursor::new(bytes)).unwrap();
        assert!(!header.scalars.keys().any(|k| k.len() > 256), "a long key is never kept");
        assert!(!header.scalars.contains_key("tokenizer.chat_template"));
        assert_eq!(header.scalars.get("general.architecture"), Some(&Scalar::Text("llama".into())));
        assert_eq!(header.unsigned("llama.block_count"), Some(1));
        assert_eq!(header.into_facts().unwrap().architecture, "llama");
    }

    #[test]
    fn single_byte_mutations_of_a_valid_header_never_panic() {
        let good = moe(3, 64);
        for i in 0..good.len().min(4096) {
            for b in [0u8, 0x01, 0x7F, 0x80, 0xFF] {
                let mut m = good.clone();
                m[i] = b;
                let _ = facts(m);
            }
        }
        // Truncation at every length too.
        for cut in 0..good.len().min(4096) {
            let _ = facts(good[..cut].to_vec());
        }
    }

    #[test]
    fn block_names_parse_only_with_a_numeric_index() {
        assert_eq!(block_of("blk.17.ffn_up.weight"), Some((17, "ffn_up.weight")));
        assert_eq!(block_of("blk..x"), None);
        assert_eq!(block_of("blk.a1.x"), None);
        assert_eq!(block_of("output.weight"), None);
    }
}
