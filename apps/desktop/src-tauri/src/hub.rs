//! Model Hub: live Hugging Face search, the GGUF files of a repository with their sizes, and
//! downloads into Cerebro's model folder with progress. The only network feature of the app; each
//! request goes to huggingface.co and nowhere else.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

const HF: &str = "https://huggingface.co";
/// Only the GGUF header is ever fetched, and only this much of it: the metadata that sizes memory
/// sits near the front; the weights never leave Hugging Face.
const MAX_HEADER: u64 = 8 << 20;

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("cerebro/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|_| "Could not start the network client.".to_owned())
}

/// Percent-encodes a query value (unreserved characters pass through).
fn encode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// One repository in the search results.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HubRepo {
    pub id: String,
    pub author: String,
    pub downloads: u64,
    pub likes: u64,
    pub updated: String,
    /// The repository's `license:` tag, when it publishes one (e.g. "apache-2.0"). Shown before a
    /// file is downloaded so the user can read what the model allows.
    pub license: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Deserialize)]
struct ApiModel {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "lastModified")]
    last_modified: String,
    #[serde(default)]
    tags: Vec<String>,
}

/// Repositories with GGUF files matching `query`, most downloaded first.
pub fn search(query: &str) -> Result<Vec<HubRepo>, String> {
    let q = query.trim();
    if q.len() > 120 {
        return Err("That search is too long.".into());
    }
    let resp = client()?
        .get(format!("{HF}/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=40", encode(q)))
        .send()
        .map_err(|_| "Hugging Face could not be reached. Check the connection.".to_owned())?;
    if !resp.status().is_success() {
        return Err(format!("Hugging Face answered {}.", resp.status().as_u16()));
    }
    let list: Vec<ApiModel> = resp.json().map_err(|_| "Hugging Face sent an unexpected answer.".to_owned())?;
    Ok(list
        .into_iter()
        .map(|m| {
            let license = m.tags.iter().find_map(|t| t.strip_prefix("license:").map(str::to_owned));
            HubRepo {
                author: m.id.split('/').next().unwrap_or_default().to_owned(),
                id: m.id,
                downloads: m.downloads,
                likes: m.likes,
                updated: m.last_modified,
                license,
                tags: m.tags.into_iter().filter(|t| !t.contains(':')).take(12).collect(),
            }
        })
        .collect())
}

/// One downloadable GGUF file (split models are listed once, with the total size).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HubFile {
    pub path: String,
    pub size: u64,
    pub parts: Vec<String>,
    pub quant: String,
    /// Already in Cerebro's model folder.
    pub downloaded: bool,
}

/// The architecture numbers and real size of one GGUF file, read from its header (hub_file_meta).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HubFileMeta {
    pub size: u64,
    pub architecture: String,
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub embedding: u64,
    pub experts: Option<u64>,
    pub active_experts: Option<u64>,
    pub context_max: u64,
    pub quant: String,
    /// Layers that keep a KV cache: all of them, or one in `full_attention_interval` for the
    /// Qwen3.5/3.8 hybrids (the others are Gated DeltaNet with a small fixed state).
    pub attention_layers: u64,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    size: u64,
}

fn valid_repo(repo: &str) -> bool {
    let mut it = repo.split('/');
    let ok = |s: &str| !s.is_empty() && s.len() <= 96 && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) && s != "." && s != "..";
    matches!((it.next(), it.next(), it.next()), (Some(a), Some(b), None) if ok(a) && ok(b))
}

fn valid_file(path: &str) -> bool {
    path.len() <= 300 && path.ends_with(".gguf") && !path.contains("..") && !path.starts_with('/') && !path.contains('\\') && path.split('/').all(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_.+".contains(c)))
}

/// The quant label in a file name, e.g. "UD-Q4_K_XL" or "Q8_0".
fn quant_of(name: &str) -> String {
    let stem = name.rsplit('/').next().unwrap_or(name).trim_end_matches(".gguf");
    let stem = stem.split("-000").next().unwrap_or(stem);
    let up = stem.to_uppercase();
    for marker in ["UD-", "IQ", "Q", "BF16", "F16", "F32", "MXFP4"] {
        if let Some(i) = up.rfind(marker) {
            let cand = &stem[i..];
            if cand.len() <= 16 && (marker != "Q" || cand[1..].starts_with(|c: char| c.is_ascii_digit())) {
                return cand.to_owned();
            }
        }
    }
    stem.to_owned()
}

pub fn models_dir() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join("Cerebro").join("models")
}

fn local_path(repo: &str, file: &str) -> PathBuf {
    let mut p = models_dir().join(repo.replace('/', "__"));
    for seg in file.split('/') {
        p.push(seg);
    }
    p
}

pub fn files(repo: &str) -> Result<Vec<HubFile>, String> {
    if !valid_repo(repo) {
        return Err("That is not a Hugging Face repository name.".into());
    }
    let resp = client()?
        .get(format!("{HF}/api/models/{repo}/tree/main?recursive=true"))
        .send()
        .map_err(|_| "Hugging Face could not be reached. Check the connection.".to_owned())?;
    if !resp.status().is_success() {
        return Err(format!("Hugging Face answered {} for that repository.", resp.status().as_u16()));
    }
    let tree: Vec<TreeEntry> = resp.json().map_err(|_| "Hugging Face sent an unexpected answer.".to_owned())?;
    let mut out: Vec<HubFile> = Vec::new();
    let mut by_key: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for e in tree.iter().filter(|e| e.kind == "file" && e.path.ends_with(".gguf") && !e.path.to_lowercase().contains("mmproj")) {
        let size = e.lfs.as_ref().map_or(e.size, |l| l.size);
        // Parts of a split model ("-00001-of-00003") are listed once, under the first part.
        let key = match (e.path.rfind("-000"), e.path.find("-of-")) {
            (Some(i), Some(_)) => e.path[..i].to_owned(),
            _ => e.path.clone(),
        };
        if let Some(&i) = by_key.get(&key) {
            let f = &mut out[i];
            f.size += size;
            f.parts.push(e.path.clone());
            f.parts.sort();
            f.path = f.parts[0].clone();
            continue;
        }
        by_key.insert(key, out.len());
        out.push(HubFile { path: e.path.clone(), size, parts: vec![e.path.clone()], quant: quant_of(&e.path), downloaded: false });
    }
    for f in &mut out {
        f.downloaded = f.parts.iter().all(|p| local_path(repo, p).exists());
    }
    out.sort_by_key(|f| f.size);
    Ok(out)
}

/// The architecture numbers and real size of one GGUF file. The header is opened directly when the
/// file is already in Cerebro's model folder, otherwise the first [`MAX_HEADER`] bytes are fetched
/// over HTTPS. Metadata only: the weights are never read.
pub fn file_meta(repo: &str, file: &str) -> Result<HubFileMeta, String> {
    if !valid_repo(repo) || !valid_file(file) {
        return Err("That is not a model file.".into());
    }
    let local = local_path(repo, file);
    if local.exists() {
        let size = std::fs::metadata(&local).map_err(|_| "That file could not be read.".to_owned())?.len();
        let g = cerebro_engine::gguf::Gguf::open(&local).map_err(|_| "That file could not be read as GGUF.".to_owned())?;
        let arch = g.str("general.architecture").unwrap_or("unknown");
        return Ok(build_meta(arch, |k| g.u64(k).ok(), size, quant_of(file)));
    }
    let resp = client()?
        .get(format!("{HF}/{repo}/resolve/main/{file}"))
        .header("Range", format!("bytes=0-{}", MAX_HEADER - 1))
        .send()
        .map_err(|_| "Hugging Face could not be reached. Check the connection.".to_owned())?;
    if !resp.status().is_success() {
        return Err(format!("Hugging Face answered {} for that file.", resp.status().as_u16()));
    }
    let size = content_total(resp.headers()).unwrap_or(0);
    let mut buf = Vec::new();
    resp.take(MAX_HEADER).read_to_end(&mut buf).map_err(|_| "The file header could not be read.".to_owned())?;
    let (arch, nums) = header_metadata(&buf)?;
    Ok(build_meta(&arch, |k| nums.get(k).copied(), size, quant_of(file)))
}

/// Total size from a `Content-Range: bytes 0-8388607/1234` header.
fn content_total(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers.get(reqwest::header::CONTENT_RANGE)?.to_str().ok()?.rsplit('/').next()?.trim().parse().ok()
}

/// Assembles the estimate's inputs from a key/value lookup, using the same GGUF keys the engine's
/// own reader does (engine.rs `describe`): `{arch}.block_count`, the attention head counts and so on.
fn build_meta(arch: &str, num: impl Fn(&str) -> Option<u64>, size: u64, quant: String) -> HubFileMeta {
    let key = |s: &str| format!("{arch}.{s}");
    let n_head = num(&key("attention.head_count")).unwrap_or(0);
    let dim = num(&key("embedding_length")).unwrap_or(0);
    let experts = num(&key("expert_count")).filter(|&e| e > 0);
    HubFileMeta {
        size,
        architecture: arch.to_owned(),
        layers: num(&key("block_count")).unwrap_or(0),
        kv_heads: num(&key("attention.head_count_kv")).unwrap_or(n_head),
        head_dim: num(&key("attention.key_length")).unwrap_or(if n_head > 0 { dim / n_head } else { 0 }),
        embedding: dim,
        experts,
        active_experts: num(&key("expert_used_count")).filter(|_| experts.is_some()),
        context_max: num(&key("context_length")).unwrap_or(0),
        quant,
        attention_layers: {
            let layers = num(&key("block_count")).unwrap_or(0);
            match num(&key("full_attention_interval")) {
                Some(i) if i > 1 => layers / i,
                _ => layers,
            }
        },
    }
}

fn header_too_big() -> String {
    "That model's header is larger than 8 MiB.".to_owned()
}

/// Reads a GGUF buffer's metadata section only (the tensor table is not parsed), so a truncated
/// header is enough: the architecture keys sit at the front, and arrays (vocabulary, token lists)
/// are stepped over without materialising them.
fn header_metadata(buf: &[u8]) -> Result<(String, HashMap<String, u64>), String> {
    let mut r = HeaderReader { buf, pos: 0 };
    if r.u32()? != 0x4655_4747 {
        return Err("That file is not a GGUF model.".to_owned());
    }
    let _version = r.u32()?;
    let _n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_kv > 1 << 20 {
        return Err(header_too_big());
    }
    let mut arch = String::new();
    let mut nums = HashMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        match ty {
            8 => {
                let s = r.string()?;
                if key == "general.architecture" {
                    arch = s;
                }
            }
            9 => r.skip_array()?,
            _ => {
                if let Some(v) = r.scalar(ty)? {
                    nums.insert(key, v);
                }
            }
        }
    }
    Ok((if arch.is_empty() { "unknown".to_owned() } else { arch }, nums))
}

struct HeaderReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> HeaderReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or_else(header_too_big)?;
        let s = self.buf.get(self.pos..end).ok_or_else(header_too_big)?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            return Err(header_too_big());
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    /// One scalar value; strings and floats return None (consumed but not needed here).
    fn scalar(&mut self, ty: u32) -> Result<Option<u64>, String> {
        Ok(match ty {
            0 | 7 => Some(self.u8()? as u64),
            1 => Some(self.u8()? as i8 as u64),
            2 | 3 => Some(self.u16()? as u64),
            4 | 5 => Some(self.u32()? as u64),
            6 => {
                self.u32()?;
                None
            }
            8 => {
                self.string()?;
                None
            }
            10 | 11 => Some(self.u64()?),
            12 => {
                self.u64()?;
                None
            }
            _ => return Err(header_too_big()),
        })
    }
    fn skip_array(&mut self) -> Result<(), String> {
        let et = self.u32()?;
        let n = self.u64()?;
        if n > 1 << 26 || et == 9 {
            return Err(header_too_big());
        }
        for _ in 0..n {
            if et == 8 {
                self.string()?;
            } else {
                self.scalar(et)?;
            }
        }
        Ok(())
    }
}

/// Download progress, streamed to the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DownloadEvent {
    Progress {
        done: u64,
        total: u64,
        #[serde(rename = "bytesPerSec")]
        bytes_per_sec: f64,
    },
    Done {
        path: String,
    },
    Error {
        message: String,
    },
}

/// Running downloads, so one can be cancelled.
#[derive(Default)]
pub struct Downloads {
    pub cancel: Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>,
}

/// Downloads every part of `file` into Cerebro's model folder (`.part` until complete, so a broken
/// download never shows up in the library). Resumes a partial file.
pub fn download(repo: &str, file: &str, parts: &[String], total: u64, events: &Channel<DownloadEvent>, cancel: &AtomicBool) -> Result<String, String> {
    if !valid_repo(repo) || !valid_file(file) || parts.is_empty() || !parts.iter().all(|p| valid_file(p)) {
        return Err("That is not a downloadable model file.".into());
    }
    let c = client()?;
    let t0 = Instant::now();
    let mut done_all: u64 = 0;
    let mut last = Instant::now();
    for part in parts {
        let dest = local_path(repo, part);
        if dest.exists() {
            done_all += std::fs::metadata(&dest).map_or(0, |m| m.len());
            continue;
        }
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir).map_err(|_| "Could not create Cerebro's model folder.".to_owned())?;
        }
        let tmp = dest.with_extension("gguf.part");
        let have = std::fs::metadata(&tmp).map_or(0, |m| m.len());
        let mut req = c.get(format!("{HF}/{repo}/resolve/main/{part}"));
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut resp = req.send().map_err(|_| "Hugging Face could not be reached. Check the connection.".to_owned())?;
        let resumed = resp.status().as_u16() == 206;
        if !resp.status().is_success() {
            return Err(format!("The download failed: Hugging Face answered {}.", resp.status().as_u16()));
        }
        let mut out = std::fs::OpenOptions::new().create(true).append(resumed).write(true).truncate(!resumed).open(&tmp).map_err(|_| "Could not write the model file.".to_owned())?;
        if resumed {
            done_all += have;
        }
        let mut buf = vec![0u8; 1 << 20];
        loop {
            if cancel.load(Ordering::SeqCst) {
                return Err("Download stopped. It resumes where it left off.".into());
            }
            let n = resp.read(&mut buf).map_err(|_| "The connection dropped. Start the download again to resume.".to_owned())?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(|_| "Could not write the model file (is the disk full?).".to_owned())?;
            done_all += n as u64;
            if last.elapsed().as_millis() >= 250 {
                last = Instant::now();
                let _ = events.send(DownloadEvent::Progress { done: done_all, total, bytes_per_sec: done_all as f64 / t0.elapsed().as_secs_f64().max(1e-3) });
            }
        }
        out.flush().ok();
        drop(out);
        std::fs::rename(&tmp, &dest).map_err(|_| "Could not finish the model file.".to_owned())?;
    }
    let first = local_path(repo, &parts[0]).to_string_lossy().into_owned();
    let _ = events.send(DownloadEvent::Progress { done: total.max(done_all), total, bytes_per_sec: done_all as f64 / t0.elapsed().as_secs_f64().max(1e-3) });
    let _ = events.send(DownloadEvent::Done { path: first.clone() });
    Ok(first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_checked() {
        assert!(valid_repo("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"));
        assert!(!valid_repo("../etc"));
        assert!(!valid_repo("a/b/c"));
        assert!(valid_file("UD-Q4_K_XL/model-00001-of-00002.gguf"));
        assert!(!valid_file("../x.gguf"));
        assert!(!valid_file("a\\b.gguf"));
        assert_eq!(quant_of("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL");
        assert_eq!(quant_of("gemma-4-e4b-it-Q8_0.gguf"), "Q8_0");
        assert_eq!(quant_of("x-IQ2_XXS-00001-of-00003.gguf"), "IQ2_XXS");
    }

    fn kv_str(key: &str, val: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&(val.len() as u64).to_le_bytes());
        b.extend_from_slice(val.as_bytes());
        b
    }

    fn kv_u32(key: &str, val: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
        b
    }

    /// A GGUF header with a string array in it, as every real tokenizer has.
    fn header() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&11u64.to_le_bytes());
        b.extend(kv_str("general.architecture", "qwen3moe"));
        b.extend(kv_u32("qwen3moe.block_count", 48));
        b.extend(kv_u32("qwen3moe.attention.head_count", 32));
        b.extend(kv_u32("qwen3moe.attention.head_count_kv", 4));
        b.extend(kv_u32("qwen3moe.attention.key_length", 128));
        b.extend(kv_u32("qwen3moe.embedding_length", 2048));
        b.extend(kv_u32("qwen3moe.expert_count", 128));
        b.extend(kv_u32("qwen3moe.expert_used_count", 8));
        b.extend(kv_u32("qwen3moe.context_length", 32768));
        b.extend(kv_u32("general.file_type", 15));
        // An array of two strings, skipped whole.
        let key = b"tokenizer.ggml.tokens";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        for t in ["<s>", "</s>"] {
            b.extend_from_slice(&(t.len() as u64).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        }
        b
    }

    #[test]
    fn header_metadata_reads_the_architecture_numbers() {
        let (arch, nums) = header_metadata(&header()).unwrap();
        assert_eq!(arch, "qwen3moe");
        let meta = build_meta(&arch, |k| nums.get(k).copied(), 18_600_000_000, quant_of("x-Q4_K_M.gguf"));
        assert_eq!(meta.size, 18_600_000_000);
        assert_eq!(meta.layers, 48);
        assert_eq!(meta.kv_heads, 4);
        assert_eq!(meta.head_dim, 128);
        assert_eq!(meta.embedding, 2048);
        assert_eq!(meta.experts, Some(128));
        assert_eq!(meta.active_experts, Some(8));
        assert_eq!(meta.context_max, 32768);
        assert_eq!(meta.quant, "Q4_K_M");
    }

    #[test]
    fn head_dim_falls_back_to_embedding_over_heads() {
        let meta = build_meta("llama", |k| if k == "llama.embedding_length" { Some(4096) } else if k == "llama.attention.head_count" { Some(32) } else { None }, 0, "Q8_0".into());
        assert_eq!(meta.head_dim, 128);
        assert_eq!(meta.kv_heads, 32);
    }

    #[test]
    fn a_header_that_is_not_gguf_or_is_cut_short_is_refused() {
        assert!(header_metadata(b"not a model").is_err());
        let cut = &header()[..20];
        assert!(header_metadata(cut).is_err());
    }
}
