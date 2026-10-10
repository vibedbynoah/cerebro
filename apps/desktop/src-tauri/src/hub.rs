//! Model Hub: live Hugging Face search, the GGUF files of a repository with their sizes, and
//! downloads into Cerebro's model folder with progress. The only network feature of the app; each
//! request goes to huggingface.co and nowhere else.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;

const HF: &str = "https://huggingface.co";

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
        .map(|m| HubRepo {
            author: m.id.split('/').next().unwrap_or_default().to_owned(),
            id: m.id,
            downloads: m.downloads,
            likes: m.likes,
            updated: m.last_modified,
            tags: m.tags.into_iter().filter(|t| !t.contains(':')).take(12).collect(),
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
}
