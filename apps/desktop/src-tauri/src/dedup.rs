//! Disk space on the Models screen: find model files stored more than once across Cerebro's,
//! another app's and other folders, then replace the extra copies on one drive with NTFS
//! hard links to the copy the user keeps — and undo that.
//!
//! The rules that make it safe:
//! - Nothing is deleted before its replacement exists. A copy is replaced by the rename dance in
//!   [`replace_with_link`]: the old file is renamed aside, the new one renamed into place, and
//!   only then is the old one deleted; any failure puts the old one back where it was.
//! - Every copy's hash is checked from disk again before anything is touched, and the whole group
//!   is refused when a file changed since the scan the user saw.
//! - Hard links only work within one volume, so copies on another drive are left alone and
//!   counted in [`DedupeResult::skipped`].
//! - Hashing streams each file in 4 MB reads (a 100 GiB model never sits in memory whole) and
//!   remembers `(path, size, modified time)` in memory, so a rescan of an unchanged library
//!   reads nothing.
//!
//! [`Disk`] remembers the last scan: the commands that follow it (`disk_deduplicate`,
//! `disk_unlink`) work on exactly the copies the user was shown, and the group ids they receive
//! are the files' SHA-256, so a stale id simply does not match.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::engine::LibraryModel;

const GIB: f64 = (1u64 << 30) as f64;
/// Hashing reads this many bytes at a time. Model files are many GiB; none is ever read whole.
const READ_CHUNK: usize = 4 << 20;

// ---- Types the UI sees (mirror ui/src/service/types.ts) -----------------------------------------

/// One copy of a model file.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DuplicateCopy {
    /// Full path, display only.
    pub path: String,
    /// The folder it was found in ("cerebro", "lmstudio", "ollama", "other").
    pub source: String,
    /// Drive as Windows names it ("C:"). Hard links only work within one drive.
    pub volume: String,
}

/// One file stored more than once.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DuplicateGroup {
    /// The file's SHA-256 as lowercase hex: the group's identity across scans.
    pub id: String,
    /// File name, for display.
    pub file: String,
    /// One copy's size, GiB.
    pub size_gi_b: f64,
    /// The same SHA-256 as `id`, named as the UI names it.
    pub sha256: String,
    /// Every copy found, sorted by path.
    pub copies: Vec<DuplicateCopy>,
    /// `"duplicate"` (separate files) or `"linked"` (all copies are one file on disk).
    pub state: &'static str,
    /// Linked only: the path the others point at.
    pub kept: Option<String>,
}

/// What `disk_scan` reports: the bytes the model folders store and the groups stored twice.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiskReport {
    /// Always `"measured"`: every number here was read from disk, none estimated.
    pub provenance: &'static str,
    /// Bytes stored under the model folders, counting a hard-linked file once.
    pub total_gi_b: f64,
    /// Files stored more than once, by name.
    pub groups: Vec<DuplicateGroup>,
}

/// What `disk_deduplicate` changed.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DedupeResult {
    /// The group afterwards (state `"linked"`).
    pub group: DuplicateGroup,
    /// Bytes the extra copies held, GiB.
    pub reclaimed_gi_b: f64,
    /// Copies left alone because they sit on another drive.
    pub skipped: u32,
}

// ---- The remembered scan and the hash memo -------------------------------------------------------

/// `(path, size, modified time)` — what makes a cached hash still true. Content that changed
/// changed the write time (or the size), so the entry is not used.
type HashKey = (PathBuf, u64, u64);

/// Memo of file hashes, so a second scan of an unchanged library reads nothing.
#[derive(Default)]
struct HashCache {
    entries: HashMap<HashKey, String>,
    /// Files actually read, ever. The cache-reuse test proves a rescan does not grow this.
    computed: u64,
}

impl HashCache {
    /// SHA-256 of `path` as lowercase hex, reused while the file's size and write time are
    /// unchanged.
    fn sha256(&mut self, path: &Path) -> std::io::Result<String> {
        let meta = std::fs::metadata(path)?;
        let modified = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
        let key = (path.to_path_buf(), meta.len(), modified);
        if let Some(known) = self.entries.get(&key) {
            return Ok(known.clone());
        }
        let hash = sha256_file(path)?;
        self.entries.insert(key, hash.clone());
        self.computed += 1;
        Ok(hash)
    }
}

/// The disk tools' state: the last scan's groups (so the calls that follow re-check exactly what
/// the user saw) and the hash memo that makes rescanning cheap.
///
/// The app manages one instance; unit tests build their own, which is why nothing here is global.
#[derive(Default)]
pub struct Disk {
    cache: Mutex<HashCache>,
    groups: Mutex<Vec<DuplicateGroup>>,
}

impl Disk {
    /// Groups the library's files that are stored more than once: by exact byte size first (one
    /// cheap stat each), then by SHA-256. Copies that are hard links of each other count as one
    /// group in state `"linked"`. Remembers the result for [`Disk::deduplicate`] and
    /// [`Disk::unlink`]. Files that cannot be read are left out rather than guessed about.
    pub fn scan(&self, library: &[LibraryModel]) -> DiskReport {
        // Hash with the memo locked: two scans at once would read the same files twice.
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let mut by_size: HashMap<u64, Vec<&LibraryModel>> = HashMap::new();
        let mut all_bytes = 0u64;
        for m in library {
            if let Ok(meta) = std::fs::metadata(&m.path) {
                all_bytes += meta.len();
                // An empty file has no bytes to reclaim; two of them are not worth a group.
                if meta.len() > 0 {
                    by_size.entry(meta.len()).or_default().push(m);
                }
            }
        }

        let mut shared = 0u64; // bytes a hard-linked group lists many times but stores once
        let mut groups: Vec<DuplicateGroup> = Vec::new();
        for (size, mut files) in by_size {
            if files.len() < 2 {
                continue;
            }
            let mut by_hash: HashMap<String, Vec<&LibraryModel>> = HashMap::new();
            for m in files.drain(..) {
                if let Ok(hash) = cache.sha256(Path::new(&m.path)) {
                    by_hash.entry(hash).or_default().push(m);
                }
            }
            for (hash, mut copies) in by_hash {
                if copies.len() < 2 {
                    continue;
                }
                copies.sort_by(|a, b| a.path.cmp(&b.path));
                let paths: Vec<&Path> = copies.iter().map(|m| Path::new(m.path.as_str())).collect();
                let linked = same_file(&paths);
                if linked {
                    shared += size * (copies.len() as u64 - 1);
                }
                let group_copies: Vec<DuplicateCopy> = copies
                    .iter()
                    .map(|m| DuplicateCopy {
                        path: m.path.clone(),
                        source: m.source.to_owned(),
                        volume: volume_of(Path::new(&m.path)),
                    })
                    .collect();
                let kept = if linked { group_copies.first().map(|c| c.path.clone()) } else { None };
                let file = copies[0].file.clone();
                groups.push(DuplicateGroup {
                    id: hash.clone(),
                    file,
                    size_gi_b: size as f64 / GIB,
                    sha256: hash,
                    copies: group_copies,
                    state: if linked { "linked" } else { "duplicate" },
                    kept,
                });
            }
        }
        groups.sort_by(|a, b| a.file.to_lowercase().cmp(&b.file.to_lowercase()).then_with(|| a.id.cmp(&b.id)));

        let report = DiskReport { provenance: "measured", total_gi_b: all_bytes.saturating_sub(shared) as f64 / GIB, groups };
        // Remember it for the calls that follow the user's look at this report.
        *self.groups.lock().unwrap_or_else(|e| e.into_inner()) = report.groups.clone();
        report
    }

    /// Keeps the copy at `keep_path` and replaces the others on its drive with hard links to it,
    /// after checking every copy's hash again from disk. Copies on other drives are skipped and
    /// counted. Refusing leaves every file exactly as it was.
    ///
    /// # Errors
    /// A plain sentence when the group is not the last scan's, a copy changed or cannot be read,
    /// the kept path is not a copy of the group, or no copy shares its drive.
    pub fn deduplicate(&self, group_id: &str, keep_path: &str) -> Result<DedupeResult, String> {
        let group = self.remembered(group_id)?;
        if group.state == "linked" {
            return Err("Those copies are already hard-linked.".to_owned());
        }
        if !group.copies.iter().any(|c| c.path == keep_path) {
            return Err("Keep one of the group's own copies.".to_owned());
        }
        // These files are about to be replaced, so their hashes are read from disk once more,
        // uncached: one changed byte (a re-download, an edit) must stop the whole thing.
        for copy in &group.copies {
            let hash = sha256_file(Path::new(&copy.path)).map_err(|_| "A copy could not be read; nothing was changed.".to_owned())?;
            if hash != group.sha256 {
                return Err("A copy changed since the scan; nothing was changed.".to_owned());
            }
        }

        let keep = Path::new(keep_path);
        let target = linkable(&group.copies, keep_path);
        if target.is_empty() {
            return Err("No copy shares a drive with the kept one.".to_owned());
        }
        let skipped = (group.copies.len() - 1 - target.len()) as u32;
        let size = std::fs::metadata(keep).map(|m| m.len()).map_err(|_| "The kept copy could not be read.".to_owned())?;
        for copy in &target {
            replace_with_link(keep, Path::new(&copy.path))?;
        }
        let linked_copies = target.len();
        drop(target);

        let linked = DuplicateGroup { state: "linked", kept: Some(keep_path.to_owned()), ..group };
        self.remember(&linked);
        Ok(DedupeResult { group: linked, reclaimed_gi_b: size as f64 * linked_copies as f64 / GIB, skipped })
    }

    /// Undoes a deduplicate: copies the bytes from the kept copy back beside every other copy, so
    /// each path is its own file again. Refuses before touching anything when a drive lacks the
    /// room the restored copies need.
    ///
    /// # Errors
    /// A plain sentence when the group is not the last scan's or is not linked, a copy changed
    /// since then, or a drive lacks the space.
    pub fn unlink(&self, group_id: &str) -> Result<DuplicateGroup, String> {
        let group = self.remembered(group_id)?;
        let Some(kept) = group.kept.clone().filter(|_| group.state == "linked") else {
            return Err("Those copies are not hard-linked.".to_owned());
        };
        let keep = Path::new(&kept);
        let size = std::fs::metadata(keep).map_err(|_| "The kept copy could not be read.".to_owned())?.len();
        let copies: Vec<&DuplicateCopy> = group.copies.iter().filter(|c| c.path != kept).collect();

        // They must still be links of the kept file: a copy rewritten in between means the state
        // the user saw changed, and replacing it would lose whatever is there now.
        if let Ok(Some(keep_id)) = cerebro_winfs::identify(keep) {
            for copy in &copies {
                if !matches!(cerebro_winfs::identify(Path::new(&copy.path)), Ok(Some(id)) if id == keep_id) {
                    return Err("A copy changed since the scan; nothing was changed.".to_owned());
                }
            }
        }
        // Room first: each restored copy needs its own size again, and copies on one drive must
        // all fit together. A volume that cannot be asked is left to fail at the write instead.
        let mut needed: HashMap<String, (PathBuf, u64)> = HashMap::new();
        for copy in &copies {
            let entry = needed.entry(copy.volume.clone()).or_insert_with(|| (PathBuf::from(&copy.path), 0));
            entry.1 += size;
        }
        for (path, need) in needed.values() {
            if let Ok(free) = cerebro_winfs::free_bytes(path) {
                if free < *need {
                    return Err("There is not enough free space to restore a copy; nothing was changed.".to_owned());
                }
            }
        }

        for copy in &copies {
            restore_copy(keep, Path::new(&copy.path))?;
        }
        drop(copies);
        let restored = DuplicateGroup { state: "duplicate", kept: None, ..group };
        self.remember(&restored);
        Ok(restored)
    }

    /// The last scan's group with this id, or a plain refusal when there is none: the id is the
    /// file's SHA-256, so a group the user is no longer looking at cannot be changed by accident.
    fn remembered(&self, group_id: &str) -> Result<DuplicateGroup, String> {
        if group_id.len() != 64 || !group_id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("That is not a duplicate group.".to_owned());
        }
        self.groups
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|g| g.id == group_id)
            .cloned()
            .ok_or_else(|| "That file is not stored more than once any more. Scan the model folders again.".to_owned())
    }

    /// Replaces the remembered copy of `group` (the state the next call must work from).
    fn remember(&self, group: &DuplicateGroup) {
        let mut groups = self.groups.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = groups.iter_mut().find(|g| g.id == group.id) {
            *slot = group.clone();
        }
    }
}

/// Opens File Explorer at the file, with the file selected. False when Explorer could not be
/// started (not the desktop app, or Explorer missing): the caller shows the path instead.
#[cfg(windows)]
pub fn reveal(path: &Path) -> bool {
    let select = format!("/select,\"{}\"", path.display());
    std::process::Command::new("explorer.exe").arg(select).spawn().is_ok()
}

/// Non-Windows builds have no File Explorer.
#[cfg(not(windows))]
pub fn reveal(_path: &Path) -> bool {
    false
}

// ---- Grouping facts -------------------------------------------------------------------------------

/// The copies a deduplicate keeping `keep_path` would replace: the group's other copies on the
/// same drive as it. The rest (another drive) cannot be hard links and are skipped.
fn linkable<'a>(copies: &'a [DuplicateCopy], keep_path: &str) -> Vec<&'a DuplicateCopy> {
    let Some(keep) = copies.iter().find(|c| c.path == keep_path) else { return Vec::new() };
    copies.iter().filter(|c| c.path != keep_path && same_volume(&c.volume, &keep.volume)).collect()
}

/// Whether every path is the same file on disk (all hard links of each other). False when the
/// platform cannot tell, the safe direction: the group is then treated as separate files.
fn same_file(paths: &[&Path]) -> bool {
    let Some((first, rest)) = paths.split_first() else { return false };
    let Ok(Some(first)) = cerebro_winfs::identify(first) else { return false };
    rest.iter().all(|p| matches!(cerebro_winfs::identify(p), Ok(Some(id)) if id == first))
}

/// The drive a path lives on, as Windows names it ("C:"), for the volume column and for the
/// same-drive test. A path with no drive prefix (relative, or another platform) has no label.
pub(crate) fn volume_of(path: &Path) -> String {
    match path.components().next() {
        Some(std::path::Component::Prefix(prefix)) => match prefix.kind() {
            std::path::Prefix::Disk(d) | std::path::Prefix::VerbatimDisk(d) => format!("{}:", (d as char).to_ascii_uppercase()),
            std::path::Prefix::UNC(server, share) | std::path::Prefix::VerbatimUNC(server, share) => format!("\\\\{}\\{}", server.to_string_lossy(), share.to_string_lossy()),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// Whether two volume labels are the same drive, so a hard link can stand in for a copy. Labels
/// compare case-insensitively ("c:" is "C:"); an unknown label matches nothing.
pub(crate) fn same_volume(a: &str, b: &str) -> bool {
    !a.is_empty() && a.eq_ignore_ascii_case(b)
}

// ---- Replacing files without ever losing one -------------------------------------------------------

/// Replaces `copy` with a hard link to `keep`.
///
/// Windows cannot rename a file over an existing one in one step, so: write the link under a
/// working name beside the copy (same folder, so the same drive), rename the copy aside, rename
/// the link into place, delete the copy-aside. Any failure renames the old file back; the bytes
/// are never lost and the path is never left empty.
fn replace_with_link(keep: &Path, copy: &Path) -> Result<(), String> {
    let link = sidecar(copy, "link");
    let old = sidecar(copy, "old");
    std::fs::hard_link(keep, &link).map_err(|_| "A hard link could not be made; nothing was changed.".to_owned())?;
    if std::fs::rename(copy, &old).is_err() {
        let _ = std::fs::remove_file(&link);
        return Err("A copy could not be moved aside; nothing was changed.".to_owned());
    }
    if std::fs::rename(&link, copy).is_err() {
        let _ = std::fs::rename(&old, copy);
        let _ = std::fs::remove_file(&link);
        return Err("A copy could not be linked; the old file is back in place.".to_owned());
    }
    if std::fs::remove_file(&old).is_err() {
        return Err("A copy was linked but its old file could not be removed; scan again.".to_owned());
    }
    Ok(())
}

/// Replaces a hard link with its own copy of the bytes: write the bytes to a working name beside
/// it, rename the link aside, rename the copy into place, delete the link-aside. The same
/// old-file-is-kept-until-the-new-one-is-in-place dance as [`replace_with_link`].
fn restore_copy(keep: &Path, copy: &Path) -> Result<(), String> {
    let fresh = sidecar(copy, "copy");
    let old = sidecar(copy, "old");
    std::fs::copy(keep, &fresh).map_err(|_| "A copy could not be written; nothing was changed.".to_owned())?;
    if std::fs::rename(copy, &old).is_err() {
        let _ = std::fs::remove_file(&fresh);
        return Err("A link could not be moved aside; nothing was changed.".to_owned());
    }
    if std::fs::rename(&fresh, copy).is_err() {
        let _ = std::fs::rename(&old, copy);
        let _ = std::fs::remove_file(&fresh);
        return Err("A copy could not be restored; the link is back in place.".to_owned());
    }
    if std::fs::remove_file(&old).is_err() {
        return Err("A copy was restored but its old link could not be removed; scan again.".to_owned());
    }
    Ok(())
}

/// A working name beside `copy`: a leading dot, the copy's own name, a fresh nonce and the step.
/// Unique per call, so it can never collide with a real file, and never anything a previous run
/// left behind (nothing here deletes a file it did not just create).
fn sidecar(copy: &Path, tag: &str) -> PathBuf {
    let name = copy.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".to_owned());
    let dir = copy.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    dir.join(format!(".{name}.cerebro-{:016x}-{tag}", nonce()))
}

/// A fresh 64-bit tag: the OS generator when it answers, else the clock, the process id and a
/// counter. Uniqueness within one folder is all a working name needs.
fn nonce() -> u64 {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if let Ok(v) = getrandom::u64() {
        return v ^ (n << 48);
    }
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    t ^ u64::from(std::process::id()) << 32 ^ n
}

/// SHA-256 of one file as lowercase hex, streamed in 4 MB reads so even a 100 GiB model never
/// sits in memory whole. Always reads: callers that must not trust a memo use this directly.
/// SHA-256 of a whole file, streamed (the Tune screen's integrity check uses it too).
pub(crate) fn sha256_of(path: &Path) -> std::io::Result<String> {
    sha256_file(path)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; READ_CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Lowercase hex, the 64-character shape the UI and the isolation hook expect.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelArch;

    /// A fresh folder under the system temp directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("cerebro-dedup-{:016x}", nonce()));
            std::fs::create_dir_all(&dir).expect("temp dir");
            TempDir(dir)
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A library entry for a file a test wrote. The scan reads `path`, `file` and `source`; the
    /// rest of the shape is filler.
    fn model(path: &Path, source: &'static str) -> LibraryModel {
        LibraryModel {
            id: "m0".to_owned(),
            name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            quant: "Q4_K_M".to_owned(),
            params: String::new(),
            architecture: "dense",
            size_gi_b: 0.0,
            context_max: 0,
            tune: "untuned",
            provenance: "measured",
            kind: "text",
            modalities: vec!["text"],
            family: String::new(),
            file: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            path: path.to_string_lossy().into_owned(),
            source,
            sha256: String::new(),
            arch: ModelArch { layers: 1, kv_heads: 1, head_dim: 1, embedding: 1, experts: None, active_experts: None },
            tuned_params: None,
            tuned_at: None,
            tune_result: None,
            projector: None,
            supported: true,
            arch_name: "qwen3".to_owned(),
        }
    }

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write");
    }

    fn sha_of(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex(&hasher.finalize())
    }

    /// A made-up copy for the pure grouping tests; the path need not exist.
    fn copy(path: &str, volume: &str) -> DuplicateCopy {
        DuplicateCopy { path: path.to_owned(), source: "other".to_owned(), volume: volume.to_owned() }
    }

    #[test]
    fn groups_copies_by_size_then_hash() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        let unique = dir.file("unique.gguf");
        let bytes = b"one model, two copies";
        write(&a, bytes);
        write(&b, bytes);
        write(&unique, b"a different size entirely");
        let disk = Disk::default();

        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio"), model(&unique, "other")]);
        assert_eq!(report.provenance, "measured");
        assert_eq!(report.groups.len(), 1);
        let g = &report.groups[0];
        assert_eq!(g.id, sha_of(bytes));
        assert_eq!(g.sha256, g.id);
        assert_eq!(g.state, "duplicate");
        assert_eq!(g.kept, None);
        assert_eq!(g.copies.len(), 2);
        assert_eq!(g.copies[0].path, a.to_string_lossy());
        assert_eq!(g.copies[1].path, b.to_string_lossy());
        assert_eq!(g.copies[0].source, "cerebro");
        assert_eq!(g.copies[1].source, "lmstudio");
        assert_eq!(g.copies[0].volume, volume_of(&a));
        assert!((g.size_gi_b - bytes.len() as f64 / GIB).abs() < 1e-12);
        // Both copies are separate files, so the disk really holds both, plus the odd one.
        let all = (bytes.len() * 2 + b"a different size entirely".len()) as f64 / GIB;
        assert!((report.total_gi_b - all).abs() < 1e-12);
    }

    #[test]
    fn same_size_different_content_is_not_a_group() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        write(&a, b"XXXXXXXXXXXXXXXX");
        write(&b, b"YYYYYYYYYYYYYYYY");
        let disk = Disk::default();
        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        assert!(report.groups.is_empty(), "same byte count, different bytes: no group");
        assert!((report.total_gi_b - 32.0 / GIB).abs() < 1e-12);
    }

    #[test]
    fn hard_linked_copies_scan_as_one_linked_group() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        write(&a, b"the same file, linked");
        std::fs::hard_link(&a, &b).expect("hard link");
        let disk = Disk::default();

        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "ollama")]);
        assert_eq!(report.groups.len(), 1);
        let g = &report.groups[0];
        assert_eq!(g.state, "linked");
        assert_eq!(g.copies.len(), 2);
        assert_eq!(g.kept.as_deref(), Some(g.copies[0].path.as_str()));
        // One file on disk, listed under both names: counted once.
        assert!((report.total_gi_b - g.size_gi_b).abs() < 1e-12);
    }

    #[test]
    fn deduplicate_reclaims_and_both_paths_keep_the_bytes() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        let bytes = b"a model file, stored twice";
        write(&a, bytes);
        write(&b, bytes);
        let disk = Disk::default();
        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        let g = report.groups[0].clone();

        let r = disk.deduplicate(&g.id, &a.to_string_lossy()).expect("deduplicate");
        assert_eq!(r.group.state, "linked");
        assert_eq!(r.group.kept.as_deref(), Some(a.to_string_lossy().as_ref()));
        assert_eq!(r.skipped, 0);
        assert!((r.reclaimed_gi_b - bytes.len() as f64 / GIB).abs() < 1e-12);
        assert_eq!(r.group.copies.len(), 2);
        assert_eq!(std::fs::read(&a).unwrap(), bytes);
        assert_eq!(std::fs::read(&b).unwrap(), bytes);
        #[cfg(windows)]
        {
            let id_a = cerebro_winfs::identify(&a).unwrap().unwrap();
            let id_b = cerebro_winfs::identify(&b).unwrap().unwrap();
            assert_eq!(id_a, id_b, "both paths are one file now");
        }
        // A rescan shows the pair as one linked file whose kept copy can be shown.
        let again = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        assert_eq!(again.groups.len(), 1);
        assert_eq!(again.groups[0].state, "linked");
        assert!(again.groups[0].kept.is_some());
    }

    #[test]
    fn a_changed_copy_refuses_deduplicate() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        write(&a, b"original content....");
        write(&b, b"original content....");
        let disk = Disk::default();
        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        let g = report.groups[0].clone();
        // The same byte count but different bytes: the re-hash before linking must catch it.
        write(&b, b"different content...");

        let err = disk.deduplicate(&g.id, &a.to_string_lossy()).expect_err("changed copy must refuse");
        assert!(err.contains("changed"), "{err}");
        assert_eq!(std::fs::read(&a).unwrap(), b"original content....");
        assert_eq!(std::fs::read(&b).unwrap(), b"different content...");
    }

    #[test]
    fn unlink_restores_independent_copies() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        let bytes = b"restore me to two files";
        write(&a, bytes);
        write(&b, bytes);
        let disk = Disk::default();
        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        let g = disk.deduplicate(&report.groups[0].id, &a.to_string_lossy()).unwrap().group;

        let back = disk.unlink(&g.id).expect("unlink");
        assert_eq!(back.state, "duplicate");
        assert_eq!(back.kept, None);
        #[cfg(windows)]
        {
            assert_ne!(cerebro_winfs::identify(&a).unwrap(), cerebro_winfs::identify(&b).unwrap());
        }
        // Writing to one leaves the other alone: they are separate files again.
        write(&a, b"written after undo");
        assert_eq!(std::fs::read(&b).unwrap(), bytes);
        // And there is nothing left to undo.
        assert!(disk.unlink(&g.id).is_err());
    }

    #[test]
    fn volumes_decide_what_can_be_linked() {
        assert!(same_volume("C:", "c:"));
        assert!(!same_volume("C:", "D:"));
        assert!(!same_volume("", "C:"));
        assert!(!same_volume("C:", ""));
        #[cfg(windows)]
        {
            assert_eq!(volume_of(Path::new("C:\\models\\a.gguf")), "C:");
            assert_eq!(volume_of(Path::new("d:\\Models\\b.gguf")), "D:");
            assert_eq!(volume_of(Path::new("\\\\server\\share\\a.gguf")), "\\\\server\\share");
            assert_eq!(volume_of(Path::new("relative\\a.gguf")), "");
            assert!(same_volume(&volume_of(Path::new("C:\\a")), &volume_of(Path::new("c:\\b"))));
            assert!(!same_volume(&volume_of(Path::new("C:\\a")), &volume_of(Path::new("D:\\b"))));
        }
    }

    #[test]
    fn cross_drive_copies_are_skipped() {
        let copies = vec![copy("C:\\cerebro\\a.gguf", "C:"), copy("C:\\lmstudio\\a.gguf", "C:"), copy("D:\\models\\a.gguf", "D:")];
        let target = linkable(&copies, "C:\\cerebro\\a.gguf");
        assert_eq!(target.len(), 1);
        assert_eq!(target[0].path, "C:\\lmstudio\\a.gguf");
        // Kept on D:: the C: copies cannot be linked to it, so nothing can be done.
        assert!(linkable(&copies, "D:\\models\\a.gguf").is_empty());
        // A path that is not a copy of the group has nothing linkable either.
        assert!(linkable(&copies, "C:\\elsewhere.gguf").is_empty());
    }

    #[test]
    fn hash_cache_skips_unchanged_files_on_a_rescan() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        write(&a, b"cached bytes");
        write(&b, b"cached bytes");
        let disk = Disk::default();

        let first = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        assert_eq!(first.groups.len(), 1);
        let reads = disk.cache.lock().unwrap().computed;
        assert_eq!(reads, 2, "the first scan reads both files");
        disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        assert_eq!(disk.cache.lock().unwrap().computed, reads, "a second scan reads nothing new");

        // A rewritten file is read again (the write time changed) and the copies no longer match;
        // the same byte count keeps it in the size bucket that gets hashed.
        write(&b, b"CACHED BYTES");
        std::fs::File::options()
            .write(true)
            .open(&b)
            .unwrap()
            .set_modified(SystemTime::now() + std::time::Duration::from_secs(2))
            .unwrap();
        let again = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        assert_eq!(disk.cache.lock().unwrap().computed, reads + 1);
        assert!(again.groups.is_empty());
    }

    #[test]
    fn deduplicate_refuses_unknown_groups_and_foreign_paths() {
        let dir = TempDir::new();
        let a = dir.file("a.gguf");
        let b = dir.file("b.gguf");
        write(&a, b"pair");
        write(&b, b"pair");
        let disk = Disk::default();

        // No scan yet (and never, a made-up id).
        assert!(disk.deduplicate(&"f".repeat(64), &a.to_string_lossy()).is_err());
        assert!(disk.deduplicate("not-a-group", &a.to_string_lossy()).is_err());

        let report = disk.scan(&[model(&a, "cerebro"), model(&b, "lmstudio")]);
        let g = report.groups[0].clone();
        // A keep path outside the group, and an unlink before anything is linked, are refused.
        assert!(disk.deduplicate(&g.id, "C:\\elsewhere.gguf").is_err());
        assert!(disk.unlink(&g.id).is_err());
        // The refusals changed nothing.
        assert_eq!(std::fs::read(&a).unwrap(), b"pair");
        assert_eq!(std::fs::read(&b).unwrap(), b"pair");
    }
}
