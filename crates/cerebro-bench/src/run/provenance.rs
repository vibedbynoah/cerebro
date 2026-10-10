//! What a measurement is keyed to: the binary, the model, and the machine's state at the time.
//!
//! `docs/CLAIMS.md`: every measurement is keyed to the exact binary that produced it (a self-built
//! fork of the same commit measured 3-5x slower than the official release), the model, and free
//! VRAM at start (observed swinging from 431 to 7285 MiB within minutes). This module computes
//! those facts cheaply enough to do before every run:
//!
//! - The model digest is cached by (path, size, modification time, volume serial, file index), so
//!   a tuning session hashes a 20 GB model once, not once per candidate configuration, while a
//!   replacement file with the same size and a restored timestamp is still re-hashed. Reading and
//!   hashing overlap on two threads, so the one hash costs about max(read, hash) rather than their
//!   sum. The handle that was hashed is kept open, deny-write, until llama-bench has exited
//!   ([`PinnedModel`]), so the file measured is the file hashed.
//! - The backend digest reuses the hashes preflight already computed for its integrity check;
//!   only when the user turned that check off is the directory hashed here.
//! - CPU topology and the GPU inventory are probed once per [`super::BenchRunner`]; only free VRAM,
//!   GPU utilization, other llama.cpp processes and the power state are sampled per run.
//! - The selected `ggml-cpu-*` variant is read from the `load_backend` line llama-bench prints on
//!   every run; the compiler comes from the build's `--version`, queried once per backend digest.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;
use std::sync::{Mutex, PoisonError};
use std::time::SystemTime;

use cerebro_probe::{CpuTopology, FileIdentity, GpuInventory, PowerState, RunningProcess};
use sha2::{Digest, Sha256};

use super::error::ProvenanceError;

/// Everything a run's result is keyed to, plus the conditions it ran under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunProvenance {
    /// SHA-256 over this machine's stable identity (CPU topology and GPU models); see
    /// [`machine_profile_id`].
    pub machine_profile_id: String,
    /// llama.cpp build number, as the binary itself reported it in its records.
    pub backend_build_number: u32,
    /// llama.cpp short commit hash, as the binary itself reported it.
    pub backend_commit: String,
    /// The compiler the build reports in `--version` (`built with ...`), asked of a sibling tool
    /// of the same build since llama-bench has no `--version`; `None` when no such tool is in the
    /// directory, its output has no compiler line, or its build number differs from the records'.
    pub backend_compiler: Option<String>,
    /// The `ggml-cpu-*` variant llama.cpp selected for this CPU (e.g. `alderlake`), from its own
    /// `load_backend` line; `None` for a build with a single, unsuffixed CPU backend.
    pub backend_cpu_variant: Option<String>,
    /// SHA-256 over every backend `.exe`/`.dll` digest; see [`backend_manifest_digest`].
    pub backend_manifest_digest: String,
    /// Whether those digests were verified against the pinned manifest before the run (false when
    /// the user turned integrity verification off; the digest is then of what was on disk).
    pub backend_integrity_verified: bool,
    /// SHA-256 of the model file, lowercase hex.
    pub model_sha256: String,
    /// Model file size in bytes.
    pub model_size_bytes: u64,
    /// Free VRAM per NVIDIA device in MiB, sampled immediately before launch. Empty on a machine
    /// without an NVIDIA GPU (or without its vendor tool).
    pub free_vram_mib_at_start: Vec<u64>,
    /// Free VRAM per device right after the run; `None` if the sample failed (the result is kept,
    /// since the start sample is what keys it).
    pub free_vram_mib_after: Option<Vec<u64>>,
    /// GPU utilization per device in percent, from the same sample as
    /// [`RunProvenance::free_vram_mib_at_start`] (`None` for a device that does not report it).
    /// A contention signal: well above zero means something else was computing on the GPU.
    pub gpu_utilization_pct_at_start: Vec<Option<u32>>,
    /// Other `llama-server`/`llama-bench`/`llama-cli` processes running when the run started;
    /// `None` if the process snapshot failed. Non-empty means the measurement was very likely
    /// contended and should be discarded or flagged, not trusted.
    pub other_llama_processes_at_start: Option<Vec<RunningProcess>>,
    /// CPU topology summary.
    pub cpu: CpuSummary,
    /// Physical GPUs, in the vendor tool's order.
    pub gpus: Vec<GpuSummary>,
    /// Mains or battery, and battery saver, sampled before launch; `None` if unavailable.
    pub power: Option<PowerState>,
}

impl RunProvenance {
    /// This provenance as `cerebro-store`'s record key. Free VRAM becomes the smallest per-device
    /// value (the tightest headroom), or `None` without a GPU.
    ///
    /// # Errors
    /// [`cerebro_store::RecordError`] if a digest is not 64 lowercase hex characters, which a
    /// value built by this module never is.
    pub fn store_key(&self) -> Result<cerebro_store::ProvenanceKey, cerebro_store::RecordError> {
        cerebro_store::ProvenanceKey::new(
            self.machine_profile_id.clone(),
            self.model_sha256.clone(),
            self.backend_manifest_digest.clone(),
            Some(self.backend_build_number),
            Some(self.backend_commit.clone()),
            self.backend_compiler.clone(),
            self.backend_cpu_variant.clone(),
            self.free_vram_mib_at_start.iter().copied().min(),
        )
    }
}

/// The CPU facts a measurement depends on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CpuSummary {
    /// The CPU's brand string (`cerebro_probe::cpu::brand_string`), `None` if it reports none.
    pub brand: Option<String>,
    /// Logical processors.
    pub logical_processors: u32,
    /// Physical cores.
    pub physical_cores: u32,
    /// Performance-class cores (all cores on a non-hybrid CPU).
    pub performance_cores: u32,
    /// Efficiency-class cores (0 on a non-hybrid CPU).
    pub efficiency_cores: u32,
    /// NUMA nodes.
    pub numa_nodes: u32,
    /// Whether the CPU mixes core classes.
    pub hybrid: bool,
}

impl CpuSummary {
    /// The summary of `cpu`, whose brand string is `brand`.
    pub fn new(cpu: &CpuTopology, brand: Option<String>) -> Self {
        Self {
            brand,
            logical_processors: cpu.logical_processors(),
            physical_cores: cpu.physical_cores(),
            performance_cores: cpu.performance_cores(),
            efficiency_cores: cpu.efficiency_cores(),
            numa_nodes: cpu.numa_nodes(),
            hybrid: cpu.is_hybrid(),
        }
    }
}

/// The GPU facts a measurement depends on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuSummary {
    /// Marketing name.
    pub name: String,
    /// Total memory in MiB.
    pub total_mib: u64,
    /// CUDA compute capability, e.g. `8.6`.
    pub compute_capability: String,
    /// Driver version.
    pub driver_version: String,
}

/// Summaries of every NVIDIA GPU in `inventory`.
pub fn gpu_summaries(inventory: &GpuInventory) -> Vec<GpuSummary> {
    inventory
        .nvidia()
        .iter()
        .map(|g| GpuSummary {
            name: g.name().to_string(),
            total_mib: g.total_mib(),
            compute_capability: g.compute_capability().to_string(),
            driver_version: g.driver_version().to_string(),
        })
        .collect()
}

/// SHA-256 over this machine's stable hardware identity, in a fixed text form:
///
/// ```text
/// cerebro-machine-profile-v2
/// cpu brand=12th Gen Intel(R) Core(TM) i9-12900H
/// cpu logical=20 physical=14 performance=6 efficiency=8 numa=1 hybrid=true
/// gpu NVIDIA GeForce RTX 3070 Ti Laptop GPU|8192|8.6
/// ```
///
/// One `gpu` line per device in vendor order. The brand line tells apart CPUs with the same
/// topology (an i7-12700H and an i9-12900H are both 6P+8E/20T); it reads `brand=?` when the CPU
/// reports none. v1 had no brand line, so a v1 id never equals a v2 id: records stored under v1
/// no longer match this machine's filter and are re-measured rather than silently merged. The
/// driver version is left out on purpose: a driver update changes performance (so it is recorded
/// in [`GpuSummary`]) but not which machine this is.
pub fn machine_profile_id(cpu: &CpuSummary, gpus: &[GpuSummary]) -> String {
    let mut text = format!(
        "cerebro-machine-profile-v2\ncpu brand={}\ncpu logical={} physical={} performance={} \
         efficiency={} numa={} hybrid={}\n",
        // A newline in the brand would forge a line of this text; CPUID returns ASCII, but the
        // field is public.
        cpu.brand.as_deref().unwrap_or("?").replace(['\n', '\r'], " "),
        cpu.logical_processors,
        cpu.physical_cores,
        cpu.performance_cores,
        cpu.efficiency_cores,
        cpu.numa_nodes,
        cpu.hybrid
    );
    for gpu in gpus {
        text.push_str(&format!(
            "gpu {}|{}|{}\n",
            gpu.name, gpu.total_mib, gpu.compute_capability
        ));
    }
    hex(&Sha256::digest(text.as_bytes()))
}

/// One SHA-256 summarizing a backend directory: the digest of `sha256sum`-style lines
/// (`<lowercase hex hash>  <lowercase file name>\n`), sorted by file name. Two directories get the
/// same digest exactly when they hold the same `.exe`/`.dll` files with the same contents, whatever
/// the case of their names.
pub fn backend_manifest_digest(hashes: &BTreeMap<String, String>) -> String {
    let mut lines: Vec<(String, String)> = hashes
        .iter()
        .map(|(name, hash)| (name.to_ascii_lowercase(), hash.to_ascii_lowercase()))
        .collect();
    lines.sort();
    let mut hasher = Sha256::new();
    for (name, hash) in lines {
        hasher.update(hash.as_bytes());
        hasher.update(b"  ");
        hasher.update(name.as_bytes());
        hasher.update(b"\n");
    }
    hex(&hasher.finalize())
}

/// The selected `ggml-cpu-*` variant, from llama.cpp's own
/// `load_backend: loaded CPU backend from <dir>\ggml-cpu-<variant>.dll` line. `lines` may carry
/// the supervisor's `stderr: ` tag; `None` when no such line names a suffixed CPU backend.
pub fn cpu_variant(lines: &[String]) -> Option<String> {
    lines.iter().find_map(|line| {
        let line = line.strip_prefix("stderr: ").unwrap_or(line);
        let path = line.strip_prefix("load_backend: loaded CPU backend from ")?.trim();
        // Not `Path::file_stem`: the path is a Windows path whatever the host, and its name is
        // what follows the last separator.
        let file = path.rsplit(['\\', '/']).next()?;
        let stem = file.strip_suffix(".dll").or_else(|| file.strip_suffix(".so"))?;
        let variant = stem.strip_prefix("ggml-cpu-")?;
        let valid = !variant.is_empty()
            && variant.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        valid.then(|| variant.to_string())
    })
}

/// The compiler each backend build reports, by backend manifest digest: one `--version` query per
/// build per runner, not one per run.
#[derive(Debug, Default)]
pub(super) struct CompilerCache {
    entries: Mutex<HashMap<String, Option<(u32, String)>>>,
}

impl CompilerCache {
    /// The compiler of the build whose manifest digest is `digest`, provided the `--version`
    /// answer (`query`, run only on a miss) names the same `build_number` the records reported.
    pub(super) fn compiler(
        &self,
        digest: &str,
        build_number: u32,
        query: impl FnOnce() -> Option<(u32, String)>,
    ) -> Option<String> {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = entries.entry(digest.to_string()).or_insert_with(query);
        entry
            .as_ref()
            .filter(|(build, _)| *build == build_number)
            .map(|(_, compiler)| compiler.clone())
    }
}

/// Identity of one version of a file on disk, as far as the cache is concerned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FileVersion {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
    identity: FileIdentity,
}

/// A model's digest plus the open handle it was computed from. While this lives, the file cannot
/// be written, renamed or deleted (it was opened sharing read access only), so a run that keeps it
/// until llama-bench exits measures exactly the bytes that were hashed. llama-bench's own
/// read-only open and memory map are compatible with it.
#[derive(Debug)]
pub struct PinnedModel {
    sha256: String,
    size: u64,
    _pin: File,
}

impl PinnedModel {
    /// Lowercase hex SHA-256 of the pinned file.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Size of the pinned file in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Close the handle, lifting the pin, and keep the digest and size.
    pub fn release(self) -> (String, u64) {
        (self.sha256, self.size)
    }
}

/// Model SHA-256 digests cached by (path, size, modification time, volume serial, file index).
///
/// A model is re-hashed whenever any of them changes. The key is read from the handle that is
/// then hashed, not from a second lookup by name, and re-read after hashing: a file that changed
/// underneath is refused ([`ProvenanceError::ModelChangedWhileHashing`]) rather than cached with
/// a digest of no single version of it. In memory only; one tuning session shares one cache.
#[derive(Debug, Default)]
pub struct ModelHashCache {
    entries: Mutex<HashMap<FileVersion, String>>,
}

impl ModelHashCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The lowercase hex SHA-256 and the size of the file at `path`, from the cache when this
    /// version of it was hashed before. The file is pinned only for the duration of the call; a
    /// run uses [`ModelHashCache::sha256_pinned`].
    pub fn sha256(&self, path: &Path) -> Result<(String, u64), ProvenanceError> {
        let pinned = self.sha256_pinned(path)?;
        Ok((pinned.sha256, pinned.size))
    }

    /// [`ModelHashCache::sha256`], keeping the hashed handle open deny-write in the returned
    /// [`PinnedModel`].
    ///
    /// # Errors
    /// [`ProvenanceError::ModelUnreadable`] if the file cannot be opened sharing read access only
    /// (including while another process holds it open for writing), read, or identified;
    /// [`ProvenanceError::ModelChangedWhileHashing`] if its version changed during the hash.
    pub fn sha256_pinned(&self, path: &Path) -> Result<PinnedModel, ProvenanceError> {
        let unreadable = |cause: String| ProvenanceError::ModelUnreadable {
            path: path.to_path_buf(),
            cause,
        };
        let io_unreadable = |e: io::Error| unreadable(e.to_string());
        let mut file = open_deny_write(path).map_err(io_unreadable)?;
        let before = version(path, &file).map_err(&unreadable)?;
        let cached = self.lock().get(&before).cloned();
        let sha256 = match cached {
            Some(hit) => hit,
            None => {
                let digest = hex(&sha256_reader(&mut file).map_err(io_unreadable)?);
                // Unreachable while the deny-write handle is held, but cheap, and it is what makes
                // the cache entry describe one version of the file whatever the platform.
                if version(path, &file).map_err(&unreadable)? != before {
                    return Err(ProvenanceError::ModelChangedWhileHashing {
                        path: path.to_path_buf(),
                    });
                }
                self.lock().insert(before.clone(), digest.clone());
                digest
            }
        };
        Ok(PinnedModel {
            sha256,
            size: before.size,
            _pin: file,
        })
    }

    /// How many file versions are cached.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether nothing is cached yet.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<FileVersion, String>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Open `path` for reading while refusing every other writer, renamer and deleter for as long as
/// the handle lives (`FILE_SHARE_READ` only; not 0, which would also refuse llama-bench's read).
fn open_deny_write(path: &Path) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x1;
        options.share_mode(FILE_SHARE_READ);
    }
    options.open(path)
}

fn version(path: &Path, file: &File) -> Result<FileVersion, String> {
    let meta = file.metadata().map_err(|e| e.to_string())?;
    Ok(FileVersion {
        path: path.to_path_buf(),
        size: meta.len(),
        modified: meta.modified().map_err(|e| e.to_string())?,
        identity: cerebro_probe::file_id::identity(file).map_err(|e| e.to_string())?,
    })
}

/// Bytes read per chunk while hashing.
const CHUNK_BYTES: usize = 4 << 20;

/// SHA-256 of everything `reader` yields, reading on one thread while hashing on this one, with
/// three recycled chunk buffers between them (12 MiB in flight at most).
fn sha256_reader<R: Read + Send>(reader: &mut R) -> io::Result<[u8; 32]> {
    std::thread::scope(|scope| {
        // Both channels live inside this closure: returning early (an error) drops them, which ends
        // the reader thread's loop before the scope joins it.
        let (full_tx, full_rx) = sync_channel::<io::Result<Vec<u8>>>(2);
        let (empty_tx, empty_rx) = sync_channel::<Vec<u8>>(3);
        for _ in 0..3 {
            // Capacity 3, so these never block; a send can only fail if the receiver is gone,
            // which it is not yet.
            let _ = empty_tx.send(vec![0; CHUNK_BYTES]);
        }
        scope.spawn(move || {
            while let Ok(mut buf) = empty_rx.recv() {
                buf.resize(CHUNK_BYTES, 0);
                match read_full(reader, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        buf.truncate(n);
                        if full_tx.send(Ok(buf)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = full_tx.send(Err(e));
                        break;
                    }
                }
            }
        });
        let mut hasher = Sha256::new();
        for chunk in full_rx {
            let chunk = chunk?;
            hasher.update(&chunk);
            // The reader may already have finished; a failed send just drops the buffer.
            let _ = empty_tx.send(chunk);
        }
        Ok(hasher.finalize().into())
    })
}

/// Fill `buf` from `reader` until it is full or the reader is exhausted; the count read.
fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while let Some(rest) = buf.get_mut(filled..).filter(|r| !r.is_empty()) {
        match reader.read(rest) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct TempFile(PathBuf);

    impl TempFile {
        fn new(bytes: &[u8]) -> Self {
            static N: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "cerebro-bench-provenance-{}-{}.bin",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, bytes).expect("write temp file");
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn the_pipelined_hash_equals_a_one_shot_hash_across_chunk_boundaries() {
        // Two full chunks plus a partial one, with a non-repeating pattern so a dropped or
        // reordered chunk changes the digest.
        let data: Vec<u8> = (0..(CHUNK_BYTES * 2 + 12_345))
            .map(|i| (i % 251) as u8)
            .collect();
        let expected: [u8; 32] = Sha256::digest(&data).into();
        assert_eq!(sha256_reader(&mut data.as_slice()).unwrap(), expected);
        assert_eq!(
            sha256_reader(&mut io::empty()).unwrap(),
            <[u8; 32]>::from(Sha256::digest(b""))
        );
    }

    #[test]
    fn a_read_error_is_returned_not_hashed_around() {
        struct Failing;
        impl Read for Failing {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("disk gone"))
            }
        }
        let err = sha256_reader(&mut Failing).expect_err("the error must surface");
        assert_eq!(err.to_string(), "disk gone");
    }

    #[test]
    fn the_cache_hashes_once_per_file_version_and_rehashes_a_changed_file() {
        let file = TempFile::new(b"GGUF version one");
        let cache = ModelHashCache::new();
        let (first, size) = cache.sha256(&file.0).unwrap();
        assert_eq!(first, hex(&Sha256::digest(b"GGUF version one")));
        assert_eq!(size, 16);
        assert_eq!(cache.sha256(&file.0).unwrap().0, first);
        assert_eq!(cache.len(), 1, "a second lookup of the same version is a cache hit");

        std::fs::write(&file.0, b"GGUF version two, longer").unwrap();
        let (second, size) = cache.sha256(&file.0).unwrap();
        assert_eq!(second, hex(&Sha256::digest(b"GGUF version two, longer")));
        assert_eq!(size, 24);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn a_pinned_model_cannot_be_written_renamed_or_deleted_but_can_still_be_read() {
        let file = TempFile::new(b"GGUF pinned");
        let cache = ModelHashCache::new();
        let pinned = cache.sha256_pinned(&file.0).unwrap();
        assert_eq!(pinned.sha256(), hex(&Sha256::digest(b"GGUF pinned")));
        assert_eq!(pinned.size(), 11);

        const ERROR_SHARING_VIOLATION: i32 = 32;
        let write = std::fs::OpenOptions::new().write(true).open(&file.0);
        assert_eq!(write.err().and_then(|e| e.raw_os_error()), Some(ERROR_SHARING_VIOLATION));
        let moved = file.0.with_extension("moved");
        assert!(std::fs::rename(&file.0, &moved).is_err(), "a rename must be refused");
        assert!(std::fs::remove_file(&file.0).is_err(), "a delete must be refused");
        // llama-bench's own open is read-only, which the pin admits.
        assert_eq!(std::fs::read(&file.0).unwrap(), b"GGUF pinned");
        // A second pin of the same version is a cache hit and coexists with the first.
        let again = cache.sha256_pinned(&file.0).unwrap();
        assert_eq!(again.sha256(), pinned.sha256());
        assert_eq!(cache.len(), 1);

        drop((pinned, again));
        std::fs::write(&file.0, b"GGUF rewritten").expect("writable once unpinned");
    }

    #[test]
    fn a_replaced_model_with_the_same_size_and_timestamp_is_rehashed() {
        let file = TempFile::new(b"GGUF version A");
        let cache = ModelHashCache::new();
        let first = cache.sha256(&file.0).unwrap().0;
        // Same length, timestamp copied back: only the file identity differs.
        let modified = std::fs::metadata(&file.0).unwrap().modified().unwrap();
        let other = TempFile::new(b"GGUF version B");
        File::options()
            .write(true)
            .open(&other.0)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        std::fs::rename(&other.0, &file.0).unwrap();
        assert_eq!(std::fs::metadata(&file.0).unwrap().modified().unwrap(), modified);
        let second = cache.sha256(&file.0).unwrap().0;
        assert_eq!(second, hex(&Sha256::digest(b"GGUF version B")));
        assert_ne!(first, second);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn the_cpu_variant_is_read_from_the_load_line() {
        let lines = [
            "stderr: load_backend: loaded CUDA backend from C:\\llama-cuda\\ggml-cuda.dll".to_string(),
            "stderr: load_backend: loaded CPU backend from C:\\llama-cuda\\ggml-cpu-alderlake.dll"
                .to_string(),
        ];
        assert_eq!(cpu_variant(&lines).as_deref(), Some("alderlake"));
        let untagged = ["load_backend: loaded CPU backend from C:\\x\\ggml-cpu-x64.dll".to_string()];
        assert_eq!(cpu_variant(&untagged).as_deref(), Some("x64"));
        for none in [
            "main: done",
            "load_backend: loaded CPU backend from C:\\x\\ggml-cpu.dll",
            "load_backend: loaded CPU backend from C:\\x\\ggml-cpu-.dll",
            "load_backend: loaded CPU backend from C:\\x\\ggml-cpu-a b.dll",
        ] {
            assert_eq!(cpu_variant(&[none.to_string()]), None, "{none}");
        }
    }

    #[test]
    fn the_compiler_is_queried_once_per_build_and_only_trusted_for_the_same_build() {
        let cache = CompilerCache::default();
        let digest = "d".repeat(64);
        let mut queries = 0;
        let mut query = || {
            queries += 1;
            Some((10751, "Clang 20.1.8 for Windows x86_64".to_string()))
        };
        assert_eq!(
            cache.compiler(&digest, 10751, &mut query).as_deref(),
            Some("Clang 20.1.8 for Windows x86_64")
        );
        assert_eq!(cache.compiler(&digest, 10751, &mut query).as_deref(), Some("Clang 20.1.8 for Windows x86_64"));
        // A sibling tool from another build does not describe this one.
        assert_eq!(cache.compiler(&digest, 10603, &mut query), None);
        assert_eq!(queries, 1, "one query per backend digest");
        // A failed query is remembered too, so a build without the tool is not asked every run.
        let other = "e".repeat(64);
        assert_eq!(cache.compiler(&other, 1, || None), None);
        assert_eq!(cache.compiler(&other, 1, || Some((1, "x".into()))), None);
    }

    fn sample_provenance() -> RunProvenance {
        let cpu = CpuSummary {
            brand: Some("12th Gen Intel(R) Core(TM) i9-12900H".into()),
            logical_processors: 20,
            physical_cores: 14,
            performance_cores: 6,
            efficiency_cores: 8,
            numa_nodes: 1,
            hybrid: true,
        };
        RunProvenance {
            machine_profile_id: machine_profile_id(&cpu, &[]),
            backend_build_number: 10751,
            backend_commit: "3466812d1".into(),
            backend_compiler: Some("Clang 20.1.8 for Windows x86_64".into()),
            backend_cpu_variant: Some("alderlake".into()),
            backend_manifest_digest: "b".repeat(64),
            backend_integrity_verified: true,
            model_sha256: "a".repeat(64),
            model_size_bytes: 988_208_640,
            free_vram_mib_at_start: Vec::new(),
            free_vram_mib_after: None,
            gpu_utilization_pct_at_start: Vec::new(),
            other_llama_processes_at_start: Some(Vec::new()),
            cpu,
            gpus: Vec::new(),
            power: None,
        }
    }

    #[test]
    fn the_store_key_takes_the_tightest_vram_headroom_and_the_build_facts() {
        let mut p = sample_provenance();
        p.free_vram_mib_at_start = vec![7285, 431];
        let key = p.store_key().unwrap();
        assert_eq!(key.free_vram_mib_at_start(), Some(431));
        assert_eq!(key.backend_cpu_variant(), Some("alderlake"));
        assert_eq!(key.backend_compiler(), Some("Clang 20.1.8 for Windows x86_64"));
        p.free_vram_mib_at_start.clear();
        p.backend_cpu_variant = None;
        let key = p.store_key().unwrap();
        assert_eq!(key.free_vram_mib_at_start(), None);
        assert_eq!(key.backend_cpu_variant(), None);
    }

    #[test]
    fn a_missing_model_is_a_typed_error() {
        let cache = ModelHashCache::new();
        let missing = std::env::temp_dir().join("cerebro-bench-no-such-model-7f3.gguf");
        assert!(matches!(
            cache.sha256(&missing),
            Err(ProvenanceError::ModelUnreadable { .. })
        ));
    }

    #[test]
    fn the_backend_digest_is_order_and_case_insensitive_and_content_sensitive() {
        let a = BTreeMap::from([
            ("llama-bench.exe".to_string(), "AA".repeat(32)),
            ("ggml.dll".to_string(), "bb".repeat(32)),
        ]);
        let b = BTreeMap::from([
            ("GGML.DLL".to_string(), "bb".repeat(32)),
            ("Llama-Bench.exe".to_string(), "aa".repeat(32)),
        ]);
        assert_eq!(backend_manifest_digest(&a), backend_manifest_digest(&b));
        // The exact form is `sha256sum` output sorted by name, so it can be reproduced by hand.
        let text = format!("{}  ggml.dll\n{}  llama-bench.exe\n", "bb".repeat(32), "aa".repeat(32));
        assert_eq!(backend_manifest_digest(&a), hex(&Sha256::digest(text.as_bytes())));
        let mut changed = a.clone();
        changed.insert("ggml.dll".to_string(), "cc".repeat(32));
        assert_ne!(backend_manifest_digest(&a), backend_manifest_digest(&changed));
    }

    #[test]
    fn the_machine_profile_ignores_the_driver_but_not_the_hardware() {
        let cpu = CpuSummary {
            brand: Some("12th Gen Intel(R) Core(TM) i9-12900H".into()),
            logical_processors: 20,
            physical_cores: 14,
            performance_cores: 6,
            efficiency_cores: 8,
            numa_nodes: 1,
            hybrid: true,
        };
        let gpu = |driver: &str, total: u64| GpuSummary {
            name: "NVIDIA GeForce RTX 3070 Ti Laptop GPU".into(),
            total_mib: total,
            compute_capability: "8.6".into(),
            driver_version: driver.into(),
        };
        let id = machine_profile_id(&cpu, &[gpu("576.02", 8192)]);
        assert_eq!(id.len(), 64);
        assert_eq!(id, machine_profile_id(&cpu, &[gpu("580.10", 8192)]));
        assert_ne!(id, machine_profile_id(&cpu, &[gpu("576.02", 16384)]));
        assert_ne!(id, machine_profile_id(&cpu, &[]));
        // Same topology, different CPU: a different machine.
        let i7 = CpuSummary {
            brand: Some("12th Gen Intel(R) Core(TM) i7-12700H".into()),
            ..cpu.clone()
        };
        assert_ne!(id, machine_profile_id(&i7, &[gpu("576.02", 8192)]));
        let text = "cerebro-machine-profile-v2\ncpu brand=12th Gen Intel(R) Core(TM) i9-12900H\n\
                    cpu logical=20 physical=14 performance=6 efficiency=8 numa=1 hybrid=true\n\
                    gpu NVIDIA GeForce RTX 3070 Ti Laptop GPU|8192|8.6\n";
        assert_eq!(id, hex(&Sha256::digest(text.as_bytes())));
        // No brand reads `?`, and a brand cannot inject a line.
        let unknown = CpuSummary { brand: None, ..cpu.clone() };
        let text = "cerebro-machine-profile-v2\ncpu brand=?\ncpu logical=20 physical=14 \
                    performance=6 efficiency=8 numa=1 hybrid=true\n";
        assert_eq!(machine_profile_id(&unknown, &[]), hex(&Sha256::digest(text.as_bytes())));
        let forged = CpuSummary { brand: Some("x\ngpu fake|1|1".into()), ..cpu };
        let text = "cerebro-machine-profile-v2\ncpu brand=x gpu fake|1|1\ncpu logical=20 \
                    physical=14 performance=6 efficiency=8 numa=1 hybrid=true\n";
        assert_eq!(machine_profile_id(&forged, &[]), hex(&Sha256::digest(text.as_bytes())));
    }
}
