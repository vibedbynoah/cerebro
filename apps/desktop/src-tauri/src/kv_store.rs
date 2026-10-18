//! Saved conversation caches ("Keep long chats ready"): the engine's `save_kv` images in one folder
//! (the app's data folder, `<LOCALAPPDATA>\Cerebro\kv`), so opening a long conversation again
//! continues it instead of reading its whole history back into the model.
//!
//! Files are named `<fingerprint>-<random>.kv`. An in-memory index (path, fingerprint, held tokens,
//! size, last use) is filled lazily, per model fingerprint, by reading each file's header once; a
//! file that cannot be read is deleted. The folder is kept under a byte cap by deleting the least
//! recently used files. Matching, choosing and eviction are plain functions that need no model;
//! only [`KvStore::scan`] touches one, through `Session::peek_kv`.

use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cerebro_engine::model::{Model, Session};

/// A saved cache must continue a prompt further than the session's own tokens by this much to be
/// worth restoring: reading and applying a saved cache is not free either.
pub const MIN_GAIN: usize = 64;

/// Caches shorter than this are not worth a file on disk (the file and its index cost more).
pub const MIN_SAVE_TOKENS: usize = 1024;

/// A cache is only written when returning to it later would save more than this much reading.
const MIN_SAVE_SECONDS: f64 = 3.0;

/// One saved cache file the index knows about.
#[derive(Debug, Clone)]
struct Entry {
    path: PathBuf,
    fingerprint: u64,
    /// The tokens the file holds (its header's token list).
    tokens: Vec<u32>,
    bytes: u64,
    /// Unix seconds of the entry's last use (or the file's own time when it was first indexed).
    last_used: u64,
}

/// A file the eviction pass considers: an indexed entry or a file found in the folder.
#[derive(Debug)]
struct Candidate {
    path: PathBuf,
    bytes: u64,
    last_used: u64,
}

/// A folder of saved conversation caches with a byte cap and an in-memory index.
pub struct KvStore {
    dir: PathBuf,
    cap_bytes: u64,
    /// The entries, across models; `best` refreshes the used one's time, `save` adds and replaces.
    entries: RefCell<Vec<Entry>>,
    /// Fingerprints whose files on disk have already been peeked into the index.
    scanned: HashSet<u64>,
}

impl KvStore {
    /// Opens (or makes) a store in `dir`, holding at most `cap_bytes` of saved caches; the oldest
    /// are deleted first when a save pushes the folder over the cap.
    pub fn new(dir: PathBuf, cap_bytes: u64) -> Self {
        if let Err(error) = std::fs::create_dir_all(&dir) {
            eprintln!("The folder for saved chats could not be created: {error}");
        }
        Self { dir, cap_bytes, entries: RefCell::new(Vec::new()), scanned: HashSet::new() }
    }

    /// Adds this model's files on disk to the index by reading each file's header once. A
    /// fingerprint is scanned only once, however often this is called; a file that cannot be read
    /// is deleted (a corrupt cache is useless and would otherwise be re-read every time).
    pub fn scan(&mut self, m: &Model) {
        self.scan_with(cerebro_engine::model::model_fingerprint(m), |path| {
            let mut file = std::fs::File::open(path).map_err(|error| format!("could not open it: {error}"))?;
            Session::peek_kv(m, &mut file).map_err(|error| error.to_string())
        });
    }

    /// [`KvStore::scan`] with the header read supplied by the caller, so the folder rules can be
    /// tested without a model. `peek` returns the tokens a file holds, or a sentence saying why it
    /// could not be read (the file is then deleted).
    fn scan_with(&mut self, fingerprint: u64, mut peek: impl FnMut(&Path) -> Result<Vec<u32>, String>) {
        if !self.scanned.insert(fingerprint) {
            return;
        }
        let prefix = format!("{fingerprint:016x}-");
        let Ok(read) = std::fs::read_dir(&self.dir) else { return };
        let mut found = Vec::new();
        for file in read.flatten() {
            let path = file.path();
            let named = path.extension() == Some(OsStr::new("kv")) && path.file_name().and_then(OsStr::to_str).is_some_and(|name| name.starts_with(&prefix));
            if !named || self.entries.borrow().iter().any(|entry| entry.path == path) {
                continue;
            }
            let (bytes, last_used) = match file.metadata() {
                Ok(meta) => (meta.len(), mtime_of(&meta)),
                Err(_) => (0, now_secs()),
            };
            match peek(&path) {
                Ok(tokens) => found.push(Entry { path, fingerprint, tokens, bytes, last_used }),
                Err(reason) => {
                    eprintln!("Deleting a saved chat cache that could not be read: {reason}");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        self.entries.borrow_mut().extend(found);
    }

    /// The saved cache that continues `prompt` best: the entry whose tokens are themselves a prefix
    /// of `prompt` and share the longest run with it (a cache that diverges from the prompt cannot
    /// be continued on a recurrent model, so it is never chosen). Returns its path and token count,
    /// and marks it as just used.
    pub fn best(&self, fingerprint: u64, prompt: &[u32]) -> Option<(PathBuf, usize)> {
        let mut entries = self.entries.borrow_mut();
        let best = pick(&entries, fingerprint, prompt)?;
        entries[best].last_used = now_secs();
        let entry = &entries[best];
        Some((entry.path.clone(), entry.tokens.len()))
    }

    /// Stores `bytes` (a `save_kv` image of `tokens`) for this model. An older entry whose tokens
    /// are a prefix of `tokens` (the same conversation, continued) is replaced: its file is deleted
    /// now. The new file is written on a background thread (temp file + rename, so a reader never
    /// sees a half-written cache), and the least recently used files are deleted until the folder
    /// is under the cap again.
    pub fn save(&mut self, fingerprint: u64, tokens: &[u32], bytes: Vec<u8>) {
        let size = bytes.len() as u64;
        let replaced: Vec<PathBuf> = self.entries.borrow().iter().filter(|entry| entry.fingerprint == fingerprint && is_prefix(&entry.tokens, tokens)).map(|entry| entry.path.clone()).collect();
        for path in &replaced {
            let _ = std::fs::remove_file(path);
        }
        self.entries.borrow_mut().retain(|entry| !replaced.contains(&entry.path));

        let path = self.dir.join(format!("{fingerprint:016x}-{}.kv", random_hex16()));
        write_on_thread(self.dir.clone(), path.clone(), bytes);
        self.entries.borrow_mut().push(Entry { path, fingerprint, tokens: tokens.to_vec(), bytes: size, last_used: now_secs() });
        self.evict();
    }

    /// Deletes every file in the folder and forgets the index.
    pub fn clear(&mut self) {
        match std::fs::read_dir(&self.dir) {
            Ok(read) => {
                for file in read.flatten() {
                    if file.path().is_file() {
                        let _ = std::fs::remove_file(file.path());
                    }
                }
            }
            Err(error) => eprintln!("The folder for saved chats could not be read: {error}"),
        }
        self.entries.borrow_mut().clear();
        self.scanned.clear();
    }

    /// How much the folder holds right now: (files, bytes) over every `.kv` file.
    pub fn usage(&self) -> (usize, u64) {
        let files = self.folder_files();
        (files.len(), files.iter().map(|file| file.bytes).sum())
    }

    /// Deletes the least recently used files until the folder is under the cap. A file that is in
    /// the index but not on disk yet (a save still being written) counts with its indexed size.
    fn evict(&mut self) {
        let mut files = self.folder_files();
        for entry in self.entries.borrow().iter() {
            if !files.iter().any(|file| file.path == entry.path) {
                files.push(Candidate { path: entry.path.clone(), bytes: entry.bytes, last_used: entry.last_used });
            }
        }
        for file in &mut files {
            if let Some(entry) = self.entries.borrow().iter().find(|entry| entry.path == file.path) {
                file.last_used = entry.last_used;
            }
        }
        for i in eviction_order(&files, self.cap_bytes) {
            let path = files[i].path.clone();
            let _ = std::fs::remove_file(&path);
            self.entries.borrow_mut().retain(|entry| entry.path != path);
        }
    }

    /// Every `.kv` file in the folder, with its size and modification time.
    fn folder_files(&self) -> Vec<Candidate> {
        let Ok(read) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let mut files = Vec::new();
        for file in read.flatten() {
            let path = file.path();
            if path.extension() != Some(OsStr::new("kv")) {
                continue;
            }
            let Ok(meta) = file.metadata() else { continue };
            if meta.is_file() {
                files.push(Candidate { path, bytes: meta.len(), last_used: mtime_of(&meta) });
            }
        }
        files
    }
}

/// How many leading tokens two runs share.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whether `tokens` is a prefix of `prompt`, strictly: a saved cache that needs more tokens than
/// the prompt has, or that diverges anywhere, cannot continue it.
fn is_prefix(tokens: &[u32], prompt: &[u32]) -> bool {
    tokens.len() <= prompt.len() && common_prefix(tokens, prompt) == tokens.len()
}

/// The best entry for `prompt` among this model's own: the longest tokens that are a prefix of the
/// prompt. Ties (the same tokens saved twice) go to the most recently used.
fn pick(entries: &[Entry], fingerprint: u64, prompt: &[u32]) -> Option<usize> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.fingerprint == fingerprint && !entry.tokens.is_empty() && is_prefix(&entry.tokens, prompt))
        .max_by_key(|(_, entry)| (entry.tokens.len(), entry.last_used))
        .map(|(i, _)| i)
}

/// The indexes of the files to delete, least recently used first, until the rest fit in `cap`
/// bytes. Ties go by path, so the same set of files always evicts in the same order.
fn eviction_order(files: &[Candidate], cap: u64) -> Vec<usize> {
    let mut total: u64 = files.iter().map(|file| file.bytes).sum();
    let mut order: Vec<usize> = (0..files.len()).collect();
    order.sort_by(|&a, &b| files[a].last_used.cmp(&files[b].last_used).then_with(|| files[a].path.cmp(&files[b].path)));
    let mut doomed = Vec::new();
    for i in order {
        if total <= cap {
            break;
        }
        total = total.saturating_sub(files[i].bytes);
        doomed.push(i);
    }
    doomed
}

/// Whether restoring this conversation later would save more than [`MIN_SAVE_SECONDS`] of reading:
/// `tokens` read at `prefill_tps`. A speed of zero (or none) means it is unknown, and then a file
/// is written: a wrong guess costs disk space, no file costs the user minutes.
pub fn worth_saving(tokens: usize, prefill_tps: f64) -> bool {
    !(prefill_tps > 0.0) || tokens as f64 / prefill_tps > MIN_SAVE_SECONDS
}

/// Writes `bytes` to `path` on a background thread, through a temp file and a rename in the same
/// folder, so a reader never sees a half-written cache. A failure logs and leaves nothing behind.
fn write_on_thread(dir: PathBuf, path: PathBuf, bytes: Vec<u8>) {
    let writer = move || {
        if let Err(error) = std::fs::create_dir_all(&dir) {
            return eprintln!("A chat cache could not be written: {error}");
        }
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or("cache").to_owned();
        let temp = dir.join(format!("{name}.tmp"));
        if let Err(error) = std::fs::write(&temp, &bytes) {
            let _ = std::fs::remove_file(&temp);
            return eprintln!("A chat cache could not be written: {error}");
        }
        if let Err(error) = std::fs::rename(&temp, &path) {
            let _ = std::fs::remove_file(&temp);
            eprintln!("A chat cache could not be written: {error}");
        }
    };
    if let Err(error) = std::thread::Builder::new().name("cerebro-kv-save".into()).spawn(writer) {
        eprintln!("A chat cache could not be written: {error}");
    }
}

/// One file's modification time as Unix seconds (0 when the clock cannot say).
fn mtime_of(meta: &std::fs::Metadata) -> u64 {
    meta.modified().ok().and_then(|time| time.duration_since(UNIX_EPOCH).ok()).map_or(0, |since| since.as_secs())
}

/// The clock as Unix seconds (0 before 1970, which a running clock never reports).
fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs())
}

/// 16 random hex digits for a file name. File names only need to be unique inside one folder, not
/// secret; when the OS CSPRNG is unavailable the clock is enough for that.
fn random_hex16() -> String {
    let mut bytes = [0u8; 8];
    if getrandom::fill(&mut bytes).is_err() {
        bytes = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_nanos() as u64).to_le_bytes();
    }
    format!("{:016x}", u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, fingerprint: u64, tokens: &[u32], last_used: u64) -> Entry {
        Entry { path: PathBuf::from(path), fingerprint, tokens: tokens.to_vec(), bytes: 0, last_used }
    }

    fn candidate(path: &str, bytes: u64, last_used: u64) -> Candidate {
        Candidate { path: PathBuf::from(path), bytes, last_used }
    }

    /// A store in a folder of its own, under the system temp folder (no model needed).
    fn temp_store(cap: u64) -> KvStore {
        KvStore::new(std::env::temp_dir().join(format!("cerebro-kv-test-{}-{}", std::process::id(), random_hex16())), cap)
    }

    /// Waits briefly for a background write, so a test never races its own thread.
    fn wait_for(mut ready: impl FnMut() -> bool, what: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if ready() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("timed out waiting for {what}");
    }

    #[test]
    fn common_prefix_counts_the_shared_start() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[1], &[]), 0);
        assert_eq!(common_prefix(&[1, 2], &[1, 2, 3]), 2);
    }

    #[test]
    fn only_a_whole_prefix_continues_a_prompt() {
        assert!(is_prefix(&[1, 2], &[1, 2, 3]));
        assert!(is_prefix(&[], &[1]));
        assert!(is_prefix(&[1, 2, 3], &[1, 2, 3]));
        assert!(!is_prefix(&[1, 2], &[1, 1, 2]));
        assert!(!is_prefix(&[1, 2, 3], &[1, 2]));
        assert!(!is_prefix(&[7], &[]));
    }

    #[test]
    fn best_picks_the_longest_saved_prefix() {
        let entries = [entry("a", 9, &[1, 2], 5), entry("b", 9, &[1, 2, 3, 4], 1), entry("c", 9, &[1, 2, 3], 9)];
        assert_eq!(pick(&entries, 9, &[1, 2, 3, 4, 5]), Some(1));
        assert_eq!(pick(&entries, 9, &[1, 2]), Some(0));
        assert_eq!(pick(&entries, 9, &[5, 5, 5]), None);
    }

    #[test]
    fn entries_that_diverge_or_belong_to_another_model_are_ignored() {
        let entries = [entry("a", 9, &[1, 2, 8], 5), entry("b", 7, &[1, 2, 3], 5)];
        assert_eq!(pick(&entries, 9, &[1, 2, 3, 4]), None);
        assert_eq!(pick(&entries, 7, &[1, 2, 3, 4]).map(|i| entries[i].path.clone()), Some(PathBuf::from("b")));
    }

    #[test]
    fn the_most_recently_used_entry_wins_a_tie() {
        let entries = [entry("old", 9, &[1, 2], 10), entry("new", 9, &[1, 2], 20)];
        assert_eq!(pick(&entries, 9, &[1, 2, 3]).map(|i| entries[i].path.clone()), Some(PathBuf::from("new")));
    }

    #[test]
    fn eviction_keeps_everything_under_the_cap() {
        let files = [candidate("a", 100, 1), candidate("b", 100, 2)];
        assert!(eviction_order(&files, 200).is_empty());
        assert!(eviction_order(&files, 1000).is_empty());
    }

    #[test]
    fn eviction_deletes_the_least_recently_used_first() {
        let files = [candidate("new", 60, 30), candidate("old", 60, 10), candidate("mid", 60, 20)];
        assert_eq!(eviction_order(&files, 150), vec![1]);
        assert_eq!(eviction_order(&files, 60), vec![1, 2]);
        assert_eq!(eviction_order(&files, 200), Vec::<usize>::new());
    }

    #[test]
    fn eviction_removes_everything_at_a_zero_cap() {
        let files = [candidate("a", 5, 1), candidate("b", 7, 2)];
        assert_eq!(eviction_order(&files, 0), vec![0, 1]);
        assert_eq!(eviction_order(&files, 11), vec![0]);
    }

    #[test]
    fn eviction_order_is_stable_when_use_times_are_equal() {
        let files = [candidate("b", 10, 5), candidate("a", 10, 5)];
        assert_eq!(eviction_order(&files, 10), vec![1]);
    }

    #[test]
    fn a_slow_read_is_worth_saving() {
        assert!(worth_saving(8192, 2.0));
        assert!(worth_saving(1024, 0.0));
        assert!(worth_saving(4096, f64::NAN));
        assert!(!worth_saving(8192, 4000.0));
        assert!(worth_saving(8192, 100.0));
    }

    #[test]
    fn usage_and_clear_see_the_folder() {
        let mut store = temp_store(1 << 20);
        std::fs::write(store.dir.join("aaaa-1111222233334444.kv"), vec![0u8; 32]).unwrap();
        std::fs::write(store.dir.join("notes.txt"), vec![0u8; 100]).unwrap();
        assert_eq!(store.usage(), (1, 32));
        store.clear();
        assert_eq!(store.usage(), (0, 0));
        assert!(!store.dir.join("aaaa-1111222233334444.kv").exists());
        assert!(!store.dir.join("notes.txt").exists());
        let _ = std::fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn scanning_indexes_one_models_files_and_deletes_broken_ones() {
        let mut store = temp_store(1 << 20);
        let good = store.dir.join("0000000000000009-0123456789abcdef.kv");
        let broken = store.dir.join("0000000000000009-fedcba9876543210.kv");
        let foreign = store.dir.join("0000000000000007-0123456789abcdef.kv");
        for path in [&good, &broken, &foreign] {
            std::fs::write(path, b"x").unwrap();
        }
        store.scan_with(9, |path| if path == broken { Err("cut short".into()) } else { Ok(vec![1, 2, 3]) });
        assert!(!broken.exists());
        assert!(good.exists() && foreign.exists());
        let paths: Vec<PathBuf> = store.entries.borrow().iter().map(|entry| entry.path.clone()).collect();
        assert_eq!(paths, vec![good.clone()]);
        // A second scan of the same fingerprint does not read anything again.
        store.scan_with(9, |_| panic!("scanned twice"));
        // Another model's scan sees only its own file.
        store.scan_with(7, |_| Ok(vec![4, 5]));
        assert_eq!(store.entries.borrow().len(), 2);
        let _ = std::fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn saving_replaces_the_same_conversations_older_cache() {
        let mut store = temp_store(1 << 20);
        let old = store.dir.join("0000000000000009-aaaaaaaaaaaaaaaa.kv");
        std::fs::write(&old, vec![0u8; 16]).unwrap();
        store.entries.borrow_mut().push(entry(old.to_str().unwrap(), 9, &[1, 2, 3], 1));
        store.save(9, &[1, 2, 3, 4, 5], vec![0u8; 64]);
        assert!(!old.exists());
        let (path, len) = store.best(9, &[1, 2, 3, 4, 5, 6]).expect("the new cache is in the index");
        assert_eq!(len, 5);
        wait_for(|| path.exists(), "the saved cache file");
        let _ = std::fs::remove_dir_all(&store.dir);
    }

    #[test]
    fn saving_evicts_the_least_recently_used_files() {
        let mut store = temp_store(100);
        let old = store.dir.join("0000000000000009-1111111111111111.kv");
        let newer = store.dir.join("0000000000000008-2222222222222222.kv");
        std::fs::write(&old, vec![0u8; 60]).unwrap();
        std::fs::write(&newer, vec![0u8; 60]).unwrap();
        store.entries.borrow_mut().push(entry(old.to_str().unwrap(), 7, &[1], 10));
        store.entries.borrow_mut().push(entry(newer.to_str().unwrap(), 8, &[2], 20));
        store.save(9, &[1, 9, 9], vec![0u8; 64]);
        // 60 + 60 + 64 bytes over a 100-byte cap: both older files go, the new one stays.
        assert!(!old.exists() && !newer.exists());
        assert_eq!(store.entries.borrow().len(), 1);
        let path = store.entries.borrow()[0].path.clone();
        wait_for(|| path.exists(), "the saved cache file");
        assert_eq!(store.usage(), (1, 64));
        let _ = std::fs::remove_dir_all(&store.dir);
    }
}
