//! `settings.json` on disk: comment stripping, the 1 MiB cap, atomic saves, and a debounced watch.
//!
//! # Why comments are stripped by a hand-written scanner rather than a crate
//!
//! JSON5/JSONC parsers exist, but they parse the *whole* document their own way; this scanner does
//! exactly one job -- turn `//` and `/* */` outside string literals into whitespace, preserving
//! every line break and every other byte's **byte offset** on its line -- and then hands the
//! result to the same `serde_json` parser every other path in this crate uses. `serde_json`
//! reports `column` as a byte offset from the start of the line, so a comment character must be
//! blanked out with as many spaces as it is bytes wide (see [`blank_out`]), not with one space per
//! `char`; otherwise a comment containing so much as one non-ASCII character would shift every
//! later column `reload()` reports on that line. That is what keeps error line/column numbers
//! meaningful, and keeps exactly one JSON parser in the crate's trust boundary.
//!
//! # Why `watch` watches the directory, not the file
//!
//! An editor's "safe save" (and Cerebro's own [`ConfigFile::save`]) does not write `settings.json`
//! in place; it writes a temp file and renames over it, which on Windows can hand a file-handle
//! watch a stale handle to the now-unlinked original. Watching the parent directory with
//! [`notify::RecursiveMode::NonRecursive`] and filtering events by file name survives both an
//! editor's replace-on-save and the file being deleted and manually recreated -- there is no
//! instant where nothing is watched.
//!
//! # Why every read opens through a reparse-point guard
//!
//! A principal who can write this directory can plant a symlink or NTFS junction named
//! `settings.json` that silently redirects every read to a file of their choosing. `settings.json`
//! is a higher-value target for exactly that than most files this workspace handles: it names
//! [`crate::schema::BackendSection::dir`] and [`crate::schema::SecuritySection::backend_path`],
//! i.e. the inputs to [`crate::profile::DefaultPreflighter::decide`]. So every read here opens with
//! `FILE_FLAG_OPEN_REPARSE_POINT` and refuses on `FILE_ATTRIBUTE_REPARSE_POINT`, mirroring
//! `cerebro_store`'s identical guard (`crates/cerebro-store/src/store.rs`) on a lower-value file for
//! the identical reason.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};

use crate::error::{cap_echo, ConfigError};
use crate::profile::Source;
use crate::schema::Settings;

/// File name of the settings document inside its directory.
pub const SETTINGS_FILE_NAME: &str = "settings.json";

/// Hard cap on settings-file size. Chosen to be generously larger than any plausible hand-edited
/// config while still being small enough that reading a whole file into memory before parsing is
/// always cheap.
pub const MAX_SETTINGS_FILE_BYTES: u64 = 1024 * 1024;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const DEBOUNCE: Duration = Duration::from_millis(200);
const POLL: Duration = Duration::from_millis(50);

/// `FILE_FLAG_OPEN_REPARSE_POINT` (winbase.h): open the reparse point itself rather than
/// transparently following it to whatever it points at. See the module documentation above.
#[cfg(windows)]
const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;

/// `FILE_ATTRIBUTE_REPARSE_POINT` (winnt.h): set on a handle's metadata when the path it was opened
/// with (via [`FILE_FLAG_OPEN_REPARSE_POINT`]) named a symlink or NTFS junction, not a regular file.
#[cfg(windows)]
const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

/// Turn an [`std::io::Error`] encountered while reading `path` into a [`ConfigError::Read`],
/// carrying the error's [`std::io::ErrorKind`] so [`ConfigFile::open`] can tell "genuinely does not
/// exist yet" apart from every other failure.
fn read_err(path: &Path, e: &std::io::Error) -> ConfigError {
    ConfigError::Read {
        path: path.to_path_buf(),
        kind: e.kind(),
        cause: e.to_string(),
    }
}

/// Open `path` by name, never through what a planted link points at, refusing outright if `path`
/// itself turns out to be a reparse point.
///
/// `MetadataExt` (for `file_attributes`) and `OpenOptionsExt` (for `custom_flags`) are imported
/// inside this function rather than at module scope: pulling them in unconditionally would be an
/// unused import on a non-Windows build under `-D warnings` -- `cerebro_store` documents this exact
/// trap next to its own copy of this guard (`crates/cerebro-store/src/store.rs`). Both are safe std
/// APIs, so this stays under the crate's `#![forbid(unsafe_code)]`.
#[cfg(windows)]
fn open_settings_file(path: &Path) -> Result<std::fs::File, ConfigError> {
    use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};

    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(|e| read_err(path, &e))?;
    let attrs = file.metadata().map_err(|e| read_err(path, &e))?.file_attributes();
    if attrs & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(ConfigError::PathIsReparsePoint { path: path.to_path_buf() });
    }
    Ok(file)
}

/// Off Windows there is no reparse-point defense (this crate, like the rest of Cerebro, is
/// Windows-first -- see `docs/ARCHITECTURE.md`), so this just opens the file normally.
#[cfg(not(windows))]
fn open_settings_file(path: &Path) -> Result<std::fs::File, ConfigError> {
    std::fs::File::open(path).map_err(|e| read_err(path, &e))
}

/// A loaded `settings.json`, tracking the last known-good [`Settings`] in memory.
#[derive(Debug)]
pub struct ConfigFile {
    path: PathBuf,
    current: Settings,
}

impl ConfigFile {
    /// Open (or, if absent, create) `settings.json` inside `dir`.
    ///
    /// A default document is bootstrapped **only** when the file genuinely does not exist yet
    /// ([`std::io::ErrorKind::NotFound`]). Every other load failure -- the file cannot parse,
    /// permission was denied, another process holds it with an incompatible share mode, a transient
    /// I/O error -- propagates untouched and writes nothing. `Path::is_file()` cannot be used to
    /// make this distinction: it is documented to return `false` for every one of those failures,
    /// not just for "does not exist," so gating the bootstrap on it (an earlier version of this
    /// function did) would silently replace a config that merely could not be stat'ed at that
    /// instant with [`Settings::default`] -- destroying a user's `security.model_roots`,
    /// `backend.dir` and `api` configuration for a reason that had nothing to do with the file's
    /// content.
    pub fn open(dir: &Path) -> Result<Self, ConfigError> {
        let path = dir.join(SETTINGS_FILE_NAME);
        match Self::load_from_disk(&path) {
            Ok(current) => Ok(Self { path, current }),
            Err(ConfigError::Read { kind: std::io::ErrorKind::NotFound, .. }) => {
                let file = Self { path, current: Settings::default() };
                file.save()?;
                Ok(file)
            }
            Err(e) => Err(e),
        }
    }

    /// The current, last known-good settings.
    pub fn current(&self) -> &Settings {
        &self.current
    }

    /// The `settings.json` path this instance was opened against.
    ///
    /// Exists so [`watch_into`] can derive the directory to watch without reaching into a private
    /// field.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Re-read `settings.json` from disk.
    ///
    /// On success, replaces the in-memory settings and returns a reference to them. On failure,
    /// the in-memory settings (and the file on disk, which this function never writes) are both
    /// left exactly as they were -- the caller sees the error, and [`current`](Self::current)
    /// keeps returning the last value that did parse.
    pub fn reload(&mut self) -> Result<&Settings, ConfigError> {
        let next = Self::load_from_disk(&self.path)?;
        self.current = next;
        Ok(&self.current)
    }

    /// Apply `patch` from `source` and persist it, or leave memory and disk both untouched.
    ///
    /// The whole patch is validated before anything is committed (see
    /// [`Settings::apply_patch`](crate::schema::Settings::apply_patch)), and a failed [`save`
    ///](Self::save) rolls `current` back to what it was before the patch: `current` must never
    /// describe a document that is not the one on disk, or a later, unrelated `save()` would
    /// silently overwrite whatever the file actually holds.
    ///
    /// # Note on feedback with [`watch`]
    ///
    /// A successful `save` here writes `settings.json`, which a concurrent [`watch`] on the same
    /// directory will observe and report as a fresh, valid reload. A caller that both `patch`es
    /// through this `ConfigFile` and drives [`apply_reload`](Self::apply_reload) from a `watch` on
    /// the same directory will see that reload of its own write; applying it is a no-op (the
    /// settings it carries are already `current`), not a loop, but callers that care should be
    /// aware of the round trip.
    pub fn patch(&mut self, patch: &serde_json::Value, source: Source) -> Result<&Settings, ConfigError> {
        let mut next = self.current.clone();
        next.apply_patch(patch, source)?;
        let previous = std::mem::replace(&mut self.current, next);
        if let Err(e) = self.save() {
            self.current = previous;
            return Err(e);
        }
        Ok(&self.current)
    }

    /// Adopt one [`watch`] result as the new last-good settings, or report why it was not one.
    ///
    /// This is where "applies only valid edits" (the watch contract) actually happens: a valid
    /// reload (`Ok`) becomes `current`; an invalid one (`Err`) is returned to the caller and changes
    /// nothing. Doing this here, rather than inside the watcher thread itself, means that thread
    /// never touches a `ConfigFile` at all -- see [`watch`]'s documentation for why that matters.
    pub fn apply_reload(&mut self, result: Result<Settings, ConfigError>) -> Result<&Settings, ConfigError> {
        self.current = result?;
        Ok(&self.current)
    }

    /// Write the current settings to disk as write-temp-in-same-dir, `fsync`, rename.
    ///
    /// The temp file lives beside the destination so the final rename is same-volume and therefore
    /// atomic on Windows (`rename` there is implemented over `MoveFileExW` with
    /// `MOVEFILE_REPLACE_EXISTING`). `fsync`ing the temp file before the rename means a crash after
    /// the rename can never observe a truncated or partially-written file at `settings.json`;
    /// `fsync`ing the containing directory afterward (to make the *rename itself* durable) has no
    /// portable Win32 equivalent and is not attempted -- this is a best-effort durability guarantee
    /// for a single-user local tool, not a database.
    pub fn save(&self) -> Result<(), ConfigError> {
        let text = serde_json::to_string_pretty(&self.current).map_err(|e| ConfigError::Write {
            path: self.path.clone(),
            cause: e.to_string(),
        })?;
        write_atomic(&self.path, text.as_bytes())
    }

    /// Read, size-check, strip comments from, parse, and structurally validate one settings file.
    /// Does not touch `self` -- shared by [`open`](Self::open), [`reload`](Self::reload), and
    /// [`watch`].
    fn load_from_disk(path: &Path) -> Result<Settings, ConfigError> {
        let mut file = open_settings_file(path)?;

        // The cap must bound bytes actually *read*, not a metadata snapshot taken before the read:
        // between an `fs::metadata` call and a subsequent `fs::read`, another writer (another
        // Cerebro surface, an editor, or a hostile local principal who can write this directory)
        // can grow the file, and the whole, now-oversized, result would still be buffered into
        // memory. `Read::take(CAP + 1)` cannot be invalidated that way: reading one byte past the
        // cap is what tells "exactly at the cap" (legal) apart from "over" without ever buffering
        // the oversized remainder.
        let mut bytes = Vec::with_capacity(64 * 1024);
        let read = file
            .by_ref()
            .take(MAX_SETTINGS_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| read_err(path, &e))? as u64;
        if read > MAX_SETTINGS_FILE_BYTES {
            // `read` is capped at `MAX_SETTINGS_FILE_BYTES + 1` by `take`, so on its own it is a
            // "more than" sentinel, not the file's actual size. Ask the already-open handle for
            // the real length; fall back to the sentinel only if that metadata call itself fails,
            // rather than turning an over-cap refusal into an unrelated I/O error.
            let size = file.metadata().map(|m| m.len()).unwrap_or(read);
            return Err(ConfigError::TooLarge {
                path: path.to_path_buf(),
                size,
            });
        }

        let text = match std::str::from_utf8(&bytes) {
            Ok(text) => text,
            Err(e) => {
                // Not valid UTF-8 at all. Reported the same shape as a JSON syntax error rather
                // than as a distinct I/O failure: from a caller's point of view this is one more
                // way the file's *content* is invalid, not a filesystem problem.
                let (line, column) = line_col_of_byte_offset(&bytes[..e.valid_up_to()]);
                return Err(ConfigError::Invalid {
                    path: path.to_path_buf(),
                    line,
                    column,
                    message: format!("file is not valid UTF-8 at byte offset {}", e.valid_up_to()),
                });
            }
        };

        let stripped = strip_comments(text);
        let settings: Settings = serde_json::from_str(&stripped).map_err(|e| ConfigError::Invalid {
            path: path.to_path_buf(),
            line: e.line(),
            column: e.column(),
            // `e.to_string()` echoes untrusted content verbatim (an unknown field's own name, or
            // the text of a wrong-typed value), so it is capped here, at construction -- see
            // `ConfigError::Invalid::message`'s doc comment.
            message: cap_echo(&e.to_string()),
        })?;
        // Structural checks `serde` alone cannot express (currently: `backend.dir` and
        // `security.backend_path` must agree) apply on every load, not just on a patch.
        settings.validate()?;
        Ok(settings)
    }
}

/// 1-based (line, column) of the byte immediately after `prefix`, by counting `\n` bytes in it.
///
/// Used only to give a non-UTF-8 failure a "plausible" position in the same (line, column) shape
/// `serde_json::Error` uses for JSON syntax failures, so every content-level [`ConfigError::Invalid`]
/// looks the same to a caller regardless of which check produced it.
fn line_col_of_byte_offset(prefix: &[u8]) -> (usize, usize) {
    let mut line = 1usize;
    let mut column = 1usize;
    for &b in prefix {
        if b == b'\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

/// Blank out one comment character, preserving its **byte** width so the stripped text has
/// byte-identical line offsets to the original.
///
/// `serde_json` reports a parse error's `column` as a byte offset from the start of its line
/// (`SliceRead::position_of_index` subtracts byte indices), not a `char` count. Replacing a
/// multi-byte character (an em dash, an accented letter, a box-drawing rule) with a single space
/// would shrink that line by `len_utf8() - 1` bytes, silently shifting every later column
/// `reload()` reports on it. Iterating `c.len_utf8()` times keeps the width exact; this only ever
/// runs on the handful of bytes inside a comment, never on the file as a whole, so allocating one
/// space at a time here is not the hot loop the rest of this function's `push`-per-`char` is.
fn blank_out(c: char, out: &mut String) {
    if c == '\n' {
        out.push('\n');
    } else {
        for _ in 0..c.len_utf8() {
            out.push(' ');
        }
    }
}

/// Replace every `//...` and `/* ... */` comment that falls outside a JSON string literal with
/// spaces (and, inside a comment, `\n` with `\n`), leaving every other byte untouched.
///
/// Operating char-by-char (not byte-by-byte) means a multi-byte UTF-8 character is never split
/// across the substitution boundary, and [`blank_out`] blanks each comment character with as many
/// bytes of whitespace as it is wide, so the stripped text's line offsets match the original
/// byte-for-byte -- see `comment_stripping_preserves_byte_columns_for_non_ascii_comments` below.
/// String-literal tracking understands `\"` and `\\` escapes, so `"https://example.com"` and a
/// string containing a literal `/*` are never mistaken for comments -- see the
/// `comment_stripping_ignores_urls_and_star_slash_inside_strings` test below, which is exactly the
/// fixture the brief calls for.
pub fn strip_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;

    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }

        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '/' if chars.peek() == Some(&'/') => {
                chars.next();
                out.push(' ');
                out.push(' ');
                for next in chars.by_ref() {
                    blank_out(next, &mut out);
                    if next == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                out.push(' ');
                out.push(' ');
                let mut prev_was_star = false;
                for next in chars.by_ref() {
                    blank_out(next, &mut out);
                    if prev_was_star && next == '/' {
                        break;
                    }
                    prev_was_star = next == '*';
                }
            }
            _ => out.push(c),
        }
    }

    out
}

/// Write-temp-in-same-dir + `fsync` + rename. See [`ConfigFile::save`] for the durability rationale.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), ConfigError> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let base_name = path.file_name().and_then(|n| n.to_str()).unwrap_or(SETTINGS_FILE_NAME);
    let unique = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = dir.join(format!(".{base_name}.tmp-{}-{unique}", std::process::id()));

    let write_result = (|| -> Result<(), ConfigError> {
        use std::io::Write as _;
        // `create_new` refuses to open if anything at all -- a regular file, a symlink, a junction
        // -- already exists at this exact name, so a principal who can write this directory cannot
        // pre-plant a link here for the write that follows to go through. The name already carries
        // this process's PID and a monotonically increasing counter, so a genuine collision would
        // mean this exact process reused a name it had already used, not an innocent coincidence.
        let mut tmp = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| ConfigError::Write {
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
        tmp.write_all(bytes).map_err(|e| ConfigError::Write {
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;
        tmp.sync_all().map_err(|e| ConfigError::Write {
            path: path.to_path_buf(),
            cause: e.to_string(),
        })
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    std::fs::rename(&tmp_path, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        ConfigError::Write {
            path: path.to_path_buf(),
            cause: e.to_string(),
        }
    })
}

/// A live watch on one directory's `settings.json`, established by [`watch`].
///
/// Dropping it stops the background thread. The stop signal is polled at the same cadence as the
/// debounce timer, so a drop blocks for at most one [`POLL`] interval.
pub struct ConfigWatcher {
    _watcher: RecommendedWatcher,
    stop: Arc<AtomicBool>,
    alive: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ConfigWatcher {
    /// Whether the background watch is still running.
    ///
    /// It goes `false` on an ordinary drop or on the file-name filter's own thread exiting, and --
    /// the case a caller cannot otherwise see -- when the channel this watch publishes to closes
    /// from the far end. That happens whenever nothing is reading `out_rx` any more, which includes
    /// a [`watch_into`] callback that panicked and unwound its drain thread: the drain thread's
    /// receiver drops, this watch's next `send` fails, and the watch stops silently unless a caller
    /// checks this. See [`watch_into`]'s documentation.
    pub fn is_running(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
}

impl Drop for ConfigWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Clears the paired [`ConfigWatcher::is_running`] flag when the watcher thread's closure returns,
/// whichever of its several `return` sites triggered that. A `Drop` guard makes "every exit path
/// clears the flag" true by construction, rather than something that has to be remembered at each
/// `return` (and re-remembered if one is ever added) -- see the
/// `a_panicking_callback_is_visible_through_is_running` test below, which exercises exactly one of
/// those paths (the closed-channel `send` failure), not the only one that exists.
struct ClearAliveOnExit(Arc<AtomicBool>);

impl Drop for ClearAliveOnExit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// Watch `dir`'s `settings.json` for changes, publishing each debounced reload result on the
/// returned channel at most once per 200 ms-quiet period.
///
/// A result is `Ok` only for an edit that parses as a valid [`Settings`]; an invalid edit is
/// published as `Err` and, on its own, never changes what any [`ConfigFile`] considers the last good
/// settings -- a caller adopts a result explicitly via [`ConfigFile::apply_reload`]. The thread
/// survives the file being deleted and recreated (see the module documentation above) and exits
/// within one [`POLL`] interval of either the returned [`ConfigWatcher`] being dropped or the
/// receiver being dropped (a send to a closed channel means nobody is listening any more, so the
/// thread exits rather than spin forever computing results nobody reads).
///
/// # Why a channel, not a callback
///
/// An earlier version of this function took an `on_change: FnMut` and invoked it directly on this
/// background thread inside `std::panic::catch_unwind`, documented as making the thread "never
/// panic regardless of what `on_change` does." That claim did not hold under this workspace's own
/// build configuration: the root `Cargo.toml` sets `[profile.release] panic = "abort"`, under which
/// there is no unwinding for `catch_unwind` to catch at all -- a panic inside a release-mode
/// `on_change` would abort the whole process, taking down the UI and any in-flight backend
/// supervision with it. Publishing to a channel instead means this thread runs no caller code, so
/// there is nothing on it that can panic in the first place: a caller that wants callback-style
/// delivery drains the channel on its own thread, where its own panic is its own problem --
/// [`watch_into`] is that drain thread, and is the brief's `watch(dir, cb)` contract.
pub fn watch(dir: PathBuf) -> Result<(ConfigWatcher, Receiver<Result<Settings, ConfigError>>), ConfigError> {
    // Canonicalize once, up front, so this thread's file-name filter (below) and `notify`'s echoed
    // event paths are always talking about the same directory regardless of how the caller spelled
    // `dir` -- a relative path, a trailing separator, a `.` component, and a canonical path all
    // resolve to this one form.
    let canonical_dir = std::fs::canonicalize(&dir).map_err(|e| ConfigError::WatchSetup {
        dir: dir.clone(),
        cause: e.to_string(),
    })?;

    let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher = notify::recommended_watcher(move |event| {
        // A send failure only means the watcher thread already exited (e.g. the `ConfigWatcher`
        // was dropped); nothing to do about that from inside notify's own callback.
        let _ = tx.send(event);
    })
    .map_err(|e| ConfigError::WatchSetup {
        dir: canonical_dir.clone(),
        cause: e.to_string(),
    })?;

    watcher
        .watch(&canonical_dir, RecursiveMode::NonRecursive)
        .map_err(|e| ConfigError::WatchSetup {
            dir: canonical_dir.clone(),
            cause: e.to_string(),
        })?;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);
    let alive = Arc::new(AtomicBool::new(true));
    let alive_for_thread = Arc::clone(&alive);
    let settings_path = canonical_dir.join(SETTINGS_FILE_NAME);
    let (out_tx, out_rx) = mpsc::channel::<Result<Settings, ConfigError>>();

    let handle = std::thread::Builder::new()
        .name("cerebro-config-watch".to_string())
        .spawn(move || {
            // Cleared on every way this closure can return -- see `ClearAliveOnExit`'s own doc.
            let _clear_alive_on_exit = ClearAliveOnExit(alive_for_thread);

            // The watch is `NonRecursive` on exactly one directory, so any event whose path's file
            // name matches can only be *this* directory's `settings.json`. Matching by file name,
            // rather than by whole-path equality against `settings_path`, keeps this working
            // whatever exact spelling `notify` happens to echo back for an event path -- a
            // difference from `settings_path`'s own spelling that whole-path equality would match
            // never, silently, forever.
            let wanted = std::ffi::OsStr::new(SETTINGS_FILE_NAME);
            let mut pending_since: Option<Instant> = None;
            loop {
                if stop_for_thread.load(Ordering::Relaxed) {
                    return;
                }
                match rx.recv_timeout(POLL) {
                    Ok(Ok(event)) => {
                        if event.paths.iter().any(|p| p.file_name() == Some(wanted)) {
                            pending_since = Some(Instant::now());
                        }
                    }
                    // A watch-backend error (e.g. a transient OS failure reported by notify) is
                    // treated the same as "no event yet"; it does not itself publish anything.
                    Ok(Err(_)) | Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }

                let Some(since) = pending_since else { continue };
                if since.elapsed() < DEBOUNCE {
                    continue;
                }
                pending_since = None;
                if out_tx.send(ConfigFile::load_from_disk(&settings_path)).is_err() {
                    return;
                }
            }
        })
        .map_err(|e| ConfigError::WatchSetup {
            dir: canonical_dir.clone(),
            cause: e.to_string(),
        })?;

    Ok((
        ConfigWatcher {
            _watcher: watcher,
            stop,
            alive,
            handle: Some(handle),
        },
        out_rx,
    ))
}

/// Watch `file`'s directory and keep it up to date, applying only edits that parse as a valid
/// [`Settings`] and invoking `on_change` for each one adopted.
///
/// This is the brief's `watch(dir, cb)` contract, built on top of the lower-level [`watch`]: "an
/// invalid edit never becomes `current`, and `file` keeps the last good settings" is a property of
/// this function rather than something every caller has to reconstruct by remembering to route
/// every channel item through [`ConfigFile::apply_reload`] themselves. `on_change` runs on the
/// drain thread this function spawns, never on the `notify` callback thread that [`watch`] itself
/// uses -- see [`watch`]'s "Why a channel, not a callback" section for why a panic there would be
/// fatal under this workspace's `panic = "abort"` release profile.
///
/// A panic inside `on_change` is the caller's own problem in the sense that this function does
/// nothing to paper over it -- but it is **not** contained to this drain thread alone. Unwinding
/// drops this thread's [`Receiver`], so [`watch`]'s background thread sees its next `send` fail and
/// exits too: the whole watch stops, silently, while the [`ConfigWatcher`] the caller still holds
/// gives no outward sign of that until [`ConfigWatcher::is_running`] is checked. (Under this
/// workspace's release `panic = "abort"` profile a panic here takes the whole process down instead,
/// so `is_running` matters for a debug build's test suite, not as a release-mode safety net.)
///
/// `file` is taken by value and handed back through the returned [`std::thread::JoinHandle`] once
/// the drain thread exits, so nothing else can hold it and interleave a `patch`/`apply_reload` call
/// with what this function is doing to it concurrently.
///
/// # Errors
/// [`ConfigError::WatchSetup`] under the same conditions as [`watch`].
///
/// # Shutting down
/// Drop the returned [`ConfigWatcher`] first, then join the [`std::thread::JoinHandle`]: dropping
/// the watcher stops `notify` and closes the channel, which lets the drain loop below finish and
/// return `file`. Joining the handle before dropping the watcher blocks forever, because the drain
/// loop never sees the channel close on its own.
pub fn watch_into(
    file: ConfigFile,
    mut on_change: impl FnMut(&Settings) + Send + 'static,
) -> Result<(ConfigWatcher, std::thread::JoinHandle<ConfigFile>), ConfigError> {
    let dir = file
        .path()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let (watcher, rx) = watch(dir)?;
    let pump = std::thread::Builder::new()
        .name("cerebro-config-apply".to_string())
        .spawn(move || {
            let mut file = file;
            // An `Err` here is dropped on purpose: an invalid edit never becomes `current`, and
            // `file` keeps the last good settings (`ConfigFile::apply_reload`'s own contract). The
            // loop exits once `rx` disconnects, i.e. once the paired `ConfigWatcher` is dropped.
            for result in rx {
                if file.apply_reload(result).is_ok() {
                    on_change(file.current());
                }
            }
            file
        })
        .map_err(|e| ConfigError::WatchSetup {
            dir: PathBuf::new(),
            cause: e.to_string(),
        })?;
    Ok((watcher, pump))
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::schema::full_settings_fixture;

    #[test]
    fn comment_stripping_ignores_urls_and_star_slash_inside_strings() {
        let input = "{\n  // a real comment\n  \"note\": \"see https://x and /* not a comment */ here\",\n  \"n\": 1 /* trailing */\n}\n";
        let stripped = strip_comments(input);
        let value: serde_json::Value = serde_json::from_str(&stripped).expect("valid JSON after stripping");
        assert_eq!(value["note"], "see https://x and /* not a comment */ here");
        assert_eq!(value["n"], 1);
    }

    #[test]
    fn unterminated_block_comment_does_not_panic() {
        // Malformed input must still come out as *some* string the scanner finishes producing;
        // whether the JSON parser downstream then accepts it is a separate question.
        let stripped = strip_comments("{ \"a\": 1 /* never closed");
        assert!(!stripped.contains("never closed"));
    }

    #[test]
    fn comment_stripping_preserves_byte_columns_for_non_ascii_comments() {
        // "é" is 2 bytes and "—" (em dash) is 3 bytes in UTF-8; naive one-space-per-*char*
        // blanking would shrink this comment's byte length and shift every column `serde_json`
        // reports for whatever follows it on the same line. `serde_json` reports `column` as a
        // byte offset from the start of the line, so the only correct fix is byte-width-preserving
        // blanking -- this test proves the stripped output has *exactly* that shape, by comparing
        // it against a manually byte-width-preserving replacement built without going through
        // `strip_comments` at all.
        let comment = "/* café note — trailing */";
        let line = format!("  \"a\": 1 {comment} ]]");
        let input = format!("{{\n{line}\n}}\n");

        let stripped = strip_comments(&input);

        let manually_blanked_comment = " ".repeat(comment.len());
        let expected_line = line.replacen(comment, &manually_blanked_comment, 1);
        let expected = format!("{{\n{expected_line}\n}}\n");

        assert_eq!(
            stripped, expected,
            "comment stripping must blank a multi-byte comment with the same number of BYTES of \
             whitespace, or every later column on the line shifts"
        );
    }

    #[test]
    fn an_error_never_echoes_more_than_the_cap_of_a_hostile_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        ConfigFile::open(dir.path()).expect("bootstrap");
        let key = "a".repeat(100_000);
        let doc = format!("{{\"{key}\": 1}}");
        std::fs::write(dir.path().join(SETTINGS_FILE_NAME), doc).expect("write hostile file");

        let err = ConfigFile::load_from_disk(&dir.path().join(SETTINGS_FILE_NAME)).unwrap_err();
        match err {
            ConfigError::Invalid { message, .. } => {
                assert!(
                    message.chars().count() <= MAX_ECHO_CHARS_TEST_BOUND,
                    "message of {} chars was not capped: {message}",
                    message.chars().count()
                );
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    /// Mirrors `error::MAX_ECHO_CHARS` (private to that module) plus one for the ellipsis marker,
    /// so this test does not need that constant to be `pub`.
    const MAX_ECHO_CHARS_TEST_BOUND: usize = 81;

    #[test]
    fn file_one_byte_over_the_cap_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(SETTINGS_FILE_NAME);
        let oversized = vec![b' '; (MAX_SETTINGS_FILE_BYTES + 1) as usize];
        std::fs::write(&path, &oversized).expect("write oversized file");

        let err = ConfigFile::load_from_disk(&path).unwrap_err();
        assert!(matches!(err, ConfigError::TooLarge { .. }), "got {err:?}");
    }

    #[test]
    fn too_large_reports_the_real_size_not_the_read_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(SETTINGS_FILE_NAME);
        let real_size = MAX_SETTINGS_FILE_BYTES + 4096;
        std::fs::write(&path, vec![b' '; real_size as usize]).expect("write oversized file");

        match ConfigFile::load_from_disk(&path).unwrap_err() {
            ConfigError::TooLarge { size, .. } => {
                assert_eq!(size, real_size, "must report the file's real size, not the read-cap sentinel");
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
    }

    #[test]
    fn file_of_exactly_the_cap_size_clears_the_size_check() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(SETTINGS_FILE_NAME);
        let exactly_cap = vec![b' '; MAX_SETTINGS_FILE_BYTES as usize];
        std::fs::write(&path, &exactly_cap).expect("write exactly-cap file");

        let err = ConfigFile::load_from_disk(&path).unwrap_err();
        // A file of whitespace is not valid JSON, so this must still fail -- but never with
        // `TooLarge`. Proving that "equal to the cap" is never mistaken for "over the cap" is what
        // this test is for; it is the size check moving from a `fs::metadata` snapshot (which
        // could not tell the two apart from a size alone without an extra syscall) to bytes
        // actually read via `Read::take(CAP + 1)`.
        assert!(!matches!(err, ConfigError::TooLarge { .. }), "got {err:?}");
    }

    #[test]
    fn open_bootstraps_a_default_file_when_none_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = ConfigFile::open(dir.path()).expect("bootstraps default settings.json");
        assert_eq!(file.current(), &Settings::default());
        assert!(dir.path().join(SETTINGS_FILE_NAME).is_file());
    }

    #[cfg(windows)]
    #[test]
    fn open_does_not_overwrite_a_settings_file_it_cannot_read() {
        use std::os::windows::fs::OpenOptionsExt as _;

        let dir = tempfile::tempdir().expect("tempdir");
        ConfigFile::open(dir.path()).expect("bootstrap default");
        let path = dir.path().join(SETTINGS_FILE_NAME);
        let before = std::fs::read(&path).expect("read bootstrapped file");

        // Hold the file with no sharing at all, so the concurrent open `ConfigFile::open` performs
        // fails with a sharing violation rather than `NotFound` -- exactly the "some other reason
        // the file could not be read" case `Path::is_file()` cannot be told apart from absence.
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&path)
            .expect("exclusively lock settings.json");

        let err = ConfigFile::open(dir.path()).unwrap_err();
        assert!(
            !matches!(err, ConfigError::Read { kind: std::io::ErrorKind::NotFound, .. }),
            "got {err:?}"
        );

        drop(lock);
        let after = std::fs::read(&path).expect("read settings.json after the failed open");
        assert_eq!(before, after, "open() must never overwrite a file it could not read");
    }

    #[cfg(windows)]
    #[test]
    fn a_failed_save_rolls_back_the_in_memory_settings() {
        use std::os::windows::fs::OpenOptionsExt as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("bootstrap default");
        let before = file.current().clone();
        let path = dir.path().join(SETTINGS_FILE_NAME);

        // No `FILE_SHARE_DELETE`: the rename `save()`'s `write_atomic` performs to replace this
        // file cannot succeed while this handle stays open.
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&path)
            .expect("exclusively lock settings.json");

        let err = file
            .patch(&serde_json::json!({"sampler": {"temp": 0.42}}), Source::Gui)
            .unwrap_err();
        assert!(matches!(err, ConfigError::Write { .. }), "got {err:?}");
        assert_eq!(file.current(), &before, "a failed save must roll the in-memory settings back");

        drop(lock);
    }

    #[test]
    fn a_document_from_a_future_schema_version_is_refused_and_keeps_last_good() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("bootstrap default");
        let last_good = file.current().clone();

        let mut future = full_settings_fixture();
        future.version = crate::schema::SCHEMA_VERSION + 1;
        std::fs::write(
            dir.path().join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&future).expect("serialize"),
        )
        .expect("write future-versioned document");

        let err = file.reload().unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedVersion { .. }), "got {err:?}");
        assert_eq!(file.current(), &last_good, "an unsupported version must keep the last good settings");
    }

    #[test]
    fn patch_then_save_then_reopen_round_trips_every_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("open");

        // Every schema key, set programmatically (not by hand-writing JSON), applied through the
        // real `patch` -> `save` path, then read back through a brand new `ConfigFile::open` --
        // the atomic write-temp/fsync/rename path this proves is otherwise untested.
        let fixture = full_settings_fixture();
        let patch_value = serde_json::to_value(&fixture).expect("fixture serializes to a patch");
        file.patch(&patch_value, Source::Gui).expect("apply the full fixture as a patch");
        drop(file);

        let reopened = ConfigFile::open(dir.path()).expect("reopen");
        assert_eq!(reopened.current(), &fixture);
    }

    /// This exact document is also part of the EVAL_PLAN §6.4 poisoned-config corpus: it is the
    /// canonical attempt to smuggle the removed `allow_unsafe_dacl` escape hatch back in through a
    /// hand-edited file rather than through code review.
    fn poisoned_allow_unsafe_dacl_document() -> String {
        let mut value = serde_json::to_value(Settings::default()).expect("serialize default");
        value["security"]
            .as_object_mut()
            .expect("security is an object")
            .insert("allow_unsafe_dacl".to_string(), serde_json::Value::Bool(true));
        serde_json::to_string_pretty(&value).expect("serialize poisoned document")
    }

    #[test]
    fn hand_written_allow_unsafe_dacl_file_is_rejected_and_keeps_last_good() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("bootstrap default");
        let last_good = file.current().clone();

        std::fs::write(dir.path().join(SETTINGS_FILE_NAME), poisoned_allow_unsafe_dacl_document())
            .expect("write poisoned file");

        let err = file.reload().unwrap_err();
        match &err {
            ConfigError::Invalid { message, .. } => {
                assert!(message.contains("allow_unsafe_dacl"), "message: {message}");
            }
            other => panic!("expected an unknown-field Invalid error, got {other:?}"),
        }
        assert_eq!(file.current(), &last_good, "reload must keep the last good settings on refusal");
    }

    /// Needs Developer Mode (or admin): unlike a directory junction, Windows has no privilege-free
    /// way to link a single *file*. Ignored for the same reason
    /// `directory_junction_escaping_the_root_is_rejected` (`profile.rs`) is: run explicitly on a
    /// machine where symlink creation is enabled.
    #[test]
    #[ignore = "creates a real file symlink with mklink; needs Developer Mode or admin, run explicitly on Windows"]
    fn settings_json_that_is_a_reparse_point_is_refused() {
        let target_dir = tempfile::tempdir().expect("tempdir");
        let real_target = target_dir.path().join("real-settings.json");
        std::fs::write(&real_target, serde_json::to_string_pretty(&full_settings_fixture()).expect("serialize"))
            .expect("write real target");

        let link_dir = tempfile::tempdir().expect("tempdir");
        let link = link_dir.path().join(SETTINGS_FILE_NAME);
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink"])
            .arg(&link)
            .arg(&real_target)
            .status()
            .expect("run mklink");
        assert!(status.success(), "mklink failed; needs Developer Mode or admin");

        let err = ConfigFile::load_from_disk(&link).unwrap_err();
        assert!(matches!(err, ConfigError::PathIsReparsePoint { .. }), "got {err:?}");
    }

    /// Deterministic byte-flip / truncation / type-confusion / duplicate-key mutations of a known
    /// valid document, from a seeded RNG so failures reproduce exactly.
    fn fuzz_mutate(base_json: &[u8], rng: &mut rand::rngs::StdRng, case: usize) -> Vec<u8> {
        match case % 4 {
            0 => {
                // Byte flips. Any byte in an ASCII JSON document XORed with 0xFF lands in
                // 0x80..=0xFF, which is not a valid standalone UTF-8 byte in this context, so this
                // reliably produces a non-UTF-8 file rather than a coincidentally-valid one.
                let mut bytes = base_json.to_vec();
                for _ in 0..5 {
                    if bytes.is_empty() {
                        break;
                    }
                    let idx = rng.random_range(0..bytes.len());
                    bytes[idx] ^= 0xFF;
                }
                bytes
            }
            1 => {
                // Truncation at a random point strictly before the end, which always removes at
                // least the final closing brace and so always breaks JSON structure.
                let cut = rng.random_range(0..base_json.len());
                base_json[..cut].to_vec()
            }
            2 => {
                // Wrong type at a randomly chosen leaf (not always `api.port`), driven by `rng` so
                // the corpus reaches nested keys and the hand-written `SamplerSection::Deserialize`
                // path too. Mutating the parsed `Value` (not string-replacing formatted text) means
                // this does not depend on the pretty-printer's exact spacing.
                // Deliberately no `PathBuf` field here (e.g. `/backend/dir`): a JSON string is a
                // perfectly valid `PathBuf`, so "confusing" it to a string produces a *different*,
                // still-valid document rather than a parse failure -- every leaf below is a field
                // whose deserializer genuinely rejects a bare string.
                const LEAVES: &[&str] = &[
                    "/api/port",
                    "/version",
                    "/runtime/n_ctx",
                    "/runtime/cache_type_k",
                    "/sampler/temp",
                    "/sampler/top_k",
                    "/ui/theme_contrast",
                    "/security/model_roots",
                    "/security/lan_mode",
                    "/models/roots",
                ];
                let mut value: serde_json::Value = serde_json::from_slice(base_json).expect("seed parses");
                let ptr = LEAVES[rng.random_range(0..LEAVES.len())];
                // A random suffix (not just the leaf choice) varies across cases too: every one of
                // the `LEAVES` targets expects a number, bool, or array, so *any* string -- with or
                // without a suffix -- is equally a type mismatch; the suffix exists purely so two
                // cases hitting the same leaf still produce two different documents.
                let marker: u64 = rng.random_range(0..u64::MAX);
                if let Some(slot) = value.pointer_mut(ptr) {
                    *slot = serde_json::json!(format!("not-the-right-type-{marker:x}"));
                }
                serde_json::to_vec(&value).expect("reserialize")
            }
            _ => {
                // Duplicate a randomly chosen top-level section key. serde's derived `Deserialize`
                // rejects a field seen twice at the same object level (`duplicate field` error),
                // which JSON's own grammar otherwise permits; which section that happens in should
                // vary across the corpus rather than always being `version`. Spliced into the raw
                // text, since `serde_json::Value` itself would just collapse the duplicate on parse
                // rather than preserving it for the next parse.
                const SECTIONS: &[&str] =
                    &["\"models\"", "\"backend\"", "\"runtime\"", "\"sampler\"", "\"api\"", "\"ui\"", "\"security\""];
                // The duplicate's own (unused, since the whole document is rejected before any
                // value is inspected) first value also varies across cases, so two cases splicing
                // the same section still land on two different documents.
                const DUMMY_VALUES: &[&str] = &["null", "0", "true", "\"x\""];
                let text = std::str::from_utf8(base_json).expect("seed is our own UTF-8 output");
                let key = SECTIONS[rng.random_range(0..SECTIONS.len())];
                let dummy = DUMMY_VALUES[rng.random_range(0..DUMMY_VALUES.len())];
                text.replacen(key, &format!("{key}: {dummy},\n  {key}"), 1).into_bytes()
            }
        }
    }

    #[test]
    fn fifty_seeded_fuzz_files_all_keep_the_last_good_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("bootstrap default");
        let last_good = file.current().clone();
        let seed_json = serde_json::to_vec_pretty(&last_good).expect("serialize seed");

        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC3EB_0104);
        let mut distinct_documents: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for case in 0..50 {
            let mutated = fuzz_mutate(&seed_json, &mut rng, case);
            distinct_documents.insert(mutated.clone());
            std::fs::write(dir.path().join(SETTINGS_FILE_NAME), &mutated).expect("write fuzz case");

            let err = match file.reload() {
                Ok(settings) => panic!("fuzz case {case} unexpectedly parsed as valid: {settings:?}"),
                Err(e) => e,
            };

            // "Plausible" means the position points somewhere inside *this* mutated file, which
            // `line >= 1` alone cannot show -- it holds for essentially every `serde_json` syntax
            // error, so it cannot distinguish a correct position from a wrong one. A non-`Invalid`
            // error is also a hard failure here: every one of these 50 cases must be a
            // content-level refusal, not e.g. a read or size failure sneaking in.
            let lines_in_case = mutated.iter().filter(|b| **b == b'\n').count() + 1;
            match &err {
                ConfigError::Invalid { line, column, .. } => {
                    assert!(
                        (1..=lines_in_case).contains(line),
                        "fuzz case {case}: line {line} outside 1..={lines_in_case} of a {}-byte file",
                        mutated.len()
                    );
                    assert!(
                        *column <= mutated.len(),
                        "fuzz case {case}: column {column} past the end of a {}-byte file",
                        mutated.len()
                    );
                }
                other => panic!("fuzz case {case}: expected a content-level Invalid refusal, got {other:?}"),
            }
            assert_eq!(file.current(), &last_good, "fuzz case {case} altered the last-good settings on failure");
        }

        // The corpus must actually explore 50 *different* documents, not repeat one or two
        // mutations dozens of times over -- see the RNG-driven leaf-pointer and section-key choices
        // in `fuzz_mutate`'s type-confusion and duplicate-key arms above.
        assert!(
            distinct_documents.len() >= 40,
            "only {} distinct fuzz documents in 50 cases",
            distinct_documents.len()
        );
    }

    #[test]
    fn watch_debounces_and_reports_only_valid_edits() {
        let dir = tempfile::tempdir().expect("tempdir");
        ConfigFile::open(dir.path()).expect("bootstrap default");

        let (_watcher, rx) = watch(dir.path().to_path_buf()).expect("watch starts");

        // An invalid edit must be reported as an error and never mistaken for a good reload.
        std::fs::write(dir.path().join(SETTINGS_FILE_NAME), "{ not json").expect("write invalid edit");
        let first = rx.recv_timeout(Duration::from_secs(5)).expect("callback fires for the invalid edit");
        assert!(first.is_err(), "expected the invalid edit to be reported as an error");

        // A valid edit that follows must be reported as Ok with its new content. Distinguished by
        // `ui.theme_accent` rather than `version`: only [`crate::schema::SCHEMA_VERSION`] itself is
        // now a valid `version`, so that field can no longer double as an arbitrary per-test marker.
        let mut edited = full_settings_fixture();
        edited.ui.theme_accent = "watch-debounce-marker".to_string();
        std::fs::write(
            dir.path().join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&edited).expect("serialize"),
        )
        .expect("write valid edit");
        let second = rx.recv_timeout(Duration::from_secs(5)).expect("callback fires for the valid edit");
        assert_eq!(second.expect("valid edit parses"), edited);
    }

    #[test]
    fn watch_survives_the_file_being_deleted_and_recreated() {
        let dir = tempfile::tempdir().expect("tempdir");
        ConfigFile::open(dir.path()).expect("bootstrap default");

        let (_watcher, rx) = watch(dir.path().to_path_buf()).expect("watch starts");

        std::fs::remove_file(dir.path().join(SETTINGS_FILE_NAME)).expect("delete settings file");

        let mut edited = full_settings_fixture();
        edited.ui.theme_accent = "watch-recreate-marker".to_string();
        std::fs::write(
            dir.path().join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&edited).expect("serialize"),
        )
        .expect("recreate settings file");

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(remaining > Duration::ZERO, "watch never reported the recreated file");
            match rx.recv_timeout(remaining) {
                Ok(Ok(settings)) if settings == edited => break,
                Ok(_) => continue,
                Err(_) => panic!("watch never reported the recreated file"),
            }
        }
    }

    #[test]
    fn watch_applies_only_valid_edits_to_the_config_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("bootstrap default");
        let last_good = file.current().clone();

        let (_watcher, rx) = watch(dir.path().to_path_buf()).expect("watch starts");

        std::fs::write(dir.path().join(SETTINGS_FILE_NAME), "{ not json").expect("write invalid edit");
        let result = rx.recv_timeout(Duration::from_secs(5)).expect("callback fires for the invalid edit");
        let err = file.apply_reload(result).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid { .. }), "got {err:?}");
        assert_eq!(file.current(), &last_good, "an invalid edit must not change current()");

        let mut edited = full_settings_fixture();
        edited.ui.theme_accent = "watch-apply-reload-marker".to_string();
        std::fs::write(
            dir.path().join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&edited).expect("serialize"),
        )
        .expect("write valid edit");
        let result = rx.recv_timeout(Duration::from_secs(5)).expect("callback fires for the valid edit");
        file.apply_reload(result).expect("a valid edit is adopted");
        assert_eq!(file.current(), &edited);
    }

    #[test]
    fn watch_observes_the_crate_s_own_atomic_replace_on_save() {
        // `save()` is temp-file + rename, not an in-place write: on Windows that surfaces as
        // FILE_ACTION_RENAMED_NEW_NAME, and the file-name filter in `watch` must match the rename
        // *destination*. Every other watch test above uses `fs::write`, which exercises a
        // different event shape entirely -- this is the one test that pins the crate's own write
        // path, which is the one path `ConfigFile::patch` actually takes.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut file = ConfigFile::open(dir.path()).expect("bootstrap default");
        let (_watcher, rx) = watch(dir.path().to_path_buf()).expect("watch starts");

        file.patch(&serde_json::json!({"ui": {"theme_accent": "violet"}}), Source::Gui)
            .expect("patch applies and saves");

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(remaining > Duration::ZERO, "watch never saw ConfigFile::save()'s rename");
            match rx.recv_timeout(remaining) {
                Ok(Ok(s)) if s.ui.theme_accent == "violet" => break,
                Ok(_) => continue,
                Err(_) => panic!("watch never saw ConfigFile::save()'s rename"),
            }
        }
    }

    #[test]
    fn a_panicking_callback_is_visible_through_is_running() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = ConfigFile::open(dir.path()).expect("bootstrap default");

        let (watcher, pump) = watch_into(file, |_| panic!("callback blew up")).expect("watch_into starts");
        assert!(watcher.is_running(), "a fresh watch must report itself running");

        // The first valid edit is what the callback panics on, which drops the drain thread's
        // receiver and unwinds it -- but `watch`'s background thread only *discovers* that the
        // channel is closed on its *next* attempted `send`, which needs a *further* file event.
        // So this keeps writing fresh edits (each with a distinct marker, so every write is a
        // genuine change `notify` reports) until the watch reports itself stopped, rather than
        // assuming one write is enough.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut n = 0u32;
        while watcher.is_running() {
            assert!(Instant::now() < deadline, "watch never noticed the closed channel after the callback panicked");
            n += 1;
            let mut edited = full_settings_fixture();
            edited.ui.theme_accent = format!("panic-trigger-{n}");
            std::fs::write(
                dir.path().join(SETTINGS_FILE_NAME),
                serde_json::to_string_pretty(&edited).expect("serialize"),
            )
            .expect("write edit");
            std::thread::sleep(DEBOUNCE * 2);
        }
        assert!(!watcher.is_running(), "a panicked callback must not leave the watch looking alive");
        assert!(pump.join().is_err(), "the drain thread must have unwound, not returned normally");
    }

    #[test]
    fn watch_into_never_applies_an_invalid_edit_and_keeps_the_last_good() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = ConfigFile::open(dir.path()).expect("bootstrap default");

        let seen: Arc<std::sync::Mutex<Vec<Settings>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_for_callback = Arc::clone(&seen);
        let (watcher, pump) = watch_into(file, move |settings| {
            seen_for_callback.lock().expect("lock").push(settings.clone());
        })
        .expect("watch_into starts");

        // An invalid edit must never reach the callback at all -- the brief's "applies only valid
        // edits" as a crate property, not a caller convention. Waited out past one debounce window
        // on its own (rather than immediately followed by the valid edit below) so this edit gets
        // its own reload attempt instead of being coalesced away.
        std::fs::write(dir.path().join(SETTINGS_FILE_NAME), "{ not json").expect("write invalid edit");
        std::thread::sleep(DEBOUNCE * 2);
        assert!(
            seen.lock().expect("lock").is_empty(),
            "callback fired for an edit that never parsed as valid Settings"
        );

        // Now a valid edit, which must be the only thing the callback ever sees.
        let mut edited = full_settings_fixture();
        edited.ui.theme_accent = "watch-into-marker".to_string();
        std::fs::write(
            dir.path().join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&edited).expect("serialize"),
        )
        .expect("write valid edit");

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(Instant::now() < deadline, "watch_into never observed the valid edit");
            if seen.lock().expect("lock").iter().any(|s| s.ui.theme_accent == "watch-into-marker") {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        // Shut down in the documented order: drop the watcher first so the drain loop's channel
        // closes and it can return `file`; joining first would block forever.
        drop(watcher);
        let returned = pump.join().expect("drain thread does not panic");
        assert_eq!(returned.current().ui.theme_accent, "watch-into-marker");

        let captured = seen.lock().expect("lock");
        assert_eq!(captured.len(), 1, "the invalid edit must never have reached the callback: {captured:?}");
        assert_eq!(captured[0].ui.theme_accent, "watch-into-marker");
    }

    #[test]
    fn watch_fires_for_a_non_canonical_directory_spelling() {
        let dir = tempfile::tempdir().expect("tempdir");
        ConfigFile::open(dir.path()).expect("bootstrap default");

        // `dir.path()` is already canonical (`tempfile` hands out an absolute path with no `.` /
        // `..` components); join a `.` onto it so this genuinely exercises the canonicalization
        // `watch` performs internally, rather than an already-canonical path that happens to also
        // be legal.
        let non_canonical = dir.path().join(".");
        let (_watcher, rx) = watch(non_canonical).expect("watch starts even from a non-canonical spelling");

        let mut edited = full_settings_fixture();
        edited.ui.theme_accent = "watch-non-canonical-marker".to_string();
        std::fs::write(
            dir.path().join(SETTINGS_FILE_NAME),
            serde_json::to_string_pretty(&edited).expect("serialize"),
        )
        .expect("write valid edit");

        let result = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("watch fires despite the non-canonical spelling");
        assert_eq!(result.expect("valid edit parses"), edited);
    }
}
