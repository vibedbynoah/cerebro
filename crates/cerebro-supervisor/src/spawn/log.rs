//! Append-only crash log for a spawned child, with a hard size cap and one-generation rotation.
//!
//! [`super::ring::LineRing`] answers "what did the child print right before it died" from memory,
//! cheaply, but it is gone the moment the process (or Cerebro itself) exits. This module is the
//! durable counterpart: every line the child prints is also appended to a file on disk, so a crash
//! can be diagnosed after the fact. The file is capped so a runaway or chatty child cannot fill the
//! disk; capped by *rotation* rather than by dropping lines, so the most recent history is not what
//! gets discarded once the cap is hit.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::SpawnError;

/// Multiplier applied to `max_bytes` to obtain the absolute ceiling [`RotatingLog`] enforces even
/// when rotation is permanently blocked.
///
/// Every other bound in this component (the ring's line count, a single line's byte length) is
/// infallible: eviction and truncation always succeed, so the bound always holds. Rotation is not
/// infallible — an AV scanner or backup agent can hold `.1` open indefinitely — so without a second,
/// unconditional ceiling behind it, this would be the one bound in the whole component that
/// disappears exactly when the system is already degraded (a chatty or looping child, nothing able
/// to rotate its log away). `4` is generous headroom above `max_bytes` for the ordinary case (a
/// transient rotation failure clears up well before the file has grown 4x), while still bounding the
/// pathological one.
const HARD_CAP_FACTOR: u64 = 4;

/// How long a failed rotation is left alone before the next [`RotatingLog::append_line`] tries it
/// again.
///
/// Without this, a child emitting thousands of lines a second while rotation stays blocked would
/// turn every single one of those lines into its own `drop(File)` + `exists()` + `rename()` +
/// reopen + `metadata()` attempt against the log directory — a syscall storm the blocking condition
/// (an AV scanner, most plausibly) has to fight through on top of whatever is already holding it.
const ROTATE_RETRY_BACKOFF: Duration = Duration::from_secs(5);

/// Written into the current generation, once, whenever the hard cap forces an in-place truncation
/// because rotation could not run — so a diagnostic session that finds this log short never mistakes
/// "short" for "the child said nothing", when what actually happened is "older lines were dropped
/// because rotation was blocked".
const HARD_CAP_MARKER: &str =
    "cerebro: log hard cap reached with rotation blocked; older lines in this generation were dropped";

/// Which access mode [`open_no_follow`] should open a path with.
///
/// Both variants share the same no-follow discipline (see [`open_no_follow`]'s own docs); only the
/// underlying `OpenOptions` differ.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenMode {
    /// Create if missing, otherwise append.
    Append,
    /// Create if missing, otherwise truncate to zero length. Combined with `create(true)` (not just
    /// `truncate(true)`) so a file removed out from under this module — an antivirus quarantine, a
    /// concurrent delete, both routine on Windows — is recreated rather than turning into a fatal
    /// `ERROR_FILE_NOT_FOUND`.
    Truncate,
}

/// Open `path` in the given `mode` (creating it if necessary), refusing rather than following a
/// reparse point at `path` itself.
///
/// A planted junction or symlink at exactly the path Cerebro was told to log to would otherwise let
/// a plain `OpenOptions` open silently write through it — or, in [`OpenMode::Truncate`], silently
/// *destroy the contents of* — wherever it points. That is the same redirection hazard
/// `super::verify_owned_dir` exists to catch for `path`'s *directory*, but that check never touches
/// the log file's own leaf component; `role` names which caller-supplied path is being opened, for
/// the error message. `FILE_FLAG_OPEN_REPARSE_POINT` makes `CreateFile` open the reparse point
/// object itself rather than following it, so the attribute check on the resulting handle below can
/// never be fooled by a swap that happens after some earlier lookup — there is no earlier lookup for
/// it to race.
///
/// This is the only place in this module an `OpenOptions` is constructed against a caller-supplied
/// path — [`RotatingLog::open`], [`RotatingLog::rotate`], and [`RotatingLog::truncate_in_place`] all
/// go through it, so a future generation-count or `FILE_SHARE_*` change has exactly one call site to
/// touch, and every open this module ever performs gets the identical no-follow treatment.
#[cfg(windows)]
fn open_no_follow(path: &Path, mode: OpenMode, role: &'static str) -> Result<File, SpawnError> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_OPEN_REPARSE_POINT,
    };

    let refuse_reparse = || SpawnError::UnsafeDir {
        role,
        path: path.display().to_string(),
        cause: "is a reparse point (junction or symlink), which is refused".to_string(),
    };

    // A pre-open, no-follow lookup classifies an existing reparse point before anything is
    // opened — the same two-step discipline `preflight::open_no_follow` uses for the model and
    // API-key files. It matters here for the same reason it matters there: a directory junction
    // planted at `path` carries `FILE_ATTRIBUTE_DIRECTORY`, and `CreateFileW` on a directory
    // without `FILE_FLAG_BACKUP_SEMANTICS` (which this function, opening what must be a plain file,
    // never sets) fails with `ERROR_ACCESS_DENIED` regardless of the reparse-point flag below —
    // without this pre-check that would surface as an opaque I/O error instead of the precise
    // hazard this function exists to name. A path that does not exist yet (the ordinary first-launch
    // case, and the ordinary case after `Truncate` just vacated it) is not refused here; `create`
    // handles it.
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(refuse_reparse());
        }
    }

    let mut opts = OpenOptions::new();
    match mode {
        OpenMode::Append => {
            opts.create(true).append(true);
        }
        OpenMode::Truncate => {
            opts.create(true).write(true).truncate(true);
        }
    }
    let file = opts
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(|e| SpawnError::LogOpen {
            path: path.display().to_string(),
            cause: e.to_string(),
        })?;

    // Closes the gap between the lookup above and this open: if `path` was swapped for a reparse
    // point in that instant, the handle's own attributes still catch it.
    let meta = file.metadata().map_err(|e| SpawnError::LogOpen {
        path: path.display().to_string(),
        cause: e.to_string(),
    })?;
    if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(refuse_reparse());
    }
    Ok(file)
}

/// [`open_no_follow`] in [`OpenMode::Append`], for [`RotatingLog::open`] and every reopen after a
/// close.
#[cfg(windows)]
fn open_append_no_follow(path: &Path) -> Result<File, SpawnError> {
    open_no_follow(path, OpenMode::Append, "crash log file")
}

/// [`open_no_follow`] in [`OpenMode::Truncate`], for [`RotatingLog::truncate_in_place`] alone.
#[cfg(windows)]
fn open_truncate_no_follow(path: &Path) -> Result<File, SpawnError> {
    open_no_follow(path, OpenMode::Truncate, "crash log file")
}

/// Non-Windows fallback: no reparse-point concept to guard against here, so this is a plain append
/// open.
#[cfg(not(windows))]
fn open_append_no_follow(path: &Path) -> Result<File, SpawnError> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| SpawnError::LogOpen {
            path: path.display().to_string(),
            cause: e.to_string(),
        })
}

/// Non-Windows fallback: no reparse-point concept to guard against here, so this is a plain
/// create-or-truncate open.
#[cfg(not(windows))]
fn open_truncate_no_follow(path: &Path) -> Result<File, SpawnError> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| SpawnError::LogOpen {
            path: path.display().to_string(),
            cause: e.to_string(),
        })
}

/// An append-only log file that rotates to a single `.1` generation once it would exceed
/// `max_bytes`, rather than growing without bound or silently dropping lines.
///
/// "One generation" is deliberate, not a shortcut: this is a live-crash-diagnostics log meant to be
/// read once, right after an unexpected exit, not a long-term audit trail — `cerebro-supervisor`
/// (the caller, not this file) is where any longer retention policy belongs, keyed to a specific
/// launch rather than to this one file's byte count.
pub(super) struct RotatingLog {
    path: PathBuf,
    // `None` only ever exists transiently inside `rotate`, between closing the old handle (which
    // Windows requires before the handle's file can be renamed) and opening the new one. If
    // `rotate` returns early on error, it can be observed as `None` afterward too; `append_line`
    // reports that as a typed error rather than assuming a `File` is always present.
    file: Option<File>,
    written: u64,
    max_bytes: u64,
    /// Absolute ceiling, `max_bytes.saturating_mul(HARD_CAP_FACTOR)` in production — see that
    /// constant's own docs for why a second, unconditional bound exists behind `max_bytes` at all.
    hard_cap: u64,
    /// `None` means the next [`RotatingLog::append_line`] may attempt a rotation if one is due;
    /// `Some(t)` means a rotation attempt already failed and the next retry is not due until `t` —
    /// see [`ROTATE_RETRY_BACKOFF`]'s own docs for why this exists at all.
    next_rotate_attempt: Option<Instant>,
    /// How long a failed rotation is left alone before being retried. Kept as a field rather than
    /// always reading [`ROTATE_RETRY_BACKOFF`] directly, so a test can inject a backoff short enough
    /// to observe the "retried on a timer, not every line" behavior without a multi-second sleep —
    /// the same reason [`cerebro_probe::child_env::build_child_env`] takes an injected parent-environment iterator
    /// instead of reading `std::env::vars_os()` directly.
    retry_backoff: Duration,
}

impl RotatingLog {
    /// Open (creating if necessary) the log file at `path`, appending to whatever is already there.
    ///
    /// `max_bytes` bounds the *current* generation only; once open, an existing file already over
    /// that size is left alone until the next line would push it further over, at which point it
    /// rotates immediately rather than growing first — see [`RotatingLog::append_line`].
    pub(super) fn open(path: PathBuf, max_bytes: u64) -> Result<Self, SpawnError> {
        Self::open_with_backoff(path, max_bytes, ROTATE_RETRY_BACKOFF)
    }

    /// [`RotatingLog::open`], with the rotation-retry backoff injectable for tests — see
    /// [`RotatingLog::retry_backoff`]'s own docs.
    fn open_with_backoff(path: PathBuf, max_bytes: u64, retry_backoff: Duration) -> Result<Self, SpawnError> {
        let file = open_append_no_follow(&path)?;
        let written = file
            .metadata()
            .map_err(|e| SpawnError::LogOpen {
                path: path.display().to_string(),
                cause: e.to_string(),
            })?
            .len();
        let max_bytes = max_bytes.max(1);
        Ok(Self {
            path,
            file: Some(file),
            written,
            max_bytes,
            hard_cap: max_bytes.saturating_mul(HARD_CAP_FACTOR),
            next_rotate_attempt: None,
            retry_backoff,
        })
    }

    /// Append `line` plus a trailing `\n`, rotating first if it would not otherwise fit.
    ///
    /// The size check runs *before* the write, not after: checking afterward would let exactly one
    /// line push the file arbitrarily far past `max_bytes` (a line can itself be up to
    /// [`super::ring::read_bounded_lines`]'s own per-line cap in size), so the file could never
    /// actually be bounded by this cap, only nudged toward it.
    ///
    /// A failed rotation is reported *after* the line is written, never instead of writing it: this
    /// is a durable crash-diagnostics log, and a transient rotation failure (an AV scanner or backup
    /// agent holding the `.1` generation open, both routine on Windows — see [`RotatingLog::rotate`])
    /// must cost only the one-line overshoot of the cap that not rotating first implies, never the
    /// line itself. An earlier version of this function propagated the rotation error with `?`
    /// before the write ran at all, which meant every line submitted while a rotation stayed blocked
    /// was silently lost — for as long as the block persisted, the durable half of this log recorded
    /// nothing, which is exactly the failure this module's rotation-recovery exists to prevent.
    ///
    /// Once a rotation attempt fails, further attempts back off on a timer (see
    /// [`ROTATE_RETRY_BACKOFF`]) instead of retrying on every subsequent line — but the *size* bound
    /// on the file is never allowed to depend on that timer, or on rotation succeeding at all: once
    /// appending would push this generation past the hard cap (see [`HARD_CAP_FACTOR`]), the current
    /// generation is truncated in place first (see [`RotatingLog::truncate_in_place`]), independent
    /// of whether a rotation was even attempted for this line.
    pub(super) fn append_line(&mut self, line: &str) -> Result<(), SpawnError> {
        let candidate_len = line.len() as u64 + 1; // +1 for the '\n'

        let over_soft_cap = self.written > 0 && self.written + candidate_len > self.max_bytes;
        let rotation_is_due = self
            .next_rotate_attempt
            .is_none_or(|next| Instant::now() >= next);

        let mut rotate_failed = None;
        if over_soft_cap && rotation_is_due {
            match self.rotate() {
                Ok(()) => self.next_rotate_attempt = None,
                Err(e) => {
                    self.next_rotate_attempt = Some(Instant::now() + self.retry_backoff);
                    rotate_failed = Some(e);
                }
            }
        }

        // The hard cap is enforced unconditionally, whether or not a rotation was even attempted
        // above (it is skipped entirely while a previous failure's backoff is still in effect) —
        // see this function's own docs and `HARD_CAP_FACTOR`'s on why this bound must never depend
        // on rotation succeeding. Neither the truncation nor the marker write is allowed to use `?`
        // here: either one failing (a concurrent delete racing the recreate, an AV scanner holding
        // the handle) must cost only the marker or the cap bound, never the line below — the same
        // "the line is never lost, only the cap overshoots" contract `rotate_failed` already gets.
        // An earlier version propagated `truncate_in_place`'s error with `?` at exactly this point,
        // which returned before `line` was ever written — see `truncate_in_place`'s own docs.
        let mut cap_failed = None;
        if self.written > 0 && self.written + candidate_len > self.hard_cap {
            match self.truncate_in_place() {
                Ok(()) => {
                    if let Err(e) = self.write_raw(HARD_CAP_MARKER) {
                        cap_failed = Some(e);
                    }
                }
                Err(e) => cap_failed = Some(e),
            }
        }

        self.write_raw(line)?;

        match rotate_failed.or(cap_failed) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Write one line plus a trailing `\n` to the current generation and account its length in
    /// [`RotatingLog::written`]. The one place both [`RotatingLog::append_line`] and
    /// [`RotatingLog::truncate_in_place`]'s marker write go through, so the two can never drift on
    /// how a write updates `written`.
    fn write_raw(&mut self, line: &str) -> Result<(), SpawnError> {
        let file = self.file.as_mut().ok_or_else(|| SpawnError::LogWrite {
            path: self.path.display().to_string(),
            cause: "no open log handle; a previous rotation did not complete".to_string(),
        })?;
        file.write_all(line.as_bytes())
            .and_then(|()| file.write_all(b"\n"))
            .map_err(|e| SpawnError::LogWrite {
                path: self.path.display().to_string(),
                cause: e.to_string(),
            })?;
        self.written += line.len() as u64 + 1;
        Ok(())
    }

    /// Reopen `self.path` through [`open_append_no_follow`] and restore [`RotatingLog::written`]
    /// from the reopened file's real length — the shared last act of both
    /// [`RotatingLog::rotate`] and [`RotatingLog::truncate_in_place`], so the invariant they both
    /// owe ("every path out of this function, success or failure, leaves `self.file` holding a live
    /// handle") is a property of this one function, not of prose repeated in two doc comments that
    /// can drift apart. Goes through the same no-follow open as [`RotatingLog::open`]: whatever
    /// vacated `self.path` just before this call (a successful rename, a successful truncate) leaves
    /// a reparse point planted in that instant exactly as much a redirection hazard as it is on
    /// first open.
    ///
    /// If this reopen itself fails, `self.file` is left `None` — there is no third handle left to
    /// fall back to once the one this module just closed cannot be reopened at all. That residual
    /// case is the same one [`RotatingLog::rotate`] always had; [`RotatingLog::write_raw`] reports it
    /// as a typed error rather than panicking.
    fn reopen_current(&mut self) -> Result<(), SpawnError> {
        let fresh = open_append_no_follow(&self.path)?;
        self.written = fresh
            .metadata()
            .map_err(|e| SpawnError::LogOpen {
                path: self.path.display().to_string(),
                cause: e.to_string(),
            })?
            .len();
        self.file = Some(fresh);
        Ok(())
    }

    /// Discard the current generation's contents in place, leaving `path` open and empty, so
    /// [`RotatingLog::append_line`]'s hard cap is an unconditional bound even when rotation cannot
    /// run at all.
    ///
    /// Every path out of this function, success or failure, leaves `self.file` holding a live handle
    /// — exactly [`RotatingLog::rotate`]'s own contract — because [`RotatingLog::append_line`] must
    /// still write the line that triggered the cap even when the truncating open itself fails (a
    /// concurrent delete or an antivirus quarantine racing this call, both routine on Windows): the
    /// truncating open's outcome is captured as a plain `Result`, never propagated with `?`, and
    /// [`RotatingLog::reopen_current`] always runs regardless of it.
    ///
    /// Reopens rather than calling `set_len(0)` on the existing append-mode handle: an append-mode
    /// handle's writes always target the current end-of-file on Windows regardless of any seek, and
    /// relying on `set_len` to also relocate that end-of-file marker back to zero is not a guarantee
    /// worth trusting without dedicated proof neither this function nor its test has — reopening the
    /// file is unambiguous instead.
    fn truncate_in_place(&mut self) -> Result<(), SpawnError> {
        drop(self.file.take());
        // Captured, not propagated: `reopen_current` below must run either way, so the line that
        // triggered this truncation is never lost merely because clearing the old generation failed.
        let truncated = open_truncate_no_follow(&self.path).map(drop);
        self.reopen_current()?;
        truncated
    }

    /// Move the current file to `<path>.1` (replacing any previous `.1`) and start a fresh, empty
    /// file at `path` — or, if the move itself fails, reopen the *same* file that was there before,
    /// so a rotation failure costs this one overshoot of the cap rather than disabling the durable
    /// log for the rest of this child's life.
    ///
    /// A `.1` generation held open by an antivirus scanner or a backup agent — both routine on
    /// Windows — is exactly the kind of transient failure this guards against: without the
    /// unconditional reopen below, `self.file` would be left `None` forever (see that field's own
    /// doc comment), `append_line` would fail every call from then on, and nothing would ever know
    /// the durable copy had gone silent. This function's own `Result` still reports that failure —
    /// [`RotatingLog::append_line`] is what decides *when* to surface it, and does so only after the
    /// line that triggered the rotation has already been written, so the cap overshoot this comment
    /// promises is the entire cost, never a lost line on top of it.
    fn rotate(&mut self) -> Result<(), SpawnError> {
        let rotated = rotated_path(&self.path);
        // Close the current handle before renaming: unlike POSIX, Windows refuses to rename a file
        // that is still open, so this is a real branch to get right rather than a formality.
        drop(self.file.take());

        // The rename outcome is this function's `Result`, but it must not gate the reopen below —
        // every path out of this function, success or failure, has to leave `self.file` holding a
        // live handle to *some* file at `self.path` (see `reopen_current`).
        let renamed: Result<(), SpawnError> = (|| {
            if rotated.exists() {
                std::fs::remove_file(&rotated).map_err(|e| SpawnError::LogRotate {
                    path: self.path.display().to_string(),
                    cause: e.to_string(),
                })?;
            }
            std::fs::rename(&self.path, &rotated).map_err(|e| SpawnError::LogRotate {
                path: self.path.display().to_string(),
                cause: e.to_string(),
            })
        })();

        self.reopen_current()?;
        renamed
    }
}

/// The single rotated-generation path for `path`: `foo.log` -> `foo.log.1`.
fn rotated_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".1");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempFile(PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let rotated = rotated_path(&self.0);
            let _ = std::fs::remove_file(&rotated);
            // `a_blocked_rotation_still_writes_the_line` and
            // `a_sustained_rotation_block_does_not_silently_disable_the_log` deliberately pre-create
            // the rotated path as a *directory* to force `rotate`'s rename to fail; clean that up
            // too.
            let _ = std::fs::remove_dir_all(&rotated);
        }
    }

    fn temp_path(tag: &str) -> TempFile {
        // Deliberately not `{:?}` of `SystemTime`: its `Debug` form (`SystemTime { intervals: .. }`
        // on this platform) contains a `:`, which Windows parses as an NTFS alternate-data-stream
        // separator rather than a plain character in the file name. `CreateFile` (what `open`
        // resolves to) happily creates that stream, but `rename`/`MoveFileEx` — what
        // `RotatingLog::rotate` calls — refuses to rename a stream and fails with
        // `ERROR_INVALID_PARAMETER`, breaking the rotation tests below with no connection to
        // `RotatingLog`'s own logic. Nanoseconds since the epoch are unique enough for a per-test
        // fixture name without that hazard.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        TempFile(std::env::temp_dir().join(format!(
            "cerebro-supervisor-log-{tag}-{}-{nanos}.log",
            std::process::id(),
        )))
    }

    #[test]
    fn lines_below_the_cap_accumulate_in_one_file() {
        let guard = temp_path("no-rotate");
        let mut log = RotatingLog::open(guard.0.clone(), 1024).expect("open log");
        log.append_line("one").expect("append");
        log.append_line("two").expect("append");
        let contents = std::fs::read_to_string(&guard.0).expect("read log");
        assert_eq!(contents, "one\ntwo\n");
        assert!(!rotated_path(&guard.0).exists());
    }

    #[test]
    fn a_line_that_would_exceed_the_cap_triggers_rotation_first() {
        let guard = temp_path("rotate");
        // Cap small enough that the second line cannot fit alongside the first.
        let mut log = RotatingLog::open(guard.0.clone(), 8).expect("open log");
        log.append_line("aaaaaa").expect("append"); // 7 bytes written, under the cap.
        log.append_line("bbbbbb").expect("append"); // Would make 14 > 8: must rotate first.

        let rotated = rotated_path(&guard.0);
        assert!(rotated.exists(), "expected a rotated generation to exist");
        assert_eq!(std::fs::read_to_string(&rotated).expect("read rotated"), "aaaaaa\n");
        assert_eq!(std::fs::read_to_string(&guard.0).expect("read current"), "bbbbbb\n");
    }

    #[test]
    fn rotation_keeps_only_one_generation() {
        let guard = temp_path("rotate-twice");
        let mut log = RotatingLog::open(guard.0.clone(), 8).expect("open log");
        log.append_line("aaaaaa").expect("append");
        log.append_line("bbbbbb").expect("append"); // rotates once: .1 = "aaaaaa"
        log.append_line("cccccc").expect("append"); // rotates again: .1 = "bbbbbb", "aaaaaa" is gone

        let rotated = rotated_path(&guard.0);
        assert_eq!(std::fs::read_to_string(&rotated).expect("read rotated"), "bbbbbb\n");
        assert_eq!(std::fs::read_to_string(&guard.0).expect("read current"), "cccccc\n");
    }

    #[test]
    fn a_blocked_rotation_still_writes_the_line() {
        // Pre-creating the `.1` generation as a *directory* forces `std::fs::remove_file` inside
        // `rotate` to fail without any platform-specific share-mode trick — this is the "a `.1`
        // held open by an AV scanner" scenario, reproduced deterministically.
        let guard = temp_path("rotate-failure");
        let rotated = rotated_path(&guard.0);
        std::fs::create_dir(&rotated).expect("create a blocking directory at the rotated path");

        // A zero backoff keeps this test's per-line semantics exactly as before the retry-backoff
        // was added: every over-cap line attempts (and, while `.1` is blocked, fails) its own
        // rotation, which is what lets this test assert on each call individually rather than only
        // on the first. `a_blocked_rotation_is_retried_on_a_timer_not_every_line` below is what
        // actually pins down the backoff itself.
        let mut log = RotatingLog::open_with_backoff(guard.0.clone(), 8, Duration::ZERO).expect("open log");
        log.append_line("aaaaaa").expect("append under the cap"); // 7 bytes written.
        let err = log
            .append_line("bbbbbb")
            .expect_err("rotation must fail while `.1` is a directory, not a file");
        assert!(matches!(err, SpawnError::LogRotate { .. }), "{err:?}");

        // The line that triggered the failed rotation must still be on disk: a blocked rotation
        // costs the documented one-line cap overshoot, never the line itself. An earlier version of
        // `append_line` propagated the rotation error with `?` *before* writing, which silently
        // dropped exactly this line — this assertion is what would have caught that.
        assert_eq!(
            std::fs::read_to_string(&guard.0).expect("read log"),
            "aaaaaa\nbbbbbb\n",
            "a blocked rotation must not cost the line that triggered it"
        );

        // A second line submitted while still blocked must also survive — the earlier bug did not
        // stop at losing one line; every line submitted for as long as the block persisted was lost.
        let err = log.append_line("dddddd").expect_err("rotation is still blocked");
        assert!(matches!(err, SpawnError::LogRotate { .. }), "{err:?}");
        assert_eq!(
            std::fs::read_to_string(&guard.0).expect("read log"),
            "aaaaaa\nbbbbbb\ndddddd\n",
        );

        // The log must not be permanently disabled by the failed rotations either: `append_line`
        // reopened the original file across each failure (its bytes are still there, `written`
        // restored from its real length), so once whatever was blocking `.1` clears up — an AV
        // scanner releasing its handle, here simulated by removing the blocking directory — the
        // very next rotation succeeds and appending resumes landing in a fresh `path`.
        std::fs::remove_dir(&rotated).expect("clear the simulated block on the rotated path");
        log.append_line("cccccc").expect("log must remain writable once rotation can succeed again");
        let contents = std::fs::read_to_string(&guard.0).expect("read log");
        assert_eq!(contents, "cccccc\n");
        assert_eq!(
            std::fs::read_to_string(&rotated).expect("read rotated generation"),
            "aaaaaa\nbbbbbb\ndddddd\n",
            "every line written while blocked must have made it into the rotated generation once \
             rotation succeeded"
        );
    }

    #[test]
    fn a_permanently_blocked_rotation_still_bounds_the_file() {
        // Unlike `a_blocked_rotation_still_writes_the_line` above (which clears the block after a
        // couple of lines), this leaves `.1` blocked for hundreds of consecutive appends — long
        // enough that, without a hard cap independent of rotation ever succeeding, the file would
        // grow without bound (that actually was this test's assertion before the hard cap existed;
        // see ROUND 4 finding on `log.rs`). `max_bytes` of 64 (so `hard_cap` is 256) leaves the
        // ~100-byte `HARD_CAP_MARKER` comfortable room, unlike the 8-byte cap the other rotation
        // tests in this file use for unrelated reasons — a marker that itself approached the cap
        // would make this test about the marker's own size, not about the bound.
        let guard = temp_path("rotate-sustained-block");
        let rotated = rotated_path(&guard.0);
        std::fs::create_dir(&rotated).expect("create a blocking directory at the rotated path");

        // Zero backoff: this test's point is the *size* bound holding under a rotation that never
        // succeeds, not the retry timing (that is `a_blocked_rotation_is_retried_on_a_timer_not_every_line`
        // below) — a nonzero backoff would just make most of these 500 attempts skip rotation
        // without changing whether the hard cap holds.
        let mut log =
            RotatingLog::open_with_backoff(guard.0.clone(), 64, Duration::ZERO).expect("open log");
        let hard_cap = 64 * HARD_CAP_FACTOR;

        log.append_line("seed").expect("the first append on an empty file never rotates");
        // Not every one of these 500 appends is even over the *soft* cap yet at the point it runs
        // (the file starts small, so the earliest few succeed outright) — the property this loop
        // pins down is not "every call errors" (that is `a_permanently_blocked_rotation...`'s sibling
        // test at the original 8-byte cap, where every line after the first is already over cap) but
        // that the file's size never exceeds `hard_cap`, for as long as rotation stays blocked, and
        // that at least one rotation attempt genuinely failed along the way (proving the block was
        // real, not merely never hit).
        let mut saw_a_rotation_failure = false;
        for i in 0..500 {
            let line = format!("l{i:04}");
            if matches!(log.append_line(&line), Err(SpawnError::LogRotate { .. })) {
                saw_a_rotation_failure = true;
            }

            // The bound is enforced on *every* append, not just checked once at the end: a version
            // that only truncated lazily, or that let one write overshoot before catching up, would
            // still pass an end-of-test-only assertion.
            let len = std::fs::metadata(&guard.0).expect("stat log").len();
            assert!(len <= hard_cap, "line {i}: log grew to {len} bytes, past hard_cap {hard_cap}");
        }
        assert!(
            saw_a_rotation_failure,
            "expected at least one rotation attempt to genuinely fail while `.1` was blocked"
        );

        let contents = std::fs::read_to_string(&guard.0).expect("read log");
        // The newest line is what a crash investigator needs most, and it must never be among what
        // a truncation drops — `truncate_in_place` runs (and drops old bytes) *before* the line that
        // triggered it is written, never after.
        assert!(contents.contains("l0499"), "{contents:?}");
        assert!(
            contents.contains(HARD_CAP_MARKER),
            "expected at least one hard-cap marker once the file was truncated in place: {contents:?}"
        );
        // The seed line and the earliest blocked-rotation lines are exactly what the hard cap is
        // supposed to be free to drop once rotation can never reclaim their space.
        assert!(!contents.contains("seed"), "{contents:?}");
    }

    #[test]
    fn a_blocked_rotation_is_retried_on_a_timer_not_every_line() {
        // A short, injected backoff (see `RotatingLog::open_with_backoff`'s own docs) makes this
        // deterministic without a multi-second real sleep: the property under test is "not every
        // line", which only needs two attempts close together and one further apart, not the
        // production 5-second value itself.
        let backoff = Duration::from_millis(200);
        let guard = temp_path("rotate-backoff");
        let rotated = rotated_path(&guard.0);
        std::fs::create_dir(&rotated).expect("create a blocking directory at the rotated path");

        let mut log =
            RotatingLog::open_with_backoff(guard.0.clone(), 8, backoff).expect("open log");
        log.append_line("aaaaaa").expect("first append on an empty file never rotates");

        // First over-cap line: no previous failure to back off from, so this attempts (and, with
        // `.1` blocked, fails) a real rotation.
        let err = log.append_line("bbbbbb").expect_err("first rotation attempt must fail");
        assert!(matches!(err, SpawnError::LogRotate { .. }), "{err:?}");

        // Immediately after: still well inside `backoff`, so this must skip the rotation attempt
        // entirely and simply succeed — the whole point of the backoff. A version with no backoff at
        // all (the pre-fix behavior) would return `Err` here too, which is exactly what this
        // assertion would have caught.
        let result = log.append_line("cccccc");
        assert!(
            result.is_ok(),
            "a line submitted immediately after a failed rotation must not retry it: {result:?}"
        );

        // Once `backoff` has actually elapsed, the next over-cap line retries and, since `.1` is
        // still blocked, fails again.
        std::thread::sleep(backoff + Duration::from_millis(50));
        let err = log.append_line("dddddd").expect_err("rotation must be retried once backoff elapses");
        assert!(matches!(err, SpawnError::LogRotate { .. }), "{err:?}");

        // Nothing submitted across any of this was ever lost, backoff-skipped or not.
        let contents = std::fs::read_to_string(&guard.0).expect("read log");
        for line in ["aaaaaa", "bbbbbb", "cccccc", "dddddd"] {
            assert!(contents.contains(line), "{line:?} missing from {contents:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_log_path_that_is_a_reparse_point_is_refused() {
        // A directory junction needs no special privilege to create on Windows, unlike a file
        // symlink — see `spawn`'s own reparse-point tests for the same reasoning. Planting the
        // junction *at* the log path itself (rather than at an ancestor, which
        // `super::verify_owned_dir` already covers for the log's parent directory) is the leaf
        // hazard `open_append_no_follow` exists to catch.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-supervisor-log-junction-root-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create junction test root");
        let target = root.join("real");
        std::fs::create_dir(&target).expect("create junction target");
        let log_path = root.join("planted-log");

        // Through the probe's own DeviceIoControl helper, not a script host: no quoting of a temp
        // path into a command string, and no window.
        cerebro_probe::backend::create_junction_for_tests(&log_path, &target)
            .expect("creating a junction needs no special privilege");

        let err = match RotatingLog::open(log_path.clone(), 1024) {
            Ok(_) => panic!("a log path that is itself a reparse point must be refused"),
            Err(e) => e,
        };
        assert!(matches!(err, SpawnError::UnsafeDir { .. }), "{err:?}");

        let _ = std::fs::remove_dir(&log_path);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn reopening_an_existing_file_resumes_appending_rather_than_truncating() {
        let guard = temp_path("resume");
        std::fs::write(&guard.0, "existing\n").expect("seed file");
        let mut log = RotatingLog::open(guard.0.clone(), 1024).expect("reopen log");
        log.append_line("new").expect("append");
        assert_eq!(
            std::fs::read_to_string(&guard.0).expect("read log"),
            "existing\nnew\n"
        );
    }

    /// Regression test for the bug an earlier version of `truncate_in_place` had: it propagated the
    /// truncating open's error with `?` *before* `reopen_current` ran and before `append_line`'s own
    /// `write_raw(line)` call, so a failed truncation both left `self.file` at `None` forever (every
    /// later `append_line` then failing with "no open log handle") and silently dropped the very line
    /// that triggered it.
    ///
    /// Denies `WriteData` (but not `AppendData`) on the log file with an explicit DACL, the shape of a genuine
    /// append-only ACL: Rust's `.append(true)` requests `FILE_GENERIC_WRITE & !FILE_WRITE_DATA` on
    /// Windows, while `.write(true).truncate(true)` requests the full `FILE_GENERIC_WRITE` — so this
    /// makes the *truncating* open fail with access denied while the *append* reopen still succeeds,
    /// which is exactly the split `truncate_in_place` has to survive. Merely deleting the file, the
    /// obvious alternative, no longer reproduces anything once the truncating open carries
    /// `.create(true)` (see the sibling fix in `open_no_follow`) — a real permission denial is needed
    /// instead.
    #[cfg(windows)]
    #[test]
    fn a_failed_truncation_leaves_the_log_writable_and_keeps_the_line() {
        let guard = temp_path("truncate-failure");
        let rotated = rotated_path(&guard.0);
        // Also block rotation (the same technique the sibling rotation-failure tests above use) so
        // the hard cap, not a successful rotation, is what this over-cap line actually hits.
        std::fs::create_dir(&rotated).expect("create a blocking directory at the rotated path");

        let mut log = RotatingLog::open_with_backoff(guard.0.clone(), 8, Duration::ZERO).expect("open log");
        log.append_line("aaaaaa").expect("append under the cap"); // 7 bytes written.

        // Staged by SID through Win32 (never `icacls` by user name, which prints into every test
        // run and can be ambiguous on a domain account): deny WriteData, allow everything else.
        let me = cerebro_probe::backend::current_user_sid_for_tests()
            .expect("the process token must be queryable");
        cerebro_probe::backend::set_dacl_from_sddl_for_tests(
            &guard.0,
            &format!("D:(D;;WD;;;{me})(A;;FA;;;{me})"),
        )
        .expect("deny WriteData on the log file");

        // Comfortably past both the soft cap (8) and the hard cap (32): `.1` is still a blocking
        // directory, so rotation fails too, and the truncating open now fails on the denied ACE.
        let line = "b".repeat(40);
        let err = log
            .append_line(&line)
            .expect_err("both the blocked rotation and the denied truncating open must surface");
        assert!(
            matches!(
                err,
                SpawnError::LogRotate { .. } | SpawnError::LogWrite { .. } | SpawnError::LogOpen { .. }
            ),
            "{err:?}"
        );

        // The line that triggered the failed truncation must still be on disk: this is the assertion
        // the pre-fix `?`-propagating version of `truncate_in_place` would have failed.
        let contents = std::fs::read_to_string(&guard.0).expect("read log despite the denied ACE");
        assert!(contents.contains(&line), "{contents:?}");

        // Clear both blocks and prove the log was not left permanently disabled: the very next line
        // must be appended without error, not rejected with "no open log handle".
        cerebro_probe::backend::set_dacl_from_sddl_for_tests(&guard.0, &format!("D:(A;;FA;;;{me})"))
            .expect("clear the denial");
        std::fs::remove_dir(&rotated).expect("clear the simulated rotation block");

        log.append_line("cccccc")
            .expect("the log must remain writable once both blocks are cleared");
    }

    /// Sibling to `a_log_path_that_is_a_reparse_point_is_refused` above, but covering the *truncate*
    /// open added for the hard cap rather than the first append open: a symlink planted at the log
    /// path in between opens must be refused, not followed and truncated in place — the worst-case
    /// version of the redirection hazard `open_no_follow` exists to catch, since a followed truncate
    /// destroys the target's contents rather than merely misdirecting a write.
    ///
    /// `#[ignore]`d because creating a *file* symlink (unlike a directory junction) needs either the
    /// `SeCreateSymbolicLink` privilege or Developer Mode enabled — not available on every machine
    /// this crate's tests run on.
    #[cfg(windows)]
    #[test]
    #[ignore = "creating a file symlink needs SeCreateSymbolicLink privilege or Developer Mode \
                enabled; run manually on a machine that has one of those"]
    fn a_log_path_swapped_for_a_symlink_before_truncation_is_refused_not_followed() {
        let guard = temp_path("truncate-symlink");
        let target = guard.0.with_extension("target");
        std::fs::write(&target, "do not touch\n").expect("seed the symlink target");

        // Open normally first (an ordinary file, not yet a symlink) so a `RotatingLog` exists with a
        // live handle, then swap the path itself for a symlink before forcing a truncation.
        let mut log = RotatingLog::open_with_backoff(guard.0.clone(), 1, Duration::ZERO)
            .expect("open the log before it is swapped for a symlink");
        drop(log.file.take());
        std::fs::remove_file(&guard.0).expect("remove the plain file before planting the symlink");
        std::os::windows::fs::symlink_file(&target, &guard.0)
            .expect("create a file symlink (requires SeCreateSymbolicLink or Developer Mode)");

        let err = log
            .truncate_in_place()
            .expect_err("a log path swapped for a symlink must be refused, not truncated through");
        assert!(matches!(err, SpawnError::UnsafeDir { .. }), "{err:?}");

        let target_contents = std::fs::read_to_string(&target).expect("read the symlink target");
        assert_eq!(target_contents, "do not touch\n", "the symlink target must be untouched");

        let _ = std::fs::remove_file(&guard.0);
        let _ = std::fs::remove_file(&target);
    }
}
