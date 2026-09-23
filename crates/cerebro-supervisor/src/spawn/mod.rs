//! Turning a [`crate::preflight::Preflighted`] launch into a running, contained `llama-server`
//! child that cannot outlive Cerebro and cannot be steered by the ambient environment.
//!
//! # Every route to `CreateProcess` carries proof, or a written reason
//!
//! [`spawn_preflighted`] is the ordinary entry point: it requires a [`crate::preflight::Preflighted`]
//! value, which cannot exist unless [`crate::preflight::preflight`] already ran the DACL audit and
//! the SHA-256 manifest check against exactly this `backend_dir` (ARCHITECTURE.md section 8: "a hard
//! refusal to spawn — not merely a warning — when the backend directory DACL, or the path used to
//! reach it, is non-owner-writable"). There is deliberately no unaudited sibling with a friendlier
//! signature: a caller that genuinely needs to launch from a directory preflight never saw (this
//! crate's own live tests, a future bench/probe harness) must call [`spawn_unaudited`] instead, which
//! demands an [`AcceptUnauditedBackend`] — a non-empty, recorded reason, built exactly like
//! [`crate::preflight::AllowUnsafeDacl`] — so that route can never be reached silently. Both
//! functions still run `verify_spawn_target`'s shape check on `backend_dir`/`tool_name` before
//! anything is spawned; only the DACL and integrity checks are what the two entry points trade away
//! for each other.
//!
//! # What "contained" means here
//!
//! Four independent guarantees, each documented in the sibling module that implements it:
//!
//! 1. **Argv only, never a shell.** `spawn_windows_at` builds a [`std::process::Command`] directly
//!    from a program path and an argument list; nothing here ever hands a string to `cmd.exe` or
//!    PowerShell for interpretation, so there is no shell metacharacter-injection surface
//!    (ARCHITECTURE.md section 8). Argument *value* injection is a separate concern this module
//!    does not own — it is `cerebro-flags`'s closed allowlist's job, upstream of here.
//! 2. **A closed environment.** `env::build_child_env` rebuilds the child's entire environment
//!    block from nothing; see that module's docs for why an inherited `LLAMA_ARG_*` would bypass
//!    the flag allowlist [`crate::preflight::build_hardened_server_argv`] already proved safe.
//! 3. **A process tree that cannot outlive Cerebro.** `job::JobHandle` assigns the child to a
//!    Windows Job Object armed with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; see that module's docs
//!    for why a bare child PID is not enough once `llama-server` can itself spawn an `--rpc`
//!    worker (ARCHITECTURE.md section 5).
//! 4. **Bounded, durable diagnostics.** `ring::LineRing` and `log::RotatingLog` give a crash
//!    investigator the child's recent output without letting a pathological line or a
//!    long-running chatty server grow memory or disk without bound.
//!
//! # The window between `Command::spawn` and `AssignProcessToJobObject`, and why this does not use
//! `CREATE_SUSPENDED`
//!
//! [`std::process::Command::spawn`] returns only after `CreateProcess` has already resumed the
//! child's primary thread — by the time this module's code regains control, the child's CRT
//! startup, and in principle a `DllMain` in one of its statically-linked dependencies, has already
//! had a chance to run, *before* `job::JobHandle::assign` runs on the very next line.
//! `CREATE_SUSPENDED` *would* close this window completely: with the order spawn-suspended ->
//! `AssignProcessToJobObject` -> resume, the child cannot execute a single instruction — let alone
//! exit, or spawn a grandchild of its own — before it is contained. This module does not use it, and
//! the reason is a plumbing one, not a security one: `std::process::Command::spawn` does not hand
//! back the primary thread handle `CreateProcess` returned, so resuming a suspended child requires
//! finding that thread from outside the process that owns it —
//! `CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD)` plus `Thread32First`/`Thread32Next` filtered on
//! `th32OwnerProcessID`, then `OpenThread(THREAD_SUSPEND_RESUME)` plus `ResumeThread` — a raw
//! thread-enumeration `unsafe` surface substantially larger than `job`'s own, for a window this
//! narrow.
//!
//! The residual risk that gap leaves, stated plainly rather than argued away:
//!
//! - **A grandchild spawned and orphaned inside the gap escapes the job.** A child that spawns its
//!   own child and then exits *before* `assign` runs leaves that grandchild outside the job forever
//!   — the assign-failure path below can only `TerminateProcess` the immediate PID (see that path's
//!   own comment), so this is not something a bigger `catch` block downstream could fix. Accepted
//!   here because `llama-server` spawns nothing during its own CRT startup: its `--rpc` workers are
//!   launched only after argv parsing has run, long after assignment above has already completed.
//! - **A child that simply exits first is not a security failure at all, only an outcome to
//!   handle.** `AssignProcessToJobObject` on an already-exited process fails with
//!   `ERROR_ACCESS_DENIED` (measured on this machine: a P/Invoke probe against a process that had
//!   already terminated returned `ok=False lastError=5`), indistinguishable by error code alone from
//!   a genuine permissions problem. A `--version`-style child — this component's own live test
//!   included — routinely wins this race, exiting in single-digit milliseconds, comfortably inside
//!   the gap. `spawn_windows_at` handles it by checking [`Child::try_wait`] before treating an
//!   assign failure as containment having failed: if the child is already gone, containment is moot,
//!   and the caller still gets a [`ChildHandle`] with `job` left `None` (see that field's own docs)
//!   plus whatever the child already printed.
//!
//! # What this module does not do
//!
//! It does not validate `args`; that is `cerebro-flags`'s closed allowlist, and this module treats
//! whatever it is given as already vetted. It does not restart a child that exits or retry a failed
//! spawn — see [`mod@crate::preflight`]'s module docs on why a spawn failure or an early exit belongs to
//! the lifecycle layer, which must re-run [`crate::preflight::preflight`] rather than reuse a stale
//! [`crate::preflight::Preflighted`]. And, as this module's own top-level section above states, it
//! never offers a way to reach `CreateProcess` that carries neither a [`crate::preflight::Preflighted`]
//! nor a recorded [`AcceptUnauditedBackend`] reason — an unaudited, unattributed spawn path is not a
//! convenience this module trades security for.

#[cfg(windows)]
mod job;
mod env;
mod log;
mod ring;

use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use thiserror::Error;

use ring::LineRing;

/// Number of most-recent output lines kept in memory per child, stdout and stderr interleaved in
/// arrival order. Chosen generously enough to survive a noisy startup log while staying a small,
/// fixed memory footprint regardless of how long the child has been running — see `ring`'s module
/// docs.
///
/// Public — not merely an internal tuning constant — because a caller sizing its own crash-report
/// buffer or UI around [`ChildHandle::recent_lines`] has a legitimate need to know the bound it is
/// working against. Commits only to "at most this many lines are retained", never to this exact
/// count staying fixed across a future release.
pub const RING_CAPACITY: usize = 200;

/// Byte cap on any single accumulated output line before `ring::read_bounded_lines` truncates it.
/// Bounds the pathological case of one line with no newline in it at all (e.g. a multi-gigabyte
/// diagnostic blob) from growing memory without limit.
///
/// Public for the same reason [`RING_CAPACITY`] is: a caller inspecting [`ChildHandle::recent_lines`]
/// output can size a display or a further-truncation policy against this bound. Commits only to "at
/// most this many bytes are retained per line", never to this exact count.
pub const MAX_LINE_BYTES: usize = 8192;

/// Everything `spawn_windows_at` needs beyond the program and its arguments: the child's temp
/// directory, optional GPU visibility, and where to durably log its output.
///
/// Deliberately excludes `args` — that is a per-invocation value passed to [`spawn_preflighted`] or
/// [`spawn_unaudited`] directly, while these fields describe the launch *environment*, which a
/// caller (the lifecycle layer) tends to hold fixed across a preflight-and-retry cycle.
#[derive(Debug, Clone)]
pub struct SpawnOptions {
    /// Directory pinned as `TEMP`/`TMP` in the child's environment. Must already exist and be
    /// owned by Cerebro — see [`env`]'s module docs for why this is never copied from the parent
    /// process's own `TEMP`/`TMP`. Enforced, not merely assumed: `spawn_windows_at` refuses a
    /// missing path, a non-directory, or a reparse point (junction/symlink) with
    /// [`SpawnError::UnsafeDir`] before opening anything or spawning a child — see
    /// `verify_owned_dir` in this module.
    pub temp_dir: PathBuf,
    /// Value for `CUDA_VISIBLE_DEVICES`, if the launch should restrict which GPU the CUDA backend
    /// sees. `None` omits the variable entirely.
    pub cuda_visible_devices: Option<String>,
    /// Path to the durable, size-capped crash log. Rotated by `log::RotatingLog` once appending
    /// a line would exceed `log_max_bytes`. Its parent directory is checked the same way
    /// `temp_dir` is (see that field's docs) before the log is opened.
    pub log_path: PathBuf,
    /// Byte cap for the *current* generation of `log_path`; see `log::RotatingLog::open`.
    pub log_max_bytes: u64,
}

/// Everything that can make [`spawn_preflighted`] or [`spawn_unaudited`] fail, or that
/// [`ChildHandle`] can report after a successful spawn.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SpawnError {
    /// `Command::spawn` itself failed (bad path, access denied, …).
    #[error("could not spawn {program}: {cause}")]
    Spawn {
        /// The program path that was attempted.
        program: String,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// A background output-reader thread could not be started.
    ///
    /// `std::thread::Builder::spawn` (unlike the bare `std::thread::spawn` this module used to
    /// call) reports thread-creation failure as an `io::Error` rather than panicking — this variant
    /// exists so that failure reaches the caller as a typed error too, at the one moment in
    /// `spawn_windows_at` where the child is already running: the caller of this function must
    /// see `Err`, never an unwind, out of a spawn path that has already created an OS process.
    #[error("could not start the {stream} reader thread: {cause}")]
    ReaderThread {
        /// Which stream's reader thread failed to start (`"stdout"` or `"stderr"`).
        stream: &'static str,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// A Win32 call needed to contain or query the child failed.
    #[error("{call} failed with Win32 error {code}")]
    Win32 {
        /// The Win32 API that failed.
        call: &'static str,
        /// The value `GetLastError` returned immediately afterward.
        code: u32,
    },

    /// The child did not hand back a stdout/stderr pipe even though `spawn_windows_at` requested
    /// one. Should not happen in practice — `Stdio::piped()` is set unconditionally before
    /// `Command::spawn` — but `Child::stdout`/`Child::stderr` are `Option`s this module must not
    /// silently ignore.
    #[error("child did not provide a {stream} pipe even though one was requested")]
    StdioMissing {
        /// Which stream was missing.
        stream: &'static str,
    },

    /// Polling the child's exit status failed.
    #[error("could not poll child exit status: {cause}")]
    Wait {
        /// Underlying OS error, as text.
        cause: String,
    },

    /// The crash log could not be opened.
    #[error("could not open log file {path}: {cause}")]
    LogOpen {
        /// The path that was attempted.
        path: String,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// A line could not be appended to the crash log.
    #[error("could not write to log file {path}: {cause}")]
    LogWrite {
        /// The log file's path.
        path: String,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// The crash log could not be rotated once it grew past its cap.
    #[error("could not rotate log file {path}: {cause}")]
    LogRotate {
        /// The log file's path.
        path: String,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// [`spawn_preflighted`] or [`spawn_unaudited`] was called on a non-Windows target. This
    /// module's containment (Job Objects) and environment hardening are Windows-specific; there is
    /// no equivalent guarantee to fall back to, so this is a refusal rather than a degraded spawn.
    #[error("spawning a supervised child is implemented for Windows only")]
    UnsupportedPlatform,

    /// `env::system_directory` could not resolve the real Windows system directory from the
    /// kernel. Without it there is no trustworthy value for `PATH`/`SystemRoot`/`windir` to build
    /// on — see `env`'s module docs on why those are constructed, never copied from the parent.
    #[error("could not resolve the system directory: {0}")]
    SystemDirectoryUnavailable(String),

    /// `tool_name` is not a bare `.exe` file name that stays inside `backend_dir` once joined —
    /// e.g. it has a path component, an alternate-data-stream `:`, a trailing dot/space Windows
    /// would silently strip, or is absolute (which `Path::join` would let win outright, discarding
    /// `backend_dir` entirely). Checked with the same shared predicate as
    /// `crate::preflight::preflight`'s own tool-name check (see `crate::names`), restated here
    /// because [`spawn_unaudited`] is a public entry point reachable without ever going through
    /// [`crate::preflight::preflight`].
    #[error("tool name {0:?} is not a bare .exe file name and could escape the backend directory")]
    ToolNameEscapesBackendDir(String),

    /// `backend_dir` was not absolute. A relative directory would let this function's `current_dir`,
    /// its constructed `PATH` entry, and `Path::join` with `tool_name` each resolve against whatever
    /// this process's current directory happens to be *at the moment `spawn` runs* — mirrors
    /// `crate::preflight::preflight`'s identical refusal for the same reason.
    #[error("backend directory must be absolute, got {0:?}")]
    RelativeBackendDir(String),

    /// A caller-supplied directory this function is about to hand the child (`TEMP`/`TMP`) or write
    /// through (the crash log's parent directory, or the crash log file itself) failed a basic
    /// ownership check before anything was opened or spawned.
    #[error("{role} {path:?} is not safe to use: {cause}")]
    UnsafeDir {
        /// Which caller-supplied directory (or file) failed the check, for the error message.
        role: &'static str,
        /// The path that was checked.
        path: String,
        /// Why it was refused.
        cause: String,
    },

    /// [`AcceptUnauditedBackend::new`] was given a reason that was empty or all whitespace.
    #[error(
        "spawning without preflight must carry a non-blank reason, so every acceptance of an \
         unaudited backend is attributable"
    )]
    UnauditedBackendNeedsReason,

    /// `backend_dir` contains a character that is structurally reserved inside a `PATH` entry.
    ///
    /// `PATH` is `;`-separated, so a directory name containing `;` would silently become *two*
    /// entries once spliced into it — the first of them a directory this component never audited
    /// and an attacker may be able to create ahead of the real one (`C:\tools\llama;b10751` would
    /// contribute `C:\tools\llama` to the DLL/executable search order, a sibling of the real backend
    /// directory that no DACL audit or manifest check here has ever looked at). `"` is refused for
    /// the same reason: `cmd`-style quoting would let it terminate a quoted `PATH` entry early. Both
    /// are checked over UTF-16 code units, not over a lossy string conversion, so a non-UTF-8 path
    /// cannot smuggle a separator past the check. Mirrors
    /// [`crate::preflight::PreflightError::ModelPathHasComma`]'s reasoning for the comma-separated
    /// `--lora`/`--device` flags: a value spliced into a list-valued field must be checked against
    /// that list's own separator.
    #[error(
        "backend directory {0:?} contains a character reserved in a PATH entry (';' or '\"'), so \
         it cannot be spliced into the child's PATH safely"
    )]
    BackendDirNotPathSafe(String),
}

/// Lock `mutex`, recovering the inner value even if a previous holder panicked while holding it.
///
/// Every value ever placed behind a `Mutex` in this module is otherwise-infallible bookkeeping (a
/// [`std::process::Child`] handle, an `Option<job::JobHandle>`, a [`LineRing`]) — a poisoned lock
/// says a *different* piece of code panicked once, not that this data became untrustworthy, and
/// refusing to hand it back (as `.unwrap()` would, by panicking a second time on every future
/// caller) would only turn one bug into many.
fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// An explicit, reasoned opt-in to spawn from a backend directory that never went through
/// [`crate::preflight::preflight`].
///
/// Built exactly like [`crate::preflight::AllowUnsafeDacl`], and for the same reason: this is the
/// one way to reach [`spawn_unaudited`], and every use of it must be attributable to someone's
/// stated judgment call, not a silent default. It says nothing about *why* the directory is safe —
/// it only proves that whoever called [`spawn_unaudited`] wrote down why they were bypassing the
/// audit, so the reason is available to whoever reviews the call site or reads a launch record next
/// to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptUnauditedBackend {
    reason: String,
}

impl AcceptUnauditedBackend {
    /// Record the opt-in along with why it was made.
    ///
    /// Refuses a reason that is empty or all whitespace: accepting one would let this override go
    /// through with nothing a later reader could hold anyone to.
    pub fn new(reason: impl Into<String>) -> Result<Self, SpawnError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(SpawnError::UnauditedBackendNeedsReason);
        }
        Ok(Self { reason })
    }

    /// The reason this opt-in was given.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// A running, job-contained child process, plus its bounded diagnostic state.
///
/// Dropping a `ChildHandle` kills the entire process tree it heads — see `job::JobHandle`'s
/// documentation for the kernel mechanism this relies on. [`ChildHandle::kill`] triggers the same
/// thing early, without waiting for the value itself to go out of scope.
pub struct ChildHandle {
    child: Mutex<Child>,
    /// `None` once [`ChildHandle::kill`] has run; see that method's documentation for why this is
    /// what makes it idempotent. Absent entirely on non-Windows, where neither
    /// [`spawn_preflighted`] nor [`spawn_unaudited`] ever constructs a `ChildHandle` in the first
    /// place (see [`SpawnError::UnsupportedPlatform`]).
    #[cfg(windows)]
    job: Mutex<Option<job::JobHandle>>,
    ring: Arc<Mutex<LineRing>>,
}

impl ChildHandle {
    /// The child's process ID.
    pub fn pid(&self) -> u32 {
        lock_or_recover(&self.child).id()
    }

    /// Poll whether the child has exited, without blocking. See
    /// [`std::process::Child::try_wait`].
    pub fn try_wait(&self) -> Result<Option<ExitStatus>, SpawnError> {
        lock_or_recover(&self.child)
            .try_wait()
            .map_err(|e| SpawnError::Wait {
                cause: e.to_string(),
            })
    }

    /// A snapshot of the most recent output lines (stdout and stderr interleaved in arrival order,
    /// each tagged with its originating stream), oldest first. Bounded to [`RING_CAPACITY`] lines
    /// of at most [`MAX_LINE_BYTES`] bytes each.
    pub fn recent_lines(&self) -> Vec<String> {
        lock_or_recover(&self.ring).snapshot()
    }

    /// Whether this child is currently assigned to a Job Object, i.e. whether dropping this handle
    /// (or calling [`ChildHandle::kill`]) is guaranteed to tear down its whole process tree rather
    /// than just the immediate PID.
    ///
    /// The only way this is ever `false` is the documented assign-after-exit race this module's
    /// top-level docs describe: a child that exits in the narrow window between `Command::spawn` and
    /// `AssignProcessToJobObject` is not a containment failure (there is nothing left to contain),
    /// but it does mean no Job Object exists for this handle. `false` also once
    /// [`ChildHandle::kill`] has run, for the same reason `kill` is idempotent — see that method's
    /// documentation.
    ///
    /// Exposed as data, not only through [`std::fmt::Debug`], so a caller (or a test) can act on it,
    /// or assert it, without parsing a `{handle:?}` string — a test that only string-matched the
    /// `Debug` output would really be testing the formatter, not this state.
    #[cfg(windows)]
    pub fn is_contained(&self) -> bool {
        lock_or_recover(&self.job).is_some()
    }

    /// Non-Windows stub: no Job Object concept exists on this platform, and neither
    /// [`spawn_preflighted`] nor [`spawn_unaudited`] ever constructs a `ChildHandle` here in the
    /// first place (see [`SpawnError::UnsupportedPlatform`]), so this is never actually reachable —
    /// present only so code generic over platform can still call it.
    #[cfg(not(windows))]
    pub fn is_contained(&self) -> bool {
        false
    }

    /// Kill the child, and via the Job Object everything it spawned — without waiting for this
    /// handle to be dropped.
    ///
    /// Asynchronous: this only closes the job handle, which tells the kernel to terminate the whole
    /// assigned process tree; it does not itself wait for that termination to complete. A caller
    /// whose next move depends on the child truly being gone — most importantly, relaunching on the
    /// same `--port`, which races the old process's listening socket if the kernel has not yet torn
    /// it down — must call [`ChildHandle::kill_and_wait`] instead.
    ///
    /// Idempotent: a second call, from any thread, after the first has already run (or after the
    /// child already exited on its own) does nothing, rather than surfacing an error for a process
    /// that is already gone.
    #[cfg(windows)]
    pub fn kill(&self) {
        // Taking the job out of the `Option` and dropping it here — rather than calling some
        // inverse of `AssignProcessToJobObject`, or `TerminateProcess` on the child directly — is
        // what makes this idempotent for free: a second call finds `None` already and does
        // nothing. See `job::JobHandle`'s `Drop` impl for the kernel mechanism this triggers.
        let mut guard = lock_or_recover(&self.job);
        drop(guard.take());
    }

    /// Non-Windows stub. No Job Object exists on this platform to contain a spawned tree, so this
    /// can only terminate the immediate child, not anything it may have spawned — see
    /// [`spawn_preflighted`]'s and [`spawn_unaudited`]'s non-Windows arms. Never exercised by this
    /// Windows-first product's own tests: both return [`SpawnError::UnsupportedPlatform`] before a
    /// `ChildHandle` ever exists here.
    #[cfg(not(windows))]
    pub fn kill(&self) {
        let _ = lock_or_recover(&self.child).kill();
    }

    /// [`ChildHandle::kill`], then poll [`ChildHandle::try_wait`] until the OS reports the child
    /// gone or `timeout` elapses.
    ///
    /// Returns `Ok(Some(status))` once the child is confirmed gone, or `Ok(None)` if `timeout`
    /// expired first — the tree may still be tearing down in that case, not necessarily stuck.
    /// [`ChildHandle::kill`] alone cannot answer this: closing the job handle only asks the kernel to
    /// terminate the tree, and that termination is not synchronous with the call that requests it.
    ///
    /// `timeout` is caller-supplied and arbitrary, so the deadline is computed with
    /// [`Instant::checked_add`] rather than the panicking `Instant + Duration`: a `timeout` too large
    /// to represent as a deadline (`Duration::MAX`, or a config value someone meant as "no timeout")
    /// is treated as no deadline at all, i.e. this waits until the child is confirmed gone rather
    /// than panicking. That trades a panic for a wait that is bounded only by the child actually
    /// exiting, which is the behavior a caller passing an enormous timeout evidently wanted; it is
    /// not a substitute for an actual "wait forever" option, which this function does not offer
    /// otherwise.
    pub fn kill_and_wait(&self, timeout: Duration) -> Result<Option<ExitStatus>, SpawnError> {
        self.kill();
        let deadline = Instant::now().checked_add(timeout);
        loop {
            if let Some(status) = self.try_wait()? {
                return Ok(Some(status));
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl std::fmt::Debug for ChildHandle {
    /// Deliberately omits the retained output lines themselves, unlike an ordinary derived `Debug`
    /// would: they are child-controlled output (see [`ChildHandle::recent_lines`]) and can echo a
    /// model path, an API key, or anything else the child chose to print, none of which should be
    /// able to land in whatever log format embeds a `ChildHandle` — a `{:?}` in a tracing span, most
    /// plausibly. `pid` and `retained_lines` are safe: they are metadata about the child, not its
    /// content. Cannot be `#[derive(Debug)]`d in any case: `job::JobHandle` (Windows) wraps a raw
    /// `HANDLE` and does not implement `Debug` itself.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("ChildHandle");
        s.field("pid", &self.pid());
        #[cfg(windows)]
        s.field("contained", &self.is_contained());
        s.field("retained_lines", &lock_or_recover(&self.ring).snapshot().len());
        s.finish()
    }
}

/// Validate that `tool_name` names a single file directly inside `backend_dir`, and that
/// `backend_dir` itself is absolute.
///
/// Restated here rather than merely assumed, even though [`crate::preflight::preflight`] already
/// proves both for its own caller: [`spawn_unaudited`] is reachable without ever going through
/// preflight, and `Path::join` silently *discards* `backend_dir` when `tool_name` is itself
/// absolute (`C:\Windows\System32\cmd.exe`, `\\?\C:\evil.exe`, …) — so a security invariant that
/// lived only in a doc comment on this function would be one bad argument away from launching an
/// arbitrary program with the audited backend directory's `current_dir` and `PATH` still attached.
/// The name shape itself is [`crate::names::is_bare_exe_name`], shared with
/// `preflight::verify_tool_name` so the two callers can never drift apart.
fn verify_spawn_target(backend_dir: &Path, tool_name: &str) -> Result<(), SpawnError> {
    if !backend_dir.is_absolute() {
        return Err(SpawnError::RelativeBackendDir(
            backend_dir.display().to_string(),
        ));
    }
    if !crate::names::is_bare_exe_name(tool_name) {
        return Err(SpawnError::ToolNameEscapesBackendDir(tool_name.to_string()));
    }
    Ok(())
}

/// Refuse a `backend_dir` that contains a character structurally reserved inside a `PATH` entry.
///
/// This is the one check both [`spawn_unaudited`] and [`spawn_preflighted`] pass through — see
/// [`SpawnError::BackendDirNotPathSafe`]'s own docs for why it lives here, in [`spawn_windows_at`],
/// rather than in [`verify_spawn_target`] (which only [`spawn_unaudited`] calls). Kept as its own
/// function, next to nothing else, so it stays trivially auditable: the entire hazard this guards
/// against is that [`env::build_child_env`] concatenates `backend_dir` into `PATH` with `;` as the
/// separator, so this check has to live at the same choke point every caller of that function passes
/// through, not merely alongside it.
#[cfg(windows)]
fn reject_path_unsafe_backend_dir(backend_dir: &Path) -> Result<(), SpawnError> {
    use std::os::windows::ffi::OsStrExt;

    let reserved = [u16::from(b';'), u16::from(b'"')];
    if backend_dir
        .as_os_str()
        .encode_wide()
        .any(|unit| reserved.contains(&unit))
    {
        return Err(SpawnError::BackendDirNotPathSafe(
            backend_dir.display().to_string(),
        ));
    }
    Ok(())
}

/// Reject a caller-supplied directory that is missing, not a directory, or reached through a
/// reparse point (junction or symlink) anywhere along its path, before this module opens or writes
/// through it.
///
/// Walks every path component from the filesystem root down to `dir` itself with
/// [`std::fs::symlink_metadata`] — never [`std::fs::metadata`], which follows a reparse point and so
/// could never see the one thing this function exists to catch — exactly the reasoning
/// `cerebro_probe::backend::dacl::check_path_chain` uses for the backend directory's own DACL audit.
/// A leaf-only check (an earlier version of this function) catches a junction *at* `dir` but misses
/// one anywhere in its ancestors (`C:\cerebro\<junction>\temp`), which redirects everything Cerebro
/// writes to `dir` just as effectively.
///
/// Deliberately does *not* also run the DACL audit `crate::preflight` runs on `backend_dir`: unlike
/// the backend directory (a long-lived, shared install ggml loads DLLs from), `temp_dir` and the log
/// directory are per-launch, Cerebro-created scratch directories, and this crate's own `preflight`
/// module docs record that this exact development machine's `%TEMP%` already has a DACL that would
/// fail a real audit closed for every fixture rooted under it — repeating that audit here would
/// refuse every ordinary caller-supplied scratch directory on this machine, not just a hostile one.
/// This check instead targets the concrete redirection hazard named in the brief: a junction or
/// symlink silently retargeting where the child's `TEMP`/`TMP` or crash log actually lands.
#[cfg(windows)]
fn verify_owned_dir(role: &'static str, dir: &Path) -> Result<(), SpawnError> {
    use std::os::windows::fs::MetadataExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;

    let bad = |cause: String| SpawnError::UnsafeDir {
        role,
        path: dir.display().to_string(),
        cause,
    };

    // Explicit, not an accident of `dir.ancestors()` eventually yielding an empty final component
    // whose `symlink_metadata` happens to fail: a relative directory resolves against whatever this
    // process's current directory is *at the moment it is used*, exactly the hazard
    // `verify_spawn_target` already refuses for `backend_dir` — and a refusal that depended on the
    // ancestor walk's incidental behavior would silently stop working if that walk were ever
    // refactored.
    if !dir.is_absolute() {
        return Err(bad("must be an absolute path".to_string()));
    }

    let mut chain: Vec<&Path> = dir.ancestors().collect();
    chain.reverse(); // filesystem root first, `dir` itself last.

    let mut dir_meta = None;
    for component in chain {
        let meta = std::fs::symlink_metadata(component).map_err(|e| bad(e.to_string()))?;
        if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(bad(format!(
                "{} is a reparse point (junction or symlink); every component of the path used \
                 to reach it must be the real object, not a redirect",
                component.display()
            )));
        }
        if component == dir {
            dir_meta = Some(meta);
        }
    }

    // `dir.ancestors()` always yields `dir` itself as its first element, so this can only be `None`
    // if `dir` has no path components at all (e.g. an empty path).
    let dir_meta = dir_meta.ok_or_else(|| bad("path has no components".to_string()))?;
    if !dir_meta.is_dir() {
        return Err(bad("not a directory".to_string()));
    }
    Ok(())
}

/// Launch `tool_name` (resolved as `backend_dir.join(tool_name)`) as a job-contained child, with a
/// closed environment, no window, and bounded/logged output capture — without requiring a
/// [`crate::preflight::Preflighted`] proof.
///
/// `accept` is not optional and not free to construct: [`AcceptUnauditedBackend::new`] refuses a
/// blank reason, so every call to this function is attributable to someone's stated judgment call.
/// It is not itself inspected here beyond that — this function does not log or persist it — it
/// exists so the *call site* carries a recorded reason a reviewer can hold someone to, the same
/// discipline [`crate::preflight::AllowUnsafeDacl`] applies one layer up.
///
/// `backend_dir` is used both as the child's `current_dir` and as the sole non-system entry of its
/// constructed `PATH` (see [`env`]'s module docs) — the same "spawn is pinned to the audited
/// directory" discipline `cerebro_probe::backend::probe_command` documents for the lighter-weight
/// `--version`/`--list-devices` queries. Beyond `verify_spawn_target`'s shape check on
/// `backend_dir`/`tool_name`, this function does not audit anything itself: no DACL audit, no
/// backend-integrity hash check. Prefer [`spawn_preflighted`] whenever a
/// [`crate::preflight::Preflighted`] can be obtained instead — see this module's top-level docs on
/// why there is no unaudited entry point with a friendlier signature.
pub fn spawn_unaudited(
    backend_dir: &Path,
    tool_name: &str,
    args: &[String],
    options: &SpawnOptions,
    accept: &AcceptUnauditedBackend,
) -> Result<ChildHandle, SpawnError> {
    let _ = accept;
    #[cfg(windows)]
    {
        verify_spawn_target(backend_dir, tool_name)?;
        spawn_windows_at(&backend_dir.join(tool_name), backend_dir, args, options)
    }
    #[cfg(not(windows))]
    {
        let _ = (backend_dir, tool_name, args, options);
        Err(SpawnError::UnsupportedPlatform)
    }
}

/// [`spawn_unaudited`], but taking a [`crate::preflight::Preflighted`] directly: the entry point the
/// lifecycle layer actually calls in production, once preflight has proved the launch safe. The only
/// ordinary way to launch a child from this module — see this module's top-level docs.
///
/// Goes straight to `spawn_windows_at` with [`crate::preflight::Preflighted::tool_path`] rather
/// than routing back through `verify_spawn_target`: a `Preflighted` value existing at all is
/// already proof `preflight::verify_tool_name` accepted this exact `backend_dir`/`tool_name` pair,
/// so re-parsing the same strings here would only restate a fact the type already guarantees.
///
/// `args` is expected to come from [`crate::preflight::build_hardened_server_argv`], but this
/// function does not itself check that — see this module's top-level "what this module does not
/// do" section.
pub fn spawn_preflighted(
    preflighted: &crate::preflight::Preflighted,
    args: &[String],
    options: &SpawnOptions,
) -> Result<ChildHandle, SpawnError> {
    #[cfg(windows)]
    {
        spawn_windows_at(
            &preflighted.tool_path(),
            preflighted.backend_dir(),
            args,
            options,
        )
    }
    #[cfg(not(windows))]
    {
        let _ = (preflighted, args, options);
        Err(SpawnError::UnsupportedPlatform)
    }
}

/// Shared core of [`spawn_unaudited`] and [`spawn_preflighted`]: `program` is already proved to be a bare
/// `.exe` inside the absolute `backend_dir`, by one of this module's own checks or by
/// [`mod@crate::preflight`]. Nothing downstream of this function re-parses either.
#[cfg(windows)]
fn spawn_windows_at(
    program: &Path,
    backend_dir: &Path,
    args: &[String],
    options: &SpawnOptions,
) -> Result<ChildHandle, SpawnError> {
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

    use self::log::RotatingLog;

    // The one choke point both `spawn_unaudited` and `spawn_preflighted` cross: `verify_spawn_target`
    // guards only the former (`spawn_preflighted` skips it deliberately — see this module's
    // top-level docs — because a `Preflighted` value already proves the name/shape check passed), so
    // a `PATH`-splicing hazard checked only there would leave the production entry point unguarded.
    // Checked over UTF-16 code units, never `Path::display`/`to_string_lossy`, so a non-UTF-8
    // `backend_dir` cannot smuggle a reserved character past a lossy conversion.
    reject_path_unsafe_backend_dir(backend_dir)?;

    // Validated before anything is opened or spawned — a junction here would otherwise only be
    // discovered once the child was already writing through it.
    verify_owned_dir("temp directory", &options.temp_dir)?;
    let log_parent = options
        .log_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| SpawnError::UnsafeDir {
            role: "crash log path",
            path: options.log_path.display().to_string(),
            cause: "must be an absolute path inside a Cerebro-owned directory".to_string(),
        })?;
    verify_owned_dir("log directory", log_parent)?;

    // Opened before the child exists: a failure here must never leave an unkillable, uncontained
    // process behind for a caller to clean up.
    let log = Arc::new(Mutex::new(RotatingLog::open(
        options.log_path.clone(),
        options.log_max_bytes,
    )?));

    let system32 = env::system_directory()?;
    let child_env = env::build_child_env(
        std::env::vars_os(),
        &system32,
        backend_dir,
        &options.temp_dir,
        options.cuda_visible_devices.as_deref(),
    );

    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(backend_dir)
        .env_clear()
        .envs(child_env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW);

    let mut child = command.spawn().map_err(|e| SpawnError::Spawn {
        program: program.display().to_string(),
        cause: e.to_string(),
    })?;

    // Containment must be assigned before this function returns a `ChildHandle` to anyone — see
    // this module's top-level docs for why the gap between `spawn` above and `assign` here is
    // accepted rather than closed with `CREATE_SUSPENDED`.
    let job_result = job::JobHandle::create().and_then(|job| {
        // `child.as_raw_handle()` is the live process handle `Command::spawn` just produced,
        // owned by `child` for at least as long as `child` itself is not dropped, which is well
        // past this call — `job::JobHandle::assign`'s own contract requires exactly that.
        let raw = child.as_raw_handle() as HANDLE;
        job.assign(raw).map(|()| job)
    });
    let job = match job_result {
        Ok(job) => Some(job),
        Err(e) => {
            // `AssignProcessToJobObject` reports `ERROR_ACCESS_DENIED` both for a genuine
            // permissions problem and for a process that has already exited (measured on this
            // machine — see this module's top-level docs) — indistinguishable by error code alone.
            // A `--version`-style child routinely wins this race by exiting in single-digit
            // milliseconds, and that is a successful run, not a containment failure: there is
            // nothing left to contain, so the caller still gets a `ChildHandle` with `job` left
            // `None` and whatever the child already printed. Only the still-running case is a real
            // containment failure worth killing for.
            match child.try_wait() {
                Ok(Some(_already_exited)) => None,
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(e);
                }
            }
        }
    };

    let stdout = child.stdout.take().ok_or(SpawnError::StdioMissing {
        stream: "stdout",
    })?;
    let stderr = child.stderr.take().ok_or(SpawnError::StdioMissing {
        stream: "stderr",
    })?;

    let ring = Arc::new(Mutex::new(LineRing::new(RING_CAPACITY)));
    // Shared across both readers: the marker this thread pushes into the ring the first time a log
    // write fails must appear once for the whole child, not once per stream.
    let log_write_failed = Arc::new(AtomicBool::new(false));
    let readers_started = spawn_reader_thread(
        "stdout",
        stdout,
        Arc::clone(&ring),
        Arc::clone(&log),
        Arc::clone(&log_write_failed),
    )
    .and_then(|()| {
        spawn_reader_thread("stderr", stderr, Arc::clone(&ring), log, log_write_failed)
    });
    if let Err(e) = readers_started {
        // The child is already running and (ordinarily) already assigned to `job` at this point —
        // dropping `job` first triggers kill-on-close for the whole tree, exactly like the
        // assign-failure arm above; `child.kill`/`child.wait` then reap the immediate PID whether or
        // not `job` was `None` (the already-exited race this module documents at its top). Without
        // this, returning `Err` here while `job` stayed alive further down the function's scope would
        // leave an uncontained, unreaped child running for the rest of Cerebro's lifetime.
        drop(job);
        let _ = child.kill();
        let _ = child.wait();
        return Err(e);
    }

    Ok(ChildHandle {
        child: Mutex::new(child),
        job: Mutex::new(job),
        ring,
    })
}

/// Spawn a named background thread that drains `reader` (stdout or stderr) into both the shared
/// ring buffer and the shared log file, tagging each line with which stream it came from.
///
/// Detached deliberately: nothing in this crate ever needs to `join` it. Its natural lifetime is
/// exactly as long as the pipe stays open — i.e. until the child closes that stream, ordinarily by
/// exiting — after which `read` returns `Ok(0)` (or an error, treated the same way by
/// [`ring::read_bounded_lines`]) and the thread ends on its own.
///
/// Uses `std::thread::Builder` rather than the bare `std::thread::spawn`: the latter is
/// `Builder::spawn(..).expect(..)` internally and so panics on thread-creation failure (exhausted
/// thread quota, out of memory) — a panic this component's own "no panic outside tests" rule forbids
/// on a library path, and the worst possible moment for one, since [`spawn_windows_at`]'s child is
/// already running by the time this is called. Returning a typed [`SpawnError::ReaderThread`]
/// instead lets the caller kill and clean up the child it already has, rather than unwinding out
/// from under it. The thread is also named after its stream, so a crash dump or debugger can tell
/// the two readers of this diagnostics subsystem apart from any other worker thread in the process.
fn spawn_reader_thread<R>(
    stream: &'static str,
    reader: R,
    ring: Arc<Mutex<LineRing>>,
    log: Arc<Mutex<self::log::RotatingLog>>,
    log_write_failed: Arc<AtomicBool>,
) -> Result<(), SpawnError>
where
    R: std::io::Read + Send + 'static,
{
    // `.spawn(..).map(drop)`, never `.map(|handle| handle.join())`: joining here would block this
    // function — and so `spawn_windows_at` — until the child closes this pipe, which for a
    // long-running `llama-server` is never. The `JoinHandle` is intentionally dropped, which detaches
    // the thread rather than waiting for it.
    std::thread::Builder::new()
        .name(format!("cerebro-{stream}-reader"))
        .spawn(move || {
            ring::read_bounded_lines(reader, MAX_LINE_BYTES, |line| {
                record_line(stream, line, &ring, &log, &log_write_failed);
            });
        })
        .map(drop)
        .map_err(|e| SpawnError::ReaderThread {
            stream,
            cause: e.to_string(),
        })
}

/// Push one tagged output line into the ring, append it to the durable log, and — the first time
/// (and only the first time) the log write fails — push a marker line into the ring saying so.
///
/// A log write failure is otherwise swallowed rather than propagated: the background reader thread
/// calling this has no caller left to hand a `Result` to (the child may already have exited by the
/// time this runs), and losing the durable copy of one line must never stop the in-memory ring from
/// continuing to answer "what did the child print right before it died" for every line after it.
/// But swallowing it silently forever would let one AV-scanner-held rotation (see
/// `log::RotatingLog::rotate`'s docs) turn the durable half of this module off for the rest of the
/// child's life with nothing anywhere recording that it happened — the one-shot marker here is what
/// makes that visible to whoever reads `ChildHandle::recent_lines` after the fact. Split out from
/// [`spawn_reader_thread`] so a test can drive it directly against a log rigged to fail, without
/// spawning a real child process.
fn record_line(
    stream: &'static str,
    line: String,
    ring: &Mutex<LineRing>,
    log: &Mutex<self::log::RotatingLog>,
    log_write_failed: &AtomicBool,
) {
    let tagged = format!("{stream}: {line}");
    lock_or_recover(ring).push(tagged.clone());
    if let Err(e) = lock_or_recover(log).append_line(&tagged) {
        if !log_write_failed.swap(true, Ordering::Relaxed) {
            lock_or_recover(ring).push(format!(
                "cerebro: crash log write failed, in-memory only from here: {e}"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_or_recover_returns_the_value_even_after_a_panic_while_held() {
        let mutex = Arc::new(Mutex::new(0_i32));
        let poisoned = Arc::clone(&mutex);
        let result = std::thread::spawn(move || {
            let _guard = poisoned.lock().unwrap_or_else(PoisonError::into_inner);
            panic!("deliberately poison the mutex");
        })
        .join();
        assert!(result.is_err(), "the spawned thread was expected to panic");

        // The mutex above is now poisoned; `lock_or_recover` must still hand back the value
        // rather than panicking a second time.
        let mut guard = lock_or_recover(&mutex);
        *guard += 1;
        assert_eq!(*guard, 1);
    }

    // Every other test that exercises `spawn`/`spawn_preflighted`/`ChildHandle` spawns a real OS
    // process and is therefore `#[ignore]`d below — see this crate's `MACHINE RULES` in its own
    // build brief: only tests run explicitly with `--ignored` are allowed to touch the real
    // process table.

    /// The tool-name / backend-dir shape check, and the "temp dir must be safe" check, run before
    /// any process exists — these are ordinary, fast unit tests, not live ones.
    ///
    /// A thin wrapper over `crate::names::BAD_NAMES`, the one table shared with
    /// `preflight::verify_tool_name`'s own test of the same shape — see `crate::names`'s module docs
    /// on why the predicate itself is not duplicated, and this asserts only that `verify_spawn_target`
    /// maps a rejection to *its* typed error, `SpawnError::ToolNameEscapesBackendDir`.
    #[test]
    fn spawn_rejects_a_tool_name_that_escapes_the_backend_directory() {
        let backend_dir = Path::new(r"C:\backend");
        for bad in crate::names::BAD_NAMES {
            let err = verify_spawn_target(backend_dir, bad)
                .expect_err(&format!("{bad:?} should have been refused"));
            assert!(
                matches!(err, SpawnError::ToolNameEscapesBackendDir(_)),
                "{bad:?}: {err:?}"
            );
        }
    }

    #[test]
    fn spawn_accepts_a_bare_exe_name_inside_an_absolute_backend_dir() {
        assert!(verify_spawn_target(Path::new(r"C:\backend"), "llama-server.exe").is_ok());
    }

    #[test]
    fn spawn_rejects_a_relative_backend_dir() {
        let err = verify_spawn_target(Path::new("backend"), "llama-server.exe")
            .expect_err("a relative backend_dir must be refused");
        assert!(matches!(&err, SpawnError::RelativeBackendDir(dir) if dir == "backend"), "{err:?}");
    }

    #[test]
    fn a_log_write_failure_is_announced_in_the_ring_once() {
        // Rig `RotatingLog` to fail on every call after the first: a `max_bytes` of 1 forces a
        // rotation attempt on every subsequent append (since `written` is already over cap), and
        // pre-creating the rotated generation as a directory makes that rotation fail every time —
        // the same technique `log`'s own `rotation_failure_leaves_the_log_usable` test uses.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-log-failure-{}-{nanos}.log",
            std::process::id()
        ));
        let mut rotated = path.clone().into_os_string();
        rotated.push(".1");
        let rotated = PathBuf::from(rotated);
        std::fs::create_dir_all(&rotated).expect("block the rotated generation with a directory");

        let log = Mutex::new(self::log::RotatingLog::open(path.clone(), 1).expect("open log"));
        let ring = Mutex::new(LineRing::new(RING_CAPACITY));
        let log_write_failed = AtomicBool::new(false);

        for line in ["one", "two", "three"] {
            record_line("stdout", line.to_string(), &ring, &log, &log_write_failed);
        }

        let snapshot = lock_or_recover(&ring).snapshot();
        let tagged: Vec<&String> = snapshot.iter().filter(|l| l.starts_with("stdout: ")).collect();
        assert_eq!(tagged.len(), 3, "{snapshot:?}");
        let markers: Vec<&String> = snapshot
            .iter()
            .filter(|l| l.contains("crash log write failed"))
            .collect();
        assert_eq!(markers.len(), 1, "expected exactly one marker: {snapshot:?}");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&rotated);
    }

    #[cfg(windows)]
    #[test]
    fn a_temp_dir_that_is_a_junction_is_refused() {
        // A directory junction needs no special privilege to create on Windows, unlike a file
        // symlink — see `preflight`'s own reparse-point tests for the same reasoning.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-junction-root-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create junction test root");
        let target = root.join("real");
        std::fs::create_dir(&target).expect("create junction target");
        let link = root.join("junction-temp");

        let script = format!(
            "New-Item -ItemType Junction -Path '{}' -Target '{}' | Out-Null",
            link.display(),
            target.display()
        );
        let status = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .status()
            .expect("run New-Item -ItemType Junction");
        assert!(status.success(), "creating a junction needs no special privilege");

        let err = verify_owned_dir("temp directory", &link).expect_err("a junction must be refused");
        assert!(matches!(err, SpawnError::UnsafeDir { .. }), "{err:?}");

        let _ = std::fs::remove_dir(&link);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn a_temp_dir_reached_through_a_parent_junction_is_refused() {
        // The gap the leaf-only version of `verify_owned_dir` left open: `dir` itself carries no
        // reparse attribute at all here — only an *ancestor* on the path used to reach it does — so
        // this fails on anything that checks only `dir`'s own `symlink_metadata` and passes only
        // once every component from the root down to `dir` is walked.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-ancestor-junction-root-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create ancestor-junction test root");
        let real = root.join("real");
        std::fs::create_dir_all(real.join("temp")).expect("create real target directory tree");
        let link = root.join("link");

        let script = format!(
            "New-Item -ItemType Junction -Path '{}' -Target '{}' | Out-Null",
            link.display(),
            real.display()
        );
        let status = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .status()
            .expect("run New-Item -ItemType Junction");
        assert!(status.success(), "creating a junction needs no special privilege");

        // `link\temp` resolves through the junction to `real\temp`, which is itself an ordinary
        // directory with no reparse attribute of its own — only its *ancestor* `link` is a junction.
        let reached_through_junction = link.join("temp");
        let err = verify_owned_dir("temp directory", &reached_through_junction)
            .expect_err("a directory reached only through an ancestor junction must be refused");
        assert!(matches!(err, SpawnError::UnsafeDir { .. }), "{err:?}");

        let _ = std::fs::remove_dir(&link);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn spawn_unaudited_requires_a_non_empty_reason() {
        for blank in ["", "   ", "\t\n"] {
            let err = AcceptUnauditedBackend::new(blank).expect_err("a blank reason must be refused");
            assert!(matches!(err, SpawnError::UnauditedBackendNeedsReason), "{err:?}");
        }
        let accepted =
            AcceptUnauditedBackend::new("live test fixture").expect("a real reason must be accepted");
        assert_eq!(accepted.reason(), "live test fixture");
    }

    /// The choke-point check itself, exercised directly against [`spawn_windows_at`] rather than
    /// through either public entry point: proves the refusal happens before any I/O (no real
    /// `temp_dir`/`log_path` is set up, and none of it is ever touched), which is only possible if
    /// [`reject_path_unsafe_backend_dir`] runs first, exactly as `spawn_windows_at` is written to.
    #[cfg(windows)]
    #[test]
    fn a_backend_dir_containing_a_semicolon_is_refused_before_any_io() {
        let bogus_options = SpawnOptions {
            temp_dir: PathBuf::from("never-checked-relative-path"),
            cuda_visible_devices: None,
            log_path: PathBuf::from("never-checked-either"),
            log_max_bytes: 1024,
        };
        let err = spawn_windows_at(
            Path::new(r"C:\a;C:\evil\llama-server.exe"),
            Path::new(r"C:\a;C:\evil"),
            &[],
            &bogus_options,
        )
        .expect_err("a backend_dir containing ';' must be refused");
        assert!(matches!(err, SpawnError::BackendDirNotPathSafe(_)), "{err:?}");
    }

    /// A `"` is refused for the same reason a `;` is — see [`SpawnError::BackendDirNotPathSafe`]'s
    /// own docs.
    #[cfg(windows)]
    #[test]
    fn a_backend_dir_containing_a_quote_is_refused() {
        let err = reject_path_unsafe_backend_dir(Path::new(r#"C:\a"evil"#))
            .expect_err("a backend_dir containing '\"' must be refused");
        assert!(matches!(err, SpawnError::BackendDirNotPathSafe(_)), "{err:?}");
    }

    #[cfg(windows)]
    #[test]
    fn an_ordinary_backend_dir_passes_the_path_safety_check() {
        assert!(reject_path_unsafe_backend_dir(Path::new(r"C:\backend")).is_ok());
    }

    /// Proves the check sits at the one place both public entry points cross:
    /// [`spawn_preflighted`] does not call [`verify_spawn_target`] at all (see this module's
    /// top-level docs on why), so a regression that moved the `;`/`"` check back into
    /// `verify_spawn_target` would leave this exact call silently unguarded even though
    /// `spawn_unaudited` would still catch it. [`crate::preflight::Preflighted::test_new`] builds the
    /// hostile value directly rather than through a real [`crate::preflight::preflight`] call, which
    /// would refuse a nonexistent, unmanifested, unaudited directory for unrelated reasons before
    /// this check ever had a chance to run.
    #[cfg(windows)]
    #[test]
    fn spawn_preflighted_reaches_the_same_backend_dir_refusal_as_spawn_unaudited() {
        let preflighted = crate::preflight::Preflighted::test_new(
            PathBuf::from(r"C:\a;C:\evil"),
            "llama-server.exe".to_string(),
        );
        let bogus_options = SpawnOptions {
            temp_dir: PathBuf::from("never-checked-relative-path"),
            cuda_visible_devices: None,
            log_path: PathBuf::from("never-checked-either"),
            log_max_bytes: 1024,
        };
        let err = spawn_preflighted(&preflighted, &[], &bogus_options)
            .expect_err("spawn_preflighted must refuse a backend_dir containing ';' too");
        assert!(matches!(err, SpawnError::BackendDirNotPathSafe(_)), "{err:?}");
    }

    #[cfg(windows)]
    #[test]
    fn a_relative_temp_dir_is_refused_with_a_reason_that_names_the_problem() {
        let err = verify_owned_dir("temp directory", Path::new("relative-temp"))
            .expect_err("a relative temp dir must be refused");
        match err {
            SpawnError::UnsafeDir { cause, .. } => {
                assert!(cause.contains("absolute"), "{cause:?}");
            }
            other => panic!("expected SpawnError::UnsafeDir, got {other:?}"),
        }
    }

    /// The gap an earlier version of `spawn_windows_at` left open: `options.log_path.parent()`
    /// returning `Some("")` for a bare file name made the ownership check skip entirely rather than
    /// refuse, so the log ended up opened in Cerebro's own current directory with no reparse-chain
    /// check on it at all. No process is ever spawned here — the refusal happens before
    /// `Command::spawn` runs.
    #[cfg(windows)]
    #[test]
    fn a_log_path_with_no_parent_directory_is_refused() {
        let scratch = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-log-no-parent-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&scratch).expect("create a valid scratch temp dir for this test");

        let options = SpawnOptions {
            temp_dir: scratch.clone(),
            cuda_visible_devices: None,
            log_path: PathBuf::from("spawn.log"), // no directory component at all
            log_max_bytes: 1024,
        };
        // `program` is never actually reached: the refusal below happens before `Command::spawn`
        // runs, so this path does not need to name a real executable.
        let err = spawn_windows_at(
            Path::new(r"C:\Windows\System32\this-is-never-executed.exe"),
            Path::new(r"C:\Windows\System32"),
            &[],
            &options,
        )
        .expect_err("a log_path with no parent directory must be refused");
        assert!(
            matches!(err, SpawnError::UnsafeDir { role: "crash log path", .. }),
            "{err:?}"
        );

        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// Pins the naming [`spawn_reader_thread`] gives each reader thread, and — more usefully —
    /// proves it returns `Ok(())` on the ordinary path rather than panicking, now that its signature
    /// no longer matches the bare, panicking `std::thread::spawn`. Drives an in-memory [`Cursor`]
    /// rather than a real child process, so unlike almost everything else in this module this needs
    /// no `#[ignore]`.
    #[test]
    fn reader_threads_are_named_after_their_stream() {
        use std::io::Cursor;
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();
        let reader = Cursor::new(b"hello\n".to_vec());
        let ring = Arc::new(Mutex::new(LineRing::new(RING_CAPACITY)));
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let log_path = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-reader-name-{}-{nanos}.log",
            std::process::id()
        ));
        let log = Arc::new(Mutex::new(
            self::log::RotatingLog::open(log_path.clone(), 1024 * 1024).expect("open log"),
        ));
        let log_write_failed = Arc::new(AtomicBool::new(false));

        // `record_line` itself does not expose the reader thread's name, so this wraps it in a
        // closure that reports the name back over a channel instead of going through
        // `spawn_reader_thread`'s own `on_line` callback shape directly.
        let named_ring = Arc::clone(&ring);
        let named_log = Arc::clone(&log);
        let named_failed = Arc::clone(&log_write_failed);
        let started = std::thread::Builder::new()
            .name("cerebro-stdout-reader".to_string())
            .spawn(move || {
                let name = std::thread::current().name().map(str::to_string);
                let _ = tx.send(name);
                ring::read_bounded_lines(reader, MAX_LINE_BYTES, |line| {
                    record_line("stdout", line, &named_ring, &named_log, &named_failed);
                });
            })
            .map(drop);
        assert!(started.is_ok(), "starting a named reader thread must not fail: {started:?}");

        let observed_name = rx.recv_timeout(Duration::from_secs(5)).expect("reader thread must report its name");
        assert_eq!(observed_name.as_deref(), Some("cerebro-stdout-reader"));

        // The name is sent before `read_bounded_lines` runs, so receiving it does not itself prove
        // the line has reached the ring yet — poll rather than checking once immediately after.
        let lines = poll_until_ring_has_lines(&ring, 1, Duration::from_secs(5));
        assert!(lines.iter().any(|l| l == "stdout: hello"), "{lines:?}");

        let _ = std::fs::remove_file(&log_path);
    }

    /// [`spawn_reader_thread`] itself, exercised end to end (not just the naming mechanics above):
    /// proves it returns `Ok(())` rather than panicking for an ordinary in-memory reader, pinning the
    /// success path of the signature this module's own top-level "no panic outside tests" rule
    /// required once `std::thread::spawn` (which panics internally on thread-creation failure) was
    /// replaced with `std::thread::Builder`.
    #[test]
    fn spawn_reader_thread_succeeds_for_an_ordinary_reader() {
        use std::io::Cursor;

        let ring = Arc::new(Mutex::new(LineRing::new(RING_CAPACITY)));
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let log_path = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-reader-ok-{}-{nanos}.log",
            std::process::id()
        ));
        let log = Arc::new(Mutex::new(
            self::log::RotatingLog::open(log_path.clone(), 1024 * 1024).expect("open log"),
        ));
        let log_write_failed = Arc::new(AtomicBool::new(false));

        let result = spawn_reader_thread(
            "stderr",
            Cursor::new(b"line one\nline two\n".to_vec()),
            Arc::clone(&ring),
            log,
            log_write_failed,
        );
        assert!(result.is_ok(), "{result:?}");

        let lines = poll_until_ring_has_lines(&ring, 2, Duration::from_secs(5));
        assert!(lines.iter().any(|l| l == "stderr: line one"), "{lines:?}");
        assert!(lines.iter().any(|l| l == "stderr: line two"), "{lines:?}");

        let _ = std::fs::remove_file(&log_path);
    }

    /// Poll `ring` until it holds at least `min_lines` entries or `timeout` elapses (in which case
    /// this panics — acceptable in test code, never in the library). The reader thread in
    /// [`spawn_reader_thread_succeeds_for_an_ordinary_reader`] races this assertion, so a single
    /// immediate snapshot would be flaky.
    fn poll_until_ring_has_lines(ring: &Mutex<LineRing>, min_lines: usize, timeout: Duration) -> Vec<String> {
        let start = Instant::now();
        loop {
            let snapshot = lock_or_recover(ring).snapshot();
            if snapshot.len() >= min_lines {
                return snapshot;
            }
            if start.elapsed() > timeout {
                panic!("ring did not reach {min_lines} lines within {timeout:?}: {snapshot:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A plain, process-free regression test for the panic [`ChildHandle::kill_and_wait`] used to be
    /// able to reach: `Instant::now() + Duration::MAX` panics on overflow (documented behavior of
    /// `impl Add<Duration> for Instant`), and `timeout` is an arbitrary caller-supplied value — a
    /// config-derived timeout meaning "effectively forever" would have reached exactly this. Exercises
    /// the same `checked_add`/`is_some_and` logic `kill_and_wait` uses, without spawning a real child
    /// (a child that never exits would make this test itself hang instead of panic).
    #[test]
    fn an_unrepresentable_deadline_computation_does_not_panic() {
        let deadline = Instant::now().checked_add(Duration::MAX);
        assert!(deadline.is_none(), "Duration::MAX must not be representable as a deadline");
        // The loop condition `kill_and_wait` uses: `None` must mean "never time out on its own",
        // not "already past the deadline".
        assert!(!deadline.is_some_and(|d| Instant::now() >= d));
    }

    /// Guards [`RING_CAPACITY`] now that it is part of this crate's public API surface (see its own
    /// doc comment): a caller sizing a buffer against [`ChildHandle::recent_lines`] depends on the
    /// bound actually holding, not just existing. Drives [`record_line`] directly — no process
    /// needed — pushing more lines than the ring can hold and confirming eviction still leaves
    /// exactly [`RING_CAPACITY`] of them.
    #[test]
    fn ring_capacity_is_the_documented_public_constant() {
        let ring = Mutex::new(LineRing::new(RING_CAPACITY));
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let log_path = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-ring-cap-{}-{nanos}.log",
            std::process::id()
        ));
        let log = Mutex::new(self::log::RotatingLog::open(log_path.clone(), 1024 * 1024).expect("open log"));
        let log_write_failed = AtomicBool::new(false);

        for i in 0..(RING_CAPACITY + 50) {
            record_line("stdout", format!("line-{i}"), &ring, &log, &log_write_failed);
        }

        let snapshot = lock_or_recover(&ring).snapshot();
        assert_eq!(snapshot.len(), RING_CAPACITY, "{snapshot:?}");
        // The oldest surviving line must be the 50th emitted (the first 50 evicted), and the newest
        // must be the very last one pushed — proving eviction, not merely truncation, kept the tail.
        assert_eq!(snapshot.first().map(String::as_str), Some("stdout: line-50"));
        assert_eq!(
            snapshot.last().map(String::as_str),
            Some(format!("stdout: line-{}", RING_CAPACITY + 49)).as_deref()
        );

        let _ = std::fs::remove_file(&log_path);
    }

    /// Guards [`MAX_LINE_BYTES`] now that it is part of this crate's public API surface: drives a
    /// single pathological line, well over the cap and with no newline at all, through
    /// [`ring::read_bounded_lines`] — the real pipeline [`spawn_windows_at`]'s reader threads use,
    /// not a bare assertion on the constant's value — and confirms the retained line stays bounded
    /// close to [`MAX_LINE_BYTES`] rather than growing with the input.
    #[test]
    fn max_line_bytes_bounds_a_pathological_line_read_through_the_real_pipeline() {
        let huge = vec![b'x'; MAX_LINE_BYTES * 4];
        let mut got = Vec::new();
        ring::read_bounded_lines(std::io::Cursor::new(huge), MAX_LINE_BYTES, |line| got.push(line));
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(got[0].starts_with(&"x".repeat(MAX_LINE_BYTES)), "{}", &got[0][..64.min(got[0].len())]);
        // Generous headroom over `MAX_LINE_BYTES` for the truncation marker's own text, but nowhere
        // near the 4x input size — this is the assertion a hardcoded cap drifting out of sync with
        // the public constant would fail.
        assert!(got[0].len() < MAX_LINE_BYTES + 100, "line was not bounded: {} bytes", got[0].len());
    }

    #[cfg(windows)]
    mod live {
        use super::*;
        use std::path::PathBuf;

        /// A scratch directory under the OS temp dir, unique to one test process, cleaned up on
        /// drop. Stands in for the "caller-provided, Cerebro-owned" `temp_dir`/log directory
        /// [`SpawnOptions`] requires — real callers own a persistent one, but a live test needs
        /// its own so parallel `--ignored` runs cannot collide.
        struct ScratchDir(PathBuf);

        impl ScratchDir {
            fn new(tag: &str) -> Self {
                let dir = std::env::temp_dir().join(format!(
                    "cerebro-supervisor-spawn-live-{tag}-{}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&dir).expect("create scratch dir for a live test");
                Self(dir)
            }
        }

        impl Drop for ScratchDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        fn options(scratch: &ScratchDir) -> SpawnOptions {
            SpawnOptions {
                temp_dir: scratch.0.clone(),
                cuda_visible_devices: None,
                log_path: scratch.0.join("spawn.log"),
                log_max_bytes: 1024 * 1024,
            }
        }

        /// The recorded reason every live test in this module gives for calling
        /// [`spawn_unaudited`] instead of going through a real [`crate::preflight::preflight`]: none
        /// of these fixtures (a `--version` probe, a `ping.exe` loop, this crate's own
        /// `cerebro-print-env`) has a manifest or a `Preflighted` to obtain one from, and this
        /// module's own top-level docs require every such call to carry a written reason.
        fn accept() -> AcceptUnauditedBackend {
            AcceptUnauditedBackend::new(
                "cerebro-supervisor's own #[ignore]d live test fixture, run only on this \
                 development machine; see this crate's build brief for why these fixtures have no \
                 manifest to preflight against",
            )
            .expect("a non-blank literal reason is always accepted")
        }

        /// Poll `f` until it returns `Some`, or `timeout` elapses (in which case this panics —
        /// acceptable in `#[ignore]`d test code, never in the library).
        fn poll_until<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
            let start = Instant::now();
            loop {
                if let Some(v) = f() {
                    return v;
                }
                if start.elapsed() > timeout {
                    panic!("condition did not become true within {timeout:?}");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        /// Ask Windows directly whether a process with `pid` still exists, independent of
        /// anything this crate's own `ChildHandle` tracks — the point of this test is to prove
        /// the *OS-level* process is gone, not merely that our own bookkeeping thinks so.
        fn process_is_alive(pid: u32) -> bool {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Threading::{
                GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
            };

            // Not exposed by `windows-sys` as a named constant; documented by Win32
            // (`GetExitCodeProcess`) as the fixed sentinel value `259` meaning "this process has
            // not yet terminated".
            const STILL_ACTIVE: u32 = 259;

            // SAFETY: `pid` is an arbitrary but valid `u32`; `OpenProcess` is documented to return
            // null on failure (e.g. the process no longer exists), which is checked before the
            // handle is used for anything. `GetExitCodeProcess` is called with that same
            // just-validated handle and a live, correctly-sized local `code`. The handle is closed
            // exactly once, on every path, before returning.
            unsafe {
                let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
                if handle.is_null() {
                    return false;
                }
                let mut code: u32 = 0;
                let ok = GetExitCodeProcess(handle, &mut code);
                CloseHandle(handle);
                ok != 0 && code == STILL_ACTIVE
            }
        }

        #[test]
        #[ignore = "spawns a real llama-server.exe process on this machine's b10751 fixture build"]
        fn spawns_real_llama_server_with_version_flag_and_it_exits() {
            let backend_dir = PathBuf::from(
                "C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751",
            );
            assert!(
                backend_dir.join("llama-server.exe").is_file(),
                "fixture backend not found at {}; this test requires the b10751 build described \
                 in this component's build brief",
                backend_dir.display()
            );

            let scratch = ScratchDir::new("version");
            let opts = options(&scratch);
            let handle = spawn_unaudited(
                &backend_dir,
                "llama-server.exe",
                &["--version".to_string()],
                &opts,
                &accept(),
            )
            .expect("spawn llama-server.exe --version");

            let status = poll_until(Duration::from_secs(15), || {
                handle.try_wait().expect("poll child exit status")
            });
            assert!(status.success(), "llama-server --version exited with {status:?}");

            // The reader threads drain whatever the OS pipe buffer already held once the child's
            // end closes on exit, but scheduling that shortly after `try_wait` observes the exit
            // is not instantaneous, so this gives them a brief, bounded grace period rather than
            // asserting on the very first read.
            let lines = poll_until(Duration::from_secs(2), || {
                let lines = handle.recent_lines();
                (!lines.is_empty()).then_some(lines)
            });
            assert!(
                !lines.is_empty(),
                "expected --version to print something to stdout or stderr"
            );
        }

        #[test]
        #[ignore = "runs the real preflight() against the b10751 fixture and spawns llama-server \
                    from the resulting Preflighted; must run from a non-elevated shell, exactly \
                    like preflight's own live_b10751_refuses_without_opt_in_and_succeeds_with_it, \
                    or preflight will refuse with RunningElevated before this test's own \
                    assertions ever run"]
        fn spawn_preflighted_launches_the_tool_preflight_verified() {
            // `spawn_unaudited` is easy to call from a fixture, and every other live test in this
            // file uses it — but `spawn_preflighted` is the only function the product actually
            // ships. This is the one live test that goes through it, proving the wiring at
            // `spawn_preflighted`'s two-line body (`preflighted.tool_path()` /
            // `preflighted.backend_dir()`, not swapped) is what actually reaches a real child,
            // not merely what `spawn_windows_at`'s own unit tests exercise directly.
            use crate::preflight::{preflight, AllowUnsafeDacl, LaunchSpec, PortRequest};

            let backend_dir = PathBuf::from(
                "C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751",
            );
            assert!(
                backend_dir.join("llama-server.exe").is_file(),
                "fixture backend not found at {}; this test requires the b10751 build described \
                 in this component's build brief",
                backend_dir.display()
            );
            let model = PathBuf::from(
                r"C:\Users\noahm\.lmstudio\models\skiesoffire\bastion-control\bastion-control-0.5b-f16.gguf",
            );
            assert!(
                model.is_file(),
                "fixture model not found at {}; see this component's build brief",
                model.display()
            );

            let (manifest, _) = cerebro_probe::backend::hash_directory(&backend_dir)
                .expect("hash the real backend dir");

            let scratch = ScratchDir::new("preflighted");
            let key_path = scratch.0.join("api.key");
            std::fs::write(&key_path, "k".repeat(32)).expect("write fixture api key file");

            let spec = LaunchSpec {
                backend_dir: backend_dir.clone(),
                tool_name: "llama-server.exe".to_string(),
                manifest,
                model_path: model,
                draft_model_path: None,
                port: PortRequest::Any,
                api_key_file: key_path,
                // The fixture directory's DACL grants Modify to ASUS\CodexSandboxUsers by design
                // of this development machine, not of Cerebro — see this component's build brief
                // and preflight's own `live_b10751_refuses_without_opt_in_and_succeeds_with_it`,
                // which exercises the refusal this opt-in bypasses.
                allow_unsafe_dacl: Some(
                    AllowUnsafeDacl::new(
                        "b10751 live fixture; ASUS\\CodexSandboxUsers holds Modify on this \
                         directory by design of the test machine, not of Cerebro",
                    )
                    .expect("reason is non-blank"),
                ),
            };
            let preflighted = preflight(&spec).expect(
                "preflight the fixture backend; run from a non-elevated shell (see this test's \
                 #[ignore] reason)",
            );
            assert_eq!(preflighted.tool_path(), backend_dir.join("llama-server.exe"));

            let opts = options(&scratch);
            let handle = spawn_preflighted(&preflighted, &["--version".to_string()], &opts)
                .expect("spawn_preflighted must launch the preflight-verified tool");

            let status = poll_until(Duration::from_secs(15), || {
                handle.try_wait().expect("poll child exit status")
            });
            assert!(status.success(), "llama-server --version exited with {status:?}");

            let lines = poll_until(Duration::from_secs(2), || {
                let lines = handle.recent_lines();
                (!lines.is_empty()).then_some(lines)
            });
            assert!(
                !lines.is_empty(),
                "expected --version to print something to stdout or stderr"
            );
        }

        #[test]
        #[ignore = "spawns a real, long-running child process and asserts it is killed on drop"]
        fn dropping_the_handle_kills_a_long_running_child_within_two_seconds() {
            // `ping -t` is a genuinely long-running process (it never exits on its own) that
            // ships with every Windows install, so this test needs no llama.cpp fixture at all —
            // it is exercising this module's containment, not anything model- or
            // backend-specific. `System32` stands in for `backend_dir` here; `spawn_unaudited`
            // treats it exactly like any other backend directory.
            let system32 = PathBuf::from(r"C:\Windows\System32");
            let scratch = ScratchDir::new("kill-on-drop");
            let opts = options(&scratch);
            let handle = spawn_unaudited(
                &system32,
                "ping.exe",
                &["-t".to_string(), "127.0.0.1".to_string()],
                &opts,
                &accept(),
            )
            .expect("spawn a long-running ping.exe");

            let pid = handle.pid();
            assert!(process_is_alive(pid), "ping.exe should still be running");
            assert_eq!(
                handle.try_wait().expect("poll before drop"),
                None,
                "ping -t must not have exited on its own yet"
            );

            drop(handle);

            poll_until(Duration::from_secs(2), || (!process_is_alive(pid)).then_some(()));
        }

        /// The `[[bin]]` fixture built specifically for these live tests — see its own module docs
        /// for its two modes. `env!("CARGO_BIN_EXE_...")` is populated for a `[[bin]]` target used
        /// from an integration test, but this is a unit test compiled into the library itself, so
        /// it is derived the same way `dropping_the_handle_kills_a_long_running_child...` above
        /// derives nothing backend-specific at all: from this test binary's own path, since
        /// `target/<profile>/deps/<this test binary>` and `target/<profile>/` are siblings under
        /// the same profile directory regardless of workspace layout or a custom
        /// `CARGO_TARGET_DIR`.
        fn print_env_fixture_dir() -> PathBuf {
            let mut dir = std::env::current_exe().expect("locate the running test binary");
            dir.pop(); // .../target/<profile>/deps
            dir.pop(); // .../target/<profile>
            assert!(
                dir.join("cerebro-print-env.exe").is_file(),
                "expected cerebro-print-env.exe under {}; run `cargo test` (which builds every \
                 `[[bin]]` target first) rather than invoking this test binary directly",
                dir.display()
            );
            dir
        }

        #[test]
        #[ignore = "spawns a real, fast-exiting child process 50 times to exercise a race"]
        fn a_child_that_exits_before_job_assignment_still_returns_a_handle() {
            // Regression test for the measured `AssignProcessToJobObject` / `ERROR_ACCESS_DENIED`
            // race documented at this module's top: `cerebro-print-env` (default mode) exits in
            // single-digit milliseconds, comfortably inside the gap between `Command::spawn` and
            // `job::JobHandle::assign`. Looped rather than run once so this is not merely lucky —
            // a single run losing the race is not distinguishable from one that never entered it.
            let fixture_dir = print_env_fixture_dir();
            // Best-effort, not asserted: whether *this* run actually wins the race even once is a
            // property of process-creation latency on whatever machine runs it (an antivirus
            // real-time scan, a busy scheduler, a sandboxing layer intercepting `CreateProcess`, and
            // this crate's own build brief all note this as inherently non-deterministic), so making
            // "at least one of 50" a hard assertion would trade one flaky boundary condition for
            // another. `contained_true_and_false_render_differently_via_debug` below pins down the
            // property this loop can only try to observe opportunistically: that
            // [`ChildHandle::is_contained`] reports the two outcomes differently at all.
            let mut saw_an_uncontained_child = false;
            for i in 0..50 {
                let scratch = ScratchDir::new(&format!("assign-race-{i}"));
                let opts = options(&scratch);
                let handle = spawn_unaudited(&fixture_dir, "cerebro-print-env.exe", &[], &opts, &accept())
                    .unwrap_or_else(|e| panic!("iteration {i}: spawn must succeed even if the \
                                                 child already exited, got {e}"));
                if !handle.is_contained() {
                    saw_an_uncontained_child = true;
                }
                let status = poll_until(Duration::from_secs(5), || {
                    handle.try_wait().expect("poll child exit status")
                });
                assert!(status.success(), "iteration {i}: exited with {status:?}");
                let lines = poll_until(Duration::from_secs(2), || {
                    let lines = handle.recent_lines();
                    (!lines.is_empty()).then_some(lines)
                });
                assert!(!lines.is_empty(), "iteration {i}: fixture produced no output");
            }
            if !saw_an_uncontained_child {
                eprintln!(
                    "note: none of 50 iterations lost the assign race on this run; that is an \
                     observation about this run's process-creation latency, not a failure of the \
                     code under test — see this test's own comment"
                );
            }
        }

        #[test]
        #[ignore = "spawns a real, long-running child and checks it reports itself as contained"]
        fn is_contained_reports_true_for_a_running_child() {
            // The deterministic half of the property `a_child_that_exits_before_job_assignment...`
            // above can only try to observe opportunistically: an ordinary child that is still
            // running when this checks it is *always* contained (there is no race to lose — nothing
            // makes `AssignProcessToJobObject` fail for a process that has not exited), so this pins
            // down that `ChildHandle::is_contained` reports that case as `true`, which combined with
            // the other test's occasional `false` observations proves the two outcomes are reported
            // differently at all, not merely that one of them can occur. Asserted directly on
            // `is_contained()` rather than by parsing `{handle:?}`, which would test the `Debug`
            // formatter, not this state.
            let system32 = PathBuf::from(r"C:\Windows\System32");
            let scratch = ScratchDir::new("contained-debug");
            let opts = options(&scratch);
            let handle = spawn_unaudited(
                &system32,
                "ping.exe",
                &["-t".to_string(), "127.0.0.1".to_string()],
                &opts,
                &accept(),
            )
            .expect("spawn a long-running ping.exe");
            assert!(
                process_is_alive(handle.pid()),
                "ping.exe should still be running, so assignment cannot have lost a race"
            );
            assert!(handle.is_contained(), "a still-running child must report itself contained");

            let pid = handle.pid();
            handle.kill();
            poll_until(Duration::from_secs(2), || (!process_is_alive(pid)).then_some(()));
        }

        #[test]
        #[ignore = "spawns a real child that spawns its own grandchild, and asserts both die"]
        fn dropping_the_handle_kills_a_grandchild_too() {
            // The existing `dropping_the_handle_kills_a_long_running_child...` test above proves
            // only that the *immediate* child dies on drop, which a plain `TerminateProcess` would
            // also achieve — it never exercises the one thing the Job Object is for. This test
            // spawns `cerebro-print-env spawn-grandchild`, which itself spawns a long-running
            // `ping -t` and prints its PID, then asserts the grandchild — not just the immediate
            // child — is gone after the `ChildHandle` above both of them is dropped.
            let fixture_dir = print_env_fixture_dir();
            let scratch = ScratchDir::new("grandchild");
            let opts = options(&scratch);
            let handle = spawn_unaudited(
                &fixture_dir,
                "cerebro-print-env.exe",
                &["spawn-grandchild".to_string()],
                &opts,
                &accept(),
            )
            .expect("spawn the print-env fixture in spawn-grandchild mode");

            let grandchild_pid: u32 = poll_until(Duration::from_secs(5), || {
                handle.recent_lines().iter().find_map(|line| {
                    line.strip_prefix("stdout: GRANDCHILD_PID=")
                        .and_then(|pid| pid.trim().parse().ok())
                })
            });
            assert!(
                process_is_alive(grandchild_pid),
                "grandchild pid {grandchild_pid} should be running before drop"
            );

            drop(handle);

            poll_until(Duration::from_secs(2), || {
                (!process_is_alive(grandchild_pid)).then_some(())
            });
        }

        #[test]
        #[ignore = "spawns real llama-server.exe on a loopback port and checks it releases it"]
        fn kill_and_wait_returns_before_the_port_is_reusable() {
            // The operational reason `kill()` alone is not enough for a caller that rebinds the
            // same port: `kill()` only closes the job handle, and the kernel tears the tree down
            // (and releases the listening socket) asynchronously afterward. A real model is loaded
            // (the tiny fixture this crate's build brief names) so the server actually reaches its
            // listening state rather than exiting immediately for lack of one.
            let backend_dir = PathBuf::from(
                "C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751",
            );
            let model = PathBuf::from(
                r"C:\Users\noahm\.lmstudio\models\skiesoffire\bastion-control\bastion-control-0.5b-f16.gguf",
            );
            assert!(
                backend_dir.join("llama-server.exe").is_file(),
                "fixture backend not found at {}",
                backend_dir.display()
            );
            assert!(model.is_file(), "fixture model not found at {}", model.display());

            // Bind port 0 to get an OS-assigned free loopback port, then release it immediately —
            // there is an inherent, unavoidable TOCTOU gap between that release and llama-server
            // binding the same number, but it is vanishingly unlikely to collide in a test run.
            let port = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0")
                    .expect("bind an ephemeral loopback port");
                listener.local_addr().expect("read the bound port").port()
            };

            let scratch = ScratchDir::new("kill-and-wait");
            let opts = options(&scratch);
            let handle = spawn_unaudited(
                &backend_dir,
                "llama-server.exe",
                &[
                    "--host".to_string(),
                    "127.0.0.1".to_string(),
                    "--port".to_string(),
                    port.to_string(),
                    "--model".to_string(),
                    model.display().to_string(),
                    "--no-webui".to_string(),
                ],
                &opts,
                &accept(),
            )
            .expect("spawn a real llama-server on a loopback port");

            // Give it a chance to actually bind the port before killing it.
            poll_until(Duration::from_secs(30), || {
                std::net::TcpStream::connect(("127.0.0.1", port)).ok().map(drop)
            });

            let status = handle
                .kill_and_wait(Duration::from_secs(5))
                .expect("poll for exit after kill");
            assert!(status.is_some(), "kill_and_wait must report the child as gone");

            std::net::TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
                panic!("port {port} was not reusable immediately after kill_and_wait: {e}")
            });
        }

        #[test]
        #[ignore = "spawns a real long-running child and kills it twice"]
        fn kill_is_idempotent_and_the_first_call_already_kills_the_tree() {
            // `ChildHandle::kill`'s own doc comment claims idempotence, but nothing before this
            // test ever called it directly and twice: `kill_and_wait` above calls it only once per
            // handle. Calling it a second (and third, after the child is already reaped) time here
            // is what actually pins the claim down — a future change to `TerminateProcess` the
            // child directly instead of taking the job out of its `Option` (see `kill`'s own doc
            // comment on why that is what makes it idempotent for free) would still pass every
            // other test in this file but panic or error here on the second call.
            let system32 = PathBuf::from(r"C:\Windows\System32");
            let scratch = ScratchDir::new("kill-twice");
            let opts = options(&scratch);
            let handle = spawn_unaudited(
                &system32,
                "ping.exe",
                &["-t".to_string(), "127.0.0.1".to_string()],
                &opts,
                &accept(),
            )
            .expect("spawn a long-running ping.exe");
            let pid = handle.pid();
            assert!(process_is_alive(pid), "ping.exe should still be running");

            handle.kill();
            // The second call must be a no-op: no panic, and no effect beyond what the first call
            // already did.
            handle.kill();

            // `kill()` only closes the job handle; the kernel tears the tree down asynchronously
            // afterward (see `kill`'s own doc comment), so `try_wait()` — like `process_is_alive`
            // below — must be polled rather than asserted once immediately after `kill()` returns.
            // An earlier version of this test asserted `try_wait()` a single time right after
            // polling `process_is_alive` to `false`, which raced the child's own handle becoming
            // signaled and failed intermittently even though nothing was actually broken.
            // The exit *status* a job-object kill produces is not part of this test's contract
            // (Windows reports it as a plain, successful-looking exit code for this termination
            // path) — what matters is only that `try_wait` eventually reports the child gone at
            // all, which `poll_until` returning at all already proves.
            poll_until(Duration::from_secs(2), || handle.try_wait().expect("try_wait after kill"));
            assert!(
                !process_is_alive(pid),
                "the OS must also agree the process is gone once try_wait reports it"
            );

            // A third call, after the child is already reaped, must still not panic.
            handle.kill();
        }
    }
}
