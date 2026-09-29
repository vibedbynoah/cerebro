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
//! 2. **A closed environment.** [`cerebro_probe::child_env::build_child_env`] rebuilds the child's
//!    entire environment block from nothing; see that module's docs for why an inherited
//!    `GGML_BACKEND_PATH` would load a DLL from outside the audited directory and an inherited
//!    `LLAMA_ARG_*` would bypass the flag allowlist
//!    [`crate::preflight::build_hardened_server_argv`] already proved safe. It is the one definition
//!    of a backend child's environment: `cerebro_probe::backend::probe_with`'s queries use the same
//!    function under the same setting, so the two cannot drift. User
//!    switchable: the one [`crate::protections::Protections`] value for this launch — read from
//!    [`crate::preflight::Preflighted::protections`] for [`spawn_preflighted`], or passed directly
//!    to [`spawn_unaudited`], never a second, independent copy on [`SpawnOptions`] that could
//!    disagree with it — having [`crate::protections::Protections::enforce_environment_allowlist`]
//!    off does not simply hand the child Cerebro's raw ambient environment — see
//!    [`effective_child_env`] for what still gets re-pinned even then — and is recorded as a
//!    [`ChildHandle::protection_warnings`] entry.
//! 3. **A process tree that cannot outlive Cerebro.** `job::JobHandle` assigns the child to a
//!    Windows Job Object armed with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; see that module's docs
//!    for why a bare child PID is not enough once `llama-server` can itself spawn an `--rpc`
//!    worker (ARCHITECTURE.md section 5). Also user switchable, the same way and for the same
//!    reason as item 2: [`crate::protections::Protections::contain_in_job_object`] off leaves the
//!    child unassigned to any job — though [`ChildHandle`]'s own `Drop` still terminates the
//!    immediate process in that case; see its documentation.
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
///
/// Also deliberately excludes [`crate::protections::Protections`]: an earlier version of this type
/// carried its own copy, which let a caller preflight a launch with every protection on and then
/// spawn it with, say, job containment off — two independent settings for one launch that could
/// disagree with each other. There is now exactly one [`crate::protections::Protections`] value per
/// launch: [`spawn_preflighted`] reads it from [`crate::preflight::Preflighted::protections`], and
/// [`spawn_unaudited`] (which has no `Preflighted` to read it from) takes it as an explicit
/// parameter instead.
#[derive(Debug, Clone)]
pub struct SpawnOptions {
    /// Directory pinned as `TEMP`/`TMP` in the child's environment. Must already exist and be
    /// owned by Cerebro — see [`cerebro_probe::child_env`]'s module docs for why this is never copied from the parent
    /// process's own `TEMP`/`TMP`. Enforced, not merely assumed: `spawn_windows_at` refuses a
    /// missing path, a non-directory, or (while
    /// [`crate::protections::Protections::refuse_redirected_paths`] is on) a path with a junction or
    /// symlink anywhere on it, with [`SpawnError::UnsafeDir`] before opening anything or spawning a
    /// child — see `verify_owned_dir` in this module. With that switch off the link is followed and
    /// the child's `TEMP`/`TMP` is the resolved target, never the link.
    pub temp_dir: PathBuf,
    /// Directory GPU drivers keep compiled kernels and pipelines in for this child:
    /// `CUDA_CACHE_PATH` is `<cache_dir>\cuda`, and `LOCALAPPDATA`/`APPDATA` are `cache_dir` itself,
    /// in both states of the environment allowlist (see [`cerebro_probe::child_env`]'s "Driver
    /// caches" for the measured cost of dropping them: 19-22 s per launch against 1.2-1.4 s warm with
    /// PTX JIT forced). Checked exactly like `temp_dir` (see that field) and refused with
    /// [`SpawnError::UnsafeDir`] rather than dropped. Use one directory per backend build, and give
    /// `cerebro_probe::backend::probe_with_cache` the same one (after [`verify_cache_dir`]) so
    /// identifying the backend warms the cache this launch reads.
    pub cache_dir: PathBuf,
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

    /// [`cerebro_probe::child_env::system_directory`] could not resolve the real Windows system
    /// directory from the kernel. Without it there is no trustworthy value for
    /// `PATH`/`SystemRoot`/`windir` to build on — see [`cerebro_probe::child_env`]'s module docs on
    /// why those are constructed, never copied from the parent.
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

/// A running, ordinarily job-contained child process, plus its bounded diagnostic state.
///
/// When this launch had [`crate::protections::Protections::contain_in_job_object`] on (the
/// default), dropping a `ChildHandle` kills the entire process tree it heads — see
/// `job::JobHandle`'s documentation for the kernel mechanism this relies on — and
/// [`ChildHandle::kill`] triggers the same thing early, without waiting for the value itself to go
/// out of scope. With that protection turned off for this launch, no Job Object exists, so nothing
/// but the immediate process is ever guaranteed to stop — but this type's own `Drop` impl (below)
/// still terminates that immediate process on an ordinary drop, exactly as [`ChildHandle::kill`]
/// would, so a launch that turned job containment off is never worse than "the immediate process
/// leaks" for every path that runs Rust drop glue at all (cancellation, an early return, a normal
/// panic unwind). Only Cerebro itself being killed or crashing — where no Drop glue runs anywhere
/// in the process — still leaves that immediate process (and anything *it* spawned) running; see
/// [`crate::protections::Protections::contain_in_job_object`]'s own documentation for that residual
/// risk, which only a real Job Object closes.
pub struct ChildHandle {
    child: Mutex<Child>,
    /// `None` once [`ChildHandle::kill`] has run; see that method's documentation for why this is
    /// what makes it idempotent. Absent entirely on non-Windows, where neither
    /// [`spawn_preflighted`] nor [`spawn_unaudited`] ever constructs a `ChildHandle` in the first
    /// place (see [`SpawnError::UnsupportedPlatform`]).
    #[cfg(windows)]
    job: Mutex<Option<job::JobHandle>>,
    ring: Arc<Mutex<LineRing>>,
    /// The spawn-owned protections ([`crate::protections::Protections::enforce_environment_allowlist`],
    /// [`crate::protections::Protections::contain_in_job_object`]) that the [`crate::protections::Protections`]
    /// value passed to [`spawn_preflighted`] or [`spawn_unaudited`] had turned off for this launch,
    /// each as a structured warning. Empty when both were on. For [`spawn_unaudited`] only, also
    /// [`crate::protections::Protections::refuse_redirected_paths`] when it is off; a preflighted
    /// launch already carries that one in [`crate::preflight::Preflighted::warnings`], so the
    /// launch record lists it once. See [`ChildHandle::protection_warnings`].
    protection_warnings: Vec<crate::protections::ProtectionWarning>,
}

impl ChildHandle {
    /// The child's process ID.
    pub fn pid(&self) -> u32 {
        lock_or_recover(&self.child).id()
    }

    /// The spawn-owned protections this launch turned off, each as a structured warning — the
    /// spawn-time half of the launch record; see [`crate::preflight::Preflighted::warnings`] for the
    /// preflight-time half and [`crate::protections::Protections`] for the full owner rule. Empty
    /// when every protection this launch's [`crate::protections::Protections`] value governs was on.
    pub fn protection_warnings(&self) -> &[crate::protections::ProtectionWarning] {
        &self.protection_warnings
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
    /// `false` in three cases: the documented assign-after-exit race this module's top-level docs
    /// describe (a child that exits in the narrow window between `Command::spawn` and
    /// `AssignProcessToJobObject` is not a containment failure — there is nothing left to contain —
    /// but it does mean no Job Object exists for this handle); this launch having
    /// [`crate::protections::Protections::contain_in_job_object`] turned off, which skips creating a
    /// Job Object at all (see [`spawn_windows_at`]); or once [`ChildHandle::kill`] has run, for the
    /// same reason `kill` is idempotent — see that method's documentation. The second case is the
    /// only one that is not itself a race or a one-way transition: it holds for this handle's entire
    /// life.
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

    /// Kill the child — and, when this launch was assigned to a Job Object, everything it spawned
    /// too — without waiting for this handle to be dropped.
    ///
    /// Closes the job handle first, when one exists, which tells the kernel to terminate the whole
    /// assigned process tree; it does not itself wait for that termination to complete. A caller
    /// whose next move depends on the child truly being gone — most importantly, relaunching on the
    /// same `--port`, which races the old process's listening socket if the kernel has not yet torn
    /// it down — must call [`ChildHandle::kill_and_wait`] instead.
    ///
    /// Also always kills the immediate child process directly, whether or not a Job Object exists
    /// for this handle. `job` can be `None` for two different reasons — the documented
    /// assign-after-exit race (nothing left alive to kill anyway), or
    /// [`crate::protections::Protections::contain_in_job_object`] having been turned off for a
    /// launch that is very much still running — and only the direct kill here covers the second
    /// case; relying on the job alone would leave this method doing nothing at all for a launch
    /// that turned that one protection off.
    ///
    /// Idempotent: a second call, from any thread, after the first has already run (or after the
    /// child already exited on its own) does nothing beyond re-attempting a kill on an
    /// already-gone process, which the OS reports as an ordinary (silently ignored) error rather
    /// than a panic — never surfacing one for a process that is already gone.
    #[cfg(windows)]
    pub fn kill(&self) {
        // Taking the job out of the `Option` and dropping it here — rather than calling some
        // inverse of `AssignProcessToJobObject` — is what makes the job-handle half of this
        // idempotent for free: a second call finds `None` already and does nothing more here. See
        // `job::JobHandle`'s `Drop` impl for the kernel mechanism this triggers.
        let mut guard = lock_or_recover(&self.job);
        drop(guard.take());
        // Always also kill the immediate process directly — see this method's own documentation for
        // why the job handle alone is not enough once job containment can be turned off.
        let _ = lock_or_recover(&self.child).kill();
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

impl Drop for ChildHandle {
    /// Guarantee the immediate child never outlives an ordinary drop of this handle, even when this
    /// launch turned [`crate::protections::Protections::contain_in_job_object`] off.
    ///
    /// When a Job Object exists (the default), it does the work: dropping `self.job`'s `Option`
    /// drops the `job::JobHandle` inside it, which closes the kernel handle and triggers
    /// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` for the whole assigned tree — this function's own
    /// explicit kill below would be redundant in that case, so it is skipped. When no Job Object
    /// exists — containment turned off for this launch, or the documented assign-after-exit race —
    /// nothing else here kills anything on an ordinary drop, which is exactly the gap a lifecycle
    /// task cancelled on runtime shutdown, an early return that drops [`ChildHandle`], or a panic
    /// unwind could fall into (see [`ChildHandle`]'s own top-level documentation): this still
    /// terminates the immediate process, if it is still running, before returning. It does not wait
    /// for the OS to confirm the kill — a caller whose next step depends on that must call
    /// [`ChildHandle::kill_and_wait`] explicitly instead, exactly as before this impl existed.
    fn drop(&mut self) {
        #[cfg(windows)]
        let contained = lock_or_recover(&self.job).is_some();
        #[cfg(not(windows))]
        let contained = false;
        if !contained {
            let mut child = lock_or_recover(&self.child);
            if matches!(child.try_wait(), Ok(None)) {
                let _ = child.kill();
            }
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
/// against is that [`cerebro_probe::child_env::build_child_env`] concatenates `backend_dir` into `PATH` with `;` as the
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

/// Whether a temp or log directory reached through a junction or symlink is refused or followed:
/// the spawn-side half of [`crate::protections::Protections::refuse_redirected_paths`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirLinks {
    /// Refuse a reparse point anywhere on the path. The default.
    Refuse,
    /// Follow every link, then use the directory's real path, read back from the handle that was
    /// opened and checked, so the child never gets the link.
    Follow,
}

/// Check a caller-supplied directory before this module opens or writes through it, and return the
/// path to use for it: `dir` itself under [`DirLinks::Refuse`], the resolved target under
/// [`DirLinks::Follow`]. Missing, relative or not-a-directory is refused either way.
///
/// Under [`DirLinks::Refuse`], walks every path component from the filesystem root down to `dir`
/// itself with [`std::fs::symlink_metadata`] — never [`std::fs::metadata`], which follows a reparse
/// point and so could never see the one thing this check exists to catch — exactly the reasoning
/// `cerebro_probe::backend::dacl::check_path_chain` uses for the backend directory's own DACL audit.
/// A leaf-only check (an earlier version of this function) catches a junction *at* `dir` but misses
/// one anywhere in its ancestors (`C:\cerebro\<junction>\temp`), which redirects everything Cerebro
/// writes to `dir` just as effectively.
///
/// Under [`DirLinks::Follow`] (the user turned the redirected-path refusal off, e.g. because
/// `%LOCALAPPDATA%` was moved to another drive with a junction), the directory is opened once,
/// following every link, checked through that handle, and its path read back from the same handle
/// ([`crate::preflight::win_path::final_path`]), so the child is handed the directory that was
/// checked. The `\\?\` prefix is dropped for a drive-letter or UNC result (see [`plain_form`]),
/// since `TEMP`/`TMP` reach programs that do not expect it.
///
/// Deliberately does *not* also run the DACL audit `crate::preflight` runs on `backend_dir`: unlike
/// the backend directory (a long-lived, shared install ggml loads DLLs from), `temp_dir` and the log
/// directory are per-launch, Cerebro-created scratch directories, and this crate's own `preflight`
/// module docs record that this exact development machine's `%TEMP%` already has a DACL that would
/// fail a real audit closed for every fixture rooted under it — repeating that audit here would
/// refuse every ordinary caller-supplied scratch directory on this machine, not just a hostile one.
/// This check instead targets the concrete redirection hazard: a junction or symlink silently
/// retargeting where the child's `TEMP`/`TMP` or crash log actually lands.
#[cfg(windows)]
fn verify_owned_dir(
    role: &'static str,
    dir: &Path,
    links: DirLinks,
) -> Result<PathBuf, SpawnError> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
    };

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

    if links == DirLinks::Follow {
        // `FILE_FLAG_BACKUP_SEMANTICS` is what lets `CreateFileW` open a directory at all; without
        // `FILE_FLAG_OPEN_REPARSE_POINT` every link on the way, the last one included, is followed.
        let handle = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(dir)
            .map_err(|e| bad(e.to_string()))?;
        if !handle.metadata().map_err(|e| bad(e.to_string()))?.is_dir() {
            return Err(bad("not a directory".to_string()));
        }
        let resolved =
            crate::preflight::win_path::final_path(&handle).map_err(|e| bad(e.to_string()))?;
        return Ok(plain_form(resolved));
    }

    let mut chain: Vec<&Path> = dir.ancestors().collect();
    chain.reverse(); // filesystem root first, `dir` itself last.

    let mut dir_meta = None;
    for component in chain {
        let meta = std::fs::symlink_metadata(component).map_err(|e| bad(e.to_string()))?;
        if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(bad(format!(
                "{} is a reparse point (junction or symlink); every component of the path used \
                 to reach it must be the real object, not a redirect ({})",
                component.display(),
                crate::protections::Protection::RedirectedPathRefusal.settings_hint()
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
    Ok(dir.to_path_buf())
}

/// The role name [`SpawnError::UnsafeDir`] reports for [`SpawnOptions::cache_dir`].
#[cfg(windows)]
const CACHE_DIR_ROLE: &str = "driver cache directory";

/// Check a driver cache directory the way a launch will ([`SpawnOptions::cache_dir`]) and return
/// the path to use: `dir` itself while [`crate::protections::Protections::refuse_redirected_paths`]
/// is on (a junction anywhere on it is refused), its resolved target while it is off. Missing,
/// relative or not-a-directory is refused either way.
///
/// For callers that start the backend outside a launch: pass the result to
/// `cerebro_probe::backend::probe_with_cache`, so the probe's queries and the server share one
/// checked cache and a junction the user did not allow never reaches either.
pub fn verify_cache_dir(
    dir: &Path,
    protections: &crate::protections::Protections,
) -> Result<PathBuf, SpawnError> {
    #[cfg(windows)]
    {
        let links = if protections.refuse_redirected_paths {
            DirLinks::Refuse
        } else {
            DirLinks::Follow
        };
        verify_owned_dir(CACHE_DIR_ROLE, dir, links)
    }
    #[cfg(not(windows))]
    {
        let _ = (dir, protections);
        Err(SpawnError::UnsupportedPlatform)
    }
}

/// `path` (from `GetFinalPathNameByHandleW`, so `\\?\`-prefixed) in its plain DOS form when it has
/// one: `\\?\C:\x` becomes `C:\x` and `\\?\UNC\srv\share\x` becomes `\\srv\share\x`. A path with no
/// plain form (a volume-GUID path) is returned unchanged; it is still absolute and names the same
/// directory. Reassembled from [`Path::components`], never by string surgery, so no path can turn
/// relative here (`\\?\UNC\...` stripped by string would become the relative `UNC\...`).
#[cfg(windows)]
fn plain_form(path: PathBuf) -> PathBuf {
    use std::path::{Component, Prefix};

    let mut components = path.components();
    let rebuilt = match components.next() {
        Some(Component::Prefix(prefix)) => match prefix.kind() {
            Prefix::VerbatimDisk(drive) => {
                if !matches!(components.next(), Some(Component::RootDir)) {
                    return path;
                }
                let mut out = PathBuf::from(format!("{}:\\", char::from(drive)));
                out.extend(components.map(|c| c.as_os_str()));
                out
            }
            Prefix::VerbatimUNC(server, share) => {
                let mut out = PathBuf::from(r"\\");
                out.push(server);
                out.push(share);
                out.extend(components.map(|c| c.as_os_str()));
                out
            }
            _ => return path,
        },
        _ => return path,
    };
    rebuilt
}

/// Launch `tool_name` (resolved as `backend_dir.join(tool_name)`) as a job-contained child, with a
/// closed environment, no window, and bounded/logged output capture — without requiring a
/// [`crate::preflight::Preflighted`] proof.
///
/// `protections` is taken explicitly, not read from `options` (which no longer carries a copy at
/// all — see [`SpawnOptions`]'s own docs): a caller with no [`crate::preflight::Preflighted`] to
/// source it from must still state the one value this launch's environment and containment
/// decisions are made against.
///
/// `accept` is not optional and not free to construct: [`AcceptUnauditedBackend::new`] refuses a
/// blank reason, so every call to this function is attributable to someone's stated judgment call.
/// It is not itself inspected here beyond that — this function does not log or persist it — it
/// exists so the *call site* carries a recorded reason a reviewer can hold someone to, the same
/// discipline [`crate::preflight::AllowUnsafeDacl`] applies one layer up.
///
/// `backend_dir` is used both as the child's `current_dir` and as the sole non-system entry of its
/// constructed `PATH` (see [`cerebro_probe::child_env`]'s module docs) — the same "spawn is pinned to the audited
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
    protections: crate::protections::Protections,
    accept: &AcceptUnauditedBackend,
) -> Result<ChildHandle, SpawnError> {
    let _ = accept;
    #[cfg(windows)]
    {
        verify_spawn_target(backend_dir, tool_name)?;
        spawn_windows_at(
            &backend_dir.join(tool_name),
            backend_dir,
            args,
            options,
            protections,
            RedirectRecord::Here,
        )
    }
    #[cfg(not(windows))]
    {
        let _ = (backend_dir, tool_name, args, options, protections);
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
/// Likewise reads [`crate::preflight::Preflighted::protections`] rather than taking a second,
/// independent value from the caller: preflight and spawn must agree on exactly one
/// [`crate::protections::Protections`] per launch, never two copies that could disagree — see
/// [`SpawnOptions`]'s own docs.
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
            preflighted.protections(),
            RedirectRecord::Preflight,
        )
    }
    #[cfg(not(windows))]
    {
        let _ = (preflighted, args, options);
        Err(SpawnError::UnsupportedPlatform)
    }
}

/// Where [`crate::protections::Protections::refuse_redirected_paths`] being off is recorded for one
/// launch. The switch is enforced by both preflight (model and key files) and spawn (temp and log
/// directories), but the launch record must list it once.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RedirectRecord {
    /// [`crate::preflight::Preflighted::warnings`] already carries it ([`spawn_preflighted`]).
    Preflight,
    /// Nothing else records it, so [`ChildHandle::protection_warnings`] does ([`spawn_unaudited`]).
    Here,
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
    protections: crate::protections::Protections,
    redirect_record: RedirectRecord,
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
    // discovered once the child was already writing through it. With the redirected-path refusal
    // off, a link is followed and the resolved target is what the child and the log get.
    let links = if protections.refuse_redirected_paths {
        DirLinks::Refuse
    } else {
        DirLinks::Follow
    };
    let bad_log_path = || SpawnError::UnsafeDir {
        role: "crash log path",
        path: options.log_path.display().to_string(),
        cause: "must be an absolute path inside a Cerebro-owned directory".to_string(),
    };
    let temp_dir = verify_owned_dir("temp directory", &options.temp_dir, links)?;
    // Refused like the temp directory, never silently dropped: without it every launch on a GPU the
    // build carries only PTX for (or a Vulkan build) recompiles its kernels (see `SpawnOptions`).
    let cache_dir = verify_owned_dir(CACHE_DIR_ROLE, &options.cache_dir, links)?;
    let log_parent = options
        .log_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(bad_log_path)?;
    let log_dir = verify_owned_dir("log directory", log_parent, links)?;
    let log_path = match links {
        DirLinks::Refuse => options.log_path.clone(),
        DirLinks::Follow => log_dir.join(options.log_path.file_name().ok_or_else(bad_log_path)?),
    };

    // Opened before the child exists: a failure here must never leave an unkillable, uncontained
    // process behind for a caller to clean up.
    let log = Arc::new(Mutex::new(RotatingLog::open(
        log_path,
        options.log_max_bytes,
    )?));

    // The two spawn-owned protections: recorded here, once, so both the environment block and the
    // job-containment decision below read the same snapshot of `protections` — passed in as one
    // argument, not read piecemeal from `options` (which carries no copy of it at all — see
    // `SpawnOptions`'s own docs), so the two decisions can never disagree.
    let mut protection_warnings = Vec::new();
    if let Some(w) = protections.environment_allowlist_warning() {
        protection_warnings.push(w);
    }
    if let Some(w) = protections.job_containment_warning() {
        protection_warnings.push(w);
    }
    // Shared with preflight: recorded here only when no preflight record exists to carry it, so a
    // preflighted launch's record lists it exactly once.
    if redirect_record == RedirectRecord::Here {
        if let Some(w) = protections.redirect_warning() {
            protection_warnings.push(w);
        }
    }

    let system32 = cerebro_probe::child_env::system_directory().map_err(|e| {
        SpawnError::SystemDirectoryUnavailable(match e {
            cerebro_probe::ProbeError::SystemDirectoryUnavailable { cause } => cause,
            other => other.to_string(),
        })
    })?;
    let child_env = effective_child_env(
        &protections,
        std::env::vars_os(),
        &system32,
        backend_dir,
        &temp_dir,
        options.cuda_visible_devices.as_deref(),
        &cache_dir,
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
    // accepted rather than closed with `CREATE_SUSPENDED`. `Protections::contain_in_job_object` off
    // skips creating a Job Object at all — the `None` this leaves `job` at is already the exact
    // representable state the assign-failure race below can also produce, so no new state is needed
    // to honour this switch, only a reason not to have tried in the first place.
    let job = if protections.contain_in_job_object {
        let job_result = job::JobHandle::create().and_then(|job| {
            // `child.as_raw_handle()` is the live process handle `Command::spawn` just produced,
            // owned by `child` for at least as long as `child` itself is not dropped, which is well
            // past this call — `job::JobHandle::assign`'s own contract requires exactly that.
            let raw = child.as_raw_handle() as HANDLE;
            job.assign(raw).map(|()| job)
        });
        match job_result {
            Ok(job) => Some(job),
            Err(e) => {
                // `AssignProcessToJobObject` reports `ERROR_ACCESS_DENIED` both for a genuine
                // permissions problem and for a process that has already exited (measured on this
                // machine — see this module's top-level docs) — indistinguishable by error code
                // alone. A `--version`-style child routinely wins this race by exiting in
                // single-digit milliseconds, and that is a successful run, not a containment
                // failure: there is nothing left to contain, so the caller still gets a
                // `ChildHandle` with `job` left `None` and whatever the child already printed. Only
                // the still-running case is a real containment failure worth killing for.
                match child.try_wait() {
                    Ok(Some(_already_exited)) => None,
                    _ => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(e);
                    }
                }
            }
        }
    } else {
        None
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
        protection_warnings,
    })
}

/// The exact environment block the child receives, honouring
/// `protections.enforce_environment_allowlist`: a thin adapter from this crate's
/// [`crate::protections::Protections`] to [`cerebro_probe::child_env::effective_child_env`], which
/// documents what "off" still pins (`PATH` first entries, `TEMP`/`TMP`, `CUDA_VISIBLE_DEVICES`, the
/// driver cache variables) and why. `cache_dir` is required here, not optional: every supervised
/// launch pins its driver caches (see [`SpawnOptions::cache_dir`]). Kept as a function of its own
/// so the on/off decision stays unit-testable without spawning a real process, against the same
/// `Protections` value `spawn_windows_at` reads.
#[cfg(windows)]
fn effective_child_env(
    protections: &crate::protections::Protections,
    parent_env: impl Iterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    system32: &Path,
    backend_dir: &Path,
    temp_dir: &Path,
    cuda_visible_devices: Option<&str>,
    cache_dir: &Path,
) -> std::collections::BTreeMap<std::ffi::OsString, std::ffi::OsString> {
    cerebro_probe::child_env::effective_child_env(
        protections.enforce_environment_allowlist,
        parent_env,
        system32,
        backend_dir,
        Some(temp_dir),
        cuda_visible_devices,
        Some(cache_dir),
    )
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

        cerebro_probe::backend::create_junction_for_tests(&link, &target)
            .expect("creating a junction needs no special privilege");

        let err = verify_owned_dir("temp directory", &link, DirLinks::Refuse)
            .expect_err("a junction must be refused");
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

        cerebro_probe::backend::create_junction_for_tests(&link, &real)
            .expect("creating a junction needs no special privilege");

        // `link\temp` resolves through the junction to `real\temp`, which is itself an ordinary
        // directory with no reparse attribute of its own — only its *ancestor* `link` is a junction.
        let reached_through_junction = link.join("temp");
        let err = verify_owned_dir(
            "temp directory",
            &reached_through_junction,
            DirLinks::Refuse,
        )
        .expect_err("a directory reached only through an ancestor junction must be refused");
        assert!(matches!(err, SpawnError::UnsafeDir { .. }), "{err:?}");

        let _ = std::fs::remove_dir(&link);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn a_temp_dir_behind_a_parent_junction_resolves_to_its_target_with_the_refusal_off() {
        // The off half of `a_temp_dir_reached_through_a_parent_junction_is_refused`: the same path
        // is accepted, and what comes back (what the child's TEMP will be) is the real directory,
        // in plain drive-letter form, never the link.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-follow-root-{}-{nanos}",
            std::process::id()
        ));
        let real = root.join("real");
        std::fs::create_dir_all(real.join("temp")).expect("create real target directory tree");
        let link = root.join("link");
        cerebro_probe::backend::create_junction_for_tests(&link, &real)
            .expect("creating a junction needs no special privilege");

        let through = link.join("temp");
        let refused = verify_owned_dir("temp directory", &through, DirLinks::Refuse);
        let followed = verify_owned_dir("temp directory", &through, DirLinks::Follow);
        let at_link = verify_owned_dir("temp directory", &link, DirLinks::Follow);
        let expected = plain_form(std::fs::canonicalize(real.join("temp")).expect("canonicalize"));
        let expected_link = plain_form(std::fs::canonicalize(&real).expect("canonicalize"));
        let _ = std::fs::remove_dir(&link);
        let _ = std::fs::remove_dir_all(&root);

        let refused = refused.expect_err("on: refused");
        assert!(
            refused.to_string().contains(
                &crate::protections::Protection::RedirectedPathRefusal.settings_hint()
            ),
            "{refused}"
        );
        let followed = followed.expect("off: followed");
        assert_eq!(followed, expected);
        assert!(
            !followed.to_string_lossy().starts_with(r"\\?\"),
            "{}",
            followed.display()
        );
        assert!(
            !followed.to_string_lossy().contains("link"),
            "{}",
            followed.display()
        );
        assert_eq!(
            at_link.expect("off: a junction at the leaf is followed too"),
            expected_link
        );
    }

    #[cfg(windows)]
    #[test]
    fn following_links_still_refuses_a_file_a_missing_path_and_a_relative_path() {
        let file = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-follow-file-{}.txt",
            std::process::id()
        ));
        std::fs::write(&file, b"not a directory").expect("create a plain file");
        let as_file = verify_owned_dir("temp directory", &file, DirLinks::Follow);
        let _ = std::fs::remove_file(&file);
        assert!(
            matches!(&as_file, Err(SpawnError::UnsafeDir { cause, .. }) if cause.contains("not a directory")),
            "{as_file:?}"
        );
        let missing = std::env::temp_dir().join("cerebro-supervisor-spawn-follow-missing-x9q");
        assert!(matches!(
            verify_owned_dir("temp directory", &missing, DirLinks::Follow),
            Err(SpawnError::UnsafeDir { .. })
        ));
        assert!(matches!(
            verify_owned_dir("temp directory", Path::new("relative"), DirLinks::Follow),
            Err(SpawnError::UnsafeDir { cause, .. }) if cause.contains("absolute")
        ));
    }

    #[cfg(windows)]
    #[test]
    fn plain_form_drops_only_a_verbatim_prefix_and_never_turns_a_path_relative() {
        assert_eq!(
            plain_form(PathBuf::from(r"\\?\C:\a\b")),
            PathBuf::from(r"C:\a\b")
        );
        assert_eq!(
            plain_form(PathBuf::from(r"\\?\UNC\srv\share\x")),
            PathBuf::from(r"\\srv\share\x")
        );
        assert_eq!(plain_form(PathBuf::from(r"C:\a")), PathBuf::from(r"C:\a"));
        let guid = PathBuf::from(r"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\t");
        assert_eq!(
            plain_form(guid.clone()),
            guid,
            "no plain form: kept, still absolute"
        );
        assert!(plain_form(PathBuf::from(r"\\?\UNC\srv\share\x")).is_absolute());
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
            cache_dir: PathBuf::from("never-checked-relative-path"),
            cuda_visible_devices: None,
            log_path: PathBuf::from("never-checked-either"),
            log_max_bytes: 1024,
        };
        let err = spawn_windows_at(
            Path::new(r"C:\a;C:\evil\llama-server.exe"),
            Path::new(r"C:\a;C:\evil"),
            &[],
            &bogus_options,
            crate::protections::Protections::default(),
            RedirectRecord::Here,
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
            cache_dir: PathBuf::from("never-checked-relative-path"),
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
        let err = verify_owned_dir(
            "temp directory",
            Path::new("relative-temp"),
            DirLinks::Refuse,
        )
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
            cache_dir: scratch.clone(),
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
            crate::protections::Protections::default(),
            RedirectRecord::Here,
        )
        .expect_err("a log_path with no parent directory must be refused");
        assert!(
            matches!(err, SpawnError::UnsafeDir { role: "crash log path", .. }),
            "{err:?}"
        );

        let _ = std::fs::remove_dir_all(&scratch);
    }

    /// A launch whose driver cache directory is unusable is refused before anything is opened or
    /// spawned, never run with the cache silently dropped (which would recompile every kernel).
    #[cfg(windows)]
    #[test]
    fn an_unusable_cache_dir_refuses_the_launch_before_anything_is_spawned() {
        let scratch = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-cache-refused-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&scratch).expect("create a valid scratch dir for this test");
        for bad in [
            PathBuf::from("relative-cache"),
            scratch.join("does-not-exist"),
        ] {
            let options = SpawnOptions {
                temp_dir: scratch.clone(),
                cache_dir: bad.clone(),
                cuda_visible_devices: None,
                log_path: scratch.join("spawn.log"),
                log_max_bytes: 1024,
            };
            let err = spawn_windows_at(
                Path::new(r"C:\Windows\System32\this-is-never-executed.exe"),
                Path::new(r"C:\Windows\System32"),
                &[],
                &options,
                crate::protections::Protections::default(),
                RedirectRecord::Here,
            )
            .expect_err("an unusable cache dir must be refused");
            assert!(
                matches!(err, SpawnError::UnsafeDir { role: CACHE_DIR_ROLE, .. }),
                "{}: {err:?}",
                bad.display()
            );
        }
        assert!(
            !scratch.join("spawn.log").exists(),
            "the refusal must come before the crash log is opened"
        );
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[cfg(windows)]
    #[test]
    fn verify_cache_dir_refuses_a_junction_on_and_follows_it_to_the_target_off() {
        let root = std::env::temp_dir().join(format!(
            "cerebro-supervisor-spawn-cache-junction-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let real = root.join("real-cache");
        std::fs::create_dir_all(&real).expect("create the real cache dir");
        let link = root.join("cache-link");
        cerebro_probe::backend::create_junction_for_tests(&link, &real)
            .expect("junctions need no privilege");

        let on = crate::protections::Protections::default();
        let off = crate::protections::Protections {
            refuse_redirected_paths: false,
            ..on
        };
        let refused = verify_cache_dir(&link, &on);
        let followed = verify_cache_dir(&link, &off);
        let plain = verify_cache_dir(&real, &on);
        let relative = verify_cache_dir(Path::new("relative-cache"), &off);
        let expected = plain_form(std::fs::canonicalize(&real).expect("canonicalize"));
        let _ = std::fs::remove_dir(&link);
        let _ = std::fs::remove_dir_all(&root);

        let refused = refused.expect_err("on: a junction is refused");
        assert!(
            matches!(&refused, SpawnError::UnsafeDir { role: CACHE_DIR_ROLE, .. }),
            "{refused:?}"
        );
        assert!(
            refused.to_string().contains(
                &crate::protections::Protection::RedirectedPathRefusal.settings_hint()
            ),
            "{refused}"
        );
        assert_eq!(followed.expect("off: followed"), expected, "the target, never the link");
        assert_eq!(plain.expect("a real directory passes"), real);
        assert!(
            matches!(relative, Err(SpawnError::UnsafeDir { .. })),
            "relative is refused in both states: {relative:?}"
        );
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

    /// [`effective_child_env`]'s own tests — exercising the allowlist on/off decision, and the
    /// settings re-pinning that must survive it off, without spawning a real process. See that
    /// function's own doc comment, which promises exactly this.
    #[cfg(windows)]
    #[test]
    fn allowlist_on_strips_llama_arg_and_lets_the_launchs_own_cuda_setting_win() {
        let parent = vec![
            (
                std::ffi::OsString::from("LLAMA_ARG_PORT"),
                std::ffi::OsString::from("9999"),
            ),
            (
                std::ffi::OsString::from("CUDA_VISIBLE_DEVICES"),
                std::ffi::OsString::from("0,1"),
            ),
        ];
        let env = effective_child_env(
            &crate::protections::Protections::default(),
            parent.into_iter(),
            Path::new(r"C:\Windows\System32"),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            Some("1"),
            Path::new(r"C:\cache"),
        );
        assert!(
            !env.keys()
                .any(|k| k.to_string_lossy().starts_with("LLAMA_ARG_")),
            "{env:?}"
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES")),
            Some(&std::ffi::OsString::from("1")),
            "the launch's own cuda_visible_devices setting must win over an ambient value"
        );
    }

    #[cfg(windows)]
    #[test]
    fn allowlist_off_inherits_ambient_names_but_keeps_settings_pinned() {
        let parent = vec![
            (
                std::ffi::OsString::from("LLAMA_ARG_PORT"),
                std::ffi::OsString::from("1"),
            ),
            (
                std::ffi::OsString::from("Path"),
                std::ffi::OsString::from(r"C:\evil"),
            ),
            (
                std::ffi::OsString::from("CUDA_VISIBLE_DEVICES"),
                std::ffi::OsString::from("0,1"),
            ),
            (
                std::ffi::OsString::from("TEMP"),
                std::ffi::OsString::from(r"C:\ambient-temp"),
            ),
        ];
        let off = crate::protections::Protections {
            enforce_environment_allowlist: false,
            ..crate::protections::Protections::default()
        };
        let env = effective_child_env(
            &off,
            parent.into_iter(),
            Path::new(r"C:\Windows\System32"),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            Some("1"),
            Path::new(r"C:\cache"),
        );
        // The one thing the allowlist protection itself governs: an ambient LLAMA_ARG_* still gets
        // through once it is off.
        assert_eq!(
            env.get(std::ffi::OsStr::new("LLAMA_ARG_PORT")),
            Some(&std::ffi::OsString::from("1")),
            "turning the allowlist off must still let an ambient LLAMA_ARG_* reach the child"
        );
        // Everything that is a *setting*, not a protection, must stay pinned regardless.
        assert_eq!(
            env.get(std::ffi::OsStr::new("TEMP")),
            Some(&std::ffi::OsString::from(r"C:\t"))
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES")),
            Some(&std::ffi::OsString::from("1")),
            "the launch's own cuda_visible_devices setting must win even with the allowlist off"
        );
        let path_keys: Vec<_> = env
            .keys()
            .filter(|k| k.eq_ignore_ascii_case(std::ffi::OsStr::new("path")))
            .collect();
        assert_eq!(
            path_keys.len(),
            1,
            "exactly one PATH-named key must survive: {env:?}"
        );
        assert_eq!(
            env.get(path_keys[0])
                .map(|v| v.to_string_lossy().into_owned()),
            Some(r"C:\backend;C:\Windows\System32;C:\evil".to_string()),
            "allowlist off: the inherited PATH reaches the child after the pinned backend_dir and \
             System32 entries, in one PATH-named key"
        );
    }

    #[cfg(windows)]
    #[test]
    fn allowlist_on_never_passes_the_inherited_path() {
        let parent = vec![(
            std::ffi::OsString::from("Path"),
            std::ffi::OsString::from(r"C:\evil"),
        )];
        let env = effective_child_env(
            &crate::protections::Protections::default(),
            parent.into_iter(),
            Path::new(r"C:\Windows\System32"),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            None,
            Path::new(r"C:\cache"),
        );
        let paths: Vec<_> = env
            .iter()
            .filter(|(k, _)| k.eq_ignore_ascii_case(std::ffi::OsStr::new("path")))
            .map(|(_, v)| v.to_string_lossy().into_owned())
            .collect();
        assert_eq!(paths, vec![r"C:\backend;C:\Windows\System32".to_string()]);
    }

    #[cfg(windows)]
    #[test]
    fn allowlist_off_with_an_empty_inherited_path_adds_no_empty_entry() {
        let parent = vec![(std::ffi::OsString::from("PATH"), std::ffi::OsString::new())];
        let off = crate::protections::Protections {
            enforce_environment_allowlist: false,
            ..crate::protections::Protections::default()
        };
        let env = effective_child_env(
            &off,
            parent.into_iter(),
            Path::new(r"C:\Windows\System32"),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            None,
            Path::new(r"C:\cache"),
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new("PATH")),
            Some(&std::ffi::OsString::from(r"C:\backend;C:\Windows\System32")),
            "an empty inherited PATH must not leave a trailing ';' (an empty entry)"
        );
    }

    #[cfg(windows)]
    #[test]
    fn allowlist_off_omits_cuda_visible_devices_when_the_launch_did_not_set_one() {
        let parent = vec![(
            std::ffi::OsString::from("CUDA_VISIBLE_DEVICES"),
            std::ffi::OsString::from("0,1"),
        )];
        let off = crate::protections::Protections {
            enforce_environment_allowlist: false,
            ..crate::protections::Protections::default()
        };
        let env = effective_child_env(
            &off,
            parent.into_iter(),
            Path::new(r"C:\Windows\System32"),
            Path::new(r"C:\backend"),
            Path::new(r"C:\t"),
            None,
            Path::new(r"C:\cache"),
        );
        assert!(
            !env.contains_key(std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES")),
            "None means omit, per SpawnOptions::cuda_visible_devices's own docs — even with the \
             allowlist off, an ambient value must not resurrect it: {env:?}"
        );
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
                cache_dir: scratch.0.clone(),
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

        /// A process watched through a handle opened while it is alive, independent of anything
        /// this crate's own `ChildHandle` tracks: the point of these tests is the *OS-level*
        /// process. Holding the handle pins the process's identity, so neither the liveness checks
        /// nor the cleanup kill can reach an unrelated process that reused the PID (which a
        /// check-by-PID after the `ChildHandle` is gone could).
        struct ProcessGuard(windows_sys::Win32::Foundation::HANDLE);

        impl ProcessGuard {
            /// Open `pid` for waiting and termination. Panics (test code) if it cannot be opened,
            /// which means the process was already gone when its PID was read.
            fn open(pid: u32) -> Self {
                use windows_sys::Win32::System::Threading::{
                    OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
                    PROCESS_TERMINATE,
                };
                // SAFETY: `OpenProcess` takes no pointers; the result is checked for null before
                // use and closed exactly once, in `Drop`.
                let handle = unsafe {
                    OpenProcess(
                        PROCESS_TERMINATE | PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                        0,
                        pid,
                    )
                };
                assert!(
                    !handle.is_null(),
                    "could not open pid {pid}; it must still be running"
                );
                Self(handle)
            }

            /// Whether the process exits within `timeout` (a zero timeout only polls).
            fn exited_within(&self, timeout: Duration) -> bool {
                use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
                use windows_sys::Win32::System::Threading::WaitForSingleObject;
                let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX - 1);
                // SAFETY: `self.0` is a live process handle opened with `PROCESS_SYNCHRONIZE`.
                unsafe { WaitForSingleObject(self.0, millis) == WAIT_OBJECT_0 }
            }

            /// Whether the process is still running right now.
            fn is_alive(&self) -> bool {
                !self.exited_within(Duration::ZERO)
            }

            /// End the process (test cleanup, for a grandchild outside any Job Object). Harmless
            /// once it has exited.
            fn terminate(&self) {
                // SAFETY: `self.0` is a live process handle opened with `PROCESS_TERMINATE`.
                unsafe {
                    windows_sys::Win32::System::Threading::TerminateProcess(self.0, 1);
                }
            }
        }

        impl Drop for ProcessGuard {
            fn drop(&mut self) {
                // Ends the process if a failed assertion left it running; harmless if it already
                // exited. The held handle guarantees this is the process that was opened.
                self.terminate();
                // SAFETY: opened in `open`, closed only here.
                unsafe {
                    windows_sys::Win32::Foundation::CloseHandle(self.0);
                }
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
                crate::protections::Protections::default(),
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
                    from the resulting Preflighted; from an elevated shell, where the public \
                    preflight() refuses with RunningElevated, the Preflighted comes from \
                    preflight_with with only the elevation answer faked"]
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
            let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));

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
                protections: crate::protections::Protections::default(),
            };
            use crate::preflight::{ElevationCheck, ProcessTokenElevation};
            let elevated = ProcessTokenElevation
                .is_elevated()
                .expect("query this test process's own token");
            let preflighted = if elevated {
                assert_eq!(
                    preflight(&spec).err(),
                    Some(crate::preflight::PreflightError::RunningElevated),
                    "the public preflight must refuse an elevated process"
                );
                crate::preflight::preflight_with(
                    &spec,
                    &crate::preflight::HostChecks {
                        elevation: &NotElevated,
                        dacl: &crate::preflight::ProbeDaclAuditor,
                    },
                )
            } else {
                preflight(&spec)
            }
            .expect("preflight the fixture backend");
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
                crate::protections::Protections::default(),
                &accept(),
            )
            .expect("spawn a long-running ping.exe");

            let child = ProcessGuard::open(handle.pid());
            assert!(child.is_alive(), "ping.exe should still be running");
            assert_eq!(
                handle.try_wait().expect("poll before drop"),
                None,
                "ping -t must not have exited on its own yet"
            );

            drop(handle);

            assert!(
                child.exited_within(Duration::from_secs(2)),
                "dropping the handle must kill the child"
            );
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
                let handle = spawn_unaudited(
                    &fixture_dir,
                    "cerebro-print-env.exe",
                    &[],
                    &opts,
                    crate::protections::Protections::default(),
                    &accept(),
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "iteration {i}: spawn must succeed even if the \
                                             child already exited, got {e}"
                    )
                });
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
                crate::protections::Protections::default(),
                &accept(),
            )
            .expect("spawn a long-running ping.exe");
            let child = ProcessGuard::open(handle.pid());
            assert!(
                child.is_alive(),
                "ping.exe should still be running, so assignment cannot have lost a race"
            );
            assert!(handle.is_contained(), "a still-running child must report itself contained");

            handle.kill();
            assert!(
                child.exited_within(Duration::from_secs(2)),
                "kill must end the child"
            );
        }

        #[test]
        #[ignore = "spawns a real, long-running child with job containment turned off"]
        fn is_contained_reports_false_when_job_containment_is_turned_off() {
            // The off half of the property `is_contained_reports_true_for_a_running_child` proves
            // for the default: with `Protections::contain_in_job_object` turned off, a still-running
            // child must report itself uncontained — this switch skips creating a Job Object at all
            // (see `spawn_windows_at`), which is a deterministic, always-observable effect, unlike
            // the opportunistic assign-after-exit race `a_child_that_exits_before_job_assignment...`
            // above can only try to hit.
            let system32 = PathBuf::from(r"C:\Windows\System32");
            let scratch = ScratchDir::new("contained-off");
            let opts = options(&scratch);
            let protections = crate::protections::Protections {
                contain_in_job_object: false,
                ..crate::protections::Protections::default()
            };
            let handle = spawn_unaudited(
                &system32,
                "ping.exe",
                &["-t".to_string(), "127.0.0.1".to_string()],
                &opts,
                protections,
                &accept(),
            )
            .expect("spawn a long-running ping.exe");
            let child = ProcessGuard::open(handle.pid());
            assert!(child.is_alive(), "ping.exe should still be running");
            assert!(
                !handle.is_contained(),
                "job containment is off, so this launch must never report itself contained"
            );
            assert_eq!(
                handle.protection_warnings().len(),
                1,
                "exactly one protection was turned off for this launch: {:?}",
                handle.protection_warnings()
            );
            assert_eq!(
                handle.protection_warnings()[0].protection(),
                crate::protections::Protection::JobObjectContainment
            );

            // Without a Job Object, only the immediate PID is guaranteed to die here — exactly the
            // risk `Protections::contain_in_job_object`'s own documentation names; `ping.exe` has no
            // children of its own, so `kill` on the immediate process is enough to clean this test up.
            handle.kill();
            assert!(
                child.exited_within(Duration::from_secs(2)),
                "kill must end the child"
            );
        }

        /// An elevation check that always answers "not elevated", so this test reaches spawn from
        /// this machine's elevated harness shell without turning `refuse_elevation` off (which would
        /// add a preflight warning unrelated to what this test checks).
        struct NotElevated;

        impl crate::preflight::ElevationCheck for NotElevated {
            fn is_elevated(&self) -> Result<bool, crate::preflight::PreflightError> {
                Ok(false)
            }
        }

        #[test]
        #[ignore = "preflights a copy of the print-env fixture under %TEMP% and spawns it through \
                    spawn_preflighted with job containment off"]
        fn spawn_preflighted_honours_the_preflighted_protections() {
            // The design claim this pins down: one `Protections` per launch, and `spawn_preflighted`
            // reads it from `Preflighted::protections()`. Every other non-default spawn-switch test
            // goes through `spawn_unaudited`, so if `spawn_preflighted` regressed to
            // `Protections::default()`, job containment would silently come back on for every real
            // launch and no other test would notice. The child must be long-running (the
            // spawn-grandchild mode parks on its `ping -t` grandchild): a child that exits before
            // job assignment reports `is_contained() == false` even with containment on, which would
            // make this assertion pass for the wrong reason.
            use crate::preflight::{
                preflight_with, AllowUnsafeDacl, HostChecks, LaunchSpec, PortRequest,
                ProbeDaclAuditor,
            };

            let scratch = ScratchDir::new("preflighted-protections");
            let backend = scratch.0.join("backend");
            std::fs::create_dir(&backend).expect("create the fixture backend dir");
            std::fs::copy(
                print_env_fixture_dir().join("cerebro-print-env.exe"),
                backend.join("cerebro-print-env.exe"),
            )
            .expect("copy the print-env fixture into its own backend dir");
            let (manifest, _) =
                cerebro_probe::backend::hash_directory(&backend).expect("hash the fixture backend");
            // Never loaded: the fixture ignores it. Preflight only checks the GGUF magic.
            let model = scratch.0.join("model.gguf");
            std::fs::write(&model, b"GGUF\x03\0\0\0").expect("write a GGUF-magic fixture file");
            let key_path = scratch.0.join("api.key");
            std::fs::write(&key_path, "k".repeat(32)).expect("write fixture api key file");

            let spec = LaunchSpec {
                backend_dir: backend.clone(),
                tool_name: "cerebro-print-env.exe".to_string(),
                manifest,
                model_path: model,
                draft_model_path: None,
                port: PortRequest::Any,
                api_key_file: key_path,
                allow_unsafe_dacl: Some(
                    AllowUnsafeDacl::new(
                        "fixture under %TEMP%, whose ancestors are writable by other principals on \
                         this development machine",
                    )
                    .expect("reason is non-blank"),
                ),
                protections: crate::protections::Protections {
                    contain_in_job_object: false,
                    // Shared by preflight and spawn, and recorded by preflight alone for this
                    // path, so the launch record lists it once (asserted below).
                    refuse_redirected_paths: false,
                    ..crate::protections::Protections::default()
                },
            };
            let preflighted = preflight_with(
                &spec,
                &HostChecks {
                    elevation: &NotElevated,
                    dacl: &ProbeDaclAuditor,
                },
            )
            .expect("preflight the print-env fixture");
            assert!(!preflighted.protections().contain_in_job_object);
            let preflight_redirects = preflighted
                .warnings()
                .iter()
                .filter(|w| {
                    matches!(w, crate::preflight::PreflightWarning::ProtectionDisabled(p)
                        if p.protection() == crate::protections::Protection::RedirectedPathRefusal)
                })
                .count();
            assert_eq!(preflight_redirects, 1, "{:?}", preflighted.warnings());

            let opts = options(&scratch);
            let handle = spawn_preflighted(&preflighted, &["spawn-grandchild".to_string()], &opts)
                .expect("spawn the preflighted fixture");
            let grandchild_pid: u32 = poll_until(Duration::from_secs(5), || {
                handle.recent_lines().iter().find_map(|line| {
                    line.strip_prefix("stdout: GRANDCHILD_PID=")
                        .and_then(|pid| pid.trim().parse().ok())
                })
            });

            let contained = handle.is_contained();
            let warnings: Vec<_> = handle
                .protection_warnings()
                .iter()
                .map(|w| w.protection())
                .collect();
            // Opened while both are alive (the fixture is parked on its grandchild), so the cleanup
            // below reaches exactly these two processes even if a PID is reused afterwards.
            let child = ProcessGuard::open(handle.pid());
            let grandchild = ProcessGuard::open(grandchild_pid);
            // Cleanup before asserting, so a failed assertion cannot leak either process: without a
            // Job Object only the immediate child dies on `kill`, so the grandchild is ended through
            // its own held handle.
            handle.kill();
            grandchild.terminate();
            let child_gone = child.exited_within(Duration::from_secs(5));
            let grandchild_gone = grandchild.exited_within(Duration::from_secs(5));
            // Without a job the grandchild inherited the output pipe, so the log reader can still be
            // finishing its last write for a moment after both processes are gone; `ScratchDir`'s
            // single best-effort removal lost that race and left the directory behind. Retried here
            // for a bounded time instead (a leftover directory is not worth failing the test over).
            drop(handle);
            let removal_started = Instant::now();
            while std::fs::remove_dir_all(&scratch.0).is_err()
                && scratch.0.exists()
                && removal_started.elapsed() < Duration::from_secs(5)
            {
                std::thread::sleep(Duration::from_millis(50));
            }

            assert!(
                child_gone && grandchild_gone,
                "cleanup must end both processes"
            );
            assert!(
                !contained,
                "Preflighted::protections must reach spawn: the child was contained"
            );
            assert_eq!(
                warnings,
                vec![crate::protections::Protection::JobObjectContainment],
                "exactly the one spawn-owned protection that was turned off must be recorded; the \
                 redirect switch is already in the preflight record and must not appear twice"
            );
        }

        #[test]
        #[ignore = "spawns the print-env fixture with its TEMP reached through a junction"]
        fn a_temp_dir_behind_a_junction_is_refused_on_and_handed_to_the_child_as_its_target_off() {
            // `%LOCALAPPDATA%` moved to another drive with a junction is the real case. On, the
            // spawn is refused before any child exists. Off, it proceeds, the child's TEMP is the
            // junction's target (never the link), and the one switch is recorded exactly once.
            let fixture_dir = print_env_fixture_dir();
            let scratch = ScratchDir::new("redirect-temp");
            let real = scratch.0.join("real");
            std::fs::create_dir_all(real.join("temp")).expect("create the junction target tree");
            let link = scratch.0.join("link");
            cerebro_probe::backend::create_junction_for_tests(&link, &real)
                .expect("creating a junction needs no special privilege");
            let opts = SpawnOptions {
                temp_dir: link.join("temp"),
                ..options(&scratch)
            };

            let refused = spawn_unaudited(
                &fixture_dir,
                "cerebro-print-env.exe",
                &[],
                &opts,
                crate::protections::Protections::default(),
                &accept(),
            );
            let refused_role = match &refused {
                Err(SpawnError::UnsafeDir { role, .. }) => Some(*role),
                _ => None,
            };

            let protections = crate::protections::Protections {
                refuse_redirected_paths: false,
                ..crate::protections::Protections::default()
            };
            let handle = spawn_unaudited(
                &fixture_dir,
                "cerebro-print-env.exe",
                &[],
                &opts,
                protections,
                &accept(),
            )
            .expect("with the refusal off, a junctioned temp dir is followed");
            let status = poll_until(Duration::from_secs(5), || {
                handle.try_wait().expect("poll child exit status")
            });
            let lines = poll_until(Duration::from_secs(2), || {
                let lines = handle.recent_lines();
                lines
                    .iter()
                    .any(|l| l.starts_with("stdout: TEMP="))
                    .then_some(lines)
            });
            let warnings: Vec<_> = handle
                .protection_warnings()
                .iter()
                .map(|w| w.protection())
                .collect();
            drop(handle);
            let _ = std::fs::remove_dir(&link);

            assert_eq!(refused_role, Some("temp directory"), "{refused:?}");
            assert!(status.success(), "{status:?}");
            let expected = plain_form(
                std::fs::canonicalize(real.join("temp")).expect("canonicalize the real temp dir"),
            );
            let temp_line = format!("stdout: TEMP={}", expected.display());
            assert!(
                lines.contains(&temp_line),
                "want {temp_line:?} in {lines:?}"
            );
            assert!(
                !lines
                    .iter()
                    .any(|l| l.starts_with("stdout: TEMP=") && l.contains("link")),
                "the child must never be handed the link: {lines:?}"
            );
            assert_eq!(
                warnings,
                vec![crate::protections::Protection::RedirectedPathRefusal],
                "spawn_unaudited has no preflight record, so it records the switch itself, once"
            );
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
                crate::protections::Protections::default(),
                &accept(),
            )
            .expect("spawn the print-env fixture in spawn-grandchild mode");

            let grandchild_pid: u32 = poll_until(Duration::from_secs(5), || {
                handle.recent_lines().iter().find_map(|line| {
                    line.strip_prefix("stdout: GRANDCHILD_PID=")
                        .and_then(|pid| pid.trim().parse().ok())
                })
            });
            let grandchild = ProcessGuard::open(grandchild_pid);
            assert!(
                grandchild.is_alive(),
                "grandchild pid {grandchild_pid} should be running before drop"
            );

            drop(handle);

            assert!(
                grandchild.exited_within(Duration::from_secs(2)),
                "the job must take the grandchild down with the handle"
            );
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
            let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));
            assert!(
                backend_dir.join("llama-server.exe").is_file(),
                "fixture backend not found at {}",
                backend_dir.display()
            );

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
                crate::protections::Protections::default(),
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
                crate::protections::Protections::default(),
                &accept(),
            )
            .expect("spawn a long-running ping.exe");
            let child = ProcessGuard::open(handle.pid());
            assert!(child.is_alive(), "ping.exe should still be running");

            handle.kill();
            // The second call must be a no-op: no panic, and no effect beyond what the first call
            // already did.
            handle.kill();

            // `kill()` only closes the job handle; the kernel tears the tree down asynchronously
            // afterward (see `kill`'s own doc comment), so `try_wait()` must be polled rather than
            // asserted once immediately after `kill()` returns.
            // An earlier version of this test asserted `try_wait()` a single time right after
            // polling a by-PID liveness check to `false`, which raced the child's own handle becoming
            // signaled and failed intermittently even though nothing was actually broken.
            // The exit *status* a job-object kill produces is not part of this test's contract
            // (Windows reports it as a plain, successful-looking exit code for this termination
            // path) — what matters is only that `try_wait` eventually reports the child gone at
            // all, which `poll_until` returning at all already proves.
            poll_until(Duration::from_secs(2), || handle.try_wait().expect("try_wait after kill"));
            assert!(
                child.exited_within(Duration::ZERO),
                "the OS must also agree the process is gone once try_wait reports it"
            );

            // A third call, after the child is already reaped, must still not panic.
            handle.kill();
        }
    }
}
