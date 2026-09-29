//! Typed failures for every probe.
//!
//! The probe's whole job is to be believed: the tuner prunes search axes and pins threads based on
//! what it reports, and a wrong answer is worse than no answer because it is silently wrong for the
//! rest of the run. So every path that could produce a plausible-but-false topology returns an
//! error instead — including paths where the kernel call itself *succeeded*.

use thiserror::Error;

/// Anything that can go wrong while establishing runtime ground truth.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProbeError {
    /// A Win32 call failed. `call` names the API so the code is actionable without a debugger.
    #[error("{call} failed with Win32 error {code}")]
    Win32 {
        /// The API that failed, e.g. `GetLogicalProcessorInformationEx`.
        call: &'static str,
        /// The value `GetLastError` returned immediately afterwards.
        code: u32,
    },

    /// Topology probing is only implemented for Windows; there is no portable substitute.
    #[error("CPU topology probing is implemented for Windows only")]
    UnsupportedPlatform,

    /// The kernel kept asking for a larger buffer than the one it had just sized.
    ///
    /// One retry is expected (a core can be hot-added between the two calls). Repeated growth is
    /// not, and looping on it forever would hang the tuner before it ran a single benchmark.
    #[error("processor topology buffer kept growing across {attempts} attempts")]
    BufferGrowthLoop {
        /// How many times the query was retried before giving up.
        attempts: u32,
    },

    /// A record claims a `Size` that cannot even contain its own header.
    ///
    /// `Size` is what advances the cursor. Zero — or anything below the fixed header — would either
    /// spin forever or re-parse the same bytes, so it is rejected before the cursor moves.
    #[error(
        "processor topology record at byte {offset} declares Size {size}, \
         below the {minimum}-byte record header"
    )]
    ImpossibleRecordSize {
        /// Byte offset of the offending record within the buffer.
        offset: usize,
        /// The `Size` field as the kernel wrote it.
        size: usize,
        /// Smallest structurally possible record.
        minimum: usize,
    },

    /// A record claims to extend past the end of the buffer the kernel said it had filled.
    ///
    /// Trusting `Size` here would read memory beyond the returned length. It comes from the kernel,
    /// but it is still length-prefixed data being walked with a cursor, so it is bounds-checked
    /// like any other.
    #[error(
        "processor topology record at byte {offset} declares Size {size}, \
         but only {remaining} bytes remain in the buffer"
    )]
    TruncatedRecord {
        /// Byte offset of the offending record within the buffer.
        offset: usize,
        /// The `Size` field as the kernel wrote it.
        size: usize,
        /// Bytes actually available from `offset` to the end of the returned data.
        remaining: usize,
    },

    /// A `RelationProcessorCore` record's `GroupCount` does not fit inside the record's own `Size`.
    #[error(
        "processor core record at byte {offset} declares GroupCount {group_count}, \
         which does not fit in its {size}-byte record"
    )]
    GroupCountOverrun {
        /// Byte offset of the offending record within the buffer.
        offset: usize,
        /// The `GroupCount` field as the kernel wrote it.
        group_count: usize,
        /// The record's own declared `Size`.
        size: usize,
    },

    /// The kernel returned no processor core records at all.
    ///
    /// Reachable when a filtered query succeeds but returns zero bytes; a machine with no cores is
    /// not a thing, so this means the query was answered from somewhere unexpected.
    #[error("the kernel reported no processor cores")]
    NoCores,

    /// A core was described with an empty affinity mask.
    ///
    /// This is the signature of reading `GROUP_AFFINITY.Mask` from the wrong offset inside the
    /// record: the call succeeds, the data is garbage, and every mask comes back `0x0`. An empty
    /// mask can never be a real core, so it is rejected here rather than propagated into a
    /// `--cpu-mask` that pins inference to no processor at all.
    #[error(
        "processor core (efficiency class {efficiency_class}) reported an empty affinity mask"
    )]
    EmptyCoreMask {
        /// Efficiency class of the core that reported nothing.
        efficiency_class: u8,
    },

    /// Two cores claim the same logical processor.
    #[error("logical processors {overlap:#x} were claimed by more than one physical core")]
    OverlappingCores {
        /// The bits claimed twice.
        overlap: u64,
    },

    /// The machine has more than one processor group, so affinity does not fit a single 64-bit
    /// mask.
    ///
    /// llama.cpp's `--cpu-mask` addresses one group's worth of logical processors, and Cerebro's
    /// mask type is a `u64` to match. Reporting only group 0 would look like a working answer while
    /// quietly hiding most of a large server, so this is an error instead.
    #[error("this machine has more than one processor group (saw group {group}); Cerebro's 64-bit affinity masks cannot address it")]
    MultipleProcessorGroups {
        /// The first non-zero group index encountered.
        group: u16,
    },

    /// A NUMA node count of zero was reported, which is not a possible machine.
    #[error("NUMA node count reported as zero")]
    ZeroNumaNodes,

    // ---- external tool queries (gpu.rs, backend.rs) ----
    //
    // Vendor tools and the backend binary are untrusted input sources: their output format changes
    // between versions, and accepting a malformed value would produce a plausible-but-wrong VRAM
    // figure or device list that silently poisons every plan built on it.
    /// A tool could not be launched at all.
    #[error("could not run {tool}: {cause}")]
    ToolSpawn {
        /// Path of the tool.
        tool: String,
        /// Underlying OS error, rendered as text so this error stays cloneable and comparable.
        cause: String,
    },

    /// A tool did not exit (and close its output) within the query deadline, so it was killed.
    ///
    /// A driver hang inside `--list-devices` would otherwise block the probe, and every caller
    /// waiting on it, forever.
    #[error("{tool} did not finish within {after:?} and was killed")]
    ToolTimedOut {
        /// Path of the tool.
        tool: String,
        /// The deadline it exceeded.
        after: std::time::Duration,
    },

    /// The real Windows system directory could not be resolved from the kernel. Without it there
    /// is no trustworthy value to build a child's `PATH`/`SystemRoot` on (see
    /// [`crate::child_env`]), so nothing is spawned.
    #[error("could not resolve the system directory: {cause}")]
    SystemDirectoryUnavailable {
        /// Why, including the Win32 error code.
        cause: String,
    },

    /// A tool ran and exited non-zero.
    #[error("{tool} exited with status {status}")]
    ToolFailed {
        /// Path of the tool.
        tool: String,
        /// Process exit code, or -1 if it was terminated by a signal.
        status: i32,
    },

    /// A tool produced output that is not valid UTF-8.
    #[error("{tool} produced output that is not valid UTF-8")]
    ToolOutputNotUtf8 {
        /// Path of the tool.
        tool: String,
    },

    /// A tool produced output that does not match the expected shape.
    ///
    /// Deliberately fatal rather than best-effort. Guessing which field is which after a format
    /// change yields a number that looks reasonable and is wrong.
    #[error("{tool} produced unexpected output: {detail}")]
    ToolOutputMalformed {
        /// Path or name of the tool.
        tool: String,
        /// What specifically did not match.
        detail: String,
    },

    /// The backend directory does not exist or is not a directory.
    #[error("backend directory does not exist: {dir}")]
    BackendDirMissing {
        /// Path that was checked.
        dir: String,
    },

    /// The backend directory could not be resolved to its canonical on-disk path while the
    /// folder-permission audit was switched off (see [`crate::backend::FolderAudit`]). Distinct from
    /// [`ProbeError::UnauditableSecurityDescriptor`] because no audit was attempted.
    #[error("could not resolve backend directory {dir}: {cause}")]
    BackendDirUnresolvable {
        /// Path that was given.
        dir: String,
        /// Why resolution failed.
        cause: String,
    },

    /// The backend directory could not be enumerated.
    #[error("backend directory {dir} could not be read: {cause}")]
    BackendDirUnreadable {
        /// Path that was checked.
        dir: String,
        /// Underlying OS error as text.
        cause: String,
    },

    /// A file inside the backend directory could not be read for hashing.
    #[error("backend file {file} could not be read: {cause}")]
    BackendFileUnreadable {
        /// Path that was checked.
        file: String,
        /// Underlying OS error as text.
        cause: String,
    },

    /// A backend directory entry's file name is not valid UTF-8.
    ///
    /// An unrepresentable name is refused rather than skipped: a planted file wearing a name the
    /// hash check silently ignores is exactly the gap a non-recursive integrity scan must not leave
    /// open.
    #[error("backend directory {dir} contains a file name that is not valid UTF-8: {name:?}")]
    BackendFileNameNotUtf8 {
        /// The backend directory.
        dir: String,
        /// Best-effort lossy rendering of the offending name.
        name: String,
    },

    /// The driver cache directory given to [`crate::backend::probe_with_cache`] is not an absolute
    /// path to an existing directory. Refused before any query runs: dropping the cache instead
    /// would make every query pay the kernel compile again with no signal.
    #[error("driver cache directory {dir} is unusable: {cause}")]
    CacheDirUnusable {
        /// The directory as given.
        dir: String,
        /// What is wrong with it.
        cause: String,
    },

    /// The named backend tool is not present in the backend directory.
    #[error("backend tool not found: {tool}")]
    BackendToolMissing {
        /// Path that was checked.
        tool: String,
    },

    /// The tool name is not a bare `.exe` file name (see [`crate::backend::is_bare_exe_name`]).
    ///
    /// The tool must be a bare `.exe` name resolved inside the backend directory. Accepting a path
    /// would let a crafted configuration point the probe at a binary somewhere else entirely, and a
    /// `.bat`/`.cmd` name would be run through `cmd.exe`.
    #[error("backend tool name must be a bare file name inside the backend directory, got {tool:?}")]
    BackendToolEscapesDir {
        /// The offending value.
        tool: String,
    },

    /// A DACL audit step against `object` failed: a security descriptor, ACE, or SID was
    /// structurally impossible to interpret, or an underlying Win32 call failed while examining it.
    ///
    /// Reachable for a malformed ACE (declared size that does not fit its ACL), a SID with more
    /// sub-authorities than it carries bytes for, an ACE type this crate does not know how to read,
    /// or a bare Win32 failure (e.g. `GetNamedSecurityInfoW`) given an object to name it against.
    /// Every one of these could grant access if guessed at, so a value here always becomes
    /// [`crate::backend::DaclAudit::Unknown`] rather than being skipped.
    #[error("could not audit {object}: {detail}")]
    UnauditableSecurityDescriptor {
        /// The path or logical object being audited, e.g. a file path or `"process token"`.
        object: String,
        /// What specifically went wrong.
        detail: String,
    },

    /// The folder-permission audit refused `object` because it is not on a local fixed volume or RAM
    /// disk: a UNC share, a mapped network drive, a removable or optical drive, a device-namespace
    /// path, or a path that resolves to one of those. The audit's trusted SIDs (`SYSTEM`,
    /// `Administrators`, ...) mean this machine's principals only when this machine issued the
    /// descriptor being read, so it cannot vouch for such a path.
    ///
    /// Distinct from [`ProbeError::UnauditableSecurityDescriptor`] because it is a fact about where
    /// the directory lives, not an I/O failure: retrying cannot change a volume's type, so a caller
    /// with a restart loop (`cerebro-supervisor`'s lifecycle) can stop at once instead of re-running
    /// a refusal that can never pass. Worded like that variant so the rendered audit verdict reads
    /// the same either way.
    #[error("could not audit {object}: {detail}")]
    BackendDirNotLocalVolume {
        /// The path that was refused, as the audit saw it.
        object: String,
        /// Which kind of non-local path it is.
        detail: String,
    },

    /// The folder-permission audit found a reparse point (symbolic link, junction, mount point, or
    /// a placeholder such as OneDrive's) on the already-resolved path to the backend directory, or
    /// directly inside it. The audit examines only the real objects on the path, never a
    /// redirect's target, so it fails closed.
    ///
    /// Distinct from [`ProbeError::UnauditableSecurityDescriptor`] because it is a fact about the
    /// layout on disk, not an I/O failure: resolution already followed every junction it could, so
    /// what remains (a volume mount folder, a cloud-files placeholder, a symlinked DLL) is there on
    /// every retry. A caller with a restart loop (`cerebro-supervisor`'s lifecycle) stops at once
    /// instead of re-hashing the backend to be refused the same way. Worded like that variant so the
    /// rendered audit verdict reads the same either way.
    #[error("could not audit {object}: {detail}")]
    BackendPathHasReparsePoint {
        /// The reparse point that was found.
        object: String,
        /// What it is and why the audit refuses it.
        detail: String,
    },

    /// [`crate::backend::probe_with`] refused to run a backend query because this process is
    /// elevated and [`crate::backend::ElevationRefusal::Enforce`] is set (the default). Nothing in
    /// the inference path needs admin rights, and `--list-devices` loads every ggml DLL, so an
    /// elevated probe would run backend code elevated. The user can turn this protection off.
    #[error(
        "refusing to run backend queries: this process is running elevated, and nothing in the \n         inference path needs admin rights (turn elevation refusal off in settings to allow it)"
    )]
    RunningElevated,

    /// [`crate::backend::probe_with`] could not determine whether this process is elevated, so it
    /// refused rather than assume "not elevated" (fail closed, like an unauditable DACL).
    #[error("could not determine whether this process is elevated: {cause}")]
    ElevationCheckFailed {
        /// Why the token query failed.
        cause: String,
    },

    /// [`crate::backend::probe`] refused to spawn anything from the backend directory because
    /// [`crate::backend::audit_dacl`] did not report [`crate::backend::DaclAudit::OwnerOnly`].
    ///
    /// This is the gate itself, not a report about one: ggml loads backend DLLs from the directory
    /// as soon as the first child process starts, so the audit must run — and pass — before that
    /// process is ever spawned.
    ///
    /// Carries [`crate::backend::UnsafeDacl`] structurally, not a pre-flattened string, so a caller
    /// — or a UI built on this crate — can recover every untrusted principal, path, and right
    /// without re-parsing text written for a human. `UnsafeDacl` has no variant meaning "safe to
    /// spawn", so this field can never hold [`crate::backend::DaclAudit::OwnerOnly`].
    #[error(
        "refusing to spawn from {dir}: {audit} (turn the folder-permission audit off in settings \
         to allow this)"
    )]
    BackendDirUnsafe {
        /// The backend directory that failed the audit.
        dir: String,
        /// Every write-capable grant found (`WritableByOthers`), or why the audit could not
        /// complete (`Unknown`).
        audit: crate::backend::UnsafeDacl,
    },
}
