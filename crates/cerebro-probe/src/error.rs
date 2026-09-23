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

    /// The named backend tool is not present in the backend directory.
    #[error("backend tool not found: {tool}")]
    BackendToolMissing {
        /// Path that was checked.
        tool: String,
    },

    /// The tool name contained path components.
    ///
    /// The tool must be a bare file name resolved inside the backend directory. Accepting a path
    /// would let a crafted configuration point the probe at a binary somewhere else entirely.
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
    #[error("refusing to spawn from {dir}: {audit}")]
    BackendDirUnsafe {
        /// The backend directory that failed the audit.
        dir: String,
        /// Every write-capable grant found (`WritableByOthers`), or why the audit could not
        /// complete (`Unknown`).
        audit: crate::backend::UnsafeDacl,
    },
}
