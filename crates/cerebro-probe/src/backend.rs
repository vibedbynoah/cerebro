//! Identity, integrity, and addressable devices of one llama.cpp build.
//!
//! # Why backend identity is part of every measurement key
//!
//! A self-built llama.cpp fork of the *same commit* measured 3-5x slower than the official release
//! on the development machine. So "which llama.cpp" is not metadata — it is an independent variable
//! that moves the number being optimized. A throughput result that does not name the exact binary
//! that produced it is not comparable to any other result, and the tuner would happily conclude that
//! a flag change caused a difference that was really a binary change.
//!
//! [`BackendIdentity`] therefore carries the build number, commit, compiler, the SHA-256 of every
//! tool and backend library, and which `ggml-cpu-*` variant the loader would select.
//!
//! # Why the directory ACL is a security property and not housekeeping
//!
//! ggml loads its backends as DLLs *from the backend directory at spawn time*. Anyone who can write
//! to that directory can therefore execute code inside Cerebro's child process. This is not
//! theoretical on the development machine: `qwen-flash-next/llama.cpp-b10751` is Modify-writable by
//! `ASUS\CodexSandboxUsers`. That is why [`DaclAudit`] is a structured value a caller can refuse to
//! spawn on, rather than a warning written to a log nobody reads.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::error::ProbeError;

/// The Win32 half of [`audit_dacl`]; kept apart so its `unsafe` surface is reviewable on its own.
#[cfg(windows)]
mod dacl;

/// A device identifier exactly as one specific backend binary reports it.
///
/// Distinct from anything in [`crate::gpu`] by type, not merely by name. A `PhysicalGpu` may exist
/// with no corresponding `AddressableDevice` — the development machine's Iris Xe is precisely that
/// case, because the shipping build has no Vulkan backend — and passing a physical device where an
/// addressable one is required must not compile.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressableDevice {
    id: String,
    description: String,
    total_mib: Option<u64>,
    free_mib: Option<u64>,
}

impl AddressableDevice {
    /// The identifier legal in `-dev` / `-devd`, e.g. `CUDA0`, `RPC0`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Human-readable description as the backend printed it.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Total memory as the backend sees it, when reported.
    pub fn total_mib(&self) -> Option<u64> {
        self.total_mib
    }

    /// Free memory as the backend sees it, when reported.
    pub fn free_mib(&self) -> Option<u64> {
        self.free_mib
    }
}

/// Identity of a llama.cpp build, as recorded in every measurement key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendIdentity {
    dir: PathBuf,
    version: String,
    build_number: Option<u32>,
    commit: Option<String>,
    compiler: Option<String>,
    hashes: BTreeMap<String, String>,
    ggml_backends: Vec<String>,
}

impl BackendIdentity {
    /// Directory the build lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Raw version line as printed.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Upstream build number, e.g. `10751`, when parseable.
    pub fn build_number(&self) -> Option<u32> {
        self.build_number
    }

    /// Short commit hash, when parseable.
    pub fn commit(&self) -> Option<&str> {
        self.commit.as_deref()
    }

    /// Compiler string, when parseable.
    pub fn compiler(&self) -> Option<&str> {
        self.compiler.as_deref()
    }

    /// SHA-256 by file name, over tools and ggml backend libraries.
    pub fn hashes(&self) -> &BTreeMap<String, String> {
        &self.hashes
    }

    /// `ggml-*.dll` backend libraries present, sorted.
    ///
    /// Absence is load-bearing: the development build contains no `ggml-vulkan.dll`, which is why
    /// its addressable set cannot include the iGPU no matter what the physical inventory says.
    pub fn ggml_backends(&self) -> &[String] {
        &self.ggml_backends
    }

    /// Whether a named ggml backend library is present, e.g. `"vulkan"`.
    pub fn has_ggml_backend(&self, name: &str) -> bool {
        let needle = format!("ggml-{}", name.to_ascii_lowercase());
        self.ggml_backends
            .iter()
            .any(|b| b.to_ascii_lowercase().starts_with(&needle))
    }
}

/// Result of auditing the write permissions on a backend directory and the files in it.
///
/// Only [`DaclAudit::OwnerOnly`] means safe. `Unknown` exists so that "we could not check" can never
/// be mistaken for "we checked and it is fine" — a security control that silently degrades to
/// permissive is worse than one that is absent, because it is trusted. `Redirected` is also a
/// refusal, kept apart from `Unknown` because it is a fixed fact about the layout, not a failed
/// check, so a caller with a restart loop can stop at once instead of retrying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaclAudit {
    /// No principal outside the trusted set (see [`audit_dacl`]) can write to the directory, to any
    /// regular file directly inside it, or delete, rename, or re-ACL any ancestor or intermediate
    /// component of the path used to reach it.
    OwnerOnly,
    /// Other principals can write. Each entry is one principal holding write-capable rights on one
    /// path. ggml loads DLLs from here at spawn time, so this is a code-execution vector and a
    /// caller should refuse to spawn.
    WritableByOthers(Vec<WritableGrant>),
    /// The check could not complete; the message names the path and, for OS failures, the Win32
    /// error code. Callers must treat this as untrusted, never as safe.
    Unknown(String),
    /// A reparse point sits on the resolved path to the directory, or directly inside it (see
    /// [`ProbeError::BackendPathHasReparsePoint`]): a volume mount folder, a OneDrive or dedup
    /// placeholder, an AppExecLink, or a symlinked file. The audit examines only real objects, never
    /// a redirect's target, so this is untrusted like `Unknown`, but it is there on every retry.
    /// Only turning the folder-permission audit off allows such a layout.
    Redirected(String),
}

impl DaclAudit {
    /// Whether it is safe to spawn a child process from this directory.
    ///
    /// `Unknown` returns false. That is the point: a caller that cannot establish safety must not
    /// proceed as though it had.
    pub fn is_safe_to_spawn(&self) -> bool {
        matches!(self, DaclAudit::OwnerOnly)
    }

    /// Distinct human-readable names of every untrusted principal with write access, sorted.
    ///
    /// Empty for `OwnerOnly`, `Unknown` and `Redirected`: only `WritableByOthers` names anyone, so
    /// an empty list means "no principal to list", never "safe"; use
    /// [`DaclAudit::is_safe_to_spawn`] for that.
    pub fn principals(&self) -> Vec<&str> {
        let DaclAudit::WritableByOthers(grants) = self else {
            return Vec::new();
        };
        let mut names: Vec<&str> = grants.iter().map(WritableGrant::principal).collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Narrow to only the outcomes that forbid spawning a child process from the directory.
    ///
    /// `Ok(())` for [`DaclAudit::OwnerOnly`]; otherwise the reason spawning must be refused, as an
    /// [`UnsafeDacl`] — a type that, unlike `DaclAudit` itself, has no variant meaning "safe". A
    /// caller building an error around "why the spawn was refused" (see
    /// [`crate::error::ProbeError::BackendDirUnsafe`]) therefore cannot construct one whose payload
    /// is, in fact, fine to spawn from.
    pub fn into_unsafe(self) -> Result<(), UnsafeDacl> {
        match self {
            DaclAudit::OwnerOnly => Ok(()),
            DaclAudit::WritableByOthers(grants) => Err(UnsafeDacl::WritableByOthers(grants)),
            DaclAudit::Unknown(message) => Err(UnsafeDacl::Unknown(message)),
            DaclAudit::Redirected(message) => Err(UnsafeDacl::Redirected(message)),
        }
    }
}

/// The outcomes of [`audit_dacl`] that forbid spawning a child process from the directory.
///
/// [`DaclAudit::into_unsafe`] is the only way to build one, and it has no arm that can produce this
/// type from [`DaclAudit::OwnerOnly`] — so a value of this type is, by construction, always a reason
/// to refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsafeDacl {
    /// See [`DaclAudit::WritableByOthers`].
    WritableByOthers(Vec<WritableGrant>),
    /// See [`DaclAudit::Unknown`].
    Unknown(String),
    /// See [`DaclAudit::Redirected`].
    Redirected(String),
}

impl std::fmt::Display for UnsafeDacl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// How many individual grants to spell out before summarizing the rest. The development
        /// build alone carries well over a hundred (one pair of ACEs repeated across every file);
        /// rendering every one of them into a log line or a UI string is not useful to a human and
        /// is not bounded in size.
        const SHOWN: usize = 5;
        match self {
            UnsafeDacl::WritableByOthers(grants) => {
                let mut principals: Vec<&str> = grants.iter().map(WritableGrant::principal).collect();
                principals.sort_unstable();
                principals.dedup();
                write!(f, "{} write-capable grants to {}", grants.len(), principals.join(", "))?;
                for g in grants.iter().take(SHOWN) {
                    write!(f, "; {g}")?;
                }
                if grants.len() > SHOWN {
                    write!(f, "; and {} more", grants.len() - SHOWN)?;
                }
                Ok(())
            }
            UnsafeDacl::Unknown(message) | UnsafeDacl::Redirected(message) => {
                write!(f, "{message}")
            }
        }
    }
}

/// Which kind of file-system object a set of rights is interpreted against.
///
/// The same access-mask bit means different things on each: `0x2` is `FILE_WRITE_DATA` on a file
/// and `FILE_ADD_FILE` on a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AuditedObject {
    /// The backend directory itself.
    Directory,
    /// A regular file, existing or (for [`GrantVia::InheritableByNewFiles`]) created later.
    File,
    /// An ancestor directory on the path used to reach the backend directory, or an intermediate
    /// path component within that chain. See [`audit_dacl`]'s "path pinning" section: `dir`'s own
    /// clean audit is worthless if the path used to reach it can be swapped out from under it.
    Ancestor,
}

/// How an untrusted principal came to hold write access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GrantVia {
    /// An access-allowed ACE (explicit or inherited) that applies to the object itself.
    Ace,
    /// An inherit-only, object-inherit ACE on the directory: it grants nothing on the directory
    /// today, but every file created there later receives it, including a planted DLL.
    InheritableByNewFiles,
    /// The object has a NULL DACL, which grants everyone full control.
    NullDacl,
    /// The object — the directory, a file inside it, or a path ancestor — is owned by a principal
    /// outside the trusted set. An owner is implicitly granted `WRITE_DAC` whatever the DACL says,
    /// so it can grant itself write access at will.
    UntrustedOwner,
}

/// One untrusted principal holding write-capable rights on one audited path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WritableGrant {
    path: PathBuf,
    applies_to: AuditedObject,
    principal: String,
    sid: String,
    access_mask: u32,
    rights: Vec<&'static str>,
    via: GrantVia,
}

impl WritableGrant {
    /// The directory, file, or path ancestor whose security descriptor carries the grant.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What the rights apply to. For [`GrantVia::InheritableByNewFiles`] this is
    /// [`AuditedObject::File`] even though [`WritableGrant::path`] is the directory.
    pub fn applies_to(&self) -> AuditedObject {
        self.applies_to
    }

    /// `DOMAIN\name` as the account database resolves it, or the SID string when it does not
    /// resolve (deleted accounts, capability SIDs, foreign domains).
    pub fn principal(&self) -> &str {
        &self.principal
    }

    /// The principal's SID in `S-1-...` form. Unlike the name, this is unambiguous.
    pub fn sid(&self) -> &str {
        &self.sid
    }

    /// Only the write-capable bits of the granted mask, not the full ACE mask.
    pub fn access_mask(&self) -> u32 {
        self.access_mask
    }

    /// Names of the bits in [`WritableGrant::access_mask`], using the directory or file spelling
    /// according to [`WritableGrant::applies_to`].
    pub fn rights(&self) -> &[&'static str] {
        &self.rights
    }

    /// How the principal came to hold these rights.
    pub fn via(&self) -> GrantVia {
        self.via
    }

    /// Build a grant directly, for tests outside this crate that need a [`DaclAudit`] value without
    /// going through the real Win32 audit (`cerebro-supervisor`'s preflight tests fake [`DaclAudit`]
    /// so they never depend on the ambient ACL of wherever `%TEMP%` happens to point on the machine
    /// running them).
    ///
    /// Gated by the `test-support` feature rather than `#[cfg(test)]`: the caller compiles this
    /// crate as an ordinary dependency, not as this crate's own test build, so `cfg(test)` would
    /// never be active for it.
    #[cfg(feature = "test-support")]
    pub fn test_new(
        path: PathBuf,
        applies_to: AuditedObject,
        principal: impl Into<String>,
        sid: impl Into<String>,
        access_mask: u32,
        rights: Vec<&'static str>,
        via: GrantVia,
    ) -> Self {
        Self {
            path,
            applies_to,
            principal: principal.into(),
            sid: sid.into(),
            access_mask,
            rights,
            via,
        }
    }
}

impl std::fmt::Display for WritableGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let how = match self.via {
            GrantVia::Ace => "",
            GrantVia::InheritableByNewFiles => " (inherited by files created there later)",
            GrantVia::NullDacl => " (NULL DACL)",
            GrantVia::UntrustedOwner => " (as owner)",
        };
        write!(
            f,
            "{} [{}] can {} on {}{how}",
            self.principal,
            self.sid,
            self.rights.join("|"),
            self.path.display()
        )
    }
}

/// Whether [`probe_with`] gates its spawns on [`audit_dacl`] reporting [`DaclAudit::OwnerOnly`].
///
/// The owner rule this exists for: any protection the user can turn off in settings must actually be
/// off everywhere it is enforced, not merely at whichever layer first added a switch for it. Without
/// this, a user who turned the folder-permission audit off in `cerebro-supervisor`'s preflight gate
/// (see `cerebro_supervisor::protections::Protections::audit_folder_permissions`) could still launch
/// a `llama-server` through the supervisor from a backend directory the audit calls unsafe, yet
/// [`probe`] — which every measurement's [`BackendIdentity`] is keyed on — would keep refusing to
/// even identify that same directory. Derive this value from the supervisor's settings with
/// `cerebro_supervisor::protections::Protections::folder_audit`, the one mapping from that setting to
/// this switch, rather than building it separately, so both crates always honour the same answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FolderAudit {
    /// Refuse with [`ProbeError::BackendDirUnsafe`] unless [`audit_dacl`] reports
    /// [`DaclAudit::OwnerOnly`]. The default, and what [`probe`] always uses.
    Enforce,
    /// Skip the audit call entirely, because the user turned this protection off in settings.
    /// Recorded on the resulting [`BackendProbe::folder_audit`] rather than merely acted on and
    /// forgotten — a settings UI listing what is currently disabled must be able to read it back.
    SkippedByUserSetting,
}

/// Whether [`probe_with`]'s `--version`/`--list-devices` queries run with a closed environment
/// ([`crate::child_env::build_child_env`]) or inherit Cerebro's own.
///
/// The same owner rule as [`FolderAudit`]: the query children load every ggml DLL, exactly like the
/// supervised server, so the user's environment-allowlist setting must reach them too. With it on,
/// an inherited `GGML_BACKEND_PATH` (which makes `ggml.dll` `LoadLibrary` a file outside the audited
/// and hashed directory) or `LLAMA_ARG_*` never reaches the query. Derive it with
/// `cerebro_supervisor::protections::Protections::probe_settings`, never separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentAllowlist {
    /// Rebuild the query child's environment from nothing: `PATH` is the backend directory then the
    /// real System32, `SystemRoot`/`windir`/`SystemDrive` come from `GetSystemDirectoryW`, and
    /// nothing else inherited survives except two scalar CPU facts. The default.
    Enforce,
    /// The query child inherits Cerebro's environment, with the backend directory and System32
    /// still first on `PATH` (see [`crate::child_env::effective_child_env`]), because the user
    /// turned this protection off in settings. Recorded on [`BackendProbe::environment_allowlist`].
    SkippedByUserSetting,
}

/// Whether [`probe_with`] refuses to run its `--version`/`--list-devices` queries while this
/// process is elevated.
///
/// The same owner rule as [`FolderAudit`]: the queries start the backend binary and load every
/// ggml DLL, exactly like the supervised server, so the elevation refusal the supervisor applies to
/// a launch must apply here too, or an elevated Cerebro would run backend code elevated just by
/// identifying it. Derive it with `cerebro_supervisor::protections::Protections::probe_settings`
/// (from `refuse_elevation`), never separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElevationRefusal {
    /// Refuse with [`ProbeError::RunningElevated`] when [`crate::token::is_elevated`] says so, and
    /// with [`ProbeError::ElevationCheckFailed`] when it cannot tell. The default.
    Enforce,
    /// Skip the token query entirely and run the queries whatever the token, because the user
    /// turned this protection off in settings. Recorded on [`BackendProbe::elevation_refusal`].
    SkippedByUserSetting,
}

/// Every user-switchable protection [`probe_with`] honours, in one value so a caller passes the
/// whole setting and cannot forget one switch. `Default` is every protection on, which is what
/// [`probe`] uses.
///
/// Build it from the supervisor's settings with
/// `cerebro_supervisor::protections::Protections::probe_settings`, the one mapping, so the probe and
/// the supervised launch always honour the same answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeSettings {
    /// See [`FolderAudit`].
    pub folder_audit: FolderAudit,
    /// See [`EnvironmentAllowlist`].
    pub environment_allowlist: EnvironmentAllowlist,
    /// See [`ElevationRefusal`].
    pub elevation_refusal: ElevationRefusal,
}

impl Default for ProbeSettings {
    /// Every protection on.
    fn default() -> Self {
        Self {
            folder_audit: FolderAudit::Enforce,
            environment_allowlist: EnvironmentAllowlist::Enforce,
            elevation_refusal: ElevationRefusal::Enforce,
        }
    }
}

/// Everything known about one backend build.
#[derive(Debug, Clone)]
pub struct BackendProbe {
    /// Build identity and file hashes.
    pub identity: BackendIdentity,
    /// Devices this binary can actually target.
    pub addressable: Vec<AddressableDevice>,
    /// Whether the folder-permission audit ran for this probe, or was skipped by user setting — see
    /// [`FolderAudit`]. Always [`FolderAudit::Enforce`] for a [`BackendProbe`] returned by [`probe`];
    /// only [`probe_with`] can produce the other value.
    pub folder_audit: FolderAudit,
    /// Whether the queries ran with the closed environment, or inherited Cerebro's by user setting —
    /// see [`EnvironmentAllowlist`]. Recorded, not merely acted on, so a settings UI can read back
    /// which protections a stored identity was taken without.
    pub environment_allowlist: EnvironmentAllowlist,
    /// Whether the queries were gated on this process not being elevated, or ran whatever the token
    /// by user setting — see [`ElevationRefusal`].
    pub elevation_refusal: ElevationRefusal,
}

/// Whether `name` is a single, bare `.exe` file name that stays inside whatever directory it is
/// joined to: the one definition every crate that launches a backend tool checks against, so a
/// hardening applied to one caller cannot leave another permissive.
///
/// Exactly one [`std::path::Component::Normal`] (rules out `..`, `.`, a bare drive letter, an
/// absolute path that would make `Path::join` discard the directory it was meant to join with, and a
/// path with any separator in it at all), no trailing `.` or ` ` (Windows silently strips both, so
/// `"llama-server.exe."` and `"llama-server.exe"` would otherwise name the same file), no `:`
/// (which would address an alternate data stream on an otherwise-legitimate file name), and a
/// `.exe` suffix. The suffix also refuses `.bat` and `.cmd`, which `std::process::Command` runs
/// through `cmd.exe`, and a `.dll`, which is not a thing to launch.
pub fn is_bare_exe_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    let is_single_normal_component = matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
    );
    is_single_normal_component
        && !name.ends_with(['.', ' '])
        && !name.contains(':')
        && name.to_ascii_lowercase().ends_with(".exe")
}

/// Every known tool-name shape [`is_bare_exe_name`] must reject, in one place. Public so each crate
/// relying on the predicate (this one's [`probe_with`], the supervisor's preflight and spawn) tests
/// its own typed refusal against the same table instead of keeping a copy that could drift.
pub const REJECTED_TOOL_NAMES: &[&str] = &[
    r"..\evil.exe",
    "../evil.exe",
    r"C:\Windows\System32\cmd.exe",
    r"\\?\C:\evil.exe",
    r"sub\tool.exe",
    "sub/dir/tool.exe",
    "llama-server.exe:ads",
    "llama-server.exe.",
    "llama-server.exe ",
    "llama-server.dll",
    "x.bat",
    "x.cmd",
    "llama-server",
    "..",
    "",
];

/// Probe a llama.cpp build directory with every protection on — see [`probe_with`] for the version
/// whose caller passes the user's settings, and for the full documentation of what this does.
pub fn probe(dir: &Path, tool_name: &str) -> Result<BackendProbe, ProbeError> {
    probe_with(dir, tool_name, ProbeSettings::default())
}

/// [`probe`], honouring the user's protection settings — see [`ProbeSettings`], and
/// [`FolderAudit`]'s docs for the owner rule this exists for.
///
/// Both queries run through this crate's private `run` helper: working directory pinned to the
/// resolved `dir`, environment per [`EnvironmentAllowlist`] (built by
/// [`crate::child_env::effective_child_env`], the same function the supervised spawn uses), no
/// console window, both output streams bounded, and a deadline after which the tool is killed and
/// [`ProbeError::ToolTimedOut`] returned.
///
/// `tool` is the executable used for the `--version` and `--list-devices` queries; it must be a bare
/// `.exe` name inside `dir` ([`is_bare_exe_name`]), which is checked rather than assumed so a caller
/// cannot be talked into running a binary from somewhere else, or a batch file through `cmd.exe`,
/// by a crafted config.
///
/// `dir` is resolved to its canonical on-disk path before anything else touches it, on Windows —
/// [`canonical_backend_dir`] (local fixed volume only, every reparse point followed, `dir`'s own
/// final component included) when
/// `settings.folder_audit` is [`FolderAudit::Enforce`], or [`canonical_backend_dir_unaudited`] (no volume
/// requirement, a resolved UNC path kept rather than refused) when it is
/// [`FolderAudit::SkippedByUserSetting`]. A
/// junction or `subst` alias anywhere in `dir`'s path (`dir` itself included) is therefore resolved
/// exactly once either way, and every step below — the DACL audit (when it runs), both queries, and the file hashes —
/// runs against that same resolved directory, which is also what [`BackendIdentity::dir`] ends up
/// recording. The audit gets it through [`audit_dacl_resolved`], which never resolves it again and
/// fails closed if it no longer resolves to itself, so a junction swapped in after this resolution
/// cannot make the audit examine a different directory from the one hashed and spawned from.
///
/// A resolution failure is reported as [`ProbeError::BackendDirUnsafe`] carrying
/// [`UnsafeDacl::Unknown`] under [`FolderAudit::Enforce`] (the audit could not complete, so the
/// directory is treated as untrusted), and as [`ProbeError::BackendDirUnresolvable`] under
/// [`FolderAudit::SkippedByUserSetting`] (no audit ran, so it is a plain resolution error).
///
/// With `settings.folder_audit` at its default, [`FolderAudit::Enforce`], the audit runs before either query
/// and gates them: if it does not report [`DaclAudit::OwnerOnly`], this returns
/// [`ProbeError::BackendDirUnsafe`] without spawning anything, because `--list-devices` alone is
/// enough to make ggml load every backend DLL an untrusted principal left in `dir`.
/// [`FolderAudit::SkippedByUserSetting`] skips that call entirely — the user has turned this
/// protection off in settings — and both queries proceed directly. Either way they are run with the
/// child's working directory pinned to `dir` (by this crate's private `probe_command` helper),
/// never anywhere else ggml's loader might also search.
///
/// # Platform
///
/// [`audit_dacl`] cannot establish safety off Windows, so on every other platform
/// [`FolderAudit::Enforce`] always returns [`ProbeError::BackendDirUnsafe`] with
/// [`UnsafeDacl::Unknown`] and never spawns anything — deliberate fail-closed behavior, not a
/// missing port. [`FolderAudit::SkippedByUserSetting`] has no such fallback to fail closed *into*,
/// so it is refused outright off Windows instead (`ProbeError::BackendDirUnsafe` with
/// `UnsafeDacl::Unknown`), rather than silently resolving `dir` unaudited on a platform this crate
/// has no path-resolution logic for at all.
///
/// # Elevation
///
/// With `settings.elevation_refusal` at its default, [`ElevationRefusal::Enforce`], the process
/// token is checked after the folder audit and before the first query: an elevated process gets
/// [`ProbeError::RunningElevated`] and nothing is spawned, and a token that cannot be queried gets
/// [`ProbeError::ElevationCheckFailed`] (fail closed). [`ElevationRefusal::SkippedByUserSetting`]
/// skips the token query entirely. This probe does not verify the backend against a pinned manifest:
/// it *computes* the identity (hashes included) that a manifest is later checked against, which the
/// supervisor's preflight does before every launch.
pub fn probe_with(
    dir: &Path,
    tool_name: &str,
    settings: ProbeSettings,
) -> Result<BackendProbe, ProbeError> {
    probe_with_token(dir, tool_name, settings, &crate::token::is_elevated, None)
}

/// [`probe_with`], with the queries' GPU driver caches pinned to `cache_dir`: `CUDA_CACHE_PATH`,
/// `LOCALAPPDATA` and `APPDATA` point there (see [`crate::child_env`]'s "Driver caches"). Pass the
/// same directory the supervised server gets (`SpawnOptions::cache_dir`), so identifying a backend
/// warms the kernel cache the server then loads from instead of compiling into a folder nothing
/// reuses; on a GPU the build carries only PTX for, [`probe_with`] recompiles on every call.
///
/// `cache_dir` must be an absolute path to an existing directory, or this returns
/// [`ProbeError::CacheDirUnusable`] before anything is spawned, rather than silently dropping the
/// cache and paying the compile again. Who owns it, and whether a junction on its path is
/// acceptable, is the caller's policy (`cerebro_supervisor::spawn::verify_cache_dir` applies the
/// user's redirected-path setting); pass the path that check returned.
pub fn probe_with_cache(
    dir: &Path,
    tool_name: &str,
    settings: ProbeSettings,
    cache_dir: &Path,
) -> Result<BackendProbe, ProbeError> {
    probe_with_token(
        dir,
        tool_name,
        settings,
        &crate::token::is_elevated,
        Some(cache_dir),
    )
}

/// [`probe_with`] with the elevation answer injected, so both states of [`ElevationRefusal`] are
/// testable without running the test suite elevated, and the optional driver cache directory of
/// [`probe_with_cache`]. The only production callers pass [`crate::token::is_elevated`].
fn probe_with_token(
    dir: &Path,
    tool_name: &str,
    settings: ProbeSettings,
    is_elevated: &dyn Fn() -> Result<bool, ProbeError>,
    cache_dir: Option<&Path>,
) -> Result<BackendProbe, ProbeError> {
    let audit = settings.folder_audit;
    if !dir.is_dir() {
        return Err(ProbeError::BackendDirMissing {
            dir: dir.display().to_string(),
        });
    }
    if !is_bare_exe_name(tool_name) {
        return Err(ProbeError::BackendToolEscapesDir {
            tool: tool_name.to_string(),
        });
    }

    // Resolved exactly once, before anything else touches disk: a junction or `subst` alias
    // anywhere in `dir`'s ancestors must not let the audit, the hash, and the spawn each reach it
    // through a different spelling, or through a second resolution of the same spelling. The audit
    // below takes this exact value and never resolves it again. See `resolve_backend_dir`.
    let resolved = resolve_backend_dir(dir, audit)?;
    let dir: &Path = &resolved;

    let tool = dir.join(tool_name);
    if !tool.is_file() {
        return Err(ProbeError::BackendToolMissing {
            tool: tool.display().to_string(),
        });
    }

    // The audit must run, and pass, *before* anything is spawned from `dir`: ggml loads every
    // backend DLL it finds there as soon as the child process starts, so a directory an untrusted
    // principal can write to must never reach `run()` at all. Skipped entirely under
    // `FolderAudit::SkippedByUserSetting` — the user has turned this protection off.
    //
    // `audit_dacl_resolved`, never `audit_dacl`: the latter would resolve `dir` a second time, and a
    // junction swapped in between would make it audit a different directory from the one hashed and
    // spawned from below. The resolved-path audit fails closed on that instead.
    #[cfg(windows)]
    if audit == FolderAudit::Enforce {
        if let Err(audit) = audit_dacl_resolved(dir).into_unsafe() {
            return Err(ProbeError::BackendDirUnsafe {
                dir: dir.display().to_string(),
                audit,
            });
        }
    }

    // Elevation, before anything is spawned: `--list-devices` loads every ggml DLL, so an elevated
    // query would run backend code elevated. Off skips the token query itself, so a token this
    // account cannot query never blocks a user who turned this protection off.
    if settings.elevation_refusal == ElevationRefusal::Enforce {
        let elevated = is_elevated().map_err(|e| ProbeError::ElevationCheckFailed {
            cause: e.to_string(),
        })?;
        if elevated {
            return Err(ProbeError::RunningElevated);
        }
    }

    // Before anything is spawned: a cache the drivers cannot use would silently cost a recompile on
    // every query instead of failing visibly.
    if let Some(cache) = cache_dir {
        if !cache.is_absolute() || !cache.is_dir() {
            return Err(ProbeError::CacheDirUnusable {
                dir: cache.display().to_string(),
                cause: "must be an absolute path to an existing directory".to_string(),
            });
        }
    }

    let env = settings.environment_allowlist;
    let version_raw = run(&tool, dir, &["--version"], env, cache_dir)?;
    let (build_number, commit, compiler) = parse_version(&version_raw);
    let devices_raw = run(&tool, dir, &["--list-devices"], env, cache_dir)?;
    let addressable = parse_list_devices(&devices_raw)?;
    let (hashes, ggml_backends) = hash_directory(dir)?;

    Ok(BackendProbe {
        identity: BackendIdentity {
            dir: dir.to_path_buf(),
            version: version_raw.lines().next().unwrap_or_default().trim().to_string(),
            build_number,
            commit,
            compiler,
            hashes,
            ggml_backends,
        },
        addressable,
        folder_audit: audit,
        environment_allowlist: env,
        elevation_refusal: settings.elevation_refusal,
    })
}

/// Resolve `dir` once to the canonical path [`probe_with`] audits, queries and hashes.
///
/// [`FolderAudit::Enforce`] uses `dacl::canonicalize` (local fixed volume only) and reports a
/// failure as [`ProbeError::BackendDirUnsafe`] with [`UnsafeDacl::Unknown`]: the audit could not
/// complete, so the directory is untrusted. [`FolderAudit::SkippedByUserSetting`] uses
/// `dacl::plain_canonicalize`, whose failure is a plain resolution error because no audit ran.
#[cfg(windows)]
fn resolve_backend_dir(dir: &Path, audit: FolderAudit) -> Result<PathBuf, ProbeError> {
    match audit {
        FolderAudit::Enforce => dacl::canonicalize(dir).map_err(|e| ProbeError::BackendDirUnsafe {
            dir: dir.display().to_string(),
            audit: UnsafeDacl::Unknown(format!(
                "DACL audit could not complete: {e}; treat as untrusted"
            )),
        }),
        FolderAudit::SkippedByUserSetting => dacl::plain_canonicalize(dir),
    }
}

/// Off Windows there is no resolution logic for either shape, so this always fails closed,
/// matching [`audit_dacl`]'s stance there. A separate function rather than an early `return` inside
/// [`probe_with`], so the rest of that body neither becomes unreachable code nor fails to type-check
/// on other platforms.
#[cfg(not(windows))]
fn resolve_backend_dir(dir: &Path, audit: FolderAudit) -> Result<PathBuf, ProbeError> {
    let _ = audit;
    Err(ProbeError::BackendDirUnsafe {
        dir: dir.display().to_string(),
        audit: UnsafeDacl::Unknown(
            "backend directory resolution is implemented for Windows only; treat as untrusted"
                .to_string(),
        ),
    })
}

/// Audit who, other than trusted principals, can change the code a spawn from `dir` would load.
///
/// Audits three things, non-recursively: every ancestor directory on `dir`'s absolute path (delete
/// or re-ACL rights only — reported as [`AuditedObject::Ancestor`]), `dir` itself, and every regular
/// file directly inside it, since a file can carry an explicit ACE its directory does not. Existing
/// subdirectories are never descended into: creating a new one already requires
/// `FILE_ADD_SUBDIRECTORY` on `dir`, which the directory's own audit already covers.
///
/// # Trusted principals
///
/// The current process user, `SYSTEM` (S-1-5-18), `Administrators` (S-1-5-32-544), and
/// `TrustedInstaller`. `CREATOR OWNER` (S-1-3-0) is trusted only on inheritable ACEs, where it
/// stands for whoever creates the file — who already needed `FILE_ADD_FILE`, audited separately.
/// Everyone else is untrusted, including `Authenticated Users`, `Users`, `Everyone`, and capability
/// or app-container SIDs. An object's owner is trusted only when the owner itself is already one of
/// these principals: ownership by any other account still carries implicit `WRITE_DAC`, so it is
/// reported as [`GrantVia::UntrustedOwner`] rather than treated as a trust path of its own. Every
/// decision is made on SIDs alone, offline; `LookupAccountSidW` is consulted only to attach a
/// display name afterward, never to decide trust.
///
/// # What counts as write access
///
/// An access-allowed ACE (plain, object, callback, or callback-object — conditions on callback ACEs
/// are not evaluated, so they are assumed to hold) whose mask intersects `FILE_WRITE_DATA`,
/// `FILE_APPEND_DATA`, `DELETE`, `WRITE_DAC`, `WRITE_OWNER`, `GENERIC_WRITE`, `GENERIC_ALL`, or
/// `MAXIMUM_ALLOWED`. On the directory, `FILE_DELETE_CHILD` also counts: it deletes a DLL regardless
/// of that file's own DACL. `FILE_WRITE_EA` and `FILE_WRITE_ATTRIBUTES` alone do not count: neither
/// changes a file's contents or who may change them. Inherit-only, object-inherit ACEs on the
/// directory count too, because they land on every file created there later
/// ([`GrantVia::InheritableByNewFiles`]); container-inherit-only ACEs do not, since subdirectories
/// are out of scope. A NULL DACL grants everyone full control; an empty DACL grants nobody anything.
/// Deny ACEs are ignored rather than subtracted, which can only over-report, never under-report.
///
/// # Scope limits, all fail-closed to [`DaclAudit::Unknown`]
///
/// Rejected outright, before a single security descriptor is read: any path that is not a local
/// disk path (a UNC share, a verbatim UNC or device-namespace path), and any path whose volume is
/// not `DRIVE_FIXED` or `DRIVE_RAMDISK` — a mapped network drive included. The well-known SIDs above
/// name *this machine's* principals only when this machine issued the descriptor being read.
///
/// `subst`/`DefineDosDevice` aliases and junctions or symlinks in any *ancestor* of `dir` are
/// resolved to the real on-disk path ([`canonical_backend_dir`]), and that canonical chain is what
/// the ancestor walk audits. The alias itself, and the directories that lead to it, are not
/// audited: whoever can repoint an ancestor junction chooses which directory the canonical path
/// names, which is why callers should record the resolved path (as `cerebro-supervisor`'s launch
/// record does) rather than trust the as-given spelling. A reparse point that is `dir`'s own final
/// component, or an entry found while enumerating `dir`, fails the result closed. So does any other
/// Win32 failure, unrecognized ACE type, or malformed ACE or SID.
///
/// This function resolves `dir` itself, so it is for callers that only want a verdict. A caller that
/// hashes or spawns must resolve once with [`canonical_backend_dir`], audit that exact value with
/// [`audit_dacl_resolved`], and hash and spawn from it. Calling this function instead would resolve
/// a second time, and a junction swapped in between the two resolutions would make the audit examine
/// a different directory from the one spawned. [`probe`] and `cerebro-supervisor`'s preflight both
/// use the resolved variant.
///
/// # Caller obligation: pin the working directory
///
/// This audits `dir` and its ancestors — never a process's current working directory. ggml's
/// backend loader scans *both* the executable directory and the current directory for
/// `ggml-<name>-*.dll` and loads every match it finds. Spawning from `dir` must therefore also set
/// the child's working directory to `dir` (as this crate's private `probe_command` helper does), or
/// a DLL sitting wherever Cerebro happened to start from can load unaudited.
pub fn audit_dacl(dir: &Path) -> DaclAudit {
    #[cfg(windows)]
    {
        audit_verdict(dacl::audit(dir))
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        DaclAudit::Unknown("DACL audit is implemented for Windows only; treat as untrusted".into())
    }
}

/// [`audit_dacl`] for a directory already resolved by [`canonical_backend_dir`], which the caller
/// will hash and spawn from. Never resolves `resolved` again.
///
/// Use this, not [`audit_dacl`], wherever a hash or a spawn follows the audit. [`audit_dacl`]
/// resolves its input itself, so between the caller's resolution and the audit's, a junction planted
/// in place of a renamed ancestor would make the audit examine the junction's target while the
/// caller goes on to spawn through the path it resolved earlier. Here, `resolved` is re-resolved
/// only to check that it still resolves to exactly itself (byte-for-byte, as both sides come from
/// `GetFinalPathNameByHandleW`); if it does not, the result is [`DaclAudit::Unknown`] saying the path
/// changed. Otherwise the audit walks exactly `resolved`'s components, so any junction inserted
/// later shows up as a reparse point on the chain and also fails closed. Once that chain has passed
/// the ancestor walk, no untrusted principal can alter it, so the caller's hashes stay valid up to
/// the spawn.
///
/// A path that is not already canonical (a `..` spelling, a different case, an 8.3 alias) fails the
/// same way: pass exactly what [`canonical_backend_dir`] returned.
///
/// # Platform
///
/// Windows only, like [`canonical_backend_dir`], the only source of a valid input.
#[cfg(windows)]
pub fn audit_dacl_resolved(resolved: &Path) -> DaclAudit {
    audit_verdict(dacl::audit_resolved(resolved))
}

/// The one mapping from the Win32 audit's result to a [`DaclAudit`], shared by [`audit_dacl`] and
/// [`audit_dacl_resolved`] so the two can never phrase or classify a result differently.
#[cfg(windows)]
fn audit_verdict(result: Result<Vec<WritableGrant>, ProbeError>) -> DaclAudit {
    match result {
        Ok(grants) if grants.is_empty() => DaclAudit::OwnerOnly,
        Ok(grants) => DaclAudit::WritableByOthers(grants),
        // No settings clause here: whoever reports this verdict names the switch once, in its own
        // words (`ProbeError::BackendDirUnsafe`, the supervisor's `BackendPathRedirected`).
        Err(error @ ProbeError::BackendPathHasReparsePoint { .. }) => DaclAudit::Redirected(
            format!("DACL audit refused the path: {error}; treat as untrusted"),
        ),
        Err(error) => DaclAudit::Unknown(format!(
            "DACL audit could not complete: {error}; treat as untrusted"
        )),
    }
}

/// Canonical on-disk path of a backend directory: local fixed volume only, every junction or `subst`
/// alias resolved once, `dir`'s own final component included (a `dir` that is itself a junction
/// resolves to its target, exactly as an ancestor junction does; the target is what gets audited).
///
/// This is exactly the resolution [`probe`] itself performs before touching disk further (see that
/// function's own documentation), exposed here so a caller that hashes, audits, or spawns from a
/// backend directory *outside* [`probe`] — `cerebro-supervisor`'s preflight gate, most importantly —
/// can resolve through the identical function and agree with `probe()` on one physical directory
/// rather than maintaining a second, potentially drifting implementation of the same resolution.
///
/// # Platform
///
/// Windows only, for the same reason [`audit_dacl`] is: there is no portable notion of an NTFS
/// junction or a `subst` alias to resolve on another platform.
#[cfg(windows)]
pub fn canonical_backend_dir(dir: &Path) -> Result<PathBuf, ProbeError> {
    dacl::canonicalize(dir)
}

/// Create an NTFS directory junction at `link` pointing at `target`. Needs no privilege and no
/// Developer Mode, which is why it is the reparse type an unprivileged attacker can plant and the one
/// the ancestor-junction tests must use.
///
/// Test support only: gated by the `test-support` feature (off by default, enabled only through a
/// dev-dependency), so it is never part of a release build's public API. Lets another crate's tests
/// (`cerebro-supervisor`'s preflight) plant a junction through the same `DeviceIoControl` helper
/// this crate tests, rather than shelling out to a script host.
#[cfg(all(windows, feature = "test-support"))]
pub fn create_junction_for_tests(link: &Path, target: &Path) -> Result<(), ProbeError> {
    dacl::create_junction(link, target)
}

/// Replace `path`'s entire DACL (protected, so nothing is inherited) with the one `sddl` describes,
/// leaving the owner alone. Lets another crate's tests stage an exact ACL through Win32 calls this
/// crate already tests, instead of shelling out to `icacls` (which prints into every test run and
/// resolves an account by name).
///
/// Test support only, like [`create_junction_for_tests`].
#[cfg(all(windows, feature = "test-support"))]
pub fn set_dacl_from_sddl_for_tests(path: &Path, sddl: &str) -> Result<(), ProbeError> {
    dacl::set_dacl_from_sddl(path, sddl)
}

/// The SID string (`S-1-5-21-...`) of the user this process's token represents, for building an
/// SDDL grant that names this account unambiguously (a bare user name can be ambiguous on a domain).
///
/// Test support only, like [`create_junction_for_tests`].
#[cfg(all(windows, feature = "test-support"))]
pub fn current_user_sid_for_tests() -> Result<String, ProbeError> {
    dacl::current_user_sid()
}

/// The SID string (`S-1-5-21-...`) of the user this process's token represents: the "current
/// user" every DACL audit here trusts.
///
/// For a caller that writes an owner-only ACL (`cerebro-config`'s secret store) and wants the
/// written grant to name exactly the principal [`audit_object_dacl`] will accept, rather than a
/// SDDL alias such as `OW` (OWNER RIGHTS, S-1-3-4) that the audit, deciding on SIDs alone, does not
/// treat as trusted.
///
/// # Platform
///
/// Windows only.
#[cfg(windows)]
pub fn current_user_sid() -> Result<String, ProbeError> {
    dacl::current_user_sid()
}

/// Audit exactly one file or directory: its own owner and DACL, with [`audit_dacl`]'s trusted
/// principals and write-capable rights, but no ancestor walk and no enumeration of a directory's
/// contents.
///
/// For objects whose own ACL is the whole question, such as an owner-only secret blob or a
/// per-launch key file Cerebro just created inside a folder it did not choose (a temp or profile
/// directory, whose ancestors on the development machine are themselves writable by a sandbox
/// group). Use [`audit_dacl`] for anything code will be loaded from.
///
/// Only [`DaclAudit::OwnerOnly`] means no untrusted principal can write, delete or re-ACL the
/// object. A path that is not on a local fixed volume, is missing, or cannot be read is
/// [`DaclAudit::Unknown`]; one that is itself a reparse point is [`DaclAudit::Redirected`].
pub fn audit_object_dacl(path: &Path) -> DaclAudit {
    #[cfg(windows)]
    {
        audit_verdict(dacl::audit_object(path))
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        DaclAudit::Unknown("DACL audit is implemented for Windows only; treat as untrusted".into())
    }
}

/// [`canonical_backend_dir`], but for a caller that turned the folder-permission audit off (see
/// [`FolderAudit::SkippedByUserSetting`]): every reparse point in `dir` and its ancestors is still
/// followed, but none of [`canonical_backend_dir`]'s trust preconditions apply — a mapped network
/// drive, a UNC path, or a drive with no letter all resolve here instead of being refused, and a
/// resolved UNC path is kept (`\\server\share\...`) rather than turned relative or rejected.
///
/// This is real path resolution, not a bypass of it: the result is still the literal, absolute,
/// on-disk path a spawn from `dir` would use. What it does *not* do is vouch for who else can write
/// to it — that vouching is exactly what the folder-permission audit provides, and a caller reaching
/// this function has, by construction, turned that off.
///
/// Exposed here for the same reason [`canonical_backend_dir`] is: so `cerebro-supervisor`'s preflight
/// gate resolves through the identical function [`probe_with`] uses internally
/// ([`FolderAudit::SkippedByUserSetting`]'s branch), rather than maintaining a second implementation.
///
/// # Platform
///
/// Windows only, for the same reason [`canonical_backend_dir`] is.
#[cfg(windows)]
pub fn canonical_backend_dir_unaudited(dir: &Path) -> Result<PathBuf, ProbeError> {
    dacl::plain_canonicalize(dir)
}

/// Build a query command for a backend tool: working directory pinned to `dir`, environment per the
/// user's [`EnvironmentAllowlist`] setting, no console window, and both output streams piped.
///
/// The working directory matters because ggml's backend loader (`ggml_backend_load_best`) scans both
/// the directory the executable lives in *and the process's current directory* for
/// `ggml-<name>-*.dll`, loading every match. Only `dir` is what [`audit_dacl`] examined, so a spawn
/// that left the child's working directory at Cerebro's own could load a DLL the audit never saw.
///
/// The environment matters for the same reason: `ggml.dll` honours `GGML_BACKEND_PATH` and
/// `LoadLibrary`s the file it names, so an inherited value would load a DLL from outside the
/// directory just resolved, audited and about to be hashed. The block comes from
/// [`crate::child_env::effective_child_env`], the one function the supervised spawn also uses, with
/// no temp directory (these queries write nothing), no `CUDA_VISIBLE_DEVICES` (every device is
/// listed), and the driver cache directory [`probe_with_cache`] was given, if any (initialising the
/// driver for `--list-devices` compiles kernels too).
///
/// `CREATE_NO_WINDOW` is set unconditionally: from a GUI parent, a console-subsystem child otherwise
/// opens a visible console window for every query.
///
/// `tool` must already be the absolute path to the executable (`dir.join(tool_name)`, as `probe`
/// builds it): `Command::current_dir` changes where the child starts, not how Windows resolves a
/// relative program name.
fn probe_command(
    tool: &Path,
    dir: &Path,
    args: &[&str],
    env: EnvironmentAllowlist,
    cache_dir: Option<&Path>,
) -> Result<Command, ProbeError> {
    let mut cmd = Command::new(tool);
    cmd.args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

        // From the kernel, never from an inherited `SystemRoot`: see `child_env`'s module docs.
        let system32 = crate::child_env::system_directory()?;
        let block = crate::child_env::effective_child_env(
            env == EnvironmentAllowlist::Enforce,
            std::env::vars_os(),
            &system32,
            dir,
            None,
            None,
            cache_dir,
        );
        cmd.env_clear().envs(block).creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        // `probe_with` refuses before any query off Windows; nothing reaches here in practice.
        let _ = (env, cache_dir);
    }
    Ok(cmd)
}

/// How long one `--version` or `--list-devices` query may take before the tool is killed. Both
/// finish in well under a second on the development machine (CUDA initialisation included); the
/// margin covers a slow first driver initialisation, while a hung driver still cannot block the
/// probe forever.
const QUERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Bytes kept from each of a query's output streams. Real `--list-devices` output is a few hundred
/// bytes; anything past this cap is drained and discarded so the child never blocks on a full pipe.
const MAX_QUERY_OUTPUT: u64 = 1 << 20;

/// Run one backend query and return its stdout and stderr, merged.
fn run(
    tool: &Path,
    dir: &Path,
    args: &[&str],
    env: EnvironmentAllowlist,
    cache_dir: Option<&Path>,
) -> Result<String, ProbeError> {
    let child = probe_command(tool, dir, args, env, cache_dir)?
        .spawn()
        .map_err(|e| tool_spawn_error(tool, &e))?;
    collect_bounded(child, tool, QUERY_TIMEOUT)
}

fn tool_spawn_error(tool: &Path, e: &std::io::Error) -> ProbeError {
    ProbeError::ToolSpawn {
        tool: tool.display().to_string(),
        cause: e.to_string(),
    }
}

/// Wait for `child` to exit and both of its output streams to close, within `timeout`, keeping at
/// most [`MAX_QUERY_OUTPUT`] bytes of each. On the deadline the child is killed and reaped, and
/// [`ProbeError::ToolTimedOut`] returned; it is never left running.
///
/// llama.cpp writes both `--version` and `--list-devices` to stderr, so the streams are merged
/// (stdout first) rather than reading stdout alone and concluding the tool printed nothing.
fn collect_bounded(child: Child, tool: &Path, timeout: Duration) -> Result<String, ProbeError> {
    let collected = collect_bounded_streams(child, tool, timeout)?;
    let mut text = String::from_utf8_lossy(&collected.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&collected.stderr));
    Ok(text)
}

/// What a bounded tool run left behind: how it exited and what it wrote (each stream capped at
/// [`MAX_QUERY_OUTPUT`] bytes; a stream that was not piped is empty).
#[derive(Debug)]
pub(crate) struct Collected {
    pub(crate) status: std::process::ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

/// The engine of [`collect_bounded`], shared with the vendor-tool queries in [`crate::gpu`]: wait
/// for `child` to exit and its piped streams to close within `timeout`, keeping at most
/// [`MAX_QUERY_OUTPUT`] bytes of each. On the deadline, or any failure, the child is killed and
/// reaped before this returns; it is never left running.
pub(crate) fn collect_bounded_streams(
    mut child: Child,
    tool: &Path,
    timeout: Duration,
) -> Result<Collected, ProbeError> {
    let started = Instant::now();
    let timed_out = || ProbeError::ToolTimedOut {
        tool: tool.display().to_string(),
        after: timeout,
    };
    // Kill and reap on every early return, so no path leaves the child running.
    let abandon = |child: &mut Child, err: ProbeError| {
        let _ = child.kill();
        let _ = child.wait();
        err
    };

    let stdout = match child.stdout.take().map(drain_bounded).transpose() {
        Ok(rx) => rx,
        Err(e) => return Err(abandon(&mut child, tool_spawn_error(tool, &e))),
    };
    let stderr = match child.stderr.take().map(drain_bounded).transpose() {
        Ok(rx) => rx,
        Err(e) => return Err(abandon(&mut child, tool_spawn_error(tool, &e))),
    };

    let status = match wait_for_exit(&mut child, timeout) {
        Ok(Some(status)) => status,
        Ok(None) => return Err(abandon(&mut child, timed_out())),
        Err(e) => return Err(abandon(&mut child, tool_spawn_error(tool, &e))),
    };

    // The child has exited, but a grandchild that inherited a pipe could still hold it open, so
    // collecting the output is bounded by the same deadline.
    let collect = |rx: Option<std::sync::mpsc::Receiver<Vec<u8>>>| match rx {
        None => Ok(Vec::new()),
        Some(rx) => rx
            .recv_timeout(timeout.saturating_sub(started.elapsed()))
            .map_err(|_| timed_out()),
    };
    let stdout = collect(stdout)?;
    let stderr = collect(stderr)?;
    Ok(Collected {
        status,
        stdout,
        stderr,
    })
}

/// Read up to [`MAX_QUERY_OUTPUT`] bytes of `reader` on a named background thread, then drain and
/// discard the rest until EOF (so the child never blocks on a full pipe), and send what was kept.
///
/// `std::thread::Builder`, not `std::thread::spawn`, which panics if the thread cannot be created.
fn drain_bounded<R>(reader: R) -> std::io::Result<std::sync::mpsc::Receiver<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("cerebro-probe-query-reader".to_string())
        .spawn(move || {
            let mut kept = Vec::new();
            let mut limited = reader.take(MAX_QUERY_OUTPUT);
            // A read error ends the stream early; whatever arrived before it is still reported.
            let _ = limited.read_to_end(&mut kept);
            let _ = std::io::copy(&mut limited.into_inner(), &mut std::io::sink());
            let _ = tx.send(kept);
        })
        .map(drop)?;
    Ok(rx)
}

/// `child`'s exit status if it exited within `timeout` (`Ok(None)` if it is still running).
///
/// On Windows this waits on the process handle itself, so it returns the moment the child exits
/// rather than on the next tick of a polling loop: the probe runs before every tune and launch, and
/// speed is a core value.
#[cfg(windows)]
fn wait_for_exit(
    child: &mut Child,
    timeout: Duration,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    // `INFINITE` is `u32::MAX`, so the largest finite wait is one less.
    let millis = u32::try_from(timeout.as_millis())
        .unwrap_or(u32::MAX)
        .min(u32::MAX - 1);
    // SAFETY: `child` owns this live process handle for the whole call; waiting on it neither
    // closes nor transfers it.
    match unsafe { WaitForSingleObject(child.as_raw_handle(), millis) } {
        WAIT_OBJECT_0 => child.try_wait(),
        WAIT_TIMEOUT => Ok(None),
        _ => Err(std::io::Error::last_os_error()),
    }
}

/// `child`'s exit status if it exited within `timeout`: a polling fallback for platforms without a
/// waitable process handle. [`probe_with`] never reaches a query there, so latency does not matter.
#[cfg(not(windows))]
fn wait_for_exit(
    child: &mut Child,
    timeout: Duration,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if started.elapsed() >= timeout {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// What `--version` reports, each field `None` when the text does not say: (build number, short
/// commit hash, compiler from the `built with ...` line). See [`parse_version`].
pub type VersionFields = (Option<u32>, Option<String>, Option<String>);

/// Build number, commit and compiler of the backend in `dir`, from `tool_name --version`, under
/// the same command construction, environment rule, window suppression, output bound and deadline
/// as [`probe_with`]'s own query, without its folder audit, hashing or `--list-devices`.
///
/// For a directory the caller has already resolved, audited and verified (a preflighted backend
/// directory): this neither resolves nor audits `dir` again. `llama-bench` has no `--version`, so a
/// benchmark run asks a sibling tool of the same build (`llama-server.exe`); the caller should
/// check the build number it gets against the one the benchmark itself reported.
///
/// # Errors
/// [`ProbeError::BackendToolEscapesDir`] for a `tool_name` that is not a bare `.exe` name,
/// [`ProbeError::BackendToolMissing`] when it is not in `dir`, and the query's own spawn and
/// timeout errors.
pub fn query_version(
    dir: &Path,
    tool_name: &str,
    env: EnvironmentAllowlist,
) -> Result<VersionFields, ProbeError> {
    if !is_bare_exe_name(tool_name) {
        return Err(ProbeError::BackendToolEscapesDir {
            tool: tool_name.to_string(),
        });
    }
    let tool = dir.join(tool_name);
    if !tool.is_file() {
        return Err(ProbeError::BackendToolMissing {
            tool: tool.display().to_string(),
        });
    }
    Ok(parse_version(&run(&tool, dir, &["--version"], env, None)?))
}

/// Run one query exactly as [`probe_with`] does — same command construction, environment rule,
/// window suppression, output bound and deadline — and return its raw merged output.
///
/// Test support only (`test-support` feature, never in a release build's API): lets
/// `cerebro-supervisor`'s environment tests, which own the print-env fixture and the process-wide
/// environment lock, prove what environment a probe query child actually receives.
#[cfg(all(windows, feature = "test-support"))]
pub fn query_tool_for_tests(
    tool: &Path,
    dir: &Path,
    args: &[&str],
    env: EnvironmentAllowlist,
    cache_dir: Option<&Path>,
) -> Result<String, ProbeError> {
    run(tool, dir, args, env, cache_dir)
}

/// Extract build number, commit, and compiler from `--version` output.
///
/// Returns `None` per field rather than failing: upstream changes this text freely, and a probe
/// that hard-errors on an unrecognized version line would refuse to run against a newer llama.cpp
/// that is otherwise perfectly usable. The consequence is recorded honestly — a measurement keyed
/// on a `None` build number is marked as having weaker provenance rather than being silently
/// treated as comparable.
pub fn parse_version(text: &str) -> VersionFields {
    let mut build = None;
    let mut commit = None;
    let mut compiler = None;

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("version:") {
            // e.g. "version: 0.3.0-dev (build 10751, commit 3466812d1)"
            if let Some(b) = between(rest, "build ", |c| !c.is_ascii_digit()) {
                build = b.parse::<u32>().ok();
            }
            if let Some(c) = between(rest, "commit ", |c| !c.is_ascii_alphanumeric()) {
                commit = Some(c.to_string());
            }
        } else if let Some(rest) = line.strip_prefix("built with ") {
            compiler = Some(rest.trim().to_string());
        }
    }
    (build, commit, compiler)
}

/// Take the run of characters after `marker` up to the first character matching `stop`.
fn between<'a>(haystack: &'a str, marker: &str, stop: impl Fn(char) -> bool) -> Option<&'a str> {
    let start = haystack.find(marker)? + marker.len();
    let tail = &haystack[start..];
    let end = tail.find(|c: char| stop(c)).unwrap_or(tail.len());
    (end > 0).then(|| &tail[..end])
}

/// Parse `--list-devices` output into the addressable device set.
///
/// A pure function so it is covered by fixtures on machines with a different backend. Lines look
/// like `  CUDA0: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8191 MiB, 7113 MiB free)`.
pub fn parse_list_devices(text: &str) -> Result<Vec<AddressableDevice>, ProbeError> {
    let mut devices = Vec::new();
    let mut in_list = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("Available devices:") {
            in_list = true;
            continue;
        }
        if !in_list || trimmed.is_empty() {
            continue;
        }
        let Some((id, rest)) = trimmed.split_once(':') else {
            continue;
        };
        let id = id.trim();
        // An id is a backend name followed by an ordinal. Anything else is a log line that
        // happened to contain a colon, and must not become a device Cerebro would later emit.
        if id.is_empty()
            || !id.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            || !id.chars().last().is_some_and(|c| c.is_ascii_digit())
            || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }

        let rest = rest.trim();
        let (description, total_mib, free_mib) = match rest.rsplit_once('(') {
            Some((desc, mem)) => {
                let mem = mem.trim_end_matches(')');
                let total = mem
                    .split(',')
                    .next()
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<u64>().ok());
                let free = mem
                    .split(',')
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<u64>().ok());
                (desc.trim().to_string(), total, free)
            }
            None => (rest.to_string(), None, None),
        };

        devices.push(AddressableDevice {
            id: id.to_ascii_uppercase(),
            description,
            total_mib,
            free_mib,
        });
    }
    Ok(devices)
}

/// SHA-256 every executable and ggml library in the directory, non-recursively.
///
/// Pure disk I/O — this reads bytes and hashes them, it never spawns anything — so it is safe to
/// call before a directory has passed [`audit_dacl`], unlike [`probe`]. That is exactly what a
/// pre-spawn integrity check (comparing the result against a pinned manifest) needs: a caller
/// must be able to detect a planted or substituted file *before* deciding whether the directory is
/// even safe to launch from.
///
/// Returns every `.exe`/`.dll` file name (as spelled on disk) mapped to its lowercase hex SHA-256,
/// plus the `ggml-*.dll` subset as backend identifiers.
///
/// Every entry is either accounted for or the call fails: a `read_dir` entry this could not
/// classify, a file this could not open or read, or a file name that is not valid UTF-8 all become
/// a [`ProbeError`] rather than being silently skipped. An integrity check that quietly drops the
/// one entry it could not read is not a check — it is exactly the gap a planted file would hide in.
/// When several files fail, the error reported is the first in name order among those that failed,
/// not whichever thread happened to finish first.
///
/// # Speed
///
/// This runs before every launch and every restart, on the critical path to first token, so it is
/// built for throughput (measured 2026-09-24 on the official CUDA build, 54 files, 1.17 GB, warm
/// cache, release): files are hashed in parallel on up to [`MAX_HASH_THREADS`] threads, largest
/// first, each streaming through a reused [`HASH_BUFFER_BYTES`] buffer rather than reading a file
/// fully into memory (CUDA runtime DLLs are hundreds of megabytes each). That took one pass from
/// 1.12 s (serial, 8 KiB reads) to 0.47 s, which is now bounded by the single largest file
/// (`ggml-cuda.dll`, 547 MB) at the hasher's ~1.9 GB/s.
pub fn hash_directory(dir: &Path) -> Result<(BTreeMap<String, String>, Vec<String>), ProbeError> {
    let mut backends = Vec::new();
    let mut files: Vec<(String, PathBuf, u64)> = Vec::new();

    let entries = std::fs::read_dir(dir).map_err(|e| ProbeError::BackendDirUnreadable {
        dir: dir.display().to_string(),
        cause: e.to_string(),
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| ProbeError::BackendDirUnreadable {
            dir: dir.display().to_string(),
            cause: e.to_string(),
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|e| ProbeError::BackendFileUnreadable {
            file: path.display().to_string(),
            cause: e.to_string(),
        })?;
        if file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().into_string().map_err(|raw| {
            ProbeError::BackendFileNameNotUtf8 {
                dir: dir.display().to_string(),
                name: raw.to_string_lossy().into_owned(),
            }
        })?;
        let lower = name.to_ascii_lowercase();
        let is_tool = lower.ends_with(".exe");
        let is_lib = lower.ends_with(".dll");
        if !is_tool && !is_lib {
            continue;
        }
        if lower.starts_with("ggml-") && is_lib {
            backends.push(name.clone());
        }
        // The size only orders the work (largest first, so the longest file starts at once); a
        // file whose size cannot be read still gets hashed, and fails there if it is unreadable.
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        files.push((name, path, size));
    }

    // Name order first, so the reported error (below) is deterministic; the work queue is a
    // separate largest-first ordering of indices into this list.
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut queue: Vec<usize> = (0..files.len()).collect();
    queue.sort_by_key(|&i| std::cmp::Reverse(files.get(i).map_or(0, |f| f.2)));

    let results = hash_files_in_parallel(&files, &queue);

    let mut hashes = BTreeMap::new();
    for ((name, _, _), result) in files.into_iter().zip(results) {
        match result {
            Some(Ok(hash)) => {
                hashes.insert(name, hash);
            }
            Some(Err(e)) => return Err(e),
            // Not attempted because another file already failed; that failure is reported when
            // this loop reaches it.
            None => {}
        }
    }
    backends.sort();
    Ok((hashes, backends))
}

/// Upper bound on hashing threads: past this, one backend directory's files are not numerous or
/// large enough to gain more, and the probe should not saturate a many-core machine.
pub const MAX_HASH_THREADS: usize = 8;

/// Read size per `ReadFile` while hashing. 1 MiB cut a serial pass over the official build from
/// 1.12 s to 0.85 s against `io::copy`'s 8 KiB (measured 2026-09-24).
pub const HASH_BUFFER_BYTES: usize = 1 << 20;

/// Hash `files[i].1` for every `i` in `queue`, in that order of starting, on up to
/// [`MAX_HASH_THREADS`] threads. The calling thread is one of the workers, so a failure to start
/// any extra thread only makes this slower, never fail. Once any file fails, no new file is
/// started. Returns one slot per entry of `files`: `None` if it was never attempted.
fn hash_files_in_parallel(
    files: &[(String, PathBuf, u64)],
    queue: &[usize],
) -> Vec<Option<Result<String, ProbeError>>> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    let slots: Vec<Mutex<Option<Result<String, ProbeError>>>> =
        files.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);

    let work = || {
        let mut buf = vec![0u8; HASH_BUFFER_BYTES];
        while !failed.load(Ordering::Relaxed) {
            let Some(&i) = queue.get(next.fetch_add(1, Ordering::Relaxed)) else {
                break;
            };
            let (Some((_, path, _)), Some(slot)) = (files.get(i), slots.get(i)) else {
                continue;
            };
            let result = hash_file(path, &mut buf);
            if result.is_err() {
                failed.store(true, Ordering::Relaxed);
            }
            // A poisoned slot can only come from a panic in this closure, which cannot happen
            // while holding it; recover the value either way.
            *slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
        }
    };

    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(MAX_HASH_THREADS)
        .min(queue.len())
        .max(1);
    std::thread::scope(|scope| {
        for _ in 1..threads {
            // A thread that cannot be created is skipped: the calling thread below drains the
            // queue regardless, so this degrades to fewer workers, never to a panic.
            let _ = std::thread::Builder::new()
                .name("cerebro-probe-hash".to_string())
                .spawn_scoped(scope, work);
        }
        work();
    });

    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        })
        .collect()
}

/// Stream one file through SHA-256 using `buf`, returning lowercase hex.
fn hash_file(path: &Path, buf: &mut [u8]) -> Result<String, ProbeError> {
    let unreadable = |e: std::io::Error| ProbeError::BackendFileUnreadable {
        file: path.display().to_string(),
        cause: e.to_string(),
    };
    let mut file = std::fs::File::open(path).map_err(unreadable)?;
    let mut hasher = Sha256::new();
    loop {
        let n = match file.read(buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(unreadable(e)),
        };
        hasher.update(buf.get(..n).unwrap_or(buf));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real captured output from the development build.
    const VERSION: &str = "version: 0.3.0-dev (build 10751, commit 3466812d1)\n\
                           built with Clang 20.1.8 for Windows x86_64\n";

    const DEVICES: &str = "Available devices:\n  \
        CUDA0: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8191 MiB, 7113 MiB free)\n";

    #[test]
    fn parses_real_version_output() {
        let (build, commit, compiler) = parse_version(VERSION);
        assert_eq!(build, Some(10751));
        assert_eq!(commit.as_deref(), Some("3466812d1"));
        assert_eq!(compiler.as_deref(), Some("Clang 20.1.8 for Windows x86_64"));
    }

    #[test]
    fn unrecognized_version_text_yields_none_rather_than_failing() {
        // Upstream changes this text freely; refusing to run against a newer llama.cpp would be
        // worse than recording weaker provenance.
        let (build, commit, compiler) = parse_version("something entirely different\n");
        assert!(build.is_none() && commit.is_none() && compiler.is_none());
    }

    #[test]
    fn parses_real_device_listing() {
        let devs = parse_list_devices(DEVICES).unwrap();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].id(), "CUDA0");
        assert_eq!(devs[0].total_mib(), Some(8191));
        assert_eq!(devs[0].free_mib(), Some(7113));
        assert!(devs[0].description().contains("3070 Ti"));
    }

    #[test]
    fn log_lines_containing_colons_do_not_become_devices() {
        // The backend prints loader chatter on the same stream. A line like
        // "load_backend: loaded CUDA backend from ..." must never become a device id that Cerebro
        // would later emit into -dev, which would fail at spawn time with a confusing error.
        let noisy = "Available devices:\n  \
            load_backend: loaded CUDA backend from ggml-cuda.dll\n  \
            CUDA0: NVIDIA GeForce RTX 3070 Ti Laptop GPU (8191 MiB, 7113 MiB free)\n";
        let devs = parse_list_devices(noisy).unwrap();
        assert_eq!(devs.len(), 1, "got {devs:?}");
        assert_eq!(devs[0].id(), "CUDA0");
    }

    #[test]
    fn text_before_the_header_is_ignored() {
        let devs = parse_list_devices("warning: blah\nCUDA9: fake (1 MiB, 1 MiB free)\n").unwrap();
        assert!(devs.is_empty());
    }

    #[test]
    fn device_ids_are_canonicalized_to_uppercase() {
        let devs = parse_list_devices("Available devices:\n  cuda0: thing (1 MiB, 1 MiB free)\n").unwrap();
        assert_eq!(devs[0].id(), "CUDA0");
    }

    #[test]
    fn hash_directory_hashes_every_exe_and_dll_and_streams_rather_than_buffering_fully() {
        let dir = std::env::temp_dir().join(format!(
            "cerebro-probe-hash-dir-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir(&dir).expect("create fixture dir");
        std::fs::write(dir.join("llama-server.exe"), b"fake server").expect("write exe");
        std::fs::write(dir.join("ggml-cuda.dll"), b"fake cuda backend").expect("write dll");
        std::fs::write(dir.join("readme.txt"), b"not a backend file").expect("write txt");
        std::fs::create_dir(dir.join("subdir")).expect("write subdir");

        let (hashes, backends) = hash_directory(&dir).expect("hash a clean directory");
        assert_eq!(hashes.len(), 2, "{hashes:?}");
        assert!(hashes.contains_key("llama-server.exe"));
        assert!(hashes.contains_key("ggml-cuda.dll"));
        assert_eq!(backends, vec!["ggml-cuda.dll".to_string()]);
        // Every hash is lowercase hex SHA-256 (64 hex digits).
        for hash in hashes.values() {
            assert_eq!(hash.len(), 64, "{hash}");
            assert!(hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parallel_hashing_matches_known_digests_across_buffer_boundaries() {
        // Values, not only shape: a known vector, and files larger than one read buffer (so a
        // chunking bug that dropped or repeated a partial read would change the digest), hashed
        // concurrently with several small files.
        let dir = std::env::temp_dir().join(format!(
            "cerebro-probe-hash-values-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir(&dir).expect("create fixture dir");
        std::fs::write(dir.join("abc.dll"), b"abc").expect("write abc");
        let big: Vec<u8> = (0..(HASH_BUFFER_BYTES * 5 / 2))
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        std::fs::write(dir.join("big.exe"), &big).expect("write big");
        let exact: Vec<u8> = vec![0x5a; HASH_BUFFER_BYTES];
        std::fs::write(dir.join("exact.dll"), &exact).expect("write exact");
        for i in 0..12 {
            std::fs::write(dir.join(format!("small{i}.dll")), format!("file {i}"))
                .expect("write small");
        }

        let (hashes, _) = hash_directory(&dir).expect("hash the fixture");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(hashes.len(), 15, "{hashes:?}");
        assert_eq!(
            hashes["abc.dll"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(hashes["big.exe"], format!("{:x}", Sha256::digest(&big)));
        assert_eq!(hashes["exact.dll"], format!("{:x}", Sha256::digest(&exact)));
        for i in 0..12 {
            let expected = format!("{:x}", Sha256::digest(format!("file {i}").as_bytes()));
            assert_eq!(hashes[&format!("small{i}.dll")], expected);
        }
    }

    #[cfg(windows)]
    #[test]
    fn hash_directory_errors_on_an_unreadable_entry_rather_than_skipping_it() {
        // A planted-DLL check that silently drops the one file it could not read is exactly the gap
        // the check exists to close, so an exclusively-locked file must fail the whole call rather
        // than being treated as absent.
        use std::os::windows::fs::OpenOptionsExt;
        let dir = std::env::temp_dir().join(format!(
            "cerebro-probe-hash-locked-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir(&dir).expect("create fixture dir");
        let locked = dir.join("ggml-cuda.dll");
        std::fs::write(&locked, b"fake cuda backend").expect("write dll");

        // share_mode(0): no other handle, including this same process's own `hash_directory`, may
        // open the file for read or write while this one is held.
        let _handle = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&locked)
            .expect("open exclusively");

        let err = hash_directory(&dir).unwrap_err();
        assert!(
            matches!(&err, ProbeError::BackendFileUnreadable { file, .. } if file.contains("ggml-cuda.dll")),
            "{err:?}"
        );

        drop(_handle);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_dacl_is_not_safe_to_spawn() {
        // The whole reason this state exists: "could not check" must never read as "checked, fine".
        assert!(!DaclAudit::Unknown("x".into()).is_safe_to_spawn());
        assert!(!DaclAudit::Redirected("x".into()).is_safe_to_spawn());
        assert!(DaclAudit::Redirected("x".into()).principals().is_empty());
        assert!(DaclAudit::OwnerOnly.is_safe_to_spawn());
        let grant = WritableGrant {
            path: PathBuf::from("C:/x"),
            applies_to: AuditedObject::Directory,
            principal: "SOMEONE".into(),
            sid: "S-1-5-32-545".into(),
            access_mask: 0x2,
            rights: vec!["FILE_ADD_FILE"],
            via: GrantVia::Ace,
        };
        let audit = DaclAudit::WritableByOthers(vec![grant.clone(), grant]);
        assert!(!audit.is_safe_to_spawn());
        // Two grants to one principal (directory + a file, say) must read as one principal.
        assert_eq!(audit.principals(), vec!["SOMEONE"]);
    }

    #[test]
    fn into_unsafe_maps_every_variant() {
        // `into_unsafe` is the only way to build an `UnsafeDacl`, and the type has no arm that can
        // produce one from `OwnerOnly` — this pins down what each `DaclAudit` variant becomes rather
        // than merely that `OwnerOnly` becomes `Ok`.
        assert_eq!(DaclAudit::OwnerOnly.into_unsafe(), Ok(()));

        let grant = WritableGrant {
            path: PathBuf::from("C:/x"),
            applies_to: AuditedObject::Directory,
            principal: "SOMEONE".into(),
            sid: "S-1-5-32-545".into(),
            access_mask: 0x2,
            rights: vec!["FILE_ADD_FILE"],
            via: GrantVia::Ace,
        };
        assert_eq!(
            DaclAudit::WritableByOthers(vec![grant.clone()]).into_unsafe(),
            Err(UnsafeDacl::WritableByOthers(vec![grant]))
        );
        assert_eq!(
            DaclAudit::Unknown("x".into()).into_unsafe(),
            Err(UnsafeDacl::Unknown("x".into()))
        );
        assert_eq!(
            DaclAudit::Redirected("y".into()).into_unsafe(),
            Err(UnsafeDacl::Redirected("y".into()))
        );
    }

    #[test]
    fn unsafe_dacl_display_is_bounded() {
        // The development build alone has well over a hundred grants (two ACEs repeated across
        // every file); the rendered text must stay a short summary, not one line per grant.
        let grants: Vec<WritableGrant> = (0..20)
            .map(|i| WritableGrant {
                path: PathBuf::from(format!("C:/x/file{i}.dll")),
                applies_to: AuditedObject::File,
                principal: format!("PRINCIPAL{i}"),
                sid: format!("S-1-5-21-1-2-3-{i}"),
                access_mask: 0x2,
                rights: vec!["FILE_WRITE_DATA"],
                via: GrantVia::Ace,
            })
            .collect();
        let text = UnsafeDacl::WritableByOthers(grants).to_string();
        assert!(text.contains("20 write-capable grants"), "{text}");
        assert!(text.contains("and 15 more"), "{text}");
        assert!(text.len() < 2000, "{} chars: {text}", text.len());
    }

    #[test]
    fn tool_name_may_not_escape_the_backend_directory() {
        // A crafted config must not be able to point the probe at a binary elsewhere, a batch file
        // (run through cmd.exe), or an alternate data stream. Driven by the shared table.
        let dir = std::env::temp_dir();
        for bad in REJECTED_TOOL_NAMES {
            assert!(!is_bare_exe_name(bad), "{bad:?} should have been rejected");
            let e = probe(&dir, bad).unwrap_err();
            assert!(
                matches!(e, ProbeError::BackendToolEscapesDir { .. }),
                "{bad:?}: {e:?}"
            );
            // Refused whatever the folder-audit setting: the name check is not the audit.
            let skip = ProbeSettings {
                folder_audit: FolderAudit::SkippedByUserSetting,
                ..ProbeSettings::default()
            };
            let e = probe_with(&dir, bad, skip).unwrap_err();
            assert!(
                matches!(e, ProbeError::BackendToolEscapesDir { .. }),
                "{bad:?}: {e:?}"
            );
        }
        assert!(is_bare_exe_name("llama-server.exe"));
        assert!(is_bare_exe_name("LLAMA-BENCH.EXE"));
    }

    #[test]
    fn probe_command_runs_with_cwd_pinned_to_the_audited_dir() {
        // Regression guard for the CWD gap: ggml's backend loader also scans the process's current
        // directory, so every backend spawn must start from the audited directory, not wherever
        // Cerebro's own process happened to be started from.
        let dir = Path::new(r"C:\backend");
        let cmd = probe_command(
            &dir.join("llama-server.exe"),
            dir,
            &["--version"],
            EnvironmentAllowlist::Enforce,
            None,
        )
        .expect("build the query command");
        assert_eq!(cmd.get_current_dir(), Some(dir));
        assert_eq!(cmd.get_program(), dir.join("llama-server.exe").as_os_str());
    }

    #[cfg(windows)]
    #[test]
    fn probe_command_clears_the_environment_only_while_the_allowlist_is_enforced() {
        // The query children load every ggml DLL, so the allowlist must reach them: on, the command
        // carries exactly the closed block (PATH = dir;System32, no inherited name); off, it still
        // pins PATH first but no longer drops what the parent had.
        let dir = Path::new(r"C:\backend");
        let tool = dir.join("llama-server.exe");
        let system32 = crate::child_env::system_directory().expect("resolve System32");
        let expected_path = {
            let mut p = dir.as_os_str().to_owned();
            p.push(";");
            p.push(system32.as_os_str());
            p
        };
        let path_of = |cmd: &Command| {
            cmd.get_envs()
                .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
                .and_then(|(_, v)| v.map(std::ffi::OsStr::to_os_string))
        };

        let on = probe_command(&tool, dir, &["--version"], EnvironmentAllowlist::Enforce, None)
            .expect("build");
        assert_eq!(path_of(&on), Some(expected_path.clone()));
        let names: Vec<String> = on
            .get_envs()
            .map(|(k, _)| k.to_string_lossy().into_owned())
            .collect();
        for name in &names {
            assert!(
                [
                    "NUMBER_OF_PROCESSORS",
                    "PROCESSOR_ARCHITECTURE",
                    "PATH",
                    "SystemRoot",
                    "windir",
                    "SystemDrive"
                ]
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(name)),
                "{name} is not in the closed block: {names:?}"
            );
        }

        let off = probe_command(
            &tool,
            dir,
            &["--version"],
            EnvironmentAllowlist::SkippedByUserSetting,
            None,
        )
        .expect("build");
        let off_path = path_of(&off)
            .expect("PATH is always pinned")
            .to_string_lossy()
            .into_owned();
        assert!(
            off_path.starts_with(&*expected_path.to_string_lossy()),
            "{off_path}"
        );
        // This test process always has more than the closed block's handful of names (USERPROFILE,
        // TEMP, ...), and with the allowlist off they pass through.
        assert!(
            off.get_envs().count() > on.get_envs().count(),
            "inherited names must pass through when off"
        );
    }

    #[test]
    fn drain_bounded_keeps_at_most_the_cap_and_still_reaches_eof() {
        // A pathological tool printing more than the cap must neither grow memory without bound nor
        // block: the reader keeps the cap and drains the rest.
        let big = vec![b'x'; (MAX_QUERY_OUTPUT as usize) * 2 + 17];
        let rx = drain_bounded(std::io::Cursor::new(big)).expect("start the reader thread");
        let kept = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the reader must finish");
        assert_eq!(kept.len() as u64, MAX_QUERY_OUTPUT);
    }

    #[cfg(windows)]
    #[test]
    fn a_query_that_outlives_its_deadline_is_killed_and_reported() {
        // `ping -n 60` runs for about a minute (loopback only), standing in for a driver hang in
        // `--list-devices`. The handle opened here, while the child is alive, pins its identity: the
        // final check cannot be fooled by a reused PID.
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{
            OpenProcess, WaitForSingleObject, PROCESS_QUERY_LIMITED_INFORMATION,
            PROCESS_SYNCHRONIZE,
        };
        let system32 = crate::child_env::system_directory().expect("resolve System32");
        let tool = system32.join("PING.EXE");
        let child = probe_command(
            &tool,
            &system32,
            &["-n", "60", "127.0.0.1"],
            EnvironmentAllowlist::Enforce,
            None,
        )
        .expect("build")
        .spawn()
        .expect("spawn ping");
        // SAFETY: no pointers; the handle is checked for null and closed exactly once below.
        let handle = unsafe {
            OpenProcess(
                PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                child.id(),
            )
        };
        assert!(!handle.is_null(), "open the running child");

        let started = Instant::now();
        let err = collect_bounded(child, &tool, Duration::from_millis(300)).unwrap_err();
        let elapsed = started.elapsed();
        // SAFETY: `handle` is live and was opened with SYNCHRONIZE; a zero timeout only polls.
        let gone = unsafe { WaitForSingleObject(handle, 0) } == WAIT_OBJECT_0;
        // SAFETY: opened above, closed only here.
        unsafe { CloseHandle(handle) };

        assert!(matches!(err, ProbeError::ToolTimedOut { .. }), "{err:?}");
        assert!(
            elapsed < Duration::from_secs(10),
            "the deadline must bound the wait: {elapsed:?}"
        );
        assert!(
            gone,
            "the timed-out child must be killed and reaped, never left running"
        );
    }

    #[test]
    #[ignore = "requires the development machine's llama.cpp b10751 build, whose directory ACL grants \
                Modify to ASUS\\CodexSandboxUsers"]
    fn live_b10751_directory_is_writable_by_codex_sandbox_users() {
        let dir = Path::new("C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751");
        let audit = audit_dacl(dir);
        let DaclAudit::WritableByOthers(grants) = &audit else {
            panic!("expected WritableByOthers, got {audit:?}");
        };
        assert!(!audit.is_safe_to_spawn());
        assert!(
            audit.principals().iter().any(|p| p.ends_with("\\CodexSandboxUsers")),
            "principals: {:?}",
            audit.principals()
        );
        // The directory grant is what allows planting a new DLL...
        assert!(grants.iter().any(|g| g.path() == dir
            && g.applies_to() == AuditedObject::Directory
            && g.principal().ends_with("\\CodexSandboxUsers")
            && g.rights().contains(&"FILE_ADD_FILE")));
        // ...and the per-file grant is what allows overwriting an existing backend in place.
        assert!(grants.iter().any(|g| g.path() == dir.join("ggml-cuda.dll")
            && g.applies_to() == AuditedObject::File
            && g.principal().ends_with("\\CodexSandboxUsers")
            && g.rights().contains(&"FILE_WRITE_DATA")));
    }

    #[test]
    #[ignore = "requires the development machine's llama.cpp b10751 build, whose directory ACL grants \
                Modify to ASUS\\CodexSandboxUsers"]
    fn live_probe_of_the_development_backend_refuses_to_spawn() {
        // This fixture is the real-world case the gate exists for. `probe()` must refuse before it
        // ever runs `--list-devices`, which would otherwise make ggml load every DLL in the
        // directory — planted or not — before anyone had a chance to say no.
        let dir = Path::new("C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751");
        let err = probe(dir, "llama-server.exe").unwrap_err();
        let ProbeError::BackendDirUnsafe { dir: reported, audit } = &err else {
            panic!("expected BackendDirUnsafe, got {err:?}");
        };
        // `probe()` resolves `dir` to its canonical path before auditing (see `probe`'s own docs),
        // so it must report that resolved, backslash-form directory — not the as-given, forward-slash
        // spelling this test happens to pass in — proving probe() reports the directory it actually
        // audited.
        assert_eq!(
            reported, r"C:\Users\noahm\Projects\qwen-flash-next\llama.cpp-b10751",
            "probe must report the canonical, resolved directory it audited"
        );
        let UnsafeDacl::WritableByOthers(grants) = audit else {
            panic!("expected WritableByOthers, got {audit:?}");
        };
        assert!(
            grants.iter().any(|g| g.principal().ends_with("\\CodexSandboxUsers")),
            "{grants:?}"
        );
    }

    #[test]
    #[ignore = "requires the development machine's llama.cpp b10751 build; spawns the real binary \
                directly (bypassing the DACL gate on purpose, as a controlled test) to keep \
                hardware-identity coverage now that probe() itself refuses to spawn from this \
                deliberately-unsafe fixture"]
    fn live_identity_extraction_against_the_real_binary() {
        let dir = Path::new("C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751");
        let tool = dir.join("llama-server.exe");

        let version_raw = run(&tool, dir, &["--version"], EnvironmentAllowlist::Enforce, None).unwrap();
        let (build_number, _commit, _compiler) = parse_version(&version_raw);
        assert_eq!(build_number, Some(10751));

        let devices_raw = run(
            &tool,
            dir,
            &["--list-devices"],
            EnvironmentAllowlist::Enforce,
            None,
        )
        .unwrap();
        let addressable = parse_list_devices(&devices_raw).unwrap();
        assert_eq!(addressable.len(), 1);
        assert_eq!(addressable[0].id(), "CUDA0");

        // The finding that invalidated the iGPU plan: no Vulkan backend in this build.
        let (_hashes, ggml_backends) = hash_directory(dir).unwrap();
        let lower: Vec<String> = ggml_backends.iter().map(|b| b.to_ascii_lowercase()).collect();
        assert!(!lower.iter().any(|b| b.contains("vulkan")));
        assert!(lower.iter().any(|b| b.contains("cuda")));
    }

    /// Everything below needs real Win32 security descriptors — SDDL-driven or on disk — and does
    /// not compile on a non-Windows target at all: `dacl` is a Windows-only module (see the `mod
    /// dacl` declaration above), and `TempTree`'s `Drop` calls into it directly.
    #[cfg(windows)]
    mod windows_tests {
        use super::*;

        #[test]
        fn unc_path_is_unknown() {
            // Both spellings are rejected purely by path prefix, before any descriptor read touches
            // even loopback — so this test needs no actual share access and cannot hang or time out.
            for path in [r"\\127.0.0.1\C$\Windows", r"\\?\UNC\127.0.0.1\C$\Windows"] {
                let DaclAudit::Unknown(msg) = audit_dacl(Path::new(path)) else {
                    panic!("a network path must never audit as anything but Unknown: {path}");
                };
                assert!(msg.contains("network") || msg.contains("disk path"), "{msg}");
            }
        }

        #[test]
        fn a_non_local_path_is_a_distinct_permanent_refusal_and_a_missing_one_is_not() {
            // A caller with a restart loop needs to tell "this volume can never be audited" (retrying
            // is pointless) from "this path could not be read right now" (it may appear later).
            // Prefix-only rejection, so no share is ever contacted.
            for path in [r"\\127.0.0.1\C$\Windows", r"\\?\UNC\127.0.0.1\C$\Windows"] {
                let err = canonical_backend_dir(Path::new(path)).unwrap_err();
                assert!(
                    matches!(err, ProbeError::BackendDirNotLocalVolume { .. }),
                    "{path}: {err:?}"
                );
            }
            let missing = TempTree::new("not-local-missing").0.join("does-not-exist");
            let err = canonical_backend_dir(&missing).unwrap_err();
            assert!(
                !matches!(err, ProbeError::BackendDirNotLocalVolume { .. }),
                "{err:?}"
            );
        }

        #[test]
        fn dotdot_path_reports_the_normalized_directory() {
            // `audit` calls `std::path::absolute` exactly once and uses that single value for every
            // Win32 call, the enumeration, and every reported grant path — so auditing
            // `<tree>\sub\..` must report exactly what auditing `<tree>` itself reports, never the
            // literal `..` spelling. Every assertion here runs unconditionally: a prior version of
            // this test only checked the path-content assertion inside an `if let WritableByOthers`,
            // which meant it silently asserted nothing at all whenever the audit came back clean.
            let tree = TempTree::new("dacl-normalize");
            std::fs::create_dir(tree.0.join("sub")).expect("create sub");
            let dotdot = tree.0.join("sub").join("..");

            let direct = audit_dacl(&tree.0);
            let via_dotdot = audit_dacl(&dotdot);
            assert_eq!(direct, via_dotdot, "auditing through a `..` component must normalize first");

            let grants = match &via_dotdot {
                DaclAudit::OwnerOnly => Vec::new(),
                DaclAudit::WritableByOthers(grants) => grants.clone(),
                DaclAudit::Unknown(msg) | DaclAudit::Redirected(msg) => {
                    panic!("a clean temp tree must be auditable: {msg}")
                }
            };
            assert!(
                grants.iter().all(|g| !g.path().display().to_string().contains("..")),
                "{grants:?}"
            );
        }

        #[test]
        fn object_audit_judges_one_file_or_directory_by_its_own_acl() {
            // The single-object audit must accept an ACL naming only the current user and SYSTEM,
            // and report anyone else holding a write-capable right on that object alone. It must
            // not walk ancestors: %TEMP% on the development machine is Modify-writable by a sandbox
            // group, and a clean object inside it must still audit clean.
            let me = current_user_sid().expect("process token must be queryable");
            let tree = TempTree::new("object-audit");
            let file = tree.0.join("secret.dpapi");
            std::fs::write(&file, b"blob").expect("create temp file");
            let sub = tree.0.join("dir");
            std::fs::create_dir(&sub).expect("create temp dir");

            let owner_only = format!("D:P(A;;FA;;;{me})(A;;FA;;;SY)");
            dacl::set_dacl_from_sddl(&file, &owner_only).expect("apply owner-only ACL");
            dacl::set_dacl_from_sddl(&sub, &owner_only).expect("apply owner-only ACL");
            assert_eq!(audit_object_dacl(&file), DaclAudit::OwnerOnly);
            assert_eq!(audit_object_dacl(&sub), DaclAudit::OwnerOnly);

            dacl::set_dacl_from_sddl(&file, &format!("D:P(A;;FW;;;BU)(A;;FA;;;{me})"))
                .expect("apply test ACL");
            let DaclAudit::WritableByOthers(grants) = audit_object_dacl(&file) else {
                panic!("BUILTIN\\Users write on the file must be reported");
            };
            assert!(
                grants.iter().all(|g| g.path() == file
                    && g.applies_to() == AuditedObject::File
                    && g.sid() == "S-1-5-32-545"),
                "{grants:?}"
            );

            // FILE_ADD_FILE is write-capable only when the object is a directory: the object's own
            // kind picks the mask, exactly as in the directory audit.
            dacl::set_dacl_from_sddl(&sub, &format!("D:P(A;;0x2;;;BU)(A;;FA;;;{me})"))
                .expect("apply test ACL");
            let DaclAudit::WritableByOthers(grants) = audit_object_dacl(&sub) else {
                panic!("BUILTIN\\Users FILE_ADD_FILE on the directory must be reported");
            };
            assert!(
                grants.iter().any(|g| g.applies_to() == AuditedObject::Directory
                    && g.rights().contains(&"FILE_ADD_FILE")),
                "{grants:?}"
            );

            // OWNER RIGHTS (the SDDL `OW` alias) is not one of the trusted SIDs.
            dacl::set_dacl_from_sddl(&file, "D:P(A;;FA;;;OW)(A;;FA;;;SY)").expect("apply OW ACL");
            assert!(matches!(audit_object_dacl(&file), DaclAudit::WritableByOthers(_)));
            dacl::set_dacl_from_sddl(&file, &owner_only).expect("restore owner-only ACL");

            let missing = tree.0.join("missing.dpapi");
            assert!(matches!(audit_object_dacl(&missing), DaclAudit::Unknown(_)));
        }

        #[test]
        fn object_audit_refuses_a_path_that_is_itself_a_reparse_point() {
            // Needs no privilege: a junction, unlike a symlink.
            let tree = TempTree::new("object-audit-junction");
            let target = tree.0.join("target");
            std::fs::create_dir(&target).expect("create target");
            let link = tree.0.join("link");
            dacl::create_junction(&link, &target).expect("create junction");
            assert!(
                matches!(audit_object_dacl(&link), DaclAudit::Redirected(_)),
                "{:?}",
                audit_object_dacl(&link)
            );
        }

        #[test]
        fn file_ace_test_passes_without_elevation() {
            // The brief's core requirement: a file's own ACE is not implied by its directory's, and a
            // regression that skipped per-file enumeration (or that matched everything to the current
            // user so nothing was ever reported) would still pass a test that only looked at the
            // directory. The b10751 live fixture cannot isolate this on its own — its per-file grants
            // are inherited from the directory — so this has to be a synthetic, non-ignored case.
            //
            // Every SDDL below includes the current user's own SID: a baseline that grants only
            // `SYSTEM`/`Administrators` would leave a non-elevated test runner with no access to its
            // own temp directory the moment it was applied.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let tree = TempTree::new("file-ace");
            let file = tree.0.join("ggml-cuda.dll");
            std::fs::write(&file, b"not a real dll").expect("create temp file");

            let baseline = format!("D:(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;{me})");
            dacl::set_dacl_from_sddl(&tree.0, &baseline).expect("apply baseline ACL");
            dacl::set_dacl_from_sddl(&file, &baseline).expect("apply baseline ACL");

            // Not `assert_eq!(before, OwnerOnly)`: once path pinning is in effect, %TEMP% on the
            // development machine is itself Modify-writable by `ASUS\CodexSandboxUsers` (the same
            // fact the b10751 fixture demonstrates), so an `Ancestor` grant naming %TEMP% is expected
            // here and must not fail this test. What this test is actually about is the
            // file-vs-directory distinction, so it asserts on that alone: neither `tree.0` nor `file`
            // grants anything.
            let before_grants = match audit_dacl(&tree.0) {
                DaclAudit::OwnerOnly => Vec::new(),
                DaclAudit::WritableByOthers(grants) => grants,
                DaclAudit::Unknown(msg) | DaclAudit::Redirected(msg) => {
                    panic!("baseline ACL must be auditable: {msg}")
                }
            };
            assert!(
                !before_grants.iter().any(|g| g.path() == tree.0 || g.path() == file),
                "a dir+file pinned to SYSTEM/Administrators/current-user-only must not itself be \
                 flagged: {before_grants:?}"
            );

            // BUILTIN\Users granted FILE_GENERIC_WRITE on the file alone; the directory's own DACL is
            // never touched again after the baseline above. The current user keeps FA so the guard can
            // still clean up regardless of what this test asserts.
            dacl::set_dacl_from_sddl(&file, &format!("D:(A;;FW;;;BU)(A;;FA;;;{me})"))
                .expect("apply test ACL");

            let after = audit_dacl(&tree.0);
            let DaclAudit::WritableByOthers(grants) = &after else {
                panic!("expected WritableByOthers once one file grants BU write, got {after:?}");
            };
            assert!(
                grants.iter().any(|g| g.path() == file
                    && g.applies_to() == AuditedObject::File
                    && g.sid() == "S-1-5-32-545"
                    && g.rights().contains(&"FILE_WRITE_DATA")),
                "{grants:?}"
            );
            assert!(
                !grants.iter().any(|g| g.path() == tree.0 && g.applies_to() == AuditedObject::Directory),
                "the directory's own DACL was never touched, so it must not appear: {grants:?}"
            );
        }

        #[test]
        fn create_junction_with_an_overlong_target_is_an_error_and_leaves_nothing_behind() {
            // `\??\` + `C:\` + 32,760 units is 32,767 UTF-16 units, 65,534 bytes: it fits the
            // `u16` byte count, but the 12 bytes `ReparseDataLength` adds do not. That addition used
            // to be unchecked (a panic in a debug build); it must be a typed error, and a failure
            // must not leave the link's directory behind.
            let tree = TempTree::new("junction-overlong");
            let link = tree.0.join("link");
            let target = PathBuf::from(format!(r"C:\{}", "a".repeat(32_760)));

            let err =
                dacl::create_junction(&link, &target).expect_err("an overlong target must fail");
            assert!(err.to_string().contains("too long"), "{err}");
            assert!(
                std::fs::symlink_metadata(&link).is_err(),
                "a failed create_junction must leave no directory behind"
            );
        }

        #[test]
        fn a_junction_inside_the_directory_makes_the_audit_redirected() {
            // A junction needs no privilege and no Developer Mode to create, unlike a symlink (see
            // `a_reparse_point_in_the_directory_makes_the_audit_redirected_when_symlinks_are_creatable`,
            // below), which is exactly why it is the reparse type an unprivileged attacker can
            // actually plant inside a directory the audit is otherwise clean on — so this test, unlike
            // that one, must never skip.
            let tree = TempTree::new("dacl-inner-junction");
            let elsewhere = TempTree::new("dacl-inner-junction-target");
            let link = tree.0.join("ggml-cuda.dll");
            dacl::create_junction(&link, &elsewhere.0).expect("junctions need no privilege");

            // `Redirected`, not `Unknown`: a planted link is a fact on disk, not a failed check, so a
            // restart loop stops at once. Either way it is untrusted.
            let DaclAudit::Redirected(msg) = audit_dacl(&tree.0) else {
                panic!("a reparse point inside the directory must fail closed as Redirected");
            };
            assert!(msg.contains("reparse point"), "{msg}");
            assert!(msg.contains("ggml-cuda.dll"), "{msg}");
        }

        #[test]
        fn a_reparse_point_in_the_directory_makes_the_audit_redirected_when_symlinks_are_creatable()
        {
            let tree = TempTree::new("dacl-reparse");
            let target = tree.0.join("real.dll");
            std::fs::write(&target, b"not a real dll").expect("create target file");
            let link = tree.0.join("ggml-cuda.dll");
            // Developer Mode (or an elevated token) is required to create a symlink; skip rather than
            // fail if this machine cannot, since that is an environment property, not a code property
            // — but say so out loud, so a skip is visible in test output rather than silently looking
            // like a pass for the wrong reason. `a_junction_inside_the_directory_makes_the_audit_redirected`,
            // above, is the unconditional coverage for this same code path.
            if std::os::windows::fs::symlink_file(&target, &link).is_err() {
                eprintln!("SKIPPED: cannot create file symlinks on this token");
                return;
            }

            let DaclAudit::Redirected(msg) = audit_dacl(&tree.0) else {
                panic!("a reparse point must fail the audit closed rather than being silently resolved");
            };
            assert!(msg.contains("reparse point"), "{msg}");
            assert!(msg.contains("ggml-cuda.dll"), "{msg}");
        }

        #[test]
        fn a_backend_dir_that_is_itself_a_junction_is_resolved_and_its_target_audited() {
            // `dir`'s own final component is the junction (scoop's `current` layout). One policy for
            // every reparse point: it resolves to the target, and auditing through it reports
            // exactly what auditing the target directly reports.
            let tree = TempTree::new("dacl-junction");
            let target = tree.0.join("real");
            std::fs::create_dir(&target).expect("create junction target");
            let link = tree.0.join("link");
            dacl::create_junction(&link, &target).expect("junctions need no privilege to create");

            let canonical = canonical_backend_dir(&target).expect("resolve the target");
            assert_eq!(
                canonical_backend_dir(&link).expect("resolve via the junction"),
                canonical
            );
            let via = audit_dacl(&link);
            if let DaclAudit::Unknown(msg) = &via {
                assert!(
                    !msg.contains("reparse point"),
                    "the final junction was not followed: {msg}"
                );
            }
            assert_eq!(audit_dacl(&target), via);
        }

        #[test]
        fn a_final_junction_target_with_an_untrusted_write_grant_is_reported_against_the_target() {
            // Following the final junction must not skip the audit: the planted BU grant on the
            // target is reported against the canonical target path.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let real = TempTree::new("final-jn-grant-target");
            let backend = real.0.join("backend");
            std::fs::create_dir(&backend).expect("create nested backend dir");
            dacl::set_dacl_from_sddl(&backend, &format!("D:(A;;FW;;;BU)(A;;FA;;;{me})"))
                .expect("apply test ACL");
            let holder = TempTree::new("final-jn-grant-holder");
            let link = holder
                .0
                .join(format!("cerebro-jn-f-{}", std::process::id()));
            dacl::create_junction(&link, &backend).expect("junctions need no privilege to create");

            let canonical = canonical_backend_dir(&backend).expect("resolve the real path");
            let audit = audit_dacl(&link);
            let DaclAudit::WritableByOthers(grants) = &audit else {
                panic!("a BU write grant on the final junction's target must be reported, got {audit:?}");
            };
            assert!(
                grants.iter().any(|g| g.path() == canonical
                    && g.applies_to() == AuditedObject::Directory
                    && g.sid() == "S-1-5-32-545"),
                "the grant must be reported against the canonical target: {grants:?}"
            );
            assert!(
                grants.iter().all(|g| !g.path().starts_with(&link)),
                "no grant may name the junction's own spelling: {grants:?}"
            );
        }

        #[test]
        fn a_junction_planted_at_the_resolved_dirs_own_name_fails_the_audit_closed() {
            // The swap-after-resolution race on the final component: resolve, rename the real
            // backend directory away, plant a junction at its name. The re-check opens the final
            // component itself, so it sees a reparse point rather than resolving through it.
            let root = TempTree::new("final-swap-after-resolve");
            let backend = root.0.join("backend");
            std::fs::create_dir(&backend).expect("create backend");
            let resolved = canonical_backend_dir(&backend).expect("resolve before the swap");

            let elsewhere = TempTree::new("final-swap-target");
            std::fs::rename(&backend, root.0.join("backend-old"))
                .expect("rename the real dir away");
            dacl::create_junction(&backend, &elsewhere.0)
                .expect("junctions need no privilege to create");

            let audit = audit_dacl_resolved(&resolved);
            let DaclAudit::Unknown(msg) = &audit else {
                panic!(
                    "a final component swapped after resolution must fail closed, got {audit:?}"
                );
            };
            assert!(msg.contains("changed"), "{msg}");
            // Unlink before `root` drops, so its DACL-restoring cleanup cannot reach `elsewhere`.
            std::fs::remove_dir(&backend).expect("remove the junction itself");
        }

        #[test]
        fn unaudited_resolution_failure_is_a_resolution_error_not_an_audit_verdict() {
            // With the folder audit off nothing is audited, so a path that cannot be resolved must
            // say so, never read as "could not audit" or as an unsafe-DACL verdict.
            let missing = TempTree::new("unaudited-missing").0.join("does-not-exist");
            let err = canonical_backend_dir_unaudited(&missing).unwrap_err();
            assert!(
                matches!(&err, ProbeError::BackendDirUnresolvable { dir, .. } if dir == &missing.display().to_string()),
                "{err:?}"
            );
        }

        #[test]
        fn unaudited_resolution_follows_an_ancestor_junction() {
            let real = TempTree::new("unaudited-junction-target");
            let backend = real.0.join("backend");
            std::fs::create_dir(&backend).expect("create nested backend dir");
            let holder = TempTree::new("unaudited-junction-holder");
            let link = holder
                .0
                .join(format!("cerebro-jn-u-{}", std::process::id()));
            dacl::create_junction(&link, &real.0).expect("junctions need no privilege to create");

            let direct = canonical_backend_dir_unaudited(&backend).expect("resolve the real path");
            let via = canonical_backend_dir_unaudited(&link.join("backend"))
                .expect("resolve via junction");
            assert_eq!(
                direct, via,
                "both spellings must resolve to the one canonical directory"
            );
        }

        #[test]
        fn an_ancestor_junction_is_resolved_and_the_canonical_path_is_re_audited() {
            // The open finding docs/RESTART.md recorded: a junction *above* the backend directory,
            // not the directory itself (that case is
            // `a_backend_dir_that_is_itself_a_junction_is_resolved_and_its_target_audited`).
            // `canonicalize` resolves an ancestor junction away — transparently, the way every Win32
            // path-taking call does — before `check_path_chain` ever runs, so the audit ends up
            // examining the *real* ancestor chain, never the junction's own spelling.
            //
            // Proven the same way `dotdot_path_reports_the_normalized_directory` proves
            // normalization: auditing through the junction must report exactly what auditing the
            // real directory directly reports, because by the time any Win32 call runs, both
            // spellings name the same resolved value. This machine's own %TEMP% ancestors are
            // themselves writable by an untrusted principal (see `file_ace_test_passes_without_elevation`),
            // so this test cannot assume the result is `OwnerOnly` — only that both paths agree, and
            // that neither ever reports a grant against the junction's own spelling.
            let real = TempTree::new("dacl-ancestor-junction-target");
            let backend = real.0.join("backend");
            std::fs::create_dir(&backend).expect("create nested backend dir");

            let holder = TempTree::new("dacl-ancestor-junction-holder");
            // A unique, unlikely-to-collide name: unlike a bare `"link"`, this cannot false-fail (or
            // false-pass) merely because %TEMP% or the account's own username happens to contain the
            // substring being checked for below.
            let link_name = format!(
                "cerebro-jn-{}-{}",
                std::process::id(),
                real.0.display().to_string().len()
            );
            let link = holder.0.join(&link_name);
            dacl::create_junction(&link, &real.0).expect("junctions need no privilege to create");

            let direct = audit_dacl(&backend);
            let via_junction = audit_dacl(&link.join("backend"));
            // Guards against the vacuous case: two audits that both fail `Unknown` for an unrelated
            // reason would still satisfy a bare `==` below without the ancestor junction ever having
            // been resolved at all.
            if let DaclAudit::Unknown(msg) = &via_junction {
                assert!(
                    !msg.contains("reparse point"),
                    "ancestor junction was not resolved: {msg}"
                );
            }
            assert_eq!(
                direct, via_junction,
                "an ancestor junction must be resolved before auditing, so both paths examine the \
                 same real ancestor chain"
            );
            if let DaclAudit::WritableByOthers(grants) = &via_junction {
                assert!(
                    grants
                        .iter()
                        .all(|g| !g.path().display().to_string().contains(&link_name)),
                    "an ancestor junction's own spelling must never appear in a reported grant \
                     path: {grants:?}"
                );
            }
        }

        #[test]
        fn ancestor_junction_re_audit_sees_the_targets_own_grant() {
            // Proves the canonical target is what gets audited, not merely that two spellings agree:
            // a known grant is planted on the real backend directory, and the audit through the
            // ancestor junction must report it against the canonical path. An audit that skipped the
            // target's own DACL, or returned the same `Unknown` text for both spellings, fails here.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let real = TempTree::new("jn-grant-target");
            let backend = real.0.join("backend");
            std::fs::create_dir(&backend).expect("create nested backend dir");
            dacl::set_dacl_from_sddl(&backend, &format!("D:(A;;FW;;;BU)(A;;FA;;;{me})"))
                .expect("apply test ACL");
            let holder = TempTree::new("jn-grant-holder");
            let link = holder
                .0
                .join(format!("cerebro-jn-g-{}", std::process::id()));
            dacl::create_junction(&link, &real.0).expect("junctions need no privilege to create");

            let canonical = canonical_backend_dir(&backend).expect("resolve the real path");
            let audit = audit_dacl(&link.join("backend"));
            let DaclAudit::WritableByOthers(grants) = &audit else {
                panic!("a BU write grant on the canonical target must be reported, got {audit:?}");
            };
            assert!(
                grants.iter().any(|g| g.path() == canonical
                    && g.applies_to() == AuditedObject::Directory
                    && g.sid() == "S-1-5-32-545"),
                "the grant must be reported against the canonical directory: {grants:?}"
            );
        }

        #[test]
        fn a_junction_planted_after_resolution_fails_the_audit_closed() {
            // The swap-after-resolution race, made deterministic by doing the swap between the two
            // calls: resolve, rename a real component away, plant a junction at its name pointing at
            // a different (clean) directory that also has a `backend` child. Re-resolving inside the
            // audit would retarget it to the swap target and pass; the resolved-path audit must
            // instead notice the path no longer resolves to itself and fail closed.
            let root = TempTree::new("swap-after-resolve");
            let parent = root.0.join("p");
            std::fs::create_dir_all(parent.join("backend")).expect("create p/backend");
            let resolved =
                canonical_backend_dir(&parent.join("backend")).expect("resolve before the swap");

            // Its own TempTree, so `root`'s Drop following the junction can only reach a tree this
            // test owns.
            let elsewhere = TempTree::new("swap-target");
            std::fs::create_dir(elsewhere.0.join("backend")).expect("create swap target backend");
            std::fs::rename(&parent, root.0.join("p-old")).expect("rename the real component away");
            dacl::create_junction(&parent, &elsewhere.0)
                .expect("junctions need no privilege to create");

            let audit = audit_dacl_resolved(&resolved);
            let DaclAudit::Unknown(msg) = &audit else {
                panic!("a path swapped after resolution must fail closed, got {audit:?}");
            };
            assert!(
                msg.contains("changed") || msg.contains("reparse point"),
                "{msg}"
            );
        }

        #[test]
        fn audit_of_an_unchanged_resolved_path_matches_the_resolving_audit() {
            // No regression for the normal case: with nothing moved, auditing the resolved path
            // without re-resolving must give exactly the answer the resolving audit gives.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let tree = TempTree::new("resolved-unchanged");
            dacl::set_dacl_from_sddl(&tree.0, &format!("D:(A;;FW;;;BU)(A;;FA;;;{me})"))
                .expect("apply test ACL");
            let resolved = canonical_backend_dir(&tree.0).expect("resolve");
            let via_resolved = audit_dacl_resolved(&resolved);
            assert!(
                matches!(via_resolved, DaclAudit::WritableByOthers(_)),
                "{via_resolved:?}"
            );
            assert_eq!(audit_dacl(&tree.0), via_resolved);
        }

        #[test]
        fn resolved_audit_refuses_a_spelling_that_is_not_already_canonical() {
            // The input must be exactly what `canonical_backend_dir` returned: a `..` spelling of the
            // same directory is not, and must fail closed rather than be silently re-resolved.
            let tree = TempTree::new("resolved-noncanonical");
            std::fs::create_dir(tree.0.join("sub")).expect("create sub");
            let dotdot = canonical_backend_dir(&tree.0)
                .expect("resolve")
                .join("sub")
                .join("..");
            let audit = audit_dacl_resolved(&dotdot);
            let DaclAudit::Unknown(msg) = &audit else {
                panic!("a non-canonical spelling must fail closed, got {audit:?}");
            };
            assert!(msg.contains("changed"), "{msg}");
        }

        #[test]
        fn probe_resolves_an_ancestor_junction_before_hashing_or_spawning() {
            // The other half of the fix: `probe()` must hash, audit, and spawn from the same
            // canonical directory the DACL audit resolves to, never the as-given spelling that still
            // names the junction. This machine's real ancestor chain may or may not pass the audit
            // (see the test above), so this cannot assert `probe()` gets all the way to a successful
            // spawn — only that whichever of the two failure shapes it hits names the canonical,
            // resolved directory and never the junction's own spelling.
            let real = TempTree::new("probe-ancestor-junction-target");
            let backend = real.0.join("backend");
            std::fs::create_dir(&backend).expect("create nested backend dir");
            std::fs::write(backend.join("tool.exe"), b"not a real exe").expect("create fake tool");

            let holder = TempTree::new("probe-ancestor-junction-holder");
            // Unique, so this cannot false-fail (or false-pass) merely because %TEMP% or the
            // account's own username happens to contain the substring checked for below.
            let link_name = format!(
                "cerebro-jn-{}-{}",
                std::process::id(),
                real.0.display().to_string().len()
            );
            let link = holder.0.join(&link_name);
            dacl::create_junction(&link, &real.0).expect("junctions need no privilege to create");

            // "Not elevated" injected: this test is about which directory is named, and the suite
            // also runs from an elevated shell, where the default elevation refusal would stop the
            // probe before either failure shape below (proven separately by the elevation tests).
            let not_elevated = || Ok(false);
            let err = probe_with_token(
                &link.join("backend"),
                "tool.exe",
                ProbeSettings::default(),
                &not_elevated,
                None,
            )
            .unwrap_err();
            let named = match &err {
                ProbeError::BackendDirUnsafe { dir, .. } => dir.clone(),
                ProbeError::ToolSpawn { tool, .. } => tool.clone(),
                other => panic!(
                    "expected either an unsafe-DACL refusal or a spawn failure on the fake tool: \
                     {other:?}"
                ),
            };
            assert!(
                !named.contains(&link_name),
                "must not report the junction's own spelling: {named}"
            );
            let tag = real
                .0
                .file_name()
                .expect("temp dir has a name")
                .to_string_lossy()
                .to_string();
            assert!(
                named.contains(&tag),
                "must report the canonical, resolved directory: {named}"
            );
        }

        /// A junction to a directory this test does not own, removed on drop as a *link only*.
        ///
        /// Deliberately not a [`TempTree`]: `TempTree`'s `Drop` rewrites the DACL of every entry it
        /// finds, and `SetNamedSecurityInfoW` follows a junction, so a junction to the real fixture
        /// inside a `TempTree` would overwrite that fixture's own ACL on cleanup. `remove_dir` on a
        /// junction removes the reparse point itself and never touches its target.
        struct ForeignJunction {
            holder: PathBuf,
            link: PathBuf,
        }

        impl ForeignJunction {
            fn new(tag: &str, target: &Path) -> Self {
                static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // The timestamp keeps a holder left by a crashed earlier run whose pid was reused
                // from colliding with this one, as in `TempTree::new`.
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                let holder = std::env::temp_dir()
                    .join(format!("cerebro-{tag}-{}-{nanos}-{n}", std::process::id()));
                std::fs::create_dir(&holder).expect("create the junction holder");
                let link = holder.join(format!("cerebro-jn-live-{}-{n}", std::process::id()));
                dacl::create_junction(&link, target)
                    .expect("junctions need no privilege to create");
                Self { holder, link }
            }
        }

        impl Drop for ForeignJunction {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir(&self.link);
                let _ = std::fs::remove_dir(&self.holder);
            }
        }

        #[test]
        #[ignore = "requires an official llama.cpp build (CEREBRO_LLAMACPP_OFFICIAL, default \
                    C:/bench/llamacpp-official); spawns it with --version and --list-devices (no \
                    model is loaded)"]
        fn live_probe_through_an_ancestor_junction_hashes_and_reports_the_canonical_dir() {
            // The success-path half of `probe_resolves_an_ancestor_junction_before_hashing_or_spawning`,
            // which can only check the path in an error: here the real fixture is probed all the way
            // through, via a junction standing in for its parent directory. The folder audit is
            // skipped (a supported user setting) because this fixture grants Modify to
            // Authenticated Users by design of this machine, so `FolderAudit::Enforce` would refuse
            // before hashing.
            let real = crate::fixtures::official_backend_dir().unwrap_or_else(|e| panic!("{e}"));
            let real = real.as_path();
            let parent = real.parent().expect("the fixture dir has a parent");
            let name = real.file_name().expect("the fixture dir has a name");
            let junction = ForeignJunction::new("probe-live-junction", parent);
            let via_junction = junction.link.join(name);

            let canonical = canonical_backend_dir_unaudited(real).expect("resolve the fixture dir");
            // Elevation refusal is off too (a supported setting, recorded below): the live tests on
            // this machine run from an elevated shell as well as a normal one, and the elevation
            // gate itself is covered deterministically by the injected-token tests.
            let settings = ProbeSettings {
                folder_audit: FolderAudit::SkippedByUserSetting,
                elevation_refusal: ElevationRefusal::SkippedByUserSetting,
                ..ProbeSettings::default()
            };
            let probed = probe_with(&via_junction, "llama-server.exe", settings)
                .expect("probe the fixture through the junction");

            assert_eq!(
                probed.identity.dir(),
                canonical.as_path(),
                "the identity must name the resolved directory, never the junction spelling"
            );
            let (expected_hashes, _) = hash_directory(&canonical).expect("hash the canonical dir");
            assert_eq!(
                probed.identity.hashes(),
                &expected_hashes,
                "hashes must come from the canonical dir"
            );
            assert!(
                probed.identity.hashes().contains_key("llama-server.exe"),
                "{:?}",
                probed.identity.hashes()
            );
            assert_eq!(probed.folder_audit, FolderAudit::SkippedByUserSetting);
            assert_eq!(probed.environment_allowlist, EnvironmentAllowlist::Enforce);
            assert_eq!(
                probed.elevation_refusal,
                ElevationRefusal::SkippedByUserSetting
            );
            // The closed environment (no TEMP, no USERPROFILE) must still let the CUDA build list
            // its devices: the allowlist costs nothing in capability.
            assert!(
                probed
                    .addressable
                    .iter()
                    .any(|d| d.id().starts_with("CUDA")),
                "{:?}",
                probed.addressable
            );
        }

        #[test]
        fn backend_dir_that_is_a_symlink_is_followed_like_a_junction_when_creatable() {
            // A directory symlink needs Developer Mode or elevation to create, unlike a junction
            // (covered, unconditionally, above) — so this one may legitimately skip on a token that has
            // neither, and says so out loud rather than looking like a pass for the wrong reason.
            // Same policy as a junction: resolved to the target, and the target audited.
            let target = TempTree::new("dacl-symlink-target");
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            dacl::set_dacl_from_sddl(&target.0, &format!("D:(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;{me})"))
                .expect("apply baseline ACL");

            let holder = TempTree::new("dacl-symlink-holder");
            let link = holder.0.join("link");
            if std::os::windows::fs::symlink_dir(&target.0, &link).is_err() {
                eprintln!("SKIPPED: cannot create directory symlinks on this token");
                return;
            }

            assert_eq!(
                canonical_backend_dir(&link).expect("resolve via the symlink"),
                canonical_backend_dir(&target.0).expect("resolve the target")
            );
            let via = audit_dacl(&link);
            if let DaclAudit::Unknown(msg) = &via {
                assert!(
                    !msg.contains("reparse point"),
                    "the symlink was not followed: {msg}"
                );
            }
            assert_eq!(audit_dacl(&target.0), via);
            std::fs::remove_dir(&link).expect("remove the symlink itself");
        }

        #[test]
        fn audit_of_a_missing_directory_is_unknown_and_carries_the_win32_code() {
            let missing = std::env::temp_dir().join("cerebro-dacl-audit-does-not-exist-7f3a");
            let DaclAudit::Unknown(message) = audit_dacl(&missing) else {
                panic!("a path that does not exist must not audit as anything but Unknown");
            };
            // ERROR_FILE_NOT_FOUND. The code is what makes the message actionable.
            assert!(message.contains("Win32 error 2"), "{message}");
            let needle = "cerebro-dacl-audit-does-not-exist-7f3a";
            assert!(message.contains(needle), "{message}");
            assert_eq!(
                message.matches(needle).count(),
                1,
                "the failing path must be named once, not doubled by an outer wrapper: {message}"
            );
        }

        /// A uniquely named temp directory whose `Drop` restores write/delete access for the current
        /// user and removes the whole tree, even when the test panicked or left an unreadable DACL
        /// behind.
        ///
        /// The name is unique per test *and* per run (PID, a nanosecond timestamp, and a per-process
        /// counter) so a reused PID across runs can never collide with a leftover directory. `Drop`
        /// always tries to hand access back to the current user, on every child first and then the
        /// directory itself, before removing anything — a locked-down file would otherwise block
        /// `remove_dir_all` even after the directory's own DACL is fixed.
        struct TempTree(PathBuf);

        impl TempTree {
            fn new(tag: &str) -> Self {
                static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("cerebro-{tag}-{}-{nanos}-{n}", std::process::id()));
                std::fs::create_dir(&path)
                    .expect("a name unique per test, per run, and per process must not already exist");
                TempTree(path)
            }
        }

        impl Drop for TempTree {
            fn drop(&mut self) {
                // Best-effort by necessity: `Drop` cannot propagate failure, and the entire point of
                // this guard is to leave nothing behind even when the DACL under test denied this
                // account everything.
                let Ok(me) = dacl::test_current_user_sid() else {
                    return;
                };
                // Object- and container-inheritable: a protected, non-inheritable ACE on a parent
                // strips the inherited ACEs from everything below it, which left grandchildren with
                // an empty DACL that `remove_dir_all` could not delete (one leaked tree per run).
                let restore = format!("D:(A;OICI;FA;;;{me})");
                restore_access(&self.0, &restore);
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        /// Hand `sddl` to `path` and, for a directory, to everything below it, parent first (the
        /// parent must be listable before its children can be found).
        ///
        /// Never re-ACLs or descends through a reparse point: `SetNamedSecurityInfoW` follows a
        /// junction, so doing either would rewrite the DACL of a target outside this tree (the
        /// swap tests plant junctions to other `TempTree`s). `remove_dir_all` then removes the link
        /// itself, never its target.
        fn restore_access(path: &Path, sddl: &str) {
            use std::os::windows::fs::MetadataExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
            let Ok(meta) = std::fs::symlink_metadata(path) else {
                return;
            };
            if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                return;
            }
            let _ = dacl::set_dacl_from_sddl(path, sddl);
            if meta.is_dir() {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        restore_access(&entry.path(), sddl);
                    }
                }
            }
        }

        #[test]
        fn temp_tree_removes_a_nested_tree_whose_grandchild_has_only_inherited_aces() {
            // The exact leak: `<root>\p-old\backend` inherits everything from `<root>`. Locking the
            // root makes the old one-level restore strip those inherited ACEs, which left `backend`
            // with an empty DACL and the tree behind. The recursive, inheritable restore must not.
            let tree = TempTree::new("nested-inherited");
            let path = tree.0.clone();
            std::fs::create_dir_all(path.join("p-old").join("backend"))
                .expect("create nested dirs");
            std::fs::write(path.join("p-old").join("backend").join("f.dll"), b"x")
                .expect("write file");
            dacl::set_dacl_from_sddl(&path, "D:").expect("lock the root");
            drop(tree);
            assert!(
                !path.exists(),
                "the whole nested tree must be removed: {path:?}"
            );
        }

        #[test]
        fn temp_tree_removes_a_tree_whose_grandchild_is_locked_to_nobody() {
            let tree = TempTree::new("locked-grandchild");
            let path = tree.0.clone();
            let grandchild = path.join("a").join("b");
            std::fs::create_dir_all(&grandchild).expect("create nested dirs");
            std::fs::write(grandchild.join("f.dll"), b"x").expect("write file");
            dacl::set_dacl_from_sddl(&grandchild, "D:").expect("lock the grandchild");
            drop(tree);
            assert!(
                !path.exists(),
                "a locked grandchild must not keep the tree alive: {path:?}"
            );
        }

        #[test]
        fn temp_tree_cleanup_never_re_acls_a_junction_target() {
            // A junction inside the tree points at a directory carrying a known BU write grant. If
            // cleanup followed the junction, it would replace that DACL with its own restore ACE and
            // the grant would vanish; it must survive, and only the link may be removed.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let target = TempTree::new("junction-cleanup-target");
            dacl::set_dacl_from_sddl(&target.0, &format!("D:(A;;FW;;;BU)(A;;FA;;;{me})"))
                .expect("apply the marker ACL");
            let tree = TempTree::new("junction-cleanup-holder");
            let path = tree.0.clone();
            dacl::create_junction(&path.join("link"), &target.0)
                .expect("junctions need no privilege");
            drop(tree);

            assert!(!path.exists(), "the holder must be removed: {path:?}");
            assert!(
                target.0.is_dir(),
                "removing the link must never remove its target"
            );
            let audit = audit_dacl(&target.0);
            let DaclAudit::WritableByOthers(grants) = &audit else {
                panic!("the marker grant must still be there, got {audit:?}");
            };
            assert!(
                grants
                    .iter()
                    .any(|g| g.path() == target.0.as_path() && g.sid() == "S-1-5-32-545"),
                "the junction target's DACL was rewritten: {grants:?}"
            );
        }

        #[test]
        fn temp_tree_is_removed_after_panic() {
            // Regression coverage for `TempTree` itself: the leftover directories that motivated it
            // were left behind by a *failed assertion*, i.e. a panic — a plain scope exit was never
            // the failure mode in question, so the guard has to be proven against an actual unwind.
            let tree = TempTree::new("panic-guard");
            let path = tree.0.clone();
            // Lock the directory down to nobody — not even the current user — to prove `Drop` restores
            // access before it tries to remove anything.
            dacl::set_dacl_from_sddl(&path, "D:").expect("apply locked-down ACL");

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _tree = tree; // Moved in so it drops (via unwind) when this closure panics.
                panic!("intentional panic to exercise TempTree's Drop guard");
            }));
            assert!(result.is_err(), "the inner closure must have panicked");
            assert!(!path.exists(), "TempTree must remove its directory even after a panic: {path:?}");
        }

        #[test]
        fn probe_refuses_to_spawn_when_the_directory_is_writable_by_others() {
            // A regression test for the fail-open gap: probe() must consult audit_dacl() *before*
            // touching the tool at all. The tool file below is not a valid executable, so if probe()
            // ever tried to spawn it, the failure would surface as `ProbeError::ToolSpawn` (CreateProcess
            // rejecting a non-PE file) rather than the `BackendDirUnsafe` this test asserts — that
            // difference in error variant is what proves the spawn attempt never happened.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let tree = TempTree::new("probe-unsafe-dir");
            std::fs::write(tree.0.join("tool.exe"), b"not a real executable").expect("create fake tool");
            // Everyone full control, plus the current user so the TempTree guard can still clean up.
            dacl::set_dacl_from_sddl(&tree.0, &format!("D:(A;;FA;;;WD)(A;;FA;;;{me})"))
                .expect("apply test ACL");

            let err = probe(&tree.0, "tool.exe").unwrap_err();
            assert!(matches!(err, ProbeError::BackendDirUnsafe { .. }), "{err:?}");
        }

        #[test]
        fn probe_error_carries_structured_audit() {
            // The structured `UnsafeDacl` must survive the `probe()` -> `ProbeError` boundary rather
            // than being flattened into a string a caller would have to re-parse to recover principals,
            // paths or rights.
            let me = dacl::test_current_user_sid().expect("process token must be queryable");
            let tree = TempTree::new("probe-structured-audit");
            std::fs::write(tree.0.join("tool.exe"), b"not a real executable").expect("create fake tool");
            dacl::set_dacl_from_sddl(&tree.0, &format!("D:(A;;FA;;;WD)(A;;FA;;;{me})"))
                .expect("apply test ACL");

            let err = probe(&tree.0, "tool.exe").unwrap_err();
            let ProbeError::BackendDirUnsafe { dir, audit } = &err else {
                panic!("expected BackendDirUnsafe, got {err:?}");
            };
            assert_eq!(dir, &tree.0.display().to_string());
            let UnsafeDacl::WritableByOthers(grants) = audit else {
                panic!("expected WritableByOthers, got {audit:?}");
            };
            assert!(grants.iter().any(|g| g.sid() == "S-1-1-0"), "{grants:?}");
        }

        /// A fake backend a query would fail to start (not a PE file), in a directory whose
        /// folder audit is skipped by setting (`%TEMP%`'s ancestors are writable by other
        /// principals on this machine, which is orthogonal to the elevation gate under test).
        /// Reaching the query shows up as `ToolSpawn`, so the error variant tells whether the
        /// elevation gate stopped the probe before anything was spawned.
        fn elevation_fixture(tag: &str) -> (TempTree, ProbeSettings) {
            let tree = TempTree::new(tag);
            std::fs::write(tree.0.join("tool.exe"), b"not a real executable")
                .expect("create fake tool");
            let settings = ProbeSettings {
                folder_audit: FolderAudit::SkippedByUserSetting,
                ..ProbeSettings::default()
            };
            (tree, settings)
        }

        #[test]
        fn an_elevated_probe_is_refused_before_any_query_spawns() {
            let (tree, settings) = elevation_fixture("probe-elevated-on");
            assert_eq!(
                settings.elevation_refusal,
                ElevationRefusal::Enforce,
                "on by default"
            );
            let asked = std::cell::Cell::new(0u32);
            let elevated = || {
                asked.set(asked.get() + 1);
                Ok(true)
            };
            let err = probe_with_token(&tree.0, "tool.exe", settings, &elevated, None).unwrap_err();
            assert_eq!(
                err,
                ProbeError::RunningElevated,
                "not ToolSpawn: nothing may be spawned"
            );
            assert_eq!(asked.get(), 1);
        }

        #[test]
        fn an_elevation_check_that_cannot_complete_refuses_the_probe() {
            let (tree, settings) = elevation_fixture("probe-elevated-unknown");
            let failing = || {
                Err(ProbeError::Win32 {
                    call: "OpenProcessToken",
                    code: 5,
                })
            };
            let err = probe_with_token(&tree.0, "tool.exe", settings, &failing, None).unwrap_err();
            assert!(
                matches!(&err, ProbeError::ElevationCheckFailed { cause } if cause.contains("OpenProcessToken")),
                "fail closed, never assume not elevated: {err:?}"
            );
        }

        #[test]
        fn a_non_elevated_probe_proceeds_to_the_query() {
            let (tree, settings) = elevation_fixture("probe-elevated-no");
            let err = probe_with_token(&tree.0, "tool.exe", settings, &|| Ok(false), None).unwrap_err();
            assert!(
                matches!(err, ProbeError::ToolSpawn { .. }),
                "the query was attempted: {err:?}"
            );
        }

        #[test]
        fn turning_elevation_refusal_off_skips_the_token_query_and_proceeds() {
            let (tree, mut settings) = elevation_fixture("probe-elevated-off");
            settings.elevation_refusal = ElevationRefusal::SkippedByUserSetting;
            let asked = std::cell::Cell::new(false);
            let elevated = || {
                asked.set(true);
                Ok(true)
            };
            let err = probe_with_token(&tree.0, "tool.exe", settings, &elevated, None).unwrap_err();
            assert!(
                matches!(err, ProbeError::ToolSpawn { .. }),
                "off must reach the query even when elevated: {err:?}"
            );
            assert!(
                !asked.get(),
                "off skips the token query itself, never merely ignores it"
            );
        }

        #[test]
        fn an_unusable_cache_dir_is_refused_before_any_query_spawns() {
            // Dropping the cache instead would make every query recompile with no signal.
            let (tree, settings) = elevation_fixture("probe-cache-dir");
            for bad in [PathBuf::from("relative-cache"), tree.0.join("missing")] {
                let err = probe_with_token(&tree.0, "tool.exe", settings, &|| Ok(false), Some(&bad))
                    .unwrap_err();
                assert!(
                    matches!(err, ProbeError::CacheDirUnusable { .. }),
                    "not ToolSpawn: nothing may be spawned ({}): {err:?}",
                    bad.display()
                );
            }
            let err = probe_with_token(&tree.0, "tool.exe", settings, &|| Ok(false), Some(&tree.0))
                .unwrap_err();
            assert!(
                matches!(err, ProbeError::ToolSpawn { .. }),
                "an existing absolute cache dir proceeds to the query: {err:?}"
            );
        }
    }
}
