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
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
/// Deliberately a three-state value. `Unknown` exists so that "we could not check" can never be
/// mistaken for "we checked and it is fine" — a security control that silently degrades to
/// permissive is worse than one that is absent, because it is trusted.
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
    /// Empty for `OwnerOnly` and for `Unknown` — an empty list says nothing about safety on its
    /// own; use [`DaclAudit::is_safe_to_spawn`] for that.
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
            UnsafeDacl::Unknown(message) => write!(f, "{message}"),
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

/// Everything known about one backend build.
///
/// Carries no `DaclAudit` field: `probe` (below) returns [`ProbeError::BackendDirUnsafe`] instead
/// of constructing a `BackendProbe` at all whenever the audit is not [`DaclAudit::OwnerOnly`], so a
/// value of this type existing would always carry that one constant answer — a field that can only
/// ever hold one value tells a caller nothing and invites code that matches on it as though it
/// might not.
#[derive(Debug, Clone)]
pub struct BackendProbe {
    /// Build identity and file hashes.
    pub identity: BackendIdentity,
    /// Devices this binary can actually target.
    pub addressable: Vec<AddressableDevice>,
}

/// Probe a llama.cpp build directory.
///
/// `tool` is the executable used for the `--version` and `--list-devices` queries; it must live
/// inside `dir`, which is checked rather than assumed so a caller cannot be talked into running a
/// binary from somewhere else by a crafted config.
///
/// [`audit_dacl`] runs before either query and gates them: if it does not report
/// [`DaclAudit::OwnerOnly`], this returns [`ProbeError::BackendDirUnsafe`] without spawning
/// anything, because `--list-devices` alone is enough to make ggml load every backend DLL an
/// untrusted principal left in `dir`. Both queries are then run with the child's working directory
/// pinned to `dir` (see [`probe_command`]) — the same audited directory, and nowhere else ggml's
/// loader might also search.
///
/// # Platform
///
/// [`audit_dacl`] cannot establish safety off Windows, so on every other platform this always
/// returns [`ProbeError::BackendDirUnsafe`] with [`UnsafeDacl::Unknown`] and never spawns anything.
/// That is deliberate fail-closed behavior, not a missing port: this crate has no code path that
/// spawns from a backend directory without going through this gate first.
pub fn probe(dir: &Path, tool_name: &str) -> Result<BackendProbe, ProbeError> {
    if !dir.is_dir() {
        return Err(ProbeError::BackendDirMissing {
            dir: dir.display().to_string(),
        });
    }
    if Path::new(tool_name).components().count() != 1 {
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

    // The audit must run, and pass, *before* anything is spawned from `dir`: ggml loads every
    // backend DLL it finds there as soon as the child process starts, so a directory an untrusted
    // principal can write to must never reach `run()` at all.
    if let Err(audit) = audit_dacl(dir).into_unsafe() {
        return Err(ProbeError::BackendDirUnsafe {
            dir: dir.display().to_string(),
            audit,
        });
    }

    let version_raw = run(&tool, dir, &["--version"])?;
    let (build_number, commit, compiler) = parse_version(&version_raw);
    let devices_raw = run(&tool, dir, &["--list-devices"])?;
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
/// name *this machine's* principals only when this machine issued the descriptor being read. A
/// `subst` or `DefineDosDevice` alias is resolved to its real target before anything else runs, so
/// the ancestor walk below examines the real path, not the alias's own. Any reparse point — `dir`
/// itself, an ancestor on its path, or an entry found while enumerating `dir` — fails the whole
/// result closed rather than being resolved and silently missing what actually controls its target.
/// So does any other Win32 failure, unrecognized ACE type, or malformed ACE or SID.
///
/// # Caller obligation: pin the working directory
///
/// This audits `dir` and its ancestors — never a process's current working directory. ggml's
/// backend loader scans *both* the executable directory and the current directory for
/// `ggml-<name>-*.dll` and loads every match it finds. Spawning from `dir` must therefore also set
/// the child's working directory to `dir` (see [`probe_command`]), or a DLL sitting wherever Cerebro
/// happened to start from can load unaudited.
pub fn audit_dacl(dir: &Path) -> DaclAudit {
    #[cfg(windows)]
    {
        match dacl::audit(dir) {
            Ok(grants) if grants.is_empty() => DaclAudit::OwnerOnly,
            Ok(grants) => DaclAudit::WritableByOthers(grants),
            Err(error) => {
                DaclAudit::Unknown(format!("DACL audit could not complete: {error}; treat as untrusted"))
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        DaclAudit::Unknown("DACL audit is implemented for Windows only; treat as untrusted".into())
    }
}

/// Build a query command for a backend tool, with its working directory pinned to `dir`.
///
/// ggml's backend loader (`ggml_backend_load_best`) scans both the directory the executable lives
/// in *and the process's current directory* for `ggml-<name>-*.dll`, loading every match. Only `dir`
/// is what [`audit_dacl`] examined, so a spawn that left the child's working directory at whatever
/// Cerebro's own happened to be could still load a DLL this audit never saw — e.g. Cerebro started
/// from `%USERPROFILE%\Downloads`, where an untrusted principal dropped one. Pinning the working
/// directory here, in the one place every backend spawn in this module goes through, keeps the audit
/// and the spawn from drifting apart.
///
/// `tool` must already be the absolute path to the executable (`dir.join(tool_name)`, as `probe`
/// builds it): `Command::current_dir` changes where the child starts, not how Windows resolves a
/// relative program name, so passing a bare file name here and relying on `current_dir` to find it
/// would not do what it looks like it does.
fn probe_command(tool: &Path, dir: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new(tool);
    cmd.args(args).current_dir(dir).stdin(Stdio::null());
    cmd
}

fn run(tool: &Path, dir: &Path, args: &[&str]) -> Result<String, ProbeError> {
    let output = probe_command(tool, dir, args)
        .output()
        .map_err(|e| ProbeError::ToolSpawn {
            tool: tool.display().to_string(),
            cause: e.to_string(),
        })?;
    // llama.cpp writes both --version and --list-devices to stderr, so merge the streams rather
    // than reading stdout and concluding the tool printed nothing.
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(text)
}

/// Extract build number, commit, and compiler from `--version` output.
///
/// Returns `None` per field rather than failing: upstream changes this text freely, and a probe
/// that hard-errors on an unrecognized version line would refuse to run against a newer llama.cpp
/// that is otherwise perfectly usable. The consequence is recorded honestly — a measurement keyed
/// on a `None` build number is marked as having weaker provenance rather than being silently
/// treated as comparable.
pub fn parse_version(text: &str) -> (Option<u32>, Option<String>, Option<String>) {
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
/// Each file's bytes are streamed through the hasher (`io::copy`) rather than read fully into
/// memory first, since this runs before every spawn and a backend directory's CUDA runtime DLLs can
/// be hundreds of megabytes each.
pub fn hash_directory(dir: &Path) -> Result<(BTreeMap<String, String>, Vec<String>), ProbeError> {
    let mut hashes = BTreeMap::new();
    let mut backends = Vec::new();

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
        let mut file = std::fs::File::open(&path).map_err(|e| ProbeError::BackendFileUnreadable {
            file: path.display().to_string(),
            cause: e.to_string(),
        })?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher).map_err(|e| ProbeError::BackendFileUnreadable {
            file: path.display().to_string(),
            cause: e.to_string(),
        })?;
        hashes.insert(name, format!("{:x}", hasher.finalize()));
    }

    backends.sort();
    Ok((hashes, backends))
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
        // A crafted config must not be able to point the probe at a binary elsewhere.
        let dir = std::env::temp_dir();
        let e = probe(&dir, "../evil.exe").unwrap_err();
        assert!(matches!(e, ProbeError::BackendToolEscapesDir { .. }));
        let e = probe(&dir, "sub/dir/tool.exe").unwrap_err();
        assert!(matches!(e, ProbeError::BackendToolEscapesDir { .. }));
    }

    #[test]
    fn probe_command_runs_with_cwd_pinned_to_the_audited_dir() {
        // Regression guard for the CWD gap: ggml's backend loader also scans the process's current
        // directory, so every backend spawn must start from the audited directory, not wherever
        // Cerebro's own process happened to be started from.
        let dir = Path::new(r"C:\backend");
        let cmd = probe_command(&dir.join("llama-server.exe"), dir, &["--version"]);
        assert_eq!(cmd.get_current_dir(), Some(dir));
        assert_eq!(cmd.get_program(), dir.join("llama-server.exe").as_os_str());
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
        assert_eq!(reported, &dir.display().to_string());
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

        let version_raw = run(&tool, dir, &["--version"]).unwrap();
        let (build_number, _commit, _compiler) = parse_version(&version_raw);
        assert_eq!(build_number, Some(10751));

        let devices_raw = run(&tool, dir, &["--list-devices"]).unwrap();
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
                DaclAudit::Unknown(msg) => panic!("a clean temp tree must be auditable: {msg}"),
            };
            assert!(
                grants.iter().all(|g| !g.path().display().to_string().contains("..")),
                "{grants:?}"
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
                DaclAudit::Unknown(msg) => panic!("baseline ACL must be auditable: {msg}"),
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
        fn a_junction_inside_the_directory_makes_the_audit_unknown() {
            // A junction needs no privilege and no Developer Mode to create, unlike a symlink (see
            // `a_reparse_point_in_the_directory_makes_the_audit_unknown_when_symlinks_are_creatable`,
            // below), which is exactly why it is the reparse type an unprivileged attacker can
            // actually plant inside a directory the audit is otherwise clean on — so this test, unlike
            // that one, must never skip.
            let tree = TempTree::new("dacl-inner-junction");
            let elsewhere = TempTree::new("dacl-inner-junction-target");
            let link = tree.0.join("ggml-cuda.dll");
            dacl::create_junction(&link, &elsewhere.0).expect("junctions need no privilege");

            let DaclAudit::Unknown(msg) = audit_dacl(&tree.0) else {
                panic!("a reparse point inside the directory must fail closed");
            };
            assert!(msg.contains("reparse point"), "{msg}");
            assert!(msg.contains("ggml-cuda.dll"), "{msg}");
        }

        #[test]
        fn a_reparse_point_in_the_directory_makes_the_audit_unknown_when_symlinks_are_creatable() {
            let tree = TempTree::new("dacl-reparse");
            let target = tree.0.join("real.dll");
            std::fs::write(&target, b"not a real dll").expect("create target file");
            let link = tree.0.join("ggml-cuda.dll");
            // Developer Mode (or an elevated token) is required to create a symlink; skip rather than
            // fail if this machine cannot, since that is an environment property, not a code property
            // — but say so out loud, so a skip is visible in test output rather than silently looking
            // like a pass for the wrong reason. `a_junction_inside_the_directory_makes_the_audit_unknown`,
            // above, is the unconditional coverage for this same code path.
            if std::os::windows::fs::symlink_file(&target, &link).is_err() {
                eprintln!("SKIPPED: cannot create file symlinks on this token");
                return;
            }

            let DaclAudit::Unknown(msg) = audit_dacl(&tree.0) else {
                panic!("a reparse point must fail the audit closed rather than being silently resolved");
            };
            assert!(msg.contains("reparse point"), "{msg}");
            assert!(msg.contains("ggml-cuda.dll"), "{msg}");
        }

        #[test]
        fn backend_dir_that_is_a_junction_is_unknown() {
            // Covers the other half of path pinning: `dir` itself being a reparse point, not merely
            // containing one. `check_path_chain` must catch this even though the target directory it
            // points at is perfectly clean.
            let tree = TempTree::new("dacl-junction");
            let target = tree.0.join("real");
            std::fs::create_dir(&target).expect("create junction target");
            let link = tree.0.join("link");
            dacl::create_junction(&link, &target).expect("junctions need no privilege to create");

            let DaclAudit::Unknown(msg) = audit_dacl(&link) else {
                panic!("a junctioned backend directory must fail the audit closed");
            };
            assert!(msg.contains("reparse point"), "{msg}");
        }

        #[test]
        fn backend_dir_that_is_a_symlink_is_unknown_when_creatable() {
            // A directory symlink needs Developer Mode or elevation to create, unlike a junction
            // (covered, unconditionally, above) — so this one may legitimately skip on a token that has
            // neither, and says so out loud rather than looking like a pass for the wrong reason.
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

            let DaclAudit::Unknown(msg) = audit_dacl(&link) else {
                panic!("a symlinked backend directory must fail the audit closed");
            };
            assert!(msg.contains("reparse point"), "{msg}");
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
                let restore = format!("D:(A;;FA;;;{me})");
                if let Ok(entries) = std::fs::read_dir(&self.0) {
                    for entry in entries.flatten() {
                        let _ = dacl::set_dacl_from_sddl(&entry.path(), &restore);
                    }
                }
                let _ = dacl::set_dacl_from_sddl(&self.0, &restore);
                let _ = std::fs::remove_dir_all(&self.0);
            }
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
    }
}
