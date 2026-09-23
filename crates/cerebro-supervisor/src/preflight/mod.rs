//! Decide whether a `llama-server` launch is allowed, before any process exists.
//!
//! [`preflight`] runs a fixed sequence of checks, each failing closed:
//!
//! 1. **Elevation** — refuse to run elevated at all (ARCHITECTURE.md section 8).
//! 2. **Backend directory shape** — `backend_dir` must be absolute (see "Why `backend_dir` must be
//!    absolute, and is never canonicalized" below), and `tool_name` must be a bare `.exe` file name
//!    with no path component, trailing dot/space, or alternate-data-stream colon.
//! 3. **Backend DACL** — the injected `DaclAuditor` must report the directory safe to spawn from,
//!    unless the caller supplied [`AllowUnsafeDacl`] with a non-blank reason, in which case launch
//!    proceeds and the acceptance is recorded as a [`PreflightWarning`]. An audit that could not
//!    complete ([`cerebro_probe::DaclAudit::Unknown`]) is refused unconditionally — the opt-in
//!    covers a *known* risk, never an unverifiable one. This metadata-only check runs *before*
//!    backend integrity below, both because it is by far the cheaper of the two (no file content is
//!    read) and because it is what makes the hash check's evidence durable: a hash proves nothing
//!    about a directory an untrusted principal could still be writing to.
//! 4. **Backend integrity** — every `.exe`/`.dll` in the backend directory must be present in, and
//!    hash-match, a pinned manifest; an unlisted file is refused exactly as a missing one is. A
//!    manifest whose keys collide once case-folded, or whose hash value is not exactly 64 hex
//!    digits, is refused outright, before any file is hashed.
//! 5. **Model files** — must exist, be regular files (not reparse points), carry the GGUF magic,
//!    contain no comma anywhere along the path actually used to reach them (including through a
//!    junction), and are canonicalized to absolute paths.
//! 6. **API key file** — must exist, be a regular file (not a reparse point), and, parsed exactly
//!    the way llama.cpp's `--api-key-file` parses it, contain at least one usable key: a non-empty,
//!    non-`#`-comment line of at least [`MIN_API_KEY_FILE_BYTES`] bytes with no control characters
//!    (which rules out a trailing `\r` from a CRLF file). See `verify_api_key_file`'s
//!    documentation for why a file that merely passes a size check is not enough.
//! 7. **Port** — loopback only; an explicit port is proven free by binding it (`0` is refused as a
//!    request, since it means "any", not "port zero"), "any" is resolved by binding port 0. See
//!    [`PortRequest`]'s documentation for the race this leaves unresolved and who must handle it.
//!
//! # `Preflighted` is proof, not a record
//!
//! Every [`Preflighted`] field is private. The only way to construct one is to pass every check
//! above; a caller cannot build a value that skips them and hand it to
//! [`build_hardened_server_argv`], the same discipline `cerebro_probe::UnsafeDacl` already applies
//! (its only constructor is [`cerebro_probe::DaclAudit::into_unsafe`]).
//!
//! # Why `backend_dir` must be absolute, and is never canonicalized
//!
//! A relative `backend_dir` would let hashing, the DACL audit, and the eventual spawn each resolve
//! it against whatever the process's current directory happens to be *at that moment* — and
//! `set_current_dir` is process-wide, so any thread can move it in between. Requiring an absolute
//! path up front removes that gap. It is deliberately never canonicalized, unlike the model and API
//! key paths: [`cerebro_probe::backend::audit_dacl`] examines the exact chain of path components
//! used to reach the directory (its "path pinning"), and resolving a junction in that chain before
//! the audit runs would make it examine an ancestor chain that is not the one actually in play.
//!
//! # Host state is injected
//!
//! Elevation and the DACL audit both read facts about the running machine that a hermetic test must
//! not depend on — this development machine's `%TEMP%` is itself writable by an untrusted
//! principal, which would make every fixture here fail the DACL check if it ran for real. Both are
//! collected in `HostChecks` and passed to `preflight_with`; [`preflight`] supplies the real
//! ones.
//!
//! # What this module does not do
//!
//! It does not scrub inherited `LLAMA_ARG_*` environment variables — upstream binds several flags
//! (`-dev`, `--port`, …) to environment names, so a variable left over from a previous launch could
//! silently override an argv value this module just proved safe. That scrubbing must happen in the
//! spawn task, immediately before `Command::spawn`, because it is a property of *that* process
//! image, not of anything checked here.
//!
//! It does not pin the child's working directory to the backend directory. That is also the spawn
//! task's job (see `cerebro_probe::backend::probe_command`'s documentation for why it matters).
//!
//! It cannot close the gap between "preflight verified this exact file, by handle" and "the spawned
//! process opens a model file by that same path, by name" — a process is handed a path, not a
//! handle, so the model (and API key) files are necessarily reopened by name downstream. Keeping
//! that reopen as close as possible to the verified [`Preflighted`] value, and treating a spawn that
//! fails or a server that never becomes ready as a reason to preflight again rather than to retry
//! the same argv, is the spawn/lifecycle layer's responsibility.

#[cfg(windows)]
mod elevation;
#[cfg(windows)]
mod win_path;

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};

use cerebro_flags::{ArgvBuilder, Dialect, EmitError, Flag};
use cerebro_probe::{DaclAudit, ProbeError, UnsafeDacl};
use thiserror::Error;

/// The GGUF container magic, always the first four bytes of a valid file.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Minimum length, in bytes, of each individual key line inside an API key file — not the file's
/// total size (a file can carry many keys, comments, and blank lines around one short one). Not a
/// strength requirement on the key material itself — just enough to reject an empty, truncated, or
/// placeholder key before it reaches `--api-key-file`. See `verify_api_key_file`.
pub const MIN_API_KEY_FILE_BYTES: u64 = 32;

/// Maximum size of an API key file this module will read. Bounds the read from an already-open
/// handle so an attacker-sized file cannot exhaust memory; llama.cpp itself imposes no such limit,
/// but no legitimate key file is anywhere near this large.
const MAX_API_KEY_FILE_BYTES: u64 = 64 * 1024;

/// The only address Cerebro ever binds a backend server to.
pub const LOOPBACK_HOST: &str = "127.0.0.1";

/// Which model slot a check applies to, so target and draft models share one set of error variants
/// instead of duplicating every model-file variant twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    /// The model being served.
    Target,
    /// The speculative-decoding draft model.
    Draft,
}

impl std::fmt::Display for ModelRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelRole::Target => "target",
            ModelRole::Draft => "draft",
        })
    }
}

/// How the server's listening port should be chosen.
///
/// # The unavoidable bind race
///
/// [`preflight`] binds the requested (or, for [`PortRequest::Any`], an OS-chosen) port and releases
/// it immediately. That proves the port was free *at that instant*, not that it will still be free
/// when the real spawn happens moments later — any process on the machine, including another
/// Cerebro instance, can claim it in between. No amount of cleverness in this module removes that
/// race: only holding the listening socket open across the child spawn would, and that is not how
/// `llama-server` binds its own port.
///
/// The spawn/lifecycle layer must therefore treat an early child exit, or its readiness probe never
/// succeeding, as an ordinary launch failure and retry from [`preflight`] — which will pick a
/// different free port for [`PortRequest::Any`], or surface [`PreflightError::PortUnavailable`]
/// again for an explicit one that is genuinely still taken — rather than assuming the port
/// [`preflight`] resolved is still good.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortRequest {
    /// Use exactly this port; refuse if it is not free. `0` is rejected (see
    /// [`PreflightError::InvalidExplicitPort`]): it does not mean "port zero", it means "let the OS
    /// choose", which is [`PortRequest::Any`].
    Explicit(u16),
    /// Let the OS pick a free ephemeral port.
    Any,
}

/// An explicit, reasoned opt-in to launch from a backend directory whose DACL is not owner-only.
///
/// Carrying a mandatory, non-blank reason is deliberate: this is the one override in the whole
/// preflight gate, and every acceptance of it must be attributable to someone's stated judgment
/// call, not a silent default. It never covers an audit that could not complete at all — see
/// [`PreflightError::BackendDirUnauditable`].
///
/// Accepting this opt-in narrows what the rest of preflight can promise: the backend integrity
/// check (step 3) still runs and still must pass, but the named untrusted principals remain free to
/// replace any `.exe`/`.dll` in the backend directory *after* that check and *before* the child
/// process loads it. The manifest match this opt-in permits past is therefore proof of integrity as
/// of preflight time only, never as of the eventual `LoadLibrary` — see
/// [`Preflighted::backend_hashes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowUnsafeDacl {
    reason: String,
}

impl AllowUnsafeDacl {
    /// Record the opt-in along with why it was made.
    ///
    /// Refuses a reason that is empty or all whitespace: accepting one would let this override go
    /// through with nothing a later reader could hold anyone to.
    pub fn new(reason: impl Into<String>) -> Result<Self, PreflightError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(PreflightError::UnsafeDaclReasonMissing);
        }
        Ok(Self { reason })
    }

    /// The reason this opt-in was given.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Everything needed to decide whether a `llama-server` launch may proceed.
#[derive(Debug, Clone)]
pub struct LaunchSpec {
    /// Directory the backend binaries live in. Must be absolute (see this module's top-level docs);
    /// `preflight_with` never resolves a relative one against the current directory.
    pub backend_dir: PathBuf,
    /// Bare file name of the tool to launch (e.g. `llama-server.exe`), resolved inside
    /// `backend_dir`. Must be a single normal path component ending in `.exe`, with no path
    /// separator, trailing dot or space, or `:` (which would name an alternate data stream).
    pub tool_name: String,
    /// Expected backend files: file name (as it appears on disk) mapped to lowercase hex SHA-256.
    /// Every `.exe`/`.dll` actually present in `backend_dir` must appear here with a matching hash,
    /// and every entry here must be present on disk — extra and missing files are both refused. Two
    /// entries whose names collide once case-folded are refused rather than silently merged.
    pub manifest: BTreeMap<String, String>,
    /// Path to the target model.
    pub model_path: PathBuf,
    /// Path to the speculative-decoding draft model, if any.
    pub draft_model_path: Option<PathBuf>,
    /// How the listening port should be chosen.
    pub port: PortRequest,
    /// Path to the file `--api-key-file` will point at.
    pub api_key_file: PathBuf,
    /// Opt-in to launch despite an unsafe backend directory DACL. `None` means the DACL must be
    /// owner-only.
    pub allow_unsafe_dacl: Option<AllowUnsafeDacl>,
}

/// A non-fatal condition recorded during a successful preflight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreflightWarning {
    /// Launch proceeded despite an unsafe backend directory DACL, because [`AllowUnsafeDacl`] was
    /// given. Carries the caller's stated reason and the untrusted principals found, so the warning
    /// is actionable without re-running the audit.
    UnsafeDaclAccepted {
        /// The reason given for accepting the risk.
        reason: String,
        /// Distinct untrusted principal names found by the audit.
        principals: Vec<String>,
        /// Full rendering of the underlying [`UnsafeDacl`], for logs.
        detail: String,
    },
}

impl std::fmt::Display for PreflightWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreflightWarning::UnsafeDaclAccepted {
                reason,
                principals,
                detail,
            } => {
                write!(
                    f,
                    "launching despite an unsafe backend directory DACL (reason: {reason}); \
                     untrusted principals: {}; these principals can replace backend binaries after \
                     this manifest check ran, so integrity is only verified as of preflight, not as \
                     of the child's DLL load; {detail}",
                    principals.join(", ")
                )
            }
        }
    }
}

/// Everything the spawner needs to launch, once every check has passed.
///
/// Every field is private; the only way to build one is [`preflight`] or `preflight_with`
/// succeeding, so a value of this type existing is itself proof every check ran. Use the accessors
/// below to read it.
#[derive(Debug, Clone)]
pub struct Preflighted {
    backend_dir: PathBuf,
    tool_name: String,
    backend_hashes: BTreeMap<String, String>,
    model_path: PathBuf,
    draft_model_path: Option<PathBuf>,
    port: u16,
    api_key_file: PathBuf,
    warnings: Vec<PreflightWarning>,
}

impl Preflighted {
    /// The backend directory that was hashed and DACL-audited, absolute but — unlike the model and
    /// API key paths — never canonicalized (see this module's top-level docs on why).
    pub fn backend_dir(&self) -> &Path {
        &self.backend_dir
    }

    /// The bare tool file name that was verified against the manifest, as given.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Absolute path of the verified tool: [`Preflighted::backend_dir`] joined with
    /// [`Preflighted::tool_name`].
    pub fn tool_path(&self) -> PathBuf {
        self.backend_dir.join(&self.tool_name)
    }

    /// File name -> lowercase hex SHA-256 for every backend file that was verified against the
    /// manifest.
    ///
    /// These hashes are only as durable as the backend directory's DACL. Under an ordinary
    /// owner-only DACL, nothing untrusted could have modified the files between this hash and now.
    /// Under [`AllowUnsafeDacl`], the named untrusted principals could still replace any of these
    /// files at any point after preflight ran — see [`AllowUnsafeDacl`]'s documentation and
    /// [`PreflightWarning::UnsafeDaclAccepted`]. Check [`Preflighted::warnings`] before treating
    /// these hashes as a guarantee that holds at spawn time.
    pub fn backend_hashes(&self) -> &BTreeMap<String, String> {
        &self.backend_hashes
    }

    /// Canonicalized, absolute, comma-free path to the target model.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Canonicalized, absolute, comma-free path to the draft model, if one was given.
    pub fn draft_model_path(&self) -> Option<&Path> {
        self.draft_model_path.as_deref()
    }

    /// The port to bind. Always loopback-only (see [`LOOPBACK_HOST`]); observed free at preflight
    /// time only — see [`PortRequest`]'s documentation for the unavoidable race between that
    /// observation and the real spawn.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Canonicalized, absolute path to the API key file.
    pub fn api_key_file(&self) -> &Path {
        &self.api_key_file
    }

    /// Non-fatal conditions recorded while preflighting.
    pub fn warnings(&self) -> &[PreflightWarning] {
        &self.warnings
    }

    /// Build a `Preflighted` directly, bypassing every check [`preflight`] normally runs.
    ///
    /// `cfg(test)`-gated and crate-private rather than exposed through a `test-support` feature
    /// (contrast `cerebro_probe::backend::WritableGrant::test_new`, gated that way because its
    /// callers are a different crate's tests): every caller that needs this is a unit test compiled
    /// into `cerebro-supervisor`'s own `cargo test` run, in `spawn`'s test module, never an external
    /// consumer, so there is no reason to widen this past `#[cfg(test)]` at all. It exists so
    /// `spawn`'s tests can prove a hostile `backend_dir` reaches the same refusal through
    /// [`crate::spawn::spawn_preflighted`] that it does through [`crate::spawn::spawn_unaudited`],
    /// without a `Preflighted` value having to survive a real (and, for a hostile path, doomed to
    /// fail for unrelated reasons — no such directory, no manifest, no DACL to audit) call to
    /// [`preflight`] first. This does not weaken [`Preflighted`]'s real guarantee: every field it
    /// still keeps private, and the only route to one that reaches outside `#[cfg(test)]` code is
    /// still [`preflight`]/[`preflight_with`] succeeding.
    #[cfg(test)]
    pub(crate) fn test_new(backend_dir: PathBuf, tool_name: String) -> Self {
        Self {
            backend_dir,
            tool_name,
            backend_hashes: BTreeMap::new(),
            model_path: PathBuf::new(),
            draft_model_path: None,
            port: 0,
            api_key_file: PathBuf::new(),
            warnings: Vec::new(),
        }
    }
}

/// Everything that can make [`preflight`] refuse a launch.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PreflightError {
    /// This process's own token is elevated. Nothing in the inference path needs admin rights
    /// (ARCHITECTURE.md section 8), so this is refused unconditionally rather than warned about.
    #[error(
        "refusing to launch: this process is running elevated, and nothing in the inference \
         path needs admin rights"
    )]
    RunningElevated,

    /// The elevation check itself could not be completed. Treated as a refusal, not as "probably
    /// not elevated" — the same fail-closed stance `cerebro-probe` takes on an unauditable DACL.
    #[error("could not determine whether this process is elevated: {0}")]
    ElevationCheckFailed(String),

    /// `backend_dir` was relative. Hashing, the DACL audit, and the eventual spawn could each
    /// resolve it against a different current directory; see this module's top-level docs.
    #[error("backend directory must be an absolute path, got {0}")]
    BackendDirNotAbsolute(String),

    /// `tool_name` was not a bare `.exe` file name: it contained a path component, a trailing dot or
    /// space, or a `:` (which would name an alternate data stream).
    #[error(
        "backend tool name must be a single, bare .exe file name with no trailing dot, space, or \
         alternate-data-stream colon, got {0:?}"
    )]
    BackendToolEscapesDir(String),

    /// `tool_name` does not name any file the manifest check just verified. Preflight has proved
    /// the manifest's files are intact, but it has proved nothing about a name outside that set.
    #[error("backend tool {tool:?} in {dir} was not among the manifest-verified files")]
    BackendToolNotVerified {
        /// The backend directory.
        dir: String,
        /// The requested tool name.
        tool: String,
    },

    /// Two manifest entries name the same file once case-folded (Windows file names are
    /// case-insensitive), which would otherwise silently collapse into one entry during the
    /// integrity check — reporting a manifest as satisfied when in fact only one of the two names
    /// it claimed to pin was ever checked.
    #[error(
        "manifest entries {a:?} and {b:?} differ only in case and would silently collapse into \
         one file"
    )]
    ManifestNameCollision {
        /// The first colliding entry, in manifest iteration order.
        a: String,
        /// The second colliding entry.
        b: String,
    },

    /// A manifest entry's hash value is not exactly 64 ASCII hex digits. Reported distinctly from
    /// [`PreflightError::BackendFileHashMismatch`]: an empty, truncated, or typo'd manifest value is
    /// a defect in the pinned manifest itself, not evidence that a binary on disk was tampered with,
    /// and an operator needs to be able to tell those two apart.
    #[error("manifest hash for {file:?} is not 64 hex digits: {value:?}")]
    ManifestHashMalformed {
        /// The manifest entry whose value is malformed.
        file: String,
        /// The malformed value itself.
        value: String,
    },

    /// The backend directory could not be enumerated or one of its files could not be read.
    #[error("could not verify backend integrity: {0}")]
    BackendIntegrityIo(#[from] ProbeError),

    /// A file in the backend directory carries a hash the manifest does not agree with.
    #[error(
        "backend file {file} in {dir} does not match its manifest hash \
         (expected {expected}, got {actual})"
    )]
    BackendFileHashMismatch {
        /// The backend directory.
        dir: String,
        /// The file whose hash did not match.
        file: String,
        /// The hash the manifest recorded.
        expected: String,
        /// The hash actually computed.
        actual: String,
    },

    /// A `.exe`/`.dll` exists in the backend directory but is not in the manifest. Exactly the
    /// planted-DLL threat ARCHITECTURE.md section 8 names.
    #[error(
        "backend file {file} in {dir} is not in the manifest; an unlisted file must be refused, \
         not silently trusted"
    )]
    BackendFileUnexpected {
        /// The backend directory.
        dir: String,
        /// The file not accounted for.
        file: String,
    },

    /// The manifest requires a file that is not present in the backend directory.
    #[error("backend file {file} is required by the manifest but is missing from {dir}")]
    BackendFileMissing {
        /// The backend directory.
        dir: String,
        /// The missing file's manifest name.
        file: String,
    },

    /// The backend directory's DACL is not owner-only, and no [`AllowUnsafeDacl`] opt-in was given.
    #[error("refusing to spawn from {dir}: {audit}")]
    BackendDirUnsafe {
        /// The backend directory that failed the audit.
        dir: String,
        /// Why it failed, structured so a caller can recover every untrusted principal.
        audit: UnsafeDacl,
    },

    /// The backend directory's DACL could not be audited at all. Refused unconditionally, even with
    /// [`AllowUnsafeDacl`]: that opt-in accepts a *known* risk it can name principals for, never an
    /// audit that produced no answer — a caller reading `principals` off the resulting warning must
    /// never be told "these are the accounts involved" when nobody actually is known.
    #[error("cannot audit backend directory {dir}'s DACL, so it cannot be launched from: {cause}")]
    BackendDirUnauditable {
        /// The backend directory.
        dir: String,
        /// Why the audit could not complete.
        cause: String,
    },

    /// A model file (target or draft) does not exist.
    #[error("{role} model not found: {path}")]
    ModelNotFound {
        /// Which model this is.
        role: ModelRole,
        /// The path that was checked.
        path: String,
    },

    /// A model path resolved to something other than a regular file (a directory, most commonly).
    #[error("{role} model path is not a regular file: {path}")]
    ModelNotRegularFile {
        /// Which model this is.
        role: ModelRole,
        /// The path that was checked.
        path: String,
    },

    /// A model path is a reparse point (symlink, junction, or mount point). Refused rather than
    /// resolved, so a hostile redirect cannot be planted between the check and the spawn.
    #[error("{role} model path is a reparse point, which is refused: {path}")]
    ModelIsReparsePoint {
        /// Which model this is.
        role: ModelRole,
        /// The path that was checked.
        path: String,
    },

    /// A model file's first four bytes are not the GGUF magic, including a file too short to hold
    /// four bytes at all.
    #[error("{role} model does not begin with the GGUF magic bytes: {path}")]
    ModelBadMagic {
        /// Which model this is.
        role: ModelRole,
        /// The path that was checked.
        path: String,
    },

    /// A model path contains a comma, which several llama.cpp flags (e.g. `-dev`, `-ts`) reserve as
    /// a list separator; a comma in a path value could be read as more than one argument. Checked
    /// both on the path as given and, separately, on its canonicalized form — a junction earlier in
    /// the path can introduce a comma that never appeared in the original spelling, and the
    /// canonical form is what actually reaches argv.
    #[error("{role} model path contains a comma, which argv list flags reserve as a separator: {path}")]
    ModelPathHasComma {
        /// Which model this is.
        role: ModelRole,
        /// The path that was checked (the as-given spelling, or the canonical one, whichever first
        /// contained the comma).
        path: String,
    },

    /// A model path could not be read or canonicalized for a reason other than "not found".
    #[error("{role} model path {path} could not be read: {cause}")]
    ModelPathIo {
        /// Which model this is.
        role: ModelRole,
        /// The path that was checked.
        path: String,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// The API key file does not exist.
    #[error("API key file does not exist: {0}")]
    ApiKeyFileMissing(String),

    /// The API key file exists but is not a regular file (most commonly, a directory).
    #[error("API key file path is not a regular file: {0}")]
    ApiKeyFileNotRegularFile(String),

    /// The API key file is a reparse point (symlink, junction, or mount point). Refused rather than
    /// resolved, for the same reason a model path is: see [`PreflightError::ModelIsReparsePoint`].
    #[error("API key file path is a reparse point, which is refused: {0}")]
    ApiKeyFileIsReparsePoint(String),

    /// The API key file is larger than `MAX_API_KEY_FILE_BYTES`. No legitimate key file needs to
    /// be this large; refusing it up front bounds the read that follows.
    #[error("API key file {path} is {len} bytes, over the {MAX_API_KEY_FILE_BYTES}-byte limit")]
    ApiKeyFileTooLarge {
        /// The path that was checked.
        path: String,
        /// Its actual size in bytes.
        len: u64,
    },

    /// Parsed the way llama.cpp's `--api-key-file` parses it (skip empty lines and `#` comments),
    /// the file contains zero usable keys. A file of only blank lines or only comments passes a bare
    /// size check but leaves llama.cpp with an empty key list, which serves every request with no
    /// authentication at all — see `verify_api_key_file`'s documentation.
    #[error(
        "API key file {0} contains no usable key once blank lines and #-comments are skipped, \
         exactly as llama.cpp's own parser would skip them; the server would run with no keys \
         configured, i.e. no authentication"
    )]
    ApiKeyFileNoKeys(String),

    /// A non-blank, non-comment line in the API key file is shorter than [`MIN_API_KEY_FILE_BYTES`].
    #[error("API key file {path} line {line} is only {len} bytes; at least {MIN_API_KEY_FILE_BYTES} are required")]
    ApiKeyTooShort {
        /// The path that was checked.
        path: String,
        /// 1-based line number within the file.
        line: usize,
        /// The line's length in bytes.
        len: usize,
    },

    /// A non-blank, non-comment line in the API key file contains a control character, most
    /// commonly a trailing `\r` left by a CRLF line ending. llama.cpp reads the file with
    /// `std::getline` splitting on `'\n'` only, so that `\r` becomes part of the key it parses; no
    /// HTTP client can send that byte in a bearer token, so trimming it here would make Cerebro's
    /// idea of the key differ from the server's rather than fixing anything.
    #[error("API key file {path} line {line} contains a control character (e.g. a CRLF '\\r')")]
    ApiKeyFileMalformed {
        /// The path that was checked.
        path: String,
        /// 1-based line number within the file.
        line: usize,
    },

    /// The API key file could not be read or canonicalized for a reason other than "not found".
    #[error("API key file {path} could not be read: {cause}")]
    ApiKeyFileIo {
        /// The path that was checked.
        path: String,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// [`AllowUnsafeDacl::new`] was given a reason that was empty or all whitespace.
    #[error(
        "an unsafe-DACL opt-in must carry a non-blank reason, so every acceptance is attributable"
    )]
    UnsafeDaclReasonMissing,

    /// `PortRequest::Explicit(0)` was requested. `0` does not mean "port zero"; it means "let the OS
    /// choose", which is [`PortRequest::Any`].
    #[error("port 0 is not a valid explicit port request; use PortRequest::Any instead")]
    InvalidExplicitPort,

    /// An explicitly requested port is not free on loopback.
    #[error("requested port {port} is not free on {LOOPBACK_HOST}: {cause}")]
    PortUnavailable {
        /// The port that was requested.
        port: u16,
        /// Underlying OS error, as text.
        cause: String,
    },

    /// No free ephemeral port could be reserved on loopback.
    #[error("could not reserve a free port on {LOOPBACK_HOST}: {cause}")]
    PortReservationFailed {
        /// Underlying OS error, as text.
        cause: String,
    },
}

/// Abstraction over "is this process elevated", so [`preflight_with`] can be exercised in tests
/// without depending on the ambient token state of whatever account runs them — CI must be able to
/// prove the refusal branch without actually running elevated.
///
/// `pub(crate)`, not `pub`: this is a test seam, not a public extension point. A [`Preflighted`]
/// value is meant to be proof that every check ran; if an outside crate could hand `preflight_with`
/// a fake [`ElevationCheck`] that always answers `false`, the proof would be worthless. The unit
/// tests and the `#[ignore]` live test below are both inside this crate, so `pub(crate)` is all
/// they need.
pub(crate) trait ElevationCheck {
    /// Whether the current process is elevated.
    fn is_elevated(&self) -> Result<bool, PreflightError>;
}

/// The real elevation check, backed by this process's own token.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ProcessTokenElevation;

impl ElevationCheck for ProcessTokenElevation {
    fn is_elevated(&self) -> Result<bool, PreflightError> {
        #[cfg(windows)]
        {
            elevation::query()
        }
        #[cfg(not(windows))]
        {
            Err(PreflightError::ElevationCheckFailed(
                "elevation check is implemented for Windows only".into(),
            ))
        }
    }
}

/// Abstraction over "what does this backend directory's DACL look like", so [`preflight_with`] can
/// be exercised in tests without depending on the ambient ACL of wherever a fixture directory
/// happens to live — this development machine's own `%TEMP%` is itself writable by an untrusted
/// principal, which would make every hermetic fixture fail a real audit regardless of its own ACL.
///
/// `pub(crate)` for the same reason as [`ElevationCheck`]: a caller outside this crate must never
/// be able to substitute a `DaclAuditor` that always reports [`DaclAudit::OwnerOnly`] and get back
/// a real [`Preflighted`].
pub(crate) trait DaclAuditor {
    /// Audit `dir` exactly as [`cerebro_probe::backend::audit_dacl`] would.
    fn audit(&self, dir: &Path) -> DaclAudit;
}

/// The real DACL auditor, backed by [`cerebro_probe::backend::audit_dacl`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ProbeDaclAuditor;

impl DaclAuditor for ProbeDaclAuditor {
    fn audit(&self, dir: &Path) -> DaclAudit {
        cerebro_probe::backend::audit_dacl(dir)
    }
}

/// Every piece of host state [`preflight_with`] needs from outside pure disk I/O, collected so a
/// caller supplies both or neither rather than injecting one and leaving the other real by
/// accident.
///
/// `pub(crate)`: kept inside the crate along with the traits it holds, so [`preflight`] is the only
/// way to obtain a [`Preflighted`] from outside this crate. See the module docs' "`Preflighted` is
/// proof, not a record" section.
pub(crate) struct HostChecks<'a> {
    /// How to decide whether this process is elevated.
    pub(crate) elevation: &'a dyn ElevationCheck,
    /// How to audit a backend directory's DACL.
    pub(crate) dacl: &'a dyn DaclAuditor,
}

/// Decide whether the launch described by `spec` may proceed.
///
/// The only public entry point into this module's checks: always uses the real process-token
/// elevation check and the real DACL audit, so a [`Preflighted`] returned from here really is proof
/// every check ran against this machine. See `preflight_with` (crate-private) for the injectable
/// version the test suite uses to force every branch without requiring an elevated CI runner or a
/// fixture directory with a specific real-world ACL.
pub fn preflight(spec: &LaunchSpec) -> Result<Preflighted, PreflightError> {
    preflight_with(
        spec,
        &HostChecks {
            elevation: &ProcessTokenElevation,
            dacl: &ProbeDaclAuditor,
        },
    )
}

/// [`preflight`] with injectable host checks. Not exported outside this crate — see [`HostChecks`].
pub(crate) fn preflight_with(
    spec: &LaunchSpec,
    host: &HostChecks<'_>,
) -> Result<Preflighted, PreflightError> {
    // 1. Elevation. Cheapest check and the most fundamental refusal, so it runs before anything
    // that touches disk or the network.
    if host.elevation.is_elevated()? {
        return Err(PreflightError::RunningElevated);
    }

    // 2. Backend directory shape. `backend_dir` must be absolute before it is used for anything —
    // see this module's top-level docs — and `tool_name` must be a bare, unambiguous `.exe` name.
    if !spec.backend_dir.is_absolute() {
        return Err(PreflightError::BackendDirNotAbsolute(
            spec.backend_dir.display().to_string(),
        ));
    }
    verify_tool_name(&spec.tool_name)?;

    // 3. DACL first: metadata-only and cheap, and it is what makes the hashes computed in step 4
    // durable. A hash proves nothing about a directory an untrusted principal could still be
    // writing to, so auditing the DACL has to come before spending the I/O to compute one.
    let mut warnings = Vec::new();
    if let Some(warning) = verify_dacl(&spec.backend_dir, host.dacl, spec.allow_unsafe_dacl.as_ref())? {
        warnings.push(warning);
    }

    // 4. Backend integrity, now taken under an owner-only (or explicitly accepted) DACL.
    let backend_hashes = verify_backend_integrity(&spec.backend_dir, &spec.manifest)?;
    let tool_lower = spec.tool_name.to_ascii_lowercase();
    if !backend_hashes
        .keys()
        .any(|name| name.to_ascii_lowercase() == tool_lower)
    {
        return Err(PreflightError::BackendToolNotVerified {
            dir: spec.backend_dir.display().to_string(),
            tool: spec.tool_name.clone(),
        });
    }

    // 5. Model files.
    let model_path = verify_model_file(ModelRole::Target, &spec.model_path)?;
    let draft_model_path = spec
        .draft_model_path
        .as_deref()
        .map(|p| verify_model_file(ModelRole::Draft, p))
        .transpose()?;

    // 6. API key file.
    let api_key_file = verify_api_key_file(&spec.api_key_file)?;

    // 7. Port.
    let port = resolve_port(spec.port)?;

    Ok(Preflighted {
        backend_dir: spec.backend_dir.clone(),
        tool_name: spec.tool_name.clone(),
        backend_hashes,
        model_path,
        draft_model_path,
        port,
        api_key_file,
        warnings,
    })
}

/// Validate that `tool_name` is a single, bare `.exe` file name that cannot escape the backend
/// directory or name something other than what it appears to.
///
/// The shape check itself — exactly one [`Component::Normal`](std::path::Component::Normal), no
/// trailing `.`/` `, no `:`, and the `.exe` extension — is [`crate::names::is_bare_exe_name`],
/// shared with `spawn::verify_spawn_target` so the two callers can never drift apart; see that
/// module's docs for why a plain component count (an earlier version of this check) was not enough.
fn verify_tool_name(tool_name: &str) -> Result<(), PreflightError> {
    if !crate::names::is_bare_exe_name(tool_name) {
        return Err(PreflightError::BackendToolEscapesDir(tool_name.to_string()));
    }
    Ok(())
}

/// Hash every `.exe`/`.dll` in `dir` and reconcile the result against `manifest`.
///
/// Matches file names case-insensitively (Windows file names are case-insensitive) but keeps the
/// on-disk spelling in the returned map and in any error, since that is what a human needs to find
/// the file. An unlisted file and a listed-but-absent file are both refused, and hashes are compared
/// as full lowercase hex.
fn verify_backend_integrity(
    dir: &Path,
    manifest: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, PreflightError> {
    validate_manifest(manifest)?;

    let (found, _ggml_backends) = cerebro_probe::backend::hash_directory(dir)?;

    let mut remaining: BTreeMap<String, String> = manifest
        .iter()
        .map(|(name, hash)| (name.to_ascii_lowercase(), hash.to_ascii_lowercase()))
        .collect();

    let mut verified = BTreeMap::new();
    for (name, actual_hash) in &found {
        let lower_name = name.to_ascii_lowercase();
        let Some(expected_hash) = remaining.remove(&lower_name) else {
            return Err(PreflightError::BackendFileUnexpected {
                dir: dir.display().to_string(),
                file: name.clone(),
            });
        };
        let actual_hash = actual_hash.to_ascii_lowercase();
        if actual_hash != expected_hash {
            return Err(PreflightError::BackendFileHashMismatch {
                dir: dir.display().to_string(),
                file: name.clone(),
                expected: expected_hash,
                actual: actual_hash,
            });
        }
        verified.insert(name.clone(), actual_hash);
    }

    // Anything left in `remaining` was required by the manifest but never found on disk.
    if let Some(missing) = remaining.into_keys().next() {
        return Err(PreflightError::BackendFileMissing {
            dir: dir.display().to_string(),
            file: missing,
        });
    }

    Ok(verified)
}

/// Validate the manifest's own shape before any file on disk is touched: every hash value must be
/// exactly 64 ASCII hex digits, and no two entries may fold to the same lowercase name.
///
/// The hash-shape check runs first and is deliberately separate from
/// [`PreflightError::BackendFileHashMismatch`]: an empty, truncated, or typo'd manifest value is a
/// defect in the pinned manifest an operator wrote, not evidence that a binary on disk was
/// tampered with, and conflating the two would tell whoever reads the refusal the wrong story.
///
/// The name-collision check exists because downstream, both this function's caller and
/// [`cerebro_probe::backend::hash_directory`]'s result are matched up by lowercase name (Windows
/// file names are case-insensitive), which means building a `BTreeMap` keyed on the lowercase form
/// — exactly what naive construction of `remaining` in [`verify_backend_integrity`] does — would let
/// a second colliding entry silently overwrite the first with no error at all. A manifest is a
/// security control; two names that were meant to pin two different-looking files but collapse into
/// one is a defect in the manifest, not something to resolve quietly by picking one.
fn validate_manifest(manifest: &BTreeMap<String, String>) -> Result<(), PreflightError> {
    for (name, hash) in manifest {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(PreflightError::ManifestHashMalformed {
                file: name.clone(),
                value: hash.clone(),
            });
        }
    }

    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for name in manifest.keys() {
        let lower = name.to_ascii_lowercase();
        if let Some(first) = seen.insert(lower, name.clone()) {
            return Err(PreflightError::ManifestNameCollision {
                a: first,
                b: name.clone(),
            });
        }
    }
    Ok(())
}

/// Audit the backend directory's DACL, refusing unless it is owner-only or the caller opted in.
///
/// An audit that could not complete ([`DaclAudit::Unknown`]) is refused even with
/// [`AllowUnsafeDacl`]: that opt-in accepts a specific, named risk, and an unauditable directory has
/// no principals to name.
fn verify_dacl(
    dir: &Path,
    dacl: &dyn DaclAuditor,
    allow_unsafe: Option<&AllowUnsafeDacl>,
) -> Result<Option<PreflightWarning>, PreflightError> {
    let grants = match dacl.audit(dir) {
        DaclAudit::OwnerOnly => return Ok(None),
        DaclAudit::WritableByOthers(grants) => grants,
        DaclAudit::Unknown(cause) => {
            return Err(PreflightError::BackendDirUnauditable {
                dir: dir.display().to_string(),
                cause,
            });
        }
    };

    let Some(allow) = allow_unsafe else {
        return Err(PreflightError::BackendDirUnsafe {
            dir: dir.display().to_string(),
            audit: UnsafeDacl::WritableByOthers(grants),
        });
    };

    let mut principals: Vec<String> = grants.iter().map(|g| g.principal().to_string()).collect();
    principals.sort_unstable();
    principals.dedup();
    let detail = UnsafeDacl::WritableByOthers(grants).to_string();

    Ok(Some(PreflightWarning::UnsafeDaclAccepted {
        reason: allow.reason().to_string(),
        principals,
        detail,
    }))
}

/// Validate one model file (target or draft) and return its canonicalized absolute path.
///
/// The reparse-point and directory classification, and the no-follow open itself, are delegated to
/// [`open_no_follow`] — the same helper [`verify_api_key_file`] uses, so both files preflight hands
/// to the child get identical resolution discipline. The GGUF magic is then read from that same
/// handle, and the canonical path is resolved from it too (see [`win_path::final_path`]'s
/// documentation for why re-resolving `path` by name instead would reopen the exact hostile-redirect
/// window this function exists to close).
fn verify_model_file(role: ModelRole, path: &Path) -> Result<PathBuf, PreflightError> {
    // Checked on the path string first, and before any filesystem access: a comma is unsafe
    // regardless of what the file turns out to contain, and this fails fast on a spec that could
    // not possibly be launched.
    reject_comma(role, path)?;

    // See `open_no_follow`'s documentation for why a reparse point (file symlink, junction, or
    // mount point) is classified by a no-follow lookup *before* anything is opened, rather than by
    // letting the open itself fail.
    #[cfg(windows)]
    let mut file = open_no_follow(path).map_err(|e| match e {
        NoFollowOpen::ReparsePoint => PreflightError::ModelIsReparsePoint {
            role,
            path: path.display().to_string(),
        },
        NoFollowOpen::Directory => PreflightError::ModelNotRegularFile {
            role,
            path: path.display().to_string(),
        },
        NoFollowOpen::Io(e) => model_io_error(role, path, e),
    })?;
    #[cfg(not(windows))]
    let mut file = fs::File::open(path).map_err(|e| model_io_error(role, path, e))?;

    let meta = file.metadata().map_err(|e| model_io_error(role, path, e))?;
    if !meta.is_file() {
        return Err(PreflightError::ModelNotRegularFile {
            role,
            path: path.display().to_string(),
        });
    }

    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) if magic == GGUF_MAGIC => {}
        Ok(()) => {
            return Err(PreflightError::ModelBadMagic {
                role,
                path: path.display().to_string(),
            });
        }
        // A file too short to hold the magic at all is a bad-magic file, not an I/O failure — the
        // earlier version conflated this with every other read error (e.g. a sharing violation),
        // which told the operator the wrong thing was wrong.
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(PreflightError::ModelBadMagic {
                role,
                path: path.display().to_string(),
            });
        }
        Err(e) => return Err(model_io_error(role, path, e)),
    }

    // Resolved from the handle already open above, not by re-resolving `path` from scratch: see
    // `win_path::final_path`'s documentation for why `fs::canonicalize(path)` cannot make this
    // guarantee. Note that this still leaves one window: the spawn task reopens the model file by
    // *name* (an OS process is handed a path, not a handle), so a swap between preflight returning
    // and that reopen is not something this function can close — that is the lifecycle layer's
    // problem, the same way the port-bind race documented on `PortRequest` is.
    #[cfg(windows)]
    let canonical = win_path::final_path(&file).map_err(|e| model_io_error(role, path, e))?;
    #[cfg(not(windows))]
    let canonical = fs::canonicalize(path).map_err(|e| model_io_error(role, path, e))?;
    // The canonical path — not the as-given spelling already checked above — is what
    // `build_hardened_server_argv` actually places in argv. A junction anywhere in `path`'s
    // ancestors can resolve to a target whose own path contains a comma that never appeared in what
    // the caller wrote, so it needs the identical check run again here.
    reject_comma(role, &canonical)?;
    Ok(canonical)
}

/// Refuse a path containing a comma, which several llama.cpp list flags (`-dev`, `-ts`, `-ot`)
/// reserve as a separator.
fn reject_comma(role: ModelRole, path: &Path) -> Result<(), PreflightError> {
    if path.to_string_lossy().contains(',') {
        return Err(PreflightError::ModelPathHasComma {
            role,
            path: path.display().to_string(),
        });
    }
    Ok(())
}

fn model_io_error(role: ModelRole, path: &Path, e: std::io::Error) -> PreflightError {
    if e.kind() == std::io::ErrorKind::NotFound {
        PreflightError::ModelNotFound {
            role,
            path: path.display().to_string(),
        }
    } else {
        PreflightError::ModelPathIo {
            role,
            path: path.display().to_string(),
            cause: e.to_string(),
        }
    }
}

/// Outcome of [`open_no_follow`], generic across every caller in this module so each can map it to
/// its own typed [`PreflightError`] variant.
#[cfg(windows)]
enum NoFollowOpen {
    /// The pre-open lookup or the open itself failed for an ordinary I/O reason (including "not
    /// found").
    Io(std::io::Error),
    /// The path is, or between the pre-open lookup and the open became, a reparse point.
    ReparsePoint,
    /// The path names a plain directory, not a reparse point.
    Directory,
}

/// Open `path` for reading without following a reparse point in its final path component,
/// classifying a reparse point and a plain directory along the way.
///
/// A no-follow `symlink_metadata` lookup classifies both *before* anything is opened. That
/// ordering matters for a directory: an earlier version of this logic (in what is now
/// [`verify_model_file`]) fell through to `File::open` and expected the attempt to fail in a way
/// that reported [`NoFollowOpen::ReparsePoint`]. That claim was never exercised against the real
/// API — `CreateFileW` on a directory, reparse point or not, fails with `ERROR_ACCESS_DENIED`
/// unless the open sets `FILE_FLAG_BACKUP_SEMANTICS`, which this function deliberately never does
/// (a directory must never be treated as an openable file at all) — so without the pre-open check a
/// directory would still be refused, but as an opaque I/O failure rather than a precise one. And a
/// junction, unlike a file symlink, needs no special privilege to create, so this is the common
/// case, not an edge case. Once the handle is open, its own attribute bits are checked again as a
/// second guard against the path being swapped for a reparse point in the gap between the two
/// lookups.
///
/// Shared by [`verify_model_file`] and [`verify_api_key_file`]: every path preflight is asked to
/// hand the child deserves the same resolution discipline — open once, check that handle, derive
/// the returned path from the handle — rather than the model path getting careful handle-based
/// treatment while another path is resolved by name.
#[cfg(windows)]
fn open_no_follow(path: &Path) -> Result<fs::File, NoFollowOpen> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_OPEN_REPARSE_POINT,
    };

    let pre = fs::symlink_metadata(path).map_err(NoFollowOpen::Io)?;
    if pre.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(NoFollowOpen::ReparsePoint);
    }
    if pre.is_dir() {
        return Err(NoFollowOpen::Directory);
    }

    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(NoFollowOpen::Io)?;

    let meta = file.metadata().map_err(NoFollowOpen::Io)?;
    if meta.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(NoFollowOpen::ReparsePoint);
    }

    Ok(file)
}

/// Validate the API key file and return its canonicalized absolute path.
///
/// A bare size check is not enough: llama.cpp's `--api-key-file` handling (`common/arg.cpp`,
/// checked against the b10751 fixture build) reads the file with `std::getline` splitting on
/// `'\n'`, skipping empty lines and lines starting with `'#'`, and if that leaves zero keys,
/// `server-http.cpp` serves every request with no authentication at all — silently, since an empty
/// key list is not treated as an error there. A file of only blank lines or only `#`-comments
/// therefore has to be refused here, not just a file that is merely short. Parsing is bounded to
/// [`MAX_API_KEY_FILE_BYTES`] read from a single no-follow handle (see [`open_no_follow`]), and
/// every surviving key must be free of control characters: llama.cpp's `getline` keeps a trailing
/// `'\r'` from a CRLF file as part of the key it parses, and no HTTP client can send that byte in a
/// bearer token, so trimming it here would make Cerebro's idea of the key differ from the server's
/// rather than fixing anything.
fn verify_api_key_file(path: &Path) -> Result<PathBuf, PreflightError> {
    #[cfg(windows)]
    let mut file = open_no_follow(path).map_err(|e| match e {
        NoFollowOpen::ReparsePoint => {
            PreflightError::ApiKeyFileIsReparsePoint(path.display().to_string())
        }
        NoFollowOpen::Directory => {
            PreflightError::ApiKeyFileNotRegularFile(path.display().to_string())
        }
        NoFollowOpen::Io(e) => api_key_io_error(path, e),
    })?;
    #[cfg(not(windows))]
    let mut file = fs::File::open(path).map_err(|e| api_key_io_error(path, e))?;

    let meta = file.metadata().map_err(|e| api_key_io_error(path, e))?;
    if !meta.is_file() {
        return Err(PreflightError::ApiKeyFileNotRegularFile(
            path.display().to_string(),
        ));
    }
    if meta.len() > MAX_API_KEY_FILE_BYTES {
        return Err(PreflightError::ApiKeyFileTooLarge {
            path: path.display().to_string(),
            len: meta.len(),
        });
    }

    // `Take` bounds the read even though the size check above already refused anything larger: the
    // same discipline `verify_model_file` applies with its fixed-size `read_exact` magic buffer,
    // not a redundant check on a value this function already trusts.
    let mut text = String::new();
    (&mut file)
        .take(MAX_API_KEY_FILE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|e| api_key_io_error(path, e))?;

    // Mirrors llama.cpp's own `--api-key-file` parser: split on '\n', skip empty lines and lines
    // starting with '#'. Line numbers below are 1-based for a human reading the refusal.
    let mut keys = 0usize;
    for (i, line) in text.split('\n').enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.bytes().any(|b| b.is_ascii_control()) {
            return Err(PreflightError::ApiKeyFileMalformed {
                path: path.display().to_string(),
                line: i + 1,
            });
        }
        if (line.len() as u64) < MIN_API_KEY_FILE_BYTES {
            return Err(PreflightError::ApiKeyTooShort {
                path: path.display().to_string(),
                line: i + 1,
                len: line.len(),
            });
        }
        keys += 1;
    }
    if keys == 0 {
        return Err(PreflightError::ApiKeyFileNoKeys(path.display().to_string()));
    }

    // Resolved from the handle already open above, for the same reason `verify_model_file` resolves
    // its canonical path from a handle rather than re-resolving `path` by name — see
    // `win_path::final_path`'s documentation.
    #[cfg(windows)]
    let canonical = win_path::final_path(&file).map_err(|e| api_key_io_error(path, e))?;
    #[cfg(not(windows))]
    let canonical = fs::canonicalize(path).map_err(|e| api_key_io_error(path, e))?;
    Ok(canonical)
}

/// Build a typed [`PreflightError`] for an API key file I/O failure, distinguishing "not found"
/// from every other cause the way [`model_io_error`] does for a model path.
fn api_key_io_error(path: &Path, e: std::io::Error) -> PreflightError {
    if e.kind() == std::io::ErrorKind::NotFound {
        PreflightError::ApiKeyFileMissing(path.display().to_string())
    } else {
        PreflightError::ApiKeyFileIo {
            path: path.display().to_string(),
            cause: e.to_string(),
        }
    }
}

/// Resolve `request` to a concrete loopback port. See [`PortRequest`]'s documentation for the
/// race this leaves unresolved and who must handle it.
fn resolve_port(request: PortRequest) -> Result<u16, PreflightError> {
    let requested = match request {
        PortRequest::Explicit(0) => return Err(PreflightError::InvalidExplicitPort),
        PortRequest::Explicit(port) => port,
        PortRequest::Any => 0,
    };
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), requested);
    let listener = TcpListener::bind(addr).map_err(|e| match request {
        PortRequest::Explicit(port) => PreflightError::PortUnavailable {
            port,
            cause: e.to_string(),
        },
        PortRequest::Any => PreflightError::PortReservationFailed {
            cause: e.to_string(),
        },
    })?;
    let resolved = listener
        .local_addr()
        .map_err(|e| PreflightError::PortReservationFailed {
            cause: e.to_string(),
        })?
        .port();
    drop(listener); // Release immediately; see the race documented on `PortRequest`.
    Ok(resolved)
}

/// Build the hardened `llama-server` argv for a preflighted launch.
///
/// This function is the only bridge from a [`Preflighted`] proof to real argv: every path flag
/// preflight verified — `--host`, `--port`, `--api-key-file`, `-m`, and `-md` when a draft model was
/// given — is emitted here, from the verified value, and nowhere else. `builder` must not already
/// carry any of those flags: `ArgvBuilder::push`'s own duplicate check only catches a *second* push
/// of the same flag, so a builder pre-loaded with, say, `Flag::DraftModel` pointing at a path
/// preflight never saw would sail through unnoticed (that path would have skipped the reparse,
/// magic and comma checks entirely). Refusing any pre-set occurrence up front closes that gap.
///
/// Takes `builder` by value rather than by `&mut` reference: a builder a caller still owns after a
/// failed call would already carry whatever this function managed to push before failing, so any
/// retry with the same builder would fail on [`cerebro_flags::EmitError::Duplicate`] instead of the
/// real problem. Consuming it makes that state impossible to observe.
///
/// Refuses a `builder` that is not [`Dialect::Server`] before touching it: [`ArgvBuilder::harden_server`]
/// only `debug_assert!`s this, which compiles out of a release build, and every flag this function
/// adds is server-only.
///
/// Does **not** scrub inherited `LLAMA_ARG_*` environment variables — see this module's top-level
/// documentation for why that is the spawn task's responsibility, not this pure function's.
pub fn build_hardened_server_argv(
    preflighted: &Preflighted,
    mut builder: ArgvBuilder,
) -> Result<Vec<String>, EmitError> {
    if builder.dialect() != Dialect::Server {
        return Err(EmitError::WrongDialect {
            expected: Dialect::Server.name(),
            actual: builder.dialect().name(),
        });
    }

    // Every one of these flags is about to be emitted from a value `preflight` verified; a caller
    // that pre-set any of them could only be supplying a value preflight never checked, which is
    // exactly the divergence this function exists to close off.
    if let Some(preset) = builder.flags().iter().find(|f| {
        matches!(
            f,
            Flag::Model(_)
                | Flag::DraftModel(_)
                | Flag::ApiKeyFile(_)
                | Flag::Host(_)
                | Flag::Port(_)
        )
    }) {
        return Err(EmitError::PreflightOwned(preset.name()));
    }

    let key_file = argv_path(preflighted.api_key_file())?;
    builder.harden_server(LOOPBACK_HOST, preflighted.port(), &key_file)?;
    builder.push(Flag::Model(argv_path(preflighted.model_path())?))?;
    if let Some(draft) = preflighted.draft_model_path() {
        builder.push(Flag::DraftModel(argv_path(draft)?))?;
    }
    builder.build()
}

/// Render a path as an argv-safe UTF-8 string.
fn argv_path(path: &Path) -> Result<String, EmitError> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| EmitError::NonUtf8Path(path.display().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cerebro_probe::{AuditedObject, GrantVia, WritableGrant};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A uniquely named temp directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "cerebro-supervisor-{tag}-{}-{nanos}-{n}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("a name unique per test, run, and process must not already exist");
            TempDir(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct FakeElevation(bool);

    impl ElevationCheck for FakeElevation {
        fn is_elevated(&self) -> Result<bool, PreflightError> {
            Ok(self.0)
        }
    }

    /// An [`ElevationCheck`] that cannot complete at all, for proving the fail-closed branch: an
    /// `Err` from the elevation check must stop the launch exactly as `Ok(true)` does, not be
    /// treated as "probably not elevated".
    struct FailingElevation;

    impl ElevationCheck for FailingElevation {
        fn is_elevated(&self) -> Result<bool, PreflightError> {
            Err(PreflightError::ElevationCheckFailed(
                "OpenProcessToken failed with Win32 error 5".to_string(),
            ))
        }
    }

    /// A [`DaclAuditor`] returning a fixed, caller-chosen [`DaclAudit`], so DACL tests never depend
    /// on the ambient ACL of wherever a fixture directory happens to be created — this development
    /// machine's own `%TEMP%` is itself writable by an untrusted principal.
    struct FakeDacl(DaclAudit);

    impl DaclAuditor for FakeDacl {
        fn audit(&self, _dir: &Path) -> DaclAudit {
            self.0.clone()
        }
    }

    /// [`preflight_with`] with both host checks faked, for tests that only care about the checks
    /// downstream of elevation and the DACL audit.
    fn preflight_for_test(
        spec: &LaunchSpec,
        elevated: bool,
        dacl: DaclAudit,
    ) -> Result<Preflighted, PreflightError> {
        let elevation = FakeElevation(elevated);
        let dacl = FakeDacl(dacl);
        preflight_with(
            spec,
            &HostChecks {
                elevation: &elevation,
                dacl: &dacl,
            },
        )
    }

    /// A backend directory with one fake tool and one fake library, plus the manifest that
    /// exactly matches it.
    struct Backend {
        dir: TempDir,
        manifest: BTreeMap<String, String>,
    }

    fn make_backend(tag: &str) -> Backend {
        let dir = TempDir::new(tag);
        fs::write(dir.0.join("llama-server.exe"), b"fake server exe").unwrap();
        fs::write(dir.0.join("ggml-cpu.dll"), b"fake cpu backend").unwrap();
        let (manifest, _) = cerebro_probe::backend::hash_directory(&dir.0).unwrap();
        Backend { dir, manifest }
    }

    fn make_model(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut bytes = GGUF_MAGIC.to_vec();
        bytes.extend_from_slice(b"padding-so-this-looks-like-a-real-file");
        fs::write(&path, bytes).unwrap();
        path
    }

    fn make_api_key(dir: &Path) -> PathBuf {
        let path = dir.join("api.key");
        fs::write(&path, "k".repeat(32)).unwrap();
        path
    }

    /// A base spec with no DACL opt-in; every test that does not care about the DACL check runs it
    /// through [`preflight_for_test`] with `DaclAudit::OwnerOnly`, so the DACL check itself never
    /// gets in the way of what the test actually exercises.
    fn base_spec(backend: &Backend, model_dir: &Path) -> LaunchSpec {
        LaunchSpec {
            backend_dir: backend.dir.0.clone(),
            tool_name: "llama-server.exe".to_string(),
            manifest: backend.manifest.clone(),
            model_path: make_model(model_dir, "target.gguf"),
            draft_model_path: None,
            port: PortRequest::Any,
            api_key_file: make_api_key(model_dir),
            allow_unsafe_dacl: None,
        }
    }

    /// A [`WritableGrant`] naming one untrusted principal on the backend directory, for the DACL
    /// tests that need a concrete [`DaclAudit::WritableByOthers`] value.
    fn writable_grant(dir: &Path, principal: &str) -> WritableGrant {
        WritableGrant::test_new(
            dir.to_path_buf(),
            AuditedObject::Directory,
            principal,
            "S-1-5-21-1-2-3-1000",
            0x2,
            vec!["FILE_ADD_FILE"],
            GrantVia::Ace,
        )
    }

    #[test]
    fn a_fully_valid_spec_preflights_successfully() {
        let backend = make_backend("valid");
        let model_dir = TempDir::new("valid-model");
        let spec = base_spec(&backend, &model_dir.0);

        let result = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).expect("must succeed");
        assert_ne!(result.port(), 0);
        assert!(result.model_path().is_absolute());
        assert_eq!(result.backend_hashes(), &backend.manifest);
        assert!(result.warnings().is_empty());
    }

    #[test]
    fn elevated_process_is_refused_before_touching_the_filesystem() {
        // Backend/model paths are deliberately bogus: elevation must be checked first and must
        // short-circuit everything else.
        let spec = LaunchSpec {
            backend_dir: PathBuf::from("does-not-exist"),
            tool_name: "llama-server.exe".to_string(),
            manifest: BTreeMap::new(),
            model_path: PathBuf::from("does-not-exist.gguf"),
            draft_model_path: None,
            port: PortRequest::Any,
            api_key_file: PathBuf::from("does-not-exist.key"),
            allow_unsafe_dacl: None,
        };
        assert_eq!(
            preflight_for_test(&spec, true, DaclAudit::OwnerOnly).unwrap_err(),
            PreflightError::RunningElevated
        );
    }

    #[test]
    fn an_elevation_check_that_cannot_complete_refuses_launch() {
        // Same bogus, nonexistent paths as `elevated_process_is_refused_before_touching_the_filesystem`:
        // an elevation check that returns `Err` must short-circuit before any of them are touched,
        // exactly as `Ok(true)` does.
        let spec = LaunchSpec {
            backend_dir: PathBuf::from("does-not-exist"),
            tool_name: "llama-server.exe".to_string(),
            manifest: BTreeMap::new(),
            model_path: PathBuf::from("does-not-exist.gguf"),
            draft_model_path: None,
            port: PortRequest::Any,
            api_key_file: PathBuf::from("does-not-exist.key"),
            allow_unsafe_dacl: None,
        };
        let elevation = FailingElevation;
        let dacl = FakeDacl(DaclAudit::OwnerOnly);
        let err = preflight_with(
            &spec,
            &HostChecks {
                elevation: &elevation,
                dacl: &dacl,
            },
        )
        .unwrap_err();
        assert!(matches!(err, PreflightError::ElevationCheckFailed(_)), "{err:?}");
    }

    #[test]
    fn relative_backend_dir_is_refused_before_hashing() {
        let spec = LaunchSpec {
            backend_dir: PathBuf::from("backend"),
            tool_name: "llama-server.exe".to_string(),
            manifest: BTreeMap::new(),
            model_path: PathBuf::from("does-not-exist.gguf"),
            draft_model_path: None,
            port: PortRequest::Any,
            api_key_file: PathBuf::from("does-not-exist.key"),
            allow_unsafe_dacl: None,
        };
        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::BackendDirNotAbsolute(dir)) if dir == "backend"
        ));
    }

    #[test]
    fn an_unmanifested_backend_file_is_refused() {
        let backend = make_backend("extra-dll");
        // Plant a file the manifest was computed before, so it is a real, unlisted extra.
        fs::write(backend.dir.0.join("ggml-cuda.dll"), b"planted").unwrap();
        let model_dir = TempDir::new("extra-dll-model");
        let spec = base_spec(&backend, &model_dir.0);

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::BackendFileUnexpected { ref file, .. } if file == "ggml-cuda.dll"),
            "{err:?}"
        );
    }

    #[test]
    fn a_manifest_entry_missing_from_disk_is_refused() {
        let backend = make_backend("missing-file");
        let model_dir = TempDir::new("missing-file-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.manifest
            .insert("ggml-vulkan.dll".to_string(), "0".repeat(64));

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::BackendFileMissing { ref file, .. } if file == "ggml-vulkan.dll"),
            "{err:?}"
        );
    }

    #[test]
    fn a_tampered_manifest_hash_is_refused() {
        let backend = make_backend("hash-mismatch");
        let model_dir = TempDir::new("hash-mismatch-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.manifest
            .insert("llama-server.exe".to_string(), "f".repeat(64));

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::BackendFileHashMismatch { ref file, .. } if file == "llama-server.exe"),
            "{err:?}"
        );
    }

    #[test]
    fn case_colliding_manifest_is_refused() {
        let backend = make_backend("case-collision");
        let model_dir = TempDir::new("case-collision-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.manifest
            .insert("GGML-CPU.DLL".to_string(), "0".repeat(64));

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ManifestNameCollision { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn malformed_manifest_hash_is_refused_distinctly() {
        // A manifest value that is not 64 hex digits is a defect in the pinned manifest, not
        // evidence a binary changed, and must be reported as such rather than as a hash mismatch.
        let backend = make_backend("malformed-hash");
        let model_dir = TempDir::new("malformed-hash-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.manifest
            .insert("llama-server.exe".to_string(), "abc".to_string());

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ManifestHashMalformed { ref value, .. } if value == "abc"),
            "{err:?}"
        );
    }

    #[test]
    fn unsafe_dacl_is_refused_before_hashing() {
        // A backend directory with an unsafe DACL and no opt-in must be refused as
        // `BackendDirUnsafe`, never `BackendFileHashMismatch` — proving the DACL audit really does
        // run before the (expensive) hash comparison, not just that both eventually refuse.
        let backend = make_backend("dacl-before-hash");
        let model_dir = TempDir::new("dacl-before-hash-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.manifest
            .insert("llama-server.exe".to_string(), "f".repeat(64));
        let grant = writable_grant(&backend.dir.0, r"EVIL\user");

        let err =
            preflight_for_test(&spec, false, DaclAudit::WritableByOthers(vec![grant])).unwrap_err();
        assert!(
            matches!(err, PreflightError::BackendDirUnsafe { .. }),
            "expected the DACL audit to run before hashing and refuse first, got {err:?}"
        );
    }

    /// A thin wrapper over `crate::names::BAD_NAMES`, the one table shared with
    /// `spawn::verify_spawn_target`'s own test of the same shape — see `crate::names`'s module docs
    /// on why the predicate itself is not duplicated. This asserts only that `verify_tool_name` maps
    /// a rejection to *its* typed error, `PreflightError::BackendToolEscapesDir`; the
    /// full-pipeline tests below cover a couple of these shapes again to prove the wiring through
    /// `preflight_for_test`, not to re-derive the predicate.
    #[test]
    fn verify_tool_name_rejects_every_shape_in_the_shared_bad_name_table() {
        for bad in crate::names::BAD_NAMES {
            let err = verify_tool_name(bad).expect_err(&format!("{bad:?} should have been refused"));
            assert!(
                matches!(err, PreflightError::BackendToolEscapesDir(_)),
                "{bad:?}: {err:?}"
            );
        }
    }

    #[test]
    fn tool_name_may_not_escape_the_backend_directory() {
        let backend = make_backend("tool-escape");
        let model_dir = TempDir::new("tool-escape-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.tool_name = "../evil.exe".to_string();

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::BackendToolEscapesDir(_))
        ));
    }

    #[test]
    fn tool_name_dot_dot_and_dll_are_refused() {
        let backend = make_backend("tool-shapes");
        let model_dir = TempDir::new("tool-shapes-model");
        for bad in ["..", "ggml-cpu.dll", "llama-server.exe."] {
            let mut spec = base_spec(&backend, &model_dir.0);
            spec.tool_name = bad.to_string();
            assert!(
                matches!(
                    preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
                    Err(PreflightError::BackendToolEscapesDir(_))
                ),
                "{bad:?} should have been refused"
            );
        }
    }

    #[test]
    fn tool_name_must_be_one_of_the_verified_files() {
        let backend = make_backend("tool-unverified");
        let model_dir = TempDir::new("tool-unverified-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.tool_name = "llama-bench.exe".to_string();

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::BackendToolNotVerified { .. })
        ));
    }

    #[test]
    fn manifest_match_is_case_insensitive_on_disk_names() {
        let dir = TempDir::new("case-insensitive-backend");
        fs::write(dir.0.join("LLAMA-SERVER.EXE"), b"fake server exe").unwrap();
        let (mut computed, _) = cerebro_probe::backend::hash_directory(&dir.0).unwrap();
        let lower_hash = computed
            .remove("LLAMA-SERVER.EXE")
            .expect("hash_directory must have hashed the file it just found");

        // The manifest key is lowercase and the hash is uppercase hex, both differing from what is
        // actually on disk: the match must still succeed by folding case on both sides.
        let mut manifest = BTreeMap::new();
        manifest.insert("llama-server.exe".to_string(), lower_hash.to_ascii_uppercase());

        let model_dir = TempDir::new("case-insensitive-model");
        let spec = LaunchSpec {
            backend_dir: dir.0.clone(),
            tool_name: "llama-server.exe".to_string(),
            manifest,
            model_path: make_model(&model_dir.0, "target.gguf"),
            draft_model_path: None,
            port: PortRequest::Any,
            api_key_file: make_api_key(&model_dir.0),
            allow_unsafe_dacl: None,
        };

        let result = preflight_for_test(&spec, false, DaclAudit::OwnerOnly)
            .expect("a manifest key differing only in case, with an uppercase-hex hash, must match");
        assert_eq!(
            result.backend_hashes().get("LLAMA-SERVER.EXE"),
            Some(&lower_hash),
            "backend_hashes must keep the on-disk spelling and lowercase hex: {:?}",
            result.backend_hashes()
        );
    }

    #[test]
    fn non_backend_manifest_entry_is_missing() {
        let backend = make_backend("non-backend-entry");
        let model_dir = TempDir::new("non-backend-entry-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        // `.txt` is neither `.exe` nor `.dll`, so `hash_directory` never looks at it — whether or
        // not it exists on disk is irrelevant to it. A manifest naming it can therefore never be
        // satisfied; this documents that only `.exe`/`.dll` are hashed at all.
        fs::write(backend.dir.0.join("readme.txt"), b"notes").unwrap();
        spec.manifest.insert("readme.txt".to_string(), "0".repeat(64));

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::BackendFileMissing { ref file, .. } if file == "readme.txt"),
            "{err:?}"
        );
    }

    #[test]
    fn a_model_with_bad_magic_is_refused() {
        let backend = make_backend("bad-magic");
        let model_dir = TempDir::new("bad-magic-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let bad = model_dir.0.join("not-gguf.bin");
        fs::write(&bad, b"NOPE this is not a gguf file").unwrap();
        spec.model_path = bad;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelBadMagic { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn truncated_model_under_four_bytes_is_bad_magic() {
        let backend = make_backend("truncated-magic");
        let model_dir = TempDir::new("truncated-magic-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let short = model_dir.0.join("short.gguf");
        fs::write(&short, b"GG").unwrap(); // Fewer than 4 bytes: read_exact hits UnexpectedEof.
        spec.model_path = short;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelBadMagic { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn draft_model_bad_magic_reports_draft_role() {
        let backend = make_backend("draft-bad-magic");
        let model_dir = TempDir::new("draft-bad-magic-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let bad = model_dir.0.join("draft-not-gguf.bin");
        fs::write(&bad, b"NOPE not a gguf file either").unwrap();
        spec.draft_model_path = Some(bad);

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelBadMagic { role: ModelRole::Draft, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_model_path_containing_a_comma_is_refused() {
        let backend = make_backend("comma-path");
        let model_dir = TempDir::new("comma-path-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.model_path = make_model(&model_dir.0, "target,evil.gguf");

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelPathHasComma { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn draft_model_with_comma_reports_draft_role() {
        let backend = make_backend("draft-comma-path");
        let model_dir = TempDir::new("draft-comma-path-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.draft_model_path = Some(make_model(&model_dir.0, "draft,evil.gguf"));

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelPathHasComma { role: ModelRole::Draft, .. }),
            "{err:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn canonical_model_path_is_from_handle() {
        // Regression guard for the switch away from `fs::canonicalize(path)` (see `win_path`'s
        // documentation for why that call was unsafe here): the path `Preflighted` reports must be
        // exactly what resolving the *same, already-verified* handle produces.
        let backend = make_backend("canonical-from-handle");
        let model_dir = TempDir::new("canonical-from-handle-model");
        let spec = base_spec(&backend, &model_dir.0);

        let result = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        let file = fs::File::open(result.model_path()).expect("reopen the resolved path");
        let expected = win_path::final_path(&file).expect("resolve final path from the reopened handle");
        assert_eq!(result.model_path(), expected);
        assert!(!result.model_path().to_string_lossy().contains(','));
    }

    #[cfg(windows)]
    #[test]
    fn comma_introduced_by_parent_junction_is_refused() {
        // `path` itself carries no comma; only the *canonical* form, reached through a junction,
        // does. Only the second `reject_comma` call inside `verify_model_file` (the one run on the
        // canonicalized path) can catch this.
        let root = TempDir::new("comma-junction-root");
        let real_dir = root.0.join("a,b");
        fs::create_dir(&real_dir).unwrap();
        let model = make_model(&real_dir, "model.gguf");

        let link = root.0.join("link");
        // `cmd /c mklink /J` cannot be used here: cmd's own command-line parser treats a comma as
        // an argument separator (the same legacy behavior that treats `;` and `=` that way for
        // internal commands), so it splits `...\a,b` into two arguments and refuses the whole
        // command with "The syntax of the command is incorrect" before `mklink` ever runs.
        // PowerShell's `New-Item` takes each path as one single-quoted string literal instead, so
        // the comma never reaches any tokenizer. The whole invocation must be one `-Command`
        // argument: passing `-ItemType`, `Junction`, etc. as separate argv entries makes
        // powershell.exe's own CLI parser collect them into an array bound to `-Path` instead of
        // running them as a command.
        let script = format!(
            "New-Item -ItemType Junction -Path '{}' -Target '{}' | Out-Null",
            link.display(),
            real_dir.display()
        );
        let status = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .status()
            .expect("run New-Item -ItemType Junction");
        assert!(status.success(), "creating a junction needs no special privilege");

        let backend = make_backend("comma-junction");
        let mut spec = base_spec(&backend, &root.0);
        spec.model_path = link.join(model.file_name().unwrap());

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelPathHasComma { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_directory_passed_as_the_model_is_refused() {
        let backend = make_backend("dir-as-model");
        let model_dir = TempDir::new("dir-as-model-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let sub = model_dir.0.join("looks-like-a-model.gguf");
        fs::create_dir(&sub).unwrap();
        spec.model_path = sub;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelNotRegularFile { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_missing_model_is_refused() {
        let backend = make_backend("missing-model");
        let model_dir = TempDir::new("missing-model-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.model_path = model_dir.0.join("nope.gguf");

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelNotFound { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_junction_passed_as_the_model_is_a_reparse_point() {
        // A directory junction, unlike a file symlink, needs no privilege to create on Windows —
        // any account can make one. This is therefore the reparse-point regression test that always
        // runs (see `a_symlinked_model_is_refused_when_symlinks_are_creatable` below for the
        // privilege-gated file-symlink variant), and it is also the shape that was getting the
        // wrong typed error before `verify_model_file` classified reparse points from
        // `symlink_metadata` up front: opening a junction without `FILE_FLAG_BACKUP_SEMANTICS`
        // fails with `ERROR_ACCESS_DENIED`, which used to surface as `ModelPathIo`.
        let backend = make_backend("model-junction");
        let root = TempDir::new("model-junction-root");
        let target = root.0.join("real");
        fs::create_dir(&target).unwrap();

        // The junction itself is the model path — it is a directory reparse point in its own
        // right, which is what `symlink_metadata` on `path` must classify before anything opens
        // it. (Pointing a model path *through* a junction at a plain file inside it, as
        // `comma_introduced_by_parent_junction_is_refused` does, is a different case: there the
        // final component is an ordinary file, and only the earlier `reject_comma` check on the
        // canonicalized path can see through the junction.)
        let link = root.0.join("junction.gguf");
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

        let mut spec = base_spec(&backend, &root.0);
        spec.model_path = link;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelIsReparsePoint { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "needs Developer Mode or SeCreateSymbolicLinkPrivilege to create a file symlink; the \
                junction test above exercises the same ModelIsReparsePoint code path without that \
                requirement, and CI must fail loudly, not silently pass with nothing checked, if \
                this one is ever run without that privilege"]
    fn a_symlinked_model_is_refused_when_symlinks_are_creatable() {
        let backend = make_backend("model-symlink");
        let model_dir = TempDir::new("model-symlink-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let target = model_dir.0.join("real.gguf");
        fs::write(&target, GGUF_MAGIC).unwrap();
        let link = model_dir.0.join("link.gguf");
        std::os::windows::fs::symlink_file(&target, &link)
            .expect("this test is #[ignore]d specifically for the token that cannot do this");
        spec.model_path = link;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ModelIsReparsePoint { role: ModelRole::Target, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_missing_api_key_file_is_refused() {
        let backend = make_backend("missing-key");
        let model_dir = TempDir::new("missing-key-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.api_key_file = model_dir.0.join("nope.key");

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::ApiKeyFileMissing(_))
        ));
    }

    #[test]
    fn a_too_short_api_key_file_is_refused() {
        let backend = make_backend("short-key");
        let model_dir = TempDir::new("short-key-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let short = model_dir.0.join("short.key");
        fs::write(&short, "short").unwrap();
        spec.api_key_file = short;

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::ApiKeyTooShort { line: 1, .. })
        ));
    }

    #[test]
    fn api_key_file_of_only_blank_lines_is_refused() {
        // 40 bytes, all newlines: long enough to pass a bare size check, but every line is empty
        // once split, so llama.cpp's own parser would keep zero keys.
        let backend = make_backend("key-blank-lines");
        let model_dir = TempDir::new("key-blank-lines-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let path = model_dir.0.join("blank.key");
        fs::write(&path, "\n".repeat(40)).unwrap();
        spec.api_key_file = path;

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::ApiKeyFileNoKeys(_))
        ));
    }

    #[test]
    fn api_key_file_of_only_comments_is_refused() {
        let backend = make_backend("key-only-comments");
        let model_dir = TempDir::new("key-only-comments-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let path = model_dir.0.join("comment.key");
        fs::write(&path, "# put the real key here, at least 32 chars long\n").unwrap();
        spec.api_key_file = path;

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::ApiKeyFileNoKeys(_))
        ));
    }

    #[test]
    fn api_key_file_with_short_key_and_padding_is_refused() {
        let backend = make_backend("key-short-plus-padding");
        let model_dir = TempDir::new("key-short-plus-padding-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let path = model_dir.0.join("short-plus-padding.key");
        // fs::write, not a shell redirect: PowerShell's Set-Content would add CRLF line endings and
        // silently change what this fixture is testing.
        fs::write(&path, format!("abc\n{}\n", "#".repeat(40))).unwrap();
        spec.api_key_file = path;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ApiKeyTooShort { line: 1, len: 3, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn api_key_file_with_crlf_is_refused() {
        let backend = make_backend("key-crlf");
        let model_dir = TempDir::new("key-crlf-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let path = model_dir.0.join("crlf.key");
        // Built with `fs::write` from an explicit `\r\n`, never PowerShell `Set-Content`, so the
        // fixture carries exactly one CRLF and nothing the shell added on its own.
        fs::write(&path, format!("{}\r\n", "k".repeat(40))).unwrap();
        spec.api_key_file = path;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ApiKeyFileMalformed { line: 1, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn api_key_file_oversized_is_refused() {
        let backend = make_backend("key-oversized");
        let model_dir = TempDir::new("key-oversized-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let path = model_dir.0.join("oversized.key");
        fs::write(&path, "k".repeat(65 * 1024)).unwrap();
        spec.api_key_file = path;

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::ApiKeyFileTooLarge { .. })
        ));
    }

    #[cfg(windows)]
    #[test]
    fn api_key_junction_is_refused() {
        // A directory junction, unlike a file symlink, needs no privilege to create on Windows —
        // mirrors `a_junction_passed_as_the_model_is_a_reparse_point` above for the API key path.
        let backend = make_backend("key-junction");
        let root = TempDir::new("key-junction-root");
        let target = root.0.join("real");
        fs::create_dir(&target).unwrap();

        let link = root.0.join("junction.key");
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

        let mut spec = base_spec(&backend, &root.0);
        spec.api_key_file = link;

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::ApiKeyFileIsReparsePoint(_)),
            "{err:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn api_key_path_is_resolved_from_handle() {
        let backend = make_backend("key-handle-resolved");
        let model_dir = TempDir::new("key-handle-resolved-model");
        let spec = base_spec(&backend, &model_dir.0);

        let result = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        let file = fs::File::open(result.api_key_file()).expect("reopen the resolved path");
        let expected =
            win_path::final_path(&file).expect("resolve final path from the reopened handle");
        assert_eq!(result.api_key_file(), expected);
    }

    #[test]
    fn api_key_directory_is_not_regular_file() {
        let backend = make_backend("key-is-dir");
        let model_dir = TempDir::new("key-is-dir-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        let dir_as_key = model_dir.0.join("key-dir");
        fs::create_dir(&dir_as_key).unwrap();
        spec.api_key_file = dir_as_key;

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::ApiKeyFileNotRegularFile(_))
        ));
    }

    #[test]
    fn an_explicit_port_already_bound_is_refused() {
        let backend = make_backend("port-taken");
        let model_dir = TempDir::new("port-taken-model");
        let mut spec = base_spec(&backend, &model_dir.0);

        let holder = TcpListener::bind((LOOPBACK_HOST, 0)).unwrap();
        let taken_port = holder.local_addr().unwrap().port();
        spec.port = PortRequest::Explicit(taken_port);

        let err = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap_err();
        assert!(
            matches!(err, PreflightError::PortUnavailable { port, .. } if port == taken_port),
            "{err:?}"
        );
        drop(holder);
    }

    #[test]
    fn explicit_port_zero_is_rejected() {
        let backend = make_backend("port-zero");
        let model_dir = TempDir::new("port-zero-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.port = PortRequest::Explicit(0);

        assert!(matches!(
            preflight_for_test(&spec, false, DaclAudit::OwnerOnly),
            Err(PreflightError::InvalidExplicitPort)
        ));
    }

    #[test]
    fn owner_only_dacl_yields_success_with_no_warnings() {
        let backend = make_backend("dacl-owner-only");
        let model_dir = TempDir::new("dacl-owner-only-model");
        let spec = base_spec(&backend, &model_dir.0);

        let result = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).expect("must succeed");
        assert!(result.warnings().is_empty(), "{:?}", result.warnings());
    }

    #[test]
    fn writable_dacl_without_opt_in_is_refused_with_structured_principals() {
        let backend = make_backend("dacl-refuse");
        let model_dir = TempDir::new("dacl-refuse-model");
        let spec = base_spec(&backend, &model_dir.0);
        let grant = writable_grant(&backend.dir.0, r"EVIL\user");

        let err = preflight_for_test(&spec, false, DaclAudit::WritableByOthers(vec![grant])).unwrap_err();
        let PreflightError::BackendDirUnsafe { audit, .. } = &err else {
            panic!("expected BackendDirUnsafe, got {err:?}");
        };
        let UnsafeDacl::WritableByOthers(grants) = audit else {
            panic!("expected WritableByOthers, got {audit:?}");
        };
        assert!(
            grants.iter().any(|g| g.principal() == r"EVIL\user"),
            "{grants:?}"
        );
    }

    #[test]
    fn writable_dacl_with_opt_in_warns_naming_principals() {
        let backend = make_backend("dacl-allow");
        let model_dir = TempDir::new("dacl-allow-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        const REASON: &str = "known-unsafe fixture, accepted for this test only";
        spec.allow_unsafe_dacl = Some(AllowUnsafeDacl::new(REASON).unwrap());
        let grant = writable_grant(&backend.dir.0, r"EVIL\user");

        let result = preflight_for_test(&spec, false, DaclAudit::WritableByOthers(vec![grant]))
            .expect("must succeed with the opt-in");
        assert_eq!(
            result.warnings(),
            &[PreflightWarning::UnsafeDaclAccepted {
                reason: REASON.to_string(),
                principals: vec![r"EVIL\user".to_string()],
                detail: UnsafeDacl::WritableByOthers(vec![writable_grant(&backend.dir.0, r"EVIL\user")])
                    .to_string(),
            }]
        );
    }

    #[test]
    fn unsafe_dacl_warning_mentions_integrity_window() {
        // The manifest match preflight just ran only holds as of preflight time when the DACL is
        // unsafe — the named principals can still swap a binary before the child loads it. The
        // warning text is the only place that caveat reaches a human reading a log, so it must say
        // so, not just name the principals.
        let backend = make_backend("dacl-allow-warning-text");
        let model_dir = TempDir::new("dacl-allow-warning-text-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.allow_unsafe_dacl =
            Some(AllowUnsafeDacl::new("known-unsafe fixture, accepted for this test only").unwrap());
        let grant = writable_grant(&backend.dir.0, r"EVIL\user");

        let result = preflight_for_test(&spec, false, DaclAudit::WritableByOthers(vec![grant]))
            .expect("must succeed with the opt-in");
        let warning = &result.warnings()[0];
        assert!(
            warning.to_string().contains("only verified as of preflight"),
            "{warning}"
        );
    }

    #[test]
    fn unknown_dacl_is_refused_even_with_opt_in() {
        let backend = make_backend("dacl-unknown");
        let model_dir = TempDir::new("dacl-unknown-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.allow_unsafe_dacl = Some(AllowUnsafeDacl::new("would accept a known risk, but this one is unknown").unwrap());

        let err = preflight_for_test(
            &spec,
            false,
            DaclAudit::Unknown("audit could not complete".to_string()),
        )
        .unwrap_err();
        assert!(
            matches!(err, PreflightError::BackendDirUnauditable { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn blank_unsafe_dacl_reason_is_rejected() {
        assert!(matches!(
            AllowUnsafeDacl::new("   "),
            Err(PreflightError::UnsafeDaclReasonMissing)
        ));
        assert!(matches!(
            AllowUnsafeDacl::new(""),
            Err(PreflightError::UnsafeDaclReasonMissing)
        ));
        assert!(AllowUnsafeDacl::new("a real reason").is_ok());
    }

    /// Locate the value immediately following `flag` in `argv`, asserting `flag` appears exactly
    /// once. Replaces substring-on-the-joined-argv assertions, which cannot tell a flag's own value
    /// from a value that merely happens to contain the same text elsewhere in the argv.
    fn value_after<'a>(argv: &'a [String], flag: &str) -> &'a str {
        let positions: Vec<usize> = argv
            .iter()
            .enumerate()
            .filter(|(_, a)| a.as_str() == flag)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions.len(), 1, "expected exactly one {flag:?} in {argv:?}");
        argv.get(positions[0] + 1)
            .unwrap_or_else(|| panic!("{flag:?} at the end of argv with no value: {argv:?}"))
    }

    #[test]
    fn argv_pairs_are_adjacent() {
        let backend = make_backend("argv");
        let model_dir = TempDir::new("argv-model");
        let spec = base_spec(&backend, &model_dir.0);
        let preflighted = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        let builder = ArgvBuilder::new(Dialect::Server);
        let argv = build_hardened_server_argv(&preflighted, builder).unwrap();

        assert_eq!(value_after(&argv, "--host"), LOOPBACK_HOST);
        assert_eq!(value_after(&argv, "--port"), preflighted.port().to_string());
        assert_eq!(
            value_after(&argv, "-m"),
            preflighted.model_path().to_str().unwrap()
        );
        assert_eq!(
            value_after(&argv, "--api-key-file"),
            preflighted.api_key_file().to_str().unwrap()
        );
        assert!(argv.iter().any(|a| a == "--no-slots"), "{argv:?}");
        assert!(argv.iter().any(|a| a == "--no-webui"), "{argv:?}");
        assert!(argv.iter().any(|a| a == "--cors-origins"), "{argv:?}");
    }

    #[test]
    fn argv_emits_verified_draft_model() {
        let backend = make_backend("argv-draft");
        let model_dir = TempDir::new("argv-draft-model");
        let mut spec = base_spec(&backend, &model_dir.0);
        spec.draft_model_path = Some(make_model(&model_dir.0, "draft.gguf"));
        let preflighted = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        // A draft model with no speculation strategy selected is itself a cross-flag violation
        // (`draft_flags_without_draft_strategy`) that `ArgvBuilder::build` rightly refuses — that
        // rule is about the caller's own flags, not about anything preflight verifies, so the
        // caller (not `build_hardened_server_argv`) is the one that must set it.
        let mut builder = ArgvBuilder::new(Dialect::Server);
        builder
            .push(Flag::SpecType(cerebro_flags::value::SpecType::DraftSimple))
            .unwrap();
        let argv = build_hardened_server_argv(&preflighted, builder).unwrap();

        assert_eq!(
            value_after(&argv, "-md"),
            preflighted
                .draft_model_path()
                .expect("spec set a draft model")
                .to_str()
                .unwrap()
        );
    }

    #[test]
    fn argv_refuses_builder_with_preset_draft_model() {
        let backend = make_backend("argv-preset-draft");
        let model_dir = TempDir::new("argv-preset-draft-model");
        let spec = base_spec(&backend, &model_dir.0);
        let preflighted = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        // Not the same path preflight ever verified — exactly the divergence the refusal exists to
        // prevent, and it would have sailed straight through the old "add flags to whatever the
        // caller already built" implementation.
        let mut builder = ArgvBuilder::new(Dialect::Server);
        builder.push(Flag::DraftModel(r"C:\x,evil.gguf".to_string())).unwrap();

        assert_eq!(
            build_hardened_server_argv(&preflighted, builder),
            Err(EmitError::PreflightOwned("-md"))
        );
    }

    #[test]
    fn argv_refuses_builder_with_preset_api_key_file_or_host() {
        let backend = make_backend("argv-preset-other");
        let model_dir = TempDir::new("argv-preset-other-model");
        let spec = base_spec(&backend, &model_dir.0);
        let preflighted = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        for (flag, name) in [
            (Flag::ApiKeyFile(r"C:\other.key".to_string()), "--api-key-file"),
            (Flag::Host("0.0.0.0".to_string()), "--host"),
        ] {
            let mut builder = ArgvBuilder::new(Dialect::Server);
            builder.push(flag).unwrap();
            assert_eq!(
                build_hardened_server_argv(&preflighted, builder),
                Err(EmitError::PreflightOwned(name)),
                "{name} must be refused as PreflightOwned, not silently accepted or reported as \
                 an ordinary Duplicate"
            );
        }
    }

    #[test]
    fn argv_builder_with_bench_dialect_is_rejected() {
        let backend = make_backend("argv-bench");
        let model_dir = TempDir::new("argv-bench-model");
        let spec = base_spec(&backend, &model_dir.0);
        let preflighted = preflight_for_test(&spec, false, DaclAudit::OwnerOnly).unwrap();

        let builder = ArgvBuilder::new(Dialect::Bench);
        assert!(matches!(
            build_hardened_server_argv(&preflighted, builder),
            Err(EmitError::WrongDialect { .. })
        ));
    }

    #[test]
    #[ignore = "requires the development machine's llama.cpp b10751 build, the tiny GGUF fixture, \
                and an unelevated test process; the build's directory ACL grants Modify to \
                ASUS\\CodexSandboxUsers, which is exactly the refusal this test exercises. Calls \
                the public preflight() (real ProcessTokenElevation and ProbeDaclAuditor both), so \
                it must be run from a non-elevated shell or the refusal branch would report \
                RunningElevated instead of the DACL refusal this test is about"]
    fn live_b10751_refuses_without_opt_in_and_succeeds_with_it() {
        let dir = Path::new("C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751");
        let (manifest, _) =
            cerebro_probe::backend::hash_directory(dir).expect("hash the real backend dir");
        let model = Path::new(
            "C:/Users/noahm/.lmstudio/models/skiesoffire/bastion-control/bastion-control-0.5b-f16.gguf",
        );

        let key_dir = TempDir::new("live-b10751-key");
        let key_path = key_dir.0.join("api.key");
        fs::write(&key_path, "k".repeat(32)).unwrap();

        let base = LaunchSpec {
            backend_dir: dir.to_path_buf(),
            tool_name: "llama-server.exe".to_string(),
            manifest,
            model_path: model.to_path_buf(),
            draft_model_path: None,
            port: PortRequest::Any,
            api_key_file: key_path,
            allow_unsafe_dacl: None,
        };

        // The one production wiring of every check, deliberately: this is the test that proves
        // `preflight()` itself — real `ProcessTokenElevation` and real `ProbeDaclAuditor` both, not
        // fakes — refuses this machine's actual (unsafe-by-design) fixture directory and then
        // accepts it with the opt-in.
        let refused = preflight(&base);
        assert!(
            matches!(refused, Err(PreflightError::BackendDirUnsafe { .. })),
            "{refused:?}"
        );

        let allowed_spec = LaunchSpec {
            allow_unsafe_dacl: Some(
                AllowUnsafeDacl::new(
                    "known-unsafe development fixture; ASUS\\CodexSandboxUsers holds Modify on this \
                     directory by design of the test machine, not of Cerebro",
                )
                .expect("reason is non-blank"),
            ),
            ..base
        };
        let allowed = preflight(&allowed_spec).expect("must succeed with the opt-in when run unelevated");
        assert!(
            allowed
                .warnings()
                .iter()
                .any(|w| matches!(w, PreflightWarning::UnsafeDaclAccepted { principals, .. } if !principals.is_empty())),
            "{:?}",
            allowed.warnings()
        );
    }
}
