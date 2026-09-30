//! Typed failures for reading, writing, patching, and resolving paths against `settings.json`.
//!
//! Every failure mode a caller can hit is one variant here. Two design rules from the brief show up
//! directly in the shape of this enum:
//!
//! * A malformed or hostile file must never take down the in-memory configuration — [`file`](
//!   crate::file) always has a variant to report, never a panic, and the caller decides what "last
//!   good" means.
//! * [`ConfigError::ModelPathNotApproved`] carries no detail at all. A path can fail
//!   [`crate::profile::resolve_model_path`] because it does not exist, because it resolves outside
//!   every approved root, or because it is a UNC/device/ADS path — and telling those apart in the
//!   error would make the error itself an oracle an attacker could use to enumerate the filesystem
//!   one probe at a time. One message, always.

use std::io::ErrorKind;
use std::path::PathBuf;

use thiserror::Error;

use crate::profile::Source;

/// Maximum length, in `char`s, of any error field echoed from untrusted input. Mirrors
/// `cerebro_store::record::cap_echo` (same value, same reason): this is the one crate in the
/// workspace whose input is by definition untrusted, and a hostile `settings.json` must not be
/// able to choose a megabyte of the text that lands in a log line or a UI toast just by naming an
/// unknown field or a JSON Patch pointer.
const MAX_ECHO_CHARS: usize = 80;

/// Truncate untrusted text to at most [`MAX_ECHO_CHARS`] characters, appending an ellipsis marker
/// when truncated. Splits on a `char` boundary, never mid-codepoint, so this is safe on non-ASCII
/// input.
pub(crate) fn cap_echo(s: &str) -> String {
    match s.char_indices().nth(MAX_ECHO_CHARS) {
        None => s.to_string(),
        Some((byte_idx, _)) => format!("{}…", &s[..byte_idx]),
    }
}

/// Anything that can go wrong while loading, saving, watching, or patching Cerebro's settings, or
/// while resolving a model path against them.
///
/// `Eq` is deliberately not derived: [`ConfigError::SamplerOutOfRange`] carries an `f64`, which has
/// no total order, so the whole enum can only be `PartialEq` -- the same trade-off
/// `cerebro_store::RecordError` makes for the same reason.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// The settings file's content is not a valid `Settings` document.
    ///
    /// Covers every content-level failure uniformly: JSON syntax errors, a value of the wrong
    /// type, an unknown field (`#[serde(deny_unknown_fields)]`), a duplicate key (a struct field,
    /// or a key of a map-backed section such as `load.presets` or `effort.levels`), and content
    /// that is not valid UTF-8 at all. All of these are "the bytes on disk do not describe a `Settings`"
    /// and all of them leave the last good in-memory configuration untouched.
    #[error("{path} is not a valid settings file at line {line}, column {column}: {message}")]
    Invalid {
        /// The file that failed to parse.
        path: PathBuf,
        /// 1-based line of the failure, best-effort for non-UTF-8 content.
        line: usize,
        /// 1-based column of the failure, best-effort for non-UTF-8 content.
        column: usize,
        /// Human-readable detail, e.g. `unknown field \`allow_unsafe_dacl\`, expected one of ...`.
        ///
        /// Echoes untrusted input (serde's own message embeds the attacker-chosen field name or
        /// value verbatim), so it is capped to [`MAX_ECHO_CHARS`] at construction -- see
        /// [`cap_echo`] -- before it ever reaches a caller, the UI, or a log.
        message: String,
    },

    /// The settings file is larger than the 1 MiB cap.
    #[error("{path} is {size} bytes, over the 1 MiB cap")]
    TooLarge {
        /// The file that was checked.
        path: PathBuf,
        /// Its size in bytes, read from the open handle's metadata. Falls back to the
        /// `Read::take` sentinel (`MAX_SETTINGS_FILE_BYTES + 1`) only if that metadata call itself
        /// fails, so this is "the real size" in the overwhelming common case rather than always a
        /// fixed "more than the cap" marker.
        size: u64,
    },

    /// The settings file (or its directory) could not be read.
    ///
    /// `kind` is what lets [`crate::file::ConfigFile::open`] tell "the file genuinely does not
    /// exist yet" ([`ErrorKind::NotFound`], the only case that bootstraps a default document) apart
    /// from every other failure (permission denied, a sharing violation, a transient I/O error),
    /// which must propagate rather than silently overwrite a file that could not be read for some
    /// other reason.
    #[error("could not read {path}: {cause}")]
    Read {
        /// The path that was being read.
        path: PathBuf,
        /// The underlying [`std::io::Error`]'s kind. `Copy`, unlike the error itself, so this stays
        /// on an enum that is still `Clone` + `PartialEq`.
        kind: ErrorKind,
        /// Underlying OS error, rendered as text so this error stays `Clone` + `PartialEq`.
        cause: String,
    },

    /// The settings file could not be written.
    #[error("could not write {path}: {cause}")]
    Write {
        /// The path that was being written (the final destination, not the temp file).
        path: PathBuf,
        /// Underlying OS error, rendered as text.
        cause: String,
    },

    /// A background watch on a directory could not be established.
    #[error("could not watch {dir}: {cause}")]
    WatchSetup {
        /// The directory that was to be watched.
        dir: PathBuf,
        /// Underlying watcher-crate error, rendered as text.
        cause: String,
    },

    /// A patch touched a security-locked JSON pointer from a [`Source`] not permitted to touch it.
    ///
    /// The whole patch is refused, never partially applied — see
    /// [`crate::schema::Settings::apply_patch`].
    #[error("{by:?} may not modify the security-locked pointer {pointer}")]
    LockedPointerRefused {
        /// The JSON pointer that was locked, e.g. `/network/lan_mode`.
        ///
        /// Echoes untrusted input -- an [`Source::Api`] caller chooses this string -- so it is
        /// capped to [`MAX_ECHO_CHARS`] at construction; see [`cap_echo`].
        pointer: String,
        /// The source the patch claimed to come from.
        by: Source,
    },

    /// A patch was individually or structurally invalid (not a locked-pointer refusal): it merged
    /// onto the current settings into something that is not a valid `Settings` document.
    #[error("patch could not be applied: {0}")]
    PatchRejected(String),

    /// A candidate model path is not under any approved model root.
    ///
    /// Deliberately carries no path, no cause, and no distinction between "the path does not
    /// exist" and "the path exists but resolves outside every approved root" — see the module
    /// documentation above.
    #[error("path is not under an approved model root")]
    ModelPathNotApproved,

    /// A path that must be a regular file (or directory) turned out to be a reparse point --
    /// a symlink or NTFS junction.
    ///
    /// A principal who can write the directory a path lives in can plant one of these to redirect
    /// a read to somewhere the caller never intended, so every read of `settings.json` opens with
    /// `FILE_FLAG_OPEN_REPARSE_POINT` and refuses here rather than transparently following it. See
    /// `cerebro_store::StoreError::PathIsReparsePoint`, which exists for the identical reason.
    #[error("{path} is a reparse point (symlink or junction), not a regular file")]
    PathIsReparsePoint {
        /// The path that turned out to be a reparse point.
        path: PathBuf,
    },

    /// A value is well-typed but breaks a rule of the settings document: an active preset that does
    /// not exist, a default level that names no level, a duplicate rank, a bound exceeded.
    #[error("{pointer}: {reason}")]
    SettingRejected {
        /// JSON pointer of the offending value, e.g. `/effort/default_level`, or the field name
        /// when the value is checked outside a document (a thread count built directly). User-chosen
        /// key names inside it are truncated to 80 characters.
        pointer: String,
        /// What rule it breaks.
        reason: String,
    },
    /// A sampler field is outside the range llama.cpp -- and `cerebro_store::SamplerProfile`, which
    /// records the run -- would accept.
    #[error("sampler.{field} = {value} is out of range")]
    SamplerOutOfRange {
        /// Which field, e.g. `"temp"`.
        field: &'static str,
        /// The rejected value.
        value: f64,
    },

    /// The document declares a schema version this build does not understand.
    ///
    /// House rule S6 ("an unknown state is a refusal") applies to the document's own version, not
    /// just to its content: `#[serde(deny_unknown_fields)]` only catches a future version that
    /// *adds* a key, not one that silently changes what an existing key *means*. See
    /// [`crate::schema::SCHEMA_VERSION`].
    #[error("settings schema version {found} is not supported (this build reads version {supported})")]
    UnsupportedVersion {
        /// The version the document declared.
        found: u32,
        /// The version this build understands.
        supported: u32,
    },
}
