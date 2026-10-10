//! Typed failures for backend installs, manifests and launch commands.

use std::path::PathBuf;

use thiserror::Error;

use crate::backend::{BackendId, LlamaCppBuild};

/// Anything that can go wrong locating, verifying or building a launch for a backend.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum BackendError {
    /// The install directory could not be listed.
    #[error("backend directory {dir} could not be read: {cause}")]
    DirUnreadable {
        /// The directory.
        dir: PathBuf,
        /// The OS error, as text.
        cause: String,
    },

    /// Files the backend needs are missing from its directory.
    #[error("{backend} in {dir} is missing {missing:?}")]
    MissingFiles {
        /// Which backend.
        backend: BackendId,
        /// The directory checked.
        dir: PathBuf,
        /// File names not found.
        missing: Vec<String>,
    },

    /// A llama.cpp folder holds a different build than the one it was registered as.
    #[error("{dir} holds the llama.cpp {found} build, not {expected}")]
    WrongBuild {
        /// The directory checked.
        dir: PathBuf,
        /// The build this backend expects.
        expected: LlamaCppBuild,
        /// The build its files identify.
        found: LlamaCppBuild,
    },

    /// A llama.cpp folder's files do not identify one build.
    #[error("{dir} is not an identifiable llama.cpp build: {detail}")]
    UnidentifiedBuild {
        /// The directory checked.
        dir: PathBuf,
        /// Why.
        detail: String,
    },

    /// Hashing the backend's files failed.
    #[error("hashing {backend}'s files failed: {cause}")]
    Hash {
        /// Which backend.
        backend: BackendId,
        /// The probe error, as text.
        cause: String,
    },

    /// The backend installs into a Python environment, whose files a pinned manifest cannot cover.
    #[error("{backend} has no integrity manifest: {reason}")]
    ManifestUnsupported {
        /// Which backend.
        backend: BackendId,
        /// Why.
        reason: &'static str,
    },

    /// Cerebro has no launch recipe for this backend.
    #[error("{backend} cannot be launched by Cerebro yet: {reason}")]
    LaunchUnsupported {
        /// Which backend.
        backend: BackendId,
        /// Why.
        reason: &'static str,
    },

    /// A launch option this backend cannot honour.
    #[error("{backend} does not support {option}: {hint}")]
    OptionUnsupported {
        /// Which backend.
        backend: BackendId,
        /// The option.
        option: &'static str,
        /// What to do instead.
        hint: &'static str,
    },

    /// The backend cannot take an API key and the user's `require_api_key` protection is on.
    /// Governed by that setting: turning it off lets the backend run keyless on loopback.
    #[error(
        "{backend} cannot take an API key; turn off protections.require_api_key in settings to \
         run it keyless on loopback"
    )]
    KeylessRefused {
        /// Which backend.
        backend: BackendId,
    },

    /// A launch request field is invalid.
    #[error("invalid launch request: {0}")]
    InvalidRequest(String),

    /// The closed llama.cpp flag allowlist refused the argv.
    #[error("llama.cpp argv refused: {0}")]
    Argv(#[from] cerebro_flags::EmitError),
}
