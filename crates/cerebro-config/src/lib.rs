//! The one schema-validated `settings.json` every Cerebro surface reads.
//!
//! CLI, HTTP API, and Tauri shell all read Cerebro's configuration through this crate and none of
//! them owns it. That single ownership is what makes "the API may not turn protections off" a
//! property one type enforces ([`profile::SECURITY_LOCKED`]) rather than a convention two parsers
//! happen to share.
//!
//! # Modules
//!
//! * [`schema`] -- [`Settings`], its sections, and the generated JSON Schema
//!   (`docs/schema/settings.schema.json`).
//! * [`load`] -- default model load parameters and named presets. Unset means llama.cpp's own
//!   automatic value, never a Cerebro guess.
//! * [`assistant`] -- the effort dial (thinking budget and verification rigor per level) and the
//!   verbosity dial (Brief by default), both fully editable.
//! * [`policy`] -- protections (every one a user setting, all on by default), network mode, and
//!   plugin permission defaults.
//! * [`file`](mod@file) -- `settings.json` on disk: comment stripping, the 1 MiB cap, atomic saves, debounced
//!   file watching, and "an invalid edit never corrupts the last known-good settings."
//! * [`profile`] -- which [`profile::Source`] may write which keys, and model-path resolution
//!   against the model folders.
//! * [`secrets`] -- DPAPI storage for the local API bearer token and an optional Hugging Face
//!   token, and the per-launch child key file.
//! * [`error`] -- [`ConfigError`], returned by every fallible settings operation.
//!
//! # No second copy of a safety check
//!
//! Launch decisions belong to `cerebro-supervisor`: [`Settings::launch_spec`] builds its
//! `LaunchSpec` with [`policy::ProtectionsSection::launch`], and its preflight audits, hashes and
//! refuses. DACL reading belongs to `cerebro-probe`: the secret store audits its files with
//! `cerebro_probe::backend::audit_object_dacl`. This crate keeps neither a spawn gate nor a DACL
//! walk of its own.
//!
//! # Why this crate is `deny`, not `forbid`, on unsafe code
//!
//! Every module except [`secrets`] is safe Rust. DPAPI (`CryptProtectData`/`CryptUnprotectData`)
//! and building an owner-only `SECURITY_ATTRIBUTES` from SDDL have no safe-Rust API, so that one
//! module opts back in with `#![allow(unsafe_code)]`, which a crate-level `forbid` would not allow.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod assistant;
pub mod error;
pub mod file;
pub mod load;
pub mod policy;
pub mod profile;
pub mod schema;
pub mod secrets;

pub use assistant::{EffortLevel, EffortSection, ThinkingBudget, VerbosityLevel, VerbositySection};
pub use error::ConfigError;
pub use file::{watch, watch_into, ConfigFile, ConfigWatcher, MAX_SETTINGS_FILE_BYTES, SETTINGS_FILE_NAME};
pub use load::{LoadParams, LoadSection};
pub use policy::{Capability, DisabledProtection, NetworkMode, NetworkSection, Permission, PluginsSection, ProtectionsSection};
pub use profile::{
    // `open_approved_model` first: it is the one a caller that will actually read the file wants,
    // because it hands back the handle the approval was made against. `resolve_model_path` is for
    // display and listing, where nothing downstream is opened.
    is_security_locked, open_approved_model, resolve_model_path, Source, SECURITY_LOCKED,
};
pub use schema::{json_schema, LaunchRequest, Settings, SCHEMA_VERSION};
pub use secrets::{KeyFile, Secret, SecretId, SecretStore, SecretsError};
