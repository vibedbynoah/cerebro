//! The one schema-validated `settings.json` every later Cerebro surface reads.
//!
//! CLI, HTTP API, and Tauri shell all read Cerebro's configuration through this crate and none of
//! them owns it. That single ownership is what makes the security guarantees in [`profile`] mean
//! anything: if the API surface could parse `settings.json` its own way, "the API may not touch
//! `/security`" would be a convention two implementations happen to share rather than a property
//! one type enforces.
//!
//! # Modules
//!
//! * [`schema`] -- [`schema::Settings`], its sections, and the generated JSON Schema. Every
//!   `runtime.*` field is a newtype or re-exported enum from [`cerebro_flags`], so a value this
//!   crate accepts is, by construction, a value `cerebro-flags` can also accept.
//! * [`file`] -- `settings.json` on disk: comment stripping, the 1 MiB cap, atomic saves, debounced
//!   file watching, and "an invalid edit never corrupts the last known-good settings."
//! * [`profile`] -- the security profile: which pointers are locked and against which [`profile::Source`],
//!   model-path canonicalization against the approved roots, and the [`profile::Preflighter`] trait
//!   a spawn decision is read through.
//! * [`error`] -- [`error::ConfigError`], the one error type every fallible operation in this crate
//!   returns.
//! * [`secrets`] -- DPAPI storage for Cerebro's only two real secrets (the local API bearer token
//!   and an optional Hugging Face token) and the per-launch child key file
//!   `cerebro-supervisor` passes to `llama-server` as `--api-key-file`.
//!
//! # What is deliberately not representable
//!
//! There is no `security.allow_unsafe_dacl` key, and none may be added: see the module
//! documentation on [`schema`] for why a *locked* key was not a strong enough guarantee and the
//! value needed to be unrepresentable instead.
//!
//! # Why this crate is `deny`, not `forbid`, on unsafe code
//!
//! Every module here except [`secrets`] is ordinary safe Rust, and a crate-wide `forbid` would
//! have been simpler to write and to audit. It had to become `deny` instead: DPAPI
//! (`CryptProtectData`/`CryptUnprotectData`), converting an SDDL string to a `SECURITY_ATTRIBUTES`,
//! and reading a DACL to audit it have no safe-Rust API, and [`secrets`] is precisely the module
//! that needs all three. `deny` can be locally overridden by an inner `#![allow(unsafe_code)]`
//! (which only [`secrets`] carries); `forbid` cannot be overridden by anything, anywhere, which is
//! the whole reason it was chosen originally and the whole reason it could not stay once a real
//! Win32 FFI need appeared. Every other module is exactly as unsafe-free as it was before this
//! change -- nothing here widens what *they* may do.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod file;
pub mod profile;
pub mod schema;
pub mod secrets;

pub use error::ConfigError;
pub use file::{watch, watch_into, ConfigFile, ConfigWatcher, MAX_SETTINGS_FILE_BYTES, SETTINGS_FILE_NAME};
pub use profile::{
    // `open_approved_model` first: it is the one a caller that will actually read the file wants,
    // because it hands back the handle the approval was made against, closing the TOCTOU window on
    // the leaf. `resolve_model_path` is for display and listing, where nothing downstream is opened
    // -- see both functions' own doc comments on `crate::profile`.
    is_security_locked, open_approved_model, resolve_model_path, DefaultPreflighter, Preflighter, Source,
    SpawnDecision, SECURITY_LOCKED,
};
pub use schema::{json_schema, Settings};
pub use secrets::{KeyFile, Secret, SecretId, SecretStore, SecretsError};
