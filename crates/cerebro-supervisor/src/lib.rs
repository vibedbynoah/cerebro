//! Cerebro's launch gate and its supervised spawner: decide whether spawning `llama-server` is
//! allowed, then run it so it cannot outlive Cerebro or be steered by the environment.
//!
//! Three parts, in sequence:
//!
//! - [`mod@preflight`] is the last checkpoint before [`cerebro_flags`] argv is handed to a child
//!   process. It exists because a spawn cannot be undone: once `CreateProcess` runs, ggml has
//!   already had the chance to load whatever DLLs sit in the backend directory, and the model file
//!   has already been mapped. Every check in [`preflight::preflight`] therefore runs — and must
//!   pass, or be explicitly overridden with a recorded reason — strictly before that point.
//! - [`mod@spawn`] is what happens on the other side of that checkpoint: it turns a
//!   [`preflight::Preflighted`] launch into a running, job-contained child with a closed
//!   environment, no window, and bounded/logged output capture. The only ordinary way in is
//!   [`spawn::spawn_preflighted`], which requires that proof; every other route to `CreateProcess`
//!   — today, only [`spawn::spawn_unaudited`], for this crate's own live tests and a future
//!   bench/probe harness — carries a written [`spawn::AcceptUnauditedBackend`] reason instead, so
//!   nothing can reach a child process silently unaudited. See [`mod@spawn`]'s module docs.
//! - [`mod@lifecycle`] is the last part: it keeps a [`spawn::ChildHandle`] supervised for as long as
//!   Cerebro wants it running — readiness polling, crash classification, and a bounded restart
//!   policy — and, per ARCHITECTURE.md section 8, runs [`preflight::preflight`] again before every
//!   single restart, never only at the first launch. See [`mod@lifecycle`]'s module docs.
//!
//! Consistent with `cerebro-probe`'s stance (see its crate docs): a check that cannot establish an
//! answer returns a typed error rather than a best guess, because a wrong "safe to launch" is worse
//! than a launch that never happens.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod names;
pub mod lifecycle;
pub mod preflight;
pub mod spawn;

pub use lifecycle::{
    FailureReason, LifecycleConfig, LifecycleError, LifecycleState, RestartPolicy, Supervisor,
};
pub use preflight::{
    build_hardened_server_argv, preflight, AllowUnsafeDacl, LaunchSpec, ModelRole, PortRequest,
    PreflightError, PreflightWarning, Preflighted, LOOPBACK_HOST, MIN_API_KEY_FILE_BYTES,
};
pub use spawn::{
    spawn_preflighted, spawn_unaudited, AcceptUnauditedBackend, ChildHandle, SpawnError,
    SpawnOptions,
};
