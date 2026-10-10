//! The one seam between [`super::run`]'s state machine and the real, synchronous
//! preflight-then-spawn path: [`Launcher`] and [`SupervisedChild`] let the driver be exercised
//! against a fake in tests without ever spawning a real OS process, while [`RealLauncher`] is what
//! production actually calls.
//!
//! Both traits are `pub(crate)`, not `pub`: like [`crate::preflight::ElevationCheck`] and
//! [`crate::preflight::DaclAuditor`], they exist purely as a test seam for this crate's own suite,
//! and an outside caller substituting its own [`Launcher`] would defeat the very guarantee
//! [`super::LifecycleConfig`] exists to state plainly — that every restart really does go back
//! through [`crate::preflight::preflight`].

use std::time::Duration;

use cerebro_flags::{ArgvBuilder, Dialect, Flag};
use thiserror::Error;

use crate::preflight::{self, LaunchSpec, PortRequest};
use crate::spawn::{self, SpawnOptions};

use super::listener::ListenerOwner;
use super::LaunchRecord;

/// A supervised child's exit, reduced to the two facts [`super::classify`] needs and nothing else —
/// not [`std::process::ExitStatus`] itself, so a fake implementation of [`SupervisedChild`] can
/// construct one directly instead of needing a real OS process (or a platform-specific
/// `ExitStatusExt::from_raw`) just to report an exit code in a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExitInfo {
    /// The process's exit code, if the OS reported one (a process killed by a signal, on a platform
    /// that has those, reports `None`).
    pub(crate) code: Option<i32>,
}

/// Abstraction over a running supervised child, so [`super::run`]'s state machine can be driven
/// against a fake in tests without spawning a real OS process. Implemented for the real
/// [`crate::spawn::ChildHandle`] below.
pub(crate) trait SupervisedChild: Send + Sync {
    /// Poll, without blocking, whether the child has exited.
    fn poll_exit(&self) -> Result<Option<ExitInfo>, LaunchError>;

    /// A snapshot of the child's most recent captured output lines, oldest first.
    fn recent_lines(&self) -> Vec<String>;

    /// Kill the child and wait up to `timeout` for the OS to confirm it is gone. `true` once
    /// confirmed gone; `false` if `timeout` elapsed first. Synchronous and potentially slow (up to
    /// `timeout`) — [`super::run`] never calls this inline; see its own `stop_child` helper, which
    /// runs it on `tokio::task::spawn_blocking`.
    fn stop(&self, timeout: Duration) -> Result<bool, LaunchError>;

    /// Who owns the TCP listener on `127.0.0.1:port`: this child's own process, another process
    /// that won the port after preflight released it, nobody yet, or unknown. Readiness polling
    /// sends nothing (the bearer key included) until this answers [`ListenerOwner::Ours`]; see
    /// `super::listener`.
    fn listener_owner(&self, port: u16) -> ListenerOwner;
}

impl SupervisedChild for spawn::ChildHandle {
    fn poll_exit(&self) -> Result<Option<ExitInfo>, LaunchError> {
        self.try_wait()
            .map(|status| status.map(|s| ExitInfo { code: s.code() }))
            .map_err(|e| LaunchError::Child(e.to_string()))
    }

    fn recent_lines(&self) -> Vec<String> {
        spawn::ChildHandle::recent_lines(self)
    }

    fn stop(&self, timeout: Duration) -> Result<bool, LaunchError> {
        self.kill_and_wait(timeout)
            .map(|status| status.is_some())
            .map_err(|e| LaunchError::Child(e.to_string()))
    }

    fn listener_owner(&self, port: u16) -> ListenerOwner {
        // `self` holds the child's process handle, so its PID cannot be reused while this runs.
        super::listener::listener_owner(self.pid(), port)
    }
}

/// A freshly preflighted and spawned child, the port it is actually listening on (the concrete
/// value [`crate::preflight::Preflighted::port`] resolved to, not merely the [`PortRequest`] asked
/// for), and this launch's own [`LaunchRecord`].
///
/// `child` is an `Arc`, not a `Box`: [`super::run`] hands a clone of it to `tokio::task::spawn_blocking`
/// every time it needs to stop the child, so that synchronous, potentially-`timeout`-long wait never
/// runs inline on the async driver task — see `super::run`'s own `stop_child` helper.
pub(crate) struct Launched {
    pub(crate) child: std::sync::Arc<dyn SupervisedChild>,
    pub(crate) port: u16,
    pub(crate) record: LaunchRecord,
}

/// Everything that can make a launch attempt fail, before the child is even running to have an
/// opinion of its own.
#[derive(Debug, Error)]
pub(crate) enum LaunchError {
    /// Preflight refused the launch.
    #[error(transparent)]
    Preflight(#[from] preflight::PreflightError),
    /// Argv construction failed.
    #[error(transparent)]
    Argv(#[from] cerebro_flags::EmitError),
    /// The OS-level spawn call failed.
    #[error(transparent)]
    Spawn(#[from] spawn::SpawnError),
    /// A [`SupervisedChild`] operation (poll, stop) failed at the OS level, after a successful
    /// spawn.
    #[error("{0}")]
    Child(String),
}

/// Run [`crate::preflight::preflight`] against `port`, then spawn the result — the one operation
/// [`super::run`] must repeat, unabridged, before every restart (ARCHITECTURE.md section 8: verified
/// before every spawn, not merely at the first launch).
pub(crate) trait Launcher: Send + Sync {
    /// Preflight and spawn a fresh launch attempt bound to `port`.
    fn launch(&self, port: PortRequest) -> Result<Launched, LaunchError>;
}

/// The production [`Launcher`]: preflights `base_spec` (with `port` substituted fresh on every
/// call) and spawns the result with `extra_flags` layered under preflight's own hardened,
/// preflight-owned flags.
pub(crate) struct RealLauncher {
    base_spec: LaunchSpec,
    extra_flags: Vec<Flag>,
    spawn_options: SpawnOptions,
}

impl RealLauncher {
    /// Build a launcher around a fixed spec/flags/options triple. `base_spec.port` is never read —
    /// every call to [`Launcher::launch`] substitutes its own `port` argument instead (see that
    /// method's docs) — but the whole `LaunchSpec` is kept, port field included, since
    /// [`crate::preflight::LaunchSpec`] is not otherwise splittable without duplicating every other
    /// field here.
    pub(crate) fn new(base_spec: LaunchSpec, extra_flags: Vec<Flag>, spawn_options: SpawnOptions) -> Self {
        Self {
            base_spec,
            extra_flags,
            spawn_options,
        }
    }
}

impl Launcher for RealLauncher {
    fn launch(&self, port: PortRequest) -> Result<Launched, LaunchError> {
        let mut spec = self.base_spec.clone();
        spec.port = port;

        // Run, in full, on every call — never cached or skipped on a retry. This is the property
        // ARCHITECTURE.md section 8 requires and the one this whole function exists to preserve:
        // a backend file replaced between restarts, or a DACL that turned unsafe, must be caught
        // again here, not waved through because it passed once.
        let preflighted = preflight::preflight(&spec)?;

        let mut builder = ArgvBuilder::new(Dialect::Server);
        for flag in self.extra_flags.iter().cloned() {
            builder.push(flag)?;
        }
        let argv = preflight::build_hardened_server_argv(&preflighted, builder)?;

        let port = preflighted.port();
        let child = spawn::spawn_preflighted(&preflighted, &argv, &self.spawn_options)?;
        let record = LaunchRecord {
            preflight_warnings: preflighted.warnings().to_vec(),
            protection_warnings: child.protection_warnings().to_vec(),
            port,
            requested_backend_dir: self.base_spec.backend_dir.clone(),
            backend_dir: preflighted.backend_dir().to_path_buf(),
            backend_hashes: preflighted.backend_hashes().clone(),
            pid: child.pid(),
        };
        Ok(Launched {
            child: std::sync::Arc::new(child),
            port,
            record,
        })
    }
}
