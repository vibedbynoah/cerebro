//! Supervising a `llama-server` child once [`mod@crate::preflight`] and [`mod@crate::spawn`] have
//! proven it safe to launch: readiness, crash classification, and a bounded restart policy,
//! observable through two channels [`Supervisor::spawn`] returns — see "Two observation channels,
//! one contract each" below for why there are two.
//!
//! # State machine
//!
//! ```text
//! Starting -> Ready                                     (readiness probe succeeds)
//! Starting -> Crashed(reason) -> Starting               (one immediate retry; port-bind failure only,
//!                                                        including another process owning the port)
//! Starting -> Crashed(reason) -> Backoff -> Starting    (ordinary counted restart)
//! Starting -> Crashed(reason) -> Failed(reason)         (restart window exceeded, or a
//!                                                        configuration error no retry can fix:
//!                                                        a permanent preflight refusal, rejected
//!                                                        argv, a 401 from our own child, an
//!                                                        unbuildable request)
//! Ready    -> Crashed(reason) -> Backoff -> Starting    (crash after readiness; same restart policy)
//! Ready    -> Crashed(reason) -> Failed(reason)
//! Starting -> Stopped                                   (Supervisor::stop while idle or launching)
//! Starting -> Crashed(reason) -> Stopped                (a stop that arrived while the failed attempt
//!                                                        was in flight outranks every retry decision)
//! Backoff  -> Stopped                                   (Supervisor::stop cancels a pending restart)
//! Backoff  -> Failed(PreviousChildNotStopped)           (the retired child is still not confirmed
//!                                                        gone; see "Every abandoned launch ...")
//! Ready    -> Stopped                                   (Supervisor::stop while serving)
//! ```
//!
//! [`Supervisor::spawn`] starts in `Starting` and keeps restarting on a crash until either
//! [`Supervisor::stop`] moves it to the terminal `Stopped`, or the restart policy's sliding-window
//! limit is exceeded and it moves to the terminal `Failed` instead.
//!
//! # Two observation channels, one contract each
//!
//! [`Supervisor::spawn`] returns both a [`tokio::sync::watch::Receiver<LifecycleState>`] and a
//! [`tokio::sync::broadcast::Receiver<LifecycleState>`], and the two promise different things.
//!
//! `watch` only ever holds the *current* state — a second `send` before a receiver's
//! `changed().await` wakes simply overwrites the first, it does not queue it — so a receiver that is
//! slower than the driver can genuinely never observe some intermediate state at all. That is fine
//! for [`LifecycleState::Crashed`] specifically, because [`LifecycleState::Backoff`] now carries the
//! same [`FailureReason`] that produced it: a `watch`-only observer that lands on `Backoff` still
//! knows why the server is restarting, even if it never saw the `Crashed` value in between.
//!
//! `broadcast` is the ordered, complete transition log: every state this module ever publishes is
//! sent to it once, in order, and a receiver that falls behind reports that explicitly —
//! [`tokio::sync::broadcast::error::RecvError::Lagged`] — rather than silently missing states the way
//! `watch` does. This is what this crate's own state-machine tests are written against; a caller that
//! needs to know a server was ever `Crashed` at all (for a UI history, or an audit log) should
//! subscribe to this one instead of polling `watch`.
//!
//! Both channels are fed by the same internal [`Publisher::publish`] call for every transition, so
//! they can never disagree about what happened, only about which subset of it a given receiver
//! managed to observe.
//!
//! # Why every restart re-preflights
//!
//! ARCHITECTURE.md section 8 requires backend integrity and the DACL audit to be "verified before
//! every spawn, not merely at install" — and this module reads that literally: [`launch::RealLauncher`]
//! calls [`crate::preflight::preflight`] fresh on every single launch attempt, including the very
//! first one, never caching a [`crate::preflight::Preflighted`] value across a restart. A backend
//! file replaced between two restarts, or a DACL that turned unsafe in between, is caught again here,
//! not waved through because it passed once.
//!
//! # Every abandoned launch is stopped before the next one
//!
//! Whatever ends a launch attempt short of a deliberate stop (the child exiting, a failed readiness
//! check, or an exit poll the OS could not answer), its process tree is stopped *before*
//! [`LifecycleState::Crashed`] is published and the backoff delay starts, so a dead launch never
//! holds VRAM or its port through the restart. If the OS does not confirm that stop, the child is
//! kept as the one lingering child, and nothing is launched beside it: once the backoff delay has
//! elapsed, the driver asks the OS once more (a cheap exit poll, then one more blocking stop). If
//! that confirms, the restart goes ahead on the configured port. If it does not,
//! [`LifecycleState::Failed`] is published with [`FailureReason::PreviousChildNotStopped`] instead
//! of launching a second server that would compete with the first for VRAM (on an 8 GB GPU, a
//! slower placement or an out-of-memory load) and for its port. Two servers therefore never run
//! side by side, and at most one unconfirmed child exists at any time.
//!
//! # A stop is `Ok` only when every child is confirmed gone
//!
//! A retired child the OS did not confirm gone is not forgotten. The driver keeps it (at most one;
//! see above) until an exit poll or a repeated stop confirms it gone, and asks the OS once more on
//! the way to any terminal state ([`LifecycleState::Stopped`] or [`LifecycleState::Failed`]). [`Supervisor::stop`] returns `Ok(())` only if that last check, and the
//! stop of the live child if there was one, both confirmed. Otherwise it returns
//! [`LifecycleError::StopNotConfirmed`], whichever branch the driver ended on: the terminal result is
//! derived from confirmation state alone.
//!
//! # The configured port is never overwritten
//!
//! [`LaunchSpec::port`] is what the user asked for (most often an explicit port an IDE or client is
//! pointed at), so [`run`] never mutates it. The one situation that justifies a different port, the
//! immediate retry after a [`FailureReason::PortBindFailure`], sets a one-shot override that the
//! very next launch attempt consumes. Every attempt after that one asks for the configured port
//! again. An unconfirmed stop needs no override: no attempt is made until the old child is
//! confirmed gone (see above), and then its port is free.
//!
//! # Readiness trusts only the child's own listener
//!
//! Preflight binds and releases the port before the child binds it, so another local process can
//! take the port in between. Readiness polling therefore asks the OS's TCP table who owns the
//! listener before it sends anything (see `listener` and `readiness::wait_ready`): no request, and
//! so no bearer key, goes out until every listener on the port belongs to the child's process. A
//! listener owned by anyone else ends the attempt as [`FailureReason::PortBindFailure`] (the one
//! fresh-port retry); no listener yet, or a table that cannot be read, keeps waiting until the
//! deadline. [`LifecycleState::Ready`] is therefore only ever published for our own child, and a 401
//! is only ever our own child rejecting the configured key.
//!
//! Readiness polling is bounded by [`LifecycleConfig::readiness_deadline`] per request as well as
//! overall, so a server that accepts connections but never answers `GET /health` still ends in
//! [`FailureReason::ReadinessDeadlineExceeded`] rather than holding the driver in `Starting` forever.
//!
//! # Why [`Launcher`](launch::Launcher) and [`SupervisedChild`](launch::SupervisedChild) exist
//!
//! [`run`] is the state machine; it never spawns a process or opens a socket itself. Both of those
//! live behind small, injectable, `pub(crate)` traits — the same test-seam discipline
//! [`crate::preflight::ElevationCheck`] and [`crate::preflight::DaclAuditor`] already use — so this
//! module's own tests can drive every transition (a crash before readiness, a bind-failure retry,
//! the restart limit) against a fake in milliseconds, deterministically, with `tokio::time` paused,
//! rather than needing a real `llama-server` process for each one.

mod backoff;
mod classify;
mod launch;
mod listener;
mod readiness;

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

use cerebro_flags::Flag;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, watch};

use crate::preflight::{self, LaunchSpec, PortRequest};
use crate::spawn::SpawnOptions;

use backoff::{backoff_delay, RestartWindow};
use launch::{ExitInfo, LaunchError, Launched, Launcher, RealLauncher, SupervisedChild};

pub use backoff::RestartPolicy;
pub use classify::FailureReason;

/// Everything recorded about one successful launch attempt: the launch record the owner rule refers
/// to when it says turning a protection off is "recorded as a structured warning in the launch
/// record" (see [`crate::protections`]'s own module docs). Carries both halves of that record —
/// [`crate::preflight::preflight`]'s own four switches and [`mod@crate::spawn`]'s two — as one
/// value, rather than leaving a caller to reassemble them from two different types, so a launch's
/// warnings are never split across two places (or, as this crate's own history had it for a while,
/// dropped on the floor because neither half was ever read past the launch that produced it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchRecord {
    /// Non-fatal conditions [`crate::preflight::preflight`] recorded for this launch — including a
    /// [`crate::preflight::PreflightWarning::ProtectionDisabled`] entry for each of preflight's own
    /// four switches ([`crate::protections::Protections::refuse_elevation`],
    /// [`crate::protections::Protections::audit_folder_permissions`],
    /// [`crate::protections::Protections::verify_backend_integrity`],
    /// [`crate::protections::Protections::refuse_redirected_paths`]) the caller turned off. The
    /// redirect switch is also enforced by spawn but recorded here only, so it appears once.
    pub preflight_warnings: Vec<preflight::PreflightWarning>,
    /// The spawn-owned protections ([`crate::protections::Protections::enforce_environment_allowlist`],
    /// [`crate::protections::Protections::contain_in_job_object`]) the caller turned off for this
    /// launch — see [`crate::spawn::ChildHandle::protection_warnings`].
    pub protection_warnings: Vec<crate::protections::ProtectionWarning>,
    /// The concrete port the child was launched on.
    pub port: u16,
    /// [`LaunchSpec::backend_dir`] exactly as configured, before any resolution.
    pub requested_backend_dir: std::path::PathBuf,
    /// The canonical directory this launch actually audited, hashed and spawned from
    /// ([`crate::preflight::Preflighted::backend_dir`]). Differs from
    /// [`LaunchRecord::requested_backend_dir`] when the configured path reaches the backend through
    /// a junction or `subst` alias. Recorded as a fact rather than warned about: whoever can repoint
    /// such an alias chooses which directory the next launch runs from, so a caller (a store, a UI)
    /// can compare this value between launches and see a changed resolution.
    pub backend_dir: std::path::PathBuf,
    /// File name -> lowercase hex SHA-256 of every backend file verified for this launch
    /// ([`crate::preflight::Preflighted::backend_hashes`]). Empty when
    /// [`crate::protections::Protections::verify_backend_integrity`] was off.
    pub backend_hashes: std::collections::BTreeMap<String, String>,
    /// OS process id of the spawned child. Only meaningful while that child is alive: Windows reuses
    /// process ids, so a caller checking it later must have opened a handle while it was running.
    pub pid: u32,
}

impl LaunchRecord {
    /// Every protection turned off for this launch, preflight- and spawn-owned alike, in one list —
    /// so a caller (a settings UI listing what is currently disabled, most plausibly) never has to
    /// know that this record's warnings happen to live in two separate fields of two separate types
    /// to read all of them.
    pub fn disabled_protections(&self) -> Vec<&crate::protections::ProtectionWarning> {
        self.preflight_warnings
            .iter()
            .filter_map(|w| match w {
                preflight::PreflightWarning::ProtectionDisabled(p) => Some(p),
                _ => None,
            })
            .chain(self.protection_warnings.iter())
            .collect()
    }
}

/// Everything needed to launch and keep a `llama-server` supervised.
///
/// `Debug` is implemented by hand so [`LifecycleConfig::api_key`] (the bearer secret for the
/// supervised server) prints as `<redacted>` and never reaches a log line, a panic message or a
/// settings dump.
#[derive(Clone)]
pub struct LifecycleConfig {
    /// Preflighted fresh before every (re-)launch. Its `port` field is the port every attempt asks
    /// for, with one one-attempt exception: the immediate retry after a detected
    /// [`FailureReason::PortBindFailure`] asks for [`PortRequest::Any`] once. The attempt after that
    /// asks for this port again — see
    /// this module's "The configured port is never overwritten" section. A second bind failure before
    /// any launch reaches [`LifecycleState::Ready`] is counted like any other failure.
    pub launch_spec: LaunchSpec,
    /// Flags layered under preflight's own hardened, preflight-owned set (`--host`, `--port`,
    /// `--api-key-file`, `-m`, `-md`, and the loopback hardening flags — see
    /// [`crate::preflight::build_hardened_server_argv`]). Must not duplicate any of those.
    pub extra_flags: Vec<Flag>,
    /// Passed to [`crate::spawn::spawn_preflighted`] on every launch.
    pub spawn_options: SpawnOptions,
    /// The bearer token sent as `Authorization: Bearer <api_key>` on every readiness probe. Must
    /// name one of the keys inside the file `launch_spec.api_key_file` points at — this is the value
    /// Cerebro itself sends, not something recovered from that file, since preflight only ever
    /// proves the file well-formed and never returns its contents.
    pub api_key: String,
    /// How long readiness polling waits for `GET /health` to return 200 before giving up.
    pub readiness_deadline: Duration,
    /// How often `GET /health` is polled while waiting for readiness.
    pub readiness_poll_interval: Duration,
    /// How often the child's own exit status is polled, both while waiting for readiness and once
    /// `Ready`.
    pub child_poll_interval: Duration,
    /// How restarts back off, and when they give up entirely.
    pub restart_policy: RestartPolicy,
    /// How long a stop (deliberate, or forced by a rejected readiness check) waits for the OS to
    /// confirm the child is gone.
    pub stop_timeout: Duration,
}

impl std::fmt::Debug for LifecycleConfig {
    /// Every field except [`LifecycleConfig::api_key`], which is printed as `<redacted>`. A field
    /// added to [`LifecycleConfig`] must be added here too; the test
    /// `debug_names_every_field_and_never_prints_the_api_key` checks that every field name appears.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Destructured so adding a field without listing it here is a compile error.
        let Self {
            launch_spec,
            extra_flags,
            spawn_options,
            api_key: _,
            readiness_deadline,
            readiness_poll_interval,
            child_poll_interval,
            restart_policy,
            stop_timeout,
        } = self;
        f.debug_struct("LifecycleConfig")
            .field("launch_spec", launch_spec)
            .field("extra_flags", extra_flags)
            .field("spawn_options", spawn_options)
            .field("api_key", &format_args!("<redacted>"))
            .field("readiness_deadline", readiness_deadline)
            .field("readiness_poll_interval", readiness_poll_interval)
            .field("child_poll_interval", child_poll_interval)
            .field("restart_policy", restart_policy)
            .field("stop_timeout", stop_timeout)
            .finish()
    }
}

/// A supervised child's lifecycle, observable through the [`watch::Receiver`] [`Supervisor::spawn`]
/// returns. See this module's top-level docs for the full state diagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleState {
    /// A launch attempt is preflighting and/or spawning, or the child is running but has not yet
    /// answered `GET /health` with 200.
    Starting,
    /// `GET /health` returned 200. The child is being supervised for as long as this state holds.
    Ready,
    /// The child stopped, or never started, for the given reason. Followed by
    /// [`LifecycleState::Starting`] directly for the one immediate port-bind retry, by
    /// [`LifecycleState::Backoff`] for an ordinary counted restart, or by
    /// [`LifecycleState::Failed`] once the restart window is exceeded (or at once, for a
    /// configuration error such as [`FailureReason::InvalidReadinessRequest`]) — see this module's
    /// top-level state diagram for exactly which. Never terminal itself.
    Crashed(FailureReason),
    /// Waiting `delay` before the restart attempt numbered `attempt` (an in-window restart count,
    /// not a lifetime total — see this crate's `backoff` module docs on why old restarts age out of
    /// both the limit and this number together). A stop requested during this state cancels the
    /// pending restart and moves straight to [`LifecycleState::Stopped`].
    Backoff {
        /// The in-window restart count this delay was computed for.
        attempt: u32,
        /// How long this state holds before the next [`LifecycleState::Starting`].
        delay: Duration,
        /// Why the restart being waited out here happened — the same value the
        /// [`LifecycleState::Crashed`] this `Backoff` follows carried. Carried here too, not only on
        /// `Crashed`, so a [`watch`]-only observer that missed the intermediate `Crashed` value (see
        /// this module's top-level "Two observation channels" section) still knows why the server is
        /// restarting.
        reason: FailureReason,
    },
    /// Stopped deliberately, via [`Supervisor::stop`] (or a [`Supervisor`] dropped without calling
    /// it). Terminal: nothing further is ever published after this. Published whether or not the OS
    /// confirmed every child gone; [`Supervisor::stop`]'s own result says which.
    Stopped,
    /// The restart policy's sliding-window limit was exceeded, or the failure was a configuration
    /// error no retry can fix; this launch is no longer retried. Terminal.
    Failed(FailureReason),
}

/// Everything a caller of [`Supervisor::stop`] can see go wrong.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LifecycleError {
    /// The background driver task panicked. Every fallible step inside it is handled with a typed
    /// [`FailureReason`] instead (see [`FailureReason::SpawnFailed`]'s "launch task panicked" case),
    /// so this should never actually happen — but awaiting [`tokio::task::JoinHandle`] can still
    /// report it, and a caller deserves a typed answer rather than an unwrapped `JoinError`.
    #[error("the lifecycle driver task panicked: {0}")]
    DriverPanicked(String),

    /// [`Supervisor::spawn`] was called outside a Tokio runtime, so there was nothing to run the
    /// driver task on. Nothing was launched.
    #[error("Supervisor::spawn must be called from inside a Tokio runtime: {0}")]
    NoRuntime(String),

    /// [`LifecycleConfig::api_key`] cannot be sent as an `Authorization: Bearer` header value (it
    /// contains CR, LF or another byte a header may not hold), so no readiness probe could ever be
    /// built. Refused before anything is launched. The message never contains the key.
    #[error("the configured API key is not a valid HTTP header value; nothing was launched")]
    InvalidApiKey,

    /// A child this driver launched was asked to stop (the live one, by [`Supervisor::stop`], or an
    /// earlier one retired after a crash or failed readiness check) and the OS did not confirm it gone
    /// within `timeout`, even when asked once more at the terminal state. [`LifecycleState::Stopped`]
    /// or [`LifecycleState::Failed`] was still published — a caller watching only the state channel
    /// cannot tell the difference — so check this error specifically when whatever the old child held
    /// (its port, VRAM) needs to be known free before doing anything else.
    #[error("the supervised child was not confirmed stopped within {timeout:?}")]
    StopNotConfirmed {
        /// The stop timeout that was waited out.
        timeout: Duration,
    },
}

/// What [`Supervisor::spawn`] hands back: the handle that stops the driver, the latest-state
/// [`watch`] receiver, and the ordered transition-log [`broadcast`] receiver (see this module's
/// top-level "Two observation channels" section for what each promises).
pub type Spawned = (
    Supervisor,
    watch::Receiver<LifecycleState>,
    broadcast::Receiver<LifecycleState>,
);

/// A running `llama-server` lifecycle. [`Supervisor::spawn`] starts the background driver task and
/// hands back this handle plus both observation channels (see [`Spawned`]). The [`watch::Receiver`]
/// always hands a new subscriber the current value first, so cloning it later never misses the fact
/// that, say, the child is already `Ready`; the [`broadcast::Receiver`] is the complete, ordered
/// transition log.
pub struct Supervisor {
    stop_tx: mpsc::UnboundedSender<()>,
    // `Option`, not a bare `JoinHandle`, purely so `stop` can `.take()` it out through `&mut self`:
    // `stop` needs to `.await` an owned `JoinHandle`, and a type with a `Drop` impl (this one, for
    // the fire-and-forget safety net below) cannot have a field moved out of it directly (`E0509`)
    // — `Option::take` mutates in place instead of partially moving `self`, which sidesteps that
    // restriction without needing `unsafe` or `ManuallyDrop`.
    driver: Option<tokio::task::JoinHandle<DriverExit>>,
    /// The most recent successful launch's [`LaunchRecord`], published by [`run`] the moment each
    /// launch attempt succeeds — see [`Supervisor::last_launch`]. `None` until the first launch
    /// succeeds; a `watch` channel always hands a new subscriber (or, here, a fresh `.borrow()`) its
    /// current value first, so this is never stale by more than the time since the last launch.
    record_rx: watch::Receiver<Option<Arc<LaunchRecord>>>,
    /// Copied from [`LifecycleConfig::stop_timeout`], for [`LifecycleError::StopNotConfirmed`]'s
    /// error message — [`run`] itself already knows this value, but by the time [`Supervisor::stop`]
    /// learns the driver's [`DriverExit`], `run`'s own copy is gone along with the finished task.
    stop_timeout: Duration,
}

/// How the lifecycle driver task ([`run`]) itself ended, distinct from the [`LifecycleState`] it
/// published on the way there. [`Supervisor::stop`] maps this to its own `Result`. Only ever produced
/// by [`terminal_exit`], so it depends on confirmation state alone, never on which branch of [`run`]
/// returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverExit {
    /// Every child this driver ever launched is confirmed gone by the OS: the live one's stop (if
    /// one was running) was confirmed, and so was every earlier child that was retired.
    Finished,
    /// At least one child (the live one, or an earlier retired one) was asked to stop and the OS did
    /// not confirm it gone within the configured timeout, even when asked again at the terminal
    /// state.
    StopUnconfirmed,
}

/// The retired child whose stop the OS did not confirm when [`retire`] asked, if any. At most one,
/// by construction rather than by the restart policy: [`run`] launches nothing while it is set.
/// Before the next launch it is cleared by an exit poll or one more blocking stop, or the driver
/// ends in [`LifecycleState::Failed`] with [`FailureReason::PreviousChildNotStopped`];
/// [`terminal_exit`] asks the OS once more before any terminal state.
type Lingering = Option<Arc<dyn SupervisedChild>>;

/// Publishes every [`LifecycleState`] transition to both channels [`Supervisor::spawn`] returns — see
/// this module's top-level "Two observation channels" section for what each promises. The one place
/// in this module that touches either sender, so the two can never be sent to independently and drift
/// out of step with each other.
struct Publisher {
    state: watch::Sender<LifecycleState>,
    log: broadcast::Sender<LifecycleState>,
}

impl Publisher {
    /// Publish one transition. `broadcast::Sender::send` errors only when there are currently no
    /// receivers at all — an ordinary state (nobody has subscribed to the log, or every subscriber
    /// dropped its receiver), not a reason to stop publishing to the `watch` channel this module's
    /// other guarantees still depend on — so that error is deliberately discarded here, exactly as
    /// the `watch` send already was.
    fn publish(&self, state: LifecycleState) {
        let _ = self.log.send(state.clone());
        self.state.send_replace(state);
    }
}

/// Capacity of the ordered transition log (see this module's top-level "Two observation channels"
/// section). A subscriber that falls more than this many transitions behind the driver gets
/// [`tokio::sync::broadcast::error::RecvError::Lagged`] instead of silently missing states. Sized
/// generously above anything a real restart storm produces in the time it takes a receiver to poll
/// once — this crate's own state-machine tests, which read every transition back, are the tightest
/// consumer this bound has to satisfy.
const TRANSITION_LOG_CAPACITY: usize = 256;

impl Supervisor {
    /// Launch and supervise a `llama-server` per `config`. See [`LifecycleConfig`] for what it
    /// controls.
    ///
    /// Must be called from inside a Tokio runtime, since the driver runs as a task on it; outside
    /// one this returns [`LifecycleError::NoRuntime`] rather than panicking the way a bare
    /// `tokio::spawn` would. An `api_key` that is not a valid header value is refused with
    /// [`LifecycleError::InvalidApiKey`] before anything is launched, rather than surfacing only once
    /// the first readiness probe fails to build.
    pub fn spawn(config: LifecycleConfig) -> Result<Spawned, LifecycleError> {
        let launcher: Arc<dyn Launcher> = Arc::new(RealLauncher::new(
            config.launch_spec.clone(),
            config.extra_flags.clone(),
            config.spawn_options.clone(),
        ));
        Self::spawn_with(launcher, config)
    }

    /// [`Supervisor::spawn`] with an injectable [`Launcher`] — the seam this crate's own tests use to
    /// drive the state machine against a fake, never a real OS process. `pub(crate)`, not `pub`: a
    /// [`Launcher`] existing at all is what makes "every restart re-preflights" true (see this
    /// module's top-level docs); letting an outside caller substitute one would let that guarantee be
    /// silently opted out of.
    pub(crate) fn spawn_with(
        launcher: Arc<dyn Launcher>,
        config: LifecycleConfig,
    ) -> Result<Spawned, LifecycleError> {
        // Both checked before any channel exists, so a refusal leaves nothing half-built behind.
        // The exact header string `readiness::probe_health` will send, so the two cannot disagree.
        if hyper::header::HeaderValue::from_str(&format!("Bearer {}", config.api_key)).is_err() {
            return Err(LifecycleError::InvalidApiKey);
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|e| LifecycleError::NoRuntime(e.to_string()))?;
        let (state_tx, state_rx) = watch::channel(LifecycleState::Starting);
        // Created, and so already subscribed, before `tokio::spawn(run(..))` runs: a `broadcast`
        // receiver only ever sees transitions sent after it was created, so subscribing any later
        // than this could miss a transition a very fast driver publishes before the caller gets
        // around to calling this function's own return value's methods.
        let (log_tx, log_rx) = broadcast::channel(TRANSITION_LOG_CAPACITY);
        let (record_tx, record_rx) = watch::channel(None);
        let (stop_tx, stop_rx) = mpsc::unbounded_channel();
        let configured_port = config.launch_spec.port;
        let stop_timeout = config.stop_timeout;
        let publisher = Publisher {
            state: state_tx,
            log: log_tx,
        };
        let driver = runtime.spawn(run(
            launcher,
            configured_port,
            config.api_key,
            config.readiness_deadline,
            config.readiness_poll_interval,
            config.child_poll_interval,
            config.restart_policy,
            stop_timeout,
            publisher,
            record_tx,
            stop_rx,
        ));
        Ok((
            Self {
                stop_tx,
                driver: Some(driver),
                record_rx,
                stop_timeout,
            },
            state_rx,
            log_rx,
        ))
    }

    /// The [`LaunchRecord`] for the most recent launch attempt that actually succeeded (preflighted
    /// and spawned; it need not have reached [`LifecycleState::Ready`]), or `None` if none has yet.
    /// This is where the owner rule's "recorded as a structured warning in the launch record" lands:
    /// a warning for a protection turned off, from either preflight or spawn, is always reachable
    /// here once the launch it belongs to has happened, never only logged and forgotten.
    pub fn last_launch(&self) -> Option<Arc<LaunchRecord>> {
        self.record_rx.borrow().clone()
    }

    /// Ask the driver to stop. During [`LifecycleState::Backoff`] this cancels the pending restart
    /// and moves straight to [`LifecycleState::Stopped`]; otherwise, if a child is running, it is
    /// stopped (via its job object, tearing down the whole assigned tree, when one exists; via a
    /// direct kill of the immediate process otherwise — see
    /// [`crate::protections::Protections::contain_in_job_object`]) and waited on (up to
    /// [`LifecycleConfig::stop_timeout`]) before [`LifecycleState::Stopped`] is published. That wait
    /// runs off the async driver task (`tokio::task::spawn_blocking`; see `stop_child` below), never
    /// inline, so it cannot stall anything else sharing this executor for up to `stop_timeout`. Waits
    /// for the driver task to actually finish before returning.
    ///
    /// Returns `Ok(())` only when every child this driver launched is confirmed gone by the OS: the
    /// running one (if any) and every earlier child retired after a crash or failed readiness check
    /// — never merely that [`LifecycleState::Stopped`] was published, which happens either way. The
    /// same holds after the driver already ended on its own in [`LifecycleState::Failed`].
    /// [`LifecycleError::StopNotConfirmed`] means the OS did not confirm some stop within
    /// `stop_timeout`; the caller must not assume the old child's port or VRAM is free.
    pub async fn stop(mut self) -> Result<(), LifecycleError> {
        let _ = self.stop_tx.send(());
        match self.driver.take() {
            Some(driver) => match driver.await {
                Ok(DriverExit::Finished) => Ok(()),
                Ok(DriverExit::StopUnconfirmed) => Err(LifecycleError::StopNotConfirmed {
                    timeout: self.stop_timeout,
                }),
                Err(e) => Err(LifecycleError::DriverPanicked(e.to_string())),
            },
            // Only reachable if `stop` were somehow called twice, which its `self`-consuming
            // signature already makes impossible from outside this module.
            None => Ok(()),
        }
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        // Fire-and-forget: `Drop` cannot `.await` the driver's own shutdown. This exists only so a
        // `Supervisor` dropped without an explicit `stop` does not leave a supervised child running
        // forever with nothing left able to reach it — a caller that needs to *know* the child is
        // gone before proceeding must still call `stop`.
        let _ = self.stop_tx.send(());
    }
}

/// Map a completed launch attempt (including a panic inside the `spawn_blocking` task it ran on)
/// down to a plain `Result<Launched, FailureReason>`, so [`run`]'s one call site for "the launch
/// itself failed" does not have to re-derive this five-way match.
fn launch_outcome(
    result: Result<Result<Launched, LaunchError>, tokio::task::JoinError>,
) -> Result<Launched, FailureReason> {
    match result {
        Err(join_err) => Err(FailureReason::SpawnFailed(format!(
            "launch task panicked: {join_err}"
        ))),
        Ok(Ok(launched)) => Ok(launched),
        // Classified here, while the typed error still exists: past this point it is only text.
        Ok(Err(LaunchError::Preflight(e))) if e.is_permanent() => {
            Err(FailureReason::PreflightRefusedPermanently {
                message: e.to_string(),
                protection: e.governing_protection(),
            })
        }
        Ok(Err(LaunchError::Preflight(e))) => Err(FailureReason::PreflightRefused {
            message: e.to_string(),
            protection: e.governing_protection(),
        }),
        Ok(Err(LaunchError::Argv(e))) => Err(FailureReason::ArgvRejected(e.to_string())),
        Ok(Err(LaunchError::Spawn(e))) => Err(FailureReason::SpawnFailed(e.to_string())),
        Ok(Err(LaunchError::Child(msg))) => Err(FailureReason::SpawnFailed(msg)),
    }
}

/// Stop `child` off the async executor and report whether the OS confirmed it gone.
///
/// `child.stop` (see [`SupervisedChild::stop`]) is synchronous and can block for up to its
/// `timeout` argument — this module's own rule (see its top-level docs on why [`Launcher`] and
/// [`SupervisedChild`] are injectable at all) is that synchronous, potentially-slow work never runs
/// inline on the driver task, exactly the reason preflight-and-spawn already goes through
/// `tokio::task::spawn_blocking` in [`run`]'s main loop. Calling `child.stop` directly on this task
/// instead — as an earlier version of every call site below did — would stall every other future
/// sharing a `current_thread` runtime (readiness polling, other supervised launches, a UI) for the
/// whole wait.
///
/// A panicked or otherwise failed blocking task is folded into `false`, the same "not confirmed"
/// outcome a genuine timeout produces: either way, the caller cannot assume the old child (and
/// whatever port it held) is actually gone.
async fn stop_child(child: Arc<dyn SupervisedChild>, timeout: Duration) -> bool {
    tokio::task::spawn_blocking(move || child.stop(timeout))
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or(false)
}

/// Tear down a launch that will not be supervised any further, before [`run`] records its failure
/// and waits out a backoff delay: whatever the reason the launch is being abandoned (an exit, a
/// failed readiness check, or an exit poll the OS could not answer), its process tree must not keep
/// holding VRAM or its port through the restart.
///
/// If the OS does not confirm the stop, the child becomes the [`Lingering`] one. [`run`] then
/// launches nothing until [`confirm_lingering_gone`] confirms it gone, so no second server is ever
/// started beside it, and the driver's terminal [`DriverExit`] still reports it (see
/// [`terminal_exit`]) rather than forgetting it once this launch is abandoned.
async fn retire(child: &Arc<dyn SupervisedChild>, timeout: Duration, lingering: &mut Lingering) {
    if !stop_child(Arc::clone(child), timeout).await {
        // `run` clears `lingering` before every launch (or ends), and this runs at most once per
        // launch, so nothing is ever overwritten here.
        *lingering = Some(Arc::clone(child));
    }
}

/// Whether the [`Lingering`] child, if any, is now confirmed gone, asked before the next launch: a
/// cheap exit poll first (a child that exited on its own during the backoff is not stopped again),
/// then one more blocking stop. `true` clears `lingering`; `false` leaves it set, and [`run`] then
/// ends in [`LifecycleState::Failed`] instead of launching a second server beside it.
async fn confirm_lingering_gone(lingering: &mut Lingering, timeout: Duration) -> bool {
    let Some(child) = lingering.as_ref() else {
        return true;
    };
    let gone =
        matches!(child.poll_exit(), Ok(Some(_))) || stop_child(Arc::clone(child), timeout).await;
    if gone {
        *lingering = None;
    }
    gone
}

/// The one way [`run`] ends: [`DriverExit::Finished`] only if `current_confirmed` (the live child's
/// stop was confirmed, or there was no live child) and the lingering child, if any, is now confirmed
/// gone as well.
///
/// The caller publishes the terminal state *before* calling this, so a `watch`-only observer is not
/// held up by the re-confirmation below, which can block for up to `timeout` (off the async
/// executor; see [`stop_child`]). The lingering child is asked even when the live one was not
/// confirmed, so each gets its kill request.
async fn terminal_exit(
    lingering: &mut Lingering,
    timeout: Duration,
    current_confirmed: bool,
) -> DriverExit {
    let mut confirmed = current_confirmed;
    if let Some(child) = lingering.take() {
        confirmed &= stop_child(child, timeout).await;
    }
    if confirmed {
        DriverExit::Finished
    } else {
        DriverExit::StopUnconfirmed
    }
}

/// Poll `child` until it exits, sleeping `poll_interval` between checks. Never returns `Ok` until
/// the child is actually gone; a poll failure at the OS level is mapped to
/// [`FailureReason::ChildPollFailed`] rather than propagated as some other error type, since by the
/// time this is called the only thing [`run`] can still do with the answer is decide a
/// [`LifecycleState::Crashed`] reason.
async fn watch_for_exit(child: &dyn SupervisedChild, poll_interval: Duration) -> Result<ExitInfo, FailureReason> {
    loop {
        match child.poll_exit() {
            Ok(Some(info)) => return Ok(info),
            Ok(None) => {}
            Err(e) => return Err(FailureReason::ChildPollFailed(e.to_string())),
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Whether a stop has been requested and not yet acted on, without waiting. A disconnected channel
/// counts as a stop too: it means every [`Supervisor`] handle is gone (the `select!`s in [`run`]
/// already treat `recv()` returning `None` the same way), and nothing could ever ask this driver to
/// stop again.
///
/// Checked before every transition [`run`] and [`handle_failure`] would publish after a launch
/// attempt ends (the loop top, and right after `Crashed`), not only raced in a few `select!`s, so a
/// stop that lands during the uncancellable `spawn_blocking` launch or a retire's blocking stop is
/// never followed by a `Starting`, `Backoff` or `Failed`.
fn stop_pending(stop_rx: &mut mpsc::UnboundedReceiver<()>) -> bool {
    !matches!(stop_rx.try_recv(), Err(mpsc::error::TryRecvError::Empty))
}

/// Record `reason` as [`LifecycleState::Crashed`], then decide and carry out what comes next: the
/// one immediate new-port retry for [`FailureReason::PortBindFailure`], an ordinary backed-off
/// restart, or [`LifecycleState::Failed`] once the restart policy's window is exceeded. A stop
/// already pending when `Crashed` is published (it arrived while the attempt was in flight), or
/// requested while this function is waiting out a backoff delay, publishes
/// [`LifecycleState::Stopped`] instead of any of those.
///
/// Returns [`ControlFlow::Break`] once a terminal state ([`LifecycleState::Stopped`] or
/// [`LifecycleState::Failed`]) has already been published — [`run`] returns immediately in that
/// case — or [`ControlFlow::Continue`] once it is time to attempt [`LifecycleState::Starting`] again.
#[allow(clippy::too_many_arguments)]
async fn handle_failure(
    reason: FailureReason,
    port_override: &mut Option<PortRequest>,
    port_retry_used: &mut bool,
    window: &mut RestartWindow,
    policy: &RestartPolicy,
    start: tokio::time::Instant,
    publisher: &Publisher,
    stop_rx: &mut mpsc::UnboundedReceiver<()>,
) -> ControlFlow<()> {
    // No `yield_now` needed here, or anywhere else in this module, any more: `Publisher::publish`
    // also sends to the ordered, bounded broadcast log (see this module's top-level "Two observation
    // channels" section), which never overwrites a value the way `watch` does, and `Backoff` now
    // carries `reason` itself so even a `watch`-only observer that lands here without ever seeing
    // `Crashed` still knows why.
    publisher.publish(LifecycleState::Crashed(reason.clone()));

    // A stop that arrived while this attempt was in flight (preflight hashing, the spawn, or a
    // retire's blocking stop) outranks every retry decision below: the docs promise
    // `Starting -> Stopped` while launching, whatever the in-flight attempt then failed with. The
    // failure itself stays visible as the `Crashed` just published.
    if stop_pending(stop_rx) {
        publisher.publish(LifecycleState::Stopped);
        return ControlFlow::Break(());
    }

    // Every retry would fail the same way, so waiting out backoff delays until the restart window
    // gives up would only waste time and report the failure late.
    if reason.is_configuration_error() {
        publisher.publish(LifecycleState::Failed(reason));
        return ControlFlow::Break(());
    }

    if reason.is_port_bind_failure() && !*port_retry_used {
        // The one immediate retry documented on `FailureReason::PortBindFailure`: preflight's own
        // port check cannot close the race between "observed free" and the real bind, so this does
        // not count against the restart window or wait out a backoff delay — it goes straight back
        // to `Starting` with a fresh OS-chosen port — for this one retry only (`port_override` is
        // consumed by the next attempt; the configured port is never overwritten).
        *port_retry_used = true;
        *port_override = Some(PortRequest::Any);
        return ControlFlow::Continue(());
    }

    let count = window.record(policy, start.elapsed());
    if count > policy.max_restarts {
        publisher.publish(LifecycleState::Failed(reason));
        return ControlFlow::Break(());
    }

    let delay = backoff_delay(policy, count);
    publisher.publish(LifecycleState::Backoff {
        attempt: count,
        delay,
        reason,
    });
    tokio::select! {
        biased;
        _ = stop_rx.recv() => {
            publisher.publish(LifecycleState::Stopped);
            return ControlFlow::Break(());
        }
        _ = tokio::time::sleep(delay) => {}
    }
    ControlFlow::Continue(())
}

/// The state machine itself. Runs until [`LifecycleState::Stopped`] or [`LifecycleState::Failed`] is
/// published, at which point the [`tokio::task::JoinHandle`] [`Supervisor::spawn_with`] wraps this in
/// finishes, resolving to a [`DriverExit`] that [`Supervisor::stop`] maps to its own `Result`.
#[allow(clippy::too_many_arguments)]
async fn run(
    launcher: Arc<dyn Launcher>,
    configured_port: PortRequest,
    api_key: String,
    readiness_deadline: Duration,
    readiness_poll_interval: Duration,
    child_poll_interval: Duration,
    restart_policy: RestartPolicy,
    stop_timeout: Duration,
    publisher: Publisher,
    record_tx: watch::Sender<Option<Arc<LaunchRecord>>>,
    mut stop_rx: mpsc::UnboundedReceiver<()>,
) -> DriverExit {
    let client = readiness::build_client();
    let start = tokio::time::Instant::now();
    let mut window = RestartWindow::new();
    let mut port_retry_used = false;
    // What the user configured (`configured_port`, never mutated) versus what the next attempt alone
    // must use instead: set by `handle_failure` (bind-failure retry) and `take()`n by the very next
    // launch attempt, so it can never outlive that one attempt.
    let mut port_override: Option<PortRequest> = None;
    // The retired child the OS did not confirm gone, if any; see `Lingering` and `terminal_exit`.
    let mut lingering: Lingering = None;

    // A tiny local macro, not a helper fn: `handle_failure`'s five call sites below all need to
    // borrow the same handful of locals (`port_override`, `window`, `stop_rx`, …) and then either
    // `return` or `continue` out of *this* function's loop depending on the answer — neither
    // of which a called fn can do on `run`'s behalf. Spelling out the five-line match at each site
    // instead would risk one of them drifting (e.g. forgetting the `continue`), which a macro makes
    // structurally impossible.
    //
    // Defined here, after every local it references — `macro_rules!` hygiene resolves a free
    // identifier inside the macro body against what is in scope *at the macro's own definition
    // site*, not merely "somewhere later in the same block", so each of `client`, `start`, `window`,
    // `port_retry_used`, `port_override` and `lingering` has to be bound above this point for the
    // macro to see it.
    macro_rules! fail {
        ($reason:expr) => {{
            match handle_failure(
                $reason,
                &mut port_override,
                &mut port_retry_used,
                &mut window,
                &restart_policy,
                start,
                &publisher,
                &mut stop_rx,
            )
            .await
            {
                // Every `ControlFlow::Break` `handle_failure` can produce has already published
                // `Failed` or `Stopped`, and happens with no *live* child: the one that failed was
                // retired before this call. A retire the OS did not confirm left that child in
                // `lingering`, so the exit is still derived from confirmation state, not assumed.
                ControlFlow::Break(()) => {
                    return terminal_exit(&mut lingering, stop_timeout, true).await
                }
                ControlFlow::Continue(()) => continue,
            }
        }};
    }

    // Whether the next loop iteration is the very first launch attempt. The `watch` channel is
    // already created holding `LifecycleState::Starting` (see `Supervisor::spawn_with`), and
    // `watch::Sender::send` marks a receiver "changed" on every call regardless of whether the value
    // actually differs — so re-sending `Starting` on this first iteration would not be a no-op, it
    // would be an extra, content-free wakeup a `receiver.changed().await` caller has no way to tell
    // apart from a real transition. Every later iteration (after `Backoff`, or an immediate
    // bind-failure retry) *is* a real `Backoff -> Starting` transition and does need to publish it.
    let mut first_launch = true;

    loop {
        // A stop requested while idle between restarts is honoured here, before anything else this
        // iteration does: before `confirm_lingering_gone` (which can block for up to `stop_timeout`
        // on a retired child) and before `Starting` is re-published, so no transition follows a stop
        // (see `stop_pending`). An in-flight `spawn_blocking` launch below is not itself cancelled
        // once started (see this module's top-level docs); `handle_failure` checks again after it.
        // No child is running at this point in the loop, so there is nothing for the OS to confirm
        // stopped beyond the lingering one, which `terminal_exit` asks about.
        if stop_pending(&mut stop_rx) {
            publisher.publish(LifecycleState::Stopped);
            return terminal_exit(&mut lingering, stop_timeout, true).await;
        }

        if first_launch {
            first_launch = false;
        } else {
            // Never launch beside a retired child the OS has not confirmed gone: it may still hold
            // VRAM and its port. Asked once more here, after the backoff delay gave it time to go;
            // see this module's "Every abandoned launch is stopped before the next one".
            if !confirm_lingering_gone(&mut lingering, stop_timeout).await {
                publisher.publish(LifecycleState::Failed(
                    FailureReason::PreviousChildNotStopped,
                ));
                return terminal_exit(&mut lingering, stop_timeout, true).await;
            }
            // No `yield_now` needed here any more — see `handle_failure`'s own comment on the same
            // change. `Publisher::publish` also sends to the ordered, bounded broadcast log, which a
            // fast-resolving next transition cannot silently overwrite the way `watch` alone could.
            publisher.publish(LifecycleState::Starting);
        }

        // Preflight (file hashing, DACL audit) and the OS-level spawn are both synchronous,
        // potentially slow work; running them inline on this task would stall every other future
        // sharing this executor thread for as long as they take. `spawn_blocking` moves that work to
        // Tokio's blocking thread pool, which is what makes this otherwise-synchronous crate usable
        // from an async supervisor at all.
        // The override, when one is pending, applies to this attempt and is gone after it.
        let attempt_port = port_override.take().unwrap_or(configured_port);
        let launcher = Arc::clone(&launcher);
        let launch_result =
            tokio::task::spawn_blocking(move || launcher.launch(attempt_port)).await;

        let launched = match launch_outcome(launch_result) {
            Ok(launched) => launched,
            Err(reason) => fail!(reason),
        };
        // Published the moment the launch itself succeeds — not deferred until `Ready` — so a
        // caller can see a `ProtectionDisabled`/protection-warning entry (or simply the port) even
        // for a launch that never becomes ready. See `Supervisor::last_launch`.
        // `send_replace`, not `send`: `watch::Sender::send` discards the value when no receiver is
        // alive, which would leave a stale record behind for any receiver created later.
        record_tx.send_replace(Some(Arc::new(launched.record.clone())));

        /// One post-launch race outcome: a stop request, the child exiting, or the readiness probe
        /// concluding one way or the other.
        enum Event {
            Stop,
            Exited(Result<ExitInfo, FailureReason>),
            Ready(Result<(), readiness::ReadinessError>),
        }

        // Readiness sends nothing until this reports the listener as the child's own; see
        // `readiness::wait_ready` and this module's "Readiness trusts only the child's own listener".
        let listener_owner = || launched.child.listener_owner(launched.port);
        let event = tokio::select! {
            biased;
            _ = stop_rx.recv() => Event::Stop,
            exit = watch_for_exit(launched.child.as_ref(), child_poll_interval) => Event::Exited(exit),
            ready = readiness::wait_ready(
                &client,
                preflight::LOOPBACK_HOST,
                launched.port,
                &api_key,
                &listener_owner,
                readiness_deadline,
                readiness_poll_interval,
            ) => Event::Ready(ready),
        };

        match event {
            Event::Stop => {
                let confirmed = stop_child(Arc::clone(&launched.child), stop_timeout).await;
                publisher.publish(LifecycleState::Stopped);
                return terminal_exit(&mut lingering, stop_timeout, confirmed).await;
            }

            Event::Exited(Ok(info)) => {
                let lines = launched.child.recent_lines();
                let reason = classify::classify_exit_before_ready(launched.port, info.code, &lines);
                // The immediate child is already gone, but its job (and any grandchild) is not: tear
                // it down now, before `fail!` waits out a backoff delay that could otherwise leave a
                // crashed launch's process tree holding VRAM or its port for no reason. If the OS
                // does not confirm it, `retire` keeps it as the lingering child, and nothing is
                // launched beside it until it is confirmed gone.
                retire(&launched.child, stop_timeout, &mut lingering).await;
                drop(launched);
                fail!(reason)
            }

            Event::Exited(Err(reason)) => {
                // The OS could not report whether the child exited, so it may well still be running
                // and holding its port and VRAM: retire it exactly like a child known to have exited,
                // before the backoff wait, rather than leaving it orphaned by the next launch.
                retire(&launched.child, stop_timeout, &mut lingering).await;
                drop(launched);
                fail!(reason)
            }

            Event::Ready(Ok(())) => {
                // A retry earned by a bind-failure race only ever applies to the *next* failure, not
                // forever: once a launch reaches readiness, whatever bound the port is proven fine,
                // so a future bind failure deserves its own retry rather than being folded into a
                // stale flag left over from several restarts ago.
                port_retry_used = false;
                publisher.publish(LifecycleState::Ready);

                let post_ready = tokio::select! {
                    biased;
                    _ = stop_rx.recv() => None,
                    exit = watch_for_exit(launched.child.as_ref(), child_poll_interval) => Some(exit),
                };
                match post_ready {
                    None => {
                        let confirmed = stop_child(Arc::clone(&launched.child), stop_timeout).await;
                        publisher.publish(LifecycleState::Stopped);
                        return terminal_exit(&mut lingering, stop_timeout, confirmed).await;
                    }
                    Some(Ok(info)) => {
                        let lines = launched.child.recent_lines();
                        let reason = FailureReason::CrashedAfterReady {
                            exit_code: info.code,
                            last_lines: lines,
                        };
                        // Same reasoning as the pre-ready `Event::Exited(Ok(_))` arm above: close the
                        // job tree before the backoff wait, not after.
                        retire(&launched.child, stop_timeout, &mut lingering).await;
                        drop(launched);
                        fail!(reason)
                    }
                    Some(Err(reason)) => {
                        // See the pre-ready `Event::Exited(Err(_))` arm: a failed poll says nothing
                        // about whether the child is gone, so it is retired before the backoff wait.
                        retire(&launched.child, stop_timeout, &mut lingering).await;
                        drop(launched);
                        fail!(reason)
                    }
                }
            }

            Event::Ready(Err(readiness_err)) => {
                // The deadline (or a 401, an unexpected status, a foreign listener) and the child
                // exiting on its own can race each other; prefer the more specific exit-based classification when the child is
                // actually gone. Whichever reason wins, the launch is retired before the backoff wait:
                // a child still running after failing its readiness check must not keep serving (or
                // holding its port) through the restart, an exited child's job tree may still hold a
                // grandchild, and a failed poll proves nothing either way.
                let reason = match launched.child.poll_exit() {
                    Ok(Some(info)) => {
                        let lines = launched.child.recent_lines();
                        classify::classify_exit_before_ready(launched.port, info.code, &lines)
                    }
                    // A 401 here is always our own child's: `wait_ready` sends nothing until the
                    // listener is attributed to the child's process, and a foreign listener is the
                    // port race the one fresh-port retry exists for.
                    Ok(None) => classify::classify_readiness_error(readiness_err, launched.port),
                    Err(e) => FailureReason::ChildPollFailed(e.to_string()),
                };
                retire(&launched.child, stop_timeout, &mut lingering).await;
                drop(launched);
                fail!(reason)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use listener::ListenerOwner;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Shared, mutable state for one [`FakeChild`], so a test can flip it to "exited" from outside
    /// the driver task without holding a `&mut` the driver itself also needs.
    struct FakeChildState {
        exit: Option<ExitInfo>,
        lines: Vec<String>,
        /// What [`FakeChild::stop`] reports as its confirmation outcome — `true` in every existing
        /// script (matching the old, unconditional `Ok(true)`), overridden to `false` by
        /// [`Script::KeepsRunningStopFails`] to exercise the "not confirmed" path.
        stop_confirms: bool,
        /// A rendezvous [`FakeChild::stop`] blocks its calling thread on before returning — `None`
        /// (the default) for every script, set only by the `stop_child` test that proves that
        /// blocking wait never runs on the async driver task. See [`StopGate`].
        stop_gate: Option<StopGate>,
        /// Set by [`FakeChild::stop`] the moment it runs. A test observing this from outside the
        /// driver task can prove `stop` was called *before* some later state is observed, not merely
        /// that it eventually ran at all.
        stopped: bool,
        /// Makes [`FakeChild::poll_exit`] fail at the "OS level" until the child is stopped — the
        /// shape of a real `try_wait` error, which says nothing about whether the child is gone.
        poll_fails: bool,
        /// How many times [`FakeChild::stop`] has been called, so a test can prove a child already
        /// known gone is not stopped (and waited on) a second time.
        stop_calls: u32,
        /// Answers [`FakeChild::listener_owner`] gives first, one per call, oldest first; once
        /// empty, `owner` answers. Lets a test script "nobody listens yet" for a few polls.
        owners: std::collections::VecDeque<ListenerOwner>,
        /// What [`FakeChild::listener_owner`] answers once `owners` is exhausted: `Ours` by default,
        /// because the fake servers these tests run live in the test process and stand in for the
        /// child's own listener.
        owner: ListenerOwner,
        /// Set the first time [`FakeChild::listener_owner`] answers `Ours`, so a fake server can
        /// prove no request reached it before ownership was established.
        ours_reported: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Default for FakeChildState {
        fn default() -> Self {
            Self {
                exit: None,
                lines: Vec::new(),
                stop_confirms: true,
                stop_gate: None,
                stopped: false,
                poll_fails: false,
                stop_calls: 0,
                owners: std::collections::VecDeque::new(),
                owner: ListenerOwner::Ours,
                ours_reported: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }
        }
    }

    /// Makes [`FakeChild::stop`] block, the way a real `SupervisedChild::stop`'s OS wait does, until
    /// a *different task on the same runtime* releases it. `stop` signals `entered` once it is
    /// blocking, then waits (bounded) on `release`, recording in `released_by_peer` whether the
    /// release actually arrived. On a `current_thread` runtime the releasing task can only run if
    /// `stop` is blocking some other OS thread, so `released_by_peer` is a deterministic, load-proof
    /// witness of where the blocking call ran, with no wall-clock threshold to tune.
    struct StopGate {
        entered: Option<tokio::sync::oneshot::Sender<()>>,
        release: std::sync::mpsc::Receiver<()>,
        released_by_peer: Arc<std::sync::atomic::AtomicBool>,
    }

    struct FakeChild(Arc<Mutex<FakeChildState>>);

    impl SupervisedChild for FakeChild {
        fn poll_exit(&self) -> Result<Option<ExitInfo>, LaunchError> {
            let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if state.poll_fails && !state.stopped {
                return Err(LaunchError::Child("fake poll failure".to_string()));
            }
            Ok(state.exit)
        }
        fn recent_lines(&self) -> Vec<String> {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).lines.clone()
        }
        fn stop(&self, _timeout: Duration) -> Result<bool, LaunchError> {
            // A real `SupervisedChild::stop` is synchronous and can block for a while (see
            // `stop_child`'s own docs) — `stop_gate` lets one test stand a genuinely thread-blocking
            // wait in for that, without spawning a real OS process, to prove `stop_child` moves it
            // off the async driver task. Taken out of the lock first so the wait never holds it.
            let gate = self
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .stop_gate
                .take();
            if let Some(mut gate) = gate {
                if let Some(entered) = gate.entered.take() {
                    let _ = entered.send(());
                }
                let released = gate.release.recv_timeout(Duration::from_secs(10)).is_ok();
                gate.released_by_peer
                    .store(released, std::sync::atomic::Ordering::SeqCst);
            }
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            state.stopped = true;
            state.stop_calls += 1;
            // Like the real `kill_and_wait`: confirmed exactly when the child is gone by the end of
            // the wait. A child rigged not to confirm keeps running (no exit recorded), so a later
            // `poll_exit` or `stop` still sees it alive, until a test sets `exit` itself.
            let confirmed = state.stop_confirms || state.exit.is_some();
            if confirmed {
                state.exit.get_or_insert(ExitInfo { code: Some(0) });
            }
            Ok(confirmed)
        }
        fn listener_owner(&self, _port: u16) -> ListenerOwner {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let owner = state.owners.pop_front().unwrap_or(state.owner);
            if owner == ListenerOwner::Ours {
                state
                    .ours_reported
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            owner
        }
    }

    /// A loopback HTTP server that answers every request with `401 Unauthorized`, standing in for a
    /// listener (ours or a foreign one) that rejects the configured key. Returns the bound port.
    async fn spawn_always_unauthorized_server() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake 401 server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf).await;
                let body = "unauthorized: Invalid API Key";
                let response = format!(
                    "HTTP/1.1 401 Unauthorized\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });
        port
    }

    /// A loopback HTTP server that answers every request with `200 OK` and counts the requests
    /// that reached it carrying an `Authorization` header (the bearer key). The count is bumped
    /// before the response is written, so any `Ready` this server caused is already counted.
    /// `ours_reported`, when given, is read as each connection is accepted: a connection arriving
    /// while it is still `false` is counted in the second counter (sent before ownership was
    /// established).
    async fn spawn_counting_ready_server(
        ours_reported: Option<Arc<std::sync::atomic::AtomicBool>>,
    ) -> (
        u16,
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the counting server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        let keyed = Arc::new(AtomicUsize::new(0));
        let early = Arc::new(AtomicUsize::new(0));
        let (keyed_srv, early_srv) = (Arc::clone(&keyed), Arc::clone(&early));
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                if ours_reported
                    .as_ref()
                    .is_some_and(|f| !f.load(Ordering::SeqCst))
                {
                    early_srv.fetch_add(1, Ordering::SeqCst);
                }
                let mut buf = [0u8; 1024];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                if head.contains("\r\nauthorization:") {
                    keyed_srv.fetch_add(1, Ordering::SeqCst);
                }
                let body = "ok";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });
        (port, keyed, early)
    }

    #[tokio::test]
    async fn a_foreign_listener_never_receives_the_key_and_takes_the_fresh_port_retry() {
        // Another process won the port after preflight released it and would answer 200. Trusting
        // it would hand it the bearer key and publish `Ready` for a server that is not ours: the
        // attempt must instead end as the port race, before anything is sent to it.
        let (foreign, keyed, _) = spawn_counting_ready_server(None).await;
        let ready_port = spawn_always_ready_server().await;
        let ours = Arc::new(Mutex::new(FakeChildState {
            owner: ListenerOwner::Foreign,
            ..FakeChildState::default()
        }));
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::KeepsRunningWithState {
                port: foreign,
                state: Arc::clone(&ours),
            },
            Script::KeepsRunning { port: ready_port },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let mut config = test_config(fast_policy());
        config.launch_spec.port = PortRequest::Explicit(foreign);
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: foreign })
        );
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        assert!(
            ours.lock().unwrap_or_else(|e| e.into_inner()).stopped,
            "our still-running child must be stopped before the retry"
        );
        // Reaching `Ready` proves the retry's launch happened before its port is read back.
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);
        supervisor.stop().await.expect("driver task must not panic");
        assert_eq!(
            fake.seen_ports(),
            vec![PortRequest::Explicit(foreign), PortRequest::Any],
            "the retry must go to a fresh OS-chosen port"
        );
        assert_eq!(
            keyed.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the foreign listener must never receive a request carrying the bearer key"
        );
    }

    #[tokio::test]
    async fn nothing_is_sent_until_the_listener_is_ours() {
        // The child has not bound its port for a few polls (and one table read fails), then has:
        // `Ready` must follow, and the server must not be contacted while ownership is unproven.
        let state = Arc::new(Mutex::new(FakeChildState {
            owners: [
                ListenerOwner::NoneYet,
                ListenerOwner::NoneYet,
                ListenerOwner::Unknown,
            ]
            .into_iter()
            .collect(),
            ..FakeChildState::default()
        }));
        let ours_reported = Arc::clone(
            &state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .ours_reported,
        );
        let (port, keyed, early) = spawn_counting_ready_server(Some(ours_reported)).await;
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningWithState {
                port,
                state: Arc::clone(&state),
            }]));
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);
        supervisor.stop().await.expect("driver task must not panic");
        assert!(
            state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .owners
                .is_empty(),
            "every scripted not-yet answer must have been asked before Ready"
        );
        assert_eq!(
            early.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a request was sent before Ours"
        );
        assert!(
            keyed.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "the probe must reach our listener"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_unattributable_listener_is_never_probed_and_times_out() {
        // The TCP table cannot be read for the whole deadline: never assume the listener is ours.
        let (port, keyed, _) = spawn_counting_ready_server(None).await;
        let state = Arc::new(Mutex::new(FakeChildState {
            owner: ListenerOwner::Unknown,
            ..FakeChildState::default()
        }));
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningWithState {
                port,
                state,
            }]));
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        supervisor.stop().await.expect("driver task must not panic");
        assert_eq!(
            keyed.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a listener that cannot be attributed must never be sent the key"
        );
    }

    #[tokio::test]
    async fn a_401_from_our_own_child_still_fails_at_once() {
        // The configuration error stays one: when the listener is our child's, a 401 means the
        // configured key is wrong and no restart can fix it.
        let port = spawn_always_unauthorized_server().await;
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunning { port }]));
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::Unauthorized)
        );
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(FailureReason::Unauthorized)
        );
        let _ = supervisor.stop().await;
    }

    /// One scripted outcome for [`FakeLauncher::launch`].
    enum Script {
        /// Fail the launch itself (no child ever exists).
        LaunchFails,
        /// Spawn a child that has already exited by the time `launch` returns, with the given exit
        /// code and captured output lines.
        ExitsImmediately { exit_code: Option<i32>, lines: Vec<String> },
        /// [`Script::ExitsImmediately`], backed by a caller-supplied [`FakeChildState`] instead of a
        /// fresh one, so a test can observe [`FakeChild::stop`] having run (via `state.stopped`) from
        /// outside the driver task — see
        /// `a_crashed_childs_job_tree_is_stopped_before_the_backoff_delay_is_observed`.
        ExitsImmediatelyWithState {
            exit_code: Option<i32>,
            lines: Vec<String>,
            state: Arc<Mutex<FakeChildState>>,
        },
        /// Spawn a child that keeps running (readiness polling then decides its fate) on `port`.
        KeepsRunning { port: u16 },
        /// [`Script::KeepsRunning`], backed by a caller-supplied [`FakeChildState`] so a test can
        /// rig it (e.g. `poll_fails`) and observe `stopped` from outside the driver task.
        KeepsRunningWithState {
            port: u16,
            state: Arc<Mutex<FakeChildState>>,
        },
        /// [`Script::KeepsRunning`], but the child's `stop` reports `false` (not confirmed gone) —
        /// exercises the lingering-child paths: no launch beside it, and `StopNotConfirmed`.
        KeepsRunningStopFails { port: u16 },
        /// [`Script::KeepsRunning`], but carrying a caller-supplied [`LaunchRecord`] instead of an
        /// empty one — exercises `Supervisor::last_launch` observing a specific set of warnings.
        KeepsRunningWithRecord { port: u16, record: LaunchRecord },
        /// Preflight refuses the launch with this exact error (no child ever exists).
        PreflightRefuses(crate::preflight::PreflightError),
        /// Block the launch (as preflight's hashing does, for up to seconds) until the test releases
        /// `gate`, then carry out `then`. Lets a test land a stop while a launch is in flight.
        BlocksThen { gate: LaunchGate, then: Box<Script> },
    }

    /// Holds [`FakeLauncher::launch`] on its blocking-pool thread: signals `entered` once blocked,
    /// then waits (bounded, so a broken test cannot hang the suite) for `release`.
    struct LaunchGate {
        entered: tokio::sync::oneshot::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
    }

    /// A [`Launcher`] driven entirely by a caller-scripted sequence, so the state machine's tests
    /// exercise every transition without a real OS process — see this module's top-level docs on why
    /// [`Launcher`] exists.
    struct FakeLauncher {
        script: Mutex<std::collections::VecDeque<Script>>,
        /// Every `port` argument [`Launcher::launch`] was actually called with, oldest first — lets
        /// a test prove which [`PortRequest`] `run` chose for a given attempt (e.g. that only the
        /// bind-failure retry switches one attempt to [`PortRequest::Any`]), and how many launches
        /// happened at all.
        seen_ports: Mutex<Vec<PortRequest>>,
    }

    impl FakeLauncher {
        fn new(script: Vec<Script>) -> Self {
            Self {
                script: Mutex::new(script.into_iter().collect()),
                seen_ports: Mutex::new(Vec::new()),
            }
        }

        fn seen_ports(&self) -> Vec<PortRequest> {
            self.seen_ports
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    }

    /// A [`LaunchRecord`] with no warnings for a script that does not care what it carries, at the
    /// given port. The directory, hash and pid fields hold distinct, recognisable values rather than
    /// defaults, so a test comparing whole records still compares something meaningful.
    fn empty_record(port: u16) -> LaunchRecord {
        LaunchRecord {
            preflight_warnings: Vec::new(),
            protection_warnings: Vec::new(),
            port,
            requested_backend_dir: std::path::PathBuf::from(r"C:\requested"),
            backend_dir: std::path::PathBuf::from(r"C:\resolved"),
            backend_hashes: [("llama-server.exe".to_string(), "ab".repeat(32))]
                .into_iter()
                .collect(),
            pid: 4242,
        }
    }

    /// A loopback HTTP server that answers every request with `200 OK`, for tests that need a fake
    /// child to reach [`LifecycleState::Ready`]. Returns the bound port.
    async fn spawn_always_ready_server() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf).await;
                let body = "ok";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });
        port
    }

    impl Launcher for FakeLauncher {
        fn launch(&self, port: PortRequest) -> Result<Launched, LaunchError> {
            self.seen_ports
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(port);
            let mut next = self
                .script
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop_front()
                .unwrap_or(Script::LaunchFails);
            while let Script::BlocksThen { gate, then } = next {
                let _ = gate.entered.send(());
                let _ = gate.release.recv_timeout(Duration::from_secs(10));
                next = *then;
            }
            match next {
                Script::BlocksThen { .. } => unreachable!("unwrapped by the loop above"),
                Script::LaunchFails => Err(LaunchError::Child("fake launch failure".to_string())),
                Script::ExitsImmediately { exit_code, lines } => {
                    let state = Arc::new(Mutex::new(FakeChildState {
                        exit: Some(ExitInfo { code: exit_code }),
                        lines,
                        ..FakeChildState::default()
                    }));
                    Ok(Launched {
                        child: Arc::new(FakeChild(state)),
                        port: 0,
                        record: empty_record(0),
                    })
                }
                Script::ExitsImmediatelyWithState {
                    exit_code,
                    lines,
                    state,
                } => {
                    {
                        let mut s = state.lock().unwrap_or_else(|e| e.into_inner());
                        s.exit = Some(ExitInfo { code: exit_code });
                        s.lines = lines;
                    }
                    Ok(Launched {
                        child: Arc::new(FakeChild(state)),
                        port: 0,
                        record: empty_record(0),
                    })
                }
                Script::KeepsRunning { port } => {
                    let state = Arc::new(Mutex::new(FakeChildState::default()));
                    Ok(Launched {
                        child: Arc::new(FakeChild(state)),
                        port,
                        record: empty_record(port),
                    })
                }
                Script::KeepsRunningWithState { port, state } => Ok(Launched {
                    child: Arc::new(FakeChild(state)),
                    port,
                    record: empty_record(port),
                }),
                Script::KeepsRunningStopFails { port } => {
                    let state = Arc::new(Mutex::new(FakeChildState {
                        stop_confirms: false,
                        ..FakeChildState::default()
                    }));
                    Ok(Launched {
                        child: Arc::new(FakeChild(state)),
                        port,
                        record: empty_record(port),
                    })
                }
                Script::KeepsRunningWithRecord { port, record } => {
                    let state = Arc::new(Mutex::new(FakeChildState::default()));
                    Ok(Launched {
                        child: Arc::new(FakeChild(state)),
                        port,
                        record,
                    })
                }
                Script::PreflightRefuses(error) => Err(LaunchError::Preflight(error)),
            }
        }
    }

    fn test_config(restart_policy: RestartPolicy) -> LifecycleConfig {
        LifecycleConfig {
            launch_spec: LaunchSpec {
                backend_dir: std::path::PathBuf::from(r"C:\unused"),
                tool_name: "llama-server.exe".to_string(),
                manifest: std::collections::BTreeMap::new(),
                model_path: std::path::PathBuf::from(r"C:\unused\model.gguf"),
                draft_model_path: None,
                port: PortRequest::Any,
                api_key_file: std::path::PathBuf::from(r"C:\unused\key.txt"),
                allow_unsafe_dacl: None,
                protections: crate::protections::Protections::default(),
            },
            extra_flags: Vec::new(),
            spawn_options: SpawnOptions {
                temp_dir: std::path::PathBuf::from(r"C:\unused\temp"),
                cache_dir: std::path::PathBuf::from(r"C:\unused\temp"),
                cuda_visible_devices: None,
                log_path: std::path::PathBuf::from(r"C:\unused\log.txt"),
                log_max_bytes: 1024,
            },
            api_key: "test-key-not-a-real-secret-value".to_string(),
            readiness_deadline: Duration::from_secs(2),
            readiness_poll_interval: Duration::from_millis(5),
            child_poll_interval: Duration::from_millis(5),
            restart_policy,
            stop_timeout: Duration::from_secs(1),
        }
    }

    fn fast_policy() -> RestartPolicy {
        RestartPolicy {
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(100),
            max_restarts: 2,
            window: Duration::from_secs(3600),
            jitter_seed: 7,
        }
    }

    /// Read the next transition off the ordered, bounded broadcast log (see this module's top-level
    /// "Two observation channels" section) — what this crate's own state-machine tests assert
    /// against, rather than the latest-value `watch` channel, so a transition a fast driver publishes
    /// in a tight burst can never be silently missed the way it could when these tests polled `watch`
    /// with a scheduler-dependent `yield_now` standing in for a real guarantee.
    async fn next_state(log_rx: &mut broadcast::Receiver<LifecycleState>) -> LifecycleState {
        log_rx
            .recv()
            .await
            .expect("no lag in a 256-slot buffer for these short scripted tests")
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_that_never_exits_and_never_answers_health_times_out_into_backoff() {
        let launcher: Arc<dyn Launcher> = Arc::new(FakeLauncher::new(vec![Script::KeepsRunning {
            port: 65535,
        }]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        // Nothing is listening on 65535, so every readiness probe fails to connect and is treated as
        // "still loading" (see `readiness::ReadinessError::Unreachable`'s docs) until the 2s
        // deadline in `test_config` elapses — advanced here rather than actually waited out, which
        // is the entire point of `start_paused = true` plus an explicit `advance`.
        tokio::time::advance(Duration::from_secs(3)).await;

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_that_exits_immediately_is_classified_and_restarted() {
        let launcher: Arc<dyn Launcher> = Arc::new(FakeLauncher::new(vec![
            Script::ExitsImmediately {
                exit_code: Some(134),
                lines: vec!["fatal: out of memory".to_string()],
            },
            Script::KeepsRunning { port: 65535 },
        ]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady {
                exit_code: Some(134),
                last_lines: vec!["fatal: out of memory".to_string()],
            })
        );
        let backoff = next_state(&mut log_rx).await;
        assert!(matches!(backoff, LifecycleState::Backoff { attempt: 1, .. }));

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_port_bind_failure_is_retried_once_immediately_then_counts_normally() {
        let bind_failure = || Script::ExitsImmediately {
            exit_code: Some(1),
            lines: vec!["srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080".to_string()],
        };
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![bind_failure(), bind_failure()]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 8080 })
        );
        // The first bind failure is retried immediately: no `Backoff` in between.
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);

        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 8080 })
        );
        // The second, in a row, is an ordinary counted restart: `Backoff` follows this time.
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_transition_is_dropped_under_a_multi_thread_runtime() {
        // The scheduler-dependence this proves gone: this module used to rely on two `yield_now`
        // calls that only worked because `#[tokio::test]`'s default `current_thread` runtime happens
        // to poll a just-woken receiver next. Nothing guarantees that ordering on a real multi-thread
        // runtime — deliberately used here, unlike every other test in this module — so this exercises
        // the exact back-to-back-transitions shape (`Crashed` immediately followed by `Starting`, with
        // no `.await` of any real duration in between) that previously depended on it. `start_paused`
        // is not used here: pausing time requires the `current_thread` runtime, and this scenario's
        // one real delay (the backoff after the second failure) is short enough that this test does
        // not need to fast-forward past it.
        let bind_failure = || Script::ExitsImmediately {
            exit_code: Some(1),
            lines: vec![
                "srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080"
                    .to_string(),
            ],
        };
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![bind_failure(), bind_failure()]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 8080 })
        );
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 8080 })
        );
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn exceeding_the_restart_window_moves_to_failed() {
        // `max_restarts: 2` in `fast_policy`, all inside one huge window: the third restart attempt
        // must exceed it.
        let crash = || Script::ExitsImmediately {
            exit_code: Some(1),
            lines: vec!["boom".to_string()],
        };
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![crash(), crash(), crash()]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        for expected_attempt in 1..=2 {
            assert!(matches!(
                next_state(&mut log_rx).await,
                LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
            ));
            assert!(matches!(
                next_state(&mut log_rx).await,
                LifecycleState::Backoff { attempt, .. } if attempt == expected_attempt
            ));
            tokio::time::advance(Duration::from_secs(1)).await;
            assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        }

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(FailureReason::ExitedBeforeReady { .. })
        ));

        drop(rx);
        // The driver already returned on its own once `Failed` was published; `stop` still completes
        // cleanly against a task that has already finished.
        supervisor.stop().await.expect("driver task must not panic even after Failed");
    }

    #[tokio::test(start_paused = true)]
    async fn stop_during_backoff_cancels_the_pending_restart() {
        let crash = || Script::ExitsImmediately {
            exit_code: Some(1),
            lines: vec!["boom".to_string()],
        };
        let launcher: Arc<dyn Launcher> = Arc::new(FakeLauncher::new(vec![crash()]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { .. }
        ));

        // Stop while the backoff delay (100ms cap, well under the 1h window) is still pending: the
        // driver must notice via the `select!` in `handle_failure`, not by waiting the delay out.
        supervisor.stop().await.expect("driver task must not panic");
        assert_eq!(*rx.borrow(), LifecycleState::Stopped);
    }

    #[tokio::test(start_paused = true)]
    async fn a_crashed_childs_job_tree_is_stopped_before_the_backoff_delay_is_observed() {
        // The minor finding this proves: a crashed child's job (and any grandchild it left behind)
        // must be torn down before `fail!` waits out the backoff delay, not after — see the
        // `Event::Exited(Ok(_))` arm's own comment in `run`. `state.stopped` is read the moment
        // `Backoff` is observed, before the (paused, but still logically pending) delay would ever
        // elapse, so this fails if the stop were deferred to the next launch attempt instead.
        let state = Arc::new(Mutex::new(FakeChildState::default()));
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::ExitsImmediatelyWithState {
                exit_code: Some(1),
                lines: vec!["boom".to_string()],
                state: Arc::clone(&state),
            }]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { .. }
        ));

        assert!(
            state.lock().unwrap_or_else(|e| e.into_inner()).stopped,
            "the crashed child must already have been stopped by the time Backoff is observed"
        );

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_whose_exit_poll_fails_is_stopped_before_the_backoff_delay() {
        // A failed `poll_exit` says nothing about whether the child is gone, so it must be retired
        // like any other abandoned launch, not left running (holding its port and VRAM) while the
        // supervisor backs off and launches a second one beside it.
        let state = Arc::new(Mutex::new(FakeChildState {
            poll_fails: true,
            ..FakeChildState::default()
        }));
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningWithState {
                port: 65535,
                state: Arc::clone(&state),
            }]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ChildPollFailed(_))
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        assert!(
            state.lock().unwrap_or_else(|e| e.into_inner()).stopped,
            "a child whose exit poll failed must be stopped before Backoff is observed"
        );

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_that_fails_readiness_is_stopped_before_the_backoff_delay() {
        let state = Arc::new(Mutex::new(FakeChildState::default()));
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningWithState {
                port: 65535,
                state: Arc::clone(&state),
            }]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        // Nothing listens on 65535: readiness runs out its (virtual) 2s deadline.
        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        assert!(
            state.lock().unwrap_or_else(|e| e.into_inner()).stopped,
            "a child that failed readiness must be stopped before Backoff is observed"
        );

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[test]
    fn spawning_outside_a_tokio_runtime_is_a_typed_error_not_a_panic() {
        let fake = Arc::new(FakeLauncher::new(vec![Script::KeepsRunning {
            port: 65535,
        }]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let result = Supervisor::spawn_with(launcher, test_config(fast_policy()));
        assert!(
            matches!(result, Err(LifecycleError::NoRuntime(_))),
            "{:?}",
            result.err()
        );
        assert!(
            fake.seen_ports().is_empty(),
            "nothing may be launched without a runtime"
        );
    }

    #[tokio::test]
    async fn dropping_the_supervisor_without_stop_still_stops_the_child() {
        // `Drop for Supervisor` is the safety net for a caller that never calls `stop`: the driver
        // must still stop the running child and publish `Stopped`, not supervise it forever.
        let state = Arc::new(Mutex::new(FakeChildState::default()));
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningWithState {
                port: 65535,
                state: Arc::clone(&state),
            }]));
        // Real clock: readiness against the unreachable port must not time out into `Crashed`
        // before the drop below is noticed, however slowly a loaded machine schedules this test.
        let mut config = test_config(fast_policy());
        config.readiness_deadline = Duration::from_secs(600);
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");
        // Dropped only once the child exists: dropping earlier would be honoured before the first
        // launch (correctly, but then there is no child whose stop this test could observe).
        tokio::time::timeout(Duration::from_secs(30), async {
            while supervisor.last_launch().is_none() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the first launch must happen");
        drop(supervisor);

        let observed = tokio::time::timeout(Duration::from_secs(30), next_state(&mut log_rx))
            .await
            .expect("the driver must react to the dropped Supervisor");
        assert_eq!(observed, LifecycleState::Stopped);
        assert!(
            state.lock().unwrap_or_else(|e| e.into_inner()).stopped,
            "the running child must be stopped once its Supervisor is dropped"
        );
    }

    #[tokio::test]
    async fn reaching_ready_and_then_stopping_publishes_ready_then_stopped() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf).await;
                let body = "ok";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });

        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunning { port }]));
        // This is the one test in this module that cannot use `start_paused = true` (real sockets
        // need a real clock advancing under them), so unlike every other test here it is exposed to
        // actual wall-clock scheduling delay. `test_config`'s 2s `readiness_deadline` is tuned for
        // the *virtual*-clock tests, where it is instantly and deterministically advanced past;
        // against a real clock it is a race — under `cargo test --workspace`'s full parallel load
        // (every other crate's test binary competing for the same CPUs), this task's single-threaded
        // runtime can go unscheduled for longer than 2s even though the loopback round trip itself is
        // instant, which was previously observed to flip this test's outcome from `Ready` to
        // `Crashed(ReadinessDeadlineExceeded)`. The fake server above answers on the very first poll
        // with no work of its own, so a generous real-time budget costs nothing in the common case
        // and only removes a spurious failure mode under load.
        let mut config = test_config(fast_policy());
        config.readiness_deadline = Duration::from_secs(30);
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);

        supervisor.stop().await.expect("driver task must not panic");
        assert_eq!(*rx.borrow(), LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn stop_after_ready_reports_unconfirmed_when_the_os_never_confirms_the_child_gone() {
        // The `Event::Stop` path's own `DriverExit` this proves: `LifecycleState::Stopped` is
        // published exactly as it is for an ordinary confirmed stop (a `watch`-only observer sees no
        // difference — see this module's top-level "Two observation channels" section), but
        // `Supervisor::stop`'s own `Result` must surface that the OS never actually confirmed the
        // child gone, since the caller cannot otherwise know whether the old port or VRAM is free.
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf).await;
                let body = "ok";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });

        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningStopFails {
                port,
            }]));
        let mut config = test_config(fast_policy());
        config.readiness_deadline = Duration::from_secs(30); // see the real-clock note above.
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);

        let result = supervisor.stop().await;
        assert!(
            matches!(result, Err(LifecycleError::StopNotConfirmed { .. })),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn a_disabled_protection_recorded_at_launch_is_observable_through_last_launch() {
        // The owner rule this proves: a warning for a protection turned off must be reachable
        // through the running `Supervisor`, not merely produced once and dropped — see
        // `Supervisor::last_launch`'s own docs.
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 512];
                let _ = stream.read(&mut buf).await;
                let body = "ok";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });

        let warning = crate::protections::Protections {
            contain_in_job_object: false,
            ..crate::protections::Protections::default()
        }
        .job_containment_warning()
        .expect("job containment is off in this fixture");
        let record = LaunchRecord {
            protection_warnings: vec![warning.clone()],
            pid: 777,
            ..empty_record(port)
        };
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningWithRecord {
                port,
                record,
            }]));

        // No `Supervisor::last_launch()` observation is possible before the first launch, and this
        // test does not need one: the moment `Ready` is published (below), the record for the
        // launch that reached it has already been sent (`run` sends it right after the launch
        // attempt itself succeeds, strictly before readiness is even polled).
        let mut config = test_config(fast_policy());
        config.readiness_deadline = Duration::from_secs(30); // see the real-clock note above.
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);

        let observed = supervisor
            .last_launch()
            .expect("a launch record must be published once a launch has succeeded");
        assert_eq!(observed.port, port);
        assert_eq!(observed.protection_warnings, vec![warning]);
        assert!(observed.preflight_warnings.is_empty());
        assert_eq!(
            observed.pid, 777,
            "the launch's own record, not some default, must be published"
        );

        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_retired_child_still_unconfirmed_after_backoff_fails_instead_of_launching_beside_it()
    {
        // Two servers must never run side by side: a retired child the OS will not confirm gone may
        // still hold its VRAM and port, so once the backoff elapses the driver asks once more and,
        // still unconfirmed, ends in `Failed` without a second launch.
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::KeepsRunningStopFails { port: 65535 },
            Script::KeepsRunning { port: 65535 },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let mut config = test_config(fast_policy());
        config.launch_spec.port = PortRequest::Explicit(65535);
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        // Nothing listens on 65535, so readiness fails at its deadline and the child is retired;
        // this fixture's `stop()` never confirms.
        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(FailureReason::PreviousChildNotStopped),
            "no `Starting` and no second launch while the first child may still be running"
        );
        assert_eq!(
            fake.seen_ports(),
            vec![PortRequest::Explicit(65535)],
            "exactly one launch: nothing may be started beside an unconfirmed child"
        );

        drop(rx);
        let result = supervisor.stop().await;
        assert!(
            matches!(result, Err(LifecycleError::StopNotConfirmed { .. })),
            "{result:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_retired_child_confirmed_on_the_second_ask_relaunches_on_the_configured_port() {
        // The retire right after the failure is not confirmed, but the extra ask after the backoff
        // is: the old child is gone, so the restart proceeds, on the configured explicit port (no
        // port override is needed once the old process is confirmed gone).
        let first = Arc::new(Mutex::new(FakeChildState {
            stop_confirms: false,
            ..FakeChildState::default()
        }));
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::KeepsRunningWithState {
                port: 65535,
                state: Arc::clone(&first),
            },
            Script::KeepsRunning { port: 65535 },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let mut config = test_config(fast_policy());
        config.launch_spec.port = PortRequest::Explicit(65535);
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(_)
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        {
            let mut s = first.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(s.stop_calls, 1, "retired once before the backoff");
            // The next stop will be confirmed, but the child is still running (no exit recorded),
            // so only a real second stop, not the cheap exit poll, can clear it.
            s.stop_confirms = true;
        }

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        // The second child's readiness deadline elapsing proves the second launch happened.
        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(_)
        ));

        assert_eq!(
            first.lock().unwrap_or_else(|e| e.into_inner()).stop_calls,
            2,
            "the second, blocking ask is what confirmed the first child gone"
        );
        assert_eq!(
            fake.seen_ports(),
            vec![PortRequest::Explicit(65535), PortRequest::Explicit(65535)],
            "a confirmed-gone child frees its port, so the configured port is asked for again"
        );
        supervisor
            .stop()
            .await
            .expect("both children are confirmed gone");
    }

    #[tokio::test]
    async fn the_explicit_port_comes_back_after_the_one_bind_failure_retry() {
        // Explicit(p) -> bind failure -> Any (the one retry) -> Ready -> crash -> Backoff ->
        // Explicit(p). Before the override was one-shot, the retry overwrote the configured port and
        // every later restart stayed on a random OS port.
        let ready_port = spawn_always_ready_server().await;
        let ready_child = Arc::new(Mutex::new(FakeChildState::default()));
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::ExitsImmediately {
                exit_code: Some(1),
                lines: vec![
                    "srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 45678"
                        .to_string(),
                ],
            },
            Script::KeepsRunningWithState {
                port: ready_port,
                state: Arc::clone(&ready_child),
            },
            Script::KeepsRunningWithState {
                port: ready_port,
                state: Arc::new(Mutex::new(FakeChildState::default())),
            },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let mut config = test_config(fast_policy());
        config.launch_spec.port = PortRequest::Explicit(45678);
        config.readiness_deadline = Duration::from_secs(30); // real clock; see the note above.
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, config).expect("called inside a tokio runtime");

        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 45678 })
        );
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);

        // The ready child now crashes.
        ready_child.lock().unwrap_or_else(|e| e.into_inner()).exit =
            Some(ExitInfo { code: Some(3) });
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::CrashedAfterReady {
                exit_code: Some(3),
                ..
            })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        // The third launch reaching `Ready` proves `launch` was called a third time.
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Ready);

        assert_eq!(
            fake.seen_ports(),
            vec![
                PortRequest::Explicit(45678),
                PortRequest::Any,
                PortRequest::Explicit(45678)
            ],
            "the bind-failure override must apply to exactly one attempt"
        );
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn an_unconfirmed_retire_makes_a_later_stop_report_unconfirmed() {
        // A child that failed readiness is retired, the OS never confirms it gone, and the driver
        // moves on to `Backoff`. A stop during that backoff has no *live* child to stop, but the
        // retired one may still hold the port and VRAM, so `Supervisor::stop` must not say `Ok`.
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningStopFails {
                port: 65535,
            }]));
        let (supervisor, rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));

        let result = supervisor.stop().await;
        assert!(
            matches!(result, Err(LifecycleError::StopNotConfirmed { .. })),
            "{result:?}"
        );
        assert_eq!(*rx.borrow(), LifecycleState::Stopped);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_after_an_unconfirmed_retire_is_reported_unconfirmed() {
        // `max_restarts: 0`: the first failure is terminal. The driver ends on its own with
        // `Failed`, but the child it retired on the way there was never confirmed gone.
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunningStopFails {
                port: 65535,
            }]));
        let policy = RestartPolicy {
            max_restarts: 0,
            ..fast_policy()
        };
        let (supervisor, _rx, mut log_rx) = Supervisor::spawn_with(launcher, test_config(policy))
            .expect("called inside a tokio runtime");

        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));

        let result = supervisor.stop().await;
        assert!(
            matches!(result, Err(LifecycleError::StopNotConfirmed { .. })),
            "{result:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_lingering_child_that_exits_later_is_pruned_and_never_stopped_again() {
        // The retired child is unconfirmed at first, then exits on its own during the backoff. The
        // next launch prunes it with a cheap `poll_exit`, so the terminal stop neither waits on it
        // again nor reports it unconfirmed.
        let lingering = Arc::new(Mutex::new(FakeChildState {
            stop_confirms: false,
            ..FakeChildState::default()
        }));
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::KeepsRunningWithState {
                port: 65535,
                state: Arc::clone(&lingering),
            },
            Script::KeepsRunning { port: 65535 },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(_)
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        {
            let mut s = lingering.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(s.stop_calls, 1, "retired once before the backoff");
            s.exit = Some(ExitInfo { code: Some(0) });
        }

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(next_state(&mut log_rx).await, LifecycleState::Starting);
        // The second child's readiness deadline elapsing proves the second launch happened.
        tokio::time::advance(Duration::from_secs(3)).await;
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(_)
        ));
        assert_eq!(fake.seen_ports().len(), 2);

        supervisor
            .stop()
            .await
            .expect("the lingering child exited, so nothing is unconfirmed");
        assert_eq!(
            lingering
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .stop_calls,
            1,
            "a child pruned as exited must not be stopped again at terminal exit"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_permanent_preflight_refusal_fails_without_restarting() {
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::PreflightRefuses(crate::preflight::PreflightError::RunningElevated),
            Script::KeepsRunning { port: 65535 },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PreflightRefusedPermanently { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(FailureReason::PreflightRefusedPermanently { .. })
        ));
        supervisor.stop().await.expect("no child ever existed");
        assert_eq!(
            fake.seen_ports().len(),
            1,
            "a permanent refusal must not be retried"
        );
    }

    /// Assert the driver published exactly `Crashed(_)` then `Stopped` and nothing after, that
    /// `Supervisor::stop` reports every child confirmed gone, and that no second launch happened.
    async fn assert_crashed_then_stopped_only(
        label: &str,
        supervisor: Supervisor,
        mut log_rx: broadcast::Receiver<LifecycleState>,
        fake: &FakeLauncher,
    ) -> FailureReason {
        let first = next_state(&mut log_rx).await;
        let LifecycleState::Crashed(reason) = first else {
            panic!("{label}: expected Crashed first, got {first:?}");
        };
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Stopped,
            "{label}: the pending stop must come straight after Crashed"
        );
        supervisor
            .stop()
            .await
            .unwrap_or_else(|e| panic!("{label}: every child must be confirmed gone: {e}"));
        // The driver has ended and dropped its sender: anything still queued would be a transition
        // published after `Stopped`.
        assert!(
            matches!(
                log_rx.try_recv(),
                Err(broadcast::error::TryRecvError::Closed)
            ),
            "{label}: nothing may be published after Stopped"
        );
        assert_eq!(fake.seen_ports().len(), 1, "{label}: no relaunch after a stop");
        reason
    }

    #[tokio::test]
    async fn a_stop_during_an_in_flight_launch_that_then_fails_is_honoured_at_once() {
        // The launch is blocked (as preflight's hashing is, for up to 2.6 s) when the stop lands,
        // then fails. Whatever the failure kind, the user's stop wins: no `Failed` for a
        // configuration error, no `Backoff` for a transient one.
        let cases = [
            (
                "permanent preflight refusal",
                Script::PreflightRefuses(crate::preflight::PreflightError::RunningElevated),
            ),
            ("transient launch failure", Script::LaunchFails),
        ];
        for (label, then) in cases {
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let fake = Arc::new(FakeLauncher::new(vec![
                Script::BlocksThen {
                    gate: LaunchGate {
                        entered: entered_tx,
                        release: release_rx,
                    },
                    then: Box::new(then),
                },
                Script::KeepsRunning { port: 65535 },
            ]));
            let launcher: Arc<dyn Launcher> = fake.clone();
            let (supervisor, _rx, log_rx) =
                Supervisor::spawn_with(launcher, test_config(fast_policy()))
                    .expect("called inside a tokio runtime");

            entered_rx.await.expect("the launch must start");
            // Sent on the handle's own channel, exactly what `Supervisor::stop` sends first, so the
            // stop is provably queued before the launch is released.
            supervisor.stop_tx.send(()).expect("the driver is alive");
            release_tx.send(()).expect("the launch is waiting");

            assert_crashed_then_stopped_only(label, supervisor, log_rx, &fake).await;
        }
    }

    #[tokio::test]
    async fn a_stop_during_a_retire_after_a_port_bind_failure_skips_the_immediate_retry() {
        // The child logged a bind failure and exited; the stop lands while `retire` is blocked in
        // the OS stop. The one immediate fresh-port retry must not run (it used to publish
        // `Starting` before noticing the stop).
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let state = Arc::new(Mutex::new(FakeChildState {
            stop_gate: Some(StopGate {
                entered: Some(entered_tx),
                release: release_rx,
                released_by_peer: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }),
            ..FakeChildState::default()
        }));
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::ExitsImmediatelyWithState {
                exit_code: Some(1),
                lines: vec![
                    "srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080"
                        .to_string(),
                ],
                state,
            },
            Script::KeepsRunning { port: 65535 },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let (supervisor, _rx, log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        entered_rx.await.expect("the retire must reach the OS stop");
        supervisor.stop_tx.send(()).expect("the driver is alive");
        release_tx.send(()).expect("the stop is waiting");

        let reason =
            assert_crashed_then_stopped_only("port-bind failure", supervisor, log_rx, &fake).await;
        assert_eq!(reason, FailureReason::PortBindFailure { port: 8080 });
    }

    #[tokio::test(start_paused = true)]
    async fn fixed_backend_layout_facts_fail_without_backoff() {
        // A backend file name that is not UTF-8, and a reparse point the folder audit finds on the
        // resolved backend path, do not change on a restart: each must reach `Failed` straight
        // from its first `Crashed`, never through `Backoff` (which would re-hash ~1.2 GB per try).
        for error in [
            crate::preflight::PreflightError::BackendIntegrityIo(
                cerebro_probe::ProbeError::BackendFileNameNotUtf8 {
                    dir: r"C:\backend".to_string(),
                    name: "a\u{fffd}.dll".to_string(),
                },
            ),
            crate::preflight::PreflightError::BackendPathRedirected {
                dir: r"C:\backend".to_string(),
                cause: r"C:\mnt\vol is a reparse point".to_string(),
            },
        ] {
            let fake = Arc::new(FakeLauncher::new(vec![
                Script::PreflightRefuses(error.clone()),
                Script::KeepsRunning { port: 65535 },
            ]));
            let launcher: Arc<dyn Launcher> = fake.clone();
            let (supervisor, _rx, mut log_rx) =
                Supervisor::spawn_with(launcher, test_config(fast_policy()))
                    .expect("called inside a tokio runtime");

            assert!(
                matches!(
                    next_state(&mut log_rx).await,
                    LifecycleState::Crashed(FailureReason::PreflightRefusedPermanently { .. })
                ),
                "{error:?}"
            );
            assert!(
                matches!(
                    next_state(&mut log_rx).await,
                    LifecycleState::Failed(FailureReason::PreflightRefusedPermanently { .. })
                ),
                "{error:?} must not back off"
            );
            supervisor.stop().await.expect("no child ever existed");
            assert_eq!(fake.seen_ports().len(), 1, "{error:?} must not be retried");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_model_path_that_is_a_reparse_point_fails_without_backoff() {
        let fake = Arc::new(FakeLauncher::new(vec![
            Script::PreflightRefuses(crate::preflight::PreflightError::ModelIsReparsePoint {
                role: crate::preflight::ModelRole::Target,
                path: r"C:\models\link.gguf".to_string(),
            }),
            Script::KeepsRunning { port: 65535 },
        ]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PreflightRefusedPermanently { .. })
        ));
        // Straight to `Failed`: no `Backoff`, and no second (backend re-hashing) attempt.
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(FailureReason::PreflightRefusedPermanently { .. })
        ));
        supervisor.stop().await.expect("no child ever existed");
        assert_eq!(
            fake.seen_ports().len(),
            1,
            "a permanent refusal must not be retried"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_transient_preflight_refusal_still_restarts() {
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::PreflightRefuses(
                crate::preflight::PreflightError::PortUnavailable {
                    port: 8080,
                    cause: "in use".to_string(),
                },
            )]));
        let (supervisor, _rx, mut log_rx) =
            Supervisor::spawn_with(launcher, test_config(fast_policy()))
                .expect("called inside a tokio runtime");

        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(FailureReason::PreflightRefused { .. })
        ));
        assert!(matches!(
            next_state(&mut log_rx).await,
            LifecycleState::Backoff { attempt: 1, .. }
        ));
        supervisor.stop().await.expect("no child ever existed");
    }

    #[test]
    fn debug_names_every_field_and_never_prints_the_api_key() {
        let mut config = test_config(fast_policy());
        config.api_key = "zz-unique-secret-zz".repeat(2);
        let printed = format!("{config:?}");
        assert!(!printed.contains("zz-unique-secret-zz"), "{printed}");
        assert!(printed.contains("<redacted>"), "{printed}");
        for field in [
            "launch_spec",
            "extra_flags",
            "spawn_options",
            "api_key",
            "readiness_deadline",
            "readiness_poll_interval",
            "child_poll_interval",
            "restart_policy",
            "stop_timeout",
        ] {
            assert!(
                printed.contains(field),
                "Debug must list `{field}`: {printed}"
            );
        }
        let pretty = format!("{config:#?}");
        assert!(!pretty.contains("zz-unique-secret-zz"), "{pretty}");
    }

    #[tokio::test]
    async fn a_header_invalid_api_key_is_refused_before_anything_launches() {
        let fake = Arc::new(FakeLauncher::new(vec![Script::KeepsRunning {
            port: 65535,
        }]));
        let launcher: Arc<dyn Launcher> = fake.clone();
        let mut config = test_config(fast_policy());
        config.api_key = "leaky\r\nX-Injected: 1".to_string();
        let result = Supervisor::spawn_with(launcher, config);
        let Err(err) = result else {
            panic!("a header-invalid api_key must be refused");
        };
        assert!(matches!(err, LifecycleError::InvalidApiKey), "{err:?}");
        assert!(
            !err.to_string().contains("leaky"),
            "the key must not appear: {err}"
        );
        assert!(
            fake.seen_ports().is_empty(),
            "nothing may be launched with an unusable key"
        );
    }

    #[tokio::test]
    async fn a_configuration_error_fails_at_once_without_a_restart() {
        // `Supervisor::spawn` refuses a header-invalid key up front, so this drives
        // `handle_failure` directly with the reason a still-unbuildable readiness request produces.
        let (state_tx, state_rx) = watch::channel(LifecycleState::Starting);
        let (log_tx, mut log_rx) = broadcast::channel(TRANSITION_LOG_CAPACITY);
        let publisher = Publisher {
            state: state_tx,
            log: log_tx,
        };
        let (_stop_tx, mut stop_rx) = mpsc::unbounded_channel();
        let mut port_override = None;
        let mut port_retry_used = false;
        let mut window = RestartWindow::new();
        let reason = FailureReason::InvalidReadinessRequest("invalid HTTP header value".into());

        let flow = handle_failure(
            reason.clone(),
            &mut port_override,
            &mut port_retry_used,
            &mut window,
            &fast_policy(),
            tokio::time::Instant::now(),
            &publisher,
            &mut stop_rx,
        )
        .await;

        assert_eq!(flow, ControlFlow::Break(()));
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Crashed(reason.clone())
        );
        assert_eq!(
            next_state(&mut log_rx).await,
            LifecycleState::Failed(reason)
        );
        assert_eq!(
            *state_rx.borrow(),
            LifecycleState::Failed(FailureReason::InvalidReadinessRequest(
                "invalid HTTP header value".into()
            ))
        );
        assert_eq!(port_override, None);
    }

    #[tokio::test]
    async fn stop_child_blocking_for_a_while_does_not_stall_a_concurrent_task() {
        // The default `#[tokio::test]` runtime is `current_thread`: one OS thread multiplexes every
        // task. `FakeChild::stop` here blocks its thread until `peer` — another task on this same
        // runtime — releases it, which `peer` can only do if that one thread is free while `stop` is
        // blocked. So `released_by_peer` ends up `true` exactly when `stop_child` ran the blocking
        // wait off the driver thread (`spawn_blocking`), and `false` (after the gate's bounded
        // 10s wait) if it ran inline. An earlier version counted wall-clock ticks of a concurrent
        // timer against a 300ms sleep instead, which a heavily loaded machine (full-workspace
        // parallel `cargo test`) could fail spuriously; this rendezvous has no timing threshold at
        // all short of the 10s safety bound.
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let released_by_peer = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let state = Arc::new(Mutex::new(FakeChildState {
            stop_gate: Some(StopGate {
                entered: Some(entered_tx),
                release: release_rx,
                released_by_peer: Arc::clone(&released_by_peer),
            }),
            ..FakeChildState::default()
        }));
        let child: Arc<dyn SupervisedChild> = Arc::new(FakeChild(Arc::clone(&state)));

        let peer = tokio::spawn(async move {
            // Released only once `stop` is provably blocking, never before, so a release that
            // merely sat queued in the channel cannot make an inline call look offloaded.
            if entered_rx.await.is_ok() {
                let _ = release_tx.send(());
            }
        });

        let confirmed = stop_child(child, Duration::from_secs(5)).await;
        assert!(confirmed, "FakeChild::stop was rigged to confirm");
        assert!(
            state.lock().unwrap_or_else(|e| e.into_inner()).stopped,
            "stop_child must actually have called SupervisedChild::stop"
        );
        assert!(
            released_by_peer.load(std::sync::atomic::Ordering::SeqCst),
            "stop_child ran the blocking stop on the runtime's only thread: the concurrent task \
             that would have released it never got to run"
        );

        peer.await.expect("peer task must not panic");
    }

    #[test]
    fn launch_outcome_maps_a_launcher_panic_to_spawn_failed() {
        // `tokio::task::JoinError` has no public constructor; a real panic inside `spawn_blocking`
        // is exercised in the async tests above only indirectly. This proves the *other* four arms
        // of `launch_outcome`'s match at least stay exhaustive and map to the reasons this module's
        // docs claim.
        use crate::protections::Protection;
        // The governing switch survives the typed error becoming text, in both classes.
        let permanent = crate::preflight::PreflightError::RunningElevated;
        let reason = launch_outcome(Ok(Err(LaunchError::Preflight(permanent))));
        assert!(
            matches!(
                &reason,
                Err(FailureReason::PreflightRefusedPermanently {
                    protection: Some(Protection::ElevationRefusal),
                    ..
                })
            ),
            "{:?}",
            reason.err()
        );
        let transient = crate::preflight::PreflightError::PortReservationFailed {
            cause: "busy".into(),
        };
        let reason = launch_outcome(Ok(Err(LaunchError::Preflight(transient))));
        assert!(
            matches!(
                &reason,
                Err(FailureReason::PreflightRefused {
                    protection: None,
                    ..
                })
            ),
            "{:?}",
            reason.err()
        );
        let governed_transient = crate::preflight::PreflightError::BackendDirUnauditable {
            dir: r"C:\b".into(),
            cause: "audit could not complete".into(),
        };
        let reason = launch_outcome(Ok(Err(LaunchError::Preflight(governed_transient))));
        assert_eq!(
            reason.as_ref().err().and_then(FailureReason::governing_protection),
            Some(Protection::FolderPermissionAudit),
            "{:?}",
            reason.err()
        );
    }

    #[test]
    fn disabled_protections_merges_both_halves_of_the_launch_record_in_order() {
        let disabled = crate::protections::Protections {
            refuse_elevation: false,
            contain_in_job_object: false,
            ..crate::protections::Protections::default()
        };
        let elevation_warning = disabled
            .elevation_warning()
            .expect("elevation refusal is off in this fixture");
        let job_warning = disabled
            .job_containment_warning()
            .expect("job containment is off in this fixture");

        let record = LaunchRecord {
            preflight_warnings: vec![preflight::PreflightWarning::ProtectionDisabled(
                elevation_warning.clone(),
            )],
            protection_warnings: vec![job_warning.clone()],
            ..empty_record(1)
        };

        let got = record.disabled_protections();
        assert_eq!(got, vec![&elevation_warning, &job_warning]);
    }

    #[test]
    fn disabled_protections_is_empty_when_every_protection_is_on() {
        let record = empty_record(1);
        assert!(record.disabled_protections().is_empty());
    }

    /// End-to-end live tests that launch a real `llama-server.exe` from this development machine's
    /// official fixture build, per this component's build brief. `#[ignore]`d: none of these run in
    /// an ordinary `cargo test`, only via `cargo test -p cerebro-supervisor -- --ignored`, which runs
    /// them in parallel — so every test here takes [`LIVE_SERIAL`], uses its own [`ScratchDir`], and
    /// checks the exact child process it launched (by the pid in its [`LaunchRecord`]), never a
    /// machine-wide scan that could see a sibling test's server or this machine's persistent one.
    mod live {
        use super::*;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU32, Ordering};

        use cerebro_flags::{DeviceList, Flag};

        use crate::preflight::{
            AllowUnsafeDacl, ElevationCheck, ProcessTokenElevation, LOOPBACK_HOST,
        };
        use crate::spawn::SpawnOptions;

        /// Serializes the live lifecycle tests: each loads the tiny model into its own
        /// `llama-server`, and running two at once only makes both slower and their timeouts less
        /// meaningful. A `tokio` mutex (not `std`) because it is held across `.await`s; it works
        /// across the separate runtimes each `#[tokio::test]` builds.
        static LIVE_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

        /// Makes every [`ScratchDir`] name unique within this process, not merely per process id:
        /// two tests in one test binary share a pid, and one finishing first must not delete the
        /// other's key file, log and TEMP while its server is still starting.
        static NEXT_SCRATCH: AtomicU32 = AtomicU32::new(0);

        /// A scratch directory under the OS temp dir, unique to one test within this process,
        /// cleaned up on drop.
        struct ScratchDir(PathBuf);

        impl ScratchDir {
            fn new(tag: &str) -> Self {
                let n = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
                let dir = std::env::temp_dir().join(format!(
                    "cerebro-supervisor-lifecycle-live-{tag}-{}-{n}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&dir).expect("create scratch dir for a live test");
                Self(dir)
            }
        }

        impl Drop for ScratchDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        /// A handle to one exact process, opened while it is known to be alive, so a later "has it
        /// exited?" check cannot be fooled by Windows reusing its pid for an unrelated process.
        struct ProcessWatch(windows_sys::Win32::Foundation::HANDLE);

        impl ProcessWatch {
            /// Open `pid` for waiting. Panics (test code) if it cannot be opened, which would mean
            /// the child named by the launch record is already gone or was never this pid.
            fn open(pid: u32) -> Self {
                use windows_sys::Win32::System::Threading::{
                    OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
                };
                // SAFETY: `OpenProcess` takes no pointers; the returned handle is checked for null
                // before use and closed exactly once, in `Drop`.
                let handle = unsafe {
                    OpenProcess(
                        PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                        0,
                        pid,
                    )
                };
                assert!(
                    !handle.is_null(),
                    "could not open the launched child (pid {pid})"
                );
                Self(handle)
            }

            /// Whether the process exits within `timeout`.
            fn exited_within(&self, timeout: Duration) -> bool {
                use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
                use windows_sys::Win32::System::Threading::WaitForSingleObject;
                let millis = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
                // SAFETY: `self.0` is a live process handle opened with `PROCESS_SYNCHRONIZE`.
                unsafe { WaitForSingleObject(self.0, millis) == WAIT_OBJECT_0 }
            }
        }

        impl Drop for ProcessWatch {
            fn drop(&mut self) {
                // SAFETY: `self.0` was opened successfully in `open` and is closed only here.
                unsafe {
                    windows_sys::Win32::Foundation::CloseHandle(self.0);
                }
            }
        }

        /// Reserve a free loopback port and release it immediately — the same bind-then-drop
        /// pattern `preflight`'s own live and unit tests use to obtain a real free port number
        /// without holding it, accepting the same unavoidable observed-free-then-rebound race
        /// [`crate::preflight::PortRequest`]'s own docs describe (exactly what this crate's
        /// [`FailureReason::PortBindFailure`] retry exists to recover from, if it is ever actually
        /// lost here).
        fn free_loopback_port() -> u16 {
            let listener =
                TcpListener::bind((LOOPBACK_HOST, 0)).expect("bind an ephemeral loopback port");
            listener.local_addr().expect("read back the bound port").port()
        }

        /// Send one `POST /completion` capped at 16 tokens over a raw blocking socket, and return
        /// the response's status line plus headers (not the full streamed body, which this test has
        /// no need to parse). Hand-rolled rather than routed through this module's own
        /// `readiness::HealthClient` (built only for bodyless `GET /health`, see that type's docs).
        fn send_one_completion(port: u16, api_key: &str) -> String {
            let body = r#"{"prompt": "The capital of France is", "n_predict": 16, "stream": false}"#;
            let mut stream = TcpStream::connect((LOOPBACK_HOST, port))
                .expect("connect to the supervised llama-server");
            stream
                .set_read_timeout(Some(Duration::from_secs(60)))
                .expect("set a read timeout so a hung response cannot block this test forever");
            let request = format!(
                "POST /completion HTTP/1.1\r\n\
                 Host: {LOOPBACK_HOST}:{port}\r\n\
                 Authorization: Bearer {api_key}\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\
                 \r\n\
                 {body}",
                body.len()
            );
            stream
                .write_all(request.as_bytes())
                .expect("send the completion request");
            let mut response = String::new();
            stream
                .read_to_string(&mut response)
                .expect("read the completion response");
            response
        }

        /// The first `Ready`, `Crashed` or `Failed` on the ordered transition log. The log, not the
        /// `watch` channel: `watch` can skip a `Crashed` that `Backoff` overwrote (see this module's
        /// "Two observation channels" section), which turned a restart loop into a silent timeout.
        async fn first_terminal(
            log: &mut broadcast::Receiver<LifecycleState>,
            timeout: Duration,
        ) -> LifecycleState {
            tokio::time::timeout(timeout, async {
                loop {
                    match log.recv().await {
                        Ok(
                            s @ (LifecycleState::Ready
                            | LifecycleState::Crashed(_)
                            | LifecycleState::Failed(_)),
                        ) => break s,
                        Ok(_) => {}
                        Err(e) => panic!("transition log: {e}"),
                    }
                }
            })
            .await
            .unwrap_or_else(|_| panic!("no Ready/Crashed/Failed state within {timeout:?}"))
        }

        #[tokio::test]
        #[ignore = "spawns a real llama-server.exe from an official build (CEREBRO_LLAMACPP_OFFICIAL, \
                    default C:/bench/llamacpp-official) with the tiny model (CEREBRO_TINY_GGUF); \
                    from an elevated shell it instead asserts the default elevation refusal"]
        async fn supervises_a_real_llama_server_through_ready_and_a_clean_stop() {
            if let Some(record) =
                supervise_real_llama_server(crate::protections::Protections::default()).await
            {
                assert!(record.disabled_protections().is_empty(), "{record:?}");
            }
        }

        #[tokio::test]
        #[ignore = "spawns a real llama-server.exe from an official build (CEREBRO_LLAMACPP_OFFICIAL, \
                    default C:/bench/llamacpp-official) with the tiny model (CEREBRO_TINY_GGUF); \
                    passes elevated or not, since elevation refusal is off"]
        async fn elevation_refusal_off_launches_a_real_server_and_records_the_warning() {
            // The switch honoured end to end: with `refuse_elevation` off, a real launch proceeds
            // even from an elevated shell, and the launch record names exactly that one protection.
            let protections = crate::protections::Protections {
                refuse_elevation: false,
                ..crate::protections::Protections::default()
            };
            let record = supervise_real_llama_server(protections)
                .await
                .expect("elevation refusal is off, so the launch must proceed");
            let disabled: Vec<_> = record
                .disabled_protections()
                .into_iter()
                .map(|w| w.protection())
                .collect();
            assert_eq!(
                disabled,
                vec![crate::protections::Protection::ElevationRefusal]
            );
        }

        /// Launch the tiny fixture model under `protections`, wait for `Ready`, run one completion,
        /// stop cleanly, prove the exact child this launch started has exited, and return the launch
        /// record. Returns `None` when this process is elevated and `protections` refuses elevation:
        /// the typed refusal is asserted instead, which is the correct default behaviour there.
        async fn supervise_real_llama_server(
            protections: crate::protections::Protections,
        ) -> Option<Arc<LaunchRecord>> {
            let _serial = LIVE_SERIAL.lock().await;

            // `CEREBRO_LLAMACPP_OFFICIAL` / `CEREBRO_TINY_GGUF` point these elsewhere.
            let backend_dir = cerebro_probe::fixtures::official_backend_dir()
                .unwrap_or_else(|e| panic!("{e}"));
            let model = cerebro_probe::fixtures::tiny_gguf().unwrap_or_else(|e| panic!("{e}"));

            let (manifest, _) = cerebro_probe::backend::hash_directory(&backend_dir)
                .expect("hash the real backend dir");

            let scratch = ScratchDir::new("e2e");
            let key_path = scratch.0.join("api.key");
            let api_key = "k".repeat(32);
            std::fs::write(&key_path, &api_key).expect("write fixture api key file");
            let log_path = scratch.0.join("llama-server.log");

            let port = free_loopback_port();

            let config = LifecycleConfig {
                launch_spec: LaunchSpec {
                    backend_dir: backend_dir.clone(),
                    tool_name: "llama-server.exe".to_string(),
                    manifest: manifest.clone(),
                    model_path: model,
                    draft_model_path: None,
                    port: PortRequest::Explicit(port),
                    api_key_file: key_path,
                    // The official fixture directory grants Modify to `Authenticated Users` (this
                    // development machine's own doing, not Cerebro's) — see this crate's `spawn` and
                    // `preflight` live tests for the same shape of opt-in against their own fixtures.
                    allow_unsafe_dacl: Some(
                        AllowUnsafeDacl::new(
                            "official llama.cpp live fixture; on the development machine \
                             Authenticated Users holds Modify on it by design of the test \
                             machine, not of Cerebro",
                        )
                        .expect("reason is non-blank"),
                    ),
                    protections,
                },
                // Force CPU (`-dev none`), per this component's build brief: this test's whole
                // point is the supervised lifecycle, not GPU offload.
                extra_flags: vec![Flag::Device(DeviceList::None)],
                spawn_options: SpawnOptions {
                    temp_dir: scratch.0.clone(),
                    cache_dir: scratch.0.clone(),
                    cuda_visible_devices: None,
                    log_path: log_path.clone(),
                    log_max_bytes: 1024 * 1024,
                },
                api_key: api_key.clone(),
                readiness_deadline: Duration::from_secs(90),
                readiness_poll_interval: Duration::from_millis(200),
                child_poll_interval: Duration::from_millis(100),
                restart_policy: RestartPolicy {
                    base_backoff: Duration::from_millis(500),
                    max_backoff: Duration::from_secs(5),
                    max_restarts: 3,
                    window: Duration::from_secs(60),
                    jitter_seed: 1,
                },
                stop_timeout: Duration::from_secs(15),
            };

            let (supervisor, _rx, mut log_rx) =
                Supervisor::spawn(config).expect("called inside a tokio runtime");
            let state = first_terminal(&mut log_rx, Duration::from_secs(120)).await;

            let elevated = ProcessTokenElevation
                .is_elevated()
                .expect("query this test process's own token");
            if protections.refuse_elevation && elevated {
                assert!(
                    matches!(
                        &state,
                        LifecycleState::Crashed(FailureReason::PreflightRefusedPermanently {
                            message,
                            protection: Some(crate::protections::Protection::ElevationRefusal),
                        }) if message.contains("elevated")
                    ),
                    "an elevated process with elevation refusal on must be refused by preflight: \
                     {state:?}"
                );
                // Permanent, so no restart: `Failed` follows at once, with no `Backoff` between.
                let next = tokio::time::timeout(Duration::from_secs(10), log_rx.recv())
                    .await
                    .expect("Failed must follow the permanent refusal at once")
                    .expect("transition log");
                assert!(
                    matches!(
                        next,
                        LifecycleState::Failed(FailureReason::PreflightRefusedPermanently { .. })
                    ),
                    "a permanent refusal must fail without restarting: {next:?}"
                );
                assert!(
                    supervisor.last_launch().is_none(),
                    "nothing may have been launched: {:?}",
                    supervisor.last_launch()
                );
                supervisor
                    .stop()
                    .await
                    .expect("stop a supervisor that never launched");
                return None;
            }

            assert_eq!(
                state,
                LifecycleState::Ready,
                "expected the supervised child to reach Ready; see {} for its own log",
                log_path.display()
            );
            let record = supervisor
                .last_launch()
                .expect("a launch that reached Ready has a launch record");
            // Opened now, while the child is certainly alive (it just answered /health), so the
            // exit check after `stop` names this exact process even if Windows later reuses its pid.
            let child = ProcessWatch::open(record.pid);

            let canonical = cerebro_probe::backend::canonical_backend_dir(&backend_dir)
                .expect("resolve the fixture dir");
            assert_eq!(record.requested_backend_dir, backend_dir);
            assert_eq!(
                record.backend_dir, canonical,
                "the record must name the resolved directory"
            );
            assert_eq!(
                record.backend_hashes, manifest,
                "the record must carry the verified hashes"
            );
            assert_eq!(record.port, port);

            // Off the current task so the driver keeps polling (health, exit) concurrently while
            // this blocking round trip runs.
            let response = tokio::task::spawn_blocking(move || send_one_completion(port, &api_key))
                .await
                .expect("the blocking completion request task must not panic");
            assert!(
                response.starts_with("HTTP/1.1 200"),
                "completion request did not succeed: {response}"
            );

            // `Ok` means the OS confirmed the child tree gone (see `Supervisor::stop`); the exact-pid
            // check below is the independent, OS-level confirmation of the same fact.
            supervisor
                .stop()
                .await
                .expect("the stop must be confirmed by the OS");
            assert!(
                child.exited_within(Duration::from_secs(10)),
                "the llama-server this test launched (pid {}) is still running after stop",
                record.pid
            );
            Some(record)
        }
    }
}
