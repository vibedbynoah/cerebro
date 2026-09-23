//! Supervising a `llama-server` child once [`mod@crate::preflight`] and [`mod@crate::spawn`] have
//! proven it safe to launch: readiness, crash classification, and a bounded restart policy, all
//! observable through one [`tokio::sync::watch`] channel.
//!
//! # State machine
//!
//! ```text
//! Starting -> Ready -> (Crashed -> Backoff -> Starting)* -> Stopped
//!                              \-> Crashed -> Failed
//! ```
//!
//! [`Supervisor::spawn`] starts in `Starting` and keeps restarting on a crash until either
//! [`Supervisor::stop`] moves it to the terminal `Stopped`, or the restart policy's sliding-window
//! limit is exceeded and it moves to the terminal `Failed` instead. Every transition is published on
//! the [`tokio::sync::watch::Receiver<LifecycleState>`] [`Supervisor::spawn`] returns.
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
mod readiness;

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

use cerebro_flags::Flag;
use thiserror::Error;
use tokio::sync::{mpsc, watch};

use crate::preflight::{self, LaunchSpec, PortRequest};
use crate::spawn::SpawnOptions;

use backoff::{backoff_delay, RestartWindow};
use launch::{ExitInfo, LaunchError, Launched, Launcher, RealLauncher, SupervisedChild};

pub use backoff::RestartPolicy;
pub use classify::FailureReason;

/// Everything needed to launch and keep a `llama-server` supervised.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Preflighted fresh before every (re-)launch. Its `port` field is only the *initial* request —
    /// a detected [`FailureReason::PortBindFailure`] overrides it with [`PortRequest::Any`] for
    /// exactly one retry, then reverts to counting like any other failure if it recurs.
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

/// A supervised child's lifecycle, observable through the [`watch::Receiver`] [`Supervisor::spawn`]
/// returns. See this module's top-level docs for the full state diagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleState {
    /// A launch attempt is preflighting and/or spawning, or the child is running but has not yet
    /// answered `GET /health` with 200.
    Starting,
    /// `GET /health` returned 200. The child is being supervised for as long as this state holds.
    Ready,
    /// The child stopped, or never started, for the given reason. Always immediately followed by
    /// either [`LifecycleState::Backoff`] or [`LifecycleState::Failed`].
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
    },
    /// Stopped deliberately, via [`Supervisor::stop`] (or a [`Supervisor`] dropped without calling
    /// it). Terminal: nothing further is ever published after this.
    Stopped,
    /// The restart policy's sliding-window limit was exceeded; this launch is no longer retried.
    /// Terminal.
    Failed(FailureReason),
}

/// Everything a caller of [`Supervisor::stop`] can see go wrong.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LifecycleError {
    /// The background driver task panicked. Every fallible step inside it is handled with a typed
    /// [`FailureReason`] instead (see [`FailureReason::SpawnFailed`]'s "launch task panicked" case),
    /// so this should never actually happen — but [`tokio::task::JoinHandle::await`] can still
    /// report it, and a caller deserves a typed answer rather than an unwrapped `JoinError`.
    #[error("the lifecycle driver task panicked: {0}")]
    DriverPanicked(String),
}

/// A running `llama-server` lifecycle. [`Supervisor::spawn`] starts the background driver task and
/// hands back both this handle and a [`watch::Receiver`] observing every [`LifecycleState`]
/// transition (a `watch` channel always hands a new subscriber the current value first, so cloning
/// the receiver later never misses the fact that, say, the child is already `Ready`).
pub struct Supervisor {
    stop_tx: mpsc::UnboundedSender<()>,
    // `Option`, not a bare `JoinHandle`, purely so `stop` can `.take()` it out through `&mut self`:
    // `stop` needs to `.await` an owned `JoinHandle`, and a type with a `Drop` impl (this one, for
    // the fire-and-forget safety net below) cannot have a field moved out of it directly (`E0509`)
    // — `Option::take` mutates in place instead of partially moving `self`, which sidesteps that
    // restriction without needing `unsafe` or `ManuallyDrop`.
    driver: Option<tokio::task::JoinHandle<()>>,
}

impl Supervisor {
    /// Launch and supervise a `llama-server` per `config`. See [`LifecycleConfig`] for what it
    /// controls.
    pub fn spawn(config: LifecycleConfig) -> (Self, watch::Receiver<LifecycleState>) {
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
    ) -> (Self, watch::Receiver<LifecycleState>) {
        let (state_tx, state_rx) = watch::channel(LifecycleState::Starting);
        let (stop_tx, stop_rx) = mpsc::unbounded_channel();
        let port_request = config.launch_spec.port;
        let driver = tokio::spawn(run(
            launcher,
            port_request,
            config.api_key,
            config.readiness_deadline,
            config.readiness_poll_interval,
            config.child_poll_interval,
            config.restart_policy,
            config.stop_timeout,
            state_tx,
            stop_rx,
        ));
        (
            Self {
                stop_tx,
                driver: Some(driver),
            },
            state_rx,
        )
    }

    /// Ask the driver to stop. During [`LifecycleState::Backoff`] this cancels the pending restart
    /// and moves straight to [`LifecycleState::Stopped`]; otherwise, if a child is running, it is
    /// killed via its job object and waited on (up to [`LifecycleConfig::stop_timeout`]) before
    /// [`LifecycleState::Stopped`] is published. Waits for the driver task to actually finish before
    /// returning, so a caller that gets `Ok(())` back knows the child is no longer being supervised.
    pub async fn stop(mut self) -> Result<(), LifecycleError> {
        let _ = self.stop_tx.send(());
        match self.driver.take() {
            Some(driver) => driver.await.map_err(|e| LifecycleError::DriverPanicked(e.to_string())),
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
        Ok(Err(LaunchError::Preflight(e))) => Err(FailureReason::PreflightRefused(e.to_string())),
        Ok(Err(LaunchError::Argv(e))) => Err(FailureReason::ArgvRejected(e.to_string())),
        Ok(Err(LaunchError::Spawn(e))) => Err(FailureReason::SpawnFailed(e.to_string())),
        Ok(Err(LaunchError::Child(msg))) => Err(FailureReason::SpawnFailed(msg)),
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

/// Record `reason` as [`LifecycleState::Crashed`], then decide and carry out what comes next: the
/// one immediate new-port retry for [`FailureReason::PortBindFailure`], an ordinary backed-off
/// restart, or [`LifecycleState::Failed`] once the restart policy's window is exceeded. A stop
/// requested while this function is waiting out a backoff delay cancels it and publishes
/// [`LifecycleState::Stopped`] instead.
///
/// Returns [`ControlFlow::Break`] once a terminal state ([`LifecycleState::Stopped`] or
/// [`LifecycleState::Failed`]) has already been published — [`run`] returns immediately in that
/// case — or [`ControlFlow::Continue`] once it is time to attempt [`LifecycleState::Starting`] again.
#[allow(clippy::too_many_arguments)]
async fn handle_failure(
    reason: FailureReason,
    port_request: &mut PortRequest,
    port_retry_used: &mut bool,
    window: &mut RestartWindow,
    policy: &RestartPolicy,
    start: tokio::time::Instant,
    state_tx: &watch::Sender<LifecycleState>,
    stop_rx: &mut mpsc::UnboundedReceiver<()>,
) -> ControlFlow<()> {
    let _ = state_tx.send(LifecycleState::Crashed(reason.clone()));
    // `watch` only ever keeps the *latest* value: it does not queue one per `send`. Every path below
    // this point sends a second state (`Starting` for the immediate port-bind retry, `Backoff` or
    // `Failed` otherwise) with no other `.await` in between on most of those paths — so without an
    // explicit yield here, a `Backoff`/`Starting` sent moments later on the same poll would overwrite
    // `Crashed` before any receiver's `changed().await` is ever woken to observe it, silently
    // dropping the very transition this module's docs promise "every transition is published".
    // Yielding once gives the executor a chance to run every receiver waiting on `changed()` with
    // `Crashed` as the current value before this task is polled again.
    tokio::task::yield_now().await;

    if reason.is_port_bind_failure() && !*port_retry_used {
        // The one immediate retry documented on `FailureReason::PortBindFailure`: preflight's own
        // port check cannot close the race between "observed free" and the real bind, so this does
        // not count against the restart window or wait out a backoff delay — it goes straight back
        // to `Starting` with a fresh OS-chosen port.
        *port_retry_used = true;
        *port_request = PortRequest::Any;
        return ControlFlow::Continue(());
    }

    let count = window.record(policy, start.elapsed());
    if count > policy.max_restarts {
        let _ = state_tx.send(LifecycleState::Failed(reason));
        return ControlFlow::Break(());
    }

    let delay = backoff_delay(policy, count);
    let _ = state_tx.send(LifecycleState::Backoff { attempt: count, delay });
    tokio::select! {
        biased;
        _ = stop_rx.recv() => {
            let _ = state_tx.send(LifecycleState::Stopped);
            return ControlFlow::Break(());
        }
        _ = tokio::time::sleep(delay) => {}
    }
    ControlFlow::Continue(())
}

/// The state machine itself. Runs until [`LifecycleState::Stopped`] or [`LifecycleState::Failed`] is
/// published, at which point the [`tokio::task::JoinHandle`] [`Supervisor::spawn_with`] wraps this in
/// simply finishes.
#[allow(clippy::too_many_arguments)]
async fn run(
    launcher: Arc<dyn Launcher>,
    mut port_request: PortRequest,
    api_key: String,
    readiness_deadline: Duration,
    readiness_poll_interval: Duration,
    child_poll_interval: Duration,
    restart_policy: RestartPolicy,
    stop_timeout: Duration,
    state_tx: watch::Sender<LifecycleState>,
    mut stop_rx: mpsc::UnboundedReceiver<()>,
) {
    let client = readiness::build_client();
    let start = tokio::time::Instant::now();
    let mut window = RestartWindow::new();
    let mut port_retry_used = false;

    // A tiny local macro, not a helper fn: `handle_failure`'s five call sites below all need to
    // borrow the same handful of locals (`port_request`, `window`, `stop_rx`, …) and then either
    // `return` or `continue` out of *this* function's loop depending on the answer — neither
    // of which a called fn can do on `run`'s behalf. Spelling out the five-line match at each site
    // instead would risk one of them drifting (e.g. forgetting the `continue`), which a macro makes
    // structurally impossible.
    //
    // Defined here, after every local it references (`port_request` aside, which is a parameter and
    // so already in scope from the top of the function) — `macro_rules!` hygiene resolves a free
    // identifier inside the macro body against what is in scope *at the macro's own definition
    // site*, not merely "somewhere later in the same block", so each of `client`, `start`, `window`
    // and `port_retry_used` has to be bound above this point for the macro to see it at all.
    macro_rules! fail {
        ($reason:expr) => {{
            match handle_failure(
                $reason,
                &mut port_request,
                &mut port_retry_used,
                &mut window,
                &restart_policy,
                start,
                &state_tx,
                &mut stop_rx,
            )
            .await
            {
                ControlFlow::Break(()) => return,
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
        if first_launch {
            first_launch = false;
        } else {
            let _ = state_tx.send(LifecycleState::Starting);
        }

        // A stop requested while idle between restarts is honored here rather than only inside the
        // post-launch `select!` below — otherwise a stop arriving during, say, a `Backoff` that just
        // elapsed would still pay for one more, entirely pointless preflight-and-spawn attempt before
        // being noticed. `try_recv` is deliberately non-blocking: an in-flight `spawn_blocking` launch
        // below is not itself cancelled once started (see this module's top-level docs), only ever
        // pre-empted before it begins.
        if stop_rx.try_recv().is_ok() {
            let _ = state_tx.send(LifecycleState::Stopped);
            return;
        }

        // Preflight (file hashing, DACL audit) and the OS-level spawn are both synchronous,
        // potentially slow work; running them inline on this task would stall every other future
        // sharing this executor thread for as long as they take. `spawn_blocking` moves that work to
        // Tokio's blocking thread pool, which is what makes this otherwise-synchronous crate usable
        // from an async supervisor at all.
        let launcher = Arc::clone(&launcher);
        let launch_result = tokio::task::spawn_blocking(move || launcher.launch(port_request)).await;

        let launched = match launch_outcome(launch_result) {
            Ok(launched) => launched,
            Err(reason) => fail!(reason),
        };

        /// One post-launch race outcome: a stop request, the child exiting, or the readiness probe
        /// concluding one way or the other.
        enum Event {
            Stop,
            Exited(Result<ExitInfo, FailureReason>),
            Ready(Result<(), readiness::ReadinessError>),
        }

        let event = tokio::select! {
            biased;
            _ = stop_rx.recv() => Event::Stop,
            exit = watch_for_exit(launched.child.as_ref(), child_poll_interval) => Event::Exited(exit),
            ready = readiness::wait_ready(
                &client,
                preflight::LOOPBACK_HOST,
                launched.port,
                &api_key,
                readiness_deadline,
                readiness_poll_interval,
            ) => Event::Ready(ready),
        };

        match event {
            Event::Stop => {
                let _ = launched.child.stop(stop_timeout);
                let _ = state_tx.send(LifecycleState::Stopped);
                return;
            }

            Event::Exited(Ok(info)) => {
                let lines = launched.child.recent_lines();
                let reason = classify::classify_exit_before_ready(launched.port, info.code, &lines);
                fail!(reason)
            }

            Event::Exited(Err(reason)) => fail!(reason),

            Event::Ready(Ok(())) => {
                // A retry earned by a bind-failure race only ever applies to the *next* failure, not
                // forever: once a launch reaches readiness, whatever bound the port is proven fine,
                // so a future bind failure deserves its own retry rather than being folded into a
                // stale flag left over from several restarts ago.
                port_retry_used = false;
                let _ = state_tx.send(LifecycleState::Ready);

                let post_ready = tokio::select! {
                    biased;
                    _ = stop_rx.recv() => None,
                    exit = watch_for_exit(launched.child.as_ref(), child_poll_interval) => Some(exit),
                };
                match post_ready {
                    None => {
                        let _ = launched.child.stop(stop_timeout);
                        let _ = state_tx.send(LifecycleState::Stopped);
                        return;
                    }
                    Some(Ok(info)) => {
                        let lines = launched.child.recent_lines();
                        let reason = FailureReason::CrashedAfterReady {
                            exit_code: info.code,
                            last_lines: lines,
                        };
                        fail!(reason)
                    }
                    Some(Err(reason)) => fail!(reason),
                }
            }

            Event::Ready(Err(readiness_err)) => {
                // The deadline (or a 401/unexpected status) and the child exiting on its own can race
                // each other; prefer the more specific exit-based classification when the child is
                // actually gone, and otherwise stop a child that is still running but has failed its
                // readiness check before recording a reason that would otherwise imply it might still
                // be serving traffic.
                let reason = match launched.child.poll_exit() {
                    Ok(Some(info)) => {
                        let lines = launched.child.recent_lines();
                        classify::classify_exit_before_ready(launched.port, info.code, &lines)
                    }
                    Ok(None) => {
                        let _ = launched.child.stop(stop_timeout);
                        classify::classify_readiness_error(readiness_err)
                    }
                    Err(e) => FailureReason::ChildPollFailed(e.to_string()),
                };
                fail!(reason)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Shared, mutable state for one [`FakeChild`], so a test can flip it to "exited" from outside
    /// the driver task without holding a `&mut` the driver itself also needs.
    #[derive(Default)]
    struct FakeChildState {
        exit: Option<ExitInfo>,
        lines: Vec<String>,
    }

    struct FakeChild(Arc<Mutex<FakeChildState>>);

    impl SupervisedChild for FakeChild {
        fn poll_exit(&self) -> Result<Option<ExitInfo>, LaunchError> {
            Ok(self.0.lock().unwrap_or_else(|e| e.into_inner()).exit)
        }
        fn recent_lines(&self) -> Vec<String> {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).lines.clone()
        }
        fn stop(&self, _timeout: Duration) -> Result<bool, LaunchError> {
            self.0.lock().unwrap_or_else(|e| e.into_inner()).exit.get_or_insert(ExitInfo { code: Some(0) });
            Ok(true)
        }
    }

    /// One scripted outcome for [`FakeLauncher::launch`].
    enum Script {
        /// Fail the launch itself (no child ever exists).
        LaunchFails,
        /// Spawn a child that has already exited by the time `launch` returns, with the given exit
        /// code and captured output lines.
        ExitsImmediately { exit_code: Option<i32>, lines: Vec<String> },
        /// Spawn a child that keeps running (readiness polling then decides its fate) on `port`.
        KeepsRunning { port: u16 },
    }

    /// A [`Launcher`] driven entirely by a caller-scripted sequence, so the state machine's tests
    /// exercise every transition without a real OS process — see this module's top-level docs on why
    /// [`Launcher`] exists.
    struct FakeLauncher {
        script: Mutex<std::collections::VecDeque<Script>>,
    }

    impl FakeLauncher {
        fn new(script: Vec<Script>) -> Self {
            Self {
                script: Mutex::new(script.into_iter().collect()),
            }
        }
    }

    impl Launcher for FakeLauncher {
        fn launch(&self, _port: PortRequest) -> Result<Launched, LaunchError> {
            let next = self
                .script
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop_front()
                .unwrap_or(Script::LaunchFails);
            match next {
                Script::LaunchFails => Err(LaunchError::Child("fake launch failure".to_string())),
                Script::ExitsImmediately { exit_code, lines } => {
                    let state = Arc::new(Mutex::new(FakeChildState {
                        exit: Some(ExitInfo { code: exit_code }),
                        lines,
                    }));
                    Ok(Launched {
                        child: Box::new(FakeChild(state)),
                        port: 0,
                    })
                }
                Script::KeepsRunning { port } => {
                    let state = Arc::new(Mutex::new(FakeChildState::default()));
                    Ok(Launched {
                        child: Box::new(FakeChild(state)),
                        port,
                    })
                }
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
            },
            extra_flags: Vec::new(),
            spawn_options: SpawnOptions {
                temp_dir: std::path::PathBuf::from(r"C:\unused\temp"),
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

    async fn next_state(rx: &mut watch::Receiver<LifecycleState>) -> LifecycleState {
        rx.changed().await.expect("driver task is still running");
        rx.borrow().clone()
    }

    #[tokio::test(start_paused = true)]
    async fn a_child_that_never_exits_and_never_answers_health_times_out_into_backoff() {
        let launcher: Arc<dyn Launcher> =
            Arc::new(FakeLauncher::new(vec![Script::KeepsRunning { port: 65535 }]));
        let (supervisor, mut rx) = Supervisor::spawn_with(launcher, test_config(fast_policy()));

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        // Nothing is listening on 65535, so every readiness probe fails to connect and is treated as
        // "still loading" (see `readiness::ReadinessError::Unreachable`'s docs) until the 2s
        // deadline in `test_config` elapses — advanced here rather than actually waited out, which
        // is the entire point of `start_paused = true` plus an explicit `advance`.
        tokio::time::advance(Duration::from_secs(3)).await;

        assert!(matches!(
            next_state(&mut rx).await,
            LifecycleState::Crashed(FailureReason::ReadinessDeadlineExceeded { .. })
        ));
        assert!(matches!(next_state(&mut rx).await, LifecycleState::Backoff { attempt: 1, .. }));

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
        let (supervisor, mut rx) = Supervisor::spawn_with(launcher, test_config(fast_policy()));

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(
            next_state(&mut rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady {
                exit_code: Some(134),
                last_lines: vec!["fatal: out of memory".to_string()],
            })
        );
        let backoff = next_state(&mut rx).await;
        assert!(matches!(backoff, LifecycleState::Backoff { attempt: 1, .. }));

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(next_state(&mut rx).await, LifecycleState::Starting);

        drop(rx);
        supervisor.stop().await.expect("driver task must not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn a_port_bind_failure_is_retried_once_immediately_then_counts_normally() {
        let bind_failure = || Script::ExitsImmediately {
            exit_code: Some(1),
            lines: vec!["srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080".to_string()],
        };
        let launcher: Arc<dyn Launcher> = Arc::new(FakeLauncher::new(vec![
            bind_failure(),
            bind_failure(),
        ]));
        let (supervisor, mut rx) = Supervisor::spawn_with(launcher, test_config(fast_policy()));

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(
            next_state(&mut rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 8080 })
        );
        // The first bind failure is retried immediately: no `Backoff` in between.
        assert_eq!(next_state(&mut rx).await, LifecycleState::Starting);

        assert_eq!(
            next_state(&mut rx).await,
            LifecycleState::Crashed(FailureReason::PortBindFailure { port: 8080 })
        );
        // The second, in a row, is an ordinary counted restart: `Backoff` follows this time.
        assert!(matches!(next_state(&mut rx).await, LifecycleState::Backoff { attempt: 1, .. }));

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
        let (supervisor, mut rx) = Supervisor::spawn_with(launcher, test_config(fast_policy()));

        for expected_attempt in 1..=2 {
            assert!(matches!(
                next_state(&mut rx).await,
                LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
            ));
            assert!(matches!(
                next_state(&mut rx).await,
                LifecycleState::Backoff { attempt, .. } if attempt == expected_attempt
            ));
            tokio::time::advance(Duration::from_secs(1)).await;
            assert_eq!(next_state(&mut rx).await, LifecycleState::Starting);
        }

        assert!(matches!(
            next_state(&mut rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
        ));
        assert!(matches!(
            next_state(&mut rx).await,
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
        let (supervisor, mut rx) = Supervisor::spawn_with(launcher, test_config(fast_policy()));

        assert!(matches!(
            next_state(&mut rx).await,
            LifecycleState::Crashed(FailureReason::ExitedBeforeReady { .. })
        ));
        assert!(matches!(next_state(&mut rx).await, LifecycleState::Backoff { .. }));

        // Stop while the backoff delay (100ms cap, well under the 1h window) is still pending: the
        // driver must notice via the `select!` in `handle_failure`, not by waiting the delay out.
        supervisor.stop().await.expect("driver task must not panic");
        assert_eq!(*rx.borrow(), LifecycleState::Stopped);
    }

    #[tokio::test]
    async fn reaching_ready_and_then_stopping_publishes_ready_then_stopped() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener.local_addr().expect("read back the bound port").port();
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

        let launcher: Arc<dyn Launcher> = Arc::new(FakeLauncher::new(vec![Script::KeepsRunning { port }]));
        let (supervisor, mut rx) = Supervisor::spawn_with(launcher, test_config(fast_policy()));

        assert_eq!(*rx.borrow(), LifecycleState::Starting);
        assert_eq!(next_state(&mut rx).await, LifecycleState::Ready);

        supervisor.stop().await.expect("driver task must not panic");
        assert_eq!(*rx.borrow(), LifecycleState::Stopped);
    }

    #[test]
    fn launch_outcome_maps_a_launcher_panic_to_spawn_failed() {
        // `tokio::task::JoinError` has no public constructor; a real panic inside `spawn_blocking`
        // is exercised in the async tests above only indirectly. This proves the *other* four arms
        // of `launch_outcome`'s match at least stay exhaustive and map to the reasons this module's
        // docs claim.
        let preflight_err = crate::preflight::PreflightError::RunningElevated;
        let reason = launch_outcome(Ok(Err(LaunchError::Preflight(preflight_err))));
        assert!(matches!(reason, Err(FailureReason::PreflightRefused(_))));
    }

    /// End-to-end live tests that launch a real `llama-server.exe` from this development machine's
    /// b10751 fixture build, per this component's build brief. `#[ignore]`d: none of these run in an
    /// ordinary `cargo test`, only via `cargo test -p cerebro-supervisor -- --ignored`.
    mod live {
        use super::*;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::path::PathBuf;

        use cerebro_flags::{DeviceList, Flag};

        use crate::preflight::{AllowUnsafeDacl, LOOPBACK_HOST};
        use crate::spawn::SpawnOptions;

        /// A scratch directory under the OS temp dir, unique to this test process, cleaned up on
        /// drop — the same pattern `spawn`'s and `preflight`'s own live tests use, so parallel
        /// `--ignored` runs cannot collide on the same key file or log path.
        struct ScratchDir(PathBuf);

        impl ScratchDir {
            fn new(tag: &str) -> Self {
                let dir = std::env::temp_dir().join(format!(
                    "cerebro-supervisor-lifecycle-live-{tag}-{}",
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

        /// True if a process literally named `llama-server.exe` (case-insensitive) exists anywhere
        /// on this machine — not merely under a PID this test itself tracked. That is the stronger,
        /// literal claim this component's build brief asks for ("asserts no llama-server process
        /// remains"), not just "the child handle this test happened to keep is gone".
        fn any_llama_server_process_exists() -> bool {
            use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
            use windows_sys::Win32::System::Diagnostics::ToolHelp::{
                CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
                TH32CS_SNAPPROCESS,
            };

            /// Closes a Toolhelp snapshot handle exactly once on every path out of the `unsafe`
            /// block below, including the early `return false`s inside its scan loop — a bare
            /// `CloseHandle` call at the bottom of that block would be skipped by those.
            struct CloseOnDrop(HANDLE);
            impl Drop for CloseOnDrop {
                fn drop(&mut self) {
                    // SAFETY: `self.0` is always a snapshot handle `any_llama_server_process_exists`
                    // itself opened successfully (see the `INVALID_HANDLE_VALUE` check just below,
                    // before this guard is ever constructed) and never closes anywhere else.
                    unsafe {
                        CloseHandle(self.0);
                    }
                }
            }

            // SAFETY: `TH32CS_SNAPPROCESS` with PID `0` snapshots every process on the system and
            // takes no pointer arguments; the returned handle is checked against
            // `INVALID_HANDLE_VALUE` before use and is wrapped in `CloseOnDrop` immediately after.
            // `entry.dwSize` is set to the struct's own size, as `Process32FirstW`/`Process32NextW`
            // require, before either is ever called, always with a valid `&mut entry`.
            unsafe {
                let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
                if snapshot == INVALID_HANDLE_VALUE {
                    panic!("CreateToolhelp32Snapshot failed");
                }
                let _guard = CloseOnDrop(snapshot);

                let mut entry: PROCESSENTRY32W = std::mem::zeroed();
                entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
                if Process32FirstW(snapshot, &mut entry) == 0 {
                    return false;
                }
                loop {
                    let name_len = entry
                        .szExeFile
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(entry.szExeFile.len());
                    let name = String::from_utf16_lossy(&entry.szExeFile[..name_len]);
                    if name.eq_ignore_ascii_case("llama-server.exe") {
                        return true;
                    }
                    if Process32NextW(snapshot, &mut entry) == 0 {
                        return false;
                    }
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
        /// `readiness::HealthClient` (built only for bodyless `GET /health`, see that type's docs):
        /// pulling in `hyper`'s request-body machinery for one `#[ignore]`d test's single POST would
        /// be more machinery than this call site needs — the same "smallest tool for the job" choice
        /// this crate's build brief already made for `readiness` itself.
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

        /// Wait until `rx`'s current state satisfies `done`, or panic once `timeout` elapses. Unlike
        /// this module's other tests' `next_state` (which asserts each individual transition against
        /// `tokio::time`'s *paused* clock), this drives a real clock against a real child process, so
        /// it only cares about the state it eventually settles into, not each step on the way there.
        async fn wait_until(
            rx: &mut watch::Receiver<LifecycleState>,
            timeout: Duration,
            done: impl Fn(&LifecycleState) -> bool,
        ) -> LifecycleState {
            tokio::time::timeout(timeout, async {
                loop {
                    {
                        let state = rx.borrow();
                        if done(&state) {
                            return state.clone();
                        }
                    }
                    rx.changed().await.expect("driver task is still running");
                }
            })
            .await
            .unwrap_or_else(|_| panic!("state did not satisfy the condition within {timeout:?}"))
        }

        #[tokio::test]
        #[ignore = "spawns a real llama-server.exe process on this machine's b10751 fixture build \
                    with the tiny CPU model, per this component's build brief"]
        async fn supervises_a_real_llama_server_through_ready_and_a_clean_stop() {
            let backend_dir = PathBuf::from("C:/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751");
            assert!(
                backend_dir.join("llama-server.exe").is_file(),
                "fixture backend not found at {}; see this component's build brief",
                backend_dir.display()
            );
            let model = PathBuf::from(
                r"C:\Users\noahm\.lmstudio\models\skiesoffire\bastion-control\bastion-control-0.5b-f16.gguf",
            );
            assert!(
                model.is_file(),
                "fixture model not found at {}; see this component's build brief",
                model.display()
            );
            assert!(
                !any_llama_server_process_exists(),
                "another llama-server.exe is already running on this machine; this test's own \
                 final assertion (none remain after stop) would be meaningless with one already \
                 running for an unrelated reason"
            );

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
                    manifest,
                    model_path: model,
                    draft_model_path: None,
                    port: PortRequest::Explicit(port),
                    api_key_file: key_path,
                    // Same known-unsafe development fixture every other live test in this crate
                    // opts into; see `spawn`'s and `preflight`'s own live tests for why this
                    // directory's ACL is unsafe by design of the test machine, not of Cerebro.
                    allow_unsafe_dacl: Some(
                        AllowUnsafeDacl::new(
                            "b10751 live fixture; ASUS\\CodexSandboxUsers holds Modify on this \
                             directory by design of the test machine, not of Cerebro",
                        )
                        .expect("reason is non-blank"),
                    ),
                },
                // Force CPU (`-dev none`), per this component's build brief: this test's whole
                // point is the supervised lifecycle, not GPU offload, and the tiny fixture model
                // has no need of it.
                extra_flags: vec![Flag::Device(DeviceList::None)],
                spawn_options: SpawnOptions {
                    temp_dir: scratch.0.clone(),
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

            let (supervisor, mut rx) = Supervisor::spawn(config);

            let state = wait_until(&mut rx, Duration::from_secs(120), |s| {
                matches!(
                    s,
                    LifecycleState::Ready | LifecycleState::Crashed(_) | LifecycleState::Failed(_)
                )
            })
            .await;
            assert_eq!(
                state,
                LifecycleState::Ready,
                "expected the supervised child to reach Ready; see {} for its own log",
                log_path.display()
            );

            // Off the current task so the driver keeps polling (health, exit) concurrently while
            // this blocking round trip runs — see this test's own `tokio-based` component brief.
            let response = tokio::task::spawn_blocking(move || send_one_completion(port, &api_key))
                .await
                .expect("the blocking completion request task must not panic");
            assert!(
                response.starts_with("HTTP/1.1 200"),
                "completion request did not succeed: {response}"
            );

            supervisor.stop().await.expect("driver task must not panic while stopping");

            // `Supervisor::stop`'s own job-object kill-and-wait already confirms *its* child tree is
            // gone before returning; this polls a little longer purely because Windows can take a
            // moment past that point to retire the process object a fresh
            // `CreateToolhelp32Snapshot` enumeration would still observe.
            let start = std::time::Instant::now();
            while any_llama_server_process_exists() {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "a llama-server.exe process still exists 10s after Supervisor::stop returned"
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
