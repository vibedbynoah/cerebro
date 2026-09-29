//! Turning a supervised child's exit, or a readiness probe's refusal, into one of the typed
//! [`FailureReason`]s the restart policy and an operator both act on.

use std::time::Duration;

use thiserror::Error;

use super::readiness::ReadinessError;

/// The exact phrase `llama-server-impl.dll` prints (`srv %s: couldn't bind HTTP server socket,
/// hostname: %s, port: %d`) when `httplib::Server::bind_to_port` fails, verified by reading the
/// b10751 fixture build's own strings rather than assumed — this crate's build brief requires
/// exactly that discipline for any claim about backend behavior. Matched as a plain substring, never
/// a regex: the log is untrusted child output, and a substring match cannot itself be steered into
/// matching something else the way an attacker-influenced regex pattern could.
const PORT_BIND_FAILURE_MARKER: &str = "couldn't bind HTTP server socket";

/// Why a supervised child stopped being — or never became — ready, precisely enough for
/// [`super::Supervisor`]'s restart policy and an operator reading [`super::LifecycleState`] to both
/// act on.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum FailureReason {
    /// The child process exited before its readiness probe ever returned 200.
    #[error("child exited before becoming ready (exit code {exit_code:?}); last output: {last_lines:?}")]
    ExitedBeforeReady {
        /// The child's exit code, if the OS reported one.
        exit_code: Option<i32>,
        /// The most recent captured stdout/stderr lines, oldest first.
        last_lines: Vec<String>,
    },

    /// The readiness probe never returned 200 within the configured deadline, and the process was
    /// still running when the deadline elapsed (it is stopped before this reason is recorded).
    #[error("readiness was not observed within {deadline:?}")]
    ReadinessDeadlineExceeded {
        /// The configured deadline that elapsed.
        deadline: Duration,
    },

    /// The child's own log announced it could not bind its listening port. Detected rather than
    /// merely inferred from a bare early exit, because it drives a different response: preflight's
    /// own port check ([`crate::preflight::PortRequest`]) proves a port free only at the instant it
    /// is checked, and cannot close the race against the real bind moments later — so this is worth
    /// one immediate retry on a new port, not an ordinary backed-off restart.
    ///
    /// Also recorded when the listener on this port belongs to another process (see
    /// `super::listener`): it won the port in that same race, and our child, still loading, never
    /// bound it. Readiness polling detects that before sending anything, so the foreign process
    /// never receives the bearer key.
    #[error("child could not bind its listening port {port}")]
    PortBindFailure {
        /// The port the child failed to bind.
        port: u16,
    },

    /// The child exited (or its connection was lost) after readiness had already been observed at
    /// least once during this launch attempt.
    #[error("child crashed after becoming ready (exit code {exit_code:?}); last output: {last_lines:?}")]
    CrashedAfterReady {
        /// The child's exit code, if the OS reported one.
        exit_code: Option<i32>,
        /// The most recent captured stdout/stderr lines, oldest first.
        last_lines: Vec<String>,
    },

    /// Preflight itself refused the (re-)launch — run fresh before every restart, never only once
    /// (ARCHITECTURE.md section 8) — for a reason a retry can plausibly outlive: a busy port, a model
    /// or key file missing for now, an I/O error, an audit that could not complete. Refusals no retry
    /// can fix are [`FailureReason::PreflightRefusedPermanently`] instead.
    #[error("preflight refused the (re-)launch: {message}")]
    PreflightRefused {
        /// The refusal, as [`crate::preflight::PreflightError`]'s `Display` wrote it.
        message: String,
        /// The setting that would let this refusal through, if any
        /// ([`crate::preflight::PreflightError::governing_protection`]), kept typed so a UI can offer
        /// that toggle without parsing `message`. An audit that could not complete is transient yet
        /// still governed by the folder audit, for example.
        protection: Option<crate::protections::Protection>,
    },

    /// Preflight refused the launch for a reason that retrying the same [`crate::preflight::LaunchSpec`]
    /// cannot fix without the user changing the settings or the files on disk (see
    /// [`crate::preflight::PreflightError::is_permanent`]): running elevated, a backend file whose
    /// hash does not match the manifest, an unsafe DACL with no opt-in, and so on. A configuration
    /// error, so it moves straight to [`super::LifecycleState::Failed`] instead of re-hashing the
    /// whole backend on every restart attempt only to be refused the same way.
    #[error("preflight refused the launch, and retrying cannot fix it: {message}")]
    PreflightRefusedPermanently {
        /// The refusal, as [`crate::preflight::PreflightError`]'s `Display` wrote it; for a governed
        /// refusal it already ends with the settings hint.
        message: String,
        /// The setting that would let this refusal through, if any
        /// ([`crate::preflight::PreflightError::governing_protection`]), kept typed so a UI can offer
        /// that toggle without parsing `message`.
        protection: Option<crate::protections::Protection>,
    },

    /// Building the argv for this launch failed (a duplicate or preflight-owned flag, a non-UTF-8
    /// path, a violated cross-flag constraint).
    #[error("could not build argv for the launch: {0}")]
    ArgvRejected(String),

    /// The OS-level spawn call itself failed.
    #[error("could not spawn the child: {0}")]
    SpawnFailed(String),

    /// The health check rejected the configured API key (401). Only ever our own child's answer:
    /// readiness polling sends nothing until the listener is attributed to the child's process, and
    /// a listener owned by another process is [`FailureReason::PortBindFailure`] instead. The child
    /// is stopped before this reason is
    /// recorded — a server answering with the wrong key accepted is not one to keep
    /// running unauthenticated.
    #[error("the health check rejected the configured API key")]
    Unauthorized,

    /// The health check returned a status neither 200 (ready) nor 503 (loading).
    #[error("the health check returned unexpected status {0}")]
    UnexpectedReadinessStatus(u16),

    /// Polling the child's own exit status failed at the OS level.
    #[error("could not poll the child's exit status: {0}")]
    ChildPollFailed(String),

    /// The readiness request itself could not be built (an API key that is not a valid header
    /// value, most plausibly). A configuration error that no restart can fix, so it moves straight
    /// to [`super::LifecycleState::Failed`] rather than through the restart policy.
    /// [`super::Supervisor::spawn`] already refuses a header-invalid key before anything launches,
    /// so this is a second line of defence, not the expected path.
    #[error("the readiness request could not be built: {0}")]
    InvalidReadinessRequest(String),

    /// A child retired after a crash or failed readiness check could not be confirmed stopped, even
    /// when asked again once the backoff delay had elapsed. Launching again would run two servers
    /// side by side, competing for VRAM and the port, so the driver ends in
    /// [`super::LifecycleState::Failed`] instead. The user can stop the stray process and start
    /// again; [`super::Supervisor::stop`] then reports [`super::LifecycleError::StopNotConfirmed`].
    #[error("a previous child could not be confirmed stopped; not relaunching beside it")]
    PreviousChildNotStopped,
}

impl FailureReason {
    /// The user setting that would let this failure through, if it is a preflight refusal a switch
    /// governs; `None` for every other reason. See
    /// [`crate::preflight::PreflightError::governing_protection`].
    pub fn governing_protection(&self) -> Option<crate::protections::Protection> {
        match self {
            FailureReason::PreflightRefused { protection, .. }
            | FailureReason::PreflightRefusedPermanently { protection, .. } => *protection,
            _ => None,
        }
    }

    /// Whether this reason names the specific bind-failure race [`PortBindFailure`] exists to
    /// recover from — [`super::run`] checks this to decide whether a failure earns the one
    /// immediate new-port retry or an ordinary backed-off restart.
    ///
    /// [`PortBindFailure`]: FailureReason::PortBindFailure
    pub(crate) fn is_port_bind_failure(&self) -> bool {
        matches!(self, FailureReason::PortBindFailure { .. })
    }

    /// Whether this reason is a configuration error that every retry would repeat identically, so
    /// [`super::run`] moves straight to [`super::LifecycleState::Failed`] instead of restarting:
    /// an unbuildable readiness request, a permanent preflight refusal (see
    /// [`crate::preflight::PreflightError::is_permanent`]), argv the flag allowlist rejects (built
    /// from the same flags and spec every attempt), or a 401 from our own child (the configured
    /// `api_key` is not in the key file the server reads). Transient refusals such as a busy port stay on the restart path.
    pub(crate) fn is_configuration_error(&self) -> bool {
        matches!(
            self,
            FailureReason::InvalidReadinessRequest(_)
                | FailureReason::PreflightRefusedPermanently { .. }
                | FailureReason::ArgvRejected(_)
                | FailureReason::Unauthorized
                | FailureReason::PreviousChildNotStopped
        )
    }
}

/// Pull the port number back out of the bind-failure log line itself (`"...port: %d"`), by finding
/// the last `"port: "` substring and reading the ASCII digits that follow it — not a regex, for the
/// same reason [`PORT_BIND_FAILURE_MARKER`] is matched as a plain substring: this is untrusted child
/// output, and a hand-rolled scan cannot itself be steered into matching something unintended the way
/// a crafted string could coax a more general pattern into doing.
///
/// The log line is the authoritative source for which port actually failed to bind — the caller-known
/// "requested" port can be [`crate::preflight::PortRequest::Any`], meaning no specific port was even
/// asked for, so this is often the only place the real number is available at all.
fn parse_bound_port(line: &str) -> Option<u16> {
    let (_, after) = line.rsplit_once("port: ")?;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Classify a child that exited (or whose recent output names a bind failure) before readiness was
/// ever observed during this launch attempt.
///
/// Checked in this order — bind-failure marker first — because a bind failure is itself one kind of
/// early exit: without checking the log first, every bind failure would be misreported as a bare
/// [`FailureReason::ExitedBeforeReady`], hiding the one condition [`super::run`] needs to tell apart
/// from every other exit reason.
///
/// `requested_port` is used only as a fallback, for the unlikely case the bind-failure log line does
/// not parse (an upstream wording change, say) — [`parse_bound_port`] is preferred whenever it
/// succeeds, since it names the port that actually failed rather than merely the one asked for.
pub(crate) fn classify_exit_before_ready(
    requested_port: u16,
    exit_code: Option<i32>,
    last_lines: &[String],
) -> FailureReason {
    match last_lines.iter().find(|line| line.contains(PORT_BIND_FAILURE_MARKER)) {
        Some(line) => FailureReason::PortBindFailure {
            port: parse_bound_port(line).unwrap_or(requested_port),
        },
        None => FailureReason::ExitedBeforeReady {
            exit_code,
            last_lines: last_lines.to_vec(),
        },
    }
}

/// Turn a [`ReadinessError`] that ended [`super::readiness::wait_ready`] into a [`FailureReason`],
/// for the case where the child was still running when the readiness probe gave up.
///
/// [`ReadinessError::Unreachable`] never reaches here: [`super::readiness::wait_ready`] treats it the
/// same as "still loading" and keeps polling internally, so it can only ever surface bundled into a
/// [`ReadinessError::DeadlineExceeded`] once the deadline itself elapses. [`ReadinessError::NonLoopbackHost`]
/// likewise cannot occur in practice — [`super::run`] always calls it with [`crate::preflight::LOOPBACK_HOST`]
/// — but is mapped to a reason rather than left for a `match` to reject, so this function stays total.
///
/// [`ReadinessError::ForeignListener`] maps to [`FailureReason::PortBindFailure`] on `port`: another
/// process won the port, which is exactly the race the one fresh-port retry exists for.
pub(crate) fn classify_readiness_error(err: ReadinessError, port: u16) -> FailureReason {
    match err {
        ReadinessError::ForeignListener => FailureReason::PortBindFailure { port },
        ReadinessError::DeadlineExceeded(deadline) => FailureReason::ReadinessDeadlineExceeded { deadline },
        ReadinessError::Unauthorized => FailureReason::Unauthorized,
        ReadinessError::UnexpectedStatus(status) => FailureReason::UnexpectedReadinessStatus(status),
        ReadinessError::InvalidRequest(cause) => FailureReason::InvalidReadinessRequest(cause),
        ReadinessError::Unreachable(cause) | ReadinessError::NonLoopbackHost(cause) => {
            FailureReason::ChildPollFailed(cause)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_naming_the_bind_failure_is_classified_as_port_bind_failure() {
        let lines = vec![
            "srv  init: initializing server".to_string(),
            "srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080".to_string(),
        ];
        let reason = classify_exit_before_ready(8080, Some(1), &lines);
        assert_eq!(reason, FailureReason::PortBindFailure { port: 8080 });
        assert!(reason.is_port_bind_failure());
    }

    #[test]
    fn the_port_in_the_log_line_wins_over_a_mismatched_requested_port() {
        // The requested port passed in (0, e.g. `PortRequest::Any`) is deliberately wrong here, to
        // prove the log line's own port number is what gets reported, not the caller's guess.
        let lines = vec!["srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080".to_string()];
        let reason = classify_exit_before_ready(0, Some(1), &lines);
        assert_eq!(reason, FailureReason::PortBindFailure { port: 8080 });
    }

    #[test]
    fn an_unparseable_bind_failure_line_falls_back_to_the_requested_port() {
        let lines = vec!["srv  init: couldn't bind HTTP server socket, hostname: 127.0.0.1".to_string()];
        let reason = classify_exit_before_ready(8080, Some(1), &lines);
        assert_eq!(reason, FailureReason::PortBindFailure { port: 8080 });
    }

    #[test]
    fn an_ordinary_exit_with_no_bind_marker_is_classified_as_exited_before_ready() {
        let lines = vec!["srv  init: loading model".to_string(), "error: out of memory".to_string()];
        let reason = classify_exit_before_ready(8080, Some(134), &lines);
        assert_eq!(
            reason,
            FailureReason::ExitedBeforeReady {
                exit_code: Some(134),
                last_lines: lines,
            }
        );
        assert!(!reason.is_port_bind_failure());
    }

    #[test]
    fn readiness_deadline_exceeded_maps_through() {
        let reason = classify_readiness_error(
            ReadinessError::DeadlineExceeded(Duration::from_secs(30)),
            8080,
        );
        assert_eq!(
            reason,
            FailureReason::ReadinessDeadlineExceeded {
                deadline: Duration::from_secs(30)
            }
        );
    }

    #[test]
    fn an_unbuildable_readiness_request_is_a_configuration_error() {
        let reason =
            classify_readiness_error(ReadinessError::InvalidRequest("bad header".into()), 8080);
        assert_eq!(
            reason,
            FailureReason::InvalidReadinessRequest("bad header".into())
        );
        assert!(reason.is_configuration_error());
    }

    #[test]
    fn unauthorized_and_argv_rejected_are_configuration_errors() {
        assert!(FailureReason::Unauthorized.is_configuration_error());
        assert!(FailureReason::ArgvRejected("duplicate flag".into()).is_configuration_error());
        assert!(
            FailureReason::PreflightRefusedPermanently {
                message: "elevated".into(),
                protection: Some(crate::protections::Protection::ElevationRefusal),
            }
            .is_configuration_error()
        );
        assert!(FailureReason::PreviousChildNotStopped.is_configuration_error());
    }

    #[test]
    fn transient_failures_stay_on_the_restart_path() {
        for reason in [
            FailureReason::PreflightRefused {
                message: "port busy".into(),
                protection: None,
            },
            FailureReason::SpawnFailed("os error".into()),
            FailureReason::ChildPollFailed("os error".into()),
            FailureReason::UnexpectedReadinessStatus(500),
            FailureReason::ReadinessDeadlineExceeded {
                deadline: Duration::from_secs(1),
            },
            FailureReason::ExitedBeforeReady {
                exit_code: Some(1),
                last_lines: Vec::new(),
            },
            FailureReason::CrashedAfterReady {
                exit_code: Some(1),
                last_lines: Vec::new(),
            },
            FailureReason::PortBindFailure { port: 8080 },
        ] {
            assert!(!reason.is_configuration_error(), "{reason:?}");
        }
    }

    #[test]
    fn readiness_unauthorized_maps_through() {
        assert_eq!(
            classify_readiness_error(ReadinessError::Unauthorized, 8080),
            FailureReason::Unauthorized
        );
    }

    #[test]
    fn a_foreign_listener_is_the_port_race_on_that_port() {
        let reason = classify_readiness_error(ReadinessError::ForeignListener, 8123);
        assert_eq!(reason, FailureReason::PortBindFailure { port: 8123 });
        assert!(reason.is_port_bind_failure());
        assert!(!reason.is_configuration_error());
    }
}
