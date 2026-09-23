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
    /// (ARCHITECTURE.md section 8): a backend file changed hash, the DACL turned unsafe, a model
    /// went missing, and so on.
    #[error("preflight refused the (re-)launch: {0}")]
    PreflightRefused(String),

    /// Building the argv for this launch failed (a duplicate or preflight-owned flag, a non-UTF-8
    /// path, a violated cross-flag constraint).
    #[error("could not build argv for the launch: {0}")]
    ArgvRejected(String),

    /// The OS-level spawn call itself failed.
    #[error("could not spawn the child: {0}")]
    SpawnFailed(String),

    /// The health check rejected the configured API key (401). The child is stopped before this
    /// reason is recorded — a server answering with the wrong key accepted is not one to keep
    /// running unauthenticated.
    #[error("the health check rejected the configured API key")]
    Unauthorized,

    /// The health check returned a status neither 200 (ready) nor 503 (loading).
    #[error("the health check returned unexpected status {0}")]
    UnexpectedReadinessStatus(u16),

    /// Polling the child's own exit status failed at the OS level.
    #[error("could not poll the child's exit status: {0}")]
    ChildPollFailed(String),
}

impl FailureReason {
    /// Whether this reason names the specific bind-failure race [`PortBindFailure`] exists to
    /// recover from — [`super::run`] checks this to decide whether a failure earns the one
    /// immediate new-port retry or an ordinary backed-off restart.
    ///
    /// [`PortBindFailure`]: FailureReason::PortBindFailure
    pub(crate) fn is_port_bind_failure(&self) -> bool {
        matches!(self, FailureReason::PortBindFailure { .. })
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
pub(crate) fn classify_readiness_error(err: ReadinessError) -> FailureReason {
    match err {
        ReadinessError::DeadlineExceeded(deadline) => FailureReason::ReadinessDeadlineExceeded { deadline },
        ReadinessError::Unauthorized => FailureReason::Unauthorized,
        ReadinessError::UnexpectedStatus(status) => FailureReason::UnexpectedReadinessStatus(status),
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
        let reason = classify_readiness_error(ReadinessError::DeadlineExceeded(Duration::from_secs(30)));
        assert_eq!(
            reason,
            FailureReason::ReadinessDeadlineExceeded {
                deadline: Duration::from_secs(30)
            }
        );
    }

    #[test]
    fn readiness_unauthorized_maps_through() {
        assert_eq!(
            classify_readiness_error(ReadinessError::Unauthorized),
            FailureReason::Unauthorized
        );
    }
}
