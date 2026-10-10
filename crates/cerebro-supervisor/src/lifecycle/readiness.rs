//! Polling a spawned `llama-server`'s `/health` endpoint until it reports ready.
//!
//! # Why hyper + hyper-util, not reqwest
//!
//! Every request this module ever makes targets `127.0.0.1` over plain HTTP — there is no TLS
//! surface to speak of. reqwest's default build pulls in a full TLS stack (a `rustls`/`ring`
//! certificate-verification chain, `webpki-roots`, …) purely to *offer* a codepath a loopback-only
//! health poll will never take. hyper 1 plus hyper-util's connection-pooled client gives the same
//! request/response shape — headers, status codes, an async body — with a smaller, better-scoped
//! footprint: no TLS crate, no cookie jar, no redirect follower, none of which this module has any
//! use for. Both are equally "small, well-maintained" per this component's own brief; hyper+hyper-util
//! is the one that does not have to be told not to speak TLS.
//!
//! # Loopback only, structurally
//!
//! [`probe_health`] never accepts an arbitrary URL from a caller — it takes a bare `host`/`port` and
//! refuses to build a request at all unless `host` is exactly [`crate::preflight::LOOPBACK_HOST`].
//! That is what "refuse any non-loopback URL construction" means here: the refusal happens before a
//! [`hyper::Uri`] is even parsed, not as a downstream connect failure a caller could mistake for
//! "not ready yet".

use std::time::Duration;

use bytes::Bytes;
use http_body_util::Empty;
use hyper::{Request, StatusCode};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use thiserror::Error;

use crate::preflight::LOOPBACK_HOST;

use super::listener::ListenerOwner;

/// A plain-HTTP client, restricted by [`probe_health`] to this module's own loopback health checks.
/// See this module's top-level docs for why hyper+hyper-util rather than reqwest.
pub(crate) type HealthClient = Client<HttpConnector, Empty<Bytes>>;

/// Build a fresh client. Cheap to call: the connector and connection pool are what actually hold
/// resources, and a [`Client`] value is a thin, cloneable handle onto them.
pub(crate) fn build_client() -> HealthClient {
    Client::builder(TokioExecutor::new()).build_http()
}

/// One `/health` probe's outcome, for the statuses `llama-server` documents itself returning while a
/// model is loading versus once it is ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Probe {
    /// `GET /health` returned 200: the server is ready.
    Ready,
    /// `GET /health` returned 503: the model is still loading. Not a failure — [`wait_ready`] keeps
    /// polling.
    Loading,
}

/// Everything that can make readiness polling give up before observing [`Probe::Ready`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ReadinessError {
    /// A request was about to be built for a host other than [`LOOPBACK_HOST`]. Every server
    /// [`crate::lifecycle`] supervises is loopback-only by construction (see
    /// [`crate::preflight::LaunchSpec`]), so a readiness probe that could reach anywhere else would
    /// be a bug; refusing to build the request at all makes that unrepresentable rather than merely
    /// unlikely.
    #[error("refusing to build a health-check request for non-loopback host {0:?}")]
    NonLoopbackHost(String),
    /// The request itself could not be built: an invalid URI, or an API key containing bytes a
    /// header value may not hold (CR, LF, other control characters). A configuration error that
    /// polling again can never fix, so [`wait_ready`] returns it at once instead of treating it like
    /// [`ReadinessError::Unreachable`] and waiting out the whole deadline. The message never contains
    /// the key itself.
    #[error("could not build the health-check request: {0}")]
    InvalidRequest(String),
    /// The request could not even be sent — most commonly, nothing is listening on the port yet,
    /// because a freshly spawned `llama-server` does not start accepting connections the instant its
    /// OS process exists. [`wait_ready`] treats this the same as [`Probe::Loading`] and keeps
    /// polling; it is surfaced as its own variant only so [`probe_health`]'s caller can tell "not
    /// listening yet" apart from a real protocol-level refusal.
    #[error("could not reach the health endpoint: {0}")]
    Unreachable(String),
    /// The server rejected the bearer token: `GET /health` returned 401.
    #[error("health check returned 401 Unauthorized: the configured API key was rejected")]
    Unauthorized,
    /// Any other, unexpected status.
    #[error("health check returned unexpected status {0}")]
    UnexpectedStatus(u16),
    /// The listener on the port belongs to another process (see [`ListenerOwner::Foreign`]): it
    /// won the port after preflight released it. Nothing was sent to it, the bearer key included.
    #[error("another process owns the listener on the health-check port")]
    ForeignListener,
    /// [`wait_ready`]'s deadline elapsed before a 200 was ever seen.
    #[error("readiness was not reached within {0:?}")]
    DeadlineExceeded(Duration),
}

/// Send one `GET /health` request with `Authorization: Bearer <api_key>` and classify the response.
///
/// `host` is checked against [`LOOPBACK_HOST`] before anything else — see this module's top-level
/// docs on why that check has to happen here, not merely be true in practice because every caller
/// today happens to pass the right constant.
pub(crate) async fn probe_health(
    client: &HealthClient,
    host: &str,
    port: u16,
    api_key: &str,
) -> Result<Probe, ReadinessError> {
    if host != LOOPBACK_HOST {
        return Err(ReadinessError::NonLoopbackHost(host.to_string()));
    }

    let uri: hyper::Uri = format!("http://{host}:{port}/health")
        .parse()
        .map_err(|e: hyper::http::uri::InvalidUri| ReadinessError::InvalidRequest(e.to_string()))?;
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .header("Authorization", format!("Bearer {api_key}"))
        .body(Empty::<Bytes>::new())
        // `hyper::http::Error`'s message names only the failure class ("invalid HTTP header
        // value"), never the offending value, so the key cannot leak through this error.
        .map_err(|e| ReadinessError::InvalidRequest(e.to_string()))?;

    let response = client
        .request(request)
        .await
        .map_err(|e| ReadinessError::Unreachable(e.to_string()))?;

    match response.status() {
        StatusCode::OK => Ok(Probe::Ready),
        StatusCode::SERVICE_UNAVAILABLE => Ok(Probe::Loading),
        StatusCode::UNAUTHORIZED => Err(ReadinessError::Unauthorized),
        other => Err(ReadinessError::UnexpectedStatus(other.as_u16())),
    }
}

/// Poll `GET /health` until it returns 200, a deadline elapses, the listener turns out to belong to
/// another process, or the server answers with something [`probe_health`] classifies as a real
/// refusal (401, or any status other than 200/503).
///
/// # Ownership before anything is sent
///
/// `owner` reports who owns the listener on `port` (see `super::listener`). No request, and so no
/// bearer key, is sent until it answers [`ListenerOwner::Ours`]: [`ListenerOwner::Foreign`] returns
/// [`ReadinessError::ForeignListener`] at once, and [`ListenerOwner::NoneYet`] or
/// [`ListenerOwner::Unknown`] keep waiting (up to `deadline`) without sending anything, so a
/// listener that cannot be attributed is never assumed to be ours. Once ownership is proven it is
/// not asked again: the child holds the port for as long as it lives, and the caller races this
/// future against the child's own exit.
///
/// A 503 keeps waiting indefinitely (up to `deadline`), matching `llama-server`'s own documented use
/// of that status for "model still loading". A connection failure ([`ReadinessError::Unreachable`])
/// is treated identically, for the reason documented on that variant. Every other error, including
/// [`ReadinessError::InvalidRequest`], is returned immediately.
///
/// `deadline` bounds each individual probe too, not only the gaps between them: the HTTP client has
/// no request timeout of its own, so a server that accepts the connection but never answers would
/// otherwise hold a single `probe_health` call open forever and the deadline would never be checked.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn wait_ready(
    client: &HealthClient,
    host: &str,
    port: u16,
    api_key: &str,
    owner: &(dyn Fn() -> ListenerOwner + Sync),
    deadline: Duration,
    poll_interval: Duration,
) -> Result<(), ReadinessError> {
    let started = tokio::time::Instant::now();
    let mut verified = false;
    loop {
        if !verified {
            match owner() {
                ListenerOwner::Ours => verified = true,
                ListenerOwner::Foreign => return Err(ReadinessError::ForeignListener),
                ListenerOwner::NoneYet | ListenerOwner::Unknown => {
                    if started.elapsed() >= deadline {
                        return Err(ReadinessError::DeadlineExceeded(deadline));
                    }
                    tokio::time::sleep(poll_interval).await;
                    continue;
                }
            }
        }
        let remaining = deadline.saturating_sub(started.elapsed());
        let Ok(outcome) =
            tokio::time::timeout(remaining, probe_health(client, host, port, api_key)).await
        else {
            return Err(ReadinessError::DeadlineExceeded(deadline));
        };
        match outcome {
            Ok(Probe::Ready) => return Ok(()),
            Ok(Probe::Loading) | Err(ReadinessError::Unreachable(_)) => {}
            Err(other) => return Err(other),
        }
        if started.elapsed() >= deadline {
            return Err(ReadinessError::DeadlineExceeded(deadline));
        }
        tokio::time::sleep(poll_interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    const TEST_KEY: &str = "test-only-fixture-key-not-a-real-secret";

    /// The ownership answer for a fake server run by this test process: it is "our child".
    fn ours() -> ListenerOwner {
        ListenerOwner::Ours
    }

    /// Read one HTTP/1.1 request off `stream` far enough to recover its `Authorization` header,
    /// discarding the rest — every request this test suite sends is a bodyless `GET`, so nothing
    /// past the header block matters.
    async fn read_authorization(stream: &mut TcpStream) -> Option<String> {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 512];
        loop {
            let n = stream.read(&mut chunk).await.ok()?;
            if n == 0 {
                return None;
            }
            buf.extend_from_slice(&chunk[..n]);
            if let Some(end) = find_header_end(&buf) {
                let head = String::from_utf8_lossy(&buf[..end]);
                // Header names are case-insensitive (RFC 9110), and hyper writes them lowercase on
                // the wire, so match the name without regard to case.
                return head.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.trim()
                        .eq_ignore_ascii_case("authorization")
                        .then(|| value.trim().to_string())
                });
            }
        }
    }

    fn find_header_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|w| w == b"\r\n\r\n")
    }

    async fn respond(stream: &mut TcpStream, status_line: &str, body: &str) {
        let response =
            format!("{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        let _ = stream.write_all(response.as_bytes()).await;
        let _ = stream.shutdown().await;
    }

    /// A minimal, hand-rolled HTTP/1.1 server, not `hyper`'s own server half: this module's
    /// production code only ever needs an HTTP *client* (see this module's top-level docs), so
    /// pulling in `hyper`'s `server` feature purely for a test fixture would widen the dependency
    /// footprint the client-only choice was made to avoid. Every request this fixture accepts is
    /// answered with `responses[i]` in order, then the listener is dropped once exhausted.
    async fn spawn_fake_server(responses: Vec<(&'static str, &'static str)>) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener.local_addr().expect("read back the bound port").port();
        tokio::spawn(async move {
            for (status_line, body) in responses {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let _ = read_authorization(&mut stream).await;
                respond(&mut stream, status_line, body).await;
            }
        });
        port
    }

    #[tokio::test]
    async fn wait_ready_keeps_polling_through_503_then_succeeds_on_200() {
        let port = spawn_fake_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "loading"),
            ("HTTP/1.1 503 Service Unavailable", "loading"),
            ("HTTP/1.1 200 OK", "ok"),
        ])
        .await;
        let client = build_client();
        let result = wait_ready(
            &client,
            LOOPBACK_HOST,
            port,
            TEST_KEY,
            &ours,
            // Real clock: generous so a loaded machine cannot turn three instant answers into a
            // spurious deadline miss. It is never waited out when the code is right.
            Duration::from_secs(30),
            Duration::from_millis(10),
        )
        .await;
        assert_eq!(result, Ok(()));
    }

    #[tokio::test]
    async fn wait_ready_fails_fast_on_401_without_waiting_for_the_deadline() {
        let port = spawn_fake_server(vec![("HTTP/1.1 401 Unauthorized", "unauthorized: Invalid API Key")]).await;
        let client = build_client();
        let started = std::time::Instant::now();
        let result = wait_ready(
            &client,
            LOOPBACK_HOST,
            port,
            "the-wrong-key",
            &ours,
            Duration::from_secs(30),
            Duration::from_millis(10),
        )
        .await;
        assert_eq!(result, Err(ReadinessError::Unauthorized));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a 401 must not wait out the readiness deadline"
        );
    }

    #[tokio::test]
    async fn wait_ready_times_out_if_only_503_is_ever_returned() {
        let port = spawn_fake_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "loading"),
            ("HTTP/1.1 503 Service Unavailable", "loading"),
            ("HTTP/1.1 503 Service Unavailable", "loading"),
        ])
        .await;
        let client = build_client();
        let result = wait_ready(
            &client,
            LOOPBACK_HOST,
            port,
            TEST_KEY,
            &ours,
            Duration::from_millis(60),
            Duration::from_millis(10),
        )
        .await;
        assert!(matches!(result, Err(ReadinessError::DeadlineExceeded(_))), "{result:?}");
    }

    #[tokio::test]
    async fn probe_health_sends_the_configured_bearer_token() {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        let (seen_tx, seen_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let auth = read_authorization(&mut stream).await;
            respond(&mut stream, "HTTP/1.1 200 OK", "ok").await;
            let _ = seen_tx.send(auth);
        });

        let client = build_client();
        let probe = probe_health(&client, LOOPBACK_HOST, port, "a-specific-token-value").await;
        assert_eq!(probe, Ok(Probe::Ready));
        let seen = seen_rx.await.expect("the fake server saw one request");
        assert_eq!(seen.as_deref(), Some("Bearer a-specific-token-value"));
    }

    #[tokio::test]
    async fn wait_ready_gives_up_at_the_deadline_on_a_server_that_never_answers() {
        // Accepts the connection, then holds it open forever without responding: the HTTP client
        // has no timeout of its own, so only `wait_ready` bounding each probe by the deadline can
        // end this. Held connections are kept alive in the task until the test ends.
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind a loopback port for the fake health server");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
            }
        });

        let client = build_client();
        let result = tokio::time::timeout(
            Duration::from_secs(60),
            wait_ready(
                &client,
                LOOPBACK_HOST,
                port,
                TEST_KEY,
                &ours,
                Duration::from_millis(200),
                Duration::from_millis(10),
            ),
        )
        .await
        .expect("wait_ready must return at its own deadline, not hang on an unanswered request");
        assert_eq!(
            result,
            Err(ReadinessError::DeadlineExceeded(Duration::from_millis(200)))
        );
    }

    #[tokio::test]
    async fn a_header_invalid_api_key_fails_fast_with_invalid_request() {
        // Port 1: nothing listens there, so if the request were ever built and sent, the probe would
        // come back `Unreachable` and `wait_ready` would poll out the whole 60s deadline instead.
        let started = std::time::Instant::now();
        let result = wait_ready(
            &build_client(),
            LOOPBACK_HOST,
            1,
            "bad\r\nkey",
            &ours,
            Duration::from_secs(60),
            Duration::from_millis(10),
        )
        .await;
        let Err(ReadinessError::InvalidRequest(message)) = &result else {
            panic!("expected InvalidRequest, got {result:?}");
        };
        assert!(
            !message.contains("bad"),
            "the key must not appear in the error: {message}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a request that cannot be built must not wait out the readiness deadline"
        );
    }

    /// A bound, never-accepted std listener: any connection a probe opened (even one closed again)
    /// is left in its accept backlog, so a non-blocking `accept` answering `WouldBlock` proves no
    /// connection was ever made, with no wall-clock wait.
    fn silent_listener() -> (std::net::TcpListener, u16) {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind a loopback port");
        listener
            .set_nonblocking(true)
            .expect("make the listener non-blocking");
        let port = listener
            .local_addr()
            .expect("read back the bound port")
            .port();
        (listener, port)
    }

    fn assert_never_contacted(listener: &std::net::TcpListener) {
        match listener.accept() {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            other => panic!("the listener must never have been contacted: {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_foreign_listener_fails_at_once_without_any_request() {
        let (listener, port) = silent_listener();
        let started = std::time::Instant::now();
        let result = wait_ready(
            &build_client(),
            LOOPBACK_HOST,
            port,
            TEST_KEY,
            &|| ListenerOwner::Foreign,
            Duration::from_secs(60),
            Duration::from_millis(10),
        )
        .await;
        assert_eq!(result, Err(ReadinessError::ForeignListener));
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "a foreign listener must not wait out the deadline"
        );
        assert_never_contacted(&listener);
    }

    #[tokio::test]
    async fn an_unattributed_listener_is_never_contacted_before_the_deadline() {
        for owner in [ListenerOwner::NoneYet, ListenerOwner::Unknown] {
            let (listener, port) = silent_listener();
            let result = wait_ready(
                &build_client(),
                LOOPBACK_HOST,
                port,
                TEST_KEY,
                &move || owner,
                Duration::from_millis(50),
                Duration::from_millis(5),
            )
            .await;
            assert_eq!(
                result,
                Err(ReadinessError::DeadlineExceeded(Duration::from_millis(50))),
                "{owner:?}"
            );
            assert_never_contacted(&listener);
        }
    }

    #[tokio::test]
    async fn ownership_is_asked_until_proven_then_never_again() {
        let port = spawn_fake_server(vec![
            ("HTTP/1.1 503 Service Unavailable", "loading"),
            ("HTTP/1.1 503 Service Unavailable", "loading"),
            ("HTTP/1.1 200 OK", "ok"),
        ])
        .await;
        let asked = std::sync::atomic::AtomicU32::new(0);
        let owner = || {
            // NoneYet twice, then Ours.
            if asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                ListenerOwner::NoneYet
            } else {
                ListenerOwner::Ours
            }
        };
        let result = wait_ready(
            &build_client(),
            LOOPBACK_HOST,
            port,
            TEST_KEY,
            &owner,
            Duration::from_secs(30),
            Duration::from_millis(5),
        )
        .await;
        assert_eq!(result, Ok(()));
        // Two not-yet answers, one Ours, then three probes with no further table reads.
        assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn probe_health_refuses_a_non_loopback_host_without_a_runtime() {
        // Exercises the refusal as a pure, synchronous check by driving `probe_health` on a
        // single-threaded block_on rather than requiring `#[tokio::test]`: the refusal happens
        // before any `.await` point at all, so no fake server or client connection is needed to
        // observe it.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build a minimal current-thread runtime");
        let client = build_client();
        let result = rt.block_on(probe_health(&client, "10.0.0.1", 8080, TEST_KEY));
        assert_eq!(result, Err(ReadinessError::NonLoopbackHost("10.0.0.1".to_string())));
    }
}
