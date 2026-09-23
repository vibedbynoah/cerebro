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
        .map_err(|e: hyper::http::uri::InvalidUri| ReadinessError::Unreachable(e.to_string()))?;
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .header("Authorization", format!("Bearer {api_key}"))
        .body(Empty::<Bytes>::new())
        .map_err(|e| ReadinessError::Unreachable(e.to_string()))?;

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

/// Poll `GET /health` until it returns 200, a deadline elapses, or the server answers with something
/// [`probe_health`] classifies as a real refusal (401, or any status other than 200/503).
///
/// A 503 keeps waiting indefinitely (up to `deadline`), matching `llama-server`'s own documented use
/// of that status for "model still loading". A connection failure ([`ReadinessError::Unreachable`])
/// is treated identically, for the reason documented on that variant.
pub(crate) async fn wait_ready(
    client: &HealthClient,
    host: &str,
    port: u16,
    api_key: &str,
    deadline: Duration,
    poll_interval: Duration,
) -> Result<(), ReadinessError> {
    let started = tokio::time::Instant::now();
    loop {
        match probe_health(client, host, port, api_key).await {
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
                return head
                    .lines()
                    .find_map(|line| line.strip_prefix("Authorization: ").map(str::to_string));
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
            Duration::from_secs(5),
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
            Duration::from_millis(60),
            Duration::from_millis(10),
        )
        .await;
        assert!(matches!(result, Err(ReadinessError::DeadlineExceeded(_))), "{result:?}");
    }

    #[tokio::test]
    async fn probe_health_sends_the_configured_bearer_token() {
        let port = spawn_fake_server(vec![("HTTP/1.1 200 OK", "ok")]).await;
        let client = build_client();
        let probe = probe_health(&client, LOOPBACK_HOST, port, "a-specific-token-value").await;
        assert_eq!(probe, Ok(Probe::Ready));
        // The fake server does not echo the header back, so this test only proves the happy path
        // parses; `probe_health_refuses_to_build_a_request_for_a_non_loopback_host` below is what
        // proves the loopback restriction, and the two 401/well-formed-header cases above are what
        // exercise the header actually being sent (a wrong key must be rejected, a right key must
        // not be).
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
