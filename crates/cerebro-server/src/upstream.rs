//! The one llama-server the gateway forwards to, over loopback HTTP/1.1.
//!
//! Requests carry llama-server's own per-launch key (`Authorization: Bearer <key>`), never anything
//! from the client's headers: the client's gateway token stops at the gateway. Only `Content-Type`
//! is copied back from upstream responses, so llama-server's own CORS or other headers never reach
//! a browser.
//!
//! The upstream is usable only while its lifecycle state is [`LifecycleState::Ready`]; at any other
//! time requests fail fast with 503 rather than queueing against a server that is starting or
//! restarting. The port can change across restarts (a port race gets one fresh-port retry), so the
//! target is read per request from a [`TargetSource`]: in `serve` that is the supervisor's own
//! launch record, which the supervisor replaces before it publishes `Ready` for a new launch. The
//! target is read before and after the state, and a request goes out only when both reads name the
//! same launch, so a request never pairs one launch's `Ready` with another launch's port.

use std::sync::Arc;
use std::time::Duration;

use axum::http::{header, HeaderValue, Method, Request, Response, StatusCode};
use bytes::Bytes;
use cerebro_supervisor::LifecycleState;
use http_body_util::{BodyExt as _, Full, Limited};
use hyper::body::Incoming;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::Value;
use tokio::sync::watch;

use crate::error::{ApiError, ServerError};
use crate::token::BearerToken;

/// Largest upstream response body the gateway buffers (non-streaming chat, which it edits to add
/// the `cerebro` field, and the internal `/props` and `/tokenize` calls).
pub const MAX_BUFFERED_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// One launch of llama-server: its process id and the loopback port it listens on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpstreamTarget {
    /// The launch's process id (tells two launches on the same port apart).
    pub pid: u32,
    /// The loopback port it listens on.
    pub port: u16,
}

/// Where the current launch listens; `None` before the first launch or after shutdown. Must be
/// updated before the lifecycle state reads `Ready` for that launch (the supervisor's launch record
/// is).
pub type TargetSource = Arc<dyn Fn() -> Option<UpstreamTarget> + Send + Sync>;

/// A llama-server reachable on loopback.
pub struct Upstream {
    client: Client<HttpConnector, Full<Bytes>>,
    target: TargetSource,
    /// `Bearer <key>`, built once and marked sensitive; cloning it per request is a refcount bump.
    authorization: HeaderValue,
    state: watch::Receiver<LifecycleState>,
}

impl std::fmt::Debug for Upstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upstream")
            .field("target", &(self.target)())
            .field("authorization", &"<redacted>")
            .field("state", &*self.state.borrow())
            .finish()
    }
}

impl Upstream {
    /// An upstream at whatever `target` names, authenticated with `key`, usable while `state` reads
    /// [`LifecycleState::Ready`].
    ///
    /// # Errors
    /// [`ServerError::InvalidConfig`] if `key` is not a valid header value (a [`BearerToken`] always
    /// is).
    pub fn new(
        target: TargetSource,
        key: &BearerToken,
        state: watch::Receiver<LifecycleState>,
        connect_timeout: Duration,
    ) -> Result<Self, ServerError> {
        let mut authorization = HeaderValue::try_from(format!("Bearer {}", key.reveal()))
            .map_err(|_| ServerError::InvalidConfig("the upstream key is not a header value".to_string()))?;
        authorization.set_sensitive(true);
        let mut connector = HttpConnector::new();
        connector.set_connect_timeout(Some(connect_timeout));
        connector.set_nodelay(true);
        connector.enforce_http(true);
        let client = Client::builder(TokioExecutor::new()).build(connector);
        Ok(Self { client, target, authorization, state })
    }

    /// The current lifecycle state.
    pub fn state(&self) -> LifecycleState {
        self.state.borrow().clone()
    }

    /// The current launch's port, `None` while unknown.
    pub fn port(&self) -> Option<u16> {
        (self.target)().map(|t| t.port)
    }

    /// The port to send to, or 503 when the server is not ready. The target is read on both sides
    /// of the state, so `Ready` and the port provably belong to the same launch.
    fn ready_port(&self) -> Result<u16, ApiError> {
        let before = (self.target)();
        let state = self.state.borrow().clone();
        if state != LifecycleState::Ready {
            return Err(ApiError::unavailable(format!("the model server is not ready ({})", describe(&state))));
        }
        match (before, (self.target)()) {
            (Some(before), Some(after)) if before == after => Ok(after.port),
            (_, None) => Err(ApiError::unavailable("the model server's port is not known yet")),
            _ => Err(ApiError::unavailable("the model server is restarting")),
        }
    }

    /// Send one request and return the response head with its body still streaming. Fails with
    /// 504 when the response head does not arrive within `head_timeout`.
    pub async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<Bytes>,
        head_timeout: Duration,
    ) -> Result<Response<Incoming>, ApiError> {
        let port = self.ready_port()?;
        let mut builder = Request::builder()
            .method(method)
            .uri(format!("http://127.0.0.1:{port}{path}"))
            .header(header::HOST, format!("127.0.0.1:{port}"))
            .header(header::AUTHORIZATION, self.authorization.clone());
        if body.is_some() {
            builder = builder.header(header::CONTENT_TYPE, "application/json");
        }
        let request = builder
            .body(Full::new(body.unwrap_or_default()))
            .map_err(|_| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "could not build the upstream request"))?;
        match tokio::time::timeout(head_timeout, self.client.request(request)).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(ApiError::bad_upstream(format!("the model server did not answer: {}", connect_error_kind(&e)))),
            Err(_) => Err(ApiError::timeout(format!("the model server did not answer within {} s", head_timeout.as_secs()))),
        }
    }

    /// Send a JSON body and buffer the JSON response (bounded by
    /// [`MAX_BUFFERED_RESPONSE_BYTES`] and by `timeout` for the whole exchange).
    pub async fn json<B: serde::Serialize + ?Sized>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
        timeout: Duration,
    ) -> Result<(StatusCode, Value), ApiError> {
        let bytes = match body {
            Some(v) => Some(Bytes::from(serde_json::to_vec(v).map_err(|_| ApiError::bad_request("unserializable body"))?)),
            None => None,
        };
        self.json_bytes(method, path, bytes, timeout).await
    }

    /// [`Upstream::json`] with the body already serialized.
    pub async fn json_bytes(
        &self,
        method: Method,
        path: &str,
        bytes: Option<Bytes>,
        timeout: Duration,
    ) -> Result<(StatusCode, Value), ApiError> {
        let exchange = async {
            let response = self.send(method, path, bytes, timeout).await?;
            let status = response.status();
            let collected = Limited::new(response.into_body(), MAX_BUFFERED_RESPONSE_BYTES)
                .collect()
                .await
                .map_err(|_| ApiError::bad_upstream("the model server's response was cut off or too large"))?
                .to_bytes();
            let value: Value = serde_json::from_slice(&collected)
                .map_err(|_| ApiError::bad_upstream("the model server's response is not JSON"))?;
            Ok::<_, ApiError>((status, value))
        };
        tokio::time::timeout(timeout, exchange)
            .await
            .map_err(|_| ApiError::timeout(format!("the model server did not finish within {} s", timeout.as_secs())))?
    }
}

/// A short, content-free description of a lifecycle state for error messages.
pub(crate) fn describe(state: &LifecycleState) -> &'static str {
    match state {
        LifecycleState::Starting => "starting",
        LifecycleState::Ready => "ready",
        LifecycleState::Crashed(_) => "crashed, restarting",
        LifecycleState::Backoff { .. } => "restarting",
        LifecycleState::Stopped => "stopped",
        LifecycleState::Failed(_) => "failed",
    }
}

fn connect_error_kind(e: &hyper_util::client::legacy::Error) -> &'static str {
    if e.is_connect() {
        "connection refused or timed out"
    } else {
        "connection closed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn upstream(target: TargetSource, state: LifecycleState) -> (Upstream, watch::Sender<LifecycleState>) {
        let (tx, rx) = watch::channel(state);
        let key = BearerToken::from_raw(&[3; 32]).expect("key");
        (Upstream::new(target, &key, rx, Duration::from_secs(1)).expect("upstream"), tx)
    }

    #[tokio::test]
    async fn ready_and_the_port_must_belong_to_the_same_launch() {
        let stable: TargetSource = Arc::new(|| Some(UpstreamTarget { pid: 10, port: 4000 }));
        let (u, tx) = upstream(stable, LifecycleState::Ready);
        assert_eq!(u.ready_port().expect("ready"), 4000);
        tx.send_replace(LifecycleState::Starting);
        assert_eq!(u.ready_port().unwrap_err().status, StatusCode::SERVICE_UNAVAILABLE);

        // A new launch replaces the target between the two reads: never sent to either port.
        let calls = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&calls);
        let moving: TargetSource = Arc::new(move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            Some(UpstreamTarget { pid: 10 + n, port: 4000 + u16::try_from(n).unwrap_or(0) })
        });
        let (u, _tx) = upstream(moving, LifecycleState::Ready);
        let err = u.ready_port().unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.message.contains("restarting"), "{}", err.message);

        let none: TargetSource = Arc::new(|| None);
        let (u, _tx) = upstream(none, LifecycleState::Ready);
        assert!(u.ready_port().unwrap_err().message.contains("not known"));
    }

    #[test]
    fn the_authorization_header_is_built_once_and_sensitive() {
        let target: TargetSource = Arc::new(|| None);
        let (u, _tx) = upstream(target, LifecycleState::Ready);
        assert!(u.authorization.is_sensitive());
        assert_eq!(u.authorization.to_str().expect("ascii"), format!("Bearer {}", "03".repeat(32)));
        assert!(!format!("{u:?}").contains("0303"));
    }
}
