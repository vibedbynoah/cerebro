//! The HTTP gateway: configuration, routes and the loopback listener.
//!
//! Routes (every one behind [`crate::guard`]):
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /v1/models` | passthrough |
//! | `POST /v1/chat/completions` | effort and verbosity applied ([`crate::effort`]); streamed (SSE) or not; XHigh/Max run [`crate::passes`] first |
//! | `POST /v1/completions` | passthrough, streamed or not (`cerebro_*` fields removed; raw prompts get no instructions) |
//! | `POST /v1/embeddings` | passthrough |
//! | `GET /cerebro/health` | 200 when the model server is ready, else 503 |
//! | `GET /cerebro/status` | model, last measured speeds, VRAM, launch facts |
//!
//! Streams are relayed frame by frame as llama-server produces them, so the gateway adds no
//! buffering to time-to-first-token; each frame must arrive within the stream idle timeout. Nothing
//! here logs: prompts, completions and tokens never reach a log line.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use cerebro_config::schema::BindAddr;
use cerebro_config::ThinkingBudget;
use http_body_util::{BodyExt as _, Limited};
use hyper::body::Incoming;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use serde_json::{json, Map, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::effort::{self, Budget, ContextFit, Dials, Plan};
use crate::error::{json_body, ApiError, ServerError};
use crate::guard::{self, normalize_origin};
use crate::passes::{self, Completer, PassKind, PassRecord};
use crate::status::{self, Speed, Telemetry};
use crate::token::BearerToken;
use crate::upstream::{describe, Upstream};

/// Bytes of a relayed stream's end kept to read llama-server's final `timings`.
const STREAM_TAIL_BYTES: usize = 8 * 1024;

/// Default interval at which a streamed XHigh/Max reply sends an SSE comment while its passes run
/// ([`Timeouts::sse_keepalive`]), so clients with read timeouts keep the connection.
pub const SSE_KEEPALIVE: Duration = Duration::from_secs(15);

/// Default cap on concurrent client connections ([`GatewayConfig::max_connections`]): generous for
/// any real local client, and a bound on what a same-machine flood of idle sockets can hold before
/// authentication runs.
pub const MAX_CONNECTIONS: usize = 256;

/// Per-request time limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// Reading a request's headers (slow-client defence).
    pub header_read: Duration,
    /// Reading a request's body.
    pub body_read: Duration,
    /// Connecting to llama-server.
    pub connect: Duration,
    /// One non-streaming upstream call, start to finish (each XHigh/Max pass gets its own).
    pub generation: Duration,
    /// Longest gap between two frames of a relayed stream (prompt processing comes before the
    /// first one, so this also bounds it).
    pub stream_idle: Duration,
    /// Interval between SSE keepalive comments while a streamed XHigh/Max reply runs its passes.
    pub sse_keepalive: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            header_read: Duration::from_secs(10),
            body_read: Duration::from_secs(60),
            connect: Duration::from_secs(5),
            generation: Duration::from_secs(30 * 60),
            stream_idle: Duration::from_secs(10 * 60),
            sse_keepalive: SSE_KEEPALIVE,
        }
    }
}

/// The gateway's request checks, each on by default. A check switched off is honoured and reported
/// once, neutrally, under `gateway.disabled_protections` in `/cerebro/status`; nothing else nags.
/// Settings map onto it from `protections.gateway_*` ([`GatewayConfig::from_settings`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayProtections {
    /// A bearer token on every route, reads included.
    pub require_token: bool,
    /// The `Host` header must name the gateway on loopback (DNS-rebinding defence). Off skips only
    /// that header test: the listener still binds loopback and still refuses a non-local peer.
    pub check_host: bool,
    /// A browser `Origin` must be on the allowlist. Off accepts any origin and echoes it in the
    /// CORS headers (never `*`).
    pub check_origin: bool,
}

impl Default for GatewayProtections {
    fn default() -> Self {
        Self { require_token: true, check_host: true, check_origin: true }
    }
}

impl GatewayProtections {
    /// The protections settings describe (`protections.gateway_*`; a missing key means on).
    pub fn from_settings(protections: &cerebro_config::ProtectionsSection) -> Self {
        Self {
            require_token: protections.gateway_require_token,
            check_host: protections.gateway_check_host,
            check_origin: protections.gateway_check_origin,
        }
    }

    /// `(setting, risk)` for each check switched off, in the order the guard runs them; the
    /// wording is `cerebro-config`'s, so the settings screen and status say the same thing.
    pub fn disabled(&self) -> Vec<(&'static str, &'static str)> {
        use cerebro_config::policy::{GATEWAY_CHECK_HOST_RISK, GATEWAY_CHECK_ORIGIN_RISK, GATEWAY_REQUIRE_TOKEN_RISK};
        [
            (self.check_host, "gateway_check_host", GATEWAY_CHECK_HOST_RISK),
            (self.check_origin, "gateway_check_origin", GATEWAY_CHECK_ORIGIN_RISK),
            (self.require_token, "gateway_require_token", GATEWAY_REQUIRE_TOKEN_RISK),
        ]
        .into_iter()
        .filter(|(on, _, _)| !on)
        .map(|(_, setting, risk)| (setting, risk))
        .collect()
    }
}

/// How the gateway listens and what it allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConfig {
    /// Loopback address to bind (the settings type cannot express anything else).
    pub bind: BindAddr,
    /// Port to bind; `0` lets the OS choose (tests).
    pub port: u16,
    /// Browser origins allowed to call the gateway (exact, normalized). Empty by default, which
    /// keeps CORS closed; settings `api.allowed_origins` holds the desktop app's origin.
    pub allowed_origins: Vec<String>,
    /// Which request checks run; all on by default.
    pub protections: GatewayProtections,
    /// Largest request body accepted, in bytes.
    pub max_body_bytes: usize,
    /// Per-request time limits.
    pub timeouts: Timeouts,
    /// Most client connections served at once; a connection accepted beyond it is closed at once
    /// instead of queued.
    pub max_connections: usize,
    /// Effort and verbosity levels and defaults.
    pub dials: Dials,
}

/// Default request body limit: 32 MiB, room for long documents and base64 images.
pub const DEFAULT_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind: BindAddr::Loopback4,
            port: cerebro_config::schema::ApiSection::default().port,
            allowed_origins: Vec::new(),
            protections: GatewayProtections::default(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            timeouts: Timeouts::default(),
            max_connections: MAX_CONNECTIONS,
            dials: Dials::default(),
        }
    }
}

impl GatewayConfig {
    /// The gateway configuration settings describe: `api.bind`, `api.port`, `api.allowed_origins`,
    /// the `protections.gateway_*` switches, and the effort and verbosity dials. Limits and
    /// timeouts keep their defaults. Origins are normalized when the gateway binds, which refuses
    /// one that is not a single concrete origin.
    pub fn from_settings(settings: &cerebro_config::Settings) -> Self {
        Self {
            bind: settings.api.bind,
            port: settings.api.port,
            allowed_origins: settings.api.allowed_origins.clone(),
            protections: GatewayProtections::from_settings(&settings.protections),
            dials: Dials { effort: settings.effort.clone(), verbosity: settings.verbosity.clone() },
            ..Self::default()
        }
    }

    fn normalized(mut self) -> Result<Self, ServerError> {
        self.allowed_origins = self
            .allowed_origins
            .iter()
            .map(|o| normalize_origin(o))
            .collect::<Result<_, _>>()?;
        if self.max_body_bytes == 0 {
            return Err(ServerError::InvalidConfig("max_body_bytes must be above zero".to_string()));
        }
        if self.max_connections == 0 {
            return Err(ServerError::InvalidConfig("max_connections must be above zero".to_string()));
        }
        if self.timeouts.sse_keepalive.is_zero() {
            return Err(ServerError::InvalidConfig("timeouts.sse_keepalive must be above zero".to_string()));
        }
        if self.dials.effort.default_settings().is_none() || self.dials.verbosity.default_settings().is_none() {
            return Err(ServerError::InvalidConfig("an effort or verbosity default names no level".to_string()));
        }
        Ok(self)
    }
}

/// Everything a request handler can reach.
pub(crate) struct AppState {
    pub(crate) config: GatewayConfig,
    /// The token clients must present; replaced in place by [`GatewayHandle::set_token`]. Read
    /// only for the constant-time compare and never held across an await. A poisoned lock refuses
    /// every request.
    pub(crate) token: RwLock<BearerToken>,
    pub(crate) upstream: Upstream,
    pub(crate) telemetry: Telemetry,
    /// The port actually bound, which the Host check requires.
    pub(crate) port: u16,
    pub(crate) ipv6: bool,
    /// `/props` of the current upstream, keyed by its port.
    props: Mutex<Option<(u16, Value)>>,
    /// Facts about the current launch, set by [`crate::serve`].
    launch: RwLock<Value>,
}

impl AppState {
    /// llama-server's `/props`, fetched once per upstream port.
    async fn props(&self) -> Option<Value> {
        let port = self.upstream.port()?;
        if let Ok(cache) = self.props.lock() {
            if let Some((cached_port, props)) = cache.as_ref() {
                if *cached_port == port {
                    return Some(props.clone());
                }
            }
        }
        let (status, props) = self
            .upstream
            .json(Method::GET, "/props", None::<&Value>, self.config.timeouts.connect.max(Duration::from_secs(10)))
            .await
            .ok()?;
        if !status.is_success() {
            return None;
        }
        // Cache only when the request went to the port the key names: a restart onto another port
        // between the two reads must not file the new server's props under the old port.
        if self.upstream.port() == Some(port) {
            if let Ok(mut cache) = self.props.lock() {
                *cache = Some((port, props.clone()));
            }
        }
        Some(props)
    }

    /// The per-slot context size, from `/props`.
    async fn n_ctx(&self) -> Option<u64> {
        let props = self.props().await?;
        props
            .pointer("/default_generation_settings/n_ctx")
            .or_else(|| props.get("n_ctx"))
            .and_then(Value::as_u64)
            .filter(|&n| n > 0)
    }

    /// Whether the model's chat template has a thinking mode (`None` when `/props` is unavailable
    /// or carries no template, in which case the budget is sent as if it had).
    async fn thinking_supported(&self) -> Option<bool> {
        let props = self.props().await?;
        props.get("chat_template").and_then(Value::as_str).map(effort::template_supports_thinking)
    }

    /// What is known about the prompt (messages and tool definitions) and the context window, for
    /// clamping the thinking budget. Tokenizes only when the cheap byte bound does not already show
    /// that everything fits; media parts are counted at [`effort::MEDIA_PART_TOKENS_BOUND`] each
    /// and never sent to the tokenizer.
    async fn context_fit(&self, body: &Map<String, Value>, plan: &Plan, client_max: Option<u64>) -> Option<ContextFit> {
        if plan.effort.thinking_budget == ThinkingBudget::Off {
            return None;
        }
        let n_ctx = self.n_ctx().await?;
        let messages = body.get("messages").and_then(Value::as_array).map_or(&[][..], Vec::as_slice);
        let tools_bound = effort::tools_bytes_bound(body);
        let bound = effort::prompt_bytes_bound(messages).saturating_add(tools_bound);
        if let ThinkingBudget::Tokens(budget) = plan.effort.thinking_budget {
            let answer = client_max.unwrap_or(u64::from(plan.answer_reserve_tokens));
            if bound.saturating_add(u64::from(budget)).saturating_add(answer) <= n_ctx {
                return Some(ContextFit { n_ctx, prompt_tokens: bound });
            }
        }
        let mut content = effort::prompt_text(messages);
        let tools_text = effort::tools_text(body);
        if let Some(tools) = &tools_text {
            content.push_str(tools);
        }
        let request = json!({ "content": content });
        let tokenized = self
            .upstream
            .json(Method::POST, "/tokenize", Some(&request), self.config.timeouts.body_read)
            .await
            .ok()
            .filter(|(status, _)| status.is_success())
            .and_then(|(_, v)| v.get("tokens").and_then(Value::as_array).map(Vec::len));
        let prompt_tokens = match tokenized {
            Some(n) => (n as u64)
                .saturating_add(effort::TEMPLATE_OVERHEAD_PER_MESSAGE.saturating_mul(messages.len() as u64))
                .saturating_add(effort::MEDIA_PART_TOKENS_BOUND.saturating_mul(effort::media_parts(messages)))
                .saturating_add(if tools_text.is_some() { effort::TOOLS_TEMPLATE_OVERHEAD } else { 0 }),
            None => bound,
        };
        Some(ContextFit { n_ctx, prompt_tokens })
    }
}

/// The gateway's bound loopback listener, before it has an upstream. Binding is cheap and the step
/// most likely to fail (another instance on the port), so `serve` binds first and launches the
/// model server only once the port is held.
pub struct GatewayListener {
    listener: TcpListener,
    config: GatewayConfig,
    port: u16,
    ipv6: bool,
}

impl std::fmt::Debug for GatewayListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayListener").field("port", &self.port).finish_non_exhaustive()
    }
}

impl GatewayListener {
    /// Validate `config` and bind its loopback listener.
    ///
    /// # Errors
    /// [`ServerError::InvalidConfig`] for a bad origin, a zero body or connection limit, a zero
    /// keepalive or a dial default naming no level; [`ServerError::Bind`] when the port cannot be
    /// bound.
    pub async fn bind(config: GatewayConfig) -> Result<Self, ServerError> {
        let config = config.normalized()?;
        let (ip, ipv6) = match config.bind {
            BindAddr::Loopback4 => (IpAddr::V4(Ipv4Addr::LOCALHOST), false),
            BindAddr::Loopback6 => (IpAddr::V6(Ipv6Addr::LOCALHOST), true),
        };
        let addr = SocketAddr::new(ip, config.port);
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Bind { addr: addr.to_string(), cause: e.to_string() })?;
        let port = listener
            .local_addr()
            .map_err(|e| ServerError::Bind { addr: addr.to_string(), cause: e.to_string() })?
            .port();
        Ok(Self { listener, config, port, ipv6 })
    }

    /// The address actually bound.
    pub fn local_addr(&self) -> SocketAddr {
        let ip = if self.ipv6 { IpAddr::V6(Ipv6Addr::LOCALHOST) } else { IpAddr::V4(Ipv4Addr::LOCALHOST) };
        SocketAddr::new(ip, self.port)
    }

    /// Attach the token clients must present, the upstream and the telemetry; the result is ready
    /// to [`Gateway::spawn`].
    pub fn attach(self, token: BearerToken, upstream: Upstream, telemetry: Telemetry) -> Gateway {
        let state = Arc::new(AppState {
            config: self.config,
            token: RwLock::new(token),
            upstream,
            telemetry,
            port: self.port,
            ipv6: self.ipv6,
            props: Mutex::new(None),
            launch: RwLock::new(Value::Null),
        });
        Gateway { listener: self.listener, state }
    }
}

/// A bound, not yet serving gateway.
pub struct Gateway {
    listener: TcpListener,
    state: Arc<AppState>,
}

impl std::fmt::Debug for Gateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gateway").field("port", &self.state.port).finish_non_exhaustive()
    }
}

impl Gateway {
    /// Bind the gateway's loopback listener and attach its upstream in one step
    /// ([`GatewayListener::bind`] then [`GatewayListener::attach`]).
    ///
    /// # Errors
    /// As [`GatewayListener::bind`].
    pub async fn bind(
        config: GatewayConfig,
        token: BearerToken,
        upstream: Upstream,
        telemetry: Telemetry,
    ) -> Result<Self, ServerError> {
        Ok(GatewayListener::bind(config).await?.attach(token, upstream, telemetry))
    }

    /// The address actually bound.
    pub fn local_addr(&self) -> SocketAddr {
        let ip = if self.state.ipv6 { IpAddr::V6(Ipv6Addr::LOCALHOST) } else { IpAddr::V4(Ipv4Addr::LOCALHOST) };
        SocketAddr::new(ip, self.state.port)
    }

    /// Start serving on the current Tokio runtime.
    pub fn spawn(self) -> GatewayHandle {
        let addr = self.local_addr();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let router = router(Arc::clone(&self.state));
        let header_read = self.state.config.timeouts.header_read;
        let max_connections = self.state.config.max_connections;
        let task = tokio::spawn(serve_loop(self.listener, router, header_read, max_connections, shutdown_rx));
        GatewayHandle { addr, shutdown_tx, task, state: self.state }
    }
}

/// A serving gateway.
pub struct GatewayHandle {
    addr: SocketAddr,
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
    state: Arc<AppState>,
}

impl std::fmt::Debug for GatewayHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayHandle").field("addr", &self.addr).finish_non_exhaustive()
    }
}

impl GatewayHandle {
    /// The loopback address clients connect to.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// A handle that replaces the launch facts `/cerebro/status` reports under `launch`.
    pub fn launch_sink(&self) -> LaunchSink {
        LaunchSink(Arc::clone(&self.state))
    }

    /// Replace the token clients must present: from the next request on, the old token is refused
    /// and `token` accepted. Requests already past the guard finish.
    pub fn set_token(&self, token: BearerToken) {
        let mut slot = self.state.token.write().unwrap_or_else(PoisonError::into_inner);
        *slot = token;
        drop(slot);
        // The value was replaced whole, so nothing a panicking holder left half-written remains.
        self.state.token.clear_poison();
    }

    /// Stop accepting connections, let in-flight requests finish (5 s at most), and return.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        let _ = self.task.await;
    }
}

/// Publishes launch facts into a gateway's `/cerebro/status` from another task.
#[derive(Clone)]
pub struct LaunchSink(Arc<AppState>);

impl std::fmt::Debug for LaunchSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LaunchSink")
    }
}

impl LaunchSink {
    /// Replace the facts reported under `launch`.
    pub fn publish(&self, info: Value) {
        if let Ok(mut slot) = self.0.launch.write() {
            *slot = info;
        }
    }
}

/// The router with every route behind the guard.
pub(crate) fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/cerebro/health", get(health))
        .route("/cerebro/status", get(status_route))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(Arc::clone(&state), guard::guard))
        .with_state(state)
}

/// Accept loop: one HTTP/1.1 connection task per client, at most `max_connections` at once (a
/// socket accepted beyond that is closed at once, before anything is read from it), header-read
/// timeout enforced, graceful shutdown when `shutdown` flips.
async fn serve_loop(
    listener: TcpListener,
    router: Router,
    header_read: Duration,
    max_connections: usize,
    mut shutdown: watch::Receiver<bool>,
) {
    let permits = Arc::new(tokio::sync::Semaphore::new(max_connections));
    let mut connections = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, peer) = match accepted {
                    Ok(pair) => pair,
                    Err(_) => {
                        // Transient (out of handles, connection reset before accept): back off
                        // briefly instead of spinning.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                // Bound to loopback, so every peer is local; checked anyway, never assumed.
                if !peer.ip().is_loopback() {
                    continue;
                }
                // At capacity: refuse (the socket closes as it drops) rather than queue.
                let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
                    continue;
                };
                let _ = stream.set_nodelay(true);
                let service = TowerToHyperService::new(router.clone());
                let mut stop = shutdown.clone();
                connections.spawn(async move {
                    let _permit = permit;
                    let connection = hyper::server::conn::http1::Builder::new()
                        .timer(TokioTimer::new())
                        .header_read_timeout(header_read)
                        .serve_connection(TokioIo::new(stream), service);
                    tokio::pin!(connection);
                    tokio::select! {
                        _ = connection.as_mut() => {}
                        _ = stop.changed() => {
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                    }
                });
            }
            _ = shutdown.changed() => break,
            Some(_) = connections.join_next(), if !connections.is_empty() => {}
        }
    }
    drop(listener);
    let drained = tokio::time::timeout(Duration::from_secs(5), async {
        while connections.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        connections.abort_all();
    }
}

async fn not_found() -> Response {
    ApiError::new(StatusCode::NOT_FOUND, "not_found", "no such route").into_response()
}

/// Read a request body within the size limit and the body-read timeout.
async fn read_body(state: &AppState, body: Body) -> Result<Bytes, ApiError> {
    let limit = state.config.max_body_bytes;
    let read = Limited::new(body, limit).collect();
    match tokio::time::timeout(state.config.timeouts.body_read, read).await {
        Err(_) => Err(ApiError::new(StatusCode::REQUEST_TIMEOUT, "timeout", "the request body was not received in time")),
        Ok(Err(e)) if e.downcast_ref::<http_body_util::LengthLimitError>().is_some() => Err(guard::payload_too_large(limit)),
        Ok(Err(_)) => Err(ApiError::bad_request("the request body could not be read")),
        Ok(Ok(collected)) => Ok(collected.to_bytes()),
    }
}

/// Parse a JSON object body.
fn parse_object(bytes: &[u8]) -> Result<Map<String, Value>, ApiError> {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(ApiError::bad_request("the request body must be a JSON object")),
        // serde_json's message names a line and column, never the content.
        Err(e) => Err(ApiError::bad_request(format!("the request body is not valid JSON: {e}"))),
    }
}

/// A JSON response with `status`.
fn json_response(status: StatusCode, value: &Value) -> Response {
    let mut response = (status, json_body(value)).into_response();
    response.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

fn into_response(result: Result<Response, ApiError>) -> Response {
    result.unwrap_or_else(IntoResponse::into_response)
}

/// Builds a streamed chat reply's `cerebro` field from the stream's final speeds and whether any
/// reasoning came back.
type ReportBuilder = Box<dyn FnOnce(Option<Speed>, bool) -> Value + Send>;

/// Relay an upstream response, streaming its body frame by frame. With `capture`, llama-server's
/// final `timings` are read from the stream's tail and recorded. With `report` on an SSE stream,
/// one extra chunk carrying the `cerebro` field is sent just before `data: [DONE]`. Frames follow
/// socket reads, not SSE events, so the search carries across frames: whether the next frame
/// starts a line, plus at most 11 held-back bytes when a frame ends in a line-start prefix of
/// `data: [DONE]` (llama-server ends every event with a blank line, so in practice nothing is held
/// back and every frame is forwarded as it arrived).
fn relay(state: Arc<AppState>, upstream: hyper::Response<Incoming>, capture: bool, report: Option<ReportBuilder>) -> Response {
    let status = upstream.status();
    let content_type = upstream.headers().get(header::CONTENT_TYPE).cloned();
    let is_sse = content_type
        .as_ref()
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    let idle = state.config.timeouts.stream_idle;
    let relay = Relay {
        body: upstream.into_body(),
        idle,
        tail: Vec::new(),
        state: capture.then_some(state),
        report: report.filter(|_| is_sse && status.is_success()),
        scan: SseScan::new(),
    };
    let stream = futures_util::stream::unfold(Some(relay), |slot| async move {
        let mut relay = slot?;
        loop {
            match tokio::time::timeout(relay.idle, relay.body.frame()).await {
                Err(_) => return Some((Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "upstream stream idle")), None)),
                Ok(None) => {
                    relay.finish();
                    // Bytes held back as a possible `data: [DONE]` prefix are the stream's own:
                    // flushed, never lost.
                    let rest = relay.scan.take_carry();
                    return (!rest.is_empty()).then(|| (Ok(Bytes::from(rest)), None));
                }
                Ok(Some(Err(_))) => return Some((Err(std::io::Error::other("upstream stream failed")), None)),
                Ok(Some(Ok(frame))) => {
                    if let Ok(data) = frame.into_data() {
                        relay.keep_tail(&data);
                        let data = relay.inject_report(data);
                        if data.is_empty() {
                            continue;
                        }
                        return Some((Ok(data), Some(relay)));
                    }
                }
            }
        }
    });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    if let Some(ct) = content_type {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    if is_sse {
        response.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    response
}

/// State of one relayed stream.
struct Relay {
    body: Incoming,
    idle: Duration,
    tail: Vec<u8>,
    state: Option<Arc<AppState>>,
    /// Pending until the `[DONE]` frame arrives.
    report: Option<ReportBuilder>,
    /// Where `[DONE]` is and whether reasoning came back, across frames (fed only while a report
    /// is pending).
    scan: SseScan,
}

/// Scans a relayed SSE stream frame by frame for the line-start `data: [DONE]` and for a non-empty
/// `reasoning_content` delta, carrying what it needs across frame boundaries.
#[derive(Debug)]
struct SseScan {
    /// Whether a non-empty `reasoning_content` delta has been seen (looked for only until the
    /// first one).
    saw_reasoning: bool,
    /// The previous frame's last bytes, so a `reasoning_content` key split across frames is seen.
    reasoning_tail: ReasoningTail,
    /// Whether the next byte to scan (the first of `carry`, else of the next frame) starts a line.
    line_start: bool,
    /// Bytes held back because they start a line and are a proper prefix of `data: [DONE]`.
    carry: Vec<u8>,
}

impl SseScan {
    fn new() -> Self {
        Self { saw_reasoning: false, reasoning_tail: ReasoningTail::default(), line_start: true, carry: Vec::new() }
    }

    /// Feed one upstream frame. Returns the bytes to forward now (held-back bytes first, a trailing
    /// line-start prefix of `data: [DONE]` held back for the next frame) and, when the marker
    /// starts a line in them, its offset. With nothing held back (every real frame) `data` is
    /// returned as it came, without a copy.
    fn feed(&mut self, data: Bytes) -> (Bytes, Option<usize>) {
        if !self.saw_reasoning {
            let straddles = self.reasoning_tail.straddles_then_keep(&data);
            self.saw_reasoning = straddles || has_reasoning_delta(&data);
        }
        // Held-back bytes start a line by construction, so a frame they prefix starts one too.
        let line_start = self.line_start;
        let data = if self.carry.is_empty() {
            data
        } else {
            let mut joined = std::mem::take(&mut self.carry);
            joined.extend_from_slice(&data);
            Bytes::from(joined)
        };
        if let Some(at) = find_done_from(&data, line_start) {
            return (data, Some(at));
        }
        let hold = held_back(&data, line_start);
        let keep = data.len() - hold;
        if hold > 0 {
            self.carry.extend_from_slice(data.get(keep..).unwrap_or(&[]));
            self.line_start = true;
        } else if let Some(&last) = data.last() {
            self.line_start = last == b'\n';
        }
        (data.slice(..keep), None)
    }

    /// The held-back bytes, for the end of the stream.
    fn take_carry(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.carry)
    }
}

/// The last [`REASONING_KEY`]`.len()` bytes of the previous frame (on the stack: no allocation).
#[derive(Debug, Default)]
struct ReasoningTail {
    bytes: [u8; REASONING_KEY.len()],
    len: usize,
}

impl ReasoningTail {
    /// Whether a non-empty `reasoning_content` delta straddles the boundary between the previous
    /// frame and `data`; then remember `data`'s end for the next frame.
    fn straddles_then_keep(&mut self, data: &[u8]) -> bool {
        const N: usize = REASONING_KEY.len();
        let prev = self.bytes.get(..self.len).unwrap_or(&[]);
        let head = data.get(..data.len().min(N)).unwrap_or(&[]);
        let mut window = [0u8; 2 * N];
        let joined = prev.len() + head.len();
        let found = match window.get_mut(..joined) {
            Some(w) => {
                let (a, b) = w.split_at_mut(prev.len());
                a.copy_from_slice(prev);
                b.copy_from_slice(head);
                has_reasoning_delta(w)
            }
            None => false,
        };
        // Keep the last N bytes seen: `data`'s end, topped up from the old tail when it is short.
        let from_data = data.len().min(N);
        let from_prev = (N - from_data).min(prev.len());
        let mut next = [0u8; N];
        if let (Some(dst), Some(src)) = (next.get_mut(..from_prev), prev.get(prev.len() - from_prev..)) {
            dst.copy_from_slice(src);
        }
        if let (Some(dst), Some(src)) = (next.get_mut(from_prev..from_prev + from_data), data.get(data.len() - from_data..)) {
            dst.copy_from_slice(src);
        }
        self.bytes = next;
        self.len = from_prev + from_data;
        found
    }
}

impl Relay {
    fn keep_tail(&mut self, data: &[u8]) {
        if self.state.is_none() {
            return;
        }
        self.tail.extend_from_slice(data);
        if self.tail.len() > STREAM_TAIL_BYTES {
            let excess = self.tail.len() - STREAM_TAIL_BYTES;
            self.tail.drain(..excess);
        }
    }

    /// What to forward for upstream frame `data`. With a report pending: any held-back bytes
    /// first, the `cerebro` chunk inserted before a line-start `data: [DONE]`, and a trailing
    /// line-start prefix of the marker held back for the next frame. Otherwise `data` unchanged.
    fn inject_report(&mut self, data: Bytes) -> Bytes {
        if self.report.is_none() {
            return data;
        }
        let (data, done_at) = self.scan.feed(data);
        let Some(at) = done_at else {
            return data;
        };
        let Some(build) = self.report.take() else {
            return data;
        };
        let speed = Speed::from_stream_tail(&self.tail);
        let chunk = json!({ "object": "chat.completion.chunk", "choices": [], "cerebro": build(speed, self.scan.saw_reasoning) });
        let event = format!("data: {}\n\n", json_body(&chunk));
        let mut out = Vec::with_capacity(data.len() + event.len());
        out.extend_from_slice(&data[..at]);
        out.extend_from_slice(event.as_bytes());
        out.extend_from_slice(&data[at..]);
        Bytes::from(out)
    }

    fn finish(&mut self) {
        if let Some(state) = &self.state {
            if let Some(speed) = Speed::from_stream_tail(&self.tail) {
                state.telemetry.record(speed);
            }
        }
    }
}

/// The SSE event that ends an OpenAI-style stream.
const DONE: &[u8] = b"data: [DONE]";

/// Whether offset `at` of `data` starts a line; offset 0 does when `line_start` says `data` itself
/// begins one.
fn starts_line(data: &[u8], at: usize, line_start: bool) -> bool {
    match at.checked_sub(1) {
        None => line_start,
        Some(before) => data.get(before) == Some(&b'\n'),
    }
}

/// Where `data: [DONE]` starts a line in `data` (SSE events start lines, and JSON strings in an
/// event cannot hold a raw newline, so this never matches inside content). `line_start` says
/// whether `data` itself begins a line: a frame that starts mid-event does not.
fn find_done_from(data: &[u8], line_start: bool) -> Option<usize> {
    let mut from = 0;
    while let Some(offset) = data.get(from..)?.windows(DONE.len()).position(|w| w == DONE) {
        let at = from + offset;
        if starts_line(data, at, line_start) {
            return Some(at);
        }
        from = at + 1;
    }
    None
}

/// How many bytes at the end of `data` start a line and are a proper prefix of `data: [DONE]`
/// (0 to 11): the marker may continue in the next frame.
fn held_back(data: &[u8], line_start: bool) -> usize {
    (1..DONE.len().min(data.len() + 1))
        .rev()
        .find(|&n| DONE.get(..n).is_some_and(|prefix| data.ends_with(prefix)) && starts_line(data, data.len() - n, line_start))
        .unwrap_or(0)
}

/// The key a non-empty reasoning delta starts with, up to its opening quote.
const REASONING_KEY: &[u8] = br#""reasoning_content":""#;

/// Whether `data` carries a non-empty `reasoning_content` delta.
fn has_reasoning_delta(data: &[u8]) -> bool {
    let mut from = 0;
    while let Some(offset) = data.get(from..).and_then(|d| d.windows(REASONING_KEY.len()).position(|w| w == REASONING_KEY)) {
        let at = from + offset;
        match data.get(at + REASONING_KEY.len()) {
            Some(&b) if b != b'"' => return true,
            _ => from = at + 1,
        }
    }
    false
}

/// Forward a body unchanged (after `cerebro_*` fields are stripped) and relay the answer.
async fn passthrough(state: Arc<AppState>, path: &'static str, request: Request) -> Result<Response, ApiError> {
    let bytes = read_body(&state, request.into_body()).await?;
    let mut body = parse_object(&bytes)?;
    let stripped = body.remove(effort::EFFORT_FIELD).is_some() | body.remove(effort::VERBOSITY_FIELD).is_some();
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let forward = if stripped {
        Bytes::from(serde_json::to_vec(&Value::Object(body)).map_err(|_| ApiError::bad_request("unserializable body"))?)
    } else {
        bytes
    };
    let head_timeout = if stream { state.config.timeouts.stream_idle } else { state.config.timeouts.generation };
    let response = state.upstream.send(Method::POST, path, Some(forward), head_timeout).await?;
    Ok(relay(state, response, true, None))
}

async fn models(State(state): State<Arc<AppState>>) -> Response {
    let result = async {
        let response = state.upstream.send(Method::GET, "/v1/models", None, state.config.timeouts.connect.max(Duration::from_secs(10))).await?;
        Ok(relay(Arc::clone(&state), response, false, None))
    }
    .await;
    into_response(result)
}

async fn completions(State(state): State<Arc<AppState>>, request: Request) -> Response {
    into_response(passthrough(state, "/v1/completions", request).await)
}

async fn embeddings(State(state): State<Arc<AppState>>, request: Request) -> Response {
    into_response(passthrough(state, "/v1/embeddings", request).await)
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let upstream = state.upstream.state();
    let ready = upstream == cerebro_supervisor::LifecycleState::Ready;
    let body = json!({ "status": if ready { "ok" } else { "unavailable" }, "upstream": describe(&upstream) });
    json_response(if ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE }, &body)
}

async fn status_route(State(state): State<Arc<AppState>>) -> Response {
    let upstream_state = state.upstream.state();
    let props = if upstream_state == cerebro_supervisor::LifecycleState::Ready { state.props().await } else { None };
    let vram_state = Arc::clone(&state);
    let vram = tokio::task::spawn_blocking(move || vram_state.telemetry.vram_blocking())
        .await
        .unwrap_or_else(|_| Err("VRAM query failed".to_string()));
    let model = props.as_ref().map(|p| {
        json!({
            "alias": p.get("model_alias"),
            "path": p.get("model_path"),
            "n_ctx": p.pointer("/default_generation_settings/n_ctx").or_else(|| p.get("n_ctx")),
            "thinking_supported": p.get("chat_template").and_then(Value::as_str).map(effort::template_supports_thinking),
            "build": p.get("build_info"),
        })
    });
    let launch = state.launch.read().map(|l| l.clone()).unwrap_or(Value::Null);
    let disabled: Vec<Value> = state
        .config
        .protections
        .disabled()
        .into_iter()
        .map(|(setting, risk)| json!({ "setting": setting, "risk": risk }))
        .collect();
    let body = json!({
        "gateway": { "port": state.port, "disabled_protections": disabled },
        "upstream": { "state": describe(&upstream_state), "port": state.upstream.port() },
        "model": model,
        "speed": state.telemetry.speed_json(),
        "vram": status::vram_json(&vram),
        "launch": launch,
        "defaults": {
            "effort": state.config.dials.effort.default_level,
            "verbosity": state.config.dials.verbosity.default_level,
        },
    });
    json_response(StatusCode::OK, &body)
}

/// Sends pass requests to the upstream's chat completions route.
struct UpstreamCompleter {
    state: Arc<AppState>,
}

impl Completer for UpstreamCompleter {
    fn complete(&self, body: &Map<String, Value>) -> impl std::future::Future<Output = Result<Value, ApiError>> + Send {
        let state = Arc::clone(&self.state);
        // Serialized here, before the future, so the future does not borrow `body`.
        let bytes = serde_json::to_vec(body).map_err(|_| ApiError::bad_request("unserializable body"));
        async move {
            let bytes = Bytes::from(bytes?);
            let (status, value) = state
                .upstream
                .json_bytes(Method::POST, "/v1/chat/completions", Some(bytes), state.config.timeouts.generation)
                .await?;
            if !status.is_success() {
                let message = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("the model server refused the request")
                    .to_string();
                return Err(ApiError::new(status, "upstream_error", message));
            }
            if let Some(speed) = Speed::from_response(&value) {
                state.telemetry.record(speed);
            }
            Ok(value)
        }
    }
}

/// Whether a chat completion carries non-empty reasoning.
fn reasoning_returned(response: &Value) -> bool {
    response
        .pointer("/choices/0/message/reasoning_content")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
}

/// The `cerebro` field of a chat reply.
fn cerebro_report(
    plan: &Plan,
    budget: &Budget,
    thinking_supported: Option<bool>,
    passes: &[PassRecord],
    skipped: Option<&str>,
    total_ms: u64,
    reasoning_returned: bool,
) -> Value {
    let draft_ms = passes.first().map_or(0, |p| p.ms);
    // Multi-path self-consistency (the level's `consistency_paths`) is not implemented yet: one
    // path runs, and the reply says so rather than implying the configured number ran.
    let consistency = (plan.effort.consistency_paths > 1)
        .then(|| json!({ "configured": plan.effort.consistency_paths, "run": 1, "note": "multi-path self-consistency is not implemented yet" }));
    json!({
        "consistency_paths": consistency,
        "effort": plan.effort_id,
        "verbosity": plan.verbosity_id,
        "thinking_supported": thinking_supported,
        "thinking_budget_tokens": budget.thinking_tokens,
        "thinking_clamped_to_context": budget.clamped_to_context,
        "answer_budget_tokens": budget.answer_tokens,
        "max_tokens": budget.max_tokens,
        "reasoning_returned": reasoning_returned,
        "passes": passes.iter().map(PassRecord::to_json).collect::<Vec<_>>(),
        "review_skipped": skipped,
        "total_ms": total_ms,
        "extra_ms": total_ms.saturating_sub(draft_ms),
    })
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The error for an upstream chat reply that is not a JSON object (nothing can be added to it).
fn not_an_object() -> ApiError {
    ApiError::bad_upstream("the model server's reply is not a JSON object")
}

async fn chat(State(state): State<Arc<AppState>>, request: Request) -> Response {
    into_response(chat_inner(state, request).await)
}

/// What a chat request resolved to before it is sent: the levels, the budget and what is known
/// about the model.
struct Prepared {
    plan: Plan,
    budget: Budget,
    client_max: Option<u64>,
    context: Option<ContextFit>,
    thinking_supported: Option<bool>,
    started: Instant,
}

async fn chat_inner(state: Arc<AppState>, request: Request) -> Result<Response, ApiError> {
    let started = Instant::now();
    let bytes = read_body(&state, request.into_body()).await?;
    let mut body = parse_object(&bytes)?;
    drop(bytes);
    let mut plan = effort::resolve(&state.config.dials, &mut body)?;
    if plan.has_passes() && body.get("n").and_then(Value::as_u64).is_some_and(|n| n > 1) {
        return Err(ApiError::bad_request("n > 1 is not supported at effort levels with review passes"));
    }
    {
        let messages = body
            .get_mut("messages")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| ApiError::bad_request("messages must be an array"))?;
        effort::inject_system(messages, &effort::instruction_text(&plan))?;
    }
    let client_max = effort::client_max_tokens(&body)?;
    // A model without a thinking mode, or a request that turned thinking off for itself, gets the
    // instructions only: llama.cpp would not think, and reserving the budget on top of the
    // client's max_tokens would only lift the answer's cap.
    let thinking_supported = state.thinking_supported().await;
    if thinking_supported == Some(false) || effort::client_disabled_thinking(&body) {
        plan.effort.thinking_budget = ThinkingBudget::Off;
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    // Borrowed, not copied: the body is not changed until the budget is applied.
    let context = state.context_fit(&body, &plan, client_max).await;
    let budget = effort::plan_budget(&plan, client_max, context);
    effort::apply_budget(&mut body, &budget);
    let prepared = Prepared { plan, budget, client_max, context, thinking_supported, started };

    if prepared.plan.has_passes() {
        if stream {
            return Ok(stream_with_passes(state, body, prepared));
        }
        let completer = UpstreamCompleter { state: Arc::clone(&state) };
        let Prepared { plan, budget, client_max, context, thinking_supported, started } = prepared;
        let outcome = passes::run(&completer, body, &plan, client_max, context, |_| {}).await?;
        let mut response = outcome.response;
        let reasoning = reasoning_returned(&response);
        let report = cerebro_report(&plan, &budget, thinking_supported, &outcome.passes, outcome.skipped, elapsed_ms(started), reasoning);
        response.as_object_mut().ok_or_else(not_an_object)?.insert("cerebro".to_string(), report);
        return Ok(json_response(StatusCode::OK, &response));
    }

    let forward = Bytes::from(serde_json::to_vec(&body).map_err(|_| ApiError::bad_request("unserializable body"))?);
    drop(body);
    let call_started = Instant::now();
    let Prepared { plan, budget, thinking_supported, started, .. } = prepared;
    if stream {
        let response = state.upstream.send(Method::POST, "/v1/chat/completions", Some(forward), state.config.timeouts.stream_idle).await?;
        let report: ReportBuilder = Box::new(move |speed: Option<Speed>, reasoning: bool| {
            let draft = PassRecord {
                kind: PassKind::Draft,
                ms: elapsed_ms(call_started),
                tokens_per_second: speed.and_then(|s| s.decode),
                found_errors: None,
            };
            cerebro_report(&plan, &budget, thinking_supported, &[draft], None, elapsed_ms(started), reasoning)
        });
        return Ok(relay(state, response, true, Some(report)));
    }
    let (status, mut response) = state
        .upstream
        .json_bytes(Method::POST, "/v1/chat/completions", Some(forward), state.config.timeouts.generation)
        .await?;
    if status.is_success() {
        let speed = Speed::from_response(&response);
        if let Some(speed) = speed {
            state.telemetry.record(speed);
        }
        let draft = PassRecord {
            kind: PassKind::Draft,
            ms: elapsed_ms(call_started),
            tokens_per_second: speed.and_then(|s| s.decode),
            found_errors: None,
        };
        let reasoning = reasoning_returned(&response);
        let report = cerebro_report(&plan, &budget, thinking_supported, &[draft], None, elapsed_ms(started), reasoning);
        if let Some(map) = response.as_object_mut() {
            map.insert("cerebro".to_string(), report);
        }
    }
    Ok(json_response(status, &response))
}

/// Run the passes, then send the final answer as an OpenAI-style SSE stream: SSE comments while
/// passes run (one per pass start, and one every [`Timeouts::sse_keepalive`]), then a content
/// chunk, a final chunk carrying `finish_reason`, `usage` and the `cerebro` field, and
/// `data: [DONE]`. A client that disconnects cancels the remaining passes.
fn stream_with_passes(state: Arc<AppState>, body: Map<String, Value>, prepared: Prepared) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    let keepalive_every = state.config.timeouts.sse_keepalive;
    tokio::spawn(async move {
        let Prepared { plan, budget, client_max, context, thinking_supported, started } = prepared;
        let completer = UpstreamCompleter { state: Arc::clone(&state) };
        let progress = tx.clone();
        let run = passes::run(&completer, body, &plan, client_max, context, move |kind| {
            let _ = progress.try_send(Ok(Bytes::from(format!(": cerebro pass {}\n\n", kind.as_str()))));
        });
        tokio::pin!(run);
        let mut keepalive = tokio::time::interval(keepalive_every);
        keepalive.tick().await;
        let outcome = loop {
            tokio::select! {
                outcome = &mut run => break outcome,
                _ = keepalive.tick() => {
                    let _ = tx.try_send(Ok(Bytes::from_static(b": cerebro keepalive\n\n")));
                }
                _ = tx.closed() => return,
            }
        };
        let events = match outcome {
            Ok(outcome) if outcome.response.is_object() => {
                let reasoning = reasoning_returned(&outcome.response);
                let report = cerebro_report(&plan, &budget, thinking_supported, &outcome.passes, outcome.skipped, elapsed_ms(started), reasoning);
                final_events(&outcome.response, report)
            }
            Ok(_) => vec![format!("data: {}\n\n", json_body(&not_an_object().body()))],
            Err(e) => vec![format!("data: {}\n\n", json_body(&e.body()))],
        };
        for event in events {
            if tx.send(Ok(Bytes::from(event))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n"))).await;
    });
    let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|item| (item, rx)) });
    let mut response = Response::new(Body::from_stream(stream));
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

/// The SSE `data:` events carrying a finished chat completion as chunks. Nothing from `response`
/// is indexed for writing: a malformed upstream value is carried as it is, never a panic.
fn final_events(response: &Value, report: Value) -> Vec<String> {
    let id = response.get("id").cloned().unwrap_or_else(|| json!("chatcmpl-cerebro"));
    let created = response.get("created").cloned().unwrap_or_else(|| json!(0));
    let model = response.get("model").cloned().unwrap_or(Value::Null);
    let message = response.pointer("/choices/0/message");
    let finish = response.pointer("/choices/0/finish_reason").cloned().unwrap_or_else(|| json!("stop"));

    let mut delta = Map::new();
    delta.insert("role".to_string(), json!("assistant"));
    for field in ["reasoning_content", "content"] {
        if let Some(v) = message.and_then(|m| m.get(field)).filter(|v| !v.is_null()) {
            delta.insert(field.to_string(), v.clone());
        }
    }
    if let Some(calls) = message.and_then(|m| m.get("tool_calls")).and_then(Value::as_array) {
        let indexed: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut c = c.clone();
                if let Some(call) = c.as_object_mut() {
                    call.insert("index".to_string(), json!(i));
                }
                c
            })
            .collect();
        delta.insert("tool_calls".to_string(), Value::Array(indexed));
    }
    let chunk = |choice: Value| json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [choice] });
    let content = chunk(json!({ "index": 0, "delta": Value::Object(delta), "finish_reason": null }));
    let mut last = Map::new();
    last.insert("id".to_string(), id.clone());
    last.insert("object".to_string(), json!("chat.completion.chunk"));
    last.insert("created".to_string(), created.clone());
    last.insert("model".to_string(), model.clone());
    last.insert("choices".to_string(), json!([{ "index": 0, "delta": {}, "finish_reason": finish }]));
    if let Some(usage) = response.get("usage") {
        last.insert("usage".to_string(), usage.clone());
    }
    last.insert("cerebro".to_string(), report);
    vec![format!("data: {}\n\n", json_body(&content)), format!("data: {}\n\n", json_body(&Value::Object(last)))]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_events_carry_answer_then_finish_with_cerebro() {
        let response = json!({"id": "x", "created": 1, "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "four"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3}});
        let events = final_events(&response, json!({"effort": "max"}));
        assert_eq!(events.len(), 2);
        let first: Value = serde_json::from_str(events[0].trim().trim_start_matches("data: ")).expect("json");
        assert_eq!(first["choices"][0]["delta"]["content"], "four");
        assert_eq!(first["object"], "chat.completion.chunk");
        let last: Value = serde_json::from_str(events[1].trim().trim_start_matches("data: ")).expect("json");
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["cerebro"]["effort"], "max");
        assert_eq!(last["usage"]["total_tokens"], 3);
    }

    #[test]
    fn final_events_carry_malformed_tool_calls_without_panicking() {
        let response = json!({"choices": [{"message": {"role": "assistant", "content": null,
            "tool_calls": ["not an object", {"id": "1", "type": "function"}]}}]});
        let events = final_events(&response, json!({}));
        let first: Value = serde_json::from_str(events[0].trim().trim_start_matches("data: ")).expect("json");
        assert_eq!(first["choices"][0]["delta"]["tool_calls"][0], "not an object");
        assert_eq!(first["choices"][0]["delta"]["tool_calls"][1]["index"], 1);
        // A reply that is not an object at all still yields well-formed chunks.
        assert_eq!(final_events(&json!([]), json!({})).len(), 2);
    }

    #[test]
    fn done_is_found_only_at_a_line_start() {
        assert_eq!(find_done_from(b"data: [DONE]\n\n", true), Some(0));
        assert_eq!(find_done_from(b"data: [DONE]\n\n", false), None, "a frame starting mid-line");
        assert_eq!(find_done_from(b"data: {\"x\":1}\n\ndata: [DONE]\n\n", true), Some(15));
        assert_eq!(find_done_from(br#"data: {"content":"data: [DONE]"}"#, true), None);
        let mut frame = br#"data: {"content":"data: [DONE]"}"#.to_vec();
        frame.extend_from_slice(b"\n\ndata: [DONE]\n\n");
        assert_eq!(find_done_from(&frame, true), Some(frame.len() - b"data: [DONE]\n\n".len()));
        assert_eq!(find_done_from(b"data: [DON", true), None);
        assert!(has_reasoning_delta(br#"{"delta":{"reasoning_content":"so"}}"#));
        assert!(!has_reasoning_delta(br#"{"delta":{"reasoning_content":""}}"#));
        assert!(!has_reasoning_delta(br#"{"delta":{"content":"so"}}"#));
        // An empty delta first does not hide a non-empty one later in the same frame.
        assert!(has_reasoning_delta(br#"{"reasoning_content":""}{"reasoning_content":"x"}"#));
    }

    #[test]
    fn held_back_is_a_line_start_proper_prefix_of_done_only() {
        assert_eq!(held_back(b"data: x\n\ndata: [DO", true), 9);
        assert_eq!(held_back(b"data: x\n\ndata: [DONE", true), 11);
        assert_eq!(held_back(b"d", true), 1);
        assert_eq!(held_back(b"d", false), 0, "mid-line");
        assert_eq!(held_back(b"x\nd", false), 1);
        assert_eq!(held_back(br#"{"content":"data: [DO"#, true), 0, "inside a JSON string, not at a line start");
        assert_eq!(held_back(b"data: x\n\n", true), 0);
        assert_eq!(held_back(b"", true), 0);
        // The whole marker is found, never held.
        assert_eq!(held_back(b"data: [DONE]", true), 0);
    }

    /// Feed `frames` through a scan, splicing `<R>` where the report goes, and return what a
    /// client receives plus whether reasoning was seen.
    fn relay_frames(frames: &[&[u8]]) -> (String, bool) {
        let mut scan = SseScan::new();
        let mut out = Vec::new();
        let mut reported = false;
        for frame in frames {
            if reported {
                out.extend_from_slice(frame);
                continue;
            }
            let (data, at) = scan.feed(Bytes::copy_from_slice(frame));
            match at {
                Some(at) => {
                    out.extend_from_slice(&data[..at]);
                    out.extend_from_slice(b"<R>");
                    out.extend_from_slice(&data[at..]);
                    reported = true;
                }
                None => out.extend_from_slice(&data),
            }
        }
        out.extend_from_slice(&scan.take_carry());
        (String::from_utf8(out).expect("utf8"), scan.saw_reasoning)
    }

    #[test]
    fn a_done_marker_split_across_frames_still_gets_the_report_before_it() {
        let (out, _) = relay_frames(&[b"data: {\"a\":1}\n\ndata: [DO", b"NE]\n\n"]);
        assert_eq!(out, "data: {\"a\":1}\n\n<R>data: [DONE]\n\n");
        // Split one byte at a time.
        let whole = b"data: {\"a\":1}\n\ndata: [DONE]\n\n";
        let bytes: Vec<&[u8]> = whole.chunks(1).collect();
        let (out, _) = relay_frames(&bytes);
        assert_eq!(out, "data: {\"a\":1}\n\n<R>data: [DONE]\n\n");
    }

    #[test]
    fn a_frame_starting_inside_a_json_string_is_never_injected_into() {
        // The event's JSON string holds the text `data: [DONE]`, and a frame boundary falls just
        // before it: the frame starts mid-line, so nothing is spliced there.
        let (out, _) = relay_frames(&[br#"data: {"content":""#, br#"data: [DONE]"}"#, b"\n\ndata: [DONE]\n\n"]);
        assert_eq!(out, "data: {\"content\":\"data: [DONE]\"}\n\n<R>data: [DONE]\n\n");
        // And a held-back look-alike inside a string is not held: a frame ending in `"data: [DO`.
        let (out, _) = relay_frames(&[br#"data: {"content":"data: [DO"#, br#"NE]"}"#, b"\n\ndata: [DONE]\n\n"]);
        assert_eq!(out, "data: {\"content\":\"data: [DONE]\"}\n\n<R>data: [DONE]\n\n");
    }

    #[test]
    fn held_back_bytes_are_flushed_when_a_stream_ends_without_done() {
        let (out, _) = relay_frames(&[b"data: {\"a\":1}\n\n", b"data: [DON"]);
        assert_eq!(out, "data: {\"a\":1}\n\ndata: [DON");
    }

    #[test]
    fn a_reasoning_delta_split_across_frames_is_seen() {
        let event = br#"data: {"choices":[{"delta":{"reasoning_content":"hm"}}]}"#;
        for cut in 1..event.len() {
            let (head, tail) = event.split_at(cut);
            let (_, saw) = relay_frames(&[head, tail, b"\n\ndata: [DONE]\n\n"]);
            assert!(saw, "missed when cut at {cut}");
        }
        let empty = br#"data: {"choices":[{"delta":{"reasoning_content":""}}]}"#;
        for cut in 1..empty.len() {
            let (head, tail) = empty.split_at(cut);
            let (_, saw) = relay_frames(&[head, tail, b"\n\ndata: [DONE]\n\n"]);
            assert!(!saw, "an empty delta counted when cut at {cut}");
        }
        // Split over many tiny frames.
        let bytes: Vec<&[u8]> = event.chunks(3).collect();
        assert!(relay_frames(&bytes).1);
    }

    #[test]
    fn config_normalizes_origins_and_refuses_bad_ones() {
        let config = GatewayConfig { allowed_origins: vec!["HTTP://Tauri.Localhost/".into()], ..GatewayConfig::default() };
        assert_eq!(config.normalized().expect("ok").allowed_origins, vec!["http://tauri.localhost"]);
        let bad = GatewayConfig { allowed_origins: vec!["*".into()], ..GatewayConfig::default() };
        assert!(bad.normalized().is_err());
        let zero = GatewayConfig { max_body_bytes: 0, ..GatewayConfig::default() };
        assert!(zero.normalized().is_err());
        let zero = GatewayConfig { max_connections: 0, ..GatewayConfig::default() };
        assert!(zero.normalized().is_err());
        let zero = GatewayConfig { timeouts: Timeouts { sse_keepalive: Duration::ZERO, ..Timeouts::default() }, ..GatewayConfig::default() };
        assert!(zero.normalized().is_err());
    }

    #[test]
    fn from_settings_takes_api_and_dials() {
        let mut settings = cerebro_config::Settings::default();
        settings.api.port = 9999;
        settings.effort.default_level = "high".into();
        let config = GatewayConfig::from_settings(&settings);
        assert_eq!(config.port, 9999);
        assert_eq!(config.dials.effort.default_level, "high");
        assert_eq!(config.dials.verbosity.default_level, "brief");
        assert!(config.allowed_origins.is_empty());
        assert_eq!(config.protections, GatewayProtections::default());
        assert!(config.protections.disabled().is_empty());
    }

    #[test]
    fn from_settings_maps_allowed_origins_normalized_and_refuses_wildcards() {
        let mut settings = cerebro_config::Settings::default();
        settings.api.allowed_origins = vec!["HTTP://Tauri.Localhost/".into()];
        let config = GatewayConfig::from_settings(&settings).normalized().expect("valid");
        assert_eq!(config.allowed_origins, vec!["http://tauri.localhost"]);
        for bad in ["*", "null"] {
            settings.api.allowed_origins = vec![bad.into()];
            let err = GatewayConfig::from_settings(&settings).normalized().unwrap_err();
            assert!(matches!(err, ServerError::InvalidConfig(_)), "{bad}: {err}");
            assert!(settings.validate().is_err(), "settings refuse {bad} too");
        }
    }

    #[test]
    fn from_settings_maps_each_gateway_switch_to_its_own_check() {
        type Setter = fn(&mut cerebro_config::ProtectionsSection);
        let cases: [(Setter, GatewayProtections, &str); 3] = [
            (|p| p.gateway_require_token = false, GatewayProtections { require_token: false, ..GatewayProtections::default() }, "gateway_require_token"),
            (|p| p.gateway_check_host = false, GatewayProtections { check_host: false, ..GatewayProtections::default() }, "gateway_check_host"),
            (|p| p.gateway_check_origin = false, GatewayProtections { check_origin: false, ..GatewayProtections::default() }, "gateway_check_origin"),
        ];
        for (set, expected, key) in cases {
            let mut settings = cerebro_config::Settings::default();
            set(&mut settings.protections);
            let config = GatewayConfig::from_settings(&settings);
            assert_eq!(config.protections, expected, "{key}");
            let disabled = config.protections.disabled();
            assert_eq!(disabled.len(), 1, "{key}");
            assert_eq!(disabled[0].0, key);
            // The settings screen's note and status carry the same key and wording.
            let note = settings.disabled_protections();
            assert_eq!(note.len(), 1, "{key}");
            assert_eq!((note[0].key, note[0].risk.as_str()), disabled[0], "{key}");
        }
    }
}
