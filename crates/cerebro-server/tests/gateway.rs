//! The gateway against a fake llama-server on loopback: authentication, Host and Origin checks,
//! body limits, SSE relay, effort and verbosity injection, the review passes, and timeouts.
//!
//! The client side is a raw HTTP/1.1 socket so every header (Host included) is exactly what the
//! test wrote.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use cerebro_server::passes::{CHECK_INSTRUCTION, CHECK_SYSTEM, REVIEW_INSTRUCTION};
use cerebro_server::{
    BearerToken, Gateway, GatewayConfig, GatewayHandle, GatewayProtections, GpuMemory, TargetSource, Telemetry, Timeouts, Upstream,
    UpstreamTarget,
};
use cerebro_supervisor::LifecycleState;
use http_body_util::BodyExt as _;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Notify};

// ---------------------------------------------------------------------------------------------
// Fake llama-server
// ---------------------------------------------------------------------------------------------

/// The key the gateway must present upstream (never the client's token).
const UPSTREAM_RAW: [u8; 32] = [7; 32];

#[derive(Default)]
struct FakeState {
    /// (path, Authorization header, body) of every request.
    seen: Mutex<Vec<(String, Option<String>, Value)>>,
    /// Scripted non-streaming chat replies (answer text), popped in order.
    replies: Mutex<VecDeque<String>>,
    /// Released by the test to let a stream finish.
    gate: Notify,
    /// Context size reported by /props.
    n_ctx: Mutex<u64>,
    /// Tokens /tokenize reports.
    token_count: Mutex<usize>,
    /// Chat template /props reports, when set.
    template: Mutex<Option<String>>,
    /// When set, chat completions take this long to answer.
    delay: Mutex<Option<Duration>>,
    /// When set, non-streaming chat completions answer with exactly this body.
    raw: Mutex<Option<String>>,
}

async fn fake_record(state: &FakeState, path: &str, request: Request) -> Value {
    let auth = request.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).map(str::to_string);
    let bytes = request.into_body().collect().await.map(|c| c.to_bytes()).unwrap_or_default();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    state.seen.lock().unwrap().push((path.to_string(), auth, body.clone()));
    body
}

fn json_reply(v: Value) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], v.to_string()).into_response()
}

async fn fake_chat(State(state): State<Arc<FakeState>>, request: Request) -> Response {
    let body = fake_record(&state, "/v1/chat/completions", request).await;
    let delay = *state.delay.lock().unwrap();
    if let Some(d) = delay {
        tokio::time::sleep(d).await;
    }
    if body["stream"] == true {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(8);
        let st = Arc::clone(&state);
        tokio::spawn(async move {
            let chunk = |text: &str| format!("data: {}\n\n", json!({"choices": [{"index": 0, "delta": {"content": text}}]}));
            let _ = tx.send(Ok(Bytes::from(chunk("Hel")))).await;
            st.gate.notified().await;
            let _ = tx.send(Ok(Bytes::from(chunk("lo")))).await;
            let last = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                              "timings": {"prompt_per_second": 812.5, "predicted_per_second": 63.25}});
            let _ = tx.send(Ok(Bytes::from(format!("data: {last}\n\ndata: [DONE]\n\n")))).await;
        });
        let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|i| (i, rx)) });
        return ([(header::CONTENT_TYPE, "text/event-stream")], Body::from_stream(stream)).into_response();
    }
    if let Some(raw) = state.raw.lock().unwrap().clone() {
        return ([(header::CONTENT_TYPE, "application/json")], raw).into_response();
    }
    let text = state.replies.lock().unwrap().pop_front().unwrap_or_else(|| "ok".to_string());
    json_reply(json!({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "tiny",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        "timings": {"prompt_per_second": 500.0, "predicted_per_second": 40.0}
    }))
}

async fn fake_passthrough(State(state): State<Arc<FakeState>>, request: Request) -> Response {
    let path = request.uri().path().to_string();
    let body = fake_record(&state, &path, request).await;
    json_reply(json!({"object": "list", "echo_path": path, "received": body}))
}

async fn fake_models(State(state): State<Arc<FakeState>>, request: Request) -> Response {
    fake_record(&state, "/v1/models", request).await;
    json_reply(json!({"object": "list", "data": [{"id": "tiny", "object": "model"}]}))
}

async fn fake_props(State(state): State<Arc<FakeState>>, request: Request) -> Response {
    fake_record(&state, "/props", request).await;
    let n_ctx = *state.n_ctx.lock().unwrap();
    let mut props = json!({"default_generation_settings": {"n_ctx": n_ctx}, "model_alias": "tiny", "model_path": "C:/m/tiny.gguf"});
    if let Some(t) = state.template.lock().unwrap().clone() {
        props["chat_template"] = json!(t);
    }
    json_reply(props)
}

async fn fake_tokenize(State(state): State<Arc<FakeState>>, request: Request) -> Response {
    fake_record(&state, "/tokenize", request).await;
    let n = *state.token_count.lock().unwrap();
    json_reply(json!({"tokens": vec![1; n]}))
}

async fn start_fake() -> (SocketAddr, Arc<FakeState>) {
    let state = Arc::new(FakeState::default());
    *state.n_ctx.lock().unwrap() = 1 << 20;
    let app = Router::new()
        .route("/v1/chat/completions", post(fake_chat))
        .route("/v1/completions", post(fake_passthrough))
        .route("/v1/embeddings", post(fake_passthrough))
        .route("/v1/models", get(fake_models))
        .route("/props", get(fake_props))
        .route("/tokenize", post(fake_tokenize))
        .with_state(Arc::clone(&state));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else { continue };
            let service = hyper_util::service::TowerToHyperService::new(app.clone());
            tokio::spawn(async move {
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(hyper_util::rt::TokioIo::new(stream), service)
                    .await;
            });
        }
    });
    (addr, state)
}

// ---------------------------------------------------------------------------------------------
// Gateway under test
// ---------------------------------------------------------------------------------------------

struct Harness {
    gateway: Option<GatewayHandle>,
    addr: SocketAddr,
    token: String,
    fake: Arc<FakeState>,
    upstream_state: watch::Sender<LifecycleState>,
}

impl Harness {
    async fn start(config: GatewayConfig) -> Self {
        let (fake_addr, fake) = start_fake().await;
        let token = BearerToken::generate().unwrap();
        let token_hex = token.reveal().to_string();
        let (state_tx, state_rx) = watch::channel(LifecycleState::Ready);
        let port = fake_addr.port();
        let target: TargetSource = Arc::new(move || Some(UpstreamTarget { pid: 1, port }));
        let upstream = Upstream::new(target, &BearerToken::from_raw(&UPSTREAM_RAW).unwrap(), state_rx, Duration::from_secs(2)).unwrap();
        let telemetry = Telemetry::new(Box::new(|| Ok(vec![GpuMemory { name: "Fake GPU".into(), total_mib: 8192, free_mib: 6000 }])));
        let gateway = Gateway::bind(GatewayConfig { port: 0, ..config }, token, upstream, telemetry).await.unwrap().spawn();
        let addr = gateway.addr();
        Self { gateway: Some(gateway), addr, token: token_hex, fake, upstream_state: state_tx }
    }

    fn host(&self) -> String {
        format!("127.0.0.1:{}", self.addr.port())
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }

    async fn get(&self, path: &str) -> Reply {
        send(self.addr, "GET", path, &[("Host", &self.host()), ("Authorization", &self.bearer())], b"").await
    }

    async fn post(&self, path: &str, body: &Value) -> Reply {
        let body = body.to_string();
        send(self.addr, "POST", path, &[("Host", &self.host()), ("Authorization", &self.bearer()), ("Content-Type", "application/json")], body.as_bytes()).await
    }

    fn seen(&self) -> Vec<(String, Option<String>, Value)> {
        self.fake.seen.lock().unwrap().clone()
    }

    fn chat_bodies(&self) -> Vec<Value> {
        self.seen().into_iter().filter(|(p, _, _)| p == "/v1/chat/completions").map(|(_, _, b)| b).collect()
    }

    async fn stop(mut self) {
        if let Some(g) = self.gateway.take() {
            g.shutdown().await;
        }
    }
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|_| panic!("not JSON: {}", String::from_utf8_lossy(&self.body)))
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// One request over a fresh connection (`Connection: close`), response read to the end.
async fn send(addr: SocketAddr, method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Reply {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let mut head = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
    for (n, v) in headers {
        head.push_str(&format!("{n}: {v}\r\n"));
    }
    if !body.is_empty() || method == "POST" {
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(body).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), stream.read_to_end(&mut raw)).await.expect("response within 30 s").ok();
    parse_reply(&raw)
}

fn parse_reply(raw: &[u8]) -> Reply {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("header end");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(n, v)| (n.trim().to_string(), v.trim().to_string()))
        .collect();
    let mut body = raw[split + 4..].to_vec();
    if headers.iter().any(|(n, v)| n.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked")) {
        body = dechunk(&body);
    }
    Reply { status, headers, body }
}

fn dechunk(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(line_end) = data.windows(2).position(|w| w == b"\r\n") {
        let size = usize::from_str_radix(String::from_utf8_lossy(&data[..line_end]).trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let start = line_end + 2;
        out.extend_from_slice(&data[start..start + size]);
        data = &data[start + size + 2..];
    }
    out
}

fn config() -> GatewayConfig {
    GatewayConfig::default()
}

fn chat_body(extra: Value) -> Value {
    let mut body = json!({"model": "tiny", "messages": [{"role": "user", "content": "What is 2+2?"}]});
    if let (Some(b), Some(e)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            b.insert(k.clone(), v.clone());
        }
    }
    body
}

// ---------------------------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn every_route_requires_the_bearer_token_including_reads() {
    let h = Harness::start(config()).await;
    let host = h.host();
    let routes = [
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("GET", "/cerebro/health"),
        ("GET", "/cerebro/status"),
        ("GET", "/no/such/route"),
    ];
    let wrong = format!("Bearer {}", "0".repeat(64));
    let basic = format!("Basic {}", h.token);
    for (method, path) in routes {
        let none = send(h.addr, method, path, &[("Host", &host)], b"{}").await;
        assert_eq!(none.status, 401, "{method} {path} without a token");
        assert_eq!(none.header("www-authenticate"), Some("Bearer"));
        for bad in [wrong.as_str(), basic.as_str(), h.token.as_str()] {
            let r = send(h.addr, method, path, &[("Host", &host), ("Authorization", bad)], b"{}").await;
            assert_eq!(r.status, 401, "{method} {path} with {bad:?}");
        }
    }
    assert!(h.seen().is_empty(), "nothing may reach the model server without a token");

    assert_eq!(h.get("/v1/models").await.status, 200);
    assert_eq!(h.get("/cerebro/health").await.status, 200);
    assert_eq!(h.get("/no/such/route").await.status, 404);
    h.stop().await;
}

#[tokio::test]
async fn upstream_sees_its_own_key_and_never_the_client_token() {
    let h = Harness::start(config()).await;
    assert_eq!(h.get("/v1/models").await.status, 200);
    assert_eq!(h.post("/v1/chat/completions", &chat_body(json!({}))).await.status, 200);
    assert_eq!(h.post("/v1/embeddings", &json!({"input": "x"})).await.status, 200);
    let expected = format!("Bearer {}", BearerToken::from_raw(&UPSTREAM_RAW).unwrap().reveal());
    let seen = h.seen();
    assert!(!seen.is_empty());
    for (path, auth, body) in &seen {
        assert_eq!(auth.as_deref(), Some(expected.as_str()), "{path}");
        assert!(!body.to_string().contains(&h.token), "{path} body carried the client token");
    }
    h.stop().await;
}

/// Rotating the token on a serving gateway takes effect from the next request: the old token is
/// refused at once, the new one accepted, and the store holds the new one.
#[cfg(windows)]
#[tokio::test]
async fn rotating_the_token_while_serving_refuses_the_old_one_at_once() {
    let h = Harness::start(config()).await;
    let dir = tempfile::tempdir().unwrap();
    let store = cerebro_config::SecretStore::open(dir.path()).unwrap();
    let old = h.bearer();
    let host = h.host();
    assert_eq!(send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &old)], b"").await.status, 200);

    let fresh = BearerToken::rotate(&store).unwrap();
    let fresh_bearer = format!("Bearer {}", fresh.reveal());
    let stored = BearerToken::load_or_create(&store).unwrap();
    assert_eq!(stored.reveal(), fresh.reveal(), "the store holds the new token");
    h.gateway.as_ref().unwrap().set_token(fresh);

    let refused = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &old)], b"").await;
    assert_eq!(refused.status, 401, "the old token is refused from the next request on");
    let accepted = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &fresh_bearer)], b"").await;
    assert_eq!(accepted.status, 200, "{}", accepted.text());
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Host and Origin
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn host_header_must_name_the_gateway_on_loopback() {
    let h = Harness::start(config()).await;
    let bearer = h.bearer();
    let port = h.addr.port();
    for host in [
        "evil.example".to_string(),
        format!("evil.example:{port}"),
        format!("127.0.0.1:{}", port.wrapping_add(1)),
        "127.0.0.1".to_string(),
        format!("[::1]:{port}"),
    ] {
        let r = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &bearer)], b"").await;
        assert_eq!(r.status, 403, "Host {host:?}");
        assert_eq!(r.json()["error"]["type"], "host_not_allowed");
    }
    let missing = send(h.addr, "GET", "/v1/models", &[("Authorization", &bearer)], b"").await;
    assert_eq!(missing.status, 403);
    let ok = send(h.addr, "GET", "/v1/models", &[("Host", &format!("localhost:{port}")), ("Authorization", &bearer)], b"").await;
    assert_eq!(ok.status, 200);
    assert!(h.seen().iter().filter(|(p, _, _)| p == "/v1/models").count() == 1, "refused requests never reach upstream");
    h.stop().await;
}

#[tokio::test]
async fn origins_are_refused_by_default_and_cors_stays_closed() {
    let h = Harness::start(config()).await;
    let host = h.host();
    let bearer = h.bearer();
    let r = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &bearer), ("Origin", "http://tauri.localhost")], b"").await;
    assert_eq!(r.status, 403);
    assert_eq!(r.header("access-control-allow-origin"), None);
    let pre = send(h.addr, "OPTIONS", "/v1/chat/completions", &[("Host", &host), ("Origin", "https://evil.example"), ("Access-Control-Request-Method", "POST")], b"").await;
    assert_eq!(pre.status, 403);
    assert_eq!(pre.header("access-control-allow-origin"), None);
    // No Origin at all (curl, SDKs): allowed with the token, no CORS headers.
    let plain = h.get("/v1/models").await;
    assert_eq!(plain.status, 200);
    assert_eq!(plain.header("access-control-allow-origin"), None);
    h.stop().await;
}

#[tokio::test]
async fn a_configured_app_origin_gets_cors_and_others_do_not() {
    let h = Harness::start(GatewayConfig { allowed_origins: vec!["http://tauri.localhost".into()], ..config() }).await;
    let host = h.host();
    let bearer = h.bearer();
    let pre = send(h.addr, "OPTIONS", "/v1/chat/completions", &[("Host", &host), ("Origin", "http://tauri.localhost"), ("Access-Control-Request-Method", "POST")], b"").await;
    assert_eq!(pre.status, 204);
    assert_eq!(pre.header("access-control-allow-origin"), Some("http://tauri.localhost"));
    assert_eq!(pre.header("access-control-allow-headers"), Some("authorization, content-type"));
    assert!(pre.body.is_empty());
    // A browser OpenAI SDK asks for its own headers; the allowed origin gets them echoed.
    let sdk = send(
        h.addr,
        "OPTIONS",
        "/v1/chat/completions",
        &[
            ("Host", &host),
            ("Origin", "http://tauri.localhost"),
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", "authorization, content-type, x-stainless-os, x-stainless-lang"),
        ],
        b"",
    )
    .await;
    assert_eq!(sdk.status, 204);
    assert_eq!(sdk.header("access-control-allow-headers"), Some("authorization, content-type, x-stainless-os, x-stainless-lang"));
    let ok = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &bearer), ("Origin", "http://tauri.localhost")], b"").await;
    assert_eq!(ok.status, 200);
    assert_eq!(ok.header("access-control-allow-origin"), Some("http://tauri.localhost"));
    assert_eq!(ok.header("vary"), Some("Origin"));
    // The allowed origin still needs the token.
    let no_token = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Origin", "http://tauri.localhost")], b"").await;
    assert_eq!(no_token.status, 401);
    let other = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &bearer), ("Origin", "http://tauri.localhost.evil.example")], b"").await;
    assert_eq!(other.status, 403);
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Switching protections off
// ---------------------------------------------------------------------------------------------

/// The `setting` keys `/cerebro/status` lists under `gateway.disabled_protections`, read with the
/// harness token.
async fn disabled_in_status(h: &Harness) -> Vec<String> {
    let status = h.get("/cerebro/status").await;
    assert_eq!(status.status, 200, "{}", status.text());
    let v = status.json();
    let list = v["gateway"]["disabled_protections"].as_array().unwrap_or_else(|| panic!("no list: {v}")).clone();
    assert!(list.iter().all(|d| d["risk"].as_str().is_some_and(|r| !r.is_empty())), "{v}");
    list.iter().map(|d| d["setting"].as_str().unwrap().to_string()).collect()
}

#[tokio::test]
async fn every_gateway_protection_is_on_by_default_and_none_is_listed() {
    let h = Harness::start(config()).await;
    assert!(disabled_in_status(&h).await.is_empty());
    h.stop().await;
}

#[tokio::test]
async fn with_the_token_check_off_a_request_without_a_token_is_served_and_host_and_origin_still_apply() {
    let off = GatewayProtections { require_token: false, ..GatewayProtections::default() };
    let h = Harness::start(GatewayConfig { protections: off, ..config() }).await;
    let host = h.host();
    let port = h.addr.port();
    assert_eq!(send(h.addr, "GET", "/v1/models", &[("Host", &host)], b"").await.status, 200);
    assert_eq!(send(h.addr, "GET", "/v1/models", &[("Host", &format!("evil.example:{port}"))], b"").await.status, 403);
    assert_eq!(send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Origin", "https://evil.example")], b"").await.status, 403);
    assert_eq!(disabled_in_status(&h).await, ["gateway_require_token"]);
    h.stop().await;
}

#[tokio::test]
async fn with_the_host_check_off_a_foreign_host_is_served_and_token_and_origin_still_apply() {
    let off = GatewayProtections { check_host: false, ..GatewayProtections::default() };
    let h = Harness::start(GatewayConfig { protections: off, ..config() }).await;
    let bearer = h.bearer();
    let port = h.addr.port();
    let foreign = format!("evil.example:{port}");
    let r = send(h.addr, "GET", "/v1/models", &[("Host", &foreign), ("Authorization", &bearer)], b"").await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(send(h.addr, "GET", "/v1/models", &[("Host", &foreign)], b"").await.status, 401);
    let origin = send(h.addr, "GET", "/v1/models", &[("Host", &foreign), ("Authorization", &bearer), ("Origin", "https://evil.example")], b"").await;
    assert_eq!(origin.status, 403);
    assert_eq!(disabled_in_status(&h).await, ["gateway_check_host"]);
    h.stop().await;
}

#[tokio::test]
async fn with_the_origin_check_off_any_origin_is_echoed_never_a_wildcard_and_token_and_host_still_apply() {
    let off = GatewayProtections { check_origin: false, ..GatewayProtections::default() };
    let h = Harness::start(GatewayConfig { protections: off, ..config() }).await;
    let host = h.host();
    let bearer = h.bearer();
    let port = h.addr.port();
    let r = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Authorization", &bearer), ("Origin", "https://some.example")], b"").await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(r.header("access-control-allow-origin"), Some("https://some.example"));
    let pre = send(h.addr, "OPTIONS", "/v1/chat/completions", &[("Host", &host), ("Origin", "https://some.example"), ("Access-Control-Request-Method", "POST")], b"").await;
    assert_eq!(pre.status, 204);
    assert_eq!(pre.header("access-control-allow-origin"), Some("https://some.example"));
    let no_token = send(h.addr, "GET", "/v1/models", &[("Host", &host), ("Origin", "https://some.example")], b"").await;
    assert_eq!(no_token.status, 401);
    let foreign = send(h.addr, "GET", "/v1/models", &[("Host", &format!("evil.example:{port}")), ("Authorization", &bearer)], b"").await;
    assert_eq!(foreign.status, 403);
    assert_eq!(disabled_in_status(&h).await, ["gateway_check_origin"]);
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Body limit and timeouts
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn bodies_over_the_limit_are_refused_declared_or_chunked() {
    let h = Harness::start(GatewayConfig { max_body_bytes: 1024, ..config() }).await;
    let big = json!({"model": "tiny", "messages": [{"role": "user", "content": "x".repeat(4096)}]});
    let r = h.post("/v1/chat/completions", &big).await;
    assert_eq!(r.status, 413);
    assert_eq!(r.json()["error"]["type"], "payload_too_large");

    // Chunked: no Content-Length, so the limit is enforced while reading.
    let mut stream = TcpStream::connect(h.addr).await.unwrap();
    let head = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: {}\r\nAuthorization: {}\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n",
        h.host(),
        h.bearer()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    let chunk = "y".repeat(600);
    for _ in 0..3 {
        stream.write_all(format!("{:x}\r\n{chunk}\r\n", chunk.len()).as_bytes()).await.unwrap();
    }
    let _ = stream.write_all(b"0\r\n\r\n").await;
    let mut raw = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut raw)).await;
    assert_eq!(parse_reply(&raw).status, 413);
    assert!(h.seen().is_empty());

    // Under the limit passes.
    assert_eq!(h.post("/v1/embeddings", &json!({"input": "small"})).await.status, 200);
    h.stop().await;
}

#[tokio::test]
async fn a_slow_model_server_times_out_with_504() {
    let timeouts = Timeouts { generation: Duration::from_millis(300), ..Timeouts::default() };
    let h = Harness::start(GatewayConfig { timeouts, ..config() }).await;
    *h.fake.delay.lock().unwrap() = Some(Duration::from_secs(3));
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "low"}))).await;
    assert_eq!(r.status, 504, "{}", r.text());
    h.stop().await;
}

#[tokio::test]
async fn a_stream_that_goes_idle_is_cut_off() {
    let timeouts = Timeouts { stream_idle: Duration::from_millis(300), ..Timeouts::default() };
    let h = Harness::start(GatewayConfig { timeouts, ..config() }).await;
    // The fake stream sends its first chunk and then waits on a gate that is never opened.
    let r = tokio::time::timeout(Duration::from_secs(10), h.post("/v1/chat/completions", &chat_body(json!({"stream": true, "cerebro_effort": "low"}))))
        .await
        .expect("the gateway must end an idle stream");
    assert_eq!(r.status, 200);
    let text = r.text();
    assert!(text.contains("Hel") && !text.contains("[DONE]"), "{text}");
    h.stop().await;
}

#[tokio::test]
async fn requests_fail_fast_while_the_model_server_is_not_ready() {
    let h = Harness::start(config()).await;
    h.upstream_state.send_replace(LifecycleState::Starting);
    let r = h.post("/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(r.status, 503);
    assert!(r.json()["error"]["message"].as_str().unwrap().contains("starting"));
    let health = h.get("/cerebro/health").await;
    assert_eq!(health.status, 503);
    assert_eq!(health.json()["upstream"], "starting");
    assert!(h.seen().is_empty());
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// SSE relay and status
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn sse_is_relayed_as_it_arrives_and_its_speed_reaches_status() {
    let h = Harness::start(config()).await;
    let body = chat_body(json!({"stream": true, "cerebro_effort": "low"})).to_string();
    let mut stream = TcpStream::connect(h.addr).await.unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nAuthorization: {}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        h.host(),
        h.bearer(),
        body.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();

    // The first chunk must arrive while upstream is still holding the rest back.
    let mut received = Vec::new();
    let mut buf = [0u8; 4096];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !String::from_utf8_lossy(&received).contains("Hel") {
        let n = tokio::time::timeout_at(deadline, stream.read(&mut buf)).await.expect("first chunk before upstream finished").unwrap();
        assert!(n > 0, "connection closed early");
        received.extend_from_slice(&buf[..n]);
    }
    let so_far = String::from_utf8_lossy(&received).into_owned();
    assert!(so_far.contains("text/event-stream"), "{so_far}");
    assert!(!so_far.contains("[DONE]"));
    h.fake.gate.notify_one();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut received)).await.unwrap().unwrap();
    let reply = parse_reply(&received);
    let text = reply.text();
    let hel = text.find("Hel").unwrap();
    let lo = text.find("\"lo\"").unwrap();
    let done = text.find("data: [DONE]").unwrap();
    assert!(hel < lo && lo < done, "{text}");

    let status = h.get("/cerebro/status").await.json();
    assert_eq!(status["speed"]["decode_tokens_per_second"], 63.25);
    assert_eq!(status["speed"]["prefill_tokens_per_second"], 812.5);
    assert_eq!(status["vram"][0]["free_mib"], 6000);
    assert_eq!(status["model"]["alias"], "tiny");
    assert_eq!(status["upstream"]["state"], "ready");
    assert_eq!(status["defaults"]["verbosity"], "brief");
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Effort and verbosity
// ---------------------------------------------------------------------------------------------

const MEDIUM: &str = "Before answering, check the key facts, numbers and any code you write. If you are unsure \
about something important, say so in one short sentence.";
const HIGH: &str = "Think the problem through before answering. Verify every claim, calculation and line of code \
you give. State your assumptions. After drafting, reread your answer once and fix any mistake.";
const BRIEF: &str = "Lead with the answer in one or two sentences. Add only the detail needed to act on it, and \
offer to expand rather than expanding unasked.";
const DETAILED: &str = "Give the answer first, then a thorough explanation: reasoning, steps, alternatives and \
caveats. Use headings or lists where they help.";

#[tokio::test]
async fn defaults_inject_medium_and_brief_into_a_new_system_message() {
    let h = Harness::start(config()).await;
    let r = h.post("/v1/chat/completions", &chat_body(json!({"max_tokens": 100}))).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let sent = &h.chat_bodies()[0];
    assert_eq!(sent["messages"][0], json!({"role": "system", "content": format!("{MEDIUM}\n\n{BRIEF}")}));
    assert_eq!(sent["messages"][1]["content"], "What is 2+2?");
    // Medium thinks up to 1024 tokens; the client's 100 is the answer budget on top.
    assert_eq!(sent["reasoning_budget_tokens"], 1024);
    assert_eq!(sent["max_tokens"], 1124);
    let reply = r.json();
    assert_eq!(reply["cerebro"]["effort"], "medium");
    assert_eq!(reply["cerebro"]["verbosity"], "brief");
    assert_eq!(reply["cerebro"]["answer_budget_tokens"], 100);
    assert_eq!(reply["cerebro"]["passes"][0]["pass"], "draft");
    assert_eq!(reply["choices"][0]["message"]["content"], "ok");
    h.stop().await;
}

#[tokio::test]
async fn requested_levels_append_exact_text_to_an_existing_system_prompt() {
    let h = Harness::start(config()).await;
    let body = json!({
        "messages": [{"role": "system", "content": "You are a calculator."}, {"role": "user", "content": "2+2"}],
        "cerebro_effort": "high",
        "cerebro_verbosity": "detailed",
    });
    assert_eq!(h.post("/v1/chat/completions", &body).await.status, 200);
    let sent = &h.chat_bodies()[0];
    assert_eq!(sent["messages"][0]["content"], format!("You are a calculator.\n\n{HIGH}\n\n{DETAILED}"));
    assert_eq!(sent["messages"].as_array().unwrap().len(), 2);
    assert!(sent.get("cerebro_effort").is_none() && sent.get("cerebro_verbosity").is_none());
    assert_eq!(sent["reasoning_budget_tokens"], 4096);
    assert!(sent.get("max_tokens").is_none(), "no client limit: the answer is not capped");

    // Low turns thinking off.
    assert_eq!(h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "low"}))).await.status, 200);
    assert_eq!(h.chat_bodies()[1]["reasoning_budget_tokens"], 0);

    // An unknown level is a client error and never reaches upstream.
    let bad = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "turbo"}))).await;
    assert_eq!(bad.status, 400);
    assert_eq!(h.chat_bodies().len(), 2);
    h.stop().await;
}

#[tokio::test]
async fn completions_and_embeddings_pass_through_without_cerebro_fields() {
    let h = Harness::start(config()).await;
    let r = h.post("/v1/completions", &json!({"prompt": "Once", "cerebro_effort": "max"})).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json()["received"], json!({"prompt": "Once"}));
    let e = h.post("/v1/embeddings", &json!({"input": ["a", "b"]})).await;
    assert_eq!(e.json()["received"], json!({"input": ["a", "b"]}));
    h.stop().await;
}

#[tokio::test]
async fn thinking_is_clamped_to_the_context_using_the_tokenizer_when_the_byte_bound_does_not_fit() {
    let h = Harness::start(config()).await;
    *h.fake.n_ctx.lock().unwrap() = 8192;
    *h.fake.token_count.lock().unwrap() = 3000;
    // A prompt whose byte bound (~6 KB) plus 16384 thinking cannot fit 8192, so it is tokenized:
    // 3000 tokens + 2 * 16 framing, answer 500: 8192 - 3032 - 500 = 4660 tokens of thinking.
    let long = "word ".repeat(1200);
    let body = json!({"messages": [{"role": "user", "content": long}], "cerebro_effort": "xhigh", "max_tokens": 500});
    h.fake.replies.lock().unwrap().extend(["draft".to_string(), "final".to_string()]);
    let r = h.post("/v1/chat/completions", &body).await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert!(h.seen().iter().any(|(p, _, _)| p == "/tokenize"));
    let sent = &h.chat_bodies()[0];
    assert_eq!(sent["reasoning_budget_tokens"], 4660);
    assert_eq!(sent["max_tokens"], 5160);
    assert_eq!(r.json()["cerebro"]["thinking_clamped_to_context"], true);

    // A short prompt with room to spare is not tokenized.
    let before = h.seen().iter().filter(|(p, _, _)| p == "/tokenize").count();
    assert_eq!(h.post("/v1/chat/completions", &chat_body(json!({}))).await.status, 200);
    assert_eq!(h.seen().iter().filter(|(p, _, _)| p == "/tokenize").count(), before);
    h.stop().await;
}

#[tokio::test]
async fn a_model_without_a_thinking_mode_keeps_the_clients_answer_limit() {
    let h = Harness::start(config()).await;
    *h.fake.template.lock().unwrap() = Some("<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>".to_string());
    h.fake.replies.lock().unwrap().extend(["a".to_string(), "b".to_string()]);
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "xhigh", "max_tokens": 24}))).await;
    assert_eq!(r.status, 200, "{}", r.text());
    for sent in h.chat_bodies() {
        assert_eq!(sent["max_tokens"], 24, "no thinking to reserve: the client's limit is the whole limit");
        assert_eq!(sent["reasoning_budget_tokens"], 0);
    }
    let reply = r.json();
    assert_eq!(reply["cerebro"]["thinking_supported"], false);
    assert_eq!(reply["cerebro"]["thinking_budget_tokens"], 0);

    h.stop().await;

    // A thinking template gets the level's budget on top of the answer budget.
    let h = Harness::start(config()).await;
    *h.fake.template.lock().unwrap() = Some("{% if enable_thinking %}<think>{% endif %}".to_string());
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "high", "max_tokens": 24}))).await;
    assert_eq!(r.json()["cerebro"]["thinking_supported"], true);
    assert_eq!(h.chat_bodies()[0]["max_tokens"], 4096 + 24);
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Review passes
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn xhigh_runs_a_self_review_before_returning_and_reports_timing() {
    let h = Harness::start(config()).await;
    h.fake.replies.lock().unwrap().extend(["2+2 is 5".to_string(), "2+2 is 4".to_string()]);
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "xhigh"}))).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let reply = r.json();
    assert_eq!(reply["choices"][0]["message"]["content"], "2+2 is 4");
    let passes: Vec<&str> = reply["cerebro"]["passes"].as_array().unwrap().iter().map(|p| p["pass"].as_str().unwrap()).collect();
    assert_eq!(passes, ["draft", "self_review"]);
    assert!(reply["cerebro"]["total_ms"].as_u64().is_some());
    assert!(reply["cerebro"]["passes"][1]["ms"].as_u64().is_some());
    assert_eq!(reply["usage"]["total_tokens"], 24);

    let bodies = h.chat_bodies();
    assert_eq!(bodies.len(), 2);
    let review = bodies[1]["messages"].as_array().unwrap();
    assert_eq!(review[review.len() - 2], json!({"role": "assistant", "content": "2+2 is 5"}));
    assert_eq!(review[review.len() - 1], json!({"role": "user", "content": REVIEW_INSTRUCTION}));
    assert!(bodies.iter().all(|b| b["stream"] == false));
    h.stop().await;
}

#[tokio::test]
async fn max_orders_draft_review_independent_check_then_correction() {
    let h = Harness::start(config()).await;
    h.fake.replies.lock().unwrap().extend([
        "draft: 5".to_string(),
        "reviewed: 5".to_string(),
        "Error: 2+2 is 4, not 5.".to_string(),
        "corrected: 4".to_string(),
    ]);
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "max"}))).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let reply = r.json();
    assert_eq!(reply["choices"][0]["message"]["content"], "corrected: 4");
    let passes: Vec<&str> = reply["cerebro"]["passes"].as_array().unwrap().iter().map(|p| p["pass"].as_str().unwrap()).collect();
    assert_eq!(passes, ["draft", "self_review", "independent_check", "correction"]);
    assert_eq!(reply["cerebro"]["passes"][2]["found_errors"], true);
    // Max is configured for 3 consistency paths; only one runs today, and the reply says so.
    assert_eq!(reply["cerebro"]["consistency_paths"]["configured"], 3);
    assert_eq!(reply["cerebro"]["consistency_paths"]["run"], 1);

    let bodies = h.chat_bodies();
    let check = bodies[2]["messages"].as_array().unwrap();
    assert_eq!(check.len(), 2, "the check runs in a fresh context");
    assert_eq!(check[0]["content"], CHECK_SYSTEM);
    assert_eq!(check[1]["content"], format!("Question:\nWhat is 2+2?\n\nAnswer:\nreviewed: 5\n\n{CHECK_INSTRUCTION}"));
    let fix = bodies[3]["messages"].as_array().unwrap();
    assert!(fix[fix.len() - 1]["content"].as_str().unwrap().contains("2+2 is 4, not 5."));
    h.stop().await;
}

/// Small models explain a clean verdict; the verdict line decides, so no correction runs.
#[tokio::test]
async fn max_with_a_clean_verdict_plus_explanation_runs_no_correction() {
    let h = Harness::start(config()).await;
    h.fake.replies.lock().unwrap().extend([
        "draft: 4".to_string(),
        "reviewed: 4".to_string(),
        "No errors found.\nThe answer correctly computes 2+2 = 4.".to_string(),
    ]);
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "max"}))).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let reply = r.json();
    assert_eq!(reply["choices"][0]["message"]["content"], "reviewed: 4");
    let passes: Vec<&str> = reply["cerebro"]["passes"].as_array().unwrap().iter().map(|p| p["pass"].as_str().unwrap()).collect();
    assert_eq!(passes, ["draft", "self_review", "independent_check"]);
    assert_eq!(reply["cerebro"]["passes"][2]["found_errors"], false);
    assert_eq!(h.chat_bodies().len(), 3, "no fourth (correction) request");
    h.stop().await;
}

#[tokio::test]
async fn streamed_max_sends_progress_then_the_reviewed_answer_then_done() {
    let h = Harness::start(config()).await;
    h.fake.replies.lock().unwrap().extend(["draft".to_string(), "reviewed".to_string(), "NO ERRORS FOUND".to_string()]);
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "max", "stream": true}))).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    let text = r.text();
    let order = [": cerebro pass draft", ": cerebro pass self_review", ": cerebro pass independent_check", "\"content\":\"reviewed\"", "\"cerebro\"", "data: [DONE]"];
    let mut at = 0;
    for marker in order {
        let found = text[at..].find(marker).unwrap_or_else(|| panic!("{marker:?} missing or out of order in:\n{text}"));
        at += found + marker.len();
    }
    assert!(!text.contains(": cerebro pass correction"));
    assert!(!text.contains("\"content\":\"draft\""), "the unreviewed draft is never streamed");
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Malformed upstream replies
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_non_object_upstream_reply_at_xhigh_is_a_502_not_a_crash() {
    let h = Harness::start(config()).await;
    *h.fake.raw.lock().unwrap() = Some("[]".to_string());
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "xhigh"}))).await;
    assert_eq!(r.status, 502, "{}", r.text());
    assert_eq!(r.json()["error"]["type"], "upstream_error");
    // Streamed, the same reply becomes an SSE error event.
    let s = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "max", "stream": true}))).await;
    assert_eq!(s.status, 200);
    assert!(s.text().contains("not a JSON object") && s.text().contains("data: [DONE]"), "{}", s.text());
    // The gateway is still up.
    *h.fake.raw.lock().unwrap() = None;
    assert_eq!(h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "low"}))).await.status, 200);
    h.stop().await;
}

#[tokio::test]
async fn a_streamed_max_tool_call_holding_a_string_is_relayed_without_a_crash() {
    let h = Harness::start(config()).await;
    *h.fake.raw.lock().unwrap() =
        Some(r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":["oops"]},"finish_reason":"tool_calls"}]}"#.to_string());
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "max", "stream": true}))).await;
    assert_eq!(r.status, 200);
    let text = r.text();
    assert!(text.contains(r#""tool_calls":["oops"]"#) && text.contains("data: [DONE]"), "{text}");
    assert_eq!(h.chat_bodies().len(), 1, "a tool call is not reviewed");
    assert_eq!(h.get("/cerebro/health").await.status, 200);
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Streamed replies below XHigh carry the cerebro field
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_streamed_medium_reply_ends_with_a_cerebro_chunk_before_done() {
    let h = Harness::start(config()).await;
    h.fake.gate.notify_one();
    let r = h.post("/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 50}))).await;
    assert_eq!(r.status, 200);
    let text = r.text();
    let lo = text.find("\"lo\"").expect("content relayed");
    let report_at = text.find("\"cerebro\"").expect("cerebro chunk");
    let done = text.find("data: [DONE]").expect("done");
    assert!(lo < report_at && report_at < done, "{text}");
    let line = text[..done].lines().rev().find(|l| l.starts_with("data: ")).expect("last event");
    let chunk: Value = serde_json::from_str(line.trim_start_matches("data: ")).expect("json");
    assert_eq!(chunk["object"], "chat.completion.chunk");
    assert_eq!(chunk["choices"], json!([]));
    assert_eq!(chunk["cerebro"]["effort"], "medium");
    assert_eq!(chunk["cerebro"]["answer_budget_tokens"], 50);
    assert_eq!(chunk["cerebro"]["passes"][0]["pass"], "draft");
    assert_eq!(chunk["cerebro"]["passes"][0]["tokens_per_second"], 63.25, "from the stream's own timings");
    assert_eq!(text.matches("\"cerebro\"").count(), 1);
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Protections: header and body timeouts, n > 1, disconnects, keepalive, connection cap
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_client_that_never_finishes_its_headers_is_dropped() {
    let timeouts = Timeouts { header_read: Duration::from_millis(300), ..Timeouts::default() };
    let h = Harness::start(GatewayConfig { timeouts, ..config() }).await;
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    s.write_all(b"GET /v1/models HTTP/1.1\r\nHost: ").await.unwrap();
    let mut buf = Vec::new();
    let closed = tokio::time::timeout(Duration::from_secs(5), s.read_to_end(&mut buf)).await;
    assert!(closed.is_ok(), "the header-read timeout must close the connection");
    assert!(h.seen().is_empty());
    h.stop().await;
}

#[tokio::test]
async fn a_body_that_stalls_gets_408() {
    let timeouts = Timeouts { body_read: Duration::from_millis(300), ..Timeouts::default() };
    let h = Harness::start(GatewayConfig { timeouts, ..config() }).await;
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nAuthorization: {}\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{{\"a\"",
        h.host(),
        h.bearer()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), s.read_to_end(&mut raw)).await;
    assert_eq!(parse_reply(&raw).status, 408);
    assert!(h.chat_bodies().is_empty());
    h.stop().await;
}

#[tokio::test]
async fn n_above_one_is_refused_at_review_levels_before_anything_goes_upstream() {
    let h = Harness::start(config()).await;
    for level in ["xhigh", "max"] {
        let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": level, "n": 2}))).await;
        assert_eq!(r.status, 400, "{level}: {}", r.text());
        assert!(r.json()["error"]["message"].as_str().unwrap().contains("n > 1"));
    }
    assert!(h.seen().is_empty(), "not even /props or /tokenize");
    // Below XHigh, n is the model server's business.
    assert_eq!(h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "low", "n": 2}))).await.status, 200);
    h.stop().await;
}

#[tokio::test]
async fn a_client_that_disconnects_during_streamed_passes_stops_further_passes() {
    let timeouts = Timeouts { sse_keepalive: Duration::from_millis(50), ..Timeouts::default() };
    let h = Harness::start(GatewayConfig { timeouts, ..config() }).await;
    *h.fake.delay.lock().unwrap() = Some(Duration::from_millis(400));
    h.fake.replies.lock().unwrap().extend(["draft".to_string(), "reviewed".to_string(), "Wrong.".to_string(), "fixed".to_string()]);
    let body = chat_body(json!({"cerebro_effort": "max", "stream": true})).to_string();
    let mut s = TcpStream::connect(h.addr).await.unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nAuthorization: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        h.host(),
        h.bearer(),
        body.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    let mut received = Vec::new();
    let mut buf = [0u8; 1024];
    while !String::from_utf8_lossy(&received).contains(": cerebro pass draft") {
        let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf)).await.expect("progress").unwrap();
        assert!(n > 0, "closed early");
        received.extend_from_slice(&buf[..n]);
    }
    drop(s);
    // Uncancelled, the four passes would take 1.6 s; wait past that.
    tokio::time::sleep(Duration::from_millis(2000)).await;
    assert_eq!(h.chat_bodies().len(), 1, "only the draft in flight at the disconnect went upstream");
    h.stop().await;
}

#[tokio::test]
async fn streamed_passes_send_keepalives_while_they_run() {
    let timeouts = Timeouts { sse_keepalive: Duration::from_millis(100), ..Timeouts::default() };
    let h = Harness::start(GatewayConfig { timeouts, ..config() }).await;
    *h.fake.delay.lock().unwrap() = Some(Duration::from_millis(350));
    h.fake.replies.lock().unwrap().extend(["draft".to_string(), "reviewed".to_string()]);
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "xhigh", "stream": true}))).await;
    assert_eq!(r.status, 200);
    let text = r.text();
    let keepalive = text.find(": cerebro keepalive").unwrap_or_else(|| panic!("no keepalive in:\n{text}"));
    assert!(keepalive < text.find("\"content\":\"reviewed\"").expect("answer"), "{text}");
    h.stop().await;
}

#[tokio::test]
async fn connections_beyond_the_cap_are_closed_and_service_resumes_when_they_go() {
    let h = Harness::start(GatewayConfig { max_connections: 2, ..config() }).await;
    let idle_a = TcpStream::connect(h.addr).await.unwrap();
    let idle_b = TcpStream::connect(h.addr).await.unwrap();
    // Give the accept loop time to take both permits.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let mut extra = TcpStream::connect(h.addr).await.unwrap();
    let mut buf = [0u8; 16];
    let read = tokio::time::timeout(Duration::from_secs(5), extra.read(&mut buf)).await.expect("the extra socket is closed at once");
    assert!(matches!(read, Ok(0) | Err(_)), "{read:?}");
    drop((idle_a, idle_b));
    // The idle connections' tasks end once their sockets close; then a real request is served.
    let mut status = 0;
    for _ in 0..50 {
        let ok = tokio::time::timeout(Duration::from_millis(500), h.get("/cerebro/health")).await;
        if let Ok(reply) = ok {
            status = reply.status;
            break;
        }
    }
    assert_eq!(status, 200);
    h.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Budget: thinking opt-out, tools and media
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_client_thinking_opt_out_keeps_its_answer_limit() {
    let h = Harness::start(config()).await;
    *h.fake.template.lock().unwrap() = Some("{% if enable_thinking %}<think>{% endif %}".to_string());
    let body = chat_body(json!({"cerebro_effort": "high", "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": false}}));
    let r = h.post("/v1/chat/completions", &body).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let sent = &h.chat_bodies()[0];
    assert_eq!(sent["max_tokens"], 24);
    assert_eq!(sent["reasoning_budget_tokens"], 0);
    assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], false);
    h.stop().await;
}

fn big_tools() -> Value {
    let properties: serde_json::Map<String, Value> =
        (0..60).map(|i| (format!("field_{i}"), json!({"type": "string", "description": "a long description of this field"}))).collect();
    json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": properties}}}])
}

#[tokio::test]
async fn tool_definitions_count_toward_the_prompt_and_can_clamp_thinking() {
    let h = Harness::start(config()).await;
    *h.fake.n_ctx.lock().unwrap() = 8192;
    *h.fake.token_count.lock().unwrap() = 6000;
    // The messages alone fit easily; with the tool schemas the byte bound does not, so the prompt
    // is tokenized with the schemas: 6000 + 2 messages * 16 + 1024 template = 7056 tokens, answer
    // 100, leaving 1036 of High's 4096 for thinking.
    let r = h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "high", "max_tokens": 100, "tools": big_tools()}))).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let tokenized = h.seen().into_iter().find(|(p, _, _)| p == "/tokenize").expect("tokenized");
    assert!(tokenized.2["content"].as_str().unwrap().contains("get_weather"));
    let sent = &h.chat_bodies()[0];
    assert_eq!(sent["reasoning_budget_tokens"], 1036);
    assert_eq!(r.json()["cerebro"]["thinking_clamped_to_context"], true);

    // Without the tools nothing is tokenized.
    let before = h.seen().iter().filter(|(p, _, _)| p == "/tokenize").count();
    assert_eq!(h.post("/v1/chat/completions", &chat_body(json!({"cerebro_effort": "high", "max_tokens": 100}))).await.status, 200);
    assert_eq!(h.seen().iter().filter(|(p, _, _)| p == "/tokenize").count(), before);
    h.stop().await;
}

fn image_body(effort: &str) -> Value {
    let image = format!("data:image/png;base64,{}", "A".repeat(1 << 20));
    json!({"cerebro_effort": effort, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": image}},
    ]}]})
}

#[tokio::test]
async fn an_attached_image_counts_a_fixed_bound_and_is_never_tokenized_as_text() {
    let h = Harness::start(config()).await;
    *h.fake.n_ctx.lock().unwrap() = 16_384;
    let r = h.post("/v1/chat/completions", &image_body("high")).await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert!(!h.seen().iter().any(|(p, _, _)| p == "/tokenize"), "a 1 MB image fits by its fixed bound");
    assert_eq!(h.chat_bodies()[0]["reasoning_budget_tokens"], 4096);
    assert_eq!(r.json()["cerebro"]["thinking_clamped_to_context"], false);

    // A tighter context is tokenized, without the image data: 20 tokens + 2 * 16 + 4096 for the
    // image, reserve 2048: 9000 - 4148 - 2048 = 2804 of thinking, not 0.
    // `/props` is cached per upstream, so the tighter context gets a fresh harness.
    h.stop().await;
    let h = Harness::start(config()).await;
    *h.fake.n_ctx.lock().unwrap() = 9000;
    *h.fake.token_count.lock().unwrap() = 20;
    let r = h.post("/v1/chat/completions", &image_body("high")).await;
    assert_eq!(r.status, 200, "{}", r.text());
    let tokenized = h.seen().into_iter().find(|(p, _, _)| p == "/tokenize").expect("tokenized");
    let content = tokenized.2["content"].as_str().unwrap().to_string();
    assert!(content.contains("What is in this picture?") && !content.contains("AAAA"), "{} bytes sent to /tokenize", content.len());
    assert_eq!(h.chat_bodies()[0]["reasoning_budget_tokens"], 2804);
    h.stop().await;
}
