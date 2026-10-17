//! Cerebro as an MCP server: other programs on this computer (Claude Code, VS Code, Cursor, the
//! DeepSeek harness...) use the local models through MCP's Streamable HTTP transport (2025-06-18).
//!
//! Off by default. Binds 127.0.0.1 only; every request needs the access token as a bearer token;
//! a browser Origin or a Host other than this loopback address is refused (DNS rebinding). Replies
//! are plain JSON (the transport's single-response form); no server-initiated stream is offered.

use std::collections::HashMap;
use std::io::BufReader;
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use subtle::ConstantTimeEq;

use crate::remote::{now_ms, random_bytes, read_request, write_response, HttpError, Request, Response};

/// What the tools reach: implemented over the engine in commands.rs, by a fake in the tests.
pub trait McpBackend: Send + Sync + 'static {
    /// One reply. `turns` are (role, text); blocking. Returns (text, output tokens, tok/s, model).
    fn generate(&self, system: &str, turns: Vec<(String, String)>, max_tokens: usize, temperature: f32) -> Result<(String, u64, f64, String), String>;
    fn models(&self) -> Value;
    fn load(&self, id: &str, context: Option<usize>) -> Result<String, String>;
    fn status(&self) -> Value;
}

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub enabled: bool,
    /// Fixed once chosen, so app configs keep working; 0 = pick on first start.
    pub port: u16,
    pub token: String,
}

/// Protocol versions this server speaks, newest first.
const VERSIONS: [&str; 2] = ["2025-06-18", "2025-03-26"];
const DEFAULT_PORT: u16 = 47_811;
const MAX_CONNECTIONS: usize = 16;
const MAX_SESSIONS: usize = 64;

#[derive(Clone)]
struct Session {
    name: String,
    version: String,
    last_seen_at: u64,
    calls: u64,
}

pub struct McpServer {
    settings: Mutex<Settings>,
    backend: Arc<dyn McpBackend>,
    save: Box<dyn Fn(&Settings) + Send + Sync>,
    running: Mutex<Option<(Arc<AtomicBool>, SocketAddr)>>,
    error: Mutex<Option<String>>,
    sessions: Mutex<HashMap<String, Session>>,
    connections: Arc<AtomicUsize>,
}

fn new_token() -> Result<String, String> {
    Ok(URL_SAFE_NO_PAD.encode(random_bytes::<32>()?))
}

impl McpServer {
    pub fn new(mut settings: Settings, backend: Arc<dyn McpBackend>, save: Box<dyn Fn(&Settings) + Send + Sync>) -> Arc<Self> {
        if settings.token.len() < 32 {
            settings.token = new_token().unwrap_or_default();
        }
        if settings.port == 0 {
            settings.port = DEFAULT_PORT;
        }
        save(&settings);
        Arc::new(McpServer {
            settings: Mutex::new(settings),
            backend,
            save,
            running: Mutex::new(None),
            error: Mutex::new(None),
            sessions: Mutex::new(HashMap::new()),
            connections: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn settings(&self) -> std::sync::MutexGuard<'_, Settings> {
        self.settings.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn persist(&self) {
        let s = self.settings().clone();
        (self.save)(&s);
    }

    fn bound(&self) -> Option<SocketAddr> {
        self.running.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|r| r.1)
    }

    pub fn view(&self) -> Value {
        let s = self.settings().clone();
        let addr = self.bound();
        let mut clients: Vec<Session> = self.sessions.lock().unwrap_or_else(|e| e.into_inner()).values().cloned().collect();
        clients.sort_by_key(|c| std::cmp::Reverse(c.last_seen_at));
        json!({
            "enabled": s.enabled,
            "running": addr.is_some(),
            "url": addr.map(|a| format!("http://{a}/mcp")),
            "port": addr.map_or(s.port, |a| a.port()),
            "token": s.token,
            "clients": clients.iter().map(|c| json!({ "name": c.name, "version": c.version, "lastSeenAt": c.last_seen_at, "calls": c.calls })).collect::<Vec<_>>(),
            "error": self.error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        })
    }

    /// Starts the server if it was left on (app start).
    pub fn resume(self: &Arc<Self>) {
        if self.settings().enabled {
            if let Err(e) = self.start() {
                *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
            }
        }
    }

    pub fn set_enabled(self: &Arc<Self>, on: bool) -> Result<(), String> {
        self.settings().enabled = on;
        self.persist();
        self.stop();
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if on {
            if let Err(e) = self.start() {
                *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e.clone());
                return Err(e);
            }
        }
        Ok(())
    }

    /// A new token; every app using the old one must be given the new one.
    pub fn rotate_token(&self) -> Result<(), String> {
        self.settings().token = new_token()?;
        self.sessions.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.persist();
        Ok(())
    }

    pub fn stop(&self) {
        if let Some((flag, _)) = self.running.lock().unwrap_or_else(|e| e.into_inner()).take() {
            flag.store(true, Ordering::SeqCst);
        }
    }

    fn start(self: &Arc<Self>) -> Result<(), String> {
        let port = self.settings().port;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).or_else(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0))).map_err(|e| format!("Could not listen on 127.0.0.1 ({e})."))?;
        let addr = listener.local_addr().map_err(|_| "Could not start the server.".to_owned())?;
        if addr.port() != port {
            // The usual port was taken: keep the new one so app configs stay valid from now on.
            self.settings().port = addr.port();
            self.persist();
        }
        listener.set_nonblocking(true).map_err(|_| "Could not start the server.".to_owned())?;
        let stop = Arc::new(AtomicBool::new(false));
        let (me, st) = (Arc::clone(self), Arc::clone(&stop));
        std::thread::Builder::new()
            .name("cerebro-mcp-server".into())
            .spawn(move || me.accept_loop(listener, st))
            .map_err(|_| "Could not start the server.".to_owned())?;
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = Some((stop, addr));
        Ok(())
    }

    fn accept_loop(self: Arc<Self>, listener: TcpListener, stop: Arc<AtomicBool>) {
        while !stop.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, peer)) => {
                    if !peer.ip().is_loopback() || self.connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                        if peer.ip().is_loopback() {
                            self.connections.fetch_sub(1, Ordering::SeqCst);
                        }
                        continue;
                    }
                    let (me, st) = (Arc::clone(&self), Arc::clone(&stop));
                    let ok = std::thread::Builder::new().name("cerebro-mcp-conn".into()).spawn(move || {
                        me.serve(stream, &st);
                        me.connections.fetch_sub(1, Ordering::SeqCst);
                    });
                    if ok.is_err() {
                        self.connections.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    }

    fn serve(&self, stream: TcpStream, stop: &AtomicBool) {
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(120)));
        let _ = stream.set_nodelay(true);
        let Ok(write) = stream.try_clone() else { return };
        let mut write = write;
        let mut io = BufReader::new(stream);
        for _ in 0..10_000 {
            let req = match read_request(&mut io) {
                Ok(r) => r,
                Err(HttpError::Closed) => return,
                Err(HttpError::Bad(code)) => {
                    let _ = write_response(&mut write, &Response::error(code, "Bad request"), false);
                    return;
                }
            };
            if stop.load(Ordering::SeqCst) {
                return;
            }
            let keep = req.keep_alive();
            let resp = self.handle(&req);
            if write_response(&mut write, &resp, keep).is_err() || !keep {
                return;
            }
        }
    }

    /// One HTTP request: transport checks, then JSON-RPC.
    pub(crate) fn handle(&self, req: &Request) -> Response {
        let port = self.bound().map_or(self.settings().port, |a| a.port());
        let host_ok = req.header("host").is_some_and(|h| h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}"));
        if !host_ok {
            return Response::error(421, "Wrong host");
        }
        // A web page could reach a loopback port; a real MCP client sends no Origin, or a local one.
        if let Some(origin) = req.header("origin") {
            let local = origin == "null" || origin.starts_with("http://127.0.0.1") || origin.starts_with("http://localhost");
            if !local {
                return Response::error(403, "Forbidden origin");
            }
        }
        if req.path != "/mcp" {
            return Response::error(404, "Not found");
        }
        let token = self.settings().token.clone();
        let given = req.header("authorization").and_then(|a| a.strip_prefix("Bearer ")).unwrap_or("");
        if given.is_empty() || !bool::from(given.as_bytes().ct_eq(token.as_bytes())) {
            return Response::error(401, "A valid access token is required").with("WWW-Authenticate", "Bearer realm=\"cerebro\"".into());
        }
        match req.method.as_str() {
            "POST" => {}
            // This server never pushes messages on its own, so there is no stream to open.
            "GET" => return Response::error(405, "No server stream").with("Allow", "POST, DELETE".into()),
            "DELETE" => {
                if let Some(id) = req.header("mcp-session-id") {
                    self.sessions.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
                }
                return Response::json(200, json!({}));
            }
            _ => return Response::error(405, "Method not allowed"),
        }
        let Ok(msg) = serde_json::from_slice::<Value>(&req.body) else {
            return Response::json(400, rpc_error(Value::Null, -32700, "Parse error"));
        };
        if msg.is_array() {
            return Response::json(400, rpc_error(Value::Null, -32600, "Batches are not supported (MCP 2025-06-18)"));
        }
        let method = msg["method"].as_str().unwrap_or("");
        let id = msg.get("id").cloned();
        if method == "initialize" {
            return self.initialize(id.unwrap_or(Value::Null), &msg["params"]);
        }
        let Some(sid) = req.header("mcp-session-id").map(str::to_owned) else {
            return Response::json(400, rpc_error(id.unwrap_or(Value::Null), -32600, "Missing Mcp-Session-Id; send initialize first"));
        };
        {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            let Some(s) = sessions.get_mut(&sid) else {
                return Response::json(404, rpc_error(id.unwrap_or(Value::Null), -32001, "Session expired; initialize again"));
            };
            s.last_seen_at = now_ms();
            if method == "tools/call" {
                s.calls += 1;
            }
        }
        // Notifications and responses get 202 with no body.
        let Some(id) = id else { return Response { status: 202, content_type: "application/json", body: Vec::new(), extra: Vec::new() } };
        let result = match method {
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tools() })),
            "tools/call" => Ok(self.call(&msg["params"])),
            _ => Err((-32601, format!("Method not found: {method}"))),
        };
        match result {
            Ok(r) => Response::json(200, json!({ "jsonrpc": "2.0", "id": id, "result": r })),
            Err((code, m)) => Response::json(200, rpc_error(id, code, &m)),
        }
    }

    fn initialize(&self, id: Value, params: &Value) -> Response {
        let asked = params["protocolVersion"].as_str().unwrap_or(VERSIONS[0]);
        let version = VERSIONS.iter().find(|v| **v == asked).copied().unwrap_or(VERSIONS[0]);
        let Ok(raw) = random_bytes::<16>() else { return Response::json(500, rpc_error(id, -32603, "Try again")) };
        let sid = URL_SAFE_NO_PAD.encode(raw);
        let clean = |v: &Value, max: usize| v.as_str().unwrap_or("").chars().filter(|c| !c.is_control()).take(max).collect::<String>();
        let session = Session { name: clean(&params["clientInfo"]["name"], 64), version: clean(&params["clientInfo"]["version"], 32), last_seen_at: now_ms(), calls: 0 };
        {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            if sessions.len() >= MAX_SESSIONS {
                if let Some(oldest) = sessions.iter().min_by_key(|(_, s)| s.last_seen_at).map(|(k, _)| k.clone()) {
                    sessions.remove(&oldest);
                }
            }
            sessions.insert(sid.clone(), session);
        }
        Response::json(
            200,
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": "cerebro", "title": "Cerebro", "version": env!("CARGO_PKG_VERSION") },
                    "instructions": "Local language models running on this computer through Cerebro. Use `generate` or `chat` for text, `status` to see what is loaded, `list_models` and `load_model` to switch models.",
                }
            }),
        )
        .with("Mcp-Session-Id", sid)
    }

    fn call(&self, params: &Value) -> Value {
        let name = params["name"].as_str().unwrap_or("");
        let args = &params["arguments"];
        let text = |t: String| json!({ "content": [{ "type": "text", "text": t }], "isError": false });
        let fail = |t: String| json!({ "content": [{ "type": "text", "text": t }], "isError": true });
        let max_tokens = args["max_tokens"].as_u64().map_or(1024, |n| n.clamp(1, 8192) as usize);
        let temperature = args["temperature"].as_f64().map_or(0.7, |t| t.clamp(0.0, 2.0) as f32);
        let system = args["system"].as_str().unwrap_or("").chars().take(32_000).collect::<String>();
        match name {
            "generate" => {
                let Some(prompt) = args["prompt"].as_str().filter(|p| !p.trim().is_empty()) else { return fail("`prompt` is required.".into()) };
                match self.backend.generate(&system, vec![("user".into(), prompt.chars().take(200_000).collect())], max_tokens, temperature) {
                    Ok((t, tokens, tps, model)) => json!({ "content": [{ "type": "text", "text": t }], "structuredContent": { "text": t, "outputTokens": tokens, "tokensPerSecond": tps, "model": model }, "isError": false }),
                    Err(e) => fail(e),
                }
            }
            "chat" => {
                let Some(msgs) = args["messages"].as_array().filter(|m| !m.is_empty() && m.len() <= 512) else { return fail("`messages` must be a non-empty list.".into()) };
                let mut turns = Vec::new();
                for m in msgs {
                    let role = m["role"].as_str().unwrap_or("");
                    let content = m["content"].as_str().unwrap_or("");
                    if role != "user" && role != "assistant" {
                        return fail("Each message needs role \"user\" or \"assistant\" and text `content`.".into());
                    }
                    turns.push((role.to_owned(), content.chars().take(200_000).collect()));
                }
                match self.backend.generate(&system, turns, max_tokens, temperature) {
                    Ok((t, tokens, tps, model)) => json!({ "content": [{ "type": "text", "text": t }], "structuredContent": { "text": t, "outputTokens": tokens, "tokensPerSecond": tps, "model": model }, "isError": false }),
                    Err(e) => fail(e),
                }
            }
            "list_models" => {
                let m = self.backend.models();
                json!({ "content": [{ "type": "text", "text": m.to_string() }], "structuredContent": { "models": m }, "isError": false })
            }
            "load_model" => {
                let Some(id) = args["id"].as_str() else { return fail("`id` is required (see list_models).".into()) };
                let ctx = args["context"].as_u64().map(|c| c.clamp(512, 1 << 20) as usize);
                match self.backend.load(id, ctx) {
                    Ok(m) => text(m),
                    Err(e) => fail(e),
                }
            }
            "status" => {
                let s = self.backend.status();
                json!({ "content": [{ "type": "text", "text": s.to_string() }], "structuredContent": s, "isError": false })
            }
            _ => fail(format!("Unknown tool: {name}")),
        }
    }
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn tools() -> Value {
    let gen_props = json!({
        "system": { "type": "string", "description": "Optional system prompt." },
        "max_tokens": { "type": "integer", "minimum": 1, "maximum": 8192, "default": 1024 },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2, "default": 0.7 },
    });
    let mut generate_props = gen_props.clone();
    generate_props["prompt"] = json!({ "type": "string", "description": "What to write about or answer." });
    let mut chat_props = gen_props;
    chat_props["messages"] = json!({ "type": "array", "items": { "type": "object", "properties": { "role": { "type": "string", "enum": ["user", "assistant"] }, "content": { "type": "string" } }, "required": ["role", "content"] } });
    let out = json!({ "type": "object", "properties": { "text": { "type": "string" }, "outputTokens": { "type": "integer" }, "tokensPerSecond": { "type": "number" }, "model": { "type": "string" } }, "required": ["text"] });
    json!([
        { "name": "generate", "title": "Generate text", "description": "Write a reply to one prompt with the model loaded in Cerebro, on this computer.",
          "inputSchema": { "type": "object", "properties": generate_props, "required": ["prompt"] }, "outputSchema": out,
          "annotations": { "readOnlyHint": true, "openWorldHint": false } },
        { "name": "chat", "title": "Chat", "description": "Reply to a conversation (user/assistant messages) with the model loaded in Cerebro.",
          "inputSchema": { "type": "object", "properties": chat_props, "required": ["messages"] }, "outputSchema": out,
          "annotations": { "readOnlyHint": true, "openWorldHint": false } },
        { "name": "list_models", "title": "List models", "description": "The models in Cerebro's library, with size, quantization and which one is loaded.",
          "inputSchema": { "type": "object", "properties": {} }, "annotations": { "readOnlyHint": true, "openWorldHint": false } },
        { "name": "load_model", "title": "Load a model", "description": "Load a model from the library (by id from list_models), replacing the loaded one.",
          "inputSchema": { "type": "object", "properties": { "id": { "type": "string" }, "context": { "type": "integer", "minimum": 512, "description": "Context length in tokens." } }, "required": ["id"] },
          "annotations": { "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false } },
        { "name": "status", "title": "Engine status", "description": "What Cerebro is doing now: loaded model, speed, memory and GPU state.",
          "inputSchema": { "type": "object", "properties": {} }, "annotations": { "readOnlyHint": true, "openWorldHint": false } },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    struct Fake;
    impl McpBackend for Fake {
        fn generate(&self, system: &str, turns: Vec<(String, String)>, max_tokens: usize, _t: f32) -> Result<(String, u64, f64, String), String> {
            Ok((format!("sys={system} turns={} last={} max={max_tokens}", turns.len(), turns.last().map_or("", |t| t.1.as_str())), 7, 123.0, "Qwen3 0.6B".into()))
        }
        fn models(&self) -> Value {
            json!([{ "id": "m1", "loaded": true }])
        }
        fn load(&self, id: &str, _c: Option<usize>) -> Result<String, String> {
            if id == "m1" { Ok("Loaded m1.".into()) } else { Err("No such model.".into()) }
        }
        fn status(&self) -> Value {
            json!({ "activity": "idle" })
        }
    }

    /// One raw HTTP exchange over a real socket.
    fn http(addr: SocketAddr, method: &str, headers: &[(&str, String)], body: &str) -> (u16, HashMap<String, String>, Value) {
        let mut s = TcpStream::connect(addr).unwrap();
        let mut req = format!("{method} /mcp HTTP/1.1\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(body);
        s.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        let (head, body) = out.split_once("\r\n\r\n").unwrap();
        let status = head[9..12].parse().unwrap();
        let hs = head.lines().skip(1).filter_map(|l| l.split_once(": ")).map(|(k, v)| (k.to_ascii_lowercase(), v.to_owned())).collect();
        (status, hs, serde_json::from_str(body).unwrap_or(Value::Null))
    }

    #[test]
    fn full_session_over_a_real_socket() {
        let server = McpServer::new(Settings { enabled: false, port: 0, token: String::new() }, Arc::new(Fake), Box::new(|_| {}));
        server.settings().port = 0;
        // Port 0 in settings means "default"; use an ephemeral port for the test instead.
        let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        server.settings().port = port;
        server.set_enabled(true).unwrap();
        let addr = server.bound().unwrap();
        let token = server.settings().token.clone();
        let host = ("Host", format!("127.0.0.1:{}", addr.port()));
        let auth = ("Authorization", format!("Bearer {token}"));
        let ct = ("Content-Type", "application/json".to_owned());

        let (st, _, _) = http(addr, "POST", &[host.clone(), ct.clone()], r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        assert_eq!(st, 401, "no token");
        let (st, _, _) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), ("Origin", "https://evil.example".into())], "{}");
        assert_eq!(st, 403, "browser origin");
        let (st, _, _) = http(addr, "POST", &[("Host", "evil.example".into()), auth.clone(), ct.clone()], "{}");
        assert_eq!(st, 421, "rebinding host");

        let (st, hs, v) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone()], r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"claude-code","version":"2.1"}}}"#);
        assert_eq!(st, 200);
        assert_eq!(v["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(v["result"]["serverInfo"]["name"], "cerebro");
        let sid = hs["mcp-session-id"].clone();
        let session = ("Mcp-Session-Id", sid.clone());

        let (st, _, _) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), session.clone()], r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        assert_eq!(st, 202);
        let (_, _, v) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone()], r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        assert!(v["error"]["message"].as_str().unwrap().contains("Mcp-Session-Id"));
        let (_, _, v) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), session.clone()], r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let names: Vec<&str> = v["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["generate", "chat", "list_models", "load_model", "status"]);

        let (_, _, v) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), session.clone()], r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"chat","arguments":{"system":"be brief","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"why?"}],"max_tokens":99999}}}"#);
        assert_eq!(v["result"]["structuredContent"]["text"], "sys=be brief turns=3 last=why? max=8192");
        assert_eq!(v["result"]["isError"], false);
        let (_, _, v) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), session.clone()], r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"load_model","arguments":{"id":"nope"}}}"#);
        assert_eq!(v["result"]["isError"], true);
        let (_, _, v) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), session.clone()], r#"{"jsonrpc":"2.0","id":5,"method":"resources/list"}"#);
        assert_eq!(v["error"]["code"], -32601);

        let view = server.view();
        assert_eq!(view["clients"][0]["name"], "claude-code");
        assert_eq!(view["clients"][0]["calls"], 2);

        server.rotate_token().unwrap();
        let (st, _, _) = http(addr, "POST", &[host.clone(), auth.clone(), ct.clone(), session.clone()], r#"{"jsonrpc":"2.0","id":6,"method":"ping"}"#);
        assert_eq!(st, 401, "the old token stops working");
        server.set_enabled(false).unwrap();
    }
}
