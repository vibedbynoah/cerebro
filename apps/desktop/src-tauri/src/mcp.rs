//! MCP (Model Context Protocol) client so local models can use MCP servers.
//!
//! Two transports (MCP 2025-06-18), chosen per entry in mcp.json: a child process with
//! newline-delimited JSON-RPC over stdio, or a Streamable HTTP session at a web address.
//! Each handshake runs on its own thread so a slow or hung server never blocks the UI.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{Manager, State};

const STDERR_LINES: usize = 200;
const STDERR_TAIL: usize = 20;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const CALL_TIMEOUT: Duration = Duration::from_secs(60);

// ---- Config -------------------------------------------------------------------------------------

/// The contents of mcp.json. A missing or unreadable file is an empty config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, ServerConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// The program to run (stdio transport); `url` is the other choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    /// A Streamable HTTP endpoint (https or http); set instead of `command`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Extra request headers for an HTTP server, e.g. an access token.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub disabled: bool,
}

// ---- Protocol -----------------------------------------------------------------------------------

type Pending = Arc<Mutex<HashMap<u64, Sender<Result<Value, String>>>>>;

/// Server-sent messages the client acts on; every other notification is ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Notification {
    ToolsListChanged,
}

/// The result of a JSON-RPC reply, or the error message it carries.
fn outcome(msg: &Value, id: u64) -> Result<Value, String> {
    if msg.get("id").and_then(|v| v.as_u64()) != Some(id) {
        return Err("the server answered a different request".to_owned());
    }
    match msg.get("result") {
        Some(res) => Ok(res.clone()),
        None => Err(msg.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()).unwrap_or("MCP error").to_owned()),
    }
}

/// One line of stdout: responses (id + result/error) go to the waiting request.
fn dispatch(line: &str, pending: &Pending) -> Option<Notification> {
    let msg: Value = serde_json::from_str(line).ok()?;
    if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
        if msg.get("result").is_some() || msg.get("error").is_some() {
            let tx = pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
            if let Some(tx) = tx {
                let _ = tx.send(outcome(&msg, id));
            }
        }
        return None;
    }
    if msg.get("method").and_then(|m| m.as_str()) == Some("notifications/tools/list_changed") {
        return Some(Notification::ToolsListChanged);
    }
    None
}

// ---- Streamable HTTP ----------------------------------------------------------------------------

/// A failed HTTP exchange. A 404 means the session is gone and initialize must run again.
enum HttpError {
    NotFound,
    Failed(String),
}

impl HttpError {
    fn into_message(self) -> String {
        match self {
            HttpError::NotFound => "the server answered 404.".to_owned(),
            HttpError::Failed(m) => m,
        }
    }
}

/// The session state of a Streamable HTTP server.
struct Http {
    client: reqwest::blocking::Client,
    session: Option<String>,
    version: Option<String>,
}

/// One response header as text.
fn header_value(resp: &reqwest::blocking::Response, name: &str) -> Option<String> {
    resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned)
}

/// The JSON-RPC reply in a finished SSE event, if that event is the one we are waiting for;
/// any other event is acted on (tool-list changes) and skipped.
fn sse_event(data: &str, id: u64, server: &Server) -> Option<Value> {
    let msg: Value = serde_json::from_str(data).ok()?;
    if msg.get("id").and_then(|v| v.as_u64()) == Some(id) {
        return Some(msg);
    }
    if msg.get("method").and_then(|m| m.as_str()) == Some("notifications/tools/list_changed") {
        server.tools_dirty.store(true, Ordering::SeqCst);
    }
    None
}

/// Reads SSE events until the reply with `id` (the spec's alternative to a JSON body).
const MAX_SSE_EVENT: usize = 16 << 20;

fn read_sse<R: BufRead>(mut reader: R, id: u64, server: &Server) -> Result<Value, String> {
    let mut data = String::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim_end_matches(['\r', '\n']);
        if n == 0 || line.is_empty() {
            if !data.is_empty() {
                if let Some(msg) = sse_event(&data, id, server) {
                    return Ok(msg);
                }
                data.clear();
            }
            if n == 0 {
                return Err("the server closed the stream before answering".to_owned());
            }
        } else if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            // One event larger than this is not an MCP reply; stop instead of growing without end.
            if data.len() > MAX_SSE_EVENT {
                return Err("the server sent an event over 16 MB".to_owned());
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ServerState {
    Starting,
    Running,
    Error(String),
    Stopped,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

pub struct Server {
    pub config: ServerConfig,
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    /// Session state of a `url` server; None means the stdio pipes above carry the protocol.
    http: Mutex<Option<Http>>,
    pending: Pending,
    next_id: AtomicU64,
    state: Mutex<ServerState>,
    tools: Mutex<Vec<Tool>>,
    tools_dirty: AtomicBool,
    stderr: Arc<Mutex<VecDeque<String>>>,
    server_info: Mutex<Value>,
}

impl Server {
    fn new(config: ServerConfig) -> Self {
        Self {
            config,
            child: Mutex::new(None),
            stdin: Mutex::new(None),
            http: Mutex::new(None),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            state: Mutex::new(ServerState::Starting),
            tools: Mutex::new(Vec::new()),
            tools_dirty: AtomicBool::new(false),
            stderr: Arc::new(Mutex::new(VecDeque::new())),
            server_info: Mutex::new(Value::Null),
        }
    }

    fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        if self.config.url.is_some() {
            return self.http_request(method, params);
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = channel();
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).insert(id, tx);
        let msg = serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        if let Err(e) = self.write_line(&msg) {
            self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
            return Err(e);
        }
        match rx.recv_timeout(REQUEST_TIMEOUT) {
            Ok(r) => r,
            Err(_) => {
                self.pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
                Err("the server did not answer in time".to_owned())
            }
        }
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        let msg = serde_json::json!({ "jsonrpc": "2.0", "method": method, "params": params });
        if self.config.url.is_some() {
            return self.http_exchange(&msg).map(|_| ()).map_err(HttpError::into_message);
        }
        self.write_line(&msg)
    }

    /// One JSON-RPC request over Streamable HTTP; keep the reply envelope to pull the result out.
    fn http_request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let mut reply = self.http_exchange(&msg);
        if matches!(reply, Err(HttpError::NotFound)) && method != "initialize" {
            // The server forgot the session; initialize once more, then repeat the call.
            self.http_forget();
            reply = self.reinitialize().and_then(|_| self.http_exchange(&msg));
        }
        match reply {
            Ok(reply) => {
                if method == "initialize" {
                    if let Some(v) = reply["result"]["protocolVersion"].as_str() {
                        self.http_remember_version(v.to_owned());
                    }
                }
                outcome(&reply, id)
            }
            Err(e) => {
                // The stderr ring is the "last error" log for HTTP servers too.
                let e = e.into_message();
                self.log(format!("{method}: {e}"));
                Err(e)
            }
        }
    }

    /// initialize + the initialized notification, without touching tools or the state machine.
    fn reinitialize(&self) -> Result<(), HttpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": "initialize", "params": initialize_params() });
        let reply = self.http_exchange(&msg)?;
        if let Some(v) = reply["result"]["protocolVersion"].as_str() {
            self.http_remember_version(v.to_owned());
        }
        outcome(&reply, id).map_err(HttpError::Failed)?;
        let notice = serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized", "params": {} });
        self.http_exchange(&notice)?;
        Ok(())
    }

    /// One POST: the JSON-RPC reply, or Null for an accepted notification. The session the server
    /// hands out is remembered; 404 has its own error so the caller can initialize again.
    fn http_exchange(&self, msg: &Value) -> Result<Value, HttpError> {
        let (client, session, version) = self.http_state()?;
        let body = serde_json::to_vec(msg).map_err(|e| HttpError::Failed(e.to_string()))?;
        let mut req = client
            .post(self.config.url.as_deref().unwrap_or_default())
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json");
        for (k, v) in &self.config.headers {
            req = req.header(k, v);
        }
        if let Some(s) = &session {
            req = req.header("Mcp-Session-Id", s);
        }
        if let Some(v) = &version {
            req = req.header("MCP-Protocol-Version", v);
        }
        let resp = req.body(body).send().map_err(|e| HttpError::Failed(format!("could not reach the server: {e}")))?;
        let status = resp.status().as_u16();
        self.http_remember_session(header_value(&resp, "mcp-session-id"));
        if status == 404 {
            return Err(HttpError::NotFound);
        }
        if !(200..300).contains(&status) {
            return Err(HttpError::Failed(format!("the server answered {status}.")));
        }
        let Some(id) = msg.get("id").and_then(|v| v.as_u64()) else {
            return Ok(Value::Null);
        };
        if header_value(&resp, "content-type").unwrap_or_default().starts_with("text/event-stream") {
            read_sse(BufReader::new(resp), id, self).map_err(HttpError::Failed)
        } else {
            let bytes = resp.bytes().map_err(|e| HttpError::Failed(format!("could not read the answer: {e}")))?;
            serde_json::from_slice(&bytes).map_err(|_| HttpError::Failed("the server sent an unexpected answer".to_owned()))
        }
    }

    /// The HTTP client plus the session headers for the next request.
    fn http_state(&self) -> Result<(reqwest::blocking::Client, Option<String>, Option<String>), HttpError> {
        let guard = self.http.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(h) => Ok((h.client.clone(), h.session.clone(), h.version.clone())),
            None => Err(HttpError::Failed("the server is not running".to_owned())),
        }
    }

    fn http_remember_session(&self, session: Option<String>) {
        let Some(session) = session else { return };
        let mut guard = self.http.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.session = Some(session);
        }
    }

    fn http_remember_version(&self, version: String) {
        let mut guard = self.http.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.version = Some(version);
        }
    }

    /// Drops the remembered session (after a 404) so initialize starts clean.
    fn http_forget(&self) {
        let mut guard = self.http.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.session = None;
            h.version = None;
        }
    }

    fn log(&self, line: String) {
        let mut buf = self.stderr.lock().unwrap_or_else(|e| e.into_inner());
        buf.push_back(line);
        while buf.len() > STDERR_LINES {
            buf.pop_front();
        }
    }

    fn write_line(&self, msg: &Value) -> Result<(), String> {
        let mut line = serde_json::to_vec(msg).map_err(|e| e.to_string())?;
        line.push(b'\n');
        let mut guard = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            Some(w) => w.write_all(&line).and_then(|_| w.flush()).map_err(|e| e.to_string()),
            None => Err("the server is not running".to_owned()),
        }
    }

    /// Re-lists tools when the server said they changed; a failure keeps the old list.
    fn refresh_tools(&self) {
        if !self.tools_dirty.swap(false, Ordering::SeqCst) {
            return;
        }
        if let Ok(tools) = list_tools(self) {
            *self.tools.lock().unwrap_or_else(|e| e.into_inner()) = tools;
        }
    }

    fn kill(&self) {
        if let Some(mut child) = self.child.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *self.stdin.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // A Streamable HTTP session ends with DELETE; best effort, the server forgets it anyway.
        let mut guard = self.http.lock().unwrap_or_else(|e| e.into_inner());
        let Some(h) = guard.as_mut() else { return };
        let Some(session) = h.session.take() else { return };
        let _ = h
            .client
            .delete(self.config.url.as_deref().unwrap_or_default())
            .header("Mcp-Session-Id", session)
            .header("MCP-Protocol-Version", h.version.take().unwrap_or_else(|| "2025-06-18".to_owned()))
            .timeout(Duration::from_secs(5))
            .send();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.kill();
    }
}

// ---- Spawning -----------------------------------------------------------------------------------

fn build_command(command: &str, cfg: &ServerConfig) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // npx/uvx are extensionless .cmd shims: go through cmd /C so Windows resolves them.
        let shim = PathBuf::from(command).extension().is_none();
        let mut c = if shim {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            Command::new(command)
        };
        c.args(&cfg.args);
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        for (k, v) in &cfg.env {
            c.env(k, v);
        }
        c
    }
    #[cfg(not(windows))]
    {
        let mut c = Command::new(command);
        c.args(&cfg.args);
        for (k, v) in &cfg.env {
            c.env(k, v);
        }
        c
    }
}

fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent(concat!("cerebro/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(CALL_TIMEOUT)
        // A few redirects (http -> https, a moved path); reqwest drops auth headers across hosts.
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
        .map_err(|_| "Could not start the network client.".to_owned())
}

fn start_server(cfg: ServerConfig) -> Arc<Server> {
    let server = Arc::new(Server::new(cfg));
    if let Some(url) = server.config.url.clone() {
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Error("the address must start with http:// or https://".to_owned());
            return server;
        }
        match http_client() {
            Ok(client) => *server.http.lock().unwrap_or_else(|e| e.into_inner()) = Some(Http { client, session: None, version: None }),
            Err(e) => *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Error(e),
        }
        return server;
    }
    let Some(command) = server.config.command.clone() else {
        *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Error("the server has no program or address".to_owned());
        return server;
    };
    let mut cmd = build_command(&command, &server.config);
    let mut child = match cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Error(format!("could not start the server: {e}"));
            return server;
        }
    };
    // All three pipes were requested, so take() yields them.
    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        let _ = child.kill();
        *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Error("no pipes to the server".to_owned());
        return server;
    };
    *server.stdin.lock().unwrap_or_else(|e| e.into_inner()) = child.stdin.take();
    *server.child.lock().unwrap_or_else(|e| e.into_inner()) = Some(child);

    let ring = server.stderr.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            let mut buf = ring.lock().unwrap_or_else(|e| e.into_inner());
            buf.push_back(line);
            while buf.len() > STDERR_LINES {
                buf.pop_front();
            }
        }
    });

    let s = server.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            if dispatch(&line, &s.pending) == Some(Notification::ToolsListChanged) {
                s.tools_dirty.store(true, Ordering::SeqCst);
            }
        }
        // stdout closed: the server is gone; fail everything still waiting.
        let mut pending = s.pending.lock().unwrap_or_else(|e| e.into_inner());
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err("server exited".to_owned()));
        }
        drop(pending);
        let mut state = s.state.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(&*state, ServerState::Stopped) {
            *state = ServerState::Error("server exited".to_owned());
        }
    });
    server
}

fn start_one(name: String, server: Arc<Server>) {
    std::thread::Builder::new()
        .name(format!("mcp-{name}"))
        .spawn(move || handshake(&server))
        .ok();
}

fn stop_server(server: &Server) {
    server.kill();
    *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Stopped;
}

fn initialize_params() -> Value {
    serde_json::json!({
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": { "name": "Cerebro", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// initialize, then the initialized notification, then the (paginated) tool list.
fn handshake(server: &Arc<Server>) {
    let result = server.request("initialize", initialize_params()).and_then(|info| {
        *server.server_info.lock().unwrap_or_else(|e| e.into_inner()) = info.get("serverInfo").cloned().unwrap_or(Value::Null);
        server.notify("notifications/initialized", serde_json::json!({}))?;
        list_tools(server)
    });
    match result {
        Ok(tools) => {
            *server.tools.lock().unwrap_or_else(|e| e.into_inner()) = tools;
            *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Running;
        }
        Err(e) => {
            *server.state.lock().unwrap_or_else(|e| e.into_inner()) = ServerState::Error(e);
        }
    }
}

fn list_tools(server: &Server) -> Result<Vec<Tool>, String> {
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let mut params = serde_json::Map::new();
        if let Some(c) = &cursor {
            params.insert("cursor".to_owned(), Value::String(c.clone()));
        }
        let res = server.request("tools/list", Value::Object(params))?;
        if let Some(arr) = res.get("tools").and_then(|t| t.as_array()) {
            for t in arr {
                tools.push(Tool {
                    name: t.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_owned(),
                    description: t.get("description").and_then(|d| d.as_str()).map(str::to_owned),
                    input_schema: t.get("inputSchema").cloned().unwrap_or(Value::Null),
                });
            }
        }
        cursor = res.get("nextCursor").and_then(|c| c.as_str()).map(str::to_owned);
        if cursor.is_none() {
            return Ok(tools);
        }
    }
}

// ---- Tauri state --------------------------------------------------------------------------------

/// Every configured server, keyed by name.
pub struct Mcp {
    inner: Mutex<HashMap<String, Arc<Server>>>,
}

impl Default for Mcp {
    fn default() -> Self {
        Self { inner: Mutex::new(HashMap::new()) }
    }
}

impl Mcp {
    /// Brings the running set in line with the config: start new/enabled servers, stop removed,
    /// disabled or changed ones.
    fn sync_servers(&self, config: &McpConfig) {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut keep: Vec<String> = Vec::new();
        for (name, server) in map.iter() {
            match config.mcp_servers.get(name) {
                Some(cfg) if !cfg.disabled && *cfg == server.config => keep.push(name.clone()),
                _ => stop_server(server),
            }
        }
        map.retain(|n, _| keep.contains(n));
        let mut to_start: Vec<(String, ServerConfig)> = Vec::new();
        for (name, cfg) in &config.mcp_servers {
            if !cfg.disabled && !map.contains_key(name) {
                to_start.push((name.clone(), cfg.clone()));
            }
        }
        for (name, cfg) in to_start {
            let server = start_server(cfg);
            map.insert(name.clone(), server.clone());
            start_one(name, server);
        }
    }

    fn restart_one(&self, name: &str, cfg: ServerConfig) {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = map.remove(name) {
            stop_server(&old);
        }
        let server = start_server(cfg);
        map.insert(name.to_owned(), server.clone());
        start_one(name.to_owned(), server);
    }

    fn status_json(&self) -> Value {
        let map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut names: Vec<&String> = map.keys().collect();
        names.sort();
        let arr = names
            .into_iter()
            .map(|name| {
                let s = map.get(name).unwrap();
                let state = s.state.lock().unwrap_or_else(|e| e.into_inner());
                let (state, error) = match &*state {
                    ServerState::Starting => ("starting", None),
                    ServerState::Running => ("running", None),
                    ServerState::Error(e) => ("error", Some(e.clone())),
                    ServerState::Stopped => ("stopped", None),
                };
                let tools = s.tools.lock().unwrap_or_else(|e| e.into_inner()).len();
                let stderr_tail: Vec<String> = {
                    let buf = s.stderr.lock().unwrap_or_else(|e| e.into_inner());
                    buf.iter().skip(buf.len().saturating_sub(STDERR_TAIL)).cloned().collect()
                };
                let server_info = s.server_info.lock().unwrap_or_else(|e| e.into_inner()).clone();
                serde_json::json!({
                    "name": name,
                    "state": state,
                    "error": error,
                    "tools": tools,
                    "stderr_tail": stderr_tail,
                    "server_info": server_info,
                })
            })
            .collect();
        Value::Array(arr)
    }

    /// The servers, sorted by name, with the map lock released (requests can take seconds).
    fn snapshot(&self) -> Vec<(String, Arc<Server>)> {
        let map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(String, Arc<Server>)> = map.iter().map(|(n, s)| (n.clone(), s.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    pub(crate) fn tools_json(&self) -> Value {
        let mut arr = Vec::new();
        for (name, s) in self.snapshot() {
            s.refresh_tools();
            let tools = s.tools.lock().unwrap_or_else(|e| e.into_inner());
            for t in tools.iter() {
                arr.push(serde_json::json!({
                    "server": name,
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }));
            }
        }
        Value::Array(arr)
    }

    pub(crate) fn call_tool(&self, server_name: &str, tool: &str, arguments: Value) -> Result<Value, String> {
        let s = self.inner.lock().unwrap_or_else(|e| e.into_inner()).get(server_name).cloned().ok_or_else(|| "unknown server".to_owned())?;
        let state = s.state.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(&*state, ServerState::Running) {
            return Err("the server is not running".to_owned());
        }
        drop(state);
        s.refresh_tools();
        let arguments = if arguments.is_null() { serde_json::json!({}) } else { arguments };
        s.request("tools/call", serde_json::json!({ "name": tool, "arguments": arguments }))
    }
}

// ---- Commands -----------------------------------------------------------------------------------

fn config_file<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("mcp.json"))
}

fn load_config<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> McpConfig {
    let Some(p) = config_file(app) else { return McpConfig::default() };
    let Ok(bytes) = std::fs::read(p) else { return McpConfig::default() };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

/// The stored mcp.json.
#[tauri::command]
pub fn mcp_config_get<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> McpConfig {
    load_config(&app)
}

/// Saves mcp.json and restarts servers so the running set matches it.
#[tauri::command]
pub fn mcp_config_save<R: tauri::Runtime>(app: tauri::AppHandle<R>, mcp: State<'_, Mcp>, config: McpConfig) -> Result<(), String> {
    let Some(p) = config_file(&app) else { return Err("the app has no config folder".to_owned()) };
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(&config).map_err(|e| e.to_string())?;
    std::fs::write(&p, bytes).map_err(|e| e.to_string())?;
    mcp.sync_servers(&config);
    Ok(())
}

/// Starts every enabled server; called once at start-up and after config changes.
#[tauri::command]
pub fn mcp_start_all<R: tauri::Runtime>(app: tauri::AppHandle<R>, mcp: State<'_, Mcp>) {
    let config = load_config(&app);
    mcp.sync_servers(&config);
}

/// One entry per server: state, tool count, last stderr/error lines and the server's own info.
#[tauri::command]
pub fn mcp_status(mcp: State<'_, Mcp>) -> Value {
    mcp.status_json()
}

/// Stops and starts one server again.
#[tauri::command]
pub fn mcp_restart<R: tauri::Runtime>(app: tauri::AppHandle<R>, mcp: State<'_, Mcp>, name: String) -> Result<(), String> {
    let config = load_config(&app);
    let cfg = config.mcp_servers.get(&name).ok_or_else(|| "unknown server".to_owned())?;
    if cfg.disabled {
        return Err("the server is disabled".to_owned());
    }
    mcp.restart_one(&name, cfg.clone());
    Ok(())
}

/// Every tool of every running server, with its input schema.
#[tauri::command]
pub fn mcp_tools(mcp: State<'_, Mcp>) -> Value {
    mcp.tools_json()
}

/// Runs one tool; returns the result object (content array, isError).
#[tauri::command]
pub fn mcp_call(mcp: State<'_, Mcp>, server: String, tool: String, arguments: Value) -> Result<Value, String> {
    mcp.call_tool(&server, &tool, arguments)
}

// ---- Tests --------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};

    /// Spawn, handshake, list and call against tests/fixtures/echo_mcp.py (skipped without Python).
    #[test]
    fn end_to_end_with_a_real_server() {
        if Command::new("python").arg("--version").output().is_err() {
            return;
        }
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/echo_mcp.py");
        let cfg = ServerConfig { command: Some("python".into()), args: vec![script.into()], env: HashMap::new(), url: None, headers: HashMap::new(), disabled: false };
        let mcp = Mcp::default();
        mcp.sync_servers(&McpConfig { mcp_servers: HashMap::from([("echo".to_owned(), cfg)]) });
        let t0 = std::time::Instant::now();
        while mcp.status_json()[0]["state"] == "starting" && t0.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(mcp.status_json()[0]["state"], "running", "{}", mcp.status_json());
        assert_eq!(mcp.tools_json()[0]["name"], "echo");
        let r = mcp.call_tool("echo", "echo", serde_json::json!({"text": "hi"})).unwrap();
        assert_eq!(r["content"][0]["text"], "hi");
    }

    #[test]
    fn config_round_trips_with_standard_key() {
        let cfg: McpConfig = serde_json::from_str(r#"{"mcpServers":{"a":{"command":"node"}}}"#).unwrap();
        assert!(serde_json::to_string(&cfg).unwrap().contains("\"mcpServers\""));
    }

    #[test]
    fn parses_sample_config() {
        let json = r#"{"mcpServers":{"fs":{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","C:/x"],"env":{"K":"V"},"disabled":false}}}"#;
        let cfg: McpConfig = serde_json::from_str(json).unwrap();
        let fs = cfg.mcp_servers.get("fs").unwrap();
        assert_eq!(fs.command.as_deref(), Some("npx"));
        assert_eq!(fs.args, vec!["-y", "@modelcontextprotocol/server-filesystem", "C:/x"]);
        assert_eq!(fs.env.get("K").unwrap(), "V");
        assert!(!fs.disabled);
        assert!(fs.url.is_none() && fs.headers.is_empty());
        // Fields left out default to empty/false.
        let minimal: McpConfig = serde_json::from_str(r#"{"mcpServers":{"a":{"command":"node"}}}"#).unwrap();
        let a = minimal.mcp_servers.get("a").unwrap();
        assert!(a.args.is_empty() && a.env.is_empty() && !a.disabled);
    }

    #[test]
    fn parses_an_http_server_with_headers() {
        let json = r#"{"mcpServers":{"docs":{"url":"https://example.com/mcp","headers":{"Authorization":"Bearer t"},"disabled":false}}}"#;
        let cfg: McpConfig = serde_json::from_str(json).unwrap();
        let docs = cfg.mcp_servers.get("docs").unwrap();
        assert_eq!(docs.url.as_deref(), Some("https://example.com/mcp"));
        assert_eq!(docs.headers.get("Authorization").unwrap(), "Bearer t");
        assert!(docs.command.is_none() && docs.args.is_empty());
        // Saved back with its address and no stray stdio keys.
        let entry = serde_json::to_value(&cfg).unwrap()["mcpServers"]["docs"].clone();
        assert_eq!(entry["url"], "https://example.com/mcp");
        assert!(entry.get("command").is_none() && entry.get("args").is_none() && entry.get("env").is_none());
    }

    /// One recorded request of the test HTTP server: method and lowercased headers.
    type Seen = (String, HashMap<String, String>);

    /// A minimal HTTP MCP server: `count` requests, one per connection, answered by `reply`.
    fn serve_http(count: usize, mut reply: impl FnMut(&str, &Value) -> crate::remote::Response + Send + 'static) -> (SocketAddr, std::thread::JoinHandle<Vec<Seen>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            let mut seen = Vec::new();
            while seen.len() < count && std::time::Instant::now() < deadline {
                let Ok((stream, _)) = listener.accept() else {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let Ok((method, headers, body)) = read_http(&stream) else { continue };
                let msg: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                seen.push((method.clone(), headers));
                let mut stream = stream;
                if crate::remote::write_response(&mut stream, &reply(&method, &msg), false).is_err() {
                    break;
                }
            }
            seen
        });
        (addr, handle)
    }

    /// Method, headers and Content-Length body of one HTTP/1.1 request.
    fn read_http(stream: &TcpStream) -> std::io::Result<(String, HashMap<String, String>, Vec<u8>)> {
        let mut r = BufReader::new(stream);
        let mut line = String::new();
        r.read_line(&mut line)?;
        let method = line.trim_end().split(' ').next().unwrap_or_default().to_owned();
        let mut headers = HashMap::new();
        loop {
            let mut l = String::new();
            r.read_line(&mut l)?;
            let l = l.trim_end();
            if l.is_empty() {
                break;
            }
            if let Some((k, v)) = l.split_once(':') {
                headers.insert(k.to_ascii_lowercase(), v.trim().to_owned());
            }
        }
        let len: usize = headers.get("content-length").map_or(0, |v| v.parse().unwrap_or(0));
        let mut body = vec![0u8; len];
        r.read_exact(&mut body)?;
        Ok((method, headers, body))
    }

    /// A JSON-RPC reply body for the test server.
    fn rpc(id: &Value, result: Value) -> Value {
        serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })
    }

    fn http_config(addr: SocketAddr) -> ServerConfig {
        ServerConfig {
            command: None,
            args: Vec::new(),
            env: HashMap::new(),
            url: Some(format!("http://{addr}/mcp")),
            headers: HashMap::from([("X-Test".to_owned(), "1".to_owned())]),
            disabled: false,
        }
    }

    /// Waits for the one server to finish its handshake and asserts it did.
    fn wait_running(mcp: &Mcp) {
        let t0 = std::time::Instant::now();
        while mcp.status_json()[0]["state"] == "starting" && t0.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(mcp.status_json()[0]["state"], "running", "{}", mcp.status_json());
    }

    #[test]
    fn end_to_end_over_streamable_http() {
        let (addr, server) = serve_http(6, |method, msg| {
            if method == "DELETE" {
                return crate::remote::Response::json(200, serde_json::json!({}));
            }
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            match msg["method"].as_str().unwrap_or("") {
                "initialize" => crate::remote::Response::json(200, rpc(&id, serde_json::json!({ "protocolVersion": "2025-06-18", "serverInfo": { "name": "web-test" } })))
                    .with("Mcp-Session-Id", "sess-1".to_owned()),
                "notifications/initialized" => crate::remote::Response { status: 202, content_type: "application/json", body: Vec::new(), extra: Vec::new() },
                // Two pages of tools, like the stdio server can have.
                "tools/list" if msg["params"]["cursor"].is_null() => {
                    crate::remote::Response::json(200, rpc(&id, serde_json::json!({ "tools": [{ "name": "ping", "inputSchema": {} }], "nextCursor": "p2" })))
                }
                "tools/list" => crate::remote::Response::json(200, rpc(&id, serde_json::json!({ "tools": [{ "name": "pong", "inputSchema": {} }] }))),
                // The call comes back as one SSE event, the transport's other reply form.
                "tools/call" => crate::remote::Response {
                    status: 200,
                    content_type: "text/event-stream",
                    body: format!("event: message\ndata: {}\n\n", rpc(&id, serde_json::json!({ "content": [{ "type": "text", "text": "hi" }] }))).into_bytes(),
                    extra: Vec::new(),
                },
                _ => crate::remote::Response::json(200, serde_json::json!({})),
            }
        });
        let mcp = Mcp::default();
        mcp.sync_servers(&McpConfig { mcp_servers: HashMap::from([("web".to_owned(), http_config(addr))]) });
        wait_running(&mcp);
        assert_eq!(mcp.tools_json().as_array().unwrap().len(), 2, "both pages listed");
        let r = mcp.call_tool("web", "ping", serde_json::json!({ "text": "hi" })).unwrap();
        assert_eq!(r["content"][0]["text"], "hi");
        // Removing the server stops it and ends the session.
        mcp.sync_servers(&McpConfig::default());
        let seen = server.join().unwrap();
        assert_eq!(seen.len(), 6, "{seen:?}");
        let header = |i: usize, k: &str| seen[i].1.get(k).map(String::as_str);
        assert_eq!(header(0, "accept"), Some("application/json, text/event-stream"));
        assert_eq!(header(0, "x-test"), Some("1"));
        assert_eq!(header(0, "mcp-session-id"), None, "no session before initialize");
        assert_eq!(header(0, "mcp-protocol-version"), None);
        assert_eq!(header(1, "mcp-session-id"), Some("sess-1"), "notification reuses the session");
        assert_eq!(header(1, "mcp-protocol-version"), Some("2025-06-18"));
        assert_eq!(header(2, "mcp-session-id"), Some("sess-1"));
        assert_eq!(header(3, "mcp-session-id"), Some("sess-1"), "paging reuses the session");
        assert_eq!(header(4, "mcp-session-id"), Some("sess-1"), "tools/call reuses the session");
        assert_eq!(seen[5].0, "DELETE");
        assert_eq!(header(5, "mcp-session-id"), Some("sess-1"));
    }

    #[test]
    fn re_initializes_once_after_a_404() {
        // The reply closure runs on the server thread, so the count is shared rather than moved.
        let inits = Arc::new(AtomicU64::new(0));
        let (addr, server) = serve_http(7, {
            let inits = Arc::clone(&inits);
            move |method, msg| {
                if method == "DELETE" {
                    return crate::remote::Response::json(200, serde_json::json!({}));
                }
                let id = msg.get("id").cloned().unwrap_or(Value::Null);
                match msg["method"].as_str().unwrap_or("") {
                    "initialize" => {
                        let n = inits.fetch_add(1, Ordering::SeqCst) + 1;
                        crate::remote::Response::json(200, rpc(&id, serde_json::json!({ "protocolVersion": "2025-06-18" }))).with("Mcp-Session-Id", format!("sess-{n}"))
                    }
                    "notifications/initialized" => crate::remote::Response { status: 202, content_type: "application/json", body: Vec::new(), extra: Vec::new() },
                    // The first tools/list hits a forgotten session; the retry runs on a new one.
                    "tools/list" if inits.load(Ordering::SeqCst) == 1 => crate::remote::Response::error(404, "session expired"),
                    "tools/list" => crate::remote::Response::json(200, rpc(&id, serde_json::json!({ "tools": [{ "name": "ok", "inputSchema": {} }] }))),
                    _ => crate::remote::Response::json(200, serde_json::json!({})),
                }
            }
        });
        let mcp = Mcp::default();
        mcp.sync_servers(&McpConfig { mcp_servers: HashMap::from([("web".to_owned(), http_config(addr))]) });
        wait_running(&mcp);
        assert_eq!(mcp.tools_json().as_array().unwrap().len(), 1);
        mcp.sync_servers(&McpConfig::default());
        let seen = server.join().unwrap();
        assert_eq!(seen.len(), 7, "{seen:?}");
        let header = |i: usize, k: &str| seen[i].1.get(k).map(String::as_str);
        assert_eq!(inits.load(Ordering::SeqCst), 2, "initialized exactly once more");
        assert_eq!(header(2, "mcp-session-id"), Some("sess-1"), "the request that 404ed");
        assert_eq!(header(3, "mcp-session-id"), None, "re-initialize starts without the dead session");
        assert_eq!(header(5, "mcp-session-id"), Some("sess-2"), "the retry runs under the new session");
        assert_eq!(seen[6].0, "DELETE");
        assert_eq!(header(6, "mcp-session-id"), Some("sess-2"));
    }

    #[test]
    fn routes_response_to_pending() {
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = channel();
        pending.lock().unwrap().insert(7, tx);
        let line = r#"{"jsonrpc":"2.0","id":7,"result":{"ok":true}}"#;
        assert_eq!(dispatch(line, &pending), None);
        let resp = rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        assert_eq!(resp, serde_json::json!({"ok": true}));
        assert!(pending.lock().unwrap().is_empty());
    }

    #[test]
    fn flags_tools_list_changed() {
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let line = r#"{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}"#;
        assert_eq!(dispatch(line, &pending), Some(Notification::ToolsListChanged));
        // Other notifications are ignored.
        let other = r#"{"jsonrpc":"2.0","method":"notifications/message","params":{"x":1}}"#;
        assert_eq!(dispatch(other, &pending), None);
    }
}
