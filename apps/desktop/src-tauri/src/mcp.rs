//! MCP (Model Context Protocol) client so local models can use MCP servers.
//!
//! One child process per server, newline-delimited JSON-RPC over stdio (MCP 2025-06-18).
//! Servers are configured in mcp.json in the app config dir; each handshake runs on its own
//! thread so a slow or hung server never blocks the UI.

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

// ---- Config -------------------------------------------------------------------------------------

/// The contents of mcp.json. A missing or unreadable file is an empty config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, ServerConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
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

/// One line of stdout: responses (id + result/error) go to the waiting request.
fn dispatch(line: &str, pending: &Pending) -> Option<Notification> {
    let msg: Value = serde_json::from_str(line).ok()?;
    if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
        if msg.get("result").is_some() || msg.get("error").is_some() {
            let tx = pending.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
            if let Some(tx) = tx {
                let r = match msg.get("result") {
                    Some(res) => Ok(res.clone()),
                    None => Err(msg
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("MCP error")
                        .to_owned()),
                };
                let _ = tx.send(r);
            }
        }
        return None;
    }
    if msg.get("method").and_then(|m| m.as_str()) == Some("notifications/tools/list_changed") {
        return Some(Notification::ToolsListChanged);
    }
    None
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
        self.write_line(&msg)
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
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.kill();
    }
}

// ---- Spawning -----------------------------------------------------------------------------------

fn build_command(cfg: &ServerConfig) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // npx/uvx are extensionless .cmd shims: go through cmd /C so Windows resolves them.
        let shim = PathBuf::from(&cfg.command).extension().is_none();
        let mut c = if shim {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(&cfg.command);
            c
        } else {
            Command::new(&cfg.command)
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
        let mut c = Command::new(&cfg.command);
        c.args(&cfg.args);
        for (k, v) in &cfg.env {
            c.env(k, v);
        }
        c
    }
}

fn start_server(cfg: ServerConfig) -> Arc<Server> {
    let server = Arc::new(Server::new(cfg));
    let mut cmd = build_command(&server.config);
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

/// initialize, then the initialized notification, then the (paginated) tool list.
fn handshake(server: &Arc<Server>) {
    let params = serde_json::json!({
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": { "name": "Cerebro", "version": env!("CARGO_PKG_VERSION") },
    });
    let result = server.request("initialize", params).and_then(|info| {
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

    fn tools_json(&self) -> Value {
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

    fn call_tool(&self, server_name: &str, tool: &str, arguments: Value) -> Result<Value, String> {
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

/// One entry per server: state, tool count, last stderr lines and the server's own info.
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

    /// Spawn, handshake, list and call against tests/fixtures/echo_mcp.py (skipped without Python).
    #[test]
    fn end_to_end_with_a_real_server() {
        if Command::new("python").arg("--version").output().is_err() {
            return;
        }
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/echo_mcp.py");
        let cfg = ServerConfig { command: "python".into(), args: vec![script.into()], env: HashMap::new(), disabled: false };
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
        assert_eq!(fs.command, "npx");
        assert_eq!(fs.args, vec!["-y", "@modelcontextprotocol/server-filesystem", "C:/x"]);
        assert_eq!(fs.env.get("K").unwrap(), "V");
        assert!(!fs.disabled);
        // Fields left out default to empty/false.
        let minimal: McpConfig = serde_json::from_str(r#"{"mcpServers":{"a":{"command":"node"}}}"#).unwrap();
        let a = minimal.mcp_servers.get("a").unwrap();
        assert!(a.args.is_empty() && a.env.is_empty() && !a.disabled);
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
