//! Local API: the loaded model behind the two request formats most apps already speak, on one port.
//!
//! - OpenAI: `GET /v1/models`, `POST /v1/chat/completions` (streamed or whole, tools, reasoning as
//!   `reasoning_content`, `stop`, usage).
//! - Anthropic: `GET /v1/models`, `POST /v1/messages` (streamed or whole, `tool_use` / `tool_result`
//!   blocks, `thinking`, `stop_sequences`, usage).
//!
//! Tools in a request belong to the calling app: when the model calls one, the call is returned in
//! that format (`tool_calls` / `tool_use`) and the app runs it, as with a hosted API. Cerebro's own
//! tools are not offered here.
//!
//! Off by default. Every protection is a setting: 127.0.0.1 only unless "Allow the local network"
//! is on (then private addresses only, never the internet); an access key (Bearer or x-api-key)
//! unless "Require the key" is off; a browser Origin or a Host name other than this machine's is
//! refused (DNS rebinding).

use std::io::{BufReader, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use subtle::ConstantTimeEq;

use crate::remote::{now_ms, random_bytes, read_request_limited, write_response, HttpError, Request, Response};
use crate::toolcall::{ToolCall, ToolSpec};

pub const DEFAULT_PORT: u16 = 47_812;
/// A whole conversation arrives in one request (agents send long contexts).
const MAX_BODY: usize = 32 << 20;
const MAX_CONNECTIONS: usize = 16;
const MAX_TOKENS_CAP: usize = 131_072;
const DEFAULT_MAX_TOKENS: usize = 8_192;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub enabled: bool,
    /// Fixed once chosen so app configs keep working; 0 = the default.
    pub port: u16,
    pub key: String,
    pub require_key: bool,
    /// Listen on the local network too (private addresses only).
    pub allow_lan: bool,
    /// A request naming another model in the library loads it.
    pub allow_model_switch: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { enabled: false, port: 0, key: String::new(), require_key: true, allow_lan: false, allow_model_switch: true }
    }
}

// ---- What the endpoints reach (commands.rs over the engine; a fake in the tests) ---------------

#[derive(Clone, Debug)]
pub struct ApiModel {
    pub id: String,
    pub name: String,
    pub loaded: bool,
}

/// One message of the conversation, already separated from either wire format.
#[derive(Clone, Debug, PartialEq)]
pub enum Msg {
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    /// A tool's result, as the app ran it.
    Tool(String),
}

#[derive(Clone, Debug)]
pub struct ApiRequest {
    /// A library model to load first (resolved by the server), or None for the loaded model.
    pub load: Option<String>,
    pub system: String,
    pub messages: Vec<Msg>,
    pub tools: Vec<ToolSpec>,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub think: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ApiEvent {
    Text(String),
    Thinking(String),
    Call { name: String, arguments: Value },
}

#[derive(Clone, Debug, Default)]
pub struct ApiDone {
    pub model: String,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    /// Reached max_tokens.
    pub length: bool,
}

#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        ApiError { status: 400, message: message.into() }
    }
}

pub trait ApiBackend: Send + Sync + 'static {
    fn models(&self) -> Vec<ApiModel>;
    /// One reply. `on` gets each piece as it is written and returns false to stop (the app left,
    /// or a stop sequence was reached). Blocking.
    fn chat(&self, req: ApiRequest, on: &mut dyn FnMut(ApiEvent) -> bool) -> Result<ApiDone, ApiError>;
}

// ---- The two wire formats ------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    OpenAi,
    Anthropic,
}

/// What a request asked for besides the conversation.
#[derive(Clone, Debug, Default)]
pub struct Opts {
    pub stream: bool,
    pub include_usage: bool,
    pub stops: Vec<String>,
    /// The model name the app asked for (echoed back), if any.
    pub model: Option<String>,
    /// Whether the client keeps the connection open (it did not send `Connection: close`).
    pub keep: bool,
}

fn text_of_openai_content(c: &Value) -> Result<String, ApiError> {
    match c {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p["type"].as_str() {
                    Some("text") | Some("input_text") => out.push_str(p["text"].as_str().unwrap_or("")),
                    Some(t @ ("image_url" | "input_image" | "input_audio" | "file")) => return Err(ApiError::bad(format!("This server does not accept {t} content yet; send text."))),
                    _ => return Err(ApiError::bad("Unknown content part.")),
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad("Message content must be a string or a list of parts.")),
    }
}

fn spec(name: &str, description: &str, schema: &Value) -> Result<ToolSpec, ApiError> {
    if name.is_empty() || name.len() > 128 {
        return Err(ApiError::bad("Every tool needs a name (at most 128 characters)."));
    }
    let schema = if schema.is_object() { schema.clone() } else { json!({ "type": "object", "properties": {} }) };
    Ok(ToolSpec { server: String::new(), name: name.to_owned(), description: description.to_owned(), input_schema: schema })
}

fn stops_of(v: &Value) -> Result<Vec<String>, ApiError> {
    let list: Vec<String> = match v {
        Value::Null => Vec::new(),
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().map(|x| x.as_str().map(str::to_owned).ok_or_else(|| ApiError::bad("Stop sequences must be strings."))).collect::<Result<_, _>>()?,
        _ => return Err(ApiError::bad("Stop sequences must be strings.")),
    };
    if list.len() > 16 || list.iter().any(|s| s.is_empty() || s.len() > 256) {
        return Err(ApiError::bad("At most 16 stop sequences, each 1 to 256 characters."));
    }
    Ok(list)
}

fn max_tokens(v: &Value, default: Option<usize>) -> Result<usize, ApiError> {
    match v.as_u64() {
        Some(n) if n >= 1 => Ok((n as usize).min(MAX_TOKENS_CAP)),
        Some(_) => Err(ApiError::bad("max_tokens must be at least 1.")),
        None if v.is_null() => default.ok_or_else(|| ApiError::bad("max_tokens is required.")),
        None => Err(ApiError::bad("max_tokens must be a whole number.")),
    }
}

fn sampling(v: &Value, think: bool) -> Result<(f32, f32), ApiError> {
    let t = match &v["temperature"] {
        Value::Null => if think { 0.6 } else { 0.7 },
        x => x.as_f64().filter(|t| (0.0..=2.0).contains(t)).ok_or_else(|| ApiError::bad("temperature must be between 0 and 2."))? as f32,
    };
    let p = match &v["top_p"] {
        Value::Null => if think { 0.95 } else { 0.8 },
        x => x.as_f64().filter(|p| *p > 0.0 && *p <= 1.0).ok_or_else(|| ApiError::bad("top_p must be above 0 and at most 1."))? as f32,
    };
    Ok((t, p))
}

/// An OpenAI chat-completions body.
pub fn parse_openai(v: &Value) -> Result<(ApiRequest, Opts), ApiError> {
    if v["n"].as_u64().is_some_and(|n| n != 1) {
        return Err(ApiError::bad("Only n = 1 is supported."));
    }
    let msgs = v["messages"].as_array().filter(|m| !m.is_empty()).ok_or_else(|| ApiError::bad("messages must be a non-empty list."))?;
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for m in msgs {
        match m["role"].as_str() {
            Some("system" | "developer") => system.push(text_of_openai_content(&m["content"])?),
            Some("user") => messages.push(Msg::User(text_of_openai_content(&m["content"])?)),
            Some("assistant") => {
                let mut calls = Vec::new();
                for c in m["tool_calls"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                    let name = c["function"]["name"].as_str().ok_or_else(|| ApiError::bad("A tool call has no function name."))?;
                    let arguments = match &c["function"]["arguments"] {
                        Value::String(s) if s.trim().is_empty() => json!({}),
                        Value::String(s) => serde_json::from_str(s).map_err(|_| ApiError::bad("Tool call arguments must be JSON."))?,
                        Value::Object(_) => c["function"]["arguments"].clone(),
                        _ => json!({}),
                    };
                    calls.push(ToolCall { name: name.to_owned(), arguments });
                }
                messages.push(Msg::Assistant { text: text_of_openai_content(&m["content"])?, calls });
            }
            Some("tool" | "function") => messages.push(Msg::Tool(text_of_openai_content(&m["content"])?)),
            _ => return Err(ApiError::bad("Each message needs a role: system, user, assistant or tool.")),
        }
    }
    let none = v["tool_choice"].as_str() == Some("none");
    let mut tools = Vec::new();
    if !none {
        for t in v["tools"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            if t["type"].as_str().unwrap_or("function") != "function" {
                return Err(ApiError::bad("Only function tools are supported."));
            }
            let f = &t["function"];
            tools.push(spec(f["name"].as_str().unwrap_or(""), f["description"].as_str().unwrap_or(""), &f["parameters"])?);
        }
    }
    let effort = v["reasoning_effort"].as_str();
    let think = matches!(effort, Some("low" | "medium" | "high" | "xhigh")) || v["chat_template_kwargs"]["enable_thinking"].as_bool() == Some(true);
    let (temperature, top_p) = sampling(v, think)?;
    let mt = if v["max_completion_tokens"].is_null() { &v["max_tokens"] } else { &v["max_completion_tokens"] };
    let req = ApiRequest { load: None, system: system.join("\n\n"), messages, tools, temperature, top_p, max_tokens: max_tokens(mt, Some(DEFAULT_MAX_TOKENS))?, think };
    let opts = Opts {
        stream: v["stream"].as_bool().unwrap_or(false),
        include_usage: v["stream_options"]["include_usage"].as_bool().unwrap_or(false),
        stops: stops_of(&v["stop"])?,
        model: v["model"].as_str().map(str::to_owned),
        keep: true,
    };
    Ok((req, opts))
}

fn text_of_anthropic_blocks(c: &Value) -> Result<String, ApiError> {
    match c {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = Vec::new();
            for b in blocks {
                match b["type"].as_str() {
                    Some("text") => out.push(b["text"].as_str().unwrap_or("").to_owned()),
                    Some(t) => return Err(ApiError::bad(format!("This server does not accept {t} content here yet; send text."))),
                    None => return Err(ApiError::bad("A content block has no type.")),
                }
            }
            Ok(out.join("\n"))
        }
        Value::Null => Ok(String::new()),
        _ => Err(ApiError::bad("Content must be a string or a list of blocks.")),
    }
}

/// An Anthropic messages body.
pub fn parse_anthropic(v: &Value) -> Result<(ApiRequest, Opts), ApiError> {
    let msgs = v["messages"].as_array().filter(|m| !m.is_empty()).ok_or_else(|| ApiError::bad("messages must be a non-empty list."))?;
    let system = text_of_anthropic_blocks(&v["system"])?;
    let mut messages = Vec::new();
    for m in msgs {
        let role = m["role"].as_str();
        let blocks: Vec<Value> = match &m["content"] {
            Value::String(s) => vec![json!({ "type": "text", "text": s })],
            Value::Array(a) => a.clone(),
            _ => return Err(ApiError::bad("Content must be a string or a list of blocks.")),
        };
        match role {
            Some("user") => {
                let mut text = Vec::new();
                for b in &blocks {
                    match b["type"].as_str() {
                        Some("text") => text.push(b["text"].as_str().unwrap_or("").to_owned()),
                        Some("tool_result") => {
                            let body = text_of_anthropic_blocks(&b["content"])?;
                            messages.push(Msg::Tool(if b["is_error"].as_bool() == Some(true) { format!("Error: {body}") } else { body }));
                        }
                        Some(t) => return Err(ApiError::bad(format!("This server does not accept {t} content yet; send text."))),
                        None => return Err(ApiError::bad("A content block has no type.")),
                    }
                }
                if !text.is_empty() {
                    messages.push(Msg::User(text.join("\n")));
                }
            }
            Some("assistant") => {
                let (mut text, mut calls) = (Vec::new(), Vec::new());
                for b in &blocks {
                    match b["type"].as_str() {
                        Some("text") => text.push(b["text"].as_str().unwrap_or("").to_owned()),
                        Some("tool_use") => calls.push(ToolCall {
                            name: b["name"].as_str().ok_or_else(|| ApiError::bad("A tool_use block has no name."))?.to_owned(),
                            arguments: if b["input"].is_object() { b["input"].clone() } else { json!({}) },
                        }),
                        // Earlier reasoning is not replayed.
                        Some("thinking" | "redacted_thinking") => {}
                        Some(t) => return Err(ApiError::bad(format!("Unknown assistant block {t}."))),
                        None => return Err(ApiError::bad("A content block has no type.")),
                    }
                }
                messages.push(Msg::Assistant { text: text.join("\n"), calls });
            }
            _ => return Err(ApiError::bad("Each message needs a role: user or assistant.")),
        }
    }
    let none = v["tool_choice"]["type"].as_str() == Some("none");
    let mut tools = Vec::new();
    if !none {
        for t in v["tools"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            // Hosted tools (web search, code execution...) have a type and no schema: not offered.
            if t["input_schema"].is_null() {
                continue;
            }
            tools.push(spec(t["name"].as_str().unwrap_or(""), t["description"].as_str().unwrap_or(""), &t["input_schema"])?);
        }
    }
    let think = matches!(v["thinking"]["type"].as_str(), Some("enabled" | "adaptive"));
    let (temperature, top_p) = sampling(v, think)?;
    let req = ApiRequest { load: None, system, messages, tools, temperature, top_p, max_tokens: max_tokens(&v["max_tokens"], None)?, think };
    let opts = Opts { stream: v["stream"].as_bool().unwrap_or(false), include_usage: true, stops: stops_of(&v["stop_sequences"])?, model: v["model"].as_str().map(str::to_owned), keep: true };
    Ok((req, opts))
}

/// Holds back text that could be the start of a stop sequence; reports the sequence once reached.
struct Stops {
    seqs: Vec<String>,
    held: String,
    hit: Option<String>,
}

impl Stops {
    fn new(seqs: Vec<String>) -> Self {
        Stops { seqs, held: String::new(), hit: None }
    }

    /// The text that can be shown now.
    fn feed(&mut self, t: &str) -> String {
        if self.hit.is_some() {
            return String::new();
        }
        if self.seqs.is_empty() {
            return t.to_owned();
        }
        self.held.push_str(t);
        if let Some((i, s)) = self.seqs.iter().filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s.clone()))).min_by_key(|(i, _)| *i) {
            let out = self.held[..i].to_owned();
            self.held.clear();
            self.hit = Some(s);
            return out;
        }
        // Keep the longest tail that is a prefix of some stop sequence.
        let keep = self
            .seqs
            .iter()
            .flat_map(|s| (1..s.len()).filter(|&k| s.is_char_boundary(k)).map(move |k| &s[..k]))
            .filter(|p| self.held.ends_with(p))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_owned();
        self.held.drain(..cut);
        out
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

fn rand_id(prefix: &str) -> String {
    let b = random_bytes::<12>().unwrap_or([7; 12]);
    format!("{prefix}{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>())
}

fn error_body(wire: Wire, status: u16, message: &str) -> Value {
    match wire {
        Wire::OpenAi => {
            let kind = match status {
                401 => "authentication_error",
                404 => "not_found_error",
                503 => "server_error",
                _ => "invalid_request_error",
            };
            json!({ "error": { "message": message, "type": kind, "code": Value::Null } })
        }
        Wire::Anthropic => {
            let kind = match status {
                401 => "authentication_error",
                403 => "permission_error",
                404 => "not_found_error",
                413 => "request_too_large",
                503 => "overloaded_error",
                500 => "api_error",
                _ => "invalid_request_error",
            };
            json!({ "type": "error", "error": { "type": kind, "message": message } })
        }
    }
}

/// Writes server-sent events on a connection that closes when the reply ends.
struct Sse<'a> {
    w: &'a mut dyn Write,
    ok: bool,
}

impl Sse<'_> {
    fn head(w: &mut dyn Write) -> std::io::Result<()> {
        w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n")?;
        w.flush()
    }
    fn data(&mut self, v: &Value) {
        if self.ok {
            self.ok = write!(self.w, "data: {v}\n\n").and_then(|_| self.w.flush()).is_ok();
        }
    }
    fn event(&mut self, v: &Value) {
        if self.ok {
            let name = v["type"].as_str().unwrap_or("message");
            self.ok = write!(self.w, "event: {name}\ndata: {v}\n\n").and_then(|_| self.w.flush()).is_ok();
        }
    }
    fn raw(&mut self, s: &str) {
        if self.ok {
            self.ok = self.w.write_all(s.as_bytes()).and_then(|_| self.w.flush()).is_ok();
        }
    }
}

/// Runs a reply and writes it in the app's format. Returns whether the connection may be reused.
fn respond(backend: &dyn ApiBackend, wire: Wire, req: ApiRequest, opts: Opts, w: &mut dyn Write) -> bool {
    let id = rand_id(if wire == Wire::OpenAi { "chatcmpl-" } else { "msg_" });
    let created = now_ms() / 1000;
    let mut stops = Stops::new(opts.stops.clone());
    let requested = opts.model.clone();
    if !opts.stream {
        let (mut text, mut thinking, mut calls) = (String::new(), String::new(), Vec::<(String, String, Value)>::new());
        let r = backend.chat(req, &mut |e| {
            match e {
                ApiEvent::Text(t) => text.push_str(&stops.feed(&t)),
                ApiEvent::Thinking(t) => thinking.push_str(&t),
                ApiEvent::Call { name, arguments } => calls.push((rand_id(if wire == Wire::OpenAi { "call_" } else { "toolu_" }), name, arguments)),
            }
            stops.hit.is_none()
        });
        let done = match r {
            Ok(d) => d,
            Err(e) => return write_response(w, &Response::json(e.status, error_body(wire, e.status, &e.message)), opts.keep).is_ok(),
        };
        if stops.hit.is_none() {
            text.push_str(&stops.finish());
        }
        let model = requested.unwrap_or_else(|| done.model.clone());
        let body = match wire {
            Wire::OpenAi => {
                let finish = if !calls.is_empty() { "tool_calls" } else if done.length && stops.hit.is_none() { "length" } else { "stop" };
                let mut message = json!({ "role": "assistant", "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(text) } });
                if !thinking.is_empty() {
                    message["reasoning_content"] = Value::String(thinking);
                }
                if !calls.is_empty() {
                    message["tool_calls"] = calls.iter().map(|(id, name, args)| json!({ "id": id, "type": "function", "function": { "name": name, "arguments": args.to_string() } })).collect();
                }
                json!({
                    "id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
                    "usage": { "prompt_tokens": done.prompt_tokens, "completion_tokens": done.output_tokens, "total_tokens": done.prompt_tokens + done.output_tokens },
                })
            }
            Wire::Anthropic => {
                let mut content = Vec::new();
                if !thinking.is_empty() {
                    content.push(json!({ "type": "thinking", "thinking": thinking, "signature": "" }));
                }
                if !text.is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
                for (id, name, args) in &calls {
                    content.push(json!({ "type": "tool_use", "id": id, "name": name, "input": args }));
                }
                let (reason, seq) = if !calls.is_empty() {
                    ("tool_use", Value::Null)
                } else if let Some(s) = &stops.hit {
                    ("stop_sequence", Value::String(s.clone()))
                } else if done.length {
                    ("max_tokens", Value::Null)
                } else {
                    ("end_turn", Value::Null)
                };
                json!({
                    "id": id, "type": "message", "role": "assistant", "model": model, "content": content,
                    "stop_reason": reason, "stop_sequence": seq,
                    "usage": { "input_tokens": done.prompt_tokens, "output_tokens": done.output_tokens },
                })
            }
        };
        return write_response(w, &Response::json(200, body), opts.keep).is_ok();
    }

    // Streamed. Errors found before the first byte are ordinary responses; later ones are an event.
    if Sse::head(w).is_err() {
        return false;
    }
    let mut sse = Sse { w, ok: true };
    let model = requested.clone().unwrap_or_default();
    match wire {
        Wire::OpenAi => {
            let chunk = |delta: Value, finish: Value| json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
            sse.data(&chunk(json!({ "role": "assistant", "content": "" }), Value::Null));
            let mut ncalls = 0usize;
            let r = backend.chat(req, &mut |e| {
                match e {
                    ApiEvent::Text(t) => {
                        let t = stops.feed(&t);
                        if !t.is_empty() {
                            sse.data(&chunk(json!({ "content": t }), Value::Null));
                        }
                    }
                    ApiEvent::Thinking(t) => sse.data(&chunk(json!({ "reasoning_content": t }), Value::Null)),
                    ApiEvent::Call { name, arguments } => {
                        sse.data(&chunk(json!({ "tool_calls": [{ "index": ncalls, "id": rand_id("call_"), "type": "function", "function": { "name": name, "arguments": arguments.to_string() } }] }), Value::Null));
                        ncalls += 1;
                    }
                }
                sse.ok && stops.hit.is_none()
            });
            match r {
                Ok(done) => {
                    if stops.hit.is_none() {
                        let rest = stops.finish();
                        if !rest.is_empty() {
                            sse.data(&chunk(json!({ "content": rest }), Value::Null));
                        }
                    }
                    let finish = if ncalls > 0 { "tool_calls" } else if done.length && stops.hit.is_none() { "length" } else { "stop" };
                    sse.data(&chunk(json!({}), Value::String(finish.into())));
                    if opts.include_usage {
                        sse.data(&json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
                            "usage": { "prompt_tokens": done.prompt_tokens, "completion_tokens": done.output_tokens, "total_tokens": done.prompt_tokens + done.output_tokens } }));
                    }
                }
                Err(e) => sse.data(&error_body(wire, e.status, &e.message)),
            }
            sse.raw("data: [DONE]\n\n");
        }
        Wire::Anthropic => {
            sse.event(&json!({ "type": "message_start", "message": { "id": id, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null, "usage": { "input_tokens": 0, "output_tokens": 0 } } }));
            // One open block at a time: thinking, then text, then each tool_use.
            let mut open: Option<&'static str> = None;
            let mut index = 0usize;
            let mut ncalls = 0usize;
            let close = |sse: &mut Sse, open: &mut Option<&'static str>, index: &mut usize| {
                if open.take().is_some() {
                    sse.event(&json!({ "type": "content_block_stop", "index": *index }));
                    *index += 1;
                }
            };
            let r = backend.chat(req, &mut |e| {
                match e {
                    ApiEvent::Thinking(t) => {
                        if open != Some("thinking") {
                            close(&mut sse, &mut open, &mut index);
                            sse.event(&json!({ "type": "content_block_start", "index": index, "content_block": { "type": "thinking", "thinking": "", "signature": "" } }));
                            open = Some("thinking");
                        }
                        sse.event(&json!({ "type": "content_block_delta", "index": index, "delta": { "type": "thinking_delta", "thinking": t } }));
                    }
                    ApiEvent::Text(t) => {
                        let t = stops.feed(&t);
                        if !t.is_empty() {
                            if open != Some("text") {
                                close(&mut sse, &mut open, &mut index);
                                sse.event(&json!({ "type": "content_block_start", "index": index, "content_block": { "type": "text", "text": "" } }));
                                open = Some("text");
                            }
                            sse.event(&json!({ "type": "content_block_delta", "index": index, "delta": { "type": "text_delta", "text": t } }));
                        }
                    }
                    ApiEvent::Call { name, arguments } => {
                        close(&mut sse, &mut open, &mut index);
                        sse.event(&json!({ "type": "content_block_start", "index": index, "content_block": { "type": "tool_use", "id": rand_id("toolu_"), "name": name, "input": {} } }));
                        sse.event(&json!({ "type": "content_block_delta", "index": index, "delta": { "type": "input_json_delta", "partial_json": arguments.to_string() } }));
                        open = Some("tool_use");
                        close(&mut sse, &mut open, &mut index);
                        ncalls += 1;
                    }
                }
                sse.ok && stops.hit.is_none()
            });
            match r {
                Ok(done) => {
                    if stops.hit.is_none() {
                        let rest = stops.finish();
                        if !rest.is_empty() {
                            if open != Some("text") {
                                close(&mut sse, &mut open, &mut index);
                                sse.event(&json!({ "type": "content_block_start", "index": index, "content_block": { "type": "text", "text": "" } }));
                                open = Some("text");
                            }
                            sse.event(&json!({ "type": "content_block_delta", "index": index, "delta": { "type": "text_delta", "text": rest } }));
                        }
                    }
                    close(&mut sse, &mut open, &mut index);
                    let (reason, seq) = if ncalls > 0 {
                        ("tool_use", Value::Null)
                    } else if let Some(s) = &stops.hit {
                        ("stop_sequence", Value::String(s.clone()))
                    } else if done.length {
                        ("max_tokens", Value::Null)
                    } else {
                        ("end_turn", Value::Null)
                    };
                    sse.event(&json!({ "type": "message_delta", "delta": { "stop_reason": reason, "stop_sequence": seq }, "usage": { "input_tokens": done.prompt_tokens, "output_tokens": done.output_tokens } }));
                    sse.event(&json!({ "type": "message_stop" }));
                }
                Err(e) => {
                    close(&mut sse, &mut open, &mut index);
                    sse.event(&error_body(wire, e.status, &e.message));
                }
            }
        }
    }
    false
}

// ---- The server ------------------------------------------------------------------------------------

pub struct LocalApi {
    settings: Mutex<Settings>,
    backend: Arc<dyn ApiBackend>,
    save: Box<dyn Fn(&Settings) + Send + Sync>,
    running: Mutex<Option<(Arc<AtomicBool>, SocketAddr)>>,
    error: Mutex<Option<String>>,
    connections: Arc<AtomicUsize>,
    requests: AtomicU64,
    last_request_at: AtomicU64,
}

fn new_key() -> Result<String, String> {
    let b = random_bytes::<24>()?;
    Ok(format!("sk-cerebro-{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>()))
}

/// Peers that may connect: this computer always; private networks when the LAN is allowed.
fn peer_ok(ip: IpAddr, lan: bool) -> bool {
    if ip.is_loopback() {
        return true;
    }
    if !lan {
        return false;
    }
    match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local() || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64),
        IpAddr::V6(v6) => v6.to_ipv4_mapped().is_some_and(|v4| peer_ok(IpAddr::V4(v4), lan)) || (v6.segments()[0] & 0xfe00) == 0xfc00 || (v6.segments()[0] & 0xffc0) == 0xfe80,
    }
}

impl LocalApi {
    pub fn new(mut settings: Settings, backend: Arc<dyn ApiBackend>, save: Box<dyn Fn(&Settings) + Send + Sync>) -> Arc<Self> {
        if settings.key.len() < 32 {
            settings.key = new_key().unwrap_or_default();
        }
        if settings.port == 0 {
            settings.port = DEFAULT_PORT;
        }
        save(&settings);
        Arc::new(LocalApi {
            settings: Mutex::new(settings),
            backend,
            save,
            running: Mutex::new(None),
            error: Mutex::new(None),
            connections: Arc::new(AtomicUsize::new(0)),
            requests: AtomicU64::new(0),
            last_request_at: AtomicU64::new(0),
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
        let port = self.bound().map_or(s.port, |a| a.port());
        let running = self.bound().is_some();
        let lan: Vec<String> = if running && s.allow_lan { crate::remote::local_addresses(false).iter().map(|ip| format!("http://{ip}:{port}/v1")).collect() } else { Vec::new() };
        json!({
            "enabled": s.enabled,
            "running": running,
            "url": running.then(|| format!("http://127.0.0.1:{port}/v1")),
            "lanUrls": lan,
            "port": port,
            "key": s.key,
            "requireKey": s.require_key,
            "allowLan": s.allow_lan,
            "allowModelSwitch": s.allow_model_switch,
            "requests": self.requests.load(Ordering::Relaxed),
            "lastRequestAt": match self.last_request_at.load(Ordering::Relaxed) { 0 => Value::Null, t => json!(t) },
            "error": self.error.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        })
    }

    pub fn resume(self: &Arc<Self>) {
        if self.settings().enabled {
            if let Err(e) = self.start() {
                *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
            }
        }
    }

    fn restart(self: &Arc<Self>) -> Result<(), String> {
        self.stop();
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if self.settings().enabled {
            if let Err(e) = self.start() {
                *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e.clone());
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn set_enabled(self: &Arc<Self>, on: bool) -> Result<(), String> {
        self.settings().enabled = on;
        self.persist();
        self.restart()
    }

    /// The three switches; changing the network one restarts the listener.
    pub fn set_options(self: &Arc<Self>, require_key: bool, allow_lan: bool, allow_model_switch: bool) -> Result<(), String> {
        let rebind = {
            let mut s = self.settings();
            let rebind = s.allow_lan != allow_lan;
            s.require_key = require_key;
            s.allow_lan = allow_lan;
            s.allow_model_switch = allow_model_switch;
            rebind
        };
        self.persist();
        if rebind {
            self.restart()?;
        }
        Ok(())
    }

    pub fn rotate_key(&self) -> Result<(), String> {
        self.settings().key = new_key()?;
        self.persist();
        Ok(())
    }

    pub fn stop(&self) {
        if let Some((flag, _)) = self.running.lock().unwrap_or_else(|e| e.into_inner()).take() {
            flag.store(true, Ordering::SeqCst);
        }
    }

    fn start(self: &Arc<Self>) -> Result<(), String> {
        let (port, lan) = {
            let s = self.settings();
            (s.port, s.allow_lan)
        };
        let ip = if lan { Ipv4Addr::UNSPECIFIED } else { Ipv4Addr::LOCALHOST };
        let listener = TcpListener::bind((ip, port)).or_else(|_| TcpListener::bind((ip, 0))).map_err(|e| format!("Could not listen on port {port} ({e})."))?;
        let addr = listener.local_addr().map_err(|_| "Could not start the API.".to_owned())?;
        if addr.port() != port {
            self.settings().port = addr.port();
            self.persist();
        }
        listener.set_nonblocking(true).map_err(|_| "Could not start the API.".to_owned())?;
        let stop = Arc::new(AtomicBool::new(false));
        let (me, st) = (Arc::clone(self), Arc::clone(&stop));
        std::thread::Builder::new().name("cerebro-api".into()).spawn(move || me.accept_loop(listener, st)).map_err(|_| "Could not start the API.".to_owned())?;
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = Some((stop, addr));
        Ok(())
    }

    fn accept_loop(self: Arc<Self>, listener: TcpListener, stop: Arc<AtomicBool>) {
        while !stop.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, peer)) => {
                    let lan = self.settings().allow_lan;
                    if !peer_ok(peer.ip(), lan) {
                        continue;
                    }
                    if self.connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                        self.connections.fetch_sub(1, Ordering::SeqCst);
                        continue;
                    }
                    let (me, st) = (Arc::clone(&self), Arc::clone(&stop));
                    let ok = std::thread::Builder::new().name("cerebro-api-conn".into()).spawn(move || {
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
        let Ok(closer) = stream.try_clone() else { return };
        self.serve_requests(stream, stop);
        // Close gracefully: end our side, then read what the client still sends until it closes
        // (at most a second). Dropping a socket with unread data makes Windows reset the
        // connection, and a client could then lose a response it had already received.
        let _ = closer.shutdown(std::net::Shutdown::Write);
        let _ = closer.set_read_timeout(Some(Duration::from_secs(1)));
        let mut sink = [0u8; 4096];
        let mut c = &closer;
        while matches!(std::io::Read::read(&mut c, &mut sink), Ok(n) if n > 0) {}
    }

    fn serve_requests(&self, stream: TcpStream, stop: &AtomicBool) {
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(300)));
        let _ = stream.set_nodelay(true);
        let Ok(mut write) = stream.try_clone() else { return };
        let mut io = BufReader::new(stream);
        for _ in 0..10_000 {
            let req = match read_request_limited(&mut io, MAX_BODY) {
                Ok(r) => r,
                Err(HttpError::Closed) => return,
                Err(HttpError::Bad(code)) => {
                    let _ = write_response(&mut write, &Response::json(code, error_body(Wire::OpenAi, code, "Bad request.")), false);
                    return;
                }
            };
            if stop.load(Ordering::SeqCst) {
                return;
            }
            let keep = req.keep_alive();
            if !self.handle(&req, &mut write) || !keep {
                return;
            }
        }
    }

    /// One request. Returns whether the connection may be reused.
    pub(crate) fn handle(&self, req: &Request, w: &mut dyn Write) -> bool {
        let wire = if req.path.starts_with("/v1/messages") || req.header("anthropic-version").is_some() { Wire::Anthropic } else { Wire::OpenAi };
        // The response says whether this connection stays open, as the client asked.
        let keep = req.keep_alive();
        let fail = |w: &mut dyn Write, status: u16, msg: &str| write_response(w, &Response::json(status, error_body(wire, status, msg)), keep).is_ok();
        let s = self.settings().clone();
        let port = self.bound().map_or(s.port, |a| a.port());
        // Host: this machine by loopback name, or (LAN allowed) an IP literal. A DNS name pointing
        // here is how a web page would reach this port, so names other than localhost are refused.
        let host_ok = req.header("host").is_some_and(|h| {
            let (name, p) = h.rsplit_once(':').unwrap_or((h, ""));
            let port_ok = p == port.to_string();
            port_ok && (name == "127.0.0.1" || name == "localhost" || name == "[::1]" || (s.allow_lan && name.parse::<Ipv4Addr>().is_ok()))
        });
        if !host_ok {
            return fail(w, 421, "Wrong host.");
        }
        if let Some(origin) = req.header("origin") {
            let local = origin == "null" || origin.starts_with("http://127.0.0.1") || origin.starts_with("http://localhost");
            if !local {
                return fail(w, 403, "Requests from web pages are not allowed.");
            }
        }
        if s.require_key {
            let given = req.header("x-api-key").or_else(|| req.header("authorization").and_then(|a| a.strip_prefix("Bearer "))).unwrap_or("");
            if given.is_empty() || !bool::from(given.as_bytes().ct_eq(s.key.as_bytes())) {
                return write_response(w, &Response::json(401, error_body(wire, 401, "A valid API key is required (Settings > Local API).")).with("WWW-Authenticate", "Bearer realm=\"cerebro\"".into()), keep).is_ok();
            }
        }
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.last_request_at.store(now_ms(), Ordering::Relaxed);
        let models = || self.backend.models();
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") => {
                let list = models();
                let body = if wire == Wire::Anthropic {
                    let data: Vec<Value> = list.iter().map(|m| json!({ "type": "model", "id": m.id, "display_name": m.name, "created_at": "2026-01-01T00:00:00Z" })).collect();
                    json!({ "data": data, "has_more": false, "first_id": list.first().map(|m| m.id.clone()), "last_id": list.last().map(|m| m.id.clone()) })
                } else {
                    let data: Vec<Value> = list.iter().map(|m| json!({ "id": m.id, "object": "model", "created": 0, "owned_by": "cerebro" })).collect();
                    json!({ "object": "list", "data": data })
                };
                write_response(w, &Response::json(200, body), keep).is_ok()
            }
            ("GET", p) if p.starts_with("/v1/models/") => {
                let id = &p["/v1/models/".len()..];
                match models().into_iter().find(|m| m.id == id) {
                    Some(m) if wire == Wire::Anthropic => write_response(w, &Response::json(200, json!({ "type": "model", "id": m.id, "display_name": m.name, "created_at": "2026-01-01T00:00:00Z" })), keep).is_ok(),
                    Some(m) => write_response(w, &Response::json(200, json!({ "id": m.id, "object": "model", "created": 0, "owned_by": "cerebro" })), keep).is_ok(),
                    None => fail(w, 404, "No such model."),
                }
            }
            ("POST", "/v1/chat/completions" | "/v1/messages") => {
                let Ok(body) = serde_json::from_slice::<Value>(&req.body) else {
                    return fail(w, 400, "The body is not JSON.");
                };
                let parsed = if req.path == "/v1/messages" { parse_anthropic(&body) } else { parse_openai(&body) };
                let (mut api_req, opts) = match parsed {
                    Ok(p) => p,
                    Err(e) => return fail(w, e.status, &e.message),
                };
                // A request naming another library model loads it (when allowed); any other name
                // (apps often send their own default) runs on the loaded model.
                if let Some(name) = &opts.model {
                    if let Some(m) = models().into_iter().find(|m| m.id.eq_ignore_ascii_case(name) || m.name.eq_ignore_ascii_case(name)) {
                        if !m.loaded {
                            if !s.allow_model_switch {
                                return fail(w, 409, &format!("{} is not loaded, and loading models from the API is off (Settings > Local API).", m.name));
                            }
                            api_req.load = Some(m.id);
                        }
                    }
                }
                respond(self.backend.as_ref(), wire, api_req, Opts { keep, ..opts }, w)
            }
            ("GET", "/" | "/v1" | "/health") => write_response(w, &Response::json(200, json!({ "status": "ok", "server": "cerebro" })), keep).is_ok(),
            _ => fail(w, 404, "Not found."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Read;

    /// Replies from a script: each event in order, then done. Records what it was asked.
    struct Fake {
        script: Vec<ApiEvent>,
        seen: Mutex<Vec<ApiRequest>>,
        stopped_at: Mutex<Option<usize>>,
    }

    impl Fake {
        fn new(script: Vec<ApiEvent>) -> Arc<Self> {
            Arc::new(Fake { script, seen: Mutex::new(Vec::new()), stopped_at: Mutex::new(None) })
        }
    }

    impl ApiBackend for Fake {
        fn models(&self) -> Vec<ApiModel> {
            vec![ApiModel { id: "qwen3-0.6b-q8_0".into(), name: "Qwen3 0.6B".into(), loaded: true }, ApiModel { id: "qwen3.8-27b".into(), name: "Qwen3.8 27B".into(), loaded: false }]
        }
        fn chat(&self, req: ApiRequest, on: &mut dyn FnMut(ApiEvent) -> bool) -> Result<ApiDone, ApiError> {
            let max = req.max_tokens;
            self.seen.lock().unwrap().push(req);
            let mut n = 0;
            for e in &self.script {
                n += 1;
                if !on(e.clone()) {
                    *self.stopped_at.lock().unwrap() = Some(n);
                    break;
                }
            }
            Ok(ApiDone { model: "qwen3-0.6b-q8_0".into(), prompt_tokens: 11, output_tokens: n, length: n >= max })
        }
    }

    fn server(fake: Arc<Fake>, f: impl FnOnce(&mut Settings)) -> (Arc<LocalApi>, SocketAddr, String) {
        let api = LocalApi::new(Settings::default(), fake, Box::new(|_| {}));
        {
            // Port 0: the OS picks a free port at bind time (picking one first races other tests).
            let mut s = api.settings();
            s.port = 0;
            f(&mut s);
        }
        api.set_enabled(true).unwrap();
        let addr = api.bound().unwrap();
        let key = api.settings().key.clone();
        (api, addr, key)
    }

    fn raw(addr: SocketAddr, method: &str, path: &str, headers: &[(&str, String)], body: &str) -> (u16, HashMap<String, String>, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        let mut req = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
        if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
            req.push_str(&format!("Host: 127.0.0.1:{}\r\n", addr.port()));
        }
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(body);
        s.write_all(req.as_bytes()).unwrap();
        let mut out = String::new();
        if let Err(e) = s.read_to_string(&mut out) {
            panic!("{method} {path}: read failed ({e:?}) after {} bytes: {out:?}", out.len());
        }
        let (head, body) = out.split_once("\r\n\r\n").unwrap();
        let status = head[9..12].parse().unwrap();
        let hs = head.lines().skip(1).filter_map(|l| l.split_once(": ")).map(|(k, v)| (k.to_ascii_lowercase(), v.to_owned())).collect();
        (status, hs, body.to_owned())
    }

    fn json_call(addr: SocketAddr, path: &str, key: &str, body: Value) -> (u16, Value) {
        let (st, _, b) = raw(addr, "POST", path, &[("Authorization", format!("Bearer {key}")), ("Content-Type", "application/json".into())], &body.to_string());
        (st, serde_json::from_str(&b).unwrap_or(Value::Null))
    }

    /// The `data:` payloads of an SSE body.
    fn events(body: &str) -> Vec<String> {
        body.split("\n\n").filter_map(|e| e.lines().find_map(|l| l.strip_prefix("data: ")).map(str::to_owned)).collect()
    }

    fn text(s: &str) -> ApiEvent {
        ApiEvent::Text(s.into())
    }

    #[test]
    fn guards_key_host_origin_and_both_error_shapes() {
        let (api, addr, key) = server(Fake::new(vec![text("hi")]), |_| {});
        let body = r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let (st, hs, b) = raw(addr, "POST", "/v1/chat/completions", &[], body);
        assert_eq!(st, 401);
        assert!(hs.contains_key("www-authenticate"));
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["error"]["type"], "authentication_error");
        let (st, _, b) = raw(addr, "POST", "/v1/messages", &[("x-api-key", "wrong".into())], r#"{"max_tokens":5,"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(st, 401);
        assert_eq!(serde_json::from_str::<Value>(&b).unwrap()["type"], "error", "Anthropic error shape");
        let (st, _, _) = raw(addr, "POST", "/v1/messages", &[("x-api-key", key.clone())], r#"{"max_tokens":5,"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(st, 200, "x-api-key works");
        let (st, _, _) = raw(addr, "GET", "/v1/models", &[("Host", "evil.example".into()), ("Authorization", format!("Bearer {key}"))], "");
        assert_eq!(st, 421, "DNS rebinding");
        let (st, _, _) = raw(addr, "GET", "/v1/models", &[("Origin", "https://evil.example".into()), ("Authorization", format!("Bearer {key}"))], "");
        assert_eq!(st, 403, "web page");
        api.set_options(false, false, true).unwrap();
        let (st, _, _) = raw(addr, "GET", "/v1/models", &[], "");
        assert_eq!(st, 200, "the key requirement is a setting");
        assert!(api.view()["requests"].as_u64().unwrap() >= 2);
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn models_in_both_shapes() {
        let (api, addr, key) = server(Fake::new(vec![]), |_| {});
        let (_, _, b) = raw(addr, "GET", "/v1/models", &[("Authorization", format!("Bearer {key}"))], "");
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "qwen3-0.6b-q8_0");
        let (_, _, b) = raw(addr, "GET", "/v1/models", &[("x-api-key", key.clone()), ("anthropic-version", "2023-06-01".into())], "");
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["data"][1]["display_name"], "Qwen3.8 27B");
        assert_eq!(v["has_more"], false);
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn openai_whole_and_streamed_with_reasoning_and_usage() {
        let fake = Fake::new(vec![ApiEvent::Thinking("hmm".into()), text("Hel"), text("lo.")]);
        let (api, addr, key) = server(fake.clone(), |_| {});
        let (st, v) = json_call(addr, "/v1/chat/completions", &key, json!({ "model": "gpt-4o", "reasoning_effort": "high", "max_tokens": 50, "messages": [{ "role": "system", "content": "Be brief." }, { "role": "user", "content": [{ "type": "text", "text": "hi" }] }] }));
        assert_eq!(st, 200);
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "gpt-4o", "the asked-for name is echoed");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello.");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "hmm");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 14);
        let seen = fake.seen.lock().unwrap().pop().unwrap();
        assert_eq!(seen.system, "Be brief.");
        assert!(seen.think);
        assert_eq!(seen.messages, vec![Msg::User("hi".into())]);
        assert_eq!(seen.load, None, "an unknown model name runs on the loaded model");

        let (st, hs, b) = raw(addr, "POST", "/v1/chat/completions", &[("Authorization", format!("Bearer {key}"))], &json!({ "stream": true, "stream_options": { "include_usage": true }, "messages": [{ "role": "user", "content": "hi" }] }).to_string());
        assert_eq!(st, 200);
        assert_eq!(hs["content-type"], "text/event-stream");
        let ev = events(&b);
        assert_eq!(ev.last().unwrap(), "[DONE]");
        let chunks: Vec<Value> = ev[..ev.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let content: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
        assert_eq!(content, "Hello.");
        assert_eq!(chunks.iter().filter_map(|c| c["choices"][0]["finish_reason"].as_str()).collect::<Vec<_>>(), ["stop"]);
        assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 3);
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn tool_calls_go_back_to_the_app_in_both_formats() {
        let call = ApiEvent::Call { name: "get_weather".into(), arguments: json!({ "city": "Oslo" }) };
        let fake = Fake::new(vec![text("Checking."), call]);
        let (api, addr, key) = server(fake.clone(), |_| {});
        let tools = json!([{ "type": "function", "function": { "name": "get_weather", "description": "Weather now", "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } } }]);
        let (_, v) = json_call(addr, "/v1/chat/completions", &key, json!({ "tools": tools, "messages": [{ "role": "user", "content": "weather?" }] }));
        let tc = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap()).unwrap(), json!({ "city": "Oslo" }));
        assert!(tc["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(fake.seen.lock().unwrap().last().unwrap().tools[0].name, "get_weather");

        let atools = json!([{ "name": "get_weather", "description": "Weather now", "input_schema": { "type": "object" } }, { "type": "web_search_20250305", "name": "web_search" }]);
        let (_, v) = json_call(addr, "/v1/messages", &key, json!({ "max_tokens": 100, "tools": atools, "messages": [{ "role": "user", "content": "weather?" }] }));
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["content"][0], json!({ "type": "text", "text": "Checking." }));
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["input"], json!({ "city": "Oslo" }));
        assert_eq!(fake.seen.lock().unwrap().last().unwrap().tools.len(), 1, "hosted tools are not offered");

        // Streamed Anthropic: the exact event order an SDK expects.
        let (_, _, b) = raw(addr, "POST", "/v1/messages", &[("x-api-key", key.clone())], &json!({ "stream": true, "max_tokens": 100, "tools": atools, "messages": [{ "role": "user", "content": "weather?" }] }).to_string());
        let kinds: Vec<String> = events(&b).iter().map(|e| serde_json::from_str::<Value>(e).unwrap()["type"].as_str().unwrap().to_owned()).collect();
        assert_eq!(kinds, ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
        let evs: Vec<Value> = events(&b).iter().map(|e| serde_json::from_str(e).unwrap()).collect();
        assert_eq!(evs[4]["index"], 1);
        assert_eq!(serde_json::from_str::<Value>(evs[5]["delta"]["partial_json"].as_str().unwrap()).unwrap(), json!({ "city": "Oslo" }));
        assert_eq!(evs[7]["delta"]["stop_reason"], "tool_use");
        assert!(b.contains("event: message_stop\n"), "named events");
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn tool_history_is_read_from_both_formats() {
        let (r, _) = parse_openai(&json!({ "messages": [
            { "role": "user", "content": "weather?" },
            { "role": "assistant", "content": null, "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "w", "arguments": "{\"city\":\"Oslo\"}" } }] },
            { "role": "tool", "tool_call_id": "call_1", "content": "12 C" },
        ] }))
        .unwrap();
        assert_eq!(r.messages, vec![Msg::User("weather?".into()), Msg::Assistant { text: String::new(), calls: vec![ToolCall { name: "w".into(), arguments: json!({ "city": "Oslo" }) }] }, Msg::Tool("12 C".into())]);
        let (r, _) = parse_anthropic(&json!({ "max_tokens": 9, "system": [{ "type": "text", "text": "S" }], "messages": [
            { "role": "user", "content": "weather?" },
            { "role": "assistant", "content": [{ "type": "thinking", "thinking": "x" }, { "type": "text", "text": "Checking." }, { "type": "tool_use", "id": "toolu_1", "name": "w", "input": { "city": "Oslo" } }] },
            { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": [{ "type": "text", "text": "rain" }], "is_error": true }, { "type": "text", "text": "and tomorrow?" }] },
        ] }))
        .unwrap();
        assert_eq!(r.system, "S");
        assert_eq!(
            r.messages,
            vec![
                Msg::User("weather?".into()),
                Msg::Assistant { text: "Checking.".into(), calls: vec![ToolCall { name: "w".into(), arguments: json!({ "city": "Oslo" }) }] },
                Msg::Tool("Error: rain".into()),
                Msg::User("and tomorrow?".into()),
            ]
        );
    }

    #[test]
    fn requests_are_validated() {
        let bad = |v: Value| parse_openai(&v).unwrap_err().message;
        assert!(bad(json!({ "messages": [] })).contains("non-empty"));
        assert!(bad(json!({ "messages": [{ "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "x" } }] }] })).contains("image_url"));
        assert!(bad(json!({ "n": 2, "messages": [{ "role": "user", "content": "x" }] })).contains("n = 1"));
        assert!(bad(json!({ "temperature": 5, "messages": [{ "role": "user", "content": "x" }] })).contains("temperature"));
        assert!(parse_anthropic(&json!({ "messages": [{ "role": "user", "content": "x" }] })).unwrap_err().message.contains("max_tokens is required"));
        let (r, o) = parse_openai(&json!({ "max_tokens": 10_000_000, "stop": "END", "tool_choice": "none", "tools": [{ "type": "function", "function": { "name": "t" } }], "messages": [{ "role": "user", "content": "x" }] })).unwrap();
        assert_eq!(r.max_tokens, MAX_TOKENS_CAP);
        assert_eq!(o.stops, ["END"]);
        assert!(r.tools.is_empty(), "tool_choice none");
        assert_eq!(r.tools.len(), 0);
    }

    #[test]
    fn stop_sequences_cut_the_text_and_stop_generation() {
        let fake = Fake::new(vec![text("one two TH"), text("REE four"), text("five")]);
        let (api, addr, key) = server(fake.clone(), |_| {});
        let (_, v) = json_call(addr, "/v1/messages", &key, json!({ "max_tokens": 50, "stop_sequences": ["THREE"], "messages": [{ "role": "user", "content": "count" }] }));
        assert_eq!(v["content"][0]["text"], "one two ");
        assert_eq!(v["stop_reason"], "stop_sequence");
        assert_eq!(v["stop_sequence"], "THREE");
        assert_eq!(*fake.stopped_at.lock().unwrap(), Some(2), "generation stopped at the sequence");
        let mut s = Stops::new(vec!["ab".into()]);
        assert_eq!(s.feed("xa"), "x");
        assert_eq!(s.feed("c"), "ac", "a held prefix that did not become a stop is released");
        assert_eq!(s.finish(), "");
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn naming_a_library_model_loads_it_unless_switching_is_off() {
        let fake = Fake::new(vec![text("ok")]);
        let (api, addr, key) = server(fake.clone(), |_| {});
        let (st, _) = json_call(addr, "/v1/chat/completions", &key, json!({ "model": "Qwen3.8 27B", "messages": [{ "role": "user", "content": "x" }] }));
        assert_eq!(st, 200);
        assert_eq!(fake.seen.lock().unwrap().last().unwrap().load.as_deref(), Some("qwen3.8-27b"));
        let (_, _) = json_call(addr, "/v1/chat/completions", &key, json!({ "model": "qwen3-0.6b-q8_0", "messages": [{ "role": "user", "content": "x" }] }));
        assert_eq!(fake.seen.lock().unwrap().last().unwrap().load, None, "already loaded");
        api.set_options(true, false, false).unwrap();
        let (st, v) = json_call(addr, "/v1/chat/completions", &key, json!({ "model": "qwen3.8-27b", "messages": [{ "role": "user", "content": "x" }] }));
        assert_eq!(st, 409);
        assert!(v["error"]["message"].as_str().unwrap().contains("loading models from the API is off"));
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn length_is_reported() {
        let (api, addr, key) = server(Fake::new(vec![text("a"), text("b")]), |_| {});
        let (_, v) = json_call(addr, "/v1/chat/completions", &key, json!({ "max_tokens": 2, "messages": [{ "role": "user", "content": "x" }] }));
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        let (_, v) = json_call(addr, "/v1/messages", &key, json!({ "max_tokens": 2, "messages": [{ "role": "user", "content": "x" }] }));
        assert_eq!(v["stop_reason"], "max_tokens");
        api.set_enabled(false).unwrap();
    }

    #[test]
    fn lan_peers_and_hosts() {
        assert!(peer_ok("127.0.0.1".parse().unwrap(), false));
        assert!(!peer_ok("192.168.1.20".parse().unwrap(), false));
        assert!(peer_ok("192.168.1.20".parse().unwrap(), true));
        assert!(!peer_ok("8.8.8.8".parse().unwrap(), true), "never the internet");
        assert!(peer_ok("fd00::5".parse().unwrap(), true));
    }
}
