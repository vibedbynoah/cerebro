//! Cloud models through the user's own API keys and plans. The user signs up and pays at the
//! provider; Cerebro only stores the provider's address and which models to show (cloud.json in
//! the app's config folder) and keeps each key in the Windows Credential Manager (never in a file,
//! a log or a reply to the window). A chat with a cloud model goes straight from this app to the
//! provider over HTTPS; nothing is proxied.
//!
//! Three wire formats cover the providers: OpenAI's chat completions (most providers and every
//! "OpenAI-compatible" endpoint), Anthropic's Messages API and Google's Gemini API. Replies stream
//! (server-sent events); reasoning, where a provider streams it, arrives as thinking.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Model ids the window uses for cloud models: "cloud:<provider id>/<model id>".
pub const PREFIX: &str = "cloud:";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    OpenAi,
    Anthropic,
    Gemini,
}

/// A provider the user added: where it is and which of its models appear in Chat. No key here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub base_url: String,
    pub models: Vec<String>,
    /// Whether a key is stored for it (filled in when listing; never the key itself).
    #[serde(default)]
    pub has_key: bool,
}

/// A provider the app knows: its official API address and where to get a key.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    pub base_url: &'static str,
    pub keys_url: &'static str,
}

pub const PRESETS: &[Preset] = &[
    Preset { id: "openai", name: "OpenAI", kind: Kind::OpenAi, base_url: "https://api.openai.com/v1", keys_url: "https://platform.openai.com/api-keys" },
    Preset { id: "anthropic", name: "Anthropic", kind: Kind::Anthropic, base_url: "https://api.anthropic.com/v1", keys_url: "https://console.anthropic.com/settings/keys" },
    Preset { id: "gemini", name: "Google Gemini", kind: Kind::Gemini, base_url: "https://generativelanguage.googleapis.com/v1beta", keys_url: "https://aistudio.google.com/apikey" },
    Preset { id: "openrouter", name: "OpenRouter", kind: Kind::OpenAi, base_url: "https://openrouter.ai/api/v1", keys_url: "https://openrouter.ai/keys" },
    Preset { id: "deepseek", name: "DeepSeek", kind: Kind::OpenAi, base_url: "https://api.deepseek.com/v1", keys_url: "https://platform.deepseek.com/api_keys" },
    Preset { id: "mistral", name: "Mistral", kind: Kind::OpenAi, base_url: "https://api.mistral.ai/v1", keys_url: "https://console.mistral.ai/api-keys" },
    Preset { id: "groq", name: "Groq", kind: Kind::OpenAi, base_url: "https://api.groq.com/openai/v1", keys_url: "https://console.groq.com/keys" },
    Preset { id: "xai", name: "xAI", kind: Kind::OpenAi, base_url: "https://api.x.ai/v1", keys_url: "https://console.x.ai" },
    Preset { id: "together", name: "Together AI", kind: Kind::OpenAi, base_url: "https://api.together.xyz/v1", keys_url: "https://api.together.ai/settings/api-keys" },
];

const KEYRING_SERVICE: &str = "Cerebro cloud models";

/// Where the provider list lives (set once at startup).
static FILE: Mutex<Option<PathBuf>> = Mutex::new(None);
/// Set by Stop: the streaming reply ends at its next piece.
static CANCEL: AtomicBool = AtomicBool::new(false);

pub fn set_file(path: PathBuf) {
    *FILE.lock().unwrap_or_else(|e| e.into_inner()) = Some(path);
}

pub fn cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}

fn file() -> Result<PathBuf, String> {
    FILE.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or_else(|| "The settings folder is unknown.".to_owned())
}

/// The providers the user added, each marked with whether its key is stored.
pub fn providers() -> Vec<Provider> {
    let Ok(path) = file() else { return Vec::new() };
    let mut list: Vec<Provider> = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    for p in &mut list {
        p.has_key = key(&p.id).is_some();
    }
    list
}

fn save_list(list: &[Provider]) -> Result<(), String> {
    let path = file()?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Could not save the providers: {e}"))?;
    }
    let clean: Vec<Provider> = list.iter().map(|p| Provider { has_key: false, ..p.clone() }).collect();
    let body = serde_json::to_vec_pretty(&clean).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, body).map_err(|e| format!("Could not save the providers: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("Could not save the providers: {e}"))
}

fn key(id: &str) -> Option<String> {
    keyring::Entry::new(KEYRING_SERVICE, id).ok()?.get_password().ok().filter(|k| !k.is_empty())
}

fn set_key(id: &str, k: &str) -> Result<(), String> {
    let e = keyring::Entry::new(KEYRING_SERVICE, id).map_err(|e| format!("The Credential Manager is not available: {e}"))?;
    e.set_password(k).map_err(|e| format!("Could not store the key in the Credential Manager: {e}"))
}

fn delete_key(id: &str) {
    if let Ok(e) = keyring::Entry::new(KEYRING_SERVICE, id) {
        let _ = e.delete_credential();
    }
}

/// Checks a provider's id, name and address.
pub fn validate(p: &Provider) -> Result<(), String> {
    if p.id.is_empty() || p.id.len() > 40 || !p.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err("A provider id uses letters, digits, - and _ (at most 40).".to_owned());
    }
    if p.name.trim().is_empty() || p.name.len() > 60 {
        return Err("Give the provider a name (at most 60 characters).".to_owned());
    }
    let url = reqwest::Url::parse(&p.base_url).map_err(|_| "The address is not a URL.".to_owned())?;
    let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"));
    if url.scheme() != "https" && !(url.scheme() == "http" && local) {
        return Err("Use an https:// address (http:// only for a server on this computer).".to_owned());
    }
    if p.models.len() > 200 || p.models.iter().any(|m| m.is_empty() || m.len() > 200) {
        return Err("Too many models, or a model name that is empty or too long.".to_owned());
    }
    Ok(())
}

/// Adds or replaces a provider; a non-empty `api_key` replaces its stored key.
pub fn save(p: Provider, api_key: Option<String>) -> Result<Vec<Provider>, String> {
    validate(&p)?;
    if let Some(k) = api_key.filter(|k| !k.trim().is_empty()) {
        set_key(&p.id, k.trim())?;
    }
    let mut list = providers();
    list.retain(|x| x.id != p.id);
    list.push(p);
    save_list(&list)?;
    Ok(providers())
}

pub fn remove(id: &str) -> Result<Vec<Provider>, String> {
    let mut list = providers();
    list.retain(|x| x.id != id);
    save_list(&list)?;
    delete_key(id);
    Ok(providers())
}

fn client(timeout: Option<Duration>) -> Result<reqwest::blocking::Client, String> {
    let mut b = reqwest::blocking::Client::builder().connect_timeout(Duration::from_secs(15)).user_agent(concat!("Cerebro/", env!("CARGO_PKG_VERSION")));
    if let Some(t) = timeout {
        b = b.timeout(t);
    }
    b.build().map_err(|e| e.to_string())
}

/// A provider's answer as a plain sentence for the user.
fn http_error(status: reqwest::StatusCode, body: &str, provider: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().or(v["error"].as_str()).or(v["message"].as_str()).map(str::to_owned))
        .unwrap_or_default();
    let what = match status.as_u16() {
        400 => "The provider did not accept the request",
        401 | 403 => "The provider refused the key: check it, or create a new one on its site",
        404 => "The provider does not know that model or address",
        408 | 504 => "The provider took too long to answer",
        413 => "The conversation is too long for this model",
        429 => "The provider's rate limit or your plan's quota was reached: wait a little or check your plan",
        500..=599 => "The provider had a problem on its side: try again shortly",
        _ => "The provider answered with an error",
    };
    let detail = detail.chars().take(300).collect::<String>();
    if detail.is_empty() { format!("{what} ({provider}, HTTP {}).", status.as_u16()) } else { format!("{what} ({provider}, HTTP {}): {detail}", status.as_u16()) }
}

fn net_error(e: &reqwest::Error, provider: &str) -> String {
    if e.is_timeout() {
        format!("{provider} did not answer in time. Check your connection and try again.")
    } else if e.is_connect() {
        format!("Could not reach {provider}. Check your internet connection.")
    } else {
        format!("The connection to {provider} failed: {e}")
    }
}

fn find(id: &str) -> Result<(Provider, String), String> {
    let p = providers().into_iter().find(|p| p.id == id).ok_or_else(|| "That cloud provider is not set up any more (Settings > Cloud models).".to_owned())?;
    let k = key(id).ok_or_else(|| format!("No API key is stored for {}. Add it in Settings > Cloud models.", p.name))?;
    Ok((p, k))
}

fn base(p: &Provider) -> String {
    p.base_url.trim_end_matches('/').to_owned()
}

/// The models a provider offers with this key (sorted), for picking which appear in Chat.
pub fn list_models(p: &Provider, api_key: Option<&str>) -> Result<Vec<String>, String> {
    validate(p)?;
    let stored;
    let k = match api_key.filter(|k| !k.trim().is_empty()) {
        Some(k) => k.trim(),
        None => {
            stored = key(&p.id).ok_or_else(|| "Paste the API key first.".to_owned())?;
            stored.as_str()
        }
    };
    let c = client(Some(Duration::from_secs(30)))?;
    let req = match p.kind {
        Kind::OpenAi => c.get(format!("{}/models", base(p))).bearer_auth(k),
        Kind::Anthropic => c.get(format!("{}/models?limit=1000", base(p))).header("x-api-key", k).header("anthropic-version", "2023-06-01"),
        Kind::Gemini => c.get(format!("{}/models?pageSize=1000", base(p))).header("x-goog-api-key", k),
    };
    let r = req.send().map_err(|e| net_error(&e, &p.name))?;
    let status = r.status();
    let body = r.text().map_err(|e| net_error(&e, &p.name))?;
    if !status.is_success() {
        return Err(http_error(status, &body, &p.name));
    }
    let v: Value = serde_json::from_str(&body).map_err(|_| format!("{} sent a model list this app could not read.", p.name))?;
    let mut ids: Vec<String> = match p.kind {
        Kind::OpenAi | Kind::Anthropic => v["data"].as_array().map(|a| a.iter().filter_map(|m| m["id"].as_str().map(str::to_owned)).collect()).unwrap_or_default(),
        Kind::Gemini => v["models"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|m| m["supportedGenerationMethods"].as_array().is_some_and(|s| s.iter().any(|x| x == "generateContent")))
                    .filter_map(|m| m["name"].as_str().map(|n| n.trim_start_matches("models/").to_owned()))
                    .collect()
            })
            .unwrap_or_default(),
    };
    ids.sort();
    ids.dedup();
    Ok(ids)
}

/// One message of the conversation sent to the provider.
pub struct Turn<'a> {
    pub role: &'a str,
    pub text: &'a str,
    /// Image data URLs (data:image/...;base64,...).
    pub images: Vec<&'a str>,
}

pub struct Settings {
    pub temperature: f32,
    pub max_tokens: usize,
}

/// What came back.
#[derive(Default)]
pub struct Outcome {
    pub output_tokens: usize,
    pub thinking_pieces: usize,
    pub stopped: bool,
}

/// The pieces a reply streams, in order.
pub enum Piece<'a> {
    Thinking(&'a str),
    Text(&'a str),
}

fn split_data_url(u: &str) -> Option<(&str, &str)> {
    let rest = u.strip_prefix("data:")?;
    let (mime, data) = rest.split_once(";base64,")?;
    Some((mime, data))
}

/// The request body for a provider's format (pure, so it is tested without a network).
pub fn request_body(kind: Kind, model: &str, system: &str, turns: &[Turn<'_>], s: &Settings) -> Value {
    match kind {
        Kind::OpenAi => {
            let mut messages = Vec::new();
            if !system.is_empty() {
                messages.push(json!({ "role": "system", "content": system }));
            }
            for t in turns {
                let content = if t.images.is_empty() {
                    json!(t.text)
                } else {
                    let mut parts = vec![json!({ "type": "text", "text": t.text })];
                    parts.extend(t.images.iter().map(|u| json!({ "type": "image_url", "image_url": { "url": u } })));
                    Value::Array(parts)
                };
                messages.push(json!({ "role": t.role, "content": content }));
            }
            json!({ "model": model, "messages": messages, "stream": true, "temperature": s.temperature, "max_tokens": s.max_tokens })
        }
        Kind::Anthropic => {
            let messages: Vec<Value> = turns
                .iter()
                .map(|t| {
                    let mut parts: Vec<Value> = t.images.iter().filter_map(|u| split_data_url(u)).map(|(mime, data)| json!({ "type": "image", "source": { "type": "base64", "media_type": mime, "data": data } })).collect();
                    parts.push(json!({ "type": "text", "text": t.text }));
                    json!({ "role": t.role, "content": parts })
                })
                .collect();
            let mut body = json!({ "model": model, "messages": messages, "stream": true, "temperature": s.temperature.min(1.0), "max_tokens": s.max_tokens });
            if !system.is_empty() {
                body["system"] = json!(system);
            }
            body
        }
        Kind::Gemini => {
            let contents: Vec<Value> = turns
                .iter()
                .map(|t| {
                    let mut parts: Vec<Value> = t.images.iter().filter_map(|u| split_data_url(u)).map(|(mime, data)| json!({ "inline_data": { "mime_type": mime, "data": data } })).collect();
                    parts.push(json!({ "text": t.text }));
                    json!({ "role": if t.role == "assistant" { "model" } else { "user" }, "parts": parts })
                })
                .collect();
            let mut body = json!({ "contents": contents, "generationConfig": { "temperature": s.temperature, "maxOutputTokens": s.max_tokens } });
            if !system.is_empty() {
                body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
            }
            body
        }
    }
}

/// The text and thinking in one streamed event of a provider's format (pure, tested on recorded
/// events). Returns (thinking, text, finished).
pub fn parse_event(kind: Kind, data: &str) -> (Option<String>, Option<String>, bool) {
    if data.trim() == "[DONE]" {
        return (None, None, true);
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else { return (None, None, false) };
    match kind {
        Kind::OpenAi => {
            let d = &v["choices"][0]["delta"];
            let think = d["reasoning_content"].as_str().or(d["reasoning"].as_str()).filter(|s| !s.is_empty()).map(str::to_owned);
            let text = d["content"].as_str().filter(|s| !s.is_empty()).map(str::to_owned);
            (think, text, false)
        }
        Kind::Anthropic => match v["type"].as_str() {
            Some("content_block_delta") => {
                let d = &v["delta"];
                match d["type"].as_str() {
                    Some("text_delta") => (None, d["text"].as_str().map(str::to_owned), false),
                    Some("thinking_delta") => (d["thinking"].as_str().map(str::to_owned), None, false),
                    _ => (None, None, false),
                }
            }
            Some("message_stop") => (None, None, true),
            _ => (None, None, false),
        },
        Kind::Gemini => {
            let (mut think, mut text) = (String::new(), String::new());
            for part in v["candidates"][0]["content"]["parts"].as_array().into_iter().flatten() {
                if let Some(t) = part["text"].as_str() {
                    if part["thought"].as_bool() == Some(true) {
                        think.push_str(t);
                    } else {
                        text.push_str(t);
                    }
                }
            }
            ((!think.is_empty()).then_some(think), (!text.is_empty()).then_some(text), false)
        }
    }
}

/// Streams a reply from `provider` / `model`; `on_piece` returns false to stop.
pub fn chat(provider_id: &str, model: &str, system: &str, turns: &[Turn<'_>], s: &Settings, mut on_piece: impl FnMut(Piece<'_>) -> bool) -> Result<Outcome, String> {
    CANCEL.store(false, Ordering::SeqCst);
    let (p, k) = find(provider_id)?;
    let body = request_body(p.kind, model, system, turns, s);
    let c = client(None)?;
    let req = match p.kind {
        Kind::OpenAi => c.post(format!("{}/chat/completions", base(&p))).bearer_auth(&k),
        Kind::Anthropic => c.post(format!("{}/messages", base(&p))).header("x-api-key", &k).header("anthropic-version", "2023-06-01"),
        Kind::Gemini => c.post(format!("{}/models/{model}:streamGenerateContent?alt=sse", base(&p))).header("x-goog-api-key", &k),
    };
    let r = req.json(&body).send().map_err(|e| net_error(&e, &p.name))?;
    let status = r.status();
    if !status.is_success() {
        let text = r.text().unwrap_or_default();
        return Err(http_error(status, &text, &p.name));
    }
    let mut out = Outcome::default();
    let reader = BufReader::new(r);
    for line in reader.lines() {
        if CANCEL.load(Ordering::SeqCst) {
            out.stopped = true;
            break;
        }
        let line = line.map_err(|e| format!("The connection to {} dropped: {e}", p.name))?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        let (think, text, done) = parse_event(p.kind, data.trim_start());
        if let Some(t) = think {
            out.thinking_pieces += 1;
            if !on_piece(Piece::Thinking(&t)) {
                out.stopped = true;
                break;
            }
        }
        if let Some(t) = text {
            out.output_tokens += 1;
            if !on_piece(Piece::Text(&t)) {
                out.stopped = true;
                break;
            }
        }
        if done {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings { temperature: 0.7, max_tokens: 512 }
    }

    #[test]
    fn openai_body_has_system_then_turns_and_images() {
        let turns = [Turn { role: "user", text: "Hi", images: vec![] }, Turn { role: "assistant", text: "Hello", images: vec![] }, Turn { role: "user", text: "What is this?", images: vec!["data:image/png;base64,AAAA"] }];
        let b = request_body(Kind::OpenAi, "gpt-x", "Be brief.", &turns, &settings());
        assert_eq!(b["messages"][0]["role"], "system");
        assert_eq!(b["messages"][1]["content"], "Hi");
        assert_eq!(b["messages"][3]["content"][1]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(b["stream"], true);
        assert_eq!(b["max_tokens"], 512);
    }

    #[test]
    fn anthropic_body_puts_system_aside_and_images_first() {
        let turns = [Turn { role: "user", text: "Describe", images: vec!["data:image/jpeg;base64,BBBB"] }];
        let b = request_body(Kind::Anthropic, "claude-x", "Sys", &turns, &settings());
        assert_eq!(b["system"], "Sys");
        assert_eq!(b["messages"][0]["content"][0]["source"]["media_type"], "image/jpeg");
        assert_eq!(b["messages"][0]["content"][1]["text"], "Describe");
    }

    #[test]
    fn gemini_body_uses_model_role_and_system_instruction() {
        let turns = [Turn { role: "user", text: "Q", images: vec![] }, Turn { role: "assistant", text: "A", images: vec![] }];
        let b = request_body(Kind::Gemini, "gemini-x", "Sys", &turns, &settings());
        assert_eq!(b["contents"][1]["role"], "model");
        assert_eq!(b["systemInstruction"]["parts"][0]["text"], "Sys");
        assert_eq!(b["generationConfig"]["maxOutputTokens"], 512);
    }

    #[test]
    fn stream_events_parse_for_each_format() {
        assert_eq!(parse_event(Kind::OpenAi, r#"{"choices":[{"delta":{"content":"Hel"}}]}"#).1.as_deref(), Some("Hel"));
        assert_eq!(parse_event(Kind::OpenAi, r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#).0.as_deref(), Some("hmm"));
        assert!(parse_event(Kind::OpenAi, "[DONE]").2);
        assert_eq!(parse_event(Kind::Anthropic, r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#).1.as_deref(), Some("Hi"));
        assert_eq!(parse_event(Kind::Anthropic, r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"t"}}"#).0.as_deref(), Some("t"));
        assert!(parse_event(Kind::Anthropic, r#"{"type":"message_stop"}"#).2);
        let g = parse_event(Kind::Gemini, r#"{"candidates":[{"content":{"parts":[{"text":"plan","thought":true},{"text":"Answer"}]}}]}"#);
        assert_eq!((g.0.as_deref(), g.1.as_deref()), (Some("plan"), Some("Answer")));
        assert_eq!(parse_event(Kind::OpenAi, "not json"), (None, None, false));
    }

    #[test]
    fn provider_addresses_are_checked() {
        let p = |url: &str| Provider { id: "x".into(), name: "X".into(), kind: Kind::OpenAi, base_url: url.into(), models: vec![], has_key: false };
        assert!(validate(&p("https://api.example.com/v1")).is_ok());
        assert!(validate(&p("http://localhost:8080/v1")).is_ok());
        assert!(validate(&p("http://api.example.com/v1")).is_err());
        assert!(validate(&p("not a url")).is_err());
        let mut bad = p("https://a.b");
        bad.id = "has space".into();
        assert!(validate(&bad).is_err());
    }

    /// The whole path on a real socket: a server on this computer answers like an OpenAI-compatible
    /// provider (model list, then a streamed reply with reasoning), with the key read back from the
    /// Windows Credential Manager. Removes its provider and key afterwards.
    #[test]
    fn streams_from_a_provider_end_to_end() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut s, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let n = s.read(&mut buf).unwrap();
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                assert!(req.contains("authorization: Bearer test-key") || req.contains("Authorization: Bearer test-key"), "{req}");
                let body = if req.starts_with("GET /v1/models") {
                    "HTTP/1.1 200 OK
content-type: application/json
connection: close

{\"data\":[{\"id\":\"b-model\"},{\"id\":\"a-model\"}]}".to_owned()
                } else {
                    assert!(req.contains("\"stream\":true"));
                    let events = [r#"{"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#, r#"{"choices":[{"delta":{"content":"Hello"}}]}"#, r#"{"choices":[{"delta":{"content":", world"}}]}"#];
                    let mut b = "HTTP/1.1 200 OK
content-type: text/event-stream
connection: close

".to_owned();
                    for e in events {
                        b.push_str(&format!("data: {e}

"));
                    }
                    b.push_str("data: [DONE]

");
                    b
                };
                s.write_all(body.as_bytes()).unwrap();
            }
        });
        let dir = std::env::temp_dir().join(format!("cerebro-cloud-test-{port}"));
        set_file(dir.join("cloud.json"));
        let p = Provider { id: format!("test{port}"), name: "Test".into(), kind: Kind::OpenAi, base_url: format!("http://127.0.0.1:{port}/v1"), models: vec!["a-model".into()], has_key: false };
        let listed = list_models(&p, Some("test-key")).unwrap();
        assert_eq!(listed, ["a-model", "b-model"]);
        save(p.clone(), Some("test-key".into())).unwrap();
        assert!(providers().iter().any(|x| x.id == p.id && x.has_key));
        let mut text = String::new();
        let mut think = String::new();
        let out = chat(&p.id, "a-model", "Be brief.", &[Turn { role: "user", text: "Hi", images: vec![] }], &settings(), |piece| {
            match piece {
                Piece::Text(t) => text.push_str(t),
                Piece::Thinking(t) => think.push_str(t),
            }
            true
        })
        .unwrap();
        server.join().unwrap();
        remove(&p.id).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(text, "Hello, world");
        assert_eq!(think, "thinking");
        assert_eq!(out.output_tokens, 2);
        assert!(!providers().iter().any(|x| x.id == p.id));
    }

    #[test]
    fn errors_read_as_sentences() {
        let e = http_error(reqwest::StatusCode::UNAUTHORIZED, r#"{"error":{"message":"Incorrect API key"}}"#, "OpenAI");
        assert!(e.starts_with("The provider refused the key") && e.contains("Incorrect API key"));
        assert!(http_error(reqwest::StatusCode::TOO_MANY_REQUESTS, "", "X").contains("quota"));
    }
}
