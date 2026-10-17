//! Tool calls in a local model's own words: the system-prompt block that advertises MCP tools, a
//! streaming scanner that pulls calls out of the reply as it is written, and the wrapper for results.
//!
//! Two dialects (docs/research/2026-10-17/D-tool-calling.md, part 1):
//! - Hermes, used by Qwen3 and Qwen3.8: `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`.
//! - Qwen3-Coder: `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`;
//!   it often drops the `<tool_call>` opener, so a bare `<function=` is a call too.
//!
//! The scanner never shows the user half a delimiter: text that could still become `<tool_call>` is
//! held back until the next bytes prove otherwise.

use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Hermes,
    QwenCoder,
}

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Text(String),
    Call(ToolCall),
    BadCall { raw: String, error: String },
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";

/// The model family's tool convention, from its GGUF chat template.
pub fn dialect_from_template(template: &str) -> Dialect {
    if template.contains("<function=") || template.contains("<parameter=") {
        Dialect::QwenCoder
    } else {
        Dialect::Hermes
    }
}

/// The system-prompt block that lists the tools, in the wording the model was trained on.
pub fn tools_prompt(dialect: Dialect, tools: &[ToolSpec]) -> String {
    let lines: Vec<String> = tools
        .iter()
        .map(|t| json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.input_schema } }).to_string())
        .collect();
    let list = lines.join("\n");
    match dialect {
        Dialect::Hermes => format!(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{list}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
        ),
        Dialect::QwenCoder => format!(
            "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{list}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Do NOT omit the initial <tool_call> tag\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n</IMPORTANT>"
        ),
    }
}

/// One tool result, as the body of the next user turn.
pub fn tool_response(content: &str) -> String {
    format!("<tool_response>\n{content}\n</tool_response>")
}

/// A call as the model itself would have written it, for replaying an assistant turn that called
/// tools (the local API receives these back in the history). The scanner reads it back unchanged.
pub fn render_call(dialect: Dialect, call: &ToolCall) -> String {
    match dialect {
        Dialect::Hermes => format!("{OPEN}\n{}\n{CLOSE}", json!({ "name": call.name, "arguments": call.arguments })),
        Dialect::QwenCoder => {
            let mut s = format!("{OPEN}\n{FN_OPEN}{}>\n", call.name);
            if let Some(args) = call.arguments.as_object() {
                for (k, v) in args {
                    let v = match v {
                        Value::String(t) => t.clone(),
                        other => other.to_string(),
                    };
                    s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
                }
            }
            s.push_str(FN_CLOSE);
            s.push('\n');
            s.push_str(CLOSE);
            s
        }
    }
}

/// Cuts a tool result to `max_chars` characters on a character boundary, saying how much went.
pub fn truncate_result(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_owned();
    }
    let cut: String = s.chars().take(max_chars).collect();
    format!("{cut}\n[... truncated {} characters]", total - max_chars)
}

/// Streaming scanner over the visible reply (reasoning already removed).
pub struct Scanner {
    dialect: Dialect,
    /// For Qwen3-Coder parameters: each tool's property types, so "5" can become 5.
    types: std::collections::HashMap<String, Map<String, Value>>,
    buf: String,
    /// Inside a call: the text since its opening delimiter, and whether it opened with <tool_call>.
    call: Option<(String, bool)>,
    /// A Coder call without its opener just ended: a following </tool_call> (even split across
    /// deltas) belongs to it and is dropped, not shown.
    after_bare: bool,
}

impl Scanner {
    pub fn new(dialect: Dialect, tools: &[ToolSpec]) -> Self {
        let types = tools.iter().map(|t| (t.name.clone(), t.input_schema.get("properties").and_then(Value::as_object).cloned().unwrap_or_default())).collect();
        Scanner { dialect, types, buf: String::new(), call: None, after_bare: false }
    }

    pub fn feed(&mut self, delta: &str) -> Vec<Piece> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        loop {
            if let Some((body, wrapped)) = self.call.as_mut() {
                // Inside a call: wait for its end.
                body.push_str(&self.buf);
                self.buf.clear();
                let end = if *wrapped { CLOSE } else { FN_CLOSE };
                let Some(i) = body.find(end) else { break };
                let inner = body[..i].to_owned();
                let rest = body[i + end.len()..].to_owned();
                let wrapped = *wrapped;
                self.call = None;
                self.after_bare = !wrapped;
                self.buf = rest;
                let raw = if wrapped { format!("{OPEN}{inner}{CLOSE}") } else { format!("{FN_OPEN}{inner}{FN_CLOSE}") };
                out.push(self.parse(&inner, wrapped, raw));
                continue;
            }
            if self.after_bare {
                let t = self.buf.trim_start();
                if t.starts_with(CLOSE) {
                    self.buf = t[CLOSE.len()..].to_owned();
                    self.after_bare = false;
                    continue;
                }
                if CLOSE.starts_with(t) {
                    break; // whitespace or part of the closer so far: wait for more
                }
                self.after_bare = false;
            }
            // Outside a call: find the earliest opener.
            let open = self.buf.find(OPEN).map(|i| (i, OPEN.len(), true));
            let fn_open = if self.dialect == Dialect::QwenCoder { self.buf.find(FN_OPEN).map(|i| (i, 0, false)) } else { None };
            let first = match (open, fn_open) {
                (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
                (a, b) => a.or(b),
            };
            if let Some((i, skip, wrapped)) = first {
                push_text(&mut out, &self.buf[..i]);
                let body = self.buf[i + skip..].to_owned();
                self.buf.clear();
                self.call = Some((String::new(), wrapped));
                self.buf = body;
                continue;
            }
            // No opener: release everything except a tail that could still start one.
            let hold = self.holdback();
            let cut = self.buf.len() - hold;
            if cut > 0 {
                let text: String = self.buf.drain(..cut).collect();
                push_text(&mut out, &text);
            }
            break;
        }
        merge(out)
    }

    /// Flushes what is left: held-back text is text; an unterminated call is a BadCall.
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        if let Some((mut body, wrapped)) = self.call.take() {
            body.push_str(&self.buf);
            self.buf.clear();
            // A Coder call that closed its function but not the wrapper is complete enough to run.
            if wrapped {
                if let Some(i) = body.find(FN_CLOSE) {
                    let inner = &body[..i + FN_CLOSE.len()];
                    out.push(self.parse(inner, true, format!("{OPEN}{body}")));
                    return out;
                }
            }
            let raw = if wrapped { format!("{OPEN}{body}") } else { format!("{FN_OPEN}{body}") };
            // Hermes JSON that is complete but missing its closing tag still counts.
            if wrapped && self.dialect == Dialect::Hermes {
                if let Piece::Call(c) = self.parse(&body, true, raw.clone()) {
                    out.push(Piece::Call(c));
                    return out;
                }
            }
            out.push(Piece::BadCall { raw, error: "the tool call was not finished".into() });
        } else if !self.buf.is_empty() && !(self.after_bare && CLOSE.starts_with(self.buf.trim_start())) {
            out.push(Piece::Text(std::mem::take(&mut self.buf)));
        }
        merge(out)
    }

    /// Bytes at the end of the buffer that are a prefix of an opener.
    fn holdback(&self) -> usize {
        let mut openers = vec![OPEN];
        if self.dialect == Dialect::QwenCoder {
            openers.push(FN_OPEN);
        }
        let b = self.buf.as_bytes();
        let mut best = 0;
        for o in openers {
            for k in (1..o.len()).rev() {
                if k <= b.len() && b[b.len() - k..] == o.as_bytes()[..k] {
                    best = best.max(k);
                    break;
                }
            }
        }
        best
    }

    /// Parses a call's inner text. `wrapped` = it was inside <tool_call>.
    fn parse(&self, inner: &str, wrapped: bool, raw: String) -> Piece {
        let t = inner.trim();
        if t.starts_with(FN_OPEN) || !wrapped {
            let body = if wrapped { t } else { &inner[..] };
            return match self.parse_coder(body, wrapped) {
                Ok(c) => Piece::Call(c),
                Err(error) => Piece::BadCall { raw, error },
            };
        }
        match parse_hermes(t) {
            Ok(c) => Piece::Call(c),
            Err(error) => Piece::BadCall { raw, error },
        }
    }

    /// `<function=NAME>` ... `</function>` (with or without the opener already consumed).
    fn parse_coder(&self, body: &str, wrapped: bool) -> Result<ToolCall, String> {
        // Unwrapped: the scanner consumed nothing, so `body` starts at "<function=" and ends before
        // "</function>". Wrapped: `body` is "<function=NAME>...</function>".
        let s = body.trim();
        let s = s.strip_prefix(FN_OPEN).ok_or("expected <function=")?;
        let gt = s.find('>').ok_or("unterminated <function=...>")?;
        let name = s[..gt].trim().to_owned();
        if name.is_empty() {
            return Err("the function has no name".into());
        }
        let mut rest = &s[gt + 1..];
        if wrapped {
            rest = rest.trim_end().strip_suffix(FN_CLOSE).unwrap_or(rest);
        }
        let types = self.types.get(&name);
        let mut args = Map::new();
        let mut r = rest;
        while let Some(i) = r.find("<parameter=") {
            let after = &r[i + "<parameter=".len()..];
            let gt = after.find('>').ok_or("unterminated <parameter=...>")?;
            let key = after[..gt].trim().to_owned();
            let val_start = &after[gt + 1..];
            let end = val_start.find("</parameter>").ok_or_else(|| format!("parameter {key} is not closed"))?;
            let raw_val = val_start[..end].strip_prefix('\n').unwrap_or(&val_start[..end]);
            let raw_val = raw_val.strip_suffix('\n').unwrap_or(raw_val);
            let ty = types.and_then(|t| t.get(&key)).and_then(|p| p.get("type")).and_then(Value::as_str).unwrap_or("string");
            args.insert(key, typed(raw_val, ty));
            r = &val_start[end + "</parameter>".len()..];
        }
        Ok(ToolCall { name, arguments: Value::Object(args) })
    }
}

/// A Qwen3-Coder parameter value converted by its schema type; anything that does not parse stays a string.
fn typed(v: &str, ty: &str) -> Value {
    let t = v.trim();
    match ty {
        "integer" => t.parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::String(v.to_owned())),
        "number" => t.parse::<f64>().ok().and_then(|f| serde_json::Number::from_f64(f).map(Value::Number)).unwrap_or_else(|| Value::String(v.to_owned())),
        "boolean" => match t.to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(v.to_owned()),
        },
        "object" | "array" | "null" => serde_json::from_str(t).unwrap_or_else(|_| Value::String(v.to_owned())),
        _ => Value::String(v.to_owned()),
    }
}

fn parse_hermes(t: &str) -> Result<ToolCall, String> {
    let v: Value = serde_json::from_str(t).map_err(|e| format!("invalid JSON: {e}"))?;
    let name = v.get("name").and_then(Value::as_str).ok_or("the call has no \"name\"")?.to_owned();
    let arguments = match v.get("arguments").or_else(|| v.get("parameters")) {
        None | Some(Value::Null) => json!({}),
        // Some models send the arguments as a JSON string holding JSON.
        Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| format!("arguments are not JSON: {e}"))?,
        Some(other) => other.clone(),
    };
    if !arguments.is_object() {
        return Err("arguments must be a JSON object".into());
    }
    Ok(ToolCall { name, arguments })
}

fn push_text(out: &mut Vec<Piece>, t: &str) {
    if !t.is_empty() {
        out.push(Piece::Text(t.to_owned()));
    }
}

/// Adjacent text pieces become one.
fn merge(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::with_capacity(pieces.len());
    for p in pieces {
        if let (Some(Piece::Text(prev)), Piece::Text(t)) = (out.last_mut(), &p) {
            prev.push_str(t);
            continue;
        }
        out.push(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, props: Value) -> ToolSpec {
        ToolSpec { server: "s".into(), name: name.into(), description: format!("{name} tool"), input_schema: json!({ "type": "object", "properties": props }) }
    }

    fn run(d: Dialect, tools: &[ToolSpec], chunks: &[&str]) -> Vec<Piece> {
        let mut s = Scanner::new(d, tools);
        let mut out = Vec::new();
        for c in chunks {
            out.extend(s.feed(c));
        }
        out.extend(s.finish());
        merge(out)
    }

    fn bytewise(d: Dialect, tools: &[ToolSpec], text: &str) -> Vec<Piece> {
        let chunks: Vec<String> = text.chars().map(String::from).collect();
        run(d, tools, &chunks.iter().map(String::as_str).collect::<Vec<_>>())
    }

    const HERMES: &str = "Let me look.\n<tool_call>\n{\"name\": \"list_directory\", \"arguments\": {\"path\": \"C:/x\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"a.md\\\"}\"}\n</tool_call>";

    #[test]
    fn hermes_two_calls_with_text_before_same_bytewise() {
        let whole = run(Dialect::Hermes, &[], &[HERMES]);
        assert_eq!(whole, bytewise(Dialect::Hermes, &[], HERMES));
        assert_eq!(whole[0], Piece::Text("Let me look.\n".into()));
        assert_eq!(whole[1], Piece::Call(ToolCall { name: "list_directory".into(), arguments: json!({ "path": "C:/x" }) }));
        assert_eq!(whole[2], Piece::Text("\n".into()));
        assert_eq!(whole[3], Piece::Call(ToolCall { name: "read_file".into(), arguments: json!({ "path": "a.md" }) }), "arguments given as a JSON string");
    }

    #[test]
    fn delimiter_split_across_deltas_never_leaks() {
        let out = run(Dialect::Hermes, &[], &["ok <tool_", "call>\n{\"name\":\"x\"}\n</tool", "_call> done"]);
        assert_eq!(out, vec![Piece::Text("ok ".into()), Piece::Call(ToolCall { name: "x".into(), arguments: json!({}) }), Piece::Text(" done".into())]);
    }

    #[test]
    fn a_lone_angle_bracket_is_released() {
        let mut s = Scanner::new(Dialect::Hermes, &[]);
        assert_eq!(s.feed("a <"), vec![Piece::Text("a ".into())]);
        assert_eq!(s.feed(" b"), vec![Piece::Text("< b".into())]);
        assert!(s.finish().is_empty());
    }

    #[test]
    fn coder_call_with_and_without_opener_and_typed_parameters() {
        let tools = [spec("search", json!({ "limit": { "type": "integer" }, "exact": { "type": "boolean" }, "filters": { "type": "object" }, "q": { "type": "string" } }))];
        let wrapped = "<tool_call>\n<function=search>\n<parameter=q>\n42 apples\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=exact>\nTrue\n</parameter>\n<parameter=filters>\n{\"lang\": \"en\"}\n</parameter>\n</function>\n</tool_call>";
        let want = Piece::Call(ToolCall { name: "search".into(), arguments: json!({ "q": "42 apples", "limit": 5, "exact": true, "filters": { "lang": "en" } }) });
        assert_eq!(run(Dialect::QwenCoder, &tools, &[wrapped]), vec![want.clone()]);
        assert_eq!(bytewise(Dialect::QwenCoder, &tools, wrapped), vec![want.clone()]);
        let bare = "Sure.\n<function=search>\n<parameter=q>\n42 apples\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n<parameter=filters>\n{\"lang\": \"en\"}\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(run(Dialect::QwenCoder, &tools, &[bare]), vec![Piece::Text("Sure.\n".into()), want.clone()], "missing opener, stray closer");
        assert_eq!(bytewise(Dialect::QwenCoder, &tools, bare), vec![Piece::Text("Sure.\n".into()), want]);
    }

    #[test]
    fn malformed_and_unterminated_calls_are_bad_calls() {
        let out = run(Dialect::Hermes, &[], &["<tool_call>\n{\"name\": \"x\", \"arguments\": {oops}}\n</tool_call>"]);
        assert!(matches!(&out[0], Piece::BadCall { error, .. } if error.contains("invalid JSON")));
        let out = run(Dialect::Hermes, &[], &["<tool_call>\n{\"name\": \"x\""]);
        assert!(matches!(&out[0], Piece::BadCall { error, .. } if error.contains("not finished")));
        let out = run(Dialect::Hermes, &[], &["<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n"]);
        assert_eq!(out, vec![Piece::Call(ToolCall { name: "x".into(), arguments: json!({}) })], "complete JSON missing only the closing tag");
        let out = run(Dialect::Hermes, &[], &["<tool_call>{\"arguments\": {}}</tool_call>"]);
        assert!(matches!(&out[0], Piece::BadCall { error, .. } if error.contains("name")));
    }

    #[test]
    fn plain_text_with_multibyte_characters_passes_through() {
        let text = "Grüße — 東京 < 5 and <b>bold</b> stays.";
        assert_eq!(bytewise(Dialect::QwenCoder, &[], text), vec![Piece::Text(text.into())]);
    }

    #[test]
    fn prompt_lists_every_tool_as_one_json_line() {
        let tools = [spec("read_file", json!({ "path": { "type": "string" } })), spec("list_directory", json!({}))];
        for d in [Dialect::Hermes, Dialect::QwenCoder] {
            let p = tools_prompt(d, &tools);
            let inner = &p[p.find("<tools>\n").unwrap() + 8..p.find("\n</tools>").unwrap()];
            let names: Vec<String> = inner.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["function"]["name"].as_str().unwrap().to_owned()).collect();
            assert_eq!(names, ["read_file", "list_directory"]);
        }
        assert!(tools_prompt(Dialect::QwenCoder, &tools).contains("Do NOT omit the initial <tool_call> tag"));
        assert_eq!(dialect_from_template("{% if x %}<function={{ name }}>"), Dialect::QwenCoder);
        assert_eq!(dialect_from_template("<tool_call>\\n{{ tool | tojson }}"), Dialect::Hermes);
    }

    #[test]
    fn rendered_calls_read_back_unchanged_in_both_dialects() {
        let tools = [spec("search", json!({ "limit": { "type": "integer" }, "exact": { "type": "boolean" }, "filters": { "type": "object" }, "q": { "type": "string" } }))];
        let call = ToolCall { name: "search".into(), arguments: json!({ "q": "two\nlines", "limit": 5, "exact": false, "filters": { "lang": "en" } }) };
        for d in [Dialect::Hermes, Dialect::QwenCoder] {
            let text = format!("Checking.\n{}", render_call(d, &call));
            assert_eq!(run(d, &tools, &[&text]), vec![Piece::Text("Checking.\n".into()), Piece::Call(call.clone())], "{d:?}: {text}");
        }
    }

    #[test]
    fn results_are_wrapped_and_truncated_on_characters() {
        assert_eq!(tool_response("ok"), "<tool_response>\nok\n</tool_response>");
        assert_eq!(truncate_result("ééééé", 3), "ééé\n[... truncated 2 characters]");
        assert_eq!(truncate_result("abc", 3), "abc");
    }
}
