//! The effort and verbosity dials applied to a chat request (`docs/specs/effort-levels.md`).
//!
//! A request may carry `cerebro_effort` and `cerebro_verbosity`; either missing takes the settings
//! default (Medium and Brief out of the box). Both fields are removed before the request reaches
//! llama-server. The gateway then:
//!
//! 1. appends the level's instruction and the verbosity instruction to the system prompt
//!    ([`instruction_text`], [`inject_system`]): to the first `system`/`developer` message when
//!    there is one, otherwise as a new leading `system` message;
//! 2. maps the level's thinking budget onto llama-server's per-request
//!    `reasoning_budget_tokens` ([`plan_budget`]): `0` for Off, a token count otherwise. llama.cpp
//!    applies it only when the model's chat template has thinking tags; the reply's `cerebro`
//!    field says whether reasoning came back;
//! 3. guards the reasoning trap measured on this machine (thinking eats the whole token budget and
//!    nothing is answered): the answer budget is reserved separately. A client `max_tokens` is the
//!    answer budget, so llama-server is sent `max_tokens = thinking + answer`; and the thinking
//!    budget is clamped so prompt + thinking + answer fit the context window when its size is known.
//!
//! The self-review passes of XHigh and Max are in [`crate::passes`].
//!
//! Cost of step 1: llama-server reuses its cached prompt only up to the first token that differs,
//! and the instructions sit at the end of the system prompt, near the start of the conversation.
//! While the effort and verbosity levels stay the same every turn reuses the cache as usual; a
//! turn that changes either level differs from that point on, so the whole conversation after the
//! system prompt is prefilled again once (the spec requires the instructions there).

use cerebro_config::{EffortLevel, EffortSection, ThinkingBudget, VerbosityLevel, VerbositySection};
use serde_json::Value;

use crate::error::ApiError;

/// Request field naming the effort level (a level id such as `"high"`, or its display name).
pub const EFFORT_FIELD: &str = "cerebro_effort";

/// Request field naming the verbosity level (`"brief"`, `"normal"`, `"detailed"`, or any level id
/// or name configured in settings).
pub const VERBOSITY_FIELD: &str = "cerebro_verbosity";

/// Thinking budget used for [`ThinkingBudget::ModelMax`] when the context size is unknown, so the
/// answer reservation still holds.
pub const MODEL_MAX_FALLBACK_TOKENS: u32 = 32_768;

/// Tokens allowed per message for chat-template framing (role markers, separators) when bounding
/// a prompt's size.
pub const TEMPLATE_OVERHEAD_PER_MESSAGE: u64 = 16;

/// The effort and verbosity settings a gateway serves with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dials {
    /// The effort dial: levels, default level and answer reserve.
    pub effort: EffortSection,
    /// The verbosity dial: levels and default level.
    pub verbosity: VerbositySection,
}

/// The levels resolved for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Settings id of the effort level, such as `"xhigh"`.
    pub effort_id: String,
    /// The effort level's settings.
    pub effort: EffortLevel,
    /// Settings id of the verbosity level, such as `"brief"`.
    pub verbosity_id: String,
    /// The verbosity level's settings.
    pub verbosity: VerbosityLevel,
    /// Answer tokens reserved apart from thinking when the client gives no `max_tokens`.
    pub answer_reserve_tokens: u32,
}

impl Plan {
    /// Self-review passes after the draft (XHigh and Max: one by default).
    pub fn review_passes(&self) -> u8 {
        self.effort.review_passes
    }

    /// Whether an independent check runs after the self-review (Max by default).
    pub fn independent_check(&self) -> bool {
        self.effort.independent_check
    }

    /// Whether any pass beyond the draft runs, so the draft cannot be streamed as it is generated.
    pub fn has_passes(&self) -> bool {
        self.review_passes() > 0 || self.independent_check()
    }
}

/// Find a level by settings id, or by id or display name ignoring ASCII case.
fn find_level<'a, L>(
    levels: &'a std::collections::BTreeMap<String, L>,
    wanted: &str,
    name_of: impl Fn(&L) -> &str,
) -> Option<(&'a String, &'a L)> {
    levels.get_key_value(wanted).or_else(|| {
        levels
            .iter()
            .find(|(id, l)| id.eq_ignore_ascii_case(wanted) || name_of(l).eq_ignore_ascii_case(wanted))
    })
}

/// Read and remove a string field; a present non-string value is a client error.
fn take_str(body: &mut serde_json::Map<String, Value>, field: &str) -> Result<Option<String>, ApiError> {
    match body.remove(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

/// Remove `cerebro_effort` and `cerebro_verbosity` from `body` and resolve them against `dials`.
///
/// Verbosity `"normal"` names the settings level `"standard"` unless a level called `"normal"` is
/// configured.
///
/// # Errors
/// 400 when a field is not a string or names no configured level (the message lists the levels);
/// 500 when the settings default names no level (settings validation prevents this).
pub fn resolve(dials: &Dials, body: &mut serde_json::Map<String, Value>) -> Result<Plan, ApiError> {
    let effort_req = take_str(body, EFFORT_FIELD)?;
    let verbosity_req = take_str(body, VERBOSITY_FIELD)?;

    let effort_wanted = effort_req.as_deref().unwrap_or(&dials.effort.default_level);
    let (effort_id, effort) = find_level(&dials.effort.levels, effort_wanted, |l| l.name.as_str())
        .ok_or_else(|| unknown_level(EFFORT_FIELD, effort_wanted, dials.effort.levels.keys()))?;

    let verbosity_wanted = verbosity_req.as_deref().unwrap_or(&dials.verbosity.default_level);
    let verbosity_found = find_level(&dials.verbosity.levels, verbosity_wanted, |l| l.name.as_str())
        .or_else(|| {
            if verbosity_wanted.eq_ignore_ascii_case("normal") {
                dials.verbosity.levels.get_key_value("standard")
            } else {
                None
            }
        });
    let (verbosity_id, verbosity) = verbosity_found
        .ok_or_else(|| unknown_level(VERBOSITY_FIELD, verbosity_wanted, dials.verbosity.levels.keys()))?;

    Ok(Plan {
        effort_id: effort_id.clone(),
        effort: effort.clone(),
        verbosity_id: verbosity_id.clone(),
        verbosity: verbosity.clone(),
        answer_reserve_tokens: dials.effort.answer_reserve_tokens,
    })
}

fn unknown_level<'a>(field: &str, wanted: &str, ids: impl Iterator<Item = &'a String>) -> ApiError {
    let ids: Vec<&str> = ids.map(String::as_str).collect();
    // The requested value is echoed only when it is short and plain, so a pasted prompt cannot be
    // reflected back through this message.
    let shown = if wanted.len() <= 32 && wanted.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
        format!("{wanted:?}")
    } else {
        "the requested value".to_string()
    };
    ApiError::bad_request(format!("{field}: {shown} is not a configured level; use one of {}", ids.join(", ")))
}

/// The text appended to the system prompt: the effort instruction, a blank line, then the
/// verbosity instruction. Empty instructions (a user may blank one in settings) are left out.
pub fn instruction_text(plan: &Plan) -> String {
    [plan.effort.instruction.trim(), plan.verbosity.instruction.trim()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Append `text` to the conversation's system prompt.
///
/// The first message, when its role is `system` or `developer`, gets `"\n\n" + text` appended
/// (string content), or a trailing `{"type":"text","text":text}` part (array content). Otherwise a
/// `{"role":"system","content":text}` message is inserted first. Empty `text` changes nothing.
///
/// # Errors
/// 400 when the first system message's content is neither a string, an array nor null.
pub fn inject_system(messages: &mut Vec<Value>, text: &str) -> Result<(), ApiError> {
    if text.is_empty() {
        return Ok(());
    }
    let first_is_system = messages.first().and_then(|m| m.get("role")).and_then(Value::as_str)
        .is_some_and(|role| role == "system" || role == "developer");
    if !first_is_system {
        messages.insert(0, serde_json::json!({ "role": "system", "content": text }));
        return Ok(());
    }
    let Some(first) = messages.first_mut().and_then(Value::as_object_mut) else {
        return Err(ApiError::bad_request("the first message must be an object"));
    };
    match first.get_mut("content") {
        None | Some(Value::Null) => {
            first.insert("content".to_string(), Value::String(text.to_string()));
        }
        Some(Value::String(existing)) => {
            if existing.trim().is_empty() {
                *existing = text.to_string();
            } else {
                existing.truncate(existing.trim_end().len());
                existing.push_str("\n\n");
                existing.push_str(text);
            }
        }
        Some(Value::Array(parts)) => {
            parts.push(serde_json::json!({ "type": "text", "text": text }));
        }
        Some(_) => return Err(ApiError::bad_request("system message content must be a string or an array")),
    }
    Ok(())
}

/// Token limits for one upstream call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Sent as `reasoning_budget_tokens`: `0` ends thinking at once.
    pub thinking_tokens: u32,
    /// The answer budget: the client's `max_tokens`, or `None` when the client set none (the
    /// answer is then bounded only by the context window, which the clamp below keeps free).
    pub answer_tokens: Option<u64>,
    /// Sent as `max_tokens`: thinking plus the client's answer budget; `None` leaves it unset.
    pub max_tokens: Option<u64>,
    /// Whether the thinking budget was lowered to fit the context window.
    pub clamped_to_context: bool,
}

/// What is known about the context window for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextFit {
    /// The server's per-slot context size in tokens.
    pub n_ctx: u64,
    /// An upper bound on the prompt's size in tokens, template framing included.
    pub prompt_tokens: u64,
}

/// Work out the token limits for a plan.
///
/// `client_max` is the request's `max_tokens` (or `max_completion_tokens`). The thinking budget is
/// the level's, or for [`ThinkingBudget::ModelMax`] whatever the context leaves after the prompt
/// and the answer reserve ([`MODEL_MAX_FALLBACK_TOKENS`] when the context is unknown). When the
/// context is known, thinking is clamped so `prompt + thinking + answer <= n_ctx`, where the answer
/// is the client's budget or the configured reserve.
pub fn plan_budget(plan: &Plan, client_max: Option<u64>, context: Option<ContextFit>) -> Budget {
    let answer_room = client_max.unwrap_or(u64::from(plan.answer_reserve_tokens));
    let free_for_thinking = context.map(|c| c.n_ctx.saturating_sub(c.prompt_tokens).saturating_sub(answer_room));
    let wanted: u64 = match plan.effort.thinking_budget {
        ThinkingBudget::Off => 0,
        ThinkingBudget::Tokens(n) => u64::from(n),
        ThinkingBudget::ModelMax => free_for_thinking.unwrap_or(u64::from(MODEL_MAX_FALLBACK_TOKENS)),
    };
    let (thinking, clamped) = match free_for_thinking {
        Some(free) if wanted > free => (free, true),
        _ => (wanted, false),
    };
    let thinking_tokens = u32::try_from(thinking).unwrap_or(u32::MAX);
    Budget {
        thinking_tokens,
        answer_tokens: client_max,
        max_tokens: client_max.map(|answer| answer.saturating_add(u64::from(thinking_tokens))),
        clamped_to_context: clamped,
    }
}

/// Tokens allowed for one image, audio or file part of a message (the multimodal projector's
/// output, not the bytes of its base64 data). Generous: common vision projectors emit a few hundred
/// to about two thousand tokens per image.
pub const MEDIA_PART_TOKENS_BOUND: u64 = 4096;

/// Tokens allowed for the chat template's own text around tool definitions (the "you may call these
/// functions" preamble and the per-tool framing), on top of the tool schemas themselves.
pub const TOOLS_TEMPLATE_OVERHEAD: u64 = 1024;

/// Whether `v` is a media content part (`image_url`, `input_audio` or `file`), which the model sees
/// as projector embeddings, never as the text of its data.
fn is_media_part(v: &Value) -> bool {
    matches!(v.get("type").and_then(Value::as_str), Some("image_url" | "input_audio" | "file"))
}

/// One token per UTF-8 byte of every string in `v`, with each media part counted as
/// [`MEDIA_PART_TOKENS_BOUND`] instead of its bytes.
fn string_bytes(v: &Value) -> u64 {
    match v {
        Value::String(s) => s.len() as u64,
        Value::Object(_) if is_media_part(v) => MEDIA_PART_TOKENS_BOUND,
        Value::Array(items) => items.iter().map(string_bytes).fold(0, u64::saturating_add),
        Value::Object(map) => map.values().map(string_bytes).fold(0, u64::saturating_add),
        _ => 0,
    }
}

/// An upper bound on a conversation's prompt size in tokens without tokenizing it: one token per
/// UTF-8 byte of every string in the messages (byte-level tokenizers never produce more), each
/// media part counted as [`MEDIA_PART_TOKENS_BOUND`], plus [`TEMPLATE_OVERHEAD_PER_MESSAGE`] per
/// message. Tool definitions are counted separately by [`tools_bytes_bound`].
pub fn prompt_bytes_bound(messages: &[Value]) -> u64 {
    messages
        .iter()
        .map(|m| string_bytes(m).saturating_add(TEMPLATE_OVERHEAD_PER_MESSAGE))
        .fold(0u64, u64::saturating_add)
}

/// The fields of a chat request the template renders besides `messages`: tool definitions and the
/// tool choice.
const TOOL_FIELDS: [&str; 2] = ["tools", "tool_choice"];

/// Counts the bytes written to it, so a value's serialized size is known without allocating it.
struct ByteCounter(u64);

impl std::io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// An upper bound on the tokens the request's tool definitions add to the prompt: twice their
/// compact JSON size (templates render them as JSON, often with `", "`/`": "` separators or
/// indentation) plus [`TOOLS_TEMPLATE_OVERHEAD`]; `0` when the request has none.
pub fn tools_bytes_bound(body: &serde_json::Map<String, Value>) -> u64 {
    let mut json_bytes = 0u64;
    for field in TOOL_FIELDS {
        if let Some(v) = body.get(field).filter(|v| !v.is_null()) {
            let mut counter = ByteCounter(0);
            // Writing a `Value` into a counter cannot fail; a failure would only undercount by
            // what was not written, so the template overhead below still applies.
            let _ = serde_json::to_writer(&mut counter, v);
            json_bytes = json_bytes.saturating_add(counter.0);
        }
    }
    if json_bytes == 0 {
        return 0;
    }
    json_bytes.saturating_mul(2).saturating_add(TOOLS_TEMPLATE_OVERHEAD)
}

/// The tool definitions as text for tokenizing (pretty-printed, which tokenizes to at least as many
/// tokens as the compact forms templates use), or `None` when the request has none.
pub fn tools_text(body: &serde_json::Map<String, Value>) -> Option<String> {
    let mut out = String::new();
    for field in TOOL_FIELDS {
        if let Some(v) = body.get(field).filter(|v| !v.is_null()) {
            out.push_str(&serde_json::to_string_pretty(v).unwrap_or_default());
            out.push('\n');
        }
    }
    (!out.is_empty()).then_some(out)
}

/// How many media parts (`image_url`, `input_audio`, `file`) the messages carry.
pub fn media_parts(messages: &[Value]) -> u64 {
    fn count(v: &Value) -> u64 {
        match v {
            Value::Object(_) if is_media_part(v) => 1,
            Value::Array(items) => items.iter().map(count).fold(0, u64::saturating_add),
            Value::Object(map) => map.values().map(count).fold(0, u64::saturating_add),
            _ => 0,
        }
    }
    messages.iter().map(count).fold(0, u64::saturating_add)
}

/// The prompt bound of the two turns a review or correction pass appends to the conversation: the
/// answer as an assistant turn and `instruction` as a user turn. The answer counts as the smaller of
/// its byte length and `completion_tokens` (both upper bounds on its tokens; the reasoning that
/// produced it is not sent back).
pub fn follow_up_bound(answer: &str, completion_tokens: Option<u64>, instruction: &str) -> u64 {
    let answer_tokens = completion_tokens.map_or(answer.len() as u64, |c| c.min(answer.len() as u64));
    let framing = ("assistant".len() + "user".len()) as u64 + 2 * TEMPLATE_OVERHEAD_PER_MESSAGE;
    answer_tokens.saturating_add(instruction.len() as u64).saturating_add(framing)
}

/// Whether a chat template has a thinking mode, judged from its source as `/props` returns it:
/// any mention of thinking or reasoning (`<think>`, `enable_thinking`, `[THINK]`,
/// `reasoning_content`, a `thinking` field). llama.cpp applies a reasoning budget only to templates
/// with thinking tags, so for a template without one the thinking budget would be ignored, and
/// adding it to `max_tokens` would let the answer run past the client's own limit.
pub fn template_supports_thinking(template: &str) -> bool {
    let lower = template.to_ascii_lowercase();
    lower.contains("think") || lower.contains("reasoning")
}

/// The text of every message, joined, for tokenizing a prompt whose byte bound does not fit. Media
/// parts are left out (their data is not text the model reads); count them with [`media_parts`].
pub fn prompt_text(messages: &[Value]) -> String {
    fn collect(v: &Value, out: &mut String) {
        match v {
            Value::String(s) => {
                out.push_str(s);
                out.push('\n');
            }
            Value::Object(_) if is_media_part(v) => {}
            Value::Array(items) => items.iter().for_each(|i| collect(i, out)),
            Value::Object(map) => map.values().for_each(|i| collect(i, out)),
            _ => {}
        }
    }
    let mut out = String::new();
    messages.iter().for_each(|m| collect(m, &mut out));
    out
}

/// Write the budget into the upstream body: `reasoning_budget_tokens` always, `max_tokens` when the
/// client gave one; the client's own `max_completion_tokens`, `n_predict` and thinking-budget
/// fields are removed so llama-server sees one consistent limit.
pub fn apply_budget(body: &mut serde_json::Map<String, Value>, budget: &Budget) {
    for field in ["max_completion_tokens", "n_predict", "thinking_budget_tokens", "max_tokens"] {
        body.remove(field);
    }
    body.insert("reasoning_budget_tokens".to_string(), Value::from(budget.thinking_tokens));
    if let Some(max) = budget.max_tokens {
        body.insert("max_tokens".to_string(), Value::from(max));
    }
}

/// Whether the request turned thinking off for itself (`chat_template_kwargs.enable_thinking:
/// false`). The model will not think, so no thinking budget may be added to its `max_tokens`.
pub fn client_disabled_thinking(body: &serde_json::Map<String, Value>) -> bool {
    body.get("chat_template_kwargs")
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(Value::as_bool)
        == Some(false)
}

/// The client's answer budget: `max_tokens`, else `max_completion_tokens`, else `n_predict`;
/// absent, null or negative (llama.cpp's "unlimited") means none.
///
/// # Errors
/// 400 when the field is present but not an integer.
pub fn client_max_tokens(body: &serde_json::Map<String, Value>) -> Result<Option<u64>, ApiError> {
    for field in ["max_tokens", "max_completion_tokens", "n_predict"] {
        match body.get(field) {
            None | Some(Value::Null) => continue,
            Some(v) => {
                if let Some(n) = v.as_u64() {
                    return Ok(Some(n));
                }
                if v.as_i64().is_some_and(|n| n < 0) {
                    return Ok(None);
                }
                return Err(ApiError::bad_request(format!("{field} must be an integer")));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(v: Value) -> serde_json::Map<String, Value> {
        v.as_object().cloned().expect("object")
    }

    #[test]
    fn defaults_are_settings_default_effort_and_brief() {
        let dials = Dials::default();
        let mut b = body(json!({"messages": []}));
        let plan = resolve(&dials, &mut b).expect("resolve");
        assert_eq!(plan.effort_id, "medium");
        assert_eq!(plan.verbosity_id, "brief");

        let mut custom = Dials::default();
        custom.effort.default_level = "high".to_string();
        let plan = resolve(&custom, &mut body(json!({}))).expect("resolve");
        assert_eq!(plan.effort_id, "high");
    }

    #[test]
    fn request_fields_select_levels_and_are_removed() {
        let dials = Dials::default();
        let mut b = body(json!({"cerebro_effort": "XHigh", "cerebro_verbosity": "normal", "messages": []}));
        let plan = resolve(&dials, &mut b).expect("resolve");
        assert_eq!(plan.effort_id, "xhigh");
        assert_eq!(plan.verbosity_id, "standard");
        assert!(!b.contains_key(EFFORT_FIELD) && !b.contains_key(VERBOSITY_FIELD));
        assert!(plan.has_passes());
        assert_eq!(plan.review_passes(), 1);
        assert!(!plan.independent_check());

        let plan = resolve(&dials, &mut body(json!({"cerebro_effort": "max", "cerebro_verbosity": "detailed"})))
            .expect("resolve");
        assert!(plan.independent_check());
        assert_eq!(plan.verbosity_id, "detailed");
    }

    #[test]
    fn unknown_or_mistyped_levels_are_client_errors() {
        let dials = Dials::default();
        let err = resolve(&dials, &mut body(json!({"cerebro_effort": "ultra"}))).unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("\"ultra\"") && err.message.contains("xhigh"), "{}", err.message);
        let err = resolve(&dials, &mut body(json!({"cerebro_verbosity": 3}))).unwrap_err();
        assert!(err.message.contains("must be a string"));
        // A long value is never echoed back.
        let long = "x".repeat(200);
        let err = resolve(&dials, &mut body(json!({"cerebro_effort": long}))).unwrap_err();
        assert!(!err.message.contains("xxxxxxxx"), "{}", err.message);
    }

    #[test]
    fn instruction_text_is_effort_then_verbosity_exactly() {
        let dials = Dials::default();
        let plan = resolve(&dials, &mut body(json!({"cerebro_effort": "low"}))).expect("resolve");
        assert_eq!(
            instruction_text(&plan),
            "Be fast and direct. Use what you already know. Only stop to double-check when an error \
             would have real consequences, such as money, safety or commands that change files.\n\n\
             Lead with the answer in one or two sentences. Add only the detail needed to act on it, \
             and offer to expand rather than expanding unasked."
        );
    }

    #[test]
    fn injection_appends_to_existing_system_or_inserts_one() {
        let mut msgs = vec![json!({"role": "user", "content": "hi"})];
        inject_system(&mut msgs, "RULES").expect("inject");
        assert_eq!(msgs[0], json!({"role": "system", "content": "RULES"}));
        assert_eq!(msgs[1]["role"], "user");

        let mut msgs = vec![json!({"role": "system", "content": "You are terse.\n "}), json!({"role": "user", "content": "hi"})];
        inject_system(&mut msgs, "RULES").expect("inject");
        assert_eq!(msgs[0]["content"], "You are terse.\n\nRULES");
        assert_eq!(msgs.len(), 2);

        let mut msgs = vec![json!({"role": "developer", "content": [{"type": "text", "text": "A"}]})];
        inject_system(&mut msgs, "RULES").expect("inject");
        assert_eq!(msgs[0]["content"], json!([{"type": "text", "text": "A"}, {"type": "text", "text": "RULES"}]));

        let mut msgs = vec![json!({"role": "system", "content": 5})];
        assert!(inject_system(&mut msgs, "RULES").is_err());
    }

    fn plan_with(budget: ThinkingBudget) -> Plan {
        let dials = Dials::default();
        let mut plan = resolve(&dials, &mut body(json!({}))).expect("resolve");
        plan.effort.thinking_budget = budget;
        plan
    }

    #[test]
    fn answer_budget_is_reserved_apart_from_thinking() {
        // Client max_tokens is the answer budget; thinking is added on top.
        let b = plan_budget(&plan_with(ThinkingBudget::Tokens(1024)), Some(200), None);
        assert_eq!((b.thinking_tokens, b.max_tokens, b.answer_tokens), (1024, Some(1224), Some(200)));
        // No client limit: max_tokens stays unset, thinking bounded by its own budget.
        let b = plan_budget(&plan_with(ThinkingBudget::Tokens(1024)), None, None);
        assert_eq!((b.thinking_tokens, b.max_tokens), (1024, None));
        // Off sends 0.
        let b = plan_budget(&plan_with(ThinkingBudget::Off), Some(50), None);
        assert_eq!((b.thinking_tokens, b.max_tokens), (0, Some(50)));
    }

    #[test]
    fn thinking_is_clamped_so_the_answer_still_fits_the_context() {
        let ctx = ContextFit { n_ctx: 4096, prompt_tokens: 1000 };
        // 4096 - 1000 - 2048 (reserve) = 1048 left for thinking out of 16384 wanted.
        let b = plan_budget(&plan_with(ThinkingBudget::Tokens(16384)), None, Some(ctx));
        assert_eq!(b.thinking_tokens, 1048);
        assert!(b.clamped_to_context);
        // With a client answer budget of 500: 4096 - 1000 - 500 = 2596.
        let b = plan_budget(&plan_with(ThinkingBudget::Tokens(16384)), Some(500), Some(ctx));
        assert_eq!((b.thinking_tokens, b.max_tokens), (2596, Some(3096)));
        // ModelMax takes what is free; a prompt that leaves nothing gives 0, never a negative.
        let b = plan_budget(&plan_with(ThinkingBudget::ModelMax), None, Some(ctx));
        assert_eq!(b.thinking_tokens, 1048);
        let full = ContextFit { n_ctx: 4096, prompt_tokens: 4000 };
        let b = plan_budget(&plan_with(ThinkingBudget::ModelMax), None, Some(full));
        assert_eq!(b.thinking_tokens, 0);
        // Unknown context: ModelMax uses the documented fallback.
        let b = plan_budget(&plan_with(ThinkingBudget::ModelMax), None, None);
        assert_eq!(b.thinking_tokens, MODEL_MAX_FALLBACK_TOKENS);
        // A budget that fits is not clamped.
        let b = plan_budget(&plan_with(ThinkingBudget::Tokens(512)), None, Some(ctx));
        assert_eq!((b.thinking_tokens, b.clamped_to_context), (512, false));
    }

    #[test]
    fn apply_budget_leaves_one_consistent_limit() {
        let mut b = body(json!({"max_completion_tokens": 10, "n_predict": 5, "thinking_budget_tokens": 9}));
        apply_budget(&mut b, &Budget { thinking_tokens: 7, answer_tokens: Some(10), max_tokens: Some(17), clamped_to_context: false });
        assert_eq!(Value::Object(b), json!({"reasoning_budget_tokens": 7, "max_tokens": 17}));
        let mut b = body(json!({"max_tokens": -1}));
        apply_budget(&mut b, &Budget { thinking_tokens: 0, answer_tokens: None, max_tokens: None, clamped_to_context: false });
        assert_eq!(Value::Object(b), json!({"reasoning_budget_tokens": 0}));
    }

    #[test]
    fn client_max_tokens_reads_each_alias() {
        assert_eq!(client_max_tokens(&body(json!({"max_tokens": 9}))).expect("ok"), Some(9));
        assert_eq!(client_max_tokens(&body(json!({"max_completion_tokens": 8}))).expect("ok"), Some(8));
        assert_eq!(client_max_tokens(&body(json!({"n_predict": -1}))).expect("ok"), None);
        assert_eq!(client_max_tokens(&body(json!({}))).expect("ok"), None);
        assert!(client_max_tokens(&body(json!({"max_tokens": "9"}))).is_err());
    }

    #[test]
    fn thinking_templates_are_recognised() {
        assert!(template_supports_thinking("{% if enable_thinking %}<think>{% endif %}"));
        assert!(template_supports_thinking("[THINK]{{ m.content }}[/THINK]"));
        assert!(template_supports_thinking("{{ message.reasoning_content }}"));
        assert!(!template_supports_thinking("<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>"));
    }

    #[test]
    fn prompt_bound_counts_bytes_and_framing() {
        let msgs = vec![json!({"role": "user", "content": "héllo"}), json!({"role": "system", "content": [{"type": "text", "text": "ab"}]})];
        // "user"(4) + "héllo"(6) + 16, "system"(6) + "text"(4) + "ab"(2) + 16
        assert_eq!(prompt_bytes_bound(&msgs), 4 + 6 + 16 + 6 + 4 + 2 + 16);
        assert!(prompt_text(&msgs).contains("héllo"));
    }

    #[test]
    fn media_parts_count_a_fixed_bound_and_are_never_tokenized_as_text() {
        let image = format!("data:image/png;base64,{}", "A".repeat(1 << 20));
        let msgs = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "input_audio", "input_audio": {"data": "UklGR", "format": "wav"}},
        ]})];
        // "user"(4) + "text"(4) + "what is this"(12) + two media parts + framing.
        assert_eq!(prompt_bytes_bound(&msgs), 4 + 4 + 12 + 2 * MEDIA_PART_TOKENS_BOUND + TEMPLATE_OVERHEAD_PER_MESSAGE);
        assert_eq!(media_parts(&msgs), 2);
        let text = prompt_text(&msgs);
        assert!(text.contains("what is this"));
        assert!(!text.contains("AAAA") && !text.contains("UklGR") && !text.contains("base64"), "{} bytes", text.len());
    }

    #[test]
    fn tool_definitions_are_bounded_and_tokenized() {
        assert_eq!(tools_bytes_bound(&body(json!({"messages": []}))), 0);
        assert_eq!(tools_text(&body(json!({"tools": null}))), None);
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
        let compact = serde_json::to_string(&tools).expect("json").len() as u64;
        let b = body(json!({"tools": tools, "tool_choice": "auto"}));
        assert_eq!(tools_bytes_bound(&b), 2 * (compact + "\"auto\"".len() as u64) + TOOLS_TEMPLATE_OVERHEAD);
        let text = tools_text(&b).expect("text");
        assert!(text.contains("get_weather") && text.contains("properties") && text.contains("auto"));
    }

    #[test]
    fn a_follow_up_counts_the_smaller_answer_bound_plus_the_instruction() {
        let framing = 9 + 4 + 2 * TEMPLATE_OVERHEAD_PER_MESSAGE;
        assert_eq!(follow_up_bound("0123456789", Some(3), "fix"), 3 + 3 + framing);
        assert_eq!(follow_up_bound("0123456789", Some(500), "fix"), 10 + 3 + framing);
        assert_eq!(follow_up_bound("0123456789", None, "fix"), 10 + 3 + framing);
    }

    #[test]
    fn a_client_thinking_opt_out_is_recognised() {
        assert!(client_disabled_thinking(&body(json!({"chat_template_kwargs": {"enable_thinking": false}}))));
        assert!(!client_disabled_thinking(&body(json!({"chat_template_kwargs": {"enable_thinking": true}}))));
        assert!(!client_disabled_thinking(&body(json!({"chat_template_kwargs": {}}))));
        assert!(!client_disabled_thinking(&body(json!({}))));
    }
}
