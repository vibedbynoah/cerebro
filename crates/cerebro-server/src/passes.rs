//! The extra passes XHigh and Max run before an answer is returned (`docs/specs/effort-levels.md`,
//! "Mechanics").
//!
//! In order:
//!
//! 1. **Draft**: the request as the client sent it (with the effort and verbosity instructions
//!    already in the system prompt), non-streaming.
//! 2. **Self-review**, `review_passes` times: the same conversation plus the draft as an assistant
//!    turn and [`REVIEW_INSTRUCTION`] as a user turn. The revised answer replaces the draft.
//! 3. **Independent check** (Max): a fresh context that holds only [`CHECK_SYSTEM`], the user's
//!    last message and the answer, asking for any error ([`CHECK_INSTRUCTION`]). Not the original
//!    conversation, so it cannot repeat the draft's reasoning; and without the client's tools and
//!    output format ([`CHECK_STRIPPED`]), so it can answer in prose.
//! 4. **Correction**, only when the check reports errors: the conversation plus the answer and the
//!    findings ([`CORRECTION_TEMPLATE`]); the corrected answer is returned. A check that gives no
//!    text (an empty reply, thinking that used its whole budget) is inconclusive: no correction.
//!
//! Each pass is its own upstream request with its own token budget (see [`run`]), so the
//! reasoning-trap guard holds for the longer review prompts too.
//!
//! A draft that is a tool call or has no text is returned as it is: there is no prose to review.
//! Every pass's wall time is recorded ([`PassRecord`]) and reported in the reply's `cerebro` field.

use std::future::Future;
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::effort::{self, ContextFit, Plan};
use crate::error::ApiError;

/// The user turn that asks for a self-review of the assistant turn before it.
pub const REVIEW_INSTRUCTION: &str = "Review your answer above critically. Check every claim, \
number, calculation and line of code, and fix any mistake you find. Reply with the corrected final \
answer only, written for the user, without commentary about the review.";

/// System prompt of the independent check's fresh context.
pub const CHECK_SYSTEM: &str = "You are an independent checker. You did not write the answer you \
are given, and you have not seen how it was produced.";

/// The instruction closing the independent check's user turn. The verdict comes alone on the
/// first line, so a small model that adds an explanation after a clean verdict is still read as
/// clean ([`is_clean`]).
pub const CHECK_INSTRUCTION: &str = "Find any error in this answer: a wrong fact, number, \
calculation, line of code or unsupported claim. On the first line write exactly NO ERRORS FOUND \
or ERRORS FOUND. If there are errors, list each one briefly on the following lines.";

/// The verdict the independent check gives when it finds nothing.
pub const CHECK_CLEAN: &str = "NO ERRORS FOUND";

/// The user turn that hands the check's findings back; `{findings}` is replaced by them.
pub const CORRECTION_TEMPLATE: &str = "An independent check found these problems in your answer:\n\n\
{findings}\n\nFix every real problem and reply with the corrected final answer only, written for \
the user. If a reported problem is not real, keep that part unchanged.";

/// Which pass a record describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    /// The first answer.
    Draft,
    /// A self-review of the current answer.
    SelfReview,
    /// The independent check in a fresh context.
    IndependentCheck,
    /// The fix after the independent check reported errors.
    Correction,
}

impl PassKind {
    /// Stable name used in the `cerebro` field.
    pub fn as_str(self) -> &'static str {
        match self {
            PassKind::Draft => "draft",
            PassKind::SelfReview => "self_review",
            PassKind::IndependentCheck => "independent_check",
            PassKind::Correction => "correction",
        }
    }
}

/// One pass's timing and outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct PassRecord {
    /// Which pass.
    pub kind: PassKind,
    /// Wall time of the upstream call, in milliseconds.
    pub ms: u64,
    /// Decode speed llama-server reported for the pass (`timings.predicted_per_second`).
    pub tokens_per_second: Option<f64>,
    /// For the independent check: whether it reported errors; `None` when it gave no reviewable
    /// verdict (inconclusive, so no correction ran).
    pub found_errors: Option<bool>,
}

impl PassRecord {
    /// The record as it appears in the `cerebro` field.
    pub fn to_json(&self) -> Value {
        let mut v = json!({ "pass": self.kind.as_str(), "ms": self.ms });
        if let Some(tps) = self.tokens_per_second {
            v["tokens_per_second"] = json!(tps);
        }
        // The check always reports its verdict: true, false, or null when it was inconclusive.
        if self.found_errors.is_some() || self.kind == PassKind::IndependentCheck {
            v["found_errors"] = json!(self.found_errors);
        }
        v
    }
}

/// What the passes produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PassOutcome {
    /// The response to return: the last answer-producing pass's response with the final answer as
    /// its message content and `usage` summed over every pass.
    pub response: Value,
    /// Every pass run, in order.
    pub passes: Vec<PassRecord>,
    /// Why review was skipped after the draft, if it was.
    pub skipped: Option<&'static str>,
}

/// Sends one non-streaming chat completion upstream and returns the parsed response.
pub trait Completer {
    /// Complete `body` (a chat completions request with `"stream": false`).
    fn complete(&self, body: &Map<String, Value>) -> impl Future<Output = Result<Value, ApiError>> + Send;
}

/// Request fields that shape or redirect a reply. The independent check must answer in plain prose
/// (`NO ERRORS FOUND` or a list), so they are removed from its body only; the review and correction
/// passes keep them, because the final answer must still honour the client's format.
pub const CHECK_STRIPPED: [&str; 7] =
    ["tools", "tool_choice", "parallel_tool_calls", "response_format", "json_schema", "grammar", "n"];

const MESSAGES: &str = "messages";

/// Run the draft and every pass `plan` asks for on `base` (the prepared request body, whose budget
/// was planned for the draft with `client_max` and `fit`). `on_pass` is called as each pass starts,
/// so a streaming caller can keep the connection alive.
///
/// Every pass gets its own token budget: a review or correction pass sends the draft's prompt plus
/// the answer and an instruction, so its thinking budget is re-planned from the grown prompt
/// ([`effort::follow_up_bound`]); the independent check is planned from its own fresh prompt. With
/// no `fit` (thinking off, or the context size unknown) every pass keeps the draft's budget.
///
/// The conversation is moved between pass bodies rather than copied.
///
/// # Errors
/// The first upstream failure; passes already run are discarded with it.
pub async fn run<C: Completer + Sync>(
    completer: &C,
    base: Map<String, Value>,
    plan: &Plan,
    client_max: Option<u64>,
    fit: Option<ContextFit>,
    mut on_pass: impl FnMut(PassKind) + Send,
) -> Result<PassOutcome, ApiError> {
    let mut body = base;
    body.insert("stream".to_string(), Value::Bool(false));
    body.remove("stream_options");
    let mut messages = match body.remove(MESSAGES) {
        Some(Value::Array(messages)) => messages,
        _ => Vec::new(),
    };
    let conversation_len = messages.len();

    let mut passes = Vec::new();
    let mut usage = UsageSum::default();

    on_pass(PassKind::Draft);
    let (mut last_response, record) = send(completer, PassKind::Draft, &mut body, &mut messages).await?;
    passes.push(record);
    usage.add(&last_response);

    let Some(mut answer) = reviewable_answer(&last_response) else {
        return Ok(PassOutcome { response: last_response, passes, skipped: Some("draft has no text to review (tool call or empty)") });
    };
    let mut answer_tokens = completion_tokens(&last_response);

    for _ in 0..plan.review_passes() {
        on_pass(PassKind::SelfReview);
        refit(&mut body, plan, client_max, fit, effort::follow_up_bound(&answer, answer_tokens, REVIEW_INSTRUCTION));
        push_follow_up(&mut messages, &answer, REVIEW_INSTRUCTION);
        let sent = send(completer, PassKind::SelfReview, &mut body, &mut messages).await;
        messages.truncate(conversation_len);
        let (response, record) = sent?;
        passes.push(record);
        usage.add(&response);
        if let Some(revised) = reviewable_answer(&response) {
            answer = revised;
            answer_tokens = completion_tokens(&response);
            last_response = response;
        }
    }

    if plan.independent_check() {
        on_pass(PassKind::IndependentCheck);
        let question = last_user_text(&messages);
        let mut check_messages = vec![
            json!({ "role": "system", "content": CHECK_SYSTEM }),
            json!({ "role": "user", "content": format!("Question:\n{question}\n\nAnswer:\n{answer}\n\n{CHECK_INSTRUCTION}") }),
        ];
        let mut check_body = check_body(&body);
        if let Some(fit) = fit {
            let fresh = ContextFit { n_ctx: fit.n_ctx, prompt_tokens: effort::prompt_bytes_bound(&check_messages) };
            effort::apply_budget(&mut check_body, &effort::plan_budget(plan, client_max, Some(fresh)));
        }
        let (check, mut record) = send(completer, PassKind::IndependentCheck, &mut check_body, &mut check_messages).await?;
        usage.add(&check);
        // No reviewable text (thinking ate the budget, an empty reply, a tool call) is
        // inconclusive, not a finding: `found_errors` stays unknown and nothing is corrected.
        let verdict = reviewable_answer(&check);
        record.found_errors = verdict.as_deref().map(|text| !is_clean(text));
        passes.push(record);
        if let Some(findings) = verdict.filter(|text| !is_clean(text)) {
            on_pass(PassKind::Correction);
            let instruction = CORRECTION_TEMPLATE.replace("{findings}", findings.trim());
            refit(&mut body, plan, client_max, fit, effort::follow_up_bound(&answer, answer_tokens, &instruction));
            push_follow_up(&mut messages, &answer, &instruction);
            let sent = send(completer, PassKind::Correction, &mut body, &mut messages).await;
            messages.truncate(conversation_len);
            let (response, record) = sent?;
            passes.push(record);
            usage.add(&response);
            if let Some(corrected) = reviewable_answer(&response) {
                answer = corrected;
                last_response = response;
            }
        }
    }

    set_answer(&mut last_response, &answer);
    usage.write_into(&mut last_response);
    Ok(PassOutcome { response: last_response, passes, skipped: None })
}

/// Send `body` with `messages` as its conversation and time the call, then move the conversation
/// back out so the next pass reuses it without a copy.
async fn send<C: Completer>(
    completer: &C,
    kind: PassKind,
    body: &mut Map<String, Value>,
    messages: &mut Vec<Value>,
) -> Result<(Value, PassRecord), ApiError> {
    body.insert(MESSAGES.to_string(), Value::Array(std::mem::take(messages)));
    let started = Instant::now();
    let result = completer.complete(body).await;
    if let Some(Value::Array(back)) = body.remove(MESSAGES) {
        *messages = back;
    }
    let response = result?;
    let ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let tokens_per_second = response.pointer("/timings/predicted_per_second").and_then(Value::as_f64);
    Ok((response, PassRecord { kind, ms, tokens_per_second, found_errors: None }))
}

/// Re-plan `body`'s token budget for a pass whose prompt is the draft's plus `added` tokens.
fn refit(body: &mut Map<String, Value>, plan: &Plan, client_max: Option<u64>, fit: Option<ContextFit>, added: u64) {
    let Some(fit) = fit else {
        return;
    };
    let grown = ContextFit { n_ctx: fit.n_ctx, prompt_tokens: fit.prompt_tokens.saturating_add(added) };
    effort::apply_budget(body, &effort::plan_budget(plan, client_max, Some(grown)));
}

/// The independent check's body: `body` without its conversation and the fields in
/// [`CHECK_STRIPPED`].
fn check_body(body: &Map<String, Value>) -> Map<String, Value> {
    body.iter()
        .filter(|(k, _)| k.as_str() != MESSAGES && !CHECK_STRIPPED.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Append `answer` as the assistant and `instruction` as the user.
fn push_follow_up(messages: &mut Vec<Value>, answer: &str, instruction: &str) {
    messages.push(json!({ "role": "assistant", "content": answer }));
    messages.push(json!({ "role": "user", "content": instruction }));
}

/// `usage.completion_tokens` of a response, if it reported one.
fn completion_tokens(response: &Value) -> Option<u64> {
    response.pointer("/usage/completion_tokens").and_then(Value::as_u64)
}

/// The first choice's message text, unless it is a tool call or empty.
fn reviewable_answer(response: &Value) -> Option<String> {
    let message = response.pointer("/choices/0/message")?;
    if message.get("tool_calls").and_then(Value::as_array).is_some_and(|c| !c.is_empty()) {
        return None;
    }
    let text = message.get("content")?.as_str()?;
    (!text.trim().is_empty()).then(|| text.to_string())
}

/// Replace the first choice's message text.
fn set_answer(response: &mut Value, answer: &str) {
    if let Some(message) = response.pointer_mut("/choices/0/message").and_then(Value::as_object_mut) {
        message.insert("content".to_string(), Value::String(answer.to_string()));
    }
}

/// The text of the last user message (string content, or its text parts joined).
fn last_user_text(messages: &[Value]) -> String {
    let Some(content) = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|m| m.get("content"))
    else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Whether the check's first line is the clean verdict (case, surrounding whitespace, quotes,
/// emphasis and a trailing period or exclamation mark ignored). Later lines are explanation and do
/// not change it; a first line that qualifies the verdict ("NO ERRORS FOUND, except ...") is not
/// clean.
pub fn is_clean(findings: &str) -> bool {
    let first = findings.trim().lines().next().unwrap_or("");
    let verdict = first
        .trim()
        .trim_matches(|c| c == '"' || c == '*' || c == '`')
        .trim_end_matches(['.', '!'])
        .trim();
    verdict.eq_ignore_ascii_case(CHECK_CLEAN)
}

/// Token usage summed over passes.
#[derive(Default)]
struct UsageSum {
    prompt: u64,
    completion: u64,
    seen: bool,
}

impl UsageSum {
    fn add(&mut self, response: &Value) {
        if let Some(usage) = response.get("usage") {
            self.seen = true;
            let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0);
            self.prompt = self.prompt.saturating_add(field("prompt_tokens"));
            self.completion = self.completion.saturating_add(field("completion_tokens"));
        }
    }

    fn write_into(&self, response: &mut Value) {
        if let (true, Some(map)) = (self.seen, response.as_object_mut()) {
            map.insert("usage".to_string(), json!({
                "prompt_tokens": self.prompt,
                "completion_tokens": self.completion,
                "total_tokens": self.prompt.saturating_add(self.completion),
            }));
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::effort::{resolve, Dials, TEMPLATE_OVERHEAD_PER_MESSAGE};
    use std::sync::Mutex;

    /// Records every request and answers from a script, in order.
    struct Scripted {
        replies: Mutex<Vec<Value>>,
        seen: Mutex<Vec<Value>>,
    }

    fn reply(text: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": text}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
               "timings": {"predicted_per_second": 50.0}})
    }

    impl Scripted {
        fn new(replies: Vec<&str>) -> Self {
            Self::raw(replies.into_iter().map(reply).collect())
        }

        fn raw(replies: Vec<Value>) -> Self {
            Self { replies: Mutex::new(replies.into_iter().rev().collect()), seen: Mutex::new(Vec::new()) }
        }

        fn seen(&self) -> Vec<Value> {
            self.seen.lock().expect("lock").clone()
        }
    }

    impl Completer for Scripted {
        fn complete(&self, body: &Map<String, Value>) -> impl Future<Output = Result<Value, ApiError>> + Send {
            self.seen.lock().expect("lock").push(Value::Object(body.clone()));
            let reply = self.replies.lock().expect("lock").pop().ok_or_else(|| ApiError::bad_upstream("script ran out"));
            async move { reply }
        }
    }

    fn plan(level: &str) -> Plan {
        let mut body = json!({"cerebro_effort": level}).as_object().cloned().expect("object");
        resolve(&Dials::default(), &mut body).expect("resolve")
    }

    fn base() -> Map<String, Value> {
        json!({"stream": true, "stream_options": {"include_usage": true},
               "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "What is 2+2?"}]})
        .as_object()
        .cloned()
        .expect("object")
    }

    async fn run_plain(c: &Scripted, level: &str, started: &mut Vec<PassKind>) -> PassOutcome {
        run(c, base(), &plan(level), None, None, |k| started.push(k)).await.expect("passes")
    }

    fn tool_call_reply() -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": null,
               "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]})
    }

    #[tokio::test]
    async fn xhigh_runs_draft_then_one_self_review_and_returns_the_revision() {
        let c = Scripted::new(vec!["draft 5", "revised 4"]);
        let mut started = Vec::new();
        let out = run_plain(&c, "xhigh", &mut started).await;
        assert_eq!(started, vec![PassKind::Draft, PassKind::SelfReview]);
        assert_eq!(out.passes.iter().map(|p| p.kind).collect::<Vec<_>>(), started);
        assert_eq!(out.response["choices"][0]["message"]["content"], "revised 4");
        assert_eq!(out.response["usage"]["total_tokens"], 30);

        let seen = c.seen();
        assert_eq!(seen[0]["stream"], false);
        assert!(seen[0].get("stream_options").is_none());
        assert_eq!(seen[0]["messages"].as_array().expect("messages").len(), 2);
        let review = seen[1]["messages"].as_array().expect("messages");
        assert_eq!(review.len(), 4);
        assert_eq!(review[2], json!({"role": "assistant", "content": "draft 5"}));
        assert_eq!(review[3], json!({"role": "user", "content": REVIEW_INSTRUCTION}));
    }

    #[tokio::test]
    async fn max_runs_review_then_independent_check_in_a_fresh_context_then_correction() {
        let c = Scripted::new(vec!["draft 5", "revised 5", "The answer 5 is wrong; 2+2 is 4.", "corrected 4"]);
        let mut started = Vec::new();
        let out = run_plain(&c, "max", &mut started).await;
        assert_eq!(started, vec![PassKind::Draft, PassKind::SelfReview, PassKind::IndependentCheck, PassKind::Correction]);
        assert_eq!(out.passes[2].found_errors, Some(true));
        assert_eq!(out.response["choices"][0]["message"]["content"], "corrected 4");

        let seen = c.seen();
        // The check sees only its own system prompt, the question and the reviewed answer.
        let check = seen[2]["messages"].as_array().expect("messages");
        assert_eq!(check.len(), 2);
        assert_eq!(check[0]["content"], CHECK_SYSTEM);
        let user = check[1]["content"].as_str().expect("text");
        assert_eq!(user, format!("Question:\nWhat is 2+2?\n\nAnswer:\nrevised 5\n\n{CHECK_INSTRUCTION}"));
        assert!(!check.iter().any(|m| m["content"] == "SYS"));
        // The correction goes back to the full conversation (not the review's) with the findings.
        let fix = seen[3]["messages"].as_array().expect("messages");
        assert_eq!(fix.len(), 4);
        assert_eq!(fix[0]["content"], "SYS");
        assert_eq!(fix[2]["content"], "revised 5");
        assert!(fix[3]["content"].as_str().expect("text").contains("2+2 is 4"));
    }

    #[tokio::test]
    async fn a_clean_check_skips_the_correction() {
        let c = Scripted::new(vec!["draft 4", "revised 4", "No errors found."]);
        let out = run_plain(&c, "max", &mut Vec::new()).await;
        assert_eq!(out.passes.len(), 3);
        assert_eq!(out.passes[2].found_errors, Some(false));
        assert_eq!(out.passes[2].to_json()["found_errors"], false);
        assert_eq!(out.response["choices"][0]["message"]["content"], "revised 4");
    }

    #[tokio::test]
    async fn the_check_drops_output_shaping_and_an_empty_check_is_inconclusive() {
        let mut shaped = base();
        for (k, v) in [
            ("response_format", json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}})),
            ("tools", json!([{"type": "function", "function": {"name": "f", "parameters": {}}}])),
            ("tool_choice", json!("auto")),
            ("parallel_tool_calls", json!(true)),
            ("grammar", json!("root ::= x")),
            ("temperature", json!(0.2)),
        ] {
            shaped.insert(k.to_string(), v);
        }
        let c = Scripted::new(vec![r#"{"a":1}"#, r#"{"a":2}"#, ""]);
        let mut started = Vec::new();
        let out = run(&c, shaped, &plan("max"), None, None, |k| started.push(k)).await.expect("passes");
        assert_eq!(started, vec![PassKind::Draft, PassKind::SelfReview, PassKind::IndependentCheck], "no correction");
        assert_eq!(out.passes[2].found_errors, None);
        let record = out.passes[2].to_json();
        assert!(record.as_object().expect("object").contains_key("found_errors"));
        assert_eq!(record["found_errors"], Value::Null);
        assert_eq!(out.response["choices"][0]["message"]["content"], r#"{"a":2}"#);

        let seen = c.seen();
        for field in CHECK_STRIPPED {
            assert!(seen[2].get(field).is_none(), "the check body kept {field}");
        }
        assert_eq!(seen[2]["temperature"], 0.2, "sampling fields stay");
        for pass in &seen[..2] {
            assert!(pass.get("response_format").is_some() && pass.get("tools").is_some(), "draft and review keep the client's format");
        }
    }

    #[tokio::test]
    async fn a_check_that_answers_with_a_tool_call_runs_no_correction() {
        let c = Scripted::raw(vec![reply("draft"), reply("revised"), tool_call_reply()]);
        let out = run_plain(&c, "max", &mut Vec::new()).await;
        assert_eq!(out.passes.len(), 3);
        assert_eq!(out.passes[2].found_errors, None);
        assert_eq!(out.response["choices"][0]["message"]["content"], "revised");
    }

    #[tokio::test]
    async fn every_pass_is_budgeted_for_its_own_prompt() {
        // XHigh wants 16384 tokens of thinking; the draft's prompt is 1000 of a 4096 context, and
        // the 2048-token answer reserve must stay free in every pass.
        let fit = ContextFit { n_ctx: 4096, prompt_tokens: 1000 };
        let c = Scripted::new(vec!["draft 5", "revised 4"]);
        run(&c, base(), &plan("xhigh"), None, Some(fit), |_| {}).await.expect("passes");
        let seen = c.seen();
        let review_thinking = seen[1]["reasoning_budget_tokens"].as_u64().expect("budget");
        // The review sends the draft's prompt plus the answer (5 tokens by usage, 7 bytes) and the
        // instruction, each with its role and framing.
        let added = 5
            + REVIEW_INSTRUCTION.len() as u64
            + ("assistant".len() + "user".len()) as u64
            + 2 * TEMPLATE_OVERHEAD_PER_MESSAGE;
        assert_eq!(review_thinking, 4096 - (1000 + added) - 2048);
        assert!(1000 + added + review_thinking + 2048 <= 4096);

        // Max: the check is planned from its own small prompt, not the conversation's.
        let c = Scripted::new(vec!["draft 5", "revised 5", "5 is wrong.", "corrected 4"]);
        let big = ContextFit { n_ctx: 32_768, prompt_tokens: 20_000 };
        run(&c, base(), &plan("max"), Some(100), Some(big), |_| {}).await.expect("passes");
        let seen = c.seen();
        let check_prompt = effort::prompt_bytes_bound(seen[2]["messages"].as_array().expect("messages"));
        assert_eq!(seen[2]["reasoning_budget_tokens"].as_u64(), Some(32_768 - check_prompt - 100));
        assert_eq!(seen[2]["max_tokens"].as_u64(), Some(32_768 - check_prompt));
        let correction_added =
            effort::follow_up_bound("revised 5", Some(5), &CORRECTION_TEMPLATE.replace("{findings}", "5 is wrong."));
        assert_eq!(seen[3]["reasoning_budget_tokens"].as_u64(), Some(32_768 - 20_000 - correction_added - 100));
    }

    #[tokio::test]
    async fn a_tool_call_or_non_object_draft_is_returned_unreviewed() {
        let c = Scripted::raw(vec![tool_call_reply()]);
        let out = run_plain(&c, "max", &mut Vec::new()).await;
        assert_eq!(out.passes.len(), 1);
        assert!(out.skipped.is_some());
        assert_eq!(out.response, tool_call_reply());

        let c = Scripted::raw(vec![json!([])]);
        let out = run_plain(&c, "xhigh", &mut Vec::new()).await;
        assert_eq!(out.response, json!([]));
        assert!(out.skipped.is_some());
    }

    #[test]
    fn clean_verdict_detection() {
        assert!(is_clean("NO ERRORS FOUND"));
        assert!(is_clean("  no errors found.\n"));
        assert!(is_clean("**NO ERRORS FOUND**"));
        assert!(!is_clean("NO ERRORS FOUND, except the sum is wrong"));
        assert!(!is_clean(""));
        // A small model's clean verdict followed by an explanation is clean.
        assert!(is_clean("No errors found.\nThe answer correctly computes 4."));
        assert!(is_clean("\n**NO ERRORS FOUND**\n\nThe arithmetic checks out!"));
        assert!(is_clean("No errors found!"));
        // Errors on the first line, or a verdict that is not first, are findings.
        assert!(!is_clean("ERRORS FOUND\n- 2+2 is 4, not 5"));
        assert!(!is_clean("The sum is wrong.\nNO ERRORS FOUND"));
        assert!(!is_clean("NO ERRORS FOUND, except the sum is wrong\nmore"));
    }
}
