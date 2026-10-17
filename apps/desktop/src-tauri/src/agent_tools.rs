//! Tools that talk to the user rather than the computer: a task list the model keeps up to date,
//! questions it can ask, and plan mode's hand-over. The engine handles them itself (engine.rs,
//! run_chat): they change nothing outside the conversation, so no permission rule applies, and the
//! two that need an answer wait for it in the chat.

use serde_json::{json, Value};

use crate::toolcall::ToolSpec;

pub const TODO: &str = "todo_write";
pub const ASK: &str = "ask_user";
pub const EXIT_PLAN: &str = "exit_plan_mode";

const MAX_TODOS: usize = 50;
const MAX_QUESTIONS: usize = 4;
const MAX_OPTIONS: usize = 6;
const MAX_PLAN_CHARS: usize = 20_000;

/// Added to the system prompt while plan mode is on.
pub const PLAN_PROMPT: &str = "You are in plan mode. Investigate first: read files and search, but do not change anything (changes and commands are refused until the plan is approved). When you have a plan, call exit_plan_mode with the complete plan in Markdown. The user approves it, or tells you what to change and you plan again.";

pub fn is_agent_tool(name: &str) -> bool {
    matches!(name, TODO | ASK | EXIT_PLAN)
}

/// The tools as the model sees them; exit_plan_mode only while plan mode is on.
pub fn specs(server: &str, plan_mode: bool) -> Vec<ToolSpec> {
    let mut v = vec![
        ToolSpec {
            server: server.to_owned(),
            name: TODO.to_owned(),
            description: "Keep a task list for multi-step work. Send the whole list every time; at most one task in_progress.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": { "todos": { "type": "array", "items": { "type": "object", "properties": {
                    "content": { "type": "string", "description": "The task, in a few words." },
                    "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] } }, "required": ["content", "status"] } } },
                "required": ["todos"],
            }),
        },
        ToolSpec {
            server: server.to_owned(),
            name: ASK.to_owned(),
            description: "Ask the user one to four questions when a decision is theirs to make. Offer options when you can, the recommended one first.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": { "questions": { "type": "array", "items": { "type": "object", "properties": {
                    "question": { "type": "string" },
                    "options": { "type": "array", "items": { "type": "string" } },
                    "multi": { "type": "boolean", "description": "More than one option may be chosen." } }, "required": ["question"] } } },
                "required": ["questions"],
            }),
        },
    ];
    if plan_mode {
        v.push(ToolSpec {
            server: server.to_owned(),
            name: EXIT_PLAN.to_owned(),
            description: "Present your plan to the user for approval. Call this once the plan is complete.".to_owned(),
            input_schema: json!({ "type": "object", "properties": { "plan": { "type": "string", "description": "The plan, in Markdown." } }, "required": ["plan"] }),
        });
    }
    v
}

/// Validates a todo_write call; the summary goes back to the model.
pub fn todo(args: &Value) -> Result<String, String> {
    let todos = args["todos"].as_array().ok_or("todos must be a list.")?;
    if todos.is_empty() || todos.len() > MAX_TODOS {
        return Err(format!("todos must hold 1 to {MAX_TODOS} tasks."));
    }
    let (mut pending, mut doing, mut done) = (0, 0, 0);
    let mut seen = std::collections::HashSet::new();
    for t in todos {
        let content = t["content"].as_str().map(str::trim).filter(|c| !c.is_empty() && c.chars().count() <= 500).ok_or("Every task needs content (at most 500 characters).")?;
        if !seen.insert(content.to_owned()) {
            return Err(format!("\"{content}\" is listed twice; every task must be different."));
        }
        match t["status"].as_str() {
            Some("pending") => pending += 1,
            Some("in_progress") => doing += 1,
            Some("completed") => done += 1,
            _ => return Err("status must be pending, in_progress or completed.".into()),
        }
    }
    if doing > 1 {
        return Err("Only one task may be in_progress at a time.".into());
    }
    Ok(format!("Updated todo list: {pending} pending, {doing} in progress, {done} completed."))
}

/// Validates an ask_user call into the questions the chat shows.
pub fn questions(args: &Value) -> Result<Value, String> {
    let qs = args["questions"].as_array().ok_or("questions must be a list.")?;
    if qs.is_empty() || qs.len() > MAX_QUESTIONS {
        return Err(format!("Ask 1 to {MAX_QUESTIONS} questions."));
    }
    let mut out = Vec::new();
    for q in qs {
        let question = q["question"].as_str().map(str::trim).filter(|s| !s.is_empty() && s.chars().count() <= 1000).ok_or("Every question needs text (at most 1000 characters).")?;
        let options: Vec<String> = match &q["options"] {
            Value::Null => Vec::new(),
            Value::Array(a) => a.iter().filter_map(Value::as_str).map(|s| s.trim().chars().take(200).collect::<String>()).filter(|s| !s.is_empty()).collect(),
            _ => return Err("options must be a list of strings.".into()),
        };
        if options.len() > MAX_OPTIONS {
            return Err(format!("At most {MAX_OPTIONS} options per question."));
        }
        out.push(json!({ "question": question, "options": options, "multi": q["multi"].as_bool().unwrap_or(false) }));
    }
    Ok(json!({ "questions": out }))
}

/// The user's answers as the model reads them; anything malformed counts as no answer.
pub fn answers_text(answer: &Value) -> Option<String> {
    let a = answer["answers"].as_array()?;
    if a.is_empty() || a.len() > MAX_QUESTIONS {
        return None;
    }
    Some(json!({ "answers": a }).to_string())
}

/// Validates an exit_plan_mode call: the plan text.
pub fn plan(args: &Value) -> Result<String, String> {
    let p = args["plan"].as_str().map(str::trim).filter(|p| !p.is_empty()).ok_or("plan must be the plan's text.")?;
    if p.chars().count() > MAX_PLAN_CHARS {
        return Err(format!("The plan is longer than {MAX_PLAN_CHARS} characters; make it shorter."));
    }
    Ok(p.to_owned())
}

/// What the model hears after the user reviewed a plan.
pub fn plan_reply(answer: &Value) -> String {
    if answer["approved"].as_bool() == Some(true) {
        "The user approved the plan. Plan mode is over: carry the plan out now.".to_owned()
    } else {
        let fb: String = answer["feedback"].as_str().unwrap_or("").trim().chars().take(4000).collect();
        if fb.is_empty() {
            "The user did not approve the plan yet. Ask what they want changed, or improve it and call exit_plan_mode again.".to_owned()
        } else {
            format!("The user wants changes before you start: {fb}\nRevise the plan and call exit_plan_mode again.")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_lists_are_validated_and_summarized() {
        let ok = json!({ "todos": [{ "content": "Read config", "status": "completed" }, { "content": "Fix bug", "status": "in_progress" }, { "content": "Test", "status": "pending" }] });
        assert_eq!(todo(&ok).unwrap(), "Updated todo list: 1 pending, 1 in progress, 1 completed.");
        assert!(todo(&json!({ "todos": [] })).is_err());
        assert!(todo(&json!({ "todos": [{ "content": "a", "status": "in_progress" }, { "content": "b", "status": "in_progress" }] })).unwrap_err().contains("one task"));
        assert!(todo(&json!({ "todos": [{ "content": "a", "status": "pending" }, { "content": "a", "status": "pending" }] })).unwrap_err().contains("twice"));
        assert!(todo(&json!({ "todos": [{ "content": "a", "status": "doing" }] })).is_err());
    }

    #[test]
    fn questions_and_answers() {
        let q = questions(&json!({ "questions": [{ "question": " Which database? ", "options": ["Postgres (recommended)", "SQLite", ""] }] })).unwrap();
        assert_eq!(q, json!({ "questions": [{ "question": "Which database?", "options": ["Postgres (recommended)", "SQLite"], "multi": false }] }));
        assert!(questions(&json!({ "questions": [] })).is_err());
        assert!(questions(&json!({ "questions": [{ "question": "x", "options": ["1", "2", "3", "4", "5", "6", "7"] }] })).is_err());
        assert_eq!(answers_text(&json!({ "answers": [{ "question": "Which database?", "answer": "SQLite" }] })).unwrap(), r#"{"answers":[{"answer":"SQLite","question":"Which database?"}]}"#);
        assert!(answers_text(&json!({ "nope": 1 })).is_none());
    }

    #[test]
    fn plans_and_reviews() {
        assert_eq!(plan(&json!({ "plan": " 1. Do it " })).unwrap(), "1. Do it");
        assert!(plan(&json!({ "plan": "" })).is_err());
        assert!(plan_reply(&json!({ "approved": true })).contains("carry the plan out"));
        assert!(plan_reply(&json!({ "approved": false, "feedback": "use SQLite" })).contains("use SQLite"));
        assert_eq!(specs("cerebro", false).len(), 2);
        assert_eq!(specs("cerebro", true).last().unwrap().name, EXIT_PLAN);
    }
}
