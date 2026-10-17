//! Orchestrate mode: GVS5H ledger-based self-orchestration (arXiv 2608.26480,
//! github.com/slee-persis/GVS5H, MIT), ported from the owner's Strata runner
//! (docs/reference/gvs5h-from-strata/gvs5h_strata.py).
//!
//! Every role is the same local model in a fresh context. They coordinate only through a ledger of
//! files (plan.md, notes.md, tasks.json, result.md, transcript.jsonl): a manager plans and curates
//! tasks, workers each take one task (several per round, each told to try a different approach),
//! rewrite the notes and the current result, and the manager decides when the goal is met. No
//! context is carried in memory between calls; the ledger is the only channel.
//!
//! On one GPU the instances take turns (fresh context each); they run side by side once the engine
//! batches several sequences.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// One model call. Implemented over the engine (engine.rs) and by a script in the tests.
pub trait Llm {
    /// A reply to `user` under `system` in a fresh context. Reasoning may be included; it is stripped.
    fn call(&mut self, role: &str, system: &str, user: &str, temperature: f32) -> Result<Reply, String>;
    fn cancelled(&self) -> bool;
}

pub struct Reply {
    pub text: String,
    /// The reply hit the token limit before it ended.
    pub cut_off: bool,
}

/// What the run reports as it goes, for the Chat ledger panel.
pub enum Progress<'a> {
    /// A role started (e.g. "plan", "worker 2", "manage").
    Step { role: &'a str, round: usize },
    /// A ledger file changed; `content` is its whole new text.
    File { name: &'static str, content: &'a str },
}

#[derive(Clone, Copy)]
pub struct Options {
    /// Workers per round, each on its own task (1..=6).
    pub workers: usize,
    /// Rounds of manage -> workers before the result is finalized (1..=8).
    pub rounds: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct Task {
    id: usize,
    desc: String,
    done: bool,
}

const SWARM: &str = "You are the PRIMARY orchestrator of a small team of experts. You own the task list and decide when the goal is met. Workers coordinate with you ONLY through files in a shared workspace, so state what they must know explicitly.";
const WORKER: &str = "You are a WORKER on a small team. You share a workspace with other workers and see only the files listed in your prompt. Build on the existing work where it helps, but if your task is to try a different approach, write a fresh version rather than patching a stuck one.";
const MAX_TASKS: usize = 12;

pub struct Ledger {
    dir: PathBuf,
}

impl Ledger {
    pub fn create(dir: &Path, goal: &str) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("Could not create the ledger folder: {e}"))?;
        let l = Ledger { dir: dir.to_owned() };
        for f in ["notes.md", "plan.md", "result.md", "tasks.json", "transcript.jsonl"] {
            l.write(f, "")?;
        }
        l.write("task.md", goal)?;
        Ok(l)
    }
    pub fn dir(&self) -> &Path {
        &self.dir
    }
    fn read(&self, f: &str) -> String {
        std::fs::read_to_string(self.dir.join(f)).unwrap_or_default()
    }
    fn write(&self, f: &str, s: &str) -> Result<(), String> {
        std::fs::write(self.dir.join(f), s).map_err(|e| format!("Could not write {f}: {e}"))
    }
    fn record(&self, role: &str, system: &str, user: &str, reply: &str) {
        let line = json!({ "role": role, "system": system, "user": user, "reply": reply }).to_string();
        let path = self.dir.join("transcript.jsonl");
        let mut s = std::fs::read_to_string(&path).unwrap_or_default();
        s.push_str(&line);
        s.push('\n');
        let _ = std::fs::write(path, s);
    }
}

/// The orchestration. Returns the final result text.
pub fn run(llm: &mut dyn Llm, ledger: &Ledger, goal: &str, context: &str, opts: Options, progress: &mut dyn FnMut(Progress<'_>)) -> Result<String, String> {
    let workers = opts.workers.clamp(1, 6);
    let rounds = opts.rounds.clamp(1, 8);
    let mut call = |llm: &mut dyn Llm, role: &str, round: usize, system: &str, user: &str, t: f32, progress: &mut dyn FnMut(Progress<'_>)| -> Result<Reply, String> {
        if llm.cancelled() {
            return Err("Stopped.".into());
        }
        progress(Progress::Step { role, round });
        let r = llm.call(role, system, user, t)?;
        ledger.record(role, system, user, &r.text);
        Ok(r)
    };
    let mut put = |name: &'static str, content: &str, progress: &mut dyn FnMut(Progress<'_>)| -> Result<(), String> {
        ledger.write(name, content)?;
        progress(Progress::File { name, content });
        Ok(())
    };

    // 1. Plan (manager).
    let sys = format!("{SWARM} Given a goal, produce a short strategy and a task list the workers can pick up. Respond with EXACTLY these sections:\n### PLAN\n<3-6 sentence strategy>\n### TASKS\n<3-6 bullets, each a concrete unit of work>");
    let r = call(llm, "plan", 0, &sys, &format!("GOAL:\n{goal}\n\nCONTEXT:\n{}", or_none(context)), 0.3, progress)?;
    let sec = sections(&r.text);
    put("plan.md", clip(sec.get("PLAN").map_or(r.text.trim(), String::as_str), 6000), progress)?;
    let mut tasks: Vec<Task> = Vec::new();
    merge_tasks(&mut tasks, &bullets(sec.get("TASKS").map_or("", String::as_str)));
    put("tasks.json", &tasks_json(&tasks), progress)?;

    // 2. Ideation (first worker): distinct approaches, no implementation.
    let sys = "You are the FIRST WORKER. Do NOT solve the task yet. Think about the goal, identify the real difficulty, and list SEVERAL DISTINCT approaches (genuinely different strategies, not variations of one idea), with pitfalls for each. Respond with EXACTLY:\n### NOTES\n<your analysis>\n### NEXT\n<bullet list of distinct approaches to try>";
    let r = call(llm, "ideate", 0, sys, &format!("GOAL:\n{goal}\n\nPLAN:\n{}", ledger.read("plan.md")), 0.4, progress)?;
    let sec = sections(&r.text);
    let notes = format!("- **Ideation:** {}\n", clip(sec.get("NOTES").map_or(r.text.trim(), String::as_str), 8000));
    put("notes.md", &notes, progress)?;
    let mut proposals = bullets(sec.get("NEXT").map_or("", String::as_str));
    let mut last = "ideation complete".to_owned();
    let mut previous: Vec<String> = Vec::new();

    for round in 1..=rounds {
        // 3. Manage: curate tasks, decide done, pick this round's tasks (one per worker).
        let (done, next) = manage(llm, &mut call, ledger, goal, context, &mut tasks, &proposals, &last, workers, round, progress)?;
        put("tasks.json", &tasks_json(&tasks), progress)?;
        if done || next.is_empty() {
            break;
        }
        // The same tasks issued again with no progress stops the loop (from the original).
        let key: Vec<String> = next.iter().map(|t| t.to_lowercase()).collect();
        if key == previous {
            break;
        }
        previous = key;
        // 4. Workers: each takes one task; later workers see earlier workers' ledger updates.
        let mut summaries = Vec::new();
        proposals.clear();
        for (i, task) in next.iter().enumerate() {
            let role = format!("worker {}", i + 1);
            let (status, nexts, summary) = worker(llm, &mut call, &mut put, ledger, goal, context, &format!("Complete this task: {task}"), &role, round, progress)?;
            proposals.extend(nexts);
            summaries.push(format!("{role} ({status}): {summary}"));
        }
        last = summaries.join("\n");
    }

    // 5. Finalize: one definitive deliverable from everything in the ledger.
    let goal_line = "Produce the DEFINITIVE final answer now, using all notes and the current result. Write it for the user: complete, correct, no mention of the team or the ledger.";
    worker(llm, &mut call, &mut put, ledger, goal, context, goal_line, "finalize", rounds + 1, progress)?;
    let result = ledger.read("result.md");
    if result.trim().is_empty() {
        return Err("The model did not produce a result.".into());
    }
    Ok(result.trim().to_owned())
}

type CallFn<'a> = dyn FnMut(&mut dyn Llm, &str, usize, &str, &str, f32, &mut dyn FnMut(Progress<'_>)) -> Result<Reply, String> + 'a;
type PutFn<'a> = dyn FnMut(&'static str, &str, &mut dyn FnMut(Progress<'_>)) -> Result<(), String> + 'a;

#[allow(clippy::too_many_arguments)]
fn manage(
    llm: &mut dyn Llm,
    call: &mut CallFn<'_>,
    ledger: &Ledger,
    goal: &str,
    context: &str,
    tasks: &mut Vec<Task>,
    proposals: &[String],
    last: &str,
    workers: usize,
    round: usize,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<(bool, Vec<String>), String> {
    let task_lines = if tasks.is_empty() { "(none)".to_owned() } else { tasks.iter().map(|t| format!("- [{}] {}", if t.done { "done" } else { "todo" }, t.desc)).collect::<Vec<_>>().join("\n") };
    let prop_lines = if proposals.is_empty() { "(none)".to_owned() } else { proposals.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n") };
    let current = ledger.read("result.md");
    let sys = format!(
        "{SWARM} Review progress and the latest worker results, then:\n- If the current result satisfies the goal, set STATUS to 'done'.\n- Otherwise curate the task list: merge duplicates, mark completed [done], drop irrelevant items, fold in only genuinely new sub-tasks.\n- If the last workers made no real progress, do NOT keep polishing the same idea: switch to a DIFFERENT approach.\n- Pick up to {workers} tasks for this round, one per worker, each different; when the best approach is unclear, give workers DIFFERENT approaches to the same problem.\nRespond with EXACTLY these sections:\n### STATUS\n<done|continue>\n### NEXT\n<bullets: the tasks for this round, one per worker; omit if done>\n### TASKS\n<curated list, '- [done] ...' or '- [todo] ...'>"
    );
    let user = format!(
        "GOAL:\n{goal}\n\nCONTEXT:\n{}\n\nCURRENT RESULT:\n{}\n\nNOTES:\n{}\n\nTASK LIST:\n{task_lines}\n\nLATEST WORKER RESULTS:\n{last}\n\nPROPOSALS:\n{prop_lines}",
        or_none(context),
        clip(if current.trim().is_empty() { "(none yet)" } else { &current }, 20000),
        clip(&ledger.read("notes.md"), 20000),
    );
    let r = call(llm, "manage", round, &sys, &user, 0.2, progress)?;
    let sec = sections(&r.text);
    let curated = parse_tasks(sec.get("TASKS").map_or("", String::as_str));
    if !curated.is_empty() {
        *tasks = curated;
    }
    let mut done = sec.get("STATUS").is_some_and(|s| s.trim().to_lowercase().starts_with("done"));
    if done && current.trim().is_empty() {
        done = false; // nothing to deliver yet
    }
    let mut next: Vec<String> = bullets(sec.get("NEXT").map_or("", String::as_str)).into_iter().take(workers).collect();
    if !done && next.is_empty() {
        next = tasks.iter().filter(|t| !t.done).map(|t| t.desc.clone()).take(workers).collect();
    }
    if !done && next.is_empty() && current.trim().is_empty() {
        next = vec!["Solve the goal completely.".into()];
    }
    Ok((done, next))
}

#[allow(clippy::too_many_arguments)]
fn worker(
    llm: &mut dyn Llm,
    call: &mut CallFn<'_>,
    put: &mut PutFn<'_>,
    ledger: &Ledger,
    goal: &str,
    context: &str,
    goal_line: &str,
    role: &str,
    round: usize,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<(String, Vec<String>, String), String> {
    let fmt = "### RESULT\n<the current full content of the deliverable: prose, code or data as appropriate>\n### NOTES\n<the COMPLETE notes file, rewritten: fold your findings in, keep what still matters, DELETE anything superseded or disproven. Use '- **Topic:** ...' bullets, under ~800 words, no markdown headings>\n### NEXT\n<bulleted remaining steps, or 'none'>\n### STATUS\n<done|continue>";
    let sys = format!("{WORKER}\nRespond with EXACTLY these sections:\n{fmt}");
    let current = ledger.read("result.md");
    let user = format!(
        "GOAL:\n{goal}\n\nCONTEXT:\n{}\n\nPLAN:\n{}\n\nNOTES:\n{}\n\nCURRENT RESULT:\n{}\n\nYOUR TASK:\n{goal_line}",
        or_none(context),
        ledger.read("plan.md"),
        ledger.read("notes.md"),
        if current.trim().is_empty() { "(none yet)" } else { &current },
    );
    let r = call(llm, role, round, &sys, &user, 0.2, progress)?;
    let sec = sections(&r.text);
    let result = sec.get("RESULT").map_or("", |s| s.trim());
    if !result.is_empty() {
        put("result.md", result, progress)?;
    } else if role == "finalize" && current.trim().is_empty() {
        // A model that ignored the format: its whole reply is the answer.
        put("result.md", r.text.trim(), progress)?;
    }
    if let Some(n) = sec.get("NOTES").filter(|n| !n.trim().is_empty() && !r.cut_off) {
        put("notes.md", clip(n.trim(), 8000), progress)?;
    }
    let nexts: Vec<String> = bullets(sec.get("NEXT").map_or("", String::as_str)).into_iter().filter(|b| !b.eq_ignore_ascii_case("none")).collect();
    let status = sec.get("STATUS").map_or("continue", |s| s.trim()).to_lowercase();
    let summary = if r.cut_off { "hit the token limit before finishing; prefer a simpler approach".to_owned() } else { format!("reported '{status}' on: {}", clip(goal_line, 90)) };
    Ok((status, nexts, summary))
}

fn or_none(s: &str) -> &str {
    if s.trim().is_empty() { "(none)" } else { s }
}

fn clip(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Drops <think>...</think> blocks (and an unclosed trailing one).
pub fn strip_think(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find("<think>") {
        out.push_str(&rest[..i]);
        match rest[i..].find("</think>") {
            Some(j) => rest = &rest[i + j + "</think>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_owned()
}

/// `### NAME` (or `**NAME**`, or `NAME:`) headed sections, names upper-cased.
fn sections(text: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let mut cur: Option<String> = None;
    let mut buf: Vec<&str> = Vec::new();
    let clean = strip_think(text);
    for line in clean.lines() {
        let s = line.trim();
        let name = s
            .strip_prefix("###")
            .or_else(|| s.strip_prefix("##"))
            .map(|n| n.trim().trim_end_matches(':').to_owned())
            .or_else(|| s.strip_prefix("**").and_then(|n| n.strip_suffix("**").or_else(|| n.strip_suffix("**:"))).map(|n| n.trim().to_owned()))
            .or_else(|| s.strip_suffix(':').filter(|n| n.len() >= 3 && n.chars().all(|c| c.is_ascii_uppercase() || c == '_')).map(str::to_owned))
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_alphabetic() || c == '_'));
        if let Some(n) = name {
            if let Some(c) = cur.take() {
                out.insert(c, buf.join("\n").trim().to_owned());
            }
            cur = Some(n.to_uppercase());
            buf.clear();
        } else {
            buf.push(line);
        }
    }
    if let Some(c) = cur {
        out.insert(c, buf.join("\n").trim().to_owned());
    }
    out
}

/// `- item`, `* item`, `1. item`, `1) item`.
fn bullets(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|l| {
            let t = l.trim_start();
            let rest = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")).or_else(|| {
                let digits = t.chars().take_while(char::is_ascii_digit).count();
                (digits > 0).then(|| &t[digits..]).and_then(|r| r.strip_prefix(". ").or_else(|| r.strip_prefix(") ")))
            })?;
            let rest = rest.trim();
            (!rest.is_empty()).then(|| rest.to_owned())
        })
        .collect()
}

fn parse_tasks(text: &str) -> Vec<Task> {
    bullets(text)
        .into_iter()
        .enumerate()
        .filter_map(|(i, b)| {
            let (done, desc) = match b.strip_prefix('[').and_then(|r| r.split_once(']')) {
                Some((mark, d)) => (mark.trim().eq_ignore_ascii_case("done") || mark.trim().eq_ignore_ascii_case("x"), d.trim().to_owned()),
                None => (false, b),
            };
            (!desc.is_empty()).then_some(Task { id: i + 1, desc, done })
        })
        .take(MAX_TASKS)
        .collect()
}

fn merge_tasks(tasks: &mut Vec<Task>, descs: &[String]) {
    for d in descs {
        let d = d.trim();
        if !d.is_empty() && tasks.len() < MAX_TASKS && !tasks.iter().any(|t| t.desc.eq_ignore_ascii_case(d)) {
            let id = tasks.iter().map(|t| t.id).max().unwrap_or(0) + 1;
            tasks.push(Task { id, desc: d.to_owned(), done: false });
        }
    }
}

fn tasks_json(tasks: &[Task]) -> String {
    serde_json::to_string_pretty(&Value::Array(tasks.iter().map(|t| json!({ "id": t.id, "desc": t.desc, "status": if t.done { "done" } else { "todo" } })).collect())).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replies by role from a script; records which roles ran.
    struct Script {
        calls: Vec<String>,
        manage_done_at: usize,
    }
    impl Llm for Script {
        fn call(&mut self, role: &str, _s: &str, user: &str, _t: f32) -> Result<Reply, String> {
            self.calls.push(role.to_owned());
            let text = match role {
                "plan" => "<think>hmm</think>### PLAN\nSplit and solve.\n### TASKS\n- Find the formula\n- Check edge cases".to_owned(),
                "ideate" => "### NOTES\nTwo ways.\n### NEXT\n- Closed form\n- Iterate".to_owned(),
                "manage" => {
                    let n = self.calls.iter().filter(|c| *c == "manage").count();
                    if n >= self.manage_done_at {
                        "### STATUS\ndone\n### TASKS\n- [done] Find the formula".to_owned()
                    } else {
                        "### STATUS\ncontinue\n### NEXT\n- Try the closed form\n- Try iteration\n- Extra task beyond workers\n### TASKS\n- [todo] Find the formula\n- [done] Check edge cases".to_owned()
                    }
                }
                r if r.starts_with("worker") => {
                    assert!(user.contains("YOUR TASK:\nComplete this task: Try"), "workers get their own task");
                    format!("### RESULT\nAnswer from {r}: 42\n### NOTES\n- **Found:** 42 by {r}\n### NEXT\n- none\n### STATUS\ndone")
                }
                "finalize" => {
                    assert!(user.contains("Answer from worker"), "finalize sees the ledger");
                    "### RESULT\nThe answer is 42.\n### NOTES\n- **Final:** 42\n### NEXT\nnone\n### STATUS\ndone".to_owned()
                }
                other => panic!("unexpected role {other}"),
            };
            Ok(Reply { text, cut_off: false })
        }
        fn cancelled(&self) -> bool {
            false
        }
    }

    #[test]
    fn plan_ideate_manage_workers_finalize_through_the_ledger() {
        let dir = std::env::temp_dir().join(format!("cerebro-gvs5h-test-{}", std::process::id()));
        let ledger = Ledger::create(&dir, "What is 6 x 7?").unwrap();
        let mut llm = Script { calls: Vec::new(), manage_done_at: 2 };
        let mut files = Vec::new();
        let out = run(&mut llm, &ledger, "What is 6 x 7?", "", Options { workers: 2, rounds: 4 }, &mut |p| {
            if let Progress::File { name, .. } = p {
                files.push(name);
            }
        })
        .unwrap();
        assert_eq!(out, "The answer is 42.");
        assert_eq!(llm.calls, ["plan", "ideate", "manage", "worker 1", "worker 2", "manage", "finalize"], "two workers per round, done on the second manage");
        assert!(ledger.read("plan.md").starts_with("Split and solve."), "reasoning stripped");
        assert!(ledger.read("notes.md").contains("**Final:** 42"));
        assert!(ledger.read("tasks.json").contains("\"done\""));
        assert_eq!(ledger.read("transcript.jsonl").lines().count(), 7);
        assert!(files.contains(&"result.md") && files.contains(&"notes.md"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_same_tasks_twice_stop_the_loop() {
        let dir = std::env::temp_dir().join(format!("cerebro-gvs5h-test2-{}", std::process::id()));
        let ledger = Ledger::create(&dir, "g").unwrap();
        let mut llm = Script { calls: Vec::new(), manage_done_at: 99 };
        run(&mut llm, &ledger, "g", "", Options { workers: 2, rounds: 8 }, &mut |_| {}).unwrap();
        let managers = llm.calls.iter().filter(|c| *c == "manage").count();
        assert_eq!(managers, 2, "second identical assignment stops it");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn parsing_helpers() {
        let s = sections("### PLAN\nA\n**TASKS**\n- one\n1. two\n2) three\nNEXT:\n* four");
        assert_eq!(s["PLAN"], "A");
        assert_eq!(bullets(&s["TASKS"]), ["one", "two", "three"]);
        assert_eq!(bullets(&s["NEXT"]), ["four"]);
        let t = parse_tasks("- [done] a\n- [todo] b\n- c");
        assert_eq!(t.iter().map(|t| (t.desc.as_str(), t.done)).collect::<Vec<_>>(), [("a", true), ("b", false), ("c", false)]);
        assert_eq!(strip_think("<think>x</think> y <think>unclosed"), "y");
        assert_eq!(clip("héllo", 2), "hé");
    }
}
