//! The Benchmark lab on Cerebro Engine: one model, three prompt depths, and a fixed 20-question
//! accuracy check. Every number is measured on this machine; a depth the model cannot hold is kept
//! in the result as "not run" with the reason, never estimated.
//!
//! Prompt reading, writing, first-token time, output length and system memory in use are recorded
//! per depth. The model is loaded once at the deepest context it supports, and `Session::clear()`
//! empties the conversation between points, so each depth starts from a fresh cache.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cerebro_engine::chat::{generate, GenConfig};
use cerebro_engine::model::{Model, Session};
use cerebro_engine::sample::SamplerConfig;
use cerebro_engine::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::engine::LibraryModel;

/// The three depths the screen asks for.
pub const CONTEXTS: [usize; 3] = [0, 4096, 16_384];
/// Tokens written at each depth. Long enough for a stable writing speed, short enough to keep the
/// whole run tolerable on a processor.
const OUTPUT_TOKENS: usize = 48;
/// Tokens kept free at the end of a full prompt for the answer.
const ANSWER_HEADROOM: usize = 64;
/// The accuracy check's answer limit.
const ANSWER_TOKENS: usize = 16;
/// Runs kept in the history file.
pub const MAX_HISTORY: usize = 200;

pub type BenchSink = Arc<dyn Fn(BenchEvent) + Send + Sync>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchModelInfo {
    pub id: String,
    pub name: String,
    pub quant: String,
    pub params: String,
    pub size_gi_b: f64,
    pub context_max: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchMachineInfo {
    pub cpu: String,
    pub gpu: String,
    pub ram_total_gi_b: f64,
    pub vram_total_gi_b: f64,
    pub build: String,
}

/// Where the model ran: the placement the user asked for, and where the weights ended up.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchPlacement {
    /// What the user chose: "auto", "gpu", "split" or "cpu".
    pub choice: String,
    /// Where the weights went: "gpu-int8", "gpu-native", "split" or "cpu".
    pub layout: String,
    /// Layers on the graphics card, and how many the model has.
    pub gpu_layers: usize,
    pub total_layers: usize,
    /// The drive that serves the part of the model that does not fit in memory.
    pub drive: String,
    /// The engine's own words about this placement.
    pub notes: Vec<String>,
}

/// The placement choices the Benchmark lab offers.
pub const PLACEMENTS: [&str; 4] = ["auto", "gpu", "split", "cpu"];

/// The number of graphics-card layers a choice asks for. "auto" keeps the placement the model is
/// loaded with (`loaded`), so the benchmark measures the placement the user chose; None means the
/// engine's own choice, as many layers as fit.
pub fn forced_layers(choice: &str, layers: usize, loaded: Option<usize>) -> Option<usize> {
    match choice {
        "gpu" => Some(layers),
        "split" => Some(layers / 2),
        "cpu" => Some(0),
        _ => loaded,
    }
}

/// The placement a run records: the choice, and the plan the engine made.
pub fn placement_of(choice: &str, plan: Option<&cerebro_engine::placement::Plan>, gpu_layers: usize, total_layers: usize) -> BenchPlacement {
    let (layout, drive, notes) = match plan {
        Some(p) => (layout_name(p.layout), drive_name(p.drive), p.notes.clone()),
        None => ("cpu".to_owned(), "unknown".to_owned(), Vec::new()),
    };
    BenchPlacement {
        choice: choice.to_owned(),
        layout,
        gpu_layers,
        total_layers,
        drive,
        notes,
    }
}

/// The plan's layout as the app names it.
fn layout_name(layout: cerebro_engine::placement::Layout) -> String {
    use cerebro_engine::placement::Layout;
    match layout {
        Layout::GpuInt8 => "gpu-int8".to_owned(),
        Layout::GpuNative => "gpu-native".to_owned(),
        Layout::Split => "split".to_owned(),
        Layout::Cpu => "cpu".to_owned(),
    }
}

/// The plan's drive as the app names it.
fn drive_name(drive: cerebro_engine::placement::Drive) -> String {
    use cerebro_engine::placement::Drive;
    match drive {
        Drive::Nvme => "nvme".to_owned(),
        Drive::SataSsd => "sata-ssd".to_owned(),
        Drive::HardDisk => "hard-disk".to_owned(),
        Drive::Removable => "removable".to_owned(),
        Drive::Unknown => "unknown".to_owned(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchPoint {
    pub context: usize,
    pub ran: bool,
    pub note: Option<String>,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub ttft_ms: f64,
    pub memory_used_gi_b: f64,
    pub seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchQuestionResult {
    pub id: String,
    pub category: String,
    pub question: String,
    pub expected: String,
    pub answer: String,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchAccuracy {
    pub asked: usize,
    pub correct: usize,
    pub questions: Vec<BenchQuestionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchRun {
    pub id: String,
    pub at: String,
    pub provenance: String,
    pub status: String,
    pub model: BenchModelInfo,
    pub machine: BenchMachineInfo,
    /// Where the model ran: the placement the user chose and the plan the engine made.
    pub placement: BenchPlacement,
    pub points: Vec<BenchPoint>,
    pub accuracy: BenchAccuracy,
    pub seconds: f64,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum BenchEvent {
    Progress { phase: &'static str, text: String, context: Option<usize>, done: usize, total: usize },
    Point { point: BenchPoint },
    Accuracy { result: BenchAccuracy },
}

#[derive(Debug, Clone, Copy)]
pub struct Question {
    pub id: &'static str,
    pub category: &'static str,
    pub question: &'static str,
    pub expected: &'static str,
}

/// Fixed, in the same order as ui/src/lib/bench.ts.
pub const QUESTIONS: [Question; 20] = [
    Question { id: "arith-1", category: "arithmetic", question: "What is 17 + 26?", expected: "43" },
    Question { id: "arith-2", category: "arithmetic", question: "What is 84 divided by 7?", expected: "12" },
    Question { id: "arith-3", category: "arithmetic", question: "What is 9 times 12?", expected: "108" },
    Question { id: "arith-4", category: "arithmetic", question: "What is 125 minus 48?", expected: "77" },
    Question { id: "units-1", category: "units", question: "How many centimetres are in 2.5 metres?", expected: "250" },
    Question { id: "units-2", category: "units", question: "How many minutes are in 3 hours?", expected: "180" },
    Question { id: "units-3", category: "units", question: "How many grams are in 1.5 kilograms?", expected: "1500" },
    Question { id: "units-4", category: "units", question: "How many millilitres are in 0.75 litres?", expected: "750" },
    Question { id: "dates-1", category: "dates", question: "What day comes 10 days after Monday?", expected: "Thursday" },
    Question { id: "dates-2", category: "dates", question: "If today is March 28, what date is 5 days later?", expected: "April 2" },
    Question { id: "dates-3", category: "dates", question: "How many days are in February 2024?", expected: "29" },
    Question { id: "dates-4", category: "dates", question: "What month is 4 months after October?", expected: "February" },
    Question { id: "letters-1", category: "letters", question: "How many times does the letter e appear in elephant?", expected: "2" },
    Question { id: "letters-2", category: "letters", question: "How many times does the letter s appear in mississippi?", expected: "4" },
    Question { id: "letters-3", category: "letters", question: "How many letters are in the word strawberry?", expected: "10" },
    Question { id: "letters-4", category: "letters", question: "How many vowels are in the word education?", expected: "5" },
    Question { id: "logic-1", category: "logic", question: "All roses are flowers. This plant is a rose. Is it a flower? Answer yes or no.", expected: "yes" },
    Question { id: "logic-2", category: "logic", question: "A is taller than B, and B is taller than C. Who is shortest? Answer A, B or C.", expected: "C" },
    Question { id: "logic-3", category: "logic", question: "Some birds can fly. Penguins are birds. Can we conclude that penguins can fly? Answer yes or no.", expected: "no" },
    Question { id: "logic-4", category: "logic", question: "If it rains, the ground gets wet. The ground is wet. Can we conclude that it rained? Answer yes or no.", expected: "no" },
];

/// Lowercase, no punctuation, single spaces; the same normalisation ui/src/lib/bench.ts uses.
pub fn normalize_answer(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        // Markdown emphasis is dropped, as the UI's grader does; everything else that is not a
        // word character becomes a space, so a full stop never hides the answer at the end.
        if ch == '`' || ch == '*' || ch == '_' {
            continue;
        }
        if ch.is_alphanumeric() || ch == '-' {
            out.extend(ch.to_lowercase());
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn push_number(current: &mut String, out: &mut Vec<f64>) {
    let text = current.trim_end_matches('.');
    if let Ok(value) = text.parse::<f64>() {
        out.push(value);
    }
    current.clear();
}

/// Every number written, in order. Thousands separators are removed first, and a trailing period
/// is punctuation, not part of the number ("43." is 43).
fn numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars().filter(|&c| c != ',') {
        if ch.is_ascii_digit() || ch == '.' || (ch == '-' && current.is_empty()) {
            current.push(ch);
        } else if !current.is_empty() {
            push_number(&mut current, &mut out);
        }
    }
    if !current.is_empty() {
        push_number(&mut current, &mut out);
    }
    out
}

/// True when `answer` gives the question's one expected answer. Numbers compare by the last number
/// written ("17 + 26 = 43"), yes/no by the first yes or no, a letter by a standalone letter, and
/// words as whole words. A short explanation around the right answer is accepted; a different
/// answer is not.
pub fn answer_matches(expected: &str, answer: &str) -> bool {
    let e = normalize_answer(expected);
    let a = normalize_answer(answer);
    if a.is_empty() {
        return false;
    }
    if let Ok(want) = e.parse::<f64>() {
        return numbers(answer).last().is_some_and(|value| (*value - want).abs() < 1e-9);
    }
    if e == "yes" || e == "no" {
        return a.split_whitespace().find(|w| *w == "yes" || *w == "no") == Some(e.as_str());
    }
    if e.len() == 1 && e.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        return a.split_whitespace().any(|w| w == e);
    }
    format!(" {a} ").contains(&format!(" {e} "))
}

fn short_answer(text: &str) -> String {
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    one_line.chars().take(120).collect()
}

/// Grades one answer, keeping what the model wrote for the report.
pub fn grade(question: &Question, answer: &str) -> BenchQuestionResult {
    BenchQuestionResult {
        id: question.id.to_owned(),
        category: question.category.to_owned(),
        question: question.question.to_owned(),
        expected: question.expected.to_owned(),
        answer: short_answer(answer),
        correct: answer_matches(question.expected, answer),
    }
}

/// Which depths this model can hold, with the reason when it cannot.
pub fn point_plan(context_max: u64) -> Vec<(usize, Option<String>)> {
    let max = context_max as usize;
    CONTEXTS
        .iter()
        .map(|&context| {
            if context == 0 {
                (context, None)
            } else if max == 0 {
                // Unknown is not the same as unlimited: only the short prompt is measured.
                (context, Some("This model does not report a context size, so this depth was not run.".to_owned()))
            } else if max >= context {
                (context, None)
            } else {
                (context, Some(format!("This model holds {max} tokens, so it cannot read a {context}-token prompt.")))
            }
        })
        .collect()
}

/// The context to load the session with: the deepest measurable point plus room for the answer.
pub fn session_ctx(context_max: u64) -> usize {
    let max = context_max as usize;
    let deepest = if max >= 16_384 {
        16_384
    } else if max >= 4_096 {
        4_096
    } else {
        0
    };
    if deepest == 0 {
        max.clamp(512, 4096)
    } else {
        (deepest + 256).min(max).max(512)
    }
}

/// The prompt size actually used at `context` (the model's window may need a little headroom).
pub fn prompt_size(context: usize, context_max: u64) -> (usize, Option<String>) {
    if context == 0 {
        return (0, None);
    }
    let max = context_max as usize;
    let want = context.min(max.saturating_sub(ANSWER_HEADROOM));
    if want < context {
        (want, Some(format!("The prompt used {} tokens so the answer fits in the model's {}-token context.", want, max)))
    } else {
        (want, None)
    }
}

/// What to measure at one planned depth: the prompt size and the note to keep beside the point,
/// or the reason the depth is not run at all. A depth the model cannot hold is never measured at a
/// shorter prompt and then stored under the depth that was asked for.
pub fn depth_prompt(context: usize, context_max: u64, unavailable: Option<String>) -> Result<(usize, Option<String>), String> {
    match unavailable {
        Some(reason) => Err(reason),
        None => Ok(prompt_size(context, context_max)),
    }
}

fn iso_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, min, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

fn model_info(m: &LibraryModel) -> BenchModelInfo {
    BenchModelInfo {
        id: m.id.clone(),
        name: m.name.clone(),
        quant: m.quant.clone(),
        params: m.params.clone(),
        size_gi_b: (m.size_gi_b * 100.0).round() / 100.0,
        context_max: m.context_max,
    }
}

fn machine_info() -> BenchMachineInfo {
    let hw = crate::engine::hardware();
    BenchMachineInfo {
        cpu: format!("{} ({} cores, {} threads)", hw.cpu.name, hw.cpu.cores, hw.cpu.threads),
        gpu: hw.gpu.as_ref().map_or("none".to_owned(), |g| g.name.clone()),
        ram_total_gi_b: (hw.ram_total_gi_b * 10.0).round() / 10.0,
        vram_total_gi_b: hw.gpu.as_ref().map_or(0.0, |g| (g.vram_total_gi_b * 10.0).round() / 10.0),
        build: crate::tune_screen::build(),
    }
}

/// System memory in use, GiB (the same figure the Monitor's RAM chart reads).
fn memory_used_gi_b() -> f64 {
    let (total, available) = cerebro_engine::hw::memory();
    ((total.saturating_sub(available)) as f64 / (1u64 << 30) as f64 * 100.0).round() / 100.0
}

/// A deterministic long document. It is only a way to fill the context: its content is not graded.
fn document(words: usize) -> String {
    const NOUNS: &[&str] = &["river", "ledger", "engine", "harbor", "garden", "signal", "archive", "bridge", "market", "lantern", "valley", "council", "orchard", "compass", "furnace", "meadow"];
    const VERBS: &[&str] = &["measures", "carries", "records", "shapes", "guards", "follows", "repairs", "divides", "gathers", "explains", "crosses", "balances"];
    const ADJ: &[&str] = &["quiet", "northern", "patient", "copper", "early", "narrow", "careful", "distant", "steady", "bright", "hollow", "modest"];
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    let mut s = String::new();
    let mut count = 0;
    let mut i = 0;
    while count < words {
        i += 1;
        let sentence = format!(
            "In section {i}, the {} {} {} the {} {} while the {} {} waits by the {}. ",
            ADJ[next(ADJ.len())],
            NOUNS[next(NOUNS.len())],
            VERBS[next(VERBS.len())],
            ADJ[next(ADJ.len())],
            NOUNS[next(NOUNS.len())],
            ADJ[next(ADJ.len())],
            NOUNS[next(NOUNS.len())],
            NOUNS[next(NOUNS.len())]
        );
        count += sentence.split_whitespace().count();
        s.push_str(&sentence);
        if i % 6 == 0 {
            s.push_str("\n\n");
        }
    }
    s
}

fn short_prompt(tok: &Tokenizer) -> Vec<u32> {
    tok.encode(&cerebro_engine::chatml(None, &[("user", "Explain in one sentence why the sky is blue.")], false))
}

/// A prompt of about `tokens` tokens: the chat wrapper, a deterministic document, then a question
/// about it. `0` is the short prompt, not an empty one.
fn long_prompt(tok: &Tokenizer, tokens: usize) -> Vec<u32> {
    if tokens == 0 {
        return short_prompt(tok);
    }
    let prefix = tok.encode("<|im_start|>user\nHere is a long document.\n\n");
    // The same tail chatml(..., false) writes: the empty think block asks for a direct answer.
    let suffix = tok.encode("\n\nIn one sentence, what is this document mostly about?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let room = tokens.saturating_sub(prefix.len() + suffix.len());
    if room < 16 {
        return short_prompt(tok);
    }
    let mut filler = tok.encode(&document(14_000));
    if filler.len() < room {
        while filler.len() < room {
            let extra = tok.encode(&document(1_000));
            if extra.is_empty() {
                break;
            }
            filler.extend(extra);
        }
    }
    let mut prompt = Vec::with_capacity(tokens);
    prompt.extend_from_slice(&prefix);
    prompt.extend_from_slice(&filler[..room.min(filler.len())]);
    prompt.extend_from_slice(&suffix);
    prompt
}

fn cfg(max_tokens: usize) -> GenConfig {
    GenConfig { sampler: SamplerConfig { temperature: 0.0, ..Default::default() }, max_tokens }
}

/// A depth that was not measured, kept in the result so the report never implies it ran.
fn not_run(context: usize, note: String) -> BenchPoint {
    BenchPoint {
        context,
        ran: false,
        note: Some(note),
        prompt_tokens: 0,
        output_tokens: 0,
        prefill_tps: 0.0,
        decode_tps: 0.0,
        ttft_ms: 0.0,
        memory_used_gi_b: 0.0,
        seconds: 0.0,
    }
}

/// Measures one depth with a prompt of `tokens` tokens. A failure at one depth is recorded honestly
/// as not run, not estimated.
fn measure(model: &Model, session: &mut Session, tok: &Tokenizer, context: usize, tokens: usize, note: Option<String>, cancel: &AtomicBool) -> BenchPoint {
    if let Err(e) = session.clear() {
        return not_run(context, format!("The session could not be cleared: {e}"));
    }
    let prompt = long_prompt(tok, tokens);
    let before = memory_used_gi_b();
    let t0 = Instant::now();
    let s = match generate(model, session, tok, &prompt, &cfg(OUTPUT_TOKENS), |_| !cancel.load(Ordering::SeqCst)) {
        Ok(stats) => stats,
        Err(e) => return not_run(context, format!("The engine could not measure this depth: {e}")),
    };
    if cancel.load(Ordering::SeqCst) {
        return not_run(context, "Stopped before this depth finished.".to_owned());
    }
    let after = memory_used_gi_b();
    BenchPoint {
        context,
        ran: true,
        note,
        prompt_tokens: s.prompt_tokens,
        output_tokens: s.output_tokens,
        prefill_tps: (s.prefill_tps * 10.0).round() / 10.0,
        decode_tps: (s.decode_tps * 10.0).round() / 10.0,
        ttft_ms: s.ttft_ms.round(),
        memory_used_gi_b: before.max(after),
        seconds: (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
    }
}

/// Runs the whole benchmark on an already loaded model and session. The caller leaves the model
/// loaded afterwards; this function clears the session between points and questions.
pub fn run(model: &Model, session: &mut Session, tok: &Tokenizer, info: &LibraryModel, placement: BenchPlacement, sink: &BenchSink, cancel: &AtomicBool) -> BenchRun {
    let started = Instant::now();
    let at = iso_now();
    let run_id = format!("bench-{}-{}", at.chars().filter(|c| c.is_ascii_digit()).collect::<String>(), info.id);
    let plan = point_plan(info.context_max);
    let total = plan.len() + QUESTIONS.len();
    let mut points: Vec<BenchPoint> = Vec::with_capacity(plan.len());
    let mut questions: Vec<BenchQuestionResult> = Vec::new();
    let mut done = 0usize;
    let mut stopped = false;

    for (context, note) in plan {
        if cancel.load(Ordering::SeqCst) {
            stopped = true;
            break;
        }
        let label = if context == 0 {
            "short prompt".to_owned()
        } else {
            format!("{} tokens", context)
        };
        let (size, note) = match depth_prompt(context, info.context_max, note) {
            Ok(prompt) => prompt,
            Err(reason) => {
                let point = not_run(context, reason);
                points.push(point.clone());
                done += 1;
                sink(BenchEvent::Point { point });
                continue;
            }
        };
        sink(BenchEvent::Progress {
            phase: "measure",
            text: if context == 0 { "Measuring a short prompt…".to_owned() } else { format!("Reading {size} tokens ({label})…") },
            context: Some(context),
            done,
            total,
        });
        let point = measure(model, session, tok, context, size, note, cancel);
        if !point.ran && cancel.load(Ordering::SeqCst) {
            stopped = true;
            break;
        }
        points.push(point.clone());
        done += 1;
        sink(BenchEvent::Point { point });
    }

    if !stopped {
        for q in &QUESTIONS {
            if cancel.load(Ordering::SeqCst) {
                stopped = true;
                break;
            }
            sink(BenchEvent::Progress {
                phase: "accuracy",
                text: format!("Question {} of {}…", questions.len() + 1, QUESTIONS.len()),
                context: None,
                done,
                total,
            });
            let _ = session.clear();
            let prompt = tok.encode(&cerebro_engine::chatml(None, &[("user", &format!("{} Answer with only the answer.", q.question))], false));
            let mut text = String::new();
            let result = generate(model, session, tok, &prompt, &cfg(ANSWER_TOKENS), |piece| {
                text.push_str(piece);
                !cancel.load(Ordering::SeqCst)
            });
            if cancel.load(Ordering::SeqCst) {
                stopped = true;
                break;
            }
            if result.is_err() && text.is_empty() {
                text.push_str("The engine could not answer this question.");
            }
            questions.push(grade(q, &text));
            done += 1;
        }
    }

    let accuracy = BenchAccuracy {
        asked: questions.len(),
        correct: questions.iter().filter(|q| q.correct).count(),
        questions,
    };
    sink(BenchEvent::Accuracy { result: accuracy.clone() });
    BenchRun {
        id: run_id,
        at,
        provenance: "measured".to_owned(),
        status: if stopped { "stopped" } else { "done" }.to_owned(),
        model: model_info(info),
        machine: machine_info(),
        placement,
        points,
        accuracy,
        seconds: (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        stop_reason: stopped.then(|| "Stopped from the Benchmark lab.".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(expected: &'static str) -> Question {
        Question { id: "t", category: "logic", question: "q", expected }
    }

    #[test]
    fn exact_answers_are_graded_by_value_not_phrasing() {
        assert!(answer_matches("43", "43"));
        assert!(answer_matches("43", "The answer is 43."));
        assert!(answer_matches("43", "17 + 26 = 43"));
        assert!(answer_matches("250", "2.5 metres is 250 centimetres"));
        assert!(answer_matches("1500", "1,500 grams"));
        assert!(!answer_matches("43", "42"));
        assert!(!answer_matches("43", "17 + 26 is 43? No, 42."));
    }

    #[test]
    fn yes_no_letters_and_words_are_graded() {
        assert!(answer_matches("yes", "Yes."));
        assert!(answer_matches("no", "No, that does not follow."));
        assert!(!answer_matches("no", "Yes, because all birds fly."));
        assert!(answer_matches("C", "C"));
        assert!(answer_matches("C", "C is shortest."));
        assert!(!answer_matches("C", "B"));
        assert!(answer_matches("April 2", "April 2"));
        assert!(answer_matches("April 2", "It will be April 2, 2024."));
        assert!(!answer_matches("April 2", "March 31"));
        assert!(answer_matches("Thursday", "The day is Thursday."));
        assert_eq!(normalize_answer("  The  Answer,  is: 43!  "), "the answer is 43");
    }

    #[test]
    fn the_answer_written_is_kept_short_and_plain() {
        let r = grade(&q("43"), "  The answer\n is 43.  ");
        assert!(r.correct);
        assert_eq!(r.answer, "The answer is 43.");
        assert_eq!(r.expected, "43");
    }

    #[test]
    fn depths_beyond_the_models_window_are_marked_not_run() {
        let p = point_plan(4096);
        assert_eq!(p[0], (0, None));
        assert_eq!(p[1], (4096, None));
        assert!(p[2].1.as_deref().unwrap().contains("cannot read a 16384"));
        let p = point_plan(2048);
        assert!(p[1].1.is_some() && p[2].1.is_some());
        let p = point_plan(0);
        assert!(p[1].1.as_deref().unwrap().contains("does not report"));
    }

    #[test]
    fn a_full_window_keeps_room_for_the_answer() {
        assert_eq!(prompt_size(4096, 4096), (4032, Some("The prompt used 4032 tokens so the answer fits in the model's 4096-token context.".into())));
        assert_eq!(prompt_size(16384, 32768), (16384, None));
        assert_eq!(session_ctx(32768), 16640);
        assert_eq!(session_ctx(4096), 4096);
        assert_eq!(session_ctx(8192), 4352);
    }

    #[test]
    fn a_depth_the_model_cannot_hold_is_not_run_at_a_shorter_prompt() {
        // A 16K point on a 4K model must never be measured as a 4032-token prompt and stored as 16K.
        let plan = point_plan(4096);
        let (context, unavailable) = plan[2].clone();
        assert_eq!(context, 16_384);
        let err = depth_prompt(context, 4096, unavailable).unwrap_err();
        assert!(err.contains("cannot read a 16384"));
        // The deepest supported point is measured at the asked-for size, not resized.
        let (context, unavailable) = plan[1].clone();
        assert_eq!(context, 4096);
        assert_eq!(depth_prompt(context, 4096, unavailable), Ok((4032, Some("The prompt used 4032 tokens so the answer fits in the model's 4096-token context.".into()))));
        // The short prompt always runs.
        let (context, unavailable) = plan[0].clone();
        assert_eq!(depth_prompt(context, 0, unavailable), Ok((0, None)));
    }

    #[test]
    fn dates_are_iso_and_stable_in_shape() {
        let at = iso_now();
        assert_eq!(at.len(), 20);
        assert_eq!(&at[4..5], "-");
        assert_eq!(&at[10..11], "T");
        assert!(at.ends_with('Z'));
    }

    #[test]
    fn the_document_is_deterministic_and_long_enough() {
        assert_eq!(document(300), document(300));
        assert!(document(300).split_whitespace().count() >= 300);
    }

    #[test]
    fn a_placement_choice_asks_for_the_layers_it_names() {
        assert_eq!(forced_layers("gpu", 40, None), Some(40));
        assert_eq!(forced_layers("split", 40, None), Some(20));
        assert_eq!(forced_layers("cpu", 40, None), Some(0));
        // "auto" keeps the placement the model is loaded with, so the user's choice is measured.
        assert_eq!(forced_layers("auto", 40, Some(12)), Some(12));
        assert_eq!(forced_layers("auto", 40, None), None);
    }

    #[test]
    fn a_run_records_where_the_weights_went() {
        let plan = cerebro_engine::placement::Plan {
            layout: cerebro_engine::placement::Layout::Split,
            gpu_layers: 20,
            total_layers: 40,
            vram_bytes: 1024,
            cpu_bytes: 2048,
            ram_resident: 4096,
            drive_bytes_per_token: 0,
            drive_layers: 0,
            est_decode_tps: 12.0,
            drive: cerebro_engine::placement::Drive::Nvme,
            notes: vec!["20 of 40 layers on the graphics card.".to_owned()],
            head_on_gpu: false,
        };
        let p = placement_of("split", Some(&plan), 20, 40);
        assert_eq!(p.choice, "split");
        assert_eq!(p.layout, "split");
        assert_eq!(p.gpu_layers, 20);
        assert_eq!(p.total_layers, 40);
        assert_eq!(p.drive, "nvme");
        assert_eq!(p.notes.len(), 1);
        // A model with no plan at all still says where it ran.
        let none = placement_of("cpu", None, 0, 40);
        assert_eq!(none.layout, "cpu");
        assert_eq!(none.drive, "unknown");
    }
}
