/**
 * Benchmark lab helpers: the fixed 20-question accuracy check, its automatic grading, and the
 * history/comparison rows the screen draws. The Rust side (src-tauri/src/bench.rs) uses the same
 * questions and the same grading rules.
 */
import type { BenchAccuracy, BenchCategory, BenchPoint, BenchQuestionResult, BenchRun, BenchStatus } from "../service/bench-types";
import { BENCH_CONTEXTS } from "../service/bench-types";

export interface BenchQuestion {
  id: string;
  category: BenchCategory;
  question: string;
  /** The exact answer the grader accepts, after normalisation. */
  expected: string;
}

/** Fixed: twenty short questions, four in each category, with one exact answer each. */
export const BENCH_QUESTIONS: readonly BenchQuestion[] = [
  { id: "arith-1", category: "arithmetic", question: "What is 17 + 26?", expected: "43" },
  { id: "arith-2", category: "arithmetic", question: "What is 84 divided by 7?", expected: "12" },
  { id: "arith-3", category: "arithmetic", question: "What is 9 times 12?", expected: "108" },
  { id: "arith-4", category: "arithmetic", question: "What is 125 minus 48?", expected: "77" },
  { id: "units-1", category: "units", question: "How many centimetres are in 2.5 metres?", expected: "250" },
  { id: "units-2", category: "units", question: "How many minutes are in 3 hours?", expected: "180" },
  { id: "units-3", category: "units", question: "How many grams are in 1.5 kilograms?", expected: "1500" },
  { id: "units-4", category: "units", question: "How many millilitres are in 0.75 litres?", expected: "750" },
  { id: "dates-1", category: "dates", question: "What day comes 10 days after Monday?", expected: "Thursday" },
  { id: "dates-2", category: "dates", question: "If today is March 28, what date is 5 days later?", expected: "April 2" },
  { id: "dates-3", category: "dates", question: "How many days are in February 2024?", expected: "29" },
  { id: "dates-4", category: "dates", question: "What month is 4 months after October?", expected: "February" },
  { id: "letters-1", category: "letters", question: "How many times does the letter e appear in elephant?", expected: "2" },
  { id: "letters-2", category: "letters", question: "How many times does the letter s appear in mississippi?", expected: "4" },
  { id: "letters-3", category: "letters", question: "How many letters are in the word strawberry?", expected: "10" },
  { id: "letters-4", category: "letters", question: "How many vowels are in the word education?", expected: "5" },
  { id: "logic-1", category: "logic", question: "All roses are flowers. This plant is a rose. Is it a flower? Answer yes or no.", expected: "yes" },
  { id: "logic-2", category: "logic", question: "A is taller than B, and B is taller than C. Who is shortest? Answer A, B or C.", expected: "C" },
  { id: "logic-3", category: "logic", question: "Some birds can fly. Penguins are birds. Can we conclude that penguins can fly? Answer yes or no.", expected: "no" },
  { id: "logic-4", category: "logic", question: "If it rains, the ground gets wet. The ground is wet. Can we conclude that it rained? Answer yes or no.", expected: "no" },
];

export const BENCH_CATEGORY_LABELS: Readonly<Record<BenchCategory, string>> = {
  arithmetic: "Arithmetic",
  units: "Units",
  dates: "Dates",
  letters: "Letters in words",
  logic: "Simple logic",
};

/** Lowercase, no punctuation, single spaces; the same normalisation the Rust grader uses. */
export function normalizeAnswer(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_]/g, "")
    .replace(/[.,!?;:'"()[\]{}]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function numbersIn(text: string): number[] {
  return (text.replace(/,/g, "").match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
}

function numberIn(text: string): number | null {
  const numbers = numbersIn(text);
  if (numbers.length === 0) return null;
  // The last number is the answer in the usual phrasing ("17 + 26 = 43").
  return numbers[numbers.length - 1]!;
}

/**
 * True when `answer` gives the question's exact answer. Numbers are compared by value, yes/no by
 * the first yes or no, a letter choice by a standalone letter, and words by whole-word match. A
 * short explanation is accepted; a different answer is not.
 */
export function answerMatches(expected: string, answer: string): boolean {
  const e = normalizeAnswer(expected);
  const a = normalizeAnswer(answer);
  if (!a) return false;
  if (/^-?\d+(?:\.\d+)?$/.test(e)) {
    const want = Number(e);
    // Compare from the raw answer: normalisation turns the thousands separator in "1,500" into a
    // space, which would read as two numbers.
    return numberIn(answer) === want;
  }
  if (e === "yes" || e === "no") {
    const found = a.match(/\b(yes|no)\b/);
    return found?.[1] === e;
  }
  if (/^[a-c]$/.test(e)) {
    return new RegExp(`(?:^|\\b)${e}(?:\\b|$)`).test(a);
  }
  return a === e || new RegExp(`(?:^|\\b)${e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\b|$)`).test(a);
}

/** Grades one answer; `answer` is kept short and plain for the report. */
export function gradeAnswer(question: BenchQuestion, answer: string): BenchQuestionResult {
  const shown = answer.replace(/\s+/g, " ").trim().slice(0, 120);
  return {
    id: question.id,
    category: question.category,
    question: question.question,
    expected: question.expected,
    answer: shown,
    correct: answerMatches(question.expected, answer),
  };
}

/** Grades a set of answers by question id; unanswered questions count as wrong. */
export function gradeAnswers(answers: Readonly<Record<string, string>>): BenchAccuracy {
  const questions = BENCH_QUESTIONS.map((q) => gradeAnswer(q, answers[q.id] ?? ""));
  return { asked: questions.length, correct: questions.filter((q) => q.correct).length, questions };
}

export const CONTEXT_LABELS: Readonly<Record<number, string>> = {
  0: "Short prompt",
  4096: "4K",
  16384: "16K",
};

export function contextLabel(context: number): string {
  return CONTEXT_LABELS[context] ?? context.toLocaleString();
}

export function contextDetail(context: number): string {
  return context === 0 ? "A short prompt (no long context)" : `${context.toLocaleString()} tokens of context`;
}

/** A point for a depth that could not be measured on this model (kept, not hidden). */
export function notRunPoint(context: number, note: string): BenchPoint {
  return { context, ran: false, note, promptTokens: 0, outputTokens: 0, prefillTps: 0, decodeTps: 0, ttftMs: 0, memoryUsedGiB: 0, seconds: 0 };
}

export interface BenchComparisonRow {
  context: number;
  a: BenchPoint | null;
  b: BenchPoint | null;
  prefillPct: number | null;
  decodePct: number | null;
  ttftPct: number | null;
}

/** Positive means run B is faster (or quicker to first token) than run A. */
export function percentChange(b: number, a: number): number | null {
  if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0) return null;
  return ((b - a) / a) * 100;
}

export function compareRuns(a: BenchRun, b: BenchRun): BenchComparisonRow[] {
  return BENCH_CONTEXTS.map((context) => {
    const pa = a.points.find((p) => p.context === context) ?? null;
    const pb = b.points.find((p) => p.context === context) ?? null;
    const both = pa?.ran && pb?.ran ? [pa, pb] : null;
    return {
      context,
      a: pa,
      b: pb,
      prefillPct: both ? percentChange(pb!.prefillTps, pa!.prefillTps) : null,
      decodePct: both ? percentChange(pb!.decodeTps, pa!.decodeTps) : null,
      ttftPct: both ? percentChange(pb!.ttftMs, pa!.ttftMs) : null,
    };
  });
}

export interface BenchChartPoint {
  runId: string;
  at: string;
  value: number;
  label: string;
  provenance: BenchRun["provenance"];
}

/** The points of a history chart: one value per run at the chosen depth, oldest first. */
export function chartPoints(runs: readonly BenchRun[], context: number, metric: "decodeTps" | "prefillTps"): BenchChartPoint[] {
  return [...runs]
    .reverse()
    .flatMap((run) => {
      const point = run.points.find((p) => p.context === context && p.ran);
      if (!point || !Number.isFinite(point[metric]) || point[metric] <= 0) return [];
      return [{ runId: run.id, at: run.at, value: point[metric], label: shortDate(run.at), provenance: run.provenance }];
    });
}

export function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function statusLabel(status: BenchStatus): string {
  return status === "done" ? "Finished" : status === "stopped" ? "Stopped" : "Failed";
}

export function runTitle(run: BenchRun): string {
  return `${run.model.name} · ${shortDate(run.at)}`;
}

/** Scores shown beside the run: accuracy first, then the fastest measured depth. */
export function runSummary(run: BenchRun): string {
  const measured = run.points.filter((p) => p.ran);
  const best = measured.reduce<BenchPoint | null>((top, p) => (top === null || p.decodeTps > top.decodeTps ? p : top), null);
  const accuracy = `${run.accuracy.correct}/${run.accuracy.asked} correct`;
  return best ? `${accuracy} · ${best.decodeTps.toFixed(1)} tok/s writing at ${contextLabel(best.context).toLowerCase()}` : accuracy;
}

export function machineLine(run: BenchRun): string {
  const gpu = run.machine.gpu && run.machine.gpu !== "none" ? run.machine.gpu : "Processor only";
  return `${gpu} · ${run.machine.cpu} · ${run.machine.ramTotalGiB.toFixed(1)} GiB RAM · ${run.machine.build}`;
}

const DRIVE_NAMES: Readonly<Record<string, string>> = {
  nvme: "an NVMe SSD",
  "sata-ssd": "a SATA SSD",
  "hard-disk": "a hard disk",
  removable: "a USB drive or memory card",
  unknown: "a drive",
};

/** One line about where the model ran: the layers on the card, the rest, and the file's drive. */
export function placementLine(run: BenchRun): string {
  const p = run.placement;
  const where =
    p.gpuLayers <= 0
      ? "Processor only"
      : p.gpuLayers >= p.totalLayers
        ? `All ${p.totalLayers} layers on the graphics card`
        : `${p.gpuLayers} of ${p.totalLayers} layers on the graphics card, the rest on the processor`;
  return `${where} · model file on ${DRIVE_NAMES[p.drive] ?? "a drive"}`;
}

/** Whether a point is the latest for its depth, so the table can mark it without extra words. */
export function latestPoint(run: BenchRun, context: number): BenchPoint | null {
  return run.points.find((p) => p.context === context) ?? null;
}
