import { describe, expect, it } from "vitest";
import {
  BENCH_QUESTIONS,
  BENCH_CATEGORY_LABELS,
  answerMatches,
  chartPoints,
  compareRuns,
  contextDetail,
  contextLabel,
  gradeAnswer,
  gradeAnswers,
  machineLine,
  normalizeAnswer,
  notRunPoint,
  percentChange,
  placementLine,
  runSummary,
  runTitle,
  statusLabel,
} from "../../ui/src/lib/bench";
import { escapeHtml, reportFilename, reportHtml } from "../../ui/src/lib/bench-report";
import { createMockBench } from "../../ui/src/mock/bench";
import { MOCK_HARDWARE } from "../../ui/src/mock/fixtures";
import { MOCK_LIBRARY } from "../../ui/src/mock/library";
import { createMockService } from "../../ui/src/mock/mock-service";
import { BENCH_CONTEXTS, type BenchAccuracy, type BenchEvent, type BenchPoint, type BenchRun } from "../../ui/src/service/bench-types";
import type { LibraryModel } from "../../ui/src/service/types";

const model = (id: string): LibraryModel => {
  const found = MOCK_LIBRARY.find((m) => m.id === id);
  if (!found) throw new Error(`fixture ${id} not found`);
  return found;
};

const BASE = model("qwen3-coder-30b-a3b-q4km");
const SMALL_CTX: LibraryModel = { ...BASE, id: "small-ctx", name: "Small context model", contextMax: 6000 };

function point(context: number, values: Partial<BenchPoint> = {}): BenchPoint {
  return {
    context,
    ran: true,
    note: null,
    promptTokens: context === 0 ? 20 : context - 64,
    outputTokens: 48,
    prefillTps: 100,
    decodeTps: 10,
    ttftMs: 1000,
    memoryUsedGiB: 5,
    seconds: 2,
    ...values,
  };
}

function accuracy(correct = 17): BenchAccuracy {
  const questions = BENCH_QUESTIONS.map((q, i) => ({
    id: q.id,
    category: q.category,
    question: q.question,
    expected: q.expected,
    answer: i < correct ? q.expected : "wrong",
    correct: i < correct,
  }));
  return { asked: questions.length, correct: questions.filter((q) => q.correct).length, questions };
}

function run(id: string, points: BenchPoint[], at = "2026-10-20T10:00:00.000Z"): BenchRun {
  return {
    id,
    at,
    provenance: "measured",
    status: "done",
    model: { id: BASE.id, name: BASE.name, quant: BASE.quant, params: BASE.params, sizeGiB: BASE.sizeGiB, contextMax: BASE.contextMax },
    machine: { cpu: "Example CPU", gpu: "Example GPU", ramTotalGiB: 32, vramTotalGiB: 8, build: "test" },
    placement: { choice: "auto", layout: "split", gpuLayers: 20, totalLayers: 40, drive: "nvme", notes: [] },
    points,
    accuracy: accuracy(),
    seconds: 10,
    stopReason: null,
  };
}

const eventTypes = (events: BenchEvent[]): string[] => events.map((e) => e.type);

describe("the question set", () => {
  it("is fixed at twenty questions, four in each category", () => {
    expect(BENCH_QUESTIONS).toHaveLength(20);
    expect(new Set(BENCH_QUESTIONS.map((q) => q.id)).size).toBe(20);
    for (const category of Object.keys(BENCH_CATEGORY_LABELS)) {
      expect(BENCH_QUESTIONS.filter((q) => q.category === category)).toHaveLength(4);
    }
    for (const q of BENCH_QUESTIONS) {
      expect(q.question.trim().length).toBeGreaterThan(0);
      expect(q.expected.trim().length).toBeGreaterThan(0);
    }
  });

  it("normalises case, punctuation and spacing", () => {
    expect(normalizeAnswer("  The  Answer,  is: 43!  ")).toBe("the answer is 43");
    expect(normalizeAnswer("**yes**")).toBe("yes");
    expect(normalizeAnswer("April  2.")).toBe("april 2");
    expect(normalizeAnswer("")).toBe("");
  });
});

describe("grading", () => {
  it("accepts the exact number, a short explanation, and a thousands separator", () => {
    const a = BENCH_QUESTIONS.find((q) => q.expected === "43")!;
    expect(answerMatches(a.expected, "43")).toBe(true);
    expect(answerMatches(a.expected, "The answer is 43.")).toBe(true);
    expect(answerMatches(a.expected, "17 + 26 = 43")).toBe(true);
    expect(answerMatches(a.expected, "42")).toBe(false);
    expect(answerMatches(a.expected, "")).toBe(false);
    const units = BENCH_QUESTIONS.find((q) => q.expected === "1500")!;
    expect(answerMatches(units.expected, "1,500 grams")).toBe(true);
    expect(answerMatches(units.expected, "1500")).toBe(true);
    expect(answerMatches(units.expected, "1.5")).toBe(false);
  });

  it("accepts a date or a day inside a sentence, but not a different one", () => {
    const date = BENCH_QUESTIONS.find((q) => q.id === "dates-2")!;
    expect(answerMatches(date.expected, "April 2")).toBe(true);
    expect(answerMatches(date.expected, "It is April 2.")).toBe(true);
    expect(answerMatches(date.expected, "April 3")).toBe(false);
    const day = BENCH_QUESTIONS.find((q) => q.id === "dates-1")!;
    expect(answerMatches(day.expected, "Thursday")).toBe(true);
    expect(answerMatches(day.expected, "The day is Thursday.")).toBe(true);
    expect(answerMatches(day.expected, "Friday")).toBe(false);
  });

  it("reads the first yes or no and a standalone letter choice", () => {
    const yes = BENCH_QUESTIONS.find((q) => q.id === "logic-1")!;
    expect(answerMatches(yes.expected, "Yes")).toBe(true);
    expect(answerMatches(yes.expected, "Yes, all roses are flowers.")).toBe(true);
    expect(answerMatches(yes.expected, "No")).toBe(false);
    expect(answerMatches(yes.expected, "no yes")).toBe(false);
    const letter = BENCH_QUESTIONS.find((q) => q.id === "logic-2")!;
    expect(answerMatches(letter.expected, "C")).toBe(true);
    expect(answerMatches(letter.expected, "Answer: C")).toBe(true);
    expect(answerMatches(letter.expected, "Because C is tallest")).toBe(true);
    expect(answerMatches(letter.expected, "B")).toBe(false);
    expect(answerMatches("yes", "Yesterday it was fine")).toBe(false);
  });

  it("keeps a short, plain answer and grades it", () => {
    const q = BENCH_QUESTIONS[0]!;
    const result = gradeAnswer(q, "  The   answer\n is 43.  ");
    expect(result).toMatchObject({ id: q.id, category: q.category, question: q.question, expected: q.expected, answer: "The answer is 43.", correct: true });
    expect(gradeAnswer(q, "x".repeat(300)).answer).toHaveLength(120);
    expect(gradeAnswer(q, "41").correct).toBe(false);
  });

  it("grades a whole set and counts an unanswered question as wrong", () => {
    const all = Object.fromEntries(BENCH_QUESTIONS.map((q) => [q.id, q.expected]));
    expect(gradeAnswers(all)).toMatchObject({ asked: 20, correct: 20 });
    expect(gradeAnswers({})).toMatchObject({ asked: 20, correct: 0 });
    const partial = gradeAnswers({ ...all, "arith-1": "" });
    expect(partial.correct).toBe(19);
    expect(partial.questions.find((q) => q.id === "arith-1")).toMatchObject({ answer: "", correct: false });
  });
});

describe("depths and comparisons", () => {
  it("labels the three depths in plain English", () => {
    expect(BENCH_CONTEXTS).toEqual([0, 4096, 16384]);
    expect(contextLabel(0)).toBe("Short prompt");
    expect(contextLabel(4096)).toBe("4K");
    expect(contextLabel(16384)).toBe("16K");
    expect(contextLabel(2048)).toBe((2048).toLocaleString());
    expect(contextDetail(0)).toBe("A short prompt (no long context)");
    expect(contextDetail(16384)).toBe(`${(16384).toLocaleString()} tokens of context`);
  });

  it("keeps a point that could not be measured, with its note", () => {
    expect(notRunPoint(16384, "The model holds fewer tokens.")).toMatchObject({ context: 16384, ran: false, note: "The model holds fewer tokens.", decodeTps: 0, seconds: 0 });
  });

  it("computes a percentage change only when the base is positive", () => {
    expect(percentChange(120, 100)).toBe(20);
    expect(percentChange(80, 100)).toBe(-20);
    expect(percentChange(10, 0)).toBeNull();
    expect(percentChange(Number.NaN, 10)).toBeNull();
  });

  it("compares two runs at all three depths and leaves an unmeasured depth empty", () => {
    const a = run("a", [point(0, { prefillTps: 100, decodeTps: 10, ttftMs: 1000 }), point(4096), notRunPoint(16384, "Too long.")]);
    const b = run("b", [point(0, { prefillTps: 120, decodeTps: 8, ttftMs: 800 }), point(4096, { prefillTps: 50 }), point(16384, { decodeTps: 4 })]);
    const rows = compareRuns(a, b);
    expect(rows.map((r) => r.context)).toEqual([0, 4096, 16384]);
    expect(rows[0]).toMatchObject({ prefillPct: 20, decodePct: -20, ttftPct: -20 });
    expect(rows[1]).toMatchObject({ prefillPct: -50, decodePct: 0, ttftPct: 0 });
    expect(rows[2]).toMatchObject({ a: expect.objectContaining({ ran: false }), b: expect.objectContaining({ ran: true }), prefillPct: null, decodePct: null, ttftPct: null });
  });
});

describe("history helpers", () => {
  it("orders chart points oldest first and skips unmeasured or zero values", () => {
    const old = run("old", [point(4096, { decodeTps: 12 })], "2026-10-19T10:00:00.000Z");
    const middle = run("middle", [notRunPoint(4096, "Too long.")], "2026-10-20T10:00:00.000Z");
    const recent = run("recent", [point(4096, { decodeTps: 18 })], "2026-10-21T10:00:00.000Z");
    const points = chartPoints([recent, middle, old], 4096, "decodeTps");
    expect(points.map((p) => p.runId)).toEqual(["old", "recent"]);
    expect(points.map((p) => p.value)).toEqual([12, 18]);
    expect(points[0]!.provenance).toBe("measured");
    expect(points[0]!.label).not.toBe("");
    expect(chartPoints([recent], 16384, "decodeTps")).toEqual([]);
  });

  it("writes the summary from the accuracy and the fastest measured depth", () => {
    const ok = run("ok", [point(0, { decodeTps: 20 }), point(4096, { decodeTps: 8 }), notRunPoint(16384, "Too long.")]);
    expect(runSummary(ok)).toBe("17/20 correct · 20.0 tok/s writing at short prompt");
    const none = run("none", [notRunPoint(0, "No model.")]);
    expect(runSummary(none)).toBe("17/20 correct");
  });

  it("describes the machine and status plainly", () => {
    expect(statusLabel("done")).toBe("Finished");
    expect(statusLabel("stopped")).toBe("Stopped");
    expect(statusLabel("failed")).toBe("Failed");
    expect(machineLine(run("x", []))).toBe("Example GPU · Example CPU · 32.0 GiB RAM · test");
    const noGpu = run("x", []);
    noGpu.machine.gpu = "none";
    expect(machineLine(noGpu)).toContain("Processor only");
    expect(runTitle(run("x", []))).toContain(BASE.name);
  });

  it("says where the model ran, in one line", () => {
    const r = run("x", []);
    expect(placementLine(r)).toBe("20 of 40 layers on the graphics card, the rest on the processor · model file on an NVMe SSD");
    r.placement.gpuLayers = 40;
    expect(placementLine(r)).toContain("All 40 layers on the graphics card");
    r.placement.gpuLayers = 0;
    r.placement.drive = "hard-disk";
    expect(placementLine(r)).toBe("Processor only · model file on a hard disk");
  });
});

describe("the placement a run measures", () => {
  it("keeps the placement the user chose in the run", async () => {
    const b = createMockBench({ models: [BASE], hardware: () => MOCK_HARDWARE, speed: 100 });
    const all = await b.run(BASE.id, "gpu", () => {}, new AbortController().signal);
    expect(all.placement).toMatchObject({ choice: "gpu", gpuLayers: BASE.arch.layers, totalLayers: BASE.arch.layers });
    expect(all.placement.layout).toBe("gpu-int8");

    const half = await b.run(BASE.id, "split", () => {}, new AbortController().signal);
    expect(half.placement).toMatchObject({ choice: "split", gpuLayers: Math.floor(BASE.arch.layers / 2) });
    expect(half.placement.layout).toBe("split");

    const cpu = await b.run(BASE.id, "cpu", () => {}, new AbortController().signal);
    expect(cpu.placement).toMatchObject({ choice: "cpu", gpuLayers: 0, layout: "cpu" });
  });
});

describe("report cards", () => {
  it("escapes every string that comes from a model", () => {
    expect(escapeHtml(`<script>& "'`)).toBe("&lt;script&gt;&amp; &quot;&#39;");
    const hostile = run("hostile", [point(0), point(4096), notRunPoint(16384, '<img src=x onerror="alert(1)">')]);
    hostile.model.name = `<script>alert(1)</script> & "model"`;
    hostile.accuracy.questions[0]!.answer = "<b>not markup</b>";
    const html = reportHtml(hostile);
    expect(html).toContain("&lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;model&quot;");
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;b&gt;not markup&lt;/b&gt;");
    expect(html).toContain(`${(4032).toLocaleString()} tokens`);
    expect(html).toContain("17/20 correct");
  });

  it("marks illustrative data as not a measurement", () => {
    const illustrative = run("i", [point(0)]);
    illustrative.provenance = "illustrative";
    const html = reportHtml(illustrative);
    expect(html).toContain("Illustrative data — not a measurement.");
    expect(html).toContain("not a measurement");
    expect(html).toContain("Example data from the web preview.");
    expect(html).not.toContain("Measured on this computer");
    expect(html).not.toContain("<script");
  });

  it("builds a safe file name", () => {
    const r = run("x", [point(0)]);
    r.model.name = 'My / model: <bad> "name"';
    expect(reportFilename(r, "html")).toMatch(/^cerebro-benchmark-My-model-bad-name-2026-10-20\.html$/);
    expect(reportFilename(r, "png")).toMatch(/\.png$/);
  });
});

describe("the web preview's benchmark", () => {
  const at = new Date(2026, 9, 21, 12, 0, 0).getTime();
  const bench = (models: readonly LibraryModel[] = [BASE, SMALL_CTX]) =>
    createMockBench({ models, hardware: () => MOCK_HARDWARE, speed: 100, now: () => at });

  it("seeds two illustrative runs so history works before the first run", async () => {
    const history = await bench().history();
    expect(history).toHaveLength(2);
    expect(history.every((r) => r.provenance === "illustrative")).toBe(true);
    expect(history[0]!.id).not.toBe(history[1]!.id);
  });

  it("runs three depths and twenty questions, then saves the run", async () => {
    const b = bench();
    const events: BenchEvent[] = [];
    const finished = await b.run(BASE.id, "auto", (e) => events.push(e), new AbortController().signal);
    expect(finished).toMatchObject({ status: "done", provenance: "illustrative" });
    expect(finished.points.map((p) => p.context)).toEqual([0, 4096, 16384]);
    expect(finished.points.every((p) => p.ran)).toBe(true);
    expect(finished.accuracy.asked).toBe(20);
    expect(finished.accuracy.correct).toBeGreaterThanOrEqual(0);
    expect(finished.accuracy.correct).toBeLessThanOrEqual(20);
    expect(eventTypes(events)).toEqual(["progress", "progress", "point", "progress", "point", "progress", "point", ...Array(20).fill("progress"), "accuracy", "done"]);
    const history = await b.history();
    expect(history).toHaveLength(3);
    expect(history[0]!.id).toBe(finished.id);
  });

  it("marks a depth the model cannot hold as not run, with a note", async () => {
    const finished = await bench().run(SMALL_CTX.id, "auto", () => {}, new AbortController().signal);
    const deep = finished.points.find((p) => p.context === 16384)!;
    expect(deep.ran).toBe(false);
    expect(deep.note).toContain("cannot read");
    expect(finished.points.find((p) => p.context === 4096)!.ran).toBe(true);
  });

  it("keeps a stopped run, labelled stopped", async () => {
    const b = bench();
    const controller = new AbortController();
    const finished = await b.run(BASE.id, "auto", (e) => {
      if (e.type === "progress" && e.phase === "measure") controller.abort();
    }, controller.signal);
    expect(finished.status).toBe("stopped");
    expect(finished.stopReason).toMatch(/Stopped/);
    const history = await b.history();
    expect(history[0]!.id).toBe(finished.id);
  });
});

describe("the service's benchmark endpoints", () => {
  it("runs through the mock service and returns the saved history", async () => {
    const service = createMockService({ benchSpeed: 100 });
    const before = await service.benchHistory();
    expect(before).toHaveLength(2);
    const finished = await service.benchRun(before[0]!.model.id, "auto", () => {}, new AbortController().signal);
    expect(finished.status).toBe("done");
    const after = await service.benchHistory();
    expect(after).toHaveLength(3);
    expect(after[0]!.id).toBe(finished.id);
    expect(after[0]!.provenance).toBe("illustrative");
  });
});
