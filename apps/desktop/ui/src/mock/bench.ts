/**
 * The web preview's benchmark lab. It runs fast and is labelled illustrative everywhere it appears:
 * its numbers are shaped like a real run so the screens, chart, comparison and report card can be
 * tried without a model, but they are not measurements.
 */
import type { HardwareProfile, LibraryModel } from "../service/types";
import type { BenchAccuracy, BenchEvent, BenchPlacement, BenchPlacementChoice, BenchPoint, BenchQuestionResult, BenchRun } from "../service/bench-types";
import { BENCH_QUESTIONS, gradeAnswer } from "../lib/bench";

export interface MockBenchOptions {
  models: readonly LibraryModel[];
  hardware: () => HardwareProfile;
  /** 1 is a few seconds per run; 100 is instant enough for a test. */
  speed?: number;
  now?: () => number;
}

export interface MockBench {
  run(modelId: string, placement: BenchPlacementChoice, onEvent: (e: BenchEvent) => void, signal: AbortSignal): Promise<BenchRun>;
  history(): Promise<BenchRun[]>;
  stop(): void;
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function modelInfo(model: LibraryModel) {
  return { id: model.id, name: model.name, quant: model.quant, params: model.params, sizeGiB: model.sizeGiB, contextMax: model.contextMax };
}

function machineInfo(hw: HardwareProfile) {
  return {
    cpu: hw.cpu.name,
    gpu: hw.gpu?.name ?? "none",
    ramTotalGiB: hw.ramTotalGiB,
    vramTotalGiB: hw.gpu?.vramTotalGiB ?? 0,
    build: "Cerebro Engine web preview",
  };
}

function mockPoint(context: number, seed: number): BenchPoint {
  const promptTokens = context === 0 ? 26 : context - 64;
  const outputTokens = 48;
  const readBase = 900 + (seed % 2500);
  const writeBase = 9 + (seed % 38);
  const depth = context / 16384;
  const prefillTps = readBase * (context === 0 ? 1 : 0.84 - depth * 0.28);
  const decodeTps = writeBase * (context === 0 ? 1 : 0.93 - depth * 0.2);
  const ttftMs = 45 + (promptTokens / Math.max(prefillTps, 1)) * 1000;
  const memoryUsedGiB = 4.2 + (seed % 160) / 10 + depth * 2.7;
  const seconds = promptTokens / Math.max(prefillTps, 1) + outputTokens / Math.max(decodeTps, 1);
  return {
    context,
    ran: true,
    note: null,
    promptTokens,
    outputTokens,
    prefillTps: Math.round(prefillTps * 10) / 10,
    decodeTps: Math.round(decodeTps * 10) / 10,
    ttftMs: Math.round(ttftMs),
    memoryUsedGiB: Math.round(memoryUsedGiB * 100) / 100,
    seconds: Math.round(seconds * 100) / 100,
  };
}

function mockAnswers(model: LibraryModel): BenchAccuracy {
  const seed = hash(model.id);
  const questions: BenchQuestionResult[] = BENCH_QUESTIONS.map((question, i) => {
    const correct = (seed + i * 37) % 100 < 84;
    let answer = question.expected;
    if (!correct) {
      if (/^\d+$/.test(question.expected)) answer = String(Number(question.expected) + (i % 3 === 0 ? 1 : -2));
      else if (question.expected === "yes") answer = "no";
      else if (question.expected === "no") answer = "yes";
      else if (question.expected === "C") answer = "B";
      else if (question.category === "dates") answer = i % 2 ? "March 31" : "Wednesday";
      else answer = "I am not sure.";
    }
    return gradeAnswer(question, answer);
  });
  return { asked: questions.length, correct: questions.filter((q) => q.correct).length, questions };
}

/** The placement the web preview pretends to use. Illustrative, like every number here. */
function mockPlacement(choice: BenchPlacementChoice, model: LibraryModel): BenchPlacement {
  const layers = model.arch.layers;
  const gpuLayers = choice === "gpu" ? layers : choice === "split" ? Math.floor(layers / 2) : choice === "cpu" ? 0 : layers;
  const notes =
    gpuLayers === 0
      ? ["No layers on the graphics card: everything runs on the processor."]
      : gpuLayers === layers
        ? ["Every layer on the graphics card."]
        : [`${gpuLayers} of ${layers} layers on the graphics card, the rest on the processor.`];
  return {
    choice,
    layout: gpuLayers === 0 ? "cpu" : gpuLayers === layers ? "gpu-int8" : "split",
    gpuLayers,
    totalLayers: layers,
    drive: "nvme",
    notes,
  };
}

function buildRun(model: LibraryModel, hw: HardwareProfile, at: number, choice: BenchPlacementChoice = "auto"): BenchRun {
  const seed = hash(model.id);
  const points: BenchPoint[] = [mockPoint(0, seed)];
  for (const context of [4096, 16384]) {
    if (model.contextMax >= context) points.push(mockPoint(context, seed));
    else
      points.push({
        context,
        ran: false,
        note: `This model holds ${model.contextMax.toLocaleString()} tokens, so it cannot read a ${context.toLocaleString()}-token prompt.`,
        promptTokens: 0,
        outputTokens: 0,
        prefillTps: 0,
        decodeTps: 0,
        ttftMs: 0,
        memoryUsedGiB: 0,
        seconds: 0,
      });
  }
  const accuracy = mockAnswers(model);
  return {
    id: `bench-${at}-${model.id}`,
    at: new Date(at).toISOString(),
    provenance: "illustrative",
    status: "done",
    model: modelInfo(model),
    machine: machineInfo(hw),
    placement: mockPlacement(choice, model),
    points,
    accuracy,
    seconds: points.reduce((s, p) => s + p.seconds, 0) + accuracy.questions.length * 0.12,
    stopReason: null,
  };
}

export function createMockBench(options: MockBenchOptions): MockBench {
  const now = options.now ?? Date.now;
  const speed = Math.max(1, options.speed ?? 8);
  const textModels = options.models.filter((m) => m.kind === "text" || m.kind === "vision");
  const history: BenchRun[] = [];
  // Two finished runs from earlier days, so the chart and comparison have something to show before
  // the first run of this visit.
  const first = textModels[0];
  const second = textModels[1] ?? textModels[0];
  if (first) history.push(buildRun(first, options.hardware(), now() - 2 * 86_400_000));
  if (second) history.push(buildRun(second, options.hardware(), now() - 86_400_000));
  let cancelled = false;
  let running = false;

  const run = async (modelId: string, placement: BenchPlacementChoice, onEvent: (e: BenchEvent) => void, signal: AbortSignal): Promise<BenchRun> => {
    if (running) throw new Error("A benchmark is already running.");
    const model = textModels.find((m) => m.id === modelId);
    if (!model) throw new Error("That model is not in the library.");
    const hw = options.hardware();
    const seed = hash(model.id);
    const at = now();
    running = true;
    cancelled = signal.aborted;
    const onAbort = () => {
      cancelled = true;
    };
    signal.addEventListener("abort", onAbort, { once: true });

    const contexts = [0, 4096, 16384].filter((c) => c === 0 || model.contextMax >= c);
    const total = contexts.length + BENCH_QUESTIONS.length;
    let done = 0;
    const points: BenchPoint[] = [];
    const questions: BenchQuestionResult[] = [];
    const pause = () => wait(Math.max(10, 420 / speed));
    const cancelledNow = () => cancelled || signal.aborted;

    try {
      onEvent({ type: "progress", phase: "load", text: `Loading ${model.name} in the web preview…`, context: null, done: 0, total });
      await pause();
      if (cancelledNow()) throw new Error("stopped");

      for (const context of [0, 4096, 16384]) {
        if (cancelledNow()) throw new Error("stopped");
        if (context !== 0 && model.contextMax < context) {
          points.push({
            context,
            ran: false,
            note: `This model holds ${model.contextMax.toLocaleString()} tokens, so it cannot read a ${context.toLocaleString()}-token prompt.`,
            promptTokens: 0,
            outputTokens: 0,
            prefillTps: 0,
            decodeTps: 0,
            ttftMs: 0,
            memoryUsedGiB: 0,
            seconds: 0,
          });
          continue;
        }
        onEvent({
          type: "progress",
          phase: "measure",
          text: context === 0 ? "Measuring a short prompt…" : `Reading ${(context - 64).toLocaleString()} tokens at ${context === 4096 ? "4K" : "16K"}…`,
          context,
          done,
          total,
        });
        await pause();
        const point = mockPoint(context, seed);
        points.push(point);
        done += 1;
        onEvent({ type: "point", point });
      }

      for (let i = 0; i < BENCH_QUESTIONS.length; i++) {
        if (cancelledNow()) throw new Error("stopped");
        onEvent({ type: "progress", phase: "accuracy", text: `Question ${i + 1} of ${BENCH_QUESTIONS.length}…`, context: null, done, total });
        await wait(Math.max(4, 80 / speed));
        const seedAnswer = mockAnswers(model).questions[i]!;
        questions.push({ ...seedAnswer, answer: seedAnswer.answer });
        done += 1;
      }
      const accuracy: BenchAccuracy = { asked: questions.length, correct: questions.filter((q) => q.correct).length, questions };
      onEvent({ type: "accuracy", result: accuracy });
      const finished: BenchRun = {
        id: `bench-${at}-${model.id}`,
        at: new Date(at).toISOString(),
        provenance: "illustrative",
        status: "done",
        model: modelInfo(model),
        machine: machineInfo(hw),
        placement: mockPlacement(placement, model),
        points,
        accuracy,
        seconds: points.reduce((s, p) => s + p.seconds, 0) + questions.length * 0.12,
        stopReason: null,
      };
      history.unshift(finished);
      onEvent({ type: "done", run: finished });
      return finished;
    } catch (error) {
      const stopped = cancelledNow();
      const accuracy: BenchAccuracy = { asked: questions.length, correct: questions.filter((q) => q.correct).length, questions };
      const finished: BenchRun = {
        id: `bench-${at}-${model.id}`,
        at: new Date(at).toISOString(),
        provenance: "illustrative",
        status: stopped ? "stopped" : "failed",
        model: modelInfo(model),
        machine: machineInfo(hw),
        placement: mockPlacement(placement, model),
        points,
        accuracy,
        seconds: points.reduce((s, p) => s + p.seconds, 0),
        stopReason: stopped ? "Stopped before the run finished." : error instanceof Error ? error.message : "The run failed.",
      };
      if (stopped) history.unshift(finished);
      onEvent(stopped ? { type: "done", run: finished } : { type: "error", message: finished.stopReason ?? "The run failed." });
      if (!stopped) throw error;
      return finished;
    } finally {
      running = false;
      signal.removeEventListener("abort", onAbort);
    }
  };

  return {
    run,
    history: () => Promise.resolve(history.map((r) => structuredClone(r))),
    stop: () => {
      cancelled = true;
    },
  };
}
