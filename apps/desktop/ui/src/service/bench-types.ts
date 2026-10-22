/**
 * Benchmark lab data. Mirrors the Rust types in src-tauri/src/bench.rs. Every number in a
 * "measured" run came from the engine on this machine; the web preview marks its runs
 * "illustrative" and never presents them as measurements.
 */

/** The prompt depths a run measures, in tokens. 0 is a short prompt, not an empty one. */
export const BENCH_CONTEXTS = [0, 4096, 16384] as const;
export type BenchContext = (typeof BENCH_CONTEXTS)[number];

export type BenchCategory = "arithmetic" | "units" | "dates" | "letters" | "logic";
export type BenchProvenance = "measured" | "illustrative";
export type BenchStatus = "done" | "stopped" | "failed";

/**
 * Where the model runs while it is measured. "auto" keeps the placement the model is loaded with
 * (what the user chose on the Models screen); the others force it.
 */
export type BenchPlacementChoice = "auto" | "gpu" | "split" | "cpu";

export const BENCH_PLACEMENTS: readonly BenchPlacementChoice[] = ["auto", "gpu", "split", "cpu"];

export const BENCH_PLACEMENT_LABELS: Readonly<Record<BenchPlacementChoice, string>> = {
  auto: "As it is loaded",
  gpu: "Graphics card",
  split: "Graphics card and processor",
  cpu: "Processor only",
};

/** The placement a run records: the choice, and where the weights ended up. */
export interface BenchPlacement {
  /** What the user chose. */
  choice: BenchPlacementChoice;
  /** Where the weights went: "gpu-int8", "gpu-native", "split" or "cpu". */
  layout: string;
  /** Layers on the graphics card, and how many the model has. */
  gpuLayers: number;
  totalLayers: number;
  /** The drive that serves the part of the model that does not fit in memory. */
  drive: string;
  /** The engine's own words about this placement. */
  notes: string[];
}

export interface BenchModelInfo {
  id: string;
  name: string;
  quant: string;
  params: string;
  sizeGiB: number;
  contextMax: number;
}

export interface BenchMachineInfo {
  cpu: string;
  gpu: string;
  ramTotalGiB: number;
  vramTotalGiB: number;
  build: string;
}

/** One measured depth. A depth the model cannot hold is kept, honestly, as not run. */
export interface BenchPoint {
  /** The depth this point asked for: 0, 4096 or 16384. */
  context: number;
  ran: boolean;
  /** Why the point did not run, or how the prompt was adjusted. Null when it ran as asked. */
  note: string | null;
  promptTokens: number;
  outputTokens: number;
  /** Prompt reading speed, tokens per second. 0 when the point did not run. */
  prefillTps: number;
  /** Writing speed after the first token, tokens per second. 0 when it did not run. */
  decodeTps: number;
  /** Time to the first written token, milliseconds. 0 when the point did not run. */
  ttftMs: number;
  /** System memory in use while measuring, GiB. 0 when the point did not run. */
  memoryUsedGiB: number;
  /** Wall time for this point, seconds. */
  seconds: number;
}

export interface BenchQuestionResult {
  id: string;
  category: BenchCategory;
  question: string;
  expected: string;
  /** What the model wrote, cut to a short answer for the report. */
  answer: string;
  correct: boolean;
}

export interface BenchAccuracy {
  asked: number;
  correct: number;
  questions: BenchQuestionResult[];
}

export interface BenchRun {
  id: string;
  /** ISO 8601 date and time, UTC. */
  at: string;
  provenance: BenchProvenance;
  status: BenchStatus;
  model: BenchModelInfo;
  machine: BenchMachineInfo;
  /** Where the model ran while it was measured. */
  placement: BenchPlacement;
  points: BenchPoint[];
  accuracy: BenchAccuracy;
  /** Whole run's wall time, seconds. */
  seconds: number;
  /** Why a stopped run stopped, or a failed run failed. Null for a finished run. */
  stopReason: string | null;
}

export type BenchPhase = "load" | "measure" | "accuracy";

export type BenchEvent =
  | { type: "progress"; phase: BenchPhase; text: string; context: number | null; done: number; total: number }
  | { type: "point"; point: BenchPoint }
  | { type: "accuracy"; result: BenchAccuracy }
  | { type: "done"; run: BenchRun }
  | { type: "error"; message: string };
