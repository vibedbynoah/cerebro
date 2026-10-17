import { TRAIN_QUANTS, type TrainEvent, type TrainJob } from "../service/train-types";

/**
 * Train screen logic: the default job, validation (the same checks as trainer.rs TrainJob::check,
 * so the form says what is wrong before anything starts), and the run view folded from events.
 */

/** Small models that train on an 8 GB GPU with QLoRA; the field also takes any Hugging Face id. */
export const BASE_MODELS: readonly { id: string; note: string }[] = [
  { id: "unsloth/Qwen3-0.6B", note: "0.6B, trains in minutes" },
  { id: "unsloth/Qwen3-4B", note: "4B, the best quality that fits 8 GB" },
  { id: "unsloth/gemma-3-4b-it", note: "4B instruction model" },
  { id: "unsloth/Llama-3.2-3B-Instruct", note: "3B instruction model" },
];

export const DEFAULT_JOB: TrainJob = {
  name: "",
  model: "unsloth/Qwen3-0.6B",
  datasetPath: null,
  datasetHf: null,
  format: "auto",
  method: "qlora",
  rank: 16,
  alpha: 16,
  lr: 2e-4,
  epochs: 1,
  maxSteps: 60,
  maxSeqLen: 2048,
  batch: 2,
  gradAccum: 4,
  export: ["q4_k_m"],
};

const HF_ID = /^[A-Za-z0-9._-]{1,96}\/[A-Za-z0-9._-]{1,96}$/;
const isHfId = (s: string): boolean => HF_ID.test(s) && !s.split("/").some((p) => p === "." || p === "..");
const NAME = /^[\p{L}\p{N} ._-]{1,64}$/u;
const DATASET_EXT = /\.(jsonl|json|csv|parquet)$/i;
const intIn = (v: number, lo: number, hi: number) => Number.isSafeInteger(v) && v >= lo && v <= hi;

export type JobField = "name" | "model" | "dataset" | "rank" | "alpha" | "lr" | "epochs" | "maxSteps" | "maxSeqLen" | "batch" | "gradAccum" | "export";

/** Every problem with the job, by field; empty when it can start. */
export function jobErrors(j: TrainJob): Partial<Record<JobField, string>> {
  const e: Partial<Record<JobField, string>> = {};
  if (!j.name.trim() || !NAME.test(j.name)) e.name = "Letters, digits, spaces, - _ or . (up to 64).";
  if (!isHfId(j.model)) e.model = "A Hugging Face id like unsloth/Qwen3-0.6B.";
  const path = j.datasetPath?.trim() ?? "";
  const hf = j.datasetHf?.trim() ?? "";
  if ((path ? 1 : 0) + (hf ? 1 : 0) !== 1) e.dataset = "Pick one dataset: a file on this computer or a Hugging Face dataset id.";
  else if (path && !DATASET_EXT.test(path)) e.dataset = "A .jsonl, .json, .csv or .parquet file.";
  else if (hf && !isHfId(hf)) e.dataset = "A Hugging Face dataset id like tatsu-lab/alpaca.";
  if (!intIn(j.rank, 1, 256)) e.rank = "1 to 256.";
  if (!intIn(j.alpha, 1, 512)) e.alpha = "1 to 512.";
  if (!(j.lr > 0 && j.lr <= 0.01)) e.lr = "Above 0, at most 0.01.";
  if (!(j.epochs > 0 && j.epochs <= 50)) e.epochs = "Above 0, at most 50.";
  if (!intIn(j.maxSteps, 0, 100_000)) e.maxSteps = "0 (use the epochs) to 100,000.";
  if (!intIn(j.maxSeqLen, 128, 32_768)) e.maxSeqLen = "128 to 32,768 tokens.";
  if (!intIn(j.batch, 1, 64)) e.batch = "1 to 64.";
  if (!intIn(j.gradAccum, 1, 256)) e.gradAccum = "1 to 256.";
  if (j.export.length > TRAIN_QUANTS.length || !j.export.every((q) => TRAIN_QUANTS.includes(q))) e.export = "Unknown format.";
  return e;
}

/** The job as sent: trimmed, with the unused dataset kind null. */
export function normalizeJob(j: TrainJob): TrainJob {
  const path = j.datasetPath?.trim() || null;
  const hf = j.datasetHf?.trim() || null;
  return { ...j, name: j.name.trim(), model: j.model.trim(), datasetPath: path, datasetHf: path ? null : hf, export: TRAIN_QUANTS.filter((q) => j.export.includes(q)) };
}

export const STAGE_LABELS: Readonly<Record<string, string>> = {
  loading_model: "Loading the model",
  preparing_data: "Reading the dataset",
  training: "Training",
  saving: "Saving the adapter",
  exporting: "Exporting GGUF",
};

export interface TrainRun {
  phase: "idle" | "running" | "done" | "failed";
  stage: string | null;
  detail: string | null;
  step: number;
  maxSteps: number;
  /** One loss per logged step (null where a step logged none). */
  losses: Array<number | null>;
  lr: number | null;
  exports: Array<{ quant: string; path: string }>;
  models: string[];
  seconds: number | null;
  stopped: boolean;
  error: string | null;
}

export const IDLE_RUN: TrainRun = { phase: "idle", stage: null, detail: null, step: 0, maxSteps: 0, losses: [], lr: null, exports: [], models: [], seconds: null, stopped: false, error: null };

/** Folds one event into the run view. Setup lines are not part of a run. */
export function applyTrainEvent(r: TrainRun, e: TrainEvent): TrainRun {
  switch (e.type) {
    case "status":
      return { ...r, phase: "running", stage: e.stage, detail: e.detail };
    case "log": {
      const losses = r.losses.slice();
      if (e.step >= 1 && e.step <= 100_000) {
        while (losses.length < e.step - 1) losses.push(null);
        losses[e.step - 1] = e.loss;
      }
      return { ...r, phase: "running", step: e.step, maxSteps: e.maxSteps, losses, lr: e.lr ?? r.lr };
    }
    case "export":
      return { ...r, exports: [...r.exports, { quant: e.quant, path: e.path }] };
    case "done":
      return { ...r, phase: "done", stage: null, models: e.models, seconds: e.seconds, stopped: e.stopped, step: e.steps || r.step };
    case "error":
      return { ...r, phase: "failed", error: e.message };
    case "setup":
      return r;
  }
}

/** The latest finite loss, for the headline figure. */
export function lastLoss(r: TrainRun): number | null {
  for (let i = r.losses.length - 1; i >= 0; i--) {
    const v = r.losses[i];
    if (v !== null && v !== undefined && Number.isFinite(v)) return v;
  }
  return null;
}

export const fileName = (p: string): string => p.split(/[\\/]/).pop() ?? p;
