/** Train mode (src-tauri/src/trainer.rs): fine-tune a model with a LoRA adapter, export it to GGUF. */

export type TrainFormat = "auto" | "alpaca" | "chatml" | "sharegpt";
export type TrainMethod = "qlora" | "lora";
export type TrainQuant = "q4_k_m" | "q5_k_m" | "q8_0" | "f16";
export const TRAIN_QUANTS: readonly TrainQuant[] = ["q4_k_m", "q5_k_m", "q8_0", "f16"];

export interface TrainJob {
  name: string;
  /** Hugging Face model id, e.g. unsloth/Qwen3-0.6B. */
  model: string;
  /** Exactly one of these two. */
  datasetPath: string | null;
  datasetHf: string | null;
  format: TrainFormat;
  method: TrainMethod;
  rank: number;
  alpha: number;
  lr: number;
  epochs: number;
  /** 0 = run the epochs. */
  maxSteps: number;
  maxSeqLen: number;
  batch: number;
  gradAccum: number;
  export: TrainQuant[];
}

export interface TrainInstalled {
  uv: string;
  python: string;
  torch: string;
  unsloth: string;
  cuda: boolean;
}

export interface TrainerStatus {
  installed: TrainInstalled | null;
  settingUp: boolean;
  running: boolean;
  root: string | null;
  /** The last lines the training process wrote to stderr (for "Details"). */
  stderrTail: string[];
}

export type TrainStage = "loading_model" | "preparing_data" | "training" | "saving" | "exporting";

export type TrainEvent =
  | { type: "setup"; text: string }
  | { type: "status"; stage: TrainStage | string; detail: string | null }
  | { type: "log"; step: number; maxSteps: number; loss: number | null; evalLoss: number | null; lr: number | null; gradNorm: number | null; epoch: number | null }
  | { type: "export"; quant: string; path: string }
  | { type: "done"; models: string[]; adapter: string | null; steps: number; seconds: number; stopped: boolean }
  | { type: "error"; message: string };
