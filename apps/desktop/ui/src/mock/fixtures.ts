import type { EngineSwitch, GpuInfo, HardwareProfile, LiveStats, ModelSummary, Settings } from "../service/types";

/**
 * ILLUSTRATIVE DATA, NOT MEASUREMENTS. Every number below is a placeholder for layout and testing.
 * Model names are real published models so the screens read naturally; sizes, speeds and VRAM
 * figures are invented and carry `provenance: "illustrative"`, which every screen shows as a flag
 * (docs/CLAIMS.md: never present an invented number as measured).
 */

export const MOCK_GPU: GpuInfo = { name: "Example laptop GPU", vramTotalGiB: 8 };

export const MOCK_MODELS: readonly ModelSummary[] = [
  {
    id: "qwen3-coder-30b-a3b-q4km",
    name: "Qwen3-Coder-30B-A3B-Instruct",
    quant: "Q4_K_M",
    params: "30B-A3B",
    architecture: "moe",
    sizeGiB: 17.3,
    contextMax: 32768,
    tune: "tuned",
    defaultsTps: 4.2,
    tunedTps: 16.8,
    provenance: "illustrative",
  },
  {
    id: "gemma-4-e4b-q4km",
    name: "Gemma-4-E4B-it",
    quant: "Q4_K_M",
    params: "E4B",
    architecture: "dense",
    sizeGiB: 5.0,
    contextMax: 32768,
    tune: "defaults-fastest",
    defaultsTps: 52.0,
    tunedTps: 52.0,
    provenance: "illustrative",
  },
  {
    id: "mistral-small-24b-q4km",
    name: "Mistral-Small-3.2-24B-Instruct",
    quant: "Q4_K_M",
    params: "24B",
    architecture: "dense",
    sizeGiB: 13.3,
    contextMax: 16384,
    tune: "untuned",
    defaultsTps: null,
    tunedTps: null,
    provenance: "illustrative",
  },
  {
    id: "qwen3-8b-q5km",
    name: "Qwen3-8B",
    quant: "Q5_K_M",
    params: "8B",
    architecture: "dense",
    sizeGiB: 5.4,
    contextMax: 32768,
    tune: "untuned",
    defaultsTps: null,
    tunedTps: null,
    provenance: "illustrative",
  },
];

export const MOCK_LIVE_BASE: LiveStats = {
  provenance: "illustrative",
  model: "Qwen3-Coder-30B-A3B-Instruct",
  tokensPerSecond: 16.8,
  vramUsedGiB: 7.1,
  vramTotalGiB: MOCK_GPU.vramTotalGiB,
  backend: { state: "running", address: "127.0.0.1:8181", build: "b6500" },
};

/** Commit of the example backend build. Illustrative, like the build number beside it. */
export const MOCK_BACKEND_COMMIT = "3f1e9c2";

/** First-run hardware scan. A generic example machine, not anyone's real computer. */
export const MOCK_HARDWARE: HardwareProfile = {
  provenance: "illustrative",
  cpu: { name: "Example 8-core laptop CPU", cores: 8, threads: 16 },
  gpu: MOCK_GPU,
  ramTotalGiB: 32,
};

/** Models setup and Model Hub list (mock/catalog.ts): current releases only, every quality and
 *  speed figure illustrative. */
export { MOCK_CATALOG } from "./catalog";

/** Safe by default: every protection on, every network feature allowed but optional. */
export const DEFAULT_SETTINGS: Settings = {
  processContainment: true,
  cleanEnvironment: true,
  integrityCheck: true,
  permissionPrompts: true,
  blockLocalFetch: true,
  hideSecretsFromTools: true,
  readBeforeWrite: true,
  loopbackOnly: true,
  offlineMode: false,
  allowModelDownloads: true,
  allowPluginInstalls: true,
};

/** Mirrors part of the engine's switch registry (crates/cerebro-engine/src/switches.rs). */
export const MOCK_SWITCHES: readonly EngineSwitch[] = [
  {
    id: "gpu_mma_matmul", name: "Tensor-core prompt reading", group: "GPU", kind: { type: "toggle" }, channel: "stable", applies: "reload",
    beginner: "Reads your prompt up to 12x faster on recent NVIDIA graphics cards.",
    advanced: "Prefill matmuls run on int8 tensor cores (mma m16n8k32, cp.async double buffer) instead of dp4a.",
    env: "CEREBRO_NO_MMA", value: 1, default: 1,
  },
  {
    id: "cpu_threads", name: "Processor threads", group: "CPU", kind: { type: "number", min: 1, max: 20, zero_auto: false, unit: "threads" }, channel: "stable", applies: "restart",
    beginner: "How many processor threads the model may use. The default leaves room for your other apps.",
    advanced: "Size of the engine's worker pool (logical CPUs minus four by default).",
    env: "CEREBRO_THREADS", value: 16, default: 16,
  },
  {
    id: "expert_vram_mb", name: "Graphics memory for experts", group: "Mixture of experts", kind: { type: "number", min: 0, max: 262144, zero_auto: true, unit: "MB" }, channel: "beta", applies: "reload",
    beginner: "How much graphics memory holds the model's most-used parts. Automatic uses everything that is free.",
    advanced: "Caps the adaptive VRAM expert cache (0 = all free VRAM minus a 600 MB margin).",
    env: "CEREBRO_EXPERT_VRAM_MB", value: 0, default: 0,
  },
];
