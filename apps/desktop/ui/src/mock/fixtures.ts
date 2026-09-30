import type { GpuInfo, LiveStats, ModelSummary, Settings } from "../service/types";

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
    tune: "tuning",
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

/** Safe by default: every protection on, every network feature allowed but optional. */
export const DEFAULT_SETTINGS: Settings = {
  processContainment: true,
  cleanEnvironment: true,
  integrityCheck: true,
  permissionPrompts: true,
  loopbackOnly: true,
  offlineMode: false,
  allowModelDownloads: true,
  allowPluginInstalls: true,
};
