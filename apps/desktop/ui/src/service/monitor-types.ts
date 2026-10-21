import type { Provenance } from "./types";

/**
 * Monitor: the engine's live /metrics snapshot, polled once a second. Same sections as the engine's
 * JSON (live, hardware, hardware_static, engine, history, totals, requests, conversation_cache);
 * the IPC bridge maps snake_case to these camelCase names. Every value the engine cannot read is
 * null and is shown as a dash, never as zero. Plain data only (survives a JSON round trip).
 */

/** What the engine is doing now. `loading` is a model being read into memory. */
/** "waiting": a tool call waits for the user's approval; "tool": a tool is running. */
export type EngineActivity = "idle" | "loading" | "reading" | "generating" | "waiting" | "tool";

export interface MetricsLive {
  state: EngineActivity;
  /** Requests waiting behind the running one. */
  queued: number;
  /** The running request, or null when idle. */
  requestId: string | null;
  /** Prompt tokens processed so far (reused tokens count as processed) and the prompt's length. */
  promptRead: number;
  promptTotal: number;
  /** Tokens of this prompt taken from the conversation cache instead of being read again. */
  promptReused: number;
  generated: number;
  maxTokens: number;
  /** Writing speed now, tok/s; null outside generation. */
  decodeTps: number | null;
  /** Mean prompt-reading speed of the running request, tok/s (new tokens only). */
  prefillTps: number | null;
  /** Seconds from arrival to the first written token, once known. */
  firstTokenS: number | null;
  /** What the model is writing: hidden reasoning, the answer, or a tool call. */
  phase: "reasoning" | "answer" | "tool-call" | null;
}

/** One graphics card's readings. Totals for a multi-GPU machine are in MetricsHardware. */
export interface GpuReading {
  index: number;
  name: string;
  util: number | null;
  memUsedGiB: number | null;
  memTotalGiB: number;
  tempC: number | null;
  powerW: number | null;
  powerLimitW: number | null;
  pcieRxMBs: number | null;
}

/** What VRAM holds, GiB. `other` is memory other programs use on the same card(s). */
export interface VramSplit {
  weights: number;
  kv: number;
  expertCache: number;
  scratch: number;
  other: number;
}

export type ThrottleReason = "thermal" | "power";

export interface MetricsHardware {
  /** Busy share of the GPU(s), percent (mean across cards). */
  gpuUtil: number | null;
  vramUsedGiB: number | null;
  vramTotalGiB: number;
  vram: VramSplit | null;
  /** Hottest card, degrees Celsius. */
  gpuTempC: number | null;
  /** Sum across cards, watts. */
  gpuPowerW: number | null;
  gpuPowerLimitW: number | null;
  /** Host-to-GPU traffic over PCIe, MB/s (sum across cards). */
  pcieRxMBs: number | null;
  pcieTxMBs: number | null;
  /** Link generation now (cards drop to a low generation when idle), the card's maximum, and lanes. */
  pcieGen: number | null;
  pcieGenMax: number | null;
  pcieWidth: number | null;
  /** Mean CPU load across all threads, percent. */
  cpu: number | null;
  ramUsedGiB: number | null;
  ramTotalGiB: number;
  /** System memory locked for model weights (MoE experts computed on the CPU live here). */
  ramPinnedGiB: number | null;
  diskReadMBs: number | null;
  diskWriteMBs: number | null;
  /** Why the GPU runs below its full clocks right now, as the driver reports it. */
  throttle: ThrottleReason[];
  gpus: GpuReading[];
}

export interface MetricsHardwareStatic {
  gpuName: string | null;
  gpuCount: number;
  driver: string | null;
  cpuName: string;
  cores: number;
  threads: number;
  ramTotalGiB: number;
  /** Temperature at which the driver starts slowing the card down, Celsius. */
  gpuThrottleTempC: number | null;
}

export type ReleaseChannel = "stable" | "beta" | "experimental";

export interface EngineModel {
  id: string;
  name: string;
  quant: string;
  params: string;
  architecture: "dense" | "moe";
  contextMax: number;
  /** Tokens in the context now: the running request's, else the last one's. */
  contextUsed: number;
  kvType: string;
  layers: number;
  gpuLayers: number;
  /** MoE: layers whose expert weights stay in system memory and run on the CPU. */
  expertLayersOnCpu: number;
}

export interface MetricsEngine {
  version: string;
  backend: "cerebro" | "strata";
  build: string;
  /** Loopback address the API listens on. */
  address: string;
  model: EngineModel | null;
  moe: {
    expertsPerToken: number;
    expertsTotal: number;
    /** Experts held in the VRAM expert cache. */
    cacheSlots: number;
    /** Share of expert lookups served from the VRAM cache, percent, over the last 60 s. */
    cacheHitPct: number | null;
    channel: ReleaseChannel;
  } | null;
  speculation: {
    /** Plain-words strategy label, e.g. "Hash of recent text". */
    method: string;
    acceptancePct: number | null;
    /** Tokens accepted per verify pass (1 means no gain). */
    tokensPerPass: number | null;
    channel: ReleaseChannel;
  } | null;
  /** What the model's last tune measured: the baseline a speed regression is judged against. */
  baseline: { decodeTps: number; prefillTps: number; depth: number; measuredAt: string } | null;
}

export const SERIES_KEYS = [
  "decodeTps",
  "prefillTps",
  "gpuUtil",
  "vramUsedGiB",
  "gpuTempC",
  "gpuPowerW",
  "pcieRxMBs",
  "cpu",
  "ramUsedGiB",
  "diskReadMBs",
  "expertHitPct",
  "draftAcceptPct",
] as const;
export type SeriesKey = (typeof SERIES_KEYS)[number];

/** Evenly spaced samples, oldest first; the last one ends at `end` (epoch ms). A null is a gap
 *  (nothing read, or nothing running for a speed). */
export interface MetricsSeries {
  stepS: number;
  end: number;
  series: Record<SeriesKey, Array<number | null>>;
}

export type HistoryRange = "1m" | "15m" | "1h" | "24h";

export interface MetricsTotals {
  /** Epoch ms the engine started counting. */
  since: number;
  requests: number;
  errors: number;
  promptTokens: number;
  reusedTokens: number;
  /** Time spent reading new prompt tokens, ms. */
  promptMs: number;
  outputTokens: number;
  decodeMs: number;
}

export type RequestState = "queued" | "loading" | "reading" | "generating" | "completed" | "error" | "cancelled";

export interface RequestSummary {
  id: string;
  /** Epoch ms. */
  startedAt: number;
  method: "POST";
  path: string;
  model: string;
  state: RequestState;
  stream: boolean;
  httpStatus: number | null;
  /** Arrival to last byte, seconds; null while running. */
  wallS: number | null;
  loadS: number | null;
  queueS: number | null;
  firstTokenS: number | null;
  promptTokens: number;
  reusedTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  decodeTps: number | null;
  prefillTps: number | null;
  responseFormat: "text" | "json_object" | "json_schema";
  /** The client session the request belongs to (a chat, an agent run, an app using the API). */
  session: string;
  /** The tool the model called in its answer, or null for a plain answer. */
  tool: string | null;
}

export interface RequestDetail extends RequestSummary {
  input: string;
  output: string;
  reasoning: string;
  response: string;
  /** Captures are cut at 256K characters each; the API response itself is never shortened. */
  truncated: { input: boolean; output: boolean; reasoning: boolean; response: boolean };
  error: string | null;
}

export interface ConversationCacheStats {
  /** Parking several conversations in RAM is on. The last conversation is always kept. */
  enabled: boolean;
  slots: number;
  parked: number;
  budgetGiB: number;
  usedGiB: number;
  requests: number;
  requestsReused: number;
  promptTokens: number;
  reusedTokens: number;
  parks: number;
  restores: number;
  evictions: number;
  lastEvent: { kind: "parked" | "restored"; tokens: number; at: number } | null;
}

export interface MetricsSnapshot {
  provenance: Provenance;
  /** Epoch ms of this snapshot. */
  at: number;
  live: MetricsLive;
  hardware: MetricsHardware;
  hardwareStatic: MetricsHardwareStatic;
  engine: MetricsEngine;
  /** The last 60 s at one sample per second. */
  history: MetricsSeries;
  totals: MetricsTotals;
  /** Newest first; at most `requestsKept`. */
  requests: RequestSummary[];
  requestsKept: number;
  conversationCache: ConversationCacheStats;
}
