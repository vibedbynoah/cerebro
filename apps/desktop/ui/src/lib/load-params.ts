import type { CpuAffinity, FlashAttention, HardwareProfile, KvCacheType, LibraryModel, LoadParams, NgramStrategy, SpeculativeMode } from "../service/types";
import { estimateFit } from "./fit";

/**
 * Load parameters: the limits, the validator and the starting settings. The same validator runs in
 * the load panel (field errors as you type) and in every CerebroService implementation (the mock
 * here, the Rust side for IPC), so nothing the panel cannot produce is ever accepted by a service.
 */

export const KV_TYPES: readonly KvCacheType[] = ["f16", "q8_0", "q4_0"];
export const FLASH_MODES: readonly FlashAttention[] = ["auto", "on", "off"];
export const AFFINITIES: readonly CpuAffinity[] = ["all", "performance", "custom"];
export const SPEC_MODES: readonly SpeculativeMode[] = ["off", "ngram", "draft"];
export const NGRAM_STRATEGIES: readonly NgramStrategy[] = ["ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"];

export const LIMITS = {
  contextMin: 256,
  /** Never above the model's own trained maximum, and never above this. */
  contextCap: 1_048_576,
  batchMin: 32,
  batchMax: 8192,
  draftMaxMin: 1,
  draftMaxMax: 32,
  coresMaxLength: 64,
  /** Thread ceiling when this machine was not detected: a sanity cap, not a claim about the CPU. */
  threadsCap: 256,
} as const;

export type LoadField = keyof LoadParams;
export type LoadErrors = Partial<Record<LoadField, string>>;

const isInt = (n: unknown): n is number => typeof n === "number" && Number.isInteger(n);
const inRange = (n: unknown, lo: number, hi: number): n is number => isInt(n) && n >= lo && n <= hi;
const oneOf = <T extends string>(v: unknown, list: readonly T[]): v is T => typeof v === "string" && (list as readonly string[]).includes(v);

/**
 * Parses a core list such as "0-5,8,10-11" into sorted, unique core indices, or null when it is not
 * one. Each index must be below `threads`; ranges run low to high; at most 64 characters.
 */
export function parseCores(text: string, threads: number): number[] | null {
  if (typeof text !== "string" || text.length === 0 || text.length > LIMITS.coresMaxLength) return null;
  if (!/^\d{1,3}(?:-\d{1,3})?(?:,\d{1,3}(?:-\d{1,3})?)*$/.test(text)) return null;
  const out = new Set<number>();
  for (const part of text.split(",")) {
    const [a, b] = part.split("-");
    const lo = Number(a);
    const hi = b === undefined ? lo : Number(b);
    if (!(lo <= hi) || hi >= threads) return null;
    for (let i = lo; i <= hi; i++) out.add(i);
  }
  return [...out].sort((x, y) => x - y);
}

/** Models that can draft for `model`: same family, text, smaller, and in the library. */
export function draftCandidates(model: LibraryModel, library: readonly LibraryModel[]): LibraryModel[] {
  return library.filter((m) => m.id !== model.id && (m.kind === "text" || m.kind === "vision") && m.family === model.family && m.sizeGiB < model.sizeGiB * 0.5);
}

/**
 * Field-by-field errors for `params` on `model` and this machine; an empty object means valid.
 * Unknown or missing fields, wrong types and out-of-range values are all errors, so a crafted
 * object can never slip through.
 */
export function validateLoadParams(
  params: unknown,
  model: Pick<LibraryModel, "architecture" | "contextMax" | "arch">,
  hw: Pick<HardwareProfile, "cpu"> | null,
  draftIds: ReadonlySet<string> = new Set(),
): LoadErrors {
  const e: LoadErrors = {};
  if (typeof params !== "object" || params === null || Array.isArray(params)) return { contextLength: "Settings are missing." };
  const p = params as Record<string, unknown>;
  const keys = Object.keys(p);
  const expected = Object.keys(STARTING_SHAPE);
  if (keys.length !== expected.length || !expected.every((k) => Object.hasOwn(p, k))) return { contextLength: "Settings are incomplete." };

  const maxCtx = Math.min(model.contextMax, LIMITS.contextCap);
  const maxThreads = maxThreadsFor(hw);
  if (!inRange(p["contextLength"], LIMITS.contextMin, maxCtx)) e.contextLength = `Between ${LIMITS.contextMin} and ${maxCtx.toLocaleString("en-US")} tokens.`;
  if (!inRange(p["gpuLayers"], 0, model.arch.layers)) e.gpuLayers = `Between 0 and ${model.arch.layers} layers.`;
  if (model.architecture === "moe") {
    if (!inRange(p["moeCpuLayers"], 0, model.arch.layers)) e.moeCpuLayers = `Between 0 and ${model.arch.layers} layers.`;
  } else if (p["moeCpuLayers"] !== 0) e.moeCpuLayers = "Only mixture-of-experts models have experts to move.";
  if (!oneOf(p["kvCacheK"], KV_TYPES)) e.kvCacheK = "Choose f16, q8_0 or q4_0.";
  if (!oneOf(p["kvCacheV"], KV_TYPES)) e.kvCacheV = "Choose f16, q8_0 or q4_0.";
  if (!oneOf(p["flashAttention"], FLASH_MODES)) e.flashAttention = "Choose auto, on or off.";
  else if (p["kvCacheV"] !== "f16" && p["flashAttention"] === "off") e.kvCacheV = "A quantized V cache needs flash attention on or auto.";
  if (!inRange(p["threads"], 1, maxThreads)) e.threads = `Between 1 and ${maxThreads} threads.`;
  if (!oneOf(p["cpuAffinity"], AFFINITIES)) e.cpuAffinity = "Choose a core set.";
  else if (p["cpuAffinity"] === "custom") {
    if (typeof p["cpuCores"] !== "string" || parseCores(p["cpuCores"], maxThreads) === null)
      e.cpuCores = `A list such as 0-5,8, using cores 0 to ${maxThreads - 1}.`;
  } else if (p["cpuCores"] !== "") e.cpuCores = "Only a custom core set takes a list.";
  if (!inRange(p["batchSize"], LIMITS.batchMin, LIMITS.batchMax)) e.batchSize = `Between ${LIMITS.batchMin} and ${LIMITS.batchMax}.`;
  if (!inRange(p["ubatchSize"], LIMITS.batchMin, LIMITS.batchMax)) e.ubatchSize = `Between ${LIMITS.batchMin} and ${LIMITS.batchMax}.`;
  else if (isInt(p["batchSize"]) && p["ubatchSize"] > p["batchSize"]) e.ubatchSize = "No larger than the batch size.";
  if (typeof p["mmap"] !== "boolean") e.mmap = "On or off.";
  if (typeof p["mlock"] !== "boolean") e.mlock = "On or off.";
  if (!oneOf(p["speculative"], SPEC_MODES)) e.speculative = "Choose off, n-gram or a draft model.";
  if (!oneOf(p["ngramStrategy"], NGRAM_STRATEGIES)) e.ngramStrategy = "Choose a strategy.";
  if (p["speculative"] === "draft") {
    if (typeof p["draftModelId"] !== "string" || !draftIds.has(p["draftModelId"])) e.draftModelId = "Choose a smaller model of the same family.";
  } else if (p["draftModelId"] !== null) e.draftModelId = "Only draft-model speculation uses a draft model.";
  if (!inRange(p["draftMax"], LIMITS.draftMaxMin, LIMITS.draftMaxMax)) e.draftMax = `Between ${LIMITS.draftMaxMin} and ${LIMITS.draftMaxMax} tokens.`;
  return e;
}

/** Highest thread count the validator accepts: this machine's, or a sanity cap when it is unknown. */
export function maxThreadsFor(hw: Pick<HardwareProfile, "cpu"> | null): number {
  return hw ? hw.cpu.threads : LIMITS.threadsCap;
}

/** Threads the CPU-only starting plan uses when this machine was not detected. */
const UNDETECTED_THREADS = 4;

/** Shape reference for the validator's exact-keys check. */
const STARTING_SHAPE: LoadParams = {
  contextLength: 8192,
  gpuLayers: 0,
  moeCpuLayers: 0,
  kvCacheK: "f16",
  kvCacheV: "f16",
  flashAttention: "auto",
  threads: 1,
  cpuAffinity: "all",
  cpuCores: "",
  batchSize: 2048,
  ubatchSize: 512,
  mmap: true,
  mlock: false,
  speculative: "off",
  ngramStrategy: "ngram-simple",
  draftModelId: null,
  draftMax: 16,
};

type StartModel = Pick<LibraryModel, "contextMax" | "arch" | "sizeGiB" | "architecture" | "projector">;

/**
 * Cerebro's starting settings for a model that has never been tuned: every value set explicitly
 * (Cerebro never inherits a llama-server default, docs/CLAIMS.md), a modest context, one thread per
 * physical core, and as much of the model on the GPU as the memory estimate says fits: for a MoE
 * model the expert weights of the first layers move to the CPU first, for a dense one whole layers
 * do. A tune replaces all of this with measured values. When this machine was not detected (`hw`
 * null) it is a conservative CPU-only plan: nothing on a GPU nobody has seen, and a few threads.
 */
export function startingParams(model: StartModel, hw: Pick<HardwareProfile, "cpu" | "gpu" | "ramTotalGiB"> | null): LoadParams {
  if (!hw) return { ...STARTING_SHAPE, contextLength: Math.min(4096, model.contextMax), gpuLayers: 0, threads: UNDETECTED_THREADS };
  const p: LoadParams = {
    ...STARTING_SHAPE,
    contextLength: Math.min(8192, model.contextMax),
    gpuLayers: hw.gpu ? model.arch.layers : 0,
    threads: Math.max(1, Math.min(hw.cpu.cores, hw.cpu.threads)),
  };
  const fits = (q: LoadParams) => estimateFit(model, q, hw).vram.status === "fits";
  if (!hw.gpu || fits(p)) return p;
  if (model.architecture === "moe") {
    for (let n = 1; n <= model.arch.layers; n++) {
      const q = { ...p, moeCpuLayers: n };
      if (fits(q)) return q;
    }
    p.moeCpuLayers = model.arch.layers;
  }
  for (let n = p.gpuLayers - 1; n >= 0; n--) {
    const q = { ...p, gpuLayers: n };
    if (fits(q)) return q;
  }
  return { ...p, gpuLayers: 0 };
}

/** The settings the Load button uses: the tuned ones, else the starting ones. Always a copy. */
export function baselineParams(model: LibraryModel, hw: Pick<HardwareProfile, "cpu" | "gpu" | "ramTotalGiB"> | null): LoadParams {
  return model.tunedParams ? { ...model.tunedParams } : startingParams(model, hw);
}

/** Fields whose value differs between two parameter sets, in display order. */
export function changedFields(a: LoadParams, b: LoadParams): LoadField[] {
  return (Object.keys(STARTING_SHAPE) as LoadField[]).filter((k) => a[k] !== b[k]);
}

export function hasErrors(errors: LoadErrors): boolean {
  return Object.keys(errors).length > 0;
}
