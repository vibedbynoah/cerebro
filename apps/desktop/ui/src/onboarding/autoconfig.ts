import { readPref, writePref } from "../lib/storage";
import { availability } from "../lib/quants";
import type { Channel } from "../lib/experience";
import type { CatalogModel, CatalogQuant, FlashAttention, HardwareProfile, KvCacheType } from "../service/types";
import { RAM_RESERVE_GIB, usableMemory, VRAM_HEADROOM_GIB } from "./recommend";

/**
 * The load settings setup writes for the first model.
 *
 * Beginner: `autoConfig` fills every value from the hardware check, and the setup step explains each
 * one in plain words. Advanced: the same object is edited field by field, and the raw backend flags
 * are shown. Switching level mid-setup shows the same object in the other view, so nothing is lost:
 * Beginner's values are Advanced's starting values, and Advanced edits are kept if the user goes
 * back to Beginner (marked as changed, with one button back to the automatic values).
 *
 * These are starting values from arithmetic on file size and memory, not measurements. Tune
 * measures and replaces them after download (docs/CLAIMS.md).
 */

export interface SetupConfig {
  modelId: string;
  quantId: string;
  contextLength: number;
  gpuLayers: number;
  /** MoE only: layers whose expert weights stay in system memory. 0 for dense models. */
  moeCpuLayers: number;
  kvCache: KvCacheType;
  flashAttention: FlashAttention;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
}

export type ConfigField = Exclude<keyof SetupConfig, "modelId">;

export const CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144] as const;
export const KV_CHOICES: readonly KvCacheType[] = ["f16", "q8_0", "q4_0"];
export const FA_CHOICES: readonly FlashAttention[] = ["auto", "on", "off"];

export const LIMITS = {
  temperature: { min: 0, max: 2, step: 0.05 },
  topP: { min: 0, max: 1, step: 0.01 },
  topK: { min: 0, max: 200, step: 1 },
  minP: { min: 0, max: 1, step: 0.01 },
} as const;

/** A file whose KL stays under this keeps close to the reference: Beginner never goes past it to fit. */
const KL_GOOD = 0.05;
/** Offloaded models keep a quarter of the memory budget free for context and the system. */
const OFFLOAD_SHARE = 0.75;

const clampInt = (n: number, lo: number, hi: number): number => (Number.isFinite(n) ? Math.min(hi, Math.max(lo, Math.round(n))) : lo);
const clampNum = (n: number, lo: number, hi: number): number => (Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : lo);

/** Files that can run here under the chosen channel. */
export function runnableQuants(model: CatalogModel, channel: Channel): CatalogQuant[] {
  return model.quants.filter((q) => availability(q, channel) === "runs");
}

/**
 * Beginner's file choice: the closest-to-reference file that fits in video memory with room for
 * context, as long as it stays under KL_GOOD; otherwise the closest that fits in three quarters of
 * video plus system memory; otherwise the smallest. Ties go to the smaller file.
 */
export function pickQuant(model: CatalogModel, hw: HardwareProfile | undefined, channel: Channel): CatalogQuant | undefined {
  const pool = runnableQuants(model, channel);
  if (pool.length === 0) return undefined;
  const byQuality = [...pool].sort((a, b) => a.klDiv - b.klDiv || a.sizeGiB - b.sizeGiB);
  if (!hw) return [...pool].sort((a, b) => a.sizeGiB - b.sizeGiB).find((q) => q.klDiv <= KL_GOOD) ?? byQuality[0];
  const budget = usableMemory(hw);
  const inVram = byQuality.find((q) => budget.vramGiB > 0 && q.sizeGiB <= budget.vramGiB && q.klDiv <= KL_GOOD);
  if (inVram) return inVram;
  const offload = byQuality.find((q) => q.sizeGiB <= budget.totalGiB * OFFLOAD_SHARE);
  return offload ?? [...pool].sort((a, b) => a.sizeGiB - b.sizeGiB)[0];
}

export type Placement = "gpu" | "moe-offload" | "split" | "cpu";

export function placementFor(model: CatalogModel, quant: CatalogQuant, hw: HardwareProfile | undefined): Placement {
  if (!hw?.gpu) return "cpu";
  const budget = usableMemory(hw);
  if (quant.sizeGiB <= budget.vramGiB) return "gpu";
  return model.architecture === "moe" ? "moe-offload" : "split";
}

/** Every value from the hardware check. Deterministic: the same machine and model give the same plan. */
export function autoConfig(model: CatalogModel, hw: HardwareProfile | undefined, channel: Channel): SetupConfig | null {
  const quant = pickQuant(model, hw, channel);
  if (!quant) return null;
  const place = placementFor(model, quant, hw);
  const L = model.layers;
  const vram = hw ? usableMemory(hw).vramGiB : 0;
  const ctxWanted = place === "gpu" ? 32768 : place === "moe-offload" ? 16384 : place === "split" ? 8192 : 4096;
  const contextLength = Math.min(ctxWanted, model.contextMax);
  let gpuLayers = 0;
  let moeCpuLayers = 0;
  if (place === "gpu") gpuLayers = L;
  else if (place === "moe-offload") {
    // Attention and shared weights stay on the card; enough layers' experts move to system memory
    // that the rest fits in about 85% of video memory.
    gpuLayers = L;
    moeCpuLayers = clampInt(Math.ceil((L * (quant.sizeGiB - vram * 0.85)) / quant.sizeGiB), 0, L);
  } else if (place === "split") gpuLayers = clampInt(Math.floor((L * vram * 0.9) / quant.sizeGiB), 0, L);
  return {
    modelId: model.id,
    quantId: quant.id,
    contextLength,
    gpuLayers,
    moeCpuLayers,
    // A full-precision cache while everything fits on the card; half the cache memory otherwise.
    kvCache: place === "gpu" && contextLength <= 32768 ? "f16" : "q8_0",
    flashAttention: place === "cpu" ? "auto" : "on",
    temperature: model.sampler.temperature,
    topP: model.sampler.topP,
    topK: model.sampler.topK,
    minP: model.sampler.minP,
  };
}

/** Makes any edited value legal for this model: clamps numbers, snaps lists, keeps a real file. */
export function sanitize(model: CatalogModel, cfg: SetupConfig): SetupConfig {
  const L = model.layers;
  const quant = model.quants.find((q) => q.id === cfg.quantId) ?? model.quants[0];
  const ctx = CONTEXT_STEPS.includes(cfg.contextLength as (typeof CONTEXT_STEPS)[number]) && cfg.contextLength <= model.contextMax ? cfg.contextLength : Math.min(4096, model.contextMax);
  return {
    modelId: model.id,
    quantId: quant?.id ?? cfg.quantId,
    contextLength: ctx,
    gpuLayers: clampInt(cfg.gpuLayers, 0, L),
    moeCpuLayers: model.architecture === "moe" ? clampInt(cfg.moeCpuLayers, 0, L) : 0,
    kvCache: KV_CHOICES.includes(cfg.kvCache) ? cfg.kvCache : "f16",
    flashAttention: FA_CHOICES.includes(cfg.flashAttention) ? cfg.flashAttention : "auto",
    temperature: Math.round(clampNum(cfg.temperature, LIMITS.temperature.min, LIMITS.temperature.max) * 100) / 100,
    topP: Math.round(clampNum(cfg.topP, LIMITS.topP.min, LIMITS.topP.max) * 100) / 100,
    topK: clampInt(cfg.topK, LIMITS.topK.min, LIMITS.topK.max),
    minP: Math.round(clampNum(cfg.minP, LIMITS.minP.min, LIMITS.minP.max) * 100) / 100,
  };
}

const FIELDS: readonly ConfigField[] = ["quantId", "contextLength", "gpuLayers", "moeCpuLayers", "kvCache", "flashAttention", "temperature", "topP", "topK", "minP"];

/** Fields that differ from the automatic plan: what Beginner marks as "changed in Advanced". */
export function changedFrom(auto: SetupConfig | null, cfg: SetupConfig | null): ConfigField[] {
  if (!auto || !cfg || auto.modelId !== cfg.modelId) return [];
  return FIELDS.filter((f) => auto[f] !== cfg[f]);
}

/** The file name a quant is saved under. Shown in flags; never a path the user typed. */
export function fileName(model: CatalogModel, quant: CatalogQuant): string {
  return `${model.name}-${quant.label.replace(/\s+/g, "-")}.gguf`;
}

/**
 * The backend command line the settings produce, one flag per line, for Advanced. Always loopback
 * (`--host 127.0.0.1`); Settings decides whether that changes.
 */
export function configFlags(model: CatalogModel, cfg: SetupConfig): string[] {
  const quant = model.quants.find((q) => q.id === cfg.quantId);
  const bin = quant?.backend === "strata" ? "strata-server" : "cerebro serve";
  const lines = [
    `${bin} -m ${quant ? fileName(model, quant) : "<file>"}`,
    `  -c ${cfg.contextLength}`,
    `  -ngl ${cfg.gpuLayers}`,
  ];
  if (model.architecture === "moe" && cfg.moeCpuLayers > 0) lines.push(`  --n-cpu-moe ${cfg.moeCpuLayers}`);
  lines.push(
    `  -ctk ${cfg.kvCache} -ctv ${cfg.kvCache}`,
    `  -fa ${cfg.flashAttention}`,
    `  --temp ${cfg.temperature} --top-p ${cfg.topP} --top-k ${cfg.topK} --min-p ${cfg.minP}`,
    "  --host 127.0.0.1",
  );
  return lines;
}

export function contextLabel(n: number): string {
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}

// ---- Persistence ---------------------------------------------------------------------------------

export const SETUP_CONFIG_KEY = "cerebro-setup-config";
const ID = /^[a-z0-9][a-z0-9._-]{0,127}$/;

/** Reads a stored plan back, field by field; anything malformed is dropped whole. */
export function parseConfig(raw: string | null): SetupConfig | null {
  if (raw === null || raw.length > 1024) return null;
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) return null;
  const r = data as Record<string, unknown>;
  const str = (k: string) => (typeof r[k] === "string" ? (r[k] as string) : null);
  const num = (k: string) => (typeof r[k] === "number" && Number.isFinite(r[k]) ? (r[k] as number) : null);
  const modelId = str("modelId");
  const quantId = str("quantId");
  const kv = str("kvCache");
  const fa = str("flashAttention");
  const nums = (["contextLength", "gpuLayers", "moeCpuLayers", "temperature", "topP", "topK", "minP"] as const).map((k) => num(k));
  if (!modelId || !ID.test(modelId) || !quantId || !ID.test(quantId)) return null;
  if (!kv || !KV_CHOICES.includes(kv as KvCacheType) || !fa || !FA_CHOICES.includes(fa as FlashAttention)) return null;
  if (nums.some((n) => n === null)) return null;
  const [contextLength, gpuLayers, moeCpuLayers, temperature, topP, topK, minP] = nums as number[];
  return {
    modelId,
    quantId,
    contextLength: contextLength ?? 4096,
    gpuLayers: gpuLayers ?? 0,
    moeCpuLayers: moeCpuLayers ?? 0,
    kvCache: kv as KvCacheType,
    flashAttention: fa as FlashAttention,
    temperature: temperature ?? 0.7,
    topP: topP ?? 0.95,
    topK: topK ?? 40,
    minP: minP ?? 0,
  };
}

export function readConfig(): SetupConfig | null {
  return parseConfig(readPref(SETUP_CONFIG_KEY));
}

export function writeConfig(cfg: SetupConfig | null): void {
  writePref(SETUP_CONFIG_KEY, cfg ? JSON.stringify(cfg) : null);
}

/** Reserve figures re-exported for the Beginner explanation. */
export { RAM_RESERVE_GIB, VRAM_HEADROOM_GIB };
