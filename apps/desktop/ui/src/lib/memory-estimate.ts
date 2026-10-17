import type { HardwareProfile, KvCacheType } from "../service/types";
import { usableMemory } from "../onboarding/recommend";

/**
 * Memory estimate for one GGUF file at a chosen context: the Model Hub's per-file "needs X GB VRAM
 * + Y GB RAM at NK context" and its "Best for this computer" ranking. Every input is real — the
 * file size from Hugging Face, the layer and attention numbers from the GGUF header, the memory
 * from the OS — and the arithmetic below is the same the engine uses (docs/CLAIMS.md). It is an
 * estimate, never a measurement; a load measures the real thing.
 *
 * Weights: the file's real size, split evenly across layers (plus ~8% for the embeddings and the
 * output head, which go to the GPU only when every layer does). KV cache: layers x kv heads x head
 * size x 2 (K and V) x context x bytes per element of the cache type, following the layers.
 * Mixture of experts: the expert share of a GPU layer may sit in RAM (Cerebro Engine's expert
 * cache), so only its attention and shared weights need video memory.
 */

const GIB = 1024 ** 3;

/** Bytes per element of each KV cache type (GGUF block size for the quantised ones). */
export const KV_BYTES: Readonly<Record<KvCacheType, number>> = { f16: 2, q8_0: 1.0625, q4_0: 0.5625 };

const NON_LAYER_SHARE = 0.08;
const EXPERT_SHARE = 0.85;
/** Driver and runtime overhead on the GPU once anything is offloaded. */
const GPU_OVERHEAD_GIB = 0.3;
/** Micro-batch the compute buffers are sized for. */
const MICRO_BATCH = 512;

/** The architecture numbers an estimate needs. As published or read from the file's GGUF header. */
export interface MemoryModel {
  architecture: "dense" | "moe";
  layers: number;
  kvHeads: number;
  headDim: number;
  embedding: number;
  contextMax: number;
  /** Layers with a KV cache when fewer than `layers` (Qwen3.5/3.8 hybrids: one in four). */
  attentionLayers?: number | undefined;
}

/** One file to estimate: its size on disk is what the weights cost. */
export interface MemoryQuant {
  label: string;
  sizeGiB: number;
}

/** How many layers go on the GPU and the memory the OS can spare for each. */
export interface MemoryPlan {
  gpuLayers: number;
  vramGiB: number;
  ramGiB: number;
}

/** Usable memory of this machine (a plan with no layers spends none of it). */
export type MemoryBudget = Omit<MemoryPlan, "gpuLayers">;

export type MemoryFit = "gpu" | "split" | "cpu" | "no";

export interface MemoryEstimate {
  /** The whole file, GiB. */
  weightsGiB: number;
  /** The KV cache for every layer at this context, GiB. */
  kvGiB: number;
  /** Compute buffers and driver overhead, GiB. */
  scratchGiB: number;
  /** Video memory this plan needs, GiB. */
  vramGiB: number;
  /** System memory this plan needs, GiB. */
  ramGiB: number;
  fits: MemoryFit;
  /** One plain sentence saying where it runs. */
  note: string;
}

const round2 = (n: number): number => Math.round(n * 100) / 100;
const finite = (n: number): number => (Number.isFinite(n) && n > 0 ? n : 0);
const int = (n: number, hi: number): number => (Number.isFinite(n) ? Math.min(hi, Math.max(0, Math.round(n))) : 0);

function noteFor(fits: MemoryFit, moe: boolean): string {
  switch (fits) {
    case "gpu":
      return "The whole file and its context fit in video memory.";
    case "split":
      return moe ? "Attention and shared weights on the card; experts stay in system memory." : "Weights split across video and system memory; the context follows the layers.";
    case "cpu":
      return "No room on the graphics card: it runs from system memory on the processor.";
    case "no":
      return "Needs more video and system memory together than this machine can spare.";
  }
}

const FIT_RANK: Readonly<Record<MemoryFit, number>> = { gpu: 0, split: 1, cpu: 2, no: 3 };

/** How one file fits a plan: what it needs in each memory, in GiB. */
export function estimate(model: MemoryModel, quant: MemoryQuant, ctx: number, kvType: KvCacheType, offload: MemoryPlan): MemoryEstimate {
  const layers = Math.max(1, int(model.layers, 1 << 20));
  const gpuLayers = int(offload.gpuLayers, layers);
  const size = finite(quant.sizeGiB);
  const context = Math.max(0, Math.round(finite(ctx)));

  const perLayer = (size * (1 - NON_LAYER_SHARE)) / layers;
  const nonLayer = size * NON_LAYER_SHARE;
  const allOnGpu = gpuLayers === layers;
  const expertInRam = model.architecture === "moe" ? gpuLayers * perLayer * EXPERT_SHARE : 0;
  // A mixture-of-experts file keeps an expert share in system memory even at full offload, so it is
  // never a whole-file-in-video-memory fit.
  const wholeOnGpu = allOnGpu && expertInRam === 0;
  const gpuWeights = Math.max(0, gpuLayers * perLayer - expertInRam + (allOnGpu ? nonLayer : 0));
  const ramWeights = Math.max(0, size - gpuWeights);

  // Hybrid models keep a KV cache only on their attention layers; spread it over all layers.
  const kvShare = model.attentionLayers && model.layers > 0 ? Math.min(1, model.attentionLayers / model.layers) : 1;
  const kvPerLayer = (model.kvHeads * model.headDim * 2 * KV_BYTES[kvType] * context * kvShare) / GIB;
  const kvGpu = gpuLayers * kvPerLayer;
  const kvCpu = (layers - gpuLayers) * kvPerLayer;

  const compute = (MICRO_BATCH * model.embedding * 4 * 6) / GIB;
  const gpuScratch = gpuLayers > 0 ? compute + GPU_OVERHEAD_GIB : 0;
  const cpuScratch = gpuLayers > 0 ? (MICRO_BATCH * model.embedding * 4) / GIB : compute;

  const vramGiB = gpuWeights + kvGpu + gpuScratch;
  const ramGiB = ramWeights + kvCpu + cpuScratch;
  const vramOk = vramGiB <= offload.vramGiB + 1e-9;
  const ramOk = ramGiB <= offload.ramGiB + 1e-9;

  let fits: MemoryFit;
  if (gpuLayers === 0) fits = ramOk ? "cpu" : "no";
  else if (vramOk && ramOk) fits = wholeOnGpu ? "gpu" : "split";
  else if (ramOk) fits = "cpu";
  else fits = "no";

  return {
    weightsGiB: round2(size),
    kvGiB: round2(kvPerLayer * layers),
    scratchGiB: round2(compute + (gpuLayers > 0 ? GPU_OVERHEAD_GIB : 0)),
    vramGiB: round2(vramGiB),
    ramGiB: round2(ramGiB),
    fits,
    note: noteFor(fits, model.architecture === "moe"),
  };
}

/** Usable memory of a machine, the same table the setup screen uses. Zero when it could not be read. */
export function budgetFor(hw: HardwareProfile | undefined): MemoryBudget {
  const m = hw ? usableMemory(hw) : { vramGiB: 0, ramGiB: 0 };
  return { vramGiB: round2(m.vramGiB), ramGiB: round2(m.ramGiB) };
}

/**
 * The plan that puts as many layers on the card as it can still hold, or none when there is no
 * card. Deterministic: the same file and machine always give the same plan.
 */
export function planFor(model: MemoryModel, quant: MemoryQuant, ctx: number, kvType: KvCacheType, budget: MemoryBudget): MemoryPlan {
  const layers = Math.max(1, int(model.layers, 1 << 20));
  if (budget.vramGiB <= 0) return { gpuLayers: 0, vramGiB: budget.vramGiB, ramGiB: budget.ramGiB };
  for (let g = layers; g >= 0; g--) {
    const e = estimate(model, quant, ctx, kvType, { gpuLayers: g, vramGiB: budget.vramGiB, ramGiB: budget.ramGiB });
    if (e.vramGiB <= budget.vramGiB + 1e-9) return { gpuLayers: g, vramGiB: budget.vramGiB, ramGiB: budget.ramGiB };
  }
  return { gpuLayers: 0, vramGiB: budget.vramGiB, ramGiB: budget.ramGiB };
}

/** Estimate with the plan this machine would choose. */
export function estimateFor(model: MemoryModel, quant: MemoryQuant, ctx: number, kvType: KvCacheType, hw: HardwareProfile | undefined): MemoryEstimate {
  const budget = budgetFor(hw);
  return estimate(model, quant, ctx, kvType, planFor(model, quant, ctx, kvType, budget));
}

// ---- Recommendations --------------------------------------------------------------------------

/**
 * How Cerebro ranks a file's quantisation for a default pick. Unsloth Dynamic first, then the
 * plain K-quants people recognise (Q4_K_M, Q5, Q8); anything else is last. Lower is preferred.
 */
export function qualityTier(label: string): number {
  const up = label.toUpperCase();
  if (up.includes("UD")) return 0;
  if (up.includes("Q4_K_M")) return 1;
  if (up.includes("Q5")) return 2;
  if (up.includes("Q8")) return 3;
  return 4;
}

export interface MemoryCandidate {
  model: MemoryModel;
  quant: MemoryQuant;
}

export interface MemoryPick extends MemoryCandidate {
  estimate: MemoryEstimate;
  /** 1-based position. */
  rank: number;
  /** One plain sentence: why this file sits at this rank for this machine. */
  why: string;
}

function whyFor(c: MemoryCandidate & { estimate: MemoryEstimate }): string {
  const size = `${finite(c.quant.sizeGiB).toFixed(1)} GiB`;
  switch (c.estimate.fits) {
    case "gpu":
      return `${c.quant.label} is the closest fit entirely in video memory (${size}).`;
    case "split":
      return `${c.quant.label} runs across video and system memory (${size}); Cerebro measures the speed after download.`;
    case "cpu":
      return `${c.quant.label} runs from system memory on this machine (${size}).`;
    case "no":
      return `${c.quant.label} needs more memory than this machine can spare (${size}).`;
  }
}

/**
 * Best files for this machine: whole-in-VRAM first, then the preferred quantisation tier, then the
 * larger (more capable) file, then the label so the order is stable. Files that will not load are
 * kept last, never recommended first. The caller's own extra fields (the model it came from) are
 * carried through untouched.
 */
export function recommend<T extends MemoryCandidate>(candidates: readonly T[], hw: HardwareProfile | undefined, ctx: number, kvType: KvCacheType, limit = 3): (T & MemoryPick)[] {
  const scored = candidates.map((c) => ({ ...c, estimate: estimateFor(c.model, c.quant, ctx, kvType, hw) }));
  scored.sort((a, b) => {
    const fit = FIT_RANK[a.estimate.fits] - FIT_RANK[b.estimate.fits];
    if (fit !== 0) return fit;
    const tier = qualityTier(a.quant.label) - qualityTier(b.quant.label);
    if (tier !== 0) return tier;
    const size = finite(b.quant.sizeGiB) - finite(a.quant.sizeGiB);
    return size !== 0 ? size : a.quant.label.localeCompare(b.quant.label);
  });
  return scored.slice(0, Math.max(0, limit)).map((c, i) => ({ ...c, rank: i + 1, why: whyFor(c) }));
}
