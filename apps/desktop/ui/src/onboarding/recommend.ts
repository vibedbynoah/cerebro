import type { CatalogModel, HardwareProfile } from "../service/types";

/**
 * Ranks catalog models for one machine by how they fit its memory, before anything is measured.
 * This is arithmetic on file size and memory, never a speed claim: every "why" says what the fit
 * implies and that Cerebro measures after download (docs/CLAIMS.md).
 */

/** Video memory kept free for the context (KV cache) and the driver. A rule of thumb. */
export const VRAM_HEADROOM_GIB = 1;
/** System memory left to Windows and other programs. A rule of thumb. */
export const RAM_RESERVE_GIB = 6;

/**
 * - `gpu`: the whole model fits in video memory.
 * - `moe-offload`: a mixture-of-experts model that fits across video and system memory.
 * - `split`: a dense model that fits only across video and system memory.
 * - `too-large`: more than both together.
 * - `unknown`: this machine could not be read, so nothing is ranked (see `unranked`).
 */
export type Fit = "gpu" | "moe-offload" | "split" | "too-large" | "unknown";

export interface Recommendation {
  model: CatalogModel;
  fit: Fit;
  /** 1-based position in the ranking (catalog position when unranked). */
  rank: number;
  /** False when the hardware is unknown and the list is in catalog order. */
  ranked: boolean;
  /** One or two plain sentences: why this model sits where it does on this machine. */
  why: string;
  /** Short label for the fit chip. */
  fitLabel: string;
  /** Model size as a share of usable video plus system memory, 0..1 (capped). */
  share: number;
  /** Usable video memory as a share of usable video plus system memory, 0..1. */
  vramShare: number;
}

export interface MemoryBudget {
  vramGiB: number;
  ramGiB: number;
  totalGiB: number;
}

const finite = (n: number): number => (Number.isFinite(n) && n > 0 ? n : 0);

export function usableMemory(hw: HardwareProfile): MemoryBudget {
  const vramGiB = Math.max(0, finite(hw.gpu?.vramTotalGiB ?? 0) - VRAM_HEADROOM_GIB);
  const ramGiB = Math.max(0, finite(hw.ramTotalGiB) - RAM_RESERVE_GIB);
  return { vramGiB, ramGiB, totalGiB: vramGiB + ramGiB };
}

export function fitFor(model: CatalogModel, budget: MemoryBudget): Fit {
  const size = finite(model.sizeGiB);
  if (budget.vramGiB > 0 && size <= budget.vramGiB) return "gpu";
  if (size <= budget.totalGiB) return model.architecture === "moe" ? "moe-offload" : "split";
  return "too-large";
}

const TIER: Record<Fit, number> = { gpu: 0, "moe-offload": 1, split: 2, "too-large": 3, unknown: 4 };

// Labels and reasons describe the memory mechanism, never a speed ranking: nothing is measured at
// this point, and which offloaded model ends up faster depends on the machine (docs/CLAIMS.md).
const LABEL: Record<Fit, string> = {
  gpu: "Fits in video memory",
  "moe-offload": "Tuning matters most",
  split: "Runs partly from system memory",
  "too-large": "Too large for this machine",
  unknown: "Not ranked",
};

const gib = (n: number): string => `${(Math.round(n * 10) / 10).toFixed(1)} GiB`;

function why(model: CatalogModel, fit: Fit, hw: HardwareProfile): string {
  const vram = hw.gpu ? gib(hw.gpu.vramTotalGiB) : null;
  switch (fit) {
    case "gpu":
      return `Fits entirely in your ${vram ?? "available"} of video memory, with room for context. Cerebro measures whether the defaults are already the fastest setting here.`;
    case "moe-offload":
      return `Larger than your video memory, but only ${model.activeParams ?? "a few"} of its parameters work per token, so reading system memory costs it less. Tuning matters most here; Cerebro measures it.`;
    case "split":
      return hw.gpu
        ? "Larger than your video memory and dense, so every token also reads system memory, which is slower. Cerebro measures what that costs after download."
        : "No graphics card the engine can use was found, so it runs on the processor from system memory. Cerebro measures what that costs after download.";
    case "too-large":
      return `Needs about ${gib(model.sizeGiB)}, more than this machine's video and system memory can spare together. It would not load here.`;
    case "unknown":
      return UNRANKED_WHY;
  }
}

const UNRANKED_WHY = "Not ranked: this machine could not be read. Tune measures it after download.";

/**
 * Best fit first: whole-model-in-VRAM, then MoE with offload, then dense with offload, then models
 * that will not load. Inside a tier, the more capable (larger) model leads when it fits in VRAM;
 * once offloading, the smaller one (less to read per token) leads. Ties keep catalog order.
 */
export function rankModels(catalog: readonly CatalogModel[], hw: HardwareProfile): Recommendation[] {
  const budget = usableMemory(hw);
  const scored = catalog.map((model, index) => ({ model, index, fit: fitFor(model, budget) }));
  scored.sort((a, b) => {
    const tier = TIER[a.fit] - TIER[b.fit];
    if (tier !== 0) return tier;
    const size = a.fit === "gpu" ? finite(b.model.sizeGiB) - finite(a.model.sizeGiB) : finite(a.model.sizeGiB) - finite(b.model.sizeGiB);
    return size !== 0 ? size : a.index - b.index;
  });
  const total = budget.totalGiB;
  return scored.map(({ model, fit }, i) => ({
    model,
    fit,
    rank: i + 1,
    ranked: true,
    why: why(model, fit, hw),
    fitLabel: LABEL[fit],
    share: total > 0 ? Math.min(1, finite(model.sizeGiB) / total) : 1,
    vramShare: total > 0 ? budget.vramGiB / total : 0,
  }));
}

/**
 * The catalog in its own order, for when the hardware check failed: nothing can be said about fit,
 * so nothing is ranked or ruled out, and the user picks by hand. Tune measures whatever they pick.
 */
export function unranked(catalog: readonly CatalogModel[]): Recommendation[] {
  return catalog.map((model, i) => ({ model, fit: "unknown", rank: i + 1, ranked: false, why: UNRANKED_WHY, fitLabel: LABEL.unknown, share: 0, vramShare: 0 }));
}

/** 0..1 to the design system's `.bar--pNN` class steps (motion.css), so widths need no inline style. */
export function pct(share: number): number {
  return Math.round(Math.min(1, Math.max(0, Number.isFinite(share) ? share : 0)) * 100);
}
