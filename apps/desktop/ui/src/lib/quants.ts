import type { CatalogQuant, QuantBackend, QuantFormat } from "../service/types";
import { channelIncludes, type Channel } from "./experience";

/**
 * Comparing quantised files of one model by quality per GiB. Quality is how close a file stays to
 * the unquantised weights (KL divergence); size is what it costs in memory. Arithmetic on the
 * catalogue's figures only: whether those are measured or illustrative is carried by each file's
 * `provenance` and shown beside the numbers.
 */

export const FORMAT_LABELS: Readonly<Record<QuantFormat, string>> = {
  "gsq-rco": "GSQ-RCO",
  "unsloth-ud": "Unsloth Dynamic",
  "imatrix-iq": "imatrix",
  "k-quant": "K-quant",
  exl3: "EXL3",
  awq: "AWQ",
  mlx: "MLX",
};

/** One plain sentence per family, shown under the table and in the Beginner explanation. */
export const FORMAT_NOTES: Readonly<Record<QuantFormat, string>> = {
  "gsq-rco": "Non-uniform: spends bits where the weights need them most, so it keeps quality at a smaller size.",
  "unsloth-ud": "Keeps the most sensitive layers at higher precision and squeezes the rest.",
  "imatrix-iq": "Calibrated on sample text (an importance matrix) so the bits go to the weights that matter.",
  "k-quant": "The classic GGUF format: simple, widely compatible, a little larger for the same quality.",
  exl3: "A GPU-only format for the ExLlamaV3 engine.",
  awq: "Activation-aware 4-bit weights for vLLM.",
  mlx: "Apple's format for Apple silicon.",
};

export const BACKEND_LABELS: Readonly<Record<QuantBackend, string>> = {
  "llama.cpp": "llama.cpp",
  strata: "Strata",
  exllamav3: "ExLlamaV3",
  vllm: "vLLM",
  mlx: "MLX",
};

export type QuantAvailability = "runs" | "channel" | "elsewhere";

/** Runs here now; runs here once the Experimental channel (Strata) is on; or needs another engine. */
export function availability(q: CatalogQuant, channel: Channel): QuantAvailability {
  if (q.unavailable !== null || q.decodeTps === null) return "elsewhere";
  if (q.backend === "strata" && !channelIncludes(channel, "experimental")) return "channel";
  return "runs";
}

/** Share of the reference quality kept, 0..100: exp(-KL), the usual way KL is read as "how close". */
export function qualityKept(q: CatalogQuant): number {
  const kl = Number.isFinite(q.klDiv) && q.klDiv >= 0 ? q.klDiv : Number.POSITIVE_INFINITY;
  return Math.exp(-kl) * 100;
}

/** Quality kept per GiB of file: higher is a better use of memory. 0 for a broken size. */
export function qualityPerGiB(q: CatalogQuant): number {
  return Number.isFinite(q.sizeGiB) && q.sizeGiB > 0 ? qualityKept(q) / q.sizeGiB : 0;
}

/**
 * Files no other file beats on both size and KL ("best at its size"). Only files that run here
 * (under the chosen channel) compete, so the badge is about this machine.
 */
export function bestAtSize(quants: readonly CatalogQuant[], channel: Channel): Set<string> {
  const pool = quants.filter((q) => availability(q, channel) === "runs");
  const out = new Set<string>();
  for (const a of pool) {
    const beaten = pool.some((b) => b !== a && b.sizeGiB <= a.sizeGiB && b.klDiv <= a.klDiv && (b.sizeGiB < a.sizeGiB || b.klDiv < a.klDiv));
    if (!beaten) out.add(a.id);
  }
  return out;
}

export function fmtKl(kl: number): string {
  return Number.isFinite(kl) ? kl.toFixed(3) : "–";
}
