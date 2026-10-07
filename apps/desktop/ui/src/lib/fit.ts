import type { HardwareProfile, KvCacheType, LibraryModel, LoadParams } from "../service/types";

/**
 * Memory fit ESTIMATE for a model at given load settings: what the load panel's bar and the
 * library's fit column show. It is arithmetic from the file size and the architecture, not a
 * measurement, and every place that shows it says "estimate". A tune measures the real thing.
 *
 * Model: the file's weights split evenly across layers (plus ~8% for embeddings and the output
 * head, which go to the GPU only when every layer does); offloaded layers sit in VRAM; for a MoE
 * model the expert share (~85%) of `moeCpuLayers` of those layers stays in RAM. KV cache memory per
 * token is layers x kv heads x head size x (bytes of K + bytes of V), split the same way as the
 * layers. Compute buffers grow with the micro-batch, and with context when flash attention is off.
 */

const GIB = 1024 ** 3;

/** Bytes per element of each KV cache type. */
export const KV_BYTES: Readonly<Record<KvCacheType, number>> = { f16: 2, q8_0: 1.0625, q4_0: 0.5625 };
const NON_LAYER_SHARE = 0.08;
const EXPERT_SHARE = 0.85;
/** Driver and runtime overhead on the GPU once anything is offloaded. */
const GPU_OVERHEAD_GIB = 0.3;
/** Headroom kept for Windows and other programs in system memory. */
const RAM_RESERVE = 0.2;

export type FitStatus = "fits" | "tight" | "over";

export interface MemoryUse {
  weights: number;
  kv: number;
  compute: number;
  /** Draft model and projector. */
  extra: number;
  total: number;
  capacity: number;
  status: FitStatus;
}

export interface FitEstimate {
  vram: MemoryUse;
  ram: MemoryUse;
  /** Worst of the two. */
  status: FitStatus;
}

const round2 = (n: number): number => Math.round(n * 100) / 100;

function status(total: number, capacity: number): FitStatus {
  if (!(capacity > 0) || total > capacity) return "over";
  return total > capacity * 0.9 ? "tight" : "fits";
}

function use(weights: number, kv: number, compute: number, extra: number, capacity: number): MemoryUse {
  const total = weights + kv + compute + extra;
  return { weights: round2(weights), kv: round2(kv), compute: round2(compute), extra: round2(extra), total: round2(total), capacity, status: status(total, capacity) };
}

export function estimateFit(
  model: Pick<LibraryModel, "sizeGiB" | "architecture" | "arch" | "projector">,
  p: LoadParams,
  hw: Pick<HardwareProfile, "gpu" | "ramTotalGiB">,
  draft?: Pick<LibraryModel, "sizeGiB"> | null,
): FitEstimate {
  const { layers, kvHeads, headDim, embedding } = model.arch;
  const vramCap = hw.gpu?.vramTotalGiB ?? 0;
  const ramCap = round2(hw.ramTotalGiB * (1 - RAM_RESERVE));
  const gpuLayers = hw.gpu ? Math.max(0, Math.min(layers, p.gpuLayers)) : 0;
  const perLayer = (model.sizeGiB * (1 - NON_LAYER_SHARE)) / layers;
  const nonLayer = model.sizeGiB * NON_LAYER_SHARE;
  const allOnGpu = gpuLayers === layers;

  const expertLayersOnCpu = model.architecture === "moe" ? Math.min(gpuLayers, Math.max(0, p.moeCpuLayers)) : 0;
  const gpuWeights = gpuLayers * perLayer - expertLayersOnCpu * perLayer * EXPERT_SHARE + (allOnGpu ? nonLayer : 0);
  const cpuWeights = Math.max(0, model.sizeGiB - gpuWeights);

  const kvPerTokenPerLayer = kvHeads * headDim * (KV_BYTES[p.kvCacheK] + KV_BYTES[p.kvCacheV]);
  const kvGpu = (gpuLayers * kvPerTokenPerLayer * p.contextLength) / GIB;
  const kvCpu = ((layers - gpuLayers) * kvPerTokenPerLayer * p.contextLength) / GIB;

  const attnScratch = p.flashAttention === "off" ? (p.ubatchSize * p.contextLength * kvHeads * 4 * 4) / GIB : 0;
  const compute = (p.ubatchSize * embedding * 4 * 6) / GIB + attnScratch;
  const gpuCompute = gpuLayers > 0 ? compute + GPU_OVERHEAD_GIB : 0;
  const cpuCompute = gpuLayers > 0 ? (p.ubatchSize * embedding * 4) / GIB : compute;

  const draftGiB = p.speculative === "draft" && draft ? draft.sizeGiB * 1.1 : 0;
  const projector = model.projector?.sizeGiB ?? 0;
  const gpuExtra = gpuLayers > 0 ? draftGiB + projector : 0;
  const cpuExtra = gpuLayers > 0 ? 0 : draftGiB + projector;

  // Locking memory with the file mapped pins every mapped page, so the whole file stays in RAM.
  // Otherwise only the weights left on the CPU occupy it.
  const ramWeights = p.mlock && p.mmap ? model.sizeGiB : cpuWeights;

  const vram = use(gpuWeights, kvGpu, gpuCompute, gpuExtra, vramCap);
  const ram = use(ramWeights, kvCpu, cpuCompute, cpuExtra, ramCap);
  const order: FitStatus[] = ["fits", "tight", "over"];
  return { vram, ram, status: order[Math.max(order.indexOf(vram.status), order.indexOf(ram.status))] ?? "over" };
}

export type FitClass = "vram" | "partial" | "too-large" | "unknown";

/** The library's fit column: whole model in VRAM, split across VRAM and RAM, or too large for both.
 *  `unknown` when this machine was not detected: no estimate is ever made for an invented machine. */
export function fitClass(model: Pick<LibraryModel, "sizeGiB">, hw: Pick<HardwareProfile, "gpu" | "ramTotalGiB"> | null): FitClass {
  if (!hw) return "unknown";
  const vram = hw.gpu?.vramTotalGiB ?? 0;
  if (model.sizeGiB + GPU_OVERHEAD_GIB + 0.5 <= vram) return "vram";
  if (model.sizeGiB <= vram + hw.ramTotalGiB * (1 - RAM_RESERVE)) return "partial";
  return "too-large";
}

export const FIT_LABELS: Readonly<Record<FitClass, string>> = {
  vram: "Fits in VRAM",
  partial: "Partial offload",
  "too-large": "Too large",
  unknown: "Unknown",
};
