import { describe, expect, it } from "vitest";
import { budgetFor, estimate, estimateFor, KV_BYTES, planFor, qualityTier, recommend, type MemoryModel } from "../../ui/src/lib/memory-estimate";
import { MOCK_HARDWARE } from "../../ui/src/mock/fixtures";
import type { HardwareProfile } from "../../ui/src/service/types";

/**
 * The memory estimate is arithmetic on real numbers: the file's size, the layer and attention
 * numbers from its GGUF header, and this machine's memory. These cases are worked out by hand from
 * those formulas so a change to any constant is caught here (docs/CLAIMS.md).
 *
 * Constants (ui/src/lib/memory-estimate.ts): weights split evenly across layers with 8% for the
 * embeddings and output head; a KV element is kvHeads x headDim x 2 x bytes(cacheType) per layer
 * per token; compute buffers are a 512-token micro-batch; 0.3 GiB of driver overhead once anything
 * is offloaded; a MoE file keeps 85% of every offloaded layer's weights (its experts) in RAM.
 */

/** Qwen3-0.6B as published: 28 blocks, 8 KV heads of 128, residual width 1024. */
const QWEN3_06B: MemoryModel = { architecture: "dense", layers: 28, kvHeads: 8, headDim: 128, embedding: 1024, contextMax: 40960 };
/** Qwen3-30B-A3B as published: 48 blocks, 4 KV heads of 128, residual width 2048. */
const QWEN3_30B_A3B: MemoryModel = { architecture: "moe", layers: 48, kvHeads: 4, headDim: 128, embedding: 2048, contextMax: 40960 };

/** The example laptop: 8 GiB card and 32 GiB of system memory, so 7 and 26 GiB usable. */
const LAPTOP: HardwareProfile = { ...MOCK_HARDWARE, gpu: { name: "Example GPU", vramTotalGiB: 8 }, ramTotalGiB: 32 };

describe("KV cache bytes per element", () => {
  it("uses the GGUF block sizes of the quantised cache types", () => {
    expect(KV_BYTES).toEqual({ f16: 2, q8_0: 1.0625, q4_0: 0.5625 });
  });
});

describe("Qwen3-0.6B Q8_0 at 4K", () => {
  const quant = { label: "Q8_0", sizeGiB: 0.61 };

  it("puts the whole file and its context in video memory when every layer is offloaded", () => {
    // Weights 0.61. Per layer the cache is 8 x 128 x 2 x 2 bytes x 4096 tokens = 16 MiB, so 28
    // layers cost 0.44 GiB. Compute buffers are 512 x 1024 x 4 x 6 bytes = 0.011 GiB, plus 0.3 GiB
    // of driver overhead. Weights + KV + scratch = 0.61 + 0.44 + 0.31 = 1.36 GiB.
    const e = estimate(QWEN3_06B, quant, 4096, "f16", { gpuLayers: 28, vramGiB: 12, ramGiB: 32 });
    expect(e).toEqual({ weightsGiB: 0.61, kvGiB: 0.44, scratchGiB: 0.31, vramGiB: 1.36, ramGiB: 0, fits: "gpu", note: "The whole file and its context fit in video memory." });
  });

  it("runs from system memory when there is no card to offload to", () => {
    // No offload: weights (0.61) + KV (0.44) + single-buffer compute (0.002) = 1.06 GiB of RAM, and
    // none of it in video memory.
    const e = estimate(QWEN3_06B, quant, 4096, "f16", { gpuLayers: 0, vramGiB: 0, ramGiB: 32 });
    expect(e).toEqual({ weightsGiB: 0.61, kvGiB: 0.44, scratchGiB: 0.01, vramGiB: 0, ramGiB: 1.06, fits: "cpu", note: "No room on the graphics card: it runs from system memory on the processor." });
  });

  it("doubles the cache with the context and shrinks it with the cache type", () => {
    const onGpu = { gpuLayers: 28, vramGiB: 12, ramGiB: 32 };
    expect(estimate(QWEN3_06B, quant, 8192, "f16", onGpu).kvGiB).toBe(0.88);
    // q4_0 is 0.5625 bytes per element, so a quarter of the f16 cache rounds to 0.12 GiB.
    expect(estimate(QWEN3_06B, quant, 4096, "q4_0", onGpu).kvGiB).toBe(0.12);
  });
});

describe("Qwen3-30B-A3B Q4_K_M at 32K", () => {
  const quant = { label: "Q4_K_M", sizeGiB: 18.6 };

  it("keeps the experts in system memory even when every layer's attention is on the card", () => {
    // Per layer 18.6 x 0.92 / 48 = 0.3565 GiB. At full offload 85% of that (the experts) stays in
    // RAM: 14.55 GiB. The attention and shared weights (4.05 GiB) plus the embeddings (1.49 GiB)
    // and the cache (48 x 4 x 128 x 2 x 2 x 32768 = 3.0 GiB) and scratch (0.32) need 7.38 GiB of
    // video memory. Both memories are used, so it is a split fit, never a whole-file one.
    const e = estimate(QWEN3_30B_A3B, quant, 32768, "f16", { gpuLayers: 48, vramGiB: 24, ramGiB: 64 });
    expect(e).toEqual({
      weightsGiB: 18.6,
      kvGiB: 3,
      scratchGiB: 0.32,
      vramGiB: 7.38,
      ramGiB: 14.55,
      fits: "split",
      note: "Attention and shared weights on the card; experts stay in system memory.",
    });
  });

  it("moves the attention and half the cache off the card as layers go back to the processor", () => {
    const e = estimate(QWEN3_30B_A3B, quant, 32768, "f16", { gpuLayers: 24, vramGiB: 6, ramGiB: 64 });
    expect(e.vramGiB).toBe(3.11);
    expect(e.ramGiB).toBe(18.82);
    expect(e.fits).toBe("split");
  });

  it("runs on the processor, and calls it impossible, when there is nowhere else to put it", () => {
    const cpu = estimate(QWEN3_30B_A3B, quant, 32768, "f16", { gpuLayers: 0, vramGiB: 0, ramGiB: 64 });
    expect(cpu).toMatchObject({ vramGiB: 0, ramGiB: 21.62, fits: "cpu", note: "No room on the graphics card: it runs from system memory on the processor." });
    // Card and system memory together are too small: the experts alone (14.55 GiB) exceed 6 GiB of RAM.
    const no = estimate(QWEN3_30B_A3B, quant, 32768, "f16", { gpuLayers: 48, vramGiB: 24, ramGiB: 6 });
    expect(no.fits).toBe("no");
    expect(no.note).toBe("Needs more video and system memory together than this machine can spare.");
  });
});

describe("usable memory, the plan, and the estimate for a machine", () => {
  it("keeps the same headroom the setup screen does", () => {
    expect(budgetFor(LAPTOP)).toEqual({ vramGiB: 7, ramGiB: 26 });
    expect(budgetFor(undefined)).toEqual({ vramGiB: 0, ramGiB: 0 });
  });

  it("offloads the most layers that still fit, or none without a card", () => {
    const plan = planFor(QWEN3_06B, { label: "Q8_0", sizeGiB: 0.61 }, 4096, "f16", budgetFor(LAPTOP));
    expect(plan).toEqual({ gpuLayers: 28, vramGiB: 7, ramGiB: 26 });
    // A mix of a card that cannot hold the whole file and one that cannot be used at all.
    const partial = planFor(QWEN3_30B_A3B, { label: "Q4_K_M", sizeGiB: 18.6 }, 32768, "f16", budgetFor(LAPTOP));
    expect(partial.gpuLayers).toBeGreaterThan(0);
    expect(partial.gpuLayers).toBeLessThan(48);
    expect(planFor(QWEN3_06B, { label: "Q8_0", sizeGiB: 0.61 }, 4096, "f16", { vramGiB: 0, ramGiB: 26 })).toMatchObject({ gpuLayers: 0 });
  });

  it("estimates with the plan this machine would choose", () => {
    expect(estimateFor(QWEN3_06B, { label: "Q8_0", sizeGiB: 0.61 }, 4096, "f16", LAPTOP)).toMatchObject({ vramGiB: 1.36, ramGiB: 0, fits: "gpu" });
    // 47 layers fit the laptop's 7 GiB of usable video memory as a split with the experts in RAM.
    expect(estimateFor(QWEN3_30B_A3B, { label: "Q4_K_M", sizeGiB: 18.6 }, 32768, "f16", LAPTOP)).toMatchObject({ vramGiB: 5.77, ramGiB: 16.15, fits: "split" });
    // No card: run from system memory. No reading at all: nothing fits, so nothing is claimed.
    expect(estimateFor(QWEN3_06B, { label: "Q8_0", sizeGiB: 0.61 }, 4096, "f16", { ...LAPTOP, gpu: null })).toMatchObject({ vramGiB: 0, fits: "cpu" });
    expect(estimateFor(QWEN3_06B, { label: "Q8_0", sizeGiB: 0.61 }, 4096, "f16", undefined).fits).toBe("no");
  });
});

describe("quantisation tier and the best files for this machine", () => {
  const small = QWEN3_06B;

  it("prefers Unsloth Dynamic, then the plain K-quants people know", () => {
    expect(qualityTier("UD-Q4_K_XL")).toBe(0);
    expect(qualityTier("Q4_K_M")).toBe(1);
    expect(qualityTier("Q5_K_M")).toBe(2);
    expect(qualityTier("Q8_0")).toBe(3);
    expect(qualityTier("IQ4_XS")).toBe(4);
    expect(qualityTier("EXL3 4.0bpw")).toBe(4);
    expect(qualityTier("ud-q4_k_xl")).toBe(0);
  });

  it("ranks whole-in-video-memory first, then tier, then the larger file, and keeps what would not load last", () => {
    const candidates = [
      { model: small, quant: { label: "Q4_K_M", sizeGiB: 40 }, tag: "too-big" },
      { model: small, quant: { label: "Q8_0", sizeGiB: 4 }, tag: "q8" },
      { model: small, quant: { label: "Q5_K_M", sizeGiB: 2.5 }, tag: "q5" },
      { model: small, quant: { label: "UD-Q4_K_XL", sizeGiB: 20 }, tag: "split-ud" },
      { model: small, quant: { label: "UD-Q4_K_XL", sizeGiB: 2 }, tag: "ud" },
    ];
    const picks = recommend(candidates, LAPTOP, 4096, "f16", 5);
    // Every 4 GiB-or-smaller file fits in video memory; among those the tier decides. The 20 GiB file
    // only fits across both memories, so it waits behind them, and the 40 GiB one cannot load at all.
    expect(picks.map((p) => p.tag)).toEqual(["ud", "q5", "q8", "split-ud", "too-big"]);
    expect(picks.map((p) => p.rank)).toEqual([1, 2, 3, 4, 5]);
    expect(picks.map((p) => p.estimate.fits)).toEqual(["gpu", "gpu", "gpu", "split", "no"]);
    expect(picks[0]!.why).toBe("UD-Q4_K_XL is the closest fit entirely in video memory (2.0 GiB).");
    expect(picks[4]!.why).toBe("Q4_K_M needs more memory than this machine can spare (40.0 GiB).");
    expect(recommend(candidates, LAPTOP, 4096, "f16", 3)).toHaveLength(3);
  });

  it("breaks a tier tie with the larger, more capable file", () => {
    const bigger = { model: small, quant: { label: "UD-Q4_K_XL", sizeGiB: 3 }, tag: "3" };
    const smaller = { model: small, quant: { label: "UD-Q4_K_XL", sizeGiB: 2 }, tag: "2" };
    expect(recommend([smaller, bigger], LAPTOP, 4096, "f16", 2).map((p) => p.tag)).toEqual(["3", "2"]);
  });

  it("carries a caller's own fields through and is deterministic", () => {
    const withExtra = [{ model: QWEN3_30B_A3B, quant: { label: "Q4_K_M", sizeGiB: 18.6 }, display: "Qwen3-30B-A3B" }];
    const picks = recommend(withExtra, LAPTOP, 32768, "f16", 1);
    expect(picks[0]).toMatchObject({ display: "Qwen3-30B-A3B", rank: 1 });
    expect(picks[0]!.why).toBe("Q4_K_M runs across video and system memory (18.6 GiB); Cerebro measures the speed after download.");
    expect(recommend(withExtra, LAPTOP, 32768, "f16", 1)).toEqual(picks);
    expect(recommend([], LAPTOP, 4096, "f16", 3)).toEqual([]);
  });
});

describe("Qwen3.5/3.8 hybrids", () => {
  // Qwen3.8-27B: 65 blocks, one in four keeps a KV cache (16 attention layers), 4 KV heads of 256.
  const QWEN38_27B: MemoryModel = { architecture: "dense", layers: 65, kvHeads: 4, headDim: 256, embedding: 5120, contextMax: 262144, attentionLayers: 16 };
  it("count KV only for the attention layers", () => {
    const quant = { label: "Q4_K_M", sizeGiB: 15.66 };
    const plan = { gpuLayers: 0, vramGiB: 0, ramGiB: 32 };
    const hybrid = estimate(QWEN38_27B, quant, 32768, "f16", plan);
    const dense = estimate({ ...QWEN38_27B, attentionLayers: undefined }, quant, 32768, "f16", plan);
    // 16 layers x 4 heads x 256 x 2 x 32768 x 2 bytes = 2 GiB, against 65 layers' 8.125 GiB.
    expect(hybrid.kvGiB).toBeCloseTo(2, 1);
    expect(dense.kvGiB).toBeCloseTo(8.13, 1);
  });
});
