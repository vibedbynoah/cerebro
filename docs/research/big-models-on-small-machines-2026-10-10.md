# Running big models fast on machines too small for them (research, 2026-10-10)

Goal (owner): people who cannot run large, smart models today, or only at 0.01 tok/s, get real speed.
All figures below are the papers' own; each technique is implemented and then measured on our
machines before any claim.

## What the literature shows

| Technique | Core idea | Reported result | Source |
|---|---|---|---|
| SpecExec | Offloaded target model verifies a large draft *tree* in one pass; with offloading, hundreds of tokens cost about the same as one | 70B on consumer GPUs at 4-6 tok/s (4-bit) / 2-3 (16-bit); up to 10.6x / 18.7x over autoregressive offloading; up to 20 tokens per target pass | arXiv 2406.02532 (main table measured on A100; check per-GPU tables) |
| Fiddler | MoE: keep non-expert layers + popular experts on GPU, compute the other experts *on the CPU* (move activations, not weights); per-layer latency model picks the device | Mixtral-8x7B FP16 (>90 GB) at >3 tok/s on one 24 GB GPU, order of magnitude over weight-shuttling | arXiv 2402.07033 (ICLR 2025) |
| PowerInfer | Dense: hot neurons resident on GPU, cold neurons on CPU, activation predictors + sparse operators | 13.2 tok/s average, 29 peak on one RTX 4090; up to 11.7x vs llama.cpp | arXiv 2312.12456 |
| LLM in a Flash | Bigger-than-DRAM: load only needed rows from flash, windowing (reuse recently active neurons), row-column bundling for large contiguous reads | Models up to 2x DRAM; 4-5x (CPU) and 20-25x (GPU) over naive loading | arXiv 2312.11514 |
| HOBBIT | Expert offloading with mixed precision: low-precision copy for cache-miss experts, prefetch up to 4 layers ahead | Up to 9.93x decode over prior offloading | arXiv 2411.01433 |
| ProMoE | Per-layer learned predictor ranks experts to prefetch | 84.7% prediction accuracy; 2.07x average decode | arXiv 2410.22134 |
| Fate | Adjacent layers' gate inputs predict next experts | ~99% expert hit rate | arXiv 2502.12224 |
| OD-MoE | On-demand expert loading, no cache | 95-99.9% recall | arXiv 2512.03927 |

## Cerebro Engine plan, in order

1. **Native 4-bit kernels** (Q4_K, Q5_K, Q6_K, then IQ/UD formats) on CPU (AVX2, int8 activations)
   and GPU (dp4a), so weights are read at their stored size. Measured on Qwen3-0.6B Q4_K_M first.
2. **MoE graph** (qwen3moe: router, top-k, normalised weights, stacked expert tensors). Test model on
   disk: Qwen3-Coder-30B-A3B Q4_K_M (17.3 GB) on the 8 GB RTX 3070 Ti laptop. Baseline to beat: our
   own earlier measurement of 29 tok/s with llama.cpp `-ncmoe 32` on this machine.
3. **Fiddler-style placement**: dense layers + KV + the most-used experts in VRAM (frequency profile,
   adaptive), remaining experts computed on the CPU in parallel with the GPU; per-layer device choice
   by a measured latency model. Telemetry: active experts, hit rate, CPU/GPU split.
4. **Expert prediction and prefetch** (cross-layer gate inputs, Fate/HOBBIT): prefetch next-layer
   experts during attention; low-precision fallback copies for misses.
5. **SpecExec-style tree speculation** with a small draft model on the GPU: the batched verify path
   (already built for prefill) checks a token tree per pass of the offloaded target.
6. **SSD streaming** (LLM-in-a-Flash): models larger than RAM, memory-mapped with large contiguous
   reads, windowing and row-column bundling of expert/neuron weights.
7. **Placement matrix benchmark** on both machines (VRAM / RAM / SSD and splits), logged to
   docs/measurements/engine/runs.jsonl with commit, machine and settings.
