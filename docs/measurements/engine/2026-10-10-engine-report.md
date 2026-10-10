# Cerebro Engine: measured results, 2026-10-10

Every number below was measured on the machine described, with the commit listed. Raw rows (one
JSON line per configuration, with commit, settings and cache statistics) are in `runs.jsonl` next to
this file. The control is the official llama.cpp release build 10603 (`c060ca974`, CUDA), run on the
same machine on the same day with `llama-bench`. It is a yardstick only; the engine contains no
llama.cpp code.

## Machine A

- CPU: Intel i9-12900H (6 performance + 8 efficiency cores, 20 threads), Windows 11
- RAM: 32 GB DDR5 configured at 4800 MT/s (about 30 of it usable; other applications left 12-17 GB free)
- GPU: NVIDIA RTX 3070 Ti Laptop, 8 GB, 448 GB/s, power-limited (80/115 W)

## Method

- `examples/bench.rs`: for each prompt length, one warm-up run, then 3 timed runs (median reported)
  of prefill and greedy decode. The same prompt text is used for every engine.
- Correctness: the GPU's greedy tokens are compared with the CPU reference (`greedy_prefix_match`).
  For mixture-of-experts models, a self-test compares the GPU expert kernels with the CPU on real
  weights (maximum relative error 4e-7).
- "First turn" numbers: a fresh process per prompt, the expert profile reset to the same starting
  file, one prompt per run (no repetition that the expert cache could learn from).

## Results

### Qwen3-0.6B Q8_0 (fits in VRAM)

| | Cerebro | llama.cpp |
|---|---|---|
| GPU decode, short context | **344 tok/s** | 293 (tg128) |
| GPU decode, 600-token context | 236 tok/s | not measured |
| GPU prefill, 512 / 2048 tokens | 1,317 / 1,228 tok/s | **12,077 / 15,896** |
| CPU decode, short context | 62 tok/s | **70** |

### Qwen3-Coder-30B-A3B Q4_K_M (17.3 GB on an 8 GB GPU)

| | Cerebro | llama.cpp (`-ngl 99 -ncmoe 32`) |
|---|---|---|
| Decode after the expert cache adapted to the content (512 / 2048 context) | **38.6 / 55.3 tok/s** (VRAM hit rate 75% / 85%) | 33.4-35.8 |
| Decode, first turn, short novel prompt | 19-23 tok/s (hit rate 31-48%) | 33.4-35.8 |
| Prefill, 512 / 2048 tokens | 165-205 / 200 tok/s | 198 / **295** |
| CPU only, decode | 13.8 tok/s | **18.5** (`-ngl 0`) |

### What moved the numbers (same machine, same model)

| Change | Before | After |
|---|---|---|
| GPU: 10 fused kernels per layer, CUDA graph replay, greedy pick on the GPU (0.6B decode) | 168 | 346 tok/s |
| GPU: batched prefill, f16 KV cache, split-K decode attention (0.6B, 2K context) | 135 / 83 | 884 / 226 tok/s (prefill / decode) |
| GPU: tiled prefill attention (0.6B prefill, 2K) | 884 | 1,228 tok/s |
| MoE hybrid: asynchronous copies, on-GPU slot lookup | 13.5 | 17.4 tok/s |
| CPU worker pool that yields instead of spinning (16 threads) | 1.9 | 11.4 tok/s |
| Touching CPU-side expert pages at load instead of faulting during decode | 30.3 | 36.2 tok/s |
| MoE: attention/output weights kept in K-quant format on the GPU (frees VRAM for experts) | 1,467 | 1,765 experts cached |
| MoE prefill: batched by expert, then uncached experts streamed to the GPU, 512-token chunks | 33 | 205 tok/s |
| Prompt-aware expert placement (2048-token prompt) | 23.7 (hit 41%) | 55.3 tok/s (hit 85%) |
| CPU: parallel router, GQA-shared attention, fast exp/round, RoPE table (30B-A3B CPU) | 9.9 | 13.8 tok/s |

## Reading the results honestly

- **Where Cerebro is ahead:** GPU decode of a model that fits (344 vs 293), and MoE decode once the
  expert cache has adapted to the conversation (38.6-55.3 vs 33-36). The adaptive VRAM expert cache
  is the reason: llama.cpp pins whole layers of experts to the GPU, Cerebro moves the experts the
  conversation actually uses.
- **Where it is behind:** GPU prefill of dense models (about 10x: llama.cpp uses tensor-core matrix
  multiplication, Cerebro's prefill kernels use dp4a); MoE first-turn decode before the cache has
  adapted (19-23 vs 33-36); CPU-only decode (13.8 vs 18.5).
- **Unexplained gap under investigation:** the CPU expert kernels reach 37-41 GB/s in isolation but
  26-28 GB/s inside a decode step.

## Next, in order

1. First-turn MoE decode: asynchronous expert swaps on the copy stream (many per token instead of 16
   per 32 tokens), and a shipped starting profile per model built from mixed prompts.
2. Tensor-core (mma) prefill kernels for dense models.
3. Dense models larger than VRAM: layer split between GPU and CPU, then speculative decoding with a
   small draft model (batched verification makes offloaded weights cost once per several tokens).
4. CPU decode: interleaved weight layout for the K-quant GEMV, closing the in-step bandwidth gap.
5. The models the product lists: Qwen3.8 (hybrid attention), Gemma 4, Qwen3.8-Flash-Next 125B MoE.
6. The same benchmark matrix on machine B (Ryzen 7 7840HS, 780M, 32 GB DDR5-5600), standing in
   for users without an NVIDIA GPU.
