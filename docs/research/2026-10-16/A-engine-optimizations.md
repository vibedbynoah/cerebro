# Engine performance catalogue (research, 2026-10-16)

Internal working catalogue of optimisation candidates for `cerebro-engine`. Every entry is a candidate,
not a claim: nothing here is shipped until it is measured on the owner's machines (docs/CLAIMS.md).
Sources are arXiv ids / URLs; anything not confirmed against a primary source is marked **UNVERIFIED**.

## Legend and method

- **Target hardware.** Machine A: i9-12900H (6 P + 8 E, AVX2, AVX-VNNI, no AVX-512, no AMX), 32 GB
  DDR5-4800, RTX 3070 Ti Laptop 8 GB, **sm_86** (Ampere), 448 GB/s, 80/115 W. Machine B: Ryzen 7 7840HS
  + Radeon 780M (RDNA3, UMA), 32 GB. Consequences used throughout: sm_86 has **mma.sync** (int8
  m16n8k32, fp16/bf16 m16n8k16), **cp.async**, `ldmatrix`, 2:4 sparse mma, **but no wgmma, no TMA,
  no FP8 tensor cores (sm_89+), no FP4 (sm_100+), no Programmatic Dependent Launch (sm_90+)**. Anything
  requiring those is flagged unavailable/Experimental.
- **Expected gain** is an estimate for Machine A on the model classes the engine already runs (Qwen3
  dense, Qwen3-Coder-30B-A3B MoE Q4_K_M), with a confidence H/M/L. Baseline numbers are from
  `docs/measurements/engine/2026-10-10-engine-report.md`.
- **The four measured gaps this catalogue must attack** (report, "Reading the results honestly" and
  "Next, in order"):
  - **G-A** dense GPU prefill ~10x slower than llama.cpp (1317/1228 vs 12077/15896 tok/s on 0.6B).
  - **G-B** MoE first-turn decode 19-23 vs 33-36 tok/s before the expert cache adapts.
  - **G-C** CPU-only decode 13.8 vs 18.5 tok/s.
  - **G-D** CPU expert kernels 26-28 GB/s in-step vs 37-41 GB/s isolated.
- **Files touched** use the real crate layout (`crates/cerebro-engine/src/`): `kernels.cu`, `gpu.rs`,
  `quant.rs`, `quant_more.rs`, `model.rs`, `pool.rs`, `gguf.rs`, `hw.rs`, `iq_tables.rs`, `sample.rs`,
  `chat.rs`, `tokenizer.rs`; plus a new `autotune`/`cache` module where noted.
- **Channels** (docs/specs/turbo-system.md): **Stable** = safe, small, no quality risk; **Beta** = measured,
  needs tuning, gated on hardware; **Experimental** = immature or quality risk, opt-in.

Item count in this document: **186** numbered optimisations (items 1-186).

---

## A. GPU kernels — prefill, mma, layouts, attention, launch (items 1-26)

### 1. int8 tensor-core (mma.sync m16n8k32) for K-quants and IQ via dequant-to-int8
- **What:** For prefill GEMM, dequantise quantised weights to int8 in registers (per 32/256-value block),
  quantise activations to int8 per-token, and run `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`
  (Ampere int8 tensor core, 2x fp16 throughput). Accumulate int32, apply block scales in the epilogue.
- **Why / gain:** Directly attacks **G-A**. The report attributes the prefill gap to dp4a prefill ("llama.cpp
  uses tensor-core matrix multiplication"). An int8 mma path is the single highest-value GPU change; llama.cpp
  itself gained large prefill speedups moving prefill to mma. Expect **3-7x prefill** on dense models
  (1317→~4000-8000 tok/s on 0.6B, closer to the 12077 yardstick), confidence **H**.
- **Impl:** New kernels in `kernels.cu`: `qmatmul_i8_mma` (per-tensor-block, tile `BMxBN=128x128`, K-step
  32) plus a new offline repack producing the mma fragment layout (see item 3). Dispatch from
  `matmul_q8_batch`/`kmatmul` in `gpu.rs` based on `DType` and a `prefill_batch` threshold. Requires an
  activation-quant kernel variant that emits int8 + per-32 scale (extend `quantize_q8`).
- **Risk:** int8 activation quant adds small quality loss (~0.5-1% KL); needs a q8 path for activations.
  Offline repack changes the on-disk/in-RAM layout, so the loader must detect and cache repacked copies.
- **Channel:** Beta.
- **Toggle:** "Use graphics-card tensor cores for prompt processing so long prompts are read several times
  faster (tiny accuracy cost)."
- **Source:** Ampere mma PTX ISA (CUDA docs); llama.cpp MMQ kernels (github.com/ggml-org/llama.cpp);
  MARLIN arXiv 2408.11743.

### 2. fp16/bf16 mma.sync (m16n8k16) with in-kernel f16 dequant (lossless path)
- **What:** Dequantise K-quants/IQ to fp16 (not int8) in registers and use fp16 tensor cores with fp32
  accumulators. This is the accuracy-preserving alternative to item 1.
- **Why / gain:** Keeps exact-ish dequant (matches the current CPU/GPU reference within tolerance),
  fixes **G-A** without activation quantisation. Slightly lower peak than int8 but no quality cost. Expect
  **2.5-5x prefill**; confidence **H**. Good default before int8 ships.
- **Impl:** `qmatmul_f16_mma` in `kernels.cu`, f16 repack in the loader; `.h` attention already uses fp16,
  reuse for K/V mma. Dispatcher in `gpu.rs`. Reference test against f32 in `model.rs` tests.
- **Risk:** fp16 dequant materialises f16 weights in smem/regs, so smem pressure limits tile size.
- **Channel:** Beta.
- **Toggle:** "Process long prompts on the GPU's math units with no change to answer quality."
- **Source:** CUTLASS docs (github.com/NVIDIA/cutlass); FlashAttention-2 arXiv 2307.08691.

### 3. Marlin-style offline weight pre-shuffle into mma fragment layout
- **What:** Repack 4/5/6-bit weights at load time into the exact register/shared layout `ldmatrix` and mma
  expect (16x64 tiles, interleaved nibble order), so the inner loop does zero shuffling.
- **Why / gain:** Removes the smem-bandwidth and instruction overhead that caps naive mma kernels; this is
  the core of why Marlin reaches near-peak. Enables items 1/2 to hit their numbers rather than stall on
  smem. Enabling change for **G-A**; confidence **H**.
- **Impl:** `repack.rs` (new) writing a cached `.cerebro-repack` next to the GGUF (or an in-RAM shadow);
  `quant.rs` gains fragment-layout constants; `gpu.rs` uploads repacked bytes.
- **Risk:** Extra disk/RAM equal to weights; cache invalidation on format change; correctness must be
  bit-checked against the reference dequant.
- **Channel:** Beta.
- **Toggle:** "Reorganise model weights once at load so the GPU reads them at full speed."
- **Source:** MARLIN arXiv 2408.11743; Machete pre-shuffle (Red Hat Developer blog, 2024-10-14;
  vllm-project/vllm PR #7174).

### 4. Use `ldmatrix` for all smem→register weight/activation loads
- **What:** Replace scalar/`int4` smem reads with `ldmatrix` (Ampere) so a warp loads 8x8 matrices in one
  instruction, matching mma operand layout.
- **Why / gain:** Shared-memory instruction count drops ~4-8x in the inner loop; prerequisite for items
  1-3 to stay compute-bound. Contributes to **G-A**; confidence **H**.
- **Impl:** inside `qmatmul_*_mma` and `attn_prefill_tiled` in `kernels.cu`.
- **Risk:** requires the swizzled smem layout (item 20) or bank conflicts negate the gain.
- **Channel:** Beta.
- **Toggle:** (no separate toggle; part of "tensor-core prefill").
- **Source:** CUDA PTX ISA `ldmatrix`; FlashAttention-2 2307.08691.

### 5. `cp.async` global→shared pipeline (double/triple buffer)
- **What:** Ampere asynchronous copies from gmem to smem that bypass registers, with 2-3 stage software
  pipelining of the K-dimension.
- **Why / gain:** Hides weight-load latency behind mma; essential for any mma kernel to reach peak. Enables
  **G-A**; confidence **H**.
- **Impl:** `__pipeline_memcpy_async` / `cp.async.cg.shared.global` with 16-byte alignment in new mma
  kernels; stage count autotuned (item 179).
- **Risk:** more smem per stage reduces occupancy; needs autotune.
- **Channel:** Beta.
- **Toggle:** (part of "tensor-core prefill").
- **Source:** NVIDIA CUDA C++ Programming Guide (async copy); MARLIN 2408.11743.

### 6. Stream-K work decomposition for prefill GEMM
- **What:** Instead of fixed thread-block tiling, decompose the K-loop across CTAs so every SM gets equal
  work (Stream-K); combine partial products with a deterministic reduction.
- **Why / gain:** Prefill matrices are often "tall-skinny" or wave-quantised (small vocab/FFN dims on a
  3070 Ti with 48 SMs); Stream-K removes the tail wave. 10-30% on top of a good mma kernel; confidence M.
- **Impl:** a scheduler template in `kernels.cu` (new `qmatmul_streamk`); reduction via fixed-order
  atomics or a workspace + second pass. Needs a small on-device work queue.
- **Risk:** nontrivial correctness (reduction order, atomics); can be slower than tiled for large grids.
  Gate on shape.
- **Channel:** Experimental.
- **Toggle:** "Spread big matrix multiplications evenly across the GPU for faster prompts."
- **Source:** NVIDIA CUTLASS Stream-K (github.com/NVIDIA/cutlass); stream-K described in CUTLASS docs
  (arXiv id **UNVERIFIED**).

### 7. Split-K with parallel reduction for prefill/decode GEMM
- **What:** Split the K dimension across blocks, accumulate partials, reduce.
- **Why / gain:** Helps when one matrix dim is small (e.g. decode GEMM, or attention output projection).
  Modest 5-20% in specific shapes; confidence M. Complements Stream-K.
- **Impl:** `qmatmul_splitk` + workspace reduction; already present in spirit for decode attention
  (`attn_partial`/`attn_combine`), generalise to GEMM.
- **Risk:** extra memory traffic; only wins for small grids.
- **Channel:** Beta.
- **Toggle:** (part of GEMM autotune).
- **Source:** CUTLASS docs; FlashAttention split-KV (item 15).

### 8. Fused epilogue: scale + bias + SwiGLU + re-quant in one kernel
- **What:** Fuse the mma epilogue (dequant scale, optional bias, activation) and the next op's input
  quantisation so intermediates never round-trip to gmem.
- **Why / gain:** Removes gmem traffic + kernel launches between GEMM and activation; 5-15% end-to-end on
  prefill, more on decode. Confidence M-H.
- **Impl:** extend `silu_mul_quant_batch` to consume mma accumulators; new `gemm_epilogue_silu_quant`.
- **Risk:** couples graph scheduling; harder to keep correct across quant formats.
- **Channel:** Beta.
- **Toggle:** "Chain GPU steps together so intermediate results don't go back to memory."
- **Source:** FlashAttention/cuBLASLt fused epilogues; CUTLASS.

### 9. Grouped GEMM (one launch, per-expert segments) for MoE prefill
- **What:** A single grouped-GEMM kernel processes all activated experts' segments (variable M per expert),
  tokens pre-sorted by expert.
- **Why / gain:** Replaces per-expert launch overhead and tiny-matmul inefficiency; MoE prefill is already
  the strongest area (205 tok/s) but this plus mma pushes further. 20-50% MoE prefill; confidence M.
- **Impl:** `moe_grouped_gemm` in `kernels.cu`; token sort/scatter added to `moe_cpu_batch` GPU mirror;
  offsets array device-side.
- **Risk:** sort overhead for small batches; needs fallback for few tokens.
- **Channel:** Beta.
- **Toggle:** "Run all the experts a prompt needs in one batched GPU pass."
- **Source:** Megablocks (arXiv 2211.15841); CUTLASS grouped GEMM.

### 10. Expert-sorted gather/scatter with on-GPU token bucketing
- **What:** Sort tokens by routed expert on the GPU (or reuse the router's sort) before grouped GEMM and
  scatter results back.
- **Why / gain:** Enables item 9; reduces redundant weight reads. Confidence M.
- **Impl:** extend `router_topk_batch` + new `bucket_scatter`; reuse `scatter_add`.
- **Risk:** extra small kernels; negligible for decode (1 token).
- **Channel:** Beta.
- **Toggle:** (part of MoE prefill batching).
- **Source:** Megablocks 2211.15841.

### 11. FlashAttention-2-style prefill attention (Ampere-safe)
- **What:** Adopt FA2's work partitioning (one CTA per (head, query-tile), K/V loop, online softmax) without
  Hopper-only features. Current `attn_prefill_tiled` is a simpler version.
- **Why / gain:** Better smem/occupancy tradeoff → 1.3-2x attention prefill → directly helps **G-A** for
  long prompts. Confidence M-H.
- **Impl:** rewrite `attn_prefill_tiled` in `kernels.cu` with FA2 tiling (Br=64/128, Bc=64), fp16 in,
  fp32 accum. Keep a reference test.
- **Risk:** masking/edge cases for GQA + per-head QK norm.
- **Channel:** Beta.
- **Toggle:** "Faster attention during prompt reading; identical output."
- **Source:** FlashAttention-2 arXiv 2307.08691.

### 12. GQA head-packing to avoid duplicating K/V
- **What:** Index Q heads onto shared KV heads in smem so K/V tiles are loaded/shared once across the query
  head group (Qwen3 GQA).
- **Why / gain:** Qwen3 proves GQA; avoid reloading K/V per Q head → 10-30% attention prefill/decode.
  Confidence M-H.
- **Impl:** adjust tensor indexing in `attn_prefill_tiled`/`attn_partial` (FA2 approach).
- **Risk:** scheduling complexity.
- **Channel:** Beta.
- **Toggle:** (part of attention kernels).
- **Source:** GQA arXiv 2305.13245; FA3 GQA packing (nvidia.com GTC25 S71368).

### 13. Variable-length (varlen) / packed attention for batched prefill
- **What:** Process several prompts/sequences of different lengths in one attention kernel with per-sequence
  offsets, no padding.
- **Why / gain:** Needed for continuous batching + chunked prefill without padding waste. Confidence M.
- **Impl:** add `cu_seqlens` to attention kernels; `gpu.rs` passes per-request offsets.
- **Risk:** index math bugs; keep scalar reference.
- **Channel:** Beta.
- **Toggle:** (part of batching).
- **Source:** FlashAttention varlen; FlashInfer arXiv 2501.01005 (**UNVERIFIED id**).

### 14. Flash-decoding (split-KV) for single-token decode
- **What:** Split the KV sequence across blocks, each computes partial attention, combine (log-sum-exp
  merge). Engine already has `attn_partial`/`attn_combine` — generalise and tune.
- **Why / gain:** Decode attention is latency-bound at long context; 1.5-3x attention decode at 2K-32K.
  Confidence M-H.
- **Impl:** tune splits by context length; `attn_partial`/`attn_combine` in `kernels.cu`.
- **Risk:** more kernels/launch overhead at short context; gate on seq_len.
- **Channel:** Beta.
- **Toggle:** "Keep decoding fast as the conversation gets long."
- **Source:** Flash-Decoding (crfm.stanford.edu/2023/10/12/flashdecoding.html); FA repo.

### 15. Paged-KV attention kernel
- **What:** Attention reads K/V from fixed-size pages via a page table (the design already promised in
  engine.md §1) instead of contiguous blocks.
- **Why / gain:** Enables shared prefix caching + memory defrag without copies; reduces memory waste from
  over-allocation → more experts/KV in VRAM. Confidence M.
- **Impl:** page table on device; `attn_*` index through it; `paged_kv` in `gpu.rs`.
- **Risk:** indirect addressing overhead ~5-10% at short context.
- **Channel:** Beta.
- **Toggle:** "Share repeated prompt history between chats without copying memory."
- **Source:** PagedAttention arXiv 2309.06180.

### 16. In-kernel RoPE fused with KV write and KV quant
- **What:** Apply rotary embedding to Q/K as part of the write path (and quantise K pre/post RoPE per
  item 32) rather than separate passes.
- **Why / gain:** Removes a full pass over Q/K; small but free end-to-end win; enables pre-RoPE K quant.
  Confidence H.
- **Impl:** fold into `qkv_post` and the KV store path in `kernels.cu`.
- **Risk:** order-of-operations must match reference.
- **Channel:** Stable.
- **Toggle:** (internal; no user toggle).
- **Source:** RoFormer arXiv 2104.09864; FlashAttention FA3 fused RoPE.

### 17. FP32 accumulate everywhere; fp16 storage only where measured safe
- **What:** Keep accumulators fp32; choose fp16 KV only when the acceptance/logits test passes.
- **Why / gain:** Prevents silent quality regressions while enabling speed. Confidence H (safety item).
- **Impl:** already true in kernels; keep as a rule + test in `model.rs`.
- **Risk:** none (guard rail).
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** engine.md rules (correctness first).

### 18. Persistent-CTA kernel scheduling
- **What:** Launch a grid sized to SM count; each CTA pulls work from a device queue.
- **Why / gain:** Amortises launch/tail latency for many small decode kernels; helps decode gaps. 5-15%;
  confidence L-M.
- **Impl:** device queue + `qmatmul_persistent`; reuse for all decode GEMMs.
- **Risk:** complexity, weak for large prefill.
- **Channel:** Experimental.
- **Toggle:** "Run many small GPU steps with less overhead." 
- **Source:** persistent kernels (CUTLASS Stream-K discussion) — **UNVERIFIED** general technique.

### 19. Shared-memory swizzle to eliminate bank conflicts
- **What:** XOR-swizzled smem layouts for weight/activation tiles.
- **Why / gain:** Required for items 4/5 to pay off; 10-30% in the inner loop. Confidence M.
- **Impl:** swizzle helper in `kernels.cu`; index math shared with repack (`item 3`).
- **Risk:** indexing bugs; test matrices.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** CUTLASS layouts; CUDA docs.

### 20. 128-bit vectorised global loads for weights and activations
- **What:** Ensure every gmem access is 16 bytes/thread where alignment permits.
- **Why / gain:** Peak memory bandwidth is the ceiling for decode; ~5-10%. Confidence M.
- **Impl:** audit `kernels.cu` loads (many already use `int4`); add alignment asserts in `Gguf`.
- **Risk:** misalignment for odd rows; fallback path.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** CUDA best practices; MARLIN 2408.11743.

### 21. Asynchronous expert/KV streaming overlap within attention (chard-style)
- **What:** Issue `cp.async` (global→shared) copies of the next layer's experts during the current layer's
  attention math, on the same stream or a copy stream.
- **Why / gain:** Directly attacks **G-B** (first-turn decode) and improves MoE prefill. Overlaps PCIe with
  compute. Confidence M-H.
- **Impl:** split `moe` step into prefetch (copy stream) + compute; extend the existing "asynchronous copies"
  in `gpu.rs` step path.
- **Risk:** PCIe contention; must bound in-flight bytes.
- **Channel:** Beta.
- **Toggle:** "Fetch the experts the next layer needs while the GPU is still busy."
- **Source:** engine-report "chunked prefill with expert prefetch"; llama.cpp async copy; ProMoE 2410.22134.

### 22. CUDA-graph capture extended to MoE and hybrid decode
- **What:** Current graph capture excludes hybrid/MoE decode (engine-report shows 168→346 with graphs for
  dense). Capture the MoE decode step too (with fixed-shape expert slots).
- **Why / gain:** Removes per-kernel launch cost on the MoE path → directly attacks **G-B**. Confidence M.
- **Impl:** `GpuState::step`/graph in `gpu.rs`; requires expert cache to present fixed addresses (slot pool).
- **Risk:** graphs break when the expert set changes; need re-capture policy per token-count.
- **Channel:** Beta.
- **Toggle:** "Record the GPU's work once and replay it, so starting a reply is quicker."
- **Source:** engine-report (CUDA graph replay 168→346); NVIDIA CUDA graphs docs.

### 23. 2:4 structured sparse tensor cores for pruned weights
- **What:** If weights are 2:4 sparse (or we sparsify), use Ampere sparse mma (2x throughput).
- **Why / gain:** 2x compute for the sparse matmul; but requires sparse weights (quality/format work).
  Confidence L for the general case.
- **Impl:** sparse mma variants + a sparsity-aware repack; `quant.rs` flag.
- **Risk:** large quality/format risk; only for models/quant formats that support it.
- **Channel:** Experimental.
- **Toggle:** "Use the GPU's sparse math units (only for models prepared for it)."
- **Source:** MARLIN Sparse-MARLIN (2408.11743); Ampere 2:4 sparsity (NVIDIA docs).

### 24. Programmatic Dependent Launch (PDL) — unavailable on sm_86
- **What:** Overlap the tail of one kernel with the head of the next via `cudaGridDependencySynchronize`.
- **Why / gain:** Ideal for chained decode kernels; but **sm_90+ only**. Marked here so it is not re-proposed.
- **Impl:** n/a on Machine A; relevant to a future Ada/Hopper/Blackwell machine.
- **Risk:** unavailable; skip.
- **Channel:** Experimental (unavailable).
- **Toggle:** (none; hardware-gated off).
- **Source:** CUDA Programming Guide (PDL, Hopper+).

### 25. Warp-specialisation (producer/consumer) within a CTA
- **What:** Some warps only issue `cp.async`/`ldmatrix` (producers), others only mma (consumers).
- **Why / gain:** FA3-style overlap without wgmma; 10-20% in mma kernels. Confidence M.
- **Impl:** specialise warps in `qmatmul_mma`/attention; explicit named barriers.
- **Risk:** complexity; marginal at small tiles.
- **Channel:** Experimental.
- **Toggle:** (part of tensor-core prefill).
- **Source:** FlashAttention-3 arXiv 2407.08608 (concept; wgmma specifics n/a on sm_86).

### 26. Occupancy/launch-config autotune per kernel per GPU
- **What:** Sweep `__launch_bounds__`, tile sizes, stage counts at first run and cache results per GPU.
- **Why / gain:** mma kernels are sensitive to tile/occupancy; autotuned configs typically add 10-30% over a
  single hardcoded config. Confidence M-H.
- **Impl:** `autotune` (new) measured on device; store in `hw.rs`/machine profile; consumed in `gpu.rs`.
- **Risk:** first-run cost; noisy measurements.
- **Channel:** Beta.
- **Toggle:** "Find the fastest settings for your card automatically."
- **Source:** TVM/Ansor arXiv 2006.06762; llama.cpp `-ngl`/kernel autotuning practice.

---

## B. Quantisation — formats, dequant-in-mma, KV/activation quant, low-bit (items 27-48)

### 27. Native Unsloth Dynamic (UD) IQ quant read path (signature format)
- **What:** First-class support for UD IQ quants (mixed per-layer bit-widths: IQ1/IQ2/Q3_K_XL/Q4_K_XL).
- **Why / gain:** The product already promises "signature quantisation: Unsloth Dynamic (UD) IQ quants"
  (engine.md). Enables the smallest files with best measured quality/GB, so bigger/better models fit the
  8 GB card → more experts cached. Enabler, plus a few % from higher KV/expert residency. Confidence H.
- **Impl:** `quant.rs`/`quant_more.rs` already carry IQ tables (`iq_tables.rs`); add per-layer bit-width
  awareness in `Model::load` and repack for mma (item 29). Tests vs stored reference logits.
- **Risk:** IQ dequant correctness; already partially present; validation effort.
- **Channel:** Stable.
- **Toggle:** "Run the smallest, highest-quality quantised models the community publishes."
- **Source:** Unsloth Dynamic v2.0 (unsloth.ai/blog/dynamic-v2, 2025-04-24).

### 28. IQ dequant directly into mma operand registers (no f32 round-trip)
- **What:** Decode IQ2/IQ3 codebook entries straight into the register layout mma needs.
- **Why / gain:** IQ dequant is the prefill bottleneck for IQ models; removes f32 materialisation →
  big **G-A** help for IQ files. 2-4x on IQ prefill; confidence M.
- **Impl:** `iq_dequant_f16_frag` in `kernels.cu` using `iq_tables.rs` grids computed on host and uploaded
  as constant/`__device__` tables.
- **Risk:** IQ grids are large; keep them in smem/const to avoid divergence.
- **Channel:** Beta.
- **Toggle:** (part of tensor-core prefill).
- **Source:** llama.cpp IQ kernels (github.com/ggml-org/llama.cpp); iq_tables in this repo.

### 29. Repack IQ/K-quants to an int8-operand layout for item 1
- **What:** Offline transform IQ/K blocks so the inner values are already int8 fragments (codebook applied
  at repack time), letting item 1 use plain int8 mma.
- **Why / gain:** Best of both: IQ's small size kept on disk, int8 tensor cores at compute time. Enabling
  change; confidence M.
- **Impl:** extends `repack.rs`; lossless for integer IQ codes if the codebook maps to int8 range.
- **Risk:** IQ codebooks are non-uniform; some formats may not fit int8 without scale changes.
- **Channel:** Experimental.
- **Toggle:** (part of tensor-core prefill).
- **Source:** MARLIN 2408.11743; llama.cpp repack practice.

### 30. q8 KV cache with paged layout (already planned; make it default)
- **What:** 8-bit KV pages, per-head scales.
- **Why / gain:** Halves KV VRAM vs fp16 → more experts resident (helps **G-B**) or longer context. 0
  quality risk at q8. Confidence H.
- **Impl:** KV quant in `gpu.rs` step path; attention dequant fused.
- **Risk:** low; small dequant cost in attention.
- **Channel:** Stable.
- **Toggle:** "Store attention memory in 8-bit so more of your conversation and more experts fit in VRAM."
- **Source:** engine.md §5; llama.cpp `-ctk q8_0`.

### 31. q4 KV pages beyond a context threshold
- **What:** Switch colder KV to 4-bit once context exceeds a threshold.
- **Why / gain:** Up to 4x KV compression for long chats → long context on 8 GB. Quality risk at q4 is
  small but nonzero. Confidence M.
- **Impl:** `kv_type` policy in `gpu.rs` + `autotune` threshold.
- **Risk:** quality; only enable past measured context.
- **Channel:** Beta.
- **Toggle:** "Compress older attention memory to 4-bit on very long chats."
- **Source:** KIVI arXiv 2402.02750; KVQuant 2401.18079.

### 32. Per-channel K / per-token V quantisation (KIVI)
- **What:** Quantise keys per channel (outlier channels) and values per token.
- **Why / gain:** Makes sub-8-bit KV viable → same as item 31 but higher quality. Confidence M.
- **Impl:** two quant schemes in the KV write/attention kernels.
- **Risk:** per-channel K needs grouped streaming (residual fp16 window).
- **Channel:** Beta.
- **Toggle:** (part of KV quant).
- **Source:** KIVI 2402.02750.

### 33. Pre-RoPE key quantisation
- **What:** Quantise K before rotary so outlier channels stay aligned.
- **Why / gain:** Better 3-4 bit KV quality. Confidence M.
- **Impl:** fold into item 16 (RoPE).
- **Risk:** must re-apply RoPE after dequant in attention.
- **Channel:** Beta.
- **Toggle:** (part of KV quant).
- **Source:** KVQuant 2401.18079.

### 34. Dense-and-sparse outlier isolation for KV (KVQuant)
- **What:** Store per-vector outliers sparsely, quantise the rest to a narrower range.
- **Why / gain:** Essential for accurate 2-3 bit KV. Confidence M.
- **Impl:** outlier detect + sparse store in KV pages.
- **Risk:** index overhead.
- **Channel:** Experimental.
- **Toggle:** (part of low-bit KV).
- **Source:** KVQuant 2401.18079.

### 35. Q-Norm / Hadamard rotation before KV quant
- **What:** Multiply K (and V) by an orthogonal/Hadamard matrix to Gaussianise outliers before quantising.
- **Why / gain:** Enables 2-bit KV with low error; also helps activation int8. Confidence M.
- **Impl:** fused `O(d log d)` Hadamard in the RoPE/write path (`kernels.cu`).
- **Risk:** extra FLOPs; must keep inverse consistent.
- **Channel:** Experimental.
- **Toggle:** (part of low-bit KV).
- **Source:** QuaRot arXiv 2404.00456; SpinQuant 2405.16406 (mark id **UNVERIFIED**); FA3 incoherent
  processing 2407.08608.

### 36. Per-token int8 activation quantisation for W8A8 (feeds item 1)
- **What:** Quantise activations to int8 with per-token (per-32) scales before the mma.
- **Why / gain:** Required for item 1's int8 path; 1 extra tiny kernel per GEMM. Enabler; confidence H.
- **Impl:** extend `quantize_q8` to emit int8 + scale arrays; `QAct` already exists in `quant.rs`.
- **Risk:** small accuracy loss; keep a W4A16 fp16 fallback.
- **Channel:** Beta.
- **Toggle:** (part of tensor-core prefill).
- **Source:** SmoothQuant 2211.10438; MARLIN 2408.11743.

### 37. SmoothQuant-style activation smoothing (offline)
- **What:** Offline rebalance activation/weight scales so activations quantise cleanly.
- **Why / gain:** Makes W8A8 accurate; enabler for item 36/1. Confidence M.
- **Impl:** host-side transform at repack; store the diagonal scales in the model header.
- **Risk:** only valid for models processed with it.
- **Channel:** Experimental.
- **Toggle:** (part of int8 prefill).
- **Source:** SmoothQuant 2211.10438.

### 38. FP8 weight/activation storage (store only on sm_86)
- **What:** Store weights/activations in FP8 (e4m3) for memory savings.
- **Why / gain:** 2x smaller than fp16 **storage**, but sm_86 has no FP8 tensor core, so compute must
  convert. Useful only for VRAM footprint (more experts cached). Confidence L on speed.
- **Impl:** fp8 store + convert-to-f16 kernel before mma.
- **Risk:** conversion cost may exceed the memory saving on sm_86.
- **Channel:** Experimental.
- **Toggle:** "Store weights in 8-bit floating point to free VRAM (your card does the conversion)."
- **Source:** FP8 formats (NVIDIA docs); FA3 FP8 (2407.08608, Hopper/Ada).

### 39. NVFP4 / FP4 — unavailable on sm_86
- **What:** 4-bit float format with hardware blocks (Blackwell).
- **Why / gain:** n/a for Machine A; logged for a future machine.
- **Impl:** n/a.
- **Risk:** unavailable.
- **Channel:** Experimental (unavailable).
- **Toggle:** (hardware-gated off).
- **Source:** NVIDIA NVFP4 (developer blog) — **UNVERIFIED**.

### 40. Ternary / 1.58-bit (BitNet) weight support
- **What:** Read and run ternary weights {-1,0,+1} (BitNet b1.58 style).
- **Why / gain:** 2-4x less weight traffic → decode speedup where models exist. Confidence M (model
  availability is the limiter).
- **Impl:** `quant.rs` DType for ternary/BF16-packed; dequant to f16 or int8 in kernels; CPU via LUT (item 124).
- **Risk:** few production ternary models yet.
- **Channel:** Experimental.
- **Toggle:** "Run ultra-compressed 1.58-bit models when they're available."
- **Source:** BitNet b1.58 arXiv 2402.17764; BitNet v2 arXiv 2504.18415.

### 41. QTIP/trellis lookup-free decoding kernels
- **What:** Decode trellis-coded (TCQ) weights on the fly with computed random-Gaussian codes.
- **Why / gain:** SOTA 2-bit quality at decode cost ~2-4 instructions/weight, ~80% of bandwidth. Good for
  very low-bit models. Confidence M.
- **Impl:** `qtip_decode` in `kernels.cu`; bitshift-trellis state per group.
- **Risk:** immature tooling to produce models; decode complexity.
- **Channel:** Experimental.
- **Toggle:** "Run new ultra-tiny trellis-quantised models at full speed."
- **Source:** QTIP arXiv 2406.11235 (NeurIPS 2024).

### 42. AQLM/QuIP# codebook weights — caution (large codebook)
- **What:** Codebook VQ weights.
- **Why / gain:** Excellent 2-bit quality, but AQLM codebooks exceed cache → slow inference (QTIP paper).
  Only viable with small codebooks (QuIP#-style).
- **Impl:** optional codebook decode kernel.
- **Risk:** often slower than QTIP; deprioritise.
- **Channel:** Experimental.
- **Toggle:** (advanced; off by default).
- **Source:** AQLM 2401.06118; QuIP# 2402.04396; QTIP 2406.11235 (comparison).

### 43. HQQ fast weight-only quant import
- **What:** Read HQQ-quantised checkpoints.
- **Why / gain:** Broader model compatibility, quick 4/3-bit files. Confidence M.
- **Impl:** scale/zero layout decode in `quant_more.rs`.
- **Risk:** format variants.
- **Channel:** Beta.
- **Toggle:** "Run HQQ-quantised models you already have."
- **Source:** HQQ arXiv 2303.08350.

### 44. GPTQ/AWQ checkpoint import
- **What:** Read GPTQ/AWQ 4-bit weights (for compatibility).
- **Why / gain:** Interop with the models users already downloaded. Confidence M.
- **Impl:** group-scale decode; EXL2-style packing variants.
- **Risk:** many sub-variants (act-order, group size).
- **Channel:** Beta.
- **Toggle:** "Run GPTQ and AWQ models you already have."
- **Source:** GPTQ 2210.17323; AWQ 2306.00978.

### 45. EXL3 (ExLlamaV3) format support
- **What:** Read EXL3 quants (which pair tensor-core mma with a compressed format).
- **Why / gain:** Strong quality/GB and a fast kernel design to mirror. Confidence M.
- **Impl:** EXL3 block decode + mma repack.
- **Risk:** spec chasing; licensing review.
- **Channel:** Experimental.
- **Toggle:** "Run EXL3-quantised models."
- **Source:** turboderp/exllamav3 (github) — **UNVERIFIED**.

### 46. Whole-model mixed-precision placement (UD 2.0 style)
- **What:** Per-layer bit widths chosen by sensitivity; the engine must independently decide its own
  placement when the file is uniform.
- **Why / gain:** Saves VRAM where it costs least quality; frees VRAM for experts/KV. Confidence M.
- **Impl:** a `quantize` CLI/background tool + metadata; `autotune` chooses per-layer bits.
- **Risk:** large tooling effort; quality validation.
- **Channel:** Experimental.
- **Toggle:** "Let Cerebro shrink the less important layers so more fits in your GPU."
- **Source:** Unsloth Dynamic v2.0 blog.

### 47. QAT checkpoint import (Gemma-3 QAT)
- **What:** Read quality-aware-trained quantised weights (better than PTQ at same bits).
- **Why / gain:** Better quality at 4-bit for the same size. Confidence M.
- **Impl:** layout decode in `quant_more.rs`.
- **Risk:** format specifics per family.
- **Channel:** Beta.
- **Toggle:** "Run QAT models that keep more quality at the same size."
- **Source:** Gemma 3 arXiv 2503.19786; Unsloth QAT notes.

### 48. Native imatrix importance-matrix metadata consumption
- **What:** Use the `imatrix` importance data some quants ship to guide any engine-side requant/placement.
- **Why / gain:** Better quality at a given size (already inside IQ files); usable to guide mixed placement.
- **Impl:** parse GGUF `imatrix` tensors; feed item 46.
- **Risk:** sparse availability.
- **Channel:** Beta.
- **Toggle:** "Use the model's importance data to place layers smarter."
- **Source:** llama.cpp imatrix docs (github) — **UNVERIFIED**.

---

## C. Speculative decoding (items 49-65)

### 49. Model-native Multi-Token Prediction (MTP) heads
- **What:** Use the extra MTP/next-N prediction heads some models ship (DeepSeek-V3, Qwen3-Next) as the
  drafter; verify N tokens per target pass.
- **Why / gain:** Free draft quality (trained with the model); Strata measured 59-66% acceptance, 1.6-1.8x
  on this PC (docs/research/strata-analysis). Directly helps **G-B** (offloaded MoE: one pass verifies
  several tokens, amortising expert loads). 1.5-1.9x decode; confidence H where heads exist.
- **Impl:** graph support in `model.rs` for `mtp.*` heads; verify path reuses the batched prefill kernels
  in `gpu.rs`; acceptance gating in `Session`.
- **Risk:** model support varies; verify must be exact (lossless).
- **Channel:** Beta.
- **Toggle:** "Let the model guess several words ahead and check them in one step, so replies come faster."
- **Source:** DeepSeek-V3 tech report arXiv 2412.19437; Qwen3-Next HF card; Strata analysis 2026-10-07.

### 50. EAGLE-3 draft head integration
- **What:** Run a small EAGLE-3 draft model (multi-layer feature fusion, training-time test) for the target.
- **Why / gain:** 3.0-6.5x speedups reported (task-dependent); realistic on small hardware 1.8-3x on code
  (templates accept well). Confidence M.
- **Impl:** draft graph in `model.rs`; tree verify in `gpu.rs`; acceptance telemetry.
- **Risk:** needs per-target trained draft weights; VRAM for the draft reduces expert cache.
- **Channel:** Experimental.
- **Toggle:** "Use a tiny helper model to write several tokens at once and verify them."
- **Source:** EAGLE-3 arXiv 2503.01840; SafeAILab/EAGLE (github).

### 51. Medusa multi-head draft
- **What:** Attach several lightweight prediction heads to the frozen target.
- **Why / gain:** Simpler than EAGLE, no separate model; 2-3x reported. Confidence M.
- **Impl:** Medusa head layers + tree attention verification.
- **Risk:** trained heads needed; moderate acceptance.
- **Channel:** Experimental.
- **Toggle:** "Add small guess-heads to the model to generate faster."
- **Source:** Medusa arXiv 2401.10774.

### 52. Lookahead decoding (Jacobi iteration)
- **What:** Generate n-grams with Jacobi iteration + verify, no draft model.
- **Why / gain:** Always available (no training); 1.5-2.3x reported for some tasks. Confidence M.
- **Impl:** parallel Jacobi step in `gpu.rs`, n-gram pool.
- **Risk:** gains depend on task; extra compute.
- **Channel:** Experimental.
- **Toggle:** "Guess ahead without a helper model using a math trick, and keep only the correct guesses."
- **Source:** Lookahead decoding arXiv 2402.02057.

### 53. Prompt-lookup / n-gram drafting
- **What:** Draft by copying n-grams from the prompt/recent text (great for edits, quoted code).
- **Why / gain:** Already in the plan (engine.md §3); Strata measured +6-11% on code edits. Cheap, no
  model. Confidence H on code/agent workloads.
- **Impl:** suffix/ngram index over the context in `Session`; verify via batched pass.
- **Risk:** low acceptance on open prose; gate on measurement.
- **Channel:** Beta.
- **Toggle:** "Copy exact repeated text from your prompt or file to answer faster when editing code."
- **Source:** apoorvumang/prompt-lookup-decoding (github); engine.md §3.

### 54. Suffix decoding / suffix automaton draft
- **What:** Model-free drafting using previously generated suffixes indexed in a suffix automaton.
- **Why / gain:** Strong on repetitive/agentic output; complements prompt-lookup. Confidence M.
- **Impl:** suffix-automaton index in `Session`/`chat.rs`.
- **Risk:** memory for the automaton; task-dependent.
- **Channel:** Experimental.
- **Toggle:** "Reuse text the model already produced to speed up repetitive output."
- **Source:** SuffixDecoding arXiv 2411.04975 (**UNVERIFIED**).

### 55. Tree verification with a single attention mask
- **What:** Verify a whole spec tree (Medusa/EAGLE/SpecExec) in one target pass using a tree-causal mask.
- **Why / gain:** Amortises the expensive target pass (crucial when target is offloaded → **G-B**). 1.2-2x
  on top of linear speculation; confidence M.
- **Impl:** tree mask + position ids in `attn_prefill_tiled`; draft tree builder in `gpu.rs`.
- **Risk:** mask correctness; throughput overhead at large tree sizes.
- **Channel:** Beta.
- **Toggle:** "Check several guessed continuations at once."
- **Source:** Medusa 2401.10774; SpecInfer arXiv 2305.09781; EAGLE-3 2503.01840.

### 56. SpecExec large-tree speculation over an offloaded target
- **What:** Build a large draft tree from a small GPU model; offloaded target verifies hundreds of tokens
  in one pass ≈ the cost of one.
- **Why / gain:** Purpose-built for big-model/small-GPU; up to 10-18x over plain offloading (paper). On
  this 8 GB card, potentially the biggest single **G-B**-adjacent win for >VRAM models. Confidence M.
- **Impl:** draft tree on GPU, batched verify path (already built for prefill) over the offloaded target;
  revisit `docs/research/big-models-on-small-machines`.
- **Risk:** large trees cost target compute; needs careful acceptance/rollback.
- **Channel:** Experimental.
- **Toggle:** "For models too big for your GPU, check a whole tree of guesses per step."
- **Source:** SpecExec arXiv 2406.02532.

### 57. Reuse the batched prefill path as the verify path
- **What:** Verification is a batched forward; route it through the same (now mma) prefill kernels.
- **Why / gain:** No new verification code; benefits from items 1/2. Enabler; confidence H.
- **Impl:** `gpu.rs` verify entry calling `prefill` with tree positions; sampling accepts/rejects in
  `sample.rs`.
- **Risk:** position-id handling for the tree.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** engine architecture (batched prefill already exists).

### 58. Adaptive draft length from measured acceptance
- **What:** Change draft length/spec mode live based on rolling acceptance and time saved.
- **Why / gain:** Avoids the cases where speculation loses (engine.md §3: "enabled only while measured
  acceptance pays"). Keeps every workload at break-even or better. Confidence H (safety/efficiency).
- **Impl:** acceptance EMA + switch in `Session`/`gpu.rs`.
- **Risk:** oscillation; hysteresis.
- **Channel:** Stable.
- **Toggle:** "Turn guessing on only when it is actually making things faster for this chat."
- **Source:** engine.md §3; Strata analysis (gating).

### 59. Small GPU draft + offloaded CPU target
- **What:** Draft model entirely on GPU; target partly on CPU.
- **Why / gain:** The draft's cheap passes hide the CPU target's per-step cost. Confidence M.
- **Impl:** two `Session`s (draft in VRAM, target split) + verify.
- **Risk:** VRAM taken by the draft hurts target residency.
- **Channel:** Experimental.
- **Toggle:** "Use a small fast model to predict, and the big model to confirm."
- **Source:** SpecExec 2406.02532; Big-models research 2026-10-10.

### 60. Layer-skip / self-speculative decoding
- **What:** Use a subset of target layers (early exit) as its own drafter; verify with full model.
- **Why / gain:** No extra model, no extra VRAM. 1.3-2x reported. Confidence M.
- **Impl:** early-exit head + verify; needs a trained/skipped exit.
- **Risk:** requires calibration; quality at exit.
- **Channel:** Experimental.
- **Toggle:** "Let the big model guess with part of its brain, then double-check."
- **Source:** LayerSkip arXiv 2404.16710; Self-speculative decoding arXiv 2309.08168.

### 61. N-gram draft table streamed from SSD
- **What:** A large offline n-gram table (Strata reads a 28.8 GB table a few rows/token from SSD).
- **Why / gain:** Cheap, high-hit drafts on repetitive text without a model. Confidence M.
- **Impl:** memory-mapped table + async row reads (item 138/142).
- **Risk:** disk latency; only worth it with a fast NVMe and cacheable access.
- **Channel:** Experimental.
- **Toggle:** "Use a large phrase table on disk to guess common text quickly."
- **Source:** Strata analysis 2026-10-07.

### 62. Speculative verification batching across requests (multi-slot)
- **What:** Verify spec trees of several concurrent requests in one batched pass.
- **Why / gain:** Amortises target cost across user sessions; throughput win in serving. Confidence M.
- **Impl:** multi-slot verify in `gpu.rs` + continuous batching (item 158).
- **Risk:** complex scheduling; tail latency.
- **Channel:** Experimental.
- **Toggle:** (server-side; no single toggle).
- **Source:** EAGLE-3 2503.01840 (batch-64 throughput); Sarathi-Serve 2403.02310.

### 63. Speculative decoding aware of the expert cache (SP-MoE / SpecMoEOff)
- **What:** When verifying multiple tokens, prefetch exactly the experts the draft tokens will need.
- **Why / gain:** Multi-token verification otherwise explodes expert traffic; SD-aware offloading reported
  1.07-3.5x TPOT over SD-agnostic offloading. Directly **G-B**. Confidence M.
- **Impl:** draft-to-target expert correspondence prefetch + cutoff-layer policy + pipelined I/O in
  `gpu.rs` MoE step.
- **Risk:** prediction accuracy; overfetch bandwidth.
- **Channel:** Experimental.
- **Toggle:** "Fetch the experts the guessed words need before checking them."
- **Source:** SP-MoE arXiv 2510.10302; SpecMoEOff arXiv 2508.21706.

### 64. Hybrid drafter selection (MTP vs EAGLE vs n-gram) per workload
- **What:** Choose the draft source per model/prompt: model-native MTP, an EAGLE head, prompt-lookup, or
  a small draft model, based on recent measured acceptance.
- **Why / gain:** No single drafter wins everywhere (Strata: MTP on prose, prompt-lookup on code edits);
  an orchestrator keeps whichever is paying. Confidence M.
- **Impl:** drafter registry + acceptance tracker in `Session`/`gpu.rs`; picks per request.
- **Risk:** switching cost; hysteresis needed.
- **Channel:** Beta.
- **Toggle:** "Automatically use the best way to predict the next words for this kind of task."
- **Source:** Strata analysis 2026-10-07; engine.md §3.

### 65. Lossless verification guard (exact speculative sampling)
- **What:** Enforce the exact accept/reject distribution (speculative sampling) so speculation never
  changes the output distribution; a test proves it.
- **Why / gain:** Speculation must be quality-neutral; this keeps every speculation toggle safe to leave
  on. Confidence H (safety).
- **Impl:** `sample.rs` rejection sampling vs the target distribution; a reference test in the style of the
  logits test in `model.rs`.
- **Risk:** none (guard rail).
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** Speculative decoding (Leviathan et al. arXiv 2211.17192; Chen et al. 2302.01318).

---

## D. MoE offload — expert cache, prediction, CPU/GPU overlap (items 66-92)

### 66. Adaptive frequency+recency expert cache (exists; harden)
- **What:** Keep the most-used experts in VRAM using LRU + frequency scoring; the signature feature.
- **Why / gain:** The report credits it for the 38.6-55.3 vs 33-36 tok/s lead once adapted. Keep and
  strengthen (items 68-79). Baseline advantage; confidence H.
- **Impl:** `MoeGpu`/expert slots in `gpu.rs`; telemetry in `model.rs` Stats.
- **Risk:** thrash on shifting topics; needs score decay.
- **Channel:** Stable.
- **Toggle:** "Keep the experts this chat uses most on the GPU."
- **Source:** engine-report; engine.md §2 (MoE expert cache).

### 67. Prompt-aware warm start from the first prompt (exists; extend)
- **What:** Profile the prompt's experts during prefill and place hot ones (report: 23.7→55.3 tok/s,
  hit 41%→85%).
- **Why / gain:** Directly fixes **G-B**'s second sentence ("before the expert cache has adapted").
  Confidence H.
- **Impl:** reuse prefill routing stats to seed the cache (`gpu.rs` prefill path).
- **Risk:** long novel prompts still cold.
- **Channel:** Stable.
- **Toggle:** "Put the experts your prompt is about to use into fast memory first."
- **Source:** engine-report ("Prompt-aware expert placement").

### 68. Shipped per-model starting profile from mixed prompts
- **What:** Bundle a measured starting expert profile per popular model so the first turn is already warm.
- **Why / gain:** Report "Next, in order" #1 second half; removes the cold first turn entirely for known
  models. Confidence H.
- **Impl:** profile file format + loader in `gpu.rs` (engine already has `save_expert_profile`).
- **Risk:** profile drift across content; stack with item 67.
- **Channel:** Stable.
- **Toggle:** "Start with experts the model usually needs, already loaded."
- **Source:** engine-report "Next #1".

### 69. Asynchronous expert swaps on the copy stream, many per token
- **What:** Replace batched 16-per-32-token swaps with fine-grained async swaps issued continuously.
- **Why / gain:** Report "Next #1"; removes swap stalls at first turn → **G-B**. Expected to move first-turn
  decode from 19-23 toward 30+ tok/s. Confidence H.
- **Impl:** copy stream + event sync in `gpu.rs` MoE step; reuse pinned staging (item 86).
- **Risk:** PCIe contention; must interleave with compute.
- **Channel:** Beta.
- **Toggle:** "Swap experts in continuously in the background instead of in bursts."
- **Source:** engine-report "Next #1"; ProMoE 2410.22134.

### 70. Cross-layer gate-input prefetch (Fate-style)
- **What:** Feed layer i's gate input into layer i+1's gate on the CPU to predict the next experts; prefetch
  during attention.
- **Why / gain:** ~97% prefetch accuracy, 99% hit rate reported; near-free (CPU computes the next gate).
  Directly **G-B**. Confidence M-H.
- **Impl:** clone gate input to CPU during `moe` step in `gpu.rs`; prediction thread in `pool.rs`.
- **Risk:** CPU load; accuracy for first layer (needs item 74).
- **Channel:** Beta.
- **Toggle:** "Predict which experts the next layer needs and fetch them early."
- **Source:** Fate arXiv 2502.12224.

### 71. Learned per-layer expert predictor (ProMoE)
- **What:** Small trained predictor ranks next-layer experts for prefetch; proactive (off critical path).
- **Why / gain:** 84.7% accuracy, 2.07x decode reported. Confidence M.
- **Impl:** predictor weights + inference in `pool.rs`/`gpu.rs`.
- **Risk:** training data per model; accuracy.
- **Channel:** Experimental.
- **Toggle:** "Let a small trained helper predict the next experts."
- **Source:** ProMoE arXiv 2410.22134.

### 72. Mixed-precision miss copies (HOBBIT)
- **What:** On a cache miss, load a low-precision copy of the expert (int4) instead of the full one; use
  higher precision only for on-demand.
- **Why / gain:** Reduces miss loading volume; HOBBIT reports up to 9.93x decode over prior offloading.
  Useful when the expert lives in RAM. Confidence M.
- **Impl:** multiple precision versions per expert in RAM + selection in `gpu.rs`; extra RAM cost.
- **Risk:** accuracy on low-precision experts; RAM doubling.
- **Channel:** Experimental.
- **Toggle:** "Fetch a lightweight version of a missing expert so it arrives sooner."
- **Source:** HOBBIT arXiv 2411.01433.

### 73. CPU-GPU concurrent expert compute (Fiddler)
- **What:** Missed experts run on the CPU in parallel with the GPU's current layer, transferring
  activations (small) not weights (large). Already the engine's core design.
- **Why / gain:** The engine.md §2 approach; report shows the MoE hybrid works (17.4 tok/s after async).
  Keep and tune. Confidence H.
- **Impl:** `moe_cpu` in `model.rs` + `pool.rs`; overlap with GPU in `gpu.rs`.
- **Risk:** CPU is the bottleneck on this machine (only 6P+8E); load balance matters.
- **Channel:** Stable.
- **Toggle:** "Compute missed experts on the CPU at the same time as the GPU works."
- **Source:** engine.md §2; Fiddler arXiv 2402.07033.

### 74. Auxiliary first-layer gate predictor
- **What:** A trained small layer predicts layer-0's experts from the previous token's last hidden state
  (there is no prior gate for layer 0).
- **Why / gain:** Closes the cold-start hole in item 70. Confidence M.
- **Impl:** tiny linear head; inference in `pool.rs`.
- **Risk:** training needed; adds a dependency.
- **Channel:** Experimental.
- **Toggle:** (advanced; part of expert prediction).
- **Source:** AdapMoE arXiv 2408.10284 (auxiliary predictive layer).

### 75. MoE-Lightning CGOPipe-style CPU-GPU-I/O pipeline with paged weights
- **What:** Fine-grained schedule overlapping CPU compute, GPU compute and several I/O directions; weights
  as pages with a page table.
- **Why / gain:** Up to 10.3x throughput over SOTA offloading on memory-constrained GPUs (batch serving).
  Confidence M.
- **Impl:** page table + staged transfers in `gpu.rs`; micro-batch scheduler in a new serving module.
- **Risk:** complex; mostly a throughput/batch win, less for single-stream latency.
- **Channel:** Experimental.
- **Toggle:** "Overlap CPU, GPU and disk work so nothing waits on the others."
- **Source:** MoE-Lightning arXiv 2411.11217 (ASPLOS 2025).

### 76. KTransformers async CPU-GPU scheduling + Expert Deferral
- **What:** AMX-optimised CPU expert kernels (n/a on 12900H) and a schedule that defers some experts to
  raise CPU utilisation from <75% to ~100%.
- **Why / gain:** Expert Deferral gives up to 1.45x extra throughput at ≤0.5% accuracy drop, and is
  CPU-ISA-independent. Confidence M.
- **Impl:** deferral policy in `gpu.rs`/`pool.rs`; keep AMX kernels for future machines.
- **Risk:** quality impact from deferral must be measured.
- **Channel:** Experimental.
- **Toggle:** "Keep the CPU busy by deferring a few experts to it intelligently."
- **Source:** KTransformers (SOSP 2025) DOI 10.1145/3731569.3764843.

### 77. Adaptive top-k gating (AdapMoE)
- **What:** Per-token sensitivity-based variable number of experts (fewer when safe).
- **Why / gain:** ~25% fewer activated experts, 1.35x speedup, no accuracy drop reported. Directly reduces
  CPU expert load (**G-C/G-D**). Confidence M.
- **Impl:** sensitivity thresholds in `router_topk` (`kernels.cu`) + host policy.
- **Risk:** needs calibration; accuracy risk if thresholds wrong.
- **Channel:** Experimental.
- **Toggle:** "Use only as many experts as a token really needs."
- **Source:** AdapMoE arXiv 2408.10284 (github.com/PKU-SEC-Lab/AdapMoE).

### 78. Shared-expert pinning in VRAM
- **What:** Always-resident shared/always-on experts (DeepSeek/Qwen MoE) never evicted.
- **Why / gain:** These are used every token; prevents pointless thrash. Confidence H (for models that
  have them).
- **Impl:** cache policy flag in `gpu.rs`.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** "Always keep the experts every token uses in memory."
- **Source:** DeepSeekMoE arXiv 2401.06066; KTransformers practice.

### 79. Score-based cache eviction (HybriMoE)
- **What:** Evict by impact-aware score, not plain LRU.
- **Why / gain:** 1.33x prefill / 1.70x decode over SOTA hybrid MoE reported; better hit rate under unstable
  routing. Confidence M.
- **Impl:** score in the cache manager (`gpu.rs`).
- **Risk:** score tuning.
- **Channel:** Beta.
- **Toggle:** "Choose which experts to keep in memory by how much they matter."
- **Source:** HybriMoE arXiv 2504.05897 (github.com/PKU-SEC-Lab/HybriMoE).

### 80. Sequence-level expert reallocation during prefill (DAOP)
- **What:** Profile the whole sequence's experts in prefill, then move the dominant ones to GPU for the
  decode phase.
- **Why / gain:** Up to 8.2x over caching/prefetch, 1.35x over offloading reported; makes decode warm.
  Complements item 67. Confidence M.
- **Impl:** prefill routing histogram → placement in `gpu.rs`.
- **Risk:** reallocation transfers during prefill; must be cheap.
- **Channel:** Beta.
- **Toggle:** "After reading your prompt, put the experts it needs into fast memory."
- **Source:** DAOP arXiv 2501.10375.

### 81. Shallow-favoring cache allocation (Fate)
- **What:** Allocate more cache to shallow layers where prediction is weaker.
- **Why / gain:** Raises hit rate to ~99% in Fate; targets the cold-start hole. Confidence M.
- **Impl:** per-layer cache budget in `gpu.rs`.
- **Risk:** interacts with item 68 profiles.
- **Channel:** Beta.
- **Toggle:** (part of expert cache policy).
- **Source:** Fate arXiv 2502.12224.

### 82. Expert cache compression (keep more experts in the same VRAM)
- **What:** Store idle cached experts in int4/int8 in VRAM, decompressing on use.
- **Why / gain:** Fits ~2-4x more experts in the same VRAM → more hits on the 8 GB card. Confidence M.
- **Impl:** compressed slot pool + dequant in `moe_matvec` (`kernels.cu`).
- **Risk:** dequant cost per expert use.
- **Channel:** Experimental.
- **Toggle:** "Store spare experts compressed so more fit in your GPU."
- **Source:** HOBBIT 2411.01433; engine-report ("attention/output weights kept in K-quant to free VRAM").

### 83. Chunked/early-preemption prefetch (ProMoE)
- **What:** Split expert fetches into chunks and preempt low-priority prefetch when a real miss appears.
- **Why / gain:** Maximises bandwidth use and hides latency; part of ProMoE's 2x. Confidence M.
- **Impl:** priority task queue + chunker in `gpu.rs`.
- **Risk:** scheduling complexity.
- **Channel:** Beta.
- **Toggle:** (part of background expert fetching).
- **Source:** ProMoE arXiv 2410.22134.

### 84. Cutoff-layer prefetch depth policy (SP-MoE)
- **What:** Bound how many layers ahead to prefetch using a latency model to avoid overfetch.
- **Why / gain:** "Just-in-time" without wasting PCIe; part of 1.07-3.5x. Confidence M.
- **Impl:** analytic model in `autotune`; consumed by prefetch in `gpu.rs`.
- **Risk:** model accuracy.
- **Channel:** Experimental.
- **Toggle:** (part of expert fetching).
- **Source:** SP-MoE arXiv 2510.10302.

### 85. Paged expert weights (page table) for SSD/RAM tiering
- **What:** Expert weights stored as pages addressed through a page table so any tier (VRAM/RAM/SSD) can
  serve them without copies.
- **Why / gain:** Unifies item 82 and SSD streaming (item 132); enables bigger-than-RAM MoE. Confidence M.
- **Impl:** page table in `gpu.rs`; `moe_matvec` (already takes `pool, slot_bytes, off`) extended.
- **Risk:** indirect addressing overhead.
- **Channel:** Beta.
- **Toggle:** "Load only the parts of big models you need, from wherever they live."
- **Source:** MoE-Lightning 2411.11217; engine.md §2.

### 86. Pinned staging buffer for expert copies (exists; tune)
- **What:** Stage expert bytes through pinned host memory for full PCIe bandwidth.
- **Why / gain:** The engine already uses `PinnedHostSlice`; tuning size/count (double-buffer) removes
  bubbles. Confidence H.
- **Impl:** `gpu.rs` pinned buffers.
- **Risk:** pinned memory exhaustion.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** CUDA pinned-memory docs; MoE-Lightning (pinned staging).

### 87. GPU grouped-GEMM MoE compute with mma (see item 9) + int8 experts
- **What:** Run the on-GPU experts with int8 mma rather than dp4a.
- **Why / gain:** Faster expert compute frees the timeline for more overlap; part of **G-B**. Confidence M.
- **Impl:** `moe_matvec` mma variant in `kernels.cu`.
- **Risk:** int8 quality (small).
- **Channel:** Beta.
- **Toggle:** (part of tensor-core).
- **Source:** item 1; MARLIN 2408.11743.

### 88. CPU router fallback (keep GPU free)
- **What:** Run part of routing on the CPU when the GPU is the bottleneck.
- **Why / gain:** Frees GPU cycles; small. Confidence L-M.
- **Impl:** `router_topk` CPU mirror in `model.rs`.
- **Risk:** sync overhead.
- **Channel:** Experimental.
- **Toggle:** (advanced).
- **Source:** engine-report ("parallel router" CPU).

### 89. Activating-expert telemetry + adaptive placement loop
- **What:** Feed the monitor's hit-rate/heat data back into cache policy live.
- **Why / gain:** Strata and the report both show placement is the lever; closed-loop tuning keeps the hit
  rate high as topics change. Enabler; confidence M.
- **Impl:** Stats → policy in `gpu.rs`; UI in the app (outside engine).
- **Risk:** control oscillation.
- **Channel:** Beta.
- **Toggle:** "Keep retuning which experts stay in memory as the chat changes."
- **Source:** positioning.md (monitor); engine-report.

### 90. Expert merging (HC-SMoE) / low-rank expert compression
- **What:** Merge similar experts or low-rank-factor their weights to shrink RAM/VRAM.
- **Why / gain:** Fewer bytes to move; but requires a preprocessing step and may change outputs. Confidence L.
- **Impl:** offline tool + merged expert reads.
- **Risk:** quality; not lossless.
- **Channel:** Experimental.
- **Toggle:** "Combine similar experts into fewer, smaller ones to fit more on the GPU."
- **Source:** HC-SMoE arXiv 2410.13202 (**UNVERIFIED id**); expert-skipping literature.

### 91. Offload-aware speculative + expert prefetch co-schedule (see item 63)
- **What:** Combine speculation with expert prefetch under one latency model.
- **Why / gain:** The two techniques compete for PCIe; co-scheduling avoids double-spending. Confidence M.
- **Impl:** shared scheduler in `gpu.rs`/`autotune`.
- **Risk:** complexity.
- **Channel:** Experimental.
- **Toggle:** (advanced; automatic).
- **Source:** SP-MoE 2510.10302; SpecMoEOff 2508.21706.

### 92. SSD expert tier with windowing / row-column bundling
- **What:** For MoE larger than RAM, read only active expert rows from NVMe, bundling to large sequential
  reads.
- **Why / gain:** LLM-in-a-Flash style 4-25x over naive loading; enables >RAM MoE. Confidence M.
- **Impl:** expert rows in `gguf.rs` prefetch path (format already bundles experts contiguously in GGUF).
- **Risk:** SSD latency; needs large reads.
- **Channel:** Experimental.
- **Toggle:** "Stream the expert parts you need from your SSD when the model doesn't fit in RAM."
- **Source:** LLM in a Flash arXiv 2312.11514; engine.md placement matrix.

---

## E. KV cache and long context (items 93-112)

### 93. Paged KV blocks (engine.md §1; is it built?)
- **What:** Fixed-size KV pages so caches share memory without copies.
- **Why / gain:** Foundation for prefix reuse, defrag and cold-page streaming. Enabler; confidence H.
- **Impl:** `gpu.rs` KV allocation + attention page table (item 15).
- **Risk:** indirect addressing cost; correctness.
- **Channel:** Beta.
- **Toggle:** "Share attention memory between chats instead of copying it."
- **Source:** engine.md §1; PagedAttention arXiv 2309.06180.

### 94. Radix prefix cache (engine.md §1; is it built?)
- **What:** Radix tree of KV blocks keyed by token prefixes; reuse the longest cached prefix.
- **Why / gain:** The single biggest win for agent/coding harnesses (agents resend 100K histories).
  Confidence H.
- **Impl:** radix tree in `Session`/`chat.rs`; paged KV backing.
- **Risk:** correctness of prefix matching; eviction policy.
- **Channel:** Stable.
- **Toggle:** "Skip re-reading text the model has already seen in this chat."
- **Source:** engine.md §1; SGLang RadixAttention arXiv 2312.07104.

### 95. Message-boundary checkpoints (engine.md §1)
- **What:** Snapshot KV at message boundaries so an agent rewriting its last message keeps history.
- **Why / gain:** Exactly the agent case; avoids rereading the whole history. Confidence H.
- **Impl:** checkpoint API in `Session`; store in RAM/SSD.
- **Risk:** memory for checkpoints.
- **Channel:** Stable.
- **Toggle:** "Keep your chat's memory so editing the last message doesn't re-read everything."
- **Source:** engine.md §1; Strata analysis.

### 96. Cold KV pages to RAM + prefetch
- **What:** Past a threshold, cold KV pages live in RAM and are prefetched for attention.
- **Why / gain:** engine.md §5; long context on 8 GB cards. Confidence M.
- **Impl:** paged KV with host backing + `cp.async` prefetch in attention.
- **Risk:** PCIe latency on attention; needs lookahead.
- **Channel:** Beta.
- **Toggle:** "Keep older chat memory in RAM and fetch it only when needed, so long chats still fit."
- **Source:** engine.md §5; Strata ("KV in RAM past 64K").

### 97. Prefix caching persisted to disk across sessions
- **What:** Save a serialised prefix cache so reopening a project keeps its KV.
- **Why / gain:** Time-to-first-token on returning to a project drops to near zero. Confidence M.
- **Impl:** serialise paged KV to `Cerebro/` cache dir; validate token hash.
- **Risk:** staleness; disk usage.
- **Channel:** Beta.
- **Toggle:** "Remember the prompt cache of your projects on disk so reopening is instant."
- **Source:** engine.md §1; llama.cpp slot save/restore.

### 98. KV cache compaction/defragmentation
- **What:** Repack pages to remove fragmentation from finished requests.
- **Why / gain:** Returns VRAM to the expert cache on the 8 GB card. Confidence M.
- **Impl:** `gpu.rs` KV allocator.
- **Risk:** compaction stalls.
- **Channel:** Beta.
- **Toggle:** "Tidy up unused chat memory so your GPU can use it elsewhere."
- **Source:** vLLM PagedAttention 2309.06180.

### 99. Attention-sink retention (StreamingLLM)
- **What:** Always keep the first few tokens' KV (attention sinks).
- **Why / gain:** Makes eviction/streaming stable; near-free. Confidence H (supporting technique).
- **Impl:** pin first pages in `gpu.rs`.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** "Always remember the very start of the conversation, even in very long chats."
- **Source:** StreamingLLM arXiv 2309.17453.

### 100. Heavy-hitter (H2O) KV eviction
- **What:** Drop low-cumulative-attention KV during decode.
- **Why / gain:** Bounds KV growth under a fixed budget → long chats on small cards. Confidence M.
- **Impl:** attention-score accumulator per K page in `gpu.rs`.
- **Risk:** quality on recall tasks.
- **Channel:** Experimental.
- **Toggle:** "Forget the chat details the model isn't using anymore to fit longer conversations."
- **Source:** H2O arXiv 2306.14048.

### 101. SnapKV prompt-KV compression
- **What:** Compress prompt KV to the important positions (observation window + per-head clustering).
- **Why / gain:** 3.6x decode speedup / 8.2x memory reported at 16K, negligible accuracy loss. Great for
  huge agent prompts. Confidence M.
- **Impl:** selection after prefill in `gpu.rs`; keep observation window (pin those pages).
- **Risk:** retrieval quality on scattered facts.
- **Channel:** Experimental.
- **Toggle:** "Shrink the memory of very long prompts to the parts the model actually looks at."
- **Source:** SnapKV arXiv 2404.14469.

### 102. Quest query-aware KV page selection
- **What:** Score pages by query criticality per step; read only the top pages.
- **Why / gain:** Reduces attention reads at long context → decode speed. Confidence M.
- **Impl:** page score + sparse page gather in attention (`kernels.cu`).
- **Risk:** approximation; extra scoring kernel.
- **Channel:** Experimental.
- **Toggle:** "Read only the chat pages relevant to the current question."
- **Source:** Quest arXiv 2406.10774.

### 103. MInference sparse attention patterns
- **What:** Detect attention patterns (vertical/slash/block) and run sparse kernels for long prefill.
- **Why / gain:** Large long-prompt prefill speedups (reported up to 10x at 1M on A100). Helps **G-A** on
  long prompts. Confidence M.
- **Impl:** offline pattern search + sparse prefill kernel.
- **Risk:** pattern instability; quality.
- **Channel:** Experimental.
- **Toggle:** "Speed up very long prompt reading by skipping attention that doesn't matter."
- **Source:** MInference arXiv 2407.02490.

### 104. PyramidKV layered KV budget
- **What:** Allocate more KV to lower layers, less to higher.
- **Why / gain:** Better quality/memory at a fixed budget than uniform eviction. Confidence M.
- **Impl:** per-layer budget in `gpu.rs`.
- **Risk:** tuning per model.
- **Channel:** Experimental.
- **Toggle:** "Give each layer just the right amount of memory in long chats."
- **Source:** PyramidKV arXiv 2406.02069.

### 105. Chunked prefill with autotuned chunk size (exists; tune)
- **What:** Long prompts in chunks (report: 8,192 chunks; tuned by sweep).
- **Why / gain:** Keeps interactive latency low and overlaps expert prefetch (item 21). Confidence H.
- **Impl:** chunk size in `gpu.rs`; `-ub`-style autotune (item 180).
- **Risk:** chunk too large stalls; too small underutilises.
- **Channel:** Stable.
- **Toggle:** (internal; part of "faster prompts").
- **Source:** engine-report; Sarathi-Serve 2403.02310.

### 106. KV-type + context-depth autotune (per model/machine)
- **What:** Sweep q8/q4 KV and context depth to the fastest stable config.
- **Why / gain:** The placement matrix already calls for this; picks the sweet spot automatically.
  Confidence H.
- **Impl:** `autotune` module; store in machine profile.
- **Risk:** sweep cost.
- **Channel:** Stable.
- **Toggle:** "Pick the fastest safe memory settings for this model on your machine."
- **Source:** engine.md placement matrix.

### 107. RoPE base scaling / YaRN for extended context
- **What:** Adjust RoPE frequencies to run beyond the trained window.
- **Why / gain:** Extends usable context; quality depends on method. Confidence M.
- **Impl:** `inv_freq` computation in `model.rs` + GGUF `rope.scaling.*` keys.
- **Risk:** quality beyond training length.
- **Channel:** Beta.
- **Toggle:** "Let the model handle much longer chats than it was trained for."
- **Source:** YaRN arXiv 2309.00071 (**UNVERIFIED id**); RoFormer 2104.09864.

### 108. Attention-sink / first-token full precision in quant KV
- **What:** Keep the first token (and sink tokens) in fp16 even under KV quant.
- **Why / gain:** KVQuant/IntactKV/SKVQ all keep them fp16; cheap accuracy insurance. Confidence H.
- **Impl:** pin first page in `gpu.rs`.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** KVQuant 2401.18079; review 2412.19442.

### 109. Variable KV precision by depth (recent tokens high precision)
- **What:** Keep the most recent R tokens in fp16, older in q4.
- **Why / gain:** KIVI/WKVQuant show recent tokens matter most; quality at fixed memory. Confidence M.
- **Impl:** residual fp16 window in the KV writer.
- **Risk:** complexity.
- **Channel:** Beta.
- **Toggle:** "Keep the newest chat memory sharp and compress the older parts."
- **Source:** KIVI 2402.02750; WKVQuant (review 2412.19442).

### 110. Preemptive swap vs recompute under KV pressure
- **What:** When VRAM is short, either swap a request's KV to RAM or recompute it later.
- **Why / gain:** Serving policy; keeps throughput up with many requests. Confidence M.
- **Impl:** scheduler in the serving module.
- **Risk:** recompute cost; swap bandwidth.
- **Channel:** Experimental.
- **Toggle:** (server-side automatic).
- **Source:** vLLM/SGLang preemption; Orca continuous batching.

### 111. Logit soft-capping and final-logit slicing
- **What:** Correct handling of Gemma-style soft-capping and computing only needed prompt logits.
- **Why / gain:** Correctness for Gemma 4 + saves a stray full-vocab pass at prefill. Confidence M.
- **Impl:** `sample.rs` + prefill argmax.
- **Risk:** per-family differences.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** Gemma 2 arXiv 2408.00118.

### 112. Context-length-aware placement (KV vs experts in VRAM)
- **What:** Decide how much VRAM goes to KV vs the expert cache from the target context depth.
- **Why / gain:** On 8 GB the tradeoff is real; report shows KV/experts are the contest. Confidence M.
- **Impl:** `autotune` + `hw.rs` probe.
- **Risk:** model.
- **Channel:** Beta.
- **Toggle:** "Balance memory between chat history and experts based on your chat length."
- **Source:** engine-report; Strata analysis (VRAM elastic).

---

## F. CPU kernels and scheduling (items 113-134)

### 113. AVX-VNNI int8 dot products (`vpdpbusd`) instead of AVX2 madd
- **What:** Replace the AVX2 int8 multiply-add reconstruction with `VPDPBUSD`/`VPDPWSSD` (VNNI on Alder
  Lake) for Q8_0/Q4_K/IQ dots.
- **Why / gain:** One instruction does 4 int8 MACs into int32; typically 1.3-2x over emulated int8 on the
  same cores. Targets **G-C** and **G-D**. Confidence H.
- **Impl:** `quant.rs` `dot_q8_0_q8`/`dot_k_q8` + new `#[target_feature(enable="avxvnni")]` variants;
  runtime dispatch on CPUID in `hw.rs`.
- **Risk:** needs per-CPU feature detection; AVX-VNNI is user-space safe.
- **Channel:** Beta.
- **Toggle:** "Use your CPU's fastest integer math for local models."
- **Source:** Intel AVX-VNNI (Intel SDM); llama.cpp AVX-VNNI path (github.com/ggml-org/llama.cpp).

### 114. Interleaved (permuted) weight layout for K-quant GEMV — the **G-D** fix
- **What:** Pre-permute Q4_K/Q5_K/Q6_K weights so the GEMV reads them in the exact order the SIMD lanes
  consume, eliminating per-step shuffles and unaligned loads.
- **Why / gain:** This is the report's "Next #4" ("interleaved weight layout for the K-quant GEMV, closing
  the in-step bandwidth gap") and the most direct attack on **G-D** (26-28 in-step vs 37-41 isolated).
  Expect the in-step number to approach the isolated one: ~1.3-1.5x on CPU experts. Confidence H.
- **Impl:** `repack.rs` (new) permuting blocks; `dot_k_q8` reads permuted bytes; loader caches the repack.
- **Risk:** extra RAM (or a repack cache); correctness must be bit-checked.
- **Channel:** Beta.
- **Toggle:** "Rearrange weights in memory once so the CPU reads them at full speed during replies."
- **Source:** engine-report "Next #4"; ik_llama.cpp interleaved quants (github.com/ikawrakow/ik_llama.cpp).

### 115. Row-split across threads matched to the interleaved layout
- **What:** Assign contiguous interleaved row-ranges per thread so each core reads a private contiguous
  stream (max hardware prefetch, no false sharing).
- **Why / gain:** Improves effective bandwidth toward the isolated figure; complements item 114. Confidence
  M.
- **Impl:** `model.rs` GEMV parallel partition + `pool.rs`.
- **Risk:** load imbalance on E-cores.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** engine-report; standard NUMA/cache practice.

### 116. P-core/E-core-aware thread pool (hybrid scheduling)
- **What:** Pin the hot decode threads to P-cores; use E-cores for prefetch/telemetry/background.
- **Why / gain:** Alder Lake P/E asymmetry means a naive pool stalls on E-cores. Avoiding 2-3 E-core
  stragglers can be worth 10-25% on CPU decode (**G-C**/**G-D**). Confidence M-H.
- **Impl:** `pool.rs` affinity (Windows `SetThreadAffinityMask`/`SetThreadIdealProcessor`); detect topology
  in `hw.rs`.
- **Risk:** topology detection on Windows (`GetLogicalProcessorInformationEx`).
- **Channel:** Beta.
- **Toggle:** "Use the fast cores for the model and keep the slow cores for background work."
- **Source:** engine.md ("rayon threads pinned to performance cores"); turbo-system.md schedulers.

### 117. Leave headroom on the last P-core for background apps / GPU feeding
- **What:** Don't saturate all P-cores; keep one for the CUDA launch thread and UI.
- **Why / gain:** Stops the launch thread being starved → smoother token cadence. Confidence M.
- **Impl:** pool thread count policy in `pool.rs`.
- **Risk:** fewer compute threads may cost a little.
- **Channel:** Beta.
- **Toggle:** "Leave a little CPU free so the GPU is always fed."
- **Source:** turbo-system.md ("fewer threads often match speed at far lower heat").

### 118. Large/huge pages for weights and KV in RAM
- **What:** Back the mmap of quantised weights with large pages (2 MB) to cut TLB misses.
- **Why / gain:** TLB pressure at 26-41 GB/s reads is real; large pages often add 5-15% on streaming
  kernels. Targets **G-C/D**. Confidence M.
- **Impl:** `gguf.rs` allocation path; enable `SeLockMemoryPrivilege` on Windows (item 170).
- **Risk:** needs a Windows privilege; fallback path.
- **Channel:** Beta.
- **Toggle:** "Use bigger memory pages so the CPU wastes less time looking things up."
- **Source:** Windows large-page docs; turbo-system (large pages idea).

### 119. Software prefetch of upcoming weight blocks
- **What:** `_mm_prefetch` the next block while computing the current one.
- **Why / gain:** Hides DRAM latency on GEMV; 5-15% at high thread counts. Confidence M.
- **Impl:** `quant.rs` dot loops; distance autotuned.
- **Risk:** over-fetch pollution.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** standard SIMD practice; llama.cpp prefetch.

### 120. Non-temporal stores / streaming loads for GEMV
- **What:** Use streaming loads where data is used once (avoids cache pollution).
- **Why / gain:** Can improve sustained bandwidth on large GEMV. Confidence L-M.
- **Impl:** `_mm256_stream_load_si128` variants in `quant.rs`.
- **Risk:** may hurt when data is reused (weights reused per token) — **weigh carefully; likely only for
  activations**.
- **Channel:** Experimental.
- **Toggle:** (internal).
- **Source:** Intel optimization manual.

### 121. CPU attention with GQA sharing + fused softmax (exists; extend)
- **What:** Attention that shares K/V across the query-head group and uses fast exp.
- **Why / gain:** Report shows CPU path improved 9.9→13.8 with these; keep. Confidence H.
- **Impl:** `model.rs` CPU attention.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** engine-report ("CPU: parallel router, GQA-shared attention, fast exp/round").

### 122. CPU batched GEMV / GEMM for multi-token prefill
- **What:** Process several tokens per weight read on the CPU (register blocking).
- **Why / gain:** CPU prefill is memory-bound; reusing each weight across k tokens gives up to k×.
  Confidence M for CPU-only prefill.
- **Impl:** `model.rs` `moe_cpu_batch` generalised to dense; register-blocked kernels in `quant.rs`.
- **Risk:** register pressure; only helps when ≥2 tokens.
- **Channel:** Beta.
- **Toggle:** "Read the model's weights once for several words at a time when using the CPU."
- **Source:** llama.cpp batched CPU prefill; standard GEMM register blocking.

### 123. LUT-based low-bit CPU kernels (T-MAC style) for ternary/IQ
- **What:** Table-lookup mpGEMM: no dequant, no multiply, just `pshufb` lookups + low-bit aggregation.
- **Why / gain:** T-MAC reports up to 4x throughput / 70% less energy vs llama.cpp on low-bit models;
  strong for ternary and tiny IQ. Targets **G-C**. Confidence M (needs the tables built).
- **Impl:** new `lut.rs` kernels; `quant.rs` dispatch for supported formats.
- **Risk:** only for low-bit formats; table build cost.
- **Channel:** Experimental.
- **Toggle:** "Run low-bit models on the CPU using fast table lookups."
- **Source:** T-MAC arXiv 2407.00088 (EuroSys 2025).

### 124. Fast f16↔f32 via F16C/FMA
- **What:** Use hardware half conversions in the dequant path.
- **Why / gain:** Removes software f16 conversion overhead; few %. Confidence M.
- **Impl:** `quant.rs` `f16_to_f32`.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** Intel F16C.

### 125. Fast vectorised exp/approx softmax (exists; guard accuracy)
- **What:** Polynomial/table exp for attention softmax on CPU.
- **Why / gain:** Report shows it helped; keeps accuracy within tolerance. Confidence M.
- **Impl:** `model.rs` CPU softmax.
- **Risk:** accuracy; keep a reference test.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** engine-report.

### 126. Unroll the K-quant inner loop to 8-16 elements with dual accumulation chains
- **What:** Multiple independent accumulators to hide FMA/int latency.
- **Why / gain:** 5-15% on GEMV. Confidence M.
- **Impl:** `quant.rs` dot kernels.
- **Risk:** code size.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** standard ILP practice; ik_llama.cpp.

### 127. Avoid the in-step bandwidth drop by interleaving expert compute with attention
- **What:** Overlap the CPU expert GEMV for layer i with the CPU attention of the same/next layer to keep
  memory pipes full (the isolated vs in-step gap may be partly latency bubbles).
- **Why / gain:** Direct attack on **G-D** (the 37-41 vs 26-28 gap). Confidence M.
- **Impl:** double-buffered phase schedule in `model.rs`; `pool.rs` barriers.
- **Risk:** register/memory port contention.
- **Channel:** Experimental.
- **Toggle:** (advanced; automatic).
- **Source:** engine-report ("Unexplained gap under investigation").

### 128. Thread pool that yields instead of spinning (exists)
- **What:** Keep the fixed 1.9→11.4 tok/s fix.
- **Why / gain:** Already proven; keep as the default pool. Confidence H.
- **Impl:** `pool.rs`.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** engine-report ("CPU worker pool that yields instead of spinning").

### 129. ISA-tier dispatch (AVX2 / AVX-VNNI / AVX-512 / NEON)
- **What:** Compile several kernel variants and pick at runtime from CPUID.
- **Why / gain:** Correct on both machines and future CPUs; no downside. Confidence H.
- **Impl:** `hw.rs` CPUID + `quant.rs` dispatch.
- **Risk:** build matrix.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** turbo-system.md (ISA tiers).

### 130. Fixed memory pool / arena for activations (avoid per-step alloc)
- **What:** Reuse activation buffers instead of allocating per token.
- **Why / gain:** Removes allocator jitter and TLB churn. Confidence M.
- **Impl:** `model.rs` `MoeScratch` extended to all buffers.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** `MoeScratch` in `model.rs`.

### 131. Higher timer resolution / avoid throttling during decode
- **What:** Raise timer resolution so Windows doesn't sleep threads mid-decode.
- **Why / gain:** Small but real for low-latency bursts. Confidence L-M.
- **Impl:** `timeBeginPeriod(1)` equivalent in a Windows module.
- **Risk:** power.
- **Channel:** Beta.
- **Toggle:** "Stop Windows from pausing the CPU during replies."
- **Source:** turbo-system.md ("timer resolution").

### 132. NUMA-awareness (prep for multi-socket; single-socket today)
- **What:** Allocate weights/threads on the local node.
- **Why / gain:** None on these laptops; logged for completeness/future. Confidence H (no-op here).
- **Impl:** `hw.rs` NUMA probe; `pool.rs` node affinity.
- **Risk:** n/a.
- **Channel:** Experimental.
- **Toggle:** (hardware-gated).
- **Source:** turbo-system.md ("NUMA placement").

### 133. Fuse rmsnorm + quant + RoPE on CPU into one pass
- **What:** Combine the small per-token elementwise ops.
- **Why / gain:** Reduces memory round-trips; few %. Confidence M.
- **Impl:** `model.rs` CPU graph.
- **Risk:** low.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** engine ("10 fused kernels per layer" on GPU, mirrored on CPU).

### 134. Batched CPU expert execution across tokens (MoE decode)
- **What:** Group tokens routed to the same CPU expert and run one GEMV per expert for the batch.
- **Why / gain:** Reuses each weight across tokens → directly helps **G-D** in decode. Confidence M.
- **Impl:** `moe_cpu_batch` in `model.rs`.
- **Risk:** sort overhead at tiny batch.
- **Channel:** Beta.
- **Toggle:** "Process all the words using the same expert in one CPU pass."
- **Source:** llama.cpp MoE batched CPU; engine-report MoE prefill batching.

---

## G. Memory and I/O (items 135-146)

### 135. Memory-map tuning and optional pre-lock of hot regions
- **What:** Tune mmap advice and lock the hottest weight regions in RAM (not the whole file).
- **Why / gain:** Avoids page-fault stalls mid-decode (report: touching pages at load gave 30.3→36.2
  tok/s). Confidence H.
- **Impl:** `gguf.rs` `touch`/`prefetch`/`release` (already present) extended to a hot-region list.
- **Risk:** memory pressure; Windows `VirtualLock` limits.
- **Channel:** Beta.
- **Toggle:** "Load the model's hot parts into memory ahead of time so replies don't stutter."
- **Source:** engine-report ("Touching CPU-side expert pages at load"); gguf.rs.

### 136. Pinned host staging buffers sized to the copy stream
- **What:** Double/triple-buffered pinned buffers for expert/KV transfers.
- **Why / gain:** Full PCIe bandwidth and overlap. Confidence H (mostly present).
- **Impl:** `gpu.rs` `PinnedHostSlice`.
- **Risk:** pinned memory exhaustion.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** CUDA pinned-memory docs.

### 137. SSD streaming with large contiguous reads (LLM-in-a-Flash)
- **What:** Read active expert/neuron rows from NVMe in large sequential chunks.
- **Why / gain:** 4-25x over naive loading; enables >RAM models. Confidence M.
- **Impl:** `gguf.rs` read path + item 92.
- **Risk:** SSD random-read latency; sequencing.
- **Channel:** Experimental.
- **Toggle:** "Stream the parts of huge models you need from your SSD."
- **Source:** LLM in a Flash arXiv 2312.11514.

### 138. Row-column bundling of neuron/expert weights
- **What:** Store the rows most often read together adjacently so one read serves several.
- **Why / gain:** Part of LLM-in-a-Flash; turns many small reads into few large ones. Confidence M.
- **Impl:** offline layout tool + `gguf.rs` reader.
- **Risk:** format support; benefit depends on activation sparsity.
- **Channel:** Experimental.
- **Toggle:** "Store weights in the order they're used so the disk reads fewer, bigger chunks."
- **Source:** LLM in a Flash 2312.11514.

### 139. Windowing to reuse recently active neurons/experts
- **What:** Track recent activation set; keep predicted next-set resident.
- **Why / gain:** Reduces disk/PCIe traffic; part of LLM-in-a-Flash. Confidence M.
- **Impl:** predictor in `gpu.rs`/`autotune` (shares with items 70-71).
- **Risk:** prediction accuracy.
- **Channel:** Experimental.
- **Toggle:** "Reuse the same parts you just used instead of reloading them."
- **Source:** LLM in a Flash 2312.11514; PowerInfer arXiv 2312.12456.

### 140. DirectStorage / GPUDirect Storage for NVIDIA + NVMe on Windows
- **What:** Bypass the CPU bounce buffer, DMA directly NVMe→VRAM via DirectStorage.
- **Why / gain:** Removes a copy and CPU involvement for >RAM streaming; potentially large for SSD-tier
  models. Confidence M (Windows/driver support is the caveat).
- **Impl:** a Windows/DirectStorage backend behind the `Device` trait; new module.
- **Risk:** API/driver availability and complexity; **UNVERIFIED** whether it works with this laptop's NVMe.
- **Channel:** Experimental.
- **Toggle:** "Send model data straight from your SSD to the GPU when streaming huge models."
- **Source:** Microsoft DirectStorage docs; NVIDIA GPUDirect Storage docs.

### 141. Overlapped (async) file I/O for SSD streaming
- **What:** Windows overlapped I/O (or io_uring on Linux) so disk reads don't block the step.
- **Why / gain:** Overlaps read with compute. Confidence M.
- **Impl:** `gguf.rs` reader on a background thread feeding pinned buffers.
- **Risk:** complexity.
- **Channel:** Experimental.
- **Toggle:** "Read from disk in the background while the model keeps working."
- **Source:** Windows overlapped I/O docs; io_uring (Linux).

### 142. PrefetchVirtualMemory + aggressive read-ahead at load
- **What:** Use the Windows prefetch API for the next layer's ranges (already hinted in gguf.rs).
- **Why / gain:** Reduces fault latency in the first tokens. Confidence M.
- **Impl:** `gguf.rs` `PrefetchVirtualMemory` (present) driven by a layer schedule.
- **Risk:** over-prefetch wastes bandwidth.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** gguf.rs; Windows `PrefetchVirtualMemory`.

### 143. UMA/zero-copy weights for machine B (Radeon 780M)
- **What:** On integrated/unified memory, map weights once and let both CPU and GPU use them.
- **Why / gain:** Removes duplicate copies on the 780M; important for machine B support. Confidence M.
- **Impl:** `gpu.rs` backend abstraction; iGPU path.
- **Risk:** needs ROCm/Vulkan backend (not yet built).
- **Channel:** Experimental.
- **Toggle:** "On integrated graphics, share one copy of the model between CPU and GPU."
- **Source:** engine.md (Vulkan/ROCm later).

### 144. Compressed on-disk weight/KV cache
- **What:** Store the engine's own repacked/compressed weights and dormant KV compressed.
- **Why / gain:** Faster reload, smaller disk. Confidence L-M (mostly load-time).
- **Impl:** cache format in `repack.rs`.
- **Risk:** decompression cost.
- **Channel:** Experimental.
- **Toggle:** (advanced).
- **Source:** this catalogue (repack cache); LLM-in-a-Flash.

### 145. Prefetch weights into L2/L3 before a decode burst
- **What:** Issue cache prefetch for the next layer's weights during the current layer.
- **Why / gain:** Hides DRAM latency; relates to **G-D**. Confidence L-M.
- **Impl:** interleave with item 127 schedule.
- **Risk:** cache thrash.
- **Channel:** Experimental.
- **Toggle:** (internal).
- **Source:** standard practice.

### 146. Right-size the KV/weights mmap to avoid double-buffering
- **What:** Ensure a device buffer isn't copied when an alias would do.
- **Why / gain:** Removes redundant copies on load. Confidence M.
- **Impl:** `gpu.rs` upload path.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** engine ("weights stay quantised in memory").

---

## H. Hybrid architectures — linear attention and sliding window (items 147-157)

### 147. Gated DeltaNet layers (Qwen3-Next)
- **What:** Implement the gated delta-rule linear-attention layer (Q/K/V projections, shortconv, SiLU,
  L2 norm, alpha/beta gates, delta-rule state update).
- **Why / gain:** Qwen3-Next's 3:1 linear:full mix makes long context cheap (fixed state, no KV growth).
  Required for the product's Qwen3.8 target. Performance enabler; correctness first. Confidence H (needed).
- **Impl:** new `linear_attn.rs` in the engine + kernels in `kernels.cu`; graph in `model.rs`.
- **Risk:** new math; must match reference.
- **Channel:** Beta.
- **Toggle:** "Run Qwen3.8-style models that stay fast on very long chats."
- **Source:** Gated DeltaNet arXiv 2412.06464; Qwen3-Next HF card (2025).

### 148. Chunked parallel scan for gated delta state
- **What:** Compute the recurrent state with the chunkwise parallel algorithm (matmul-heavy, not sequential).
- **Why / gain:** Sequential recurrence would kill prefill; the chunkwise form makes it tensor-core friendly.
  Confidence M-H.
- **Impl:** chunked scan kernel in `kernels.cu` / Triton-free CUDA; follow Flash Linear Attention design.
- **Risk:** numerical stability of the delta rule.
- **Channel:** Beta.
- **Toggle:** (part of Qwen3.8 support).
- **Source:** Gated DeltaNet 2412.06464; flash-linear-attention (github).

### 149. Recurrent state cache across sessions (linear-attn state paging)
- **What:** Persist each session's fixed-size recurrent state so reopening a chat resumes instantly.
- **Why / gain:** Linear-attn state is tiny; ideal for caching. Confidence M.
- **Impl:** state save/load in `Session`.
- **Risk:** correctness of "resume from state".
- **Channel:** Beta.
- **Toggle:** "Resume a Qwen3.8 chat with its memory intact."
- **Source:** Qwen3-Next / Mamba2 practice; vLLM hybrid KV manager (blog 2025-09-11).

### 150. Hybrid KV manager (linear state + full-attention KV share VRAM)
- **What:** One allocator balancing fixed linear states and paged full-attention KV so both fit the 8 GB.
- **Why / gain:** vLLM tunes block sizes so both occupy equal physical memory; avoids fragmentation.
  Confidence M.
- **Impl:** `gpu.rs` allocator (with items 15/93).
- **Risk:** tuning per model.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** vLLM Qwen3-Next blog (2025-09-11).

### 151. Sliding-window attention (Gemma family)
- **What:** Local attention with a window for most layers (Gemma-2/3 use 5:1 local:global).
- **Why / gain:** Bounds KV per layer → long context on small cards; required for Gemma 4. Confidence H
  (required); perf win M.
- **Impl:** window mask in `attn_*`; config from GGUF.
- **Risk:** correctness of the local/global interleave.
- **Channel:** Beta.
- **Toggle:** "Run Gemma-style models efficiently with mostly local attention."
- **Source:** Gemma 2 arXiv 2408.00118; Gemma 3 arXiv 2503.19786.

### 152. Reuse of global-attention layers' KV only
- **What:** Only global layers keep long-range KV; local layers drop beyond the window.
- **Why / gain:** Halves KV memory vs full attention everywhere → more experts resident. Confidence M.
- **Impl:** per-layer KV budget in `gpu.rs`.
- **Risk:** low.
- **Channel:** Beta.
- **Toggle:** (part of sliding-window support).
- **Source:** Gemma 2/3.

### 153. Mamba-2 SSD kernel (optional linear layers)
- **What:** Semi-separable state-space (Mamba-2) block kernel.
- **Why / gain:** Some hybrids (and machine-B interest) use Mamba-2. Confidence M (optional).
- **Impl:** `mamba2.rs` + kernels.
- **Risk:** extra architecture surface.
- **Channel:** Experimental.
- **Toggle:** "Run Mamba-style models."
- **Source:** Mamba-2 arXiv 2405.21060.

### 154. Short-convolution kernel for linear layers
- **What:** Depthwise short conv applied before Q/K/V in Gated DeltaNet blocks.
- **Why / gain:** Needed for the architecture; cheap. Confidence H (required).
- **Impl:** fused in `linear_attn.rs`.
- **Risk:** low.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** Qwen3-Next card; Gated DeltaNet 2412.06464.

### 155. Hybrid-aware MoE + linear attention scheduling
- **What:** Co-schedule the MoE FFN and the linear-attn block so expert prefetch overlaps the scan.
- **Why / gain:** Reuses the overlap machinery (items 21/69) for the new architecture. Confidence M.
- **Impl:** `gpu.rs` step for Qwen3.8.
- **Risk:** complexity.
- **Channel:** Experimental.
- **Toggle:** (internal).
- **Source:** Qwen3-Next (MoE + hybrid); engine.md M6.

### 156. MTP support for Qwen3-Next
- **What:** Run the Qwen3-Next MTP head (item 49) for the hybrid model.
- **Why / gain:** The model ships MTP for speed; big decode gain on the target model. Confidence M.
- **Impl:** graph + verify; shares item 49.
- **Risk:** acceptance.
- **Channel:** Beta.
- **Toggle:** "Use Qwen3.8's built-in look-ahead to answer faster."
- **Source:** Qwen3-Next card; DeepSeek-V3 2412.19437.

### 157. Controlled-forgetting / gating ratio guidance in placement
- **What:** Use the hybrid ratio (3:1 etc.) to decide memory and kernel choice.
- **Why / gain:** Practical: tells the tuner which layers dominate. Confidence M.
- **Impl:** `autotune` + `model.rs` config.
- **Risk:** low.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** hybrid survey arXiv 2507.06457.

---

## I. Batching and serving (items 158-167)

### 158. Continuous (iteration-level) batching
- **What:** Several conversations decode in one pass; engine.md §6.
- **Why / gain:** Throughput for multi-chat use; lets the GPU do useful work. Confidence H.
- **Impl:** multi-slot in `gpu.rs` + server route.
- **Risk:** tail latency.
- **Channel:** Beta.
- **Toggle:** "Answer several chats at once so each one is faster overall."
- **Source:** Orca continuous batching; engine.md §6.

### 159. Stall-free chunked-prefill scheduling (Sarathi-Serve)
- **What:** Coalesce decode + a bounded prefill chunk so new requests join without stalling decodes.
- **Why / gain:** 2.6-5.6x serving capacity at latency SLO reported. Confidence M.
- **Impl:** scheduler module (new) over `gpu.rs` batch.
- **Risk:** complexity.
- **Channel:** Experimental.
- **Toggle:** "Add new questions without slowing the replies already running."
- **Source:** Sarathi-Serve arXiv 2403.02310.

### 160. Prefix-aware request scheduling
- **What:** Schedule requests to maximise radix-cache reuse.
- **Why / gain:** Big win for agent traffic (shared system prompts). Confidence M.
- **Impl:** scheduler + item 94.
- **Risk:** scheduling complexity.
- **Channel:** Beta.
- **Toggle:** "Group similar requests so shared context is computed once."
- **Source:** SGLang arXiv 2312.07104.

### 161. Interactive-first queue policy
- **What:** Prioritise interactive over batch jobs (engine.md "interactive requests first").
- **Why / gain:** Latency for the human. Confidence H.
- **Impl:** queue in the serving module.
- **Risk:** starvation of batch.
- **Channel:** Stable.
- **Toggle:** "Always answer the chat you're waiting on before background work."
- **Source:** turbo-system.md schedulers.

### 162. Multi-slot paged KV with copy-on-write prefixes
- **What:** Slots share prefix pages until they diverge.
- **Why / gain:** Memory efficiency for many similar chats. Confidence M.
- **Impl:** paged KV + refcounts in `gpu.rs`.
- **Risk:** lifetime bugs.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** PagedAttention 2309.06180.

### 163. Preemption: swap vs recompute decision
- **What:** Under memory pressure, swap a request's KV to RAM or recompute (see item 110).
- **Why / gain:** Keeps throughput when slots exceed VRAM. Confidence M.
- **Impl:** scheduler.
- **Risk:** recompute cost.
- **Channel:** Experimental.
- **Toggle:** (server automatic).
- **Source:** vLLM preemption; Orca.

### 164. Request-level CUDA graph (capture per batch shape)
- **What:** Capture graphs keyed by batch size/slot count.
- **Why / gain:** Keeps launch overhead low as batches vary. Confidence M.
- **Impl:** `gpu.rs` graph cache keyed by shape.
- **Risk:** re-capture cost; graph memory.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** item 22; CUDA graphs.

### 165. Micro-batching of MoE experts across requests
- **What:** Batch experts from different requests into shared grouped GEMMs.
- **Why / gain:** Bigger per-expert batches → better GPU utilisation. Confidence M.
- **Impl:** with items 9/87.
- **Risk:** complexity.
- **Channel:** Experimental.
- **Toggle:** (server automatic).
- **Source:** MoE-Lightning 2411.11217; grouped GEMM.

### 166. Autotuned batch size / token budget
- **What:** Pick batch size and token budget from measured throughput/latency.
- **Why / gain:** Matches workload; avoids paging. Confidence M.
- **Impl:** `autotune`.
- **Risk:** sweep cost.
- **Channel:** Beta.
- **Toggle:** (advanced).
- **Source:** Sarathi-Serve 2403.02310; turbo-system schedulers.

### 167. Response streaming backpressure and cancel
- **What:** Cancel and free KV promptly on user stop.
- **Why / gain:** Frees VRAM for others; report notes quiet cancel. Confidence M.
- **Impl:** server + `Session` teardown.
- **Risk:** low.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** Strata analysis (quiet cancel).

---

## J. Windows-specific tuning (items 168-176)

### 168. Thread affinity + priority on Windows hybrid CPUs
- **What:** Explicit affinity masks and `THREAD_PRIORITY_HIGHEST` for decode threads; E-core affinity for
  background.
- **Why / gain:** Alder Lake scheduling on Windows 11 can mis-place threads; explicit affinity recovers
  the P-core advantage (feeds **G-C/D**). Confidence M-H.
- **Impl:** Windows module + `pool.rs`.
- **Risk:** must detect topology robustly; revert on failure.
- **Channel:** Beta.
- **Toggle:** "Pin the model to your fastest CPU cores."
- **Source:** turbo-system.md ("P-core/E-core pinning, process priority class").

### 169. Power-plan / processor-boost control
- **What:** Switch to High Performance and set min/max processor state during inference; restore on exit.
- **Why / gain:** Prevents downclocking mid-decode; measured per-machine. Confidence M.
- **Impl:** `powercfg`-equivalent + restoration; app-level.
- **Risk:** heat/battery; auto-revert if it doesn't help (turbo-system rule).
- **Channel:** Beta.
- **Toggle:** "Tell Windows to run the CPU at full speed while generating."
- **Source:** turbo-system.md §3 machine profiles.

### 170. Large-page support via SeLockMemoryPrivilege
- **What:** Enable the privilege so item 118 can allocate 2 MB pages.
- **Why / gain:** Enabler for the TLB win. Confidence M.
- **Impl:** token adjust in the Windows module.
- **Risk:** requires admin/privilege; fallback.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** Windows large-page docs.

### 171. NVML power/clock/fan profiles (measured, auto-revert)
- **What:** Set GPU power limit/clock offsets via NVML within vendor range; benchmark each change.
- **Why / gain:** On a power-limited 80/115 W laptop GPU, correct power limit can raise sustained clocks;
  turbo-system warns clock offsets can *lower* speed, so measure and revert. Confidence M.
- **Impl:** NVML module (Strata uses ctypes NVML; we can use the driver API); app-level.
- **Risk:** instability/heat; must restore on exit/crash.
- **Channel:** Beta.
- **Toggle:** "Let Cerebro tune your GPU's power for the best speed, and undo it if it doesn't help."
- **Source:** turbo-system.md §3.

### 172. TDR (Timeout Detection & Recovery) awareness
- **What:** Avoid configs that trip the Windows GPU timeout on long kernels; optionally extend it.
- **Why / gain:** Prevents driver resets during long prefill/stream-K. Confidence M.
- **Impl:** keep kernels bounded; document the registry knob.
- **Risk:** registry changes are advanced.
- **Channel:** Experimental.
- **Toggle:** (advanced; off).
- **Source:** Windows TDR docs.

### 173. Hardware-accelerated GPU scheduling (HAGS) awareness
- **What:** Detect HAGS state; it changes launch latency characteristics.
- **Why / gain:** Informational/tuning; small. Confidence L.
- **Impl:** registry read.
- **Risk:** low.
- **Channel:** Experimental.
- **Toggle:** (informational).
- **Source:** Windows HAGS docs.

### 174. Working-set / memory-pressure handling
- **What:** Trim Cerebro's working set when idle so the OS doesn't page out weights.
- **Why / gain:** Avoids page-fault stalls (**G-D** adjacent). Confidence M.
- **Impl:** Windows module; `gguf.rs` release.
- **Risk:** trimming hot pages hurts.
- **Channel:** Beta.
- **Toggle:** "Ask Windows not to page the model out of memory."
- **Source:** Windows memory docs; gguf.rs `release`.

### 175. Timer resolution for low-latency bursts
- **What:** See item 131 on Windows.
- **Impl/notes:** duplicate intentionally (Windows-specific), restore on exit.
- **Confidence:** L-M.
- **Channel:** Beta.
- **Toggle:** (internal).
- **Source:** turbo-system.md.

### 176. Exclusive fullscreen / foreground priority for the app
- **What:** Raise process priority when the user is waiting; lower when background.
- **Why / gain:** Smoother tokens; small. Confidence L.
- **Impl:** app-level; `pool.rs` priority.
- **Risk:** low.
- **Channel:** Beta.
- **Toggle:** "Give Cerebro more CPU when you're waiting on an answer."
- **Source:** turbo-system.md ("background efficiency mode for idle").

---

## K. Auto-tuner methodology (items 177-186)

### 177. Per-machine ISA-tier build + install-time probe
- **What:** Ship AVX2 / AVX-VNNI / AVX-512 / NEON variants and pick by CPUID at install.
- **Why / gain:** Correct + fast on every machine; no downside. Confidence H.
- **Impl:** `hw.rs` probe + build feature flags; download prebuilt tiers.
- **Risk:** build matrix size.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** turbo-system.md §1.

### 178. Kernel-config autotune (tile/occupancy/stage)
- **What:** Sweep kernel hyperparameters on first run per GPU.
- **Why / gain:** 10-30% over a hardcoded config. Confidence M.
- **Impl:** `autotune` (new) writing to the machine profile.
- **Risk:** first-run cost.
- **Channel:** Beta.
- **Toggle:** "Test a few GPU settings once and keep the fastest."
- **Source:** TVM/Ansor 2006.06762; item 26.

### 179. Placement autotune (VRAM / RAM / SSD / split)
- **What:** Search layer/expert placements to maximise measured speed (engine.md placement matrix).
- **Why / gain:** The single biggest knob for >VRAM models; automated = always-best defaults.
  Confidence H.
- **Impl:** `autotune` + `gpu.rs` placement; results logged to `docs/measurements/engine/runs.jsonl`.
- **Risk:** search time.
- **Channel:** Beta.
- **Toggle:** "Find the best way to spread the model across GPU, RAM and disk for your machine."
- **Source:** engine.md placement matrix; big-models research 2026-10-10.

### 180. Prefill chunk / ub / b sweep
- **What:** Sweep chunk size and micro-batch for prefill throughput and latency.
- **Why / gain:** Report shows chunking moved prefill 33→205 tok/s; keep tuning. Confidence H.
- **Impl:** `autotune`.
- **Risk:** overfitting to a prompt set.
- **Channel:** Stable.
- **Toggle:** "Pick the fastest prompt-processing batch size for your machine."
- **Source:** engine-report; Strata (8,192 chunks).

### 181. Expert-cache size sweep
- **What:** Find the VRAM expert-cache size that maximises decode speed (vs KV tradeoff, item 112).
- **Why / gain:** Directly tunes **G-B**. Confidence M-H.
- **Impl:** `autotune` + `gpu.rs`.
- **Risk:** search time; content dependence.
- **Channel:** Beta.
- **Toggle:** "Fill spare GPU memory with the experts that help most."
- **Source:** Strata analysis ("every extra GB holds ~700 experts").

### 182. KV-type sweep (f16/q8/q4 × context depth)
- **What:** Sweep KV formats and depths for best speed/quality (measure quality, not just speed).
- **Why / gain:** Picks the safe-fast point automatically. Confidence M.
- **Impl:** `autotune` + a small perplexity/KL check.
- **Risk:** quality measurement needs calibration data.
- **Channel:** Beta.
- **Toggle:** "Balance answer quality and speed automatically for long chats."
- **Source:** engine.md placement matrix; KIVI 2402.02750.

### 183. Measurement hygiene (median-of-N, warm-up, noise floor)
- **What:** Codify the report's method: one warm-up, 3 timed runs, median; fresh process for first-turn;
  report variance.
- **Why / gain:** Prevents tuning to noise; foundation for "never slower than the previous milestone".
  Confidence H.
- **Impl:** `examples/bench.rs` + runs.jsonl.
- **Risk:** none.
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** engine-report method section.

### 184. Shipped measured defaults per popular model
- **What:** Bundle a `tuned.json` per model (the repo already stores `*__tuned*.json`).
- **Why / gain:** Users get the tuned config without waiting for a local sweep. Confidence H.
- **Impl:** per-model profile store; consumed by `autotune` as the starting point.
- **Risk:** machine dependence (override by local sweep).
- **Channel:** Stable.
- **Toggle:** "Start with settings measured to be fast for this model."
- **Source:** `docs/measurements/` (`*__tuned*.json`), engine-report.

### 185. Continuous regression gate in CI
- **What:** A benchmark set that must not regress per commit (engine rule "never slower than previous
  milestone").
- **Why / gain:** Stops silent slowdowns. Confidence H.
- **Impl:** CI running `examples/bench.rs` on a fixed prompt set; compare to baseline.
- **Risk:** CI hardware variance (use the dev box or a recorded baseline).
- **Channel:** Stable.
- **Toggle:** (internal).
- **Source:** engine.md rules; Strata "evidence culture".

### 186. Profile-guided, hardware-tiered releases (PGO/LTO where measured)
- **What:** PGO/LTO builds per ISA tier, only if they beat the baseline build on this machine.
- **Why / gain:** Free single-digit %; turbo-system rule keeps only wins. Confidence M.
- **Impl:** build pipeline.
- **Risk:** build complexity.
- **Channel:** Experimental.
- **Toggle:** "Use a build tuned for your exact CPU."
- **Source:** turbo-system.md §1 ("LTO/PGO where it measures faster"; a self-built fork was once 3-5x slower).

---

## Ranked "do next" — top 25 by gain/effort for this engine (Machine A)

Ordering is for `cerebro-engine` as it exists today (CPU + CUDA, own kernels) against the four measured
gaps. **G-A** dense GPU prefill 10x slow; **G-B** MoE first-turn decode 19-23 vs 33-36; **G-C** CPU decode
13.8 vs 18.5; **G-D** CPU expert in-step 26-28 vs isolated 37-41 GB/s. Effort = S/M/L.

| # | Do | Item(s) | Gap | Expected | Effort | Why first |
|---|---|---|---|---|---|---|
| 1 | int8 tensor-core prefill (m16n8k32) with int8 activations | 1, 36 | G-A | 3-7x prefill | L | The report names mma absence as the cause of the ~10x prefill gap; biggest single number on the board |
| 2 | fp16 tensor-core prefill as the accuracy-safe first step | 2, 11 | G-A | 2.5-5x prefill | M | Lower risk than int8; ships quality-neutral before int8 lands |
| 3 | Offline weight repack into mma fragment layout | 3, 4, 19 | G-A | enables 1/2 | M | Without it the mma kernels stall on smem; the other GPU work depends on it |
| 4 | Async expert swaps on the copy stream (many per token) | 69 | G-B | 19-23→30+ tok/s | M | Report "Next #1"; removes first-turn swap stalls directly |
| 5 | Interleaved K-quant weight layout for CPU GEMV | 114 | G-D, G-C | in-step 26-28→~37 GB/s | M | Report "Next #4"; named fix for the in-step bandwidth gap |
| 6 | AVX-VNNI int8 dots with CPUID dispatch | 113, 129 | G-C, G-D | 1.3-2x CPU dots | S | Small, safe, high-confidence CPU win on Alder Lake |
| 7 | P-core pinning + E-core background split | 116, 117 | G-C, G-D | 10-25% CPU decode | S | Cheap and removes E-core stragglers on the i9-12900H |
| 8 | Shipped per-model expert starting profile | 68 | G-B | removes cold first turn | S | Report "Next #1"; immediate, no runtime risk |
| 9 | Cross-layer gate-input expert prefetch (Fate) | 70, 74 | G-B | ~99% hit | M | Prediction for free on the CPU; sustains the cache after turn 1 |
| 10 | Reuse batched prefill as the verify path + MTP | 49, 57 | G-B | 1.5-1.9x | M | MTP measured 1.6-1.8x on this PC; amortises expert loads |
| 11 | Stream-K work decomposition for prefill GEMM | 6 | G-A | +10-30% | M | Tail-wave loss is real with 48 SMs; complements mma |
| 12 | CPU large pages for weights/KV | 118, 170 | G-C, G-D | +5-15% | M | TLB pressure at 26-41 GB/s reads |
| 13 | Fused epilogue (scale/SwiGLU/requant) | 8 | G-A, G-B | +5-15% | M | Removes gmem round-trips between GEMM and activation |
| 14 | Grouped GEMM + on-GPU expert token bucketing | 9, 10, 87 | G-B | +20-50% MoE prefill | L | Makes MoE prefill efficient; feeds item 65 |
| 15 | Continuous batching + multi-slot paged KV | 158, 162 | serving | throughput | L | Multi-chat throughput; needed for the harness story |
| 16 | Flash-decoding tuning (split-KV by context) | 14 | long ctx | 1.5-3x attention decode | S | Already half-built (`attn_partial`/`attn_combine`); tune splits |
| 17 | q8 KV default (paged) | 30, 93 | G-B | +experts resident | M | Halves KV; frees VRAM for experts on the 8 GB card |
| 18 | Placement autotune (VRAM/RAM/SSD/split) | 179, 181, 180 | all | best configs auto | M | Turns the report's manual matrix into shipped defaults |
| 19 | Prompt-aware warm start (extend) | 67, 80 | G-B | 23.7→55.3 path | S | Already partly built; stack with shipped profile (item 8) |
| 20 | Softmax/loop prefetch + fused CPU rmsnorm/RoPE/quant | 119, 133 | G-C, G-D | +5-15% | S | Low-risk CPU micro-optimisations |
| 21 | FA2-style + GQA-packed prefill attention | 11, 12 | G-A | +30-100% attn prefill | M | Attention share of prefill grows with prompt length |
| 22 | Qwen3-Next support (Gated DeltaNet + hybrid KV) | 147, 148, 150, 156 | product | long-context speed | L | M6 target model; correctness-first, then speed |
| 23 | SSD streaming + windowing for >RAM MoE | 92, 137, 139 | big models | 4-25x vs naive | L | The "big models on small machines" story |
| 24 | SpecExec tree speculation over offloaded target | 56, 63 | G-B | up to 10x offload | L | Purpose-built for the 8 GB card on >VRAM dense models |
| 25 | Continuous regression gate + shipped tuned defaults | 183, 184, 185 | process | protects every gain | S | Keeps the engine "never slower than the previous milestone" |

### Quick notes on the ranking

- Items 1-3 are one project: they must land together or none helps. That is why they occupy three slots.
- The CPU gap has a cheap half (items 5-7, 12, 20: layout + VNNI + pinning + pages + prefetch) that should
  be done before the expensive CPU-batching work; the report's in-step bandwidth gap is most likely layout
  + page-fault + scheduling, not raw ISA.
- Items 4, 8, 9, 10, 19 are the first-turn MoE project (**G-B**); they are complementary and mostly
  independent of the GPU mma work, so they can proceed in parallel with items 1-3.
- Items 14-16 and 22-24 are larger product bets (MoE prefill efficiency, serving, new architectures,
  big-model streaming); they are ranked after the four-gap fixes but are the ones the product roadmap
  (engine.md M3-M6) will need.

### Verification status of sources

Primary sources were checked against arXiv/abstract pages for: Marlin (2408.11743), Machete (Red Hat/
vLLM PR), QTIP (2406.11235), FA3 (2407.08608), Sarathi-Serve (2403.02310), EAGLE-3 (2503.01840), KVQuant
(2401.18079), KIVI (2402.02750), SnapKV (2404.14469), Gated DeltaNet (2412.06464), T-MAC (2407.00088),
HOBBIT (2411.01433), ProMoE (2410.22134), AdapMoE (2408.10284), Fate (2502.12224), HybriMoE (2504.05897),
DAOP (2501.10375), MoE-Lightning (2411.11217; ASPLOS DOI 10.1145/3669940.3707267), KTransformers (SOSP
2025 DOI 10.1145/3731569.3764843), SP-MoE (2510.10302), SpecMoEOff (2508.21706), Unsloth Dynamic v2.0,
BitNet v2 (2504.18415), hybrid survey (2507.06457). Items carrying an explicit **UNVERIFIED** tag
(source id or claim not confirmed against a primary page) are: 6 (stream-K arXiv id), 13 (FlashInfer id),
35 (SpinQuant id), 39 (NVFP4), 45 (EXL3), 48 (imatrix), 54 (SuffixDecoding id), 90 (HC-SMoE id), 107
(YaRN id), 140 (DirectStorage on this laptop), and the general "work in progress" HF model cards.
All expected-gain figures are estimates for Machine A with the stated confidence and must be replaced by
measured numbers before any public claim (docs/CLAIMS.md).

