# Review brief: make Cerebro faster than llama.cpp everywhere, and find the bugs

Reviewers (including `/code-review ultra`): this is the owner's standing request. Besides correctness bugs,
look hard for **speed** problems and propose concrete fixes. Cerebro is a Tauri desktop app (`app` branch,
`apps/desktop`) with its own Rust/CUDA inference engine (`engine` branch, `crates/cerebro-engine`). The bar:
**beat llama.cpp at its best settings on every benchmark** on the reference laptop (i9-12900H, RTX 3070 Ti
Laptop 8 GB at 6001 MHz memory clock = ~384 GB/s real peak, 32 GB RAM, Windows 11, WDDM). Our own kernels only:
no cuBLAS, no llama.cpp code paths for NVIDIA.

## Where we lose today (512-token prompt, read / write tok/s; llama.cpp at its best -ngl)

| Model | Cerebro | llama.cpp best |
|---|---|---|
| Qwen2.5 14B Q4_K_M (split) | 829 / 24.7 | **1070** / 24.5 (-ngl 44) |
| Gemma 4 E4B Q4_K_M (whole GPU) | 2806 / 81.9 | **3369** / 80.9 |
| Bonsai 27B Q1_0 (whole GPU, DeltaNet) | 575 / 53.4 | **705** / 40.9 |
| DeepSeek-V4.1 IQ1_M (processor only) | ~1.6 / 1.6 | llama.cpp b10751 cannot load it |

Wins today: Qwen3.8-27B (371/4.45 vs 279/4.04), DavidAU 27B (424/4.64 vs 300/4.00), and Coder-Next,
Flash-Next and Llama-70B (llama.cpp runs out of memory). Always compare against an **-ngl sweep** of llama.cpp,
never a single flag set: its fastest setting places the last N layers and the output head on the GPU and
slightly over-commits VRAM (Windows pages the excess to shared memory).

## Speed work, highest value first

1. **Prompt reading: the int8 tensor-core GEMM** (`kernels_mma.cu`, `matmul_q8_mma_x`, used via `gpu.rs` and
   `gpu_g4.rs` `gemm`). It runs at ~40 TOPS, ~20% of the int8 peak, on every model. Likely cause: every
   m16n8k32 MMA is followed by i2f + scale multiply + FMA for each output (weights and activations both
   carry one scale per 32 values), so the FP pipe, not the tensor cores, is the limit. Ideas to evaluate:
   fewer FP ops per MMA (hoisted or packed scale products), deeper cp.async pipelining, tile shapes per
   matrix size, overlapping `k_to_q8` (Q4_K -> int8 conversion, ~11% of Gemma's prompt time) with the
   previous GEMM on a second stream. The fused Q4_K path (`mmq_q4k_x`, behind `CEREBRO_MMQ`) measured
   slower. Prove any change with identical greedy output and an nsys kernel table.
2. **DeltaNet prompt reading is sequential per token** (`delta_rule128` in `kernels.cu`, ~1 ms/layer for
   512 tokens on Bonsai). A chunked (WY / chunk-parallel gated delta rule) algorithm, as llama.cpp and FLA
   use, should cut it several-fold. Keep `delta_rule` as the reference; outputs must match.
3. **DeepSeek-V4.1 has no GPU path** (`ds41.rs`, processor only, one token at a time, ~1.6 tok/s;
   prompts are read token by token too). Candidates: the dense per-layer matrices (q_a, q_b, kv, wo_a,
   wo_b, shared expert, router; ~5 GB) on the card, an expert cache in VRAM like `gpu.rs` `load_moe`,
   batched prompt reading, attention on the GPU. Expert weights (IQ1_S/IQ1_M/IQ2) stream from the SSD:
   check `prefetch` / page-cache behaviour.
4. **Split models' VRAM footprint** (`gpu.rs` `load_with`, `placement.rs`): Q6_K is widened to a 224-byte
   GPU block (+6.7%), prompt conversion scratch ~180 MB, the streaming pipe's two int8 layer slots
   ~250 MB+. llama.cpp fits 44 layers + head of Qwen2.5-14B in 7.9 GB; we need the spill allowance
   (`placement::spill_allowance`, a cube-law heuristic fitted on two models) plus an evictable head to get
   there, and prompt reading suffers. Native 210-byte Q6_K and per-matrix (not per-layer) streaming would
   free room.
5. **Decode kernels are near peak** (Q4_K `matvec_k` ~83% of 384 GB/s); small fixed costs per layer
   (Gemma: 15 launches/layer; attention ~26 us/layer at short context) are what is left there.

## Bugs and risks to check

- **DeepSeek chat turns re-read the whole conversation**: `Session::feed` for `ds41` only continues an
  exact token prefix, otherwise it resets. If the app re-renders earlier turns differently from what was
  generated (thinking dropped, `</think>` added), every turn re-reads everything at ~1.7 tok/s.
- `cerebro_engine::STOP` is a process-wide static: two sessions (Arena, cluster) can stop each other;
  it is cleared at job start, so a stop pressed while a job is queued is lost.
- `evict_head` / `head_split` re-upload and `pipe` slot release (`gpu.rs`): ordering with the copy stream,
  and what happens if the re-upload allocation fails.
- The spill retry in `Session::new_gpu_with` sets `SPILL_OFF` for the whole process after one failure.
- `ds41.rs` candidate-block selection (contexts > 16K tokens) is implemented but untested; the Engram row id
  mapping across the two tables assumes the GGUF table order matches the metadata offsets.
- Pre-existing failing e2e specs (`apps/desktop/tests/e2e`): orchestrate, hub-live, hub-verified, train,
  tune, machine-tune, models disk-space undo, tools Escape.

## How to verify

- Engine: `cargo test --release -p cerebro-engine --lib`; benchmarks with `examples/bench.rs`
  (`--backends gpu --lengths 512 --decode 64 --reps 2`); per-token split timing with `CEREBRO_SPLIT_PROF=1`.
- Kernel changes: identical greedy text before and after (`examples/greedy.rs`) on Qwen2.5-14B, a Qwen3.8
  DeltaNet model and Gemma 4.
- App: `cargo test --release -p cerebro-desktop`; real-model check
  `CEREBRO_TEST_MODEL=<gguf> cargo test --release -p cerebro-desktop a_real_model_answers -- --ignored`;
  UI `npx tsc --noEmit -p .`, `npx vitest run --config ui/vite.config.ts`, Playwright e2e.
- House rules: interface copy in the third person, plain adult language; no seeded personal data; every
  protection is a user switch; Windows paths; tok/s shown after replies.
