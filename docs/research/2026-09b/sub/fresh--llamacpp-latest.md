# llama.cpp b10751 → b11028 (2026-09-22) — harness-relevant changes

Window: builds after b10751 up to ~b11028 (2026-09-22). Spot-checked against
master at b11118 (2026-09-23). llama.cpp ships per-commit pre-release builds;
each "bXXXXX" tag is one merged PR, so the changelog IS the release list:
https://github.com/ggml-org/llama.cpp/releases
Speculative-decoding reference: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
Multimodal reference: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
Server reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## 0. Version / asset notes (Windows-first)

- Newest in scope ~b11028 (2026-09-22); master was b11118 on 2026-09-23, so the
  delta below is current to within a day.
- Every release publishes Windows x64 zips: CPU, CUDA 12.4, CUDA 13.4, Vulkan,
  OpenVINO, SYCL, ROCm 10.0 — plus arm64 CPU/CUDA13/OpenCL-Adreno. The
  reference laptop (RTX 3070 Ti, Ampere 8.6) wants `llama-b11XXX-bin-win-cuda-13.4-x64.zip`
  + matching `cudart` DLL zip. The local b10751 build (CUDA+RPC+CPU, no Vulkan)
  is now ~270 commits behind; a self-built fork measured 3–5x slower than the
  official binary on this machine, so re-baseline on the official b11026 CUDA
  build at C:\bench\llamacpp-official before trusting any old number.
- `-hf <user>/<model>[:quant]` / `-hfd` download direct from Hugging Face; HF
  downloads now live in the standard HF cache dir (shared with other HF tools).
  `llama-gguf-split --merge` rejoins sharded GGUFs (matters for Qwen3.5 122B 4-part).
- SPEED-Bench client (`tools/server/bench/speed-bench`) runs baseline-vs-spec
  comparisons against a live server — adopt its method for the harness autotuner.

## 1. Speculative decoding: three new draft-model types

- **EAGLE-3 merged** (`--spec-type draft-eagle3`, `-md` draft GGUF):
  https://github.com/ggml-org/llama.cpp/pull/18039 (commit 88a3927).
  One-layer autoregressive draft reading target hidden states; needs
  `--target-model-dir` conversion. Reported 2–3x on LLaMA-8B class (PR: 121→274 t/s
  Q4_K_M quicksort, datacenter GPU, batch 1 — do not quote as laptop numbers).
  Drafts exist for Qwen3 dense+MoE (AngelSlim/Tengyunw/RedHatAI), Gemma4, gpt-oss.
  MoE targets gain less (more experts fire during parallel verify).
  Long-context caveat: Gemma4-26B EAGLE3 acceptance collapses by ~30K tokens
  (draft rope_theta=10000, draft-model limit, verified vs vLLM):
  https://github.com/ggml-org/llama.cpp/pull/24707
- **DFlash merged** (`--spec-type draft-dflash`):
  https://github.com/ggml-org/llama.cpp/pull/22105 (z-lab block diffusion).
  Emits a whole block per draft step; up to ~8x claimed on Qwen3 (datacenter,
  greedy). Qwen3-8B bf16: DFlash geomean ~2.9x vs baseline across 5 domains
  (PR numbers, RTX 4090). `--spec-draft-n-max` clamps to trained block size.
  Drafts: `z-lab/Qwen3*-DFlash`, Gemma4, gpt-oss, Kimi, MiniMax.
- **DFlash 2** (local-conv + candidate selector, `incoai/Qwen3.8-27B-DFlash2`):
  PR https://github.com/ggml-org/llama.cpp/pull/27342 ; merge reported
  2026-08-27 (https://www.llms.blog/posts/llama-cpp-merges-dflash-2-support-for-up-to-2x-faster-speculative-decoding-across-long-contexts).
  Independent end-to-end study (RTX-class GPU, Qwen3.8-27B Q4_K_M): 68→154 t/s
  (2.26x), optimum `--spec-draft-n-max 5`, hard cap 7:
  https://github.com/lukaLLM/DFlash2_Qwen3.8_3.6_27B_LlamaCPP
  and https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF
- **DSpark merged 2026-07-28** (`--spec-type draft-dspark`, `--spec-draft-conf-min`):
  https://github.com/ggml-org/llama.cpp/pull/25173 (DeepSeek+PKU DeepSpec).
  DFlash + semi-autoregressive Markov head; greedy stays lossless. RTX 4090,
  Qwen3-8B bf16: 1.88x overall vs baseline, 1.21x over DFlash; Q4_K_M geomean
  1.81x. Drafts: `deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7`, speculators-format
  RedHatAI Gemma4 drafts (#26275). Only Qwen3 backbone so far.
- n-gram family unchanged in kind, now selectable as a comma list
  (`ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod`) and combinable
  with a draft model; `--spec-default` enables ngram-mod. New: backend sampling
  flags (`--backend-sampling`, `--spec-draft-backend-sampling`) and synthetic
  acceptance bench flags (`--spec-synth-rates`, `--spec-synth-len`) — outputs
  under synthetic mode are NOT valid model output.
- Harness rule: greedy (temp 0/top-k 1) spec is output-lossless modulo
  floating-point; always verify acceptance + tok/s on the 3070 Ti, since MoE
  verify cost and draft VRAM change the tradeoff vs datacenter cards.

## 2. Hybrid/recurrent models: speculation + prompt cache now work (with caveats)

- Speculative checkpointing for SSM/recurrent state: #19493 (server),
  #22227 (speculative-simple), #22114/#22168 (logic/reset):
  https://github.com/ggml-org/llama.cpp/pull/19493
  Flags: `--ctx-checkpoints N`, `--spec-use-checkpoints on`, `--spec-ckpt-num-tries`.
- Multi-turn cache reuse fixes for Qwen3.5 (DeltaNet/Gated-DeltaNet hybrid):
  #23121 and #20428 threads; turn-2 prefill 12K tok → ~tens of tokens when fixed.
  Maintainers note most "full reprocessing" reports are clients mutating the
  prefix — verify with fixed prompts before blaming the runtime.
- Known traps on this hardware class: checkpoint "Chunk not found" crash under
  parallel load fixed by #20232
  (https://github.com/ggml-org/llama.cpp/issues/20222); tensor-split + hybrid
  assert `split_state.n_segments == 1`
  (https://github.com/ggml-org/llama.cpp/issues/21765); SSM checkpoints (~150 MB
  each) can permanently degrade a VRAM-constrained GPU until restart
  (https://github.com/ggml-org/llama.cpp/issues/20145) — cap `--ctx-checkpoints`
  and `--parallel` on 8 GB VRAM.
- DFlash + mmproj multimodal crashes (draft KV positional hole, HTTP 500);
  community zero-fill patch stops the crash but gives zero image speedup
  (https://github.com/ggml-org/llama.cpp/issues/27408). Recommendation: force
  `--spec-type none` on image/audio turns.

## 3. Router / multi-model mode (experimental, hardened in window)

- Run `llama-server` with no `-m`: router dispatches by `model` field, one child
  process per model, LRU eviction (`--models-dir`, `--models-preset` INI,
  `--models-max` default 4, `--no-models-autoload`, `/models/load|/unload`):
  https://huggingface.co/blog/ggml-org/model-management-in-llamacpp and
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md
- In-window fixes: router eviction races (#29217, b11114), child log-file leak
  (#29212, b11113). Only one model resident per worker — alternating models =
  full reload each time; group requests by model. Crash isolation favors
  one-process-per-model (llama-swap pattern) for a small office.
- Harness mapping: router replaces hand-rolled model swapping; pin
  `--models-max 1–2` on 8 GB VRAM and pass through per-model presets
  (ngl, ctx, spec flags) from the Rust supervisor.

## 4. Tool calling, MCP, agent mode (all EXPERIMENTAL, localhost-only)

- Server exposes function calling/tool use for ~any model (Jinja templates),
  `/tools` endpoint, Anthropic Messages compat, responses API:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- `--tools <list|all>` / `--agent` (`-ag`) built-ins: read/write/edit/glob/grep/
  shell/get_info; `--mcp-servers-config` (Cursor JSON) spawns MCP stdio servers
  as children and merges their tools (#26062, superseding #25736).
  WebUI adds MCP client + agentic loop (#18655).
- Security changed in window: `--tools/--agent` flips `--cors-origins` default to
  localhost; new `--cors-origins/--cors-methods/--cors-headers/--cors-credentials`
  (#25655). `/cors-proxy` (`--ui-mcp-proxy`) is an open SSRF-able forwarder —
  never expose. Critical: b11000 fixed an unauthenticated RPC RCE
  (use-after-free via cached compute graph, #24292):
  https://github.com/ggml-org/llama.cpp/releases
- `input_image` in `function_call_output` (b11112, #22575) — vision tool results.
- Harness mapping: keep agent mode OFF by default; allowlist `--tools` subset,
  force `--cors-origins localhost`, never spawn MCP servers from model text;
  surface tool calls through the existing approval UX.

## 5. Reasoning controls

- Per-model Jinja + `--chat-template-kwargs '{"enable_thinking": false}'` is the
  supported thinking toggle (Qwen3.x); `preserve_thinking=true` keeps reasoning
  serialized across turns (needed for Qwen3.6 checkpoint reuse claims).
  No universal `--reasoning-effort` flag in scope — effort knobs are template/
  model-specific (e.g. DFlash2 benches use Qwen3.8 `reasoning=xhigh`).
  Treat any "disable thinking" claim per-model and verify output, not just speed.

## 6. Multimodal (mtmd): audio + video join image

- libmtmd serves image, audio, video via `/chat/completions`; `--mmproj`,
  `--no-mmproj-offload`, `--no-mmproj`:
  https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- New prequants in scope: Gemma4 family (E2B/E4B/26B-A4B/31B, audio+vision on
  small ones), Qwen3-Omni-30B-A3B, Qwen3-ASR 0.6B/1.7B, Voxtral-Mini-3B;
  OCR models (PaddleOCR-VL #18825, GLM-OCR #19677, DeepSeek-OCR #17400,
  Dots.OCR #17575, HunyuanOCR #21395).
- 8 GB VRAM reality: mmproj defaults to GPU offload; budget it explicitly or
  pass `--no-mmproj-offload` when the LLM layers need the room.

## 7. KV cache / memory

- Quantized KV (`-ctk/-ctv`: f16/q8_0/q4_0/…) with Hadamard attention rotation
  (`attn_rot`, disable via `LLAMA_ATTN_ROT_DISABLE`) remains the lever for big
  ctx on 8 GB; SWA/ISWA sliding caches + `--ctx-checkpoints`,
  `--checkpoint-every-nb`, `--cache-prompt`, `--cache-ram`, `--fit` control
  retention. Mixed-precision KV design: `docs/mixed-kv-cache-design.md`.
- Online KV requantize (`POST /cache/requantize`, `llama_requantize_memory`) is
  a PoC PR only (#24134) — not merged; approximate with restart-at-lower-ctk.
- Research-only, NOT upstream: TurboQuant 3-bit KV (ICLR 2026) in forks —
  promising on paper, ik_llama.cpp tests show worse PPL than q4_0+Hadamard on
  several models: https://github.com/ikawrakow/ik_llama.cpp/issues/1509 and
  https://github.com/ggml-org/llama.cpp/discussions/20969

## 8. CUDA / backend perf in window (Ampere notes)

- b11009 fused-QKV fix for Gemma4/Qwen35 (#28965); b11007 CUDA graphs for MTP
  draft (#28549); b11002 CUDA/HIP im2col access (#28013); concurrent streams per
  split for multi-GPU (#28198); OpenCL speculative/MTP GEMV kernels (#26477);
  Metal sparse FA (#28098); Vulkan Intel-Xe FA kernels (#24406); Hexagon Q4_K/Q6_K
  (#28994). No Ampere-specific kernel work in window — expect small, not step,
  gains on the 3070 Ti; NVFP4/MXFP4 paths are Blackwell-or-newer (Qwen3.5 NVFP4
  wiring #20506) and irrelevant to this laptop, consistent with the ruled-out list.
- Qwen3.5 support itself (#19468, Gated-DeltaNet hybrid, Feb 2026) predates the
  window; in-window work is the fixes above plus MTP/spec integration.

## 9. New model coverage in window

- DFM Mimir 1B `HrmTextForCausalLM` (b11003, #27625); DeepSeek-V4 arch support
  (#24162, July); speculators-format DSpark checkpoints (#26275); gpt-oss +
  native MXFP4 and Ollama/vendor-sync fallout predate the window (Aug 2025–early
  2026) but matter: gpt-oss has no Q4_K_M — MXFP4 IS its native precise format.
- Research-only, not in llama.cpp: DDTree speculative layer over DFlash
  (https://arxiv.org/abs/2604.12989); MiniMax-M3 EAGLE3 fork; ik_llama.cpp
  extra quants/IQK-FA variants (emerging fork, CPU+CUDA focused).

## 10. What to benchmark first on the reference laptop (batch 1)

1. Official b1102x CUDA13 vs local b10751 on Qwen3-Coder-30B-A3B Q4_K_M + Gemma-4-E4B.
2. `--spec-default` (ngram-mod) everywhere — free on repetitive coding/agent turns.
3. EAGLE-3 (`AngelSlim/Qwen3-30B-A3B_eagle3`) vs MTP vs DSpark-adjacent Qwen3 drafts
   on HumanEval-style tasks; record accept rate, tok/s, VRAM.
4. `--ctx-checkpoints 1–2` + `--parallel 1` for Qwen3.5/Gemma4 multi-turn; watch
   for the #20145 degradation signature.
5. Keep `--moe-expert-cache-size` OFF (measured slower), min_p 0.05 stays.
