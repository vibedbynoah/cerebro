# llama.cpp changes since b10751

_Subtask fresh--llamacpp-latest_

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



# Newest open models worth offering

_Subtask fresh--new-models_

# fresh--new-models — Newest open models worth offering (Aug–Sep 2026)

Focus: what changed since 2026-09-01, with July/August releases included where they are the current best local pick. All claims checked against primary sources opened 2026-09-23. Datacenter batch numbers do NOT transfer to batch-1 laptop; flagged below.

## 1. Qwen3.8-27B dense — the default local pick (2026-08-14)

- What: 27B dense native vision-language model (text+image+video in, text out), hybrid Gated DeltaNet + Gated Attention (3 linear : 1 full), MTP trained, thinking mode on by default with `reasoning_effort` (xhigh/medium/low) and `preserve_thinking`. Context 262,144 native, YaRN-extensible to ~1M. License Apache-2.0 ([HF model card](https://huggingface.co/Qwen/Qwen3.8-27B)).
- Why it matters: vendor table claims Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7, LiveCodeBench v6 90.3, GPQA-Diamond 89.2, OSWorld-Verified 84.3 — all Claude-Code harness, temp 1.0/top_p 0.95 ([HF model card](https://huggingface.co/Qwen/Qwen3.8-27B)). Self-reported; treat as directional.
- GGUF: Unsloth Dynamic 3.0 repo `unsloth/Qwen3.8-27B-GGUF` — UD-IQ1_S 6.19 GB, UD-IQ2_XXS 7.27 GB, UD-Q2_K_XL 9.83 GB, UD-Q4_K_M 16.5 GB, Q8_0 29 GB, BF16 54.7 GB, plus mmproj + MTP quants ([Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)). GSQ-RCO non-uniform alternative claims IQ3_XXS 10.1 GB within ~1 pt of BF16 on GPQA/LiveCodeBench (community report, unverified). Bartowski/Distillio imatrix Q4_K_M 17.77 GB also listed.
- Fit 8 GB VRAM + 32 GB RAM: NOT fully in VRAM at usable quants. Correct offer: 2-bit IQ2 (~8–10 GB file) split GPU/CPU, or Q4_K_M with partial offload + short context; best on 24 GB cards. Sampler: thinking mode temp 1.0/top_p 0.95/top_k 20; non-thinking 0.7/0.8/presence 1.5 per card.
- Harness: one-click entry `Qwen3.8-27B (UD-Q4_K_M / IQ2_XXS fallback)`, download mmproj-F16 for vision, expose `--mmproj`, `-c` tiers (8K/32K/128K), MTP draft toggle, thinking toggle.

## 2. Qwen3.8-2.4T-A95B (Max-class open weights, 2026-08-12/13)

- What: first open Qwen-Max-class MoE: 2.4T total / 95B active, hybrid linear+full attention, 262K native (~1M via YaRN). Weights on HF + ModelScope; managed 1M-context API on QwenCloud ([Qwen blog](https://qwen.ai/blog?id=qwen3.8), [AWS deploy post](https://aws.amazon.com/blogs/machine-learning/deploying-qwen3-8-2-4t-a95b-on-amazon-sagemaker-hyperpod-with-vllm/)).
- Licence: mixed reports — some secondary coverage says Apache-2.0, other trackers say custom Qwen3.8-Max licence ($50M MaaS / attribution clauses). Treat as **unverified — read LICENSE file before shipping**.
- GGUF: Unsloth `Qwen3.8-2.4T-A95B-GGUF` UD-Q1_0 (IQ1_XXXS) ~397 GB file, needs ~450 GB RAM; BF16 ~4.9 TB. No laptop fit.
- Harness: API-only entry; do NOT offer local install on reference machine. Note NVFP4/MXFP4 path needs Blackwell B300-class nodes.

## 3. Qwen3.8-Flash-Next (2026-08-27, Qwen4 arch preview)

- What: open-weight multimodal MoE, early preview of Qwen4 architecture (evolution of Qwen3-Next Gated DeltaNet+Gated Attention). Served as Qwen3.8-Flash on QwenCloud ($0.16 in / $0.47 out per 1M) ([Alibaba Cloud blog](https://www.alibabacloud.com/blog/qwen3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501)). Weights on HF + ModelScope.
- Harness: list as emerging efficiency pick; GGUF maturity unverified — gate behind "experimental" flag until llama.cpp support confirmed.

## 4. Qwen3.8-Omni-Flash (2026-09-22, research)

- What: native omni MoE (text+image+audio+video), Thinker-Talker, 1M context, Qwen-MM-Plugins + Qwen-Live-Harness frameworks released alongside ([arXiv 2609.25611](https://arxiv.org/abs/2609.25611)). Status research-only; no stable GGUF/llama.cpp path confirmed.
- Harness: skip catalog; track for future audio/video agent work.

## 5. DeepSeek-V4.1-Flash (2026-09-10) — datacenter efficiency, not laptop

- What: 552B-backbone multimodal MoE, Causal Encoder-Decoder: 8B active prefill / 16B decode, CSA2 sparse attention + FP4 KV → 890 bytes/token global (claimed 1/4 HBM, 1/8 SSD vs V4-Flash), 1M context, MIT licence ([DeepSeek news](https://www.deepseek.com/en/news/deepseek-v4-1-flash/), [HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). Pretrained 45T tokens.
- Vendor numbers (internal harness, max effort): Terminal-Bench 2.1 90.6, DeepSWE 74.2, CyberGym 88.1, GPQA 90.9 — ahead of V4-Pro on cheap agentic tier, behind Opus/GPT-5.6 on hard sets ([HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). No chat template shipped; Rust `deepseek-recipe` + `encoding/` reference instead.
- Fit: HF file listing shows ~763B-param footprint class; no mature small GGUF at release; needs 2000-GPU+storage-cluster class serving per vendor. **No local offer** on 8/32 GB machine.

## 6. GLM-5.3 flagship (API 2026-08-14, weights 2026-08-28)

- What: same base as GLM-5.2 (744–753B MoE, 40B active, 78 layers, 256 experts top-8, DSA), all gains from post-training. Text-only, 1M context / 128K out. Weights BF16 + FP8 (F8_E4M3) at `zai-org/GLM-5.3` ([HF card](https://huggingface.co/zai-org/GLM-5.3)). Licence custom `glm-5.3` (MIT-style + security review above $10B MaaS revenue) — NOT plain MIT.
- Vendor claims: +50% on in-house code bench vs 5.2; CyberGym 84.5 SOTA, ExploitBench 54.4 (2.2× 5.2), Terminal-Bench 2.1 88.2 ([HF card](https://huggingface.co/zai-org/GLM-5.3)). Self-reported, Claude-Code harness.
- Fit: FP8 ~756 GB / 141 shards — server-only. Harness: API entry only.

## 7. GLM-5.3-Flash (2026-08-26) — MIT, still server-scale

- What: new base, 320–321B total / 18B active, hybrid linear+sparse, 45 layers, text+image+video in, 1M context (1,048,576 config), MIT licence, SGLang/vLLM/TokenSpeed support ([GLM-5.3-Flash blog](https://z.ai/blog/glm-5.3-flash)). Stealth-ran as "Ox Alpha" on OpenRouter.
- Fit: FP8 ~331 GB — still far beyond reference laptop. Harness: API entry; local only for 128 GB+ workstations at 3-bit (unverified).

## 8. Tencent Hy4 preview (2026-08-28)

- What: Hunyuan MoE 770B / 49B active, 78 layers (256 experts top-8 after first dense layer), 1M+ context, BF16 (~1.56 TB) + FP8 (~814 GB) at `tencent/Hy4-preview`, Apache-2.0 ([Tencent announcement](https://www.tencent.com/tencent-releases-and-open-sources-tencent-hy4-preview/)). Internal blind test (163 experts, 203 tasks): 2.99/4 vs GLM-5.3 2.92, Kimi K3 2.94 — company test, no external replication.
- Fit: official vLLM recipe 8-way TP (16×B200 / 8×B300). No mature GGUF at launch. API-only for Cerebro.

## 9. Gemma 4 family (Apr 2026 + 12B June + July 15 refresh) — best small/VL picks

- What: Apache-2.0, native multimodal (image/video all sizes; audio on E2B/E4B/12B), 128K (edge) / 256K context, 140+ langs, thinking mode, MTP drafters, llama.cpp/MLX/Ollama day-0 ([Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [HF blog](https://huggingface.co/blog/gemma4)). Sizes: E2B (2.3B eff), E4B (4.5B eff), 12B Unified encoder-free (11.95B), 26B-A4B MoE (3.8B active), 31B dense. Arena ~1452 (31B) / 1441 (26B-MoE).
- GGUF: `ggml-org/gemma-4-*-it-GGUF` official; E2B ~3 GB QAT, E4B 5.5–6 GB 4-bit, 12B ~8 GB 4-bit, 26B-A4B ~18 GB 4-bit, 31B ~20 GB 4-bit.
- Fit: E2B/E4B/12B fully in 8 GB VRAM; 26B-A4B partial offload viable (only 3.8B active → fast tok/s); 31B needs Q3/offload. **Offer all five.**
- Harness: default offline VL picks; expose `--mmproj` equivalent (llama-mtmd), image-token budget (70–1120), MTP assistant toggle.

## 10. DiffusionGemma 26B-A4B (2026, alongside Gemma 4)

- What: discrete-diffusion text generator on 26B-A4B MoE foundation (25.2B/3.8B active), text+image in, 256K, Apache-2.0 ([HF blog](https://huggingface.co/blog/gemma4)). Block-parallel denoising: ~15–20 tokens/forward, >1100 tok/s claimed on H100 FP8 (vendor, low batch). Quality cost: trails autoregressive 26B (MMLU-Pro 77.6 vs 82.6, AIME 69.1 vs 88.3).
- Status emerging; llama.cpp support unverified. Track only.

## 11. OpenAI gpt-oss-20b / 120b (2025-08, still current small reasoning pick)

- What: MoE reasoning models, Apache-2.0: 20b (21B/3.6B active, MXFP4 ~12 GB, 128K), 120b (117B/5.1B active, ~61–63 GB, 80 GB GPU) ([gpt-oss repo](https://github.com/openai/gpt-oss)). Harmony format, adjustable reasoning, function-calling; no vision.
- GGUF: `ggml-org/gpt-oss-*-GGUF` MXFP4; llama.cpp guide documents `--n-cpu-moe` splits, 20b at 14.9 GB @8K / 120b at 64 GB @8K, RTX 2060 8 GB runnable with 22 CPU-MoE layers (community guide).
- Fit: **20b is the only frontier-ish reasoning model fully usable on 8 GB VRAM** (partial offload). Offer as default offline coder alongside Gemma.

## 12. Poolside Laguna XS 2.1 (2026-07-02, boundary — best local coding MoE)

- What: 33B MoE / 3B active, 40 layers (10 global + 30 SWA-512), 256 experts + shared, 256K (docs say 262,144), interleaved thinking, text-only, OpenMDW-1.1 permissive ([Poolside blog](https://poolside.ai/blog/introducing-laguna-xs-2-1)). Checkpoints BF16/FP8/INT4/NVFP4 + official GGUF (BF16, Q4_K_M) + DFlash 0.5B speculator (~70% accept, claimed ~2× tok/s).
- Benchmarks vendor-only: SWE-bench Verified 70.9%, Multilingual 63.1% (+5.4 vs XS.2). llama.cpp support landed via PR #25165 — require recent build.
- Fit: Q4_K_M ~20 GB file → runs on 36 GB Mac; on 8/32 laptop use INT4 + DFlash + partial offload, 8–32K context. Offer as "fast local coder" experimental.

## 13. Mistral Medium 3.5 128B (2026-05, boundary — newest Mistral open)

- What: dense 128B merged instruct+reasoning+coding, 256K, custom vision encoder, modified MIT (large-revenue carve-out), SWE-bench Verified 77.6% vendor ([Mistral news](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5/)). Self-host ~4 GPUs; Ollama `mistral-medium-3.5:128b-q4_K_M`.
- Fit: no laptop fit (~70+ GB at Q4). API-only; flag licence carve-out.

## 14. Vision/OCR + embeddings specialists

- DeepSeek-OCR-2 (Jan 2026): 3B MoE (570M active) document/OCR VLM, Apache-2.0, ~4 GB at 4-bit, single-GPU friendly. Best cheap RAG-ingest companion; llama.cpp VL support unverified — offer via Python sidecar first.
- Embeddings: **no new Aug–Sep 2026 open embedding release verified** — keep Qwen3-Embedding / bge-m3 / nomic-embed as stable catalog entries (unverified for "newest" claim). Do not advertise a new embedder without a primary source.
- Kimi K3 (2.8T/104B, July 16/27): frontier API-only, custom licence; no local offer. Tencent Hy3/GLM-5.2 era models superseded — drop from front page.

## Catalog recommendation for Cerebro (8 GB + 32 GB laptop)

1. gpt-oss-20b MXFP4 (offline reasoning/coding default) 2. Gemma 4 E2B/E4B (tiny + audio) 3. Gemma 4 12B Unified (best 8 GB VL) 4. Gemma 4 26B-A4B Q4 (fast MoE, partial offload) 5. Qwen3.8-27B IQ2_XXS/Q4 (max quality, offload) 6. Laguna XS 2.1 INT4+DFlash (agentic coding experimental) 7. API-only: Qwen3.8-Max, GLM-5.3/Flash, DeepSeek-V4.1-Flash, Hy4, Mistral Medium 3.5, Kimi K3.



# Local image generation backends

_Subtask fresh--image-generation_

# Local image generation backends — research notes (2026-09-22)

Focus: what changed since 2026-09-01, for a Windows-first Rust harness
(spawn-backend-as-child-process, like llama.cpp) on the reference laptop
(i9-12900H, 31.7 GB RAM, RTX 3070 Ti 8 GB Ampere, no NPU).
All URLs below were opened during this pass. Anything not verified is marked
"unverified".

## 1. stable-diffusion.cpp is the obvious embeddable core

Repo: https://github.com/leejet/stable-diffusion.cpp (MIT, pure C/C++, ggml-based,
"working in the same way as llama.cpp"). News entries run through 2026-09-20
(Day-0 Qwen-Image-2.1), i.e. actively maintained within the focus window.
Supported image models now include SD1.x/2.x/Turbo, SDXL/Turbo, SD3/SD3.5,
FLUX.1-dev/schnell, FLUX.2-dev/klein, Qwen-Image (+2.1), Z-Image, Chroma,
Chroma1-Radiance, Lens, PiD, LongCat, MiniT2I, Ovis, Anima, ERNIE-Image, Krea2,
Mage-Flow, HiDream-O1, Ideogram4, LLaDA-Image; edit models (FLUX Kontext,
Qwen-Image-Edit series); video (Wan2.1/2.2, LTX-2.3/2.5, MiniMax-H3,
HunyuanVideo 1.5, LingBot-Video). Backends: CPU (AVX/AVX2/AVX512), CUDA,
Vulkan, Metal, OpenCL, SYCL. Weight formats: ckpt, safetensors, GGUF, plus a
convert mode (weights to GGUF/safetensors). Prebuilt Windows binaries
(CUDA/Vulkan/CPU) are published per-release, so Cerebro can redistribute or
sidecar them like it does llama.cpp.
Source: https://github.com/leejet/stable-diffusion.cpp

## 2. Server/API mode exists and is the harness seam

`sd-server` exposes three families
(source: https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md):
OpenAI-compatible (`POST /v1/images/generations`, `POST /v1/images/edits`
multipart incl. mask, `GET /v1/models`); WebUI-compatible
(`POST /sdapi/v1/txt2img`, `/sdapi/v1/img2img`, plus loras/upscalers/samplers/
schedulers/sd-models/options discovery); and a native async `sdcpp API`
(`GET /sdcpp/v1/capabilities`, `POST /sdcpp/v1/img_gen`, `POST
/sdcpp/v1/vid_gen`, `GET /sdcpp/v1/jobs/{id}`, cancel endpoint). Native-only
controls can be smuggled through the compat APIs inside
`<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` in the prompt. Prompt-embedded
`<lora:...>` tags are intentionally unsupported server-side; LoRA must go
through structured fields. Since 2026-04-11 there is also a brand-new embedded
web UI served by the binary itself
(source: https://github.com/leejet/stable-diffusion.cpp). Caveat from a
community server-mode write-up: model is fixed at server start (restart to
swap) — unverified in this pass beyond that report.

## 3. Speed per image at 1024px: sd.cpp trails ComfyUI, leads on footprint

Hard numbers all come from one sd.cpp issue thread (batch 1; Linux where
stated; samplers mostly unstated — treat as indicative, not transferable):
https://github.com/leejet/stable-diffusion.cpp/issues/1215
- Reporter: FLUX.2-klein-4B, 1024x1024, 4 steps, cfg 1 — ComfyUI ~5 s vs
  sd.cpp ~25 s (Vulkan and ROCm builds).
- RX 7800XT, Arch, mesa-git, same thread: SDXL (cfg 4, 30 steps) ComfyUI
  12.36 s warm (16.86 s cold) vs sd.cpp Vulkan 29.63 s (~2.4x slower);
  Z-Image-Turbo (cfg 1, 9 steps) 20.84 s vs 34.57 s (~1.65x); FLUX.2-klein-4B
  6.24 s vs 23.32 s (~3.7x), then after upstream optimization PR ~31.8%
  faster (15.90 s); klein-9B ~1.9x slower, ~29% faster after the fix.
- RTX 4090, CUDA backend, same thread: sd.cpp ~6.7 it/s vs ComfyUI ~10.5
  it/s (~1.6x gap).
No RTX 3070 Ti-class 1024px timing was verified in this pass (unverified).
Rule of thumb for Cerebro's benchmark table: sd.cpp ≈ 1.5–2.5x slower per
image than ComfyUI/PyTorch on the same discrete GPU, but with far lower
VRAM and zero Python/CUDA-toolkit dependencies.

## 4. VRAM: what fits the 8 GB Ampere card

- FLUX.1 diffusion-weights-only table
  (source: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux.md):
  q8_0 12068 MB / q4_0 6394 MB / q4_k 6395 MB / q3_k 4888 MB / q2_k 3735 MB —
  plus VAE + CLIP-L + T5-XXL on top. Doc says 6 GB, "even 4 GB" GPUs can run
  FLUX with offload; cfg-scale 1.0 recommended; schnell needs only 4 steps.
  Practical 8 GB recipe: schnell q4/q2_k GGUF + `--clip-on-cpu`
  (+ `--offload-to-cpu` if needed), T5 fp8 variant.
- Qwen-Image (20B MMDiT; Apache-2.0; source: https://github.com/QwenLM/Qwen-Image):
  Unsloth's sd.cpp tutorial
  (source: https://unsloth.ai/docs/models/tutorials/qwen-image-2512/stable-diffusion.cpp)
  uses Q4_K_M diffusion (~13.1 GB for the Edit-2512 GGUF) + Qwen2.5-VL-7B
  Q4 text encoder + safetensors VAE, rule of thumb "combined RAM+VRAM >
  GGUF size", example 1024px/40 steps/euler/cfg 2.5/flow-shift 3 with
  `--diffusion-fa --offload-to-cpu`. Fits the reference laptop only with
  CPU offload (slow); not the default pick.
- SDXL/SDXL-Turbo remains the safe 1024px default on 8 GB (few-GB
  footprint); SD3.5-Medium/Large-Turbo is the quality step-up. Exact
  sd.cpp VRAM figures for these were not re-verified this pass (unverified).

## 5. Feature coverage: img2img, inpainting, upscaling all present

sd.cpp API supports init image, mask image (incl. inpainting invert flag),
denoising strength, control/IP-adapter/reference images, tiled VAE, ESRGAN
upscale models, TAESD fast decode, hi-res fix (upscaler incl. latent modes),
LoRA, LCM/LCM-LoRA, PhotoMaker, IP-Adapter (SD1.5/SDXL), ControlNet (SD1.5
only), ADetailer, Flash Attention, negative prompt
(sources: https://github.com/leejet/stable-diffusion.cpp and
https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md).
KoboldCpp's image-gen settings page confirms the same surface from a
shipper's view (quantize/compress weights, TAE fallback VAE, no-VAE-tiling
toggle, per-model CLIP/T5/VAE slots):
https://koboldai.com/Guides/Kobold_CPP/KoboldCpp_ImageGen/

## 6. Licences (all primary sources opened)

- FLUX.1-schnell: Apache-2.0, commercial OK
  (https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- FLUX.1-dev (and Kontext etc.): Non-Commercial License v1.1.1 — no
  commercial/production use without a BFL deal; outputs are yours but can't
  train competing models
  (https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
- FLUX.2-klein is described as Apache-2.0 and FLUX.2-dev as non-commercial
  only in a secondary listing (https://lmstudio.ai/maticulous75/image) —
  primary BFL terms unverified this pass.
- SD3.5 suite: Stability Community License — free incl. commercial use under
  $1M annual revenue, Enterprise above; attribution/"Powered by Stability AI"
  required (https://stability.ai/license).
- Qwen-Image (v1): Apache-2.0 (https://github.com/QwenLM/Qwen-Image).
- Qwen-Image-2.1: Qwen Research License, NON-COMMERCIAL ONLY, commercial
  licence via model-business@notice.qwencloud.com
  (https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE). Licence
  regime changed between v1 and 2.x — Cerebro must gate these separately.
- sd.cpp itself is MIT; KoboldCpp is AGPL-3.0 (careful with bundling:
  https://github.com/LostRuins/koboldcpp/); Comfy Desktop is AGPL/commercial
  dual (FAQ below); diffusion-rs is MIT (its repo page).

## 7. ComfyUI: headless API yes, Desktop app no

Comfy Desktop "does not support headless, server-only, or remote operation
modes… For headless or server deployments, use a manual ComfyUI installation
or the Windows Portable build instead"
(source: https://docs.comfy.org/installation/desktop/faq). The installed
`C:/Users/noahm/AppData/Local/Comfy-Desktop` copy therefore cannot be driven
as Cerebro's backend; the portable/manual install exposes the `/prompt` +
`/history` (+ websocket) API that wrappers like comfy-headless POST
workflow JSON to (workflow API format described in search results —
unverified in this pass). ComfyUI is the speed leader (§3) but drags a full
Python + PyTorch + CUDA stack and model-folder management — an optional
"bring your own ComfyUI" target, not the bundled default.

## 8. Other embeddable engines

- KoboldCpp (single-file exe, AGPL): bundles sd.cpp image generation
  (SD1.5/SDXL/SD3/FLUX/Qwen-Image/Z-Image/Klein per its readme) with an
  A1111-compatible API and bundled StableUI; proves the "llama.cpp-style
  single binary + image tab" pattern Cerebro wants
  (sources: https://github.com/LostRuins/koboldcpp/ and
  https://koboldai.com/Guides/Kobold_CPP/KoboldCpp_ImageGen/).
- diffusion-rs: Rust bindings to sd.cpp on crates.io; CUDA+Vulkan marked
  working on Windows (source: https://github.com/newfla/diffusion-rs) —
  the fastest path if Cerebro wants in-process instead of argv-spawned
  `sd-server` (F FI risk vs process isolation trade-off).
- sd.cpp readme also lists Python/C#/Go/Dart bindings and UIs
  (sd.cpp-webui, LocalAI, GIMP plugins)
  (source: https://github.com/leejet/stable-diffusion.cpp) — not evaluated
  further (unverified beyond the listing).

## 9. Competitors: LM Studio and Jan do NOT do local image generation

- LM Studio: no native local diffusion runtime found. Image generation
  happens only via cloud: an MCP workaround calling Hugging Face Spaces, or
  a Hub plugin that calls the Hugging Face Inference API (FLUX.1-schnell
  default, dev models need Pro credits, cold starts 20–60 s)
  (source: https://lmstudio.ai/maticulous75/image). Fully offline local
  image gen is a genuine Cerebro differentiator.
- Jan: v0.6.9 "image support" is image *input* (multimodal understanding:
  auto-detected vision for local Gemma3/Qwen3 etc.), not image synthesis
  (source: https://jan.ai/changelog/2025-08-28-image-support). A Jan server
  doc snippet suggests provider-routed `/v1/images/generations|edits`
  proxying to an external image provider — unverified (doc URL 404'd when
  opened). Either way, no bundled local diffusion.

## 10. Cutting-edge 2025–2026 deltas for the harness

Day-0/1 model support is now a release-note race (Qwen-Image-2.1 Day-0
2026-09-20; FLUX.2-dev/klein, Z-Image, Wan2.1/2.2, LTX-2.3/2.5, MiniMax-H3
all added since Sep 2025: https://github.com/leejet/stable-diffusion.cpp).
Notable: sd.cpp docs list an inference-acceleration-via-caching guide and
INT8-convrot safetensors (listing only — unverified); Wan/LTX video jobs
run through the same async `vid_gen` API but are VRAM-heavy and not a sane
default on the 8 GB card (unverified timings). Qwen-Image-2.x's licence flip
to non-commercial is the sharpest gotcha of the month (§6).

## 11. Recommendation for Cerebro

Bundle `sd-server` (prebuilt CUDA Windows binary, pinned version + SHA) as
a second argv-spawned child beside llama.cpp; talk to it over
`sdcpp API` (async jobs) with OpenAI-compat fallback. Defaults for the
reference machine: SDXL-Turbo (fastest) / FLUX.1-schnell-Q4 (quality), both
Apache/Community-clean; hide dev/Qwen-2.x weights behind a non-commercial
acknowledgement. Keep ComfyUI portable as an optional external engine via
its `/prompt` API. Benchmark table must state hardware/model/quant/steps/
backend per image, and re-measure on the 3070 Ti — the 5 s vs 25 s class of
gaps above do not transfer across GPUs.



# Local speech-to-text and text-to-speech

_Subtask fresh--audio_

# Local speech stack for Cerebro — late Sept 2026 (fresh--audio)

Scope: best fully-offline Windows desktop STT + TTS for a Rust harness that spawns
`llama-server`/`llama.cpp` with argv only. Reference machine: i9-12900H, 32 GB RAM,
RTX 3070 Ti Laptop 8 GB (Ampere sm_86, ~80–115 W), no NPU. All figures are batch-1
laptop-relevant unless stated; datacenter batch-128 numbers are flagged non-transferable.
Cutoff focus: changes since 2026-09-01, preferring 2025–2026 primary sources.

## 1. Speech-to-text

### 1.1 whisper.cpp (C/C++, GGML) + server mode — default offline STT
- Current: v1.9.4 (2026-09-11) fixes decoder reseeding, adds language in `/inference`
detect response; v1.9.3 (2026-08-20) nightly b4938; v1.8.5+ hardened server
(token_timestamps flag, carry_initial_prompt, speaker JSON, params-leak fixes).
Sources: https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.3 ,
https://freedom.tech/posts/2026-09-11-whisper-cpp-1-9-4/ ,
https://github.com/ggml-org/whisper.cpp/releases/tag/v1.8.5
- Server (`whisper-server`) is file-POST `/inference`, not chunked streaming; one
community thread prototypes `new_segment_callback` chunked streaming but crashes on
segment 2 (mutex/lifetime bug, unverified for production).
Source: https://github.com/ggml-org/whisper.cpp/discussions/3306
- Live mic path is `whisper-stream` (SDL2, `--step/--length`, sliding-window `--step 0`
+ `-vth ~0.6` VAD). Source: https://github.com/ggerganov/whisper.cpp/blob/master/examples/stream/README.md
- whisper.cpp vs faster-whisper (Aug 2026 comparison): same weights (tiny 39M … large-v3
1550M); RTX 4070 large-v3 faster-whisper int8 ~12xRT/2.5 GB vs whisper.cpp CUDA ~8xRT/~3 GB;
M-series whisper.cpp Metal ~10xRT wins on Mac; x86 CPU tiny/base faster-whisper int8 ~20xRT
vs whisper.cpp ~15xRT. No RTX 3070 Ti laptop figure found (unverified); treat 4070 ratios
as upper bounds (power-limited Ampere will be lower).
Source: https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026
- Weights choice: large-v3-turbo (809M, decoder 32→4 layers, Sep 2024) is ~5x Apple-Silicon
/ ~2.7x GPU faster than large-v3 at +0.39pp mean WER (7.83 vs 7.44); faster-whisper 13-min
GPU run: turbo fp16 19.2 s/2.5 GB vs large-v3 fp16 52.0 s/4.5 GB; turbo int8 1.5 GB.
Sources: https://huggingface.co/openai/whisper-large-v3-turbo ,
https://github.com/SYSTRAN/faster-whisper/issues/1030 ,
https://whispernotes.app/blog/introducing-whisper-large-v3-turbo
- distil-large-v3 (756M, English-only, 6.3x relative latency, short-form WER 9.7 vs 8.4
large-v3) is fastest for English-only; needs GGML conversion for whisper.cpp, native in
faster-whisper. Source: https://weesperneonflow.ai/en/blog/2026-08-27-whisper-v3-turbo-vs-distil-whisper-comparison-2026/
- faster-whisper itself: CTranslate2 port, up to ~4x vs openai/whisper, int8 ~2x vs fp16
at <0.1pp WER cost; needs CUDA 12 + cuDNN 9 on current ctranslate2; 2026 fork
`faster-whisper2` v2.1.1 (2026-06-18) same claim. Python-only, CUDA-only GPU path.
Sources: https://github.com/SYSTRAN/faster-whisper , https://pypi.org/project/faster-whisper2 ,
https://localaimaster.com/blog/faster-whisper-guide

### 1.2 NVIDIA Parakeet TDT 0.6B v2/v3 — fastest accurate EN/European STT
- 600M FastConformer+TDT; v2 English-only, v3 25 European langs + auto-detect; native
PnC/caps, word timestamps, up to 24-min single pass (full attention, A100 80 GB).
License CC-BY-4.0. Sources: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 ,
https://catalog.ngc.nvidia.com/orgs/nvidia/collections/parakeet-tdt-0.6b ,
https://docs.nvidia.com/nemo/speech/3.0.0/asr/asr_checkpoints.html
- Local C++ path: `NeMo-Speech.cpp` + `parakeet-tdt-0.6b-v3.q8_0.gguf`
(`hf download …; nemo-speech transcribe`). Python path is NeMo 2.2+/Riva; HF Transformers
also supported. Streaming = chunked-inference script (`chunk_secs=2`, unverified latency).
Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/main/README.md
- Speed claims (non-transferable as stated): RTFx 3380 at batch 128 on
HF-Open-ASR leaderboard; community i7 CPU INT8 2.4 s clip → 307 ms (~8xRT), M5 Neural
Engine 33xRT STT. Sources: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 ,
https://www.reddit.com/r/artificial/comments/1u3r85g/i_built_a_100_local_cpuonly_voice_loop_for_any/
- Caveat: NeMo/Riva docs target Linux + H100/A100/L40; Windows is community/Docker only
(unverified). 8 GB Ampere fits q8 GGUF easily (model ~0.6–1.2 GB), but plan CPU fallback.

### 1.3 Moonshine (Useful Sensors) — tiny streaming-first STT
- Offline tiny/base: tiny 27M/~190 MB, base 61M/~400 MB, ONNX, 5x faster than Whisper per
10-s segment at same-or-better WER (vendor claim, hardware unspecified — treat as
unverified). WebRTC/VAD-ready plugin API. Sources: https://pypi.org/project/getstream-plugins-moonshine/ ,
https://github.com/moonshine-ai/moonshine-js
- Streaming family: tiny 34M / small 123M / medium 245M, 50 Hz frontend, sliding-window
encoder, 80 ms lookahead; Transformers support exists but "not yet fully efficient
streaming" (uses flash-attn sliding window). Live browser demo via MoonshineJS+WebRTC.
Sources: https://huggingface.co/UsefulSensors/moonshine-streaming-tiny ,
https://huggingface.co/docs/transformers/v5.9.0/model_doc/moonshine_streaming ,
https://webrtc.moonshine.ai/
- 2026 direction is the `moonshine-ai/moonshine` "Moonshine Voice" toolkit (Python/JS/WASM,
1 MB–size options, unverified benchmarks). Source: https://github.com/moonshine-ai/moonshine

### 1.4 Kyutai STT (delayed-streams) — low-latency streaming baseline
- `stt-1b-en_fr` (~1B, EN+FR, 0.5 s delay + semantic VAD) and `stt-2.6b-en` (~2.6B,
EN-only, 2.5 s delay); decoder-only over Mimi tokens (12.5 Hz, 32 tokens/frame);
word timestamps = frame offset − delay. Weights CC-BY-4.0; code MIT/Apache.
Sources: https://kyutai.org/stt/ , https://huggingface.co/kyutai/stt-2.6b-en ,
https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/README.md
- Runtimes: `moshi.run_inference`, `moshi-server` (CUDA, TOML configs), MLX (Apple).
Paper: https://arxiv.org/abs/2509.08753 (preprint Sep 2025, unverified peer review).
No Windows-native server documented (unverified); run as sidecar process.

### 1.5 llama.cpp audio input (mtmd) — one-process audio+text+vision
- Since b8769 (2026-04-12): Qwen3-Omni-30B-A3B (MoE 30B/3B active, text+image+audio+video)
and Qwen3-ASR supported in mtmd; run via `llama-mtmd-cli`. Earlier: Qwen2-Audio,
SeaLLM-Audio (poor quant results noted), Qwen2.5-Omni 3B/7B, Ultravox, Voxtral, GLM.
Sources: https://www.creativeainews.com/blog/llama-cpp-audio-qwen3-omni-multimodal/ ,
https://github.com/ggml-org/llama.cpp/pull/19441 ,
https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Gemma 3n (E2B/E4B, USM audio encoder, 160 ms chunks, 140 text / 35 multimodal langs)
is text-only in llama.cpp as of Jun 2026; audio needs MLX/transformers.
Sources: https://huggingface.co/blog/gemma3n , https://github.com/ggml-org/llama.cpp/discussions/15194
- Fit: Qwen3-ASR/Omni GGUFs are large for 8 GB VRAM alongside an LLM; prefer dedicated
STT sidecar and reserve mtmd audio for "audio understanding" turns. Quality/latency on
3070 Ti unverified.

## 2. Text-to-speech

### 2.1 Kokoro-82M — default local TTS (quality + licence + size)
- 82M StyleTTS2-class, Apache-2.0 weights+code, 24 kHz, 54 voices/8 langs (v1.0),
int8 ONNX ~92 MB + voices ~28 MB; q-variants listed on HF. Sources:
https://huggingface.co/hexgrad/Kokoro-82M , https://github.com/hexgrad/kokoro ,
https://huggingface.co/onnx-community/Kokoro-82M-ONNX
- Windows-ready ONNX: `kokoro-onnx` (MIT runtime, CPU/CUDA/DirectML providers),
`pykokoro` (DirectML extra), NVIDIA `kokoro-82M-onnx-opt` (win-x64 CUDA13, Ampere
listed). CPU RTF ~0.45–0.51 (~2xRT) on 4-core EPYC; quality rated above Supertonic.
Sources: http://www.github.com/thewh1teagle/kokoro-onnx , https://pypi.org/project/pykokoro/ ,
https://huggingface.co/nvidia/kokoro-82M-onnx-opt , https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts

### 2.2 Piper — cheapest CPU voice prompts
- VITS voices as ONNX (tens of MB), tiers x_low/low 16 kHz, medium/high 22.05 kHz;
100+ voices/30+ langs; ~10xRT desktop CPU, RT on Pi 5 (guide claim, board-dependent).
Maintained fork `OHF-Voice/piper1-gpl` v1.8.0 (2026-09-04); legacy standalone
`rhasspy/piper` Windows zip is 2023 MIT but stale. Source: https://localaimaster.com/blog/piper-tts-setup-guide
- Streaming: `streaming-vits` v0.1.0 (2026-09-01) reuses Piper/VITS graphs, TTFA ~163 ms
vs 701 ms non-streaming on 15.2 s utterance (M-series Mac, 4 threads; unverified on
Windows). Sherpa-ONNX SAPI5 engine reports ~0.5 s latency/250 MB peak (unverified).
Sources: https://pypi.org/project/streaming-vits/ , https://github.com/willwade/sherpaonnxazuresapi-installer/
- Licence caution: `piper1-gpl` is GPL-licensed (name + repo); verify before bundling.

### 2.3 Orpheus 3B — expressive LLM-TTS over llama.cpp (experimental)
- Llama-3.2-3B finetune emitting SNAC 24 kHz tokens as text tokens; GGUF floats in
community (`mmwillet2/Orpheus_GGUF`, isaiahbjork local server recipe `-ngl 29 -c 8192`);
`orpheus-cpp` streams via llama-cpp-python + WebRTC/FastRTC demo. Sources:
https://github.com/canopyai/Orpheus-TTS/blob/main/additional_inference_options/no_gpu/README.md ,
https://github.com/markuryy/orpheus-cpp , https://news.ycombinator.com/item?id=43419983
- Upstream llama.cpp support is OPEN: issue #12476 (SNAC decoder missing), draft PR
#12487 stalled; TTS.cpp PR #90 is the working C++ reference. Sources:
https://github.com/ggml-org/llama.cpp/issues/12476 ,
https://github.com/ggml-org/llama.cpp/pull/12487 ,
https://github.com/mmwillet/TTS.cpp/pull/90
- Fit: 3B Q4 fits 8 GB but contends with the LLM; quality/streaming on 3070 Ti unverified.

### 2.4 Dia 1.6B / Dia2 — dialogue-native, heavy
- Dia 1.6B: 1.6B + Descript codec, `[S1]/[S2]` dialogue + laugh/cough tokens, EN-only,
Apache-2.0, ~10 GB VRAM full, ~40 tok/s on A4000 (~0.46xRT; 86 tok ≈ 1 s audio).
Sources: https://github.com/nari-labs/dia , https://huggingface.co/nari-labs/Dia-1.6B-0626
- Dia2 (2025-11-19, 1B/2B, streaming from first tokens, ≤2 min) is CUDA 12.8+ only as of
Apr 2026; no MLX port. 8 GB laptop cannot hold Dia 1.6B + LLM together; batch-only.
Source: https://codersera.com/blog/run-nari-dia-16b-on-mac-installation-guide/

### 2.5 Kyutai TTS 1.6B (+Pocket TTS) — true text-streaming TTS
- 1.6B-name/1.8B-actual, Mimi 12.5 Hz, audio delayed 16 steps (1.28 s), text-streaming:
220 ms token→audio, batched 32-stream 350 ms on L40S, 16 streams >2xRTF; voices via
precomputed embeddings; custom-voice weights private (cloning issue open).
CC-BY-4.0 weights. Sources: https://kyutai.org/blog/2025-07-03-kyutai-tts-1-6b/ ,
https://huggingface.co/kyutai/tts-1.6b-en_fr , https://github.com/kyutai-labs/moshi/issues/404
- Runtimes: PyTorch streaming script, Rust `moshi-server` + `configs/config-tts.toml`,
MLX `--quantize 8/4`. Pocket TTS (Jan 2026 per thread) is tiny CPU 5-s cloner but
non-streaming batch. Source: https://github.com/kyutai-labs/delayed-streams-modeling
- No Windows-native Rust server docs (unverified); sidecar on WSL2/Linux or CPU PyTorch.

### 2.6 Chatterbox family (Resemble AI, MIT + watermark) — clone + control
- Turbo 350M EN: 1-step distilled decoder, ~6xRT GPU/~75 ms, paralinguistic tags;
Nano 110M: 3xRT on 8 CPU cores; Multilingual v3 0.5B Llama: 23–25 langs, better
similarity/fewer hallucinations; Single-Language Pack (zh/es-mx/es-es/pt-br/pt-pt/hi).
Every clip carries PerTh watermark. Sources: https://www.resemble.ai/learn/models/chatterbox-turbo ,
https://github.com/resemble-ai/chatterbox , https://huggingface.co/ResembleAI/chatterbox ,
https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages
- Caveat: v3 Korean/Vietnamese CER ~71–75% (data-starved, not production); Czech/Japanese/
Finnish 18–32% CER need per-content eval. NVIDIA NIM gives 2–39x vs raw PyTorch
(datacenter, non-transferable). Same v3 source.

### 2.7 Qwen3-TTS (Alibaba, Apache-2.0) — lowest first-packet latency
- 0.6B/1.7B × 12 Hz multi-codebook + 25 Hz semantic variants; 10 langs; 3-s clone;
97 ms first packet (0.6B) / 101 ms (1.7B) at concurrency 1 (paper Table, GPU
unspecified — treat as lab figure); EN WER 1.24 (1.7B-12Hz). Sources:
https://arxiv.org/html/2601.15621v1 , https://github.com/QwenLM/Qwen3-TTS ,
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
- vLLM-Omni day-0 offline support; streaming Python "coming soon" per FAQ; Windows/3070 Ti
numbers unverified. Source: https://qwenlm-qwen3-tts.mintlify.app/resources/faq

### 2.8 Supertonic 3 — tiny multilingual CPU TTS
- ~99M flow-matching ONNX, 404 MB assets, 31 langs, 44.1 kHz; 5-step CPU RTF ~0.31
(~3xRT, "good/clear"), 2-step ~0.165 but robotic; Kokoro wins quality, Apache wins
licence (Supertonic is OpenRAIL-M). Fixed-voice open weights; cloning via Voice
Builder JSON only. Sources: https://heyneo.com/blog/kokoro-tts-vs-supertonic-3-tts ,
https://github.com/supertone-inc/supertonic , https://www.kiadev.net/news/2026-05-15-supertonic-v3-release

## 3. Voice-mode design (Windows harness)

- VAD: Silero v4 1.72 MB / v5 2.22 MB (16/8 kHz), ~1 ms per 30 ms chunk on 1 CPU thread;
sherpa-onnx bundles Silero + ASR/TTS/VAD C++ with DTMF/endpointing knobs.
Sources: https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html , https://github.com/k2-fsa/sherpa-onnx/
- Barge-in recipe (2026 production bar <150 ms): dual-pass WebRTC-GMM instant trigger
(~15–20 ms) + Silero confirm (~30 ms + 150–300 ms smoothing); streaming TTS buffer
200–300 ms with <20 ms flush; 250 ms post-flush cooldown against echo re-trigger;
browser/Windows AEC before VAD; record only heard words on interrupt.
Sources: https://www.syncsoft.ai/en/blog/voice-agent-barge-in-vad-tuning-2026 ,
https://rajatpandit.com/agentic-ai/real-time-audio-vad
- Modes: push-to-talk (simplest, no VAD tuning, best for office/noise) → VAD-gated
auto-turn (500 ms→dynamic semantic endpointing) → full-duplex barge-in (needs AEC +
streaming TTS). Local CPU loop proof (Silero + Parakeet INT8 + Supertonic, 2.4 s→307 ms
STT on i7) shows office-viable CPU-only chain. Source: https://www.reddit.com/r/artificial/comments/1u3r85g/i_built_a_100_local_cpuonly_voice_loop_for_any/
- Recommended Cerebro default: whisper.cpp large-v3-turbo-q5_0 (or Parakeet q8 GGUF where
EN/European-only) + Silero VAD + Kokoro-int8 ONNX; push-to-talk first, VAD auto-turn
second, barge-in behind a flag. All-spawned-as-child-process (argv only, no shell),
loopback-only ports, opt-in update check only (matches offline policy).

## 4. Gaps / what to verify on the reference laptop
- No published RTX 3070 Ti Laptop batch-1 STT/TTS latencies found (unverified); measure:
turbo-q5_0 whisper.cpp CUDA, Parakeet q8 NeMo-speech.cpp, Kokoro-int8 DirectML vs CPU,
Qwen3-TTS-0.6B CUDA.
- Whisper short-bench overstatement (~3x vs soak, per brief) applies to Parakeet/Chatterbox
RTFx too — soak-test before claiming wins.
- Licences to re-check at bundle time: Piper GPL fork, Supertonic OpenRAIL-M, Kyutai/Parakeet
CC-BY-4.0 attribution, Chatterbox PerTh disclosure (EU AI Act Art. 50).



# Hugging Face one-click downloads

_Subtask fresh--hf-downloads_

# Hugging Face one-click downloads — research notes (2026-09-22)

How a desktop app (Cerebro, Rust + spawned llama.cpp) should implement LM Studio-style
one-click model downloads from the Hugging Face Hub. All facts verified Sep 2026 from
primary sources linked inline. Focus: what changed since 2026-09-01, plus durable design.

## 1. Hub HTTP API: search, model info, file listing

- Search/list: `GET /api/models` with `search`, `author`, `filter` (tags), `sort`
  (`downloads`/`likes`/`lastModified`), `direction`, `limit`, `full`, `config`. Paginated
  via `Link` header. Documented at https://huggingface.co/docs/huggingface_hub/en/guides/search
  and the OpenAPI spec https://huggingface.co/.well-known/openapi.md
  (docs index moved to the OpenAPI playground: https://huggingface.co/docs/hub/en/api).
- Python equivalent is `HfApi.list_models()`; CLI equivalent is `hf models ls --search llama
  --sort downloads --limit 5`; file listing `hf models ls <repo> -R --tree -h`.
  See https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Per-file metadata comes from the Hub "paths-info" / tree endpoints; `huggingface.js`
  `downloadFileToCacheDir` resolves etag via `pathsInfo({paths:[path], revision, expand:true})`
  reading `info.lfs.oid`, `info.xetHash`, or `info.oid`:
  https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/download-file-to-cache-dir.ts
- `snapshot_download` pins one commit hash (`repo_info.sha`), stores `refs/<branch>`, and
  caches the per-commit file list in `trees/<commit>.json` so repeat downloads cost one
  network call (branch→hash resolve). Offline + incomplete snapshot raises
  `IncompleteSnapshotError` (with `snapshot_path` attr):
  https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- GGUF-quant repos: `repo_info.siblings` (rfilename/size/lfs/xet) is how apps enumerate
  `Q4_K_M` etc. variants; LM Studio additionally curates a catalog. `allow_patterns` /
  `ignore_patterns` globs filter snapshot downloads:
  https://huggingface.co/docs/huggingface_hub/en/guides/download

## 2. Resolve URLs + resumable Range downloads (the LFS path)

- Canonical file URL: `https://huggingface.co/{ns}/{repo}/resolve/{rev}/{path}`
  (`hf_hub_url()` builds it). Files > few MB 302-redirect to CloudFront/CDN; client must
  follow redirects. HEAD on the resolve URL returns `ETag` (= git-sha1 for git files,
  sha256 for LFS files), `X-Linked-Size`, `X-Repo-Commit`:
  https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download
  and impl https://github.com/huggingface/huggingface_hub/blob/v1.22.0/src/huggingface_hub/file_download.py
- Resume: `http_get` sends `Range: bytes=<resume_size>-`, writes to `blobs/<etag>.incomplete`,
  atomically renames on completion, guards concurrent same-file downloads with a file lock
  (`WeakFileLock`, `.locks/` dir), retries transient errors ~5x, checks `expected_size`:
  same file_download.py source above.
- Known edge (May 2026, PR #4209): CloudFront may ignore `Range` and return 200 (gzip+chunked,
  no `Content-Length` for small text files); client must truncate + roll back progress:
  https://github.com/huggingface/huggingface_hub/pull/4209
- `hf_transfer` (parallel-range Rust downloader, `HF_HUB_ENABLE_HF_TRANSFER=1`) is
  **deprecated** — Xet backend replaces it; it also never supported resume/proxies:
  https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

## 3. Xet storage backend — why plain HTTP clients stall at 0 bytes

- Since `huggingface_hub` 0.30–0.32 (2025), large files live on Xet, not LFS. With `hf_xet`
  installed (bundled since 0.32.0) downloads go: auth → CAS `GET /v2/reconstructions/{file_id}`
  → signed multi-range xorb fetches → chunk reassembly:
  https://huggingface.co/docs/xet/download-protocol and https://huggingface.co/docs/xet/api
- Critical interop trap: `GET <resolve-URL>` returns an `X-Xet-Hash` header; following the
  302 downloads via the legacy LFS-compat bridge
  (`cas-bridge.xethub.hf.co`), NOT the Xet protocol — and the bridge can stall/hang at
  0 bytes (matches the observed `hf download` 0-byte stall on this machine). Doc:
  https://github.com/huggingface/hub-docs/blob/main/docs/xet/file-id.md
- CAS endpoint is discovered per-repo via `/xet-read-token` (`casUrl` =
  `https://cas-server.xethub.hf.co/`, plus xorb buckets `transfer.xethub.hf.co`,
  `us.aws.cdn.hf.co` / `us.gcp.cdn.hf.co`); short-lived signed URLs must not be cached:
  https://github.com/huggingface/huggingface_hub/issues/4475
- `HF_HUB_DISABLE_XET=1` forces the legacy path; `HF_XET_CACHE` (`$HF_HOME/xet`) holds
  chunk/shard caches; `HF_XET_HIGH_PERFORMANCE=1` needs ~64 GB RAM — unsuitable for the
  32 GB reference laptop: https://huggingface.co/docs/hub/en/xet/using-xet-storage
- Reference impl is Rust `xet-core` (`hf-xet`, `xet-client`, `xet-data` crates):
  https://github.com/huggingface/xet-core
- Resume caveat (Jun 2025 forum): Xet-enabled repos can restart from 0% after interrupt;
  Xet resume differs from LFS `.incomplete` resume:
  https://discuss.huggingface.co/t/strange-problem-with-huggingface-cli/160083
- Implication for Cerebro: a pure-`reqwest` Range downloader works for small/git files but
  is second-class for multi-GB weights. Options: (a) depend on `hf-xet`/`xet-core` crates,
  (b) shell to `hf download`/`hfrs`, (c) implement CAS reconstruction (spec is open).

## 4. Rust client: `hf-hub` crate 1.0.0 (Jul 2026)

- Official Rust equivalent of `huggingface_hub`: `HFClient` (async) + `HFClientSync`
  (`blocking` feature), cache-compatible with Python, `HFError::RateLimited` (429),
  Xet high-performance transfers, `snapshot_download` with allow/ignore globs, wasm support:
  https://crates.io/crates/hf-hub and https://github.com/huggingface/hf-hub
- Download-builder semantics (range = Rust `start..end`, ETag/`If-None-Match`, `.no_exist`
  markers, transient-error→cache fallback): https://docs.rs/hf-hub/latest/src/hf_hub/repository/download.rs.html
- Ships an `hfrs` CLI (`cargo install --git … hfrs`). Token order: explicit → `HF_TOKEN` →
  token file; `HF_ENDPOINT` override supported (mirror-friendly).

## 5. Gated models + tokens

- Gated repos require login + "Agree" (username+email shared with authors); auto vs manual
  approval; request flow is browser-only, downloads need a token:
  https://huggingface.co/docs/hub/en/models-gated
- Tokens: fine-grained (scoped, preferred for shipping), `read` (download private/gated),
  `write`; pass as `Authorization: Bearer hf_…` or `?token=`; never bake into binaries:
  https://huggingface.co/docs/hub/en/security-tokens
- `hf auth login` (browser device flow or `--token`), token at `$HF_HOME/token`:
  https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Ollama precedent for private HF GGUFs: per-machine SSH key (`~/.ollama/id_ed25519.pub`
  registered in HF settings) — no bearer token in app:
  https://huggingface.co/docs/hub/ollama

## 6. Rate limits (Sep 2025 tiers, still current doc)

- Three buckets over 5-min windows: Hub APIs / **Resolvers** (`/resolve/…`, highest limits)
  / Pages. Free user: 1,000 API + 5,000 resolver reqs; anon (per IP): 500 + 3,000:
  https://huggingface.co/docs/hub/en/rate-limits
- Exceed → 429 with IETF `RateLimit` / `RateLimit-Policy` headers; `huggingface_hub` ≥1.2.0
  auto-waits-until-reset on 429. Guidance: always send `HF_TOKEN` downstream, prefer
  resolver calls over API calls, back off with jitter. Ollama users hit 429 on unauthenticated
  `hf.co` pulls (0.14.x): https://github.com/ollama/ollama/issues/7240

## 7. Checksums

- Cache identity IS the checksum: `blobs/<etag>` where etag = LFS sha256 (or git-sha1);
  `snapshots/<commit>/<path>` symlinks to blobs:
  https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- `hf cache verify <repo> [--revision …] [--local-dir …]` (added Oct 2025, PR #3461)
  recomputes sha256 (LFS) / git-sha1 and reports mismatches:
  https://github.com/huggingface/huggingface_hub/commit/80025cdf5c3ea7e0739f1233e6b510ff802a0a0c
- Deliberately NO on-download hash check (full re-read of GBs too costly; size check only) —
  Dec 2025 decision after Xet corruption reports:
  https://github.com/huggingface/huggingface_hub/issues/3643
  Cerebro should offer post-download verify as an explicit quality-gate step.

## 8. How LM Studio / Jan / Ollama do it

- **LM Studio**: curated catalog + variant picker (Q3_K_S…Q8), search accepts keywords or full
  HF URLs, nested-subfolder repos supported (0.3.9, Jan 2025), optional HF proxy setting,
  `~/.lmstudio/models/<publisher>/<model>/` layout preserving HF structure, `lms get <url>`
  CLI, `POST /api/v1/models/download` REST:
  https://lmstudio.ai/docs/app/basics/download-model ,
  https://lmstudio.ai/blog/lmstudio-v0.3.9 ,
  https://huggingface.co/docs/hub/en/lmstudio ,
  https://lmstudio.ai/docs/app/advanced/import-model ,
  https://github.com/lmstudio-ai/lms/blob/9902c3a0/src/subcommands/get.ts
- **Jan** (open source, Tauri+Rust): download manager in Rust backend (single-file +
  whole-HF-repo, progress via Tauri events, cancel, `HF_TOKEN` setting, path-traversal
  sanitization) — merged May 2025 for the llama.cpp extension:
  https://github.com/menloresearch/jan/pull/4933 ; models stored as plain `.gguf` files.
- **Ollama**: OCI-style registry (`registry.ollama.ai`): manifest JSON → content-addressed
  `blobs/sha256-<digest>`; chunked parallel pull with per-chunk resume (`-partial-N` state),
  NDJSON progress; `hf.co/<user>/<repo>[:quant]` bridge runs any of ~45k HF GGUFs (server-side
  manifest synthesis, requires GGUF/llama.cpp-compatible):
  https://github.com/ollama/ollama/blob/e434a938/server/images.go ,
  https://huggingface.co/docs/hub/ollama ,
  https://github.com/ollama/ollama/issues/8632 (CDN stall precedent)

## 9. Mirrors

- `HF_ENDPOINT` (read at import/startup) redirects all Hub traffic; community mirror
  `https://hf-mirror.com` documents `export HF_ENDPOINT=https://hf-mirror.com`:
  https://hf-mirror.com/ ; maintainer-confirmed pattern:
  https://github.com/huggingface/huggingface_hub/issues/2830
- Cerebro (offline-first, no mirrors by default) should still accept an endpoint override
  for lab/proxy use; never hardcode third-party mirrors.

## 10. Licence display requirements

- Licence is model-card YAML: `license:` (+ `license_name`/`license_link` for custom) —
  surfaced on the model page and filterable: https://huggingface.co/docs/hub/en/model-cards
- One-click UX must show `license` (+ `gated` state) BEFORE download; fetch via
  `hf models card <repo> --metadata [--format json]`:
  https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Gated "Agree" text (`extra_gated_*` fields) is part of the licence consent surface:
  https://huggingface.co/docs/hub/en/models-gated

## 11. Recommended Cerebro design (Windows-first, offline, Rust)

1. **Search**: `GET {endpoint}/api/models?search=…&filter=gguf&sort=downloads&limit=…`
   (+ `Link` pagination), custom `User-Agent: Cerebro/<ver>`. Cache results; respect 429
   headers with backoff. No network except opt-in update check — gate the whole feature
   behind explicit user action + token entry.
2. **Variant pick**: `GET /api/models/{id}` (siblings: name/size/lfs-sha256/xet flag) →
   filter `*.gguf` → rank by fit (VRAM 8 GB: prefer Q4_K_M of ≤8B; show est. RAM/VRAM).
   Pin `revision` = commit hash; show licence + gated badge first.
3. **Download**: prefer Xet-capable path (depend on `hf-hub` 1.x/`xet-core` or shell
   `hf download --local-dir`); else Range-resume GET to `.part` + atomic rename,
   `Range` re-request on 200-without-`Content-Range`, `HF_HUB_DOWNLOAD_TIMEOUT`-style
   generous timeouts, token from OS credential store (never disk-plaintext).
4. **Verify**: size check always; optional `sha256` vs LFS etag post-pass (quality gate).
5. **Layout**: `models/<publisher>/<repo>/<rev-short>/<file>.gguf` + sidecar JSON
   (repo, rev, etag, size, licence, URL, timestamp) — LM Studio-like, offline-readable.



# MCP and plugin ecosystems

_Subtask fresh--mcp-plugins_

# MCP and plugin ecosystems — late Sept 2026 (Cerebro research)

Date: 2026-09-22. Cutoff focus: what changed since 2026-09-01; spec baseline is 2026-07-28.
All URLs below were opened via search/fetch. Unverified = marked.

## 1. Spec: 2026-07-28 is current stable (stateless core)

- The 2026-07-28 spec is the stable release; RC locked 2026-05-21, final 2026-07-28
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Headline: stateless protocol core. `initialize`/`notifications/initialized` handshake
  and `Mcp-Session-Id` removed (SEP-2575, SEP-2567). Every request carries protocol
  version + client capabilities in `_meta`; optional `server/discover` RPC replaces
  handshake; any request can hit any instance behind round-robin LB
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/)
  (https://modelcontextprotocol.io/specification/2026-07-28/changelog).
- Streamable HTTP must include `Mcp-Method` / `Mcp-Name` headers for routing without
  body parsing (SEP-2243) (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- `tools/list`, `prompts/list`, `resources/list`, `resources/read` carry `ttlMs` +
  `cacheScope` (SEP-2549) (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- MRTR (multi-round-trip requests, SEP-2322): server returns
  `resultType: "input_required"` + needed requests; client retries with
  `inputResponses`. Replaces held-open server-initiated `elicitation/create`,
  `sampling/createMessage`, `roots/list` (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Tasks moved to `io.modelcontextprotocol/tasks` extension with poll `tasks/get` +
  new `tasks/update`; notifications via opt-in `subscriptions/listen`
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Deprecated (12-month minimum window): Roots, Sampling, Logging (SEP-2577);
  legacy HTTP+SSE transport; DCR (see auth)
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Extensions framework formalized (SEP-2133 lineage); MCP Apps + Tasks + Enterprise
  Managed Authorization are extensions, negotiated via capabilities
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/)
  (https://modelcontextprotocol.io/extensions/apps/overview).

## 2. SDKs: TypeScript v2, Rust rmcp 3.x

- Tier 1 (2026-07-28 on day one): TypeScript, Python, Go, C#. Rust = Tier 2, beta→stable
  support after (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- TypeScript v2 (stable 2026-07-28): monolith `@modelcontextprotocol/sdk` split into
  `@modelcontextprotocol/server` 2.0.0 + `@modelcontextprotocol/client` 2.0.0;
  Standard Schema input/output (Zod v4, ArkType, Valibot); `serveStdio()` /
  `createMcpHandler()` web-standard Request/Response (Node/Bun/Deno/Workers);
  thin Express/Hono/Fastify/Node adapters; v1.x security fixes ≥6 months
  (https://github.com/modelcontextprotocol/typescript-sdk)
  (https://www.npmjs.com/package/@modelcontextprotocol/server).
- Rust `rmcp`: crate `rmcp` 3.4.0 published 2026-09-15 (after cutoff window; 3.0.0 cut
  2026-07-28); implements stable 2026-07-28 + back-compat 2025-11-25; features:
  discover/legacy/auto lifecycle modes, MRTR, tasks, caching, routing headers,
  stateless Streamable HTTP, schemars JSON Schema 2020-12, tokio
  (https://github.com/modelcontextprotocol/rust-sdk) (https://crates.io/crates/rmcp).
- Cerebro relevance: Rust harness should depend on `rmcp` 3.x with `server` feature
  for in-process servers and `TokioChildProcess` client for spawned stdio servers.

## 3. Hosts: how desktop apps install/run servers

- Claude Desktop: `claude_desktop_config.json` (legacy manual) + one-click `.mcpb`
  desktop extensions (Settings → Extensions → Browse/Install); built-in Node;
  secrets in OS keychain; curated directory auto-updates; Team/Enterprise can
  push/block/private-list (https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)
  (https://www.anthropic.com/engineering/desktop-extensions).
- Claude Code CLI: `claude mcp add --transport http|sse|stdio`, `claude mcp add-json`,
  `claude mcp add-from-claude-desktop`; scopes local/project (`.mcp.json`)/user;
  plugins bundle servers via `.mcp.json` or inline `plugin.json`
  (https://code.claude.com/docs/en/mcp).
- Cursor: Settings → MCP panel + `~/.cursor/mcp.json` (global) or `.cursor/mcp.json`
  (project), `mcpServers` key; stdio + Streamable HTTP (≥v0.48); per-server restart;
  one-click `cursor.com/en/install-mcp?name=..&config=..` deeplinks
  (https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-cursor.md).
- VS Code (≥1.99): Extensions view `@mcp` gallery; `.vscode/mcp.json` with
  `"servers"` key (not `mcpServers`); user/profile/remote/devcontainer variants;
  `code --add-mcp`; auto-discovery from Claude Desktop/Cursor/Windsurf/Copilot CLI;
  resources/prompts/MCP Apps surfaced in Chat
  (https://code.visualstudio.com/docs/agent-customization/mcp-servers).
- LM Studio (≥0.3.17, host): Program tab → Install → Edit `mcp.json`
  (`~/.lmstudio/mcp.json`, `%USERPROFILE%/.lmstudio/mcp.json` on Windows); follows
  Cursor `mcpServers` notation; stdio spawned per server + remote URL/headers;
  "Add to LM Studio" deeplink; API exposes ephemeral vs `mcp.json` integrations
  with `allowed_tools`; caution: local models + Claude-oriented servers can flood
  context (https://lmstudio.ai/docs/app/mcp).
- DeepSeek Harness (DSH = "DSH Desktop" in task): `@deepseek-ai/dsh-mcp-client`,
  one plugin instance = one MCP server in `cordis.yml`/patch; stdio +
  `streamable-http`; tools namespaced `mcp__<server>__<tool>` through unified
  gating/timeout pipeline; only Tools bridged (no Resources/Prompts); dsh-desktop
  Electron/Tauri wrappers add plugin/MCP/skills markets + paste-and-convert MCP JSON
  (https://deepseekdocs.com/en/docs/features/mcp).

## 4. Packaging + registries: MCPB, server.json, one-click

- MCPB (was `.dxt`, renamed late 2025; both install): zip + `manifest.json`
  (`mcpb_version`, `name`, `server.type` node/python/binary, `mcp_config`,
  `user_config` with `${__dirname}`/`${user_config.x}` templating, secrets → OS
  vault); CLI `npm i -g @anthropic-ai/mcpb; mcpb init; mcpb pack`; Node recommended
  (ships in Claude Desktop); open spec + toolchain
  (https://github.com/modelcontextprotocol/mcpb)
  (https://www.anthropic.com/engineering/desktop-extensions).
- Official MCP Registry (preview 2025-09-08): `registry.modelcontextprotocol.io`,
  open API + `server.json` + `mcp-publisher` CLI; packages point at npm/PyPI/NuGet/
  Docker/MCPB/GitHub releases; public + private sub-registries; community
  denylist moderation
  (https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/).
- GitHub MCP Registry one-click: server page → Install in VS Code, pre-filled config
  (https://github.blog/ai-and-ml/generative-ai/how-to-find-install-and-manage-mcp-servers-with-the-github-mcp-registry/).
- Smithery (`smithery.ai/server/...`) and MCPFind-style catalogs add scanned metadata
  + per-client install snippets; route through hosted gateway vs local stdio tradeoff
  (https://github.com/sena-labs/replicate-mcp-server).

## 5. Permissions/sandboxing (host controls Cerebro should copy)

- Spec line: human-in-the-loop SHOULD hold deny on tool invocation; hosts SHOULD show
  exposed tools + indicators + confirmations (cited in
  https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks context;
  Simon Willison summary unverified-fetch, treat as secondary).
- VS Code: workspace trust + per-server trust dialog on first start/config change;
  `MCP: Reset Trust`; autostart policy (`never`/`onlyNew`/`newAndOutdated`);
  per-tool toggles; `sandboxEnabled` + `sandbox.filesystem.allowWrite` /
  `sandbox.network.allowedDomains` on macOS/Linux (not Windows); enterprise GitHub
  policy gating (https://code.visualstudio.com/docs/agent-customization/mcp-servers).
- Claude Code: project `.mcp.json` approval prompt; workspace-trust interaction;
  `disabledMcpServers`/`disabledMcpjsonServers`; per-call timeouts + idle timeouts;
  OAuth token purge on remove (https://code.claude.com/docs/en/mcp).
- LM Studio: no confirm before first spawn — 2026-04-17 bug report shows enabled
  stdio entry spawns on enable + every launch, handshake after; recommends
  first-enable dialog, re-prompt on command hash change, kill process group
  (unverified-fix; issue text from search, treat as report not spec).
- Cerebro: hash-pin `command+args`, first-run consent showing argv/cwd, re-consent
  on change, allowlist, per-tool approve, kill process group on failure, log all
  tool I/O to SIEM-style local log.

## 6. Auth for remote servers: OAuth 2.1 + CIMD (2026-07-28 hardening)

- Base: OAuth 2.1, RFC 8414/9728 discovery, RFC 8707 `resource` indicators, PKCE S256
  mandatory; server validates aud/iss/signature per request; tokens never in query.
- 2026-07-28 deltas: `iss` per RFC 9207 validated (SEP-2468, SHOULD→future MUST);
  credentials bound per issuer, re-register on AS change (SEP-2352);
  `application_type` native/web in DCR fixes localhost rejects (SEP-837); refresh
  guidance (SEP-2207); step-up scope accumulation is client duty (SEP-2350);
  standard `oauth-authorization-server` suffix (SEP-2351); DCR formally deprecated
  for CIMD (self-hosted HTTPS `client_id` URL doc; fetch+validate, no registry-write
  endpoint) (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Cerebro (fully offline default): remote OAuth path is opt-in only; local stdio
  servers need no OAuth; if remote enabled, implement CIMD-first + DCR fallback,
  `resource`-bound tokens, per-issuer credential store.

## 7. MCP Apps (extension `io.modelcontextprotocol/ui`, stable 2026-01-26)

- Tools link `_meta.ui.resourceUri` → `ui://` HTML resource
  (`text/html;profile=mcp-app`); host prefetches via `resources/read`, renders in
  sandboxed iframe, postMessage JSON-RPC (`ui/initialize`, `tools/call` proxy,
  `ui/update-model-context`); CSP/permission declarations; graceful text fallback
  if host lacks support (https://modelcontextprotocol.io/extensions/apps/overview).
- Supported by Claude (web+desktop), VS Code Copilot, M365 Copilot, Goose, Postman,
  MCPJam, Archestra (https://modelcontextprotocol.io/extensions/apps/overview).
- Cerebro: render-deserves-care on Windows-first offline host — iframe sandbox +
  allowlist CSP; Apps increase prompt-injection surface (rendered HTML from server).

## 8. Security: tool poisoning, rug pulls, shadowing, output injection

- Tool poisoning (Invariant Labs, Apr 2025): hidden directives in tool
  descriptions/params invisible in simplified UI exfiltrate `~/.cursor/mcp.json`,
  `~/.ssh/id_rsa`, hijack other servers' tools (e.g. reroute `send_email`);
  demonstrated vs Cursor; mitigations: show full descriptions, pin versions/hashes,
  cross-server boundaries (https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks).
- OWASP framing: connect-time vs runtime trust gap; tool outputs flow into context
  unvalidated; mitigations: structured-output schemas, isolate privileged tools,
  server-side enforcement (not prompt-only), allowlist servers, out-of-band confirm
  for sensitive ops (https://github.com/OWASP/www-community/blob/master/pages/attacks/MCP_Tool_Poisoning.md).
- Rug pulls: approved tool silently redefined server-side; Microsoft Zero-Trust
  catalog maps to LLM01/LLM03/LLM06/LLM07/LLM08 + MITRE AML.T0010/AML.T0051.001/
  AML.T0053/AML.T0110; controls: RBAC identity, supply-chain vetting/signing,
  gateway filters (Prompt Shields), DLP labels, sandboxing, SIEM logging, block+escalate
  (https://learn.microsoft.com/en-us/security/zero-trust/catalog-ai-attack-techniques/rug-pull-attack).
- Scale numbers (secondary, via CSA 2026-07-01/02 notes): MCPTox benchmark 45 live
  servers × 20 LLMs avg 36.5% poisoning success, max 72.8% (o1-mini); MCPoison
  CVE-2025-54136 in Cursor (approve-by-name, config swap = team RCE, fixed 1.3.9);
  TrustFall May 2026: 4 coding CLIs auto-executed project MCP on folder-trust.
  Treat numbers as reported-benchmark, not laptop batch-1 transfer.
- Research mitigation ETDI (arXiv 2506.01333, Jun 2025): signed versioned tool
  definitions (+ API-contract hash), OAuth scopes in signed def, PDP policy check
  (Cedar/OPA), call-stack verification; prototype fork only
  (https://arxiv.org/html/2506.01333v1).

## 9. Most useful servers today (+ image gen)

- Reference set (`modelcontextprotocol/servers`): filesystem (scoped dirs + Roots),
  fetch (URL→markdown), memory (knowledge-graph JSONL), git, sequential-thinking,
  time, everything (test); community: Brave Search, Postgres/SQLite, Playwright
  (browser), GitHub official `ghcr.io/github/github-mcp-server` (remote
  `https://api.githubcopilot.com/mcp/` or Docker; old npm `server-github`
  deprecated Apr 2025) (https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-cursor.md).
- Image gen (all networked, API-key, NOT offline — Cerebro conflict to disclose):
  `mcpimg` (OpenRouter/Together/Replicate/fal, stdio `npx -y mcpimg`); fal single-purpose
  `mcp-fal-ai-image` (`generate-image`, saves `~/Downloads/fal_ai`); Replicate family:
  official thin REST bridge vs `sena-labs/replicate-mcp-server` curated 36-task-tool
  studio (waits + auto-downloads to `~/Downloads/replicate-mcp/`, cost estimator,
  batch/DAG, `.mcpb` + Smithery + HTTP/SSE)
  (https://github.com/sena-labs/replicate-mcp-server).
- Local-model warning (LM Studio docs): Claude/ChatGPT-oriented servers can bloat
  context and overflow small local windows — gate with `allowed_tools`/scoped servers
  (https://lmstudio.ai/docs/app/mcp).

## 10. Cerebro integration sketch (Rust + argv-only llama.cpp)

- Ship `rmcp` client (TokioChildProcess, argv arrays, no shell) + optional `rmcp`
  server for Cerebro's own tools; support stdio first, Streamable HTTP opt-in.
- Config: `mcp.json`-compatible (`mcpServers`), VS Code `servers` alias accepted;
  per-server `command/args/env/cwd`, `allowed_tools`, timeouts, hash-pinned approvals.
- Distribute curated local servers as signed `.mcpb`-style zips; verify manifest hash
  before spawn; OS-vault secrets; enterprise allowlist/blocklist.
- Default-deny network: image-gen and remote servers behind explicit opt-in +
  red "network" badge; document per-call $/latency since offline is the promise.



# Tool permission systems

_Subtask fresh--permissions_

# Tool permission systems — fresh research, late Sept 2026

Scope: Claude Code, OpenAI Codex CLI, opencode, Cline, Gemini CLI, Cursor.
Focus: modes, rule syntax, approval UI, diff previews, Windows sandboxing, audit logs, praise/complaints.
Date: 2026-09-22. All URLs below were opened during research.

## 1. Claude Code

Permission model: three rule lists — `allow`, `ask`, `deny` — in `settings.json`, evaluated in
fixed order **deny → ask → allow**; specificity never changes the order. A broad deny cannot carry
allowlist exceptions (https://code.claude.com/docs/en/permissions). Rule shape is `Tool` or
`Tool(specifier)`, e.g. `Bash(npm run *)`, `Read(./.env)`, `WebFetch(domain:example.com)`.
A bare-name deny (`Bash`) removes the tool from the model's context entirely; a scoped deny
(`Bash(rm *)`) leaves the tool visible and blocks matching calls. Managed enterprise settings
override everything; otherwise lists merge across managed → CLI flags → local → project → user,
and a deny at any level beats an allow anywhere else.

Modes (`permissions.defaultMode`, switchable with Shift+Tab): `default`/`manual` (prompt on first
use), `acceptEdits` (auto-accept file edits + basic fs commands in workspace), `plan` (read-only
recon), `auto` (a classifier model reviews actions instead of the user), `dontAsk` (auto-deny
anything not pre-approved — for CI), `bypassPermissions` (`--dangerously-skip-permissions`, for
throwaway containers only). Deny and explicit ask rules still apply even in bypass mode, plus a
hard circuit breaker prompting on `rm -rf /`-style critical-path removals
(https://code.claude.com/docs/en/permission-modes?c=mikannn&fcdaa149_sort_date=desc).
Manage rules via the `/permissions` dialog, which shows each rule's source file.

Bash rules are **prefix/text matches, not a security boundary** — the docs say so explicitly:
`/usr/bin/curl`, `sh -c 'curl …'`, `git -C . push` all dodge `Bash(curl *)` / `Bash(git push *)`.
Compound commands are split per subcommand; a fixed wrapper list (`timeout`, `nice`, …) is stripped;
there is a built-in read-only command set; output redirects are checked against Edit/Read rules
(https://code.claude.com/docs/en/permissions). For real enforcement the docs point at the Bash
sandbox (macOS, Linux, WSL2 only — no native Windows sandbox) and at `PreToolUse` hooks
(https://code.claude.com/docs/en/hooks.md, https://code.claude.com/docs/en/agent-sdk/hooks.md).
Hooks (`PreToolUse`, `PostToolUse`, `PermissionRequest`, `PermissionDenied`, …) can allow/deny/ask,
rewrite tool input, and are the community's audit-log mechanism — but hook decisions never override
deny/ask rules, and there are open Windows reports of PreToolUse hooks silently never firing
(https://github.com/anthropics/claude-code/issues/25981) or of `ask` verdicts being ignored under
`defaultMode: auto` (https://github.com/anthropics/claude-code/issues/85904). Workspace trust:
project-level allow rules only apply after accepting the trust dialog.

Third-party audit/policy hooks (all local-only, hook-based, not OS sandboxes): `cc-audit`
append-only JSONL log + YAML blocklists, fail-closed policy (https://pypi.org/project/cc-audit/0.1.2/);
`sneakoscope`/`agentguard` with hash-chained log, live dashboard, and `run` sandbox on Linux/macOS
only — "on Windows there is no clean native equivalent, so `run` refuses" (https://pypi.org/project/sneakoscope/0.4.1/);
`PolicyApprovalGate`, a deterministic LLM-free PreToolUse gate with audit log, rotation, hashing,
redaction; Windows support explicitly experimental (https://github.com/nobuo-miura/PolicyApprovalGate).

## 2. OpenAI Codex CLI

Two stacked controls: **sandbox mode** (what code *can* do: `read-only`, `workspace-write`,
`danger-full-access`/`--yolo`) × **approval policy** (when it must *ask*: `on-request` (default),
`never`, `unless-trusted`, `on-failure`, plus granular) (https://learn.chatgpt.com/docs/agent-approvals-security,
https://github.com/openai/codex/blob/13c42a077c88a0d04ae7680a9891d2daf4558577/docs/sandbox.md).
Approval UI offers accept-once / accept-for-session / accept-and-add-to-policy (writes an execpolicy
rule); read-only + never is the CI preset (https://openai-codex.mintlify.app/concepts/approvals).
`execpolicy` rules in Starlark (`prefix_rule(pattern=[…], decision=allow|prompt|forbidden)`)
persist trusted commands per project (same source). Newer **permission profiles** (beta,
Codex ≥0.138) replace sandbox_mode with named profiles (`:read-only`, `:workspace`,
`:danger-full-access`, or custom with `extends`), filesystem `read|write|deny` entries
(`**/*.env = deny` carve-outs), and network domain allow/deny enforced via a proxy feature
(https://developers.openai.com/codex/permissions). **Auto-review** routes sandbox-escalation
approvals to a separate reviewer agent instead of the user; it grants nothing, only changes who
reviews (https://developers.openai.com/codex/concepts/sandboxing/auto-review).

Windows is the weak platform everywhere: Codex's native Windows sandbox is experimental
(AppContainer restricted token; cannot stop writes where the Everyone SID already has access),
with `elevated` vs weaker `unelevated` levels — docs recommend WSL for the Linux model
(https://developers.openai.com/codex/permissions,
https://github.com/openai/codex/blob/13c42a077c88a0d04ae7680a9891d2daf4558577/docs/sandbox.md).
Biggest recent complaint: `approval_policy = "untrusted"` (run only known-safe commands
automatically) was retired in 0.149.0 with a hard startup error telling users to delete it, silently
downgrading trusted projects to weaker `on-request`; docs still recommended it at filing time
(https://github.com/openai/codex/issues/39973).

## 3. opencode

Simplest model of the six: a `permission` map (`allow|ask|deny`) keyed by tool (`bash`, `edit`,
`read`, `glob`, `grep`, `webfetch`, …) plus guards (`external_directory`, `doom_loop`), with
per-pattern objects (`"bash": {"*": "ask", "git *": "allow", "rm *": "deny"}`). **Last matching
rule wins** — opposite of Claude Code's deny-first order (https://opencode.ai/docs/permissions/).
Defaults are permissive (`allow` almost everywhere; only `doom_loop`/`external_directory` ask and
`.env` reads denied). `--auto` flips asks to approvals while still enforcing denies; the TUI shows
an `auto` indicator. Ask UI: `once` / `always` / `reject`, where `always` writes a durable
project-scoped allow rule that still can never override `deny`. Agent identities get their own
permission overlays (a `review` subagent can be `edit: deny`), and v2 moved to an ordered
`permissions[]` array of `{action, resource, effect}` with `shell`/`subagent` renames
(https://opencode.ai/v2/docs/permissions/). Above that sit `experimental.policies`: binary,
never-prompt, organization-wins statements (e.g. `permission:shell:git push * → deny`) delivered
also via OpenCode Console workspaces (https://opencode.ai/v2/docs/policies/). The rework history
(`tools` boolean merged into `permission` in v1.1.1) is in (https://github.com/anomalyco/opencode/pull/6319).
No OS-level command sandbox was found in the docs — confinement is approval-gating plus the
`external_directory` boundary. Praise centers on the tiny config surface; the risk side is the
permissive default plus last-match-wins ordering, which rewards putting the catch-all first.

## 4. Cline (VS Code / JetBrains / CLI / SDK)

Category toggles, not regex rules: Read/Edit project files + outside-workspace extensions, Execute
safe/all commands, browser, MCP, notifications (https://docs.cline.bot/features/auto-approve).
Critically, **the model itself marks each command `requires_approval`** — there is no fixed
allowlist, and the "safe" examples are explicitly "not guarantees." YOLO mode auto-approves
everything with no confirmation dialog; enterprises can kill the toggle remotely via
`yoloModeAllowed: false` (https://docs.cline.bot/enterprise-solutions/configuration/infrastructure-configuration/control-other-cline-features/yolo-mode).
Compensating control is **checkpoints**: a shadow git repo snapshots the workspace after every tool
call, with Compare (multi-file diff) and Restore files / task / both per step
(https://docs.cline.bot/core-workflows/checkpoints,
https://github.com/cline/cline/blob/65e9727c/src/integrations/checkpoints/index.ts). Approvals show
editable diffs first (VS Code diff preview work: https://github.com/cline/cline/commit/2ac5c85e6950528de2a2a69a9e0e68307eb6d78d);
SDK embeds use `toolPolicies: {tool: {autoApprove}}` with conditional-approval callbacks, default
enabled+auto-approved so policies must be set explicitly
(https://docs.cline.bot/sdk/guides/permission-handling); CLI has `--yolo` / `--auto-approve`,
plus `CLINE_COMMAND_PERMISSIONS` allow/deny globs where deny wins
(https://github.com/cline/cline/blob/fd8cecdd/cli/man/cline.1.md). No OS sandbox; isolation = Git
rollback + human gates. Users like the low friction; compliance-minded admins are pointed at
disabling YOLO, telemetry/task-history/audit logs, and non-production restriction.

## 5. Gemini CLI

Modes: `default` (prompt), `auto_edit` (edits auto-approved), `plan` (read-only + plan-artifact
rules via a `plan.toml` policy tier), and `yolo` — CLI-flag-only, never a persisted default
(https://geminicli.com/docs/reference/configuration/, https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/plan-mode.md).
**Trusted Folders**: enabling `security.folderTrust` gates project config/`.env`/extensions/MCP on a
per-folder trust decision; untrusted folders force a restricted safe mode (no workspace settings,
no auto-accept, YOLO/AUTO_EDIT forcibly downgraded to DEFAULT per PR
https://github.com/google-gemini/gemini-cli/pull/7041; yolo auto-trust fix
https://github.com/google-gemini/gemini-cli/pull/19839). Standing rules live in a **policy engine**
(TOML `[[rule]]` with toolName/decision/priority/modes/argsPattern)
(https://google-gemini.github.io/gemini-cli/docs/get-started/configuration.html). Sandboxing is the
broadest here: Seatbelt (macOS), Docker/Podman (cross-platform), `windows-native` (icacls Low
integrity — persistent side effects, documented reset command), gVisor/runsc, LXC, plus
per-tool sandboxing and **sandbox expansion** (a denial proactively becomes a scoped approval
dialog) (https://geminicli.com/docs/cli/sandbox/). Hooks (`BeforeTool`/`AfterTool`, same exit-2
contract as Claude) and checkpointing/session history round it out. Note: as of June 2026 the site
banners Gemini CLI's transition to Antigravity CLI for unpaid/Google One tiers — track which binary
your comparison targets (same page). Admin kill-switches: `disableYoloMode`, `disableAlwaysAllow`,
`secureModeEnabled`.

## 6. Cursor

**Run Modes** (Settings → Agents → Approvals & Execution): `Auto-review` (default since 3.6:
allowlisted calls run, other shell goes to the sandbox when possible, the rest to an LLM classifier
— currently Claude 4.5 Haiku or GPT-5.4 Mini — which can allow, redirect, or prompt),
`Allowlist` (deterministic, optional sandbox), `Run Everything` (no prompts, no sandbox). The old
`Ask Every Time` was deprecated in 3.5 (https://cursor.com/docs/agent/security/run-modes,
https://cursor.com/changelog/auto-review). Steering (not enforcement) comes from
`permissions.json`: `terminalAllowlist` (prefix strings), `mcpAllowlist` (`server:tool`), and
`autoRun.allow/block_instructions` — plain-English classifier hints
(https://cursor.com/docs/reference/permissions, https://cursor.com/docs/enterprise/deployment-patterns).
`permissions.json` + `sandbox.json` (network domains, extra read/write paths, temp dirs) layer at
user and project scope; team dashboard policy wins (same refs). Sandboxing: Seatbelt on macOS,
Landlock+seccomp on Linux (kernel ≥6.2), **Windows only via WSL2**; plus always-on guards (browser,
file-deletion, external-file protections) and `.cursorignore` (same run-modes page). Approval UI
shows the full command with approve/deny/edit; allowlisted entries can be added inline. Open gap:
the CLI exposed only `allowlist`/`unrestricted` approval modes with no classifier — a forum feature
request for IDE/CLI parity (report says CLI Auto-review followed in July 2026)
(https://forum.cursor.com/t/auto-review-run-mode/161922). Docs stress Auto-review is "best-effort
convenience, not a security boundary" — use Allowlist + manual approval for strict control.

## 7. Cross-cutting: Windows, audit, sentiment

- **Windows sandboxing is the industry-wide hole.** Native enforcement exists only as Codex's
  experimental AppContainer token and Gemini's icacls-integrity approach (with persistent
  side effects); Claude and Cursor defer to WSL2; Cline and opencode have none; agentguard's `run`
  refuses on Windows outright. Any Cerebro comparison on this laptop must test WSL2 vs native paths
  separately and state which was used.
- **Approval UX converges** on once/session-or-saved-always/deny + inline diff or command preview;
  Cline's checkpoint Compare/Restore is the most forgiving undo model, while opencode's durable
  `always` and Codex's accept-and-persist-to-execpolicy are the most permanent.
- **Auditability is hook/transcript-based, never a SIEM.** Best patterns: hash-chained JSONL
  (agentguard), rotation + redaction (PolicyApprovalGate), fail-closed policy with log-only mode
  (cc-audit), session transcripts (Codex `~/.codex/sessions`, Gemini 30-day retention,
  Cline task history). Cerebro can adopt all of these cheaply since it already owns process spawn.
- **Two design poles:** deny-first fixed precedence (Claude, Codex profiles) vs last-match-wins
  (opencode) vs model-graded safety (Cline safe/all, Cursor classifier, Claude/Codox auto-review).
  Classifier modes cut prompt fatigue but are explicitly non-deterministic and non-boundaries —
  datacenter "safe" claims do not transfer to batch-1 laptop use; say so in any comparison.
- **Praise:** fewer prompts via classifiers/auto-approve; Cline checkpoints; Gemini sandbox
  expansion; opencode's tiny config. **Complaints:** Codex's untrusted-policy removal; Claude's
  prefix-match bypasses and silent Windows hook failures; Cursor's CLI parity gap; Cline/Claude
  YOLO footguns and compliance friction.



# Tauri v2 and WebView2 state

_Subtask fresh--tauri_

# Tauri v2 + WebView2 state — research notes (2026-09-22)

Focus: what changed since 2026-09-01, with emphasis on 2025–2026 primary sources.
Reference machine: Win11 Pro laptop, i9-12900H, 32 GB RAM, RTX 3070 Ti 8 GB, no NPU.
Cerebro is offline-first; only an opt-in update check may touch the network.

## 1. Latest releases (verified Sept 2026)

- Tauri core latest is **v2.11.5** (2026-07-01; v2.11.4 2026-06-28/30, v2.11.3 2026-06-17, v2.11.2 2026-05-16, v2.11.0 2026-04-30). v2.11.5 was a dependency unpin (`time` 0.3.53); v2.11.4 pinned `time <0.3.52` and fixed a Windows HDC-handle leak in DPI queries (also improves undecorated-window resize speed).
  Sources: https://v2.tauri.app/release/tauri/ , https://github.com/tauri-apps/tauri/releases
- **wry 0.56.0** (2026-07-30; 0.56.1 fixes a Windows WebView2 parent-subclass re-entrancy crash); tao 0.35.x/0.36. Bundler 2.9.4 (2026-06-28). `@tauri-apps/api` 2.11.x.
  Sources: https://v2.tauri.app/release/wry/v0.56.0/ , https://github.com/tauri-apps/wry/releases , https://v2.tauri.app/release/tauri-bundler/
- Tauri 2.11.0 features (2026-04-30): `command(rename)`, `eval_with_callback`, `data-tauri-drag-region="deep"`, WebView general-autofill toggle (Windows/WebView2 only), Android/iOS multi-window + file association, macOS content-process-termination handler, `dbus` feature flag for Linux theme detection.
  Source: https://v2.tauri.app/release/tauri/v2.11.0/
- wry 0.56.0: expanded cross-backend **permission-handler API** (12 WebView2 permission kinds, Allow/Deny/Default+Prompt), Windows **named WebView2 profiles** (`with_profile_name`), removed macOS/iOS `transparent`/`fullscreen` feature flags (always on) and the `protocol` feature flag (custom protocols always available).
  Sources: https://github.com/tauri-apps/wry/releases/tag/wry-v0.56.0 , https://github.com/tauri-apps/wry/pull/1654 , https://github.com/tauri-apps/wry/pull/1738

## 2. Security: capabilities, isolation, CSP

- **Capabilities** (`src-tauri/capabilities/*.json`, referenced from `tauri.conf.json`) are the allow/deny boundary: per-window/webview glob (`*`, `admin-*`), per-platform, plus `remote.urls` (URLPattern) for remote origins. No match = no IPC at all. Since 2026-04 (commit 7a2672f), remote origins ALWAYS go through ACL even without an AppManifest (breaking change for apps that let remote pages call custom commands).
  Sources: https://v2.tauri.app/security/capabilities/ , https://github.com/tauri-apps/tauri/commit/7a2672f7cf11652eab794781c63a5bb0124deb68
- **Permissions/scopes**: `plugin:allow-command` + `scope.allow/deny` (deny wins). E.g. `shell:allow-execute` with `{name, cmd, args, sidecar}`; fs scopes use `$HOME`-style globs. `assetProtocol.scope` (`allow`/`deny`/`requireLiteralLeadingDot`) gates `asset:`/`convertFileSrc` loads; dot-directories (`.cache`) need literal segments or `requireLiteralLeadingDot:false`.
  Sources: https://v2.tauri.app/security/permissions/ , https://v2.tauri.app/security/asset-protocol/
- **Isolation pattern** (opt-in `app.security.pattern.use="isolation"`): sandboxed iframe intercepts all IPC, AES-GCM encrypts via SubtleCrypto with per-run keys. Recommended but costs per-message overhead; ES modules don't load in the isolation app on Windows (build-time script inlining only).
  Source: https://v2.tauri.app/concept/inter-process-communication/isolation/
- **CSP**: opt-in; Tauri appends nonces/hashes for bundled code at compile time. `dangerousDisableAssetCspModification` disables that hardening. `freezePrototype:false` is the default (prototype-pollution → IPC-hijack vector); set `true`. `withGlobalTauri:false` default shrinks exposure.
  Sources: https://v2.tauri.app/security/csp/ , https://v2.tauri.app/reference/config/ , https://snyk.io/articles/5-common-security-misconfigurations/
- Threat model is explicit: capabilities mitigate frontend compromise, NOT malicious Rust, lax scopes, WebView 0-days, or supply-chain attacks; Tauri deliberately uses the OS WebView (evergreen patches) instead of bundling Chromium.
  Source: https://v2.tauri.app/security/

## 3. Updater + signing (relevant to Cerebro's opt-in update check)

- `tauri-plugin-updater` 2.x: mandatory **minisign Ed25519** signature (`signer generate`, `TAURI_SIGNING_PRIVATE_KEY`, pubkey embedded in binary). TLS enforced; `{{target}}/{{arch}}/{{current_version}}` endpoint variables; static JSON or 204/200 dynamic server; per-platform `OS-ARCH` keys; NSIS `.nsis.zip` must use **Store (method 0)** and BOM-free JSON or install fails.
  Sources: https://v2.tauri.app/plugin/updater/ , https://github.com/tauri-apps/plugins-workspace/blob/v2/plugins/updater/src/updater.rs , https://github.com/tauri-apps/tauri/issues/15601
- `bundle.createUpdaterArtifacts: true` (new apps) / `"v1Compatible"` (v1 migrants; removed in v3). Windows `installMode`: passive (default) / basicUI / quiet.
  Source: https://v2.tauri.app/plugin/updater/
- Fit for Cerebro: updater is the ONLY sanctioned network path — keep it off by default, gate behind explicit opt-in capability (`updater:default` split into check/download/install), ship a static JSON on a trusted HTTPS endpoint. Key loss = no future updates for installed users.

## 4. Sidecars (llama-server / whisper.cpp / SD-cpp pattern)

- `bundle.externalBin: ["binaries/my-sidecar"]` + per-triple binaries (`my-sidecar-x86_64-pc-windows-msvc.exe`); spawn via `app.shell().sidecar("my-sidecar")` (Rust) or `Command.sidecar('binaries/my-sidecar')` (JS); `execute()` vs `spawn()` need `shell:allow-execute` vs `shell:allow-spawn`; args allowlisted exactly or by regex validator (`"args": true` = footgun).
  Sources: https://v2.tauri.app/develop/sidecar/ , https://github.com/tauri-apps/plugins-workspace/tree/v2/plugins/shell
- Shell plugin kills spawned children on `RunEvent::Exit`. `shell.open` moved to opener plugin; scope validation RCE (CVE-2025-31477, fixed in shell 2.2.1).
  Sources: https://docs.rs/crate/tauri-plugin-shell/latest/source/src/lib.rs , https://github.com/advisories/GHSA-C9PR-Q8GX-3MGP
- Fit: llama-server is an ideal sidecar (single persistent process, HTTP on localhost, argv-only). Prefer spawning from **Rust** (no JS capability needed for the binary itself), stream SSE/stdout via Channel/events. One long-lived server ≫ per-inference respawn (process spawn ~50–200 ms + model reload seconds).

## 5. IPC streaming performance (token streaming matters most)

- `invoke` is JSON-based; raw `InvokeResponseBody` avoids one `serde_json::Value` round-trip (PR #10641: 150 MB response 29 s → 23 s custom-protocol, 54 s → 48 s postMessage, macOS).
  Source: https://www.github.com/tauri-apps/tauri/commit/d0510f52eb7efeabe00df5030cf10be16f99e178
- **Channel**: payloads < **8192 B JSON / 1024 B raw** go via direct `eval` (2× faster than fetch on WebView2 v135); larger payloads are stored server-side and pulled via `fetch` (`FETCH_CHANNEL_DATA_COMMAND`). `emit_raw`/`emit_raw_to` + `payload_raw` avoid JSON for events (PR #14269).
  Sources: https://docs.rs/tauri/latest/src/tauri/ipc/channel.rs.html , https://github.com/tauri-apps/tauri/pull/14269
- Third-party `tauri-conduit` benches (Rust-side only, NOT end-to-end): 25 B struct 722 ns → 330 ns (JSON-direct) → 78 ns (binary); 64 KB blob 2.27 ms → 0.83 ms → 0.20 ms; end-to-end small payloads are bridge-dominated (~300 µs, 1.3×), 64 KB 6.7 ms → 3.2 ms (JSON) / 0.6 ms (binary). Datacenter-style codec numbers do NOT transfer to batch-1 token streaming; per-token payloads (~10–50 B) are bridge-dominated.
  Source: https://github.com/userFRM/tauri-conduit/blob/master/BENCHMARKS.md
- `tauri-ipc-bench` (image streaming): returning `Vec<u8>`/`Response` from `invoke` arrives as a **JSON number array** (worse than base64); custom `bench-img://` protocol serving bytes fetched natively is fastest; never push big blobs via eval — signal via eval, pull via fetch.
  Sources: https://github.com/iceinvein/tauri-ipc-bench , https://github.com/tauri-apps/tauri/issues/13758
- Fit: token deltas are tiny → plain `Channel` is optimal (stays on the fast eval path); batch deltas if >8 KB JSON. Images/audio → custom protocol + `asset:` scope, not invoke.

## 6. WebView2 on Windows 11

- Evergreen runtime **preinstalled on Win11** (and pushed to Win10); Tauri installers still handle missing runtime: `downloadBootstrapper` (default, +0 MB, needs net) / `embedBootstrapper` (+1.8 MB) / `offlineInstaller` (+127 MB) / `fixedVersion` (+180 MB, you own patching — worse security) / `skip` (broken if absent).
  Sources: https://v2.tauri.app/distribute/windows-installer/ , https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution , https://v2.tauri.app/reference/webview-versions/
- `bundle.windows.minimumWebview2Version` (now top-level `windows`, NSIS-only key deprecated in CLI 2.11.0) forces bootstrapper when runtime is too old.
  Source: https://github.com/tauri-apps/tauri/releases/tag/@tauri-apps%2Fcli-v2.11.0
- wry 0.56 named profiles isolate cookies/IndexedDB/cache per webview on a shared runtime — useful for separating chat sessions or untrusted RAG previews.
  Source: https://github.com/tauri-apps/wry/pull/1738
- Pitfall: background windows + frequent eval pushes leak unmanaged WebView2 IPC memory → OOM crash (~2 GB WebView limit); prefer frontend-pull. No built-in OOM-reload event.
  Source: https://github.com/tauri-apps/tauri/issues/13758

## 7. SolidJS + installers + pitfalls

- `create-tauri-app` ships an official **SolidJS** template (TS flavor); community: `riipandi/tauri-start-solid` (Tauri v2 + plugins + TanStack Router + Tailwind, pushed 2026-03), `kvnxiao` TanStack-Start template (SPA default; SSR-prerender only for SEO), `tauri-controls-v2` Solid titlebar/controls.
  Sources: https://v2.tauri.app/start/create-project/ , https://github.com/riipandi/tauri-start-solid
- Solid's fine-grained reactivity (no VDOM diff) suits 60 fps token streaming; keep Tauri JS bridge (`@tauri-apps/api`, plugin packages) pinned to matching 2.11.x minor.
- Installers: **MSI = WiX v3, Windows-only builds, needs VBSCRIPT optional feature**; NSIS `-setup.exe` cross-compilable (cargo-xwin, last resort) with custom templates, hooks (`NSIS_HOOK_*`), per-user/per-machine/both modes, i18n. Sign via `windows.certificateThumbprint`/`signCommand`; since 2025 bundler signs WiX DLLs + NSIS plugin DLLs (7 signing calls — painful with cloud HSM; post-build signing of just the installer + regenerating `.nsis.zip`/`.sig` is a working workaround).
  Sources: https://v2.tauri.app/distribute/windows-installer/ , https://www.github.com/tauri-apps/tauri/commit/8d994f60fe05ec0f45cbe926506bbe10b0d36e3c , https://github.com/tauri-apps/tauri/issues/15601 , https://github.com/tauri-apps/tauri/pull/15422
- Must-fix CVEs/configs: **CVE-2026-42184** (origin confusion on Windows/Android `http://<proto>.localhost` mapping; fixed in 2.11.1 — Cerebro MUST use ≥2.11.1); shell-open RCE CVE-2025-31477 (use opener plugin, pin shell ≥2.2.1); iframe IPC CVE-2024-35222 (fixed); 2026-04 remote-ACL enforcement. Default-deny checklist: no `"args": true`, no `windows:["*"]` + remote, no `asset assess **/*`, `freezePrototype:true`.
  Sources: https://github.com/tauri-apps/tauri/security/advisories/GHSA-7gmj-67g7-phm9 , https://github.com/advisories/GHSA-C9PR-Q8GX-3MGP , https://github.com/tauri-apps/tauri/commit/7a2672f7cf11652eab794781c63a5bb0124deb68



# What rivals shipped recently

_Subtask fresh--rival-updates_

# Rival updates — Aug-Sep 2026 (checked 2026-09-22/23)

Method: websearch (11 queries) + webfetch opens of primary release pages. Every bullet cites a page actually opened in this run. Anything else is marked `unverified`. Batch-1 laptop numbers do not transfer from server/GPU-cluster claims.

## Cross-cutting backend: llama.cpp speculative decoding

- `llama-server --spec-type` now supports `draft-eagle3 | draft-dflash | draft-dspark | draft-mtp | ngram-mod | ngram-simple | ngram-map-k | ngram-map-k4v | ngram-cache | none`; `--spec-default` enables `ngram-mod`. Draft + draftless can be mixed; draftless wins on conflict. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- EAGLE-3: 1-layer draft reading target hidden states, needs `--target-model-dir` convert; listed drafts include Qwen3 1.7–32B + 30B-MoE, Gemma-4 31B/26B-A4B, gpt-oss 20B/120B. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- DFlash: block-diffusion draft emits whole block per forward pass (`--spec-draft-n-max` clamped to trained block); DSpark adds semi-autoregressive Markov head + `--spec-draft-conf-min P`; currently Qwen3 backbone only, Gemma4 planned. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Upstream EAGLE-3 PR claims 2–3x (e.g. Llama-3.1-8B Q4_K_M 121→274 tok/s) on contributor hardware — datacenter/GPU numbers, not batch-1 laptop; treat as upper bound, verify with soak not short bench. (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Cerebro takeaway: LM Studio 0.4.22 + KoboldCpp 1.119 both productized DFlash/DSpark/MTP in Aug 2026 (see below). Cerebro's allowlist should add `draft-dflash/dspark/mtp + ngram-mod` with argv-only emission before any autotuner.

## LM Studio — 0.4.25 (Sep 19), 0.4.24 (Sep 9), 0.4.23/0.4.22 (Aug 28), 0.4.21 (Aug 12)

- 0.4.22: DFlash/DSpark/MTP assistant drafters (needs llama.cpp engine 2.29.1+); tool-image results in OpenAI `/v1/responses|/chat/completions` + Anthropic `/v1/messages`; Linux AppImage/`.deb` fixes + in-app updates. (https://lmstudio.ai/changelog/lmstudio)
- 0.4.24: advanced llama.cpp argv overrides for GGUF load; DSpark/DFlash compat heuristics; Loaded-Instances ctx-length fix. (https://lmstudio.ai/changelog/lmstudio)
- 0.4.25: Mac Splash engine (M3+, macOS 26.4+). 0.4.23: Qwen 3.8 Flash Next reliability. (https://lmstudio.ai/changelog/lmstudio)
- Cerebro takeaway: direct rival already exposes argv overrides + new drafters; parity item.

## Ollama — v0.34.3 (Sep 19, latest stable), v0.34.4-rc0 (Sep 23), v0.34.2 (Sep 15), v0.34.1 (Sep 14), v0.34.0 (Sep 5), v0.33.3 (Sep 2), v0.33.2/0.33.1/0.33.0 (Aug)

- v0.34.0: ChatGPT Desktop reuse of Ollama models; OpenAI tool-search + response compaction; faster structured output on Apple Silicon. (https://github.com/ollama/ollama/releases)
- v0.34.1: MLX safetensors `ollama create` GA; GGUF creation now requires llama.cpp tooling; `typical_p` deprecated for new models; `/api/tags` 3.1s→294ms cold; MLX memory fixes. (https://github.com/ollama/ollama/releases)
- v0.33.3: Gemma4 images+audio on MLX; cached prompt tokens; honor GGUF default params. (https://github.com/ollama/ollama/releases)
- v0.32.15 (Aug 19): metadata cache halves TTFT ~995→524ms in their bench (hardware unverified — do not quote as laptop figure). (https://github.com/ollama/ollama/releases)
- Assets include `Ollama-darwin.zip`, `ollama-linux-*.tar.zst`, `install.ps1/.sh` (Windows via ps1). Backend: bundled llama.cpp + MLX/MLX-C. (https://github.com/ollama/ollama/releases)

## KoboldCpp — 1.121 (Sep 15, latest), 1.120 (Aug 29), 1.119 (Aug 16)

- 1.121: Minimax H3 media refs (audio + multi-image); `--ffncpu` (= `--n-cpu-ffn` partial offload); Adaptive-P sampler opt; MusicUI rework; SDUI runtime LoRA selector. (https://github.com/LostRuins/koboldcpp/releases)
- 1.120: `--usedirectio` (+ mlock/mmap combo); Qwen3.8-Flash-Next + Ling-3.0-flash support (warns bad quants exist). (https://github.com/LostRuins/koboldcpp/releases)
- 1.119: Minimax H3 video/I2V; Muse Glimmer + Qwen3.8 + jinja tool-calling; DSpark/DFlash support; 64 image/audio attachments. (https://github.com/LostRuins/koboldcpp/releases)
- Windows: `koboldcpp.exe` (CUDA pyinstaller), `oldpc` (CUDA11+AVX1), `nocuda` (Vulkan), mac-arm64. Default `http://localhost:5001`. (https://github.com/LostRuins/koboldcpp/releases)

## Open WebUI — v0.11.4 (Sep 21, latest), v0.11.3/v0.11.2 (Aug 31), v0.11.1 (Aug 25)

- v0.11.1: `Tool Permissions` Full/Approval cards (Allow/Deny, per-conversation default); `ask_user` builtin (2–3 options + free text, survives reload); streaming deltas (250k-char reply ~1.46GiB→1.90MiB, 787x in their 20-char-chunk test); Manage Models (load/unload/remove) for llama.cpp + LM Studio; release notes state security fixes withheld for embargo. (https://openwebui.com/blog/v0-11-1-the-model-learns-to-stop-and-ask) (https://github.com/open-webui/open-webui/releases)
- v0.11.4: slim image ~175MB (~89% smaller); standard image −170MB; terminal skills + AGENTS.md per-turn; auto skill discovery; `/skills:create`; terminal tabs; Staan EU search. (https://github.com/open-webui/open-webui/releases)
- Security: upgrade promptly; 16-CVE Sep-9 batch details `unverified` (not opened this run).
- Cerebro takeaway: copy approval-gate + ask_user + delta-streaming patterns; note Docker-only distribution (no portable zip).

## Jan — v0.8.4 (Jul 23, latest): nothing in Aug–Sep window

- v0.8.4: settings/creds → backend store + OS keyring (localStorage kept as downgrade snapshot); native `web_search/web_fetch`; per-model chat-template kwargs (llama.cpp); token counter (remote/MLX); handlebars 4.7.9 RCE bump; MCP approval-hang fix. (https://github.com/janhq/jan/releases)
- Windows: `Jan_0.8.4_x64-setup.exe` (55.1MB) + AppImage/deb/dmg listed. (https://github.com/janhq/jan/releases)
- Cerebro takeaway: keyring-over-localStorage + per-model template kwargs are the two ideas to steal.

## AnythingLLM — v1.16.2 (Sep 22, latest), v1.16.1 (Aug 27), v1.16.0 (Aug 13)

- v1.16.2: `LLMman-OSS` replaces Docker Model Runner; Google Vertex AI; `generate-image` agent skill; You.com default (ZDR); Ollama runtime 0.33.3 bump; KoboldCPP provider overhaul. (https://github.com/Mintplex-Labs/anything-llm/releases)
- v1.16.1: Foundry Local embedded (WinML CPU/GPU/NPU, Win 11 24H2+); GenieX Snapdragon (replaces old NPU engine, re-download needed); CoT/activity rollup; abort warns before killing background inference. (https://github.com/Mintplex-Labs/anything-llm/releases)
- v1.16.0: `/img` generation; folder drag hierarchy; mid-session tool toggle; abort now kills inference (no ghost inference). (https://github.com/Mintplex-Labs/anything-llm/releases)

## Msty Studio — no Aug–Sep Studio release (latest opened: 2.9.6/2.9.5 Jul 24, 2.9.4 Jul 17, 2.9.1 Jul 10)

- 2.9.1: GGUF ready-without-restart; Llama.cpp preset refresh fix. 2.9.0: Context Studio + Dynamic pull mode. 2.8.0: Discover Hub + One Model Hub + provider tool presets. (https://msty.ai/resources/changelog/studio/)
- Nexus 0.5.0 (Sep 8 Console+Pro) and Go 0.15.0 (Aug 6 local STT) reported in vendor blog/newsletter — `unverified` (pages not opened this run).
- Cerebro takeaway: One Model Hub + pull-mode context are UX to beat; local-GGUF-import without restart is a parity bug to avoid.

## goose — v1.51.0 (Sep 17, latest), v1.50.0 (Sep 8), v1.49.0 (Sep 3), v1.48.0 (Aug 27), v1.47.0 (Aug 21)

- v1.51.0: Desktop→state-machine loop via ACP; EUrouter declarative provider; GPT-live API; gateway pairing allowlist; opt-in terminal bell; streamed-thinking ordering. (https://github.com/block/goose/releases)
- v1.49.0/1.48.0: auto-updater (desktop); background extension load; web-search/browser-use built-ins; PreToolUse `on_failure`; ~30-entry hardening batch (tool allowlists, secret-suffix redaction, symlink confinement, permission-denies-precedence). (https://github.com/block/goose/releases)
- Install: 39 assets incl. `download_cli.sh`, rpm/deb/tarbz2 (Windows via portable extract + `Goose.exe` per docs — `unverified` installer name).

## Cline — SDK harness migration at 100% (Aug 23)

- Dual-bundle loader (`legacy/`+`next/`+46KB loader, PostHog flag, auto-fallback, shared state, union `package.json`) worked around no-gradual-rollout Marketplace. (https://cline.bot/blog/how-we-migrated-11-million-users-to-clines-biggest-harness-upgrade)
- `task.mistake_limit_reached`: 6.34%→0.62% (10x); per-model 6–11x (Sonnet-5 11x, DeepSeek V4 Flash 10x). Open-weights tbench.ai: Kimi K3 82.02% vs 71.9/76.4; GLM 5.3 Flash 64.0%; DS V4 Flash 60.67% — vendor-run, verify independently. (https://cline.bot/blog/how-we-migrated-11-million-users-to-clines-biggest-harness-upgrade)
- Thesis: 2024 XML-tag harness taxes 2026 RL-native-tool models — validates Cerebro's harness-first bet. Extension/Desktop/CLI/SDK version numbers `unverified` (release pages not opened).

## DSH Desktop (anywhere-labs) — community DeepSeek-Harness desktop

- MIT; Windows x64 NSIS (`https://www.dshdesktop.cn/api/downloads/windows`) + macOS Universal DMG; no extra env; Electron supervisor spawns Harness child with backoff recovery; everything-a-plugin incl. desktop shell via `dsh-community-fabric` typed IPC. (https://github.com/anywhere-labs/dsh-desktop)
- 28.6k stars / 1.4k forks at open; mobile (iOS/Android) via encrypted WebSocket + short-lived tokens (v2.0.9+); LAN access explicitly unauthenticated — trusted-network-only warning. (https://github.com/anywhere-labs/dsh-desktop)
- Update check sends `X-DSH-Desktop-Version/Channel/Installation-Id` (random UUID, not HW-derived). Latest version number `unverified` (releases page not opened).



# Self-verification and reasoning effort

_Subtask fresh--verification_

# Self-verification and reasoning effort — fresh research (late Sept 2026)

Focus: what changed since 2026-09-01, plus the 2025–2026 baseline that matters for a
Rust harness spawning llama.cpp on the reference laptop (i9-12900H, RTX 3070 Ti 8 GB,
32 GB RAM, batch 1, fully offline). All numbers carry their own conditions; datacenter or
large-batch figures are flagged as non-transferable.

## 1. Reasoning effort levels in open models

**gpt-oss (OpenAI, Apache-2.0, Aug 2025): low / medium / high.** Trained variable-effort
reasoning; the level is set by a `Reasoning: low|medium|high` line in the Harmony system
message, default medium (https://openai.com/index/introducing-gpt-oss/,
https://arxiv.org/html/2508.10925v1, https://github.com/openai/gpt-oss).
Model-card test-time scaling is smooth but log-linear: gpt-oss-120b AIME-2024 no-tools
goes 56.3 → 80.4 → 95.8 across low/med/high; 20b goes 42.1 → 80.0 → 92.1; SWE-Bench
Verified 120b goes 47.9 → 52.6 → 62.4 (https://openai.com/index/gpt-oss-model-card/).
Cost implication at batch 1: longer CoT = linearly more decode tokens at ~same tok/s, so
high effort costs roughly low's tokens ×3–5 on hard tasks. The Harmony format is mandatory;
without it the model "will not work correctly"
(https://github.com/openai/openai-cookbook/blob/main/articles/openai-harmony.md,
https://github.com/openai/gpt-oss/blob/main/README.md). The reference `gpt_oss.chat`
CLI exposes `-r/--reasoning-effort` defaulting to low (https://github.com/openai/gpt-oss).

**Qwen3 hybrid thinking (Apr 2025): hard switch + soft switch.** `enable_thinking=True/False`
in `apply_chat_template` selects thinking vs non-thinking mode; with thinking enabled,
`/think` and `/no_think` suffixes flip per-turn, last instruction wins
(https://qwenlm.github.io/blog/qwen3/, https://huggingface.co/Qwen/Qwen3-8B).
Local note: llama.cpp did not originally expose the hard switch; the workaround was a custom
`--chat-template-file` forcing `enable_thinking=False`
(https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html). Qwen3 support in
llama.cpp dates to build b5092
(https://qwen.readthedocs.io/en/stable/run_locally/llama.cpp.html).

**Qwen3.8-Max (Aug 2026): tiered `reasoning_effort`.** API levels `low` (fast), `medium`
(balanced), `xhigh` (default, deep); `reasoning_effort` and `thinking_budget` cannot be set
together; `preserve_thinking` on by default (https://qwen.ai/blog?id=qwen3.8,
https://docs.qwencloud.com/developer-guides/text-generation/thinking). The qwen-code CLI
grew a 5-tier `/effort` ladder (low/medium/high/xhigh/max) with per-provider
translate-and-clamp, e.g. OpenAI `max→xhigh`, DeepSeek `low/medium→high, xhigh→max`
(https://qwenlm.github.io/qwen-code-docs/en/design/2026-06-30-unified-reasoning-effort-cli/,
https://github.com/QwenLM/qwen-code/pull/8472). Only `qwen3.8-max*` reads the tiered field;
older Qwen hybrids read only the `enable_thinking` boolean
(https://github.com/QwenLM/qwen-code/pull/8488).

**DeepSeek R1: no clean effort knob.** The R1 API exposes reasoning as `reasoning_content`
but no dedicated budget parameter; `max_tokens` caps reasoning+answer jointly, so a long
trace can starve the visible answer (observed failure mode, mid-2026 API guide:
https://pastagi.com/guides/control-llm-reasoning-effort-api-guide/). R1 training used
16 samples/query, up to 32K tokens pre-checkpoint and 65K after
(https://arxiv.org/pdf/2501.12948v2); evals recommend temperature 0.6, top-p 0.95
(https://github.com/deepseek-ai/DeepSeek-R1/). A 2025 feature request proposes s1-style
budget forcing (`max_thinking_tokens`, append-"Wait") for R1/GRPO but it is unverified
discussion, not shipped code (https://github.com/deepseek-ai/DeepSeek-R1/issues/252).

## 2. What llama.cpp actually exposes ( Sept 2026 state)

- `--reasoning on|off|auto` sets `enable_thinking` via template kwargs; `--reasoning-budget N`
enforces a real token budget via a sampler grammar with delayed launch, plus
`--reasoning-budget-message` (e.g. "...budget exceeded, need to answer") to land the
forced stop coherently (https://github.com/ggml-org/llama.cpp/pull/20297,
https://aihaven.com/news/llama-cpp-reasoning-budget-update/).
- Per-request override: `thinking_budget_tokens` in the OpenAI-compatible request body
(applies only if no CLI budget set), and `chat_template_kwargs: {enable_thinking: false}`
for Qwen-family per-request control
(https://github.com/ggml-org/llama.cpp/discussions/21445,
https://github.com/ggml-org/llama.cpp/pull/13196).
- gpt-oss support: native MXFP4 + Harmony parsing landed Aug 2025
(https://github.com/ggml-org/llama.cpp/pull/15091,
https://github.com/ggml-org/llama.cpp/pull/15181); effort via
`--chat-template-kwargs '{"reasoning_effort":"low"}'` per request
(https://huggingface.co/openai/gpt-oss-20b/discussions/47).
- Sharp edges (all primary-source issues): `--reasoning-budget 0` needed `--jinja` to take
effect and behaved differently in CLI vs server
(https://github.com/ggml-org/llama.cpp/pull/13771); `--reasoning-budget 0` did nothing for
gpt-oss early on (https://github.com/ggml-org/llama.cpp/issues/15266); as of Mar 2026
`--reasoning off` still mapped to `Reasoning: medium` instead of low for gpt-oss, and
`--reasoning-budget 0` likewise (https://github.com/ggml-org/llama.cpp/issues/20458);
Qwen3.5 `enable_thinking:false` repeatedly ignored without the exact flag combination
(https://github.com/ggml-org/llama.cpp/issues/20182). Budget without the transition message
cratered Qwen3.5-9B HumanEval 94% → 79% in maintainer tests; with it, ~89%
(https://github.com/ggml-org/llama.cpp/pull/20297). For Cerebro's allowlist this means:
template-kwarg path and budget path must both be allowlisted and soak-tested per model.

## 3. Self-refine / intrinsic self-correction: the negative result stands

- Huang et al. (ICLR 2024): without oracle labels, self-correction *degrades* reasoning on
GSM8K/CommonSenseQA/HotpotQA; GPT-3.5 flips correct→incorrect (8.8%) more than
incorrect→correct (7.6%); multi-agent debate ≡ self-consistency at equal sample count
(https://arxiv.org/abs/2310.01798).
- Kamoi et al. (TACL): no prior work shows successful prompt-only self-correction on general
tasks under fair settings; it works with reliable external feedback or large-scale
fine-tuning (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/When-Can-LLMs-Actually-Correct-Their-Own-Mistakes).
- Kambhampati planning study: GPT-4 self-critique collapses Game-of-24 5%→3% and graph
coloring 16%→2% (false-negative rates 80–97%); sound external verifier restores gains
(https://www.alphaxiv.org/abs/2402.08115).
- ACL 2025 dark-side analysis: failure mechanisms are answer wavering, prompt bias (failed
cases have 4.4–6.1× longer prompts), and human-like biases (overthinking, cognitive
overload, perfectionism with 1.7–3.1× longer outputs); mitigations: repeat-the-question
and <10-sample SFT (https://p.rst.im/q/aclanthology.org/2025.acl-long.1314.pdf).
- Small models are worse, not better, at this: LLaMA-1B overthinks 45.4% more than LLaMA-70B
(https://doi.org/10.1609/aaai.v40i40.40723); ≤3B models lose up to 75% accuracy after 8K
long-CoT SFT examples via error accumulation (https://doi.org/10.18653/v1/2025.emnlp-main.251);
SCORE: small-model self-refine needs a *strong* (GPT-4-class) verifier; weak self-verifiers
mislead (https://doi.org/10.18653/v1/2024.findings-acl.924).

## 4. What does work: train it in, or verify from outside

- **SCoRe** (online multi-turn RL on self-generated traces, two-stage + correctness-flip
bonus): +15.6% MATH / +9.1% HumanEval self-correction on Gemini 1.0 Pro / 1.5 Flash;
plain SFT on correction traces collapses to trivial edits (https://arxiv.org/html/2409.12917v1).
- **ReVISE** (ICML 2025): curriculum preference learning for verify-then-rethink plus
confidence-aware decoding; efficient test-time scaling without an external verifier
(https://proceedings.mlr.press/v267/lee25ab.html).
- **Self-verification RL** (NeurIPS 2025, Qwen2.5-Math-7B / R1-Distill-1.5B): unifies
generation+verification in one RL run with an online policy-aligned buffer; weighted
aggregation over N samples beats external-RM best-of-N/beam search at inference
(https://proceedings.neurips.cc/paper_files/paper/2025/file/07320f63b9d44f0f128f68ab56721729-Paper-Conference.pdf).
Complementary 2026 result: training *only* to self-verify matches generation-training on
math and shortens traces (https://arxiv.org/html/2602.07594v1).
- **SSR** (Nov 2025): Socratic (sub-question, sub-answer) decomposition + step-level
re-solve confidence + targeted refine; beats holistic self-refine on 5 benchmarks but at
"substantially higher" compute (https://arxiv.org/abs/2511.10621).
- **External grounding wins**: code-execution feedback (self-debug with unit tests) is the
consistently positive case in every survey above; treat the compiler/test-runner as the
verifier, not the model's own opinion.
- **Generative verifiers**: GenRM (next-token `Yes/No` + CoT rationale, Best-of-N):
5%→45.3% algorithmic, 73%→93.4% GSM8K (Gemma2-9B verifier over Gemini 1.0 Pro solutions)
(https://arxiv.org/abs/2408.15240); GenPRM adds step-level CoT+code verification, 1.5B
beats GPT-4o on ProcessBench (https://ojs.aaai.org/index.php/AAAI/article/view/40797).
Caveat for batch 1: at fixed FLOPs, plain self-consistency beats GenRM until ~8× compute;
GenRM needs ~128× more for +3.8% (https://doi.org/10.48550/arxiv.2504.01005). HSRM (Aug 2026)
cuts verifier cost by reading frozen-generator hidden states with a ~2M-param head instead
of re-reading text (https://arxiv.org/abs/2608.30841). None of these ship in llama.cpp;
all are research-only for Cerebro.

## 5. Multi-path self-consistency: effective, linear-cost, now with early exits

Classic SC (sample N, majority vote) still helps weak-model/hard-task combos, but costs N×
tokens at batch 1. Proven economizers: ESC early-stops on window agreement (GSM8K −80.1%,
MATH −33.8% samples at parity: https://proceedings.iclr.cc/paper_files/paper/2024/file/3fe2a777282299ecb4f9e7ebb531f0ab-Paper-Conference.pdf);
DSC routes easy questions to single-sample via LLM difficulty ranking (−65% cost on GPT-4
vs SC: https://doi.org/10.18653/v1/2025.findings-naacl.383); CISC weights votes by P(True)
confidence (−41–46% paths: https://aclanthology.org/2025.findings-acl.1030.pdf); Slim-SC
prunes similar chains step-wise (−45% latency, −26% KV usage on R1-Distill:
https://aclanthology.org/2025.emnlp-main.1750.pdf); ranked voting (Borda/IRV over ranked
answers) adds +2.7–5% over majority vote on small models
(https://aclanthology.org/2025.findings-acl.744.pdf). Counterpoint (Nov 2025): on strong
models near ceiling (Gemini 2.5 Pro, MATH-500 98% single-pass), 15–20 samples buy only
+0.4–1.6% at ~15–20× tokens, sometimes *hurting* (https://arxiv.org/html/2511.00751).
And under a fixed token budget, parallel independent paths beat one ever-longer "Wait..."
trace by up to ~20% (https://proceedings.neurips.cc/paper_files/paper/2025/file/fc067ac218430c409d6f65403328f740-Paper-Conference.pdf).
Cerebro rule of thumb: single pass by default; spend parallel samples only when single-pass
reliability is demonstrably low (hard coding tasks), with ESC-style early stop.

## 6. Concise reasoning: the cheapest verified win

- CCoT ("think step-by-step + be concise"): −48.7% response length at parity accuracy on GPT-4;
GPT-3.5 loses −27.7% on math only; −22.7% per-token cost
(https://arxiv.org/pdf/2401.05618v3.pdf).
- Chain-of-Draft (≤5 words per step): matches CoT at ~7.6–20% of tokens; GSM8K 91% vs 95%
at 44 vs 205 tokens; sports-understanding −92% tokens
(https://www.alphaxiv.org/abs/2502.18600).
- Token-complexity study (31 prompts × 6 LLMs): all brevity prompts sit on one universal
length–accuracy curve — length matters more than wording; optimal per-question compression
would be 3–11× but prompts only achieve ~1.3×; a verifier router gets closest
(https://arxiv.org/html/2503.01141).
- Overthinking is real and small-model-skewed: "2+3?" can trigger thousand-token traces;
verbosity can flip right answers wrong (https://arxiv.org/html/2503.16419,
https://arxiv.org/pdf/2505.00127). For Cerebro this is pure prompt engineering —
no runtime change, no allowlist impact — and it multiplies every other saving.

## 7. Budget forcing ("Wait" / forced stop)

s1 (Qwen2.5-32B + 1K traces): cap thinking by appending end-of-thinking delimiter; extend by
suppressing it and appending "Wait" → AIME24 50%→57%, beats o1-preview by up to 27% on
MATH/AIME24; flattens after ~6 extensions, loops beyond that
(https://arxiv.org/abs/2501.19393, https://simplescaling.github.io/). Prompt-only
length control fails (models can't count tokens); "Wait" beats "Hmm"/"Alternatively".
Cerebro can approximate the *cap* side today with `--reasoning-budget N` + transition
message; the *extend* side has no llama.cpp equivalent (would need client-side re-prompt).

## 8. Batch-1 laptop cost model (applies to every item)

At batch 1 there is no throughput amortization: wall time ≈ prefill + tokens/tok/s, and the
reference numbers (29 tok/s MoE offloaded, 2–12 tok/s dense) dominate everything. So: halving
thinking tokens halves latency; N parallel samples cost N× latency unless run concurrently
against VRAM headroom the 8 GB card does not have for 20–30B models; reasoning tokens bill
against the same context window as answers (DeepSeek R1 shares `max_tokens`;
Qwen3.5 agents starved at max_tokens 600–700: https://github.com/ggml-org/llama.cpp/issues/20182).
Short benchmarks overstate MoE tok/s ~3× vs sustained soak (existing Cerebro measurement) —
apply the same discount to any paper tok/s claim. Default Cerebro policy implied by the
evidence: effort Low/off for chat and easy code, Medium for standard coding, High only for
verifiably hard tasks; concise-style system prompts everywhere; self-repair loops gated on
compiler/test output, never on the model's own "looks good".



# Local memory and document search

_Subtask fresh--memory-rag_

# Local memory + document search (fresh research, 2026-09-22)

Scope: persistent cross-chat memory + local doc search for Cerebro (offline Windows-first Rust harness spawning llama.cpp via argv). Focus on changes since 2026-09-01, preferring 2025-2026 primary sources. All claims cite pages actually opened.

## 1. Embeddings that run in llama.cpp

**Qwen3-Embedding (0.6B/4B/8B) is the current default pick.** The family offers 0.6B, 4B, 8B sizes, 32K context, Matryoshka (MRL) dims and instruction-aware queries; the 8B model scored 70.58 on MTEB multilingual (June 2025) (https://github.com/QwenLM/Qwen3-Embedding). Key detail: the embedding models use last-token pooling and callers should prepend `Instruct: <task>\nQuery:` on the query side; dropping the instruction costs ~1-5% retrieval (https://github.com/QwenLM/Qwen3-Embedding). llama.cpp embedding support was merged in PR #15023 and works with `--embedding --pooling last` (reported in https://github.com/ggml-org/llama.cpp/pull/14029). MemX (Mar 2026 Rust paper) uses Qwen3-Embedding-0.6B at 1024 dims as its live embedder (https://arxiv.org/html/2603.16171v1).
Cerebro fit: 0.6B Q4_K_M (~0.4 GB) easily fits the 8 GB Ampere card alongside a coder model; serve a second `llama-server` on 127.0.0.1 with `-ngl 99 --embedding --pooling last`.

**BGE-M3 GGUF remains the versatile fallback.** BGE-M3 is 568M params, 1024 dims, 8192 context, MIT, and uniquely emits dense + sparse (lexical) + ColBERT multi-vector from one pass; the model card recommends hybrid retrieval + rerank pipeline (https://huggingface.co/gpustack/bge-m3-GGUF). GGUF quants (Q4_K_M ~438 MB, Q8_0 ~635 MB) run under `llama serve -hf gpustack/bge-m3-GGUF:Q4_K_M` (https://huggingface.co/gpustack/bge-m3-GGUF). Use when multilingual + sparse without a second index matters; otherwise Qwen3-0.6B is newer and smaller.

**Nomic v2 MoE GGUF (multilingual, 512 ctx).** nomic-embed-text-v2-moe is 475M total / 305M active MoE (8 experts, top-2), 768 dims with Matryoshka down to 256, Apache-2.0, GGUF from Q2_K (~261 MB) to F16 (~913 MB); served via `llama-server -m <file> --embeddings` (https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF). Requires `search_query:` / `search_document:` prefixes and caps at 512 tokens (https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF). Good for cheap prefix-truncated vectors; bad for long docs (prefer Qwen3/BGE-M3 8K-32K).

**Jina v5 GGUF is cutting-edge but fork-gated (unverified upstream).** Search snippets describe jina-embeddings-v5 omni/text GGUF families with imatrix quants and `--pooling last/mean`, but note the omni multimodal build needs the `feat-v5-omni` llama.cpp fork, not upstream. Treat as emerging; pin only if you re-verify against the HF repos. Skepticism: datacenter batch-64 throughput numbers do not transfer to batch-1 laptop ingest; verify locally with `-b 64 -ub 64` sweeps.

## 2. Rerankers in llama.cpp (the biggest 2025-09 change)

**Qwen3-Reranker now works in llama.cpp (`--reranking`).** The original PR #14029 framed the model as a 2-label (`yes`/`no`) classifier whose softmaxed `yes` score is the relevance score (https://github.com/ggml-org/llama.cpp/pull/14029). The successor PR #15824 was merged (commit b5bd037, 2025-09-25): converter detects `# Qwen3-Reranker` README, writes `PoolingType.RANK` + `cls.output` head + a baked `rerank` chat template, graph applies last-token pooling for QWEN3 RANK plus softmax (https://github.com/ggerganov/llama.cpp/commit/b5bd037832bcb8ed3086dfe26ce9090bea989af1). Server exposes `POST /reranking` (aliases `/rerank`, `/v1/rerank`, `/v1/reranking`) behind `--rerank/--reranking`, mutually exclusive with `--embedding` (https://github.com/ggml-org/llama.cpp/blob/873c825611d9cb76427931b5e74642bade4853dd/tools/server/README.md). Qwen3-Reranker scores: 0.6B reaches 65.80 MTEB-R / 67.28 MLDR; 4B reaches 69.76 / 69.97 / 81.20 code (https://github.com/QwenLM/Qwen3-Embedding). Practical note from the thread: run two servers (embed on :8080, rerank on :8081) and for `llama-embedding` add `--embd-normalize -1` (https://github.com/ggml-org/llama.cpp/pull/14029). A follow-up PR adds optional per-request `instruction` override for the rerank template (emerging, unverified merge status). Cerebro: spawn rerank server with `--reranking --alias qwen3-reranker-0.6b`, POST `{query, documents}`, sort by `relevance_score`, cap at top 5-10 to bound latency on the 3070 Ti.

## 3. Hybrid search: BM25 + vectors + RRF

**The consensus pipeline is vector recall + keyword recall fused by Reciprocal Rank Fusion (k=60).** MemX formalizes it: cosine vector path + FTS5 keyword path, RRF merge, four-factor re-rank (semantic 0.45, recency 0.25, frequency 0.05, importance 0.10), z-score+sigmoid normalize, dedup, and a low-confidence rejection gate (https://arxiv.org/html/2603.16171v1). Mem0 v3 (Apr 2026) independently moved to multi-signal retrieval (semantic + BM25 + entity matching fused) with single-pass ADD-only extraction on the managed platform (https://github.com/mem0ai/mem0). Graphiti retrieves via semantic + BM25 + graph traversal with no LLM in the loop (https://github.com/getzep/graphiti). Rust precedent crates do exactly this: `kwaai_rag::bm25` (tantivy BM25 + RRF k=60) and the `hybrid-search` crate (tantivy BM25 + nalgebra cosine + weighted RRF) per search results. Cerebro can implement this in-process with zero new servers.

**Rejection gate matters more than recall tuning.** MemX rejects when keyword set is empty AND max vector sim < tau (recommended tau=0.50 from a {0.48,0.50,0.52,0.64} sweep); on 43 custom queries Hit@1 was 91-100% with Miss-Empty 50-75% (https://arxiv.org/html/2603.16171v1). Copy this: never inject low-confidence memories; return empty and let the chat model say it does not know.

**Chunking beats pipeline tweaks.** On LongMemEval (500 Qs, up to 220K records) fact-level atomic storage hit 51.6% Hit@5 / 0.38 MRR, roughly doubling session-level storage; knowledge-update queries jumped 30.8%->75.6% (https://arxiv.org/html/2603.16171v1). Tag-signature dedup helped tagged template data (+2.7pp Hit@3) but hurt tag-free atomic facts (-5pp), so make dedup adaptive (https://arxiv.org/html/2603.16171v1). FTS5 cut keyword latency ~1100x vs LIKE at 100K rows (3305 ms -> 2.9 ms), total search 88 ms at 220K rows (https://arxiv.org/html/2603.16171v1).

## 4. Rust-native vector stores (no server)

**tantivy 0.26.x is the Rust BM25 workhorse.** tantivy is a Lucene-inspired Rust full-text crate with BM25 scoring, incremental indexing, mmap directories, and Windows/Linux/macOS support; latest 0.26.2 shipped 2026-09-08 (https://docs.rs/crate/tantivy/latest). Use it for the keyword leg and for code-identifier search.

**sqlite-vec = exact brute-force, honest and linear.** sqlite-vec is a C SQLite extension (`vec0` table) doing SIMD brute-force KNN with perfect recall and zero index to corrupt, but no ANN release (tracking issue #25 open since 2024, as of mid-2026) (https://dreaming.press/posts/lancedb-vs-sqlite-vec-vs-duckdb.html). Comfort zone is ~100K to low-millions of 1024-d vectors; past ~1M rows linear scans hurt. Best when memory already lives in SQLite and scale is modest.

**LanceDB = embedded ANN + versioning for churning memory.** Rust core on the Lance columnar format with IVF-PQ/HNSW indexes, MVCC time-travel, ACID, zero-copy column evolution, larger-than-RAM mmap (https://dreaming.press/posts/lancedb-vs-sqlite-vec-vs-duckdb.html). Choose it when memory is appended/re-embedded/pruned all day or exceeds RAM. Cost: a second format/dependency vs a SQLite extension.

**DuckDB vss = analytical, with a persistence asterisk.** Real HNSW but disk persistence needs `hnsw_enable_experimental_persistence` with no WAL recovery and whole-index checkpoint rewrites (https://dreaming.press/posts/lancedb-vs-sqlite-vec-vs-duckdb.html). Fine for batch analytical search; wrong for an agent mutating memory all day. Rule of thumb: sqlite-vec until ~1M vectors or P99 pain, LanceDB to stay embedded at scale, Qdrant server only when memory becomes shared infrastructure.

## 5. Memory extraction + editing UX (what to copy)

**Mem0 v3 + OpenMemory MCP (production).** Mem0 reports 92.5 LoCoMo / 94.4 LongMemEval on its managed stack with ~7K tokens/call (platform-only optimizations; OSS directionally similar) (https://github.com/mem0ai/mem0). The portable idea is OpenMemory: a local MCP memory server (`add_memories`, `search_memory`, etc.) shared across Claude/Cursor/Windsurf with a dashboard for browse/edit/delete (per search results). Cerebro: expose the same tool trio over its local API + GUI panel; keep store local-first.

**MemU (14.4k stars): Markdown-wiki memory, agent does the judging.** Core logic ~500 lines; per-host sidecar binaries mine session logs (e.g. `~/.claude/projects/*.jsonl`, `~/.codex/sessions/**/*.jsonl`, SQLite for OpenClaw/Hermes) and patch instruction files (`AGENTS.md`/`CLAUDE.md`/`SOUL.md`); local backends are sqlite (brute-force cosine) or Postgres+pgvector (https://github.com/NevaMind-AI/MemU). Copy: scheduled background distill -> Markdown skill/memory files -> embed name+description -> `retrieve` injection before answering.

**Letta: filesystem beats fancy tools; sleep-time moves work off the critical path.** A Letta agent with only file tools (`search_files`, `grep`, `open`) hit 74.0% on LoCoMo with gpt-4o-mini, above Mem0's 68.5% graph variant, arguing retrieval UX matters more than graph-vs-vector choice (https://www.letta.com/blog/benchmarking-ai-agent-memory/). Sleep-time compute adds a background agent that rewrites core-memory blocks between turns so consolidation never blocks replies; primary can be a fast small model while sleeper is stronger (https://www.letta.com/blog/sleep-time-compute/). Cerebro: nightly/off-idle summarizer process that compacts facts; primary chat never waits on extraction.

**Graphiti temporal graph (31.1k stars) for facts that change.** Entities/edges carry validity windows; contradictions invalidate (not delete) old edges; retrieval is hybrid without LLM summarization at query time (https://github.com/getzep/graphiti). Needs Neo4j/FalkorDB/Neptune/Kuzu (Kuzu deprecated) plus OpenAI-style LLM+embedder endpoints, including Ollama/local OpenAI-compatible ones (https://github.com/getzep/graphiti). Too heavy as a default for an 8 GB laptop, but its bi-temporal invalidation + episode provenance is the right model for entitlements/pricing/project facts; approximate with `supersedes` links in SQLite.

## 6. Document parsing on Windows (Rust-first)

**docling.rs is the standout Rust-native path.** It ports Python docling to a single static binary with declarative formats 20-60x faster at ~60x less RAM and PDF ML 4.3x faster at 2.3-2.6x less peak RAM (methodology in-repo), covering DOCX/PPTX/XLSX/legacy Office/iWork/EPUB/ODF/HTML/PDF/images/audio/video plus Markdown/docling-JSON/DocLang output and a `docling-rag` chunk->embed->search layer on SQLite+sqlite-vec or Postgres+pgvector (https://github.com/docling-project/docling.rs/blob/master/README.md). Caveats: PDF ML needs pdfium + ONNX downloads, pdfium is Linux-x64-only for now; declarative formats work out of the box; WASM build converts Office formats fully client-side (https://github.com/docling-project/docling.rs/blob/master/README.md). Cerebro: shell out to `docling-rs` CLI or bind the `docling` crate for Office/HTML/PDF-text; keep Python docling or VLM fallback for scanned PDFs until pdfium-on-Windows lands.

## 7. Privacy controls (offline == feature)

MemX's template is one libSQL file on the user's machine, no third-party dependency (https://arxiv.org/html/2603.16171v1). Harden it: (a) default-off update check only (already Cerebro policy); (b) per-user scoping + auth on any local server port (Mem0 self-host now enables auth by default); (c) explicit read vs retrieval counters so admin views do not pollute ranking (https://arxiv.org/html/2603.16171v1); (d) episode provenance per fact + validity windows so users can audit/delete what produced a memory (https://github.com/getzep/graphiti); (e) PII redaction gate before embedding (regex + small classifier; unverified as a named crate — implement in Rust, do not claim otherwise); (f) memory dashboard with edit/delete per item (OpenMemory pattern). All stores above (sqlite-vec, tantivy, LanceDB embedded) open no network port, removing the server-CVE family.

## Recommendation for the reference laptop

Ship v1 as: Qwen3-Embedding-0.6B Q4_K_M (`--embedding --pooling last`) + Qwen3-Reranker-0.6B Q4_K_M (`--reranking`) as two pinned `llama-server` children; tantivy BM25 + cosine vector + RRF(k=60) + tau=0.50 rejection in-process; single libSQL/SQLite file with FTS5 + `vec0`; fact-level atomic memories with `supersedes` links; idle-time summarizer (Letta-style); docling.rs for ingestion; GUI memory panel with provenance + delete. Defer Graphiti-server and Jina-omni fork until fit is proven. Re-verify every throughput claim at batch 1, stated ctx depth and sampler — short benches overstate MoE ~3x.



# New security issues in local AI tools

_Subtask fresh--security_

# fresh--security — New security issues in local AI tools (Aug–Sep 2026)

Date: 2026-09-22. Focus: what changed since 2026-09-01, plus the Aug 2026 backlog that still matters.
All URLs below were opened and read. Anything not verifiable is marked `unverified`.
Scope note: Cerebro spawns llama.cpp as a child process with argv only, Windows-first, fully offline.
So the issues that matter most are: (a) llama.cpp/Ollama memory-safety bugs, (b) malicious model
files, (c) DNS rebinding against localhost APIs, (d) MCP/tool poisoning, (e) Tauri/WebView2 IPC,
(f) coding-agent supply chain.

## 1. llama.cpp

**RPC GRAPH_RECOMPUTE use-after-free → unauthenticated RCE (CVE-2026-39909, CVSS 4.0 9.2).**
llama.cpp before b8585: storing a computation graph, freeing referenced buffers, then triggering
re-execution leaves dangling pointers → arbitrary read/write, full RCE, no auth, no user
interaction. Fixed in b8585 (commit `389c7d4`, PR #21175). Only exposes hosts running the RPC
backend (rpc-server / `--rpc`), not standalone llama-server.
Source: https://www.vulncheck.com/advisories/llama-cpp-use-after-free-in-rpc-graph-recompute-handler
Cerebro relevance: HIGH — Cerebro plans CUDA+RPC+CPU backends; never bind RPC to LAN, pin ≥ b8585
(local build here is b10751, i.e. past the fix — verify at runtime anyway).

**GGUF parser heap overflows: two patch floors, not one.** CVE-2026-27940 (CVSS 7.8, integer
overflow → undersized heap alloc in `gguf_init_from_file_impl()`, bypass of the 2025 CVE-2025-53630
fix, fixed in b8146) and CVE-2026-33298 (CVSS 7.8, overflow in `ggml_nbytes()`, fixed in b7824).
Ollama separately ships its own Go GGUF loader with CVE-2026-7482 ("Bleeding Llama", CVSS 9.1,
unauthenticated OOB read over `/api/create`, fixed only in Ollama 0.17.1 — no llama.cpp build
fixes it). LM Studio and Jan vendor the same ggml parser. Rule: llama.cpp ≥ b8146 AND Ollama ≥
0.17.1; treat every untrusted GGUF like an untrusted binary; sandbox model loading.
Source: https://www.stingrai.io/blog/malicious-gguf-file-llama-cpp-parser-rce

**JSON-schema-to-grammar recursion DoS (CVE-2026-52130, CVSS 7.5, published 2026-09-01)** affecting
b5693 and earlier — `unverified` (seen only in search-result snippets, advisory page not opened).
Plausible but do not cite as verified.

**Cyera batch (CVE-2026-43627…43632, CVE-2026-70638…70640: batch init overflow, DRY sampler
underflow, KV-cache restore overflow, server UAFs) — `unverified`** (seen only in snippets; public
PR reportedly closed unmerged). Do not cite without opening the source.

## 2. Ollama

**Tensor-blob redirect SSRF (CVE-2026-85180, CVSS 8.7, published 2026-09-03).** Ollama 0.30.0–0.33.2
fails to validate redirect destinations for tensor-layer blob downloads (`x/transfer/download.go`
`resolve()` returns cross-host `Location` unchecked), so a malicious registry redirects pulls to
internal hosts incl. cloud metadata. No patched release at disclosure; fixes proposed in PR #17082
and PR #18021 (both open/unmerged). Separate path from the earlier CVE-2026-5530 `server/download.go` fix.
Source: https://www.vulncheck.com/advisories/ollama-0.30.0-through-0.33.2-ssrf-via-cross-host-tensor-blob-redirect

**Go GGUF decoder integer overflow (CVE-2026-86289, ≤ 0.31.1, fixed in 0.31.2-rc1).**
`readGGUFV1String` in `fs/ggml/gguf.go`: unbounded length arithmetic wraps, corrupting allocation
sizes; crafted GGUF triggers it (user must load the file). Patch adds overflow-safe element-count
math plus `MaxStringLength` (16 MB), `MaxArraySize` (64 MB), `MaxTensorDims` (4).
Source: https://www.sentinelone.com/vulnerability-database/cve-2026-86289/

**DNS-rebinding → persistent chat-template poisoning (NemoClaw + Ollama, CVE-2026-65105, CVSS 8.1,
disclosed 2026-08-25, Oasis Security via NVIDIA PSIRT).** NemoClaw starts Ollama with
`OLLAMA_HOST=0.0.0.0:11434`; non-loopback bind skips Ollama's Host-header check (the CVE-2024-28224
fix), so a single malicious webpage rebinds DNS to 127.0.0.1 and uses unauthenticated
`/api/show` + `/api/create` to rewrite the model's Go chat template — persistent, invisible,
survives system prompts. Fixed on macOS/Linux (v0.0.35, loopback + token proxy); Windows/WSL path
still exposed at disclosure. Cerebro lesson: bind inference loopback-only, add template-hash
integrity check (`ollama show --template` vs baseline).
Source: https://thehackernews.com/2026/08/a-malicious-webpage-could-poison-your.html

## 3. Open WebUI (Sept batch: upgrade to v0.11.1)

v0.11.1 (2026-08-25, `d3e8bf3`) is a security rollup: channel message-overwrite via chat-completion
API (CVE-2026-87994), same-origin XSS→account-takeover via terminal port-preview iframe
(`PortPreview.svelte`, allow-scripts + allow-same-origin; fix commit `54d7a22`), DNS-rebinding SSRF
in `SafePlaywrightURLLoader` (CVE-2026-87996 — Python validates hostname, Playwright re-resolves it;
fix `27402ff`), knowledge-base metadata disclosure across 11 vector backends (CVE-2026-87017, fix
`1d6d4e6`), plus opt-in human-in-the-loop tool approval.
Source: https://github.com/open-webui/open-webui/releases/tag/v0.11.1
Also: Vega/Vega-Lite chart rendering client-side SSRF (CVE-2026-70480, 0.6.34–0.11.0, fixed 0.11.0 —
unrestricted resource loader lets a chat code-block make the *viewer's* browser issue
attacker-chosen GETs).
Source: https://osv.dev/vulnerability/CVE-2026-70480

## 4. ComfyUI — unauthenticated pickle RCE (CVE-2026-68771, CVSS 9.3/9.8)

`LoadTrainingDataset` passes attacker-uploaded `shard_*.pkl` to `torch.load()` without
`weights_only=True`; chain unauthenticated `POST /upload/image` + `POST /prompt`. Fix: one-arg patch
(commit `94ee49b`, PR #14543). Public exploit exists (堡 2026-08-03). Cerebro rule: never
`torch.load` untrusted files without `weights_only=True`; never expose workflow APIs unauthenticated.
Source: https://www.vulncheck.com/advisories/comfyui-unauthenticated-rce-via-loadtrainingdataset-pickle-deserialization

## 5. MCP ecosystem — the dominant trending technique

**Tool poisoning (OWASP, canonical).** Malicious server embeds instructions in tool descriptions /
responses; agent treats them as trusted context → file reads, exfiltration, prompt bypass. Root
cause: descriptions reviewed once at connect time, responses never. Defenses: structured-output
schemas, privileged-tool isolation, server-side enforcement, allowlists, out-of-band confirmation.
Source: https://owasp.org/www-community/attacks/MCP_Tool_Poisoning

**Deadbugz (active Aug–Sep 2026, Pillar Security via CSA 2026-09-02).** Malicious MCP server pushed
via 23 unsolicited GitHub PRs (2026-08-10); benign for first 3 `tools/call`s, then rewrites its own
tool metadata to harvest SSH keys / AWS creds / kubeconfigs / shell history. No CVE. Defeats
one-time install review → requires continuous per-call schema fingerprinting + hash pinning +
allowlisting.
Source: https://labs.cloudsecurityalliance.org/research/csa-research-note-deadbugz-mcp-metadata-poisoning-20260902-c/

**MCP cache + instructions chaining (MCP-2026-008 / MCP-2026-015, both open/unpatched at
disclosure).** July-28 spec added `CacheableResult` with attacker-controlled `cacheScope: public`,
so a shared gateway re-serves one poisoned `tools/list` to every user; the unsanitized
`instructions` field is concatenated into victims' system prompts. Mitigation: never cache across
identities; treat server metadata as untrusted input; pin servers; log every tool list.
Source: https://datapace.ai/blog/mcp-cache-poisoning-prompt-injection

**LM Studio: MCP commands execute before handshake validation (issue #1819, Apr 2026, still open).**
Enabling (or auto-starting on launch) an `mcp.json` STDIO entry spawns the command first; a failed
handshake cannot un-run it, and forked children persist. `mcp.json` writability = persistence.
Wishlist: first-enable consent dialog with resolved path+hash, kill process group, strip env
(`LD_PRELOAD`, `PYTHONSTARTUP`, …), block shell wrappers, per-platform sandboxing.
Source: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1819

## 6. Tauri / WebView2 — origin confusion (CVE-2026-42184, fixed 2.11.1)

`is_local_url()` on Windows/Android maps `app://` to `http://<scheme>.localhost/` but checks only
the first DNS label (`split_once('.')`), so `http://app.evil.com/` is classified `Origin::Local`
and remote pages invoke local-only IPC commands. Windows is Cerebro's primary target → pin
Tauri ≥ 2.11.1, keep capabilities least-privilege, never render remote content in the privileged
WebView.
Source: https://github.com/tauri-apps/tauri/security/advisories/GHSA-7gmj-67g7-phm9

## 7. AI coding agents — Sept 2026 cluster

**GitSpawn (Manifold Security, 2026-09-01/02): 8 flaws, 7 agents.** Malicious `.git/config`
`core.fsmonitor` value executes when the agent runs background `git status` — outside the sandbox,
no approval prompt. Affects Claude Code (2.1.193→fixed 2.1.196; second `ultrareview` path still open
at 2.1.252), Codex CLI 0.102–0.130 (CVE-2026-19592, CVSS 7.3, fixed 0.131.0), Cursor (patched),
Goose <1.44.0 (CVE-2026-72718, fixed 1.44.0), Hermes/Qwen Code/Grok Build (unpatched at disclosure).
Clone-vector only via intact `.git/` file copies (zips, USB, sync folders), not `git clone`.
Defense: `git config --global core.fsmonitor false`, fresh clones only.
Source: https://shattered.io/gitspawn-ai-coding-agent-vulnerability-2026/

**Plugin4Shell (AIR, disclosed 2026-09-18).** Same missing check in 4 products: SHA pin verified for
the *requested* commit but checkout landing never verified → zero-click RCE at plugin load.
Claude Code 2.1.179 and Codex 0.146.0 patched silently (no CVE); Copilot unpatched; Gemini CLI
deprecated instead of patched.
Source: https://easternherald.com/2026/09/20/plugin4shell-ai-coding-agents-supply-chain-vulnerability/

**PhantomFix / Sentry Seer (CERT/CC VU#212479, CVE-2026-90999, CVSS 9.8, 2026-09-16, unpatched,
vendor status Unknown).** Public Sentry DSN accepts attacker-crafted error events; Seer Autofix
embeds the fabricated root-cause analysis into the coding-agent prompt; agent installs attacker's
package with repo access. Predecessor: Tenet "agentjacking" (June 2026, 85% success across Claude
Code/Cursor/Codex, 2,388 exposed DSNs). Mitigations: disable Seer auto-remediation/handoff, deny
unprompted installs, sandbox + egress-deny, proxy the DSN server-side.
Source: https://dev.to/jamilxt/a-fake-bug-report-can-make-your-ai-coding-agent-run-attacker-code-55pg

**Explosive prompts (arXiv 2609.22510, 2026-09-18, research-only).** Conditional dormant payloads
("when X, do Y") beat imperative injections 43–83% vs ≤3% across 9 production agents (Codex, Gemini
CLI, Claude Code, Cursor, Copilot, Devin, Kiro, Qwen Code, Assistant; n=30 each) and evade Prompt
Guard 2 / PIGuard / PIShield; proposed DeFuse detector (AUC 0.9994, 1.16 ms/doc) cuts live
execution 34.3%→3.0%. Batch-1 laptop irrelevance note: numbers are attack-success rates, not
throughput; no perf claim transfers.
Source: https://arxiv.org/abs/2609.22510

## 8. Malicious model files (cross-cutting)

**GGUF Jinja2 SSTI class.** `tokenizer.chat_template` is executable Jinja2; unsandboxed
`jinja2.Environment()` rendering = RCE. Known cases: llama-cpp-python < 0.72 ("Llama Drama",
CVE-2024-34359 — patched); SGLang `/v1/rerank` (CVE-2026-5760, CVSS 9.8, unpatched at 2026-04-21,
maintainers unresponsive; ZMQ pickle RCEs CVE-2026-3059/3060 fixed in 0.5.10). Scan templates for
`__class__`/`os`/`subprocess`/`eval`/`exec`; render only in `ImmutableSandboxedEnvironment`.
Sources: https://research.jfrog.com/model-threats/gguf-ssti/ ;
https://labs.cloudsecurityalliance.org/research/csa-research-note-sglang-cve-2026-5760-llm-serving-rce-20260/

## 9. Supply chain (AI toolchain)

**LiteLLM/Trivy (TeamPCP/Shai-Hulud, Mar 2026; fallout quantified Aug 2026).** Poisoned
`litellm==1.82.7/1.82.8` on PyPI (~40 min window, 10:39 UTC 2026-03-24); 1.82.8's `litellm_init.pth`
runs at every interpreter startup, harvesting SSH/cloud/k8s/LLM keys to `models.litellm.cloud`.
Aug 2026 analyses: 153 GB archive, 433,909 files, 118,829 runner dumps, 2,488 orgs (Hudson Rock);
2,500+ companies / 434k pipelines (CloudSEK); FBI FLASH-20260702-01 warns harvested creds will be
reused. Vendor: last clean 1.82.6, safe ≥ 1.83.0 via CI/CD v2 + cosign-signed images; Docker-proxy
users unaffected. Cerebro lesson (offline harness): pin + hash every dependency, verify signatures.
Source: https://docs.litellm.ai/blog/security-update-march-2026

## 10. Trending techniques & recommended mitigations (summary)

1. Prompt/tool injection through tool metadata (poisoning, Deadbugz runtime-gating, MCP cache →
   cross-user) — pin + continuously fingerprint schemas, allowlist servers, confirm sensitive ops
   out-of-band, key caches by identity.
2. Malicious model files (GGUF parser overflows, pickle `torch.load`, Jinja2 chat-template SSTI) —
   patched builds, `weights_only=True`, sandboxed rendering, provenance + checksums.
3. DNS rebinding vs localhost APIs (Ollama 11434, Open WebUI Playwright loader) — loopback-only
   bind, Host-header validation, auth reverse proxy, firewall the port, template integrity hashes.
4. Supply chain (poisoned PyPI, SHA-pin bypass, malicious `.git/config`, rogue PRs adding MCP
   servers) — hash-pinned deps, verify checkout landing, fresh clones, review MCP config as code.
5. Over-privileged local surfaces (Tauri IPC origin confusion, MCP pre-handshake exec, agent
   background commands) — least-privilege capabilities, consent gates, sandbox/sanitize subprocesses.

