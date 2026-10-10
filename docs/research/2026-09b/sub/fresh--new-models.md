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
