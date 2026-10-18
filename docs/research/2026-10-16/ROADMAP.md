# Build order from the 2026-10-16 research (A engine, B Strata, C product)

Status: [x] done and measured, [ ] next. Numbers only from docs/measurements.

## Engine (paper A top-25, B section 3)
- [x] int8 tensor-core prefill matmul (A#1) — 0.6B 512-token prefill 1,309 -> 15,853 tok/s
- [x] f16 tensor-core flash attention for prefill (A#21) — 23,215 @512, 18,231 @2048 (llama.cpp 12,077 / 15,896)
- [x] AVX-VNNI Q8_0 dot (A#6) — decode unchanged (memory-bound); pays off in batched CPU prefill
- [x] Engine switches registry (every optimisation a user toggle with explanations; C section 2)
- [x] Batched CPU prefill (dense, 4-token VNNI kernel) — 0.6B CPU prefill 57 -> 183 tok/s @512, bit-identical logits to 256 tokens
- [ ] Async expert swaps + shipped expert profile per model (A#4, A#8, B S-01) — first-turn MoE 19-23 -> 30+
- [ ] MTP / prompt-lookup speculation reusing batched prefill as verifier (A#10, B C-01) — Strata measures 1.6-1.8x
- [ ] Interleaved K-quant CPU layout, large pages (A#5, A#12) — CPU expert 26-28 -> ~37 GB/s in-step
- [ ] q8 KV pages; KV tier to RAM past 64K (A#17, B E-01/E-02)
- [ ] Continuous batching + multi-slot KV (A#15) — needed by Orchestrate below
- [x] Qwen3.8 hybrid (Gated DeltaNet) on CPU and GPU — 4B 5.9 -> 65 tok/s; 27B split 2.0 -> 2.8 tok/s
- [ ] Gemma 4 graphs (A#22)
- [x] Placement planner + GPU/CPU layer split, drive-aware (NVMe/SATA/HDD/USB-SD) — Qwen3-14B 3.2 -> 11.8 tok/s on 8 GB
- [ ] MMA flash prefill for head size 256 (Qwen3.8) — from docs/research/2026-10-17/J-feature-roadmap-triage.md
- [ ] Vulkan backend for AMD / Intel GPUs behind the same planner (J triage #3)
- [ ] Block-hash prefix cache + persistent KV store on disk (J triage #4-5)

## App
- [x] Turbo: real engine switches, beginner/advanced text, channel badges, saved per machine
- [x] MCP client (stdio, mcp.json, tools/list, tools/call) + Tools screen section (status, logs, presets, try a tool)
- [x] Models call MCP tools from Chat (tool-call loop behind the Tools rules) - real Qwen3-0.6B run passes
- [x] Cerebro itself as an MCP server (Streamable HTTP, loopback + token)
- [x] Orchestrate mode backend (GVS5H ledger; real 0.6B run correct in 10.7 s)
- [x] Monitor: real GPU load/temp/power/PCIe/throttle, CPU load, driver (NVML at run time), 60 s history
- [ ] Monitor: live prefill tok/s, per-tier memory %, disk read
- [ ] Orchestrate mode (GVS5H, arXiv 2608.26480, github.com/slee-persis/GVS5H, MIT): a manager instance
      splits a task, fresh worker instances of the same local model solve parts, all coordinate through a
      ledger folder (plan, notes, current solution) in the user's workspace. Reported: local Qwen3.8-27B
      66.8% -> 92.4% pass@1 on 100 hard LiveCodeBench problems; some models gain nothing, so the app
      measures it per model before recommending it. Depends on continuous batching and the conversation cache.
- [x] Remove Security screen; protections stay in Settings
- [x] Phone access: LAN-only HTTPS, QR + password, revocable phones
- [x] Real data only on desktop; 3 Hz live polling (rates per second)
- [x] GPU decode attention (GQA): 331/297/250 tok/s at 32/2048/4096 ctx, ahead of llama.cpp
- [ ] CPU decode: 66.9 vs llama.cpp 72.5 (t=16) on 0.6B - next engine item
- [x] Duplicate Flash-Next copy removed after SHA-256 match (93.7 GB freed)

## Drafting policy (credits)
Boilerplate drafted by free opencode models (`opencode run -m opencode/<model>-free`), reviewed and fixed
here. Local Flash-Next on Strata reads prompts at 3.5 tok/s on this laptop when RAM is full, so it only
takes short prompts.

## DeepSeek queue (2026-10-17; prompts in docs/prompts/deepseek-v4.1.md, one worktree each)
- [x] mcp-http (Streamable HTTP client)   - [x] builtin-tools (+ SSRF guard, toggle)   - [x] orchestrate-ui
- [x] mcp-catalogue (from paper H)        - [x] memory-estimate (paper G s.3-4)
- [x] training v1: Unsloth sidecar, Train screen. Written by Claude after DeepSeek hit its rate limit. Verified end to end:
      5-step Qwen3-0.6B -> Q4_K_M in 87 s, runs on Cerebro Engine. Pinned uv 0.12.22 + llama.cpp b11361 tools.
- [ ] next: permission presets / sandbox ladder, todo/ask-user/plan tools, OpenAI/Anthropic-compatible local API,
      mmproj multimodal, GPU kernels for qwen35 hybrid layers, rows4 benchmark on a quiet machine.
- [ ] permissions: allow/ask/deny broker fail-closed, preset ladder, sandbox (paper I #1-2, #4)
- [ ] tool hygiene: output bounds + spill files, read-before-write (paper I #5-6)
- [ ] agent tools: todo, ask-user, plan mode (paper I #12, #14, #16)
- [ ] local API: OpenAI /v1/chat/completions + Anthropic /v1/messages on one loopback port (paper G #18)
- [ ] multimodal: mmproj loader, image input in Chat (paper G #14) - engine work, Claude
- [ ] CPU decode: rows4 kernel written + bit-identical, benchmark on a quiet machine - Claude
