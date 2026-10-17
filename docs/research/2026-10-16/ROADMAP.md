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
- [ ] Qwen3.8 hybrid (Gated DeltaNet) and Gemma 4 graphs (A#22) — product's listed models

## App
- [x] Turbo: real engine switches, beginner/advanced text, channel badges, saved per machine
- [x] MCP client (stdio, mcp.json, tools/list, tools/call) + Tools screen section (status, logs, presets, try a tool)
- [ ] Models call MCP tools from Chat (tool-call loop behind the Tools rules); Cerebro itself as an MCP server
- [x] Monitor: real GPU load/temp/power/PCIe/throttle, CPU load, driver (NVML at run time), 60 s history
- [ ] Monitor: live prefill tok/s, per-tier memory %, disk read
- [ ] Orchestrate mode (GVS5H, arXiv 2608.26480, github.com/slee-persis/GVS5H, MIT): a manager instance
      splits a task, fresh worker instances of the same local model solve parts, all coordinate through a
      ledger folder (plan, notes, current solution) in the user's workspace. Reported: local Qwen3.8-27B
      66.8% -> 92.4% pass@1 on 100 hard LiveCodeBench problems; some models gain nothing, so the app
      measures it per model before recommending it. Depends on continuous batching and the conversation cache.
- [ ] Remove Security screen (owner 2026-10-09); protections stay in Settings

## Drafting policy (credits)
Boilerplate drafted by free opencode models (`opencode run -m opencode/<model>-free`), reviewed and fixed
here. Local Flash-Next on Strata reads prompts at 3.5 tok/s on this laptop when RAM is full, so it only
takes short prompts.
