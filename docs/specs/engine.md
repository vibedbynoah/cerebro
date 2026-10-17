# Cerebro Engine: our own inference backend (owner, 2026-10-09)

Cerebro runs models on its own engine, written in Rust with its own CPU and CUDA kernels. It does not
wrap or ship llama.cpp, Strata or any other inference server. Other engines appear only as named,
measured baselines on the comparison page (docs/CLAIMS.md). Every optimisation below is measured on
the user's machine and gated by channel (Stable / Beta / Experimental) per docs/specs/turbo-system.md.

## Architecture

- `cerebro-engine` crate, no inference dependencies: own GGUF reader (memory-mapped, every tensor
  bounds-checked), own tokenizer (BPE from GGUF metadata, chat templates), own model graphs.
- Backends behind one `Device` trait: CPU (AVX2/AVX-512/NEON kernels, rayon threads pinned to
  performance cores) and CUDA (kernels written in CUDA C, compiled per architecture, launched through
  the driver API). Vulkan/ROCm/Metal later through the same trait.
- Weights stay quantised in memory; matmuls dequantise in registers (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS,
  then ternary and FP8/NVFP4 on hardware that has them).

## Signature technologies

1. **Conversation cache (prefix cache).** A radix tree of KV blocks keyed by token prefixes. A new
   request reuses the longest cached prefix and computes only new tokens; parked conversations move
   to RAM and back. Message-boundary checkpoints so an agent rewriting its last message keeps the
   history. Paged KV blocks (fixed-size pages) so caches share memory without copying.
2. **MoE expert cache.** All experts in RAM (or memory-mapped from SSD); the most-used experts live
   in VRAM in an adaptive LRU/frequency cache. Experts that miss are computed on the CPU in parallel
   with the GPU's work. The hit rate, active experts per token and residency are exported to the
   Monitor.
3. **Speculative decoding.** Model-native MTP heads where present; prompt-lookup drafts for edits and
   quoted text; a small draft model otherwise. Each enabled only while measured acceptance pays.
4. **Chunked prefill with expert prefetch.** Long prompts in large chunks; the next layer's experts
   stream to the GPU during the current layer's attention.
5. **KV quantisation and long context.** q8/q4 KV pages; past a threshold, cold KV pages live in RAM
   and are prefetched.
6. **Continuous batching.** Several conversations decode together in one pass; interactive requests
   first.
7. **Telemetry built in.** Every step reports timings, cache hits, acceptance and memory to the
   Monitor through the same snapshot the app already renders.

## Milestones (each ends with a measured run on this PC)

- **M1 CPU core:** GGUF reader, tokenizer, Qwen3-family dense graph, Q8_0/Q4_K/Q6_K kernels,
  sampling (temperature, top-k/p, min-p). Output matches a reference within tolerance on a small model.
- **M2 Conversation cache:** paged KV, radix prefix cache, streaming OpenAI-compatible server route.
- **M3 MoE + expert cache:** Qwen3 MoE graph (local test model: Qwen3-Coder-30B-A3B Q4_K_M), expert
  cache with CPU fallback, hit-rate telemetry.
- **M4 CUDA:** dense and MoE kernels on sm_86 (RTX 3070 Ti laptop), attention with paged KV.
- **M5 Speculation, KV quant, chunked prefill, batching.**
- **M6 Current architectures:** Qwen3.8 (hybrid linear attention + MoE + MTP), Gemma 4 (dense and
  MoE, E-series), image and speech models.

## Rules

- Correctness first: every kernel has a reference test against plain f32 math; every graph has a
  logits test against stored reference values.
- Never slower than the previous milestone on any tracked benchmark.
- Hostile model files cannot crash or overrun: every offset and size in a GGUF is validated.

## Placement and hardware test matrix (owner, 2026-10-09)

Every model is benchmarked across placements and machines, the way users will run it:
VRAM only; RAM only (CPU); SSD streaming (memory-mapped, not preloaded); VRAM + RAM split (by layer
and by expert); VRAM + RAM + SSD; plus thread counts, KV types and context depths. Test machines:
i9-12900H + RTX 3070 Ti laptop 8 GB (main), Ryzen 7 7840HS + Radeon 780M, 32 GB DDR5-5600 (second;
stands in for iGPU and AMD users, not a distributed worker). The tuner picks the fastest stable
placement per machine; results feed the site only when measured.

## Product rules (owner, 2026-10-09)
- Everything lives inside the desktop app: no CLI for users (internal dev harnesses only).
- Signature quantisation: Unsloth Dynamic (UD) IQ quants, supported natively; a Cerebro quant
  format later if measurably better.
- Separate workspaces for coding, images, audio and video chats; a chosen folder the model can
  see and work in; MCP servers configured in an mcp.json the app manages.
