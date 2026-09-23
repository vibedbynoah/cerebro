PRODUCT: Cerebro Engine ("Cerebro"). Open-source (Apache-2.0), Windows-first local LLM harness for small businesses and AI enthusiasts. Fully offline; only an opt-in update check touches the network. Owner's goals: maximum tokens/sec, maximum real coding and reasoning ability out of local models, extreme security, professional-grade visual design and UX, and beating the 10 best competing harnesses on every metric in blind side-by-side comparison.

EXISTING CODE: Rust workspace. A closed allowlist of llama.cpp flags with safe argv emission, and a hardware probe (CPU topology, GPU inventory, backend identity + SHA-256). Everything else (process supervisor, benchmark harness, speculation tuner, autotuner, quality gate, config, local API, CLI, GUI, chat, coding agent, RAG, website, packaging) is unbuilt.

BACKEND: llama.cpp is the compute backend, spawned as a child process with argv only (no shell). Local build b10751 with CUDA + RPC + CPU backends (no Vulkan). Latest upstream release on 2026-09-22 is about b11028; the official b11026 CUDA build is installed at C:\bench\llamacpp-official.

REFERENCE MACHINE: laptop, Windows 11 Pro. Intel i9-12900H (6 P-cores / 12 threads + 8 E-cores, no AVX-512), 31.7 GB RAM, RTX 3070 Ti Laptop 8 GB VRAM (Ampere, compute 8.6, power-limited 80-115 W), Intel Iris Xe iGPU, NO NPU, NVMe PCIe 4.

MEASURED ON THIS MACHINE (trust over generic claims):
- Flag selection alone moved one model from 2.22 tok/s (LM Studio defaults) to 29 tok/s (-ngl 999, -ncmoe, q8_0 KV, P-core pinning).
- Qwen3-Coder-30B-A3B MoE with -ncmoe 32: 78% HumanEval at 29 tok/s; a 2-bit dense 27B: 37% at 12 tok/s.
- Short benchmarks overstate MoE throughput ~3x vs a sustained soak.
- --moe-expert-cache-size was slower than no cache at every size.
- 1-bit/2-bit quants and depth pruning destroyed models outright.
- Repetition collapse came from greedy decoding; min_p 0.05 fixed it.
- A self-built llama.cpp fork ran 3-5x slower than the official release binary.
- Dense Qwen3.8-27B Q4_K_M: 7.0/8 on a hard coding benchmark at ~2 tok/s; Gemma-4-E4B Q4_K_M: 6.25/8 at 61 tok/s; a 125B MoE scored 3/8 at 3.4 tok/s.

ALREADY RULED OUT (only revisit with NEW primary evidence, and say so): NPU offload (no NPU); retrofitting Mixture-of-Depths; inference-time KV-to-linear-attention conversion; k-means KV collapse; "bit-exact" speculative decoding (it is distribution-preserving); MXFP4/NVFP4 acceleration on Ampere; "1M token context" on this hardware.

CLAIMS POLICY: every claim must be checkable by a hostile reviewer; any speedup figure must state hardware, model, quant, runtime version, context depth and sampler.

DATE: 2026-09-22. Focus on what changed since 2026-09-01. Prefer 2025-2026 primary sources.
