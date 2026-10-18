# Triage: "Cross-Platform Local LLM Inference: Feature Roadmap" (owner-supplied, 2026-10-17)

The source document lists ~50 features in 10 groups. It says citations are provided but contains
none, so its specific figures (e.g. "33 -> 10 fused kernels, ~1.3x", "llama.cpp implements nvfp4x4")
are unverified and are not used as targets here. This file maps each group to Cerebro's actual
state on 2026-10-17 and ranks the gaps for this product: a Windows desktop app, own Rust engine,
fastest local inference on whatever CPU, GPU and drive the user has.

Status: **done** (shipped and measured), **partial**, **gap**, **n/a** (does not fit a desktop app).

## 1. Core
| Item | Status | Notes |
|---|---|---|
| Efficient transformer inference | done | Own CUDA kernels (int8 dp4a + tensor-core MMA prefill, GQA decode), CPU AVX-VNNI Q8 dot, batched CPU prefill |
| KV cache, prefix reuse | partial | Conversation cache reuses the longest common prefix of one session; no multi-prefix block cache |
| Multi-precision | done | f16 KV on GPU, f32 on CPU; K-quants, IQ/UD quants, Q8_0, F16 |
| Multi-backend | **gap** | CUDA + CPU only. AMD and Intel GPUs run CPU-only: the largest gap for "every mainstream GPU" |
| Threaded queuing | done | Engine thread, separate HTTP/IPC threads, Job Object tools |

## 2. Scheduling and cache
| Item | Status | Notes |
|---|---|---|
| Block-hash prefix cache | gap | P1: system prompts and tool lists repeat across chats and API calls |
| In-flight (continuous) batching | gap | P1: needed for Orchestrate workers and concurrent local-API clients |
| Multi-step decode in one launch | done | Whole decode step captured as a CUDA graph; greedy picks on the GPU |
| LRU eviction across tiers | partial | MoE expert cache rebalances by use; no KV tiering |
| Persistent KV store (disk) | gap | P1: resume a long conversation without re-reading the prompt |
| Cache salting | n/a | One user per machine; the local API is keyed and loopback by default |

## 3. Quantization and kernels
| Item | Status | Notes |
|---|---|---|
| INT8 / K-quants / 2-bit IQ | done | Native K-quant GPU layout when VRAM is short (placement.rs) |
| FP8 / NVFP4 | n/a here | Needs Ada / Blackwell tensor cores; reference GPU is Ampere. Revisit with hardware |
| Kernel fusion | partial | rmsnorm+quant, q/k norm+RoPE+cache write, SiLU+quant, DeltaNet core fused per stage |
| Flash attention | partial | MMA flash prefill for head size 128 only; **head size 256 (Qwen3.8) uses the generic kernel**: P0 |
| Long-context attention variants | gap | P2 |
| MoE kernels | done | Hybrid expert placement, VRAM expert cache, streamed experts during prefill |

## 4. Speculative decoding
| Item | Status | Notes |
|---|---|---|
| Draft-model / MTP speculation | gap | P0 on the engine roadmap: verify with batched prefill (Qwen3.8 ships MTP heads) |
| Prompt-lookup (n-gram) drafts | gap | P0: free speedup for code and quoting, no second model |
| Online draft training | n/a for now | Research-grade |

## 5. Streaming and oversubscription
| Item | Status | Notes |
|---|---|---|
| GPU/CPU layer split | **done today** | Planner picks int8 / native / split; Qwen3-14B 3.2 -> 11.8 tok/s, Qwen3.8-27B 2.0 -> 2.8 tok/s on 8 GB |
| Drive-aware placement | **done today** | NVMe / SATA / HDD / USB-SD detected; per-token drive bytes and warnings |
| Per-layer PCIe weight streaming (dense) | measured-not-worth-it here | Laptop PCIe 4 x8 ~ 12 GB/s: streaming the 27B's 9 GB tail costs ~0.75 s/token vs ~0.33 s on the CPU. Revisit for desktop x16 + small CPUs |
| Activation/KV swapping | gap | P2, tied to KV tiering |

## 6. Distributed
| Item | Status | Notes |
|---|---|---|
| Tensor / pipeline parallel across machines | gap | P2. The owner's BOSGAME mini PC (LAN RPC worker) is the real target; pipeline split by layer reuses the split machinery |
| NCCL / RDMA | n/a | Consumer Windows hardware |

## 7. Security
| Item | Status | Notes |
|---|---|---|
| Isolation | done | Job Objects, tool access ladder, secret-free tool env, read-before-write, SSRF guard |
| Audit log | partial | Tools activity log (session); persistent append-only log is a gap (P2) |
| Access control | done | Local API / MCP keys, loopback default, phone pairing + password |
| Encrypted weights | n/a | |

## 8. Testing and CI
| Item | Status | Notes |
|---|---|---|
| Cross-backend equality | done | Greedy CPU = GPU = split checks; examples/layerdiff per-layer residuals |
| Benchmarks | done | bench harness -> docs/measurements, official llama.cpp release as control |
| CI | gap | No GPU runner; local suites only. P2 |

## 9. Packaging
PyPI / conda / Docker: n/a (desktop app). Windows installer: Tauri bundler. macOS: gap until a Metal
backend exists.

## 10. Experimental
2-bit: done (UD IQ2). NVFP4x4, sparsity, NPU: gaps, P2, hardware-dependent.

## Build order taken from this triage
1. Flash (MMA) prefill attention for head size 256: Qwen3.8 prompt reading.
2. Speculation: prompt-lookup drafts, then MTP heads, verified by batched prefill.
3. Vulkan backend (AMD / Intel GPUs) behind the same placement planner.
4. Continuous batching + block prefix cache (Orchestrate workers, local API clients).
5. Persistent KV store on disk (instant resume of long chats).
