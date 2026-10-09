# Turbo: the CachyOS layer (owner, 2026-10-08)

Settings presets exist in other apps. Cerebro goes further, the way CachyOS goes past a stock
distro: optimised builds, better scheduling and machine profiles, all measured on the user's machine,
all optional, all reversible. Every item below is a feature with a channel (Stable / Beta /
Experimental), a hardware gate and a measured before/after.

## 1. Optimised builds ("custom kernels")
- Per-machine llama.cpp/ggml builds: CPU ISA tiers (x86-64-v3 AVX2, v4 AVX-512, AVX-VNNI, AMX;
  ARM NEON/SVE), CUDA arch-specific (sm_75..sm_120), HIP gfx targets, Vulkan, SYCL, Metal; flash
  attention and fused kernels on, LTO/PGO where it measures faster.
- Alternative engines and forks as selectable "kernels": official llama.cpp, ik_llama.cpp, Strata
  (MoE turbo), vLLM/SGLang/TensorRT-LLM where hardware allows, MLX on Apple, ExLlamaV3.
- The official release always stays installed as the control: a custom build ships only if it beats
  the official one on this machine (measured: a self-built fork was once 3-5x slower).
- Prebuilt, signed variants downloaded by hardware tier, or built locally on request.

## 2. Schedulers
- CPU: thread count chosen by measurement (fewer threads often match speed at far lower heat),
  P-core/E-core pinning, process priority class, NUMA placement, timer resolution, background
  efficiency mode for idle.
- GPU: offload and expert placement, tensor split across cards, KV placement and streaming.
- Requests: queue policy (FIFO, shortest-first, interactive-first), batching/parallel slots,
  conversation-state parking, prefill chunk size.

## 3. Machine profiles (CPU and GPU)
- Profiles: Max speed / Balanced / Quiet / Battery / Custom, switchable live, per model.
- CPU: Windows power plan and processor boost mode, min/max state; Linux governor and EPP.
- GPU: power limit and clock locks through NVML (within the vendor's allowed range), memory clock,
  fan curve where the driver exposes it; AMD/Intel equivalents. Measured, never assumed: on a
  power-limited laptop GPU, clock offsets can lower speed, so each change is benchmarked and auto-
  reverted if it does not help or if temperatures/power hit limits.
- Every change shows what it touched and has one-click revert; Cerebro restores the previous state
  on exit or crash.

## 4. Profiler
- Per-token timeline: prefill vs decode, GPU kernels vs CPU experts vs PCIe transfers, cache hits,
  draft acceptance; bottleneck verdict ("CPU-bound on experts: +2 GB VRAM would give ~+700 experts").
- Comparison runs: profile A vs B on the same prompt set, with the method saved.

## Safety stays optional
Sandbox, integrity checks, loopback-only, permission prompts and every other protection are enabled
by default and each can be switched off by the user (Settings > Advanced protections), with a plain
explanation of what changes. Cerebro never argues about it.
