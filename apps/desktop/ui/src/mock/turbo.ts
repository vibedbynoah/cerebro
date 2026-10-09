import type { EngineBuild, GpuRange, MachineProfile, Measurement, TokenProfile } from "../lib/turbo";

/**
 * Illustrative Turbo data shaped like the engine's future API (crates cerebro-builds and
 * cerebro-profiles). Every number shown from here is labelled illustrative on screen.
 */

export const ENGINE_BUILDS: readonly EngineBuild[] = [
  { id: "llama-official", name: "llama.cpp official release", kind: "control", target: "Generic CUDA 12 build", version: "b11026", channel: "stable", installed: true, speedVsControl: 1, qualityOk: true, note: "Always installed. Every other build has to beat it on this machine." },
  { id: "llama-native", name: "llama.cpp, tuned for this machine", kind: "tuned-build", target: "AVX2 + CUDA sm_86, flash attention, LTO", version: "b11026", channel: "stable", installed: true, speedVsControl: 1.11, qualityOk: true, note: "Compiled for this CPU's instruction set and this GPU's architecture." },
  { id: "ik-llama", name: "ik_llama.cpp", kind: "fork", target: "CUDA sm_86, fused MoE kernels", version: "2026.10", channel: "beta", installed: true, speedVsControl: 1.24, qualityOk: true, note: "A fork with faster MoE and i-quant kernels." },
  { id: "strata", name: "Strata", kind: "engine", target: "Qwen3.8-Flash-Next family only", version: "0.1.40", channel: "beta", installed: true, speedVsControl: 1.9, qualityOk: true, note: "Expert cache in VRAM, CPU experts in parallel, model-native speculation. One model family." },
  { id: "llama-pgo", name: "llama.cpp, profile-guided build", kind: "tuned-build", target: "PGO from this machine's own runs", version: "b11026", channel: "experimental", installed: false, speedVsControl: null, qualityOk: null, note: "Built from a profile of your own workloads. Takes about 20 minutes to build locally." },
  { id: "llama-self-fork", name: "llama.cpp, expert-cache fork", kind: "fork", target: "CUDA sm_86", version: "dev", channel: "experimental", installed: true, speedVsControl: 0.31, qualityOk: true, note: "Measured slower than the control here, so it is never promoted." },
];

export const GPU_RANGE: GpuRange = { minPowerW: 60, maxPowerW: 115, defaultPowerW: 80 };

export const PROFILES: readonly MachineProfile[] = [
  { id: "max", name: "Max speed", summary: "Highest sustained speed. Louder and warmer.", cpu: { powerPlan: "best-performance", boost: true, maxStatePct: 100 }, gpu: { powerLimitW: 115, coreLockMHz: null, memoryOffsetMHz: 0, fan: "max" } },
  { id: "balanced", name: "Balanced", summary: "Nearly full speed at much lower heat. The default.", cpu: { powerPlan: "balanced", boost: true, maxStatePct: 100 }, gpu: { powerLimitW: 80, coreLockMHz: null, memoryOffsetMHz: 0, fan: "auto" } },
  { id: "quiet", name: "Quiet", summary: "Fans stay low. Good for long chats at a desk.", cpu: { powerPlan: "balanced", boost: false, maxStatePct: 90 }, gpu: { powerLimitW: 65, coreLockMHz: 1500, memoryOffsetMHz: 0, fan: "quiet" } },
  { id: "battery", name: "Battery", summary: "Longest runtime unplugged.", cpu: { powerPlan: "best-efficiency", boost: false, maxStatePct: 70 }, gpu: { powerLimitW: 60, coreLockMHz: 1200, memoryOffsetMHz: 0, fan: "auto" } },
  { id: "custom", name: "Custom", summary: "Your own limits. Every change is measured and can be reverted.", cpu: { powerPlan: "balanced", boost: true, maxStatePct: 100 }, gpu: { powerLimitW: 80, coreLockMHz: null, memoryOffsetMHz: 0, fan: "auto" } },
];

/** What re-measuring after each profile shows on this machine (illustrative). */
export const PROFILE_MEASUREMENTS: Readonly<Record<string, Measurement>> = {
  max: { tokPerSec: 18.6, gpuTempC: 83, throttled: false },
  balanced: { tokPerSec: 17.9, gpuTempC: 71, throttled: false },
  quiet: { tokPerSec: 15.2, gpuTempC: 64, throttled: false },
  battery: { tokPerSec: 11.4, gpuTempC: 58, throttled: false },
  custom: { tokPerSec: 17.9, gpuTempC: 71, throttled: false },
};

export const TOKEN_PROFILE: TokenProfile = { gpuKernels: 0.34, cpuExperts: 0.48, pcie: 0.12, other: 0.06, expertCacheHit: 0.07, draftAcceptance: 0.63, vramFreeGiB: 0.4 };

export const LOGICAL_CORES = 16;
export const MEASURED_SWEET_SPOT = 4;
