import type { Channel } from "./experience";

/**
 * Turbo: the performance layer (docs/specs/turbo-system.md). Engine builds, schedulers, machine
 * profiles and the profiler. Everything here is pure so the rules are unit-tested: which build may
 * be promoted over the control, what a profile changes, when a change is kept or reverted, and the
 * plain-words bottleneck verdict.
 */

export type EngineKind = "control" | "tuned-build" | "fork" | "engine";

export interface EngineBuild {
  id: string;
  name: string;
  kind: EngineKind;
  /** What this build is tuned for, e.g. "AVX-512 + CUDA sm_86". */
  target: string;
  version: string;
  channel: Channel;
  installed: boolean;
  /** Measured decode speed relative to the control on this machine (1 = same); null if not measured. */
  speedVsControl: number | null;
  /** Did it pass the quality gate (same outputs within tolerance) on this machine? */
  qualityOk: boolean | null;
  note: string;
}

/** The official release is always kept as the control and can never be removed or demoted. */
export function isControl(b: EngineBuild): boolean {
  return b.kind === "control";
}

/** A build may become the default only when it was measured faster than the control by a margin
 *  larger than run-to-run noise, passed the quality gate, and is allowed on the chosen channel. */
export const PROMOTE_MARGIN = 1.03;

export function channelAllows(selected: Channel, required: Channel): boolean {
  const rank: Record<Channel, number> = { stable: 0, beta: 1, experimental: 2 };
  return rank[required] <= rank[selected];
}

export type PromoteVerdict = { ok: true } | { ok: false; reason: string };

export function canPromote(b: EngineBuild, channel: Channel): PromoteVerdict {
  if (isControl(b)) return { ok: true };
  if (!channelAllows(channel, b.channel)) return { ok: false, reason: `Needs the ${b.channel} channel.` };
  if (!b.installed) return { ok: false, reason: "Not installed yet." };
  if (b.speedVsControl === null || b.qualityOk === null) return { ok: false, reason: "Not measured on this machine yet." };
  if (!b.qualityOk) return { ok: false, reason: "Failed the quality gate here, so the control stays." };
  if (b.speedVsControl < PROMOTE_MARGIN) return { ok: false, reason: "Not faster than the control here, so the control stays." };
  return { ok: true };
}

export function formatRatio(r: number | null): string {
  if (r === null) return "not measured";
  const pct = Math.round((r - 1) * 100);
  return pct === 0 ? "same as control" : `${pct > 0 ? "+" : ""}${pct}% vs control`;
}

// ---- Schedulers ---------------------------------------------------------------------------------

export type CoreSet = "performance" | "all" | "efficiency";
export type Priority = "normal" | "above-normal" | "high";
export type QueuePolicy = "fifo" | "interactive-first" | "shortest-first";

export interface Schedule {
  /** null = the measured sweet spot. */
  threads: number | null;
  cores: CoreSet;
  priority: Priority;
  queue: QueuePolicy;
  parallelSlots: number;
  /** Prompt tokens read per step; null = chosen automatically. */
  prefillChunk: number | null;
  parkConversations: boolean;
}

export const DEFAULT_SCHEDULE: Schedule = {
  threads: null,
  cores: "performance",
  priority: "above-normal",
  queue: "interactive-first",
  parallelSlots: 1,
  prefillChunk: null,
  parkConversations: true,
};

export function clampSchedule(s: Schedule, logicalCores: number): Schedule {
  const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(v)));
  return {
    ...s,
    threads: s.threads === null ? null : clamp(s.threads, 1, Math.max(1, logicalCores)),
    parallelSlots: clamp(s.parallelSlots, 1, 8),
    prefillChunk: s.prefillChunk === null ? null : clamp(s.prefillChunk, 256, 32768),
  };
}

// ---- Machine profiles ---------------------------------------------------------------------------

export type ProfileId = "max" | "balanced" | "quiet" | "battery" | "custom";

export interface MachineProfile {
  id: ProfileId;
  name: string;
  summary: string;
  cpu: { powerPlan: "best-performance" | "balanced" | "best-efficiency"; boost: boolean; maxStatePct: number };
  /** GPU limits are always inside the range the driver reports. */
  gpu: { powerLimitW: number; coreLockMHz: number | null; memoryOffsetMHz: number; fan: "auto" | "quiet" | "max" };
}

export interface GpuRange {
  minPowerW: number;
  maxPowerW: number;
  defaultPowerW: number;
}

export interface ProfileChange {
  what: string;
  from: string;
  to: string;
}

const fmtPlan = { "best-performance": "Best performance", balanced: "Balanced", "best-efficiency": "Best efficiency" } as const;

/** Every setting a profile switch touches, as plain from/to lines (shown before and after applying). */
export function diffProfiles(a: MachineProfile, b: MachineProfile): ProfileChange[] {
  const out: ProfileChange[] = [];
  const push = (what: string, from: string, to: string) => {
    if (from !== to) out.push({ what, from, to });
  };
  push("Windows power plan", fmtPlan[a.cpu.powerPlan], fmtPlan[b.cpu.powerPlan]);
  push("CPU boost", a.cpu.boost ? "On" : "Off", b.cpu.boost ? "On" : "Off");
  push("CPU maximum state", `${a.cpu.maxStatePct}%`, `${b.cpu.maxStatePct}%`);
  push("GPU power limit", `${a.gpu.powerLimitW} W`, `${b.gpu.powerLimitW} W`);
  push("GPU clock lock", a.gpu.coreLockMHz ? `${a.gpu.coreLockMHz} MHz` : "Driver default", b.gpu.coreLockMHz ? `${b.gpu.coreLockMHz} MHz` : "Driver default");
  push("GPU memory clock", offset(a.gpu.memoryOffsetMHz), offset(b.gpu.memoryOffsetMHz));
  push("Fan curve", cap(a.gpu.fan), cap(b.gpu.fan));
  return out;
}

const offset = (n: number) => (n === 0 ? "Default" : `${n > 0 ? "+" : ""}${n} MHz`);
const cap = (s: string) => s[0]!.toUpperCase() + s.slice(1);

/** A profile's GPU power limit is clamped to the driver's range; nothing outside it is ever sent. */
export function clampProfile(p: MachineProfile, range: GpuRange): MachineProfile {
  const w = Math.min(range.maxPowerW, Math.max(range.minPowerW, Math.round(p.gpu.powerLimitW)));
  return { ...p, gpu: { ...p.gpu, powerLimitW: w } };
}

export interface Measurement {
  tokPerSec: number;
  gpuTempC: number;
  /** True when the GPU hit its power or thermal limit during the run. */
  throttled: boolean;
}

export type KeepVerdict = { keep: true; gainPct: number } | { keep: false; reason: string };

/** After a change, Cerebro measures again. It keeps the change only if speed rose beyond noise and
 *  the run stayed under the temperature ceiling without throttling; otherwise it reverts at once.
 *  (On power-limited laptop GPUs clock offsets often lower speed, so nothing is assumed.) */
export function keepOrRevert(before: Measurement, after: Measurement, tempCeilingC = 87, noisePct = 2): KeepVerdict {
  if (after.gpuTempC >= tempCeilingC) return { keep: false, reason: `GPU reached ${after.gpuTempC} °C, over the ${tempCeilingC} °C ceiling. Reverted.` };
  if (after.throttled && !before.throttled) return { keep: false, reason: "The GPU started throttling. Reverted." };
  const gainPct = ((after.tokPerSec - before.tokPerSec) / before.tokPerSec) * 100;
  if (gainPct <= noisePct) return { keep: false, reason: gainPct < 0 ? `Slower by ${Math.abs(gainPct).toFixed(1)}%. Reverted.` : `No gain beyond noise (${gainPct.toFixed(1)}%). Reverted.` };
  return { keep: true, gainPct: Math.round(gainPct * 10) / 10 };
}

// ---- Profiler -----------------------------------------------------------------------------------

export interface TokenProfile {
  /** Share of one decode step's time, 0-1 each, summing to ~1. */
  gpuKernels: number;
  cpuExperts: number;
  pcie: number;
  other: number;
  expertCacheHit: number;
  draftAcceptance: number | null;
  vramFreeGiB: number;
}

/** One plain sentence naming the bottleneck and the most useful next step. */
export function bottleneck(p: TokenProfile): string {
  const top = Math.max(p.gpuKernels, p.cpuExperts, p.pcie);
  if (top === p.cpuExperts && p.cpuExperts >= 0.4) {
    return `CPU-bound on experts: ${Math.round(p.cpuExperts * 100)}% of each step runs experts on the processor and only ${Math.round(p.expertCacheHit * 100)}% hit the graphics card's cache. More VRAM for the expert cache, or fewer offloaded layers, is the biggest lever.`;
  }
  if (top === p.pcie && p.pcie >= 0.25) {
    return `Transfer-bound: ${Math.round(p.pcie * 100)}% of each step waits on PCIe. A larger expert cache or a bigger prefill chunk cuts the copies.`;
  }
  if (p.draftAcceptance !== null && p.draftAcceptance < 0.45) {
    return `GPU-bound, and draft guesses are accepted only ${Math.round(p.draftAcceptance * 100)}% of the time: turning speculation off may be faster for this kind of text.`;
  }
  return `GPU-bound: ${Math.round(p.gpuKernels * 100)}% of each step is the graphics card computing. This is the efficient state; a faster build or a lower quantisation is what is left.`;
}
