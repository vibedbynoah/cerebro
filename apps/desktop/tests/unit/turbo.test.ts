import { describe, expect, it } from "vitest";
import {
  bottleneck,
  canPromote,
  channelAllows,
  clampProfile,
  clampSchedule,
  DEFAULT_SCHEDULE,
  diffProfiles,
  formatRatio,
  keepOrRevert,
  type EngineBuild,
} from "../../ui/src/lib/turbo";
import { ENGINE_BUILDS, GPU_RANGE, PROFILES } from "../../ui/src/mock/turbo";

const build = (over: Partial<EngineBuild>): EngineBuild => ({
  id: "x", name: "X", kind: "tuned-build", target: "t", version: "v", channel: "stable", installed: true, speedVsControl: 1.2, qualityOk: true, note: "", ...over,
});

describe("engine promotion", () => {
  it("always allows the control", () => {
    expect(canPromote(ENGINE_BUILDS.find((b) => b.kind === "control")!, "stable")).toEqual({ ok: true });
  });
  it("requires the build's channel", () => {
    expect(canPromote(build({ channel: "experimental" }), "beta")).toEqual({ ok: false, reason: "Needs the experimental channel." });
    expect(canPromote(build({ channel: "beta" }), "experimental").ok).toBe(true);
  });
  it("never promotes a slower, unmeasured, failed or uninstalled build", () => {
    expect(canPromote(build({ speedVsControl: 0.31 }), "stable").ok).toBe(false);
    expect(canPromote(build({ speedVsControl: 1.02 }), "stable").ok).toBe(false);
    expect(canPromote(build({ speedVsControl: null }), "stable").ok).toBe(false);
    expect(canPromote(build({ qualityOk: false }), "stable").ok).toBe(false);
    expect(canPromote(build({ installed: false }), "stable").ok).toBe(false);
  });
  it("orders channels", () => {
    expect(channelAllows("stable", "beta")).toBe(false);
    expect(channelAllows("experimental", "stable")).toBe(true);
  });
  it("formats the ratio", () => {
    expect(formatRatio(1.24)).toBe("+24% vs control");
    expect(formatRatio(0.31)).toBe("-69% vs control");
    expect(formatRatio(1)).toBe("same as control");
    expect(formatRatio(null)).toBe("not measured");
  });
});

describe("schedules and profiles", () => {
  it("clamps every scheduler value", () => {
    const s = clampSchedule({ ...DEFAULT_SCHEDULE, threads: 99, parallelSlots: 0, prefillChunk: 10 }, 16);
    expect(s.threads).toBe(16);
    expect(s.parallelSlots).toBe(1);
    expect(s.prefillChunk).toBe(256);
    expect(clampSchedule(DEFAULT_SCHEDULE, 16).threads).toBeNull();
  });
  it("keeps the GPU power limit inside the driver's range", () => {
    const max = PROFILES.find((p) => p.id === "max")!;
    expect(clampProfile({ ...max, gpu: { ...max.gpu, powerLimitW: 400 } }, GPU_RANGE).gpu.powerLimitW).toBe(GPU_RANGE.maxPowerW);
    expect(clampProfile({ ...max, gpu: { ...max.gpu, powerLimitW: 1 } }, GPU_RANGE).gpu.powerLimitW).toBe(GPU_RANGE.minPowerW);
  });
  it("lists exactly the settings a switch touches", () => {
    const [max, balanced] = [PROFILES.find((p) => p.id === "max")!, PROFILES.find((p) => p.id === "balanced")!];
    const d = diffProfiles(balanced, max);
    expect(d.map((c) => c.what)).toEqual(["Windows power plan", "GPU power limit", "Fan curve"]);
    expect(diffProfiles(max, max)).toEqual([]);
  });
});

describe("keep or revert", () => {
  const base = { tokPerSec: 17.9, gpuTempC: 70, throttled: false };
  it("keeps a real gain", () => {
    expect(keepOrRevert(base, { ...base, tokPerSec: 19 })).toEqual({ keep: true, gainPct: 6.1 });
  });
  it("reverts noise, slowdowns, heat and new throttling", () => {
    expect(keepOrRevert(base, { ...base, tokPerSec: 18.1 }).keep).toBe(false);
    expect(keepOrRevert(base, { ...base, tokPerSec: 15 }).keep).toBe(false);
    expect(keepOrRevert(base, { ...base, tokPerSec: 20, gpuTempC: 90 }).keep).toBe(false);
    expect(keepOrRevert(base, { ...base, tokPerSec: 20, throttled: true }).keep).toBe(false);
  });
});

describe("bottleneck verdict", () => {
  it("names CPU-bound experts", () => {
    expect(bottleneck({ gpuKernels: 0.3, cpuExperts: 0.5, pcie: 0.1, other: 0.1, expertCacheHit: 0.07, draftAcceptance: 0.6, vramFreeGiB: 0.4 })).toMatch(/^CPU-bound on experts: 50%/);
  });
  it("names transfer-bound", () => {
    expect(bottleneck({ gpuKernels: 0.3, cpuExperts: 0.2, pcie: 0.4, other: 0.1, expertCacheHit: 0.5, draftAcceptance: 0.6, vramFreeGiB: 1 })).toMatch(/^Transfer-bound/);
  });
  it("flags low draft acceptance, else GPU-bound", () => {
    expect(bottleneck({ gpuKernels: 0.8, cpuExperts: 0.1, pcie: 0.05, other: 0.05, expertCacheHit: 0.9, draftAcceptance: 0.3, vramFreeGiB: 1 })).toMatch(/speculation off/);
    expect(bottleneck({ gpuKernels: 0.8, cpuExperts: 0.1, pcie: 0.05, other: 0.05, expertCacheHit: 0.9, draftAcceptance: null, vramFreeGiB: 1 })).toMatch(/^GPU-bound: 80%/);
  });
});
