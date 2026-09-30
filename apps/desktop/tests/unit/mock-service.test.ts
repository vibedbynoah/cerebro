import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SETTINGS, MOCK_MODELS } from "../../ui/src/mock/fixtures";
import { createMockService, isSettingKey } from "../../ui/src/mock/mock-service";
import type { LiveStats, SettingKey, Settings } from "../../ui/src/service/types";

describe("mock service", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("labels every number it produces as illustrative, never measured", async () => {
    const service = createMockService();
    expect((await service.getLiveStats()).provenance).toBe("illustrative");
    for (const m of await service.listModels()) expect(m.provenance).toBe("illustrative");
  });

  it("is safe by default: every protection on, offline until told otherwise", async () => {
    const s = await createMockService().getSettings();
    expect(s.processContainment && s.cleanEnvironment && s.integrityCheck && s.permissionPrompts && s.loopbackOnly).toBe(true);
    expect(await createMockService().getNetwork()).toBe("offline");
    expect(await createMockService({ network: "online" }).getNetwork()).toBe("online");
  });

  it("returns copies, so a caller cannot mutate the service's state", async () => {
    const service = createMockService();
    const a = await service.getLiveStats();
    a.backend.state = "error";
    a.vramUsedGiB = 999;
    const b = await service.getLiveStats();
    expect(b.backend.state).toBe("running");
    expect(b.vramUsedGiB).not.toBe(999);

    const models = await service.listModels();
    models[0]!.name = "tampered";
    expect((await service.listModels())[0]!.name).toBe(MOCK_MODELS[0]!.name);

    const settings = await service.getSettings();
    settings.integrityCheck = false;
    expect((await service.getSettings()).integrityCheck).toBe(true);
  });

  it("ticks only while subscribed, with values inside physical bounds", () => {
    let n = 0;
    const service = createMockService({ tickMs: 100, random: () => (n++ % 2 === 0 ? 0 : 1) });
    const seen: LiveStats[] = [];
    const off = service.subscribeLiveStats((s) => seen.push(s));
    vi.advanceTimersByTime(350);
    expect(seen).toHaveLength(3);
    for (const s of seen) {
      expect(s.vramUsedGiB).toBeGreaterThanOrEqual(0);
      expect(s.vramUsedGiB).toBeLessThanOrEqual(s.vramTotalGiB);
      expect(s.tokensPerSecond).toBeGreaterThan(0);
    }
    off();
    off(); // idempotent
    vi.advanceTimersByTime(1000);
    expect(seen).toHaveLength(3);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("keeps ticking for the remaining subscribers when one leaves", () => {
    const service = createMockService({ tickMs: 100 });
    const a: LiveStats[] = [];
    const b: LiveStats[] = [];
    const offA = service.subscribeLiveStats((s) => a.push(s));
    const offB = service.subscribeLiveStats((s) => b.push(s));
    vi.advanceTimersByTime(100);
    offA();
    vi.advanceTimersByTime(200);
    expect(a).toHaveLength(1);
    expect(b).toHaveLength(3);
    offB();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("ignores a nonsensical tick interval instead of spinning", () => {
    const service = createMockService({ tickMs: -5 });
    const seen: LiveStats[] = [];
    const off = service.subscribeLiveStats((s) => seen.push(s));
    vi.advanceTimersByTime(999);
    expect(seen).toHaveLength(0);
    vi.advanceTimersByTime(1);
    expect(seen).toHaveLength(1);
    off();
  });

  it("lets the user switch off any protection", async () => {
    const service = createMockService();
    for (const key of Object.keys(DEFAULT_SETTINGS) as SettingKey[]) {
      const next = await service.updateSetting(key, false);
      expect(next[key]).toBe(false);
    }
    expect(Object.values(await service.getSettings()).every((v) => v === false)).toBe(true);
  });

  it("rejects unknown keys and non-boolean values, like the Rust side will", async () => {
    const service = createMockService();
    const update = service.updateSetting.bind(service) as (k: unknown, v: unknown) => Promise<Settings>;
    await expect(update("__proto__", true)).rejects.toThrow(TypeError);
    await expect(update("constructor", true)).rejects.toThrow(TypeError);
    await expect(update("sandbox", false)).rejects.toThrow(/Unknown setting/);
    await expect(update("integrityCheck", "false")).rejects.toThrow(/true or false/);
    await expect(update("integrityCheck", 0)).rejects.toThrow(TypeError);
    expect((await service.getSettings()).integrityCheck).toBe(true);
    expect(isSettingKey("hasOwnProperty")).toBe(false);
  });
});
