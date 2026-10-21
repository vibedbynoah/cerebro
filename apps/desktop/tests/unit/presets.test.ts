import { beforeEach, describe, expect, it } from "vitest";
import { loadPresets, PRESETS_KEY, removePreset, savePreset } from "../../ui/src/lib/chat/presets";
import { DEFAULT_PREFS } from "../../ui/src/lib/chat/prompt";
import { DEFAULT_SAMPLER } from "../../ui/src/service/chat-types";

describe("per-model presets", () => {
  beforeEach(() => localStorage.clear());

  it("saves, loads and forgets a model's settings", () => {
    savePreset("m1", { ...DEFAULT_PREFS, effort: "high" }, { ...DEFAULT_SAMPLER, temperature: 0.2 });
    const p = loadPresets()["m1"]!;
    expect(p.prefs.effort).toBe("high");
    expect(p.sampler.temperature).toBe(0.2);
    removePreset("m1");
    expect(loadPresets()["m1"]).toBeUndefined();
  });

  it("drops anything unreadable or out of range", () => {
    localStorage.setItem(PRESETS_KEY, JSON.stringify({ ok: { prefs: { ...DEFAULT_PREFS }, sampler: DEFAULT_SAMPLER }, bad: { prefs: { effort: "ultra" } }, worse: 3 }));
    expect(Object.keys(loadPresets())).toEqual(["ok"]);
    localStorage.setItem(PRESETS_KEY, "not json");
    expect(loadPresets()).toEqual({});
  });
});
