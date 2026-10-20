import { describe, expect, it } from "vitest";
import { ACCURATE_PROMPT, applyMode, FASTEST_PROMPT, MODE_PRESETS, MODES, modeOf, parseSampler } from "../../ui/src/lib/chat/modes";
import { buildRequest, DEFAULT_PREFS, EFFORT_LEVELS } from "../../ui/src/lib/chat/prompt";
import { DEFAULT_SAMPLER } from "../../ui/src/service/chat-types";

describe("the Speed <-> Accuracy slider", () => {
  it("a fresh install sits at Balanced", () => {
    expect(modeOf(DEFAULT_PREFS, DEFAULT_SAMPLER)).toBe(3);
  });

  it("every level is detected after it is applied, from any other level", () => {
    for (const from of MODES) {
      const start = applyMode(from, DEFAULT_PREFS, DEFAULT_SAMPLER);
      for (const to of MODES) {
        const s = applyMode(to, start.prefs, start.sampler);
        expect(modeOf(s.prefs, s.sampler)).toBe(to);
      }
    }
  });

  it("Fastest picks words greedily, so the engine's on-card loop runs, with thinking off and a one-line prompt", () => {
    const s = applyMode(1, DEFAULT_PREFS, DEFAULT_SAMPLER);
    expect(s.sampler.temperature).toBe(0);
    expect(s.sampler.repeatPenalty).toBe(1);
    expect(s.sampler.maxTokens).toBe(1024);
    const req = buildRequest({ levels: EFFORT_LEVELS, base: s.prefs.base, effort: s.prefs.effort, verbosity: s.prefs.verbosity, source: s.prefs.source, addInstructions: s.prefs.addInstructions, memories: [{ id: "m", text: "Likes tea.", scope: "you" }], memoryOn: s.prefs.memoryOn, nativeSupported: true, modelMaxThinking: 32768 });
    expect(req.native?.thinkingBudget).toBe(0);
    expect(req.passes).toBe(0);
    expect(req.systemPrompt.startsWith(FASTEST_PROMPT)).toBe(true);
    expect(req.systemPrompt).not.toContain("Likes tea.");
    // Far shorter than the Accurate level's prompt.
    const acc = applyMode(4, DEFAULT_PREFS, DEFAULT_SAMPLER);
    const accReq = buildRequest({ levels: EFFORT_LEVELS, base: acc.prefs.base, effort: acc.prefs.effort, verbosity: acc.prefs.verbosity, source: acc.prefs.source, addInstructions: true, memories: [], memoryOn: true, nativeSupported: true, modelMaxThinking: 32768 });
    expect(req.systemPrompt.length * 5).toBeLessThan(accReq.systemPrompt.length);
    expect(accReq.systemPrompt.startsWith(ACCURATE_PROMPT)).toBe(true);
    expect(accReq.passes).toBe(1);
  });

  it("Maximum turns on Orchestrate with five workers", () => {
    const s = applyMode(5, DEFAULT_PREFS, DEFAULT_SAMPLER);
    expect(s.prefs.orchestrate).toBe(true);
    expect(s.prefs.workers).toBe(5);
    expect(s.prefs.effort).toBe("max");
  });

  it("a prompt the user wrote is kept; the default follows the slider", () => {
    const own = { ...DEFAULT_PREFS, base: "You are my pirate." };
    expect(applyMode(1, own, DEFAULT_SAMPLER).prefs.base).toBe("You are my pirate.");
    expect(applyMode(1, DEFAULT_PREFS, DEFAULT_SAMPLER).prefs.base).toBe(FASTEST_PROMPT);
    expect(modeOf(applyMode(4, own, DEFAULT_SAMPLER).prefs, applyMode(4, own, DEFAULT_SAMPLER).sampler)).toBe(4);
  });

  it("changing one control by hand shows Custom", () => {
    const s = applyMode(2, DEFAULT_PREFS, DEFAULT_SAMPLER);
    expect(modeOf({ ...s.prefs, effort: "high" }, s.sampler)).toBeNull();
    expect(modeOf(s.prefs, { ...s.sampler, temperature: 0.9 })).toBeNull();
  });

  it("every level says what it does and what changes", () => {
    for (const m of MODES) {
      expect(MODE_PRESETS[m].summary.length).toBeGreaterThan(20);
      expect(MODE_PRESETS[m].changes.length).toBeGreaterThan(0);
    }
  });

  it("a stored sampler is re-checked field by field", () => {
    expect(parseSampler(null)).toEqual(DEFAULT_SAMPLER);
    expect(parseSampler("not json")).toEqual(DEFAULT_SAMPLER);
    expect(parseSampler(JSON.stringify({ temperature: 0, maxTokens: 1024, topK: -3 }))).toEqual({ ...DEFAULT_SAMPLER, temperature: 0, maxTokens: 1024 });
  });
});
