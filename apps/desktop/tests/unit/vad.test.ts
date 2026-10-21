import { describe, expect, it } from "vitest";
import { EndOfSpeech, thresholdFor } from "../../ui/src/lib/chat/vad";

const run = (levels: number[], stepMs = 100) => {
  const v = new EndOfSpeech({ threshold: 0.1, silenceMs: 800, minSpeechMs: 250 });
  return levels.map((l, i) => v.feed(l, i * stepMs));
};

describe("end of speech", () => {
  it("waits through silence, hears speech, and ends after 0.8 s of quiet", () => {
    const states = run([0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(states.slice(0, 4)).toEqual(["waiting", "waiting", "waiting", "waiting"]);
    expect(states[5]).toBe("speaking");
    expect(states.indexOf("done")).toBe(14);
  });

  it("a click is not speech, and a pause shorter than the silence keeps the turn going", () => {
    expect(run([0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).every((s) => s === "waiting")).toBe(true);
    const s = run([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(s.indexOf("done")).toBe(16);
  });

  it("more sensitivity means a lower threshold", () => {
    expect(thresholdFor(1)).toBeLessThan(thresholdFor(0));
    expect(thresholdFor(2)).toBe(thresholdFor(1));
  });
});
