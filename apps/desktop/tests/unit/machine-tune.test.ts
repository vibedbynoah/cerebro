import { describe, expect, it } from "vitest";
import { formatGain, formatSeconds, groupTrials, MACHINE_TUNE_PHASES, phaseTitle, settingWords, settingsInWords, speedLine, trialLine } from "../../ui/src/lib/machine-tune";
import { cloneMachineTuneOutcome, MACHINE_TUNE_OUTCOME, MACHINE_TUNE_TRIALS } from "../../ui/src/mock/machine-tune";
import type { MachineTuneSetting, MachineTuneTrial } from "../../ui/src/service/types";

const setting = (id: string, value: number): MachineTuneSetting => ({ id, value });
const trial = (phase: MachineTuneTrial["phase"], label: string, tps: number): MachineTuneTrial => ({ phase, label, settings: [setting("cpu_threads", 12)], tps, reps: [tps, tps, tps] });

describe("phase titles", () => {
  it("names every phase of a run in plain words", () => {
    expect(MACHINE_TUNE_PHASES).toEqual(["threads", "splitting", "rows4", "confirm"]);
    expect(phaseTitle("threads")).toBe("Threads");
    expect(phaseTitle("splitting")).toBe("Work splitting");
    expect(phaseTitle("rows4")).toBe("Four-row dots");
    expect(phaseTitle("confirm")).toBe("Confirm");
    // An unknown phase from a newer engine falls back to its own name rather than breaking.
    expect(phaseTitle("future" as never)).toBe("future");
  });
});

describe("grouping finished trials", () => {
  it("keeps the engine's phase order and leaves out phases without trials", () => {
    const groups = groupTrials([trial("confirm", "The winner", 48.2), trial("threads", "6 threads", 43.4), trial("splitting", "4 pieces per thread", 48.6)]);
    expect(groups.map((g) => g.phase)).toEqual(["threads", "splitting", "confirm"]);
    expect(groups.map((g) => g.title)).toEqual(["Threads", "Work splitting", "Confirm"]);
    expect(groupTrials([])).toEqual([]);
  });

  it("marks the fastest setting in every phase", () => {
    const groups = groupTrials([
      trial("threads", "6 threads", 43.4),
      trial("threads", "12 threads", 48.3),
      trial("threads", "16 threads", 47.6),
      trial("splitting", "1 pieces per thread", 43.7),
      trial("splitting", "4 pieces per thread", 48.6),
      trial("splitting", "8 pieces per thread", 47.2),
    ]);
    const [threads, splitting] = groups;
    expect(threads!.fastest).toBe(1);
    expect(threads!.trials[threads!.fastest]!.label).toBe("12 threads");
    expect(splitting!.fastest).toBe(1);
    expect(splitting!.trials[splitting!.fastest]!.label).toBe("4 pieces per thread");
  });

  it("gives a single-trial phase to the only setting there is", () => {
    const groups = groupTrials([trial("confirm", "The winner against the starting settings", 48.2)]);
    expect(groups).toHaveLength(1);
    expect(groups[0]!.fastest).toBe(0);
  });
});

describe("settings in words", () => {
  it("says each known setting the way the engine describes it", () => {
    expect(settingWords(setting("cpu_threads", 12))).toBe("12 threads");
    expect(settingWords(setting("cpu_decode_threads", 20))).toBe("20 threads while writing");
    expect(settingWords(setting("cpu_chunk_div", 4))).toBe("4 pieces per thread");
    expect(settingWords(setting("cpu_rows4", 0))).toBe("four-row dots off");
    expect(settingWords(setting("cpu_rows4", 1))).toBe("four-row dots on");
    // A setting this version does not know is shown as is, not hidden.
    expect(settingWords(setting("cpu_mystery", 3))).toBe("cpu_mystery 3");
  });

  it("joins the winner's three settings in the engine's own order", () => {
    expect(settingsInWords([setting("cpu_threads", 12), setting("cpu_chunk_div", 4), setting("cpu_rows4", 0)])).toBe("12 threads, 4 pieces per thread, four-row dots off");
  });
});

describe("speed and duration lines", () => {
  it("shows the gain with its sign", () => {
    expect(formatGain(0.022)).toBe("+2.2%");
    expect(formatGain(-0.004)).toBe("-0.4%");
    expect(formatGain(0)).toBe("+0.0%");
  });

  it("copies the outcome into one confirmation line", () => {
    expect(speedLine({ ...MACHINE_TUNE_OUTCOME, startTps: 47.2, tunedTps: 48.2, gain: 0.022 })).toBe("47.2 → 48.2 tok/s (+2.2%)");
  });

  it("writes one trial as label and speed", () => {
    expect(trialLine(trial("threads", "12 threads", 48.3))).toBe("12 threads — 48.3 tok/s");
  });

  it("writes a duration in seconds under a minute and minutes above", () => {
    expect(formatSeconds(52)).toBe("52 s");
    expect(formatSeconds(59.6)).toBe("1 min");
    expect(formatSeconds(100)).toBe("1 min 40 s");
    expect(formatSeconds(120)).toBe("2 min");
  });
});

describe("the recorded tune", () => {
  it("holds every trial the engine timed, with the median of its three runs", () => {
    expect(MACHINE_TUNE_TRIALS).toHaveLength(12);
    for (const t of MACHINE_TUNE_TRIALS) {
      expect(t.reps, t.label).toHaveLength(3);
      const median = [...t.reps].sort((a, b) => a - b)[1]!;
      expect(t.tps, t.label).toBe(median);
    }
  });

  it("was run on this machine's processor and a real model and kept 12 threads", () => {
    expect(MACHINE_TUNE_OUTCOME.machine).toContain("i9-12900H");
    expect(MACHINE_TUNE_OUTCOME.model).toBe("8f4b1c0d9a6e2f37");
    expect(MACHINE_TUNE_OUTCOME.kept).toBe(true);
    expect(MACHINE_TUNE_OUTCOME.seconds).toBe(52);
    expect(settingsInWords(MACHINE_TUNE_OUTCOME.settings)).toBe("12 threads, 4 pieces per thread, four-row dots off");
    expect(MACHINE_TUNE_OUTCOME.note).toBe("As fast with 4 fewer threads, so it runs cooler.");
    expect(cloneMachineTuneOutcome(MACHINE_TUNE_OUTCOME).trials).toEqual(MACHINE_TUNE_TRIALS.map((t) => ({ ...t, settings: t.settings.map((s) => ({ ...s })) })));
  });

  it("hands callers a copy they cannot change the recording through", () => {
    const copy = cloneMachineTuneOutcome(MACHINE_TUNE_OUTCOME);
    copy.settings[0]!.value = 99;
    copy.trials[0]!.reps.push(1);
    expect(MACHINE_TUNE_OUTCOME.settings[0]!.value).toBe(12);
    expect(MACHINE_TUNE_OUTCOME.trials[0]!.reps).toHaveLength(3);
  });
});
