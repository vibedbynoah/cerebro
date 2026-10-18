import type { MachineTuneOutcome, MachineTuneSetting, MachineTuneTrial } from "../service/types";

/**
 * A real recorded tune, replayed by the web preview and tests: the engine's tune module
 * (cerebro_engine::tune) measured a Qwen3-0.6B Q8_0 model on the processor only, on this
 * developer machine — a 12th Gen Intel(R) Core(TM) i9-12900H (6 performance cores, 20 threads)
 * with an NVIDIA GeForce RTX 3070 Ti Laptop GPU — on 2026-10-18. Every number below is copied
 * from that run's log; one trial is emitted every 300 ms, then the outcome.
 */

const setting = (id: string, value: number): MachineTuneSetting => ({ id, value });

/** The three processor settings the tune measures, at the values one trial ran with. */
const at = (threads: number, div: number, rows4: number): MachineTuneSetting[] => [setting("cpu_threads", threads), setting("cpu_chunk_div", div), setting("cpu_rows4", rows4)];

const trial = (phase: MachineTuneTrial["phase"], label: string, settings: MachineTuneSetting[], tps: number, reps: number[]): MachineTuneTrial => ({ phase, label, settings, tps, reps });

/** Every timed setting of the recorded run, in the order the engine emitted it. */
export const MACHINE_TUNE_TRIALS: readonly MachineTuneTrial[] = [
  trial("threads", "6 threads", at(6, 4, 0), 43.4, [43.0, 43.4, 43.4]),
  trial("threads", "12 threads", at(12, 4, 0), 48.3, [48.3, 48.2, 48.9]),
  trial("threads", "16 threads", at(16, 4, 0), 47.6, [48.2, 47.6, 47.4]),
  trial("threads", "18 threads", at(18, 4, 0), 48.1, [48.1, 48.1, 47.4]),
  trial("threads", "20 threads", at(20, 4, 0), 47.1, [46.0, 47.1, 47.6]),
  trial("splitting", "1 pieces per thread", at(12, 1, 0), 43.7, [44.9, 43.7, 42.8]),
  trial("splitting", "2 pieces per thread", at(12, 2, 0), 47.4, [47.4, 41.8, 49.0]),
  trial("splitting", "4 pieces per thread", at(12, 4, 0), 48.6, [48.1, 48.6, 49.1]),
  trial("splitting", "8 pieces per thread", at(12, 8, 0), 47.2, [47.2, 47.7, 46.3]),
  trial("rows4", "Four-row dots off", at(12, 4, 0), 46.7, [46.7, 45.1, 48.3]),
  trial("rows4", "Four-row dots on", at(12, 4, 1), 46.4, [46.4, 47.5, 45.6]),
  trial("confirm", "The winner against the starting settings", at(12, 4, 0), 48.2, [48.2, 47.9, 49.0]),
];

/** The run's finish, copied from the engine's outcome (48.2 against 47.2 tok/s in the confirmation). */
export const MACHINE_TUNE_OUTCOME: MachineTuneOutcome = {
  machine: "12th Gen Intel(R) Core(TM) i9-12900H | 6P20L | NVIDIA GeForce RTX 3070 Ti Laptop GPU 8191 MiB",
  model: "8f4b1c0d9a6e2f37",
  gpuLayers: 0,
  settings: at(12, 4, 0),
  start: at(16, 4, 0),
  startTps: 47.2,
  tunedTps: 48.2,
  gain: 0.022,
  kept: true,
  busy: 0.06,
  note: "As fast with 4 fewer threads, so it runs cooler.",
  trials: [...MACHINE_TUNE_TRIALS],
  seconds: 52,
};

/** A deep copy, so callers can never change the recording. */
export function cloneMachineTuneOutcome(outcome: MachineTuneOutcome): MachineTuneOutcome {
  return {
    ...outcome,
    settings: outcome.settings.map((s) => ({ ...s })),
    start: outcome.start.map((s) => ({ ...s })),
    trials: outcome.trials.map((t) => ({ ...t, settings: t.settings.map((s) => ({ ...s })), reps: [...t.reps] })),
  };
}

/** Resolves after `ms`, or at once when the signal aborts (the caller checks `signal.aborted`). */
const wait = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(t), resolve()), { once: true });
  });

export interface MachineTunePlayerOptions {
  /** One trial every this many milliseconds. Default 300, the recording's pace. */
  trialMs?: number;
}

/** Replays the recorded run: one finished trial every `trialMs`, then the outcome. */
export function createMachineTunePlayer(options: MachineTunePlayerOptions = {}) {
  const trialMs = options.trialMs ?? 300;
  return {
    /** Aborting `signal` rejects with the same sentence the engine sends when cancelled. */
    async play(onTrial: (t: MachineTuneTrial) => void, signal: AbortSignal): Promise<MachineTuneOutcome> {
      for (const t of MACHINE_TUNE_TRIALS) {
        await wait(trialMs, signal);
        if (signal.aborted) throw new Error("Tuning was stopped.");
        onTrial({ ...t, settings: t.settings.map((s) => ({ ...s })), reps: [...t.reps] });
      }
      await wait(trialMs, signal);
      if (signal.aborted) throw new Error("Tuning was stopped.");
      return cloneMachineTuneOutcome(MACHINE_TUNE_OUTCOME);
    },
  };
}
