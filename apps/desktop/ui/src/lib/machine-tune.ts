import type { MachineTuneOutcome, MachineTuneSetting, MachineTuneTrial } from "../service/types";

/**
 * "Tune for this machine" (Turbo screen): the plain-words view of a tuning run
 * (cerebro_engine::tune). Everything here is pure so the rules are unit-tested: the phase titles,
 * grouping the finished trials, the settings in words, and the speed and duration lines.
 */

/** The phases in the order the engine runs them. */
export const MACHINE_TUNE_PHASES = ["threads", "splitting", "rows4", "confirm"] as const;
export type MachineTunePhase = (typeof MACHINE_TUNE_PHASES)[number];

const PHASE_TITLES: Record<MachineTunePhase, string> = {
  threads: "Threads",
  splitting: "Work splitting",
  rows4: "Four-row dots",
  confirm: "Confirm",
};

/** The plain-words title of one phase. */
export function phaseTitle(phase: MachineTuneTrial["phase"]): string {
  return PHASE_TITLES[phase] ?? phase;
}

export interface MachineTuneGroup {
  phase: MachineTunePhase;
  title: string;
  trials: MachineTuneTrial[];
  /** Index in `trials` of the phase's fastest setting; every phase has at least one trial. */
  fastest: number;
}

/** The finished trials grouped by phase, in the engine's order; phases without trials are left out. */
export function groupTrials(trials: readonly MachineTuneTrial[]): MachineTuneGroup[] {
  const groups: MachineTuneGroup[] = [];
  for (const phase of MACHINE_TUNE_PHASES) {
    const inPhase = trials.filter((t) => t.phase === phase);
    if (inPhase.length === 0) continue;
    let fastest = 0;
    inPhase.forEach((t, i) => {
      if (t.tps > inPhase[fastest]!.tps) fastest = i;
    });
    groups.push({ phase, title: PHASE_TITLES[phase], trials: inPhase, fastest });
  }
  return groups;
}

/** One processor setting in words: "12 threads", "4 pieces per thread", "four-row dots off". */
export function settingWords(s: MachineTuneSetting): string {
  switch (s.id) {
    case "cpu_threads":
      return `${s.value} threads`;
    case "cpu_decode_threads":
      return `${s.value} threads while writing`;
    case "cpu_chunk_div":
      return `${s.value} pieces per thread`;
    case "cpu_rows4":
      return `four-row dots ${s.value !== 0 ? "on" : "off"}`;
    default:
      return `${s.id} ${s.value}`;
  }
}

/** The settings in words, in the engine's own order: "12 threads, 4 pieces per thread, four-row dots off". */
export function settingsInWords(settings: readonly MachineTuneSetting[]): string {
  return settings.map(settingWords).join(", ");
}

/** A gain (tuned / start - 1) as a signed percentage: 0.022 is "+2.2%". */
export function formatGain(gain: number): string {
  return `${gain >= 0 ? "+" : ""}${(gain * 100).toFixed(1)}%`;
}

/** The confirmation line: "47.2 → 48.2 tok/s (+2.2%)". */
export function speedLine(outcome: MachineTuneOutcome): string {
  return `${outcome.startTps.toFixed(1)} → ${outcome.tunedTps.toFixed(1)} tok/s (${formatGain(outcome.gain)})`;
}

/** One finished trial: "12 threads — 48.3 tok/s". */
export function trialLine(t: MachineTuneTrial): string {
  return `${t.label} — ${t.tps.toFixed(1)} tok/s`;
}

/** How long a tune took, in words: "52 s", "1 min 40 s" or "2 min". */
export function formatSeconds(seconds: number): string {
  const total = Math.round(seconds);
  if (total < 60) return `${total} s`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} s`;
}
