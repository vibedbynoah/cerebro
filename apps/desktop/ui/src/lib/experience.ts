import { readPref, writePref } from "./storage";

/**
 * Experience level and optimisation channel (docs/specs/positioning.md).
 *
 * - Beginner: Cerebro sets the model file, context, offload, KV cache type and sampler from the
 *   hardware check, hides raw flags and explains every term where it appears.
 * - Advanced: every backend flag, the file choice, every load parameter, the Stable / Beta /
 *   Experimental channels and the dense live metrics.
 *
 * Both are per-machine preferences in local storage, like the theme: they grant nothing, a missing
 * or tampered value falls back to the default, and every value read back is checked against a
 * closed list. Switching level never discards a value: the other view shows the same settings.
 */

export type ExperienceLevel = "beginner" | "advanced";
export const EXPERIENCE_KEY = "cerebro-experience";
/** Pros by default (positioning.md); Beginner is one click away during setup and in Settings. */
export const DEFAULT_LEVEL: ExperienceLevel = "advanced";

export const LEVEL_LABELS: Readonly<Record<ExperienceLevel, string>> = { beginner: "Beginner", advanced: "Advanced" };

export function parseLevel(raw: string | null): ExperienceLevel | null {
  return raw === "beginner" || raw === "advanced" ? raw : null;
}

export function readLevel(): ExperienceLevel {
  return parseLevel(readPref(EXPERIENCE_KEY)) ?? DEFAULT_LEVEL;
}

export function writeLevel(level: ExperienceLevel): void {
  writePref(EXPERIENCE_KEY, level);
}

export function otherLevel(level: ExperienceLevel): ExperienceLevel {
  return level === "beginner" ? "advanced" : "beginner";
}

// ---- Channels ------------------------------------------------------------------------------------

export type Channel = "stable" | "beta" | "experimental";
export const CHANNEL_KEY = "cerebro-channel";
export const DEFAULT_CHANNEL: Channel = "stable";
export const CHANNELS: readonly Channel[] = ["stable", "beta", "experimental"];

export const CHANNEL_LABELS: Readonly<Record<Channel, string>> = { stable: "Stable", beta: "Beta", experimental: "Experimental" };
export const CHANNEL_NOTES: Readonly<Record<Channel, string>> = {
  stable: "Optimisations that passed every check on many machines.",
  beta: "Adds optimisations that pass on most machines. Each is measured here before it is kept.",
  experimental: "Adds the newest optimisations. Fastest when they work; each one still has to pass the quality gate.",
};

export function parseChannel(raw: string | null): Channel | null {
  return raw === "stable" || raw === "beta" || raw === "experimental" ? raw : null;
}

export function readChannel(): Channel {
  return parseChannel(readPref(CHANNEL_KEY)) ?? DEFAULT_CHANNEL;
}

export function writeChannel(channel: Channel): void {
  writePref(CHANNEL_KEY, channel);
}

/** True when a feature released on `feature` is switched on by the chosen channel. */
export function channelIncludes(chosen: Channel, feature: Channel): boolean {
  return CHANNELS.indexOf(feature) <= CHANNELS.indexOf(chosen);
}

export interface Optimisation {
  id: string;
  label: string;
  channel: Channel;
  /** What the user gets, in plain words. */
  note: string;
}

/** The optimisations each channel turns on. Every one is measured on this machine before it is kept. */
export const OPTIMISATIONS: readonly Optimisation[] = [
  { id: "measured-speculation", label: "Measured speculation", channel: "stable", note: "Speculative decoding switches on only where Tune measured a gain on this machine." },
  { id: "prefill-chunking", label: "Autotuned prefill chunking", channel: "stable", note: "Reads long prompts in the chunk size measured fastest here." },
  { id: "expert-cache", label: "MoE expert cache", channel: "beta", note: "Keeps the experts this session uses most in video memory and streams the rest." },
  { id: "state-reuse", label: "Conversation-state reuse", channel: "beta", note: "Keeps a processed conversation between turns, so an agent's next step skips re-reading it." },
  { id: "kv-streaming", label: "KV streaming past 64K", channel: "experimental", note: "Moves older context to system memory so long contexts fit on small cards." },
  { id: "strata", label: "Strata turbo backend", channel: "experimental", note: "Runs the models it supports, including GSQ-RCO files, on Cerebro's own backend." },
];
