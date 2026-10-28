import { DEFAULT_SAMPLER, type Sampler } from "../../service/chat-types";
import { readPref, writePref } from "../storage";
import { DEFAULT_BASE_PROMPT, type ChatPrefs } from "./prompt";

/**
 * The Speed <-> Accuracy slider: five presets over Chat's existing controls (effort, verbosity,
 * reasoning source, Orchestrate, the system prompt and the sampler). Moving it sets every one of
 * them; the individual controls stay available, and changing one shows the slider as "Custom".
 * Pure, so the presets and their detection are tested.
 */

export type Mode = 1 | 2 | 3 | 4 | 5;
export const MODES: readonly Mode[] = [1, 2, 3, 4, 5];

/** The system prompts the speed and accuracy ends use in place of the default one. ACCURATE_PROMPT
 *  was drafted for small local models and chosen by measurement: 8 of 8 right against 7 of 8 for
 *  the shorter one it replaced (Qwen3.5-4B, docs/measurements/engine/2026-10-19-multimodal.md). */
export const FASTEST_PROMPT = "Answer directly and briefly. No preamble, no restating the question.";
export const FAST_PROMPT = "You are a helpful assistant on this computer. Be concise and correct.";
export const ACCURATE_PROMPT = [
  "You are a careful, accurate assistant. Your job is to give the correct answer, not the fastest one. Small errors matter, so slow down and check your work before you reply.",
  "First, restate the question to yourself in your own words. Identify exactly what is being asked: what is given, what must be found or decided, what units or format the answer needs, and how many parts the question has. If the request is unclear, choose the most reasonable reading and say briefly what you assumed.",
  "Then answer the question that was asked. Put the direct answer first, in plain language, followed by only the explanation needed to support it. Do not pad with background, repeat the question at length, or describe your process.",
  "For math, logic, and code, work through the problem one step at a time. Write each operation and its result so an error is visible, and keep the arithmetic exact until the final rounding. Check units, signs, order of operations, rounding, edge values, and special cases such as zero, one, negatives, empty input, very large input, and repeated values. For code, state the expected input and output, give complete runnable code rather than a fragment, trace it on a simple case and an edge case, and check syntax, variable names, and off-by-one errors. For a proof or argument, check that each conclusion follows from the ones before it.",
  "For facts, dates, names, numbers, and quotes, use the exact value you are confident of. Prefer an exact fact over a vague summary. Do not invent details or fill gaps with guesses. If you do not know, say so plainly and give the best partial answer you can, marked as uncertain. If the answer depends on a condition, state the condition. Never pretend to have checked something you did not check.",
  "Before you finish, check the answer against the original question. Re-read the question. Confirm that you answered every part, used the right units and format, and did not say something different from earlier. Work out the key result a second way when possible. Remove any sentence that does not help the user. Keep the restatement and checks private; give only the answer.",
  "Reply in clear, plain language, with short sentences. Never tell the user to wait. If you are unsure, say what is uncertain and why. Accuracy matters more than speed.",
].join("\n\n");

export interface ModePreset {
  label: string;
  /** One line for the notice and the hint: what it does and what it costs. */
  summary: string;
  /** What actually changes, for the notice. */
  changes: readonly string[];
  prefs: Pick<ChatPrefs, "effort" | "effortCustom" | "thinking" | "verbosity" | "source" | "addInstructions" | "memoryOn" | "orchestrate" | "workers" | "rounds">;
  base: string;
  sampler: Partial<Sampler>;
}

export const MODE_PRESETS: Readonly<Record<Mode, ModePreset>> = {
  1: {
    label: "Fastest",
    summary: "The quickest reply: no thinking, a one-line system prompt, and the word picker running entirely on the graphics card. Fine for simple questions; it gets more multi-step problems wrong.",
    changes: [
      "Thinking off",
      "System prompt cut to one line, so less to read before the first word",
      "Memories not added",
      "Greedy decoding: on an NVIDIA card every word is picked on the card, with no round trip per word",
      "Replies up to 8,192 tokens",
    ],
    prefs: { effortCustom: null, thinking: "off", effort: "low", verbosity: "brief", source: "native", addInstructions: true, memoryOn: false, orchestrate: false, workers: 3, rounds: 3 },
    base: FASTEST_PROMPT,
    sampler: { temperature: 0, repeatPenalty: 1, topK: 1, topP: 1, minP: 0, maxTokens: 8192 },
  },
  2: {
    label: "Fast",
    summary: "Short, direct answers with thinking off and a short system prompt.",
    changes: ["Thinking off", "A short system prompt", "Brief answers", "Replies up to 8,192 tokens"],
    prefs: { effortCustom: null, thinking: "off", effort: "low", verbosity: "brief", source: "native", addInstructions: true, memoryOn: true, orchestrate: false, workers: 3, rounds: 3 },
    base: FAST_PROMPT,
    sampler: { temperature: 0.5, repeatPenalty: 1.05, topK: 40, topP: 0.95, minP: 0.05, maxTokens: 8192 },
  },
  3: {
    label: "Balanced",
    summary: "The default: the model thinks first (up to 1,024 tokens) where it supports it, and checks key facts and numbers. Much more often right on multi-step problems, and many times slower than Fast.",
    changes: ["Thinking Low: up to 1,024 thinking tokens", "Medium effort", "Key facts and numbers checked", "Saved memories added"],
    prefs: { effortCustom: null, thinking: "low", effort: "medium", verbosity: "brief", source: "native", addInstructions: true, memoryOn: true, orchestrate: false, workers: 3, rounds: 3 },
    base: DEFAULT_BASE_PROMPT,
    sampler: { temperature: 0.7, repeatPenalty: 1.05, topK: 40, topP: 0.95, minP: 0.05, maxTokens: 16384 },
  },
  4: {
    label: "Accurate",
    summary: "Long thinking, a detailed correctness prompt and a review pass: about 3-4 times slower than Balanced.",
    changes: ["Thinking High: up to 16,384 thinking tokens", "Extra-high effort", "A detailed prompt for checking every step", "The draft is reviewed once and corrected", "Lower temperature"],
    prefs: { effortCustom: null, thinking: "high", effort: "xhigh", verbosity: "normal", source: "both", addInstructions: true, memoryOn: true, orchestrate: false, workers: 3, rounds: 3 },
    base: ACCURATE_PROMPT,
    sampler: { temperature: 0.4, repeatPenalty: 1.05, topK: 40, topP: 0.95, minP: 0.05, maxTokens: 16384 },
  },
  5: {
    label: "Maximum",
    summary: "Orchestrate: the model plans, five fresh instances take on parts each round through a shared ledger, and a final pass writes the answer. Many model calls per reply: for hard problems.",
    changes: [
      "Orchestrate on: a manager plans and curates tasks in a shared ledger (plan, notes, tasks, result)",
      "Five workers per round, each a fresh instance trying a different approach, up to three rounds",
      "Thinking Max, maximum effort and two review passes in each call",
      "On one graphics card the instances take turns, so this is the slowest setting",
    ],
    prefs: { effortCustom: null, thinking: "max", effort: "max", verbosity: "normal", source: "both", addInstructions: true, memoryOn: true, orchestrate: true, workers: 5, rounds: 3 },
    base: ACCURATE_PROMPT,
    sampler: { temperature: 0.4, repeatPenalty: 1.05, topK: 40, topP: 0.95, minP: 0.05, maxTokens: 16384 },
  },
};

const PRESET_BASES = new Set(MODES.map((m) => MODE_PRESETS[m].base));

/** The preferences and sampler after moving the slider to `m`. The system prompt follows the
 *  slider only while it is the default or another level's; a prompt the user wrote is kept. */
export function applyMode(m: Mode, prefs: ChatPrefs, sampler: Sampler): { prefs: ChatPrefs; sampler: Sampler } {
  const p = MODE_PRESETS[m];
  const base = PRESET_BASES.has(prefs.base.trim()) ? p.base : prefs.base;
  return { prefs: { ...prefs, ...p.prefs, base }, sampler: { ...sampler, ...p.sampler } };
}

/** The level the current settings match, or null ("Custom") once any of them was changed by hand. */
export function modeOf(prefs: ChatPrefs, sampler: Sampler): Mode | null {
  for (const m of MODES) {
    const p = MODE_PRESETS[m];
    const prefsMatch = (Object.keys(p.prefs) as (keyof ModePreset["prefs"])[]).every((k) => (k === "workers" || k === "rounds") && !p.prefs.orchestrate ? true : prefs[k] === p.prefs[k]);
    const samplerMatch = (Object.keys(p.sampler) as (keyof Sampler)[]).every((k) => sampler[k] === p.sampler[k]);
    const baseMatch = !PRESET_BASES.has(prefs.base.trim()) || prefs.base.trim() === p.base;
    if (prefsMatch && samplerMatch && baseMatch) return m;
  }
  return null;
}

// ---- The sampler, kept between sessions (so the slider's level survives a restart) ---------------

export const SAMPLER_KEY = "cerebro-chat-sampler";

/** A stored sampler, each field re-checked against its range; anything else falls back per field. */
export function parseSampler(raw: string | null): Sampler {
  let v: Record<string, unknown> = {};
  try {
    const p: unknown = raw ? JSON.parse(raw) : null;
    if (p && typeof p === "object") v = p as Record<string, unknown>;
  } catch {
    return { ...DEFAULT_SAMPLER };
  }
  const num = (x: unknown, min: number, max: number, d: number): number => (typeof x === "number" && Number.isFinite(x) && x >= min && x <= max ? x : d);
  const int = (x: unknown, min: number, max: number, d: number): number => (typeof x === "number" && Number.isInteger(x) && x >= min && x <= max ? x : d);
  const d = DEFAULT_SAMPLER;
  return {
    temperature: num(v.temperature, 0, 2, d.temperature),
    topP: num(v.topP, 0, 1, d.topP),
    minP: num(v.minP, 0, 1, d.minP),
    topK: int(v.topK, 0, 1000, d.topK),
    repeatPenalty: num(v.repeatPenalty, 0.5, 2, d.repeatPenalty),
    // The old level caps (1,024 / 2,048 / 4,096) cut code and long answers off mid-file: a saved
    // sampler still holding one of them moves to the level's new limit.
    maxTokens: OLD_CAPS[int(v.maxTokens, 16, 131072, d.maxTokens)] ?? int(v.maxTokens, 16, 131072, d.maxTokens),
  };
}

const OLD_CAPS: Readonly<Record<number, number>> = { 1024: 8192, 2048: 8192, 4096: 16384 };

export function loadSampler(): Sampler {
  return parseSampler(readPref(SAMPLER_KEY));
}

export function saveSampler(s: Sampler): void {
  writePref(SAMPLER_KEY, JSON.stringify(s));
}
