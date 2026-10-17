import { readPref, writePref } from "../storage";

/**
 * What Chat sends with each message: the system prompt the user can see and edit, and the native
 * reasoning parameters. Implements docs/specs/effort-levels.md exactly: five effort levels, each
 * setting a thinking budget, a verification instruction and self-check passes; and the reasoning
 * source (native model effort, Cerebro prompts, both, or neither). Pure, so every rule is tested.
 */

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";
export const EFFORTS: readonly Effort[] = ["low", "medium", "high", "xhigh", "max"];
export const EFFORT_LABELS: Readonly<Record<Effort, string>> = { low: "Low", medium: "Medium", high: "High", xhigh: "XHigh", max: "Max" };

export type Verbosity = "brief" | "normal" | "detailed";
export const VERBOSITIES: readonly Verbosity[] = ["brief", "normal", "detailed"];
export const VERBOSITY_LABELS: Readonly<Record<Verbosity, string>> = { brief: "Brief", normal: "Normal", detailed: "Detailed" };

/** Where reasoning control comes from (effort-levels.md, "Reasoning source"). */
export type ReasoningSource = "native" | "prompts" | "both" | "none";
export const REASONING_SOURCES: readonly ReasoningSource[] = ["native", "prompts", "both", "none"];
export const REASONING_LABELS: Readonly<Record<ReasoningSource, string>> = {
  native: "Native only",
  prompts: "Cerebro prompts",
  both: "Both",
  none: "Neither",
};
export const REASONING_NOTES: Readonly<Record<ReasoningSource, string>> = {
  native: "The model's own reasoning control. Nothing is added to the system prompt.",
  prompts: "Cerebro's verification instruction and review passes, without the model's own control.",
  both: "The model's own control plus Cerebro's verification instruction and review passes.",
  none: "The raw model: no added prompt, no review passes, default reasoning.",
};

export interface EffortLevel {
  /** Thinking budget in tokens; 0 = off. Max uses the model's own ceiling. */
  thinkingTokens: number;
  /** reasoning_effort for models that take one. */
  nativeEffort: "low" | "medium" | "high";
  instruction: string;
  /** Review passes after the draft: 0 none, 1 self-review, 2 self-review plus an independent check. */
  passes: 0 | 1 | 2;
  /** Illustrative slowdown vs Low, shown until this machine has measured its own. */
  slowdown: string;
}

/** The prompt drafts from effort-levels.md, verbatim. */
export const EFFORT_LEVELS: Readonly<Record<Effort, EffortLevel>> = {
  low: {
    thinkingTokens: 0,
    nativeEffort: "low",
    instruction: "Be fast and direct. Use what you already know. Only stop to double-check when an error would have real consequences, such as money, safety or commands that change files.",
    passes: 0,
    slowdown: "fastest",
  },
  medium: {
    thinkingTokens: 1024,
    nativeEffort: "low",
    instruction: "Before answering, check the key facts, numbers and any code you write. If you are unsure about something important, say so in one short sentence.",
    passes: 0,
    slowdown: "about 1.2-1.5x",
  },
  high: {
    thinkingTokens: 4096,
    nativeEffort: "medium",
    instruction: "Think the problem through before answering. Verify every claim, calculation and line of code you give. State your assumptions. After drafting, reread your answer once and fix any mistake.",
    passes: 1,
    slowdown: "about 2x",
  },
  xhigh: {
    thinkingTokens: 16384,
    nativeEffort: "high",
    instruction: "Assume every part of your answer is wrong until you have checked it. Recompute every number, trace every code path line by line, consider edge cases, and verify each claim. After drafting, review the whole answer critically and correct it. Clearly mark anything you could not verify.",
    passes: 1,
    slowdown: "about 3-4x",
  },
  max: {
    thinkingTokens: Number.POSITIVE_INFINITY,
    nativeEffort: "high",
    instruction: "Verify everything; leave nothing unchecked. Recompute, retrace and re-derive every part of your answer. When your draft is done, review it as a demanding expert who is trying to find a single flaw in any line, number, claim or assumption, and fix what you find. Only give the answer once it would survive that review. If something cannot be confirmed, say precisely what and why instead of hiding it.",
    passes: 2,
    slowdown: "slowest",
  },
};

export const VERBOSITY_INSTRUCTIONS: Readonly<Record<Verbosity, string>> = {
  brief: "Answer first, in as few words as the question needs. Give details only when asked.",
  normal: "Answer clearly with the context a careful reader needs, without padding.",
  detailed: "Give a thorough answer: the reasoning, the alternatives you considered and worked examples.",
};

export const DEFAULT_BASE_PROMPT = "You are a helpful assistant running locally on this computer.";

export interface Memory {
  id: string;
  text: string;
  scope: "you" | "project";
}

export interface PromptInputs {
  /** The levels in force (the defaults, or the user's edits from Settings). */
  levels?: Readonly<Record<Effort, EffortLevel>>;
  base: string;
  effort: Effort;
  verbosity: Verbosity;
  source: ReasoningSource;
  /** False when the user chose "no added prompt": only `base` is sent. */
  addInstructions: boolean;
  memories: readonly Memory[];
  /** Per-chat "don't remember this", or the global memory switch off. */
  memoryOn: boolean;
  /** Does the loaded model expose a native reasoning control? */
  nativeSupported: boolean;
  /** The model's own ceiling for thinking tokens (Max uses it). */
  modelMaxThinking: number;
}

export interface BuiltRequest {
  systemPrompt: string;
  /** Sent as backend parameters; null when the model has no native control or the source excludes it. */
  native: { reasoningEffort: "low" | "medium" | "high"; thinkingBudget: number } | null;
  passes: 0 | 1 | 2;
  /** Tokens reserved for the answer itself, separate from thinking (the reasoning-budget trap). */
  answerReserve: number;
  /** Plain notes shown in the composer, e.g. that the model has no native control. */
  notes: string[];
}

export const ANSWER_RESERVE = 2048;

export function buildRequest(i: PromptInputs): BuiltRequest {
  const level = (i.levels ?? EFFORT_LEVELS)[i.effort];
  const notes: string[] = [];
  const usePrompts = i.addInstructions && (i.source === "prompts" || i.source === "both");
  const wantNative = i.source === "native" || i.source === "both";

  const parts = [i.base.trim()];
  if (i.addInstructions) {
    parts.push(VERBOSITY_INSTRUCTIONS[i.verbosity]);
    if (usePrompts) parts.push(level.instruction);
    if (i.memoryOn && i.memories.length) parts.push(`What you know about the user:\n${i.memories.map((m) => `- ${m.text}`).join("\n")}`);
  }

  let native: BuiltRequest["native"] = null;
  if (wantNative) {
    if (i.nativeSupported) {
      const budget = Number.isFinite(level.thinkingTokens) ? Math.min(level.thinkingTokens, i.modelMaxThinking) : i.modelMaxThinking;
      native = { reasoningEffort: level.nativeEffort, thinkingBudget: budget };
    } else {
      notes.push("This model has no native reasoning control, so only the verification instruction applies.");
    }
  }
  // Native only on a model without native control would otherwise mean no effort at all: fall back
  // to the instruction so the chosen level still does something, and say so.
  const fallbackPrompt = i.source === "native" && !i.nativeSupported && i.addInstructions;
  if (fallbackPrompt) parts.push(level.instruction);

  const passes = i.source === "none" || !(usePrompts || fallbackPrompt) ? 0 : level.passes;
  if (!i.addInstructions) notes.push("No added prompt: the model sees only your system prompt.");

  return { systemPrompt: parts.filter(Boolean).join("\n\n"), native, passes, answerReserve: ANSWER_RESERVE, notes };
}

// ---- Persisted chat preferences -----------------------------------------------------------------

export interface ChatPrefs {
  effort: Effort;
  verbosity: Verbosity;
  source: ReasoningSource;
  addInstructions: boolean;
  base: string;
  memoryOn: boolean;
  /** Orchestrate mode on: several instances of the model through a shared ledger (GVS5H). */
  orchestrate: boolean;
  /** Workers per round, 1-6. */
  workers: number;
  /** Rounds of manage -> workers before the result is finalized, 1-8. */
  rounds: number;
}

export const DEFAULT_PREFS: ChatPrefs = { effort: "medium", verbosity: "brief", source: "native", addInstructions: true, base: DEFAULT_BASE_PROMPT, memoryOn: true, orchestrate: false, workers: 3, rounds: 3 };

const isOneOf = <T extends string>(list: readonly T[], v: unknown): v is T => typeof v === "string" && (list as readonly string[]).includes(v);

/** Anything stored is re-checked against closed lists; a tampered or partial value falls back per field. */
export function parsePrefs(raw: string | null): ChatPrefs {
  if (!raw) return { ...DEFAULT_PREFS };
  let v: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_PREFS };
    v = parsed as Record<string, unknown>;
  } catch {
    return { ...DEFAULT_PREFS };
  }
  const inRange = (x: unknown, min: number, max: number, d: number): number => (typeof x === "number" && Number.isInteger(x) && x >= min && x <= max ? x : d);
  return {
    effort: isOneOf(EFFORTS, v.effort) ? v.effort : DEFAULT_PREFS.effort,
    verbosity: isOneOf(VERBOSITIES, v.verbosity) ? v.verbosity : DEFAULT_PREFS.verbosity,
    source: isOneOf(REASONING_SOURCES, v.source) ? v.source : DEFAULT_PREFS.source,
    addInstructions: typeof v.addInstructions === "boolean" ? v.addInstructions : DEFAULT_PREFS.addInstructions,
    base: typeof v.base === "string" && v.base.length <= 8000 ? v.base : DEFAULT_PREFS.base,
    memoryOn: typeof v.memoryOn === "boolean" ? v.memoryOn : DEFAULT_PREFS.memoryOn,
    orchestrate: typeof v.orchestrate === "boolean" ? v.orchestrate : DEFAULT_PREFS.orchestrate,
    workers: inRange(v.workers, 1, 6, DEFAULT_PREFS.workers),
    rounds: inRange(v.rounds, 1, 8, DEFAULT_PREFS.rounds),
  };
}

/** Model families whose chat template exposes a native reasoning control (thinking on/off, budget
 *  or reasoning_effort). The engine reads this from the GGUF chat template; this list is the mock's. */
const NATIVE_REASONING = /qwen3|qwq|gpt-oss|deepseek|glm-4\.[5-9]|magistral|gemma-?4|phi-4-reasoning|kimi|minimax/i;
export function supportsNativeReasoning(nameOrFamily: string): boolean {
  return NATIVE_REASONING.test(nameOrFamily);
}

// ---- Shared storage (Chat and Settings read and write the same values) --------------------------


export const PREFS_KEY = "cerebro-chat-prefs";
export const LEVELS_KEY = "cerebro-effort-levels";

export function loadPrefs(): ChatPrefs {
  return parsePrefs(readPref(PREFS_KEY));
}

export function savePrefs(p: ChatPrefs): void {
  writePref(PREFS_KEY, JSON.stringify(p));
}

export type LevelEdits = Partial<Record<Effort, Partial<Pick<EffortLevel, "instruction" | "thinkingTokens" | "passes">>>>;

/** User edits to each effort level (Settings > Effort), validated field by field. */
export function parseLevelEdits(raw: string | null): LevelEdits {
  if (!raw) return {};
  try {
    const v = JSON.parse(raw) as Record<string, unknown>;
    const out: LevelEdits = {};
    for (const e of EFFORTS) {
      const x = v?.[e] as Record<string, unknown> | undefined;
      if (!x || typeof x !== "object") continue;
      const edit: Partial<Pick<EffortLevel, "instruction" | "thinkingTokens" | "passes">> = {};
      if (typeof x.instruction === "string" && x.instruction.length <= 4000) edit.instruction = x.instruction;
      if (typeof x.thinkingTokens === "number" && Number.isInteger(x.thinkingTokens) && x.thinkingTokens >= 0 && x.thinkingTokens <= 262144) edit.thinkingTokens = x.thinkingTokens;
      if (x.passes === 0 || x.passes === 1 || x.passes === 2) edit.passes = x.passes;
      if (Object.keys(edit).length) out[e] = edit;
    }
    return out;
  } catch {
    return {};
  }
}

export function loadLevelEdits(): LevelEdits {
  return parseLevelEdits(readPref(LEVELS_KEY));
}

export function saveLevelEdits(e: LevelEdits): void {
  writePref(LEVELS_KEY, Object.keys(e).length ? JSON.stringify(e) : null);
}

export function effectiveLevels(edits: LevelEdits): Record<Effort, EffortLevel> {
  const out = {} as Record<Effort, EffortLevel>;
  for (const e of EFFORTS) out[e] = { ...EFFORT_LEVELS[e], ...(edits[e] ?? {}) };
  return out;
}
