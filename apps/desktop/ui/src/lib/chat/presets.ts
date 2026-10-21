import type { Sampler } from "../../service/chat-types";
import { readPref, writePref } from "../storage";
import type { ChatPrefs } from "./prompt";
import { parseSampler } from "./modes";

/**
 * Per-model presets: a model's own chat settings (the slider's level comes with them), applied
 * when that model is picked in Chat. Saved from the live panel; nothing is saved until the user
 * asks. Stored as one JSON object keyed by model id; anything unreadable is dropped.
 */

export const PRESETS_KEY = "cerebro-model-presets";

export interface ModelPreset {
  prefs: Pick<ChatPrefs, "effort" | "verbosity" | "source" | "addInstructions" | "base" | "memoryOn" | "orchestrate" | "workers" | "rounds">;
  sampler: Sampler;
  savedAt: number;
}

const EFFORTS = ["low", "medium", "high", "xhigh", "max"];
const VERBOSITIES = ["brief", "normal", "detailed"];
const SOURCES = ["native", "prompts", "both", "none"];

function valid(v: unknown): ModelPreset | null {
  if (!v || typeof v !== "object") return null;
  const o = v as Record<string, unknown>;
  const p = o.prefs as Record<string, unknown> | undefined;
  if (!p || typeof p !== "object") return null;
  if (!EFFORTS.includes(p.effort as string) || !VERBOSITIES.includes(p.verbosity as string) || !SOURCES.includes(p.source as string)) return null;
  if (typeof p.addInstructions !== "boolean" || typeof p.base !== "string" || p.base.length > 8000 || typeof p.orchestrate !== "boolean" || typeof p.memoryOn !== "boolean") return null;
  const int = (x: unknown, lo: number, hi: number) => (typeof x === "number" && Number.isInteger(x) && x >= lo && x <= hi ? x : null);
  const workers = int(p.workers, 1, 6);
  const rounds = int(p.rounds, 1, 8);
  if (workers === null || rounds === null) return null;
  return {
    prefs: { effort: p.effort as ChatPrefs["effort"], verbosity: p.verbosity as ChatPrefs["verbosity"], source: p.source as ChatPrefs["source"], addInstructions: p.addInstructions, base: p.base, memoryOn: p.memoryOn, orchestrate: p.orchestrate, workers, rounds },
    sampler: parseSampler(JSON.stringify(o.sampler ?? {})),
    savedAt: typeof o.savedAt === "number" ? o.savedAt : 0,
  };
}

export function loadPresets(): Record<string, ModelPreset> {
  const out: Record<string, ModelPreset> = {};
  try {
    const raw: unknown = JSON.parse(readPref(PRESETS_KEY) ?? "{}");
    if (raw && typeof raw === "object") {
      for (const [id, v] of Object.entries(raw as Record<string, unknown>)) {
        const p = valid(v);
        if (p && id.length <= 300) out[id] = p;
      }
    }
  } catch {
    // An unreadable store is the same as none.
  }
  return out;
}

export function savePreset(modelId: string, prefs: ChatPrefs, sampler: Sampler): void {
  const all = loadPresets();
  const { effort, verbosity, source, addInstructions, base, memoryOn, orchestrate, workers, rounds } = prefs;
  all[modelId] = { prefs: { effort, verbosity, source, addInstructions, base, memoryOn, orchestrate, workers, rounds }, sampler, savedAt: Date.now() };
  writePref(PRESETS_KEY, JSON.stringify(all));
}

export function removePreset(modelId: string): void {
  const all = loadPresets();
  delete all[modelId];
  writePref(PRESETS_KEY, JSON.stringify(all));
}
