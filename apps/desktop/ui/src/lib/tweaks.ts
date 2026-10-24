import type { LibraryModel, LoadParams, PatchProfile, RopeScaling, Sampler, Tweaks } from "../service/types";
import { KV_TYPES, LIMITS } from "./load-params";

/**
 * Tweak Studio rules: what a tweak may hold, a light check of chat templates, a line diff for the
 * template view, and the import/export format of shareable patch profiles. Imported profiles are
 * untrusted text: they are parsed with JSON.parse, size-capped, and rebuilt field by field from
 * validated values, so no unexpected key (including `__proto__`) ever reaches app state.
 */

export const TWEAK_LIMITS = {
  templateMaxChars: 32_768,
  stopTokenMaxChars: 32,
  stopTokensMax: 8,
  profileNameMax: 40,
  profileMaxBytes: 65_536,
  contextMin: 256,
  contextMax: 1_048_576,
  ropeFactorMax: 16,
} as const;

export const ROPE_SCALINGS: readonly RopeScaling[] = ["none", "linear", "yarn"];

export const EMPTY_TWEAKS: Tweaks = { kvQuant: null, contextOverride: null, rope: null, chatTemplate: null, stopTokens: [], sampler: null };

const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const int = (n: unknown): n is number => typeof n === "number" && Number.isInteger(n);

export const SAMPLER_LIMITS: Readonly<Record<keyof Sampler, { min: number; max: number; step: number; int?: boolean }>> = {
  temperature: { min: 0, max: 2, step: 0.05 },
  topK: { min: 0, max: 200, step: 1, int: true },
  topP: { min: 0, max: 1, step: 0.01 },
  minP: { min: 0, max: 1, step: 0.01 },
  repeatPenalty: { min: 1, max: 2, step: 0.01 },
};

/** Returns the reason a stop token is not acceptable, or null. Control characters other than a
 *  newline are refused: they cannot be typed and only arrive by paste. */
export function stopTokenError(token: string, existing: readonly string[]): string | null {
  if (token.length === 0) return "Type the text that should end a reply.";
  if (token.length > TWEAK_LIMITS.stopTokenMaxChars) return `At most ${TWEAK_LIMITS.stopTokenMaxChars} characters.`;
  if (/[\u0000-\u0009\u000b-\u001f\u007f]/.test(token)) return "Control characters are not allowed.";
  if (existing.includes(token)) return "Already in the list.";
  if (existing.length >= TWEAK_LIMITS.stopTokensMax) return `At most ${TWEAK_LIMITS.stopTokensMax} stop tokens.`;
  return null;
}

/**
 * A light structural check of a Jinja chat template: balanced `{{ }}` and `{% %}` delimiters and
 * matched block tags. It does not run the template (nothing in the UI ever does).
 */
export function checkTemplate(text: string): string[] {
  const issues: string[] = [];
  if (text.trim().length === 0) return ["The template is empty."];
  if (text.length > TWEAK_LIMITS.templateMaxChars) issues.push(`Longer than ${TWEAK_LIMITS.templateMaxChars.toLocaleString("en-US")} characters.`);
  const count = (re: RegExp) => text.match(re)?.length ?? 0;
  if (count(/\{\{/g) !== count(/\}\}/g)) issues.push("Unbalanced {{ }} delimiters.");
  if (count(/\{%/g) !== count(/%\}/g)) issues.push("Unbalanced {% %} delimiters.");
  const stack: string[] = [];
  for (const m of text.matchAll(/\{%-?\s*(\w+)/g)) {
    const tag = m[1] ?? "";
    if (tag === "if" || tag === "for" || tag === "macro" || tag === "raw" || tag === "generation") stack.push(tag);
    else if (tag.startsWith("end")) {
      const open = stack.pop();
      if (open !== tag.slice(3)) {
        issues.push(open ? `{% ${tag} %} closes a {% ${open} %} block.` : `{% ${tag} %} has no opening tag.`);
        break;
      }
    }
  }
  if (issues.length === 0 && stack.length > 0) issues.push(`{% ${stack[stack.length - 1]} %} is never closed.`);
  return issues;
}

export type DiffLine = { kind: "same" | "add" | "del"; text: string };

/** Line diff by longest common subsequence. Inputs above `maxLines` lines return null (too large
 *  to diff in the UI thread without a visible pause). */
export function diffLines(before: string, after: string, maxLines = 600): DiffLine[] | null {
  const a = before.split("\n");
  const b = after.split("\n");
  if (a.length > maxLines || b.length > maxLines) return null;
  const n = a.length;
  const m = b.length;
  const lcs: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    const row = lcs[i]!;
    const below = lcs[i + 1]!;
    for (let j = m - 1; j >= 0; j--) row[j] = a[i] === b[j] ? below[j + 1]! + 1 : Math.max(below[j]!, row[j + 1]!);
  }
  const out: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ kind: "same", text: a[i]! });
      i++;
      j++;
    } else if (lcs[i + 1]![j]! >= lcs[i]![j + 1]!) out.push({ kind: "del", text: a[i++]! });
    else out.push({ kind: "add", text: b[j++]! });
  }
  while (i < n) out.push({ kind: "del", text: a[i++]! });
  while (j < m) out.push({ kind: "add", text: b[j++]! });
  return out;
}

/** Validates and rebuilds a Tweaks object from untrusted input. Returns null when anything is off. */
export function sanitizeTweaks(input: unknown): Tweaks | null {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return null;
  const t = input as Record<string, unknown>;
  const kv = t["kvQuant"];
  if (kv !== null && !(typeof kv === "string" && (KV_TYPES as readonly string[]).includes(kv))) return null;
  const ctx = t["contextOverride"];
  if (ctx !== null && !(int(ctx) && ctx >= TWEAK_LIMITS.contextMin && ctx <= TWEAK_LIMITS.contextMax)) return null;

  let rope: Tweaks["rope"] = null;
  if (t["rope"] !== null) {
    const r = t["rope"];
    if (typeof r !== "object" || r === null || Array.isArray(r)) return null;
    const rr = r as Record<string, unknown>;
    const scaling = rr["scaling"];
    if (!(typeof scaling === "string" && (ROPE_SCALINGS as readonly string[]).includes(scaling))) return null;
    if (!(finite(rr["factor"]) && rr["factor"] >= 1 && rr["factor"] <= TWEAK_LIMITS.ropeFactorMax)) return null;
    if (!(int(rr["originalContext"]) && rr["originalContext"] >= TWEAK_LIMITS.contextMin && rr["originalContext"] <= TWEAK_LIMITS.contextMax)) return null;
    rope = { scaling: scaling as RopeScaling, factor: rr["factor"], originalContext: rr["originalContext"] };
  }

  const tpl = t["chatTemplate"];
  if (tpl !== null && !(typeof tpl === "string" && tpl.length > 0 && tpl.length <= TWEAK_LIMITS.templateMaxChars)) return null;

  const stops = t["stopTokens"];
  if (!Array.isArray(stops) || stops.length > TWEAK_LIMITS.stopTokensMax) return null;
  const cleanStops: string[] = [];
  for (const s of stops) {
    if (typeof s !== "string" || stopTokenError(s, cleanStops) !== null) return null;
    cleanStops.push(s);
  }

  let sampler: Sampler | null = null;
  if (t["sampler"] !== null) {
    const s = t["sampler"];
    if (typeof s !== "object" || s === null || Array.isArray(s)) return null;
    const ss = s as Record<string, unknown>;
    const out: Partial<Sampler> = {};
    for (const key of Object.keys(SAMPLER_LIMITS) as (keyof Sampler)[]) {
      const v = ss[key];
      const lim = SAMPLER_LIMITS[key];
      if (!finite(v) || v < lim.min || v > lim.max || (lim.int && !int(v))) return null;
      out[key] = v;
    }
    sampler = out as Sampler;
  }
  return { kvQuant: kv as Tweaks["kvQuant"], contextOverride: ctx as number | null, rope, chatTemplate: tpl as string | null, stopTokens: cleanStops, sampler };
}

export function profileNameError(name: string, taken: readonly string[]): string | null {
  const trimmed = name.trim();
  if (trimmed.length === 0) return "Give it a name.";
  if (trimmed.length > TWEAK_LIMITS.profileNameMax) return `At most ${TWEAK_LIMITS.profileNameMax} characters.`;
  if (/[\u0000-\u001f\u007f]/.test(trimmed)) return "Control characters are not allowed.";
  if (taken.some((t) => t.toLowerCase() === trimmed.toLowerCase())) return "That name is taken.";
  return null;
}

const HEX64 = /^[0-9a-f]{64}$/;
const FIX_ID = /^[a-z0-9-]{1,48}$/;

/** The text a user copies to share a profile. */
export function exportProfile(p: Omit<PatchProfile, "id">): string {
  return JSON.stringify({ format: "cerebro-patch-profile", version: 1, name: p.name, modelSha256: p.modelSha256, tweaks: p.tweaks, fixes: p.fixes }, null, 2);
}

export type ImportResult = { ok: true; profile: Omit<PatchProfile, "id"> } | { ok: false; error: string };

/** Parses shared profile text. Never evaluates it; rebuilds only known, valid fields. */
export function importProfile(text: string): ImportResult {
  if (new TextEncoder().encode(text).length > TWEAK_LIMITS.profileMaxBytes) return { ok: false, error: "That text is too large to be a patch profile." };
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { ok: false, error: "That is not a patch profile (it is not valid JSON)." };
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) return { ok: false, error: "That is not a patch profile." };
  const d = data as Record<string, unknown>;
  if (d["format"] !== "cerebro-patch-profile" || d["version"] !== 1) return { ok: false, error: "That is not a Cerebro patch profile, or it is from a newer version." };
  const name = typeof d["name"] === "string" ? d["name"].trim() : "";
  if (profileNameError(name, []) !== null) return { ok: false, error: "The profile's name is missing or too long." };
  const sha = d["modelSha256"];
  if (typeof sha !== "string" || !HEX64.test(sha)) return { ok: false, error: "The profile does not name a model file." };
  const tweaks = sanitizeTweaks(d["tweaks"]);
  if (!tweaks) return { ok: false, error: "The profile's tweaks are not valid." };
  const fixes = d["fixes"];
  if (!Array.isArray(fixes) || fixes.length > 32 || !fixes.every((f): f is string => typeof f === "string" && FIX_ID.test(f)))
    return { ok: false, error: "The profile's fixes are not valid." };
  return { ok: true, profile: { name, modelSha256: sha, tweaks, fixes: [...new Set(fixes)] } };
}

/** How many tweaks differ from "as the file says". */
export function activeTweakCount(t: Tweaks): number {
  return (t.kvQuant ? 1 : 0) + (t.contextOverride !== null ? 1 : 0) + (t.rope ? 1 : 0) + (t.chatTemplate !== null ? 1 : 0) + (t.stopTokens.length > 0 ? 1 : 0) + (t.sampler ? 1 : 0);
}

/**
 * The one rule for what a load actually uses: the load panel's settings (tuned, starting or edited)
 * with the active tweaks laid over them. Tweaks win for the fields they set (KV cache type for both
 * K and V, context length); everything else comes from `p`. Returns a new object; `p` is untouched,
 * so "Reset to tuned" still restores the measured values.
 */
export function applyTweaks(p: LoadParams, t: Tweaks): LoadParams {
  return {
    ...p,
    kvCacheK: t.kvQuant ?? p.kvCacheK,
    kvCacheV: t.kvQuant ?? p.kvCacheV,
    contextLength: t.contextOverride ?? p.contextLength,
  };
}

/** The load fields a tweak currently decides, so the load panel can show them as set elsewhere. */
export function tweakedFields(t: Tweaks): ReadonlySet<keyof LoadParams> {
  const out = new Set<keyof LoadParams>();
  if (t.kvQuant) (out.add("kvCacheK"), out.add("kvCacheV"));
  if (t.contextOverride !== null) out.add("contextLength");
  return out;
}

/**
 * The longest context a load may ask for: the file's trained maximum, or, with RoPE scaling
 * switched on, the stretched length (original context x factor). Never above the global cap.
 */
export function contextCeiling(m: Pick<LibraryModel, "contextMax" | "contextExtended">, t: Pick<Tweaks, "rope">): number {
  const stretched = t.rope && t.rope.scaling !== "none" ? Math.round(t.rope.originalContext * t.rope.factor) : 0;
  // The desktop engine stretches any model with YaRN past its trained length (contextExtended).
  return Math.min(LIMITS.contextCap, Math.max(m.contextMax, stretched, m.contextExtended ?? 0));
}
