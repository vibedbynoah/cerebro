import type { GgufEdit, GgufEntry } from "../service/types";

/**
 * GGUF metadata editing rules, shared by the editor and every service. A value is checked against
 * the key's stored type; structural keys (marked not editable) are refused, because a changed layer
 * count or tensor layout would make the file unreadable. Edits are only ever written to a copy.
 */

export const GGUF_STRING_MAX = 65_536;

/** Parses the editor's text for `entry` into a typed value, or returns the reason it cannot. */
export function parseGgufValue(entry: Pick<GgufEntry, "type" | "key">, text: string): { ok: true; value: string | number | boolean } | { ok: false; error: string } {
  switch (entry.type) {
    case "bool":
      if (text === "true" || text === "false") return { ok: true, value: text === "true" };
      return { ok: false, error: "true or false." };
    case "uint32":
    case "int32": {
      const t = text.trim();
      if (!/^-?\d{1,10}$/.test(t)) return { ok: false, error: "A whole number." };
      const n = Number(t);
      const [lo, hi] = entry.type === "uint32" ? [0, 4_294_967_295] : [-2_147_483_648, 2_147_483_647];
      if (n < lo || n > hi) return { ok: false, error: `Between ${lo.toLocaleString("en-US")} and ${hi.toLocaleString("en-US")}.` };
      if (/context_length$/.test(entry.key) && n < 1) return { ok: false, error: "At least 1." };
      return { ok: true, value: n };
    }
    case "float32": {
      const t = text.trim();
      if (!/^-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(t)) return { ok: false, error: "A number, such as 1000000 or 1e-6." };
      const n = Number(t);
      if (!Number.isFinite(n) || Math.abs(n) > 3.4e38) return { ok: false, error: "Out of range for a 32-bit float." };
      return { ok: true, value: n };
    }
    case "string":
      if (text.length > GGUF_STRING_MAX) return { ok: false, error: `At most ${GGUF_STRING_MAX.toLocaleString("en-US")} characters.` };
      if (text.includes("\u0000")) return { ok: false, error: "Null characters are not allowed." };
      return { ok: true, value: text };
    case "array":
      return { ok: false, error: "Arrays are read-only here." };
  }
}

/** Checks a full edit list against the file's entries. Returns the first problem, or null. */
export function validateGgufEdits(edits: unknown, entries: readonly GgufEntry[]): string | null {
  if (!Array.isArray(edits) || edits.length === 0) return "Nothing to write.";
  if (edits.length > 64) return "Too many edits at once.";
  const byKey = new Map(entries.map((e) => [e.key, e]));
  const seen = new Set<string>();
  for (const raw of edits as unknown[]) {
    if (typeof raw !== "object" || raw === null) return "An edit is malformed.";
    const { key, value } = raw as Partial<GgufEdit>;
    if (typeof key !== "string" || seen.has(key)) return "An edit is malformed or repeated.";
    seen.add(key);
    const entry = byKey.get(key);
    if (!entry) return `The file has no key ${key}.`;
    if (!entry.editable) return `${key} is structural and cannot be changed.`;
    const parsed = parseGgufValue(entry, typeof value === "string" ? value : String(value));
    if (!parsed.ok) return `${key}: ${parsed.error}`;
    if (typeof value !== typeof parsed.value || (typeof value === "number" && value !== parsed.value)) return `${key}: wrong type.`;
  }
  return null;
}

/** Display text of a value, as the editor shows and edits it. */
export function ggufText(value: string | number | boolean): string {
  return typeof value === "string" ? value : String(value);
}
