import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { GLOSSARY } from "../../ui/src/lib/glossary";
import { MOCK_LIBRARY } from "../../ui/src/mock/library";
import { NGRAM_LABELS } from "../../ui/src/screens/models/LoadPanel";

/**
 * docs/CLAIMS.md is binding on every word the app shows. These are the overclaims it names
 * explicitly; none may appear in the glossary or anywhere in the UI source.
 */
const BANNED: readonly RegExp[] = [
  // Speculative decoding is distribution-preserving, never "the same output" (CLAIMS.md "Correctness
  // overclaims": a batched verify pass can flip a near-tie argmax).
  /same tokens/i,
  /identical/i,
  /bit-exact/i,
  /never changes which tokens/i,
  // A speedup ratio without its conditions is invalid (rule zero).
  /\d(?:\.\d+)?\s*×\s*defaults/i,
  /×\s*defaults/i,
  /\bx\s*faster\b|\d(?:\.\d+)?×\s*faster/i,
];

const SRC = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../ui/src");

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) out.push(...sourceFiles(full));
    else if (/\.(tsx?|css|html)$/.test(name)) out.push(full);
  }
  return out;
}

describe("UI copy follows docs/CLAIMS.md", () => {
  it("the glossary never overclaims speculative decoding or speedups", () => {
    for (const [key, entry] of Object.entries(GLOSSARY)) {
      for (const re of BANNED) expect(`${entry.term} ${entry.text}`, `${key} matches ${re}`).not.toMatch(re);
    }
    expect(GLOSSARY.speculative.text).toMatch(/distribution-preserving/);
  });

  it("no UI source file contains a banned phrase", () => {
    const files = sourceFiles(SRC);
    expect(files.length).toBeGreaterThan(20);
    for (const file of files) {
      const text = readFileSync(file, "utf8");
      for (const re of BANNED) expect(text, `${path.relative(SRC, file)} matches ${re}`).not.toMatch(re);
    }
  });

  it("every draft-free strategy has a plain-words label", () => {
    for (const [id, label] of Object.entries(NGRAM_LABELS)) {
      expect(label, id).not.toBe(id);
      expect(label.length).toBeGreaterThan(5);
    }
  });

  it("the library only carries speeds as three numbers with their conditions", () => {
    for (const m of MOCK_LIBRARY) {
      expect(Object.hasOwn(m, "tunedTps"), m.id).toBe(false);
      expect(Object.hasOwn(m, "defaultsTps"), m.id).toBe(false);
      const r = m.tuneResult;
      if (!r) continue;
      expect(r.prefillTps, m.id).toBeGreaterThan(0);
      expect(r.decodeTps0, m.id).toBeGreaterThan(0);
      expect(r.decodeTpsDepth, m.id).toBeGreaterThan(0);
      // Writing slows as the context fills; a depth number above the empty-context one is a bug.
      expect(r.decodeTpsDepth, m.id).toBeLessThanOrEqual(r.decodeTps0);
      expect(r.depth, m.id).toBeGreaterThan(0);
      for (const condition of [r.build, r.gpu, r.sampler, r.measuredAt]) expect(condition.length, m.id).toBeGreaterThan(0);
      expect(r.provenance).toBe("illustrative");
    }
  });
});
