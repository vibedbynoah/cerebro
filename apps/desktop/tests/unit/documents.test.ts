import { describe, expect, it } from "vitest";
import { checkFile, makeDoc, MAX_CHARS, PASSAGE_CHARS, parseDocs, passages, search } from "../../ui/src/lib/documents";

describe("documents index", () => {
  it("splits on paragraphs and sentences, never past the passage size", () => {
    const text = Array.from({ length: 40 }, (_, i) => `Sentence number ${i} talks about topic ${i % 7}.`).join(" ") + "\n\n" + "x".repeat(PASSAGE_CHARS * 2 + 10);
    const ps = passages(text);
    expect(ps.length).toBeGreaterThan(3);
    for (const p of ps) expect(p.length).toBeLessThanOrEqual(PASSAGE_CHARS);
    expect(ps.join("").replace(/\s/g, "")).toBe(text.replace(/\s/g, ""));
  });

  it("counts the passages it really made and marks long files as truncated", () => {
    const d = makeDoc("1", "a.txt", "text/plain", 10, "One. Two.", 0);
    expect(d.chunks).toBe(1);
    expect(d.truncated).toBe(false);
    const big = makeDoc("2", "b.txt", "text/plain", 10, "word ".repeat(MAX_CHARS), 0);
    expect(big.truncated).toBe(true);
    expect(big.text.length).toBe(MAX_CHARS);
  });

  it("refuses formats it cannot read, with the reason", () => {
    expect(checkFile("report.pdf", 100)).toMatch(/PDF and Office/);
    expect(checkFile("tool.exe", 100)).toMatch(/Supported/);
    expect(checkFile("notes.md", 100)).toBeNull();
    expect(checkFile("empty.txt", 0)).toMatch(/under 20 MB/);
  });

  it("ranks the passage that answers first, across files, matching word forms", () => {
    const docs = [
      makeDoc("a", "ops.txt", "", 1, "Backups run every night at 2am. The router is in the hall.", 0),
      makeDoc("b", "misc.txt", "", 1, "The office has a backup generator. Lunch is at noon.", 0),
      makeDoc("c", "other.txt", "", 1, "Nothing relevant here at all.", 0),
    ];
    const hits = search(docs, "when do backups run");
    expect(hits.map((h) => h.doc.name)).toEqual(["ops.txt", "misc.txt"]);
    expect(hits[0]?.matched).toBe(2);
    expect(hits[0]?.parts.filter((p) => p.hit).map((p) => p.v.toLowerCase())).toEqual(["backups", "run"]);
  });

  it("the desktop app starts empty; the preview shows examples", () => {
    expect(parseDocs(null, false)).toEqual([]);
    expect(parseDocs(null, true).length).toBe(3);
  });
});
