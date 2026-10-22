import { describe, expect, it } from "vitest";
import { makeDoc, type Doc } from "../../ui/src/lib/documents";
import { assembleContext, approxTokens, ragPrompt, retrieve } from "../../ui/src/lib/rag";

const docs: Doc[] = [
  makeDoc("a", "handbook.txt", "text/plain", 10, "Remote work is allowed up to three days a week with manager approval. Expense claims must be filed within 30 days with a receipt.", 0),
  makeDoc("b", "runbook.txt", "text/plain", 10, "To restart the inference server, stop the service, clear the cache folder and start it again. Backups run every night at 2am.", 0),
  makeDoc("c", "pricing.md", "text/markdown", 10, "The starter plan is 19 euros a month for up to three seats. Annual billing gives two months free.", 0),
];

describe("retrieval", () => {
  it("finds the passage that answers, and leaves the rest out", () => {
    const found = retrieve(docs, "when do backups run");
    expect(found.length).toBeGreaterThan(0);
    expect(found[0]?.docId).toBe("b");
    expect(found[0]?.text).toMatch(/Backups run every night/);
    expect(found.some((p) => p.docId === "c")).toBe(false);
  });

  it("gives one file no more than a couple of passages", () => {
    const many = [makeDoc("long", "long.txt", "text/plain", 10, Array.from({ length: 40 }, (_, i) => `Sentence ${i} about backups and the server.`).join(" "), 0)];
    const found = retrieve(many, "backups server", { limit: 10, perDoc: 2 });
    expect(found.length).toBe(2);
  });

  it("finds nothing when the words are not in the index", () => {
    expect(retrieve(docs, "zzzz")).toEqual([]);
    expect(retrieve([], "backups")).toEqual([]);
  });
});

describe("the context that goes into the request", () => {
  it("numbers the passages and names the file each came from", () => {
    const found = retrieve(docs, "plan and days");
    const context = assembleContext(found);
    expect(context.sources.length).toBeGreaterThanOrEqual(2);
    expect(context.sources[0]?.n).toBe(1);
    expect(context.sources.map((s) => s.n)).toEqual(context.sources.map((_, i) => i + 1));
    expect(context.text).toContain("[1] ");
    expect(context.text).toContain("[2] ");
    expect(context.text).toMatch(/\[1\] (handbook|pricing|runbook)\.txt/);
  });

  it("keeps the same passage only once", () => {
    const passage = { docId: "a", docName: "handbook.txt", text: "Expenses must be filed within 30 days.", index: 0, score: 1 };
    const context = assembleContext([passage, { ...passage }, { ...passage, docId: "b", docName: "other.txt" }]);
    expect(context.sources.length).toBe(2);
    expect(context.sources.map((s) => s.docId)).toEqual(["a", "b"]);
  });

  it("stays inside the token budget", () => {
    const found = Array.from({ length: 30 }, (_, i) => ({ docId: `d${i}`, docName: `file${i}.txt`, text: `Passage ${i}: ${"word ".repeat(60)}`, index: 0, score: 1 }));
    const context = assembleContext(found, 200);
    expect(context.tokens).toBeLessThanOrEqual(200);
    expect(context.sources.length).toBeGreaterThan(0);
    expect(context.sources.length).toBeLessThan(found.length);
    expect(approxTokens(context.text)).toBeLessThanOrEqual(context.tokens + 40);
  });

  it("always keeps one passage, even when it is longer than the budget", () => {
    const context = assembleContext([{ docId: "a", docName: "a.txt", text: "word ".repeat(500), index: 0, score: 1 }], 10);
    expect(context.sources.length).toBe(1);
  });

  it("tells the model how to quote the passages", () => {
    const context = assembleContext(retrieve(docs, "when do backups run"));
    const prompt = ragPrompt(context);
    expect(prompt).toContain("[1]");
    expect(prompt).toMatch(/quote the passage number/i);
    expect(ragPrompt(assembleContext([]))).toBe("");
  });
});
