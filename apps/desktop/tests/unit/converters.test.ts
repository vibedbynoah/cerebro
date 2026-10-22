import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { deflateSync } from "node:zlib";
import { convertBytes, convertText, canConvert } from "../../ui/src/lib/convert/index";
import { inflate, inflateRaw } from "../../ui/src/lib/convert/inflate";
import { readZip } from "../../ui/src/lib/convert/zip";

/** The checked-in fixture files, found from wherever the tests were started. */
const fixtures = (): string => {
  let dir = process.cwd();
  for (let i = 0; i < 6; i++) {
    const candidate = path.join(dir, "tests", "fixtures");
    if (existsSync(candidate)) return candidate;
    dir = path.dirname(dir);
  }
  throw new Error("The fixtures folder was not found.");
};
const fixture = (name: string): Uint8Array => new Uint8Array(readFileSync(path.join(fixtures(), name)));
const fixtureText = (name: string): string => readFileSync(path.join(fixtures(), name), "utf8");

describe("converters", () => {
  it("reads the formats it says it can, and refuses the others", () => {
    expect(canConvert("report.pdf")).toBe(true);
    expect(canConvert("plan.docx")).toBe(true);
    expect(canConvert("notes.md")).toBe(true);
    expect(canConvert("data.csv")).toBe(true);
    expect(canConvert("plan.json")).toBe(true);
    expect(canConvert("page.html")).toBe(true);
    expect(canConvert("notes.txt")).toBe(true);
    expect(canConvert("tool.exe")).toBe(false);
  });

  it("turns plain text into itself", () => {
    const out = convertText("handbook.txt", fixtureText("handbook.txt"));
    expect(out.kind).toBe("text");
    expect(out.text).toContain("Annual leave is 25 days");
    expect(out.chars).toBe(out.text.length);
  });

  it("takes the markup out of Markdown but keeps the words", () => {
    const out = convertText("pricing.md", fixtureText("pricing.md"));
    expect(out.text).toContain("The starter plan is 19 euros a month");
    expect(out.text).not.toContain("#");
    expect(out.text).not.toContain("**");
    expect(out.markdown).toContain("# Pricing notes");
  });

  it("reads HTML as text and as Markdown, and leaves out the script", () => {
    const out = convertText("runbook.html", fixtureText("runbook.html"));
    expect(out.text).toContain("Server runbook");
    expect(out.text).toContain("Clear the cache folder");
    expect(out.text).not.toContain("alert(");
    expect(out.markdown).toContain("# Server runbook");
    expect(out.markdown).toContain("- Clear the cache folder");
  });

  it("turns a CSV into rows and a Markdown table", () => {
    const out = convertText("team.csv", fixtureText("team.csv"));
    expect(out.text.split("\n")[0]).toBe("name | role | seats");
    expect(out.text).toContain("Ada | engineer | 3");
    expect(out.markdown).toContain("| name | role | seats |");
    expect(out.markdown).toContain("| --- | --- | --- |");
  });

  it("turns JSON into one value a line and into a fenced block", () => {
    const out = convertText("plan.json", fixtureText("plan.json"));
    expect(out.text).toContain("plan: starter");
    expect(out.text).toContain("seats: 3");
    expect(out.markdown).toContain("```json");
    expect(() => convertText("bad.json", "{not json")).toThrow(/not valid JSON/);
  });

  it("reads the text of a PDF, page by page", () => {
    const out = convertBytes("sample.pdf", fixture("sample.pdf"));
    expect(out.kind).toBe("pdf");
    expect(out.pages).toBe(2);
    expect(out.text).toContain("Cerebro test document");
    expect(out.text).toContain("Annual leave is 25 days plus public holidays.");
    // A kerned run inside one show operation still reads as two words.
    expect(out.text).toContain("Backups run every night at 2am.");
    expect(out.text).toContain("The router is in the hall.");
    expect(out.text).toContain("Backups are checked every Monday morning.");
    expect(out.markdown).toContain("## Page 1");
    expect(out.markdown).toContain("## Page 2");
  });

  it("reads a Word file: headings, lists and tables", () => {
    const out = convertBytes("sample.docx", fixture("sample.docx"));
    expect(out.kind).toBe("docx");
    expect(out.text).toContain("Quarterly plan");
    expect(out.text).toContain("The team ships the new search on Friday.");
    expect(out.text).toContain("Write the release notes");
    expect(out.text).toContain("Task | Owner");
    expect(out.markdown).toContain("# Quarterly plan");
    expect(out.markdown).toContain("- Write the release notes");
    expect(out.markdown).toContain("| Task | Owner |");
    expect(out.markdown).toContain("| Search | Noah |");
  });

  it("says why a file it cannot read failed", () => {
    expect(() => convertBytes("tool.exe", new Uint8Array([1, 2, 3]))).toThrow(/not one Cerebro can read/);
    expect(() => convertBytes("broken.pdf", new Uint8Array([1, 2, 3]))).toThrow(/not a PDF/);
  });
});

describe("compressed data", () => {
  it("inflates a zlib stream back to the bytes it came from", () => {
    const source = Buffer.from("The quick brown fox jumps over the lazy dog. ".repeat(40), "utf8");
    const packed = deflateSync(source);
    expect(Buffer.from(inflate(new Uint8Array(packed))).toString("utf8")).toBe(source.toString("utf8"));
  });

  it("inflates a raw DEFLATE stream, as ZIP entries use", () => {
    const source = Buffer.from("word ".repeat(500), "utf8");
    const packed = deflateSync(source, { level: 9 });
    // A zlib stream without its two-byte header is a raw stream.
    const raw = new Uint8Array(packed.subarray(2));
    expect(Buffer.from(inflateRaw(raw)).toString("utf8")).toBe(source.toString("utf8"));
  });

  it("opens a ZIP archive and reads its entries", () => {
    const zip = readZip(fixture("sample.docx"));
    expect([...zip.keys()]).toContain("word/document.xml");
    expect(new TextDecoder().decode(zip.get("word/document.xml")!.data)).toContain("Quarterly plan");
  });
});
