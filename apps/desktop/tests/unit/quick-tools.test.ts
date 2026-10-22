import { describe, expect, it } from "vitest";
import { buildQuickToolPrompt, checkJson, fieldList, QUICK_TOOLS, quickTool, QUICK_TOOL_MARKER, REWRITE_STYLES, TRANSLATE_LANGUAGES, type QuickToolId } from "../../ui/src/lib/quick-tools";

describe("quick tools", () => {
  it("has one card and one command for each tool", () => {
    expect(QUICK_TOOLS.length).toBeGreaterThanOrEqual(9);
    const ids = QUICK_TOOLS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const t of QUICK_TOOLS) {
      expect(quickTool(t.id)).toBe(t);
      expect(t.command).toBe(t.id);
      expect(t.blurb.length).toBeGreaterThan(5);
      expect(t.inputLabel.length).toBeGreaterThan(3);
    }
  });

  it("tells the model what the job is, and what to do with the text", () => {
    const tool = quickTool("summarise")!;
    const prompt = buildQuickToolPrompt(tool.id, { text: "A long document about backups." });
    expect(prompt.system).toContain(`${QUICK_TOOL_MARKER} summarise`);
    expect(prompt.system).toMatch(/summary/i);
    expect(prompt.user).toBe("A long document about backups.");
  });

  it("names the language to translate into", () => {
    const prompt = buildQuickToolPrompt("translate", { text: "Hello", language: "Spanish" });
    expect(prompt.system).toContain("Spanish");
    expect(prompt.system).toMatch(/translation only/i);
  });

  it("names the style to rewrite in", () => {
    const prompt = buildQuickToolPrompt("rewrite", { text: "Hello", style: "shorter" });
    expect(prompt.system).toContain("shorter");
    expect(REWRITE_STYLES.map((s) => s.id)).toEqual(["shorter", "clearer", "formal", "friendly"]);
  });

  it("puts the fields to extract in the request", () => {
    const prompt = buildQuickToolPrompt("extract", { text: "Ada is the engineer. Grace is the analyst.", fields: "name, role" });
    expect(prompt.system).toMatch(/JSON only/i);
    expect(prompt.user).toContain("Fields: name, role");
    expect(prompt.user).toContain("Ada is the engineer.");
    expect(fieldList("name, role, name")).toEqual(["name", "role"]);
    expect(fieldList(undefined)).toEqual([]);
  });

  it("sends both texts when comparing", () => {
    const prompt = buildQuickToolPrompt("compare", { text: "First text", second: "Second text" });
    expect(prompt.user).toContain("First text");
    expect(prompt.user).toContain("Second text");
    expect(prompt.system).toMatch(/agree|differ|changed/i);
  });

  it("asks for a summary, decisions and actions from a transcript", () => {
    const prompt = buildQuickToolPrompt("notes", { text: "We agreed to ship on Friday." });
    expect(prompt.system).toMatch(/summary/i);
    expect(prompt.system).toMatch(/decisions/i);
    expect(prompt.system).toMatch(/actions/i);
  });

  it("asks for a description when the tool has an image", () => {
    const prompt = buildQuickToolPrompt("describe", { text: "" });
    expect(prompt.system).toContain(`${QUICK_TOOL_MARKER} describe`);
    expect(prompt.system).toMatch(/describe the image/i);
  });

  it("sends nothing to a model when the tool only reads aloud", () => {
    const prompt = buildQuickToolPrompt("read", { text: "Hello there" });
    expect(prompt.system).toBe("");
    expect(prompt.user).toBe("Hello there");
  });

  it("every tool's instruction says which tool it is", () => {
    const tools: QuickToolId[] = ["summarise", "translate", "rewrite", "extract", "explain", "compare", "notes", "describe"];
    for (const id of tools) {
      const prompt = buildQuickToolPrompt(id, { text: "x", second: "y", fields: "a" });
      expect(prompt.system.startsWith(`${QUICK_TOOL_MARKER} ${id}`)).toBe(true);
    }
  });

  it("offers languages to translate into", () => {
    expect(TRANSLATE_LANGUAGES).toContain("Spanish");
    expect(TRANSLATE_LANGUAGES).toContain("Japanese");
    expect(TRANSLATE_LANGUAGES.length).toBeGreaterThan(10);
  });
});

describe("checking an answer that should be JSON", () => {
  it("reads JSON out of the reply, even in a code fence", () => {
    expect(checkJson('[{"name":"Ada"}]')).toMatchObject({ ok: true });
    expect(checkJson('```json\n[{"name":"Ada"}]\n```')).toMatchObject({ ok: true });
    expect(checkJson('{"name":"Ada"}')).toMatchObject({ ok: true });
  });

  it("says what is wrong when the answer is not JSON", () => {
    const bad = checkJson("Sorry, I could not do that.");
    expect(bad.ok).toBe(false);
    expect(bad.error).toMatch(/not valid JSON/i);
    expect(checkJson("").error).toMatch(/no JSON/i);
    expect(checkJson("42").error).toMatch(/not an object or a list/i);
  });
});
