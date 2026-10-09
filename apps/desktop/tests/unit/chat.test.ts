import { describe, expect, it } from "vitest";
import { highlight } from "../../ui/src/lib/chat/highlight";
import { parseInline, parseMarkdown, plainText, safeHref, type Block, type Inline } from "../../ui/src/lib/chat/markdown";
import { buildRequest, DEFAULT_PREFS, EFFORT_LEVELS, parsePrefs, supportsNativeReasoning, VERBOSITY_INSTRUCTIONS, type PromptInputs } from "../../ui/src/lib/chat/prompt";
import { memoryCandidate, parseConversations, parseMemories, titleFrom } from "../../ui/src/lib/chat/store";
import { createMockChat, tokenize } from "../../ui/src/mock/chat";
import { DEFAULT_SAMPLER, type ChatEvent } from "../../ui/src/service/chat-types";

const base: PromptInputs = {
  base: "You are helpful.",
  effort: "high",
  verbosity: "brief",
  source: "native",
  addInstructions: true,
  memories: [{ id: "a", text: "Uses Rust.", scope: "you" }],
  memoryOn: true,
  nativeSupported: true,
  modelMaxThinking: 32768,
};

describe("effort and reasoning source (effort-levels.md)", () => {
  it("native only sends the model's own control and adds no verification text", () => {
    const r = buildRequest(base);
    expect(r.native).toEqual({ reasoningEffort: "medium", thinkingBudget: 4096 });
    expect(r.systemPrompt).not.toContain(EFFORT_LEVELS.high.instruction);
    expect(r.passes).toBe(0);
  });
  it("Cerebro prompts adds the instruction and passes, no native params", () => {
    const r = buildRequest({ ...base, source: "prompts" });
    expect(r.native).toBeNull();
    expect(r.systemPrompt).toContain(EFFORT_LEVELS.high.instruction);
    expect(r.passes).toBe(1);
  });
  it("both sends native params and the instruction", () => {
    const r = buildRequest({ ...base, source: "both", effort: "max" });
    expect(r.native).toEqual({ reasoningEffort: "high", thinkingBudget: 32768 });
    expect(r.systemPrompt).toContain(EFFORT_LEVELS.max.instruction);
    expect(r.passes).toBe(2);
  });
  it("neither is the raw model plus verbosity and memory only", () => {
    const r = buildRequest({ ...base, source: "none", effort: "max" });
    expect(r.native).toBeNull();
    expect(r.passes).toBe(0);
    expect(r.systemPrompt).not.toContain(EFFORT_LEVELS.max.instruction);
  });
  it("no added prompt sends only the user's own system prompt", () => {
    const r = buildRequest({ ...base, source: "both", addInstructions: false });
    expect(r.systemPrompt).toBe("You are helpful.");
    expect(r.passes).toBe(0);
    expect(r.notes.join(" ")).toMatch(/No added prompt/);
  });
  it("falls back to the instruction when the model has no native control, and says so", () => {
    const r = buildRequest({ ...base, nativeSupported: false });
    expect(r.native).toBeNull();
    expect(r.systemPrompt).toContain(EFFORT_LEVELS.high.instruction);
    expect(r.notes[0]).toMatch(/no native reasoning control/);
  });
  it("adds verbosity and memory, and leaves memory out when off", () => {
    const r = buildRequest(base);
    expect(r.systemPrompt).toContain(VERBOSITY_INSTRUCTIONS.brief);
    expect(r.systemPrompt).toContain("Uses Rust.");
    expect(buildRequest({ ...base, memoryOn: false }).systemPrompt).not.toContain("Uses Rust.");
  });
  it("caps thinking at the model's ceiling and reserves the answer budget", () => {
    const r = buildRequest({ ...base, effort: "xhigh", modelMaxThinking: 8000 });
    expect(r.native?.thinkingBudget).toBe(8000);
    expect(r.answerReserve).toBeGreaterThan(0);
  });
  it("low effort has no thinking and no passes", () => {
    expect(EFFORT_LEVELS.low.thinkingTokens).toBe(0);
    expect(EFFORT_LEVELS.low.passes).toBe(0);
    expect(EFFORT_LEVELS.max.passes).toBe(2);
  });
  it("knows which families have native reasoning", () => {
    expect(supportsNativeReasoning("Qwen3 Qwen3-30B-A3B")).toBe(true);
    expect(supportsNativeReasoning("gpt-oss 120b")).toBe(true);
    expect(supportsNativeReasoning("Mistral Nemo")).toBe(false);
  });
  it("re-checks stored preferences field by field", () => {
    expect(parsePrefs("{bad")).toEqual(DEFAULT_PREFS);
    expect(parsePrefs(JSON.stringify({ effort: "ultra", source: "both", base: 5 }))).toEqual({ ...DEFAULT_PREFS, source: "both" });
  });
});

// ---- Markdown: structure and the XSS corpus ----------------------------------------------------

const kinds = (blocks: Block[]) => blocks.map((b) => b.t);
function allInline(blocks: Block[]): Inline[] {
  const out: Inline[] = [];
  const walk = (l: Inline[]) => l.forEach((x) => (out.push(x), "c" in x && walk(x.c)));
  for (const b of blocks) {
    if (b.t === "p" || b.t === "h") walk(b.c);
    if (b.t === "ul" || b.t === "ol") b.items.forEach(walk);
    if (b.t === "quote") out.push(...allInline(b.c));
  }
  return out;
}

describe("markdown", () => {
  it("parses the supported blocks", () => {
    const md = "# Title\n\nPara **bold** and *em* and `code`.\n\n- a\n- b\n\n1. one\n2. two\n\n> quote\n\n```rust\nfn main() {}\n```\n\n---";
    expect(kinds(parseMarkdown(md))).toEqual(["h", "p", "ul", "ol", "quote", "code", "hr"]);
  });
  it("keeps an unclosed fence open while streaming", () => {
    const b = parseMarkdown("```py\nprint(1)");
    expect(b[0]).toEqual({ t: "code", lang: "py", v: "print(1)", open: true });
  });
  it("leaves snake_case alone", () => {
    expect(parseInline("call my_long_name now")).toEqual([{ t: "text", v: "call my_long_name now" }]);
  });
  it("gives plain text for copy and read-aloud", () => {
    expect(plainText(parseMarkdown("Hi **there**\n\n- a"))).toBe("Hi there\n\n- a");
  });

  const corpus = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    '<a href="javascript:alert(1)">x</a>',
    "[click](javascript:alert(1))",
    "[click](JaVaScRiPt:alert(1))",
    "[click]( javascript:alert(1))",
    "[click](java\tscript:alert(1))",
    "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "[x](vbscript:msgbox(1))",
    "[x](//evil.example/path)",
    '<svg onload="alert(1)">',
    "<iframe src=javascript:alert(1)>",
    '"><img src=x onerror=alert(1)>',
    "**<b onclick=alert(1)>bold</b>**",
    "`<script>`",
    "<style>body{display:none}</style>",
    "[x](https://ok.example/\"onmouseover=\"alert(1))",
  ];
  for (const sample of corpus) {
    it(`renders hostile input as text: ${JSON.stringify(sample).slice(0, 50)}`, () => {
      const tree = parseMarkdown(sample);
      for (const x of allInline(tree)) {
        if (x.t === "link") {
          expect(x.href).toMatch(/^(https?:\/\/|mailto:)/i);
          expect(x.href).not.toMatch(/["'<>\s]/);
        }
      }
      // Every angle bracket survives only inside text or code nodes, never as an element.
      expect(JSON.stringify(tree)).not.toMatch(/"t":"(script|img|svg|iframe|style|a)"/);
    });
  }
  it("allows only http(s) and mailto links", () => {
    expect(safeHref("https://example.com/a?b=c")).toBe("https://example.com/a?b=c");
    expect(safeHref("mailto:a@b.c")).toBe("mailto:a@b.c");
    for (const bad of ["javascript:alert(1)", "data:text/html,x", "/relative", "ftp://x", "https://a b", "https://x\u0000y"]) expect(safeHref(bad)).toBeNull();
  });
});

describe("highlight", () => {
  it("tags keywords, strings, numbers, comments and calls", () => {
    const t = highlight('def f(x):\n    # note\n    return "s" + 1', "python");
    expect(t.find((x) => x.v === "def")?.k).toBe("kw");
    expect(t.find((x) => x.v === '"s"')?.k).toBe("str");
    expect(t.find((x) => x.v === "1")?.k).toBe("num");
    expect(t.find((x) => x.v === "# note")?.k).toBe("com");
    expect(t.find((x) => x.v === "f")?.k).toBe("fn");
    expect(t.map((x) => x.v).join("")).toBe('def f(x):\n    # note\n    return "s" + 1');
  });
  it("passes unknown languages through as one plain token", () => {
    expect(highlight("<b>x</b>", "html")).toEqual([{ k: "plain", v: "<b>x</b>" }]);
  });
});

describe("chat store", () => {
  it("drops malformed or hostile stored conversations", () => {
    expect(parseConversations("not json")).toEqual([]);
    const ok = parseConversations(
      JSON.stringify([
        { id: "c1", title: "T", noMemory: true, updatedAt: 1, messages: [{ id: "m1", role: "user", text: "hi", attachments: [{ name: "a.png", type: "image/png", bytes: 3, preview: "javascript:alert(1)" }] }, { id: "m2", role: "system", text: "x" }] },
        { id: 5 },
      ]),
    );
    expect(ok).toHaveLength(1);
    expect(ok[0]!.messages).toHaveLength(1);
    expect(ok[0]!.messages[0]!.attachments[0]!.preview).toBeNull();
    expect(ok[0]!.noMemory).toBe(true);
  });
  it("parses memories and finds 'remember that' facts", () => {
    expect(parseMemories(JSON.stringify([{ id: "1", text: "x", scope: "project" }, { id: "2", text: "" }]))).toEqual([{ id: "1", text: "x", scope: "project" }]);
    expect(memoryCandidate("Remember that I use Arch Linux.")).toBe("I use Arch Linux.");
    expect(memoryCandidate("what is rust")).toBeNull();
  });
  it("titles a chat from its first message", () => {
    expect(titleFrom("  hello   world ")).toBe("hello world");
    expect(titleFrom("x".repeat(80))).toHaveLength(48);
  });
});

describe("mock chat stream", () => {
  const req = (text: string, extra = {}) => ({
    modelId: "m",
    systemPrompt: "s",
    turns: [{ role: "user" as const, text }],
    sampler: DEFAULT_SAMPLER,
    native: { reasoningEffort: "medium" as const, thinkingBudget: 4096 },
    passes: 1 as const,
    effort: "high" as const,
    verbosity: "brief" as const,
    source: "both" as const,
    ...extra,
  });
  it("streams start, thinking, tokens, a check and done", async () => {
    const events: ChatEvent[] = [];
    await createMockChat({ speed: 1000 }).chat(req("hello"), (e) => events.push(e), new AbortController().signal);
    const types = new Set(events.map((e) => e.type));
    expect([...types]).toEqual(["start", "thinking", "token", "check", "done"]);
    const done = events.at(-1) as Extract<ChatEvent, { type: "done" }>;
    expect(done.stats.stopped).toBe(false);
    expect(done.stats.thinkingTokens).toBeGreaterThan(0);
  });
  it("stops when aborted and says so", async () => {
    const ctl = new AbortController();
    const events: ChatEvent[] = [];
    const p = createMockChat({ speed: 1 }).chat(req("hello"), (e) => (events.push(e), e.type === "token" && ctl.abort()), ctl.signal);
    await p;
    const done = events.at(-1) as Extract<ChatEvent, { type: "done" }>;
    expect(done.type).toBe("done");
    expect(done.stats.stopped).toBe(true);
  });
  it("returns an image for /imagine and echoes for /echo", async () => {
    const ev: ChatEvent[] = [];
    await createMockChat({ speed: 1000 }).chat(req("/imagine a fox", { native: null, passes: 0 }), (e) => ev.push(e), new AbortController().signal);
    expect(ev.some((e) => e.type === "image" && e.prompt === "a fox")).toBe(true);
    const ev2: ChatEvent[] = [];
    await createMockChat({ speed: 1000 }).chat(req("/echo **x**", { native: null, passes: 0 }), (e) => ev2.push(e), new AbortController().signal);
    expect(ev2.filter((e) => e.type === "token").map((e) => (e as { text: string }).text).join("")).toBe("**x**");
  });
  it("rejects a malformed request", async () => {
    await expect(createMockChat().chat({ ...req("x"), turns: [] }, () => {}, new AbortController().signal)).rejects.toThrow();
  });
  it("tokenizes keeping all text", () => {
    expect(tokenize("a b\nc").join("")).toBe("a b\nc");
  });
});
