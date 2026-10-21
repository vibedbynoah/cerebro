import { describe, expect, it } from "vitest";
import {
  activeMessages,
  branchPosition,
  focusBranch,
  parseConversations,
  switchBranch,
  treeFromMessages,
  type ChatMessage,
} from "../../ui/src/lib/chat/store";
import { highlightRuns, searchConversations, snippetAround } from "../../ui/src/lib/chat/search";
import {
  STARTER_PROMPTS,
  exportPrompts,
  fillPromptTemplate,
  mergePrompts,
  parsePromptJson,
  promptCategories,
  promptVariables,
  type SavedPrompt,
} from "../../ui/src/lib/chat/prompts";

function message(id: string, role: "user" | "assistant", text: string): ChatMessage {
  return {
    id,
    role,
    text,
    thinking: "",
    checks: [],
    image: null,
    tools: [],
    ledger: null,
    attachments: [],
    stats: null,
    effort: "medium",
    verbosity: "brief",
    source: "native",
    streaming: false,
    error: null,
  };
}

describe("conversation branches", () => {
  it("migrates a chat saved before branching into a readable single path", () => {
    const raw = JSON.stringify([
      {
        id: "c1",
        title: "Old chat",
        updatedAt: 5,
        messages: [
          { id: "u1", role: "user", text: "hello" },
          { id: "a1", role: "assistant", text: "hi" },
        ],
        pinned: true,
        folder: "  Work  ",
        tags: ["blue", "blue", "  later ", 7],
        archived: true,
      },
    ]);
    const [chat] = parseConversations(raw);
    expect(chat!.messages.map((m) => m.id)).toEqual(["u1", "a1"]);
    expect(chat!.tree.nodes.map((n) => [n.id, n.parentId])).toEqual([
      ["u1", null],
      ["a1", "u1"],
    ]);
    expect(chat!.pinned).toBe(true);
    expect(chat!.folder).toBe("Work");
    expect(chat!.tags).toEqual(["blue", "later"]);
    expect(chat!.archived).toBe(true);
    expect(branchPosition(chat!.tree, "a1")).toEqual({ index: 1, total: 1 });
  });

  it("switches to a sibling and follows that branch's chosen child", () => {
    const u1 = message("u1", "user", "first wording");
    const a1 = message("a1", "assistant", "first reply");
    const u2 = message("u2", "user", "edited wording");
    const a2 = message("a2", "assistant", "second reply");
    const tree = treeFromMessages([u1, a1]);
    tree.nodes.push({ id: "u2", parentId: null, message: u2 }, { id: "a2", parentId: "u2", message: a2 });
    tree.selected[""] = "u2";
    tree.selected["u2"] = "a2";

    expect(activeMessages(tree).map((m) => m.id)).toEqual(["u2", "a2"]);
    expect(branchPosition(tree, "u2")).toEqual({ index: 2, total: 2 });
    const older = switchBranch(tree, "u2", -1);
    expect(activeMessages(older).map((m) => m.id)).toEqual(["u1", "a1"]);
    const newer = switchBranch(older, "u1", 1);
    expect(activeMessages(newer).map((m) => m.id)).toEqual(["u2", "a2"]);
  });

  it("opens the branch that contains a searched message", () => {
    const u1 = message("u1", "user", "old");
    const a1 = message("a1", "assistant", "old answer");
    const u2 = message("u2", "user", "new");
    const a2 = message("a2", "assistant", "new answer");
    const tree = treeFromMessages([u1, a1]);
    tree.nodes.push({ id: "u2", parentId: null, message: u2 }, { id: "a2", parentId: "u2", message: a2 });
    tree.selected[""] = "u2";
    tree.selected["u2"] = "a2";
    const focused = focusBranch(tree, "a1");
    expect(activeMessages(focused).map((m) => m.id)).toEqual(["u1", "a1"]);
    expect(activeMessages(focusBranch(focused, "a2")).map((m) => m.id)).toEqual(["u2", "a2"]);
  });

  it("keeps the newest part of a very long chat when storage is trimmed", () => {
    const messages = Array.from({ length: 510 }, (_, i) => message(`m${i}`, i % 2 === 0 ? "user" : "assistant", `text ${i}`));
    const [chat] = parseConversations(JSON.stringify([{ id: "long", title: "Long chat", updatedAt: 1, messages }]));
    expect(chat!.messages).toHaveLength(500);
    expect(chat!.messages[0]!.id).toBe("m10");
    expect(chat!.messages[chat!.messages.length - 1]!.id).toBe("m509");
  });

  it("keeps a stored tree, and falls back to the message list when the tree is hostile", () => {
    const raw = JSON.stringify([
      {
        id: "c1",
        title: "T",
        updatedAt: 1,
        messages: [{ id: "u1", role: "user", text: "fallback" }],
        tree: {
          nodes: [
            { id: "u1", parentId: null, message: { id: "u1", role: "user", text: "root" } },
            { id: "u2", parentId: "u1", message: { id: "u2", role: "user", text: "edited" } },
          ],
          selected: { "": "u2" },
        },
      },
      {
        id: "c2",
        title: "Cycle",
        updatedAt: 1,
        messages: [{ id: "m1", role: "user", text: "safe" }],
        tree: {
          nodes: [
            { id: "x", parentId: "y", message: { id: "x", role: "user", text: "x" } },
            { id: "y", parentId: "x", message: { id: "y", role: "user", text: "y" } },
          ],
          selected: {},
        },
      },
    ]);
    const [kept, cycle] = parseConversations(raw);
    expect(kept!.messages.map((m) => m.text)).toEqual(["root", "edited"]);
    expect(cycle!.messages.map((m) => m.text)).toEqual(["safe"]);
  });
});

describe("search across chats", () => {
  const list = [
    {
      ...parseConversations(
        JSON.stringify([
          {
            id: "c1",
            title: "Model speed",
            updatedAt: 10,
            messages: [
              { id: "u1", role: "user", text: "Why is the mixture of experts slow?" },
              { id: "a1", role: "assistant", text: "The experts run on the processor." },
            ],
          },
        ]),
      )[0]!,
    },
    {
      ...parseConversations(
        JSON.stringify([
          {
            id: "c2",
            title: "Shopping",
            updatedAt: 20,
            messages: [{ id: "u2", role: "user", text: "Buy milk and bread." }],
          },
        ]),
      )[0]!,
    },
  ];

  it("finds words in titles and messages, newest chat first", () => {
    const byMessage = searchConversations(list, "processor");
    expect(byMessage).toHaveLength(1);
    expect(byMessage[0]).toMatchObject({ conversationId: "c1", messageId: "a1", role: "assistant" });
    const byTitle = searchConversations(list, "shopping");
    expect(byTitle[0]).toMatchObject({ conversationId: "c2", messageId: null, titleMatch: true });
  });

  it("requires every word, ignores case, and marks the hits", () => {
    expect(searchConversations(list, "EXPERTS processor")).toHaveLength(1);
    expect(searchConversations(list, "experts milk")).toHaveLength(0);
    const runs = highlightRuns("Mixture of Experts, experts everywhere", "experts mixture");
    expect(runs.filter((r) => r.hit).map((r) => r.text.toLowerCase())).toEqual(["mixture", "experts", "experts"]);
  });

  it("shows a short snippet around the first hit", () => {
    const text = `${"a ".repeat(80)}needle${" b".repeat(80)}`;
    const snippet = snippetAround(text, "needle", 10, 20);
    expect(snippet).toContain("needle");
    expect(snippet.length).toBeLessThan(text.length);
    expect(snippet.startsWith("…")).toBe(true);
  });
});

describe("prompt library", () => {
  it("ships 15 starter prompts with unique ids and useful categories", () => {
    expect(STARTER_PROMPTS).toHaveLength(15);
    expect(new Set(STARTER_PROMPTS.map((p) => p.id)).size).toBe(15);
    expect(new Set(STARTER_PROMPTS.map((p) => p.category))).toEqual(new Set(["Writing", "Coding", "Study", "Work", "Everyday"]));
    expect(STARTER_PROMPTS.map((p) => p.title)).toContain("Explain code");
    expect(STARTER_PROMPTS.map((p) => p.title)).toContain("Translate text");
    expect(STARTER_PROMPTS.map((p) => p.title)).toContain("Make study notes");
  });

  it("finds each {{variable}} once and fills it in", () => {
    const text = "Write about {{ topic }} for {{level}}, then repeat {{topic}}.";
    expect(promptVariables(text)).toEqual(["topic", "level"]);
    expect(fillPromptTemplate(text, { topic: "caches", level: "a beginner" })).toBe("Write about caches for a beginner, then repeat caches.");
    expect(fillPromptTemplate(text, { topic: "caches" })).toContain("{{level}}");
  });

  it("imports an array or the exported object and drops malformed entries", () => {
    const good: SavedPrompt = { id: "p1", title: "One", text: "Say {{x}}", category: "Custom" };
    expect(parsePromptJson(JSON.stringify([good, { id: "bad", title: "", text: "x" }]))).toEqual([good]);
    const round = parsePromptJson(exportPrompts([good]));
    expect(round).toEqual([good]);
    expect(parsePromptJson("{not json")).toBeNull();
    expect(parsePromptJson(JSON.stringify({ prompts: "no" }))).toBeNull();
  });

  it("merges by id and lists custom categories after the built-in ones", () => {
    const a: SavedPrompt = { id: "a", title: "A", text: "a", category: "Writing" };
    const b: SavedPrompt = { id: "b", title: "B", text: "b", category: "Games" };
    const changed: SavedPrompt = { ...a, title: "A changed" };
    expect(mergePrompts([a], [changed, b])).toEqual([changed, b]);
    expect(promptCategories([a, b])).toEqual(["Writing", "Coding", "Study", "Work", "Everyday", "Games"]);
  });
});
