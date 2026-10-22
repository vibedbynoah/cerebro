import { LEDGER_FILES, type ChatStats, type LedgerFileName, type ToolActivity } from "../../service/chat-types";
import { readPref, writePref } from "../storage";
import type { Effort, Memory, ReasoningSource, Verbosity } from "./prompt";
import type { Source } from "../rag";

/**
 * Conversations and memories, kept on this machine in local storage. Everything read back is
 * validated field by field; a tampered or oversized entry is dropped, never rendered as markup
 * (replies are always parsed by lib/chat/markdown.ts).
 *
 * A conversation is a tree of messages: each node names its parent and the tree remembers which
 * child was chosen at every step. `messages` is the active path, kept in the stored record so an
 * older build can still read the chat; the tree is what makes branches possible. Chats saved before
 * branching have no tree and are migrated into a single path when they load.
 */

export interface Attachment {
  name: string;
  type: string;
  bytes: number;
  /** data: URL preview for images only (CSP allows data: images), else null. */
  preview: string | null;
}

/** Orchestrate mode state of one reply: the ledger folder, the role running now, the files so far. */
export interface LedgerState {
  /** The ledger folder on disk, from the first event; null until it arrives. */
  dir: string | null;
  /** The role that started last ("plan", "worker 2", "finalize"), with its round. */
  step: { role: string; round: number } | null;
  /** The whole text of each ledger file that changed. */
  files: Partial<Record<LedgerFileName, string>>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  thinking: string;
  checks: string[];
  image: { prompt: string; seed: number } | null;
  /** Tool calls the model made in this reply, in order. */
  tools: ToolActivity[];
  /** Orchestrate mode's ledger, null for a normal reply. */
  ledger: LedgerState | null;
  attachments: Attachment[];
  stats: ChatStats | null;
  effort: Effort;
  verbosity: Verbosity;
  source: ReasoningSource;
  streaming: boolean;
  error: string | null;
  /** The passages of the user's documents this reply used, as chips under the reply. */
  sources: Source[];
}

/** One message in a conversation's tree. The message id is also the node id. */
export interface BranchNode {
  id: string;
  parentId: string | null;
  message: ChatMessage;
}

/** Every message ever sent in a chat, plus the child chosen at each step. The root's parent is "". */
export interface BranchTree {
  nodes: BranchNode[];
  selected: Record<string, string>;
}

export interface Conversation {
  id: string;
  title: string;
  /** The active path through the tree: what the screen shows and what is sent to the model. */
  messages: ChatMessage[];
  tree: BranchTree;
  /** Per-chat "don't remember this". */
  noMemory: boolean;
  updatedAt: number;
  pinned: boolean;
  /** One optional folder, null for none. */
  folder: string | null;
  tags: string[];
  archived: boolean;
}

export const CONVERSATIONS_KEY = "cerebro-chats";
export const MEMORIES_KEY = "cerebro-memories";
const MAX_CONVERSATIONS = 50;
const MAX_TEXT = 200_000;
const MAX_PREVIEW = 2_000_000;
const MAX_BRANCH_NODES = 500;
const MAX_FOLDER = 60;
const MAX_TAG = 30;
const MAX_TAGS = 8;

export const newId = (): string => {
  const b = new Uint8Array(8);
  crypto.getRandomValues(b);
  return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
};

export function titleFrom(text: string): string {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t) return "New chat";
  return t.length > 48 ? `${t.slice(0, 47)}…` : t;
}

export function emptyConversation(): Conversation {
  return {
    id: newId(),
    title: "New chat",
    messages: [],
    tree: { nodes: [], selected: {} },
    noMemory: false,
    updatedAt: Date.now(),
    pinned: false,
    folder: null,
    tags: [],
    archived: false,
  };
}

const str = (v: unknown, max = MAX_TEXT): string | null => (typeof v === "string" && v.length <= max ? v : null);
const oneOf = <T extends string>(v: unknown, list: readonly T[], d: T): T => (typeof v === "string" && (list as readonly string[]).includes(v) ? (v as T) : d);
const num = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0);

const MAX_LEDGER_FILE = 20_000;
const LEDGER_STEP = /^(plan|ideate|manage|finalize|worker [1-6])$/;

function parseLedger(v: unknown): LedgerState | null {
  if (!v || typeof v !== "object") return null;
  const x = v as Record<string, unknown>;
  const files: Partial<Record<LedgerFileName, string>> = {};
  if (x.files && typeof x.files === "object") {
    for (const name of LEDGER_FILES) {
      const text = str((x.files as Record<string, unknown>)[name], MAX_LEDGER_FILE);
      if (text !== null) files[name] = text;
    }
  }
  const s = x.step && typeof x.step === "object" ? (x.step as Record<string, unknown>) : null;
  const step = s && str(s.role, 40) !== null && LEDGER_STEP.test(s.role as string) ? { role: s.role as string, round: num(s.round) } : null;
  return { dir: str(x.dir, 1000), step, files };
}

function parseSources(v: unknown): Source[] {
  if (!Array.isArray(v)) return [];
  return v
    .flatMap((raw): Source[] => {
      if (!raw || typeof raw !== "object") return [];
      const s = raw as Record<string, unknown>;
      const n = num(s.n);
      const docId = str(s.docId, 64);
      const docName = str(s.docName, 260);
      const text = str(s.text, 4000);
      if (!docId || !docName || text === null || n < 1) return [];
      return [{ n, docId, docName, text, index: num(s.index) }];
    })
    .slice(0, 20);
}

function parseMessage(v: unknown): ChatMessage | null {
  if (!v || typeof v !== "object") return null;
  const m = v as Record<string, unknown>;
  const id = str(m.id, 64);
  const text = str(m.text);
  if (!id || text === null || (m.role !== "user" && m.role !== "assistant")) return null;
  const atts = Array.isArray(m.attachments)
    ? m.attachments.flatMap((a): Attachment[] => {
        if (!a || typeof a !== "object") return [];
        const x = a as Record<string, unknown>;
        const name = str(x.name, 260);
        const type = str(x.type, 100);
        const preview = x.preview === null ? null : str(x.preview, MAX_PREVIEW);
        if (name === null || type === null) return [];
        return [{ name, type, bytes: num(x.bytes), preview: preview && /^data:image\/(png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+$/.test(preview) ? preview : null }];
      })
    : [];
  const img = m.image && typeof m.image === "object" ? (m.image as Record<string, unknown>) : null;
  const s = m.stats && typeof m.stats === "object" ? (m.stats as Record<string, unknown>) : null;
  return {
    id,
    role: m.role,
    text,
    thinking: str(m.thinking) ?? "",
    checks: Array.isArray(m.checks) ? m.checks.flatMap((c) => (str(c, 2000) !== null ? [c as string] : [])).slice(0, 2) : [],
    image: img && str(img.prompt, 400) !== null ? { prompt: img.prompt as string, seed: num(img.seed) } : null,
    tools: Array.isArray(m.tools)
      ? m.tools
          .flatMap((t): ToolActivity[] => {
            if (!t || typeof t !== "object") return [];
            const x = t as Record<string, unknown>;
            const id = str(x.id, 64);
            const name = str(x.name, 200);
            const server = str(x.server, 200);
            if (id === null || name === null || server === null) return [];
            // A call still waiting or running when the app closed did not finish.
            const state = oneOf(x.state, ["approval", "question", "plan", "running", "done", "denied", "error"] as const, "error");
            const live = state === "approval" || state === "question" || state === "plan" || state === "running";
            return [{ id, name, server, arguments: x.arguments ?? {}, state: live ? "error" : state, result: str(x.result, 20000), error: live ? "Interrupted." : str(x.error, 2000) }];
          })
          .slice(0, 32)
      : [],
    attachments: atts.slice(0, 10),
    ledger: parseLedger(m.ledger),
    stats: s
      ? { outputTokens: num(s.outputTokens), thinkingTokens: num(s.thinkingTokens), tokPerSec: num(s.tokPerSec), ttftMs: num(s.ttftMs), totalMs: num(s.totalMs), extraMs: num(s.extraMs), stopped: s.stopped === true }
      : null,
    effort: oneOf(m.effort, ["low", "medium", "high", "xhigh", "max"] as const, "medium"),
    verbosity: oneOf(m.verbosity, ["brief", "normal", "detailed"] as const, "brief"),
    source: oneOf(m.source, ["native", "prompts", "both", "none"] as const, "native"),
    sources: parseSources(m.sources),
    // A reply that was streaming when the app closed is finished as stopped.
    streaming: false,
    error: str(m.error, 500),
  };
}

/** A plain chain for a chat saved before branching existed. */
export function treeFromMessages(messages: readonly ChatMessage[]): BranchTree {
  const nodes: BranchNode[] = [];
  const selected: Record<string, string> = {};
  let parentId: string | null = null;
  for (const message of messages) {
    nodes.push({ id: message.id, parentId, message });
    selected[parentId ?? ""] = message.id;
    parentId = message.id;
  }
  return { nodes, selected };
}

function parseTree(v: unknown): BranchTree | null {
  if (!v || typeof v !== "object") return null;
  const x = v as Record<string, unknown>;
  if (!Array.isArray(x.nodes)) return null;
  const nodes: BranchNode[] = [];
  const ids = new Set<string>();
  // Keep the newest nodes: a very long chat should reopen near where the person left it.
  for (const raw of x.nodes.slice(-MAX_BRANCH_NODES)) {
    if (!raw || typeof raw !== "object") continue;
    const n = raw as Record<string, unknown>;
    const id = str(n.id, 64);
    const message = parseMessage(n.message);
    if (!id || !message || ids.has(id)) continue;
    let parentId: string | null;
    if (n.parentId === null || n.parentId === undefined) parentId = null;
    else {
      const parent = str(n.parentId, 64);
      if (parent === null) continue;
      parentId = parent;
    }
    message.id = id;
    ids.add(id);
    nodes.push({ id, parentId, message });
  }
  if (!nodes.length) return null;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  // A node whose parent vanished (or a parent cycle) cannot be reached; drop it rather than hang.
  const valid: BranchNode[] = [];
  for (const node of nodes) {
    if (node.parentId !== null && !byId.has(node.parentId)) continue;
    const seen = new Set<string>([node.id]);
    let parent = node.parentId;
    while (parent !== null) {
      if (seen.has(parent)) break;
      seen.add(parent);
      parent = byId.get(parent)?.parentId ?? null;
    }
    if (parent === null) valid.push(node);
  }
  if (!valid.length) return null;
  const validById = new Map(valid.map((n) => [n.id, n]));
  const selected: Record<string, string> = {};
  if (x.selected && typeof x.selected === "object") {
    for (const [key, value] of Object.entries(x.selected as Record<string, unknown>)) {
      if (typeof value !== "string" || !validById.has(value)) continue;
      const parent = validById.get(value)!.parentId;
      if ((key === "" && parent === null) || (key !== "" && validById.has(key) && parent === key)) selected[key] = value;
    }
  }
  return { nodes: valid, selected };
}

function parseTags(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  const out: string[] = [];
  for (const raw of v) {
    const tag = str(raw, MAX_TAG)?.trim();
    if (tag && !out.includes(tag)) out.push(tag);
    if (out.length >= MAX_TAGS) break;
  }
  return out;
}

/** The path from a root to a leaf, following the child chosen at each step. */
export function branchPath(tree: BranchTree): BranchNode[] {
  const byParent = new Map<string, BranchNode[]>();
  for (const node of tree.nodes) {
    const key = node.parentId ?? "";
    const list = byParent.get(key);
    if (list) list.push(node);
    else byParent.set(key, [node]);
  }
  const out: BranchNode[] = [];
  const seen = new Set<string>();
  let parent = "";
  for (;;) {
    const children = byParent.get(parent);
    if (!children?.length) break;
    const wanted = tree.selected[parent];
    const node = children.find((n) => n.id === wanted) ?? children[children.length - 1]!;
    if (seen.has(node.id)) break;
    seen.add(node.id);
    out.push(node);
    parent = node.id;
  }
  return out;
}

export function activeMessages(tree: BranchTree): ChatMessage[] {
  return branchPath(tree).map((n) => n.message);
}

/** Which of its siblings a message is: 1-based index and the total. */
export function branchPosition(tree: BranchTree, id: string): { index: number; total: number } {
  const node = tree.nodes.find((n) => n.id === id);
  if (!node) return { index: 1, total: 1 };
  const siblings = tree.nodes.filter((n) => (n.parentId ?? "") === (node.parentId ?? ""));
  return { index: Math.max(0, siblings.findIndex((n) => n.id === id)) + 1, total: siblings.length };
}

/** Switches one message to its previous (-1) or next (+1) sibling, wrapping at the ends. */
export function switchBranch(tree: BranchTree, id: string, delta: number): BranchTree {
  const node = tree.nodes.find((n) => n.id === id);
  if (!node) return tree;
  const siblings = tree.nodes.filter((n) => (n.parentId ?? "") === (node.parentId ?? ""));
  if (siblings.length < 2) return tree;
  const at = siblings.findIndex((n) => n.id === id);
  const next = siblings[(at + delta + siblings.length) % siblings.length];
  if (!next) return tree;
  return { nodes: tree.nodes, selected: { ...tree.selected, [node.parentId ?? ""]: next.id } };
}

/** Makes the path to a message (and through it, the last child chosen below it) active. */
export function focusBranch(tree: BranchTree, id: string): BranchTree {
  const byId = new Map(tree.nodes.map((n) => [n.id, n]));
  let node = byId.get(id);
  if (!node) return tree;
  const selected = { ...tree.selected };
  while (node) {
    selected[node.parentId ?? ""] = node.id;
    node = node.parentId === null ? undefined : byId.get(node.parentId);
  }
  return { nodes: tree.nodes, selected };
}

export function parseConversations(raw: string | null): Conversation[] {
  if (!raw) return [];
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(v)) return [];
  return v
    .flatMap((c): Conversation[] => {
      if (!c || typeof c !== "object") return [];
      const x = c as Record<string, unknown>;
      const id = str(x.id, 64);
      const title = str(x.title, 200);
      if (!id || title === null || !Array.isArray(x.messages)) return [];
      const seen = new Set<string>();
      const messages = x.messages
        .map(parseMessage)
        .filter((m): m is ChatMessage => m !== null)
        .filter((m) => (seen.has(m.id) ? false : (seen.add(m.id), true)))
        .slice(-MAX_BRANCH_NODES);
      const parsedTree = parseTree(x.tree);
      let tree = parsedTree ?? treeFromMessages(messages);
      if (!branchPath(tree).length) tree = treeFromMessages(messages);
      const active = activeMessages(tree);
      const folder = str(x.folder, MAX_FOLDER)?.trim() ?? "";
      return [
        {
          id,
          title,
          messages: active.length ? active : messages,
          tree,
          noMemory: x.noMemory === true,
          updatedAt: num(x.updatedAt),
          pinned: x.pinned === true,
          folder: folder || null,
          tags: parseTags(x.tags),
          archived: x.archived === true,
        },
      ];
    })
    .slice(0, MAX_CONVERSATIONS);
}

export function loadConversations(): Conversation[] {
  return parseConversations(readPref(CONVERSATIONS_KEY));
}

export function saveConversations(list: readonly Conversation[]): void {
  writePref(CONVERSATIONS_KEY, JSON.stringify(list.slice(0, MAX_CONVERSATIONS)));
}

export function parseMemories(raw: string | null): Memory[] {
  if (!raw) return [];
  try {
    const v: unknown = JSON.parse(raw);
    if (!Array.isArray(v)) return [];
    return v
      .flatMap((m): Memory[] => {
        if (!m || typeof m !== "object") return [];
        const x = m as Record<string, unknown>;
        const id = str(x.id, 64);
        const text = str(x.text, 500);
        if (!id || !text || !text.trim()) return [];
        return [{ id, text, scope: x.scope === "project" ? "project" : "you" }];
      })
      .slice(0, 200);
  } catch {
    return [];
  }
}

export const DEFAULT_MEMORIES: Memory[] = [
  { id: "m-seed-1", text: "Works mostly in Rust and TypeScript.", scope: "you" },
  { id: "m-seed-2", text: "Prefers short answers with the command or code first.", scope: "you" },
  { id: "m-seed-3", text: "Cerebro project: Windows first, everything offline.", scope: "project" },
];

export function loadMemories(): Memory[] {
  const raw = readPref(MEMORIES_KEY);
  return raw === null ? DEFAULT_MEMORIES.map((m) => ({ ...m })) : parseMemories(raw);
}

export function saveMemories(list: readonly Memory[]): void {
  writePref(MEMORIES_KEY, JSON.stringify(list));
}

/** Facts worth remembering from a user message (illustrative: the engine will extract these). */
export function memoryCandidate(text: string): string | null {
  const m = /^\s*(?:remember(?: that)?|note that)\s+(.{3,300})$/i.exec(text);
  return m ? m[1]!.replace(/[.\s]+$/, "") + "." : null;
}
