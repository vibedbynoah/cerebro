import { LEDGER_FILES, type ChatStats, type LedgerFileName, type ToolActivity } from "../../service/chat-types";
import { readPref, writePref } from "../storage";
import type { Effort, Memory, ReasoningSource, Verbosity } from "./prompt";

/**
 * Conversations and memories, kept on this machine in local storage. Everything read back is
 * validated field by field; a tampered or oversized entry is dropped, never rendered as markup
 * (replies are always parsed by lib/chat/markdown.ts).
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
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  /** Per-chat "don't remember this". */
  noMemory: boolean;
  updatedAt: number;
}

export const CONVERSATIONS_KEY = "cerebro-chats";
export const MEMORIES_KEY = "cerebro-memories";
const MAX_CONVERSATIONS = 50;
const MAX_TEXT = 200_000;
const MAX_PREVIEW = 2_000_000;

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
  return { id: newId(), title: "New chat", messages: [], noMemory: false, updatedAt: Date.now() };
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
            const state = oneOf(x.state, ["approval", "running", "done", "denied", "error"] as const, "error");
            const live = state === "approval" || state === "running";
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
    // A reply that was streaming when the app closed is finished as stopped.
    streaming: false,
    error: str(m.error, 500),
  };
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
      return [{ id, title, messages: x.messages.map(parseMessage).filter((m): m is ChatMessage => m !== null), noMemory: x.noMemory === true, updatedAt: num(x.updatedAt) }];
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
