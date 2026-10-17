import { createSignal } from "solid-js";
import type { LogEntry } from "./permissions";

/**
 * Every tool call that finished in Chat this session, for the Tools screen's Activity list. Chat
 * appends as calls end (done, denied, failed); the list lives for the session and holds the newest
 * 500.
 */
const [entries, setEntries] = createSignal<LogEntry[]>([]);

export const toolActivity = entries;

export function recordToolActivity(e: LogEntry): void {
  setEntries((l) => [e, ...l].slice(0, 500));
}

/** A short line for the log: the call's main argument, as a person would describe it. */
export function summarizeArgs(name: string, args: unknown): string {
  const a = (args ?? {}) as Record<string, unknown>;
  const pick = ["command", "path", "url", "pattern", "query", "plan"].find((k) => typeof a[k] === "string");
  if (pick) return String(a[pick]).replace(/\s+/g, " ").slice(0, 160);
  if (name === "todo_write" && Array.isArray(a.todos)) return `${a.todos.length} task${a.todos.length === 1 ? "" : "s"}`;
  if (name === "ask_user" && Array.isArray(a.questions)) return String((a.questions[0] as { question?: unknown })?.question ?? "").slice(0, 160);
  const s = JSON.stringify(a);
  return s === "{}" ? "" : s.slice(0, 160);
}
