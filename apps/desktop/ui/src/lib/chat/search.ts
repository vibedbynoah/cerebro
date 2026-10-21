import type { Conversation } from "./store";

/**
 * Search across every chat on this machine. Matching is case-insensitive and every word in the
 * query must appear in the title or one message; results carry a short snippet around the first
 * match so the caller can show it with <mark> around the hits. No markup is ever produced here.
 */

export interface ChatSearchResult {
  conversationId: string;
  conversationTitle: string;
  /** The message to jump to, or null when the chat's title itself matched. */
  messageId: string | null;
  role: "user" | "assistant" | null;
  snippet: string;
  /** True when the title matched (the row opens the chat at its current position). */
  titleMatch: boolean;
}

export function queryTerms(query: string): string[] {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, 8);
}

const hasAll = (text: string, terms: readonly string[]): boolean => {
  const lower = text.toLowerCase();
  return terms.every((t) => lower.includes(t));
};

/** Splits text into runs, marking every occurrence of any query word. Adjacent hits merge. */
export function highlightRuns(text: string, query: string): Array<{ text: string; hit: boolean }> {
  const terms = [...new Set(queryTerms(query))].sort((a, b) => b.length - a.length);
  if (!terms.length || !text) return [{ text, hit: false }];
  const marks = new Array<boolean>(text.length).fill(false);
  const lower = text.toLowerCase();
  for (const term of terms) {
    let from = 0;
    for (;;) {
      const at = lower.indexOf(term, from);
      if (at < 0) break;
      for (let i = at; i < at + term.length && i < marks.length; i++) marks[i] = true;
      from = at + Math.max(1, term.length);
    }
  }
  const out: Array<{ text: string; hit: boolean }> = [];
  let start = 0;
  for (let i = 1; i <= text.length; i++) {
    if (i === text.length || marks[i] !== marks[start]) {
      out.push({ text: text.slice(start, i), hit: marks[start] === true });
      start = i;
    }
  }
  return out;
}

/** A short window around the first hit, on word edges where possible. */
export function snippetAround(text: string, query: string, before = 60, after = 120): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const terms = queryTerms(query);
  const lower = flat.toLowerCase();
  let at = -1;
  for (const term of terms) {
    const i = lower.indexOf(term);
    if (i >= 0 && (at < 0 || i < at)) at = i;
  }
  if (at < 0 || flat.length <= before + after + 1) return flat;
  let start = Math.max(0, at - before);
  let end = Math.min(flat.length, at + after);
  if (start > 0) {
    const space = flat.indexOf(" ", start);
    if (space > 0 && space < at) start = space + 1;
  }
  if (end < flat.length) {
    const space = flat.lastIndexOf(" ", end);
    if (space > at) end = space;
  }
  return `${start > 0 ? "…" : ""}${flat.slice(start, end)}${end < flat.length ? "…" : ""}`;
}

/** All matching chats, title hits first, then messages newest chat first. */
export function searchConversations(list: readonly Conversation[], query: string): ChatSearchResult[] {
  const terms = queryTerms(query);
  if (!terms.length) return [];
  const out: ChatSearchResult[] = [];
  const ordered = [...list].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const conversation of ordered) {
    if (hasAll(conversation.title, terms)) {
      out.push({
        conversationId: conversation.id,
        conversationTitle: conversation.title,
        messageId: null,
        role: null,
        snippet: conversation.title,
        titleMatch: true,
      });
    }
    const seen = new Set<string>();
    const messages = conversation.tree?.nodes.length ? conversation.tree.nodes.map((n) => n.message) : conversation.messages;
    for (const message of messages) {
      if (seen.has(message.id)) continue;
      seen.add(message.id);
      if (!hasAll(message.text, terms)) continue;
      out.push({
        conversationId: conversation.id,
        conversationTitle: conversation.title,
        messageId: message.id,
        role: message.role,
        snippet: snippetAround(message.text, query),
        titleMatch: false,
      });
    }
    if (out.length >= 100) break;
  }
  return out.slice(0, 100);
}
