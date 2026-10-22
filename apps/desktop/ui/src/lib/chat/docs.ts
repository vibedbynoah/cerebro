import { readPref, writePref } from "../storage";
import { scopeFromKey, type DocScope } from "../collections";
import type { Collection } from "../collections";

/**
 * What Chat remembers about the user's documents: whether to use them when answering, and which
 * files to search. Off by default: the documents are only sent to a model when the user asks for
 * it, and the passages stay on the computer unless a cloud model is picked.
 */

export interface DocsPrefs {
  /** Use the indexed documents when answering. Off until the user turns it on. */
  on: boolean;
  /** Which files to search: "all", "c:<collection id>", "f:<folder>" or "t:<tag>". */
  scope: string;
}

export const DOCS_PREFS_KEY = "cerebro-chat-docs";

export const DEFAULT_DOCS_PREFS: DocsPrefs = { on: false, scope: "all" };

export function parseDocsPrefs(raw: string | null): DocsPrefs {
  if (!raw) return { ...DEFAULT_DOCS_PREFS };
  try {
    const v = JSON.parse(raw) as Record<string, unknown>;
    const scope = typeof v.scope === "string" && v.scope.length <= 200 ? v.scope : "all";
    return { on: v.on === true, scope };
  } catch {
    return { ...DEFAULT_DOCS_PREFS };
  }
}

export function loadDocsPrefs(): DocsPrefs {
  return parseDocsPrefs(readPref(DOCS_PREFS_KEY));
}

export function saveDocsPrefs(p: DocsPrefs): void {
  writePref(DOCS_PREFS_KEY, JSON.stringify(p));
}

/** The scope the settings name, or everything when the name is not one we know. */
export function scopeOf(prefs: DocsPrefs, collections: readonly Collection[]): DocScope {
  return scopeFromKey(prefs.scope, collections) ?? { kind: "all" };
}

/** What the scope covers, in words: "All documents", "Collection: Notes", "Folder: Work". */
export function scopeLabel(prefs: DocsPrefs, collections: readonly Collection[]): string {
  const scope = scopeOf(prefs, collections);
  switch (scope.kind) {
    case "all":
      return "All documents";
    case "collection":
      return `Collection: ${collections.find((c) => c.id === scope.id)?.name ?? "unknown"}`;
    case "folder":
      return `Folder: ${scope.name}`;
    case "tag":
      return `Tag: ${scope.name}`;
  }
}
