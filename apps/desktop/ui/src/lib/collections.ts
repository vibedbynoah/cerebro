import { readPref, writePref } from "./storage";
import { newId } from "./chat/store";
import type { Doc } from "./documents";

/**
 * Collections: named groups of indexed files. A file can sit in several collections, and Chat can
 * search one collection instead of everything. Everything is kept on this machine, next to the
 * documents themselves; nothing is uploaded.
 */

export interface Collection {
  id: string;
  name: string;
}

export const COLLECTIONS_KEY = "cerebro-doc-collections";
const MAX_COLLECTIONS = 50;
const MAX_NAME = 60;

export function parseCollections(raw: string | null): Collection[] {
  if (!raw) return [];
  try {
    const v: unknown = JSON.parse(raw);
    if (!Array.isArray(v)) return [];
    const seen = new Set<string>();
    return v
      .flatMap((x): Collection[] => {
        if (!x || typeof x !== "object") return [];
        const c = x as Record<string, unknown>;
        const id = typeof c.id === "string" ? c.id.slice(0, 64) : "";
        const name = typeof c.name === "string" ? c.name.trim().slice(0, MAX_NAME) : "";
        if (!id || !name || seen.has(id)) return [];
        seen.add(id);
        return [{ id, name }];
      })
      .slice(0, MAX_COLLECTIONS);
  } catch {
    return [];
  }
}

export function loadCollections(): Collection[] {
  return parseCollections(readPref(COLLECTIONS_KEY));
}

export function saveCollections(list: readonly Collection[]): void {
  writePref(COLLECTIONS_KEY, JSON.stringify(list.slice(0, MAX_COLLECTIONS)));
}

/** A collection with a fresh id and a clean name. */
export function makeCollection(name: string, id = newId()): Collection {
  return { id, name: name.trim().slice(0, MAX_NAME) };
}

export function collectionName(list: readonly Collection[], id: string): string {
  return list.find((c) => c.id === id)?.name ?? "";
}

/** The files in one collection, in the order they were indexed. */
export function docsInCollection(docs: readonly Doc[], id: string): Doc[] {
  return docs.filter((d) => d.collections.includes(id));
}

/** The files in one collection, with the collection's name. */
export function collectionNames(list: readonly Collection[]): string[] {
  return list.map((c) => c.name);
}

/** Puts files into a collection. Files already in it are left alone. */
export function addToCollection(docs: readonly Doc[], docIds: readonly string[], collectionId: string): Doc[] {
  const ids = new Set(docIds);
  return docs.map((d) => (ids.has(d.id) && !d.collections.includes(collectionId) ? { ...d, collections: [...d.collections, collectionId] } : d));
}

/** Takes files out of a collection. */
export function removeFromCollection(docs: readonly Doc[], docIds: readonly string[], collectionId: string): Doc[] {
  const ids = new Set(docIds);
  return docs.map((d) => (ids.has(d.id) ? { ...d, collections: d.collections.filter((c) => c !== collectionId) } : d));
}

/** Which files a scope covers: everything, one collection, one folder or one tag. */
export type DocScope =
  | { kind: "all" }
  | { kind: "collection"; id: string }
  | { kind: "folder"; name: string }
  | { kind: "tag"; name: string };

export function scopeKey(scope: DocScope): string {
  switch (scope.kind) {
    case "all":
      return "all";
    case "collection":
      return `c:${scope.id}`;
    case "folder":
      return `f:${scope.name.toLowerCase()}`;
    case "tag":
      return `t:${scope.name.toLowerCase()}`;
  }
}

/** The scope a key names, or null when the key names none. */
export function scopeFromKey(key: string, collections: readonly Collection[]): DocScope | null {
  if (key === "all") return { kind: "all" };
  if (key.startsWith("c:")) {
    const id = key.slice(2);
    return collections.some((c) => c.id === id) ? { kind: "collection", id } : null;
  }
  if (key.startsWith("f:")) return { kind: "folder", name: key.slice(2) };
  if (key.startsWith("t:")) return { kind: "tag", name: key.slice(2) };
  return null;
}

export function docsInScope(docs: readonly Doc[], scope: DocScope): Doc[] {
  switch (scope.kind) {
    case "all":
      return [...docs];
    case "collection":
      return docsInCollection(docs, scope.id);
    case "folder":
      return docs.filter((d) => (d.folder ?? "").toLowerCase() === scope.name.toLowerCase());
    case "tag":
      return docs.filter((d) => d.tags.some((t) => t.toLowerCase() === scope.name.toLowerCase()));
  }
}

/** The folders the indexed files use, sorted. */
export function folderNames(docs: readonly Doc[]): string[] {
  return [...new Set(docs.flatMap((d) => (d.folder ? [d.folder] : [])))].sort((a, b) => a.localeCompare(b));
}

/** The tags the indexed files use, sorted. */
export function tagNames(docs: readonly Doc[]): string[] {
  return [...new Set(docs.flatMap((d) => d.tags))].sort((a, b) => a.localeCompare(b));
}
