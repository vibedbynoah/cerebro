import { describe, expect, it } from "vitest";
import { addToCollection, docsInCollection, docsInScope, folderNames, loadCollections, makeCollection, parseCollections, removeFromCollection, saveCollections, scopeFromKey, scopeKey, tagNames } from "../../ui/src/lib/collections";
import { makeDoc, type Doc } from "../../ui/src/lib/documents";

const docs: Doc[] = [
  makeDoc("a", "handbook.txt", "text/plain", 10, "Remote work is allowed up to three days a week.", 0, { folder: "Work", tags: ["policy"] }),
  makeDoc("b", "pricing.md", "text/markdown", 10, "The starter plan is 19 euros a month.", 0, { folder: "Work", tags: ["money", "policy"] }),
  makeDoc("c", "notes.txt", "text/plain", 10, "Milk, bread, coffee.", 0, { tags: ["shopping"] }),
];

describe("collections", () => {
  it("reads back only well-formed collections", () => {
    expect(parseCollections(null)).toEqual([]);
    expect(parseCollections(JSON.stringify([{ id: "c1", name: "Policy" }]))).toEqual([{ id: "c1", name: "Policy" }]);
    expect(parseCollections(JSON.stringify([{ id: "", name: "No id" }]))).toEqual([]);
    expect(parseCollections(JSON.stringify([{ id: "c1", name: 42 }]))).toEqual([]);
    expect(parseCollections("not json")).toEqual([]);
  });

  it("keeps the collection list on this machine", () => {
    saveCollections([makeCollection("Policy", "c1")]);
    expect(loadCollections().map((c) => c.name)).toEqual(["Policy"]);
    saveCollections([]);
  });

  it("puts files in a collection and takes them out again", () => {
    const withFiles = addToCollection(docs, ["a", "b"], "c1");
    expect(docsInCollection(withFiles, "c1").map((d) => d.id)).toEqual(["a", "b"]);
    // A file already in the collection is not added twice.
    expect(addToCollection(withFiles, ["a"], "c1")[0]?.collections).toEqual(["c1"]);
    const without = removeFromCollection(withFiles, ["a"], "c1");
    expect(docsInCollection(without, "c1").map((d) => d.id)).toEqual(["b"]);
  });

  it("names the files a scope covers", () => {
    expect(docsInScope(docs, { kind: "all" }).length).toBe(3);
    expect(docsInScope(docs, { kind: "folder", name: "work" }).map((d) => d.id)).toEqual(["a", "b"]);
    expect(docsInScope(docs, { kind: "tag", name: "policy" }).map((d) => d.id)).toEqual(["a", "b"]);
    expect(docsInScope(docs, { kind: "collection", id: "c1" }).length).toBe(0);
  });

  it("turns a scope into a key and back", () => {
    const collections = [makeCollection("Policy", "c1")];
    expect(scopeKey({ kind: "all" })).toBe("all");
    expect(scopeKey({ kind: "collection", id: "c1" })).toBe("c:c1");
    expect(scopeKey({ kind: "folder", name: "Work" })).toBe("f:work");
    expect(scopeKey({ kind: "tag", name: "Policy" })).toBe("t:policy");
    expect(scopeFromKey("c:c1", collections)).toEqual({ kind: "collection", id: "c1" });
    expect(scopeFromKey("c:missing", collections)).toBeNull();
    expect(scopeFromKey("all", collections)).toEqual({ kind: "all" });
  });

  it("lists the folders and tags the files use", () => {
    expect(folderNames(docs)).toEqual(["Work"]);
    expect(tagNames(docs)).toEqual(["money", "policy", "shopping"]);
  });
});
