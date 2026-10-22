import { describe, expect, it } from "vitest";
import { isVerified, publisherOf, relevance, sortRepos, verifiedPublisher, VERIFIED_PUBLISHERS, VERIFIED_TOOLTIP } from "../../ui/src/lib/hub-verified";
import type { HubRepo } from "../../ui/src/service/types";

const repo = (id: string, downloads = 1000, likes = 10): HubRepo => ({
  id,
  author: id.split("/")[0] ?? id,
  downloads,
  likes,
  updated: "2026-09-01T00:00:00Z",
  license: null,
  tags: ["gguf"],
});

describe("the checked list of publishers", () => {
  it("names every publisher with an id, a name and what they publish", () => {
    expect(VERIFIED_PUBLISHERS.length).toBeGreaterThan(10);
    for (const p of VERIFIED_PUBLISHERS) {
      expect(p.id).toMatch(/^[A-Za-z0-9_.-]+$/);
      expect(p.name.length).toBeGreaterThan(1);
      expect(p.publishes.length).toBeGreaterThan(10);
    }
    // No publisher is listed twice.
    expect(new Set(VERIFIED_PUBLISHERS.map((p) => p.id.toLowerCase())).size).toBe(VERIFIED_PUBLISHERS.length);
  });

  it("covers the model makers and the GGUF publishers people know", () => {
    for (const id of ["Qwen", "google", "meta-llama", "mistralai", "microsoft", "deepseek-ai", "openai", "nvidia", "ibm-granite", "HuggingFaceTB", "unsloth", "ggml-org", "lmstudio-community", "bartowski"]) {
      expect(isVerified(`${id}/some-model-GGUF`), id).toBe(true);
    }
  });

  it("finds the publisher of a repository and leaves an unknown one alone", () => {
    expect(publisherOf("unsloth/Qwen3-8B-GGUF")).toBe("unsloth");
    expect(publisherOf("Qwen")).toBe("Qwen");
    expect(verifiedPublisher("unsloth/Qwen3-8B-GGUF")?.name).toBe("Unsloth");
    // Hugging Face account names do not care about case.
    expect(verifiedPublisher("Unsloth/Qwen3-8B-GGUF")?.id).toBe("unsloth");
    expect(verifiedPublisher("someone-else/Qwen3-8B-GGUF")).toBeNull();
    expect(isVerified("someone-else/Qwen3-8B-GGUF")).toBe(false);
  });

  it("says what the badge means, and does not promise more", () => {
    expect(VERIFIED_TOOLTIP).toMatch(/Cerebro's list of known publishers/);
    expect(VERIFIED_TOOLTIP).toMatch(/not a guarantee/i);
  });
});

describe("the order of the search results", () => {
  it("puts a verified publisher first at equal relevance", () => {
    const quiet = repo("unsloth/Qwen3-8B-GGUF", 100);
    const busy = repo("someone/Qwen3-8B-GGUF", 900_000);
    const sorted = sortRepos([busy, quiet], "Qwen3-8B GGUF");
    expect(sorted.map((r) => r.id)).toEqual([quiet.id, busy.id]);
  });

  it("keeps the better match first, whatever the publisher", () => {
    const exact = repo("someone/Qwen3-8B-Instruct-GGUF", 10);
    const loose = repo("unsloth/Qwen3-Coder-GGUF", 900_000);
    const sorted = sortRepos([loose, exact], "Qwen3-8B-Instruct GGUF");
    expect(sorted.map((r) => r.id)[0]).toBe(exact.id);
    expect(relevance(exact, "Qwen3-8B-Instruct GGUF")).toBeGreaterThan(relevance(loose, "Qwen3-8B-Instruct GGUF"));
  });

  it("falls back to downloads when relevance and publisher are equal", () => {
    const a = repo("someone/a-GGUF", 500);
    const b = repo("other/b-GGUF", 900);
    expect(sortRepos([a, b], "GGUF").map((r) => r.id)).toEqual([b.id, a.id]);
  });

  it("does not change the list it is given", () => {
    const list = [repo("someone/a-GGUF"), repo("unsloth/b-GGUF")];
    const copy = [...list];
    sortRepos(list, "GGUF");
    expect(list).toEqual(copy);
  });
});
