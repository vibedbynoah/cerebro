import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { noticeText, npmNotices } from "../../scripts/third-party.mjs";
import { DOCS } from "../../ui/src/legal/content";
import { LEGAL_DOCS } from "../../ui/src/legal/types";
import { MOCK_CATALOG, MOCK_HARDWARE } from "../../ui/src/mock/fixtures";
import { firstStep, needsOnboarding, parseRecord, STEPS, TERMS_STEP, TERMS_VERSION, type OnboardingRecord } from "../../ui/src/onboarding/progress";
import { fitFor, pct, rankModels, unranked, usableMemory } from "../../ui/src/onboarding/recommend";
import type { CatalogModel, HardwareProfile } from "../../ui/src/service/types";

const done: OnboardingRecord = { v: 1, completed: true, termsVersion: TERMS_VERSION, model: "qwen3-8b-q5km" };

describe("onboarding record", () => {
  it("round-trips a valid record", () => {
    expect(parseRecord(JSON.stringify(done))).toEqual(done);
    expect(parseRecord(JSON.stringify({ ...done, model: null, termsVersion: null, completed: false }))).toEqual({
      v: 1,
      completed: false,
      termsVersion: null,
      model: null,
    });
  });

  it("rejects anything malformed, oversized or tampered with", () => {
    const bad = [
      null,
      "",
      "not json",
      "[]",
      "null",
      "42",
      JSON.stringify({ ...done, v: 2 }),
      JSON.stringify({ ...done, completed: "yes" }),
      JSON.stringify({ ...done, termsVersion: "latest" }),
      JSON.stringify({ ...done, termsVersion: 20260930 }),
      JSON.stringify({ ...done, model: "<img src=x onerror=alert(1)>" }),
      JSON.stringify({ ...done, model: "../../etc/passwd" }),
      JSON.stringify({ ...done, model: "A".repeat(10) }),
      JSON.stringify({ ...done, pad: "x".repeat(600) }),
    ];
    for (const raw of bad) expect(parseRecord(raw), String(raw)).toBeNull();
  });

  it("copies only known fields, so extra keys and __proto__ never reach the app", () => {
    const parsed = parseRecord(`{"v":1,"completed":true,"termsVersion":"${TERMS_VERSION}","model":null,"__proto__":{"polluted":true},"admin":true}`);
    expect(parsed).toEqual({ v: 1, completed: true, termsVersion: TERMS_VERSION, model: null });
    expect(Object.keys(parsed ?? {})).toEqual(["v", "completed", "termsVersion", "model"]);
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });

  it("shows setup until it is finished with the current terms; old terms reopen at the terms step", () => {
    expect(needsOnboarding(null)).toBe(true);
    expect(needsOnboarding({ ...done, completed: false })).toBe(true);
    expect(needsOnboarding(done)).toBe(false);
    const oldTerms = { ...done, termsVersion: "2020-01-01" };
    expect(needsOnboarding(oldTerms)).toBe(true);
    expect(firstStep(oldTerms)).toBe(TERMS_STEP);
    expect(firstStep(null)).toBe(0);
    expect(firstStep({ ...done, completed: false, termsVersion: null })).toBe(0);
    expect(STEPS[TERMS_STEP]?.id).toBe("terms");
    expect(STEPS.map((s) => s.id)).toEqual(["welcome", "hardware", "model", "privacy", "terms", "finish"]);
  });
});

describe("model recommendation", () => {
  it("ranks the mock catalog by fit on the mock machine", () => {
    const recs = rankModels(MOCK_CATALOG, MOCK_HARDWARE);
    expect(recs.map((r) => [r.model.id, r.fit, r.rank])).toEqual([
      ["qwen3-8b-q5km", "gpu", 1],
      ["gemma-4-e4b-q4km", "gpu", 2],
      ["qwen3-coder-30b-a3b-q4km", "moe-offload", 3],
      ["mistral-small-24b-q4km", "split", 4],
      ["llama-3.3-70b-q4km", "too-large", 5],
    ]);
    for (const r of recs) {
      expect(r.why.length).toBeGreaterThan(40);
      expect(r.share).toBeGreaterThan(0);
      expect(r.share).toBeLessThanOrEqual(1);
    }
  });

  it("never states a speed: every why is about memory fit, and promises a measurement where speed is implied", () => {
    for (const r of rankModels(MOCK_CATALOG, MOCK_HARDWARE)) {
      expect(r.why).not.toMatch(/tok\/s|\d+\s*[x×]|faster than/i);
      if (r.fit === "gpu") expect(r.why).toMatch(/Cerebro measures/);
    }
  });

  it("never ranks speed without a measurement: comparative speed words only next to 'measure'", () => {
    const cpuOnly: HardwareProfile = { ...MOCK_HARDWARE, gpu: null, ramTotalGiB: 64 };
    const strings = [...rankModels(MOCK_CATALOG, MOCK_HARDWARE), ...rankModels(MOCK_CATALOG, cpuOnly), ...unranked(MOCK_CATALOG)].flatMap((r) => [r.why, r.fitLabel]);
    for (const s of strings) {
      if (/(slow|slower|slowest|slows|fast|faster|fastest|quick|quicker|quickest|speedier)/i.test(s)) expect(s, s).toMatch(/measure/i);
    }
    expect(rankModels(MOCK_CATALOG, MOCK_HARDWARE).find((r) => r.fit === "split")?.fitLabel).toBe("Runs partly from system memory");
  });

  it("lists the catalog unranked, in catalog order, when the hardware is unknown", () => {
    const recs = unranked(MOCK_CATALOG);
    expect(recs.map((r) => r.model.id)).toEqual(MOCK_CATALOG.map((m) => m.id));
    for (const r of recs) {
      expect(r.ranked).toBe(false);
      expect(r.fit).toBe("unknown");
      expect(r.why).toMatch(/could not be read/);
    }
    expect(rankModels(MOCK_CATALOG, MOCK_HARDWARE).every((r) => r.ranked)).toBe(true);
  });

  it("handles a machine with no usable graphics card", () => {
    const cpuOnly: HardwareProfile = { ...MOCK_HARDWARE, gpu: null, ramTotalGiB: 16 };
    expect(usableMemory(cpuOnly)).toEqual({ vramGiB: 0, ramGiB: 10, totalGiB: 10 });
    const recs = rankModels(MOCK_CATALOG, cpuOnly);
    expect(recs.some((r) => r.fit === "gpu")).toBe(false);
    const small = recs.find((r) => r.model.id === "gemma-4-e4b-q4km");
    expect(small?.fit).toBe("split");
    expect(small?.why).toMatch(/processor/);
  });

  it("treats nonsense sizes and memory as zero instead of producing NaN", () => {
    const weird: CatalogModel = { ...(MOCK_CATALOG[0] as CatalogModel), id: "weird", sizeGiB: Number.NaN };
    const hw: HardwareProfile = { ...MOCK_HARDWARE, ramTotalGiB: Number.POSITIVE_INFINITY, gpu: { name: "x", vramTotalGiB: -4 } };
    const budget = usableMemory(hw);
    expect(budget).toEqual({ vramGiB: 0, ramGiB: 0, totalGiB: 0 });
    expect(fitFor(weird, budget)).toBe("split");
    const [r] = rankModels([weird], hw);
    expect(Number.isFinite(r?.share)).toBe(true);
    expect(pct(Number.NaN)).toBe(0);
    expect(pct(-1)).toBe(0);
    expect(pct(2)).toBe(100);
    expect(pct(0.183)).toBe(18);
  });
});

describe("bundled legal documents", () => {
  it("has every document the terms step links to, as plain text", () => {
    for (const { id } of LEGAL_DOCS) {
      const doc = DOCS[id];
      expect(doc.id).toBe(id);
      expect(doc.sections.length).toBeGreaterThan(0);
      for (const s of doc.sections) {
        for (const b of s.blocks) {
          const texts =
            b.kind === "p"
              ? [b.text]
              : b.kind === "list"
                ? [...b.items]
                : b.kind === "notices"
                  ? [b.title, b.detail, ...b.notices.paragraphs, ...b.notices.packages.flatMap((p) => [p.name, p.version, p.license, p.note ?? ""])]
                  : [b.name, b.detail, b.license, b.text ?? ""];
          for (const t of texts) expect(typeof t).toBe("string");
        }
      }
    }
    expect(DOCS.terms.summary).toContain(TERMS_VERSION);
  });

  it("covers the points the setup summary makes", () => {
    const all = (id: "terms" | "privacy" | "licenses") =>
      DOCS[id].sections
        .flatMap((s) => s.blocks.map((b) => (b.kind === "p" ? b.text : b.kind === "list" ? b.items.join(" ") : b.kind === "notices" ? b.title : `${b.name} ${b.license}`)))
        .join(" ");
    const terms = all("terms");
    expect(terms).toMatch(/as is/i);
    expect(terms).toMatch(/Apache License/);
    expect(terms).toMatch(/models? .*own licen[cs]e/i);
    expect(terms).toMatch(/plugins/i);
    expect(DOCS.privacy.summary).toMatch(/no telemetry/i);
    expect(all("privacy")).toMatch(/does not send them anywhere/);
    const licenses = all("licenses");
    for (const name of ["Cerebro Engine", "JavaScript packages", "Rust crates", "Inter", "Fraunces", "JetBrains Mono", "llama.cpp"]) expect(licenses).toContain(name);
    // The full Apache text is bundled, not linked to a website.
    const apache = DOCS.licenses.sections[0]?.blocks[0];
    expect(apache?.kind === "license" && (apache.text ?? "").includes("TERMS AND CONDITIONS FOR USE")).toBe(true);
  });

  it("never claims more than the protections do (docs/CLAIMS.md)", () => {
    const text = JSON.stringify(DOCS);
    expect(text).not.toMatch(/sandboxed|can(not|'t) (read|access) (your )?files|no network access|bit-exact/i);
  });
});

describe("generated third-party notices", () => {
  it("lists every crate compiled into the desktop app, each with licence text", () => {
    const rust = DOCS.licenses.sections.flatMap((s) => s.blocks).find((b) => b.kind === "notices" && b.id === "rust");
    expect(rust?.kind).toBe("notices");
    if (rust?.kind !== "notices") return;
    const names = rust.notices.packages.map((p) => p.name);
    for (const crate of ["tauri", "serde", "ed25519-dalek", "subtle", "getrandom", "zeroize"]) expect(names).toContain(crate);
    for (const p of rust.notices.packages) {
      expect(p.texts.length, p.name).toBeGreaterThan(0);
      // Some crates add a one-line LICENSE ("MIT OR Apache-2.0") beside the full texts.
      expect(Math.max(...p.texts.map((t) => noticeText(rust.notices, t).length)), p.name).toBeGreaterThan(100);
    }
    // A crate that ships no licence file gets the standard text and says so.
    for (const p of rust.notices.packages.filter((q) => q.note)) expect(p.note).toMatch(/no licence file/);
  });

  it("marks the JavaScript list as build-generated where nothing is bundled (Vitest, dev server)", () => {
    const js = DOCS.licenses.sections.flatMap((s) => s.blocks).find((b) => b.kind === "notices" && b.id === "bundled");
    expect(js?.kind === "notices" && js.complete).toBe(false);
  });

  it("collects bundled packages from module ids, nested copies included, and refuses one without a licence", () => {
    const root = mkdtempSync(path.join(os.tmpdir(), "cerebro-notices-"));
    try {
      const pkg = (dir: string, name: string, licence: string | null) => {
        mkdirSync(dir, { recursive: true });
        writeFileSync(path.join(dir, "package.json"), JSON.stringify({ name, version: "1.2.3", license: "MIT" }));
        if (licence !== null) writeFileSync(path.join(dir, "LICENSE"), licence);
      };
      const nm = path.join(root, "node_modules");
      pkg(path.join(nm, "plain"), "plain", "MIT License\r\n\r\nCopyright (c) Plain\r\n\r\nShared body.");
      pkg(path.join(nm, "@scope", "pkg"), "@scope/pkg", "MIT License\n\nCopyright (c) Scope\n\nShared body.");
      pkg(path.join(nm, "plain", "node_modules", "inner"), "inner", "Inner licence");
      const ids = [
        path.join(nm, "plain", "dist", "index.js"),
        `\0${path.join(nm, "@scope", "pkg", "index.js")}?commonjs-proxy`,
        path.join(nm, "plain", "node_modules", "inner", "x.js"),
        path.join(root, "src", "app.ts"),
      ];
      const n = npmNotices(ids);
      expect(n.packages.map((p) => p.name)).toEqual(["@scope/pkg", "inner", "plain"]);
      const plain = n.packages.find((p) => p.name === "plain");
      expect(noticeText(n, plain?.texts[0] ?? -1)).toBe("MIT License\n\nCopyright (c) Plain\n\nShared body.");
      // Shared paragraphs are stored once.
      expect(n.paragraphs.filter((p) => p === "Shared body.")).toHaveLength(1);

      pkg(path.join(nm, "bare"), "bare", null);
      expect(() => npmNotices([path.join(nm, "bare", "index.js")])).toThrow(/publishes no licence file/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
