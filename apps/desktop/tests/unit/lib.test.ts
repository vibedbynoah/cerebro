import { describe, expect, it } from "vitest";
import { fmt1, fmtContext, ratio } from "../../ui/src/lib/format";
import { fuzzyFilter, fuzzyMatch, highlightRuns } from "../../ui/src/lib/fuzzy";
import { GLOSSARY } from "../../ui/src/lib/glossary";
import { hrefFor, parseHash, ROUTES } from "../../ui/src/lib/routes";
import { SETTINGS_META } from "../../ui/src/lib/settings-meta";
import { DEFAULT_SETTINGS } from "../../ui/src/mock/fixtures";
import { ICON_NAMES } from "../../ui/src/components/icons";

describe("fuzzy matching", () => {
  it("matches subsequences case-insensitively and rejects the rest", () => {
    expect(fuzzyMatch("tne", "Tune")?.indices).toEqual([0, 2, 3]);
    expect(fuzzyMatch("MH", "Model Hub")?.indices).toEqual([0, 6]);
    expect(fuzzyMatch("xyz", "Tune")).toBeNull();
    expect(fuzzyMatch("   ", "Tune")).toEqual({ score: 0, indices: [] });
  });

  it("ranks prefix and word-start hits above scattered ones", () => {
    const items = ["Theme: Mono", "Documents", "Models", "Model Hub", "Tools"];
    const ranked = fuzzyFilter("mo", items, (s) => [s]).map((r) => r.item);
    expect(ranked).toEqual(["Models", "Model Hub", "Theme: Mono"]);
    // "Documents" has no "o" after its "m", so it is not a subsequence match at all.
    expect(ranked).not.toContain("Tools");
  });

  it("matches keywords but never highlights them in the label", () => {
    const [hit] = fuzzyFilter("sandbox", [{ label: "Security", kw: "sandbox integrity" }], (i) => [i.label, i.kw]);
    expect(hit?.item.label).toBe("Security");
    expect(hit?.match.indices).toEqual([]);
  });

  it("keeps equal scores in authored order", () => {
    const ranked = fuzzyFilter("", ["b", "a", "c"], (s) => [s]).map((r) => r.item);
    expect(ranked).toEqual(["b", "a", "c"]);
  });

  it("splits highlight runs as text, so markup in a label stays text", () => {
    expect(highlightRuns("<b>", [0])).toEqual([
      { text: "<", hit: true },
      { text: "b>", hit: false },
    ]);
    expect(highlightRuns("Tune", [0, 2, 3])).toEqual([
      { text: "T", hit: true },
      { text: "u", hit: false },
      { text: "ne", hit: true },
    ]);
  });
});

describe("routing", () => {
  it("round-trips every route", () => {
    for (const r of ROUTES) expect(parseHash(hrefFor(r.id))).toBe(r.id);
  });

  it("sends anything unknown or hostile to Home", () => {
    for (const h of ["", "#", "#/", "#/nope", "#/../../etc/passwd", "#/Settings", "#/settings/extra", "#/javascript:alert(1)", "#/__proto__", "#/constructor"]) {
      expect(parseHash(h)).toBe("home");
    }
    expect(parseHash("#/settings/")).toBe("settings");
  });

  it("has every screen the briefs name, each with a real icon and a plain summary", () => {
    expect(ROUTES.map((r) => r.label)).toEqual([
      "Home",
      "Model Hub",
      "Models",
      "Tune",
      "Monitor",
      "Chat",
      "Image Studio",
      "Documents",
      "Plugins",
      "Tools",
      "Security",
      "Settings",
      "About",
    ]);
    for (const r of ROUTES) {
      expect(ICON_NAMES).toContain(r.icon);
      expect(r.summary.endsWith(".")).toBe(true);
    }
  });
});

describe("formatting", () => {
  it("formats readouts with a fixed locale and clamps ratios", () => {
    expect(fmt1(16.84)).toBe("16.8");
    expect(fmt1(1234.5)).toBe("1,234.5");
    expect(fmtContext(32768)).toBe("32k");
    expect(fmtContext(1000)).toBe("1,000");
    expect(ratio(7, 8)).toBe(0.875);
    expect(ratio(9, 8)).toBe(1);
    expect(ratio(-1, 8)).toBe(0);
    expect(ratio(1, 0)).toBe(0);
    expect(ratio(Number.NaN, 8)).toBe(0);
  });
});

describe("settings copy", () => {
  it("describes every setting exactly once, with one short neutral note", () => {
    expect(SETTINGS_META.map((m) => m.key).sort()).toEqual(Object.keys(DEFAULT_SETTINGS).sort());
    for (const m of SETTINGS_META) {
      expect(m.note.length).toBeLessThanOrEqual(110);
      // User control rule: explain, never nag.
      expect(m.note).not.toMatch(/!|warning|dangerous|not recommended|are you sure|risk/i);
      if (m.term) expect(GLOSSARY[m.term]).toBeDefined();
    }
  });

  it("never claims more than the mechanisms do (docs/CLAIMS.md)", () => {
    const all = [...SETTINGS_META.flatMap((m) => [m.note, m.limits ?? ""]), ...Object.values(GLOSSARY).map((g) => g.text)].join(" ");
    expect(all).not.toMatch(/bit-exact|byte-identical|1M token|NPU|uses your whole machine|sandboxed from|cannot access/i);
    expect(GLOSSARY.jobObject.text).toMatch(/does not limit/i);
  });
});
