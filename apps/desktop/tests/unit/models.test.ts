import { describe, expect, it } from "vitest";
import { estimateFit, fitClass } from "../../ui/src/lib/fit";
import { parseGgufValue, validateGgufEdits } from "../../ui/src/lib/gguf";
import { baselineParams, changedFields, draftCandidates, LIMITS, parseCores, startingParams, validateLoadParams } from "../../ui/src/lib/load-params";
import { modelFromHash, parseHash } from "../../ui/src/lib/routes";
import {
  applyTweaks,
  checkTemplate,
  contextCeiling,
  diffLines,
  EMPTY_TWEAKS,
  exportProfile,
  importProfile,
  sanitizeTweaks,
  stopTokenError,
  TWEAK_LIMITS,
  tweakedFields,
} from "../../ui/src/lib/tweaks";
import { MOCK_HARDWARE } from "../../ui/src/mock/fixtures";
import { detailFor, MOCK_LIBRARY } from "../../ui/src/mock/library";
import { createMockService } from "../../ui/src/mock/mock-service";
import { matchesQuery } from "../../ui/src/screens/models/Library";
import { importedName } from "../../ui/src/screens/models/TweakStudio";
import { windowRange } from "../../ui/src/screens/models/virtual";
import type { GgufEntry, LibraryModel, LoadParams } from "../../ui/src/service/types";

const find = (pred: (m: LibraryModel) => boolean): LibraryModel => {
  const m = MOCK_LIBRARY.find(pred);
  if (!m) throw new Error("fixture not found");
  return m;
};
const MOE = find((m) => m.id === "qwen3-coder-30b-a3b-q4km");
const DENSE = find((m) => m.id === "qwen3-8b-q5km");
const draftIds = (m: LibraryModel) => new Set(draftCandidates(m, MOCK_LIBRARY).map((d) => d.id));

describe("load settings validation", () => {
  it("accepts the tuned and starting settings of every text and vision model", () => {
    for (const m of MOCK_LIBRARY.filter((x) => x.kind === "text" || x.kind === "vision")) {
      expect(validateLoadParams(baselineParams(m, MOCK_HARDWARE), m, MOCK_HARDWARE, draftIds(m)), m.id).toEqual({});
      expect(validateLoadParams(startingParams(m, MOCK_HARDWARE), m, MOCK_HARDWARE, draftIds(m)), m.id).toEqual({});
    }
  });

  it("rejects missing, extra and wrongly typed fields", () => {
    const ok = baselineParams(MOE, MOCK_HARDWARE);
    expect(validateLoadParams(null, MOE, MOCK_HARDWARE)).not.toEqual({});
    expect(validateLoadParams([], MOE, MOCK_HARDWARE)).not.toEqual({});
    const { threads: _t, ...missing } = ok;
    expect(validateLoadParams(missing, MOE, MOCK_HARDWARE)).not.toEqual({});
    expect(validateLoadParams({ ...ok, extra: "--rpc 1.2.3.4" }, MOE, MOCK_HARDWARE)).not.toEqual({});
    expect(validateLoadParams({ ...ok, contextLength: "8192" }, MOE, MOCK_HARDWARE).contextLength).toBeDefined();
    expect(validateLoadParams({ ...ok, contextLength: 8192.5 }, MOE, MOCK_HARDWARE).contextLength).toBeDefined();
    expect(validateLoadParams({ ...ok, gpuLayers: MOE.arch.layers + 1 }, MOE, MOCK_HARDWARE).gpuLayers).toBeDefined();
    expect(validateLoadParams({ ...ok, threads: MOCK_HARDWARE.cpu.threads + 1 }, MOE, MOCK_HARDWARE).threads).toBeDefined();
    expect(validateLoadParams({ ...ok, kvCacheK: "q2" }, MOE, MOCK_HARDWARE).kvCacheK).toBeDefined();
    expect(validateLoadParams({ ...ok, mmap: "yes" }, MOE, MOCK_HARDWARE).mmap).toBeDefined();
  });

  it("explains combinations llama.cpp cannot run", () => {
    const ok = baselineParams(MOE, MOCK_HARDWARE);
    expect(validateLoadParams({ ...ok, kvCacheV: "q4_0", flashAttention: "off" }, MOE, MOCK_HARDWARE).kvCacheV).toMatch(/flash attention/);
    expect(validateLoadParams({ ...ok, batchSize: 256, ubatchSize: 512 }, MOE, MOCK_HARDWARE).ubatchSize).toBeDefined();
    const dense = baselineParams(DENSE, MOCK_HARDWARE);
    expect(validateLoadParams({ ...dense, moeCpuLayers: 2 }, DENSE, MOCK_HARDWARE).moeCpuLayers).toBeDefined();
    expect(validateLoadParams({ ...dense, speculative: "draft", draftModelId: "not-a-model" }, DENSE, MOCK_HARDWARE, draftIds(DENSE)).draftModelId).toBeDefined();
    expect(validateLoadParams({ ...dense, cpuAffinity: "all", cpuCores: "0-3" }, DENSE, MOCK_HARDWARE).cpuCores).toBeDefined();
  });

  it("parses core lists strictly", () => {
    expect(parseCores("0-3,8,10-11", 16)).toEqual([0, 1, 2, 3, 8, 10, 11]);
    expect(parseCores("3-1", 16)).toBeNull();
    expect(parseCores("0-16", 16)).toBeNull();
    expect(parseCores("0;calc", 16)).toBeNull();
    expect(parseCores("", 16)).toBeNull();
    expect(parseCores("1,".repeat(40), 16)).toBeNull();
  });

  it("lists changed fields in display order", () => {
    const a = baselineParams(MOE, MOCK_HARDWARE);
    expect(changedFields(a, { ...a })).toEqual([]);
    expect(changedFields({ ...a, threads: 1, contextLength: 1024 }, a)).toEqual(["contextLength", "threads"]);
  });
});

describe("memory fit estimate", () => {
  it("keeps every illustrative tuned result inside the example machine's VRAM", () => {
    for (const m of MOCK_LIBRARY.filter((x) => x.tunedParams)) {
      const draft = m.tunedParams?.draftModelId ? MOCK_LIBRARY.find((d) => d.id === m.tunedParams?.draftModelId) : null;
      expect(estimateFit(m, m.tunedParams as LoadParams, MOCK_HARDWARE, draft ?? null).vram.status, m.id).not.toBe("over");
    }
  });

  it("moves memory the way the settings say", () => {
    const p = baselineParams(MOE, MOCK_HARDWARE);
    const base = estimateFit(MOE, p, MOCK_HARDWARE);
    expect(estimateFit(MOE, { ...p, contextLength: p.contextLength * 2 }, MOCK_HARDWARE).vram.kv).toBeGreaterThan(base.vram.kv);
    expect(estimateFit(MOE, { ...p, kvCacheK: "f16", kvCacheV: "f16" }, MOCK_HARDWARE).vram.kv).toBeGreaterThan(estimateFit(MOE, { ...p, kvCacheK: "q4_0", kvCacheV: "q4_0" }, MOCK_HARDWARE).vram.kv);
    const fewerOnCpu = estimateFit(MOE, { ...p, moeCpuLayers: 0 }, MOCK_HARDWARE);
    expect(fewerOnCpu.vram.weights).toBeGreaterThan(base.vram.weights);
    expect(fewerOnCpu.ram.weights).toBeLessThan(base.ram.weights);
    const cpuOnly = estimateFit(MOE, { ...p, gpuLayers: 0 }, MOCK_HARDWARE);
    expect(cpuOnly.vram.total).toBe(0);
  });

  it("estimates nothing for a machine that was not detected", () => {
    expect(fitClass(DENSE, null)).toBe("unknown");
    // A conservative CPU-only plan, valid under the absolute thread cap.
    const p = startingParams(DENSE, null);
    expect(p.gpuLayers).toBe(0);
    expect(p.moeCpuLayers).toBe(0);
    expect(validateLoadParams(p, DENSE, null)).toEqual({});
    expect(baselineParams(DENSE, null)).toEqual(p);
    expect(validateLoadParams({ ...p, threads: LIMITS.threadsCap }, DENSE, null)).toEqual({});
    expect(validateLoadParams({ ...p, threads: LIMITS.threadsCap + 1 }, DENSE, null).threads).toBeDefined();
    // A tuned model keeps its measured settings either way.
    expect(baselineParams(MOE, null)).toEqual(MOE.tunedParams);
  });

  it("puts nothing on a GPU the machine does not have", () => {
    const noGpu = { ...MOCK_HARDWARE, gpu: null };
    const p = startingParams(DENSE, noGpu);
    expect(p.gpuLayers).toBe(0);
    expect(estimateFit(DENSE, { ...p, gpuLayers: 10 }, noGpu).vram.total).toBe(0);
    expect(fitClass(DENSE, noGpu)).not.toBe("vram");
  });
});

describe("tweaks laid over load settings (one rule for what a load uses)", () => {
  it("a KV tweak changes both K and V and leaves the tuned settings untouched", () => {
    const tuned = baselineParams(MOE, MOCK_HARDWARE);
    const before = { ...tuned };
    const eff = applyTweaks(tuned, { ...EMPTY_TWEAKS, kvQuant: "q4_0" });
    expect(eff.kvCacheK).toBe("q4_0");
    expect(eff.kvCacheV).toBe("q4_0");
    expect(tuned).toEqual(before);
    expect(applyTweaks(tuned, EMPTY_TWEAKS)).toEqual(tuned);
    expect([...tweakedFields({ ...EMPTY_TWEAKS, kvQuant: "q8_0", contextOverride: 4096 })].sort()).toEqual(["contextLength", "kvCacheK", "kvCacheV"]);
  });

  it("validates the combination: a quantized V cache from a tweak with flash attention off is refused", () => {
    const tuned = { ...baselineParams(DENSE, MOCK_HARDWARE), flashAttention: "off" as const };
    expect(validateLoadParams(tuned, DENSE, MOCK_HARDWARE)).toEqual({});
    const eff = applyTweaks(tuned, { ...EMPTY_TWEAKS, kvQuant: "q4_0" });
    expect(validateLoadParams(eff, DENSE, MOCK_HARDWARE).kvCacheV).toMatch(/flash attention/);
  });

  it("RoPE scaling raises the context ceiling, and only then", () => {
    const yarn = { scaling: "yarn" as const, factor: 4, originalContext: 32768 };
    expect(contextCeiling({ contextMax: 32768 }, { rope: yarn })).toBe(131072);
    expect(contextCeiling({ contextMax: 32768 }, { rope: null })).toBe(32768);
    expect(contextCeiling({ contextMax: 32768 }, { rope: { ...yarn, scaling: "none" } })).toBe(32768);
    const eff = applyTweaks(baselineParams(DENSE, MOCK_HARDWARE), { ...EMPTY_TWEAKS, contextOverride: 131072, rope: yarn });
    expect(validateLoadParams(eff, DENSE, MOCK_HARDWARE).contextLength).toBeDefined();
    expect(validateLoadParams(eff, { ...DENSE, contextMax: contextCeiling(DENSE, { rope: yarn }) }, MOCK_HARDWARE)).toEqual({});
  });
});

describe("routes that name a model", () => {
  it("keep the screen and accept only well-formed ids", () => {
    expect(parseHash("#/tune?model=qwen3-8b-q5km")).toBe("tune");
    expect(modelFromHash("#/tune?model=qwen3-8b-q5km")).toBe("qwen3-8b-q5km");
    expect(modelFromHash("#/tune")).toBeNull();
    expect(modelFromHash("#/tune?model=..%2F..%2Fetc")).toBeNull();
    expect(modelFromHash("#/tune?model=%3Cimg%3E")).toBeNull();
    expect(parseHash("#/tune?model=x#/settings")).toBe("home");
  });
});

describe("Tweak Studio rules", () => {
  it("checks template structure without running it", () => {
    expect(checkTemplate("{% for m in messages %}{{ m.content }}{% endfor %}")).toEqual([]);
    expect(checkTemplate("{{ broken")[0]).toMatch(/Unbalanced/);
    expect(checkTemplate("{% if x %}open")[0]).toMatch(/never closed/);
    expect(checkTemplate("{% for x in y %}{% endif %}")[0]).toMatch(/closes a/);
    expect(checkTemplate("   ")[0]).toMatch(/empty/);
  });

  it("diffs by line and refuses inputs too large to diff", () => {
    expect(diffLines("a\nb\nc", "a\nc\nd")).toEqual([
      { kind: "same", text: "a" },
      { kind: "del", text: "b" },
      { kind: "same", text: "c" },
      { kind: "add", text: "d" },
    ]);
    expect(diffLines("x\n".repeat(700), "y")).toBeNull();
  });

  it("refuses control characters, duplicates and overlong stop tokens", () => {
    expect(stopTokenError("</s>", [])).toBeNull();
    expect(stopTokenError("a\u0007b", [])).not.toBeNull();
    expect(stopTokenError("</s>", ["</s>"])).not.toBeNull();
    expect(stopTokenError("x".repeat(TWEAK_LIMITS.stopTokenMaxChars + 1), [])).not.toBeNull();
  });

  it("round-trips a shared profile and rebuilds only known fields", () => {
    const profile = { name: "Long docs", modelSha256: DENSE.sha256, tweaks: { ...EMPTY_TWEAKS, kvQuant: "q8_0" as const, stopTokens: ["</answer>"] }, fixes: ["qwen3-im-end"] };
    const back = importProfile(exportProfile(profile));
    expect(back).toEqual({ ok: true, profile });

    const hostile = `{"format":"cerebro-patch-profile","version":1,"name":"x","modelSha256":"${"a".repeat(64)}","__proto__":{"polluted":1},"tweaks":{"kvQuant":null,"contextOverride":null,"rope":null,"chatTemplate":null,"stopTokens":[],"sampler":null,"__proto__":{"polluted":1}},"fixes":[]}`;
    const r = importProfile(hostile);
    expect(r.ok).toBe(true);
    if (r.ok) expect(Object.keys(r.profile.tweaks).sort()).toEqual(Object.keys(EMPTY_TWEAKS).sort());
    expect(({} as Record<string, unknown>)["polluted"]).toBeUndefined();
  });

  it("names an imported profile whose name is taken so that it always fits and never collides", () => {
    expect(importedName("Long docs", ["Long docs"])).toBe("Long docs (import)");
    expect(importedName("Long docs", ["Long docs", "long docs (import)"])).toBe("Long docs (import 2)");
    const long = "x".repeat(TWEAK_LIMITS.profileNameMax);
    const named = importedName(long, [long]);
    expect(named.length).toBeLessThanOrEqual(TWEAK_LIMITS.profileNameMax);
    expect(named.endsWith(" (import)")).toBe(true);
  });

  it("rejects malformed, oversized and out-of-range profiles", () => {
    expect(importProfile("not json").ok).toBe(false);
    expect(importProfile(" ".repeat(TWEAK_LIMITS.profileMaxBytes + 1)).ok).toBe(false);
    expect(importProfile(JSON.stringify({ format: "cerebro-patch-profile", version: 2 })).ok).toBe(false);
    expect(sanitizeTweaks({ ...EMPTY_TWEAKS, contextOverride: 10 })).toBeNull();
    expect(sanitizeTweaks({ ...EMPTY_TWEAKS, rope: { scaling: "yarn", factor: 99, originalContext: 32768 } })).toBeNull();
    expect(sanitizeTweaks({ ...EMPTY_TWEAKS, sampler: { temperature: 0.7, topK: 2.5, topP: 0.9, minP: 0, repeatPenalty: 1 } })).toBeNull();
    expect(sanitizeTweaks({ ...EMPTY_TWEAKS, stopTokens: "</s>" })).toBeNull();
  });
});

describe("GGUF metadata editing", () => {
  const entries = detailFor(DENSE).gguf as GgufEntry[];

  it("parses values against the stored type", () => {
    expect(parseGgufValue({ key: "x.context_length", type: "uint32" }, "40960")).toEqual({ ok: true, value: 40960 });
    expect(parseGgufValue({ key: "x.context_length", type: "uint32" }, "-1").ok).toBe(false);
    expect(parseGgufValue({ key: "x.context_length", type: "uint32" }, "0").ok).toBe(false);
    expect(parseGgufValue({ key: "x", type: "float32" }, "1e-6")).toEqual({ ok: true, value: 1e-6 });
    expect(parseGgufValue({ key: "x", type: "float32" }, "1e40").ok).toBe(false);
    expect(parseGgufValue({ key: "x", type: "bool" }, "yes").ok).toBe(false);
    expect(parseGgufValue({ key: "x", type: "string" }, "a\u0000b").ok).toBe(false);
    expect(parseGgufValue({ key: "x", type: "array" }, "[]").ok).toBe(false);
  });

  it("refuses structural keys, unknown keys, repeats and wrong types", () => {
    const structural = entries.find((e) => !e.editable);
    const editable = entries.find((e) => e.editable && e.type === "uint32");
    expect(structural && editable).toBeTruthy();
    if (!structural || !editable) return;
    expect(validateGgufEdits([], entries)).not.toBeNull();
    expect(validateGgufEdits([{ key: structural.key, value: structural.value }], entries)).toMatch(/structural/);
    expect(validateGgufEdits([{ key: "general.nope", value: 1 }], entries)).toMatch(/no key/);
    expect(validateGgufEdits([{ key: editable.key, value: 1 }, { key: editable.key, value: 2 }], entries)).not.toBeNull();
    expect(validateGgufEdits([{ key: editable.key, value: "4096" }], entries)).toMatch(/wrong type/);
    expect(validateGgufEdits([{ key: editable.key, value: 4096 }], entries)).toBeNull();
  });
});

describe("library list", () => {
  it("windows only the rows in view, plus overscan", () => {
    expect(windowRange(0, 600, 60, 593, 6)).toEqual([0, 16]);
    expect(windowRange(6000, 600, 60, 593, 6)).toEqual([94, 116]);
    expect(windowRange(1e9, 600, 60, 593, 6)[1]).toBe(593);
    expect(windowRange(-50, 600, 60, 0, 6)).toEqual([0, 0]);
  });

  it("matches every search word as plain text", () => {
    expect(matchesQuery(DENSE, "qwen3 q5_k_m")).toBe(true);
    expect(matchesQuery(DENSE, "qwen3 gemma")).toBe(false);
    expect(matchesQuery(DENSE, "(.*)")).toBe(false);
    expect(matchesQuery(DENSE, "   ")).toBe(true);
  });
});

describe("mock service: Models screen arguments are checked like the Rust side will", () => {
  it("refuses unknown models, invalid settings, and settings for engines without any", async () => {
    const s = createMockService();
    await expect(s.loadModel("../../etc", null)).rejects.toThrow();
    await expect(s.loadModel(MOE.id, { ...baselineParams(MOE, MOCK_HARDWARE), gpuLayers: -1 })).rejects.toThrow();
    const stt = find((m) => m.kind === "stt");
    await expect(s.loadModel(stt.id, baselineParams(MOE, MOCK_HARDWARE))).rejects.toThrow();
    await expect(s.savePreset(MOE.id, "\u0000", baselineParams(MOE, MOCK_HARDWARE))).rejects.toThrow();
  });

  it("keeps one chat model loaded and puts speech engines beside it", async () => {
    const s = createMockService();
    const stt = find((m) => m.kind === "stt");
    await s.loadModel(DENSE.id, null);
    const loaded = await s.loadModel(stt.id, null);
    expect(loaded.map((l) => l.slot).sort()).toEqual(["llm", "stt"]);
    expect(loaded.find((l) => l.slot === "llm")?.id).toBe(DENSE.id);
  });

  it("keeps the tweaks in effect per file, validated, with only that file's known fixes", async () => {
    const s = createMockService();
    expect((await s.getActivePatch(DENSE.sha256)).fixes).toEqual([]);
    await expect(s.getActivePatch("f".repeat(64))).rejects.toThrow();
    await expect(s.setActivePatch(DENSE.sha256, { tweaks: { ...EMPTY_TWEAKS, contextOverride: 1 }, fixes: [] })).rejects.toThrow();
    await expect(s.setActivePatch(DENSE.sha256, { tweaks: EMPTY_TWEAKS, fixes: ["not-a-fix"] })).rejects.toThrow();
    const fix = detailFor(DENSE).quirks[0]?.id;
    const saved = await s.setActivePatch(DENSE.sha256, { tweaks: { ...EMPTY_TWEAKS, kvQuant: "q8_0" }, fixes: fix ? [fix, fix] : [] });
    expect(saved.fixes.length).toBe(fix ? 1 : 0);
    expect((await s.getActivePatch(DENSE.sha256)).tweaks.kvQuant).toBe("q8_0");
    expect((await s.getActivePatch(MOE.sha256)).tweaks.kvQuant).toBeNull();
  });

  it("a second write merges into the copy instead of replacing it", async () => {
    const s = createMockService();
    const gguf = detailFor(DENSE).gguf ?? [];
    const ctx = gguf.find((e) => e.key.endsWith(".context_length"));
    const name = gguf.find((e) => e.key === "general.name");
    expect(ctx && name).toBeTruthy();
    if (!ctx || !name) return;
    await s.writeGgufCopy(DENSE.id, [{ key: ctx.key, value: 40960 }]);
    const second = await s.writeGgufCopy(DENSE.id, [{ key: name.key, value: "Renamed" }]);
    expect(Object.fromEntries(second.edits.map((e) => [e.key, e.value]))).toEqual({ [ctx.key]: 40960, [name.key]: "Renamed" });
    // Setting a key back to its original value takes it out of the copy.
    const third = await s.writeGgufCopy(DENSE.id, [{ key: ctx.key, value: ctx.value }]);
    expect(third.edits).toEqual([{ key: name.key, value: "Renamed" }]);
    // A copy that would equal the original is refused; revert deletes it instead.
    await expect(s.writeGgufCopy(DENSE.id, [{ key: name.key, value: name.value }])).rejects.toThrow();
    expect((await s.getModelDetail(DENSE.id)).copy?.edits).toEqual([{ key: name.key, value: "Renamed" }]);
  });

  it("loads the tuned settings with the file's active tweaks, and accepts a stretched context only with RoPE", async () => {
    const s = createMockService();
    const yarn = { scaling: "yarn" as const, factor: 4, originalContext: 32768 };
    const tuned = MOE.tunedParams as LoadParams;
    await expect(s.loadModel(MOE.id, { ...tuned, contextLength: 131072 })).rejects.toThrow();
    await s.setActivePatch(MOE.sha256, { tweaks: { ...EMPTY_TWEAKS, kvQuant: "q4_0", rope: yarn }, fixes: [] });
    const loaded = await s.loadModel(MOE.id, null);
    const llm = loaded.find((l) => l.slot === "llm");
    expect(llm?.params?.kvCacheV).toBe("q4_0");
    expect(llm?.tuned).toBe(false);
    await expect(s.loadModel(MOE.id, { ...tuned, contextLength: 131072 })).resolves.toBeDefined();
  });

  it("writes metadata edits to a copy only after validating them, and reverts", async () => {
    const s = createMockService();
    const structural = (detailFor(DENSE).gguf ?? []).find((e) => !e.editable);
    if (structural) await expect(s.writeGgufCopy(DENSE.id, [{ key: structural.key, value: structural.value }])).rejects.toThrow();
    const ctxKey = (detailFor(DENSE).gguf ?? []).find((e) => e.key.endsWith(".context_length"))?.key ?? "";
    const copy = await s.writeGgufCopy(DENSE.id, [{ key: ctxKey, value: 40960 }]);
    expect(copy.path).not.toBe(DENSE.path);
    expect((await s.getModelDetail(DENSE.id)).copy?.path).toBe(copy.path);
    await s.revertGgufCopy(DENSE.id);
    expect((await s.getModelDetail(DENSE.id)).copy).toBeNull();
  });

  it("deduplicates once, keeps the chosen copy, never links across drives, and undoes", async () => {
    const s = createMockService();
    const report = await s.scanDisk();
    expect(report.provenance).toBe("illustrative");
    // A believable total, not the 500-file scroll-test list.
    expect(report.totalGiB).toBeGreaterThan(20);
    expect(report.totalGiB).toBeLessThan(320);
    const g = report.groups.find((x) => x.state === "duplicate" && x.copies.length > 2 && x.copies.some((c) => c.volume !== x.copies[0]?.volume));
    expect(g).toBeDefined();
    if (!g) return;
    const lm = g.copies.find((c) => c.source === "lmstudio");
    expect(lm).toBeDefined();
    if (!lm) return;
    await expect(s.deduplicate(g.id, "C:\\not\\in\\group.gguf")).rejects.toThrow();
    const r = await s.deduplicate(g.id, lm.path);
    expect(r.group.kept).toBe(lm.path);
    expect(r.skipped).toBeGreaterThan(0);
    expect(r.reclaimedGiB).toBeCloseTo(g.sizeGiB * (g.copies.filter((c) => c.volume === lm.volume).length - 1), 1);
    await expect(s.deduplicate(g.id, lm.path)).rejects.toThrow();
    const back = await s.unlink(g.id);
    expect(back.state).toBe("duplicate");
    expect(back.kept).toBeNull();
    await expect(s.unlink(g.id)).rejects.toThrow();
    // A group whose only other copy is on another drive has nothing to link.
    const crossOnly = report.groups.find((x) => x.copies.length === 2 && x.copies[0]?.volume !== x.copies[1]?.volume);
    if (crossOnly?.copies[0]) await expect(s.deduplicate(crossOnly.id, crossOnly.copies[0].path)).rejects.toThrow();
  });
});
