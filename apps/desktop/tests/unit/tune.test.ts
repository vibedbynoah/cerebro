import { describe, expect, it } from "vitest";
import {
  announcement,
  answerSeconds,
  applyTuneEvent,
  emptyTuneView,
  finalThirdSlope,
  formatClock,
  isTuneProfile,
  markTerms,
  NOISE_MARGIN,
  objectiveScore,
  pickFit,
  PROFILE_IDS,
  PROFILES,
  reduceTune,
  screeningSpeed,
  SETTING_TERMS,
  tuneProgress,
  type TuneView,
} from "../../ui/src/lib/tune";
import { GLOSSARY } from "../../ui/src/lib/glossary";
import { draftCandidates, validateLoadParams } from "../../ui/src/lib/load-params";
import { MOCK_HARDWARE, MOCK_MODELS } from "../../ui/src/mock/fixtures";
import { MOCK_LIBRARY } from "../../ui/src/mock/library";
import { createMockService } from "../../ui/src/mock/mock-service";
import { buildTuneScript, createTunePlayer, type TuneScript } from "../../ui/src/mock/tune";
import type { LibraryModel, TuneEvent, TuneProfile } from "../../ui/src/service/types";

const lib = (id: string): LibraryModel => {
  const m = MOCK_LIBRARY.find((x) => x.id === id);
  if (!m) throw new Error(`fixture ${id} not found`);
  return m;
};
const MOE = lib("qwen3-coder-30b-a3b-q4km");
const SMALL = lib("gemma-4-e4b-q4km");
const DENSE_BIG = lib("mistral-small-24b-q4km");
const LLMS = MOCK_MODELS.map((m) => lib(m.id));

/** The script's events as the service would number them. */
const eventsOf = (s: TuneScript): TuneEvent[] => s.events.map((e, i) => ({ ...e.body, seq: i + 1, at: e.at }) as TuneEvent);
const script = (m: LibraryModel, p: TuneProfile, extra: { fail?: boolean; integrityCheck?: boolean } = {}) =>
  buildTuneScript(m, p, { runId: "r1", integrityCheck: extra.integrityCheck ?? true, ...(extra.fail ? { fail: true } : {}) });

/** What a view shows, without the clock and sequence bookkeeping. */
const outcome = (v: TuneView) => ({
  status: v.status,
  winner: v.winner,
  quality: v.quality,
  soaks: v.soaks.map((s) => ({ id: s.candidateId, samples: s.samples, result: s.result })),
  candidates: v.candidates.map((c) => ({ id: c.id, status: c.status, screenTps: c.screenTps, result: c.result, soak: c.soak, quality: c.quality, winner: c.winner, samples: c.samples })),
});

const flush = (ms = 30) => new Promise((r) => setTimeout(r, ms));

describe("tune maths", () => {
  it("averages screening speed over every repetition after the first", () => {
    expect(
      screeningSpeed([
        { rep: 1, t: 1, tps: 2 },
        { rep: 2, t: 2, tps: 10 },
        { rep: 3, t: 3, tps: 12 },
      ]),
    ).toBe(11);
    expect(screeningSpeed([{ rep: 1, t: 1, tps: 50 }])).toBeNull();
  });

  it("fits the free-VRAM trend over the final third only", () => {
    const T = 600;
    // Falls fast early, then flat: only the final third counts, so the trend is steady.
    const early = [0, 100, 200, 300, 400, 500, 600].map((t) => ({ t, vramFreeGiB: t < 400 ? 2 - t / 400 : 1 }));
    expect(finalThirdSlope(early, T)).toBeCloseTo(0, 6);
    const falling = [400, 500, 600].map((t) => ({ t, vramFreeGiB: 1 - (t - 400) / 60 / 1024 }));
    expect(finalThirdSlope(falling, T)).toBeCloseTo(-1, 6);
    expect(finalThirdSlope([{ t: 500, vramFreeGiB: 1 }], T)).toBeNull();
    expect(finalThirdSlope([], T)).toBeNull();
  });

  it("formats a run clock and survives nonsense", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(65.9)).toBe("1:05");
    expect(formatClock(3723)).toBe("1:02:03");
    expect(formatClock(Number.NaN)).toBe("0:00");
    expect(formatClock(-5)).toBe("0:00");
  });

  it("ranks long documents on reading and writing together", () => {
    const fastReader = { prefillTps: 2000, decodeTps0: 10, decodeTpsDepth: 8 };
    const fastWriter = { prefillTps: 300, decodeTps0: 14, decodeTpsDepth: 12 };
    expect(objectiveScore("long-docs", fastReader, 32768)).toBeGreaterThan(objectiveScore("long-docs", fastWriter, 32768));
    expect(objectiveScore("chat", fastWriter, 4096)).toBeGreaterThan(objectiveScore("chat", fastReader, 4096));
    expect(answerSeconds(fastReader, 32768)).toBeCloseTo(32768 / 2000 + 500 / 8, 6);
  });

  it("accepts only the three workload profiles", () => {
    for (const p of PROFILE_IDS) expect(isTuneProfile(p)).toBe(true);
    for (const bad of ["Chat", "", "long_docs", null, 3, {}, "__proto__"]) expect(isTuneProfile(bad)).toBe(false);
  });
});

describe("tune event reducer", () => {
  it("ignores events before a run starts and events it has already applied", () => {
    const events = eventsOf(script(MOE, "chat"));
    const v = emptyTuneView();
    applyTuneEvent(v, events[3]!);
    expect(v.run).toBeNull();
    const once = reduceTune(events);
    const twice = reduceTune([...events, ...events.slice(1)]);
    expect(outcome(twice)).toEqual(outcome(once));
  });

  it("a new run's started event clears the previous run", () => {
    const a = eventsOf(script(MOE, "chat"));
    const v = reduceTune(a);
    expect(v.status).toBe("done");
    const b = script(SMALL, "chat");
    applyTuneEvent(v, { ...b.events[0]!.body, seq: a.length + 1, at: 0 } as TuneEvent);
    expect(v.status).toBe("running");
    expect(v.run?.modelId).toBe(SMALL.id);
    expect(v.candidates).toEqual([]);
    expect(v.winner).toBeNull();
  });

  it("progress only moves forward and reaches 100% exactly when the winner is in", () => {
    const events = eventsOf(script(DENSE_BIG, "long-docs"));
    const v = emptyTuneView();
    let last = 0;
    for (const e of events) {
      applyTuneEvent(v, e);
      const p = tuneProgress(v);
      expect(p).toBeGreaterThanOrEqual(last - 1e-9);
      expect(p).toBeLessThanOrEqual(1);
      if (e.kind !== "winner") expect(p).toBeLessThan(1);
      last = p;
    }
    expect(last).toBe(1);
  });

  it("cancelling marks the step in flight as interrupted and keeps every finished one", () => {
    const events = eventsOf(script(MOE, "chat"));
    const mid = events.findIndex((e) => e.kind === "soak-sample");
    const v = reduceTune(events.slice(0, mid + 3));
    applyTuneEvent(v, { kind: "cancelled", seq: 10_000, at: 999 });
    expect(v.status).toBe("cancelled");
    expect(v.activeId).toBeNull();
    expect(v.soaks.at(-1)?.interrupted).toBe(true);
    expect(v.candidates.some((c) => c.soak === "interrupted")).toBe(true);
    expect(v.candidates.filter((c) => c.result !== null).length).toBeGreaterThan(1);
  });
});

describe("illustrative tune scripts", () => {
  for (const m of LLMS) {
    for (const p of PROFILE_IDS) {
      it(`${m.id} / ${p}: a coherent run whose winner passed every check`, () => {
        const s = script(m, p);
        const events = eventsOf(s);
        expect(s.run.provenance).toBe("illustrative");
        expect(events[0]!.kind).toBe("started");
        expect(events.at(-1)!.kind).toBe("winner");
        // The run clock never runs backwards.
        for (let i = 1; i < events.length; i++) expect(events[i]!.at).toBeGreaterThanOrEqual(events[i - 1]!.at);
        // Every reference is to a candidate announced before it.
        const known = new Set<string>();
        for (const e of events) {
          if (e.kind === "candidate") known.add(e.candidate.id);
          else if ("candidateId" in e) expect(known.has(e.candidateId), `${e.kind} ${e.candidateId}`).toBe(true);
        }
        const v = reduceTune(events);
        expect(v.status).toBe("done");
        expect(v.candidates[0]!.candidate.baseline).toBe(true);
        expect(v.candidates.filter((c) => c.candidate.baseline)).toHaveLength(1);
        expect(v.candidates).toHaveLength(s.run.plannedCandidates);
        expect(v.soaks).toHaveLength(s.run.plannedSoaks);
        const w = v.candidates.find((c) => c.winner)!;
        expect(w.soak).toBe("passed");
        const base = v.candidates[0]!;
        if (!w.candidate.baseline) {
          expect(w.quality).toBe("passed");
          // It beat llama.cpp's own settings by more than the noise margin, on this profile's measure.
          expect(objectiveScore(p, w.result!, s.run.depth)).toBeGreaterThan(objectiveScore(p, base.result!, s.run.depth) * (1 + NOISE_MARGIN));
        }
        // A soak passes exactly when its final-third trend is not negative.
        for (const soak of v.soaks) {
          expect(soak.result!.passed).toBe(soak.result!.slopeMiBPerMin >= 0);
          expect(soak.samples.length).toBeGreaterThan(10);
        }
        for (const q of v.quality) expect(q.passed).toBe(q.value <= q.gate);
        // Nothing fails to load except what needs more video memory than is free, and never the baseline.
        expect(base.status).toBe("advanced");
        // Wording the claims policy forbids never appears.
        const text = JSON.stringify(events);
        expect(text).not.toMatch(/bit-exact|byte-identical|identical output|lossless/i);
        if (w.candidate.settings.some((x) => x.name === "Speculation")) expect(v.winner!.reasons.join(" ")).toContain("distribution-preserving");
      });
    }
  }

  it("is deterministic for a model and profile", () => {
    expect(eventsOf(script(MOE, "coding"))).toEqual(eventsOf(script(MOE, "coding")));
  });

  it("tells both stories: a large MoE beats the defaults, a small model that fits keeps them", () => {
    const moe = reduceTune(eventsOf(script(MOE, "chat")));
    expect(moe.candidates.find((c) => c.winner)!.candidate.baseline).toBe(false);
    expect(moe.winner!.reasons.join(" ")).toMatch(/automatic fit placed \d+ of \d+ layers/);
    expect(moe.candidates.some((c) => c.status === "failed")).toBe(true);
    expect(moe.soaks.some((s) => !s.result!.passed)).toBe(true);

    const small = reduceTune(eventsOf(script(SMALL, "chat")));
    expect(small.candidates.find((c) => c.winner)!.candidate.baseline).toBe(true);
    expect(small.winner!.reasons.join(" ")).toContain("keeps llama.cpp's own settings");
  });

  it("says when the integrity check was switched off instead of claiming it ran", () => {
    const off = reduceTune(eventsOf(script(MOE, "chat", { integrityCheck: false })));
    expect(off.run!.integrityCheck).toBe(false);
    expect(off.probe!.integrity).toBe("off");
    expect(reduceTune(eventsOf(script(MOE, "chat"))).probe!.integrity).toBe("verified");
  });

  it("a crash ends the run with a reason, and the step in flight is interrupted", () => {
    const v = reduceTune(eventsOf(script(lib("qwen3-8b-q5km"), "coding", { fail: true })));
    expect(v.status).toBe("failed");
    expect(v.failure).toMatch(/stopped unexpectedly/);
    expect(v.winner).toBeNull();
    expect(v.candidates.some((c) => c.status === "interrupted")).toBe(true);
  });

  it("states every condition a speed needs beside it (claims rule zero)", () => {
    const { run } = script(MOE, "long-docs");
    expect(run.depth).toBe(32768);
    expect(run.sampler).toMatch(/temperature/);
    for (const m of LLMS) {
      for (const p of PROFILE_IDS) {
        const r = script(m, p).run;
        // The machine is the GPU and the CPU and RAM; the build is the exact binary.
        for (const text of [r.gpu, r.cpu, r.modelName, r.quant, r.build, r.backendCommit, r.sampler]) expect(text.trim().length, `${m.id} ${p}`).toBeGreaterThan(0);
        expect(r.cpu, m.id).toMatch(/cores/);
        expect(r.vramTotalGiB).toBeGreaterThan(0);
        expect(r.ramTotalGiB).toBeGreaterThan(0);
        expect(r.depth).toBeGreaterThan(0);
        // Never deeper than the model can hold.
        expect(r.depth, m.id).toBeLessThanOrEqual(m.contextMax);
      }
    }
  });

  it("every setting the engine names has a plain-words definition", () => {
    for (const m of LLMS) {
      for (const p of PROFILE_IDS) {
        const names = script(m, p).events.flatMap((e) => (e.body.kind === "candidate" ? e.body.candidate.settings.map((x) => x.name) : []));
        expect(names.length).toBeGreaterThan(0);
        for (const n of names) {
          expect(SETTING_TERMS[n], `${m.id} ${p}: "${n}"`).toBeDefined();
          expect(GLOSSARY[SETTING_TERMS[n]!]).toBeDefined();
        }
      }
    }
  });

  it("every candidate's settings are valid load settings for that model", () => {
    for (const m of LLMS) {
      const draftIds = new Set(draftCandidates(m, MOCK_LIBRARY).map((d) => d.id));
      for (const p of PROFILE_IDS) {
        const s = script(m, p);
        expect(Object.keys(s.params).length).toBeGreaterThan(0);
        for (const [id, params] of Object.entries(s.params)) expect(validateLoadParams(params, m, MOCK_HARDWARE, draftIds), `${m.id} ${p} ${id}`).toEqual({});
      }
    }
  });

  it("marks each technical term in a winner's reasons once, and leaves already-defined ones plain", () => {
    const lines = ["A q8_0 KV cache needs less.", "The KV cache and flash attention help.", "llama.cpp's automatic fit placed 20 layers."];
    const parts = markTerms(lines);
    const terms = parts.flat().filter((x): x is { term: keyof typeof GLOSSARY; text: string } => typeof x !== "string");
    expect(terms.map((t) => t.term)).toEqual(["kvCache", "flashAttention", "autofit"]);
    expect(parts.map((p) => p.map((x) => (typeof x === "string" ? x : x.text)).join(""))).toEqual(lines);
    expect(markTerms(lines, ["autofit"]).flat().some((x) => typeof x !== "string" && x.term === "autofit")).toBe(false);
    // The noise-margin definition states the same margin the code applies.
    expect(GLOSSARY.noiseMargin.text).toContain(`${Math.round(NOISE_MARGIN * 100)}%`);
  });

  it("the model card's memory line uses the Models screen's estimate at the workload's depth", () => {
    const qwen8 = lib("qwen3-8b-q5km");
    // 5.4 GiB fits with a 4K cache, but not with a 32K one: the tune's dense scenario agrees.
    expect(pickFit(qwen8, MOCK_HARDWARE, "chat")?.fits).toBe(true);
    const long = pickFit(qwen8, MOCK_HARDWARE, "long-docs");
    expect(long?.fits).toBe(false);
    expect(long?.text).toMatch(/^Needs \d+\.\d GiB at 32K/);
    expect(reduceTune(eventsOf(script(qwen8, "long-docs"))).candidates.find((c) => c.winner)?.candidate.baseline).toBe(false);
    expect(pickFit(MOE, MOCK_HARDWARE, "chat")?.fits).toBe(false);
    // No detected machine: no estimate, never one for an invented machine.
    expect(pickFit(qwen8, null, "chat")).toBeNull();
    // A model that holds less than the profile's depth is estimated at its own limit.
    expect(PROFILES["long-docs"].depth).toBe(32768);
    expect(pickFit(DENSE_BIG, MOCK_HARDWARE, "long-docs")?.text).toContain("16K");
  });

  it("a crash is announced once, by the run's alert, not again by the status line", () => {
    const v = reduceTune(eventsOf(script(lib("qwen3-8b-q5km"), "coding", { fail: true })));
    expect(v.status).toBe("failed");
    expect(announcement(v)).toBe("");
  });
});

describe("tune player", () => {
  const view = (events: TuneEvent[][]) => reduceTune(events.flat());

  it("replays the whole run to a late subscriber, then streams live, never interleaved", async () => {
    const player = createTunePlayer({ speed: Number.POSITIVE_INFINITY, hold: "screen" });
    player.start((runId) => buildTuneScript(MOE, "chat", { runId, integrityCheck: true }));
    await flush();
    const got: Array<{ events: TuneEvent[]; replay: boolean }> = [];
    player.subscribe((events, replay) => got.push({ events, replay }));
    await flush();
    expect(got).toHaveLength(1);
    expect(got[0]!.replay).toBe(true);
    expect(got[0]!.events[0]!.kind).toBe("started");
    const v = reduceTune(got[0]!.events);
    expect(v.status).toBe("running");
    expect(v.candidates.find((c) => c.status === "screening")?.id).toBe("c4");
    player.dispose();
  });

  it("allows one tune at a time, and checks run ids", async () => {
    const player = createTunePlayer({ speed: Number.POSITIVE_INFINITY, hold: "soak" });
    const info = player.start((runId) => buildTuneScript(MOE, "chat", { runId, integrityCheck: true }));
    await flush();
    expect(() => player.start((runId) => buildTuneScript(SMALL, "chat", { runId, integrityCheck: true }))).toThrow(/already running/);
    expect(() => player.cancel("tune-999")).toThrow(/Unknown tune run/);
    expect(() => player.cancel({ toString: () => info.runId })).toThrow(/Unknown tune run/);
    expect(() => player.resume(info.runId)).toThrow(/Only a cancelled tune/);
    player.cancel(info.runId);
    expect(() => player.cancel(info.runId)).toThrow(/not running/);
    // A cancelled run can be replaced by a new one.
    const next = player.start((runId) => buildTuneScript(SMALL, "chat", { runId, integrityCheck: true }));
    expect(next.runId).not.toBe(info.runId);
    expect(() => player.resume(info.runId)).toThrow(/Unknown tune run/);
    player.dispose();
  });

  it("cancel then resume ends exactly where an uninterrupted run ends", async () => {
    const straight: TuneEvent[][] = [];
    const p1 = createTunePlayer({ speed: Number.POSITIVE_INFINITY });
    p1.subscribe((e) => straight.push(e));
    await flush();
    p1.start((runId) => buildTuneScript(MOE, "chat", { runId, integrityCheck: true }));
    await flush(60);
    expect(view(straight).status).toBe("done");

    // The same run, frozen mid-soak, cancelled, and resumed with the freeze removed.
    let held: TuneScript | null = null;
    const resumed: TuneEvent[][] = [];
    const p2 = createTunePlayer({ speed: Number.POSITIVE_INFINITY, hold: "soak" });
    p2.subscribe((e) => resumed.push(e));
    await flush();
    const info = p2.start((runId) => (held = buildTuneScript(MOE, "chat", { runId, integrityCheck: true })));
    await flush();
    expect(view(resumed).status).toBe("running");
    p2.cancel(info.runId);
    const mid = view(resumed);
    expect(mid.status).toBe("cancelled");
    expect(mid.soaks.at(-1)!.interrupted).toBe(true);
    for (const e of held!.events) delete e.mark;
    p2.resume(info.runId);
    await flush(60);
    const end = view(resumed);
    expect(end.status).toBe("done");
    expect(outcome(end)).toEqual(outcome(view(straight)));
    // Time spent replaying the interrupted step is on the clock, and the clock never ran backwards.
    const ats = resumed.flat().map((e) => e.at);
    for (let i = 1; i < ats.length; i++) expect(ats[i]!).toBeGreaterThanOrEqual(ats[i - 1]!);
    expect(end.clock).toBeGreaterThan(view(straight).clock);
    p1.dispose();
    p2.dispose();
  });

  it("delivers copies, so one listener cannot change what another sees", async () => {
    const player = createTunePlayer({ speed: Number.POSITIVE_INFINITY });
    const a: TuneEvent[] = [];
    const b: TuneEvent[] = [];
    player.subscribe((e) => {
      for (const x of e) (x as { at: number }).at = -1;
      a.push(...e);
    });
    player.subscribe((e) => b.push(...e));
    await flush();
    player.start((runId) => buildTuneScript(SMALL, "chat", { runId, integrityCheck: true }));
    await flush(60);
    expect(a.length).toBe(b.length);
    expect(b.every((e) => e.at >= 0)).toBe(true);
    player.dispose();
  });

  it("stops delivering after unsubscribe", async () => {
    const player = createTunePlayer({ speed: Number.POSITIVE_INFINITY, hold: "screen" });
    let calls = 0;
    const off = player.subscribe(() => calls++);
    off();
    off();
    player.start((runId) => buildTuneScript(MOE, "chat", { runId, integrityCheck: true }));
    await flush();
    expect(calls).toBe(0);
    player.dispose();
  });
});

describe("mock service tune methods", () => {
  it("validate their arguments like the Rust side will", async () => {
    const service = createMockService({ tuneSpeed: Number.POSITIVE_INFINITY, tuneHold: "screen" });
    const stt = MOCK_LIBRARY.find((m) => m.kind === "stt")!;
    await expect(service.startTune("not-a-model", "chat")).rejects.toThrow();
    await expect(service.startTune(stt.id, "chat")).rejects.toThrow(/text and vision/);
    await expect(service.startTune(MOE.id, "fast" as TuneProfile)).rejects.toThrow(/profile/);
    await expect(service.startTune({} as unknown as string, "chat")).rejects.toThrow();
    await expect(service.cancelTune("nope")).rejects.toThrow();
    await expect(service.resumeTune("nope")).rejects.toThrow();
    const info = await service.startTune(MOE.id, "coding");
    expect(info.profile).toBe("coding");
    expect(info.provenance).toBe("illustrative");
    await expect(service.startTune(SMALL.id, "chat")).rejects.toThrow(/already running/);
    await service.cancelTune(info.runId);
  });

  it("honours the integrity check being switched off in Settings", async () => {
    const service = createMockService({ tuneSpeed: Number.POSITIVE_INFINITY });
    await service.updateSetting("integrityCheck", false);
    const info = await service.startTune(MOE.id, "chat");
    expect(info.integrityCheck).toBe(false);
    const got: TuneEvent[] = [];
    service.subscribeTune((e) => got.push(...e));
    await flush(60);
    expect(reduceTune(got).probe?.integrity).toBe("off");
  });
});

describe("a finished tune is the library's truth", () => {
  it("writes the winner's numbers, settings and state back to the library and the summaries", async () => {
    const service = createMockService({ tuneSpeed: Number.POSITIVE_INFINITY });
    const other = createMockService({ tuneSpeed: Number.POSITIVE_INFINITY });
    const qwen8 = "qwen3-8b-q5km";
    expect((await service.listLibrary()).find((m) => m.id === qwen8)?.tune).toBe("untuned");
    const got: TuneEvent[] = [];
    service.subscribeTune((e) => got.push(...e));
    await service.startTune(qwen8, "coding");
    await flush(80);
    const v = reduceTune(got);
    expect(v.status).toBe("done");
    const win = v.candidates.find((c) => c.winner)!;
    const base = v.candidates.find((c) => c.id === v.winner!.baselineId)!;
    const m = (await service.listLibrary()).find((x) => x.id === qwen8)!;
    expect(m.tune).toBe(win.candidate.baseline ? "defaults-fastest" : "tuned");
    expect(m.tuneResult).toMatchObject({ ...win.result, depth: v.run!.depth, build: v.run!.build, gpu: v.run!.gpu, sampler: v.run!.sampler, provenance: "illustrative" });
    expect(m.tunedParams).not.toBeNull();
    expect(m.tunedAt).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    const summary = (await service.listModels()).find((x) => x.id === qwen8)!;
    expect(summary.tune).toBe(m.tune);
    expect(summary.defaultsTps).toBe(base.result!.decodeTps0);
    expect(summary.tunedTps).toBe(win.result!.decodeTps0);
    // Another service (another window, another test) is untouched.
    expect((await other.listLibrary()).find((x) => x.id === qwen8)?.tune).toBe("untuned");
  });

  it("shows the model as measuring while it runs, and restores its state when cancelled", async () => {
    const service = createMockService({ tuneSpeed: Number.POSITIVE_INFINITY, tuneHold: "soak" });
    const info = await service.startTune(SMALL.id, "chat");
    await flush(60);
    expect((await service.listLibrary()).find((m) => m.id === SMALL.id)?.tune).toBe("tuning");
    await service.cancelTune(info.runId);
    await flush();
    expect((await service.listLibrary()).find((m) => m.id === SMALL.id)?.tune).toBe("defaults-fastest");
    expect((await service.listModels()).find((m) => m.id === SMALL.id)?.tune).toBe("defaults-fastest");
  });
});
