import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { applyTrainEvent, DEFAULT_JOB, IDLE_RUN, jobErrors, lastLoss, normalizeJob } from "../../ui/src/lib/train";
import { createMockTrainer } from "../../ui/src/mock/mock-train";
import type { TrainEvent, TrainJob } from "../../ui/src/service/train-types";

const ok: TrainJob = { ...DEFAULT_JOB, name: "support bot", datasetPath: String.raw`C:\data\x.jsonl` };

describe("train job validation", () => {
  it("accepts a complete job and names each bad field", () => {
    expect(jobErrors(ok)).toEqual({});
    expect(Object.keys(jobErrors({ ...DEFAULT_JOB }))).toEqual(expect.arrayContaining(["name", "dataset"]));
    expect(jobErrors({ ...ok, model: "../evil" }).model).toBeTruthy();
    expect(jobErrors({ ...ok, model: "a/b/c" }).model).toBeTruthy();
    expect(jobErrors({ ...ok, name: String.raw`x\..\y` }).name).toBeTruthy();
    expect(jobErrors({ ...ok, datasetHf: "tatsu-lab/alpaca" }).dataset).toMatch(/one dataset/);
    expect(jobErrors({ ...ok, datasetPath: "C:/notes.txt" }).dataset).toMatch(/jsonl/);
    expect(jobErrors({ ...ok, lr: 1 }).lr).toBeTruthy();
    expect(jobErrors({ ...ok, rank: Number.NaN }).rank).toBeTruthy();
    expect(jobErrors({ ...ok, batch: 1.5 }).batch).toBeTruthy();
    expect(jobErrors({ ...ok, export: ["q2_k" as never] }).export).toBeTruthy();
  });

  it("normalizes: trims, keeps one dataset kind, orders and dedupes exports", () => {
    const n = normalizeJob({ ...ok, name: "  a  ", datasetHf: "x/y", export: ["f16", "q4_k_m", "f16"] });
    expect(n.name).toBe("a");
    expect(n.datasetHf).toBeNull();
    expect(n.export).toEqual(["q4_k_m", "f16"]);
    expect(normalizeJob({ ...ok, datasetPath: "  ", datasetHf: " tatsu-lab/alpaca " })).toMatchObject({ datasetPath: null, datasetHf: "tatsu-lab/alpaca" });
  });

  it("the IPC validator refuses what the form refuses", () => {
    const src = readFileSync("isolation/index.js", "utf8");
    for (const key of ["name", "model", "datasetPath", "datasetHf", "format", "method", "rank", "alpha", "lr", "epochs", "maxSteps", "maxSeqLen", "batch", "gradAccum", "export"]) {
      expect(src).toContain(`"${key}"`);
    }
    expect(src).toMatch(/trainer_start: \(p\) =>.*isTrainJob\(p\.job\)/);
  });
});

describe("train run view", () => {
  it("folds stages, logs (with gaps), exports and the end", () => {
    const evs: TrainEvent[] = [
      { type: "status", stage: "training", detail: "64 examples" },
      { type: "log", step: 1, maxSteps: 4, loss: 2.5, evalLoss: null, lr: 1e-4, gradNorm: 1, epoch: 0.25 },
      { type: "log", step: 3, maxSteps: 4, loss: 1.1, evalLoss: null, lr: null, gradNorm: 1, epoch: 0.75 },
      { type: "export", quant: "q4_k_m", path: "C:/m/a.gguf" },
      { type: "done", models: ["C:/m/a.gguf"], adapter: null, steps: 4, seconds: 12, stopped: false },
    ];
    const r = evs.reduce(applyTrainEvent, IDLE_RUN);
    expect(r.losses).toEqual([2.5, null, 1.1]);
    expect(r.lr).toBe(1e-4);
    expect(lastLoss(r)).toBe(1.1);
    expect(r).toMatchObject({ phase: "done", step: 4, models: ["C:/m/a.gguf"], seconds: 12 });
    expect(applyTrainEvent(r, { type: "error", message: "CUDA out of memory" })).toMatchObject({ phase: "failed", error: "CUDA out of memory" });
  });
});

describe("mock trainer", () => {
  it("needs setup, runs the real event sequence, refuses a second run and stops on request", async () => {
    const t = createMockTrainer({ stepMs: 0 });
    await expect(t.trainerStart(ok, () => {})).rejects.toThrow(/Set up/);
    const lines: string[] = [];
    await t.trainerSetup((e) => e.type === "setup" && lines.push(e.text));
    expect(lines.length).toBeGreaterThan(2);
    expect((await t.trainerStatus()).installed?.cuda).toBe(true);
    const evs: TrainEvent[] = [];
    await t.trainerStart({ ...ok, maxSteps: 5 }, (e) => evs.push(e));
    expect(evs.filter((e) => e.type === "log")).toHaveLength(5);
    expect(evs.at(-1)).toMatchObject({ type: "done", steps: 5, stopped: false });

    const slow = createMockTrainer({ stepMs: 5 });
    await slow.trainerSetup(() => {});
    const seen: TrainEvent[] = [];
    const p = slow.trainerStart({ ...ok, maxSteps: 500 }, (e) => {
      seen.push(e);
      if (e.type === "log" && e.step === 3) void slow.trainerCancel();
    });
    await expect(slow.trainerStart(ok, () => {})).rejects.toThrow(/already running/);
    await p;
    expect(seen.at(-1)).toMatchObject({ type: "done", stopped: true, models: [] });
  });
});
