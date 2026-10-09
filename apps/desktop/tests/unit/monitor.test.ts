import { describe, expect, it } from "vitest";
import {
  ALERT_RULES,
  breakdown,
  compact,
  csvCell,
  emptyAlerts,
  exportFilename,
  exportPayload,
  filterRequests,
  HISTORY_RANGES,
  isHistoryRange,
  NO_TOOL_LABEL,
  pillView,
  QUERY_MAX,
  seconds,
  seriesRows,
  seriesStats,
  sparkGeometry,
  stateView,
  stepAlerts,
  throughput,
  toCsv,
  totalsLine,
  type AlertTracker,
} from "../../ui/src/lib/monitor";
import { createMockService, mockOptionsFromQuery } from "../../ui/src/mock/mock-service";
import { SERIES_KEYS, type MetricsLive, type MetricsSnapshot, type RequestSummary } from "../../ui/src/service/monitor-types";

const LIVE_IDLE: MetricsLive = { state: "idle", queued: 0, requestId: null, promptRead: 0, promptTotal: 0, promptReused: 0, generated: 0, maxTokens: 0, decodeTps: null, prefillTps: null, firstTokenS: null, phase: null };

/** A clock the test moves by hand, so the simulation advances exactly as far as asked. */
function clock(start = Date.UTC(2026, 9, 9, 12, 0, 0)) {
  let t = start;
  return { now: () => t, advance: (ms: number) => (t += ms) };
}

function seeded(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function req(over: Partial<RequestSummary>): RequestSummary {
  return {
    id: "req_0001",
    startedAt: 0,
    method: "POST",
    path: "/v1/chat/completions",
    model: "M",
    state: "completed",
    stream: true,
    httpStatus: 200,
    wallS: 2,
    loadS: 0,
    queueS: 0,
    firstTokenS: 0.5,
    promptTokens: 1000,
    reusedTokens: 900,
    outputTokens: 100,
    reasoningTokens: 0,
    decodeTps: 20,
    prefillTps: 500,
    responseFormat: "text",
    session: "Chat · A",
    tool: null,
    ...over,
  };
}

describe("header pill", () => {
  it("names each state the way the brief does", () => {
    expect(pillView(LIVE_IDLE, true)).toEqual({ tone: "idle", text: "Idle", queued: 0 });
    expect(pillView({ ...LIVE_IDLE, state: "reading", promptRead: 430, promptTotal: 1000 }, true).text).toBe("Reading prompt 43%");
    expect(pillView({ ...LIVE_IDLE, state: "generating", decodeTps: 21.44 }, true).text).toBe("Generating 21.4 tok/s");
    expect(pillView({ ...LIVE_IDLE, state: "generating", decodeTps: 21.4, queued: 2 }, true).queued).toBe(2);
    expect(pillView({ ...LIVE_IDLE, state: "loading" }, true).text).toBe("Loading model");
    expect(pillView(LIVE_IDLE, false)).toEqual({ tone: "offline", text: "Engine not reachable", queued: 0 });
    expect(pillView(null, true).text).toBe("Connecting");
  });

  it("never reports more than 100% of a prompt or a negative queue", () => {
    expect(pillView({ ...LIVE_IDLE, state: "reading", promptRead: 2000, promptTotal: 1000 }, true).text).toBe("Reading prompt 100%");
    expect(pillView({ ...LIVE_IDLE, queued: -3 }, true).queued).toBe(0);
  });
});

describe("state card", () => {
  it("shows prompt read while reading and written of max while writing", () => {
    const reading = stateView({ ...LIVE_IDLE, state: "reading", promptRead: 500, promptTotal: 2000, promptReused: 400 }, undefined);
    expect(reading.progress).toBe(25);
    expect(reading.detail).toContain("400 reused");
    const writing = stateView({ ...LIVE_IDLE, state: "generating", generated: 512, maxTokens: 4096, decodeTps: 17.9, phase: "tool-call" }, undefined);
    expect(writing.progress).toBe(12.5);
    expect(writing.label).toBe("Writing a tool call");
    expect(stateView(LIVE_IDLE, req({ outputTokens: 310, decodeTps: 18.2 })).detail).toBe("Last request: 310 tokens written at 18.2 tok/s");
  });
});

describe("formats", () => {
  it("formats compact counts, durations and throughput with a fixed locale", () => {
    expect(compact(940)).toBe("940");
    expect(compact(88_400)).toBe("88.4K");
    expect(compact(1_234_567)).toBe("1.2M");
    expect(compact(null)).toBe("–");
    expect(seconds(0.421)).toBe("0.42 s");
    expect(seconds(12.34)).toBe("12.3 s");
    expect(seconds(245)).toBe("4 min 05 s");
    expect(throughput(9_400)).toEqual(["9.4", "GB/s"]);
    expect(throughput(3.25)).toEqual(["3.3", "MB/s"]);
    expect(throughput(null)).toEqual(["–", "MB/s"]);
  });

  it("writes the totals line only after the first request, reused tokens excluded from read", () => {
    const base = { since: Date.UTC(2026, 9, 6, 14, 2), requests: 0, errors: 0, promptTokens: 0, reusedTokens: 0, promptMs: 0, outputTokens: 0, decodeMs: 0 };
    expect(totalsLine(base)).toBe("");
    const line = totalsLine({ ...base, requests: 41, promptTokens: 5_100_000, reusedTokens: 3_900_000, promptMs: 845_000, outputTokens: 88_400, decodeMs: 4_150_000 });
    expect(line).toMatch(/^Since \w{3} \d\d:\d\d: 41 requests · 1\.2M prompt tokens read at 1,420 tok\/s \(3\.9M reused\) · 88\.4K written at 21\.3 tok\/s$/);
  });
});

describe("sparklines", () => {
  it("breaks the line at gaps instead of drawing a fake zero", () => {
    const g = sparkGeometry([1, 2, null, 3, 4], 4, 100, 40, 0);
    expect(g.line.match(/M/g)).toHaveLength(2);
    expect(g.line).not.toContain("NaN");
    expect(g.area.match(/Z/g)).toHaveLength(2);
  });

  it("draws a lone sample as a short tick and nothing for an empty series", () => {
    expect(sparkGeometry([null, 5, null], 10, 100, 40).line).toMatch(/^M[\d.]+ [\d.]+L[\d.]+ [\d.]+$/);
    expect(sparkGeometry([], 10, 100, 40)).toEqual({ line: "", area: "" });
    expect(sparkGeometry([null, null], 10, 100, 40)).toEqual({ line: "", area: "" });
  });

  it("clamps values above the scale top inside the box", () => {
    const g = sparkGeometry([0, 1000], 10, 100, 40, 2);
    const ys = [...g.line.matchAll(/[ML][\d.]+ ([\d.-]+)/g)].map((m) => Number(m[1]));
    for (const y of ys) {
      expect(y).toBeGreaterThanOrEqual(2);
      expect(y).toBeLessThanOrEqual(38);
    }
  });

  it("summarises a series, skipping gaps", () => {
    expect(seriesStats([null, 3, 1, null, 2, null])).toEqual({ min: 1, max: 3, mean: 2, last: 2, count: 3 });
    expect(seriesStats([null]).count).toBe(0);
  });
});

/** The parts of a snapshot the alert rules read, over a healthy baseline. */
function snap(over: { at: number; temp?: number; throttle?: Array<"thermal" | "power">; vram?: number; live?: Partial<MetricsLive> }): MetricsSnapshot {
  return {
    provenance: "illustrative",
    at: over.at,
    live: { ...LIVE_IDLE, ...over.live },
    hardware: {
      gpuUtil: 50,
      vramUsedGiB: over.vram ?? 6,
      vramTotalGiB: 8,
      vram: { weights: 4, kv: 1, expertCache: 0.5, scratch: 0.3, other: 0.2 },
      gpuTempC: over.temp ?? 60,
      gpuPowerW: 80,
      gpuPowerLimitW: 115,
      pcieRxMBs: 100,
      pcieTxMBs: 4,
      pcieGen: 4,
      pcieGenMax: 4,
      pcieWidth: 8,
      cpu: 40,
      ramUsedGiB: 20,
      ramTotalGiB: 32,
      ramPinnedGiB: 10,
      diskReadMBs: 0,
      diskWriteMBs: 0,
      throttle: over.throttle ?? [],
      gpus: [],
    },
    hardwareStatic: { gpuName: "G", gpuCount: 1, driver: null, cpuName: "C", cores: 8, threads: 16, ramTotalGiB: 32, gpuThrottleTempC: 87 },
    engine: {
      version: "0",
      backend: "llama.cpp",
      build: "b1",
      address: "127.0.0.1:1",
      model: null,
      moe: null,
      speculation: null,
      baseline: { decodeTps: 20, prefillTps: 500, depth: 8192, measuredAt: "2026-09-28" },
    },
    history: { stepS: 1, end: over.at, series: Object.fromEntries(SERIES_KEYS.map((k) => [k, []])) as unknown as MetricsSnapshot["history"]["series"] },
    totals: { since: 0, requests: 0, errors: 0, promptTokens: 0, reusedTokens: 0, promptMs: 0, outputTokens: 0, decodeMs: 0 },
    requests: [],
    requestsKept: 200,
    conversationCache: { enabled: false, slots: 0, parked: 0, budgetGiB: 0, usedGiB: 0, requests: 0, requestsReused: 0, promptTokens: 0, reusedTokens: 0, parks: 0, restores: 0, evictions: 0, lastEvent: null },
  };
}

function run(samples: MetricsSnapshot[], from: AlertTracker = emptyAlerts()) {
  let t = from;
  const raised: string[] = [];
  for (const s of samples) {
    const r = stepAlerts(t, s);
    t = r.tracker;
    raised.push(...r.raised.map((a) => a.kind));
  }
  return { tracker: t, raised };
}

describe("alerts", () => {
  it("raises thermal throttling only after it holds, once, and clears with hysteresis", () => {
    const hot = (i: number) => snap({ at: i * 1000, temp: 88 });
    expect(run([hot(0), hot(1)]).raised).toEqual([]);
    const up = run([0, 1, 2, 3, 4].map(hot));
    expect(up.raised).toEqual(["thermal"]);
    expect(up.tracker.active.thermal?.detail).toContain("88 °C");
    // 86 °C is below the limit but inside the 3 °C band: still active.
    const band = run([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map((i) => snap({ at: i * 1000, temp: 86 })), up.tracker);
    expect(band.tracker.active.thermal).toBeDefined();
    const cool = run(Array.from({ length: ALERT_RULES.thermal.clear }, (_, i) => snap({ at: (20 + i) * 1000, temp: 70 })), band.tracker);
    expect(cool.tracker.active.thermal).toBeUndefined();
    expect(cool.tracker.recent[0]?.kind).toBe("thermal");
    expect(cool.tracker.recent[0]?.resolvedAt).toBe((20 + ALERT_RULES.thermal.clear - 1) * 1000);
  });

  it("treats a driver-reported thermal throttle as throttling whatever the temperature reads", () => {
    expect(run([0, 1, 2].map((i) => snap({ at: i * 1000, temp: 70, throttle: ["thermal"] }))).raised).toEqual(["thermal"]);
    expect(run([0, 1, 2, 3].map((i) => snap({ at: i * 1000, throttle: ["power"] }))).raised).toEqual([]);
  });

  it("raises VRAM pressure at 95% and clears only below 92%", () => {
    const up = run([0, 1, 2].map((i) => snap({ at: i * 1000, vram: 7.7 })));
    expect(up.raised).toEqual(["vram"]);
    const still = run([3, 4, 5, 6, 7, 8].map((i) => snap({ at: i * 1000, vram: 7.5 })), up.tracker);
    expect(still.tracker.active.vram).toBeDefined();
    const down = run([9, 10, 11, 12, 13].map((i) => snap({ at: i * 1000, vram: 7.0 })), still.tracker);
    expect(down.tracker.active.vram).toBeUndefined();
  });

  it("judges speed only while writing past the first tokens, against the tuned baseline", () => {
    const slow = (i: number, generated = 200) => snap({ at: i * 1000, live: { state: "generating", generated, decodeTps: 14 } });
    expect(run(Array.from({ length: 9 }, (_, i) => slow(i))).raised).toEqual([]);
    expect(run(Array.from({ length: 10 }, (_, i) => slow(i))).raised).toEqual(["regression"]);
    // The first tokens of a reply are not judged; neither is a model with no tune.
    expect(run(Array.from({ length: 20 }, (_, i) => slow(i, 10))).raised).toEqual([]);
    const untuned = Array.from({ length: 20 }, (_, i) => {
      const s = slow(i);
      s.engine.baseline = null;
      return s;
    });
    expect(run(untuned).raised).toEqual([]);
    // A request that ends clears it.
    const up = run(Array.from({ length: 10 }, (_, i) => slow(i)));
    const idle = run(Array.from({ length: ALERT_RULES.regression.clear }, (_, i) => snap({ at: (30 + i) * 1000 })), up.tracker);
    expect(idle.tracker.active.regression).toBeUndefined();
  });

  it("never mutates the tracker it was given", () => {
    const t0 = emptyAlerts();
    stepAlerts(t0, snap({ at: 0, temp: 99 }));
    expect(t0).toEqual(emptyAlerts());
  });
});

describe("requests", () => {
  const list = [
    req({ id: "req_0003", state: "generating", wallS: null, session: "Coding agent · app", tool: "edit_file" }),
    req({ id: "req_0002", state: "error", httpStatus: 400, stream: false, session: "API · invoices" }),
    req({ id: "req_0001", state: "cancelled", httpStatus: 499 }),
  ];

  it("filters by state, kind and a case-insensitive query", () => {
    expect(filterRequests(list, "", "running").map((r) => r.id)).toEqual(["req_0003"]);
    expect(filterRequests(list, "", "failed").map((r) => r.id)).toEqual(["req_0002", "req_0001"]);
    expect(filterRequests(list, "", "json").map((r) => r.id)).toEqual(["req_0002"]);
    expect(filterRequests(list, "EDIT_file", "all").map((r) => r.id)).toEqual(["req_0003"]);
    expect(filterRequests(list, "400", "all").map((r) => r.id)).toEqual(["req_0002"]);
    expect(filterRequests(list, "  ", "all")).toHaveLength(3);
  });

  it("caps the query, so a pasted megabyte cannot match or stall", () => {
    const long = `req_0003${"x".repeat(QUERY_MAX * 10)}`;
    expect(filterRequests(list, long, "all")).toEqual([]);
  });

  it("breaks requests down by session and by tool, weighting speed by tokens", () => {
    const rows = breakdown(
      [
        req({ session: "A", outputTokens: 100, decodeTps: 10 }),
        req({ session: "A", outputTokens: 300, decodeTps: 30, firstTokenS: 1.5 }),
        req({ session: "B", state: "error", outputTokens: 0, decodeTps: null, firstTokenS: null, tool: "run_tests" }),
      ],
      "session",
    );
    expect(rows.map((r) => [r.label, r.requests, r.failed])).toEqual([
      ["A", 2, 0],
      ["B", 1, 1],
    ]);
    // 400 tokens over 10 s + 10 s = 20 tok/s, not the mean of 10 and 30.
    expect(rows[0]?.decodeTps).toBeCloseTo(20);
    expect(rows[0]?.meanFirstTokenS).toBeCloseTo(1);
    expect(rows[1]?.decodeTps).toBeNull();
    const tools = breakdown([req({}), req({ tool: "edit_file" }), req({ tool: "edit_file" })], "tool");
    expect(tools.map((r) => r.label)).toEqual(["edit_file", NO_TOOL_LABEL]);
  });
});

describe("export", () => {
  it("neutralises spreadsheet formulas and quotes CSV cells", () => {
    expect(csvCell("=HYPERLINK(\"x\")")).toBe("\"'=HYPERLINK(\"\"x\"\")\"");
    expect(csvCell("+cmd")).toBe("'+cmd");
    expect(csvCell("@SUM(A1)")).toBe("'@SUM(A1)");
    expect(csvCell("-12.5")).toBe("-12.5");
    expect(csvCell(-3)).toBe("-3");
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell("line\nbreak")).toBe('"line\nbreak"');
    expect(csvCell(null)).toBe("");
    expect(csvCell(Number.NaN)).toBe("");
  });

  it("writes one row per sample with an ISO time and every series", () => {
    const series = { stepS: 5, end: Date.UTC(2026, 9, 9, 12, 0, 0), series: Object.fromEntries(SERIES_KEYS.map((k) => [k, [1.234, null]])) as unknown as MetricsSnapshot["history"]["series"] };
    const rows = seriesRows(series);
    expect(rows).toHaveLength(2);
    expect(rows[0]?.["time"]).toBe("2026-10-09T11:59:55.000Z");
    expect(rows[1]?.["decode_tok_s"]).toBeNull();
    expect(rows[0]?.["decode_tok_s"]).toBe(1.23);
    const csv = toCsv(rows);
    expect(csv.split("\r\n")[0]).toBe(["time", "decode_tok_s", "prefill_tok_s", "gpu_util_pct", "vram_used_gib", "gpu_temp_c", "gpu_power_w", "pcie_rx_mb_s", "cpu_pct", "ram_used_gib", "disk_read_mb_s", "expert_cache_hit_pct", "draft_acceptance_pct"].join(","));
    expect(csv.split("\r\n")).toHaveLength(4);
    const payload = exportPayload(series, "15m", snap({ at: 0 }), Date.UTC(2026, 9, 9));
    expect(payload.provenance).toBe("illustrative");
    expect(payload.samples).toHaveLength(2);
  });

  it("builds a safe file name whatever it is given", () => {
    expect(exportFilename("15m", "csv", new Date(2026, 9, 9, 14, 30, 5).getTime())).toBe("cerebro-metrics-15m-20261009-143005.csv");
    expect(exportFilename("../../x" as never, "json", 0)).toMatch(/^cerebro-metrics-1m-\d{8}-\d{6}\.json$/);
  });

  it("knows exactly the four ranges", () => {
    expect(HISTORY_RANGES.map((r) => r.id)).toEqual(["1m", "15m", "1h", "24h"]);
    for (const bad of ["", "2h", "__proto__", 15, null]) expect(isHistoryRange(bad)).toBe(false);
  });
});

describe("mock engine (/metrics)", () => {
  it("advances one sample per second of clock and labels everything illustrative", async () => {
    const c = clock();
    const service = createMockService({ now: c.now, random: seeded(3) });
    const a = await service.getMetrics();
    c.advance(5000);
    const b = await service.getMetrics();
    expect(b.at - a.at).toBe(5000);
    expect(b.provenance).toBe("illustrative");
    expect(b.history.stepS).toBe(1);
    for (const k of SERIES_KEYS) expect(b.history.series[k]).toHaveLength(60);
    expect(b.engine.model?.name).toBe("Qwen3-Coder-30B-A3B-Instruct");
    expect(b.requests.length).toBeGreaterThanOrEqual(16);
    expect(b.totals.requests).toBeGreaterThanOrEqual(16);
  });

  it("stays inside physical bounds over ten simulated minutes", async () => {
    const c = clock();
    const service = createMockService({ now: c.now, random: seeded(7) });
    const states = new Set<string>();
    for (let i = 0; i < 600; i++) {
      c.advance(1000);
      const s = await service.getMetrics();
      states.add(s.live.state);
      const hw = s.hardware;
      expect(hw.vramUsedGiB ?? 0).toBeLessThanOrEqual(hw.vramTotalGiB + 1e-9);
      expect(hw.gpuUtil ?? 0).toBeGreaterThanOrEqual(0);
      expect(hw.gpuUtil ?? 0).toBeLessThanOrEqual(100);
      expect(hw.gpuPowerW ?? 0).toBeLessThanOrEqual((hw.gpuPowerLimitW ?? 0) + 1e-9);
      expect(s.live.promptRead).toBeLessThanOrEqual(s.live.promptTotal);
      if (s.engine.model) expect(s.engine.model.contextUsed).toBeLessThanOrEqual(s.engine.model.contextMax);
      for (const k of SERIES_KEYS) for (const v of s.history.series[k]) if (v !== null) expect(Number.isFinite(v)).toBe(true);
    }
    expect(states).toContain("reading");
    expect(states).toContain("generating");
    expect(states).toContain("idle");
  });

  it("serves each history range at its own step and point count", async () => {
    const service = createMockService({ now: clock().now, random: seeded(1) });
    for (const r of HISTORY_RANGES) {
      const h = await service.getMetricsHistory(r.id);
      expect(h.stepS).toBe(r.stepS);
      for (const k of SERIES_KEYS) expect(h.series[k]).toHaveLength(r.points);
    }
    await expect(service.getMetricsHistory("2h" as never)).rejects.toThrow(/range/);
  });

  it("returns a kept request's capture and rejects malformed or unknown ids", async () => {
    const service = createMockService({ now: clock().now, random: seeded(2) });
    const s = await service.getMetrics();
    const first = s.requests[0];
    expect(first).toBeDefined();
    const d = await service.getRequest(first?.id ?? "");
    expect(d.id).toBe(first?.id);
    expect(() => JSON.parse(d.input)).not.toThrow();
    d.input = "tampered";
    expect((await service.getRequest(first?.id ?? "")).input).not.toBe("tampered");
    for (const bad of ["", "req_", "../req_0001", "req_0001; DROP", "REQ_0001", 42]) await expect(service.getRequest(bad as never)).rejects.toThrow(/Malformed/);
    await expect(service.getRequest("req_99999999")).rejects.toThrow(/no longer kept/);
  });

  it("can fail, show two GPUs, or start with alerts firing", async () => {
    await expect(createMockService({ failMetrics: true }).getMetrics()).rejects.toThrow();
    const multi = await createMockService({ multiGpu: true, now: clock().now }).getMetrics();
    expect(multi.hardware.gpus).toHaveLength(2);
    expect(multi.hardwareStatic.gpuCount).toBe(2);
    expect(mockOptionsFromQuery("?mock=monitor-alerts").monitorScenario).toBe("alerts");

    const c = clock();
    const service = createMockService({ monitorScenario: "alerts", now: c.now, random: seeded(5) });
    let tracker = emptyAlerts();
    const raised = new Set<string>();
    for (let i = 0; i < 25; i++) {
      c.advance(1000);
      const r = stepAlerts(tracker, await service.getMetrics());
      tracker = r.tracker;
      for (const a of r.raised) raised.add(a.kind);
    }
    expect([...raised].sort()).toEqual(["regression", "thermal", "vram"]);
  });

  it("serves an idle engine with no model when none is loaded", async () => {
    const c = clock();
    const service = createMockService({ now: c.now });
    await service.unloadModel("qwen3-coder-30b-a3b-q4km");
    c.advance(3000);
    const s = await service.getMetrics();
    expect(s.engine.model).toBeNull();
    expect(s.live.state).toBe("idle");
    expect(s.hardware.vram).toBeNull();
  });
});
