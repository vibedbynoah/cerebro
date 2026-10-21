import type {
  ConversationCacheStats,
  GpuReading,
  HistoryRange,
  MetricsEngine,
  MetricsHardware,
  MetricsLive,
  MetricsSeries,
  MetricsSnapshot,
  MetricsTotals,
  RequestDetail,
  RequestSummary,
  SeriesKey,
  ThrottleReason,
} from "../service/monitor-types";
import { SERIES_KEYS } from "../service/monitor-types";
import { rangeDef } from "../lib/monitor";
import { contextError, inputBody, responseBody, TEMPLATES, type RequestTemplate } from "./monitor-requests";

/**
 * ILLUSTRATIVE ENGINE for the Monitor: a small simulation of a model serving requests on an example
 * machine, so every card moves the way it would against the real engine's /metrics. Nothing here is
 * a measurement. Time comes from the clock, not a timer: each snapshot advances the simulation to
 * "now" one second at a time, so an unwatched page runs no timers and a slow poll loses nothing.
 * Before the service started, history comes from a deterministic workload pattern (sampleAt), so
 * the 15 min / 1 h / 24 h ranges have something to show from the first second.
 */

/** The loaded model as the simulation needs it (built by the mock service from the library). */
export interface SimModel {
  id: string;
  name: string;
  quant: string;
  params: string;
  architecture: "dense" | "moe";
  contextMax: number;
  kvType: string;
  layers: number;
  gpuLayers: number;
  expertLayersOnCpu: number;
  expertsTotal: number | null;
  expertsPerToken: number | null;
  /** VRAM the model takes, GiB (from the fit estimate of its tuned settings). */
  vram: { weights: number; kv: number; scratch: number };
  /** RAM pinned for weights left on the CPU, GiB. */
  ramWeights: number;
  sizeGiB: number;
  /** Writing speed with an empty context, and the tune's numbers at its working depth. */
  decode0: number;
  baseline: { decodeTps: number; prefillTps: number; depth: number; measuredAt: string } | null;
  speculation: string | null;
}

export type MonitorScenario = "default" | "alerts" | "idle";

export interface MonitorSimOptions {
  random: () => number;
  now?: () => number;
  scenario?: MonitorScenario;
  gpus: ReadonlyArray<{ name: string; vramGiB: number; powerLimitW: number }>;
  cpu: { name: string; cores: number; threads: number };
  ramTotalGiB: number;
  build: string;
  address: string;
}

export interface MonitorSim {
  snapshot(model: SimModel | null): MetricsSnapshot;
  /** Just what the status bar needs, from the same simulation: writing speed now (or of the last
   *  finished request) and VRAM in use. */
  hud(model: SimModel | null): { decodeTps: number | null; vramUsedGiB: number | null };
  history(range: HistoryRange, model: SimModel | null): MetricsSeries;
  request(id: string): RequestDetail | null;
}

/** Requests the engine keeps for the inspector, and the capture cap per body. */
export const REQUESTS_KEPT = 200;
export const CAPTURE_CAP = 256 * 1024;
const RING_MAX = 3600;
const THROTTLE_C = 87;
const OTHER_VRAM_GIB = 0.38;

const r1 = (n: number) => Math.round(n * 10) / 10;
const r2 = (n: number) => Math.round(n * 100) / 100;
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

/** Deterministic 0..1 noise for a time and a channel. */
function noise(t: number, channel: number): number {
  let h = (Math.floor(t) ^ Math.imul(channel + 1, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

type Phase = "idle" | "reading" | "generating";

/** Workload before the service started: 45 s blocks, busier in working hours. */
function pastPhase(tMs: number): { phase: Phase; heat: number } {
  const s = Math.floor(tMs / 1000);
  const block = Math.floor(s / 45);
  const hour = new Date(tMs).getHours();
  const p = hour >= 9 && hour < 19 ? 0.58 : 0.12;
  const busy = (b: number) => noise(b, 7) < p;
  const off = s - block * 45;
  const phase: Phase = !busy(block) ? "idle" : off < 7 ? "reading" : off < 37 ? "generating" : "idle";
  let heat = 0;
  for (let k = 0; k < 6; k++) heat += busy(block - k) ? 1 / 6 : 0;
  return { phase, heat };
}

/** Reading speed, writing speed at a context depth, and the effect of throttling. */
function decodeAt(m: SimModel, depth: number): number {
  const b = m.baseline;
  if (!b || b.depth <= 0 || b.decodeTps >= m.decode0) return m.decode0;
  const k = (m.decode0 / b.decodeTps - 1) / b.depth;
  return m.decode0 / (1 + k * depth);
}

const prefillOf = (m: SimModel) => m.baseline?.prefillTps ?? m.decode0 * 20;

/** Headroom the expert cache leaves free on the card, GiB. */
const EXPERT_HEADROOM_GIB = 1.1;

/** The VRAM expert cache fills what the model leaves free (MoE with experts on the CPU only). */
function expertCacheGiB(m: SimModel, vramTotal: number): number {
  if (m.architecture !== "moe" || m.expertLayersOnCpu === 0) return 0;
  return clamp(vramTotal - m.vram.weights - m.vram.kv - m.vram.scratch - OTHER_VRAM_GIB - EXPERT_HEADROOM_GIB, 0, 1.6);
}

/** Which template comes next: the everyday work in turn, with a failed and a cancelled request now
 *  and then (one in about thirteen each), the way a real session goes. */
const SCHEDULE: readonly number[] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 8, 5, 6, 7, 0, 1, 2, 3, 4, 5, 9, 6, 7];

interface Active {
  rec: RequestDetail;
  t: RequestTemplate;
  arrivedAt: number;
  phase: "queued" | "reading" | "generating";
  promptRead: number;
  generated: number;
  target: number;
  decodeSum: number;
  decodeN: number;
  readStartedAt: number;
}

export function createMonitorSim(o: MonitorSimOptions): MonitorSim {
  const now = o.now ?? Date.now;
  const scenario = o.scenario ?? "default";
  const vramTotal = o.gpus.reduce((a, g) => a + g.vramGiB, 0);
  const powerLimit = o.gpus.reduce((a, g) => a + g.powerLimitW, 0);
  const start = Math.floor(now() / 1000) * 1000;
  let t = start;
  let seq = 1;
  let templateAt = Math.floor(o.random() * SCHEDULE.length);
  let idleLeft = scenario === "idle" ? Number.POSITIVE_INFINITY : 2;
  let modelId: string | null = null;

  // Hardware state that carries from second to second.
  let temp = scenario === "alerts" ? 86 : 52;
  let heatLeft = scenario === "alerts" ? 600 : 0;
  let otherVram = scenario === "alerts" ? 1.32 : OTHER_VRAM_GIB;
  let otherLeft = scenario === "alerts" ? 600 : 0;
  let diskBurst = 0;

  const ring: Record<SeriesKey, Array<number | null>> = Object.fromEntries(SERIES_KEYS.map((k) => [k, []])) as unknown as Record<SeriesKey, Array<number | null>>;
  /** Time of ring[*][0]. */
  let ringStart = start + 1000;

  const requests: RequestDetail[] = [];
  const queue: Active[] = [];
  let active: Active | null = null;

  const totals: MetricsTotals = { since: start - 26 * 60_000, requests: 0, errors: 0, promptTokens: 0, reusedTokens: 0, promptMs: 0, outputTokens: 0, decodeMs: 0 };
  const cc: ConversationCacheStats = {
    enabled: true,
    slots: 4,
    parked: 2,
    budgetGiB: 8,
    usedGiB: 0,
    requests: 0,
    requestsReused: 0,
    promptTokens: 0,
    reusedTokens: 0,
    parks: 3,
    restores: 2,
    evictions: 0,
    lastEvent: null,
  };
  let lastSession = "";

  let live: MetricsLive = idleLive();
  let lastHw: MetricsHardware | null = null;
  let lastModel: SimModel | null = null;

  function idleLive(): MetricsLive {
    return { state: "idle", queued: 0, requestId: null, promptRead: 0, promptTotal: 0, promptReused: 0, generated: 0, maxTokens: 0, decodeTps: null, prefillTps: null, firstTokenS: null, phase: null };
  }

  const nextTemplate = (): RequestTemplate => {
    const tpl = TEMPLATES[SCHEDULE[templateAt % SCHEDULE.length] ?? 0] as RequestTemplate;
    templateAt++;
    return tpl;
  };

  function newRecord(tpl: RequestTemplate, m: SimModel, arrivedAt: number): RequestDetail {
    const id = `req_${String(seq++).padStart(4, "0")}`;
    const vary = 0.85 + o.random() * 0.3;
    const prompt = tpl.error ? tpl.prompt : Math.min(m.contextMax - tpl.maxTokens, Math.round(tpl.prompt * vary));
    const reused = Math.round(prompt * tpl.reusedShare);
    return {
      id,
      startedAt: arrivedAt,
      method: "POST",
      path: tpl.path,
      model: m.name,
      state: "queued",
      stream: tpl.stream,
      httpStatus: null,
      wallS: null,
      loadS: null,
      queueS: null,
      firstTokenS: null,
      promptTokens: prompt,
      reusedTokens: reused,
      outputTokens: 0,
      reasoningTokens: 0,
      decodeTps: null,
      prefillTps: null,
      responseFormat: tpl.responseFormat,
      session: tpl.session,
      tool: tpl.tool,
      input: inputBody(tpl, m.name),
      output: "",
      reasoning: "",
      response: "",
      truncated: { input: false, output: false, reasoning: false, response: false },
      error: null,
    };
  }

  function keep(rec: RequestDetail) {
    requests.unshift(rec);
    if (requests.length > REQUESTS_KEPT) requests.length = REQUESTS_KEPT;
  }

  function finish(a: Active, at: number, m: SimModel) {
    const rec = a.rec;
    const tpl = a.t;
    rec.wallS = r2((at - a.arrivedAt) / 1000);
    if (rec.error === null) {
      rec.state = tpl.cancelAt !== undefined ? "cancelled" : "completed";
      rec.httpStatus = tpl.cancelAt !== undefined ? 499 : 200;
      if (tpl.cancelAt !== undefined) rec.error = `The client closed the stream after ${a.generated} tokens.`;
      rec.decodeTps = a.decodeN ? r1(a.decodeSum / a.decodeN) : null;
      rec.outputTokens = Math.round(a.generated);
      rec.reasoningTokens = Math.min(rec.outputTokens, tpl.reasoning);
      rec.reasoning = tpl.reasoningText;
      rec.output = tpl.cancelAt !== undefined ? tpl.outputText.slice(0, Math.round(tpl.outputText.length * tpl.cancelAt)) : tpl.outputText;
    }
    rec.response = responseBody(tpl, rec.id, m.name, rec.output, rec.promptTokens, rec.outputTokens, true, rec.error && rec.httpStatus === 400 ? rec.error : null);
    // Totals and the conversation cache count every request that reached the model.
    totals.requests++;
    if (rec.state !== "completed") totals.errors++;
    if (rec.httpStatus !== 400) {
      const read = rec.promptTokens - rec.reusedTokens;
      totals.promptTokens += rec.promptTokens;
      totals.reusedTokens += rec.reusedTokens;
      totals.promptMs += rec.prefillTps ? (read / rec.prefillTps) * 1000 : 0;
      totals.outputTokens += rec.outputTokens;
      totals.decodeMs += rec.decodeTps ? (rec.outputTokens / rec.decodeTps) * 1000 : 0;
      cc.requests++;
      cc.promptTokens += rec.promptTokens;
      cc.reusedTokens += rec.reusedTokens;
      if (rec.reusedTokens > 0) cc.requestsReused++;
    }
  }

  /** Records of the session so far, so the inspector and the breakdowns open with content. */
  function seed(m: SimModel) {
    const count = 16;
    for (let i = count; i >= 1; i--) {
      const tpl = nextTemplate();
      const arrived = start - i * 95_000 - Math.floor(o.random() * 30_000);
      const rec = newRecord(tpl, m, arrived);
      const a: Active = { rec, t: tpl, arrivedAt: arrived, phase: "generating", promptRead: rec.promptTokens, generated: 0, target: 0, decodeSum: 0, decodeN: 0, readStartedAt: arrived };
      rec.queueS = i % 5 === 0 ? r2(1 + o.random() * 6) : 0;
      if (tpl.error) {
        rec.state = "error";
        rec.httpStatus = 400;
        rec.error = contextError(rec.promptTokens, m.contextMax);
        finish(a, arrived + 40, m);
      } else {
        const prefill = prefillOf(m) * (0.92 + o.random() * 0.12);
        const read = rec.promptTokens - rec.reusedTokens;
        const readS = read / prefill;
        const out = Math.round((tpl.reasoning + tpl.output) * (tpl.cancelAt ?? 1));
        const decode = decodeAt(m, rec.promptTokens) * (0.94 + o.random() * 0.08);
        rec.prefillTps = Math.round(prefill);
        rec.firstTokenS = r2((rec.queueS ?? 0) + readS + 1 / decode);
        a.generated = out;
        a.decodeSum = decode;
        a.decodeN = 1;
        finish(a, arrived + ((rec.queueS ?? 0) + readS + out / decode) * 1000, m);
      }
      keep(rec);
    }
    lastSession = requests[0]?.session ?? "";
  }

  function begin(a: Active, at: number, m: SimModel) {
    const rec = a.rec;
    rec.queueS = r2((at - a.arrivedAt) / 1000);
    if (a.t.error) {
      rec.state = "error";
      rec.httpStatus = 400;
      rec.error = contextError(rec.promptTokens, m.contextMax);
      finish(a, at + 40, m);
      active = null;
      idleLeft = 2;
      return;
    }
    a.phase = "reading";
    a.readStartedAt = at;
    a.promptRead = rec.reusedTokens;
    rec.state = "reading";
    if (a.t.session !== lastSession && rec.reusedTokens > 0) {
      // Switching conversations: the previous one is parked in RAM and this one's state restored.
      cc.parks++;
      if (cc.parked >= cc.slots) cc.evictions++;
      else cc.parked++;
      cc.restores++;
      cc.parked = Math.max(0, cc.parked - 1);
      cc.lastEvent = { kind: "restored", tokens: rec.reusedTokens, at };
    }
    lastSession = a.t.session;
    active = a;
  }

  function arrive(m: SimModel, at: number, tpl: RequestTemplate = nextTemplate()): Active {
    const rec = newRecord(tpl, m, at);
    keep(rec);
    const out = Math.round((tpl.reasoning + tpl.output) * (tpl.cancelAt ?? 1) * (0.9 + o.random() * 0.2));
    return { rec, t: tpl, arrivedAt: at, phase: "queued", promptRead: 0, generated: 0, target: Math.max(1, out), decodeSum: 0, decodeN: 0, readStartedAt: at };
  }

  function stepOnce(m: SimModel | null) {
    t += 1000;
    const rand = o.random;
    // ---- Work ----
    if (m) {
      if (!active) {
        const waiting = queue.shift();
        if (waiting) begin(waiting, t, m);
        else if (--idleLeft <= 0) {
          begin(arrive(m, t), t, m);
          if (!active) idleLeft = 2;
        }
      }
      const a = active as Active | null;
      const throttled = temp >= THROTTLE_C;
      if (a && a.phase === "reading") {
        const speed = prefillOf(m) * (0.94 + rand() * 0.1) * (throttled ? 0.85 : 1);
        a.promptRead = Math.min(a.rec.promptTokens, a.promptRead + speed);
        a.rec.prefillTps = Math.round(speed);
        if (a.promptRead >= a.rec.promptTokens) {
          a.phase = "generating";
          a.rec.state = "generating";
          a.rec.firstTokenS = r2((t - a.arrivedAt) / 1000 + 0.06);
        }
      } else if (a && a.phase === "generating") {
        const depth = a.rec.promptTokens + a.generated;
        const speed = Math.max(0.5, decodeAt(m, depth) * (0.95 + rand() * 0.08) * (throttled ? 0.8 : 1));
        a.generated = Math.min(a.target, a.generated + speed);
        a.decodeSum += speed;
        a.decodeN++;
        const share = a.generated / Math.max(1, a.target);
        const tpl = a.t;
        a.rec.outputTokens = Math.round(a.generated);
        a.rec.reasoningTokens = Math.min(a.rec.outputTokens, tpl.reasoning);
        a.rec.reasoning = tpl.reasoningText.slice(0, Math.round(tpl.reasoningText.length * Math.min(1, a.generated / Math.max(1, tpl.reasoning))));
        const outShare = tpl.reasoning > 0 ? Math.max(0, (a.generated - tpl.reasoning) / Math.max(1, a.target - tpl.reasoning)) : share;
        a.rec.output = tpl.outputText.slice(0, Math.round(tpl.outputText.length * Math.min(1, outShare) * (tpl.cancelAt ?? 1)));
        a.rec.decodeTps = r1(a.decodeSum / a.decodeN);
        if (a.generated >= a.target) {
          finish(a, t, m);
          active = null;
          idleLeft = 3 + Math.floor(rand() * 6);
        }
      }
      // Another client arrives while one is running: it waits its turn.
      if (active && queue.length < 2 && scenario !== "idle" && rand() < 0.05) queue.push(arrive(m, t));
    }

    // ---- Live view ----
    const a = active as Active | null;
    if (!m || !a) {
      live = { ...idleLive(), queued: queue.length };
    } else {
      const depth = a.rec.promptTokens + a.generated;
      const reasoningLeft = a.t.reasoning - a.generated;
      live = {
        state: a.phase === "generating" ? "generating" : "reading",
        queued: queue.length,
        requestId: a.rec.id,
        promptRead: Math.round(a.promptRead),
        promptTotal: a.rec.promptTokens,
        promptReused: a.rec.reusedTokens,
        generated: Math.round(a.generated),
        maxTokens: a.t.maxTokens,
        decodeTps: a.phase === "generating" ? r1(Math.max(0.5, decodeAt(m, depth) * (0.95 + o.random() * 0.08) * (temp >= THROTTLE_C ? 0.8 : 1))) : null,
        prefillTps: a.rec.prefillTps,
        firstTokenS: a.rec.firstTokenS,
        phase: a.phase !== "generating" ? null : reasoningLeft > 0 ? "reasoning" : a.t.tool ? "tool-call" : "answer",
      };
    }

    // ---- Hardware ----
    const phase: Phase = live.state === "generating" ? "generating" : live.state === "reading" ? "reading" : "idle";
    if (scenario === "default" && heatLeft <= 0 && phase !== "idle" && o.random() < 1 / 600) heatLeft = 45;
    if (scenario === "default" && otherLeft <= 0 && o.random() < 1 / 900) otherLeft = 40;
    heatLeft = Math.max(0, heatLeft - 1);
    otherLeft = Math.max(0, otherLeft - 1);
    const otherTarget = otherLeft > 0 ? 1.32 : OTHER_VRAM_GIB;
    otherVram += (otherTarget - otherVram) * 0.5;
    const tempTarget = (phase === "reading" ? 76 : phase === "generating" ? 71 : 46) + (heatLeft > 0 ? 18 : 0);
    temp += (tempTarget - temp) * 0.08 + (o.random() - 0.5) * 0.4;
    // Now and then Windows trims the mapped model file and a few pages are read back from disk.
    if (diskBurst > 0) diskBurst--;
    else if (m && o.random() < 1 / 150) diskBurst = 2;
    lastHw = hardware(m, phase);
    lastModel = m;
    record(lastHw, live, m);
  }

  /**
   * Illustrative readings per phase. Levels are shares of the card's limits chosen to look like a
   * MoE model with experts on the CPU: reading a prompt keeps the GPU near full and streams experts
   * over PCIe; writing leans on the CPU (its experts) with the GPU partly idle; idle is near zero.
   * Each reading gets a few percent of noise so the charts move like real telemetry.
   */
  function hardware(m: SimModel | null, phase: Phase): MetricsHardware {
    const rand = o.random;
    const busy = m !== null && phase !== "idle";
    const util = phase === "reading" ? 94 + rand() * 5 : phase === "generating" ? 70 + rand() * 12 : 1 + rand() * 3;
    const power = Math.min(powerLimit, (phase === "reading" ? 0.95 : phase === "generating" ? 0.74 : 0.08) * powerLimit * (0.96 + rand() * 0.06));
    const pcie = phase === "reading" ? 8_600 + rand() * 1_400 : phase === "generating" ? (m?.expertLayersOnCpu ? 1_500 + rand() * 600 : 40 + rand() * 30) : 1 + rand() * 3;
    const cpu = phase === "reading" ? 30 + rand() * 10 : phase === "generating" ? (m?.expertLayersOnCpu ? 62 + rand() * 14 : 18 + rand() * 6) : 2 + rand() * 3;
    const expertCache = m ? expertCacheGiB(m, vramTotal) : 0;
    const split = m ? { weights: m.vram.weights, kv: m.vram.kv, expertCache: r2(expertCache), scratch: m.vram.scratch, other: r2(otherVram) } : null;
    const used = split ? r2(split.weights + split.kv + split.expertCache + split.scratch + split.other) : r2(otherVram);
    const tempC = r1(temp);
    const throttle: ThrottleReason[] = [];
    if (tempC >= THROTTLE_C) throttle.push("thermal");
    if (power >= powerLimit * 0.98) throttle.push("power");
    const ramUsed = 9.6 + (m ? m.ramWeights : 0) + rand() * 0.08;
    const gpus: GpuReading[] = o.gpus.map((g, i) => {
      const share = g.vramGiB / vramTotal;
      const lag = i === 0 ? 1 : 0.86;
      return {
        index: i,
        name: g.name,
        util: r1(clamp(util * lag, 0, 100)),
        memUsedGiB: r2(used * share),
        memTotalGiB: g.vramGiB,
        tempC: r1(tempC - i * 4),
        powerW: Math.round(power * share * lag),
        powerLimitW: g.powerLimitW,
        pcieRxMBs: Math.round(pcie * share),
      };
    });
    return {
      gpuUtil: r1(util),
      vramUsedGiB: used,
      vramTotalGiB: vramTotal,
      vram: split,
      gpuTempC: tempC,
      gpuPowerW: Math.round(power),
      gpuPowerLimitW: powerLimit,
      pcieRxMBs: Math.round(pcie),
      pcieTxMBs: Math.round(pcie * 0.04),
      pcieGen: busy ? 4 : 1,
      pcieGenMax: 4,
      pcieWidth: o.gpus.length > 1 ? 16 : 8,
      cpu: r1(cpu),
      ramUsedGiB: r2(ramUsed),
      ramTotalGiB: o.ramTotalGiB,
      ramPinnedGiB: m ? r2(m.ramWeights) : 0,
      diskReadMBs: r1(diskBurst > 0 ? 180 + rand() * 60 : rand() * 0.6),
      diskWriteMBs: r1(rand() * 1.2),
      throttle,
      gpus,
    };
  }

  function moeHit(m: SimModel | null, phase: Phase): number | null {
    if (!m || m.architecture !== "moe" || m.expertLayersOnCpu === 0 || phase === "idle") return null;
    return r1(28 + o.random() * 8);
  }
  function draftAccept(m: SimModel | null, phase: Phase): number | null {
    if (!m || !m.speculation || phase !== "generating") return null;
    return r1(52 + o.random() * 12);
  }

  let lastHit: number | null = null;
  let lastAccept: number | null = null;

  function record(hw: MetricsHardware, l: MetricsLive, m: SimModel | null) {
    const phase: Phase = l.state === "generating" ? "generating" : l.state === "reading" ? "reading" : "idle";
    lastHit = moeHit(m, phase) ?? lastHit;
    lastAccept = draftAccept(m, phase) ?? lastAccept;
    const sample: Record<SeriesKey, number | null> = {
      decodeTps: l.decodeTps,
      prefillTps: l.state === "reading" ? l.prefillTps : null,
      gpuUtil: hw.gpuUtil,
      vramUsedGiB: hw.vramUsedGiB,
      gpuTempC: hw.gpuTempC,
      gpuPowerW: hw.gpuPowerW,
      pcieRxMBs: hw.pcieRxMBs,
      cpu: hw.cpu,
      ramUsedGiB: hw.ramUsedGiB,
      diskReadMBs: hw.diskReadMBs,
      expertHitPct: moeHit(m, phase),
      draftAcceptPct: draftAccept(m, phase),
    };
    for (const k of SERIES_KEYS) ring[k].push(sample[k]);
    if (ring.decodeTps.length > RING_MAX) {
      for (const k of SERIES_KEYS) ring[k].shift();
      ringStart += 1000;
    }
  }

  /** One second before the service started, from the workload pattern. */
  function pastSample(m: SimModel | null, tMs: number): Record<SeriesKey, number | null> {
    const { phase: p0, heat } = pastPhase(tMs);
    const phase: Phase = m ? p0 : "idle";
    const n = (c: number) => noise(tMs / 1000, c);
    const moe = m && m.architecture === "moe" && m.expertLayersOnCpu > 0;
    const vramUsed = m ? m.vram.weights + m.vram.kv + m.vram.scratch + OTHER_VRAM_GIB + expertCacheGiB(m, vramTotal) : OTHER_VRAM_GIB;
    return {
      decodeTps: m && phase === "generating" ? r1(decodeAt(m, 6000 + n(1) * 6000) * (0.95 + n(2) * 0.08)) : null,
      prefillTps: m && phase === "reading" ? Math.round(prefillOf(m) * (0.94 + n(3) * 0.1)) : null,
      gpuUtil: r1(phase === "reading" ? 94 + n(4) * 5 : phase === "generating" ? 70 + n(4) * 12 : 1 + n(4) * 3),
      vramUsedGiB: r2(vramUsed + (n(5) - 0.5) * 0.02),
      gpuTempC: r1(46 + (m ? heat * 24 : 0) + (n(6) - 0.5) * 1.2),
      gpuPowerW: Math.round((phase === "reading" ? 0.95 : phase === "generating" ? 0.74 : 0.08) * powerLimit * (0.96 + n(7) * 0.06)),
      pcieRxMBs: Math.round(phase === "reading" ? 8_600 + n(8) * 1_400 : phase === "generating" ? (moe ? 1_500 + n(8) * 600 : 40 + n(8) * 30) : 1 + n(8) * 3),
      cpu: r1(phase === "reading" ? 30 + n(9) * 10 : phase === "generating" ? (moe ? 62 + n(9) * 14 : 18 + n(9) * 6) : 2 + n(9) * 3),
      ramUsedGiB: r2(9.6 + (m ? m.ramWeights : 0) + n(10) * 0.08),
      diskReadMBs: r1(n(11) < 0.01 ? 180 + n(12) * 60 : n(12) * 0.6),
      expertHitPct: moe && phase !== "idle" ? r1(28 + n(13) * 8) : null,
      draftAcceptPct: m?.speculation && phase === "generating" ? r1(52 + n(14) * 12) : null,
    };
  }

  function valueAt(m: SimModel | null, k: SeriesKey, tMs: number): number | null {
    if (tMs >= ringStart) {
      const i = Math.round((tMs - ringStart) / 1000);
      if (i < ring[k].length) return ring[k][i] ?? null;
    }
    return pastSample(m, tMs)[k];
  }

  function advance(m: SimModel | null) {
    if (m?.id !== modelId) {
      // A different model in the slot (or none): its history starts here, as the engine's would.
      if (m && modelId === null && requests.length === 0) seed(m);
      if (m === null) {
        active = null;
        queue.length = 0;
      }
      modelId = m?.id ?? null;
      if (scenario === "alerts" && m && !active) {
        // Mid-way through a long agent request on a hot card, with another program holding VRAM.
        // The agent's longest turn: deep context, so writing is slowest here.
        const a = arrive(m, t - 20_000, TEMPLATES[3] as RequestTemplate);
        a.target = 900;
        begin(a, t - 18_000, m);
        const aa = active as Active | null;
        if (aa) {
          aa.phase = "generating";
          aa.rec.state = "generating";
          aa.promptRead = aa.rec.promptTokens;
          aa.rec.prefillTps = Math.round(prefillOf(m));
          aa.rec.firstTokenS = 2.4;
          aa.generated = 240;
        }
      }
    }
    const target = Math.floor(now() / 1000) * 1000;
    let steps = 0;
    while (t < target && steps < 120) {
      stepOnce(m);
      steps++;
    }
    if (t < target) {
      // Asleep for minutes (a suspended laptop): skip ahead instead of replaying it all.
      const gap = target - t;
      t = target;
      for (const k of SERIES_KEYS) ring[k].push(...Array.from({ length: Math.min(RING_MAX, Math.round(gap / 1000)) }, () => null));
      while (ring.decodeTps.length > RING_MAX) {
        for (const k of SERIES_KEYS) ring[k].shift();
        ringStart += 1000;
      }
    }
    if (!lastHw) {
      lastHw = hardware(m, "idle");
      lastModel = m;
    }
  }

  function engine(m: SimModel | null): MetricsEngine {
    const lastDone = requests.find((r) => r.state === "completed");
    const used = active ? Math.round(active.rec.promptTokens + active.generated) : lastDone ? lastDone.promptTokens + lastDone.outputTokens : 0;
    const expertGiB = lastHw?.vram?.expertCache ?? 0;
    const expertSize = m && m.expertsTotal ? (m.sizeGiB * 0.92 * 0.85) / (m.layers * m.expertsTotal) : 0;
    return {
      version: "0.1.0",
      backend: "cerebro",
      build: o.build,
      address: o.address,
      model: m
        ? {
            id: m.id,
            name: m.name,
            quant: m.quant,
            params: m.params,
            architecture: m.architecture,
            contextMax: m.contextMax,
            contextUsed: Math.min(m.contextMax, used),
            kvType: m.kvType,
            layers: m.layers,
            gpuLayers: m.gpuLayers,
            expertLayersOnCpu: m.expertLayersOnCpu,
          }
        : null,
      moe:
        m && m.architecture === "moe" && m.expertsTotal && m.expertsPerToken
          ? { expertsPerToken: m.expertsPerToken, expertsTotal: m.expertsTotal, cacheSlots: expertSize > 0 ? Math.floor(expertGiB / expertSize) : 0, cacheHitPct: m.expertLayersOnCpu > 0 ? lastHit : null, channel: "experimental" }
          : null,
      speculation: m && m.speculation ? { method: m.speculation, acceptancePct: lastAccept, tokensPerPass: lastAccept === null ? null : r2(1 + (lastAccept / 100) * 2.2), channel: "beta" } : null,
      baseline: m?.baseline ?? null,
    };
  }

  const copyDetail = (r: RequestDetail): RequestDetail => ({ ...r, truncated: { ...r.truncated } });
  const summary = (r: RequestDetail): RequestSummary => {
    const { input: _i, output: _o, reasoning: _r, response: _p, truncated: _t, error: _e, ...rest } = r;
    return { ...rest };
  };

  function series60(m: SimModel | null): MetricsSeries {
    const end = t;
    const series = Object.fromEntries(SERIES_KEYS.map((k) => [k, Array.from({ length: 60 }, (_, i) => valueAt(m, k, end - (59 - i) * 1000))])) as unknown as Record<SeriesKey, Array<number | null>>;
    return { stepS: 1, end, series };
  }

  return {
    snapshot(m) {
      advance(m);
      const hw = lastHw as MetricsHardware;
      cc.usedGiB = r2(cc.parked * 1.4);
      return {
        provenance: "illustrative",
        at: t,
        live: { ...live },
        hardware: { ...hw, vram: hw.vram ? { ...hw.vram } : null, throttle: [...hw.throttle], gpus: hw.gpus.map((g) => ({ ...g })) },
        hardwareStatic: {
          gpuName: o.gpus.length > 1 ? `${o.gpus.length} × ${o.gpus[0]?.name ?? "GPU"}` : (o.gpus[0]?.name ?? null),
          gpuCount: o.gpus.length,
          driver: "Example driver 581.42",
          cpuName: o.cpu.name,
          cores: o.cpu.cores,
          threads: o.cpu.threads,
          ramTotalGiB: o.ramTotalGiB,
          gpuThrottleTempC: THROTTLE_C,
        },
        engine: engine(lastModel ?? m),
        history: series60(m),
        totals: { ...totals },
        requests: requests.map(summary),
        requestsKept: REQUESTS_KEPT,
        conversationCache: { ...cc, lastEvent: cc.lastEvent ? { ...cc.lastEvent } : null },
      };
    },

    hud(m) {
      advance(m);
      const last = requests.find((r) => r.state === "completed" && r.decodeTps !== null);
      return { decodeTps: live.decodeTps ?? last?.decodeTps ?? null, vramUsedGiB: lastHw?.vramUsedGiB ?? null };
    },

    history(range, m) {
      advance(m);
      const def = rangeDef(range);
      if (def.stepS === 1) return series60(m);
      const step = def.stepS * 1000;
      const end = Math.floor(t / step) * step;
      const out = Object.fromEntries(SERIES_KEYS.map((k) => [k, [] as Array<number | null>])) as unknown as Record<SeriesKey, Array<number | null>>;
      for (let j = 0; j < def.points; j++) {
        const bEnd = end - (def.points - 1 - j) * step;
        const bStart = bEnd - step;
        // Real seconds where the ring has them; otherwise up to 10 evenly spaced samples of the
        // past workload, which is plenty for a mean.
        const real = bEnd > ringStart;
        const times: number[] = [];
        if (real) for (let x = Math.max(bStart + 1000, ringStart); x <= Math.min(bEnd, t); x += 1000) times.push(x);
        if (times.length === 0) {
          const sub = Math.min(10, def.stepS);
          for (let s = 1; s <= sub; s++) times.push(bStart + Math.round((s * step) / sub));
        }
        const past = times.map((x) => (x < ringStart ? pastSample(m, x) : null));
        for (const k of SERIES_KEYS) {
          let sum = 0;
          let n = 0;
          times.forEach((x, idx) => {
            const v = x < ringStart ? (past[idx] as Record<SeriesKey, number | null>)[k] : valueAt(m, k, x);
            if (v !== null) {
              sum += v;
              n++;
            }
          });
          out[k].push(n ? r2(sum / n) : null);
        }
      }
      return { stepS: def.stepS, end, series: out };
    },

    request(id) {
      const r = requests.find((x) => x.id === id);
      if (!r) return null;
      const d = copyDetail(r);
      for (const key of ["input", "output", "reasoning", "response"] as const) {
        if (d[key].length > CAPTURE_CAP) {
          d[key] = d[key].slice(0, CAPTURE_CAP);
          d.truncated[key] = true;
        }
      }
      return d;
    },
  };
}
