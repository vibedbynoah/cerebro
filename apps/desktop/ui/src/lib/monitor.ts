import type {
  HistoryRange,
  MetricsLive,
  MetricsSeries,
  MetricsSnapshot,
  MetricsTotals,
  RequestState,
  RequestSummary,
  SeriesKey,
} from "../service/monitor-types";
import { SERIES_KEYS } from "../service/monitor-types";

/**
 * The Monitor's logic, free of any DOM: header pill text, readout formats, sparkline geometry,
 * alert rules, request filtering, per-session and per-tool breakdowns, and export. Unit-tested in
 * tests/unit/monitor.test.ts; the screen only renders what these return.
 */

// ---- History ranges -------------------------------------------------------------------------

export interface RangeDef {
  id: HistoryRange;
  label: string;
  /** Spoken and exported form. */
  long: string;
  stepS: number;
  points: number;
}

export const HISTORY_RANGES: readonly RangeDef[] = [
  { id: "1m", label: "1 min", long: "the last minute", stepS: 1, points: 60 },
  { id: "15m", label: "15 min", long: "the last 15 minutes", stepS: 5, points: 180 },
  { id: "1h", label: "1 h", long: "the last hour", stepS: 15, points: 240 },
  { id: "24h", label: "24 h", long: "the last 24 hours", stepS: 300, points: 288 },
];

export function isHistoryRange(v: unknown): v is HistoryRange {
  return typeof v === "string" && HISTORY_RANGES.some((r) => r.id === v);
}

export function rangeDef(id: HistoryRange): RangeDef {
  return HISTORY_RANGES.find((r) => r.id === id) ?? (HISTORY_RANGES[0] as RangeDef);
}

// ---- Formats (fixed locale, so readouts never change width between locales) -----------------

const nf = (min: number, max: number) => new Intl.NumberFormat("en-US", { minimumFractionDigits: min, maximumFractionDigits: max });
const F0 = nf(0, 0);
const F1 = nf(1, 1);
const F2 = nf(2, 2);

const ok = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

export const DASH = "–";

export function num(v: number | null | undefined, digits: 0 | 1 | 2 = 0): string {
  if (!ok(v)) return DASH;
  return (digits === 0 ? F0 : digits === 1 ? F1 : F2).format(v);
}

/** 940 -> "940", 88_400 -> "88.4K", 1_234_567 -> "1.2M". */
export function compact(v: number | null | undefined): string {
  if (!ok(v)) return DASH;
  const a = Math.abs(v);
  if (a >= 1e9) return `${F1.format(v / 1e9)}B`;
  if (a >= 1e6) return `${F1.format(v / 1e6)}M`;
  if (a >= 1e4) return `${F1.format(v / 1e3)}K`;
  return F0.format(v);
}

/** Seconds as "0.42 s", "12.3 s", "4 min 05 s". */
export function seconds(v: number | null | undefined): string {
  if (!ok(v)) return DASH;
  if (v < 10) return `${F2.format(v)} s`;
  if (v < 60) return `${F1.format(v)} s`;
  const m = Math.floor(v / 60);
  const s = Math.round(v - m * 60);
  return `${m} min ${String(s).padStart(2, "0")} s`;
}

/** Throughput in MB/s, switching to GB/s from 1,000 MB/s. Returns [value, unit]. */
export function throughput(mbs: number | null | undefined): [string, string] {
  if (!ok(mbs)) return [DASH, "MB/s"];
  if (mbs >= 1000) return [F1.format(mbs / 1000), "GB/s"];
  return [mbs < 10 ? F1.format(mbs) : F0.format(mbs), "MB/s"];
}

export function pct(part: number, whole: number): number {
  if (!(whole > 0) || !Number.isFinite(part)) return 0;
  return Math.min(100, Math.max(0, (100 * part) / whole));
}

export function clockTime(ms: number): string {
  return new Date(ms).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function sinceLabel(ms: number): string {
  return new Date(ms).toLocaleString("en-GB", { weekday: "short", hour: "2-digit", minute: "2-digit" });
}

// ---- Header pill and state card -------------------------------------------------------------

export type PillTone = "idle" | "loading" | "reading" | "generating" | "offline";

export interface PillView {
  tone: PillTone;
  text: string;
  queued: number;
}

/** "Idle" / "Loading model" / "Reading prompt 43%" / "Generating 21.4 tok/s", plus a queue count. */
export function pillView(live: MetricsLive | null, reachable: boolean): PillView {
  if (!reachable) return { tone: "offline", text: "Engine not reachable", queued: 0 };
  if (!live) return { tone: "idle", text: "Connecting", queued: 0 };
  const queued = Math.max(0, Math.floor(live.queued));
  switch (live.state) {
    case "loading":
      return { tone: "loading", text: "Loading model", queued };
    case "reading":
      return { tone: "reading", text: live.promptTotal > 0 ? `Reading prompt ${Math.floor(pct(live.promptRead, live.promptTotal))}%` : "Reading prompt", queued };
    case "generating":
      return { tone: "generating", text: ok(live.decodeTps) ? `Generating ${F1.format(live.decodeTps)} tok/s` : "Generating", queued };
    case "waiting":
      return { tone: "loading", text: "Waiting for your approval", queued };
    case "tool":
      return { tone: "reading", text: "Running a tool", queued };
    default:
      return { tone: "idle", text: "Idle", queued };
  }
}

export interface StateView {
  label: string;
  detail: string;
  /** 0..100: prompt read, or written of the request's maximum. */
  progress: number;
  progressLabel: string;
}

export function stateView(live: MetricsLive, last: RequestSummary | undefined): StateView {
  if (live.state === "loading") return { label: "Loading model", detail: "Reading the model file into memory", progress: 0, progressLabel: "Loading" };
  if (live.state === "waiting") return { label: "Waiting for you", detail: "A tool call needs your approval in Chat", progress: 0, progressLabel: "Waiting" };
  if (live.state === "tool") return { label: "Running a tool", detail: "The model asked for a tool; its result goes back to the model", progress: 0, progressLabel: "Tool" };
  if (live.state === "reading") {
    const p = pct(live.promptRead, live.promptTotal);
    const reused = live.promptReused > 0 ? ` · ${compact(live.promptReused)} reused` : "";
    return {
      label: "Reading prompt",
      detail: `${num(live.promptRead)} of ${num(live.promptTotal)} tokens${reused}`,
      progress: p,
      progressLabel: `Prompt read ${Math.floor(p)}%`,
    };
  }
  if (live.state === "generating") {
    const p = pct(live.generated, live.maxTokens);
    const what = live.phase === "reasoning" ? "Reasoning" : live.phase === "tool-call" ? "Writing a tool call" : "Writing the answer";
    return {
      label: what,
      detail: `${num(live.generated)} of ${num(live.maxTokens)} tokens max · ${num(live.decodeTps, 1)} tok/s`,
      progress: p,
      progressLabel: `Written ${num(live.generated)} of ${num(live.maxTokens)} tokens`,
    };
  }
  const detail = last && last.outputTokens > 0 ? `Last request: ${num(last.outputTokens)} tokens written${ok(last.decodeTps) ? ` at ${F1.format(last.decodeTps)} tok/s` : ""}` : "Requests to the API appear here as they arrive";
  return { label: "Waiting for a request", detail, progress: 0, progressLabel: "Idle" };
}

/** "Since Thu 14:02: 41 requests · 1.2M prompt tokens read at 1,420 tok/s (3.9M reused) · 88.4K
 *  written at 21.3 tok/s". Empty before the first request. */
export function totalsLine(t: MetricsTotals): string {
  if (!(t.requests > 0)) return "";
  const read = Math.max(0, t.promptTokens - t.reusedTokens);
  const pSpeed = t.promptMs > 0 && read > 0 ? ` at ${F0.format(read / (t.promptMs / 1000))} tok/s` : "";
  const oSpeed = t.decodeMs > 0 && t.outputTokens > 0 ? ` at ${F1.format(t.outputTokens / (t.decodeMs / 1000))} tok/s` : "";
  const errors = t.errors > 0 ? ` · ${num(t.errors)} failed` : "";
  return `Since ${sinceLabel(t.since)}: ${num(t.requests)} request${t.requests === 1 ? "" : "s"}${errors} · ${compact(read)} prompt tokens read${pSpeed} (${compact(t.reusedTokens)} reused) · ${compact(t.outputTokens)} written${oSpeed}`;
}

// ---- Sparklines -----------------------------------------------------------------------------

export interface SeriesStats {
  min: number | null;
  max: number | null;
  mean: number | null;
  last: number | null;
  count: number;
}

export function seriesStats(values: ReadonlyArray<number | null>): SeriesStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  for (const v of values) {
    if (!ok(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    count++;
  }
  let last: number | null = null;
  for (let i = values.length - 1; i >= 0; i--) {
    const v = values[i];
    if (ok(v)) {
      last = v;
      break;
    }
  }
  return count ? { min, max, mean: sum / count, last, count } : { min: null, max: null, mean: null, last: null, count: 0 };
}

/** 0.8 -> 1, 13 -> 20, 42 -> 50: a round axis maximum. */
export function niceCeil(v: number): number {
  if (!(v > 0) || !Number.isFinite(v)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
}

export interface SparkGeometry {
  line: string;
  area: string;
}

/**
 * Path data for a sparkline in a `w` x `h` box (y grows down). Nulls break the line (a gap, never a
 * fake zero); a lone point between gaps draws as a short tick so it is still visible. The area
 * closes each run down to the baseline.
 */
export function sparkGeometry(values: ReadonlyArray<number | null>, top: number, w: number, h: number, pad = 2): SparkGeometry {
  const n = values.length;
  if (n === 0 || !(top > 0)) return { line: "", area: "" };
  const dx = n > 1 ? w / (n - 1) : 0;
  const y = (v: number) => pad + (1 - Math.min(1, Math.max(0, v / top))) * (h - pad * 2);
  let line = "";
  let area = "";
  let run: Array<[number, number]> = [];
  const flush = () => {
    if (run.length === 0) return;
    if (run.length === 1) {
      const [x0, y0] = run[0] as [number, number];
      run = [
        [Math.max(0, x0 - dx / 2), y0],
        [Math.min(w, x0 + dx / 2), y0],
      ];
    }
    const seg = run.map(([x, yy], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${yy.toFixed(1)}`).join("");
    line += seg;
    const first = run[0] as [number, number];
    const lastPt = run[run.length - 1] as [number, number];
    area += `${seg}L${lastPt[0].toFixed(1)} ${h}L${first[0].toFixed(1)} ${h}Z`;
    run = [];
  };
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (ok(v)) run.push([i * dx, y(v)]);
    else flush();
  }
  flush();
  return { line, area };
}

/** The sample time of index `i` in a series ending at `end`. */
export function sampleTime(series: Pick<MetricsSeries, "stepS" | "end">, length: number, i: number): number {
  return series.end - (length - 1 - i) * series.stepS * 1000;
}

/** "now", "12 s ago", "4 min ago", "2 h 05 min ago". */
export function agoLabel(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 1) return "now";
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  const h = Math.floor(s / 3600);
  return `${h} h ${String(Math.round((s - h * 3600) / 60)).padStart(2, "0")} min ago`;
}

// ---- Alerts ---------------------------------------------------------------------------------

export type AlertKind = "thermal" | "vram" | "regression";

export interface MonitorAlert {
  kind: AlertKind;
  title: string;
  detail: string;
  since: number;
  resolvedAt: number | null;
}

export interface AlertTracker {
  active: Partial<Record<AlertKind, MonitorAlert>>;
  /** Resolved alerts, newest first, at most RECENT_MAX. */
  recent: MonitorAlert[];
  /** Consecutive samples meeting the raise condition, and the clear condition, per kind. */
  hot: Record<AlertKind, number>;
  cool: Record<AlertKind, number>;
}

/** Samples a condition must hold before an alert is raised, and before it clears. One sample is a
 *  second, so a one-second spike never toasts. */
export const ALERT_RULES = {
  thermal: { raise: 3, clear: 10 },
  vram: { raise: 3, clear: 5, raiseShare: 0.95, clearShare: 0.92 },
  regression: { raise: 10, clear: 5, below: 0.8, recovered: 0.9, minTokens: 64 },
} as const;

const RECENT_MAX = 20;

export function emptyAlerts(): AlertTracker {
  return { active: {}, recent: [], hot: { thermal: 0, vram: 0, regression: 0 }, cool: { thermal: 0, vram: 0, regression: 0 } };
}

interface Condition {
  /** Raise condition holds this sample. */
  hot: boolean;
  /** Clear condition holds this sample (hysteresis: not simply !hot). */
  cool: boolean;
  title: string;
  detail: string;
}

function conditions(s: MetricsSnapshot): Record<AlertKind, Condition> {
  const hw = s.hardware;
  const limit = s.hardwareStatic.gpuThrottleTempC;
  const temp = hw.gpuTempC;
  const thermalNow = hw.throttle.includes("thermal") || (ok(temp) && ok(limit) && temp >= limit);
  const thermalCool = !hw.throttle.includes("thermal") && (!ok(temp) || !ok(limit) || temp < limit - 3);

  const share = ok(hw.vramUsedGiB) && hw.vramTotalGiB > 0 ? hw.vramUsedGiB / hw.vramTotalGiB : null;
  const other = hw.vram?.other;

  const base = s.engine.baseline;
  const live = s.live;
  const generating = live.state === "generating" && live.generated >= ALERT_RULES.regression.minTokens && ok(live.decodeTps);
  const ratioNow = generating && base && base.decodeTps > 0 ? (live.decodeTps as number) / base.decodeTps : null;
  const depth = s.engine.model?.contextUsed ?? 0;

  return {
    thermal: {
      hot: thermalNow,
      cool: thermalCool,
      title: "Graphics card is too hot, so it is slowing down",
      detail: `${num(temp)} °C${ok(limit) ? `, at or above the ${num(limit)} °C point where the driver slows the card` : ""}. Speed drops until it cools; check airflow or lower the power limit.`,
    },
    vram: {
      hot: share !== null && share >= ALERT_RULES.vram.raiseShare,
      cool: share === null || share < ALERT_RULES.vram.clearShare,
      title: "Graphics memory almost full",
      detail: `${num(hw.vramUsedGiB, 1)} of ${num(hw.vramTotalGiB, 1)} GiB in use${ok(other) && other >= 0.5 ? `; other programs hold ${num(other, 1)} GiB` : ""}. If it fills, the driver moves memory to system RAM and speed falls sharply.`,
    },
    regression: {
      hot: ratioNow !== null && ratioNow < ALERT_RULES.regression.below,
      cool: live.state !== "generating" || (ratioNow !== null && ratioNow >= ALERT_RULES.regression.recovered),
      title: "Writing slower than the tuned baseline",
      detail:
        base && ratioNow !== null
          ? `${num(live.decodeTps, 1)} tok/s is ${num((1 - ratioNow) * 100)}% below the ${num(base.decodeTps, 1)} tok/s measured at ${compact(base.depth)} context; this request is at ${compact(depth)}. Long contexts, heat or other graphics card work slow writing.`
          : "Writing speed fell below the tuned baseline.",
    },
  };
}

/**
 * Advances the alert state by one snapshot. Returns the new tracker (the input is not changed) and
 * the alerts raised by this sample, which the caller toasts once each.
 */
export function stepAlerts(prev: AlertTracker, s: MetricsSnapshot): { tracker: AlertTracker; raised: MonitorAlert[] } {
  const next: AlertTracker = { active: { ...prev.active }, recent: prev.recent, hot: { ...prev.hot }, cool: { ...prev.cool } };
  const raised: MonitorAlert[] = [];
  const cond = conditions(s);
  for (const kind of ["thermal", "vram", "regression"] as const) {
    const c = cond[kind];
    const rule = ALERT_RULES[kind];
    next.hot[kind] = c.hot ? next.hot[kind] + 1 : 0;
    next.cool[kind] = c.cool ? next.cool[kind] + 1 : 0;
    const current = next.active[kind];
    if (current) {
      if (next.cool[kind] >= rule.clear) {
        delete next.active[kind];
        next.recent = [{ ...current, resolvedAt: s.at }, ...next.recent].slice(0, RECENT_MAX);
      } else if (c.hot) {
        next.active[kind] = { ...current, detail: c.detail };
      }
    } else if (next.hot[kind] >= rule.raise) {
      const alert: MonitorAlert = { kind, title: c.title, detail: c.detail, since: s.at, resolvedAt: null };
      next.active[kind] = alert;
      raised.push(alert);
    }
  }
  return { tracker: next, raised };
}

// ---- Requests -------------------------------------------------------------------------------

export type RequestFilter = "all" | "running" | "completed" | "failed" | "stream" | "json";

export const REQUEST_FILTERS: ReadonlyArray<{ id: RequestFilter; label: string }> = [
  { id: "all", label: "All requests" },
  { id: "running", label: "Running and queued" },
  { id: "completed", label: "Completed" },
  { id: "failed", label: "Failed or cancelled" },
  { id: "stream", label: "Streamed" },
  { id: "json", label: "JSON responses" },
];

export function isRequestFilter(v: unknown): v is RequestFilter {
  return typeof v === "string" && REQUEST_FILTERS.some((f) => f.id === v);
}

const RUNNING: ReadonlySet<RequestState> = new Set(["queued", "loading", "reading", "generating"]);

export function isRunning(state: RequestState): boolean {
  return RUNNING.has(state);
}

export const STATE_LABELS: Readonly<Record<RequestState, string>> = {
  queued: "Queued",
  loading: "Loading",
  reading: "Reading",
  generating: "Writing",
  completed: "Done",
  error: "Error",
  cancelled: "Cancelled",
};

/** Search box input is capped, so a pasted megabyte cannot stall every keystroke. */
export const QUERY_MAX = 200;

export function filterRequests(list: readonly RequestSummary[], query: string, filter: RequestFilter): RequestSummary[] {
  const q = query.slice(0, QUERY_MAX).trim().toLowerCase();
  return list.filter((r) => {
    if (filter === "running" && !isRunning(r.state)) return false;
    if (filter === "completed" && r.state !== "completed") return false;
    if (filter === "failed" && r.state !== "error" && r.state !== "cancelled") return false;
    if (filter === "stream" && !r.stream) return false;
    if (filter === "json" && r.stream) return false;
    if (!q) return true;
    return `${r.id} ${r.path} ${r.model} ${r.session} ${r.tool ?? ""} ${r.httpStatus ?? ""}`.toLowerCase().includes(q);
  });
}

// ---- Breakdowns -----------------------------------------------------------------------------

export interface BreakdownRow {
  key: string;
  label: string;
  requests: number;
  failed: number;
  promptTokens: number;
  reusedTokens: number;
  outputTokens: number;
  /** Output tokens over writing time across the group's finished requests. */
  decodeTps: number | null;
  meanFirstTokenS: number | null;
  meanWallS: number | null;
}

export const NO_TOOL_LABEL = "No tool (plain answer)";

export function breakdown(list: readonly RequestSummary[], by: "session" | "tool"): BreakdownRow[] {
  const groups = new Map<string, { row: BreakdownRow; decodeS: number; ftSum: number; ftN: number; wallSum: number; wallN: number }>();
  for (const r of list) {
    const key = by === "session" ? r.session : (r.tool ?? "");
    let g = groups.get(key);
    if (!g) {
      g = {
        row: { key, label: by === "tool" && !r.tool ? NO_TOOL_LABEL : key, requests: 0, failed: 0, promptTokens: 0, reusedTokens: 0, outputTokens: 0, decodeTps: null, meanFirstTokenS: null, meanWallS: null },
        decodeS: 0,
        ftSum: 0,
        ftN: 0,
        wallSum: 0,
        wallN: 0,
      };
      groups.set(key, g);
    }
    const row = g.row;
    row.requests++;
    if (r.state === "error" || r.state === "cancelled") row.failed++;
    row.promptTokens += r.promptTokens;
    row.reusedTokens += r.reusedTokens;
    row.outputTokens += r.outputTokens;
    if (ok(r.decodeTps) && r.decodeTps > 0 && r.outputTokens > 0) g.decodeS += r.outputTokens / r.decodeTps;
    if (ok(r.firstTokenS)) {
      g.ftSum += r.firstTokenS;
      g.ftN++;
    }
    if (ok(r.wallS)) {
      g.wallSum += r.wallS;
      g.wallN++;
    }
  }
  const rows = [...groups.values()].map((g) => {
    const decodedTokens = g.decodeS > 0 ? g.row.outputTokens : 0;
    return {
      ...g.row,
      decodeTps: g.decodeS > 0 ? decodedTokens / g.decodeS : null,
      meanFirstTokenS: g.ftN ? g.ftSum / g.ftN : null,
      meanWallS: g.wallN ? g.wallSum / g.wallN : null,
    };
  });
  return rows.sort((a, b) => b.requests - a.requests || a.label.localeCompare(b.label));
}

// ---- Export ---------------------------------------------------------------------------------

export const SERIES_LABELS: Readonly<Record<SeriesKey, string>> = {
  decodeTps: "decode_tok_s",
  prefillTps: "prefill_tok_s",
  gpuUtil: "gpu_util_pct",
  vramUsedGiB: "vram_used_gib",
  gpuTempC: "gpu_temp_c",
  gpuPowerW: "gpu_power_w",
  pcieRxMBs: "pcie_rx_mb_s",
  cpu: "cpu_pct",
  ramUsedGiB: "ram_used_gib",
  diskReadMBs: "disk_read_mb_s",
  expertHitPct: "expert_cache_hit_pct",
  draftAcceptPct: "draft_acceptance_pct",
};

/** One row per sample: ISO time, then every series by its export name. */
export function seriesRows(series: MetricsSeries): Array<Record<string, string | number | null>> {
  const n = Math.max(0, ...SERIES_KEYS.map((k) => series.series[k]?.length ?? 0));
  const rows: Array<Record<string, string | number | null>> = [];
  for (let i = 0; i < n; i++) {
    const row: Record<string, string | number | null> = { time: new Date(sampleTime(series, n, i)).toISOString() };
    for (const k of SERIES_KEYS) {
      const v = series.series[k]?.[i];
      row[SERIES_LABELS[k]] = ok(v) ? Math.round(v * 100) / 100 : null;
    }
    rows.push(row);
  }
  return rows;
}

/**
 * RFC 4180 CSV. A cell that a spreadsheet would run as a formula (starts with = + - @, tab or CR)
 * is prefixed with an apostrophe, unless it is a plain number, so an exported file opened in Excel
 * can never execute anything that reached it from a request or a model name.
 */
export function csvCell(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "";
  let s = v;
  if (/^[=+\-@\t\r]/.test(s) && !/^[+-]?\d+(\.\d+)?$/.test(s)) s = `'${s}`;
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

export function toCsv(rows: ReadonlyArray<Record<string, string | number | null>>): string {
  const first = rows[0];
  if (!first) return "";
  const cols = Object.keys(first);
  return [cols.map(csvCell).join(","), ...rows.map((r) => cols.map((c) => csvCell(r[c])).join(","))].join("\r\n") + "\r\n";
}

export interface ExportPayload {
  format: "cerebro-metrics";
  version: 1;
  provenance: MetricsSnapshot["provenance"];
  range: HistoryRange;
  stepS: number;
  exportedAt: string;
  model: string | null;
  gpu: string | null;
  samples: Array<Record<string, string | number | null>>;
}

export function exportPayload(series: MetricsSeries, range: HistoryRange, snap: Pick<MetricsSnapshot, "provenance" | "engine" | "hardwareStatic">, at: number): ExportPayload {
  return {
    format: "cerebro-metrics",
    version: 1,
    provenance: snap.provenance,
    range,
    stepS: series.stepS,
    exportedAt: new Date(at).toISOString(),
    model: snap.engine.model?.name ?? null,
    gpu: snap.hardwareStatic.gpuName,
    samples: seriesRows(series),
  };
}

/** "cerebro-metrics-15m-20261009-143005.csv": only safe characters, whatever the inputs. */
export function exportFilename(range: HistoryRange, ext: "json" | "csv", at: number): string {
  const d = new Date(at);
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  return `cerebro-metrics-${isHistoryRange(range) ? range : "1m"}-${stamp}.${ext}`;
}
