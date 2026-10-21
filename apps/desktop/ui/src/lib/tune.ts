import type { HardwareProfile, LibraryModel, TuneCandidate, TuneEvent, TuneMeasurement, TunePhase, TuneProfile, TuneRunInfo } from "../service/types";
import { estimateFit } from "./fit";
import type { GlossaryKey } from "./glossary";
import { startingParams } from "./load-params";

/**
 * The Tune screen's state, rebuilt from the service's event stream by one reducer. Because the view
 * is a pure function of the events, leaving the screen and coming back (a replay from the first
 * event) and cancelling then resuming (the interrupted step is replayed) both land on the same view.
 */

/** Copy with technical terms marked: plain strings, and terms rendered with their definition. */
export type TermText = ReadonlyArray<string | { term: GlossaryKey; text: string }>;

export interface ProfileMeta {
  label: string;
  /** Who it is for, in one short phrase. */
  summary: string;
  /** Working context depth, tokens. */
  depth: number;
  /** What "fastest" means for it, in plain words. */
  objective: TermText;
}

export const PROFILE_IDS: readonly TuneProfile[] = ["chat", "coding", "long-docs"];

export const PROFILES: Readonly<Record<TuneProfile, ProfileMeta>> = {
  chat: {
    label: "Chat",
    summary: "Questions, drafts and quick replies.",
    depth: 4096,
    objective: ["Fastest ", { term: "decode", text: "writing speed" }, " with 4K tokens of conversation in view."],
  },
  coding: {
    label: "Coding",
    summary: "An agent reading and editing code.",
    depth: 16384,
    objective: ["Fastest ", { term: "decode", text: "writing speed" }, " at 16K tokens, where repeated code can make ", { term: "ngram", text: "n-gram speculation" }, " pay."],
  },
  "long-docs": {
    label: "Long documents",
    summary: "Reports, contracts and manuals.",
    depth: 32768,
    objective: ["Shortest time to read a 32K-token document (", { term: "prefill", text: "prompt reading" }, ") and write a 500-token answer."],
  },
};

/**
 * Every setting name a tune reports, mapped to its plain-words definition, so a setting is never
 * shown without one. The engine's names are a fixed vocabulary; a unit test fails when the mock
 * engine emits a name missing here, and the UI shows an unknown name as plain text, never broken.
 */
export const SETTING_TERMS: Readonly<Record<string, GlossaryKey>> = {
  Placement: "autofit",
  "GPU layers": "offload",
  "Experts on CPU": "moeCpu",
  "KV cache": "kvCache",
  "Flash attention": "flashAttention",
  Threads: "threads",
  "Micro-batch": "batch",
  Speculation: "ngram",
};

export function settingTerm(name: string): GlossaryKey | null {
  return Object.hasOwn(SETTING_TERMS, name) ? SETTING_TERMS[name]! : null;
}

/**
 * Phrases in an engine-written explanation (a winner's reasons) that get their definition on first
 * use. Matched case-insensitively on word boundaries; longer phrases first.
 */
const REASON_TERMS: ReadonlyArray<readonly [RegExp, GlossaryKey]> = [
  [/\bautomatic (?:fit|placement|settings)\b/i, "autofit"],
  [/\bprompt-lookup speculation\b/i, "ngram"],
  [/\bKV cache\b/i, "kvCache"],
  [/\bflash attention\b/i, "flashAttention"],
  [/\bn-gram speculation\b/i, "ngram"],
  [/\bexpert weights\b/i, "moeCpu"],
  [/\bmicro-batches\b/i, "batch"],
  [/\bthreads\b/i, "threads"],
  [/\bnoise margin\b/i, "noiseMargin"],
  [/\bquality gate\b/i, "klDiv"],
  [/\bdistribution-preserving\b/i, "speculative"],
];

/**
 * Splits explanations into text and terms: each known phrase is marked the first time it appears
 * across all of `lines` (later mentions stay plain, so a definition is offered once, not per line).
 * Terms the surrounding copy already defines are passed in `alreadyDefined` and left plain.
 */
export function markTerms(lines: readonly string[], alreadyDefined: readonly GlossaryKey[] = []): TermText[] {
  const used = new Set<GlossaryKey>(alreadyDefined);
  return lines.map((line) => {
    const out: Array<string | { term: GlossaryKey; text: string }> = [];
    let rest = line;
    for (;;) {
      let best: { index: number; length: number; key: GlossaryKey } | null = null;
      for (const [re, key] of REASON_TERMS) {
        if (used.has(key)) continue;
        const m = re.exec(rest);
        if (m && (best === null || m.index < best.index)) best = { index: m.index, length: m[0].length, key };
      }
      if (!best) break;
      if (best.index > 0) out.push(rest.slice(0, best.index));
      out.push({ term: best.key, text: rest.slice(best.index, best.index + best.length) });
      used.add(best.key);
      rest = rest.slice(best.index + best.length);
    }
    if (rest) out.push(rest);
    return out;
  });
}

/** The working depth a tune measures at: the profile's, capped at what the model can hold. */
export function tuneDepth(profile: TuneProfile, contextMax: number): number {
  return Math.min(PROFILES[profile].depth, contextMax);
}

/**
 * The model card's memory line on the Tune screen: whether the whole model plus its KV cache at the
 * workload's depth fits in video memory, from the same estimate the Models screen uses (lib/fit.ts),
 * so the two screens never disagree. Null when this machine was not detected: nothing is estimated
 * for an invented machine.
 */
export function pickFit(
  model: Pick<LibraryModel, "sizeGiB" | "architecture" | "arch" | "projector" | "contextMax">,
  hw: Pick<HardwareProfile, "cpu" | "gpu" | "ramTotalGiB"> | null,
  profile: TuneProfile,
): { fits: boolean; text: string } | null {
  if (!hw) return null;
  const depth = tuneDepth(profile, model.contextMax);
  const at = depthLabel(depth);
  if (!hw.gpu) return { fits: false, text: `No graphics card found: runs on the processor at ${at}` };
  const est = estimateFit(model, { ...startingParams(model, hw), gpuLayers: model.arch.layers, moeCpuLayers: 0, contextLength: depth, kvCacheK: "f16", kvCacheV: "f16" }, hw);
  const cap = depth < PROFILES[profile].depth ? ` (its most)` : "";
  const vram = Math.round(hw.gpu.vramTotalGiB * 10) / 10;
  if (est.vram.status === "over") return { fits: false, text: `Needs ${est.vram.total.toFixed(1)} GiB at ${at}${cap}: part runs on the processor` };
  return { fits: true, text: `${est.vram.status === "tight" ? "Just fits" : "Fits"} in ${vram} GiB video memory at ${at}${cap}` };
}

export function isTuneProfile(value: unknown): value is TuneProfile {
  return typeof value === "string" && (PROFILE_IDS as readonly string[]).includes(value);
}

/** A candidate must beat Cerebro Engine's automatic settings by more than this share to replace them:
 *  repeated runs of one configuration differ by about this much. */
export const NOISE_MARGIN = 0.03;

export interface PhaseMeta {
  label: string;
  text: string;
}

export const PHASE_IDS: readonly TunePhase[] = ["probe", "screen", "measure", "quality", "soak", "decide"];

/** Wording from docs/CLAIMS.md "About measurement". */
export const PHASES: Readonly<Record<TunePhase, PhaseMeta>> = {
  probe: { label: "Probe", text: "Read the graphics card, its free video memory and the backend build." },
  screen: { label: "Screen", text: "Try each candidate briefly. The first repetition is always discarded." },
  measure: { label: "Measure", text: "Three numbers per survivor: prompt reading, writing from empty, and writing at your working depth." },
  quality: { label: "Quality gate", text: "Compare each faster candidate's answers with the engine's automatic settings. A setting that hurts them is rejected." },
  soak: { label: "Soak", text: "Run the fastest remaining candidate under sustained load. Free video memory must hold steady over the final third." },
  decide: { label: "Lock in", text: "Keep the fastest configuration that passed every check, or the automatic settings if nothing beat it." },
};

export type TuneStatus = "idle" | "running" | "cancelled" | "done" | "failed";

export type CandidateStatus = "queued" | "screening" | "interrupted" | "advanced" | "discarded" | "failed";

export interface SpeedSample {
  rep: number;
  t: number;
  tps: number;
}

export interface CandidateView {
  id: string;
  candidate: TuneCandidate;
  status: CandidateStatus;
  samples: SpeedSample[];
  /** Screening speed (repetitions after the first), or null. */
  screenTps: number | null;
  reason: string | null;
  result: TuneMeasurement | null;
  soak: "none" | "running" | "interrupted" | "passed" | "failed";
  quality: "none" | "passed" | "failed";
  winner: boolean;
}

export interface SoakSample {
  t: number;
  vramFreeGiB: number;
  tps: number;
}

export interface SoakView {
  candidateId: string;
  samples: SoakSample[];
  interrupted: boolean;
  result: { passed: boolean; slopeMiBPerMin: number; reason: string } | null;
}

export interface QualityView {
  candidateId: string;
  metric: string;
  value: number;
  gate: number;
  passed: boolean;
  reference: string;
}

export interface TuneView {
  run: TuneRunInfo | null;
  status: TuneStatus;
  phase: TunePhase | null;
  probe: { vramFreeGiB: number; integrity: "verified" | "off" } | null;
  candidates: CandidateView[];
  /** The candidate being screened or soaked right now, or null. */
  activeId: string | null;
  soaks: SoakView[];
  quality: QualityView[];
  winner: { candidateId: string; baselineId: string; reasons: string[] } | null;
  failure: string | null;
  /** Run clock of the newest event, seconds. */
  clock: number;
  /** Sequence number of the newest event applied. */
  seq: number;
}

export function emptyTuneView(): TuneView {
  return { run: null, status: "idle", phase: null, probe: null, candidates: [], activeId: null, soaks: [], quality: [], winner: null, failure: null, clock: 0, seq: 0 };
}

const findCandidate = (v: TuneView, id: string) => v.candidates.find((c) => c.id === id);

/**
 * Applies one event to `v` in place (plain objects in tests, a Solid `produce` draft in the screen).
 * Events already applied (seq not newer) are ignored, except `started`, which resets the view for a
 * new run. Re-applying the first events of an interrupted step (a resume replays it) is safe: each
 * step opens with an event that resets what the step builds (`screen-start`, `soak-start`).
 */
export function applyTuneEvent(v: TuneView, e: TuneEvent): void {
  if (e.kind === "started") {
    Object.assign(v, emptyTuneView(), { run: e.run, status: "running", seq: e.seq, clock: e.at });
    return;
  }
  if (v.run === null || e.seq <= v.seq) return;
  v.seq = e.seq;
  v.clock = Math.max(v.clock, e.at);
  switch (e.kind) {
    case "phase":
      v.phase = e.phase;
      if (e.phase !== "screen" && e.phase !== "soak") v.activeId = null;
      break;
    case "probe":
      v.probe = { vramFreeGiB: e.vramFreeGiB, integrity: e.integrity };
      break;
    case "candidate": {
      if (findCandidate(v, e.candidate.id)) break;
      v.candidates.push({
        id: e.candidate.id,
        candidate: e.candidate,
        status: "queued",
        samples: [],
        screenTps: null,
        reason: null,
        result: null,
        soak: "none",
        quality: "none",
        winner: false,
      });
      break;
    }
    case "screen-start": {
      const c = findCandidate(v, e.candidateId);
      if (!c) break;
      c.status = "screening";
      c.samples = [];
      c.screenTps = null;
      c.reason = null;
      v.activeId = c.id;
      break;
    }
    case "sample": {
      const c = findCandidate(v, e.candidateId);
      if (c && c.status === "screening") c.samples.push({ rep: e.rep, t: e.t, tps: e.tps });
      break;
    }
    case "screened": {
      const c = findCandidate(v, e.candidateId);
      if (!c) break;
      c.status = e.outcome === "advance" ? "advanced" : e.outcome === "discard" ? "discarded" : "failed";
      c.screenTps = e.tps;
      c.reason = e.reason;
      if (v.activeId === c.id) v.activeId = null;
      break;
    }
    case "measured": {
      const c = findCandidate(v, e.candidateId);
      if (c) c.result = { prefillTps: e.prefillTps, decodeTps0: e.decodeTps0, decodeTpsDepth: e.decodeTpsDepth };
      break;
    }
    case "soak-start": {
      const c = findCandidate(v, e.candidateId);
      if (!c) break;
      c.soak = "running";
      v.activeId = c.id;
      const existing = v.soaks.find((s) => s.candidateId === c.id);
      if (existing) {
        existing.samples = [];
        existing.interrupted = false;
        existing.result = null;
      } else {
        v.soaks.push({ candidateId: c.id, samples: [], interrupted: false, result: null });
      }
      break;
    }
    case "soak-sample": {
      const s = v.soaks.find((x) => x.candidateId === e.candidateId);
      if (s && s.result === null && !s.interrupted) s.samples.push({ t: e.t, vramFreeGiB: e.vramFreeGiB, tps: e.tps });
      break;
    }
    case "soak-end": {
      const s = v.soaks.find((x) => x.candidateId === e.candidateId);
      if (s) s.result = { passed: e.passed, slopeMiBPerMin: e.slopeMiBPerMin, reason: e.reason };
      const c = findCandidate(v, e.candidateId);
      if (c) c.soak = e.passed ? "passed" : "failed";
      if (v.activeId === e.candidateId) v.activeId = null;
      break;
    }
    case "quality": {
      const q: QualityView = { candidateId: e.candidateId, metric: e.metric, value: e.value, gate: e.gate, passed: e.passed, reference: e.reference };
      const i = v.quality.findIndex((x) => x.candidateId === e.candidateId);
      if (i >= 0) v.quality[i] = q;
      else v.quality.push(q);
      const c = findCandidate(v, e.candidateId);
      if (c) c.quality = e.passed ? "passed" : "failed";
      break;
    }
    case "winner": {
      v.winner = { candidateId: e.candidateId, baselineId: e.baselineId, reasons: [...e.reasons] };
      for (const c of v.candidates) c.winner = c.id === e.candidateId;
      v.status = "done";
      v.activeId = null;
      break;
    }
    case "cancelled": {
      v.status = "cancelled";
      for (const c of v.candidates) {
        if (c.status === "screening") c.status = "interrupted";
        if (c.soak === "running") c.soak = "interrupted";
      }
      for (const s of v.soaks) if (s.result === null) s.interrupted = true;
      v.activeId = null;
      break;
    }
    case "resumed":
      v.status = "running";
      break;
    case "failed":
      v.status = "failed";
      v.failure = e.reason;
      for (const c of v.candidates) if (c.status === "screening") c.status = "interrupted";
      for (const s of v.soaks) if (s.result === null) s.interrupted = true;
      v.activeId = null;
      break;
  }
}

/** Builds a view from a list of events (tests, and a fresh subscription's replay). */
export function reduceTune(events: readonly TuneEvent[]): TuneView {
  const v = emptyTuneView();
  for (const e of events) applyTuneEvent(v, e);
  return v;
}

/** Least-squares line through the final third of a soak: slope in GiB per second, through the
 *  point (meanT, meanY). Null until the final third has two samples at different times. */
export function finalThirdFit(samples: ReadonlyArray<Pick<SoakSample, "t" | "vramFreeGiB">>, soakSeconds: number): { slope: number; meanT: number; meanY: number; fromT: number; toT: number } | null {
  const tail = samples.filter((s) => s.t >= (soakSeconds * 2) / 3);
  if (tail.length < 2) return null;
  const n = tail.length;
  const meanT = tail.reduce((a, s) => a + s.t, 0) / n;
  const meanY = tail.reduce((a, s) => a + s.vramFreeGiB, 0) / n;
  let num = 0;
  let den = 0;
  for (const s of tail) {
    num += (s.t - meanT) * (s.vramFreeGiB - meanY);
    den += (s.t - meanT) ** 2;
  }
  if (den === 0) return null;
  return { slope: num / den, meanT, meanY, fromT: tail[0]!.t, toT: tail[n - 1]!.t };
}

/**
 * Trend of free video memory over the final third of a soak, MiB per minute. The soak passes only
 * when this is not negative (docs/CLAIMS.md). Null until the final third has two samples.
 */
export function finalThirdSlope(samples: ReadonlyArray<Pick<SoakSample, "t" | "vramFreeGiB">>, soakSeconds: number): number | null {
  const fit = finalThirdFit(samples, soakSeconds);
  // GiB per second -> MiB per minute.
  return fit ? fit.slope * 1024 * 60 : null;
}

/** Screening speed: the mean of every repetition after the first (the first is warm-up). */
export function screeningSpeed(samples: readonly SpeedSample[]): number | null {
  const kept = samples.filter((s) => s.rep > 1);
  if (kept.length === 0) return null;
  return kept.reduce((a, s) => a + s.tps, 0) / kept.length;
}

/** "4:05", "1:02:03": a run clock. */
export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

/** Share of the whole run done, 0..1, from the phase and how far it has got. */
export function tuneProgress(v: TuneView): number {
  if (v.status === "done") return 1;
  if (!v.run || !v.phase) return 0;
  const weights: Record<TunePhase, number> = { probe: 0.04, screen: 0.46, measure: 0.08, quality: 0.06, soak: 0.32, decide: 0.04 };
  let done = 0;
  for (const p of PHASE_IDS) {
    if (p === v.phase) break;
    done += weights[p];
  }
  let within = 0;
  if (v.phase === "screen") {
    const finished = v.candidates.filter((c) => c.status === "advanced" || c.status === "discarded" || c.status === "failed").length;
    const active = v.candidates.find((c) => c.status === "screening");
    const partial = active ? Math.min(1, active.samples.length / 10) : 0;
    within = Math.min(1, (finished + partial) / Math.max(1, v.run.plannedCandidates));
  } else if (v.phase === "soak") {
    const total = v.run.soakSeconds * Math.max(1, v.run.plannedSoaks);
    const seconds = v.soaks.reduce((a, s) => a + (s.result ? v.run!.soakSeconds : (s.samples.at(-1)?.t ?? 0)), 0);
    within = Math.min(1, seconds / total);
  } else if (v.phase === "measure") {
    within = 0.5;
  }
  return Math.min(1, done + weights[v.phase] * within);
}

/** "4K", "16K", "32K". */
export function depthLabel(tokens: number): string {
  return tokens >= 1024 && tokens % 1024 === 0 ? `${tokens / 1024}K` : tokens.toLocaleString("en-US");
}

/**
 * The single number a profile ranks candidates by (higher is better): writing speed at depth for
 * chat and coding; for long documents, the inverse of the time to read a depth-sized document and
 * write a 500-token answer.
 */
export function objectiveScore(profile: TuneProfile, m: TuneMeasurement, depth: number): number {
  if (profile === "long-docs") return 1 / (depth / m.prefillTps + 500 / m.decodeTpsDepth);
  return m.decodeTpsDepth;
}

/** Seconds to read a `depth`-token document and write 500 tokens: the long-documents figure. */
export function answerSeconds(m: TuneMeasurement, depth: number): number {
  return depth / m.prefillTps + 500 / m.decodeTpsDepth;
}

/** The newest status line a screen reader should hear: phase changes and outcomes, never samples. */
export function announcement(v: TuneView): string {
  if (!v.run) return "";
  switch (v.status) {
    case "cancelled":
      return "Tune cancelled. Finished steps are kept; resume any time.";
    case "failed":
      // The run's inline alert (role="alert") announces the reason once; saying it here too would
      // read it twice.
      return "";
    case "done": {
      const w = v.candidates.find((c) => c.winner);
      return w ? (w.candidate.baseline ? "Tune finished. The automatic settings were fastest and are kept." : `Tune finished. ${w.candidate.label} locked in.`) : "Tune finished.";
    }
    default:
      return v.phase ? `${PHASES[v.phase].label}: ${PHASES[v.phase].text}` : "Starting the tune.";
  }
}
