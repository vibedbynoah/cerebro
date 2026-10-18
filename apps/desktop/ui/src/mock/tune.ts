import { finalThirdSlope, NOISE_MARGIN, objectiveScore, PROFILES, tuneDepth } from "../lib/tune";
import { startingParams } from "../lib/load-params";
import type { LibraryModel, LoadParams, TuneCandidate, TuneEvent, TuneEventBody, TuneMeasurement, TuneProfile, TuneRunInfo, TuneSetting } from "../service/types";
import { MOCK_BACKEND_COMMIT, MOCK_GPU, MOCK_HARDWARE, MOCK_LIVE_BASE, MOCK_MODELS } from "./fixtures";

/**
 * ILLUSTRATIVE DATA, NOT MEASUREMENTS. A scripted tune for the Tune screen's mock event stream.
 *
 * The run is computed up front from the model's published shape (layers, KV heads, experts) and a
 * deliberately simple memory-and-bandwidth model of the example 8 GiB laptop GPU, so the story is
 * coherent for any model: what fits, what runs out of memory, what wins and why. Every number is
 * invented and every run carries `provenance: "illustrative"`. Nothing here describes real
 * llama.cpp performance; the real engine sends the same event shapes with measured values.
 */

/** Where a test can freeze playback (mid-screen, mid-soak), for deterministic screenshots. */
export type TuneHold = "screen" | "soak";

export interface ScheduledEvent {
  /** Playback delay before this event at speed 1, ms. */
  gap: number;
  /** Run clock, seconds. */
  at: number;
  /** Events of one step are replayed together when a cancelled run resumes. */
  step: number;
  body: TuneEventBody;
  mark?: TuneHold;
}

export interface TuneScript {
  run: TuneRunInfo;
  events: ScheduledEvent[];
  /** The load settings behind each candidate, by id: what the mock library keeps when it wins. */
  params: Record<string, LoadParams>;
}

// ---- The example machine ----------------------------------------------------------------------

const VRAM = MOCK_GPU.vramTotalGiB;
/** What the desktop and driver keep: the free VRAM the probe reports. */
const VRAM_FREE = Math.round((VRAM - 0.45) * 100) / 100;
/** A configuration that needs more than this does not load at all. */
const LOAD_LIMIT = VRAM_FREE;
/** Below this much headroom a configuration loads, but its free VRAM drifts down under load. */
const SAFE_HEADROOM = 0.45;
const GPU_BW = 300; // GiB/s, effective
const CPU_BW = 26; // GiB/s, effective for the CPU's share of the weights
/** Extra time per token for each layer whose attention runs on the CPU, s. */
const CPU_LAYER_COST = 0.003;
const SPLIT_OVERHEAD = 0.012; // s per token when work crosses the PCIe bus
const SOAK_SECONDS = 600;
const QUALITY_GATE = 0.01;
const SAMPLER = "temperature 0 (greedy), set explicitly";

type Kv = "f16" | "q8_0" | "q4_0";

interface Config {
  /** Layers on the GPU; "fit" lets llama.cpp's automatic fit decide. */
  gpuLayers: number | "fit";
  /** MoE: layers whose expert weights stay in system memory. */
  moeCpu: number;
  kv: Kv;
  fa: boolean;
  /** Pinned to performance cores, or null for llama.cpp's own choice. */
  threads: number | null;
  ubatch: number;
  ngram: boolean;
}

const BASE: Config = { gpuLayers: "fit", moeCpu: 0, kv: "f16", fa: false, threads: null, ubatch: 512, ngram: false };

const KV_BYTES: Record<Kv, number> = { f16: 2, q8_0: 1.0625, q4_0: 0.5625 };
const KL: Record<Kv, number> = { f16: 0.0005, q8_0: 0.0032, q4_0: 0.024 };

const isMoe = (m: LibraryModel) => m.arch.experts !== null && m.arch.activeExperts !== null;
const expertShare = 0.9; // of an MoE file's weights

function kvGiB(m: LibraryModel, depth: number, kv: Kv): number {
  return (2 * m.arch.layers * m.arch.kvHeads * m.arch.headDim * depth * KV_BYTES[kv]) / 2 ** 30;
}

function computeBuffer(c: Config): number {
  return 0.35 + 0.15 * (c.ubatch / 512 - 1);
}

/** Layers the automatic fit puts on the GPU: as many whole layers (with their KV cache) as fit. */
function fitLayers(m: LibraryModel, depth: number): number {
  const L = m.arch.layers;
  const perLayer = (m.sizeGiB + kvGiB(m, depth, "f16")) / L;
  return Math.max(0, Math.min(L, Math.floor((LOAD_LIMIT - SAFE_HEADROOM - computeBuffer(BASE)) / perLayer)));
}

function resolvedLayers(m: LibraryModel, c: Config, depth: number): number {
  return c.gpuLayers === "fit" ? fitLayers(m, depth) : Math.min(m.arch.layers, c.gpuLayers);
}

/** GPU memory a configuration needs, GiB. */
function vramNeeded(m: LibraryModel, c: Config, depth: number): number {
  const L = m.arch.layers;
  const ngl = resolvedLayers(m, c, depth);
  let weights = (m.sizeGiB * ngl) / L;
  if (isMoe(m) && c.moeCpu > 0) weights -= (m.sizeGiB * expertShare * Math.min(c.moeCpu, ngl)) / L;
  return weights + (kvGiB(m, depth, c.kv) * ngl) / L + computeBuffer(c);
}

/** Bytes read per generated token on the GPU and on the CPU, GiB. */
function activeSplit(m: LibraryModel, c: Config, depth: number): { gpu: number; cpu: number } {
  const L = m.arch.layers;
  const ngl = resolvedLayers(m, c, depth);
  if (!isMoe(m)) return { gpu: (m.sizeGiB * ngl) / L, cpu: (m.sizeGiB * (L - ngl)) / L };
  const shared = m.sizeGiB * (1 - expertShare);
  const experts = m.sizeGiB * expertShare * ((m.arch.activeExperts ?? 1) / (m.arch.experts ?? 1));
  const expertsOnGpu = Math.max(0, ngl - c.moeCpu);
  return {
    gpu: (shared * ngl) / L + (experts * expertsOnGpu) / L,
    cpu: (shared * (L - ngl)) / L + (experts * (L - expertsOnGpu)) / L,
  };
}

const NGRAM_GAIN: Record<TuneProfile, number> = { chat: 0.95, coding: 1.24, "long-docs": 1.05 };

/** The three numbers for a configuration, before noise. Invented, deliberately simple. */
function model3(m: LibraryModel, c: Config, profile: TuneProfile, depth: number): TuneMeasurement {
  const { gpu, cpu } = activeSplit(m, c, depth);
  const cpuLayers = m.arch.layers - resolvedLayers(m, c, depth);
  const cpuTime = (cpu / CPU_BW + cpuLayers * CPU_LAYER_COST) * (c.threads !== null && cpu > 0 ? 0.86 : 1);
  const perToken = gpu / GPU_BW + cpuTime + (cpu > 0.01 ? SPLIT_OVERHEAD : 0) + 0.004;
  let decode0 = 1 / perToken;
  if (c.ngram) decode0 *= NGRAM_GAIN[profile];
  // Writing slows as the context fills; flash attention and a smaller cache slow it less.
  const depthCost = depth / (c.fa ? 80_000 : 60_000) / (c.kv === "f16" ? 1 : 1.08);
  const decodeDepth = decode0 / (1 + depthCost);
  const active = gpu + cpu;
  const gpuShare = active > 0 ? gpu / active : 1;
  const prefillBase = 1 / (gpuShare / (5200 / active) + (1 - gpuShare) / (420 / active));
  const batchGain = 1 + (cpu > 0.01 ? 0.2 : 0.04) * Math.log2(c.ubatch / 512);
  return { prefillTps: prefillBase * batchGain * (c.fa ? 1.06 : 1), decodeTps0: decode0, decodeTpsDepth: decodeDepth };
}

// ---- Deterministic noise ------------------------------------------------------------------------

function hash32(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const r1 = (n: number) => Math.round(n * 10) / 10;
const r2 = (n: number) => Math.round(n * 100) / 100;
const r3 = (n: number) => Math.round(n * 1000) / 1000;
const r4 = (n: number) => Math.round(n * 10000) / 10000;

// ---- Candidates ---------------------------------------------------------------------------------

interface Planned {
  candidate: TuneCandidate;
  config: Config;
}

function settingsFor(m: LibraryModel, c: Config, depth: number): TuneSetting[] {
  const L = m.arch.layers;
  if (c.gpuLayers === "fit") {
    return [
      { name: "Placement", value: "automatic", flag: null },
      { name: "GPU layers", value: `${fitLayers(m, depth)} of ${L} on the GPU`, flag: null },
    ];
  }
  const out: TuneSetting[] = [{ name: "GPU layers", value: c.gpuLayers >= L ? `all ${L}` : `${c.gpuLayers} of ${L}`, flag: `-ngl ${c.gpuLayers >= L ? 999 : c.gpuLayers}` }];
  if (c.moeCpu > 0) out.push({ name: "Experts on CPU", value: `${c.moeCpu} layers`, flag: `-ncmoe ${c.moeCpu}` });
  if (c.kv !== "f16") out.push({ name: "KV cache", value: c.kv, flag: `-ctk ${c.kv} -ctv ${c.kv}` });
  if (c.fa) out.push({ name: "Flash attention", value: "on", flag: "-fa on" });
  if (c.threads !== null) out.push({ name: "Threads", value: `${c.threads}, performance cores`, flag: `-t ${c.threads}` });
  if (c.ubatch !== 512) out.push({ name: "Micro-batch", value: c.ubatch.toLocaleString("en-US"), flag: `-ub ${c.ubatch}` });
  if (c.ngram) out.push({ name: "Speculation", value: "n-gram (ngram-mod)", flag: null });
  return out;
}

/** Smallest number of expert layers on the CPU that loads with headroom, for one KV type. */
function moeFrontier(m: LibraryModel, base: Config, depth: number): number {
  for (let n = 0; n <= m.arch.layers; n++) if (vramNeeded(m, { ...base, moeCpu: n }, depth) <= LOAD_LIMIT - SAFE_HEADROOM) return n;
  return m.arch.layers;
}

/** Most GPU layers that load with headroom, for one configuration. */
function denseFrontier(m: LibraryModel, base: Config, depth: number): number {
  for (let n = m.arch.layers; n > 0; n--) if (vramNeeded(m, { ...base, gpuLayers: n }, depth) <= LOAD_LIMIT - SAFE_HEADROOM) return n;
  return 0;
}

type Scenario = "fits" | "moe" | "dense";

function scenarioFor(m: LibraryModel, depth: number): Scenario {
  if (vramNeeded(m, { ...BASE, gpuLayers: m.arch.layers }, depth) <= LOAD_LIMIT - SAFE_HEADROOM) return "fits";
  return isMoe(m) ? "moe" : "dense";
}

function plan(m: LibraryModel, profile: TuneProfile, depth: number, scenario: Scenario): Planned[] {
  const L = m.arch.layers;
  const all: Array<{ label: string; config: Config }> = [{ label: "Cerebro Engine automatic", config: BASE }];
  const P = 6; // performance cores on the example CPU
  if (scenario === "moe") {
    const f16 = { ...BASE, gpuLayers: L };
    const q8 = { ...f16, kv: "q8_0" as const, fa: true };
    const n16 = moeFrontier(m, f16, depth);
    const n8 = moeFrontier(m, q8, depth);
    // The smallest step that no longer loads: "only N layers' experts on the CPU".
    let over = Math.max(0, n8 - 2);
    while (over > 0 && vramNeeded(m, { ...q8, moeCpu: over }, depth) <= LOAD_LIMIT) over--;
    all.push({ label: `Experts split, ${Math.min(L, n16 + 4)} layers on CPU`, config: { ...f16, moeCpu: Math.min(L, n16 + 4) } });
    all.push({ label: `Experts split, ${n16} layers on CPU`, config: { ...f16, moeCpu: n16 } });
    if (vramNeeded(m, { ...q8, moeCpu: over }, depth) > LOAD_LIMIT) all.push({ label: `Experts split, only ${over} layers on CPU`, config: { ...q8, moeCpu: over } });
    all.push({ label: "q8_0 KV cache and flash attention", config: { ...q8, moeCpu: n8 } });
    all.push({ label: `Plus ${P} performance-core threads`, config: { ...q8, moeCpu: n8, threads: P } });
    if (n8 > 0) all.push({ label: "One more expert layer on the GPU", config: { ...q8, moeCpu: n8 - 1, threads: P } });
    all.push({ label: "Plus n-gram speculation", config: { ...q8, moeCpu: n8, threads: P, ngram: true } });
    all.push({ label: "Plus 1,024-token micro-batches", config: { ...q8, moeCpu: Math.min(L, n8 + 1), threads: P, ubatch: 1024 } });
  } else if (scenario === "dense") {
    const fa = { ...BASE, fa: true };
    const q8 = { ...fa, kv: "q8_0" as const };
    const q4 = { ...fa, kv: "q4_0" as const };
    const g16 = denseFrontier(m, { ...fa, gpuLayers: L }, depth);
    const g8 = denseFrontier(m, { ...q8, gpuLayers: L }, depth);
    const g4 = denseFrontier(m, { ...q4, gpuLayers: L }, depth);
    all.push({ label: "Flash attention", config: { ...fa, gpuLayers: g16 } });
    all.push({ label: "q8_0 KV cache, more layers on the GPU", config: { ...q8, gpuLayers: g8 } });
    if (g8 + 2 <= L) all.push({ label: "Two more layers on the GPU", config: { ...q8, gpuLayers: g8 + 2 } });
    all.push({ label: `Plus ${P} performance-core threads`, config: { ...q8, gpuLayers: g8, threads: P } });
    all.push({ label: "q4_0 KV cache, most layers on the GPU", config: { ...q4, gpuLayers: g4, threads: P } });
    all.push({ label: "Plus n-gram speculation", config: { ...q8, gpuLayers: g8, threads: P, ngram: true } });
    all.push({ label: "Plus 1,024-token micro-batches", config: { ...q8, gpuLayers: Math.max(1, g8 - 1), threads: P, ubatch: 1024 } });
  } else {
    const gpu = { ...BASE, gpuLayers: L };
    all.push({ label: "Flash attention", config: { ...gpu, fa: true } });
    all.push({ label: "q8_0 KV cache and flash attention", config: { ...gpu, fa: true, kv: "q8_0" } });
    all.push({ label: `${P} performance-core threads`, config: { ...gpu, threads: P } });
    all.push({ label: "1,024-token micro-batches", config: { ...gpu, ubatch: 1024 } });
    all.push({ label: "n-gram speculation", config: { ...gpu, ngram: true } });
  }
  void profile;
  return all.map((x, i) => ({ config: x.config, candidate: { id: `c${i}`, label: x.label, baseline: i === 0, settings: settingsFor(m, x.config, depth) } }));
}

/** A candidate's configuration as the load settings the Models screen uses. */
function loadParamsFor(m: LibraryModel, c: Config, depth: number): LoadParams {
  const start = startingParams(m, MOCK_HARDWARE);
  return {
    ...start,
    contextLength: Math.min(depth, m.contextMax),
    gpuLayers: resolvedLayers(m, c, depth),
    moeCpuLayers: isMoe(m) ? c.moeCpu : 0,
    kvCacheK: c.kv,
    kvCacheV: c.kv,
    // llama.cpp's own choice is "auto"; a candidate that names it turns it on.
    flashAttention: c.fa ? "on" : "auto",
    threads: c.threads ?? start.threads,
    cpuAffinity: c.threads !== null ? "performance" : "all",
    cpuCores: "",
    ubatchSize: c.ubatch,
    batchSize: Math.max(start.batchSize, c.ubatch),
    speculative: c.ngram ? "ngram" : "off",
    ngramStrategy: c.ngram ? "ngram-mod" : start.ngramStrategy,
    draftModelId: null,
  };
}

// ---- Explanation --------------------------------------------------------------------------------

function pct(a: number, b: number): string {
  return `${Math.round(Math.abs(a / b - 1) * 100)}%`;
}

function reasonsFor(m: LibraryModel, depth: number, profile: TuneProfile, scenario: Scenario, winner: Planned, baseline: Planned, best: Planned | null, rejected: string[], results: Map<string, TuneMeasurement>): string[] {
  const L = m.arch.layers;
  const fit = fitLayers(m, depth);
  const out: string[] = [];
  if (winner.candidate.baseline) {
    if (scenario === "fits") out.push(`The whole model and its KV cache fit in video memory at ${depth / 1024}K tokens, so the automatic placement already put all ${L} layers on the graphics card.`);
    else out.push(`The automatic placement put ${fit} of ${L} layers on the graphics card, and no candidate that passed every check did better.`);
    if (best && !best.candidate.baseline) {
      const b = results.get(best.candidate.id);
      const base = results.get(baseline.candidate.id);
      if (b && base) {
        const gain = objectiveScore(profile, b, depth) / objectiveScore(profile, base, depth) - 1;
        if (gain <= NOISE_MARGIN) out.push(`The best candidate, "${best.candidate.label}", was within ${Math.max(0, Math.round(gain * 100))}% of it: inside the ${Math.round(NOISE_MARGIN * 100)}% noise margin, so changing settings would add risk for no measurable gain.`);
      }
    }
    if (rejected.length > 0) out.push(`Faster candidates were rejected: ${rejected.join("; ")}.`);
    out.push("Cerebro keeps the engine's automatic settings rather than change something that did not help.");
    return out;
  }
  const c = winner.config;
  const ngl = resolvedLayers(m, c, depth);
  if (scenario === "moe") {
    out.push(`The automatic placement put ${fit} of ${L} layers on the graphics card and ran the other ${L - fit} entirely on the processor. That is the most it could fit, not the fastest it could run.`);
    out.push(
      `The winner keeps every layer's attention on the graphics card and moves only the expert weights of ${c.moeCpu} layers to system memory. This model reads ${m.arch.activeExperts} of its ${m.arch.experts} experts per token, so those reads cost little.`,
    );
  } else if (ngl !== fit) {
    out.push(`The automatic placement put ${fit} of ${L} layers on the graphics card. The winner places ${ngl}, so less of each token's work waits on the processor.`);
  } else {
    out.push(`The automatic placement put ${fit} of ${L} layers on the graphics card. The winner keeps that placement and changes how the work runs.`);
  }
  if (c.kv !== "f16") {
    const saved = kvGiB(m, depth, "f16") - kvGiB(m, depth, c.kv);
    out.push(`A ${c.kv} KV cache needs ${r1(saved)} GiB less than f16 at ${depth / 1024}K tokens${scenario === "moe" || ngl > fit ? ", which made that room" : ""}.`);
  }
  if (c.fa && scenario !== "moe") out.push("Flash attention slows writing less as the context fills.");
  if (c.threads !== null) out.push(`Pinning ${c.threads} threads to the performance cores ran the processor's share faster.`);
  if (c.ngram) out.push("N-gram speculation guessed often enough from repeated text to pay for checking its guesses. It is distribution-preserving: the model verifies every guess.");
  if (c.ubatch !== 512) out.push(`Micro-batches of ${c.ubatch.toLocaleString("en-US")} tokens read a long document faster, which matters most for this profile.`);
  if (rejected.length > 0) out.push(`Faster on paper but rejected: ${rejected.join("; ")}.`);
  return out;
}

// ---- The script ---------------------------------------------------------------------------------

export interface ScriptOptions {
  runId: string;
  integrityCheck: boolean;
  /** Stop the run partway through screening with a crash, for the error path. */
  fail?: boolean;
}

/** The whole illustrative run for one model and profile, computed up front. Deterministic. */
export function buildTuneScript(m: LibraryModel, profile: TuneProfile, opts: ScriptOptions): TuneScript {
  // The profile's depth, or less when the model cannot hold that much text.
  const depth = tuneDepth(profile, m.contextMax);
  const scenario = scenarioFor(m, depth);
  const planned = plan(m, profile, depth, scenario);
  const rnd = mulberry32(hash32(`${m.id}|${profile}`));
  const jitter = (spread: number) => 1 + (rnd() * 2 - 1) * spread;
  // Scale so the baseline lands on the illustrative defaults speed the rest of the app shows.
  const summary = MOCK_MODELS.find((s) => s.id === m.id);
  const rawBase = model3(m, BASE, "chat", PROFILES.chat.depth).decodeTps0;
  const scale = summary?.defaultsTps ? summary.defaultsTps / rawBase : 1;

  const events: ScheduledEvent[] = [];
  let at = 0;
  let step = 0;
  const push = (gap: number, dt: number, body: TuneEventBody, mark?: TuneHold) => {
    at += dt;
    events.push({ gap, at: Math.round(at * 10) / 10, step, body, ...(mark ? { mark } : {}) });
  };
  const nextStep = () => {
    step += 1;
  };

  const soakPlanned: Planned[] = [];
  const params: Record<string, LoadParams> = {};
  for (const p of planned) params[p.candidate.id] = loadParamsFor(m, p.config, depth);
  const run: TuneRunInfo = {
    runId: opts.runId,
    modelId: m.id,
    modelName: m.name,
    quant: m.quant,
    architecture: m.architecture,
    sizeGiB: m.sizeGiB,
    profile,
    depth,
    sampler: SAMPLER,
    build: MOCK_LIVE_BASE.backend.build,
    backendCommit: MOCK_BACKEND_COMMIT,
    gpu: MOCK_GPU.name,
    vramTotalGiB: VRAM,
    cpu: `${MOCK_HARDWARE.cpu.name} (${MOCK_HARDWARE.cpu.cores} cores, ${MOCK_HARDWARE.cpu.threads} threads)`,
    ramTotalGiB: MOCK_HARDWARE.ramTotalGiB,
    soakSeconds: SOAK_SECONDS,
    plannedCandidates: planned.length,
    plannedSoaks: 0, // filled in below, once the soaks are known
    integrityCheck: opts.integrityCheck,
    provenance: "illustrative",
  };

  push(0, 0, { kind: "started", run });
  push(260, 0.4, { kind: "phase", phase: "probe" });
  push(700, 6.2, { kind: "probe", vramFreeGiB: VRAM_FREE, integrity: opts.integrityCheck ? "verified" : "off" });
  nextStep();
  push(500, 1.5, { kind: "phase", phase: "screen" });
  for (const p of planned) push(110, 0, { kind: "candidate", candidate: p.candidate });
  nextStep();

  // Screening: short bursts, first repetition discarded. Bursts read faster than sustained speed,
  // most of all on MoE models, so they only rank; Measure and Soak produce the numbers that count.
  const burst = isMoe(m) ? 1.18 : 1.04;
  const truth = new Map<string, TuneMeasurement>();
  const advanced: Planned[] = [];
  let bestScreen = 0;
  const loadSeconds = 8 + m.sizeGiB * 1.6;
  for (const [index, p] of planned.entries()) {
    const id = p.candidate.id;
    push(320, 4, { kind: "screen-start", candidateId: id });
    const need = vramNeeded(m, p.config, depth);
    if (!p.candidate.baseline && need > LOAD_LIMIT) {
      push(900, loadSeconds * 0.7, {
        kind: "screened",
        candidateId: id,
        tps: null,
        outcome: "fail",
        reason: `Ran out of video memory while loading: needs ${r1(need)} GiB, ${VRAM_FREE} GiB free.`,
      });
      nextStep();
      continue;
    }
    const raw = model3(m, p.config, profile, depth);
    const t3: TuneMeasurement = { prefillTps: raw.prefillTps * scale, decodeTps0: raw.decodeTps0 * scale, decodeTpsDepth: raw.decodeTpsDepth * scale };
    truth.set(id, t3);
    const v = t3.decodeTps0 * burst;
    const samples: Array<{ rep: number; tps: number }> = [];
    for (const f of [0.62, 0.81, 0.93]) samples.push({ rep: 1, tps: v * f * jitter(0.02) });
    for (let i = 0; i < 4; i++) samples.push({ rep: 2, tps: v * jitter(0.025) });
    for (let i = 0; i < 4; i++) samples.push({ rep: 3, tps: v * jitter(0.025) });
    let t = 0;
    for (const [i, s] of samples.entries()) {
      t += 0.9 + rnd() * 0.25;
      push(i === 0 ? 700 : 190, i === 0 ? loadSeconds : 4.5, { kind: "sample", candidateId: id, rep: s.rep, t: r2(t), tps: r2(s.tps) }, !opts.fail && index === 4 && i === 6 ? "screen" : undefined);
      if (opts.fail && index === 3 && i === 4) {
        push(600, 2, { kind: "failed", reason: "The model process stopped unexpectedly while screening a candidate (exit code 0xC0000005). Nothing on this machine was changed" });
        run.plannedSoaks = 2;
        return { run, events, params };
      }
    }
    const kept = samples.filter((s) => s.rep > 1);
    const screen = kept.reduce((a, s) => a + r2(s.tps), 0) / kept.length;
    const isBaseline = p.candidate.baseline;
    // Long documents rank on reading and writing together, so a slower writer is not discarded on
    // its writing burst alone.
    const keepRatio = profile === "long-docs" ? 0.75 : 0.88;
    const advance = isBaseline || screen >= bestScreen * keepRatio;
    const reason = isBaseline
      ? "Reference: every candidate is compared with it."
      : advance
        ? screen > bestScreen
          ? "Fastest so far."
          : pct(screen, bestScreen) === "0%"
            ? "As fast as the fastest so far; measured in full."
            : `Within ${pct(screen, bestScreen)} of the fastest; measured in full.`
        : `${pct(screen, bestScreen)} slower than the fastest so far.`;
    push(420, 3, { kind: "screened", candidateId: id, tps: r1(screen), outcome: advance ? "advance" : "discard", reason });
    bestScreen = Math.max(bestScreen, screen);
    if (advance) advanced.push(p);
    nextStep();
  }

  // Measure: the three numbers for every survivor.
  push(500, 2, { kind: "phase", phase: "measure" });
  nextStep();
  const results = new Map<string, TuneMeasurement>();
  for (const p of advanced) {
    const t3 = truth.get(p.candidate.id)!;
    const m3: TuneMeasurement = { prefillTps: Math.round(t3.prefillTps * jitter(0.02)), decodeTps0: r1(t3.decodeTps0 * jitter(0.015)), decodeTpsDepth: 0 };
    m3.decodeTpsDepth = Math.min(m3.decodeTps0, r1(t3.decodeTpsDepth * jitter(0.015)));
    results.set(p.candidate.id, m3);
    push(480, 45, { kind: "measured", candidateId: p.candidate.id, ...m3 });
    nextStep();
  }

  const score = (p: Planned) => objectiveScore(profile, results.get(p.candidate.id)!, depth);
  const ranked = [...advanced].sort((a, b) => score(b) - score(a));
  const baseline = planned[0]!;
  const baseScore = score(baseline);
  // Only a candidate that beats llama.cpp's own settings by more than the noise margin is worth
  // checking further; the rest would change settings for no measurable gain.
  const contenders = ranked.filter((p) => !p.candidate.baseline && score(p) > baseScore * (1 + NOISE_MARGIN));
  const rejected: string[] = [];

  // Quality gate first (minutes), so no soak (much longer) is spent on a configuration that would
  // fail it. At most four are checked.
  push(500, 2, { kind: "phase", phase: "quality" });
  nextStep();
  const passedQuality: Planned[] = [];
  for (const p of contenders.slice(0, 4)) {
    const value = r4(KL[p.config.kv] * jitter(0.15));
    const passed = value <= QUALITY_GATE;
    push(1000, 90, {
      kind: "quality",
      candidateId: p.candidate.id,
      metric: "KL divergence",
      value,
      gate: QUALITY_GATE,
      passed,
      reference: "next-token probabilities from the engine's automatic settings with an f16 KV cache, on 64 fixed prompts",
    });
    nextStep();
    if (passed) passedQuality.push(p);
    else rejected.push(`"${p.candidate.label}" failed the quality gate`);
  }

  // Soak: the fastest remaining candidate under sustained load; if its free video memory falls,
  // the next one (at most three). When none is left, llama.cpp's own settings are confirmed the
  // same way before they are kept.
  push(500, 2, { kind: "phase", phase: "soak" });
  nextStep();
  const soak = (p: Planned): boolean => {
    soakPlanned.push(p);
    const id = p.candidate.id;
    const headroom = VRAM_FREE - vramNeeded(m, p.config, depth);
    const leaking = headroom < SAFE_HEADROOM;
    const start = r2(leaking ? Math.max(0.3, headroom) : headroom);
    const sustained = results.get(id)!.decodeTpsDepth * 0.97;
    push(420, 6, { kind: "soak-start", candidateId: id });
    const samples: Array<{ t: number; vramFreeGiB: number }> = [];
    for (let i = 1; i <= 30; i++) {
      const t = (SOAK_SECONDS / 30) * i;
      // GiB: a slow climb while healthy; a steady fall from the first quarter while leaking.
      const drift = leaking ? -Math.max(0, t - SOAK_SECONDS / 4) * (0.03 / 60) : (t / 60) * 0.003;
      const free = r3(Math.max(0, start + drift + (rnd() * 2 - 1) * 0.0015));
      samples.push({ t, vramFreeGiB: free });
      push(150, SOAK_SECONDS / 30, { kind: "soak-sample", candidateId: id, t, vramFreeGiB: free, tps: r1(sustained * jitter(0.02)) });
    }
    const slope = finalThirdSlope(samples, SOAK_SECONDS) ?? 0;
    const passed = slope >= 0;
    push(500, 2, {
      kind: "soak-end",
      candidateId: id,
      passed,
      slopeMiBPerMin: r1(slope),
      reason: passed
        ? "Free video memory held steady over the final third."
        : `Free video memory kept falling over the final third (${r1(slope)} MiB a minute). At that rate it runs out within the hour.`,
    });
    nextStep();
    if (!passed) rejected.push(`"${p.candidate.label}" failed the soak`);
    return passed;
  };
  let winner: Planned | null = null;
  for (const p of passedQuality.slice(0, 3)) {
    if (soak(p)) {
      winner = p;
      break;
    }
  }
  if (!winner) {
    soak(baseline);
    winner = baseline;
  }
  // A test can freeze the last soak past its two-thirds mark, where the trend line has appeared.
  const lastSoak = events.findLastIndex((e) => e.body.kind === "soak-start");
  const holdAt = events[lastSoak + 23];
  if (holdAt?.body.kind === "soak-sample") holdAt.mark = "soak";

  push(500, 2, { kind: "phase", phase: "decide" });
  nextStep();
  const best = ranked.find((p) => !p.candidate.baseline) ?? null;
  push(900, 3, {
    kind: "winner",
    candidateId: winner.candidate.id,
    baselineId: baseline.candidate.id,
    reasons: reasonsFor(m, depth, profile, scenario, winner, baseline, best, rejected, results),
  });
  run.plannedSoaks = soakPlanned.length;
  return { run, events, params };
}

/** Plain copy of an event for delivery (listeners never share the log's objects). */
export function cloneEvent(e: TuneEvent): TuneEvent {
  return JSON.parse(JSON.stringify(e)) as TuneEvent;
}

// ---- Playback -----------------------------------------------------------------------------------

export interface TunePlayerOptions {
  /** Playback speed multiplier (1: a full run in under a minute). Infinity plays everything at once. */
  speed: number;
  hold?: TuneHold | undefined;
  fail?: boolean | undefined;
}

interface ActiveRun {
  script: TuneScript;
  status: "running" | "cancelled" | "done" | "failed";
  /** Index of the next event to emit. */
  next: number;
  /** Where a resume restarts: the first event of the interrupted step. */
  restart: number;
  /** Added to every scripted run clock value, so time spent replaying a step is counted once. */
  offset: number;
  holding: boolean;
  log: TuneEvent[];
}

/**
 * Plays scripts as a live event stream for CerebroService's tune methods. Validation of model ids
 * and profiles is the caller's; run ids are checked here. Only one run exists at a time.
 */
export function createTunePlayer(options: TunePlayerOptions) {
  const speed = Number.isFinite(options.speed) || options.speed === Number.POSITIVE_INFINITY ? Math.max(0.01, options.speed) : 1;
  const listeners = new Map<(events: TuneEvent[], replay: boolean) => void, { ready: boolean }>();
  let run: ActiveRun | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let seq = 0;
  let runs = 0;

  const deliver = (batch: TuneEvent[]) => {
    for (const [listener, state] of listeners) if (state.ready) listener(batch.map(cloneEvent), false);
  };

  const emit = (r: ActiveRun, body: TuneEventBody, at: number): TuneEvent => {
    const event = { ...body, seq: ++seq, at: Math.round(at * 10) / 10 } as TuneEvent;
    r.log.push(event);
    if (body.kind === "winner") r.status = "done";
    if (body.kind === "failed") r.status = "failed";
    return event;
  };

  const schedule = (r: ActiveRun) => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (r.status !== "running" || r.holding || r.next >= r.script.events.length) return;
    const gap = r.script.events[r.next]!.gap / speed;
    timer = setTimeout(() => pump(r), Number.isFinite(gap) ? gap : 0);
  };

  /** Emits the next event, plus every following one due within a few milliseconds, as one batch. */
  const pump = (r: ActiveRun) => {
    timer = null;
    if (run !== r || r.status !== "running") return;
    const batch: TuneEvent[] = [];
    while (r.status === "running" && r.next < r.script.events.length) {
      const item = r.script.events[r.next]!;
      if (options.hold !== undefined && item.mark === options.hold) {
        r.holding = true;
        break;
      }
      if (batch.length > 0 && item.gap / speed >= 4) break;
      batch.push(emit(r, item.body, item.at + r.offset));
      r.next += 1;
    }
    if (batch.length > 0) deliver(batch);
    schedule(r);
  };

  const stepStart = (r: ActiveRun, index: number): number => {
    const step = r.script.events[index]?.step;
    if (step === undefined) return index;
    let i = index;
    while (i > 0 && r.script.events[i - 1]!.step === step) i--;
    return i;
  };

  const current = (runId: unknown): ActiveRun => {
    if (run === null || typeof runId !== "string" || runId !== run.script.run.runId) throw new TypeError("Unknown tune run");
    return run;
  };

  return {
    isRunning: () => run !== null && run.status === "running",

    /** Starts a new run from a script built by the caller for this run id. */
    start(build: (runId: string) => TuneScript): TuneRunInfo {
      if (run !== null && run.status === "running") throw new Error("A tune is already running. Cancel it first.");
      if (timer !== null) clearTimeout(timer);
      runs += 1;
      const script = build(`tune-${runs}`);
      const r: ActiveRun = { script, status: "running", next: 0, restart: 0, offset: 0, holding: false, log: [] };
      run = r;
      schedule(r);
      return { ...script.run };
    },

    subscribe(listener: (events: TuneEvent[], replay: boolean) => void): () => void {
      const state = { ready: false };
      listeners.set(listener, state);
      // The replay is sent before any live event reaches this listener, so the two never interleave.
      queueMicrotask(() => {
        if (listeners.get(listener) !== state) return;
        state.ready = true;
        listener(run ? run.log.map(cloneEvent) : [], true);
      });
      let active = true;
      return () => {
        if (!active) return;
        active = false;
        if (listeners.get(listener) === state) listeners.delete(listener);
      };
    },

    cancel(runId: unknown): void {
      const r = current(runId);
      if (r.status !== "running") throw new TypeError("This tune is not running");
      if (timer !== null) clearTimeout(timer);
      timer = null;
      r.holding = false;
      r.status = "cancelled";
      r.restart = stepStart(r, r.next);
      deliver([emit(r, { kind: "cancelled" }, r.log.at(-1)?.at ?? 0)]);
    },

    resume(runId: unknown): void {
      const r = current(runId);
      if (r.status !== "cancelled") throw new TypeError("Only a cancelled tune can be resumed");
      const lastAt = r.log.at(-1)?.at ?? 0;
      r.status = "running";
      deliver([emit(r, { kind: "resumed" }, lastAt)]);
      r.next = r.restart;
      const restartAt = r.script.events[r.restart]?.at ?? 0;
      // The replayed step starts a second after the resume on the run clock.
      r.offset = lastAt + 1 - restartAt;
      schedule(r);
    },

    dispose(): void {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      listeners.clear();
    },
  };
}
