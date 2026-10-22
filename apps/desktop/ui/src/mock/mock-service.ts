import { validateGgufEdits } from "../lib/gguf";
import { draftCandidates, hasErrors, validateLoadParams } from "../lib/load-params";
import { activeTweakCount, applyTweaks, contextCeiling, EMPTY_TWEAKS, profileNameError, sanitizeTweaks } from "../lib/tweaks";
import type { CerebroService, Unsubscribe } from "../service/service";
import { createMockChat } from "./chat";
import { createMockTrainer } from "./mock-train";
import { cloneMachineTuneOutcome, createMachineTunePlayer } from "./machine-tune";
import type {
  CloudProvider,
  ClusterShare,
  EngineSwitch,
  LocalApiView,
  McpServerView,
  RemoteView,
  McpConfig,
  McpServerStatus,
  McpTool,
  ActivePatch,
  CatalogModel,
  DuplicateGroup,
  GgufCopy,
  GpuInfo,
  HardwareProfile,
  KvStoreUsage,
  LibraryModel,
  LiveStats,
  LoadedModel,
  LoadParams,
  LoadPreset,
  MachineTuneOutcome,
  ModelSummary,
  NetworkState,
  PatchProfile,
  SettingKey,
  Settings,
  TuneEvent,
  TuneState,
} from "../service/types";
import { DEFAULT_SETTINGS, MOCK_SWITCHES, MOCK_CATALOG, MOCK_GPU, MOCK_HARDWARE, MOCK_LIVE_BASE, MOCK_MODELS } from "./fixtures";
import { detailFor, MOCK_LIBRARY, mockDuplicates } from "./library";
import { buildTuneScript, createTunePlayer, type TuneHold, type TuneScript } from "./tune";
import { applyTuneEvent, emptyTuneView, isTuneProfile } from "../lib/tune";
import { estimateFit } from "../lib/fit";
import { isHistoryRange } from "../lib/monitor";
import type { MetricsSeries, MetricsSnapshot, RequestDetail } from "../service/monitor-types";
import { createMonitorSim, type MonitorScenario, type SimModel } from "./monitor";

export interface MockServiceOptions {
  /** HUD sample interval, ms. Default 1000. */
  tickMs?: number;
  /** 0..1 random source; inject a seeded one for deterministic tests. Default Math.random. */
  random?: () => number;
  /** Starting settings (merged over the safe defaults). */
  settings?: Partial<Settings>;
  network?: NetworkState;
  /** Failure injection for tests: how many detectHardware calls reject before one succeeds
   *  (Infinity: every call). Default 0. */
  failHardware?: number;
  /** detectHardware resolves with a malformed profile (no cpu), as a broken backend might. */
  malformedHardware?: boolean;
  /** listCatalog rejects. */
  failCatalog?: boolean;
  /** listLibrary resolves with no models (the empty state). */
  emptyLibrary?: boolean;
  /** listLibrary rejects. */
  failLibrary?: boolean;
  /** Simulated latency in ms (default 0: resolve at once). The app sets realistic values so loading
   *  states are visible; tests leave them at 0 or ask for `slow-library`. */
  latency?: { library?: number; load?: number; test?: number; scan?: number };
  /** Tune playback speed (1: a whole run in under a minute; Infinity: at once). Default 1. */
  tuneSpeed?: number;
  /** Freeze a tune partway (mid-screen, mid-soak) for screenshots of a run in progress. */
  tuneHold?: TuneHold;
  /** The tune's model process crashes partway through screening. */
  failTune?: boolean;
  /** Monitor: what the simulated engine is doing (default: serving a working session). */
  monitorScenario?: MonitorScenario;
  /** Monitor: a machine with two graphics cards instead of one laptop GPU. */
  multiGpu?: boolean;
  /** Monitor: getMetrics rejects, as an engine that stopped answering would. */
  failMetrics?: boolean;
  /** Chat playback speed multiplier (tests use 20). */
  chatSpeed?: number;
  /** Orchestrate playback speed multiplier (tests use 1: the steps are visible). */
  orchestrateSpeed?: number;
  /** Milliseconds per simulated training step (tests use small values). */
  trainStepMs?: number;
  /** Clock for the Monitor's simulation (epoch ms). Default Date.now. */
  now?: () => number;
}

/** Every failure or test state a web/test build can ask the mock for, by `?mock=` value. */
const MOCK_FAILURES: Readonly<Record<string, MockServiceOptions>> = {
  "fail-hw": { failHardware: Number.POSITIVE_INFINITY },
  "fast-train": { trainStepMs: 15 },
  "fail-hw-once": { failHardware: 1 },
  "bad-hw": { malformedHardware: true },
  "fail-catalog": { failCatalog: true },
  "empty-library": { emptyLibrary: true },
  "fail-library": { failLibrary: true },
  "slow-library": { latency: { library: 60_000 } },
  "fast-tune": { tuneSpeed: 60 },
  "tune-hold-screen": { tuneHold: "screen" },
  "tune-hold-soak": { tuneHold: "soak" },
  "fail-tune": { failTune: true },
  "monitor-alerts": { monitorScenario: "alerts" },
  "monitor-idle": { monitorScenario: "idle" },
  "monitor-multi-gpu": { multiGpu: true },
  "fail-metrics": { failMetrics: true },
  "fast-chat": { chatSpeed: 20 },
  // Orchestrate at real speed: fast enough to test, slow enough to watch each role start.
  orchestrate: { orchestrateSpeed: 1 },
  online: { network: "online" },
};

/**
 * Failure switches from the page's query string (`?mock=fail-hw&mock=fail-catalog`), for headless
 * tests of the error paths. Only exact known values count; anything else is ignored. main.tsx calls
 * this in the web build only (`__DESKTOP__` compiles the call out of the desktop app), and nothing
 * here reads storage, so no persisted value can turn a failure on.
 */
export function mockOptionsFromQuery(search: string): MockServiceOptions {
  const out: MockServiceOptions = {};
  for (const value of new URLSearchParams(search).getAll("mock")) {
    if (Object.hasOwn(MOCK_FAILURES, value)) Object.assign(out, MOCK_FAILURES[value]);
  }
  return out;
}

const SETTING_KEYS: ReadonlySet<string> = new Set(Object.keys(DEFAULT_SETTINGS));

export function isSettingKey(key: unknown): key is SettingKey {
  return typeof key === "string" && SETTING_KEYS.has(key);
}

const round1 = (n: number): number => Math.round(n * 10) / 10;
const round2 = (n: number): number => Math.round(n * 100) / 100;

function cloneStats(s: LiveStats): LiveStats {
  return { ...s, backend: { ...s.backend } };
}

/** Resolves with `fn()` after `ms` (at once when 0), or rejects with what it throws. */
function later<T>(ms: number | undefined, fn: () => T): Promise<T> {
  const run = (resolve: (v: T) => void, reject: (e: unknown) => void) => {
    try {
      resolve(fn());
    } catch (err) {
      reject(err);
    }
  };
  if (ms === undefined || !(ms > 0)) return new Promise(run);
  return new Promise((resolve, reject) => setTimeout(() => run(resolve, reject), ms));
}

const cloneParams = (p: LoadParams): LoadParams => ({ ...p });
const cloneLoaded = (l: LoadedModel): LoadedModel => ({ ...l, params: l.params ? cloneParams(l.params) : null });
const cloneLibrary = (m: LibraryModel): LibraryModel => ({
  ...m,
  modalities: [...m.modalities],
  arch: { ...m.arch },
  tunedParams: m.tunedParams ? cloneParams(m.tunedParams) : null,
  projector: m.projector ? { ...m.projector } : null,
});
const clonePatch = (p: ActivePatch): ActivePatch => ({
  tweaks: { ...p.tweaks, stopTokens: [...p.tweaks.stopTokens], rope: p.tweaks.rope ? { ...p.tweaks.rope } : null, sampler: p.tweaks.sampler ? { ...p.tweaks.sampler } : null },
  fixes: [...p.fixes],
});
const cloneProfile = (p: PatchProfile): PatchProfile => ({
  ...p,
  tweaks: { ...p.tweaks, stopTokens: [...p.tweaks.stopTokens], rope: p.tweaks.rope ? { ...p.tweaks.rope } : null, sampler: p.tweaks.sampler ? { ...p.tweaks.sampler } : null },
  fixes: [...p.fixes],
});
const clonePreset = (p: LoadPreset): LoadPreset => ({ ...p, params: cloneParams(p.params) });
const cloneGroup = (g: DuplicateGroup): DuplicateGroup => ({ ...g, copies: g.copies.map((c) => ({ ...c })) });
/** Copies on the kept copy's drive other than the kept one: the ones a hard link can replace. */
const linkableCopies = (g: DuplicateGroup, keepPath: string) => {
  const keep = g.copies.find((c) => c.path === keepPath);
  return keep ? g.copies.filter((c) => c !== keep && c.volume === keep.volume) : [];
};
const FIX_ID = /^[a-z0-9-]{1,48}$/;
const REQUEST_ID = /^req_\d{1,8}$/;

/** Plain labels for the draft-free strategies, as the Models screen names them. */
const NGRAM_PLAIN: Readonly<Record<LoadParams["ngramStrategy"], string>> = {
  "ngram-simple": "Repeat recent text",
  "ngram-map-k": "Match the last few tokens",
  "ngram-map-k4v": "Longer 4-token matches",
  "ngram-mod": "Hash of recent text",
  "ngram-cache": "Cache of past replies",
};

/** Two example workstation cards for the multi-GPU Monitor view. */
const MULTI_GPUS = [
  { name: "Example workstation GPU", vramGiB: 24, powerLimitW: 350 },
  { name: "Example workstation GPU", vramGiB: 24, powerLimitW: 350 },
];

/** The built-in tools (src-tauri/src/builtin_tools.rs) as the Tools screen reads them. Illustrative. */
const MOCK_BUILTIN_TOOLS: readonly McpTool[] = [
  { server: "cerebro", name: "read_file", description: "Read a text file from a workspace folder.", input_schema: {} },
  { server: "cerebro", name: "list_directory", description: "List the entries of a folder in the workspace.", input_schema: {} },
  { server: "cerebro", name: "search_files", description: "Find lines containing a pattern in the workspace files.", input_schema: {} },
  { server: "cerebro", name: "write_file", description: "Write (create or replace) a text file in a workspace folder.", input_schema: {} },
  { server: "cerebro", name: "edit_file", description: "Replace one exact occurrence of old_text in a file.", input_schema: {} },
  { server: "cerebro", name: "run_command", description: "Run a PowerShell command in the first workspace folder.", input_schema: {} },
  { server: "cerebro", name: "fetch_url", description: "Fetch a web page as text.", input_schema: {} },
];

/** The loaded model as the Monitor's simulation needs it: placement and memory from the fit
 *  estimate of its tuned settings, speeds from its (illustrative) tune result. */
function simModelFor(m: LibraryModel, loadedParams: LoadParams | null, vramGiB: number): SimModel | null {
  const p = loadedParams ?? m.tunedParams;
  if (!p) return null;
  // With room for the whole model (two big cards), everything goes to the GPUs.
  let params: LoadParams = vramGiB > MOCK_GPU.vramTotalGiB ? { ...p, gpuLayers: m.arch.layers, moeCpuLayers: 0 } : p;
  const hw = { gpu: { name: MOCK_GPU.name, vramTotalGiB: vramGiB }, ramTotalGiB: MOCK_HARDWARE.ramTotalGiB };
  // A MoE model with experts on the CPU gives up about 1.5 GiB of resident expert weights to the
  // expert cache, which keeps the most-used experts on the card instead of a fixed set.
  if (m.architecture === "moe" && params.moeCpuLayers > 0) {
    const before = estimateFit(m, params, hw).vram.weights;
    for (let n = params.moeCpuLayers; n <= params.gpuLayers; n++) {
      params = { ...params, moeCpuLayers: n };
      if (estimateFit(m, params, hw).vram.weights <= before - 1.5) break;
    }
  }
  const fit = estimateFit(m, params, hw);
  const r = m.tuneResult;
  return {
    id: m.id,
    name: m.name,
    quant: m.quant,
    params: m.params,
    architecture: m.architecture,
    contextMax: params.contextLength,
    kvType: params.kvCacheK === params.kvCacheV ? params.kvCacheK : `${params.kvCacheK} / ${params.kvCacheV}`,
    layers: m.arch.layers,
    gpuLayers: params.gpuLayers,
    expertLayersOnCpu: m.architecture === "moe" ? params.moeCpuLayers : 0,
    expertsTotal: m.arch.experts,
    expertsPerToken: m.arch.activeExperts,
    vram: { weights: fit.vram.weights, kv: fit.vram.kv, scratch: round2(fit.vram.compute + fit.vram.extra) },
    ramWeights: fit.ram.weights,
    sizeGiB: m.sizeGiB,
    decode0: r?.decodeTps0 ?? 10,
    baseline: r ? { decodeTps: r.decodeTpsDepth, prefillTps: r.prefillTps, depth: r.depth, measuredAt: r.measuredAt } : null,
    speculation: params.speculative === "ngram" ? NGRAM_PLAIN[params.ngramStrategy] : params.speculative === "draft" ? "Draft model" : null,
  };
}
const IMAGE_TYPE = /^image\/(png|jpeg|webp|gif)$/;

/**
 * In-memory CerebroService for web mode and tests. All data is illustrative (see fixtures.ts).
 * The HUD ticker only runs while someone is subscribed, so an idle page schedules no timers.
 */
export function createMockService(options: MockServiceOptions = {}): CerebroService {
  let cloudList: CloudProvider[] = [];
  let recordingSince = 0;
  let clusterShare: ClusterShare = { sharing: false, model: null, code: null, port: null, addresses: ["192.168.1.10"], computer: "THIS-PC" };
  const hubDone = new Set<string>();
  const hubCancelled = new Set<string>();
  const switches: EngineSwitch[] = MOCK_SWITCHES.map((x) => ({ ...x }));
  let remote: Omit<RemoteView, "running" | "url" | "address" | "port"> = {
    enabled: false, addressChoice: null, addresses: ["192.168.1.20"], passwordSet: false, allowChat: false, allowTailscale: false,
    sessionDays: 30, fingerprint: "AB:CD:EF:01:23:45:67:89", devices: [], error: null,
  };
  const remoteView = (): RemoteView => ({
    ...remote,
    running: remote.enabled,
    url: remote.enabled ? `https://${remote.addressChoice ?? "192.168.1.20"}:51234/` : null,
    address: remote.addressChoice ?? "192.168.1.20",
    port: 51234,
  });
  let mcpSrv: McpServerView = { enabled: false, running: false, url: null, port: 47811, token: "mock-token-0123456789abcdefghijklmnop", clients: [], error: null };
  let mcp: McpConfig = {
    mcpServers: { files: { command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/me/Projects"], env: {}, disabled: false } },
  };
  // The folders the built-in tools may touch (Settings > Tools); illustrative in web mode.
  let folders: string[] = ["C:\\Users\\me\\Projects"];
  let localApi: LocalApiView = { enabled: false, running: false, url: null, lanUrls: [], port: 47812, key: "sk-cerebro-mock0123456789abcdef0123456789abcdef", requireKey: true, allowLan: false, allowModelSwitch: true, requests: 0, lastRequestAt: null, error: null };
  const apiView = (): LocalApiView => ({ ...localApi, lanUrls: localApi.running && localApi.allowLan ? ["http://192.168.1.20:47812/v1"] : [] });
  const trainer = createMockTrainer(options.trainStepMs !== undefined ? { stepMs: options.trainStepMs } : {});
  const chatEngine = createMockChat({ ...(options.chatSpeed !== undefined ? { speed: options.chatSpeed } : {}), ...(options.orchestrateSpeed !== undefined ? { orchestrateSpeed: options.orchestrateSpeed } : {}) });
  const tickMs = options.tickMs !== undefined && Number.isFinite(options.tickMs) && options.tickMs > 0 ? options.tickMs : 1000;
  const random = options.random ?? Math.random;
  const network: NetworkState = options.network ?? "offline";
  let settings: Settings = { ...DEFAULT_SETTINGS, ...options.settings };
  // The saved-chat folder starts empty in web mode; the desktop service reads the real one.
  let kvUsage: KvStoreUsage = { files: 0, bytes: 0 };
  let hardwareFailuresLeft = options.failHardware !== undefined && options.failHardware > 0 ? options.failHardware : 0;
  let current: LiveStats = cloneStats(MOCK_LIVE_BASE);
  const listeners = new Set<(stats: LiveStats) => void>();
  let timer: ReturnType<typeof setInterval> | null = null;

  // ---- Models screen state. Every method checks its arguments the way the Rust side will. ----
  const latency = options.latency ?? {};
  // This service's own copies: a finished tune writes its result into them, and other services
  // (one per test) must not see it.
  const library: readonly LibraryModel[] = options.emptyLibrary ? [] : MOCK_LIBRARY.map(cloneLibrary);
  const byId = new Map(library.map((m) => [m.id, m]));
  const summaries: ModelSummary[] = MOCK_MODELS.map((m) => ({ ...m }));
  // Matches the HUD's starting sample: the tuned coding model is loaded.
  const startModel = byId.get("qwen3-coder-30b-a3b-q4km");
  let loaded: LoadedModel[] = startModel ? [{ id: startModel.id, slot: "llm", params: startModel.tunedParams ? cloneParams(startModel.tunedParams) : null, tuned: true }] : [];
  if (!startModel) current = { ...current, model: null, tokensPerSecond: null };
  const presets = new Map<string, LoadPreset[]>();
  const profiles = new Map<string, PatchProfile[]>();
  const active = new Map<string, ActivePatch>();
  const copies = new Map<string, GgufCopy>();
  let groups: DuplicateGroup[] = mockDuplicates();
  const tune = createTunePlayer({ speed: options.tuneSpeed ?? 1, hold: options.tuneHold, fail: options.failTune });
  // Tuning for this machine (Turbo): replays the recorded run kept in machine-tune.ts.
  const machineTune = createMachineTunePlayer();
  let lastTuneOutcome: MachineTuneOutcome | null = null;
  let nextId = 1;
  let speechTimer: ReturnType<typeof setTimeout> | undefined;
  let speechDone: (() => void) | undefined;
  const multiGpu = options.multiGpu === true;
  const monitor = createMonitorSim({
    random,
    ...(options.now ? { now: options.now } : {}),
    scenario: options.monitorScenario ?? "default",
    gpus: multiGpu ? MULTI_GPUS : [{ name: MOCK_GPU.name, vramGiB: MOCK_GPU.vramTotalGiB, powerLimitW: 115 }],
    cpu: MOCK_HARDWARE.cpu,
    ramTotalGiB: MOCK_HARDWARE.ramTotalGiB,
    build: MOCK_LIVE_BASE.backend.build,
    address: MOCK_LIVE_BASE.backend.address ?? "127.0.0.1:8181",
  });
  /** The text or vision model in the llm slot, as the Monitor sees it. */
  const servedModel = (): SimModel | null => {
    const inSlot = loaded.find((l) => l.slot === "llm");
    const m = inSlot ? byId.get(inSlot.id) : undefined;
    return m ? simModelFor(m, inSlot?.params ?? null, multiGpu ? 48 : MOCK_GPU.vramTotalGiB) : null;
  };

  // ---- A tune's outcome is the library's truth: the service follows its own run and writes the
  // model's state (measuring, then tuned or defaults-fastest), its three numbers and its kept
  // settings back, so Models, Home and the HUD show what the Tune screen just locked in. Every
  // number copied keeps its "illustrative" provenance. ----
  let tuneScript: TuneScript | null = null;
  const followed = emptyTuneView();
  /** The model's tune state before the current run, restored when the run stops without a result. */
  let stateBefore: { id: string; tune: TuneState } | null = null;
  const setTuneState = (id: string, state: TuneState) => {
    const m = byId.get(id);
    if (m) m.tune = state;
    const s = summaries.find((x) => x.id === id);
    if (s) s.tune = state;
  };
  const keepWinner = (e: Extract<TuneEvent, { kind: "winner" }>) => {
    const run = followed.run;
    if (!run || !tuneScript || tuneScript.run.runId !== run.runId) return;
    const m = byId.get(run.modelId);
    const win = followed.candidates.find((c) => c.id === e.candidateId);
    const base = followed.candidates.find((c) => c.id === e.baselineId);
    const params = tuneScript.params[e.candidateId];
    if (!m || !win?.result || !base?.result || !params) return;
    const kept = win.candidate.baseline;
    const today = new Date().toISOString().slice(0, 10);
    m.tuneResult = {
      prefillTps: win.result.prefillTps,
      decodeTps0: win.result.decodeTps0,
      decodeTpsDepth: win.result.decodeTpsDepth,
      depth: run.depth,
      build: run.build,
      gpu: run.gpu,
      sampler: run.sampler,
      measuredAt: today,
      provenance: "illustrative",
    };
    m.tunedParams = cloneParams(params);
    m.tunedAt = today;
    // A model running on its tuned settings now runs on the new ones.
    loaded = loaded.map((l) => (l.id === m.id && l.tuned ? { ...l, params: cloneParams(params) } : l));
    setTuneState(m.id, kept ? "defaults-fastest" : "tuned");
    const s = summaries.find((x) => x.id === m.id);
    if (s) {
      // Home's summary pair is writing speed with an empty context, as the HUD shows it.
      s.defaultsTps = base.result.decodeTps0;
      s.tunedTps = win.result.decodeTps0;
    }
    stateBefore = null;
  };
  tune.subscribe((events) => {
    for (const e of events) {
      applyTuneEvent(followed, e);
      const run = followed.run;
      if (!run) continue;
      switch (e.kind) {
        case "started":
          stateBefore = { id: run.modelId, tune: byId.get(run.modelId)?.tune ?? "untuned" };
          setTuneState(run.modelId, "tuning");
          break;
        case "resumed":
          setTuneState(run.modelId, "tuning");
          break;
        case "cancelled":
        case "failed":
          if (stateBefore) setTuneState(stateBefore.id, stateBefore.tune);
          break;
        case "winner":
          keepWinner(e);
          break;
      }
    }
  });
  const model = (id: unknown): LibraryModel => {
    const m = typeof id === "string" ? byId.get(id) : undefined;
    if (!m) throw new TypeError(`Unknown model: ${String(id)}`);
    return m;
  };
  const isLlm = (m: LibraryModel) => m.kind === "text" || m.kind === "vision";
  const tweaksFor = (m: LibraryModel) => active.get(m.sha256)?.tweaks ?? EMPTY_TWEAKS;
  // The same rule as the UI: the context ceiling grows only with the file's active RoPE tweak.
  const checkParams = (m: LibraryModel, params: unknown) => {
    const draftIds = new Set(draftCandidates(m, library).map((d) => d.id));
    if (hasErrors(validateLoadParams(params, { ...m, contextMax: contextCeiling(m, tweaksFor(m)) }, MOCK_HARDWARE, draftIds))) throw new TypeError("Invalid load settings");
  };

  const next = (): LiveStats => {
    const base = MOCK_LIVE_BASE;
    const jitter = (spread: number) => (random() * 2 - 1) * spread;
    // The HUD names whatever is in the llm slot; its speed is that model's illustrative tuned speed.
    const inSlot = loaded.find((l) => l.slot === "llm");
    const m = inSlot ? byId.get(inSlot.id) : undefined;
    // Writing with an empty context (a short chat), from the model's last tune, so a re-tune shows here too.
    const speed = m ? (m.tuneResult?.decodeTps0 ?? (m.name === base.model ? base.tokensPerSecond : 10)) : null;
    // The status bar reads the same simulated engine as the Monitor, so the two never disagree; a
    // model the simulation cannot describe (no settings) falls back to its tuned speed.
    const sim = m && !multiGpu ? monitor.hud(servedModel()) : null;
    current = {
      ...cloneStats(base),
      model: m ? m.name : null,
      tokensPerSecond: speed === null ? null : (sim?.decodeTps ?? round1(speed + jitter(0.6))),
      vramUsedGiB: m ? round2(Math.min(base.vramTotalGiB, Math.max(0, sim?.vramUsedGiB ?? base.vramUsedGiB + jitter(0.08)))) : 0.4,
    };
    return cloneStats(current);
  };

  const start = () => {
    if (timer !== null) return;
    timer = setInterval(() => {
      const sample = next();
      for (const listener of listeners) listener(cloneStats(sample));
    }, tickMs);
  };
  const stop = () => {
    if (timer === null) return;
    clearInterval(timer);
    timer = null;
  };

  return {
    getLiveStats: () => Promise.resolve(cloneStats(current)),

    subscribeLiveStats(listener): Unsubscribe {
      listeners.add(listener);
      start();
      let active = true;
      return () => {
        if (!active) return;
        active = false;
        listeners.delete(listener);
        if (listeners.size === 0) stop();
      };
    },

    listModels: (): Promise<ModelSummary[]> => Promise.resolve(summaries.map((m) => ({ ...m }))),

    getGpu: (): Promise<GpuInfo> => Promise.resolve({ ...MOCK_GPU }),

    detectHardware: (): Promise<HardwareProfile> => {
      if (hardwareFailuresLeft > 0) {
        hardwareFailuresLeft -= 1;
        return Promise.reject(new Error("Mock: hardware detection failed"));
      }
      if (options.malformedHardware) return Promise.resolve({ ...MOCK_HARDWARE, cpu: null } as unknown as HardwareProfile);
      return Promise.resolve({
        ...MOCK_HARDWARE,
        cpu: { ...MOCK_HARDWARE.cpu },
        gpu: MOCK_HARDWARE.gpu === null ? null : { ...MOCK_HARDWARE.gpu },
      });
    },

    listCatalog: (): Promise<CatalogModel[]> =>
      options.failCatalog ? Promise.reject(new Error("Mock: catalog unavailable")) : Promise.resolve(MOCK_CATALOG.map((m) => structuredClone(m))),

    getNetwork: () => Promise.resolve(network),

    getSettings: () => Promise.resolve({ ...settings }),

    updateSetting(key, value) {
      // Same checks the Rust side will make: a known key and a boolean value, nothing else.
      if (!isSettingKey(key)) return Promise.reject(new TypeError(`Unknown setting: ${String(key)}`));
      if (typeof value !== "boolean") return Promise.reject(new TypeError(`Setting ${key} takes true or false`));
      settings = { ...settings, [key]: value };
      return Promise.resolve({ ...settings });
    },

    kvStoreUsage: () => Promise.resolve({ ...kvUsage }),

    kvStoreClear: () => {
      const held = { ...kvUsage };
      kvUsage = { files: 0, bytes: 0 };
      return Promise.resolve(held);
    },

    listLibrary: () =>
      later(latency.library, () => {
        if (options.failLibrary) throw new Error("Mock: library scan failed");
        return library.map(cloneLibrary);
      }),

    getModelDetail: (id) =>
      later(0, () => {
        const m = model(id);
        const copy = copies.get(m.id);
        const d = detailFor(m);
        return {
          ...d,
          gguf: d.gguf ? d.gguf.map((e) => ({ ...e })) : null,
          copy: copy ? { path: copy.path, edits: copy.edits.map((e) => ({ ...e })) } : null,
        };
      }),

    listLoaded: () => later(0, () => loaded.map(cloneLoaded)),

    loadModel: (id, params) =>
      later(latency.load, () => {
        const m = model(id);
        if (isLlm(m)) {
          if (params !== null) checkParams(m, params);
          // null: the tuned settings with this file's active tweaks, the same rule the UI shows.
          const t = tweaksFor(m);
          const used = params ?? (m.tunedParams ? applyTweaks(m.tunedParams, t) : null);
          const untweaked = activeTweakCount(t) === 0 && (active.get(m.sha256)?.fixes.length ?? 0) === 0;
          const entry: LoadedModel = { id: m.id, slot: "llm", params: used ? cloneParams(used) : null, tuned: params === null && untweaked };
          loaded = [...loaded.filter((l) => l.slot !== "llm"), entry];
        } else {
          if (params !== null) throw new TypeError("This engine takes no load settings");
          const slot = m.kind === "stt" ? "stt" : m.kind === "tts" ? "tts" : "image";
          loaded = [...loaded.filter((l) => l.slot !== slot), { id: m.id, slot, params: null, tuned: false }];
        }
        return loaded.map(cloneLoaded);
      }),

    unloadModel: (id) =>
      later(0, () => {
        const m = model(id);
        loaded = loaded.filter((l) => l.id !== m.id);
        return loaded.map(cloneLoaded);
      }),

    listPresets: (modelId) => later(0, () => (presets.get(model(modelId).id) ?? []).map(clonePreset)),

    savePreset: (modelId, name, params) =>
      later(0, () => {
        const m = model(modelId);
        const list = presets.get(m.id) ?? [];
        const err = typeof name === "string" ? profileNameError(name, list.map((p) => p.name)) : "Invalid name";
        if (err) throw new TypeError(err);
        checkParams(m, params);
        const next = [...list, { id: `preset-${nextId++}`, modelId: m.id, name: name.trim(), params: cloneParams(params) }];
        presets.set(m.id, next);
        return next.map(clonePreset);
      }),

    deletePreset: (modelId, presetId) =>
      later(0, () => {
        const m = model(modelId);
        const next = (presets.get(m.id) ?? []).filter((p) => p.id !== presetId);
        presets.set(m.id, next);
        return next.map(clonePreset);
      }),

    getActivePatch: (sha) =>
      later(0, () => {
        const m = typeof sha === "string" ? library.find((x) => x.sha256 === sha) : undefined;
        if (!m) throw new TypeError("Unknown model file");
        return clonePatch(active.get(m.sha256) ?? { tweaks: EMPTY_TWEAKS, fixes: [] });
      }),

    setActivePatch: (sha, patch) =>
      later(0, () => {
        const m = typeof sha === "string" ? library.find((x) => x.sha256 === sha) : undefined;
        if (!m) throw new TypeError("Unknown model file");
        const tweaks = sanitizeTweaks(patch?.tweaks);
        if (!tweaks) throw new TypeError("Invalid tweaks");
        // Only fixes the quirks list has for this file can be switched on.
        const known = new Set(detailFor(m).quirks.map((q) => q.id));
        if (!Array.isArray(patch.fixes) || patch.fixes.length > 32 || !patch.fixes.every((f) => typeof f === "string" && known.has(f))) throw new TypeError("Invalid fixes");
        const next: ActivePatch = { tweaks, fixes: [...new Set(patch.fixes)] };
        active.set(m.sha256, next);
        return clonePatch(next);
      }),

    listPatchProfiles: (sha) => later(0, () => (profiles.get(String(sha)) ?? []).map(cloneProfile)),

    savePatchProfile: (profile) =>
      later(0, () => {
        const sha: unknown = profile?.modelSha256;
        if (typeof sha !== "string" || !library.some((m) => m.sha256 === sha)) throw new TypeError("Unknown model file");
        const list = profiles.get(sha) ?? [];
        const err = typeof profile.name === "string" ? profileNameError(profile.name, list.map((p) => p.name)) : "Invalid name";
        if (err) throw new TypeError(err);
        const tweaks = sanitizeTweaks(profile.tweaks);
        if (!tweaks) throw new TypeError("Invalid tweaks");
        if (!Array.isArray(profile.fixes) || profile.fixes.length > 32 || !profile.fixes.every((f) => typeof f === "string" && FIX_ID.test(f))) throw new TypeError("Invalid fixes");
        const next = [...list, { id: `profile-${nextId++}`, name: profile.name.trim(), modelSha256: sha, tweaks, fixes: [...new Set(profile.fixes)] }];
        profiles.set(sha, next);
        return next.map(cloneProfile);
      }),

    deletePatchProfile: (sha, id) =>
      later(0, () => {
        const key = String(sha);
        const next = (profiles.get(key) ?? []).filter((p) => p.id !== id);
        profiles.set(key, next);
        return next.map(cloneProfile);
      }),

    writeGgufCopy: (id, edits) =>
      later(latency.test, () => {
        const m = model(id);
        const entries = detailFor(m).gguf;
        if (!entries) throw new TypeError("Not a GGUF file");
        const err = validateGgufEdits(edits, entries);
        if (err) throw new TypeError(err);
        // Merge into the existing copy: newest value per key; a key set back to its original value
        // leaves the copy. A copy that would equal the original is refused (revert deletes it).
        const original = new Map(entries.map((e) => [e.key, e.value]));
        const next = new Map((copies.get(m.id)?.edits ?? []).map((e) => [e.key, e.value]));
        for (const e of edits) {
          if (e.value === original.get(e.key)) next.delete(e.key);
          else next.set(e.key, e.value);
        }
        if (next.size === 0) throw new TypeError("The copy would match the original; revert it instead");
        const copy: GgufCopy = { path: m.path.replace(/(\.gguf)?$/i, ".edited.gguf"), edits: [...next].map(([key, value]) => ({ key, value })) };
        copies.set(m.id, copy);
        return { path: copy.path, edits: copy.edits.map((e) => ({ ...e })) };
      }),

    revertGgufCopy: (id) =>
      later(0, () => {
        copies.delete(model(id).id);
      }),

    scanDisk: () =>
      later(latency.scan, () => ({
        provenance: "illustrative" as const,
        // A believable laptop, not the 500-file scroll-test list: the four featured models, a
        // sample of up to 26 more that a laptop would keep (20 GiB or less each), and the extra
        // bytes the duplicate groups still store.
        totalGiB:
          library.length === 0
            ? 0
            : round1(
                library.filter((m, i) => i < 4 || (i % 20 === 0 && m.sizeGiB <= 20)).slice(0, 30).reduce((sum, m) => sum + m.sizeGiB, 0) +
                  // A linked group still stores its copies on other drives (they cannot be hard links).
                  groups.reduce((sum, g) => sum + g.sizeGiB * (g.copies.length - 1 - (g.state === "linked" && g.kept ? linkableCopies(g, g.kept).length : 0)), 0),
              ),
        groups: library.length === 0 ? [] : groups.map(cloneGroup),
      })),

    deduplicate: (groupId, keepPath) =>
      later(latency.scan, () => {
        const g = groups.find((x) => x.id === groupId);
        if (!g) throw new TypeError("Unknown duplicate group");
        if (g.state === "linked") throw new TypeError("Already deduplicated");
        if (typeof keepPath !== "string" || !g.copies.some((c) => c.path === keepPath)) throw new TypeError("Keep one of the group's own copies");
        const linkable = linkableCopies(g, keepPath).length;
        if (linkable === 0) throw new TypeError("No copy shares a drive with the kept one");
        const skipped = g.copies.length - 1 - linkable;
        const next: DuplicateGroup = { ...cloneGroup(g), state: "linked", kept: keepPath };
        groups = groups.map((x) => (x.id === g.id ? next : x));
        return { group: cloneGroup(next), reclaimedGiB: Math.round(g.sizeGiB * linkable * 10) / 10, skipped };
      }),

    unlink: (groupId) =>
      later(latency.scan, () => {
        const g = groups.find((x) => x.id === groupId);
        if (!g) throw new TypeError("Unknown duplicate group");
        if (g.state !== "linked" || g.kept === null) throw new TypeError("Not linked");
        const next: DuplicateGroup = { ...cloneGroup(g), state: "duplicate", kept: null };
        groups = groups.map((x) => (x.id === g.id ? next : x));
        return cloneGroup(next);
      }),

    revealInFolder: (id) =>
      later(0, () => {
        model(id);
        // The browser preview cannot open File Explorer; the desktop bridge will.
        return false;
      }),

    describeImage: (id, image) =>
      later(latency.test, () => {
        const m = model(id);
        if (m.kind !== "vision") throw new TypeError("Not a vision model");
        if (!m.projector) throw new TypeError("No projector paired");
        if (typeof image?.type !== "string" || !IMAGE_TYPE.test(image.type)) throw new TypeError("Unsupported image type");
        if (typeof image.bytes !== "number" || !(image.bytes > 0) || image.bytes > 20 * 1024 * 1024) throw new TypeError("Image too large");
        const name = typeof image.name === "string" ? image.name.slice(0, 120) : "the image";
        return {
          text: `A sample description of ${name}: the model would describe the picture here in a sentence or two. The image stayed on this computer.`,
          provenance: "illustrative" as const,
        };
      }),

    startRecording: () =>
      later(0, () => {
        recordingSince = Date.now();
      }),
    // Model Hub (the live Hub, shown in the browser build with ?hub=live): example repositories and
    // files, a GGUF header per file, and a download that progresses over about two seconds.
    hubSearch: (q) =>
      later(250, () => {
        const all = [
          { id: "example-org/Qwen3.5-4B-GGUF", author: "example-org", downloads: 412000, likes: 930, updated: "2026-09-30T10:00:00Z", license: "apache-2.0", tags: ["gguf", "text-generation", "image-text-to-text"] },
          { id: "example-org/Qwen3-Coder-30B-A3B-Instruct-GGUF", author: "example-org", downloads: 268000, likes: 610, updated: "2026-09-12T10:00:00Z", license: "apache-2.0", tags: ["gguf", "text-generation"] },
          { id: "example-org/gemma-4-E4B-it-GGUF", author: "example-org", downloads: 190000, likes: 455, updated: "2026-08-20T10:00:00Z", license: "gemma", tags: ["gguf", "image-text-to-text"] },
          { id: "example-org/Llama-3.1-8B-Instruct-GGUF", author: "example-org", downloads: 1200000, likes: 2100, updated: "2025-11-02T10:00:00Z", license: "llama3.1", tags: ["gguf", "text-generation"] },
          { id: "example-org/Qwen3-0.6B-GGUF", author: "example-org", downloads: 98000, likes: 220, updated: "2026-05-01T10:00:00Z", license: "apache-2.0", tags: ["gguf"] },
        ];
        const words = q.toLowerCase().replace(/gguf/g, "").split(/\s+/).filter(Boolean);
        const hits = all.filter((r) => words.every((w) => r.id.toLowerCase().includes(w.replace(/\s/g, ""))));
        return hits.length ? hits : all.slice(0, 3);
      }),
    hubFiles: (repo) =>
      later(200, () => {
        const base = repo.split("/").pop()!.replace(/-GGUF$/, "");
        const gib = /30B/.test(base) ? 17.3 : /4B|E4B/.test(base) ? 2.6 : /8B/.test(base) ? 4.6 : 0.5;
        return [
          { path: `${base}-Q4_K_M.gguf`, size: Math.round(gib * 2 ** 30), parts: [`${base}-Q4_K_M.gguf`], quant: "Q4_K_M", downloaded: hubDone.has(`${repo}/${base}-Q4_K_M.gguf`) },
          { path: `${base}-Q8_0.gguf`, size: Math.round(gib * 1.9 * 2 ** 30), parts: [`${base}-Q8_0.gguf`], quant: "Q8_0", downloaded: hubDone.has(`${repo}/${base}-Q8_0.gguf`) },
        ];
      }),
    hubFileMeta: (_repo, file) =>
      later(60, () => ({ size: /Q8/.test(file) ? 4.9 * 2 ** 30 : 2.6 * 2 ** 30, architecture: "qwen3", layers: 36, kvHeads: 8, headDim: 128, embedding: 2560, experts: null, activeExperts: null, contextMax: 40960, attentionLayers: 36, quant: /Q8/.test(file) ? "Q8_0" : "Q4_K_M" })),
    hubDownload: (repo, file, onEvent) =>
      new Promise((resolve) => {
        let done = 0;
        const step = Math.max(1, Math.floor(file.size / 10));
        const t = setInterval(() => {
          done = Math.min(file.size, done + step);
          onEvent({ type: "progress", done, total: file.size, bytesPerSec: step * 5 });
          if (done >= file.size || hubCancelled.delete(`${repo}/${file.path}`)) {
            clearInterval(t);
            if (done >= file.size) {
              hubDone.add(`${repo}/${file.path}`);
              onEvent({ type: "done", path: `C:\\models\\${file.path}` });
            } else onEvent({ type: "error", message: "Stopped. Resume picks up where it left off." });
            resolve(`C:\\models\\${file.path}`);
          }
        }, 200);
      }),
    hubCancel: (repo, path) => later(0, () => void hubCancelled.add(`${repo}/${path}`)),
    // The official MCP registry: two example servers (the preview contacts nothing).
    mcpRegistrySearch: (q) =>
      later(300, () =>
        [
          { id: "weather-example", name: "Weather (example)", category: "Registry", description: "Forecasts for a place. An example of a registry server.", install: { command: "npx", args: ["-y", "@example/weather-mcp@1.2.0"] }, env: [{ name: "WEATHER_KEY", label: "Your API key", secret: true, url: "https://example.com" }], risk: "run" as const, offline: false, homepage: "https://example.com/weather-mcp" },
          { id: "notes-example", name: "Notes (example)", category: "Registry", description: "Search and write notes. An example of a registry server.", install: { command: "uvx", args: ["notes-mcp==0.3.1"] }, env: [], risk: "run" as const, offline: false, homepage: "https://example.com/notes-mcp" },
        ].filter((e) => !q || `${e.name} ${e.description}`.toLowerCase().includes(q.toLowerCase())),
      ),
    // Cluster: sharing is an in-memory flag; two example computers answer the search, and a load
    // returns example link figures (the preview connects to nothing).
    clusterShareGet: () => later(0, () => ({ ...clusterShare })),
    clusterShareStart: (id) =>
      later(150, () => {
        clusterShare = { ...clusterShare, sharing: true, model: MOCK_LIBRARY.find((m) => m.id === id)?.name ?? id, code: "482913", port: 50080 };
        return { ...clusterShare };
      }),
    clusterShareStop: () =>
      later(50, () => {
        clusterShare = { ...clusterShare, sharing: false, model: null, code: null, port: null };
        return { ...clusterShare };
      }),
    clusterDiscover: () =>
      later(600, () => [
        { name: "OFFICE-PC", addr: "192.168.1.20:50080", fingerprint: "ab".repeat(32) },
        { name: "MINI-PC", addr: "192.168.1.34:50080", fingerprint: "ab".repeat(32) },
      ]),
    clusterLoad: (_id, local, workers) =>
      later(500, () => {
        if (workers.some((w) => w.code.replace(/\D/g, "") !== "482913")) throw new Error("the worker at 192.168.1.20:50080 refused: this computer is not paired with the worker (wrong pairing code)");
        const n = 36;
        let lo = Math.max(1, Math.round(n * local / (local + workers.reduce((s, w) => s + w.share, 0))));
        return {
          links: workers.map((w, i) => {
            const hi = i === workers.length - 1 ? n : Math.min(n - 1, lo + Math.round((n - lo) / (workers.length - i)));
            const out = { addr: w.addr, lo, hi, rttMs: 0.9 + i * 0.4, mbPerS: 110 - i * 20 };
            lo = hi;
            return out;
          }),
        };
      }),
    // Cloud models: an in-memory list; "Test" offers example model names (nothing is contacted).
    cloudPresets: () =>
      later(0, () => [
        { id: "openai", name: "OpenAI", kind: "openAi" as const, baseUrl: "https://api.openai.com/v1", keysUrl: "https://platform.openai.com/api-keys" },
        { id: "anthropic", name: "Anthropic", kind: "anthropic" as const, baseUrl: "https://api.anthropic.com/v1", keysUrl: "https://console.anthropic.com/settings/keys" },
        { id: "gemini", name: "Google Gemini", kind: "gemini" as const, baseUrl: "https://generativelanguage.googleapis.com/v1beta", keysUrl: "https://aistudio.google.com/apikey" },
        { id: "openrouter", name: "OpenRouter", kind: "openAi" as const, baseUrl: "https://openrouter.ai/api/v1", keysUrl: "https://openrouter.ai/keys" },
      ]),
    cloudProviders: () => later(0, () => cloudList.map((p) => ({ ...p }))),
    saveCloudProvider: (p, apiKey) =>
      later(120, () => {
        if (!p.hasKey && !apiKey) throw new Error("Paste the API key first.");
        cloudList = [...cloudList.filter((x) => x.id !== p.id), { ...p, hasKey: true }];
        return cloudList.map((x) => ({ ...x }));
      }),
    removeCloudProvider: (id) =>
      later(80, () => {
        cloudList = cloudList.filter((x) => x.id !== id);
        return cloudList.map((x) => ({ ...x }));
      }),
    listCloudModels: (p, apiKey) =>
      later(300, () => {
        if (!apiKey && !p.hasKey) throw new Error("Paste the API key first.");
        if (apiKey === "bad") throw new Error("The provider refused the key: check it, or create a new one on its site.");
        return ["example-large", "example-medium", "example-small", "example-vision"];
      }),
    // The preview draws its own illustrative artwork (screens/ImageStudio.tsx); it never calls these.
    generateImage: () => Promise.reject(new Error("The browser preview does not make images.")),
    revealImage: () => later(0, () => false),
    // The preview has no audio: reading aloud takes about as long as speaking would.
    readAloud: (text) =>
      new Promise<void>((resolve) => {
        speechTimer = setTimeout(resolve, Math.min(15000, text.split(/\s+/).length * 380));
        speechDone = resolve;
      }),
    stopSpeaking: () =>
      later(0, () => {
        clearTimeout(speechTimer);
        speechDone?.();
      }),
    // A short phrase after the microphone starts (0.3 s to 1.5 s), then quiet: what voice mode's
    // end-of-speech detection hears in the preview.
    recordingLevel: () =>
      later(0, () => {
        const t = Date.now() - recordingSince;
        return t > 300 && t < 1500 ? 0.3 + Math.random() * 0.6 : 0.01;
      }),

    transcribeSample: (id, seconds) =>
      later(latency.test, () => {
        if (model(id).kind !== "stt") throw new TypeError("Not a speech-to-text model");
        if (typeof seconds !== "number" || !(seconds > 0) || seconds > 30) throw new TypeError("Recording length out of range");
        return { text: "This is a sample transcript. Your words would appear here, transcribed on this machine.", provenance: "illustrative" as const };
      }),

    speakSample: (id, text) =>
      later(latency.test, () => {
        if (model(id).kind !== "tts") throw new TypeError("Not a text-to-speech model");
        if (typeof text !== "string" || text.trim().length === 0 || text.length > 280) throw new TypeError("Sentence must be 1 to 280 characters");
        return Math.round(Math.max(1, text.trim().split(/\s+/).length * 0.38) * 10) / 10;
      }),

    startTune: (modelId, profile) =>
      later(0, () => {
        const m = model(modelId);
        if (!isLlm(m)) throw new TypeError("Only text and vision models can be tuned");
        if (!isTuneProfile(profile)) throw new TypeError("Unknown workload profile");
        return tune.start((runId) => {
          tuneScript = buildTuneScript(m, profile, { runId, integrityCheck: settings.integrityCheck, ...(options.failTune ? { fail: true } : {}) });
          return tuneScript;
        });
      }),

    subscribeTune: (listener) => tune.subscribe(listener),

    cancelTune: (runId) => later(0, () => tune.cancel(runId)),

    resumeTune: (runId) => later(0, () => tune.resume(runId)),

    getMetrics: (): Promise<MetricsSnapshot> =>
      options.failMetrics ? Promise.reject(new Error("Mock: engine not answering")) : later(0, () => monitor.snapshot(servedModel())),

    getMetricsHistory: (range): Promise<MetricsSeries> =>
      later(0, () => {
        if (!isHistoryRange(range)) throw new TypeError("Unknown history range");
        return monitor.history(range, servedModel());
      }),

    getRequest: (id): Promise<RequestDetail> =>
      later(0, () => {
        if (typeof id !== "string" || !REQUEST_ID.test(id)) throw new TypeError("Malformed request id");
        const d = monitor.request(id);
        if (!d) throw new Error("Request no longer kept");
        return d;
      }),

    listSwitches: () => later(0, () => switches.map((x) => ({ ...x }))),
    setSwitch: (id, value) =>
      later(0, () => {
        const sw = switches.find((x) => x.id === id);
        if (!sw) throw new Error(`unknown switch ${id}`);
        const v = sw.kind.type === "toggle" ? (value !== 0 ? 1 : 0) : sw.kind.zero_auto && value === 0 ? 0 : Math.min(sw.kind.max, Math.max(sw.kind.min, Math.trunc(value)));
        sw.value = v;
        return v;
      }),
    resetSwitches: () =>
      later(0, () => {
        for (const x of switches) x.value = x.default;
        return switches.map((x) => ({ ...x }));
      }),

    async tuneMachine(onTrial, signal) {
      // The engine needs a model in the llm slot; it says so with the same sentence the Rust side
      // uses when nothing is loaded.
      if (!loaded.some((l) => l.slot === "llm")) throw new Error("Load a model first (Models screen).");
      const outcome = await machineTune.play(onTrial, signal);
      // The engine keeps the winner through its switches; the mock applies the recorded values to
      // the switches it lists, so Turbo's list can pick them up after a refresh.
      for (const s of outcome.settings) {
        const sw = switches.find((x) => x.id === s.id);
        if (sw) sw.value = sw.kind.type === "toggle" ? (s.value !== 0 ? 1 : 0) : s.value;
      }
      lastTuneOutcome = outcome;
      return cloneMachineTuneOutcome(outcome);
    },
    lastMachineTune: () => later(0, () => (lastTuneOutcome ? cloneMachineTuneOutcome(lastTuneOutcome) : null)),

    mcpConfig: () => later(0, () => structuredClone(mcp)),
    mcpSaveConfig: (config) =>
      later(0, () => {
        mcp = structuredClone(config);
      }),
    mcpStatus: () =>
      later(0, () =>
        Object.entries(mcp.mcpServers)
          .filter(([, c]) => !c.disabled)
          .map(([name]): McpServerStatus => ({ name, state: "running", error: null, tools: name === "files" ? 2 : 0, stderr_tail: [], server_info: { name } })),
      ),
    mcpRestart: () => later(0, () => undefined),
    mcpTools: () =>
      later(0, (): McpTool[] =>
        mcp.mcpServers.files && !mcp.mcpServers.files.disabled
          ? [
              { server: "files", name: "read_file", description: "Read a file in the allowed folder", input_schema: {} },
              { server: "files", name: "list_directory", description: "List a folder", input_schema: {} },
            ]
          : [],
      ),
    mcpCall: (_server, tool) => later(0, () => ({ content: [{ type: "text", text: `Mock result of ${tool}` }], isError: false })),

    builtinTools: () => later(0, () => MOCK_BUILTIN_TOOLS.map((t) => ({ ...t }))),
    builtinFoldersGet: () => later(0, () => [...folders]),
    builtinFoldersSet: (list) =>
      later(0, () => {
        if (!Array.isArray(list) || list.length > 32 || !list.every((f) => typeof f === "string" && f.length > 0 && f.length <= 1024)) throw new TypeError("Invalid workspace folders");
        folders = [...new Set(list)];
        return [...folders];
      }),

    ...trainer,
    // Illustrative: the layer share that fits an 8 GB card, the rest in RAM (no drive reads).
    placementPreview: (id, contextLength, gpuLayers) =>
      later(0, () => {
        const m = MOCK_LIBRARY.find((x) => x.id === id);
        if (!m) throw new Error("No such model.");
        const total = m.arch.layers;
        const bytes = m.sizeGiB * 1024 ** 3;
        const per = bytes / total;
        const fit = Math.min(total, Math.floor((7.4 * 1024 ** 3 - contextLength * 1024) / per));
        const layers = Math.max(0, Math.min(fit, gpuLayers));
        const layout = layers === 0 ? "cpu" : layers >= total ? "gpu-int8" : "split";
        const cpu = bytes - layers * per;
        const est = 1 / ((layers * per) / 3e11 + cpu / 1.9e10);
        return {
          layout,
          gpuLayers: layers,
          totalLayers: total,
          vramBytes: layers * per,
          cpuBytes: cpu,
          ramResident: cpu,
          driveBytesPerToken: 0,
          estDecodeTps: est,
          drive: "nvme",
          notes: [layout === "split" ? `${layers} of ${total} layers on the GPU, the rest on the processor.` : layout === "cpu" ? "Everything runs on the processor." : "Every layer fits on the GPU."],
        } as const;
      }),
    localApiGet: () => later(0, apiView),
    localApiEnable: (on) =>
      later(0, () => {
        localApi = { ...localApi, enabled: on, running: on, url: on ? "http://127.0.0.1:47812/v1" : null };
        return apiView();
      }),
    localApiOptions: (requireKey, allowLan, allowModelSwitch) =>
      later(0, () => {
        localApi = { ...localApi, requireKey, allowLan, allowModelSwitch };
        return apiView();
      }),
    localApiRotateKey: () =>
      later(0, () => {
        localApi = { ...localApi, key: `sk-cerebro-${Math.random().toString(16).slice(2).padEnd(40, "0")}` };
        return apiView();
      }),
    mcpServerGet: () => later(0, () => ({ ...mcpSrv })),
    mcpServerEnable: (on) =>
      later(0, () => {
        mcpSrv = { ...mcpSrv, enabled: on, running: on, url: on ? "http://127.0.0.1:47811/mcp" : null };
        return { ...mcpSrv };
      }),
    mcpServerRotateToken: () =>
      later(0, () => {
        mcpSrv = { ...mcpSrv, token: `mock-${Math.random().toString(36).slice(2)}-token-0123456789abcdef`, clients: [] };
        return { ...mcpSrv };
      }),
    remoteGet: () => later(0, () => remoteView()),
    remoteSetPassword: (password) =>
      later(0, () => {
        if (password.length < 8) throw new Error("Use at least 8 characters.");
        remote = { ...remote, passwordSet: true, devices: [] };
        return remoteView();
      }),
    remoteEnable: (on) =>
      later(0, () => {
        if (on && !remote.passwordSet) throw new Error("Set a password first.");
        remote = { ...remote, enabled: on };
        return remoteView();
      }),
    remoteOptions: (o) =>
      later(0, () => {
        remote = { ...remote, allowChat: o.allowChat, allowTailscale: o.allowTailscale, addressChoice: o.address, sessionDays: o.sessionDays };
        return remoteView();
      }),
    remotePair: () =>
      later(0, () => {
        if (!remote.enabled) throw new Error("Turn phone access on first.");
        // A plain square stands in for the QR: the web build has no server to pair with.
        return { url: "https://192.168.1.20:51234/#pair=example", qrSvg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="#fff"/><rect x="1" y="1" width="3" height="3"/></svg>', expiresAt: Date.now() + 600_000 };
      }),
    remoteRevoke: (id) =>
      later(0, () => {
        remote = { ...remote, devices: id === "*" ? [] : remote.devices.filter((d) => d.id !== id) };
        return remoteView();
      }),

    chat: (request, onEvent, signal) => chatEngine.chat(request, onEvent, signal),
    chatToolDecide: (id, allow) => chatEngine.decide(id, allow),
    chatToolAnswer: (id, answer) => chatEngine.answer(id, answer),

    transcribe: (seconds) => chatEngine.transcribe(seconds),
    // The preview cannot run a speech model: it returns an example transcript, and the screen says
    // the data is illustrative.
    transcribeAudio: (audio) =>
      Promise.resolve(
        `The weekly sync agreed three things: the search release on Friday, the new pricing page, and who owns the support rota. (Example transcript for ${audio.name}; the desktop app transcribes the real file.)`,
      ),
  };
}
