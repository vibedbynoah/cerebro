/**
 * Domain types shared by the UI and every CerebroService implementation (the mock today, the Tauri
 * IPC bridge later). Plain data only: everything here must survive a JSON round trip over IPC.
 */

/** Where a value came from. Anything not `measured` must be labelled as such wherever it is shown
 *  (docs/CLAIMS.md): the mock only ever produces `illustrative`. */
export type Provenance = "measured" | "illustrative";

export type BackendState = "stopped" | "starting" | "running" | "error";

export interface BackendStatus {
  state: BackendState;
  /** Loopback address the OpenAI-compatible API listens on while running, e.g. "127.0.0.1:8181". */
  address: string | null;
  /** Cerebro Engine build the backend was verified against, e.g. "b6500". */
  build: string;
}

export interface GpuInfo {
  name: string;
  vramTotalGiB: number;
  /** Measured tensor-core rates (an overclock or an unlock mod shows up here); absent when not measured. */
  tensor?: TensorInfo | null;
}

export interface TensorInfo {
  int8Tops: number;
  int4Tops: number;
  f16Tflops: number;
  /** FP16 into FP32 runs at the full (professional-card) rate: a GeForce card with the limit unlocked. */
  f16FullRate: boolean;
  smClockMhz: number | null;
}

/** One HUD sample: what the status bar shows right now. */
export interface LiveStats {
  provenance: Provenance;
  /** Name of the loaded model, or null when none is loaded. */
  model: string | null;
  /** Decode speed of the last generation, tokens per second; null when idle and never run. */
  tokensPerSecond: number | null;
  vramUsedGiB: number;
  vramTotalGiB: number;
  backend: BackendStatus;
}

export type TuneState = "untuned" | "tuning" | "tuned" | "defaults-fastest";

export interface ModelSummary {
  id: string;
  name: string;
  /** Quantization label, e.g. "Q4_K_M". */
  quant: string;
  /** Parameter count as published, e.g. "30B-A3B". */
  params: string;
  architecture: "dense" | "moe";
  sizeGiB: number;
  contextMax: number;
  /** The longest context Cerebro runs it at, stretched with YaRN past `contextMax` (desktop). */
  contextExtended?: number;
  tune: TuneState;
  /** Decode tok/s on defaults and tuned, when a tune has finished. */
  defaultsTps: number | null;
  tunedTps: number | null;
  provenance: Provenance;
}

export type NetworkState = "online" | "offline";

/**
 * Every protection and network feature the user controls. Safe by default; every one of these can
 * be switched off in Settings, and the UI never blocks or nags when one is.
 */
export interface Settings {
  /** Start the model process in a Windows Job Object that ends it, and anything it started, when
   *  Cerebro stops. Does not restrict file or network access (docs/CLAIMS.md). */
  processContainment: boolean;
  /** Rebuild the model process's environment from a closed allowlist instead of inheriting it. */
  cleanEnvironment: boolean;
  /** Verify the backend binary's hash before every start. */
  integrityCheck: boolean;
  /** Ask before a tool or plugin performs an action. */
  permissionPrompts: boolean;
  /** Web fetches by a model may not reach this computer or the local network. */
  blockLocalFetch: boolean;
  /** Commands a model runs do not inherit environment variables that look like secrets. */
  hideSecretsFromTools: boolean;
  /** A model must read a file (and it must be unchanged since) before changing it. */
  readBeforeWrite: boolean;
  /** Save a long conversation's reading of its history to disk, so returning to it continues
   *  instead of running the model through every message again. */
  keepChatCaches: boolean;
  /** Serve the API on 127.0.0.1 only. Off lets other devices on the network reach it. */
  loopbackOnly: boolean;
  /** Block every network feature (model downloads, plugin installs). */
  offlineMode: boolean;
  allowModelDownloads: boolean;
  allowPluginInstalls: boolean;
}

export type SettingKey = keyof Settings;

/** What the saved-chat folder holds ("Keep long chats ready"): the cache files and their total size. */
export interface KvStoreUsage {
  files: number;
  bytes: number;
}

export interface AppInfo {
  name: string;
  version: string;
  /** "desktop" inside the Tauri shell, "web" in a plain browser with the mock service. */
  mode: "desktop" | "web";
}

/** What first-run setup reads to size models. The mock only ever produces `illustrative`. */
export interface HardwareProfile {
  provenance: Provenance;
  cpu: { name: string; cores: number; threads: number };
  /** The graphics card Cerebro Engine can offload to, or null when there is none it can use. */
  gpu: GpuInfo | null;
  ramTotalGiB: number;
}

/** A model setup can recommend before anything is downloaded. */
export interface CatalogModel {
  id: string;
  name: string;
  /** Parameter count as published, e.g. "30B-A3B". */
  params: string;
  /** For a mixture-of-experts model, the parameters active per token as published, e.g. "3B". */
  activeParams: string | null;
  architecture: "dense" | "moe";
  quant: string;
  /** File size of the quantized GGUF, GiB. */
  sizeGiB: number;
  /** One short phrase, e.g. "Coding agent". */
  goodFor: string;
  provenance: Provenance;
  /** Transformer blocks, for offload arithmetic (as published). */
  layers: number;
  /** Attention heads and the size of each, for the KV cache (as published). */
  kvHeads: number;
  headDim: number;
  /** Residual width, for the compute buffers (as published). */
  embedding: number;
  /** Longest context the model was trained for, tokens (as published). */
  contextMax: number;
  /** The publisher's recommended sampler values, Beginner's starting point. */
  sampler: CatalogSampler;
  /** Every quantised file Cerebro lists for this model, best formats first. `quant`/`sizeGiB` above
   *  name the general-purpose one setup ranks by. */
  quants: CatalogQuant[];
}

export interface CatalogSampler {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
}

/** Quantisation families the catalogue lists. Each is a different way to store fewer bits per weight. */
export type QuantFormat = "gsq-rco" | "unsloth-ud" | "imatrix-iq" | "k-quant" | "exl3" | "awq" | "mlx";

/** The engine a file needs. Only `Cerebro Engine` and `strata` files run in this build. */
export type QuantBackend = "cerebro" | "strata" | "exllamav3" | "vllm" | "mlx";

/**
 * One downloadable file of a model. Quality is measured against the unquantised (BF16) weights:
 * KL divergence of the next-token distribution and the perplexity increase. Speed is decode tok/s on
 * this machine, null where the file cannot run here. All three carry `provenance`.
 */
export interface CatalogQuant {
  id: string;
  /** File label as published, e.g. "UD-Q4_K_XL". */
  label: string;
  format: QuantFormat;
  backend: QuantBackend;
  bitsPerWeight: number;
  sizeGiB: number;
  /** Mean KL divergence against BF16. 0 is no change. */
  klDiv: number;
  /** Perplexity increase against BF16, percent. */
  pplDeltaPct: number;
  /** Decode tok/s on this machine with Cerebro's starting settings, or null when it cannot run here. */
  decodeTps: number | null;
  /** Why it cannot run here, when `decodeTps` is null. */
  unavailable: string | null;
  provenance: Provenance;
}

// ---------------------------------------------------------------------------------------------
// Models screen: library, load parameters, tweaks, GGUF metadata, disk space
// ---------------------------------------------------------------------------------------------

/** What a model can take in or produce. Shown as badges on every model. */
export type Modality = "text" | "vision" | "audio-in" | "audio-out" | "image-gen";

/**
 * How a model runs. `text` and `vision` load into Cerebro Engine (one slot: loading another
 * replaces it); `stt`, `tts` and `image` each load into their own lightweight engine beside it.
 */
export type ModelKind = "text" | "vision" | "stt" | "tts" | "image";

/** The folder a model file was found in. Cerebro reads other apps' folders; it never moves files. */
export type ModelSource = "cerebro" | "lmstudio" | "ollama" | "other";

export type KvCacheType = "f16" | "q8_0" | "q4_0";
export type FlashAttention = "auto" | "on" | "off";
export type CpuAffinity = "all" | "performance" | "custom";
export type SpeculativeMode = "off" | "ngram" | "draft";
/** The draft-free speculative strategies (docs/CLAIMS.md). */
export type NgramStrategy = "ngram-simple" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod" | "ngram-cache";

/** Every Cerebro Engine load parameter the load panel exposes. Validated by lib/load-params.ts on both
 *  sides of the service boundary. */
export interface LoadParams {
  /** Tokens the model can keep in view at once. */
  contextLength: number;
  /** Layers placed on the GPU, 0..arch.layers. */
  gpuLayers: number;
  /** MoE only: the expert weights of this many layers stay in system memory. 0 for dense models. */
  moeCpuLayers: number;
  kvCacheK: KvCacheType;
  kvCacheV: KvCacheType;
  flashAttention: FlashAttention;
  threads: number;
  cpuAffinity: CpuAffinity;
  /** Core list for `custom` affinity, e.g. "0-5,8". Empty otherwise. */
  cpuCores: string;
  batchSize: number;
  ubatchSize: number;
  mmap: boolean;
  mlock: boolean;
  speculative: SpeculativeMode;
  ngramStrategy: NgramStrategy;
  /** Library id of the draft model for `draft` speculation, else null. */
  draftModelId: string | null;
  draftMax: number;
}

export interface ModelArch {
  layers: number;
  /** Key/value attention heads and the size of each: these set the KV cache's memory per token. */
  kvHeads: number;
  headDim: number;
  embedding: number;
  /** MoE only: experts per layer and experts used per token. */
  experts: number | null;
  activeExperts: number | null;
}

/**
 * What a finished tune measured for one file (docs/CLAIMS.md "About measurement"): three numbers,
 * never one, and every condition they were measured under. A speed is never shown without these.
 */
export interface TuneResult {
  /** Prompt reading (prefill), tok/s. */
  prefillTps: number;
  /** Writing (decode) with an empty context, tok/s. */
  decodeTps0: number;
  /** Writing (decode) with `depth` tokens already in the context, tok/s. */
  decodeTpsDepth: number;
  /** The working context depth `decodeTpsDepth` was measured at, tokens. */
  depth: number;
  /** Cerebro Engine build that produced the numbers, e.g. "b6500". */
  build: string;
  /** The graphics card (or "CPU only") it ran on. */
  gpu: string;
  /** Sampler settings during the run, e.g. "greedy" or "temp 0.6, top-p 0.95". */
  sampler: string;
  /** ISO date of the run. */
  measuredAt: string;
  provenance: Provenance;
}

/** A model file in the library. The single-number `defaultsTps`/`tunedTps` summary is left out on
 *  purpose: the Models screen only ever shows a speed as a full `TuneResult` with its conditions. */
export interface LibraryModel extends Omit<ModelSummary, "defaultsTps" | "tunedTps"> {
  kind: ModelKind;
  modalities: Modality[];
  /** Publisher family for grouping and draft-model compatibility, e.g. "Qwen3". */
  family: string;
  file: string;
  /** Full path, display only. */
  path: string;
  source: ModelSource;
  /** SHA-256 of the file, hex. Patch profiles are keyed by it. */
  sha256: string;
  arch: ModelArch;
  /** The fastest safe settings the last tune kept, or null when never tuned. */
  tunedParams: LoadParams | null;
  /** ISO date of the last tune, or null. */
  tunedAt: string | null;
  /** What the last tune measured with `tunedParams`, or null when never tuned. */
  tuneResult: TuneResult | null;
  /** Vision only: the projector (mmproj) file paired with it, or null when none was found. */
  projector: { file: string; sizeGiB: number } | null;
  /** False when the engine cannot run its architecture yet (desktop); absent means it can. */
  supported?: boolean;
}

/** Engine slot a loaded model occupies. */
export type EngineSlot = "llm" | "stt" | "tts" | "image";

export interface LoadedModel {
  id: string;
  slot: EngineSlot;
  /** The settings it was loaded with; null for engines without load parameters. */
  params: LoadParams | null;
  /** True when it was loaded with the tuned settings unchanged. */
  tuned: boolean;
  /** Where the engine put it (desktop). */
  placement?: Placement | null;
}

/** Where a model's weights go (cerebro_engine::placement::Plan). Bytes; speeds are estimates. */
export interface Placement {
  layout: "gpu-int8" | "gpu-native" | "split" | "cpu";
  gpuLayers: number;
  totalLayers: number;
  vramBytes: number;
  /** Weights the processor reads per token, how many stay in RAM, and how many come from the drive. */
  cpuBytes: number;
  ramResident: number;
  driveBytesPerToken: number;
  estDecodeTps: number;
  drive: "nvme" | "sata-ssd" | "hard-disk" | "removable" | "unknown";
  notes: string[];
}

export interface LoadPreset {
  id: string;
  modelId: string;
  name: string;
  params: LoadParams;
}

export interface Sampler {
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  repeatPenalty: number;
}

export type RopeScaling = "none" | "linear" | "yarn";

/**
 * Small, reversible tweaks applied when the model loads. The file itself never changes. `null`
 * means "as the file says".
 */
export interface Tweaks {
  kvQuant: KvCacheType | null;
  contextOverride: number | null;
  rope: { scaling: RopeScaling; factor: number; originalContext: number } | null;
  chatTemplate: string | null;
  stopTokens: string[];
  sampler: Sampler | null;
}

/** The tweaks and switched-on fixes in effect for one exact file: what the next load applies. */
export interface ActivePatch {
  tweaks: Tweaks;
  fixes: string[];
}

/** A named, shareable set of tweaks and switched-on fixes for one exact file (by SHA-256). */
export interface PatchProfile {
  id: string;
  name: string;
  modelSha256: string;
  tweaks: Tweaks;
  fixes: string[];
}

/** One entry from the quirks database: a known problem with a model family and its fix. */
export interface QuirkFix {
  id: string;
  title: string;
  /** What goes wrong without it, in plain words. */
  problem: string;
  /** What switching it on changes. */
  change: string;
  provenance: Provenance;
}

export type GgufType = "string" | "uint32" | "int32" | "float32" | "bool" | "array";

export interface GgufEntry {
  key: string;
  type: GgufType;
  /** Arrays are summarised ("151,936 strings"), never sent whole. */
  value: string | number | boolean;
  /** False for structural keys: changing them would make the file unreadable. */
  editable: boolean;
}

export interface GgufEdit {
  key: string;
  value: string | number | boolean;
}

/** A metadata-edited copy of a model file. The original is never written. */
export interface GgufCopy {
  path: string;
  edits: GgufEdit[];
}

export interface ModelDetail {
  id: string;
  gguf: GgufEntry[] | null;
  /** The chat template stored in the file, or null when it has none. */
  chatTemplate: string | null;
  quirks: QuirkFix[];
  /** What the model's authors publish for long context and sampling, when they do. */
  author: { rope: Tweaks["rope"]; sampler: Sampler | null };
  copy: GgufCopy | null;
}

export interface DuplicateCopy {
  path: string;
  source: ModelSource;
  /** Drive, as Windows names it, e.g. "C:". Hard links only work within one drive. */
  volume: string;
}

export interface DuplicateGroup {
  id: string;
  file: string;
  sizeGiB: number;
  sha256: string;
  copies: DuplicateCopy[];
  state: "duplicate" | "linked";
  /** Linked only: the path of the copy the others now point to. */
  kept: string | null;
}

export interface DiskReport {
  provenance: Provenance;
  totalGiB: number;
  groups: DuplicateGroup[];
}

export interface DedupeResult {
  group: DuplicateGroup;
  reclaimedGiB: number;
  /** Copies that could not be linked because they sit on another drive. */
  skipped: number;
}

/** Output of a "try it" test. The mock's is always illustrative. */
/** Cluster: whether this computer is shared as a worker (screens/Cluster.tsx). */
export interface ClusterShare {
  sharing: boolean;
  model: string | null;
  code: string | null;
  port: number | null;
  addresses: string[];
  computer: string;
}

/** A sharing computer heard on the local network. */
export interface ClusterWorker {
  name: string;
  addr: string;
  fingerprint: string;
}

/** How one link measured when a model loaded across computers. */
export interface ClusterLink {
  addr: string;
  lo: number;
  hi: number;
  rttMs: number;
  mbPerS: number;
}

/** A cloud provider Cerebro knows: its official API address and where to get a key. */
export interface CloudPreset {
  id: string;
  name: string;
  kind: "openAi" | "anthropic" | "gemini";
  baseUrl: string;
  keysUrl: string;
}

/** A cloud provider the user added (Settings > Cloud models). The key is never sent back. */
export interface CloudProvider {
  id: string;
  name: string;
  kind: "openAi" | "anthropic" | "gemini";
  baseUrl: string;
  /** The provider's model ids shown in Chat. */
  models: string[];
  hasKey: boolean;
}

/** An image Image Studio made on this machine (desktop). */
export interface GeneratedImage {
  /** The PNG as a data: URL. */
  dataUrl: string;
  /** Where it was saved. */
  path: string;
  seed: number;
  /** Time to make it, measured. */
  seconds: number;
  width: number;
  height: number;
}

export interface TestOutput {
  text: string;
  provenance: Provenance;
}

// ---------------------------------------------------------------------------------------------
// Tune screen: a tune run, streamed as events
// ---------------------------------------------------------------------------------------------

/** What the model will mostly be used for. It sets the working depth and what "fastest" means. */
export type TuneProfile = "chat" | "coding" | "long-docs";

/** The stages of a tune, in order (docs/CLAIMS.md "About measurement"; lib/tune.ts PHASE_IDS). The
 *  quality gate runs before the soak, so no long soak is spent on a setting that fails it. */
export type TunePhase = "probe" | "screen" | "measure" | "quality" | "soak" | "decide";

/** One setting a candidate configuration changes: plain words, plus a Cerebro Engine flag when it has
 *  a well-known one (null otherwise, never an invented flag). */
export interface TuneSetting {
  name: string;
  value: string;
  flag: string | null;
}

export interface TuneCandidate {
  id: string;
  /** Plain-words name, e.g. "Experts of 30 layers on the CPU". */
  label: string;
  /** Cerebro Engine's own automatic settings (its built-in fit): the baseline every candidate must beat. */
  baseline: boolean;
  settings: TuneSetting[];
}

/** Everything a run was started with and is measured under. Rule zero of docs/CLAIMS.md: a speed is
 *  never shown without the machine, model, quant, build, depth and sampler, so all of them are here.
 *  "The machine" is the graphics card AND the processor and system memory: expert weights and pinned
 *  threads run there. "The build" is the exact binary (build number plus commit). The desktop bridge
 *  must check every one of these is present and well-formed before it forwards a run to the UI. */
export interface TuneRunInfo {
  runId: string;
  modelId: string;
  modelName: string;
  quant: string;
  architecture: "dense" | "moe";
  sizeGiB: number;
  profile: TuneProfile;
  /** Working context depth, tokens: decode is measured with this much already in view. */
  depth: number;
  /** Sampler during measurement, as values. */
  sampler: string;
  /** Cerebro Engine build the backend runs, e.g. "b6500". */
  build: string;
  /** Commit (or binary hash) of that exact backend build: measurements are keyed to the binary. */
  backendCommit: string;
  gpu: string;
  vramTotalGiB: number;
  /** The processor, with its core and thread counts: expert weights and pinned threads run here. */
  cpu: string;
  /** System memory, GiB: where offloaded layers and expert weights live. */
  ramTotalGiB: number;
  /** Length of one soak, seconds of run time. */
  soakSeconds: number;
  /** How many candidates will be screened and how many survivors soaked (for overall progress). */
  plannedCandidates: number;
  plannedSoaks: number;
  /** False when the integrity check is switched off in Settings; the probe then says so. */
  integrityCheck: boolean;
  provenance: Provenance;
}

export interface TuneMeasurement {
  prefillTps: number;
  decodeTps0: number;
  decodeTpsDepth: number;
}

/** Payload of one tune event. `TuneEvent` adds the sequence number and run clock. */
export type TuneEventBody =
  | { kind: "started"; run: TuneRunInfo }
  | { kind: "phase"; phase: TunePhase }
  | { kind: "probe"; vramFreeGiB: number; integrity: "verified" | "off" }
  | { kind: "candidate"; candidate: TuneCandidate }
  | { kind: "screen-start"; candidateId: string }
  /** One live writing-speed reading. Repetition 1 is warm-up and is always discarded. */
  | { kind: "sample"; candidateId: string; rep: number; t: number; tps: number }
  | { kind: "screened"; candidateId: string; tps: number | null; outcome: "advance" | "discard" | "fail"; reason: string }
  | ({ kind: "measured"; candidateId: string } & TuneMeasurement)
  | { kind: "soak-start"; candidateId: string }
  | { kind: "soak-sample"; candidateId: string; t: number; vramFreeGiB: number; tps: number }
  | { kind: "soak-end"; candidateId: string; passed: boolean; slopeMiBPerMin: number; reason: string }
  | { kind: "quality"; candidateId: string; metric: string; value: number; gate: number; passed: boolean; reference: string }
  | { kind: "winner"; candidateId: string; baselineId: string; reasons: string[] }
  | { kind: "cancelled" }
  | { kind: "resumed" }
  | { kind: "failed"; reason: string };

export type TuneEvent = TuneEventBody & {
  /** Increases by one per event across every run, so a replayed event is recognised. */
  seq: number;
  /** Run clock, seconds since the run started (time spent cancelled does not count). */
  at: number;
};

/** The long-context check's outcome (engine longctx.rs). */
export interface ContextCheckResult {
  context: number;
  trained: number;
  /** YaRN stretch, 1 within the trained length. */
  factor: number;
  probes: Array<{ tokens: number; depth: number; found: boolean; reply: string; seconds: number }>;
}

/** A mixture-of-experts model's experts as they fire (engine activity.rs). */
export interface MoeActivity {
  layers: number;
  experts: number;
  /** 0-255 per (layer, expert), layer-major, scaled by the hottest. */
  heat: number[];
  /** The experts the latest token picked, per layer. */
  last: number[][];
  tokens: number;
}

/** How a Model Hub search is ordered (Hugging Face's sort keys). */
export type HubSort = "downloads" | "trendingScore" | "likes" | "createdAt" | "lastModified";

/** A Hugging Face repository with GGUF files (Model Hub search result). */
export interface HubRepo {
  id: string;
  author: string;
  downloads: number;
  likes: number;
  /** ISO date of the last change. */
  updated: string;
  /** The repository's `license:` tag, e.g. "apache-2.0", or null when it publishes none. */
  license: string | null;
  tags: string[];
}

/** One downloadable GGUF file; a split model is one entry with all its parts. */
export interface HubFile {
  path: string;
  /** Bytes, all parts together. */
  size: number;
  parts: string[];
  quant: string;
  downloaded: boolean;
}

export type HubDownloadEvent =
  | { type: "progress"; done: number; total: number; bytesPerSec: number }
  | { type: "done"; path: string }
  | { type: "error"; message: string };

/** The architecture numbers and real size the Hub reads from a GGUF header (hub_file_meta). */
export interface HubFileMeta {
  /** Bytes of the whole file. */
  size: number;
  /** `general.architecture`, e.g. "qwen3moe". */
  architecture: string;
  layers: number;
  kvHeads: number;
  headDim: number;
  embedding: number;
  /** MoE only: experts per layer, or null for a dense model. */
  experts: number | null;
  /** MoE only: experts used per token. */
  activeExperts: number | null;
  /** Longest context the model was trained for, tokens. */
  contextMax: number;
  /** Layers with a KV cache (fewer than `layers` for Qwen3.5/3.8 hybrids). */
  attentionLayers: number;
  /** Quant label from the file name, e.g. "UD-Q4_K_XL". */
  quant: string;
}

/** One engine optimisation the user can change (engine switches registry, Turbo screen). */
export type SwitchKind = { type: "toggle" } | { type: "number"; min: number; max: number; zero_auto: boolean; unit: string };
export interface EngineSwitch {
  id: string;
  name: string;
  group: string;
  kind: SwitchKind;
  channel: "stable" | "beta" | "experimental";
  /** When a change takes effect. */
  applies: "live" | "reload" | "restart";
  beginner: string;
  advanced: string;
  env: string;
  value: number;
  default: number;
}

/** One processor setting and its value (cerebro_engine::tune::Setting). */
export interface MachineTuneSetting {
  id: string;
  value: number;
}

/** One timed setting from a tune for this machine (cerebro_engine::tune::Trial). */
export interface MachineTuneTrial {
  phase: "threads" | "splitting" | "rows4" | "confirm";
  /** Plain words, e.g. "12 threads". */
  label: string;
  settings: MachineTuneSetting[];
  /** Median writing speed, tokens per second. */
  tps: number;
  reps: number[];
}

/** A finished tune for this machine (cerebro_engine::tune::Outcome, camelCase over IPC). */
export interface MachineTuneOutcome {
  /** The processor, graphics card and core layout the result belongs to. */
  machine: string;
  /** Model fingerprint, as hex. */
  model: string;
  /** Layers on the GPU when measured (0 = processor only). */
  gpuLayers: number;
  /** The settings in force afterwards (the winner, or the starting ones). */
  settings: MachineTuneSetting[];
  start: MachineTuneSetting[];
  startTps: number;
  tunedTps: number;
  /** tuned / start - 1, from the alternating confirmation. */
  gain: number;
  kept: boolean;
  /** How busy the processor was (0..1) from other programs, or null. */
  busy: number | null;
  /** One plain sentence about the result. */
  note: string;
  trials: MachineTuneTrial[];
  seconds: number;
}

/** Streamed while tuning: each finished trial, then the outcome or the reason it stopped. */
export type MachineTuneEvent = ({ type: "trial" } & MachineTuneTrial) | { type: "done"; outcome: MachineTuneOutcome } | { type: "error"; message: string };

/** MCP servers (mcp.json, standard format). A server is a local program (command) or a web address (url). */
export interface McpServerConfig {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  headers?: Record<string, string>;
  disabled: boolean;
}
export interface McpConfig {
  mcpServers: Record<string, McpServerConfig>;
}
export interface McpServerStatus {
  name: string;
  state: "starting" | "running" | "error" | "stopped";
  error: string | null;
  tools: number;
  stderr_tail: string[];
  server_info: unknown;
}
export interface McpTool {
  server: string;
  name: string;
  description: string | null;
  input_schema: unknown;
}
export interface McpCallResult {
  content?: Array<{ type: string; text?: string }>;
  isError?: boolean;
}

/** Phone access (src-tauri/src/remote.rs): a paired phone and the settings screen's view. */
export interface RemoteDevice {
  id: string;
  name: string;
  createdAt: number;
  lastSeenAt: number;
  ip: string;
}
export interface RemoteView {
  enabled: boolean;
  running: boolean;
  url: string | null;
  /** Where it listens now, or the saved choice while off. */
  address: string | null;
  /** The saved choice; null = automatic. */
  addressChoice: string | null;
  port: number;
  addresses: string[];
  passwordSet: boolean;
  allowChat: boolean;
  allowTailscale: boolean;
  sessionDays: number;
  fingerprint: string | null;
  devices: RemoteDevice[];
  error: string | null;
}
export interface RemotePairing {
  url: string;
  qrSvg: string;
  expiresAt: number;
}
export interface RemoteOptions {
  allowChat: boolean;
  allowTailscale: boolean;
  address: string | null;
  sessionDays: number;
}

/** Cerebro as an MCP server for other apps on this computer (src-tauri/src/mcp_server.rs). */
/** The local OpenAI/Anthropic-compatible API (src-tauri/src/local_api.rs). */
export interface LocalApiView {
  enabled: boolean;
  running: boolean;
  /** Base URL on this computer, ending in /v1. */
  url: string | null;
  /** Base URLs on the local network (when allowed and running). */
  lanUrls: string[];
  port: number;
  key: string;
  requireKey: boolean;
  allowLan: boolean;
  allowModelSwitch: boolean;
  requests: number;
  lastRequestAt: number | null;
  error: string | null;
}

export interface McpServerView {
  enabled: boolean;
  running: boolean;
  url: string | null;
  port: number;
  token: string;
  clients: Array<{ name: string; version: string; lastSeenAt: number; calls: number }>;
  error: string | null;
}
