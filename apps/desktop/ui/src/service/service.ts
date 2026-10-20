import type { TrainerStatus, TrainEvent, TrainInstalled, TrainJob } from "./train-types";
import type { Placement } from "./types";
import type { EngineSwitch, LocalApiView, McpServerView, RemoteOptions, RemotePairing, RemoteView, McpCallResult, McpConfig, McpServerStatus, McpTool,
  ActivePatch,
  CatalogModel,
  DedupeResult,
  DiskReport,
  DuplicateGroup,
  GgufCopy,
  GgufEdit,
  GpuInfo,
  HardwareProfile,
  LibraryModel,
  LiveStats,
  LoadedModel,
  LoadParams,
  LoadPreset,
  MachineTuneOutcome,
  MachineTuneTrial,
  ModelDetail,
  ModelSummary,
  NetworkState,
  PatchProfile,
  SettingKey,
  Settings,
  KvStoreUsage,
  TestOutput,
  GeneratedImage,
  TuneEvent,
  TuneProfile,
  TuneRunInfo,
  HubDownloadEvent,
  HubFile,
  HubFileMeta,
  HubRepo,
} from "./types";

import type { ChatEvent, ChatRequest } from "./chat-types";
import type { HistoryRange, MetricsSeries, MetricsSnapshot, RequestDetail } from "./monitor-types";

export type Unsubscribe = () => void;

/**
 * Everything the UI reads or changes. The mock (src/mock) implements it for web mode and tests; the
 * Tauri IPC bridge will implement the same interface, so no screen ever knows which one it has.
 *
 * Rules for implementations:
 * - Never throw synchronously; report failure through a rejected promise.
 * - Return copies, never shared mutable objects.
 * - `updateSetting` validates both key and value and rejects anything else.
 */
export interface CerebroService {
  getLiveStats(): Promise<LiveStats>;
  /** Calls `listener` with each new HUD sample until unsubscribed. */
  subscribeLiveStats(listener: (stats: LiveStats) => void): Unsubscribe;
  listModels(): Promise<ModelSummary[]>;
  getGpu(): Promise<GpuInfo>;
  /** Processor, graphics card and memory, for first-run setup and model sizing. */
  detectHardware(): Promise<HardwareProfile>;
  /** Models setup may recommend. Listing needs no network; downloading one does. */
  listCatalog(): Promise<CatalogModel[]>;
  getNetwork(): Promise<NetworkState>;
  getSettings(): Promise<Settings>;
  updateSetting<K extends SettingKey>(key: K, value: Settings[K]): Promise<Settings>;
  /** What the saved-chat folder holds ("Keep long chats ready"). */
  kvStoreUsage(): Promise<KvStoreUsage>;
  /** Deletes every saved chat cache; resolves with what was removed. */
  kvStoreClear(): Promise<KvStoreUsage>;

  // ---- Models screen. Every id argument must name a library model; every other argument is
  // validated with the same rules as the UI (lib/load-params.ts, lib/tweaks.ts, lib/gguf.ts). ----

  /** Every model file found in Cerebro's, LM Studio's, Ollama's and the user's added folders. */
  listLibrary(): Promise<LibraryModel[]>;
  /** Heavy per-model data the list does not need: GGUF metadata, template, quirks. */
  getModelDetail(id: string): Promise<ModelDetail>;
  listLoaded(): Promise<LoadedModel[]>;
  /** `params` null loads the tuned settings (or Cerebro's starting settings when never tuned).
   *  A text or vision model replaces the one in the llm slot; other engines load beside it. */
  loadModel(id: string, params: LoadParams | null): Promise<LoadedModel[]>;
  unloadModel(id: string): Promise<LoadedModel[]>;
  listPresets(modelId: string): Promise<LoadPreset[]>;
  savePreset(modelId: string, name: string, params: LoadParams): Promise<LoadPreset[]>;
  deletePreset(modelId: string, presetId: string): Promise<LoadPreset[]>;
  /** The tweaks in effect for one exact file (none until the user sets some). */
  getActivePatch(sha256: string): Promise<ActivePatch>;
  /** Replaces them. The next load of that file applies them; the file itself is never changed. */
  setActivePatch(sha256: string, patch: ActivePatch): Promise<ActivePatch>;
  /** Profiles saved for one exact file. */
  listPatchProfiles(sha256: string): Promise<PatchProfile[]>;
  savePatchProfile(profile: Omit<PatchProfile, "id">): Promise<PatchProfile[]>;
  deletePatchProfile(sha256: string, id: string): Promise<PatchProfile[]>;
  /** Writes the edits to a copy beside the original; the original is never opened for writing.
   *  When a copy exists the edits are merged into it: a key set to its original value is dropped
   *  from the copy, every other key keeps its newest value. */
  writeGgufCopy(id: string, edits: GgufEdit[]): Promise<GgufCopy>;
  /** Deletes the copy, so the original is the only file again. */
  revertGgufCopy(id: string): Promise<void>;
  scanDisk(): Promise<DiskReport>;
  /** Keeps the copy at `keepPath` and replaces the others on its drive with hard links to it, after
   *  checking every hash. */
  deduplicate(groupId: string, keepPath: string): Promise<DedupeResult>;
  /** Undoes a deduplicate: copies the bytes back so every app has its own file again. Needs the
   *  file's size free once per linked copy. */
  unlink(groupId: string): Promise<DuplicateGroup>;
  /** Opens File Explorer at the file (desktop). Resolves false where that is not possible. */
  revealInFolder(id: string): Promise<boolean>;
  /** The image test: `data` is the image as a data: URL (desktop reads it; the preview does not). */
  describeImage(id: string, image: { name: string; type: string; bytes: number; data?: string }): Promise<TestOutput>;
  /** Stops the recording `startRecording` began and transcribes it with model `id`. */
  transcribeSample(id: string, seconds: number): Promise<TestOutput>;
  /** Starts recording from the microphone (push-to-talk, or a speech model's test); `transcribe`
   *  and `transcribeSample` stop it. */
  startRecording(): Promise<void>;
  /** The microphone's recent peak level while recording, 0..1, for the waveform. */
  recordingLevel(): Promise<number>;
  /** Resolves with the length of the spoken audio, in seconds, as it starts playing. */
  speakSample(id: string, text: string): Promise<number>;
  /** Image Studio: makes an image with an image model (desktop). */
  generateImage(req: { modelId: string; prompt: string; size: number; steps: number; seed: number | null }): Promise<GeneratedImage>;
  /** Opens File Explorer at a generated image. */
  revealImage(path: string): Promise<boolean>;
  /** Reads a reply aloud with the speech model in use; resolves when it has finished or stopped. */
  readAloud(text: string): Promise<void>;
  /** Stops speech that is playing or being read. */
  stopSpeaking(): Promise<void>;

  // ---- Tune screen. One tune at a time: it uses the GPU fully. ----

  /** Starts measuring `modelId` (a text or vision library model) for `profile`. Rejects while
   *  another tune is running; a finished, failed or cancelled run is replaced. */
  startTune(modelId: string, profile: TuneProfile): Promise<TuneRunInfo>;
  /** Delivers the current run's events from its first (`replay` true, one batch), then every new
   *  one as it happens, until unsubscribed. A new run begins with a `started` event. */
  subscribeTune(listener: (events: TuneEvent[], replay: boolean) => void): Unsubscribe;
  /** Stops the running tune. Every finished step is kept. */
  cancelTune(runId: string): Promise<void>;
  /** Continues a cancelled run: finished steps are kept and the interrupted one starts over. */
  resumeTune(runId: string): Promise<void>;

  // ---- Monitor. The app polls getMetrics once a second; history and request bodies on demand. ----

  /** The engine's live snapshot: state, hardware, model, the last 60 s, totals, recent requests. */
  getMetrics(): Promise<MetricsSnapshot>;
  /** Longer history, averaged into the range's step (1 min: 1 s, 15 min: 5 s, 1 h: 15 s, 24 h:
   *  5 min). Rejects an unknown range. */
  getMetricsHistory(range: HistoryRange): Promise<MetricsSeries>;
  /** One kept request with its captured bodies (each cut at 256K characters). Rejects a malformed
   *  or unknown id. */
  getRequest(id: string): Promise<RequestDetail>;

  // ---- Engine switches (Turbo). Values are saved on this machine by the app. ----
  listSwitches(): Promise<EngineSwitch[]>;
  /** Resolves to the stored value (clamped to the switch's range). */
  setSwitch(id: string, value: number): Promise<number>;
  resetSwitches(): Promise<EngineSwitch[]>;

  // ---- Tune for this machine (Turbo). A loaded model is required; the model is busy meanwhile. ----

  /** Times the processor settings that depend on this machine on the loaded model, calling
   *  `onTrial` after each finished setting. Aborting the signal stops tuning, through the same
   *  stop Chat uses. Rejects with a plain sentence when no model is loaded or tuning cannot run. */
  tuneMachine(onTrial: (t: MachineTuneTrial) => void, signal: AbortSignal): Promise<MachineTuneOutcome>;
  /** The last finished tune kept on this machine, or null before the first one. */
  lastMachineTune(): Promise<MachineTuneOutcome | null>;

  // ---- MCP servers. The config is saved as mcp.json; saving restarts servers to match. ----
  mcpConfig(): Promise<McpConfig>;
  mcpSaveConfig(config: McpConfig): Promise<void>;
  mcpStatus(): Promise<McpServerStatus[]>;
  mcpRestart(name: string): Promise<void>;
  mcpTools(): Promise<McpTool[]>;
  mcpCall(server: string, tool: string, args: unknown): Promise<McpCallResult>;

  // ---- Built-in tools: always offered to the model (server "cerebro"), limited to the folders the
  // user added on the Tools screen. Empty folder list means every built-in tool refuses. ----
  builtinTools(): Promise<McpTool[]>;
  builtinFoldersGet(): Promise<string[]>;
  /** Validates, saves and returns the new list; rejects when a path does not exist or is not a folder. */
  builtinFoldersSet(folders: string[]): Promise<string[]>;

  // ---- Train: fine-tune with LoRA in a managed Python environment, export GGUF to the library. ----
  trainerStatus(): Promise<TrainerStatus>;
  /** Builds the training environment (several GB, once). Streams "setup" events. */
  trainerSetup(onEvent: (e: TrainEvent) => void): Promise<TrainInstalled>;
  /** Runs one job to the end; rejects with the reason when it fails. */
  trainerStart(job: TrainJob, onEvent: (e: TrainEvent) => void): Promise<void>;
  /** Stops the running job at its next step; what it learned so far is still saved. */
  trainerCancel(): Promise<boolean>;

  // ---- Phone access: off by default; LAN only, QR pairing plus password. ----
  // ---- Cerebro as an MCP server: 127.0.0.1 only, bearer token. ----
  // ---- Local API: OpenAI + Anthropic formats on one port; off by default, every protection a switch. ----
  localApiGet(): Promise<LocalApiView>;
  localApiEnable(on: boolean): Promise<LocalApiView>;
  localApiOptions(requireKey: boolean, allowLan: boolean, allowModelSwitch: boolean): Promise<LocalApiView>;
  localApiRotateKey(): Promise<LocalApiView>;
  mcpServerGet(): Promise<McpServerView>;
  mcpServerEnable(on: boolean): Promise<McpServerView>;
  mcpServerRotateToken(): Promise<McpServerView>;

  remoteGet(): Promise<RemoteView>;
  remoteSetPassword(password: string): Promise<RemoteView>;
  remoteEnable(on: boolean): Promise<RemoteView>;
  remoteOptions(options: RemoteOptions): Promise<RemoteView>;
  remotePair(): Promise<RemotePairing>;
  /** "*" signs out every phone. */
  remoteRevoke(id: string): Promise<RemoteView>;

  // ---- Chat. One reply streams at a time per conversation. ----

  /** Streams a reply: `start`, optional `thinking`, `token`s, optional `check` passes, then `done`
   *  (or `error`). Aborting the signal stops generation; `done` still arrives with `stopped: true`
   *  and the text so far is kept. Rejects a malformed request before streaming. */
  chat(request: ChatRequest, onEvent: (e: ChatEvent) => void, signal: AbortSignal): Promise<void>;
  /** Answers a tool call that waits for approval in Chat. False when nothing was waiting. */
  chatToolDecide(id: string, allow: boolean): Promise<boolean>;
  /** The user's answer to a question (ask_user) or plan (exit_plan_mode); false if nothing waited. */
  chatToolAnswer(id: string, answer: Record<string, unknown>): Promise<boolean>;
  /** Transcribes a push-to-talk recording on this machine. */
  transcribe(seconds: number): Promise<string>;

  // ---- Model Hub (desktop: live Hugging Face). Absent in the web preview. ----

  /** Repositories with GGUF files matching the query, most downloaded first. */
  hubSearch?(query: string): Promise<HubRepo[]>;
  /** A repository's GGUF files with their real sizes. */
  hubFiles?(repo: string): Promise<HubFile[]>;
  /** The architecture numbers and real size of one file's GGUF header (metadata only). */
  hubFileMeta?(repo: string, file: string): Promise<HubFileMeta>;
  /** Downloads one file into Cerebro's model folder; resolves with its path when complete. */
  hubDownload?(repo: string, file: HubFile, onEvent: (e: HubDownloadEvent) => void): Promise<string>;
  /** Stops a download; the partial file resumes next time. */
  hubCancel?(repo: string, path: string): Promise<void>;
  /** Where a model would go if loaded now with this context and GPU layer count (desktop). */
  placementPreview?(id: string, contextLength: number, gpuLayers: number): Promise<Placement>;
}
