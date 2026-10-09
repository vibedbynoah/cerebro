import type {
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
  ModelDetail,
  ModelSummary,
  NetworkState,
  PatchProfile,
  SettingKey,
  Settings,
  TestOutput,
  TuneEvent,
  TuneProfile,
  TuneRunInfo,
} from "./types";

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
  describeImage(id: string, image: { name: string; type: string; bytes: number }): Promise<TestOutput>;
  transcribeSample(id: string, seconds: number): Promise<TestOutput>;
  /** Resolves with the length of the spoken audio, in seconds. */
  speakSample(id: string, text: string): Promise<number>;

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
}
