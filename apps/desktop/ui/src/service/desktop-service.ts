/**
 * The desktop app's CerebroService: every number and model comes from Cerebro Engine running in the
 * app process (src-tauri/src/engine.rs), through IPC commands. Nothing here is invented: a feature
 * the engine does not have yet rejects with a plain sentence, so the screen shows that honestly
 * instead of example data.
 *
 * `local` supplies what is only the user's own state kept on this machine (settings, presets,
 * per-file tweaks); it never supplies models, hardware or measurements.
 */
import { Channel, invoke } from "@tauri-apps/api/core";
import type { CerebroService, Unsubscribe } from "./service";
import type { ChatEvent, ChatRequest } from "./chat-types";
import type { HistoryRange, MetricsSeries, MetricsSnapshot, RequestDetail } from "./monitor-types";
import type { CatalogModel, GpuInfo, HardwareProfile, LibraryModel, LiveStats, LoadedModel, LoadParams, ModelDetail, ModelSummary, NetworkState } from "./types";
import { CURRENT_CATALOG } from "./catalog";
import type { TrainerStatus, TrainEvent, TrainInstalled } from "./train-types";
import type { ActivePatch, EngineSwitch, McpServerView, RemotePairing, RemoteView, McpCallResult, McpConfig, McpServerStatus, McpTool, HubDownloadEvent, HubFile, HubFileMeta, HubRepo, LoadPreset, PatchProfile } from "./types";
import { EMPTY_TWEAKS, profileNameError, sanitizeTweaks } from "../lib/tweaks";

const NOT_YET = (what: string) => Promise.reject(new Error(`${what} is not available in this build yet.`));

export function createDesktopService(local: CerebroService): CerebroService {
  let library: LibraryModel[] = [];
  // The user's own presets and per-file tweaks, kept for this session.
  const presets = new Map<string, LoadPreset[]>();
  const patches = new Map<string, ActivePatch>();
  const profiles = new Map<string, PatchProfile[]>();
  let nextId = 1;
  const known = (id: string) => {
    if (!library.some((m) => m.id === id)) throw new TypeError("That model is not in the library.");
  };
  const knownFile = (sha: string) => {
    if (!library.some((m) => m.sha256 === sha)) throw new TypeError("Unknown model file");
  };
  let hardware: Promise<HardwareProfile> | null = null;

  const hw = () => (hardware ??= invoke<HardwareProfile>("hw_detect").catch((e) => {
    hardware = null;
    throw e;
  }));

  const summary = (m: LibraryModel): ModelSummary => ({
    id: m.id,
    name: m.name,
    quant: m.quant,
    params: m.params,
    architecture: m.architecture,
    sizeGiB: m.sizeGiB,
    contextMax: m.contextMax,
    tune: m.tune,
    defaultsTps: null,
    tunedTps: null,
    provenance: "measured",
  });

  const service: CerebroService = {
    getLiveStats: () => invoke<LiveStats>("live_stats"),
    subscribeLiveStats(listener): Unsubscribe {
      // 3 Hz like the metrics feed: VRAM and state are instant, tok/s changes once a second in
      // the backend. A slow answer is never overlapped and a hidden window skips polls.
      let alive = true;
      let inFlight = false;
      const tick = () => {
        if (inFlight || document.visibilityState === "hidden") return;
        inFlight = true;
        invoke<LiveStats>("live_stats")
          .then((s) => alive && listener(s))
          .catch(() => {})
          .finally(() => {
            inFlight = false;
          });
      };
      tick();
      const timer = setInterval(tick, 333);
      return () => {
        alive = false;
        clearInterval(timer);
      };
    },
    async listModels() {
      if (library.length === 0) library = await invoke<LibraryModel[]>("library_list");
      return library.map(summary);
    },
    async getGpu(): Promise<GpuInfo> {
      const h = await hw();
      return h.gpu ?? { name: "No NVIDIA GPU found", vramTotalGiB: 0 };
    },
    detectHardware: () => hw(),
    listCatalog: (): Promise<CatalogModel[]> => Promise.resolve(CURRENT_CATALOG.map((c) => ({ ...c }))),
    getNetwork: (): Promise<NetworkState> => Promise.resolve(navigator.onLine ? "online" : "offline"),
    getSettings: () => local.getSettings(),
    updateSetting: (key, value) => local.updateSetting(key, value),

    async listLibrary() {
      library = await invoke<LibraryModel[]>("library_list");
      return library.map((m) => ({ ...m }));
    },
    async getModelDetail(id): Promise<ModelDetail> {
      if (!library.some((m) => m.id === id)) throw new Error("That model is not in the library.");
      return { id, gguf: null, chatTemplate: null, quirks: [], author: { rope: null, sampler: null }, copy: null };
    },
    listLoaded: () => invoke<LoadedModel[]>("models_loaded"),
    loadModel: (id: string, params: LoadParams | null) => invoke<LoadedModel[]>("model_load", { id, context: params?.contextLength ?? null }),
    unloadModel: () => invoke<LoadedModel[]>("model_unload"),
    async listPresets(modelId) {
      known(modelId);
      return (presets.get(modelId) ?? []).map((p) => ({ ...p, params: { ...p.params } }));
    },
    async savePreset(modelId, name, params) {
      known(modelId);
      const list = presets.get(modelId) ?? [];
      const err = profileNameError(name, list.map((p) => p.name));
      if (err) throw new TypeError(err);
      const next = [...list, { id: `preset-${nextId++}`, modelId, name: name.trim(), params: { ...params } }];
      presets.set(modelId, next);
      return next.map((p) => ({ ...p }));
    },
    async deletePreset(modelId, presetId) {
      known(modelId);
      const next = (presets.get(modelId) ?? []).filter((p) => p.id !== presetId);
      presets.set(modelId, next);
      return next.map((p) => ({ ...p }));
    },
    async getActivePatch(sha) {
      knownFile(sha);
      const p = patches.get(sha) ?? { tweaks: EMPTY_TWEAKS, fixes: [] };
      return { tweaks: { ...p.tweaks }, fixes: [...p.fixes] };
    },
    async setActivePatch(sha, patch) {
      knownFile(sha);
      const tweaks = sanitizeTweaks(patch?.tweaks);
      if (!tweaks) throw new TypeError("Invalid tweaks");
      const next: ActivePatch = { tweaks, fixes: [] };
      patches.set(sha, next);
      return { tweaks: { ...next.tweaks }, fixes: [] };
    },
    async listPatchProfiles(sha) {
      return (profiles.get(sha) ?? []).map((p) => ({ ...p }));
    },
    async savePatchProfile(profile) {
      knownFile(profile.modelSha256);
      const list = profiles.get(profile.modelSha256) ?? [];
      const err = profileNameError(profile.name, list.map((p) => p.name));
      if (err) throw new TypeError(err);
      const tweaks = sanitizeTweaks(profile.tweaks);
      if (!tweaks) throw new TypeError("Invalid tweaks");
      const next = [...list, { ...profile, id: `profile-${nextId++}`, name: profile.name.trim(), tweaks, fixes: [] }];
      profiles.set(profile.modelSha256, next);
      return next.map((p) => ({ ...p }));
    },
    async deletePatchProfile(sha, id) {
      const next = (profiles.get(sha) ?? []).filter((p) => p.id !== id);
      profiles.set(sha, next);
      return next.map((p) => ({ ...p }));
    },
    writeGgufCopy: () => NOT_YET("Editing a model's metadata"),
    revertGgufCopy: () => NOT_YET("Editing a model's metadata"),
    // Total size of the library is measured; finding duplicate copies is not built yet, so no
    // duplicate groups are reported.
    scanDisk: async () => ({ provenance: "measured", totalGiB: library.reduce((t, m) => t + m.sizeGiB, 0), groups: [] }),
    deduplicate: () => NOT_YET("Deduplication"),
    unlink: () => NOT_YET("Deduplication"),
    revealInFolder: () => Promise.resolve(false),
    describeImage: () => NOT_YET("Image input"),
    transcribeSample: () => NOT_YET("Speech to text"),
    speakSample: () => NOT_YET("Text to speech"),

    startTune: () => NOT_YET("Tune"),
    subscribeTune: () => () => {},
    cancelTune: () => Promise.resolve(),
    resumeTune: () => NOT_YET("Tune"),

    getMetrics: () => invoke<MetricsSnapshot>("metrics"),
    async getMetricsHistory(range: HistoryRange): Promise<MetricsSeries> {
      if (range !== "1m") return NOT_YET("History longer than a minute");
      return (await invoke<MetricsSnapshot>("metrics")).history;
    },
    getRequest: (): Promise<RequestDetail> => NOT_YET("Request bodies"),

    listSwitches: () => invoke<EngineSwitch[]>("switches_list"),
    setSwitch: (id, value) => invoke<number>("switch_set", { id, value }),
    resetSwitches: () => invoke<EngineSwitch[]>("switches_reset"),

    mcpConfig: () => invoke<McpConfig>("mcp_config_get"),
    mcpSaveConfig: (config) => invoke<void>("mcp_config_save", { config }),
    mcpStatus: () => invoke<McpServerStatus[]>("mcp_status"),
    mcpRestart: (name) => invoke<void>("mcp_restart", { name }),
    mcpTools: () => invoke<McpTool[]>("mcp_tools"),
    mcpCall: (server, tool, args) => invoke<McpCallResult>("mcp_call", { server, tool, arguments: args }),

    // Built-in tools: specs in the same shape as mcp_tools; the folder list is saved by the app.
    builtinTools: () => invoke<McpTool[]>("builtin_tools"),
    builtinFoldersGet: () => invoke<string[]>("builtin_folders_get"),
    builtinFoldersSet: (folders) => invoke<string[]>("builtin_folders_set", { folders }),

    trainerStatus: () => invoke<TrainerStatus>("trainer_status"),
    async trainerSetup(onEvent) {
      const channel = new Channel<TrainEvent>();
      channel.onmessage = onEvent;
      return invoke<TrainInstalled>("trainer_setup", { onEvent: channel });
    },
    async trainerStart(job, onEvent) {
      const channel = new Channel<TrainEvent>();
      channel.onmessage = onEvent;
      await invoke<void>("trainer_start", { job, onEvent: channel });
      library = [];
    },
    trainerCancel: () => invoke<boolean>("trainer_cancel"),
    mcpServerGet: () => invoke<McpServerView>("mcp_server_get"),
    mcpServerEnable: (on) => invoke<McpServerView>("mcp_server_enable", { on }),
    mcpServerRotateToken: () => invoke<McpServerView>("mcp_server_rotate_token"),
    remoteGet: () => invoke<RemoteView>("remote_get"),
    remoteSetPassword: (password) => invoke<RemoteView>("remote_set_password", { password }),
    remoteEnable: (on) => invoke<RemoteView>("remote_enable", { on }),
    remoteOptions: (o) => invoke<RemoteView>("remote_options", { allowChat: o.allowChat, allowTailscale: o.allowTailscale, address: o.address, sessionDays: o.sessionDays }),
    remotePair: () => invoke<RemotePairing>("remote_pair"),
    remoteRevoke: (id) => invoke<RemoteView>("remote_revoke", { id }),

    async chat(request: ChatRequest, onEvent: (e: ChatEvent) => void, signal: AbortSignal) {
      const channel = new Channel<ChatEvent>();
      channel.onmessage = onEvent;
      const stop = () => {
        invoke("chat_stop").catch(() => {});
      };
      signal.addEventListener("abort", stop, { once: true });
      try {
        await invoke("chat_send", { request, onEvent: channel });
      } finally {
        signal.removeEventListener("abort", stop);
      }
    },
    chatToolDecide: (id, allow) => invoke<boolean>("chat_tool_decide", { id, allow }),
    transcribe: () => NOT_YET("Speech to text"),

    hubSearch: (query) => invoke<HubRepo[]>("hub_search", { query }),
    hubFiles: (repo) => invoke<HubFile[]>("hub_files", { repo }),
    hubFileMeta: (repo, file) => invoke<HubFileMeta>("hub_file_meta", { repo, file }),
    async hubDownload(repo, file, onEvent) {
      const channel = new Channel<HubDownloadEvent>();
      channel.onmessage = onEvent;
      const path = await invoke<string>("hub_download", { repo, file: file.path, parts: file.parts, total: file.size, onEvent: channel });
      library = [];
      return path;
    },
    hubCancel: (repo, path) => invoke<void>("hub_cancel", { repo, file: path }),
  };
  return service;
}
