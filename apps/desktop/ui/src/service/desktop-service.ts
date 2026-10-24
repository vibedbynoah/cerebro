import type { McpCatalogueEntry } from "../lib/mcp-catalogue";
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
import type { BenchEvent, BenchRun } from "./bench-types";
import type { HistoryRange, MetricsSeries, MetricsSnapshot, RequestDetail } from "./monitor-types";
import type { CatalogModel, DedupeResult, DiskReport, DuplicateGroup, GpuInfo, HardwareProfile, KvStoreUsage, LibraryModel, LiveStats, LoadedModel, LoadParams, ModelDetail, ModelSummary, NetworkState, GgufCopy, GeneratedImage, CloudPreset, CloudProvider, ClusterShare, ClusterWorker, ClusterLink } from "./types";
import { CURRENT_CATALOG } from "./catalog";
import type { TrainerStatus, TrainEvent, TrainInstalled } from "./train-types";
import type { Placement } from "./types";
import type { ActivePatch, EngineSwitch, LocalApiView, McpServerView, RemotePairing, RemoteView, McpCallResult, McpConfig, McpServerStatus, McpTool, HubDownloadEvent, HubFile, HubFileMeta, HubRepo, LoadPreset, PatchProfile } from "./types";
import type { MachineTuneEvent, MachineTuneOutcome, TuneEvent, TuneProfile, TuneRunInfo } from "./types";
import { EMPTY_TWEAKS, profileNameError, sanitizeTweaks } from "../lib/tweaks";


export function createDesktopService(local: CerebroService): CerebroService {
  // The Tune screen's feed: one channel for the app's life. The engine sends the run so far, then
  // every event; a new run's log starts at its "started" event.
  let tuneLog: TuneEvent[] = [];
  const tuneListeners = new Set<{ listener: (events: TuneEvent[], replay: boolean) => void; ready: boolean }>();
  let tuneReady: Promise<void> | null = null;
  const ensureTuneFeed = (): Promise<void> => {
    if (tuneReady) return tuneReady;
    const channel = new Channel<{ events: TuneEvent[]; replay: boolean }>();
    channel.onmessage = ({ events, replay }) => {
      if (replay) {
        tuneLog = [...events];
        return;
      }
      const fresh: TuneEvent[] = [];
      for (const e of events) {
        if (e.kind === "started") tuneLog = [];
        if (tuneLog.some((x) => x.seq === e.seq)) continue;
        tuneLog.push(e);
        fresh.push(e);
      }
      if (fresh.length === 0) return;
      for (const l of tuneListeners) if (l.ready) l.listener(fresh.map((e) => ({ ...e })), false);
    };
    tuneReady = invoke<void>("tune_subscribe", { onEvent: channel }).catch(() => {});
    return tuneReady;
  };

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
    setOffline: (on) => invoke<void>("network_offline_set", { on }),
    kvStoreUsage: () => invoke<KvStoreUsage>("kv_store_usage"),
    kvStoreClear: () => invoke<KvStoreUsage>("kv_store_clear"),

    async listLibrary() {
      library = await invoke<LibraryModel[]>("library_list");
      return library.map((m) => ({ ...m }));
    },
    getModelDetail: (id) => invoke<ModelDetail>("model_detail", { id }),
    listLoaded: () => invoke<LoadedModel[]>("models_loaded"),
    loadModel: (id: string, params: LoadParams | null) => invoke<LoadedModel[]>("model_load", { id, context: params?.contextLength ?? null, gpuLayers: params?.gpuLayers ?? null }),
    placementPreview: (id, contextLength, gpuLayers) => invoke<Placement>("placement_preview", { id, context: contextLength, gpuLayers }),
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
    writeGgufCopy: (id, edits) => invoke<GgufCopy>("gguf_copy_write", { id, edits }),
    revertGgufCopy: (id) => invoke<void>("gguf_copy_revert", { id }),
    // The engine's scan groups the model files by size and SHA-256 and remembers the result, so
    // deduplicate and undo work on the copies the last scan reported.
    scanDisk: () => invoke<DiskReport>("disk_scan"),
    deduplicate: (groupId, keepPath) => invoke<DedupeResult>("disk_deduplicate", { groupId, keepPath }),
    unlink: (groupId) => invoke<DuplicateGroup>("disk_unlink", { groupId }),
    revealInFolder: (id) => invoke<boolean>("model_reveal", { id }),
    async describeImage(id, image) {
      if (!image.data) throw new Error("The image could not be read.");
      const text = await invoke<string>("model_describe_image", { id, data: image.data });
      return { text, provenance: "measured" };
    },
    async transcribeSample(id) {
      const text = await invoke<string>("stt_transcribe", { modelId: id });
      return { text, provenance: "measured" };
    },
    startRecording: () => invoke<void>("mic_start"),
    recordingLevel: () => invoke<number>("mic_level"),
    speakSample: (id, text) => invoke<number>("tts_sample", { modelId: id, text }),
    mcpRegistrySearch: (query) => invoke<McpCatalogueEntry[]>("mcp_registry_search", { query }),
    clusterShareGet: () => invoke<ClusterShare>("cluster_share_get"),
    clusterShareStart: (id) => invoke<ClusterShare>("cluster_share_start", { id }),
    clusterShareStop: () => invoke<ClusterShare>("cluster_share_stop"),
    clusterDiscover: () => invoke<ClusterWorker[]>("cluster_discover"),
    clusterLoad: (id, localShare, workers) => invoke<{ links: ClusterLink[] }>("cluster_load", { id, localShare, workers }),
    cloudPresets: () => invoke<CloudPreset[]>("cloud_presets"),
    cloudProviders: () => invoke<CloudProvider[]>("cloud_providers"),
    saveCloudProvider: (provider, apiKey) => invoke<CloudProvider[]>("cloud_provider_save", { provider, apiKey }),
    removeCloudProvider: (id) => invoke<CloudProvider[]>("cloud_provider_remove", { id }),
    listCloudModels: (provider, apiKey) => invoke<string[]>("cloud_models_list", { provider, apiKey }),
    generateImage: (req) => invoke<GeneratedImage>("image_generate", { modelId: req.modelId, prompt: req.prompt, size: req.size, steps: req.steps, seed: req.seed }),
    editImage: (req) => invoke<GeneratedImage>("image_edit", { modelId: req.modelId, prompt: req.prompt, source: req.source, strength: req.strength, size: req.size, steps: req.steps, seed: req.seed }),
    revealImage: (path) => invoke<boolean>("image_reveal", { path }),
    readAloud: (text) => invoke<void>("tts_read", { text: text.slice(0, 100_000) }),
    stopSpeaking: () => invoke<void>("tts_stop"),

    async startTune(modelId: string, profile: TuneProfile): Promise<TuneRunInfo> {
      const { integrityCheck } = await local.getSettings();
      ensureTuneFeed();
      return invoke<TuneRunInfo>("tune_start", { modelId, profile, integrityCheck });
    },
    subscribeTune(listener) {
      ensureTuneFeed();
      const entry = { listener, ready: false };
      tuneListeners.add(entry);
      // The run so far first (one batch), then live events; the two never interleave.
      void ensureTuneFeed().then(() => {
        if (!tuneListeners.has(entry)) return;
        entry.ready = true;
        listener(tuneLog.map((e) => ({ ...e })), true);
      });
      return () => {
        tuneListeners.delete(entry);
      };
    },
    cancelTune: (runId: string) => invoke<void>("tune_cancel", { runId }),
    resumeTune: (runId: string) => invoke<void>("tune_resume", { runId }),

    async benchRun(modelId, placement, onEvent, signal) {
      const channel = new Channel<BenchEvent>();
      channel.onmessage = onEvent;
      const stop = () => {
        invoke("bench_stop").catch(() => {});
      };
      signal.addEventListener("abort", stop, { once: true });
      try {
        return await invoke<BenchRun>("bench_run", { modelId, placement, onEvent: channel });
      } finally {
        signal.removeEventListener("abort", stop);
      }
    },
    benchHistory: () => invoke<BenchRun[]>("bench_history"),
    benchStop: () => invoke<void>("bench_stop"),

    getMetrics: () => invoke<MetricsSnapshot>("metrics"),
    getMetricsHistory: (range: HistoryRange) => invoke<MetricsSeries>("metrics_history", { range }),
    getRequest: (id: string) => invoke<RequestDetail>("request_detail", { id }),

    listSwitches: () => invoke<EngineSwitch[]>("switches_list"),
    setSwitch: (id, value) => invoke<number>("switch_set", { id, value }),
    resetSwitches: () => invoke<EngineSwitch[]>("switches_reset"),

    // Tuning for this machine streams its trials through a channel; stopping goes through the same
    // chat_stop the Chat screen's stop uses (the engine's cancel switch is shared).
    async tuneMachine(onTrial, signal) {
      const channel = new Channel<MachineTuneEvent>();
      channel.onmessage = (event) => {
        if (event.type !== "trial") return;
        const { type: _type, ...trial } = event;
        onTrial(trial);
      };
      const stop = () => {
        invoke("chat_stop").catch(() => {});
      };
      signal.addEventListener("abort", stop, { once: true });
      try {
        return await invoke<MachineTuneOutcome>("tune_machine_start", { onEvent: channel });
      } finally {
        signal.removeEventListener("abort", stop);
      }
    },
    lastMachineTune: () => invoke<MachineTuneOutcome | null>("tune_machine_last"),

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
    localApiGet: () => invoke<LocalApiView>("local_api_get"),
    localApiEnable: (on) => invoke<LocalApiView>("local_api_enable", { on }),
    localApiOptions: (requireKey, allowLan, allowModelSwitch) => invoke<LocalApiView>("local_api_options", { requireKey, allowLan, allowModelSwitch }),
    localApiRotateKey: () => invoke<LocalApiView>("local_api_rotate_key"),
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
    chatToolAnswer: (id, answer) => invoke<boolean>("chat_tool_answer", { id, answer }),
    transcribe: () => invoke<string>("stt_transcribe", {}),
    transcribeAudio: (audio) => invoke<string>("stt_transcribe_file", { data: audio.data, modelId: null }),

    hubSearch: (query) => invoke<HubRepo[]>("hub_search", { query }),
    hubFiles: (repo) => invoke<HubFile[]>("hub_files", { repo }),
    hubAvatar: (owner) => invoke<string>("hub_avatar", { owner }),
    saveDataset: (name, text) => invoke<string>("train_dataset_save", { name, text }),
    hubPipelineFiles: (repo) => invoke<{ files: string[]; total: number }>("hub_pipeline_files", { repo }),
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
