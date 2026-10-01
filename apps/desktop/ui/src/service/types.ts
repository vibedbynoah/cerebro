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
  /** llama.cpp build the backend was verified against, e.g. "b6500". */
  build: string;
}

export interface GpuInfo {
  name: string;
  vramTotalGiB: number;
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
  /** Serve the API on 127.0.0.1 only. Off lets other devices on the network reach it. */
  loopbackOnly: boolean;
  /** Block every network feature (model downloads, plugin installs). */
  offlineMode: boolean;
  allowModelDownloads: boolean;
  allowPluginInstalls: boolean;
}

export type SettingKey = keyof Settings;

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
  /** The graphics card llama.cpp can offload to, or null when there is none it can use. */
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
}
