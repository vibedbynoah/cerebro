import type { GpuInfo, LiveStats, ModelSummary, NetworkState, SettingKey, Settings } from "./types";

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
  getNetwork(): Promise<NetworkState>;
  getSettings(): Promise<Settings>;
  updateSetting<K extends SettingKey>(key: K, value: Settings[K]): Promise<Settings>;
}
