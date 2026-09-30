import type { CerebroService, Unsubscribe } from "../service/service";
import type { GpuInfo, LiveStats, ModelSummary, NetworkState, SettingKey, Settings } from "../service/types";
import { DEFAULT_SETTINGS, MOCK_GPU, MOCK_LIVE_BASE, MOCK_MODELS } from "./fixtures";

export interface MockServiceOptions {
  /** HUD sample interval, ms. Default 1000. */
  tickMs?: number;
  /** 0..1 random source; inject a seeded one for deterministic tests. Default Math.random. */
  random?: () => number;
  /** Starting settings (merged over the safe defaults). */
  settings?: Partial<Settings>;
  network?: NetworkState;
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

/**
 * In-memory CerebroService for web mode and tests. All data is illustrative (see fixtures.ts).
 * The HUD ticker only runs while someone is subscribed, so an idle page schedules no timers.
 */
export function createMockService(options: MockServiceOptions = {}): CerebroService {
  const tickMs = options.tickMs !== undefined && Number.isFinite(options.tickMs) && options.tickMs > 0 ? options.tickMs : 1000;
  const random = options.random ?? Math.random;
  const network: NetworkState = options.network ?? "offline";
  let settings: Settings = { ...DEFAULT_SETTINGS, ...options.settings };
  let current: LiveStats = cloneStats(MOCK_LIVE_BASE);
  const listeners = new Set<(stats: LiveStats) => void>();
  let timer: ReturnType<typeof setInterval> | null = null;

  const next = (): LiveStats => {
    const base = MOCK_LIVE_BASE;
    const jitter = (spread: number) => (random() * 2 - 1) * spread;
    current = {
      ...cloneStats(base),
      tokensPerSecond: base.tokensPerSecond === null ? null : round1(base.tokensPerSecond + jitter(0.6)),
      vramUsedGiB: round2(Math.min(base.vramTotalGiB, Math.max(0, base.vramUsedGiB + jitter(0.08)))),
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

    listModels: (): Promise<ModelSummary[]> => Promise.resolve(MOCK_MODELS.map((m) => ({ ...m }))),

    getGpu: (): Promise<GpuInfo> => Promise.resolve({ ...MOCK_GPU }),

    getNetwork: () => Promise.resolve(network),

    getSettings: () => Promise.resolve({ ...settings }),

    updateSetting(key, value) {
      // Same checks the Rust side will make: a known key and a boolean value, nothing else.
      if (!isSettingKey(key)) return Promise.reject(new TypeError(`Unknown setting: ${String(key)}`));
      if (typeof value !== "boolean") return Promise.reject(new TypeError(`Setting ${key} takes true or false`));
      settings = { ...settings, [key]: value };
      return Promise.resolve({ ...settings });
    },
  };
}
