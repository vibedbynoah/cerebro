import type { CerebroService, Unsubscribe } from "../service/service";
import type { CatalogModel, GpuInfo, HardwareProfile, LiveStats, ModelSummary, NetworkState, SettingKey, Settings } from "../service/types";
import { DEFAULT_SETTINGS, MOCK_CATALOG, MOCK_GPU, MOCK_HARDWARE, MOCK_LIVE_BASE, MOCK_MODELS } from "./fixtures";

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
}

/** Every failure a web/test build can ask the mock for, by `?mock=` value. */
const MOCK_FAILURES: Readonly<Record<string, MockServiceOptions>> = {
  "fail-hw": { failHardware: Number.POSITIVE_INFINITY },
  "fail-hw-once": { failHardware: 1 },
  "bad-hw": { malformedHardware: true },
  "fail-catalog": { failCatalog: true },
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

/**
 * In-memory CerebroService for web mode and tests. All data is illustrative (see fixtures.ts).
 * The HUD ticker only runs while someone is subscribed, so an idle page schedules no timers.
 */
export function createMockService(options: MockServiceOptions = {}): CerebroService {
  const tickMs = options.tickMs !== undefined && Number.isFinite(options.tickMs) && options.tickMs > 0 ? options.tickMs : 1000;
  const random = options.random ?? Math.random;
  const network: NetworkState = options.network ?? "offline";
  let settings: Settings = { ...DEFAULT_SETTINGS, ...options.settings };
  let hardwareFailuresLeft = options.failHardware !== undefined && options.failHardware > 0 ? options.failHardware : 0;
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
      options.failCatalog ? Promise.reject(new Error("Mock: catalog unavailable")) : Promise.resolve(MOCK_CATALOG.map((m) => ({ ...m }))),

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
