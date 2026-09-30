/**
 * The few things the UI needs from its host that are not app data: window controls and app info.
 * Web mode gets a browser implementation; the desktop build swaps in the Tauri one (tauri.ts), which
 * is only ever bundled when `__DESKTOP__` is true.
 */
import type { AppInfo } from "../service/types";

export interface Platform {
  readonly mode: AppInfo["mode"];
  /** False in a browser: the window controls explain themselves instead of acting. */
  readonly hasWindowControls: boolean;
  minimize(): Promise<void>;
  toggleMaximize(): Promise<void>;
  /** Whether the window is maximized now (always false in a browser), for the Maximize/Restore
   *  control. Callers re-check it after a resize. */
  isMaximized(): Promise<boolean>;
  close(): Promise<void>;
  appInfo(): Promise<AppInfo>;
}

export const APP_VERSION = "0.1.0";

export function createWebPlatform(): Platform {
  const none = () => Promise.resolve();
  return {
    mode: "web",
    hasWindowControls: false,
    minimize: none,
    toggleMaximize: none,
    isMaximized: () => Promise.resolve(false),
    close: none,
    appInfo: () => Promise.resolve({ name: "Cerebro", version: APP_VERSION, mode: "web" }),
  };
}

export async function createPlatform(): Promise<Platform> {
  if (__DESKTOP__) {
    const { createTauriPlatform } = await import("./tauri");
    return createTauriPlatform();
  }
  return createWebPlatform();
}
