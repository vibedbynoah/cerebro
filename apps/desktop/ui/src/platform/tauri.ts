/**
 * Tauri implementation of Platform. Bundled into the desktop build only (see platform.ts).
 *
 * Every command here is registered in src-tauri/src/commands.rs, granted to the main window in
 * src-tauri/capabilities/main-window.json and allowlisted in the isolation hook (isolation/index.js).
 * None takes an argument from the webview: each acts on the window that called it.
 */
import { invoke } from "@tauri-apps/api/core";
import type { AppInfo } from "../service/types";
import type { Platform } from "./platform";

function isAppInfo(value: unknown): value is AppInfo {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v["name"] === "string" && typeof v["version"] === "string" && v["mode"] === "desktop";
}

export function createTauriPlatform(): Platform {
  return {
    mode: "desktop",
    hasWindowControls: true,
    minimize: () => invoke<void>("window_minimize"),
    toggleMaximize: () => invoke<void>("window_toggle_maximize"),
    async isMaximized() {
      const maximized: unknown = await invoke("window_is_maximized");
      if (typeof maximized !== "boolean") throw new TypeError("window_is_maximized returned an unexpected shape");
      return maximized;
    },
    close: () => invoke<void>("window_close"),
    async appInfo() {
      const info: unknown = await invoke("app_info");
      if (!isAppInfo(info)) throw new TypeError("app_info returned an unexpected shape");
      return info;
    },
  };
}
