// @ts-check
/**
 * Isolation hook: runs in Tauri's sandboxed isolation iframe and sees every IPC message the app
 * sends before it is encrypted for the Rust side. Anything not on the allowlist is dropped here,
 * so even script injected into the main webview cannot reach a command this app never uses.
 *
 * The allowlist must match src-tauri/build.rs, src-tauri/capabilities/main-window.json and the
 * `invoke` calls in ui/src/platform/tauri.ts (tests/unit/tauri-config.test.ts checks all four).
 * None of these commands takes arguments, so any payload with a key in it is refused as well.
 */

/** @type {ReadonlySet<string>} */
const ALLOWED = new Set([
  "window_minimize",
  "window_toggle_maximize",
  "window_is_maximized",
  "window_close",
  "app_info",
  // Tauri's own drag-region script (data-tauri-drag-region on the title bar).
  "plugin:window|start_dragging",
  "plugin:window|internal_toggle_maximize",
]);

/**
 * @param {unknown} payload
 * @returns {boolean} true for "no arguments": undefined, or a plain object with no own keys.
 */
function isEmptyPayload(payload) {
  if (payload === undefined) return true;
  if (typeof payload !== "object" || payload === null) return false;
  const proto = Object.getPrototypeOf(payload);
  if (proto !== Object.prototype && proto !== null) return false;
  return Reflect.ownKeys(payload).length === 0;
}

/**
 * @param {{ cmd?: unknown; payload?: unknown }} message
 * @returns {boolean}
 */
function isAllowedMessage(message) {
  return typeof message === "object" && message !== null && typeof message.cmd === "string" && ALLOWED.has(message.cmd) && isEmptyPayload(message.payload);
}

// A classic script (Tauri inlines it), so no `export`: tests run this file in a VM context and call
// the hook through the fake window they pass in.
/** @param {{ cmd?: unknown; payload?: unknown }} message */
const hook = (message) => {
  if (isAllowedMessage(message)) return message;
  // Throwing stops Tauri's handler before it encrypts or sends anything.
  throw new Error("IPC message refused by the isolation allowlist");
};
Object.defineProperty(window, "__TAURI_ISOLATION_HOOK__", { value: Object.freeze(hook), writable: false, configurable: false });
