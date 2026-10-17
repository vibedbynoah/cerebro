// @ts-check
/**
 * Isolation hook: runs in Tauri's sandboxed isolation iframe and sees every IPC message the app
 * sends before it is encrypted for the Rust side. Anything not on the allowlist is dropped here,
 * so even script injected into the main webview cannot reach a command this app never uses.
 *
 * The allowlist must match src-tauri/build.rs, src-tauri/capabilities/main-window.json and the
 * `invoke` calls in the UI (tests/unit/tauri-config.test.ts checks them). Commands without
 * arguments refuse any payload with a key in it; commands with arguments have a shape check each.
 */

/** Commands that take no arguments. @type {ReadonlySet<string>} */
const ALLOWED = new Set([
  "window_minimize",
  "window_toggle_maximize",
  "window_is_maximized",
  "window_close",
  "app_info",
  "hw_detect",
  "library_list",
  "model_unload",
  "models_loaded",
  "chat_stop",
  "live_stats",
  "metrics",
  "switches_list",
  "switches_reset",
  "mcp_config_get",
  "mcp_status",
  "remote_get",
  "remote_pair",
  "mcp_server_get",
  "mcp_server_rotate_token",
  "mcp_tools",
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

/** @param {unknown} v @returns {v is Record<string, unknown>} */
function isPlain(v) {
  if (typeof v !== "object" || v === null) return false;
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
}

/** @param {Record<string, unknown>} o @param {string[]} required @param {string[]} optional */
function hasOnlyKeys(o, required, optional = []) {
  const keys = Reflect.ownKeys(o);
  return required.every((k) => keys.includes(k)) && keys.every((k) => typeof k === "string" && (required.includes(k) || optional.includes(k)));
}

/** @param {unknown} v */
const isModelId = (v) => typeof v === "string" && /^[A-Za-z0-9]{1,64}$/.test(v);

/** A Tauri channel, as serialized for IPC ("__CHANNEL__:<id>") or still as its object. @param {unknown} v */
const isChannel = (v) => (typeof v === "string" && /^__CHANNEL__:\d+$/.test(v)) || (typeof v === "object" && v !== null && typeof (/** @type {{id?: unknown}} */ (v)).id === "number");

/** A Hugging Face "owner/name". @param {unknown} v */
const isRepo = (v) => typeof v === "string" && /^[A-Za-z0-9._-]{1,96}\/[A-Za-z0-9._-]{1,96}$/.test(v) && !v.includes("..");

/** A .gguf path inside a repository, no traversal. @param {unknown} v */
const isGguf = (v) => typeof v === "string" && v.length <= 300 && /^[A-Za-z0-9._+-]+(\/[A-Za-z0-9._+-]+)*\.gguf$/.test(v) && !v.includes("..");

/** An MCP server name in mcp.json. @param {unknown} v */
const isServerName = (v) => typeof v === "string" && /^[A-Za-z0-9 ._-]{1,64}$/.test(v);

/** mcp.json: { mcpServers: { name: { command, args[], env{}, disabled } } }. @param {unknown} c */
function isMcpConfig(c) {
  if (!isPlain(c) || !hasOnlyKeys(c, ["mcpServers"]) || !isPlain(c.mcpServers)) return false;
  const servers = Object.entries(c.mcpServers);
  return (
    servers.length <= 64 &&
    servers.every(([name, s]) => {
      if (!isServerName(name) || !isPlain(s) || !hasOnlyKeys(s, ["command"], ["args", "env", "disabled"])) return false;
      if (typeof s.command !== "string" || s.command.length === 0 || s.command.length > 512) return false;
      if (s.args !== undefined && !(Array.isArray(s.args) && s.args.length <= 64 && s.args.every((a) => typeof a === "string" && a.length <= 1024))) return false;
      if (s.env !== undefined && !(isPlain(s.env) && Object.values(s.env).every((v) => typeof v === "string"))) return false;
      return s.disabled === undefined || typeof s.disabled === "boolean";
    })
  );
}

/** @param {unknown} r */
function isChatRequest(r) {
  if (!isPlain(r) || !isModelId(r.modelId) || typeof r.systemPrompt !== "string" || !Array.isArray(r.turns) || !isPlain(r.sampler)) return false;
  return r.turns.every((t) => isPlain(t) && (t.role === "user" || t.role === "assistant") && typeof t.text === "string") && (r.tools === undefined || r.tools === null || isToolsRequest(r.tools));
}

const TOOL_RULES = new Set(["allow", "ask", "deny"]);
/** Chat's tool rules: { rules: { "server/tool": allow|ask|deny }, defaultRule }. @param {unknown} t */
function isToolsRequest(t) {
  if (!isPlain(t) || !hasOnlyKeys(t, ["rules", "defaultRule"]) || !TOOL_RULES.has(/** @type {string} */ (t.defaultRule)) || !isPlain(t.rules)) return false;
  const entries = Object.entries(t.rules);
  return entries.length <= 4096 && entries.every(([k, v]) => k.length <= 200 && typeof v === "string" && TOOL_RULES.has(v));
}

/** Commands with arguments, each with its own shape check. @type {Record<string, (p: unknown) => boolean>} */
const WITH_ARGS = {
  model_load: (p) => isPlain(p) && hasOnlyKeys(p, ["id"], ["context"]) && isModelId(p.id) && (p.context === undefined || p.context === null || (Number.isInteger(p.context) && /** @type {number} */ (p.context) > 0)),
  hub_search: (p) => isPlain(p) && hasOnlyKeys(p, ["query"]) && typeof p.query === "string" && p.query.length <= 120,
  hub_files: (p) => isPlain(p) && hasOnlyKeys(p, ["repo"]) && isRepo(p.repo),
  hub_file_meta: (p) => isPlain(p) && hasOnlyKeys(p, ["repo", "file"]) && isRepo(p.repo) && isGguf(p.file),
  hub_download: (p) => isPlain(p) && hasOnlyKeys(p, ["repo", "file", "parts", "total", "onEvent"]) && isRepo(p.repo) && isGguf(p.file) && Array.isArray(p.parts) && p.parts.length > 0 && p.parts.length <= 64 && p.parts.every(isGguf) && Number.isSafeInteger(p.total) && isChannel(p.onEvent),
  hub_cancel: (p) => isPlain(p) && hasOnlyKeys(p, ["repo", "file"]) && isRepo(p.repo) && isGguf(p.file),
  switch_set: (p) => isPlain(p) && hasOnlyKeys(p, ["id", "value"]) && typeof p.id === "string" && /^[a-z0-9_]{1,48}$/.test(p.id) && Number.isSafeInteger(p.value),
  mcp_config_save: (p) => isPlain(p) && hasOnlyKeys(p, ["config"]) && isMcpConfig(p.config),
  mcp_restart: (p) => isPlain(p) && hasOnlyKeys(p, ["name"]) && isServerName(p.name),
  remote_set_password: (p) => isPlain(p) && hasOnlyKeys(p, ["password"]) && typeof p.password === "string" && p.password.length >= 8 && p.password.length <= 1024,
  mcp_server_enable: (p) => isPlain(p) && hasOnlyKeys(p, ["on"]) && typeof p.on === "boolean",
  remote_enable: (p) => isPlain(p) && hasOnlyKeys(p, ["on"]) && typeof p.on === "boolean",
  remote_options: (p) =>
    isPlain(p) &&
    hasOnlyKeys(p, ["allowChat", "allowTailscale", "address", "sessionDays"]) &&
    typeof p.allowChat === "boolean" &&
    typeof p.allowTailscale === "boolean" &&
    (p.address === null || (typeof p.address === "string" && /^\d{1,3}(\.\d{1,3}){3}$/.test(p.address))) &&
    typeof p.sessionDays === "number" &&
    Number.isSafeInteger(p.sessionDays) &&
    p.sessionDays >= 0 &&
    p.sessionDays <= 3650,
  chat_tool_decide: (p) => isPlain(p) && hasOnlyKeys(p, ["id", "allow"]) && typeof p.id === "string" && /^t\d{1,16}-\d{1,3}$/.test(p.id) && typeof p.allow === "boolean",
  remote_revoke: (p) => isPlain(p) && hasOnlyKeys(p, ["id"]) && typeof p.id === "string" && /^(\*|[A-Za-z0-9_-]{1,32})$/.test(p.id),
  mcp_call: (p) => isPlain(p) && hasOnlyKeys(p, ["server", "tool", "arguments"]) && isServerName(p.server) && typeof p.tool === "string" && p.tool.length > 0 && p.tool.length <= 128 && (p.arguments === null || isPlain(p.arguments)),
  chat_send: (p) => isPlain(p) && hasOnlyKeys(p, ["request", "onEvent"]) && isChatRequest(p.request) && isChannel(p.onEvent),
};

/**
 * @param {{ cmd?: unknown; payload?: unknown }} message
 * @returns {boolean}
 */
function isAllowedMessage(message) {
  if (typeof message !== "object" || message === null || typeof message.cmd !== "string") return false;
  if (ALLOWED.has(message.cmd)) return isEmptyPayload(message.payload);
  const check = Object.prototype.hasOwnProperty.call(WITH_ARGS, message.cmd) ? WITH_ARGS[message.cmd] : undefined;
  return check !== undefined && check(message.payload);
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
