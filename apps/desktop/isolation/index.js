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
  "disk_scan",
  "chat_stop",
  "kv_store_usage",
  "kv_store_clear",
  "live_stats",
  "metrics",
  "switches_list",
  "switches_reset",
  "tune_machine_last",
  "mcp_config_get",
  "mcp_status",
  "remote_get",
  "remote_pair",
  "mcp_server_get",
  "mcp_server_rotate_token",
  "mcp_tools",
  "builtin_tools",
  "builtin_folders_get",
  "trainer_status",
  "local_api_get",
  "local_api_rotate_key",
  "trainer_cancel",
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

/** A duplicate group: the file's SHA-256 as lowercase hex (disk_scan's group id). @param {unknown} v */
const isGroupId = (v) => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);

/** A model file path. The engine matches it exactly against the copies its own scan found; the
 *  bound is Windows' maximum path length. @param {unknown} v */
const isPath = (v) => typeof v === "string" && v.length > 0 && v.length <= 32767;

/** A Tauri channel, as serialized for IPC ("__CHANNEL__:<id>") or still as its object. @param {unknown} v */
const isChannel = (v) => (typeof v === "string" && /^__CHANNEL__:\d+$/.test(v)) || (typeof v === "object" && v !== null && typeof (/** @type {{id?: unknown}} */ (v)).id === "number");
/** Absent, null, or a whole number from `min` to 2^20. @param {unknown} v @param {number} min */
const optCount = (v, min) => v === undefined || v === null || (typeof v === "number" && Number.isInteger(v) && v >= min && v <= 1 << 20);
const TRAIN_QUANTS = new Set(["q4_k_m", "q5_k_m", "q8_0", "f16"]);
/** @param {unknown} v @param {number} lo @param {number} hi */
const isNum = (v, lo, hi) => typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi;
/** @param {unknown} v @param {number} lo @param {number} hi */
const isInt = (v, lo, hi) => typeof v === "number" && Number.isSafeInteger(v) && v >= lo && v <= hi;
/** @param {unknown} v */
const isHfId = (v) => typeof v === "string" && /^[A-Za-z0-9._-]{1,96}\/[A-Za-z0-9._-]{1,96}$/.test(v);
/** A training job (trainer.rs TrainJob::check repeats every check). @param {unknown} j */
const isTrainJob = (j) =>
  isPlain(j) &&
  hasOnlyKeys(j, ["name", "model", "datasetPath", "datasetHf", "format", "method", "rank", "alpha", "lr", "epochs", "maxSteps", "maxSeqLen", "batch", "gradAccum", "export"]) &&
  typeof j.name === "string" && j.name.length > 0 && j.name.length <= 64 &&
  isHfId(j.model) &&
  ((typeof j.datasetPath === "string" && j.datasetPath.length > 0 && j.datasetPath.length <= 1024 && j.datasetHf === null) || (j.datasetPath === null && isHfId(j.datasetHf))) &&
  ["auto", "alpaca", "chatml", "sharegpt"].includes(/** @type {string} */ (j.format)) &&
  ["qlora", "lora"].includes(/** @type {string} */ (j.method)) &&
  isInt(j.rank, 1, 256) && isInt(j.alpha, 1, 512) && isNum(j.lr, 1e-8, 0.01) && isNum(j.epochs, 0.01, 50) &&
  isInt(j.maxSteps, 0, 100000) && isInt(j.maxSeqLen, 128, 32768) && isInt(j.batch, 1, 64) && isInt(j.gradAccum, 1, 256) &&
  Array.isArray(j.export) && j.export.length <= 4 && j.export.every((q) => typeof q === "string" && TRAIN_QUANTS.has(q));

/** A Hugging Face "owner/name". @param {unknown} v */
const isRepo = (v) => typeof v === "string" && /^[A-Za-z0-9._-]{1,96}\/[A-Za-z0-9._-]{1,96}$/.test(v) && !v.includes("..");

/** A .gguf path inside a repository, no traversal. @param {unknown} v */
const isGguf = (v) => typeof v === "string" && v.length <= 300 && /^[A-Za-z0-9._+-]+(\/[A-Za-z0-9._+-]+)*\.gguf$/.test(v) && !v.includes("..");

/** An MCP server name in mcp.json. @param {unknown} v */
const isServerName = (v) => typeof v === "string" && /^[A-Za-z0-9 ._-]{1,64}$/.test(v);

/** One mcp.json server: a local program (command, args, env) or a web address (url, headers). @param {unknown} s */
function isMcpServer(s) {
  if (!isPlain(s) || !hasOnlyKeys(s, [], ["command", "args", "env", "url", "headers", "disabled"])) return false;
  if (s.disabled !== undefined && typeof s.disabled !== "boolean") return false;
  if (s.command !== undefined && s.url === undefined) {
    if (typeof s.command !== "string" || s.command.length === 0 || s.command.length > 512) return false;
    if (s.args !== undefined && !(Array.isArray(s.args) && s.args.length <= 64 && s.args.every((a) => typeof a === "string" && a.length <= 1024))) return false;
    if (s.env !== undefined && !(isPlain(s.env) && Object.values(s.env).every((v) => typeof v === "string"))) return false;
    return true;
  }
  if (s.url !== undefined && s.command === undefined) {
    if (typeof s.url !== "string" || s.url.length > 2048 || !/^https?:\/\/\S+$/.test(s.url)) return false;
    if (s.headers !== undefined && !(isPlain(s.headers) && Object.entries(s.headers).every(([k, v]) => k.length > 0 && k.length <= 128 && typeof v === "string" && v.length <= 2048))) return false;
    return true;
  }
  return false;
}

/** mcp.json: { mcpServers: { name: { command, args[], env{} } | { url, headers{} } (+ disabled) } }. @param {unknown} c */
function isMcpConfig(c) {
  if (!isPlain(c) || !hasOnlyKeys(c, ["mcpServers"]) || !isPlain(c.mcpServers)) return false;
  const servers = Object.entries(c.mcpServers);
  return servers.length <= 64 && servers.every(([name, s]) => isServerName(name) && isMcpServer(s));
}

/** A turn's attachment: name, type and size, plus an image's data (a base64 image data: URL of at
 *  most 3 MB, which the composer's downscaling keeps it under). @param {unknown} a */
function isAttachment(a) {
  if (!isPlain(a) || !hasOnlyKeys(a, ["name", "type", "bytes", "data"])) return false;
  const { name, type, bytes, data } = /** @type {Record<string, unknown>} */ (a);
  return (
    typeof name === "string" && name.length <= 260 &&
    typeof type === "string" && type.length <= 100 &&
    typeof bytes === "number" && Number.isFinite(bytes) && bytes >= 0 &&
    (data === undefined || (typeof data === "string" && data.length <= 3_000_000 && /^data:image\/(png|jpeg|gif|webp);base64,/.test(data)))
  );
}

/** @param {unknown} r */
function isChatRequest(r) {
  if (!isPlain(r) || !isModelId(r.modelId) || typeof r.systemPrompt !== "string" || !Array.isArray(r.turns) || !isPlain(r.sampler)) return false;
  return (
    r.turns.every(
      (t) =>
        isPlain(t) &&
        (t.role === "user" || t.role === "assistant") &&
        typeof t.text === "string" &&
        (t.attachments === undefined || (Array.isArray(t.attachments) && t.attachments.length <= 10 && t.attachments.every(isAttachment))),
    ) &&
    (r.tools === undefined || r.tools === null || isToolsRequest(r.tools)) &&
    (r.orchestrate === undefined || r.orchestrate === null || isOrchestrateRequest(r.orchestrate)) &&
    (r.keepCache === undefined || typeof r.keepCache === "boolean")
  );
}

/** Orchestrate mode: workers per round and rounds, both plain integers in range. @param {unknown} o */
function isOrchestrateRequest(o) {
  if (!isPlain(o) || !hasOnlyKeys(o, ["workers", "rounds"])) return false;
  const { workers, rounds } = /** @type {{ workers: unknown; rounds: unknown }} */ (o);
  return Number.isInteger(workers) && /** @type {number} */ (workers) >= 1 && /** @type {number} */ (workers) <= 6 && Number.isInteger(rounds) && /** @type {number} */ (rounds) >= 1 && /** @type {number} */ (rounds) <= 8;
}

const TOOL_RULES = new Set(["allow", "ask", "deny"]);
/** Chat's tool rules: { rules: { "server/tool": allow|ask|deny }, defaultRule, offline? }. @param {unknown} t */
function isToolsRequest(t) {
  if (!isPlain(t) || !hasOnlyKeys(t, ["rules", "defaultRule"], ["offline", "allowLocalFetch", "access", "showSecrets", "blindWrites", "planMode"]) || !TOOL_RULES.has(/** @type {string} */ (t.defaultRule)) || !isPlain(t.rules)) return false;
  if (t.offline !== undefined && typeof t.offline !== "boolean") return false;
  if (t.allowLocalFetch !== undefined && typeof t.allowLocalFetch !== "boolean") return false;
  if (t.access !== undefined && !["read-only", "workspace", "full"].includes(/** @type {string} */ (t.access))) return false;
  if (t.showSecrets !== undefined && typeof t.showSecrets !== "boolean") return false;
  if (t.blindWrites !== undefined && typeof t.blindWrites !== "boolean") return false;
  if (t.planMode !== undefined && typeof t.planMode !== "boolean") return false;
  const entries = Object.entries(t.rules);
  return entries.length <= 4096 && entries.every(([k, v]) => k.length <= 200 && typeof v === "string" && TOOL_RULES.has(v));
}

/** Commands with arguments, each with its own shape check. @type {Record<string, (p: unknown) => boolean>} */
const WITH_ARGS = {
  model_load: (p) => isPlain(p) && hasOnlyKeys(p, ["id"], ["context", "gpuLayers"]) && isModelId(p.id) && optCount(p.context, 1) && optCount(p.gpuLayers, 0),
  placement_preview: (p) => isPlain(p) && hasOnlyKeys(p, ["id"], ["context", "gpuLayers"]) && isModelId(p.id) && optCount(p.context, 1) && optCount(p.gpuLayers, 0),
  disk_deduplicate: (p) => isPlain(p) && hasOnlyKeys(p, ["groupId", "keepPath"]) && isGroupId(p.groupId) && isPath(p.keepPath),
  disk_unlink: (p) => isPlain(p) && hasOnlyKeys(p, ["groupId"]) && isGroupId(p.groupId),
  model_reveal: (p) => isPlain(p) && hasOnlyKeys(p, ["id"]) && isModelId(p.id),
  hub_search: (p) => isPlain(p) && hasOnlyKeys(p, ["query"]) && typeof p.query === "string" && p.query.length <= 120,
  hub_files: (p) => isPlain(p) && hasOnlyKeys(p, ["repo"]) && isRepo(p.repo),
  hub_file_meta: (p) => isPlain(p) && hasOnlyKeys(p, ["repo", "file"]) && isRepo(p.repo) && isGguf(p.file),
  hub_download: (p) => isPlain(p) && hasOnlyKeys(p, ["repo", "file", "parts", "total", "onEvent"]) && isRepo(p.repo) && isGguf(p.file) && Array.isArray(p.parts) && p.parts.length > 0 && p.parts.length <= 64 && p.parts.every(isGguf) && Number.isSafeInteger(p.total) && isChannel(p.onEvent),
  hub_cancel: (p) => isPlain(p) && hasOnlyKeys(p, ["repo", "file"]) && isRepo(p.repo) && isGguf(p.file),
  switch_set: (p) => isPlain(p) && hasOnlyKeys(p, ["id", "value"]) && typeof p.id === "string" && /^[a-z0-9_]{1,48}$/.test(p.id) && Number.isSafeInteger(p.value),
  mcp_config_save: (p) => isPlain(p) && hasOnlyKeys(p, ["config"]) && isMcpConfig(p.config),
  mcp_restart: (p) => isPlain(p) && hasOnlyKeys(p, ["name"]) && isServerName(p.name),
  remote_set_password: (p) => isPlain(p) && hasOnlyKeys(p, ["password"]) && typeof p.password === "string" && p.password.length >= 8 && p.password.length <= 1024,
  local_api_enable: (p) => isPlain(p) && hasOnlyKeys(p, ["on"]) && typeof p.on === "boolean",
  local_api_options: (p) =>
    isPlain(p) && hasOnlyKeys(p, ["requireKey", "allowLan", "allowModelSwitch"]) && typeof p.requireKey === "boolean" && typeof p.allowLan === "boolean" && typeof p.allowModelSwitch === "boolean",
  trainer_setup: (p) => isPlain(p) && hasOnlyKeys(p, ["onEvent"]) && isChannel(p.onEvent),
  trainer_start: (p) => isPlain(p) && hasOnlyKeys(p, ["job", "onEvent"]) && isChannel(p.onEvent) && isTrainJob(p.job),
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
  chat_tool_answer: (p) =>
    isPlain(p) && hasOnlyKeys(p, ["id", "answer"]) && typeof p.id === "string" && /^t\d{1,16}-\d{1,3}$/.test(p.id) && isPlain(p.answer) && JSON.stringify(p.answer).length <= 65536,
  chat_tool_decide: (p) => isPlain(p) && hasOnlyKeys(p, ["id", "allow"]) && typeof p.id === "string" && /^t\d{1,16}-\d{1,3}$/.test(p.id) && typeof p.allow === "boolean",
  builtin_folders_set: (p) =>
    isPlain(p) && hasOnlyKeys(p, ["folders"]) && Array.isArray(p.folders) && p.folders.length <= 32 && p.folders.every((f) => typeof f === "string" && f.length > 0 && f.length <= 1024),
  remote_revoke: (p) => isPlain(p) && hasOnlyKeys(p, ["id"]) && typeof p.id === "string" && /^(\*|[A-Za-z0-9_-]{1,32})$/.test(p.id),
  mcp_call: (p) => isPlain(p) && hasOnlyKeys(p, ["server", "tool", "arguments"]) && isServerName(p.server) && typeof p.tool === "string" && p.tool.length > 0 && p.tool.length <= 128 && (p.arguments === null || isPlain(p.arguments)),
  chat_send: (p) => isPlain(p) && hasOnlyKeys(p, ["request", "onEvent"]) && isChatRequest(p.request) && isChannel(p.onEvent),
  tune_machine_start: (p) => isPlain(p) && hasOnlyKeys(p, ["onEvent"]) && isChannel(p.onEvent),
  metrics_history: (p) => isPlain(p) && hasOnlyKeys(p, ["range"]) && ["1m", "15m", "1h", "24h"].includes(/** @type {string} */ (p.range)),
  request_detail: (p) => isPlain(p) && hasOnlyKeys(p, ["id"]) && typeof p.id === "string" && p.id.length > 0 && p.id.length <= 64,
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
