import { readPref, writePref } from "./storage";

/**
 * Tool permissions. Each tool has a rule (ask every time, allow for this session, always allow,
 * deny); unset tools follow the global default. If the user switches permission prompts off in
 * Settings, every tool runs without asking: that is their choice and nothing argues with it. A
 * decision made in the approval dialog can set the rule. Everything is logged.
 */

export type Rule = "ask" | "session" | "always" | "deny";
export const RULES: readonly Rule[] = ["ask", "session", "always", "deny"];
export const RULE_LABELS: Readonly<Record<Rule, string>> = { ask: "Ask every time", session: "Allow for this session", always: "Always allow", deny: "Deny" };

export type Risk = "read" | "write" | "run" | "network";
export const RISK_LABELS: Readonly<Record<Risk, string>> = { read: "Reads files", write: "Changes files", run: "Runs commands", network: "Uses the network" };

export interface ToolDef {
  id: string;
  name: string;
  source: "built-in" | "plugin";
  plugin?: string;
  risk: Risk;
  description: string;
}

/** The server name the engine lists the built-in tools under (src-tauri/src/builtin_tools.rs SERVER);
 *  their policy keys are "cerebro/<tool>", the same "server/tool" shape as an MCP tool's. */
export const BUILTIN_SERVER = "cerebro";
const builtinKey = (tool: string): string => `${BUILTIN_SERVER}/${tool}`;

export const TOOLS: readonly ToolDef[] = [
  { id: builtinKey("read_file"), name: "Read file", source: "built-in", risk: "read", description: "Reads a file you point it at." },
  { id: builtinKey("list_directory"), name: "List folder", source: "built-in", risk: "read", description: "Lists the files in a folder." },
  { id: builtinKey("search_files"), name: "Search files", source: "built-in", risk: "read", description: "Searches file contents in a folder." },
  { id: builtinKey("edit_file"), name: "Edit file", source: "built-in", risk: "write", description: "Changes a file, shown as a diff first." },
  { id: builtinKey("write_file"), name: "Create file", source: "built-in", risk: "write", description: "Creates a new file." },
  { id: builtinKey("run_command"), name: "Run command", source: "built-in", risk: "run", description: "Runs a terminal command in a folder you choose." },
  { id: builtinKey("run_tests"), name: "Run tests", source: "built-in", risk: "run", description: "Runs your project's test command." },
  { id: builtinKey("fetch_url"), name: "Fetch web page", source: "built-in", risk: "network", description: "Downloads one web page. Off while offline mode is on." },
  { id: builtinKey("generate_image"), name: "Generate image", source: "built-in", risk: "write", description: "Makes an image with a local image model and saves it." },
  { id: "web_search", name: "Web search", source: "plugin", plugin: "Search", risk: "network", description: "Searches the web through the installed search plugin." },
  { id: "git_commit", name: "Git commit", source: "plugin", plugin: "Git", risk: "write", description: "Commits staged changes in a repository." },
  { id: "send_email", name: "Send email", source: "plugin", plugin: "Mail", risk: "network", description: "Sends an email from your configured account." },
];

export type Decision = "allow-once" | "allow-session" | "always" | "deny";
export const DECISION_LABELS: Readonly<Record<Decision, string>> = { "allow-once": "Allowed once", "allow-session": "Allowed for session", always: "Always allowed", deny: "Denied" };

/** How far Cerebro's built-in tools reach (engine.rs Access). */
export type Access = "read-only" | "workspace" | "full";
export const ACCESS: readonly Access[] = ["read-only", "workspace", "full"];
export const ACCESS_LABELS: Readonly<Record<Access, string>> = { "read-only": "Read only", workspace: "Workspace folders", full: "Full access" };
export const ACCESS_NOTES: Readonly<Record<Access, string>> = {
  "read-only": "Tools can read inside your workspace folders. Nothing can be changed and no command runs.",
  workspace: "Tools can read and change files inside your workspace folders. Commands start there, but are not confined to them by Windows.",
  full: "Tools can read and change any file your Windows account can.",
};

export interface Policy {
  defaultRule: Rule;
  rules: Record<string, Rule>;
  access: Access;
}

/** Safe defaults: reading is always allowed, the tools that change files or run commands ask, and
 *  everything else follows the default rule. */
export const DEFAULT_POLICY: Policy = {
  defaultRule: "ask",
  access: "workspace",
  rules: {
    [builtinKey("read_file")]: "always",
    [builtinKey("list_directory")]: "always",
    [builtinKey("search_files")]: "always",
    [builtinKey("run_command")]: "ask",
    [builtinKey("write_file")]: "ask",
    [builtinKey("edit_file")]: "ask",
  },
};
export const POLICY_KEY = "cerebro-tool-policy";

const isRule = (v: unknown): v is Rule => typeof v === "string" && (RULES as readonly string[]).includes(v);
const TOOL_IDS: ReadonlySet<string> = new Set(TOOLS.map((t) => t.id));

export function parsePolicy(raw: string | null): Policy {
  if (!raw) return structuredClone(DEFAULT_POLICY);
  try {
    const v = JSON.parse(raw) as Record<string, unknown>;
    const rules: Record<string, Rule> = {};
    if (v && typeof v.rules === "object" && v.rules) {
      for (const [k, r] of Object.entries(v.rules as Record<string, unknown>)) if ((TOOL_IDS.has(k) || MCP_KEY.test(k)) && isRule(r)) rules[k] = r;
    }
    const access = (ACCESS as readonly unknown[]).includes(v?.access) ? (v.access as Access) : DEFAULT_POLICY.access;
    return { defaultRule: isRule(v?.defaultRule) ? v.defaultRule : DEFAULT_POLICY.defaultRule, rules, access };
  } catch {
    return structuredClone(DEFAULT_POLICY);
  }
}

/** An MCP tool's key in the policy: "server/tool" (server names as mcp.json allows them). */
export const mcpToolKey = (server: string, tool: string): string => `${server}/${tool}`;
const MCP_KEY = /^[A-Za-z0-9 ._-]{1,64}\/[^\s/]{1,128}$/;

/** Tools allowed "for this session" (until Cerebro restarts), shared by Chat and the Tools screen. */
export const SESSION_GRANTS = new Set<string>();

/** Chat's per-request tool rules: every known tool resolved to allow / ask / deny now. */
export function chatToolRules(p: Policy, keys: readonly string[], promptsOn: boolean): { rules: Record<string, Verdict>; defaultRule: Verdict; access: Access } {
  const rules: Record<string, Verdict> = {};
  for (const k of keys) rules[k] = resolve(p, k, SESSION_GRANTS, promptsOn);
  const d = p.defaultRule === "deny" ? "deny" : !promptsOn || p.defaultRule === "always" || p.defaultRule === "session" ? "allow" : "ask";
  return { rules, defaultRule: d, access: p.access };
}

// ---- Presets: one choice that sets the access level and the built-in tools' rules together ------

export type PresetId = "read-only" | "ask" | "auto-edit" | "full-auto";
export interface Preset {
  id: PresetId;
  label: string;
  note: string;
  access: Access;
  defaultRule: Rule;
  rules: Record<string, Rule>;
}

const READS = ["read_file", "list_directory", "search_files"].map(builtinKey);
const WRITES = ["write_file", "edit_file"].map(builtinKey);
const RUN = builtinKey("run_command");
const NET = builtinKey("fetch_url");
const all = (keys: readonly string[], r: Rule): Record<string, Rule> => Object.fromEntries(keys.map((k) => [k, r]));

export const PRESETS: readonly Preset[] = [
  { id: "read-only", label: "Read only", note: "The model looks but never changes anything.", access: "read-only", defaultRule: "deny", rules: { ...all(READS, "always"), ...all(WRITES, "deny"), [RUN]: "deny", [NET]: "ask" } },
  { id: "ask", label: "Ask before changes", note: "Reading is free; every change, command and web fetch asks you first.", access: "workspace", defaultRule: "ask", rules: { ...all(READS, "always"), ...all(WRITES, "ask"), [RUN]: "ask", [NET]: "ask" } },
  { id: "auto-edit", label: "Auto-edit", note: "Files in your workspace folders change without asking; commands and web fetches still ask.", access: "workspace", defaultRule: "ask", rules: { ...all(READS, "always"), ...all(WRITES, "always"), [RUN]: "ask", [NET]: "ask" } },
  { id: "full-auto", label: "Full auto", note: "Everything runs without asking, anywhere on this computer.", access: "full", defaultRule: "always", rules: { ...all(READS, "always"), ...all(WRITES, "always"), [RUN]: "always", [NET]: "always" } },
];

/** The preset a policy matches exactly (access, default and every built-in rule), or null. */
export function presetOf(p: Policy): PresetId | null {
  const builtin = [...READS, ...WRITES, RUN, NET];
  const hit = PRESETS.find((x) => x.access === p.access && x.defaultRule === p.defaultRule && builtin.every((k) => (p.rules[k] ?? p.defaultRule) === x.rules[k]));
  return hit?.id ?? null;
}

/** A policy with the preset applied; rules for MCP tools the user set are kept. */
export function applyPreset(p: Policy, id: PresetId): Policy {
  const x = PRESETS.find((q) => q.id === id) ?? PRESETS[1]!;
  return { access: x.access, defaultRule: x.defaultRule, rules: { ...p.rules, ...x.rules } };
}

export function loadPolicy(): Policy {
  return parsePolicy(readPref(POLICY_KEY));
}

export function savePolicy(p: Policy): void {
  writePref(POLICY_KEY, JSON.stringify(p));
}

export function ruleFor(p: Policy, toolId: string): Rule {
  return p.rules[toolId] ?? p.defaultRule;
}

export type Verdict = "allow" | "deny" | "ask";

/** What happens when `toolId` is called. `sessionGrants` holds tools allowed for this session. */
export function resolve(p: Policy, toolId: string, sessionGrants: ReadonlySet<string>, promptsOn: boolean): Verdict {
  const rule = ruleFor(p, toolId);
  if (rule === "deny") return "deny";
  if (!promptsOn) return "allow";
  // "session" allows the tool until Cerebro restarts; an "Allow for session" answer in the dialog
  // grants the same for a tool whose rule is "ask".
  if (rule === "always" || rule === "session" || sessionGrants.has(toolId)) return "allow";
  return "ask";
}

/** Applies a dialog decision: the new policy and session grants. */
export function applyDecision(p: Policy, grants: ReadonlySet<string>, toolId: string, d: Decision): { policy: Policy; grants: Set<string> } {
  const next = new Set(grants);
  const policy: Policy = { defaultRule: p.defaultRule, rules: { ...p.rules }, access: p.access };
  if (d === "allow-session") next.add(toolId);
  if (d === "always") policy.rules[toolId] = "always";
  return { policy, grants: next };
}

// ---- Mock tool calls (drive the approval dialog in web mode and tests) --------------------------

export type Preview =
  | { kind: "command"; cwd: string; command: string }
  | { kind: "diff"; path: string; lines: Array<{ t: "+" | "-" | " "; v: string }> }
  | { kind: "url"; method: "GET"; url: string }
  | { kind: "plugin"; plugin: string; action: string; details: string };

export interface ToolCall {
  id: string;
  toolId: string;
  preview: Preview;
}

export const SAMPLE_CALLS: readonly ToolCall[] = [
  { id: "call-cmd", toolId: builtinKey("run_command"), preview: { kind: "command", cwd: "C:\\Users\\you\\code\\app", command: "cargo test --workspace" } },
  {
    id: "call-diff",
    toolId: builtinKey("edit_file"),
    preview: {
      kind: "diff",
      path: "src/config.rs",
      lines: [
        { t: " ", v: "pub fn load(path: &Path) -> Result<Config> {" },
        { t: "-", v: "    let text = fs::read_to_string(path).unwrap();" },
        { t: "+", v: "    let text = fs::read_to_string(path)?;" },
        { t: " ", v: "    toml::from_str(&text).map_err(Error::Parse)" },
      ],
    },
  },
  { id: "call-url", toolId: builtinKey("fetch_url"), preview: { kind: "url", method: "GET", url: "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct" } },
  { id: "call-plugin", toolId: "git_commit", preview: { kind: "plugin", plugin: "Git", action: "Commit", details: 'Commit 2 staged files with message "Fix config error handling"' } },
];

export interface LogEntry {
  at: number;
  toolId: string;
  summary: string;
  /** "ran" and "failed" are real calls from Chat; the rest come from the approval dialog. */
  outcome: Decision | "auto-allowed" | "auto-denied" | "ran" | "failed";
}

export function summarize(p: Preview): string {
  switch (p.kind) {
    case "command":
      return p.command;
    case "diff":
      return `Edit ${p.path}`;
    case "url":
      return `${p.method} ${p.url}`;
    case "plugin":
      return `${p.plugin}: ${p.details}`;
  }
}
