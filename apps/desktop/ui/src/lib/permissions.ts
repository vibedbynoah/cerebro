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

export const TOOLS: readonly ToolDef[] = [
  { id: "read_file", name: "Read file", source: "built-in", risk: "read", description: "Reads a file you point it at." },
  { id: "list_dir", name: "List folder", source: "built-in", risk: "read", description: "Lists the files in a folder." },
  { id: "search_files", name: "Search files", source: "built-in", risk: "read", description: "Searches file contents in a folder." },
  { id: "edit_file", name: "Edit file", source: "built-in", risk: "write", description: "Changes a file, shown as a diff first." },
  { id: "write_file", name: "Create file", source: "built-in", risk: "write", description: "Creates a new file." },
  { id: "run_command", name: "Run command", source: "built-in", risk: "run", description: "Runs a terminal command in a folder you choose." },
  { id: "run_tests", name: "Run tests", source: "built-in", risk: "run", description: "Runs your project's test command." },
  { id: "fetch_url", name: "Fetch web page", source: "built-in", risk: "network", description: "Downloads one web page. Off while offline mode is on." },
  { id: "generate_image", name: "Generate image", source: "built-in", risk: "write", description: "Makes an image with a local image model and saves it." },
  { id: "web_search", name: "Web search", source: "plugin", plugin: "Search", risk: "network", description: "Searches the web through the installed search plugin." },
  { id: "git_commit", name: "Git commit", source: "plugin", plugin: "Git", risk: "write", description: "Commits staged changes in a repository." },
  { id: "send_email", name: "Send email", source: "plugin", plugin: "Mail", risk: "network", description: "Sends an email from your configured account." },
];

export type Decision = "allow-once" | "allow-session" | "always" | "deny";
export const DECISION_LABELS: Readonly<Record<Decision, string>> = { "allow-once": "Allowed once", "allow-session": "Allowed for session", always: "Always allowed", deny: "Denied" };

export interface Policy {
  defaultRule: Rule;
  rules: Record<string, Rule>;
}

export const DEFAULT_POLICY: Policy = { defaultRule: "ask", rules: { read_file: "always", list_dir: "always", search_files: "always" } };
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
    return { defaultRule: isRule(v?.defaultRule) ? v.defaultRule : DEFAULT_POLICY.defaultRule, rules };
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
export function chatToolRules(p: Policy, keys: readonly string[], promptsOn: boolean): { rules: Record<string, Verdict>; defaultRule: Verdict } {
  const rules: Record<string, Verdict> = {};
  for (const k of keys) rules[k] = resolve(p, k, SESSION_GRANTS, promptsOn);
  const d = p.defaultRule === "deny" ? "deny" : !promptsOn || p.defaultRule === "always" || p.defaultRule === "session" ? "allow" : "ask";
  return { rules, defaultRule: d };
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
  const policy: Policy = { defaultRule: p.defaultRule, rules: { ...p.rules } };
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
  { id: "call-cmd", toolId: "run_command", preview: { kind: "command", cwd: "C:\\Users\\you\\code\\app", command: "cargo test --workspace" } },
  {
    id: "call-diff",
    toolId: "edit_file",
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
  { id: "call-url", toolId: "fetch_url", preview: { kind: "url", method: "GET", url: "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct" } },
  { id: "call-plugin", toolId: "git_commit", preview: { kind: "plugin", plugin: "Git", action: "Commit", details: 'Commit 2 staged files with message "Fix config error handling"' } },
];

export interface LogEntry {
  at: number;
  toolId: string;
  summary: string;
  outcome: Decision | "auto-allowed" | "auto-denied";
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
