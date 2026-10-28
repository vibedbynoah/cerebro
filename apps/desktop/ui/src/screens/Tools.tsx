import "../styles/tools.css";
import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { ApprovalDialog } from "../components/ApprovalDialog";
import { Icon } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { hrefFor } from "../lib/routes";
import {
  ACCESS,
  ACCESS_LABELS,
  ACCESS_NOTES,
  applyDecision,
  applyPreset,
  PRESETS,
  presetOf,
  type Access,
  DECISION_LABELS,
  loadPolicy,
  mcpToolKey,
  resolve,
  RISK_LABELS,
  RULE_LABELS,
  RULES,
  SAMPLE_CALLS,
  savePolicy,
  summarize,
  TOOLS,
  type Decision,
  type LogEntry,
  type Policy,
  type Rule,
  type ToolCall,
} from "../lib/permissions";
import { notify } from "../lib/toast";
import { useService } from "../service/context";
import type { McpTool } from "../service/types";
import { useAppState } from "../state";
import { toolActivity } from "../lib/tool-activity";
import { McpServers } from "./tools/McpServers";

/**
 * Tools: every action a model may take, each behind a rule the user sets. The approval dialog
 * previews exactly what a call will do. Rules can be edited or removed, and prompts can be
 * switched off entirely in Settings.
 */
export function Tools(): JSX.Element {
  const app = useAppState();
  const service = useService();
  const promptsOn = () => app.settings()?.permissionPrompts ?? true;
  const offline = () => app.settings()?.offlineMode ?? false;

  // The built-in tools are fixed at startup: the desktop app lists them beside the MCP servers'
  // tools. The web preview has no engine, so it shows the example tools instead.
  const [builtins, setBuiltins] = createSignal<McpTool[]>([]);
  if (__DESKTOP__) {
    service
      .builtinTools()
      .then(setBuiltins)
      .catch(() => {});
  }

  // The tools the MCP servers offer, refreshed each second; replaced only when the list changes, so
  // an open rule menu is never rebuilt under the pointer.
  const [mcpTools, setMcpTools] = createSignal<McpTool[]>([]);
  let mcpSig = "";
  let mcpInFlight = false;
  const loadMcpTools = () => {
    if (mcpInFlight || document.visibilityState === "hidden") return;
    mcpInFlight = true;
    service
      .mcpTools()
      .then((list) => {
        const sig = list.map((t) => `${t.server}/${t.name}/${t.description ?? ""}`).join("|");
        if (sig !== mcpSig) {
          mcpSig = sig;
          setMcpTools(list);
        }
      })
      .catch(() => {})
      .finally(() => {
        mcpInFlight = false;
      });
  };
  loadMcpTools();
  const mcpPoll = setInterval(loadMcpTools, 1000);
  onCleanup(() => clearInterval(mcpPoll));

  /** One row per tool: the built-in tools (desktop), the MCP servers' tools, and, in the browser
   *  preview only, the built-in and plugin examples. */
  const rows = () => [
    ...(__DESKTOP__
      ? builtins().map((t) => {
          const id = mcpToolKey(t.server, t.name);
          const def = TOOLS.find((d) => d.id === id);
          return { id, name: def?.name ?? t.name, chip: "Built-in", risk: def?.risk ?? null, description: t.description ?? def?.description ?? "" };
        })
      : []),
    ...mcpTools().map((t) => ({ id: mcpToolKey(t.server, t.name), name: t.name, chip: t.server, risk: null, description: t.description ?? "" })),
    ...(__DESKTOP__ ? [] : TOOLS.map((t) => ({ id: t.id, name: t.name, chip: t.source === "plugin" ? (t.plugin ?? "Plugin") : "Built-in", risk: t.risk, description: t.description }))),
  ];
  const [policy, setPolicyS] = createSignal<Policy>(loadPolicy());
  const [grants, setGrants] = createSignal<Set<string>>(new Set());
  const [pending, setPending] = createSignal<ToolCall | null>(null);
  const [log, setLog] = createSignal<LogEntry[]>([]);
  const [q, setQ] = createSignal("");

  const setPolicy = (p: Policy) => {
    setPolicyS(p);
    savePolicy(p);
  };
  const setRule = (toolId: string, r: Rule | "default") => {
    const rules = { ...policy().rules };
    if (r === "default") delete rules[toolId];
    else rules[toolId] = r;
    setPolicy({ ...policy(), rules });
  };
  const record = (call: ToolCall, outcome: LogEntry["outcome"]) => setLog((l) => [{ at: Date.now(), toolId: call.toolId, summary: summarize(call.preview), outcome }, ...l].slice(0, 500));

  const simulate = (call: ToolCall) => {
    const v = resolve(policy(), call.toolId, grants(), promptsOn());
    if (v === "ask") setPending(call);
    else {
      record(call, v === "allow" ? "auto-allowed" : "auto-denied");
      notify(v === "allow" ? "Ran without asking" : "Denied by a rule", summarize(call.preview), v === "allow" ? "info" : "warning");
    }
  };
  const decide = (d: Decision) => {
    const call = pending();
    if (!call) return;
    const next = applyDecision(policy(), grants(), call.toolId, d);
    setPolicy(next.policy);
    setGrants(next.grants);
    record(call, d);
    setPending(null);
  };

  const toolName = (id: string) => TOOLS.find((t) => t.id === id)?.name ?? id;
  // Real calls from Chat this session, with the preview's sample calls, newest first.
  const allLog = () => [...toolActivity(), ...log()].sort((a, b) => b.at - a.at);
  const shown = () => {
    const s = q().toLowerCase().trim();
    return s ? allLog().filter((e) => `${toolName(e.toolId)} ${e.summary} ${e.outcome}`.toLowerCase().includes(s)) : allLog();
  };

  return (
    <div class="screen tools">
      <ScreenHeader route="tools" />
      <Show when={!promptsOn()}>
        <p class="notice" role="status">
          <Icon name="info" size={18} />
          <span>
            Permission prompts are off: tools run without asking, except those set to Deny. <a class="text-link" href={hrefFor("settings")}>Settings</a>
          </span>
        </p>
      </Show>
      <Show when={offline()}>
        <p class="notice" role="status">
          <Icon name="offline" size={18} />
          <span>
            Offline mode is on: Fetch web page refuses until it is turned off. <a class="text-link" href={hrefFor("settings")}>Settings</a>
          </span>
        </p>
      </Show>

      <Section id="tool-presets" title="How much the model may do" description="One choice sets the access level and the rules for Cerebro's own tools together. Change any single rule below and the choice shows as Custom.">
        <fieldset class="segmented tools-presets">
          <legend class="sr-only">Preset</legend>
          <For each={PRESETS}>
            {(x) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="tool-preset" value={x.id} checked={presetOf(policy()) === x.id} onChange={() => setPolicy(applyPreset(policy(), x.id))} />
                <span>{x.label}</span>
              </label>
            )}
          </For>
        </fieldset>
        <p class="hint" data-testid="preset-note">
          {PRESETS.find((x) => x.id === presetOf(policy()))?.note ?? "Custom: a mix of rules."}
        </p>
        <label class="tools-access">
          <span>Access</span>
          <span class="select">
            <select class="input" aria-label="Tool access" value={policy().access} onChange={(e) => setPolicy({ ...policy(), access: e.currentTarget.value as Access })}>
              <For each={ACCESS}>{(a) => <option value={a}>{ACCESS_LABELS[a]}</option>}</For>
            </select>
          </span>
        </label>
        <p class="hint" data-testid="access-note">
          {ACCESS_NOTES[policy().access]}
        </p>
      </Section>

      <McpServers />
      <Section id="default-rule" title="Default for tools without a rule">
        <fieldset class="segmented">
          <legend class="sr-only">Default rule</legend>
          <For each={RULES}>
            {(r) => (
              <label class="segmented__option" data-press>
                <input type="radio" name="default-rule" value={r} checked={policy().defaultRule === r} onChange={() => setPolicy({ ...policy(), defaultRule: r })} />
                <span>{RULE_LABELS[r]}</span>
              </label>
            )}
          </For>
        </fieldset>
      </Section>

      <Section id="tool-rules" title="Tools" description={__DESKTOP__ ? "Cerebro's built-in tools and every tool the MCP servers offer. Default follows the setting above." : "Built-in tools and tools from installed plugins. Change any rule; Default follows the setting above."}>
        <table class="tools-table">
          <thead>
            <tr>
              <th scope="col">Tool</th>
              <th scope="col">What it does</th>
              <th scope="col">Rule</th>
              <th scope="col">
                <span class="sr-only">Try it</span>
              </th>
            </tr>
          </thead>
          <tbody>
            <For each={rows()} fallback={<tr><td colspan="4" class="hint">No tools yet. Add an MCP server above and its tools appear here.</td></tr>}>
              {(t) => {
                // Sample calls are a browser-preview demo; the desktop app logs only real calls.
                const call = __DESKTOP__ ? undefined : SAMPLE_CALLS.find((c) => c.toolId === t.id);
                return (
                  <tr data-tool={t.id}>
                    <th scope="row">
                      <span class="tools-table__name">{t.name}</span>
                      <span class="chip" data-tone="neutral">
                        {t.chip}
                      </span>
                    </th>
                    <td>
                      <Show when={t.risk}>
                        {(r) => (
                          <span class="tools-table__risk" data-risk={r()}>
                            {RISK_LABELS[r()]}
                          </span>
                        )}
                      </Show>
                      <span class="hint">{t.description}</span>
                    </td>
                    <td>
                      <select aria-label={`Rule for ${t.name}`} value={policy().rules[t.id] ?? "default"} onChange={(e) => setRule(t.id, e.currentTarget.value as Rule | "default")}>
                        <option value="default">Default ({RULE_LABELS[policy().defaultRule]})</option>
                        <For each={RULES}>{(r) => <option value={r}>{RULE_LABELS[r]}</option>}</For>
                      </select>
                    </td>
                    <td>
                      <Show when={call}>
                        {(c) => (
                          <button class="btn btn--tertiary btn--sm" type="button" onClick={() => simulate(c())} aria-label={`Try a sample ${t.name} call`}>
                            Try
                          </button>
                        )}
                      </Show>
                    </td>
                  </tr>
                );
              }}
            </For>
          </tbody>
        </table>
        <Show when={!__DESKTOP__}>
          <p class="hint">Sample calls are illustrative: nothing actually runs.</p>
        </Show>
        <span class="sr-only" aria-live="polite">
          {log()[0] ? `${toolName(log()[0]!.toolId)}: ${log()[0]!.outcome}` : ""}
        </span>
      </Section>

      <Section id="tool-log" title="Activity" description="Every tool call and what happened to it, newest first.">
        <input class="tools-search" type="search" placeholder="Search calls" aria-label="Search tool activity" value={q()} onInput={(e) => setQ(e.currentTarget.value)} />
        <ol class="tools-log" role="list">
          <For each={shown()} fallback={<li class="hint">{allLog().length ? "Nothing matches." : __DESKTOP__ ? "No tool calls yet. They appear here as models use tools in Chat." : "No tool calls yet. Try one above."}</li>}>
            {(e) => (
              <li data-outcome={e.outcome}>
                <span class="mono tools-log__at">{new Date(e.at).toLocaleTimeString()}</span>
                <span class="tools-log__tool">{toolName(e.toolId)}</span>
                <span class="mono tools-log__sum">{e.summary}</span>
                <span class="chip" data-tone={e.outcome === "deny" || e.outcome === "auto-denied" || e.outcome === "failed" ? "off" : "on"}>
                  {e.outcome === "auto-allowed" ? "Ran (rule)" : e.outcome === "auto-denied" ? "Blocked (rule)" : e.outcome === "ran" ? "Ran" : e.outcome === "failed" ? "Failed" : DECISION_LABELS[e.outcome]}
                </span>
              </li>
            )}
          </For>
        </ol>
      </Section>

      <ApprovalDialog call={pending()} onDecide={decide} />
    </div>
  );
}
