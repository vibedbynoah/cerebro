import "../styles/tools.css";
import { createSignal, For, Show, type JSX } from "solid-js";
import { ApprovalDialog } from "../components/ApprovalDialog";
import { Icon } from "../components/icons";
import { ScreenHeader, Section } from "../components/Screen";
import { hrefFor } from "../lib/routes";
import {
  applyDecision,
  DECISION_LABELS,
  loadPolicy,
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
import { useAppState } from "../state";

/**
 * Tools: every action a model may take, each behind a rule the user sets. The approval dialog
 * previews exactly what a call will do. Rules can be edited or removed, and prompts can be
 * switched off entirely in Settings.
 */
export function Tools(): JSX.Element {
  const app = useAppState();
  const promptsOn = () => app.settings()?.permissionPrompts ?? true;
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
      notify(v === "allow" ? "Ran without asking" : "Denied by your rule", summarize(call.preview), v === "allow" ? "info" : "warning");
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
  const shown = () => {
    const s = q().toLowerCase().trim();
    return s ? log().filter((e) => `${toolName(e.toolId)} ${e.summary} ${e.outcome}`.toLowerCase().includes(s)) : log();
  };

  return (
    <div class="screen tools">
      <ScreenHeader route="tools" />
      <Show when={!promptsOn()}>
        <p class="notice" role="status">
          <Icon name="info" size={18} />
          <span>
            Permission prompts are off: tools run without asking, except those you set to Deny. <a class="text-link" href={hrefFor("settings")}>Settings</a>
          </span>
        </p>
      </Show>

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

      <Section id="tool-rules" title="Tools" description="Built-in tools and tools from installed plugins. Change any rule; Default follows the setting above.">
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
            <For each={TOOLS}>
              {(t) => {
                const call = SAMPLE_CALLS.find((c) => c.toolId === t.id);
                return (
                  <tr data-tool={t.id}>
                    <th scope="row">
                      <span class="tools-table__name">{t.name}</span>
                      <span class="chip" data-tone="neutral">
                        {t.source === "plugin" ? t.plugin : "Built-in"}
                      </span>
                    </th>
                    <td>
                      <span class="tools-table__risk" data-risk={t.risk}>
                        {RISK_LABELS[t.risk]}
                      </span>
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
        <p class="hint">Sample calls are illustrative: nothing actually runs.</p>
        <span class="sr-only" aria-live="polite">
          {log()[0] ? `${toolName(log()[0]!.toolId)}: ${log()[0]!.outcome}` : ""}
        </span>
      </Section>

      <Section id="tool-log" title="Activity" description="Every tool call and what happened to it, newest first.">
        <input class="tools-search" type="search" placeholder="Search calls" aria-label="Search tool activity" value={q()} onInput={(e) => setQ(e.currentTarget.value)} />
        <ol class="tools-log" role="list">
          <For each={shown()} fallback={<li class="hint">{log().length ? "Nothing matches." : "No tool calls yet. Try one above."}</li>}>
            {(e) => (
              <li data-outcome={e.outcome}>
                <span class="mono tools-log__at">{new Date(e.at).toLocaleTimeString()}</span>
                <span class="tools-log__tool">{toolName(e.toolId)}</span>
                <span class="mono tools-log__sum">{e.summary}</span>
                <span class="chip" data-tone={e.outcome === "deny" || e.outcome === "auto-denied" ? "off" : "on"}>
                  {e.outcome === "auto-allowed" ? "Ran (rule)" : e.outcome === "auto-denied" ? "Blocked (rule)" : DECISION_LABELS[e.outcome]}
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
