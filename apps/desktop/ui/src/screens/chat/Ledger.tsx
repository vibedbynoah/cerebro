import { createMemo, createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { LEDGER_FILES, type LedgerFileName } from "../../service/chat-types";
import type { LedgerState } from "../../lib/chat/store";

/**
 * Orchestrate mode's ledger: the roles run one after another (GVS5H), so the panel shows the role
 * running now and the files the team has written so far. Plain disclosure and tabs, no markup from
 * the model: every file is shown as text (tasks.json as a checklist).
 */

const TAB_LABELS: Readonly<Record<LedgerFileName, string>> = { "plan.md": "Plan", "notes.md": "Notes", "tasks.json": "Tasks", "result.md": "Result" };

/** "worker 2" is the model's role name; the chip says "Worker 2, round 1". */
function roleLabel(role: string): string {
  const w = /^worker ([1-6])$/.exec(role);
  return w ? `Worker ${w[1]}` : role.charAt(0).toUpperCase() + role.slice(1);
}

interface TaskRow {
  id: number;
  desc: string;
  done: boolean;
}

/** The manager curates tasks.json; each row is { id, desc, status }. Malformed JSON shows as text. */
function parseTasks(text: string): TaskRow[] | null {
  try {
    const v: unknown = JSON.parse(text);
    if (!Array.isArray(v)) return null;
    return v.flatMap((t): TaskRow[] => {
      if (!t || typeof t !== "object") return [];
      const x = t as Record<string, unknown>;
      return typeof x.desc === "string" && Number.isInteger(x.id) ? [{ id: x.id as number, desc: x.desc, done: x.status === "done" }] : [];
    });
  } catch {
    return null;
  }
}

export function LedgerPanel(props: { id: string; ledger: LedgerState; streaming: boolean; ref?: (el: HTMLDetailsElement) => void }): JSX.Element {
  const [tab, setTab] = createSignal<LedgerFileName | null>(null);
  const [copied, setCopied] = createSignal(false);
  const l = () => props.ledger;
  // Two replies in one thread each have their own tabs and panel, so the ids carry the message id.
  const uid = () => `l-${props.id.replace(/[^A-Za-z0-9]/g, "")}`;
  const written = createMemo(() => LEDGER_FILES.filter((f) => l().files[f] !== undefined));
  // No tab chosen: follow the file the team is writing now, otherwise the newest written one.
  const active = createMemo<LedgerFileName | null>(() => {
    const chosen = tab();
    return (chosen && l().files[chosen] !== undefined ? chosen : null) ?? [...written()].reverse().find((f) => (l().files[f] ?? "").trim() !== "") ?? written().at(-1) ?? null;
  });
  const chip = createMemo(() => {
    const s = l().step;
    return s ? `${roleLabel(s.role)}, round ${s.round}` : null;
  });
  const tasks = createMemo(() => (active() === "tasks.json" ? parseTasks(l().files["tasks.json"] ?? "") : null));

  const copyDir = async () => {
    try {
      await navigator.clipboard.writeText(l().dir ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the path stays selectable */
    }
  };

  return (
    <details class="ledger" ref={(el) => props.ref?.(el)}>
      <summary class="ledger__summary">
        <Icon name="chip" size={14} />
        <span>Ledger</span>
        <span class="mono ledger__count">
          {written().length}/{LEDGER_FILES.length} files
        </span>
        <Show when={props.streaming && chip()}>
          {(c) => (
            <span class="chip" data-tone="active" data-testid="ledger-step" aria-live="polite">
              {c()}
            </span>
          )}
        </Show>
      </summary>
      <div class="ledger__body">
        <Show when={l().dir}>
          {(dir) => (
            <p class="ledger__dir">
              <span class="mono" data-testid="ledger-dir">
                {dir()}
              </span>
              <button class="icon-btn icon-btn--sm" type="button" onClick={() => void copyDir()} aria-label={copied() ? "Copied" : "Copy ledger folder path"}>
                <Icon name={copied() ? "check" : "copy"} size={14} />
              </button>
            </p>
          )}
        </Show>
        <div class="ledger__tabs" role="tablist" aria-label="Ledger files">
          <For each={written()}>
            {(f) => (
              <button class="ledger__tab" type="button" role="tab" id={`${uid()}-tab-${f}`} aria-selected={active() === f} aria-controls={`${uid()}-file`} onClick={() => setTab(f)}>
                {TAB_LABELS[f]}
                <span class="mono ledger__tab-size">{(l().files[f] ?? "").length}</span>
              </button>
            )}
          </For>
        </div>
        <div id={`${uid()}-file`} role="tabpanel" aria-labelledby={active() ? `${uid()}-tab-${active()}` : undefined} tabindex="0">
          <Show when={active()} fallback={<p class="hint">The team has not written a file yet.</p>}>
            {(f) => (
              <Show when={tasks()} fallback={<pre class="ledger__text">{l().files[f()] ?? ""}</pre>}>
                {(rows) => (
                  <ul class="ledger__tasks" role="list" data-testid="ledger-tasks">
                    <For each={rows()} fallback={<li class="hint">No tasks yet.</li>}>
                      {(t) => (
                        <li class="ledger__task" data-done={t.done}>
                          <input type="checkbox" checked={t.done} disabled aria-label={t.done ? "Done" : "To do"} />
                          <span>{t.desc}</span>
                        </li>
                      )}
                    </For>
                  </ul>
                )}
              </Show>
            )}
          </Show>
        </div>
      </div>
    </details>
  );
}
