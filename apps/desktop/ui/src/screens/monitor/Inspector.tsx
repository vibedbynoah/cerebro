import { createEffect, createMemo, createResource, createSignal, createUniqueId, For, onCleanup, Show, type JSX } from "solid-js";
import { createStore, reconcile } from "solid-js/store";
import { Icon } from "../../components/icons";
import { Term } from "../../components/Term";
import { copyText } from "../../lib/download";
import {
  clockTime,
  filterRequests,
  isRequestFilter,
  isRunning,
  num,
  QUERY_MAX,
  REQUEST_FILTERS,
  seconds,
  STATE_LABELS,
  type RequestFilter,
} from "../../lib/monitor";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { RequestDetail, RequestSummary } from "../../service/monitor-types";

/**
 * Every API request the engine keeps, live: a searchable, filterable list and the selected one's
 * timings, token counts and captured bodies. Bodies come from any local client and are untrusted:
 * they are only ever rendered as text (Solid escapes it), never as HTML, and each is capped at
 * 256K characters by the engine. Rows are reconciled by id, so a new sample updates a row in place
 * and keyboard focus stays where it was.
 */

const PANES = ["input", "output", "reasoning", "response"] as const;
type Pane = (typeof PANES)[number];
const PANE_LABELS: Readonly<Record<Pane, string>> = { input: "Input", output: "Output", reasoning: "Reasoning", response: "Response" };

/** Pretty-prints JSON output for reading; anything else is shown as it came. */
function readable(text: string): string {
  const t = text.trim();
  if (!(t.startsWith("{") || t.startsWith("["))) return text;
  try {
    return JSON.stringify(JSON.parse(t), null, 2);
  } catch {
    return text;
  }
}

export function Inspector(props: { requests: readonly RequestSummary[]; selected: string | null; onSelect: (id: string | null) => void; at: number; illustrative: boolean }): JSX.Element {
  const service = useService();
  const id = createUniqueId();
  const [query, setQuery] = createSignal("");
  const [filter, setFilter] = createSignal<RequestFilter>("all");
  const [pane, setPane] = createSignal<Pane>("output");
  const [copied, setCopied] = createSignal(false);
  let copiedTimer: ReturnType<typeof setTimeout> | undefined;
  onCleanup(() => clearTimeout(copiedTimer));

  const [rows, setRows] = createStore<RequestSummary[]>([]);
  createEffect(() => setRows(reconcile(filterRequests(props.requests, query(), filter()), { key: "id" })));

  const selected = createMemo(() => props.requests.find((r) => r.id === props.selected) ?? null);
  // A running request is fetched again every sample, so its output grows on screen; a finished one
  // once per state change.
  const fetchKey = createMemo(() => {
    const s = selected();
    if (!s) return null;
    return isRunning(s.state) ? `${s.id}|${props.at}` : `${s.id}|${s.state}`;
  });
  // A request can leave the engine's list between two polls; a failed fetch resolves to null (never
  // an errored resource, whose read would throw into the screen's error boundary).
  const [detail] = createResource(fetchKey, (key) => service.getRequest(key.split("|")[0] ?? "").catch((): RequestDetail | null => null));
  const current = (): RequestDetail | null => {
    const d = detail.latest;
    return d && d.id === props.selected ? d : null;
  };

  const content = () => {
    const d = current();
    if (!d) return "";
    const p = pane();
    const raw = d[p];
    if (raw) return p === "output" || p === "response" ? readable(raw) : raw;
    if (p === "reasoning") return "No separate reasoning for this request.";
    if (p === "response" && d.stream && isRunning(d.state)) return "Still streaming. The Output tab shows the reply as it arrives.";
    if (p === "output" && d.error) return "No output: the request ended before the model wrote anything.";
    return isRunning(d.state) ? "Nothing yet." : "No content.";
  };
  const note = () => {
    const d = current();
    const p = pane();
    const base =
      p === "input"
        ? "The request body the client sent."
        : p === "output"
          ? "What the model wrote. JSON is formatted here for reading."
          : p === "reasoning"
            ? "The model's separate reasoning, when it produces one."
            : d?.stream
              ? "The last events of the stream, as the client received them."
              : "The response body the client received.";
    return `${base}${d?.truncated[p] ? " This capture stops at 256K characters; the client received all of it." : ""}`;
  };

  const copy = async () => {
    const ok = await copyText(content());
    if (!ok) {
      notify("Could not copy", "Select the text and copy it by hand.", "error");
      return;
    }
    setCopied(true);
    clearTimeout(copiedTimer);
    copiedTimer = setTimeout(() => setCopied(false), 1500);
  };

  let list: HTMLUListElement | undefined;
  const onListKey = (e: KeyboardEvent) => {
    if (!list || !["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    const buttons = Array.from(list.querySelectorAll<HTMLButtonElement>("button.mon-req"));
    const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (at === -1) return;
    e.preventDefault();
    const last = buttons.length - 1;
    const next = e.key === "Home" ? 0 : e.key === "End" ? last : e.key === "ArrowDown" ? Math.min(last, at + 1) : Math.max(0, at - 1);
    buttons[next]?.focus();
  };

  const tabEls: HTMLButtonElement[] = [];
  const onTabKey = (e: KeyboardEvent, i: number) => {
    const n = PANES.length;
    const next = e.key === "ArrowRight" ? (i + 1) % n : e.key === "ArrowLeft" ? (i - 1 + n) % n : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    setPane(PANES[next] as Pane);
    tabEls[next]?.focus();
  };

  return (
    <div class="mon-inspector">
      <div class="mon-inspector__list card">
        <div class="mon-inspector__tools">
          <label class="sr-only" for={`${id}-q`}>
            Search requests
          </label>
          <span class="mon-search">
            <Icon name="search" size={14} />
            <input
              id={`${id}-q`}
              class="input mon-search__input"
              type="search"
              placeholder="Search id, path, session, tool"
              maxLength={QUERY_MAX}
              autocomplete="off"
              spellcheck={false}
              value={query()}
              onInput={(e) => setQuery(e.currentTarget.value.slice(0, QUERY_MAX))}
            />
          </span>
          <label class="sr-only" for={`${id}-f`}>
            Show
          </label>
          <span class="select">
            <select id={`${id}-f`} class="input" value={filter()} onChange={(e) => isRequestFilter(e.currentTarget.value) && setFilter(e.currentTarget.value)}>
              <For each={REQUEST_FILTERS}>{(f) => <option value={f.id}>{f.label}</option>}</For>
            </select>
          </span>
        </div>
        <p class="mon-inspector__count">
          {num(rows.length)} of {num(props.requests.length)} kept requests
        </p>
        <Show
          when={rows.length > 0}
          fallback={<p class="mon-empty">{props.requests.length ? "No request matches." : "No requests yet. Calls to the API appear here as they arrive."}</p>}
        >
          <ul class="mon-reqs" ref={list} onKeyDown={onListKey} aria-label="Requests, newest first">
            <For each={rows}>
              {(r) => (
                <li>
                  <button type="button" class="mon-req" aria-pressed={props.selected === r.id} data-request={r.id} onClick={() => props.onSelect(r.id)}>
                    <span class="mon-req__row">
                      <span class="mono mon-req__time">{clockTime(r.startedAt)}</span>
                      <span class="mon-tag" data-state={r.state}>
                        {STATE_LABELS[r.state]}
                      </span>
                    </span>
                    <span class="mon-req__path mono">
                      {r.method} {r.path}
                    </span>
                    <span class="mon-req__meta">
                      {r.id} · {r.wallS === null ? "running" : seconds(r.wallS)} · {r.stream ? "stream" : "JSON"}
                      {r.httpStatus !== null ? ` · HTTP ${r.httpStatus}` : ""}
                    </span>
                  </button>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </div>

      <div class="mon-inspector__detail card">
        <Show
          when={selected()}
          fallback={
            <div class="mon-inspector__none">
              <Icon name="activity" size={20} />
              <p>Choose a request to see its timings, token counts and what was sent and received.</p>
            </div>
          }
        >
          {(r) => (
            <>
              <header class="mon-detail__head">
                <h3 class="mon-detail__title mono">
                  {r().id} · {r().method} {r().path}
                </h3>
                <span class="mon-tag" data-state={r().state}>
                  {STATE_LABELS[r().state]}
                </span>
              </header>
              <p class="mon-detail__sub">
                {r().session} · started {clockTime(r().startedAt)}
                <Show when={props.illustrative}> · illustrative capture</Show>
              </p>
              <dl class="mon-stats">
                <div>
                  <dt>Wall time</dt>
                  <dd>{r().wallS === null ? "running" : seconds(r().wallS)}</dd>
                </div>
                <div>
                  <dt>Load</dt>
                  <dd>{seconds(r().loadS ?? 0)}</dd>
                </div>
                <div>
                  <dt>Queue</dt>
                  <dd>{seconds(r().queueS)}</dd>
                </div>
                <div>
                  <dt>
                    <Term k="firstToken">First token</Term>
                  </dt>
                  <dd>{seconds(r().firstTokenS)}</dd>
                </div>
                <div>
                  <dt>Prompt</dt>
                  <dd>
                    {num(r().promptTokens)} <span class="muted">({num(r().reusedTokens)} reused)</span>
                  </dd>
                </div>
                <div>
                  <dt>Written</dt>
                  <dd>
                    {num(r().outputTokens)} <span class="muted">({num(r().reasoningTokens)} reasoning)</span>
                  </dd>
                </div>
                <div>
                  <dt>Writing</dt>
                  <dd>{r().decodeTps === null ? "–" : `${num(r().decodeTps, 1)} tok/s`}</dd>
                </div>
                <div>
                  <dt>Reading</dt>
                  <dd>{r().prefillTps === null ? "–" : `${num(r().prefillTps)} tok/s`}</dd>
                </div>
                <div>
                  <dt>Response</dt>
                  <dd>
                    {r().stream ? "Stream" : "JSON"} · {r().responseFormat}
                  </dd>
                </div>
                <div>
                  <dt>HTTP</dt>
                  <dd>{r().httpStatus ?? "–"}</dd>
                </div>
              </dl>
              <Show when={current()?.error}>
                {(err) => (
                  <p class="mon-detail__error" role="note">
                    <Icon name="alert" size={16} />
                    <span>{err()}</span>
                  </p>
                )}
              </Show>
              <div class="mon-tabs-bar">
                <div class="mon-tabs" role="tablist" aria-label="Captured bodies">
                  <For each={PANES}>
                    {(p, i) => (
                      <button
                        ref={(el) => (tabEls[i()] = el)}
                        type="button"
                        role="tab"
                        class="mon-tab"
                        id={`${id}-tab-${p}`}
                        aria-selected={pane() === p}
                        aria-controls={`${id}-panel`}
                        tabindex={pane() === p ? 0 : -1}
                        onClick={() => setPane(p)}
                        onKeyDown={(e) => onTabKey(e, i())}
                      >
                        {PANE_LABELS[p]}
                      </button>
                    )}
                  </For>
                </div>
                <button type="button" class="btn btn--tertiary mon-copy" data-press onClick={() => void copy()} disabled={!current()}>
                  <Icon name={copied() ? "check" : "copy"} size={14} />
                  {copied() ? "Copied" : "Copy"}
                </button>
              </div>
              <div class="mon-pane" role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-tab-${pane()}`}>
                <Show when={current()} fallback={<p class="mon-empty">{!detail.loading && detail.latest === null ? "This request is no longer kept." : "Loading the capture…"}</p>}>
                  <pre class="mon-pre" tabindex="0" aria-label={`${PANE_LABELS[pane()]} of ${r().id}`}>
                    {content()}
                  </pre>
                </Show>
                <p class="mon-pane__note">{note()}</p>
              </div>
            </>
          )}
        </Show>
      </div>
    </div>
  );
}
