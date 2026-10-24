import { createEffect, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { parseMarkdown, plainText } from "../../lib/chat/markdown";
import { EFFORT_LABELS } from "../../lib/chat/prompt";
import type { ChatMessage } from "../../lib/chat/store";
import type { Source } from "../../lib/rag";
import { LedgerPanel } from "./Ledger";
import { Markdown } from "./Markdown";
import { useService } from "../../service/context";
import { notify } from "../../lib/toast";
import { ToolCard, type ToolAnswer, type ToolDecision } from "./ToolCard";

const ms = (n: number) => (n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(1)} s`);

export function MessageView(props: {
  m: ChatMessage;
  last: boolean;
  branch?: { index: number; total: number };
  busy?: boolean;
  found?: boolean;
  onRegenerate: (id: string) => void;
  onMore: () => void;
  onEdit?: (id: string, text: string) => void;
  onBranch?: (id: string, delta: number) => void;
  onToolDecide?: (id: string, decision: ToolDecision) => void;
  onToolAnswer?: (id: string, answer: ToolAnswer) => void;
  /** Opens a passage of the user's documents in the Documents screen. */
  onOpenSource?: (source: Source) => void;
}): JSX.Element {
  return (
    <Show when={props.m.role === "assistant"} fallback={<UserMessage {...props} />}>
      <AssistantMessage {...props} />
    </Show>
  );
}

function BranchControl(props: { index: number; total: number; busy?: boolean | undefined; onBranch?: ((delta: number) => void) | undefined }): JSX.Element {
  return (
    <span class="msg__branches">
      <button class="msg__branch-btn" type="button" aria-label="Previous branch" disabled={props.busy} onClick={() => props.onBranch?.(-1)}>
        &lt;
      </button>
      <span class="mono msg__branch-count" aria-label={`Branch ${props.index} of ${props.total}`}>
        {props.index}/{props.total}
      </span>
      <button class="msg__branch-btn" type="button" aria-label="Next branch" disabled={props.busy} onClick={() => props.onBranch?.(1)}>
        &gt;
      </button>
    </span>
  );
}

function UserMessage(props: {
  m: ChatMessage;
  branch?: { index: number; total: number };
  busy?: boolean;
  found?: boolean;
  onEdit?: (id: string, text: string) => void;
  onBranch?: (id: string, delta: number) => void;
}): JSX.Element {
  const [editing, setEditing] = createSignal(false);
  const [draft, setDraft] = createSignal("");
  let area: HTMLTextAreaElement | undefined;
  const start = () => {
    setDraft(props.m.text);
    setEditing(true);
    requestAnimationFrame(() => area?.focus());
  };
  const save = () => {
    const text = draft().trim();
    if (!text || !props.onEdit) return;
    setEditing(false);
    props.onEdit(props.m.id, text);
  };
  return (
    <article class="msg msg--user" classList={{ "msg--found": props.found }} aria-label="You said" data-message-id={props.m.id} tabindex="-1">
      <Show when={props.m.attachments.length}>
        <ul class="msg__atts" role="list">
          <For each={props.m.attachments}>
            {(a) => (
              <li class="att">
                <Show when={a.preview} fallback={<Icon name="text" size={16} />}>
                  {(src) => <img src={src()} alt="" width="48" height="48" />}
                </Show>
                <span class="att__name">{a.name}</span>
              </li>
            )}
          </For>
        </ul>
      </Show>
      <Show
        when={editing()}
        fallback={
          <p class="msg__text" onDblClick={() => !props.busy && start()}>
            {props.m.text}
          </p>
        }
      >
        <form
          class="msg__edit"
          onSubmit={(e) => {
            e.preventDefault();
            save();
          }}
        >
          <label class="sr-only" for={`edit-${props.m.id}`}>
            Edit message
          </label>
          <textarea
            id={`edit-${props.m.id}`}
            ref={area}
            class="input msg__edit-input"
            rows="3"
            value={draft()}
            maxlength="100000"
            onInput={(e) => setDraft(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                setEditing(false);
              } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                save();
              }
            }}
          />
          <span class="msg__edit-actions">
            <button class="btn btn--primary btn--sm" type="submit" disabled={!draft().trim()} data-press>
              Save and branch
            </button>
            <button class="btn btn--tertiary btn--sm" type="button" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </span>
        </form>
      </Show>
      <span class="msg__meta-row">
        <Show when={props.onEdit && !editing()}>
          <button class="icon-btn icon-btn--sm" type="button" aria-label="Edit message" onClick={start} disabled={props.busy}>
            <Icon name="edit" size={14} />
          </button>
        </Show>
        <Show when={(props.branch?.total ?? 1) > 1}>
          <BranchControl index={props.branch!.index} total={props.branch!.total} busy={props.busy} onBranch={(d) => props.onBranch?.(props.m.id, d)} />
        </Show>
      </span>
    </article>
  );
}

function AssistantMessage(props: {
  m: ChatMessage;
  last: boolean;
  branch?: { index: number; total: number };
  busy?: boolean;
  found?: boolean;
  onRegenerate: (id: string) => void;
  onMore: () => void;
  onBranch?: (id: string, delta: number) => void;
  onToolDecide?: (id: string, decision: ToolDecision) => void;
  onToolAnswer?: (id: string, answer: ToolAnswer) => void;
  onOpenSource?: (source: Source) => void;
}): JSX.Element {
  const m = () => props.m;
  const [speaking, setSpeaking] = createSignal(false);
  const service = useService();
  const [copied, setCopied] = createSignal(false);
  let timer: ReturnType<typeof setTimeout> | undefined;
  onCleanup(() => clearTimeout(timer));

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(plainText(parseMarkdown(m().text)));
      setCopied(true);
      timer = setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the text stays selectable */
    }
  };
  // The ledger folds away once the reply is done, so a long thread stays calm; opening it by hand
  // keeps it open.
  let ledgerEl: HTMLDetailsElement | undefined;
  let wasStreaming = m().streaming;
  createEffect(() => {
    const now = m().streaming;
    if (wasStreaming && !now && ledgerEl) ledgerEl.open = false;
    wasStreaming = now;
  });

  // Read aloud with the speech model in use (the desktop plays it; the preview only times it).
  const readAloud = () => {
    if (speaking()) {
      void service.stopSpeaking().catch(() => {});
      setSpeaking(false);
      return;
    }
    setSpeaking(true);
    service
      .readAloud(plainText(parseMarkdown(m().text)))
      .catch((e: unknown) => notify("Read aloud failed", e instanceof Error && e.message ? e.message : "Try again.", "error"))
      .finally(() => setSpeaking(false));
  };

  return (
    <article class="msg msg--bot" classList={{ "msg--found": props.found }} aria-label="Reply" aria-busy={m().streaming} data-streaming={m().streaming} data-message-id={m().id} tabindex="-1">
      <Show when={m().ledger}>
        {(ledger) => <LedgerPanel id={m().id} ledger={ledger()} streaming={m().streaming} ref={(el) => void (ledgerEl = el)} />}
      </Show>
      <Show when={m().thinking}>
        <details class="msg__fold">
          <summary>
            <Icon name="chip" size={14} /> Thinking <span class="mono">{m().stats?.thinkingTokens ?? "…"} tokens</span>
          </summary>
          <p class="msg__thinking">{m().thinking}</p>
        </details>
      </Show>
      <Show when={m().tools.length}>
        <div class="msg__tools">
          <For each={m().tools}>{(t) => <ToolCard tool={t} onDecide={props.onToolDecide} onAnswer={props.onToolAnswer} />}</For>
        </div>
      </Show>
      <Show when={m().text} fallback={<Show when={m().streaming && !m().tools.some((t) => t.state === "approval" || t.state === "question" || t.state === "plan")}><p class="msg__typing" aria-label="Writing">●●●</p></Show>}>
        <Markdown text={m().text} />
      </Show>
      <Show when={m().sources.length}>
        <ul class="msg__sources" role="list" aria-label="Passages from your documents">
          <For each={m().sources}>
            {(s) => (
              <li>
                <button type="button" class="chip chip--source" onClick={() => props.onOpenSource?.(s)} title={s.text} data-testid="source-chip">
                  <span class="mono">{s.n}</span>
                  <span class="msg__source-name">{s.docName}</span>
                </button>
              </li>
            )}
          </For>
        </ul>
      </Show>
      <Show when={m().image}>{(img) => <ImageCard prompt={img().prompt} seed={img().seed} />}</Show>
      <Show when={m().checks.length}>
        <details class="msg__fold">
          <summary>
            <Icon name="check" size={14} /> Self-check <span class="mono">{m().checks.length} pass{m().checks.length > 1 ? "es" : ""}</span>
          </summary>
          <ol class="msg__checks">
            <For each={m().checks}>{(c) => <li>{c}</li>}</For>
          </ol>
        </details>
      </Show>
      <Show when={(props.branch?.total ?? 1) > 1}>
        <span class="msg__meta-row msg__meta-row--bot">
          <BranchControl index={props.branch!.index} total={props.branch!.total} busy={props.busy} onBranch={(d) => props.onBranch?.(m().id, d)} />
        </span>
      </Show>
      <Show when={m().error}>
        <p class="notice" role="alert">
          <Icon name="alert" size={16} /> <span>{m().error}</span>
        </p>
      </Show>
      <Show when={!m().streaming && m().stats}>
        {(s) => (
          <footer class="msg__foot">
            <span class="msg__meta" data-testid="reply-meta">
              {EFFORT_LABELS[m().effort]} effort · <span class="mono">{s().tokPerSec.toFixed(1)} tok/s</span> · first token{" "}
              <span class="mono">{ms(s().ttftMs)}</span>
              <Show when={s().thinkingTokens > 0}>
                {" "}
                · thinking <span class="mono">{s().thinkingTokens}</span> tokens, <span class="mono">+{ms(s().extraMs)}</span>
              </Show>
              <Show when={s().stopped}> · stopped</Show>
            </span>
            <span class="msg__actions">
              <button class="icon-btn" type="button" onClick={copy} aria-label={copied() ? "Copied" : "Copy reply"}>
                <Icon name={copied() ? "check" : "copy"} size={16} />
              </button>
              <button class="icon-btn" type="button" onClick={readAloud} aria-label={speaking() ? "Stop reading aloud" : "Read aloud"} aria-pressed={speaking()}>
                <Icon name={speaking() ? "stop" : "speaker"} size={16} />
              </button>
              <button class="icon-btn" type="button" onClick={() => props.onRegenerate(m().id)} aria-label="Regenerate">
                <Icon name="refresh" size={16} />
              </button>
            </span>
            <Show when={props.last && m().verbosity === "brief" && !s().stopped}>
              <button class="btn btn--tertiary btn--sm msg__more" type="button" onClick={() => props.onMore()} data-press>
                Tell me more
              </button>
            </Show>
          </footer>
        )}
      </Show>
    </article>
  );
}

/** An illustrative picture drawn from the seed, standing in for the local image model's output. */
function ImageCard(props: { prompt: string; seed: number }): JSX.Element {
  const rnd = (i: number) => {
    const x = Math.sin(props.seed * 9301 + i * 49297) * 233280;
    return x - Math.floor(x);
  };
  const hue = Math.round(rnd(1) * 360);
  const blobs = Array.from({ length: 6 }, (_, i) => ({ cx: 20 + rnd(i + 2) * 216, cy: 20 + rnd(i + 9) * 216, r: 40 + rnd(i + 17) * 90, h: (hue + i * 37) % 360 }));
  return (
    <figure class="img-card">
      <svg viewBox="0 0 256 256" role="img" aria-label={`Generated image: ${props.prompt}`}>
        <rect width="256" height="256" fill={`oklch(28% 0.06 ${hue})`} />
        <For each={blobs}>{(b) => <circle cx={b.cx} cy={b.cy} r={b.r} fill={`oklch(70% 0.14 ${b.h} / 0.55)`} />}</For>
      </svg>
      <figcaption>
        <span>{props.prompt}</span>
        <span class="mono">1024×1024 · seed {props.seed} · illustrative</span>
      </figcaption>
    </figure>
  );
}
