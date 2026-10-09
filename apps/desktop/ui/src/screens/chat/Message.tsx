import { createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { parseMarkdown, plainText } from "../../lib/chat/markdown";
import { EFFORT_LABELS, REASONING_LABELS } from "../../lib/chat/prompt";
import type { ChatMessage } from "../../lib/chat/store";
import { Markdown } from "./Markdown";

const ms = (n: number) => (n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(1)} s`);

export function MessageView(props: {
  m: ChatMessage;
  last: boolean;
  onRegenerate: () => void;
  onMore: () => void;
}): JSX.Element {
  return (
    <Show when={props.m.role === "assistant"} fallback={<UserMessage m={props.m} />}>
      <AssistantMessage {...props} />
    </Show>
  );
}

function UserMessage(props: { m: ChatMessage }): JSX.Element {
  return (
    <article class="msg msg--user" aria-label="You said">
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
      <p class="msg__text">{props.m.text}</p>
    </article>
  );
}

function AssistantMessage(props: { m: ChatMessage; last: boolean; onRegenerate: () => void; onMore: () => void }): JSX.Element {
  const m = () => props.m;
  const [speaking, setSpeaking] = createSignal(false);
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
  // Read-aloud is simulated in this build (no audio leaves a sound; the engine's TTS will play it).
  const readAloud = () => {
    if (speaking()) {
      clearTimeout(timer);
      setSpeaking(false);
      return;
    }
    setSpeaking(true);
    const words = plainText(parseMarkdown(m().text)).split(/\s+/).length;
    timer = setTimeout(() => setSpeaking(false), Math.min(15000, words * 380));
  };

  return (
    <article class="msg msg--bot" aria-label="Reply" aria-busy={m().streaming} data-streaming={m().streaming}>
      <Show when={m().thinking}>
        <details class="msg__fold">
          <summary>
            <Icon name="chip" size={14} /> Thinking <span class="mono">{m().stats?.thinkingTokens ?? "…"} tokens</span>
          </summary>
          <p class="msg__thinking">{m().thinking}</p>
        </details>
      </Show>
      <Show when={m().text} fallback={<Show when={m().streaming}><p class="msg__typing" aria-label="Writing">●●●</p></Show>}>
        <Markdown text={m().text} />
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
      <Show when={m().error}>
        <p class="notice" role="alert">
          <Icon name="alert" size={16} /> <span>{m().error}</span>
        </p>
      </Show>
      <Show when={!m().streaming && m().stats}>
        {(s) => (
          <footer class="msg__foot">
            <span class="msg__meta" data-testid="reply-meta">
              {EFFORT_LABELS[m().effort]} effort · {REASONING_LABELS[m().source]} · <span class="mono">{s().tokPerSec.toFixed(1)} tok/s</span> · first token{" "}
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
              <Show when={props.last}>
                <button class="icon-btn" type="button" onClick={() => props.onRegenerate()} aria-label="Regenerate">
                  <Icon name="refresh" size={16} />
                </button>
              </Show>
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
