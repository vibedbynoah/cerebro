import { createMemo, createSignal, For, Index, Match, Show, Switch, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { highlight } from "../../lib/chat/highlight";
import { parseMarkdown, splitChunks, type Block, type Inline } from "../../lib/chat/markdown";

/**
 * Renders a reply from the parsed tree with a closed set of elements and text nodes only. No HTML
 * string ever reaches the DOM, so a reply cannot inject markup or script.
 */
export function Markdown(props: { text: string }): JSX.Element {
  // Chunks are keyed by position: while a reply streams only the last one changes, so only it is
  // re-parsed and re-rendered (60 fps on long replies).
  const chunks = createMemo(() => splitChunks(props.text));
  return (
    <div class="md">
      <Index each={chunks()}>{(chunk) => <Chunk src={chunk()} />}</Index>
    </div>
  );
}

function Chunk(props: { src: string }): JSX.Element {
  const blocks = createMemo(() => parseMarkdown(props.src));
  return <For each={blocks()}>{(b) => <BlockView b={b} />}</For>;
}

function BlockView(props: { b: Block }): JSX.Element {
  return (
    <Switch>
      <Match when={props.b.t === "p" && props.b}>{(b) => <p>{inlines(b().c)}</p>}</Match>
      <Match when={props.b.t === "h" && props.b}>
        {(b) => (
          <Switch>
            <Match when={b().level <= 2}>
              <h3 class="md__h">{inlines(b().c)}</h3>
            </Match>
            <Match when={b().level > 2}>
              <h4 class="md__h">{inlines(b().c)}</h4>
            </Match>
          </Switch>
        )}
      </Match>
      <Match when={props.b.t === "ul" && props.b}>
        {(b) => (
          <ul>
            <For each={b().items}>{(it) => <li>{inlines(it)}</li>}</For>
          </ul>
        )}
      </Match>
      <Match when={props.b.t === "ol" && props.b}>
        {(b) => (
          <ol start={b().start}>
            <For each={b().items}>{(it) => <li>{inlines(it)}</li>}</For>
          </ol>
        )}
      </Match>
      <Match when={props.b.t === "quote" && props.b}>
        {(b) => (
          <blockquote>
            <For each={b().c}>{(x) => <BlockView b={x} />}</For>
          </blockquote>
        )}
      </Match>
      <Match when={props.b.t === "code" && props.b}>{(b) => <CodeView lang={b().lang} code={b().v} open={b().open} />}</Match>
      <Match when={props.b.t === "hr"}>
        <hr />
      </Match>
    </Switch>
  );
}

function inlines(list: readonly Inline[]): JSX.Element {
  return <For each={list}>{(x) => <InlineView x={x} />}</For>;
}

function InlineView(props: { x: Inline }): JSX.Element {
  return (
    <Switch>
      <Match when={props.x.t === "text" && props.x}>{(x) => x().v}</Match>
      <Match when={props.x.t === "code" && props.x}>{(x) => <code class="md__code">{x().v}</code>}</Match>
      <Match when={props.x.t === "strong" && props.x}>{(x) => <strong>{inlines(x().c)}</strong>}</Match>
      <Match when={props.x.t === "em" && props.x}>{(x) => <em>{inlines(x().c)}</em>}</Match>
      <Match when={props.x.t === "del" && props.x}>{(x) => <del>{inlines(x().c)}</del>}</Match>
      <Match when={props.x.t === "link" && props.x}>
        {(x) => (
          <a href={x().href} target="_blank" rel="noopener noreferrer nofollow">
            {inlines(x().c)}
          </a>
        )}
      </Match>
    </Switch>
  );
}

export function CodeView(props: { lang: string; code: string; open?: boolean }): JSX.Element {
  const tokens = createMemo(() => highlight(props.code, props.lang));
  const [copied, setCopied] = createSignal(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(props.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  return (
    <figure class="md-code" data-lang={props.lang || "text"}>
      <figcaption class="md-code__bar">
        <span class="md-code__lang mono">{props.lang || "text"}</span>
        <button class="md-code__copy" type="button" onClick={copy} aria-label={copied() ? "Copied" : "Copy code"} disabled={props.open === true}>
          <Icon name={copied() ? "check" : "copy"} size={14} />
          <span>{copied() ? "Copied" : "Copy"}</span>
        </button>
      </figcaption>
      <pre class="md-code__pre" tabindex="0">
        <code>
          <For each={tokens()}>{(t) => (t.k === "plain" ? t.v : <span class={`tok-${t.k}`}>{t.v}</span>)}</For>
        </code>
      </pre>
      <Show when={props.open}>
        <span class="sr-only">Code still streaming</span>
      </Show>
    </figure>
  );
}
