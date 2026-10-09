import { createSignal, For, onCleanup, type JSX } from "solid-js";
import { Icon } from "./icons";

/**
 * Design-system code block with a copy button. Lines are rendered as text nodes; the copy writes
 * plain text only. Clipboard access can be refused (permissions, headless browsers): the status line
 * then says so instead of pretending.
 */
export function CodeBlock(props: {
  title?: string;
  lines: readonly string[];
  label?: string;
  large?: boolean;
  /** The lines are one command wrapped for reading: one prompt, and Copy gives a single line. */
  oneCommand?: boolean;
}): JSX.Element {
  const [status, setStatus] = createSignal("");
  const [done, setDone] = createSignal(false);
  let timer: ReturnType<typeof setTimeout> | undefined;
  onCleanup(() => clearTimeout(timer));

  const copy = async () => {
    clearTimeout(timer);
    try {
      await navigator.clipboard.writeText(props.oneCommand ? props.lines.map((l) => l.trim()).join(" ") : props.lines.join("\n"));
      setDone(true);
      setStatus("Copied");
    } catch {
      setDone(false);
      setStatus("Copy unavailable here");
    }
    timer = setTimeout(() => {
      setDone(false);
      setStatus("");
    }, 2000);
  };

  return (
    <figure class={props.large ? "code-block code-block--lg" : "code-block"}>
      <figcaption class="code-block__title">
        <span>{props.title ?? "Terminal"}</span>
        <span class="code-block__status" role="status" aria-live="polite">
          {status()}
        </span>
        <button
          class="btn btn--icon code-block__copy"
          classList={{ "is-done": done() }}
          type="button"
          data-press
          aria-label={props.label ?? "Copy commands"}
          onClick={() => void copy()}
        >
          <Icon name="copy" class="icon-copy" size={16} />
          <Icon name="check" class="icon-check" size={16} />
        </button>
      </figcaption>
      <pre tabindex="0">
        <code>
          <For each={props.lines}>
            {(line, i) => (
              <>
                <span class="code-block__prompt" aria-hidden="true">
                  {props.oneCommand && i() > 0 ? "  " : "> "}
                </span>
                {line}
                {i() < props.lines.length - 1 ? "\n" : ""}
              </>
            )}
          </For>
        </code>
      </pre>
    </figure>
  );
}
