import { For, Show, type JSX } from "solid-js";
import { Icon, type IconName } from "./icons";

export type PreviewArt = "chat" | "catalog" | "gallery" | "documents" | "plugins" | "tools";

export interface PreviewFeature {
  icon: IconName;
  title: string;
  text: JSX.Element;
}

/**
 * Body of a screen that arrives in a later build. It says plainly that the screen is planned, what it
 * will do (three concrete capabilities), and shows a wireframe sketch of the layout that assembles
 * when it scrolls into view. The sketch is decorative (aria-hidden) and holds no numbers, so nothing
 * in it can read as a measurement.
 */
export function Preview(props: {
  art: PreviewArt;
  title: string;
  lede: JSX.Element;
  features: readonly PreviewFeature[];
  children?: JSX.Element;
}): JSX.Element {
  return (
    <section class="preview card reveal" data-reveal aria-labelledby={`preview-${props.art}-title`}>
      <div class="preview__text">
        <span class="chip preview__status" data-tone="active">
          <span class="preview__status-dot" aria-hidden="true" />
          Planned for a later build
        </span>
        <h2 class="preview__title" id={`preview-${props.art}-title`}>
          {props.title}
        </h2>
        <p class="preview__lede">{props.lede}</p>
        <ul class="preview__features">
          <For each={props.features}>
            {(f) => (
              <li class="preview__feature">
                <span class="preview__feature-icon" aria-hidden="true">
                  <Icon name={f.icon} size={16} />
                </span>
                <span class="preview__feature-text">
                  <span class="preview__feature-title">{f.title}</span>
                  <span class="preview__feature-desc">{f.text}</span>
                </span>
              </li>
            )}
          </For>
        </ul>
        <Show when={props.children}>
          <div class="preview__extra">{props.children}</div>
        </Show>
      </div>
      <div class={`preview__art preview__art--${props.art}`} aria-hidden="true">
        <Art kind={props.art} />
      </div>
    </section>
  );
}

/** Skeleton pieces: `.sk` blocks that rise in one after another (CSS stagger, transform/opacity). */
function Art(props: { kind: PreviewArt }): JSX.Element {
  const lines = (n: number, cls = "") => (
    <For each={Array.from({ length: n }, (_, i) => i)}>{(i) => <span class={`sk-line sk-line--${(i % 3) + 1} ${cls}`} />}</For>
  );
  switch (props.kind) {
    case "chat":
      return (
        <div class="sk-stack">
          <div class="sk sk-bubble sk-bubble--in">{lines(2)}</div>
          <div class="sk sk-bubble sk-bubble--out">{lines(1)}</div>
          <div class="sk sk-bubble sk-bubble--in">
            {lines(3)}
            <span class="sk-code">{lines(2, "sk-line--code")}</span>
          </div>
          <div class="sk sk-input">
            <span class="sk-line sk-line--2" />
            <span class="sk-caret" />
            <span class="sk-send">
              <Icon name="arrowRight" size={14} />
            </span>
          </div>
        </div>
      );
    case "catalog":
      return (
        <div class="sk-stack">
          <div class="sk sk-search">
            <Icon name="search" size={14} />
            <span class="sk-line sk-line--3" />
          </div>
          <For each={[0.45, 0.7, 0.92]}>
            {(fit, i) => (
              <div class="sk sk-row">
                <span class="sk-avatar" />
                <span class="sk-row__text">
                  <span class={`sk-line sk-line--${i() + 1}`} />
                  <span class="sk-line sk-line--meta" />
                </span>
                <span class="sk-fit" data-fit={fit > 0.9 ? "tight" : "ok"}>
                  <span class={`sk-fit__fill sk-fit__fill--${Math.round(fit * 100)}`} />
                </span>
              </div>
            )}
          </For>
        </div>
      );
    case "gallery":
      return (
        <div class="sk-stack">
          <div class="sk-grid">
            <For each={[1, 2, 3, 4, 5, 6]}>{(n) => <span class={`sk sk-tile sk-tile--${n}`} />}</For>
          </div>
          <div class="sk sk-input">
            <span class="sk-line sk-line--1" />
            <span class="sk-send">
              <Icon name="images" size={14} />
            </span>
          </div>
        </div>
      );
    case "documents":
      return (
        <div class="sk-split">
          <div class="sk-stack">
            <For each={[1, 2, 3, 4]}>
              {(n) => (
                <div class="sk sk-file" classList={{ "is-hit": n === 2 }}>
                  <Icon name="documents" size={14} />
                  <span class={`sk-line sk-line--${(n % 3) + 1}`} />
                </div>
              )}
            </For>
          </div>
          <div class="sk sk-bubble sk-bubble--in sk-answer">
            {lines(4)}
            <span class="sk-cites">
              <span class="sk-cite">1</span>
              <span class="sk-cite">2</span>
            </span>
          </div>
        </div>
      );
    case "plugins":
      return (
        <div class="sk-grid sk-grid--2">
          <For each={[true, true, false, true]}>
            {(on, i) => (
              <div class="sk sk-card" classList={{ "sk-card--focus": i() === 0 }}>
                <span class="sk-avatar sk-avatar--square" />
                <span class={`sk-line sk-line--${(i() % 3) + 1}`} />
                <span class="sk-line sk-line--meta" />
                <span class="sk-toggle" data-on={on} />
              </div>
            )}
          </For>
        </div>
      );
    case "tools":
      return (
        <div class="sk-stack">
          <div class="sk sk-bubble sk-bubble--in">{lines(2)}</div>
          <div class="sk sk-prompt">
            <span class="sk-prompt__head">
              <Icon name="lock" size={14} />
              <span class="sk-line sk-line--2" />
            </span>
            <span class="sk-line sk-line--meta" />
            <span class="sk-prompt__actions">
              <span class="sk-btn" />
              <span class="sk-btn sk-btn--primary" />
            </span>
          </div>
          <div class="sk sk-bubble sk-bubble--in">{lines(1)}</div>
        </div>
      );
  }
}
