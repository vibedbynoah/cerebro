import { Show, type JSX } from "solid-js";
import { routeById, type RouteId } from "../lib/routes";

/**
 * Shared screen scaffolding: one heading tier per level (h1 screen title and h2 section
 * titles both in the interface font), one spacing rhythm (--space-8 between sections, --space-4 inside cards).
 */
export function ScreenHeader(props: { route: RouteId; title?: string; lede?: JSX.Element; actions?: JSX.Element }): JSX.Element {
  const def = () => routeById(props.route);
  return (
    <header class="screen-header">
      <div class="screen-header__text">
        <h1 class="screen-title">{props.title ?? def().label}</h1>
        <Show when={props.lede}>
          <p class="screen-lede">{props.lede}</p>
        </Show>
      </div>
      <Show when={props.actions}>
        <div class="screen-header__actions">{props.actions}</div>
      </Show>
    </header>
  );
}

export function Section(props: { title: string; description?: JSX.Element; children: JSX.Element; id?: string }): JSX.Element {
  return (
    <section class="section reveal" id={props.id} data-reveal aria-labelledby={props.id ? `${props.id}-title` : undefined}>
      <div class="section__head">
        <h2 class="section__title" id={props.id ? `${props.id}-title` : undefined}>
          {props.title}
        </h2>
        <Show when={props.description}>
          <p class="section__desc">{props.description}</p>
        </Show>
      </div>
      {props.children}
    </section>
  );
}

/** "Illustrative data — not a measurement": always inside the figure it qualifies. */
export function IllustrativeFlag(): JSX.Element {
  // The desktop app shows only real data; the badge belongs to the web preview's sample data.
  if (__DESKTOP__) return null;
  return <span class="stat-tile__flag">Illustrative data — not a measurement</span>;
}
