import { Show, type JSX } from "solid-js";
import { readPref, writePref } from "../lib/storage";
import { Icon } from "./icons";
import { routeById, type RouteId } from "../lib/routes";

/**
 * Shared screen scaffolding: one heading tier per level (h1 screen title and h2 section
 * titles both in the interface font), one spacing rhythm (--space-8 between sections, --space-4 inside cards).
 */
/** Every screen opens with its title alone (and its actions): the same quiet header everywhere.
 *  `lede` is accepted for older call sites and not shown; sections carry the explanations. */
export function ScreenHeader(props: { route: RouteId; title?: string; lede?: JSX.Element; actions?: JSX.Element }): JSX.Element {
  const def = () => routeById(props.route);
  return (
    <header class="screen-header">
      <div class="screen-header__text">
        <h1 class="screen-title">{props.title ?? def().label}</h1>
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

/**
 * A large, collapsible group of related sections: a title, one line on what is inside, then the
 * sections themselves (each a smaller titled block). Open or closed is remembered per viewer;
 * a search can open it (`data-group`).
 */
export function Group(props: { id: string; title: string; summary: string; defaultOpen?: boolean; children: JSX.Element }): JSX.Element {
  const key = `cerebro-group-${props.id}`;
  const stored = readPref(key);
  return (
    <details
      class="group"
      id={props.id}
      data-group
      open={stored === null ? (props.defaultOpen ?? false) : stored === "1"}
      onToggle={(e) => writePref(key, e.currentTarget.open ? "1" : "0")}
    >
      <summary class="group__head">
        <span class="group__text">
          <span class="group__title">{props.title}</span>
          <span class="group__summary">{props.summary}</span>
        </span>
        <Icon name="chevron" size={16} />
      </summary>
      <div class="group__body">{props.children}</div>
    </details>
  );
}

/** "Illustrative data — not a measurement": always inside the figure it qualifies. */
export function IllustrativeFlag(): JSX.Element {
  // The desktop app shows only real data; the badge belongs to the web preview's sample data.
  if (__DESKTOP__) return null;
  return <span class="stat-tile__flag">Illustrative data — not a measurement</span>;
}
