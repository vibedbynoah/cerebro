import { Show, type JSX } from "solid-js";
import { routeById, type RouteId } from "../lib/routes";
import { Icon } from "./icons";

/**
 * Shared screen scaffolding: one heading tier per level (h1 screen title in Fraunces, h2 section
 * titles in Inter), one spacing rhythm (--space-8 between sections, --space-4 inside cards).
 */
export function ScreenHeader(props: { route: RouteId; title?: string; lede?: JSX.Element; actions?: JSX.Element }): JSX.Element {
  const def = () => routeById(props.route);
  return (
    <header class="screen-header">
      <div class="screen-header__text">
        <p class="eyebrow">
          <Icon name={def().icon} size={14} />
          {def().label}
        </p>
        <h1 class="screen-title">{props.title ?? def().label}</h1>
        <p class="screen-lede">{props.lede ?? def().summary}</p>
      </div>
      <Show when={props.actions}>
        <div class="screen-header__actions">{props.actions}</div>
      </Show>
    </header>
  );
}

export function Section(props: { title: string; description?: JSX.Element; children: JSX.Element; id?: string }): JSX.Element {
  return (
    <section class="section reveal" data-reveal aria-labelledby={props.id ? `${props.id}-title` : undefined}>
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
  return <span class="stat-tile__flag">Illustrative data — not a measurement</span>;
}
