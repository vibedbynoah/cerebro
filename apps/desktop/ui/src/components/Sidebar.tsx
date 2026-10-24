import { Tooltip } from "@ark-ui/solid/tooltip";
import { createEffect, createSignal, For, on, onCleanup, onMount, type JSX } from "solid-js";
import { Portal } from "solid-js/web";
import { hrefFor, ROUTES, type RouteDef, type RouteGroup, type RouteId } from "../lib/routes";
import { Icon } from "./icons";

const GROUPS: readonly RouteGroup[] = ["Run", "Create", "Extend", "System"];

/**
 * Collapsible primary navigation. Every entry is a real link (`#/id`). Collapsed, labels stay in the
 * accessibility tree (visually hidden) and a tooltip shows them on hover and keyboard focus.
 * Up/Down/Home/End move focus between entries; Tab still reaches each one.
 */
export function Sidebar(props: { route: RouteId; collapsed: boolean; onToggle: () => void }): JSX.Element {
  let list: HTMLDivElement | undefined;
  let indicator: HTMLSpanElement | undefined;

  // The active marker slides between entries (transform only). Positions are read once per route or
  // collapse change, never per frame.
  const place = () => {
    if (!list || !indicator) return;
    const active = list.querySelector<HTMLElement>('a[aria-current="page"]');
    if (!active) return;
    indicator.style.setProperty("--indicator-y", `${active.offsetTop}px`);
    indicator.style.setProperty("--indicator-h", `${active.offsetHeight}px`);
  };
  onMount(() => {
    place();
    // Enable the slide only after the first placement, so the marker never flies in on load.
    const raf = requestAnimationFrame(() => indicator?.classList.add("is-ready"));
    onCleanup(() => cancelAnimationFrame(raf));
  });
  createEffect(on([() => props.route, () => props.collapsed], () => queueMicrotask(place), { defer: true }));

  const onKeyDown = (e: KeyboardEvent) => {
    if (!list || !["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    const links = Array.from(list.querySelectorAll<HTMLAnchorElement>("a.nav-link"));
    const at = links.indexOf(document.activeElement as HTMLAnchorElement);
    if (at === -1) return;
    e.preventDefault();
    const last = links.length - 1;
    const next = e.key === "Home" ? 0 : e.key === "End" ? last : e.key === "ArrowDown" ? (at === last ? 0 : at + 1) : at === 0 ? last : at - 1;
    links[next]?.focus();
  };

  // Controlled `open` instead of `disabled`: the asChild trigger keeps the event handlers from its
  // first render, so a `disabled` that flips at runtime (Ctrl+B) would never reach them. The machine
  // always tracks the pointer; the tooltip is only allowed to show while the sidebar is collapsed.
  const link = (r: RouteDef) => {
    const [wants, setWants] = createSignal(false);
    createEffect(on(() => props.collapsed, () => setWants(false), { defer: true }));
    return (
    <Tooltip.Root
      openDelay={250}
      closeDelay={0}
      open={props.collapsed && wants()}
      onOpenChange={(d) => setWants(d.open)}
      positioning={{ placement: "right", gutter: 10 }}
    >
      <Tooltip.Trigger
        asChild={(triggerProps) => (
          <a
            {...triggerProps()}
            class="nav-link"
            href={hrefFor(r.id)}
            aria-current={props.route === r.id ? "page" : undefined}
            data-route={r.id}
          >
            <Icon name={r.icon} class="nav-link__icon" />
            <span class="nav-link__label">{r.label}</span>
          </a>
        )}
      />
      <Portal>
        <Tooltip.Positioner>
          <Tooltip.Content class="tooltip">{r.label}</Tooltip.Content>
        </Tooltip.Positioner>
      </Portal>
    </Tooltip.Root>
    );
  };

  return (
    <nav id="sidebar" class="sidebar" classList={{ "is-collapsed": props.collapsed }} aria-label="Main">
      <div class="sidebar__list" ref={list} onKeyDown={onKeyDown}>
        <span class="nav-indicator" ref={indicator} aria-hidden="true" />
        <For each={GROUPS}>
          {(group) => (
            <div class="nav-group">
              <p class="nav-group__title" id={`nav-${group}`}>
                <span class="nav-group__text">
                  {group}
                </span>
              </p>
              <ul class="nav-group__list" aria-labelledby={`nav-${group}`}>
                <For each={ROUTES.filter((r) => r.group === group && r.inSidebar !== false)}>{(r) => <li>{link(r)}</li>}</For>
              </ul>
            </div>
          )}
        </For>
      </div>
      <div class="sidebar__foot">
        <button
          type="button"
          class="sidebar-toggle"
          data-press
          aria-controls="sidebar"
          aria-expanded={!props.collapsed}
          aria-keyshortcuts="Control+B"
          onClick={() => props.onToggle()}
        >
          <Icon name="sidebar" class="sidebar-toggle__icon" />
          <span class="sidebar-toggle__label">{props.collapsed ? "Expand sidebar" : "Collapse sidebar"}</span>
        </button>
      </div>
    </nav>
  );
}
