import { Show, type JSX } from "solid-js";
import { pillView } from "../lib/monitor";
import { hrefFor } from "../lib/routes";
import { useMetrics } from "../service/metrics";

/**
 * The engine's state in the title bar, on every screen: Idle, Loading model, Reading prompt 43%,
 * Generating 21.4 tok/s, and how many requests wait. A link to the Monitor. Not a live region: a
 * value changing every second would flood a screen reader; the link's name carries the state.
 */
export function StatusPill(): JSX.Element {
  const feed = useMetrics();
  const view = () => pillView(feed.snapshot()?.live ?? null, feed.reachable());
  const label = () => {
    const v = view();
    return `Engine: ${v.text}${v.queued > 0 ? `, ${v.queued} queued` : ""}. Open Monitor`;
  };
  return (
    <a class="engine-pill" href={hrefFor("monitor")} data-tone={view().tone} data-press aria-label={label()} title="Open Monitor">
      <span class="engine-pill__dot" aria-hidden="true" />
      <span class="engine-pill__text">{view().text}</span>
      <Show when={view().queued > 0}>
        <span class="engine-pill__queue">{view().queued} queued</span>
      </Show>
    </a>
  );
}
