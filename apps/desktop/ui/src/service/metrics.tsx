import { createContext, createSignal, onCleanup, onMount, useContext, type Accessor, type JSX } from "solid-js";
import { emptyAlerts, stepAlerts, type AlertKind, type AlertTracker } from "../lib/monitor";
import { notify } from "../lib/toast";
import { useService } from "./context";
import type { MetricsSnapshot } from "./monitor-types";

/** The engine's /metrics snapshot, polled once a second for the whole app (header pill, Monitor,
 *  alerts), so every view shows the same sample. */
export interface MetricsFeed {
  snapshot: Accessor<MetricsSnapshot | null>;
  /** False after three polls in a row failed; true again on the next success. */
  reachable: Accessor<boolean>;
  alerts: Accessor<AlertTracker>;
}

export const POLL_MS = 1000;
const FAILURES_BEFORE_UNREACHABLE = 3;
/** One toast per kind of alert at most this often, however often it flaps. */
const TOAST_COOLDOWN_MS = 120_000;

const FeedContext = createContext<MetricsFeed>();

export function MetricsProvider(props: { children: JSX.Element }): JSX.Element {
  const service = useService();
  const [snapshot, setSnapshot] = createSignal<MetricsSnapshot | null>(null);
  const [reachable, setReachable] = createSignal(true);
  const [alerts, setAlerts] = createSignal<AlertTracker>(emptyAlerts());

  onMount(() => {
    let alive = true;
    let inFlight = false;
    let failures = 0;
    const lastToast: Partial<Record<AlertKind, number>> = {};
    const tick = () => {
      // A hidden window skips polls (nothing to draw); a slow answer is never overlapped.
      if (inFlight || document.visibilityState === "hidden") return;
      inFlight = true;
      service
        .getMetrics()
        .then((s) => {
          if (!alive) return;
          failures = 0;
          setReachable(true);
          setSnapshot(s);
          const { tracker, raised } = stepAlerts(alerts(), s);
          setAlerts(tracker);
          for (const a of raised) {
            const last = lastToast[a.kind];
            if (last !== undefined && s.at - last < TOAST_COOLDOWN_MS) continue;
            lastToast[a.kind] = s.at;
            notify(a.title, s.provenance === "illustrative" ? `${a.detail} Illustrative data.` : a.detail, "warning");
          }
        })
        .catch(() => {
          if (!alive) return;
          failures++;
          if (failures >= FAILURES_BEFORE_UNREACHABLE) setReachable(false);
        })
        .finally(() => {
          inFlight = false;
        });
    };
    tick();
    const timer = setInterval(tick, POLL_MS);
    onCleanup(() => {
      alive = false;
      clearInterval(timer);
    });
  });

  return <FeedContext.Provider value={{ snapshot, reachable, alerts }}>{props.children}</FeedContext.Provider>;
}

export function useMetrics(): MetricsFeed {
  const feed = useContext(FeedContext);
  if (!feed) throw new Error("useMetrics() called outside <MetricsProvider>");
  return feed;
}
