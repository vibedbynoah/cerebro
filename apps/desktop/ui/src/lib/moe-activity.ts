import { createSignal, onCleanup, onMount, type Accessor } from "solid-js";
import { useService } from "../service/context";
import { useMetrics } from "../service/metrics";
import type { MoeActivity } from "../service/types";

/**
 * Which experts a mixture-of-experts model is firing, read from the engine about seven times a
 * second while it reads a prompt or writes, and not at all otherwise. One feed for the whole app:
 * the title bar's logo and Chat's heatmap share it.
 */
const [activity, setActivity] = createSignal<MoeActivity | null>(null);
const [active, setActive] = createSignal(false);
let users = 0;
let timer: ReturnType<typeof setTimeout> | undefined;

/** The latest map (null for a dense model), and whether the model is working right now. */
export function useMoeActivity(): { activity: Accessor<MoeActivity | null>; active: Accessor<boolean> } {
  const service = useService();
  const feed = useMetrics();
  onMount(() => {
    users += 1;
    if (users > 1 || !service.moeActivity) return;
    const tick = async () => {
      const s = feed.snapshot();
      const working = !!s?.engine.moe && (s.live.state === "generating" || s.live.state === "reading");
      setActive(working);
      if (working) {
        try {
          setActivity((await service.moeActivity!()) ?? null);
        } catch {
          /* the next tick tries again */
        }
      }
      if (users > 0) timer = setTimeout(() => void tick(), working ? 140 : 500);
    };
    void tick();
  });
  onCleanup(() => {
    users -= 1;
    if (users === 0) {
      clearTimeout(timer);
      setActive(false);
    }
  });
  return { activity, active };
}

/** The expert to show as firing: the first pick of the latest token in the middle layer. */
export function firingExpert(a: MoeActivity | null): number | null {
  if (!a || a.last.length === 0) return null;
  const row = a.last[Math.floor(a.last.length / 2)] ?? a.last[0];
  return row && row.length > 0 ? row[0]! : null;
}
