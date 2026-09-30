import { createSignal, onCleanup, onMount, type Accessor } from "solid-js";
import type { CerebroService } from "./service";
import type { LiveStats } from "./types";

/**
 * The current HUD sample, kept current by the service's subscription for as long as the calling
 * component lives. Every view that shows live numbers (the status bar, the Home tiles) reads it
 * this way, so they always show the same sample: two disagreeing readouts would undercut the whole
 * "measured, not guessed" point.
 *
 * The one-off `getLiveStats()` only seeds the value; if a pushed sample has already arrived, the
 * (possibly older) seed is dropped.
 */
export function createLiveStats(service: CerebroService): Accessor<LiveStats | null> {
  const [stats, setStats] = createSignal<LiveStats | null>(null);
  onMount(() => {
    let alive = true;
    let pushed = false;
    service
      .getLiveStats()
      .then((s) => {
        if (alive && !pushed) setStats(s);
      })
      .catch(() => {
        if (alive && !pushed) setStats(null);
      });
    const off = service.subscribeLiveStats((s) => {
      if (!alive) return;
      pushed = true;
      setStats(s);
    });
    onCleanup(() => {
      alive = false;
      off();
    });
  });
  return stats;
}
