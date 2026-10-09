import { createSignal, onCleanup, onMount, type Accessor } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { applyTuneEvent, emptyTuneView, type TuneView } from "../../lib/tune";
import { notify } from "../../lib/toast";
import type { CerebroService } from "../../service/service";
import type { TuneProfile } from "../../service/types";

export type TuneAction = "start" | "cancel" | "resume";

export interface TuneSession {
  /** The run as the events describe it (a Solid store: rows keep their identity as they change). */
  view: TuneView;
  /** False until the service has replayed the current run, so the screen never flashes "no tune". */
  ready: Accessor<boolean>;
  /** The action waiting on the service, or null. Buttons show it as busy and ignore repeats. */
  busy: Accessor<TuneAction | null>;
  start(modelId: string, profile: TuneProfile): Promise<boolean>;
  cancel(): Promise<void>;
  resume(): Promise<void>;
}

/**
 * Follows the service's tune for as long as the calling component lives. The tune itself keeps
 * running in the service when the screen is left; coming back replays it from the first event.
 */
export function createTuneSession(service: CerebroService): TuneSession {
  const [view, setView] = createStore<TuneView>(emptyTuneView());
  const [ready, setReady] = createSignal(false);
  const [busy, setBusy] = createSignal<TuneAction | null>(null);

  onMount(() => {
    const off = service.subscribeTune((events, replay) => {
      if (events.length > 0)
        setView(
          produce((v) => {
            for (const e of events) applyTuneEvent(v, e);
          }),
        );
      // After the replay has rendered: rows that arrive from now on are new and animate in; the
      // replayed ones (coming back to the screen) appear at once.
      if (replay) queueMicrotask(() => setReady(true));
    });
    onCleanup(off);
  });

  const run = async (action: TuneAction, fn: () => Promise<unknown>, failure: [string, string]): Promise<boolean> => {
    if (busy() !== null) return false;
    setBusy(action);
    try {
      await fn();
      return true;
    } catch {
      notify(failure[0], failure[1], "error");
      return false;
    } finally {
      setBusy(null);
    }
  };

  return {
    view,
    ready,
    busy,
    start: (modelId, profile) =>
      run("start", () => service.startTune(modelId, profile), ["The tune did not start", "Nothing changed. If another tune is running, cancel it first."]),
    async cancel() {
      const id = view.run?.runId;
      if (!id) return;
      await run("cancel", () => service.cancelTune(id), ["The tune could not be cancelled", "It is still running. Try again."]);
    },
    async resume() {
      const id = view.run?.runId;
      if (!id) return;
      await run("resume", () => service.resumeTune(id), ["The tune could not be resumed", "Its finished steps are kept. Try again, or start a new tune."]);
    },
  };
}
