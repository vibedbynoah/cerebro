import { createResource, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Section } from "../../components/Screen";
import { formatSeconds, groupTrials, settingsInWords, speedLine, trialLine } from "../../lib/machine-tune";
import { useService } from "../../service/context";
import type { MachineTuneOutcome, MachineTuneTrial } from "../../service/types";

/**
 * Tune for this machine: times the processor settings that depend on the hardware (threads, how
 * work is split, four-row dots) on the loaded model and keeps the fastest (cerebro_engine::tune).
 * Stop goes through the same stop Chat uses (the engine's cancel switch is shared). The last
 * finished tune is shown again on the next visit.
 */
export function MachineTune(props: { onTuned?: () => void }): JSX.Element {
  const service = useService();
  const [running, setRunning] = createSignal(false);
  const [trials, setTrials] = createSignal<MachineTuneTrial[]>([]);
  const [finished, setFinished] = createSignal<MachineTuneOutcome | null>(null);
  const [problem, setProblem] = createSignal<string | null>(null);
  const [stopped, setStopped] = createSignal(false);
  let abort: AbortController | null = null;

  // A tune needs a model in the llm slot; the hint says so before the button does anything.
  const [loaded] = createResource(() =>
    service
      .listLoaded()
      .then((list) => list.some((l) => l.slot === "llm"))
      .catch(() => false),
  );
  // The last finished tune, kept beside the app's switches on this machine.
  const [last] = createResource(() => service.lastMachineTune().catch(() => null));

  // A run that was stopped or failed has nothing to report beyond its note, so the saved result
  // stays out of the way until the next visit.
  const shown = (): MachineTuneOutcome | null => (running() || stopped() || problem() !== null ? null : finished() ?? last() ?? null);
  // Live trials while a run is in flight (or stopped part-way); otherwise the shown outcome's own.
  const shownTrials = (): MachineTuneTrial[] => (trials().length > 0 ? trials() : shown()?.trials ?? []);
  const groups = () => groupTrials(shownTrials());

  const start = async () => {
    const controller = new AbortController();
    abort = controller;
    setRunning(true);
    setTrials([]);
    setFinished(null);
    setProblem(null);
    setStopped(false);
    try {
      const result = await service.tuneMachine((t) => setTrials((prev) => [...prev, t]), controller.signal);
      setFinished(result);
      // The engine's switches now hold the kept settings; the list below shows the new values.
      props.onTuned?.();
    } catch (e) {
      if (controller.signal.aborted) setStopped(true);
      else setProblem(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      abort = null;
    }
  };

  // Leaving the screen stops a running tune rather than letting it run unseen.
  onCleanup(() => abort?.abort());

  return (
    <Section
      id="machine-tune"
      title="Tune for this machine"
      description="Times the processor settings on the loaded model and keeps the fastest. About a minute; the model is busy meanwhile."
    >
      <div class="machine-tune card">
        <div class="machine-tune__actions">
          <button class="btn btn--primary" type="button" onClick={() => void start()} disabled={running() || loaded() !== true}>
            Tune for this machine
          </button>
          <Show when={running()}>
            <button class="btn btn--secondary" type="button" onClick={() => abort?.abort()}>
              Stop
            </button>
          </Show>
          <Show when={loaded() === false && !running()}>
            <span class="hint">Load a model first.</span>
          </Show>
        </div>
        <Show when={running()}>
          <p class="machine-tune__status" role="status">
            Timing one setting at a time. The model is busy until this finishes.
          </p>
        </Show>
        <Show when={stopped()}>
          <p class="machine-tune__status" role="status">
            Tuning was stopped. Settings are unchanged.
          </p>
        </Show>
        <Show when={problem()}>
          <p class="machine-tune__status" role="status">
            Tuning could not finish: {problem()}
          </p>
        </Show>
        <Show when={shown()}>
          {(o) => (
            <div class="machine-tune__result">
              <p class="machine-tune__note" data-testid="machine-tune-note">
                <Show when={!finished()}>
                  <span class="machine-tune__last">Last tune: </span>
                </Show>
                {o().note}
              </p>
              {/* No trials: nothing on the processor to tune (the whole model is on the GPU). */}
              <Show when={o().trials.length > 0}>
                <p class="machine-tune__speed mono">{speedLine(o())}</p>
                <p class="machine-tune__settings">
                  {o().kept ? "Kept settings" : "Current settings"}: {settingsInWords(o().settings)}
                </p>
              </Show>
              <Show when={o().trials.length > 0}>
                <p class="machine-tune__machine">{o().machine}</p>
                <p class="machine-tune__took">Took {formatSeconds(o().seconds)}.</p>
              </Show>
            </div>
          )}
        </Show>
        <Show when={groups().length > 0}>
          <div class="machine-tune__trials" aria-live="polite">
            <For each={groups()}>
              {(g) => (
                <div class="machine-tune__phase">
                  <h3 class="machine-tune__phase-title">{g.title}</h3>
                  <ul class="machine-tune__list">
                    <For each={g.trials}>
                      {(t, i) => (
                        <li class="machine-tune__trial" data-fastest={i() === g.fastest}>
                          <span class="machine-tune__trial-text">{trialLine(t)}</span>
                          <Show when={i() === g.fastest}>
                            <span class="chip" data-tone="on">
                              Fastest
                            </span>
                          </Show>
                        </li>
                      )}
                    </For>
                  </ul>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </Section>
  );
}
