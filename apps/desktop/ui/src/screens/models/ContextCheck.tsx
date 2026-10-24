import { createSignal, For, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { useService } from "../../service/context";
import type { ContextCheckResult } from "../../service/types";

/**
 * "Does it still work this long?" for the loaded model: a passcode hidden at the start, middle and
 * end of documents a quarter, half and nine tenths of its context long, and whether it found each
 * one. Past the trained length (YaRN) this is how to know the stretch holds for this model.
 */
export function ContextCheck(props: { trained: number }): JSX.Element {
  const service = useService();
  const [busy, setBusy] = createSignal(false);
  const [result, setResult] = createSignal<ContextCheckResult | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const run = () => {
    if (!service.contextCheck || busy()) return;
    setBusy(true);
    setError(null);
    service
      .contextCheck()
      .then(setResult)
      .catch((e: unknown) => setError(e instanceof Error && e.message ? e.message : "The check could not run."))
      .finally(() => setBusy(false));
  };
  const found = () => (result()?.probes ?? []).filter((p) => p.found).length;
  return (
    <Show when={service.contextCheck}>
      <section class="ctxcheck card" aria-labelledby="ctxcheck-title">
        <div class="ctxcheck__head">
          <h3 id="ctxcheck-title">Long-context check</h3>
          <button type="button" class="btn btn--secondary btn--sm" disabled={busy()} aria-busy={busy() ? "true" : undefined} onClick={run}>
            {busy() ? "Checking…" : result() ? "Check again" : "Check long context"}
          </button>
        </div>
        <p class="hint">
          Hides a passcode at the start, middle and end of three long documents and asks the model for it. A model that reads its whole context finds all nine.
          {props.trained > 0 ? ` It was trained for ${props.trained.toLocaleString("en-US")} tokens; past that Cerebro stretches it with YaRN.` : ""} Long contexts take a few minutes; the chat's cache is cleared.
        </p>
        <Show when={error()}>
          <p class="notice" role="alert">
            <Icon name="alert" size={16} /> <span>{error()}</span>
          </p>
        </Show>
        <Show when={result()}>
          {(r) => (
            <>
              <p class="ctxcheck__verdict" data-ok={found() === r().probes.length ? "true" : "false"}>
                <strong>
                  {found()} of {r().probes.length} found
                </strong>{" "}
                at a {r().context.toLocaleString("en-US")}-token context{r().factor > 1 ? ` (${r().factor.toFixed(2)}x the trained length, YaRN)` : ""}.
              </p>
              <table class="ctxcheck__table">
                <thead>
                  <tr>
                    <th scope="col">Document</th>
                    <th scope="col">Hidden at</th>
                    <th scope="col">Result</th>
                    <th scope="col">Answer</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={r().probes}>
                    {(p) => (
                      <tr data-found={p.found ? "true" : "false"}>
                        <td class="mono">{p.tokens.toLocaleString("en-US")} tokens</td>
                        <td>{p.depth < 0.3 ? "start" : p.depth > 0.7 ? "end" : "middle"}</td>
                        <td>{p.found ? "Found" : "Missed"}</td>
                        <td class="mono ctxcheck__reply">{p.reply || "(nothing)"}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
            </>
          )}
        </Show>
      </section>
    </Show>
  );
}
