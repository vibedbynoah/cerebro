import { createResource, For, Match, Show, Switch, type JSX } from "solid-js";
import { CodeBlock } from "../components/CodeBlock";
import { IllustrativeFlag, ScreenHeader, Section } from "../components/Screen";
import { Term } from "../components/Term";
import { fmt1, fmtContext } from "../lib/format";
import { useService } from "../service/context";
import type { ModelSummary } from "../service/types";

export function TuneChip(props: { model: ModelSummary }): JSX.Element {
  return (
    <Switch>
      <Match when={props.model.tune === "tuned"}>
        <span class="chip" data-tone="on">
          Tuned
        </span>
      </Match>
      <Match when={props.model.tune === "defaults-fastest"}>
        <span class="chip" data-tone="neutral">
          Defaults were fastest
        </span>
      </Match>
      <Match when={props.model.tune === "tuning"}>
        <span class="chip" data-tone="active">
          Measuring
        </span>
      </Match>
      <Match when={props.model.tune === "untuned"}>
        <span class="chip" data-tone="off">
          Not tuned yet
        </span>
      </Match>
    </Switch>
  );
}

export function Models(): JSX.Element {
  const service = useService();
  const [models] = createResource(() => service.listModels());

  return (
    <div class="screen">
      <ScreenHeader route="models" />
      <Section
        id="library"
        title="On this machine"
        description={
          <>
            Each file is one model at one <Term k="quant" /> level. Speeds are decode <Term k="tokps">tok/s</Term>, always shown beside quality.
          </>
        }
      >
        <div class="card table-card">
          <div class="table-card__head">
            <IllustrativeFlag />
          </div>
          <div class="table-scroll" tabindex="0" role="region" aria-label="Models on this machine">
            <table class="table">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Quant</th>
                  <th scope="col" class="num">
                    Size
                  </th>
                  <th scope="col" class="num">
                    Context
                  </th>
                  <th scope="col">Tune</th>
                  <th scope="col" class="num">
                    Defaults → tuned
                  </th>
                  <th scope="col" class="num">
                    <Term k="quality">Quality</Term>
                  </th>
                </tr>
              </thead>
              <tbody>
                <For each={models()} fallback={<tr><td colSpan={7} class="table__empty">Loading models…</td></tr>}>
                  {(m) => (
                    <tr data-model={m.id}>
                      <th scope="row">
                        <span class="model-name">{m.name}</span>
                        <span class="model-meta">
                          {m.params} · {m.architecture === "moe" ? "MoE" : "dense"}
                        </span>
                      </th>
                      <td class="mono">{m.quant}</td>
                      <td class="num mono">
                        {fmt1(m.sizeGiB)}
                        <span class="unit">GiB</span>
                      </td>
                      <td class="num mono">{fmtContext(m.contextMax)}</td>
                      <td>
                        <TuneChip model={m} />
                      </td>
                      <td class="num mono">
                        <Show when={m.defaultsTps !== null && m.tunedTps !== null} fallback={<span class="muted">–</span>}>
                          {fmt1(m.defaultsTps ?? 0)} → {fmt1(m.tunedTps ?? 0)}
                        </Show>
                      </td>
                      <td class="num mono">
                        {/* No quality run exists yet; speed is never shown as if it were the whole story. */}
                        <span class="muted" aria-hidden="true">
                          –
                        </span>
                        <span class="sr-only">Not measured yet</span>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </div>
      </Section>
      <Section id="cli" title="From the command line" description="List these models with their file paths, or measure one that has not been tuned yet.">
        <CodeBlock lines={["cerebro models", "cerebro tune qwen3-8b-q5km"]} />
      </Section>
    </div>
  );
}
