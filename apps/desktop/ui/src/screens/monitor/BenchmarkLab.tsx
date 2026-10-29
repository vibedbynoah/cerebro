import { modelLabel } from "../../lib/model-label";
import "../../styles/bench.css";
import { createEffect, createMemo, createResource, createSignal, For, onCleanup, onMount, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import {
  BENCH_CATEGORY_LABELS,
  chartPoints,
  compareRuns,
  contextDetail,
  contextLabel,
  machineLine,
  placementLine,
  runSummary,
  runTitle,
  statusLabel,
  type BenchChartPoint,
} from "../../lib/bench";
import { drawReportCard, reportFilename, reportHtml } from "../../lib/bench-report";
import { saveDataUrl, saveText } from "../../lib/download";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import type { BenchAccuracy, BenchPlacementChoice, BenchPoint, BenchRun } from "../../service/bench-types";
import { BENCH_CONTEXTS, BENCH_PLACEMENTS, BENCH_PLACEMENT_LABELS } from "../../service/bench-types";
import type { LibraryModel } from "../../service/types";

/** What each placement means, in one line, under the choice. */
const PLACEMENT_NOTES: Readonly<Record<BenchPlacementChoice, string>> = {
  auto: "Where the model is loaded now. Load it differently on the Models screen to measure another placement.",
  gpu: "Every layer on the graphics card. The whole model has to fit in graphics memory.",
  split: "Half the layers on the graphics card, half on the processor.",
  cpu: "Every layer on the processor, in system memory.",
};

interface Progress {
  phase: string;
  text: string;
  done: number;
  total: number;
}

function categoryScores(accuracy: BenchAccuracy): Array<{ label: string; correct: number; asked: number }> {
  const groups = new Map<string, { correct: number; asked: number }>();
  for (const q of accuracy.questions) {
    const group = groups.get(q.category) ?? { correct: 0, asked: 0 };
    group.asked += 1;
    if (q.correct) group.correct += 1;
    groups.set(q.category, group);
  }
  return [...groups.entries()].map(([category, score]) => ({
    label: BENCH_CATEGORY_LABELS[category as keyof typeof BENCH_CATEGORY_LABELS] ?? category,
    ...score,
  }));
}

function changeTone(value: number | null, lowerIsBetter = false): string {
  if (value === null || value === 0) return "";
  const good = lowerIsBetter ? value < 0 : value > 0;
  return good ? "bench-change--good" : "bench-change--bad";
}

function percent(value: number | null): string {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

/** The history chart: one value per saved run at one depth, oldest first. */
function HistoryChart(props: { points: BenchChartPoint[]; label: string }): JSX.Element {
  const width = 640;
  const height = 190;
  const padX = 34;
  const padY = 24;
  const values = () => props.points.map((p) => p.value);
  const low = () => Math.min(...values());
  const high = () => Math.max(...values());
  const span = () => Math.max(0.001, high() - low());
  const x = (index: number) => (props.points.length === 1 ? width / 2 : padX + (index * (width - padX * 2)) / (props.points.length - 1));
  const y = (value: number) => height - padY - ((value - low()) / span()) * (height - padY * 2);
  const line = () => props.points.map((p, i) => `${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const first = () => props.points[0];
  const last = () => props.points[props.points.length - 1];
  return (
    <figure class="bench-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="bench-chart-title bench-chart-desc">
        <title id="bench-chart-title">{props.label} across saved runs</title>
        <desc id="bench-chart-desc">
          {props.points.length} measured runs, from {first()?.label ?? "none"} to {last()?.label ?? "none"}. The lowest value is {low().toFixed(1)} and the highest is {high().toFixed(1)}.
        </desc>
        <line class="bench-chart__axis" x1={padX} y1={height - padY} x2={width - padX} y2={height - padY} />
        <line class="bench-chart__axis" x1={padX} y1={padY} x2={padX} y2={height - padY} />
        <Show when={props.points.length > 1}>
          <polyline class="bench-chart__line" points={line()} />
        </Show>
        <For each={props.points}>
          {(point, index) => (
            <circle class="bench-chart__dot" cx={x(index())} cy={y(point.value)} r="4">
              <title>
                {point.label}: {point.value.toFixed(1)} ({point.provenance})
              </title>
            </circle>
          )}
        </For>
      </svg>
      <figcaption class="bench-chart__foot">
        <span>{first()?.label ?? ""}</span>
        <span class="hint">
          {props.points.length} run{props.points.length === 1 ? "" : "s"} · {props.label}
        </span>
        <span>{last()?.label ?? ""}</span>
      </figcaption>
    </figure>
  );
}

function PointsTable(props: { points: readonly BenchPoint[] }): JSX.Element {
  return (
    <div class="bench-table-wrap">
      <table class="bench-table">
        <caption class="sr-only">Speed at three prompt depths</caption>
        <thead>
          <tr>
            <th scope="col">Depth</th>
            <th scope="col">Prompt read</th>
            <th scope="col">Reading</th>
            <th scope="col">Writing</th>
            <th scope="col">First token</th>
            <th scope="col">Memory in use</th>
          </tr>
        </thead>
        <tbody>
          <For each={BENCH_CONTEXTS}>
            {(context) => {
              const point = () => props.points.find((p) => p.context === context) ?? null;
              return (
                <tr data-ran={point()?.ran ?? false}>
                  <th scope="row">
                    <span class="bench-table__depth">{contextLabel(context)}</span>
                    <span class="bench-table__detail">{contextDetail(context)}</span>
                  </th>
                  <Show
                    when={point()?.ran}
                    fallback={
                      <td colSpan="5" class="bench-table__note">
                        {point()?.note ?? "Not measured in this run."}
                      </td>
                    }
                  >
                    <td class="mono">{point()!.promptTokens.toLocaleString()} tokens</td>
                    <td class="mono">{point()!.prefillTps.toFixed(1)} tok/s</td>
                    <td class="mono">{point()!.decodeTps.toFixed(1)} tok/s</td>
                    <td class="mono">{Math.round(point()!.ttftMs).toLocaleString()} ms</td>
                    <td class="mono">{point()!.memoryUsedGiB.toFixed(2)} GiB</td>
                  </Show>
                </tr>
              );
            }}
          </For>
        </tbody>
      </table>
    </div>
  );
}

function AccuracyCard(props: { accuracy: BenchAccuracy }): JSX.Element {
  const scores = createMemo(() => categoryScores(props.accuracy));
  return (
    <section class="card bench-accuracy" aria-labelledby="bench-accuracy-title">
      <div class="bench-accuracy__head">
        <div>
          <h2 class="bench-card__title" id="bench-accuracy-title">
            Accuracy check
          </h2>
          <p class="hint">Twenty fixed short questions, graded automatically against one exact answer each.</p>
        </div>
        <p class="bench-score">
          {props.accuracy.correct}/{props.accuracy.asked} correct
        </p>
      </div>
      <ul class="bench-categories" role="list">
        <For each={scores()}>
          {(score) => (
            <li class="bench-category">
              <span>{score.label}</span>
              <strong class="mono">
                {score.correct}/{score.asked}
              </strong>
            </li>
          )}
        </For>
      </ul>
      <Show
        when={props.accuracy.questions.length}
        fallback={<p class="hint">This run stopped before the accuracy check asked any questions.</p>}
      >
        <details class="bench-questions">
          <summary>Show every question and answer</summary>
          <div class="bench-table-wrap">
            <table class="bench-table bench-table--questions">
              <thead>
                <tr>
                  <th scope="col">Question</th>
                  <th scope="col">Answer</th>
                  <th scope="col">Expected</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                <For each={props.accuracy.questions}>
                  {(question) => (
                    <tr>
                      <td>{question.question}</td>
                      <td>{question.answer || "No answer"}</td>
                      <td>{question.expected}</td>
                      <td>
                        <span class="chip" data-tone={question.correct ? "on" : "off"}>
                          {question.correct ? "Correct" : "Wrong"}
                        </span>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </details>
      </Show>
    </section>
  );
}

/**
 * Benchmark lab: measures one model at a short prompt, 4K and 16K, then asks twenty fixed
 * questions. Runs are saved with the model and machine; history, two-run comparison and report
 * cards (PNG and HTML) are built only from what was measured.
 */
export function BenchmarkLab(): JSX.Element {
  const service = useService();
  const [library] = createResource(() => service.listLibrary().catch(() => [] as LibraryModel[]));
  const models = createMemo(() => (library() ?? []).filter((m) => m.kind === "text" || m.kind === "vision"));
  const [modelId, setModelId] = createSignal("");
  createEffect(() => {
    if (!modelId() && models().length) setModelId(models()[0]!.id);
  });
  const model = () => models().find((m) => m.id === modelId()) ?? null;

  const [history, setHistory] = createSignal<BenchRun[]>([]);
  const [historyError, setHistoryError] = createSignal<string | null>(null);
  const [current, setCurrent] = createSignal<BenchRun | null>(null);
  const [active, setActive] = createSignal(false);
  const [progress, setProgress] = createSignal<Progress | null>(null);
  const [livePoints, setLivePoints] = createSignal<BenchPoint[]>([]);
  const [compareA, setCompareA] = createSignal("");
  const [compareB, setCompareB] = createSignal("");
  const [chartContext, setChartContext] = createSignal(0);
  const [chartMetric, setChartMetric] = createSignal<"decodeTps" | "prefillTps">("decodeTps");
  const [placement, setPlacement] = createSignal<BenchPlacementChoice>("auto");
  let abort: AbortController | null = null;
  let acceptedId = "";

  const accept = (run: BenchRun) => {
    if (run.id === acceptedId) return;
    acceptedId = run.id;
    const previous = history().find((r) => r.id !== run.id);
    setHistory((runs) => [run, ...runs.filter((r) => r.id !== run.id)].slice(0, 200));
    setCurrent(run);
    if (previous) setCompareA(previous.id);
    setCompareB(run.id);
    setProgress(null);
    setLivePoints([]);
  };

  onCleanup(() => {
    // Leaving the lab (or Monitor) stops a run instead of leaving heavy work hidden. The desktop
    // service turns this abort into bench_stop; no run means no stop reaches the engine.
    abort?.abort();
  });

  onMount(() => {
    void service
      .benchHistory()
      .then((runs) => {
        setHistory(runs);
        if (runs[0]) {
          setCurrent(runs[0]);
          setCompareB(runs[0].id);
          setCompareA(runs[1]?.id ?? "");
        }
      })
      .catch(() => setHistoryError("Saved benchmark runs could not be read. New runs still work; they are not saved until the folder can be read."));
  });

  const start = async () => {
    const chosen = model();
    if (!chosen || active()) return;
    acceptedId = "";
    const controller = new AbortController();
    abort = controller;
    setActive(true);
    setLivePoints([]);
    setProgress({ phase: "load", text: "Getting the model ready…", done: 0, total: BENCH_CONTEXTS.length + 20 });
    try {
      const run = await service.benchRun(
        chosen.id,
        placement(),
        (event) => {
          switch (event.type) {
            case "progress":
              setProgress({ phase: event.phase, text: event.text, done: event.done, total: event.total });
              break;
            case "point":
              setLivePoints((points) => [...points, event.point]);
              break;
            case "done":
              accept(event.run);
              break;
            case "error":
              setProgress({ phase: "error", text: event.message, done: 0, total: 1 });
              break;
            default:
              break;
          }
        },
        controller.signal,
      );
      accept(run);
    } catch (error) {
      if (!controller.signal.aborted) {
        const message = error instanceof Error ? error.message : "The benchmark could not finish.";
        notify("Benchmark failed", message, "error");
      }
    } finally {
      setActive(false);
      setProgress(null);
      if (abort === controller) abort = null;
    }
  };

  const stop = () => {
    abort?.abort();
    void service.benchStop().catch(() => {});
  };

  const comparison = createMemo(() => {
    const a = history().find((r) => r.id === compareA());
    const b = history().find((r) => r.id === compareB());
    return a && b ? { a, b, rows: compareRuns(a, b) } : null;
  });

  const chart = createMemo(() => chartPoints(history(), chartContext(), chartMetric()));
  const chartLabel = () => `${chartMetric() === "decodeTps" ? "Writing" : "Reading"} speed at ${contextLabel(chartContext())}`;
  const anyIllustrative = createMemo(() => history().some((r) => r.provenance === "illustrative"));

  const exportReport = (kind: "png" | "html") => {
    const run = current();
    if (!run) return;
    const name = reportFilename(run, kind);
    try {
      if (kind === "html") {
        if (saveText(name, "text/html", reportHtml(run))) notify("Report card saved", `${name}${run.provenance === "illustrative" ? " (illustrative data)" : ""}`, "success");
        else notify("Export failed", "The file could not be created. Nothing was saved.", "error");
      } else {
        const data = drawReportCard(run);
        if (saveDataUrl(name, data)) notify("Report card saved", `${name}${run.provenance === "illustrative" ? " (illustrative data)" : ""}`, "success");
        else notify("Export failed", "The image could not be created. Nothing was saved.", "error");
      }
    } catch {
      notify("Export failed", "The report card could not be drawn. Nothing was saved.", "error");
    }
  };

  const selectedModel = model;

  return (
    <div class="bench">
      <div class="bench-intro card">
        <div>
          <h2 class="bench-card__title">Benchmark this computer</h2>
          <p class="hint">
            One model is measured with a short prompt, a 4K prompt and a 16K prompt, then asked twenty fixed questions. A depth the model cannot hold is shown as not run, never estimated.
          </p>
        </div>
        <Show when={!__DESKTOP__}>
          <IllustrativeFlag />
        </Show>
      </div>

      <form
        class="card bench-setup"
        onSubmit={(event) => {
          event.preventDefault();
          void start();
        }}
      >
        <label class="bench-field bench-field--model">
          <span>Model</span>
          <select value={modelId()} onChange={(event) => setModelId(event.currentTarget.value)} disabled={active()}>
            <For each={models()}>
              {(m) => <option value={m.id}>{modelLabel(m)}</option>}
            </For>
          </select>
          <Show when={selectedModel()} fallback={<span class="hint">No supported text model was found. Add one on the Models screen.</span>}>
            {(m) => (
              <span class="hint">
                {[m().params, `${m().sizeGiB.toFixed(1)} GiB`, `holds ${m().contextMax.toLocaleString()} tokens`].filter(Boolean).join(" · ")}
              </span>
            )}
          </Show>
        </label>
        <label class="bench-field bench-field--placement">
          <span>Placement</span>
          <select value={placement()} onChange={(event) => setPlacement(event.currentTarget.value as BenchPlacementChoice)} disabled={active()}>
            <For each={BENCH_PLACEMENTS}>
              {(p) => (
                <option value={p}>{BENCH_PLACEMENT_LABELS[p]}</option>
              )}
            </For>
          </select>
          <span class="hint">{PLACEMENT_NOTES[placement()]}</span>
        </label>
        <div class="bench-setup__actions">
          <span class="bench-setup__spacer" aria-hidden="true" />
          <button class="btn btn--primary" type="submit" disabled={!selectedModel() || active()} data-press data-testid="bench-run">
            <Icon name="bolt" size={15} />
            {active() ? "Measuring…" : "Run benchmark"}
          </button>
          <Show when={active()}>
            <button class="btn btn--secondary" type="button" onClick={stop} data-press data-testid="bench-stop">
              <Icon name="stop" size={14} />
              Stop
            </button>
          </Show>
        </div>
        <p class="hint bench-setup__note">The run loads the model once, at the deepest depth it measures, and keeps it loaded. Keep other heavy work closed while it measures.</p>
      </form>

      <Show when={progress()}>
        {(state) => (
          <section class="card bench-progress" role="status" aria-live="polite" data-testid="bench-progress">
            <div class="bench-progress__head">
              <div>
                <h2 class="bench-card__title">Measuring</h2>
                <p class="hint">{state().text}</p>
              </div>
              <span class="mono bench-progress__count">
                {state().done}/{state().total}
              </span>
            </div>
            <progress class="bench-progress__bar" max={Math.max(1, state().total)} value={state().done} aria-label="Benchmark progress">
              {state().done} of {state().total}
            </progress>
            <Show when={livePoints().length}>
              <ul class="bench-progress__points" role="list">
                <For each={livePoints()}>
                  {(point) => (
                    <li>
                      <span>{contextLabel(point.context)}</span>
                      <span class="mono">{point.decodeTps.toFixed(1)} tok/s writing</span>
                    </li>
                  )}
                </For>
              </ul>
            </Show>
          </section>
        )}
      </Show>

      <Show
        when={current()}
        fallback={
          <Show when={!active()}>
            <p class="card bench-empty hint" data-testid="bench-empty">
              No benchmark runs yet. Choose a model and run the benchmark to see measured speed, accuracy and a report card.
            </p>
          </Show>
        }
      >
        {(run) => (
          <section class="card bench-run" aria-labelledby="bench-run-title" data-testid="bench-run-result">
            <div class="bench-run__head">
              <div class="bench-run__title">
                <h2 class="bench-card__title" id="bench-run-title">
                  {run().model.name}
                </h2>
                <p class="hint">
                  {run().model.quant} · {run().model.params || "size not reported"} · {new Date(run().at).toLocaleString()}
                </p>
              </div>
              <div class="bench-run__badges">
                <Show when={run().provenance === "illustrative"}>
                  <IllustrativeFlag />
                </Show>
                <span class="chip" data-tone={run().status === "done" ? "on" : run().status === "stopped" ? "off" : "active"}>
                  {statusLabel(run().status)}
                </span>
              </div>
            </div>
            <p class="hint bench-run__machine">{machineLine(run())}</p>
            <p class="hint bench-run__placement" data-testid="bench-placement">
              {placementLine(run())}
            </p>
            <div class="bench-run__report">
              <div class="bench-run__summary">
                <Icon name="activity" size={17} />
                <span>{runSummary(run())}</span>
              </div>
              <div class="bench-run__actions">
                <button class="btn btn--secondary btn--sm" type="button" onClick={() => exportReport("png")} data-press data-testid="bench-export-png">
                  <Icon name="download" size={14} />
                  PNG report
                </button>
                <button class="btn btn--secondary btn--sm" type="button" onClick={() => exportReport("html")} data-press data-testid="bench-export-html">
                  <Icon name="download" size={14} />
                  HTML report
                </button>
              </div>
            </div>
            <PointsTable points={run().points} />
            <AccuracyCard accuracy={run().accuracy} />
          </section>
        )}
      </Show>

      <section class="card bench-history" aria-labelledby="bench-history-title">
        <div class="bench-card__head">
          <div>
            <h2 class="bench-card__title" id="bench-history-title">
              History
            </h2>
            <p class="hint">One point per saved run, oldest to newest. Compare runs from the same machine and the same model family.</p>
          </div>
          <div class="bench-history__controls">
            <label class="bench-field bench-field--compact">
              <span>Depth</span>
              <select value={String(chartContext())} onChange={(event) => setChartContext(Number(event.currentTarget.value))}>
                <For each={BENCH_CONTEXTS}>{(context) => <option value={context}>{contextLabel(context)}</option>}</For>
              </select>
            </label>
            <label class="bench-field bench-field--compact">
              <span>Metric</span>
              <select value={chartMetric()} onChange={(event) => setChartMetric(event.currentTarget.value as "decodeTps" | "prefillTps")}>
                <option value="decodeTps">Writing speed</option>
                <option value="prefillTps">Reading speed</option>
              </select>
            </label>
          </div>
        </div>
        <Show when={historyError()}>
          <p class="notice notice--inline" role="alert">
            {historyError()}
          </p>
        </Show>
        <Show
          when={chart().length}
          fallback={<p class="hint">No measured point at {contextLabel(chartContext())} yet. Run a benchmark, or pick another depth.</p>}
        >
          <HistoryChart points={chart()} label={`${chartLabel()} (tok/s)`} />
        </Show>
        <Show when={anyIllustrative()}>
          <p class="hint">Some points are from the web preview and are illustrative, not measurements.</p>
        </Show>
      </section>

      <section class="card bench-compare" aria-labelledby="bench-compare-title">
        <h2 class="bench-card__title" id="bench-compare-title">
          Compare two runs
        </h2>
        <Show when={history().length >= 2} fallback={<p class="hint">Two saved runs are needed for a comparison. Run the benchmark again, or save another run.</p>}>
          <div class="bench-compare__pickers">
            <label class="bench-field">
              <span>Run A</span>
              <select value={compareA()} onChange={(event) => setCompareA(event.currentTarget.value)}>
                <For each={history()}>{(run) => <option value={run.id}>{runTitle(run)}</option>}</For>
              </select>
            </label>
            <label class="bench-field">
              <span>Run B</span>
              <select value={compareB()} onChange={(event) => setCompareB(event.currentTarget.value)}>
                <For each={history()}>{(run) => <option value={run.id}>{runTitle(run)}</option>}</For>
              </select>
            </label>
          </div>
          <Show when={comparison()} fallback={<p class="hint">Choose two different saved runs.</p>}>
            {(pair) => (
              <div class="bench-table-wrap">
                <table class="bench-table bench-table--compare">
                  <caption class="sr-only">
                    Run B compared with run A; a positive speed change means run B is faster.
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Depth</th>
                      <th scope="col">Reading A → B</th>
                      <th scope="col">Writing A → B</th>
                      <th scope="col">First token A → B</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={pair().rows}>
                      {(row) => {
                        const both = () => row.a?.ran && row.b?.ran;
                        return (
                          <tr>
                            <th scope="row">{contextLabel(row.context)}</th>
                            <td class="mono">
                              <Show when={both()} fallback={<span class="hint">Not run</span>}>
                                {row.a!.prefillTps.toFixed(1)} → {row.b!.prefillTps.toFixed(1)}
                                <span class={`bench-change ${changeTone(row.prefillPct)}`}>{percent(row.prefillPct)}</span>
                              </Show>
                            </td>
                            <td class="mono">
                              <Show when={both()} fallback={<span class="hint">Not run</span>}>
                                {row.a!.decodeTps.toFixed(1)} → {row.b!.decodeTps.toFixed(1)}
                                <span class={`bench-change ${changeTone(row.decodePct)}`}>{percent(row.decodePct)}</span>
                              </Show>
                            </td>
                            <td class="mono">
                              <Show when={both()} fallback={<span class="hint">Not run</span>}>
                                {Math.round(row.a!.ttftMs).toLocaleString()} → {Math.round(row.b!.ttftMs).toLocaleString()} ms
                                <span class={`bench-change ${changeTone(row.ttftPct, true)}`}>{percent(row.ttftPct)}</span>
                              </Show>
                            </td>
                          </tr>
                        );
                      }}
                    </For>
                  </tbody>
                </table>
              </div>
            )}
          </Show>
        </Show>
      </section>

      <section class="card bench-saved" aria-labelledby="bench-saved-title">
        <h2 class="bench-card__title" id="bench-saved-title">
          Saved runs
        </h2>
        <Show when={history().length} fallback={<p class="hint">Runs are saved on this computer and appear here after the first one.</p>}>
          <ul class="bench-saved__list" role="list">
            <For each={history().slice(0, 8)}>
              {(run) => (
                <li>
                  <button
                    class="bench-saved__run"
                    type="button"
                    aria-pressed={current()?.id === run.id}
                    onClick={() => {
                      setCurrent(run);
                      setCompareB(run.id);
                    }}
                    data-testid="bench-saved-run"
                  >
                    <span class="bench-saved__name">{run.model.name}</span>
                    <span class="hint">
                      {new Date(run.at).toLocaleString()} · {runSummary(run)}
                    </span>
                  </button>
                </li>
              )}
            </For>
          </ul>
        </Show>
        <Show when={history().length > 8}>
          <p class="hint">Showing the eight most recent runs. The history chart and comparison include every saved run.</p>
        </Show>
      </section>
    </div>
  );
}
