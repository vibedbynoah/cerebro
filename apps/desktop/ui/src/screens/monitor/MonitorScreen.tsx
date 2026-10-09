import "../../styles/monitor.css";
import { createEffect, createSignal, createUniqueId, For, on, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../../components/icons";
import { IllustrativeFlag, ScreenHeader, Section } from "../../components/Screen";
import { saveText } from "../../lib/download";
import { exportFilename, exportPayload, HISTORY_RANGES, isHistoryRange, rangeDef, seriesRows, toCsv, totalsLine } from "../../lib/monitor";
import { readPref, writePref } from "../../lib/storage";
import { notify } from "../../lib/toast";
import { useService } from "../../service/context";
import { useMetrics } from "../../service/metrics";
import type { HistoryRange, MetricsSeries } from "../../service/monitor-types";
import { Breakdown } from "./Breakdown";
import { ActiveAlerts, AlertsCard, CacheCard, MetricGrid, ModelCard, StateCard } from "./cards";
import { Inspector } from "./Inspector";

const RANGE_KEY = "cerebro-monitor-range";

/**
 * The Monitor: the model and the machine, live. Reads the app-wide metrics feed (one poll a
 * second, shared with the header pill and the alerts), fetches longer history for the chosen range
 * on the range's own step, and exports that range as JSON or CSV. Loaded as its own chunk.
 */
export function MonitorScreen(): JSX.Element {
  const feed = useMetrics();
  const service = useService();
  const id = createUniqueId();
  const stored = readPref(RANGE_KEY);
  const [range, setRange] = createSignal<HistoryRange>(isHistoryRange(stored) ? stored : "1m");
  const [long, setLong] = createSignal<MetricsSeries | null>(null);
  const [selected, setSelected] = createSignal<string | null>(null);

  // Longer ranges: fetched when chosen, then again each time a new bucket closes (at most every 5 s).
  createEffect(
    on(range, (r) => {
      writePref(RANGE_KEY, r === "1m" ? null : r);
      setLong(null);
      if (r === "1m") return;
      let alive = true;
      const load = () =>
        service
          .getMetricsHistory(r)
          .then((s) => alive && setLong(s))
          .catch(() => alive && notify("History could not be loaded", "The last minute is still shown live.", "error"));
      load();
      const timer = setInterval(load, Math.max(5, rangeDef(r).stepS) * 1000);
      onCleanup(() => {
        alive = false;
        clearInterval(timer);
      });
    }),
  );

  const series = () => (range() === "1m" ? (feed.snapshot()?.history ?? null) : long());

  const exportAs = (ext: "json" | "csv") => {
    const s = series();
    const snap = feed.snapshot();
    if (!s || !snap) {
      notify("Nothing to export yet", "Wait for the history to load, then try again.", "info");
      return;
    }
    const at = Date.now();
    const name = exportFilename(range(), ext, at);
    const text = ext === "json" ? JSON.stringify(exportPayload(s, range(), snap, at), null, 2) : toCsv(seriesRows(s));
    if (saveText(name, ext === "json" ? "application/json" : "text/csv", text)) notify(`Exported ${rangeDef(range()).long}`, `${name}${snap.provenance === "illustrative" ? " (illustrative data)" : ""}`, "success");
    else notify("Export failed", "The file could not be created. Nothing was saved.", "error");
  };

  const showRequest = (rid: string) => {
    setSelected(rid);
    requestAnimationFrame(() => document.querySelector<HTMLElement>(`button.mon-req[data-request="${CSS.escape(rid)}"]`)?.focus());
  };

  const actions = (
    <div class="mon-actions">
      <fieldset class="segmented mon-range">
        <legend class="sr-only">History range</legend>
        <For each={HISTORY_RANGES}>
          {(r) => (
            <label class="segmented__option">
              <input type="radio" name={`${id}-range`} value={r.id} checked={range() === r.id} onChange={() => setRange(r.id)} />
              {r.label}
            </label>
          )}
        </For>
      </fieldset>
      <div class="mon-export" role="group" aria-label="Export the chosen range">
        <button type="button" class="btn btn--secondary mon-export__btn" data-press onClick={() => exportAs("json")} aria-label={`Export ${rangeDef(range()).long} as JSON`}>
          <Icon name="download" size={14} />
          JSON
        </button>
        <button type="button" class="btn btn--secondary mon-export__btn" data-press onClick={() => exportAs("csv")} aria-label={`Export ${rangeDef(range()).long} as CSV`}>
          <Icon name="download" size={14} />
          CSV
        </button>
      </div>
    </div>
  );

  return (
    <div class="screen monitor">
      <ScreenHeader route="monitor" actions={actions} />
      <Show
        when={feed.snapshot()}
        fallback={
          <Show
            when={!feed.reachable()}
            fallback={
              <p class="sr-only" role="status">
                Connecting to the engine…
              </p>
            }
          >
            <div class="notice" role="alert">
              <Icon name="alert" size={18} />
              <p>The engine is not answering. Cerebro keeps trying every second; nothing else is affected.</p>
            </div>
          </Show>
        }
      >
        {(snap) => (
          <>
            <Show when={!feed.reachable()}>
              <div class="notice" role="alert">
                <Icon name="alert" size={18} />
                <p>The engine stopped answering. The numbers below are the last ones it sent; Cerebro keeps trying every second.</p>
              </div>
            </Show>
            <div class="mon-summary">
              <p class="mon-totals">{totalsLine(snap().totals) || "No requests since the engine started."}</p>
              <Show when={snap().provenance === "illustrative"}>
                <IllustrativeFlag />
              </Show>
            </div>
            <ActiveAlerts alerts={feed.alerts()} illustrative={snap().provenance === "illustrative"} />
            <div class="mon-top">
              <StateCard snap={snap()} onShowRequest={showRequest} />
              <ModelCard snap={snap()} />
            </div>
            <section class="mon-section" aria-labelledby="mon-live-title">
              <div class="mon-section__head">
                <h2 class="section__title" id="mon-live-title">
                  Live metrics
                </h2>
                <p class="section__desc">
                  {range() === "1m" ? "One sample a second." : `Averaged over ${rangeDef(range()).stepS >= 60 ? `${rangeDef(range()).stepS / 60} minutes` : `${rangeDef(range()).stepS} seconds`} per point.`} Focus a chart and use the arrow keys to read its samples.
                </p>
              </div>
              <Show when={series()} fallback={<div class="mon-grid mon-grid--pending" aria-hidden="true" />}>
                {(s) => <MetricGrid snap={snap()} series={s()} range={rangeDef(range())} alerts={feed.alerts()} />}
              </Show>
            </section>
            <div class="mon-pair-row">
              <CacheCard snap={snap()} />
              <AlertsCard alerts={feed.alerts()} />
            </div>
            <Section id="mon-breakdown" title="By session and by tool" description="Where tokens and time went across every request the engine keeps, grouped by the client session or by the tool the model called.">
              <Breakdown requests={snap().requests} illustrative={snap().provenance === "illustrative"} />
            </Section>
            <Section id="mon-requests" title="Requests" description="Every call to the API, live. Captures stay on this machine.">
              <Inspector requests={snap().requests} selected={selected()} onSelect={setSelected} at={snap().at} illustrative={snap().provenance === "illustrative"} />
            </Section>
          </>
        )}
      </Show>
    </div>
  );
}
