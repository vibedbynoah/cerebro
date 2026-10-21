import { createEffect, For, Index, Show, type JSX } from "solid-js";
import { setBarValue } from "@design/motion";
import { Icon, type IconName } from "../../components/icons";
import { Term } from "../../components/Term";
import { hrefFor } from "../../lib/routes";
import {
  agoLabel,
  clockTime,
  compact,
  DASH,
  num,
  pct,
  seconds,
  stateView,
  throughput,
  type AlertTracker,
  type RangeDef,
} from "../../lib/monitor";
import type { MetricsSeries, MetricsSnapshot, ReleaseChannel, SeriesKey } from "../../service/monitor-types";
import { Sparkline } from "./Sparkline";

/**
 * The Monitor's cards. Each reads the latest snapshot through props, so a new sample rewrites text
 * nodes and path data in place; no card is rebuilt per tick.
 */

const CHANNEL_LABEL: Readonly<Record<ReleaseChannel, string>> = { stable: "Stable", beta: "Beta", experimental: "Experimental" };

/** The release channel a feature ships in (Stable, Beta, Experimental), as a small outlined tag. */
export function Channel(props: { channel: ReleaseChannel }): JSX.Element {
  return (
    <span class="mon-channel" data-channel={props.channel}>
      {CHANNEL_LABEL[props.channel]}
    </span>
  );
}

/** A number and its unit as one unit of type (DESIGN.md). */
export function Readout(props: { value: string; unit?: string; of?: string }): JSX.Element {
  return (
    <>
      <span class="mon-num">{props.value}</span>
      <Show when={props.of}>
        <span class="mon-of">/{props.of}</span>
      </Show>
      <Show when={props.unit && props.value !== DASH}>
        <span class="unit">{props.unit}</span>
      </Show>
    </>
  );
}

// ---- State ----------------------------------------------------------------------------------

const BADGES = [
  { id: "idle", label: "Idle" },
  { id: "loading", label: "Loading" },
  { id: "reading", label: "Reading prompt" },
  { id: "generating", label: "Generating" },
] as const;

/** What the engine is doing now: state badges, prompt-read or written-of-max progress, first token,
 *  context fill, and a link to the running request in the inspector. */
export function StateCard(props: { snap: MetricsSnapshot; onShowRequest: (id: string) => void }): JSX.Element {
  let bar: HTMLSpanElement | undefined;
  const live = () => props.snap.live;
  const view = () => stateView(live(), props.snap.requests.find((r) => r.state === "completed" || r.state === "cancelled"));
  createEffect(() => {
    const p = view().progress;
    if (bar) setBarValue(bar, p / 100);
  });
  const model = () => props.snap.engine.model;
  const ctxPct = () => {
    const m = model();
    return m ? pct(m.contextUsed, m.contextMax) : 0;
  };
  const running = () => props.snap.requests.find((r) => r.id === live().requestId);
  return (
    <article class="mon-panel mon-state reveal" data-reveal aria-labelledby="mon-state-title" data-state={live().state}>
      <div class="mon-state__top">
        <h2 class="mon-panel__title" id="mon-state-title">
          Model state
        </h2>
        <ul class="mon-badges" aria-label="Engine states">
          <For each={BADGES}>
            {(b) => (
              <li class="mon-badge" classList={{ "is-on": live().state === b.id }} aria-current={live().state === b.id ? "true" : undefined}>
                {b.label}
              </li>
            )}
          </For>
          <li class="mon-badge mon-badge--queue" classList={{ "is-on": live().queued > 0 }}>
            {live().queued} queued
          </li>
        </ul>
      </div>
      <p class="mon-state__label">{view().label}</p>
      <p class="mon-state__detail">{view().detail}</p>
      <span
        class="mon-track"
        role="progressbar"
        aria-label={view().progressLabel}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(view().progress)}
        data-tone={live().state === "reading" ? "reading" : "writing"}
      >
        <span class="bar mon-bar" ref={bar} />
      </span>
      <dl class="mon-facts mon-facts--row">
        <div>
          <dt>
            <Term k="firstToken">First token</Term>
          </dt>
          <dd class="mono">{seconds(live().firstTokenS ?? running()?.firstTokenS ?? props.snap.requests[0]?.firstTokenS ?? null)}</dd>
        </div>
        <div>
          <dt>
            <Term k="context">Context</Term>
          </dt>
          <dd class="mono">
            <Show when={model()} fallback={DASH}>
              {(m) => (
                <>
                  {compact(m().contextUsed)} of {compact(m().contextMax)} · {num(ctxPct())}%
                </>
              )}
            </Show>
          </dd>
        </div>
        <div>
          <dt>Request</dt>
          <dd>
            <Show when={running()} fallback={<span class="muted">None running</span>}>
              {(r) => (
                <button type="button" class="text-link text-link--btn mono" onClick={() => props.onShowRequest(r().id)}>
                  {r().id}
                </button>
              )}
            </Show>
          </dd>
        </div>
      </dl>
    </article>
  );
}

// ---- Model ----------------------------------------------------------------------------------

/** The loaded model: quant, backend, context, placement, active experts, speculation and the tuned
 *  baseline that speed alerts are judged against. */
export function ModelCard(props: { snap: MetricsSnapshot }): JSX.Element {
  const e = () => props.snap.engine;
  return (
    <article class="mon-panel mon-model reveal" data-reveal aria-labelledby="mon-model-title">
      <h2 class="mon-panel__title" id="mon-model-title">
        Model
      </h2>
      <Show
        when={e().model}
        fallback={
          <p class="mon-empty">
            No model is loaded. <a class="text-link" href={hrefFor("models")}>Load one from Models</a>.
          </p>
        }
      >
        {(m) => (
          <>
            <p class="mon-model__name">{m().name}</p>
            <p class="mon-model__chips">
              <span class="chip" data-tone="neutral">
                {m().quant}
              </span>
              <span class="chip" data-tone="off">
                {m().params}
              </span>
              <span class="chip" data-tone="off">
                {m().architecture === "moe" ? "Mixture of experts" : "Dense"}
              </span>
            </p>
            <dl class="mon-facts">
              <div>
                <dt>
                  <Term k="backend">Backend</Term>
                </dt>
                <dd>
                  {e().backend} <span class="mono">{e().build}</span> · <span class="mono">{e().address}</span> (<Term k="loopback" />)
                </dd>
              </div>
              <div>
                <dt>Context</dt>
                <dd>
                  <span class="mono">{num(m().contextMax)}</span> tokens · <Term k="kvCache">KV cache</Term> <span class="mono">{m().kvType}</span>
                </dd>
              </div>
              <div>
                <dt>
                  <Term k="offload">Placement</Term>
                </dt>
                <dd>
                  <span class="mono">
                    {m().gpuLayers} of {m().layers}
                  </span>{" "}
                  layers on the GPU
                  <Show when={m().expertLayersOnCpu > 0}>
                    ; experts of <span class="mono">{m().expertLayersOnCpu}</span> layers in RAM, computed on the CPU
                  </Show>
                </dd>
              </div>
              <Show when={e().moe}>
                {(moe) => (
                  <div>
                    <dt>
                      <Term k="moe">Active experts</Term>
                    </dt>
                    <dd>
                      <span class="mono">
                        {moe().expertsPerToken} of {moe().expertsTotal}
                      </span>{" "}
                      per token (<span class="mono">{num(pct(moe().expertsPerToken, moe().expertsTotal), 1)}%</span>)
                      <Show when={moe().cacheSlots > 0}>
                        {" "}
                        · <span class="mono">{num(moe().cacheSlots)}</span> in the <Term k="expertCache" /> <Channel channel={moe().channel} />
                      </Show>
                    </dd>
                  </div>
                )}
              </Show>
              <div>
                <dt>
                  <Term k="speculative">Speculation</Term>
                </dt>
                <dd>
                  <Show when={e().speculation} fallback={<>Off for this model</>}>
                    {(sp) => (
                      <>
                        {sp().method} · <span class="mono">{num(sp().acceptancePct)}%</span> accepted · <span class="mono">{num(sp().tokensPerPass, 1)}</span> tokens per pass{" "}
                        <Channel channel={sp().channel} />
                      </>
                    )}
                  </Show>
                </dd>
              </div>
              <div>
                <dt>Tuned baseline</dt>
                <dd>
                  <Show when={e().baseline} fallback={<>Not tuned yet</>}>
                    {(b) => (
                      <>
                        Writing <span class="mono">{num(b().decodeTps, 1)}</span> tok/s at <span class="mono">{compact(b().depth)}</span> context, reading <span class="mono">{num(b().prefillTps)}</span> tok/s · {b().measuredAt}
                      </>
                    )}
                  </Show>
                </dd>
              </div>
            </dl>
          </>
        )}
      </Show>
    </article>
  );
}

// ---- Metric cards ---------------------------------------------------------------------------

function MetricCard(props: {
  metric: string;
  icon: IconName;
  label: JSX.Element;
  value: JSX.Element;
  sub: JSX.Element;
  badge?: JSX.Element;
  wide?: boolean;
  alert?: boolean;
  children?: JSX.Element;
}): JSX.Element {
  return (
    <article class="mon-card reveal" data-reveal classList={{ "mon-card--wide": props.wide === true, "is-alert": props.alert === true }} data-metric={props.metric}>
      <header class="mon-card__head">
        <Icon name={props.icon} size={16} />
        <h3 class="mon-card__label">{props.label}</h3>
        {props.badge}
      </header>
      <div class="mon-card__value">{props.value}</div>
      <p class="mon-card__sub">{props.sub}</p>
      {props.children}
    </article>
  );
}

const fmt1 = (v: number) => num(v, 1);
const fmt0 = (v: number) => num(v);

/** The live metric cards with sparklines for the chosen range; per-GPU values on multi-GPU machines. */
export function MetricGrid(props: { snap: MetricsSnapshot; series: MetricsSeries | null; range: RangeDef; alerts: AlertTracker }): JSX.Element {
  const hw = () => props.snap.hardware;
  const st = () => props.snap.hardwareStatic;
  const live = () => props.snap.live;
  const e = () => props.snap.engine;
  const vals = (k: SeriesKey) => props.series?.series[k] ?? [];
  const step = () => props.series?.stepS ?? 1;
  const end = () => props.series?.end ?? props.snap.at;
  const rangeLabel = () => `Last ${props.range.label}`;
  const multi = () => hw().gpus.length > 1;
  const perGpu = (f: (g: MetricsSnapshot["hardware"]["gpus"][number]) => string) => hw().gpus.map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const lastDone = () => props.snap.requests.find((r) => r.state === "completed");

  const decodeNow = () => live().decodeTps ?? lastDone()?.decodeTps ?? null;
  const prefillNow = () => (live().state !== "idle" ? live().prefillTps : (lastDone()?.prefillTps ?? null));
  const spark = (key: string, label: string, k: SeriesKey, format: (v: number) => string, top?: number, threshold?: { value: number; label: string } | null) => (
    <Sparkline
      label={label}
      lines={[{ key, label, values: vals(k), tone: "accent", area: true, format, ...(top !== undefined ? { top } : {}) }]}
      stepS={step()}
      end={end()}
      threshold={threshold ?? null}
      rangeLabel={rangeLabel()}
    />
  );

  const vram = () => hw().vram;
  const vramParts = () => {
    const v = vram();
    if (!v) return [];
    return [
      { key: "weights", label: "Weights", value: v.weights },
      { key: "kv", label: "KV cache", value: v.kv },
      { key: "expert", label: "Expert cache", value: v.expertCache },
      { key: "scratch", label: "Scratch", value: v.scratch },
      { key: "other", label: "Other programs", value: v.other },
    ];
  };
  const vramOffsets = () => {
    let x = 0;
    return vramParts().map((p) => {
      const w = (100 * p.value) / Math.max(0.001, hw().vramTotalGiB);
      const out = { ...p, x, w };
      x += w;
      return out;
    });
  };
  const kvFill = () => {
    const m = e().model;
    return m ? pct(m.contextUsed, m.contextMax) : null;
  };

  return (
    <div class="mon-grid">
      <MetricCard
        metric="speed"
        icon="bolt"
        wide
        alert={props.alerts.active.regression !== undefined}
        label={
          <>
            <Term k="decode">Writing</Term> and <Term k="prefill">reading</Term> speed
          </>
        }
        value={
          <div class="mon-pair">
            <div>
              <Readout value={num(decodeNow(), 1)} unit="tok/s" />
              <span class="mon-pair__cap">
                <span class="mon-key" data-tone="accent" aria-hidden="true" />
                {live().state === "generating" ? "Writing now" : "Writing, last request"}
              </span>
            </div>
            <div>
              <Readout value={num(prefillNow())} unit="tok/s" />
              <span class="mon-pair__cap">
                <span class="mon-key" data-tone="muted" aria-hidden="true" />
                {live().state === "reading" ? "Reading now" : live().state === "generating" ? "Reading, this request" : "Reading, last request"}
              </span>
            </div>
          </div>
        }
        sub={
          <Show when={e().baseline} fallback={<>Each line on its own scale.</>}>
            {(b) => (
              <>
                Tuned baseline: writing <span class="mono">{num(b().decodeTps, 1)}</span> tok/s at <span class="mono">{compact(b().depth)}</span> context. Each line on its own scale.
              </>
            )}
          </Show>
        }
      >
        <Sparkline
          label="Writing and reading speed"
          lines={[
            { key: "decode", label: "Writing", values: vals("decodeTps"), tone: "accent", area: true, format: (v) => `${num(v, 1)} tok/s` },
            { key: "prefill", label: "Reading", values: vals("prefillTps"), tone: "muted", dashed: true, format: (v) => `${num(v)} tok/s` },
          ]}
          stepS={step()}
          end={end()}
          threshold={null}
          rangeLabel={rangeLabel()}
        />
      </MetricCard>

      <MetricCard
        metric="vram"
        icon="chip"
        wide
        alert={props.alerts.active.vram !== undefined}
        label={<Term k="vram" />}
        value={<Readout value={num(hw().vramUsedGiB, 1)} of={num(hw().vramTotalGiB, 1)} unit="GiB" />}
        sub={
          <>
            <span class="mono">{num(pct(hw().vramUsedGiB ?? 0, hw().vramTotalGiB))}%</span> in use
            <Show when={kvFill() !== null}>
              {" "}
              · <Term k="kvCache">KV cache</Term> <span class="mono">{num(kvFill())}%</span> full
            </Show>
            <Show when={multi()}> · {perGpu((g) => `${num(g.memUsedGiB, 1)} GiB`)}</Show>
          </>
        }
      >
        <Show when={vram()}>
          <svg class="mon-split" viewBox="0 0 100 8" preserveAspectRatio="none" role="img" aria-label={`Graphics memory split: ${vramParts().map((p) => `${p.label} ${num(p.value, 2)} GiB`).join(", ")}, of ${num(hw().vramTotalGiB, 1)} GiB`}>
            <Index each={vramOffsets()}>{(p) => <rect class="mon-split__part" data-part={p().key} x={p().x} y="0" width={Math.max(0, p().w)} height="8" />}</Index>
          </svg>
          <ul class="mon-legend">
            <Index each={vramParts()}>
              {(p) => (
                <li>
                  <span class="mon-swatch" data-part={p().key} aria-hidden="true" />
                  {p().key === "expert" ? <Term k="expertCache">Expert cache</Term> : p().label}
                  <span class="mono">{num(p().value, 2)}</span>
                </li>
              )}
            </Index>
          </ul>
        </Show>
        {spark("vram", "Graphics memory in use", "vramUsedGiB", (v) => `${num(v, 2)} GiB`, hw().vramTotalGiB, { value: hw().vramTotalGiB * 0.95, label: "Pressure at 95%" })}
      </MetricCard>

      <MetricCard
        metric="gpu"
        icon="gpu"
        label={<Term k="gpu">GPU load</Term>}
        value={<Readout value={num(hw().gpuUtil)} unit="%" />}
        sub={multi() ? perGpu((g) => `${num(g.util)}%`) : (st().gpuName ?? "No GPU detected")}
      >
        {spark("gpu", "GPU load", "gpuUtil", (v) => `${fmt0(v)}%`, 100)}
      </MetricCard>

      <MetricCard
        metric="temp"
        icon="thermometer"
        alert={props.alerts.active.thermal !== undefined}
        label="GPU temperature"
        value={<Readout value={num(hw().gpuTempC)} unit="°C" />}
        sub={
          multi() ? (
            perGpu((g) => `${num(g.tempC)} °C`)
          ) : (
            <>
              <Show when={hw().throttle.includes("thermal")} fallback={<>Driver slows the card at <span class="mono">{num(st().gpuThrottleTempC)}</span> °C</>}>
                <Term k="throttling">Throttling</Term> now
              </Show>
            </>
          )
        }
      >
        {spark("temp", "GPU temperature", "gpuTempC", (v) => `${fmt1(v)} °C`, Math.max(95, (st().gpuThrottleTempC ?? 87) + 8), st().gpuThrottleTempC ? { value: st().gpuThrottleTempC ?? 0, label: `Throttles at ${num(st().gpuThrottleTempC)} °C` } : null)}
      </MetricCard>

      <MetricCard
        metric="power"
        icon="bolt"
        label="GPU power"
        value={<Readout value={num(hw().gpuPowerW)} unit="W" />}
        sub={
          multi() ? (
            perGpu((g) => `${num(g.powerW)} W`)
          ) : (
            <>
              of <span class="mono">{num(hw().gpuPowerLimitW)}</span> W limit · <span class="mono">{num(pct(hw().gpuPowerW ?? 0, hw().gpuPowerLimitW ?? 0))}%</span>
              <Show when={hw().throttle.includes("power")}> · at the limit</Show>
            </>
          )
        }
      >
        {spark("power", "GPU power", "gpuPowerW", (v) => `${fmt0(v)} W`, (hw().gpuPowerLimitW ?? 100) * 1.06, hw().gpuPowerLimitW ? { value: hw().gpuPowerLimitW ?? 0, label: "Line: power limit" } : null)}
      </MetricCard>

      <MetricCard
        metric="pcie"
        icon="link"
        label={<Term k="pcie" />}
        value={<Readout value={throughput(hw().pcieRxMBs)[0]} unit={throughput(hw().pcieRxMBs)[1]} />}
        sub={
          multi() ? (
            perGpu((g) => throughput(g.pcieRxMBs).join(" "))
          ) : (
            <>
              To the GPU · <span class="mono">Gen{hw().pcieGen ?? DASH}</span>
              <Show when={hw().pcieGenMax && hw().pcieGen !== hw().pcieGenMax}>
                {" "}
                now, <span class="mono">Gen{hw().pcieGenMax}</span> under load
              </Show>{" "}
              <span class="mono">x{hw().pcieWidth ?? DASH}</span>
            </>
          )
        }
      >
        {spark("pcie", "PCIe traffic to the GPU", "pcieRxMBs", (v) => throughput(v).join(" "))}
      </MetricCard>

      <MetricCard metric="cpu" icon="chip" label={<Term k="cpu">CPU</Term>} value={<Readout value={num(hw().cpu)} unit="%" />} sub={`${st().cores} cores · ${st().threads} threads`}>
        {spark("cpu", "CPU load", "cpu", (v) => `${fmt0(v)}%`, 100)}
      </MetricCard>

      <MetricCard
        metric="ram"
        icon="memory"
        label={<Term k="ram" />}
        value={<Readout value={num(hw().ramUsedGiB, 1)} of={num(hw().ramTotalGiB)} unit="GiB" />}
        sub={
          <>
            <span class="mono">{num(hw().ramPinnedGiB, 1)}</span> GiB held for model weights
          </>
        }
      >
        {spark("ram", "RAM in use", "ramUsedGiB", (v) => `${num(v, 1)} GiB`, hw().ramTotalGiB)}
      </MetricCard>

      <MetricCard
        metric="disk"
        icon="disk"
        label="Disk read"
        value={<Readout value={throughput(hw().diskReadMBs)[0]} unit={throughput(hw().diskReadMBs)[1]} />}
        sub={
          <>
            Write <span class="mono">{throughput(hw().diskWriteMBs).join(" ")}</span>
          </>
        }
      >
        {spark("disk", "Disk read", "diskReadMBs", (v) => throughput(v).join(" "))}
      </MetricCard>

      <MetricCard
        metric="experts"
        icon="sliders"
        label={
          <>
            <Term k="expertCache">Experts</Term> and <Term k="acceptance">drafts</Term>
          </>
        }
        value={
          <div class="mon-pair mon-pair--tight">
            <div>
              <Readout value={e().moe ? num(e().moe?.cacheHitPct) : DASH} unit="%" />
              <span class="mon-pair__cap">
                <span class="mon-key" data-tone="accent" aria-hidden="true" />
                Cache hits
              </span>
            </div>
            <div>
              <Readout value={e().speculation ? num(e().speculation?.acceptancePct) : DASH} unit="%" />
              <span class="mon-pair__cap">
                <span class="mon-key" data-tone="muted" aria-hidden="true" />
                Accepted
              </span>
            </div>
          </div>
        }
        sub={
          <Show when={e().moe} fallback={<>{e().speculation ? e().speculation?.method : "Not used by this model"}</>}>
            {(moe) => (
              <>
                <span class="mono">
                  {moe().expertsPerToken}/{moe().expertsTotal}
                </span>{" "}
                experts active (<span class="mono">{num(pct(moe().expertsPerToken, moe().expertsTotal), 1)}%</span>)
              </>
            )}
          </Show>
        }
      >
        <Sparkline
          label="Expert cache hits and draft acceptance"
          lines={[
            { key: "hit", label: "Expert cache hits", values: vals("expertHitPct"), tone: "accent", area: true, top: 100, format: (v) => `${fmt0(v)}%` },
            { key: "accept", label: "Drafts accepted", values: vals("draftAcceptPct"), tone: "muted", dashed: true, top: 100, format: (v) => `${fmt0(v)}%` },
          ]}
          stepS={step()}
          end={end()}
          threshold={null}
          rangeLabel={rangeLabel()}
        />
      </MetricCard>
    </div>
  );
}

// ---- Conversation cache ---------------------------------------------------------------------

/** The conversation cache: prompts reused, tokens not read again, parked and restored conversations. */
export function CacheCard(props: { snap: MetricsSnapshot }): JSX.Element {
  const c = () => props.snap.conversationCache;
  let slots: HTMLSpanElement | undefined;
  let mem: HTMLSpanElement | undefined;
  createEffect(() => {
    if (slots) setBarValue(slots, c().slots > 0 ? c().parked / c().slots : 0);
    if (mem) setBarValue(mem, c().budgetGiB > 0 ? c().usedGiB / c().budgetGiB : 0);
  });
  return (
    <article class="mon-panel mon-cache reveal" data-reveal aria-labelledby="mon-cache-title">
      <h2 class="mon-panel__title" id="mon-cache-title">
        <Term k="conversationCache">Conversation cache</Term>
      </h2>
      <p class="mon-cache__lede">
        <Show when={c().requests > 0} fallback={<>No requests yet.</>}>
          <span class="mono">{num(c().requestsReused)}</span> of <span class="mono">{num(c().requests)}</span> requests reused part of their prompt: <span class="mono">{compact(c().reusedTokens)}</span> tokens not read again (
          <span class="mono">{num(pct(c().reusedTokens, c().promptTokens))}%</span> of all prompt tokens).
        </Show>
      </p>
      <Show when={c().enabled}>
        <div class="mon-meters">
          <div class="mon-meter">
            <span class="mon-meter__label">Conversations parked</span>
            <span class="mon-meter__value mono">
              {c().parked} / {c().slots}
            </span>
            <span class="mon-track mon-track--thin" role="meter" aria-label="Conversations parked" aria-valuemin={0} aria-valuemax={c().slots} aria-valuenow={c().parked}>
              <span class="bar mon-bar" ref={slots} />
            </span>
          </div>
          <div class="mon-meter">
            <span class="mon-meter__label">Memory used</span>
            <span class="mon-meter__value mono">
              {num(c().usedGiB, 1)} / {num(c().budgetGiB)} GiB
            </span>
            <span class="mon-track mon-track--thin" role="meter" aria-label="Conversation cache memory" aria-valuemin={0} aria-valuemax={c().budgetGiB} aria-valuenow={c().usedGiB}>
              <span class="bar mon-bar" ref={mem} />
            </span>
          </div>
        </div>
      </Show>
      <dl class="mon-facts mon-facts--row">
        <div>
          <dt>Parked / restored</dt>
          <dd class="mono">
            {num(c().parks)} / {num(c().restores)}
            <Show when={c().evictions > 0}> · {num(c().evictions)} evicted</Show>
          </dd>
        </div>
        <div>
          <dt>Last switch</dt>
          <dd>
            <Show when={c().lastEvent} fallback={<span class="muted">None yet</span>}>
              {(ev) => (
                <>
                  {ev().kind === "parked" ? "Parked" : "Restored"} <span class="mono">{compact(ev().tokens)}</span> tokens, {agoLabel(props.snap.at - ev().at)}
                </>
              )}
            </Show>
          </dd>
        </div>
      </dl>
    </article>
  );
}

// ---- Alerts ---------------------------------------------------------------------------------

const ALERT_ICON: Readonly<Record<string, IconName>> = { thermal: "thermometer", vram: "chip", regression: "bolt" };

/** Alerts that hold right now, above the cards. Not a live region: the toast already announced each. */
export function ActiveAlerts(props: { alerts: AlertTracker; illustrative: boolean }): JSX.Element {
  const active = () => Object.values(props.alerts.active);
  return (
    <Show when={active().length > 0}>
      <div class="mon-alerts" aria-label="Active alerts" role="region">
        <Index each={active()}>
          {(a) => (
            <div class="mon-alert" data-kind={a()?.kind}>
              <Icon name={ALERT_ICON[a()?.kind ?? "vram"] ?? "alert"} size={18} />
              <div>
                <p class="mon-alert__title">
                  {a()?.title} <span class="mon-alert__since">since {clockTime(a()?.since ?? 0)}</span>
                </p>
                <p class="mon-alert__detail">
                  {a()?.detail}
                  <Show when={props.illustrative}> Illustrative data.</Show>
                </p>
              </div>
            </div>
          )}
        </Index>
      </div>
    </Show>
  );
}

/** What the Monitor watches for, and the alerts that have come and gone since it started. */
export function AlertsCard(props: { alerts: AlertTracker }): JSX.Element {
  const recent = () => props.alerts.recent.slice(0, 5);
  return (
    <article class="mon-panel mon-alerts-card reveal" data-reveal aria-labelledby="mon-alerts-title">
      <h2 class="mon-panel__title" id="mon-alerts-title">
        Alerts
      </h2>
      <p class="mon-cache__lede">
        Cerebro watches for <Term k="throttling" />, graphics memory above 95% and writing more than 20% below the tuned baseline, and shows one quiet notice when one starts.
      </p>
      <Show when={recent().length > 0} fallback={<p class="mon-empty">Nothing has gone wrong since the Monitor started watching.</p>}>
        <ul class="mon-recent">
          <Index each={recent()}>
            {(a) => (
              <li>
                <Icon name={ALERT_ICON[a().kind] ?? "alert"} size={14} />
                <span>{a().title}</span>
                <span class="mono muted">
                  {clockTime(a().since)} · {seconds(((a().resolvedAt ?? a().since) - a().since) / 1000)}
                </span>
              </li>
            )}
          </Index>
        </ul>
      </Show>
    </article>
  );
}
