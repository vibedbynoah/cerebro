import { createEffect, createMemo, createSignal, createUniqueId, For, Match, on, onCleanup, Show, Switch, untrack, type JSX } from "solid-js";
import { setBarValue } from "@design/motion";
import { Icon } from "../../components/icons";
import { IllustrativeFlag } from "../../components/Screen";
import { Term } from "../../components/Term";
import {
  depthLabel,
  finalThirdSlope,
  formatClock,
  NOISE_MARGIN,
  PHASE_IDS,
  PHASES,
  PROFILES,
  tuneProgress,
  type CandidateView,
  type SoakView,
  type TuneView,
} from "../../lib/tune";
import { Spinner } from "../models/parts";
import { Spark, SpeedChart, VramChart } from "./charts";
import type { TuneSession } from "./session";
import { fmt, Num, ProgressBar, SettingName } from "./bits";
import { WinnerCard } from "./Winner";
import { useAppState } from "../../state";

// ---- Phases -------------------------------------------------------------------------------------

/**
 * The six stages. With no run it is the explainer; during a run each step is done, current or
 * still to come, and a stopped run marks where it stopped.
 */
export function PhaseStepper(props: { view: TuneView | null }): JSX.Element {
  // The explainer (no run) assembles as it scrolls into view: the steps arrive in order and the rail
  // draws across behind them (scroll reveal and comparison-bar primitives: transform and opacity
  // only, shown at once under reduced motion). A run's stepper shows its state, no entrance.
  const idle = untrack(() => !props.view?.run);
  const index = () => (props.view?.phase ? PHASE_IDS.indexOf(props.view.phase) : -1);
  const stateOf = (i: number): "done" | "current" | "stopped" | "todo" => {
    const v = props.view;
    if (!v || !v.run) return "todo";
    if (v.status === "done") return "done";
    if (i < index()) return "done";
    if (i === index()) return v.status === "running" ? "current" : "stopped";
    return "todo";
  };
  const rail = () => {
    const v = props.view;
    if (!v || !v.run) return 0;
    if (v.status === "done") return 1;
    return Math.max(0, index()) / (PHASE_IDS.length - 1);
  };
  let fill: HTMLSpanElement | undefined;
  createEffect(
    on(
      rail,
      (r) => {
        if (fill) setBarValue(fill, r);
      },
      { defer: true },
    ),
  );
  return (
    <div class="phases" classList={{ "phases--idle": !props.view?.run }}>
      <span class="phases__rail" aria-hidden="true">
        <Show when={!idle} fallback={<span class="bar bar--p100 phases__draw" data-bar data-bar-to="1" data-bar-delay="160" />}>
          <span ref={fill} class={`bar bar--p${Math.round(untrack(rail) * 100)}`} />
        </Show>
      </span>
      <ol class="phases__list">
        <For each={PHASE_IDS}>
          {(id, i) => (
            <li class="phase" classList={{ reveal: idle }} data-reveal={idle ? "" : undefined} data-state={stateOf(i())} aria-current={stateOf(i()) === "current" ? "step" : undefined}>
              <span class="phase__node" aria-hidden="true">
                <Switch fallback={<span class="phase__num">{i() + 1}</span>}>
                  <Match when={stateOf(i()) === "done"}>
                    <Icon name="check" size={14} />
                  </Match>
                  <Match when={stateOf(i()) === "stopped"}>
                    <Icon name="stop" size={12} />
                  </Match>
                </Switch>
              </span>
              <span class="phase__label">
                {PHASES[id].label}
                <span class="sr-only">{stateOf(i()) === "done" ? " (done)" : stateOf(i()) === "current" ? " (in progress)" : stateOf(i()) === "stopped" ? " (stopped here)" : ""}</span>
              </span>
              <span class="phase__text">{PHASES[id].text}</span>
            </li>
          )}
        </For>
      </ol>
      <Show when={props.view?.run && props.view.phase && props.view.status !== "done" ? props.view.phase : null}>
        {(phase) => (
          <p class="phases__now">
            <strong>{PHASES[phase()].label}.</strong> {PHASES[phase()].text}
          </p>
        )}
      </Show>
    </div>
  );
}

// ---- Header -------------------------------------------------------------------------------------

function StatusChip(props: { view: TuneView }): JSX.Element {
  const winner = () => props.view.candidates.find((c) => c.winner);
  return (
    <Switch>
      <Match when={props.view.status === "running"}>
        <span class="chip chip--live" data-tone="active">
          <span class="chip__pulse" aria-hidden="true" />
          Measuring
        </span>
      </Match>
      <Match when={props.view.status === "cancelled"}>
        <span class="chip" data-tone="off">
          Cancelled
        </span>
      </Match>
      <Match when={props.view.status === "failed"}>
        <span class="chip" data-tone="danger">
          <Icon name="alert" size={12} />
          Stopped
        </span>
      </Match>
      <Match when={props.view.status === "done" && winner()?.candidate.baseline}>
        <span class="chip" data-tone="neutral">
          Defaults were fastest
        </span>
      </Match>
      <Match when={props.view.status === "done"}>
        <span class="chip" data-tone="on">
          <Icon name="check" size={12} />
          Locked in
        </span>
      </Match>
    </Switch>
  );
}

/**
 * The run clock. Events can be a minute apart (a Measure step, soak samples), so while the run is
 * running it ticks locally once a second from the newest event's clock, and snaps back to the
 * event clock whenever an event arrives (a resume offsets the clock). It stops on cancel, failure
 * and the result, and then shows the event clock exactly.
 */
function useRunClock(view: () => TuneView): () => number {
  const [now, setNow] = createSignal(performance.now());
  let anchor = { clock: untrack(() => view().clock), at: performance.now() };
  createEffect(
    on(
      () => view().clock,
      (clock) => {
        anchor = { clock, at: performance.now() };
        setNow(anchor.at);
      },
      { defer: true },
    ),
  );
  createEffect(() => {
    if (view().status !== "running") return;
    anchor = { clock: untrack(() => view().clock), at: performance.now() };
    const id = setInterval(() => setNow(performance.now()), 1000);
    onCleanup(() => clearInterval(id));
  });
  return () => {
    const v = view();
    if (v.status !== "running") return v.clock;
    return Math.max(v.clock, anchor.clock + (now() - anchor.at) / 1000);
  };
}

function RunHeader(props: { session: TuneSession; onNewTune: () => void }): JSX.Element {
  const v = () => props.session.view;
  const run = () => v().run!;
  const clock = useRunClock(v);
  let side: HTMLDivElement | undefined;
  // Cancel and Resume replace each other. When the pressed button goes away, focus would fall to the
  // page; hand it to the button that took its place instead.
  createEffect(
    on(
      () => v().status,
      () => {
        const active = document.activeElement;
        if (side && (active === null || active === document.body)) side.querySelector<HTMLButtonElement>("button")?.focus({ preventScroll: true });
      },
      { defer: true },
    ),
  );
  return (
    <div class="run-head">
      <div class="run-head__text">
        <p class="run-head__eyebrow">
          <span>{PROFILES[run().profile].label}</span>
          <span aria-hidden="true">·</span>
          <span>{`depth ${depthLabel(run().depth)}`}</span>
          <span aria-hidden="true">·</span>
          <span class="mono">{run().build}</span>
        </p>
        <h3 class="run-head__model">
          {run().modelName} <span class="mono run-head__quant">{run().quant}</span>
        </h3>
      </div>
      <div class="run-head__side" ref={side}>
        <StatusChip view={v()} />
        <span class="run-head__clock" title="Run time">
          <Icon name="clock" size={14} />
          <span class="mono">{formatClock(clock())}</span>
          <span class="sr-only"> run time</span>
        </span>
        <Switch>
          <Match when={v().status === "running"}>
            <button
              type="button"
              class="btn btn--destructive btn--sm"
              data-press="firm"
              aria-busy={props.session.busy() === "cancel" ? "true" : undefined}
              onClick={() => void props.session.cancel()}
            >
              <Spinner />
              <span class="btn__label">
                <Icon name="stop" size={14} />
                Cancel tune
              </span>
            </button>
          </Match>
          <Match when={v().status === "cancelled"}>
            <button
              type="button"
              class="btn btn--secondary btn--sm"
              data-press
              aria-busy={props.session.busy() === "resume" ? "true" : undefined}
              onClick={() => void props.session.resume()}
            >
              <Spinner />
              <span class="btn__label">
                <Icon name="play" size={14} />
                Resume
              </span>
            </button>
          </Match>
          <Match when={v().status === "done" || v().status === "failed"}>
            <button type="button" class="btn btn--secondary btn--sm" data-press onClick={() => props.onNewTune()}>
              <Icon name="refresh" size={14} />
              {v().status === "failed" ? "Start again" : "Tune again"}
            </button>
          </Match>
        </Switch>
      </div>
    </div>
  );
}

// ---- Candidates ---------------------------------------------------------------------------------

type RowState = { label: string; tone: "off" | "active" | "on" | "neutral" | "danger"; live?: boolean };

function rowState(c: CandidateView, done: boolean): RowState {
  if (c.winner) return c.candidate.baseline ? { label: "Kept", tone: "neutral" } : { label: "Locked in", tone: "on" };
  if (c.status === "screening") return { label: "Screening", tone: "active", live: true };
  if (c.soak === "running") return { label: "Soaking", tone: "active", live: true };
  if (c.status === "failed") return { label: "Did not load", tone: "danger" };
  if (c.soak === "failed") return { label: "Failed soak", tone: "danger" };
  if (c.quality === "failed") return { label: "Failed quality gate", tone: "danger" };
  if (c.status === "discarded") return { label: "Discarded", tone: "off" };
  if (c.status === "interrupted" || c.soak === "interrupted") return { label: "Interrupted", tone: "off" };
  if (c.soak === "passed") return { label: "Passed soak", tone: "neutral" };
  if (c.quality === "passed") return { label: "Passed quality gate", tone: "neutral" };
  if (c.candidate.baseline && c.status === "advanced") return { label: "Reference", tone: "neutral" };
  if (c.status === "advanced") return { label: done ? "Measured" : c.result ? "Measured" : "Survived screen", tone: "neutral" };
  return { label: "Queued", tone: "off" };
}

function CandidateRow(props: { c: CandidateView; index: number; view: TuneView; sparkMax: number; animate: boolean; onHighlight: (id: string | null) => void }): JSX.Element {
  const [open, setOpen] = createSignal(false);
  const id = createUniqueId();
  const c = () => props.c;
  const state = () => rowState(c(), props.view.status === "done");
  // Beginner hides raw backend flags (lib/experience.ts): the row reads in plain words instead.
  const app = useAppState();
  const showFlags = () => app.level() === "advanced";
  const flags = () =>
    showFlags()
      ? c()
          .candidate.settings.map((s) => s.flag)
          .filter((f): f is string => f !== null)
          .join(" ")
      : "";
  const plain = () =>
    c()
      .candidate.settings.filter((s) => !showFlags() || s.flag === null)
      .map((s) => s.value)
      .join(" · ");
  const lastTps = () => c().samples.at(-1)?.tps ?? null;
  const depth = () => props.view.run?.depth ?? 0;
  return (
    <li
      class="cand"
      classList={{ "cand--enter": props.animate, "cand--active": c().id === props.view.activeId, "cand--win": c().winner, "cand--out": state().tone === "off" || state().tone === "danger" }}
      data-candidate={c().id}
      data-status={c().status}
      onPointerEnter={() => props.onHighlight(c().id)}
      onPointerLeave={() => props.onHighlight(null)}
    >
      <button
        type="button"
        class="cand__head"
        aria-labelledby={`${id}-label ${id}-chip`}
        aria-describedby={`${id}-value`}
        aria-expanded={open()}
        aria-controls={`${id}-detail`}
        onClick={() => setOpen(!open())}
        onFocus={() => props.onHighlight(c().id)}
        onBlur={() => props.onHighlight(null)}
      >
        <span class="cand__index mono" aria-hidden="true">
          {String(props.index + 1).padStart(2, "0")}
        </span>
        <span class="cand__name">
          <span class="cand__label" id={`${id}-label`}>
            {c().candidate.label}
            <Show when={c().candidate.baseline}>
              <span class="cand__ref">baseline</span>
            </Show>
          </span>
          <span class="cand__flags mono" title={flags() || plain()} aria-hidden="true">
            {flags() || plain()}
          </span>
        </span>
        <Spark samples={c().samples} max={props.sparkMax} />
        <span class="cand__value" id={`${id}-value`}>
          <Switch fallback={<span class="muted">–</span>}>
            <Match when={c().result}>
              {(r) => (
                <>
                  <Num value={r().decodeTpsDepth} unit="tok/s" />
                  <span class="cand__at mono">{`@${depthLabel(depth())}`}</span>
                </>
              )}
            </Match>
            <Match when={c().status === "screening" && lastTps() !== null}>
              <Num value={lastTps() ?? 0} unit="tok/s" />
            </Match>
            <Match when={c().screenTps !== null}>
              <Num value={c().screenTps ?? 0} unit="tok/s" />
              <span class="cand__at">burst</span>
            </Match>
          </Switch>
        </span>
        <span class="chip cand__chip" id={`${id}-chip`} data-tone={state().tone}>
          <Show when={state().live}>
            <span class="chip__pulse" aria-hidden="true" />
          </Show>
          <Show when={state().tone === "danger"}>
            <Icon name="alert" size={12} />
          </Show>
          {state().label}
        </span>
        <span class="cand__chevron" aria-hidden="true">
          <Icon name="chevron" size={16} />
        </span>
      </button>
      <div class="cand__detail" id={`${id}-detail`} hidden={!open()}>
        <Show when={c().reason}>
          <p class="cand__reason">{c().reason}</p>
        </Show>
        <dl class="cand__settings">
          <For each={c().candidate.settings}>
            {(s) => (
              <div class="cand__setting">
                <dt>
                  <SettingName name={s.name} />
                </dt>
                <dd>
                  {s.value}
                  <Show when={showFlags() && s.flag}>
                    <code class="mono">{s.flag}</code>
                  </Show>
                </dd>
              </div>
            )}
          </For>
        </dl>
        <Show when={c().result}>
          {(r) => (
            <dl class="cand__three">
              <div>
                <dt>Prompt reading</dt>
                <dd>
                  <Num value={r().prefillTps} digits={0} unit="tok/s" />
                </dd>
              </div>
              <div>
                <dt>Writing, empty context</dt>
                <dd>
                  <Num value={r().decodeTps0} unit="tok/s" />
                </dd>
              </div>
              <div>
                <dt>{`Writing at ${depthLabel(depth())}`}</dt>
                <dd>
                  <Num value={r().decodeTpsDepth} unit="tok/s" />
                </dd>
              </div>
            </dl>
          )}
        </Show>
      </div>
    </li>
  );
}

function Candidates(props: { session: TuneSession; onHighlight: (id: string | null) => void }): JSX.Element {
  const v = () => props.session.view;
  // Rows already there when this list mounts (coming back to the screen replays the run) appear at
  // once; only rows that arrive live, or belong to a newer run, animate in.
  const mountedRun = untrack(() => v().run?.runId ?? null);
  const replayed = new Set(untrack(() => v().candidates.map((c) => c.id)));
  const isNew = (id: string) => untrack(() => v().run?.runId !== mountedRun || !replayed.has(id));
  const sparkMax = createMemo(() => Math.max(1, ...v().candidates.flatMap((c) => c.samples.map((s) => s.tps))));
  const counts = createMemo(() => {
    const cs = v().candidates;
    return {
      screened: cs.filter((c) => c.status === "advanced" || c.status === "discarded" || c.status === "failed").length,
      total: v().run?.plannedCandidates ?? cs.length,
    };
  });
  return (
    <div class="tune-card tune-card--cands">
      <div class="tune-card__head">
        <h3 class="tune-card__title">Candidate configurations</h3>
        <span class="tune-card__meta mono">{`${counts().screened} of ${counts().total} screened`}</span>
      </div>
      <Show when={v().candidates.length > 0} fallback={<p class="tune-card__empty">Candidates appear here once the machine has been probed.</p>}>
        <ol class="cands">
          <For each={v().candidates}>
            {(c, i) => <CandidateRow c={c} index={i()} view={v()} sparkMax={sparkMax()} animate={isNew(c.id)} onHighlight={props.onHighlight} />}
          </For>
        </ol>
      </Show>
    </div>
  );
}

// ---- Live speed ---------------------------------------------------------------------------------

function SpeedCard(props: { view: TuneView; highlightId: string | null }): JSX.Element {
  return (
    <div class="tune-card tune-card--speed">
      <div class="tune-card__head">
        <h3 class="tune-card__title">
          Live <Term k="decode">writing speed</Term>
        </h3>
        <span class="tune-card__meta">
          <Term k="tokps" />
        </span>
      </div>
      <SpeedChart candidates={props.view.candidates} activeId={props.view.activeId} highlightId={props.highlightId} />
      <ul class="legend" aria-label="Chart key">
        <li>
          <span class="legend__swatch legend__swatch--active" aria-hidden="true" />
          Candidate being screened
        </li>
        <li>
          <span class="legend__swatch legend__swatch--ref" aria-hidden="true" />
          Automatic settings (reference)
        </li>
        <li>
          <span class="legend__swatch legend__swatch--other" aria-hidden="true" />
          Other candidates
        </li>
        <li>
          <span class="legend__swatch legend__swatch--warm" aria-hidden="true" />
          Warm-up repetition, discarded
        </li>
        <Show when={props.view.winner && !props.view.candidates.find((c) => c.winner)?.candidate.baseline}>
          <li>
            <span class="legend__swatch legend__swatch--win" aria-hidden="true" />
            Locked in
          </li>
        </Show>
      </ul>
      <p class="tune-card__note">Screening readings are short bursts, used only to rank. Bursts read faster than sustained speed, so the numbers that count come from Measure and Soak.</p>
    </div>
  );
}

// ---- Quality gate -------------------------------------------------------------------------------

function QualityCard(props: { view: TuneView }): JSX.Element {
  const v = () => props.view;
  const reached = () => {
    const p = v().phase;
    return v().status === "done" || (p !== null && PHASE_IDS.indexOf(p) >= PHASE_IDS.indexOf("quality"));
  };
  const label = (id: string) => v().candidates.find((c) => c.id === id)?.candidate.label ?? id;
  const emptyText = (): JSX.Element => {
    const inQuality = v().phase === "quality";
    if (inQuality && v().status === "cancelled") return "Cancelled before any comparison finished. The quality gate starts over on resume.";
    if (inQuality && v().status === "failed") return "The run stopped before any comparison finished.";
    if (reached() && !inQuality)
      return (
        <>
          No candidate beat the automatic settings by more than the {Math.round(NOISE_MARGIN * 100)}% <Term k="noiseMargin" />, so there was nothing to check.
        </>
      );
    if (reached()) return "Comparing answers…";
    return "After Measure, every candidate faster than the automatic settings has its answers compared with theirs.";
  };
  return (
    <div class="tune-card tune-card--quality">
      <div class="tune-card__head">
        <h3 class="tune-card__title">Quality gate</h3>
        <span class="tune-card__meta">
          <Term k="klDiv" />
        </span>
      </div>
      <Show
        when={v().quality.length > 0}
        fallback={
          <p class="tune-card__empty">
            {emptyText()}
          </p>
        }
      >
        <ul class="gates">
          <For each={v().quality}>
            {(q) => {
              const share = () => Math.min(1, q.value / (q.gate * 3));
              return (
                <li class="gate" data-passed={String(q.passed)}>
                  <div class="gate__top">
                    <span class="gate__label">{label(q.candidateId)}</span>
                    <span class="chip" data-tone={q.passed ? "neutral" : "danger"}>
                      <Icon name={q.passed ? "check" : "alert"} size={12} />
                      {q.passed ? "Passed" : "Rejected"}
                    </span>
                  </div>
                  <div class="gate__meter" aria-hidden="true">
                    <span class="gate__track">
                      <span class={`bar bar--p${Math.round(share() * 100)}`} classList={{ "gate__fill--fail": !q.passed }} data-bar data-bar-to={String(share())} />
                    </span>
                    <span class="gate__limit" />
                  </div>
                  <p class="gate__nums mono">
                    {`${q.metric} ${fmt(q.value, 4)}`}
                    <span class="gate__gate">{`limit ${fmt(q.gate, 3)}`}</span>
                  </p>
                </li>
              );
            }}
          </For>
        </ul>
        <p class="tune-card__note">{`Compared with ${v().quality[0]?.reference ?? "the reference"}. Lower is closer.`}</p>
      </Show>
    </div>
  );
}

// ---- Soak ---------------------------------------------------------------------------------------

function SoakCard(props: { view: TuneView }): JSX.Element {
  const v = () => props.view;
  const T = () => v().run?.soakSeconds ?? 600;
  const label = (id: string) => v().candidates.find((c) => c.id === id)?.candidate.label ?? id;
  const current = createMemo<SoakView | null>(() => v().soaks.at(-1) ?? null);
  const slope = createMemo(() => {
    const s = current();
    return s ? (s.result?.slopeMiBPerMin ?? finalThirdSlope(s.samples, T())) : null;
  });
  return (
    <div class="tune-card tune-card--soak">
      <div class="tune-card__head">
        <h3 class="tune-card__title">
          <Term k="soak">Soak</Term>
        </h3>
        <span class="tune-card__meta">
          free <Term k="vram" />
        </span>
      </div>
      <Show
        when={current()}
        fallback={<p class="tune-card__empty">After the quality gate, the fastest remaining candidate runs under sustained load while free graphics memory is watched.</p>}
      >
        {(s) => {
          const last = () => s().samples.at(-1);
          return (
            <>
              <div class="soak-now">
                <p class="soak-now__label">{label(s().candidateId)}</p>
                <ProgressBar value={s().result ? 1 : (last()?.t ?? 0) / T()} label="Soak progress" tone="neutral" />
                <dl class="soak-now__stats">
                  <div>
                    <dt>Run time</dt>
                    <dd class="mono">{`${formatClock(s().result ? T() : (last()?.t ?? 0))} of ${formatClock(T())}`}</dd>
                  </div>
                  <div>
                    <dt>Sustained writing</dt>
                    <dd>{last() ? <Num value={last()!.tps} unit="tok/s" /> : "–"}</dd>
                  </div>
                  <div>
                    <dt>Final-third trend</dt>
                    <dd class="mono" data-trend={slope() === null ? "none" : slope()! >= 0 ? "steady" : "falling"}>
                      {slope() === null ? "–" : `${slope()! >= 0 ? "+" : "−"}${fmt(Math.abs(slope()!), 1)} MiB/min`}
                    </dd>
                  </div>
                </dl>
              </div>
              <VramChart soaks={v().soaks} soakSeconds={T()} label={label} />
              <ul class="soak-list">
                <For each={v().soaks}>
                  {(x) => (
                    <li class="soak-item" data-state={x.result ? (x.result.passed ? "passed" : "failed") : x.interrupted ? "interrupted" : "running"}>
                      <span class="soak-item__swatch" aria-hidden="true" />
                      <span class="soak-item__label">{label(x.candidateId)}</span>
                      <span class="soak-item__verdict">
                        {x.result ? x.result.reason : x.interrupted ? "Interrupted; it starts over on resume." : "Running…"}
                      </span>
                    </li>
                  )}
                </For>
              </ul>
            </>
          );
        }}
      </Show>
    </div>
  );
}

// ---- The whole run ------------------------------------------------------------------------------

export function RunPanel(props: { session: TuneSession; onNewTune: () => void }): JSX.Element {
  const v = () => props.session.view;
  const [highlight, setHighlight] = createSignal<string | null>(null);
  // A result already there when the panel mounts (coming back to the screen) appears at once; a
  // result that arrives live, or belongs to a newer run, makes its entrance.
  const mountedRun = untrack(() => v().run?.runId ?? null);
  const replayedWinner = untrack(() => v().winner !== null);
  const winnerIsLive = () => v().run?.runId !== mountedRun || !replayedWinner;
  return (
    <div class="run" data-status={v().status}>
      <figure class="run__figure">
        <figcaption class="run__caption">
          <IllustrativeFlag />
          <span class="run__provenance">
            {`${v().run?.gpu ?? ""} (${fmt(v().run?.vramTotalGiB ?? 0, 0)} GiB) · ${v().run?.cpu ?? ""}, ${fmt(v().run?.ramTotalGiB ?? 0, 0)} GiB RAM · ${v().run?.build ?? ""} (${v().run?.backendCommit ?? ""}) · ${v().run?.sampler ?? ""}`}
          </span>
        </figcaption>
        <RunHeader session={props.session} onNewTune={props.onNewTune} />
        <ProgressBar value={tuneProgress(v())} label="Tune progress" />
        <PhaseStepper view={v()} />
        <Show when={v().probe}>
          {(p) => (
            <p class="probe-line">
              <Icon name="gpu" size={14} />
              <span>
                <span class="mono">{fmt(p().vramFreeGiB, 2)} GiB</span> of {fmt(v().run?.vramTotalGiB ?? 0, 0)} GiB graphics memory free.
              </span>
              <Show
                when={p().integrity === "verified"}
                fallback={
                  <span class="probe-line__note">
                    <Icon name="info" size={14} />
                    Backend <Term k="integrity" /> is off in Settings, so it was skipped.
                  </span>
                }
              >
                <span class="probe-line__ok">
                  <Icon name="check" size={14} />
                  Engine {v().run?.build ?? ""} verified.
                </span>
              </Show>
            </p>
          )}
        </Show>
        <Show when={v().status === "failed"}>
          <div class="notice notice--inline" role="alert">
            <Icon name="alert" size={18} />
            <p>{`${v().failure ?? "The tune stopped"}.`}</p>
          </div>
        </Show>
        <Show when={v().status === "cancelled"}>
          <p class="hint">
            <Icon name="info" size={16} />
            Cancelled. Finished steps are kept; Resume starts the interrupted step over.
          </p>
        </Show>
        <div class="run__grid">
          <SpeedCard view={v()} highlightId={highlight()} />
          <Candidates session={props.session} onHighlight={setHighlight} />
        </div>
        <div class="run__grid run__grid--checks">
          <QualityCard view={v()} />
          <SoakCard view={v()} />
        </div>
      </figure>
      <Show when={v().winner}>
        <WinnerCard view={v()} animate={untrack(winnerIsLive)} />
      </Show>
    </div>
  );
}
