import { createMemo, createSignal, For, onCleanup, onMount, Show, type JSX } from "solid-js";
import { finalThirdFit, type CandidateView, type SoakView } from "../../lib/tune";

/**
 * Live charts for a tune, drawn in SVG at the container's measured pixel size (never a stretched
 * viewBox, so labels keep their shape). Updates only rewrite path data and a few attributes: no
 * layout work per sample, and no transition on the data itself (a new sample simply extends the
 * line), so reduced motion needs no special case here.
 */

/** 0.8 -> 1, 13 -> 20, 42 -> 50: a round axis maximum. */
export function niceCeil(v: number): number {
  if (!(v > 0) || !Number.isFinite(v)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const n = v / p;
  const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  return m * p;
}

/** Width of the element, kept current with a ResizeObserver. */
function useWidth(initial: number): [() => number, (el: HTMLElement) => void] {
  const [w, setW] = createSignal(initial);
  let el: HTMLElement | undefined;
  let ro: ResizeObserver | undefined;
  onMount(() => {
    if (!el) return;
    const target = el;
    const measure = () => {
      const cw = Math.round(target.getBoundingClientRect().width);
      if (cw > 0) setW(cw);
    };
    measure();
    if (typeof ResizeObserver === "function") {
      ro = new ResizeObserver(measure);
      ro.observe(target);
    }
  });
  onCleanup(() => ro?.disconnect());
  return [w, (e) => (el = e)];
}

const fmt = (v: number, digits: number) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

// ---- Writing speed per candidate ----------------------------------------------------------------

const SPEED_H = 240;
const PAD = { l: 44, r: 20, t: 16, b: 30 };

export function SpeedChart(props: { candidates: readonly CandidateView[]; activeId: string | null; highlightId: string | null }): JSX.Element {
  const [w, ref] = useWidth(560);
  const traced = createMemo(() => props.candidates.filter((c) => c.samples.length > 0));
  const tMax = createMemo(() => Math.max(10, Math.ceil(Math.max(0, ...traced().flatMap((c) => c.samples.map((s) => s.t))))));
  const yMax = createMemo(() => niceCeil(Math.max(1, ...traced().flatMap((c) => c.samples.map((s) => s.tps))) * 1.12));
  const x = (t: number) => PAD.l + (t / tMax()) * Math.max(1, w() - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / yMax()) * (SPEED_H - PAD.t - PAD.b);
  const line = (pts: ReadonlyArray<{ t: number; tps: number }>) => pts.map((s, i) => `${i ? "L" : "M"}${x(s.t).toFixed(1)} ${y(s.tps).toFixed(1)}`).join("");
  const yTicks = createMemo(() => [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax()));
  const xTicks = createMemo(() => {
    const step = tMax() <= 12 ? 2 : tMax() <= 24 ? 4 : 10;
    const out: number[] = [];
    for (let t = 0; t <= tMax(); t += step) out.push(t);
    return out;
  });
  const baseline = createMemo(() => props.candidates.find((c) => c.candidate.baseline && c.screenTps !== null) ?? null);
  const active = createMemo(() => props.candidates.find((c) => c.id === props.activeId && c.samples.length > 0) ?? null);
  const summary = createMemo(() => {
    const a = active();
    const last = a?.samples.at(-1);
    const b = baseline();
    return [
      "Writing speed of each candidate during its screen, in tokens per second.",
      a && last ? `${a.candidate.label} is being screened, now at ${fmt(last.tps, 1)} tok/s.` : "",
      b?.screenTps != null ? `llama.cpp automatic screened at ${fmt(b.screenTps, 1)} tok/s.` : "",
      `${traced().length} of ${props.candidates.length} candidates traced.`,
    ]
      .filter(Boolean)
      .join(" ");
  });
  const decimals = () => (yMax() < 10 ? 1 : 0);

  return (
    <div class="chart" ref={ref}>
      <svg class="chart__svg" width={w()} height={SPEED_H} role="img" aria-label={summary()}>
        <g class="chart__grid" aria-hidden="true">
          <For each={yTicks()}>
            {(v) => (
              <>
                <line x1={PAD.l} x2={w() - PAD.r} y1={y(v)} y2={y(v)} classList={{ "chart__rule--base": v === 0 }} />
                <text class="chart__tick" x={PAD.l - 8} y={y(v) + 4} text-anchor="end">
                  {fmt(v, decimals())}
                </text>
              </>
            )}
          </For>
          <For each={xTicks()}>
            {(t) => (
              <text class="chart__tick" x={x(t)} y={SPEED_H - 10} text-anchor="middle">
                {`${t}s`}
              </text>
            )}
          </For>
        </g>
        <Show when={baseline()}>
          {(b) => (
            <g class="chart__ref" aria-hidden="true">
              <line x1={PAD.l} x2={w() - PAD.r} y1={y(b().screenTps ?? 0)} y2={y(b().screenTps ?? 0)} />
              <text x={w() - PAD.r} y={y(b().screenTps ?? 0) - 6} text-anchor="end">
                llama.cpp automatic
              </text>
            </g>
          )}
        </Show>
        <g aria-hidden="true">
          <For each={traced()}>
            {(c) => {
              const warm = () => {
                const firstKept = c.samples.findIndex((s) => s.rep > 1);
                return firstKept < 0 ? c.samples : c.samples.slice(0, firstKept + 1);
              };
              const kept = () => c.samples.filter((s) => s.rep > 1);
              return (
                <g
                  class="trace"
                  data-candidate={c.id}
                  classList={{
                    "trace--active": c.id === props.activeId,
                    "trace--highlight": c.id === props.highlightId && c.id !== props.activeId,
                    "trace--baseline": c.candidate.baseline,
                    "trace--out": c.status === "discarded" || c.status === "interrupted",
                    "trace--win": c.winner,
                  }}
                >
                  <path class="trace__warm" d={line(warm())} />
                  <path class="trace__line" d={line(kept())} />
                </g>
              );
            }}
          </For>
        </g>
        <Show when={active()}>
          {(a) => {
            const last = () => a().samples.at(-1)!;
            const lx = () => Math.min(x(last().t), w() - PAD.r - 2);
            return (
              <g class="chart__now" aria-hidden="true">
                <circle cx={x(last().t)} cy={y(last().tps)} r="4" />
                <text x={lx() - 8} y={y(last().tps) - 10} text-anchor="end">
                  {`${fmt(last().tps, 1)} tok/s`}
                </text>
              </g>
            );
          }}
        </Show>
      </svg>
    </div>
  );
}

/** A small per-candidate trace for the candidate list. Scales to its own range. */
export function Spark(props: { samples: ReadonlyArray<{ rep: number; tps: number }>; max: number }): JSX.Element {
  const W = 72;
  const H = 22;
  const d = createMemo(() => {
    const pts = props.samples;
    if (pts.length < 2) return "";
    const max = props.max > 0 ? props.max : 1;
    return pts.map((s, i) => `${i ? "L" : "M"}${((i / (pts.length - 1)) * (W - 2) + 1).toFixed(1)} ${(H - 2 - (s.tps / max) * (H - 4)).toFixed(1)}`).join("");
  });
  return (
    <svg class="spark" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <path d={d()} />
    </svg>
  );
}

// ---- Free video memory during the soak ----------------------------------------------------------

const VRAM_H = 200;

export function VramChart(props: { soaks: readonly SoakView[]; soakSeconds: number; label: (candidateId: string) => string }): JSX.Element {
  const [w, ref] = useWidth(520);
  const T = () => Math.max(60, props.soakSeconds);
  const yMax = createMemo(() => niceCeil(Math.max(0.5, ...props.soaks.flatMap((s) => s.samples.map((p) => p.vramFreeGiB))) * 1.3));
  const x = (t: number) => PAD.l + (t / T()) * Math.max(1, w() - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / yMax()) * (VRAM_H - PAD.t - PAD.b);
  const yTicks = createMemo(() => [0, 0.5, 1].map((f) => f * yMax()));
  const xTicks = createMemo(() => {
    const out: number[] = [];
    const step = T() >= 480 ? 120 : 60;
    for (let t = 0; t <= T(); t += step) out.push(t);
    return out;
  });
  const summary = createMemo(() => {
    const parts = props.soaks.map((s) => {
      const last = s.samples.at(-1);
      const state = s.result ? (s.result.passed ? "passed" : "failed") : s.interrupted ? "interrupted" : "running";
      return `${props.label(s.candidateId)}: ${state}${last ? `, ${fmt(last.vramFreeGiB, 2)} GiB free` : ""}.`;
    });
    return `Free video memory during each soak, in GiB. The shaded band is the final third, where the trend must not fall. ${parts.join(" ")}`;
  });

  return (
    <div class="chart" ref={ref}>
      <svg class="chart__svg" width={w()} height={VRAM_H} role="img" aria-label={summary()}>
        <g aria-hidden="true">
          <rect class="chart__band" x={x((T() * 2) / 3)} y={PAD.t} width={Math.max(0, x(T()) - x((T() * 2) / 3))} height={VRAM_H - PAD.t - PAD.b} />
          <text class="chart__band-label" x={x(T()) - 6} y={PAD.t + 14} text-anchor="end">
            final third
          </text>
        </g>
        <g class="chart__grid" aria-hidden="true">
          <For each={yTicks()}>
            {(v) => (
              <>
                <line x1={PAD.l} x2={w() - PAD.r} y1={y(v)} y2={y(v)} classList={{ "chart__rule--base": v === 0 }} />
                <text class="chart__tick" x={PAD.l - 8} y={y(v) + 4} text-anchor="end">
                  {fmt(v, 2)}
                </text>
              </>
            )}
          </For>
          <For each={xTicks()}>
            {(t) => (
              <text class="chart__tick" x={x(t)} y={VRAM_H - 10} text-anchor="middle">
                {`${t / 60}m`}
              </text>
            )}
          </For>
        </g>
        <g aria-hidden="true">
          <For each={props.soaks}>
            {(s) => {
              const d = () => s.samples.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.vramFreeGiB).toFixed(1)}`).join("");
              const fit = createMemo(() => finalThirdFit(s.samples, props.soakSeconds));
              const state = () => (s.result ? (s.result.passed ? "passed" : "failed") : s.interrupted ? "interrupted" : "running");
              return (
                <g class="soak-trace" data-state={state()} data-candidate={s.candidateId}>
                  <path class="soak-trace__line" d={d()} />
                  <Show when={fit()}>
                    {(f) => {
                      const at = (t: number) => f().meanY + f().slope * (t - f().meanT);
                      return <line class="soak-trace__trend" x1={x(f().fromT)} y1={y(at(f().fromT))} x2={x(f().toT)} y2={y(at(f().toT))} />;
                    }}
                  </Show>
                  <Show when={s.samples.at(-1)}>
                    {(p) => (
                      // A failed soak ends in a cross, not a dot: its state reads without colour.
                      <Show when={state() === "failed"} fallback={<circle class="soak-trace__end" cx={x(p().t)} cy={y(p().vramFreeGiB)} r="3.5" />}>
                        <path
                          class="soak-trace__cross"
                          d={`M${(x(p().t) - 4).toFixed(1)} ${(y(p().vramFreeGiB) - 4).toFixed(1)}l8 8M${(x(p().t) + 4).toFixed(1)} ${(y(p().vramFreeGiB) - 4).toFixed(1)}l-8 8`}
                        />
                      </Show>
                    )}
                  </Show>
                </g>
              );
            }}
          </For>
        </g>
      </svg>
    </div>
  );
}
