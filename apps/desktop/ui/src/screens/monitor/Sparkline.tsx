import { createMemo, createSignal, createUniqueId, Index, Show, type JSX } from "solid-js";
import { agoLabel, sampleTime, seriesStats, sparkGeometry } from "../../lib/monitor";

/**
 * A metric's history as an SVG sparkline. One fixed box (viewBox 0 0 W H stretched to the card,
 * non-scaling strokes), so a new sample only rewrites path data: no layout, no transition on the
 * data, and nothing for reduced motion to stop. Each line may have its own scale (decode and
 * prefill differ a hundredfold), and the legend says so.
 *
 * Keyboard: the chart is one tab stop; Left/Right step through samples, Home/End jump to the
 * oldest/newest, Escape leaves. The value under the cursor is shown below the chart and spoken
 * politely. Pointer hover does the same.
 */

/** One series in a sparkline: its samples (null is a gap), scale, look, and how a value reads. */
export interface SparkLine {
  key: string;
  label: string;
  values: ReadonlyArray<number | null>;
  /** Scale top. Defaults to a little above the series' own maximum. */
  top?: number;
  tone: "accent" | "muted" | "warning";
  format: (v: number) => string;
  /** Draw the area under the line (first line only, usually). */
  area?: boolean;
  dashed?: boolean;
}

const W = 300;
const H = 48;

export function Sparkline(props: {
  label: string;
  lines: readonly SparkLine[];
  stepS: number;
  end: number;
  /** A horizontal reference (a throttle point, a power limit) on the first line's scale. */
  threshold?: { value: number; label: string } | null;
  rangeLabel: string;
}): JSX.Element {
  const id = createUniqueId();
  const [cursor, setCursor] = createSignal<number | null>(null);
  // Spoken only when the user moves the cursor, never because a new sample arrived under it.
  const [spoken, setSpoken] = createSignal("");
  const length = () => Math.max(0, ...props.lines.map((l) => l.values.length));

  const tops = createMemo(() =>
    props.lines.map((l, i) => {
      if (l.top !== undefined && l.top > 0) return l.top;
      const s = seriesStats(l.values);
      const th = i === 0 && props.threshold ? props.threshold.value : 0;
      return Math.max(1e-6, (s.max ?? 0) * 1.15, th * 1.08);
    }),
  );
  const geom = createMemo(() => props.lines.map((l, i) => sparkGeometry(l.values, tops()[i] ?? 1, W, H)));
  const thresholdY = () => {
    const t = props.threshold;
    const top = tops()[0];
    if (!t || !top) return null;
    return 2 + (1 - Math.min(1, t.value / top)) * (H - 4);
  };

  const summary = createMemo(() => {
    const parts = props.lines.map((l) => {
      const s = seriesStats(l.values);
      if (s.count === 0) return `${l.label}: no samples`;
      return `${l.label} now ${s.last === null ? "none" : l.format(s.last)}, low ${l.format(s.min ?? 0)}, high ${l.format(s.max ?? 0)}`;
    });
    return `${props.label}, ${props.rangeLabel}. ${parts.join(". ")}. Use the arrow keys to read samples.`;
  });

  const readout = createMemo(() => {
    const i = cursor();
    const n = length();
    if (i === null || n === 0) return null;
    const t = sampleTime({ stepS: props.stepS, end: props.end }, n, i);
    const when = agoLabel(props.end - t);
    const vals = props.lines.map((l) => {
      const v = l.values[i];
      return `${l.label} ${v === null || v === undefined ? "–" : l.format(v)}`;
    });
    return { when, text: vals.join(" · "), x: n > 1 ? (i / (n - 1)) * 100 : 0 };
  });

  const onKey = (e: KeyboardEvent) => {
    const n = length();
    if (n === 0) return;
    const cur = cursor();
    let next: number | null = cur;
    if (e.key === "ArrowLeft") next = cur === null ? n - 1 : Math.max(0, cur - 1);
    else if (e.key === "ArrowRight") next = cur === null ? n - 1 : Math.min(n - 1, cur + 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    else if (e.key === "Escape") next = null;
    else return;
    e.preventDefault();
    setCursor(next);
    const r = readout();
    setSpoken(r ? `${r.when}: ${r.text}` : "");
  };
  const onPointer = (e: PointerEvent) => {
    const el = e.currentTarget as HTMLElement;
    const r = el.getBoundingClientRect();
    const n = length();
    if (n === 0 || r.width === 0) return;
    setCursor(Math.round(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * (n - 1)));
  };

  return (
    <div class="spark">
      <div
        class="spark__plot"
        tabindex="0"
        role="group"
        aria-roledescription="chart"
        aria-label={summary()}
        aria-describedby={`${id}-read`}
        onKeyDown={onKey}
        onBlur={() => setCursor(null)}
        onPointerMove={onPointer}
        onPointerLeave={() => setCursor(null)}
      >
        <svg class="spark__svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <Show when={thresholdY() !== null}>
            <line class="spark__threshold" x1="0" x2={W} y1={thresholdY() ?? 0} y2={thresholdY() ?? 0} vector-effect="non-scaling-stroke" />
          </Show>
          {/* Index, not For: the lines array is rebuilt every sample, and keying by position keeps
              the same <path> elements, so only their d attribute changes. */}
          <Index each={props.lines}>
            {(l, i) => (
              <g class="spark__series" data-tone={l().tone}>
                <Show when={l().area}>
                  <path class="spark__area" d={geom()[i]?.area ?? ""} />
                </Show>
                <path class="spark__line" classList={{ "is-dashed": l().dashed === true }} d={geom()[i]?.line ?? ""} vector-effect="non-scaling-stroke" />
              </g>
            )}
          </Index>
        </svg>
        <Show when={readout()}>{(r) => <span class="spark__cursor" style={{ "--x": `${r().x}%` }} aria-hidden="true" />}</Show>
      </div>
      <p class="spark__foot" id={`${id}-read`}>
        <Show
          when={readout()}
          fallback={
            <>
              <span>{props.rangeLabel}</span>
              <Show when={props.threshold}>{(t) => <span class="spark__legend-threshold">{t().label}</span>}</Show>
            </>
          }
        >
          {(r) => (
            <>
              <span class="spark__when">{r().when}</span>
              <span class="spark__val">{r().text}</span>
            </>
          )}
        </Show>
      </p>
      <span class="sr-only" aria-live="polite">
        {spoken()}
      </span>
    </div>
  );
}
