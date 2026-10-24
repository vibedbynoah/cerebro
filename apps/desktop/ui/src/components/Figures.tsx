import { createEffect, For, Index, on, Show, untrack, type Accessor, type JSX } from "solid-js";
import { prefersReducedMotion } from "@design/motion";
import { IllustrativeFlag } from "./Screen";
import type { Provenance } from "../service/types";

const formatters = new Map<number, Intl.NumberFormat>();

/** Same fixed-locale formatting as the HUD (lib/format.ts) and the count-up (motion.ts), so a tile
 *  and the status bar never round one sample differently. Precision is kept ("16.8", "7.0"). */
function countText(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return "–";
  let fmt = formatters.get(decimals);
  if (!fmt) {
    fmt = new Intl.NumberFormat("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    formatters.set(decimals, fmt);
  }
  return fmt.format(value);
}

/** One digit of a live readout. When its character changes it dips briefly in opacity (an
 *  instrument tick, never a replayed count-up); unchanged digits stay still. */
function LiveDigit(props: { ch: Accessor<string> }): JSX.Element {
  let el: HTMLSpanElement | undefined;
  createEffect(
    on(
      props.ch,
      () => {
        if (!el || prefersReducedMotion() || typeof el.animate !== "function") return;
        el.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 160, easing: "cubic-bezier(0.2, 0, 0, 1)" });
      },
      { defer: true },
    ),
  );
  return (
    <span class="live-digit" ref={el}>
      {props.ch()}
    </span>
  );
}

/**
 * Design-system stat tile, in two kinds:
 * - Static (default): `value` renders final in markup (no-JS and screen readers get the real
 *   number) and motion.ts counts it up once when it scrolls into view. The count-up owns the
 *   element's text from then on, so a static tile shows the value it was created with.
 * - `live`: the value follows its source (Solid props are reactive, so pass the sample directly).
 *   No `data-count-to`, so motion.ts never touches it; tabular digits change in place.
 *
 * `accent` marks the one live value on a screen (accent colour is never on every tile).
 */
export function StatTile(props: {
  value: number;
  decimals?: number;
  unit?: string;
  of?: string;
  label: JSX.Element;
  live?: boolean;
  accent?: boolean;
  provenance: Provenance;
  source?: string;
}): JSX.Element {
  const text = () => countText(props.value, props.decimals ?? 0);
  return (
    <div class="stat-tile" classList={{ "stat-tile--live": props.accent ?? false, "stat-tile--updating": props.live ?? false }}>
      <p class="stat-tile__value">
        <Show
          when={props.live}
          fallback={
            <span class="count" data-count-to={untrack(() => props.value.toFixed(props.decimals ?? 0))}>
              {untrack(text)}
            </span>
          }
        >
          <span class="live-value" data-value={text()}>
            <span class="sr-only">{text()}</span>
            <span class="live-value__digits" aria-hidden="true">
              <Index each={text().split("")}>{(ch) => <LiveDigit ch={ch} />}</Index>
            </span>
          </span>
        </Show>
        <Show when={props.of}>
          <span class="stat-tile__of">{props.of}</span>
        </Show>
        <Show when={props.unit}>
          <span class="stat-tile__unit">{props.unit}</span>
        </Show>
      </p>
      <p class="stat-tile__label">{props.label}</p>
      <Show when={props.provenance === "illustrative"} fallback={<span class="stat-tile__source">{props.source ?? "measured"}</span>}>
        <IllustrativeFlag />
      </Show>
    </div>
  );
}

export interface ComparisonRow {
  label: string;
  value: number;
  /** Bar length as a share of the longest row, 0..1. */
  share: number;
  /** The faster configuration, drawn in the accent colour (tuned or defaults, whichever won). */
  winner?: boolean;
}

/** Bar lengths from the values themselves: the largest row fills the track, the rest scale to it. */
export function shares(rows: ReadonlyArray<Omit<ComparisonRow, "share">>): ComparisonRow[] {
  const max = Math.max(0, ...rows.map((r) => (Number.isFinite(r.value) ? r.value : 0)));
  return rows.map((r) => ({ ...r, share: max > 0 && Number.isFinite(r.value) ? Math.max(0, r.value) / max : 0 }));
}

/** Default-vs-tuned bars that slide in on scroll, with the disclosure inside the same figure. */
export function Comparison(props: { rows: readonly ComparisonRow[]; unit: string; label: string; note?: JSX.Element }): JSX.Element {
  return (
    <figure class="comparison-figure" aria-label={props.label}>
      <Show when={!__DESKTOP__}>
        <figcaption class="stat-tile__flag">Illustrative data — not a measurement</figcaption>
      </Show>
      <div class="comparison">
        <For each={props.rows}>
          {(row, i) => {
            const pct = Math.round(Math.min(1, Math.max(0, row.share)) * 100);
            const text = row.value.toFixed(1);
            return (
              <div class="comparison__row">
                <span class="comparison__label">{row.label}</span>
                <span class="comparison__track">
                  <span
                    class={`bar bar--p${pct}`}
                    classList={{ "bar--tuned": row.winner ?? false }}
                    data-bar
                    data-bar-to={String(Math.min(1, Math.max(0, row.share)))}
                    data-bar-delay={String(i() * 120)}
                  />
                </span>
                <span class="comparison__value">
                  <span class="count" data-count-to={text}>
                    {text}
                  </span>
                  <span class="comparison__unit">{props.unit}</span>
                </span>
              </div>
            );
          }}
        </For>
      </div>
      <Show when={props.note}>
        <p class="comparison-figure__note">{props.note}</p>
      </Show>
    </figure>
  );
}
