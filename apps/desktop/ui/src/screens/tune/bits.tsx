import { createEffect, For, on, Show, untrack, type JSX } from "solid-js";
import { setBarValue } from "@design/motion";
import { Term } from "../../components/Term";
import { settingTerm, type TermText } from "../../lib/tune";

export const fmt = (v: number, digits: number) => v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** A number and its unit as one unit of type (DESIGN.md). */
export function Num(props: { value: number; digits?: number; unit: string }): JSX.Element {
  return (
    <span class="num-unit mono">
      {fmt(props.value, props.digits ?? 1)}
      <span class="unit">{props.unit}</span>
    </span>
  );
}

/** A progress fill that slides from its old value to the new one (setBarValue), or jumps under
 *  reduced motion. The value is also exposed as a progressbar for assistive tech. */
export function ProgressBar(props: { value: number; label: string; tone?: "accent" | "neutral" }): JSX.Element {
  let fill: HTMLSpanElement | undefined;
  const pct = () => Math.round(Math.min(1, Math.max(0, props.value)) * 100);
  const initial = untrack(pct);
  createEffect(
    on(
      () => props.value,
      (v) => {
        if (fill) setBarValue(fill, v);
      },
      { defer: true },
    ),
  );
  return (
    <span class="tune-progress" role="progressbar" aria-label={props.label} aria-valuemin="0" aria-valuemax="100" aria-valuenow={pct()}>
      <span ref={fill} class={`bar bar--p${initial}`} classList={{ "bar--tuned": props.tone !== "neutral" }} />
    </span>
  );
}


/** A setting's name with its plain-words definition (lib/tune SETTING_TERMS); an unknown name from
 *  the engine is shown as plain text. */
export function SettingName(props: { name: string }): JSX.Element {
  return (
    <Show when={settingTerm(props.name)} fallback={props.name}>
      {(key) => <Term k={key()}>{props.name}</Term>}
    </Show>
  );
}

/** Copy with its technical terms defined inline. */
export function TermLine(props: { parts: TermText }): JSX.Element {
  return <For each={props.parts}>{(part) => (typeof part === "string" ? part : <Term k={part.term}>{part.text}</Term>)}</For>;
}
