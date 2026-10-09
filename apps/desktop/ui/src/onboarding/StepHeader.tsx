import type { JSX } from "solid-js";
import { Icon, type IconName } from "../components/icons";
import { STEPS, type StepId } from "./progress";

/** "Step 3 · This machine": numbered from the step list, so adding a step never leaves a stale number. */
export function eyebrowFor(id: StepId): string {
  const i = STEPS.findIndex((s) => s.id === id);
  return `Step ${i + 1} · ${STEPS[i]?.label ?? ""}`;
}

/** One step's heading block: eyebrow, the h1 that is focused on arrival, and the lede. */
export function StepHeader(props: { eyebrow: string; icon: IconName; title: string; lede: JSX.Element; wide?: boolean }): JSX.Element {
  return (
    <header class="onb-head" classList={{ "onb-head--wide": props.wide ?? false }}>
      <p class="eyebrow">
        <Icon name={props.icon} size={14} />
        {props.eyebrow}
      </p>
      <h1 class="onb-title" id="onb-step-title" tabindex="-1" aria-describedby="onb-step-count">
        {props.title}
      </h1>
      <p class="onb-lede">{props.lede}</p>
    </header>
  );
}
