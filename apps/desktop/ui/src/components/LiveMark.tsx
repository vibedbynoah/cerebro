import { createMemo, For, type JSX } from "solid-js";
import { firingExpert, useMoeActivity } from "../lib/moe-activity";
import { LOCKUP_W, MARK_IDLE, MARK_PICKED, MARK_ROUTER, MARK_STEM, WORDMARK_D } from "./Brand";

/** The mark's seven expert squares, the picked one first: the C the router chooses among. */
const SLOTS: ReadonlyArray<readonly [number, number]> = [MARK_PICKED, ...MARK_IDLE];

/**
 * The title bar's lockup, alive while a mixture-of-experts model works: the route leaves the
 * router for the square of the expert that is firing (expert number modulo the seven squares), as
 * the mark depicts. At rest, and for dense models or under reduced motion, it is the brand mark
 * exactly (Brand.tsx).
 */
export function LiveLockup(props: { height?: number; class?: string }): JSX.Element {
  const moe = useMoeActivity();
  const h = () => Math.max(20, props.height ?? 24);
  const slot = createMemo(() => {
    if (!moe.active()) return 0;
    const e = firingExpert(moe.activity());
    return e === null ? 0 : e % SLOTS.length;
  });
  const at = () => SLOTS[slot()]!;
  return (
    <svg
      class={props.class ? `brand-lockup live-mark ${props.class}` : "brand-lockup live-mark"}
      viewBox={`0 0 ${LOCKUP_W} 24`}
      width={Math.round((LOCKUP_W * h()) / 24)}
      height={h()}
      role="img"
      aria-label="Cerebro"
      data-brand="lockup"
      data-firing={moe.active() ? String(slot()) : undefined}
    >
      <g class="brand__idle">
        <For each={SLOTS}>{([x, y], i) => <rect x={x} y={y} width="6" height="6" rx="1.5" class={i() === slot() ? "live-mark__hidden" : undefined} />}</For>
      </g>
      <g class="brand__route">
        <rect class="live-mark__picked" x={at()[0]} y={at()[1]} width="6" height="6" rx="1.5" />
        {slot() === 0 ? (
          <path d={MARK_STEM} />
        ) : (
          <line class="live-mark__route" x1={MARK_ROUTER.cx} y1={MARK_ROUTER.cy} x2={at()[0] + 3} y2={at()[1] + 3} />
        )}
        <circle cx={MARK_ROUTER.cx} cy={MARK_ROUTER.cy} r={MARK_ROUTER.r} />
      </g>
      <path class="brand__word" transform="translate(32 0)" d={WORDMARK_D} />
    </svg>
  );
}
