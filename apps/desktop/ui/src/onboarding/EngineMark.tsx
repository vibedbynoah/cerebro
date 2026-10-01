import { For, onCleanup, onMount, type JSX } from "solid-js";
import { prefersReducedMotion } from "@design/motion";

/**
 * The welcome instrument: Cerebro's dial (the brand mark, at hero size). On arrival it assembles
 * once and tells the product in one gesture: the needle sweeps the whole range (trying
 * configurations), passes the llama.cpp defaults mark, and settles on the fastest safe setting it
 * found. No numbers: it is a picture of the method, not a result.
 *
 * Motion runs once, through the Web Animations API (allowed by the strict CSP, unlike inline
 * styles): transform and opacity, plus stroke-dashoffset for the draw-on (the documented
 * exception). Every animation uses fill "backwards", so the static CSS is the final picture: with
 * reduced motion, no JS or a failed module, the finished dial simply shows.
 */

const C = 160;
/** Needle angles, degrees clockwise from 12 o'clock. The arc spans -135..135. */
export const DEFAULTS_DEG = -30;
export const BEST_DEG = 60;

function polar(r: number, deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180;
  return [Math.round((C + r * Math.sin(rad)) * 100) / 100, Math.round((C - r * Math.cos(rad)) * 100) / 100];
}

function arc(r: number, from: number, to: number): string {
  const [x1, y1] = polar(r, from);
  const [x2, y2] = polar(r, to);
  return `M${x1} ${y1}A${r} ${r} 0 ${to - from > 180 ? 1 : 0} 1 ${x2} ${y2}`;
}

const TICKS = Array.from({ length: 28 }, (_, i) => {
  const deg = -135 + i * 10;
  const major = i % 3 === 0;
  const [x1, y1] = polar(major ? 92 : 97, deg);
  const [x2, y2] = polar(104, deg);
  return { d: `M${x1} ${y1}L${x2} ${y2}`, major };
});

const TRACK = arc(118, -135, 135);
const VALUE = arc(118, -135, BEST_DEG);
const [DX, DY] = polar(118, DEFAULTS_DEG);
const [BX, BY] = polar(118, BEST_DEG);

export function EngineMark(): JSX.Element {
  let root: SVGSVGElement | undefined;
  let glow: HTMLDivElement | undefined;

  onMount(() => {
    if (!root || prefersReducedMotion() || typeof root.animate !== "function") return;
    const q = <T extends Element>(sel: string) => Array.from(root?.querySelectorAll<T & Element>(sel) ?? []);
    const running: Animation[] = [];
    const run = (el: Element | undefined, frames: Keyframe[], opts: KeyframeAnimationOptions) => {
      if (el) running.push(el.animate(frames, { fill: "backwards", ...opts }));
    };
    const draw = [
      { strokeDasharray: "1 1", strokeDashoffset: 1 },
      { strokeDasharray: "1 1", strokeDashoffset: 0 },
    ];
    const emphasized = "cubic-bezier(0.2, 0, 0, 1)";
    const spring = "cubic-bezier(0.34, 1.56, 0.64, 1)";

    run(glow, [{ opacity: 0, transform: "scale(0.7)" }, { opacity: 1, transform: "scale(1)" }], { duration: 1400, easing: emphasized });
    run(q("[data-engine='bezel']")[0], draw, { duration: 900, easing: emphasized });
    run(q("[data-engine='track']")[0], draw, { duration: 900, delay: 120, easing: emphasized });
    q("[data-engine='tick']").forEach((t, i) => run(t, [{ opacity: 0 }, { opacity: 1 }], { duration: 240, delay: 300 + i * 18, easing: "linear" }));
    run(q("[data-engine='hub']")[0], [{ opacity: 0, transform: "scale(0.4)" }, { opacity: 1, transform: "scale(1)" }], { duration: 420, delay: 420, easing: spring });
    // The search: full sweep, overshoot back, settle on the best found. One pass, never looped.
    run(
      q("[data-engine='needle']")[0],
      [
        { transform: "rotate(-135deg)", opacity: 0, offset: 0 },
        { transform: "rotate(-135deg)", opacity: 1, offset: 0.06 },
        { transform: "rotate(128deg)", opacity: 1, offset: 0.58, easing: "cubic-bezier(0.45, 0, 0.2, 1)" },
        { transform: `rotate(${BEST_DEG - 8}deg)`, opacity: 1, offset: 0.84 },
        { transform: `rotate(${BEST_DEG}deg)`, opacity: 1, offset: 1 },
      ],
      { duration: 1700, delay: 600, easing: "cubic-bezier(0.4, 0, 0.2, 1)" },
    );
    run(q("[data-engine='defaults']")[0], [{ opacity: 0, transform: "scale(0.3)" }, { opacity: 1, transform: "scale(1)" }], { duration: 320, delay: 1000, easing: spring });
    run(q("[data-engine='value']")[0], draw, { duration: 700, delay: 2150, easing: emphasized });
    run(q("[data-engine='best']")[0], [{ opacity: 0, transform: "scale(0.2)" }, { opacity: 1, transform: "scale(1)" }], { duration: 420, delay: 2300, easing: spring });
    // One ping when the best setting is found.
    run(
      q("[data-engine='ping']")[0],
      [
        { opacity: 0, transform: "scale(1)", offset: 0 },
        { opacity: 0.7, transform: "scale(1)", offset: 0.02 },
        { opacity: 0, transform: "scale(3)", offset: 1 },
      ],
      { duration: 900, delay: 2400, easing: "cubic-bezier(0, 0, 0.2, 1)" },
    );
    onCleanup(() => running.forEach((a) => a.cancel()));
  });

  return (
    <div class="engine">
      <div class="engine__glow" ref={glow} aria-hidden="true" />
      <svg class="engine__svg" viewBox="0 0 320 320" ref={root} aria-hidden="true">
        <circle class="engine__bezel" data-engine="bezel" cx={C} cy={C} r="150" pathLength="1" />
        <circle class="engine__inner" cx={C} cy={C} r="66" />
        <path class="engine__track" data-engine="track" d={TRACK} pathLength="1" />
        <For each={TICKS}>{(t) => <path class="engine__tick" classList={{ "engine__tick--major": t.major }} data-engine="tick" d={t.d} />}</For>
        <path class="engine__value" data-engine="value" d={VALUE} pathLength="1" />
        <circle class="engine__defaults" data-engine="defaults" cx={DX} cy={DY} r="5" />
        <circle class="engine__ping" data-engine="ping" cx={BX} cy={BY} r="7" />
        <circle class="engine__best" data-engine="best" cx={BX} cy={BY} r="7" />
        <g class="engine__needle" data-engine="needle">
          <path class="engine__needle-line" d={`M${C} ${C + 16}L${C} ${C - 100}`} />
        </g>
        <g class="engine__hub" data-engine="hub">
          <circle class="engine__hub-ring" cx={C} cy={C} r="15" />
          <circle class="engine__hub-dot" cx={C} cy={C} r="5" />
        </g>
      </svg>
    </div>
  );
}
