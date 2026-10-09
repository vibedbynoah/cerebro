import { For, onCleanup, onMount, type JSX } from "solid-js";
import { prefersReducedMotion } from "@design/motion";
import { MARK_IDLE, MARK_PICKED, MARK_ROUTER, MARK_STEM, RouteMotif } from "../components/Brand";

/**
 * The welcome picture: Cerebro's mark at hero size over the routing motif. On arrival it assembles
 * once and tells the idea in one gesture: the seven experts settle onto the grid, the router
 * appears at the centre, the route reaches up, and the one expert it picked lights blue with a
 * single ping. No numbers: it is the brand's picture of routing, not a result.
 *
 * Motion runs once through the Web Animations API (allowed by the strict CSP, unlike inline
 * styles), transform and opacity only. Every animation uses fill "backwards", so the static CSS is
 * the final picture: with reduced motion, no JS or a failed module, the finished mark simply shows.
 */
export function BrandHero(): JSX.Element {
  let root: SVGSVGElement | undefined;

  onMount(() => {
    if (!root || prefersReducedMotion() || typeof root.animate !== "function") return;
    const q = (sel: string) => Array.from(root?.querySelectorAll(sel) ?? []);
    const running: Animation[] = [];
    const run = (el: Element | undefined, frames: Keyframe[], opts: KeyframeAnimationOptions) => {
      if (el) running.push(el.animate(frames, { fill: "backwards", ...opts }));
    };
    const emphasized = "cubic-bezier(0.2, 0, 0, 1)";
    const spring = "cubic-bezier(0.34, 1.56, 0.64, 1)";

    // Experts settle in around the C, the open side last.
    q("[data-hero='expert']").forEach((el, i) =>
      run(el, [{ opacity: 0, transform: "translateY(10%) scale(0.7)" }, { opacity: 1, transform: "none" }], { duration: 520, delay: 120 + i * 70, easing: emphasized }),
    );
    run(q("[data-hero='router']")[0], [{ opacity: 0, transform: "scale(0.3)" }, { opacity: 1, transform: "scale(1)" }], { duration: 460, delay: 700, easing: spring });
    run(q("[data-hero='stem']")[0], [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }], { duration: 360, delay: 1050, easing: emphasized });
    run(q("[data-hero='picked']")[0], [{ opacity: 0, transform: "scale(0.7)" }, { opacity: 1, transform: "scale(1)" }], { duration: 440, delay: 1320, easing: spring });
    run(
      q("[data-hero='ping']")[0],
      [
        { opacity: 0, transform: "scale(1)", offset: 0 },
        { opacity: 0.6, transform: "scale(1)", offset: 0.04 },
        { opacity: 0, transform: "scale(2.4)", offset: 1 },
      ],
      { duration: 900, delay: 1420, easing: "cubic-bezier(0, 0, 0.2, 1)" },
    );
    onCleanup(() => running.forEach((a) => a.cancel()));
  });

  return (
    <div class="brand-hero">
      <RouteMotif class="brand-hero__motif" />
      <svg class="brand-hero__mark" viewBox="0 0 24 24" width="168" height="168" ref={root} aria-hidden="true" data-brand="mark">
        <g class="brand__idle">
          <For each={MARK_IDLE}>{([x, y]) => <rect data-hero="expert" x={x} y={y} width="6" height="6" rx="1.5" />}</For>
        </g>
        <g class="brand__route">
          <path class="brand-hero__stem" data-hero="stem" d={MARK_STEM} />
          <rect class="brand-hero__ping" data-hero="ping" x={MARK_PICKED[0]} y={MARK_PICKED[1]} width="6" height="6" rx="1.5" />
          <rect data-hero="picked" x={MARK_PICKED[0]} y={MARK_PICKED[1]} width="6" height="6" rx="1.5" />
          <circle data-hero="router" cx={MARK_ROUTER.cx} cy={MARK_ROUTER.cy} r={MARK_ROUTER.r} />
        </g>
      </svg>
    </div>
  );
}
