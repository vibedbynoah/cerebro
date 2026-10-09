import { createUniqueId, For, onCleanup, onMount, type JSX } from "solid-js";

/**
 * Cerebro's mark, wordmark and routing motif, drawn inline from the masters in design/brand/svg
 * (BRAND.md) so they re-theme with the app: idle experts and the wordmark take the ink token, the
 * route, router and picked expert take the accent. tests/unit/brand.test.ts checks this geometry
 * against the master files, so the two can never drift apart.
 *
 * The mark: seven experts (rounded squares on a 3 x 3 grid) form a C; the router (circle) sits in
 * its centre; one route (the stem) connects it to the one expert it picked. Only the route is blue.
 */

/** Idle experts: [x, y] of each 6-unit square on the 24-unit grid. */
export const MARK_IDLE: ReadonlyArray<readonly [number, number]> = [
  [0, 0],
  [18, 0],
  [0, 9],
  [0, 18],
  [9, 18],
  [18, 18],
];
/** The expert the router picked. */
export const MARK_PICKED: readonly [number, number] = [9, 0];
export const MARK_STEM = "M10.5 5h3v5h-3z";
export const MARK_ROUTER = { cx: 12, cy: 12, r: 3 } as const;
/** The drawn wordmark (2.4-unit monoline on a 12-unit x-height, 75.6 x 18). Never typeset it. */
export const WORDMARK_D =
  "M9.57 8.79A4.8 4.8 0 1 0 9.57 15.21M12.8 12H22.4A4.8 4.8 0 1 0 21.28 15.09M26.4 18V6M26.4 11.6A4.4 4.4 0 0 1 30.8 7.2H31.8M32 12H41.6A4.8 4.8 0 1 0 40.48 15.09M45.6 0V18M45.6 12A4.8 4.8 0 1 1 55.2 12A4.8 4.8 0 1 1 45.6 12M59.2 18V6M59.2 11.6A4.4 4.4 0 0 1 63.6 7.2H64.6M64.8 12A4.8 4.8 0 1 1 74.4 12A4.8 4.8 0 1 1 64.8 12";
export const WORDMARK_W = 75.6;
/** Horizontal lockup: mark 24, gap 8, wordmark (ascender on the mark's top edge). */
export const LOCKUP_W = 32 + WORDMARK_W;

/** The mark's shapes, for use inside any 24-unit coordinate space. */
export function MarkShapes(props: { idleClass?: string; routeClass?: string }): JSX.Element {
  return (
    <>
      <g class={props.idleClass ?? "brand__idle"}>
        <For each={MARK_IDLE}>{([x, y]) => <rect x={x} y={y} width="6" height="6" rx="1.5" />}</For>
      </g>
      <g class={props.routeClass ?? "brand__route"}>
        <rect x={MARK_PICKED[0]} y={MARK_PICKED[1]} width="6" height="6" rx="1.5" />
        <path d={MARK_STEM} />
        <circle cx={MARK_ROUTER.cx} cy={MARK_ROUTER.cy} r={MARK_ROUTER.r} />
      </g>
    </>
  );
}

/**
 * The mark alone. Sizes on a multiple of 8 px put every edge on a whole pixel (16 px: 4 px experts,
 * 2 px channels); never below 16 px. Decorative unless `label` is given.
 */
export function BrandMark(props: { size?: number; label?: string; class?: string }): JSX.Element {
  const size = () => Math.max(16, props.size ?? 24);
  return (
    <svg
      class={props.class ? `brand-mark ${props.class}` : "brand-mark"}
      viewBox="0 0 24 24"
      width={size()}
      height={size()}
      role={props.label ? "img" : undefined}
      aria-label={props.label}
      aria-hidden={props.label ? undefined : "true"}
      data-brand="mark"
    >
      <MarkShapes />
    </svg>
  );
}

/**
 * Horizontal lockup (the default): mark, then the drawn wordmark. `height` is the mark's height;
 * 20 px is the smallest that keeps the wordmark stroke at 2 px.
 */
export function BrandLockup(props: { height?: number; label?: string; class?: string }): JSX.Element {
  const h = () => Math.max(20, props.height ?? 24);
  return (
    <svg
      class={props.class ? `brand-lockup ${props.class}` : "brand-lockup"}
      viewBox={`0 0 ${LOCKUP_W} 24`}
      width={Math.round((LOCKUP_W * h()) / 24)}
      height={h()}
      role="img"
      aria-label={props.label ?? "Cerebro"}
      data-brand="lockup"
    >
      <MarkShapes />
      <path class="brand__word" transform="translate(32 0)" d={WORDMARK_D} />
    </svg>
  );
}

/** Seamless 192 x 96 motif tile: idle experts (layer 1) and the bus with four routes (layer 2). */
const MOTIF_CELLS: ReadonlyArray<readonly [number, number]> = [
  [8, 8], [32, 8], [56, 8], [80, 8], [104, 8], [128, 8], [152, 8], [176, 8],
  [8, 32], [56, 32], [80, 32], [104, 32], [128, 32], [152, 32],
  [8, 56], [32, 56], [80, 56], [128, 56], [152, 56], [176, 56],
  [8, 80], [32, 80], [56, 80], [80, 80], [104, 80], [128, 80], [152, 80], [176, 80],
];
export const MOTIF_ROUTE_D = "M0 48H192M24 48V36H32M72 48V60H64M120 48V60H112M168 48V36H176";
const MOTIF_PICKED: ReadonlyArray<readonly [number, number]> = [
  [32, 32],
  [176, 32],
  [56, 56],
  [104, 56],
];
const MOTIF_ROUTERS = [24, 72, 120, 168];

/**
 * The routing motif, behind empty states and the first-run screen only, never behind data
 * (BRAND.md "Motif"). Experts take the border token, routes the accent at 55%. The routes light
 * once when the motif scrolls into view; they never animate continuously, and under reduced motion
 * they are simply lit.
 */
export function RouteMotif(props: { class?: string }): JSX.Element {
  const id = `motif-${createUniqueId()}`;
  let root: SVGSVGElement | undefined;
  onMount(() => {
    const el = root;
    if (!el) return;
    if (typeof IntersectionObserver !== "function") {
      el.classList.add("is-lit");
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          el.classList.add("is-lit");
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    onCleanup(() => io.disconnect());
  });
  return (
    <svg class={props.class ? `motif ${props.class}` : "motif"} ref={root} aria-hidden="true" data-brand="motif">
      <defs>
        <pattern id={`${id}-cells`} width="192" height="96" patternUnits="userSpaceOnUse">
          <For each={MOTIF_CELLS}>{([x, y]) => <rect x={x} y={y} width="8" height="8" rx="2" />}</For>
        </pattern>
        <pattern id={`${id}-routes`} width="192" height="96" patternUnits="userSpaceOnUse">
          <path class="motif__bus" d={MOTIF_ROUTE_D} />
          <For each={MOTIF_PICKED}>{([x, y]) => <rect x={x} y={y} width="8" height="8" rx="2" />}</For>
          <For each={MOTIF_ROUTERS}>{(x) => <circle cx={x} cy="48" r="2.5" />}</For>
        </pattern>
      </defs>
      <rect class="motif__cells" width="100%" height="100%" fill={`url(#${id}-cells)`} />
      <rect class="motif__routes" width="100%" height="100%" fill={`url(#${id}-routes)`} />
    </svg>
  );
}
