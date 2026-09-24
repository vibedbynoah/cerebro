/**
 * Cerebro shared motion primitives. Framework-agnostic, dependency-free; used by the Astro site and
 * the Tauri/SolidJS app.
 *
 * Rules:
 *  - Reduced motion (`prefers-reduced-motion: reduce`) shows every final state instantly. The
 *    preference is re-checked live; data that already arrived never animates twice.
 *  - Animate `transform` and `opacity` only. One exception: draw-on (`stroke-dashoffset`).
 *  - Never write `style.transform`. Primitives write custom properties (`--mx`, `--my`,
 *    `--press-scale`) and one rule in motion.css composes them.
 *  - Hidden starting states live in motion.css behind `motion-pending` (set before first paint by
 *    theme-init.js, with a CSS failsafe) and, after that, behind a per-element arm: an `init*`
 *    sets `data-motion-armed` on each element it observes, in the same task. Only armed elements
 *    are ever hidden, so markup no primitive owns (a list item mounted later, an island that never
 *    called init) always renders final. Without JS the page is always complete.
 *  - Every `init*` returns a cleanup function (which disarms, so anything it no longer watches
 *    shows its final state) and is safe to call again on the same elements.
 */

const REDUCE_QUERY = "(prefers-reduced-motion: reduce)";
/** Must match the failsafe delay in motion.css. */
const FAILSAFE_MS = 2500;

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia(REDUCE_QUERY).matches;
}

/** Hover-only effects (magnetic) run for a real mouse only, never touch or pen. */
function hasFinePointer(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  return window.matchMedia("(hover: hover) and (pointer: fine)").matches;
}

/** Calls `cb` now and on every change of the reduced-motion preference. Returns an unsubscribe. */
export function onReducedMotionChange(cb: (reduced: boolean) => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    cb(false);
    return () => {};
  }
  const mql = window.matchMedia(REDUCE_QUERY);
  const listener = () => cb(mql.matches);
  listener();
  mql.addEventListener("change", listener);
  return () => mql.removeEventListener("change", listener);
}

const clamp01 = (n: number): number => Math.min(1, Math.max(0, n));

/** Set on an element a primitive has taken ownership of; motion.css hides only armed elements. */
const ARMED = "data-motion-armed";
const arm = (el: Element): void => el.setAttribute(ARMED, "");
const disarm = (el: Element): void => el.removeAttribute(ARMED);
const isArmed = (el: Element): boolean => el.hasAttribute(ARMED);

/** Every element matching `selector` under `root`, plus `root` itself when it matches (watch mode
 *  hands each added node in as its own root). */
function queryAll<T extends Element>(root: ParentNode, selector: string): T[] {
  const found = Array.from(root.querySelectorAll<T>(selector));
  if (typeof Element !== "undefined" && root instanceof Element && root.matches(selector)) found.unshift(root as unknown as T);
  return found;
}
const clamp = (n: number, min: number, max: number): number => Math.min(max, Math.max(min, n));

function inViewport(el: Element): boolean {
  const r = el.getBoundingClientRect();
  return r.bottom > 0 && r.top < window.innerHeight && r.right > 0 && r.left < window.innerWidth;
}

/** Count-ups that already started. A later setup shows their final value instead of replaying. */
const counted = new WeakSet<Element>();
/** Count-ups a live initCountUps() currently drives. */
const countOwned = new WeakSet<Element>();

// ---------------------------------------------------------------------------
// Scroll reveal
// ---------------------------------------------------------------------------

export interface RevealOptions {
  /** Default: `[data-reveal]`. */
  selector?: string;
  /** Delay between elements that enter the viewport together, ms. Default 60. */
  staggerMs?: number;
  /** Stagger steps before the delay stops growing. Default 6 (so at most 360 ms). */
  maxSteps?: number;
  rootMargin?: string;
  threshold?: number;
  /** Reveal once (default) or re-hide when scrolled back out. */
  once?: boolean;
}

/** Document order, for sorting entries that arrive in one IntersectionObserver batch. */
const byDocumentOrder = (a: Element, b: Element): number =>
  a === b ? 0 : a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;

/** Interactive elements carry their own `transition` (press spring, magnetic, hover lift). A reveal
 *  class on the same element declares a competing `transition` at the same specificity, so one of
 *  the two silently loses: the card snaps in, or the button stops springing. */
const INTERACTIVE = "[data-press], [data-magnetic], [data-sweep], .btn, a[href], button, input, select, textarea, summary";
const warnedReveal = new WeakSet<Element>();

/** Warns once per element when a reveal class sits on an interactive element instead of a
 *  wrapper around it (DESIGN.md, motion principle 10). Exported for tests. */
export function warnInteractiveReveal(elements: Iterable<Element>): void {
  for (const el of elements) {
    if (warnedReveal.has(el) || !el.matches(INTERACTIVE)) continue;
    warnedReveal.add(el);
    console.warn("[cerebro/motion] Put .reveal/.pop-in on a wrapper, not on an interactive element; their transitions conflict:", el);
  }
}

/** Adds `.is-visible` to `.reveal`/`.reveal-emphasized`/`.pop-in` elements as they enter the
 *  viewport. The stagger (`--reveal-delay`) is set per batch: elements that enter together
 *  arrive as a group in document order, and an element that scrolls in alone gets no delay.
 *  Reveal goes on a wrapper, never on an interactive element (see warnInteractiveReveal). */
export function initScrollReveal(root: ParentNode = document, options: RevealOptions = {}): () => void {
  const selector = options.selector ?? "[data-reveal]";
  const staggerMs = options.staggerMs ?? 60;
  const maxSteps = options.maxSteps ?? 6;
  const once = options.once ?? true;
  const all = queryAll<HTMLElement>(root, selector);
  warnInteractiveReveal(all);
  // Skips elements another live init already owns, so a repeated call never double-observes.
  const elements = all.filter((el) => !isArmed(el) && !(once && el.classList.contains("is-visible")));
  if (elements.length === 0) return () => {};

  if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
    for (const el of elements) el.classList.add("is-visible");
    return () => {};
  }

  const observer = new IntersectionObserver(
    (entries) => {
      const entering = entries
        .filter((e) => e.isIntersecting)
        .map((e) => e.target as HTMLElement)
        .filter((el) => !el.classList.contains("is-visible"))
        .sort(byDocumentOrder);
      entering.forEach((el, i) => {
        el.style.setProperty("--reveal-delay", `${Math.min(i, maxSteps) * staggerMs}ms`);
        el.classList.add("is-visible");
        if (once) observer.unobserve(el);
      });
      if (!once) {
        for (const entry of entries) {
          if (entry.isIntersecting) continue;
          const el = entry.target as HTMLElement;
          el.classList.remove("is-visible");
          el.style.removeProperty("--reveal-delay");
        }
      }
    },
    { root: null, rootMargin: options.rootMargin ?? "0px 0px -10% 0px", threshold: options.threshold ?? 0.15 },
  );
  for (const el of elements) {
    arm(el);
    observer.observe(el);
  }
  return () => {
    observer.disconnect();
    for (const el of elements) disarm(el);
  };
}

// ---------------------------------------------------------------------------
// Count-up numbers
// ---------------------------------------------------------------------------

export interface CountUpOptions {
  from?: number;
  to: number;
  /** ms. Default 900 (--duration-count). Non-positive or non-finite falls back to 900; capped at
   *  3000. */
  duration?: number;
  /** Default: en-US integer formatting. */
  formatter?: (n: number) => string;
  /** 0..1 -> 0..1. Default: ease-out cubic. */
  easing?: (t: number) => number;
}

const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3);

/** Longest a count-up may run: data arriving, not a show. */
const MAX_COUNT_MS = 3000;
const DEFAULT_COUNT_MS = 900;

/** Only a finite, positive duration is accepted, capped at MAX_COUNT_MS. Anything else (negative,
 *  zero, NaN, Infinity) would never let progress reach 1 or would stall the digits for minutes. */
function safeDuration(d: number | undefined): number {
  return d !== undefined && Number.isFinite(d) && d > 0 ? Math.min(d, MAX_COUNT_MS) : DEFAULT_COUNT_MS;
}

/** Animates `el`'s text from `from` to `to`. Writes `from` synchronously, so the next painted frame
 *  is never a stale value. Returns a cancel function. */
export function countUp(el: HTMLElement, options: CountUpOptions): () => void {
  const from = options.from ?? 0;
  const to = options.to;
  const duration = safeDuration(options.duration);
  const format = options.formatter ?? ((n: number) => Math.round(n).toLocaleString("en-US"));
  const easing = options.easing ?? easeOutCubic;

  if (!Number.isFinite(to) || prefersReducedMotion() || typeof requestAnimationFrame === "undefined") {
    if (Number.isFinite(to)) el.textContent = format(to);
    return () => {};
  }

  el.textContent = format(from);
  let raf = 0;
  let start = 0;
  const step = (t: number) => {
    if (!start) start = t;
    const progress = clamp01((t - start) / duration);
    el.textContent = format(progress < 1 ? from + (to - from) * easing(progress) : to);
    if (progress < 1) raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  return () => cancelAnimationFrame(raf);
}

interface CountTarget {
  visible: HTMLElement;
  from: number;
  to: number;
  duration: number;
  fmt: Intl.NumberFormat;
}

/** Splits a `[data-count-to]` element into an `.sr-only` final value (read by assistive tech) and
 *  an `aria-hidden` span that animates. Idempotent. Returns null for invalid data. */
function prepareCount(el: HTMLElement): CountTarget | null {
  const rawTo = el.dataset["countTo"] ?? "";
  const to = Number(rawTo);
  if (rawTo === "" || !Number.isFinite(to)) return null;
  const rawFrom = el.dataset["countFrom"];
  const parsedFrom = rawFrom === undefined ? 0 : Number(rawFrom);
  const from = Number.isFinite(parsedFrom) ? parsedFrom : 0;
  const rawDuration = el.dataset["countDuration"];
  const duration = safeDuration(rawDuration === undefined ? undefined : Number(rawDuration));
  // Keep the authored precision ("17.9" stays one decimal) and a fixed locale, so animated digits
  // never disagree with the server-rendered markup.
  const decimals = (rawTo.split(".")[1] ?? "").length;
  const fmt = new Intl.NumberFormat("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  const finalText = fmt.format(to);

  let visible = el.querySelector<HTMLElement>(':scope > [aria-hidden="true"]');
  if (!visible) {
    const sr = document.createElement("span");
    sr.className = "sr-only";
    sr.textContent = finalText;
    visible = document.createElement("span");
    visible.setAttribute("aria-hidden", "true");
    visible.textContent = el.textContent ?? "";
    el.replaceChildren(sr, visible);
  } else {
    const sr = el.querySelector<HTMLElement>(":scope > .sr-only");
    if (sr) sr.textContent = finalText;
  }
  // Reserve the final width (tabular mono digits are 1ch) so neighbouring text never shifts.
  el.style.setProperty("display", "inline-block");
  el.style.setProperty("min-width", `${finalText.length}ch`);
  return { visible, from, to, duration, fmt };
}

/**
 * Counts every `[data-count-to]` element up when it scrolls into view. Markup contract: the
 * authored text is already the final value (`<span data-count-to="17.9">17.9</span>`), so no-JS
 * visitors and screen readers always get the real number. Optional: `data-count-from`,
 * `data-count-duration` (ms).
 */
export function initCountUps(root: ParentNode = document, selector = "[data-count-to]"): () => void {
  // An element a live init already drives is left alone: a second init would cut its count short.
  const elements = queryAll<HTMLElement>(root, selector).filter((el) => !countOwned.has(el));
  if (elements.length === 0) return () => {};
  const cancels: Array<() => void> = [];
  const targets = new Map<HTMLElement, CountTarget>();
  for (const el of elements) {
    const t = prepareCount(el);
    if (t) targets.set(el, t);
  }

  const showFinal = (t: CountTarget) => {
    t.visible.textContent = t.fmt.format(t.to);
  };
  const start = (el: HTMLElement) => {
    const t = targets.get(el);
    if (!t) return;
    if (counted.has(el)) return showFinal(t);
    counted.add(el);
    cancels.push(countUp(t.visible, { from: t.from, to: t.to, duration: t.duration, formatter: (n) => t.fmt.format(n) }));
  };

  if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
    for (const [el, t] of targets) {
      counted.add(el); // shown final = arrived; a later switch to full motion must not replay it
      showFinal(t);
    }
    return () => {};
  }

  for (const el of targets.keys()) countOwned.add(el);
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        start(entry.target as HTMLElement);
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.4 },
  );
  for (const [el, t] of targets) {
    if (counted.has(el)) showFinal(t);
    else if (inViewport(el)) start(el); // same task: the first frame after un-hiding shows `from`
    else {
      t.visible.textContent = t.fmt.format(t.from); // never final-then-reset when it scrolls in
      observer.observe(el);
    }
  }

  return () => {
    observer.disconnect();
    for (const cancel of cancels) cancel();
    // Unwatched means final: a count cut short, or one still waiting at `from`, shows the real value.
    for (const [el, t] of targets) {
      countOwned.delete(el);
      showFinal(t);
    }
  };
}

// ---------------------------------------------------------------------------
// Draw-on SVG strokes
// ---------------------------------------------------------------------------

export interface DrawOnOptions {
  delayMs?: number;
}

/** Must match --duration-draw in tokens.css; the done-fallback waits this plus a margin. */
const DRAW_MS = 900;
const DRAW_DONE_MARGIN_MS = 300;

/** Adds `.is-draw-done` (motion.css: `stroke-dasharray: none`) once the stroke has drawn, so the
 *  final state never depends on dash arithmetic and a later resize can never re-truncate it. A timer
 *  backs up `transitionend`, which never fires for a transition that did not run. */
function markDoneAfterDraw(el: SVGElement, delayMs: number): void {
  let timer = 0;
  const done = () => {
    el.removeEventListener("transitionend", onEnd);
    clearTimeout(timer);
    el.classList.add("is-draw-done");
  };
  const onEnd = (e: Event) => {
    if ((e as TransitionEvent).propertyName === "stroke-dashoffset") done();
  };
  el.addEventListener("transitionend", onEnd);
  timer = window.setTimeout(done, delayMs + DRAW_MS + DRAW_DONE_MARGIN_MS);
}

/**
 * A `vector-effect: non-scaling-stroke` path lays its dashes out in screen pixels, while
 * `pathLength="1"` normalises them in user units: in a non-uniformly stretched SVG a "full-length"
 * dash of 1 then covers only part of the path. For such an element this writes an upper bound for
 * the dash (`--draw-len`, in pathLength units: the largest axis scale of its screen transform) and
 * adds `.draw--scaled`, whose CSS uses it. Over-estimating is always safe: a dash longer than the
 * path still paints the whole path. Returns false when the length can't be measured (not rendered).
 */
function prepareDrawLength(el: SVGElement): boolean {
  if (getComputedStyle(el).vectorEffect !== "non-scaling-stroke") return true;
  const geom = el as unknown as Partial<SVGGeometryElement>;
  const ctm = typeof geom.getScreenCTM === "function" ? geom.getScreenCTM() : null;
  if (!ctm || typeof geom.getTotalLength !== "function") return false;
  const total = geom.getTotalLength();
  const declared = Number(el.getAttribute("pathLength"));
  const units = declared > 0 ? declared : total; // without pathLength, dash values are user units
  const scale = Math.max(Math.hypot(ctm.a, ctm.b), Math.hypot(ctm.c, ctm.d), 1);
  if (!Number.isFinite(scale) || !(total > 0)) return false;
  el.style.setProperty("--draw-len", String(Math.ceil(units * scale * 1.02 * 1000) / 1000));
  el.classList.add("draw--scaled");
  return true;
}

const warnedDraw = new WeakSet<Element>();

/** Warns once per element for a non-scaling-stroke `[data-draw]` whose screen length could not be
 *  measured. It is then left unarmed: shown drawn, without animation, never truncated. */
function warnUnmeasuredDraw(el: Element): void {
  if (warnedDraw.has(el)) return;
  warnedDraw.add(el);
  console.warn("[cerebro/motion] [data-draw] with vector-effect: non-scaling-stroke has no measurable screen length (is it rendered?); showing it drawn without animation:", el);
}

/**
 * Marks an SVG path drawn (`.is-drawn`); motion.css animates `stroke-dashoffset`, then
 * `.is-draw-done` clears the dash. Documented exception to transform/opacity-only: nothing else
 * reveals a stroke along its own geometry, so use it sparingly. Every `[data-draw]` element carries
 * `pathLength="1"` so the hidden state is static CSS with no measurement.
 */
export function drawOn(path: Element, options: DrawOnOptions = {}): void {
  const el = path as SVGElement;
  const delayMs = options.delayMs ?? 0;
  el.style.setProperty("--draw-delay", `${delayMs}ms`);
  el.classList.add("is-drawn");
  if (!el.classList.contains("is-draw-done")) markDoneAfterDraw(el, delayMs);
}

/** Draws every `[data-draw]` when its nearest `[data-draw-trigger]` (or itself) scrolls into view,
 *  staggering paths that share a trigger. */
export function initDrawOnReveal(root: ParentNode = document, selector = "[data-draw]", staggerMs = 120): () => void {
  const candidates = queryAll<SVGElement>(root, selector).filter((el) => !el.classList.contains("is-drawn") && !isArmed(el));
  if (candidates.length === 0) return () => {};

  if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
    for (const el of candidates) drawOn(el);
    return () => {};
  }

  const elements: SVGElement[] = [];
  for (const el of candidates) {
    if (prepareDrawLength(el)) {
      arm(el);
      elements.push(el);
    } else {
      warnUnmeasuredDraw(el);
    }
  }
  if (elements.length === 0) return () => {};

  const groups = new Map<Element, SVGElement[]>();
  for (const el of elements) {
    const trigger = el.closest("[data-draw-trigger]") ?? el;
    const list = groups.get(trigger) ?? [];
    list.push(el);
    groups.set(trigger, list);
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        (groups.get(entry.target) ?? []).forEach((el, i) => drawOn(el, { delayMs: i * staggerMs }));
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.3 },
  );
  for (const trigger of groups.keys()) observer.observe(trigger);
  return () => {
    observer.disconnect();
    for (const el of elements) disarm(el);
  };
}

// ---------------------------------------------------------------------------
// Magnetic hover
// ---------------------------------------------------------------------------

export interface MagneticOptions {
  /** Fraction of the pointer offset the element travels. Default 0.3. */
  strength?: number;
  /** Max travel in px. Default 14. */
  maxOffset?: number;
}

const magneticWired = new WeakSet<Element>();

/** Nudges `el` toward a hovering mouse via `--mx`/`--my`. No-op for touch/pen, reduced motion, or
 *  an element that is already wired. */
export function initMagnetic(el: HTMLElement, options: MagneticOptions = {}): () => void {
  if (prefersReducedMotion() || !hasFinePointer() || magneticWired.has(el)) return () => {};
  magneticWired.add(el);
  const strength = options.strength ?? 0.3;
  const maxOffset = options.maxOffset ?? 14;

  // Measured once per pointerenter, before any offset: measuring on every move would chase the
  // element's own transform.
  let rect: DOMRect | null = null;
  // One style write per frame, however fast the mouse reports.
  let frame = 0;
  let pendingX = 0;
  let pendingY = 0;

  const reset = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    el.style.setProperty("--mx", "0px");
    el.style.setProperty("--my", "0px");
  };
  const onEnter = (e: PointerEvent) => {
    if (e.pointerType !== "mouse") return;
    el.classList.add("is-tracking");
    reset();
    rect = el.getBoundingClientRect();
  };
  const onMove = (e: PointerEvent) => {
    if (e.pointerType !== "mouse" || !rect) return;
    pendingX = clamp((e.clientX - (rect.left + rect.width / 2)) * strength, -maxOffset, maxOffset);
    pendingY = clamp((e.clientY - (rect.top + rect.height / 2)) * strength, -maxOffset, maxOffset);
    if (!frame) {
      frame = requestAnimationFrame(() => {
        frame = 0;
        el.style.setProperty("--mx", `${pendingX}px`);
        el.style.setProperty("--my", `${pendingY}px`);
      });
    }
  };
  const onLeave = () => {
    el.classList.remove("is-tracking");
    reset();
    rect = null;
  };

  el.addEventListener("pointerenter", onEnter);
  el.addEventListener("pointermove", onMove);
  el.addEventListener("pointerleave", onLeave);
  return () => {
    onLeave();
    el.removeEventListener("pointerenter", onEnter);
    el.removeEventListener("pointermove", onMove);
    el.removeEventListener("pointerleave", onLeave);
    magneticWired.delete(el);
  };
}

export function initMagneticAll(root: ParentNode = document, selector = "[data-magnetic]"): () => void {
  const cleanups = queryAll<HTMLElement>(root, selector).map((el) => initMagnetic(el));
  return () => cleanups.forEach((fn) => fn());
}

// ---------------------------------------------------------------------------
// Press spring
// ---------------------------------------------------------------------------

const pressWired = new WeakSet<Element>();

/** Toggles `.is-pressed` on pointer and keyboard activation of `[data-press]` (Enter and Space on a
 *  button, Enter only on a link).
 *  `.is-pressed` only sets `--press-scale`, so it composes with magnetic hover. */
export function initPressSpring(root: ParentNode = document, selector = "[data-press]"): () => void {
  const teardown: Array<() => void> = [];

  for (const el of queryAll<HTMLElement>(root, selector)) {
    if (pressWired.has(el)) continue;
    pressWired.add(el);
    // Space on a focused link scrolls the page instead of following it, so a link only shows the
    // press for the key that activates it (Enter). Buttons and role=button take both.
    const isLink = (el instanceof HTMLAnchorElement && el.hasAttribute("href")) || el.getAttribute("role") === "link";
    const activates = (key: string) => key === "Enter" || (key === " " && !isLink);
    const press = () => el.classList.add("is-pressed");
    const release = () => el.classList.remove("is-pressed");
    const onKeydown = (e: KeyboardEvent) => {
      if (!e.repeat && activates(e.key)) press();
    };
    const onKeyup = (e: KeyboardEvent) => {
      if (e.key === " " || e.key === "Enter") release();
    };
    const events: Array<[string, EventListener]> = [
      ["pointerdown", press],
      ["pointerup", release],
      ["pointercancel", release],
      ["pointerleave", release],
      ["blur", release], // focus left mid-press: no keyup will arrive
      ["keydown", onKeydown as EventListener],
      ["keyup", onKeyup as EventListener],
    ];
    for (const [type, fn] of events) el.addEventListener(type, fn);
    // Enter on a link can navigate before keyup arrives; without this the page would come back
    // from the back/forward cache with the control still squashed.
    const win = el.ownerDocument.defaultView;
    win?.addEventListener("pagehide", release);
    teardown.push(() => {
      for (const [type, fn] of events) el.removeEventListener(type, fn);
      win?.removeEventListener("pagehide", release);
      release();
      pressWired.delete(el);
    });
  }

  return () => teardown.forEach((fn) => fn());
}

// ---------------------------------------------------------------------------
// Button activation guard (busy / disabled)
// ---------------------------------------------------------------------------

/** Checked live at each event: a button's busy/disabled state can change while the page is open. */
function isBlockedActivation(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  const btn = target.closest(".btn");
  if (!btn) return false;
  return btn.getAttribute("aria-busy") === "true" || btn.getAttribute("aria-disabled") === "true" || btn.hasAttribute("disabled");
}

/**
 * `pointer-events: none` stops a mouse click on a busy button, but keyboard Enter/Space activates
 * the focused element directly. Capture-phase listeners on `root` block both paths for any `.btn`
 * that is busy or disabled at the moment of the event.
 */
export function initButtonGuards(root: ParentNode = document): () => void {
  const target = root as unknown as EventTarget; // Document, Element and DocumentFragment all are
  const onClick = (e: Event) => {
    if (isBlockedActivation(e.target)) {
      e.preventDefault();
      e.stopPropagation();
    }
  };
  const onKeydown = (e: Event) => {
    const key = (e as KeyboardEvent).key;
    if ((key === " " || key === "Enter") && isBlockedActivation(e.target)) {
      e.preventDefault();
      e.stopPropagation();
    }
  };
  target.addEventListener("click", onClick, true);
  target.addEventListener("keydown", onKeydown, true);
  return () => {
    target.removeEventListener("click", onClick, true);
    target.removeEventListener("keydown", onKeydown, true);
  };
}

// ---------------------------------------------------------------------------
// Light sweep
// ---------------------------------------------------------------------------

/** The sweep itself is pure CSS (`.sweep` in motion.css); this opts `[data-sweep]` elements in. */
export function applyLightSweep(root: ParentNode = document, selector = "[data-sweep]"): void {
  for (const el of queryAll<HTMLElement>(root, selector)) el.classList.add("sweep");
}

// ---------------------------------------------------------------------------
// Comparison morph (default vs tuned bars)
// ---------------------------------------------------------------------------

/** Slides each `[data-bar]` (a pill whose box is its final length, moved with
 *  `transform: translateX()` from behind the left edge of its overflow-hidden track, so both caps
 *  stay round) into place when its track scrolls into view; `data-bar-delay` (ms) staggers the
 *  tuned bar after the default one.
 *
 *  The final width's source of truth in markup is a `.bar--pNN` class (motion.css), which works in
 *  every engine with no JS. `data-bar-to` (0..1), when present, is written through the CSSOM so a
 *  runtime value (a live measurement in the app) can be finer than the 1% class steps. */
export function initComparisonBars(root: ParentNode = document, selector = "[data-bar]"): () => void {
  const elements = queryAll<HTMLElement>(root, selector).filter((el) => !el.classList.contains("is-visible") && !isArmed(el));
  if (elements.length === 0) return () => {};

  const apply = (el: HTMLElement) => {
    const raw = el.dataset["barTo"];
    const to = raw === undefined || raw === "" ? NaN : Number(raw);
    if (Number.isFinite(to)) el.style.setProperty("--bar-to", String(clamp01(to)));
    const delay = Number(el.dataset["barDelay"] ?? "0");
    el.style.setProperty("--bar-delay", `${Number.isFinite(delay) && delay > 0 ? Math.min(delay, 1000) : 0}ms`);
    el.classList.add("is-visible");
  };

  if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
    for (const el of elements) apply(el);
    return () => {};
  }

  // Observe the track, not the bar: an armed bar sits translated out of its track's clip, so its
  // own intersection says nothing about whether the figure is on screen.
  const barsByTrack = new Map<Element, HTMLElement[]>();
  for (const el of elements) {
    arm(el);
    const track = el.parentElement ?? el;
    const list = barsByTrack.get(track) ?? [];
    list.push(el);
    barsByTrack.set(track, list);
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const el of barsByTrack.get(entry.target) ?? []) apply(el);
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.4 },
  );
  for (const track of barsByTrack.keys()) observer.observe(track);
  return () => {
    observer.disconnect();
    for (const el of elements) disarm(el);
  };
}

/** Changes a bar's value live (a progress fill during a run, a measurement that just updated): the
 *  fill's right end slides from the old value to the new one while its left end stays on the track
 *  start, with a transform-only Web Animation (allowed by the strict CSP, unlike inline style
 *  transitions). Under reduced motion it simply jumps. `to` is clamped to 0..1; returns it. */
export function setBarValue(el: HTMLElement, to: number): number {
  const next = clamp01(Number.isFinite(to) ? to : 0);
  const prev = clamp01(Number.parseFloat(getComputedStyle(el).getPropertyValue("--bar-to")) || 0);
  for (const running of el.getAnimations()) if (!(running instanceof CSSTransition)) running.cancel();
  if (prefersReducedMotion() || prev === next) {
    el.style.setProperty("--bar-to", String(next));
    return next;
  }
  const root = getComputedStyle(document.documentElement);
  const timing = {
    duration: Number.parseFloat(root.getPropertyValue("--duration-slow")) || 600,
    easing: root.getPropertyValue("--ease-emphasized").trim() || "ease-out",
  };
  if (next > prev) {
    // Growing: the box takes its new length now and starts shifted left so its right end sits
    // where the old value ended (the hidden part is behind the track's clip).
    el.style.setProperty("--bar-to", String(next));
    el.animate([{ transform: `translateX(${(prev / next - 1) * 100}%)` }, { transform: "translateX(0)" }], timing);
  } else {
    // Shrinking: keep the old length while it slides left, then take the new length at the end, in
    // the same frame the animation is removed, so the left end never leaves the track start.
    const anim = el.animate([{ transform: "translateX(0)" }, { transform: `translateX(${(next / prev - 1) * 100}%)` }], { ...timing, fill: "forwards" });
    anim.addEventListener("finish", () => {
      el.style.setProperty("--bar-to", String(next));
      anim.cancel();
    });
  }
  return next;
}

// ---------------------------------------------------------------------------
// Wire everything at once
// ---------------------------------------------------------------------------

// `motion-ready` and the button guards are document-wide, so several initMotion() callers (one per
// SolidJS component, or a repeated Astro page-load hook) share them by reference count.
let readyRefs = 0;
const guardRefs = new WeakMap<Document, { refs: number; off: () => void }>();

function acquireReady(): void {
  if (readyRefs++ === 0) document.documentElement.classList.add("motion-ready");
}

function releaseReady(): void {
  if (readyRefs > 0 && --readyRefs === 0) document.documentElement.classList.remove("motion-ready");
}

function acquireGuards(doc: Document): () => void {
  const entry = guardRefs.get(doc) ?? { refs: 0, off: () => {} };
  if (entry.refs++ === 0) entry.off = initButtonGuards(doc);
  guardRefs.set(doc, entry);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    if (--entry.refs === 0) entry.off();
  };
}

/** If the CSS failsafe already un-hid the page (the module arrived after FAILSAFE_MS), content in
 *  view is treated as already arrived, so it doesn't vanish and replay. */
function settleIfLate(root: ParentNode): void {
  const html = document.documentElement;
  if (!html.classList.contains("motion-pending") || performance.now() < FAILSAFE_MS) return;
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-reveal], [data-bar]"))) {
    if (inViewport(el)) el.classList.add("is-visible");
  }
  for (const el of Array.from(root.querySelectorAll("[data-draw]"))) {
    if (inViewport(el)) el.classList.add("is-drawn");
  }
  for (const el of Array.from(root.querySelectorAll("[data-count-to]"))) {
    if (inViewport(el)) counted.add(el);
  }
}

export interface MotionOptions {
  /**
   * Also wire elements added under `root` later (a SolidJS `<For>` list, chat messages, a route
   * change), with one MutationObserver that `cleanup()` disconnects. Without it, later markup is
   * simply never armed and renders final, never stuck hidden.
   */
  watch?: boolean;
}

/** Every attribute a primitive in wireAll() wires. Watch mode skips added nodes with none of them
 *  (most streamed chat text), so an insertion costs one selector match, not six queries. */
const MOTION_SELECTOR = "[data-reveal],[data-count-to],[data-draw],[data-magnetic],[data-press],[data-bar],[data-sweep]";

function hasMotion(node: Element): boolean {
  return node.matches(MOTION_SELECTOR) || node.querySelector(MOTION_SELECTOR) !== null;
}

/** Every primitive for one subtree. Each init skips elements a live init already owns. */
function wireAll(root: ParentNode): () => void {
  applyLightSweep(root);
  const cleanups = [
    initScrollReveal(root),
    initCountUps(root),
    initDrawOnReveal(root),
    initMagneticAll(root),
    initPressSpring(root),
    initComparisonBars(root),
  ];
  return () => cleanups.forEach((fn) => fn());
}

/**
 * Wires every primitive under `root`, then adds `motion-ready` (and drops `motion-pending`) in the
 * same task, so the first frame after un-hiding already shows each primitive's start state.
 * Re-runs when the reduced-motion preference changes.
 */
export function initMotion(root: ParentNode = document, options: MotionOptions = {}): () => void {
  const doc = root instanceof Document ? root : (root as Node).ownerDocument ?? document;
  const releaseGuards = acquireGuards(doc);
  settleIfLate(root);
  let cleanups: Array<() => void> = [];
  // Watch mode: one cleanup per added subtree root, so a subtree that leaves the DOM (a deleted or
  // re-rendered chat message) releases its listeners and observer targets instead of living for
  // the whole session.
  const subtrees = new Map<Element, () => void>();
  const dropSubtrees = () => {
    subtrees.forEach((off) => off());
    subtrees.clear();
  };
  let holdingReady = false;

  const setup = (reduced: boolean) => {
    cleanups.forEach((fn) => fn());
    // The full re-wire below covers every subtree already in the DOM.
    dropSubtrees();
    cleanups = [wireAll(root)];
    if (!reduced && !holdingReady) {
      acquireReady();
      holdingReady = true;
    } else if (reduced && holdingReady) {
      releaseReady();
      holdingReady = false;
    }
    document.documentElement.classList.remove("motion-pending");
  };

  const unsubscribe = onReducedMotionChange(setup);

  // Mutation callbacks run as a microtask after the insertion, before the next paint, so an added
  // element is armed before it can show its final state and then vanish.
  let watcher: MutationObserver | null = null;
  if (options.watch && typeof MutationObserver !== "undefined") {
    watcher = new MutationObserver((records) => {
      // Connection is judged once per batch, after every record: a node moved within one batch
      // (removed, then re-added) is still connected and keeps its wiring.
      let removed = false;
      const added = new Set<Element>();
      for (const record of records) {
        if (record.removedNodes.length) removed = true;
        for (const node of Array.from(record.addedNodes)) {
          if (node instanceof Element) added.add(node);
        }
      }
      if (removed) {
        for (const [el, off] of subtrees) {
          if (!el.isConnected) {
            off();
            subtrees.delete(el);
          }
        }
      }
      for (const node of added) {
        if (!node.isConnected || subtrees.has(node) || !hasMotion(node)) continue;
        subtrees.set(node, wireAll(node));
      }
    });
    watcher.observe(root as unknown as Node, { childList: true, subtree: true });
  }

  return () => {
    watcher?.disconnect();
    watcher = null;
    unsubscribe();
    releaseGuards();
    dropSubtrees();
    cleanups.forEach((fn) => fn());
    cleanups = [];
    if (holdingReady) releaseReady();
    holdingReady = false;
  };
}
