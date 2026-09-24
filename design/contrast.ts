/**
 * Browser-only WCAG contrast helpers, shared by preview.ts (to label swatches with a live,
 * honest ratio) and design/tests/preview.spec.ts (to assert AA compliance).
 *
 * Resolving through a real, connected DOM element (rather than handing the raw string straight
 * to a canvas 2D context) is load-bearing, not stylistic: tokens.css declares every colour once
 * via `light-dark(LIGHT, DARK)`, and `getComputedStyle(root).getPropertyValue('--color-bg')`
 * returns that literal, UNEVALUATED text — custom properties store a token sequence, not a
 * resolved value, so `light-dark()` inside one is only ever evaluated once it is substituted
 * into a real CSS property on a real element that has a used `color-scheme`. A hidden probe
 * element appended to the document, with `color-scheme` inherited normally from `:root`/
 * `[data-theme]`, gives the browser everything it needs to resolve `light-dark()` (and any other
 * new colour syntax) exactly the way it resolves colour for the rest of the page — so the number
 * this file reports is always the actual rendered pixel, not a guess about what the token says.
 */

export interface RGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

let probeEl: HTMLElement | null = null;

function getProbe(): HTMLElement {
  if (typeof document === "undefined") {
    throw new Error("contrast.ts is browser-only (it resolves colour through a live DOM element) — it cannot run during SSR/build.");
  }
  if (probeEl && probeEl.isConnected) return probeEl;
  const el = document.createElement("span");
  el.setAttribute("aria-hidden", "true");
  el.style.setProperty("position", "fixed");
  el.style.setProperty("top", "-9999px");
  el.style.setProperty("left", "-9999px");
  el.style.setProperty("pointer-events", "none");
  document.body.appendChild(el);
  probeEl = el;
  return el;
}

let sharedCanvas: HTMLCanvasElement | null = null;

/** Rasterizes an already-concrete CSS colour string (no `light-dark()` or other
 *  context-dependent function left in it) to real 8-bit sRGB via a 1x1 canvas. This is the step
 *  that makes the result independent of how a given browser version chooses to *serialize* a
 *  computed colour — Chromium has serialized computed `color` as `rgb()`, and, more recently, in
 *  whatever colour function it was specified with (e.g. `oklch(0.52 0.02 75)`); the canvas always
 *  resolves either to the same real pixel. */
function rasterize(cssColor: string): RGBA {
  if (!sharedCanvas) sharedCanvas = document.createElement("canvas");
  sharedCanvas.width = 1;
  sharedCanvas.height = 1;
  const ctx = sharedCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas context unavailable");
  ctx.clearRect(0, 0, 1, 1);
  ctx.fillStyle = cssColor;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
  return { r: r ?? 0, g: g ?? 0, b: b ?? 0, a: (a ?? 255) / 255 };
}

/** Resolves any valid CSS colour string (oklch(), light-dark(), hex, rgb(), a custom property's
 *  literal value, ...) to the actual sRGB pixel the browser would paint. First flattens it
 *  through a real, connected element (see the probe helper above) so a context-dependent function
 *  like `light-dark()` picks the right branch for the current `color-scheme`; then rasterizes
 *  whatever concrete colour comes back. */
export function getRenderedColor(cssColor: string): RGBA {
  const el = getProbe();
  el.style.setProperty("color", cssColor);
  const resolved = getComputedStyle(el).color;
  return rasterize(resolved);
}

function channelToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/** WCAG relative luminance, 0 (black) .. 1 (white). */
export function relativeLuminance({ r, g, b }: RGBA): number {
  return 0.2126 * channelToLinear(r) + 0.7152 * channelToLinear(g) + 0.0722 * channelToLinear(b);
}

/** WCAG contrast ratio, 1 (no contrast) .. 21 (black on white). */
export function contrastRatio(a: RGBA, b: RGBA): number {
  const l1 = relativeLuminance(a);
  const l2 = relativeLuminance(b);
  const [hi, lo] = l1 > l2 ? [l1, l2] : [l2, l1];
  return (hi + 0.05) / (lo + 0.05);
}

/** Convenience: contrast ratio between two CSS colour strings, as actually rendered. */
export function contrastOf(cssColorA: string, cssColorB: string): number {
  return contrastRatio(getRenderedColor(cssColorA), getRenderedColor(cssColorB));
}

export interface OKLCH {
  l: number; // 0..1
  c: number; // chroma, typically 0..~0.4
  h: number; // hue, degrees, 0..360 (NaN for an achromatic colour — see below)
}

function srgbToLinear(c: number): number {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

/**
 * sRGB (0..255 per channel) -> OKLCH, via Björn Ottosson's OKLab. Used to check that the ACTUAL
 * rendered accent and danger colours stay far enough apart in hue/lightness for a colour-deficient
 * viewer, from the real painted pixel (through the same canvas probe `getRenderedColor` uses),
 * not from the declared token source — see design/tests/preview.spec.ts.
 */
export function toOklch({ r, g, b }: RGBA): OKLCH {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);

  const l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
  const m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
  const s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);

  const L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
  const a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
  const bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

  const c = Math.sqrt(a * a + bb * bb);
  // An (almost) achromatic colour has no meaningful hue angle; NaN signals "don't compare hue
  // here," which callers should treat as "compare lightness instead."
  const h = c < 0.0001 ? NaN : (Math.atan2(bb, a) * 180) / Math.PI;
  return { l: L, c, h: h < 0 ? h + 360 : h };
}

/** Convenience: resolves a CSS colour string to the OKLCH of its actual rendered pixel. */
export function oklchOf(cssColor: string): OKLCH {
  return toOklch(getRenderedColor(cssColor));
}

/** Smallest angle between two hue angles in degrees, 0..180. */
export function hueDelta(h1: number, h2: number): number {
  const d = Math.abs(h1 - h2) % 360;
  return d > 180 ? 360 - d : d;
}
