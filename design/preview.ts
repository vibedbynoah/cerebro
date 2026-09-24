/**
 * Behaviour for the design specimen page (preview.html): theme toggle, live contrast labels
 * (computed from actually-rendered pixels, not asserted numbers), live token-value labels, the
 * motion-token picker, the code-block copy button, and wiring every shared motion primitive
 * from motion.ts.
 *
 * Every DOM write here uses the CSSOM (`el.style.setProperty`) or `textContent`/attributes —
 * never `setAttribute("style", ...)` or innerHTML — so this page keeps working unmodified under
 * the strict CSP declared in preview.html's `<meta>` tag (style-src 'self', no 'unsafe-inline').
 */
import { initMotion, onReducedMotionChange, prefersReducedMotion } from "./motion.js";
import { initTerms } from "./terms.js";
import { contrastOf, getRenderedColor } from "./contrast.js";

type Theme = "dark" | "light";

const THEME_STORAGE_KEY = "cerebro-theme";

/** Reads the visitor's own EXPLICIT prior choice, if any — never the OS default, which is
 *  intentionally NOT persisted so it keeps following live OS changes (see initThemeToggle). */
function storedTheme(): Theme | null {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    return v === "dark" || v === "light" ? v : null;
  } catch {
    return null; // private browsing / blocked storage: fall through to the OS preference
  }
}

function osTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function currentTheme(): Theme {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "dark" || attr === "light") return attr;
  return osTheme();
}

function setTheme(theme: Theme, persist: boolean): void {
  document.documentElement.setAttribute("data-theme", theme);
  const button = document.querySelector<HTMLButtonElement>("[data-theme-toggle]");
  // A real ARIA toggle button: the accessible name ("Dark theme", from the .sr-only child in
  // preview.html) names the FEATURE, not the action, and never changes; aria-pressed carries the
  // on/off state. This also sidesteps the previous bug where the visible label was hard-coded
  // "Switch to light" even when the OS (and therefore the actually-rendered page) preferred
  // light, until JS ran a tick later and corrected it.
  if (button) button.setAttribute("aria-pressed", String(theme === "dark"));
  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Storage blocked/full: the toggle still works for this page view, it just won't persist.
    }
  }
  labelContrastPairs();
  labelTokenValues();
}

function initThemeToggle(): void {
  const button = document.querySelector<HTMLButtonElement>("[data-theme-toggle]");
  if (!button) return;

  const stored = storedTheme();
  // Only an explicit click (or a previously-stored explicit click) pins the theme. Until that
  // happens, the page keeps following the OS preference live — a visitor who never touched the
  // toggle and then changes their OS theme sees this page follow along, rather than being stuck
  // on whatever was true at load (the previous behaviour, since data-theme was set unconditionally
  // on every load regardless of whether the visitor had ever chosen anything).
  let explicit = stored !== null;
  setTheme(stored ?? currentTheme(), false);

  // Clicking back to the theme the OS already prefers returns the page to "follow system" (the
  // stored choice is cleared), so an explicit pick is never a one-way door.
  button.addEventListener("click", () => {
    const next: Theme = currentTheme() === "dark" ? "light" : "dark";
    if (next === osTheme()) {
      try {
        localStorage.removeItem(THEME_STORAGE_KEY);
      } catch {
        // Storage blocked: nothing was stored either.
      }
      explicit = false;
      setTheme(next, false);
    } else {
      explicit = true;
      setTheme(next, true);
    }
  });

  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    // osTheme(), not currentTheme(): once set, data-theme would just hand back the old theme.
    if (!explicit) setTheme(osTheme(), false);
  });
}

/** Labels every `[data-contrast-fg][data-contrast-bg]` element with its live, rendered ratio. */
function labelContrastPairs(root: ParentNode = document): void {
  const styles = getComputedStyle(document.documentElement);
  const elements = Array.from(root.querySelectorAll<HTMLElement>("[data-contrast-fg][data-contrast-bg]"));
  for (const el of elements) {
    const fgVar = el.dataset["contrastFg"];
    const bgVar = el.dataset["contrastBg"];
    if (!fgVar || !bgVar) continue;
    const min = Number(el.dataset["contrastMin"] ?? "4.5");
    const fg = styles.getPropertyValue(fgVar).trim();
    const bg = styles.getPropertyValue(bgVar).trim();
    if (!fg || !bg) continue;
    const ratio = contrastOf(fg, bg);
    el.textContent = `${ratio.toFixed(2)}:1 vs ${min}:1 required`;
    el.dataset["pass"] = String(ratio >= min);
  }
}

function toHex(n: number): string {
  return Math.round(n).toString(16).padStart(2, "0");
}

/** Top-level comma count + 1: the number of layers in a shadow list. */
function countLayers(value: string): number {
  let depth = 0;
  let layers = 1;
  for (const c of value) {
    if (c === "(") depth++;
    else if (c === ")") depth--;
    else if (c === "," && depth === 0) layers++;
  }
  return layers;
}

/** Labels every `[data-token]` element with its value as rendered in the current theme:
 *  - a colour shows its hex (with alpha), the raw light-dark() source collapsed below it;
 *  - `data-token-kind="list"` (a shadow or ring) shows its layer count, source collapsed below;
 *  - anything else (a font stack) shows its value as written. */
function labelTokenValues(root: ParentNode = document): void {
  const styles = getComputedStyle(document.documentElement);
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-token]"))) {
    const token = el.dataset["token"];
    if (!token) continue;
    const raw = styles.getPropertyValue(token).trim().replace(/\s+/g, " ");
    if (!raw) continue;
    let rendered: string | null = null;
    if (el.dataset["tokenKind"] === "list") {
      const n = countLayers(raw);
      rendered = `${n} ${n === 1 ? "layer" : "layers"}`;
    } else {
      // getRenderedColor ignores an unparseable value (the probe keeps its last colour), so only
      // a value the engine accepts as a colour is resolved.
      if (CSS.supports("color", raw)) {
        const { r, g, b, a } = getRenderedColor(raw);
        rendered = a < 1 ? `#${toHex(r)}${toHex(g)}${toHex(b)} / ${a.toFixed(2)}` : `#${toHex(r)}${toHex(g)}${toHex(b)}`;
      }
    }
    const resolvedEl = document.createElement("span");
    resolvedEl.className = "swatch__value-resolved";
    if (rendered === null) {
      // Nothing to resolve: the value itself, wrapping, with no duplicate source.
      resolvedEl.classList.add("swatch__value-resolved--wrap");
      resolvedEl.textContent = raw;
      el.replaceChildren(resolvedEl);
      continue;
    }
    resolvedEl.textContent = rendered;
    const wasOpen = el.querySelector("details")?.open ?? false;
    const details = document.createElement("details");
    details.className = "swatch__source";
    details.open = wasOpen;
    const summary = document.createElement("summary");
    summary.textContent = "source";
    const code = document.createElement("code");
    code.textContent = raw;
    details.append(summary, code);
    el.replaceChildren(resolvedEl, details);
  }
}

/** The duration/easing token picker on the motion-tokens demo rail, plus a Play button so
 *  keyboard and touch users (not just mouse-hover on the rail) can run the demo.
 *
 *  Both `<select>`s are rebuilt from the ACTUAL computed token values (`getComputedStyle` on
 *  `:root`) rather than left as the hardcoded `<option value="...">` markup in preview.html: a
 *  future edit to a `--duration-*`/`--ease-*` token in tokens.css would otherwise silently
 *  desynchronize from this picker's hardcoded copies, so the demo would keep advertising a value
 *  the token no longer actually holds. The hardcoded options stay in the markup as the no-JS/
 *  pre-hydration fallback (they match today's tokens); this function's `replaceChildren` calls
 *  overwrite them for every JS-enabled visitor with values that can never drift. */
function initMotionTokenPicker(): void {
  const demo = document.querySelector<HTMLElement>("[data-motion-demo]");
  if (!demo) return;
  const track = demo.querySelector<HTMLElement>("[data-motion-track]");
  const durationSelect = document.querySelector<HTMLSelectElement>("[data-motion-duration]");
  const easingSelect = document.querySelector<HTMLSelectElement>("[data-motion-easing]");
  const playButton = document.querySelector<HTMLButtonElement>("[data-motion-play]");
  const rootStyles = getComputedStyle(document.documentElement);

  if (durationSelect) {
    durationSelect.replaceChildren();
    for (const name of ["instant", "fast", "base", "slow"]) {
      const value = rootStyles.getPropertyValue(`--duration-${name}`).trim();
      durationSelect.add(new Option(`--duration-${name} (${value})`, value, name === "base", name === "base"));
    }
  }
  if (easingSelect) {
    easingSelect.replaceChildren();
    for (const name of ["standard", "emphasized", "spring", "exit"]) {
      const value = rootStyles.getPropertyValue(`--ease-${name}`).trim();
      easingSelect.add(new Option(`--ease-${name}`, value, name === "standard", name === "standard"));
    }
  }

  // The chip's full travel distance is the rail's own rendered width (see `.motion-demo__chip` in
  // preview.css) — measured, not assumed, so it can never drift from what's actually on screen at
  // the current viewport width, and re-measured on resize since the rail is fluid-width.
  const updateTrackWidth = () => {
    if (track) demo.style.setProperty("--track-w", `${track.clientWidth}px`);
  };
  updateTrackWidth();
  // A ResizeObserver, not window "resize": the rail also changes width when fonts swap in or the
  // grid reflows around the selects, with no window resize at all.
  if (track && typeof ResizeObserver !== "undefined") new ResizeObserver(updateTrackWidth).observe(track);
  else window.addEventListener("resize", updateTrackWidth);

  const apply = () => {
    if (durationSelect) demo.style.setProperty("--motion-demo-duration", durationSelect.value);
    if (easingSelect) demo.style.setProperty("--motion-demo-ease", easingSelect.value);
  };
  durationSelect?.addEventListener("change", apply);
  easingSelect?.addEventListener("change", apply);
  apply();

  // Play is an action, not a toggle: one there-and-back run with the selected duration on each leg
  // and the selected easing, via the Web Animations API (not governed by style-src, so the strict
  // CSP holds). Hovering the rail itself stays the CSS preview.
  //
  // Under reduced motion the control stays useful without movement: it jumps the chip between its
  // start and end state (no transition) and says so in a role="status" hint, and its label names
  // what it does. Both follow the preference live.
  const chip = track?.querySelector<HTMLElement>(".motion-demo__chip") ?? null;
  const hint = document.getElementById("motion-hint");
  const setHint = (text: string) => {
    if (hint) hint.textContent = text;
  };
  const REDUCED_HINT = "Animations are off because your system prefers reduced motion, so this shows the end state instead of the movement.";
  onReducedMotionChange((reduced) => {
    chip?.classList.remove("is-at-end");
    if (playButton) playButton.textContent = reduced ? "Show end state" : "Play";
    setHint(reduced ? REDUCED_HINT : "");
  });

  playButton?.addEventListener("click", () => {
    if (!track || !chip) return;
    if (prefersReducedMotion()) {
      for (const running of chip.getAnimations()) running.cancel();
      const atEnd = chip.classList.toggle("is-at-end");
      playButton.textContent = atEnd ? "Show start state" : "Show end state";
      setHint(atEnd ? "Reduced motion: the chip is at its end state." : "Reduced motion: the chip is back at its start state.");
      return;
    }
    const dx = Math.max(0, track.clientWidth - 24);
    const legMs = Number.parseFloat(durationSelect?.value ?? "320") || 320;
    const easing = easingSelect?.value || "ease";
    // Restart cleanly on a second click; leave the hover transition to CSS.
    for (const running of chip.getAnimations()) if (!(running instanceof CSSTransition)) running.cancel();
    chip.animate(
      [
        { transform: "translateX(0)", easing },
        { transform: `translateX(${dx}px)`, offset: 0.45 },
        { transform: `translateX(${dx}px)`, offset: 0.55, easing },
        { transform: "translateX(0)" },
      ],
      { duration: (legMs * 2) / 0.9 },
    );
  });
}

function initReducedMotionBadge(): void {
  const badge = document.querySelector<HTMLElement>("[data-reduced-motion-badge]");
  if (!badge) return;
  onReducedMotionChange((reduced) => {
    badge.textContent = reduced ? "prefers-reduced-motion: reduce (animations off)" : "prefers-reduced-motion: no-preference";
    badge.dataset["reduced"] = String(reduced);
  });
}

/** Copy-to-clipboard button on the code-block component. Strips the `>` prompt glyphs (they're
 *  `aria-hidden` decoration, not part of the commands) before writing to the clipboard.
 *
 *  The status region (`role="status"`, an implicit `aria-live="polite"` region) starts and stays
 *  EMPTY except for the ~1.8s after a result: a live region announces content CHANGES, and the
 *  previous version held the literal text "Copied" at all times and only toggled its opacity, so a
 *  screen reader had nothing to announce on an actual successful copy — and a clipboard failure was
 *  swallowed with no feedback of any kind. Writing, then clearing, real text on every result (both
 *  success and failure) is what makes this audible, not just visible. */
function initCodeBlockCopy(): void {
  for (const button of Array.from(document.querySelectorAll<HTMLButtonElement>("[data-copy]"))) {
    const block = button.closest(".code-block");
    const code = block?.querySelector("code");
    const status = block?.querySelector<HTMLElement>("[data-copy-status]");
    if (!code) continue;

    let clearTimer = 0;
    const announce = (message: string, success: boolean) => {
      if (status) {
        // Clearing then re-setting on the next frame (rather than just overwriting) guarantees a
        // screen reader treats two consecutive identical results (copy succeeds twice in a row) as
        // two separate announcements, not a no-op text assignment it might coalesce away.
        status.textContent = "";
        requestAnimationFrame(() => {
          status.textContent = message;
        });
      }
      button.classList.toggle("is-done", success);
      window.clearTimeout(clearTimer);
      clearTimer = window.setTimeout(() => {
        if (status) status.textContent = "";
        button.classList.remove("is-done");
      }, 1800);
    };

    button.addEventListener("click", async () => {
      const prompts = Array.from(code.querySelectorAll(".code-block__prompt"));
      const text = prompts.length
        ? code.textContent?.replace(/^\s*>\s?/gm, "").trim()
        : code.textContent?.trim();
      try {
        await navigator.clipboard.writeText(text ?? "");
        announce("Copied", true);
      } catch {
        // Clipboard access can be denied (permissions, insecure context) — this is now told to the
        // visitor instead of failing silently with no feedback at all.
        announce("Copy failed, select the text manually", false);
      }
    });
  }
}

/**
 * Positions the mobile pipeline connector (`.pipeline__connector--v`) so it runs exactly from the
 * centre of the FIRST `.pipeline__n` circle to the centre of the LAST one — measured from the real,
 * rendered layout rather than assumed from a fixed offset, because the four pipeline nodes are not
 * all the same height (nodes 2 and 4 carry a `.pipeline__sub` line, 1 and 3 don't), so a static
 * `top`/`height` in CSS can't stay centred on every circle at once the way a live measurement can.
 * Only does anything under the mobile (<= 40rem) layout the connector is actually shown in;
 * otherwise it clears its inline overrides so a resize back to desktop doesn't leave a stale value
 * behind. Re-run on resize (debounced to one measurement per animation frame, since `resize` can
 * fire many times during a drag) and once after fonts finish loading, since a late font swap can
 * shift text-driven node heights after the very first measurement.
 */
/** `el`'s untransformed box relative to `ancestor` (which must be positioned), summed along the
 *  offsetParent chain. Null when `ancestor` is not on that chain. */
function layoutBox(el: HTMLElement, ancestor: HTMLElement): { top: number; left: number; width: number; height: number } | null {
  let top = 0;
  let left = 0;
  let node: HTMLElement | null = el;
  while (node && node !== ancestor) {
    top += node.offsetTop + (node === el ? 0 : node.clientTop);
    left += node.offsetLeft + (node === el ? 0 : node.clientLeft);
    node = node.offsetParent as HTMLElement | null;
  }
  if (node !== ancestor) return null;
  return { top, left, width: el.offsetWidth, height: el.offsetHeight };
}

function positionPipelineConnector(): void {
  const wrap = document.querySelector<HTMLElement>(".pipeline-wrap");
  const connector = document.querySelector<HTMLElement>(".pipeline__connector--v");
  const nodes = Array.from(document.querySelectorAll<HTMLElement>(".pipeline__n"));
  if (!wrap || !connector || nodes.length < 2) return;

  const isMobile = window.matchMedia("(max-width: 40rem)").matches;
  if (!isMobile) {
    connector.style.removeProperty("--pipeline-v-top");
    connector.style.removeProperty("--pipeline-v-left");
    connector.style.removeProperty("--pipeline-v-height");
    return;
  }

  // Layout boxes, never getBoundingClientRect(): the nodes pop in (`.pop-in` scales each <li> about
  // its own centre), and a measurement taken mid-animation would pin the connector off the circles.
  const first = layoutBox(nodes[0]!, wrap);
  const last = layoutBox(nodes[nodes.length - 1]!, wrap);
  if (!first || !last) return;
  const firstCenterY = first.top + first.height / 2;
  const lastCenterY = last.top + last.height / 2;
  const firstCenterX = first.left + first.width / 2;
  const connectorWidth = 4;

  connector.style.setProperty("--pipeline-v-top", `${firstCenterY}px`);
  connector.style.setProperty("--pipeline-v-left", `${firstCenterX - connectorWidth / 2}px`);
  connector.style.setProperty("--pipeline-v-height", `${Math.max(0, lastCenterY - firstCenterY)}px`);
}

function initPipelineConnector(): void {
  positionPipelineConnector();
  let frame = 0;
  const scheduleReposition = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      positionPipelineConnector();
    });
  };
  window.addEventListener("resize", scheduleReposition);
  void document.fonts.ready.then(scheduleReposition);
}

function init(): void {
  initThemeToggle();
  initMotionTokenPicker();
  initReducedMotionBadge();
  initCodeBlockCopy();
  initPipelineConnector();
  labelContrastPairs();
  labelTokenValues();
  initTerms(document);
  initMotion(document);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
