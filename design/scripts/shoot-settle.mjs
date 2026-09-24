// @ts-check
/**
 * Brings the specimen to its finished, default (full-motion) state the way a visitor does: a real
 * scroll pass in viewport steps, so every IntersectionObserver-driven primitive (reveal, count-up,
 * draw-on, comparison bars) fires on its own trigger, then a wait for the longest transition.
 * Shared by `npm run shoot` and the Playwright smoke test that keeps it honest.
 */

/** Longest one-shot motion (--duration-count / --duration-draw, 900ms) + the capped reveal stagger
 *  (360ms) + the tuned bar's delay (120ms), rounded up. */
const SETTLE_MS = 1500;

/** @param {import("@playwright/test").Page} page */
export async function settleFullMotion(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(async () => {
    document.documentElement.style.setProperty("scroll-behavior", "auto");
    const step = Math.max(200, Math.round(window.innerHeight * 0.6));
    for (let y = 0; y < document.documentElement.scrollHeight; y += step) {
      window.scrollTo(0, y);
      await new Promise((r) => setTimeout(r, 180));
    }
    window.scrollTo(0, document.documentElement.scrollHeight);
    await new Promise((r) => setTimeout(r, 180));
    window.scrollTo(0, 0);
  });
  await page.waitForTimeout(SETTLE_MS);
}

/**
 * Reports anything not in its final state. With `strict`, throws instead, so a screenshot of a
 * half-finished page can never be written.
 * @param {import("@playwright/test").Page} page
 * @param {boolean} strict
 * @param {boolean} [reduced] the badge text expected for this capture
 */
export async function assertSettled(page, strict, reduced = false) {
  const report = await page.evaluate(() => {
    const hidden = Array.from(document.querySelectorAll("[data-reveal]"))
      .filter((el) => !el.classList.contains("is-visible") || getComputedStyle(el).opacity !== "1")
      .map((el) => `${el.tagName.toLowerCase()}.${el.className}`);
    // The animating digits live in the aria-hidden child (the .sr-only twin always holds the final).
    const shown = (/** @type {Element} */ el) => (el.querySelector(':scope > [aria-hidden="true"]') ?? el).textContent?.trim();
    const unfinishedCounts = Array.from(document.querySelectorAll(".count[data-count-to]"))
      .filter((el) => shown(el) !== el.getAttribute("data-count-to"))
      .map((el) => `${shown(el)} != ${el.getAttribute("data-count-to")}`);
    // A drawn stroke has no dash left (.is-draw-done, or reduced motion), so it can't be truncated.
    const undrawn = Array.from(document.querySelectorAll("[data-draw]"))
      .filter((el) => el.getClientRects().length > 0 && getComputedStyle(el).strokeDasharray !== "none")
      .map((el) => `${el.tagName.toLowerCase()}.${el.getAttribute("class") ?? ""} dasharray=${getComputedStyle(el).strokeDasharray}`);
    const badge = document.querySelector("[data-reduced-motion-badge]")?.textContent?.trim() ?? "";
    return { hidden, unfinishedCounts, undrawn, badge };
  });
  const expectedBadge = reduced ? "prefers-reduced-motion: reduce (animations off)" : "prefers-reduced-motion: no-preference";
  if (strict && (report.hidden.length || report.unfinishedCounts.length || report.undrawn.length || report.badge !== expectedBadge)) {
    throw new Error(`specimen not settled before capture: ${JSON.stringify(report)}`);
  }
  return report;
}
