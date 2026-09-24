#!/usr/bin/env node
// @ts-check
/**
 * `npm run refs` — captures full-viewport 1440x900 headless reference screenshots of five
 * best-in-class marketing sites for visual comparison only, into design/review/refs/ (gitignored:
 * these are someone else's copyrighted pages, kept locally for our own review, never shipped).
 *
 * This is the one script in design/ that is allowed to make real network requests.
 */
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const OUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "review", "refs");

const SITES = [
  { name: "anthropic", url: "https://www.anthropic.com" },
  { name: "linear", url: "https://linear.app" },
  { name: "raycast", url: "https://raycast.com" },
  { name: "vercel", url: "https://vercel.com" },
  { name: "stripe", url: "https://stripe.com" },
];

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const browser = await chromium.launch();
  try {
    for (const site of SITES) {
      // `reducedMotion: "reduce"` (a real browser context flag, distinct from
      // `page.screenshot({ animations: "disabled" })` below) is the one thing that stops a site's
      // OWN intro animation from running at all: `waitUntil: "load"` plus a fixed timeout only
      // waits for the page to finish LOADING, which says nothing about whether an entrance
      // animation that started on load has finished PLAYING by the time the timeout elapses — an
      // earlier capture of linear.app landed mid hero-intro-animation, blurred and unusable for
      // comparison, for exactly this reason.
      const context = await browser.newContext({
        viewport: { width: 1440, height: 900 },
        reducedMotion: "reduce",
      });
      const page = await context.newPage();
      try {
        // `networkidle` (no more than 2 in-flight requests for 500ms) is a materially stronger
        // signal than `load` that a JS-heavy marketing site has actually finished fetching the
        // assets its hero needs, not just fired the `load` event while a hero image or web font is
        // still in flight.
        await page.goto(site.url, { waitUntil: "networkidle", timeout: 45_000 });
        // Waits for every @font-face the page requested to finish loading — a hero captured before
        // its display font swaps in shows the fallback system font, at different metrics, which
        // reads as a different (and wrong) layout entirely.
        await page.evaluate(() => document.fonts.ready);
        // A fixed settle time beyond `networkidle`/fonts.ready still matters: `reducedMotion` skips
        // most CSS/Web Animations transitions, but some sites drive their intro with rAF-stepped
        // JS instead, which reduced-motion does not touch. Three seconds is enough for those to
        // reach whatever their own final/idle frame is.
        await page.waitForTimeout(3000);
        const file = path.join(OUT_DIR, `${site.name}.png`);
        // `animations: "disabled"` (Playwright finishes any CSS transition/animation still running
        // at capture time instantly, rather than mid-frame) is the last line of defense for
        // anything the two waits above didn't fully settle.
        await page.screenshot({ path: file, fullPage: false, animations: "disabled" });
        console.log("Saved", file);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.error(`Failed to capture ${site.name} (${site.url}):`, message);
      } finally {
        await context.close();
      }
    }
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
