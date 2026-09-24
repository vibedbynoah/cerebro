#!/usr/bin/env node
// @ts-check
/**
 * `npm run shoot` — headless screenshots of the design specimen (preview.html) in dark and light,
 * at desktop (1440x900) and mobile (390x844), into design/review/shots/.
 *
 *  - preview-<theme>-<size>.png: the DEFAULT experience, full motion. The page is brought to its
 *    finished state by a real scroll pass (shoot-settle.mjs), so reveals, count-ups, draw-ons and
 *    bars each fire on their own trigger; capture refuses to run if anything is unfinished.
 *  - preview-<theme>-<size>-reduced.png: the prefers-reduced-motion path (every primitive snaps to
 *    its final state without moving).
 */
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { startServer, DESIGN_ROOT } from "./serve.mjs";
import { settleFullMotion, assertSettled } from "./shoot-settle.mjs";

const OUT_DIR = path.join(DESIGN_ROOT, "review", "shots");

const VIEWPORTS = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "mobile", width: 390, height: 844 },
];
/** @type {Array<"dark" | "light">} */
const THEMES = ["dark", "light"];

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const { url, close } = await startServer(0);
  const browser = await chromium.launch();
  try {
    for (const theme of THEMES) {
      for (const vp of VIEWPORTS) {
        for (const reduced of [false, true]) {
          const context = await browser.newContext({
            viewport: { width: vp.width, height: vp.height },
            colorScheme: theme,
            reducedMotion: reduced ? "reduce" : "no-preference",
          });
          const page = await context.newPage();
          await page.goto(`${url}/preview.html`, { waitUntil: "networkidle" });
          // colorScheme above means the page follows the OS theme with nothing stored, exactly as a
          // first-time visitor with that preference sees it.
          if (reduced) {
            await page.evaluate(() => document.fonts.ready);
            await page.waitForTimeout(100);
          } else {
            await settleFullMotion(page);
          }
          await assertSettled(page, true, reduced);
          const file = path.join(OUT_DIR, `preview-${theme}-${vp.name}${reduced ? "-reduced" : ""}.png`);
          // animations: "disabled" only fast-forwards anything still running (the busy spinner);
          // every one-shot motion has already finished above.
          await page.screenshot({ path: file, fullPage: true, animations: "disabled" });
          console.log("Saved", file);
          await context.close();
        }
      }
    }
  } finally {
    await browser.close();
    await close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
