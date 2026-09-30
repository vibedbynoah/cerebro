#!/usr/bin/env node
// @ts-check
/**
 * `npm run shoot` — headless screenshots of every screen of the web-mode build (ui/dist) in dark
 * and light at 1280x800, into review/shots/ (gitignored). Each screen is first scrolled through so
 * its scroll reveals and count-ups fire, then returned to the top and captured once nothing is
 * animating. Two extra states per theme: the command palette open, and the collapsed sidebar.
 *
 * Headless only (nothing appears on screen); every request that leaves the origin is aborted and
 * fails the run, same as the Playwright suite.
 */
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(HERE, "../review/shots");
const ROUTES = ["home", "hub", "models", "tune", "chat", "images", "documents", "plugins", "tools", "security", "settings"];
/** @type {Array<"dark" | "light">} */
const THEMES = ["dark", "light"];

/** @param {import("@playwright/test").Page} page */
async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(async () => {
    const main = document.querySelector("main");
    if (!main) return;
    const step = Math.max(200, main.clientHeight * 0.6);
    for (let y = 0; y <= main.scrollHeight; y += step) {
      main.scrollTo({ top: y });
      await new Promise((r) => setTimeout(r, 90));
    }
    main.scrollTo({ top: 0 });
  });
  // Infinite loops (a spinner, the live-dot pulse) are fast-forwarded by the screenshot itself.
  await page
    .waitForFunction(
      () => document.getAnimations().every((a) => a.playState !== "running" || a.effect?.getComputedTiming().iterations === Infinity),
      undefined,
      { timeout: 6000 },
    )
    .catch(() => console.warn("  (some motion was still running; captured anyway)"));
  await page.waitForTimeout(150);
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const { url, close } = await startServer(0);
  const origin = new URL(url).origin;
  /** @type {string[]} */
  const offOrigin = [];
  /** @type {string[]} */
  const problems = [];
  const browser = await chromium.launch({ headless: true });
  try {
    for (const theme of THEMES) {
      const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, colorScheme: theme, reducedMotion: "no-preference" });
      await context.route("**/*", (route) => {
        const u = route.request().url();
        if (u.startsWith("data:") || u.startsWith("blob:") || new URL(u).origin === origin) return route.continue();
        offOrigin.push(u);
        return route.abort("blockedbyclient");
      });
      const page = await context.newPage();
      page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && problems.push(`${theme}: ${m.text()}`));
      page.on("pageerror", (e) => problems.push(`${theme}: ${e.message}`));

      for (const id of ROUTES) {
        await page.goto(`${url}/#/${id}`);
        await page.locator("main h1").waitFor();
        await page.locator(".statusbar [data-hud='speed']").waitFor();
        await settle(page);
        const file = path.join(OUT_DIR, `${id}-${theme}.png`);
        await page.screenshot({ path: file, animations: "disabled" });
        console.log("Saved", file);
      }

      // Command palette open, with a query typed.
      await page.goto(`${url}/#/home`);
      await page.locator("main h1").waitFor();
      await settle(page);
      await page.keyboard.press("Control+k");
      await page.getByRole("combobox").waitFor();
      await page.keyboard.type("mo");
      await page.waitForTimeout(400);
      await page.screenshot({ path: path.join(OUT_DIR, `palette-${theme}.png`), animations: "disabled" });
      console.log("Saved", path.join(OUT_DIR, `palette-${theme}.png`));
      await page.keyboard.press("Escape");

      // Collapsed sidebar on the Models screen.
      await page.goto(`${url}/#/models`);
      await page.locator("main h1").waitFor();
      await page.keyboard.press("Control+b");
      await settle(page);
      await page.mouse.move(640, 400);
      await page.screenshot({ path: path.join(OUT_DIR, `collapsed-${theme}.png`), animations: "disabled" });
      console.log("Saved", path.join(OUT_DIR, `collapsed-${theme}.png`));
      await context.close();
    }
  } finally {
    await browser.close();
    await close();
  }
  if (offOrigin.length) throw new Error(`requests left the origin:\n${offOrigin.join("\n")}`);
  if (problems.length) throw new Error(`console problems:\n${problems.join("\n")}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
