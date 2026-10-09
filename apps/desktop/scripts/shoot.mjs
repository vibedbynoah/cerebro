#!/usr/bin/env node
// @ts-check
/**
 * `npm run shoot` — headless screenshots of every screen of the web-mode build (ui/dist) in dark
 * and light at 1280x800, into review/shots/ (gitignored). Each screen is first scrolled through so
 * its scroll reveals and count-ups fire, then returned to the top and captured once nothing is
 * animating. Two extra states per theme: the command palette open, and the collapsed sidebar.
 * Then the Models, Tune and Monitor screens' states (models-*, tune-*, monitor-*), and every first-run setup step
 * (onboarding-*), in a fresh profile that has never finished it,
 * driven by the keyboard the way a user would.
 *
 * Headless only (nothing appears on screen); every request that leaves the origin is aborted and
 * fails the run, same as the Playwright suite.
 */
import { chromium } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.resolve(HERE, "../review/shots");
const ROUTES = ["home", "hub", "models", "tune", "monitor", "turbo", "chat", "images", "documents", "plugins", "tools", "security", "settings", "about"];
/** @type {Array<"dark" | "light">} */
const THEMES = ["dark", "light"];
/** What a finished setup leaves in local storage. Key and terms version are read from the app's
 *  source (ui/src/onboarding/progress.ts), so a terms bump cannot leave this seeding a stale record. */
const PROGRESS_SRC = await readFile(path.resolve(HERE, "../ui/src/onboarding/progress.ts"), "utf8");
const constant = (/** @type {string} */ name) => {
  const m = new RegExp(`export const ${name} = "([^"]+)"`).exec(PROGRESS_SRC);
  if (!m?.[1]) throw new Error(`${name} not found in progress.ts`);
  return m[1];
};
const ONBOARDING_KEY = constant("ONBOARDING_KEY");
const DONE_RECORD = JSON.stringify({ v: 1, completed: true, termsVersion: constant("TERMS_VERSION"), model: "gemma-4-12b-it" });

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

/**
 * @param {import("@playwright/test").Browser} browser
 * @param {string} origin
 * @param {string} url
 * @param {"dark" | "light"} theme
 * @param {string[]} offOrigin
 * @param {string[]} problems
 */
async function shootOnboarding(browser, origin, url, theme, offOrigin, problems) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, colorScheme: theme, reducedMotion: "no-preference" });
  await context.route("**/*", (route) => {
    const u = route.request().url();
    if (u.startsWith("data:") || u.startsWith("blob:") || new URL(u).origin === origin) return route.continue();
    offOrigin.push(u);
    return route.abort("blockedbyclient");
  });
  const page = await context.newPage();
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && problems.push(`onboarding ${theme}: ${m.text()}`));
  page.on("pageerror", (e) => problems.push(`onboarding ${theme}: ${e.message}`));
  const shot = async (/** @type {string} */ name) => {
    await page.evaluate(() => document.fonts.ready);
    await page
      .waitForFunction(
        () => document.getAnimations().every((a) => a.playState !== "running" || a.effect?.getComputedTiming().iterations === Infinity),
        undefined,
        { timeout: 8000 },
      )
      .catch(() => console.warn("  (some motion was still running; captured anyway)"));
    await page.mouse.move(1270, 790);
    await page.waitForTimeout(150);
    const file = path.join(OUT_DIR, `onboarding-${name}-${theme}.png`);
    await page.screenshot({ path: file, animations: "disabled" });
    console.log("Saved", file);
  };
  const next = async (/** @type {string} */ step) => {
    await page.locator("[data-primary]").click();
    await page.locator(`.onb[data-step="${step}"]`).waitFor();
  };

  await page.goto(`${url}/`);
  await page.locator('.onb[data-step="welcome"]').waitFor();
  await shot("welcome");
  await next("level");
  await shot("level");
  await next("hardware");
  await page.locator('.scan-panel[data-state="done"]').waitFor({ timeout: 8000 });
  await shot("hardware");
  await next("model");
  await shot("model");
  await next("configure");
  await shot("configure-advanced");
  // The escape hatch: the same values in the Beginner view, then back.
  await page.locator(".onb-level-switch").click();
  await page.locator(".plan").waitFor();
  await shot("configure-beginner");
  await page.locator(".onb-level-switch").click();
  await page.locator(".adv").waitFor();
  await next("privacy");
  await shot("privacy");
  await next("terms");
  await shot("terms");
  await page.getByRole("button", { name: "Licenses" }).click();
  await page.locator(".legal .legal__license").first().waitFor();
  await shot("legal");
  await page.keyboard.press("Escape");
  await page.locator(".legal").waitFor({ state: "detached" });
  await next("finish");
  await shot("finish");

  // Off the happy path (the web build's mock failure switches): the hardware check failing, the
  // model step unranked because of it, and the fallback shown if setup itself breaks.
  await page.goto(`${url}/?mock=fail-hw`);
  await page.locator('.onb[data-step="welcome"]').waitFor();
  await next("level");
  await next("hardware");
  await page.locator('.scan-panel[data-state="error"]').waitFor();
  await shot("hardware-error");
  await next("model");
  await shot("model-unranked");
  await page.goto(`${url}/?mock=bad-hw`);
  await page.locator('.onb[data-step="welcome"]').waitFor();
  await next("level");
  await page.locator("[data-primary]").click();
  await page.locator(".setup-fallback").waitFor();
  await shot("fallback");
  await context.close();
}

/**
 * The Models screen's states (models-*): the detail drawer on each tab, the vision and speech
 * "try it" panels, the disk panel, and the loading and empty library.
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {"dark" | "light"} theme
 */
async function shootModels(page, url, theme) {
  const save = async (/** @type {string} */ name) => {
    await page.mouse.move(1270, 790);
    await page
      .waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running" || a.effect?.getComputedTiming().iterations === Infinity), undefined, {
        timeout: 6000,
      })
      .catch(() => console.warn("  (some motion was still running; captured anyway)"));
    await page.waitForTimeout(150);
    const file = path.join(OUT_DIR, `models-${name}-${theme}.png`);
    await page.screenshot({ path: file, animations: "disabled" });
    console.log("Saved", file);
  };
  /** A fresh load of the Models screen (a hash-only navigation would keep an open drawer). */
  const fresh = async (/** @type {string} */ query = "") => {
    await page.goto(`${url}/${query}#/models`);
    await page.reload();
  };
  /** Opens one model's drawer by searching its file, the way a user would. */
  const open = async (/** @type {string} */ id, /** @type {string} */ query) => {
    await fresh();
    await page.locator(".lib-body > .lib-row").first().waitFor();
    await page.getByRole("searchbox", { name: "Search models" }).fill(query);
    await page.locator(`.lib-row[data-model="${id}"] .lib-name__text`).click();
    await page.locator(".drawer__title").waitFor();
  };
  const tab = async (/** @type {string} */ name) => {
    await page.getByRole("tab", { name }).click();
    await page.waitForTimeout(250);
  };

  await open("qwen3-coder-30b-a3b-q4km", "coder 30b q4_k_m");
  await save("drawer-overview");
  await tab("Load settings");
  await save("drawer-load");
  await page.getByLabel("Context length", { exact: true }).fill("65536");
  await page.getByLabel("Experts on CPU", { exact: true }).fill("4");
  await save("drawer-load-custom");
  await open("qwen3-8b-q5km", "qwen3 8b q5_k_m");
  await tab("Tweak Studio");
  await page.getByLabel("KV cache type").selectOption("q8_0");
  await save("drawer-tweak");
  await tab("Load settings");
  await save("drawer-load-tweaked");
  await tab("Metadata");
  await save("drawer-metadata");
  await page.getByRole("button", { name: "Edit qwen3.context_length" }).click();
  await page.getByLabel("New value for qwen3.context_length").fill("40960");
  await page.keyboard.press("Enter");
  await page.getByRole("button", { name: "Write to a copy" }).click();
  await page.locator(".meta__copy").waitFor();
  await page.getByRole("button", { name: "Edit general.name" }).click();
  await page.getByLabel("New value for general.name").fill("Qwen3-8B (renamed)");
  await page.keyboard.press("Enter");
  await save("drawer-metadata-copy");

  // This machine not detected: nothing is estimated for it.
  await page.goto(`${url}/?mock=fail-hw#/models`);
  await page.reload();
  await page.locator(".lib-body > .lib-row").first().waitFor();
  await page.getByRole("searchbox", { name: "Search models" }).fill("qwen3 8b q5_k_m");
  await page.locator('.lib-row[data-model="qwen3-8b-q5km"] .lib-name__text').click();
  await page.locator(".drawer__title").waitFor();
  await tab("Load settings");
  await save("drawer-load-nohw");

  await fresh();
  await page.locator(".lib-body > .lib-row").first().waitFor();
  await page.locator(".lib-toolbar label", { hasText: "Vision" }).click();
  await page.locator(".lib-body > .lib-row").first().locator(".lib-name__text").click();
  await page.locator(".drawer__title").waitFor();
  await save("drawer-vision");
  await page.keyboard.press("Escape");
  await page.locator(".lib-toolbar label", { hasText: "Audio" }).click();
  await page.locator(".lib-body > .lib-row").first().locator(".lib-name__text").click();
  await page.locator(".drawer__title").waitFor();
  await save("drawer-audio");
  await page.keyboard.press("Escape");

  await fresh();
  await page.locator(".dup-list").waitFor();
  await page.locator("#disk-title").evaluate((el) => el.scrollIntoView({ block: "start" }));
  await page.waitForTimeout(400);
  await save("disk");

  await fresh("?mock=slow-library");
  await page.locator(".lib-grid--skeleton").waitFor();
  await page.waitForTimeout(500);
  await save("loading");
  await fresh("?mock=empty-library");
  await page.locator(".lib-empty").waitFor();
  await save("empty");
}

/**
 * The Tune screen's states (tune-*): a run mid-screen and mid-soak (the mock freezes playback at a
 * marked event), a finished MoE tune scrolled to its candidates and to the result, llama.cpp's own
 * settings kept for a small model, a cancelled run and a crashed one.
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {"dark" | "light"} theme
 */
async function shootTune(page, url, theme) {
  const save = async (/** @type {string} */ name) => {
    await page.mouse.move(1270, 790);
    await page
      .waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running" || a.effect?.getComputedTiming().iterations === Infinity), undefined, {
        timeout: 6000,
      })
      .catch(() => console.warn("  (some motion was still running; captured anyway)"));
    await page.waitForTimeout(200);
    const file = path.join(OUT_DIR, `tune-${name}-${theme}.png`);
    await page.screenshot({ path: file, animations: "disabled" });
    console.log("Saved", file);
  };
  /** Scrolls <main> so `selector` sits near the top, firing the reveals on the way. */
  const scrollToEl = async (/** @type {string} */ selector) => {
    await page.locator(selector).first().evaluate((el) => el.scrollIntoView({ block: "start" }));
    await page.evaluate(() => document.querySelector("main")?.scrollBy({ top: -16 }));
    await page.waitForTimeout(500);
  };
  /** A fresh page with the given mock switches, a model and profile chosen, and the tune started. */
  const start = async (/** @type {string} */ query, /** @type {string} */ model, /** @type {string} */ profile) => {
    await page.goto(`${url}/${query}#/tune`);
    await page.reload();
    await page.locator(`.pick-card[data-model="${model}"]`).click();
    await page.locator(`.pick-card[data-profile="${profile}"]`).click();
    await page.getByRole("button", { name: "Start tune" }).click();
    await page.locator(".run").waitFor();
  };

  await start("?mock=fast-tune&mock=tune-hold-screen", "qwen3-coder-30b-a3b-q4km", "chat");
  await page.locator('.cand[data-candidate="c4"][data-status="screening"]').waitFor();
  await page.waitForTimeout(600);
  await scrollToEl("#run-title");
  await save("screening");
  await scrollToEl(".tune-card--cands");
  await save("candidates-live");

  await start("?mock=fast-tune&mock=tune-hold-soak", "qwen3-coder-30b-a3b-q4km", "chat");
  await page.locator(".soak-item").nth(1).waitFor({ timeout: 15000 });
  await page.waitForTimeout(600);
  await scrollToEl(".run__grid--checks");
  await save("soak");

  await start("?mock=fast-tune", "qwen3-coder-30b-a3b-q4km", "coding");
  await page.locator(".winner").waitFor({ timeout: 15000 });
  await scrollToEl(".winner");
  await save("result");
  await scrollToEl(".winner__body");
  await save("result-why");

  await start("?mock=fast-tune", "gemma-4-e4b-q4km", "chat");
  await page.locator(".winner").waitFor({ timeout: 15000 });
  await scrollToEl(".winner");
  await save("defaults-kept");

  await start("?mock=fast-tune&mock=tune-hold-soak", "mistral-small-24b-q4km", "long-docs");
  await page.locator(".soak-item").first().waitFor({ timeout: 15000 });
  await page.getByRole("button", { name: "Cancel tune" }).click();
  await page.getByRole("button", { name: "Resume" }).waitFor();
  await scrollToEl("#run-title");
  await save("cancelled");

  await start("?mock=fast-tune&mock=fail-tune", "qwen3-8b-q5km", "coding");
  await page.locator(".run[data-status='failed']").waitFor({ timeout: 15000 });
  await scrollToEl("#run-title");
  await save("failed");
}

/**
 * The Monitor's states: the live view at the top, the metric cards, alerts firing (thermal, VRAM
 * pressure, speed below baseline), a request open in the inspector, a two-GPU machine, the 24 h
 * range, and an engine that stopped answering.
 * @param {import("@playwright/test").Page} page
 * @param {string} url
 * @param {"dark" | "light"} theme
 */
async function shootMonitor(page, url, theme) {
  const save = async (/** @type {string} */ name) => {
    await page.mouse.move(1270, 790);
    await page.waitForTimeout(700);
    const file = path.join(OUT_DIR, `monitor-${name}-${theme}.png`);
    await page.screenshot({ path: file, animations: "disabled" });
    console.log("Saved", file);
  };
  const scrollToEl = async (/** @type {string} */ selector) => {
    await page.locator(selector).first().evaluate((el) => el.scrollIntoView({ block: "start" }));
    await page.evaluate(() => document.querySelector("main")?.scrollBy({ top: -16 }));
    await page.waitForTimeout(600);
  };
  const open = async (/** @type {string} */ query) => {
    await page.goto(`${url}/${query}#/monitor`);
    await page.reload();
    await page.locator(".mon-grid .mon-card").first().waitFor();
    await settle(page);
  };

  await open("");
  await page.waitForTimeout(2500);
  await save("top");
  await scrollToEl("#mon-live-title");
  await save("metrics");
  await scrollToEl(".mon-pair-row");
  await save("cache");
  await scrollToEl("#mon-requests-title");
  await page.locator(".mon-tag[data-state='completed']").first().click();
  await page.getByRole("tab", { name: "Output" }).waitFor();
  await page.waitForTimeout(400);
  await save("inspector");

  await open("?mock=monitor-alerts");
  await page.locator(".mon-alert[data-kind='regression']").waitFor({ timeout: 20000 });
  await page.evaluate(() => document.querySelector("main")?.scrollTo({ top: 0 }));
  await save("alerts");

  await open("?mock=monitor-multi-gpu");
  await scrollToEl("#mon-live-title");
  await save("multi-gpu");

  await open("");
  await page.locator(".mon-range label", { hasText: "24 h" }).click();
  await page.waitForTimeout(800);
  await scrollToEl("#mon-live-title");
  await save("24h");
  await page.locator(".mon-range label", { hasText: "1 min" }).click();

  await page.goto(`${url}/?mock=fail-metrics#/monitor`);
  await page.reload();
  await page.locator(".engine-pill[data-tone='offline']").waitFor({ timeout: 10000 });
  await save("unreachable");
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
      await context.addInitScript(([key, record]) => window.localStorage.setItem(key, record), /** @type {[string, string]} */ ([ONBOARDING_KEY, DONE_RECORD]));
      const page = await context.newPage();
      page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && problems.push(`${theme}: ${m.text()}`));
      page.on("pageerror", (e) => problems.push(`${theme}: ${e.message}`));

      if (process.env["SHOOT_ONLY"] === "monitor") {
        await page.goto(`${url}/#/monitor`);
        await page.locator("main h1").waitFor();
        await settle(page);
        await page.screenshot({ path: path.join(OUT_DIR, `monitor-${theme}.png`), animations: "disabled" });
        await shootMonitor(page, url, theme);
        await context.close();
        continue;
      }
      for (const id of ROUTES) {
        await page.goto(`${url}/#/${id}`);
        await page.locator("main h1").waitFor();
        await page.locator(".statusbar [data-hud='speed']").waitFor();
        await settle(page);
        const file = path.join(OUT_DIR, `${id}-${theme}.png`);
        await page.screenshot({ path: file, animations: "disabled" });
        console.log("Saved", file);
      }

      // Beginner view of the screens the experience level changes, then back to Advanced.
      await page.evaluate(() => window.localStorage.setItem("cerebro-experience", "beginner"));
      for (const id of ["hub", "settings"]) {
        await page.goto(`${url}/#/${id}`);
        await page.reload();
        await page.locator("main h1").waitFor();
        await page.locator(".statusbar [data-hud='speed']").waitFor();
        await settle(page);
        const file = path.join(OUT_DIR, `${id}-beginner-${theme}.png`);
        await page.screenshot({ path: file, animations: "disabled" });
        console.log("Saved", file);
      }
      await page.evaluate(() => window.localStorage.removeItem("cerebro-experience"));

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

      await shootModels(page, url, theme);
      await shootTune(page, url, theme);
      await shootMonitor(page, url, theme);

      // Collapsed sidebar on the Models screen.
      await page.goto(`${url}/#/models`);
      await page.locator("main h1").waitFor();
      await page.keyboard.press("Control+b");
      await settle(page);
      await page.mouse.move(640, 400);
      await page.screenshot({ path: path.join(OUT_DIR, `collapsed-${theme}.png`), animations: "disabled" });
      console.log("Saved", path.join(OUT_DIR, `collapsed-${theme}.png`));
      await context.close();

      await shootOnboarding(browser, origin, url, theme, offOrigin, problems);
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
