#!/usr/bin/env node
// @ts-nocheck
/**
 * `node scripts/shot.mjs <hash-route> <out.png> [width] [height] [theme] [setup.js]` — one screenshot of
 * the web build (ui/dist) past setup and the feature guide, after an optional setup script runs in
 * the page (for example to open a panel). For reviewing a screen while building it.
 */
import { chromium } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const [route = "home", out = "review/shot.png", width = "1440", height = "900", theme = "dark", setup = ""] = process.argv.slice(2);
const src = (f) => readFileSync(path.resolve(HERE, f), "utf8");
const constant = (s, n) => new RegExp(`export const ${n} = "([^"]+)"`).exec(s)?.[1];
const progress = src("../ui/src/onboarding/progress.ts");
const record = JSON.stringify({ v: 1, completed: true, termsVersion: constant(progress, "TERMS_VERSION"), model: "gemma-4-12b-it" });
const guide = constant(src("../ui/src/lib/features.ts"), "GUIDE_VERSION");
const { url, close } = await startServer(0);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: Number(width), height: Number(height) }, colorScheme: theme === "light" ? "light" : "dark" });
await page.addInitScript(([k, v, g, t]) => {
  localStorage.setItem(k, v);
  localStorage.setItem("cerebro-guide-seen", g);
  localStorage.setItem("cerebro-theme", t);
}, [constant(progress, "ONBOARDING_KEY"), record, guide, theme]);
await page.goto(`${url}/#/${route}`);
await page.waitForTimeout(1200);
if (setup) await page.evaluate(setup);
await page.waitForTimeout(1500);
await page.screenshot({ path: out });
await browser.close();
await close();
console.log(out);
