#!/usr/bin/env node
// @ts-nocheck
/** `node scripts/shot-one.mjs <route> [advanced] [out.png]` — one screenshot of one screen of the web build (ui/dist), dark theme, setup finished. */
import { chromium } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const [route = "home", level = "beginner", out = path.resolve(HERE, `../review/${route}-${level}.png`)] = process.argv.slice(2);
const SRC = await readFile(path.resolve(HERE, "../ui/src/onboarding/progress.ts"), "utf8");
const c = (/** @type {string} */ n) => /** @type {string} */ (new RegExp(`export const ${n} = "([^"]+)"`).exec(SRC)?.[1]);
const done = JSON.stringify({ v: 1, completed: true, termsVersion: c("TERMS_VERSION"), model: "gemma-4-12b-it" });
const { url: origin, close } = await startServer(0);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 1000 }, colorScheme: "dark" });
await page.addInitScript(([k, v, lvl]) => { localStorage.setItem(k, v); localStorage.setItem("cerebro-experience", lvl); }, [c("ONBOARDING_KEY"), done, level]);
await page.goto(`${origin}/#/${route}`);
await page.waitForTimeout(1500);
await page.screenshot({ path: out, fullPage: false });
console.log(out);
await browser.close();
await close();
