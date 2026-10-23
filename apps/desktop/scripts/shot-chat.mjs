#!/usr/bin/env node
// @ts-nocheck
/**
 * `node scripts/shot-chat.mjs [out-prefix] [theme]` — screenshots of Chat in the web build (ui/dist,
 * mock engine): empty, then after one question with a formatted answer. Setup finished, advanced level.
 */
import { chromium } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const [prefix = path.resolve(HERE, "../review/chat"), theme = "dark"] = process.argv.slice(2);
const SRC = await readFile(path.resolve(HERE, "../ui/src/onboarding/progress.ts"), "utf8");
const c = (n) => new RegExp(`export const ${n} = "([^"]+)"`).exec(SRC)?.[1];
const done = JSON.stringify({ v: 1, completed: true, termsVersion: c("TERMS_VERSION"), model: "gemma-4-12b-it" });
const { url: origin, close } = await startServer(0);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: theme });
await page.addInitScript(([k, v]) => { localStorage.setItem(k, v); localStorage.setItem("cerebro-experience", "advanced"); }, [c("ONBOARDING_KEY"), done]);
await page.goto(`${origin}/#/chat`);
await page.waitForTimeout(1500);
const guide = page.getByTestId("feature-guide");
if (await guide.isVisible().catch(() => false)) await guide.getByRole("button", { name: "Close" }).click();
await page.screenshot({ path: `${prefix}-empty.png` });
const box = page.getByRole("textbox").first();
await box.fill("Explain how a split model runs across a graphics card and a processor, with a short code example.");
await page.keyboard.press("Enter");
await page.waitForTimeout(6000);
await page.screenshot({ path: `${prefix}-reply.png` });
// The message box's focus style, for checking it draws no ring of its own.
console.log(await page.locator("textarea.composer__input").evaluate((el) => { const s = getComputedStyle(el); return `focus: outline ${s.outline}, shadow ${s.boxShadow}, border ${s.border}, matches :focus-visible ${el.matches(":focus-visible")}`; }));
console.log(`${prefix}-empty.png ${prefix}-reply.png`);
await browser.close();
await close();
