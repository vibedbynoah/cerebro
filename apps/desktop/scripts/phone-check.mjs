#!/usr/bin/env node
// @ts-nocheck
/**
 * `node scripts/phone-check.mjs <url-file>` — drives the real phone page (src-tauri/src/remote/) served
 * by the real Rust server (the ignored test `serve_for_browser_check`) in emulated phones: pairing
 * link, sign-in, dashboard, chat, stop dialog, sign-out. Fails on any console error or CSP report and
 * saves screenshots to review/phone/.
 */
import { chromium, devices } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "../review/phone");
const urls = (await readFile(process.argv[2], "utf8")).trim().split("\n");
await mkdir(OUT, { recursive: true });

const browser = await chromium.launch();
const problems = [];
const cases = [
  { name: "iphone-dark", device: devices["iPhone 13"], scheme: "dark" },
  { name: "android-light", device: devices["Pixel 7"], scheme: "light" },
];
for (const [i, c] of cases.entries()) {
  const ctx = await browser.newContext({ ...c.device, colorScheme: c.scheme, ignoreHTTPSErrors: true });
  const page = await ctx.newPage();
  // 401s are expected (the first session check, the deliberate wrong password); nothing else is.
  page.on("console", (m) => m.type() === "error" && !m.text().includes("status of 401") && problems.push(`${c.name} console: ${m.text()}`));
  page.on("response", (r) => r.status() >= 400 && r.status() !== 401 && problems.push(`${c.name}: ${r.request().method()} ${new URL(r.url()).pathname} -> ${r.status()}`));
  page.on("pageerror", (e) => problems.push(`${c.name} pageerror: ${e}`));
  const url = urls[i];
  await page.goto(url);
  await page.waitForSelector("#viewSignIn:not([hidden])");
  if (page.url().includes("pair=")) problems.push(`${c.name}: pairing code left in the address bar`);
  await page.screenshot({ path: path.join(OUT, `${c.name}-signin.png`) });

  // Wrong password first: the message must stay on screen.
  await page.fill("#password", "wrong password");
  await page.click("#signInSubmit");
  await page.waitForSelector("#signInError:not([hidden])");
  await page.waitForTimeout(1500);
  if (await page.isHidden("#signInError")) problems.push(`${c.name}: sign-in error vanished`);

  await page.fill("#password", "correct horse");
  await page.click("#signInSubmit");
  await page.waitForSelector("#viewDash:not([hidden])", { timeout: 5000 });
  await page.waitForTimeout(1500);
  const decode = await page.textContent("#decode");
  if (!decode || decode.includes("—")) problems.push(`${c.name}: decode readout empty (${decode})`);
  const pill = await page.textContent("#pillText");
  if (pill !== "Generating") problems.push(`${c.name}: pill says "${pill}"`);
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
  if (sideways) problems.push(`${c.name}: page scrolls sideways`);
  await page.screenshot({ path: path.join(OUT, `${c.name}-dashboard.png`), fullPage: true });

  await page.click("#tabChat");
  await page.fill("#chatInput", "How fast is my model?");
  await page.click("#sendButton");
  await page.waitForFunction(() => document.querySelector("#chatLog .msg.assistant .body")?.textContent?.includes("tokens a second."), null, { timeout: 10000 });
  await page.screenshot({ path: path.join(OUT, `${c.name}-chat.png`) });

  await page.click("#tabStatus");
  await page.click("#stopButton");
  await page.waitForSelector("#stopScrim:not([hidden])");
  await page.screenshot({ path: path.join(OUT, `${c.name}-stop.png`) });
  await page.click("#stopConfirm");

  await page.click("#signOut");
  await page.waitForSelector("#viewSignIn:not([hidden])");
  await ctx.close();
}
await browser.close();
for (const p of problems) console.log("PROBLEM", p);
console.log(problems.length ? `${problems.length} problem(s)` : "phone page OK");
process.exit(problems.length ? 1 : 0);
