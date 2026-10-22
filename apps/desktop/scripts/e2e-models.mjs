// @ts-nocheck
// Every model, through the real desktop exe, the way a person uses it: Models screen -> Load,
// Chat -> pick the model, move the slider to Fastest, ask one question, stop as soon as the reply is
// in, then Unload and go on to the next model. Nothing is called behind the UI's back except reading
// the library list to know which rows to look for. Saves a report and a screenshot per model.
//
// Usage: node scripts/e2e-models.mjs [name-substring ...]
//   With substrings, only matching models run (in library order); without, every runnable text model.
//   CEREBRO_E2E_EXE overrides the exe (default: the release build).

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright-core";

const exe = process.env.CEREBRO_E2E_EXE ?? new URL("../../../target/release/cerebro-desktop.exe", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
const wants = process.argv.slice(2).map((s) => s.toLowerCase());
const out = new URL("../test-results/e2e-models/", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
mkdirSync(out, { recursive: true });
const QUESTION = "What is the capital of France? Answer in one word.";
const LOAD_TIMEOUT = 15 * 60_000;
const REPLY_TIMEOUT = 20 * 60_000;

const app = spawn(exe, [], { env: { ...process.env, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: "--remote-debugging-port=9334" }, stdio: ["ignore", "pipe", "pipe"] });
let log = "";
app.stdout.on("data", (d) => (log += d));
app.stderr.on("data", (d) => (log += d));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const report = { started: new Date().toISOString(), exe, question: QUESTION, results, console: [] };

try {
  let browser;
  for (let i = 0; i < 60 && !browser; i++) {
    await sleep(500);
    browser = await chromium.connectOverCDP("http://127.0.0.1:9334").catch(() => undefined);
  }
  if (!browser) throw new Error("the app did not open a debugging port");
  const page = browser.contexts()[0].pages().find((p) => !p.url().includes("isolation")) ?? browser.contexts()[0].pages()[0];
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && report.console.push(m.text().slice(0, 300)));
  page.on("pageerror", (e) => report.console.push("pageerror: " + String(e).slice(0, 300)));
  await page.waitForLoadState("domcontentloaded");
  await sleep(1500);

  // First run, changed terms, and the feature guide: through the UI.
  if (await page.getByText("Skip setup").first().isVisible().catch(() => false)) {
    await page.getByText("Skip setup").first().click();
    await sleep(800);
  }
  for (let i = 0; i < 8 && (await page.getByText("SET UP CEREBRO").first().isVisible().catch(() => false)); i++) {
    const accept = page.getByRole("button", { name: /accept and continue/i });
    if (await accept.isVisible().catch(() => false)) await accept.click();
    else await page.keyboard.press("Enter");
    await sleep(1000);
  }
  const guide = page.getByTestId("feature-guide");
  if (await guide.isVisible().catch(() => false)) await guide.getByRole("button", { name: "Close" }).click();

  // Which rows to look for (reading only).
  const lib = await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("library_list", {}));
  const models = lib.filter((m) => m.supported && (m.kind === "text" || m.kind === "vision") && (!wants.length || wants.some((w) => `${m.name} ${m.quant} ${m.file}`.toLowerCase().includes(w))));
  report.models = models.map((m) => `${m.name} ${m.quant} (${m.sizeGiB.toFixed(1)} GiB)`);

  for (const m of models) {
    const r = { model: `${m.name} ${m.quant}`, sizeGiB: +m.sizeGiB.toFixed(1) };
    results.push(r);
    const t0 = Date.now();
    try {
      // Models screen -> Load.
      await page.evaluate(() => (location.hash = "#/models"));
      await sleep(1200);
      const loadBtn = page.getByRole("button", { name: `Load ${m.name} ${m.quant}`, exact: true });
      await loadBtn.scrollIntoViewIfNeeded();
      await loadBtn.click();
      await page.getByRole("button", { name: `Unload ${m.name} ${m.quant}`, exact: true }).waitFor({ timeout: LOAD_TIMEOUT });
      r.loadSeconds = (Date.now() - t0) / 1000;
      r.placement = await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("model_loaded", {}).catch(() => null)).then((x) => (x ? JSON.stringify(x).slice(0, 300) : null)).catch(() => null);

      // Chat -> pick it, Fastest, ask.
      await page.evaluate(() => (location.hash = "#/chat"));
      await sleep(1200);
      await page.getByRole("button", { name: "New chat" }).first().click().catch(() => {});
      await page.getByRole("combobox", { name: "Model" }).selectOption(m.id);
      await page.locator("#speed-mode").fill("1");
      await sleep(400);
      const box = page.getByRole("textbox", { name: "Message" });
      await box.fill(QUESTION);
      const t1 = Date.now();
      await box.press("Enter");
      const reply = page.locator(".msg--bot").last();
      await reply.waitFor({ timeout: 60_000 });
      await page.waitForFunction(() => {
        const el = [...document.querySelectorAll(".msg--bot")].pop();
        return el && el.getAttribute("data-streaming") === "false";
      }, null, { timeout: REPLY_TIMEOUT });
      r.replySeconds = (Date.now() - t1) / 1000;
      const text = (await reply.innerText()).replace(/\s+/g, " ").trim();
      r.reply = text.slice(0, 400);
      r.coherent = /paris/i.test(text);
      await page.screenshot({ path: `${out}${m.id.replace(/[^a-z0-9]+/gi, "_").slice(0, 60)}.png` });

      // Models screen -> Unload.
      await page.evaluate(() => (location.hash = "#/models"));
      await sleep(1000);
      await page.getByRole("button", { name: `Unload ${m.name} ${m.quant}`, exact: true }).click();
      await page.getByRole("button", { name: `Load ${m.name} ${m.quant}`, exact: true }).waitFor({ timeout: 120_000 });
    } catch (e) {
      r.error = String(e?.message ?? e).split("\n")[0].slice(0, 300);
      await page.screenshot({ path: `${out}${m.id.replace(/[^a-z0-9]+/gi, "_").slice(0, 60)}-error.png` }).catch(() => {});
    }
    r.totalSeconds = (Date.now() - t0) / 1000;
    console.log(JSON.stringify(r));
    writeFileSync(`${out}report.json`, JSON.stringify(report, null, 2));
  }
  await browser.close().catch(() => {});
} catch (e) {
  report.error = String(e?.stack ?? e);
} finally {
  app.kill();
  await sleep(500);
  report.finished = new Date().toISOString();
  writeFileSync(`${out}report.json`, JSON.stringify(report, null, 2));
  writeFileSync(`${out}app.log`, log);
  const ok = results.filter((r) => r.coherent).length;
  console.log(`${ok} of ${results.length} models answered coherently${report.error ? `; error: ${report.error.split("\n")[0]}` : ""}`);
}
