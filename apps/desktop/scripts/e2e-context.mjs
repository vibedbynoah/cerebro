// @ts-nocheck
// The long-context check through the real desktop exe: loads each model at E2E_CONTEXT tokens
// (default 65536, past most small models' trained length, so YaRN is on), opens its drawer, presses
// "Check long context" and records how many of the nine hidden passcodes it found.
//
// Usage: node scripts/e2e-context.mjs [name-substring ...]
//   CEREBRO_E2E_EXE overrides the exe (default: the release build).

import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { chromium } from "playwright-core";

const exe = process.env.CEREBRO_E2E_EXE ?? new URL("../../../target/release/cerebro-desktop.exe", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
const wants = process.argv.slice(2).map((s) => s.toLowerCase());
const out = new URL("../test-results/e2e-context/", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
mkdirSync(out, { recursive: true });
const QUESTION = "What is the capital of France? Answer in one word.";
const LOAD_TIMEOUT = 15 * 60_000;
const REPLY_TIMEOUT = 20 * 60_000;

// WebView2 reuses a running browser process for the same profile and ignores new arguments, so a
// WebView left over from an earlier run kept its port and this script drove it until the timeout.
// Leftover app and WebView processes go first, the app gets a port nothing holds, and the whole
// process tree goes at the end (killing only the exe leaves its WebView running).
const ps = (cmd) => { try { execFileSync("powershell.exe", ["-NoProfile", "-Command", cmd], { stdio: "ignore" }); } catch {} };
ps("Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cerebro-desktop.exe' -or ($_.Name -eq 'msedgewebview2.exe' -and $_.CommandLine -match 'remote-debugging-port') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }");
const port = await new Promise((res) => { const s = createServer().listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); }); });
// A WebView profile of its own, emptied each run: the test's chats, settings and first-run choices
// never land in the user's app (they did: a test chat about France showed up in the owner's Chat).
const profile = `${out}webview-profile`;
rmSync(profile, { recursive: true, force: true });
const app = spawn(exe, [], { env: { ...process.env, CEREBRO_E2E_PORT: String(port), WEBVIEW2_USER_DATA_FOLDER: profile, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}` }, stdio: ["ignore", "pipe", "pipe"] });
const killTree = () => { if (app.pid) try { execFileSync("taskkill", ["/T", "/F", "/PID", String(app.pid)], { stdio: "ignore" }); } catch {} };
let log = "";
app.stdout.on("data", (d) => (log += d));
app.stderr.on("data", (d) => (log += d));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const report = { started: new Date().toISOString(), exe, question: QUESTION, results, console: [] };
const save = () => writeFileSync(`${out}report.json`, JSON.stringify(report, null, 2));
// A call into the app that never answers must not hang the run: every model gets its load and reply
// budget plus ten minutes, then the run writes what it has and stops.
let deadline;
const watchdog = (ms, what) => {
  clearTimeout(deadline);
  deadline = setTimeout(() => {
    report.error = `stuck for ${Math.round(ms / 60_000)} min: ${what}`;
    report.finished = new Date().toISOString();
    save();
    writeFileSync(`${out}app.log`, log);
    killTree();
    console.log(`stopped: ${report.error}`);
    process.exit(2);
  }, ms);
};
watchdog(5 * 60_000, "starting the app");

try {
  let browser;
  for (let i = 0; i < 60 && !browser; i++) {
    await sleep(500);
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 10_000 }).catch(() => undefined);
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
  const lib = await Promise.race([page.evaluate(() => window.__TAURI_INTERNALS__.invoke("library_list", {})), sleep(60_000).then(() => { throw new Error("library_list did not answer in 60 s"); })]);
  const models = lib.filter((m) => m.supported && (m.kind === "text" || m.kind === "vision") && (!wants.length || wants.some((w) => `${m.name} ${m.quant} ${m.file}`.toLowerCase().includes(w))));
  report.models = models.map((m) => `${m.name} ${m.quant} (${m.sizeGiB.toFixed(1)} GiB)`);

  const findRow = async (m) => {
    await page.evaluate(() => (location.hash = "#/models"));
    await sleep(800);
    const search = page.getByRole("searchbox").or(page.getByPlaceholder(/Search name/)).first();
    await search.fill(m.file.split(/[\\/]/).pop().replace(/\.gguf$/i, ""));
    await sleep(500);
    const row = page.locator(`.lib-row[data-model="${m.id}"]`);
    await row.waitFor({ timeout: 15_000 });
    return row;
  };

  // E2E_CONTEXT tokens (default 65536): past most small models' trained length, so YaRN is on.
  const CTX = Number(process.env.E2E_CONTEXT ?? 65536);
  for (const m of models) {
    const r = { model: `${m.name} ${m.quant}`, trained: m.contextMax, extended: m.contextExtended, context: CTX };
    results.push(r);
    const t0 = Date.now();
    watchdog(LOAD_TIMEOUT + REPLY_TIMEOUT + 10 * 60_000, r.model);
    try {
      await Promise.race([page.evaluate(([id, ctx]) => window.__TAURI_INTERNALS__.invoke("model_load", { id, context: ctx }), [m.id, CTX]), sleep(LOAD_TIMEOUT).then(() => { throw new Error("load timed out"); })]);
      r.loadSeconds = (Date.now() - t0) / 1000;
      // Models screen -> the model's drawer -> Load tab -> Check long context.
      const row = await findRow(m);
      // The row opens the drawer when a click lands outside its buttons.
      await row.locator(".lib-cell--model").click({ position: { x: 90, y: 10 } });
      await sleep(800);
      await page.getByRole("tab", { name: /^Load/ }).first().click();
      const btn = page.getByRole("button", { name: /Check long context/ });
      await btn.waitFor({ timeout: 30_000 });
      const t1 = Date.now();
      await btn.click();
      await page.locator(".ctxcheck__verdict").waitFor({ timeout: REPLY_TIMEOUT });
      r.checkSeconds = (Date.now() - t1) / 1000;
      r.verdict = (await page.locator(".ctxcheck__verdict").innerText()).replace(/\s+/g, " ").trim();
      r.rows = await page.locator(".ctxcheck__table tbody tr").allInnerTexts();
      r.coherent = /^(\d+) of \1 found/.test(r.verdict);
      await page.screenshot({ path: `${out}${m.id.replace(/[^a-z0-9]+/gi, "_").slice(0, 60)}.png` });
      await page.evaluate(() => window.__TAURI_INTERNALS__.invoke("model_unload", {})).catch(() => {});
    } catch (e) {
      r.error = String(e?.message ?? e).split("\n")[0].slice(0, 300);
      await page.screenshot({ path: `${out}${m.id.replace(/[^a-z0-9]+/gi, "_").slice(0, 60)}-error.png` }).catch(() => {});
    }
    r.totalSeconds = (Date.now() - t0) / 1000;
    console.log(JSON.stringify(r));
    save();
  }
  clearTimeout(deadline);
  await browser.close().catch(() => {});
} catch (e) {
  report.error = String(e?.stack ?? e);
} finally {
  clearTimeout(deadline);
  killTree();
  await sleep(500);
  report.finished = new Date().toISOString();
  save();
  writeFileSync(`${out}app.log`, log);
  const ok = results.filter((r) => r.coherent).length;
  console.log(`${ok} of ${results.length} models passed the long-context check${report.error ? `; error: ${report.error.split("\n")[0]}` : ""}`);
}
