// @ts-nocheck
// End-to-end check of the real desktop exe (no mock): starts target/release/cerebro-desktop.exe with
// WebView2 remote debugging, drives the UI over CDP, calls the engine through the app's own IPC,
// saves screenshots, then closes the app. Usage: node scripts/e2e-real.mjs [model-name-substring]
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright-core";

const exe = process.env.CEREBRO_E2E_EXE ?? new URL("../../../target/release/cerebro-desktop.exe", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
const want = (process.argv[2] ?? "0.6B").toLowerCase();
const out = new URL("../test-results/e2e-real/", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
mkdirSync(out, { recursive: true });

// Its own WebView profile, so the test's chats never land in the user's app.
const profile = join(tmpdir(), "cerebro-e2e-real-profile");
rmSync(profile, { recursive: true, force: true });
const app = spawn(exe, [], { env: { ...process.env, WEBVIEW2_USER_DATA_FOLDER: profile, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: "--remote-debugging-port=9333" }, stdio: ["ignore", "pipe", "pipe"] });
let log = "";
app.stdout.on("data", (d) => (log += d));
app.stderr.on("data", (d) => (log += d));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const report = {};
try {
  let browser;
  for (let i = 0; i < 60 && !browser; i++) {
    await sleep(500);
    browser = await chromium.connectOverCDP("http://127.0.0.1:9333").catch(() => undefined);
  }
  if (!browser) throw new Error("app did not expose a debugging port");
  const page = browser.contexts()[0].pages().find((p) => !p.url().includes("isolation")) ?? browser.contexts()[0].pages()[0];
  report.console = [];
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && report.console.push(m.text().slice(0, 400)));
  page.on("pageerror", (e) => report.console.push("pageerror: " + String(e).slice(0, 400)));
  await page.waitForLoadState("domcontentloaded");
  await sleep(1500);
  const call = (cmd, args) => page.evaluate(([c, a]) => window.__TAURI_INTERNALS__.invoke(c, a ?? {}), [cmd, args]);
  const shot = async (name) => page.screenshot({ path: `${out}${name}.png` });

  // First run shows setup: walk to "This machine" (real hardware), then skip the rest.
  if (await page.getByText("Skip setup").first().isVisible().catch(() => false)) {
    await page.getByText("Get started").first().click();
    await sleep(800);
    await page.keyboard.press("Enter");
    await sleep(2500);
    await shot("setup-this-machine");
    await page.keyboard.press("Escape");
    await sleep(800);
  }
  // The rest of setup, or accepting changed terms again.
  for (let i = 0; i < 8 && (await page.getByText("SET UP CEREBRO").first().isVisible().catch(() => false)); i++) {
    const accept = page.getByRole("button", { name: /accept and continue/i });
    if (await accept.isVisible().catch(() => false)) await accept.click();
    else await page.keyboard.press("Enter");
    await sleep(1000);
  }
  await shot("after-setup");
  // The feature guide opens the first time; read it, then close it.
  const guide = page.getByTestId("feature-guide");
  report.guide = await guide.isVisible().catch(() => false);
  if (report.guide) {
    await shot("feature-guide");
    await guide.getByRole("button", { name: "Close" }).click();
    await sleep(500);
  }
  report.hardware = await call("hw_detect");
  const lib = await call("library_list");
  report.library = lib.map((m) => `${m.name} ${m.quant} ${m.sizeGiB.toFixed(1)} GiB ${m.supported ? "" : "(unsupported arch " + m.archName + ")"}`);
  await page.evaluate(() => (location.hash = "#/models"));
  await sleep(1500);
  await shot("models");

  const model = lib.find((m) => m.supported && (m.name.toLowerCase().includes(want) || m.file.toLowerCase().includes(want)));
  if (!model) throw new Error(`no supported model matching ${want}`);
  const t0 = Date.now();
  report.loaded = await call("model_load", { id: model.id, context: 4096 });
  report.loadSeconds = (Date.now() - t0) / 1000;

  await page.evaluate(() => (location.hash = "#/chat"));
  await sleep(1200);
  await shot("chat-empty");
  const box = page.locator("textarea").first();
  await box.fill("What is the capital of France? Answer in one sentence.");
  await box.press("Enter");
  await sleep(model.architecture === "moe" ? 20000 : 6000);
  await shot("chat-reply");
  report.chatText = (await page.locator("main").innerText()).slice(-600);

  // Model Hub: live search, a repository's files, and one small real download.
  const repos = await call("hub_search", { query: "Qwen3-0.6B GGUF" });
  report.hubTop = repos.slice(0, 5).map((r) => `${r.id} (${r.downloads} downloads)`);
  const repo = repos.find((r) => /unsloth\/Qwen3-0\.6B-GGUF/i.test(r.id)) ?? repos[0];
  const files = await call("hub_files", { repo: repo.id });
  report.hubFiles = files.map((f) => `${f.quant} ${(f.size / 2 ** 30).toFixed(2)} GiB${f.downloaded ? " (downloaded)" : ""}`);
  await page.evaluate(() => (location.hash = "#/hub"));
  await sleep(2000);
  await page.getByRole("searchbox").fill("Qwen3-0.6B GGUF");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await sleep(4000);
  await page.getByText(repo.id, { exact: true }).first().click().catch(() => {});
  await sleep(4000);
  await shot("hub");
  if (process.env.E2E_DOWNLOAD) {
    const small = [...files].sort((a, b) => a.size - b.size)[0];
    const t1 = Date.now();
    const row = page.locator("tr", { hasText: small.path }).first();
    await row.getByRole("button", { name: /download|resume/i }).click();
    await sleep(3000);
    await shot("hub-downloading");
    while (Date.now() - t1 < 600000 && !(await row.getByText("In your library").isVisible().catch(() => false))) await sleep(2000);
    report.download = { file: small.path, gib: small.size / 2 ** 30, seconds: (Date.now() - t1) / 1000 };
    await shot("hub-downloaded");
  }
  report.metrics = await call("metrics");
  report.live = await call("live_stats");
  await page.evaluate(() => (location.hash = "#/monitor"));
  await sleep(2500);
  await shot("monitor");
  await page.evaluate(() => (location.hash = "#/home"));
  await sleep(1500);
  await shot("home");
  await browser.close().catch(() => {});
} catch (e) {
  report.error = String(e?.stack ?? e);
} finally {
  app.kill();
  await sleep(500);
  writeFileSync(`${out}report.json`, JSON.stringify(report, null, 2));
  writeFileSync(`${out}app.log`, log);
  const m = report.metrics;
  console.log(JSON.stringify({
    hardware: report.hardware,
    library: report.library,
    loadSeconds: report.loadSeconds,
    chatTail: report.chatText,
    engine: m?.engine?.model?.name,
    moe: m?.engine?.moe,
    lastRequest: m?.requests?.[0],
    error: report.error,
    console: report.console,
    hubTop: report.hubTop,
    hubFiles: report.hubFiles,
    download: report.download,
  }, null, 2));
}
