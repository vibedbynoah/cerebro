// @ts-nocheck
// Screenshots of every screen of the real desktop exe, for reviewing how the app looks with real
// data. A throwaway WebView profile and its own debugging port, like e2e-models.mjs, so nothing
// lands in the user's app.
//
// Usage: node scripts/shoot-real.mjs OUT_DIR [route ...]
//   CEREBRO_E2E_EXE overrides the exe (default: the e2e build).

import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { chromium } from "playwright-core";

const exe = process.env.CEREBRO_E2E_EXE ?? new URL("../../../target-e2e/release/cerebro-desktop.exe", import.meta.url).pathname.replace(/^\/(\w:)/, "$1");
const out = process.argv[2];
if (!out) throw new Error("usage: node scripts/shoot-real.mjs OUT_DIR [route ...]");
mkdirSync(out, { recursive: true });
const routes = process.argv.slice(3).length ? process.argv.slice(3) : ["home", "hub", "chat", "models", "cluster", "monitor", "turbo", "images", "documents", "train", "tools", "plugins", "settings", "about"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const port = await new Promise((res) => { const s = createServer().listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); }); });
const profile = `${out}/webview-profile`;
rmSync(profile, { recursive: true, force: true });
const app = spawn(exe, [], { env: { ...process.env, CEREBRO_E2E_PORT: String(port), WEBVIEW2_USER_DATA_FOLDER: profile, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}` }, stdio: "ignore" });
const killTree = () => { try { execFileSync("taskkill", ["/T", "/F", "/PID", String(app.pid)], { stdio: "ignore" }); } catch {} };
try {
  let browser;
  for (let i = 0; i < 60 && !browser; i++) {
    await sleep(500);
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 10_000 }).catch(() => undefined);
  }
  if (!browser) throw new Error("the app did not open a debugging port");
  const page = browser.contexts()[0].pages().find((p) => !p.url().includes("isolation")) ?? browser.contexts()[0].pages()[0];
  page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && console.log(`console ${m.type()}: ${m.text().slice(0, 300)}`));
  page.on("pageerror", (e) => console.log(`pageerror: ${String(e).slice(0, 300)}`));
  await page.setViewportSize({ width: 1440, height: 900 }).catch(() => {});
  await sleep(2000);
  if (await page.getByText("Skip setup").first().isVisible().catch(() => false)) await page.getByText("Skip setup").first().click();
  for (let i = 0; i < 8 && (await page.getByText("SET UP CEREBRO").first().isVisible().catch(() => false)); i++) {
    const accept = page.getByRole("button", { name: /accept and continue/i });
    if (await accept.isVisible().catch(() => false)) await accept.click();
    else await page.keyboard.press("Enter");
    await sleep(800);
  }
  const guide = page.getByTestId("feature-guide");
  if (await guide.isVisible().catch(() => false)) await guide.getByRole("button", { name: "Close" }).click();
  for (const r of routes) {
    await page.evaluate((h) => (location.hash = `#/${h}`), r);
    await sleep(3500);
    await page.screenshot({ path: `${out}/${r}.png`, fullPage: false });
    console.log(r);
  }
  await browser.close().catch(() => {});
} finally {
  killTree();
}
