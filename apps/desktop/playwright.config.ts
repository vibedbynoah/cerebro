import os from "node:os";
import { defineConfig, devices } from "@playwright/test";

// Fixed port so every worker agrees on the base URL; startup fails loudly if something holds it.
const PORT = 4391;
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  // A quarter of the logical CPUs, capped at 4: axe scans starve at the default on this laptop.
  workers: process.env["CI"] ? 2 : Math.max(1, Math.min(4, Math.floor(os.cpus().length / 4))),
  timeout: 45_000,
  forbidOnly: !!process.env["CI"],
  retries: process.env["CI"] ? 1 : 0,
  reporter: [["list"]],
  outputDir: "./test-results",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
    // Headless always: nothing may appear on the owner's screen.
    headless: true,
  },
  webServer: {
    command: "node scripts/serve.mjs",
    url: `${BASE_URL}/index.html`,
    reuseExistingServer: false,
    timeout: 20_000,
    env: { PORT: String(PORT) },
  },
  projects: [
    { name: "chromium", testIgnore: /\.perf\.spec\.ts$|-perf\.spec\.ts$/, use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } } },
    // Main-thread budgets (no task over 50 ms) are measured after every other test has finished, so
    // parallel workers competing for the same CPU cannot stretch a task past the budget.
    {
      name: "perf",
      testMatch: /\.perf\.spec\.ts$|-perf\.spec\.ts$/,
      dependencies: ["chromium"],
      use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } },
    },
  ],
});
