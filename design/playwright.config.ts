import os from "node:os";
import { defineConfig, devices } from "@playwright/test";

// Fixed port, never reused: if anything already holds it, startup fails loudly. (This config is
// re-evaluated in every worker, so a per-run random port would differ between workers.)
const PORT = 4321;
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  // Playwright's default (half the logical CPUs = 10 on the owner's 20-thread laptop) starved the
  // axe scans into timeouts. A quarter, capped at 4, keeps every browser responsive; heavy tests
  // also set their own budget.
  workers: process.env["CI"] ? 2 : Math.max(1, Math.min(4, Math.floor(os.cpus().length / 4))),
  timeout: 45_000,
  forbidOnly: !!process.env["CI"],
  retries: process.env["CI"] ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  webServer: {
    command: "node scripts/serve.mjs",
    url: `${BASE_URL}/preview.html`,
    reuseExistingServer: false,
    timeout: 20_000,
    env: { PORT: String(PORT) },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
