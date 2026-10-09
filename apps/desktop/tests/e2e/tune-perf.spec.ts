import { expect, test } from "./fixtures";

// Untraced and in the perf project (after every other test), like models-perf.spec.ts: tracing and
// parallel workers would stretch tasks past the budget for reasons that are not the screen's.
test.use({ trace: "off" });

test("a tune streaming from probe to result never blocks the main thread for more than 50 ms", async ({ page }) => {
  test.setTimeout(120_000);
  // A long task is a frame the user feels drop. The screen's own work repeats on every run; a stall
  // the operating system causes (antivirus, a driver) does not. So up to three runs stream, and at
  // least one must finish with no long task at all.
  const runs: number[][] = [];
  for (let attempt = 0; attempt < 3; attempt++) {
    await page.goto("/?mock=fast-tune#/tune");
    await page.locator('.pick-card[data-model="qwen3-coder-30b-a3b-q4km"]').click();
    await page.locator('.pick-card[data-profile="coding"]').click();
    await page.evaluate(() => {
      const w = window as unknown as { __longTasks: number[] };
      w.__longTasks = [];
      new PerformanceObserver((list) => list.getEntries().forEach((e) => w.__longTasks.push(Math.round(e.duration)))).observe({ type: "longtask", buffered: false });
    });
    await page.getByRole("button", { name: "Start tune" }).click();
    await expect(page.locator(".winner")).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(400);
    const tasks = await page.evaluate(() => (window as unknown as { __longTasks: number[] }).__longTasks);
    runs.push(tasks);
    if (tasks.length === 0) break;
  }
  expect(
    runs.some((t) => t.length === 0),
    `every run had a long task: ${runs.map((t) => `[${t.join(", ")}]`).join(" ")}`,
  ).toBe(true);
});
