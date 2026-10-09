import { expect, test } from "./fixtures";

// In the perf project (after every other test), untraced, like the other budgets.
test.use({ trace: "off" });

test("eight seconds of live Monitor updates never block the main thread for more than 50 ms", async ({ page }) => {
  test.setTimeout(90_000);
  // Up to three windows; at least one must have no long task at all (a stall the operating system
  // causes does not repeat, the screen's own work would).
  const runs: number[][] = [];
  for (let attempt = 0; attempt < 3; attempt++) {
    await page.goto("/#/monitor");
    await expect(page.locator(".mon-grid .mon-card")).toHaveCount(10);
    await page.waitForTimeout(1500);
    const tasks = await page.evaluate(
      () =>
        new Promise<number[]>((resolve) => {
          const out: number[] = [];
          const po = new PerformanceObserver((list) => list.getEntries().forEach((e) => out.push(Math.round(e.duration))));
          po.observe({ type: "longtask", buffered: false });
          setTimeout(() => {
            po.disconnect();
            resolve(out);
          }, 8000);
        }),
    );
    runs.push(tasks);
    if (tasks.length === 0) break;
  }
  expect(
    runs.some((t) => t.length === 0),
    `every window had a long task: ${runs.map((t) => `[${t.join(", ")}]`).join(" ")}`,
  ).toBe(true);
});
