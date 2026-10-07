import { expect, openScreen, test } from "./fixtures";

// Trace recording observes every DOM mutation and multiplies main-thread work several-fold, so this
// file runs untraced. Playwright only allows `trace` per file, which is why this one measurement
// lives on its own (and in its own project, after every other test, so no parallel worker competes
// for the CPU: see playwright.config.ts).
test.use({ trace: "off" });

interface Pass {
  /** Durations of every long task (main thread busy over 50 ms) seen during the pass, ms. */
  longTasks: number[];
  /** Median time from one scroll step to the next frame, ms. */
  medianFrame: number;
  steps: number;
}

test("scrolling the whole 500+ model library never blocks the main thread for more than 50 ms", async ({ page }) => {
  await openScreen(page, "models");
  const grid = page.getByRole("grid", { name: "Models on this machine" });
  await expect(grid.locator(".lib-body > .lib-row").first()).toBeVisible();

  // A long task is what a user feels as a dropped frame or a stuck scroll. Step through the whole
  // list one 240 px notch per frame, as a wheel would, and record every long task the browser
  // reports. Work the list itself causes happens on every pass; a stall caused by the operating
  // system (an antivirus scan, a driver) does not repeat. So up to three passes run, every pass
  // must keep its median frame near 60 Hz, and at least one full pass must have no long task.
  const passes: Pass[] = [];
  for (let attempt = 0; attempt < 3; attempt++) {
    const pass = await grid.evaluate(async (el): Promise<Pass> => {
      el.scrollTop = 0;
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const longTasks: number[] = [];
      const po = new PerformanceObserver((list) => list.getEntries().forEach((e) => longTasks.push(Math.round(e.duration))));
      po.observe({ type: "longtask", buffered: false });
      const frames: number[] = [];
      for (let y = 0; y < el.scrollHeight; y += 240) {
        const t0 = performance.now();
        el.scrollTop = y;
        await new Promise((r) => requestAnimationFrame(r));
        frames.push(performance.now() - t0);
      }
      // Long-task entries are delivered asynchronously; give the last ones time to arrive.
      await new Promise((r) => setTimeout(r, 100));
      po.disconnect();
      frames.sort((a, b) => a - b);
      return { longTasks, medianFrame: frames[Math.floor(frames.length / 2)] ?? Infinity, steps: frames.length };
    });
    passes.push(pass);
    expect(pass.steps).toBeGreaterThan(100);
    expect(pass.medianFrame, `median frame in pass ${attempt + 1}`).toBeLessThan(25);
    if (pass.longTasks.length === 0) break;
  }
  expect(
    passes.some((p) => p.longTasks.length === 0),
    `every pass had a long task: ${passes.map((p) => `[${p.longTasks.join(", ")}]`).join(" ")}`,
  ).toBe(true);
});
