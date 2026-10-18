import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, openScreen, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

/** Runs a tune to the end. The mock replays the recorded run: one trial every 300 ms, then the outcome. */
async function tuneToEnd(page: Page): Promise<void> {
  await openScreen(page, "turbo");
  // The mock starts with a model in the chat slot (like chat.spec.ts), so a tune can run at once.
  const start = page.getByRole("button", { name: "Tune for this machine" });
  await expect(start).toBeEnabled();
  await start.click();
  await expect(page.getByTestId("machine-tune-note")).toHaveText("As fast with 4 fewer threads, so it runs cooler.", { timeout: 20_000 });
}

test.describe("machine tune", () => {
  test("times the processor settings on the loaded model and keeps the fastest", async ({ page }) => {
    await openScreen(page, "turbo");
    await expect(page.locator("#machine-tune-title")).toHaveText("Tune for this machine");
    const start = page.getByRole("button", { name: "Tune for this machine" });
    await expect(start).toBeEnabled();
    await start.click();

    // While it runs: a Stop button and a live region announcing every finished trial.
    await expect(page.getByRole("button", { name: "Stop" })).toBeVisible();
    await expect(page.locator(".machine-tune__trials")).toHaveAttribute("aria-live", "polite");
    await expect(page.locator(".machine-tune__trial").first()).toBeVisible({ timeout: 10_000 });

    // The finish: the note as the headline, the speed line, the kept settings and the machine.
    await expect(page.getByTestId("machine-tune-note")).toHaveText("As fast with 4 fewer threads, so it runs cooler.", { timeout: 20_000 });
    await expect(page.locator(".machine-tune__speed")).toHaveText("47.2 → 48.2 tok/s (+2.2%)");
    await expect(page.locator(".machine-tune__settings")).toContainText("12 threads, 4 pieces per thread, four-row dots off");
    await expect(page.locator(".machine-tune__machine")).toContainText("i9-12900H");
    await expect(page.locator(".machine-tune__took")).toHaveText("Took 52 s.");
    await expect(page.getByRole("button", { name: "Stop" })).toHaveCount(0);

    // Every trial of the recorded run, grouped under the four phases, one marked fastest per group.
    await expect(page.locator(".machine-tune__phase-title")).toHaveText(["Threads", "Work splitting", "Four-row dots", "Confirm"]);
    await expect(page.locator(".machine-tune__trial")).toHaveCount(12);
    await expect(page.locator('.machine-tune__trial[data-fastest="true"]')).toHaveCount(4);

    // The kept setting reached the engine switches list, which refetches after a tune.
    await expect(page.locator("#sw-cpu_threads")).toHaveValue("12");

    // Coming back to the screen (without a reload) shows the saved outcome again.
    await page.locator('a.nav-link[data-route="home"]').click();
    await page.locator('a.nav-link[data-route="turbo"]').click();
    await expect(page.getByTestId("machine-tune-note")).toHaveText("As fast with 4 fewer threads, so it runs cooler.");
    await expect(page.locator(".machine-tune__settings")).toContainText("12 threads, 4 pieces per thread, four-row dots off");
  });

  for (const theme of ["dark", "light"] as const) {
    test(`no axe violations with a finished tune (${theme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: theme });
      await tuneToEnd(page);
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
