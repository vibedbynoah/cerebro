import AxeBuilder from "@axe-core/playwright";
import { expect, openScreen, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("turbo", () => {
  test("advanced shows profiles, engines, schedulers, profiler and channels", async ({ page }) => {
    await openScreen(page, "turbo");
    await expect(page.locator("main h1")).toHaveText("Turbo");
    for (const id of ["profiles", "engines", "schedulers", "profiler", "channels"]) await expect(page.locator(`#${id}-title`)).toBeVisible();
    await expect(page.locator(".turbo-engine")).toHaveCount(6);
    await expect(page.getByTestId("bottleneck")).toContainText("CPU-bound on experts");
  });

  test("a profile switch lists what changed and reverts in one click", async ({ page }) => {
    await openScreen(page, "turbo");
    await page.locator(".turbo-profile", { hasText: "Max speed" }).click();
    await expect(page.locator(".turbo-table tbody tr")).toHaveCount(3);
    await expect(page.locator("#profile-result")).toContainText("Max speed");
    await page.getByRole("button", { name: "Revert" }).click();
    await expect(page.locator("#profile-result")).toContainText("Back to Balanced");
    await expect(page.locator(".turbo-profile[data-active='true']")).toContainText("Balanced");
  });

  test("a slower build is never promoted and says why", async ({ page }) => {
    await openScreen(page, "turbo");
    const slow = page.locator(".turbo-engine[data-engine='cerebro-expert-cache']");
    await expect(slow).toContainText("-69% vs control");
    await expect(slow.locator(".hint")).toContainText(/channel|control stays/);
  });

  test("beginner sees only the profile picker", async ({ page }) => {
    await page.goto("/#/settings");
    await page.locator("label.segmented__option", { hasText: "Beginner" }).click();
    await openScreen(page, "turbo");
    await expect(page.getByTestId("turbo-beginner")).toBeVisible();
    await expect(page.locator("#engines-title")).toHaveCount(0);
    await page.getByRole("button", { name: "Show everything" }).click();
    await expect(page.locator("#engines-title")).toBeVisible();
  });

  for (const theme of ["dark", "light"] as const) {
    test(`no axe violations (${theme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: theme });
      await openScreen(page, "turbo");
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
