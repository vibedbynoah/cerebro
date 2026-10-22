import AxeBuilder from "@axe-core/playwright";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const open = (page: import("@playwright/test").Page) => page.goto("/?hub=live&mock=online#/hub");

for (const scheme of ["dark", "light"] as const) {
  test.describe(`the Model Hub's verified publishers and recommended file (${scheme})`, () => {
    test.use({ colorScheme: scheme });

    test("a verified publisher carries the badge, with the tooltip that says what it means", async ({ page }) => {
      await open(page);
      await page.getByRole("button", { name: "Coding" }).click();
      const repo = page.locator(".livehub__repo").first();
      await expect(repo).toBeVisible();
      const badge = repo.locator(".verified");
      await expect(badge).toBeVisible();
      await expect(badge).toContainText("Verified");
      await expect(badge).toHaveAttribute("title", /Cerebro's list of known publishers/);
      await expect(badge).toHaveAttribute("title", /not a guarantee/);
      // The icon is there too, so the badge never depends on colour alone.
      await expect(badge.locator(".icon")).toBeVisible();
    });

    test("a repository from an unknown publisher shows its downloads and its age", async ({ page }) => {
      await open(page);
      await page.locator("input[type=search]").fill("Mystery");
      await page.getByRole("button", { name: "Search", exact: true }).click();
      const repo = page.locator(".livehub__repo").first();
      await expect(repo).toBeVisible();
      await expect(repo.locator(".verified")).toHaveCount(0);
      await expect(repo).toContainText("downloads");
      await expect(repo).toContainText("ago");
    });

    test("the filter chip is off by default and leaves only verified publishers", async ({ page }) => {
      await open(page);
      await page.locator("input[type=search]").fill("Mystery");
      await page.getByRole("button", { name: "Search", exact: true }).click();
      await expect(page.locator(".livehub__repo")).toHaveCount(1);
      const chip = page.getByRole("button", { name: "Verified publishers only" });
      await expect(chip).toHaveAttribute("aria-pressed", "false");
      await chip.click();
      await expect(chip).toHaveAttribute("aria-pressed", "true");
      await expect(page.locator(".livehub__repo")).toHaveCount(0);
      await expect(page.locator(".livehub__results")).toContainText("No repository from a verified publisher");
      // The badge is explained in words on the page, not only in a tooltip.
      await expect(page.locator(".livehub__filters")).toContainText("Cerebro's list of known publishers");
    });

    test("one file is recommended for this computer, marked in the list and ready to download", async ({ page }) => {
      await open(page);
      await page.getByRole("button", { name: "Coding" }).click();
      await page.locator(".livehub__repo").first().click();
      const rec = page.locator(".livehub__rec");
      await expect(rec).toBeVisible();
      await expect(rec).toContainText("Recommended for this computer");
      await expect(rec.locator("[data-rec-why]")).toContainText("the best quality that fits your 8 GB graphics card and 32 GB of memory");
      // The same file is the one marked in the list, and it is a split model: one download.
      const row = page.locator(".livehub__table tbody tr[data-recommended='true']");
      await expect(row).toHaveCount(1);
      await expect(row).toContainText("Recommended for this computer");
      await expect(row).toContainText("2 parts, one download");
      // Every other file is still there, one click away.
      await expect(page.locator(".livehub__table tbody tr")).toHaveCount(3);
      // The recommended file is the one the page downloads.
      await rec.getByRole("button", { name: /Download the recommended file/ }).click();
      await expect(row.getByRole("progressbar")).toBeVisible();
      await expect(row).toContainText("In your library", { timeout: 10_000 });
    });

    test("is accessible", async ({ page }) => {
      await open(page);
      await page.getByRole("button", { name: "Coding" }).click();
      await page.locator(".livehub__repo").first().click();
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).include(".livehub").analyze();
      expect(r.violations).toEqual([]);
    });
  });
}
