import AxeBuilder from "@axe-core/playwright";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const open = (page: import("@playwright/test").Page) => page.goto("/?hub=live&mock=online#/hub");

test.describe("the live Model Hub", () => {
  test("opens on recommendations ranked by fit, each with a family tile, abilities and a fit badge", async ({ page }) => {
    await open(page);
    const grid = page.locator(".livehub__pick-grid");
    await expect(grid.locator(".livehub__pick")).toHaveCount(8);
    await expect(grid.locator(".livehub__pick").first().locator(".family-tile")).toBeVisible();
    const tones = await grid.locator(".livehub__pick").evaluateAll((els) => els.map((e) => e.getAttribute("data-fit")));
    const rank = { good: 0, ok: 1, slow: 2, no: 3 } as Record<string, number>;
    expect(tones.map((t) => rank[t!])).toEqual([...tones.map((t) => rank[t!])].sort((a, b) => (a ?? 0) - (b ?? 0)));
  });

  test("a category searches, a repository lists its files, and a download lands in the library", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "Coding" }).click();
    await expect(page.getByRole("button", { name: "Coding" })).toHaveAttribute("aria-pressed", "true");
    const repo = page.locator(".livehub__repo").first();
    await expect(repo).toBeVisible();
    await expect(repo.locator(".livehub__badge").first()).toBeVisible();
    await repo.click();
    const row = page.locator(".livehub__table tbody tr").first();
    await expect(row).toContainText("Q4_K_M");
    await row.getByRole("button", { name: "Download" }).click();
    await expect(row.getByRole("progressbar")).toBeVisible();
    await expect(row).toContainText("In your library", { timeout: 10_000 });
    await page.getByRole("button", { name: "Back to recommendations" }).click();
    await expect(page.getByRole("heading", { name: "Recommended for this computer" })).toBeVisible();
  });

  test("a pick searches for its files", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "See files for Qwen3.5 4B" }).click();
    await expect(page.locator("input[type=search]")).toHaveValue("Qwen3.5 4B GGUF");
    await expect(page.locator(".livehub__repo").first()).toContainText("Qwen3.5-4B");
  });

  test("is accessible", async ({ page }) => {
    await open(page);
    await settle(page);
    const r = await new AxeBuilder({ page }).withTags(TAGS).include(".livehub").analyze();
    expect(r.violations).toEqual([]);
  });
});
