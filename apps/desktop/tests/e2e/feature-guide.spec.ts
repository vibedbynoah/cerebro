import AxeBuilder from "@axe-core/playwright";
import { FEATURE_GROUPS } from "../../ui/src/lib/features";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("the feature guide", () => {
  test.use({ guideFirst: true });

  test("opens on entering the app, lists every feature, and the X closes it for good", async ({ page }) => {
    await page.goto("/#/home");
    const guide = page.getByTestId("feature-guide");
    await expect(guide).toBeVisible();
    await expect(guide.getByRole("heading", { name: "What Cerebro does" })).toBeVisible();
    const count = FEATURE_GROUPS.reduce((n, g) => n + g.features.length, 0);
    await expect(guide.locator(".guide__item")).toHaveCount(count);
    await guide.getByRole("button", { name: "Close" }).click();
    await expect(guide).toHaveCount(0);
    await page.reload();
    await expect(page.locator(".app")).toBeVisible();
    await expect(page.getByTestId("feature-guide")).toHaveCount(0);
  });

  test("Open takes you to the feature's screen, and the palette brings the guide back", async ({ page }) => {
    await page.goto("/#/home");
    await page.getByRole("button", { name: "Open Image Studio" }).click();
    await expect(page).toHaveURL(/#\/images/);
    await expect(page.getByTestId("feature-guide")).toHaveCount(0);
    await page.keyboard.press("Control+k");
    await page.getByRole("option", { name: /^What Cerebro does/ }).first().click();
    await expect(page.getByTestId("feature-guide")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("feature-guide")).toHaveCount(0);
  });

  test("Home's What's new opens the guide", async ({ page }) => {
    await page.goto("/#/home");
    await page.getByTestId("feature-guide").getByRole("button", { name: "Close" }).click();
    await page.getByRole("button", { name: "See everything Cerebro does" }).click();
    await expect(page.getByTestId("feature-guide")).toBeVisible();
  });

  test("is accessible", async ({ page }) => {
    await page.goto("/#/home");
    await expect(page.getByTestId("feature-guide")).toBeVisible();
    await settle(page);
    const r = await new AxeBuilder({ page }).withTags(TAGS).include("[data-testid=feature-guide]").analyze();
    expect(r.violations).toEqual([]);
  });
});
