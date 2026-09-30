import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, openScreen, ROUTE_IDS, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function scan(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

for (const scheme of ["dark", "light"] as const) {
  test.describe(`axe (${scheme})`, () => {
    test.use({ colorScheme: scheme });
    for (const id of ROUTE_IDS) {
      test(`${id} has no axe violations`, async ({ page }) => {
        test.setTimeout(60_000);
        await openScreen(page, id);
        await scrollThrough(page);
        await settle(page);
        expect(await scan(page)).toEqual([]);
      });
    }

    test("command palette open has no axe violations", async ({ page }) => {
      await openScreen(page, "home");
      await page.keyboard.press("Control+k");
      await page.getByRole("combobox").fill("se");
      await settle(page);
      expect(await scan(page)).toEqual([]);
    });

    test("collapsed sidebar with a toast has no axe violations", async ({ page }) => {
      await openScreen(page, "settings");
      await page.keyboard.press("Control+b");
      // The collapse animates as a View Transition; let it land before clicking into the new layout.
      await expect(page.locator("nav.sidebar")).toHaveClass(/is-collapsed/);
      await settle(page);
      await page.getByRole("switch", { name: "Offline mode" }).click({ force: true });
      await expect(page.locator(".toast")).toBeVisible();
      await settle(page);
      expect(await scan(page)).toEqual([]);
    });
  });
}
