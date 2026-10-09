import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function open(page: Page): Promise<void> {
  await page.goto("/?mock=fast-plugins&mock=online#/plugins");
  await expect(page.locator("main h1")).toHaveText("Plugins");
}

test.describe("plugins", () => {
  test("one-click install animates into Enabled and lands in Installed", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "Install SQLite" }).click();
    await expect(page.locator(".plug-card[data-plugin='sqlite'] .plug-enabled")).toBeVisible({ timeout: 10_000 });
    await page.getByRole("tab", { name: /Installed/ }).click();
    await expect(page.locator(".plug-row[data-plugin='sqlite']")).toBeVisible();
    await page.reload();
    await page.getByRole("tab", { name: /Installed/ }).click();
    await expect(page.locator(".plug-row[data-plugin='sqlite']")).toBeVisible();
  });

  test("category and search filter the catalogue", async ({ page }) => {
    await open(page);
    await page.locator(".plug-cats").getByRole("button", { name: "Voice" }).click();
    await expect(page.locator(".plug-grid .plug-card")).toHaveCount(2);
    await page.locator(".plug-cats").getByRole("button", { name: "All" }).click();
    await page.getByRole("searchbox", { name: "Search plugins" }).fill("postgres");
    await expect(page.locator(".plug-grid .plug-card")).toHaveCount(1);
  });

  test("details list abilities, permissions, licence and the third-party note", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "Details for Mail" }).click();
    const d = page.getByRole("dialog", { name: "Mail" });
    await expect(d).toContainText("Permissions it asks for");
    await expect(d).toContainText("OS keychain");
    await expect(d).toContainText("MIT");
    await expect(d).toContainText("authors' responsibility");
    await page.getByRole("button", { name: "Close details" }).click();
    await expect(d).toHaveCount(0);
  });

  test("installed plugins can be switched off, updated and removed; a local file installs offline", async ({ page }) => {
    await open(page);
    await page.getByRole("tab", { name: /Installed/ }).click();
    await expect(page.locator(".plug-row[data-plugin='git']")).toContainText("update to 1.1.2 available");
    await page.locator('[data-setting="plugin-git"]').click();
    await expect(page.locator('[data-setting="plugin-git"]')).toHaveAttribute("data-state", "unchecked");
    await page.locator('.plug-file input[type="file"]').setInputFiles({ name: "my-tool.cbplugin", mimeType: "application/zip", buffer: Buffer.from("PK") });
    await expect(page.locator(".plug-installed")).toContainText("my-tool.cbplugin");
    await page.locator(".plug-row[data-plugin='git']").getByRole("button", { name: "Remove" }).click();
    await expect(page.locator(".plug-row[data-plugin='git']")).toHaveCount(0);
  });

  test("with offline mode on, catalogue installs explain why and file installs still work", async ({ page }) => {
    await open(page);
    await page.getByRole("link", { name: "Settings" }).first().click();
    await page.locator('[data-setting="setting-offlineMode"]').click();
    await page.getByRole("link", { name: "Plugins" }).first().click();
    await expect(page.locator(".notice")).toContainText("Offline");
    await page.getByRole("button", { name: "Install SQLite" }).click();
    await expect(page.locator(".plug-card[data-plugin='sqlite'] .plug-enabled")).toHaveCount(0);
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await open(page);
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
