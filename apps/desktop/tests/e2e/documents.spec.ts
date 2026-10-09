import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function open(page: Page): Promise<void> {
  await page.goto("/?mock=fast-docs#/documents");
  await expect(page.locator("main h1")).toHaveText("Documents");
}

test.describe("documents", () => {
  test("search returns the passage that answers, with its file, as text", async ({ page }) => {
    await open(page);
    await page.getByRole("searchbox", { name: "Search documents" }).fill("annual leave days");
    const hit = page.locator(".doc-hit").first();
    await expect(hit).toContainText("Employee handbook 2026.pdf");
    await expect(hit.locator("mark").first()).toHaveText(/annual/i);
  });

  test("a dropped text file indexes with progress, becomes searchable and can be removed", async ({ page }) => {
    await open(page);
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("The wifi password for the office is printed on the router. Backups run every night at 2am.") });
    await expect(page.locator(".doc-row", { hasText: "notes.txt" }).locator(".chip")).toHaveText("Ready", { timeout: 10_000 });
    await page.getByRole("searchbox", { name: "Search documents" }).fill("when do backups run");
    await expect(page.locator(".doc-hit").first()).toContainText("notes.txt");
    await page.getByRole("button", { name: "Remove notes.txt" }).click();
    await expect(page.locator(".doc-row", { hasText: "notes.txt" })).toHaveCount(0);
  });

  test("unsupported files are refused with a reason; markup in a file stays text", async ({ page }) => {
    await open(page);
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "tool.exe", mimeType: "application/octet-stream", buffer: Buffer.from("MZ") });
    await expect(page.getByText("tool.exe not added")).toBeVisible();
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "x.html", mimeType: "text/html", buffer: Buffer.from("<img src=x onerror=alert(1)> secret plan inside.") });
    await expect(page.locator(".doc-row", { hasText: "x.html" }).locator(".chip")).toHaveText("Ready", { timeout: 10_000 });
    await page.getByRole("searchbox", { name: "Search documents" }).fill("secret plan");
    await expect(page.locator(".doc-hit img")).toHaveCount(0);
    await expect(page.locator(".doc-hit").first()).toContainText("<img src=x onerror=alert(1)>");
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with results (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await open(page);
      await page.getByRole("searchbox", { name: "Search documents" }).fill("restart server");
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
