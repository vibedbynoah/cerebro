import AxeBuilder from "@axe-core/playwright";
import { expect, openScreen, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("security screen", () => {
  test("shows integrity, folder audit, containment, key, updates and activity", async ({ page }) => {
    await openScreen(page, "security");
    for (const id of ["protections", "integrity", "audit", "sandbox", "api-key", "updates", "activity"]) await expect(page.locator(`#${id}-title`)).toBeVisible();
    await expect(page.locator("#integrity-title").locator("..").locator("..")).toContainText("5 of 5 files match");
  });

  test("hashes are hidden until asked for", async ({ page }) => {
    await openScreen(page, "security");
    await expect(page.locator("#hash-table")).toHaveCount(0);
    await page.getByRole("button", { name: "Show file hashes" }).click();
    await expect(page.locator("#hash-table tbody tr")).toHaveCount(5);
  });

  test("a new API key is shown once, masked until revealed, copied only after a confirm, and never stored", async ({ page }) => {
    // The page may write the clipboard but never read it (Permissions-Policy), so the test records
    // what the page writes instead of reading it back.
    await page.addInitScript(() => {
      const w = window as unknown as { __copied: string[] };
      w.__copied = [];
      Object.defineProperty(navigator, "clipboard", { value: { writeText: async (t: string) => void w.__copied.push(t) }, configurable: true });
    });
    await openScreen(page, "security");
    const key = page.getByTestId("api-key");
    await expect(key).toHaveText(/^cbr_\w+…\w{4}$/);
    await page.getByRole("button", { name: "Create new key" }).click();
    await expect(key).toHaveText(/^cbr_\w{0,2}…\w{4}$/);
    await page.getByRole("button", { name: "Reveal" }).click();
    const full = await key.innerText();
    expect(full).toMatch(/^cbr_[0-9a-f]{64}$/);
    await page.getByRole("button", { name: "Copy" }).click();
    await expect(page.getByRole("button", { name: "Click again to copy" })).toBeVisible();
    await page.getByRole("button", { name: "Click again to copy" }).click();
    expect(await page.evaluate(() => (window as unknown as { __copied: string[] }).__copied)).toEqual([full]);
    const stored = await page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
    expect(stored).not.toContain(full.slice(4));
    // Leaving the screen drops it: coming back shows only the mask.
    await openScreen(page, "home");
    await openScreen(page, "security");
    await expect(page.getByTestId("api-key")).not.toHaveText(full);
    await expect(page.getByRole("button", { name: "Reveal" })).toHaveCount(0);
  });

  test("activity can be filtered", async ({ page }) => {
    await openScreen(page, "security");
    await page.getByRole("searchbox", { name: "Filter security activity" }).fill("rotated");
    await expect(page.locator(".sec-log li")).toHaveCount(1);
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with hashes shown (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openScreen(page, "security");
      await page.getByRole("button", { name: "Show file hashes" }).click();
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
