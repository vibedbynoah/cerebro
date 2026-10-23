import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, settle, test, openSettings } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function openChat(page: Page): Promise<void> {
  await page.goto("/#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
}
const dock = (page: Page) => page.getByTestId("live-dock");

test.describe("the live panel in Chat", () => {
  test("opens from the header and with Ctrl+., shows live numbers, and is remembered", async ({ page }) => {
    await openChat(page);
    await expect(dock(page)).toHaveCount(0);
    await page.getByRole("button", { name: /^Live panel/ }).click();
    await expect(dock(page)).toBeVisible();
    await expect(dock(page).getByTestId("dock-tps")).not.toHaveText("");
    await expect(dock(page).getByRole("meter", { name: "Graphics memory used" })).toBeVisible();
    await page.reload();
    await expect(dock(page)).toBeVisible();
    await page.keyboard.press("Control+.");
    await expect(dock(page)).toHaveCount(0);
    await page.keyboard.press("Control+.");
    await expect(dock(page)).toBeVisible();
  });

  test("quick settings change the same settings as the composer", async ({ page }) => {
    await openChat(page);
    await page.getByRole("button", { name: /^Live panel/ }).click();
    const speed = dock(page).getByRole("slider", { name: /Speed or accuracy/ });
    await speed.fill("1");
    await expect(dock(page).getByRole("checkbox", { name: "Think before answering" })).not.toBeChecked();
    // The Settings tab shows the same level.
    await openSettings(page);
    await expect(page.getByTestId("speed-mode")).toHaveText("Fastest");
    await page.getByRole("button", { name: /^Live panel/ }).click();
    await dock(page).getByRole("checkbox", { name: "Think before answering" }).check();
    await openSettings(page);
    await expect(page.getByTestId("speed-mode")).toHaveText("Custom");
    await page.getByRole("button", { name: /^Live panel/ }).click();
    await dock(page).getByRole("button", { name: "System prompt" }).click();
    await expect(page.locator("#base-prompt")).toBeVisible();
  });

  test("resizes with the keyboard and stays within its limits", async ({ page }) => {
    await openChat(page);
    await page.getByRole("button", { name: /^Live panel/ }).click();
    const side = page.locator(".chat-side--dock");
    const w0 = (await side.boundingBox())!.width;
    const handle = page.getByRole("separator", { name: "Resize the live panel" });
    await handle.focus();
    for (let i = 0; i < 3; i++) await handle.press("ArrowLeft");
    const w1 = (await side.boundingBox())!.width;
    expect(w1).toBeGreaterThan(w0 + 40);
    for (let i = 0; i < 40; i++) await handle.press("ArrowRight");
    expect((await side.boundingBox())!.width).toBeGreaterThanOrEqual(279);
  });

  test("is accessible in both themes", async ({ page }) => {
    await openChat(page);
    await page.getByRole("button", { name: /^Live panel/ }).click();
    for (const scheme of ["dark", "light"] as const) {
      await page.emulateMedia({ colorScheme: scheme });
      await page.reload();
      await expect(page.getByTestId("live-dock")).toBeVisible();
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).include(".chat-side").analyze();
      expect(r.violations, scheme).toEqual([]);
    }
  });
});
