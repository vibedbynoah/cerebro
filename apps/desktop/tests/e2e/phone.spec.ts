import AxeBuilder from "@axe-core/playwright";
import { expect, openScreen, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("phone access", () => {
  test("off until a password is set, then pairs with a QR code", async ({ page }) => {
    await openScreen(page, "settings");
    const phone = page.getByTestId("phone-access");
    await phone.scrollIntoViewIfNeeded();
    await expect(page.getByTestId("phone-status")).toHaveText(/Off/);
    await expect(page.getByRole("switch", { name: "Phone access" })).toBeDisabled();

    await phone.getByRole("button", { name: "Generate" }).click();
    const generated = await phone.locator(".phone__generated-value").textContent();
    expect(generated?.split("-")).toHaveLength(6);
    await phone.getByRole("button", { name: "Set password" }).click();
    await expect(phone.locator("summary", { hasText: "Change password" })).toBeVisible();

    await page.getByRole("switch", { name: "Phone access" }).click({ force: true });
    await expect(page.getByTestId("phone-status")).toContainText("https://");
    await phone.getByRole("button", { name: "Pair a phone" }).click();
    await expect(page.getByTestId("phone-qr")).toBeVisible();
    await expect(page.getByTestId("phone-url")).toContainText("#pair=");
    await expect(phone.locator(".phone__qr-caption")).toContainText(/Expires in \d+:\d\d/);
  });

  test("options keep the automatic address when another option changes", async ({ page }) => {
    await openScreen(page, "settings");
    const phone = page.getByTestId("phone-access");
    await phone.scrollIntoViewIfNeeded();
    await phone.locator("summary", { hasText: "Advanced" }).click();
    const chat = page.getByRole("switch", { name: "Allow chat from the phone" });
    await chat.click({ force: true });
    await expect(chat).toBeChecked();
    await expect(phone.locator("select")).toHaveValue("");
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openScreen(page, "settings");
      await scrollThrough(page);
      await page.getByTestId("phone-access").scrollIntoViewIfNeeded();
      await settle(page);
      const results = await new AxeBuilder({ page }).include("[data-testid='phone-access']").withTags(TAGS).analyze();
      expect(results.violations).toEqual([]);
    });
  }
});
