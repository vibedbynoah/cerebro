import type { Page } from "@playwright/test";
import { COMMANDS } from "../../ui/src/lib/chat/commands";
import { expect, test } from "./fixtures";

async function openChat(page: Page): Promise<void> {
  await page.goto("/#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
}
const box = (page: Page) => page.getByRole("textbox", { name: "Message" });
const menu = (page: Page) => page.getByRole("listbox", { name: "Commands" });

test.describe("/ commands", () => {
  test("typing / opens a filtered menu; arrows and Enter pick, Esc closes", async ({ page }) => {
    await openChat(page);
    await box(page).fill("/");
    await expect(menu(page)).toBeVisible();
    await box(page).fill("/sp");
    await expect(menu(page).getByRole("option").first()).toContainText("/speed");
    await box(page).press("Enter");
    await expect(box(page)).toHaveValue("/speed ");
    await expect(menu(page)).toHaveCount(0);
    await box(page).fill("/");
    await box(page).press("Escape");
    await expect(menu(page)).toHaveCount(0);
  });

  test("/speed moves the slider, /help lists every command, a typo says what was meant", async ({ page }) => {
    await openChat(page);
    await box(page).fill("/speed fastest");
    await box(page).press("Enter");
    await expect(page.getByTestId("speed-mode")).toHaveText("Fastest");
    await box(page).fill("/sped 3");
    await box(page).press("Enter");
    await expect(page.getByRole("alert")).toContainText("Did you mean /speed?");
    await box(page).fill("/help");
    await box(page).press("Enter");
    await expect(page.getByTestId("command-help")).toBeVisible();
    await expect(page.getByTestId("command-help").locator("dt")).toHaveCount(COMMANDS.length);
    await page.keyboard.press("Escape");
  });

  test("a mouse click picks a command; commands never reach the model", async ({ page }) => {
    await openChat(page);
    await box(page).fill("/te");
    await menu(page).getByRole("option", { name: /\/temp/ }).click();
    await expect(box(page)).toHaveValue("/temp ");
    await box(page).fill("/temp 0.3");
    await box(page).press("Enter");
    await expect(page.getByText("Temperature 0.3")).toBeVisible();
    await expect(page.locator(".msg")).toHaveCount(0);
  });

  test("/clear empties the chat and Undo brings it back", async ({ page }) => {
    await openChat(page);
    await box(page).fill("Hello there");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).not.toHaveAttribute("aria-busy", "true", { timeout: 15_000 });
    const n = await page.locator(".msg").count();
    expect(n).toBeGreaterThan(1);
    await box(page).fill("/clear");
    await box(page).press("Enter");
    await expect(page.locator(".msg")).toHaveCount(0);
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.locator(".msg")).toHaveCount(n);
  });
});
