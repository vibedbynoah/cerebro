import type { Page } from "@playwright/test";
import { expect, openFineTune, test } from "./fixtures";

async function openChat(page: Page): Promise<void> {
  await page.goto("/?mock=fast-chat#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
}

const box = (page: Page) => page.getByRole("textbox", { name: "Message" });
const docsSwitch = (page: Page) => page.getByRole("checkbox", { name: "Use my documents" });

test.describe("answering from your documents", () => {
  test("the documents are off until the user asks for them", async ({ page }) => {
    await openChat(page);
    await openFineTune(page);
    await expect(docsSwitch(page)).not.toBeChecked();
    await box(page).fill("how many days of annual leave?");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).not.toHaveAttribute("aria-busy", "true", { timeout: 15_000 });
    await expect(page.getByTestId("source-chip")).toHaveCount(0);
  });

  test("/docs turns them on, and the reply shows the passages it used", async ({ page }) => {
    await openChat(page);
    await box(page).fill("/docs ");
    await box(page).press("Enter");
    await expect(page.getByText("Using your documents")).toBeVisible();
    await openFineTune(page);
    await expect(docsSwitch(page)).toBeChecked();

    await box(page).fill("how many days of annual leave?");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).not.toHaveAttribute("aria-busy", "true", { timeout: 15_000 });
    const chip = page.getByTestId("source-chip").first();
    await expect(chip).toBeVisible();
    await expect(chip).toContainText("Employee handbook 2026.txt");
  });

  test("a source chip opens the passage in Documents", async ({ page }) => {
    await openChat(page);
    await box(page).fill("/docs ");
    await box(page).press("Enter");
    await box(page).fill("how many days of annual leave?");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).not.toHaveAttribute("aria-busy", "true", { timeout: 15_000 });
    await page.getByTestId("source-chip").first().click();
    await expect(page.locator("main h1")).toHaveText("Documents");
    await expect(page.locator("#doc-opened")).toContainText("Employee handbook 2026.txt");
    await expect(page.locator("#doc-opened")).toContainText(/annual leave/i);
  });

  test("/docs off turns them off again", async ({ page }) => {
    await openChat(page);
    await box(page).fill("/docs ");
    await box(page).press("Enter");
    await openFineTune(page);
    await expect(docsSwitch(page)).toBeChecked();
    await box(page).fill("/docs off");
    await box(page).press("Enter");
    await expect(page.getByText("Not using your documents")).toBeVisible();
    await expect(docsSwitch(page)).not.toBeChecked();
  });
});

test.describe("quick tools in Chat", () => {
  test("/summarise answers with a summary of the last message", async ({ page }) => {
    await openChat(page);
    await box(page).fill("Backups run every night at 2am and the router is in the hall.");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).not.toHaveAttribute("aria-busy", "true", { timeout: 15_000 });
    await box(page).fill("/summarise ");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).toContainText("Summary", { timeout: 15_000 });
  });

  test("/translate answers in the language that was asked for", async ({ page }) => {
    await openChat(page);
    await box(page).fill("Good morning.");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).not.toHaveAttribute("aria-busy", "true", { timeout: 15_000 });
    await box(page).fill("/translate Spanish");
    await box(page).press("Enter");
    await expect(page.locator(".msg--bot").last()).toContainText("Translation", { timeout: 15_000 });
  });
});
