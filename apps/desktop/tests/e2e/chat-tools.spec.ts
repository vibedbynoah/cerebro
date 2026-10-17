import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function ask(page: Page): Promise<void> {
  await page.goto("/?mock=fast-chat#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
  const box = page.getByRole("textbox", { name: "Message" });
  await box.fill("What files are in my projects folder?");
  await box.press("Enter");
}

const card = (page: Page) => page.locator(".msg--bot").last().locator(".toolcard");

test.describe("tools in chat", () => {
  test("an Ask rule pauses the reply on a card; Allow once runs the tool and the answer uses it", async ({ page }) => {
    await ask(page);
    await expect(card(page)).toHaveAttribute("data-state", "approval", { timeout: 10_000 });
    await expect(card(page)).toContainText("list_directory");
    await expect(card(page)).toContainText("from files");
    await card(page).getByRole("button", { name: "Allow once" }).click();
    await expect(card(page)).toHaveAttribute("data-state", "done");
    await card(page).locator("summary", { hasText: "Result" }).click();
    await expect(card(page).locator("pre").last()).toContainText("todo.md");
    await expect(page.locator(".msg--bot").last()).toContainText("two folders, cerebro and notes", { timeout: 10_000 });
  });

  test("Deny tells the model no, and the reply says so", async ({ page }) => {
    await ask(page);
    await card(page).getByRole("button", { name: "Deny" }).click();
    await expect(card(page)).toHaveAttribute("data-state", "denied");
    await expect(page.locator(".msg--bot").last()).toContainText("was not allowed", { timeout: 10_000 });
  });

  test("Always allow saves the rule: the next call runs without asking, and Tools shows it", async ({ page }) => {
    await ask(page);
    await card(page).getByRole("button", { name: "Always allow" }).click();
    await expect(card(page)).toHaveAttribute("data-state", "done");
    await expect(page.locator(".msg--bot").last().getByTestId("reply-meta")).toBeVisible({ timeout: 10_000 });
    const box = page.getByRole("textbox", { name: "Message" });
    await box.fill("And list the folder again");
    await box.press("Enter");
    await expect(card(page)).toHaveAttribute("data-state", "done", { timeout: 10_000 });
    await page.goto("/#/tools");
    await expect(page.getByRole("combobox", { name: "Rule for list_directory" })).toHaveValue("always");
  });

  test("the waiting card has no axe violations", async ({ page }) => {
    await ask(page);
    await expect(card(page)).toHaveAttribute("data-state", "approval", { timeout: 10_000 });
    await settle(page);
    const r = await new AxeBuilder({ page }).include(".toolcard").withTags(TAGS).analyze();
    expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
  });
});
