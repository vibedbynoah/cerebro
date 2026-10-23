import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, openFineTune, openSettings, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function openChat(page: Page): Promise<void> {
  await page.goto("/#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
}

const slider = (page: Page) => page.getByRole("slider", { name: "Speed" });
const level = (page: Page) => page.getByTestId("speed-mode");

test.describe("the Speed <-> Accuracy slider", () => {
  test("starts at Balanced, and each level says what it does and sets the controls it covers", async ({ page }) => {
    await openChat(page);
    await expect(level(page)).toHaveText("Balanced");
    await expect(page.locator("#mode-note")).toContainText("The default");

    await slider(page).fill("1");
    await expect(level(page)).toHaveText("Fastest");
    await expect(page.locator("#mode-note")).toContainText("The quickest reply");
    // The notice names what changed.
    await expect(page.getByText("Greedy decoding", { exact: false }).first()).toBeVisible();
    // The fine controls are folded away by default, but the slider still set them.
    await openFineTune(page);
    await expect(page.locator('input[name="effort"][value="low"]')).toBeChecked();
    await expect(page.locator('input[name="verbosity"][value="brief"]')).toBeChecked();
    // The system prompt follows: one line.
    await page.getByRole("button", { name: "System prompt" }).click();
    await expect(page.locator("#base-prompt")).toHaveValue("Answer directly and briefly. No preamble, no restating the question.");

    await openSettings(page);
    await slider(page).fill("5");
    await expect(level(page)).toHaveText("Maximum");
    await expect(page.locator(".composer__check").filter({ hasText: "Orchestrate" }).locator("input")).toBeChecked();
    await expect(page.getByTestId("orchestrate-workers")).toHaveText("5");
    await expect(page.locator('input[name="effort"][value="max"]')).toBeChecked();
  });

  test("changing a control by hand shows Custom; moving the slider picks a level again", async ({ page }) => {
    await openChat(page);
    await slider(page).fill("2");
    await expect(level(page)).toHaveText("Fast");
    await openFineTune(page);
    await page.locator('input[name="effort"][value="high"]').check({ force: true });
    await expect(level(page)).toHaveText("Custom");
    await expect(page.locator("#mode-note")).toContainText("you changed one of the settings in Fine-tune by hand");
    await slider(page).fill("4");
    await expect(level(page)).toHaveText("Accurate");
  });

  test("the level survives a reload", async ({ page }) => {
    await openChat(page);
    await slider(page).fill("1");
    await expect(level(page)).toHaveText("Fastest");
    await page.reload();
    await expect(level(page)).toHaveText("Fastest");
  });

  test("Fine-tune remembers whether it was open", async ({ page }) => {
    await openChat(page);
    // Open by default in the side panel, where there is room.
    const more = page.locator("details.composer__more");
    await expect(more).toHaveJSProperty("open", true);
    await page.locator(".composer__more > summary").click();
    await page.reload();
    await expect(page.locator("details.composer__more")).toHaveJSProperty("open", false);
    await openFineTune(page);
    await page.reload();
    await expect(page.locator("details.composer__more")).toHaveJSProperty("open", true);
  });

  test("is accessible", async ({ page }) => {
    await openChat(page);
    await slider(page).fill("4");
    await openFineTune(page);
    await settle(page);
    const r = await new AxeBuilder({ page }).withTags(TAGS).include(".composer").analyze();
    expect(r.violations).toEqual([]);
  });
});
