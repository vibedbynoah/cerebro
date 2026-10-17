import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

// The orchestrate mock runs at watchable speed: every role lasts about two seconds.
async function openChat(page: Page): Promise<void> {
  await page.goto("/?mock=orchestrate#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
}

const toggle = (page: Page) => page.locator(".composer__check").filter({ hasText: "Orchestrate" }).locator("input");
const bot = (page: Page) => page.locator(".msg--bot").last();
const step = (page: Page) => bot(page).getByTestId("ledger-step");

async function send(page: Page, text: string): Promise<void> {
  const box = page.getByRole("textbox", { name: "Message" });
  await box.fill(text);
  await box.press("Enter");
}

test.describe("orchestrate in chat", () => {
  test("the toggle explains itself and sends the run: steps advance, Notes open, the answer arrives, the panel folds away", async ({ page }) => {
    await openChat(page);
    await expect(page.getByText("Several instances of this model plan, split the work and check each other through a shared ledger.")).toHaveCount(0);
    await toggle(page).check();
    await expect(page.getByText("Several instances of this model plan, split the work and check each other through a shared ledger. Slower, often better on hard problems.")).toBeVisible();
    await expect(page.getByTestId("orchestrate-workers")).toHaveText("3");
    await expect(page.getByTestId("orchestrate-rounds")).toHaveText("3");
    await page.getByRole("button", { name: "More workers" }).click();
    await page.getByRole("button", { name: "Fewer workers" }).click();
    await expect(page.getByTestId("orchestrate-workers")).toHaveText("3");

    await send(page, "What is 14 times 7?");
    // Every role takes its turn through the ledger, one after another.
    await expect(step(page)).toHaveText("Plan, round 0", { timeout: 10_000 });
    await expect(page.locator(".ledger__count")).toHaveText("1/4 files");
    await expect(step(page)).toHaveText("Ideate, round 0");
    await expect(step(page)).toHaveText("Manage, round 1");
    await expect(step(page)).toHaveText("Worker 1, round 1");
    await page.locator(".ledger__summary").click();
    await page.locator(".ledger__tab", { hasText: "Notes" }).click();
    await expect(page.locator(".ledger__text")).toContainText("Ideation:");
    await expect(step(page)).toHaveText("Worker 2, round 1");
    await expect(step(page)).toHaveText("Manage, round 2", { timeout: 20_000 });
    await expect(step(page)).toHaveText("Finalize, round 3", { timeout: 20_000 });

    // The final result.md is the reply itself.
    await expect(bot(page).getByTestId("reply-meta")).toBeVisible({ timeout: 30_000 });
    await expect(bot(page).locator(".md")).toContainText("The same result comes from regrouping");
    // Done: the panel is collapsed again, with the tasks checklist one tab away.
    await expect(bot(page).locator(".ledger")).toHaveJSProperty("open", false);
    await page.locator(".ledger__summary").click();
    await page.locator(".ledger__tab", { hasText: "Tasks" }).click();
    await expect(page.getByTestId("ledger-tasks").locator(".ledger__task")).toHaveCount(3);
    await expect(page.getByTestId("ledger-tasks").locator("li[data-done='true']")).toHaveCount(3);
  });

  test("Stop mid-run keeps the partial ledger and the folder path", async ({ page }) => {
    await openChat(page);
    await toggle(page).check();
    await send(page, "A goal that takes a while");
    await expect(step(page)).toHaveText("Plan, round 0", { timeout: 10_000 });
    await expect(step(page)).toHaveText(/Ideate/);
    await page.getByRole("button", { name: "Stop generating" }).click();
    await expect(bot(page).getByTestId("reply-meta")).toContainText("stopped", { timeout: 10_000 });
    // No answer was produced, but the team's work so far is still readable.
    await expect(bot(page).locator(".md")).toHaveCount(0);
    await page.locator(".ledger__summary").click();
    await expect(bot(page).getByTestId("ledger-dir")).toContainText("orchestrate");
    await expect(bot(page).locator(".ledger__tab", { hasText: "Plan" })).toBeVisible();
    await expect(bot(page).locator(".ledger__text")).toContainText("Split the product");
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with a finished Orchestrate reply (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openChat(page);
      await toggle(page).check();
      await send(page, "What is 14 times 7?");
      await expect(bot(page).getByTestId("reply-meta")).toBeVisible({ timeout: 40_000 });
      await page.locator(".ledger__summary").click();
      await page.locator(".ledger__tab", { hasText: "Tasks" }).click();
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
