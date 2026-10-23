import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, openFineTune, settle, test, enableTools } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function send(page: Page, text: string, plan = false): Promise<void> {
  await page.goto("/?mock=fast-chat#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
  await enableTools(page);
  // Plan first is a saved preference: set it either way.
  await openFineTune(page);
  await page.getByRole("checkbox", { name: "Plan first" }).setChecked(plan);
  const box = page.getByRole("textbox", { name: "Message" });
  await box.fill(text);
  await box.press("Enter");
}

const cards = (page: Page) => page.locator(".msg--bot").last().locator(".toolcard");

test.describe("agent tools in chat", () => {
  test("a question waits for the answer, which reaches the model", async ({ page }) => {
    await send(page, "Set up storage, ask me what you need.");
    const card = cards(page).first();
    await expect(card).toHaveAttribute("data-state", "question", { timeout: 10_000 });
    await expect(card).toContainText("Which database should the app use?");
    const sendBtn = card.getByRole("button", { name: "Send answer" });
    await expect(sendBtn).toBeDisabled();
    await card.getByRole("radio", { name: "SQLite (recommended)" }).check();
    await sendBtn.click();
    await expect(card).toHaveAttribute("data-state", "done");
    await expect(page.locator(".msg--bot").last()).toContainText("Going with SQLite (recommended).", { timeout: 10_000 });
  });

  test("a typed answer replaces the chosen option", async ({ page }) => {
    await send(page, "ask me first");
    const card = cards(page).first();
    await expect(card).toHaveAttribute("data-state", "question", { timeout: 10_000 });
    await card.getByRole("radio", { name: "Postgres" }).check();
    await card.getByRole("textbox", { name: /Another answer/ }).fill("DuckDB");
    await card.getByRole("button", { name: "Send answer" }).click();
    await expect(page.locator(".msg--bot").last()).toContainText("Going with DuckDB.", { timeout: 10_000 });
  });

  test("plan first: a task list, then a plan to approve; approving turns plan mode off", async ({ page }) => {
    await send(page, "Fix the config crash", true);
    const todo = cards(page).first();
    await expect(todo.getByRole("list", { name: "Task list" })).toContainText("Draft the change", { timeout: 10_000 });
    await expect(todo.locator("li[data-status=completed]")).toHaveText(/Read the config loader/);
    const plan = cards(page).nth(1);
    await expect(plan).toHaveAttribute("data-state", "plan", { timeout: 10_000 });
    await expect(plan.locator(".toolcard__plan-body")).toContainText("Return an error from load()");
    await plan.getByRole("button", { name: "Approve plan" }).click();
    await expect(page.getByRole("checkbox", { name: "Plan first" })).not.toBeChecked();
    await expect(page.locator(".msg--bot").last()).toContainText("Starting with step 1", { timeout: 10_000 });
  });

  test("keep planning sends what to change", async ({ page }) => {
    await send(page, "Fix the config crash", true);
    const plan = cards(page).nth(1);
    await expect(plan).toHaveAttribute("data-state", "plan", { timeout: 10_000 });
    await plan.getByRole("textbox", { name: "What should change in the plan" }).fill("skip the Settings screen");
    await plan.getByRole("button", { name: "Keep planning" }).click();
    await expect(page.getByRole("checkbox", { name: "Plan first" })).toBeChecked();
    await expect(page.locator(".msg--bot").last()).toContainText("Revising the plan: skip the Settings screen", { timeout: 10_000 });
  });

  test("finished calls show up in the Tools screen's activity", async ({ page }) => {
    await send(page, "ask me which one");
    const card = cards(page).first();
    await card.getByRole("radio", { name: "Postgres" }).check({ timeout: 10_000 });
    await card.getByRole("button", { name: "Send answer" }).click();
    await expect(card).toHaveAttribute("data-state", "done");
    await page.locator('a.nav-link[data-route="tools"]').click();
    await expect(page.locator(".tools-log li").first()).toContainText("Which database should the app use?");
    await expect(page.locator(".tools-log li").first().locator(".chip")).toHaveText("Ran");
  });

  test("a plan card has no accessibility violations", async ({ page }) => {
    await send(page, "Fix the config crash", true);
    await expect(cards(page).nth(1)).toHaveAttribute("data-state", "plan", { timeout: 10_000 });
    await settle(page);
    expect((await new AxeBuilder({ page }).include(".msg--bot").withTags(TAGS).analyze()).violations).toEqual([]);
  });

  test("a question card has no accessibility violations", async ({ page }) => {
    await send(page, "ask me");
    await expect(cards(page).first()).toHaveAttribute("data-state", "question", { timeout: 10_000 });
    await settle(page);
    expect((await new AxeBuilder({ page }).include(".msg--bot").withTags(TAGS).analyze()).violations).toEqual([]);
  });
});
