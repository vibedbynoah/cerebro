import AxeBuilder from "@axe-core/playwright";
import type { Download, Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function axe(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

async function openArena(page: Page): Promise<void> {
  await page.goto("/?mock=fast-chat#/chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
  await page.getByTestId("arena-tab").click();
  await expect(page.locator("main h1")).toHaveText("Arena");
  await expect(page.locator(".arena-model input:checked")).toHaveCount(2);
}

/** Runs one prompt through the two models Arena picks on a first visit. */
async function compare(page: Page, prompt: string): Promise<void> {
  await page.getByTestId("arena-prompt").fill(prompt);
  await page.getByTestId("arena-run").click();
  const replies = page.getByTestId("arena-reply");
  await expect(replies).toHaveCount(2);
  await expect(replies.nth(0).locator(".chip")).toHaveText("Done", { timeout: 25_000 });
  await expect(replies.nth(1).locator(".chip")).toHaveText("Done", { timeout: 25_000 });
}

async function downloadText(download: Download): Promise<string> {
  const stream = await download.createReadStream();
  return Buffer.concat(await stream.toArray()).toString("utf8");
}

test.describe("arena", () => {
  test("asks for two models before it can compare", async ({ page }) => {
    await openArena(page);
    await expect(page.getByRole("button", { name: "Compare answers" })).toBeDisabled();
    await expect(page.getByText("Choose at least two models.")).toHaveCount(0);
    await page.locator(".arena-model input:checked").nth(1).uncheck();
    await expect(page.locator(".arena-model input:checked")).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Compare answers" })).toBeDisabled();
    await expect(page.getByText("Choose at least two models.")).toBeVisible();
    await expect(page.locator(".arena-board")).toContainText("No votes in general yet.");
  });

  test("filters the library but keeps the chosen models in view", async ({ page }) => {
    await openArena(page);
    const search = page.getByPlaceholder("Search models");
    const before = await page.locator(".arena-model input").count();
    await search.fill("Qwen3-0.6B");
    const filtered = await page.locator(".arena-model input").count();
    expect(filtered).toBeGreaterThan(0);
    expect(filtered).toBeLessThan(before);
    await expect(page.locator(".arena-model input:checked")).toHaveCount(2);
    await expect(page.locator(".arena-models__grid")).toHaveCSS("overflow-y", "auto");
    await search.fill("");
    await expect(page.locator(".arena-model input")).toHaveCount(before);
  });

  test("compares blind, reveals the names on a vote, keeps the board and exports the round", async ({ page }) => {
    await openArena(page);
    const prompt = "Explain what a KV cache is in two sentences.";
    await compare(page, prompt);

    // Blind by default: the answer card says so, but no model name appears.
    await expect(page.getByText("Names hidden until the vote")).toBeVisible();
    await expect(page.locator(".arena-reply__model")).toHaveCount(0);
    await expect(page.locator(".arena-reply__body")).toHaveCount(2);
    for (const body of await page.locator(".arena-reply__body").all()) await expect(body).not.toBeEmpty();
    await expect(page.locator(".arena-reply__metrics dd").first()).toContainText(/\d+ ms|—/);

    await page.getByTestId("arena-vote-a").click();
    await expect(page.getByText(/Vote saved for/)).toBeVisible();
    await expect(page.locator(".arena-reply__model")).toHaveCount(2);
    await expect(page.locator(".arena-vote__done")).toContainText("The leaderboard is updated.");

    const rows = page.getByTestId("arena-leader-row");
    await expect(rows).toHaveCount(2);
    await expect(rows.first().locator(".arena-board__rating")).toHaveText((1016).toLocaleString());
    await expect(rows.first().locator("td").nth(2)).toHaveText("1");

    const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("arena-export").click()]);
    expect(download.suggestedFilename()).toMatch(/^cerebro-arena-\d{4}-\d{2}-\d{2}-\d{4}\.json$/);
    const body = JSON.parse(await downloadText(download));
    expect(body).toMatchObject({ format: "cerebro-arena", version: 1, category: "general", categoryLabel: "General", prompt, blind: true });
    expect(body.vote).not.toBe("none");
    expect(body.replies).toHaveLength(2);
    expect(body.replies[0]).toHaveProperty("modelName");
    expect(body.leaderboard).toHaveLength(2);
    expect(body.leaderboard[0]).toMatchObject({ rating: 1016, wins: 1, losses: 0, ties: 0 });

    await page.reload();
    await page.getByTestId("arena-tab").click();
    await expect(page.getByTestId("arena-leader-row")).toHaveCount(2);
    await expect(page.getByTestId("arena-leader-row").first().locator(".arena-board__rating")).toHaveText((1016).toLocaleString());
  });

  test("no axe violations with two finished answers and a saved vote", async ({ page }) => {
    await page.emulateMedia({ colorScheme: "dark" });
    await openArena(page);
    await compare(page, "What is one small way to make a code review kinder?");
    await page.getByTestId("arena-vote-tie").click();
    await expect(page.getByText("Vote saved as a tie.")).toBeVisible();
    await expect(page.getByTestId("arena-leader-row")).toHaveCount(2);
    await scrollThrough(page);
    await settle(page);
    expect(await axe(page)).toEqual([]);
  });
});
