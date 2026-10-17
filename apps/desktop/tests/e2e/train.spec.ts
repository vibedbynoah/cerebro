import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function open(page: Page): Promise<void> {
  await page.goto("/?mock=fast-train#/train");
  await expect(page.locator("main h1")).toHaveText("Train");
}

test.describe("train", () => {
  test("setup, then a run streams steps and loss and lands a model in the library", async ({ page }) => {
    await open(page);
    const start = page.getByRole("button", { name: "Start training" });
    await expect(start).toBeDisabled();
    await page.getByRole("button", { name: "Set up training" }).click();
    await expect(page.locator(".train-ready .chip")).toHaveText("Ready", { timeout: 10_000 });
    await expect(page.locator(".train-ready")).toContainText("PyTorch 2.11.0+cu128");

    await page.getByLabel("Name").fill("support bot");
    await page.getByLabel("Dataset file path").fill("C:/data/examples.jsonl");
    await page.getByLabel("Steps").fill("12");
    await start.click();

    const run = page.locator(".train-run");
    await expect(run.locator(".chip")).toHaveText("Finished", { timeout: 15_000 });
    await expect(run.locator(".train-figures")).toContainText("12 / 12");
    await expect(run.getByRole("progressbar", { name: "Training steps" })).toHaveAttribute("aria-valuenow", "12");
    await expect(run.locator(".train-models")).toContainText("unsloth.Q4_K_M.gguf");
    await expect(page.getByText("support bot is in your model library.")).toBeVisible();
  });

  test("the form names each problem before anything starts", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "Set up training" }).click();
    await expect(page.locator(".train-ready .chip")).toHaveText("Ready", { timeout: 10_000 });
    await page.getByLabel("Base model").fill("../escape");
    await page.getByRole("button", { name: "Start training" }).click();
    await expect(page.getByText("Fix the highlighted settings first.")).toBeVisible();
    await expect(page.getByText("A Hugging Face id like unsloth/Qwen3-0.6B.")).toBeVisible();
    await expect(page.getByText("Pick one dataset")).toBeVisible();
    await expect(page.locator(".train-run")).toHaveCount(0);
  });

  test("stop ends the run after the current step", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "Set up training" }).click();
    await expect(page.locator(".train-ready .chip")).toHaveText("Ready", { timeout: 10_000 });
    await page.getByLabel("Name").fill("long run");
    await page.getByRole("radio", { name: "A Hugging Face dataset" }).check();
    await page.getByLabel("Hugging Face dataset id").fill("tatsu-lab/alpaca");
    await page.getByLabel("Steps").fill("5000");
    await page.getByRole("button", { name: "Start training" }).click();
    await expect(page.locator(".train-run .chip")).toHaveText("Training", { timeout: 10_000 });
    await page.getByRole("button", { name: "Stop" }).click();
    await expect(page.locator(".train-run .chip")).toHaveText("Stopped", { timeout: 10_000 });
    await expect(page.locator(".train-run")).toContainText("Only the adapter was saved.");
  });

  test("has no accessibility violations, idle and finished", async ({ page }) => {
    await open(page);
    await scrollThrough(page);
    await settle(page);
    expect((await new AxeBuilder({ page }).withTags(TAGS).analyze()).violations).toEqual([]);
    await page.getByRole("button", { name: "Set up training" }).click();
    await expect(page.locator(".train-ready .chip")).toHaveText("Ready", { timeout: 10_000 });
    await page.getByLabel("Name").fill("a11y");
    await page.getByLabel("Dataset file path").fill("C:/d.jsonl");
    await page.getByLabel("Steps").fill("4");
    await page.getByRole("button", { name: "Start training" }).click();
    await expect(page.locator(".train-run .chip")).toHaveText("Finished", { timeout: 10_000 });
    await scrollThrough(page);
    await settle(page);
    expect((await new AxeBuilder({ page }).withTags(TAGS).analyze()).violations).toEqual([]);
  });
});
