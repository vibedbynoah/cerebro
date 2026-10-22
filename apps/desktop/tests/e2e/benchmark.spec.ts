import AxeBuilder from "@axe-core/playwright";
import type { Download, Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function axe(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

async function downloadText(download: Download): Promise<string> {
  const stream = await download.createReadStream();
  return Buffer.concat(await stream.toArray()).toString("utf8");
}

async function openLab(page: Page): Promise<void> {
  await page.goto("/?mock=fast-bench#/monitor");
  await expect(page.locator("main h1")).toHaveText("Monitor");
  await page.locator("label.segmented__option", { hasText: "Benchmark lab" }).click();
  await expect(page.locator(".bench")).toBeVisible();
  await expect(page.getByTestId("bench-run-result")).toBeVisible();
}

test.describe("benchmark lab", () => {
  test("runs three depths and twenty questions, saves the run, and draws the history and comparison", async ({ page }) => {
    await openLab(page);

    // The web preview starts with two illustrative runs so the chart has something to show.
    await expect(page.getByTestId("bench-saved-run")).toHaveCount(2);
    await expect(page.locator(".bench-chart__dot")).toHaveCount(2);
    await expect(page.locator(".bench-table--compare tbody tr")).toHaveCount(3);
    await expect(page.locator('[data-testid="bench-run-result"]')).toContainText("Illustrative data — not a measurement");

    await page.getByTestId("bench-run").click();
    await expect(page.getByTestId("bench-saved-run")).toHaveCount(3, { timeout: 20_000 });
    await expect(page.locator(".bench-chart__dot")).toHaveCount(3);
    await expect(page.locator('[data-testid="bench-run-result"] .bench-score')).toHaveText(/^\d+\/20 correct$/);
    await expect(page.locator('[data-testid="bench-run-result"] > .bench-table-wrap tbody tr')).toHaveCount(3);
    await expect(page.locator('[data-testid="bench-run-result"] > .bench-table-wrap tbody tr').first()).toContainText("tokens");
    await expect(page.locator('[data-testid="bench-run-result"] .bench-category')).toHaveCount(5);

    await page.locator('[data-testid="bench-run-result"] details.bench-questions summary').click();
    await expect(page.locator(".bench-table--questions tbody tr")).toHaveCount(20);
    await expect(page.locator(".bench-table--questions .chip").first()).toHaveText(/Correct|Wrong/);

    // Both report cards contain the same measured run and say what the data is.
    const [png] = await Promise.all([page.waitForEvent("download"), page.getByTestId("bench-export-png").click()]);
    expect(png.suggestedFilename()).toMatch(/^cerebro-benchmark-.+\.png$/);
    const bytes = Buffer.concat(await (await png.createReadStream()).toArray());
    expect([...bytes.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    await expect(page.locator(".toast").first()).toContainText("Report card saved");

    const [html] = await Promise.all([page.waitForEvent("download"), page.getByTestId("bench-export-html").click()]);
    expect(html.suggestedFilename()).toMatch(/^cerebro-benchmark-.+\.html$/);
    const report = await downloadText(html);
    const optionLabel = (await page.locator(".bench-field--model select option:checked").textContent())?.trim() ?? "";
    const modelName = optionLabel.split(" · ")[0] ?? "";
    expect(modelName).not.toBe("");
    expect(report).toContain("<!doctype html>");
    expect(report).toContain("Cerebro benchmark");
    expect(report).toContain("Speed at three depths");
    expect(report).toContain("Accuracy check");
    expect(report).toContain("Illustrative data — not a measurement.");
    expect(report).toContain("Example data from the web preview.");
    expect(report).not.toContain("Measured on this computer");
    expect(report).toContain(modelName);
    expect(report).not.toContain("<script");

    // The history controls change the chart without touching the stored run.
    await page.locator(".bench-history__controls select").first().selectOption("16384");
    await expect(page.locator(".bench-chart__foot")).toContainText("16K");
    await page.locator(".bench-history__controls select").nth(1).selectOption("prefillTps");
    await expect(page.locator(".bench-chart__foot")).toContainText("Reading speed");

    // A saved run can be reopened from the list.
    await page.getByTestId("bench-saved-run").nth(1).click();
    await expect(page.getByTestId("bench-saved-run").nth(1)).toHaveAttribute("aria-pressed", "true");
  });

  test("keeps Monitor's live view as the default and returns to it", async ({ page }) => {
    await page.goto("/?mock=fast-bench#/monitor");
    await expect(page.locator(".mon-grid .mon-card")).toHaveCount(10);
    await expect(page.locator(".bench")).toHaveCount(0);
    await page.locator("label.segmented__option", { hasText: "Benchmark lab" }).click();
    await expect(page.getByTestId("bench-run-result")).toBeVisible();
    await page.locator("label.segmented__option", { hasText: "Live" }).click();
    await expect(page.locator(".mon-grid .mon-card")).toHaveCount(10);
    await expect(page.locator(".bench")).toHaveCount(0);
  });

  test("the /bench command in Chat opens the lab, and the address says which view is open", async ({ page }) => {
    await page.goto("/?mock=fast-bench#/chat");
    await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
    const input = page.getByLabel("Message");
    await input.fill("/bench");
    // The composer says what Enter will do, so the message is not lost silently.
    await expect(page.getByTestId("composer-command")).toContainText("runs bench on this computer");
    await input.press("Enter");
    await expect(page.locator("main h1")).toHaveText("Monitor");
    await expect(page.locator(".bench")).toBeVisible();
    await expect(page).toHaveURL(/#\/monitor\?tab=benchmark$/);
    // Choosing the live view puts that in the address too.
    await page.locator("label.segmented__option", { hasText: "Live" }).click();
    await expect(page).toHaveURL(/#\/monitor$/);
  });

  test("measures the placement the user picked and says where the model ran", async ({ page }) => {
    await openLab(page);
    const placement = page.locator(".bench-field--placement select");
    await expect(placement).toBeVisible();
    await expect(placement.locator("option")).toHaveCount(4);
    await placement.selectOption("cpu");
    await expect(page.locator(".bench-field--placement .hint")).toContainText("processor");
    await page.getByTestId("bench-run").click();
    await expect(page.getByTestId("bench-saved-run")).toHaveCount(3, { timeout: 20_000 });
    await expect(page.getByTestId("bench-placement")).toContainText("Processor only");
  });

  test("no axe violations with a finished run and every question shown", async ({ page }) => {
    await page.emulateMedia({ colorScheme: "light" });
    await openLab(page);
    await page.getByTestId("bench-run").click();
    await expect(page.getByTestId("bench-saved-run")).toHaveCount(3, { timeout: 20_000 });
    await page.locator('[data-testid="bench-run-result"] details.bench-questions summary').click();
    await scrollThrough(page);
    await settle(page);
    expect(await axe(page)).toEqual([]);
  });
});
