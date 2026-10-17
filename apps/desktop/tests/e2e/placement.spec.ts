import AxeBuilder from "@axe-core/playwright";
import { MOCK_LIBRARY } from "../../ui/src/mock/library";
import { expect, openScreen, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const BIG = [...MOCK_LIBRARY].filter((m) => m.kind === "text" && m.arch.layers > 0).sort((a, b) => b.sizeGiB - a.sizeGiB)[0]!;

test.describe("placement", () => {
  test("the load panel says where the model would run, and fewer GPU layers move it to the processor", async ({ page }) => {
    await openScreen(page, "models");
    await page.getByRole("searchbox", { name: "Search models" }).fill(BIG.file);
    await page.locator(`.lib-row[data-model="${BIG.id}"] .lib-name__text`).click();
    await page.getByRole("tab", { name: "Load settings" }).click();
    const card = page.getByRole("region", { name: "Where it would run" });
    await expect(card).toBeVisible({ timeout: 5_000 });
    await expect(card).toContainText("tok/s");
    await expect(card).toContainText("An estimate from each part's speed");
    const gpu = page.getByLabel("GPU layers", { exact: true });
    await gpu.fill("0");
    await gpu.press("Tab");
    await expect(card).toContainText("Processor only", { timeout: 5_000 });
    await settle(page);
    expect((await new AxeBuilder({ page }).include(".placement").withTags(TAGS).analyze()).violations).toEqual([]);
  });
});
