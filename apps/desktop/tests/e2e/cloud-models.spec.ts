import AxeBuilder from "@axe-core/playwright";
import { expect, settle, test, gotoSettings } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("cloud models", () => {
  test("add a provider: a bad key says why, a good key lists models, the chosen ones appear in Chat with a badge", async ({ page }) => {
    await gotoSettings(page);
    const panel = page.getByTestId("cloud-models");
    await panel.getByRole("combobox", { name: "Add a provider" }).selectOption("openai");
    await expect(panel.getByRole("link", { name: /Create one on OpenAI's site/ })).toHaveAttribute("href", "https://platform.openai.com/api-keys");
    await panel.getByLabel(/API key/).fill("bad");
    await panel.getByRole("button", { name: "Test and list models" }).click();
    await expect(panel.getByRole("alert")).toContainText("refused the key");
    await panel.getByLabel(/API key/).fill("sk-good");
    await panel.getByRole("button", { name: "Test and list models" }).click();
    await expect(panel.getByRole("checkbox", { name: "example-large" })).toBeChecked();
    await panel.getByRole("checkbox", { name: "example-vision" }).uncheck();
    await panel.getByRole("button", { name: "Save" }).click();
    await expect(panel.getByText("key stored in Windows Credential Manager")).toBeVisible();

    await page.goto("/#/chat");
    const select = page.getByRole("combobox", { name: "Model" });
    await expect(select.locator("optgroup[label='Cloud (API keys)'] option")).toHaveCount(3);
    await select.selectOption({ label: "example-large · OpenAI" });
    await expect(page.getByTestId("cloud-badge")).toHaveText("Sent to OpenAI");
  });

  test("Settings panel is accessible", async ({ page }) => {
    await gotoSettings(page);
    await page.getByTestId("cloud-models").getByRole("combobox", { name: "Add a provider" }).selectOption("anthropic");
    await settle(page);
    const r = await new AxeBuilder({ page }).withTags(TAGS).include("[data-testid=cloud-models]").analyze();
    expect(r.violations).toEqual([]);
  });
});
