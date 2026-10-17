import AxeBuilder from "@axe-core/playwright";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("local API", () => {
  test("off by default; on shows the address, the key stays hidden until shown, and every protection is a switch", async ({ page }) => {
    await page.goto("/#/settings");
    const card = page.getByTestId("local-api");
    await card.scrollIntoViewIfNeeded();
    await expect(page.getByTestId("local-api-status")).toHaveText("Off");
    await card.getByRole("switch", { name: "Local API" }).click({ force: true });
    await expect(page.getByTestId("local-api-status")).toContainText("On at http://127.0.0.1:47812/v1");
    await expect(page.getByTestId("local-api-key-value")).not.toContainText("mock0123456789abcdef0123456789abcdef");
    await expect(page.getByTestId("local-api-snippet")).toHaveCount(0);
    await card.getByRole("button", { name: "Show" }).click();
    await expect(page.getByTestId("local-api-snippet")).toContainText('base_url="http://127.0.0.1:47812/v1"');
    await card.getByRole("radio", { name: "Anthropic SDK" }).click();
    await expect(page.getByTestId("local-api-snippet")).toContainText('base_url="http://127.0.0.1:47812"');

    await card.getByRole("switch", { name: "Allow the local network" }).click({ force: true });
    await expect(card).toContainText("http://192.168.1.20:47812/v1");
    await card.getByRole("switch", { name: "Require the key" }).click({ force: true });
    await expect(card.getByRole("alert")).toHaveText("Anyone on your network can use this computer's model without a key.");
    await expect(page.getByTestId("local-api-key-value")).toHaveCount(0);
    await expect(page.getByTestId("local-api-snippet")).toContainText('api_key="not-needed"');
  });

  test("has no accessibility violations when on", async ({ page }) => {
    await page.goto("/#/settings");
    const card = page.getByTestId("local-api");
    await card.scrollIntoViewIfNeeded();
    await card.getByRole("switch", { name: "Local API" }).click({ force: true });
    await expect(page.getByTestId("local-api-status")).toContainText("On at");
    await scrollThrough(page);
    await settle(page);
    expect((await new AxeBuilder({ page }).include('[data-testid="local-api"]').withTags(TAGS).analyze()).violations).toEqual([]);
  });
});
