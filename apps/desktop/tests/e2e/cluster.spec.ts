import AxeBuilder from "@axe-core/playwright";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("cluster", () => {
  test("sharing shows a pairing code and the address; stopping hides it", async ({ page }) => {
    await page.goto("/#/cluster");
    await page.getByRole("button", { name: "Start sharing" }).click();
    await expect(page.getByTestId("pairing-code")).toHaveText("482 913");
    await expect(page.getByTestId("cluster-sharing")).toContainText("192.168.1.10:50080");
    await page.getByRole("button", { name: "Stop sharing" }).click();
    await expect(page.getByTestId("cluster-sharing")).toHaveCount(0);
  });

  test("find computers, a wrong code says why, the right code loads across computers with link figures", async ({ page }) => {
    await page.goto("/#/cluster");
    await page.getByRole("button", { name: "Look for computers" }).click();
    await expect(page.getByText("OFFICE-PC")).toBeVisible();
    await page.getByRole("button", { name: "Use" }).first().click();
    await page.getByRole("textbox", { name: "Computer address" }).fill("10.0.0.5:50080");
    await page.getByRole("button", { name: "Add", exact: true }).click();
    await expect(page.locator(".cluster__chosen li")).toHaveCount(2);
    await page.getByRole("button", { name: "Remove 10.0.0.5:50080" }).click();
    const load = page.getByRole("button", { name: /Load across 2 computers/ });
    await load.click();
    await expect(page.getByRole("alert")).toContainText("6-digit pairing code");
    await page.getByPlaceholder("000 000").fill("111111");
    await load.click();
    await expect(page.getByRole("alert")).toContainText("wrong pairing code");
    await page.getByPlaceholder("000 000").fill("482 913");
    await load.click();
    const links = page.getByTestId("cluster-links");
    await expect(links).toBeVisible();
    await expect(links.locator("tbody tr")).toHaveCount(2);
    await expect(links).toContainText("MB/s");
  });

  test("is accessible", async ({ page }) => {
    await page.goto("/#/cluster");
    await page.getByRole("button", { name: "Look for computers" }).click();
    await page.getByRole("button", { name: "Use" }).first().click();
    await settle(page);
    const r = await new AxeBuilder({ page }).withTags(TAGS).include(".cluster").analyze();
    expect(r.violations).toEqual([]);
  });
});
