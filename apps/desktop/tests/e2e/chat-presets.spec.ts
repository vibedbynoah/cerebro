import { expect, test } from "./fixtures";

test("a model's saved settings come back when it is picked again", async ({ page }) => {
  await page.goto("/#/chat");
  const select = page.getByRole("combobox", { name: "Model" });
  await expect(select.locator("option").nth(1)).toBeAttached();
  const first = await select.locator("option").nth(0).getAttribute("value");
  const second = await select.locator("option").nth(1).getAttribute("value");
  await page.getByRole("button", { name: /^Live panel/ }).click();
  const dock = page.getByTestId("live-dock");
  await dock.getByRole("slider", { name: /Speed or accuracy/ }).fill("1");
  await expect(page.getByTestId("speed-mode")).toHaveText("Fastest");
  await dock.getByRole("button", { name: "Save as this model's settings" }).click();
  await expect(dock.getByRole("button", { name: "Forget them" })).toBeVisible();
  // Another model, other settings.
  await select.selectOption(second!);
  await dock.getByRole("slider", { name: /Speed or accuracy/ }).fill("4");
  await expect(page.getByTestId("speed-mode")).toHaveText("Accurate");
  // Back to the first: its settings return.
  await select.selectOption(first!);
  await expect(page.getByTestId("speed-mode")).toHaveText("Fastest");
  await dock.getByRole("button", { name: "Forget them" }).click();
  await expect(dock.getByRole("button", { name: "Forget them" })).toHaveCount(0);
});
