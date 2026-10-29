import { expect, test } from "./fixtures";

test("Chat starts on the loaded model, and sending with another one loads it first", async ({ page }) => {
  await page.goto("/#/chat");
  const select = page.getByRole("combobox", { name: "Model" });
  await expect(select.locator("option").nth(1)).toBeAttached();
  // The status bar names the model in memory; the picker starts on it.
  // Options read "name · quant · publisher"; the status bar names the model.
  const picked = await select.locator("option:checked").textContent();
  await expect(page.locator(".statusbar")).toContainText(picked!.split(" · ")[0]!.trim());
  const other = await select.locator("option").nth(1).getAttribute("value");
  const current = await select.inputValue();
  await select.selectOption(other === current ? (await select.locator("option").nth(0).getAttribute("value"))! : other!);
  const box = page.getByRole("textbox", { name: "Message" });
  await box.fill("Hello there");
  await box.press("Enter");
  await expect(page.getByText(/^Loading /).first()).toBeVisible();
  await expect(page.locator(".msg--bot").last()).not.toContainText("Load this model first");
  await expect(page.locator(".msg--bot").last()).toHaveAttribute("data-streaming", "false", { timeout: 15000 });
});
