import { expect, test } from "./fixtures";

test.describe("voice mode", () => {
  test("listens, hears a phrase, answers aloud and listens again; Esc ends it", async ({ page }) => {
    await page.goto("/#/chat");
    await page.getByRole("button", { name: "Voice mode" }).click();
    const v = page.getByTestId("voice-mode");
    await expect(v).toBeVisible();
    await expect(v.getByRole("status")).toHaveText(/Listening|Hearing you/);
    // The mock speaks for about a second, then is quiet: the turn ends and the reply is read aloud.
    await expect(v).toHaveAttribute("data-phase", "speaking", { timeout: 20_000 });
    await expect(v.locator(".voice__heard")).not.toBeEmpty();
    await expect(page.locator(".msg--user")).toHaveCount(1);
    await v.getByRole("button", { name: "Interrupt" }).click();
    await expect(v).toHaveAttribute("data-phase", /listening|hearing/, { timeout: 5_000 });
    await page.keyboard.press("Escape");
    await expect(v).toHaveCount(0);
  });

  test("/voice opens it", async ({ page }) => {
    await page.goto("/#/chat");
    await page.getByRole("textbox", { name: "Message" }).fill("/voice");
    await page.getByRole("textbox", { name: "Message" }).press("Enter");
    await expect(page.getByTestId("voice-mode")).toBeVisible();
    await page.getByRole("button", { name: "End voice mode" }).click();
    await expect(page.getByTestId("voice-mode")).toHaveCount(0);
  });
});
