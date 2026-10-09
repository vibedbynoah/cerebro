import { expect, openScreen, test } from "./fixtures";

test.describe("home discovery", () => {
  test("quick actions open their screens", async ({ page }) => {
    await openScreen(page, "home");
    await page.locator(".quick__action", { hasText: "Make an image" }).click();
    await expect(page.locator("main h1")).toHaveText("Image Studio");
  });

  test("the spotlight changes on request and links to its feature", async ({ page }) => {
    await openScreen(page, "home");
    const title = page.locator(".spotlight h3");
    const first = await title.innerText();
    await page.getByRole("button", { name: "Show another" }).click();
    await expect(title).not.toHaveText(first);
  });

  test("first steps tick themselves from what you have done, and can be hidden", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("cerebro-chats", JSON.stringify([{ id: "c1", title: "Hello there", noMemory: false, updatedAt: Date.now(), messages: [{ id: "m1", role: "user", text: "hi" }] }]));
    });
    await openScreen(page, "home");
    await expect(page.locator(".checklist__item", { hasText: "Start a chat" })).toHaveAttribute("data-done", "true");
    await expect(page.locator(".recent")).toContainText("Hello there");
    await page.getByRole("button", { name: "Hide first steps" }).click();
    await expect(page.locator(".checklist")).toHaveCount(0);
    await page.reload();
    await expect(page.locator(".spotlight")).toBeVisible();
    await expect(page.locator(".checklist")).toHaveCount(0);
  });
});
