import { expect, openScreen, test } from "./fixtures";

test.describe("home", () => {
  test("quick actions open their screens", async ({ page }) => {
    await openScreen(page, "home");
    await page.locator(".quick__action", { hasText: "Make an image" }).click();
    await expect(page.locator("main h1")).toHaveText("Image Studio");
  });

  test("recent chats link to Chat, and nothing seeded shows up", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("cerebro-chats", JSON.stringify([{ id: "c1", title: "Hello there", noMemory: false, updatedAt: Date.now(), messages: [{ id: "m1", role: "user", text: "hi" }] }]));
    });
    await openScreen(page, "home");
    await expect(page.locator(".spotlight")).toHaveCount(0);
    await expect(page.locator(".news")).toHaveCount(0);
    await page.locator(".recent a", { hasText: "Hello there" }).click();
    await expect(page.locator("main h1")).toHaveText("Chat");
  });
});
