import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
// 1x1 transparent PNG.
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", "base64");

async function openChat(page: Page): Promise<void> {
  await page.goto("/?mock=fast-chat#/chat");
  await expect(page.locator(".chat-head__title")).toHaveText("Chat");
  await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
}

async function send(page: Page, text: string): Promise<void> {
  const box = page.getByRole("textbox", { name: "Message" });
  await box.fill(text);
  await box.press("Enter");
}

const lastReply = (page: Page) => page.locator(".msg--bot").last();
const waitDone = (page: Page) => expect(lastReply(page).getByTestId("reply-meta")).toBeVisible({ timeout: 15_000 });

test.describe("chat", () => {
  test("streams a reply with speed, first-token time and the effort used", async ({ page }) => {
    await openChat(page);
    await page.locator(".chat-empty__idea").first().click();
    await expect(page.locator(".msg--user")).toHaveCount(1);
    await waitDone(page);
    await expect(lastReply(page).getByTestId("reply-meta")).toContainText(/Medium effort · Native only · \d+\.\d tok\/s · first token \d+ ms/);
    await expect(page.locator(".chat-list__item")).toHaveCount(1);
    await expect(page.locator(".chat-list__item[data-current='true']")).toContainText("Why is my 30B model");
  });

  test("Escape stops a reply and keeps what was written", async ({ page }) => {
    await page.goto("/#/chat");
    await expect(page.locator(".chat-head__model select option").first()).toBeAttached();
    await send(page, "Tell me more about expert caches");
    await expect(lastReply(page).locator(".md p").first()).toBeVisible({ timeout: 10_000 });
    await page.getByRole("textbox", { name: "Message" }).press("Escape");
    await expect(lastReply(page).getByTestId("reply-meta")).toContainText("stopped");
  });

  test("regenerate replaces the last reply; brief replies offer Tell me more", async ({ page }) => {
    await openChat(page);
    await send(page, "Is a 4-bit quant fine?");
    await waitDone(page);
    await expect(page.getByRole("button", { name: "Tell me more" })).toBeVisible();
    await page.getByRole("button", { name: "Regenerate" }).click();
    await waitDone(page);
    await expect(page.locator(".msg--bot")).toHaveCount(1);
    await page.getByRole("button", { name: "Tell me more" }).click();
    await waitDone(page);
    await expect(lastReply(page).locator("h3")).toHaveText("In more detail");
  });

  test("Max effort with Both shows thinking and two self-check passes, and the prompt shows exactly what is sent", async ({ page }) => {
    await openChat(page);
    await page.locator("label.segmented__option", { hasText: "Max" }).click();
    await page.getByRole("combobox", { name: "Reasoning" }).selectOption("both");
    await send(page, "Check my math");
    await waitDone(page);
    await expect(lastReply(page).locator("summary", { hasText: "Thinking" })).toBeVisible();
    await expect(lastReply(page).locator("summary", { hasText: "Self-check" })).toContainText("2 passes");
    await expect(lastReply(page).getByTestId("reply-meta")).toContainText(/Max effort · Both · .* thinking \d+ tokens/);
    await page.getByRole("button", { name: "System prompt" }).click();
    await expect(page.getByTestId("prompt-preview")).toContainText("Verify everything; leave nothing unchecked.");
    await expect(page.getByTestId("native-params")).toContainText("effort high");
    // One click to send nothing but the user's own prompt.
    await page.locator(".prompt-panel .composer__check input").check();
    await expect(page.getByTestId("prompt-preview")).toHaveText("You are a helpful assistant running locally on this computer.");
  });

  test("hostile Markdown in a reply never becomes markup", async ({ page }) => {
    let dialogs = 0;
    page.on("dialog", (d) => {
      dialogs++;
      void d.dismiss();
    });
    await openChat(page);
    const evil = "<script>alert(1)</script> <img src=x onerror=alert(1)> [a](javascript:alert(1)) [b](https://example.com) <svg onload=alert(1)> **<b onclick=alert(1)>x</b>**";
    await send(page, `/echo ${evil}`);
    await waitDone(page);
    const reply = lastReply(page);
    await expect(reply.locator(".md script, .md img, .md svg, .md iframe, .md b[onclick], .md [onerror], .md [onload]")).toHaveCount(0);
    const hrefs = await reply.locator(".md a").evaluateAll((as) => as.map((a) => a.getAttribute("href")));
    expect(hrefs).toEqual(["https://example.com"]);
    await expect(reply).toContainText("<script>alert(1)</script>");
    expect(dialogs).toBe(0);
  });

  test("/imagine renders an image card", async ({ page }) => {
    await openChat(page);
    await send(page, "/imagine a lighthouse at dusk");
    await waitDone(page);
    await expect(lastReply(page).locator(".img-card svg")).toHaveAttribute("aria-label", "Generated image: a lighthouse at dusk");
  });

  test("images attach with a preview and go with the message", async ({ page }) => {
    await openChat(page);
    await page.locator('.composer input[type="file"]').setInputFiles({ name: "chart.png", mimeType: "image/png", buffer: PNG });
    await expect(page.locator(".composer__atts .att img")).toHaveAttribute("src", /^data:image\/png;base64,/);
    await send(page, "What is in this picture?");
    await expect(page.locator(".msg--user .att")).toContainText("chart.png");
    await waitDone(page);
  });

  test("push-to-talk fills the composer with the transcript", async ({ page }) => {
    await openChat(page);
    const mic = page.getByRole("button", { name: "Hold to talk" });
    await mic.hover();
    await page.mouse.down();
    await expect(page.locator(".mic[aria-pressed='true'] .wave__bar")).toHaveCount(16);
    await page.waitForTimeout(400);
    await page.mouse.up();
    await expect(page.getByRole("textbox", { name: "Message" })).toHaveValue(/graphics memory/);
  });

  test("memory: remembered facts are listed, editable, and can be switched off per chat", async ({ page }) => {
    await openChat(page);
    await send(page, "Remember that I use an RTX 3070 Ti laptop.");
    await waitDone(page);
    await page.getByRole("button", { name: "Memory" }).click();
    await expect(page.locator(".memory__text").last()).toHaveValue("I use an RTX 3070 Ti laptop.");
    await page.getByRole("button", { name: /^Forget: I use an RTX/ }).click();
    await expect(page.locator(".memory__text")).toHaveCount(3);
    await page.getByRole("textbox", { name: "New memory" }).fill("Likes dark mode.");
    await page.getByRole("button", { name: "Add", exact: true }).click();
    await expect(page.locator(".memory__text").last()).toHaveValue("Likes dark mode.");
    await page.locator('[data-setting="memory-chat"]').click();
    await page.getByRole("button", { name: "System prompt" }).click();
    await expect(page.getByTestId("prompt-preview")).not.toContainText("Likes dark mode.");
  });

  test("conversations persist across a reload and can be deleted", async ({ page }) => {
    await openChat(page);
    await send(page, "First question");
    await waitDone(page);
    await page.waitForTimeout(600);
    await page.reload();
    await expect(page.locator(".chat-list__item")).toHaveCount(1);
    await expect(page.locator(".msg--bot")).toHaveCount(1);
    await page.getByRole("button", { name: "Delete First question" }).click();
    await expect(page.locator(".chat-list__item")).toHaveCount(0);
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with a finished reply (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openChat(page);
      await send(page, "Write a Python function to count lines");
      await waitDone(page);
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
