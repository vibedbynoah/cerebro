import AxeBuilder from "@axe-core/playwright";
import { readFile } from "node:fs/promises";
import type { Page } from "@playwright/test";
import { expect, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

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

async function seedChats(page: Page, chats: unknown[]): Promise<void> {
  await page.addInitScript((value) => window.localStorage.setItem("cerebro-chats", JSON.stringify(value)), chats);
}

const user = (id: string, text: string) => ({ id, role: "user", text });
const reply = (id: string, text: string) => ({ id, role: "assistant", text });

test.describe("chat branches", () => {
  test("editing an earlier message keeps both wordings and the turn after it", async ({ page }) => {
    await openChat(page);
    await send(page, "First wording");
    await waitDone(page);
    await send(page, "Follow up question");
    await waitDone(page);
    await expect(page.locator(".msg--user")).toHaveCount(2);

    await page.locator(".msg--user").first().getByRole("button", { name: "Edit message" }).click();
    await page.getByRole("textbox", { name: "Edit message" }).fill("Second wording");
    await page.getByRole("button", { name: "Save and branch" }).click();
    await expect(page.locator(".msg--bot")).toHaveCount(1);
    await waitDone(page);

    const words = page.locator(".msg--user .msg__text");
    await expect(words).toHaveText("Second wording");
    const branches = page.locator(".msg--user .msg__branches");
    await expect(branches.locator(".msg__branch-count")).toHaveText("2/2");

    await branches.getByRole("button", { name: "Previous branch" }).click();
    await expect(words).toHaveText(["First wording", "Follow up question"]);
    await expect(branches.locator(".msg__branch-count")).toHaveText("1/2");
    await expect(page.locator(".msg--bot")).toHaveCount(2);

    await branches.getByRole("button", { name: "Next branch" }).click();
    await expect(words).toHaveText("Second wording");
    await expect(page.locator(".msg--bot")).toHaveCount(1);
  });

  test("regenerating any reply adds a branch instead of replacing it", async ({ page }) => {
    await openChat(page);
    await send(page, "Is a 4-bit quant fine?");
    await waitDone(page);
    await send(page, "And what about a 2-bit quant?");
    await waitDone(page);
    await expect(page.locator(".msg--bot")).toHaveCount(2);

    await page.locator(".msg--bot").first().getByRole("button", { name: "Regenerate" }).click();
    await expect(page.locator(".msg--bot")).toHaveCount(1);
    await waitDone(page);
    await expect(page.locator(".msg--user .msg__text")).toHaveText("Is a 4-bit quant fine?");

    const branch = page.locator(".msg--bot").first().locator(".msg__branches");
    await expect(branch.locator(".msg__branch-count")).toHaveText("2/2");
    await branch.getByRole("button", { name: "Previous branch" }).click();
    await expect(page.locator(".msg--bot").first().locator(".msg__branch-count")).toHaveText("1/2");
    await expect(page.locator(".msg--bot")).toHaveCount(2);
    await expect(page.locator(".msg--user .msg__text").last()).toHaveText("And what about a 2-bit quant?");
  });
});

test.describe("chat search", () => {
  const chats = [
    {
      id: "beta",
      title: "Beta plan",
      updatedAt: 100,
      messages: [],
      tree: {
        nodes: [
          { id: "u1", parentId: null, message: user("u1", "Plan the beta launch") },
          { id: "a1", parentId: "u1", message: reply("a1", "The beta answer mentions May.") },
          { id: "u2", parentId: null, message: user("u2", "Plan the gamma launch") },
          { id: "a2", parentId: "u2", message: reply("a2", "The launch checklist mentions Thursday.") },
        ],
        selected: { "": "u1", u1: "a1" },
      },
    },
    { id: "garden", title: "Garden notes", updatedAt: 200, messages: [user("g1", "Buy seeds on Thursday")] },
  ];

  test("the chat list finds words in another branch and jumps to the message", async ({ page }) => {
    await seedChats(page, chats);
    await page.goto("/#/chat");
    await expect(page.locator(".msg--user .msg__text")).toHaveText("Plan the beta launch");

    const search = page.getByRole("searchbox", { name: "Search chats and messages" });
    await search.fill("checklist");
    const result = page.locator(".chat-list__result", { hasText: "checklist" });
    await expect(result).toHaveCount(1);
    await expect(result.locator("mark").first()).toHaveText("checklist");

    await result.click();
    await expect(page.locator(".msg--user .msg__text")).toHaveText("Plan the gamma launch");
    await expect(page.locator("[data-message-id='a2']")).toHaveClass(/msg--found/);
    await expect(page.locator("[data-message-id='a2']")).toContainText("launch checklist mentions Thursday");
    await expect(search).toHaveValue("");
  });

  test("Ctrl+Shift+F searches every chat from another screen", async ({ page }) => {
    await seedChats(page, chats);
    await page.goto("/#/settings");
    await expect(page.locator("main h1")).toHaveText("Settings");

    await page.keyboard.press("Control+Shift+F");
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await dialog.getByRole("combobox", { name: "Search chats" }).fill("checklist");
    await expect(dialog.getByRole("option")).toHaveCount(1);
    await dialog.getByRole("option").first().click();

    await expect(page).toHaveURL(/#\/chat$/);
    await expect(page.locator("[data-message-id='a2']")).toContainText("launch checklist mentions Thursday");
    await expect(page.locator("[data-message-id='a2']")).toHaveClass(/msg--found/);
  });
});

test.describe("chat organization", () => {
  test("pins, folders, tags and the archive sort and filter the list", async ({ page }) => {
    await seedChats(page, [
      { id: "alpha", title: "Alpha work", updatedAt: 300, messages: [user("a1", "Alpha chat")] },
      { id: "beta", title: "Beta home", updatedAt: 400, messages: [user("b1", "Beta chat")] },
    ]);
    await page.goto("/#/chat");

    const items = page.locator(".chat-list__item");
    await expect(items).toHaveCount(2);
    await expect(items.first()).toContainText("Beta home");

    const alpha = page.locator(".chat-list__item", { hasText: "Alpha work" });
    await alpha.getByRole("button", { name: "Pin Alpha work" }).click();
    await expect(alpha.getByRole("button", { name: "Unpin Alpha work" })).toHaveAttribute("aria-pressed", "true");
    await expect(items.first()).toContainText("Alpha work");

    await alpha.locator("summary").click();
    await alpha.getByLabel("Folder").fill("Work");
    await alpha.getByLabel("Folder").press("Tab");
    await alpha.getByLabel("Tags, separated by commas").fill("later, urgent");
    await alpha.getByLabel("Tags, separated by commas").press("Tab");
    await alpha.locator("summary").click();

    const folderFilter = page.getByRole("combobox", { name: "Filter by folder" });
    await folderFilter.selectOption("Work");
    await expect(items).toHaveCount(1);
    await expect(items.first()).toContainText("Alpha work");
    await folderFilter.selectOption("all");

    const tagFilter = page.getByRole("combobox", { name: "Filter by tag" });
    await tagFilter.selectOption("urgent");
    await expect(items).toHaveCount(1);
    await expect(items.first()).toContainText("Alpha work");
    await tagFilter.selectOption("all");

    await alpha.locator("summary").click();
    // The row leaves the list as soon as it is archived, so a plain click is enough.
    await alpha.getByRole("checkbox", { name: "Archived" }).click();
    await expect(items).toHaveCount(1);
    await expect(items.first()).toContainText("Beta home");
    await expect(page.locator(".msg--user .msg__text")).toHaveText("Beta chat");

    await page.getByRole("checkbox", { name: "Archived" }).check();
    await expect(items).toHaveCount(1);
    await expect(items.first()).toContainText("Alpha work");
  });
});

test.describe("prompt library", () => {
  test("fills variables, creates prompt text, and imports and exports JSON", async ({ page }, info) => {
    await openChat(page);
    await page.getByRole("button", { name: "Prompt library" }).click();
    await expect(page.getByRole("heading", { name: "Prompt library" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Use prompt" })).toBeVisible();
    await page.locator(".prompt-library__use").getByRole("button", { name: "Close" }).click();
    await expect(page.getByRole("button", { name: "Use prompt" })).toHaveCount(0);

    await page.locator(".prompt-library__item", { hasText: "Explain it simply" }).click();
    await page.getByRole("button", { name: "Use prompt" }).click();
    await expect(page.getByText("Fill in the variable")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Prompt library" })).toBeVisible();
    await page.getByLabel("topic", { exact: true }).fill("caches");
    await page.getByLabel("level", { exact: true }).fill("beginner");
    await page.getByRole("button", { name: "Use prompt" }).click();
    const box = page.getByRole("textbox", { name: "Message" });
    await expect(box).toHaveValue(/^Explain caches to a beginner\./);
    await expect(page.getByRole("heading", { name: "Prompt library" })).toHaveCount(0);

    await box.fill("/prompt");
    await box.press("Enter");
    await expect(page.getByRole("heading", { name: "Prompt library" })).toBeVisible();
    await expect(box).toHaveValue("");

    await page.getByRole("button", { name: "New prompt" }).click();
    const editor = page.locator(".prompt-library__editor");
    await editor.getByLabel("Title").fill("Test prompt");
    await editor.getByLabel("Category").fill("Custom");
    await editor.getByLabel("Prompt text, with variables in double braces").fill("Say hello to {{name}}.");
    await editor.getByRole("button", { name: "Save prompt" }).click();
    await expect(page.locator(".prompt-library__item", { hasText: "Test prompt" })).toBeVisible();

    await page.getByRole("combobox", { name: "Prompt category" }).selectOption("Custom");
    await page.locator(".prompt-library__item", { hasText: "Test prompt" }).click();
    await page.getByLabel("name", { exact: true }).fill("Ada");
    await page.getByRole("button", { name: "Use prompt" }).click();
    await expect(box).toHaveValue(/^Say hello to Ada\./);

    await box.fill("/prompt");
    await box.press("Enter");
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export" }).click()]);
    const path = info.outputPath("cerebro-prompts.json");
    await download.saveAs(path);
    const exported = JSON.parse(await readFile(path, "utf8")) as { prompts: Array<{ title: string }> };
    expect(exported.prompts.map((p) => p.title)).toContain("Test prompt");

    const imported = { id: "p-imported", title: "Imported prompt", text: "Write a note about {{subject}}.", category: "Custom" };
    await page.locator(".prompt-library input[type='file']").setInputFiles({
      name: "prompts.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify({ version: 1, prompts: [imported] })),
    });
    await expect(page.locator(".prompt-library__item", { hasText: "Imported prompt" })).toBeVisible();
    await expect(page.getByText("Imported 1 prompt")).toBeVisible();
  });
});

for (const scheme of ["dark", "light"] as const) {
  test.describe(`chat features: axe (${scheme})`, () => {
    test.use({ colorScheme: scheme });

    test("branches and the prompt library have no axe violations", async ({ page }) => {
      await seedChats(page, [
        {
          id: "branch",
          title: "Two endings",
          updatedAt: 1,
          messages: [],
          tree: {
            nodes: [
              { id: "u1", parentId: null, message: user("u1", "Tell me a story") },
              { id: "a1", parentId: "u1", message: reply("a1", "Once upon a time, the first ending.") },
              { id: "u2", parentId: null, message: user("u2", "Tell me a poem") },
              { id: "a2", parentId: "u2", message: reply("a2", "Roses are red, the second ending.") },
            ],
            selected: { "": "u1", u1: "a1" },
          },
        },
      ]);
      await page.goto("/#/chat");
      await expect(page.locator(".msg__branches")).toBeVisible();

      await page.getByRole("button", { name: "Prompt library" }).click();
      await page.locator(".prompt-library__item", { hasText: "Explain it simply" }).click();
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });

    test("the global chat search has no axe violations", async ({ page }) => {
      await seedChats(page, [{ id: "garden", title: "Garden notes", updatedAt: 1, messages: [user("g1", "Buy seeds on Thursday")] }]);
      await page.goto("/#/home");
      await page.keyboard.press("Control+Shift+F");
      await page.getByRole("dialog").getByRole("combobox", { name: "Search chats" }).fill("Thursday");
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  });
}
