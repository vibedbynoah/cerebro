import AxeBuilder from "@axe-core/playwright";
import { readFile } from "node:fs/promises";
import { expect, openScreen, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("settings: replies, search, configuration file", () => {
  test("search hides sections that do not match", async ({ page }) => {
    await openScreen(page, "settings");
    await page.getByRole("searchbox", { name: "Search settings" }).fill("thinking budget");
    await expect(page.locator("#effort-title")).toBeVisible();
    await expect(page.locator("#appearance-title")).toBeHidden();
    await expect(page.locator(".settings-search .hint")).toHaveText("1 section match");
    await page.getByRole("searchbox", { name: "Search settings" }).fill("");
    await expect(page.locator("#appearance-title")).toBeVisible();
  });

  test("keep long chats ready: a switch, with what the saved-chat folder holds below it", async ({ page }) => {
    await openScreen(page, "settings");
    const sw = page.getByRole("switch", { name: "Keep long chats ready" });
    await expect(sw).toBeChecked();
    await expect(page.getByTestId("saved-chats")).toHaveText("No saved chats yet.");
    // The line sits outside the switch: reading it never toggles the setting.
    await page.getByTestId("saved-chats").click();
    await expect(sw).toBeChecked();
  });

  test("an edited effort level and the default verbosity reach Chat's system prompt", async ({ page }) => {
    await openScreen(page, "settings");
    await page.locator("label.segmented__option", { hasText: "Detailed" }).click();
    await page.getByRole("combobox", { name: "Default effort" }).selectOption("high");
    await page.getByRole("combobox", { name: "Reasoning source" }).selectOption("prompts");
    const high = page.locator(".effort-level[data-level='high']");
    await high.locator("summary").click();
    await high.getByRole("textbox", { name: "Instruction" }).fill("Double-check every number twice.");
    await high.getByRole("textbox", { name: "Instruction" }).blur();
    await expect(high.locator("summary")).toContainText("edited");
    await openScreen(page, "chat");
    await page.getByRole("button", { name: "System prompt" }).click();
    await expect(page.getByTestId("prompt-preview")).toContainText("Double-check every number twice.");
    await expect(page.getByTestId("prompt-preview")).toContainText("Give a thorough answer");
  });

  test("turning off Cerebro's instructions sends the base prompt alone", async ({ page }) => {
    await openScreen(page, "settings");
    await page.locator('[data-setting="add-instructions"]').click();
    await openScreen(page, "chat");
    await page.getByRole("button", { name: "System prompt" }).click();
    await expect(page.getByTestId("prompt-preview")).toHaveText("You are a helpful assistant running locally on this computer.");
  });

  test("exports the configuration and re-imports it, ignoring anything unknown", async ({ page }, info) => {
    await openScreen(page, "settings");
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export" }).click()]);
    const path = info.outputPath("cerebro-config.json");
    await download.saveAs(path);
    const cfg = JSON.parse(await readFile(path, "utf8"));
    expect(cfg.cerebroConfig).toBe(1);
    expect(cfg.settings.integrityCheck).toBe(true);
    cfg.settings.integrityCheck = false;
    cfg.settings.evil = "<script>";
    cfg.level = "wizard";
    await page.locator('#config-file-title').locator("..").locator("..").locator('input[type="file"]').setInputFiles({ name: "c.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(cfg)) });
    await expect(page.locator('[data-setting="setting-integrityCheck"]')).toHaveAttribute("data-state", "unchecked");
    await expect(page.getByText(/Ignored: settings\.evil, level/)).toBeVisible();
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with an effort level open (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openScreen(page, "settings");
      await page.locator(".effort-level[data-level='max'] summary").click();
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
