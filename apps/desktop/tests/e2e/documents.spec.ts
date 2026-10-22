import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const fixture = (name: string): string => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));

async function open(page: Page): Promise<void> {
  await page.goto("/?mock=fast-chat#/documents");
  await expect(page.locator("main h1")).toHaveText("Documents");
}

/** Opens one of the screen's tabs (Search, Files, Convert, Quick tools). */
async function tab(page: Page, name: string): Promise<void> {
  await page.getByRole("tab", { name }).click();
  await expect(page.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
}

test.describe("documents", () => {
  test("search returns the passage that answers, with its file, as text", async ({ page }) => {
    await open(page);
    await page.getByRole("searchbox", { name: "Search documents" }).fill("annual leave days");
    const hit = page.locator(".doc-hit").first();
    await expect(hit).toContainText("Employee handbook 2026.txt");
    await expect(hit.locator("mark").first()).toHaveText(/annual/i);
  });

  test("a dropped text file indexes with progress, becomes searchable and can be removed", async ({ page }) => {
    await open(page);
    await tab(page, "Files");
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("The wifi password for the office is printed on the router. Backups run every night at 2am.") });
    await expect(page.locator(".doc-row", { hasText: "notes.txt" }).locator(".chip")).toHaveText("Ready", { timeout: 10_000 });
    // The result says what was read.
    await expect(page.locator(".doc-result", { hasText: "notes.txt" })).toContainText("characters");
    await tab(page, "Search");
    await page.getByRole("searchbox", { name: "Search documents" }).fill("when do backups run");
    await expect(page.locator(".doc-hit").first()).toContainText("notes.txt");
    await tab(page, "Files");
    await page.getByRole("button", { name: "Remove notes.txt" }).click();
    await expect(page.locator(".doc-row", { hasText: "notes.txt" })).toHaveCount(0);
  });

  test("unsupported files are refused with a reason; markup in a file is read as text", async ({ page }) => {
    await open(page);
    await tab(page, "Files");
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "tool.exe", mimeType: "application/octet-stream", buffer: Buffer.from("MZ") });
    await expect(page.locator(".doc-result", { hasText: "tool.exe" })).toContainText("Supported");
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "x.html", mimeType: "text/html", buffer: Buffer.from("<img src=x onerror=alert(1)> secret plan inside.") });
    await expect(page.locator(".doc-row", { hasText: "x.html" }).locator(".chip")).toHaveText("Ready", { timeout: 10_000 });
    await tab(page, "Search");
    await page.getByRole("searchbox", { name: "Search documents" }).fill("secret plan");
    await expect(page.locator(".doc-hit img")).toHaveCount(0);
    // The markup is read as text, so it can never run.
    await expect(page.locator(".doc-hit").first()).toContainText("secret plan inside");
    await expect(page.locator(".doc-hit").first()).not.toContainText("<img");
  });

  test("a PDF is converted to clean text, and the result says how much was read", async ({ page }) => {
    await open(page);
    await tab(page, "Files");
    await page.locator('.doc-drop input[type="file"]').setInputFiles(fixture("sample.pdf"));
    const result = page.locator(".doc-result", { hasText: "sample.pdf" });
    await expect(result).toBeVisible({ timeout: 10_000 });
    await expect(result).toContainText("characters");
    await expect(result).toContainText("2 pages");
    await tab(page, "Search");
    await page.getByRole("searchbox", { name: "Search documents" }).fill("annual leave");
    await expect(page.locator(".doc-hit").first()).toContainText("sample.pdf");
  });

  test("files are grouped in collections, with a folder and tags", async ({ page }) => {
    await open(page);
    await tab(page, "Files");
    await page.getByRole("textbox", { name: "New collection" }).fill("Policy");
    await page.getByRole("button", { name: "Add collection" }).click();
    await expect(page.locator(".doc-collection", { hasText: "Policy" })).toBeVisible();
    await page.locator('.doc-drop input[type="file"]').setInputFiles({ name: "policy.txt", mimeType: "text/plain", buffer: Buffer.from("Expenses must be filed within 30 days.") });
    await expect(page.locator(".doc-row", { hasText: "policy.txt" })).toBeVisible({ timeout: 10_000 });
    await page.locator('.doc-row:has-text("policy.txt") summary').click();
    await page.getByRole("textbox", { name: "Folder" }).fill("Work");
    await page.getByRole("textbox", { name: "Tags, separated by commas" }).fill("money, policy");
    await page.getByRole("checkbox", { name: "Policy" }).check();
    await expect(page.locator(".doc-row", { hasText: "policy.txt" })).toContainText("Policy");
  });

  test("the Convert tab turns a file into Markdown, and offers to save it", async ({ page }) => {
    await open(page);
    await tab(page, "Convert");
    await page.locator('#doc-panel-convert input[type="file"]').setInputFiles({ name: "pricing.md", mimeType: "text/markdown", buffer: Buffer.from("# Pricing\n\nThe starter plan is 19 euros a month.\n") });
    await expect(page.locator("#doc-convert-result")).toBeVisible({ timeout: 10_000 });
    await expect(page.locator(".doc-preview")).toContainText("The starter plan is 19 euros a month.");
    await expect(page.getByRole("button", { name: "Save" })).toBeVisible();
    await page.getByLabel("Save as").selectOption("text");
    await expect(page.locator(".doc-preview")).not.toContainText("# Pricing");
  });

  test("the Quick tools tab runs Summarise, Extract and Translate", async ({ page }) => {
    await open(page);
    await tab(page, "Quick tools");

    const summarise = page.locator('[data-tool="summarise"]');
    await summarise.getByRole("textbox", { name: "Text to summarise" }).fill("Backups run every night at 2am and the router is in the hall.");
    await summarise.getByRole("button", { name: "Summarise" }).click();
    await expect(summarise.locator(".tool-card__result")).toContainText("Summary", { timeout: 15_000 });

    const extract = page.locator('[data-tool="extract"]');
    await extract.getByRole("textbox", { name: "Text to read" }).fill("Ada is the engineer. Grace is the analyst.");
    await extract.getByRole("textbox", { name: "Fields to extract" }).fill("name, role");
    await extract.getByRole("button", { name: "Extract" }).click();
    await expect(extract.locator(".tool-card__result")).toContainText("name", { timeout: 15_000 });
    await expect(extract.locator(".tool-card__result")).toContainText("role");

    const translate = page.locator('[data-tool="translate"]');
    await translate.getByRole("textbox", { name: "Text to translate" }).fill("Good morning.");
    await translate.getByLabel("Language").selectOption("Spanish");
    await translate.getByRole("button", { name: "Translate" }).click();
    await expect(translate.locator(".tool-card__result")).toContainText("Translation", { timeout: 15_000 });
  });

  test("a tool that needs a model says so, with a way to the Models screen", async ({ page }) => {
    await open(page);
    await tab(page, "Quick tools");
    const describe = page.locator('[data-tool="describe"]');
    await expect(describe.getByRole("button", { name: "Open Models" })).toBeVisible();
    await expect(describe).toContainText("No vision model is loaded");
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with results (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await open(page);
      await page.getByRole("searchbox", { name: "Search documents" }).fill("restart server");
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });

    test(`no axe violations on the other tabs (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await open(page);
      await tab(page, "Files");
      await scrollThrough(page);
      await settle(page);
      const files = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(files.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
      await tab(page, "Quick tools");
      await scrollThrough(page);
      await settle(page);
      const tools = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(tools.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
