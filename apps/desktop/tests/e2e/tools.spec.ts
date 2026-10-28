import AxeBuilder from "@axe-core/playwright";
import { expect, openScreen, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("tools and permissions", () => {
  test("a command asks first, previews the exact command, and Allow once logs it", async ({ page }) => {
    await openScreen(page, "tools");
    await page.getByRole("button", { name: "Try a sample Run command call" }).click();
    const dialog = page.getByRole("alertdialog");
    await expect(dialog).toContainText("Allow Run command?");
    await expect(dialog.locator("pre")).toContainText("cargo test --workspace");
    await page.keyboard.press("y");
    await expect(dialog).toHaveCount(0);
    await expect(page.locator(".tools-log li").first()).toContainText("Allowed once");
  });

  test("an edit previews its diff; Always sets the rule so the next call runs without asking", async ({ page }) => {
    await openScreen(page, "tools");
    await page.getByRole("button", { name: "Try a sample Edit file call" }).click();
    await expect(page.locator(".diff-line[data-t='add']")).toContainText("fs::read_to_string(path)?;");
    await page.getByRole("button", { name: /Always/ }).click();
    await expect(page.getByRole("combobox", { name: "Rule for Edit file" })).toHaveValue("always");
    await page.getByRole("button", { name: "Try a sample Edit file call" }).click();
    await expect(page.getByRole("alertdialog")).toHaveCount(0);
    await expect(page.locator(".tools-log li").first()).toContainText("Ran (rule)");
  });

  test("Escape denies; a Deny rule blocks without asking; rules persist", async ({ page }) => {
    await openScreen(page, "tools");
    await page.getByRole("button", { name: "Try a sample Fetch web page call" }).click();
    await page.keyboard.press("Escape");
    await expect(page.locator(".tools-log li").first()).toContainText("Denied");
    await page.getByRole("combobox", { name: "Rule for Git commit" }).selectOption("deny");
    await page.getByRole("button", { name: "Try a sample Git commit call" }).click();
    await expect(page.getByRole("alertdialog")).toHaveCount(0);
    await expect(page.locator(".tools-log li").first()).toContainText("Blocked (rule)");
    await page.reload();
    await expect(page.getByRole("combobox", { name: "Rule for Git commit" })).toHaveValue("deny");
  });

  test("with prompts switched off in Settings, tools run without asking", async ({ page }) => {
    await openScreen(page, "settings");
    await page.locator('[data-setting="setting-permissionPrompts"]').click();
    await openScreen(page, "tools");
    await expect(page.locator(".notice")).toContainText("Permission prompts are off");
    await page.getByRole("button", { name: "Try a sample Run command call" }).click();
    await expect(page.getByRole("alertdialog")).toHaveCount(0);
    await expect(page.locator(".tools-log li").first()).toContainText("Ran (rule)");
  });

  test("activity is searchable", async ({ page }) => {
    await openScreen(page, "tools");
    await page.getByRole("button", { name: "Try a sample Run command call" }).click();
    await page.keyboard.press("n");
    await page.getByRole("button", { name: "Try a sample Fetch web page call" }).click();
    await page.keyboard.press("y");
    await page.getByRole("searchbox", { name: "Search tool activity" }).fill("huggingface");
    await expect(page.locator(".tools-log li")).toHaveCount(1);
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with the dialog open (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openScreen(page, "tools");
      await page.getByRole("button", { name: "Try a sample Edit file call" }).click();
      await expect(page.getByRole("alertdialog")).toBeVisible();
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});

test.describe("MCP servers", () => {
  test("the Add form saves a web address server and shows it running", async ({ page }) => {
    await openScreen(page, "tools");
    const mcp = page.getByTestId("mcp-servers");
    await mcp.getByText("Add a custom server").click();
    await mcp.getByRole("radio", { name: "Web address" }).click();
    await expect(mcp.getByLabel("Server command")).toHaveCount(0);
    await mcp.getByLabel("Server name").fill("webdocs");
    await mcp.getByLabel("Server address").fill("https://example.com/mcp");
    await mcp.getByLabel("Headers, one KEY=VALUE per line").fill("Authorization=Bearer token123");
    await mcp.getByRole("button", { name: "Add server" }).click();
    const item = mcp.getByTestId("mcp-server-webdocs");
    await expect(item.locator(".mcp__command")).toHaveText("https://example.com/mcp");
    await expect(item.locator(".pill")).toHaveText("running");
  });

  test("a web address without http(s) is refused", async ({ page }) => {
    await openScreen(page, "tools");
    const mcp = page.getByTestId("mcp-servers");
    await mcp.getByText("Add a custom server").click();
    await mcp.getByRole("radio", { name: "Web address" }).click();
    await mcp.getByLabel("Server name").fill("badweb");
    await mcp.getByLabel("Server address").fill("example.com/mcp");
    await mcp.getByRole("button", { name: "Add server" }).click();
    await expect(page.getByText("A web address must start with http:// or https://")).toBeVisible();
    await expect(mcp.getByTestId("mcp-server-badweb")).toHaveCount(0);
  });
});
