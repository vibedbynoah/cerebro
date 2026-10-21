import AxeBuilder from "@axe-core/playwright";
import { expect, openScreen, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

test.describe("Cerebro as an MCP server", () => {
  test("off by default; on shows the address, a masked token and ready-to-paste configs", async ({ page }) => {
    await openScreen(page, "settings");
    const section = page.getByTestId("mcp-server-access");
    await section.scrollIntoViewIfNeeded();
    await expect(page.getByTestId("mcp-server-status")).toHaveText(/Off/);
    await page.getByRole("switch", { name: "Cerebro MCP server" }).click({ force: true });
    await expect(page.getByTestId("mcp-server-status")).toContainText("http://127.0.0.1:47811/mcp");
    await expect(page.getByTestId("mcp-server-token")).toContainText("•");
    await expect(page.getByTestId("mcp-server-snippet")).toHaveCount(0);
    await section.getByRole("button", { name: "Show" }).click();
    await expect(page.getByTestId("mcp-server-snippet")).toContainText('"type": "http"');
    await expect(page.getByTestId("mcp-server-snippet")).toContainText('"Authorization": "Bearer ');
    await section.getByRole("radio", { name: "Other" }).click();
    await expect(page.getByTestId("mcp-server-snippet")).toContainText("Transport: Streamable HTTP (MCP 2025-06-18)");
    const before = await page.getByTestId("mcp-server-token").textContent();
    await section.getByRole("button", { name: "New token" }).click();
    await section.getByRole("button", { name: "Click again to replace" }).click();
    await expect(page.getByTestId("mcp-server-token")).not.toHaveText(before ?? "");
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await openScreen(page, "settings");
      await page.getByRole("switch", { name: "Cerebro MCP server" }).click({ force: true });
      await scrollThrough(page);
      await page.getByTestId("mcp-server-access").scrollIntoViewIfNeeded();
      await settle(page);
      const r = await new AxeBuilder({ page }).include("[data-testid='mcp-server-access']").withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
    });
  }
});
