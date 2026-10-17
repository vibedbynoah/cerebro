import { expect, openScreen, test } from "./fixtures";

test.describe("MCP server catalogue", () => {
  test("searching the catalogue and adding a server puts it in the server list", async ({ page }) => {
    await openScreen(page, "tools");
    const panel = page.getByTestId("mcp-catalogue");
    await panel.scrollIntoViewIfNeeded();
    await expect(panel.getByTestId("mcp-catalogue-filesystem")).toBeVisible();

    await page.getByLabel("Search servers").fill("duckduckgo");
    await expect(panel.getByTestId("mcp-catalogue-duckduckgo")).toBeVisible();
    await expect(panel.getByTestId("mcp-catalogue-filesystem")).toHaveCount(0);
    await expect(panel.getByTestId("mcp-catalogue-count")).toHaveText(/^1 of \d+ servers$/);

    await panel.getByRole("button", { name: "Add DuckDuckGo" }).click();
    await expect(page.getByLabel("Server name")).toHaveValue("duckduckgo");
    await expect(page.getByLabel("Server command")).toHaveValue("uvx");
    await expect(page.getByLabel("Server arguments, quoted parts stay together")).toHaveValue("duckduckgo-mcp-server");
    await expect(page.getByTestId("mcp-catalogue-keys")).toContainText("needs no keys");

    await page.getByRole("button", { name: "Add server" }).click();
    const server = page.getByTestId("mcp-server-duckduckgo");
    await expect(server).toBeVisible();
    await expect(server.locator("code")).toHaveText("uvx duckduckgo-mcp-server");
    await expect(server).toContainText("running");
  });

  test("category chips filter the grid, and a server that needs a key shows secret fields", async ({ page }) => {
    await openScreen(page, "tools");
    const panel = page.getByTestId("mcp-catalogue");
    await panel.scrollIntoViewIfNeeded();

    await panel.getByRole("button", { name: "Databases", exact: true }).click();
    await expect(panel.getByTestId("mcp-catalogue-sqlite")).toBeVisible();
    await expect(panel.getByTestId("mcp-catalogue-memory")).toHaveCount(0);

    await panel.getByRole("button", { name: "All", exact: true }).click();
    await page.getByLabel("Search servers").fill("sentry");
    await panel.getByRole("button", { name: "Add Sentry" }).click();

    const token = page.getByLabel("Access token");
    await expect(token).toHaveAttribute("type", "password");
    await expect(page.getByTestId("mcp-catalogue-keys")).toContainText("From the catalogue");

    await page.getByRole("button", { name: "Clear" }).click();
    await expect(page.getByTestId("mcp-catalogue-keys")).toHaveCount(0);
  });
});
