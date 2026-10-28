import { expect, openScreen, test } from "./fixtures";

test.describe("MCP server catalogue", () => {
  test("searching the catalogue and one click on Add starts the server", async ({ page }) => {
    await openScreen(page, "tools");
    const panel = page.getByTestId("mcp-catalogue");
    await panel.scrollIntoViewIfNeeded();
    await expect(panel.getByTestId("mcp-catalogue-filesystem")).toBeVisible();

    await page.getByLabel("Search servers").fill("duckduckgo");
    await expect(panel.getByTestId("mcp-catalogue-duckduckgo")).toBeVisible();
    await expect(panel.getByTestId("mcp-catalogue-filesystem")).toHaveCount(0);
    await expect(panel.getByTestId("mcp-catalogue-count")).toHaveText(/^1 of \d+ servers$/);

    await panel.getByRole("button", { name: "Add DuckDuckGo" }).click();
    const server = page.getByTestId("mcp-server-duckduckgo");
    await expect(server).toBeVisible();
    await expect(server.locator("code")).toHaveText("uvx duckduckgo-mcp-server");
    await expect(server).toContainText("running");
    await expect(panel.getByRole("button", { name: "Add DuckDuckGo" })).toHaveText("Added");
  });

  test("a folder argument gets the workspace folder or Documents, with no form", async ({ page }) => {
    await openScreen(page, "tools");
    const panel = page.getByTestId("mcp-catalogue");
    await panel.scrollIntoViewIfNeeded();
    await page.getByLabel("Search servers").fill("filesystem");
    await panel.getByRole("button", { name: "Add Filesystem" }).click();
    const server = page.getByTestId("mcp-server-filesystem");
    await expect(server).toBeVisible();
    await expect(server.locator("code")).not.toContainText("<folder");
  });

  test("category chips filter the grid, and a server that needs a key asks for it on its card", async ({ page }) => {
    await openScreen(page, "tools");
    const panel = page.getByTestId("mcp-catalogue");
    await panel.scrollIntoViewIfNeeded();

    await panel.getByRole("button", { name: "Databases", exact: true }).click();
    await expect(panel.getByTestId("mcp-catalogue-sqlite")).toBeVisible();
    await expect(panel.getByTestId("mcp-catalogue-memory")).toHaveCount(0);

    await panel.getByRole("button", { name: "All", exact: true }).click();
    await page.getByLabel("Search servers").fill("sentry");
    await panel.getByRole("button", { name: "Add Sentry" }).click();

    const keys = panel.getByTestId("mcp-card-keys-sentry");
    const token = keys.getByLabel("Access token");
    await expect(token).toHaveAttribute("type", "password");
    await expect(token).toBeFocused();
    await expect(page.getByTestId("mcp-server-sentry")).toHaveCount(0);

    await token.fill("sntrys_test_token");
    await panel.getByRole("button", { name: "Add Sentry" }).click();
    await expect(page.getByTestId("mcp-server-sentry")).toBeVisible();
  });
});
