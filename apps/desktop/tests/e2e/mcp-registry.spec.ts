import { MCP_CATALOGUE } from "../../ui/src/lib/mcp-catalogue";
import { expect, test } from "./fixtures";

test.describe("MCP library: the official registry", () => {
  test("switching to the registry searches on Enter; Add fills the form with the command and keys", async ({ page }) => {
    await page.goto("/#/tools");
    const browse = page.getByTestId("mcp-catalogue");
    await expect(browse.getByTestId("mcp-catalogue-count")).toContainText("servers");
    await browse.getByRole("radio", { name: "Official MCP registry" }).check({ force: true });
    await expect(browse.getByTestId("mcp-catalogue-count")).toContainText("not checked by Cerebro");
    await browse.getByRole("searchbox", { name: "Search servers" }).fill("weather");
    await browse.getByRole("searchbox", { name: "Search servers" }).press("Enter");
    await expect(browse.getByTestId("mcp-catalogue-count")).toHaveText("1 servers from the registry");
    await browse.getByRole("button", { name: "Add Weather (example)" }).click();
    await expect(page.getByTestId("mcp-catalogue-keys")).toContainText("Weather (example)");
    await expect(page.getByTestId("mcp-catalogue-keys")).toContainText("Your API key");
    await browse.getByRole("radio", { name: /Checked list/ }).check({ force: true });
    await browse.getByRole("searchbox", { name: "Search servers" }).fill("");
    await expect(browse.getByTestId("mcp-catalogue-count")).toHaveText(`${MCP_CATALOGUE.length} of ${MCP_CATALOGUE.length} servers`);
  });
});
