import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, scrollThrough, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", "base64");

async function open(page: Page): Promise<void> {
  await page.goto("/?mock=fast-images#/images");
  await expect(page.locator("main h1")).toHaveText("Image Studio");
}

test.describe("image studio", () => {
  test("text to image shows a live denoising preview, then saves to the gallery", async ({ page }) => {
    await open(page);
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.locator(".img-error")).toHaveText("Describe the image.");
    await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("A lighthouse at dusk");
    await page.locator("label.img-chip", { hasText: "16:9" }).click();
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.getByRole("progressbar", { name: "Generating" })).toBeVisible();
    await expect(page.locator(".img-tile")).toHaveCount(1, { timeout: 15_000 });
    await expect(page.locator(".img-result figcaption")).toContainText("1344×768");
    await page.reload();
    await expect(page.locator(".img-tile")).toHaveCount(1);
  });

  test("favourite, filter, reuse the prompt and delete", async ({ page }) => {
    await open(page);
    await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("A red fox in snow");
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.locator(".img-tile")).toHaveCount(1, { timeout: 15_000 });
    await page.getByRole("button", { name: "Favourite A red fox in snow" }).click();
    await page.locator("label.segmented__option", { hasText: "Favourites" }).click();
    await expect(page.locator(".img-tile")).toHaveCount(1);
    await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("");
    await page.getByRole("button", { name: "Reuse prompt: A red fox in snow" }).click();
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toHaveValue("A red fox in snow");
    await expect(page.getByRole("link", { name: "Export A red fox in snow" })).toHaveAttribute("href", /^data:image\/svg\+xml/);
    await page.getByRole("button", { name: "Delete A red fox in snow" }).click();
    await expect(page.locator(".img-tile")).toHaveCount(0);
  });

  test("inpainting needs a source and a painted mask", async ({ page }) => {
    await open(page);
    await page.getByRole("tab", { name: "Inpaint" }).click();
    await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("Replace the sky");
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.locator(".img-error").last()).toHaveText("Add a source image first.");
    await page.locator('.img-drop input[type="file"]').setInputFiles({ name: "photo.png", mimeType: "image/png", buffer: PNG });
    await expect(page.getByRole("img", { name: "Mask: paint over the area to change" })).toBeVisible();
    await expect(page.locator(".img-error").last()).toHaveText("Paint over the area to change first.");
    await page.locator(".img-mask canvas").scrollIntoViewIfNeeded();
    const box = (await page.locator(".img-mask canvas").boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2 + 10);
    await page.mouse.up();
    await expect(page.locator(".img-error")).toHaveCount(0);
    await page.getByRole("button", { name: "Generate" }).click();
    await expect(page.locator(".img-tile")).toHaveCount(1, { timeout: 15_000 });
  });

  test("advanced settings validate their values", async ({ page }) => {
    await open(page);
    const details = page.locator(".img-advanced-details");
    if (!(await details.evaluate((el) => (el as HTMLDetailsElement).open))) {
      await page.locator(".img-advanced-details > summary").click();
    }
    await expect(details).toHaveAttribute("open", "");
    await page.getByRole("spinbutton", { name: "Steps" }).fill("500");
    await page.getByRole("spinbutton", { name: "Steps" }).blur();
    await expect(page.locator(".img-advanced .img-error")).toHaveText("Steps must be 1 to 150.");
  });

  for (const scheme of ["dark", "light"] as const) {
    test(`no axe violations with an image made (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await open(page);
      await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("Mountains");
      await page.getByRole("button", { name: "Generate" }).click();
      await expect(page.locator(".img-tile")).toHaveCount(1, { timeout: 15_000 });
      await scrollThrough(page);
      await settle(page);
      const r = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
    });
  }
});
