import type { Page } from "@playwright/test";
import { expect, openScreen, settle, test } from "./fixtures";

/**
 * The brand in the running app, in both themes: the lockup in the title bar, the mono mark on the
 * About entry, the About screen, the motif behind the empty library, and one accent (the website's
 * blue) wherever the mark is drawn. Colours are read back as rendered sRGB pixels, so a token that
 * silently resolves to another hue fails here.
 */

/** Renders a CSS colour to one canvas pixel and returns its HSL hue and saturation. */
async function hueOf(page: Page, color: string): Promise<{ hue: number; sat: number; rgb: number[] }> {
  return page.evaluate((c) => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    ctx.fillStyle = c;
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b] = Array.from(ctx.getImageData(0, 0, 1, 1).data).map((v) => v / 255) as [number, number, number];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    const l = (max + min) / 2;
    const sat = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
    let hue = 0;
    if (d !== 0) hue = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return { hue: (hue * 60 + 360) % 360, sat, rgb: [r, g, b].map((v) => Math.round(v * 255)) };
  }, color);
}

for (const scheme of ["dark", "light"] as const) {
  test.describe(`brand (${scheme})`, () => {
    test.use({ colorScheme: scheme });

    test("the title bar shows the lockup: ink experts and wordmark, one blue route", async ({ page }) => {
      await openScreen(page, "home");
      const lockup = page.locator(".titlebar [data-brand='lockup']");
      await expect(lockup).toHaveAttribute("aria-label", "Cerebro");
      const box = await lockup.boundingBox();
      expect(box?.height).toBe(24);
      const fills = await lockup.evaluate((el) => {
        const cs = (sel: string, prop: "fill" | "stroke") => getComputedStyle(el.querySelector(sel) as Element)[prop];
        const probe = document.createElement("span");
        probe.style.color = "var(--color-accent)";
        document.body.append(probe);
        const accent = getComputedStyle(probe).color;
        probe.style.color = "var(--color-text-primary)";
        const ink = getComputedStyle(probe).color;
        probe.style.color = "var(--color-accent-text)";
        const link = getComputedStyle(probe).color;
        probe.remove();
        return { route: cs(".brand__route", "fill"), idle: cs(".brand__idle", "fill"), word: cs(".brand__word", "stroke"), accent, ink, link };
      });
      expect(fills.route).toBe(fills.accent);
      expect(fills.idle).toBe(fills.ink);
      expect(fills.word).toBe(fills.ink);
      const blue = await hueOf(page, fills.route);
      // #0069e8 (light) sits near 213 deg, #38b8f6 (dark) near 200 deg; never a warm hue.
      expect(blue.hue, `route colour ${blue.rgb.join(",")}`).toBeGreaterThan(190);
      expect(blue.hue).toBeLessThan(235);
      expect(blue.sat).toBeGreaterThan(0.6);
      const link = await hueOf(page, fills.link);
      expect(link.hue).toBeGreaterThan(190);
      expect(link.hue).toBeLessThan(235);
    });

    test("About shows the lockup, the mark's anatomy and every document", async ({ page }) => {
      await openScreen(page, "about");
      await expect(page).toHaveTitle("About · Cerebro");
      await expect(page.locator("main h1")).toHaveText("About Cerebro");
      await expect(page.locator(".about-hero [data-brand='lockup']")).toBeVisible();
      await expect(page.locator(".about-hero [data-brand='motif']")).toHaveCount(1);
      await expect(page.locator(".about-hero")).toContainText("0.1.0");
      await expect(page.locator(".mark-anatomy rect")).toHaveCount(7);
      for (const label of ["Terms of use", "Privacy", "Licenses"]) await expect(page.getByRole("button", { name: label })).toBeVisible();
      await page.getByRole("button", { name: "Licenses" }).click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.getByRole("dialog")).toHaveCount(0);
      // The sidebar entry draws the mark in one ink, like every other nav icon.
      const icon = page.locator("a.nav-link[data-route='about'] [data-icon='about']");
      await expect(icon).toBeVisible();
      await expect(icon.locator("rect")).toHaveCount(7);
    });

    test("the motif's routes light once in view and the mark never sits behind data", async ({ page }) => {
      await openScreen(page, "about");
      await settle(page);
      await expect(page.locator(".about-hero .motif")).toHaveClass(/is-lit/);
      // Screens with data carry no motif.
      for (const id of ["home", "tune", "models"] as const) {
        await openScreen(page, id);
        await expect(page.locator("main .motif")).toHaveCount(0);
      }
    });
  });
}

test.describe("first-run welcome", () => {
  test.use({ onboarded: false });

  test("the mark assembles once over the motif, and the title bar carries the lockup", async ({ page }) => {
    await page.goto("/");
    await page.locator('.onb[data-step="welcome"]').waitFor();
    await expect(page.locator(".titlebar [data-brand='lockup']")).toBeVisible();
    const hero = page.locator(".brand-hero");
    await expect(hero.locator("[data-brand='mark'] rect[data-hero='expert']")).toHaveCount(6);
    await expect(hero.locator("[data-brand='motif']")).toHaveCount(1);
    await settle(page);
    // Final picture after the one-time assembly: everything opaque, nothing still moving or looping.
    const loops = await page.evaluate(() => document.getAnimations().filter((a) => a.effect?.getComputedTiming().iterations === Infinity && (a.effect as KeyframeEffect).target?.closest?.(".brand-hero")).length);
    expect(loops).toBe(0);
    expect(await hero.locator("[data-hero='picked']").evaluate((el) => getComputedStyle(el).opacity)).toBe("1");
  });
});

test.describe("reduced motion", () => {
  test.use({ onboarded: false, contextOptions: { reducedMotion: "reduce" } });

  test("the welcome mark shows finished at once", async ({ page }) => {
    await page.goto("/");
    await page.locator('.onb[data-step="welcome"]').waitFor();
    const running = await page.evaluate(() => document.getAnimations().filter((a) => (a.effect as KeyframeEffect | null)?.target?.closest?.(".brand-hero") && a.playState === "running").length);
    expect(running).toBe(0);
    expect(await page.locator(".brand-hero .motif__routes").evaluate((el) => getComputedStyle(el).opacity)).toBe("0.55");
  });
});

test("Settings links to About", async ({ page }) => {
  await openScreen(page, "settings");
  await page.getByRole("link", { name: "More about Cerebro" }).click();
  await expect(page).toHaveURL(/#\/about$/);
  await expect(page.locator("main h1")).toHaveText("About Cerebro");
});
