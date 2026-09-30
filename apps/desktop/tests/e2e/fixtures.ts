import { test as base, expect, type Page } from "@playwright/test";

export const ROUTE_IDS = ["home", "hub", "models", "tune", "chat", "images", "documents", "plugins", "tools", "security", "settings"] as const;
export type RouteIdE2E = (typeof ROUTE_IDS)[number];

export interface Guard {
  /** Console errors/warnings, uncaught page errors and CSP violations seen so far. */
  problems: string[];
  /** Requests whose origin differs from the app's. They are aborted, never sent. */
  offOrigin: string[];
  /** Mark console messages matching this pattern as expected for the current test. */
  allow(pattern: RegExp): void;
}

/**
 * Every test runs under a guard: any console error or warning, uncaught exception, CSP violation or
 * request to another origin fails the test after it finishes. Off-origin requests are also aborted
 * at the network layer, so nothing can leave the machine even if a check is missed.
 */
export const test = base.extend<{ guard: Guard }>({
  guard: [
    async ({ page, baseURL }, use) => {
      const origin = new URL(baseURL ?? "http://127.0.0.1").origin;
      const allowed: RegExp[] = [];
      const guard: Guard = {
        problems: [],
        offOrigin: [],
        allow: (p) => allowed.push(p),
      };
      await page.route("**/*", async (route) => {
        const url = route.request().url();
        if (url.startsWith("data:") || url.startsWith("blob:") || new URL(url).origin === origin) {
          await route.continue();
        } else {
          guard.offOrigin.push(url);
          await route.abort("blockedbyclient");
        }
      });
      page.on("console", (msg) => {
        if (msg.type() !== "error" && msg.type() !== "warning") return;
        const text = `${msg.type()}: ${msg.text()}`;
        if (!allowed.some((p) => p.test(text))) guard.problems.push(text);
      });
      page.on("pageerror", (err) => guard.problems.push(`pageerror: ${err.message}`));
      await page.addInitScript(() => {
        document.addEventListener("securitypolicyviolation", (e) => {
          console.error(`CSP violation: ${e.violatedDirective} ${e.blockedURI}`);
        });
      });
      await use(guard);
      expect(guard.offOrigin, "requests left the origin").toEqual([]);
      expect(guard.problems, "console errors, warnings, page errors or CSP violations").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };

/** Loads a screen and waits until it has rendered and settled (fonts, first data, entry motion). */
export async function openScreen(page: Page, id: RouteIdE2E): Promise<void> {
  await page.goto(`/#/${id}`);
  await expect(page.locator("main h1")).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await expect(page.locator(".statusbar [data-hud='speed']")).toBeVisible();
}

/** Scrolls <main> through its full height so every scroll-triggered reveal and count-up fires. */
export async function scrollThrough(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const main = document.querySelector("main");
    if (!main) return;
    const step = Math.max(200, main.clientHeight * 0.6);
    for (let y = 0; y <= main.scrollHeight; y += step) {
      main.scrollTo({ top: y });
      await new Promise((r) => setTimeout(r, 80));
    }
    main.scrollTo({ top: main.scrollHeight });
    await new Promise((r) => setTimeout(r, 120));
  });
}

/** Waits until no CSS animation or transition is running anywhere in the document. */
export async function settle(page: Page): Promise<void> {
  await page.waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running"), undefined, { timeout: 5000 });
}
