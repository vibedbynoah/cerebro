import { test as base, expect, type Page } from "@playwright/test";
import { ONBOARDING_KEY, TERMS_VERSION } from "../../ui/src/onboarding/progress";
import { GUIDE_VERSION } from "../../ui/src/lib/features";

export const ROUTE_IDS = ["home", "hub", "models", "cluster", "tune", "monitor", "turbo", "chat", "images", "documents", "train", "plugins", "tools", "settings", "about"] as const;
export type RouteIdE2E = (typeof ROUTE_IDS)[number];

/** What a finished first-run setup leaves in local storage, from the app's own constants, so a
 *  terms bump never leaves the suite seeding a stale record. */
export { ONBOARDING_KEY, TERMS_VERSION };
export const DONE_RECORD = JSON.stringify({ v: 1, completed: true, termsVersion: TERMS_VERSION, model: "qwen3-8b-q5km" });

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
export const test = base.extend<{ guard: Guard; onboarded: boolean; guideFirst: boolean }>({
  /** Every test starts past first-run setup unless it opts out with `test.use({ onboarded: false })`,
   *  which gives it a fresh profile that has never finished setup. */
  onboarded: [true, { option: true }],
  /** The feature guide opens on entering the app; tests skip it unless they set this. */
  guideFirst: [false, { option: true }],
  guard: [
    async ({ page, baseURL, onboarded, guideFirst }, use) => {
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
      if (onboarded) {
        // Only when nothing is stored yet, so a test that changes the record keeps its change.
        await page.addInitScript(
          ([key, record]) => {
            if (window.localStorage.getItem(key) === null) window.localStorage.setItem(key, record);
          },
          [ONBOARDING_KEY, DONE_RECORD] as const,
        );
      }
      // The feature guide opens on entering the app; every spec but its own starts with it seen.
      if (!guideFirst) await page.addInitScript((v) => window.localStorage.setItem("cerebro-guide-seen", v), GUIDE_VERSION);
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

/** Opens Chat's "Fine-tune" disclosure, which holds the finer composer controls. */
export async function openFineTune(page: Page): Promise<void> {
  const more = page.locator("details.composer__more");
  if (!(await more.evaluate((d) => (d as HTMLDetailsElement).open))) {
    await page.locator(".composer__more > summary").click();
  }
  await expect(more).toHaveJSProperty("open", true);
}

/** Waits until no CSS animation or transition is running anywhere in the document. */
export async function settle(page: Page): Promise<void> {
  await page.waitForFunction(() => document.getAnimations().every((a) => a.playState !== "running"), undefined, { timeout: 5000 });
}
