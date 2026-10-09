import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { DONE_RECORD, expect, ONBOARDING_KEY, settle, test, TERMS_VERSION } from "./fixtures";

// Every test here starts from a profile that has never finished first-run setup.
test.use({ onboarded: false });

const step = (page: Page, id: string) => page.locator(`.onb[data-step="${id}"]`);
const primary = (page: Page) => page.locator("[data-primary]");
const stored = (page: Page) => page.evaluate((key) => window.localStorage.getItem(key), ONBOARDING_KEY);

async function openSetup(page: Page): Promise<void> {
  await page.goto("/");
  await expect(step(page, "welcome")).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
}

test("a fresh profile walks the whole setup by keyboard and lands in the app, remembered", async ({ page }) => {
  await openSetup(page);
  await expect(page.locator("h1")).toHaveText("Your models, measured on this machine.");
  await expect(page).toHaveTitle("Welcome · Set up Cerebro");
  // No app chrome yet: no sidebar, no palette trigger.
  await expect(page.locator("nav.sidebar")).toHaveCount(0);
  await expect(page.locator(".palette-trigger")).toHaveCount(0);

  await page.keyboard.press("Enter");
  // Experience level: Advanced (pros by default) is preselected; Enter keeps it.
  await expect(step(page, "level")).toBeVisible();
  await expect(page.getByRole("radio", { name: /Advanced/ })).toBeChecked();
  await page.keyboard.press("Enter");
  await expect(step(page, "hardware")).toBeVisible();
  await expect(page.locator("#onb-step-title")).toBeFocused();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
  await expect(page.locator(".scan__row[data-state='done']")).toHaveCount(4);
  await expect(page.locator(".scan-panel")).toContainText("Illustrative data");

  await page.keyboard.press("ArrowRight");
  await expect(step(page, "model")).toBeVisible();
  // Ranked for this machine, best loadable fit preselected, every card explains why.
  const radios = page.getByRole("radio");
  // Six that load here, plus "Decide later".
  await expect(radios).toHaveCount(7);
  await expect(radios.last()).toHaveAccessibleName("Decide later");
  await expect(radios.first()).toBeChecked();
  await expect(page.locator(".rec").first()).toContainText("Gemma-4-12B-it");
  await expect(page.locator(".rec").first()).toContainText("Best fit");
  await expect(page.locator(".rec-legend")).toContainText("not speed");
  for (const why of await page.locator(".rec__why").allTextContents()) expect(why.length).toBeGreaterThan(40);
  await expect(page.locator(".rec-skip")).toContainText("Mistral-Medium-3.5-128B");
  // Arrow keys move the choice like any radio group.
  await radios.first().focus();
  await page.keyboard.press("ArrowDown");
  await expect(radios.nth(1)).toBeChecked();
  const chosen = await radios.nth(1).getAttribute("value");

  await page.locator("#onb-step-title").focus();
  await page.keyboard.press("Enter");
  // Load settings, Advanced view: the automatic plan as editable fields and the backend flags.
  await expect(step(page, "configure")).toBeVisible();
  await expect(page.locator(".adv")).toBeVisible();
  await expect(page.locator(".adv pre").first()).toContainText("--host 127.0.0.1");
  await page.locator("#onb-step-title").focus();
  await page.keyboard.press("Enter");
  await expect(step(page, "privacy")).toBeVisible();
  await expect(page.locator(".promise")).toHaveCount(4);
  // Wording stays inside docs/CLAIMS.md: containment is described with what it does not do.
  await expect(page.locator(".promise").nth(1)).toContainText("does not limit which files or network addresses");

  await page.keyboard.press("Enter");
  await expect(step(page, "terms")).toBeVisible();
  // Accepting is a deliberate press, never a stray Enter or arrow.
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await expect(step(page, "terms")).toBeVisible();
  expect(await stored(page)).toBeNull();
  await expect(page.getByRole("button", { name: "Skip setup" })).toHaveCount(0);
  await page.getByRole("button", { name: "Accept and continue" }).click();

  await expect(step(page, "finish")).toBeVisible();
  await expect(page.locator(".summary")).toContainText("Gemma-4-E4B-it");
  await expect(page.locator(".code-block, pre").first()).toContainText(`cerebro tune ${chosen}`);
  await page.locator("#onb-step-title").focus();
  await page.keyboard.press("Enter");

  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  await expect(page.locator("main")).toBeFocused();
  expect(JSON.parse((await stored(page)) ?? "null")).toEqual({ v: 1, completed: true, termsVersion: TERMS_VERSION, model: chosen });

  await page.reload();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  await expect(page.locator(".onb")).toHaveCount(0);
});

test("hardware readings appear one at a time while the scan runs", async ({ page }) => {
  await openSetup(page);
  await primary(page).click();
  await primary(page).click();
  const rows = page.locator(".scan__row");
  await expect(rows).toHaveCount(4);
  // Mid-scan: something is being read while the last row still waits.
  await expect(page.locator(".scan__row[data-state='scanning']")).toHaveCount(1);
  await expect(rows.last()).toHaveAttribute("data-state", "waiting");
  const seen = new Set<string>();
  await expect
    .poll(
      async () => {
        const states = await rows.evaluateAll((els) => els.map((e) => e.getAttribute("data-state") ?? ""));
        seen.add(states.join(","));
        return states.every((s) => s === "done");
      },
      { timeout: 8000, intervals: [50] },
    )
    .toBe(true);
  // Rows land in order: never a later row done while an earlier one is not.
  for (const snapshot of seen) {
    const states = snapshot.split(",");
    const firstNotDone = states.findIndex((s) => s !== "done");
    if (firstNotDone >= 0) expect(states.slice(firstNotDone).includes("done"), snapshot).toBe(false);
  }
  expect(seen.size).toBeGreaterThan(2);
  await expect(page.locator(".memmap")).toBeVisible();
});

test("with reduced motion the scan shows every reading at once", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openSetup(page);
  await primary(page).click();
  await primary(page).click();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 1500 });
  await expect(page.locator(".scan__row[data-state='done']")).toHaveCount(4);
});

test("Escape skips to the terms; Back and the step rail move around; finishing is remembered", async ({ page }) => {
  await openSetup(page);
  await page.keyboard.press("Escape");
  await expect(step(page, "terms")).toBeVisible();
  // Escape does nothing more on the terms step.
  await page.keyboard.press("Escape");
  await expect(step(page, "terms")).toBeVisible();
  // Steps after the terms are not reachable before accepting.
  await expect(page.locator(".onb-rail").getByRole("button", { name: "Ready" })).toHaveCount(0);

  await page.locator("#onb-step-title").focus();
  await page.keyboard.press("ArrowLeft");
  await expect(step(page, "privacy")).toBeVisible();
  await page.getByRole("button", { name: "Back" }).click();
  await expect(step(page, "configure")).toBeVisible();
  await page.locator(".onb-rail").getByRole("button", { name: "Welcome" }).click();
  await expect(step(page, "welcome")).toBeVisible();

  await page.getByRole("button", { name: "Skip setup" }).click();
  await expect(step(page, "terms")).toBeVisible();
  await primary(page).click();
  await expect(step(page, "finish")).toBeVisible();
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  const record = JSON.parse((await stored(page)) ?? "null") as { completed: boolean; termsVersion: string };
  expect(record.completed).toBe(true);
  expect(record.termsVersion).toBe(TERMS_VERSION);
});

test("new terms reopen setup at the terms step only, with a notice", async ({ page }) => {
  await page.addInitScript(
    ([key, record]) => window.localStorage.setItem(key, record),
    [ONBOARDING_KEY, JSON.stringify({ v: 1, completed: true, termsVersion: "2020-01-01", model: null })] as const,
  );
  await page.goto("/");
  await expect(step(page, "terms")).toBeVisible();
  await expect(page.locator(".onb-changed")).toContainText("The terms changed");
  await page.getByRole("button", { name: "Accept and continue" }).click();
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
});

test("a malformed or tampered record just shows setup again", async ({ page }) => {
  await page.addInitScript((key) => window.localStorage.setItem(key, '{"v":1,"completed":true,"termsVersion":"<script>","model":"x"}'), ONBOARDING_KEY);
  await page.goto("/");
  await expect(step(page, "welcome")).toBeVisible();
});

test("the full Terms, Privacy and Licenses open in place from the bundle, by keyboard", async ({ page, guard }) => {
  const requests: string[] = [];
  page.on("request", (r) => requests.push(r.url()));
  await openSetup(page);
  await page.keyboard.press("Escape");
  await expect(step(page, "terms")).toBeVisible();

  const open = page.getByRole("button", { name: "Full terms" });
  await open.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("heading", { level: 2 })).toHaveText("Terms of use");
  await expect(dialog).toContainText("Provided as is");
  await expect(dialog.getByRole("tab", { name: "Terms of use" })).toBeFocused();

  await page.keyboard.press("ArrowRight");
  await expect(dialog.getByRole("tab", { name: "Privacy" })).toHaveAttribute("aria-selected", "true");
  await expect(dialog).toContainText("no telemetry");
  await page.keyboard.press("End");
  await expect(dialog.getByRole("tab", { name: "Licenses" })).toBeFocused();
  // Cerebro, the generated JavaScript list, three typefaces, the generated Rust list, llama.cpp.
  await expect(dialog.locator(".legal__license")).toHaveCount(7);
  await dialog.getByText("Full licence text").first().click();
  await expect(dialog.locator(".legal__pre").first()).toContainText("Apache License");

  // Setup keys stay inert while the dialog is open.
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(step(page, "terms")).toBeVisible();
  await expect(open).toBeFocused();

  // Everything came from this origin (the guard fails the test otherwise, too).
  const origin = new URL(page.url()).origin;
  expect(requests.filter((u) => !u.startsWith("data:") && new URL(u).origin !== origin)).toEqual([]);
  expect(guard.offOrigin).toEqual([]);
});

test("setup can be run again from Settings and from the command palette", async ({ page }) => {
  await page.addInitScript(([key, record]) => window.localStorage.getItem(key) ?? window.localStorage.setItem(key, record), [ONBOARDING_KEY, DONE_RECORD] as const);
  await page.goto("/#/settings");
  await expect(page.locator("main h1")).toHaveText("Settings");
  await page.getByRole("button", { name: "Run setup again" }).click();
  await expect(step(page, "welcome")).toBeVisible();
  // Terms already accepted: Skip finishes straight away.
  await page.getByRole("button", { name: "Skip setup" }).click();
  await expect(page.locator("main h1")).toHaveText("Settings");

  await page.keyboard.press("Control+k");
  await page.getByRole("combobox").fill("run setup");
  await page.keyboard.press("Enter");
  await expect(step(page, "welcome")).toBeVisible();
});

test("legal documents are also reachable from Settings", async ({ page }) => {
  await page.addInitScript(([key, record]) => window.localStorage.setItem(key, record), [ONBOARDING_KEY, DONE_RECORD] as const);
  await page.goto("/#/settings");
  await page.locator(".about__doc", { hasText: "Licenses" }).click();
  await expect(page.getByRole("dialog").getByRole("heading", { level: 2 })).toHaveText("Licenses");
});

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
async function scan(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

for (const scheme of ["dark", "light"] as const) {
  test(`every setup step has no axe violations (${scheme})`, async ({ page }) => {
    test.setTimeout(90_000);
    await page.emulateMedia({ colorScheme: scheme });
    await openSetup(page);
    for (const id of ["welcome", "level", "hardware", "model", "configure", "privacy", "terms", "finish"]) {
      await expect(step(page, id)).toBeVisible();
      if (id === "hardware") await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
      await settle(page).catch(() => undefined);
      expect(await scan(page), `step ${id}`).toEqual([]);
      if (id === "configure") {
        // Both views of the load settings.
        await page.getByRole("button", { name: "Switch to Beginner" }).click();
        await expect(page.locator(".plan")).toBeVisible();
        await settle(page).catch(() => undefined);
        expect(await scan(page), "configure, Beginner").toEqual([]);
        await page.getByRole("button", { name: "Switch to Advanced" }).click();
        await expect(page.locator(".adv")).toBeVisible();
      }
      if (id === "terms") {
        await page.getByRole("button", { name: "Licenses" }).click();
        await expect(page.locator(".legal__license").first()).toBeVisible();
        await settle(page).catch(() => undefined);
        expect(await scan(page), "legal dialog").toEqual([]);
        await page.keyboard.press("Escape");
        await expect(page.getByRole("dialog")).toHaveCount(0);
      }
      if (id !== "finish") await primary(page).click();
    }
  });
}
