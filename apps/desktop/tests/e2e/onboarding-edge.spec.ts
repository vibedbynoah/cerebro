import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, ONBOARDING_KEY, settle, test, TERMS_VERSION } from "./fixtures";

/**
 * First-run setup off the happy path: a service that fails (the mock's `?mock=` switches, web build
 * only), a stale stored choice, keyboard and screen-reader details, small windows, and the generated
 * Licenses lists. Every test runs under the fixtures' guard, so any uncaught error (pageerror) or
 * console error fails it.
 */
test.use({ onboarded: false });

const step = (page: Page, id: string) => page.locator(`.onb[data-step="${id}"]`);
const primary = (page: Page) => page.locator("[data-primary]");
const stored = async (page: Page) => JSON.parse((await page.evaluate((key) => window.localStorage.getItem(key), ONBOARDING_KEY)) ?? "null") as { model: string | null } | null;

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
/** axe violations in both colour schemes (the error tints differ per theme). */
async function axe(page: Page): Promise<string[]> {
  const out: string[] = [];
  for (const colorScheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme });
    await settle(page).catch(() => undefined);
    const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    out.push(...results.violations.map((v) => `${colorScheme} ${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`));
  }
  return out;
}

async function open(page: Page, query = ""): Promise<void> {
  await page.goto(`/${query}`);
  await expect(step(page, "welcome")).toBeVisible();
}

async function toTermsAndFinish(page: Page): Promise<void> {
  await primary(page).click();
  await expect(step(page, "terms")).toBeVisible();
  await page.getByRole("button", { name: "Accept and continue" }).click();
  await expect(step(page, "finish")).toBeVisible();
}

test("hardware detection fails: setup still runs, says so, and lists every model unranked to pick by hand", async ({ page }) => {
  await open(page, "?mock=fail-hw");
  await primary(page).click();
  await primary(page).click();
  await expect(step(page, "hardware")).toBeVisible();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "error");
  await expect(page.getByRole("alert")).toContainText("The hardware check did not finish");
  expect(await axe(page), "hardware error").toEqual([]);

  await primary(page).click();
  await expect(step(page, "model")).toBeVisible();
  await expect(page.locator("h1")).toHaveText("Pick a first model.");
  await expect(page.locator(".rec-legend")).toContainText("listed unranked");
  // The whole catalog, in catalog order, nothing ruled out, nothing preselected, no "best".
  const radios = page.getByRole("radio");
  await expect(radios).toHaveCount(8);
  await expect(page.locator(".rec__name").first()).toHaveText("Gemma-4-12B-it");
  await expect(page.locator(".rec").nth(6)).toContainText("Mistral-Medium-3.5-128B");
  await expect(page.locator(".rec[data-rank='1']")).toHaveCount(0);
  await expect(page.locator(".rec-skip")).toHaveCount(0);
  await expect(page.getByText("No model in the catalog fits")).toHaveCount(0);
  for (let i = 0; i < 8; i++) await expect(radios.nth(i)).not.toBeChecked();
  expect(await axe(page), "model step, unranked").toEqual([]);

  await page.locator(".rec").nth(2).click();
  await expect(radios.nth(2)).toBeChecked();
  const chosen = await radios.nth(2).getAttribute("value");
  await primary(page).click();
  // Without a hardware reading the load settings still have a plan (sized from the file alone).
  await expect(step(page, "configure")).toBeVisible();
  await expect(page.locator(".adv")).toBeVisible();
  expect(await axe(page), "configure, no hardware").toEqual([]);
  await primary(page).click();
  await expect(step(page, "privacy")).toBeVisible();
  await toTermsAndFinish(page);
  await expect(page.locator(".summary")).toContainText("Not checked");
  await expect(page.locator(".finish-payoff")).toContainText("gpt-oss-20b");
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  expect((await stored(page))?.model).toBe(chosen);
});

test("Scan again after a failed check reads the machine and finishes the scan", async ({ page }) => {
  await open(page, "?mock=fail-hw-once");
  await primary(page).click();
  await primary(page).click();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "error");
  await page.getByRole("button", { name: "Scan again" }).click();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
  await expect(page.locator(".scan__row[data-state='done']")).toHaveCount(4);
  await expect(page.getByRole("alert")).toHaveCount(0);
  // Ranked again, with the best fit preselected.
  await primary(page).click();
  await expect(page.locator(".rec[data-rank='1']")).toContainText("Gemma-4-12B-it");
  await expect(page.getByRole("radio").first()).toBeChecked();
});

test("the model list fails: setup says so plainly and still finishes, with no model remembered", async ({ page }) => {
  await open(page, "?mock=fail-catalog");
  await primary(page).click();
  await primary(page).click();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
  await primary(page).click();
  await expect(step(page, "model")).toBeVisible();
  await expect(page.locator(".rec-notice")).toContainText("The model list could not be read");
  await expect(page.getByRole("radio")).toHaveCount(0);
  await primary(page).click();
  await expect(step(page, "configure")).toBeVisible();
  await expect(page.locator(".rec-notice")).toContainText("No first model chosen");
  await primary(page).click();
  await expect(step(page, "privacy")).toBeVisible();
  await toTermsAndFinish(page);
  await expect(page.locator(".summary")).toContainText("None yet");
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  expect((await stored(page))?.model).toBeNull();
});

test("if setup itself breaks, a fallback still offers the terms and Accept, and the app opens", async ({ page }) => {
  // A malformed hardware answer makes the hardware step throw while rendering.
  await open(page, "?mock=bad-hw");
  await primary(page).click();
  await primary(page).click();
  await expect(page.locator(".setup-fallback")).toBeVisible();
  await expect(page.locator("h1")).toHaveText("Setup could not finish.");
  await expect(page.locator("#setup-fallback-title")).toBeFocused();
  expect(await axe(page), "setup fallback").toEqual([]);
  await page.getByRole("button", { name: "Read the terms" }).click();
  await expect(page.getByRole("dialog").getByRole("heading", { level: 2 })).toHaveText("Terms of use");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.getByRole("button", { name: "Accept and open Cerebro" }).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  const record = await page.evaluate((key) => window.localStorage.getItem(key), ONBOARDING_KEY);
  expect(JSON.parse(record ?? "null")).toEqual({ v: 1, completed: true, termsVersion: TERMS_VERSION, model: null });
});

test("on the model step, Down then Enter picks the second model and continues", async ({ page }) => {
  await open(page);
  await primary(page).click();
  await primary(page).click();
  await primary(page).click();
  await expect(step(page, "model")).toBeVisible();
  const radios = page.getByRole("radio");
  await radios.first().focus();
  await page.keyboard.press("ArrowDown");
  await expect(radios.nth(1)).toBeChecked();
  const chosen = await radios.nth(1).getAttribute("value");
  await page.keyboard.press("Enter");
  await expect(step(page, "configure")).toBeVisible();
  await page.keyboard.press("Escape"); // skip to the terms
  await expect(step(page, "terms")).toBeVisible();
  await page.getByRole("button", { name: "Accept and continue" }).click();
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  expect((await stored(page))?.model).toBe(chosen);
});

test("each radio is named by model and fit only, and described by its reason and size", async ({ page }) => {
  await open(page);
  await primary(page).click();
  await primary(page).click();
  await primary(page).click();
  await expect(step(page, "model")).toBeVisible();
  const first = page.getByRole("radio").first();
  await expect(first).toHaveAccessibleName("Gemma-4-12B-it Best fit Fits in video memory");
  await expect(first).toHaveAccessibleDescription(/^Fits entirely in your .* 6\.9 GiB$/);
});

test("a stored model that no longer loads is never written back; Decide later stores none", async ({ page }) => {
  await page.addInitScript(
    ([key, record]) => window.localStorage.getItem(key) ?? window.localStorage.setItem(key, record),
    [ONBOARDING_KEY, JSON.stringify({ v: 1, completed: false, termsVersion: null, model: "old-model-q2" })] as const,
  );
  // Skipping straight through: the stale id is dropped.
  await open(page);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Accept and continue" }).click();
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  expect((await stored(page))?.model).toBeNull();

  // Visiting the model step: the stale id is replaced by the best fit, and Decide later stores null.
  await page.evaluate((key) => window.localStorage.setItem(key, JSON.stringify({ v: 1, completed: false, termsVersion: null, model: "old-model-q2" })), ONBOARDING_KEY);
  await page.reload();
  await expect(step(page, "welcome")).toBeVisible();
  await primary(page).click();
  await primary(page).click();
  await primary(page).click();
  await expect(step(page, "model")).toBeVisible();
  await expect(page.getByRole("radio").first()).toBeChecked();
  await page.locator(".rec--later").click();
  await expect(page.getByRole("radio", { name: "Decide later" })).toBeChecked();
  await primary(page).click();
  await expect(step(page, "configure")).toBeVisible();
  await primary(page).click();
  await expect(step(page, "privacy")).toBeVisible();
  await toTermsAndFinish(page);
  await expect(page.locator(".finish-payoff")).toContainText("No first model yet");
  await primary(page).click();
  await expect(page.locator("main h1")).toHaveText("This machine, measured.");
  expect((await stored(page))?.model).toBeNull();
});

test("steps are announced once (heading focus, with the count as its description); the rail says what is done", async ({ page }) => {
  await open(page);
  await expect(page.locator("#onb-step-title")).toHaveAttribute("aria-describedby", "onb-step-count");
  await primary(page).click();
  await expect(page.locator("#onb-step-title")).toBeFocused();
  await expect(page.locator("#onb-step-title")).toHaveAccessibleDescription("Step 2 of 8");
  // The live region stays quiet on a plain step change (the scan's own result may speak later).
  await expect(page.locator(".onb > [role='status']")).not.toContainText("Step 2");
  const rail = page.locator(".onb-rail");
  // (Chromium inserts a space before the visually hidden state; screen readers read it the same.)
  await expect(rail.getByRole("button", { name: /^Welcome\s?, completed$/ })).toBeVisible();
  await expect(rail.locator("[aria-current='step']")).toContainText(/Experience\s?, current step/);
  // Skipping still explains itself.
  await page.keyboard.press("Escape");
  await expect(page.locator(".onb > [role='status']")).toHaveText("Setup skipped. One step left: the terms.");
});

for (const size of [
  { width: 1024, height: 640 },
  { width: 900, height: 700 },
]) {
  test(`every step fits a ${size.width}x${size.height} window: no sideways scroll, footer reachable`, async ({ page }) => {
    await page.setViewportSize(size);
    await open(page);
    const rail = page.locator(".onb-rail");
    if (size.width <= 900) await expect(rail).toBeHidden();
    else await expect(rail).toBeVisible();
    for (const id of ["welcome", "level", "hardware", "model", "configure", "privacy", "terms", "finish"]) {
      await expect(step(page, id)).toBeVisible();
      if (id === "hardware") await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
      // Measured at rest: a step's entry slide is a transform inside an overflow-hidden stage.
      await settle(page).catch(() => undefined);
      const overflow = await page.evaluate(() => {
        const scroller = document.querySelector(".onb-scroll");
        return {
          doc: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          stage: scroller ? scroller.scrollWidth - scroller.clientWidth : 0,
        };
      });
      expect(overflow, id).toEqual({ doc: 0, stage: 0 });
      await expect(primary(page)).toBeInViewport();
      if (id === "terms") await page.getByRole("button", { name: "Accept and continue" }).click();
      else if (id !== "finish") await primary(page).click();
    }
  });
}

test("the Licenses page lists every package the build bundled, and every Rust crate, with texts", async ({ page, request }) => {
  const shipped = (await (await request.get("/third-party-packages.json")).json()) as Array<{ name: string }>;
  expect(shipped.map((p) => p.name)).toEqual(expect.arrayContaining(["solid-js", "@ark-ui/solid", "@floating-ui/dom", "@floating-ui/core", "@zag-js/dialog"]));
  await open(page);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Licenses" }).click();
  const dialog = page.getByRole("dialog");
  const js = dialog.locator("[data-notices='bundled']");
  await expect(js).toContainText(`${shipped.length} packages`);
  await js.getByText(`Show all ${shipped.length}`).click();
  // Rows are built when the group first opens (the toggle event): wait for them before reading.
  await expect(js.locator(".legal__pkg-name")).toHaveCount(shipped.length);
  const names = await js.locator(".legal__pkg-name").allTextContents();
  expect(names.sort()).toEqual(shipped.map((p) => p.name).sort());
  // A row's licence text appears when it is opened, as plain text.
  const zag = js.locator(".legal__pkg", { hasText: "@zag-js/dialog" });
  await zag.locator("summary").click();
  await expect(zag.locator(".legal__pre")).toContainText("Chakra UI");

  const rust = dialog.locator("[data-notices='rust']");
  await rust.locator("summary").first().click();
  await expect.poll(() => rust.locator(".legal__pkg-name").count()).toBeGreaterThan(100);
  const crates = await rust.locator(".legal__pkg-name").allTextContents();
  expect(crates.length).toBeGreaterThan(100);
  expect(crates).toEqual(expect.arrayContaining(["tauri", "serde", "ed25519-dalek"]));
  const tauri = rust.locator(".legal__pkg").filter({ has: page.locator(".legal__pkg-name", { hasText: /^tauri$/ }) });
  await tauri.locator("summary").click();
  await expect(tauri.locator(".legal__pre").first()).toContainText(/Apache License|MIT/);
});

test("switching legal documents opens each at its top", async ({ page }) => {
  await open(page);
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Licenses" }).click();
  const body = page.locator(".legal__body");
  await page.locator("[data-notices='rust'] summary").first().click();
  await expect.poll(() => page.locator("[data-notices='rust'] .legal__pkg").count()).toBeGreaterThan(100);
  await body.evaluate((el) => el.scrollTo({ top: el.scrollHeight }));
  await expect.poll(() => body.evaluate((el) => el.scrollTop)).toBeGreaterThan(200);
  await page.getByRole("tab", { name: "Privacy" }).click();
  await expect(page.getByRole("dialog").getByRole("heading", { level: 2 })).toHaveText("Privacy");
  await expect.poll(() => body.evaluate((el) => el.scrollTop)).toBe(0);
});
