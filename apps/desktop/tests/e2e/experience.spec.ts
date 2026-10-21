import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { EXPERIENCE_KEY, CHANNEL_KEY } from "../../ui/src/lib/experience";
import { SETUP_CONFIG_KEY } from "../../ui/src/onboarding/autoconfig";
import { DONE_RECORD, expect, ONBOARDING_KEY, openScreen, settle, test } from "./fixtures";

/**
 * Experience levels (docs/specs/positioning.md): Beginner or Advanced is chosen during setup, the
 * escape hatch on every setup step switches views in place without losing a value, and the choice
 * persists and can be changed in Settings. Every test runs under the fixtures' guard.
 */

const step = (page: Page, id: string) => page.locator(`.onb[data-step="${id}"]`);
const primary = (page: Page) => page.locator("[data-primary]");
const hatch = (page: Page) => page.locator(".onb-level-switch");
const pref = (page: Page, key: string) => page.evaluate((k) => window.localStorage.getItem(k), key);
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function axe(page: Page): Promise<string[]> {
  await settle(page).catch(() => undefined);
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

/** Anything in <main> poking past its right edge, or the page scrolling sideways. Wide tables scroll
 *  inside their own region, so they never count. */
async function overflow(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const bad: string[] = [];
    if (document.documentElement.scrollWidth > window.innerWidth) bad.push("document scrolls sideways");
    const main = document.querySelector("main");
    if (!main) return bad;
    if (main.scrollWidth > main.clientWidth + 1) bad.push(`main overflows by ${main.scrollWidth - main.clientWidth}px`);
    const edge = main.getBoundingClientRect().right;
    for (const el of Array.from(main.querySelectorAll("*"))) {
      if (el.closest(".qtable-scroll") && !el.classList.contains("qtable-scroll")) continue;
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.right > edge + 1) bad.push(`${el.tagName.toLowerCase()}.${String(el.getAttribute("class"))} ends at ${Math.round(r.right)}`);
    }
    return bad.slice(0, 5);
  });
}

/** From the welcome step to the load-settings step, choosing `level` on the way. */
async function toConfigure(page: Page, level: "beginner" | "advanced"): Promise<void> {
  await page.goto("/");
  await expect(step(page, "welcome")).toBeVisible();
  await primary(page).click();
  await expect(step(page, "level")).toBeVisible();
  await page.locator(`.level-card[data-level="${level}"]`).click();
  await expect(page.locator(`#level-${level}`)).toBeChecked();
  await primary(page).click();
  await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
  await primary(page).click();
  await expect(step(page, "model")).toBeVisible();
  await expect(page.getByRole("radio").first()).toBeChecked();
  await primary(page).click();
  await expect(step(page, "configure")).toBeVisible();
}

test.describe("during setup", () => {
  test.use({ onboarded: false });

  test("Beginner: Cerebro sets every value from the hardware check, explains each, and shows no raw flags", async ({ page }) => {
    await toConfigure(page, "beginner");
    await expect(page.locator("#onb-step-title")).toHaveText("Cerebro set this up for you.");
    const plan = page.locator(".plan__item");
    await expect(plan).toHaveCount(5);
    await expect(page.locator('.plan__item[data-field="quantId"] .plan__value')).toHaveText("UD-Q4_K_XL, 6.9 GiB");
    await expect(page.locator('.plan__item[data-field="contextLength"] .plan__value')).toHaveText("32K tokens");
    await expect(page.locator('.plan__item[data-field="gpuLayers"] .plan__value')).toHaveText("Graphics card");
    // Every row explains itself and defines its term inline.
    for (const text of await page.locator(".plan__text").allTextContents()) expect(text.length).toBeGreaterThan(25);
    await expect(page.locator(".plan__label .term")).toHaveCount(5);
    // No flags, no file table, no channel picker.
    await expect(step(page, "configure")).not.toContainText("llama-server");
    await expect(step(page, "configure")).not.toContainText("-ngl");
    await expect(page.locator(".qtable")).toHaveCount(0);
    await expect(page.locator('input[name="setup-channel"]')).toHaveCount(0);
    // Starting values, not measurements.
    await expect(step(page, "configure")).toContainText("not measurements");
    expect(await axe(page)).toEqual([]);

    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Accept and continue" }).click();
    await expect(step(page, "finish")).toBeVisible();
    await expect(page.locator(".summary")).toContainText("Beginner");
    await primary(page).click();
    await expect(page.locator("main h1")).toHaveText("This machine, measured.");
    expect(await pref(page, EXPERIENCE_KEY)).toBe("beginner");
    expect(JSON.parse((await pref(page, SETUP_CONFIG_KEY)) ?? "null")).toMatchObject({ modelId: "gemma-4-12b-it", quantId: "gemma-4-12b-it--ud-q4-k-xl", contextLength: 32768, gpuLayers: 48 });
  });

  test("Advanced: every file with KL, perplexity and speed, every load parameter, channels and live flags", async ({ page }) => {
    await toConfigure(page, "advanced");
    await expect(page.locator("#onb-step-title")).toHaveText("Every load setting, in view.");
    const table = page.locator(".qtable");
    await expect(table.getByRole("columnheader", { name: "KL" })).toBeVisible();
    await expect(table.getByRole("columnheader", { name: "PPL Δ" })).toBeVisible();
    await expect(table.getByRole("columnheader", { name: "Quality per GiB" })).toBeVisible();
    for (const family of ["Unsloth Dynamic", "GSQ-RCO", "imatrix", "EXL3", "AWQ", "MLX"]) await expect(table).toContainText(family);
    await expect(page.locator(".qtable-wrap")).toContainText("Illustrative data");
    // Beginner's file is the starting value; files for other engines are listed but cannot be picked.
    await expect(page.locator('input[name="setup-quant"]:checked')).toHaveValue("gemma-4-12b-it--ud-q4-k-xl");
    await expect(page.locator('.qrow[data-quant="gemma-4-12b-it--exl3-4-0bpw"] input')).toBeDisabled();
    await expect(page.locator('.qrow[data-quant="gemma-4-12b-it--exl3-4-0bpw"]')).toContainText("ExLlamaV3");
    // GSQ-RCO runs on Strata, which is on the Experimental channel.
    const gsq = page.locator('.qrow[data-quant="gemma-4-12b-it--gsq-rco-3-5"]');
    await expect(gsq.locator("input")).toBeDisabled();
    await expect(gsq).toContainText("Needs the Experimental channel");
    await page.locator('input[name="setup-channel"][value="experimental"]').check();
    await expect(gsq.locator("input")).toBeEnabled();
    await gsq.locator("label").click();
    await expect(gsq.locator("input")).toBeChecked();
    const flags = page.locator(".adv pre");
    await expect(flags).toContainText("strata-server -m Gemma-4-12B-it-GSQ-RCO-3.5.gguf");
    // Back to Stable: the GSQ-RCO file cannot run any more, so the automatic file comes back.
    await page.locator('input[name="setup-channel"][value="stable"]').check();
    await expect(page.locator('input[name="setup-quant"]:checked')).toHaveValue("gemma-4-12b-it--ud-q4-k-xl");
    await expect(flags).toContainText("cerebro serve -m Gemma-4-12B-it-UD-Q4_K_XL.gguf");
    // Every edit flows into the flags; values outside the model's range are clamped.
    await page.locator("#cfg-ngl").fill("999");
    await page.locator("#cfg-ngl").blur();
    await expect(page.locator("#cfg-ngl")).toHaveValue("48");
    await page.locator("#cfg-kv").selectOption("q4_0");
    await expect(flags).toContainText("-ctk q4_0 -ctv q4_0");
    await expect(flags).toContainText("--host 127.0.0.1");
    await expect(page.locator(".config-changed")).toContainText("1 value differs");
    expect(await axe(page)).toEqual([]);
  });

  test("the escape hatch is on every step and switches in place, without leaving the step", async ({ page }) => {
    await page.goto("/");
    const ids = ["welcome", "level", "hardware", "model", "configure", "privacy", "terms", "finish"];
    for (const id of ids) {
      await expect(step(page, id)).toBeVisible();
      if (id === "hardware") await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
      await expect(hatch(page), id).toBeVisible();
      await expect(hatch(page), id).toBeInViewport();
      await expect(hatch(page)).toHaveText("Switch to Beginner");
      await hatch(page).click();
      await expect(step(page, id), `${id} stays put`).toBeVisible();
      await expect(hatch(page)).toHaveText("Switch to Advanced");
      expect(await pref(page, EXPERIENCE_KEY)).toBe("beginner");
      if (id === "level") await expect(page.locator("#level-beginner")).toBeChecked();
      await hatch(page).click();
      await expect(hatch(page)).toHaveText("Switch to Beginner");
      if (id === "terms") await page.getByRole("button", { name: "Accept and continue" }).click();
      else if (id !== "finish") await primary(page).click();
    }
    // Announced politely, and it says nothing is lost.
    await hatch(page).click();
    await expect(page.locator(".onb > [role='status']")).toHaveText("Beginner view. Every value is kept.");
  });

  test("picked Beginner by mistake: switch mid-setup, Beginner's values start Advanced, edits survive switching back", async ({ page }) => {
    await toConfigure(page, "beginner");
    await expect(page.locator(".plan")).toBeVisible();

    await page.getByRole("button", { name: "Switch to Advanced" }).click();
    await expect(step(page, "configure")).toBeVisible();
    await expect(page.locator(".adv")).toBeVisible();
    await expect(page.locator('input[name="setup-quant"]:checked')).toHaveValue("gemma-4-12b-it--ud-q4-k-xl");
    await expect(page.locator("#cfg-ctx")).toHaveValue("32768");
    await expect(page.locator("#cfg-ngl")).toHaveValue("48");
    await expect(page.locator(".config-changed")).toHaveCount(0);

    await page.locator("#cfg-ctx").selectOption("8192");
    await page.locator("#cfg-temp").fill("0.6");
    await page.locator("#cfg-temp").blur();
    await expect(page.locator(".adv pre")).toContainText("-c 8192");
    await expect(page.locator(".adv pre")).toContainText("--temp 0.6");
    await expect(page.locator(".config-changed")).toContainText("2 values differ");

    // Back to Beginner: the edits are kept and marked.
    await page.getByRole("button", { name: "Switch to Beginner" }).click();
    await expect(page.locator(".plan")).toBeVisible();
    await expect(page.locator('.plan__item[data-field="contextLength"] .plan__value')).toHaveText("8K tokens");
    await expect(page.locator('.plan__item[data-field="contextLength"]')).toHaveAttribute("data-changed", "true");
    await expect(page.locator('.plan__item[data-field="temperature"]')).toHaveAttribute("data-changed", "true");
    await expect(page.locator('.plan__item[data-field="quantId"]')).toHaveAttribute("data-changed", "false");
    await expect(page.locator(".config-changed")).toContainText("changed in Advanced. They are kept.");

    // Leaving the step and coming back keeps them too.
    await page.getByRole("button", { name: "Back" }).click();
    await expect(step(page, "model")).toBeVisible();
    await primary(page).click();
    await expect(page.locator('.plan__item[data-field="contextLength"] .plan__value')).toHaveText("8K tokens");

    // One button back to the automatic values.
    await page.getByRole("button", { name: "Use automatic values" }).click();
    await expect(page.locator('.plan__item[data-field="contextLength"] .plan__value')).toHaveText("32K tokens");
    await expect(page.locator(".config-changed")).toHaveCount(0);

    // Advanced again, one edit, and finish: the plan and the level are remembered.
    await page.getByRole("button", { name: "Switch to Advanced" }).click();
    await page.locator("#cfg-ctx").selectOption("16384");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Accept and continue" }).click();
    await expect(page.locator(".summary")).toContainText("Advanced");
    await expect(page.locator(".summary")).toContainText("16K context");
    await primary(page).click();
    await expect(page.locator("main h1")).toHaveText("This machine, measured.");
    expect(await pref(page, EXPERIENCE_KEY)).toBe("advanced");
    expect(JSON.parse((await pref(page, SETUP_CONFIG_KEY)) ?? "null")).toMatchObject({ quantId: "gemma-4-12b-it--ud-q4-k-xl", contextLength: 16384, temperature: 1 });
  });

  test("a level chosen mid-setup survives a restart of setup, and the saved plan comes back", async ({ page }) => {
    // A reload mid-setup keeps the level.
    await page.goto("/");
    await primary(page).click();
    await page.locator('.level-card[data-level="beginner"]').click();
    await page.reload();
    await expect(step(page, "welcome")).toBeVisible();
    await expect(hatch(page)).toHaveText("Switch to Advanced");

    await toConfigure(page, "beginner");
    await page.getByRole("button", { name: "Switch to Advanced" }).click();
    await page.locator("#cfg-ctx").selectOption("8192");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Accept and continue" }).click();
    await primary(page).click();
    await expect(page.locator("main h1")).toHaveText("This machine, measured.");

    // Run setup again: Advanced is still chosen and the stored plan is the starting point.
    await page.goto("/#/settings");
    await page.getByRole("button", { name: "Run setup again" }).click();
    await expect(step(page, "welcome")).toBeVisible();
    await expect(hatch(page)).toHaveText("Switch to Beginner");
    await primary(page).click();
    await expect(page.locator("#level-advanced")).toBeChecked();
    await primary(page).click();
    await expect(page.locator(".scan-panel")).toHaveAttribute("data-state", "done", { timeout: 8000 });
    await primary(page).click();
    await primary(page).click();
    await expect(page.locator("#cfg-ctx")).toHaveValue("8192");
  });
});

test.describe("after setup", () => {
  test("Settings changes the level and, in Advanced, the channel; both persist across a reload", async ({ page }) => {
    await openScreen(page, "settings");
    const section = page.locator("section", { has: page.locator("#experience-title") });
    await expect(section.getByRole("radio", { name: "Advanced" })).toBeChecked();
    await expect(section.getByRole("radio", { name: "Stable" })).toBeChecked();
    await expect(section.locator('.opt-list__item[data-on="true"]')).toHaveCount(2);

    await section.getByRole("radio", { name: "Experimental" }).check();
    await expect(section.locator('.opt-list__item[data-on="true"]')).toHaveCount(6);
    await expect(page.locator(".toast").last()).toContainText("Experimental channel");

    await section.getByRole("radio", { name: "Beginner" }).check();
    await expect(section.getByRole("radio", { name: "Beginner" })).toBeChecked();
    // Beginner shows the channel but does not offer to change it.
    await expect(section.getByRole("radio", { name: "Experimental" })).toHaveCount(0);
    await expect(section.locator(".channel-card__locked")).toContainText("Experimental. Switch to Advanced to change it.");
    expect(await overflow(page)).toEqual([]);
    expect(await axe(page)).toEqual([]);

    await page.reload();
    await expect(page.locator("main h1")).toHaveText("Settings");
    await expect(section.getByRole("radio", { name: "Beginner" })).toBeChecked();
    expect(await pref(page, CHANNEL_KEY)).toBe("experimental");
    await section.getByRole("radio", { name: "Advanced" }).check();
    await expect(section.getByRole("radio", { name: "Experimental" })).toBeChecked();
  });

  test("Model Hub: current models, every file by quality per GiB in Advanced, Cerebro's pick in Beginner, nothing downloaded", async ({ page }) => {
    const requests: string[] = [];
    page.on("request", (r) => requests.push(r.url()));
    await openScreen(page, "hub");
    const cards = page.locator(".hub-card");
    await expect(cards).toHaveCount(7);
    await expect(cards.first()).toHaveAttribute("data-model", "gemma-4-12b-it");
    await expect(page.locator(".hub-list")).not.toContainText(/Llama-3|Qwen3-8B|Mistral-Small|Phi-4|DeepSeek-R1/);
    await expect(page.locator(".qtable")).toHaveCount(7);
    const first = cards.first();
    for (const family of ["Unsloth Dynamic", "GSQ-RCO", "imatrix", "K-quant", "EXL3", "AWQ", "MLX"]) await expect(first).toContainText(family);
    await expect(first.locator(".qrow__best").first()).toBeVisible();
    await expect(page.locator(".hub-meta")).toContainText("Illustrative data");
    await expect(page.locator(".hub-card[data-fit='too-large']")).toContainText("Too large for this machine");
    expect(await overflow(page)).toEqual([]);
    expect(await axe(page)).toEqual([]);

    await page.evaluate((k) => window.localStorage.setItem(k, "beginner"), EXPERIENCE_KEY);
    await page.reload();
    await expect(page.locator(".hub-card").first()).toBeVisible();
    await expect(page.locator(".qtable")).toHaveCount(0);
    await expect(first.locator(".hub-pick__file")).toHaveText("UD-Q4_K_XL");
    await expect(first.locator(".hub-pick__nums")).toContainText("tok/s here");
    const more = first.locator(".hub-card__more");
    await expect(more).toHaveAttribute("aria-expanded", "false");
    await more.click();
    await expect(more).toHaveAttribute("aria-expanded", "true");
    await expect(first.locator(".qtable")).toBeVisible();
    expect(await overflow(page)).toEqual([]);
    expect(await axe(page)).toEqual([]);
    // A narrow window: the cards reflow and the tables scroll inside themselves.
    await page.setViewportSize({ width: 900, height: 700 });
    await settle(page).catch(() => undefined);
    expect(await overflow(page)).toEqual([]);
    // Nothing left this origin and nothing was fetched as a model file.
    expect(requests.filter((u) => /\.gguf|huggingface|download/i.test(u))).toEqual([]);
  });

  test("Tune hides raw flags in Beginner and shows them in Advanced", async ({ page }) => {
    await page.addInitScript(([k]) => window.localStorage.setItem(k, "beginner"), [EXPERIENCE_KEY] as const);
    await page.goto("/?mock=fast-tune&mock=tune-hold-screen#/tune");
    await expect(page.locator("main h1")).toHaveText("Tune");
    await page.locator('.pick-card[data-model="qwen3-coder-30b-a3b-q4km"]').click();
    await page.locator('.pick-card[data-profile="chat"]').click();
    await page.getByRole("button", { name: "Start tune" }).click();
    const row = page.locator('.cand[data-candidate="c1"]');
    await expect(row).toHaveAttribute("data-status", "advanced", { timeout: 15_000 });
    await row.locator(".cand__head").click();
    await expect(row.locator(".cand__detail")).toBeVisible();
    await expect(row.locator(".cand__detail")).not.toContainText("-ncmoe");
    await expect(row.locator(".cand__flags")).not.toContainText("-ncmoe");
  });
});

test("the onboarding record and the level are separate: a done record with no level opens in Advanced", async ({ page }) => {
  await page.addInitScript(([key, record]) => window.localStorage.setItem(key, record), [ONBOARDING_KEY, DONE_RECORD] as const);
  await openScreen(page, "settings");
  await expect(page.locator("section", { has: page.locator("#experience-title") }).getByRole("radio", { name: "Advanced" })).toBeChecked();
});
