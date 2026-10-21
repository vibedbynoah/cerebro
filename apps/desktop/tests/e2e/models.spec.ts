import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { MOCK_LIBRARY } from "../../ui/src/mock/library";
import type { LibraryModel } from "../../ui/src/service/types";
import { expect, openScreen, settle, test } from "./fixtures";

/** Brings one model's row to the top of the library (by searching its file name) and opens its drawer. */
async function openModel(page: Page, m: LibraryModel, query = ""): Promise<void> {
  if (query) {
    await page.goto(`/${query}#/models`);
    await expect(page.locator("main h1")).toBeVisible();
  } else await openScreen(page, "models");
  await page.getByRole("searchbox", { name: "Search models" }).fill(m.file);
  const row = page.locator(`.lib-row[data-model="${m.id}"]`);
  await row.locator(".lib-name__text").click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.locator(".drawer__title")).toHaveText(m.name);
}

const first = <T>(items: readonly T[], pred: (t: T) => boolean): T => {
  const found = items.find(pred);
  if (!found) throw new Error("fixture not found");
  return found;
};

const TUNED_MOE = first(MOCK_LIBRARY, (m) => m.id === "qwen3-coder-30b-a3b-q4km");
const UNTUNED = first(MOCK_LIBRARY, (m) => m.id === "qwen3-8b-q5km");
const VISION = first(MOCK_LIBRARY, (m) => m.kind === "vision" && m.projector !== null);
const VISION_NO_PROJECTOR = first(MOCK_LIBRARY, (m) => m.kind === "vision" && m.projector === null);
const STT = first(MOCK_LIBRARY, (m) => m.kind === "stt");
const TTS = first(MOCK_LIBRARY, (m) => m.kind === "tts");

test.describe("library", () => {
  test("lists 500+ models while keeping only the rows in view in the DOM", async ({ page }) => {
    expect(MOCK_LIBRARY.length).toBeGreaterThan(500);
    await openScreen(page, "models");
    const grid = page.getByRole("grid", { name: "Models on this machine" });
    await expect(grid).toHaveAttribute("aria-rowcount", String(MOCK_LIBRARY.length + 1));
    await expect(page.locator(".lib-count")).toHaveText(`${MOCK_LIBRARY.length} models`);
    const rendered = await page.locator(".lib-body > .lib-row").count();
    expect(rendered).toBeGreaterThan(5);
    expect(rendered).toBeLessThan(40);

    // Scroll to the very end: the last row is rendered and the DOM stays small.
    await grid.evaluate((el) => (el.scrollTop = el.scrollHeight));
    await expect(page.locator(`.lib-body > .lib-row[aria-rowindex="${MOCK_LIBRARY.length + 1}"]`)).toBeVisible();
    expect(await page.locator(".lib-body > .lib-row").count()).toBeLessThan(40);
  });

  test("search filters as you type, counts the result, and clears", async ({ page }) => {
    await openScreen(page, "models");
    const search = page.getByRole("searchbox", { name: "Search models" });
    await search.fill("gemma q8");
    await expect(page.locator(".lib-count")).toHaveText(/^\d+ of \d+ models$/);
    for (const name of await page.locator(".lib-body .lib-name__text").allTextContents()) expect(name.toLowerCase()).toContain("gemma");
    for (const q of await page.locator(".lib-body .lib-sub .mono").allTextContents()) expect(q).toBe("Q8_0");
    await page.getByRole("button", { name: "Clear filters" }).first().click();
    await expect(search).toHaveValue("");
    await expect(page.locator(".lib-count")).toHaveText(`${MOCK_LIBRARY.length} models`);
  });

  test("a search that is markup is shown as text and matches nothing", async ({ page }) => {
    await openScreen(page, "models");
    const payload = '<img src=x onerror="window.__xss=1">';
    await page.getByRole("searchbox", { name: "Search models" }).fill(payload);
    await expect(page.locator(".lib-state")).toContainText("No models match");
    await expect(page.locator(".lib-state .mono")).toHaveText(payload);
    expect(await page.locator(".lib-state img").count()).toBe(0);
    expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
  });

  test("modality and fit filters narrow the list and every row carries its badges", async ({ page }) => {
    await openScreen(page, "models");
    await page.locator(".lib-toolbar label", { hasText: "Vision" }).click();
    const rows = page.locator(".lib-body > .lib-row");
    await expect(rows.first()).toBeVisible();
    for (const row of await rows.all()) await expect(row.locator('.badge[data-modality="vision"]')).toHaveCount(1);
    await page.locator(".lib-toolbar label", { hasText: "Audio" }).click();
    await expect(rows.first().locator('.badge[data-modality="audio-in"], .badge[data-modality="audio-out"]')).toHaveCount(1);
    await page.getByRole("combobox", { name: "Memory fit" }).selectOption("vram");
    for (const chip of await page.locator(".lib-body .fit-chip").allTextContents()) expect(chip).toBe("Fits in graphics memory");
  });

  test("keyboard: arrows move, End reaches the last row, Enter opens details, Escape returns", async ({ page }) => {
    await openScreen(page, "models");
    const firstRow = page.locator('.lib-body > .lib-row[aria-rowindex="2"]');
    await firstRow.focus();
    await page.keyboard.press("ArrowDown");
    await expect(page.locator('.lib-body > .lib-row[aria-rowindex="3"]')).toBeFocused();
    await page.keyboard.press("End");
    const last = page.locator(`.lib-body > .lib-row[aria-rowindex="${MOCK_LIBRARY.length + 1}"]`);
    await expect(last).toBeFocused();
    await expect(last).toBeInViewport();
    await page.keyboard.press("Home");
    await expect(firstRow).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect(firstRow.locator(".lib-load")).toBeFocused();
    await page.keyboard.press("ArrowLeft");
    await expect(firstRow).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toBeHidden();
    await expect(firstRow).toBeFocused();
  });

  test("one-click Load from a row shows busy, then loaded, and the HUD follows", async ({ page }) => {
    await openScreen(page, "models");
    await page.getByRole("searchbox", { name: "Search models" }).fill(UNTUNED.file);
    const row = page.locator(`.lib-row[data-model="${UNTUNED.id}"]`);
    await row.getByRole("button", { name: `Load ${UNTUNED.name} ${UNTUNED.quant}` }).click();
    await expect(row.locator(".lib-loaded")).toBeVisible();
    await expect(page.locator(".toast", { hasText: `${UNTUNED.name} loaded` })).toBeVisible();
    await expect(page.locator(".loaded-item")).toContainText(UNTUNED.name);
    // The coding model was in the chat slot; it was replaced, not stacked.
    await expect(page.locator(".loaded-item")).not.toContainText(TUNED_MOE.name);
    await expect(page.locator(".statusbar [data-hud='model'] .hud-item__value")).toHaveText(UNTUNED.name, { timeout: 4000 });
    await row.getByRole("button", { name: `Unload ${UNTUNED.name} ${UNTUNED.quant}` }).click();
    await expect(page.locator(".loaded__none")).toHaveText("Nothing loaded");
  });
});

test.describe("states", () => {
  test("the empty library explains where models come from", async ({ page }) => {
    await page.goto("/?mock=empty-library#/models");
    await expect(page.locator(".lib-empty")).toContainText("No models on this machine yet");
    await expect(page.getByRole("link", { name: "Open Model Hub" })).toHaveAttribute("href", "#/hub");
    await expect(page.locator(".dup-list")).toHaveCount(0);
  });

  test("skeleton rows show while the folders are read", async ({ page }) => {
    await page.goto("/?mock=slow-library#/models");
    await expect(page.locator(".lib-grid--skeleton")).toBeVisible();
    await expect(page.locator(".lib-grid--skeleton")).toHaveAttribute("aria-hidden", "true");
    await expect(page.locator(".lib-count")).toHaveText("Reading model folders…");
  });

  test("a failed scan says so and offers a retry", async ({ page }) => {
    await page.goto("/?mock=fail-library#/models");
    await expect(page.getByRole("alert").filter({ hasText: "could not be read" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
  });
});

test.describe("drawer", () => {
  test("overview shows file facts, three tune numbers with their conditions beside quality, and the actions", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    const dialog = page.getByRole("dialog");
    await expect(dialog.locator(".facts")).toContainText(TUNED_MOE.sha256);
    const metrics = dialog.locator(".ov-metrics");
    await expect(metrics).toContainText("Not measured yet");
    await expect(metrics).toContainText("Prompt reading");
    await expect(metrics).toContainText("empty context");
    await expect(metrics).toContainText("Writing at 8K context");
    // Every condition rule zero names: machine, build, quant, depth, sampler (the model is the drawer).
    const r = TUNED_MOE.tuneResult;
    expect(r).not.toBeNull();
    const conditions = dialog.locator(".ov-conditions");
    for (const part of [r?.gpu ?? "", r?.build ?? "", TUNED_MOE.quant, "8,192-token depth", `sampler ${r?.sampler ?? ""}`]) await expect(conditions).toContainText(part);
    await expect(dialog).not.toContainText("× defaults");
    await expect(dialog.locator(".ov-tune .stat-tile__flag")).toBeVisible();
    await expect(dialog.getByRole("link", { name: "Tune" })).toHaveAttribute("href", `#/tune?model=${TUNED_MOE.id}`);
    await expect(dialog.getByRole("button", { name: "Chat" })).toBeVisible();
    await dialog.getByRole("button", { name: "Reveal in folder" }).click();
    await expect(page.locator(".toast", { hasText: "Reveal in folder works in the desktop app" })).toBeVisible();
    await expect(dialog.locator(".code-block")).toContainText(`cerebro tune ${TUNED_MOE.id}`);
  });

  test("Tune opens the Tune screen for this model; Chat loads it and opens Chat", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("dialog").getByRole("link", { name: "Tune" }).click();
    await expect(page).toHaveURL(new RegExp(`#/tune\\?model=${UNTUNED.id}$`));
    await expect(page.locator(`.pick-card[data-model="${UNTUNED.id}"] input`)).toBeChecked();
    await expect(page.locator(`.pick-card[data-model="${UNTUNED.id}"]`)).toContainText(UNTUNED.name);
    await expect(page.locator(".code-block").last()).toContainText(`cerebro tune ${UNTUNED.id}`);

    await openModel(page, UNTUNED);
    await page.getByRole("dialog").getByRole("button", { name: "Chat" }).click();
    await expect(page.locator(".toast", { hasText: `${UNTUNED.name} loaded` })).toBeVisible();
    await expect(page).toHaveURL(/#\/chat$/);
    await expect(page.locator(".statusbar [data-hud='model'] .hud-item__value")).toHaveText(UNTUNED.name, { timeout: 4000 });
  });

  test("a hand-made Tune link to an unknown model is ignored", async ({ page }) => {
    await page.goto("/#/tune?model=not-a-model");
    await expect(page.locator("main h1")).toBeVisible();
    await expect(page.locator('.pick-card[data-model="not-a-model"]')).toHaveCount(0);
    // The first model on offer is chosen instead, and the command line follows the choice.
    const checked = page.locator(".pick__grid--models .pick-card:has(input:checked)");
    await expect(checked).toHaveCount(1);
    const id = await checked.getAttribute("data-model");
    await expect(page.locator(".code-block").last()).toContainText(`cerebro tune ${id}`);
    expect(await page.locator(".code-block").last().textContent()).not.toContain("not-a-model");
  });

  test("tabs follow the arrow keys", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    const overview = page.getByRole("tab", { name: "Overview" });
    await overview.focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Load settings" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("End");
    await expect(page.getByRole("tab", { name: "Metadata" })).toBeFocused();
  });
});

test.describe("load panel", () => {
  test("every change is marked, moves the estimate, and resets to tuned", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    await page.getByRole("tab", { name: "Load settings" }).click();
    const vram = page.getByRole("meter", { name: "Estimated graphics memory use" });
    const before = Number(await vram.getAttribute("aria-valuenow"));
    await expect(page.locator(".drawer__caption")).toContainText("Tuned settings");

    const ctx = page.getByLabel("Context length", { exact: true });
    await ctx.fill("4096");
    await expect(page.locator(".lfield.is-changed")).toHaveCount(1);
    await expect(page.locator(".drawer__caption")).toContainText("Custom settings · 1 change");
    await expect.poll(async () => Number(await vram.getAttribute("aria-valuenow"))).not.toBe(before);

    // Fewer experts on the CPU puts more weight on the GPU.
    const ram = page.getByRole("meter", { name: "Estimated system memory use" });
    const ramBefore = Number(await ram.getAttribute("aria-valuenow"));
    await page.getByLabel("Experts on CPU", { exact: true }).fill("0");
    await expect.poll(async () => Number(await ram.getAttribute("aria-valuenow"))).toBeLessThan(ramBefore);
    await expect(page.locator(".fit-status")).toHaveText(/Does not fit|Tight/);

    await page.getByRole("button", { name: "Reset to tuned" }).click();
    await expect(page.locator(".lfield.is-changed")).toHaveCount(0);
    await expect(page.locator(".drawer__caption")).toContainText("Tuned settings");
    expect(Number(await vram.getAttribute("aria-valuenow"))).toBe(before);
  });

  test("an impossible combination is explained and Load waits for a fix", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    await page.getByRole("tab", { name: "Load settings" }).click();
    await page.getByLabel("V cache", { exact: true }).selectOption("q4_0");
    await page.locator(".lfield", { hasText: "Flash attention" }).locator("label", { hasText: "Off" }).click();
    await expect(page.locator(".field__error").first()).toContainText("needs flash attention");
    await expect(page.locator(".drawer__load-btn")).toBeDisabled();
    await expect(page.locator(".drawer__caption")).toContainText("Fix 1 setting to load");
    await page.getByLabel("Context length", { exact: true }).fill("10");
    await expect(page.locator(".drawer__caption")).toContainText("Fix 2 settings to load");
  });

  test("a loaded model offers Reload only while the panel differs from what is running", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    const btn = page.locator(".drawer__load-btn");
    await expect(btn).toHaveText(/Loaded · Unload/);
    await page.getByRole("tab", { name: "Load settings" }).click();
    await page.getByLabel("Context length", { exact: true }).fill("8192");
    await expect(btn).toHaveText(/Reload/);
    await expect(page.locator(".drawer__caption")).toContainText("reload to apply");
    await btn.click();
    await expect(page.locator(".toast").filter({ hasText: `${TUNED_MOE.name} loaded` })).toContainText("with your settings");
    await expect(btn).toHaveText(/Loaded · Unload/);
    await expect(page.locator(".loaded-item")).toContainText("custom");
  });

  test("an invalid field is marked invalid and described by its error, politely, with no axe violations", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    await page.getByRole("tab", { name: "Load settings" }).click();
    const ctx = page.getByLabel("Context length", { exact: true });
    await expect(ctx).not.toHaveAttribute("aria-invalid", "true");
    await ctx.fill("65536");
    await expect(ctx).toHaveAttribute("aria-invalid", "true");
    const ids = (await ctx.getAttribute("aria-describedby"))?.split(" ") ?? [];
    const errId = ids.find((i) => i.endsWith("-err"));
    expect(errId).toBeDefined();
    const err = page.locator(`[id="${errId}"]`);
    await expect(err).toContainText("Between 256 and");
    // Polite, never assertive: the message is in a live region, not role=alert.
    await expect(err).not.toHaveAttribute("role", "alert");
    await expect(err.locator("xpath=..")).toHaveAttribute("aria-live", "polite");
    await settle(page);
    const results = await new AxeBuilder({ page }).include(".drawer").withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]).analyze();
    expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`)).toEqual([]);
  });

  test("custom settings load, and a named preset is saved and chosen again", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Load settings" }).click();
    await expect(page.getByRole("button", { name: "Reset to starting" })).toBeDisabled();
    await page.getByLabel("Threads", { exact: true }).fill("4");
    await page.getByRole("button", { name: "Save as preset" }).click();
    await page.getByLabel("Preset name").fill("Quiet fans");
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(page.locator(".toast", { hasText: "Preset saved: Quiet fans" })).toBeVisible();
    const picker = page.locator(".preset-bar select");
    await expect(picker.locator("option:checked")).toHaveText("Quiet fans");
    await picker.selectOption({ label: "Starting settings" });
    await expect(page.getByLabel("Threads", { exact: true })).not.toHaveValue("4");
    await picker.selectOption({ label: "Quiet fans" });
    await expect(page.getByLabel("Threads", { exact: true })).toHaveValue("4");
    await page.locator(".drawer__load-btn").click();
    await expect(page.locator(".toast").filter({ hasText: `${UNTUNED.name} loaded` })).toContainText("with your settings");
    await expect(page.locator(".loaded-item")).toContainText("custom");
  });
});

test.describe("hardware not detected", () => {
  test("nothing is estimated for an invented machine: fit is Unknown and the panel says so", async ({ page }) => {
    await page.goto("/?mock=fail-hw#/models");
    await expect(page.locator("main h1")).toBeVisible();
    await expect(page.locator(".lib-body > .lib-row").first()).toBeVisible();
    for (const chip of await page.locator(".lib-body .fit-chip").allTextContents()) expect(chip.trim()).toBe("Unknown");
    await expect(page.locator(".fit-chip", { hasText: "Fits in graphics memory" })).toHaveCount(0);

    await page.getByRole("searchbox", { name: "Search models" }).fill(UNTUNED.file);
    await page.locator(`.lib-row[data-model="${UNTUNED.id}"] .lib-name__text`).click();
    await expect(page.locator(".drawer__caption")).toContainText("CPU only");
    await page.getByRole("tab", { name: "Load settings" }).click();
    await expect(page.locator(".fit-unknown")).toContainText("This machine was not detected");
    await expect(page.getByRole("meter")).toHaveCount(0);
    await expect(page.getByLabel("GPU layers", { exact: true })).toHaveValue("0");
    await expect(page.locator(".lfield", { hasText: "Threads" })).toContainText("Processor not detected");
    await expect(page.getByRole("button", { name: "Detect again" })).toBeVisible();
  });

  test("Detect again recovers when the second detection works", async ({ page }) => {
    await page.goto("/?mock=fail-hw-once#/models");
    await expect(page.locator(".lib-body > .lib-row").first()).toBeVisible();
    await page.getByRole("searchbox", { name: "Search models" }).fill(UNTUNED.file);
    await page.locator(`.lib-row[data-model="${UNTUNED.id}"] .lib-name__text`).click();
    await page.getByRole("tab", { name: "Load settings" }).click();
    await page.getByRole("button", { name: "Detect again" }).click();
    await expect(page.getByRole("meter", { name: "Estimated graphics memory use" })).toBeVisible();
    await expect(page.locator(".fit-unknown")).toHaveCount(0);
  });
});

test.describe("tweaks and load settings agree", () => {
  test("a KV tweak shows in the caption, shrinks the estimate, and is what Load sends", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Load settings" }).click();
    const vram = page.getByRole("meter", { name: "Estimated graphics memory use" });
    const kvOf = async () => Number(/KV cache ([\d.]+)/.exec((await vram.getAttribute("aria-valuetext")) ?? "")?.[1] ?? Number.NaN);
    const kvBefore = await kvOf();
    expect(kvBefore).toBeGreaterThan(0);
    await expect(page.locator(".drawer__caption")).toContainText("Starting settings");

    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await page.getByLabel("KV cache type").selectOption("q8_0");
    await page.getByRole("tab", { name: "Load settings" }).click();
    await expect(page.locator(".drawer__caption")).toContainText("+ 1 tweak · not measured with these");
    await expect.poll(kvOf).toBeLessThan(kvBefore);
    // The K and V fields show the tweak's value and say where it comes from.
    await expect(page.getByLabel("V cache", { exact: true })).toHaveValue("q8_0");
    await expect(page.getByLabel("V cache", { exact: true })).toBeDisabled();
    await expect(page.locator(".lfield").filter({ has: page.locator("label", { hasText: /^V cache$/ }) })).toContainText("Set by Tweak Studio");

    // Flash attention off with the tweak's quantized V cache is caught on the combined settings.
    await page.locator(".lfield", { hasText: "Flash attention" }).locator("label", { hasText: "Off" }).click();
    await expect(page.locator(".drawer__caption")).toContainText("Fix 1 setting to load");
    await page.locator(".lfield", { hasText: "Flash attention" }).locator("label", { hasText: "Auto" }).click();

    await page.locator(".drawer__load-btn").click();
    await expect(page.locator(".toast").filter({ hasText: `${UNTUNED.name} loaded` })).toContainText("with your settings");
    await expect(page.locator(".loaded-item")).toContainText("custom");
  });

  test("a sampler value out of range is shown as an error and never saved", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await page.getByRole("button", { name: "Set a default sampler" }).click();
    const temp = page.getByLabel("Temperature", { exact: true });
    const before = await temp.inputValue();
    await temp.fill("5");
    await expect(temp).toHaveAttribute("aria-invalid", "true");
    await expect(page.locator(".sampler__field", { hasText: "Temperature" }).locator(".field__error")).toHaveText("0 to 2");
    await temp.blur();
    await expect(temp).toHaveValue(before);
    await expect(temp).not.toHaveAttribute("aria-invalid", "true");
    // Close and reopen: the stored patch still holds the earlier value.
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toBeHidden();
    await page.locator(`.lib-row[data-model="${UNTUNED.id}"] .lib-name__text`).click();
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await expect(page.getByLabel("Temperature", { exact: true })).toHaveValue(before);
  });
});

test.describe("vision and audio", () => {
  test("a vision model shows its paired projector and reads a dropped image", async ({ page }) => {
    await openModel(page, VISION);
    const dialog = page.getByRole("dialog");
    await expect(dialog.locator(".projector")).toContainText(VISION.projector?.file ?? "");
    await expect(dialog.locator(".badge[data-modality='vision']")).toBeVisible();
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64");
    await dialog.locator(".dropzone input[type=file]").setInputFiles({ name: "photo.png", mimeType: "image/png", buffer: png });
    await expect(dialog.locator(".dropzone__img")).toHaveAttribute("src", /^data:image\/png;base64,/);
    await expect(dialog.locator(".try-output__text")).toContainText("stayed on this computer");
    await dialog.locator(".dropzone input[type=file]").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hi") });
    await expect(dialog.locator(".field__error")).toContainText("PNG, JPEG, WebP or GIF");
  });

  test("a second image dropped while the first is read gets only its own description", async ({ page }) => {
    await openModel(page, VISION);
    const dialog = page.getByRole("dialog");
    // Record every text the output ever shows, so a brief stale description is caught too.
    await page.evaluate(() => {
      const seen: string[] = [];
      (window as unknown as { __seen: string[] }).__seen = seen;
      new MutationObserver(() => {
        for (const el of Array.from(document.querySelectorAll(".try-output__text"))) seen.push(el.textContent ?? "");
      }).observe(document.body, { subtree: true, childList: true, characterData: true });
    });
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64");
    const input = dialog.locator(".dropzone input[type=file]");
    await input.setInputFiles({ name: "first.png", mimeType: "image/png", buffer: png });
    await page.waitForTimeout(300);
    await input.setInputFiles({ name: "second.png", mimeType: "image/png", buffer: png });
    await expect(dialog.locator(".try-output__text")).toContainText("second.png");
    await page.waitForTimeout(400);
    await expect(dialog.locator(".try-output__text")).toContainText("second.png");
    const seen = await page.evaluate(() => (window as unknown as { __seen: string[] }).__seen);
    expect(seen.some((t) => t.includes("first.png"))).toBe(false);
  });

  test("a vision model without a projector says what is missing", async ({ page }) => {
    await openModel(page, VISION_NO_PROJECTOR);
    await expect(page.locator(".projector--missing")).toContainText("Text chat works");
    await expect(page.locator(".dropzone")).toHaveCount(0);
  });

  test("speech to text records and transcribes in its own engine", async ({ page }) => {
    await openModel(page, STT);
    await expect(page.getByRole("tab")).toHaveCount(1);
    await page.getByRole("button", { name: "Record" }).click();
    await page.getByRole("button", { name: /Stop/ }).click();
    await expect(page.locator(".try-output__text")).toContainText("sample transcript");
    await page.locator(".drawer__load-btn").click();
    await expect(page.locator(".loaded-item[data-slot='stt']")).toBeVisible();
    // The chat model is still loaded beside it.
    await expect(page.locator(".loaded-item[data-slot='llm']")).toBeVisible();
  });

  test("text to speech speaks a sentence and validates it", async ({ page }) => {
    await openModel(page, TTS);
    const speak = page.getByRole("button", { name: "Speak" });
    await page.getByLabel("Sentence").fill("");
    await expect(speak).toBeDisabled();
    await page.getByLabel("Sentence").fill("Hello from this machine.");
    await speak.click();
    await expect(page.locator(".playback")).toContainText("s");
  });
});

test.describe("Tweak Studio", () => {
  test("fixes switch, tweaks diff, and a profile saves, copies and re-imports", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    const summary = page.locator(".studio__count");
    await expect(summary).toHaveText("0");

    await page.getByRole("switch", { name: "Stop at <|im_end|>" }).click({ force: true });
    await expect(summary).toHaveText("1");

    await page.getByLabel("KV cache type").selectOption("q8_0");
    await page.getByLabel("New stop token").fill("</answer>");
    await page.getByRole("button", { name: "Add", exact: true }).click();
    await expect(page.getByRole("list", { name: "Stop tokens" })).toContainText("</answer>");
    await page.getByRole("button", { name: /Use the authors' values/ }).click();
    await expect(summary).toHaveText("5");

    const template = page.getByRole("textbox", { name: "Chat template" });
    await template.fill(`${await template.inputValue()}\n{{- '<|im_end|>' }}`);
    await page.getByRole("button", { name: /^Diff/ }).click();
    await expect(page.locator(".diff__line[data-kind='add']")).toHaveCount(1);
    await expect(page.locator(".template__status")).toContainText("Structure looks right");
    await page.getByRole("button", { name: "Edit", exact: true }).click();
    await template.fill("{% if x %}broken");
    await expect(page.locator(".template__status")).toContainText("{% if %} is never closed");
    await template.fill("{{ broken");
    await expect(page.locator(".template__status")).toContainText("Unbalanced");

    await page.getByLabel("Profile name").fill("Long docs");
    await page.getByRole("button", { name: "Save profile" }).click();
    await expect(page.locator(".profile")).toContainText("Long docs");

    await page.getByRole("button", { name: "Revert all" }).click();
    await expect(summary).toHaveText("0");
    await page.locator(".profile").getByRole("button", { name: "Apply" }).click();
    await expect(summary).toHaveText("6");
  });

  test("tweaks survive switching tabs and closing the drawer, and Revert all clears them", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await page.getByLabel("KV cache type").selectOption("q8_0");
    await expect(page.locator(".studio__count")).toHaveText("1");
    await page.getByRole("tab", { name: "Load settings" }).click();
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await expect(page.getByLabel("KV cache type")).toHaveValue("q8_0");
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toBeHidden();
    await page.locator(`.lib-row[data-model="${UNTUNED.id}"] .lib-name__text`).click();
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await expect(page.locator(".studio__count")).toHaveText("1");
    await page.getByRole("button", { name: "Revert all" }).click();
    await expect(page.locator(".studio__count")).toHaveText("0");
  });

  test("imported profiles are validated and cannot pollute the page", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Tweak Studio" }).click();
    await page.getByRole("button", { name: "Import" }).click();
    const box = page.getByLabel("Profile text to import");
    await box.fill("not json");
    await page.getByRole("button", { name: "Import profile" }).click();
    await expect(page.getByRole("alert")).toContainText("not valid JSON");

    // A raw "__proto__" key, as an attacker would paste it (an object literal would not serialize it).
    const hostile = `{"format":"cerebro-patch-profile","version":1,"name":"<img src=x onerror=alert(1)>","modelSha256":"${UNTUNED.sha256}","__proto__":{"polluted":true},"tweaks":{"kvQuant":null,"contextOverride":null,"rope":null,"chatTemplate":null,"stopTokens":[],"sampler":null,"__proto__":{"polluted":true},"extra":1},"fixes":[]}`;
    await box.fill(hostile);
    await page.getByRole("button", { name: "Import profile" }).click();
    await expect(page.locator(".profile")).toContainText("<img src=x onerror=alert(1)>");
    expect(await page.locator(".profile img").count()).toBe(0);
    expect(await page.evaluate(() => ({} as Record<string, unknown>)["polluted"])).toBeUndefined();

    // A successful import closes the box; open it again for the next one.
    await expect(box).toBeHidden();
    await page.getByRole("button", { name: "Import" }).click();
    await box.fill(JSON.stringify({ format: "cerebro-patch-profile", version: 1, name: "Other file", modelSha256: "a".repeat(64), tweaks: { kvQuant: "q8_0", contextOverride: null, rope: null, chatTemplate: null, stopTokens: [], sampler: null }, fixes: [] }));
    await page.getByRole("button", { name: "Import profile" }).click();
    await expect(page.locator(".notice--inline")).toContainText("made for a different file");
    await page.getByRole("button", { name: "Use it for this file anyway" }).click();
    await expect(page.locator(".profile", { hasText: "Other file" })).toContainText("1 tweaks");
  });
});

test.describe("GGUF metadata, copies", () => {
  test("a second write adds to the copy; the table shows both as in the copy; Revert removes both", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Metadata" }).click();
    // Grouped by namespace, with the action on each row.
    await expect(page.locator(".meta-group__title", { hasText: "General" })).toBeVisible();
    await expect(page.locator(".meta-group__title", { hasText: "Tokenizer" })).toBeVisible();

    const ctxRow = page.locator(".meta-row", { hasText: "qwen3.context_length" });
    await ctxRow.getByRole("button", { name: "Edit qwen3.context_length" }).click();
    await page.getByLabel("New value for qwen3.context_length").fill("40960");
    await page.keyboard.press("Enter");
    await expect(page.locator(".meta-diff__item")).toHaveCount(1);
    await expect(page.locator(".meta-diff .hint")).toContainText(`${UNTUNED.sizeGiB.toFixed(1)} GiB free on C:`);
    await page.getByRole("button", { name: "Write to a copy" }).click();
    await expect(page.locator(".meta__copy")).toContainText("(1 change)");

    const nameRow = page.locator(".meta-row", { hasText: "general.name" });
    await nameRow.getByRole("button", { name: "Edit general.name" }).click();
    await page.getByLabel("New value for general.name").fill("Qwen3-8B renamed");
    await page.keyboard.press("Enter");
    // The diff shows everything the copy will hold: the earlier change and the new one.
    await expect(page.locator(".meta-diff__item")).toHaveCount(2);
    await expect(page.locator(".meta-diff__item.is-new")).toHaveCount(1);
    await page.getByRole("button", { name: "Update the copy" }).click();
    await expect(page.locator(".meta__copy")).toContainText("(2 changes)");
    await expect(ctxRow).toContainText("40960");
    await expect(ctxRow.locator(".meta-row__mark")).toHaveText("changed in copy");
    await expect(nameRow).toContainText("Qwen3-8B renamed");
    await expect(nameRow.locator(".meta-row__mark")).toHaveText("changed in copy");

    await page.locator(".meta__copy").getByRole("button", { name: "Revert" }).click();
    await expect(page.locator(".meta__copy")).toHaveCount(0);
    await expect(page.locator(".meta-row__mark")).toHaveCount(0);
    await expect(ctxRow).toContainText(String(UNTUNED.contextMax));
  });
});

test.describe("GGUF metadata", () => {
  test("edits are type-checked, shown as a diff, written to a copy and reverted", async ({ page }) => {
    await openModel(page, UNTUNED);
    await page.getByRole("tab", { name: "Metadata" }).click();
    const blockCount = page.locator(".meta-row", { hasText: "qwen3.block_count" });
    await expect(blockCount.locator(".meta-lock")).toBeVisible();
    await expect(blockCount.getByRole("button")).toHaveCount(0);

    const ctxRow = page.locator(".meta-row", { hasText: "qwen3.context_length" });
    await ctxRow.getByRole("button", { name: "Edit qwen3.context_length" }).click();
    const input = page.getByLabel("New value for qwen3.context_length");
    await input.fill("forty thousand");
    await page.keyboard.press("Enter");
    await expect(ctxRow.locator(".field__error")).toHaveText("A whole number.");
    await input.fill("40960");
    await page.keyboard.press("Enter");
    await expect(page.locator(".meta-diff__item")).toContainText("40960");
    await page.getByRole("button", { name: "Write to a copy" }).click();
    await expect(page.locator(".meta__copy")).toContainText(".edited.gguf");
    await expect(page.locator(".toast", { hasText: "The original file is unchanged" })).toBeVisible();
    await page.locator(".meta__copy").getByRole("button", { name: "Revert" }).click();
    await expect(page.locator(".meta__copy")).toHaveCount(0);
  });
});

test.describe("disk space", () => {
  test("deduplicate asks once, links same-drive copies, and leaves other drives alone", async ({ page }) => {
    await openScreen(page, "models");
    const llama = page.locator(".dup", { hasText: "Llama-3.1-8B-Instruct-Q8_0.gguf" });
    await expect(llama.getByRole("button", { name: "Deduplicate" })).toBeDisabled();
    await expect(llama).toContainText("Drive D: stays a copy");

    const coder = page.locator(".dup", { hasText: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" });
    await expect(coder).toContainText("Drive D: stays a copy");
    await coder.getByRole("button", { name: "Deduplicate" }).click();
    await expect(coder.locator(".dup__confirm")).toContainText("Link 2 copies to Cerebro's copy?");
    await coder.getByRole("button", { name: "Link" }).click();
    await expect(coder.locator(".dup__done")).toContainText("34.6 GiB freed");
    await expect(page.locator(".toast", { hasText: "Freed 34.6 GiB" })).toContainText("1 copy on another drive stays separate");
  });

  test("the copy to keep can be chosen, and Undo turns the links back into copies", async ({ page }) => {
    await openScreen(page, "models");
    const qwen = page.locator(".dup", { hasText: "Qwen3-8B-Q5_K_M.gguf" });
    await expect(qwen.getByRole("radio", { name: /Keep this copy: Cerebro — / })).toBeChecked();
    await qwen.getByRole("radio", { name: /Keep this copy: Another app's folder — .*\.lmstudio/ }).check();
    await qwen.getByRole("button", { name: "Deduplicate" }).click();
    await expect(qwen.locator(".dup__confirm")).toContainText("to the copy in another app's folder?");
    await qwen.getByRole("button", { name: "Link" }).click();
    await expect(qwen.locator(".dup__done")).toBeVisible();
    await expect(qwen.locator(".dup-copy.is-kept")).toContainText("Another app's folder");
    await qwen.getByRole("button", { name: "Undo" }).click();
    await expect(qwen.locator(".dup__confirm")).toContainText("GiB free on C:");
    await qwen.getByRole("button", { name: "Restore copies" }).click();
    await expect(page.locator(".toast", { hasText: "Copies restored" })).toBeVisible();
    await expect(qwen.getByRole("button", { name: "Deduplicate" })).toBeEnabled();
  });

  test("pressing Enter twice on Link deduplicates once, with one success and no error", async ({ page }) => {
    await openScreen(page, "models");
    const gemma = page.locator(".dup", { hasText: "gemma-3-12b-it-Q4_K_M.gguf" });
    await gemma.getByRole("button", { name: "Deduplicate" }).click();
    const link = gemma.getByRole("button", { name: "Link" });
    await expect(link).toBeFocused();
    await page.keyboard.press("Enter");
    await page.keyboard.press("Enter");
    await expect(gemma.locator(".dup__done")).toBeVisible();
    await page.waitForTimeout(900);
    await expect(page.locator(".toast", { hasText: "Freed" })).toHaveCount(1);
    await expect(page.locator(".toast", { hasText: "Nothing was deduplicated" })).toHaveCount(0);
  });
});

test.describe("motion", () => {
  test.use({ contextOptions: { reducedMotion: "reduce" } });
  test("with reduced motion the drawer opens without animating", async ({ page }) => {
    await openModel(page, TUNED_MOE);
    const anim = await page.locator(".drawer").evaluate((el) => getComputedStyle(el).animationName);
    expect(anim).toBe("none");
    await page.getByRole("tab", { name: "Load settings" }).click();
    // The memory bars jump to their new size instead of sliding.
    const seg = await page.locator(".meter__seg").first().evaluate((el) => getComputedStyle(el).transitionProperty);
    expect(seg).not.toContain("transform");
    await settle(page);
  });
});
