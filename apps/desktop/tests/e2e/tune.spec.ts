import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, settle, test } from "./fixtures";

const MOE = { id: "qwen3-coder-30b-a3b-q4km", name: "Qwen3-Coder-30B-A3B-Instruct" };
const SMALL = { id: "gemma-4-e4b-q4km", name: "Gemma-4-E4B-it" };
const DENSE = { id: "qwen3-8b-q5km", name: "Qwen3-8B" };
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

async function axe(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

/** Horizontal overflow anywhere in the window: the document, the main column, or an element of the
 *  screen poking past the main column's right edge. */
async function overflow(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const bad: string[] = [];
    const doc = document.documentElement;
    if (doc.scrollWidth > window.innerWidth) bad.push("document scrolls sideways");
    const main = document.querySelector("main");
    if (main && main.scrollWidth > main.clientWidth + 1) bad.push(`main overflows by ${main.scrollWidth - main.clientWidth}px`);
    const edge = main ? main.getBoundingClientRect().right : window.innerWidth;
    for (const el of Array.from(document.querySelectorAll("main *"))) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.right > edge + 1) bad.push(`${el.tagName.toLowerCase()}.${String(el.getAttribute("class"))} ends at ${Math.round(r.right)}`);
    }
    return bad.slice(0, 5);
  });
}

/** The Tune screen's own status line (polite live region). */
const statusLine = (page: Page) => page.locator(".screen--tune > [role='status']");

/** Opens the Tune screen with the given `?mock=` switches, picks a model and profile, and starts. */
async function startTune(page: Page, mocks: string[], model: string, profile: "chat" | "coding" | "long-docs"): Promise<void> {
  const query = mocks.map((m) => `mock=${m}`).join("&");
  await page.goto(`/${query ? `?${query}` : ""}#/tune`);
  await expect(page.locator("main h1")).toHaveText("Speed tune");
  await page.locator(`.pick-card[data-model="${model}"]`).click();
  await page.locator(`.pick-card[data-profile="${profile}"]`).click();
  await page.getByRole("button", { name: "Start tune" }).click();
  await expect(page.locator(".run")).toBeVisible();
}

test.describe("before a run", () => {
  test("explains the stages, offers models and profiles, and shows the matching command", async ({ page }) => {
    await page.goto("/#/tune");
    await expect(page.getByRole("heading", { name: "Start a tune" })).toBeVisible();
    await expect(page.locator(".phases__list .phase")).toHaveCount(6);
    await expect(page.locator(".pick__grid--profiles .pick-card")).toHaveCount(3);
    await page.locator(`.pick-card[data-model="${SMALL.id}"]`).click();
    await expect(page.locator(".code-block").last()).toContainText(`cerebro tune ${SMALL.id}`);
    await page.locator('.pick-card[data-profile="long-docs"]').click();
    await expect(page.locator(".objective")).toContainText("32K");
    // The choice is remembered for next time.
    await page.reload();
    await expect(page.locator(`.pick-card[data-model="${SMALL.id}"] input`)).toBeChecked();
    await expect(page.locator('.pick-card[data-profile="long-docs"] input')).toBeChecked();
  });

  test("a tampered saved choice is ignored, never rendered", async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("cerebro-tune-setup", JSON.stringify({ model: '<img src=x onerror="window.__pwned=1">', profile: "__proto__" }));
    });
    await page.goto("/#/tune");
    await expect(page.locator(".pick__grid--models .pick-card:has(input:checked)")).toHaveCount(1);
    await expect(page.locator('.pick-card[data-profile="chat"] input')).toBeChecked();
    expect(await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned)).toBeUndefined();
    await expect(page.locator("main img")).toHaveCount(0);
  });

  test("the model cards estimate memory at the workload's depth, and the stages assemble on scroll", async ({ page }) => {
    await page.goto("/#/tune");
    const card = page.locator(`.pick-card[data-model="${DENSE.id}"]`);
    await page.locator('.pick-card[data-profile="chat"]').click();
    await expect(card.locator(".pick-card__fit-text")).toHaveAttribute("data-fits", "true");
    await page.locator('.pick-card[data-profile="long-docs"]').click();
    await expect(card.locator(".pick-card__fit-text")).toHaveAttribute("data-fits", "false");
    await expect(card.locator(".pick-card__fit-text")).toContainText("32K");
    // The card labels and the objective define their terms, outside the radio labels.
    await expect(page.locator(".pick__terms .term")).not.toHaveCount(0);
    await expect(page.locator(".pick-card .term")).toHaveCount(0);
    await page.locator('.pick-card[data-profile="coding"]').click();
    await expect(page.locator(".objective .term", { hasText: "n-gram speculation" })).toHaveCount(1);
    // The idle stages reveal, and the rail draws, once scrolled into view.
    await page.locator("#how-title").scrollIntoViewIfNeeded();
    await expect(page.locator(".phases--idle .phase.is-visible")).toHaveCount(6);
    await expect(page.locator(".phases--idle .phases__draw")).toHaveClass(/is-visible/);
  });

  test("the profile and model pickers work from the keyboard", async ({ page }) => {
    await page.goto("/#/tune");
    const chat = page.locator('.pick-card[data-profile="chat"] input');
    await chat.focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.locator('.pick-card[data-profile="coding"] input')).toBeChecked();
    await expect(page.locator(".objective")).toContainText("16K");
  });
});

test.describe("a live run", () => {
  test("candidates appear, the chart traces them, and the start button waits", async ({ page }) => {
    await startTune(page, ["fast-tune", "tune-hold-screen"], MOE.id, "chat");
    await expect(page.locator('.cand[data-candidate="c4"][data-status="screening"]')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".cands .cand")).toHaveCount(9);
    expect(await page.locator(".trace").count()).toBeGreaterThanOrEqual(4);
    await expect(page.locator(".trace--active")).toHaveCount(1);
    await expect(page.locator(".trace--baseline")).toHaveCount(1);
    await expect(page.locator(".chart__ref")).toContainText("Automatic settings");
    await expect(page.locator(".run__caption")).toContainText("Illustrative");
    await expect(page.locator(".chip--live")).toContainText("Measuring");
    await expect(page.getByRole("button", { name: "Start tune" })).toBeDisabled();
    await expect(page.locator('.cand[data-candidate="c3"]')).toContainText("Did not load");
    // The chart's text alternative names the candidate in flight.
    await expect(page.locator(".tune-card--speed svg")).toHaveAttribute("aria-label", /Experts|KV|Plus|being screened/);
    // The stage in progress is marked for assistive tech.
    await expect(page.locator('.phase[aria-current="step"]')).toContainText("Screen");
    await expect(page.getByRole("progressbar", { name: "Tune progress" })).toHaveAttribute("aria-valuenow", /^[1-9]\d?$/);
  });

  test("a candidate opens to show its settings and flags", async ({ page }) => {
    await startTune(page, ["fast-tune", "tune-hold-screen"], MOE.id, "chat");
    const row = page.locator('.cand[data-candidate="c1"]');
    await expect(row).toHaveAttribute("data-status", "advanced", { timeout: 15_000 });
    const head = row.locator(".cand__head");
    await head.focus();
    await page.keyboard.press("Enter");
    await expect(head).toHaveAttribute("aria-expanded", "true");
    await expect(row.locator(".cand__detail")).toBeVisible();
    await expect(row.locator(".cand__detail")).toContainText("-ncmoe");
    // Every setting name carries its plain-words definition.
    const names = row.locator(".cand__settings dt");
    expect(await names.count()).toBeGreaterThan(0);
    await expect(row.locator(".cand__settings dt:not(:has(.term))")).toHaveCount(0);
    // The row's name is its label and state; raw flags are not read out on every row.
    const name = await head.evaluate((el) => {
      const ids = (el.getAttribute("aria-labelledby") ?? "").split(" ");
      return ids.map((id) => document.getElementById(id)?.textContent ?? "").join(" ");
    });
    expect(name).toContain("Experts split");
    expect(name).not.toContain("-ngl");
    await page.keyboard.press("Enter");
    await expect(row.locator(".cand__detail")).toBeHidden();
  });

  test("the soak shows its VRAM trend and the quality gate its results", async ({ page }) => {
    await startTune(page, ["fast-tune", "tune-hold-soak"], MOE.id, "chat");
    await expect(page.locator(".soak-item")).toHaveCount(2, { timeout: 20_000 });
    await expect(page.locator('.soak-item[data-state="failed"]')).toContainText("kept falling");
    await expect(page.locator('.soak-item[data-state="running"]')).toBeVisible();
    await expect(page.locator(".soak-trace__trend").first()).toBeVisible();
    await expect(page.locator(".gate")).not.toHaveCount(0);
    await expect(page.locator(".gate").first()).toContainText("KL divergence");
    await expect(page.getByRole("progressbar", { name: "Soak progress" })).toBeVisible();
  });

  test("finishes with the winner locked in and why it beat the automatic settings", async ({ page }) => {
    await startTune(page, ["fast-tune"], MOE.id, "coding");
    const winner = page.locator(".winner");
    await expect(winner).toBeVisible({ timeout: 25_000 });
    await expect(winner).toHaveAttribute("data-kept", "tuned");
    await expect(winner.locator(".winner__eyebrow")).toContainText("Locked in");
    await expect(winner.locator(".why li").first()).toContainText("automatic placement put");
    await expect(winner.locator(".versus__metric")).toHaveCount(3);
    await expect(winner.locator(".winner__caption")).toContainText("Illustrative");
    await expect(winner.locator(".winner__conditions")).toContainText("depth 16K");
    await expect(winner.locator(".winner__conditions")).toContainText("temperature 0");
    // The machine includes the processor and memory; the build names its exact commit.
    await expect(winner.locator(".winner__conditions")).toContainText("GiB RAM");
    await expect(winner.locator(".winner__conditions")).toContainText("cores");
    await expect(winner.locator(".winner__conditions")).toContainText(/Cerebro Engine \d+\.\d+\.\d+ \([0-9a-f]{7,}\)/);
    await expect(page.locator(".run__provenance")).toContainText("GiB RAM");
    // One hero figure, flagged as illustrative right beside it, then the reasons with their terms.
    await expect(winner.locator(".winner__hero .stat-tile__flag")).toBeVisible();
    await expect(winner.locator(".winner__hero-text")).toContainText("×");
    await expect(winner.locator(".why .term")).not.toHaveCount(0);
    await expect(winner.locator(".winner__settings dt:not(:has(.term))")).toHaveCount(0);
    await expect(winner.locator(".code-block")).toContainText(`cerebro serve ${MOE.id}`);
    await expect(page.locator(".run")).toHaveAttribute("data-status", "done");
    await expect(page.locator(".cand--win")).toHaveCount(1);
    await expect(page.getByRole("status").filter({ hasText: "Tune finished" })).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Start tune" })).toBeEnabled();
  });

  test("Open in Models shows the model with the numbers the result just locked in", async ({ page }) => {
    await startTune(page, ["fast-tune"], DENSE.id, "coding");
    const winner = page.locator(".winner");
    await expect(winner).toBeVisible({ timeout: 25_000 });
    const atDepth = await winner.locator(".versus__metric").nth(2).locator(".versus__row--win .versus__value").innerText();
    const value = /[\d.]+/.exec(atDepth.replaceAll(",", ""))?.[0] ?? "";
    expect(value).not.toBe("");
    await winner.getByRole("link", { name: /Open in Models/ }).click();
    await expect(page.locator("main h1")).toHaveText("Models");
    const drawer = page.locator(".drawer");
    await expect(drawer).toBeVisible();
    await expect(drawer).toContainText(DENSE.name);
    await expect(drawer.locator(".ov-metrics__win .count")).toHaveAttribute("data-count-to", value);
    await expect(drawer.locator(".ov-conditions")).toContainText("16,384-token depth");
    // Its state agrees: no longer untuned.
    await expect(drawer).not.toContainText("Not tuned yet");
    await page.keyboard.press("Escape");
    await expect(drawer).toBeHidden();
    // Closing the drawer drops the link, so a reload does not reopen it.
    expect(new URL(page.url()).hash).toBe("#/models");
  });

  test("a small model that fits keeps the automatic settings, and says so", async ({ page }) => {
    await startTune(page, ["fast-tune"], SMALL.id, "chat");
    const winner = page.locator(".winner");
    await expect(winner).toBeVisible({ timeout: 25_000 });
    await expect(winner).toHaveAttribute("data-kept", "defaults");
    await expect(winner.locator(".winner__title")).toHaveText("Defaults were fastest");
    await expect(winner.locator(".why")).toContainText("keeps the engine's automatic settings");
  });
});

test.describe("cancel, resume and leaving", () => {
  test("cancel keeps what finished; resume carries on", async ({ page }) => {
    await startTune(page, ["fast-tune", "tune-hold-soak"], DENSE.id, "long-docs");
    await expect(page.locator(".soak-item").first()).toBeVisible({ timeout: 20_000 });
    const measured = await page.locator(".cand .num-unit").count();
    await page.getByRole("button", { name: "Cancel tune" }).click();
    await expect(page.locator(".run")).toHaveAttribute("data-status", "cancelled");
    await expect(page.locator(".soak-item").last()).toHaveAttribute("data-state", "interrupted");
    await expect(page.locator('.phase[data-state="stopped"]')).toHaveCount(1);
    expect(await page.locator(".cand .num-unit").count()).toBe(measured);
    await expect(page.getByRole("button", { name: "Start tune" })).toBeEnabled();
    const resume = page.getByRole("button", { name: "Resume" });
    await expect(resume).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.locator(".run")).toHaveAttribute("data-status", "running");
    await expect(page.locator(".soak-item").last()).toHaveAttribute("data-state", "running", { timeout: 15_000 });
    await expect(page.getByRole("button", { name: "Cancel tune" })).toBeFocused();
  });

  test("leaving the screen keeps the tune running; coming back replays it", async ({ page }) => {
    await startTune(page, ["fast-tune", "tune-hold-screen"], MOE.id, "chat");
    await expect(page.locator('.cand[data-candidate="c4"][data-status="screening"]')).toBeVisible({ timeout: 15_000 });
    const before = await page.locator(".cand").evaluateAll((rows) => rows.map((r) => `${r.getAttribute("data-candidate")}:${r.getAttribute("data-status")}`));
    await page.getByRole("link", { name: "Models" }).first().click();
    await expect(page.locator("main h1")).toHaveText("Models");
    await page.getByRole("link", { name: "Tune" }).first().click();
    await expect(page.locator(".run")).toBeVisible();
    const after = await page.locator(".cand").evaluateAll((rows) => rows.map((r) => `${r.getAttribute("data-candidate")}:${r.getAttribute("data-status")}`));
    expect(after).toEqual(before);
    // Replayed rows appear at once; only rows that arrive live animate in.
    await expect(page.locator(".cand--enter")).toHaveCount(0);
  });

  test("coming back to a finished tune shows the result at once, with nothing replaying", async ({ page }) => {
    await startTune(page, ["fast-tune"], SMALL.id, "chat");
    await expect(page.locator(".winner")).toBeVisible({ timeout: 25_000 });
    await expect(page.locator(".winner--enter")).toHaveCount(1);
    await page.getByRole("link", { name: "Home" }).first().click();
    await expect(page.locator("main h1")).not.toHaveText("Speed tune");
    await page.getByRole("link", { name: "Tune" }).first().click();
    await expect(page.locator(".winner")).toBeVisible();
    await expect(page.locator(".winner--enter")).toHaveCount(0);
    await expect(page.locator(".winner [data-count-to]")).toHaveCount(0);
    await page.waitForTimeout(100);
    const running = await page.evaluate(() =>
      document
        .getAnimations()
        .filter((a) => a.playState === "running")
        .map((a) => String((a.effect as KeyframeEffect | null)?.target instanceof Element ? ((a.effect as KeyframeEffect).target as Element).closest(".winner") !== null : false))
        .filter((inWinner) => inWinner === "true"),
    );
    expect(running).toEqual([]);
  });

  test("the run clock ticks between events, and stops when cancelled", async ({ page }) => {
    await startTune(page, ["fast-tune", "tune-hold-soak"], MOE.id, "chat");
    await expect(page.locator(".soak-item")).toHaveCount(2, { timeout: 20_000 });
    await expect(page.locator('.soak-item[data-state="running"]')).toBeVisible();
    const clock = page.locator(".run-head__clock .mono");
    await expect(page.locator(".run-head__clock [data-icon='clock']")).toHaveCount(1);
    // Playback is frozen here, so no event arrives: the clock must still move.
    const held = await clock.innerText();
    await expect(clock).not.toHaveText(held, { timeout: 2_500 });
    await page.getByRole("button", { name: "Cancel tune" }).click();
    await expect(page.locator(".run")).toHaveAttribute("data-status", "cancelled");
    const stopped = await clock.innerText();
    await page.waitForTimeout(1_600);
    await expect(clock).toHaveText(stopped);
  });

  test("a tune that finishes while another screen is open says so in one toast", async ({ page }) => {
    await startTune(page, ["fast-tune"], SMALL.id, "chat");
    await page.getByRole("link", { name: "Home" }).first().click();
    await expect(page.locator("main h1")).not.toHaveText("Speed tune");
    const toast = page.locator(".toast", { hasText: "Tune finished" });
    await expect(toast).toBeVisible({ timeout: 25_000 });
    await expect(toast).toContainText(SMALL.name);
    await expect(page.locator(".toast", { hasText: "Tune finished" })).toHaveCount(1);
  });

  test("a crash stops the run with its reason and offers to start again", async ({ page }) => {
    await startTune(page, ["fast-tune", "fail-tune"], DENSE.id, "coding");
    await expect(page.locator(".run")).toHaveAttribute("data-status", "failed", { timeout: 20_000 });
    await expect(page.locator(".run [role='alert']")).toContainText("stopped unexpectedly");
    // Announced once: by the alert, not a second time by the status line.
    await expect(statusLine(page)).toHaveText("");
    await expect(page.locator(".chip[data-tone='danger']").first()).toContainText("Stopped");
    await page.getByRole("button", { name: "Start again" }).click();
    await expect(page.locator(".setup .pick-card input:checked").first()).toBeFocused();
  });

  test("with the integrity check switched off, the probe says it was skipped", async ({ page }) => {
    await page.goto("/#/settings");
    const sw = page.getByRole("switch", { name: /integrity/i });
    await expect(sw).toBeChecked();
    await sw.click({ force: true });
    await expect(sw).not.toBeChecked();
    await page.getByRole("link", { name: "Tune" }).first().click();
    await page.locator(`.pick-card[data-model="${MOE.id}"]`).click();
    await page.getByRole("button", { name: "Start tune" }).click();
    await expect(page.locator(".probe-line__note")).toContainText("off in Settings", { timeout: 15_000 });
    await expect(page.locator(".probe-line__ok")).toHaveCount(0);
  });
});

test.describe("layout", () => {
  test("no horizontal scroll at the smallest desktop window, in every state", async ({ page }) => {
    test.setTimeout(90_000);
    await page.setViewportSize({ width: 1024, height: 640 });
    await page.goto("/#/tune");
    await expect(page.locator(".pick-card").first()).toBeVisible();
    expect(await overflow(page), "idle").toEqual([]);

    await startTune(page, ["fast-tune", "tune-hold-screen"], MOE.id, "chat");
    await expect(page.locator('.cand[data-candidate="c4"][data-status="screening"]')).toBeVisible({ timeout: 15_000 });
    await page.locator('.cand[data-candidate="c1"] .cand__head').click();
    expect(await overflow(page), "screening").toEqual([]);

    await startTune(page, ["fast-tune", "tune-hold-soak"], MOE.id, "chat");
    await expect(page.locator(".soak-item")).toHaveCount(2, { timeout: 20_000 });
    expect(await overflow(page), "soak").toEqual([]);

    await startTune(page, ["fast-tune"], MOE.id, "long-docs");
    await expect(page.locator(".winner")).toBeVisible({ timeout: 25_000 });
    await page.locator(".winner").scrollIntoViewIfNeeded();
    expect(await overflow(page), "result").toEqual([]);
  });
});

test.describe("accessibility and motion", () => {
  for (const scheme of ["dark", "light"] as const) {
    test(`a run in progress and a finished one have no axe violations (${scheme})`, async ({ page }) => {
      test.setTimeout(90_000);
      await page.emulateMedia({ colorScheme: scheme });
      await startTune(page, ["fast-tune", "tune-hold-soak"], MOE.id, "chat");
      await expect(page.locator(".soak-item")).toHaveCount(2, { timeout: 20_000 });
      await page.locator('.cand[data-candidate="c5"] .cand__head').click();
      await settle(page);
      expect(await axe(page)).toEqual([]);

      await startTune(page, ["fast-tune"], SMALL.id, "long-docs");
      await expect(page.locator(".winner")).toBeVisible({ timeout: 25_000 });
      await page.locator(".winner").scrollIntoViewIfNeeded();
      await settle(page);
      expect(await axe(page)).toEqual([]);
    });
  }

  test("with reduced motion, nothing animates while the run streams in", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await startTune(page, ["fast-tune"], MOE.id, "chat");
    for (let i = 0; i < 6; i++) {
      await page.waitForTimeout(250);
      const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running").map((a) => (a.effect as KeyframeEffect | null)?.target?.className ?? "?"));
      expect(running).toEqual([]);
    }
    await expect(page.locator(".winner")).toBeVisible({ timeout: 25_000 });
  });
});
