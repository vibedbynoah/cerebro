import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { expect, openScreen, settle, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];
const PILL_TEXT = /^(Idle|Loading model|Reading prompt \d{1,3}%|Generating \d+\.\d tok\/s)/;

async function axe(page: Page): Promise<string[]> {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return results.violations.map((v) => `${v.id} (${v.impact ?? "?"}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join(" | ")}`);
}

async function openMonitor(page: Page, query = ""): Promise<void> {
  await page.goto(`/${query}#/monitor`);
  await expect(page.locator("main h1")).toHaveText("Monitor");
  await expect(page.locator(".mon-grid .mon-card")).toHaveCount(10);
}

test.describe("header pill", () => {
  test("shows the engine state on every screen and opens the Monitor", async ({ page }) => {
    await openScreen(page, "settings");
    const pill = page.locator(".titlebar .engine-pill");
    await expect(pill.locator(".engine-pill__text")).toHaveText(PILL_TEXT);
    await expect(pill).toHaveAttribute("aria-label", /^Engine: .+\. Open Monitor$/);
    await pill.click();
    await expect(page).toHaveURL(/#\/monitor$/);
    await expect(page.locator("main h1")).toHaveText("Monitor");
  });

  test("says so when the engine stops answering", async ({ page }) => {
    await page.goto("/?mock=fail-metrics#/monitor");
    await expect(page.locator(".engine-pill")).toHaveAttribute("data-tone", "offline", { timeout: 8000 });
    await expect(page.locator(".engine-pill__text")).toHaveText("Engine not reachable");
    await expect(page.locator("main .notice[role='alert']")).toContainText("not answering");
  });
});

test.describe("monitor", () => {
  test("shows every required card, flagged illustrative, and keeps updating", async ({ page }) => {
    await openMonitor(page);
    await expect(page.locator(".mon-state")).toBeVisible();
    await expect(page.locator(".mon-model")).toContainText("Qwen3-Coder-30B-A3B-Instruct");
    await expect(page.locator(".mon-cache")).toBeVisible();
    await expect(page.locator(".mon-alerts-card")).toBeVisible();
    await expect(page.locator(".mon-summary .stat-tile__flag")).toHaveText("Illustrative data — not a measurement");
    await expect(page.locator(".mon-totals")).toHaveText(/^Since \w{3} \d\d:\d\d: \d+ requests/);
    for (const m of ["speed", "vram", "gpu", "temp", "power", "pcie", "cpu", "ram", "disk", "experts"]) {
      await expect(page.locator(`[data-metric="${m}"] .spark__plot`)).toBeVisible();
    }
    // VRAM is split into weights, KV cache and expert cache (and scratch, other programs).
    await expect(page.locator("[data-metric='vram'] .mon-legend li")).toHaveCount(5);
    await expect(page.locator("[data-metric='vram'] .mon-legend")).toContainText("KV cache");
    await expect(page.locator("[data-metric='vram'] .mon-legend")).toContainText("Expert cache");
    // Decode and prefill share one card.
    await expect(page.locator("[data-metric='speed'] .spark__series")).toHaveCount(2);
    const d0 = await page.locator("[data-metric='gpu'] .spark__line").getAttribute("d");
    await expect.poll(async () => page.locator("[data-metric='gpu'] .spark__line").getAttribute("d"), { timeout: 4000 }).not.toBe(d0);
  });

  test("a new sample never shifts the layout", async ({ page }) => {
    await openMonitor(page);
    await settle(page).catch(() => {});
    const shift = await page.evaluate(
      () =>
        new Promise<number>((resolve) => {
          let total = 0;
          new PerformanceObserver((list) => {
            for (const e of list.getEntries() as Array<PerformanceEntry & { value: number; hadRecentInput: boolean }>) if (!e.hadRecentInput) total += e.value;
          }).observe({ type: "layout-shift", buffered: false });
          setTimeout(() => resolve(total), 4500);
        }),
    );
    expect(shift).toBeLessThan(0.01);
  });

  test("reduced motion: live updates run no animation at all", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await openMonitor(page);
    await page.waitForTimeout(1200);
    const running = await page.evaluate(async () => {
      let seen = 0;
      for (let i = 0; i < 25; i++) {
        seen = Math.max(seen, document.getAnimations().filter((a) => a.playState === "running").length);
        await new Promise((r) => setTimeout(r, 100));
      }
      return seen;
    });
    expect(running).toBe(0);
  });

  test("the range switch changes every chart and is remembered", async ({ page }) => {
    await openMonitor(page);
    await page.locator(".mon-range label", { hasText: "24 h" }).click();
    await expect(page.locator("[data-metric='cpu'] .spark__foot")).toContainText("Last 24 h");
    await expect(page.locator("#mon-live-title + .section__desc")).toContainText("Averaged over 5 minutes");
    await page.reload();
    await expect(page.locator(".mon-range input[value='24h']")).toBeChecked();
    await page.locator(".mon-range label", { hasText: "1 min" }).click();
    await expect(page.locator("[data-metric='cpu'] .spark__foot")).toContainText("Last 1 min");
  });

  test("charts can be read with the keyboard", async ({ page }) => {
    await openMonitor(page);
    const plot = page.locator("[data-metric='gpu'] .spark__plot");
    await plot.focus();
    await expect(plot).toHaveAttribute("aria-label", /^GPU load, Last 1 min\. GPU load now \d+%/);
    await page.keyboard.press("Home");
    await expect(page.locator("[data-metric='gpu'] .spark__when")).toHaveText(/ago$/);
    await page.keyboard.press("End");
    await expect(page.locator("[data-metric='gpu'] .spark__when")).toHaveText("now");
    await page.keyboard.press("Escape");
    await expect(page.locator("[data-metric='gpu'] .spark__when")).toHaveCount(0);
  });

  test("exports the chosen range as CSV and JSON, labelled illustrative", async ({ page }) => {
    await openMonitor(page);
    await page.locator(".mon-range label", { hasText: "15 min" }).click();
    await expect(page.locator("[data-metric='cpu'] .spark__foot")).toContainText("Last 15 min");
    const [csv] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export the last 15 minutes as CSV" }).click()]);
    expect(csv.suggestedFilename()).toMatch(/^cerebro-metrics-15m-\d{8}-\d{6}\.csv$/);
    const text = await (await csv.createReadStream()).toArray().then((b) => Buffer.concat(b).toString("utf8"));
    const lines = text.trim().split("\r\n");
    expect(lines[0]).toMatch(/^time,decode_tok_s,prefill_tok_s,gpu_util_pct/);
    expect(lines).toHaveLength(181);
    const [json] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export the last 15 minutes as JSON" }).click()]);
    const body = JSON.parse(await (await json.createReadStream()).toArray().then((b) => Buffer.concat(b).toString("utf8")));
    expect(body.provenance).toBe("illustrative");
    expect(body.range).toBe("15m");
    expect(body.samples).toHaveLength(180);
    await expect(page.locator(".toast").first()).toContainText("Exported the last 15 minutes");
  });
});

test.describe("request inspector", () => {
  test("searches, filters, opens a request and copies its capture", async ({ page }) => {
    await openMonitor(page);
    const list = page.locator(".mon-reqs");
    await list.scrollIntoViewIfNeeded();
    expect(await page.locator(".mon-req").count()).toBeGreaterThanOrEqual(16);

    await page.getByLabel("Show").selectOption("failed");
    const tags = await page.locator(".mon-req .mon-tag").allTextContents();
    expect(tags.length).toBeGreaterThan(0);
    for (const t of tags) expect(["Error", "Cancelled"]).toContain(t);
    await page.getByLabel("Show").selectOption("all");

    await page.getByLabel("Search requests").fill("invoice");
    await expect(page.locator(".mon-req").first()).toBeVisible();
    await page.getByLabel("Search requests").fill("no such request anywhere");
    await expect(page.locator(".mon-inspector__list .mon-empty")).toHaveText("No request matches.");
    await page.getByLabel("Search requests").fill("");

    const done = page.locator(".mon-req:has(.mon-tag[data-state='completed'])").first();
    const rid = await done.getAttribute("data-request");
    await done.click();
    await expect(done).toHaveAttribute("aria-pressed", "true");
    const detail = page.locator(".mon-inspector__detail");
    await expect(detail.locator(".mon-detail__title")).toContainText(rid ?? "?");
    for (const label of ["Wall time", "Load", "Queue", "First token", "Prompt", "Written", "Writing", "Reading", "Response", "HTTP"]) {
      await expect(detail.locator("dt", { hasText: label }).first()).toBeVisible();
    }
    // Tabs: arrow keys move between Input, Output, Reasoning and Response.
    const output = page.getByRole("tab", { name: "Output" });
    await output.focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Reasoning" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Home");
    await expect(page.getByRole("tab", { name: "Input" })).toBeFocused();
    await expect(detail.locator(".mon-pre")).toContainText('"model": ');
    // Reading the clipboard back is denied by the app's own permissions policy, so the write is
    // observed instead.
    await page.evaluate(() => {
      const w = window as unknown as { __copied: string };
      w.__copied = "";
      navigator.clipboard.writeText = async (t: string) => {
        w.__copied = t;
      };
    });
    await detail.getByRole("button", { name: "Copy" }).click();
    await expect(detail.getByRole("button", { name: "Copied" })).toBeVisible();
    expect(await page.evaluate(() => (window as unknown as { __copied: string }).__copied)).toContain('"model"');
  });

  test("a running request can be opened from the state card", async ({ page }) => {
    await openMonitor(page);
    const link = page.locator(".mon-state .text-link--btn");
    await expect(link).toBeVisible({ timeout: 15000 });
    const rid = (await link.textContent()) ?? "";
    await link.click();
    await expect(page.locator(".mon-detail__title")).toContainText(rid);
    await expect(page.locator(`button.mon-req[data-request="${rid}"]`)).toBeFocused();
  });

  test("is free of axe violations with a request open (dark and light)", async ({ page }) => {
    for (const scheme of ["dark", "light"] as const) {
      await page.emulateMedia({ colorScheme: scheme });
      await openMonitor(page);
      await page.locator(".mon-req").first().click();
      await settle(page).catch(() => {});
      expect(await axe(page), scheme).toEqual([]);
    }
  });
});

test.describe("alerts and machines", () => {
  test("thermal throttling, VRAM pressure and a speed regression each raise one quiet notice", async ({ page }) => {
    await openMonitor(page, "?mock=monitor-alerts");
    // One toast as the alert starts (it fades after a few seconds), even though the condition holds.
    await expect(page.locator(".toast").filter({ hasText: "VRAM almost full" })).toBeVisible({ timeout: 10000 });
    await page.waitForTimeout(1500);
    await expect(page.locator(".toast").filter({ hasText: "VRAM almost full" })).toHaveCount(1);
    for (const kind of ["thermal", "vram", "regression"]) await expect(page.locator(`.mon-alert[data-kind='${kind}']`)).toBeVisible({ timeout: 20000 });
    await expect(page.locator(".mon-alert[data-kind='regression']")).toContainText(/below the \d+\.\d tok\/s measured at 8,192 context/);
    await expect(page.locator("[data-metric='temp']")).toHaveClass(/is-alert/);
    await settle(page).catch(() => {});
    expect(await axe(page)).toEqual([]);
  });

  test("a two-GPU machine shows each card's values", async ({ page }) => {
    await openMonitor(page, "?mock=monitor-multi-gpu");
    for (const m of ["gpu", "temp", "power", "pcie", "vram"]) {
      await expect(page.locator(`[data-metric='${m}'] .mon-card__sub`)).toContainText("GPU 0");
      await expect(page.locator(`[data-metric='${m}'] .mon-card__sub`)).toContainText("GPU 1");
    }
  });
});
