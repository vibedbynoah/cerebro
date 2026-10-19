import { expect, openScreen, ROUTE_IDS, scrollThrough, settle, test } from "./fixtures";

const TITLES: Record<(typeof ROUTE_IDS)[number], string> = {
  home: "This machine, measured.",
  hub: "Model Hub",
  models: "Models",
  tune: "Tune",
  monitor: "Monitor",
  turbo: "Turbo",
  chat: "Chat",
  images: "Image Studio",
  documents: "Documents",
  train: "Train",
  plugins: "Plugins",
  tools: "Tools",
  settings: "Settings",
  about: "About Cerebro",
};

for (const scheme of ["dark", "light"] as const) {
  test.describe(`every screen renders (${scheme})`, () => {
    test.use({ colorScheme: scheme });
    for (const id of ROUTE_IDS) {
      test(`${id} renders with zero console errors`, async ({ page }) => {
        await openScreen(page, id);
        await expect(page.locator("main h1")).toHaveText(TITLES[id]);
        await expect(page.locator(`a.nav-link[data-route="${id}"]`)).toHaveAttribute("aria-current", "page");
        await scrollThrough(page);
        await settle(page);
        // Frame is complete: title bar and its lockup, the engine pill, one nav entry per screen, the HUD and its disclosure.
        await expect(page.locator(".titlebar")).toBeVisible();
        await expect(page.locator(".titlebar [data-brand='lockup']")).toBeVisible();
        await expect(page.locator(".titlebar .engine-pill")).toBeVisible();
        await expect(page.locator("a.nav-link")).toHaveCount(ROUTE_IDS.length);
        await expect(page.locator(".statusbar")).toContainText("Illustrative data");
      });
    }
  });
}

test("navigating by sidebar link updates the route, title and current entry", async ({ page }) => {
  await openScreen(page, "home");
  await page.getByRole("link", { name: "Models" }).click();
  await expect(page).toHaveURL(/#\/models$/);
  await expect(page.locator("main h1")).toHaveText("Models");
  await expect(page).toHaveTitle("Models · Cerebro");
  await expect(page.locator('a.nav-link[aria-current="page"]')).toHaveCount(1);
});

test("an unknown route falls back to Home", async ({ page }) => {
  await page.goto("/#/../../etc/passwd");
  await expect(page.locator("main h1")).toHaveText(TITLES.home);
});

test.describe("keyboard", () => {
  test("skip link, then focus rings on every stop of the frame", async ({ page }) => {
    await openScreen(page, "home");
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "Skip to content" });
    await expect(skip).toBeFocused();
    await expect(skip).toBeInViewport();
    await page.keyboard.press("Enter");
    await expect(page.locator("main")).toBeFocused();

    // Walk the first stops from the top and check each shows a visible ring.
    await page.locator("body").focus();
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    for (let i = 0; i < 8; i++) {
      await page.keyboard.press("Tab");
      const ring = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el || el === document.body) return "none";
        const own = getComputedStyle(el).boxShadow;
        // Ark's switch focuses a hidden input and rings its visible control instead.
        const control = el.closest(".switch")?.querySelector(".switch__control");
        return own !== "none" ? own : control ? getComputedStyle(control).boxShadow : "none";
      });
      expect(ring, `focus ring on tab stop ${i + 1}`).not.toBe("none");
    }
  });

  test("arrow keys move between sidebar entries", async ({ page }) => {
    await openScreen(page, "home");
    await page.locator('a.nav-link[data-route="home"]').focus();
    await page.keyboard.press("ArrowDown");
    await expect(page.locator('a.nav-link[data-route="hub"]')).toBeFocused();
    await page.keyboard.press("End");
    await expect(page.locator('a.nav-link[data-route="about"]')).toBeFocused();
    await page.keyboard.press("ArrowDown");
    await expect(page.locator('a.nav-link[data-route="home"]')).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/#\/home$/);
  });

  test("Ctrl+B collapses the sidebar and labels stay accessible", async ({ page }) => {
    await openScreen(page, "home");
    await page.keyboard.press("Control+b");
    await expect(page.locator("nav.sidebar")).toHaveClass(/is-collapsed/);
    await expect(page.locator("nav.sidebar").getByRole("link", { name: "Documents" })).toBeVisible();
    const width = await page.locator("nav.sidebar").evaluate((el) => el.getBoundingClientRect().width);
    expect(width).toBeLessThan(80);
    // Hovering a collapsed entry shows its tooltip (a real pointer moves more than once).
    const tune = page.locator('a.nav-link[data-route="tune"]');
    // Collapsed entries are exactly as wide as the column: the whole hit area is inside the sidebar.
    const box = await tune.boundingBox();
    const side = await page.locator("nav.sidebar").boundingBox();
    expect(box && side && box.x + box.width <= side.x + side.width).toBe(true);
    await tune.hover();
    await tune.hover({ position: { x: 10, y: 12 } });
    await expect(page.locator(".tooltip", { hasText: "Tune" })).toBeVisible();
    await page.reload();
    await expect(page.locator("nav.sidebar")).toHaveClass(/is-collapsed/);
  });
});

test.describe("command palette", () => {
  test("Ctrl+K opens it, fuzzy search filters, Enter navigates", async ({ page }) => {
    await openScreen(page, "home");
    await page.keyboard.press("Control+k");
    const dialog = page.getByRole("dialog", { name: "Command palette" });
    await expect(dialog).toBeVisible();
    const input = page.getByRole("combobox", { name: "Search screens and commands" });
    await expect(input).toBeFocused();
    await input.fill("tne");
    const first = page.getByRole("option").first();
    await expect(first).toHaveAttribute("data-command", "go-tune");
    await expect(first).toHaveAttribute("aria-selected", "true");
    // "T" and "ne" are matched: contiguous hits share one <mark>.
    await expect(first.locator("mark")).toHaveText(["T", "ne"]);
    await page.keyboard.press("Enter");
    await expect(dialog).toBeHidden();
    await expect(page).toHaveURL(/#\/tune$/);
    await expect(page.locator("main h1")).toHaveText("Tune");
    await expect(page.locator("main")).toBeFocused();
  });

  test("arrow keys move the selection; Escape closes and returns focus", async ({ page }) => {
    await openScreen(page, "home");
    const trigger = page.getByRole("button", { name: /Search screens and commands/ });
    await trigger.click();
    const options = page.getByRole("option");
    await expect(options.first()).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("ArrowDown");
    await expect(options.nth(1)).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("ArrowUp");
    await page.keyboard.press("ArrowUp");
    await expect(options.last()).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toBeHidden();
    await expect(trigger).toBeFocused();
  });

  test("no matches says so, and nothing runs", async ({ page }) => {
    await openScreen(page, "home");
    await page.keyboard.press("Control+k");
    await page.getByRole("combobox").fill("zzzzqqq");
    await expect(page.getByRole("option")).toHaveCount(0);
    await expect(page.locator(".palette__empty")).toContainText("No screens or commands match");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("dialog")).toBeVisible();
  });

  test("switches theme and the choice survives a reload", async ({ page }) => {
    await openScreen(page, "home");
    await page.keyboard.press("Control+k");
    await page.getByRole("combobox").fill("light");
    await page.keyboard.press("Enter");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await expect(page.locator(".toast")).toContainText("Theme: Light");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    expect(bg).not.toBe("rgba(0, 0, 0, 0)");
  });
});

test.describe("user control", () => {
  test("every protection can be switched off from Settings, with a neutral toast and no prompt", async ({ page }) => {
    await openScreen(page, "settings");
    const containment = page.getByRole("switch", { name: "Process containment" });
    await expect(containment).toBeChecked();
    await containment.click({ force: true });
    await expect(containment).not.toBeChecked();
    await expect(page.locator(".toast")).toContainText("Process containment off");
    await expect(page.getByRole("dialog")).toHaveCount(0);
    for (const name of ["Clean environment", "Backend integrity check", "Permission prompts", "API on loopback only"]) {
      await page.getByRole("switch", { name }).click({ force: true });
      await expect(page.getByRole("switch", { name })).not.toBeChecked();
    }
  });

  test("keyboard toggles a switch", async ({ page }) => {
    await openScreen(page, "settings");
    const offline = page.getByRole("switch", { name: "Offline mode" });
    await offline.focus();
    await page.keyboard.press("Space");
    await expect(offline).toBeChecked();
    await expect(page.locator(".statusbar")).toContainText("Offline mode");
  });

  test("theme radio group in Settings", async ({ page }) => {
    await openScreen(page, "settings");
    await page.getByRole("radio", { name: "Dark" }).check({ force: true });
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByRole("radio", { name: "Match system" }).check({ force: true });
    await expect(page.locator("html")).not.toHaveAttribute("data-theme", /./);
  });
});

test("window controls explain themselves in the browser preview", async ({ page }) => {
  await openScreen(page, "home");
  await page.getByRole("button", { name: "Close" }).click();
  await expect(page.locator(".toast")).toContainText("Window controls work in the desktop app");
});

test("Model Hub degrades gracefully offline", async ({ page }) => {
  await openScreen(page, "hub");
  await expect(page.locator(".notice")).toContainText("downloading a file needs a connection");
});

test("the HUD shows mock live data and updates it", async ({ page }) => {
  await openScreen(page, "home");
  const speed = page.locator(".statusbar [data-hud='speed'] .hud-item__value");
  await expect(speed).toContainText("tok/s");
  await expect(page.locator(".statusbar [data-hud='backend']")).toContainText("127.0.0.1:8181");
  await expect(page.getByRole("meter", { name: "VRAM in use" })).toHaveAttribute("aria-valuemax", "8");
});

test("the Home tiles and the HUD always show the same live sample", async ({ page }) => {
  await openScreen(page, "home");
  // Past two HUD ticks (1 s each), so both have moved off the first sample.
  await page.waitForTimeout(2500);
  const read = () =>
    page.evaluate(() => {
      const tiles = Array.from(document.querySelectorAll<HTMLElement>(".stat-tile .live-value"));
      const hud = (sel: string) => document.querySelector(`.statusbar [data-hud='${sel}'] .hud-item__value`)?.firstChild?.textContent?.trim();
      return {
        tileSpeed: tiles[0]?.dataset["value"],
        tileVram: tiles[1]?.dataset["value"],
        hudSpeed: hud("speed"),
        hudVram: hud("vram"),
        visibleSpeed: tiles[0]?.querySelector(".live-value__digits")?.textContent,
      };
    });
  // Read in one synchronous pass, so a tick cannot land between the two readouts.
  for (let i = 0; i < 3; i++) {
    const r = await read();
    expect(r.tileSpeed).toMatch(/^\d+\.\d$/);
    expect(r.tileSpeed).toBe(r.hudSpeed);
    expect(r.tileVram).toBe(r.hudVram);
    expect(r.visibleSpeed).toBe(r.tileSpeed);
    await page.waitForTimeout(700);
  }
  // Live tiles are never count-up targets.
  await expect(page.locator(".stat-tile--updating [data-count-to]")).toHaveCount(0);
});

test("at the narrowest window (1040x700) the HUD still names the loaded model, truncated, with no overflow", async ({ page }) => {
  await page.setViewportSize({ width: 1040, height: 700 });
  await openScreen(page, "home");
  const model = page.locator(".statusbar [data-hud='model'] .hud-item__value");
  await expect(model).toBeVisible();
  const box = await model.evaluate((el) => ({ w: el.getBoundingClientRect().width, text: el.textContent ?? "", title: el.getAttribute("title") }));
  expect(box.w).toBeGreaterThan(40);
  expect(box.text.length).toBeGreaterThan(0);
  expect(box.title).toBe(box.text.trim());
  const overflow = await page.locator(".statusbar").evaluate((el) => el.scrollWidth - el.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  // Every HUD item stays inside the window.
  const right = await page.locator(".statusbar [data-hud='network']").evaluate((el) => el.getBoundingClientRect().right);
  expect(right).toBeLessThanOrEqual(1040);
});

test("the Models library reports quality beside speed", async ({ page }) => {
  await openScreen(page, "models");
  await expect(page.getByRole("columnheader", { name: /Quality/ })).toBeVisible();
  const first = page.locator(".lib-body > .lib-row").first();
  // Cells: size, fit, last tune, quality, load.
  await expect(first.getByRole("gridcell").nth(3)).toContainText("Not measured yet");
});

test("the maximize control reads Maximize in the browser preview", async ({ page }) => {
  await openScreen(page, "home");
  await expect(page.getByRole("button", { name: "Maximize" })).toBeVisible();
  await expect(page.locator(".window-control [data-icon='maximize']")).toHaveCount(1);
});

test.describe("reduced motion", () => {
  test.use({ contextOptions: { reducedMotion: "reduce" } });

  test("final states show at once and nothing animates", async ({ page }) => {
    await page.goto("/#/home");
    await expect(page.locator("main h1")).toBeVisible();
    await expect(page.locator("html")).not.toHaveClass(/motion-pending|motion-ready/);
    // Count-ups show their real value immediately, with no scroll.
    await expect(page.locator(".stat-tile .count").first()).toContainText(/\d/);
    const running = await page.evaluate(() => document.getAnimations().filter((a) => a.playState === "running").length);
    expect(running).toBe(0);
    const anim = await page.locator(".screen").evaluate((el) => getComputedStyle(el).animationName);
    expect(anim).toBe("none");
    // Sections below the fold are already visible without scrolling.
    const opacity = await page.locator(".section").last().evaluate((el) => getComputedStyle(el).opacity);
    expect(opacity).toBe("1");
    // The palette opens and closes without animating.
    await page.keyboard.press("Control+k");
    await expect(page.getByRole("dialog")).toBeVisible();
    const paletteAnim = await page.locator(".palette").evaluate((el) => getComputedStyle(el).animationName);
    expect(paletteAnim).toBe("none");
  });
});

test.describe("full motion", () => {
  test("scroll reveals and count-ups arrive when scrolled into view", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 600 });
    await openScreen(page, "home");
    const last = page.locator(".section").last();
    await expect(last).toHaveAttribute("data-motion-armed", "");
    await scrollThrough(page);
    await expect(last).toHaveClass(/is-visible/);
    await settle(page);
    await expect(page.locator(".comparison .bar").first()).toHaveClass(/is-visible/);
  });
});

for (const viewport of [
  { width: 1280, height: 800 },
  { width: 1920, height: 1080 },
]) {
  test(`no overflow at ${viewport.width}x${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    for (const id of ROUTE_IDS) {
      await openScreen(page, id);
      await scrollThrough(page);
      const overflow = await page.evaluate(() => {
        const bad: string[] = [];
        const doc = document.documentElement;
        if (doc.scrollWidth > window.innerWidth || doc.scrollHeight > window.innerHeight) bad.push("document scrolls");
        for (const sel of [".titlebar", ".sidebar", "main", ".statusbar"]) {
          const el = document.querySelector(sel);
          if (el && el.scrollWidth > el.clientWidth + 1) bad.push(`${sel} overflows horizontally by ${el.scrollWidth - el.clientWidth}px`);
        }
        // Nothing in the screen pokes out past the main column.
        const main = document.querySelector("main");
        const edge = main ? main.getBoundingClientRect().right : window.innerWidth;
        for (const el of Array.from(document.querySelectorAll("main *"))) {
          const r = el.getBoundingClientRect();
          if (r.width > 0 && r.right > edge + 1 && !el.closest(".table-scroll")) bad.push(`${el.tagName.toLowerCase()}.${el.className} ends at ${r.right}`);
        }
        return bad.slice(0, 5);
      });
      expect(overflow, `${id} at ${viewport.width}x${viewport.height}`).toEqual([]);
    }
  });
}

test("still works with Object.prototype frozen (Tauri freezePrototype)", async ({ page }) => {
  await page.addInitScript(() => Object.freeze(Object.prototype));
  await openScreen(page, "home");
  await page.keyboard.press("Control+k");
  await page.getByRole("combobox").fill("sett");
  await page.keyboard.press("Enter");
  await expect(page.locator("main h1")).toHaveText("Settings");
  await expect(page.getByRole("dialog")).toBeHidden();
  await page.getByRole("switch", { name: "Offline mode" }).click({ force: true });
  await expect(page.locator(".toast")).toContainText("Offline mode on");
});
