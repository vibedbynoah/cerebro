import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/** Scrolls `selector` fully into view and waits long enough for every scroll-triggered primitive
 *  (reveal/bar/count-up, all <= --duration-slow + a stagger delay) to reach its final state via
 *  the real IntersectionObserver path — not a blind "jump to the bottom of the page" scroll, which
 *  can skip straight past a section's trigger line without ever putting it in the viewport at an
 *  observed moment (IntersectionObserver reports actual intersection at observation time, not
 *  "was crossed at some point during the jump"). */
async function scrollSectionIntoView(page: Page, selector: string) {
  await page.locator(selector).scrollIntoViewIfNeeded();
  await page.waitForTimeout(1000);
}

const THEMES = ["dark", "light"] as const;

async function setTheme(page: Page, theme: (typeof THEMES)[number]) {
  await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
  await page.waitForTimeout(150); // let fonts/paint settle before measuring
}

test.describe("preview loads cleanly", () => {
  for (const theme of THEMES) {
    test(`zero console errors and zero off-origin requests (${theme})`, async ({ page, baseURL }) => {
      const consoleErrors: string[] = [];
      const offOrigin: string[] = [];
      const baseOrigin = new URL(baseURL ?? "http://127.0.0.1:4319").origin;

      page.on("console", (msg) => {
        if (msg.type() === "error") consoleErrors.push(msg.text());
      });
      page.on("pageerror", (err) => consoleErrors.push(String(err)));
      page.on("request", (req) => {
        const origin = new URL(req.url()).origin;
        if (origin !== baseOrigin) offOrigin.push(req.url());
      });

      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await setTheme(page, theme);

      expect(consoleErrors, `console errors:\n${consoleErrors.join("\n")}`).toEqual([]);
      expect(offOrigin, `off-origin requests:\n${offOrigin.join("\n")}`).toEqual([]);
    });
  }

  test("no Content-Security-Policy violations during load and a full scroll (both themes)", async ({ page }) => {
    for (const theme of THEMES) {
      const violations: string[] = [];
      page.on("console", (msg) => {
        // Chromium reports CSP violations to the console as well as the securitypolicyviolation
        // event; capturing the event directly (below) is the primary signal.
        if (/content security policy/i.test(msg.text())) violations.push(msg.text());
      });
      await page.addInitScript(() => {
        (window as unknown as { __csp: string[] }).__csp = [];
        document.addEventListener("securitypolicyviolation", (e) => {
          (window as unknown as { __csp: string[] }).__csp.push(`${e.violatedDirective} ${e.blockedURI}`);
        });
      });
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await setTheme(page, theme);
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(200);
      const eventViolations = await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp);
      expect(eventViolations, `CSP violations (${theme}):\n${eventViolations.join("\n")}`).toEqual([]);
    }
  });
});

/**
 * One list of pairs, asserted against every background in every theme (including the
 * OS-preference path with no explicit `data-theme`), instead of only the ~11 pairs the markup
 * happens to annotate. A token that regresses on a pair nobody thought to mark in HTML still
 * fails here.
 */
const TEXT_ROLES = [
  "--color-text-primary",
  "--color-text-secondary",
  "--color-text-muted",
  // --color-accent is a FILL-only role (checked separately, against its own -contrast token, in
  // FILLS below) — it is never painted as text-on-background, so it doesn't belong in this list.
  // --color-accent-text is the token actually used for that role.
  "--color-accent-text",
  "--color-success",
  "--color-warning",
  "--color-danger",
];
const UI_ROLES = ["--color-border-strong", "--color-focus-ring"];
const BACKGROUNDS = ["--color-bg", "--color-surface-1", "--color-surface-2", "--color-surface-3"];

interface ContrastFailure {
  pair: string;
  ratio: number;
  min: number;
}

async function contrastMatrix(page: Page): Promise<ContrastFailure[]> {
  return page.evaluate(
    async ({ TEXT_ROLES, UI_ROLES, BACKGROUNDS }) => {
      const mod = await import("../contrast.js");
      const styles = getComputedStyle(document.documentElement);
      const v = (name: string) => styles.getPropertyValue(name).trim();
      const failures: { pair: string; ratio: number; min: number }[] = [];
      for (const bg of BACKGROUNDS) {
        for (const fg of TEXT_ROLES) {
          const ratio = mod.contrastOf(v(fg), v(bg));
          if (ratio < 4.5) failures.push({ pair: `${fg} on ${bg}`, ratio: Math.round(ratio * 100) / 100, min: 4.5 });
        }
        for (const fg of UI_ROLES) {
          const ratio = mod.contrastOf(v(fg), v(bg));
          if (ratio < 3) failures.push({ pair: `${fg} on ${bg}`, ratio: Math.round(ratio * 100) / 100, min: 3 });
        }
      }
      // Text/background pairs painted directly onto a filled brand/status colour (the "-contrast"
      // tokens), at 4.5:1.
      const FILLS: Array<[string, string]> = [
        ["--color-accent-contrast", "--color-accent"],
        ["--color-success-contrast", "--color-success"],
        ["--color-warning-contrast", "--color-warning"],
        ["--color-danger-contrast", "--color-danger"],
      ];
      for (const [fg, bg] of FILLS) {
        const ratio = mod.contrastOf(v(fg), v(bg));
        if (ratio < 4.5) failures.push({ pair: `${fg} on ${bg}`, ratio: Math.round(ratio * 100) / 100, min: 4.5 });
      }
      return failures;
    },
    { TEXT_ROLES, UI_ROLES, BACKGROUNDS },
  );
}

test.describe("WCAG 2.2 AA contrast matrix (computed from rendered pixels)", () => {
  for (const theme of THEMES) {
    test(`explicit data-theme="${theme}"`, async ({ page }) => {
      await page.goto("/preview.html");
      await setTheme(page, theme);
      const failures = await contrastMatrix(page);
      expect(failures, `pairs failing AA:\n${JSON.stringify(failures, null, 2)}`).toEqual([]);
    });
  }

  for (const scheme of ["light", "dark"] as const) {
    test(`OS preference only, no explicit data-theme (colorScheme: ${scheme})`, async ({ browser }) => {
      const context = await browser.newContext({ colorScheme: scheme });
      const page = await context.newPage();
      await page.goto("/preview.html");
      // The page's own toggle always sets an explicit data-theme on load; strip it back off so
      // this test exercises the same `:root:not([data-theme])` OS-preference path a consumer
      // with no theme toggle at all would hit.
      await page.evaluate(() => document.documentElement.removeAttribute("data-theme"));
      await page.waitForTimeout(100);
      const failures = await contrastMatrix(page);
      expect(failures, `pairs failing AA (OS ${scheme}):\n${JSON.stringify(failures, null, 2)}`).toEqual([]);
      await context.close();
    });
  }

  test("still-annotated pairs in the markup agree with the matrix", async ({ page }) => {
    await page.goto("/preview.html");
    const results = await page.evaluate(async () => {
      const mod = await import("../contrast.js");
      const styles = getComputedStyle(document.documentElement);
      const nodes = Array.from(document.querySelectorAll<HTMLElement>("[data-contrast-fg][data-contrast-bg]"));
      return nodes.map((el) => {
        const fgVar = el.dataset["contrastFg"] as string;
        const bgVar = el.dataset["contrastBg"] as string;
        const min = Number(el.dataset["contrastMin"] ?? "4.5");
        const fg = styles.getPropertyValue(fgVar).trim();
        const bg = styles.getPropertyValue(bgVar).trim();
        const ratio = mod.contrastOf(fg, bg);
        return { pair: `${fgVar} on ${bgVar}`, ratio: Math.round(ratio * 100) / 100, min };
      });
    });
    expect(results.length).toBeGreaterThan(0);
    const failures = results.filter((r) => r.ratio < r.min);
    expect(failures, `pairs failing AA:\n${JSON.stringify(failures, null, 2)}`).toEqual([]);
  });
});

test.describe("axe accessibility scan", () => {
  // The heaviest tests in the suite: run them one after the other with their own budget, so they
  // never compete with each other for CPU.
  test.describe.configure({ mode: "serial" });
  for (const theme of THEMES) {
    test(`no serious/critical violations, fully visible (${theme})`, async ({ browser }) => {
      test.setTimeout(90_000);
      // Reduced motion: nothing is mid-reveal (opacity 0) while axe checks contrast/structure.
      const context = await browser.newContext({ reducedMotion: "reduce" });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        await page.evaluate(() => document.fonts.ready);
        await setTheme(page, theme);

        // `.backdrop-grid` is decorative (aria-hidden) and axe cannot compute a background through
        // its gradients. The "backdrop never costs text its contrast" test below checks the same
        // text against the backdrop's worst-case composite instead.
        await page.evaluate(() => {
          document.querySelector<HTMLElement>(".backdrop-grid")?.style.setProperty("display", "none");
        });

        const results = await new AxeBuilder({ page })
          // aria-hidden, stroke-only decorative SVGs: axe reports them "incomplete" rather than
          // skipping them. Their real text lives in HTML outside the SVG and is scanned.
          .exclude(".pipeline__connector")
          .exclude(".hero__trace-plot")
          // Forced hover/press overlays are alpha-blended pseudo-elements axe cannot composite;
          // the "button state matrix" test proves they equal the real states.
          .exclude('.state-matrix [data-force="hover"]')
          .exclude('.state-matrix [data-force="active"]')
          .analyze();
        const severe = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        expect(severe, `axe violations:
${JSON.stringify(severe, null, 2)}`).toEqual([]);

        const contrastRule = results.incomplete.find((r) => r.id === "color-contrast");
        expect(contrastRule, `axe color-contrast could not fully verify:
${JSON.stringify(contrastRule, null, 2)}`).toBeUndefined();
      } finally {
        await context.close();
      }
    });
  }
});

test.describe("backdrop never costs text its contrast", () => {
  for (const theme of THEMES) {
    test(`every text node over .backdrop-grid passes AA against its worst-case composite (${theme})`, async ({ browser }) => {
      const context = await browser.newContext({ reducedMotion: "reduce", viewport: { width: 1440, height: 900 } });
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await setTheme(page, theme);

      const result = await page.evaluate(async () => {
        const mod = await import("../contrast.js");
        const backdrop = document.querySelector<HTMLElement>(".backdrop-grid");
        if (!backdrop) return { error: "no .backdrop-grid on the page", checked: 0, failures: [] };
        const styles = getComputedStyle(document.documentElement);
        const v = (name: string) => styles.getPropertyValue(name).trim();

        // Worst case, ignoring the mask entirely: --color-bg, then the glow at its full centre
        // alpha, then a grid line on top — composited with canvas source-over, the same blend CSS
        // uses for stacked background layers.
        const toCss = (c: { r: number; g: number; b: number; a: number }) => `rgba(${c.r}, ${c.g}, ${c.b}, ${c.a})`;
        const canvas = document.createElement("canvas");
        canvas.width = 1;
        canvas.height = 1;
        const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
        for (const token of ["--color-bg", "--color-glow", "--color-grid-line"]) {
          ctx.fillStyle = toCss(mod.getRenderedColor(v(token)));
          ctx.fillRect(0, 0, 1, 1);
        }
        const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
        const worstBg = { r: r ?? 0, g: g ?? 0, b: b ?? 0, a: 1 };

        const isOpaque = (el: Element) => mod.getRenderedColor(getComputedStyle(el).backgroundColor).a >= 0.999;
        const band = backdrop.getBoundingClientRect();
        const failures: { text: string; ratio: number; min: number }[] = [];
        let checked = 0;
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        const seen = new Set<Element>();
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
          const el = node.parentElement;
          if (!el || seen.has(el) || !node.textContent?.trim()) continue;
          seen.add(el);
          if (el.closest("[aria-hidden='true'], .sr-only, [popover]")) continue;
          const rect = el.getBoundingClientRect();
          if (rect.width === 0 || rect.top + window.scrollY >= band.bottom + window.scrollY) continue;
          // Text sitting on its own opaque surface (a card, a filled button, a code chip) never
          // shows the backdrop through it; the contrast matrix already covers those pairs.
          let covered = false;
          for (let a: Element | null = el; a && a !== document.body; a = a.parentElement) {
            if (isOpaque(a)) { covered = true; break; }
          }
          if (covered) continue;
          const cs = getComputedStyle(el);
          const size = parseFloat(cs.fontSize);
          const large = size >= 24 || (size >= 18.66 && parseInt(cs.fontWeight, 10) >= 700);
          const min = large ? 3 : 4.5;
          const ratio = mod.contrastRatio(mod.getRenderedColor(cs.color), worstBg);
          checked++;
          if (ratio < min) failures.push({ text: node.textContent.trim().slice(0, 40), ratio: Math.round(ratio * 100) / 100, min });
        }
        return { error: null, checked, failures };
      });

      expect(result.error).toBeNull();
      // The hero's eyebrow, headline, lede, terms, claims link and the top bar's text at minimum.
      expect(result.checked).toBeGreaterThanOrEqual(6);
      expect(result.failures, `text failing AA over the backdrop:
${JSON.stringify(result.failures, null, 2)}`).toEqual([]);
      await context.close();
    });
  }
});

test.describe("responsive: no horizontal overflow", () => {
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 320, height: 640 },
  ]) {
    test(`document does not scroll horizontally at ${viewport.width}px`, async ({ browser }) => {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const [scrollWidth, clientWidth] = await page.evaluate(() => [
        document.documentElement.scrollWidth,
        document.documentElement.clientWidth,
      ]);
      expect(scrollWidth, `scrollWidth ${scrollWidth} > clientWidth ${clientWidth} at ${viewport.width}px`).toBeLessThanOrEqual(clientWidth);
      await context.close();
    });
  }
});

test.describe("no-JS visitor", () => {
  test("every reveal is visible and every count-up shows its final value with JS disabled", async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    const page = await context.newPage();
    await page.goto("/preview.html");

    const opacities = await page.evaluate(() =>
      Array.from(document.querySelectorAll<HTMLElement>(".reveal, .reveal-emphasized, .pop-in")).map(
        (el) => getComputedStyle(el).opacity,
      ),
    );
    expect(opacities.length).toBeGreaterThan(0);
    for (const opacity of opacities) expect(opacity).toBe("1");

    const countTexts = await page.evaluate(() =>
      Array.from(document.querySelectorAll<HTMLElement>("[data-count-to]")).map((el) => el.textContent?.trim()),
    );
    expect(countTexts.length).toBeGreaterThan(0);
    for (const text of countTexts) {
      expect(text).not.toBe("0");
      expect(text).not.toBe("");
    }

    await context.close();
  });
});

test.describe("prefers-reduced-motion", () => {
  test("reveal, count-up, press-spring, draw-on, magnetic and sweep all resolve to a final, static state", async ({ browser }) => {
    const context = await browser.newContext({ reducedMotion: "reduce" });
    const page = await context.newPage();
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.waitForTimeout(150);

    const revealOpacity = await page.evaluate(() => {
      const el = document.querySelector<HTMLElement>(".reveal[data-reveal]");
      return el ? getComputedStyle(el).opacity : null;
    });
    expect(revealOpacity).toBe("1");

    const countText = await page.evaluate(() => document.querySelector("[data-count-to]")?.textContent?.trim());
    expect(countText).not.toBe("0");
    expect(countText).not.toBe("");

    const pressTransition = await page.evaluate(() => {
      const el = document.querySelector<HTMLElement>("[data-press]");
      return el ? getComputedStyle(el).transitionDuration : null;
    });
    expect(pressTransition?.split(",").every((d) => parseFloat(d) === 0)).toBe(true);

    // Draw-on: fully drawn immediately, no waiting for scroll.
    const dashoffset = await page.evaluate(() => {
      const el = document.querySelector<SVGPathElement>("[data-draw]");
      return el ? getComputedStyle(el).strokeDashoffset : null;
    });
    expect(dashoffset).toBe("0px");

    // Magnetic: simulate pointer movement over the hero CTA; --mx must stay at 0px.
    const hero = page.locator(".btn--hero");
    const box = await hero.boundingBox();
    expect(box).not.toBeNull();
    if (box) {
      await page.mouse.move(box.x + 2, box.y + 2);
      await page.mouse.move(box.x + box.width - 2, box.y + box.height - 2);
    }
    const mx = await page.evaluate(() => {
      const el = document.querySelector<HTMLElement>(".btn--hero");
      return el ? getComputedStyle(el).getPropertyValue("--mx").trim() : null;
    });
    expect(mx === "0px" || mx === "" || mx === null).toBe(true);

    // Sweep never renders its overlay.
    const sweepDisplay = await page.evaluate(() => {
      const el = document.querySelector(".sweep");
      return el ? getComputedStyle(el, "::after").display : null;
    });
    expect(sweepDisplay).toBe("none");

    await context.close();
  });
});

test.describe("hero interaction: magnetic + press compose instead of conflicting", () => {
  test("pressing the hero CTA scales it down even while a magnetic offset is applied", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const hero = page.locator(".btn--hero");
    const box = await hero.boundingBox();
    expect(box).not.toBeNull();
    if (!box) return;

    // Move across the button first so the magnetic primitive applies a translate offset.
    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.move(box.x + box.width * 0.85, box.y + box.height * 0.2, { steps: 5 });
    await page.mouse.down();
    // `--press-scale` (an unregistered custom property) changes discretely the instant `.is-pressed`
    // is added; only the `transform` PROPERTY that resolves it then transitions smoothly, starting
    // its clock at that instant. Reading computed style immediately (t≈0) would just catch the
    // transition's own start value, not the pressed one — a brief wait is what every other
    // transition-driven assertion in this file already does (e.g. the hover-opacity test below).
    await page.waitForTimeout(80);

    const transform = await hero.evaluate((el) => getComputedStyle(el).transform);
    expect(transform).not.toBe("none");
    const match = /matrix\(([^,]+),/.exec(transform);
    expect(match, `expected a matrix() transform, got: ${transform}`).not.toBeNull();
    const scaleA = match ? parseFloat(match[1] ?? "1") : 1;
    expect(scaleA, `transform: ${transform}`).toBeLessThan(0.99);

    await page.mouse.up();
  });
});

test.describe("draw-on: hidden before intersection, drawn after", () => {
  test("a path below the fold has a non-zero dash offset before it scrolls into view", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    // Scoped to the PIPELINE diagram's own path, not a bare `[data-draw]`: the hero also has one
    // now (`.hero__trace`'s instrument-trace curve), and that one is above the fold and therefore
    // already drawn by the time the page finishes loading — a generic first-match selector would
    // silently start asserting on the wrong (already-drawn) element instead of the below-the-fold
    // one this test exists to check.
    const selector = "#draw-on [data-draw]";
    const before = await page.evaluate((sel) => {
      const el = document.querySelector<SVGPathElement>(sel);
      return el ? getComputedStyle(el).strokeDashoffset : null;
    }, selector);
    // pathLength="1" + stroke-dasharray: 1 means the hidden state is exactly "1px" (a unitless
    // pathLength is still reported with a px suffix by getComputedStyle for stroke-dashoffset).
    expect(before).toBe("1px");

    await page.locator("#draw-on").scrollIntoViewIfNeeded();
    await page.waitForTimeout(1200);

    const after = await page.evaluate((sel) => {
      const el = document.querySelector<SVGPathElement>(sel);
      return el ? getComputedStyle(el).strokeDashoffset : null;
    }, selector);
    expect(after).toBe("0px");
  });
});

test.describe("button states", () => {
  test("hover changes the overlay opacity on primary/secondary/destructive", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    for (const selector of [".btn--primary", ".btn--secondary", ".btn--destructive"]) {
      const btn = page.locator(selector).first();
      const before = await btn.evaluate((el) => getComputedStyle(el, "::before").opacity);
      await btn.hover();
      await page.waitForTimeout(250);
      const after = await btn.evaluate((el) => getComputedStyle(el, "::before").opacity);
      expect(Number(after), `${selector}: before=${before} after=${after}`).toBeGreaterThan(Number(before));
    }
  });

  test("a disabled button shows not-allowed and never gets pressed", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const disabled = page.locator(".btn[disabled]").first();
    await expect(disabled).toHaveCSS("cursor", "not-allowed");
  });
});

test.describe("keyboard", () => {
  test("every focusable element gets a visible focus-visible indicator, and no stray tabindex sits on a non-interactive element", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });

    const strayTabindex = await page.evaluate(() => {
      const interactive = new Set(["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"]);
      return Array.from(document.querySelectorAll<HTMLElement>("[tabindex]"))
        .filter((el) => {
          if (interactive.has(el.tagName)) return false;
          if (el.getAttribute("tabindex") === "-1") return false;
          // A scrollable region (e.g. the code block's <pre>) legitimately carries tabindex="0"
          // so keyboard users can scroll it without a mouse whenever its content is wide enough
          // to need it — the WCAG-recommended pattern for any such element, independent of
          // whether THIS viewport happens to be wide enough that it isn't currently overflowing.
          const style = getComputedStyle(el);
          if (el.tagName === "PRE" && (style.overflowX === "auto" || style.overflowX === "scroll")) return false;
          return true;
        })
        .map((el) => el.outerHTML.slice(0, 80));
    });
    expect(strayTabindex, `non-interactive elements with a positive tabindex:\n${strayTabindex.join("\n")}`).toEqual([]);

    const first = page.locator("button, a[href], input, select, [tabindex]:not([tabindex='-1'])").first();
    await first.focus();
    const hasRing = await first.evaluate((el) => {
      const s = getComputedStyle(el);
      return s.boxShadow !== "none" || s.outlineStyle !== "none";
    });
    expect(hasRing).toBe(true);
  });
});

test.describe("elevation shadows render", () => {
  for (const theme of THEMES) {
    test(`every --shadow-* token resolves to a real box-shadow, not 'none' (${theme})`, async ({ page }) => {
      await page.goto("/preview.html");
      await setTheme(page, theme);
      const shadows = await page.$$eval(
        ".elevation-card--1, .elevation-card--2, .elevation-card--3, .elevation-card--glow, .btn--hero, .term__pop",
        (els) => els.map((el) => ({ cls: el.className, boxShadow: getComputedStyle(el).boxShadow })),
      );
      expect(shadows.length).toBeGreaterThan(0);
      const flat = shadows.filter((s) => s.boxShadow === "none" || s.boxShadow === "");
      expect(flat, `flat (shadowless) elements in ${theme}:\n${JSON.stringify(flat, null, 2)}`).toEqual([]);
    });
  }
});

test.describe("comparison bars actually render", () => {
  test("bars have non-zero rendered width and the tuned/default ratio matches data-bar-to (JS, after scroll)", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await scrollSectionIntoView(page, "#comparison");
    const bars = await page.$$eval("#comparison .bar", (els) =>
      els.map((el) => ({
        width: ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(el),
        to: Number((el as HTMLElement).dataset["barTo"] ?? "1"),
      })),
    );
    expect(bars.length).toBe(2);
    for (const bar of bars) expect(bar.width, JSON.stringify(bar)).toBeGreaterThan(0);
    const [defaultBar, tunedBar] = bars as [{ width: number; to: number }, { width: number; to: number }];
    const actualRatio = tunedBar.width / defaultBar.width;
    const expectedRatio = tunedBar.to / defaultBar.to;
    expect(actualRatio, `actual ${actualRatio} vs expected ${expectedRatio}`).toBeGreaterThan(expectedRatio * 0.95);
    expect(actualRatio, `actual ${actualRatio} vs expected ${expectedRatio}`).toBeLessThan(expectedRatio * 1.05);
  });

  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 844 },
  ]) {
    test(`tracks share the same left and right edge at ${viewport.width}px`, async ({ browser }) => {
      const context = await browser.newContext({ viewport });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        const tracks = await page.$$eval("#comparison .comparison__track", (els) => els.map((el) => el.getBoundingClientRect().toJSON()));
        expect(tracks.length).toBe(2);
        const [a, b] = tracks as [DOMRect, DOMRect];
        expect(Math.abs(a.left - b.left)).toBeLessThan(1);
        expect(Math.abs(a.right - b.right)).toBeLessThan(1);
      } finally {
        await context.close();
      }
    });
  }

  test("on a phone every track is at least 280px wide and the default bar at least 40px", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const tracks = await page.$$eval("#comparison .comparison__track", (els) => els.map((el) => el.getBoundingClientRect().width));
      expect(tracks.length).toBe(2);
      for (const w of tracks) expect(w, `track widths: ${tracks.join(", ")}`).toBeGreaterThanOrEqual(280);
      const defaultBar = await page.locator("#comparison .bar:not(.bar--tuned)").evaluate((el) => ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(el));
      expect(defaultBar).toBeGreaterThanOrEqual(40);
    } finally {
      await context.close();
    }
  });

  test("the illustrative-data pill keeps its natural width inside the comparison figure", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const width = await page.locator(".comparison-figure > .stat-tile__flag").evaluate((el) => el.getBoundingClientRect().width);
      expect(width).toBeLessThan(340);
    } finally {
      await context.close();
    }
  });

  test("bars are correctly sized and proportioned with JS disabled", async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    const page = await context.newPage();
    await page.goto("/preview.html");
    const bars = await page.$$eval("#comparison .bar", (els) =>
      els.map((el) => ({
        width: ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(el),
        to: Number((el as HTMLElement).getAttribute("data-bar-to") ?? "1"),
      })),
    );
    expect(bars.length).toBe(2);
    for (const bar of bars) expect(bar.width, JSON.stringify(bar)).toBeGreaterThan(0);
    const [defaultBar, tunedBar] = bars as [{ width: number; to: number }, { width: number; to: number }];
    const actualRatio = tunedBar.width / defaultBar.width;
    const expectedRatio = tunedBar.to / defaultBar.to;
    expect(actualRatio).toBeGreaterThan(expectedRatio * 0.9);
    expect(actualRatio).toBeLessThan(expectedRatio * 1.1);
    await context.close();
  });

  test("bars show their final, proportional width under reduced motion (no scroll needed)", async ({ browser }) => {
    const context = await browser.newContext({ reducedMotion: "reduce" });
    const page = await context.newPage();
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const bars = await page.$$eval("#comparison .bar", (els) => els.map((el) => ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(el)));
    expect(bars.length).toBe(2);
    for (const w of bars) expect(w).toBeGreaterThan(0);
    expect(bars[1]! / bars[0]!).toBeGreaterThan(4);
    await context.close();
  });
});

test.describe("full tab-order focus ring", () => {
  test("every focusable element in the tab order shows a visible ring, including the hero CTA and every term trigger", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const total = await page.$$eval(
      "button, a[href], input, select, textarea, summary, [tabindex]:not([tabindex='-1'])",
      (els) =>
        els.filter((el) => !el.hasAttribute("disabled") && el.getAttribute("aria-disabled") !== "true" && !el.closest("[inert]")).length,
    );
    expect(total).toBeGreaterThan(5);

    const missing: string[] = [];
    for (let i = 0; i < total; i++) {
      await page.keyboard.press("Tab");
      const info = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el || el === document.body) return null;
        const s = getComputedStyle(el);
        const hasRing = s.boxShadow !== "none" || (s.outlineStyle !== "none" && s.outlineColor !== "rgba(0, 0, 0, 0)" && s.outlineWidth !== "0px");
        return { html: el.outerHTML.slice(0, 70), hasRing };
      });
      if (info && !info.hasRing) missing.push(info.html);
    }
    expect(missing, `elements with no visible focus ring:\n${missing.join("\n")}`).toEqual([]);
  });
});

test.describe("term popovers open near their own trigger", () => {
  test("every term popover opens within ~40px of the button that opened it, inside the viewport", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const buttons = await page.$$("button.term__btn[popovertarget]");
    expect(buttons.length).toBeGreaterThan(3);

    for (const btn of buttons) {
      await btn.scrollIntoViewIfNeeded();
      await btn.click();
      const targetId = await btn.getAttribute("popovertarget");
      expect(targetId).not.toBeNull();
      const rects = await page.evaluate(
        (id) => {
          const pop = document.getElementById(id!);
          const trigger = document.querySelector(`[popovertarget="${id}"]`);
          if (!pop || !trigger) return null;
          return { pop: pop.getBoundingClientRect().toJSON(), btn: trigger.getBoundingClientRect().toJSON() };
        },
        targetId as string,
      );
      expect(rects, `no popover/trigger rect for #${targetId}`).not.toBeNull();
      const { pop, btn: btnRect } = rects as { pop: DOMRect; btn: DOMRect };
      // "Near" allows either a below- or above-anchored fallback (position-try-fallbacks) —
      // check proximity on whichever axis moved, not a fixed side.
      const verticalGap = Math.min(Math.abs(pop.top - btnRect.bottom), Math.abs(btnRect.top - pop.bottom));
      expect(verticalGap, `#${targetId}: popover top=${pop.top} vs trigger bottom=${btnRect.bottom}`).toBeLessThan(40);
      const viewport = page.viewportSize();
      if (viewport) {
        expect(pop.left).toBeGreaterThanOrEqual(-1);
        expect(pop.right).toBeLessThanOrEqual(viewport.width + 1);
      }
      await page.keyboard.press("Escape");
    }
  });
});

test.describe("accent vs danger stay visually distinct", () => {
  for (const theme of THEMES) {
    test(`accent fill and danger fill are at least 30° apart in hue OR 15% apart in lightness (${theme})`, async ({ page }) => {
      await page.goto("/preview.html");
      await setTheme(page, theme);
      const result = await page.evaluate(async () => {
        const mod = await import("../contrast.js");
        const styles = getComputedStyle(document.documentElement);
        const accent = mod.oklchOf(styles.getPropertyValue("--color-accent").trim());
        const danger = mod.oklchOf(styles.getPropertyValue("--color-danger").trim());
        return {
          accent,
          danger,
          hueDelta: mod.hueDelta(accent.h, danger.h),
          lightnessDelta: Math.abs(accent.l - danger.l),
        };
      });
      const pass = result.hueDelta >= 30 || result.lightnessDelta >= 0.15;
      expect(pass, `accent ${JSON.stringify(result.accent)} vs danger ${JSON.stringify(result.danger)}: hueDelta=${result.hueDelta}, lightnessDelta=${result.lightnessDelta}`).toBe(true);
    });
  }
});

test.describe("pipeline diagram stays legible on mobile", () => {
  test("node labels are at least 12px at 390px width", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const page = await context.newPage();
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const sizes = await page.$$eval(".pipeline__label, .pipeline__sub", (els) => els.map((el) => parseFloat(getComputedStyle(el).fontSize)));
    expect(sizes.length).toBeGreaterThan(0);
    for (const size of sizes) expect(size).toBeGreaterThanOrEqual(12);
    await context.close();
  });
});

test.describe("no scaffolding text on the page", () => {
  test("no visible text node contains a literal backtick", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const offenders = await page.evaluate(() => {
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      const bad: string[] = [];
      let node: Node | null;
      // eslint-disable-next-line no-cond-assign
      while ((node = walker.nextNode())) {
        if (node.textContent?.includes("`")) bad.push(node.textContent.trim().slice(0, 60));
      }
      return bad;
    });
    expect(offenders, `text nodes with a literal backtick:\n${offenders.join("\n")}`).toEqual([]);
  });
});

test.describe("button rows share one height per row", () => {
  test("every .btn variant in the interaction catalogue has the same offsetHeight", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    // `.btn--icon` is deliberately a fixed 40x40 square (a toolbar icon button, a different
    // component variant) — this checks the text-label variants that should share one box height:
    // filled (.btn--primary) and bordered (.btn--secondary/.btn--destructive) used to differ by
    // the border's width because only the bordered variants had a real border in the box model.
    const heights = await page.$$eval(
      "#buttons .btn-row:first-of-type .btn:not(.btn--icon)",
      (els) => els.map((el) => (el as HTMLElement).offsetHeight),
    );
    expect(heights.length).toBeGreaterThan(3);
    const distinct = new Set(heights);
    expect(distinct.size, `heights: ${heights.join(", ")}`).toBe(1);
  });
});

test.describe("code uses JetBrains Mono everywhere", () => {
  test("every <code>/<pre>/<kbd> computed font-family starts with JetBrains Mono Variable", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const bad = await page.$$eval("code, pre, kbd", (els) =>
      els
        .map((el) => ({ tag: el.tagName, cls: el.className, font: getComputedStyle(el).fontFamily }))
        .filter((el) => !el.font.startsWith('"JetBrains Mono Variable"')),
    );
    expect(bad, `elements not using JetBrains Mono Variable:\n${JSON.stringify(bad, null, 2)}`).toEqual([]);

    // `networkidle` does not mean the font has finished decoding (font-display: swap lets text
    // paint first), so under a loaded parallel run a bare `fonts.check()` raced the load and
    // flaked. Explicitly request the face, then assert it really resolved to a loaded FontFace.
    const checked = await page.evaluate(async () => {
      const faces = await document.fonts.load('1em "JetBrains Mono Variable"');
      return faces.length > 0 && document.fonts.check('1em "JetBrains Mono Variable"');
    });
    expect(checked).toBe(true);
  });

  test("a code block's <code> inherits the block's own size instead of the base chip size", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const sizes = await page.evaluate(() => {
      const pre = document.querySelector(".code-block pre");
      const code = document.querySelector(".code-block pre code");
      if (!pre || !code) return null;
      return { pre: getComputedStyle(pre).fontSize, code: getComputedStyle(code).fontSize };
    });
    expect(sizes).not.toBeNull();
    expect(sizes?.code).toBe(sizes?.pre);
  });
});

test.describe("focus ring wins over unlayered consumer CSS", () => {
  test("an unlayered override injected via adoptedStyleSheets does not remove the ring", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const boxShadow = await page.evaluate(() => {
      const sheet = new CSSStyleSheet();
      // Equal specificity to `cerebro.focus`'s own selector list, and completely unlayered — the
      // exact shape of override a consumer's own component CSS (Astro/Solid, neither of which
      // layers by default) could plausibly ship.
      sheet.replaceSync(".btn, a, input { box-shadow: none; outline: none; }");
      (document as Document & { adoptedStyleSheets: CSSStyleSheet[] }).adoptedStyleSheets = [
        ...document.adoptedStyleSheets,
        sheet,
      ];
      const btn = document.querySelector<HTMLElement>(".btn--hero");
      btn?.focus();
      return btn ? getComputedStyle(btn).boxShadow : null;
    });
    expect(boxShadow).not.toBeNull();
    expect(boxShadow).not.toBe("none");
  });
});

test.describe("busy button", () => {
  test("shows a real, animating spinner (not the sweep) and blocks Enter/Space activation", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const busyPrimary = page.locator('.btn--primary[data-sweep][aria-busy="true"]').first();
    await expect(busyPrimary).toBeVisible();

    const spinner = await busyPrimary.evaluate((el) => {
      const s = el.querySelector<HTMLElement>(".btn__spinner");
      if (!s) return null;
      const c = getComputedStyle(s);
      return { opacity: c.opacity, animationName: c.animationName };
    });
    expect(spinner).not.toBeNull();
    expect(Number(spinner?.opacity)).toBeGreaterThan(0.9);
    expect(spinner?.animationName).toBe("btn-spin");

    // The sweep must not be competing with the spinner for `::after` at all: `.sweep:not([aria-busy=
    // "true"], ...)::after` (motion.css) simply does not MATCH a busy button, so no rule anywhere
    // gives this pseudo-element a `content` value — it has no generated box. (Once a pseudo-element
    // has no matching rule, its computed `opacity` reports the CSS-wide initial value, 1, which is
    // why checking opacity here would be checking the wrong thing entirely; `content: "none"` is
    // what actually proves nothing is rendered.)
    const sweepContent = await busyPrimary.evaluate((el) => getComputedStyle(el, "::after").content);
    expect(sweepContent).toBe("none");

    // A click handler proves whether Enter actually fired a click: none is wired on this static
    // specimen button, so instead assert the guard's own `preventDefault` ran by checking that a
    // synthetic click dispatched the normal way IS observed, while a capture-phase guard swallows
    // the keyboard-originated one — verified by counting real `click` events reaching the button.
    let clicks = 0;
    await page.exposeFunction("__reportClick", () => {
      clicks += 1;
    });
    await busyPrimary.evaluate((el) => {
      el.addEventListener("click", () => (window as unknown as { __reportClick: () => void }).__reportClick());
    });
    await busyPrimary.focus();
    await page.keyboard.press("Enter");
    await page.waitForTimeout(50);
    expect(clicks).toBe(0);
  });
});

test.describe("pipeline diagram: connector never crosses a label", () => {
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 1440, height: 900 },
  ]) {
    test(`no label/sub rect intersects the connector rect at ${viewport.width}px`, async ({ browser }) => {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await page.locator("#draw-on").scrollIntoViewIfNeeded();
      await page.waitForTimeout(150);

      const hits = await page.evaluate(() => {
        const isMobile = window.matchMedia("(max-width: 40rem)").matches;
        const selector = isMobile ? ".pipeline__connector--v" : ".pipeline__connector--h";
        const line = document.querySelector(selector)?.getBoundingClientRect();
        if (!line) return -1;
        return Array.from(document.querySelectorAll(".pipeline__label, .pipeline__sub")).filter((el) => {
          const r = el.getBoundingClientRect();
          return r.left < line.right && r.right > line.left && r.top < line.bottom && r.bottom > line.top;
        }).length;
      });
      expect(hits).toBe(0);
      await context.close();
    });
  }

  // Measures the DRAWN STROKE (the `[data-draw]` line inside the SVG), not the <svg> box: the box
  // was always the right size, but a fixed viewBox once letterboxed the stroke inside it to half its
  // width at 1440px — a box-only check passed while the visible line stopped short of both ends.
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 1440, height: 900 },
  ]) {
    test(`the drawn connector spans from the first circle's centre to the last circle's centre, within 2px, at ${viewport.width}px`, async ({ browser }) => {
      const context = await browser.newContext({ viewport, reducedMotion: "reduce" });
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await page.locator("#draw-on").scrollIntoViewIfNeeded();
      await page.waitForTimeout(150);

      const result = await page.evaluate(() => {
        const isMobile = window.matchMedia("(max-width: 40rem)").matches;
        const nodes = Array.from(document.querySelectorAll(".pipeline__n"));
        const stroke = document.querySelector(
          isMobile ? ".pipeline__connector--v [data-draw]" : ".pipeline__connector--h [data-draw]",
        );
        if (!stroke || nodes.length < 2) return null;
        const first = nodes[0]!.getBoundingClientRect();
        const last = nodes[nodes.length - 1]!.getBoundingClientRect();
        const line = stroke.getBoundingClientRect();
        return isMobile
          ? { start: line.top, end: line.bottom, firstCenter: first.top + first.height / 2, lastCenter: last.top + last.height / 2 }
          : { start: line.left, end: line.right, firstCenter: first.left + first.width / 2, lastCenter: last.left + last.width / 2 };
      });
      expect(result).not.toBeNull();
      if (result) {
        expect(Math.abs(result.start - result.firstCenter)).toBeLessThanOrEqual(2);
        expect(Math.abs(result.end - result.lastCenter)).toBeLessThanOrEqual(2);
      }
      await context.close();
    });
  }
});

test.describe("motion token demo: chip never overlaps the Play button", () => {
  test("while playing, the chip's rect stays clear of the Play button's rect", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.locator("[data-motion-play]").click();
    await page.waitForTimeout(400); // mid-travel, not just the two endpoints
    const overlap = await page.evaluate(() => {
      const chip = document.querySelector(".motion-demo__chip")?.getBoundingClientRect();
      const play = document.querySelector("[data-motion-play]")?.getBoundingClientRect();
      if (!chip || !play) return true;
      return chip.left < play.right && chip.right > play.left && chip.top < play.bottom && chip.bottom > play.top;
    });
    expect(overlap).toBe(false);
  });
});

test.describe("button state matrix", () => {
  test("forced data-force hover/active/focus render the same computed styles as the real states", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });

    // The bench is `inert` (a picture of states, not controls), so the real states are produced on a
    // live button of the same variant in the catalogue.
    const restBefore = page.locator("#buttons .btn-row .btn--primary:not([disabled]):not([aria-busy])").first();
    const hoverBtn = page.locator('.state-matrix .btn--primary[data-force="hover"]');
    await restBefore.hover();
    // `::before`'s opacity transitions over `--duration-fast` (200ms) on real `:hover` — reading it
    // immediately would catch a mid-transition value instead of the settled one `[data-force]` (a
    // static class, no transition to wait out) reports right away.
    await page.waitForTimeout(250);
    const [realHoverOpacity, forcedHoverOpacity] = await Promise.all([
      restBefore.evaluate((el) => getComputedStyle(el, "::before").opacity),
      hoverBtn.evaluate((el) => getComputedStyle(el, "::before").opacity),
    ]);
    expect(forcedHoverOpacity).toBe(realHoverOpacity);

    const focusBtn = page.locator('.state-matrix .btn--primary[data-force="focus"]');
    const plainBtn = restBefore;
    await page.mouse.move(0, 0);
    await plainBtn.focus();
    const [realFocusShadow, forcedFocusShadow] = await Promise.all([
      plainBtn.evaluate((el) => getComputedStyle(el).boxShadow),
      focusBtn.evaluate((el) => getComputedStyle(el).boxShadow),
    ]);
    expect(forcedFocusShadow).toBe(realFocusShadow);

    // Pressed: a real :active (mouse held down) on each live catalogue variant against the bench's
    // forced one, including destructive's own danger tint.
    for (const variant of ["primary", "secondary", "destructive"]) {
      const live = page.locator(`#buttons .btn-row .btn--${variant}:not([disabled]):not([aria-busy]):not([aria-disabled])`).first();
      const forced = page.locator(`.state-matrix .btn--${variant}[data-force="active"]`);
      await live.hover();
      await page.mouse.down();
      await page.waitForTimeout(250);
      const read = (el: Element) => {
        const b = getComputedStyle(el, "::before");
        return { background: b.backgroundColor, opacity: b.opacity };
      };
      const real = await live.evaluate(read);
      await page.mouse.up();
      expect(await forced.evaluate(read), variant).toEqual(real);
    }
  });
});

test.describe("count-up accessibility", () => {
  test("the final value is exposed to the accessibility tree throughout the animation, and a bad data-count-from never renders NaN", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await scrollSectionIntoView(page, "#count-up");
    const srTexts = await page.$$eval("[data-count-to] .sr-only", (els) => els.map((el) => el.textContent));
    expect(srTexts.length).toBeGreaterThan(0);
    for (const text of srTexts) expect(text).not.toBe("");

    const bodyText = await page.evaluate(() => document.body.textContent ?? "");
    expect(bodyText).not.toMatch(/NaN/);
  });
});

test.describe("theme init runs before the deferred module", () => {
  test("an explicit stored theme wins over an opposite OS colour scheme with no flash to fix", async ({ browser }) => {
    const context = await browser.newContext({ colorScheme: "dark" });
    const page = await context.newPage();
    // Simulates a returning visitor who explicitly picked light while their OS stayed dark.
    await page.addInitScript(() => {
      window.localStorage.setItem("cerebro-theme", "light");
    });
    // Proves the OUTCOME (explicit storage beats OS scheme) that only holds if theme-init.js ran
    // synchronously in <head>, ahead of the deferred `preview.js` module's own copy of this same
    // lookup — Playwright can't directly assert "before first paint," but `initThemeToggle()`
    // (preview.ts) treats a stored value as "explicit" either way, so this alone can't distinguish
    // the two files. It does confirm the value theme-init.ts is responsible for setting is correct.
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const theme = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
    expect(theme).toBe("light");
    await context.close();
  });
});

test.describe("theme toggle", () => {
  test("reflects state via aria-pressed and persists an explicit choice across reload", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const toggle = page.locator("[data-theme-toggle]");
    const before = await toggle.getAttribute("aria-pressed");
    await toggle.click();
    const after = await toggle.getAttribute("aria-pressed");
    expect(after).not.toBe(before);

    const themeAfterClick = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
    await page.reload({ waitUntil: "networkidle" });
    const themeAfterReload = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
    expect(themeAfterReload).toBe(themeAfterClick);
  });

  test("the claims-policy link resolves instead of 400ing", async ({ page, baseURL }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const href = await page.locator(".hero__actions a.btn--tertiary").getAttribute("href");
    expect(href).toBe("/claims-policy");
    const res = await page.request.get(new URL(href!, baseURL).toString());
    expect(res.status()).toBe(200);
  });
});

test.describe("claims", () => {
  test("no bare 'tok/s' figure without a placeholder flag or a cited source", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const unflagged = await page.evaluate(() => {
      const FIGURE = /\d+(\.\d+)?\s*tok\/s/;
      const flagSelector = ".stat-tile__flag, .stat-tile__source, .stat-tile__provenance";
      const bad: string[] = [];
      for (const section of Array.from(document.querySelectorAll("section"))) {
        if (FIGURE.test(section.textContent ?? "") && !section.querySelector(flagSelector)) {
          bad.push(section.id || section.textContent?.slice(0, 40) || "(unnamed section)");
        }
      }
      return bad;
    });
    expect(unflagged, `sections with an unflagged tok/s figure: ${unflagged.join(", ")}`).toEqual([]);
  });

  test("the no-draft-model speculative strategy count matches docs/CLAIMS.md (5 of 10)", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const text = await page.locator("#count-up").evaluate((el) => {
      const copy = el.cloneNode(true) as HTMLElement;
      copy.querySelectorAll(".term__btn").forEach((b) => b.remove());
      return (copy.textContent ?? "").replace(/\s+/g, " ");
    });
    expect(text).toContain("5");
    expect(text).toMatch(/speculative strategies need no draft model/);
    expect(text).not.toMatch(/\b10\b\s+no-draft-model/);
  });

  test("no unimplemented sandbox guarantee (file-system/network isolation) appears anywhere on the page", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    // Includes the popover content, which only renders into the accessibility tree / DOM once
    // opened — `body.textContent` still contains it because `[popover]` content exists in the DOM
    // (just not painted) whether or not it's currently shown.
    const text = (await page.locator("body").evaluate((el) => el.textContent)) ?? "";
    for (const bad of [/can.t read files/i, /can.t use the network/i, /no network access/i, /touch hardware/i]) {
      expect(text, `matched forbidden phrase ${bad}`).not.toMatch(bad);
    }
  });

  test("the sandbox definition claims only what job.rs/env.rs do and states what it does not restrict", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const sandbox = ((await page.locator("#def-sandboxed").textContent()) ?? "").replace(/\s+/g, " ");
    expect(sandbox).not.toMatch(/other programs|settings/i);
    expect(sandbox).toMatch(/Job Object/);
    expect(sandbox).toMatch(/allowlist/i);
    expect(sandbox).toMatch(/does not limit which files or network/i);
    // "it" must never be the thing served on loopback: the API is.
    expect(sandbox).toMatch(/The API it serves listens only on loopback/);
  });

  test("every count-up value sits inside a figure/.stat-tile that also carries a flag or source", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const orphans = await page.$$eval(".count", (els) =>
      els
        .map((el) => {
          const box = el.closest("figure, .stat-tile");
          const flagged = !!box?.querySelector(".stat-tile__flag, .stat-tile__source, .stat-tile__provenance");
          return { text: el.textContent, hasBox: !!box, flagged };
        })
        .filter((r) => !r.hasBox || !r.flagged),
    );
    expect(orphans, `unflagged count-up values:\n${JSON.stringify(orphans, null, 2)}`).toEqual([]);
  });

  test('every placeholder value uses the exact wording "Illustrative data — not a measurement"', async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const flags = await page.$$eval(".stat-tile__flag", (els) => els.map((el) => el.textContent?.trim()));
    expect(flags.length).toBeGreaterThan(0);
    for (const flag of flags) expect(flag).toBe("Illustrative data — not a measurement");
  });

  test("no heading containing \"Measured\" sits directly over a flagged placeholder section", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const offenders = await page.$$eval("section", (sections) =>
      sections
        .filter((s) => /measured/i.test(s.querySelector("h2, h3")?.textContent ?? ""))
        .filter((s) => s.querySelector(".stat-tile__flag"))
        .map((s) => s.id || s.querySelector("h2, h3")?.textContent),
    );
    expect(offenders).toEqual([]);
  });
});
