import { test, expect, type Page } from "@playwright/test";

/**
 * Motion gate, layout stability, content rules and CSP tightness. Split from preview.spec.ts so
 * each file stays readable.
 */

const THEMES = ["dark", "light"] as const;

async function scrollSectionIntoView(page: Page, selector: string) {
  await page.locator(selector).scrollIntoViewIfNeeded();
  await page.waitForTimeout(1000);
}

interface Sample {
  t: number;
  text: string;
  opacity: number;
}

test.describe("motion gate: nothing flashes its final state before animating", () => {
  test("the hero count-up never shows its final value and then a smaller one", async ({ page }) => {
    await page.addInitScript(() => {
      const samples: Sample[] = [];
      (window as unknown as { __samples: Sample[] }).__samples = samples;
      const effectiveOpacity = (el: Element) => {
        let o = 1;
        for (let n: Element | null = el; n; n = n.parentElement) o *= Number(getComputedStyle(n).opacity);
        return o;
      };
      const tick = () => {
        const count = document.querySelector(".hero .count");
        if (count) {
          const visible = count.querySelector('[aria-hidden="true"]') ?? count;
          samples.push({ t: performance.now(), text: (visible.textContent ?? "").trim(), opacity: effectiveOpacity(visible) });
        }
        if (performance.now() < 3500) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.waitForTimeout(2500);
    const samples = await page.evaluate(() => (window as unknown as { __samples: Sample[] }).__samples);
    expect(samples.length).toBeGreaterThan(10);

    // Only frames where the number is actually visible matter: a hidden "34.0" is not a flash.
    const seen = samples.filter((s) => s.opacity > 0.01);
    expect(seen.length, "the hero number never became visible").toBeGreaterThan(0);
    for (let i = 1; i < seen.length; i++) {
      const prev = seen[i - 1]!;
      const cur = seen[i]!;
      expect(Number(cur.text), `visible value went ${prev.text} -> ${cur.text} at ${Math.round(cur.t)}ms`).toBeGreaterThanOrEqual(Number(prev.text));
    }
    const last = seen[seen.length - 1]!;
    expect(last.text).toBe("34.0");
    expect(last.opacity).toBeGreaterThan(0.99);
  });

  test("CSS failsafe: with the motion module blocked, everything is visible within 3s", async ({ page }) => {
    await page.route("**/preview.js", (route) => route.abort());
    await page.goto("/preview.html");
    const early = await page.evaluate(() => ({
      pending: document.documentElement.classList.contains("motion-pending"),
      // The hero's secondary pieces (proof row, trace panel) wait behind the gate...
      heroRevealOpacity: Array.from(document.querySelectorAll(".hero :is(.reveal, .reveal-emphasized)")).map((el) => getComputedStyle(el).opacity),
      // ...while the LCP heading, lede and CTA paint on the first frame (DESIGN.md, principle 13).
      lcpOpacity: Array.from(document.querySelectorAll(".hero__title, .hero__lede, .hero__actions")).map((el) => {
        let o = 1;
        for (let n: Element | null = el; n; n = n.parentElement) o *= Number(getComputedStyle(n).opacity);
        return o;
      }),
    }));
    expect(early.pending, "theme-init.js should set motion-pending before first paint").toBe(true);
    expect(early.heroRevealOpacity.length).toBeGreaterThanOrEqual(2);
    for (const o of early.heroRevealOpacity) expect(o).toBe("0");
    expect(early.lcpOpacity).toEqual([1, 1, 1]);

    await page.waitForTimeout(3000);
    const hidden = await page.$$eval(".reveal, .reveal-emphasized, .pop-in, [data-count-to]", (els) =>
      els.filter((el) => getComputedStyle(el).opacity !== "1").map((el) => el.className),
    );
    expect(hidden, `still hidden after the failsafe:\n${hidden.join("\n")}`).toEqual([]);
  });

  test("toggling reduced motion mid-session never re-animates a finished count-up", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    for (const selector of ["#in-context", ".result-card--measuring", ".result-card--defaults-won", "#count-up", "#comparison"]) await scrollSectionIntoView(page, selector);
    const readAll = () =>
      page.$$eval("[data-count-to]", (els) => els.map((el) => el.querySelector('[aria-hidden="true"]')?.textContent?.trim() ?? ""));
    const finals = await readAll();
    expect(finals).toEqual(["34.0", "412", "34.0", "27.5", "412", "34.0", "27.5", "31.2", "52.0", "50.4", "6.0", "34.0", "5", "6.0", "34.0"]);

    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.waitForTimeout(100);
    await page.emulateMedia({ reducedMotion: "no-preference" });
    const frames = await page.evaluate(
      () =>
        new Promise<string[][]>((resolve) => {
          const out: string[][] = [];
          const start = performance.now();
          const tick = () => {
            out.push(
              Array.from(document.querySelectorAll("[data-count-to]")).map(
                (el) => el.querySelector('[aria-hidden="true"]')?.textContent?.trim() ?? "",
              ),
            );
            if (performance.now() - start < 700) requestAnimationFrame(tick);
            else resolve(out);
          };
          requestAnimationFrame(tick);
        }),
    );
    for (const frame of frames) expect(frame).toEqual(finals);
  });
});

test.describe("layout stability", () => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 844 },
  ]) {
    for (const theme of THEMES) {
      // +0ms: fonts arrive as fast as the local server allows. +800ms: the swap lands after first
      // paint, which is what actually tests the metric-matched fallbacks.
      for (const fontDelay of [0, 800]) {
        test(`CLS under 0.02 through load and font swap at ${viewport.width}px, ${theme}, fonts +${fontDelay}ms`, async ({ browser }) => {
          const context = await browser.newContext({ viewport, colorScheme: theme });
          try {
            const page = await context.newPage();
            if (fontDelay) {
              await page.route("**/*.woff2", async (route) => {
                await new Promise((r) => setTimeout(r, fontDelay));
                await route.continue();
              });
            }
            await page.addInitScript(() => {
              const w = window as unknown as { __cls: number; __shifts: string[] };
              w.__cls = 0;
              w.__shifts = [];
              new PerformanceObserver((list) => {
                type Shift = PerformanceEntry & { value: number; hadRecentInput: boolean; sources?: Array<{ node?: Node | null }> };
                for (const entry of list.getEntries() as Shift[]) {
                  if (entry.hadRecentInput) continue;
                  w.__cls += entry.value;
                  const nodes = (entry.sources ?? []).map((s) => (s.node instanceof Element ? s.node.className || s.node.tagName : s.node?.nodeName));
                  w.__shifts.push(`${entry.value.toFixed(4)} at ${Math.round(entry.startTime)}ms: ${nodes.join(", ")}`);
                }
              }).observe({ type: "layout-shift", buffered: true });
            });
            await page.goto("/preview.html", { waitUntil: "networkidle" });
            await page.evaluate(() => document.fonts.ready);
            await page.waitForTimeout(1200);
            const { cls, shifts } = await page.evaluate(() => {
              const w = window as unknown as { __cls: number; __shifts: string[] };
              return { cls: w.__cls, shifts: w.__shifts };
            });
            expect(cls, `layout shifts:\n${shifts.join("\n")}`).toBeLessThan(0.02);
          } finally {
            await context.close();
          }
        });
      }
    }
  }
});

test.describe("content rules", () => {
  test("every hero technical term has a ? button whose popover exists", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const terms = await page.$$eval(".hero .term", (els) =>
      els.map((el) => {
        const id = el.querySelector(".term__btn")?.getAttribute("popovertarget") ?? "";
        const pop = id ? document.getElementById(id) : null;
        return { term: el.querySelector("dfn")?.textContent?.trim(), ok: !!pop?.hasAttribute("popover") && !!pop.textContent?.trim() };
      }),
    );
    expect(terms.map((t) => t.term)).toEqual(["llama.cpp", "sandboxed", "integrity-checked", "loopback", "decode", "tok/s"]);
    for (const t of terms) expect(t.ok, `${t.term} has no popover`).toBe(true);
  });

  test("the integrity definition stays inside docs/CLAIMS.md (no signature or tamper-proof claim)", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const text = (await page.locator("#def-integrity").textContent()) ?? "";
    expect(text).toMatch(/before every start/i);
    for (const bad of [/signature/i, /signed/i, /tamper-?proof/i, /guarantee/i]) expect(text).not.toMatch(bad);
  });

  test("a term popover fades in (opacity/transform only) instead of snapping", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const pop = page.locator("#def-loopback");
    const transition = await pop.evaluate((el) => getComputedStyle(el).transitionProperty);
    for (const prop of ["opacity", "transform"]) expect(transition).toContain(prop);
    expect(transition).not.toMatch(/\b(top|left|width|height|margin)\b/);
    await page.locator('[popovertarget="def-loopback"]').click();
    await page.waitForTimeout(400);
    expect(await pop.evaluate((el) => getComputedStyle(el).opacity)).toBe("1");
  });
});

test.describe("CSP is tight", () => {
  test("the policy denies by default, enforces Trusted Types, allows no data: URLs, and the page triggers no violation", async ({ page }) => {
    await page.addInitScript(() => {
      (window as unknown as { __csp: string[] }).__csp = [];
      document.addEventListener("securitypolicyviolation", (e) => {
        (window as unknown as { __csp: string[] }).__csp.push(`${e.violatedDirective} ${e.blockedURI}`);
      });
    });
    const response = await page.goto("/preview.html", { waitUntil: "networkidle" });
    const header = response?.headers()["content-security-policy"] ?? "";
    const meta = (await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content")) ?? "";
    for (const policy of [header, meta]) {
      expect(policy).toContain("default-src 'none'");
      expect(policy).toContain("require-trusted-types-for 'script'");
      expect(policy).toContain("trusted-types 'none'");
      expect(policy).not.toContain("data:");
      expect(policy).not.toContain("unsafe-");
      // Grant nothing unused: the page makes no fetch/XHR and loads no images.
      expect(policy).toContain("connect-src 'none'");
      expect(policy).toContain("img-src 'none'");
    }

    // Exercise every path that writes to the DOM, then check for violations.
    await page.locator("[data-theme-toggle]").click();
    await page.locator("#color summary").first().click();
    await page.locator("[data-motion-play]").click();
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(400);
    const violations = await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp);
    expect(violations).toEqual([]);
  });
});
