import { test, expect, type Page } from "@playwright/test";
import { execFile } from "node:child_process";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

/**
 * The shared primitives exercised the way their real consumers use them (one init call per SolidJS
 * component or Astro island, long card grids, diagrams outside a .reveal), not only the way the
 * specimen initialises everything once from `document`. Plus the generated-CSS, build-drift,
 * content and mobile-polish checks from design round 3.
 */

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const run = promisify(execFile);

/** Appends a fresh container at the end of <main> using DOM APIs only (the page enforces Trusted
 *  Types, so no HTML strings). */
async function appendContainer(page: Page, id: string): Promise<void> {
  await page.evaluate((cid) => {
    const box = document.createElement("div");
    box.id = cid;
    document.querySelector("main")?.append(box);
  }, id);
}

// ---------------------------------------------------------------------------------------------
// terms.ts
// ---------------------------------------------------------------------------------------------

test.describe("terms: separate init calls never share an anchor name", () => {
  test("two roots initialised separately get unique names and each popover opens within 40px of its own trigger", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    for (const id of ["root-a", "root-b"]) await appendContainer(page, id);
    const names = await page.evaluate(async () => {
      const mod = await import("../terms.js");
      for (const rootId of ["root-a", "root-b"]) {
        const root = document.getElementById(rootId)!;
        for (let i = 0; i < 3; i++) {
          const p = document.createElement("p");
          p.style.setProperty("margin", "160px 0");
          const term = document.createElement("span");
          term.className = "term";
          const dfn = document.createElement("dfn");
          dfn.textContent = `${rootId} term ${i}`;
          const btn = document.createElement("button");
          btn.className = "term__btn";
          btn.type = "button";
          btn.textContent = "?";
          btn.setAttribute("aria-label", `What is ${rootId} term ${i}?`);
          btn.setAttribute("popovertarget", `pop-${rootId}-${i}`);
          term.append(dfn, btn);
          p.append(term);
          const pop = document.createElement("div");
          pop.id = `pop-${rootId}-${i}`;
          pop.className = "term__pop";
          pop.setAttribute("popover", "");
          pop.textContent = `Definition ${rootId} ${i}`;
          root.append(p, pop);
        }
        mod.initTerms(root); // one call per "component"
      }
      mod.initTerms(document.getElementById("root-a")!); // re-init must keep, not renumber, names
      return Array.from(document.querySelectorAll<HTMLElement>(".term__btn[popovertarget]")).map((b) =>
        b.style.getPropertyValue("anchor-name"),
      );
    });
    expect(names.every((n) => n.startsWith("--cerebro-term-"))).toBe(true);
    expect(new Set(names).size, `anchor names: ${names.join(", ")}`).toBe(names.length);

    for (const rootId of ["root-a", "root-b"]) {
      for (let i = 0; i < 3; i++) {
        const btn = page.locator(`[popovertarget="pop-${rootId}-${i}"]`);
        await btn.scrollIntoViewIfNeeded();
        await btn.click();
        await page.waitForTimeout(250);
        const r = await page.evaluate((id) => {
          const pop = document.getElementById(id)!.getBoundingClientRect();
          const b = document.querySelector(`[popovertarget="${id}"]`)!.getBoundingClientRect();
          return { gap: Math.min(Math.abs(pop.top - b.bottom), Math.abs(b.top - pop.bottom)) };
        }, `pop-${rootId}-${i}`);
        expect(r.gap, `pop-${rootId}-${i} opened ${r.gap}px from its trigger`).toBeLessThan(40);
        await page.keyboard.press("Escape");
      }
    }

    // Cleanup removes both properties.
    const leftovers = await page.evaluate(async () => {
      const mod = await import("../terms.js");
      const root = document.createElement("div");
      const btn = document.createElement("button");
      btn.className = "term__btn";
      btn.setAttribute("popovertarget", "pop-cleanup");
      const pop = document.createElement("div");
      pop.id = "pop-cleanup";
      pop.setAttribute("popover", "");
      root.append(btn, pop);
      document.body.append(root);
      const off = mod.initTerms(root);
      off();
      return [btn.style.getPropertyValue("anchor-name"), pop.style.getPropertyValue("position-anchor")];
    });
    expect(leftovers).toEqual(["", ""]);
  });

  test("on a phone every term popover keeps a gutter from both viewport edges", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const ids = await page.$$eval("button.term__btn[popovertarget]", (els) => els.map((e) => e.getAttribute("popovertarget")!));
      expect(ids.length).toBeGreaterThan(8);
      for (const id of ids) {
        const btn = page.locator(`main [popovertarget="${id}"]`);
        await btn.scrollIntoViewIfNeeded();
        await btn.click();
        await page.waitForTimeout(250);
        const r = await page.evaluate((pid) => {
          const pop = document.getElementById(pid)!.getBoundingClientRect();
          const b = document.querySelector(`[popovertarget="${pid}"]`)!.getBoundingClientRect();
          return { left: pop.left, right: pop.right, gap: Math.min(Math.abs(pop.top - b.bottom), Math.abs(b.top - pop.bottom)) };
        }, id);
        expect(r.left, `#${id} left edge`).toBeGreaterThanOrEqual(8);
        expect(r.right, `#${id} right edge`).toBeLessThanOrEqual(390 - 8);
        expect(r.gap, `#${id} distance from trigger`).toBeLessThan(40);
        await page.keyboard.press("Escape");
      }
    } finally {
      await context.close();
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Scroll reveal
// ---------------------------------------------------------------------------------------------

test.describe("reveal: stagger is per arriving batch, capped", () => {
  test("a card that scrolls in alone has no delay and is visible within 700ms; a batch never exceeds 6 steps", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await appendContainer(page, "tall-list");
      await appendContainer(page, "small-batch");
      await page.evaluate(async () => {
        const mod = await import("../motion.js");
        const tall = document.getElementById("tall-list")!;
        const batch = document.getElementById("small-batch")!;
        for (let i = 0; i < 12; i++) {
          const card = document.createElement("div");
          card.className = "elevation-card elevation-card--1 reveal";
          card.dataset["reveal"] = "";
          card.dataset["card"] = String(i + 1);
          card.textContent = `Card ${i + 1}`;
          card.style.setProperty("height", "760px");
          card.style.setProperty("margin-bottom", "120px");
          tall.append(card);
        }
        for (let i = 0; i < 12; i++) {
          const chip = document.createElement("div");
          chip.className = "reveal";
          chip.dataset["reveal"] = "";
          chip.textContent = `Chip ${i + 1}`;
          chip.style.setProperty("height", "24px");
          batch.append(chip);
        }
        mod.initScrollReveal(tall);
        mod.initScrollReveal(batch);
      });

      // Timed inside the page (scroll to first frame at full opacity), so test-runner round trips
      // under a loaded parallel run don't count against the primitive.
      const elapsed = await page.evaluate(
        () =>
          new Promise<number>((resolve) => {
            const card = document.querySelector<HTMLElement>('[data-card="12"]')!;
            card.scrollIntoView({ block: "start", behavior: "instant" });
            const t0 = performance.now();
            const tick = () => {
              if (getComputedStyle(card).opacity === "1") resolve(performance.now() - t0);
              else if (performance.now() - t0 > 3000) resolve(Infinity);
              else requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
          }),
      );
      const delay = await page.evaluate(() => (document.querySelector('[data-card="12"]') as HTMLElement).style.getPropertyValue("--reveal-delay"));
      expect(delay).toBe("0ms");
      expect(elapsed, `card 12 took ${elapsed}ms`).toBeLessThan(700);

      await page.evaluate(() => document.getElementById("small-batch")!.scrollIntoView({ block: "center", behavior: "instant" }));
      await page.waitForTimeout(300);
      const delays = await page.$$eval("#small-batch .reveal", (els) =>
        els.map((el) => parseFloat((el as HTMLElement).style.getPropertyValue("--reveal-delay") || "0")),
      );
      expect(delays[0]).toBe(0);
      expect(Math.max(...delays), `delays: ${delays.join(", ")}`).toBeLessThanOrEqual(6 * 60);
      // Group arrival is in document order.
      for (let i = 1; i < delays.length; i++) expect(delays[i]!).toBeGreaterThanOrEqual(delays[i - 1]!);
    } finally {
      await context.close();
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Draw-on gate
// ---------------------------------------------------------------------------------------------

test.describe("draw-on: never flashes drawn before motion-ready", () => {
  test("a [data-draw] path outside any .reveal, in view at load, is undrawn in every frame before motion-ready", async ({ page }) => {
    await page.route("**/preview.html", async (route) => {
      const res = await route.fetch();
      const html = (await res.text()).replace(
        '<main class="wrap">',
        '<main class="wrap"><svg id="probe-draw-svg" viewBox="0 0 100 10" width="300" height="30" aria-hidden="true"><path id="probe-draw" data-draw pathLength="1" d="M0 5 H100" stroke="currentColor" stroke-width="2" /></svg>',
      );
      await route.fulfill({ response: res, body: html });
    });
    await page.addInitScript(() => {
      const samples: Array<{ ready: boolean; offset: string }> = [];
      (window as unknown as { __drawSamples: typeof samples }).__drawSamples = samples;
      const tick = () => {
        const el = document.getElementById("probe-draw");
        if (el) {
          samples.push({
            ready: document.documentElement.classList.contains("motion-ready"),
            offset: getComputedStyle(el).strokeDashoffset,
          });
        }
        if (samples.length < 240) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    const samples = await page.evaluate(() => (window as unknown as { __drawSamples: Array<{ ready: boolean; offset: string }> }).__drawSamples);
    const before = samples.filter((s) => !s.ready);
    expect(samples.length).toBeGreaterThan(0);
    for (const s of before) expect(s.offset, "painted drawn before motion-ready").not.toBe("0px");
    // And it does draw once motion arrives.
    expect(samples.at(-1)?.offset).toBe("0px");
  });
});

// ---------------------------------------------------------------------------------------------
// Count-up durations
// ---------------------------------------------------------------------------------------------

test.describe("count-up: bad durations can't stall or spin", () => {
  test("data-count-duration of -5 and 1e9 still end on the final value within 3.1s and stop requesting frames", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await appendContainer(page, "count-probe");
    await page.evaluate(async () => {
      const mod = await import("../motion.js");
      const box = document.getElementById("count-probe")!;
      box.scrollIntoView({ block: "center", behavior: "instant" });
      for (const d of ["-5", "1e9"]) {
        const el = document.createElement("span");
        el.dataset["countTo"] = "34.0";
        el.dataset["countDuration"] = d;
        el.dataset["probe"] = d;
        el.textContent = "34.0";
        box.append(el);
      }
      (window as unknown as { __t0: number }).__t0 = performance.now();
      mod.initCountUps(box);
    });
    await page.waitForFunction(
      () => Array.from(document.querySelectorAll("#count-probe [aria-hidden='true']")).every((el) => el.textContent === "34.0"),
      null,
      { timeout: 4000 },
    );
    const elapsed = await page.evaluate(() => performance.now() - (window as unknown as { __t0: number }).__t0);
    expect(elapsed, `took ${elapsed}ms`).toBeLessThan(3100 + 100);
    // The rounded digits can read final a few frames before progress reaches 1 (33.97 shows as
    // "34.0"), so count frames only once the 3000ms cap plus a frame has certainly passed.
    await page.waitForTimeout(Math.max(0, 3150 - elapsed));

    const frames = await page.evaluate(async () => {
      let n = 0;
      const original = window.requestAnimationFrame.bind(window);
      window.requestAnimationFrame = (cb) => {
        n++;
        return original(cb);
      };
      await new Promise((r) => setTimeout(r, 300));
      window.requestAnimationFrame = original;
      return n;
    });
    expect(frames).toBe(0);
  });
});

// ---------------------------------------------------------------------------------------------
// Motion token demo
// ---------------------------------------------------------------------------------------------

async function chipX(page: Page): Promise<number> {
  return page.locator(".motion-demo__chip").evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).m41);
}

test.describe("motion token demo: Play actually plays", () => {
  test("with the mouse over Play, clicking it moves the chip over time and brings it back", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const play = page.locator("[data-motion-play]");
    await play.scrollIntoViewIfNeeded();
    await play.hover();
    await page.waitForTimeout(400);
    expect(await chipX(page), "hovering Play must not pre-move the chip").toBe(0);
    expect(await play.getAttribute("aria-pressed")).toBeNull();

    await play.click();
    await page.waitForTimeout(120);
    const a = await chipX(page);
    await page.waitForTimeout(120);
    const b = await chipX(page);
    expect(a).toBeGreaterThan(0);
    expect(b).not.toBe(a);
    await page.waitForTimeout(900); // 320ms legs: the whole run is ~711ms
    expect(await chipX(page)).toBe(0);
  });
  // Reduced motion: tests/consistency.spec.ts "motion demo under reduced motion" (Play jumps to the
  // end state and a role="status" hint says why, instead of a silent dead button).
});

// ---------------------------------------------------------------------------------------------
// Comparison bars without typed attr()
// ---------------------------------------------------------------------------------------------

test.describe("bars: honest without typed attr() and without JS", () => {
  test("with JS off and the attr() rule removed, the default bar is still much narrower than the tuned bar", async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    try {
      const page = await context.newPage();
      let stripped = false;
      await page.route("**/motion.css", async (route) => {
        const res = await route.fetch();
        const css = await res.text();
        const next = css.replace(/@supports \(scale: attr\([\s\S]*?\r?\n\}\r?\n/, "");
        stripped = next !== css && !next.includes("attr(data-bar-to");
        await route.fulfill({ response: res, body: next });
      });
      await page.goto("/preview.html");
      expect(stripped, "the typed attr() rule was not found to strip").toBe(true);
      const [def, tuned] = await page.$$eval("#comparison .comparison .bar", (els) => els.map((el) => ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(el)));
      expect(def!).toBeGreaterThan(0);
      expect(tuned! / def!).toBeGreaterThan(4);
    } finally {
      await context.close();
    }
  });
});

// ---------------------------------------------------------------------------------------------
// light-dark() fallback
// ---------------------------------------------------------------------------------------------

/** Returns the body of the first `{ ... }` block following `start` (brace-matched). */
function blockAfter(css: string, start: number): string {
  const open = css.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < css.length; i++) {
    if (css[i] === "{") depth++;
    else if (css[i] === "}" && --depth === 0) return css.slice(open + 1, i);
  }
  throw new Error("unbalanced braces");
}

function declarations(body: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of body.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/(--[\w-]+)\s*:\s*([^;{}]+);/g)) {
    out.set(m[1]!, m[2]!.replace(/\s+/g, " ").trim());
  }
  return out;
}

/** Independent re-implementation of light-dark() resolution (not the generator's code). */
function pick(value: string, theme: "light" | "dark"): string {
  let s = value;
  for (;;) {
    const at = s.lastIndexOf("light-dark(");
    if (at === -1) return s;
    let depth = 0;
    let comma = -1;
    let end = -1;
    for (let i = at + "light-dark(".length - 1; i < s.length; i++) {
      if (s[i] === "(") depth++;
      else if (s[i] === ")" && --depth === 0) {
        end = i;
        break;
      } else if (s[i] === "," && depth === 1 && comma === -1) comma = i;
    }
    const a = s.slice(at + "light-dark(".length, comma).trim();
    const b = s.slice(comma + 1, end).trim();
    s = s.slice(0, at) + (theme === "light" ? a : b) + s.slice(end + 1);
  }
}

test.describe("tokens: light-dark() fallback can't drift", () => {
  test("the @supports-not block defines every light-dark() token with identical values for each theme", async () => {
    const css = (await readFile(path.join(DESIGN_ROOT, "tokens.css"), "utf8")).replace(/\r\n/g, "\n");
    const sourceStart = css.indexOf(':root,\n[data-theme="light"],\n[data-theme="dark"] {');
    expect(sourceStart).toBeGreaterThan(-1);
    const source = [...declarations(blockAfter(css, sourceStart))].filter(([, v]) => v.includes("light-dark("));
    expect(source.length).toBeGreaterThan(20);

    const fbStart = css.indexOf("@supports not (color: light-dark(#000, #fff))");
    expect(fbStart).toBeGreaterThan(-1);
    const fallback = blockAfter(css, fbStart);
    const dark = declarations(blockAfter(fallback, fallback.indexOf('[data-theme="dark"]')));
    const light = declarations(blockAfter(fallback, fallback.indexOf('[data-theme="light"]')));
    const osLight = declarations(blockAfter(fallback, fallback.indexOf(":root:not([data-theme])")));

    expect([...dark.keys()]).toEqual(source.map(([k]) => k));
    expect([...light.keys()]).toEqual(source.map(([k]) => k));
    expect([...osLight.keys()]).toEqual(source.map(([k]) => k));
    for (const [name, value] of source) {
      expect(dark.get(name), name).toBe(pick(value, "dark"));
      expect(light.get(name), name).toBe(pick(value, "light"));
      expect(osLight.get(name), name).toBe(pick(value, "light"));
      expect(dark.get(name)).not.toContain("light-dark(");
    }
  });

  for (const theme of ["dark", "light"] as const) {
    test(`forcing the fallback block renders the same colours (${theme})`, async ({ page }) => {
      const sample = () =>
        page.evaluate(() => {
          const pick = (sel: string) => {
            const s = getComputedStyle(document.querySelector(sel)!);
            return [s.backgroundColor, s.color, s.borderTopColor];
          };
          return [pick("body"), pick(".btn--primary"), pick(".input"), pick(".result-card[data-theme='light']"), pick(".result-card[data-theme='dark']")];
        });
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
      const native = await sample();

      await page.route("**/tokens.css", async (route) => {
        const res = await route.fetch();
        const css = (await res.text()).replace("@supports not (color: light-dark(#000, #fff))", "@supports (color: red)");
        await route.fulfill({ response: res, body: css });
      });
      await page.reload({ waitUntil: "networkidle" });
      await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
      const forced = await sample();
      expect(forced).toEqual(native);
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Mobile polish
// ---------------------------------------------------------------------------------------------

test.describe("mobile polish at 390px", () => {
  test("the focus-demo ring sits fully inside its swatch and no <select> truncates its text", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const ring = await page.evaluate(() => {
        const btn = document.querySelector(".is-focus-demo")!.getBoundingClientRect();
        const box = document.querySelector(".is-focus-demo")!.closest(".swatch")!.getBoundingClientRect();
        const R = 4; // 2px gap + 2px ring (--focus-ring)
        return { left: btn.left - R - box.left, right: box.right - (btn.right + R), top: btn.top - R - box.top };
      });
      expect(ring.left, "ring clipped on the left").toBeGreaterThanOrEqual(0);
      expect(ring.right, "ring clipped on the right").toBeGreaterThanOrEqual(0);
      expect(ring.top).toBeGreaterThanOrEqual(0);

      const selects = await page.$$eval("select", (els) =>
        els.map((el) => {
          const s = el as HTMLSelectElement;
          const probe = document.createElement("span");
          const cs = getComputedStyle(s);
          probe.style.setProperty("font", cs.font);
          probe.style.setProperty("position", "absolute");
          probe.style.setProperty("white-space", "nowrap");
          probe.textContent = s.selectedOptions[0]?.text ?? "";
          document.body.append(probe);
          const textW = probe.getBoundingClientRect().width;
          probe.remove();
          const inner = s.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) - 16; // native arrow
          return { text: probe.textContent, textW, inner, scroll: s.scrollWidth, client: s.clientWidth };
        }),
      );
      expect(selects.length).toBeGreaterThan(0);
      for (const s of selects) {
        expect(s.scroll, JSON.stringify(s)).toBeLessThanOrEqual(s.client);
        expect(s.textW, `"${s.text}" truncated: ${JSON.stringify(s)}`).toBeLessThanOrEqual(s.inner);
      }
    } finally {
      await context.close();
    }
  });

  test("no token name in the z-stack wraps across lines", async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const wrapped = await page.$$eval(".z-stack__card code, .z-stack__card:not(.z-stack__card--top) span", (els) =>
        els.filter((el) => el.getClientRects().length > 1).map((el) => el.textContent),
      );
      expect(wrapped).toEqual([]);
    } finally {
      await context.close();
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Content and accessibility
// ---------------------------------------------------------------------------------------------

test.describe("content: every term is defined where it first appears", () => {
  test("the first visible occurrence of each <dfn> term in reading order is that <dfn>", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const problems = await page.evaluate(() => {
      const texts: Array<{ node: Text; text: string }> = [];
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode: (n) =>
          (n.parentElement?.closest("[popover], script, style, .sr-only, [aria-hidden='true']") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
      });
      let n: Node | null;
      while ((n = walker.nextNode())) texts.push({ node: n as Text, text: n.textContent ?? "" });
      const bad: string[] = [];
      const dfns = Array.from(document.querySelectorAll("dfn"));
      for (const dfn of dfns) {
        const term = (dfn.textContent ?? "").trim();
        const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const re = new RegExp(`(^|[^\\w-])${escaped}(?![\\w-])`, "i");
        const first = texts.find((t) => re.test(t.text));
        if (!first || !dfn.contains(first.node)) bad.push(`${term}: first used in "${first?.text.trim().slice(0, 60)}"`);
      }
      return { bad, count: dfns.length };
    });
    expect(problems.count).toBeGreaterThanOrEqual(13);
    expect(problems.bad, problems.bad.join("\n")).toEqual([]);
    const ids = await page.$$eval("dfn", (els) => els.map((e) => e.textContent?.trim()));
    for (const required of ["tok/s", "decode", "context depth", "speculative strategies", "draft model", "quant", "prefill"]) {
      expect(ids).toContain(required);
    }
  });
});

test.describe("a11y: bench and CTA semantics", () => {
  test("nothing inside the state-matrix bench is in the tab order, and the hero CTA is a link", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    expect(await page.locator(".state-matrix").getAttribute("inert")).not.toBeNull();
    const reachable: string[] = [];
    for (let i = 0; i < 80; i++) {
      await page.keyboard.press("Tab");
      const inBench = await page.evaluate(() => !!document.activeElement?.closest(".state-matrix"));
      if (inBench) reachable.push(await page.evaluate(() => document.activeElement?.outerHTML.slice(0, 60) ?? ""));
    }
    expect(reachable).toEqual([]);
    const cta = page.getByRole("link", { name: "Get started" });
    await expect(cta).toHaveCount(1);
    expect(await cta.getAttribute("href")).toBe("#install");
    expect(await page.locator("#install").count()).toBe(1);
    expect(await page.locator(".hero button.btn--hero").count()).toBe(0);
  });
});

// ---------------------------------------------------------------------------------------------
// Hygiene
// ---------------------------------------------------------------------------------------------

test.describe("hygiene: committed build output and generated CSS can't drift", () => {
  test("check-build fails on a theme-init.js that differs from the compiled theme-init.ts, and passes on the real one", async () => {
    test.setTimeout(90_000);
    const script = path.join(DESIGN_ROOT, "scripts", "check-build.mjs");
    await run(process.execPath, [script], { cwd: DESIGN_ROOT });
    const dir = await mkdtemp(path.join(os.tmpdir(), "cerebro-drift-"));
    try {
      const stale = path.join(dir, "theme-init.js");
      await writeFile(stale, (await readFile(path.join(DESIGN_ROOT, "theme-init.js"), "utf8")) + "\n// stale edit\n");
      let failed = false;
      try {
        await run(process.execPath, [script, "--against", stale], { cwd: DESIGN_ROOT });
      } catch {
        failed = true;
      }
      expect(failed).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("the generated CSS blocks are current (gen-css --check)", async () => {
    await run(process.execPath, [path.join(DESIGN_ROOT, "scripts", "gen-css.mjs"), "--check"], { cwd: DESIGN_ROOT });
  });
});
