import { test, expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Design round 4: the small seams that decide whether the system reads as one designed thing
 * (controls sized to their content at every width, one vertical rhythm, distinguishable text roles,
 * one number-to-unit rule, press that reads pushed in) and content that can't contradict the real
 * product (fictional names beside illustrative numbers, Cerebro's real default port).
 */

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const REPO_ROOT = path.resolve(DESIGN_ROOT, "..");
const THEMES = ["dark", "light"] as const;

async function setTheme(page: Page, theme: (typeof THEMES)[number]) {
  await page.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
  await page.waitForTimeout(150);
}

async function openAt(page: Page, width: number, height = 900) {
  await page.setViewportSize({ width, height });
  await page.goto("/preview.html", { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
}

// ---------------------------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------------------------

test.describe("selects fit their text at every width", () => {
  for (const width of [1440, 1024, 768, 390]) {
    test(`no <select> clips any of its options at ${width}px`, async ({ page }) => {
      await openAt(page, width);
      const clipped = await page.evaluate(() => {
        const out: string[] = [];
        for (const s of Array.from(document.querySelectorAll("select"))) {
          const cs = getComputedStyle(s);
          const ctx = document.createElement("canvas").getContext("2d");
          if (!ctx) throw new Error("no 2d context");
          ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
          const room = s.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
          // Every option, not only the selected one: the width must not depend on the selection.
          for (const o of Array.from(s.options)) {
            const need = ctx.measureText(o.text).width;
            if (need > room + 0.5) out.push(`"${o.text}" needs ${need.toFixed(1)}px, has ${room.toFixed(1)}px`);
          }
        }
        return out;
      });
      expect(clipped).toEqual([]);
    });
  }

  test("the chevron sits inside the end padding, clear of the widest option's text", async ({ page }) => {
    await openAt(page, 1440);
    const overlaps = await page.evaluate(() => {
      const out: string[] = [];
      for (const wrap of Array.from(document.querySelectorAll<HTMLElement>(".select"))) {
        const s = wrap.querySelector("select");
        if (!s) continue;
        const cs = getComputedStyle(s);
        const after = getComputedStyle(wrap, "::after");
        const box = s.getBoundingClientRect();
        const textEnd = box.right - parseFloat(cs.paddingRight) - parseFloat(cs.borderRightWidth);
        const chevronStart = box.right - parseFloat(after.insetInlineEnd) - parseFloat(after.width) * Math.SQRT2;
        if (chevronStart < textEnd) out.push(`${s.className}: chevron starts ${chevronStart} before text end ${textEnd}`);
        if (after.content === "none" || after.content === "normal") out.push("no chevron rendered");
      }
      return out;
    });
    expect(overlaps).toEqual([]);
  });
});

test.describe("controls in one row share one height", () => {
  for (const width of [1440, 1024, 390]) {
    test(`the motion demo's Play button and selects are the same height (±1px) at ${width}px`, async ({ page }) => {
      await openAt(page, width);
      const heights = await page.$$eval(".motion-demo :is(.btn, select)", (els) => els.map((el) => el.getBoundingClientRect().height));
      expect(heights.length).toBe(3);
      expect(Math.max(...heights) - Math.min(...heights), JSON.stringify(heights)).toBeLessThanOrEqual(1);
    });
  }
});

test.describe("result-card readouts stay in their own channel", () => {
  for (const width of [1440, 1200, 1030, 1024, 900, 600, 390]) {
    test(`no metric label or value spills out of its metric at ${width}px`, async ({ page }) => {
      await openAt(page, width);
      const spills = await page.$$eval(".result-card .metric", (metrics) =>
        metrics.flatMap((m) => {
          const box = m.getBoundingClientRect();
          return Array.from(m.querySelectorAll(".metric__label, .metric__value, .metric__qual"))
            .filter((el) => {
              const r = el.getBoundingClientRect();
              return r.right > box.right + 1 || r.left < box.left - 1 || (el as HTMLElement).scrollWidth > (el as HTMLElement).clientWidth + 1;
            })
            .map((el) => `${el.className} "${el.textContent}"`);
        }),
      );
      expect(spills).toEqual([]);
    });
  }
});

test.describe("one vertical rhythm", () => {
  for (const width of [1440, 1024, 768, 390]) {
    test(`stacked blocks in a section never sit closer than 16px at ${width}px`, async ({ page }) => {
      await openAt(page, width);
      const touching = await page.evaluate(() => {
        const isText = (el: Element) => /^(P|H[1-6])$/.test(el.tagName);
        const out: string[] = [];
        for (const s of Array.from(document.querySelectorAll("main section"))) {
          const kids = Array.from(s.children).filter((k) => k.getBoundingClientRect().height > 0);
          for (let i = 1; i < kids.length; i++) {
            const a = kids[i - 1] as Element;
            const b = kids[i] as Element;
            // Layout boxes, not transformed ones: a reveal that hasn't fired yet is still 16px low.
            const ha = a as HTMLElement;
            const hb = b as HTMLElement;
            const gap = hb.offsetTop - (ha.offsetTop + ha.offsetHeight);
            // Only stacked pairs: the two columns of a plate sit side by side, not one above the other.
            const ra = a.getBoundingClientRect();
            const rb = b.getBoundingClientRect();
            if (rb.left >= ra.right || rb.right <= ra.left) continue;
            // Text-to-text pairs (eyebrow, title, desc) keep their own tighter heading rhythm.
            if (isText(a) && isText(b)) {
              if (gap < 0) out.push(`#${s.id}: ${a.className} overlaps ${b.className}`);
              continue;
            }
            // offsetTop/offsetHeight are integers, so allow 1px of rounding.
            if (gap < 16 - 1.5) out.push(`#${s.id}: ${a.className} -> ${b.className} gap ${gap.toFixed(1)}px`);
          }
        }
        return out;
      });
      expect(touching).toEqual([]);
    });
  }
});

test.describe("icon-button geometry", () => {
  test("the theme toggle is the catalog's 40x40 standalone icon button", async ({ page }) => {
    await openAt(page, 1440);
    // Standalone icon buttons only: the code block's copy button is an inline control (DESIGN.md:
    // 24px minimum inline, 40px standalone).
    const size = (sel: string) =>
      page.locator(sel).first().evaluate((el) => {
        const r = el.getBoundingClientRect();
        return { w: Math.round(r.width), h: Math.round(r.height) };
      });
    expect(await size("#buttons .btn--icon")).toEqual({ w: 40, h: 40 });
    expect(await size("[data-theme-toggle]")).toEqual({ w: 40, h: 40 });
  });

  for (const os of ["dark", "light"] as const) {
    test(`toggling back to the OS theme (${os}) returns the page to following the system`, async ({ browser }) => {
      const context = await browser.newContext({ colorScheme: os });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        const toggle = page.locator("[data-theme-toggle]");
        const stored = () => page.evaluate(() => localStorage.getItem("cerebro-theme"));
        expect(await stored()).toBeNull();
        await toggle.click();
        expect(await stored()).toBe(os === "dark" ? "light" : "dark");
        await toggle.click();
        expect(await stored(), "the second click should clear the explicit choice").toBeNull();
        expect(await page.evaluate(() => document.documentElement.getAttribute("data-theme"))).toBe(os);
        // And it follows the OS again: flip the emulated preference.
        await page.emulateMedia({ colorScheme: os === "dark" ? "light" : "dark" });
        await expect.poll(() => page.evaluate(() => document.documentElement.getAttribute("data-theme"))).toBe(os === "dark" ? "light" : "dark");
      } finally {
        await context.close();
      }
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------------------------

const TEXT_ROLES = ["--color-accent-text", "--color-warning", "--color-danger", "--color-success"];

test.describe("text roles stay distinct", () => {
  for (const theme of THEMES) {
    test(`every pair of accent-text/warning/danger/success is 30deg of hue or 0.08 lightness apart (${theme})`, async ({ page }) => {
      await page.goto("/preview.html");
      await setTheme(page, theme);
      const close = await page.evaluate(async (roles) => {
        const m = await import("../contrast.js");
        const st = getComputedStyle(document.documentElement);
        const o = roles.map((r) => m.oklchOf(st.getPropertyValue(r).trim()));
        const bad: string[] = [];
        for (let i = 0; i < o.length; i++) {
          for (let j = i + 1; j < o.length; j++) {
            const a = o[i] as { l: number; h: number };
            const b = o[j] as { l: number; h: number };
            const dh = m.hueDelta(a.h, b.h);
            const dl = Math.abs(a.l - b.l);
            if (dh < 30 && dl < 0.08) bad.push(`${roles[i]} ~ ${roles[j]} (dh ${dh.toFixed(1)}, dl ${dl.toFixed(3)})`);
          }
        }
        return bad;
      }, TEXT_ROLES);
      expect(close).toEqual([]);
    });
  }
});

test.describe("press reads pushed in", () => {
  for (const theme of THEMES) {
    test(`pressed buttons keep AA text contrast, and a pressed fill is darker than its rest (${theme})`, async ({ page }) => {
      await page.goto("/preview.html");
      await setTheme(page, theme);
      const rows = await page.evaluate(async () => {
        const m = await import("../contrast.js");
        type C = { r: number; g: number; b: number; a: number };
        const over = (top: C, under: C): C => ({
          r: top.r * top.a + under.r * (1 - top.a),
          g: top.g * top.a + under.g * (1 - top.a),
          b: top.b * top.a + under.b * (1 - top.a),
          a: 1,
        });
        /** The opaque colour painted behind `el`: its own background over its ancestors'. */
        const backdrop = (el: Element | null): C => {
          const layers: C[] = [];
          for (let n = el; n; n = n.parentElement) {
            const c = m.getRenderedColor(getComputedStyle(n).backgroundColor);
            if (c.a > 0) layers.push(c);
            if (c.a >= 1) break;
          }
          return layers.reverse().reduce<C>((acc, c) => over(c, acc), { r: 255, g: 255, b: 255, a: 1 });
        };
        /** The opaque colour of `el` with its ::before overlay composited on top. */
        const withOverlay = (el: Element): C => {
          const pre = getComputedStyle(el, "::before");
          const overlay = m.getRenderedColor(pre.backgroundColor);
          overlay.a *= Number(pre.opacity);
          return over(overlay, backdrop(el));
        };
        /** OKLab distance (Euclidean over L, a, b). */
        const oklabDist = (x: C, y: C) => {
          const ab = (o: { c: number; h: number }) =>
            Number.isNaN(o.h) ? [0, 0] : [o.c * Math.cos((o.h * Math.PI) / 180), o.c * Math.sin((o.h * Math.PI) / 180)];
          const p = m.toOklch(x);
          const q = m.toOklch(y);
          const [pa = 0, pb = 0] = ab(p);
          const [qa = 0, qb = 0] = ab(q);
          return Math.hypot(p.l - q.l, pa - qa, pb - qb);
        };
        const out: Array<{ variant: string; ratio: number; restL: number; hoverL: number; pressedL: number; restToPressed: number; filled: boolean }> = [];
        for (const variant of ["primary", "secondary", "destructive"]) {
          const rest = document.querySelector(`.state-matrix .btn--${variant}:not([data-force]):not([disabled]):not([aria-busy])`);
          const hover = document.querySelector(`.state-matrix .btn--${variant}[data-force="hover"]`);
          const pressed = document.querySelector(`.state-matrix .btn--${variant}[data-force="active"]`);
          if (!rest || !hover || !pressed) throw new Error(`missing ${variant} in the state matrix`);
          const restBg = backdrop(rest);
          const hoverBg = withOverlay(hover);
          const pressedBg = withOverlay(pressed);
          const fg = m.getRenderedColor(getComputedStyle(pressed).color);
          const lum = (c: C) => m.relativeLuminance(c);
          out.push({
            variant,
            ratio: m.contrastRatio(fg, pressedBg),
            restL: lum(restBg),
            hoverL: lum(hoverBg),
            pressedL: lum(pressedBg),
            restToPressed: oklabDist(restBg, pressedBg),
            filled: m.getRenderedColor(getComputedStyle(rest).backgroundColor).a > 0,
          });
        }
        return out;
      });
      for (const r of rows) {
        expect(r.ratio, `${r.variant} pressed text contrast`).toBeGreaterThanOrEqual(4.5);
        if (r.filled) expect(r.pressedL, `${r.variant}: pressed must be darker than rest`).toBeLessThan(r.restL);
        // Pushed in, never lit up: a press is never brighter than the hover it follows.
        expect(r.pressedL, `${r.variant}: pressed must not be lighter than hover`).toBeLessThanOrEqual(r.hoverL);
        // DESIGN.md: on touch the press is the only feedback, so it must be visible against rest.
        expect(r.restToPressed, `${r.variant}: rest to pressed OKLab distance`).toBeGreaterThanOrEqual(0.02);
      }
      expect(rows.find((r) => r.variant === "primary")?.filled).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Readouts
// ---------------------------------------------------------------------------------------------

test.describe("a number and its unit are one unit of type", () => {
  for (const width of [1440, 390]) {
    test(`the value-to-unit gap is the same share of the value size in every readout at ${width}px`, async ({ page }) => {
      await openAt(page, width);
      const gaps = await page.evaluate(() => {
        const out: Array<{ where: string; ratio: number; literalSpace: boolean }> = [];
        for (const count of Array.from(document.querySelectorAll<HTMLElement>(".count"))) {
          const value = count.parentElement;
          const unit = count.nextElementSibling;
          if (!value || !unit || !/(__unit|__of)\b/.test(unit.className)) continue;
          if (unit.className.includes("__of")) continue; // "5/10" is a fraction, not a unit
          const a = count.getBoundingClientRect();
          const b = unit.getBoundingClientRect();
          const size = parseFloat(getComputedStyle(value).fontSize);
          const between = count.nextSibling;
          out.push({
            where: value.className,
            ratio: (b.left - a.right) / size,
            literalSpace: between !== unit && /\s/.test(between?.textContent ?? ""),
          });
        }
        return out;
      });
      expect(gaps.length).toBeGreaterThanOrEqual(8);
      for (const g of gaps) {
        expect(g.literalSpace, `${g.where}: literal whitespace between value and unit`).toBe(false);
        expect(Math.abs(g.ratio - 0.3), `${g.where}: gap is ${g.ratio.toFixed(3)}em, want 0.3em`).toBeLessThanOrEqual(0.05);
      }
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Content that can't contradict the real product
// ---------------------------------------------------------------------------------------------

test.describe("illustrative content stays fictional", () => {
  test("no illustrative figure or result card names a real model", async ({ page }) => {
    await page.goto("/preview.html");
    const hits = await page.evaluate(() => {
      const flagged = new Set<Element>();
      for (const flag of Array.from(document.querySelectorAll(".stat-tile__flag"))) {
        const holder = flag.closest("figure, .stat-tile, .result-card");
        if (holder) flagged.add(holder);
      }
      for (const card of Array.from(document.querySelectorAll(".result-card"))) flagged.add(card);
      return Array.from(flagged)
        .filter((e) => /qwen|llama-?[23]|gemma|mistral|mixtral|deepseek|phi-?\d|gpt-oss/i.test(e.textContent ?? ""))
        .map((e) => e.className);
    });
    expect(hits).toEqual([]);
  });

  test("the status chip in each result card sits inside the flagged figure", async ({ page }) => {
    await page.goto("/preview.html");
    const outside = await page.$$eval(".result-card .result-card__status", (els) =>
      els.filter((el) => !el.closest("figure")?.querySelector(".stat-tile__flag")).length,
    );
    expect(outside).toBe(0);
  });

  test("the loopback endpoint shows Cerebro's own default API port (crates/cerebro-config)", async ({ page }) => {
    const src = await readFile(path.join(REPO_ROOT, "crates", "cerebro-config", "src", "schema.rs"), "utf8");
    const impl = src.slice(src.indexOf("impl Default for ApiSection"));
    const port = /port:\s*(\d{2,5})/.exec(impl)?.[1];
    expect(port, "ApiSection default port not found in schema.rs").toBeTruthy();
    await page.goto("/preview.html");
    const endpoints = await page.locator(".result-card__endpoint").allTextContents();
    expect(endpoints.length).toBeGreaterThan(0);
    for (const t of endpoints) expect(t.trim()).toBe(`127.0.0.1:${port}`);
  });

  test("one spelling system (US, like the token names)", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    // Open every <details> so collapsed text is checked too.
    await page.evaluate(() => document.querySelectorAll("details").forEach((d) => (d.open = true)));
    const text = await page.evaluate(() => document.body.innerText);
    const british = text.match(/\b(colour|characteris|quantis|catalogue|behaviour|optimis|labell|favour|centre|analys(e|ing)\b|licen[cs]e(d)? under)\w*/gi) ?? [];
    expect(british.filter((w) => !/^licen/i.test(w))).toEqual([]);
  });
});

// ---------------------------------------------------------------------------------------------
// Motion composability and the reduced-motion Play
// ---------------------------------------------------------------------------------------------

test.describe("reveal composes safely", () => {
  test("the specimen puts no reveal class on an interactive element, and init warns when a consumer does", async ({ page }) => {
    const warnings: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "warning") warnings.push(msg.text());
    });
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    expect(warnings.filter((w) => w.includes("[cerebro/motion]"))).toEqual([]);

    await page.evaluate(async () => {
      const box = document.createElement("div");
      box.id = "reveal-misuse";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn--secondary reveal";
      btn.setAttribute("data-reveal", "");
      btn.setAttribute("data-press", "");
      btn.textContent = "Misused";
      const wrap = document.createElement("div");
      wrap.className = "reveal";
      wrap.setAttribute("data-reveal", "");
      box.append(btn, wrap);
      document.querySelector("main")?.append(box);
      const m = await import("../motion.js");
      m.initScrollReveal(box)();
      m.initScrollReveal(box)(); // a second init must not warn again for the same element
    });
    const motionWarnings = warnings.filter((w) => w.includes("[cerebro/motion]"));
    expect(motionWarnings).toHaveLength(1);
    expect(motionWarnings[0]).toContain("wrapper");
  });
});

test.describe("motion demo under reduced motion", () => {
  test("Play stays useful: it jumps the chip to its end state and a status hint says why", async ({ browser }) => {
    const context = await browser.newContext({ reducedMotion: "reduce" });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const hint = page.locator("#motion-hint");
      await expect(hint).toHaveAttribute("role", "status");
      await expect(hint).not.toHaveText("");
      const play = page.locator("[data-motion-play]");
      await expect(play).toHaveAttribute("aria-describedby", "motion-hint");
      await play.scrollIntoViewIfNeeded();
      await play.click();
      const state = await page.evaluate(() => {
        const chip = document.querySelector<HTMLElement>(".motion-demo__chip");
        const track = document.querySelector<HTMLElement>("[data-motion-track]");
        if (!chip || !track) throw new Error("demo missing");
        return {
          x: new DOMMatrixReadOnly(getComputedStyle(chip).transform).m41,
          travel: track.clientWidth - 24,
          running: chip.getAnimations().length,
        };
      });
      expect(state.running, "no animation may run under reduced motion").toBe(0);
      expect(Math.abs(state.x - state.travel)).toBeLessThanOrEqual(1);
      await expect(hint).toContainText("end state");
      // A second press returns it to the start, still without motion.
      await play.click();
      expect(await page.locator(".motion-demo__chip").evaluate((el) => new DOMMatrixReadOnly(getComputedStyle(el).transform).m41)).toBe(0);
    } finally {
      await context.close();
    }
  });

  test("with full motion the hint is empty and hidden", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await expect(page.locator("#motion-hint")).toHaveText("");
    await expect(page.locator("#motion-hint")).toBeHidden();
  });
});

// ---------------------------------------------------------------------------------------------
// The screenshot pipeline records the default experience
// ---------------------------------------------------------------------------------------------

test.describe("shoot.mjs captures the default, full-motion state", () => {
  test("after the scroll pass every reveal is visible, every count-up is final and the badge says no-preference", async ({ page }) => {
    const { settleFullMotion, assertSettled } = await import("../scripts/shoot-settle.mjs");
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await settleFullMotion(page);
    const report = await assertSettled(page, false);
    expect(report.hidden).toEqual([]);
    expect(report.unfinishedCounts).toEqual([]);
    expect(report.undrawn).toEqual([]);
    expect(report.badge).toBe("prefers-reduced-motion: no-preference");
  });
});

// ---------------------------------------------------------------------------------------------
// The hero chart reads as data
// ---------------------------------------------------------------------------------------------

test.describe("hero trace is a readable chart", () => {
  for (const width of [1440, 390]) {
    test(`y ticks sit on their grid lines and the defaults line is labeled, inside the flagged figure, at ${width}px`, async ({ page }) => {
      await openAt(page, width);
      const r = await page.evaluate(() => {
        const fig = document.querySelector(".hero__trace");
        const svg = fig?.querySelector<SVGSVGElement>(".hero__trace-plot");
        if (!fig || !svg) throw new Error("hero trace missing");
        const box = svg.getBoundingClientRect();
        // viewBox is 0 0 320 140; with height:auto the scale is uniform.
        const yOf = (u: number) => box.top + (u / 140) * box.height;
        const mid = (el: Element | null) => {
          if (!el) return NaN;
          const b = el.getBoundingClientRect();
          return (b.top + b.bottom) / 2;
        };
        const ticks = [
          { label: "40", y: 10 },
          { label: "20", y: 70 },
          { label: "0", y: 130 },
        ].map((t) => {
          const el = fig.querySelector(`.hero__trace-tick--${t.label}`);
          return { label: t.label, text: el?.textContent?.trim(), off: Math.abs(mid(el) - yOf(t.y)) };
        });
        const base = fig.querySelector(".hero__trace-base-label");
        const baseBox = base?.getBoundingClientRect();
        return {
          flagged: !!fig.querySelector(".stat-tile__flag"),
          ticks,
          baseText: base?.textContent?.trim(),
          // The label sits just above the dashed line (y=112), never across it.
          baseAboveLine: baseBox ? baseBox.bottom <= yOf(112) + 0.5 && baseBox.bottom >= yOf(112) - 16 : false,
          baseInside: baseBox ? baseBox.right <= box.right + 0.5 && baseBox.left >= box.left - 0.5 : false,
        };
      });
      expect(r.flagged).toBe(true);
      for (const t of r.ticks) {
        expect(t.text).toBe(t.label);
        expect(t.off, `tick ${t.label} is ${t.off.toFixed(1)}px off its grid line`).toBeLessThanOrEqual(1.5);
      }
      expect(r.baseText).toBe("defaults 6.0");
      expect(r.baseAboveLine).toBe(true);
      expect(r.baseInside).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Elevation reads in both themes
// ---------------------------------------------------------------------------------------------

test.describe("elevation = surface + shadow", () => {
  for (const theme of THEMES) {
    test(`each elevation level sits one surface further from the page background (${theme})`, async ({ page }) => {
      await page.goto("/preview.html");
      await setTheme(page, theme);
      const rows = await page.evaluate(async () => {
        const m = await import("../contrast.js");
        const bg = m.getRenderedColor(getComputedStyle(document.body).backgroundColor);
        return [1, 2, 3].map((n) => {
          const el = document.querySelector(`#radii-elevation .elev-${n}`);
          if (!el) throw new Error(`.elev-${n} missing`);
          const cs = getComputedStyle(el);
          return {
            n,
            // Distance from the page background, whichever direction the theme steps in.
            distance: m.contrastRatio(m.getRenderedColor(cs.backgroundColor), bg),
            shadow: cs.boxShadow,
          };
        });
      });
      for (const r of rows) expect(r.shadow, `.elev-${r.n} has a shadow`).not.toBe("none");
      expect(rows[0]!.distance).toBeLessThan(rows[1]!.distance);
      expect(rows[1]!.distance).toBeLessThan(rows[2]!.distance);
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Nothing scrolls under the sticky topbar
// ---------------------------------------------------------------------------------------------

test.describe("scroll targets clear the sticky topbar", () => {
  for (const width of [1440, 390]) {
    test(`anchor jumps, scrollIntoView and keyboard focus never land a target behind the topbar at ${width}px`, async ({ browser }) => {
      // Reduced motion: the scroll is instant, so positions are final when read.
      const context = await browser.newContext({ viewport: { width, height: 800 }, reducedMotion: "reduce" });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        await page.evaluate(() => document.documentElement.style.setProperty("scroll-behavior", "auto"));
        const topbarBottom = await page.locator(".topbar").evaluate((el) => el.getBoundingClientRect().bottom);

        // 1. The hero CTA's same-page link.
        await page.locator('.hero a[href="#install"]').click();
        await expect.poll(() => page.evaluate(() => location.hash)).toBe("#install");
        const installTop = await page.locator("#install").evaluate((el) => el.getBoundingClientRect().top);
        expect(installTop, "#install lands behind the topbar").toBeGreaterThanOrEqual(topbarBottom);

        // 2. scrollIntoView (what audit tools and app code use) on text deep in the page.
        const tops = await page.evaluate(() =>
          ["#comparison .comparison__label", "#forms .field__label", "#draw-on .pipeline__label", "#count-up .stat-tile__value"].map((sel) => {
            const el = document.querySelector(sel);
            if (!el) throw new Error(`${sel} missing`);
            el.scrollIntoView();
            return { sel, top: el.getBoundingClientRect().top };
          }),
        );
        for (const t of tops) expect(t.top, `${t.sel} lands behind the topbar`).toBeGreaterThanOrEqual(topbarBottom);

        // 3. Keyboard focus moving to a control below the fold scrolls it into view clear of the bar.
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.locator("#forms .field__empty .btn").focus();
        const focusTop = await page.locator("#forms .field__empty .btn").evaluate((el) => el.getBoundingClientRect().top);
        expect(focusTop, "focused control is obscured by the topbar").toBeGreaterThanOrEqual(topbarBottom);
      } finally {
        await context.close();
      }
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Bars: the finished box is the visible fill, and live changes slide
// ---------------------------------------------------------------------------------------------

test.describe("bars never hide behind their track's clip", () => {
  for (const reducedMotion of ["no-preference", "reduce"] as const) {
    test(`every finished bar's box stays inside its track, so it can't overlap the label beside it (${reducedMotion})`, async ({ browser }) => {
      const context = await browser.newContext({ reducedMotion, viewport: { width: 1440, height: 900 } });
      try {
        const page = await context.newPage();
        await page.goto("/preview.html", { waitUntil: "networkidle" });
        if (reducedMotion === "no-preference") {
          const { settleFullMotion } = await import("../scripts/shoot-settle.mjs");
          await settleFullMotion(page);
        }
        const out = await page.$$eval(".bar", (bars) =>
          bars.map((b) => {
            const r = b.getBoundingClientRect();
            const t = b.parentElement!.getBoundingClientRect();
            return { cls: b.className, left: r.left - t.left, right: t.right - r.right };
          }),
        );
        expect(out.length).toBeGreaterThanOrEqual(5);
        for (const o of out) {
          expect(o.left, `${o.cls} starts before its track`).toBeGreaterThanOrEqual(-0.5);
          expect(o.right, `${o.cls} ends past its track`).toBeGreaterThanOrEqual(-0.5);
        }
      } finally {
        await context.close();
      }
    });
  }

  test("setBarValue slides the right end from the old value to the new one, left end anchored, both directions", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const r = await page.evaluate(async () => {
      const m = await import("../motion.js");
      const bar = document.querySelector<HTMLElement>(".tune-progress .bar");
      if (!bar) throw new Error("no progress bar");
      bar.scrollIntoView({ block: "center" });
      const track = bar.parentElement!;
      const visible = () => {
        const b = bar.getBoundingClientRect();
        const t = track.getBoundingClientRect();
        return { left: Math.max(b.left, t.left) - t.left, share: (Math.min(b.right, t.right) - Math.max(b.left, t.left)) / t.width, boxLeft: b.left - t.left };
      };
      const run = async (to: number) => {
        m.setBarValue(bar, to);
        const anims = bar.getAnimations().filter((a) => !(a instanceof CSSTransition));
        const a = anims[0];
        if (!a) return { animated: false, mid: visible(), end: visible() };
        a.pause();
        a.currentTime = Number(a.effect?.getComputedTiming().duration ?? 600) / 2;
        const mid = visible();
        a.finish();
        await new Promise((res) => requestAnimationFrame(() => res(null)));
        return { animated: true, mid, end: visible() };
      };
      const start = visible().share;
      const grow = await run(0.8);
      const shrink = await run(0.3);
      return { start, grow, shrink, left: bar.getAnimations().length };
    });
    expect(r.start).toBeCloseTo(0.55, 2);
    expect(r.grow.animated).toBe(true);
    expect(r.grow.mid.share).toBeGreaterThan(0.56);
    expect(r.grow.mid.share).toBeLessThan(0.8);
    expect(r.grow.mid.left).toBeLessThanOrEqual(0.5);
    expect(r.grow.end.share).toBeCloseTo(0.8, 2);
    expect(r.grow.end.boxLeft).toBeCloseTo(0, 0);
    expect(r.shrink.animated).toBe(true);
    expect(r.shrink.mid.share).toBeGreaterThan(0.3);
    expect(r.shrink.mid.share).toBeLessThan(0.79);
    expect(r.shrink.mid.left, "the fill's left end must stay on the track start").toBeLessThanOrEqual(0.5);
    expect(r.shrink.end.share).toBeCloseTo(0.3, 2);
    expect(r.shrink.end.boxLeft, "after shrinking the box is its new length, inside the track").toBeCloseTo(0, 0);
    expect(r.left).toBe(0);
  });

  test("under reduced motion setBarValue jumps without animating", async ({ browser }) => {
    const context = await browser.newContext({ reducedMotion: "reduce" });
    try {
      const page = await context.newPage();
      await page.goto("/preview.html", { waitUntil: "networkidle" });
      const r = await page.evaluate(async () => {
        const m = await import("../motion.js");
        const bar = document.querySelector<HTMLElement>(".tune-progress .bar")!;
        m.setBarValue(bar, 0.9);
        const b = bar.getBoundingClientRect();
        const t = bar.parentElement!.getBoundingClientRect();
        return { animations: bar.getAnimations().length, share: (Math.min(b.right, t.right) - Math.max(b.left, t.left)) / t.width };
      });
      expect(r.animations).toBe(0);
      expect(r.share).toBeCloseTo(0.9, 2);
    } finally {
      await context.close();
    }
  });
});
