import { test, expect, type Browser, type Page } from "@playwright/test";
import { settleFullMotion } from "../scripts/shoot-settle.mjs";

/**
 * Design round 5: a primitive's end state must equal its final state for every element on the page
 * (the non-scaling-stroke sparkline once stopped at 57% while every "is-drawn" assertion passed),
 * and only elements a primitive has armed may ever be hidden.
 */

type Shot = { name: string; png: string; stroke: string };

/** Screenshots every visible [data-draw]'s <svg> once the page has settled in the given mode. */
async function drawShots(browser: Browser, viewport: { width: number; height: number }, theme: "dark" | "light", reduced: boolean): Promise<Shot[]> {
  const context = await browser.newContext({ viewport, colorScheme: theme, reducedMotion: reduced ? "reduce" : "no-preference" });
  try {
    const page = await context.newPage();
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    if (reduced) await page.evaluate(() => document.fonts.ready);
    else await settleFullMotion(page);
    await page.mouse.move(0, 0);
    const count = await page.locator("[data-draw]").count();
    const shots: Shot[] = [];
    for (let i = 0; i < count; i++) {
      const path = page.locator("[data-draw]").nth(i);
      if (!(await path.evaluate((el) => el.getClientRects().length > 0))) continue; // the other breakpoint's connector
      const svg = path.locator("xpath=ancestor::*[local-name()='svg'][1]");
      await svg.scrollIntoViewIfNeeded();
      await page.waitForTimeout(50);
      // Index and authored class only: state classes (is-draw-done, draw--scaled) differ by mode.
      const { name, stroke } = await path.evaluate((el, n) => ({ name: `${n}:${el.tagName}.${(el.getAttribute("class") ?? "").split(" ")[0]}`, stroke: getComputedStyle(el).stroke }), i);
      shots.push({ name, stroke, png: (await svg.screenshot({ animations: "disabled" })).toString("base64") });
    }
    return shots;
  } finally {
    await context.close();
  }
}

/** Decodes a PNG on a blank page (the specimen's CSP rightly forbids data: images) and returns the
 *  extent of the pixels painted in the stroke's own colour: where the drawn line actually reaches. */
async function strokeExtent(page: Page, png: string, stroke: string) {
  return page.evaluate(
    async ({ png, stroke }) => {
      const probe = new OffscreenCanvas(1, 1).getContext("2d")!;
      probe.fillStyle = stroke;
      probe.fillRect(0, 0, 1, 1);
      const [sr, sg, sb] = probe.getImageData(0, 0, 1, 1).data;
      const bytes = Uint8Array.from(atob(png), (c) => c.charCodeAt(0));
      const bmp = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
      const ctx = new OffscreenCanvas(bmp.width, bmp.height).getContext("2d")!;
      ctx.drawImage(bmp, 0, 0);
      const { data } = ctx.getImageData(0, 0, bmp.width, bmp.height);
      let count = 0;
      let minX = Infinity;
      let maxX = -1;
      let minY = Infinity;
      let maxY = -1;
      for (let i = 0; i < data.length; i += 4) {
        const d = Math.max(Math.abs(data[i]! - sr!), Math.abs(data[i + 1]! - sg!), Math.abs(data[i + 2]! - sb!));
        if (d > 40) continue;
        const x = (i / 4) % bmp.width;
        const y = Math.floor(i / 4 / bmp.width);
        count++;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
      return { count, minX, maxX, minY, maxY, width: bmp.width, height: bmp.height };
    },
    { png, stroke },
  );
}

test.describe("draw-on: the settled full-motion stroke equals the reduced-motion stroke", () => {
  for (const [viewport, theme] of [
    [{ width: 1440, height: 900 }, "dark"],
    [{ width: 390, height: 844 }, "light"],
  ] as const) {
    test(`every visible [data-draw] reaches the same pixels in both modes at ${viewport.width}px (${theme})`, async ({ browser }) => {
      test.setTimeout(90_000);
      const full = await drawShots(browser, viewport, theme, false);
      const reduced = await drawShots(browser, viewport, theme, true);
      expect(full.map((s) => s.name)).toEqual(reduced.map((s) => s.name));
      // Hero trace, both sparklines, one pipeline connector.
      expect(full.length).toBeGreaterThanOrEqual(4);
      const blank = await browser.newPage();
      try {
        for (let i = 0; i < full.length; i++) {
          const a = await strokeExtent(blank, full[i]!.png, full[i]!.stroke);
          const b = await strokeExtent(blank, reduced[i]!.png, reduced[i]!.stroke);
          const msg = `${full[i]!.name}: full ${JSON.stringify(a)} vs reduced ${JSON.stringify(b)}`;
          expect(b.count, msg).toBeGreaterThan(10);
          // A truncated stroke stops short: its far edge (rightmost / lowest painted pixel) moves.
          for (const edge of ["minX", "maxX", "minY", "maxY"] as const) expect(Math.abs(a[edge] - b[edge]), `${edge} ${msg}`).toBeLessThanOrEqual(2);
          expect(a.count / b.count, msg).toBeGreaterThan(0.9);
          expect(a.count / b.count, msg).toBeLessThan(1.1);
        }
      } finally {
        await blank.close();
      }
    });
  }
});

test.describe("draw-on: non-scaling strokes", () => {
  test("a stretched non-scaling path gets a measured dash, draws fully, then drops its dash", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const result = await page.evaluate(async () => {
      const mod = await import("../motion.js");
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", "0 0 100 10");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("width", "600");
      svg.setAttribute("height", "10");
      svg.style.setProperty("position", "fixed");
      svg.style.setProperty("top", "40%");
      svg.style.setProperty("left", "0");
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", "M0 5 H100");
      path.setAttribute("pathLength", "1");
      path.setAttribute("data-draw", "");
      path.style.setProperty("vector-effect", "non-scaling-stroke");
      path.style.setProperty("stroke", "currentColor");
      svg.append(path);
      document.body.append(svg);
      mod.initDrawOnReveal(svg);
      const armed = { scaled: path.classList.contains("draw--scaled"), len: Number(path.style.getPropertyValue("--draw-len")), armed: path.hasAttribute("data-motion-armed") };
      await new Promise((r) => setTimeout(r, 1600));
      return { ...armed, done: path.classList.contains("is-draw-done"), dash: getComputedStyle(path).strokeDasharray };
    });
    expect(result.armed).toBe(true);
    expect(result.scaled).toBe(true);
    // 600px of screen over 100 user units: the dash must cover at least 6 path lengths.
    expect(result.len).toBeGreaterThanOrEqual(6);
    expect(result.done).toBe(true);
    expect(result.dash).toBe("none");
  });

  test("an unmeasurable non-scaling path warns once and is shown drawn, never armed; the specimen warns about nothing", async ({ page }) => {
    const warnings: string[] = [];
    page.on("console", (m) => {
      if (m.type() === "warning" && m.text().includes("[cerebro/motion]")) warnings.push(m.text());
    });
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await settleFullMotion(page);
    expect(warnings, "the specimen itself must trigger no motion warning").toEqual([]);

    const result = await page.evaluate(async () => {
      const mod = await import("../motion.js");
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", "0 0 100 10");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("width", "600");
      const path = document.createElementNS(NS, "path") as SVGPathElement;
      path.setAttribute("d", "M0 5 H100");
      path.setAttribute("pathLength", "1");
      path.setAttribute("data-draw", "");
      path.style.setProperty("vector-effect", "non-scaling-stroke");
      svg.append(path);
      document.querySelector("main")!.append(svg);
      // Simulates an element the engine can't place on screen yet.
      Object.defineProperty(path, "getScreenCTM", { value: () => null });
      mod.initDrawOnReveal(svg);
      mod.initDrawOnReveal(svg); // a second init must not warn again
      return { armed: path.hasAttribute("data-motion-armed"), dash: getComputedStyle(path).strokeDasharray };
    });
    await page.waitForTimeout(50);
    expect(result.armed).toBe(false);
    expect(result.dash).toBe("none");
    expect(warnings).toHaveLength(1);
  });
});

test.describe("arming: only elements a primitive owns are ever hidden", () => {
  test("markup added after initMotion, with no re-init, renders final within 100ms", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.waitForTimeout(500);
    const state = await page.evaluate(async () => {
      const box = document.createElement("div");
      box.id = "late";
      const reveal = document.createElement("div");
      reveal.className = "reveal";
      reveal.dataset["reveal"] = "";
      reveal.textContent = "late reveal";
      const pop = document.createElement("div");
      pop.className = "pop-in";
      pop.dataset["reveal"] = "";
      pop.textContent = "late pop";
      const track = document.createElement("span");
      track.className = "comparison__track";
      track.style.setProperty("width", "400px");
      const bar = document.createElement("span");
      bar.className = "bar bar--p50";
      bar.dataset["bar"] = "";
      track.append(bar);
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", "0 0 100 10");
      svg.setAttribute("width", "300");
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", "M0 5 H100");
      path.setAttribute("pathLength", "1");
      path.setAttribute("data-draw", "");
      svg.append(path);
      box.append(reveal, pop, track, svg);
      document.querySelector(".hero")!.after(box);
      await new Promise((r) => setTimeout(r, 100));
      const dash = getComputedStyle(path).strokeDasharray;
      return {
        ready: document.documentElement.classList.contains("motion-ready"),
        reveal: getComputedStyle(reveal).opacity,
        pop: getComputedStyle(pop).opacity,
        barShare: ((b: Element) => { const r = b.getBoundingClientRect(); const t = b.parentElement!.getBoundingClientRect(); return Math.max(0, Math.min(r.right, t.right) - Math.max(r.left, t.left)); })(bar) / track.getBoundingClientRect().width,
        drawn: dash === "none" || getComputedStyle(path).strokeDashoffset === "0px",
      };
    });
    expect(state.ready).toBe(true);
    expect(state.reveal).toBe("1");
    expect(state.pop).toBe("1");
    expect(state.barShare).toBeGreaterThan(0.49);
    expect(state.barShare).toBeLessThan(0.51);
    expect(state.drawn).toBe(true);
  });

  test("cleanup disarms: an element a primitive stops watching shows its final state", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    const state = await page.evaluate(async () => {
      const mod = await import("../motion.js");
      const box = document.createElement("div");
      box.style.setProperty("margin-top", "4000px");
      const reveal = document.createElement("div");
      reveal.className = "reveal";
      reveal.dataset["reveal"] = "";
      reveal.textContent = "below the fold";
      box.append(reveal);
      document.querySelector("main")!.append(box);
      const off = mod.initScrollReveal(box);
      await new Promise((r) => requestAnimationFrame(() => r(null)));
      const before = { armed: reveal.hasAttribute("data-motion-armed"), opacity: getComputedStyle(reveal).opacity };
      off();
      const armedAfter = reveal.hasAttribute("data-motion-armed");
      await new Promise((r) => setTimeout(r, 800)); // it fades to final over --duration-slow
      return { before, after: { armed: armedAfter, opacity: getComputedStyle(reveal).opacity } };
    });
    expect(state.before).toEqual({ armed: true, opacity: "0" });
    expect(state.after).toEqual({ armed: false, opacity: "1" });
  });

  test("watch mode arms nodes added later, reveals them on scroll, and cleanup disconnects the observer", async ({ page }) => {
    await page.goto("/preview.html", { waitUntil: "networkidle" });
    await page.evaluate(async () => {
      const mod = await import("../motion.js");
      const list = document.createElement("div");
      list.id = "watched";
      document.querySelector("main")!.append(list);
      (window as unknown as { __offWatch: () => void }).__offWatch = mod.initMotion(list, { watch: true });
      const item = document.createElement("div");
      item.className = "reveal";
      item.dataset["reveal"] = "";
      item.id = "watched-1";
      item.textContent = "added after init";
      list.append(item);
    });
    await page.waitForTimeout(100);
    const armed = await page.locator("#watched-1").evaluate((el) => ({ armed: el.hasAttribute("data-motion-armed"), opacity: getComputedStyle(el).opacity }));
    expect(armed).toEqual({ armed: true, opacity: "0" });

    await page.locator("#watched-1").scrollIntoViewIfNeeded();
    await page.waitForTimeout(900);
    expect(await page.locator("#watched-1").evaluate((el) => getComputedStyle(el).opacity)).toBe("1");

    const after = await page.evaluate(async () => {
      (window as unknown as { __offWatch: () => void }).__offWatch();
      const item = document.createElement("div");
      item.className = "reveal";
      item.dataset["reveal"] = "";
      item.style.setProperty("margin-top", "3000px");
      item.textContent = "added after cleanup";
      document.getElementById("watched")!.append(item);
      await new Promise((r) => setTimeout(r, 50));
      return { armed: item.hasAttribute("data-motion-armed"), opacity: getComputedStyle(item).opacity, ready: document.documentElement.classList.contains("motion-ready") };
    });
    // The page's own initMotion(document) still holds motion-ready; the watcher is gone.
    expect(after).toEqual({ armed: false, opacity: "1", ready: true });
  });
});
