#!/usr/bin/env node
// @ts-check
/**
 * `npm run fonts:metrics`: measures, in headless Chromium, each web font against the exact local
 * face (by full name) that fonts.css uses as its metric-matched fallback for that weight band.
 * Prints size-adjust (advance-width ratio, per type size) and the ascent/descent overrides
 * (web-font metric / size-adjust) to copy into fonts.css.
 */
import { chromium } from "@playwright/test";
import { startServer } from "./serve.mjs";

/** @type {Array<[string, number, number[], string]>} web family, weight, sizes (px), local face */
const CASES = [
  ["Inter Variable", 400, [16, 18], "Segoe UI"],
  ["Inter Variable", 500, [16], "Segoe UI Semibold"],
  ["Inter Variable", 600, [22, 30], "Segoe UI Semibold"],
  ["Inter Variable", 700, [16], "Segoe UI Bold"],
  ["Inter Variable", 400, [16, 18], "Arial"],
  ["Inter Variable", 600, [22], "Arial Bold"],
  ["Fraunces Variable", 400, [22, 30], "Georgia"],
  ["Fraunces Variable", 600, [30, 46, 52, 80], "Georgia Bold"],
  ["Fraunces Variable", 400, [22, 30], "Times New Roman"],
  ["Fraunces Variable", 600, [30, 46, 52, 80], "Times New Roman Bold"],
  ["JetBrains Mono Variable", 400, [14], "Consolas"],
];

const { url, close } = await startServer(0);
const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(`${url}/preview.html`, { waitUntil: "networkidle" });
  const rows = await page.evaluate(async (cases) => {
    await document.fonts.ready;
    const text = "A calm instrument for local inference. Every configuration, characterized. Measure first 0123456789";
    const span = document.createElement("span");
    span.textContent = text;
    span.style.setProperty("position", "absolute");
    span.style.setProperty("white-space", "nowrap");
    document.body.append(span);
    /** @param {string} family @param {number} weight @param {number} size */
    const width = (family, weight, size) => {
      span.style.setProperty("font", `${weight} ${size}px "${family}"`);
      return span.getBoundingClientRect().width;
    };
    const ctx = /** @type {CanvasRenderingContext2D} */ (document.createElement("canvas").getContext("2d"));
    const out = [];
    let n = 0;
    for (const [web, weight, sizes, localName] of cases) {
      const family = `probe-${n++}`;
      const face = new FontFace(family, `local("${localName}")`);
      try {
        await face.load();
      } catch {
        out.push({ web, weight, local: localName, "size-adjust %": "missing" });
        continue;
      }
      document.fonts.add(face);
      ctx.font = `${weight} 1000px "${web}"`;
      const m = ctx.measureText("H");
      for (const size of sizes) {
        const sizeAdjust = width(web, weight, size) / width(family, weight, size);
        out.push({
          web,
          weight,
          size,
          local: localName,
          "size-adjust %": +(sizeAdjust * 100).toFixed(1),
          "ascent-override %": +((m.fontBoundingBoxAscent / 1000 / sizeAdjust) * 100).toFixed(1),
          "descent-override %": +((m.fontBoundingBoxDescent / 1000 / sizeAdjust) * 100).toFixed(1),
        });
      }
    }
    span.remove();
    return out;
  }, CASES);
  console.table(rows);
} finally {
  await browser.close();
  await close();
}
