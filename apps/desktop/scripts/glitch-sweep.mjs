#!/usr/bin/env node
// @ts-nocheck
/**
 * `node scripts/glitch-sweep.mjs` — headless sweep of every screen of the web build (ui/dist) at 1280 and
 * 820 px, dark and light. Reports console errors, uncaught exceptions, horizontal page overflow and
 * elements wider than their scroll container, and saves a shot per case to review/sweep/.
 * Exit code 1 when anything is reported.
 */
import { chromium } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./serve.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "../review/sweep");
const ROUTES = ["home", "hub", "models", "tune", "monitor", "turbo", "chat", "images", "documents", "plugins", "tools", "settings", "about"];
const SRC = await readFile(path.resolve(HERE, "../ui/src/onboarding/progress.ts"), "utf8");
const c = (n) => new RegExp(`export const ${n} = "([^"]+)"`).exec(SRC)?.[1];
const done = JSON.stringify({ v: 1, completed: true, termsVersion: c("TERMS_VERSION"), model: "gemma-4-12b-it" });

await mkdir(OUT, { recursive: true });
const { url: origin, close } = await startServer(0);
const browser = await chromium.launch();
const problems = [];
for (const theme of ["dark", "light"]) {
  for (const width of [1280, 820]) {
    for (const level of ["beginner", "advanced"]) {
      const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: theme });
      const errs = [];
      page.on("console", (m) => m.type() === "error" && errs.push(m.text()));
      page.on("pageerror", (e) => errs.push(String(e)));
      await page.addInitScript(([k, v, lvl]) => {
        localStorage.setItem(k, v);
        localStorage.setItem("cerebro-experience", lvl);
      }, [c("ONBOARDING_KEY"), done, level]);
      for (const route of ROUTES) {
        errs.length = 0;
        await page.goto(`${origin}/#/${route}`);
        await page.waitForTimeout(1200);
        const found = await page.evaluate(() => {
          const out = [];
          const de = document.documentElement;
          if (de.scrollWidth > de.clientWidth + 1) out.push(`page scrolls sideways (${de.scrollWidth} > ${de.clientWidth})`);
          const main = document.querySelector("main");
          if (main) {
            const mr = main.getBoundingClientRect();
            for (const el of main.querySelectorAll("*")) {
              const r = el.getBoundingClientRect();
              if (r.width === 0 || getComputedStyle(el).position === "fixed") continue;
              // Inside a horizontal scroller is fine.
              let p = el.parentElement, scroller = false;
              while (p && p !== main) {
                const ox = getComputedStyle(p).overflowX;
                if (ox === "auto" || ox === "scroll" || ox === "hidden" || ox === "clip") { scroller = true; break; }
                p = p.parentElement;
              }
              if (!scroller && r.right > mr.right + 2) {
                const id = el.className && typeof el.className === "string" ? "." + el.className.split(" ")[0] : el.tagName;
                out.push(`overflows main by ${Math.round(r.right - mr.right)}px: ${id} "${(el.textContent || "").trim().slice(0, 40)}"`);
              }
            }
          }
          // Status bar readouts must never paint over each other.
          const items = [...document.querySelectorAll(".statusbar > *, .statusbar > * > *")].filter((e) => e.getBoundingClientRect().width > 0);
          for (let i = 1; i < items.length; i++) {
            const a = items[i - 1].getBoundingClientRect(), b = items[i].getBoundingClientRect();
            if (items[i - 1].contains(items[i]) || items[i].contains(items[i - 1])) continue;
            if (b.left < a.right - 1 && a.left < b.right - 1) out.push(`status bar overlap: "${items[i - 1].textContent.trim().slice(0, 20)}" / "${items[i].textContent.trim().slice(0, 20)}"`);
          }
          // Text wider than its own clipped box without an ellipsis.
          for (const el of document.querySelectorAll(".statusbar .hud-item")) {
            if (el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflow !== "hidden") out.push(`status bar item spills: "${el.textContent.trim().slice(0, 30)}"`);
          }
          return [...new Set(out)].slice(0, 8);
        });
        const all = [...found, ...errs.map((e) => "console: " + e.slice(0, 200))];
        if (all.length) problems.push({ case: `${route} ${theme} ${width} ${level}`, all });
        await page.screenshot({ path: path.join(OUT, `${route}-${theme}-${width}-${level}.png`) });
      }
      await page.close();
    }
  }
}
await browser.close();
await close();
for (const p of problems) console.log(`\n# ${p.case}\n  - ${p.all.join("\n  - ")}`);
console.log(`\n${problems.length} case(s) with problems`);
process.exit(problems.length ? 1 : 0);
