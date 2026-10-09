#!/usr/bin/env node
// @ts-check
/**
 * Post-build gate, run by `npm run build` / `build:desktop`. Fails the build when:
 *  - the JavaScript budget is exceeded (gzipped):
 *      - 150 KB for everything loaded before first paint: theme-init.js, the entry module and every
 *        chunk index.html modulepreloads. This is what every launch pays;
 *      - 50 KB for any one chunk loaded on demand (a screen opened later, setup, the licence text);
 *      - 260 KB for all JavaScript together.
 *    Screens that only some sessions open (Models, setup, Licenses) load as their own chunks from
 *    local disk, so they are held to a per-chunk cap instead of the first-paint budget.
 *    Why the total moved from the old single 150 KB gate to 260 KB: that gate counted every chunk
 *    as if it loaded at launch. The launch cost is still held to the same 150 KB (first paint);
 *    the extra room is only for on-demand chunks, each capped at 50 KB, so no one screen can grow
 *    unnoticed and nothing new reaches first paint;
 *  - index.html has an inline script, an inline event handler, a style attribute, or loads anything
 *    from another origin;
 *  - CSS pulls anything from another origin (@import or url());
 * *  - JavaScript contains a dynamic-code or raw-HTML sink other than Solid's compile-time template
 *    cloning (the one `innerHTML=` in solid-js/web, fed only by constant template strings);
 *  - the desktop build (ui/dist-desktop) still contains a test-only mock failure switch.
 */
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { gzipSync } from "node:zlib";

const INITIAL_BUDGET = 150 * 1024;
const LAZY_CHUNK_BUDGET = 50 * 1024;
const TOTAL_BUDGET = 260 * 1024;
const dist = path.resolve(process.argv[2] ?? "ui/dist");

/** @param {string} dir @returns {Promise<string[]>} */
async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(full)));
    else out.push(full);
  }
  return out;
}

const failures = [];
const files = await walk(dist);
const rel = (/** @type {string} */ f) => path.relative(dist, f).replaceAll("\\", "/");

// ---- Budget ----
const indexHtml = await readFile(path.join(dist, "index.html"), "utf8");
/** Files index.html loads before first paint: classic and module scripts, and modulepreloads. */
const initial = new Set(
  [...indexHtml.matchAll(/<(?:script[^>]*\ssrc|link[^>]*rel="modulepreload"[^>]*\shref)="\.\/([^"]+)"/g)].map((m) => path.resolve(dist, m[1] ?? "")),
);
let gz = 0;
let initialGz = 0;
for (const file of files.filter((f) => f.endsWith(".js"))) {
  const size = gzipSync(await readFile(file)).length;
  gz += size;
  if (initial.has(file)) initialGz += size;
  else if (size > LAZY_CHUNK_BUDGET) failures.push(`${rel(file)} is ${(size / 1024).toFixed(1)} KB gzipped, over the ${LAZY_CHUNK_BUDGET / 1024} KB on-demand chunk budget`);
}
if (initial.size < 2) failures.push("index.html names no entry script: the first-paint budget cannot be measured");
if (initialGz > INITIAL_BUDGET) failures.push(`first-paint JS is ${(initialGz / 1024).toFixed(1)} KB gzipped, over the ${INITIAL_BUDGET / 1024} KB budget`);
if (gz > TOTAL_BUDGET) failures.push(`all JS together is ${(gz / 1024).toFixed(1)} KB gzipped, over the ${TOTAL_BUDGET / 1024} KB budget`);

// ---- HTML ----
for (const file of files.filter((f) => f.endsWith(".html"))) {
  const html = await readFile(file, "utf8");
  for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
    const attrs = m[1] ?? "";
    if (!/\bsrc\s*=/.test(attrs) || (m[2] ?? "").trim() !== "") failures.push(`${rel(file)}: inline <script>`);
  }
  if (/<[a-z][^>]*\son[a-z]+\s*=/i.test(html)) failures.push(`${rel(file)}: inline event handler attribute`);
  if (/<[a-z][^>]*\sstyle\s*=/i.test(html)) failures.push(`${rel(file)}: style attribute`);
  if (/<style\b/i.test(html)) failures.push(`${rel(file)}: inline <style>`);
  for (const m of html.matchAll(/\s(?:src|href|action|poster|data)\s*=\s*["']?\s*(?:https?:)?\/\//gi)) {
    failures.push(`${rel(file)}: off-origin reference ${m[0].trim()}`);
  }
}

// ---- CSS ----
for (const file of files.filter((f) => f.endsWith(".css"))) {
  const css = await readFile(file, "utf8");
  if (/@import\s+(?:url\()?\s*["']?(?:https?:)?\/\//i.test(css)) failures.push(`${rel(file)}: off-origin @import`);
  if (/url\(\s*["']?(?:https?:)?\/\//i.test(css)) failures.push(`${rel(file)}: off-origin url()`);
}

// ---- JS sinks ----
for (const file of files.filter((f) => f.endsWith(".js"))) {
  const js = await readFile(file, "utf8");
  if (/\beval\s*\(/.test(js)) failures.push(`${rel(file)}: eval(`);
  if (/new\s+Function\s*\(/.test(js)) failures.push(`${rel(file)}: new Function(`);
  if (/\b(?:insertAdjacentHTML|outerHTML\s*=|document\.write)/.test(js)) failures.push(`${rel(file)}: raw HTML sink`);
  const innerHtmlWrites = js.match(/\.innerHTML\s*=/g)?.length ?? 0;
  if (innerHtmlWrites > 1) failures.push(`${rel(file)}: ${innerHtmlWrites} innerHTML writes (only Solid's template cloning is allowed)`);
}

// ---- Test-only switches never ship in the desktop app ----
// The mock's failure switches (`?mock=fail-hw`, mock-service.ts) are read only when !__DESKTOP__;
// their query values must have been compiled out of the desktop build.
if (path.basename(dist) === "dist-desktop") {
  for (const file of files.filter((f) => f.endsWith(".js"))) {
    const js = await readFile(file, "utf8");
    for (const key of ["fail-hw", "fail-catalog", "bad-hw", "empty-library", "fail-library", "slow-library", "fast-tune", "tune-hold-screen", "tune-hold-soak", "fail-tune"]) if (js.includes(key)) failures.push(`${rel(file)}: test-only mock switch "${key}" in the desktop build`);
  }
}

if (failures.length > 0) {
  console.error(`check-bundle: ${failures.length} problem(s) in ${dist}`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(
  `check-bundle: OK (gzipped JS: first paint ${(initialGz / 1024).toFixed(1)} of ${INITIAL_BUDGET / 1024} KB, all ${(gz / 1024).toFixed(1)} of ${TOTAL_BUDGET / 1024} KB; no inline code, no off-origin references)`,
);
