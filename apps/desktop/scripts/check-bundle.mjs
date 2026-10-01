#!/usr/bin/env node
// @ts-check
/**
 * Post-build gate, run by `npm run build` / `build:desktop`. Fails the build when:
 *  - gzipped JavaScript (all chunks plus theme-init.js) exceeds the 150 KB budget;
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

const BUDGET_BYTES = 150 * 1024;
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
let gz = 0;
for (const file of files.filter((f) => f.endsWith(".js"))) gz += gzipSync(await readFile(file)).length;
if (gz > BUDGET_BYTES) failures.push(`gzipped JS is ${(gz / 1024).toFixed(1)} KB, over the ${BUDGET_BYTES / 1024} KB budget`);

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
    for (const key of ["fail-hw", "fail-catalog", "bad-hw"]) if (js.includes(key)) failures.push(`${rel(file)}: test-only mock switch "${key}" in the desktop build`);
  }
}

if (failures.length > 0) {
  console.error(`check-bundle: ${failures.length} problem(s) in ${dist}`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check-bundle: OK (gzipped JS ${(gz / 1024).toFixed(1)} KB of ${BUDGET_BYTES / 1024} KB; no inline code, no off-origin references)`);
