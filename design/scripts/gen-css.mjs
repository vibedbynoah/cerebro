#!/usr/bin/env node
// @ts-check
/**
 * Writes the two generated CSS blocks from their single sources, so no copy can drift:
 *
 *  1. tokens.css `light-dark fallback`: every colour token is authored once as
 *     `light-dark(LIGHT, DARK)`. Engines older than Chrome 123 / Safari 17.5 reject light-dark(), so
 *     every such token would compute to nothing (transparent buttons, UA black-on-white). This block
 *     repeats the same values as plain per-theme declarations under
 *     `@supports not (color: light-dark(#000, #fff))`, parsed from the authored declarations.
 *  2. motion.css `bar widths`: `.bar--p0` … `.bar--p100`, the no-JS, any-engine final width of a
 *     comparison bar (`--bar-to: 0.18` for `.bar--p18`).
 *
 * Usage: `node scripts/gen-css.mjs` rewrites both blocks; `--check` exits 1 if either is stale.
 */
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Splits `s` on top-level commas (not inside parentheses). @param {string} s */
function splitTopLevel(s) {
  /** @type {string[]} */
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
    else if (c === "," && depth === 0) {
      parts.push(s.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(s.slice(start));
  return parts.map((p) => p.trim());
}

/** Replaces every `light-dark(A, B)` in `value` with A (light) or B (dark).
 *  @param {string} value @param {"light" | "dark"} theme */
export function resolveLightDark(value, theme) {
  let out = "";
  let i = 0;
  const key = "light-dark(";
  while (i < value.length) {
    const at = value.indexOf(key, i);
    if (at === -1) {
      out += value.slice(i);
      break;
    }
    out += value.slice(i, at);
    let depth = 1;
    let j = at + key.length;
    for (; j < value.length && depth > 0; j++) {
      if (value[j] === "(") depth++;
      else if (value[j] === ")") depth--;
    }
    const inner = value.slice(at + key.length, j - 1);
    const args = splitTopLevel(inner);
    if (args.length !== 2) throw new Error(`light-dark() needs exactly 2 arguments: ${value}`);
    out += resolveLightDark(theme === "light" ? /** @type {string} */ (args[0]) : /** @type {string} */ (args[1]), theme);
    i = j;
  }
  return out;
}

/** Every `--name: value` declaration in `css` whose value uses light-dark(), in source order,
 *  outside the generated block. @param {string} css */
export function lightDarkDeclarations(css) {
  const authored = css.replace(/\/\* BEGIN GENERATED[\s\S]*?END GENERATED[^*]*\*\//g, "");
  const noComments = authored.replace(/\/\*[\s\S]*?\*\//g, "");
  /** @type {Array<{ name: string; value: string }>} */
  const decls = [];
  const re = /(--[\w-]+)\s*:\s*([^;{}]*light-dark\([^;{}]*);/g;
  for (const m of noComments.matchAll(re)) {
    decls.push({ name: /** @type {string} */ (m[1]), value: /** @type {string} */ (m[2]).replace(/\s+/g, " ").trim() });
  }
  return decls;
}

/** @param {Array<{ name: string; value: string }>} decls @param {"light" | "dark"} theme @param {string} indent */
function block(decls, theme, indent) {
  return decls.map((d) => `${indent}${d.name}: ${resolveLightDark(d.value, theme)};`).join("\n");
}

/** @param {string} tokensCss */
export function tokensFallback(tokensCss) {
  const decls = lightDarkDeclarations(tokensCss);
  if (decls.length === 0) throw new Error("no light-dark() declarations found in tokens.css");
  return [
    "@supports not (color: light-dark(#000, #fff)) {",
    "  :root,",
    '  [data-theme="dark"] {',
    block(decls, "dark", "    "),
    "  }",
    "",
    '  [data-theme="light"] {',
    block(decls, "light", "    "),
    "  }",
    "",
    "  @media (prefers-color-scheme: light) {",
    "    :root:not([data-theme]) {",
    block(decls, "light", "      "),
    "    }",
    "  }",
    "}",
  ].join("\n");
}

export function barWidths() {
  const lines = [];
  for (let p = 0; p <= 100; p++) lines.push(`.bar--p${p} { --bar-to: ${String(p / 100)}; }`);
  return lines.join("\n");
}

/** @param {string} css @param {string} label @param {string} body */
export function replaceBlock(css, label, body) {
  const begin = `/* BEGIN GENERATED: ${label} */`;
  const end = `/* END GENERATED: ${label} */`;
  const a = css.indexOf(begin);
  const b = css.indexOf(end);
  if (a === -1 || b === -1 || b < a) throw new Error(`markers for "${label}" not found`);
  return `${css.slice(0, a + begin.length)}\n${body}\n${css.slice(b)}`;
}

async function main() {
  const check = process.argv.includes("--check");
  const jobs = [
    { file: "tokens.css", label: "light-dark fallback", make: tokensFallback },
    { file: "motion.css", label: "bar widths", make: () => barWidths() },
  ];
  let stale = false;
  for (const job of jobs) {
    const file = path.join(DESIGN_ROOT, job.file);
    const raw = await readFile(file, "utf8");
    // Line-ending agnostic (git's autocrlf checks files out as CRLF on Windows): generate and
    // compare on LF, write back with the file's own line ending throughout.
    const eol = raw.includes("\r\n") ? "\r\n" : "\n";
    const css = raw.replace(/\r\n/g, "\n");
    const next = replaceBlock(css, job.label, job.make(css));
    if (next === css) continue;
    if (check) {
      console.error(`${job.file}: generated block "${job.label}" is stale. Run: npm run gen`);
      stale = true;
    } else {
      await writeFile(file, next.replace(/\n/g, eol));
      console.log(`${job.file}: rewrote "${job.label}"`);
    }
  }
  if (stale) process.exit(1);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main();
