#!/usr/bin/env node
// @ts-check
/**
 * Fails if the committed theme-init.js is not exactly what theme-init.ts compiles to.
 *
 * theme-init.js is the one build output that is committed (consumers load it as a classic
 * `<script src>` before first paint, with no bundler in between), so an edit to the .ts that was
 * never rebuilt would ship stale pre-paint code. This compiles the sources into a temp directory
 * with the same tsconfig.build.json and compares bytes (line endings normalised).
 *
 * `--against <file>` compares a different file instead of design/theme-init.js (used by the test
 * that proves this check actually fails on drift).
 */
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const tsc = require.resolve("typescript/bin/tsc");

const i = process.argv.indexOf("--against");
const committed = i !== -1 && process.argv[i + 1] ? path.resolve(/** @type {string} */ (process.argv[i + 1])) : path.join(DESIGN_ROOT, "theme-init.js");

const out = await mkdtemp(path.join(os.tmpdir(), "cerebro-build-check-"));
try {
  execFileSync(process.execPath, [tsc, "-p", path.join(DESIGN_ROOT, "tsconfig.build.json"), "--outDir", out, "--sourceMap", "false"], {
    cwd: DESIGN_ROOT,
    stdio: "inherit",
  });
  const norm = (/** @type {string} */ s) => s.replace(/\r\n/g, "\n");
  const fresh = norm(await readFile(path.join(out, "theme-init.js"), "utf8"));
  const current = norm(await readFile(committed, "utf8"));
  if (fresh !== current) {
    console.error(`${path.relative(DESIGN_ROOT, committed) || committed} is stale: it differs from what theme-init.ts compiles to. Run: npm run build`);
    process.exit(1);
  }
  console.log("theme-init.js matches theme-init.ts");
} finally {
  await rm(out, { recursive: true, force: true });
}
