#!/usr/bin/env node
// @ts-check
/**
 * `npm run fonts:sync` — copies the woff2 files and OFL licence this design system actually
 * ships from the pinned `@fontsource-variable/*` devDependencies into design/fonts/, and writes
 * design/fonts/SOURCES.json recording exactly which package version and file each one came from,
 * plus its sha256. `npm test`'s font-provenance check re-hashes every shipped woff2 and compares
 * it against this file, so a font file can no longer sit in the repo with no record of where it
 * came from or whether it has been tampered with since.
 *
 * Run this after bumping a `@fontsource-variable/*` version in package.json, then re-run
 * `npm test` before committing the result.
 */
import { createHash } from "node:crypto";
import { copyFile, readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const NODE_MODULES = path.join(DESIGN_ROOT, "node_modules");
const FONTS_ROOT = path.join(DESIGN_ROOT, "fonts");

/**
 * Which files come out of which package, and where they land. Inter and JetBrains Mono use
 * Fontsource's registered-axis-only ("wght") build. Fraunces uses the "standard" build
 * (wght + opsz — enough for `font-optical-sizing: auto` to pick up its large-size glyphs at
 * display sizes) rather than the "full" build (wght + opsz + SOFT + WONK, ~2x the bytes for two
 * axes this system never varies).
 */
const MAP = [
  { pkg: "@fontsource-variable/inter", dir: "inter", file: "files/inter-latin-wght-normal.woff2" },
  { pkg: "@fontsource-variable/inter", dir: "inter", file: "files/inter-latin-ext-wght-normal.woff2" },
  { pkg: "@fontsource-variable/inter", dir: "inter", file: "files/inter-latin-wght-italic.woff2" },
  { pkg: "@fontsource-variable/inter", dir: "inter", file: "files/inter-latin-ext-wght-italic.woff2" },
  { pkg: "@fontsource-variable/fraunces", dir: "fraunces", file: "files/fraunces-latin-standard-normal.woff2" },
  { pkg: "@fontsource-variable/fraunces", dir: "fraunces", file: "files/fraunces-latin-ext-standard-normal.woff2" },
  { pkg: "@fontsource-variable/jetbrains-mono", dir: "jetbrains-mono", file: "files/jetbrains-mono-latin-wght-normal.woff2" },
  { pkg: "@fontsource-variable/jetbrains-mono", dir: "jetbrains-mono", file: "files/jetbrains-mono-latin-ext-wght-normal.woff2" },
];

/** @param {string} filePath */
async function sha256(filePath) {
  const buf = await readFile(filePath);
  return createHash("sha256").update(buf).digest("hex");
}

async function main() {
  /** @type {Array<{package: string, version: string, file: string, sha256: string}>} */
  const sources = [];
  const seenDirs = new Set();

  for (const { pkg, dir, file } of MAP) {
    const pkgRoot = path.join(NODE_MODULES, ...pkg.split("/"));
    const src = path.join(pkgRoot, file);
    const destDir = path.join(FONTS_ROOT, dir);
    await mkdir(destDir, { recursive: true });
    const destFile = path.join(destDir, path.basename(file));
    await copyFile(src, destFile);

    if (!seenDirs.has(dir)) {
      seenDirs.add(dir);
      await copyFile(path.join(pkgRoot, "LICENSE"), path.join(destDir, "OFL.txt"));
    }

    const { version } = JSON.parse(await readFile(path.join(pkgRoot, "package.json"), "utf8"));
    sources.push({
      package: pkg,
      version,
      file: path.posix.join(dir, path.basename(file)),
      sha256: await sha256(destFile),
    });
    console.log("synced", pkg, "->", path.relative(DESIGN_ROOT, destFile));
  }

  await writeFile(path.join(FONTS_ROOT, "SOURCES.json"), JSON.stringify(sources, null, 2) + "\n");
  console.log("wrote", path.relative(DESIGN_ROOT, path.join(FONTS_ROOT, "SOURCES.json")));
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
