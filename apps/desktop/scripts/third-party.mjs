// @ts-check
/**
 * Third-party notices for the bundled Licenses page, derived from what actually ships rather than
 * from a hand-kept list (vite.config.ts wires both collectors into `virtual:cerebro-third-party`).
 *
 * - `npmNotices`: every node_modules package with at least one module in the built JS chunks, with
 *   its package.json licence field and every licence/notice file it publishes. A package without
 *   one fails the build.
 * - `rustNotices`: every crate compiled into the desktop binary (the normal-dependency closure of
 *   cerebro-desktop from `cargo metadata --locked --offline`, Windows target), with every licence
 *   and notice file in its published package. A crate that publishes none falls back to the
 *   standard text of its declared licence from ../licenses/, and the page says so; a declared
 *   licence with no standard text there fails the build.
 *
 * Both return the same compact shape: licence texts are split into paragraphs and each distinct
 * paragraph is stored once (hundreds of crates share the Apache and MIT bodies), which keeps the
 * lazily loaded legal chunk inside the JavaScript budget. `noticeText` rebuilds a text exactly
 * (line endings normalised to \n).
 */
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

/**
 * @typedef {{ name: string; version: string; license: string; texts: number[]; note?: string }} NoticePackage
 * @typedef {{ paragraphs: string[]; docs: number[][]; packages: NoticePackage[] }} Notices
 */

/** Licence, notice and copyright files a package publishes at its root. */
const LICENSE_FILE = /^(licen[cs]e|copying|notice|copyright|unlicense)([-._].*)?$/i;
/** Files that match the pattern but are code or metadata, not licence text. */
const NOT_TEXT = /\.(js|cjs|mjs|ts|rs|json|toml|spdx|html?)$/i;

/** @param {string} dir */
function licenseFiles(dir) {
  return readdirSync(dir)
    .filter((f) => LICENSE_FILE.test(f) && !NOT_TEXT.test(f) && statSync(path.join(dir, f)).isFile())
    .sort();
}

/** Builds the paragraph-deduplicated shape. */
function createTable() {
  /** @type {Map<string, number>} */
  const paraIndex = new Map();
  /** @type {Map<string, number>} */
  const docIndex = new Map();
  /** @type {number[][]} */
  const docs = [];
  return {
    /** @param {string} text @returns {number} */
    add(text) {
      const norm = text.replace(/\r\n?/g, "\n").replace(/^﻿/, "").trimEnd();
      const known = docIndex.get(norm);
      if (known !== undefined) return known;
      const ids = norm.split("\n\n").map((p) => {
        let i = paraIndex.get(p);
        if (i === undefined) {
          i = paraIndex.size;
          paraIndex.set(p, i);
        }
        return i;
      });
      docs.push(ids);
      docIndex.set(norm, docs.length - 1);
      return docs.length - 1;
    },
    /** @param {NoticePackage[]} packages @returns {Notices} */
    finish(packages) {
      return { paragraphs: [...paraIndex.keys()], docs, packages };
    },
  };
}

/** Rebuilds licence text `i` of a Notices table. */
export function noticeText(/** @type {{ paragraphs: readonly string[]; docs: ReadonlyArray<readonly number[]> }} */ n, /** @type {number} */ i) {
  return (n.docs[i] ?? []).map((p) => n.paragraphs[p] ?? "").join("\n\n");
}

/** Last `node_modules/<pkg>` in a module id, so nested copies resolve to the copy actually used. */
const PKG_IN_ID = /^(.*[\\/]node_modules[\\/])((?:@[^\\/]+[\\/])?[^\\/]+)/;

/**
 * @param {Iterable<string>} moduleIds every module id in the built chunks
 * @returns {Notices}
 */
export function npmNotices(moduleIds) {
  /** @type {Map<string, string>} package name -> directory */
  const found = new Map();
  for (const raw of moduleIds) {
    const id = raw.replace(/^\0/, "").split("?")[0] ?? "";
    // Greedy prefix: the last node_modules segment wins.
    const m = PKG_IN_ID.exec(id);
    if (!m?.[1] || !m[2]) continue;
    const name = m[2].replaceAll("\\", "/");
    if (!found.has(name)) found.set(name, path.join(m[1], m[2]));
  }
  const table = createTable();
  /** @type {NoticePackage[]} */
  const packages = [];
  for (const [name, dir] of [...found].sort(([a], [b]) => a.localeCompare(b))) {
    const pj = /** @type {{ version?: string; license?: unknown }} */ (JSON.parse(readFileSync(path.join(dir, "package.json"), "utf8")));
    const files = licenseFiles(dir);
    if (files.length === 0) throw new Error(`third-party notices: bundled package ${name} publishes no licence file`);
    packages.push({
      name,
      version: pj.version ?? "unknown",
      license: typeof pj.license === "string" ? pj.license : "See licence text",
      texts: files.map((f) => table.add(readFileSync(path.join(dir, f), "utf8"))),
    });
  }
  return table.finish(packages);
}

/**
 * @param {string} workspaceRoot directory holding the workspace Cargo.toml and Cargo.lock
 * @param {string} fallbackDir directory of standard licence texts (MIT.txt, ...)
 * @param {string} apacheFile the Apache-2.0 text
 * @returns {Notices}
 */
export function rustNotices(workspaceRoot, fallbackDir, apacheFile) {
  const out = execFileSync(
    "cargo",
    ["metadata", "--format-version", "1", "--locked", "--offline", "--filter-platform", "x86_64-pc-windows-msvc", "--manifest-path", path.join(workspaceRoot, "Cargo.toml")],
    { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"] },
  );
  /** @type {{ packages: Array<{ id: string; name: string; version: string; license: string | null; license_file: string | null; authors: string[]; repository: string | null; manifest_path: string }>; workspace_members: string[]; resolve: { nodes: Array<{ id: string; deps: Array<{ pkg: string; dep_kinds: Array<{ kind: string | null }> }> }> } }} */
  const meta = JSON.parse(out);
  const byId = new Map(meta.packages.map((p) => [p.id, p]));
  const nodes = new Map(meta.resolve.nodes.map((n) => [n.id, n]));
  const root = meta.packages.find((p) => p.name === "cerebro-desktop" && meta.workspace_members.includes(p.id));
  if (!root) throw new Error("third-party notices: cerebro-desktop is not a workspace member");
  const workspace = new Set(meta.workspace_members);

  // Normal dependencies only: build-script and dev dependencies are not compiled into the binary.
  const seen = new Set([root.id]);
  const queue = [root.id];
  while (queue.length > 0) {
    const node = nodes.get(/** @type {string} */ (queue.shift()));
    for (const dep of node?.deps ?? []) {
      if (!dep.dep_kinds.some((k) => k.kind === null) || seen.has(dep.pkg)) continue;
      seen.add(dep.pkg);
      queue.push(dep.pkg);
    }
  }

  /** @type {Record<string, string>} */
  const standard = {
    MIT: readFileSync(path.join(fallbackDir, "MIT.txt"), "utf8"),
    "BSD-3-Clause": readFileSync(path.join(fallbackDir, "BSD-3-Clause.txt"), "utf8"),
    "MPL-2.0": readFileSync(path.join(fallbackDir, "MPL-2.0.txt"), "utf8"),
    "Apache-2.0": readFileSync(apacheFile, "utf8"),
  };

  const table = createTable();
  /** @type {NoticePackage[]} */
  const packages = [];
  for (const id of seen) {
    if (workspace.has(id)) continue;
    const p = byId.get(id);
    if (!p) continue;
    const dir = path.dirname(p.manifest_path);
    const files = licenseFiles(dir);
    if (p.license_file) {
      const f = path.resolve(dir, p.license_file);
      if (existsSync(f) && !files.some((g) => path.resolve(dir, g) === f)) files.push(path.relative(dir, f));
    }
    const license = p.license ?? (p.license_file ? "See licence text" : "Unknown");
    if (files.length > 0) {
      packages.push({ name: p.name, version: p.version, license, texts: files.map((f) => table.add(readFileSync(path.join(dir, f), "utf8"))) });
      continue;
    }
    // No file published: the standard text of each declared licence, said plainly.
    const ids = (p.license ?? "").split(/\s+(?:OR|AND|WITH)\s+|[()/]/).map((s) => s.trim()).filter(Boolean);
    const texts = ids.filter((s) => standard[s] !== undefined).map((s) => table.add(/** @type {string} */ (standard[s])));
    if (texts.length === 0) throw new Error(`third-party notices: crate ${p.name} ${p.version} publishes no licence file and declares "${license}", which has no standard text in ${fallbackDir}`);
    const who = p.authors.length > 0 ? p.authors.map((a) => a.replace(/\s*<[^>]*>/, "")).join(", ") : p.repository ? `the authors of ${p.repository}` : `the ${p.name} authors`;
    packages.push({
      name: p.name,
      version: p.version,
      license,
      texts,
      note: `The published crate includes no licence file. Shown below is the standard text of its declared licence; the copyright belongs to ${who}.`,
    });
  }
  packages.sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));
  return table.finish(packages);
}
