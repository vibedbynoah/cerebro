import { test, expect } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFile, readdir, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FONTS_ROOT = path.join(DESIGN_ROOT, "fonts");

interface Source {
  package: string;
  version: string;
  file: string;
  sha256: string;
}

// Pure Node file-hashing — no browser/page needed, so this runs in the Node context Playwright
// test bodies execute in by default.
test("every shipped woff2's sha256 matches fonts/SOURCES.json", async () => {
  const sources: Source[] = JSON.parse(await readFile(path.join(FONTS_ROOT, "SOURCES.json"), "utf8"));
  expect(sources.length).toBeGreaterThan(0);

  const mismatches: string[] = [];
  for (const entry of sources) {
    const filePath = path.join(FONTS_ROOT, entry.file);
    const buf = await readFile(filePath);
    const actual = createHash("sha256").update(buf).digest("hex");
    if (actual !== entry.sha256) {
      mismatches.push(`${entry.file}: expected ${entry.sha256}, got ${actual}`);
    }
  }
  expect(mismatches, mismatches.join("\n")).toEqual([]);
});

test("every family folder under design/fonts contains OFL.txt, and every shipped woff2 is recorded", async () => {
  const sources: Source[] = JSON.parse(await readFile(path.join(FONTS_ROOT, "SOURCES.json"), "utf8"));
  const recorded = new Set(sources.map((s) => s.file.replace(/\\/g, "/")));

  const entries = await readdir(FONTS_ROOT, { withFileTypes: true });
  const familyDirs = entries.filter((e) => e.isDirectory());
  expect(familyDirs.length).toBeGreaterThan(0);

  const missingLicense: string[] = [];
  const unrecorded: string[] = [];
  for (const dir of familyDirs) {
    const licensePath = path.join(FONTS_ROOT, dir.name, "OFL.txt");
    await access(licensePath).catch(() => missingLicense.push(dir.name));

    const files = await readdir(path.join(FONTS_ROOT, dir.name));
    for (const file of files) {
      if (!file.endsWith(".woff2")) continue;
      const rel = `${dir.name}/${file}`;
      if (!recorded.has(rel)) unrecorded.push(rel);
    }
  }
  expect(missingLicense, `family folders missing OFL.txt: ${missingLicense.join(", ")}`).toEqual([]);
  expect(unrecorded, `woff2 files not recorded in SOURCES.json: ${unrecorded.join(", ")}`).toEqual([]);
});
