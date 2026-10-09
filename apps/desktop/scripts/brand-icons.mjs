#!/usr/bin/env node
// @ts-check
/**
 * `npm run brand` — installs Cerebro's brand rasters (design/brand/out, generated from the masters
 * by the website's brand build; design/brand/BRAND.md) as the desktop app's icons and installer art:
 *
 *   src-tauri/icons/32x32.png, 128x128.png, 128x128@2x.png, icon.png (512)   Linux bundles, window icon
 *   src-tauri/icons/icon.ico                                               Windows exe, taskbar, NSIS
 *   src-tauri/icons/icon.icns                                              macOS bundle
 *   src-tauri/installer/header.bmp (150 x 57), sidebar.bmp (164 x 314)     NSIS MUI2 art
 *
 * Deterministic and dependency-free: PNG and BMP bytes are copied as they are; the ICO is the same
 * images reordered so the 32 px entry comes first (Tauri's codegen takes the first ICO entry as the
 * default window icon, and 32 px is what Windows asks for there); the ICNS is assembled from the
 * macOS iconset PNGs. tests/unit/brand.test.ts checks every output against its source.
 *
 * `--sync <dir>` first copies missing files from a brand folder (the website's site/brand).
 */
import { copyFile, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, "..");
export const BRAND = path.resolve(APP, "../../design/brand");
const ICONS = path.join(APP, "src-tauri/icons");
const INSTALLER = path.join(APP, "src-tauri/installer");

/** Tauri icon file <- brand raster (relative to design/brand/out). */
export const PNG_ICONS = /** @type {const} */ ({
  "32x32.png": "windows/cerebro-32.png",
  "128x128.png": "windows/cerebro-128.png",
  "128x128@2x.png": "windows/cerebro-256.png",
  "icon.png": "png/cerebro-512.png",
});

export const INSTALLER_ART = /** @type {const} */ ({
  "header.bmp": "installer/header.bmp",
  "sidebar.bmp": "installer/sidebar.bmp",
});

/** ICNS element type <- iconset file. PNG payloads; every type here accepts PNG on macOS 10.7+. */
export const ICNS_TYPES = /** @type {const} */ ([
  ["icp4", "icon_16x16.png"],
  ["icp5", "icon_32x32.png"],
  ["ic11", "icon_16x16@2x.png"],
  ["ic12", "icon_32x32@2x.png"],
  ["ic07", "icon_128x128.png"],
  ["ic13", "icon_128x128@2x.png"],
  ["ic08", "icon_256x256.png"],
  ["ic14", "icon_256x256@2x.png"],
  ["ic09", "icon_512x512.png"],
  ["ic10", "icon_512x512@2x.png"],
]);

/**
 * @typedef {{ width: number, height: number, colors: number, planes: number, bpp: number, data: Buffer }} IcoEntry
 * @param {Buffer} buf
 * @returns {IcoEntry[]}
 */
export function parseIco(buf) {
  if (buf.length < 6 || buf.readUInt16LE(0) !== 0 || buf.readUInt16LE(2) !== 1) throw new Error("not an ICO file");
  const count = buf.readUInt16LE(4);
  if (count === 0 || buf.length < 6 + count * 16) throw new Error("ICO directory is truncated");
  /** @type {IcoEntry[]} */
  const out = [];
  for (let i = 0; i < count; i++) {
    const e = 6 + i * 16;
    const size = buf.readUInt32LE(e + 8);
    const offset = buf.readUInt32LE(e + 12);
    if (offset + size > buf.length || offset < 6 + count * 16) throw new Error(`ICO entry ${i} points outside the file`);
    out.push({
      width: buf[e] || 256,
      height: buf[e + 1] || 256,
      colors: buf.readUInt8(e + 2),
      planes: buf.readUInt16LE(e + 4),
      bpp: buf.readUInt16LE(e + 6),
      data: buf.subarray(offset, offset + size),
    });
  }
  return out;
}

/** @param {IcoEntry[]} entries */
export function writeIco(entries) {
  const head = Buffer.alloc(6 + entries.length * 16);
  head.writeUInt16LE(1, 2);
  head.writeUInt16LE(entries.length, 4);
  let offset = head.length;
  entries.forEach((e, i) => {
    const d = 6 + i * 16;
    head[d] = e.width >= 256 ? 0 : e.width;
    head[d + 1] = e.height >= 256 ? 0 : e.height;
    head[d + 2] = e.colors;
    head.writeUInt16LE(e.planes, d + 4);
    head.writeUInt16LE(e.bpp, d + 6);
    head.writeUInt32LE(e.data.length, d + 8);
    head.writeUInt32LE(offset, d + 12);
    offset += e.data.length;
  });
  return Buffer.concat([head, ...entries.map((e) => e.data)]);
}

/** 32 px first, the rest in ascending size. @param {IcoEntry[]} entries */
export function windowIconFirst(entries) {
  const first = entries.find((e) => e.width === 32);
  if (!first) throw new Error("the ICO has no 32 px entry");
  return [first, ...entries.filter((e) => e !== first).sort((a, b) => a.width - b.width)];
}

/** @param {Array<[string, Buffer]>} parts */
export function writeIcns(parts) {
  const chunks = parts.map(([type, data]) => {
    const h = Buffer.alloc(8);
    h.write(type, 0, "ascii");
    h.writeUInt32BE(data.length + 8, 4);
    return Buffer.concat([h, data]);
  });
  const head = Buffer.alloc(8);
  head.write("icns", 0, "ascii");
  head.writeUInt32BE(8 + chunks.reduce((n, c) => n + c.length, 0), 4);
  return Buffer.concat([head, ...chunks]);
}

/** @param {string} p */
const exists = (p) =>
  stat(p).then(
    () => true,
    () => false,
  );

/** Copies every file of `from` that `to` does not have yet. @param {string} from @param {string} to */
async function syncMissing(from, to) {
  let copied = 0;
  for (const entry of await readdir(from, { withFileTypes: true })) {
    const a = path.join(from, entry.name);
    const b = path.join(to, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "scripts" || entry.name === "node_modules") continue;
      await mkdir(b, { recursive: true });
      copied += await syncMissing(a, b);
    } else if (!(await exists(b))) {
      await copyFile(a, b);
      copied++;
    }
  }
  return copied;
}

async function main() {
  const at = process.argv.indexOf("--sync");
  if (at !== -1) {
    const from = process.argv[at + 1];
    if (!from) throw new Error("--sync needs a folder");
    await mkdir(BRAND, { recursive: true });
    console.log(`brand: copied ${await syncMissing(path.resolve(from), BRAND)} missing file(s) into design/brand`);
  }
  const out = path.join(BRAND, "out");
  await mkdir(ICONS, { recursive: true });
  await mkdir(INSTALLER, { recursive: true });
  for (const [name, src] of Object.entries(PNG_ICONS)) await copyFile(path.join(out, src), path.join(ICONS, name));
  for (const [name, src] of Object.entries(INSTALLER_ART)) await copyFile(path.join(out, src), path.join(INSTALLER, name));
  const ico = windowIconFirst(parseIco(await readFile(path.join(out, "windows/cerebro.ico"))));
  await writeFile(path.join(ICONS, "icon.ico"), writeIco(ico));
  /** @type {Array<[string, Buffer]>} */
  const icns = [];
  for (const [type, file] of ICNS_TYPES) icns.push([type, await readFile(path.join(out, "macos/cerebro.iconset", file))]);
  await writeFile(path.join(ICONS, "icon.icns"), writeIcns(icns));
  console.log(`brand: wrote ${Object.keys(PNG_ICONS).length + 2} icons into src-tauri/icons and the NSIS art into src-tauri/installer`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
