/**
 * Brand identity (design/brand/BRAND.md) as data: the brand files are in the repo, the app's inline
 * mark and wordmark match the masters exactly, every Tauri icon and the installer art are the brand
 * rasters, the app CSS introduces no accent but the blue tokens, and the copy keeps BRAND.md's voice.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ICNS_TYPES, INSTALLER_ART, parseIco, PNG_ICONS } from "../../scripts/brand-icons.mjs";
import { MARK_IDLE, MARK_PICKED, MARK_ROUTER, MARK_STEM, MOTIF_ROUTE_D, WORDMARK_D } from "../../ui/src/components/Brand";

const APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const BRAND = path.resolve(APP, "../../design/brand");
const read = (p: string) => readFileSync(p, "utf8");
const svg = (name: string) => read(path.join(BRAND, "svg", name));

/** [x, y, w, h] of every <rect> in an SVG fragment (absent x/y are 0). */
function rects(fragment: string): number[][] {
  return [...fragment.matchAll(/<rect\b([^>]*)\/?>/g)].map((m) => {
    const attr = (n: string, d = 0) => Number(new RegExp(`\\s${n}="([^"]+)"`).exec(m[1] ?? "")?.[1] ?? d);
    return [attr("x"), attr("y"), attr("width"), attr("height")];
  });
}

function pngSize(buf: Buffer): [number, number] {
  expect(buf.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

describe("brand files", () => {
  it("design/brand holds BRAND.md, the masters, the motif and the generated icon set", () => {
    expect(read(path.join(BRAND, "BRAND.md"))).toMatch(/## Voice/);
    for (const f of ["mark.svg", "mark-on-dark.svg", "mark-mono.svg", "mark-small.svg", "wordmark.svg", "motif-cells.svg", "motif-routes.svg", "lockup-horizontal.svg", "lockup-stacked.svg"]) {
      expect(statSync(path.join(BRAND, "svg", f)).size, f).toBeGreaterThan(100);
    }
    for (const f of ["windows/cerebro.ico", "png/cerebro-1024.png", "macos/cerebro.iconset/icon_512x512@2x.png", "installer/sidebar.bmp"]) {
      expect(statSync(path.join(BRAND, "out", f)).size, f).toBeGreaterThan(100);
    }
  });

  it("the lockups carry the real wordmark path", () => {
    for (const f of ["lockup-horizontal.svg", "lockup-horizontal-on-dark.svg", "lockup-stacked.svg", "lockup-stacked-on-dark.svg"]) {
      expect(svg(f), f).toContain(`d="${WORDMARK_D}"`);
    }
  });
});

describe("the app draws the masters exactly", () => {
  const master = svg("mark.svg");
  const [ink, blue] = [/<g fill="#101622">(.*?)<\/g>/s.exec(master)?.[1] ?? "", /<g fill="#0069e8">(.*?)<\/g>/s.exec(master)?.[1] ?? ""];

  it("idle experts, picked expert, stem and router match mark.svg", () => {
    expect(rects(ink)).toEqual(MARK_IDLE.map(([x, y]) => [x, y, 6, 6]));
    expect(rects(blue)).toEqual([[MARK_PICKED[0], MARK_PICKED[1], 6, 6]]);
    expect(blue).toContain(`d="${MARK_STEM}"`);
    expect(blue).toContain(`<circle cx="${MARK_ROUTER.cx}" cy="${MARK_ROUTER.cy}" r="${MARK_ROUTER.r}"/>`);
    // Seven experts, one of them picked: the idea of the mark.
    expect(MARK_IDLE.length + 1).toBe(7);
  });

  it("the wordmark and the motif routes match their masters", () => {
    expect(svg("wordmark.svg")).toContain(`d="${WORDMARK_D}"`);
    expect(svg("motif-routes.svg")).toContain(`d="${MOTIF_ROUTE_D}"`);
  });

  it("the old dial mark is gone from the app", () => {
    const icons = read(path.join(APP, "ui/src/components/icons.tsx"));
    expect(icons).not.toMatch(/instrument dial|BrandMark/);
  });
});

describe("Tauri icons and installer art are the brand rasters", () => {
  const conf = JSON.parse(read(path.join(APP, "src-tauri/tauri.conf.json"))) as {
    bundle: { icon: string[]; windows?: { nsis?: { installerIcon?: string; headerImage?: string; sidebarImage?: string } } };
  };

  it("every PNG icon is byte-identical to its brand source and the right size", () => {
    const sizes: Record<string, number> = { "32x32.png": 32, "128x128.png": 128, "128x128@2x.png": 256, "icon.png": 512 };
    for (const [name, src] of Object.entries(PNG_ICONS)) {
      const buf = readFileSync(path.join(APP, "src-tauri/icons", name));
      expect(buf.equals(readFileSync(path.join(BRAND, "out", src))), name).toBe(true);
      expect(pngSize(buf), name).toEqual([sizes[name], sizes[name]]);
    }
  });

  it("icon.ico holds every Windows size, 32 px first (the window and taskbar icon)", () => {
    const entries = parseIco(readFileSync(path.join(APP, "src-tauri/icons/icon.ico")));
    expect(entries[0]?.width).toBe(32);
    expect(entries.map((e) => e.width).sort((a, b) => a - b)).toEqual([16, 20, 24, 32, 40, 48, 64, 96, 128, 256]);
    const source = parseIco(readFileSync(path.join(BRAND, "out/windows/cerebro.ico")));
    for (const e of entries) expect(source.some((s) => s.width === e.width && s.data.equals(e.data)), `${e.width}px`).toBe(true);
  });

  it("icon.icns holds Apple's ten sizes from the macOS iconset", () => {
    const buf = readFileSync(path.join(APP, "src-tauri/icons/icon.icns"));
    expect(buf.subarray(0, 4).toString("ascii")).toBe("icns");
    expect(buf.readUInt32BE(4)).toBe(buf.length);
    const types: string[] = [];
    for (let at = 8; at < buf.length; at += buf.readUInt32BE(at + 4)) {
      types.push(buf.subarray(at, at + 4).toString("ascii"));
      pngSize(buf.subarray(at + 8));
    }
    expect(types).toEqual(ICNS_TYPES.map(([t]) => t));
  });

  it("tauri.conf.json bundles the icons for every platform and the NSIS art", () => {
    expect(conf.bundle.icon).toEqual(["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.png", "icons/icon.icns", "icons/icon.ico"]);
    for (const icon of conf.bundle.icon) expect(statSync(path.join(APP, "src-tauri", icon)).size, icon).toBeGreaterThan(0);
    const nsis = conf.bundle.windows?.nsis;
    expect(nsis).toEqual({ installerIcon: "icons/icon.ico", headerImage: "installer/header.bmp", sidebarImage: "installer/sidebar.bmp" });
    const dims: Record<string, [number, number]> = { "header.bmp": [150, 57], "sidebar.bmp": [164, 314] };
    for (const [name, src] of Object.entries(INSTALLER_ART)) {
      const buf = readFileSync(path.join(APP, "src-tauri/installer", name));
      expect(buf.equals(readFileSync(path.join(BRAND, "out", src))), name).toBe(true);
      expect(buf.subarray(0, 2).toString("ascii")).toBe("BM");
      expect([buf.readInt32LE(18), buf.readInt32LE(22)], name).toEqual(dims[name]);
      expect(buf.readUInt16LE(28), `${name} is 24-bit`).toBe(24);
    }
  });
});

const SRC = path.join(APP, "ui/src");
function files(dir: string, re: RegExp): string[] {
  return readdirSync(dir).flatMap((n) => {
    const full = path.join(dir, n);
    return statSync(full).isDirectory() ? files(full, re) : re.test(n) ? [full] : [];
  });
}

describe("one accent: the website's blue", () => {
  it("the app CSS adds no colour of its own outside the cool slate and blue hues", () => {
    for (const file of files(path.join(SRC, "styles"), /\.css$/)) {
      const css = read(file);
      for (const m of css.matchAll(/oklch\(\s*([\d.]+)%?\s+([\d.]+)\s+([\d.]+)/g)) {
        const [chroma, hue] = [Number(m[2]), Number(m[3])];
        if (chroma >= 0.02) expect(hue, `${path.basename(file)}: ${m[0]}`).toBeGreaterThanOrEqual(230);
        if (chroma >= 0.02) expect(hue, `${path.basename(file)}: ${m[0]}`).toBeLessThanOrEqual(275);
      }
      // Hex only as mask ink (#000 / #0000), never as a colour anyone sees.
      for (const m of css.matchAll(/#[0-9a-f]{3,8}\b/gi)) expect(["#000", "#0000"], `${path.basename(file)}: ${m[0]}`).toContain(m[0]);
      expect(css, path.basename(file)).not.toMatch(/\b(amber|orange)\b/i);
    }
  });

  it("the design tokens keep the blue accent in both themes", () => {
    const tokens = read(path.resolve(APP, "../../design/tokens.css"));
    const accent = /--color-accent:\s*light-dark\(oklch\([\d.]+% [\d.]+ (\d+)\), oklch\([\d.]+% [\d.]+ (\d+)\)\)/.exec(tokens);
    expect(accent?.slice(1).map(Number)).toEqual([258, 235]);
  });
});

describe("copy keeps BRAND.md's voice", () => {
  const ui = files(SRC, /\.(tsx|ts)$/).filter((f) => !f.includes(`${path.sep}mock${path.sep}`));

  it("no hype words and no feature described by pointing at another product", () => {
    for (const file of ui) {
      const text = read(file);
      expect(text, path.relative(SRC, file)).not.toMatch(/\b(revolutionary|magical|supercharge[ds]?|unleash|blazing)\b/i);
      expect(text, path.relative(SRC, file)).not.toMatch(/(LM Studio|Ollama)[- ]style|like (LM Studio|Ollama)|in the style of|inspired by|LM Studio's load panel/i);
    }
  });

  it("other products appear only where the user needs the interoperability fact", () => {
    const allowed = new Set(["service/service.ts", "screens/models/Library.tsx", "screens/models/ModelsScreen.tsx", "screens/models/DiskPanel.tsx", "screens/models/parts.tsx"]);
    for (const file of ui) {
      const rel = path.relative(SRC, file).split(path.sep).join("/");
      if (/LM Studio|Ollama/.test(read(file))) expect(allowed, rel).toContain(rel);
    }
  });

  it("Cerebro is never written in capitals or with a suffix", () => {
    for (const file of ui) expect(read(file), path.relative(SRC, file)).not.toMatch(/CEREBRO\b|Cerebro AI|CerebroGPT|the Cerebro\b/);
  });
});
