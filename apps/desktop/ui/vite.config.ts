/// <reference types="vitest/config" />
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import solid from "vite-plugin-solid";
import { CSP_META, headersFile } from "../security.mjs";
import { npmNotices, rustNotices } from "../scripts/third-party.mjs";

/** @see scripts/third-party.mjs */
type Notices = ReturnType<typeof npmNotices>;

const uiDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(uiDir, "..");
const designDir = path.resolve(appDir, "../../design");

/** Attribute-safe text: the policy is fixed, but it is escaped rather than trusted to stay quote-free. */
function escapeAttr(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("'", "&#39;").replaceAll("<", "&lt;");
}

/**
 * - Serves and emits the design system's theme-init.js as a real file: it must be a classic,
 *   render-blocking `<script src>` in <head> (a module runs after first paint), and an external
 *   file is the only kind `script-src 'self'` allows.
 * - Web build only: injects the meta CSP fallback and emits `_headers` for static hosts. The desktop
 *   build carries neither; the Tauri shell sends its own CSP (src-tauri/tauri.conf.json), and a
 *   second, narrower meta policy would block the IPC origins it allows.
 */
function cerebroShell(mode: string): Plugin {
  const themeInit = readFileSync(path.join(designDir, "theme-init.js"), "utf8");
  const web = mode !== "desktop";
  return {
    name: "cerebro-shell",
    configureServer(server) {
      server.middlewares.use("/theme-init.js", (_req, res) => {
        res.setHeader("Content-Type", "text/javascript; charset=utf-8");
        res.end(themeInit);
      });
    },
    // Injected here rather than written in index.html, because Vite warns about (and skips) any
    // classic script it finds in the source. Placed directly after `<meta charset>`, which must stay
    // the first thing in <head> (the parser settles the encoding before reading anything else).
    // Order: charset, CSP meta, theme-init, then the rest.
    transformIndexHtml: {
      order: "post",
      handler(html, ctx) {
        const charset = /<meta charset="utf-8"\s*\/?>/i.exec(html);
        if (!charset) throw new Error('index.html must start <head> with <meta charset="utf-8" />');
        const tags: string[] = [];
        if (web && !ctx.server) tags.push(`<meta http-equiv="Content-Security-Policy" content="${escapeAttr(CSP_META)}">`);
        tags.push('<script src="./theme-init.js"></script>');
        const at = charset.index + charset[0].length;
        return `${html.slice(0, at)}\n    ${tags.join("\n    ")}${html.slice(at)}`;
      },
    },
    generateBundle() {
      this.emitFile({ type: "asset", fileName: "theme-init.js", source: themeInit });
      if (web) this.emitFile({ type: "asset", fileName: "_headers", source: headersFile() });
    },
  };
}

/**
 * `virtual:cerebro-licenses`: everything the bundled Licenses page shows (ui/src/legal/content.ts),
 * compiled in as constants so the page works offline with no request at all (the desktop CSP allows
 * no same-origin fetch).
 * - `apache`: Cerebro's own licence.
 * - `rust`: notices for every crate compiled into the desktop binary (scripts/third-party.mjs).
 * - `bundled`: notices for every node_modules package that ends up in the built JS. Only knowable
 *   once the bundle exists, so the module carries a placeholder that generateBundle replaces with
 *   the list from the final chunks' module ids. The build fails if the placeholder is not found
 *   exactly once, or if any bundled package publishes no licence file.
 * Under the dev server and Vitest nothing is bundled: `bundled` is empty and `bundledComplete` is
 * false, and the page says the list is produced by a build.
 */
const LICENSES_ID = "virtual:cerebro-licenses";
const BUNDLED_TOKEN = "__CEREBRO_BUNDLED_NOTICES__";
function cerebroLicenses(): Plugin {
  let isBuild = false;
  let rust: Notices | null = null;
  return {
    name: "cerebro-licenses",
    configResolved(config) {
      isBuild = config.command === "build";
    },
    resolveId: (id) => (id === LICENSES_ID ? `\0${LICENSES_ID}` : null),
    load(id) {
      if (id !== `\0${LICENSES_ID}`) return null;
      rust ??= rustNotices(path.resolve(appDir, "../.."), path.join(appDir, "licenses"), path.resolve(appDir, "../../LICENSE"));
      const apache = readFileSync(path.resolve(appDir, "../../LICENSE"), "utf8");
      return [
        `export const apache = ${JSON.stringify(apache)};`,
        `export const rust = JSON.parse(${JSON.stringify(JSON.stringify(rust))});`,
        isBuild
          ? `export const bundled = JSON.parse("${BUNDLED_TOKEN}");\nexport const bundledComplete = true;`
          : `export const bundled = { paragraphs: [], docs: [], packages: [] };\nexport const bundledComplete = false;`,
      ].join("\n");
    },
    generateBundle(_options, bundle) {
      const chunks = Object.values(bundle).filter((c) => c.type === "chunk");
      const holders = chunks.filter((c) => c.code.includes(BUNDLED_TOKEN));
      if (holders.length === 0) return; // the Licenses page is not part of this bundle
      if (holders.length !== 1) this.error(`cerebro-licenses: placeholder found in ${holders.length} chunks`);
      const ids = chunks.flatMap((c) => Object.keys(c.modules));
      const notices = npmNotices(ids);
      if (notices.packages.length === 0) this.error("cerebro-licenses: no bundled package found; module ids are not file paths");
      const holder = holders[0]!;
      const pattern = new RegExp(`(["'\`])${BUNDLED_TOKEN}\\1`, "g");
      const matches = holder.code.match(pattern)?.length ?? 0;
      if (matches !== 1) this.error(`cerebro-licenses: expected one quoted placeholder in ${holder.fileName}, found ${matches}`);
      // A JS string literal holding the JSON: JSON.stringify escapes quotes, backslashes and
      // control characters, and the result is parsed by JSON.parse, never evaluated as code.
      const literal = JSON.stringify(JSON.stringify(notices));
      holder.code = holder.code.replace(pattern, () => literal);
      this.emitFile({
        type: "asset",
        fileName: "third-party-packages.json",
        source: JSON.stringify(notices.packages.map((p) => ({ name: p.name, version: p.version, license: p.license })), null, 2),
      });
    },
  };
}

export default defineConfig(({ mode }) => ({
  root: uiDir,
  // Relative asset URLs: the same build works from the Tauri asset protocol and any static host path.
  base: "./",
  publicDir: false,
  // No HMR under Vitest: solid-refresh's virtual module does not resolve in the test runner.
  plugins: [solid({ hot: !process.env["VITEST"] }), cerebroShell(mode), cerebroLicenses()],
  resolve: {
    alias: { "@design": designDir },
  },
  define: {
    __DESKTOP__: JSON.stringify(mode === "desktop"),
  },
  server: {
    port: 5173,
    strictPort: true,
    host: "127.0.0.1",
    fs: { allow: [uiDir, designDir] },
  },
  build: {
    outDir: path.join(uiDir, mode === "desktop" ? "dist-desktop" : "dist"),
    emptyOutDir: true,
    target: "chrome120",
    // Never inline assets as data: URLs (fonts would then need `font-src data:`).
    assetsInlineLimit: 0,
    // The polyfill is only for engines without modulepreload; Chromium and WebView2 have it.
    modulePreload: { polyfill: false },
    sourcemap: false,
    reportCompressedSize: false,
  },
  test: {
    dir: path.join(appDir, "tests/unit"),
    environment: "jsdom",
    globals: false,
    setupFiles: [path.join(appDir, "tests/unit/setup.ts")],
    server: { deps: { inline: [/@ark-ui/, /@zag-js/, /solid-js/, /@solidjs/, /@solid-primitives/] } },
    // Worker threads start far faster than forked processes on Windows, and two workers are enough
    // for five files: the default forks pool timed out before any test ran on a busy machine.
    pool: "threads",
    maxWorkers: 2,
    fileParallelism: true,
  },
}));
