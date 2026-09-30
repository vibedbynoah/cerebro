/// <reference types="vitest/config" />
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import solid from "vite-plugin-solid";
import { CSP_META, headersFile } from "../security.mjs";

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

export default defineConfig(({ mode }) => ({
  root: uiDir,
  // Relative asset URLs: the same build works from the Tauri asset protocol and any static host path.
  base: "./",
  publicDir: false,
  // No HMR under Vitest: solid-refresh's virtual module does not resolve in the test runner.
  plugins: [solid({ hot: !process.env["VITEST"] }), cerebroShell(mode)],
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
