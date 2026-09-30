#!/usr/bin/env node
// @ts-check
/**
 * Minimal static server for the web-mode build (ui/dist), used by Playwright, `npm run shoot` and
 * `npm run preview`. Sends exactly the production security headers (security.mjs), binds to
 * 127.0.0.1 only, serves files strictly inside the dist folder, and answers GET/HEAD only.
 */
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { SECURITY_HEADERS } from "../security.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const DIST = path.resolve(HERE, "../ui/dist");

/** @type {Record<string, string>} */
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".woff2": "font/woff2",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".json": "application/json",
  ".txt": "text/plain; charset=utf-8",
};

/**
 * @param {number} port 0 picks a free port.
 * @returns {Promise<{ url: string; close: () => Promise<void> }>}
 */
export function startServer(port = 0) {
  const server = createServer(async (req, res) => {
    for (const [name, value] of Object.entries(SECURITY_HEADERS)) res.setHeader(name, value);
    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405, { Allow: "GET, HEAD" }).end();
      return;
    }
    try {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const rel = decodeURIComponent(url.pathname).replace(/^\/+/, "") || "index.html";
      const file = path.resolve(DIST, rel);
      if (file !== DIST && !file.startsWith(DIST + path.sep)) {
        res.writeHead(403).end();
        return;
      }
      const info = await stat(file);
      if (!info.isFile()) throw new Error("not a file");
      const body = await readFile(file);
      res.writeHead(200, {
        "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream",
        "Content-Length": body.length,
        "Cache-Control": "no-store",
      });
      res.end(req.method === "HEAD" ? undefined : body);
    } catch {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not found");
    }
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      const addr = server.address();
      const actual = typeof addr === "object" && addr ? addr.port : port;
      resolve({
        url: `http://127.0.0.1:${actual}`,
        close: () => new Promise((done) => server.close(() => done())),
      });
    });
  });
}

// Run directly: `node scripts/serve.mjs` (PORT env, default 4173).
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const port = Number(process.env["PORT"] ?? 4173);
  const { url, close } = await startServer(Number.isInteger(port) && port > 0 && port < 65536 ? port : 4173);
  console.log(`Serving ${DIST} at ${url}`);
  const stop = () => void close().then(() => process.exit(0));
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
}
