#!/usr/bin/env node
// @ts-check
/**
 * Tiny dependency-free static file server for design/, used by `npm run shoot`, the Playwright
 * `webServer` in playwright.config.ts, and manual `npm run serve` while iterating.
 *
 * This is a dev-only tool, but it listens on a real TCP port that any page open in the owner's
 * browser can reach via `fetch('http://127.0.0.1:<port>/...')` — there is no browser same-origin
 * protection against that. So it is hardened as if it mattered:
 *  - every request is handled inside a try/catch; a malformed URL (bad percent-encoding) returns
 *    400 instead of throwing an uncaught URIError that would crash the whole process and take
 *    every other test/screenshot down with it.
 *  - only an explicit allowlist of paths is servable — not "everything under design/", which
 *    would include node_modules, package.json, tsconfig, and test-results.
 *  - the traversal check compares against the root PLUS a trailing separator, so a sibling
 *    directory that merely starts with the same characters (e.g. `design-x`) cannot pass.
 *  - the Host header must name 127.0.0.1 or localhost on the port we're actually listening on,
 *    which defeats DNS rebinding (a public page pointing a subdomain's DNS at 127.0.0.1 and then
 *    asking the browser to fetch it with a Host header the attacker controls).
 *  - every response carries nosniff, no-referrer, same-origin CORP and COOP, a Permissions-Policy
 *    that denies device APIs, and the same strict CSP as preview.html's <meta> tag.
 */
import http from "node:http";
import { createReadStream } from "node:fs";
import { stat, realpath } from "node:fs/promises";
import { pipeline } from "node:stream";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT_WITH_SEP = DESIGN_ROOT + path.sep;
// The one deliberate exception to "only design/ is servable": preview.html's "Read the claims
// policy" link needs docs/CLAIMS.md, one directory above design/. This is a single hardcoded
// target resolved once here — not something derived from a request path — so it can't be turned
// into a traversal primitive the way accepting an arbitrary `../` segment from a URL would be.
const CLAIMS_PATH = path.resolve(DESIGN_ROOT, "..", "docs", "CLAIMS.md");

/** @type {Record<string, string>} */
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/plain; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".ico": "image/x-icon",
};

// Only these paths are ever servable — the page itself, its built JS/CSS/source-maps at the
// design root, and font files/licences one directory level down. Nothing else in design/
// (package.json, tsconfig*.json, node_modules, test-results, scripts/**) is reachable over HTTP.
const ALLOWED_PATH = /^\/(?:preview\.html|claims-policy|[\w-]+\.(?:css|js|mjs|js\.map)|fonts\/[\w-]+\/[\w.-]+\.(?:woff2|txt))$/;

// Only what the page uses: no data: URLs, no inline code, and Trusted Types enforced (the page
// never writes HTML strings). preview.html's <meta> carries the same policy minus frame-ancestors,
// which only works as a header.
export const CSP =
  "default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; require-trusted-types-for 'script'; trusted-types 'none'";

const SECURITY_HEADERS = {
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-read=(), usb=(), serial=(), hid=(), payment=()",
  "Content-Security-Policy": CSP,
};

/** Resolves a request path to an on-disk path, or null if it should be rejected. Never throws —
 *  malformed percent-encoding is caught and treated as a rejection, not a crash.
 *  @param {string} rawUrl
 *  @returns {string | null} */
function resolveRequestPath(rawUrl) {
  let decoded;
  try {
    decoded = decodeURIComponent(new URL(rawUrl, "http://internal").pathname);
  } catch {
    return null;
  }
  const rel = decoded === "/" ? "/preview.html" : decoded;
  if (rel.includes("\0") || !ALLOWED_PATH.test(rel)) return null;
  // A single hardcoded exception (see CLAIMS_PATH above) — not built from the request path, so it
  // can't be widened into a general traversal primitive.
  if (rel === "/claims-policy") return CLAIMS_PATH;
  const resolved = path.resolve(DESIGN_ROOT, "." + rel);
  if (resolved !== DESIGN_ROOT && !resolved.startsWith(ROOT_WITH_SEP)) return null;
  return resolved;
}

/** True if `real` (an already-symlink-resolved absolute path) is the allowed target itself, or
 *  falls inside it when it's a directory root.
 *  @param {string} root @param {string} real */
function isWithin(root, real) {
  return real === root || real.startsWith(root + path.sep);
}

/** @param {string | undefined} hostHeader @param {number} port */
function isAllowedHost(hostHeader, port) {
  if (!hostHeader) return false;
  return hostHeader === `127.0.0.1:${port}` || hostHeader === `localhost:${port}`;
}

/** Starts the server on `port` (0 = ephemeral). Resolves with the base URL and a close() fn. */
export function startServer(port = 4319) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      // Every branch below is synchronous-looking but the file I/O is async; wrapping the whole
      // handler (including that async tail) in try/catch means no request — malformed URL,
      // unreadable file, anything — can throw past this function and crash the process.
      (async () => {
        try {
          for (const [key, value] of Object.entries(SECURITY_HEADERS)) res.setHeader(key, value);

          const actualPort = getActualPort(server, port);
          if (!isAllowedHost(req.headers.host, actualPort)) {
            res.writeHead(421, { "Content-Type": "text/plain; charset=utf-8" }).end("Misdirected Request");
            return;
          }

          if (req.method !== "GET" && req.method !== "HEAD") {
            res.writeHead(405, { "Content-Type": "text/plain; charset=utf-8" }).end("Method Not Allowed");
            return;
          }

          const filePath = resolveRequestPath(req.url ?? "/");
          if (!filePath) {
            res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" }).end("Bad Request");
            return;
          }

          const info = await stat(filePath);
          if (!info.isFile()) throw new Error("not a file");

          // stat() above already follows symlinks for existence/type, but the traversal check in
          // resolveRequestPath ran against the UNRESOLVED request path — a symlink placed
          // anywhere under an otherwise-allowed directory (e.g. design/fonts/some-family/) could
          // point outside design/ entirely and that check would never see it. realpath() resolves
          // every symlink in the chain, so this check runs against where the bytes actually come
          // from, not where the URL claims they come from.
          const real = await realpath(filePath);
          const allowed = filePath === CLAIMS_PATH ? isWithin(CLAIMS_PATH, real) : isWithin(DESIGN_ROOT, real);
          if (!allowed) throw new Error("resolved outside its allowed root");

          const ext = path.extname(real).toLowerCase();
          res.writeHead(200, {
            "Content-Type": MIME[ext] ?? "application/octet-stream",
            "Content-Length": info.size,
            "Cache-Control": "no-cache",
          });
          if (req.method === "HEAD") {
            res.end();
            return;
          }
          // `stream.pipeline` (not `.pipe()`) so a read error mid-stream — the file is deleted,
          // or Windows AV/indexer briefly locks it (EBUSY) — is handled HERE instead of throwing
          // an unhandled `error` event that would crash the whole process and take every other
          // request/test/screenshot down with it.
          await new Promise((resolveStream) => {
            pipeline(createReadStream(real), res, (err) => {
              if (err && !res.writableEnded) res.destroy();
              resolveStream(undefined);
            });
          });
        } catch {
          if (!res.headersSent) {
            res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not Found");
          } else {
            res.end();
          }
        }
      })();
    });
    server.on("error", reject);
    server.on("clientError", (_err, socket) => {
      // A malformed HTTP request line/headers (not a malformed URL — that's handled above)
      // reaches here before `req` even exists. Respond and move on instead of letting the
      // socket (and the process default handler) tear the server down.
      if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    });
    server.listen(port, "127.0.0.1", () => {
      const actualPort = getActualPort(server, port);
      resolve({
        url: `http://127.0.0.1:${actualPort}`,
        close: () => new Promise((res) => server.close(() => res(undefined))),
      });
    });
  });
}

/** @param {import("node:http").Server} server @param {number} fallback */
function getActualPort(server, fallback) {
  const address = server.address();
  return typeof address === "object" && address ? address.port : fallback;
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1].replace(/\\/g, "/")}`;
if (isMain || process.argv[1]?.endsWith("serve.mjs")) {
  const port = Number(process.env.PORT ?? 4319);
  startServer(port).then(({ url }) => {
    console.log(`Serving ${DESIGN_ROOT} at ${url}`);
  });
}
