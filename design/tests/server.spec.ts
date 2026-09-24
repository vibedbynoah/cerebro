import { test, expect } from "@playwright/test";
import http from "node:http";
import fs from "node:fs/promises";
import fsSync from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DESIGN_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/**
 * These exercise design/scripts/serve.mjs directly at the HTTP level — including malformed
 * requests a browser's `fetch()`/`URL` would normalize or refuse to send at all — so they use
 * Node's raw `http` module rather than Playwright's request fixture.
 */
function rawGet(baseURL: string, rawPath: string, headers: Record<string, string> = {}): Promise<{ status: number; headers: http.IncomingHttpHeaders }> {
  const url = new URL(baseURL);
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: url.hostname,
        port: url.port,
        path: rawPath,
        method: "GET",
        headers: { host: `${url.hostname}:${url.port}`, ...headers },
      },
      (res) => {
        res.resume(); // drain, we only need status/headers
        res.on("end", () => resolve({ status: res.statusCode ?? 0, headers: res.headers }));
      },
    );
    req.on("error", reject);
    req.end();
  });
}

test.describe("serve.mjs hardening", () => {
  test("malformed percent-encoding returns 400 and the server keeps serving afterward", async ({ baseURL }) => {
    const bad = await rawGet(baseURL!, "/%E0%A4%A");
    expect(bad.status).toBe(400);

    const ok = await rawGet(baseURL!, "/preview.html");
    expect(ok.status).toBe(200);
  });

  test("paths outside the allowlist (traversal, sibling dir, node_modules, package.json) are rejected", async ({ baseURL }) => {
    for (const path of ["/../design-x/a", "/node_modules/typescript/package.json", "/package.json", "/tsconfig.json", "/scripts/serve.mjs"]) {
      const res = await rawGet(baseURL!, path);
      expect([400, 404], `${path} -> ${res.status}`).toContain(res.status);
    }
  });

  test("an unrecognized Host header is rejected (DNS-rebinding defence)", async ({ baseURL }) => {
    const res = await rawGet(baseURL!, "/preview.html", { host: "evil.example" });
    expect(res.status).toBe(421);
  });

  test("every response carries the hardening headers", async ({ baseURL }) => {
    const res = await rawGet(baseURL!, "/preview.html");
    expect(res.headers["x-content-type-options"]).toBe("nosniff");
    expect(res.headers["content-security-policy"]).toContain("default-src 'none'");
    expect(res.headers["content-security-policy"]).toContain("frame-ancestors 'none'");
    expect(res.headers["referrer-policy"]).toBe("no-referrer");
    expect(res.headers["cross-origin-opener-policy"]).toBe("same-origin");
    expect(res.headers["cross-origin-resource-policy"]).toBe("same-origin");
    expect(res.headers["permissions-policy"]).toContain("camera=()");
  });

  test("the claims-policy exception serves docs/CLAIMS.md read-only, nothing else outside design/", async ({ baseURL }) => {
    const res = await rawGet(baseURL!, "/claims-policy");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toContain("text/plain");
  });

  test("a symlink under design/fonts/ pointing outside design/ is rejected, not served", async ({ baseURL }) => {
    // realpath()-based defence (see serve.mjs): the traversal check in resolveRequestPath only
    // ever sees the UNRESOLVED request path, which for a symlinked directory looks perfectly
    // inside design/fonts/ — only resolving the symlink reveals it points elsewhere.
    const outsideDir = await fs.mkdtemp(path.join(os.tmpdir(), "cerebro-escape-"));
    const outsideFile = path.join(outsideDir, "escape.woff2");
    await fs.writeFile(outsideFile, "not a real font");
    const linkDir = path.join(DESIGN_ROOT, "fonts", "__escape_test__");
    let linked = false;
    try {
      await fs.symlink(outsideDir, linkDir, "junction");
      linked = true;
    } catch {
      // Symlink/junction creation can be blocked by policy on some machines/CI runners; skip
      // rather than fail the suite over an environment limitation unrelated to the server code.
    }
    test.skip(!linked, "could not create a directory symlink/junction on this machine");

    try {
      const res = await rawGet(baseURL!, "/fonts/__escape_test__/escape.woff2");
      expect([400, 404], `escape.woff2 -> ${res.status}`).toContain(res.status);

      const ok = await rawGet(baseURL!, "/preview.html");
      expect(ok.status, "server should still be serving normal requests afterward").toBe(200);
    } finally {
      await fs.rm(linkDir, { recursive: true, force: true }).catch(() => {});
      await fs.rm(outsideDir, { recursive: true, force: true }).catch(() => {});
    }
  });

  test("a file that disappears mid-request does not crash the server (stream error is handled)", async ({ baseURL }) => {
    const tmpName = `__race_test__.css`;
    const tmpPath = path.join(DESIGN_ROOT, tmpName);
    // Large enough that createReadStream's first chunk hasn't necessarily been flushed by the
    // time we unlink below, giving the deliberately-racy delete below a real chance of landing
    // mid-stream rather than before the stream even opens.
    await fs.writeFile(tmpPath, "/* race */\n".repeat(200_000));
    try {
      const req = rawGet(baseURL!, `/${tmpName}`);
      // Best-effort race: delete the file while the response may still be streaming. Whether or
      // not this actually lands inside the read (vs. before it opens, in which case the request
      // just 404s normally), the assertion below — the server survives and keeps serving — holds
      // either way, which is the actual property this test protects: no request, timed how it
      // may, can throw an unhandled 'error' event past serve.mjs and crash the process.
      queueMicrotask(() => {
        fsSync.rm(tmpPath, { force: true }, () => {});
      });
      await req.catch(() => {});
    } finally {
      await fs.rm(tmpPath, { force: true }).catch(() => {});
    }

    const ok = await rawGet(baseURL!, "/preview.html");
    expect(ok.status, "server should still be serving normal requests afterward").toBe(200);
  });

  test("only GET/HEAD are allowed", async ({ baseURL }) => {
    const url = new URL(baseURL!);
    const status = await new Promise<number>((resolve, reject) => {
      const req = http.request(
        { host: url.hostname, port: url.port, path: "/preview.html", method: "POST", headers: { host: `${url.hostname}:${url.port}` } },
        (res) => {
          res.resume();
          res.on("end", () => resolve(res.statusCode ?? 0));
        },
      );
      req.on("error", reject);
      req.end();
    });
    expect(status).toBe(405);
  });
});
