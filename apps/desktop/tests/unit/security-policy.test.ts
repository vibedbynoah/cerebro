import { describe, expect, it } from "vitest";
import { CSP_HEADER, CSP_META, headersFile, SECURITY_HEADERS } from "../../security.mjs";

describe("web-mode security policy", () => {
  it("is the strict CSP the brief requires, with no unsafe or remote source", () => {
    const d = Object.fromEntries(CSP_HEADER.split("; ").map((part) => [part.split(" ")[0], part.split(" ").slice(1).join(" ")]));
    expect(d["default-src"]).toBe("'self'");
    expect(d["script-src"]).toBe("'self'");
    expect(d["style-src"]).toBe("'self'");
    expect(d["img-src"]).toBe("'self' data:");
    expect(d["object-src"]).toBe("'none'");
    expect(d["base-uri"]).toBe("'none'");
    expect(d["frame-ancestors"]).toBe("'none'");
    expect(d["form-action"]).toBe("'none'");
    expect(CSP_HEADER).not.toMatch(/unsafe|https?:|\*/);
  });

  it("leaves header-only directives out of the meta fallback, where browsers ignore them", () => {
    expect(CSP_META).not.toMatch(/frame-ancestors/);
    expect(CSP_META).toMatch(/default-src 'self'/);
  });

  it("sends every hardening header, with HSTS preload", () => {
    expect(SECURITY_HEADERS["Strict-Transport-Security"]).toBe("max-age=63072000; includeSubDomains; preload");
    expect(SECURITY_HEADERS["X-Content-Type-Options"]).toBe("nosniff");
    expect(SECURITY_HEADERS["Referrer-Policy"]).toBe("strict-origin-when-cross-origin");
    expect(SECURITY_HEADERS["Cross-Origin-Opener-Policy"]).toBe("same-origin");
    expect(SECURITY_HEADERS["Cross-Origin-Resource-Policy"]).toBe("same-origin");
    const pp = SECURITY_HEADERS["Permissions-Policy"] ?? "";
    for (const feature of ["camera", "microphone", "geolocation", "usb", "payment"]) expect(pp).toContain(`${feature}=()`);
    expect(headersFile().startsWith("/*\n")).toBe(true);
    expect(headersFile()).toContain(`  Content-Security-Policy: ${CSP_HEADER}`);
  });
});
