// @ts-check
/**
 * The one source of truth for the web-mode security policy. Read by ui/vite.config.ts (meta CSP in
 * index.html, and the `_headers` file emitted into dist for static hosts), by scripts/serve.mjs
 * (the test/preview server sends exactly these headers) and by tests/e2e/security.spec.ts.
 *
 * The desktop shell has its own CSP in src-tauri/tauri.conf.json, because it must also allow the
 * Tauri IPC origins; tests/unit/tauri-config.test.ts keeps the two aligned on every shared directive.
 */

/** @type {Readonly<Record<string, readonly string[]>>} */
export const CSP_DIRECTIVES = Object.freeze({
  "default-src": ["'self'"],
  "script-src": ["'self'"],
  "style-src": ["'self'"],
  "img-src": ["'self'", "data:"],
  "font-src": ["'self'"],
  "connect-src": ["'self'"],
  "media-src": ["'none'"],
  "worker-src": ["'none'"],
  "manifest-src": ["'none'"],
  "frame-src": ["'none'"],
  "object-src": ["'none'"],
  "base-uri": ["'none'"],
  // No <form> posts anywhere in the app; every input is handled in script.
  "form-action": ["'none'"],
  "frame-ancestors": ["'none'"],
});

/** Browsers ignore these in a <meta> policy and log an error for them, so the meta fallback omits
 *  them (the header carries them). */
const HEADER_ONLY = new Set(["frame-ancestors", "report-uri", "report-to", "sandbox"]);

/** @param {(name: string) => boolean} keep */
function serialize(keep) {
  return Object.entries(CSP_DIRECTIVES)
    .filter(([name]) => keep(name))
    .map(([name, values]) => `${name} ${values.join(" ")}`)
    .join("; ");
}

/** The full policy, for the Content-Security-Policy header. */
export const CSP_HEADER = serialize(() => true);

/** The same policy minus header-only directives, for `<meta http-equiv>` on hosts without headers. */
export const CSP_META = serialize((name) => !HEADER_ONLY.has(name));

/**
 * Every feature is denied except clipboard-write for this origin (the copy buttons on code blocks).
 * Only features Chromium recognises are listed: an unknown name logs a console error.
 */
export const PERMISSIONS_POLICY = [
  "accelerometer=()",
  "autoplay=()",
  "bluetooth=()",
  "browsing-topics=()",
  "camera=()",
  "clipboard-read=()",
  "clipboard-write=(self)",
  "display-capture=()",
  "encrypted-media=()",
  "fullscreen=()",
  "gamepad=()",
  "geolocation=()",
  "gyroscope=()",
  "hid=()",
  "idle-detection=()",
  "local-fonts=()",
  "magnetometer=()",
  "microphone=()",
  "midi=()",
  "payment=()",
  "picture-in-picture=()",
  "publickey-credentials-get=()",
  "screen-wake-lock=()",
  "serial=()",
  "usb=()",
  "xr-spatial-tracking=()",
].join(", ");

/** @type {Readonly<Record<string, string>>} */
export const SECURITY_HEADERS = Object.freeze({
  "Content-Security-Policy": CSP_HEADER,
  "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy": PERMISSIONS_POLICY,
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "X-Frame-Options": "DENY",
});

/** Netlify / Cloudflare Pages `_headers` format: the same headers on every path. */
export function headersFile() {
  const lines = ["/*"];
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) lines.push(`  ${name}: ${value}`);
  return `${lines.join("\n")}\n`;
}
