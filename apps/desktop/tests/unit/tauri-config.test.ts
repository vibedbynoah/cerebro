/**
 * The desktop shell's security configuration, checked as data: the CSP, the isolation pattern, the
 * capability grants, and that the four places naming the IPC commands agree (build.rs ACL,
 * generate_handler!, capability, isolation allowlist) with what the UI actually invokes.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { describe, expect, it } from "vitest";
import { CSP_DIRECTIVES, PERMISSIONS_POLICY } from "../../security.mjs";

const APP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const read = (rel: string) => readFileSync(path.join(APP, rel), "utf8");

interface TauriConf {
  identifier: string;
  build: { frontendDist: string; devUrl?: string; removeUnusedCommands?: boolean };
  app: {
    withGlobalTauri?: boolean;
    windows: Array<{ label: string; create?: boolean; decorations?: boolean; devtools?: boolean; url?: string }>;
    security: {
      csp: Record<string, string>;
      devCsp?: unknown;
      freezePrototype?: boolean;
      dangerousDisableAssetCspModification?: boolean;
      assetProtocol?: { enable?: boolean };
      pattern: { use: string; options?: { dir: string } };
      capabilities: string[];
      headers: Record<string, string>;
    };
  };
}

const conf = JSON.parse(read("src-tauri/tauri.conf.json")) as TauriConf;
const capability = JSON.parse(read("src-tauri/capabilities/main-window.json")) as {
  identifier: string;
  windows: string[];
  remote?: unknown;
  permissions: string[];
};
const cargo = read("src-tauri/Cargo.toml");
const buildRs = read("src-tauri/build.rs");
const libRs = read("src-tauri/src/lib.rs");
const isolationJs = read("isolation/index.js");
const tauriTs = read("ui/src/platform/tauri.ts");
const desktopTs = read("ui/src/service/desktop-service.ts");

/** build.rs is the single list; every other declaration must match it. */
const APP_COMMANDS = [...(/const COMMANDS[^;]+;/.exec(buildRs)?.[0] ?? "").matchAll(/"([a-z_]+)"/g)].map((m) => m[1] as string);
const CORE_COMMANDS = ["plugin:window|start_dragging", "plugin:window|internal_toggle_maximize"];

describe("tauri.conf.json", () => {
  const csp = conf.app.security.csp;

  it("has a strict CSP with no unsafe sources and no remote script, style or frame", () => {
    const all = Object.values(csp).join(" ");
    expect(all).not.toMatch(/unsafe-inline|unsafe-eval|unsafe-hashes|wasm-unsafe-eval|\*(?!\.localhost)|https?:\/\/(?!ipc\.localhost|127\.0\.0\.1:\d+|\*\.localhost)/);
    expect(csp["default-src"]).toBe("'self'");
    expect(csp["script-src"]).toBe("'self'");
    expect(csp["object-src"]).toBe("'none'");
    expect(csp["base-uri"]).toBe("'none'");
    expect(csp["frame-ancestors"]).toBe("'none'");
    expect(csp["form-action"]).toBe("'none'");
    // Only Tauri's own isolation-iframe style is hashed (its textContent is a fixed constant).
    expect(csp["style-src"]).toMatch(/^'self'( 'sha256-[A-Za-z0-9+/=]{44}')?$/);
    // IPC plus the loopback API, nothing else.
    expect(csp["connect-src"]?.split(" ")).toEqual(["ipc:", "http://ipc.localhost", "http://127.0.0.1:8181"]);
    expect(conf.app.security.devCsp).toBeUndefined();
  });

  it("frames nothing but Tauri's isolation iframe, as narrowly as CSP can say it", () => {
    // tauri-codegen names the isolation scheme `isolation-<random uuid>` on every build, so on
    // Windows the frame's origin is http://isolation-<uuid>.localhost. CSP host wildcards only
    // cover a whole leading label, so `http://*.localhost` is the narrowest source that matches it
    // (a fixed `http://isolation.localhost` would silently break IPC). Other *.localhost origins are
    // the app's own custom protocols, and every app page sends `frame-ancestors 'none'`.
    expect(csp["frame-src"]).toBe("http://*.localhost");
  });

  it("shares every directive with the web policy except the ones the shell must widen", () => {
    const widened = new Set(["connect-src", "style-src", "frame-src"]);
    for (const [name, values] of Object.entries(CSP_DIRECTIVES)) {
      if (widened.has(name)) continue;
      expect(csp[name], name).toBe(values.join(" "));
    }
  });

  it("turns on the isolation pattern, prototype freezing, and nothing that loosens the CSP", () => {
    expect(conf.app.security.pattern).toEqual({ use: "isolation", options: { dir: "../isolation" } });
    expect(conf.app.security.freezePrototype).toBe(true);
    expect(conf.app.security.dangerousDisableAssetCspModification).toBe(false);
    expect(conf.app.security.assetProtocol?.enable).toBe(false);
    expect(conf.app.withGlobalTauri).toBe(false);
    expect(conf.build.removeUnusedCommands).toBe(true);
    expect(cargo).toMatch(/features = \[[^\]]*"isolation"/);
  });

  it("loads only bundled assets: no dev server URL, no remote window URL", () => {
    expect(conf.build.devUrl).toBeUndefined();
    expect(conf.build.frontendDist).toBe("../ui/dist-desktop");
    for (const w of conf.app.windows) expect(w.url).toBeUndefined();
  });

  it("never compiles devtools into release builds", () => {
    expect(cargo).not.toMatch(/"devtools"/);
    for (const w of conf.app.windows) expect(w.devtools).not.toBe(true);
  });

  it("builds the main window in code, where the navigation and new-window guards attach", () => {
    expect(conf.app.windows).toHaveLength(1);
    expect(conf.app.windows[0]).toMatchObject({ label: "main", create: false, decorations: false });
    expect(libRs).toMatch(/\.on_navigation\(navigation::is_app_url\)/);
    expect(libRs).toMatch(/\.on_new_window\(\|_url, _features\| NewWindowResponse::Deny\)/);
  });

  it("sends the hardening headers the platform supports", () => {
    const h = conf.app.security.headers;
    expect(h["Cross-Origin-Opener-Policy"]).toBe("same-origin");
    expect(h["Cross-Origin-Resource-Policy"]).toBe("same-origin");
    expect(h["X-Content-Type-Options"]).toBe("nosniff");
    expect(h["Permissions-Policy"]).toBe(PERMISSIONS_POLICY.replace("browsing-topics=(), ", ""));
  });
});

describe("IPC surface", () => {
  it("grants the main window exactly the commands it uses, and no plugin it does not", () => {
    expect(conf.app.security.capabilities).toEqual([capability.identifier]);
    expect(capability.windows).toEqual(["main"]);
    expect(capability.remote).toBeUndefined();
    expect(capability.permissions.sort()).toEqual(
      [...APP_COMMANDS.map((c) => `allow-${c.replaceAll("_", "-")}`), "core:window:allow-start-dragging", "core:window:allow-internal-toggle-maximize"].sort(),
    );
    expect(capability.permissions.join(" ")).not.toMatch(/default|shell|fs:|http:|opener|process/);
    expect(cargo).not.toMatch(/tauri-plugin-(shell|fs|http|opener|process)/);
  });

  it("declares the same commands in build.rs, generate_handler!, the isolation hook and the UI", () => {
    const quoted = (src: string) => [...src.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]);
    const acl = quoted(/const COMMANDS[^;]+;/.exec(buildRs)?.[0] ?? "");
    const handler = [...(/generate_handler!\[([^\]]+)\]/.exec(libRs)?.[1] ?? "").matchAll(/(?:commands|mcp)::([a-z_]+)/g)].map((m) => m[1]);
    const noArgs = [...(/new Set\(\[([\s\S]*?)\]\)/.exec(isolationJs)?.[1] ?? "").matchAll(/"([^"]+)"/g)].map((m) => m[1]);
    const withArgs = [...(/const WITH_ARGS = \{([\s\S]*?)\n\};/.exec(isolationJs)?.[1] ?? "").matchAll(/^ {2}([a-z_]+): \(p\) =>/gm)].map((m) => m[1]);
    const invoked = new Set([...(tauriTs + desktopTs).matchAll(/invoke(?:<[^>]+>)?\("([a-z_]+)"/g)].map((m) => m[1]));
    expect(APP_COMMANDS.length).toBeGreaterThan(5);
    expect(acl).toEqual(APP_COMMANDS);
    expect([...handler].sort()).toEqual([...APP_COMMANDS].sort());
    expect([...noArgs, ...withArgs].sort()).toEqual([...APP_COMMANDS, ...CORE_COMMANDS].sort());
    expect([...invoked].sort()).toEqual([...APP_COMMANDS].sort());
  });
});

describe("isolation hook", () => {
  const win: Record<string, unknown> = {};
  vm.runInNewContext(isolationJs, { window: win, Object, Reflect, Set, Error });
  const hook = win["__TAURI_ISOLATION_HOOK__"] as (m: unknown) => unknown;

  it("passes allowlisted commands with no arguments through unchanged", () => {
    const noArgs = [...(/new Set\(\[([\s\S]*?)\]\)/.exec(isolationJs)?.[1] ?? "").matchAll(/"([^"]+)"/g)].map((m) => m[1] as string);
    expect(noArgs).toEqual(expect.arrayContaining(CORE_COMMANDS));
    for (const cmd of noArgs) {
      const message = { cmd, callback: 1, error: 2, payload: {} };
      expect(hook(message)).toBe(message);
    }
    expect(hook({ cmd: "app_info", callback: 1, error: 2 })).toBeTruthy();
  });

  it("checks the shape of engine-switch and MCP arguments", () => {
    const ok = (cmd: string, payload: unknown) => expect(() => hook({ cmd, callback: 1, error: 2, payload })).not.toThrow();
    const no = (cmd: string, payload: unknown) => expect(() => hook({ cmd, callback: 1, error: 2, payload })).toThrow();
    ok("switch_set", { id: "gpu_mma_matmul", value: 0 });
    no("switch_set", { id: "gpu_mma_matmul", value: 0.5 });
    no("switch_set", { id: "../x", value: 1 });
    ok("mcp_restart", { name: "files" });
    no("mcp_restart", { name: "a;b" });
    ok("mcp_call", { server: "files", tool: "read_file", arguments: { path: "a.txt" } });
    no("mcp_call", { server: "files", tool: "", arguments: {} });
    const server = { command: "npx", args: ["-y", "pkg"], env: { K: "V" }, disabled: false };
    ok("mcp_config_save", { config: { mcpServers: { files: server } } });
    no("mcp_config_save", { config: { mcpServers: { files: { ...server, extra: 1 } } } });
    no("mcp_config_save", { config: { mcpServers: { files: { ...server, args: [1] } } } });
    // A web address server: url and headers replace command, args and env.
    const web = { url: "https://example.com/mcp", headers: { Authorization: "Bearer t" }, disabled: false };
    ok("mcp_config_save", { config: { mcpServers: { web } } });
    ok("mcp_config_save", { config: { mcpServers: { web: { url: "http://127.0.0.1:9000/mcp" } } } });
    no("mcp_config_save", { config: { mcpServers: { web: { url: "ftp://example.com" } } } });
    no("mcp_config_save", { config: { mcpServers: { web: { url: "https://example.com/mcp", headers: { A: 1 } } } } });
    no("mcp_config_save", { config: { mcpServers: { web: { ...web, command: "npx" } } } });
    no("mcp_config_save", { config: { mcpServers: { web: { disabled: false } } } });
  });

  it("refuses every other command, and any argument smuggled into an allowed one", () => {
    for (const message of [
      { cmd: "plugin:shell|execute", payload: {} },
      { cmd: "plugin:fs|read_file", payload: {} },
      { cmd: "plugin:window|set_position", payload: {} },
      { cmd: "window_close ", payload: {} },
      { cmd: "WINDOW_CLOSE", payload: {} },
      { cmd: ["window_close"], payload: {} },
      { cmd: "window_close", payload: { label: "other" } },
      { cmd: "window_close", payload: "x" },
      { cmd: "window_close", payload: null },
      { cmd: "window_close", payload: [] },
      { cmd: "window_close", payload: Object.create({ inherited: true }) },
      null,
      "window_close",
    ]) {
      expect(() => hook(message), JSON.stringify(message)).toThrow(/refused/);
    }
  });

  it("cannot be replaced by script running in the isolation frame", () => {
    expect(() => {
      "use strict";
      (win as { __TAURI_ISOLATION_HOOK__: unknown }).__TAURI_ISOLATION_HOOK__ = () => true;
    }).toThrow(TypeError);
    expect(win["__TAURI_ISOLATION_HOOK__"]).toBe(hook);
  });
});
