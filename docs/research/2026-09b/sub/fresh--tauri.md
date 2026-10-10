# Tauri v2 + WebView2 state — research notes (2026-09-22)

Focus: what changed since 2026-09-01, with emphasis on 2025–2026 primary sources.
Reference machine: Win11 Pro laptop, i9-12900H, 32 GB RAM, RTX 3070 Ti 8 GB, no NPU.
Cerebro is offline-first; only an opt-in update check may touch the network.

## 1. Latest releases (verified Sept 2026)

- Tauri core latest is **v2.11.5** (2026-07-01; v2.11.4 2026-06-28/30, v2.11.3 2026-06-17, v2.11.2 2026-05-16, v2.11.0 2026-04-30). v2.11.5 was a dependency unpin (`time` 0.3.53); v2.11.4 pinned `time <0.3.52` and fixed a Windows HDC-handle leak in DPI queries (also improves undecorated-window resize speed).
  Sources: https://v2.tauri.app/release/tauri/ , https://github.com/tauri-apps/tauri/releases
- **wry 0.56.0** (2026-07-30; 0.56.1 fixes a Windows WebView2 parent-subclass re-entrancy crash); tao 0.35.x/0.36. Bundler 2.9.4 (2026-06-28). `@tauri-apps/api` 2.11.x.
  Sources: https://v2.tauri.app/release/wry/v0.56.0/ , https://github.com/tauri-apps/wry/releases , https://v2.tauri.app/release/tauri-bundler/
- Tauri 2.11.0 features (2026-04-30): `command(rename)`, `eval_with_callback`, `data-tauri-drag-region="deep"`, WebView general-autofill toggle (Windows/WebView2 only), Android/iOS multi-window + file association, macOS content-process-termination handler, `dbus` feature flag for Linux theme detection.
  Source: https://v2.tauri.app/release/tauri/v2.11.0/
- wry 0.56.0: expanded cross-backend **permission-handler API** (12 WebView2 permission kinds, Allow/Deny/Default+Prompt), Windows **named WebView2 profiles** (`with_profile_name`), removed macOS/iOS `transparent`/`fullscreen` feature flags (always on) and the `protocol` feature flag (custom protocols always available).
  Sources: https://github.com/tauri-apps/wry/releases/tag/wry-v0.56.0 , https://github.com/tauri-apps/wry/pull/1654 , https://github.com/tauri-apps/wry/pull/1738

## 2. Security: capabilities, isolation, CSP

- **Capabilities** (`src-tauri/capabilities/*.json`, referenced from `tauri.conf.json`) are the allow/deny boundary: per-window/webview glob (`*`, `admin-*`), per-platform, plus `remote.urls` (URLPattern) for remote origins. No match = no IPC at all. Since 2026-04 (commit 7a2672f), remote origins ALWAYS go through ACL even without an AppManifest (breaking change for apps that let remote pages call custom commands).
  Sources: https://v2.tauri.app/security/capabilities/ , https://github.com/tauri-apps/tauri/commit/7a2672f7cf11652eab794781c63a5bb0124deb68
- **Permissions/scopes**: `plugin:allow-command` + `scope.allow/deny` (deny wins). E.g. `shell:allow-execute` with `{name, cmd, args, sidecar}`; fs scopes use `$HOME`-style globs. `assetProtocol.scope` (`allow`/`deny`/`requireLiteralLeadingDot`) gates `asset:`/`convertFileSrc` loads; dot-directories (`.cache`) need literal segments or `requireLiteralLeadingDot:false`.
  Sources: https://v2.tauri.app/security/permissions/ , https://v2.tauri.app/security/asset-protocol/
- **Isolation pattern** (opt-in `app.security.pattern.use="isolation"`): sandboxed iframe intercepts all IPC, AES-GCM encrypts via SubtleCrypto with per-run keys. Recommended but costs per-message overhead; ES modules don't load in the isolation app on Windows (build-time script inlining only).
  Source: https://v2.tauri.app/concept/inter-process-communication/isolation/
- **CSP**: opt-in; Tauri appends nonces/hashes for bundled code at compile time. `dangerousDisableAssetCspModification` disables that hardening. `freezePrototype:false` is the default (prototype-pollution → IPC-hijack vector); set `true`. `withGlobalTauri:false` default shrinks exposure.
  Sources: https://v2.tauri.app/security/csp/ , https://v2.tauri.app/reference/config/ , https://snyk.io/articles/5-common-security-misconfigurations/
- Threat model is explicit: capabilities mitigate frontend compromise, NOT malicious Rust, lax scopes, WebView 0-days, or supply-chain attacks; Tauri deliberately uses the OS WebView (evergreen patches) instead of bundling Chromium.
  Source: https://v2.tauri.app/security/

## 3. Updater + signing (relevant to Cerebro's opt-in update check)

- `tauri-plugin-updater` 2.x: mandatory **minisign Ed25519** signature (`signer generate`, `TAURI_SIGNING_PRIVATE_KEY`, pubkey embedded in binary). TLS enforced; `{{target}}/{{arch}}/{{current_version}}` endpoint variables; static JSON or 204/200 dynamic server; per-platform `OS-ARCH` keys; NSIS `.nsis.zip` must use **Store (method 0)** and BOM-free JSON or install fails.
  Sources: https://v2.tauri.app/plugin/updater/ , https://github.com/tauri-apps/plugins-workspace/blob/v2/plugins/updater/src/updater.rs , https://github.com/tauri-apps/tauri/issues/15601
- `bundle.createUpdaterArtifacts: true` (new apps) / `"v1Compatible"` (v1 migrants; removed in v3). Windows `installMode`: passive (default) / basicUI / quiet.
  Source: https://v2.tauri.app/plugin/updater/
- Fit for Cerebro: updater is the ONLY sanctioned network path — keep it off by default, gate behind explicit opt-in capability (`updater:default` split into check/download/install), ship a static JSON on a trusted HTTPS endpoint. Key loss = no future updates for installed users.

## 4. Sidecars (llama-server / whisper.cpp / SD-cpp pattern)

- `bundle.externalBin: ["binaries/my-sidecar"]` + per-triple binaries (`my-sidecar-x86_64-pc-windows-msvc.exe`); spawn via `app.shell().sidecar("my-sidecar")` (Rust) or `Command.sidecar('binaries/my-sidecar')` (JS); `execute()` vs `spawn()` need `shell:allow-execute` vs `shell:allow-spawn`; args allowlisted exactly or by regex validator (`"args": true` = footgun).
  Sources: https://v2.tauri.app/develop/sidecar/ , https://github.com/tauri-apps/plugins-workspace/tree/v2/plugins/shell
- Shell plugin kills spawned children on `RunEvent::Exit`. `shell.open` moved to opener plugin; scope validation RCE (CVE-2025-31477, fixed in shell 2.2.1).
  Sources: https://docs.rs/crate/tauri-plugin-shell/latest/source/src/lib.rs , https://github.com/advisories/GHSA-C9PR-Q8GX-3MGP
- Fit: llama-server is an ideal sidecar (single persistent process, HTTP on localhost, argv-only). Prefer spawning from **Rust** (no JS capability needed for the binary itself), stream SSE/stdout via Channel/events. One long-lived server ≫ per-inference respawn (process spawn ~50–200 ms + model reload seconds).

## 5. IPC streaming performance (token streaming matters most)

- `invoke` is JSON-based; raw `InvokeResponseBody` avoids one `serde_json::Value` round-trip (PR #10641: 150 MB response 29 s → 23 s custom-protocol, 54 s → 48 s postMessage, macOS).
  Source: https://www.github.com/tauri-apps/tauri/commit/d0510f52eb7efeabe00df5030cf10be16f99e178
- **Channel**: payloads < **8192 B JSON / 1024 B raw** go via direct `eval` (2× faster than fetch on WebView2 v135); larger payloads are stored server-side and pulled via `fetch` (`FETCH_CHANNEL_DATA_COMMAND`). `emit_raw`/`emit_raw_to` + `payload_raw` avoid JSON for events (PR #14269).
  Sources: https://docs.rs/tauri/latest/src/tauri/ipc/channel.rs.html , https://github.com/tauri-apps/tauri/pull/14269
- Third-party `tauri-conduit` benches (Rust-side only, NOT end-to-end): 25 B struct 722 ns → 330 ns (JSON-direct) → 78 ns (binary); 64 KB blob 2.27 ms → 0.83 ms → 0.20 ms; end-to-end small payloads are bridge-dominated (~300 µs, 1.3×), 64 KB 6.7 ms → 3.2 ms (JSON) / 0.6 ms (binary). Datacenter-style codec numbers do NOT transfer to batch-1 token streaming; per-token payloads (~10–50 B) are bridge-dominated.
  Source: https://github.com/userFRM/tauri-conduit/blob/master/BENCHMARKS.md
- `tauri-ipc-bench` (image streaming): returning `Vec<u8>`/`Response` from `invoke` arrives as a **JSON number array** (worse than base64); custom `bench-img://` protocol serving bytes fetched natively is fastest; never push big blobs via eval — signal via eval, pull via fetch.
  Sources: https://github.com/iceinvein/tauri-ipc-bench , https://github.com/tauri-apps/tauri/issues/13758
- Fit: token deltas are tiny → plain `Channel` is optimal (stays on the fast eval path); batch deltas if >8 KB JSON. Images/audio → custom protocol + `asset:` scope, not invoke.

## 6. WebView2 on Windows 11

- Evergreen runtime **preinstalled on Win11** (and pushed to Win10); Tauri installers still handle missing runtime: `downloadBootstrapper` (default, +0 MB, needs net) / `embedBootstrapper` (+1.8 MB) / `offlineInstaller` (+127 MB) / `fixedVersion` (+180 MB, you own patching — worse security) / `skip` (broken if absent).
  Sources: https://v2.tauri.app/distribute/windows-installer/ , https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution , https://v2.tauri.app/reference/webview-versions/
- `bundle.windows.minimumWebview2Version` (now top-level `windows`, NSIS-only key deprecated in CLI 2.11.0) forces bootstrapper when runtime is too old.
  Source: https://github.com/tauri-apps/tauri/releases/tag/@tauri-apps%2Fcli-v2.11.0
- wry 0.56 named profiles isolate cookies/IndexedDB/cache per webview on a shared runtime — useful for separating chat sessions or untrusted RAG previews.
  Source: https://github.com/tauri-apps/wry/pull/1738
- Pitfall: background windows + frequent eval pushes leak unmanaged WebView2 IPC memory → OOM crash (~2 GB WebView limit); prefer frontend-pull. No built-in OOM-reload event.
  Source: https://github.com/tauri-apps/tauri/issues/13758

## 7. SolidJS + installers + pitfalls

- `create-tauri-app` ships an official **SolidJS** template (TS flavor); community: `riipandi/tauri-start-solid` (Tauri v2 + plugins + TanStack Router + Tailwind, pushed 2026-03), `kvnxiao` TanStack-Start template (SPA default; SSR-prerender only for SEO), `tauri-controls-v2` Solid titlebar/controls.
  Sources: https://v2.tauri.app/start/create-project/ , https://github.com/riipandi/tauri-start-solid
- Solid's fine-grained reactivity (no VDOM diff) suits 60 fps token streaming; keep Tauri JS bridge (`@tauri-apps/api`, plugin packages) pinned to matching 2.11.x minor.
- Installers: **MSI = WiX v3, Windows-only builds, needs VBSCRIPT optional feature**; NSIS `-setup.exe` cross-compilable (cargo-xwin, last resort) with custom templates, hooks (`NSIS_HOOK_*`), per-user/per-machine/both modes, i18n. Sign via `windows.certificateThumbprint`/`signCommand`; since 2025 bundler signs WiX DLLs + NSIS plugin DLLs (7 signing calls — painful with cloud HSM; post-build signing of just the installer + regenerating `.nsis.zip`/`.sig` is a working workaround).
  Sources: https://v2.tauri.app/distribute/windows-installer/ , https://www.github.com/tauri-apps/tauri/commit/8d994f60fe05ec0f45cbe926506bbe10b0d36e3c , https://github.com/tauri-apps/tauri/issues/15601 , https://github.com/tauri-apps/tauri/pull/15422
- Must-fix CVEs/configs: **CVE-2026-42184** (origin confusion on Windows/Android `http://<proto>.localhost` mapping; fixed in 2.11.1 — Cerebro MUST use ≥2.11.1); shell-open RCE CVE-2025-31477 (use opener plugin, pin shell ≥2.2.1); iframe IPC CVE-2024-35222 (fixed); 2026-04 remote-ACL enforcement. Default-deny checklist: no `"args": true`, no `windows:["*"]` + remote, no `asset assess **/*`, `freezePrototype:true`.
  Sources: https://github.com/tauri-apps/tauri/security/advisories/GHSA-7gmj-67g7-phm9 , https://github.com/advisories/GHSA-C9PR-Q8GX-3MGP , https://github.com/tauri-apps/tauri/commit/7a2672f7cf11652eab794781c63a5bb0124deb68
