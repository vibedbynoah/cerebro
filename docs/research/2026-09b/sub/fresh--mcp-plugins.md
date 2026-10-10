# MCP and plugin ecosystems — late Sept 2026 (Cerebro research)

Date: 2026-09-22. Cutoff focus: what changed since 2026-09-01; spec baseline is 2026-07-28.
All URLs below were opened via search/fetch. Unverified = marked.

## 1. Spec: 2026-07-28 is current stable (stateless core)

- The 2026-07-28 spec is the stable release; RC locked 2026-05-21, final 2026-07-28
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Headline: stateless protocol core. `initialize`/`notifications/initialized` handshake
  and `Mcp-Session-Id` removed (SEP-2575, SEP-2567). Every request carries protocol
  version + client capabilities in `_meta`; optional `server/discover` RPC replaces
  handshake; any request can hit any instance behind round-robin LB
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/)
  (https://modelcontextprotocol.io/specification/2026-07-28/changelog).
- Streamable HTTP must include `Mcp-Method` / `Mcp-Name` headers for routing without
  body parsing (SEP-2243) (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- `tools/list`, `prompts/list`, `resources/list`, `resources/read` carry `ttlMs` +
  `cacheScope` (SEP-2549) (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- MRTR (multi-round-trip requests, SEP-2322): server returns
  `resultType: "input_required"` + needed requests; client retries with
  `inputResponses`. Replaces held-open server-initiated `elicitation/create`,
  `sampling/createMessage`, `roots/list` (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Tasks moved to `io.modelcontextprotocol/tasks` extension with poll `tasks/get` +
  new `tasks/update`; notifications via opt-in `subscriptions/listen`
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Deprecated (12-month minimum window): Roots, Sampling, Logging (SEP-2577);
  legacy HTTP+SSE transport; DCR (see auth)
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Extensions framework formalized (SEP-2133 lineage); MCP Apps + Tasks + Enterprise
  Managed Authorization are extensions, negotiated via capabilities
  (https://blog.modelcontextprotocol.io/posts/2026-07-28/)
  (https://modelcontextprotocol.io/extensions/apps/overview).

## 2. SDKs: TypeScript v2, Rust rmcp 3.x

- Tier 1 (2026-07-28 on day one): TypeScript, Python, Go, C#. Rust = Tier 2, beta→stable
  support after (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- TypeScript v2 (stable 2026-07-28): monolith `@modelcontextprotocol/sdk` split into
  `@modelcontextprotocol/server` 2.0.0 + `@modelcontextprotocol/client` 2.0.0;
  Standard Schema input/output (Zod v4, ArkType, Valibot); `serveStdio()` /
  `createMcpHandler()` web-standard Request/Response (Node/Bun/Deno/Workers);
  thin Express/Hono/Fastify/Node adapters; v1.x security fixes ≥6 months
  (https://github.com/modelcontextprotocol/typescript-sdk)
  (https://www.npmjs.com/package/@modelcontextprotocol/server).
- Rust `rmcp`: crate `rmcp` 3.4.0 published 2026-09-15 (after cutoff window; 3.0.0 cut
  2026-07-28); implements stable 2026-07-28 + back-compat 2025-11-25; features:
  discover/legacy/auto lifecycle modes, MRTR, tasks, caching, routing headers,
  stateless Streamable HTTP, schemars JSON Schema 2020-12, tokio
  (https://github.com/modelcontextprotocol/rust-sdk) (https://crates.io/crates/rmcp).
- Cerebro relevance: Rust harness should depend on `rmcp` 3.x with `server` feature
  for in-process servers and `TokioChildProcess` client for spawned stdio servers.

## 3. Hosts: how desktop apps install/run servers

- Claude Desktop: `claude_desktop_config.json` (legacy manual) + one-click `.mcpb`
  desktop extensions (Settings → Extensions → Browse/Install); built-in Node;
  secrets in OS keychain; curated directory auto-updates; Team/Enterprise can
  push/block/private-list (https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)
  (https://www.anthropic.com/engineering/desktop-extensions).
- Claude Code CLI: `claude mcp add --transport http|sse|stdio`, `claude mcp add-json`,
  `claude mcp add-from-claude-desktop`; scopes local/project (`.mcp.json`)/user;
  plugins bundle servers via `.mcp.json` or inline `plugin.json`
  (https://code.claude.com/docs/en/mcp).
- Cursor: Settings → MCP panel + `~/.cursor/mcp.json` (global) or `.cursor/mcp.json`
  (project), `mcpServers` key; stdio + Streamable HTTP (≥v0.48); per-server restart;
  one-click `cursor.com/en/install-mcp?name=..&config=..` deeplinks
  (https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-cursor.md).
- VS Code (≥1.99): Extensions view `@mcp` gallery; `.vscode/mcp.json` with
  `"servers"` key (not `mcpServers`); user/profile/remote/devcontainer variants;
  `code --add-mcp`; auto-discovery from Claude Desktop/Cursor/Windsurf/Copilot CLI;
  resources/prompts/MCP Apps surfaced in Chat
  (https://code.visualstudio.com/docs/agent-customization/mcp-servers).
- LM Studio (≥0.3.17, host): Program tab → Install → Edit `mcp.json`
  (`~/.lmstudio/mcp.json`, `%USERPROFILE%/.lmstudio/mcp.json` on Windows); follows
  Cursor `mcpServers` notation; stdio spawned per server + remote URL/headers;
  "Add to LM Studio" deeplink; API exposes ephemeral vs `mcp.json` integrations
  with `allowed_tools`; caution: local models + Claude-oriented servers can flood
  context (https://lmstudio.ai/docs/app/mcp).
- DeepSeek Harness (DSH = "DSH Desktop" in task): `@deepseek-ai/dsh-mcp-client`,
  one plugin instance = one MCP server in `cordis.yml`/patch; stdio +
  `streamable-http`; tools namespaced `mcp__<server>__<tool>` through unified
  gating/timeout pipeline; only Tools bridged (no Resources/Prompts); dsh-desktop
  Electron/Tauri wrappers add plugin/MCP/skills markets + paste-and-convert MCP JSON
  (https://deepseekdocs.com/en/docs/features/mcp).

## 4. Packaging + registries: MCPB, server.json, one-click

- MCPB (was `.dxt`, renamed late 2025; both install): zip + `manifest.json`
  (`mcpb_version`, `name`, `server.type` node/python/binary, `mcp_config`,
  `user_config` with `${__dirname}`/`${user_config.x}` templating, secrets → OS
  vault); CLI `npm i -g @anthropic-ai/mcpb; mcpb init; mcpb pack`; Node recommended
  (ships in Claude Desktop); open spec + toolchain
  (https://github.com/modelcontextprotocol/mcpb)
  (https://www.anthropic.com/engineering/desktop-extensions).
- Official MCP Registry (preview 2025-09-08): `registry.modelcontextprotocol.io`,
  open API + `server.json` + `mcp-publisher` CLI; packages point at npm/PyPI/NuGet/
  Docker/MCPB/GitHub releases; public + private sub-registries; community
  denylist moderation
  (https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/).
- GitHub MCP Registry one-click: server page → Install in VS Code, pre-filled config
  (https://github.blog/ai-and-ml/generative-ai/how-to-find-install-and-manage-mcp-servers-with-the-github-mcp-registry/).
- Smithery (`smithery.ai/server/...`) and MCPFind-style catalogs add scanned metadata
  + per-client install snippets; route through hosted gateway vs local stdio tradeoff
  (https://github.com/sena-labs/replicate-mcp-server).

## 5. Permissions/sandboxing (host controls Cerebro should copy)

- Spec line: human-in-the-loop SHOULD hold deny on tool invocation; hosts SHOULD show
  exposed tools + indicators + confirmations (cited in
  https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks context;
  Simon Willison summary unverified-fetch, treat as secondary).
- VS Code: workspace trust + per-server trust dialog on first start/config change;
  `MCP: Reset Trust`; autostart policy (`never`/`onlyNew`/`newAndOutdated`);
  per-tool toggles; `sandboxEnabled` + `sandbox.filesystem.allowWrite` /
  `sandbox.network.allowedDomains` on macOS/Linux (not Windows); enterprise GitHub
  policy gating (https://code.visualstudio.com/docs/agent-customization/mcp-servers).
- Claude Code: project `.mcp.json` approval prompt; workspace-trust interaction;
  `disabledMcpServers`/`disabledMcpjsonServers`; per-call timeouts + idle timeouts;
  OAuth token purge on remove (https://code.claude.com/docs/en/mcp).
- LM Studio: no confirm before first spawn — 2026-04-17 bug report shows enabled
  stdio entry spawns on enable + every launch, handshake after; recommends
  first-enable dialog, re-prompt on command hash change, kill process group
  (unverified-fix; issue text from search, treat as report not spec).
- Cerebro: hash-pin `command+args`, first-run consent showing argv/cwd, re-consent
  on change, allowlist, per-tool approve, kill process group on failure, log all
  tool I/O to SIEM-style local log.

## 6. Auth for remote servers: OAuth 2.1 + CIMD (2026-07-28 hardening)

- Base: OAuth 2.1, RFC 8414/9728 discovery, RFC 8707 `resource` indicators, PKCE S256
  mandatory; server validates aud/iss/signature per request; tokens never in query.
- 2026-07-28 deltas: `iss` per RFC 9207 validated (SEP-2468, SHOULD→future MUST);
  credentials bound per issuer, re-register on AS change (SEP-2352);
  `application_type` native/web in DCR fixes localhost rejects (SEP-837); refresh
  guidance (SEP-2207); step-up scope accumulation is client duty (SEP-2350);
  standard `oauth-authorization-server` suffix (SEP-2351); DCR formally deprecated
  for CIMD (self-hosted HTTPS `client_id` URL doc; fetch+validate, no registry-write
  endpoint) (https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- Cerebro (fully offline default): remote OAuth path is opt-in only; local stdio
  servers need no OAuth; if remote enabled, implement CIMD-first + DCR fallback,
  `resource`-bound tokens, per-issuer credential store.

## 7. MCP Apps (extension `io.modelcontextprotocol/ui`, stable 2026-01-26)

- Tools link `_meta.ui.resourceUri` → `ui://` HTML resource
  (`text/html;profile=mcp-app`); host prefetches via `resources/read`, renders in
  sandboxed iframe, postMessage JSON-RPC (`ui/initialize`, `tools/call` proxy,
  `ui/update-model-context`); CSP/permission declarations; graceful text fallback
  if host lacks support (https://modelcontextprotocol.io/extensions/apps/overview).
- Supported by Claude (web+desktop), VS Code Copilot, M365 Copilot, Goose, Postman,
  MCPJam, Archestra (https://modelcontextprotocol.io/extensions/apps/overview).
- Cerebro: render-deserves-care on Windows-first offline host — iframe sandbox +
  allowlist CSP; Apps increase prompt-injection surface (rendered HTML from server).

## 8. Security: tool poisoning, rug pulls, shadowing, output injection

- Tool poisoning (Invariant Labs, Apr 2025): hidden directives in tool
  descriptions/params invisible in simplified UI exfiltrate `~/.cursor/mcp.json`,
  `~/.ssh/id_rsa`, hijack other servers' tools (e.g. reroute `send_email`);
  demonstrated vs Cursor; mitigations: show full descriptions, pin versions/hashes,
  cross-server boundaries (https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks).
- OWASP framing: connect-time vs runtime trust gap; tool outputs flow into context
  unvalidated; mitigations: structured-output schemas, isolate privileged tools,
  server-side enforcement (not prompt-only), allowlist servers, out-of-band confirm
  for sensitive ops (https://github.com/OWASP/www-community/blob/master/pages/attacks/MCP_Tool_Poisoning.md).
- Rug pulls: approved tool silently redefined server-side; Microsoft Zero-Trust
  catalog maps to LLM01/LLM03/LLM06/LLM07/LLM08 + MITRE AML.T0010/AML.T0051.001/
  AML.T0053/AML.T0110; controls: RBAC identity, supply-chain vetting/signing,
  gateway filters (Prompt Shields), DLP labels, sandboxing, SIEM logging, block+escalate
  (https://learn.microsoft.com/en-us/security/zero-trust/catalog-ai-attack-techniques/rug-pull-attack).
- Scale numbers (secondary, via CSA 2026-07-01/02 notes): MCPTox benchmark 45 live
  servers × 20 LLMs avg 36.5% poisoning success, max 72.8% (o1-mini); MCPoison
  CVE-2025-54136 in Cursor (approve-by-name, config swap = team RCE, fixed 1.3.9);
  TrustFall May 2026: 4 coding CLIs auto-executed project MCP on folder-trust.
  Treat numbers as reported-benchmark, not laptop batch-1 transfer.
- Research mitigation ETDI (arXiv 2506.01333, Jun 2025): signed versioned tool
  definitions (+ API-contract hash), OAuth scopes in signed def, PDP policy check
  (Cedar/OPA), call-stack verification; prototype fork only
  (https://arxiv.org/html/2506.01333v1).

## 9. Most useful servers today (+ image gen)

- Reference set (`modelcontextprotocol/servers`): filesystem (scoped dirs + Roots),
  fetch (URL→markdown), memory (knowledge-graph JSONL), git, sequential-thinking,
  time, everything (test); community: Brave Search, Postgres/SQLite, Playwright
  (browser), GitHub official `ghcr.io/github/github-mcp-server` (remote
  `https://api.githubcopilot.com/mcp/` or Docker; old npm `server-github`
  deprecated Apr 2025) (https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-cursor.md).
- Image gen (all networked, API-key, NOT offline — Cerebro conflict to disclose):
  `mcpimg` (OpenRouter/Together/Replicate/fal, stdio `npx -y mcpimg`); fal single-purpose
  `mcp-fal-ai-image` (`generate-image`, saves `~/Downloads/fal_ai`); Replicate family:
  official thin REST bridge vs `sena-labs/replicate-mcp-server` curated 36-task-tool
  studio (waits + auto-downloads to `~/Downloads/replicate-mcp/`, cost estimator,
  batch/DAG, `.mcpb` + Smithery + HTTP/SSE)
  (https://github.com/sena-labs/replicate-mcp-server).
- Local-model warning (LM Studio docs): Claude/ChatGPT-oriented servers can bloat
  context and overflow small local windows — gate with `allowed_tools`/scoped servers
  (https://lmstudio.ai/docs/app/mcp).

## 10. Cerebro integration sketch (Rust + argv-only llama.cpp)

- Ship `rmcp` client (TokioChildProcess, argv arrays, no shell) + optional `rmcp`
  server for Cerebro's own tools; support stdio first, Streamable HTTP opt-in.
- Config: `mcp.json`-compatible (`mcpServers`), VS Code `servers` alias accepted;
  per-server `command/args/env/cwd`, `allowed_tools`, timeouts, hash-pinned approvals.
- Distribute curated local servers as signed `.mcpb`-style zips; verify manifest hash
  before spawn; OS-vault secrets; enterprise allowlist/blocklist.
- Default-deny network: image-gen and remote servers behind explicit opt-in +
  red "network" badge; document per-call $/latency since offline is the promise.
