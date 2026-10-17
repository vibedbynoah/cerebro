# Topic H — A Curated MCP Server Catalogue for Cerebro

**A build-ready reference for adding a first-class MCP catalogue to Cerebro (Tauri 2 + SolidJS, Rust inference engine, Windows).**

Scope: 142 catalogued MCP servers in the 15 requested categories (files, git, code, shell, browser/web, search, databases, docs/notes, productivity, design, cloud/devops, media, memory, math/science, OS/Windows), the MCP **2025-06-18** Streamable HTTP transport and its OAuth 2.1 authorization model in detail, how a desktop client should present a catalogue (search, categories, one-click install, key entry, health checks), and a ranked top-25 build list.

Verification legend used throughout: anything I could confirm against primary docs/repos while writing is stated plainly (sources at the end). Anything I could not confirm to that standard is marked **UNVERIFIED** — treat those install commands as "verify before shipping in the one-click installer". MCP ecosystem packages move fast; every install string must be re-checked at release time by Cerebro's catalogue CI (see §9.1).

---

## 1. Executive summary

1. **The catalogue is feasible as a shipped feature.** The ecosystem now has a machine-readable source of truth: the official MCP Registry (`https://registry.modelcontextprotocol.io`, `GET /v0.1/servers`, cursor pagination, `search`, `updated_since`, `version=latest`). Cerebro should mirror it nightly into SQLite and overlay a hand-curated "Windows-ready" layer, because registry entries rarely include the Windows command wrapping (`cmd /c npx …`) that Cerebro needs.
2. **Two transports cover ~all real servers.** stdio (subprocess) for ~95% of local servers, Streamable HTTP (MCP 2025-06-18) for hosted vendor servers (Linear, Atlassian, Sentry, Datadog, Cloudflare, GitHub remote, PagerDuty, …). Cerebro must implement both; the existing stdio client extends naturally, and the HTTP client needs session IDs, SSE parsing, `Last-Event-ID` resumption, and OAuth 2.1 with Protected Resource Metadata (RFC 9728).
3. **Offline is mostly a packaging problem.** ~60 of the 142 entries are fully local (files, git, shell, local DBs, memory, Windows tooling) and work with zero network once installed; Cerebro should pre-warm `npm`/`uv`/Docker caches at install time so "offline" is actually true on a plane. Everything SaaS-bound is `NET` by construction.
4. **Risk typing must be first-class.** Every entry below carries a risk class (read-only → writes files → runs commands → network → account access). Cerebro already has per-tool allow/ask/deny; the catalogue must map risk class → default per-tool policy and hash tool schemas so a silently-changed tool definition re-prompts the user.
5. **Protocol drift is real.** The 2025-06-18 transport (sessions, SSE, `Last-Event-ID`) is being superseded: newer revisions (2025-11-25, and the 2026-07-28 draft) remove sessions and standalone GET streams and add request-embedded flows (MRTR). Cerebro should negotiate `MCP-Protocol-Version`, keep the 2025-06-18 behavior, and stay version-tolerant (see §7).

---

## 2. Method, sources, and definition of "maintained"

- Primary sources consulted while writing (fetched, not remembered): official `modelcontextprotocol/servers` README, the 2025-06-18 transports and authorization spec pages, the draft transports page, GitHub's `github-mcp-server` docs, Atlassian Rovo MCP docs, Notion's local server npm page, Linear's MCP docs, Sentry's MCP repo/docs, Stripe, Supabase, Cloudflare's MCP server docs, Playwright MCP docs, Chrome DevTools MCP docs, Context7 npm page, Brave/Exa/Tavily/Firecrawl docs, Basic Memory docs, AWS MCP installation docs, Azure MCP repo, Windows-MCP PyPI page, Docker MCP Toolkit docs, Kubernetes MCP repo, Grafana MCP docs, Terraform MCP docs, PagerDuty MCP repo/docs, Redis/Brave/MongoDB/Neo4j/Qdrant/Obsidian/Excel/Google Workspace/Slack(zencoder)/Jupyter/Datadog/Figma docs, the official Registry API docs. All URLs in §12.
- **Maintained** means: vendor-owned or a well-known community repo with recent releases at the time of writing. Archived-but-functional servers (the modelcontextprotocol "Archived" list) are included only where they remain the de-facto option, and are flagged.
- Official reference servers still maintained by the steering group: **Everything, Fetch, Filesystem, Git, Memory, Sequential Thinking, Time** (everything else in that repo is archived per its README).
- I did **not** re-run every server; install strings are as-published, and categories of risk are static analysis of what the tool set can do.

---

## 3. Taxonomy used in every table

**Risk class** (composable):

| Tag | Meaning | Cerebro default policy implication |
|---|---|---|
| `RO` | read-only access to data | auto-allow (still logged) |
| `WF` | writes/deletes files on disk | ask by default; scope paths via server args |
| `EX` | executes commands / arbitrary code (local or sandbox) | deny by default, per-session "ask" opt-in |
| `NW` | unrestricted network egress | ask for first use; log destinations where possible |
| `AC` | acts as you via API key / OAuth session (account access) | ask; show the identity that is connected |

**Offline**:

| Tag | Meaning |
|---|---|
| `Y` | fully offline after install (local data/processes only) |
| `Y*` | offline after first-run package/browser/model download; pre-warmable by Cerebro |
| `P` | partial: core works offline, key features need a service |
| `N` | requires a remote service by design (API key/OAuth) |

Transport: `stdio` = subprocess over stdin/stdout (works locally); `http` = remote Streamable HTTP (2025-06-18; older servers may expose legacy HTTP+SSE `/sse`); `local-http` = server binds localhost and speaks Streamable HTTP/SSE (e.g., Figma desktop, Playwright HTTP mode).

**Windows note that applies to every `npx` row:** the official docs instruct wrapping `npx` as `cmd /c npx …` on Windows (Tauri must spawn `cmd.exe` or resolve `npx.cmd`; see §9.4).

---

## 4. The catalogue

> Rows are numbered globally (1–142). Cross-references like "(see #2)" are not new entries. Install strings are the canonical published form; Cerebro stores its own Windows-normalized variant (see §9.1).

### 4.1 Files & local documents (6 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 1 | Filesystem (official) | `npx -y @modelcontextprotocol/server-filesystem C:\allow\dir` | none (dir allowlist as args) | stdio | `WF` | `Y` | Sandboxed read/write/search/move within explicitly allowed directories. |
| 2 | Desktop Commander | `npx -y @wonderwhy-er/desktop-commander@latest` | none | stdio (+remote mode) | `WF EX NW` | `Y` | Terminal + diff-based file editing + long-running process control + PDF→MD in one server. |
| 3 | Excel MCP Server (haris-musa) | `uvx excel-mcp-server stdio --allow-dir C:\workbooks` | none | stdio / http | `WF` | `Y` | Create/read/edit `.xlsx` (formulas, tables, charts, PivotTables) without Excel installed. |
| 4 | MarkItDown MCP (Microsoft) | `uvx markitdown-mcp` **UNVERIFIED** (entrypoint name) | none | stdio | `RO WF` | `Y` | Converts Office/PDF/HTML to Markdown for LLM ingestion (microsoft/markitdown). |
| 5 | Office-Word MCP (GongRzhe) | `uvx office-word-mcp-server` **UNVERIFIED** | none | stdio | `WF` | `Y` | Read/create/edit `.docx` documents. |
| 6 | mcp-pandoc (vivekVells) | `uvx mcp-pandoc` **UNVERIFIED** | none | stdio | `WF` | `Y` | Format conversion between doc formats via pandoc. |

**Cerebro notes.** #1, #2, #3 are day-one presets: #2 subsumes most file work but is `EX+NW`, so Cerebro should ship #1 as the safe default and #2 as an explicit opt-in with the directory scope visible in the card. All three should get a native "Allowed folders" picker that rewrites the args array (Cerebro already knows user folders). #4/#5/#6 are convenience converters; for #4 the safer Cerebro integration is a *built-in* converter using its own Rust toolchain, since the MCP surface is one tool and the package is thin.

### 4.2 Git & version control (8 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 7 | Git (official) | `uvx mcp-server-git --repository C:\repo` | none | stdio | `RO WF` | `Y` | status/diff/log/show/branch/commit/add for a local repo. |
| 8 | GitHub (official) | `docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server` (or OAuth login; remote `https://api.githubcopilot.com/mcp/`) | `GITHUB_PERSONAL_ACCESS_TOKEN`, `GITHUB_TOOLSETS` (e.g. `repos,issues,pull_requests,actions,code_security`), `GITHUB_HOST` for GHES | stdio (Docker) / remote http | `RO WF NW AC` | `N` | Repos, issues, PRs, Actions, code security; toolset-scoped. |
| 9 | GitLab (community, zereight) | `npx -y @zereight/mcp-gitlab` **UNVERIFIED** | `GITLAB_PERSONAL_ACCESS_TOKEN`, `GITLAB_API_URL` | stdio (+sse) | `RO WF NW AC` | `N` | MRs, issues, pipelines, wiki on self-hosted or gitlab.com. |
| 10 | GitLab (official, beta) | remote endpoint **UNVERIFIED** (docs reference a hosted GitLab MCP) | OAuth | http | `RO WF NW AC` | `N` | First-party GitLab context; prefer once GA. |
| 11 | Bitbucket Cloud | via Atlassian Rovo (see #74) | OAuth | http | `RO WF NW AC` | `N` | Repos/PRs/pipelines inside the Atlassian umbrella server. |
| 12 | Gitea/Forgejo MCP (community) | repo/package **UNVERIFIED** (community `gitea-mcp` variants) | `GITEA_ACCESS_TOKEN` | stdio | `RO WF NW AC` | `N` | Self-hosted Forgejo/Gitea issue+repo automation. |
| 13 | DeepWiki MCP (Devin/Cognition) | remote `https://mcp.deepwiki.com/mcp` **UNVERIFIED** (free public endpoint) | none | http | `RO NW` | `N` | Ask questions about any public GitHub repo (auto-generated wiki). |
| 14 | Gitingest MCP (cyclotruc) | `uvx gitingest-mcp` **UNVERIFIED** | none | stdio | `RO NW WF(digest files)` | `N` | Repo → prompt-friendly digest for feeding local models. |

**Cerebro notes.** #8 is the flagship: prefer the Docker path when Docker is present (isolated, OAuth-capable), fall back to remote with OAuth, fall back to a PAT prompt. Cerebro's twin-pane UI should render `GITHUB_TOOLSETS` as checkboxes (each toolset is a visible permission surface). #7 pairs with the existing file server to create a "project workspace" bundle (repo folder + git + filesystem + code search).

### 4.3 Code intelligence & dev tools (10 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 15 | Context7 (Upstash) | `npx -y @upstash/context7-mcp --api-key YOUR_KEY` | `CONTEXT7_API_KEY` (optional; free tier w/o key) | stdio (+http) | `RO NW` | `N` | Injects version-accurate library docs into the model's context. |
| 16 | Serena (oraios) | `uvx --from git+https://github.com/oraios/serena serena start-mcp-server` **UNVERIFIED** | none | stdio | `RO WF EX(LSP)` | `Y*` | LSP-powered semantic code navigation/editing toolkit. |
| 17 | JetBrains MCP Proxy | `npx -y @jetbrains/mcp-proxy` **UNVERIFIED** | none (needs IDE plugin) | stdio | `RO EX(IDE actions)` | `Y` | Bridges a running JetBrains IDE's actions/refactors. |
| 18 | Semgrep MCP (semgrep/mcp) | `uvx semgrep-mcp` **UNVERIFIED** | optional `SEMGREP_APP_TOKEN` | stdio | `RO NW` | `Y*` | SAST scans, custom rule authoring, findings. |
| 19 | SonarQube MCP (SonarSource, official) | Docker image `mcp/sonarqube` **UNVERIFIED** | `SONARQUBE_TOKEN`, `SONARQUBE_URL` | stdio (Docker) | `RO NW AC` | `N` | Issues, quality gates, hotspots from SonarQube. |
| 20 | Snyk MCP | `snyk mcp` via Snyk CLI **UNVERIFIED** | `SNYK_TOKEN` | stdio | `RO NW AC` | `N` | Dependency + code vulnerability scanning. |
| 21 | Sequential Thinking (official) | `npx -y @modelcontextprotocol/server-sequential-thinking` | none | stdio | `RO` (pure reasoning) | `Y` | Structured multi-step reasoning scratchpad tool. |
| 22 | Everything (official) | `npx -y @modelcontextprotocol/server-everything` | none | stdio (+http) | `EX` (test only) | `Y` | Reference server exercising every MCP feature; use for conformance tests. |
| 23 | Repomix MCP (yamadashy) | `npx -y repomix --mcp` **UNVERIFIED** | none | stdio | `RO WF` | `Y` | Packs a repo into one AI-friendly file (xml/md/plain). |
| 24 | claude-context (Zilliz) | `npx -y @zilliz/claude-context-mcp@latest` **UNVERIFIED** | `OPENAI_API_KEY` or `MILVUS_*`/Zilliz endpoint | stdio | `RO NW EX(index)` | `P` | Semantic codebase search via Milvus/Zilliz vector index. |

**Cerebro notes.** #21 and #22 should be *built into* Cerebro's installer as hidden health-check servers (they are exactly what you use to validate the client). #15 has an obvious synergy with Cerebro's local models: cache the fetched docs text in the chat context to avoid re-fetching. #16 is the "code agent" heavyweight; because Serena edits files, map its write tools to `ask` and pair with the git server so every change is diffable. #19/#20 are enterprise-flavored; gate behind "enterprise" in the UI.

### 4.4 Shell & process control (5 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 25 | PowerShell MCP (community) | repo/package **UNVERIFIED** (several exist; e.g. `PowershellMCPServer`) | none | stdio | `EX NW` | `Y` | Run PowerShell commands with structured output. |
| 26 | Windows CLI MCP (community) | repo/package **UNVERIFIED** (cmd/PowerShell/git-bash wrapper variants) | none | stdio | `EX NW` | `Y` | Command allowlist/denylist execution on Windows shells. |
| 27 | mcp-shell (community) | repo/package **UNVERIFIED** | none | stdio | `EX NW` | `Y` | Minimal "run a shell command" server. |
| 28 | Docker MCP Gateway (official Docker) | `docker mcp gateway run --profile <profile>` | none | stdio gateway → containerized servers | `EX(in containers) NW` | `Y*` (images cached) | One stdio endpoint fronting a whole catalog of containerized MCP servers. |
| 29 | E2B MCP (e2b) | package **UNVERIFIED** (`@e2b/mcp-server` variants) | `E2B_API_KEY` | stdio | `EX NW AC` | `N` | Run untrusted code in cloud sandboxes. |

**Cerebro notes.** This is the category where I recommend building **first-party**: a `cerebro-shell` server in Rust whose design is already in your codebase's wheelhouse — job-object-supervised child processes, timeout per call, output clipping (e.g., 64 KB), command allowlist, and "ask every time" default that hooks into your existing per-tool policy UI. Third-party shell servers are all `EX` with wide blast radius and churn quickly; shipping the curated third-party list is fine, but mark them "community" and keep the first-party one as the recommended preset. #28 is strategic: if Docker Desktop is installed, Cerebro can offer *one* connection that fans out to dozens of catalog servers with Docker's isolation defaults, which is the cleanest sandbox story on Windows.

### 4.5 Browser & web automation (8 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 30 | Playwright MCP (Microsoft, official) | `npx @playwright/mcp@latest` (`--headless`, or `--port 8931` for HTTP mode → `http://localhost:8931/mcp`) | none; `PLAYWRIGHT_MCP_*` envs for tuning | stdio / local-http | `RO NW WF(screenshots/downloads)` | `Y*` (browser downloads on first use) | Accessibility-tree-driven browser automation (no screenshots required). |
| 31 | Chrome DevTools MCP (Google, official) | `npx -y chrome-devtools-mcp@latest` (`--slim --headless`, `--autoConnect`, `--browser-url=…`) | none | stdio | `RO NW` | `Y*` (Chrome required) | Control/inspect a live Chrome: console, network, perf traces, automation. |
| 32 | Puppeteer (official, **archived**) | `npx -y @modelcontextprotocol/server-puppeteer` | none | stdio | `RO NW` | `Y*` | Legacy browser automation/scraping; kept for compatibility only. |
| 33 | Browserbase MCP | `npx -y @browserbasehq/mcp` **UNVERIFIED** | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` | stdio | `RO NW AC` | `N` | Cloud headless browsers (stealth, sessions, proxies). |
| 34 | Cloudflare Browser Rendering (remote, official) | remote `https://browser.mcp.cloudflare.com/mcp` | OAuth | http | `RO NW AC` | `N` | Fetch/convert/screenshot pages at the edge. |
| 35 | Fetch (official) | `uvx mcp-server-fetch` | none | stdio | `RO NW` | `N` | URL → Markdown for LLMs; the canonical reference fetch tool. |
| 36 | Chrome MCP (hangwin) | package **UNVERIFIED** (`mcp-chrome` extension + bridge) | none | stdio | `RO NW AC(your browser)` | `Y` (local browser) | Drives your *existing* Chrome profile via extension. |
| 37 | Browser Use | package **UNVERIFIED** | none | stdio | `RO EX NW` | `Y*` | Vision-based web agent (screenshot→action loop). |

**Cerebro notes.** #30 is the best default web preset (fast, deterministic, headless-friendly); #31 is the debugging companion and pairs with Cerebro's "Turbo" toggle — when a page renders wrong, the model can pull console+network instead of guessing. Both need a one-click "install browser" step in the wizard (Playwright: `npx playwright install chromium`; Chrome DevTools MCP wants stable Chrome; detect via registry `HKLM\SOFTWARE\Google\Update` or `where chrome`). Keep #32 as archived with a deprecation badge. #33/#34/#37 are cloud sandboxes for when the user doesn't want their own browser touched — surface them as "cloud browser" options with per-session cost notes.

### 4.6 Search & research (8 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 38 | Brave Search (official Brave) | `npx -y @brave/brave-search-mcp-server --transport stdio` (or `--transport http`) | `BRAVE_API_KEY` (or `BRAVE_API_KEY_FILE`) | stdio / local-http | `RO NW` | `N` | Web/local/image/video/news search + AI summarizer. |
| 39 | Exa | `npx -y exa-mcp-server` (hosted: `https://mcp.exa.ai/mcp`, OAuth, or `?exaApiKey=…`, `x-api-key`) | `EXA_API_KEY` | stdio / http | `RO NW` | `N` | Neural web search, page fetch, multi-step research; hosted tier is keyless with rate limits. |
| 40 | Tavily | `npx -y tavily-mcp@latest` (remote: `https://mcp.tavily.com/mcp/?tavilyApiKey=…`) | `TAVILY_API_KEY`, optional `DEFAULT_PARAMETERS` JSON | stdio / http | `RO NW` | `N` | Search + extract + crawl tuned for agents. |
| 41 | Firecrawl | `npx -y firecrawl-mcp` (self-hosted: `FIRECRAWL_API_URL`) | `FIRECRAWL_API_KEY` (optional when self-hosted) | stdio / local-http (`HTTP_STREAMABLE_SERVER=true`) | `RO NW` | `P` (fully offline if self-hosted instance) | Crawl/scrape/extract entire sites, structured outputs. |
| 42 | Perplexity Ask | `npx -y @perplexity-ai/mcp-server` **UNVERIFIED** (package naming) | `PERPLEXITY_API_KEY` | stdio | `RO NW` | `N` | One-shot web-grounded Q&A. |
| 43 | DuckDuckGo MCP | `uvx duckduckgo-mcp-server` **UNVERIFIED** | none | stdio | `RO NW` | `N` | Keyless web search (rate-limited). |
| 44 | SearXNG MCP | community package **UNVERIFIED** | `SEARXNG_URL` | stdio | `RO NW` | `P` (fully local if you host SearXNG) | Metasearch over your own SearXNG instance — the privacy/offline answer. |
| 45 | Kagi MCP | community package **UNVERIFIED** | `KAGI_API_KEY` | stdio | `RO NW` | `N` | Premium paid search index. |

**Cerebro notes.** Ship #38, #39, #40 as the three search presets (different quality/cost tradeoffs; the catalogue card should show price tier as a badge). #41 doubles as the page-fetcher for RAG-ish workflows and can be pointed at a self-hosted instance — the only "truly offline-capable at scale" crawl option. Because each key is a separate billing surface, Cerebro's key-entry dialog should link straight to the vendor key page and validate the key immediately with one cheap call before saving (see §9.6).

### 4.7 Databases & data (18 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 46 | PostgreSQL (official, **archived**) | `npx -y @modelcontextprotocol/server-postgres postgresql://host/db` | conn string (args) | stdio | `RO` | `P` | Read-only SQL + schema inspection. |
| 47 | Postgres MCP Pro (crystaldba) | `uvx postgres-mcp --access-mode=restricted postgresql://…` **UNVERIFIED** exact flags | conn string | stdio | `RO WF` | `Y` (local PG) | Index tuning, health checks, explain plans, safe SQL access modes. |
| 48 | SQLite (official, **archived**) | `uvx mcp-server-sqlite --db-path C:\data\app.db` | none | stdio | `RO WF` | `Y` | Query/BI on a local SQLite file (also creates tables/rows). |
| 49 | MySQL (designcomputer) | `uvx mysql-mcp-server` **UNVERIFIED** | `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` | stdio | `RO WF` | `Y` (local) | MySQL schema/query execution. |
| 50 | MongoDB (official) | `npx -y mongodb-mcp-server@latest --readOnly` | `MDB_MCP_CONNECTION_STRING` (or Atlas API client id/secret) | stdio | `RO WF` | `Y` (local mongod) | Mongo + Atlas management; `--readOnly` strongly recommended. |
| 51 | Redis (official) | `uvx --from redis-mcp-server@latest redis-mcp-server --url redis://localhost:6379/0` | `REDIS_URL` or `--host/--port/--password/--ssl*` | stdio | `RO WF` | `Y` | Keys, hashes, streams, cluster ops on Redis. |
| 52 | ClickHouse (official) | `uvx mcp-clickhouse` **UNVERIFIED** args | `CLICKHOUSE_HOST`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_DATABASE` **UNVERIFIED** names | stdio | `RO` | `Y` (local CH) | OLAP queries + schema exploration. |
| 53 | MotherDuck / DuckDB | `uvx mcp-server-motherduck --db-path md:<db> --motherduck-token <t>` **UNVERIFIED** | token (or none for pure local DuckDB) | stdio | `RO WF` | `Y` (local `.duckdb` file) | Analytics SQL over local DuckDB or MotherDuck cloud. |
| 54 | Snowflake (Snowflake-Labs) | `uvx snowflake-labs-mcp --service-config-file <cfg>` **UNVERIFIED** | Snowflake account/user/key config | stdio | `RO WF NW AC` | `N` | Snowflake SQL + Cortex/warehouse ops. |
| 55 | BigQuery (community) | `uvx mcp-server-bigquery` **UNVERIFIED** | Google ADC credentials | stdio | `RO NW AC` | `N` | BigQuery queries + dataset introspection. |
| 56 | Elasticsearch (official Elastic) | `npx -y @elastic/mcp-server-elasticsearch` **UNVERIFIED** | `ES_URL`, `ES_API_KEY` | stdio | `RO` | `Y` (local ES) | Indices, mappings, search DSL. |
| 57 | Neo4j Cypher (official-ish, neo4j-contrib) | `uvx mcp-neo4j-cypher --transport stdio` | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `NEO4J_READ_ONLY` | stdio (+http) | `RO WF` | `Y` (local Neo4j) | Cypher queries + schema; `NEO4J_READ_ONLY=true` supported. |
| 58 | Memgraph | Docker image `mcp/memgraph` **UNVERIFIED** | Memgraph URL/user/pass | stdio (Docker) | `RO WF` | `Y` (local) | Cypher + graph algorithms on Memgraph. |
| 59 | Qdrant (official) | `uvx mcp-server-qdrant` | `QDRANT_URL` or `QDRANT_LOCAL_PATH`, `COLLECTION_NAME`, `QDRANT_READ_ONLY` | stdio (+sse) | `RO WF` | `Y` (local path mode) | Vector store/find as long-term memory. |
| 60 | Chroma (official-ish) | `uvx chroma-mcp` **UNVERIFIED** | `CHROMA_CLIENT_TYPE`, `CHROMA_DATA_DIR`, `CHROMA_HOST` | stdio | `RO WF` | `Y` (persistent local) | Local vector collections for RAG/memory. |
| 61 | Pinecone | package **UNVERIFIED** | `PINECONE_API_KEY` | stdio | `RO WF NW AC` | `N` | Hosted vector DB query/upsert. |
| 62 | Supabase (official) | `npx -y @supabase/mcp-server-supabase@latest --read-only --project-ref=<ref>` (remote: `https://mcp.supabase.com/mcp` with scoped URL) | `SUPABASE_ACCESS_TOKEN` | stdio / http | `RO WF NW AC` | `N` | Postgres SQL, migrations, Edge Functions, branches, logs. |
| 63 | Neon (official) | `npx -y @neondatabase/mcp-server-neon start <api-key>` **UNVERIFIED** exact CLI | `NEON_API_KEY` | stdio (+remote) | `RO WF NW AC` | `N` | Serverless Postgres project/branch/SQL management. |

**Cerebro notes.** Local DBs (#46–#53, #57, #59, #60) are the offline-capable subset and should be surfaced as "local data" presets; the wizard should offer a file picker for SQLite/DuckDB/Qdrant-local and a connection-string field with a "test connection" probe before saving. For hosted DBs, default every one of them to read-only flags where supported (verified for MongoDB `--readOnly`, Supabase `--read-only`, Neo4j `NEO4J_READ_ONLY`, Qdrant `QDRANT_READ_ONLY`) — write access for databases should require an explicit second toggle in Cerebro's install wizard. #62's remote variant has a URL-based scoping widget upstream; Cerebro should strip gadgetry and generate the scoped URL itself once the user picks a project.

### 4.8 Docs, notes & knowledge bases (8 entries)

| # | Server | Install (canonical) | Env / keys | Transport | Risk | Offline | One-liner |
|---|---|---|---|---|---|---|---|
| 64 | Notion (official) | **Recomm ended: remote `https://mcp.notion.com/mcp` (OAuth).** Local fallback: `npx -y @notionhq/notion-mcp-server` (maintenance mode) | `NOTION_TOKEN` (local) | stdio / http | `RO WF NW AC` | `N` | Semantic search + read/edit pages in Markdown. |
| 65 | Obsidian (MarkusPfundstein) | `uvx mcp-obsidian` | `OBSIDIAN_API_KEY`, `OBSIDIAN_HOST` (default 127.0.0.1), `OBSIDIAN_PORT` (default 27124) | stdio | `RO WF` | `Y` | Notes CRUD/search over the Obsidian Local REST API plugin (works on Windows). |
| 66 | Basic Memory (basicmachines) | `uvx --prerelease=allow basic-memory mcp` | none (writes `~/basic-memory` by default; optional sync creds) | stdio / https | `RO WF` | `Y` | Markdown knowledge graph with observations/wikilinks + semantic search; Obsidian-compatible. |
| 67 | Google Workspace (taylorwilsdon) | `uvx workspace-mcp --tools gmail drive calendar` | `GOOGLE_OAUTH_CLIENT_ID`, `GO
## Objective
- Produce a finished markdown paper ("Topic H — A curated MCP server catalogue for Cerebro") on stdout, per the task file `C:\Users\noahm\AppData\Local\Temp\claude\C--Users-noahm\28e64578-597b-41cb-8458-a14d4af849a4\scratchpad\rq_H.txt`.
- Constraints: do NOT create/edit any files; output ONLY the finished paper; cite primary sources; mark unconfirmed items UNVERIFIED; 120+ real MCP servers in 15 given categories with install command/env/transport/risk/offline/one-liner each; cover MCP 2025-06-18 Streamable HTTP + OAuth 2.1 resource metadata; desktop catalogue UX (search, categories, one-click install, key entry, health checks); end with ranked top-25 build list.

## Important Details
- Cerebro context: Windows desktop app, Tauri 2 + SolidJS, own Rust inference engine (CUDA + AVX2/VNNI CPU, GGUF, MoE offload), i9-12900H / 32 GB RAM / RTX 3070 Ti Laptop 8 GB; existing Model Hub, Tune, Monitor, Turbo, Chat with per-tool allow/ask/deny MCP rules, MCP stdio client with 5 presets, phone access.
- Paper taxonomy already committed in written output: risk tags `RO`/`WF`/`EX`/`NW`/`AC`; offline tags `Y`/`Y*`/`P`/`N`; transports `stdio`/`http`/`local-http`; 142 numbered entries planned across the 15 categories; per-category "Cerebro notes" paragraphs.
- Planned structure of remaining paper: §5 Streamable HTTP deep dive, §6 OAuth 2.1/resource metadata, §7 protocol drift (2025-11-25 / 2026-07-28: sessions removed, stateless, MRTR), §8 catalogue UX/data model, §9 Rust blueprint (rmcp crate, tokio, keyring→Windows Credential Manager, SQLite FTS5), §10 health/supervision/policy, §11 ranked top-25 (must be last), §12 sources.
- Key verified facts to reuse: Windows npx must wrap as `cmd /c npx`; uv Windows install `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`; registry API `https://registry.modelcontextprotocol.io/v0.1/servers?search=&updated_since=&version=latest` (cursor pagination, /v0.1/ping); spec pages `https://modelcontextprotocol.io/specification/2025-06-18/basic/transports` and `.../basic/authorization`; Streamable HTTP = single endpoint POST/GET/DELETE, `Accept: application/json, text/event-stream`, `Mcp-Session-Id`, `Last-Event-ID` resumption; OAuth refs draft-ietf-oauth-v2-1-13, RFC 8414, RFC 7591, RFC 9728, PKCE + RFC 8707 resource param.
- Design decisions for continuation: backoff `min(30s, 0.5s*2^n)+jitter`; health check 60 s interval, 3 misses → restart; schema hash re-prompt on tool change; prefer Docker isolation → uv → npx in install wizard.
- Entry numbering written so far: #1–6 Files, #7–14 Git, #15–24 Code, #25–29 Shell, #30–37 Browser, #38–45 Search, #46–63 Databases, #64–67 of Docs/notes.
- Remaining planned numbering: #68 MS365, #69 Joplin, #70 Zotero, #71 Readwise; #72–84 Productivity; #85–91 Design; #92–113 Cloud/DevOps; #114–121 Media; #122–127 Memory; #128–135 Math/science; #136–142 OS/Windows.
- Language: conversation is English; paper must stay English.

## Work State
### Completed
- Read task file rq_H.txt.
- ~40 web searches/fetches verifying: official servers README (active reference servers: Everything, Fetch, Filesystem, Git, Memory, Sequential Thinking, Time), 2025-06-18 transports + authorization spec, registry API docs, and vendor/community servers — GitHub (`ghcr.io/github/github-mcp-server`, remote `https://api.githubcopilot.com/mcp/`), Atlassian (`https://mcp.atlassian.com/v2/mcp`), Linear (`https://mcp.linear.app/mcp`), Notion, Sentry (`https://mcp.sentry.dev/mcp`), Stripe, Supabase, Cloudflare (`mcp.cloudflare.com/mcp`, `docs.mcp.cloudflare.com/mcp`), Playwright, Chrome DevTools MCP, Context7, Brave, Exa, Tavily, Firecrawl, Basic Memory, AWS awslabs (uvx), Azure (`npx -y @azure/mcp@latest server start`), Windows-MCP (`uvx windows-mcp`), Docker MCP gateway, Kubernetes MCP, Grafana, Figma Dev Mode (`http://127.0.0.1:3845/mcp`), Blender, MongoDB, Neo4j, Redis, Terraform (hashicorp docker), PagerDuty (`uvx pagerduty-mcp`), Google Workspace (`uvx workspace-mcp`), Slack zencoder, Jupyter, Obsidian, Excel, Qdrant, Datadog, Desktop Commander.
- Paper partially written on stdout: §1 Executive summary, §2 Method/sources/maintained definition, §3 Taxonomy, §4.1–4.7 complete (entries #1–63), §4.8 started through #67 (Google Workspace) — output truncated mid-table at "`GO".

### Active
- Finishing the markdown paper: continue §4.8 from #68, then §4.9–4.15 (#72–142), §5–§12, ending with the ranked top-25 build list.

### Blocked
- Exa-backed search hit 429 rate limits on several queries (Wolfram, arXiv, Elastic, Snowflake, ClickHouse, Chroma/Pinecone, Gmail/Calendar, some productivity servers) — those entries are marked UNVERIFIED in the paper.
- Previous output truncation cut the paper mid-table at §4.8 #67; continuation must not duplicate already-written sections.

## Next Move
1. Continue the paper exactly where truncated: finish §4.8 (#68 Microsoft 365, #69 Joplin, #70 Zotero, #71 Readwise) with risk/offline columns and UNVERIFIED tags, then write §4.9–4.15 using the planned numbering above.
2. Write §5 (2025-06-18 Streamable HTTP: sessions, SSE, resumption), §6 (OAuth 2.1 + RFC 9728 discovery), §7 (protocol drift note), §8 (catalogue UX: search/categories/one-click install/key entry/health checks), §9 (Rust/rmcp blueprint), §10 (health/policy formulas), §11 (ranked top-25, must be the final section), §12 (source list); keep total ≥120 entries and no file writes.

## Relevant Files
- `C:\Users\noahm\AppData\Local\Temp\claude\C--Users-noahm\28e64578-597b-41cb-8458-a14d4af849a4\scratchpad\rq_H.txt`: task file with full Topic H requirements and constraints.
- `https://registry.modelcontextprotocol.io/v0.1/servers`: official MCP Registry API (catalogue sync source).
- `https://modelcontextprotocol.io/specification/2025-06-18/basic/transports` and `.../basic/authorization`: primary spec sources for §5/§6.
Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.
