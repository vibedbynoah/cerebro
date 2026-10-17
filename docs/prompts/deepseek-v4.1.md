# Prompts for DeepSeek V4.1 (Cerebro grunt work)

How to use: open a NEW DeepSeek V4.1 session per prompt (DSH Desktop or `opencode run -m opencode/deepseek-v4.1-flash-free --variant max`),
paste ONE prompt, let it finish. Each prompt works in its own git worktree and ends with a commit on its own branch, so
nothing touches the main checkout; Claude reviews and merges. One task per session keeps its context small (faster, fewer
mistakes). If a prompt is long, save it to a file and say: "Read the full task in <file> and do exactly what it says"
(prompts over ~8 KB passed on the command line are silently dropped on Windows).

Every prompt starts with the shared preamble below. Paste preamble + task together.

---

## Shared preamble (paste first, every time)

```
You are working on Cerebro, a Windows desktop app for running local AI models fast. Read before you write; never guess an API.

Repos (git worktrees of one repo per product):
- App: C:\Users\noahm\Projects\cerebro-app  (Tauri 2 + SolidJS; Rust in apps\desktop\src-tauri\src, UI in apps\desktop\ui\src)
- Engine: C:\Users\noahm\Projects\cerebro-engine  (Rust crate crates\cerebro-engine; CUDA kernels in src\kernels.cu)

Rules:
1. Create your own worktree first and work ONLY there:
   git -C <repo> worktree add ..\<repo-name>-ds-<task> -b ds/<task>
   Never edit the main checkouts. Never run git stash, reset --hard, push, or delete files you did not create.
2. Match the surrounding code: naming, comment density, error style. Comments say why, one line.
3. No new dependencies unless the task says so. No placeholder code, no TODOs, no fake data: every number shown in
   the UI comes from the engine or the OS. Copy never names other products.
4. UI: design tokens only (var(--color-*), var(--space-*), var(--radius-*)); no hex/rgb colours; every flex/grid child
   min-width: 0; primary buttons "btn btn--primary btn--sm". New IPC commands must be added in FOUR places:
   src-tauri\build.rs COMMANDS, capabilities\main-window.json, isolation\index.js (no-arg set or a WITH_ARGS shape
   check), and generate_handler! in lib.rs. tests\unit\tauri-config.test.ts checks this.
5. Build and test before you commit (from apps\desktop unless noted):
   - Rust app:  cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && cargo test -p cerebro-desktop"   (run from the worktree root)
   - UI:        npx tsc --noEmit -p tsconfig.json && npx vitest run --config ui/vite.config.ts
   - E2E:       npx vite build --config ui/vite.config.ts && npx playwright test
   - Engine:    set PATH to MSVC 14.51 bin + Windows Kits 10.0.26100 bin, then cargo test --release --features cuda --lib
   Fix every failure and warning you caused. Do not weaken or delete existing tests.
6. Finish with ONE commit on your branch (message: what changed and why, plus measured numbers if any), then print:
   branch name, files changed, test results (pass counts), anything you could not finish and why. Keep it under 25 lines.
```

---

## 1. MCP client: Streamable HTTP transport

```
Task: add the MCP Streamable HTTP transport (spec 2025-06-18) to Cerebro's MCP client, next to stdio.
Read first: apps\desktop\src-tauri\src\mcp.rs (whole file), ui\src\screens\tools\McpServers.tsx, isolation\index.js (isMcpConfig).
- Config: a server entry may have "url" (http/https) and "headers" instead of "command"/"args"; same mcp.json shape
  others use ({"mcpServers": {"name": {"url": "...", "headers": {...}}}}).
- Client: POST JSON-RPC with Accept: application/json, text/event-stream; handle both a JSON reply and an SSE reply
  (read events until the response with our id); keep the Mcp-Session-Id from initialize and send it on every request;
  send MCP-Protocol-Version header after initialize; DELETE the session on stop; 404 on a request = re-initialize once.
  Use the reqwest blocking client already in Cargo.toml (rustls). Timeouts: 30 s connect, 60 s per call.
- Same Server state machine, status, stderr-tail equivalent ("last error"), tools/list paging and tools/call as stdio.
- UI: the Add form gets a "Local program / Web address" choice; isMcpConfig accepts url+headers.
- Tests: a Rust test that runs a tiny in-process HTTP MCP server on 127.0.0.1 (std TcpListener, JSON replies, one SSE
  reply) and checks initialize -> tools/list -> tools/call -> session header reuse. Unit test for isMcpConfig.
Task name for the branch: mcp-http
```

## 2. MCP catalogue (one-click servers)

```
Task: replace the 5 MCP presets with a searchable catalogue of real MCP servers.
Read first: ui\src\screens\tools\McpServers.tsx, ui\src\styles\tools.css, and (if present) the research paper
C:\Users\noahm\Projects\cerebro\docs\research\2026-10-17\H-mcp-catalogue.md.
- Data: ui\src\lib\mcp-catalogue.ts exporting a typed readonly array: id, name, category, description (one plain
  sentence), install {command, args[]} or {url}, env keys the user must fill (name, label, secret: bool, url to get it),
  risk ("read" | "write" | "run" | "network" | "account"), offline: bool, homepage. Only servers you can verify exist
  (official modelcontextprotocol/servers and well-known vendor servers). At least 40 entries. No made-up packages.
- UI: a "Browse servers" panel in the MCP section: search box, category chips, cards with risk badge and offline badge,
  "Add" opens the existing form pre-filled with env fields to fill (secrets in password inputs).
- Tests: vitest that every entry has the required fields, unique ids, no duplicate names; an e2e test that searches,
  adds one entry and sees it in the server list (web build uses the mock service).
Task name: mcp-catalogue
```

## 3. Memory estimator and model recommendations

```
Task: a real memory estimator and hardware-based model recommendations, shown in Model Hub.
Read first: ui\src\screens\Hub.tsx and its hub components, src-tauri\src\hub.rs, src-tauri\src\engine.rs (LibraryModel,
hardware()), and C:\Users\noahm\Projects\cerebro\docs\research\2026-10-17\G-unsloth.md if present.
- ui\src\lib\memory-estimate.ts: estimate(model, quant, ctx, kvType, offload) -> {weightsGiB, kvGiB, scratchGiB,
  vramGiB, ramGiB, fits: "gpu" | "split" | "cpu" | "no", note}. Weights from the GGUF file size (real), KV from layers x
  kv_heads x head_dim x 2 x ctx x bytes(kvType), MoE: experts may sit in RAM. Read layer/head numbers from the GGUF
  metadata the Rust side already parses (add an IPC command hub_file_meta(repo, file) that range-reads the GGUF header
  over HTTPS if needed; header only, max 8 MB).
- Hub shows per file: "Needs X GB VRAM + Y GB RAM at 8K context" and a context slider that updates it live.
- Recommendations: on the Hub home, "Best for this computer": rank by fits + size + quality tier (UD/Q4_K_M/Q5/Q8),
  using the machine's real VRAM/RAM from hardware().
- Tests: vitest for estimate() against hand-computed cases (Qwen3-0.6B Q8_0 at 4096 ctx, a 30B-A3B MoE Q4_K_M at 32K);
  e2e that moves the slider and sees the number change.
Task name: memory-estimate
```

## 4. Built-in tools for Chat

```
Task: built-in tools models can use in Chat without installing an MCP server: read_file, list_directory, search_files,
write_file, edit_file (exact string replace), run_command (PowerShell), fetch_url (GET, text only).
Read first: src-tauri\src\engine.rs (ToolHost, run_chat), src-tauri\src\commands.rs (McpToolHost), src-tauri\src\toolcall.rs,
ui\src\lib\permissions.ts (Policy, rules, mcpToolKey), ui\src\screens\Tools.tsx.
- Rust module src-tauri\src\builtin_tools.rs; a composite ToolHost that lists built-ins (server name "cerebro") plus MCP
  tools. Built-ins work only inside folders the user added ("workspace folders", a new list in Settings > Tools);
  any path outside is refused with a clear message. Paths canonicalized; no symlink escape.
- run_command: PowerShell with -NoProfile, working dir = first workspace folder, 60 s timeout, output capped at 16 KB,
  process tree killed on timeout (Windows Job Object). Default rule for run_command and write/edit tools: ask.
- fetch_url: http/https only, 2 MB cap, text/html converted to plain text, honours the Offline mode setting.
- Tests: Rust tests for path confinement (.., absolute paths, symlinks), edit_file uniqueness check, timeout kill.
Task name: builtin-tools
```

## 5. CPU decode speed (engine; measure everything)

```
Task: make CPU decode of dense Q8_0 models faster. Today Qwen3-0.6B Q8_0 decodes at 66.9 tok/s with 16 threads;
llama.cpp b10751 does 72.5. Goal: beat 72.5 without changing outputs.
Read first: crates\cerebro-engine\src\quant.rs (dot_q8_0_q8_vnni, dot_q8_0_q8_x4), model.rs (matmul, Session::step),
pool.rs (thread pool), switches.rs. Model: C:\models\dev\Qwen3-0.6B-Q8_0.gguf.
Measure with: target\release\examples\bench.exe --model C:\models\dev\Qwen3-0.6B-Q8_0.gguf --out runs.jsonl --machine A
  --reps 5 --decode 256 --lengths 32,512 --backends cpu
Ideas to try ONE AT A TIME, keep only what measures faster (median of 5, report before/after):
  a) row-interleaved weight repack at load (4 rows x 32 per block) so one activation load feeds 4 rows of VNNI;
  b) pin pool threads to P-cores first (GetLogicalProcessorInformationEx), E-cores only if they help;
  c) spin-then-park barrier in pool.rs instead of condvar wake per matmul;
  d) fuse rmsnorm+quantize and silu*up+quantize on CPU;
  e) prefetch the next weight rows (_mm_prefetch) in the dot loop.
Every kept change: behind a switch in switches.rs (Beta), bit-identical logits vs before (examples\batchcheck.rs style
check over 64 tokens), and a line in the commit message with the numbers. If nothing beats the baseline, commit nothing
and report the measurements.
Task name: cpu-decode
```

## 6. Orchestrate mode UI (GVS5H) - only after Claude lands the backend

```
Task: the Chat UI for Orchestrate mode. The backend command chat_orchestrate already exists (read src-tauri\src\orchestrate.rs
and the ChatEvent::Ledger events in engine.rs first; if they do not exist yet, stop and say so).
- Composer: a "Orchestrate" toggle with a workers count (2-6) and rounds (1-8), saved in chat prefs.
- While running: a "Ledger" panel in the reply showing plan.md, notes.md and the task list live (text only), each worker's
  current step, and the final answer streaming as the reply text.
- Tests: e2e against the mock (add the mock events in mock\chat.ts), axe clean in both themes.
Task name: orchestrate-ui
```

---

## Why these prompts save credits

- One task per session: DeepSeek's context stays small, so it is faster and makes fewer mistakes.
- Exact file list to read first: no blind exploration.
- Its own branch and worktree: Claude only reviews a diff instead of redoing the work.
- Built-in measurement and tests: a change that does not measure faster or does not pass never reaches review.
- Short final report: the review starts from facts (branch, tests, numbers), not a long narrative.
