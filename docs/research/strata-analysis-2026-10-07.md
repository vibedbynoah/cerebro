# Strata: what Cerebro takes from it (2026-10-07)

Source: https://github.com/Niko1221/Strata (MIT, C++/CUDA/HIP engine + Python server, 16.9k stars,
pushed daily). Owner's install: `C:\Users\noahm\strata` (commit 6f32ec0, IQ3_XXS pack, port 8080).
Owner's verdict: the engine and its monitor are excellent; as a coding harness it is weak. Cerebro is
the harness; Strata is a backend to host and a monitor to learn from.

## Measured on this PC (from strata-iq3_xxs.log)

At ~100K context, IQ3_XXS, RTX 3070 Ti 8 GB: prefill of new tokens 11-14 tok/s with ~97% of the prompt
reused from the conversation cache, decode 5.6-7.7 tok/s, MTP drafts accepted 59-66%, decode expert
cache hit rate 5.6-7.6%. The owner reports ~20 tok/s on short prompts. The low hit rate is the 8 GB
card: every extra GB holds ~700 experts (their docs), so this machine is CPU-bound on experts.

## Backend ideas worth adopting

| Idea | What Strata does | Cerebro use |
|---|---|---|
| Tiered placement | Dense parts + KV + an adaptive expert cache in VRAM; all experts pinned in RAM and computed by the CPU in place, concurrently with the GPU; a 28.8 GB n-gram table read from SSD a few rows per token | Tune: treat Strata as the MoE backend for models it supports; for llama.cpp, keep `-ncmoe` placement but learn the "fill remaining VRAM with hot experts" policy |
| Speculation | Model's own MTP layer drafts up to 3 tokens (2.4-3.2 accepted per pass, 1.6-1.8x); prompt-lookup drafts up to 5 only where measured acceptance pays (code edits +6-11%) | Engine: enable MTP / prompt-lookup per model automatically, gated on measured acceptance, never by default when it loses |
| Big prefill chunks | 8,192-token chunks with next layer's experts streamed over PCIe during attention; byte-sized ring | Tune: expose and auto-pick `-ub`/`-b` per model from a measured sweep |
| Conversation cache | Keeps last conversation state (reads only new tokens); optional parking of several conversations in RAM (`--conversation-cache-mib`); message-boundary checkpoints so an agent that rewrites its last message does not reread history | Server: this is the single biggest win for a coding harness (agents resend 100K-token histories). Use llama.cpp slot save/restore + prompt cache; park per-session state |
| KV in RAM past 64K | Only the most-read part of KV in VRAM, rest streamed with prefetch | Long-context mode for 8 GB cards |
| VRAM elastic | Engine gives back VRAM on request so games/other apps can share the GPU | App: "share GPU" toggle |
| Fault recovery | Reasoning-loop rescue, fatal-recovery restart with waiters, quiet cancel, Windows Job Object | Supervisor already has Job Objects; add loop detection + auto restart without dropping queued requests |
| Install by AI / MCP | `docs/AI_SETUP.md` written for coding agents; `tools/strata_mcp.py` installs/starts/stops it | Cerebro MCP server: "install model X", "start", "stop", "status" for any assistant |
| APIs | OpenAI chat, Anthropic messages, Responses API (Codex CLI), MCP tool host, JSON response formats | Server: all four, same as planned; Responses API is new for us |
| Evidence culture | Every claim has a dated bench folder with method, matrix and community runs | Matches docs/CLAIMS.md; copy the `bench/results/<date>-<topic>/README.md` layout |

## The monitor (what makes it good)

Files: `serve/web/app.js` (Monitor tab), `serve/web/monitor.html/js` (request inspector),
`serve/telemetry.py` (1 Hz sampler), `serve/prometheus.py` + `docs/monitoring/grafana-strata.json`.

- One `GET /metrics` JSON polled every second carries everything: `live` (state reading/generating/idle,
  prompt_read/prompt_total, generated/max_tokens, tok_s, prefill mean, queued), `hardware` (now),
  `hardware_static` (GPU name, totals), `engine` (expert slots cached, version), `history` (last 60
  samples per series for sparklines), `requests` (last 12 or all), `totals`, `conversation_cache`.
- Header pill: Idle / "Reading prompt 43%" / "Generating 21.4 tok/s" / "2 queued".
- State card with badges and a progress bar (prompt read %, or generated / max_tokens).
- Eight metric cards with 60 s SVG sparklines: Speed (decode and prefill together), GPU load, VRAM
  (with "N experts cached"), GPU temp, power, PCIe RX, CPU, disk read. Multi-GPU shows per-card values.
- Telemetry without dependencies: NVML loaded with ctypes from the driver's `nvml.dll` (load, VRAM,
  temp, power + limit, PCIe gen/width/throughput); AMD via Linux sysfs; CPU/RAM/disk from psutil or OS
  fallbacks. Anything unreadable is None and never stops the server.
- Conversation-cache card: requests that reused their prompt, tokens reused (% of all), parked/restored,
  last switch.
- Totals line: "Since Tue 14:02: 41 requests, 1.2M prompt tokens read at 1,420 tok/s (3.9M reused),
  88K written at 21.3 tok/s".
- Request inspector: every API call listed live (state tag, path, wall time, stream/JSON, HTTP status),
  search + filter, detail with load / queue / first-token times, prompt and output tokens, speed,
  response format, and tabs for input body, output, reasoning, response (256K capture cap), copy button,
  load/unload buttons, API key prompt, theme toggle.
- Prometheus exporter + Grafana dashboard + Kubernetes ServiceMonitor for people who want history.

Its weaknesses (our opening): one model family, one request at a time by default, a chat page rather
than a coding harness (no tools/permissions/diffs/projects), monitor history only 60 s, no per-session
cost/latency breakdown, no alerts.

## Cerebro tasks this creates

1. engine `telemetry` crate: 1 Hz sampler, NVML via dynamic load (no CUDA dependency), AMD (sysfs on
   Linux, ADLX/D3DKMT on Windows), Intel, CPU/RAM/disk/thermals; 60 s ring + 24 h downsampled history;
   never fails the server.
2. server `/metrics` JSON (same shape idea) + Prometheus text endpoint + request capture store
   (input, output, reasoning, timings, 256K cap, redaction toggle) with a live event stream.
3. app **Monitor** screen: header pill, state card, 8+ sparkline cards (decode+prefill, GPU, VRAM with
   expert/KV split, temp, power, PCIe, CPU, RAM, disk), conversation-cache card, totals, request
   inspector with tabs; plus what Strata lacks: history beyond 60 s, per-session and per-tool breakdowns,
   alerts (thermal, VRAM pressure, speed regression), export.
4. backend adapter: run Strata (`strata.exe` + run config) under cerebro-supervisor as an alternative to
   llama-server for Qwen3.8-Flash-Next; keep llama.cpp for everything else. MIT licence: credit it.
5. conversation cache + message-boundary checkpoints in the server for agent workloads.
6. Cerebro MCP server for install/start/stop/status (Strata's MCP_SERVER pattern) and an AI_SETUP.md.
