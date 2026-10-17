# Cerebro product features: monitor, MCP, customisation, turbo toggles

Task C, 2026-10-16. Companion to the engine and turbo papers of the same date. Input read first:
`docs/specs/engine.md`, `docs/specs/turbo-system.md`, `docs/specs/positioning.md`,
`docs/specs/effort-levels.md`, `docs/measurements/engine/2026-10-10-engine-report.md`,
`docs/CLAIMS.md`, `docs/FEATURE_CATALOG.md`, and the app worktree (`cerebro-app`: `design/`,
`apps/desktop/ui/src`, `apps/desktop/src-tauri/src`).

**270 numbered features**, grouped as required. Counts: Monitor 35 (MON), Turbo toggles 45 (TRB),
MCP 35 (MCP), Customisation 30 (CUS), Workspaces 20 (WSP), Agent harness 25 (AGT), Model hub 20
(HUB), Onboarding 15 (ONB), Reliability 15 (REL), Privacy 15 (PRV), Accessibility 15 (ACC).
Ends with the ordered P0 build list (30 items).

## Conventions

- **Schema per feature:** id · name — what it is; user value; UI placement; engine data needed;
  effort (S/M/L); priority (P0/P1/P2); channel (Stable/Beta/Experimental); risk; source. Turbo
  toggles additionally carry: expected gain on this hardware (estimate + confidence), an
  implementation sketch for the Rust+CUDA engine (data structures, kernel shape, files), and a
  one-sentence user-facing toggle explanation in plain language (**copy rule from positioning.md:
  benefit only, never names another product**).
- **Effort** is relative to this codebase: S ≈ days (one screen/command, existing primitives),
  M ≈ 1-3 weeks (new subsystem or engine path + tests), L ≈ multi-month (new engine capability or
  platform surface).
- **Gain estimates** are only for this machine (i9-12900H 6P+8E AVX2+AVX-VNNI, 32 GB DDR5-4800,
  RTX 3070 Ti Laptop 8 GB sm_86 @80-115 W) unless stated. Confidence: high (measured on this box or
  arithmetic from measured data), medium (mechanism well understood, magnitude uncertain), low
  (no local data yet).
- **`(exists: path)`** marks what the app already ships today, from reading the app worktree; a
  field written as `null` in `metrics()` (`apps/desktop/src-tauri/src/commands.rs:173`) is data the
  engine must start producing — those are marked `engine: NEW →`.
- **UNVERIFIED** marks anything not checked against a primary source or a local measurement.
  All external numbers are cited; numbers from `2026-10-10-engine-report.md` are local measurements
  (method and raw rows in `runs.jsonl` next to it).
- Comparison numbers against other engines appear only inside `gain:` fields (internal research,
  the comparison-page exception in positioning.md). No user-facing toggle string names a product.

## 0. What exists today (grounding)

| Area | Shipped in the app now | Missing (the gap this paper fills) |
|---|---|---|
| Monitor | Screen shell with cards, breakdown, inspector, sparkline (`ui/src/screens/monitor/*`); `metrics()` snapshot with live/history/requests/totals | ~40% of fields are `null` (GPU util, temps, power, PCIe, CPU, prefill, TTFT, speculation, conversation-cache stats); no alerts, no export, no retention beyond 60 s |
| Turbo | Machine profiles, engines/builds list, schedulers, profiler share bars, channels (Stable/Beta/Experimental) (`ui/src/screens/Turbo.tsx`); Tune autotuner screens (`screens/tune/*`) | Individual optimisation toggles with per-toggle channel, measured-impact badge, beginner/advanced explanations, auto-revert |
| Tune | Setup → Run → Winner flow, bits/charts | Feeding winners into toggle profiles and Monitor badges |
| MCP | Nothing MCP-specific; Tools screen has built-in tool rules + activity log (`screens/Tools.tsx`); plugins screen/library (`screens/Plugins.tsx`, `lib/plugins.ts`) | MCP client, servers, catalogue, mcp.json, per-server permissions |
| Customisation | Theme island (dark/light), experience level, config-file JSON section, command palette, sidebar collapse, reduced motion (design system) | Density, profiles/presets, keybindings, per-model settings pages, plugins API contract |
| Workspaces | Chat / Image Studio / Documents as separate screens; folder picker inside individual screens | Workspace concept: kinds, per-workspace mcp.json, settings, history, permissions |
| Agent | Not yet a harness in the app (catalog specs exist in `docs/FEATURE_CATALOG.md`, coding-agent 66 items) | Manager/worker, ledger, verifier, routing |
| Model hub | `hub_search` / `hub_files` / `hub_download` with parts+progress (`commands.rs:277-310`), LiveHub screen | Fit prediction, quant ranking, speed prediction, local import |

## 1. Monitor — every live metric worth showing (35)

Positioning binds this group: "Live metrics are a signature feature, not a debug page… show the
model and the system together with percentages." Every item below shows a value, a percentage of a
meaningful whole, and a plain-language meaning.

**MON-01 · Decode tokens/sec, live** — what: tok/s of the current generation, 10 s moving average,
session peak; value: the headline number, honest and sampled (labelled as such per CLAIMS); UI:
Monitor → Speed card (exists: `screens/monitor/cards.tsx`); engine: `metrics().live.decodeTps`
(exists) — needs smoothing + peak; effort **S** · **P0** · Stable; risk: timer jitter (state the
sampling window); source: positioning.md §Show the machine thinking.

**MON-02 · Prefill tok/s with prompt-progress %** — what: prompt tokens read / prompt total as a
progress % plus prefill tok/s when it finishes; value: long prompts stop being a black box, user
sees "reading your prompt: 62%"; UI: Monitor → Speed card secondary line + live phase bar (exists
`screens/monitor/Breakdown.tsx`); engine: `live.promptRead/promptTotal` exists, `prefillTps` is
`null` → **NEW**; effort **S** · **P0** · Stable; risk: chunked prefill makes % stepwise (show per
chunk); source: engine.md §4 chunked prefill.

**MON-03 · Time to first token, per request and p50/p95** — what: TTFT for this message, plus
session median and p95 in history; value: the latency users actually feel; UI: Monitor → request row
+ Speed card "first token" tile; engine: `requests[].firstTokenS` exists, `live.firstTokenS`
`null` → **NEW**; effort **S** · **P0** · Stable; risk: low; source: 2026-10-10 report method
(first-token timing already recorded in `runs.jsonl`).

**MON-04 · Prompt reuse %** — what: reused prompt tokens / prompt tokens for the request and cumulated
for the session ("68% of this prompt was already computed"); value: explains why turn 2 is faster;
UI: Monitor → Cache card + per-request inspector row; engine: `conversationCache.reusedTokens`
exists, per-request `reusedTokens` exists; effort **S** · **P0** · Stable; risk: low; source:
engine.md §1 conversation cache; prefix-caching TTFT evidence arXiv:2510.15152, arXiv:2601.06007
(cost −41-80%, TTFT −6-31% measured on API providers — external numbers, UNVERIFIED for our engine).

**MON-05 · MoE expert-cache hit rate %** — what: VRAM expert hit rate now and averaged over the
session; value: the single number that explains MoE speed after adaptation; UI: Monitor → Experts
card (exists, `moe.cacheHitPct`); engine: `moe.cacheHitPct` exists (31-48% first turn → 75-85%
adapted, measured); effort **S** · **P0** · Stable; risk: low; source: engine-report rows 41-44.

**MON-06 · Active experts per token, with % of model** — what: e.g. "8 of 512 = 1.6% of the model is
used per token"; value: makes MoE legible to beginners, grounds the VRAM discussion; UI: Monitor →
Experts card; engine: `moe.expertsPerToken`, `expertsTotal` exist; effort **S** · **P1** · Stable;
risk: low; source: positioning.md MoE bullet.

**MON-07 · Expert residency split (VRAM / RAM / SSD) as counts and %** — what: experts cached in
VRAM vs mapped in RAM vs streamed from SSD, per tier, plus % of total experts; value: shows exactly
what the adaptive cache bought (report: 1,467 → 1,765 experts resident from one optimisation); UI:
Monitor → Experts card, expandable bar; engine: `moe.cacheSlots` exists, RAM/SSD split **NEW**;
effort **M** · **P1** · Stable; risk: requires residency accounting in `pool.rs` (expert pool
owner); source: engine-report "What moved the numbers" row on K-quant attention weights.

**MON-08 · Per-layer routing heat map** — what: layer × time heat map of expert-cache misses;
value: pinpoints which layers force streaming (the 26-28 GB/s in-step gap investigation becomes
visible); UI: Monitor → Experts → heat map (new sub-panel); engine: **NEW** (per-layer miss counters
exported in `metrics().moe.layerMisses`); effort **M** · **P1** · Experimental channel for the panel
until counters are cheap (per-layer atomics must not cost decode); risk: telemetry overhead can eat
the very gain being shown — sample every N tokens; source: positioning.md heat-map bullet.

**MON-09 · Hottest experts this session** — what: top-N expert ids by residency time/usage with
layer origin; value: debugging + explains cache churn; UI: Monitor → Experts → list; engine: **NEW**
(frequency counters in the cache, reuse `pool.rs` LRU metadata); effort **S** · **P2** · Stable;
risk: low; source: engine.md §2.

**MON-10 · Speculation: draft acceptance % and tokens per verify pass** — what: % of drafted tokens
accepted, effective speed-up ×; value: tells the user whether speculation is earning its keep;
UI: Monitor → Speed card "speculation" tile; engine: `engine.speculation` exists but `null` →
**NEW** (acceptance counter at verify site in `model.rs`); effort **M** · **P1** · Beta (feature is
Experimental); risk: none to display; source: engine.md §3 "enabled only while measured acceptance
pays".

**MON-11 · VRAM split: weights / KV / expert cache / scratch, each as % of total** — what: four-way
split bar of the 8 GB with headroom %; value: the "would +2 GB help?" question answered at a glance;
UI: Monitor → Memory card (exists: `hardware.vram*`); engine: aggregate VRAM exists, per-pool split
**NEW** (allocator counters); effort **M** · **P0** · Stable; risk: pool counters must be updated on
every alloc/free (test against `nvidia-smi` total); source: turbo-system §4 bottleneck verdict.

**MON-12 · KV fill % of context window** — what: contextUsed/contextMax with projected messages
remaining at current usage; value: prevents surprise truncation, motivates parking/quant KV; UI:
Monitor → Memory card second bar + composer ring; engine: `model.contextUsed/contextMax` exist;
effort **S** · **P0** · Stable; risk: low; source: engine.md §1/§5.

**MON-13 · RAM used / pinned / available, with % of physical** — what: system RAM plus Cerebro's
pinned (non-pageable) allocation as % of total; value: explains paging risk when experts live in RAM
(report: 12-17 GB free while running); UI: Monitor → Memory card; engine: `hardware.ramUsedGiB`
exists, `ramPinnedGiB` **NEW**; effort **M** · **P1** · Stable; risk: Windows query needs
`QueryProcessMemoryInfo`/pool accounting; source: engine-report Machine A notes.

**MON-14 · Disk streaming rate + SSD headroom per tier** — what: current read MB/s of expert/model
streaming, plus free space on the model volume as % (warn <10%); value: SSD streaming is a first-class
placement (engine matrix), users see when the disk is the bottleneck; UI: Monitor → Memory card +
alerts; engine: `hardware.diskReadMBs` **NEW** (ReadProcessMemoryCounters or PDH counters), disk
free from `hw.rs`; effort **M** · **P1** · Stable; risk: low; source: engine.md placement matrix
"SSD streaming (memory-mapped)".

**MON-15 · PCIe throughput vs link maximum %** — what: Rx/Tx MB/s with negotiated gen/width and %
of link capacity; value: expert swapping lives or dies on PCIe; verdict phrasing like "expert swaps
use 41% of the link"; UI: Monitor → System card; engine: `hardware.pcieRxMBs/pcieGen/pcieGenMax/
pcieWidth` all **NEW** (NVML `nvmlDeviceGetPcieThroughput` + `nvmlDeviceGetPcieUtilization`;
gen/width from `nvmlDeviceGetPcieCurrLinkGen/Width`); effort **M** · **P1** · Stable; risk: NVML
throughput is a sampled counter (label sampling window); source: engine-report MoE async-copy work;
`examples/pcie.rs` already exists for local measurement.

**MON-16 · GPU power vs limit %, temperature, clocks, throttle reason** — what: watts drawn / watts
allowed (%), °C with throttle threshold, SM/memory clock; value: on a power-limited 80/115 W laptop
this is the honesty panel — turbo profiles must show what they are fighting; UI: Monitor → System
card (tiles exist, values `null`); engine: `hardware.gpuPowerW/gpuPowerLimitW/gpuTempC` **NEW** via
NVML (`nvmlDeviceGetPowerUsage/GetPowerLimit/GetTemperature/GetClockInfo`); effort **M** · **P0** ·
Stable; risk: NVML init on some laptop drivers returns unsupported for fan only — degrade to power
+ temp; source: turbo-system §3 "power limit and clock locks through NVML".

**MON-17 · GPU utilisation % and SM busy %** — what: graphics-card busy % while generating; value:
distinguishes compute-bound (util ≈ 100) from PCIe/CPU-bound (util low) at a glance; UI: Monitor →
System card; engine: `hardware.gpuUtil` **NEW** (NVML utilization; SM-busy needs Nsight-level counters
— UNVERIFIED that driver API exposes them, mark as util% only at first); effort **S** · **P1** ·
Stable; risk: util% is coarse (samples over ~1 s); source: positioning.md System bullet.

**MON-18 · CPU per-core load with P/E-core split %** — what: 20 threads grouped 6P+8E, package
power if exposed; value: shows thread-pinning and worker-pool decisions working (measured 1.9 →
11.4 tok/s from a yielding pool); UI: Monitor → System card, expandable core grid; engine:
`hardware.cpu` **NEW** (PDH `Processor Information(\Processor Information(*)\% Processor Utility)`,
per-core; E-core grouping from `hw.rs` topology); effort **M** · **P1** · Stable; risk: sampling cost
— 1 s default; source: turbo-system §2 P/E pinning; engine-report worker-pool row.

**MON-19 · Conversation-cache statistics** — what: parked slots, budget used %, parks/restores/
evictions, last event; value: users understand why switching back to an old chat is instant; UI:
Monitor → Cache card; engine: `conversationCache.*` shape exists but all `enabled:false/0` → **NEW**
(wire to radix-tree stats in engine §1); effort **M** · **P1** · Stable; risk: low; source: engine.md
§1; external evidence for cache economics arXiv:2510.09665 (LMCache, up to 15× throughput in
serving settings — UNVERIFIED for our single-user engine).

**MON-20 · Per-request inspector** — what: every request with wall time, queue time, load time,
TTFT, prompt tokens, reused tokens, output tokens, decode/prefill tok/s, phase, session, tool; value:
"which message was slow, and why" becomes a query, not a guess; UI: Monitor → Requests table →
row detail (exists: `screens/monitor/Inspector.tsx`, rows from `metrics().requests`); engine: rows
exist, `loadS/queueS/prefillTps/httpStatus` null → **NEW**; effort **M** · **P0** · Stable; risk:
retention size (cap 100 rows exists — add ring file for history); source: engine.md §7 telemetry.

**MON-21 · History retention and ranges (60 s / 5 min / 1 h / session)** — what: sparkline ring
buffer switches range, backed by a compact on-disk series; value: "was it slower this evening?" is
answerable; UI: Monitor → history cards (exists `Sparkline.tsx`); engine: `history.series` 60-point
ring exists → **NEW** persistent series (append-only JSONL/CSV, 1 Hz, compress, cap by age); effort
**M** · **P1** · Stable; risk: disk writes while generating — append on a background thread, never on
the decode path; source: 2026-10-10 report ("raw rows… runs.jsonl") as the pattern to keep.

**MON-22 · Alerts with thresholds (temp, power, VRAM, KV full, tok/s regression)** — what: user-set
thresholds → toast + status-bar chip + log row (e.g. "GPU at 87 °C, clocks reduced"); value: heat and
throttling become visible before users blame the model; UI: Monitor → Alerts strip + Settings >
Monitor; engine: needs MON-13..17 signals; effort **M** · **P1** · Stable; risk: alert fatigue —
default only 3 alerts (temp, OOM-risk, engine restart); source: turbo-system §3 auto-revert on
temperature/power limits.

**MON-23 · Export: CSV / JSON of history and requests** — what: one click exports the selected range
+ request table with machine/model/build context per CLAIMS; value: users can prove claims, file
issues with data; UI: Monitor → Export button; engine: series + requests serializer; effort **S** ·
**P1** · Stable; risk: exports may contain prompts — default export metrics-only, prompts off
(see PRV-06); source: docs/CLAIMS.md (every number ships with machine, model, build, context).

**MON-24 · Per-token phase timeline (GPU kernels / CPU experts / PCIe / cache hits)** — what: the
profiler's per-token timeline attached to a live request; value: matches the Turbo profiler but on
real traffic; UI: Monitor → request detail → timeline (new); engine: profiler parts exist in Turbo
(`Turbo.tsx` profiler share bars) → **NEW** live sampling API; effort **L** · **P2** · Experimental;
risk: sampling overhead, must be disabled by default under 1% CPU budget; source: turbo-system §4
profiler.

**MON-25 · Bottleneck verdict card (plain language)** — what: one sentence, e.g. "CPU-bound on
experts: +2 GB VRAM would give about +700 experts"; value: turns metrics into an action; UI: Monitor
→ top insight card; engine: rule table over MON-05/07/11/15/16 (verdict strings live in UI layer);
effort **M** · **P1** · Beta; risk: wrong verdicts erode trust — only fire when a signal is
unambiguous (>=2 corroborating counters), else show nothing; source: turbo-system §4 verbatim
example.

**MON-26 · Session totals and "time saved"** — what: tokens generated, prompt tokens, reused tokens
and the prefill seconds saved by reuse (computed from measured prefill tok/s), messages, wall time;
value: the cache shows its worth in seconds, not percentages; UI: Monitor → Totals strip (exists:
`totals`); engine: `totals` exists → add `prefillSecondsSaved` **NEW**; effort **S** · **P1** ·
Stable; risk: label as estimate (derived, not timed A/B); source: MON-04 pattern; CLAIMS labelling
rule.

**MON-27 · Baseline strip: this run vs your tuned baseline** — what: today's median tok/s vs the
Tune-screen winner run for the same model/context, as a % (e.g. "96% of tuned"); value: regressions
and profile wins are visible without a spreadsheet; UI: Monitor → Speed card footer; engine:
`engine.baseline` exists (`null`) → **NEW** (read Tune results store); effort **M** · **P1** ·
Stable; risk: context must match or the % lies — key baselines by (model, quant, ctx, profile) and
hide when keys differ; source: turbo-system §4 "comparison runs… method saved".

**MON-28 · Model facts card** — what: quant, params, architecture, layers on GPU, KV type, backend
build/version; value: trust and bug reports; UI: Monitor → Model card (exists: `engine.model`);
engine: fields exist except `kvType` when split (currently hardcoded f16/f32) → improve **NEW**;
effort **S** · **P1** · Stable; risk: low; source: engine-report correctness section.

**MON-29 · Cross-model speed history** — what: per (model, quant) best/median tok/s from past
sessions; value: buying/quant choice informed by own data; UI: Models screen → per-model "measured
here" row, linked from Monitor; engine: keyed series from MON-21 store; effort **S** · **P2** ·
Stable; risk: low; source: engine.md placement matrix (tuner results feed the site when measured).

**MON-30 · Latency distribution (inter-token latency histogram)** — what: ITL p50/p95/p99 for the
last message; value: average tok/s hides stutters (expert swaps show as tail latency); UI: Monitor →
request detail; engine: sample ITLs in decode loop **NEW**; effort **M** · **P1** · Beta; risk: ring
of per-token timestamps (~8 B/token — negligible); source: engine-report first-turn decode problem.

**MON-31 · Queue depth, active slots, batch size** — what: requests waiting, slots decoding
together, tokens in the current batch; value: explains why a second message waits (interactive-first
queue); UI: Monitor → System row; engine: **NEW** (scheduler counters from engine §6); effort **S** ·
**P2** · Stable; risk: low; source: turbo-system §2 queue policy.

**MON-32 · Machine power state (plan, boost, throttle flags)** — what: active power plan /
processor boost state / GPU throttle flags with % time throttled; value: ties MON-16 to the turbo
profile that changed it; UI: Monitor → System card footer; engine: **NEW** (profile store +
`powercfg`/NVML throttle reasons); effort **S** · **P2** · Stable; risk: only if profile control
ships (TRB-33); source: turbo-system §3.

**MON-33 · Engine self-test status** — what: last kernel reference check, greedy prefix match vs
reference, GGUF validation result, timestamp; value: correctness is a feature users can see; UI:
Monitor → footer status chip / Security screen cross-link; engine: test harness exists
(`greedy_prefix_match`, expert self-test 4e-7 in report) → **NEW** persist last result; effort **S** ·
**P2** · Stable; risk: low; source: engine-report §Method; engine.md §Rules correctness-first.

**MON-34 · Load progress % (weights mapped, experts warm, KV allocated)** — what: real percentages
during model load, not a spinner; value: 17 GB loads take seconds-to-a-minute, users see progress;
UI: overlay + Monitor state card; engine: **NEW** (load progress events from `gguf.rs` mmap + warm
touch loop); effort **M** · **P1** · Stable; risk: progress from file bytes is honest, progress from
"touch pages" is not — report bytes only; source: engine-report "touching CPU-side expert pages at
load" (warm-touch exists as an optimisation, expose its phase).

**MON-35 · Always-on status bar + compact overlay** — what: persistent bottom bar (exists:
`components/StatusBar.tsx`) showing model, tok/s, VRAM %, temp, power %; optional always-on-top
mini-window; value: users watch the machine while doing other work; UI: StatusBar (exists) + new
overlay window; engine: `live_stats()` exists (subset) → reuse `metrics()`; effort **M** · **P1** ·
Stable; risk: overlay is a second webview (Tauri window, `src-tauri/src/commands.rs` window
commands); source: positioning "Show the machine thinking".

## 2. Turbo / optimisation toggles (45)

Every engine optimisation as a user toggle: beginner sentence (shown by default), advanced sentence
(expanded row), safe default, measured-impact badge (before → after on THIS machine with method one
click away), channel gate, auto-revert. TRB-40..45 are the meta-features that make toggles a system.
Gain figures: measured = local number from `2026-10-10-engine-report.md`; estimated = this paper.

**TRB-01 · Tensor-core prefill (mma)** — what: dense-prefill GEMM on warp-level matrix multiply-
accumulate (f16 inputs, f32 accumulate) instead of the current dp4a path; gain: closes the measured
~10× dense-prefill gap (0.6B, 512/2048 tok: 1,317/1,228 → target ≥ 8,000-10,000 tok/s control
class; MoE prefill 200 → est. 350-600 tok/s, confidence **medium-high** — mechanism proven on
sm_86, exact tile efficiency unknown); sketch: new warp-level `mma.m16n8k8`/`m8n8k4` kernels in
`kernels_mma.cu` (file already exists), 64×64×32 tiles, shared-memory double buffering, epilogue
bias+activation in registers, dispatch by shape in `gpu.rs`, per-shape A/B fallback to the dp4a
kernel, reference test vs f32 in `examples/bench.rs`; risk: medium — numeric drift on accumulation
order (tolerance test), register pressure, occupancy cliffs per quant (Q4_K dequant into MMA fragments
is the hard part: dequant in shared memory as fragments are staged); channel **Experimental** →
Beta after a measured win on both machines; toggle (beginner): "Use the graphics card's built-in
matrix units when reading a long prompt, which can make prompt processing much faster."; toggle
(advanced): "Prefill GEMM via mma.sync on sm_86 with per-shape A/B against the dp4a path; disabled
automatically when the measured run loses."; effort **L** · **P0**; source: engine-report "Where it
is behind… about 10×", "Next, in order" #2.

**TRB-02 · Fused per-layer kernels** — what: the 10 fused kernels per layer already measured
(RoPE+attention+ MLP pieces in fewer passes); gain: **measured** 168 → 346 tok/s (0.6B GPU decode,
part of that combined row); sketch: current `kernels.cu` fusion points kept as default, toggle
selects unfused reference path for A/B; risk: low (fusion bugs caught by logits test); channel
**Stable** (default on); toggle: "Combine several small steps of each layer into one, which the
measurements show makes generation faster."; effort **S** · **P0**; source: engine-report "What
moved the numbers" row 1.

**TRB-03 · CUDA graph replay** — what: capture the decode step once and replay it instead of
launching kernels individually; gain: **measured** (part of 168 → 346 tok/s row; launch overhead on
small batches is the dominant term at short context); sketch: `gpu.rs` graph capture for the decode
shape (batch ≤ N, fixed KV pointers via paged-KV stable addresses), invalidate on shape change,
fallback to eager; risk: medium — pointer instability with paged KV/pool resizes forces graph
re-capture (budget ≤1 recapture per request); channel **Stable**; toggle: "Reuse the GPU's launch
plan for every token instead of re-planning it each time, which speeds up generation.";
effort **M** · **P0**; source: engine-report row 1.

**TRB-04 · Sampler on the GPU** — what: greedy/top-k/min-p/logits-softmax picked on-device so the
logit tensor never crosses PCIe; gain: **measured** (included in 168 → 346 row: "greedy pick on the
GPU"); sketch: `sample.rs` → device variant in `kernels.cu` (top-k bitonic per block + argmax),
result is one token id copied back; risk: low for greedy, medium for full sampler parity (test every
sampler against CPU reference); channel **Stable** for greedy, **Beta** for sampling parity; toggle:
"Let the graphics card choose the next word itself, avoiding a slow trip back to the processor.";
effort **M** · **P1**; source: engine-report row 1.

**TRB-05 · Batched prefill** — what: prefill computed in token batches instead of one token at a
time; gain: **measured** (0.6B prefill 135 → 884 tok/s in the same combined row); sketch: existing
batched path in `model.rs` with token-batch dimension, autotuned chunk (see TRB-35); risk: low;
channel **Stable**; toggle: "Read the prompt in batches instead of one word at a time, which makes
waiting for the first reply much shorter."; effort **S** · **P0**; source: engine-report row 2.

**TRB-06 · f16 KV cache** — what: half-precision keys/values (vs f32) for GPU-resident context;
gain: **measured** (part of row 2: decode 83 → 226 tok/s at 2K context, plus 2× KV capacity); sketch:
KV pages allocated f16 in `gpu.rs`/paged pool, attention kernels read f16 with f32 accumulate;
risk: low-medium (quality impact negligible at f16 — but show a KV-quality note, never claim without
measurement); channel **Stable**; toggle: "Store the context in half precision, which fits more
conversation on the graphics card and was measured faster."; effort **M** · **P0**; source:
engine-report row 2; engine.md §5.

**TRB-07 · q8 KV cache** — what: 8-bit KV pages (VRAM headroom for experts/context); gain: est.
2× context capacity per GB at ≤1% decode slowdown for most models (confidence **medium** — must be
measured per model; headroom converts directly into expert residency on this 8 GB card: MON-07);
sketch: new page type in paged pool (`gpu.rs`), dequant-on-load in attention kernel, per-layer
toggle, quality gate = per-model perplexity/logits A/B in Tune; risk: medium — some models degrade
visibly at q8/q4 (measure, do not assume); channel **Beta**; toggle: "Store the context in 8-bit
precision to leave more room for the model's experts; Cerebro measures the quality cost for your
model before keeping it."; effort **M** · **P1**; source: engine.md §5 (q8/q4 KV pages).

**TRB-08 · q4 KV cache (long-context mode)** — what: 4-bit KV pages past a context threshold;
 gain: est. 4× KV density → on this card enables ~2× context at equal VRAM (confidence **medium**);
sketch: same page machinery as TRB-07 with q4 block layout, hybrid pages (q4 cold / f16 hot) with
promotion on reuse (radix tree already tracks access); risk: high — visible quality loss on some
tasks; **Beta→Experimental**, per-model opt-in with measured badge; toggle: "For very long
conversations, compress older context to 4 bits to keep it on the graphics card, measured for your
model first."; effort **L** · **P2**; source: engine.md §5; external: KV-compression tradeoffs
survey UNVERIFIED (no local data).

**TRB-09 · Split-K decode attention** — what: parallelise the attention reduction across more
thread-blocks with a workspace (helps large batch/long ctx decode); gain: **measured** (row 2: decode
at 2K 83 → 226 tok/s alongside f16 KV); sketch: split-K path in `kernels.cu` with atomic/final
reduce pass, chosen by shape heuristic in `gpu.rs`; risk: low; channel **Stable**; toggle: "Split one
attention calculation across more graphics-card cores, which helps at long context."; effort **M** ·
**P1**; source: engine-report row 2.

**TRB-10 · Tiled prefill attention** — what: cache-friendly tiling of the prefill attention (Q×K
tiles in shared memory); gain: **measured** 884 → 1,228 tok/s (0.6B, 2K); sketch: tiled kernel in
`kernels.cu`/`kernels_mma.cu` with causal masking per tile; risk: low; channel **Stable**; toggle:
"Tile the prompt-reading attention so data is reused from fast memory, measured faster on this
machine."; effort **M** · **P0**; source: engine-report row 3.

**TRB-11 · Asynchronous expert swaps (copy stream)** — what: expert loads run on a dedicated CUDA
copy stream, many small transfers per token instead of 16 per 32 tokens; gain: targets the measured
first-turn MoE gap 19-23 → 33-36 tok/s (est. first-turn 19-23 → 27-32, confidence **medium** —
depends on PCIe and hit rate; report lists it as next #1); sketch: `pool.rs` per-expert prefetch
into staging → `cudaMemcpyAsync` on stream 1 with events; decode kernel on stream 0 waits only when
the requested expert is not resident; risk: medium-high — many small copies can saturate PCIe
(MON-15 becomes the diagnostic), and event churn adds overhead at high hit rates (disable when
hit rate > 80%); channel **Experimental** → Beta; toggle: "Start loading the parts of the model you
are about to need while the graphics card keeps working, which speeds up the first reply of a new
conversation."; effort **L** · **P0**; source: engine-report "Next, in order" #1.

**TRB-12 · Shipped starting expert profile** — what: a per-model expert-residency profile built
from mixed prompts, shipped with the model's catalog entry so turn 1 starts warm; gain: targets
first-turn 19-23 → 27-33 tok/s (confidence **medium-high** — prompt-aware placement already showed
41% → 85% hit rate when given the prompt); sketch: profile = JSON sidecar (top-K experts per layer
by frequency), loaded by `pool.rs` at model load, blended with adaptive LRU; risk: low — a bad
profile only costs startup, adaptation dominates within turns; channel **Stable** once profiles are
measured per model; toggle: "Ship a starting set of the most-used parts of each model so the first
message is already fast."; effort **M** · **P0**; source: engine-report "Next, in order" #1.

**TRB-13 · Prompt-aware expert placement** — what: scan the upcoming prompt's token ids and place
the experts the router will need before generation starts; gain: **measured** 23.7 → 55.3 tok/s on a
2,048-token prompt (hit 41% → 85%); sketch: tokenizer-only pass (no model forward) → run the MoE
router's static expert-affinity table (built offline per model from a probe corpus) → `pool.rs`
residency pre-stage; risk: medium — affinity table build cost per model (cache it in the model's
library metadata); channel **Stable**; toggle: "Look at your prompt first and preload the model
parts it will use, which was measured to nearly double speed on long prompts."; effort **M** · **P0**;
source: engine-report "What moved the numbers" (prompt-aware placement row).

**TRB-14 · CPU+GPU expert overlap** — what: experts missing from VRAM are computed on the CPU in
parallel with the GPU's other work (signature technology); gain: converts misses from stalls into
overlap — the difference between 19-23 and "CPU-speed floor"; on this machine CPU expert kernels
37-41 GB/s isolated vs 26-28 GB/s in-step, so true gain is gated by closing TRB-19 (confidence
**medium**); sketch: worker pool (`pool.rs` + rayon) pinned to P-cores (TRB-31), double-buffered
activation exchange via pinned host memory, event-driven merge; risk: medium — the measured in-step
bandwidth gap (unexplained, report §unexplained) means overlap may under-deliver; channel **Beta**;
toggle: "Let the processor compute the model parts that are not on the graphics card at the same
time, instead of waiting."; effort **L** · **P0**; source: engine.md §2 signature #2; engine-report
unexplained gap.

**TRB-15 · Warm-touch expert pages at load** — what: touch every CPU-side expert page when the
model loads so paging faults never happen during decode; gain: **measured** 30.3 → 36.2 tok/s (CPU
MoE); sketch: sequential prefault pass in `gguf.rs`/`pool.rs` load path (already implemented — the
toggle exists to let users trade load time for RSS pressure); risk: low (slower startup, larger
working set); channel **Stable** (default on); toggle: "Read the processor-resident parts of the
model into memory when it loads, so nothing stalls mid-sentence."; effort **S** · **P1**; source:
engine-report row 5.

**TRB-16 · K-quant attention/output weights on the GPU** — what: keep attention and output weights
in K-quant format in VRAM instead of dequantising, freeing room for experts; gain: **measured**
1,467 → 1,765 experts resident (+20%, direct hit-rate lift on an 8 GB card); sketch: quantized matvec
path for those tensors in `kernels.cu` + `quant.rs`; risk: low-medium (slightly slower per-op vs f16
for those tensors — measured net positive already); channel **Stable**; toggle: "Keep the attention
part of the model in its compact stored form so more experts fit on the graphics card — measured
+200 experts here."; effort **M** · **P0**; source: engine-report row 6.

**TRB-17 · Yielding CPU worker pool** — what: worker threads block (not spin) when idle; gain:
**measured** 1.9 → 11.4 tok/s (16-thread MoE CPU path) and it is what keeps the UI responsive
during decode; sketch: pool in `pool.rs` with condvar parking (implemented); risk: low (wakeup
latency — measured already); channel **Stable**; toggle: "Let idle helper threads sleep instead of
burning power, which was measured to make the processor path far faster and keeps your laptop
cool."; effort **S** · **P0**; source: engine-report row 4.

**TRB-18 · Fast CPU paths (parallel router, shared GQA attention, fast exp/RoPE table)** — what:
the CPU decode micro-optimisation bundle; gain: **measured** 9.9 → 13.8 tok/s (30B-A3B CPU); note
still behind control 18.5 — remaining work is TRB-19; sketch: current `model.rs`/`quant.rs` paths;
risk: low; channel **Stable**; toggle: "Use the faster attention and routing paths Cerebro measured
for the processor — this is the reason CPU-only speed improved by 40% on this machine."; effort
**M** · **P0**; source: engine-report row 8.

**TRB-19 · Interleaved weight layout for K-quant GEMV** — what: reorder K-quant blocks in memory so
the CPU decode kernel's working set streams sequentially (closes the 37-41 → 26-28 GB/s in-step
gap); gain: est. CPU decode 13.8 → 16.5-18 tok/s if the gap closes (confidence **medium** — the gap
is real and measured, the fix's coverage is unknown); sketch: layout transform at load in
`gguf.rs`+`quant.rs` (block-interleaved by column tile), GEMV kernel in `kernels`-equivalent CPU
code updated to match, cache-aligned tiles sized to L2 (1.25 MB/P-core on this chip); risk: medium —
layout interacts with mmap sharing and paging (TRB-37); channel **Beta**; toggle: "Lay out the
model's numbers so the processor reads them in one smooth stream, aiming to close the measured
speed gap."; effort **L** · **P1**; source: engine-report unexplained gap + "Next, in order" #4.

**TRB-20 · Expert-batched MoE prefill** — what: prefill groups tokens by expert, then streams
uncached experts, in 512-token chunks; gain: **measured** 33 → 205 tok/s (MoE prefill); sketch:
already implemented (`model.rs` prefill path); risk: low; channel **Stable**; toggle: "Group the
prompt by which model parts it uses so the prompt is read in one smooth pass — measured six times
faster for mixture models."; effort **M** · **P0**; source: engine-report row 7.

**TRB-21 · Chunked prefill with expert prefetch** — what: long prompts processed in large chunks;
next layer's experts stream during the current layer's attention; gain: est. hides most expert-miss
latency in prefill (confidence **medium**; interacts with TRB-11); sketch: chunk loop in `model.rs`,
prefetch issued from `pool.rs` during attention event, chunk size from TRB-35 autotune; risk: medium —
over-prefetch wastes VRAM/policy (cap by budget %); channel **Beta**; toggle: "While the graphics
card reads one part of your prompt, fetch the next part in the background."; effort **L** · **P1**;
source: engine.md §4.

**TRB-22 · Continuous batching** — what: several conversations decode together in one pass,
interactive requests first; gain: est. 1.5-2.5× aggregate throughput at 3+ concurrent requests,
near-1× for one user (confidence **high** mechanism; single-user benefit is latency scheduling not
tok/s); sketch: slot scheduler in `chat.rs` + batch assembly in `model.rs`, paged KV already shared;
risk: medium — fairness/starvation with interactive-first (define max-batch, expose in MON-31);
channel **Beta**; toggle: "Run several conversations together in one pass so nothing waits longer
than it has to."; effort **L** · **P1**; source: engine.md §6.

**TRB-23 · Conversation-state parking** — what: idle conversations' KV moves to RAM and back on
resume; gain: est. keeps 5-10 conversations warm for near-instant resume instead of re-prefill
(re-prefill of 4K tokens ≈ 2-6 s measured-class prefill on this box; confidence **high** arithmetic);
sketch: radix-tree subtree serialize to pinned host buffers, restore on message (engine.md §1);
risk: medium — park/restore time must be async (background stream) or it trades one stall for
another; channel **Stable**; toggle: "Keep unfinished conversations in the computer's memory so
switching back to them is instant."; effort **M** · **P1**; source: engine.md §1; `Turbo.tsx`
already exposes "Keep recent conversations ready".

**TRB-24 · Radix prefix cache (conversation cache)** — what: KV blocks keyed by token prefixes, new
requests reuse the longest cached prefix; gain: **measured** in spirit — report shows reuse rows
(55.3 tok/s at 85% hit vs 23.7 at 41%); externally, prompt caching cuts TTFT 6-31% and cost 41-80%
on providers (arXiv:2601.06007, UNVERIFIED for our engine); our number to prove: repeat-turn prefill
→ near-zero, est. >90% reuse on multi-turn chat (confidence **high** given prefix identity); sketch:
radix tree over paged KV in `chat.rs`/pool (engine.md §1); risk: low (eviction policy ties to TRB-34);
channel **Stable** (default on); toggle: "Reuse what you already computed at the start of a
conversation instead of reading the same prompt again."; effort **L** · **P0**; source: engine.md §1.

**TRB-25 · Message-boundary checkpoints** — what: KV snapshots at message boundaries so an agent
rewriting its last message keeps the history valid; gain: est. saves re-prefill of the whole
conversation on every edit/regenerate (seconds per edit at 4K context; confidence **high**
arithmetic); sketch: checkpoint id + prefix-length marker in radix tree, invalidated from edited
token onward; risk: low; channel **Stable**; toggle: "When a message is edited, keep the work already
done for everything before it."; effort **M** · **P1**; source: engine.md §1; positioning.md
(signature list).

**TRB-26 · Speculation: native MTP head** — what: use the model's own multi-token-prediction head
as drafter where present; gain: est. 1.3-1.8× decode when acceptance ≥ 50-70% (confidence **medium**;
gated on measured acceptance — MON-10); sketch: MTP module forward in `model.rs` (removed from small
UD quants sometimes — detect presence in `gguf.rs`, see Unsloth note that UD quants may drop MTP
module), verify batch in one pass; risk: medium — acceptance varies wildly per content (report says
gate on measured acceptance); channel **Experimental** → Beta; toggle: "Where the model predicts
ahead by itself, check several words at once — only while measurements show it pays."; effort **L** ·
**P1**; source: engine.md §3; positioning (MTP gated on acceptance).

**TRB-27 · Speculation: prompt-lookup drafting** — what: draft tokens by copying from the existing
prompt/history (edits, quoted code, repeated text); gain: est. 1.5-3× on agent/code edits with high
overlap (confidence **medium** — acceptance depends on task mix; near-1.0× elsewhere, so gate per
request); sketch: n-gram/substring lookup over tokenized prompt + short suffix cache in `model.rs`
(draft) + batch verify; risk: low (verify always protects quality); channel **Beta**; toggle: "When
the model quotes or rewrites text you already sent, draft from that text and verify — free speed for
coding edits."; effort **M** · **P1**; source: engine.md §3; external: SuffixDecoding-class
approach (see `docs/FEATURE_CATALOG.md` #55, corpus-built n-gram cache — claims UNVERIFIED).

**TRB-28 · Speculation: small draft model** — what: a tiny local model drafts k tokens, verified in
one batched pass (also the plan for dense models larger than VRAM); gain: est. 1.4-1.9× on dense
decode, and it makes partial-offload verify affordable (batched verification pays offloaded weights
once per several tokens — report "Next" #3; confidence **medium**); sketch: second weight set loaded
on demand (must fit beside main model — on 8 GB this usually means the main model is partly in RAM),
draft loop + verify in `model.rs`, per-request accept stats; risk: high — VRAM pressure can make it
slower net (auto-disable when measured acceptance × k < 1.1); channel **Experimental**; toggle: "Use
a small model to guess the next few words and only check them, if measuring shows it saves time on
your model."; effort **L** · **P1**; source: engine.md §3; engine-report "Next" #3.

**TRB-29 · KV streaming (cold pages to RAM + prefetch)** — what: past a threshold, cold KV pages
move to RAM and prefetch back on reuse; gain: est. removes the hard context-VRAM ceiling — enables
>64K context on this 8 GB card (positioning lists KV streaming past 64K as a signature; confidence
**medium**, cost = PCIe traffic when patterns are bad); sketch: paged pool tier tags + prefetch
scheduler keyed by radix-tree access recency; risk: medium — thrashing (add per-request budget +
MON-15 verdict); channel **Beta**; toggle: "Let very long conversations spill into the computer's
memory and fetch back only what is needed next."; effort **L** · **P1**; source: engine.md §5;
positioning.md signature list.

**TRB-30 · ISA-specific CPU build (AVX2 / AVX-VNNI tiers)** — what: compile kernels for the exact
CPU tier (x86-64-v3 + AVX-VNNI on this i9; NEON/SVE on ARM); gain: est. +10-30% on CPU GEMV/dequant
paths (AVX-VNNI speeds int4 dot-products) — confidence **medium** (no local A/B of VNNI path yet;
report's CPU gap is bandwidth-bound first, so VNNI helps only after TRB-19); sketch: `build.rs`
`target-cpu` feature tiers + runtime CPUID gate in `hw.rs`, per-tier dispatch table; risk: low (ship
baseline + tier builds, pick by measurement — turbo-system's own rule: a custom build ships only if
it wins on this machine); channel **Stable**; toggle: "Compile the processor code for exactly the
instruction set your CPU has, then keep it only if it measures faster here."; effort **M** · **P1**;
source: turbo-system §1; engine.md backend section.

**TRB-31 · Thread count autotune + P/E pinning** — what: measured thread count (fewer often matches
speed at far less heat), performance-core pinning, worker affinity; gain: **measured-class** — the
yielding pool row (1.9 → 11.4) shows thread policy dominates; est. quiet-mode parity within 5% of
max-speed with −10-20 W package power (confidence **medium**); sketch: Tune runner sweeps
{8,12,16,20} threads × {P-only, all} × priorities, stores winner keyed by machine (cerebro-probe +
tune session); risk: low; channel **Stable** (default Balanced = measured winner); toggle: "Pick the
number of processor cores that measured fastest for your machine — fewer cores can be just as fast
and much cooler."; effort **M** · **P0**; source: turbo-system §2.

**TRB-32 · Process priority, timer resolution, background efficiency** — what: priority class,
Windows timer resolution, and an idle mode that stops burning wakeups when no request is running;
gain: est. lower tail latency (first-token jitter −10-30% under load, confidence **low**) and
measurable idle power savings (confidence **medium**: timer resolution 1 ms vs 15.6 ms affects CPU
idle power); sketch: `SetPriorityClass`, `timeBeginPeriod` wrapped in `cerebro-flags`-adjacent app
crate, restored on exit (turbo-system §3 revert rule); risk: low (priority raise needs no admin;
never SYSTEM); channel **Stable**; toggle: "Raise Cerebro's priority while it generates and relax it
when idle, so other programs get the machine back."; effort **S** · **P1**; source: turbo-system §2.

**TRB-33 · Machine power profiles (Max / Balanced / Quiet / Battery / Custom)** — what: coordinated
CPU plan+boost and GPU power/clock/fan targets, switchable live, per model; gain: on this laptop,
80 vs 115 W ≈ 30% more GPU power headroom → est. +10-25% clock-bound throughput in Max vs Battery
(confidence **medium**; note the report's warning: clock offsets on power-limited laptop GPUs can
*lower* speed — every change is benchmarked and auto-reverted); sketch: NVML
`nvmlDeviceSetPowerLimit`/`nvmlDeviceSetClockOffset`-class calls behind an unsafe-RAII guard that
restores prior values on drop/crash, Windows power plan via `powercfg` (app crate `cerebro-probe`/
new `turbo` module); risk: high-touch — vendor ranges only, temps/power limits monitored
(MON-16/22), one-click revert; channel **Beta** (Stable for plan-only, no clock locks); toggle:
"Pick how hard this machine works — Max speed, Balanced, Quiet or Battery — every change is measured
and undone if it does not help."; effort **L** · **P1**; source: turbo-system §3 (verbatim
auto-revert rule).

**TRB-34 · Offload & placement autotune (layers, experts, tensor split, KV home)** — what: tuner
chooses how many layers on GPU, expert residency budget, multi-GPU split, KV placement per model;
gain: **measured-class** — the whole report is placement effects (e.g. expert placement 23.7 → 55.3
tok/s); per model on this box est. +15-40% over naive defaults (confidence **high** — Tune screen
already built for exactly this); sketch: Tune runner (`ui/src/screens/tune/*`, `cerebro-bench`)
drives `model_load` variants, stores winner in library metadata; risk: low (never slower rule —
report a regression as a badge, do not apply); channel **Stable**; toggle: "Measure where each part
of your model should live — graphics card, memory or disk — and keep the fastest arrangement.";
effort **M** · **P0**; source: engine.md placement matrix; turbo-system §2 GPU.

**TRB-35 · Prefill chunk-size autotune** — what: chunk size for prefill chosen by measurement (also
sets prefetch pacing); gain: est. +5-20% prefill on long prompts (confidence **medium** — positioning
lists "autotuned prefill chunking" as an optimisation others lack; report used 512-token chunks for
MoE prefill); sketch: sweep {128,256,512,1024,2048} in Tune, stored per (model,placement); risk: low;
channel **Stable**; toggle: "Choose the prompt-reading batch size by measurement instead of a fixed
guess."; effort **S** · **P1**; source: positioning.md; engine-report MoE prefill row (512-token
chunks).

**TRB-36 · Request queue policy (interactive-first / FIFO / shortest-first)** — what: order of
waiting requests; gain: no single-request tok/s change; est. interactive wait −30-80% under
multi-request load (confidence **high** mechanism); sketch: scheduler in `chat.rs` (Turbo.tsx already
exposes the choice — wire to engine); risk: low (starvation guard = max 2 consecutive background
jobs); channel **Stable**; toggle: "Answer the message you are watching first when several requests
wait."; effort **S** · **P1**; source: turbo-system §2; exists in `Turbo.tsx` UI.

**TRB-37 · SSD streaming placement (mmap, never preloaded)** — what: model/expert weights
memory-mapped from NVMe, faulted in on demand (with TRB-15 warm-touch option); gain: lets 17-40 GB
models run in 32 GB RAM at native NVMe read speed (est. 3-7 GB/s sequential vs 25 GB/s DDR4-class
— so streaming is the floor for big MoE; confidence **high** mechanism, per-model numbers from
placement matrix); sketch: `gguf.rs` mmap + residency tiers in `pool.rs`; risk: medium — Windows
page-fault storms if TRB-15 off (default warm for CPU-resident hot set); channel **Stable**; toggle:
"Load big models straight from the drive as they are needed, so a model larger than your memory can
still run."; effort **M** · **P1**; source: engine.md placement matrix.

**TRB-38 · KV-type autotune (f16 / q8 / q4 per model+context)** — what: tuner picks KV precision by
measuring quality proxy and speed per model; gain: headroom → expert residency (2× KV density at q8
frees VRAM for hundreds of experts: report shows +200 experts ≈ measurable hit-rate gain); confidence
**medium**, quality gate required; sketch: Tune adds KV variants (TRB-06/07/08 kernels) + logits
divergence check; risk: medium (auto-pick only when quality within tolerance; else default f16);
channel **Beta**; toggle: "Choose context precision per model by measuring both speed and quality,
defaulting to the safe option."; effort **M** · **P2**; source: engine.md §5.

**TRB-39 · Idle efficiency mode** — what: when no request runs for N seconds: release GPU clocks to
idle, stop background prefetch, park worker threads; gain: battery/heat (est. −5-15 W system idle
during mixed use, confidence **medium** — mechanism: unlock power state restored by TRB-33); sketch:
idle detector in app supervisor → NVML power-limit restore + pool park; risk: low (wake latency
~10-50 ms on first token — show it honestly); channel **Stable**; toggle: "When nothing is
generating, let the machine cool down; the first word after idle may take a moment longer."; effort
**M** · **P2**; source: turbo-system §2 background efficiency mode.

**TRB-40 · Auto-tune runner (Tune → winners → toggles)** — what: one button runs the benchmark
matrix for the loaded model (placement, threads, chunk, KV, sampler) and applies only winners;
gain: the aggregate of TRB-31/34/35/38 — expected +15-40% over defaults on this box (confidence
**high** — the individual rows are measured); sketch: Tune screens (exists) extended to write toggle
profile; risk: medium (tune time — show ETA, run in idle, cache by (model,machine,build)); channel
**Stable**; toggle: "Let Cerebro measure the fastest safe settings for your model on this machine
and keep only what wins."; effort **L** · **P0**; source: turbo-system §4; `screens/tune/*` exists.

**TRB-41 · Measured-impact badge** — what: every optimisation row shows before → after tok/s on
this machine, with method one click away; gain: turns marketing into evidence (the positioning
requirement: every number measured or labelled illustrative); UI: Turbo toggle rows; data: Tune
results + `runs.jsonl`-style run records; effort **M** · **P0** · Stable; risk: none if missing data
renders as "not measured on this machine yet" (never an invented number); source: CLAIMS.md; design
rule "Flag placeholder numbers".

**TRB-42 · Safe defaults + auto-revert** — what: every toggle change is benchmarked in the
background; if slower or hitting power/temp limits, it reverts and says why; gain: protection of the
core promise ("never slower than the previous milestone"); UI: Turbo rows + toast + MON-22 alert;
effort **M** · **P0** · Stable; risk: medium (revert racing an in-flight generation — apply between
requests); source: turbo-system §3 ("auto-reverted if it does not help").

**TRB-43 · Channel gate per toggle (Stable / Beta / Experimental)** — what: each optimisation
carries a channel; default = Stable only, opt into higher channels globally or per feature; gain:
users get experimental speed without instability (report's first-turn MoE and mma work are not
stable-class yet); UI: Turbo → Channels section (exists) becomes per-row badges + global selector;
effort **M** · **P0** · Stable; risk: low (channel defaults documented per release); source:
turbo-system §1 + `Turbo.tsx` channels section.

**TRB-44 · Two-layer explanations (beginner / advanced)** — what: each toggle shows a plain sentence
by default and the engineering sentence on expand (experience level from positioning controls which
is shown first); gain: beginners can act, advanced users can audit — no jargon in default copy; UI:
every toggle row (`.term` primitive for terms); effort **S** · **P0** · Stable; risk: copy review per
CLAIMS (no other product named); source: positioning.md experience levels; effort-levels.md pattern.

**TRB-45 · Per-model toggle profiles** — what: toggles are stored per (model, quant) as well as
globally — what helps a 0.6B dense model may hurt a 30B MoE; gain: prevents one-size regressions
(e.g. speculation gates differ per model); UI: Turbo header model selector; effort **M** · **P1** ·
Stable; risk: low (show scope: "applies to this model"); source: turbo-system §3 "per model".

## 3. MCP (35)

Protocol baseline: MCP stable spec **2026-07-28** (stateless core, `Mcp-Session-Id` removed,
Streamable HTTP, MRTR, tasks in `io.modelcontextprotocol/tasks` extension, Roots/Sampling/Logging
deprecated) — https://blog.modelcontextprotocol.io/posts/2026-07-28/ and
https://modelcontextprotocol.io/specification/2026-07-28/changelog. Rust client/server: `rmcp` 3.x
(crates.io/crates/rmcp; Tier-2 on day one of the 2026-07-28 release — UNVERIFIED current maturity,
re-check at build time). Official registry: https://registry.modelcontextprotocol.io (preview;
metadata `server.json` → package/URL, DNS-verified namespaces — see
https://modelcontextprotocol.io/registry/about). Security is a first-class section because tool
poisoning / rug-pull / response-injection are documented and successful against undefended agents
(arXiv:2508.14925 MCPTox; arXiv:2601.07395 MCP-ITP; arXiv:2601.17549; arXiv:2508.12538; OWASP
https://owasp.org/www-community/attacks/MCP_Tool_Poisoning; Invariant Labs April 2025 PoC —
exfiltration with no user action; ACL Industry 2026 SHIELDMCP reports 74% → 9% attack success with
runtime interception, <120 ms median overhead — external numbers UNVERIFIED).

**MCP-01 · Built-in filesystem server** — what: in-process server exposing read/write/list/glob over
the workspace folder only (engine.md: "a chosen folder the model can see and work in"); value: the
#1 thing users connect MCP for, zero install, sandbox-consistent; UI: Tools → Servers (new) +
permission rules reuse `screens/Tools.tsx` grid; engine data: none (app crate, `rmcp` server
feature); effort **M** · **P0** · Stable; risk: path escapes — enforce canonicalised roots, deny
`..` and symlink-out (audit test); source: engine.md §Product rules; MCP reference servers list
(https://github.com/modelcontextprotocol/servers).

**MCP-02 · Built-in git server** — what: status/diff/log/branch/commit read-operations plus guarded
mutating ops behind approval; value: coding workspaces get history without shell risk; UI: Tools →
Servers; effort **M** · **P1** · Stable; risk: mutating ops (commit/push) always prompt; never expose
credentials (use `git` via argv in sandbox — existing `cerebro-supervisor` pattern); source:
`docs/FEATURE_CATALOG.md` #8 (per-edit commits) as the native precedent.

**MCP-03 · Built-in shell server** — what: argv-only command execution inside the workspace sandbox
(Job Object, CPU/memory caps, no network by default — Security screen already owns this story);
value: agents need a terminal; parity between built-in tools and MCP is one permission engine; UI:
Tools → Servers + ApprovalDialog (exists); effort **M** · **P0** · Stable (sandbox on by default,
off-able per positioning "protections optional"); risk: highest-risk server — default rule = ask,
never allow-all; source: catalog sandbox section; Security screen (`screens/Security.tsx`).

**MCP-04 · Built-in browser server** — what: navigate/read-DOM/click/screenshot over an isolated
browser profile (Playwright is already a dev dependency of the design system — reuse where licensing
allows; runtime bundles its own browser only on opt-in download); value: "read this page into my
context" is a top MCP use case; UI: Tools → Servers; effort **L** · **P1** · Beta; risk: large
download, untrusted-page injection into context (tool results marked untrusted — catalog #83);
source: fresh--mcp-plugins.md host survey.

**MCP-05 · Built-in search server** — what: web search through a user-chosen provider (network is
opt-in per Settings → Network); value: grounding without sending the whole conversation (send the
query only, show exactly what leaves — PRV-13); UI: Tools → Servers + Chat composer "search" chip;
effort **M** · **P1** · Stable; risk: provider API keys — keychain only (MCP-21); source: Settings
network section exists.

**MCP-06 · Built-in sqlite server** — what: read-only-by-default SQL over user-selected .db files;
value: local data is the point of a local model; UI: Tools → Servers; effort **M** · **P2** ·
Stable; risk: destructive SQL — parse statements, deny write ops unless approved; source: MCP
ecosystem common-server list (registry).

**MCP-07 · Built-in documents server (RAG)** — what: expose the Documents screen index (search,
chunks, citations) to any agent via MCP; value: one index, many consumers (chat, agents, external
hosts); UI: Documents + Tools → Servers; engine data: Documents store (exists screen, `Documents.tsx`);
effort **M** · **P1** · Stable; risk: low; source: engine.md workspaces; catalog RAG items.

**MCP-08 · Built-in memory server** — what: structured long-term memory (entities, notes,
preferences) with trust levels; value: continuity across sessions without re-explaining; UI: new
Memory panel in Chat + Tools → Servers; effort **M** · **P1** · Beta; risk: memory poisoning —
untrusted sources never promote to pinned core without confirmation (catalog #82); source:
catalog security section; fresh--memory-rag.md (2026-09-22, claims UNVERIFIED).

**MCP-09 · Built-in image server** — what: generate/list/read images from the Image Studio pipeline
for agents; value: agents can produce and inspect visuals; UI: Image Studio + Tools → Servers;
effort **M** · **P2** · Beta; risk: prompt-injection via image metadata — strip EXIF, re-encode
(catalog #77); source: engine.md M6 image models.

**MCP-10 · Cerebro as an MCP server** — what: Cerebro itself speaks MCP to *other* hosts: chat
completion, model list, Monitor snapshot, agent start/stop — loopback-only, per-launch key; value:
users already in other MCP-capable tools get Cerebro's local models and telemetry without a bridge;
UI: Settings → Network → "Serve MCP (loopback)"; engine data: existing in-process engine + metrics;
effort **M** · **P1** · Stable (loopback-only, key per launch — same posture as the API gateway);
risk: any exposure beyond loopback needs explicit user action + warning; source: catalog #70-71
pattern (loopback-only + per-launch random key already binding for the HTTP API).

**MCP-11 · Server catalogue (official registry mirror + curated shelf)** — what: browse/search the
official registry feed plus a curated shelf of servers rated for local models; value: discovery
without leaving the app; UI: Tools → Catalogue; engine data: none (network to
registry.modelcontextprotocol.io, cached, offline mode = cached list only); effort **M** · **P1** ·
Stable; risk: registry is preview (breaking changes possible — pin API version, cache schema);
source: https://registry.modelcontextprotocol.io + /registry/about.

**MCP-12 · One-click install** — what: from catalogue row → review card (tools list, requested
permissions, package provenance) → install (npx/uvx/pip/docker or remote URL) into workspace or
global scope; value: the "easy customisation" promise; UI: catalogue → install sheet; effort **M** ·
**P0** · Stable; risk: supply chain — show package name/version/publisher, hash pin where possible,
never auto-update (MCP-32); source: fresh--mcp-plugins.md host install patterns (desktop-extension
precedents); registry install metadata.

**MCP-13 · Per-workspace mcp.json managed by the app** — what: each workspace has an mcp.json the
app owns and edits (engine.md requires exactly this); value: project MCP sets travel with folders;
UI: workspace settings → MCP tab + raw JSON view; effort **M** · **P0** · Stable; risk: hand-edited
JSON drifts from UI — validate on save, show diff; source: engine.md §Product rules ("MCP servers
configured in an mcp.json the app manages").

**MCP-14 · Scope merge (global / workspace / session)** — what: precedence session > workspace >
global, with conflict display; value: users understand which server is active where; UI: MCP tab
scope chips; effort **S** · **P1** · Stable; risk: low; source: host-scope precedents in
fresh--mcp-plugins.md (user/project scopes).

**MCP-15 · Per-tool permission prompts (allow / ask / deny + globs)** — what: every MCP tool call
goes through the existing permission engine (Tools screen rules, ApprovalDialog exists); value: the
model may not act outside policy — MCP tools are not a bypass; UI: Tools → Rules (exists, extend to
MCP tools); effort **M** · **P0** · Stable; risk: prompt fatigue — rules persist "always allow this
tool+args-shape", never "always allow server"; source: catalog #68 permission engine; OWASP MCP
page (least-privilege for tool responses).

**MCP-16 · Tool-call activity log for MCP** — what: every MCP call with server, tool, args hash,
duration, verdict, result size; value: auditability identical to built-in tools (Activity section
exists); UI: Tools → Activity (exists — add MCP rows); effort **S** · **P0** · Stable; risk: args may
contain secrets — hash/redact by rule (PRV-06); source: `screens/Tools.tsx` tool-log section.

**MCP-17 · Server health & supervision** — what: crash restart (bounded), stdio pipe supervision,
timeouts, backoff, last-error visible; value: a bad server never takes the app down; UI: server row
status chips; effort **M** · **P1** · Stable; risk: restart loops — cap 3/10 min then disable with
notice; source: rmcp `TokioChildProcess` client (fresh--mcp-plugins.md §2).

**MCP-18 · Context budget meter for tools** — what: tokens each server's tool schemas + resources
consume vs context window (%); value: local models have small contexts — tool bloat is a real failure
mode (the LM Studio note in fresh research: "local models + Claude-oriented servers can flood
context"); UI: MCP tab + composer warning chip; effort **M** · **P1** · Stable; risk: low — needs
`/tokenize`-equivalent (tokenizer is in-engine); source: fresh--mcp-plugins.md cautions; engine.md
tokenizer.

**MCP-19 · Manifest diffing (rug-pull detection)** — what: hash every server's tool
names/descriptions/schemas at install and on every connect; on change, freeze the server and show the
diff before re-enabling; value: stops post-approval description swaps (documented attack); UI: server
row alert + diff modal; effort **M** · **P0** · Stable; risk: false positives on auto-updating
servers — that is the point (approve updates via MCP-32); source: Invariant Labs April 2025 "rug
pulls"; arXiv:2508.14925 (post-registration mutation).

**MCP-20 · Install-time poisoning scan** — what: static scan of tool descriptions + package metadata
for injection patterns (hidden instructions, "ignore previous", credential-file mentions), plus
optional local-model judge; value: documented attack class with 70%+ success rates against
undefended agents (external, UNVERIFIED); UI: install sheet verdict chip (green/amber/red + reasons);
effort **M** · **P1** · Beta (local-model judge Experimental); risk: scanner is not a guarantee —
copy must say what it does not catch (CLAIMS rule); source: arXiv:2508.14925; arXiv:2601.07395;
arXiv:2508.12538 §III; ACL 2026 SHIELDMCP (runtime+static combo).

**MCP-21 · Secrets in the OS keychain** — what: env-var secrets referenced from mcp.json resolve at
launch from Windows Credential Manager/DPAPI; value: shareable mcp.json without leaking keys; UI:
install sheet secret slots; effort **M** · **P1** · Stable; risk: low; source: desktop-extension
precedent (keychain storage, fresh--mcp-plugins.md §3).

**MCP-22 · Enable/disable per workspace, per agent, per session** — what: fine-grained activation;
value: "browser server for research agent only"; UI: MCP tab toggles + agent recipes; effort **S** ·
**P1** · Stable; risk: low; source: agent section of this paper (AGT-21).

**MCP-23 · MCP prompts & resources in the composer** — what: servers' prompts appear as composer
templates; resources appear as attachable context chips; value: server authors ship reusable flows;
UI: composer "+" menu; effort **M** · **P2** · Beta; risk: prompts are untrusted text (treat as data,
catalog #83); source: server-concepts docs
(https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts).

**MCP-24 · Resource browser (read-only)** — what: list/read server-exposed resources with size and
token cost preview before adding to context; value: control over context flood; UI: composer side
panel; effort **M** · **P2** · Beta; risk: low; source: MCP resources API; MON-18 budget meter.

**MCP-25 · Remote servers (Streamable HTTP)** — what: connect to hosted MCP servers with header/auth
handling, per 2026-07-28 transport; value: enterprise/API servers; UI: add-server sheet with URL
+ auth; effort **M** · **P2** · Beta; risk: egress — default allowlist off, every URL shown in
PRV-13 network log; source: spec 2026-07-28 changelog (routing headers `Mcp-Method`/`Mcp-Name`).

**MCP-26 · Deeplink install (`cerebro://mcp/install?…`)** — what: catalogue links on the web open
the app's install sheet (Tauri deep link; navigation policy blocks foreign *webview* navigation —
this is an OS-level scheme handler, unaffected by `navigation.rs`); value: one click from a list to a
reviewed install; UI: install sheet; effort **M** · **P2** · Stable; risk: scheme hijack — validate
payload against registry metadata, never execute on open; source: host deeplink precedents
(fresh--mcp-plugins.md); `src-tauri/src/navigation.rs` as the security context to preserve.

**MCP-27 · Export / import mcp.json** — what: copy a workspace's MCP set to clipboard/file or from
one; value: team sharing, backups; UI: MCP tab overflow; effort **S** · **P2** · Stable; risk:
secrets must export as `keychain://` references, never plaintext; source: config-file section
pattern (Settings → Configuration file exists).

**MCP-28 · Server sandboxing** — what: each spawned server runs in its own Job Object / restricted
token with CPU, memory, and (default) no network; value: a compromised server cannot roam the
machine; UI: Security screen section (honest "what it does not do" line required); effort **L** ·
**P1** · Stable; risk: some servers need network (http registry ones are remote already) — default
deny, per-server grant with explanation; source: catalog #81 parser-sandbox precedent; security
design (Security screen exists).

**MCP-29 · Per-server usage stats** — what: calls, wall time, tokens consumed, prompts approved;
value: users see which servers earn their context; UI: server row + weekly summary; effort **S** ·
**P2** · Stable; risk: low; source: MON-20 request inspector pattern.

**MCP-30 · Budget caps per server (tokens/day, calls/day)** — what: hard caps with pause-on-reach;
value: a runaway agent cannot drown a small context or the day; UI: server settings; effort **M** ·
**P2** · Stable; risk: false trips — warn at 80%; source: AGT-06 step-budget pattern.

**MCP-31 · Local-model compatibility ratings** — what: catalogue rows show context cost of the tool
set, tested behaviours with small models (does the server's tool description bloat break 8K
contexts?), and a "fits local" badge; value: honest guidance for this audience; UI: catalogue badge;
effort **M** · **P2** · Beta; risk: ratings need a test harness (run server against a scripted local
agent — reuse Tune infrastructure); source: fresh--mcp-plugins.md caution note.

**MCP-32 · Version pinning and reviewed updates** — what: installed servers pin versions; updates
arrive as reviewable changesets (manifest diff + changelog), never auto-applied; value: closes the
rug-pull window in the other direction; UI: server row "update available → Review"; effort **M** ·
**P1** · Stable; risk: security patches need a fast path — flag advisories, one-click update with
diff still shown; source: MCP-19 threat model; registry versioning
(https://modelcontextprotocol.io/registry/versioning).

**MCP-33 · Tasks extension support** — what: long-running MCP tasks (`io.modelcontextprotocol/tasks`
extension with poll/`tasks/update`) shown as background jobs with progress; value: servers can do
long work without holding a request; UI: Monitor → jobs strip + server row; effort **M** · **P2** ·
Experimental (extension is new; UNVERIFIED ecosystem support); risk: spec churn — isolate behind an
adapter; source: spec 2026-07-28 changelog (SEP tasks extension, MRTR SEP-2322).

**MCP-34 · Sampling / elicitation policy (deny by default)** — what: server-initiated sampling
requests are refused unless the user enables per server; value: server-side prompt injection through
sampling is a documented protocol-level hole; UI: server settings + prompt on attempt; effort **S** ·
**P1** · Stable; risk: none (deny is safe; legacy deprecation noted in spec); source:
arXiv:2601.17549 (sampling without origin authentication); spec deprecates Sampling — deny matches
the direction of travel.

**MCP-35 · Provenance & namespace verification** — what: catalogue entries display DNS-verified
publisher namespaces and package hashes from the registry; value: users know who published a server;
UI: catalogue/install badges; effort **S** · **P2** · Stable; risk: registry preview data quality —
show "unverified" honestly; source: registry about (DNS verification).

## 4. Customisation (30)

Grounded in the design system (`design/DESIGN.md`: one accent, tokens.css light-dark pairs, WCAG AA
tested) and the Settings screen that exists (appearance, experience level, protections, network,
config file, first-run re-entry).

**CUS-01 · Theme: dark / light / system** — what: two finished themes + follow-OS, per-island
re-theming; value: equal finish in both (design rule — not an afterthought); UI: Settings →
Appearance (exists); effort **S** · **P0** · Stable; risk: low (contrast tests gate every change);
source: DESIGN.md color roles; `theme-init.js`.

**CUS-02 · High-contrast variant** — what: token overrides raising borders/text separation beyond
AA to AAA-class where possible; value: accessibility for low vision without hacks; UI: Appearance;
effort **M** · **P2** · Stable; risk: must not break accent semantics (status colors keep icon+label
— design rule "color is never the only signal"); source: DESIGN.md accessibility rules; WCAG 2.2
https://www.w3.org/TR/WCAG22/.

**CUS-03 · Density: comfortable / compact** — what: spacing scale shift (4 px base kept; row heights
and paddings tighten); value: pros fit more metrics per screen; UI: Appearance; effort **M** · **P2**
· Stable; risk: hit targets must stay ≥40 px (ACC-09) — compact uses 40 px rows minimum; source:
DESIGN.md spacing (4 px scale).

**CUS-04 · Sidebar & panel layout** — what: collapse (exists), section pinning, Monitor dock side,
message list density; value: users compose their workspace; UI: sidebar toggle (exists) + Settings →
Layout; effort **M** · **P1** · Stable; risk: low; source: Sidebar.tsx exists.

**CUS-05 · Named profiles** — what: bundles of (experience level, channels, toggles, theme, metric
density) saved and switchable ("Work", "Lab", "Guest"); value: one click changes the whole posture;
UI: Settings → Profiles; effort **M** · **P1** · Beta; risk: profile vs per-model overrides
precedence must be explicit; source: positioning experience-level model.

**CUS-06 · Sampler presets ("presets" done right)** — what: named (temperature, top-p, min-p, seed,
effort) bundles applied per message or as default, each showing the measured-impact path via Tune;
value: "code", "creative", "precise" without flag archaeology; UI: composer preset chips + Settings;
effort **M** · **P1** · Stable; risk: never call a preset "defaults" without provenance — CLAIMS
rule ("never inherits a server default"); source: DESIGN.md do/don't row on sampler values.

**CUS-07 · Keybindings** — what: rebindable shortcuts, command palette (exists) as the full
fallback; value: muscle memory; UI: Settings → Keys; effort **M** · **P2** · Stable; risk: collision
detection on save; source: CommandPalette.tsx exists.

**CUS-08 · Prompt templates & snippets** — what: user-defined system prompts/snippets with {variables},
insertable in composer, per-workspace scoped; value: repeatable workflows; UI: composer snippets menu
+ Settings → Templates; effort **M** · **P1** · Stable; risk: templates are plain text (no code
execution); source: effort-levels.md prompt-draft pattern.

**CUS-09 · Per-model settings** — what: per (model, quant): sampler card, context default, offload,
threads, KV type, effort default, chat template overrides; value: every model gets its best life on
this hardware; UI: Models → model detail (exists: `screens/models/`); effort **M** · **P0** ·
Stable; risk: settings sprawl — group by Beginner/Advanced with TRB-44 pattern; source: engine.md
placement matrix; Models screen exists.

**CUS-10 · Plugins API contract** — what: stable manifest (capabilities: tools, screens, hooks,
settings panes), versioned, sandboxed, offline after install; value: third parties extend Cerebro
without forking it; UI: Plugins screen (exists) becomes the store + dev mode; effort **L** · **P1** ·
Beta; risk: API stability burden — ship contract tests with the SDK; source: `lib/plugins.ts` +
Plugins.tsx exist; catalog #60-66 taste of plugin-shaped features.

**CUS-11 · User styles (custom CSS) per theme** — what: opt-in user stylesheet loaded after tokens
(power-user escape hatch); value: total visual control for those who accept the risk; UI: Settings →
Advanced; effort **S** · **P2** · Experimental; risk: breaks contrast guarantees — warn that
accessibility tests do not cover user CSS; source: positioning "options everywhere".

**CUS-12 · Locale & units** — what: UI language + GiB/GB, °C/°F, MT/s choices; value: comprehension;
UI: Settings → Appearance/About; effort **M** · **P2** · Stable (English-only v1, plumbing now);
risk: font subsets must cover the language (design glyph rule); source: DESIGN.md font subsetting
rule.

**CUS-13 · Text size & chat readability** — what: UI scale 90-150%, message width, code font size;
value: accessibility + long-session comfort; UI: Appearance; effort **S** · **P1** · Stable; risk:
layout reflow at 200% zoom (ACC-06) — test; source: ACC section.

**CUS-14 · Motion control** — what: follow-system / reduce / off (design already re-checks reduced
motion live); value: vestibular comfort; UI: Appearance (exists via CSS media query — make it an
explicit setting); effort **S** · **P1** · Stable; risk: low; source: DESIGN.md motion principle 5.

**CUS-15 · Notification preferences** — what: which alerts toast, which only badge, sound on/off,
do-not-disturb windows; value: alerts without nagging; UI: Settings → Notifications; effort **S** ·
**P2** · Stable; risk: low; source: MON-22 alerts.

**CUS-16 · Composer behaviour** — what: send key (Enter vs Ctrl+Enter), streaming render mode,
stop/regenerate placement, per-workspace default effort (effort-levels.md composer control exists in
spec); value: typing ergonomics; UI: Settings → Chat (exists: `screens/settings/ChatSettings.tsx`);
effort **S** · **P1** · Stable; risk: low; source: effort-levels.md mechanics.

**CUS-17 · Full settings export/import** — what: one JSON (exists: Settings → Configuration file)
extended to profiles, templates, MCP scope, keybinding maps; value: backup, migration between
machines (machine-specific parts flagged); UI: Settings → Config file (exists); effort **M** · **P1**
· Stable; risk: never include conversations/memories/keys (existing rule in that section); source:
Settings.tsx config-file section.

**CUS-18 · Workspace defaults & inheritance** — what: global → workspace → session cascade with
visible "where did this value come from"; value: no hidden overrides; UI: settings headers show
scope chips; effort **M** · **P1** · Stable; risk: complexity — show effective value + source
everywhere; source: WSP-04/18.

**CUS-19 · Model naming, favourites, tags, notes** — what: rename local models, star, tag, annotate
(quant quirks, best use); value: library becomes personal; UI: Models list (exists); effort **S** ·
**P1** · Stable; risk: low; source: Models screen exists.

**CUS-20 · Local model folder watch** — what: watch user-chosen folders for GGUF drops and import
automatically with validation; value: "models you already downloaded" just appear (interoperability
is the allowed product-mention case); UI: Models → add folder; effort **M** · **P2** · Stable; risk:
low (validate before listing); source: positioning interoperability exception.

**CUS-21 · Tune winner → named preset** — what: save a Tune result as a toggle profile with the
measured badge attached; value: measurement becomes configuration; UI: Tune → Winner (exists) "Save
as preset"; effort **S** · **P1** · Stable; risk: low; source: screens/tune/Winner.tsx exists.

**CUS-22 · Per-session overrides** — what: temporary per-chat changes (sampler, effort, model) that
never write back unless saved; value: experimentation without profile pollution; UI: chat header
"session settings"; effort **M** · **P1** · Stable; risk: confusion — chip shows "session only";
source: CUS-05 precedence.

**CUS-23 · Slash commands** — what: user-defined `/name` → template or tool invocation; value: power
users live here; UI: composer autocomplete; effort **M** · **P2** · Beta; risk: commands may invoke
tools — they inherit the permission engine; source: CUS-08/MCP-15.

**CUS-24 · Path conventions** — what: configurable roots (models, history, artifacts, exports) with
defaults under LOCALAPPDATA; value: users with small C: drives survive; UI: Settings → Advanced →
Paths; effort **S** · **P2** · Stable; risk: moving paths needs migration helper; source: catalog
shadow-git path pattern (%LOCALAPPDATA%/Cerebro/history/<project-hash>).

**CUS-25 · Startup behaviour** — what: restore last session / open home / autostart with OS /
minimise to tray; value: fits daily routine; UI: Settings → Startup; effort **M** · **P2** · Stable;
risk: autostart + idle mode must respect TRB-39; source: MON-35.

**CUS-26 · Multi-window** — what: detachable Monitor and Chat windows (Tauri multi-webview, same
navigation policy); value: monitor on second screen while coding; UI: window controls; effort **M** ·
**P2** · Beta; risk: every window must obey `navigation.rs` (tests exist as the template); source:
`src-tauri/src/navigation.rs`; DESIGN motion rule (per-window allocators).

**CUS-27 · Always-on-top mini monitor (OSD)** — what: 240×120 window: tok/s, VRAM %, temp, power %;
value: glanceable machine state; UI: from MON-35; effort **M** · **P2** · Beta; risk: low; source:
MON-35.

**CUS-28 · Editable effort-level definitions** — what: rename/rewrite the five levels' instructions,
budgets, passes, or add levels (effort-levels.md requires this); value: users own the verification
contract; UI: Settings → Effort; effort **M** · **P1** · Stable; risk: the answer-budget guard must
stay non-editable-off (report: thinking-eats-budget trap is guarded); source: effort-levels.md
"Everything is editable".

**CUS-29 · Configurable Monitor columns** — what: choose request-table columns, card order, sparkline
metrics; value: each pro's dashboard; UI: Monitor → customise; effort **M** · **P2** · Stable; risk:
defaults must stay CLAIMS-clean; source: MON-21/20.

**CUS-30 · Settings search** — what: full-text search across every setting with scope chips
(what-it-does sentence per setting); value: 250 features need a find; UI: command palette + Settings
header; effort **M** · **P1** · Stable; risk: low; source: CommandPalette keywords infrastructure
exists (`routes.ts` keywords pattern).

## 5. Workspaces (20)

Engine rule (binding): "Separate workspaces for coding, images, audio and video chats; a chosen
folder the model can see and work in; MCP servers configured in an mcp.json the app manages."

**WSP-01 · Workspace kinds (coding / images / audio / video / documents / plain)** — what: first-class
workspace objects with kind-specific defaults (screens, tools, MCP sets); value: the app reshapes
itself per job; UI: workspace switcher in sidebar (new); effort **L** · **P0** · Beta (coding +
documents first); risk: over-abstraction — ship coding+documents fully before audio/video tabs;
source: engine.md §Product rules.

**WSP-02 · Folder binding with visible scope** — what: each workspace binds a root folder; every tool
read/write shows the bound path; value: the model's reach is legible and enforceable; UI: workspace
header path chip; engine data: path canonicalisation in supervisor; effort **M** · **P0** · Stable;
risk: symlink escapes (deny-out — MCP-01 rule); source: engine.md "a chosen folder the model can see".

**WSP-03 · Per-workspace mcp.json** — what: MCP-13 scoped to the workspace object; UI: workspace →
MCP tab; effort **M** · **P0** · Stable; risk: MCP-13; source: engine.md.

**WSP-04 · Per-workspace model + settings** — what: default model, sampler card, effort, turbo
profile (e.g. coding = interactive-first queue, TRB-36); value: right tool per job; UI: workspace
settings; effort **M** · **P1** · Stable; risk: CUS-18 precedence; source: CUS-09/18.

**WSP-05 · Per-workspace system prompt / templates** — what: scoped prompt templates (CUS-08) and
AGENTS.md-style instruction file honoured in that folder; value: project memory; UI: workspace →
Instructions; effort **M** · **P1** · Stable; risk: file on disk may change under the app — show
"file changed, reloaded"; source: catalog compaction items re-inject AGENTS.md from disk (#19).

**WSP-06 · Permission presets per workspace (read-only / build / auto)** — what: one choice sets all
tool rules; value: sane defaults per project risk; UI: workspace → Permissions; effort **M** · **P1**
· Stable; risk: never block features behind a protection — per-positioning, presets can be loosened;
source: catalog #14 plan-mode precedent; positioning protections-optional rule.

**WSP-07 · Workspace switcher & recent list** — what: quick switcher (command palette + sidebar);
value: multi-project users live here; UI: sidebar top; effort **M** · **P1** · Stable; risk: low;
source: CUS-30 palette.

**WSP-08 · Coding workspace extras** — what: repo map panel, diff review surface, terminal, git
status strip; value: the flagship agent surface; UI: coding-kind layout; effort **L** · **P0** ·
Beta; risk: catalog-level scope — reuse coding-agent catalog items as the spec (#1-66); source:
docs/FEATURE_CATALOG.md coding-agent section.

**WSP-09 · Image workspace** — what: generation queue, seed history, asset grid, edit sessions
(prompt + source image); value: Image Studio becomes project-shaped; UI: images screen (exists)
scoped by workspace; effort **M** · **P1** · Beta; risk: model availability (image models are M6 —
until then the workspace organises assets only); source: engine.md M6; ImageStudio.tsx exists.

**WSP-10 · Audio workspace** — what: transcription sessions, TTS drafts, per-file history; value:
local speech is a real local-first win; UI: new audio tab (hidden until an audio model is installed);
effort **M** · **P2** · Experimental; risk: model support lags (M6) — do not promise dates; source:
engine.md M6; fresh--audio.md (2026-09 research, claims UNVERIFIED).

**WSP-11 · Video workspace** — what: clip import, key-frame chat, storyboard outputs; value:
organises video work around local models when they land; UI: hidden tab; effort **L** · **P2** ·
Experimental; risk: none if inert until models exist; source: engine.md §Product rules (video named).

**WSP-12 · Documents workspace** — what: index status, chunk browser, citation-linked answers,
re-index controls; value: trustable local RAG; UI: documents screen (exists) + workspace glue;
effort **M** · **P1** · Stable; risk: stale indexes — show freshness per folder; source: Documents.tsx.

**WSP-13 · Artifacts panel** — what: every output (patch, image, transcript, export) lands in the
workspace artifacts list with provenance (request id); value: work is findable after the chat
scrolls away; UI: workspace right panel; effort **M** · **P2** · Stable; risk: low; source: AGT-09
trajectory pattern.

**WSP-14 · Workspace-scoped search** — what: search across conversations, artifacts, and index of
one workspace; value: project memory; UI: palette scope chip; effort **M** · **P2** · Beta; risk:
low (local index); source: CUS-30.

**WSP-15 · Shared memory scope selector** — what: memories can be workspace-private, user-global, or
shared with explicit links; value: no cross-project contamination by default; UI: memory settings;
effort **M** · **P2** · Beta; risk: MCP-08 poisoning rules apply; source: catalog #82.

**WSP-16 · Workspace templates & export** — what: export a workspace definition (kind, folder
convention, mcp.json, prompts, presets) as a shareable file; value: teams standardise setups; UI:
workspace overflow; effort **M** · **P2** · Stable; risk: paths must relativise; source: CUS-27
export patterns.

**WSP-17 · Disk usage per workspace** — what: artifacts + index + shadow-git history sizes with
purge; value: agent runs can eat tens of GB — visible; UI: workspace settings → Storage; effort **M** ·
**P2** · Stable; risk: purge must confirm deeply (shadow history is the undo story, AGT-13); source:
REL-06 disk guards.

**WSP-18 · Workspace turbo profile binding** — what: which measured profile (TRB-33) and queue
policy applies in this workspace; value: coding gets interactive-first, batch jobs get throughput;
UI: workspace → Performance; effort **S** · **P2** · Stable; risk: low; source: TRB-36.

**WSP-19 · Guest workspace** — what: throwaway workspace: no MCP, no memory writes, no history
persist, optional folder; value: safe experimentation and lending the machine; UI: switcher → "Guest";
effort **M** · **P2** · Beta; risk: leaks via plugins — disable CUS-10 plugins in guest by default;
source: PRV-11.

**WSP-20 · Workspace health strip** — what: one row: bound folder (exists?), MCP servers up/down,
model loaded, permissions preset, index fresh; value: "why isn't my agent working" answered at a
glance; UI: workspace header; effort **S** · **P1** · Stable; risk: low; source: MON/MCP status
chips.

## 6. Agent harness (25)

Spec base: `docs/FEATURE_CATALOG.md` coding-agent section (66 reviewed items — this group is the
product layer over that engine of record). External claims about harness techniques come from the
owner paste `docs/research/harness-techniques-2026-10-09.md` and are **UNVERIFIED** until measured.
Routing literature: RouteLLM arXiv:2406.18665 (up to 3.66× cost reduction at 95% quality on
MT-Bench — API setting, UNVERIFIED locally), FrugalGPT arXiv:2305.05176, routing survey
arXiv:2603.04445, benchmark LLMRouterBench (ACL 2026 Findings — top routers ≤4% accuracy gain over
best single, up to 31.7% cost reduction; several routers fail to beat best-single — use as caution),
router-fragility analysis EACL 2026 (routers often stop routing). For *local-first* routing the
"cost" is user time + cloud credits, and the weak/strong pair is (small local model, large local or
cloud model).

**AGT-01 · Manager/worker agents** — what: a manager agent plans and delegates; worker agents run
read-only exploration or isolated edits with their own contexts; value: long tasks finish without
blowing one context; UI: Chat → agent sidebar (task tree); effort **L** · **P0** · Beta; risk:
complexity — depth 1-2 first, sequential workers before parallel (catalog #24 precedent); source:
catalog #24/#54; engine conversation cache makes slot juggling cheap.

**AGT-02 · Task ledger** — what: persistent per-task checklist (id, status, next action) the model
writes via a tool and the UI renders live; value: progress survives compaction and restarts; UI:
task panel above composer; effort **M** · **P0** · Stable; risk: ledger lies — verify against
filesystem state where possible (AGT-03); source: catalog #16 todo tool.

**AGT-03 · Verifier pass** — what: before "done", run the task's checks (compile, schema, tests the
user enabled); failures feed back with a retry cap; value: local models need external ground truth;
UI: ledger row status + diff review; effort **L** · **P0** · Beta; risk: slow suites — default to
compile+lint, tests opt-in (catalog #7 defaults precedent); source: catalog #7/#41/#56.

**AGT-04 · Local-first model routing** — what: classify → cheap local model first → escalate to
stronger local, then (only if enabled) cloud; value: answers cost nothing most of the time, cloud
credits reserved for hard steps; UI: chat header routing chip ("routed to X because…"); effort **L** ·
**P0** · Beta; risk: router-fragility (EACL 2026) — start with transparent heuristics (task class,
step count, verifier failures) over a learned router, log every decision (AGT-17); source: RouteLLM
mechanics; local-cost framing from arXiv:2603.20895 §4.5 (local deployment cost = tokens + ITL +
TTFT + GPU time).

**AGT-05 · Cloud-credits guard** — what: daily/monthly cloud budget in the app; cloud calls require
budget remaining or explicit per-call approval; value: the "don't burn my API key" feature; UI:
Settings → Routing → budget; effort **M** · **P0** · Stable; risk: none (default budget = 0);
source: AGT-04 economics.

**AGT-06 · Step budget & loop guards** — what: caps on steps, repeated (action,observation) pairs,
A-B alternation, no-diff progress; re-plan once then abort with the patch saved; value: agents stop
hammering; UI: ledger warning states; effort **M** · **P0** · Stable; risk: false trips on
legitimately repetitive work — per-task overrides; source: catalog #28.

**AGT-07 · Diff-before-apply review** — what: every mutation staged, shown as unified/side-by-side
diff, accept/edit/reject per hunk; value: the human stays the editor-in-chief; UI: review surface
(CodeMirror merge, catalog #59); effort **M** · **P0** · Stable; risk: low; source: catalog #18.

**AGT-08 · Sandboxed tool execution** — what: argv-only, Job Object limits, no network by default,
writes confined to workspace; value: agents are powerful without being trusted; UI: Security screen
(exists) + per-call badges; effort **L** · **P0** · Stable (sandbox default on, off-able); risk: the
protections-optional rule means copy must never moralise; source: catalog sandbox section;
positioning §Options everywhere.

**AGT-09 · Trajectory log (JSONL) & replay** — what: every step, tool call, verdict, token count
written to a per-task JSONL exportable for debugging/issue reports; value: debugging agents becomes
evidence-based; UI: task → Export trajectory; effort **M** · **P1** · Stable; risk: contains
prompts/files — redact on export (PRV-06); source: catalog #31 linear transcript.

**AGT-10 · Read-only explore workers** — what: subagents with read/grep/glob only, fresh context,
token-capped summaries returned; value: repo understanding without polluting the main context; UI:
task tree worker rows; effort **M** · **P1** · Beta; risk: summary lossiness — return file:line
anchors so the manager can drill; source: catalog #24.

**AGT-11 · Parallel workers in isolated worktrees** — what: N candidate patches in separate git
worktrees, verified then ranked; value: best-of-N quality without double-writing the user's tree;
UI: task → attempts list; effort **L** · **P2** · Experimental; risk: disk (worktrees of big repos)
— size guard; source: catalog #44/#45.

**AGT-12 · Threshold compaction with state preservation** — what: at ~80% context, summarise with a
fixed schema (goal, decisions, files, errors, next action) and re-inject state files; value: tasks
longer than the context window; UI: compact event in ledger; effort **M** · **P0** · Stable; risk:
summary drift — keep plan/todo/AGENTS.md verbatim from disk (never summarised); source: catalog #19.

**AGT-13 · Checkpoint & rollback** — what: shadow-git checkpoints before mutating batches, one-click
rollback of Cerebro-authored changes only; value: safe undo; UI: task → Restore point; effort **M** ·
**P0** · Stable; risk: HEAD-moved refusal (catalog rule); source: catalog #17.

**AGT-14 · Lifecycle hooks** — what: deterministic events (session start, pre/post tool, pre/post
compact, stop) running user argv commands with JSON stdin, exit 2 = block; value: users wire in
their own policy/tests; UI: workspace → Hooks; effort **M** · **P2** · Stable; risk: hooks are code
execution — user-owned, shown in activity log; source: catalog #15.

**AGT-15 · Plan mode → build mode** — what: enforced read-only phase producing a markdown plan the
user approves before any mutation; value: trust-building default for a local model; UI: composer mode
toggle; effort **M** · **P0** · Stable; risk: low; source: catalog #14.

**AGT-16 · Task cost accounting** — what: per-task: wall time, tokens by model, estimated cloud $,
estimated electricity (kWh via MON-16 power integration × time, labelled estimate); value: local vs
cloud tradeoffs become visible numbers; UI: task summary card; effort **M** · **P2** · Beta; risk:
electricity price assumptions — let the user set their rate, default off; source: CLAIMS (label
estimates).

**AGT-17 · Routing transparency** — what: every routing/escalation decision logged with the signals
that drove it, shown inline; value: trust + debugging (and the cure for router fragility); UI: chip →
explanation popover; effort **M** · **P1** · Stable; risk: none; source: EACL 2026 fragility paper
(understandable failures).

**AGT-18 · User-task evaluation set** — what: users pin 5-20 of their own recurring tasks; a weekly
(or on-demand) run scores success with the verifier; value: "did my setup get better?" answered with
my tasks, not a public benchmark; UI: Settings → Agent → Eval; effort **L** · **P2** · Experimental;
risk: eval flakiness — show pass-rate with n; source: catalog #49-51 benchmark machinery (mini
suites) as the internal base.

**AGT-19 · Crash-resume** — what: on app/engine crash, the ledger + trajectory restore the task
mid-flight with a "resume?" prompt; value: long tasks survive reality; UI: startup dialog; effort
**M** · **P1** · Beta; risk: resuming into a dirty tree — run AGT-13 diff check first; source: REL-01.

**AGT-20 · Observation truncation policy** — what: tool outputs head+tail truncated with elision
markers and per-observation caps, configurable; value: contexts survive noisy tools; UI: workspace →
Agent → output caps; effort **S** · **P1** · Stable; risk: losing the error line in the middle — keep
middle-capture option for stderr; source: catalog #31.

**AGT-21 · Agent recipes** — what: parameterised task templates ("add tests for this module", "review
this diff") with pre-bound tools, model, effort; value: repeatable workflows without prompt
engineering; UI: task launcher; effort **M** · **P1** · Stable; risk: low; source: catalog #60-62.

**AGT-22 · Approval levels per task** — what: watch (approve everything) / default (mutations
approve) / auto-read (read-only tools free) / auto-build (mutations free inside workspace) — chosen
at task start, visible chip; value: explicit trust calibration per task; UI: task header; effort
**M** · **P1** · Stable; risk: auto-build + MCP shell = danger — require MCP-15 rules to allow;
source: catalog #68 permission modes; positioning protections-optional.

**AGT-23 · Bounded context passing between workers** — what: workers pass artifacts (file pointers,
structured facts), not raw logs; value: token efficiency; UI: task tree shows artifact edges; effort
**M** · **P2** · Experimental; risk: information loss — artifact schema per kind (source: UNVERIFIED
harness-techniques paste); source: docs/research/harness-techniques-2026-10-09.md (UNVERIFIED).

**AGT-24 · Failure-log mining ("harness-zero" lite)** — what: weekly local job mines failed
trajectories and proposes prompt/tool/rule patches for user review; value: the harness improves on
the user's own failure mode; UI: Settings → Agent → Suggestions; effort **L** · **P2** ·
Experimental; risk: noisy suggestions — nothing auto-applies; source: harness-techniques paste
(UNVERIFIED).

**AGT-25 · Trajectory distillation to a local fine-tune** — what: export accepted trajectories as
SFT data for local fine-tuning of the user's own small model; value: the dream of a personal model;
UI: Settings → Agent → Export; effort **L** · **P2** · Experimental (far); risk: privacy (local only),
quality (needs tooling we do not own yet); source: harness-techniques paste (UNVERIFIED); mark
**roadmap-only**.

## 7. Model hub (20)

Grounded in existing `hub_search`/`hub_files`/`hub_download` commands and LiveHub screen.

**HUB-01 · Hub search with local-fit sort** — what: search GGUF repos (exists), default sort blends
relevance with fit-to-this-machine; value: users see what runs *here* first; UI: Hub (exists); effort
**M** · **P1** · Stable; risk: fit ≠ quality — show both axes; source: LiveHub.tsx; HUB-02.

**HUB-02 · Fit-to-hardware prediction** — what: given (file size, quant, ctx, placement policy) →
predicted viability on this box: VRAM/RAM/SSD per tier, expected tok/s class, headroom warnings;
value: zero wasted downloads; UI: hub card + Models import; engine data: hw probe (exists) + KV-cost
formula (layers, KV type, ctx) + placement matrix results cache; effort **M** · **P0** · Stable; risk:
predictions must carry an error story — "estimate, measured after load" (CLAIMS label); source:
engine.md placement matrix; MON-07/11 vocabulary.

**HUB-03 · Quant ladder with quality-per-GB ranking** — what: for each model, list available quants
sorted by measured/published quality per GB (UD dynamic ladders, imatrix IQ-quants, standard K-quants)
with KL/benchmark numbers where sourced and labelled; value: the positioning promise ("current,
best-in-class quantisations… by measured quality per GB"); UI: model detail quant table
(QuantTable.tsx exists); effort **M** · **P1** · Stable; risk: third-party numbers must cite source
and label UNVERIFIED/vendor; source: positioning.md audience section; Unsloth Dynamic 3.0 docs
(https://unsloth.ai/docs/basics/dynamic-3.0-ggufs — vendor claims ">10% better top-1 at same size",
treat as vendor); arXiv:2408.15240 (quant survey, cited in repo).

**HUB-04 · UD quants as first-class citizens** — what: Unsloth Dynamic IQ quants natively supported
and preferred where quality-per-GB wins (engine.md: "Signature quantisation: Unsloth Dynamic (UD) IQ
quants, supported natively"); value: best models-per-GB for an 8 GB card; UI: badge "UD" in quant
table; engine data: `iq_tables.rs` exists (IQ dequant tables in engine); effort **M** · **P0** ·
Stable; risk: UD3 may change per-layer schemes — parse dynamically from tensor types, not names;
source: engine.md §Product rules; engine `iq_tables.rs`; Unsloth docs (1-bit retains ~72-77% top-1
on Qwen3.8-27B — vendor claim, UNVERIFIED).

**HUB-05 · Speed prediction per quant on this machine** — what: tok/s class per (model, quant) from
past Tune runs on this box, plus interpolations from the placement model; value: "this Q5 will be
~34 tok/s here" before download; UI: hub card "measured here" or "estimated" chip; effort **M** ·
**P1** · Beta; risk: never fake measured — separate chips; source: MON-27/29 stores.

**HUB-06 · Context budget calculator** — what: slider for context length → KV GB + resulting headroom
% on this card; value: users stop OOM-ing by choosing 256K blindly; UI: model detail; effort **S** ·
**P1** · Stable; risk: formula must match engine allocation (share code with HUB-02); source:
engine.md §5; catalog #9 (explicit context sizing, no silent truncation).

**HUB-07 · Resumable multi-part downloads** — what: parts+progress+pause/resume/cancel (exists:
`hub_download` with parts); value: big files on imperfect networks; UI: Hub downloads; effort **S** ·
**P1** · Stable (exists — polish); risk: low; source: commands.rs:288.

**HUB-08 · Integrity at import** — what: SHA-256 streaming hash, GGUF header/bounds validation,
one-time tensor check capture; value: hostile or corrupt files cannot crash the engine; UI: import
result row; effort **M** · **P0** · Stable; risk: hash cost on 17 GB (~30-60 s NVMe) — run
asynchronously with progress; source: catalog #80; engine.md Rules ("Hostile model files cannot
crash or overrun").

**HUB-09 · Import from disk & other local stores** — what: point at folders; Cerebro reads models
already on the machine (interop is a copy-rule exception); value: no re-downloads; UI: Models →
Import; effort **M** · **P1** · Stable; risk: duplicate files — dedupe by hash (HUB-17); source:
positioning interoperability exception.

**HUB-10 · Model cards (facts that matter)** — what: params, arch, context, licence, multimodal,
reasoning modes, chat template presence, MoE topology (experts/layers/active), file list with sizes;
value: informed choice without leaving the app; UI: model detail; effort **M** · **P1** · Stable;
risk: metadata lies in the wild — prefer GGUF-embedded metadata, verify template on load; source:
hub.rs metadata fetch; catalog #32 preset pattern.

**HUB-11 · Personal ranking** — what: "best on this machine for you" orders by (your measured speed
× your task success), not popularity; value: honest discovery; UI: Hub home section; effort **M** ·
**P2** · Beta; risk: cold start — fall back to fit+quality blend; source: AGT-18 evals.

**HUB-12 · First-run recommendation** — what: after the hardware probe, one recommended model+quant
with the reasoning shown (exists: `onboarding/recommend.ts`); value: beginners never face a list;
UI: onboarding (exists); effort **S** · **P1** · Stable (exists — keep measured-based); risk: stale
recommendations — refresh from hub+measurements; source: onboarding/recommend.ts.

**HUB-13 · Disk guards & cleanup** — what: free-space check before download, duplicate/old-quant
suggestions with one-click free; value: 17 GB surprises; UI: downloads + Models → Storage; effort
**M** · **P1** · Stable; risk: deletion needs confirmation + trash, not unlink; source: WSP-17.

**HUB-14 · Provenance badges** — what: every speed/quality claim on a card carries machine/model/
build or an "illustrative" disclosure; value: CLAIMS compliance at the UI layer; UI: badge component
(`.stat-tile__flag` exists); effort **S** · **P0** · Stable; risk: none; source: DESIGN.md disclosure
token; CLAIMS.md.

**HUB-15 · MoE placement preview** — what: for MoE models: experts resident estimate at chosen quant
+ ctx on this card (report: 1,765 experts resident today at Q4_K_M with K-quant attention); value:
first-turn speed expectations set honestly; UI: model detail; effort **M** · **P2** · Beta; risk:
estimates only until measured (HUB-05); source: engine-report resident-expert row.

**HUB-16 · "+2 GB VRAM" simulator** — what: profiler verdict links to a simulator: with N more GB
VRAM, predicted experts resident / layers offloaded / tok/s delta; value: turns upgrade thoughts into
math (turbo-system's own example: "+700 experts"); UI: verdict card popover; effort **M** · **P2** ·
Beta; risk: model must be labelled illustrative unless backed by placement runs; source:
turbo-system §4 verbatim verdict example.

**HUB-17 · Shard dedupe** — what: content-addressed storage so shared shards across models are stored
once; value: multi-model users halve disk; UI: storage view; effort **M** · **P2** · Experimental;
risk: complexity — start with hardlink detection (same volume); source: HUB-08 hashing.

**HUB-18 · Installed-model update watch** — what: check repo revisions for installed models, show
diffs, never auto-download; value: security/quality updates noticed; UI: Models badges; effort **M** ·
**P2** · Stable (network opt-in); risk: low; source: hub.rs network scope (huggingface only).

**HUB-19 · Offline catalogue mode** — what: hub works from cache with a clear offline chip; value:
Cerebro works fully offline (Settings → Network promises this); UI: chip; effort **S** · **P1** ·
Stable; risk: stale prices/files — timestamp the cache; source: Settings network copy (exists).

**HUB-20 · Model-card sampler application** — what: on load, apply the vendor-recommended sampler
card (e.g. thinking vs instruct temperatures) as the starting preset, shown as explicit values; value:
no more silent bad defaults; UI: load dialog; effort **S** · **P1** · Stable; risk: vendor cards
conflict with user presets — user value wins after first run; source: DESIGN.md sampler-values rule;
HUB-10 metadata.

## 8. Onboarding (15)

Existing: `ui/src/onboarding/` (Onboarding.tsx, ExperienceSteps, autoconfig, recommend, progress).
Binding rules: experience level at install, changeable any time; an "Advanced" escape hatch on every
install step without losing Beginner choices (positioning.md §Audience).

**ONB-01 · Experience-level choice (exists)** — what: Beginner (Cerebro edits configs, hides raw
flags, explains terms) vs Advanced (everything); value: right-sized start; UI: onboarding step 1
(exists); effort **S** · **P0** · Stable; risk: wrong pick — solved by ONB-05; source: positioning.

**ONB-02 · Hardware probe with explanation (exists, extend)** — what: CPU ISA (AVX2/VNNI), cores,
RAM, GPU+VRAM+power limit, NVMe — each shown with what it means for models; value: the machine
becomes legible before the first download; UI: onboarding step + Settings; engine data:
`hw_detect()` exists (`commands.rs:87`, `cerebro-probe`); effort **S** · **P0** · Stable; risk:
driver quirks — label unknowns "not detected"; source: engine.md placement matrix.

**ONB-03 · First model + download (exists, extend)** — what: one recommended model+quant (HUB-12)
with fit prediction shown; value: first success in minutes; UI: onboarding; effort **M** · **P0** ·
Stable; risk: large download on bad network — pause/resume (HUB-07) from minute one; source:
recommend.ts.

**ONB-04 · Auto-configuration with visible reasoning** — what: after probe, show the chosen context,
offload, threads, sampler as explicit values with the honest badge "Defaults were fastest" or the
measured reason; value: trust through transparency (design's own example of the honest outcome);
UI: onboarding summary card; effort **M** · **P0** · Stable; risk: none if never invented (CLAIMS);
source: DESIGN.md "In context" result-card rule.

**ONB-05 · Advanced escape hatch on every step (exists pattern — enforce everywhere)** — what: every
onboarding step shows "Advanced" that expands in place; choices become starting values, nothing lost;
value: experienced users never trapped; UI: every step; effort **S** · **P0** · Stable; risk: none;
source: positioning verbatim.

**ONB-06 · First-run micro-benchmark** — what: after load, a 2-minute silent bench (prefill+decode)
establishes the machine's baseline row (MON-27 origin); value: all later badges have an origin story;
UI: onboarding completion + Monitor; engine data: Tune runner (exists) minimal profile; effort **M** ·
**P1** · Stable; risk: adds setup time — show progress with skip (skip = badges say "not measured");
source: engine-report method (warm-up + 3 timed runs, median).

**ONB-07 · Protections explainer (plain, non-preachy)** — what: each default-on protection with one
line on what it does AND what it does not do; value: positioning requires honest mechanism writing;
UI: onboarding step + Security screen; effort **M** · **P1** · Stable; risk: copy must follow CLAIMS
line-by-line style; source: DESIGN.md security-writing rule; turbo-system §Safety stays optional.

**ONB-08 · Terms & consent (exists)** — what: local processing statement, optional network features
listed off by default; value: informed start; UI: exists (`legal/`); effort **S** · **P0** · Stable;
risk: low; source: onboarding exists.

**ONB-09 · Monitor tour** — what: after first message, a guided highlight of the live metrics (tok/s,
VRAM split, expert hit rate) with the plain-language meanings; value: the signature feature is
discovered, not stumbled on; UI: spotlight overlay (respecting reduced motion); effort **M** · **P1** ·
Stable; risk: annoying — skippable, once; source: positioning "Show the machine thinking".

**ONB-10 · Import from existing local setups** — what: scan for model folders from other local tools
on this machine and offer import (interoperability copy exception); value: zero re-downloading; UI:
onboarding optional step; effort **M** · **P2** · Stable; risk: only read known paths, never write
outside Cerebro dirs; source: positioning interoperability exception; HUB-09.

**ONB-11 · Channel opt-in explanation** — what: one screen explaining Stable/Beta/Experimental with
the global default (Stable) and what opting in means; value: the turbo layer is understandable
before it is switched on; UI: onboarding (Advanced) + Turbo; effort **S** · **P1** · Stable; risk:
none; source: turbo-system.

**ONB-12 · Telemetry choice (default: everything local)** — what: nothing leaves the machine by
default; opt-in crash-report sharing exists as an explicit choice; value: trust; UI: onboarding +
Settings → Network; effort **S** · **P0** · Stable; risk: none; source: Settings network copy
("Cerebro works fully offline").

**ONB-13 · Accessibility check at setup** — what: text size, contrast theme, motion preference
available during setup, not buried later; value: accessibility users are not second-class at first
run; UI: onboarding step; effort **S** · **P2** · Stable; risk: none; source: ACC section.

**ONB-14 · Offline readiness check** — what: after setup, verify the app works with network off
(model loaded, chat works, hub shows cached); value: the promise is demonstrated, not claimed; UI:
onboarding final screen; effort **S** · **P2** · Stable; risk: none; source: Settings network promise.

**ONB-15 · Re-run setup & contextual hints** — what: Settings → First-run setup re-entry (exists)
plus `.term` `?` tooltips everywhere a term first appears (design primitive exists); value: learning
is continuous; UI: exists primitives; effort **S** · **P1** · Stable; risk: low; source: Settings.tsx
setup section; DESIGN.md term rules.

## 9. Reliability (15)

**REL-01 · Crash recovery & session restore** — what: conversations, ledger, downloads resume after
app or engine crash; value: agents run for hours — they must survive; UI: startup "restored" toast;
effort **M** · **P0** · Beta; risk: restoring mid-generation is wrong — resume at last completed
message only; source: AGT-19.

**REL-02 · Engine watchdog** — what: if no token advances for N seconds (N = 5× p99 ITL), abort the
request, capture a diagnostic snapshot (thread states, NVML, last events), restart the engine, offer
the report; value: hung generations stop being force-kills; UI: alert + report card; effort **M** ·
**P0** · Stable; risk: false trips during long prefill — use phase-aware timeouts (MON-02 phase);
source: engine-report; REL-08.

**REL-03 · Auto-revert on any slower/hotter change** — what: TRB-42 extended to every system change
(profiles, priority, thread counts); value: the app can never leave the machine worse; UI: toast with
before/after; effort **M** · **P0** · Stable; risk: measurement noise — revert only if worse than
−5% with n≥3; source: turbo-system §3.

**REL-04 · Settings backup & restore** — what: dated automatic backups (7 daily), one-click restore
diffed; value: config courage; UI: Settings → Config file (exists) + Backup section; effort **M** ·
**P1** · Stable; risk: none; source: CUS-17.

**REL-05 · Signed updates (exists: `update_sig.rs`)** — what: verify update signatures before apply,
staged rollback; value: supply-chain safety for the app itself; UI: silent + About row; effort **M** ·
**P1** · Stable; risk: key management (document rotation); source: src-tauri/src/update_sig.rs exists.

**REL-06 · Disk-full & temp guards** — what: pre-flight space checks (downloads, artifacts, shadow
history), temp sweeps on exit; value: no mysterious write failures mid-task; UI: warnings; effort
**M** · **P1** · Stable; risk: low; source: HUB-13.

**REL-07 · Hostile-file validation (exists as rule)** — what: every GGUF offset/size validated
before mapping; corrupt downloads fail closed with a named tensor; value: engine safety rule
(engine.md Rules); UI: import errors; effort **M** · **P0** · Stable; risk: parser bugs — fuzz the
reader (catalog ASVS/fuzz precedent #78); source: engine.md Rules.

**REL-08 · Benchmark regression gate (dev-facing, user-visible result)** — what: CI and release runs
fail if any tracked benchmark regresses >2% vs previous build on both machines; value: "never slower
than the previous milestone" is enforced, not aspired to; UI: About → build benchmarks view; effort
**M** · **P0** · Stable; risk: noisy benches — median-of-3 as in `examples/bench.rs`; source:
engine.md Rules; runs.jsonl method.

**REL-09 · Health dashboard** — what: last self-tests, watchdog restarts, revert events, download
verifications with timestamps; value: trust at a glance; UI: About or Settings → Health; effort **M** ·
**P2** · Stable; risk: low; source: MON-33.

**REL-10 · Graceful degradation on OOM** — what: on VRAM/RAM pressure: park conversations first,
then suggest lower KV quant/ctx, then offer reload with fewer experts — never a silent crash; value:
users understand limits instead of fearing them; UI: guided dialog; effort **M** · **P0** · Beta;
risk: heuristics must be conservative (act at 90% pressure, MON-22); source: engine.md placements.

**REL-11 · Reproducible run records** — what: every bench/Tune/badge run stores commit, settings,
cache stats as JSONL (runs.jsonl pattern) viewable in-app; value: claims are checkable; UI: badge →
method sheet (TRB-41); effort **M** · **P1** · Stable; risk: low; source: engine-report §Method.

**REL-12 · Backpressure & queueing** — what: requests queue with visible position and estimated
start (interactive-first, TRB-36); value: the UI never freezes and promises are visible; UI: composer
queue chip; effort **M** · **P1** · Stable; risk: estimation wrong under load — show range; source:
MON-31.

**REL-13 · Sleep/hibernate handling** — what: on OS sleep: pause generation, park state, resume
cleanly; no corrupt downloads (resume parts); value: laptops close lids; UI: status chip; effort **M** ·
**P1** · Stable; risk: NVML state after resume — re-init device; source: HUB-07/TRB-23 machinery.

**REL-14 · Single-instance & port hygiene** — what: one app instance per user profile; in-process
engine means no port for chat, but MCP-10 and API servers pick free ports and record them; value: no
confusing failures; UI: focus-existing-instance; effort **S** · **P2** · Stable; risk: low; source:
in-process engine (commands.rs backend address "in-process").

**REL-15 · Diagnostic bundle** — what: one-click redacted support zip: versions, health (REL-09),
last N run records, logs (PII-redacted, PRV-05/06), hardware probe; value: bug reports that can be
actioned; UI: About → Export diagnostics; effort **M** · **P1** · Stable; risk: redaction must be
tested — default deny content (prompts/paths) with opt-in include; source: catalog #73 audit
logging patterns.

## 10. Privacy (15)

**PRV-01 · Offline by default** — what: every network feature (hub, updates, search, remote MCP)
individually opt-in; value: a local harness that actually stays local; UI: Settings → Network
(exists); effort **S** · **P0** · Stable; risk: feature discoverability — hub is the one promoted
exception; source: Settings network copy; positioning.

**PRV-02 · No account, no telemetry by default** — what: no sign-in anywhere; usage telemetry (if
ever offered) is opt-in, aggregate, and documented; value: the default posture; UI: Settings; effort
**S** · **P0** · Stable; risk: none; source: Settings; ONB-12.

**PRV-03 · Per-feature network toggles (exists, enforce)** — what: hub / updates / search / remote
MCP each their own switch with last-connection timestamps shown; value: granular consent; UI:
Settings → Network; effort **S** · **P1** · Stable; risk: none; source: Settings exists.

**PRV-04 · Local-only secrets** — what: API keys and MCP secrets in OS keychain / DPAPI; per-launch
random keys for local endpoints; value: nothing in plaintext config; UI: key slots; effort **M** ·
**P0** · Stable; risk: keychain prompt UX on first use; source: catalog #71; MCP-21.

**PRV-05 · Deterministic PII recognizers for logs/exports** — what: Rust recognizers (Luhn, IBAN,
national IDs, email/phone patterns) applied to logs, diagnostics, exports with per-entity
replace/mask/hash; value: support bundles do not become data leaks; UI: export preview; effort **M** ·
**P1** · Beta; risk: false negatives — copy states scope honestly; source: catalog #75 (spec'd).

**PRV-06 · Redaction-by-default exports** — what: exports contain metrics + hashes; content
(prompts, paths, file contents) only via explicit toggles per export; value: sharing numbers is safe;
UI: export sheet checkboxes; effort **S** · **P0** · Stable; risk: none; source: MON-23.

**PRV-07 · Folder-scope enforcement** — what: tools and MCP servers can only touch bound workspace
folders unless per-call approved; value: the blast radius is a directory; UI: approval dialog shows
the exact path; effort **M** · **P0** · Stable; risk: symlink/hardlink escapes — canonicalise and
deny; source: WSP-02; catalog external-dir guard (#68).

**PRV-08 · Optional conversation encryption at rest** — what: DPAPI-protected store for chat
history and memories; value: shared-machine users get real separation; UI: Settings → Privacy;
effort **M** · **P2** · Experimental; risk: lost key = lost history — export warnings; source:
catalog DPAPI audit-key precedent (#74).

**PRV-09 · Memory trust levels** — what: content from tools/RAG/web is `untrusted` and never
auto-promoted to pinned memory; value: the model's "memory" cannot be poisoned by a webpage; UI:
memory panel trust chips; effort **M** · **P1** · Stable; risk: friction for legitimate saves —
one-click promote; source: catalog #82; MCP-20 threat base.

**PRV-10 · Prompt provenance display** — what: every message shows what was injected around it:
system additions, retrieved chunks (with sources), tool outputs, memories; value: users see the
whole input, not just their sentence; UI: message → "context" popover; effort **M** · **P1** ·
Stable; risk: verbose — collapsed by default with counts; source: catalog #83; effort-levels.md
(added prompts always visible).

**PRV-11 · Incognito session** — what: one switch: no history persisted, no memory writes, no
trajectory mining; value: sensitive asks stay on screen only; UI: composer toggle; effort **M** ·
**P1** · Stable; risk: agent checkpoints still touch disk — disable shadow-git in incognito with
notice; source: WSP-19.

**PRV-12 · Tool-call audit trail** — what: append-only activity (exists: Tools → Activity) with
export; value: accountability for agents; UI: exists; effort **S** · **P1** · Stable; risk: args
redaction (PRV-06); source: Tools.tsx activity section exists.

**PRV-13 · Outbound network ledger** — what: every outbound connection the app makes, with
feature, host, bytes, timestamp — visible in Settings; value: "what left this machine" is answerable
in seconds; UI: Settings → Network → Ledger; effort **M** · **P1** · Beta; risk: completeness —
route all egress through one supervisor layer (app crate); source: positioning; MCP-25.

**PRV-14 · Licence & local compliance notes** — what: per-model licence shown at download; workspace
export includes the licences of models used; value: teams stay compliant; UI: hub card + workspace
export; effort **S** · **P2** · Stable; risk: licence metadata may be missing — show "unknown"; source:
HUB-10.

**PRV-15 · Shred** — what: one-click secure delete of conversations, memories, artifacts, caches,
downloads (passes + unlink); value: clean machine handover; UI: Settings → Privacy → Shred with
scope picker; effort **M** · **P2** · Stable; risk: destructive — typed confirmation + scope preview;
source: WSP-17 storage accounting.

## 11. Accessibility (15)

Design system already commits to WCAG 2.2 AA (measured from rendered pixels by `npm test`),
keyboard reachability, term tooltips, reduced motion, live-value semantics. These features make that
a product.

**ACC-01 · Contrast regression suite (exists — protect it)** — what: pixel-measured contrast tests
for every token pair in both themes, run in CI; value: accessibility cannot rot; UI: none (CI);
effort **S** · **P0** · Stable; risk: none; source: DESIGN.md contrast.js; design package.json test.

**ACC-02 · Full keyboard operation + visible focus** — what: every action reachable, two-tone focus
ring on all `:focus-visible`; value: keyboard users are first-class; UI: everywhere; effort **M** ·
**P0** · Stable (exists as rule — audit per new screen); risk: custom widgets (charts) need roles;
source: DESIGN.md a11y rules.

**ACC-03 · Command palette (exists)** — what: every screen/setting/action by name; value: motor and
cognitive shortcut; UI: exists; effort **S** · **P0** · Stable; risk: low; source: CommandPalette.tsx.

**ACC-04 · Streaming announcements** — what: `aria-live` politeness for generation status, alerts,
and request completion (design warns about busy-region holdback — follow its pattern); value: screen
reader users get the same liveness; UI: status line outside busy regions; effort **M** · **P1** ·
Stable; risk: chatter — announce phase changes, not tokens; source: DESIGN.md aria-busy rule.

**ACC-05 · Reduced motion (exists — setting, CUS-14)** — what: instant final states, no replayed
data animation; value: vestibular comfort; UI: exists CSS + new explicit setting; effort **S** ·
**P1** · Stable; risk: none; source: DESIGN.md motion principles 5/11.

**ACC-06 · Reflow at 200% zoom / 400% narrow** — what: no horizontal scroll of essential content at
200% text zoom (WCAG 1.4.10); value: low-vision users; UI: layout; effort **M** · **P1** · Beta;
risk: dense tables (requests) need card fallback; source: WCAG 2.2 SC 1.4.10.

**ACC-07 · Colour-blind-safe metric palettes** — what: charts use shape+hatch+label, never hue
alone (design rule "color is never the only signal" extended to sparklines/heat maps); value: MON-08
heat maps stay readable; UI: chart components (`Sparkline.tsx`, figures); effort **M** · **P2** ·
Stable; risk: aesthetic pushback — keep design tokens; source: DESIGN.md; CUS-02.

**ACC-08 · Term explanations on first use (exists primitive)** — what: every technical term has a
keyboard/touch-reachable `?` definition; value: beginners welcome without dumbing defaults; UI:
Term.tsx (exists); effort **S** · **P0** · Stable; risk: copy volume — link definitions to About
glossary; source: DESIGN.md term rule; positioning audience.

**ACC-09 · Hit targets 40×40 / 24×24 inline (exists rule)** — what: enforced per component; value:
touch users; UI: components.css; effort **S** · **P1** · Stable; risk: compact density (CUS-03) must
respect it; source: DESIGN.md.

**ACC-10 · Captions & transcripts in audio/video workspaces** — what: every audio output has a
readable transcript view; value: deaf/hard-of-hearing users + searchability; UI: workspace panels;
effort **M** · **P2** · Experimental (model-gated, WSP-10/11); risk: none; source: WSP-10.

**ACC-11 · Forced-colors / Windows High Contrast support** — what: system-forced palettes respected
(border-visible controls); value: OS-level accessibility users are not broken by custom colors; UI:
CSS; effort **M** · **P2** · Beta; risk: accent-only affordances need borders; source: WCAG + design
"color never the only signal".

**ACC-12 · Number+unit typography (exists — protect)** — what: value and unit as one type unit,
monospace figures; value: readouts stay readable and screen-reader-sane (unit announced with value);
UI: design system; effort **S** · **P1** · Stable; risk: none; source: DESIGN.md spacing/type rule.

**ACC-13 · Alert announcements with icons/labels** — what: every status chip has icon+text, never
color alone; live-region announcements for MON-22 alerts; value: same information for all; UI:
StatusPill.tsx (exists); effort **S** · **P1** · Stable; risk: none; source: DESIGN.md status roles.

**ACC-14 · Monitor pause/record shortcuts** — what: keyboard control of history capture and
inspector focus; value: capture moments without mouse gymnastics; UI: Monitor; effort **S** · **P2** ·
Stable; risk: none; source: MON-21/23.

**ACC-15 · Plain-language mode** — what: global toggle: all explanations render the beginner
sentence (TRB-44), abbreviations expanded, jargon terms auto-annotated; value: cognitive accessibility
without a separate "simplified app"; UI: Settings + command palette; effort **M** · **P2** · Beta;
risk: experts want density — reversible per-screen; source: TRB-44; positioning experience levels.

## 12. Sources

Local (binding or measured): `docs/specs/engine.md`; `docs/specs/turbo-system.md`;
`docs/specs/positioning.md`; `docs/specs/effort-levels.md`;
`docs/measurements/engine/2026-10-10-engine-report.md` (+ `runs.jsonl`); `docs/CLAIMS.md`;
`docs/FEATURE_CATALOG.md`; `docs/research/harness-techniques-2026-10-09.md` (claims UNVERIFIED);
`docs/research/2026-09b/sub/fresh--mcp-plugins.md` (2026-09-22 spec/host survey);
`docs/research/2026-09b/sub/fresh--memory-rag.md`; `docs/research/2026-09b/sub/fresh--audio.md`;
app worktree as read 2026-10-16 (`design/DESIGN.md`, `apps/desktop/ui/src/**`,
`apps/desktop/src-tauri/src/{commands.rs,engine.rs,hub.rs,navigation.rs,update_sig.rs}`);
engine worktree `crates/cerebro-engine/src/{kernels.cu,kernels_mma.cu,gpu.rs,model.rs,pool.rs,
chat.rs,gguf.rs,hw.rs,quant.rs,iq_tables.rs,sample.rs}`.

External (checked 2026-10-16): MCP stable spec 2026-07-28 —
https://blog.modelcontextprotocol.io/posts/2026-07-28/ ;
https://modelcontextprotocol.io/specification/2026-07-28/changelog ;
official registry — https://registry.modelcontextprotocol.io , https://modelcontextprotocol.io/registry/about ,
https://modelcontextprotocol.io/registry/versioning ;
Rust SDK `rmcp` — https://crates.io/crates/rmcp , https://github.com/modelcontextprotocol/rust-sdk ;
server concepts — https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts .
MCP security: arXiv:2508.14925 (MCPTox); arXiv:2601.07395 (MCP-ITP);
arXiv:2601.17549 (protocol analysis); arXiv:2508.12538 (systematic analysis);
https://owasp.org/www-community/attacks/MCP_Tool_Poisoning ;
https://labs.invariantlabs.ai (Invariant Labs, April 2025 tool-poisoning disclosure —
https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks );
ACL Industry 2026 SHIELDMCP — https://aclanthology.org/2026.acl-industry.58.pdf (74%→9% ASR claim
UNVERIFIED). Routing: arXiv:2406.18665 (RouteLLM); arXiv:2305.05176 (FrugalGPT);
arXiv:2603.04445 (survey); arXiv:2603.20895 (prefill-activation router, COLM 2026);
https://aclanthology.org/2026.findings-acl.1881.pdf (LLMRouterBench);
https://aclanthology.org/2026.eacl-long.351.pdf (router fragility). Caching:
arXiv:2601.06007 (prompt caching 41-80% cost, 6-31% TTFT on APIs — UNVERIFIED locally);
arXiv:2510.15152; arXiv:2510.09665 (LMCache); arXiv:2410.03065 (Cake);
arXiv:2505.21919; NeurIPS 2025 Learned Prefix Caching
(https://proceedings.neurips.cc/paper_files/paper/2025/file/414f642a1ea9350006669774cba9bcd4-Paper-Conference.pdf).
Quantisation: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs (vendor claims UNVERIFIED);
https://unsloth.ai/blog/dynamic-v2 ; https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/74 .
Observability naming: OpenTelemetry GenAI semantic conventions —
https://opentelemetry.io/docs/specs/semconv/gen-ai/ and
https://github.com/open-telemetry/semantic-conventions (use for export schemas later; not a v1
dependency). Accessibility: W3C WCAG 2.2 — https://www.w3.org/TR/WCAG22/ .

## 13. P0 build list (30, in order)

P0-tagged features in the catalogue exceed 30; the sequence below is the binding first build order
(dependencies first: measurement + safety rails before optimisation switches, switches before the
tuner that writes them, Monitor before badges, MCP substrate before the agent harness). Any feature
tagged P0 above but not listed here is **P0-queued behind item 30** (notably the agent-harness core
AGT-01..08/12/13/15, which needs the MCP/tool substrate and permission engine in items 27-29).

| # | Id | Item | Why it is here (dependency) |
|---|---|---|---|
| 1 | TRB-43 | Channel gate per toggle + global Stable/Beta/Experimental selector | Everything else ships gated |
| 2 | TRB-44 | Two-layer (beginner/advanced) toggle explanations | Copy framework; positioning rule |
| 3 | TRB-42 | Safe defaults + measured auto-revert harness | No optimisation may leave the machine slower |
| 4 | TRB-41 | Measured-impact badge + method sheet (CLAIMS at UI layer) | Evidence layer for every later toggle |
| 5 | MON-16 | GPU power / limit % / temp / clocks / throttle via NVML | Feed for auto-revert + all power features |
| 6 | MON-01 | Decode tok/s live with 10 s average + peak | Headline metric |
| 7 | MON-03 | TTFT per request + session p50/p95 | Latency truth; routing/queue debug |
| 8 | MON-02 | Prefill tok/s + prompt-progress % | Makes mma-prefill work visible (item 20) |
| 9 | MON-04 | Prompt reuse % (request + session) | Prefix-cache evidence |
| 10 | MON-05 | Expert cache hit-rate polish (wire to profile data) | MoE story; gates TRB-11..13 tuning |
| 11 | MON-11 | VRAM split (weights/KV/experts/scratch) + headroom % | Placement + upgrade math |
| 12 | MON-12 | KV fill % of context window | Truncation trust |
| 13 | MON-20 | Per-request inspector completion (null fields) | Every later "why was it slow" |
| 14 | TRB-12 | Shipped per-model expert starting profiles | First-turn MoE fix, report next-#1 |
| 15 | TRB-13 | Prompt-aware expert placement (productise measured win) | 23.7 → 55.3 tok/s measured |
| 16 | TRB-11 | Asynchronous expert swaps on the copy stream (Experimental) | First-turn fix part 2 |
| 17 | TRB-31 | Thread-count / P-E-core / priority autotune | Cheapest large CPU win; Quiet profile base |
| 18 | TRB-34 | Offload & placement autotune (Tune integration) | Per-model speed floor |
| 19 | TRB-40 | One-button auto-tune runner applying only winners | Productises 17-18 |
| 20 | TRB-01 | Tensor-core (mma) prefill kernels (Experimental) | Largest measured gap (dense prefill ~10×) |
| 21 | TRB-24 | Prefix/conversation cache wiring + stats (engine M2) | Multi-turn speed; MON-04 backend |
| 22 | HUB-08 | Integrity at import (SHA-256, GGUF bounds, one-time tensor check) | Hostile-file rule before wider downloads |
| 23 | HUB-02 | Fit-to-hardware prediction on hub cards | Kills wasted downloads |
| 24 | HUB-14 | Provenance badges (measured vs illustrative) | CLAIMS compliance across hub/monitor |
| 25 | CUS-09 | Per-model settings page (sampler/ctx/offload/threads/effort) | Home for tuner output |
| 26 | WSP-02 | Workspace folder binding with visible scope (coding+documents kinds first, WSP-01) | Agent/MCP containment boundary |
| 27 | MCP-13 | Per-workspace mcp.json managed by the app | Engine.md product rule |
| 28 | MCP-15 | MCP tools in the permission engine (allow/ask/deny + globs) | No MCP call without policy |
| 29 | MCP-01 | Built-in filesystem + shell servers under the sandbox (MCP-03) | Agent substrate |
| 30 | MCP-12 | Catalogue + one-click reviewed install (+ MCP-19 manifest diffing as the gate) | Ecosystem surface, gated by 28-29 |

Next queue after 30 (already P0-tagged): AGT-15 plan mode → AGT-02 ledger → AGT-07 diff review →
AGT-08 sandbox exposure → AGT-03 verifier → AGT-01 manager/worker → AGT-12 compaction → AGT-13
checkpoints → AGT-04 local-first routing + AGT-05 credits guard; then the remaining Monitor
P0s (MON-16 follow-ons: MON-13/14/15/34) and HUB-04, TRB-02/03/05/06/10/17/18/20 toggle exposure
(already implemented in-engine — product work is badges + per-model profiles only).

*End of Task C. 270 features. Anything marked UNVERIFIED above must be measured on Machine A
(and Machine B where the feature is hardware-portable) before it appears in public copy
(docs/CLAIMS.md remains binding).*
