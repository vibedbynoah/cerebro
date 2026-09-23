# Cerebro autopilot (restart era)

## Running

| Job | Script | Run id | Resume args | Started |
|---|---|---|---|---|
| Front ends: design system, then website + desktop app in parallel | tools/build/frontend.workflow.js | wf_84912725-b06 | {"skip": [ids of tasks already committed]} | 2026-09-22 23:4x |
| Engine wave A (worktree cerebro-core, branch core): hardening, config, bench-run, models, hardware-backends, tune, server, hub, cli | tools/build/core.workflow.js | wf_932a2a40-60a | {"skip": [ids already committed]} | 2026-09-23 00:1x |
| Fresh research (free models, detached python, PID in docs/research/2026-09b/driver.out) | tools/research/run_research.py | - | rerun the script; it skips finished subtasks | 2026-09-23 00:0x |

Worktrees: `C:\Users\noahm\Projects\cerebro` (main, design system), `cerebro-site` (branch site),
`cerebro-app` (branch app). Merge site and app into main when both tracks finish.

## Rules

- Models: builders `claude-sonnet-5`, critics `claude-opus-5-5`. Never plain `opus`.
- Pass = 10/10 with zero findings (owner rule), max 8 rounds, then commit with open findings logged.
- Free first passes on local Gemma-4-E4B (`tools/local/local_review.py`), boilerplate on local Qwen
  (`tools/local/qwen_task.py`). Local server stops itself after 15 idle minutes.
- A halted run (usage limit) resumes with its run id; add committed task ids to `skip`.
- Never stop a healthy run just to apply script edits (owner rule: no wasted tokens). Edits wait for
  the next natural halt; resume then replays finished agents from the journal at no cost. Pending
  edit as of 2026-09-22: expanded data/performance sections in site-home and site-performance.
- A run with no agent-*.jsonl writes for 20+ minutes, or none since its last limit failure, is dead:
  stop it and resume.

## Direction (owner, 2026-09-23)

The product should be HUGE: hundreds to thousands of features (think ComfyUI's breadth), fully
self-contained (downloads, runtimes, model configs all in-app), every OS and hardware. Build the
definitive Windows app first and test it on Windows; cross-platform conversion comes after. The
catalog (docs/FEATURE_CATALOG.md, 627 features: 504 reviewed + 123 added) is the backlog; later
waves implement it section by section in both the engine and the app.

## Queued after the front ends

1. Red-team: owner's abliterated local model attacks the website and the desktop app (owner to
   name the model); findings become fix tasks.
2. Slice 1 CLI (models, tune, serve, chat) on the kept crates; then wire the app's mock layer to it.
3. Triage the owner's incoming research and boilerplate against docs/RESTART.md.
4. Daily auto-research: schedule tools/research/run_research.py-style free-model runs (from the marrow archive) that scan arXiv, Reddit and vendor blogs for new local-inference features and write a dated report for review (owner's research report, 2026-09-22).
5. Wave B onward: turn FEATURE_CATALOG.md sections into engine + app task batches (image generation, voice, plugins/MCP host, permissions engine, memory, RAG, workflow builder, OS-wide quick actions, Tweak Studio backend), Windows first; then macOS and Linux conversion.
