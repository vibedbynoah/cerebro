# Cerebro restart (2026-09-22)

The first attempt lives on, untouched, in `C:\Users\noahm\Projects\marrow` (branches `cerebro-build`,
`cerebro-build-bench`, `cerebro-build-ui`). This repository starts over and carries forward only
what earned its place.

## Why restart

Ten days of work produced high-quality, heavily reviewed low-level crates and a very large
research corpus, but nothing a user can run. The cause was ordering, not quality: every module was
hardened to production standard (up to five Opus review rounds each) before any vertical slice
existed, and parallel worktrees drifted apart and duplicated each other. The restart flips the
order: a thin working product first, then hardening driven by what it actually needs.

## Ledger

### Kept code (in `crates/`, all green: 377 tests pass)

| Crate | Source | Why kept | Known issues |
|---|---|---|---|
| `cerebro-flags` | marrow `cerebro-build` | Closed flag allowlist and argv emission: the core safety idea, small and tested. | None. |
| `cerebro-probe` | marrow `cerebro-build` | CPU topology, GPU inventory, backend identity and SHA-256, DACL audit. | DACL audit follows ancestor junctions instead of failing closed (open major finding). |
| `cerebro-supervisor` | marrow `cerebro-build` | Preflight gate and job-contained spawn with an allowlisted environment; lifecycle (readiness, restart policy) is a near-complete draft. | Lifecycle never passed review; one timing-sensitive test fails under full-workspace parallel load. |
| `cerebro-bench` | marrow `cerebro-build-bench` | llama-bench JSON parsing and nonparametric stats (first rep discarded, bootstrap CI, pp/tg/tg-at-depth triple). | None known. |
| `cerebro-store` | marrow `cerebro-build-bench` | Append-only, hash-chained, crash-safe result records keyed by provenance. | Overbuilt for v1 but finished. |

### Kept but parked

| Item | Why parked |
|---|---|
| `crates/cerebro-config` (in tree, not a workspace member) | Settings schema, file handling and DPAPI secrets are needed, but it duplicates the DACL walk and a second spawn gate that belong to probe and supervisor. Rejoins the build after de-duplication. |

### Kept data and docs

| Item | Location |
|---|---|
| Claims policy (binding) | `docs/CLAIMS.md` |
| Architecture thesis (throughput measured, not fitted; speculation as the uncontested axis) | `docs/ARCHITECTURE.md` (to be revised for the product-first plan) |
| Measured baselines | `docs/measurements/` (contended set is provisional; see below) |
| Research corpus: feature catalog (667 decisions), competitors (10 rivals, 86 metrics), eval plan | marrow `docs/research/2026-09/` (reference only; not copied, too large to be a working document) |
| Rival harnesses installed for comparison | `C:\bench`, plus the Qwen3-Coder-30B MoE in `C:\models` |

### Chucked

| Item | Why |
|---|---|
| `cerebro-theme` (2.9k lines of OKLCH colour science) | WebView2 renders CSS `oklch()` natively; a token stylesheet does the job. |
| Wave-2 build plan and 23 micro-briefs | Built bottom-up with no user-visible result; replaced by the slice plan below. |
| Five-round review per micro-module | Spent a usage window per module. Review now happens per slice, with free local first passes. |
| Parallel worktrees on overlapping crates | Caused the duplication above. One repository, one branch per slice. |

## What the measurements already say

Provisional, taken while builds were compiling (quiet re-run pending):

- Qwen3-Coder-30B-A3B MoE: llama.cpp defaults 3.1 tok/s, tuned (`-ngl 999 -ncmoe 32`, q8_0 KV) 17.9 tok/s.
- Gemma-4-E4B (fits in 8 GB VRAM): llama.cpp defaults 80.6 tok/s beat hand-tuned flags at 74.8.

Tuning helps a lot when a model does not fit and can hurt when it does, so Cerebro must measure,
never assume. That is the product's thesis, now with first evidence.

## Plan

**Slice 1: it runs (CLI).** `cerebro models` lists local GGUFs with size and fit. `cerebro tune <model>`
measures llama.cpp's own fit against a few candidate flag sets and stores the winner. `cerebro serve
<model>` launches it through preflight and the supervisor on loopback with an API key and an
OpenAI-compatible endpoint. `cerebro chat` talks to it. Done when a non-developer can go from a
GGUF file to a faster, safe local endpoint with two commands.

**Slice 2: it looks good (desktop).** Tauri shell with model library, one-click tune with live
progress, and a chat screen with a tok/s and VRAM HUD, built to the design bar in the research
(Linear, Raycast, Claude desktop).

**Slice 3: it wins.** The comparison campaign against the 10 rivals, quiet-machine benchmarks,
coding and reasoning suites; every lost metric becomes build work.

Hardening items (config de-duplication, DACL ancestor fix, lifecycle review, flaky test) are
pulled into whichever slice first needs them.

## Process

- Builders: Sonnet 5. Critics and QA: Opus 5.5, to 10/10 with zero findings, judged functionally
  (tests always run) and, for UI, visually from headless screenshots.
- Free local first passes before any paid review: Gemma-4-E4B rates code and screenshots
  (`tools/local/local_review.py`); Qwen3-Coder-30B drafts low-stakes boilerplate
  (`tools/local/qwen_task.py`), which the builder then verifies.
- Incoming boilerplate and research from the owner is triaged against this ledger before anything
  is merged.
