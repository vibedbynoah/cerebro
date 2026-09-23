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

## Owner research report triage (Downloads/harness_research_report.html, 2026-09-22)

A 12 KB summary (17 rated features, a sample prompt, two reviewer notes); the 275 MB of crawled
source data it describes was not included, and no sources are cited, so its ratings are not used.

- Kept: auto-research loop (becomes a daily scheduled free-model research run, not every 30 min,
  because the free tier rate-limits); multi-path self-consistency, reflexion and plan-and-execute
  (folded into the Effort dial's High to Max levels); persistent cross-session memory (core app
  feature); reviewer notes (explain jargon for non-technical readers, real CLI commands in code
  blocks, heading tiers, spacing rhythm, mobile menu, narrow-screen tables) as critic rules.
- Chucked: training-time methods (Constitutional AI, DPO, ORPO, RLVR, federated learning) because
  Cerebro runs models and does not train them; model architectures (MoE, SSM, RetNet) because the
  harness runs whatever llama.cpp supports; FlashAttention 3 (Hopper-only); FP8/AWQ/GPTQ (not
  accelerated on Ampere / not GGUF); neuromorphic, quantum-inspired, neuro-symbolic and causal
  engines (not implementable); the sample prompt ("4x+ throughput", "Revolutionary") and the
  self-assigned 96/100 because they violate docs/CLAIMS.md; testimonials until real users exist,
  because placeholder quotes read as fabricated reviews.

## Product requirements added 2026-09-22 (owner)

- Fully self-contained and offline (no Wi-Fi needed); plugins optional.
- One-click model downloads from Hugging Face (Qwen, Google, Meta, Mistral, DeepSeek, ...) like
  LM Studio: users press Download, never browse the hub.
- Plugin and MCP marketplace with clean icons and one-click install (image generation, web search,
  files, git, ...), installable from a local file for offline use.
- Tools with Claude Code / Codex style permission prompts (ask, session, always, deny), editable.
- Everything user-editable, including switching off the sandbox and integrity checks. Safe by
  default; the user decides. This overrides the parked cerebro-config design that made an unsafe
  DACL override unrepresentable: it must become a user setting.
- Beginner-friendly, with daily feature discovery (spotlight, What's New, checklist).
- Speed and performance are core values alongside compatibility.
- Brief answers first by default (low upfront verbosity), with a user verbosity setting.
- Every model load parameter is a user option (tuned values are the default, with presets and reset).
- Model loading: DSH-style one-click simplicity plus an LM Studio-style load panel for text models;
  a better dedicated path for vision (auto-paired mmproj) and audio (separate STT/TTS engines).
- Effort dial (Low to Max) that sets thinking budget and verification rigor together, user-chosen
  and editable: see docs/specs/effort-levels.md.
- Support every model type natively, not via plugins: text, coding, vision, image generation
  (text-to-image, img2img, inpainting, upscaling; FLUX, SDXL, SD 3.5, Qwen-Image incl. GGUF),
  speech-to-text, text-to-speech, embeddings. The owner specifically dislikes that LM Studio and
  DSH lack image generation. Candidate backends: stable-diffusion.cpp, ComfyUI (installed locally),
  whisper.cpp, Kokoro/Piper.

## Process

- Builders: Sonnet 5. Critics and QA: Opus 5.5, to 10/10 with zero findings, judged functionally
  (tests always run) and, for UI, visually from headless screenshots.
- Free local first passes before any paid review: Gemma-4-E4B rates code and screenshots
  (`tools/local/local_review.py`); Qwen3-Coder-30B drafts low-stakes boilerplate
  (`tools/local/qwen_task.py`), which the builder then verifies.
- Incoming boilerplate and research from the owner is triaged against this ledger before anything
  is merged.
