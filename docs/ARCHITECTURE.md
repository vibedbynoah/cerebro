# Cerebro — Locked Architecture

Status: locked 2026-09-12. Derived from six verified recon dossiers against llama.cpp b10751
(commit 3466812d1) on the reference machine. Every claim here is either verified against the local
binaries or explicitly marked UNVERIFIED.

## 1. Thesis

Cerebro is a **throughput-objective configuration engine** for local LLM inference.

It does not implement inference. llama.cpp is the compute backend and remains so.

The precise gap Cerebro fills:

> llama.cpp's built-in autofitter (`common/fit.cpp`, `--fit`, on by default) maximizes offload
> subject to a **memory** constraint. It never measures a token. The memory-optimal configuration
> and the throughput-optimal configuration are frequently not the same point — the reference
> machine has a measured case of 2.22 tok/s vs 29 tok/s for the same model, where the difference
> was entirely flag selection that a memory-fitter has no reason to find.

Cerebro substitutes a **measured objective function** for the fitted one.

### The differentiator, stated narrowly enough to survive a reviewer

An earlier draft of this document claimed Cerebro "searches the axes that no existing tool searches
at all." That is false and trivially falsifiable, and it is corrected here rather than softened.

`llama-optimus` (Optuna search over `llama-bench`) and `TurboLLM auto-tune` already perform
throughput-objective tuning over `-ngl`, `-ncmoe`, batch size and context. **Axes 2, 3 and 4 below
are contested.** Cerebro competes there on packaging, Windows-awareness, sustained-load verification
and quality gating — not on originality.

**Axis 1 — speculation — is uncontested, and it is uncontested structurally rather than by
neglect.** `tools/llama-bench/llama-bench.cpp` contains zero occurrences of `spec_type`,
`model_draft`, or `n_draft`. Both competing tuners are `llama-bench` wrappers, so they are not
declining to tune speculation; they cannot reach it with the harness they are built on. Any tool
that wants this axis must drive `llama-server` and own a second measurement path.

The honest one-line positioning is therefore **not** "the tool that measures instead of fits". It is
"the only tuner that can reach the one axis the standard benchmark harness cannot express."

## 2. What Cerebro deliberately does not build

These were candidate components, eliminated by recon because upstream already does them. Building
them would be waste that loses to upstream on its next release.

| Not built | Because |
|---|---|
| Memory-fit planner | `common/fit.cpp` does context reduction, per-device layer counts, MoE-to-host probing, and sub-layer ATTN/GATE/UP splitting. Already better than a reimplementation would be. |
| Benchmark runner **for axes 2-4** | `llama-bench` sweeps with range/step/mult syntax over `-ncmoe -ctk -ctv -t -C --poll -ngl -b -ub -ot -dev -ts` and emits csv/json/jsonl/sql. Cerebro drives it, does not replace it. **Carve-out: `llama-bench` cannot express Axis 1 at all** — zero `spec_type`/`model_draft`/`n_draft` occurrences in its source — so Cerebro owns a second, `llama-server`-driven harness for speculation. See §6. |
| Model swapper | `--models-dir`, `--models-preset`, `--models-max`, `--models-autoload` absorbed llama-swap's function upstream. Cerebro emits a preset INI. |
| Sampler defaults | `min_p 0.05` plus the penalties/dry/top_n_sigma chain are upstream defaults. Claiming credit here is a false differentiator to an audience that reads release notes. |
| KV cache features | `-ctk/-ctv`, windowed SWA by default, `-ctxcp`, `-kvu`, `-cram`, `--cache-reuse` all ship. Cerebro **chooses** KV settings by measurement; it does not implement them. |
| An inference kernel | Non-goal, permanently. |

## 3. The four original feature ideas, adjudicated

| # | Original idea | Verdict | What ships instead |
|---|---|---|---|
| A | Offload compressed state to the NPU | **Impossible.** i9-12900H is Alder Lake; Intel NPUs begin at Meteor Lake. No NPU exists. No NPU runtime performs live KV compression on any hardware. | Heterogeneous device assignment across the devices that *do* exist: dGPU, CPU-as-RPC-device, E-cores, NVMe. |
| B | Mixture-of-Depths token skipping | **Impossible to retrofit.** MoD is a train-time architecture; router weights exist in no GGUF. Retrofitting layer-skip is depth pruning, measured on this machine to destroy model quality. | **Speculative decoding.** Output-distribution-preserving (see §7 — *not* bit-exact, and the difference matters), 10 strategies plus off, complete flag surface, unreachable by any llama-bench-based tuner. This becomes the product's core. |
| C | Convert KV cache to linear-attention state at inference | **Impossible** without a distillation training run (MOHAWK-class, GPU-hours). | Measured selection across the 9 real KV types, flash-attention modes, SWA, and context checkpoints. |
| D | K-means collapse of 50 KV blocks into 1 centroid | Destroys positional information. | Measured `-ctk/-ctv` type selection gated on numerical equivalence. |

The honest engine is faster than the proposed one, because speculative decoding works and
retrofitted MoD does not.

## 4. Search space

The axes Cerebro searches, in priority order by (unclaimed value × measured leverage).

### Axis 0 — Sampler entropy (searched *first*, because it conditions Axis 1)

This axis is not optional and is not a quality preference. It is a correctness precondition for
every speculation measurement.

Because llama.cpp verifies by **sample-then-compare** rather than by rejection sampling (§7), the
acceptance rate is a direct function of how peaked the target's output distribution is. Under greedy
decoding the target's next token is deterministic and a correct draft is accepted nearly always; as
temperature and the truncation samplers widen the distribution, the probability that the target
happens to sample exactly the drafted token falls, and the speculative speedup falls with it.

Consequence: **a speculative throughput number measured at one sampler setting does not transfer to
another.** Every Axis 1 result record must carry `--temp`, `--top-k`, `--top-p` and `--min-p`, and
Cerebro must tune speculation *per sampler profile* rather than once per model. A tuner that reports
"speculation gave 1.8x" without stating the sampler it measured under is reporting noise.

Cerebro ships two sampler profiles by default — a deterministic profile for code and extraction, and
a creative profile at upstream defaults — and tunes Axis 1 separately under each.

### Axis 1 — Speculation (primary differentiator)

`--spec-type` accepts **ten strategies plus `none`** (eleven enum members total, `common.h:171-182`;
stated this way because "eleven strategies" invites a reviewer to find the off-by-one):

    none
    draft-simple  draft-eagle3  draft-mtp  draft-dflash  draft-dspark
    ngram-simple  ngram-map-k   ngram-map-k4v  ngram-mod  ngram-cache

Two properties make this the core of the product:

1. The `ngram-*` family requires **no draft model**. It speculates from context alone, so
   speculative decoding is available to a user with exactly one model and no vocab-compatible pair.
   Every other tool in the category treats speculation as requiring a curated model pair.
2. `llama-bench` has **no speculative flags at all**. This axis is unmeasurable with the standard
   harness, which is precisely why nothing automates it.

Sub-parameters per strategy: `--spec-draft-n-max` (default 3), `--spec-draft-n-min`,
`--spec-draft-p-min`, `--spec-draft-p-split`, `--spec-draft-backend-sampling`,
`--spec-ngram-simple-size-n`, `--spec-ngram-simple-size-m`, `--spec-ngram-simple-min-hits`,
`--spec-ngram-map-k-size-m`, `--spec-ngram-map-k4v-size-m`, `--spec-ngram-mod-n-min`,
`--spec-ngram-mod-n-max`, `--spec-ngram-mod-n-match`.

Draft placement is independently tunable: `-devd`, `-ngld`, `-Cd` (draft CPU mask),
`--cpu-strict-draft`, `-otd`, `-ncmoed`, `-ctkd`, `-ctvd`, `--prio-draft`.

**Draft placement applies to only five of the ten strategies.** The `ngram-*` family uses no draft
model, so the entire draft-placement flag family and the loopback-RPC apparatus of §5 are inert for
it. This drives search *sequencing*, not just coverage: the n-gram strategies need zero additional
VRAM, zero additional download, no vocabulary-compatible pair, and no RPC worker, and they apply to
every model in a user's library. They are therefore searched **first**, and a user with exactly one
model still gets a tuned speculation result.

#### Making Axis 1 tractable

The search is large and each measurement requires a live `llama-server`. Two mechanisms bound it:

- `--spec-synth-len` and `--spec-synth-rates` synthesize acceptance behavior, letting Cerebro compute
  **this machine's break-even acceptance rate** — the rate at which drafting stops paying for the
  verification batch — before spending anything on real draft models. Strategies whose plausible
  acceptance rate cannot clear break-even are pruned without ever being run.
- Acceptance rate is measured from the live server and fed back, so draft length (`--spec-draft-n-max`)
  is tuned against observed behavior rather than swept blindly, and speculation is auto-disabled when
  it is net-negative.

### Axis 2 — Offload plan

`-ngl` (offloads the **top** N layers; the output layer counts as one, so the VRAM model uses
`n_layer + 1` slots indexed from the end), `-ncmoe`, `-ot`, `-lm/--load-mode` over
`{auto, mmap, mlock, mmap+mlock, none, dio}`.

**`-ncffn` / `--n-cpu-ffn` is mandatory here, not optional.** It offloads dense FFN weights, and
without it *every dense model has no tunable offload boundary at all* — `-ncmoe` applies only to MoE
routed experts. The reference library's dense 27B is unreachable by Axis 2 without this flag. This
was the single largest hole in the original axis definition.

Also searched, all natively sweepable by `llama-bench` at zero marginal cost: `-b`/`-ub` (the
primary prefill and time-to-first-token lever), `-nkvo`/`--no-kv-offload` (directly moves the
KV-versus-weights VRAM boundary, which is *the* binding constraint on an 8 GB card), and `-ts`/`-sm`
(without which §5's RPC worker is given work but no tuned proportion).

**Explicitly rejected, recorded rather than omitted:** `--moe-expert-cache-size`. A controlled sweep
on the reference machine measured it slower than no cache at every size. The mechanism generalizes —
at batch 1, a handful of cache slots against the full expert table cannot reach a hit rate that
amortizes the bookkeeping. Recorded here so it is not rediscovered as a promising idea.

### Axis 3 — KV and context

`-ctk`/`-ctv` over the nine legal types `{f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1}`
(default f16), `-fa {on,off,auto}`, `-c`, `-ctxcp`, `--checkpoint-min-step` (checkpoint memory cost
is the product of count and spacing, and on 8 GB that product is what binds), `--cache-reuse`,
`-cram`/`--cache-ram`, and `--swa-full`.

**`-fa` is a prerequisite, not an independent toggle.** Quantized KV with flash attention off forces
per-step dequantization and is frequently slower than plain `f16`. Treating the two as independent
generates a large block of configurations that cannot win, and under the two-tier protocol every one
of them costs screening time. The constraint is encoded in the search space: non-`f16` KV types are
only paired with `-fa on`.

**`--swa-full` carries a silent-regression risk.** On sliding-window models, losing prompt-cache
reuse to SWA pruning is invisible in a depth-0 benchmark and highly visible in multi-turn use. It is
evaluated against the depth-D and cache-reuse metrics, never against depth 0 alone.

`-cram`/`--cache-ram` and `--cache-reuse` are lossless wins that exploit the reference machine's
memory asymmetry (8 GB VRAM against 31.7 GB RAM) and are free to search.

`--slot-save-path` plus `POST /slots/{id}?action=save|restore` turns prefill of a fixed large context
— a codebase, a long system prompt — into a one-time cost. Lossless, and the substrate for a
named-context feature.

**RoPE scaling** (`--rope-scaling`, `--rope-scale`, `--yarn-orig-ctx`, `--yarn-ext-factor`) is
included for completeness of the context story but is **excluded from the optimizer's objective**.
Beyond roughly 2-4x native training context it degrades quality in ways throughput cannot see, and
no fine-tune is available locally to repair it. The optimizer may not select it to win a benchmark.

### Axis 4 — CPU placement

`-t`, `--cpu-mask`, `--cpu-strict 1`, `--poll`, `--prio`.

Two P-core masks are searched as **distinct candidates**, because they are not equivalent for a
memory-bandwidth-bound workload: `0xFFF` (`-t 12`, all P-core logical processors including SMT
siblings) and `0x555` (`-t 6`, one thread per physical P-core). E-core mask is `0xFF000`.

Masks are derived at runtime from `GetSystemCpuSetInformation` grouped by `EfficiencyClass`, taking
the highest class as P-cores. Never hardcoded.

`--numa` is pruned entirely on single-node systems. The reference machine has one node, so the axis
cannot move and must not consume soak-test hours.

## 5. Heterogeneous device assignment

Cerebro's device model is **two-tier**, and conflating the tiers is a bug class:

- **Physical inventory** — from `vulkaninfo`, `nvidia-smi`, CPU topology. What silicon exists.
- **Addressable set** — from *that specific binary's* `--list-devices`. What the backend can target.

These sets differ. On the reference machine the physical inventory contains an Intel Iris Xe that
the addressable set does not, because the local build ships no `ggml-vulkan.dll`. Cerebro must never
emit a device string absent from the addressable set of the binary it is about to spawn.

### Device assignment available today (CUDA + RPC + CPU backend)

| Role | Placement | Mechanism |
|---|---|---|
| Target model, attention + KV | dGPU | `-ngl 999` |
| MoE routed experts | System RAM via CPU | `-ncmoe N` (deliberately spares `_shexp` and `ffn_gate_inp` — shared expert and router are small, hot every token, and belong on GPU) |
| Draft model | CPU as an addressable device | `ggml-rpc-server -d CPU` bound to loopback, then `-devd RPC0` |
| Draft model threads | E-cores | `-Cd FF000 --cpu-strict-draft 1` |
| Weight streaming | NVMe | `-lm` selection |

The loopback RPC worker is the one mechanism that makes "target on dGPU, draft on CPU" expressible
as an explicit device assignment, because `-dev CPU` is rejected at parse time while `-ot ...=CPU`
is accepted. If Cerebro spawns an RPC worker it **must** pass `-d` explicitly — the default binds
the local dGPU, which would make the planner believe it has a second 8 GiB accelerator that is
actually the same card, contending for the exact VRAM the target model needs.

`--rpc` **hard-aborts** the process on an unreachable endpoint (`ggml-rpc.cpp:547`) rather than
degrading. Every RPC endpoint gets a TCP connect probe first, and an abort is classified as an
infrastructure error, never as a benchmark result.

### iGPU path (deferred, honestly)

Reaching the Iris Xe requires a Vulkan-enabled llama.cpp. This is a real work item and it reopens a
measured hazard: a self-built llama.cpp fork on this machine ran **3-5× slower** than the official
release. Therefore: prefer the official upstream Vulkan release binary over a self-build, keep the
CUDA-only build as a control, and treat the binary as a tuned variable rather than a constant.

UNVERIFIED and gating: the Iris Xe driver (30.0.101.1660) caps at Vulkan 1.3.205 and may not satisfy
the ggml Vulkan backend's required extension and cooperative-matrix surface.

If a Vulkan build also carries CUDA, the same dGPU appears **twice** (as `CUDA0` and as a Vulkan
device). The planner must deduplicate by PCI identity or it will double-count 8 GiB of VRAM.

## 6. Measurement substrate

Getting this wrong produces confidently wrong configurations, so it is specified before any search
code exists.

- **Primary source**: `llama-bench` JSON. Parse `samples_ns`/`samples_ts` — the raw per-repetition
  samples — not `avg`/`stddev`. Raw samples permit nonparametric statistics instead of trusting a
  mean over five repetitions.
- **Discard the first repetition, always.** Measured: rep1 = 124 t/s vs rep2 = 173 t/s on a 0.5B
  model *with warmup already enabled*. Set `-r` to `desired_reps + 2` and drop the first.
- **Never report a single number.** Every configuration is characterized by a triple: `pp_tok_s` at
  depth 0, `tg_tok_s` at depth 0, and `tg_tok_s` at depth D where D is the target working context
  (via `-d`). Measured decay: 35.08 → 31.49 t/s between d0 and d64 on a small model. A depth-0-only
  number systematically overstates real use.
- **Two-tier protocol.** Tier 1 screening: `llama-bench`, `-r 7`, `-d {0, ctx/2}`, `--delay 5`.
  Tier 2 confirmation: only the top ~3 survivors go to a sustained soak. This exists because
  short benchmarks on this machine overstate MoE throughput by roughly **3×**.
- **VRAM trend is a pass criterion.** A soak passes only if free VRAM has a non-negative trend over
  its final third. This catches the failure mode where a configuration runs for hours and then dies.
- **Speculation is measured through `llama-server`, not `llama-bench`**, which has no draft flags.
  This is a second, separate harness path and is budgeted as such.
- **Preconditions are recorded, not assumed.** Free VRAM swung from 431 MiB to 7285 MiB within
  minutes on an ordinary desktop purely from other processes. A configuration measured once against
  a quiet machine will OOM the moment the user opens a browser. Every result records free VRAM at
  start, and the recommended configuration carries a headroom margin.
- **Search with CUDA sysmem fallback disabled.** When VRAM is exhausted the driver silently spills
  to host memory over PCIe and throughput collapses by roughly an order of magnitude *with no
  error*. Disabling it converts a silent 10× cliff into a clean, fast, unambiguous OOM signal.
  UNVERIFIED: the NVAPI DRS setting ID for this policy is not yet confirmed; `nvidia-smi` is
  verified blind to it.

### Provenance is part of the result key

A self-built fork measured 3-5× slower than the official release. Therefore every measurement record
is keyed by `(backend build number, commit, compiler, selected ggml-cpu-*.dll variant, backend
SHA-256, model SHA-256, free VRAM at start)`. Binary identity is a dimension, not a footnote.

## 7. Quality gate

Throughput without a correctness gate is how you ship a fast, broken product.

- **Primary (lossless gate)**: `llama-results` — prefill-logit NMSE against a reference. Fast,
  single prompt, one scalar, hard OK/FAIL.
- **Secondary (quality gate)**: `llama-perplexity --kl-divergence` against a `--kl-divergence-base`
  reference-logits file. Also available: `--hellaswag`, `--winogrande`, `--multiple-choice`.
- **Empirical tolerance, measured locally**: moving the entire computation from CPU to CUDA moved
  NMSE only to `8.454e-07` and still returned OK. Switching to `q8_0` KV moved it to `3.3e-03` and
  returned FAIL. That is roughly a 400× gap between "different arithmetic, same model" and "lossy
  cache" — an unusually clean band to set a threshold in.
  UNVERIFIED: the exact upstream NMSE pass constant is bracketed to `(8.454e-07, 3.324e-03]` by
  measurement but should be read from `tools/results/results.cpp` before being quoted in user-facing
  documentation.
- **Rejected definition**: "identical greedy token sequence under a fixed seed". Too weak (a short
  prompt has few chances to diverge) and brittle in the wrong direction (one near-tie logit flips
  the entire downstream sequence).

### What speculative decoding actually guarantees

This was verified in `common/sampling.cpp` (`common_sampler_sample_and_accept_n`) rather than
assumed, because the claim is load-bearing for both correctness and marketing, and because two
independent research passes contradicted each other about it — one asserting the n-gram variants are
lossy, the other asserting llama.cpp performs Leviathan rejection sampling. **Both are wrong.**

The verification loop is *sample-then-compare*:

```c
for (i = 0; i < draft.size(); i++) {
    const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
    common_sampler_accept(gsmpl, id, true);
    result.push_back(id);
    if (draft[i] != id) break;
}
```

At every drafted position the target model samples its **own** token through the full sampler chain
from its own logits. The emitted token is always the target's. The draft token is used solely as an
equality predicate controlling early exit. There is no acceptance ratio `min(1, p/q)` and no
residual-distribution resample; the draft's probabilities never enter the computation, and a draft
token is never substituted into the output.

Three consequences, all of which the product depends on:

1. **The guarantee is uniform across all ten strategies.** There is exactly one call site
   (`tools/server/server-context.cpp:3798`) and it is spec-type agnostic. Whether the draft came
   from a draft model, an EAGLE-3 head, MTP layers, block diffusion, or any of the five n-gram
   matchers, it meets the same equality test. A worse draft source lowers the acceptance rate and
   therefore the speedup; it cannot change the output distribution. **The n-gram family carries
   exactly the same guarantee as `draft-simple` — there is no weaker tier.**
2. **A hostile draft model cannot steer output.** Correctness rests on an integer equality test, not
   on the draft being well-calibrated or trustworthy. A malicious draft can waste time and nothing
   else. This materially reduces the risk of the second untrusted GGUF that §8 flags.
3. **It is distribution-preserving but NOT bit-exact**, and the product must say the former. The
   verify pass evaluates `n_draft+1` positions in one batched target forward while ordinary decode
   runs single-token forwards; different batch shapes select different kernels and different
   reduction orders, so logits differ in the low bits and a near-tie argmax can flip. This is the
   same hazard that makes the "identical greedy token sequence" definition unusable above.
   Additionally, `--spec-draft-backend-sampling` defaults to **enabled** and relocates where
   sampling happens, which is a second independent source of divergence — and a searched axis.

**Product rule: speed and quality are always reported together.** A dense 27B scores ~7.0/8 on the
reference reasoning benchmark at ~2 tok/s; Gemma-4-E4B scores 6.25/8 at 61 tok/s. A speed-only
leaderboard actively misleads. Every result row carries both.

## 8. Security posture

Three concrete adversaries, not "hackers":

1. **A hostile file** — GGUF, config, or results file.
2. **A hostile web origin** in a browser tab the user already has open, reaching loopback.
3. **A hostile other local principal.** This is not hypothetical on the reference machine: the
   backend directory is Modify-writable by `ASUS\CodexSandboxUsers`, and ggml loads backends as DLLs
   from that directory. That is a live DLL-plant vector.

Requirements:

- **Backend integrity verified before every spawn**, not once at install. SHA-256 manifest check,
  plus a hard refusal to spawn — not merely a warning — when the backend directory DACL, or the
  path used to reach it, is non-owner-writable.
- **Never inherit a llama-server default.** Explicitly emit `--host`, `--port`, `--cors-origins`,
  `--no-cors-credentials`, `--api-key-file`, `--no-slots`, `--no-webui` on every spawn. `--slots`
  defaults to **enabled** and exposes slot state to anything on loopback.
- **Argv construction is the highest-leverage code in the product.** No shell is involved, so there
  is no metacharacter injection — but *argument* injection is live: 141 flags take values and
  several take comma-separated lists (`--lora`, `--device`, `--override-tensor`, `--rpc`). The flag
  space Cerebro can emit is a **closed allowlist**, shipped as a reviewed data file keyed to backend
  build, written **before** the argv builder.
- **The autotuner is itself an attack surface.** It is a machine that generates flag combinations
  and persists them. Without a closed allowlist, a poisoned config or results file steers it into
  emitting dangerous flags.
- **Speculation doubles the untrusted-file surface.** The draft model is a second untrusted GGUF in
  the same process. Every GGUF mitigation applies to it identically.
- **Cerebro refuses to run elevated.** Nothing in the inference path needs admin. Consented tuning
  changes escalate a separate, short-lived helper per change.
- **Windows tuning is N independent consented operations**, each carrying a stored revert action —
  never one "optimize" button. VBS/HVCI are currently ON on the reference machine; any option to
  disable them removes live protection and is rendered as a visually distinct security warning, is
  never a default, and is never bundled with performance changes.
- **Secrets are nearly nonexistent, and the design says so plainly.** The only real ones are the
  local API bearer token and an optional HF token. DPAPI CurrentUser protects those two. Encrypting
  a config file with a key stored beside it is theater and will not be shipped as if it were
  security.
- **Cerebro's own HTTP API**: bearer token on every route including reads, Origin allowlist,
  a required custom header on state-changing routes (so no-preflight simple requests cannot reach
  them), Host-header validation, request size limits.
- **Update check**: opt-in, default OFF. One outbound HTTPS GET for a static version string. No
  identifiers, no payload, no auto-download, no auto-execute. Its irreducible leak is DNS + TLS SNI
  — "a Cerebro user at this IP checked at time T" — which cannot be engineered away while the feature
  is on, and therefore belongs in the opt-in prompt text, not in a footnote.

## 9. Flag-emission hazards

Verified behaviors that will silently corrupt generated configurations:

- **A spawn from an audited backend directory must pin the child's working directory to that same
  directory.** ggml's backend loader (`ggml_backend_load_best`) scans both the executable's own
  directory and the process's *current* directory for `ggml-<name>-*.dll`, loading every match.
  `cerebro-probe::backend::audit_dacl` only examines the directory it is given; a supervisor or bench
  harness that spawns `llama-server`/`llama-bench` without also setting `current_dir` to that
  directory can load an unaudited DLL from wherever Cerebro's own process happened to start —
  `cerebro-probe::backend::probe_command` is the reference pattern.
- **Two dialects, two emitters.** `llama-server`/`llama-cli` use comma-separated device lists;
  `llama-bench` uses `/` for `-dev` and `-ts` and `;` for `-ot`, because comma is reserved for its
  sweep syntax. Reusing one serializer across both is a latent bug: `-dev CUDA0,CUDA1` means *two
  benchmark runs* in bench and *two devices* in server.
- **`-devd` has no environment binding** while `-dev` has `LLAMA_ARG_DEVICE`. A half-env config
  silently applies the target device and drops the draft device. The config format is argv-based.
- **`-dev CPU` is rejected; `-ot ...=CPU` is accepted.** CPU placement routes through `-ot`,
  `-cmoe`/`-ncmoe`, `-ncffn`, and `-dev none` — never through `-dev`.
- **Device names are case-insensitive; the `none` sentinel is case-sensitive** and lowercase-only.
  `-dev None` hard-fails. Emit canonical `CUDA0`/`RPC0` and lowercase `none`.
- **`-mg` and `-ts` are not validated at parse time.** `-mg 1` on a single-device machine, `-ts 1;1`
  with the wrong separator — all parse cleanly and fail later at model load. Parse-time dry-run
  validation catches everything else but not these two.
- **`-ot` is first-match-wins under unanchored `regex_search`.** Emit fully escaped patterns
  (`blk\.17\.`, not `blk.17.`) and narrow-before-broad. `=` and `,` are structurally reserved in the
  `-ot` value, so a generated pattern must never contain a comma — bounded repetition like `{1,2}`
  is forbidden. Enumerate per-layer patterns instead.
- **`--fit` is poison for reproducibility, and that is desirable here.** It is ON by default and
  adjusts only UNSET arguments; setting `-ngl` or any `-ot` hard-aborts its fitting. Cerebro sets
  both explicitly on every measured run — which means **Cerebro owns OOM safety** for every value it
  pins, because it has disabled the safety net.
- **Validate every generated config with a cheap `--list-devices` dry run** before spending a soak
  test on it. Parse-time validation is free and catches `-dev`/`-devd`/`-sm`/`-ot` errors in under a
  second.
- **Emit `-lm/--load-mode`, never `--mlock`/`--mmap`/`--no-mmap`/`-dio`** — those are deprecated in
  b10751 and emit deprecation noise that Cerebro would have to parse around.
- **Grammar plus backend sampling: investigated, disproven, and recorded so it is not re-added.**
  A source reading reported `GGML_ASSERT(!gsmpl->grmr && "using grammar in combination with backend
  sampling is not supported")` in `common/sampling.cpp`, and since `--spec-draft-backend-sampling`
  defaults to **enabled**, the inference was that structured output plus speculation aborts the
  server. That inference was written into the flag layer as a hard mutual exclusion.

  It was wrong. Six cases were run against b10751 with a 0.5B model — `--grammar` with
  `--spec-type ngram-simple` at the default draft-sampling setting, `--grammar` with explicit `-bs`,
  and `-bs` alone, each on both the CPU backend (`-dev none`) and the CUDA backend
  (`-dev CUDA0 -ngl 999`). All six completed with `rc=0` and correct grammar-constrained output.
  The assert is not reachable through the documented flag surface in this build.

  The constraint was removed and replaced with regression tests. Blocking it would have denied
  JSON-mode output to precisely the users most likely to need it, for no reason. **The governing
  rule, now in `docs/CLAIMS.md`: read the source to form a hypothesis, run the binary to make a
  claim.**

## 10. Honest limitations

Stated here so they can be stated on the website.

- No NPU support. Not "coming soon" — the reference class of hardware has no NPU, and no NPU runtime
  performs the operation the original design imagined.
- No Mixture-of-Depths. It cannot be retrofitted onto trained weights.
- No inference-time linear-attention conversion. It requires a training run.
- Context length is bounded by arithmetic, not by ambition. Cerebro reports the maximum context it
  measured on the actual machine, never a headline number the hardware cannot hold.
- iGPU utilization is gated on a Vulkan-enabled backend and on driver capability that is currently
  unverified. Measured hybrid-graphics data shows decode throughput flat or regressed, so this is
  not presented as a pending win.
- **No MXFP4 or NVFP4.** MXFP4 exists in the `ik_llama.cpp` fork, not mainline — an easy thing to
  misread as upstream support. NVFP4 targets Blackwell FP4 tensor cores; the reference RTX 3070 Ti
  is Ampere (compute 8.6) and would run it unaccelerated at best.
- **No sub-2-bit quantization claims.** QuIP#/AQLM/BitNet-style "lossless 2-bit" results are
  unreproduced here and contradicted by the reference machine's own measurement of a 1-bit 27B
  emitting whitespace.
- StreamingLLM / attention sinks / `--keep` bound memory for indefinite streaming by forgetting the
  middle of the context. That is a do-not-crash feature, not a context-length feature, and is not
  presented as one.
- Cerebro is not comparable to vLLM, SGLang or TensorRT-LLM in either direction. Their published
  figures are aggregate multi-request throughput on datacenter GPUs; the comparison is
  category-invalid and will not appear in Cerebro's materials.
- Cerebro improves configuration. It does not make a small model smart or a large model fit.
