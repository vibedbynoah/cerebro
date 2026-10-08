# Cerebro — Claims Policy

Binding on the website, the README, release notes, and any launch post.

Every claim Cerebro makes must be one a hostile reviewer can check and confirm. This file exists
because an earlier design round nearly shipped two false statements onto the homepage — one of them
the load-bearing correctness claim for the product's core feature.

**Rule zero: any speedup figure is invalid unless it is accompanied by the machine, model, quant,
backend build, context depth, and sampler settings it was measured under.** No exceptions, including
in social copy where the detail "doesn't fit."

---

## Claims we may make

### About the differentiator

- Cerebro tunes llama.cpp's speculative decoding by measured throughput. `llama-bench` — the standard
  llama.cpp benchmarking tool — contains no speculative parameters at all, so no llama-bench-based
  tuner can reach this axis.
- Five of the ten speculative strategies (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`,
  `ngram-mod`, `ngram-cache`) need no draft model, no extra VRAM, and no second download.
  Speculative decoding is therefore available for every model in your library, not just ones with a
  published vocabulary-compatible partner.
- llama.cpp's built-in autofitter (`common/fit.cpp`, `--fit`, on by default) maximizes offload
  subject to a memory constraint. The memory-optimal configuration and the throughput-optimal
  configuration are frequently different points.

### About correctness

- Speculative decoding in llama.cpp is **output-distribution-preserving** for every strategy. The
  verifier samples from the target model and uses the draft only as an equality test, so the draft
  cannot change which token is emitted.
- Because the draft never substitutes a token, a bad — or even hostile — draft model cannot steer
  output. It can only waste time. Correctness rests on an integer equality test, not on the draft
  being well-calibrated.

### About measurement

- Every configuration is characterized by three numbers, never one: prefill tok/s, decode tok/s at
  depth 0, and decode tok/s at your working context depth. Depth-0-only numbers systematically
  overstate real use.
- Short benchmarks on this class of hardware overstate MoE throughput by roughly 3×. Cerebro screens
  quickly, then confirms survivors under a sustained soak, and a soak passes only if free VRAM shows
  a non-negative trend across its final third.
- We discard the first benchmark repetition, always. We measured rep 1 at 124 t/s against rep 2 at
  173 t/s with warmup already enabled.
- Speed and quality are always reported together. A dense 27B scored ~7.0/8 on our reasoning
  benchmark at ~2 tok/s; Gemma-4-E4B scored 6.25/8 at 61 tok/s. A speed-only leaderboard actively
  misleads.
- Every measurement is keyed to the exact binary that produced it — build number, commit, compiler,
  backend hash, model hash, and free VRAM at start. We measured a self-built llama.cpp fork running
  3-5× slower than the official release of the same commit.

### About privacy and safety

- Cerebro verifies backend integrity before every spawn and never inherits a llama-server default.
  It refuses to run elevated, and every Windows tuning change is a separate consented operation
  carrying its own revert action.
- Cerebro starts the model-running process inside a Windows Job Object created with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`crates/cerebro-supervisor/src/spawn/job.rs`): the moment
  Cerebro's own handle to that job closes — an explicit stop or Cerebro's own process dying
  unexpectedly — every process assigned to it, including anything the model process itself spawned,
  is terminated by the kernel. The child's environment is never inherited: it is rebuilt from a
  closed allowlist (`env.rs`), so it starts with none of Cerebro's own environment variables or
  secrets, and its `PATH`/`SystemRoot` are resolved fresh from the OS rather than copied from a
  value that could be poisoned upstream. Neither mechanism restricts the child's file-system or
  network access — see "Claims we may not make" below for the wording this rules out.
- Cerebro reports the maximum context it actually measured on your machine, never a headline number
  your hardware cannot hold.
- We publish what we do not do: no NPU support, no Mixture-of-Depths, no inference-time
  linear-attention conversion, and no iGPU use until a Vulkan-capable backend is verified.
- Apache-2.0. Cerebro configures llama.cpp; it does not fork, patch, or replace it.

---

## Claims we may not make

Each of these is false, unverifiable, or category-invalid. Several were in a draft of our own
architecture document and were removed.

### Correctness overclaims

- ❌ **"Bit-exact identical output" / "byte-identical" / "produces exactly the same tokens"** for
  speculative decoding. The verify pass batches `n_draft+1` positions into a single target forward,
  changing kernel selection and reduction order; a near-tie argmax can flip. The correct word is
  *distribution-preserving*. This was in our own draft and is the single most important correction
  in this file.

### Originality overclaims

- ❌ **"Nobody tunes llama.cpp by measured throughput" / "the first tool to measure instead of fit"
  / "no automatic flag tuning exists."** False and trivially falsifiable: `llama-optimus` and
  `TurboLLM auto-tune` both already do throughput-objective tuning.
- ❌ **"Cerebro searches axes nothing else searches"** as a blanket statement. Only the speculation
  axis qualifies. `-ngl`, `-ncmoe`, batch size and context are searched by existing tools.

### Unsupported numbers

- ❌ Any specific speculative speedup multiplier — "2× faster", "2-4× with EAGLE-3" — as an expected
  result. Those figures come from datacenter GPUs at small batch under vLLM/SGLang.
- ❌ Quoting XGrammar's structured-decoding speedups. XGrammar is not integrated into llama.cpp.
- ❌ Any claim that Cerebro is faster than vLLM, SGLang or TensorRT-LLM. Their published figures are
  aggregate multi-request throughput on datacenter GPUs; the comparison is category-invalid in both
  directions.

### Hardware overclaims

- ❌ **"Uses your whole machine" / "harnesses the idle iGPU alongside the dGPU."** The shipping
  CUDA-only build cannot see the Iris Xe at all, and measured hybrid-graphics data shows decode
  throughput flat or regressed.
- ❌ Any NPU claim, **including "NPU support coming."**
- ❌ Advertising MXFP4 or NVFP4 support. MXFP4 is in the `ik_llama.cpp` fork, not mainline; NVFP4
  targets Blackwell FP4 tensor cores.

### Capability overclaims

- ❌ **Any "1M token context"** or similar headline number for this hardware class. KV arithmetic
  rules it out, and cross-context reasoning quality degrades well before any nominal limit.
- ❌ Presenting StreamingLLM / attention sinks / `--keep` as extending usable context. They bound
  memory for indefinite streaming by forgetting the middle. That is a do-not-crash feature.
- ❌ "Lossless 2-bit" or any near-lossless sub-2-bit quantization claim. Unreproduced, and
  contradicted by our own measurement of a 1-bit 27B emitting whitespace.
- ❌ Marketing `--moe-expert-cache-size` as an optimization. Our controlled sweep measured it slower
  than no cache at every size on an 8 GB card.

### Sandbox overclaims

- ❌ **"Sandboxed" described as filesystem or network isolation** — "can't read files," "can't use
  the network," "no network access," "touch hardware beyond what it's explicitly given," or similar.
  The implementation (`crates/cerebro-supervisor/src/spawn/job.rs`) is a Windows Job Object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, which guarantees process-tree cleanup, not I/O restriction —
  a Job Object with only this limit set does not restrict what the process it contains can open,
  read, or connect to. Say only what the Job Object (process-tree kill-on-close) and the
  clean-environment build (`env.rs`, a closed allowlist rebuilt from scratch, never inherited)
  actually do.

### Credit we have not earned

- ❌ Claiming credit for upstream defaults — `min_p 0.05`, the penalties/dry/top_n_sigma chain, KV
  quantization types, windowed SWA, context checkpoints, prompt-cache reuse, or the model swapper.
  These ship in llama.cpp and our audience reads release notes.

---

## Standard of evidence

| Claim type | Required backing |
|---|---|
| Throughput number | Our own measurement, with full configuration disclosed, first repetition discarded, sustained not burst |
| Quality number | Task benchmark or KL-divergence against reference logits, with the gate and tolerance stated |
| "Lossless" | Distribution-preserving argument from mechanism, never "we saw the same output once" |
| Upstream behavior | Verified against the binary we ship against, by running it — not by reading a summary of it |
| Competitor comparison | Same machine, same model, same quant, or no comparison at all |

The last row of the first column is not a formality. A hard error was written into this project's
own constraint layer on the strength of a source reading, and running the binary disproved it on the
CPU backend within minutes. Read the source to form the hypothesis; run the binary to make the claim.
