# Topic F — Orchestrate Mode and Product Features for Cerebro
### The fastest local AI platform on one RTX 3070 Ti Laptop (8 GB, sm_86), i9-12900H, 32 GB DDR5, Windows 11

**Scope.** This paper specifies (1) an Orchestrate mode modeled on GVS5H ledger-based zero-shot self-orchestration and how to run it on one local model with continuous batching, (2) mechanics from Strata worth matching or beating, (3) a Monitor screen with every metric worth showing plus cheap Windows sampling methods, (4) Turbo/customization switches with beginner and advanced explanations, (5) onboarding and per-machine auto-tuning, and (6) an automated glitch-hunting checklist for a Tauri 2 + SolidJS app.

**Verification status (checked 2026-10-17).**
- **1 — Sources verified today.** arXiv:2608.26480 (GVS5H, v2, CC BY 4.0) and github.com/slee-persis/GVS5H (MIT, 750 stars at fetch time, 55 commits) were fetched and read; details from both are marked VERIFIED below. github.com/Niko1221/Strata (MIT, 26.4k stars, 2,174 commits, created 2026-09-24) plus its docs `HOW_IT_WORKS.md` and `BATCHING.md` were fetched and read; details from them are marked VERIFIED. Everything else is engineering design (marked DESIGN) or UNVERIFIED where I could not confirm it.
- **2 — Status legend.** VERIFIED = fetched from the primary source today. CITED-IN-PAPER = identifier taken from the GVS5H reference list but not independently fetched. DESIGN = this paper's engineering proposal. UNVERIFIED = plausible but unconfirmed; validate before shipping.
- **3 — Naming.** "GVS5H" is expanded by its authors as "Gorilla vs. 5 Humans": five cheap self-orchestrated instances vs. one expensive frontier model. "Strata" is an independent local inference engine for Qwen3.8-Flash-Next; its README is written for end users, not spec language — where its docs make quantitative claims I quote them as repo claims.
- **4 — Product frame.** Cerebro's Orchestrate mode should be a **throughput mode** (many problems in flight, batched across roles), not a single-problem latency mode, except when the user explicitly trades wall-clock for accuracy on one hard task. Every design decision below is sorted by that frame.

---

## Part 1 — Orchestrate mode: GVS5H ledger-based zero-shot self-orchestration, localized to one engine

**5 — What GVS5H is (VERIFIED).** A training-free scaffold in which fresh instances of one model decompose a problem and coordinate through a shared filesystem. On the 100 latest hard LiveCodeBench problems (release_v6), it yields up to +25.6 points; Qwen3.8-27B rises 66.8 → 92.4 pass@1, Qwen3.8-Flash-Next 84.2 → 93.0, against Claude Fable 5's 90.4 single-call. Gains are conditional: some configurations are unchanged or worse. Source: arXiv:2608.26480; benchmark arXiv:2403.07974.

**6 — Why it works (VERIFIED, from transcript analysis).** Two mechanisms: (a) decomposition commits a reduction or bounding argument to disk before any code is written, so the next call builds on it; (b) persistent context — state lives in `plan.md`, `notes.md`, `solution.py` rather than one ever-growing transcript, so a token cap cannot truncate the work. The paper documents a Qwen3.8-27B single call that repeated a phrase 7,743 times while re-checking an answer it had already found (abc399_e) — bounded calls that write to disk avoid this.

**7 — Gains are conditional (VERIFIED).** Biggest gains go to models that are weak at organizing their own reasoning: small models with reasoning on, or models with reasoning disabled that otherwise jump straight to code (Kimi-K3 +31.6 points at 16k reasoning-off, per the paper's OpenRouter-served set). A model that both plans internally and stops on its own gains little. One documented regression: on LCB 3765 (Qwen3.6-35B, 128k, reasoning off) the ideation stage talked itself out of the correct optimization that the single call had used, and the worker implemented a worse plan with an extra bug.

**8 — Cost profile (VERIFIED).** The scaffold roughly triples output tokens vs. one call ($/pass: Qwen3.8-Flash-Next single $2.73 → manager $5.76; Qwen3.8-27B single $18.95 → manager $47.82). The paper's framing is that the orchestrated arm of a cheaper model can beat a single call of an expensive model — for a local-only product, the analog currency is wall-clock seconds and joules, not dollars (item 38).

**9 — Roles (VERIFIED).** Every role except the verifier is the same underlying model in a fresh context. There are five model roles: manager (plan and manage), worker-brainstorm, worker-execute, finalizer, and cut-off summarizer. The verifier is harness code, not a model: it runs the problem's public stdin tests. Of the 100 LCB problems, 73 carry stdin-format public tests; the other 27 are LeetCode-style with call-based tests the engine does not run, so they get no verifier signal.

**10 — Ledger/workspace layout (VERIFIED exactly as described in the paper).**

```
<ws>/task.md      # the problem statement
<ws>/plan.md      # the manager's overarching plan
<ws>/tasks.json   # a debug mirror of the task list
<ws>/notes.md     # curated ideas / findings / partial proofs
<ws>/solution.py  # the current solution
```

Workers never talk to each other; each sees the ledger plus the one task the manager assigns. `plan.md` and `notes.md` are capped when written at 4,000 and 8,000 characters so neither grows without bound. In v1 workers appended to notes; in v2 each execute-worker rewrites `notes.md` as a curated whole.

**11 — Ledger v2 for Cerebro (DESIGN).** Keep the paper's five files as the human-visible source of truth, and add an append-only event log so Monitor, replay, and crash recovery get a machine record:

```
ws/<run_id>/
  meta.json      # {schema:1, created, model, engine_build, sampler, budgets}
  task.md  plan.md  notes.md  solution.py  tasks.json
  ledger.jsonl   # append-only; one JSON object per line
  run.lock       # exclusive lock
```

Appends use one `WriteFile` of a whole line + `FlushFileBuffers`; file replacements (`solution.py`) use write-temp + `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` for atomicity. Lock with `LockFileEx` on `run.lock`; the second Cerebro window opens the run read-only. Failure mode this prevents: torn `solution.py` when the app is killed mid-write — the worst possible artifact to corrupt.

**12 — Event schema (DESIGN).**

```json
{"seq":417,"ts":"2026-10-17T09:14:22.913Z","role":"manager","type":"task.assign",
 "task":"t-11","rev":37,"prompt_hash":"9f2c...","result_hash":"d81a...",
 "tokens":{"in":3120,"out":214},"finish":"stop","ms":812}
```

`role ∈ {manager, brainstorm, worker, verifier, finalizer, summarizer}`; `type ∈ {plan.write, task.assign, task.done, task.issue, note.write, solution.write, verify.fail, verify.pass, iter, stop}`. `prompt_hash`/`result_hash` enable cycle detection and prefix-cache auditing (item 33). Measure: replay the log and assert the reconstructed `tasks.json` equals the on-disk one — a cheap consistency test.

**13 — tasks.json schema (DESIGN; the paper only says it is a debug mirror).**

```json
{"schema":1,"rev":37,
 "tasks":[{"id":"t-07","status":"done","desc":"...","deps":["t-04"],
           "evidence":["test 12/13 pass","artifact sha1:ab12…"],
           "rev_created":30,"rev_closed":33}],
 "next":"t-11"}
```

Keep the task list *dynamically generated and updated*, exactly as the paper describes — it is not predetermined (VERIFIED as a property of the method). Status vocabulary: `open|doing|done|blocked`, and a `fix-or-switch` pseudo-task the manager must create when the verifier reports a failure.

**14 — Verifier feedback is ground truth (VERIFIED).** Whenever the round's worker produced a fresh candidate, the public tests run; pass/fail plus the first failing case is fed back, and a failing run **overrides** a manager "done" verdict. Locally, extend this with sandboxed execution: run the candidate under a job object with a CPU-time cap (e.g. `JOBOBJECT_CPU_RATE_CONTROL`, 2 s), memory cap, no network, temp-dir cwd. Failure mode: an infinite-loop candidate hanging the orchestrator thread; the job object kills it and the timeout is itself a verifier verdict.

**15 — Ledger as external memory, prompt-injection hygiene (DESIGN).** Notes and solutions are written by the same model, but any tool output (file contents, test output) that lands in the ledger is untrusted data. Delimit untrusted spans with per-run random tags, never execute ledger content as prompt instructions, and cap each untrusted span at N chars. Measure: a red-team suite of 12 planted "ignore previous instructions" strings in test output; assert none change the manager's next task semantics.

**16 — Manager plan prompt (VERIFIED content, tightened template).** The manager reads the problem and writes a 3–6 sentence strategy plus 3–6 concrete seed tasks (paper's exact shape). Temperature 0.3 (VERIFIED).

```
SYSTEM: You are the manager. You never write the final solution yourself.
Read task.md. Write (a) a strategy of 3-6 sentences, (b) 3-6 concrete seed tasks.
Respond as JSON: {"plan":"...","tasks":["...","..."]}
```

Local max_tokens: 512 is ample; the paper's 128k cap is a cloud luxury. Hard-cap the JSON with a streaming partial parser (item 32) so a truncation still yields usable seeds.

**17 — Worker-brainstorm prompt (VERIFIED role, tightened template).** The first worker does not write a solution; it identifies the core difficulty, lists candidate approaches and pitfalls, appends them to notes, and proposes next steps. Temperature 0.4 (VERIFIED). Locally this is your cheapest call — keep it short (≤400 output tokens) to prime the ledger.

**18 — Worker-execute prompt (VERIFIED role, tightened template).** A fresh worker executes exactly one task, rewrites `solution.py` and `notes.md` in full (curated, not appended — VERIFIED v2 behavior), then proposes remaining steps. Temperature 0.2 (VERIFIED). It receives: subagent preamble wrapping the solver prompt, plan, current notes, current solution, and the one task (VERIFIED as the paper's construction). Locally, budget output at 1,500–3,000 tokens per task; force end-of-file writes to be the last thing it does so a truncation can be salvaged.

**19 — Verifier (harness) — exact local behavior (DESIGN).** Steps: (1) static parse + import of `solution.py` in a throwaway process; (2) run public tests with a per-case timeout; (3) on fail, write `verify.fail` with only the first failing case into the ledger; (4) a failing run forces a `fix-or-switch` task. Measure the verifier's own false-negative rate with a mutation suite: 20 mutants of known-good solutions; every mutant must fail at least one public test, or your test coverage is the bottleneck, not the model.

**20 — Finalizer (VERIFIED conditions).** A finalization worker emits the definitive solution whenever the loop ends without a clean sign-off: round budget spent, manager reissued a task it just handed out, or manager named no task. It is **skipped** when the manager declared done and a usable solution is already on disk. Local nuance: "usable" = verifier pass, or absent a verifier, non-empty + parses + smoke-runs.

**21 — Cut-off summarizer (VERIFIED).** A worker that hits its token cap mid-attempt has its partial reasoning summarized by a fresh short call so its ideas still reach the manager. Local version: keep the raw partial in `ledger.jsonl` and summarize into `notes.md` with a 200-token cap; the summarizer is the cheapest model call in the loop and the easiest place to lose information — log both partial and summary hashes so you can A/B the summarizer prompt.

**22 — Sampling profiles per role (VERIFIED temps + DESIGN extras).** Manager plan 0.3, brainstorm 0.4, execute 0.2, curating task list 0.2, single-call baseline 0.2. Locally add: min-p 0.05 for execute to cut tail damage; repeat penalty 1.05 only for brainstorm (novel candidate approaches) and 1.0 for execute (code may legitimately repeat identifiers). With a fixed seed, the whole loop should be replayable on the same engine build — assert determinism in CI for greedy decode.

**23 — The control loop, v2, as pseudocode (VERIFIED flow, formalized).**

```python
state = load(ws)                       # ledger files
if not state.plan: state = manager_plan(state)          # step 1
if not state.brainstormed: state = worker_brainstorm(state)  # step 2
for it in range(MAX_ITERS):            # MAX_ITERS = 10 in v2 (4 in v1)
    state = manager_manage(state)      # step 3: merge, mark, add, declare or pick
    if state.done and verifier_ok(state): break
    if state.no_task or state.reissued_same_task: break          # no-progress guard
    out   = worker_execute(state, state.next_task)               # step 4
    v     = verifier_run(state, out)                             # step 5 (harness)
    if v.failed: state.tasks[state.next_task].status = "open"
    state = apply(state, out, v)
else: pass
if not (state.done and verifier_ok(state)): state = finalizer(state)  # step 6
```

**24 — Termination taxonomy (VERIFIED guards + DESIGN additions).** Five clean stops: (a) manager declares done and the solution is usable; (b) iteration budget exhausted (`MAX_ITERS`); (c) no-progress guard — manager reissues the exact task just handed out; (d) manager names no task; (e) new: wall-clock budget. Plus two kill conditions for local UX: user cancel, and stagnation detector (item 29). Every stop writes a `stop` event with reason; Monitor shows the reason — "solved" vs "budget" vs "no-progress" changes how much the user should trust the answer.

**25 — Cycle and stagnation detection (DESIGN).** Maintain a set of `sha1(task_text)` for issued tasks; a reissue stops the loop (that IS the paper's no-progress guard, VERIFIED). Extend to plan-level oscillation: hash the concatenation of plan+notes each manage round; if the last three hashes alternate A,B,A, stop and finalize. Measure: unit test with a scripted fake model that ping-pongs two tasks; the harness must stop at iteration 3, not 10.

**26 — Budgets as first-class config (DESIGN; paper has MAX_ITERS=10 and char caps).** Expose: `max_iters` (default 10 to match v2), `plan_chars` 4000, `notes_chars` 8000 (paper values, VERIFIED), plus local-only `wall_clock_s`, `total_tokens`, and `max_concurrent_problems`. On the 3070 Ti Laptop, a token budget is the honest control: roughly 100k–400k total tokens is a good default per hard problem.

**27 — One model, many roles: the serving contract (DESIGN).** Each role call is an engine request with an immutable prefix: `role_system_prompt + "\n" + task.md + "\n" + plan.md + "\n" + notes.md + "\n" + solution.py` for workers; manager sees all but solution last; brainstorm sees plan but not solution. Prefixes are constructed byte-for-byte identically across calls (no timestamps, no random ids in prompt text) so prefix caching can hit (item 33). Failure mode: putting the iteration number in the prompt text — cache misses on every call.

**28 — Role-aware continuous batching: why it is the core local problem (DESIGN, informed by VERIFIED Strata measurements).** Manager calls are short and latency-critical (they gate the loop); worker calls are long and throughput-friendly; the verifier is CPU-side. A single naive queue lets a 3,000-token worker prefill block the manager for seconds. Strata's measurements show the stakes on one 12 GB GPU: with one request at a time, the fourth of four requests waited 11.2 s for its first token; with `parallel: 4` it waited 1.8 s, at an 11–24% decode-rate cost (VERIFIED, repo table). Cerebro's scheduler should make the same trade, but priority-aware instead of round-robin.

**29 — Scheduler sketch: three classes, aging, reserved prefill (DESIGN).**

```
class PRIO: MANAGER=0, FINALIZER=1, WORKER=2, BATCH=3      # lower = sooner
admit(req):
  if req.role == MANAGER: req.max_out = 384; req.ttft_sla_ms = 400
  if req.role == WORKER and pending_manager_call: req.defer = True
  if req.state == PREFILL and prefill_running_us > 1500:
      req.chunk = 2048; decode_first(0.5)     # decode-share 0.5 analog
  aging: req.prio -= 1 per 5s waiting          # starvation guard
```

Reserve ~15% of prefill bandwidth for manager-class requests; if a worker prefill is running and a manager call arrives, apply Strata's cooperative-yield shape: stop the worker prefill at its next chunk boundary, copy partial KV into its slot, admit the short manager prompt, then resume (BATCHING.md documents exactly this `BYIELD` mechanism; VERIFIED as a repo mechanism).

**30 — KV/prefix reuse across roles (DESIGN).** Key the prefix cache by `sha1(role_system_prompt + task.md + plan.md + notes.md + solution.py)`; store a radix tree over token blocks; workers executing successive tasks share everything through the plan but diverge at notes (rewritten, not appended — VERIFIED v2), so expect reuse through the plan and miss from the notes boundary onward. Measure hit tokens / total prefill tokens per call in the ledger; target ≥ 60% on worker calls after iteration 2. On the Gated DeltaNet hybrid, attention layers use KV while the linear-attention layers carry recurrent state — reuse applies to the KV part; the recurrent state must be replayed for any prefix beyond the reused boundary (see item 34).

**31 — Role sessions and parking (DESIGN, Strata-verified analog).** The manager is a recurring conversation: park its session between rounds so its own analysis rejoins cheaply instead of being re-prefilled. Strata's parking cache (`--conversation-cache-mib/slots`) restored follow-up turns at 0.53 s/0.46 s vs 3.9 s/5.7 s first turns (VERIFIED repo measurement). Cerebro equivalent: keep manager KV as a first-class cached session with LRU eviction; worker sessions are disposable.

**32 — Streaming plan parse to hide manager latency (DESIGN).** Parse the manager's JSON plan incrementally; when the `tasks` array's first 2 elements are complete, speculatively admit worker tasks (subject to a `speculate ≤ 2` cap and cancellation if the manager later marks them "done"). Failure mode: speculative worker burns GPU on a task the plan deletes — accept the waste, cap it, log `speculation_waste_tokens` in the ledger as a tunable.

**33 — Iteration-over-problems batching (DESIGN).** Orchestrate multiple problems simultaneously: N problem-workspaces, each at a different stage of the loop, all feeding one batch. This is what makes Orchestrate a *throughput* mode: with N=4 on this laptop you keep the GPU busy while managers think, verifiers run, and workers decode. The paper's harness ran `--parallel 8` (VERIFIED in the repo's example command); locally start with 2–4 and expose the knob.

**34 — Hybrid-model session state (DESIGN, correct-by-construction note).** For Qwen3.8 hybrid (Gated DeltaNet layers; cf. arXiv:2412.06464 for gated delta rule, and Mamba-2 arXiv:2405.21060 for the SSD formulation), each session carries: attention KV, linear-recursion state per GDN layer, and (in the Qwen3-Next family, per Strata's docs the model has "GDN recurrence, QSA K/V and indexer, PLE history" per slot — VERIFIED that Strata tracks these five per-slot state kinds). Batch slots must copy these states atomically on admission; a torn GDN state produces subtle garbage rather than a crash — test by checksumming state before/after a slot round-trip.

**35 — Wall-clock model of the loop (DESIGN).** Per iteration: `t = TTFT_manager + decode_manager + TTFT_worker + decode_worker + verify_ms`. With prefix reuse and a manager parked, TTFT_manager ≈ 50–150 ms locally; the worker dominates. Ceiling: 10 iterations ≈ 10 × (worker call). Rule of thumb for the user: Orchestrate costs ≈ 3–6× the tokens of one call (paper: ~3×, VERIFIED) and roughly the same multiple of wall-clock at equal concurrency; it buys accuracy only where planning was the bottleneck.

**36 — Measuring gain per model: the protocol (DESIGN, mirroring the paper's statistics).** For each local model: same benchmark split (start with LCB-100 release_v6 subset of 30 problems to fit the laptop's time budget), 3–5 passes per arm, single-call arm and orchestrate arm, identical sampler except the paper's per-role temps, 128k cap analog = generous local cap. Compute per-problem mean Δ, paired sign-flip permutation test (the paper uses 2×10⁵ resamples, Holm-corrected; VERIFIED), 95% t-intervals across passes. Local additions: tokens/pass, wall-clock/solve, joules/solve (item 72), and pass@1-at-equal-wall-clock (item 37).

**37 — Budget-matched comparisons (DESIGN, addresses the paper's own limitation).** The paper explicitly does not attempt equal-token comparison (VERIFIED limitation). Locally, add two matched arms: (a) *same total tokens* — single-call arm gets `k×` samples of one call where k ≈ scaffold's token multiplier, best-of-n with verifier; (b) *same wall-clock* — every method gets the same seconds on this machine. Report the Pareto frontier (accuracy vs. seconds) per model. Related work: "Reasoning in Token Economies" (Wang et al., EMNLP 2024; CITED-IN-PAPER) and Tran & Kiela, arXiv:2604.02460 (CITED-IN-PAPER), which finds single-agent LLMs can outperform multi-agent systems under equal thinking-token budgets.

**38 — Local currency conversion (DESIGN).** Convert the paper's cost table to your machine: joules/solve = ∫P dt over the solve (GPU W from NVML + CPU package W estimate). Since local inference is capital-cost (electricity + time), the headline metric for Orchestrate should be **solved-per-hour at fixed quality floor** and **joules per solved problem**. Example target bands for the 3070 Ti Laptop on 30B-A3B Q4: single-call ~15–25 tok/s decode (estimate, UNVERIFIED — measure during onboarding), so a 3× token multiplier is a ~3× wall-clock multiplier at concurrency 1; concurrency 3–4 restores most of it.

**39 — When to enable Orchestrate by default (DESIGN gating rule).** Ship it off by default; recommend it via an in-app badge only when the per-model sweep shows both: `Δpass@1 ≥ +4pp` (Holm-significant) **and** `tokens/solve ≤ 2.5×` the single arm. This matches the paper's conditional finding (VERIFIED) and prevents the LCB-3765 pattern (a strong single-call model made worse).

**40 — Comparison matrix vs the alternatives (DESIGN, each method grounded in its paper).**

| Method | Call count | Needs verifier | Best at | Local cost shape |
|---|---|---|---|---|
| Single call | 1 | no | strong models, short tasks | baseline |
| Self-consistency (arXiv:2203.11171) | n | answer extraction | verifiable answers | linear n; batch-friendly |
| Best-of-n + exec/LM verifier (arXiv:2306.05685) | n | yes (tests/judge) | code, math | linear n; GPU-bound |
| Reflection/Refine (arXiv:2303.17651, 2303.11366) | 2–5 | soft | prose, small fixes | sequential |
| Debate (arXiv:2305.14325) | k×r | judge | factuality | high; equal-budget findings negative (2604.02460) |
| ToT/LATS/GoT (arXiv:2305.10601, 2310.04406, 2308.09687) | 10s–100s | value model | puzzles, small branching | latency-monster locally |
| MoA (arXiv:2406.04692) | n×layers | aggregator | quality at any cost | heaviest |
| GVS5H | 2–13 | public tests | planning-limited models | ~3× tokens, sequential per problem |

**41 — Where each wins locally (DESIGN decision table).** Verifiable + long tail of hard cases → best-of-n with execution verifier, or GVS5H when you also want progress written down. Non-verifiable prose → self-consistency or single call; GVS5H adds little. Small-search puzzles → budgeted ToT with a cheap value head; skip on laptops under 16 GB. Mixed workloads → router: a 3-feature difficulty classifier (prompt length, logit entropy of the first 16 tokens, self-check "is this one-shot solvable?") picks single vs orchestrate; train it from your own ledger data (you get labels free: which runs the single call failed).

**42 — GVS5H failure modes (VERIFIED where quoted, DESIGN where extended).** (a) Wrong-plan anchoring: wrong early notes poison all downstream workers — the paper's own suggested fix is fresh-perspective workers with the raw statement and no context (VERIFIED suggestion); implement as `fresh_worker_every: 2` iterations. (b) Confidently-wrong worker overwrites a correct intermediate — the paper suggests running candidates against a generated test suite before acceptance (VERIFIED); locally the public-test verifier already covers this where tests exist. (c) Unbounded fan-out is prevented by `MAX_ITERS` and char caps (VERIFIED guards). (d) Reward hacking of a soft verifier — never let an LLM-only verifier block a worker whose artifact fails execution; execution always outranks it. (e) Ledger drift — `tasks.json` and the manager's private state diverge; the replay consistency test (item 12) catches it.

**43 — Orchestrate UX in Cerebro (DESIGN).** The user sees: a progress tree (tasks/status from `tasks.json`), the current ledger diff (what changed this round), stop-reason on completion, and a live "tokens spent vs. single-call estimate" meter. Kill switch must finalize, not discard: on cancel, run the finalizer against the current ledger so the user keeps the best artifact.

**44 — Security of a local orchestrator (DESIGN).** Workspace sandboxing (path canonicalization + reject `..`), code execution only inside the job-object sandbox (item 14), network disabled for executed code, and a loud UI notice that Orchestrate runs model-generated code. This is table stakes for a desktop product that executes LLM output on the user's machine.

---

## Part 2 — Strata mechanics worth matching or beating

**45 — What Strata is (VERIFIED).** An MIT-licensed C++/CUDA/HIP engine + Python server + web app that runs Qwen3.8-Flash-Next (125B total, 6B active per token) on one 12 GB consumer GPU with 32 GB+ RAM by keeping hot experts in VRAM, all experts in RAM, the CPU computing non-cached experts in place concurrently, and only a lookup table on SSD. One-click `START-HERE.bat` on Windows; OpenAI/Anthropic/Responses APIs on `127.0.0.1:8080`; an MCP server; a Monitor tab.

**46 — Expert tiering numbers (VERIFIED repo claims).** 24,576 experts total, ~10 routed per token; ~700 experts per extra GB of VRAM; VRAM holds an adaptive cache of a few thousand hot experts; RAM pins all experts; CPU computes misses in place at the same time the GPU computes cached ones; the SSD holds a 28.8 GB n-gram table read a few rows per token through the OS cache. Cerebro's MoE offload should expose the same economics: **VRAM GB → expert slots → hit rate → decode speed** as a first-class calculator in Turbo.

**47 — Guess-and-check spec decoding (VERIFIED repo claims).** Strata's model has a built-in MTP layer drafting up to 3 tokens; a single pass over all 48 layers verifies them, yielding 2.4–3.2 tokens per pass and 1.6–1.8× faster decode. Cerebro should match with two mechanisms: its own MTP/draft head if available in the model, plus a tiny 0.5–1.5B draft model option; acceptance-rate telemetry belongs in Monitor (item 66).

**48 — Prompt-lookup drafting (VERIFIED repo claim).** When output copies context (code edits, quoted text), lookup drafts up to 5 tokens from the earlier copy — gated on measured acceptance and cost: +6–11% on code edits, unchanged elsewhere. This is nearly free to implement (suffix automaton over the prompt) and should ship enabled for agent/code workloads.

**49 — Prefill in 8,192-token chunks with layer-streaming overlap (VERIFIED repo claim).** Big chunks keep prefill >1,000 tok/s; experts of the next layer stream over PCIe while the current layer's attention runs. Cerebro's int8 mma prefill path plus a copy/compute overlap pipeline is the direct analog. Measurement: Monitor's PCIe chart should show steady streaming during prefill, not bursts (bursts = no overlap).

**50 — KV residency policy (VERIFIED repo claim).** Up to 64K context the KV stays in VRAM; beyond that only the most-read part is resident and the rest streams from RAM; `--kv-resident` flips this (attention's 32K window in VRAM, whole cache in pinned RAM — 1.6 GB at 128K). Cerebro should expose the same three-way choice (all-VRAM / windowed / streaming) with live KV occupancy in Monitor.

**51 — Batch slots and the honest measured trade (VERIFIED repo measurements).** On one RTX 5070 12 GB, Q2_0, 32K context: 4 concurrent requests through slots cut first-token wait from 11.2 s to 1.8 s but aggregate decode from 70.7 to 63.1 tok/s and per-request rate from 79.6 to 16.9 tok/s; a single request alone loses 11% (parallel 2) or 24% (parallel 4) because slot sessions (0.56 GiB each at 32K/8-bit KV) shrink the expert cache. This is the exact trade Cerebro's Turbo slider must express: **slots buy waiting time, not speed, on small-VRAM cards.**

**52 — Pipelined groups across GPUs (VERIFIED repo measurements).** With a layer split, batch groups form a pipeline across GPUs: 8 clients, total tok/s 166 (4×R9700) vs 86 with one group; 109 vs 78 (2 GPUs); 110 vs 78 (3 GPUs). Single-GPU Cerebro can still pipeline the *copy* stage with the compute stage; the product lesson is to name the mode ("throughput" vs "lowest latency") instead of exposing `--batch-groups`.

**53 — Slot as conversation cache + parking (VERIFIED repo claims/measurements).** A finished slot keeps prompt, answer, and the turn-boundary checkpoint; next turn reuses it (50–60 ms copy-back); a parking cache with 4 slots made follow-up turns 0.53/0.46 s to first token vs 3.9/5.7 s cold. Orchestrate mode should reuse this: the manager's session is the canonical parked conversation.

**54 — Cooperative preemption and decode-share (VERIFIED repo mechanisms).** Long prompts yield to short ones at chunk boundaries (`BYIELD`, `YIELDED` protocol lines); between prompt chunks the slots decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5). Cerebro should copy both numbers as defaults — they are the cheapest known anti-head-of-line-blocking recipe.

**55 — Adaptive expert cache with quantities to beat (VERIFIED repo measurements).** Tier adapts every 4 windows by default (`--adapt-every`), swapping most-routed non-resident experts for least-routed residents, with a min-gain threshold (default 1.5×). Measured on 2×RTX 3080: async adapt 89.2 aggregate tok/s, routed served from VRAM 92.1%; blocking tier 74.7; frozen learned cache 69.5 at 77.2%; no tier 45.2 at 52.7%. Cerebro's expert scheduler should target the same public metrics: **% routed experts served from VRAM** and **adapt stall per window** (Strata logs `adapt wait`; their async version makes it 0.2 ms).

**56 — Windows-specific wins (VERIFIED repo claims).** `STRATA_PIN_BY_NEED=1`: Windows can lock only about half the RAM for the GPU under WDDM, so pin the experts the card actually needs first — measured +4.4% prompt reading on an RTX 5070. `--adapt-async 1`: +4–5% decode, +7–10% on code/long documents. These are precisely the kinds of Windows-only optimizations Cerebro should ship on by default and verify on 12900H + 3070 Ti Laptop.

**57 — Exactness knobs for benchmarking (VERIFIED repo mechanisms).** To make batch-vs-solo token comparisons exact, Strata requires `STRATA_IQ_MT_MIN=1`, `--pcie-frac 0`, `--adapt-every 1000000`, `--no-prefill-borrow` because expert rows round differently when their batch group size changes. Cerebro's own A/B benchmarks need the same class of "freeze the adaptive tiers" flags or every measurement is confounded. This is a product feature, not just a dev tool: a "Benchmark lock" switch (item 108).

**58 — Strata match-or-beat checklist (DESIGN, targets).** Ship: 8k prefill chunks with copy/compute overlap; ≥80% routed-experts-from-VRAM steady state; adapt stalls <1 ms; prompt-lookup drafting on; slot parking for the manager session; batch-slots default off on 8 GB with a one-click "more responsive" toggle; a Monitor that reports hit rate, PCIe, and adapt wait. Beat: Windows pin-by-need automated per-machine (they need a config edit — VERIFIED: two opt-in JSON settings), and a heads-up alert when thermal throttle erases the gains of a tuning choice.

---

## Part 3 — The Monitor screen

**59 — Sampling principle (DESIGN).** Two cadences: **fast tier** (2–4 Hz: util, mem, tokens, queue, TTFT deltas) and **slow tier** (0.5–1 Hz: power, clocks, throttle bits, PCIe, thermals, disk). One sampler thread in Rust; ring buffers in shared state; the UI reads snapshots pushed over a Tauri `Channel` (not events-per-sample); the webview never calls sampling APIs directly. Failure mode this prevents: UI jank during decode, the classic desktop-AI sin.

**60 — Core engine metrics (DESIGN).** Live prefill tok/s and decode tok/s (per stream and aggregate), tokens-in-flight, batch occupancy (rows used / max rows), queue depth by role, KV occupancy (tokens resident / capacity), slot states (idle/reading/decoding), acceptance rate (draft tokens accepted per verify window), expert hit rate, and per-request ttft_ms. Cerebro already has an engine — these are counters, not sampled quantities; emit them from the engine at 4 Hz and forward. Measurement of *prefill tok/s*: `prompt_tokens / ttft`; *decode tok/s*: `(n_out - 1) / (t_last - t_first)`; both per request, then aggregated as median over active requests. Report p50 and p95, never mean alone.

**61 — TTFT with a queue/wait split (DESIGN).** TTFT = queue_wait + admission + prefill + sampling-to-first-token. Chart a stacked bar of the split for the worst request right now, plus a histogram over the last 5 minutes. This single chart answers 90% of "why is it slow" tickets. Measure: timestamps at each stage boundary in the engine; assert their sum equals measured TTFT within 2 ms in a test.

**62 — Memory tiers (DESIGN).** VRAM breakdown: weights (dense), expert cache (resident experts), KV cache (per slot), draft/verify buffers, compute workspace, everything-else (fragmentation + driver overhead). RAM: total, available, and **Cerebro-pinned** vs process working set (pinned host buffers matter for PCIe streaming). Disk: GB read/s from the GGUF (mmap page-ins) and SSD-cache hit behavior; page-file activity should be near zero — nonzero means the OS is paging and decode will collapse.

**63 — WDDM VRAM budget vs usage (DESIGN; API VERIFIED as public).** Query `IDXGIAdapter3::QueryVideoMemoryInfo` (`DXGI_MEMORY_SEGMENT_GROUP_LOCAL`, usage type `CURRENT_USAGE`) for Budget/CurrentUsage/AvailableForReservation. Show "usage / budget", not "usage / 8 GB": Windows may hand you less than the sticker VRAM when the desktop, browser, or another app holds memory — and it changes live. Failure mode: Cerebro plans expert cache from *total* VRAM and OOMs when the budget drops; plan from budget minus a reserve (Strata hardcodes ~5 GB reserve per card in its slot recommendation logic — VERIFIED behavior in BATCHING.md).

**64 — NVML: exact calls and cadence (DESIGN).** (docs.nvidia.com/deploy/nvml-api)

```text
4 Hz  nvmlDeviceGetUtilizationRates        # GPU + memory controller util
4 Hz  nvmlDeviceGetMemoryInfo_v2           # VRAM used/total/reserved
2 Hz  nvmlDeviceGetTemperature(gpu)        # memory-junction where supported
2 Hz  nvmlDeviceGetPowerUsage              # mW
2 Hz  nvmlDeviceGetEnforcedPowerLimit      # mW — show limit, not just draw
2 Hz  nvmlDeviceGetClockInfo(sm/mem)       # MHz
1 Hz  nvmlDeviceGetCurrentClocksThrottleReasons
1 Hz  nvmlDeviceGetPcieThroughput(tx/rx)   # KB/s
```

Each call costs roughly tens of microseconds when batched on one NVML handle; budget total monitor cost <0.5% of one CPU core. Measure the monitor itself: a hidden diagnostics chart of sampler-thread CPU time per tick (item 76).

**65 — Prefer NVML events to polling where possible (DESIGN).** `nvmlDeviceRegisterEvents` supports Xid and clock-change events (`nvmlDeviceGetSupportedEventTypes`); subscribe for Xid (GPU errors/TDR-adjacent), ignore per-frame polling for anything event-shaped. This is how you catch a driver-level stall with the exact timestamp instead of discovering it as a mystery TTFT spike.

**66 — Expert hit rate: definition and instrumentation (DESIGN).** Per decode step, for each MoE layer: `hits += 1` if the routed expert's weights were resident in VRAM at use time; `miss` otherwise (computed on CPU or streamed). Report global hit rate = hits/(hits+misses), per-layer heat strip (48 layers × colors), and **bytes moved per miss** (weights + eviction cost). Add swap counters (experts moved VRAM↔RAM per second) and adapt-stall ms per window (the Strata `adapt wait` analog, item 55). Measure: correlate hit rate vs decode tok/s over a session; a good cache shows a rising hit rate in the first 2–3 minutes of a repeated workload (Strata observes exactly this learning behavior — VERIFIED repo claim).

**67 — PCIe (DESIGN).** NVML gives TX/RX bytes/s; also `nvmlDeviceGetPcieReplayCounter` (errors) and link generation/width (`nvmlDeviceGetCurrPcieLinkGeneration/Width`) — on laptop GA104 expect Gen4 x8 (verify with GPU-Z; UNVERIFIED for this specific SKU). Chart PCIe RX during prefill (expert streaming) and correlate spikes with hit-rate dips. Warning rule: sustained RX > ~12 GB/s with hit rate <70% means the offload split is mis-tuned — surface "reduce experts on GPU or lower context" advice.

**68 — CPU load: cheap and exact (DESIGN).** Totals via `GetSystemTimes` deltas (`cpu% = 1 - idleΔ/(kernelΔ+userΔ)`) — verified stable Win32 API, ~1 µs. Per-core via `NtQuerySystemInformation(SystemProcessorPerformanceInformation)` (undocumented but stable since XP; widely used — mark UNVERIFIED for future Windows builds) or PDH `\Processor Information(_Total)\% Processor Utility` (better than `% Processor Time` on modern Windows because it accounts for frequency scaling). Show per-core heat a P/E-aware strip (6P + 8E on the 12900H) — users immediately see when decode threads spilled onto E-cores, which on this CPU costs 30–50% on matmul (estimate; measure during tuning).

**69 — PDH: the right way (DESIGN).** Open one query per group of counters; `PdhCollectQueryData` at tier cadence; `PdhGetFormattedCounterValue` with `PDH_FMT_DOUBLE`; treat first-sample `PDH_INVALID_DATA` as "no data". Counter paths to ship:

```text
\GPU Engine(*engtype_Compute)\Utilization Percentage
\GPU Adapter Memory(*)\Dedicated Usage
\Process(cerebro-engine)\Working Set
\Processor Information(_Total)\% Processor Utility
\PhysicalDisk(_Total)\Disk Read Bytes/sec
\Memory\Available MBytes
\Paging File(_Total)\% Usage
```

Never use WMI for live telemetry (COM overhead, hibernation stalls, handle leaks). `Collect` cost is tens–hundreds of µs per query; one query, many counters.

**70 — GPU engine counters for attribution (DESIGN).** `\GPU Engine(*engtype_Compute)\Utilization Percentage` per engine instance distinguishes Cerebro's compute engtype from browsers' 3D — the honest answer to "the GPU says 90% but Cerebro says idle" (someone else's app is using it; show it). Per-process VRAM attribution on Windows: `nvmlDeviceGetComputeRunningProcesses` covers compute contexts (works on Windows for CUDA processes; UNVERIFIED for graphics contexts) and PDH `\GPU Process Memory(pid_*)\Local Usage` complements it.

**71 — Thermals (DESIGN).** GPU core/mem via NVML. CPU package + SSD via LibreHardwareMonitor (github.com/LibreHardwareMonitor/LibreHardwareMonitor; requires its kernel driver — bundle carefully, mark UNVERIFIED for its current signing/distribution status on Win11) or HWiNFO's shared-memory API (hwinfo.com) where installed. Laptop ECs vary; treat missing sensor as "unavailable," never show a fake zero. Throttle taxonomy: NVML throttle-reason bits (idle, app clocks, sw power cap, hw slowdown, hw thermal, hw power brake, display clock) shown as a colored ribbon under the throughput chart; Intel side, detect PL2 taper by watching sustained all-core frequency collapse 2–5 s into a sweep.

**72 — Power and energy (DESIGN).** GPU watts from NVML; CPU watts via LHM where available (Intel Power Gadget is discontinued — do not depend on it, UNVERIFIED project state); else estimate `P ≈ P_idle + (P_max−P_idle)×load_fraction` for a fallback bar clearly labeled "estimate". Integrate power into energy per session, per request, and per solved problem (the Orchestrate currency, item 38). These numbers make "fastest local AI platform" falsifiable: tokens per joule at a fixed quality floor.

**73 — Throttle & power-limit timeline (DESIGN).** A single timeline with: SM clock, power draw vs enforced limit, temperature, and the throttle ribbon — the four signals users actually need to understand "why did it slow down at minute 3". On this laptop expect a pattern: PL2 burst ~2–5 s, then PL1 plateau, GPU at 80–115 W depending on Dynamic Boost and AC state (UNVERIFIED specifics per chassis vendor). Monitor should learn each machine's sustained envelope during onboarding and draw the *sustained* target line, not the peak.

**74 — Fault signals (DESIGN).** TDR/watchdog: WDDM resets a hung GPU after ~2 s by default (learn.microsoft.com "Timeout Detection and Recovery"); register an engine-side watchdog that logs the last 3 s of monitor snapshots when a CUDA call exceeds 1.5 s. Warnings: NVML Xid events, engine crash event, model-file hash mismatch on load. Every monitor snapshot ring buffer is dumped to the crash report (item 146).

**75 — Sampling architecture and cost measurement (DESIGN).**

```rust
// one OS thread, fixed cadence, no allocation after init
let fast = RingBuf::new(4 * 3600); // 1 h @ 4 Hz
let slow = RingBuf::new(1 * 3600);
loop {
  let t0 = Instant::now();
  let s = sample_all(&mut self.state);   // NVML + PDH + GetSystemTimes
  self.fast.push(s.fast()); self.slow.push(s.slow());
  if self.tick % 8 == 0 { channel.send(Frame::Snapshot(s)); } // 0.5 Hz UI push
  self.cost.push(t0.elapsed());
  sleep_until(next_tick());
}
```

Push full snapshots at 0.5 Hz for panels plus deltas at 4 Hz for the one live chart that needs it; the UI must never request data — backpressure and focus-pause come free (items 76, 127).

**76 — Overhead budget and proof (DESIGN).** Target: sampler <0.3% of one CPU core total; no GPU work. Prove it: a self-test that runs the sampler alone for 60 s and asserts median tick cost < 1.5 ms with all counters enabled; run in CI on the reference laptop. If a counter is expensive (per-process GPU), demote it to the slow tier.

**77 — Charts: engine choice (DESIGN).** uPlot (github.com/leeoniya/uPlot, canvas-based) fed by min/max decimated ring data: pre-decimate to one pixel bucket = 2 points (min,max). Re-render at 4 Hz max; never re-render on every sample. Resize via ResizeObserver, handle devicePixelRatio (item 138). Avoid SVG-per-point and avoid re-creating chart instances in SolidJS — create once in `onMount`, update via `setData`.

**78 — Chart set A: throughput timeline (DESIGN).** Left axis decode tok/s (line), right axis prefill tok/s (line, often 10–100× scale — log or dual axis), shaded band = batch occupancy %, markers = request start/end. This chart is the product's signature: it visibly shows batching, chunked prefill, and drafting working (or not).

**79 — Chart set B: memory tiers (DESIGN).** Stacked area: VRAM = dense + experts + KV + draft + workspace, with the WDDM budget as a dashed ceiling; second stacked area: RAM pinned / process working set / available; disk read rate bottom strip. Overlay: expert hit rate on the same x-axis — the correlation is the insight.

**80 — Chart set C: latency and routing heat (DESIGN).** TTFT histogram (last 5 min, split stacked by queue vs prefill), queue-depth line by role, and the 48-layer expert hit heat strip. The heat strip catches layer-specific offload mistakes (e.g., a mistakenly CPU-placed hot layer shows as a cold row).

**81 — Chart set D: thermals, power, throttle (DESIGN).** SM clock + power + temps on one timeline; throttle ribbon as a color-coded strip (green none, yellow power-cap, red thermal). Correlate with a "tokens lost to throttling" counter: estimated `∫(clock_peak − clock_actual)/clock_peak × tok/s dt` — a number users immediately understand and act on.

**82 — Chart set E: PCIe and swaps (DESIGN).** TX/RX area chart with expert-swap rate and adapt-stall annotations. Healthy steady state: modest RX during prefill, near-zero during pure decode if experts fit; orchestrate workloads with N problems will show constant moderate RX — fine. Warning: replay-counter growth.

**83 — Alerts with hysteresis (DESIGN).** Rules (all with 5–10 s hysteresis and one-per-session dedupe): thermal throttle active >10 s; VRAM usage > 95% of budget for >5 s; hit rate < 60% after 3 min warm; page file bytes/s > 1 MB/s; PCIe replay counter growing; TTFT p95 > 3× user-configured ceiling. Each alert offers one action button opening the relevant Turbo switch.

**84 — Export and replay (DESIGN).** "Save session" writes monitor JSONL + engine ledger + config to a zip; a lightweight replay view scrubs the session with charts synchronized to ledger events (task.assign, verify.fail, throttle start). This is both a support tool and a benchmark artifact: the same zip format feeds the auto-tune evidence (item 121).

**85 — Laptop-specific indicators (DESIGN).** AC vs battery (powercfg battery status / `GetSystemPowerStatus`), dGPU power ceiling indicator (draw vs enforced limit — on battery expect a hard cap), MUX/dGPU-direct mode note (advertise the expected mismatch in external-display vs internal-panel performance), Dynamic Boost state (UNVERIFIED API availability — likely only inferable from watts). Show "power ceiling reached" instead of pretending there is more performance available.

---

## Part 4 — Turbo: the switches every power user wants

**86 — Sampler presets (DESIGN).** Four named presets, each a full bundle, plus Expert mode exposing every parameter:

| Preset | temp | top-p | top-k | min-p | repeat | notes |
|---|---|---|---|---|---|---|
| Precise (code) | 0.2 | 0.95 | 40 | 0.05 | 1.0 | matches GVS5H execute temp 0.2 (VERIFIED default there) |
| Balanced | 0.7 | 0.95 | 40 | 0.05 | 1.05 | chat default |
| Creative | 1.0 | 0.98 | 80 | 0.02 | 1.1 | brainstorm/temps 0.4+ analog |
| Deterministic | 0.0 | — | — | — | 1.0 | seed pinned; enables exact A/B tests (item 57) |

Beginner text: "How adventurous the model is." Advanced: expose seed (64-bit), DRY/XTC-style penalties if implemented, and per-preset overrides for Orchestrate roles (item 22).

**87 — Context size (DESIGN).** Slider 4K–128K with a live VRAM delta estimate. Formula: `KV bytes/token = 2 × n_kv_heads × head_dim × n_attn_layers × bytes_per_element` (+ recurrent state size per GDN layer for the hybrid). Example (illustrative config, VERIFY against the actual GGUF metadata): 48 layers, GQA 4 heads × 128 dim → 98 KB/token at f16 → 32K = 3.1 GB, 128K = 12.6 GB — which is exactly why KV quant and windowed residency exist. Beginner: "How much text the model remembers." Show the trade as a sentence: "128K context costs ~X GB of your 8 GB."

**88 — KV quant (DESIGN).** f16 → q8_0 (≈2× saving, near-lossless in practice) → q4_0 (≈4×, visible degradation on long context, especially V). Ship q8_0 as the default for headroom above 16K, keep K at higher precision than V when split-allowed (quality asymmetry is well-established practice; UNVERIFIED as a controlled measurement on this exact model — gate it behind the quality harness in item 109). Measure: long-context needle test + KL on a 16-prompt calibration set vs f16.

**89 — Offload split (DESIGN).** For MoE: an "expert budget" control that allocates VRAM after dense/KV/draft overhead, computing expected expert slots at ~700/GB (Strata's number, VERIFIED as repo claim) and showing projected hit rate from the last session's routing histogram. Beginner: "More GPU = faster, if it fits." Advanced: per-layer tensor override and `--n-cpu-moe`-style knobs; auto-recommend split from the onboarding sweep. Failure mode: budget computed from total VRAM while the WDDM budget is lower — compute from `QueryVideoMemoryInfo` budget minus reserve (item 63).

**90 — Flash attention / prefill kernel choice (DESIGN).** Cerebro has f16 flash attention and int8 mma.sync m16n8k32 prefill. Expose: `prefill: int8-fastest | f16-exactish`, `decode-attn: f16` — the int8 prefill is the TTFT lever (2×+ on prefill in the best case; measure per driver/model). Quality gate this switch with the item-109 KL test; int8 prefill errors compound in long chains (orchestrate!), so verify per model once and cache the verdict in the profile.

**91 — Threads and affinity (DESIGN).** Default `threads = P-cores` (6 on the 12900H), decode pinned to P-cores via `SetThreadAffinityMask`, E-cores left for PCIe copies/IO (they are slower per-core and share L2 in clusters). Advanced: separate `prefill_threads` and `decode_threads`, and a "hybrid-aware" toggle. Failure mode: 20 threads thrashing E-cores/HT and *slowing* decode 10–30% (estimate; measure in sweep). Measure: sweep threads ∈ {4,6,8,10,12,14,16} × {P-only, all} at fixed context; keep the median-of-3 winner.

**92 — Priority class and scheduler citizenship (DESIGN, exact APIs).** `SetPriorityClass(HIGH_PRIORITY_CLASS)` for the engine process (NOT realtime — starves the compositor), `SetThreadPriority(THREAD_PRIORITY_ABOVE_NORMAL)` for decode threads, **disable EcoQoS** via `SetProcessInformation(ProcessPowerThrottling)` with `PROCESS_POWER_THROTTLING_EXECUTION_SPEED` off — this alone can 2×-difference background-launched processes when Windows decides to throttle. Optional: MMCSS `AvSetMmThreadCharacteristics("Games")` for the decode thread for scheduled priority. Beginner: "Give Cerebro priority over background apps."

**93 — Power plans and power mode (DESIGN).** `powercfg /setactive SCHEME_MIN` (High performance) is blunt; better: toggle the Windows "power mode" overlay (Best performance) while the engine runs, restore on exit; expose the Ultimate Performance scheme GUID creation (`powercfg -duplicatescheme e9a42b02-...`) as an advanced button. Also set per-app GPU preference ("High performance") in the NVIDIA App/driver settings. Beginner: "Maximum performance while generating."

**94 — GPU clocks and power limits (DESIGN, risk-flagged).** "Prefer maximum performance" via driver settings is safe; clock-locking via NVML (`nvmlDeviceSetGpuLockedClocks` needs admin; UNVERIFIED behavior on laptop driver stacks) or nvidia-smi is advanced, non-persistent, and risky on a laptop's thermal envelope. Ship the driver-setting path, mention locking as UNSUPPORTED. Warning text: locking clocks on battery/Dynamic Boost laptops usually *reduces* total throughput by forcing a worse thermal operating point.

**95 — Disk and memory placement (DESIGN).** Keep GGUF on NVMe with >50 GB free (SLC cache), enable `mmap` with a preload/warmup pass (count sequential read at startup and show it), set the data dir out of OneDrive/Dropbox sync (item 142), and pin host buffers for PCIe streaming via `cudaHostAlloc` (advanced). Beginner: "Put models on your fastest drive." Measure: startup-to-first-token time and disk read rate in Monitor.

**96 — Drafting switches (DESIGN).** Enable/disable: MTP head (if model has one), external draft model (pick file), prompt-lookup (item 48). Targets to display: acceptance rate, tokens/pass, and net tok/s delta vs drafting-off measured this session — never let a draft path silently lose (Strata measured MTP-batching can be *net negative* in some configs: -21.9% on a 2×R9700 all-in-VRAM setup — VERIFIED repo measurement). Auto-disable drafting when measured net delta < 0 for 60 s.

**97 — Parallel/slots (DESIGN).** Slider 1–8 with the honest caption from Strata's own data (VERIFIED): extra slots cut waiting time and per-request speed on small-VRAM cards, add speed only when experts mostly fit in VRAM. Default on this laptop: 1 for single-user interactive; Orchestrate mode automatically raises it to 2–4 while active.

**98 — Advanced misc (DESIGN).** ubatch (128–1024; 512 default), `mlock`/pin weights (careful: 32 GB machine, 30B Q4 ≈ 18 GB — pinning everything starves the OS; pin experts-only per item 56), PCIe share controls and prefill-borrow analogs (item 54), and a "freeze adaptive tiers" benchmark lock (item 57). Every advanced knob gets an inline one-line failure-mode tooltip: "toast: lowering ubatch increases scheduler overhead."

**99 — Two-tier UI pattern (DESIGN).** Turbo screen = Beginner layer (5 presets: Fast / Balanced / Quality / Long-context / Max-concurrency) + Advanced drawer (everything above), with an impact preview ("expected: +18% decode, −2 GB KV") computed from this machine's own sweep data, and one-click revert to last-known-good. No switch without (a) plain-language explanation, (b) expected effect, (c) reset path.

**100–106 — Per-switch failure modes and 60-second verification (DESIGN).** For each of ctx size, KV quant, offload split, threads, priority, power mode, drafting: ship a **micro-benchmark button** that A/Bs the switch in <60 s (256-token decode + 8K prefill at fixed seed) and reports the measured delta plus the quality gate result. This converts every power-user tweak from folklore into evidence and produces the training data for auto-tune (item 121). Failure-mode table to encode: ctx too big → KV spills, decode collapses; KV q4 → long-context drift (run needle test); split too GPU-heavy → OOM on next driver update; threads too high → thermal throttle earlier; priority realtime → UI freeze; drafting on non-repetitive text → net slowdown.

---

## Part 5 — Onboarding and per-machine auto-tuning

**107 — First-run flow (DESIGN).** Steps: (1) hardware detect: GPU via NVML (`nvmlDeviceGetName`, VRAM, CC), CPU model/cores via `GetNativeSystemInfo` + registry, RAM via `GlobalMemoryStatusEx`, disk type via `Get-PhysicalDisk` (DeviceMediaType/SSD flag), driver version via NVML/registry. (2) Scan/import model files with GGUF metadata parsing (arch, layers, kv heads, expert count — spec: github.com/ggml-org/ggml/blob/master/docs/gguf.md). (3) "Will it fit" math (item 115). (4) 3–10 minute benchmark sweep (items 108–113). (5) Save profile + benchmark evidence + a shareable report (Strata's `BENCH.bat` community-report pattern is the model to copy — VERIFIED that they ship exactly this).

**108 — Sweep design (DESIGN).** Search space, ordered by impact: ① n_gpu_layers / expert budget (5 values), ② KV type {f16, q8_0} × ctx {8K, 32K, 64K} (subset), ③ ubatch {256, 512, 1024}, ④ threads {P-cores, P-cores+2, all}, ⑤ drafting {on, off}. Use **successive halving**: round 1 = 12 configs × 8 s each (warmup + 64-token decode + 2K prefill); keep top 5; round 2 = 30 s each (128 tokens + 8K prefill); keep top 2; round 3 = 60 s verification of the winner. Hard budget 10 minutes; thermal guard pauses 20 s whenever any temp > 85 °C (item 111). Benchmark-lock flags (item 57) on for every run; record the lock state in the profile.

**109 — Objective and quality gate (DESIGN).** Objective: `score = decode_tps@8K × w1 + prefill_tps@8K × w2 − OOM/variance penalties`, with w defaults 0.7/0.3 (interactive bias). Quality gate (per model, once): KL divergence of next-token distribution vs the f16/no-offload reference on 16 fixed calibration prompts (measure mean KL of top-64 logits; threshold ≈ 0.02, tuned once), plus 20-prompt smoke suite with exact-match checks for deterministic prompts. Any config failing the gate is excluded regardless of speed — this makes "auto" trustworthy.

**110 — Sweep algorithm pseudocode (DESIGN).**

```python
def sweep(hw, model):
    cfgs = coarse_grid(hw, model)                  # ~12 configs
    results = []
    for c in cfgs:
        if thermal_hot(): cooldown(20)
        run_warmup(c, 8)                           # 2 iterations, discard
        t = median([run(c, seconds=8) for _ in range(3)])
        results.append((score(c, t), c, t))
    for rounds in (5, 2):
        if budget_exhausted(): break
        keep = top(results, rounds)
        results = [refine_and_rereun(c, longer=rounds_time()) for _, c, _ in keep]
    best = argmax(results)
    verify_quality(best)                           # KL + smoke, once
    save_profile(hw_fingerprint(hw), best)
```

**111 — Confound control (DESIGN).** Sweep refuses to start on battery; asks user to close GPU apps (check per-process GPU utilization via PDH); aborts and retries later if any other process >10% compute engine for >10 s; discards a config if throttle reasons were non-zero >20% of its run (retry once after cooldown). Log all of it into the profile so a weird result can be explained, not just observed.

**112 — Profile artifact (DESIGN).**

```json
{"schema":1,"fingerprint":"sha1:gpu|driver|cpu|ram|disk",
 "engine":"cerebro 0.4.2+cuda86","model":"Qwen3-Coder-30B-A3B-Q4_K_M",
 "config":{"n_cpu_moe":28,"kv_type":"q8_0","ctx":32768,"ubatch":512,"threads":12,
           "flash":"int8-prefill","slots":2,"draft":"mtp"},
 "measured":{"decode_tps":18.7,"prefill_tps":410,"ttft_ms_8k":19500,"joules_per_token":0.041},
 "quality":{"kl_vs_f16":0.0031,"smoke":"20/20"},"bench_lock":true,"date":"2026-10-17"}
```

Profiles are versioned; re-tune triggers: driver change, engine update changing kernels, model file change, or user "feels slow" report with monitor evidence.

**113 — Per-architecture tuning notes (DESIGN).** Dense Qwen3: n_gpu_layers + KV quant are the whole game. MoE (Coder-30B-A3B):
