# GVS5H — findings

**What I read (2026-10-19).** The repo [README](https://github.com/slee-persis/GVS5H), the v2 source ([`multiagent.py`](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/multiagent.py), [`orchestrator.py`](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/orchestrator.py), [`run_bench.py`](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/run_bench.py)), the v1 source, the [paper v2](https://arxiv.org/html/2608.26480v2) and [paper v1](https://arxiv.org/html/2608.26480v1). **Bottom line:** there is **no ablation by instance count, role set, ledger format, or stopping rule** in either paper version or the repo. The only scaffold comparison is v1 vs v2, five bundled changes, confounded by serving path. Anything below labelled **UNVERIFIED** was not reported or could not be found.

## 1. The exact loop

One model, fresh context per call. `MULTIAGENT_MODEL` selects it (default `groq:qwen/qwen3.6-27b`); all roles are the same model. `MAX_ITERS=10`, `MAX_TASKS=12`, plan capped at 4,000 chars, notes at 8,000 ([source L18, L34](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/multiagent.py#L18)). Calls are **sequential**; there is no in-problem thread pool.

| file | written by / when | read by |
|---|---|---|
| `task.md` | harness at start | **never read by agents**; the problem is passed as a Python string |
| `plan.md` | manager-plan, after its call | brainstorm, manager-manage, worker |
| `tasks.json` | harness after every manage (`{id,desc,status,result}`) | **never read by agents**; debug mirror only |
| `notes.md` | brainstorm appends; every worker **replaces** it | manager-manage and worker |
| `solution.py` | worker (code mode) | manager-manage, worker; executed by verifier |
| `answer.md` | worker (math mode only) | manager-manage, worker |
| `transcript.jsonl` | harness records every request/response/token | not read by agents; used for analysis |

Exact control flow ([`multiagent_solve`, L412-476](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/multiagent.py#L412-L476)):

1. Harness creates `<ws>/<md5(problem)[:12]>` and empties the files; writes `task.md`.
2. **Manager — plan** (temp 0.3): reads only the problem; writes a 3-6 sentence `plan.md` and returns 3-6 seed tasks. These tasks are not yet saved to `tasks.json`.
3. **Worker — brainstorm** (temp 0.4): gets `problem + plan.md`; must not solve or write code; appends `## ideation` to `notes.md`; returns distinct approaches.
4. **Manager — manage** (temp 0.2): gets problem, current solution/answer, all notes, the task list, last result, and proposals; returns `STATUS` (done/continue), one `NEXT` task, and a curated task list; harness saves `tasks.json`.
5. **Worker — execute** (temp 0.2): gets problem, plan, notes, current artifact, and exactly one task; writes `solution.py` and **rewrites** `notes.md` (not append); returns next steps and its own status.
6. **Verifier — no model**: if code was written and public stdin tests exist, runs `solution.py` as a subprocess with a 10 s timeout per test; prepends PASS/FAIL + first failing case to the summary fed back to the manager. A failure **overrides** `STATUS: done`.
7. Back to step 4. Stops when the manager says done with a usable artifact, when `MAX_ITERS=10` is reached, or when the manager reissues the exact same task (no-progress guard). If not cleanly done, a **finalizer** worker is called with goal “Produce the DEFINITIVE final solution now…” and task name `finalize`.

Pseudocode: `plan(); brainstorm(); status,nxt,tasks=manage("ideation complete"); while status=="continue" and nxt and i<10: if nxt==prev: break; worker(nxt); verifier?; status,nxt,tasks=manage(); if verify_failed and status=="done": status="continue"; i+=1; if not(status=="done" and has_answer()): finalize()`.

**How many instances?** One model instance; one worker call at a time. The loop is manager→worker→manager→worker, not N agents voting. `run_bench.py --parallel 8` parallelizes **different problems**, not roles ([run_bench.py L21-38](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/run_bench.py#L21-L38)). The paper’s “five passes” are five **independent evaluation runs** per condition, not five collaborating instances. The “5” in GVS5H (“Gorilla vs. 5 Humans”, five boys on shoulders) is branding. **No 1/3/5/10-instance sweep exists.**

## 2. Prompts per role (verbatim)

**Manager — plan** (system; user = problem only):
> You are the PRIMARY orchestrator (manager) of a small team of workers, all expert at competitive programming. Given a problem, produce a short overarching plan to solve it, then a task list the workers can pick up. Respond with EXACTLY these sections:
> ### PLAN
> <3-6 sentence strategy>
> ### TASKS
> <3-6 bullet tasks, each a concrete unit of work>

**Worker — brainstorm** (system; user = `PROBLEM:\n{problem}\n\nPLAN:\n{plan.md}`):
> You are the FIRST WORKER. Do NOT solve the problem and do NOT write any code. Just think about it: identify the core difficulty, then list SEVERAL DISTINCT candidate approaches (genuinely different algorithms / data structures / problem reductions, not variations of one idea), and note pitfalls for each. Describe each approach in prose only -- absolutely no code blocks; a later worker will implement. Respond with EXACTLY:
> ### NOTES
> <your analysis>
> ### NEXT
> <bullet list of distinct approaches to try next>

**Manager — manage** (system; user = problem, current solution, notes, task list, last result, proposals):
> You are the PRIMARY orchestrator and manager. You OWN the task list and decide when the problem is solved. Review the current progress and the latest worker's result, then:
> - The LATEST WORKER RESULT may include a SAMPLE TESTS verdict from actually running the code. Treat it as ground truth: only set STATUS 'done' if the solution PASSED the sample tests; if it FAILED, you MUST set STATUS 'continue' and choose a task that fixes the failing case or switches to a different approach.
> - If the current solution/answer is complete and correct, set STATUS to 'done'.
> - Otherwise CURATE the task list: merge duplicates, drop finished or irrelevant items, mark completed ones [done], and fold in ONLY genuinely new sub-tasks from the proposals. Then choose the single most valuable next task.
> - IMPORTANT: if the current solution keeps failing, or the last worker made no real progress, do NOT keep refining the same idea. Switch to a DIFFERENT approach (a different algorithm / data structure / reduction) from the notes, or ask for a new one. You have many rounds -- use them to try distinct approaches, not to polish a stuck one.
> Respond with EXACTLY these sections:
> ### STATUS
> <done|continue>
> ### NEXT
> <exact text of the ONE task to do next; omit if done>
> ### TASKS
> <curated list, one per line, each '- [done] ...' or '- [todo] ...'>

**Worker — execute / finalizer** (system; `{solver_system}` is the code baseline prompt below; user = `PROBLEM…\n\nPLAN…\n\nNOTES…\n\nCURRENT WORK…\n\nYOUR TASK: {goal}`; normal goal `Complete this task: {task.desc}`, final goal `Produce the DEFINITIVE final solution now, using all notes and current work.`):
> You are a WORKER subagent, You are an elite competitive programmer. Solve the given problem in Python. Think carefully about algorithmic complexity and edge cases. Output EXACTLY ONE complete, self-contained Python program inside a single ```python ...``` fenced block, and nothing else after it. You share a workspace with the team. Build on the current work and notes where useful -- but if your task is to try a different approach, write a FRESH solution for that approach instead of patching the stuck one. Respond with EXACTLY these sections:
> ### CODE
> ```python
> <the FULL updated self-contained program>
> ```
> ### NOTES
> <the COMPLETE notes file, rewritten. You are shown the current NOTES above: fold your findings into them, keep what still matters, and DELETE anything superseded, disproven, or now obvious. This REPLACES the file, so whatever you omit is gone. Organise it as '- **Topic:** ...' bullets, under ~800 words. Do NOT use markdown headings (#, ##), bold-only lines, or ALLCAPS: lines anywhere inside this section -- the reply is split on those, so they would truncate your notes.>
> ### NEXT
> <bullet list of remaining steps, or 'none'>
> ### STATUS
> <solved|continue>

**Cut-off summarizer** (fresh call when a worker hits the token cap; system; user = `TASK: …\n\nCUT-OFF ATTEMPT:\n…`):
> A worker's solution attempt was CUT OFF when it hit the token limit. Summarize its partial attempt in 3-5 sentences: which approach it was pursuing, what it established or ruled out, how far it got, and what remained unfinished. Be concrete so another worker can resume or judge it. Do NOT try to finish the solution yourself.

**Verifier** (no prompt): feedback strings are `[SAMPLE TESTS: PASSED all {n} public samples -- the solution looks correct.]` or `[SAMPLE TESTS: FAILED -- passed {p}/{t}. The current solution is WRONG. First failing case: input=… expected=… got=…. Fix the bug or, if this approach keeps failing, switch to a DIFFERENT approach.]`. It only covers **73/100** problems with stdin public tests; the 27 LeetCode-style problems get no verifier signal.

**Single-call baseline**: exactly the `solver_system` text above, one user message = problem, temp 0.2 ([`orchestrator.py` L387-394](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/orchestrator.py#L387-L394)). The optional `_format_rule` is appended only when the model name contains “muse” or “glimmer” (or `MULTIAGENT_STRICT_FORMAT=1`); it is not part of the Qwen/GPT runs.

## 3. Measured gains, token cost, latency

**Gains by instance count: none exists.** No experiment varies the number of agents/instances. If by “instance count” you mean the five **passes**, Table 1 reports only the five-pass mean; there is no gain-vs-N curve. The code cannot run five workers in parallel: one `_worker` call per loop iteration.

Pinned-backend v2 results, LCB-100, 128k cap, reasoning on, five passes ([paper v2 Table 1](https://arxiv.org/html/2608.26480v2#S3.T1)):

| model | single call | with manager | Δ |
|---|---:|---:|---:|
| Claude Fable 5 | 90.4 ± 1.4 | — (single-only) | — |
| GPT-5.6-Terra | 80.8 ± 1.0 | 88.0 ± 1.2 | +7.2 |
| GPT-5.6-Luna | 70.4 ± 4.6 | 81.2 ± 2.8 | +10.8 |
| Qwen3.8-27B | 66.8 ± 5.2 | 92.4 ± 2.3 | **+25.6** |
| Qwen3.8-Flash-Next | 84.2 ± 1.8 | 93.0 ± 1.2 | +8.8 |
| DeepSeek-V4.1-Flash | 88.4 ± 2.6 | 92.0 ± 1.8 | +3.6 |

All five manager deltas are significant (p ≤ 0.002; four at <2.5e-5). OpenRouter-served one-pass set: Opus-5 88→95, Kimi-K3 86→86, MiniMax-M3 62→69, Qwen3.6-35B 26→44; Qwen3.5-9B reasoning-on was unusable. Reasoning-off, 16k, five passes: Kimi-K3 33.4→65.0 (+31.6), MiniMax-M3 22.2→33.2 (+11.0), Qwen3.5-9B 15.0→22.6 (+7.6), **Qwen3.6-35B 28.2→27.4 (−0.8)** — gains are not universal ([App. C, Table 6](https://arxiv.org/html/2608.26480v2#A3.T6)).

**Token/cost** ([paper v2 Table 2](https://arxiv.org/html/2608.26480v2#S3.T2)); list rates × actual tokens, retries billed:

| model | single $/pass | manager $/pass | output MTok, single→manager | $ multiplier |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 18.95 | 47.82 | 7.42→18.63 | 2.5× |
| Qwen3.8-Flash-Next | 2.73 | 5.76 | 5.78→11.84 | 2.1× |
| DeepSeek-V4.1-Flash | 3.46 | 15.43 | 2.86→12.49 | 4.5× |
| GPT-5.6-Luna | 0.41 | 1.50 | 0.33→1.05 | 3.7× |
| GPT-5.6-Terra | 3.41 | 11.71 | 0.27→0.79 | 3.4× |
| Claude Fable 5 | 61.11 | — | 1.20 | — |

The paper’s summary: output tokens rise to **about 3×** baseline and the scaffold **“roughly triples the bill”**; input tokens rise ~16-20× (0.07→1.1-1.5 MTok). Per-arm $ increase is 2.1-4.5×, so “3×” is an average, not a constant. Cost per solved problem also rises (e.g., Qwen 3.8-27B $0.28→$0.52; Flash-Next $0.032→$0.062).

**Latency: UNVERIFIED / not reported.** The paper, README, and result JSONs report no wall-clock time, per-call latency, or end-to-end duration. `--parallel 8` makes any wall-clock number depend on concurrency, and the paper never gives one. Arithmetic proxy from Table 3: manager arms used **3,141-3,525 model calls per 500 problem-passes**, i.e. **6.3-7.1 sequential calls per problem** versus 1 for the baseline (max possible ≈24: plan + brainstorm + 10×(manage+worker) + finalizer); actual time per call is provider-dependent and unmeasured.

## 4. Which parts matter most

**No component ablation exists, so no causal ranking is possible.** The only comparison is v1 vs v2, which changes **five things at once**: 4→10 rounds, adds the public-test verifier, adds the cut-off summarizer, notes append→rewrite, and adds 4k/8k file caps ([v2 §2.1](https://arxiv.org/html/2608.26480v2#S2.SS1)). v1 also ran on OpenRouter and v2 on pinned backends; the paper explicitly says the two sets are not directly comparable. You cannot attribute the gain to the verifier, the round budget, the notes format, or the caps from this design.

What the paper does support, without isolation:
- **Decomposition + persistent context** are its stated mechanisms, from a hand-picked transcript analysis (8 examples): committing the reduction/bounding argument to disk before code, splitting coupled objectives, and bounding each call so work reaches disk ([§4](https://arxiv.org/html/2608.26480v2#S4)). This is qualitative, not an ablation.
- **Truncation rescue** is measured for Qwen3.8-27B: 35/500 single-call cells had no code; the manager solved 28, failed 7, left 0 → **5.6 points**, ~22% of its +25.6 ([§3.3](https://arxiv.org/html/2608.26480v2#S3.SS3)). For models with no truncation (Terra, Luna) this channel is zero, so it cannot explain their gains.
- **Model condition is the strongest moderator**: gains are largest where the model cannot self-plan/stop — Kimi +31.6 with reasoning off, MiniMax +11, Qwen3.5 +7.6 — and negative for Qwen3.6-35B with reasoning off. This is a model×condition interaction, not a role ablation.
- **Verification is not proven to help.** It covers only 73/100 problems, the manager otherwise trusts worker self-reports, and Appendix B shows the harness’s `sys.stdin` mock could make the verifier certify a program the hidden grader fails. Its net contribution is unquantified and potentially negative.

## 5. What a chat app should copy (extrapolation; paper tested code only)

The paper’s own limitation: all results are code generation; math/knowledge benchmarks were abandoned (near-ceiling). So copy the **control pattern**, not the competitive-programming prompts, and treat gains as conditional.

- **Default to a single call; add a “deep work” mode.** The scaffold costs 2-4.5× and adds 6-7 sequential calls; it is not worth it for simple questions. Trigger it for long deliverables, multi-step research, code, or when the model is weak/ reasoning-disabled.
- **Use a small persistent ledger, not one growing chat transcript**: `plan.md` (3-6 sentence strategy + 3-6 tasks), `notes.md` (curated findings, rewritten each round), `draft.md` (current answer/code), and a hidden task list. Cap the files (the paper uses 4k/8k chars) and keep the task list to ~12.
- **Roles**: planner/manager, a brainstorm call that lists genuinely distinct approaches and **does not answer**, manager-curator that picks exactly one next task, one fresh executor per task, a verifier if a real oracle exists, and a finalizer. Same model is fine; fresh context per call is the point.
- **Run the roles in sequence, not as five parallel instances.** The evidence supports serial decomposition and context resets; it does **not** test N-sample voting or parallel agents on the same question. If you parallelize, parallelize independent sub-questions, not copies of the same one.
- **Make the manager curate, not accumulate.** Merge/drop tasks, mark done, add only genuinely new sub-tasks; rewrite notes and delete superseded material. This is what keeps context bounded and is the clearest transferable lesson.
- **Stop rules**: done only with a usable artifact; verifier failure overrides done; stop if the same task is reissued; cap rounds (10 in the paper, 3-5 is more sensible for chat latency); otherwise run a finalizer. Do not let the model declare done with no answer on disk.
- **Verification needs an oracle.** For general questions use citations, calculations, unit checks, or tool output; otherwise label uncertainty. Do not copy the “manager trusts the worker except for sample tests” design — the paper itself flags confidently-wrong propagation as a failure mode.
- **Implementation warning**: the repo parses Markdown headings (`### STATUS`, `### NEXT`, `### CODE`, `### NOTES`) and forbids `#` inside notes because the parser would split there. For a chat app, use a structured schema (JSON fields) rather than brittle heading parsing; the paper did not compare ledger formats.
- **Borrow the paper’s proposed fixes only as hypotheses**: fresh-perspective workers with no prior context, and generated test suites, are discussed as future work, **not implemented or measured**. They are plausible for anchoring/regression problems but not validated here.

**Sources:** [repo](https://github.com/slee-persis/GVS5H) · [v2 `multiagent.py`](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/multiagent.py) · [v2 `orchestrator.py`](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/orchestrator.py) · [v2 `run_bench.py`](https://github.com/slee-persis/GVS5H/blob/master/codebase/v2-current/escalation/run_bench.py) · [paper abs](https://arxiv.org/abs/2608.26480) · [paper v2 HTML](https://arxiv.org/html/2608.26480v2) · [paper v1 HTML](https://arxiv.org/html/2608.26480v1)
