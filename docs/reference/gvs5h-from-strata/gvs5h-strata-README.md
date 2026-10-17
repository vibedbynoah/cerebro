# GVS5H on Strata

Ledger-based zero-shot self-orchestration, ported from
[slee-persis/GVS5H](https://github.com/slee-persis/GVS5H) (MIT) to run against the
local Strata server on `127.0.0.1:8081`.

## What it actually is

Not a model upgrade. Every role is **the same model in a fresh context**, and they
coordinate only through files on disk:

| file | written by | read by |
| --- | --- | --- |
| `plan.md` | manager | manager, all workers |
| `notes.md` | workers (each rewrites it whole) | manager, all workers |
| `tasks.json` | manager | manager |
| `result.md` | workers | manager, all workers |
| `transcript.jsonl` | harness | you |

Because no context is carried in memory between calls, the ledger is the only
channel — which is what makes it training-free, and what the paper credits for the
gain (decomposition plus persistent context).

## Flow

```
plan  ->  ideate  ->  [ manage -> worker ] x N  ->  finalize
```

The manager curates the task list each round and picks exactly one next task. A
worker implements it, rewrites `notes.md`, and reports status. Repeat until the
manager says `done`, a task repeats with no progress, or `--iters` is hit. The
same task issued twice in a row stops the loop — that guard is from the original.

## Usage

```powershell
# simple
python gvs5h_strata.py "Add retry-with-backoff to Projects/throttle's HTTP client"

# with context and a verification command workers may run
python gvs5h_strata.py "Fix the flaky test in cerebro-core" `
  --context "Root: C:\Users\noahm\Projects\cerebro-core. Fails only on Windows." `
  --verify "python -m pytest tests/test_x.py -q"

# faster, no extended thinking on workers
python gvs5h_strata.py --no-thinking --iters 4 "..."
```

Watch progress:

```powershell
Get-Content C:\Users\noahm\strata\gvs5h-strata\run.log -Wait -Tail 20
```

## Options

| flag | default | note |
| --- | --- | --- |
| `--iters` | 8 | max worker iterations; each is one full model call |
| `--verify` | none | command a worker may run to check its own work |
| `--no-thinking` | off | disables extended thinking on workers, faster but weaker |
| `--model` | `qwen3.8-flash-next-ud-iq4_xs` | must match what Strata has loaded |

Env: `STRATA_URL`, `STRATA_MODEL`, `STRATA_TOKEN`, `STRATA_MAX_TOKENS`,
`STRATA_TIMEOUT`, `GVS5H_WS`.

## Expected cost on this machine

Each iteration is a **sequential** Strata call, not parallel: plan and ideation
took a few minutes each on a 3070 Ti at 1M context. A 4-iteration run is roughly
5 calls. There is no 5x speedup here and no local "Fable 5 parity" claim — the
paper's 93% vs 90.4% was measured on pinned cloud backends, and it explicitly
reports gains are "not universal."

What you can expect is the mechanism: decomposition and a durable shared ledger,
at the cost of several sequential passes.

## Layout

- `gvs5h_strata.py` — the harness
- `run.ps1` / `run.log` — last test invocation and its output
- `ws/current/` — the ledger for the most recent run