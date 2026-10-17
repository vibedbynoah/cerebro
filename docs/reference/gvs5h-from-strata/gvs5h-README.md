# GVS5H bench, ported to a local Strata server

Port of slee-persis/GVS5H (`ref/GVS5H`, cloned 2026-10-09) that measures whether their
ledger-based multi-agent orchestration gain reproduces on Strata-served models. The bench
itself is unmodified; everything Strata-specific lives in this file and
`run_gvs5h_strata.ps1`.

## How the adapter works

Their `orchestrator.chat()` routes calls by provider prefix. The `groq:` prefix posts to
`ESCALATION_OPENAI_BASE`, which is how their own LAN/litellm runs pointed it at a private
endpoint. Strata serves an OpenAI-compatible API on 127.0.0.1:8080 with no API key, so:

```
ESCALATION_OPENAI_BASE=http://127.0.0.1:8080/v1/chat/completions
GROQ_API_KEY=strata-local          # any non-empty string; the server needs no key
MULTIAGENT_MODEL=groq:strata
```

...is the whole adapter. No code changes to the bench.

## What has been verified so far

- One chat round-trip through `orchestrator.chat()` -> Strata: works (2026-10-09).
- `load_code_generation_dataset(release_v6)`: 1055 problems, loads from the HF cache.
- Pinned problem set `lcb100_hardest_v6.json`: 100 ids, same set the paper reports on.
- **A full engine run has NOT been measured yet.** The first smoke died when another
  process restarted :8080 with the iq3_xxs config mid-run (see "Hazards").

## Measured constraints (on this box)

- `--parallel` in the server: `strata serve` runs one request at a time unless the model
  config sets `"parallel": N`, so bench parallelism only queues. Keep `-Parallel 1`.
- coder-IQ1_M: `--max-context 32768`, ~7.4 tok/s (HumanEval run, `humaneval-coder.txt`).
- IQ3_XXS: 131072 context, ~2.2 tok/s (`humaneval-iq3_xxs.txt`) - multiagent needs
  ~12-45 calls per problem, which is days per problem at that speed.
- `CLOUD_TIMEOUT` is the per-call timeout and `CLOUD_MAX_TOKENS` the per-call output cap.
  A 4096-token cap at coder speed is ~9 minutes worst case; 1800s is the floor.

## Running it

The Strata server must already be loaded and healthy on :8080 (it is not started here):

```
powershell -File bench\run_gvs5h_strata.ps1 -Engine single     -N 2 -Tag name-single
powershell -File bench\run_gvs5h_strata.ps1 -Engine multiagent -N 2 -MaxIters 3 -Tag name-multi
```

Problem set is always the pinned ids file (`--ids-file`, the 100 hardest from release_v6),
results land in `bench/results/gvs5h-<tag>-<engine>.json`, workspaces and per-call
transcripts in `runs/ws/<slug>/transcript.jsonl` inside the ref clone.

## Hazards

- **RAM, measured 2026-10-09:** with IQ3_XXS resident on :8080 (it streams its 76 GB of
  experts off NVMe), `Win32_OperatingSystem.FreePhysicalMemory` read 199.5 MB, and a bench
  python building the LiveCodeBench cache was killed in that window -- Windows logged
  `RADAR_PRE_LEAK_64` for `python.exe` at 22:34. Both failed runs died before their first
  solve, one to a model swap and one to this. No full engine run has completed yet. Count
  free RAM before launching; coder-IQ1_M (resident ~17 GB) leaves room, IQ3_XXS does not.
- **Model swaps kill runs.** Anything that restarts :8080 with a different config (the
  user's run-*.bat habits, the scheduled DSH supervisor is on :8099 and harmless here)
  takes the port, and the bench process dies with the connection. Check
  `http://127.0.0.1:8080/health` reports the model you intend before launching.
- BOSGAME (192.168.1.34:8080, the second box that could offload the measurement) was
  unreachable at 23:51 -- the override only works when that machine is up.
- Cost for a local run is wall time, not dollars. Total it from the transcripts
  (`completion_tokens` per call), not from any price table.
- `ref/GVS5H/` and `bench/gvs5h/` are not in .gitignore yet - add them before any commit
  if you do not want the clone and its venv tracked.
