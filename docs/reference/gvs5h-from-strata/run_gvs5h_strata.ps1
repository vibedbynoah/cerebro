# Ported GVS5H bench driver for a LOCAL Strata server.
#
# slee-persis/GVS5H (ref/GVS5H) routes every model call through provider prefixes
# ("groq:", "openai:", ...). The "groq:" prefix sends to ESCALATION_OPENAI_BASE,
# which is how their own LAN/litellm runs pointed it at a private endpoint. Strata
# serves an OpenAI-compatible API on 127.0.0.1:8080 with no API key, so "groq:strata"
# plus ESCALATION_OPENAI_BASE is the whole adapter -- no code changes to the bench.
#
# Usage (Strata server must already be loaded and healthy):
#   powershell -File bench\run_gvs5h_strata.ps1 -Engine single -N 2 -Tag smoke-single
#   powershell -File bench\run_gvs5h_strata.ps1 -Engine multiagent -N 2 -MaxIters 3 -Tag smoke-multi
#
# Notes that matter for a LOCAL run:
#  - The Strata engine decodes one request at a time unless "parallel": N is in the
#    model config, so keep -Parallel 1 (a queue, not a speedup).
#  - CLOUD_MAX_TOKENS caps output per call. Stratum runs at 32k context; multiagent
#    prompts grow with notes + current solution, so 4096 leaves room to breathe.
#  - CLOUD_TIMEOUT is the per-call timeout. Local decode at single-digit tok/s with a
#    4096-token cap can take ~10 minutes on a long solution; 1800s is the floor.
#  - Cost for a local run is wall time, not dollars. The transcript.jsonl per
#    workspace has completion_tokens/prompt_tokens/finish_reason per call to total up.

param(
    [ValidateSet('single', 'multiagent')] [string] $Engine = 'single',
    [int] $N = 2,
    [int] $MaxIters = 3,
    [int] $Cap = 4096,
    [int] $Parallel = 1,
    [int] $Timeout = 1800,
    [string] $Tag = 'run',
    [string] $Model = 'strata'
)

$ErrorActionPreference = 'Stop'
$Root    = 'C:\Users\noahm\strata'
$Ref     = "$Root\ref\GVS5H\codebase\v2-current\escalation"
$Py      = "$Root\bench\gvs5h\.venv\Scripts\python.exe"
$Results = "$Root\bench\results"
$Log     = "$Results\gvs5h-$Tag-$Engine.log"

if (-not (Test-Path $Results)) { New-Item -ItemType Directory -Path $Results | Out-Null }

# --- health check: a hung engine wastes an hour of bench time ----------------------
try {
    $health = (Invoke-WebRequest http://127.0.0.1:8080/health -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    if (-not $health.loaded) { throw "server up but not loaded" }
} catch {
    Write-Output "Strata server not ready: $($_.Exception.Message)"
    Write-Output "Start it first: run-coder-iq1_m.bat (or run-iq3_xxs.bat), then retry."
    exit 1
}
Write-Output "Strata healthy: $($health.model) ctx=$($health.max_context)"

# --- the adapter: point their groq: provider at Strata ----------------------------
$env:ESCALATION_OPENAI_BASE   = 'http://127.0.0.1:8080/v1/chat/completions'
$env:GROQ_API_KEY             = 'strata-local'      # any non-empty string; server needs no key
$env:ESCALATION_GROQ_REASONING = 'none'             # don't pay for hidden reasoning tokens
$env:MULTIAGENT_MODEL         = "groq:$Model"
$env:LCB_RELEASE              = 'release_v6'        # the pinned hard set from the paper
$env:ESCALATION_CLOUD_MAX_TOKENS = "$Cap"
$env:ESCALATION_CLOUD_TIMEOUT    = "$Timeout"
$env:MULTIAGENT_MAX_ITERS     = "$MaxIters"

$out = "$Results\gvs5h-$Tag-$Engine.json"
Write-Output "engine=$Engine model=groq:$Model n=$N cap=$Cap timeout=$Timeout max_iters=$MaxIters log=$Log"

& $Py "$Ref\run_bench.py" --engine $Engine --only lcb --lcb $N --parallel $Parallel --out $out *> $Log
$rc = $LASTEXITCODE
Get-Content $Log -Tail 15
Write-Output "exit=$rc results=$out"
