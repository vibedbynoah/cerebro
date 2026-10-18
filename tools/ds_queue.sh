#!/usr/bin/env bash
# DeepSeek job queue for opencode. Each job is a worktree ../cerebro-app-ds-<job> (or an absolute
# path) holding a .ds-task.md spec. Each attempt is a FRESH opencode session (resuming with -c made
# the API fail); the prompt points at the partial work already in the worktree. At most MAX opencode
# processes run at once (more corrupt opencode's shared session database), with a backoff after an
# attempt that failed fast (rate limit / API error).
# Usage: tools/ds_queue.sh BASE_COMMIT job1 job2 ...
LOG=${LOG:-$HOME/ds-logs}
mkdir -p "$LOG"
BASE=$1
shift
MAX=${MAX:-3}
PROMPT='Read the full task in .ds-task.md in this folder and do exactly what it says. Earlier attempts may have left work in this worktree: first run `git status` and `git diff`, keep what is correct, fix what is not, and finish the task. Engine source is readable at .engine-src\ (read only). Never use paths outside this folder. Build, run the tests, fix failures, and commit on your branch. Do not stop to ask.'

wt() { case "$1" in /*) echo "$1" ;; *) echo "/c/Users/noahm/Projects/cerebro-app-ds-$1" ;; esac; }
committed() { [ "$(git -C "$(wt "$1")" log --oneline "$BASE"..HEAD 2>/dev/null | wc -l)" -gt 0 ]; }
# Only this queue's `opencode run` processes count: the owner's own opencode windows and service do not.
running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='opencode.exe'\" | Where-Object { \$_.CommandLine -match ' run ' }).Count" 2>/dev/null | tr -d ''; }

run_job() {
  local t=$1 dir
  dir=$(wt "$t")
  local name=$(basename "$dir")
  for attempt in 1 2 3 4 5; do
    committed "$t" && { echo "$name COMMITTED: $(git -C "$dir" log --oneline -1)"; return; }
    while [ "$(running)" -ge "$MAX" ]; do sleep 20; done
    echo "$name attempt $attempt $(date +%T)"
    local t0=$(date +%s)
    (cd "$dir" && timeout 5400 opencode run -m "opencode/deepseek-v4.1-flash-free#max" --auto "$PROMPT" >> "$LOG/$name.out" 2>> "$LOG/$name.err")
    [ $(( $(date +%s) - t0 )) -lt 120 ] && sleep $(( attempt * 120 ))
  done
  committed "$t" && echo "$name COMMITTED: $(git -C "$dir" log --oneline -1)" || echo "$name GAVE UP after 5 attempts"
}

for t in "$@"; do
  run_job "$t" &
  sleep 60
done
wait
echo "QUEUE DONE"
