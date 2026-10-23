#!/usr/bin/env bash
# DeepSeek job queue for opencode. Each job is a worktree ../cerebro-app-ds-<job> (or an absolute
# path) holding a .ds-task.md spec. Each attempt is a FRESH opencode session (resuming with -c made
# the API fail); the prompt points at the partial work already in the worktree. At most MAX opencode
# processes run at once (more corrupt opencode's shared session database), with a backoff after an
# attempt that failed fast (rate limit / API error).
# Usage: tools/ds_queue.sh BASE_COMMIT job1 job2[=OTHER_BASE] ...
# Jobs run in the order given; a rate-limited attempt waits 10 minutes before the next one.
LOG=${LOG:-$HOME/ds-logs}
mkdir -p "$LOG"
BASE=$1
shift
MAX=${MAX:-3}
# The models to use, in order: after an attempt that hit a rate limit the next attempt moves on
# to the next model instead of waiting (the free tiers limit each model separately).
MODELS=(${DS_MODELS:-opencode/deepseek-v4.1-flash-free#max opencode/glm-5.3-flash-free opencode/minimax-m3.1-flash-free opencode/nemotron-3-ultra-free})
PROMPT='Read the full task in .ds-task.md in this folder and do exactly what it says. Earlier attempts may have left work in this worktree: first run `git status` and `git diff`, keep what is correct, fix what is not, and finish the task. Engine source is readable at .engine-src\ (read only). Never use paths outside this folder. Build, run the tests, fix failures, and commit on your branch. Do not stop to ask.'

wt() { local j=${1%%=*}; case "$j" in /*) echo "$j" ;; *) echo "/c/Users/noahm/Projects/cerebro-app-ds-$j" ;; esac; }
base() { case "$1" in *=*) echo "${1#*=}" ;; *) echo "$BASE" ;; esac; }
committed() { [ "$(git -C "$(wt "$1")" log --oneline "$(base "$1")"..HEAD 2>/dev/null | wc -l)" -gt 0 ]; }
# Only this queue's `opencode run` processes count: the owner's own opencode windows and service do not.
running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='opencode.exe'\" | Where-Object { \$_.CommandLine -match ' run ' }).Count" 2>/dev/null | tr -d '
'; }

run_job() {
  local t=$1 dir
  dir=$(wt "$t")
  local name=$(basename "$dir")
  local mi=0
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    committed "$t" && { echo "$name COMMITTED: $(git -C "$dir" log --oneline -1)"; return; }
    while [ "$(running)" -ge "$MAX" ]; do sleep 20; done
    local model=${MODELS[$((mi % ${#MODELS[@]}))]}
    echo "$name attempt $attempt with $model $(date +%T)"
    local t0=$(date +%s)
    (cd "$dir" && timeout 5400 opencode run -m "$model" --auto "$PROMPT" >> "$LOG/$name.out" 2>> "$LOG/$name.err")
    [ $(( $(date +%s) - t0 )) -lt 120 ] && sleep $(( attempt * 120 ))
    if tail -c 2000 "$LOG/$name.err" 2>/dev/null | grep -q "Rate limit exceeded"; then
      echo "$name rate-limited on $model $(date +%T)"
      mi=$((mi + 1))
      # Every model tried once since the last wait: wait before going round again.
      [ $((mi % ${#MODELS[@]})) -eq 0 ] && sleep 600
      : > "$LOG/$name.err"
    fi
  done
  committed "$t" && echo "$name COMMITTED: $(git -C "$dir" log --oneline -1)" || echo "$name GAVE UP after 12 attempts"
}

if [ "$MAX" -le 1 ]; then
  for t in "$@"; do run_job "$t"; done
else
  for t in "$@"; do
    run_job "$t" &
    sleep 60
  done
  wait
fi
echo "QUEUE DONE"
