#!/usr/bin/env bash
# Runs every Cerebro test suite, one after another, and keeps going when one fails. Writes one
# short report (REPORT.md: a line per suite, then the last lines of each failure) and the full
# output of every suite next to it, so the result reads in one place.
#
# Usage: bash tools/test-all.sh [OUT_DIR] [--quick]
#   --quick skips the slow suites (browser end-to-end, real exe, engine model checks).
# Exit code: the number of failed suites.
set -u
OUT=${1:-/c/Users/noahm/Projects/cerebro/test-runs/$(date +%Y%m%d-%H%M%S)}
QUICK=0; for a in "$@"; do [ "$a" = "--quick" ] && QUICK=1; done
mkdir -p "$OUT"
ENGINE=/c/Users/noahm/Projects/cerebro-engine
APP=/c/Users/noahm/Projects/cerebro-app
DESK=$APP/apps/desktop
PORT=${CEREBRO_E2E_PORT:-4392}
# Runs a command with the MSVC build tools on the path (Rust builds need them).
MSVC=/c/Users/noahm/Projects/cerebro/tools/msvc.cmd
REPORT="$OUT/REPORT.md"
FAILED=0
: > "$OUT/summary.tsv"

# run NAME TIMEOUT_S DIR COMMAND...  (COMMAND runs in bash; MSVC ones go through msvc)
run() {
  local name=$1 limit=$2 dir=$3; shift 3
  local log="$OUT/$(echo "$name" | tr ' /' '__').log"
  local t0=$(date +%s)
  (cd "$dir" && timeout "$limit" bash -c "$*") > "$log" 2>&1
  local code=$?
  local secs=$(( $(date +%s) - t0 ))
  local status=pass
  [ $code -ne 0 ] && status=FAIL && FAILED=$((FAILED + 1))
  [ $code -eq 124 ] && status="FAIL (timed out after ${limit}s)"
  printf "%s\t%s\t%ss\t%s\n" "$name" "$status" "$secs" "$(basename "$log")" >> "$OUT/summary.tsv"
  echo "$(date +%T) $status  $name (${secs}s)"
}
msvc() { echo "$MSVC $*"; }

# ---- Engine
run "engine: unit tests" 1800 "$ENGINE" "$(msvc cargo test --release -p cerebro-engine --lib 2>&1)"
run "engine: build tools" 1800 "$ENGINE" "$(msvc cargo build --release --features cuda,portable --example greedy --example bench --example ppl 2>&1)"
if [ $QUICK -eq 0 ]; then
  # Small models through every backend: the text must stay correct (each line checks one).
  for m in 'C:\models\dev\Qwen3-0.6B-Q8_0.gguf' 'C:\models\dev\qwen35\Qwen3.5-0.8B-Q8_0.gguf'; do
    [ -f "$m" ] || continue
    for be in cpu gpu; do
      run "engine: $(basename "$m") $be" 900 "$ENGINE" "target/release/examples/greedy.exe '$m' 'The capital of France is' 12 $be 2>&1 | tee /dev/stderr | grep -q 'Paris'"
    done
  done
fi

# ---- App: UI
run "ui: type check" 600 "$DESK" "npx tsc --noEmit -p tsconfig.json"
run "ui: unit tests" 900 "$DESK" "npx vitest run --config ui/vite.config.ts"
run "ui: build" 600 "$DESK" "npx vite build --config ui/vite.config.ts"
if [ $QUICK -eq 0 ]; then
  run "ui: browser end-to-end" 3600 "$DESK" "CEREBRO_E2E_PORT=$PORT npx playwright test --reporter=line"
  run "ui: glitch sweep" 1200 "$DESK" "node scripts/glitch-sweep.mjs"
fi

# ---- App: Rust
run "app: rust tests" 3600 "$APP" "$(msvc cargo test --release -p cerebro-desktop 2>&1)"
if [ $QUICK -eq 0 ]; then
  run "app: desktop build" 3600 "$DESK" "npm run build:desktop && $(msvc cargo build --release -p cerebro-desktop 2>&1)"
  # The real exe, driven like a user (opens, setup, guide, load, chat, close).
  run "app: real exe end-to-end" 1200 "$DESK" "node scripts/e2e-real.mjs 0.6B; grep -q '\"error\"' test-results/e2e-real/report.json && exit 1 || exit 0"
fi
# Line-ending-only changes a Rust build leaves in permission files.
(cd "$APP" && for f in $(git diff --name-only -- apps/desktop/src-tauri/permissions); do git diff --quiet --ignore-cr-at-eol -- "$f" && git checkout -q -- "$f"; done)

# ---- Report: a line per suite, then the end of each failure's output.
{
  echo "# Test run $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "$FAILED failed of $(wc -l < "$OUT/summary.tsv")."
  echo
  echo "| Suite | Result | Time | Log |"
  echo "|---|---|---|---|"
  while IFS=$'\t' read -r n s t l; do echo "| $n | $s | $t | $l |"; done < "$OUT/summary.tsv"
  while IFS=$'\t' read -r n s t l; do
    case "$s" in FAIL*)
      echo; echo "## $n"; echo; echo '```'
      grep -aE "error|Error|FAIL|failed|panicked|✘|×|  x " "$OUT/$l" | tail -25
      echo "--- last lines ---"; tail -15 "$OUT/$l"
      echo '```' ;;
    esac
  done < "$OUT/summary.tsv"
} > "$REPORT"
echo "report: $REPORT"
exit $FAILED
