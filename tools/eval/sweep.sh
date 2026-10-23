#!/usr/bin/env bash
# Full placement sweep: every text model in the app's model folders (smallest first), every placement
# this machine offers, on Cerebro's engine and on the official llama.cpp release as the control.
# Nothing here names a model or a GPU: models come from the folders the app scans, GPUs from the
# machine, layer counts from fractions of each model's own depth.
#
# Usage: bash tools/eval/sweep.sh OUT_DIR [MAX_GB] [MIN_GB]
#   Only models between MIN_GB (default 0) and MAX_GB (default 999) run.
# Results: OUT_DIR/cerebro.jsonl and OUT_DIR/llamacpp.jsonl (one line per run), progress in OUT_DIR/progress.log.
set -u
OUT=${1:?out dir}
MAX_GB=${2:-999}
MIN_GB=${3:-0}
mkdir -p "$OUT"
CB=/c/Users/noahm/Projects/cerebro-engine/target/release/examples/bench.exe
LB=/c/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751/llama-bench.exe            # CUDA release
LV=/c/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751-vulkan/llama-bench.exe     # Vulkan release
LF=/c/Users/noahm/Qwen38/runtime/llama.cpp/build-cuda-ninja/bin/llama-bench.exe       # fork, for architectures the release lacks
CJ="$OUT/cerebro.jsonl"; LJ="$OUT/llamacpp.jsonl"; LOG="$OUT/progress.log"
T=${SWEEP_TIMEOUT:-2400}
say() { echo "$(date +%T) $*" | tee -a "$LOG" >&2; }

# The machine, as the app sees it.
GPUS=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name -join ';'" | tr -d '\r')
IGPU=$(echo "$GPUS" | tr ';' '\n' | grep -iE "Intel.*(Iris|UHD|Arc.*Graphics$|Graphics)|AMD Radeon\(TM\)|Radeon Graphics" | head -1)
DGPU=$(echo "$GPUS" | tr ';' '\n' | grep -iE "NVIDIA|Radeon RX|Arc A" | head -1)
MACHINE="$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" | tr -d '\r'), $(echo "$GPUS" | tr ';' ','), $(powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)" | tr -d '\r') GB RAM"
say "machine: $MACHINE | dGPU: ${DGPU:-none} | iGPU: ${IGPU:-none}"
VK_I=$("$LV" --list-devices 2>&1 | grep -iF "${IGPU:-NONE}" | grep -oE ": [0-9]+ = " | head -1 | tr -dc '0-9')

# The models: SWEEP_MODELS (paths separated by ';', in that order) or every text model in the
# app's folders, smallest first (tools/eval/list_models.py).
HERE=$(dirname "$0")
if [ -n "${SWEEP_MODELS:-}" ]; then
  mapfile -t MODELS < <(python -I "$HERE/list_models.py" "$MAX_GB" "$MIN_GB" --paths "$SWEEP_MODELS" | tr -d '\r')
else
  mapfile -t MODELS < <(python -I "$HERE/list_models.py" "$MAX_GB" "$MIN_GB" | tr -d '\r')
fi
say "${#MODELS[@]} models"

cerebro() { # LABEL BACKEND DECODE [ENV=..] [-- extra bench args]
  local label=$1 be=$2 dec=$3; shift 3
  local envs=() extra=()
  while [ $# -gt 0 ]; do case "$1" in --*) extra+=("$1" "$2"); shift 2 ;; *) envs+=("$1"); shift ;; esac; done
  say "  cerebro $label"
  if ! timeout "$T" env "${envs[@]}" "$CB" --model "$MODEL" --out "$CJ" --machine "$MACHINE" --label "$label" --backends "$be" --lengths 512 --decode "$dec" --reps 3 "${extra[@]}" > /dev/null 2>> "$OUT/cerebro.err"; then
    echo "{\"label\":\"$label\",\"model\":\"$(basename "$MODEL")\",\"failed\":true,\"why\":\"$(tail -c 300 "$OUT/cerebro.err" | tr -d '\"\n\r' | tail -c 200)\"}" >> "$CJ"
  fi
}
llama() { # BIN LABEL [args]
  local bin=$1 label=$2; shift 2
  say "  llama.cpp $label"
  local o
  # -d 0,512: reading from an empty context and writing after a 512-token prompt, the work
  # Cerebro's bench does (verdict.py and report.py pair them that way).
  o=$(timeout "$T" "$bin" -m "$MODEL" -p 512 -n 128 -r 3 -fa 1 -d 0,512 -o jsonl "$@" 2>> "$OUT/llamacpp.err")
  if [ -n "$o" ]; then echo "$o" | sed "s|^{|{\"label\":\"$label\",|" >> "$LJ"; return 0; fi
  echo "{\"label\":\"$label\",\"model_filename\":\"$(basename "$MODEL")\",\"failed\":true}" >> "$LJ"; return 1
}

for entry in "${MODELS[@]}"; do
  SIZE=${entry%%$'\t'*}; MODEL=${entry#*$'\t'}
  GB=$(( SIZE / 1000000000 ))
  say "== $(basename "$MODEL") (${GB} GB)"
  # Fewer written tokens where each one is slow: big models and anything reading from the drive.
  D=128; [ "$GB" -ge 10 ] && D=32; DS=16
  MOE=$("$CB" --model "$MODEL" --print-experts 2>/dev/null | tr -dc '0-9'); [ "${MOE:-0}" -gt 0 ] && MOE=1 || MOE=0
  # The seven combinations of processor (RAM), graphics card and drive, then the integrated GPU.
  # "gpu" asks for every layer on the card (a model too big for it is split: gpu_layers says so).
  # "ssd" = layers read from the drive on every pass, computed by the processor; for a mixture of
  # experts, the experts come from the drive with the RAM expert cache kept to 1 GB.
  SSD_ALL=(X=1 --drive-frac 1.0); SSD_HALF=(X=1 --drive-frac 0.5); SSD_THIRD=(X=1 --drive-frac 0.33)
  if [ "${MOE:-0}" = 1 ]; then SSD_ALL=(CEREBRO_EXPERT_CACHE_MB=1024); SSD_HALF=(CEREBRO_EXPERT_CACHE_MB=1024); SSD_THIRD=(CEREBRO_EXPERT_CACHE_MB=1024); fi
  cerebro "cpu" cpu "$D" X=1
  [ -n "$DGPU" ] && cerebro "gpu" gpu "$D" X=1 --gpu-frac 1.0
  cerebro "ssd" cpu "$DS" "${SSD_ALL[@]}"
  [ -n "$DGPU" ] && cerebro "cpu+gpu" gpu "$D" X=1 --gpu-frac 0.5
  cerebro "cpu+ssd" cpu "$DS" "${SSD_HALF[@]}"
  [ -n "$DGPU" ] && cerebro "gpu+ssd" gpu "$DS" "${SSD_HALF[@]}" --gpu-frac 0.5
  [ -n "$DGPU" ] && cerebro "cpu+gpu+ssd" gpu "$DS" "${SSD_THIRD[@]}" --gpu-frac 0.33
  # The integrated GPU shares system memory and holds an int8 copy of the weights (about 1.9x a
  # Q4_K file): skip it, with a row saying why, when that copy would not fit in free RAM with 4 GB
  # to spare (it got the whole sweep stopped for low memory on a 32 GB laptop).
  FREE_GB=$(powershell.exe -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '')
  IGPU_OK=$(python -I -c "print(1 if $SIZE * 1.9 / 1e9 + 4 < ${FREE_GB:-0} else 0)")
  if [ -n "$IGPU" ] && [ "$IGPU_OK" = 1 ]; then
    cerebro "igpu" portable "$D" "CEREBRO_PORTABLE_GPU=$IGPU" CEREBRO_PORTABLE_INTEGRATED=1
    cerebro "igpu+cpu" portable "$D" "CEREBRO_PORTABLE_GPU=$IGPU" CEREBRO_PORTABLE_INTEGRATED=1 --gpu-frac 0.5
  elif [ -n "$IGPU" ]; then
    say "  cerebro igpu skipped: the int8 copy needs about $(( SIZE * 19 / 10 / 1000000000 )) GB, ${FREE_GB} GB free"
    for lab in igpu igpu+cpu; do echo "{\"label\":\"$lab\",\"model\":\"$(basename "$MODEL")\",\"failed\":true,\"why\":\"skipped: not enough free memory for the integrated GPU's copy\"}" >> "$CJ"; done
  fi
  # Control: the official release on the same model (the fork where the release lacks the architecture).
  BIN=$LB
  if ! CUDA_VISIBLE_DEVICES=-1 timeout "$T" "$LB" -m "$MODEL" -p 8 -n 1 -r 1 -ngl 0 > /dev/null 2>&1; then BIN=$LF; say "  (release cannot load it: using the fork)"; fi
  NL=$("$CB" --model "$MODEL" --print-layers 2>/dev/null | tr -dc '0-9'); NL=${NL:-32}
  CUDA_VISIBLE_DEVICES=-1 llama "$BIN" "cpu" -ngl 0 -t 6,12,14,20
  if [ -n "$DGPU" ]; then
    # All layers when they fit; otherwise the most that load (the release has no automatic fit here).
    llama "$BIN" "gpu" -ngl 999 || for f in 85 75 60 50; do llama "$BIN" "gpu (${f}% of layers fit)" -ngl "$(( NL * f / 100 ))" && break; done
    llama "$BIN" "cpu+gpu" -ngl "$(( NL / 2 ))"
  fi
  if [ -n "$IGPU" ] && [ -n "$VK_I" ]; then
    llama "$LV" "igpu" -dev "Vulkan$VK_I" -ngl 999
    llama "$LV" "igpu+cpu" -dev "Vulkan$VK_I" -ngl "$(( NL / 2 ))"
  fi
done
say "done"
