#!/usr/bin/env bash
# Head-to-head placement matrix: Cerebro engine vs the official llama.cpp release on the same
# machine, models and prompt. Every raw result line is kept (jsonl); the report is built from them.
# Usage: bash tools/eval/placement-matrix.sh OUT_DIR
set -u
OUT=${1:?out dir}
mkdir -p "$OUT"
CB=/c/Users/noahm/Projects/cerebro-engine/target/release/examples/bench.exe
LB=/c/Users/noahm/Projects/qwen-flash-next/llama.cpp-b10751/llama-bench.exe
M06='C:\models\dev\Qwen3-0.6B-Q8_0.gguf'
M8='C:\Users\noahm\AppData\Local\Cerebro\models\Qwen__Qwen3-8B-GGUF\Qwen3-8B-Q4_K_M.gguf'
M14='C:\models\dev\dense\Qwen3-14B-Q4_K_M.gguf'
M27='C:\Users\noahm\.lmstudio\models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf'
MACHINE="i9-12900H + RTX 3070 Ti Laptop 8 GB, 32 GB RAM, Micron 3400 NVMe (2.10 GB/s sequential)"
CJ="$OUT/cerebro.jsonl"
LJ="$OUT/llamacpp.jsonl"

# cerebro LABEL BACKEND DECODE [ENV...]
cerebro() {
  local label=$1 be=$2 dec=$3; shift 3
  echo "cerebro: $label" >&2
  env "$@" "$CB" --model "$MODEL" --out "$CJ" --machine "$MACHINE" --label "$label" --backends "$be" --lengths 512 --decode "$dec" --reps 3 > /dev/null 2>> "$OUT/cerebro.err" || echo "{\"label\":\"$label\",\"model\":\"$MODEL\",\"failed\":true}" >> "$CJ"
}
# llama LABEL [llama-bench args...]  (CPU-only runs hide the GPU)
llama() {
  local label=$1; shift
  echo "llama.cpp: $label" >&2
  "$LB" -m "$MODEL" -p 512 -n 128 -r 3 -fa 1 -o jsonl "$@" 2>> "$OUT/llamacpp.err" | sed "s|^{|{\"label\":\"$label\",|" >> "$LJ" || echo "{\"label\":\"$label\",\"model\":\"$MODEL\",\"failed\":true}" >> "$LJ"
}
llama_cpu() {
  local label=$1; shift
  echo "llama.cpp: $label" >&2
  CUDA_VISIBLE_DEVICES=-1 "$LB" -m "$MODEL" -p 512 -n 128 -r 3 -fa 1 -ngl 0 -o jsonl "$@" 2>> "$OUT/llamacpp.err" | sed "s|^{|{\"label\":\"$label\",|" >> "$LJ"
}

# --- Qwen3-0.6B Q8_0: fits everywhere.
MODEL=$M06
for t in 6 12 14; do cerebro "cpu t$t" cpu 128 CEREBRO_THREADS=$t; done
cerebro "gpu" gpu 128 X=1
cerebro "gpu14+cpu" gpu 128 CEREBRO_GPU_LAYERS=14
llama_cpu "cpu" -t 6,12,14
llama "gpu" -ngl 99
llama "gpu14+cpu" -ngl 14

# --- Qwen3-8B Q4_K_M: fits the GPU, so every tier alone and every combination.
MODEL=$M8
for t in 6 12 14; do cerebro "cpu t$t" cpu 64 CEREBRO_THREADS=$t; done
cerebro "gpu" gpu 64 X=1
cerebro "gpu18+cpu18" gpu 64 CEREBRO_GPU_LAYERS=18
cerebro "ssd36+head" cpu 16 CEREBRO_STREAM_LAYERS=36
cerebro "gpu18+ssd18" gpu 16 CEREBRO_GPU_LAYERS=18 CEREBRO_STREAM_LAYERS=18
cerebro "cpu18+ssd18" cpu 16 CEREBRO_STREAM_LAYERS=18
cerebro "gpu12+cpu12+ssd12" gpu 16 CEREBRO_GPU_LAYERS=12 CEREBRO_STREAM_LAYERS=12
llama_cpu "cpu" -t 6,12,14
llama "gpu" -ngl 99
llama "gpu18+cpu18" -ngl 18

# --- Qwen3-14B Q4_K_M (8.4 GiB): does not fit 8 GB, so split.
MODEL=$M14
cerebro "cpu t12" cpu 32 CEREBRO_THREADS=12
cerebro "gpu auto+cpu" gpu 64 X=1
cerebro "gpu auto+ssd rest" gpu 16 CEREBRO_STREAM_LAYERS=9
llama_cpu "cpu" -t 12
llama "gpu+cpu sweep" -ngl 28,31,34

# --- Qwen3.8-27B Q4_K_M (15.4 GiB): split or processor.
MODEL=$M27
cerebro "cpu t12" cpu 16 CEREBRO_THREADS=12
cerebro "gpu auto+cpu" gpu 32 X=1
llama_cpu "cpu" -t 12
llama "gpu+cpu sweep" -ngl 16,20,24

echo done >&2
