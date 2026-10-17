@echo off
rem A/B benchmark of the four-row Q8_0 decode kernel (switch cpu_rows4) on the CPU, alternating
rem off/on so thermal drift hits both equally, with the official llama.cpp release as the control.
rem Usage: tools\bench-rows4.cmd MODEL.gguf LLAMA_BENCH.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
for /f %%i in ('git rev-parse --short HEAD') do set COMMIT=%%i
set OUT=C:\Users\noahm\Projects\cerebro\docs\measurements\engine\rows4.jsonl
set CEREBRO_THREADS=16
cargo build --release -q -p cerebro-engine --example bench || exit /b 1
for %%r in (1 2 3) do (
  set CEREBRO_ROWS4=0
  target\release\examples\bench.exe --model %1 --backends cpu --lengths 32 --decode 128 --reps 3 --commit %COMMIT%-rows4off --out %OUT% --machine "i9-12900H+RTX3070Ti-8GB"
  set CEREBRO_ROWS4=1
  target\release\examples\bench.exe --model %1 --backends cpu --lengths 32 --decode 128 --reps 3 --commit %COMMIT%-rows4on --out %OUT% --machine "i9-12900H+RTX3070Ti-8GB"
  %2 -m %1 -t 16 -p 0 -n 128 -r 3 -ngl 0 -o jsonl >> C:\Users\noahm\Projects\cerebro\docs\measurements\engine\rows4-llamacpp.jsonl
)
