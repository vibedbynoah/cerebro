@echo off
rem Runs the engine benchmark with the MSVC environment and logs to docs/measurements/engine.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
for /f %%i in ('git rev-parse --short HEAD') do set COMMIT=%%i
cargo run --release -q -p cerebro-engine --example bench -- --commit %COMMIT% --out C:\Users\noahm\Projects\cerebro\docs\measurements\engine\runs.jsonl %*
