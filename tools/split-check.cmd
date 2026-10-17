@echo off
rem Greedy output must match across CPU only, whole GPU and a forced layer split.
rem Usage: tools\split-check.cmd MODEL.gguf SPLIT_LAYERS
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cargo build --release -q -p cerebro-engine --example greedy || exit /b 1
set P=Explain why the sky is blue in two sentences.
echo == cpu
set CEREBRO_GPU_LAYERS=
target\release\examples\greedy.exe %1 "%P%" 48
echo == gpu
target\release\examples\greedy.exe %1 "%P%" 48 gpu
echo == split %2
set CEREBRO_GPU_LAYERS=%2
target\release\examples\greedy.exe %1 "%P%" 48 gpu
set CEREBRO_GPU_LAYERS=
