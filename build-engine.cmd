@echo off
rem Builds Cerebro Engine with the MSVC host compiler on PATH (nvcc needs cl.exe for the CUDA kernels).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cargo build --release -p cerebro-engine %*
