@echo off
rem Builds the desktop shell with the MSVC environment (the engine's CUDA kernels need nvcc and cl.exe).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cargo build -p cerebro-desktop %*
