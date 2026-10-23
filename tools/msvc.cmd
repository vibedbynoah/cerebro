@echo off
rem Runs a command with the MSVC build tools on the path: tools\msvc.cmd cargo build ...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
%*
