# Cerebro Engine

Open-source, Windows-first, offline local LLM harness for small businesses and AI enthusiasts.
Cerebro measures the fastest safe configuration of llama.cpp for your exact machine instead of
guessing, then runs it contained and serves it on loopback.

Status: restarted 2026-09-22. See `docs/RESTART.md` for what was kept, what was dropped, and the
plan. Claims made anywhere about Cerebro must satisfy `docs/CLAIMS.md`.

Build (PowerShell; the MSVC linker must come first on PATH on this machine):

    $vc = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
    $sdk = "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.26100.0\x64"
    $env:PATH = "$vc;$sdk;$env:PATH"
    cargo test --workspace
