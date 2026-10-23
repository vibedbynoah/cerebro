# Test run 2026-10-21 22:32

3 failed of 6.

| Suite | Result | Time | Log |
|---|---|---|---|
| engine: unit tests | FAIL | 0s | engine:_unit_tests.log |
| engine: build tools | FAIL | 1s | engine:_build_tools.log |
| ui: type check | pass | 19s | ui:_type_check.log |
| ui: unit tests | pass | 114s | ui:_unit_tests.log |
| ui: build | pass | 9s | ui:_build.log |
| app: rust tests | FAIL | 1s | app:_rust_tests.log |

## engine: unit tests

```
bash: -c: line 1: syntax error near unexpected token `('
--- last lines ---
bash: -c: line 1: syntax error near unexpected token `('
bash: -c: line 1: `powershell.exe -NoProfile -Command "& \"\$env:SystemRoot\System32\cmd.exe\" /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cargo test --release -p cerebro-engine --lib'"'
```

## engine: build tools

```
bash: -c: line 1: syntax error near unexpected token `('
--- last lines ---
bash: -c: line 1: syntax error near unexpected token `('
bash: -c: line 1: `powershell.exe -NoProfile -Command "& \"\$env:SystemRoot\System32\cmd.exe\" /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cargo build --release --features cuda,portable --example greedy --example bench --example ppl'"'
```

## app: rust tests

```
bash: -c: line 1: syntax error near unexpected token `('
--- last lines ---
bash: -c: line 1: syntax error near unexpected token `('
bash: -c: line 1: `powershell.exe -NoProfile -Command "& \"\$env:SystemRoot\System32\cmd.exe\" /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && cargo test --release -p cerebro-desktop'"'
```
