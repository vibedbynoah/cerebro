# Install the competitor harnesses that need an installer, pip or npm.
# Owner approved installs on 2026-09-17. GUI installers run silently so nothing sits on screen
# (the laptop's OLED panel must not hold a static window). Everything is logged, and the log
# records how to remove each one.

$ErrorActionPreference = 'Continue'
$log = 'C:\Users\noahm\Projects\marrow\docs\research\2026-09\rivals-install.log'
function Note($msg) {
  $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Write-Output $line
  Add-Content -Path $log -Value $line -Encoding utf8
}

Note "start; C: free $([math]::Round((Get-PSDrive C).Free/1GB)) GB"

# 1. Jan (NSIS, silent). Uninstall: Apps & Features, then remove %APPDATA%\Jan.
$jan = 'C:\bench\downloads\Jan_0.8.4_x64-setup.exe'
if (Test-Path 'C:\Users\noahm\AppData\Local\Programs\Jan\Jan.exe') {
  Note 'Jan already installed'
} elseif (Test-Path $jan) {
  Note 'installing Jan (silent)'
  $p = Start-Process -FilePath $jan -ArgumentList '/S' -Wait -PassThru
  Note "Jan installer exit $($p.ExitCode)"
}

# 2. AnythingLLM Desktop (NSIS, silent). Uninstall: Apps & Features + %APPDATA%\anythingllm-desktop.
$allm = 'C:\bench\downloads\AnythingLLMDesktop.exe'
if (Test-Path "$env:LOCALAPPDATA\Programs\AnythingLLM\AnythingLLM.exe") {
  Note 'AnythingLLM already installed'
} elseif (Test-Path $allm) {
  Note 'installing AnythingLLM (silent)'
  $p = Start-Process -FilePath $allm -ArgumentList '/S' -Wait -PassThru
  Note "AnythingLLM installer exit $($p.ExitCode)"
}

# 3. Open WebUI in its own venv, with its data directory pinned so uninstall is one folder delete.
$venv = 'C:\bench\openwebui-venv'
if (Test-Path "$venv\Scripts\open-webui.exe") {
  Note 'Open WebUI already installed'
} else {
  Note 'creating Open WebUI venv (python 3.12)'
  & 'C:\Users\noahm\AppData\Local\Programs\Python\Python312\python.exe' -m venv $venv
  & "$venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
  Note 'pip install open-webui (several minutes, ~2-3 GB)'
  & "$venv\Scripts\pip.exe" install open-webui --quiet
  Note "open-webui installed: $(Test-Path "$venv\Scripts\open-webui.exe")"
}

# 4. Cline CLI. Uninstall: npm uninstall -g cline, then remove ~/.cline.
if (Get-Command cline -ErrorAction SilentlyContinue) {
  Note 'cline already installed'
} else {
  Note 'npm install -g cline'
  npm install -g cline 2>&1 | Select-Object -Last 3 | ForEach-Object { Note "  npm: $_" }
}

Note 'versions:'
foreach ($probe in @(
  @('ollama', 'C:\bench\ollama\ollama.exe', '--version'),
  @('koboldcpp', 'C:\bench\koboldcpp\koboldcpp.exe', '--version'),
  @('llama-server (official b11026)', 'C:\bench\llamacpp-official\llama-server.exe', '--version'),
  @('goose cli', 'C:\bench\goose\goose.exe', '--version')
)) {
  if (Test-Path $probe[1]) {
    $out = & $probe[1] $probe[2] 2>&1 | Select-Object -First 2
    Note "  $($probe[0]): $($out -join ' | ')"
  } else {
    Note "  $($probe[0]): MISSING at $($probe[1])"
  }
}
Note "finished; C: free $([math]::Round((Get-PSDrive C).Free/1GB)) GB"
