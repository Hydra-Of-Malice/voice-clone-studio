# One-shot environment setup for Windows (Python 3.11 + CUDA 12.6 wheels, Node 20 for the UI).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Needs: uv, Node.js, and MSVC Build Tools (pkuseg, a Chatterbox dependency, compiles from source).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Step($msg, $block) { Write-Host "==> $msg"; & $block; if ($LASTEXITCODE -ne 0) { throw "$msg failed" } }

if (-not (Test-Path .venv)) { Step "Create venv" { uv venv --python 3.11 .venv } }
Step "PyTorch (CUDA 12.6)" { uv pip install --python .venv torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 }
Step "Dependencies" { uv pip install --python .venv -r requirements.txt cython wheel }
# pkuseg does not declare numpy as a build dependency, so it must be built against the installed one
Step "pkuseg" { uv pip install --python .venv --no-build-isolation pkuseg==0.0.25 }
Step "Chatterbox" { uv pip install --python .venv chatterbox-tts }
Step "Pin setuptools" { uv pip install --python .venv "setuptools<81" }
Step "Web UI" { Push-Location web; npm install --no-audit --no-fund; npm run build; Pop-Location }
Write-Host "Done. Start the app with:  .venv\Scripts\python -m vc.main"
