# One-shot development setup for Windows (Python 3.11 + CUDA 12.8 wheels, Node 20 for the UI).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Needs: uv, Node.js, and MSVC Build Tools (pkuseg, a Chatterbox dependency, compiles from source).
param([string]$Cuda = "cu128", [string]$Torch = "2.7.1")
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Step($msg, $block) { Write-Host "==> $msg"; & $block; if ($LASTEXITCODE -ne 0) { throw "$msg failed" } }
$index = "https://download.pytorch.org/whl/$Cuda"
New-Item -ItemType Directory -Force build | Out-Null
# Chatterbox pins torch 2.6.0 (no RTX 50 support); the override keeps the CUDA 12.8 build in place
"torch==$Torch+$Cuda`ntorchaudio==$Torch+$Cuda" | Set-Content -Encoding ascii build\torch-overrides.txt

if (-not (Test-Path .venv)) { Step "Create venv" { uv venv --python 3.11 .venv } }
Step "PyTorch $Torch ($Cuda)" { uv pip install --python .venv "torch==$Torch" "torchaudio==$Torch" --index-url $index }
Step "Dependencies" { uv pip install --python .venv --override build\torch-overrides.txt --extra-index-url $index --index-strategy unsafe-best-match -r requirements.txt cython wheel }
# pkuseg does not declare numpy as a build dependency, so it must be built against the installed one
Step "pkuseg" { uv pip install --python .venv --no-build-isolation pkuseg==0.0.25 }
Step "Chatterbox" { uv pip install --python .venv --override build\torch-overrides.txt --extra-index-url $index --index-strategy unsafe-best-match chatterbox-tts }
Step "Pin setuptools" { uv pip install --python .venv "setuptools<81" }
Step "Web UI" { Push-Location web; npm install --no-audit --no-fund; npm run build; Pop-Location }
Write-Host "Done. Start the app with:  .venv\Scripts\python -m vc.main"
