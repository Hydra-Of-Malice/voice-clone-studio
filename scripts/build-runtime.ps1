# Builds the self-contained Python runtime that the installer ships (nothing is downloaded on the user's PC):
#   build\runtime\python.exe + Lib\site-packages with PyTorch CUDA 12.6, Chatterbox and the app dependencies.
# python-build-standalone is relocatable, so the installer can copy the folder anywhere.
# Needs MSVC Build Tools on the BUILD machine only (pkuseg compiles from source); users do not need them.
#   powershell -ExecutionPolicy Bypass -File scripts\build-runtime.ps1 [-Clean]
param([switch]$Clean, [string]$PythonTag = "3.11.16+20260924", [string]$Cuda = "cu126")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build"
$runtime = Join-Path $build "runtime"
$dl = Join-Path $build "downloads"
New-Item -ItemType Directory -Force $build, $dl | Out-Null
if ($Clean -and (Test-Path $runtime)) { Remove-Item -Recurse -Force $runtime }
$uv = (Get-Command uv -ErrorAction Stop).Source
function Uv { & $uv @args; if ($LASTEXITCODE -ne 0) { throw "uv $($args -join ' ') failed" } }

if (-not (Test-Path (Join-Path $runtime "python.exe"))) {
    $ver, $rel = $PythonTag -split "\+"
    $name = "cpython-$PythonTag-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
    $url = "https://github.com/astral-sh/python-build-standalone/releases/download/$rel/" + $name.Replace("+", "%2B")
    $tgz = Join-Path $dl $name
    if (-not (Test-Path $tgz)) { Write-Host "== downloading $name"; Invoke-WebRequest -Uri $url -OutFile $tgz }
    $tmp = Join-Path $build "pbs-extract"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Force $tmp | Out-Null
    & "$env:SystemRoot\System32\tar.exe" -xzf $tgz -C $tmp
    Move-Item (Join-Path $tmp "python") $runtime
    Remove-Item -Recurse -Force $tmp
}
$py = Join-Path $runtime "python.exe"
Write-Host "== runtime python: $(& $py -c 'import sys; print(sys.version.split()[0])')"

Write-Host "== PyTorch ($Cuda)"
Uv pip install --python $py --index-url "https://download.pytorch.org/whl/$Cuda" torch==2.6.0 torchaudio==2.6.0
Write-Host "== dependencies"
Uv pip install --python $py -r (Join-Path $root "requirements.txt") cython wheel
Write-Host "== pkuseg (built against the installed numpy)"
Uv pip install --python $py --no-build-isolation pkuseg==0.0.25
Write-Host "== Chatterbox"
Uv pip install --python $py chatterbox-tts
Uv pip install --python $py "setuptools<81"
Uv pip uninstall --python $py pytest cython wheel

Write-Host "== pruning caches"
Get-ChildItem -Path $runtime -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "== verifying"
$env:PYTHONPATH = $root
& $py -c "import torch, chatterbox, faster_whisper, speechbrain, df.enhance, audioseal, c2pa, peft, wordfreq, perth, fastapi; assert perth.PerthImplicitWatermarker is not None; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "runtime verification failed" }
$size = (Get-ChildItem -Recurse -File $runtime | Measure-Object -Property Length -Sum).Sum
Write-Host ("== runtime ready: {0} ({1:N1} GB)" -f $runtime, ($size / 1GB))
