# Builds the self-contained Python runtime that the installer ships (nothing is downloaded on the user's PC):
#   build\runtime-<cuda>\python.exe + Lib\site-packages with PyTorch CUDA, Chatterbox and the app dependencies.
# python-build-standalone is relocatable, so the installer can copy the folder anywhere.
# Needs MSVC Build Tools on the BUILD machine only (pkuseg compiles from source); users do not need them.
#
# CUDA 12.8 (cu128) is the default: it is the first build that runs on RTX 50 (Blackwell) cards and it still
# supports RTX 20 / 30 / 40. Chatterbox pins torch 2.6.0, so the pin is overridden (scripts\torch-overrides.txt
# is generated below).
#   powershell -ExecutionPolicy Bypass -File scripts\build-runtime.ps1 [-Cuda cu128] [-Torch 2.7.1]
param([string]$Cuda = "cu128", [string]$Torch = "2.7.1", [string]$PythonTag = "3.11.16+20260924")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build"
$runtime = Join-Path $build "runtime-$Cuda"
$dl = Join-Path $build "downloads"
New-Item -ItemType Directory -Force $build, $dl | Out-Null
$uv = (Get-Command uv -ErrorAction Stop).Source
function Uv { & $uv @args; if ($LASTEXITCODE -ne 0) { throw "uv $($args -join ' ') failed" } }

if (-not (Test-Path (Join-Path $runtime "python.exe"))) {
    $ver, $rel = $PythonTag -split "\+"
    $name = "cpython-$PythonTag-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
    $url = "https://github.com/astral-sh/python-build-standalone/releases/download/$rel/" + $name.Replace("+", "%2B")
    $tgz = Join-Path $dl $name
    if (-not (Test-Path $tgz)) { Write-Host "== downloading $name"; Invoke-WebRequest -Uri $url -OutFile $tgz }
    $tmp = Join-Path $build "pbs-extract"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    & "$env:SystemRoot\System32\tar.exe" -xzf $tgz -C $tmp
    Move-Item (Join-Path $tmp "python") $runtime
}
$py = Join-Path $runtime "python.exe"
Write-Host "== runtime python: $(& $py -c 'import sys; print(sys.version.split()[0])')"

$index = "https://download.pytorch.org/whl/$Cuda"
$overrides = Join-Path $build "torch-overrides.txt"
"torch==$Torch+$Cuda`ntorchaudio==$Torch+$Cuda" | Set-Content -Encoding ascii $overrides

Write-Host "== PyTorch $Torch ($Cuda)"
Uv pip install --python $py --index-url $index "torch==$Torch" "torchaudio==$Torch"
Write-Host "== dependencies"
Uv pip install --python $py --override $overrides --extra-index-url $index --index-strategy unsafe-best-match -r (Join-Path $root "requirements.txt") cython wheel
Write-Host "== pkuseg (built against the installed numpy)"
Uv pip install --python $py --no-build-isolation pkuseg==0.0.25
Write-Host "== Chatterbox (torch pin overridden)"
Uv pip install --python $py --override $overrides --extra-index-url $index --index-strategy unsafe-best-match chatterbox-tts
Uv pip install --python $py "setuptools<81"
Uv pip uninstall --python $py pytest cython wheel

Write-Host "== pruning caches"
Get-ChildItem -Path $runtime -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "== verifying"
$env:PYTHONPATH = $root
& $py -c "import torch, torchaudio, chatterbox, faster_whisper, speechbrain, df.enhance, audioseal, c2pa, peft, wordfreq, perth, fastapi; assert perth.PerthImplicitWatermarker is not None; assert torch.__version__.startswith('$Torch'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'archs', torch.cuda.get_arch_list())"
if ($LASTEXITCODE -ne 0) { throw "runtime verification failed" }
$size = (Get-ChildItem -Recurse -File $runtime | Measure-Object -Property Length -Sum).Sum
Write-Host ("== runtime ready: {0} ({1:N1} GB)" -f $runtime, ($size / 1GB))
