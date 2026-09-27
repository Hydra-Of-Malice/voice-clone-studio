# Builds build\installer\VoiceCloneStudio-Setup-<version>.exe plus its -N.bin slices (2 GB each).
# Prerequisites: scripts\build-runtime.ps1, scripts\build-models.py, Node.js, Inno Setup 6.3+
#   powershell -ExecutionPolicy Bypass -File scripts\build-installer.ps1 [-SkipUi]
param([switch]$SkipUi, [string]$Iscc)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
$outDir = Join-Path $build 'installer'
$runtime = Join-Path $build 'runtime-cu128'
$models = Join-Path $build 'models'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$m = Select-String -Path (Join-Path $root 'vc\__init__.py') -Pattern '__version__\s*=\s*"([^"]+)"' | Select-Object -First 1
$version = $m.Matches[0].Groups[1].Value
Write-Host "== Voice Clone Studio $version"

if (-not (Test-Path (Join-Path $runtime 'python.exe'))) { throw 'build\runtime is missing: run scripts\build-runtime.ps1' }
if (-not (Test-Path (Join-Path $models 'bundle.json'))) { throw 'build\models is missing: run scripts\build-models.py' }

if (-not $SkipUi) {
    Write-Host '== web UI'
    Push-Location (Join-Path $root 'web')
    try {
        & npm.cmd install --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
        & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'npm run build failed' }
    } finally { Pop-Location }
}
if (-not (Test-Path (Join-Path $root 'web\dist\index.html'))) { throw 'web\dist\index.html is missing' }

Write-Host '== app icon'
& (Join-Path $runtime 'python.exe') (Join-Path $PSScriptRoot 'make-icon.py') (Join-Path $root 'installer\app.ico')

if (-not $Iscc) {
    $cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
    $Iscc = if ($cmd) { $cmd.Source } else {
        @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
            Where-Object { Test-Path $_ } | Select-Object -First 1
    }
}
if (-not $Iscc) { throw 'Inno Setup not found: winget install --id JRSoftware.InnoSetup -e' }

Get-ChildItem $outDir -Filter "VoiceCloneStudio-Setup-$version*" | Remove-Item -Force
Write-Host '== pre-reading runtime + models so the antivirus scan cache is warm for the compiler'
& (Join-Path $runtime 'python.exe') (Join-Path $PSScriptRoot 'prewarm.py') $runtime $models
Write-Host "== Inno Setup compile ($Iscc)"
& $Iscc "/DAppVersion=$version" "/DOutputDir=$outDir" "/DRuntimeDir=$runtime" "/DModelsDir=$models" '/Qp' (Join-Path $root 'installer\VoiceCloneStudio.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }

$files = Get-ChildItem $outDir -Filter "VoiceCloneStudio-Setup-$version*"
$gb = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1GB, 2)
Write-Host ("Done: {0} files, {1} GB in {2}" -f $files.Count, $gb, $outDir) -ForegroundColor Green
$files | ForEach-Object { "{0,10:N0} MB  {1}" -f ($_.Length / 1MB), $_.Name }
