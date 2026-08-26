$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv-datasets"
$VirtualPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$DatasetPackage = Join-Path $ProjectRoot "ml"

try {
    $PythonCommand = (Get-Command python -ErrorAction Stop).Source
} catch {
    throw "Python 3.11 or newer is required and was not found on PATH."
}

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    Write-Host "Creating dataset-tooling environment at $VirtualEnvironment ..." -ForegroundColor Cyan
    & $PythonCommand -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) { throw "Dataset virtual environment creation failed." }
}

Write-Host "Installing FloodSight dataset tooling ..." -ForegroundColor Cyan
& $VirtualPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Dataset-tooling pip upgrade failed." }
& $VirtualPython -m pip install -e ("{0}[dev]" -f $DatasetPackage)
if ($LASTEXITCODE -ne 0) { throw "Dataset-tooling dependency installation failed." }

Write-Host "FloodSight Phase 3 dataset tooling is installed." -ForegroundColor Green
Write-Host "Run .\scripts\datasets\doctor.ps1 before importing data."
