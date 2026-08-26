$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VirtualPython = Join-Path $ProjectRoot ".venv-datasets\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    throw "Dataset tooling is missing. Run .\scripts\datasets\setup.ps1 first."
}

& $VirtualPython -m floodsight_data.cli prepare-all @args
if ($LASTEXITCODE -ne 0) { throw "One or more datasets are not ready for preparation." }
