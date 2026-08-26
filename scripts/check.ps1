$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VirtualPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FrontendPath = Join-Path $ProjectRoot "frontend"
$NodeModules = Join-Path $FrontendPath "node_modules"

if (-not (Test-Path -LiteralPath $VirtualPython) -or -not (Test-Path -LiteralPath $NodeModules)) {
    throw "Dependencies are missing. Run .\scripts\setup.ps1 before running checks."
}

try {
    $NpmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
} catch {
    throw "npm.cmd was not found on PATH. Install Node.js and retry."
}

Write-Host "Checking backend formatting and imports ..." -ForegroundColor Cyan
& $VirtualPython -m ruff check (Join-Path $ProjectRoot "backend\app") (Join-Path $ProjectRoot "backend\tests")
if ($LASTEXITCODE -ne 0) { throw "Backend lint failed." }

Write-Host "Running backend tests and shared-schema validation ..." -ForegroundColor Cyan
& $VirtualPython -m pytest (Join-Path $ProjectRoot "backend\tests")
if ($LASTEXITCODE -ne 0) { throw "Backend tests failed." }

Push-Location $FrontendPath
try {
    Write-Host "Linting frontend ..." -ForegroundColor Cyan
    & $NpmCommand run lint
    if ($LASTEXITCODE -ne 0) { throw "Frontend lint failed." }

    Write-Host "Running frontend tests ..." -ForegroundColor Cyan
    & $NpmCommand run test
    if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed." }

    Write-Host "Building frontend production bundle ..." -ForegroundColor Cyan
    & $NpmCommand run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
} finally {
    Pop-Location
}

Write-Host "All FloodSight Phase 1 checks passed." -ForegroundColor Green
