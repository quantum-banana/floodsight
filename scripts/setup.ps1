$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$VirtualPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$BackendPath = Join-Path $ProjectRoot "backend"
$FrontendPath = Join-Path $ProjectRoot "frontend"

try {
    $PythonCommand = (Get-Command python -ErrorAction Stop).Source
} catch {
    throw "Python 3.11 or newer is required and was not found on PATH."
}

try {
    $NpmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
} catch {
    throw "Node.js and npm are required. Install a supported Node.js release and retry."
}

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    Write-Host "Creating Python virtual environment at $VirtualEnvironment ..." -ForegroundColor Cyan
    & $PythonCommand -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}

Write-Host "Installing FloodSight backend dependencies ..." -ForegroundColor Cyan
& $VirtualPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $VirtualPython -m pip install -e ("{0}[dev]" -f $BackendPath)
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }

Write-Host "Installing FloodSight frontend dependencies ..." -ForegroundColor Cyan
Push-Location $FrontendPath
try {
    & $NpmCommand install
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
} finally {
    Pop-Location
}

Write-Host "FloodSight Phase 0 setup completed successfully." -ForegroundColor Green
Write-Host "Run .\scripts\dev.ps1 to start the development services."

