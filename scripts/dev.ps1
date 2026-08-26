$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackendPath = Join-Path $ProjectRoot "backend"
$FrontendPath = Join-Path $ProjectRoot "frontend"
$VirtualPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$NodeModules = Join-Path $FrontendPath "node_modules"

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    throw "Python environment not found. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $NodeModules)) {
    throw "Frontend dependencies not found. Run .\scripts\setup.ps1 first."
}

try {
    $NpmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
} catch {
    throw "npm.cmd was not found on PATH. Install Node.js and retry."
}

Write-Host "Starting FloodSight API at http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$BackendProcess = Start-Process -FilePath $VirtualPython `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $BackendPath -PassThru -NoNewWindow

Write-Host "Starting FloodSight frontend at http://127.0.0.1:5173 ..." -ForegroundColor Cyan
$FrontendProcess = Start-Process -FilePath $NpmCommand `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") `
    -WorkingDirectory $FrontendPath -PassThru -NoNewWindow

Write-Host "Both services are running. Press Ctrl+C to stop them." -ForegroundColor Green

try {
    while (-not $BackendProcess.HasExited -and -not $FrontendProcess.HasExited) {
        Start-Sleep -Seconds 1
        $BackendProcess.Refresh()
        $FrontendProcess.Refresh()
    }

    if ($BackendProcess.HasExited) {
        throw "The backend process exited with code $($BackendProcess.ExitCode)."
    }
    throw "The frontend process exited with code $($FrontendProcess.ExitCode)."
} finally {
    foreach ($Process in @($BackendProcess, $FrontendProcess)) {
        if ($null -ne $Process -and -not $Process.HasExited) {
            Stop-Process -Id $Process.Id
        }
    }
}

