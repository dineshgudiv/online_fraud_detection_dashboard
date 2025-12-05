$ErrorActionPreference = "Stop"

# Ensure we run from the project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPath = Join-Path $scriptDir ".venv"

# 1) Create venv if missing
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment in .venv..."
    python -m venv $venvPath
}

# 2) Activate venv
Write-Host "Activating virtual environment..."
. "$venvPath\Scripts\Activate.ps1"

# 3) Install dependencies once (skipped on subsequent runs if streamlit exists)
$reqFile = Join-Path $scriptDir "requirements.txt"
$streamlitExe = Join-Path $venvPath "Scripts\streamlit.exe"

$shouldInstall = (Test-Path $reqFile) -and -not (Test-Path $streamlitExe)
if ($shouldInstall) {
    Write-Host "Installing Python dependencies from requirements.txt ..."
    python -m pip install --upgrade pip
    python -m pip install -r $reqFile
}

# 4) Start FastAPI backend
Write-Host "Starting FastAPI backend on http://127.0.0.1:8000 ..."
$apiCmd = "cd `"$scriptDir`"; . .\.venv\Scripts\Activate.ps1; uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd

Start-Sleep -Seconds 2

# 5) Start Streamlit dashboard
Write-Host "Starting Streamlit dashboard on http://localhost:8502 ..."
$dashCmd = "cd `"$scriptDir`"; . .\.venv\Scripts\Activate.ps1; streamlit run dashboard/app.py --server.port 8502"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $dashCmd

Write-Host ""
Write-Host "Backend + dashboard launched. Close the opened PowerShell windows to stop them."
