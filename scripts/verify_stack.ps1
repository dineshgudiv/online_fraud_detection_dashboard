$ErrorActionPreference = "Continue"

$repoRoot = Resolve-Path $PSScriptRoot\..
Set-Location $repoRoot

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host $Title
    Write-Host ("-" * $Title.Length)
}

Write-Section "Starting stack"
docker compose down -v | Out-Null
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: docker compose up"
    exit 1
}

$services = @("db", "backend", "web", "grafana", "prometheus")
$timeoutSeconds = 180
$deadline = (Get-Date).AddSeconds($timeoutSeconds)
$status = @{}

Write-Section "Waiting for health checks"
while ((Get-Date) -lt $deadline) {
    $allHealthy = $true
    foreach ($svc in $services) {
        $cid = (docker compose ps -q $svc) 2>$null
        if (-not $cid) {
            $status[$svc] = "missing"
            $allHealthy = $false
            continue
        }
        $health = (docker inspect --format "{{.State.Health.Status}}" $cid) 2>$null
        if (-not $health) {
            $health = "no-healthcheck"
        }
        $status[$svc] = $health
        if ($health -ne "healthy" -and $health -ne "no-healthcheck") {
            $allHealthy = $false
        }
    }
    if ($allHealthy) { break }
    Start-Sleep -Seconds 5
}

$unhealthy = $status.GetEnumerator() | Where-Object { $_.Value -ne "healthy" -and $_.Value -ne "no-healthcheck" }
if ($unhealthy) {
    Write-Host "Healthcheck status:"
    $status.GetEnumerator() | Sort-Object Name | ForEach-Object { "{0}: {1}" -f $_.Name, $_.Value } | Write-Host
}

Write-Section "Endpoint checks"
function Test-Endpoint {
    param([string]$Label, [string]$Url)
    try {
        $resp = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 5 -UseBasicParsing
        Write-Host ("PASS: {0} ({1})" -f $Label, $resp.StatusCode)
        return $true
    } catch {
        Write-Host ("FAIL: {0} - {1}" -f $Label, $_.Exception.Message)
        return $false
    }
}

$endpointFailures = @()
if (-not (Test-Endpoint -Label "Web" -Url "http://localhost:5173")) { $endpointFailures += "web" }
if (-not (Test-Endpoint -Label "Backend health" -Url "http://localhost:8001/health")) { $endpointFailures += "backend" }
if (-not (Test-Endpoint -Label "Grafana health" -Url "http://localhost:3000/api/health")) { $endpointFailures += "grafana" }
if (-not (Test-Endpoint -Label "Prometheus ready" -Url "http://localhost:9090/-/ready")) { $endpointFailures += "prometheus" }

$allOk = (-not $unhealthy) -and ($endpointFailures.Count -eq 0)
if ($allOk) {
    Write-Host "PASS: Stack is healthy."
    exit 0
}

Write-Section "Recent logs (tail 50) for failing services"
$servicesToLog = @()
if ($unhealthy) {
    $servicesToLog += $unhealthy | ForEach-Object { $_.Name }
}
$servicesToLog += $endpointFailures
$servicesToLog = $servicesToLog | Sort-Object -Unique

foreach ($svc in $servicesToLog) {
    Write-Host ""
    Write-Host "Logs for $svc"
    Write-Host "--------------"
    docker compose logs --tail 50 $svc
}

exit 1
