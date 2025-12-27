$ErrorActionPreference = "Continue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

$inputPath = Join-Path $repoRoot "grafana\\dashboards\\fraud_ai_model_training_dashboard.json"
$outputPath = Join-Path $repoRoot "grafana\\dashboards\\fraud_ai_model_training_dashboard.UI.json"
$converter = Join-Path $repoRoot "tools\\convert_grafana_dashboard.py"

$hadFailure = $false

python $converter --input $inputPath --output $outputPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: Dashboard conversion"
    $hadFailure = $true
} else {
    Write-Host "OK: Dashboard conversion -> $outputPath"
}

function Test-Endpoint {
    param(
        [string]$Label,
        [string]$Url
    )
    try {
        $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 5 -UseBasicParsing
        Write-Host ("OK: {0} ({1})" -f $Label, $response.StatusCode)
        return $true
    } catch {
        Write-Host ("FAIL: {0} - {1}" -f $Label, $_.Exception.Message)
        return $false
    }
}

if (-not (Test-Endpoint -Label "Grafana health" -Url "http://127.0.0.1:3000/api/health")) {
    $hadFailure = $true
}

if (-not (Test-Endpoint -Label "Prometheus health" -Url "http://127.0.0.1:9090/-/healthy")) {
    $hadFailure = $true
}

if (-not (Test-Endpoint -Label "Prometheus up query" -Url "http://127.0.0.1:9090/api/v1/query?query=up")) {
    $hadFailure = $true
}

if ($hadFailure) {
    exit 1
}
