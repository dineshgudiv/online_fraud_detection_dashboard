$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

$inputPath = Join-Path $repoRoot "grafana\\dashboards\\fraud_ai_model_training_dashboard.json"
$outputPath = Join-Path $repoRoot "grafana\\dashboards\\fraud_ai_model_training_dashboard.UI.json"
$converter = Join-Path $repoRoot "tools\\convert_grafana_dashboard.py"

python $converter --input $inputPath --output $outputPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dashboard conversion failed."
    exit $LASTEXITCODE
}

$item = Get-Item -LiteralPath $outputPath
Write-Host "UI dashboard written: $($item.FullName)"
Write-Host ("Size: {0} bytes" -f $item.Length)
