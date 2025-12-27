$ErrorActionPreference = "Stop"

$container = "fraud-frontend"
$webroot = "/usr/share/nginx/html"

# Target Grafana settings
$grafanaHost = "http://127.0.0.1:3000"
$targetUid = "fraud-ai-training"

# Temp folder
$tmp = Join-Path $PWD "_tmp_frontend_patch"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null

Write-Host "1) Copying frontend build out of container..."
docker cp "$container`:$webroot" $tmp

$localRoot = Join-Path $tmp "html"
if (-not (Test-Path $localRoot)) {
  # docker cp may create a folder named after last path element
  $localRoot = Join-Path $tmp "html"
}

Write-Host "2) Searching for embed patterns..."
$patterns = @("d-solo", "panelId", "grafana", "Model Training", ":3000", "localhost:3000")
$hits = Get-ChildItem -Path $localRoot -Recurse -File |
  Where-Object { $_.Extension -in @(".html", ".js", ".css", ".map") } |
  Select-String -Pattern $patterns -SimpleMatch -CaseSensitive:$false -ErrorAction SilentlyContinue

if (-not $hits) {
  Write-Host "No direct matches found with simple patterns. Running a broader case-insensitive scan for 'solo'/'panel'/'iframe'..." -ForegroundColor Yellow
  $hits = Get-ChildItem -Path $localRoot -Recurse -File |
    Where-Object { $_.Extension -in @(".html", ".js", ".css", ".map") } |
    Select-String -Pattern @("solo", "panel", "iframe", "graf") -CaseSensitive:$false -ErrorAction SilentlyContinue
}

if ($hits) {
  Write-Host ("Found {0} matching lines (showing first 30):" -f $hits.Count)
  $hits | Select-Object -First 30 | ForEach-Object { "{0}:{1}  {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim() } | Write-Host
} else {
  Write-Host "Still no matches in built assets. This frontend may be fetching embed URLs from a backend API at runtime." -ForegroundColor Yellow
  Write-Host "Proceeding with a safe patch anyway: replace localhost:3000 and any /d-solo/<uid>/ occurrences near panelId 61-65." -ForegroundColor Yellow
}

# Regex for panelId variants (plain, URL-encoded, or JS escaped)
$panelRegex = [regex]'panelId(?:=|%3D|\\u003d)(6[1-5])'

$navItems = @(
  @{ id = "dashboard"; label = "Home"; prefix = "01" },
  @{ id = "live"; label = "Live Monitor"; prefix = "02" },
  @{ id = "investigations"; label = "Predict / Investigate"; prefix = "03" },
  @{ id = "security"; label = "Data Preprocessing & Security"; prefix = "04" },
  @{ id = "modelTraining"; label = "Model Training"; prefix = "05" },
  @{ id = "alerts"; label = "Alerts & Audit Logs"; prefix = "06" },
  @{ id = "realTimeStream"; label = "Real-time Stream"; prefix = "07" },
  @{ id = "liveGraphs"; label = "Live Graphs"; prefix = "08" },
  @{ id = "advanced"; label = "Advanced"; prefix = "09" },
  @{ id = "transactionAnalysis"; label = "Transaction Analysis"; prefix = "10" },
  @{ id = "thresholds"; label = "Threshold & Mode Tuning"; prefix = "11" },
  @{ id = "modelHealth"; label = "Model Health & Drift"; prefix = "12" },
  @{ id = "dataExplorer"; label = "Data Explorer"; prefix = "13" },
  @{ id = "ensemble"; label = "Ensemble Status"; prefix = "14" },
  @{ id = "globalNetwork"; label = "Enhanced Global Fraud Network"; prefix = "15" },
  @{ id = "fraudRings"; label = "Fraud Rings"; prefix = "16" },
  @{ id = "fraudInvestigation"; label = "Fraud Investigation"; prefix = "17" },
  @{ id = "ruleEngine"; label = "Rule Engine"; prefix = "18" },
  @{ id = "caseManagement"; label = "Case Management"; prefix = "19" },
  @{ id = "batchScoring"; label = "Batch Scoring"; prefix = "20" },
  @{ id = "executiveReport"; label = "Executive Report"; prefix = "21" },
  @{ id = "whatIf"; label = "What-If Simulator"; prefix = "22" },
  @{ id = "dataArchive"; label = "Data Archive"; prefix = "23" },
  @{ id = "pipeline"; label = "Real-Time Fraud Decision Pipeline"; prefix = "24" }
)

$navArray = $navItems | ForEach-Object {
  "{id:`"$($_.id)`",label:`"$($_.label)`",prefix:`"$($_.prefix)`"}"
}
$navLiteral = "const RT=[" + ($navArray -join ",") + "];"

$placeholder = "const HT=Object.fromEntries(RT.map(r=>[r.id,r.label]));function qY({page:e}){const t=HT[e]||`"Coming Soon`";return C.jsxs(`"section`",{className:`"space-y-4`",children:[C.jsx(`"h1`",{className:`"text-2xl font-semibold text-white`",children:t}),C.jsx(`"p`",{className:`"text-sm text-slate-400`",children:`"This page is enabled in the menu. Implementation is pending.`"})]})}"

$gxExtra = "realTimeStream:qY,liveGraphs:qY,advanced:qY,transactionAnalysis:qY,thresholds:qY,modelHealth:qY,dataExplorer:qY,ensemble:qY,globalNetwork:qY,fraudRings:qY,fraudInvestigation:qY,ruleEngine:qY,caseManagement:qY,batchScoring:qY,executiveReport:qY,whatIf:qY,dataArchive:qY,pipeline:qY"

# Patch files
Write-Host "3) Patching built files..."
$files = Get-ChildItem -Path $localRoot -Recurse -File | Where-Object { $_.Extension -in @(".html", ".js", ".css", ".map") }

$changed = 0
foreach ($f in $files) {
  $text = Get-Content $f.FullName -Raw -ErrorAction SilentlyContinue
  if ($null -eq $text) { continue }

  $orig = $text

  # Replace Grafana host if hardcoded
  $text = $text -replace "http://localhost:3000", $grafanaHost
  $text = $text -replace "https://localhost:3000", $grafanaHost

  # Replace UID ONLY in places that reference panelIds 61-65
  if ($panelRegex.IsMatch($text)) {
    # Replace /d-solo/<someuid>/... with /d-solo/fraud-ai-training/...
    $text = [regex]::Replace($text, "(d-solo/)([A-Za-z0-9_-]+)(/)", "`$1$targetUid`$3")
    # Also handle absolute paths like /d-solo/<uid>/
    $text = [regex]::Replace($text, "(/d-solo/)([A-Za-z0-9_-]+)(/)", "`$1$targetUid`$3")
  }

  # Replace hardcoded dashboard slug for Grafana embeds
  $text = $text -replace 'qX="[^"]+"', 'qX="fraud-ai-model-training"'

  # Expand navigation items
  if ($text -match 'const RT=\\[') {
    $text = [regex]::Replace(
      $text,
      'const RT=\\[[^\\]]*\\];',
      "$navLiteral$placeholder",
      1
    )
  }

  # Ensure router has placeholders for extra pages
  if ($text -match 'const GX=\\{') {
    $text = [regex]::Replace(
      $text,
      'const GX=\\{[^\\}]*\\};',
      'const GX={dashboard:nG,live:iG,investigations:aG,security:zX,modelTraining:KX,alerts:UX,' + $gxExtra + '};',
      1
    )
  }

  # Pass current page id to routed component
  $text = $text -replace 'C\\.jsx\\(r,\\{\\}\\)', 'C.jsx(r,{page:e})'

  if ($text -ne $orig) {
    Set-Content -Path $f.FullName -Value $text -Encoding UTF8
    $changed++
  }
}

Write-Host "Patched files count: $changed"

Write-Host "4) Copying patched build back into container..."
docker cp "$localRoot\\." "$container`:$webroot"

Write-Host "5) Restarting container..."
docker restart $container | Out-Null

Write-Host "Done. Open: http://127.0.0.1:5173 and re-check the Model Training page."
