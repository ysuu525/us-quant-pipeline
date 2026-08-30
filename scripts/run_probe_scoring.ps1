# Re-score probe folds 40-42 for BOTH arms under ONE scoring config, so the
# lb90 / lb200 / ensemble comparison is apples-to-apples. Re-scoring the lb90
# arm also serves as the post-speedup control the ledger requires (2026-08-27:
# "留待全部定案后再动并重跑对照").
param([string]$Amp = "bf16", [int]$BatchSize = 384)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\probe_scoring.log -Append

$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

function Invoke-WithRetry {
    param([string]$Label, [scriptblock]$Action, [int]$MaxAttempts = 3)
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        if ($a -lt $MaxAttempts) { Start-Sleep -Seconds 60 }
    }
    throw "$Label failed $MaxAttempts times"
}

$folds = @(
    @{ n = "fold40"; vs = "2022-07-01"; ve = "2022-12-30" },
    @{ n = "fold41"; vs = "2023-01-03"; ve = "2023-06-30" },
    @{ n = "fold42"; vs = "2023-07-03"; ve = "2023-12-29" }
)

foreach ($lb in @(90, 200)) {
    foreach ($f in $folds) {
        $md = "outputs\$($f.n)_lb${lb}_s0_poolB_universe"
        if (-not (Test-Path "$md\predictor_final")) { throw "missing model: $md" }
        $tag = "amp_lb${lb}_$($f.n)"
        Write-Host "=== SCORE $tag (amp=$Amp bs=$BatchSize) ===" -ForegroundColor Cyan
        Invoke-WithRetry $tag {
            & $PY scripts\evaluate_fold.py --model-dir $md --processed $P `
                --val-start $f.vs --val-end $f.ve --lookback $lb `
                --tag $tag --batch-size $BatchSize --amp $Amp
        }
    }
}

Set-Content -Path outputs\probe_scoring.DONE -Value (Get-Date -Format o)
Write-Host "=== probe scoring complete (6 evals, amp=$Amp) ===" -ForegroundColor Green
Stop-Transcript
