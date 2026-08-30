# Ensemble validation (ledger 2026-08-29): train lb200 on the SAME probe folds
# 40-42 so the "lb90+lb200 ensemble +20%" finding can be tested out of the
# selection sample (2022H2-2023H2 instead of the 2003-04 screening folds).
# Training only - scoring is run separately so every arm shares one AMP config.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline

$lock = "outputs\probe_lb200.lock"
if (Test-Path $lock) {
    $other = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($other -and (Get-Process -Id ([int]$other) -ErrorAction SilentlyContinue)) {
        Write-Host "Another run is active (PID $other). Exiting."; exit 0
    }
}
Set-Content -Path $lock -Value $PID -Encoding utf8
Start-Transcript -Path outputs\probe_lb200.log -Append

$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

function Invoke-WithRetry {
    param([string]$Label, [scriptblock]$Action, [int]$MaxAttempts = 3)
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        if ($a -lt $MaxAttempts) {
            Write-Host "!! $Label attempt $a failed (exit $LASTEXITCODE); retry in 60s" -ForegroundColor Yellow
            Start-Sleep -Seconds 60
        }
    }
    throw "$Label failed $MaxAttempts times - manual intervention needed"
}

$folds = @(
    @{ n = "fold40"; ts = "2019-07-01"; te = "2022-06-22" },
    @{ n = "fold41"; ts = "2020-01-02"; te = "2022-12-21" },
    @{ n = "fold42"; ts = "2020-07-01"; te = "2023-06-22" }
)

foreach ($f in $folds) {
    $out = "outputs\$($f.n)_lb200_s0_poolB_universe"
    Write-Host "=== TRAIN $($f.n) lb200 poolB ===" -ForegroundColor Cyan
    Invoke-WithRetry "TRAIN $($f.n) lb200" {
        & $PY -m kronos_ft.train --panel "$P\panel_kronos_adj.parquet" `
            --index-parquet "$P\market_index.parquet" `
            --train-start $f.ts --train-end $f.te `
            --lookback 200 --seed 0 --stage both --out $out `
            --universe-parquet "$P\universe.parquet" `
            --index-cache "$P\index_cache\lb200_full.parquet"
    }
}

Set-Content -Path outputs\probe_lb200_TRAIN.DONE -Value (Get-Date -Format o)
Write-Host "=== lb200 probe training complete (folds 40-42) ===" -ForegroundColor Green
Remove-Item $lock -ErrorAction SilentlyContinue
Stop-Transcript
