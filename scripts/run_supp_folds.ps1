# Supplemental modern folds (user-approved, ledger 2026-08-30): folds 36-39 x
# {lb90, lb200}, pool B, seed 0. Purpose: raise the modern-era sample from 3 to
# 7 folds so the paired criterion has the power to settle lb90 vs lb200.
# All validation windows end 2022-06-30 => the sealed OOS window (2024-01-01)
# is untouched. Training only; scoring runs separately under one AMP config.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline

$lock = "outputs\supp_folds.lock"
if (Test-Path $lock) {
    $other = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($other -and (Get-Process -Id ([int]$other) -ErrorAction SilentlyContinue)) {
        Write-Host "Another run is active (PID $other). Exiting."; exit 0
    }
}
Set-Content -Path $lock -Value $PID -Encoding utf8
Start-Transcript -Path outputs\supp_folds.log -Append

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
    @{ n = "fold36"; ts = "2017-07-03"; te = "2020-06-22" },
    @{ n = "fold37"; ts = "2018-01-02"; te = "2020-12-22" },
    @{ n = "fold38"; ts = "2018-07-02"; te = "2021-06-22" },
    @{ n = "fold39"; ts = "2019-01-02"; te = "2021-12-22" }
)

$i = 0; $total = 8
foreach ($f in $folds) {
    foreach ($lb in @(90, 200)) {
        $i++
        $out = "outputs\$($f.n)_lb${lb}_s0_poolB_universe"
        Write-Host "=== [$i/$total] TRAIN $($f.n) lb$lb poolB ===" -ForegroundColor Cyan
        Invoke-WithRetry "TRAIN $($f.n) lb$lb" {
            & $PY -m kronos_ft.train --panel "$P\panel_kronos_adj.parquet" `
                --index-parquet "$P\market_index.parquet" `
                --train-start $f.ts --train-end $f.te `
                --lookback $lb --seed 0 --stage both --out $out `
                --universe-parquet "$P\universe.parquet" `
                --index-cache "$P\index_cache\lb${lb}_full.parquet"
        }
    }
}

Set-Content -Path outputs\supp_folds_TRAIN.DONE -Value (Get-Date -Format o)
Write-Host "=== supplemental training complete (folds 36-39 x 2 arms) ===" -ForegroundColor Green
Remove-Item $lock -ErrorAction SilentlyContinue
Stop-Transcript
