# Training-length probe (ledger 2026-08-30, user-approved). Two questions:
#   Q1 Is the early-stopping metric (inner generative loss) aligned with RankIC?
#      -> score existing epoch checkpoints and plot RankIC vs epoch.
#   Q2 Is 30 epochs a binding ceiling on folds that never early-stop?
#      -> retrain one fold with max_epochs=50 and compare.
# Q1 needs no training at all; it runs first. Waits for the supplemental-fold
# host to finish so it never contends for the GPU.
param([int]$WaitPid = 0)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline

$lock = "outputs\epoch_probe.lock"
if (Test-Path $lock) {
    $o = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($o -and (Get-Process -Id ([int]$o) -ErrorAction SilentlyContinue)) {
        Write-Host "Another run active (PID $o)."; exit 0
    }
}
Set-Content -Path $lock -Value $PID -Encoding utf8
Start-Transcript -Path outputs\epoch_probe.log -Append

if ($WaitPid -gt 0) {
    Write-Host "Waiting for PID $WaitPid ..."
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 120 }
    Start-Sleep -Seconds 120
}

$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

function Invoke-WithRetry {
    param([string]$Label, [scriptblock]$Action, [int]$Max = 3)
    for ($a = 1; $a -le $Max; $a++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        if ($a -lt $Max) { Start-Sleep -Seconds 60 }
    }
    throw "$Label failed $Max times - manual intervention needed"
}

# ---- Q1: RankIC vs epoch on two contrasting folds ----
# fold36 lb90 stopped at epoch 6 (best 1) - inner window spans the COVID crash.
# fold40 lb90 ran the full 30 (best 26) - inner window is the 2022 selloff.
$jobs = @(
    @{ md = "outputs\fold36_lb90_s0_poolB_universe"; eps = @(1,3,6);      vs = "2020-07-01"; ve = "2020-12-31"; n = "fold36" },
    @{ md = "outputs\fold40_lb90_s0_poolB_universe"; eps = @(1,5,10,20,30); vs = "2022-07-01"; ve = "2022-12-30"; n = "fold40" }
)
foreach ($j in $jobs) {
    foreach ($e in $j.eps) {
        $dst = "outputs\ckpt_probe\$($j.n)_lb90_e$('{0:d3}' -f $e)"
        $tag = "ckpt_$($j.n)_e$('{0:d3}' -f $e)"
        Write-Host "=== MATERIALIZE $($j.n) epoch $e ===" -ForegroundColor Cyan
        Invoke-WithRetry "materialize $tag" {
            & $PY scripts\score_checkpoint.py --model-dir $j.md --epoch $e --out $dst
        }
        Write-Host "=== SCORE $tag ===" -ForegroundColor Cyan
        Invoke-WithRetry "score $tag" {
            & $PY scripts\evaluate_fold.py --model-dir $dst --processed $P `
                --val-start $j.vs --val-end $j.ve --lookback 90 `
                --tag $tag --batch-size 128 --amp bf16
        }
    }
}
Set-Content -Path outputs\epoch_probe_Q1.DONE -Value (Get-Date -Format o)

# ---- Q2: 50-epoch budget on a fold that never early-stopped ----
Write-Host "=== TRAIN fold40 lb200 max_epochs=50 ===" -ForegroundColor Cyan
Invoke-WithRetry "train e50" {
    & $PY -m kronos_ft.train --panel "$P\panel_kronos_adj.parquet" `
        --index-parquet "$P\market_index.parquet" `
        --train-start 2019-07-01 --train-end 2022-06-22 `
        --lookback 200 --seed 0 --stage both --max-epochs 50 `
        --out outputs\fold40_lb200_s0_e50_poolB_universe `
        --universe-parquet "$P\universe.parquet" `
        --index-cache "$P\index_cache\lb200_full.parquet"
}
Invoke-WithRetry "score e50" {
    & $PY scripts\evaluate_fold.py --model-dir outputs\fold40_lb200_s0_e50_poolB_universe `
        --processed $P --val-start 2022-07-01 --val-end 2022-12-30 --lookback 200 `
        --tag e50_lb200_fold40 --batch-size 128 --amp bf16
}

Set-Content -Path outputs\epoch_probe.DONE -Value (Get-Date -Format o)
Write-Host "=== epoch probe complete ===" -ForegroundColor Green
Remove-Item $lock -ErrorAction SilentlyContinue
Stop-Transcript
