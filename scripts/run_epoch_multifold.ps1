# Multi-fold confirmation of the epoch-vs-RankIC finding (ledger 2026-08-30).
# Q1 showed on 2 folds that RankIC at epoch 1 already matches or beats the
# fully-trained model, while the inner generative loss keeps improving. Two
# folds and one score per point cannot separate that from sampling noise
# (score reliability ~0.75), so this re-scores e1 / e20 / e30 (or the last
# available epoch) across the remaining modern folds, lb90, one config.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\epoch_multifold.log -Append

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

# eps chosen per fold: epoch 1, a mid/late point, and the last saved epoch.
$jobs = @(
    @{ n = "fold37"; vs = "2021-01-04"; ve = "2021-06-30"; eps = @(1,18,23) },
    @{ n = "fold38"; vs = "2021-07-01"; ve = "2021-12-31"; eps = @(1,20,30) },
    @{ n = "fold39"; vs = "2022-01-03"; ve = "2022-06-30"; eps = @(1,20,30) },
    @{ n = "fold41"; vs = "2023-01-03"; ve = "2023-06-30"; eps = @(1,20,28) },
    @{ n = "fold42"; vs = "2023-07-03"; ve = "2023-12-29"; eps = @(1,20,30) }
)

$i = 0; $total = 15
foreach ($j in $jobs) {
    foreach ($e in $j.eps) {
        $i++
        $es = '{0:d3}' -f $e
        $dst = "outputs\ckpt_probe\$($j.n)_lb90_e$es"
        $tag = "ckpt_$($j.n)_e$es"
        $md = "outputs\$($j.n)_lb90_s0_poolB_universe"
        if (Test-Path "$dst\eval_$tag\metrics.json") {
            Write-Host "=== [$i/$total] SKIP $tag ==="; continue
        }
        Write-Host "=== [$i/$total] $tag ===" -ForegroundColor Cyan
        Invoke-WithRetry "materialize $tag" {
            & $PY scripts\score_checkpoint.py --model-dir $md --epoch $e --out $dst
        }
        Invoke-WithRetry "score $tag" {
            & $PY scripts\evaluate_fold.py --model-dir $dst --processed $P `
                --val-start $j.vs --val-end $j.ve --lookback 90 `
                --tag $tag --batch-size 128 --amp bf16
        }
    }
}

Set-Content -Path outputs\epoch_multifold.DONE -Value (Get-Date -Format o)
Write-Host "=== multi-fold epoch probe complete ($total evals) ===" -ForegroundColor Green
Stop-Transcript
