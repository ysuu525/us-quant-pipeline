# Score the supplemental folds 36-39 (both arms) under the SAME config as the
# probe folds (bf16 / bs128 / sample_count 5), so folds 36-42 form one
# comparable 7-fold modern-era set for the lb90-vs-lb200 paired criterion.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\supp_scoring.log -Append

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

$folds = @(
    @{ n = "fold36"; vs = "2020-07-01"; ve = "2020-12-31" },
    @{ n = "fold37"; vs = "2021-01-04"; ve = "2021-06-30" },
    @{ n = "fold38"; vs = "2021-07-01"; ve = "2021-12-31" },
    @{ n = "fold39"; vs = "2022-01-03"; ve = "2022-06-30" }
)

$i = 0
foreach ($lb in @(90, 200)) {
    foreach ($f in $folds) {
        $i++
        $md = "outputs\$($f.n)_lb${lb}_s0_poolB_universe"
        $tag = "amp_lb${lb}_$($f.n)"
        if (Test-Path "$md\eval_$tag\metrics.json") {
            Write-Host "=== [$i/8] SKIP $tag (already scored) ==="; continue
        }
        Write-Host "=== [$i/8] SCORE $tag ===" -ForegroundColor Cyan
        Invoke-WithRetry $tag {
            & $PY scripts\evaluate_fold.py --model-dir $md --processed $P `
                --val-start $f.vs --val-end $f.ve --lookback $lb `
                --tag $tag --batch-size 128 --amp bf16
        }
    }
}

Set-Content -Path outputs\supp_scoring.DONE -Value (Get-Date -Format o)
Write-Host "=== supplemental scoring complete (8 evals) ===" -ForegroundColor Green
Stop-Transcript
