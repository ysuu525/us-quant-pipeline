# E12 extended to all 7 modern folds. The 3-fold read (zero-shot +0.02240 vs
# fine-tuned +0.02563) is what the "fine-tuning barely helps" conclusion rests
# on, and this project has twice been burned by 3-fold conclusions. Skips folds
# already scored.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
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
    @{ n = "fold39"; vs = "2022-01-03"; ve = "2022-06-30" },
    @{ n = "fold40"; vs = "2022-07-01"; ve = "2022-12-30" },
    @{ n = "fold41"; vs = "2023-01-03"; ve = "2023-06-30" },
    @{ n = "fold42"; vs = "2023-07-03"; ve = "2023-12-29" }
)
foreach ($f in $folds) {
    $tag = "zeroshot_$($f.n)"
    if (Test-Path "outputs\zeroshot_base\eval_$tag\metrics.json") {
        Write-Host "SKIP $tag (done)"; continue
    }
    Write-Host "=== ZERO-SHOT $($f.n) ===" -ForegroundColor Cyan
    Invoke-WithRetry $tag {
        & $PY scripts\evaluate_fold.py --model-dir outputs\zeroshot_base --processed $P `
            --val-start $f.vs --val-end $f.ve --lookback 90 `
            --tag $tag --batch-size 128 --amp bf16
    }
}
Set-Content -Path outputs\zeroshot_all7.DONE -Value (Get-Date -Format o)
Write-Host "=== zero-shot 7 folds complete ===" -ForegroundColor Green
