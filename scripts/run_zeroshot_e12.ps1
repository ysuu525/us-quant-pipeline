# E12: zero-shot baseline. Score the UNMODIFIED pretrained Kronos-small on the
# same validation windows as the fine-tuned models, one config (bf16/bs128/lb90).
# Decides what fine-tuning actually contributes:
#   zero-shot ~= e1 ~= e30  -> fine-tuning adds nothing; value is in pretraining
#   zero-shot << e1         -> fine-tuning matters but saturates after 1 epoch
# Three folds spanning the fine-tuned range (fold39 strongest, fold42 weakest).
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\zeroshot_e12.log -Append

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
    @{ n = "fold40"; vs = "2022-07-01"; ve = "2022-12-30" },
    @{ n = "fold39"; vs = "2022-01-03"; ve = "2022-06-30" },
    @{ n = "fold42"; vs = "2023-07-03"; ve = "2023-12-29" }
)

$i = 0
foreach ($f in $folds) {
    $i++
    $tag = "zeroshot_$($f.n)"
    if (Test-Path "outputs\zeroshot_base\eval_$tag\metrics.json") {
        Write-Host "=== [$i/3] SKIP $tag ==="; continue
    }
    Write-Host "=== [$i/3] ZERO-SHOT $($f.n) ===" -ForegroundColor Cyan
    Invoke-WithRetry $tag {
        & $PY scripts\evaluate_fold.py --model-dir outputs\zeroshot_base --processed $P `
            --val-start $f.vs --val-end $f.ve --lookback 90 `
            --tag $tag --batch-size 128 --amp bf16
    }
}

Set-Content -Path outputs\zeroshot_e12.DONE -Value (Get-Date -Format o)
Write-Host "=== E12 zero-shot complete (3 folds) ===" -ForegroundColor Green
Stop-Transcript
