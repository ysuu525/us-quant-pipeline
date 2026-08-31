# sample_count 5 -> 20 on the FROZEN backbone (the likely final design, since
# 7-fold E12 shows fine-tuning adds <8%). Measured score reliability at
# sample_count=5 is ~0.75; Spearman-Brown predicts ~0.92 at 20 paths, i.e.
# roughly +11% observed RankIC purely from averaging away sampling noise.
# Runs alongside the CPU-bound ridge probe - the GPU is idle (210 MHz, 15 W).
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\sc20_zeroshot.log -Append
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

$folds = @(
    @{ n = "fold40"; vs = "2022-07-01"; ve = "2022-12-30" },
    @{ n = "fold39"; vs = "2022-01-03"; ve = "2022-06-30" },
    @{ n = "fold36"; vs = "2020-07-01"; ve = "2020-12-31" }
)
foreach ($f in $folds) {
    if (Test-Path "outputs\STOP_GPU") { Write-Host "STOP_GPU present, exiting."; break }
    $tag = "zs_sc20_$($f.n)"
    if (Test-Path "outputs\zeroshot_base\eval_$tag\metrics.json") { Write-Host "SKIP $tag"; continue }
    Write-Host "=== $tag (sample_count=20) ===" -ForegroundColor Cyan
    for ($a = 1; $a -le 3; $a++) {
        & $PY scripts\evaluate_fold.py --model-dir outputs\zeroshot_base --processed $P `
            --val-start $f.vs --val-end $f.ve --lookback 90 `
            --tag $tag --batch-size 32 --amp bf16 --sample-count 20
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 60
    }
}
Set-Content -Path outputs\sc20_zeroshot.DONE -Value (Get-Date -Format o)
Write-Host "=== sample_count=20 zero-shot done ===" -ForegroundColor Green
Stop-Transcript
