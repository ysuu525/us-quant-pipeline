# GPU queue stage 4: does raising sample_count actually buy IC?
#
# Measured score reliability is ~0.75 at sample_count=5 (re-scoring the same
# model with a different batch size correlates only 0.75). Spearman-Brown then
# predicts 5->20 paths lifts reliability to ~0.92, i.e. observed RankIC ~+11%.
# That is a mechanical, model-free gain and the cheapest IC lever left - but it
# has never been measured. bf16 (1.7x) roughly pays for the 4x sampling cost.
#
# sample_count is a frozen official sampling parameter; this run is a
# measurement, not a config change. Any adoption must be recorded in the
# deviation list first.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\gpu_stage4.log -Append

$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

Write-Host "[stage4] waiting for the overnight queue to finish..."
while (-not (Test-Path "outputs\overnight.DONE")) {
    if (Test-Path "outputs\STOP_GPU") { Write-Host "[stage4] STOP_GPU present, exiting."; Stop-Transcript; exit 0 }
    Start-Sleep -Seconds 120
}
Start-Sleep -Seconds 60
Write-Host "[stage4] overnight queue done, starting."

function Invoke-WithRetry {
    param([string]$Label, [scriptblock]$Action, [int]$Max = 3)
    for ($a = 1; $a -le $Max; $a++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        if ($a -lt $Max) { Start-Sleep -Seconds 60 }
    }
    Write-Host "[stage4] $Label failed $Max times"
}

# Two folds x {5, 20} paths. fold40 and fold39 both have strong fine-tuned
# readings, so a noise-reduction gain should be visible if it exists.
$jobs = @(
    @{ n = "fold40"; vs = "2022-07-01"; ve = "2022-12-30" },
    @{ n = "fold39"; vs = "2022-01-03"; ve = "2022-06-30" }
)
foreach ($j in $jobs) {
    foreach ($sc in @(20)) {
        $tag = "sc${sc}_lb90_$($j.n)"
        $md = "outputs\$($j.n)_lb90_s0_poolB_universe"
        if (Test-Path "$md\eval_$tag\metrics.json") { Write-Host "SKIP $tag"; continue }
        Write-Host "=== $tag ===" -ForegroundColor Cyan
        Invoke-WithRetry $tag {
            & $PY scripts\evaluate_fold.py --model-dir $md --processed $P `
                --val-start $j.vs --val-end $j.ve --lookback 90 `
                --tag $tag --batch-size 64 --amp bf16 --sample-count $sc
        }
    }
}

Set-Content -Path outputs\gpu_stage4.DONE -Value (Get-Date -Format o)
Write-Host "[stage4] done $(Get-Date -Format o)"
Stop-Transcript
