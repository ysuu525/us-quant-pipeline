# Recency probe (user-approved deviation, ledger 2026-08-28): folds 40-42,
# lb90 + pool B + seed 0. Answers "does the signal still exist in the 2020s"
# without touching the sealed OOS window (all val windows end 2023-12-29).
# Waits for the lb200 screening host (PID passed as -WaitPid) to exit first.
# Idempotent: rerun resumes from checkpoints (train.py completion detection).
param([int]$WaitPid = 0)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline

# Single-instance lock. Two concurrent runs write the same checkpoint files and
# corrupt them (happened 2026-08-28: a stale waiter and a fresh launch collided).
$lock = "outputs\probe_recent.lock"
if (Test-Path $lock) {
    $other = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($other -and (Get-Process -Id ([int]$other) -ErrorAction SilentlyContinue)) {
        Write-Host "Another probe run is active (PID $other). Exiting."
        exit 0
    }
}
Set-Content -Path $lock -Value $PID -Encoding utf8

Start-Transcript -Path outputs\probe_recent.log -Append

$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

if ($WaitPid -gt 0) {
    Write-Host "Waiting for screening host PID $WaitPid to exit..."
    while ($true) {
        $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
        if ($null -eq $proc -or $proc.ProcessName -ne "powershell") { break }
        Start-Sleep -Seconds 120
    }
    Write-Host "Screening host exited. Cooling down 180s before training."
    Start-Sleep -Seconds 180
}

# Wait (max 30 min) for the auto-generated round-1 comparison, then pick the
# probe arm: lb200 only if the frozen criterion says it beats lb90 (see
# choose_probe_arm.py); tie or missing report falls back to lb90.
$cmp = "outputs\screen_lookback_round1.json"
$deadline = (Get-Date).AddMinutes(30)
while (-not (Test-Path $cmp) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 60 }
$LB = (& $PY scripts\choose_probe_arm.py).Trim()
if ($LB -ne "200") { $LB = "90" }
Write-Host "Probe arm chosen: lb$LB (compare json present: $(Test-Path $cmp))"

# Retry wrapper. Training resumes from checkpoints, so a retry is nearly free.
# Motivation: driver-level faults (nvlddmkm 153, CUDA illegal memory access when
# a co-running CUDA process is force-killed) took this batch down twice on
# 2026-08-28; transient hardware faults should not kill an overnight run.
function Invoke-WithRetry {
    param([string]$Label, [scriptblock]$Action, [int]$MaxAttempts = 3)
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        if ($a -lt $MaxAttempts) {
            Write-Host "!! $Label attempt $a failed (exit $LASTEXITCODE); retrying in 60s" -ForegroundColor Yellow
            Start-Sleep -Seconds 60
        }
    }
    throw "$Label failed $MaxAttempts times - manual intervention needed"
}

# Boundaries from crsp_pipeline.splits.walk_forward_folds (oos_start=2024-01-01)
$folds = @(
    @{ n = "fold40"; ts = "2019-07-01"; te = "2022-06-22"; vs = "2022-07-01"; ve = "2022-12-30" },
    @{ n = "fold41"; ts = "2020-01-02"; te = "2022-12-21"; vs = "2023-01-03"; ve = "2023-06-30" },
    @{ n = "fold42"; ts = "2020-07-01"; te = "2023-06-22"; vs = "2023-07-03"; ve = "2023-12-29" }
)

foreach ($f in $folds) {
    $out = "outputs\$($f.n)_lb${LB}_s0_poolB_universe"
    Write-Host "=== TRAIN probe $($f.n) lb$LB poolB ===" -ForegroundColor Cyan
    Invoke-WithRetry "TRAIN $($f.n)" {
        & $PY -m kronos_ft.train --panel "$P\panel_kronos_adj.parquet" `
            --index-parquet "$P\market_index.parquet" `
            --train-start $f.ts --train-end $f.te `
            --lookback $LB --seed 0 --stage both --out $out `
            --universe-parquet "$P\universe.parquet" `
            --index-cache "$P\index_cache\lb${LB}_full.parquet"
    }

    Write-Host "=== EVAL probe $($f.n) ===" -ForegroundColor Cyan
    Invoke-WithRetry "EVAL $($f.n)" {
        & $PY scripts\evaluate_fold.py --model-dir $out --processed $P `
            --val-start $f.vs --val-end $f.ve --lookback $LB `
            --tag "probe_recent_lb${LB}_$($f.n)"
    }
}

Set-Content -Path outputs\probe_recent.DONE -Value (Get-Date -Format o)
Write-Host "=== Recency probe complete (folds 40-42 trained + evaluated) ===" -ForegroundColor Green
Remove-Item $lock -ErrorAction SilentlyContinue
Stop-Transcript
