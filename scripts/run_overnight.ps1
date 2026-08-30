# Overnight queue (2026-08-31). Runs sequentially so the GPU is never shared -
# co-running CUDA jobs already caused one driver-level crash on this machine.
# Every stage is idempotent (skips finished work), retries transient failures,
# and writes a .DONE marker.
#
# Order is by decision value:
#   1. zero-shot on all 7 folds  -> completes "does fine-tuning add anything"
#      (currently rests on 3 folds; this project has been burned twice by that)
#   2. epoch probe on 5 more folds -> confirms "e1 ~= e30" beyond 2 folds
#   3. ridge linear probe, 7 folds -> does the frozen representation carry any
#      cross-sectional signal at all (the MLP head overfit; 513 params cannot)
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
Set-Location F:\quant\us-quant-pipeline
Start-Transcript -Path outputs\overnight.log -Append

$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

# Wait for any GPU job still running (the fold40 diagnostic) to finish.
Write-Host "[overnight] waiting for GPU to free up..."
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*diag_representation*' -or
                       $_.CommandLine -like '*evaluate_fold*' -or
                       $_.CommandLine -like '*train_rank_head*' }
    if (-not $busy) { break }
    Start-Sleep -Seconds 60
}
Write-Host "[overnight] GPU free, starting."

function Stage {
    param([string]$Name, [string]$Marker, [scriptblock]$Action)
    if (Test-Path $Marker) { Write-Host "[overnight] SKIP $Name (marker exists)"; return }
    Write-Host "[overnight] ===== START $Name  $(Get-Date -Format o) ====="
    try { & $Action } catch { Write-Host "[overnight] $Name FAILED: $_" }
    Write-Host "[overnight] ===== END   $Name  $(Get-Date -Format o) ====="
}

Stage "1/3 zero-shot 7 folds" "outputs\zeroshot_all7.DONE" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_zeroshot_all7.ps1
}

Stage "2/3 epoch probe multi-fold" "outputs\epoch_multifold.DONE" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_epoch_multifold.ps1
}

Stage "3/3 ridge probe 7 folds" "outputs\ridge_probe.json" {
    & $PY scripts\ridge_probe_folds.py --processed $P --pool mean `
        --out-json outputs\ridge_probe.json
}

Set-Content -Path outputs\overnight.DONE -Value (Get-Date -Format o)
Write-Host "[overnight] ALL DONE $(Get-Date -Format o)"
Stop-Transcript
