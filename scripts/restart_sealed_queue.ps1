# Safe restart of the sealed confirmation queue after removing fold43.
# ASCII-only on purpose: no BOM needed, avoids PS 5.1 encoding traps.
#
# The PowerShell driver was terminated, but its Start-Process child (the
# fold05 training) is an orphan that keeps running and will write
# predictor_final normally.  NEVER hard-kill a running CUDA process
# (CLAUDE.md section 8: killing one crashes the driver).
#
# This script waits for every kronos_ft.train OR evaluate_fold python to exit, lets the GPU
# settle, then relaunches the queue with the new 33-fold list.  The queue is
# idempotent, so fold05 training is skipped if it completed.
$REPO = "F:\quant\us-quant-pipeline"
$LOG  = "$REPO\outputs\sealed_confirm\_logs\restart.log"

function Get-TrainProcs {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -and ($_.CommandLine.Contains("kronos_ft.train") -or $_.CommandLine.Contains("evaluate_fold")) })
}

Write-Host "Waiting for orphaned training process(es) to finish..."
$waited = 0
while ($true) {
    $procs = Get-TrainProcs
    if ($procs.Count -eq 0) { break }
    if ($waited % 600 -eq 0) {
        Write-Host ("  still running: {0} proc(s), waited {1}s" -f $procs.Count, $waited)
        Add-Content -Path $LOG -Value ("{0} waiting, {1} procs" -f (Get-Date -Format o), $procs.Count) -Encoding utf8
    }
    Start-Sleep -Seconds 30
    $waited += 30
}
Write-Host "All training processes exited. Letting the GPU settle for 30s..."
Start-Sleep -Seconds 30
Add-Content -Path $LOG -Value ("{0} relaunching queue (33 folds, fold43 removed)" -f (Get-Date -Format o)) -Encoding utf8
Write-Host "Relaunching queue (33 folds, fold43 removed)"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$REPO\scripts\run_sealed_confirm_queue.ps1"
