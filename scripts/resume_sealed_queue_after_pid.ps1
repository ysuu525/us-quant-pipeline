param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

# One-shot handoff: let the already-running CUDA training finish naturally,
# then resume the idempotent sealed queue. This script never reads sealed data.
$ErrorActionPreference = "Stop"
$REPO = "F:\quant\us-quant-pipeline"
$QUEUE = "$REPO\scripts\run_sealed_confirm_queue.ps1"

$training = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
if ($training) {
    Wait-Process -Id $TrainingPid
}

Start-Sleep -Seconds 5
Set-Location $REPO
& $QUEUE
