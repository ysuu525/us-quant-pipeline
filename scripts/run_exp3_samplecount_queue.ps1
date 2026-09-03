# 实验 3：严格单 GPU 串行；判据与用途限制见 exp3_samplecount_curve.py。
# 输出使用 Start-Process 的 stdout/stderr 分离重定向，不使用 PowerShell *>。
$ErrorActionPreference = "Stop"
$REPO = "F:\quant\us-quant-pipeline"
$PY = "$REPO\.venv\Scripts\python.exe"
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$LOGDIR = "$REPO\outputs\exp3_queue_logs"
$LOCK = "$REPO\outputs\exp3_samplecount.queue.lock"
$QUEUE_STARTED = Get-Date

Set-Location $REPO
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

if (Test-Path -LiteralPath $LOCK) {
    $oldPid = Get-Content -LiteralPath $LOCK -ErrorAction SilentlyContinue
    $alive = $false
    if ($oldPid) {
        try {
            Get-Process -Id ([int]$oldPid) -ErrorAction Stop | Out-Null
            $alive = $true
        }
        catch {}
    }
    if ($alive) {
        Write-Host "另一个实验 3 队列进程（PID ${oldPid}）正在运行。" -ForegroundColor Yellow
        exit 0
    }
    Remove-Item -LiteralPath $LOCK -Force
}
Set-Content -LiteralPath $LOCK -Value $PID -Encoding utf8

function Wait-MemoryGate {
    while ($true) {
        $committed = (Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue / 1GB
        if ($committed -le 40.0) {
            Write-Host ("  committed={0:N2} GB，允许启动" -f $committed)
            return
        }
        Write-Host ("  committed={0:N2} GB > 40 GB，等待 60 秒" -f $committed) -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
}

function Test-Complete {
    param(
        [string]$ModelDir,
        [string]$Tag,
        [int]$SampleCount,
        [string]$ValStart,
        [string]$ValEnd
    )
    $dir = Join-Path $ModelDir "eval_$Tag"
    $score = Join-Path $dir "scores.parquet"
    $metric = Join-Path $dir "metrics.json"
    if (-not (Test-Path -LiteralPath $score) -or -not (Test-Path -LiteralPath $metric)) {
        return $false
    }
    try {
        $m = Get-Content -Raw -LiteralPath $metric | ConvertFrom-Json
        return (
            [int]$m.scoring_config.sample_count -eq $SampleCount -and
            [int]$m.scoring_config.batch_size -eq 128 -and
            [int]$m.scoring_config.lookback -eq 90 -and
            [int]$m.scoring_config.predict -eq 6 -and
            [string]$m.scoring_config.amp -eq "bf16" -and
            [string]$m.val_window[0] -eq $ValStart -and
            [string]$m.val_window[1] -eq $ValEnd
        )
    }
    catch {
        return $false
    }
}

function Invoke-Exp3 {
    param(
        [string]$Label,
        [string]$ModelDir,
        [string]$ValStart,
        [string]$ValEnd,
        [string]$Tag,
        [int]$SampleCount,
        [int]$MaxAttempts = 4
    )
    if (Test-Complete $ModelDir $Tag $SampleCount $ValStart $ValEnd) {
        Write-Host "${Label}：完整产物已存在，跳过。" -ForegroundColor Green
        return
    }
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Wait-MemoryGate
        $stdout = Join-Path $LOGDIR "$Label.log"
        $stderr = Join-Path $LOGDIR "$Label.err.log"
        $arguments = @(
            "scripts\evaluate_fold.py",
            "--model-dir", $ModelDir,
            "--processed", $P,
            "--val-start", $ValStart,
            "--val-end", $ValEnd,
            "--lookback", "90",
            "--tag", $Tag,
            "--predict", "6",
            "--batch-size", "128",
            "--sample-count", [string]$SampleCount,
            "--amp", "bf16",
            "--device", "cuda"
        )
        $started = Get-Date
        Write-Host "${Label}：第 $attempt 次启动 $($started.ToString('o'))" -ForegroundColor Cyan
        $process = Start-Process -FilePath $PY -ArgumentList $arguments `
            -WorkingDirectory $REPO -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr -NoNewWindow -PassThru -Wait
        $elapsed = (Get-Date) - $started
        if ($process.ExitCode -eq 0 -and (Test-Complete $ModelDir $Tag $SampleCount $ValStart $ValEnd)) {
            Write-Host ("${Label}：成功，耗时 {0}" -f $elapsed) -ForegroundColor Green
            return
        }
        Write-Host ("${Label}：失败 exit={0}，耗时 {1}" -f $process.ExitCode, $elapsed) -ForegroundColor Yellow
        if ($attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds 60
        }
    }
    throw "${Label} 连续失败 $MaxAttempts 次"
}

try {
    Invoke-Exp3 "sc20_fold36" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e1_sc20" 20
    Invoke-Exp3 "sc20_fold37" "$REPO\outputs\fold37_lb90_s0_poolB_universe" "2021-01-04" "2021-06-30" "e1_sc20" 20
    Invoke-Exp3 "sc20_fold38" "$REPO\outputs\fold38_lb90_s0_poolB_universe" "2021-07-01" "2021-12-31" "e1_sc20" 20
    Invoke-Exp3 "sc20_fold39" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e1_sc20" 20
    Invoke-Exp3 "sc20_fold40" "$REPO\outputs\fold40_lb90_s0_poolB_universe" "2022-07-01" "2022-12-30" "e1_sc20" 20
    Invoke-Exp3 "sc20_fold41" "$REPO\outputs\fold41_lb90_s0_poolB_universe" "2023-01-03" "2023-06-30" "e1_sc20" 20
    Invoke-Exp3 "sc20_fold42" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e1_sc20" 20
    Invoke-Exp3 "sc10_fold36" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e1_sc10" 10
    Invoke-Exp3 "sc10_fold39" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e1_sc10" 10
    Invoke-Exp3 "sc10_fold42" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e1_sc10" 10
    Invoke-Exp3 "sc40_fold36" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e1_sc40" 40
    Invoke-Exp3 "sc40_fold39" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e1_sc40" 40
    Invoke-Exp3 "sc40_fold42" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e1_sc40" 40
    $completed = Get-Date
    $done = [ordered]@{
        started = $QUEUE_STARTED.ToString("o")
        completed = $completed.ToString("o")
        elapsed_seconds = ($completed - $QUEUE_STARTED).TotalSeconds
    } | ConvertTo-Json
    Set-Content -LiteralPath "$LOGDIR\QUEUE.DONE" -Value $done -Encoding utf8
    Write-Host "实验 3 GPU 队列完成。" -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $LOCK -Force -ErrorAction SilentlyContinue
}
