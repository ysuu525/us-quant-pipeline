# 实验 9：严格单 GPU 串行；12 个逻辑 r1 复用旧产物，只新跑 12 个 r2。
# 判据与用途限制见 exp9_test_retest_reliability.py。stdout/stderr 分离重定向。
$ErrorActionPreference = "Stop"
$REPO = "F:\quant\us-quant-pipeline"
$PY = "$REPO\.venv\Scripts\python.exe"
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$LOGDIR = "$REPO\outputs\exp9_queue_logs"
$LOCK = "$REPO\outputs\exp9_reliability.queue.lock"
$EXP3LOCK = "$REPO\outputs\exp3_samplecount.queue.lock"
$QUEUE_STARTED = Get-Date
$lockStream = $null

Set-Location $REPO
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

function Test-LiveLock {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $otherPid = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $otherPid) { return $false }
    try {
        Get-Process -Id ([int]$otherPid) -ErrorAction Stop | Out-Null
        return $true
    }
    catch { return $false }
}

if (Test-Path -LiteralPath $LOCK) {
    if (Test-LiveLock $LOCK) {
        Write-Host "另一个实验 9 队列正在运行。" -ForegroundColor Yellow
        exit 0
    }
    Remove-Item -LiteralPath $LOCK -Force
}
try {
    $lockStream = [System.IO.File]::Open($LOCK, [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    $pidBytes = [System.Text.Encoding]::UTF8.GetBytes([string]$PID)
    $lockStream.Write($pidBytes, 0, $pidBytes.Length)
    $lockStream.Flush()
}
catch {
    Write-Host "无法取得实验 9 GPU 互斥锁。" -ForegroundColor Yellow
    exit 0
}

function Wait-KnownGpuQueue {
    while (Test-LiveLock $EXP3LOCK) {
        Write-Host "实验 3 GPU 队列仍在运行；实验 9 等待 60 秒。" -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
    if (Test-Path -LiteralPath $EXP3LOCK) {
        throw "发现失效的实验 3 锁；为避免误并发，须人工核查后再启动。"
    }
}

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
    param([string]$Dir, [int]$SampleCount, [string]$ValStart, [string]$ValEnd)
    $score = Join-Path $Dir "scores.parquet"
    $metric = Join-Path $Dir "metrics.json"
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
    catch { return $false }
}

function Assert-Reuse {
    param([string]$Label, [string]$Dir, [int]$SampleCount, [string]$ValStart, [string]$ValEnd)
    if (-not (Test-Complete $Dir $SampleCount $ValStart $ValEnd)) {
        throw "复用格 ${Label} 不完整或口径不符：$Dir。不得重复跑同配置；先完成实验 3。"
    }
    Write-Host "复用 ${Label}：$Dir" -ForegroundColor Green
}

function Invoke-Exp9 {
    param(
        [string]$Label, [string]$ModelDir, [string]$ValStart, [string]$ValEnd,
        [string]$Tag, [int]$SampleCount, [int]$MaxAttempts = 4
    )
    $dir = Join-Path $ModelDir "eval_$Tag"
    if (Test-Complete $dir $SampleCount $ValStart $ValEnd) {
        Write-Host "${Label}：完整产物已存在，跳过。" -ForegroundColor Green
        return
    }
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Wait-KnownGpuQueue
        Wait-MemoryGate
        $stdout = Join-Path $LOGDIR "$Label.log"
        $stderr = Join-Path $LOGDIR "$Label.err.log"
        $arguments = @(
            "scripts\evaluate_fold.py", "--model-dir", $ModelDir,
            "--processed", $P, "--val-start", $ValStart, "--val-end", $ValEnd,
            "--lookback", "90", "--tag", $Tag, "--predict", "6",
            "--batch-size", "128", "--sample-count", [string]$SampleCount,
            "--amp", "bf16", "--device", "cuda"
        )
        $started = Get-Date
        Write-Host "${Label}：第 $attempt 次启动 $($started.ToString('o'))" -ForegroundColor Cyan
        $process = Start-Process -FilePath $PY -ArgumentList $arguments `
            -WorkingDirectory $REPO -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr -NoNewWindow -PassThru -Wait
        $elapsed = (Get-Date) - $started
        if ($process.ExitCode -eq 0 -and (Test-Complete $dir $SampleCount $ValStart $ValEnd)) {
            Write-Host ("${Label}：成功，耗时 {0}" -f $elapsed) -ForegroundColor Green
            return
        }
        Write-Host ("${Label}：失败 exit={0}，耗时 {1}" -f $process.ExitCode, $elapsed) -ForegroundColor Yellow
        if ($attempt -lt $MaxAttempts) { Start-Sleep -Seconds 60 }
    }
    throw "${Label} 连续失败 $MaxAttempts 次"
}

try {
    Wait-KnownGpuQueue
    Assert-Reuse "fold36_sc5_r1" "$REPO\outputs\fold36_lb90_s0_poolB_universe\eval_amp_lb90_fold36" 5 "2020-07-01" "2020-12-31"
    Assert-Reuse "fold39_sc5_r1" "$REPO\outputs\fold39_lb90_s0_poolB_universe\eval_amp_lb90_fold39" 5 "2022-01-03" "2022-06-30"
    Assert-Reuse "fold42_sc5_r1" "$REPO\outputs\fold42_lb90_s0_poolB_universe\eval_amp_lb90_fold42" 5 "2023-07-03" "2023-12-29"
    Assert-Reuse "fold36_sc10_r1" "$REPO\outputs\fold36_lb90_s0_poolB_universe\eval_e1_sc10" 10 "2020-07-01" "2020-12-31"
    Assert-Reuse "fold39_sc10_r1" "$REPO\outputs\fold39_lb90_s0_poolB_universe\eval_e1_sc10" 10 "2022-01-03" "2022-06-30"
    Assert-Reuse "fold42_sc10_r1" "$REPO\outputs\fold42_lb90_s0_poolB_universe\eval_e1_sc10" 10 "2023-07-03" "2023-12-29"
    Assert-Reuse "fold36_sc20_r1" "$REPO\outputs\fold36_lb90_s0_poolB_universe\eval_e1_sc20" 20 "2020-07-01" "2020-12-31"
    Assert-Reuse "fold39_sc20_r1" "$REPO\outputs\fold39_lb90_s0_poolB_universe\eval_e1_sc20" 20 "2022-01-03" "2022-06-30"
    Assert-Reuse "fold42_sc20_r1" "$REPO\outputs\fold42_lb90_s0_poolB_universe\eval_e1_sc20" 20 "2023-07-03" "2023-12-29"
    Assert-Reuse "fold36_sc40_r1" "$REPO\outputs\fold36_lb90_s0_poolB_universe\eval_e1_sc40" 40 "2020-07-01" "2020-12-31"
    Assert-Reuse "fold39_sc40_r1" "$REPO\outputs\fold39_lb90_s0_poolB_universe\eval_e1_sc40" 40 "2022-01-03" "2022-06-30"
    Assert-Reuse "fold42_sc40_r1" "$REPO\outputs\fold42_lb90_s0_poolB_universe\eval_e1_sc40" 40 "2023-07-03" "2023-12-29"

    Invoke-Exp9 "sc5_fold36_r2" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e9_sc5_r2" 5
    Invoke-Exp9 "sc5_fold39_r2" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e9_sc5_r2" 5
    Invoke-Exp9 "sc5_fold42_r2" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e9_sc5_r2" 5
    Invoke-Exp9 "sc10_fold36_r2" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e9_sc10_r2" 10
    Invoke-Exp9 "sc10_fold39_r2" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e9_sc10_r2" 10
    Invoke-Exp9 "sc10_fold42_r2" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e9_sc10_r2" 10
    Invoke-Exp9 "sc20_fold36_r2" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e9_sc20_r2" 20
    Invoke-Exp9 "sc20_fold39_r2" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e9_sc20_r2" 20
    Invoke-Exp9 "sc20_fold42_r2" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e9_sc20_r2" 20
    Invoke-Exp9 "sc40_fold36_r2" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e9_sc40_r2" 40
    Invoke-Exp9 "sc40_fold39_r2" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e9_sc40_r2" 40
    Invoke-Exp9 "sc40_fold42_r2" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e9_sc40_r2" 40
    $completed = Get-Date
    [ordered]@{
        started = $QUEUE_STARTED.ToString("o")
        completed = $completed.ToString("o")
        elapsed_seconds = ($completed - $QUEUE_STARTED).TotalSeconds
        logical_cells = 24
        reused_cells = 12
        new_gpu_calls = 12
        reuse_rule = "sc5 r1=eval_amp; sc10/20/40 r1=experiment3 e1; all r2=new e9 tags"
    } | ConvertTo-Json | Set-Content -LiteralPath "$LOGDIR\QUEUE.DONE" -Encoding utf8
    Write-Host "实验 9 GPU 队列完成。" -ForegroundColor Green
}
finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
    Remove-Item -LiteralPath $LOCK -Force -ErrorAction SilentlyContinue
}
