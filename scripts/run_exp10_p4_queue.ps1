# 实验 10 / P4：开发折 36 / 39 / 42 的 FT 微调**重跑**队列（GPU，隔夜串行）。
#
# 判据 / 用途限制：诊断，无阈值，不作判定。曲线读法的限定见
# scripts\exp10_p4_adaptation.py 的 docstring（只有开发折可微调、无对照臂）。
#
# ── 先读这一段再决定要不要跑 ────────────────────────────────────────────────
# **本队列多半不需要跑。** src\kronos_ft\train.py:220-224 每个 epoch 就已经把完整
# state_dict 存成 {stage}_epoch{NNN}.pt，逐轮 train/inner loss 也已在
# {stage}_summary.json 里；折 36 / 39 / 42 的这些产物**当前全部在磁盘上**
# （36: tok 8 + pred 6 轮；39/42: 各 30 + 30 轮）。因此 P4 的两条曲线可以由
#     .venv\Scripts\python.exe scripts\exp10_p4_adaptation.py
# 纯事后算出，**一个 GPU 小时都不用花**。本队列只在需要一次「独立重跑以验证
# 可复现性」时才有意义。
#
# ── 超参溯源（一字不改；三处独立来源互相印证）────────────────────────────
# 训练窗与命令行来自：
#   fold36  2017-07-03..2020-06-22  scripts\run_supp_folds.ps1 + ledger.md:98
#   fold39  2019-01-02..2021-12-22  scripts\run_supp_folds.ps1 + ledger.md:108
#   fold42  2020-07-01..2023-06-22  scripts\run_probe_recent.ps1 + ledger.md:71
# 其余全部走 kronos_ft.train 的默认值（batch_size=50, max_epochs=30, patience=5,
# n_train_batches=2000, n_val_batches=400, swa_k=3, inner_months=6, seed=0,
# lookback=90, stage=both, 池 B = universe anchor），与原运行的
# outputs\fold*\run_summary.json 的 config 逐字段一致。
#
# **已知分歧（不自行裁决，只披露）**：scripts\emit_folds.py 今天机械生成的边界与
# 上面三行不完全相同（fold36 te=2020-06-24、fold42 ts=2020-07-06 / ve=2024-01-02）。
# 原训练是用旧脚本里写死的边界（那批 .ps1 注明 oos_start=2024-01-01）跑的。
# P4 的要求是「与原 FT 训练一字不改」，故本队列用**原运行的边界**（上面三行）。
#
# ── 副作用（必须知情）───────────────────────────────────────────────────────
# kronos_ft.train.run() 在 ledger=True（CLI 默认）时会往 experiments\ledger.md
# **追加**一行 `train | ...`（预注册 §3 的既有行为，不是本实验的改动）。
# 若不希望重跑污染登记簿流水，请先与用户确认。
#
# 产物写到 outputs\exp10_p4_fold{f}_lb90_s0_poolB_universe（**不写进原折目录**，
# 否则 train.py 的 _completed() 会直接跳过重训）。
$ErrorActionPreference = "Stop"
$REPO = "F:\quant\us-quant-pipeline"
$PY = "$REPO\.venv\Scripts\python.exe"
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$LOGDIR = "$REPO\outputs\exp10_p4_queue_logs"
$LOCK = "$REPO\outputs\exp10_p4.queue.lock"
$EXP9LOCK = "$REPO\outputs\exp9_reliability.queue.lock"
$EXP3LOCK = "$REPO\outputs\exp3_samplecount.queue.lock"
$QUEUE_STARTED = Get-Date
$lockStream = $null

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

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
        Write-Host "另一个实验 10 P4 队列正在运行。" -ForegroundColor Yellow
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
    Write-Host "无法取得实验 10 P4 GPU 互斥锁。" -ForegroundColor Yellow
    exit 0
}

function Wait-KnownGpuQueue {
    while ((Test-LiveLock $EXP3LOCK) -or (Test-LiveLock $EXP9LOCK)) {
        Write-Host "实验 3 / 9 的 GPU 队列仍在运行；实验 10 等待 60 秒。" -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
    foreach ($stale in @($EXP3LOCK, $EXP9LOCK)) {
        if (Test-Path -LiteralPath $stale) {
            throw "发现失效的锁 $stale；为避免误并发，须人工核查后再启动。"
        }
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
    param([string]$Dir)
    return ((Test-Path -LiteralPath (Join-Path $Dir "tokenizer_summary.json")) -and
            (Test-Path -LiteralPath (Join-Path $Dir "predictor_summary.json")) -and
            (Test-Path -LiteralPath (Join-Path $Dir "run_summary.json")))
}

function Invoke-P4Train {
    param(
        [string]$Label, [string]$TrainStart, [string]$TrainEnd, [string]$Out,
        [int]$MaxAttempts = 3
    )
    if (Test-Complete $Out) {
        Write-Host "${Label}：完整产物已存在，跳过。" -ForegroundColor Green
        return
    }
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Wait-KnownGpuQueue
        Wait-MemoryGate
        # 每次尝试单独命名，重试不覆盖上一次的 err.log
        $stdout = Join-Path $LOGDIR "$Label.a$attempt.log"
        $stderr = Join-Path $LOGDIR "$Label.a$attempt.err.log"
        $arguments = @(
            "-m", "kronos_ft.train",
            "--panel", "$P\panel_kronos_adj.parquet",
            "--index-parquet", "$P\market_index.parquet",
            "--train-start", $TrainStart, "--train-end", $TrainEnd,
            "--lookback", "90", "--seed", "0", "--stage", "both",
            "--out", $Out,
            "--universe-parquet", "$P\universe.parquet",
            "--index-cache", "$P\index_cache\lb90_full.parquet"
        )
        $started = Get-Date
        Write-Host "${Label}：第 $attempt 次启动 $($started.ToString('o'))" -ForegroundColor Cyan
        $process = Start-Process -FilePath $PY -ArgumentList $arguments `
            -WorkingDirectory $REPO -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr -NoNewWindow -PassThru -Wait
        $elapsed = (Get-Date) - $started
        if ($process.ExitCode -eq 0 -and (Test-Complete $Out)) {
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
    Invoke-P4Train "p4_fold36" "2017-07-03" "2020-06-22" "outputs\exp10_p4_fold36_lb90_s0_poolB_universe"
    Invoke-P4Train "p4_fold39" "2019-01-02" "2021-12-22" "outputs\exp10_p4_fold39_lb90_s0_poolB_universe"
    Invoke-P4Train "p4_fold42" "2020-07-01" "2023-06-22" "outputs\exp10_p4_fold42_lb90_s0_poolB_universe"

    Wait-MemoryGate
    & $PY "scripts\exp10_p4_adaptation.py" `
        --extra-dir "outputs\exp10_p4_fold36_lb90_s0_poolB_universe" `
        --extra-dir "outputs\exp10_p4_fold39_lb90_s0_poolB_universe" `
        --extra-dir "outputs\exp10_p4_fold42_lb90_s0_poolB_universe" `
        --out "outputs\exp10_p4_adaptation_rerun.json" `
        --svg "outputs\exp10_p4_adaptation_rerun.svg"
    if ($LASTEXITCODE -ne 0) { throw "exp10_p4_adaptation.py 失败 exit=$LASTEXITCODE" }

    $completed = Get-Date
    [ordered]@{
        started = $QUEUE_STARTED.ToString("o")
        completed = $completed.ToString("o")
        elapsed_seconds = ($completed - $QUEUE_STARTED).TotalSeconds
        folds = @("fold36", "fold39", "fold42")
        note = "重跑队列；原折目录的逐轮 checkpoint 已足够算 P4，本队列仅供独立复现"
    } | ConvertTo-Json | Set-Content -LiteralPath "$LOGDIR\QUEUE.DONE" -Encoding utf8
    Write-Host "实验 10 P4 队列完成。" -ForegroundColor Green
}
finally {
    if ($null -ne $lockStream) { $lockStream.Dispose() }
    Remove-Item -LiteralPath $LOCK -Force -ErrorAction SilentlyContinue
}
