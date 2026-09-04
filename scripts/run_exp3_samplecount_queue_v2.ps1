# 实验 3 GPU 队列 v2 —— 方案 A（显存封顶）的交接版。
#
# 背景：v1（run_exp3_samplecount_queue.ps1）每折实测 3.2–4.0 小时，预期 68 分钟。
# 根因已实测：打分进程独占显存 14.82 GiB 并溢出 0.83 GiB 到共享内存，
# Windows WDDM 上 cudaMalloc 不会失败而是分到系统内存，PyTorch 缓存分配器
# 永远收不到「回收缓存」的信号 → 整卡撑满 → 换页 → 掉速 3–4 倍。
# 处置（用户裁定，方案 A）：给打分进程加 --gpu-mem-fraction 0.70，让分配器
# 触顶时释放缓存而不是溢出。数值、RNG 调用序列、scoring_config 全部不变。
#
# 使用方式（重要）：本脚本由主会话在 v1 的 PowerShell 已被结束、
# **但其子进程 python（fold40）仍在跑**时启动。因此：
#   - 启动后先 Wait-GpuIdle：只要还有任何 evaluate_fold.py 进程活着就等，
#     绝不与之抢显存（并发 CUDA 进程可共存，但强杀会引发驱动层崩溃）；
#   - 锁文件与 v1 同路径；锁里的 PID 若已死则删锁接管。
#   - Test-Complete 口径与 v1 完全一致，故折 36–40 会被自动跳过。
#
# 输出仍用 Start-Process 的 stdout/stderr 分离重定向，不使用 PowerShell *>。
param(
    [int]$LimitObs = 0,          # 可选：传给 --limit-obs（冒烟用；两个冒烟项另有写死的 6000）
    [double]$MemFraction = 0.70, # --gpu-mem-fraction 的值（工程参数，不进 scoring_config）
    [double]$Sc40GateGib = 8.5   # sc40 冒烟 allocated 超过它就跳过 sc40 三项
)
# 2026-09-04 第二轮（用户裁定选项 1）：sc40/b128 活跃峰值约 9.7 GiB，0.70 封顶下 OOM；
# 改以 -MemFraction 0.85 -Sc40GateGib 13.0 重跑本脚本：已完成项全部由 Test-Complete 跳过，
# 只重做 sc40 冒烟与 sc40 三折。门槛 13.0 ≈「冒烟没 OOM 就放行」（上限 0.85×16=13.6 GiB）。

$ErrorActionPreference = "Stop"
$REPO = "F:\quant\us-quant-pipeline"
$PY = "$REPO\.venv\Scripts\python.exe"
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$LOGDIR = "$REPO\outputs\exp3_queue_logs"
$LOCK = "$REPO\outputs\exp3_samplecount.queue.lock"
$QUEUE_STARTED = Get-Date

# 方案 A 的配套：分配器内部策略，只在设了 memory fraction 时才起作用
# （触顶前先跑一轮缓存回收）。不影响数值，子进程继承本变量。
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8"
if ($MemFraction -le 0 -or $MemFraction -gt 1) { throw "-MemFraction 必须落在 (0,1]，收到 $MemFraction" }
$MEM_FRACTION = $MemFraction.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture)

$SMOKE_MODEL = "$REPO\outputs\fold36_lb90_s0_poolB_universe"
$SMOKE_START = "2020-07-01"
$SMOKE_END = "2020-12-31"
$SMOKE_LIMIT = 6000
$SC40_ALLOC_GATE_GIB = $Sc40GateGib
Write-Host ("队列 v2 参数：gpu-mem-fraction={0}  sc40 门槛={1} GiB  PYTORCH_CUDA_ALLOC_CONF={2}" -f $MEM_FRACTION, $SC40_ALLOC_GATE_GIB, $env:PYTORCH_CUDA_ALLOC_CONF) -ForegroundColor DarkCyan

Set-Location $REPO
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null

if ($LimitObs -gt 0) {
    Write-Host ("警告：-LimitObs {0} 会让全部正式折也走冒烟口径；" -f $LimitObs) -ForegroundColor Red
    Write-Host "      产出的 metrics.json 仍会通过 Test-Complete，后续正式跑会被误跳过。" -ForegroundColor Red
    Write-Host "      仅供代码路径冒烟，读数不作判定用。" -ForegroundColor Red
}

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
    Write-Host "锁文件里的 PID ${oldPid} 已不存在，删锁接管。" -ForegroundColor Yellow
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

function Wait-GpuIdle {
    # v1 的 PowerShell 被结束后，其子进程 python 仍在打分。等它自然退出，
    # 绝不强杀（并发 CUDA 进程强杀会引发驱动层崩溃，已发生过一次）。
    while ($true) {
        $busy = @()
        try {
            # 必须同时是 python.exe：实测只按 CommandLine 匹配会命中任何
            # 「命令行里提到 evaluate_fold.py」的 powershell（例如主会话的一次
            # 盘点命令），把队列白等；pytest / tests 路径同理排除。
            $busy = @(Get-CimInstance Win32_Process -ErrorAction Stop |
                Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -and
                               $_.CommandLine -like "*evaluate_fold.py*" -and
                               $_.CommandLine -notlike "*pytest*" -and
                               $_.CommandLine -notlike "*\tests\*" })
        }
        catch {
            Write-Host "  Win32_Process 查询失败：未核，按「GPU 空闲」放行" -ForegroundColor DarkGray
            return
        }
        # 排除自己（本脚本不是 python 进程，这里只会命中打分子进程）
        $busy = @($busy | Where-Object { $_.ProcessId -ne $PID })
        if ($busy.Count -eq 0) {
            Write-Host ("  [{0}] 未见 evaluate_fold.py 进程，GPU 视为空闲" -f (Get-Date).ToString("HH:mm:ss"))
            return
        }
        foreach ($b in $busy) {
            Write-Host ("  [{0}] 等待打分进程 PID {1} 退出（60 秒后重查）" -f `
                (Get-Date).ToString("HH:mm:ss"), $b.ProcessId) -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 60
    }
}

function Show-GpuMemory {
    # 只读诊断：读失败就打「未核」，绝不阻塞队列。
    try {
        $adapter = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
        $totalGb = ($adapter | Measure-Object -Property CookedValue -Sum).Sum / 1GB
        $top = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
            Where-Object { $_.CookedValue -gt 0 } |
            Sort-Object -Property CookedValue -Descending | Select-Object -First 3
        $parts = @()
        foreach ($s in $top) {
            $parts += ("{0}={1:N2}GB" -f $s.InstanceName, ($s.CookedValue / 1GB))
        }
        if ($parts.Count -eq 0) { $parts = @("（无非零进程实例）") }
        Write-Host ("  GPU 专用显存合计 {0:N2} GB；占用前三 {1}" -f $totalGb, ($parts -join "  ")) -ForegroundColor DarkCyan
    }
    catch {
        Write-Host "  GPU 显存计数器：未核（性能计数器读取失败）" -ForegroundColor DarkGray
    }
}

function Test-Complete {
    # 与 v1 逐字相同：口径核对靠 scoring_config（CLAUDE.md §八）。
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

function Get-RuntimeBlock {
    # 读 metrics.json 的新顶层键 runtime（工程口径，不属于 scoring_config）。
    param([string]$ModelDir, [string]$Tag)
    $metric = Join-Path (Join-Path $ModelDir "eval_$Tag") "metrics.json"
    if (-not (Test-Path -LiteralPath $metric)) { return $null }
    try {
        $m = Get-Content -Raw -LiteralPath $metric | ConvertFrom-Json
        return $m.runtime
    }
    catch {
        return $null
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
        [int]$Limit = 0,
        [int]$MaxAttempts = 4
    )
    if (Test-Complete $ModelDir $Tag $SampleCount $ValStart $ValEnd) {
        Write-Host "${Label}：完整产物已存在，跳过。" -ForegroundColor Green
        return
    }
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Wait-GpuIdle
        Wait-MemoryGate
        Show-GpuMemory
        $stdout = Join-Path $LOGDIR "${Label}.a${attempt}.log"
        $stderr = Join-Path $LOGDIR "${Label}.a${attempt}.err.log"
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
            "--device", "cuda",
            "--gpu-mem-fraction", $MEM_FRACTION
        )
        if ($Limit -gt 0) {
            $arguments += @("--limit-obs", [string]$Limit)
        }
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
        Write-Host ("${Label}：失败 exit={0}，耗时 {1}（日志 {2}）" -f $process.ExitCode, $elapsed, $stdout) -ForegroundColor Yellow
        if ($attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds 60
        }
    }
    throw "${Label} 连续失败 $MaxAttempts 次"
}

try {
    # ---- a/b：两个冒烟（fold36 模型 + fold36 窗口，--limit-obs 6000）----
    # 冒烟失败不拖垮整队：sc40 前置门槛读不到数字时会自行跳过 sc40。
    $smokeError = @()
    foreach ($sc in @(20, 40)) {
        $tag = "memfrac_smoke_sc$sc"
        try {
            Invoke-Exp3 "smoke_sc$sc" $SMOKE_MODEL $SMOKE_START $SMOKE_END $tag $sc $SMOKE_LIMIT 2
        }
        catch {
            $smokeError += ("sc{0}: {1}" -f $sc, $_.Exception.Message)
            Write-Host ("冒烟 sc{0} 失败：{1}" -f $sc, $_.Exception.Message) -ForegroundColor Red
        }
    }

    # ---- c：把两次冒烟的 runtime 打到控制台并落盘 ----
    $smokeReport = [ordered]@{
        generated = (Get-Date).ToString("o")
        gpu_mem_fraction = [double]$MEM_FRACTION
        pytorch_cuda_alloc_conf = $env:PYTORCH_CUDA_ALLOC_CONF
        limit_obs = $SMOKE_LIMIT
        model_dir = $SMOKE_MODEL
        val_window = @($SMOKE_START, $SMOKE_END)
        errors = $smokeError
        runs = [ordered]@{}
    }
    $sc40Alloc = $null
    foreach ($sc in @(20, 40)) {
        $tag = "memfrac_smoke_sc$sc"
        $rt = Get-RuntimeBlock $SMOKE_MODEL $tag
        if ($null -eq $rt) {
            Write-Host ("冒烟 sc{0}：metrics.json 的 runtime 未核（缺文件或缺键）" -f $sc) -ForegroundColor Red
            $smokeReport.runs["sc$sc"] = "未核"
            continue
        }
        Write-Host ("冒烟 sc{0}：allocated={1} GiB  reserved={2} GiB  scoring={3} s  device={4}" -f `
            $sc, $rt.max_memory_allocated_gib, $rt.max_memory_reserved_gib, `
            $rt.scoring_seconds, $rt.device) -ForegroundColor Cyan
        $smokeReport.runs["sc$sc"] = [ordered]@{
            max_memory_allocated_gib = $rt.max_memory_allocated_gib
            max_memory_reserved_gib = $rt.max_memory_reserved_gib
            scoring_seconds = $rt.scoring_seconds
            device = $rt.device
            gpu_mem_fraction = $rt.gpu_mem_fraction
        }
        if ($sc -eq 40) { $sc40Alloc = $rt.max_memory_allocated_gib }
    }
    Set-Content -LiteralPath "$LOGDIR\MEMFRAC_SMOKE.json" `
        -Value ($smokeReport | ConvertTo-Json -Depth 6) -Encoding utf8

    # ---- f 的前置门槛：sc40 是否放行（在此判定，跑在队尾）----
    $runSc40 = $true
    $sc40Reason = ""
    if ($null -eq $sc40Alloc) {
        $runSc40 = $false
        $sc40Reason = "sc40 冒烟的 runtime.max_memory_allocated_gib 未核（缺 metrics.json 或缺键）"
    }
    elseif ([double]$sc40Alloc -gt $SC40_ALLOC_GATE_GIB) {
        $runSc40 = $false
        $sc40Reason = ("sc40 冒烟 allocated={0} GiB > 门槛 {1} GiB" -f $sc40Alloc, $SC40_ALLOC_GATE_GIB)
    }
    if (-not $runSc40) {
        Write-Host "===============================================================" -ForegroundColor Red
        Write-Host ("跳过全部 sc40 项：{0}" -f $sc40Reason) -ForegroundColor Red
        Write-Host "===============================================================" -ForegroundColor Red
        $skip = [ordered]@{
            generated = (Get-Date).ToString("o")
            reason = $sc40Reason
            gate_gib = $SC40_ALLOC_GATE_GIB
            observed_max_memory_allocated_gib = $sc40Alloc
            smoke_report = "$LOGDIR\MEMFRAC_SMOKE.json"
        } | ConvertTo-Json -Depth 4
        Set-Content -LiteralPath "$LOGDIR\SC40_SKIPPED.json" -Value $skip -Encoding utf8
    }

    # ---- d：sc20 折 36–42（Test-Complete 口径不变，已完成的自动跳过）----
    Invoke-Exp3 "sc20_fold36" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e1_sc20" 20 $LimitObs
    Invoke-Exp3 "sc20_fold37" "$REPO\outputs\fold37_lb90_s0_poolB_universe" "2021-01-04" "2021-06-30" "e1_sc20" 20 $LimitObs
    Invoke-Exp3 "sc20_fold38" "$REPO\outputs\fold38_lb90_s0_poolB_universe" "2021-07-01" "2021-12-31" "e1_sc20" 20 $LimitObs
    Invoke-Exp3 "sc20_fold39" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e1_sc20" 20 $LimitObs
    Invoke-Exp3 "sc20_fold40" "$REPO\outputs\fold40_lb90_s0_poolB_universe" "2022-07-01" "2022-12-30" "e1_sc20" 20 $LimitObs
    Invoke-Exp3 "sc20_fold41" "$REPO\outputs\fold41_lb90_s0_poolB_universe" "2023-01-03" "2023-06-30" "e1_sc20" 20 $LimitObs
    Invoke-Exp3 "sc20_fold42" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e1_sc20" 20 $LimitObs

    # ---- e：sc10 折 36 / 39 / 42 ----
    Invoke-Exp3 "sc10_fold36" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e1_sc10" 10 $LimitObs
    Invoke-Exp3 "sc10_fold39" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e1_sc10" 10 $LimitObs
    Invoke-Exp3 "sc10_fold42" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e1_sc10" 10 $LimitObs

    # ---- f：sc40 折 36 / 39 / 42（受上面的前置门槛控制）----
    if ($runSc40) {
        Invoke-Exp3 "sc40_fold36" "$REPO\outputs\fold36_lb90_s0_poolB_universe" "2020-07-01" "2020-12-31" "e1_sc40" 40 $LimitObs
        Invoke-Exp3 "sc40_fold39" "$REPO\outputs\fold39_lb90_s0_poolB_universe" "2022-01-03" "2022-06-30" "e1_sc40" 40 $LimitObs
        Invoke-Exp3 "sc40_fold42" "$REPO\outputs\fold42_lb90_s0_poolB_universe" "2023-07-03" "2023-12-29" "e1_sc40" 40 $LimitObs
    }
    else {
        Write-Host ("sc40 三项按前置门槛跳过，详见 {0}\SC40_SKIPPED.json" -f $LOGDIR) -ForegroundColor Red
    }

    # ---- g：收尾 ----
    $completed = Get-Date
    $done = [ordered]@{
        started = $QUEUE_STARTED.ToString("o")
        completed = $completed.ToString("o")
        elapsed_seconds = ($completed - $QUEUE_STARTED).TotalSeconds
        queue_version = "v2 (gpu-mem-fraction $MEM_FRACTION)"
        sc40_executed = $runSc40
        smoke_errors = $smokeError
    } | ConvertTo-Json -Depth 4
    Set-Content -LiteralPath "$LOGDIR\QUEUE.DONE" -Value $done -Encoding utf8
    Write-Host "实验 3 GPU 队列 v2 完成。" -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $LOCK -Force -ErrorAction SilentlyContinue
}
