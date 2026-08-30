# lookback 粗筛第一轮：守着 run_screen_lookback.ps1，训练+评估全绿后自动跑
# 三档配对判据（预注册 §2 第一轮，99% 宽边界）。命令逐字取自 HANDOFF.md §2。
# 用法: powershell -ExecutionPolicy Bypass -File scripts\after_screen_compare.ps1 -WatchPid <PID>
param([int]$WatchPid = 0)

$ErrorActionPreference = "Stop"
Set-Location "F:\quant\us-quant-pipeline"
$PY  = ".\.venv\Scripts\python.exe"
$LOG = "outputs\after_screen_compare.log"

function Log([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $LOG -Value $line -Encoding utf8
}

# ---- 12 个评估目录（3 档 x 4 折，池 B）----
# lb90 那一档复用池子消融已训好的模型，故 tag 命名与 lb60/lb200 不同。
$evals = [ordered]@{
    "lb90=outputs\fold01_lb90_s0_poolB_universe\eval_poolB_universe"       = $true
    "lb90=outputs\fold02_lb90_s0_poolB_universe\eval_poolB_universe_fold02" = $true
    "lb90=outputs\fold03_lb90_s0_poolB_universe\eval_poolB_universe_fold03" = $true
    "lb90=outputs\fold04_lb90_s0_poolB_universe\eval_poolB_universe_fold04" = $true
}
foreach ($lb in @(60, 200)) {
    foreach ($n in @("fold01", "fold02", "fold03", "fold04")) {
        $evals["lb${lb}=outputs\${n}_lb${lb}_s0_poolB_universe\eval_screen_lb${lb}_${n}"] = $true
    }
}
$dirs = @($evals.Keys | ForEach-Object { $_.Split("=", 2)[1] })

Log "开始守候：等 12 个评估目录齐全（WatchPid=$WatchPid）"

# ---- 等训练脚本跑完 ----
while ($true) {
    $missing = @($dirs | Where-Object {
        -not ((Test-Path (Join-Path $_ "daily_ic.parquet")) -and (Test-Path (Join-Path $_ "report.md")))
    })
    if ($missing.Count -eq 0) { break }

    # 还活着吗？先看被守的 PID，再退而看有没有任何 kronos_ft.train 在跑
    # （容许用户中途 Ctrl-C 后原样重跑续训，换了新 PID 也不误判）
    $alive = $false
    if ($WatchPid -gt 0 -and $null -ne (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue)) { $alive = $true }
    if (-not $alive) {
        $procs = @(Get-CimInstance Win32_Process -Filter "Name like '%python%' or Name like '%powershell%'" |
                   Where-Object { $_.CommandLine -like "*kronos_ft.train*" -or $_.CommandLine -like "*run_screen_lookback*" -or $_.CommandLine -like "*evaluate_fold.py*" })
        if ($procs.Count -gt 0) { $alive = $true }
    }
    if (-not $alive) {
        Log "训练进程已退出，但仍缺 $($missing.Count)/12 个评估结果 —— 中止，不跑 compare_arms。"
        $missing | ForEach-Object { Log "    缺: $_" }
        Log "处理：原样重跑 scripts\run_screen_lookback.ps1（幂等续训），再重挂本脚本。"
        exit 1
    }
    Start-Sleep -Seconds 120
}

Log "12/12 评估齐全，开始跑 compare_arms.py（三档配对，基准 lb90，99%/95% 双边界都会打印）"

# ---- 三档配对判据 ----
$cmpArgs = @("scripts\compare_arms.py")
foreach ($k in $evals.Keys) { $cmpArgs += @("--arm", $k) }
$cmpArgs += @("--baseline", "lb90", "--out", "outputs\screen_lookback_round1.md", "--ledger")

& $PY @cmpArgs *>&1 | Tee-Object -FilePath "outputs\compare_arms_stdout.log"
if ($LASTEXITCODE -ne 0) {
    Log "compare_arms.py 失败，退出码 $LASTEXITCODE（见 outputs\compare_arms_stdout.log）"
    exit $LASTEXITCODE
}

Log "完成 → outputs\screen_lookback_round1.md（结论已追加进 experiments\ledger.md）"
Add-Content -Path $LOG -Value (Get-Content "outputs\screen_lookback_round1.md" -Raw) -Encoding utf8
