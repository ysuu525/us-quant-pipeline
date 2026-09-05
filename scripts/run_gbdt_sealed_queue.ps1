# 树基线『计算专用』队列（2026-09-05 授权：计算 != 读取）
# XGBoost 主口径，折 05-35，CPU only，串行 + 互斥锁 + 失败重试 + 断点续跑。
# 控制台只显示：折号 / 阶段 / 成功失败 / 耗时。绝不打印任何指标。
# 用 Start-Process 重定向：PS5.1 的 *> 会把原生 exe 的 stderr 包成 NativeCommandError。
$ErrorActionPreference = "Continue"
$REPO   = "F:\quant\us-quant-pipeline"
$PY     = "$REPO\.venv-gbdt\Scripts\python.exe"
$PYMAIN = "$REPO\.venv\Scripts\python.exe"      # emit_folds 用主 venv（需要 crsp_pipeline 全栈）
$P      = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$CONFIG = "$REPO\configs\gbdt_strong_v2_sealed.json"
$OUT    = "$REPO\outputs\gbdt_strong_jkp_v2"
$CACHE  = "$OUT\cache_sealed"
$FOLDS  = "$OUT\folds_05_35.json"
$SEALED = "$OUT\xgboost\sealed"
$LOGDIR = "$SEALED\_logs"
$LOCK   = "$SEALED\.queue.lock"
$FIRST  = 5
$LAST   = 35

Set-Location $REPO
New-Item -ItemType Directory -Force $SEALED | Out-Null
New-Item -ItemType Directory -Force $LOGDIR | Out-Null

if (Test-Path $LOCK) {
    $pidTxt = Get-Content $LOCK -ErrorAction SilentlyContinue
    $alive = $false
    if ($pidTxt) { try { Get-Process -Id ([int]$pidTxt) -ErrorAction Stop | Out-Null; $alive = $true } catch {} }
    if ($alive) { Write-Host "另一个队列进程 (PID $pidTxt) 在跑，退出。" -ForegroundColor Yellow; exit 0 }
    Remove-Item $LOCK -Force
}
Set-Content -Path $LOCK -Value $PID -Encoding utf8

function Invoke-Sealed {
    param([string]$Label, [string[]]$CmdArgs, [string]$Log, [int]$MaxAttempts = 4, [int]$WaitSeconds = 120)
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        $p = Start-Process -FilePath $PY -ArgumentList $CmdArgs -WorkingDirectory $REPO `
             -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
             -NoNewWindow -PassThru -Wait
        if ($p.ExitCode -eq 0) { return }
        Write-Host "  !! $Label 第 $a 次失败 (exit $($p.ExitCode))，$WaitSeconds 秒后原命令重跑（内存不足会自动等到位）" -ForegroundColor Yellow
        Start-Sleep -Seconds $WaitSeconds
    }
    throw "$Label 连续失败 $MaxAttempts 次，需人工介入"
}

try {
    # 折表机械产出，不得手写
    $json = & $PYMAIN "$REPO\scripts\emit_folds.py" --processed $P --first $FIRST --last $LAST
    if ($LASTEXITCODE -ne 0) { throw "emit_folds.py 失败" }
    Set-Content -Path $FOLDS -Value $json -Encoding utf8
    $todo = $json | ConvertFrom-Json
    Write-Host "队列：$($todo.Count) 折（fold$('{0:d2}' -f $FIRST)-fold$('{0:d2}' -f $LAST)），每折 = 内层选参 + 三 seed 拟合 + 打分封存" -ForegroundColor Cyan

    # 年度特征缓存（含训练目标列 y，落在自带哨兵的 cache_sealed 下）
    if (Test-Path "$CACHE\base\manifest.json") {
        Write-Host "缓存：已完成，跳过"
    } else {
        Write-Host "缓存：开始 $(Get-Date -Format HH:mm:ss)"
        Invoke-Sealed "PREPARE" @(
            "$REPO\scripts\gbdt_baseline.py","--config",$CONFIG,"--out-dir",$OUT,
            "--cache-dir",$CACHE,"--folds-json",$FOLDS,"--models","xgboost",
            "--sealed","--prepare-only"
        ) "$LOGDIR\prepare.log"
        Write-Host "缓存：成功 $(Get-Date -Format HH:mm:ss)"
    }

    $i = 0
    foreach ($f in $todo) {
        $i++
        $dir = "$SEALED\$($f.n)"
        if (Test-Path "$dir\SEALED_MANIFEST.json") {
            Write-Host "[$i/$($todo.Count)] $($f.n)  已完成，跳过"
            continue
        }
        Write-Host "[$i/$($todo.Count)] $($f.n)  train=[$($f.ts)..$($f.te)] val=[$($f.vs)..$($f.ve)]  开始 $(Get-Date -Format HH:mm:ss)" -ForegroundColor Cyan
        $t0 = Get-Date
        # --append-ledger 由脚本内部按 append_ledger 的格式写（LF、无 BOM、按 tag 幂等）
        Invoke-Sealed "FOLD $($f.n)" @(
            "$REPO\scripts\gbdt_baseline.py","--config",$CONFIG,"--out-dir",$OUT,
            "--cache-dir",$CACHE,"--folds-json",$FOLDS,"--models","xgboost",
            "--folds",$f.n,"--sealed","--append-ledger"
        ) "$LOGDIR\fold_$($f.n).log"
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        Write-Host "  成功 $(Get-Date -Format HH:mm:ss)（$mins 分钟）" -ForegroundColor Green
    }
    Set-Content -Path "$SEALED\QUEUE.DONE" -Value (Get-Date -Format o) -Encoding utf8
    Write-Host "=== 队列完成。结果处于封存状态，读取须另行授权。 ===" -ForegroundColor Green
}
finally {
    Remove-Item $LOCK -Force -ErrorAction SilentlyContinue
}
