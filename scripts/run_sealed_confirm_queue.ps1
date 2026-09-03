# 封存确认队列（2026-09-02 授权：计算 != 读取）
# 严格单 GPU 串行 + 互斥锁 + 失败重试 + 原命令续训。绝不强杀正在跑的 CUDA 进程。
# 控制台只显示：折号 / 阶段 / 成功失败。训练 inner loss 等一律进封存日志。
# 用 Start-Process 重定向：PS5.1 的 *> 会把原生 exe 的 stderr 包成 NativeCommandError。
$ErrorActionPreference = "Continue"
$REPO = "F:\quant\us-quant-pipeline"
$PY   = "$REPO\.venv\Scripts\python.exe"
$P    = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$LB   = 90
$SEALROOT = "$REPO\outputs\sealed_confirm"
$LOGDIR   = "$SEALROOT\_logs"
$LOCK     = "$SEALROOT\.queue.lock"

Set-Location $REPO
New-Item -ItemType Directory -Force $SEALROOT | Out-Null
New-Item -ItemType Directory -Force $LOGDIR   | Out-Null

if (Test-Path $LOCK) {
    $pidTxt = Get-Content $LOCK -ErrorAction SilentlyContinue
    $alive = $false
    if ($pidTxt) { try { Get-Process -Id ([int]$pidTxt) -ErrorAction Stop | Out-Null; $alive = $true } catch {} }
    if ($alive) { Write-Host "另一个队列进程 (PID $pidTxt) 在跑，退出。" -ForegroundColor Yellow; exit 0 }
    Remove-Item $LOCK -Force
}
Set-Content -Path $LOCK -Value $PID -Encoding utf8

function Invoke-Sealed {
    param([string]$Label, [string[]]$CmdArgs, [string]$Log, [int]$MaxAttempts = 4)
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        $p = Start-Process -FilePath $PY -ArgumentList $CmdArgs -WorkingDirectory $REPO `
             -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" `
             -NoNewWindow -PassThru -Wait
        if ($p.ExitCode -eq 0) { return }
        Write-Host "  !! $Label 第 $a 次失败 (exit $($p.ExitCode))，60s 后原命令重跑（OOM 自动续训）" -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
    throw "$Label 连续失败 $MaxAttempts 次，需人工介入"
}

function Test-Sealed { param([string]$Dir) return (Test-Path "$Dir\SEALED_MANIFEST.json") }

try {
    $json  = & $PY scripts\emit_folds.py --processed $P
    $folds = $json | ConvertFrom-Json
    # fold43 (val 2024-01-03..2024-07-02) 落在 Kronos 语料窗内（截至 2024-06），
    # 不是干净证据；用户 2026-09-02 裁定移除。干净封存窗 = fold44 + fold45。
    $todo  = $folds | Where-Object { ($_.n -ge "fold05" -and $_.n -le "fold35") -or ($_.n -ge "fold44") }
    Write-Host "队列：$($todo.Count) 折（05-35 与 44-45），每折 = FT 训练 + FT 打分 + ZS 打分" -ForegroundColor Cyan

    $i = 0
    foreach ($f in $todo) {
        $i++
        $out    = "$REPO\outputs\$($f.n)_lb" + $LB + "_s0_poolB_universe"
        $ftSeal = "$out\eval_sealed_ft_$($f.n)"
        $zsSeal = "$REPO\outputs\zeroshot_base\eval_sealed_zs_$($f.n)"
        Write-Host "[$i/$($todo.Count)] $($f.n)  train=[$($f.ts)..$($f.te)] val=[$($f.vs)..$($f.ve)]" -ForegroundColor Cyan

        if (Test-Path "$out\predictor_final") {
            Write-Host "  训练：已完成，跳过"
        } else {
            Write-Host "  训练：开始 $(Get-Date -Format HH:mm:ss)"
            Invoke-Sealed "TRAIN $($f.n)" @(
                "-m","kronos_ft.train",
                "--panel","$P\panel_kronos_adj.parquet",
                "--index-parquet","$P\market_index.parquet",
                "--train-start",$f.ts,"--train-end",$f.te,
                "--lookback","$LB","--seed","0","--stage","both","--out",$out,
                "--universe-parquet","$P\universe.parquet",
                "--index-cache","$P\index_cache\lb$($LB)_full.parquet"
            ) "$LOGDIR\train_$($f.n).log"
            Write-Host "  训练：成功 $(Get-Date -Format HH:mm:ss)"
        }

        if (Test-Sealed $ftSeal) {
            Write-Host "  FT 打分：已完成，跳过"
        } else {
            Write-Host "  FT 打分：开始 $(Get-Date -Format HH:mm:ss)"
            Invoke-Sealed "SCORE-FT $($f.n)" @(
                "scripts\evaluate_fold.py","--model-dir",$out,"--processed",$P,
                "--val-start",$f.vs,"--val-end",$f.ve,"--lookback","$LB",
                "--tag","sealed_ft_$($f.n)","--amp","bf16","--batch-size","128",
                "--sample-count","5","--device","cuda","--sealed"
            ) "$LOGDIR\score_ft_$($f.n).log"
            Write-Host "  FT 打分：成功 $(Get-Date -Format HH:mm:ss)"
        }

        if (Test-Sealed $zsSeal) {
            Write-Host "  ZS 打分：已完成，跳过"
        } else {
            Write-Host "  ZS 打分：开始 $(Get-Date -Format HH:mm:ss)"
            Invoke-Sealed "SCORE-ZS $($f.n)" @(
                "scripts\evaluate_fold.py","--model-dir","$REPO\outputs\zeroshot_base","--processed",$P,
                "--val-start",$f.vs,"--val-end",$f.ve,"--lookback","$LB",
                "--tag","sealed_zs_$($f.n)","--amp","bf16","--batch-size","128",
                "--sample-count","5","--device","cuda","--sealed"
            ) "$LOGDIR\score_zs_$($f.n).log"
            Write-Host "  ZS 打分：成功 $(Get-Date -Format HH:mm:ss)"
        }
    }
    Set-Content -Path "$SEALROOT\QUEUE.DONE" -Value (Get-Date -Format o) -Encoding utf8
    Write-Host "=== 队列完成。结果处于封存状态，读取须另行授权。 ===" -ForegroundColor Green
}
finally {
    Remove-Item $LOCK -Force -ErrorAction SilentlyContinue
}
