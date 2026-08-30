# lookback 粗筛第二轮（预注册 §2 第 2 步）：存活档补 seeds {1,2}。
# 第一轮无淘汰 → 存活档 = {60, 90, 200}，共 3 × 4 折 × 2 seed = 24 次训练+评估。
# 幂等：已完成的阶段自动跳过，中断后原样重跑即续。

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

# 单步重试：训练可断点续（已完成阶段跳过、未完成阶段逐位续上），故重试
# 几乎零成本。动机：2026-08-28 一次 nvlddmkm(153) 显卡驱动闪断在第 1/24 格
# 就把整批毙了——偶发硬件层故障不该终止 2 天的跑批。
function Invoke-WithRetry {
    param([string]$Label, [scriptblock]$Action, [int]$MaxAttempts = 3)
    for ($a = 1; $a -le $MaxAttempts; $a++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        if ($a -lt $MaxAttempts) {
            Write-Host "!! $Label 第 $a 次失败（exit $LASTEXITCODE），30s 后重试（续训）" `
                -ForegroundColor Yellow
            Start-Sleep -Seconds 30
        }
    }
    throw "$Label 连续 $MaxAttempts 次失败——需人工介入"
}

$folds = @(
    @{ n = "fold01"; ts = "2000-01-03"; te = "2002-12-20"; vs = "2003-01-02"; ve = "2003-06-30" },
    @{ n = "fold02"; ts = "2000-07-03"; te = "2003-06-20"; vs = "2003-07-01"; ve = "2003-12-31" },
    @{ n = "fold03"; ts = "2001-01-02"; te = "2003-12-22"; vs = "2004-01-02"; ve = "2004-06-30" },
    @{ n = "fold04"; ts = "2001-07-02"; te = "2004-06-22"; vs = "2004-07-01"; ve = "2004-12-31" }
)

$total = 3 * 4 * 2
$i = 0
foreach ($seed in @(1, 2)) {
    foreach ($lb in @(60, 90, 200)) {
        $cache = "$P\index_cache\lb${lb}_full.parquet"
        foreach ($f in $folds) {
            $i++
            $out = "outputs\$($f.n)_lb${lb}_s${seed}_poolB_universe"
            Write-Host "=== [$i/$total] TRAIN lb=$lb seed=$seed $($f.n) ===" -ForegroundColor Cyan
            Invoke-WithRetry "TRAIN lb=$lb seed=$seed $($f.n)" {
                & $PY -m kronos_ft.train --panel "$P\panel_kronos_adj.parquet" `
                    --index-parquet "$P\market_index.parquet" `
                    --train-start $f.ts --train-end $f.te `
                    --lookback $lb --seed $seed --stage both --out $out `
                    --universe-parquet "$P\universe.parquet" --index-cache $cache
            }

            Write-Host "=== [$i/$total] EVAL  lb=$lb seed=$seed $($f.n) ===" -ForegroundColor Cyan
            Invoke-WithRetry "EVAL lb=$lb seed=$seed $($f.n)" {
                & $PY scripts\evaluate_fold.py --model-dir $out --processed $P `
                    --val-start $f.vs --val-end $f.ve --lookback $lb `
                    --tag "screen_lb${lb}_s${seed}_$($f.n)"
            }
        }
    }
}

Write-Host ""
Write-Host "=== 粗筛第二轮完成（24 格）。下一步：3 seed 平均后跑 compare_arms 95% 判据 ===" -ForegroundColor Green
