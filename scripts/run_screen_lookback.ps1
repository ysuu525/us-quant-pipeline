# lookback 粗筛第一轮（预注册 §2）：3 档 × 最早 4 折 × seed0，池 = B（universe）。
# lb90 × fold01-04 已在池子消融中训练并评估完毕，此处只补 lb60 与 lb200。
# 幂等：已完成的阶段自动跳过（train.py 的完成检测），中断后原样重跑即续。

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$P = "F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z"
$PY = ".\.venv\Scripts\python.exe"

# 折边界（splits.walk_forward_folds 生成，oos_start=2024-01-01）
$folds = @(
    @{ n = "fold01"; ts = "2000-01-03"; te = "2002-12-20"; vs = "2003-01-02"; ve = "2003-06-30" },
    @{ n = "fold02"; ts = "2000-07-03"; te = "2003-06-20"; vs = "2003-07-01"; ve = "2003-12-31" },
    @{ n = "fold03"; ts = "2001-01-02"; te = "2003-12-22"; vs = "2004-01-02"; ve = "2004-06-30" },
    @{ n = "fold04"; ts = "2001-07-02"; te = "2004-06-22"; vs = "2004-07-01"; ve = "2004-12-31" }
)

foreach ($lb in @(60, 200)) {
    $cache = "$P\index_cache\lb${lb}_full.parquet"
    foreach ($f in $folds) {
        $out = "outputs\$($f.n)_lb${lb}_s0_poolB_universe"
        Write-Host "=== TRAIN lb=$lb $($f.n) ===" -ForegroundColor Cyan
        & $PY -m kronos_ft.train --panel "$P\panel_kronos_adj.parquet" `
            --index-parquet "$P\market_index.parquet" `
            --train-start $f.ts --train-end $f.te `
            --lookback $lb --seed 0 --stage both --out $out `
            --universe-parquet "$P\universe.parquet" --index-cache $cache
        if ($LASTEXITCODE -ne 0) { throw "训练失败: lb=$lb $($f.n)（原命令重跑即续训）" }

        Write-Host "=== EVAL  lb=$lb $($f.n) ===" -ForegroundColor Cyan
        & $PY scripts\evaluate_fold.py --model-dir $out --processed $P `
            --val-start $f.vs --val-end $f.ve --lookback $lb `
            --tag "screen_lb${lb}_$($f.n)"
        if ($LASTEXITCODE -ne 0) { throw "评估失败: lb=$lb $($f.n)" }
    }
}

Write-Host ""
Write-Host "=== 粗筛第一轮训练+评估完成。下一步跑 compare_arms.py 三档配对判据 ===" -ForegroundColor Green
