"""在真实开发折（36–42）上对拍 `portfolio.construction` 与 `compare_arms_money`。

用途
----
`tests/test_portfolio.py` 已经用合成数据证明两者逐位一致，但合成数据不可能覆盖
真实分数/ADV 的全部并列模式（真实 PERMNO 的 dict 键序、真实 ADV 的整数并列、
真实 scores.parquet 的行序）。本脚本是**上线前的最后一道核对**：拿真实的
`scores.parquet` 跑一遍，断言 `r` 列 `np.array_equal`。

只读**已消耗**的开发折 36–42（CLAUDE.md §四：这七折已被数十次决策读数消耗，
读数为方向性证据、非无偏估计）。本脚本**不产生任何新读数**——它只比较两个
实现的输出是否相同，不打印收益、不打印夏普、不写 JSON。折 05–35 与 2024-07
起的封存窗**一眼都不看**。

前置条件（今晚不跑，等 GPU 队列结束）
-------------------------------------
1. **≥ 6 GB 空闲提交内存**。`load_prices()` 要读 2020-06-01..2024-01-05 的
   `panel_raw` 六列（CLAUDE.md §七：提交上限常年吃紧，已多次 OOM）。
   跑之前先查：
   `(Get-Counter '\\Memory\\Committed Bytes').CounterSamples[0].CookedValue/1GB`
2. 读任何 parquet 之前先过 `crsp_pipeline.sealed.assert_readable` 守卫
   （2026-09-02 裁定：计算授权 != 读取授权）。本脚本对价格面板与每个折的
   分数目录都调了守卫；命中哨兵会直接抛错退出，不会静默跳过。
3. 不修改 `scripts/compare_arms_money.py`——本脚本以模块方式加载它，只调
   `load_prices()` 与 `arm_returns()`，一个字节都不写回。

用法
----
    .venv\\Scripts\\python.exe scripts\\check_construction_parity.py
    .venv\\Scripts\\python.exe scripts\\check_construction_parity.py --lb 90 --topn 500

退出码 0 = 逐位一致；1 = 有差异（此时构造已经漂移，**所有既有臂间读数作废**，
先查 diff 再谈别的）。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.sealed import assert_readable            # noqa: E402
from portfolio.construction import (                        # noqa: E402
    frozen_long_only_returns,
    scores_frame_to_by_day,
)


def _load_reference_module():
    """以模块方式加载 scripts/compare_arms_money.py（只读，不改动）。"""
    path = REPO_ROOT / "scripts" / "compare_arms_money.py"
    spec = importlib.util.spec_from_file_location("_compare_arms_money_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _new_impl_returns(cam, lb, ret, oc, adv, topn, cost_bp) -> pd.DataFrame:
    """用新实现走一遍同样的七折，列与 `arm_returns` 对齐（date / r / fold）。"""
    parts = []
    for fold in cam.FOLDS:
        d = cam.OUT / f"{fold}_lb{lb}_s0_poolB_universe" / f"eval_amp_lb{lb}_{fold}"
        assert_readable(d)
        sc = pd.read_parquet(d / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"])
        by_day = scores_frame_to_by_day(sc)
        del sc
        one = frozen_long_only_returns(by_day, ret, oc, adv,
                                       topn=topn, cost_bp=cost_bp)
        parts.append(one.assign(fold=fold))
        del by_day, one
    return pd.concat(parts).sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lb", type=int, nargs="+", default=[90, 200])
    ap.add_argument("--topn", type=int, default=500)
    ap.add_argument("--cost-bp", type=float, default=8.0)
    args = ap.parse_args()

    cam = _load_reference_module()
    # 守卫要在**任何** parquet 打开之前跑完：原脚本的 arm_returns() 自己不带守卫。
    assert_readable(cam.P)
    for lb in args.lb:
        for fold in cam.FOLDS:
            assert_readable(cam.OUT / f"{fold}_lb{lb}_s0_poolB_universe")

    print("读取价格面板（列裁剪 + 日期下推，同原脚本）...", flush=True)
    ret, oc, adv = cam.load_prices()

    all_ok = True
    for lb in args.lb:
        print(f"\n=== lb{lb} / top{args.topn} / {args.cost_bp}bp ===", flush=True)
        ref = cam.arm_returns(lb, ret, oc, adv, args.topn, args.cost_bp)
        new = _new_impl_returns(cam, lb, ret, oc, adv, args.topn, args.cost_bp)

        idx_ok = list(ref.index) == list(new.index)
        fold_ok = list(ref["fold"]) == list(new["fold"])
        print(f"  行数  ref={len(ref)}  new={len(new)}")
        print(f"  成交日序列一致: {idx_ok}    折标签一致: {fold_ok}")
        if not (idx_ok and fold_ok):
            all_ok = False
            print("  !! 索引就已经不一致，逐位比较无意义，先查 by_day 构造")
            continue

        a, b = ref["r"].to_numpy(), new["r"].to_numpy()
        exact = bool(np.array_equal(a, b))
        max_abs = float(np.nanmax(np.abs(a - b))) if len(a) else 0.0
        n_diff = int((a != b).sum())
        print(f"  np.array_equal: {exact}")
        print(f"  最大绝对差: {max_abs:.3e}    不等元素数: {n_diff}/{len(a)}")
        if not exact:
            all_ok = False
            bad = np.flatnonzero(a != b)[:10]
            for t in bad:
                print(f"    {ref.index[t].date()}  ref={a[t]!r}  new={b[t]!r}")

    print(f"\n结论：{'逐位一致' if all_ok else '存在差异——构造已漂移'}")
    print("本脚本不产出任何收益读数；折 05–35 与 2024-07 起的窗口未触碰。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
