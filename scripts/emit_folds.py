"""机械输出 walk-forward 折边界（JSON）。**不得手写窗口** —— 队列脚本调用此文件。

    python scripts/emit_folds.py --processed <P> [--first 5 --last 35]

边界完全来自 crsp_pipeline.splits.walk_forward_folds（训 3 年 / 验 6 个月 /
滚动 6 个月 / purge 6 交易日），不传 oos_start 以便同时得到 2024 年后的折。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.splits import walk_forward_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=10_000)
    args = ap.parse_args()

    P = Path(args.processed)
    cal = TradingCalendar.from_market_index(
        pd.read_parquet(P / "market_index.parquet"), "caldt")
    folds = walk_forward_folds(cal, "2000-01-03", cal.dates[-1])
    out = []
    for i, f in enumerate(folds, start=1):
        if not (args.first <= i <= args.last):
            continue
        out.append({
            "n": f"fold{i:02d}",
            "ts": str(f.train_start.date()), "te": str(f.train_end.date()),
            "vs": str(f.val_start.date()), "ve": str(f.val_end.date()),
        })
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
