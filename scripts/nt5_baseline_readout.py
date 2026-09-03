"""NT=5 冻结构造在开发折 36–42 上的基准读数（预注册执行器）。

依据：experiments/ledger.md 2026-09-03「持仓期 6 → 5 日（NT=5）」条目——
「新构造在折 36–42 的基准读数在本条登记之后方可运行，且只作新冻结构造的基准，
**不作与 NT=6 的比较**（B 层）」。本脚本因此：

* **只计算、只打印 NT=5**。不计算、不打印任何其它 NT；也不读取 NT=6 的既有结果。
* 只读已消耗的开发折 36–42（经 `signals.kronos_adapter` 的折号白名单与
  `assert_readable` 守卫），FT 与 ZS 两臂各算一遍（k=2 裁定）。
* 价格/成交额加载复用 `scripts/compare_arms_money.py::load_prices`（同一口径：
  2020-06-01..2024-01-05、dropna(DlyRet)、oc = Close/|Open|−1、adv20 shift(1)）。
* 构造用 `src/portfolio/construction.frozen_long_only_returns(nt=5)`——该函数已于
  2026-09-03 在 NT=6 上与原脚本逐位对拍通过（ledger tooling 条目）。

读数（按 CLAUDE.md §五：主指标为冻结构造下的净夏普；RankIC 不在本脚本内）：
毛年化（算术）、成本网格 2/4/8/12/16/22bp 下净年化、每 bp 拖累、breakeven 单边成本、
年单边交易次数（= 2×252×mean(turn)/NT，与 HANDOFF 的「48×」口径一致）、
净(8bp)年化波动与夏普、最大回撤、最长水下、毛日收益 NW(5) t、逐折毛年化与正折数。

用途：K12 功效重算、小试协议 v2（NT=5）的规模预算、确认协议 v4 的开发折 BE。
**不得用于选择 NT、退出线、universe 或任何构造参数。**
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable          # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat    # noqa: E402
from portfolio.construction import (                      # noqa: E402
    frozen_long_only_returns, scores_frame_to_by_day,
)
from signals.kronos_adapter import scores_path            # noqa: E402

NT = 5
TOPN, EXIT_PCT, MIN_NAMES = 500, 0.30, 50
FOLDS = list(range(36, 43))
ARMS = ("ft", "zs")
COST_GRID_BP = (2, 4, 8, 12, 16, 22)
NW_LAG = 5
OUT_JSON = REPO / "outputs" / "nt5_baseline_readout.json"


def log(m: str) -> None:
    print(m, flush=True)


def _load_cam():
    spec = importlib.util.spec_from_file_location(
        "compare_arms_money", REPO / "scripts" / "compare_arms_money.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def max_drawdown_and_underwater(daily: np.ndarray) -> tuple[float, int]:
    nav = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    mdd = float(dd.min())
    # 最长水下：连续 dd<0 的最长段
    longest = cur = 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    return mdd, int(longest)


def summarize(gross: pd.Series, turn: pd.Series, fold: pd.Series) -> dict:
    g = gross.to_numpy(dtype=float)
    t = turn.to_numpy(dtype=float)
    n = len(g)
    drag_per_bp_daily = 2.0 * (1.0 / 1e4) * t / NT          # 每 bp 单边成本的日拖累
    drag_per_bp_annual_pct = float(drag_per_bp_daily.mean() * 252 * 100)
    gross_annual_pct = float(g.mean() * 252 * 100)
    be_bp = gross_annual_pct / drag_per_bp_annual_pct if drag_per_bp_annual_pct > 0 else np.nan
    one_way_trades_per_year = float(2.0 * 252 * t.mean() / NT)
    nw = newey_west_tstat(pd.Series(g), NW_LAG)
    out = {
        "n_days": n,
        "gross_annual_pct": gross_annual_pct,
        "gross_daily_nw_t": float(nw["t"]),
        "gross_daily_nw_se_annual_pct": float(nw["se"] * 252 * 100),
        "drag_per_bp_annual_pct": drag_per_bp_annual_pct,
        "breakeven_oneway_bp": float(be_bp),
        "oneway_trades_per_year": one_way_trades_per_year,
        "mean_sleeve_turn": float(t.mean()),
        "daily_book_turnover_oneway": float(t.mean() / NT),
        "net_by_cost_bp": {},
    }
    for bp in COST_GRID_BP:
        net = g - bp * drag_per_bp_daily
        vol = float(net.std(ddof=1) * np.sqrt(252) * 100)
        ann = float(net.mean() * 252 * 100)
        geo = float((np.prod(1.0 + net) ** (252.0 / n) - 1.0) * 100)
        mdd, uw = max_drawdown_and_underwater(net)
        out["net_by_cost_bp"][str(bp)] = {
            "net_annual_pct": ann, "net_geo_annual_pct": geo,
            "vol_annual_pct": vol, "sharpe": ann / vol if vol > 0 else np.nan,
            "max_drawdown_pct": mdd * 100, "longest_underwater_days": uw,
        }
    per_fold = {}
    for f, grp in pd.DataFrame({"g": g, "fold": fold.to_numpy()}).groupby("fold"):
        per_fold[str(f)] = float(grp["g"].mean() * 252 * 100)
    out["per_fold_gross_annual_pct"] = per_fold
    out["folds_positive"] = int(sum(v > 0 for v in per_fold.values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    cam = _load_cam()
    assert_readable(cam.P)
    log("加载价格/成交额（复用 compare_arms_money.load_prices，同口径）...")
    ret, oc, adv = cam.load_prices()

    result = {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "construction": {"NT": NT, "TOPN": TOPN, "EXIT_PCT": EXIT_PCT,
                             "MIN_NAMES": MIN_NAMES, "entry_pct": 0.10,
                             "execution": "t close score -> t+1 open, fresh names eat open->close"},
            "folds": FOLDS, "arms": list(ARMS), "cost_grid_bp": list(COST_GRID_BP),
            "nw_lag": NW_LAG,
            "preregistration": "ledger 2026-09-03 持仓期 6→5 条目；只报 NT=5，不与 NT=6 比较",
        },
        "arms": {},
    }
    for arm in ARMS:
        parts = []
        for f in FOLDS:
            p = scores_path(f, arm)
            assert_readable(p)
            sc = pd.read_parquet(p, columns=["PERMNO", "signal_date", "score"])
            by_day = scores_frame_to_by_day(sc, min_names=MIN_NAMES)
            del sc
            df = frozen_long_only_returns(by_day, ret, oc, adv, topn=TOPN,
                                          cost_bp=0.0, exit_pct=EXIT_PCT,
                                          nt=NT, min_names=MIN_NAMES)
            # 自检：函数内部 cost_bp=8 的 r 应等于 毛 − 8bp 拖累
            df8 = frozen_long_only_returns(by_day, ret, oc, adv, topn=TOPN,
                                           cost_bp=8.0, exit_pct=EXIT_PCT,
                                           nt=NT, min_names=MIN_NAMES)
            derived = df["r"] - 8.0 / 1e4 * 2.0 * df["turn"] / NT
            if not np.allclose(derived.to_numpy(), df8["r"].to_numpy(), atol=1e-15):
                raise AssertionError(f"{arm} fold{f}: turn 列与成本公式不一致")
            df["fold"] = f
            parts.append(df)
            log(f"  {arm} fold{f}: days={len(df)}")
        allf = pd.concat(parts).sort_index()
        result["arms"][arm] = summarize(allf["r"], allf["turn"], allf["fold"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    log(f"\n=== NT={NT} / top{TOPN} / 进前10% 出前{int(EXIT_PCT*100)}% / 折 36–42 ===")
    for arm, s in result["arms"].items():
        n8 = s["net_by_cost_bp"]["8"]
        log(f"[{arm.upper()}] days={s['n_days']}  毛年化 {s['gross_annual_pct']:.2f}%  "
            f"NW t {s['gross_daily_nw_t']:.2f}  正折 {s['folds_positive']}/7")
        log(f"       每bp拖累 {s['drag_per_bp_annual_pct']:.4f}%/年  年单边交易 {s['oneway_trades_per_year']:.1f}×  "
            f"BE {s['breakeven_oneway_bp']:.1f}bp")
        log(f"       净(8bp) {n8['net_annual_pct']:.2f}%  波动 {n8['vol_annual_pct']:.2f}%  "
            f"夏普 {n8['sharpe']:.2f}  MDD {n8['max_drawdown_pct']:.1f}%  水下 {n8['longest_underwater_days']} 日")
        log("       净年化网格: " + "  ".join(
            f"{bp}bp={v['net_annual_pct']:.2f}%" for bp, v in s["net_by_cost_bp"].items()))
    log(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
