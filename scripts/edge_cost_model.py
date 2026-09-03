"""用 EDGE 替换「全池统一 8bp」，并量化持仓相对池均值的成本高估。

依赖 outputs/edge_monthly.parquet（由 scripts/edge_spread.py 生成，已与官方
参考实现逐位对拍通过）。

三个交付
--------
A. **年代曲线**：top500 池的有效价差逐年（等权均值 + 中位），2000-2025。
   直接回答「8bp 这个假设在哪些年代成立」。
B. **持仓加权 vs 池均值**：K6b 查出持仓在 top500 内偏向低价/不流动端
   （corr(1/价格)=+0.382 等），故池均值必然低估。本节给出高估倍数。
C. **按逐持仓 EDGE 重算净超额与 breakeven**，对照 8bp 口径。

口径
----
* 单边成本 = EDGE / 2（有效价差的一半），用**上月**估计避免点时性争议；
* 零售规模（<=$1M）下冲击项 <=1.3bp（见 scratchpad/cap.py 的测算），故略去；
  Alpaca 免佣，故 fees≈0。**成本 ≈ 半价差**；
* 组合构造与 analysis_sevenfold.py 逐行一致（top500 / 六档错位 / 进前 10% /
  跌出前 30% 才卖 / t+1 开盘），换手按**纯多腿**计（compare_arms_money.py 口径）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
EXIT_PCT, NT, TOPN = 0.30, 6, 500

ERAS = {
    "2003-04": (["fold01", "fold02", "fold03", "fold04"], "2002-10-01", "2005-01-05",
                {"fold01": "eval_poolB_universe", "fold02": "eval_poolB_universe_fold02",
                 "fold03": "eval_poolB_universe_fold03", "fold04": "eval_poolB_universe_fold04"}),
    "2020H2-23": ([f"fold{f}" for f in range(36, 43)], "2020-04-01", "2024-01-05",
                  {f"fold{f}": f"eval_amp_lb90_fold{f}" for f in range(36, 43)}),
}


def log(m):
    print(m, flush=True)


def load_edge():
    E = pd.read_parquet(OUT / "edge_monthly.parquet")
    E = E[E.edge.notna() & (E.edge > 0)].copy()
    # 用上月估计：把 ym 前移一个月作为「可用月」
    per = pd.PeriodIndex(E.ym, freq="M") + 1
    E["use_ym"] = per.astype(str)
    return {(int(p), y): float(e) for p, y, e in
            zip(E.PERMNO, E.use_ym, E.edge)}


def era_book(folds, lo, hi, evd, EDGE):
    """重建持仓，逐日记录：池均值半价差、持仓加权半价差、纯多腿换手、毛超额。"""
    p = pd.read_parquet(P / "panel_raw.parquet",
                        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose",
                                 "DlyRet", "DlyPrcVol"],
                        filters=[("DlyCalDt", ">=", pd.Timestamp(lo)),
                                 ("DlyCalDt", "<=", pd.Timestamp(hi))])
    p["DlyCalDt"] = pd.to_datetime(p["DlyCalDt"])
    p = p.dropna(subset=["DlyRet"]).sort_values(["PERMNO", "DlyCalDt"])
    p["oc"] = np.where(p.DlyOpen.abs() > 0, p.DlyClose / p.DlyOpen.abs() - 1.0, np.nan)
    p["adv20"] = (p.groupby("PERMNO")["DlyPrcVol"].rolling(20, min_periods=10).mean()
                   .reset_index(level=0, drop=True)).groupby(p["PERMNO"]).shift(1)
    ret = {d: dict(zip(g.PERMNO, g.DlyRet)) for d, g in p.groupby("DlyCalDt")}
    oc = {d: dict(zip(g.PERMNO, g.oc)) for d, g in p.groupby("DlyCalDt")}
    adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in p.groupby("DlyCalDt")}
    del p

    rows = []
    for fold in folds:
        d = OUT / f"{fold}_lb90_s0_poolB_universe" / evd[fold]
        sc = pd.read_parquet(d / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"]).dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        by_day = {day: dict(zip(g.PERMNO, g.score))
                  for day, g in sc.groupby("signal_date") if len(g) >= 50}
        del sc
        days = sorted(by_day)
        book = [None] * NT
        for i, day in enumerate(days):
            s = by_day[day]
            a = adv.get(day, {})
            elig = [x for x in s if x in a and np.isfinite(a[x])]
            if len(elig) > TOPN:
                elig = sorted(elig, key=lambda x: -a[x])[:TOPN]
            elig = set(elig)
            s = {x: v for x, v in s.items() if x in elig}
            n = len(s)
            if n < 50:
                continue
            pct = (pd.Series(s).rank() / n).to_dict()
            k = max(1, n // 10)
            order = sorted(pct, key=lambda x: -pct[x])
            j = i % NT
            ym = pd.Period(day, freq="M").strftime("%Y-%m")
            prev = book[j]
            if prev is None:
                nb, fresh, turn = list(order[:k]), set(order[:k]), 0.0
            else:
                keepn = [x for x in prev if x in pct and pct[x] >= 1 - EXIT_PCT][:k]
                add = [x for x in order if x not in set(keepn)][:k - len(keepn)]
                nb, fresh = keepn + add, set(add)
                turn = (k - len(keepn)) / k
            book[j] = nb
            if i + 1 >= len(days) or i < NT:
                continue
            nd = days[i + 1]
            rm, om = ret.get(nd, {}), oc.get(nd, {})
            if not rm:
                continue
            held = [x for t in range(NT) if book[t] for x in book[t]]
            e_pool = [EDGE[(x, ym)] for x in s if (x, ym) in EDGE]
            e_held = [EDGE[(x, ym)] for x in held if (x, ym) in EDGE]
            e_trade = [EDGE[(x, ym)] for x in fresh if (x, ym) in EDGE]
            vals = []
            for t in range(NT):
                nm = book[t]
                if not nm:
                    continue
                rs = [(om.get(x) if (t == j and x in fresh) else rm.get(x)) for x in nm]
                rs = [v for v in rs if v is not None and np.isfinite(v)]
                if rs:
                    vals.append(np.mean(rs))
            rl = float(np.mean(vals)) if vals else 0.0
            bench = float(np.mean([rm[x] for x in pct if x in rm])) if pct else 0.0
            rows.append(dict(
                fold=fold, date=nd, gross=rl - bench, turn=turn,
                sp_pool=np.mean(e_pool) / 2 if e_pool else np.nan,
                sp_held=np.mean(e_held) / 2 if e_held else np.nan,
                sp_trade=np.mean(e_trade) / 2 if e_trade else np.nan,
                edge_cov=len(e_held) / max(1, len(held))))
    return pd.DataFrame(rows)


def main():
    log("载入 EDGE 月度面板（用上月估计）...")
    EDGE = load_edge()
    log(f"  {len(EDGE):,} 个 (股, 可用月)")

    E = pd.read_parquet(OUT / "edge_monthly.parquet")
    E = E[E.edge.notna() & (E.edge > 0)]
    E["year"] = E.ym.str[:4].astype(int)

    log("\n" + "=" * 72)
    log("A. 有效价差的年代曲线（全池，单边成本 = EDGE/2）")
    log("=" * 72)
    log(f"  {'年':>6}{'股数':>8}{'中位半价差':>12}{'均值半价差':>12}")
    curve = {}
    for y, g in E.groupby("year"):
        med, mean = g.edge.median() / 2 * 1e4, g.edge.mean() / 2 * 1e4
        curve[int(y)] = {"median_bp": float(med), "mean_bp": float(mean),
                         "n": int(len(g))}
        if y % 2 == 0 or y >= 2020:
            log(f"  {y:>6}{len(g):>8,}{med:>11.1f}bp{mean:>11.1f}bp")

    res = {"era_curve_full_pool": curve, "eras": {}}
    log("\n" + "=" * 72)
    log("B/C. 持仓加权 vs 池均值，并按逐持仓 EDGE 重算")
    log("=" * 72)
    for era, (folds, lo, hi, evd) in ERAS.items():
        R = era_book(folds, lo, hi, evd, EDGE)
        R = R.dropna(subset=["sp_pool", "sp_held", "sp_trade"])
        g_ann = R.gross.mean() * 252 * 100
        pool, held, trade = R.sp_pool.mean(), R.sp_held.mean(), R.sp_trade.mean()
        # 成本：每天换 1/NT 档、换手 turn、往返 2 次，按**成交名字**的半价差计
        drag_edge = (2.0 * (R.sp_trade * R.turn) / NT).mean() * 252 * 100
        drag_pool = (2.0 * (R.sp_pool * R.turn) / NT).mean() * 252 * 100
        drag_8bp = (2.0 * (8 / 1e4) * R.turn / NT).mean() * 252 * 100
        be = g_ann / (drag_edge / (trade * 1e4)) if trade > 0 else np.nan
        log(f"\n  [{era}]  {len(R)} 天   毛超额 {g_ann:+.2f}%/年   EDGE 覆盖率 {R['edge_cov'].mean():.1%}")
        log(f"    池均值半价差   {pool*1e4:6.1f}bp")
        log(f"    持仓加权半价差 {held*1e4:6.1f}bp   （相对池均值 {held/pool:.2f}×）")
        log(f"    **成交名字半价差 {trade*1e4:6.1f}bp   （相对池均值 {trade/pool:.2f}×）**")
        log(f"    成本拖累：EDGE 口径 {drag_edge:5.2f}%/年  |  池均值口径 {drag_pool:5.2f}%"
            f"  |  固定8bp口径 {drag_8bp:5.2f}%")
        log(f"    **净超额：EDGE {g_ann-drag_edge:+6.2f}%  |  固定8bp {g_ann-drag_8bp:+6.2f}%**")
        log(f"    breakeven 单边成本 {be:.1f}bp   （实测成交半价差 {trade*1e4:.1f}bp）"
            f"  -> 余量 {be/(trade*1e4):.2f}×")
        res["eras"][era] = {
            "n_days": int(len(R)), "gross_ann": float(g_ann),
            "half_spread_pool_bp": float(pool * 1e4),
            "half_spread_held_bp": float(held * 1e4),
            "half_spread_traded_bp": float(trade * 1e4),
            "tilt_traded_over_pool": float(trade / pool),
            "drag_edge_pct": float(drag_edge), "drag_flat8bp_pct": float(drag_8bp),
            "net_edge_pct": float(g_ann - drag_edge),
            "net_flat8bp_pct": float(g_ann - drag_8bp),
            "breakeven_bp": float(be), "edge_coverage": float(R['edge_cov'].mean())}
    (OUT / "edge_cost_model.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/edge_cost_model.json")


if __name__ == "__main__":
    main()
