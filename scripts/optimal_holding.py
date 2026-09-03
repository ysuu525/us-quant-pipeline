"""持仓期用闭式解算，不在折上搜（Gârleanu-Pedersen 2013 的原则）。

动机（2026-08-31，对抗性复核提出）：持仓期 6 日是在折 36-42 上网格搜出来的，
这消耗了评估折的自由度；而给定 (信号衰减曲线, 换手, 成本曲线)，最优持仓期是
**算得出来的**，不需要在收益上搜。GP (JF 2013) 证明 alpha 指数衰减 + 交易成本下
最优交易速率有闭式解（"aim in front of the target"）。本脚本做该原则的离散版本：

  日频组合收益 = (1/T) * sum_{h=1..T} e(h)          e(h)=入选后第 h 天的边际超额
  日频成本     = 2 * bp * turn(T) / T               每天换一档、换手 turn(T)
  net(T)       = 252 * [ mean(e(1..T)) - 2*bp*turn(T)/T ]

三个输入全部是**机械测量的原语**，不是在 net(T) 上做的搜索：
  * e(h)   —— 逐日取当天分数前 10% 的名字，跟踪其后第 h 天相对池均值的超额；
  * turn(T)—— 只跑建仓/换仓逻辑，不看任何收益；
  * bp     —— 外生成本假设，逐档报告。

判读（先写）：若解析最优 T* 与已在折上搜出的 T=6 一致 → 那次网格搜索没有额外
消耗自由度，可如实声明；若 T* 明显不同 → 现行 T=6 是在噪声上选的，应改用 T*，
并把这次改动记为一次**由先验/解析而非数据驱动**的设计修订。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
FOLDS = list(range(36, 43))
EXIT_PCT, TOPN, HMAX = 0.30, 500, 14
LO, HI = "2020-04-01", "2024-03-01"


def log(m):
    print(m, flush=True)


def load_prices():
    p = pd.read_parquet(P / "panel_raw.parquet",
                        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose",
                                 "DlyRet", "DlyPrcVol"],
                        filters=[("DlyCalDt", ">=", pd.Timestamp(LO)),
                                 ("DlyCalDt", "<=", pd.Timestamp(HI))])
    p["DlyCalDt"] = pd.to_datetime(p["DlyCalDt"])
    p = p.dropna(subset=["DlyRet"]).sort_values(["PERMNO", "DlyCalDt"])
    p["adv20"] = (p.groupby("PERMNO")["DlyPrcVol"].rolling(20, min_periods=10).mean()
                   .reset_index(level=0, drop=True)).groupby(p["PERMNO"]).shift(1)
    ret = {d: dict(zip(g.PERMNO, g.DlyRet)) for d, g in p.groupby("DlyCalDt")}
    adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in p.groupby("DlyCalDt")}
    return ret, adv


def load_scores():
    by_day = {}
    for f in FOLDS:
        d = OUT / f"fold{f}_lb90_s0_poolB_universe" / f"eval_amp_lb90_fold{f}"
        sc = pd.read_parquet(d / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"]).dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        for day, g in sc.groupby("signal_date"):
            if len(g) >= 50:
                by_day[day] = dict(zip(g.PERMNO, g.score))
    return by_day


def eligible(s, a):
    e = [p for p in s if p in a and np.isfinite(a[p])]
    if len(e) > TOPN:
        e = sorted(e, key=lambda p: -a[p])[:TOPN]
    return {p: v for p, v in s.items() if p in set(e)}


def edge_profile(by_day, days, ret, adv):
    """e(h)：入选当天前 10% 的名字，其后第 h 个交易日相对同池等权的超额。"""
    acc = {h: [] for h in range(1, HMAX + 1)}
    for i, day in enumerate(days):
        s = eligible(by_day[day], adv.get(day, {}))
        if len(s) < 50:
            continue
        k = max(1, len(s) // 10)
        top = set(sorted(s, key=lambda p: -s[p])[:k])
        pool = list(s)
        for h in range(1, HMAX + 1):
            if i + h >= len(days):
                break
            rm = ret.get(days[i + h], {})
            if not rm:
                continue
            rt = [rm[p] for p in top if p in rm]
            rp = [rm[p] for p in pool if p in rm]
            if rt and rp:
                acc[h].append(np.mean(rt) - np.mean(rp))
    return {h: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v))
            for h, v in acc.items() if v}


def turnover(by_day, days, adv, T):
    """只跑换仓逻辑测 turn(T)，不看任何收益。"""
    book = [None] * T
    ts = []
    for i, day in enumerate(days):
        s = eligible(by_day[day], adv.get(day, {}))
        n = len(s)
        if n < 50:
            continue
        pct = (pd.Series(s).rank() / n).to_dict()
        k = max(1, n // 10)
        order = sorted(pct, key=lambda p: -pct[p])
        j = i % T
        prev = book[j]
        if prev is None:
            book[j] = list(order[:k])
        else:
            keep = [p for p in prev if p in pct and pct[p] >= 1 - EXIT_PCT][:k]
            add = [p for p in order if p not in set(keep)][:k - len(keep)]
            book[j] = keep + add
            if i >= T:
                ts.append((k - len(keep)) / k)
    return float(np.mean(ts)) if ts else np.nan


def main():
    log("读取价格与分数...")
    ret, adv = load_prices()
    by_day = load_scores()
    days = sorted(by_day)
    log(f"  {len(days)} 个信号日  [{days[0].date()}..{days[-1].date()}]")

    log("\n【原语一】边际边际衰减曲线 e(h)：前 10% 名字入选后第 h 天的超额（bp/日）")
    E = edge_profile(by_day, days, ret, adv)
    e1 = E[1][0]
    log("   h :  " + "  ".join(f"{h:>6d}" for h in sorted(E)))
    log("  bp :  " + "  ".join(f"{E[h][0]*1e4:>6.2f}" for h in sorted(E)))
    log("  ±SE:  " + "  ".join(f"{E[h][1]*1e4:>6.2f}" for h in sorted(E)))
    log("  相对:  " + "  ".join(f"{E[h][0]/e1:>6.2f}" for h in sorted(E)))
    hl = next((h for h in sorted(E) if E[h][0] <= e1 / 2), None)
    log(f"   -> 半衰期约 {hl} 个交易日（e(h) 首次跌破 e(1)/2）")

    log("\n【原语二】turn(T)：只跑换仓逻辑，不看收益")
    Ts = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    TU = {T: turnover(by_day, days, adv, T) for T in Ts}
    log("   T   :  " + "  ".join(f"{T:>6d}" for T in Ts))
    log("  turn :  " + "  ".join(f"{TU[T]:>6.3f}" for T in Ts))

    log("\n【解析】net(T) = 252 * [ mean(e(1..T)) - 2*bp*turn(T)/T ]   单位 %/年")
    res = {"edge_profile_bp": {h: E[h][0] * 1e4 for h in E},
           "edge_se_bp": {h: E[h][1] * 1e4 for h in E},
           "turnover": TU, "net_by_bp": {}}
    hdr = "   T  |" + "".join(f"  毛{'':>2}" if False else "" for _ in [0])
    log("   T  |    毛%  |" + "".join(f"   {bp}bp" for bp in (4, 8, 15, 25)) + "   | 最优?")
    best = {}
    for T in Ts:
        g = 252 * np.mean([E[h][0] for h in range(1, T + 1) if h in E]) * 100
        row = {}
        for bp in (4, 8, 15, 25):
            c = 252 * 2 * (bp / 1e4) * TU[T] / T * 100
            row[bp] = g - c
        res["net_by_bp"][T] = {"gross_pct": float(g),
                               **{f"net_{bp}bp": float(v) for bp, v in row.items()}}
        log(f"  {T:>3}  | {g:+7.2f} |" + "".join(f" {row[bp]:+7.2f}" for bp in (4, 8, 15, 25)))
    for bp in (4, 8, 15, 25):
        b = max(Ts, key=lambda T: res["net_by_bp"][T][f"net_{bp}bp"])
        best[bp] = b
    log("\n  ** 解析最优持仓期 T* ：" +
        "   ".join(f"{bp}bp -> T*={best[bp]}" for bp in (4, 8, 15, 25)) + " **")
    log(f"  ** 现行 T=6（在折 36-42 上网格搜出）；8bp 下解析最优 T*={best[8]} **")
    res["T_star"] = {str(k): int(v) for k, v in best.items()}
    (OUT / "optimal_holding.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/optimal_holding.json")


if __name__ == "__main__":
    main()
