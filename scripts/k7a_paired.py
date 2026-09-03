"""K7a 的正确统计口径：逐日**配对**的 IC_ZS − IC_FT，而非折级推断。

为什么要重做
------------
初版判读用了折级 SD（n=4，SD=0.029 -> SE=0.0145 -> t=+0.34）并据此宣布
「功效不足、不可判读」。**这是错的**：零样本与微调是在**同日、同名字集**上打的分，
故 IC_ZS,t − IC_FT,t 是**配对差**，当日市场状态、截面离散度等公共成分在差分中相消。
折级 SD 同时包含了这些公共成分，因此系统性高估了配对量的不确定性。

本项目此前已经踩过同一个坑并有明确教训（2026-08-31 登记簿）：
lb90 vs lb200 按**水平**比较判「不可分」，按**配对**比较则 NW t=+1.94、95%CI 全体>0。

正确做法
--------
* 逐日在 **同一名字集**（两个分数文件的交集 ∩ 当日 top500 by ADV20）上分别算 RankIC；
* d_t = IC_ZS,t − IC_FT,t，对 d_t 做 Newey-West（6 日重叠标签 + 六档错位 -> lag=10）
  与 stationary block bootstrap（块长均值 10）；
* **折数只负责一致性判定，不用于构造标准误**；
* 同法处理 top-decile 毛超额差。

预注册阈值（ledger, 2026-08-31，未改）
  ① ΔIC >= +0.007   ② >=3/4 折同向   ③ Δtop-decile 毛 >= +4.8%/年
本脚本只更换**标准误的估计方式**，不改阈值。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
TOPN, B, MEAN_BLOCK, SEED, NWLAG = 500, 10000, 10, 20260831, 10

FT = {"fold01": "fold01_lb90_s0_poolB_universe/eval_poolB_universe",
      "fold02": "fold02_lb90_s0_poolB_universe/eval_poolB_universe_fold02",
      "fold03": "fold03_lb90_s0_poolB_universe/eval_poolB_universe_fold03",
      "fold04": "fold04_lb90_s0_poolB_universe/eval_poolB_universe_fold04"}
ZS = {f"fold0{i}": f"zeroshot_base/eval_zs_fold0{i}" for i in range(1, 5)}


def log(m):
    print(m, flush=True)


def nw(x, lags=NWLAG):
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    e = x - mu
    S = e @ e
    for L in range(1, lags + 1):
        S += 2 * (1 - L / (lags + 1.0)) * (e[L:] @ e[:-L])
    se = math.sqrt(max(S, 0)) / n
    return mu, se, (mu / se if se > 0 else np.nan)


def stat_idx(n, rng, mb=MEAN_BLOCK):
    p = 1.0 / mb
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def main():
    rng = np.random.default_rng(SEED)
    p = pd.read_parquet(P / "panel_raw.parquet",
                        columns=["PERMNO", "DlyCalDt", "DlyPrcVol"],
                        filters=[("DlyCalDt", ">=", pd.Timestamp("2002-10-01")),
                                 ("DlyCalDt", "<=", pd.Timestamp("2005-01-05"))])
    p["DlyCalDt"] = pd.to_datetime(p["DlyCalDt"])
    p = p.sort_values(["PERMNO", "DlyCalDt"])
    p["adv20"] = (p.groupby("PERMNO")["DlyPrcVol"].rolling(20, min_periods=10).mean()
                   .reset_index(level=0, drop=True)).groupby(p["PERMNO"]).shift(1)
    adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in p.groupby("DlyCalDt")}
    del p

    rows = []
    for f in FT:
        lab = pd.read_parquet(OUT / FT[f] / "labels.parquet")
        sf = pd.read_parquet(OUT / FT[f] / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"])
        sz = pd.read_parquet(OUT / ZS[f] / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"])
        m = (sf.rename(columns={"score": "ft"})
               .merge(sz.rename(columns={"score": "zs"}), on=["PERMNO", "signal_date"])
               .merge(lab[["PERMNO", "signal_date", "label", "status"]],
                      on=["PERMNO", "signal_date"]))
        m = m[(m.status == "ok") & m.label.notna() & m.ft.notna() & m.zs.notna()]
        m["signal_date"] = pd.to_datetime(m["signal_date"])
        for dt, g in m.groupby("signal_date"):
            a = adv.get(dt, {})
            g = g[[np.isfinite(a.get(x, np.nan)) for x in g.PERMNO]]
            if len(g) > TOPN:
                g = g.assign(_a=[a[x] for x in g.PERMNO]).nlargest(TOPN, "_a")
            if len(g) < 50:
                continue
            lr = g.label.rank()
            ic_z = g.zs.rank().corr(lr)
            ic_f = g.ft.rank().corr(lr)
            mu = g.label.mean()
            tz = g.label[g.zs >= g.zs.quantile(0.9)].mean() - mu
            tf = g.label[g.ft >= g.ft.quantile(0.9)].mean() - mu
            rows.append((f, dt, ic_z, ic_f, ic_z - ic_f, tz, tf, tz - tf, len(g)))
    D = pd.DataFrame(rows, columns=["fold", "date", "ic_zs", "ic_ft", "d_ic",
                                    "top_zs", "top_ft", "d_top", "n"])
    log(f"配对样本：{len(D)} 个交易日 / {D.fold.nunique()} 折  "
        f"（每日同名字集，中位 {int(D.n.median())} 只）")

    res = {}
    log("\n" + "=" * 80)
    log("逐日配对差（正确口径） vs 折级推断（初版的错误口径）")
    log("=" * 80)
    for col, nm, thr, scale in (("d_ic", "ΔIC (ZS−FT)", 0.007, 1.0),
                                ("d_top", "Δtop-decile 毛(年化%)", 4.8, 252 / 6 * 100)):
        v = D[col].to_numpy() * scale
        mu, se, t = nw(v)
        bs = np.array([v[stat_idx(len(v), rng)].mean() for _ in range(B)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        byf = D.groupby("fold")[col].mean() * scale
        se_fold = byf.std(ddof=1) / np.sqrt(len(byf))
        log(f"\n  {nm}")
        log(f"    逐日配对：均值 {mu:+.5f}  NW(lag={NWLAG}) SE {se:.5f}  **t = {t:+.2f}**")
        log(f"              bootstrap 95%CI [{lo:+.5f}, {hi:+.5f}]  "
            f"P(>0) = {float((bs>0).mean()):.3f}")
        log(f"    折级(n=4)：SE {se_fold:.5f}  t = {mu/se_fold:+.2f}   "
            f"<- 初版用的口径，SE 大 {se_fold/se:.1f} 倍")
        mde_d, mde_f = 1.96 * se, 2.35 * se_fold
        log(f"    最小可检出效应：逐日配对 {mde_d:.5f}  |  折级 {mde_f:.5f}   "
            f"（门槛 {thr}）")
        log(f"    -> 门槛 {'在' if mde_d <= thr else '不在'}逐日配对的可检出范围内"
            f"（折级口径下{'在' if mde_f <= thr else '不在'}）")
        res[col] = {"mean": float(mu), "nw_se": float(se), "nw_t": float(t),
                    "ci95": [float(lo), float(hi)], "p_gt0": float((bs > 0).mean()),
                    "fold_se": float(se_fold), "mde_paired": float(mde_d),
                    "mde_fold": float(mde_f), "threshold": thr,
                    "per_fold": {k: float(v) for k, v in byf.items()},
                    "folds_pos": int((byf > 0).sum())}

    log("\n" + "=" * 80)
    log("按预注册三条阈值重新判读（阈值未改，只换标准误口径）")
    log("=" * 80)
    a, b = res["d_ic"], res["d_top"]
    c1 = a["mean"] >= 0.007
    c2 = a["folds_pos"] >= 3
    c3 = b["mean"] >= 4.8
    for lbl, ok, txt in (("①", c1, f"ΔIC >= +0.007 ? 实测 {a['mean']:+.5f} "
                          f"(t={a['nw_t']:+.2f}, CI [{a['ci95'][0]:+.5f},{a['ci95'][1]:+.5f}])"),
                         ("②", c2, f">=3/4 折同向 ? 实测 {a['folds_pos']}/4"),
                         ("③", c3, f"Δtop-decile >= +4.8%/年 ? 实测 {b['mean']:+.2f}% "
                          f"(t={b['nw_t']:+.2f}, CI [{b['ci95'][0]:+.2f},{b['ci95'][1]:+.2f}])")):
        log(f"  {lbl} {txt}  -> {'通过' if ok else '不通过'}")
    log(f"\n  **{'零样本明显救回' if (c1 and c2 and c3) else '未达「明显救回」标准'}**")
    res["verdict"] = {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3),
                      "all": bool(c1 and c2 and c3)}
    (OUT / "k7a_paired.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/k7a_paired.json")


if __name__ == "__main__":
    main()
