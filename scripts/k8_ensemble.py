"""K8：零样本 + 微调的等权秩合成（判据见 ledger 的 K8 pre-registration 条）。

动机：逐日分数秩相关实测 2003-04 +0.056 / 2020H2-23 **-0.045**，而分数信度约 0.75
——两者是**真正正交**的信号，却有相当的 RankIC（零样本 +0.01919 / 微调 +0.02071）。
两个 IC≈0.02、相关≈0 的信号等权合成，理论 IC = (ρ1+ρ2)/sqrt(2(1+c)) ≈ 0.0282（+36%）。

此前被否决的「合成方案」测的是 lb90+lb200（两者均为微调版、高度相关，仅 +3.1%），
**ZS+FT 的合成从未测过**。

判据（先于运行落笔）
  ① 配对 ΔIC（合成 − lb90 微调）>= +0.004 且 bootstrap 95%CI 下界 > 0
  ② >= 5/7 折同向
  ③ top-decile 减池均值 >= lb90 单臂的 90%（形状不被破坏）
MDE 前置检查：近代 881 天同类配对量 NW SE 约 0.003 -> MDE 约 0.006；预期效应 +0.0075。

合成方式：逐日在同名字集上各自转**百分位秩**后等权相加。零自由参数。
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
TOPN, B, MEAN_BLOCK, SEED, NWLAG = 500, 10000, 10, 20260831, 5

MODERN = [(f"fold{f}", f"fold{f}_lb90_s0_poolB_universe/eval_amp_lb90_fold{f}",
           f"zeroshot_base/eval_zeroshot_fold{f}") for f in range(36, 43)]
EARLY = [("fold01", "fold01_lb90_s0_poolB_universe/eval_poolB_universe",
          "zeroshot_base/eval_zs_fold01"),
         ("fold02", "fold02_lb90_s0_poolB_universe/eval_poolB_universe_fold02",
          "zeroshot_base/eval_zs_fold02"),
         ("fold03", "fold03_lb90_s0_poolB_universe/eval_poolB_universe_fold03",
          "zeroshot_base/eval_zs_fold03"),
         ("fold04", "fold04_lb90_s0_poolB_universe/eval_poolB_universe_fold04",
          "zeroshot_base/eval_zs_fold04")]


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


def build_adv(lo, hi):
    p = pd.read_parquet(P / "panel_raw.parquet",
                        columns=["PERMNO", "DlyCalDt", "DlyPrcVol"],
                        filters=[("DlyCalDt", ">=", pd.Timestamp(lo)),
                                 ("DlyCalDt", "<=", pd.Timestamp(hi))])
    p["DlyCalDt"] = pd.to_datetime(p["DlyCalDt"])
    p = p.sort_values(["PERMNO", "DlyCalDt"])
    p["adv20"] = (p.groupby("PERMNO")["DlyPrcVol"].rolling(20, min_periods=10).mean()
                   .reset_index(level=0, drop=True)).groupby(p["PERMNO"]).shift(1)
    return {d: dict(zip(g.PERMNO, g.adv20)) for d, g in p.groupby("DlyCalDt")}


def run(pairs, adv, label):
    rows = []
    for f, ftd, zsd in pairs:
        lab = pd.read_parquet(OUT / ftd / "labels.parquet")
        sf = pd.read_parquet(OUT / ftd / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"])
        sz = pd.read_parquet(OUT / zsd / "scores.parquet",
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
            comb = g.ft.rank(pct=True) + g.zs.rank(pct=True)     # 等权秩合成
            lr = g.label.rank()
            mu = g.label.mean()
            def topx(s):
                return g.label[s >= s.quantile(0.9)].mean() - mu
            rows.append((f, dt,
                         g.ft.rank().corr(lr), g.zs.rank().corr(lr), comb.rank().corr(lr),
                         topx(g.ft), topx(g.zs), topx(comb),
                         g.ft.rank().corr(g.zs.rank())))
    D = pd.DataFrame(rows, columns=["fold", "date", "ic_ft", "ic_zs", "ic_cb",
                                    "top_ft", "top_zs", "top_cb", "scorr"])
    D["d_ic"] = D.ic_cb - D.ic_ft
    log(f"\n{'='*78}\n{label}：{len(D)} 天 / {D.fold.nunique()} 折"
        f"   逐日分数秩相关中位 {D.scorr.median():+.3f}\n{'='*78}")
    log(f"  RankIC   微调 {D.ic_ft.mean():+.5f}   零样本 {D.ic_zs.mean():+.5f}   "
        f"**合成 {D.ic_cb.mean():+.5f}**   （合成/微调 {D.ic_cb.mean()/D.ic_ft.mean():.2f}×）")
    th = (0.01919 + 0.02071) / math.sqrt(2 * (1 + D.scorr.median()))
    log(f"  理论预期（(ρ1+ρ2)/sqrt(2(1+c))，用实测 c={D.scorr.median():+.3f}）= {th:+.5f}")
    log(f"  top-decile 减池均值(bp)  微调 {D.top_ft.mean()*1e4:.1f}  "
        f"零样本 {D.top_zs.mean()*1e4:.1f}  合成 {D.top_cb.mean()*1e4:.1f}   "
        f"（合成/微调 {D.top_cb.mean()/D.top_ft.mean():.2f}×）")
    return D


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for pairs, lo, hi, lbl, key in (
            (MODERN, "2020-04-01", "2024-01-05", "近代七折（主判据）", "modern"),
            (EARLY, "2002-10-01", "2005-01-05", "折01-04（附带报告）", "early")):
        D = run(pairs, build_adv(lo, hi), lbl)
        v = D.d_ic.to_numpy()
        mu, se, t = nw(v)
        bs = np.array([v[stat_idx(len(v), rng)].mean() for _ in range(B)])
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        byf = D.groupby("fold").d_ic.mean()
        pos = int((byf > 0).sum())
        shape = D.top_cb.mean() / D.top_ft.mean()
        log(f"\n  配对 ΔIC（合成 − 微调）= {mu:+.5f}   NW(lag={NWLAG}) SE {se:.5f}   t = {t:+.2f}")
        log(f"    bootstrap 95%CI [{lo_:+.5f}, {hi_:+.5f}]   P(>0) = {float((bs>0).mean()):.3f}")
        log(f"    逐折 ΔIC>0: {pos}/{len(byf)}   " +
            "  ".join(f"{k[-2:]}:{x:+.4f}" for k, x in byf.items()))
        res[key] = {"n_days": int(len(D)), "ic_ft": float(D.ic_ft.mean()),
                    "ic_zs": float(D.ic_zs.mean()), "ic_cb": float(D.ic_cb.mean()),
                    "score_corr_median": float(D.scorr.median()),
                    "d_ic": float(mu), "nw_se": float(se), "nw_t": float(t),
                    "ci95": [float(lo_), float(hi_)], "p_gt0": float((bs > 0).mean()),
                    "folds_pos": pos, "n_folds": int(len(byf)),
                    "shape_ratio": float(shape),
                    "per_fold": {k: float(x) for k, x in byf.items()}}
        if key == "modern":
            c1 = (mu >= 0.004) and (lo_ > 0)
            c2 = pos >= 5
            c3 = shape >= 0.90
            log(f"\n  按预注册判据：")
            log(f"    ① ΔIC>=+0.004 且 CI 下界>0 ? {mu:+.5f} / {lo_:+.5f}  -> {'通过' if c1 else '不通过'}")
            log(f"    ② >=5/7 折同向 ? {pos}/7  -> {'通过' if c2 else '不通过'}")
            log(f"    ③ 形状保留 >=90% ? {shape:.2f}×  -> {'通过' if c3 else '不通过'}")
            v = ("合成有效，进入冻结清单候选" if (c1 and c2 and c3) else
                 "IC 有效但需查形状" if (c1 and c2) else "不成立")
            log(f"    **判读：{v}**")
            res["verdict"] = {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "text": v}
    (OUT / "k8_ensemble.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/k8_ensemble.json")


if __name__ == "__main__":
    main()
