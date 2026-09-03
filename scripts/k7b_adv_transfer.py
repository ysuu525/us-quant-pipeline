"""K7b：年代 × 流动性传输审计（判据见 ledger 的 K7a/K7b pre-registration 条）。

要回答的重构后的问题（来自 codex 复核）：
  全池 IC 跨年代只低 14%（+0.0178 vs +0.0207），top500 IC 低 55%（+0.0116 vs +0.0258）。
  top500 相对全池的增益从早期 -0.0062 变成近代 +0.0051，交互差 0.0113。
  => 更可能是「top500 能放大信号」这一关系只在近代成立，而不是基座整体不适配早期。

本脚本做四件事：
  A. **交互项的显著性**：逐日 Δ_t = IC(top500)_t − IC(全池)_t，两年代的 DiD +
     stationary block bootstrap CI。上面那个 0.0113 是 7 个折均值上眼看的，
     本项目今天已两次因此把噪声读成结构，故必须先装误差棒。
  B. **ADV 五档传输图**：每天按 point-in-time ADV20 分五档，逐档算 RankIC、
     Q10−档均值、十分位曲线；两年代对照 + 早期折数同向。
  C. **截面交互系数**：逐日 OLS  label ~ rank(score) + ADV_pct + rank(score)×ADV_pct，
     取交互系数的时序均值与 NW t。
  D. **|score| 是否在预测离散度而非方向**：逐日 Spearman(|score 去中位|, |label 去中位|)，
     用于解释 2003-04 十分位曲线「两端皆正」。

只用折 01-04 与折 36-42。不动折 05-35。
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
NQ, TOPN, B, MEAN_BLOCK, SEED = 5, 500, 5000, 10, 20260831

ERAS = {
    "2003-04": (["fold01", "fold02", "fold03", "fold04"], "2002-10-01", "2005-01-05",
                {"fold01": "eval_poolB_universe", "fold02": "eval_poolB_universe_fold02",
                 "fold03": "eval_poolB_universe_fold03", "fold04": "eval_poolB_universe_fold04"}),
    "2020H2-23": ([f"fold{f}" for f in range(36, 43)], "2020-04-01", "2024-01-05",
                  {f"fold{f}": f"eval_amp_lb90_fold{f}" for f in range(36, 43)}),
}


def log(m):
    print(m, flush=True)


def stationary_idx(n, rng, mb=MEAN_BLOCK):
    p = 1.0 / mb
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def nw_t(x, lags=5):
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    e = x - mu
    S = e @ e
    for L in range(1, lags + 1):
        S += 2 * (1 - L / (lags + 1.0)) * (e[L:] @ e[:-L])
    se = math.sqrt(max(S, 0)) / n
    return mu, (mu / se if se > 0 else np.nan)


def era_daily(folds, lo, hi, evd):
    """逐日指标表。"""
    p = pd.read_parquet(P / "panel_raw.parquet",
                        columns=["PERMNO", "DlyCalDt", "DlyPrcVol"],
                        filters=[("DlyCalDt", ">=", pd.Timestamp(lo)),
                                 ("DlyCalDt", "<=", pd.Timestamp(hi))])
    p["DlyCalDt"] = pd.to_datetime(p["DlyCalDt"])
    p = p.sort_values(["PERMNO", "DlyCalDt"])
    p["adv20"] = (p.groupby("PERMNO")["DlyPrcVol"].rolling(20, min_periods=10).mean()
                   .reset_index(level=0, drop=True)).groupby(p["PERMNO"]).shift(1)
    adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in p.groupby("DlyCalDt")}
    del p

    rows, dec = [], {q: [] for q in range(NQ)}
    for fold in folds:
        d = OUT / f"{fold}_lb90_s0_poolB_universe" / evd[fold]
        l = pd.read_parquet(d / "labels.parquet")
        s = pd.read_parquet(d / "scores.parquet")
        m = s.merge(l[["PERMNO", "signal_date", "label", "status"]],
                    on=["PERMNO", "signal_date"])
        m = m[(m.status == "ok") & m.label.notna() & m.score.notna()]
        m["signal_date"] = pd.to_datetime(m["signal_date"])
        for dt, g in m.groupby("signal_date"):
            if len(g) < 100:
                continue
            a = adv.get(dt, {})
            g = g.assign(_a=[a.get(x, np.nan) for x in g.PERMNO])
            g = g[np.isfinite(g._a)]
            if len(g) < 100:
                continue
            sr, lr = g.score.rank(), g.label.rank()
            ic_full = sr.corr(lr)
            g500 = g.nlargest(min(TOPN, len(g)), "_a")
            ic_500 = g500.score.rank().corr(g500.label.rank()) if len(g500) >= 50 else np.nan
            # 截面交互：label ~ rank(score) + ADV_pct + 交互
            x1 = (sr / len(g)).to_numpy() - 0.5
            x2 = g._a.rank(pct=True).to_numpy() - 0.5
            y = (lr / len(g)).to_numpy() - 0.5
            X = np.column_stack([np.ones(len(g)), x1, x2, x1 * x2])
            try:
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                inter = float(b[3])
            except np.linalg.LinAlgError:
                inter = np.nan
            # |score| 预测离散度？
            disp = pd.Series((g.score - g.score.median()).abs()).rank().corr(
                pd.Series((g.label - g.label.median()).abs()).rank())
            # ADV 五档
            q = pd.qcut(g._a.rank(method="first"), NQ, labels=False)
            per_q = {}
            for qi in range(NQ):
                gq = g[q == qi]
                if len(gq) < 40:
                    per_q[qi] = (np.nan, np.nan)
                    continue
                icq = gq.score.rank().corr(gq.label.rank())
                thr = gq.score.quantile(0.9)
                topq = gq.label[gq.score >= thr].mean() - gq.label.mean()
                per_q[qi] = (icq, topq)
                r = gq.score.rank(pct=True)
                mu = gq.label.mean()
                dec[qi].append([gq.label[(r > i / 10) & (r <= (i + 1) / 10)].mean() - mu
                                for i in range(10)])
            rows.append(dict(fold=fold, date=dt, ic_full=ic_full, ic_500=ic_500,
                             inter=inter, disp=disp,
                             **{f"ic_q{qi}": per_q[qi][0] for qi in range(NQ)},
                             **{f"top_q{qi}": per_q[qi][1] for qi in range(NQ)}))
    return pd.DataFrame(rows), {q: np.array(v) for q, v in dec.items()}


def main():
    rng = np.random.default_rng(SEED)
    D, DEC = {}, {}
    for era, (folds, lo, hi, evd) in ERAS.items():
        log(f"\n构造 {era} 的逐日指标表...")
        D[era], DEC[era] = era_daily(folds, lo, hi, evd)
        log(f"  {len(D[era])} 天")

    res = {}
    log("\n" + "=" * 78)
    log("A. 交互项 Δ = IC(top500) − IC(全池) 的显著性")
    log("=" * 78)
    log(f"  {'年代':<12}{'IC全池':>10}{'IC top500':>12}{'Δ':>10}{'NW t(Δ)':>10}")
    dd = {}
    for era in ERAS:
        d = D[era].dropna(subset=["ic_full", "ic_500"])
        delta = (d.ic_500 - d.ic_full).to_numpy()
        mu, t = nw_t(delta)
        dd[era] = delta
        log(f"  {era:<12}{d.ic_full.mean():>+10.5f}{d.ic_500.mean():>+12.5f}"
            f"{mu:>+10.5f}{t:>10.2f}")
    did = dd["2020H2-23"].mean() - dd["2003-04"].mean()
    bs = []
    for _ in range(B):
        a = dd["2003-04"][stationary_idx(len(dd["2003-04"]), rng)]
        b = dd["2020H2-23"][stationary_idx(len(dd["2020H2-23"]), rng)]
        bs.append(b.mean() - a.mean())
    bs = np.array(bs)
    lo_, hi_ = np.percentile(bs, [2.5, 97.5])
    log(f"\n  ** DiD（近代Δ − 早期Δ） = {did:+.5f}   95% CI [{lo_:+.5f}, {hi_:+.5f}]"
        f"   P(>0)={float((bs>0).mean()):.3f} **")
    log(f"  -> 交互效应{'显著' if lo_ > 0 else '不显著（零在 CI 内）'}")
    res["interaction_did"] = {"point": float(did), "ci95": [float(lo_), float(hi_)],
                              "p_gt0": float((bs > 0).mean())}

    log("\n" + "=" * 78)
    log("B. ADV 五档传输图（q0=最不流动 … q4=最流动）")
    log("=" * 78)
    res["adv_buckets"] = {}
    for era in ERAS:
        d = D[era]
        log(f"\n  [{era}]")
        log("    档位      RankIC   NW t   Q10−档均值   早期折同向")
        res["adv_buckets"][era] = {}
        for qi in range(NQ):
            ic = d[f"ic_q{qi}"].dropna().to_numpy()
            tp = d[f"top_q{qi}"].dropna().to_numpy()
            mu, t = nw_t(ic)
            byf = d.groupby("fold")[f"ic_q{qi}"].mean()
            pos = int((byf > 0).sum())
            log(f"    q{qi}     {mu:>+9.5f}{t:>7.2f}   {tp.mean()*1e4:>+8.1f}bp"
                f"      {pos}/{len(byf)}")
            res["adv_buckets"][era][f"q{qi}"] = {
                "rank_ic": float(mu), "nw_t": float(t),
                "top_minus_mean_bp": float(tp.mean() * 1e4),
                "folds_pos": pos, "n_folds": int(len(byf))}

    log("\n  逐档 IC 的年代差（近代 − 早期）：")
    for qi in range(NQ):
        a = res["adv_buckets"]["2003-04"][f"q{qi}"]["rank_ic"]
        b = res["adv_buckets"]["2020H2-23"][f"q{qi}"]["rank_ic"]
        log(f"    q{qi}: {a:+.5f} -> {b:+.5f}   差 {b-a:+.5f}")

    log("\n" + "=" * 78)
    log("C. 日截面交互系数  label ~ rank(score) + ADV_pct + rank(score)×ADV_pct")
    log("=" * 78)
    res["xsec_interaction"] = {}
    for era in ERAS:
        v = D[era].inter.dropna().to_numpy()
        mu, t = nw_t(v)
        log(f"  {era:<12} 交互系数均值 {mu:+.5f}   NW t = {t:+.2f}")
        res["xsec_interaction"][era] = {"mean": float(mu), "nw_t": float(t)}

    log("\n" + "=" * 78)
    log("D. |score| 是否在预测离散度而非方向")
    log("=" * 78)
    res["dispersion_test"] = {}
    for era in ERAS:
        v = D[era].disp.dropna().to_numpy()
        mu, t = nw_t(v)
        log(f"  {era:<12} Spearman(|score−中位|, |label−中位|) = {mu:+.5f}   NW t = {t:+.2f}")
        res["dispersion_test"][era] = {"mean": float(mu), "nw_t": float(t)}

    log("\n" + "=" * 78)
    log("附：各 ADV 档的十分位曲线（相对档均值，bp，最低分档在左）")
    log("=" * 78)
    for era in ERAS:
        log(f"\n  [{era}]")
        for qi in range(NQ):
            a = DEC[era][qi]
            if len(a) == 0:
                continue
            log(f"    q{qi}: " + " ".join(f"{np.nanmean(a[:, i])*1e4:>6.1f}" for i in range(10)))

    (OUT / "k7b_adv_transfer.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/k7b_adv_transfer.json")


if __name__ == "__main__":
    main()
