"""给关键量装误差棒：832 天上的 stationary block bootstrap + Hansen SPA。

动机（2026-08-31）：本会话两次把噪声读成结构（eta 的比值假象、K6 的 4/7 一致性
门槛），共同机制都是 **在 n=7 上做推断，而手上有 832 天**。本脚本一次性替换掉
所有折级的眼看判断。

四个输出：
  A. eta（实测毛十分位价差 对 桥公式 3.51*IC*sigma 的无截距斜率）的 CI
  B. sigma_cs 时间趋势斜率的 CI（此前折级读数 -0.19pp/半年, t=-2.99）
  C. K6 三个规格 alpha 的 CI（此前只有 NW t）
  D. Hansen (2005) SPA_c：对**实际跑过的配置集**做多重检验，替代手数试验数的
     Bonferroni。SPA 不要求判断哪些试验「还算数」，也不像 Bonferroni 那样对
     高度相关的试验过度惩罚。

bootstrap：stationary bootstrap（Politis-Romano），几何块长均值 10 个交易日
（覆盖 6 日重叠标签 + 六档错位的持仓重叠）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
J = Path(r"F:\quant\external\jkp")
FOLDS = [f"fold{f}" for f in range(36, 43)]
B, MEAN_BLOCK, SEED = 5000, 10, 20260831


def _guard(p):
    """封存守卫：拒绝读取带 SEALED 哨兵的目录（2026-09-02 计算授权 != 读取授权）。"""
    import sys as _s
    from pathlib import Path as _P
    _r = _P(__file__).resolve().parents[1] / "src"
    if str(_r) not in _s.path:
        _s.path.insert(0, str(_r))
    from crsp_pipeline.sealed import assert_readable
    assert_readable(p)
    return p


def stationary_idx(n, rng, mean_block=MEAN_BLOCK):
    """Politis-Romano stationary bootstrap 的一组下标。"""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(n)
        else:
            i = (i + 1) % n
    return idx


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def daily_panel():
    """逐日 RankIC / sigma_cs / 毛十分位价差（lb90，全池，与 eta 诊断同口径）。"""
    rows = []
    for f in range(36, 43):
        d = OUT / f"fold{f}_lb90_s0_poolB_universe" / f"eval_amp_lb90_fold{f}"
        l = pd.read_parquet(d / "labels.parquet")
        s = pd.read_parquet(_guard(d) / "scores.parquet")
        m = s.merge(l[["PERMNO", "signal_date", "label", "status"]],
                    on=["PERMNO", "signal_date"])
        m = m[(m.status == "ok") & m.label.notna() & m.score.notna()]
        for dt, g in m.groupby("signal_date"):
            if len(g) < 50:
                continue
            ic = g.score.rank().corr(g.label.rank())
            sd = g.label.std()
            q = g.score.quantile([0.1, 0.9])
            sp = g.label[g.score >= q.loc[0.9]].mean() - g.label[g.score <= q.loc[0.1]].mean()
            rows.append((f"fold{f}", pd.Timestamp(dt), ic, sd, sp, 3.51 * ic * sd))
    return pd.DataFrame(rows, columns=["fold", "date", "ic", "sd", "spread", "pred"]) \
             .sort_values("date").reset_index(drop=True)


def discover_configs():
    """折 36-42 上都有逐日 IC 序列的配置（SPA 的试验全集）。"""
    per = {}
    for f in range(36, 43):
        for p in OUT.rglob("daily_ic.parquet"):
            if f"fold{f}" not in str(p):
                continue
            tag = p.parent.name
            key = (tag.replace(f"fold{f}", "").replace(f"_{f}", "")
                      .replace(str(f), "").strip("_"))
            per.setdefault(key, {})[f] = p
    return {k: v for k, v in per.items() if len(v) == 7}


def load_ic_series(paths):
    out = []
    for f in sorted(paths):
        df = pd.read_parquet(paths[f])
        c = [x for x in df.columns if "ic" in x.lower()]
        if not c:
            return None
        s = df.set_index(df.columns[0])[c[0]] if df.index.name is None else df[c[0]]
        s.index = pd.to_datetime(s.index)
        out.append(s)
    return pd.concat(out).sort_index()


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    D = daily_panel()
    n = len(D)
    print(f"逐日面板 {n} 天  [{D.date.min().date()}..{D.date.max().date()}]")
    pred, spr, sd = D.pred.to_numpy(), D.spread.to_numpy(), D.sd.to_numpy()
    tt = np.arange(n, dtype=float)
    tt = (tt - tt.mean()) / 252.0                     # 单位：年

    # ---- 预生成 bootstrap 下标（三项共用，保证可比）
    print(f"生成 {B} 组 stationary bootstrap 下标（均值块长 {MEAN_BLOCK}）...")
    IDX = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        IDX[b] = stationary_idx(n, rng)

    # ---- A. eta
    eta = float(pred @ spr / (pred @ pred))
    be = np.array([float(pred[i] @ spr[i] / (pred[i] @ pred[i])) for i in IDX])
    lo, hi = ci(be)
    print(f"\nA. eta（IC->毛十分位价差的乘子） = {eta:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"   eta=1 是否在 CI 内: {'是' if lo <= 1 <= hi else '否'}"
          f"   eta=0: {'在' if lo <= 0 <= hi else '不在'}")
    res["eta"] = {"point": eta, "ci95": [lo, hi], "contains_1": bool(lo <= 1 <= hi)}

    # ---- B. sigma_cs 的时间趋势
    def slope(y, x):
        xc = x - x.mean()
        return float(xc @ (y - y.mean()) / (xc @ xc))
    s0 = slope(sd, tt) * 100
    bs = np.array([slope(sd[i], tt[i]) * 100 for i in IDX])
    lo, hi = ci(bs)
    print(f"\nB. sigma_cs 年化趋势斜率 = {s0:+.3f} pp/年   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"   零在 CI 内: {'是（趋势不显著）' if lo <= 0 <= hi else '否（趋势显著）'}"
          f"   相对水平 {sd.mean()*100:.2f}% -> 每年 {s0/(sd.mean()*100)*100:+.1f}%")
    res["sigma_trend_pp_per_yr"] = {"point": s0, "ci95": [lo, hi]}

    # ---- C. K6 三规格 alpha
    S = pd.read_parquet(OUT / "k6_strategy_daily.parquet")
    C = pd.read_parquet(OUT / "k6_control_daily.parquet")
    mk = pd.read_csv(J / "usa_mkt_daily_vw_cap.csv")
    mk["date"] = pd.to_datetime(mk["date"]); mk = mk.set_index("date")
    mc = [c for c in mk.columns if c.lower() in ("ret", "mkt", "mktrf")][0]
    th = pd.read_csv(J / "usa_all_themes_daily_vw_cap.csv", usecols=["name", "date", "ret"])
    th["date"] = pd.to_datetime(th["date"])
    TH = th.pivot_table(index="date", columns="name", values="ret").dropna(axis=1, how="all")
    CORE6 = ["rev1", "rev5", "amihud", "volshock", "idiovol", "hlrange"]
    print("\nC. K6 三个规格的 alpha（年化 %）")
    res["k6_alpha"] = {}
    for spec, cols, extra in (("S1", CORE6, []), ("S2", CORE6 + ["invprice"], []),
                              ("S3", CORE6, [TH])):
        F = pd.concat([C[cols], mk[mc].rename("market")] + extra, axis=1, sort=True)
        d = pd.concat([S["long"].rename("y"), F], axis=1, sort=True).dropna()
        y = d["y"].to_numpy()
        X = np.column_stack([np.ones(len(d)), d.drop(columns=["y"]).to_numpy()])
        a0 = float(np.linalg.pinv(X.T @ X) @ (X.T @ y))[0] if False else \
             float((np.linalg.pinv(X.T @ X) @ (X.T @ y))[0]) * 252 * 100
        m2 = len(d)
        I2 = IDX[:, :m2] % m2
        ba = []
        for i in I2:
            Xi, yi = X[i], y[i]
            ba.append(float((np.linalg.pinv(Xi.T @ Xi) @ (Xi.T @ yi))[0]) * 252 * 100)
        ba = np.array(ba); lo, hi = ci(ba)
        raw = y.mean() * 252 * 100
        print(f"   {spec}: alpha {a0:+.2f}%  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"保留 {100*a0/raw:.0f}%  P(alpha>0)={float((ba>0).mean()):.3f}")
        res["k6_alpha"][spec] = {"point": a0, "ci95": [lo, hi],
                                 "p_gt0": float((ba > 0).mean())}

    # ---- D. Hansen SPA_c
    cfgs = discover_configs()
    print(f"\nD. Hansen SPA_c —— 折36-42 上七折齐全的配置：{len(cfgs)} 个")
    series = {}
    for k, paths in cfgs.items():
        s = load_ic_series(paths)
        if s is not None and len(s) > 700:
            series[k] = s
    if len(series) < 2:
        print("   可用配置不足 2 个，跳过 SPA（覆盖不足，需先补齐逐日 IC 产物）")
        res["spa"] = {"skipped": True, "n_configs": len(series)}
    else:
        M = pd.concat(series, axis=1).dropna()
        d = M.to_numpy(); nn, K = d.shape
        print(f"   参与 SPA 的配置 {K} 个 / 共同交易日 {nn}：{list(M.columns)}")
        mu = d.mean(0)
        I3 = IDX[:, :nn] % nn
        bm = np.array([d[i].mean(0) for i in I3])
        omega = bm.std(0, ddof=1) * np.sqrt(nn)
        omega = np.where(omega > 0, omega, np.inf)
        Z = np.sqrt(nn) * mu / omega
        T = max(Z.max(), 0.0)
        thr = -np.sqrt(2 * np.log(np.log(nn)))
        muc = np.where(Z >= thr, mu, 0.0)             # Hansen 的一致性再中心化
        Tb = np.maximum((np.sqrt(nn) * (bm - muc) / omega).max(1), 0.0)
        p = float((Tb >= T).mean())
        best = M.columns[int(np.argmax(Z))]
        print(f"   最优配置 = {best}（逐日 IC 均值 {mu.max():+.5f}，studentized Z={Z.max():.2f}）")
        print(f"   **SPA_c p 值 = {p:.4f}**（H0：所有配置的真实 IC <= 0）")
        res["spa"] = {"n_configs": int(K), "n_days": int(nn), "best": str(best),
                      "T_spa": float(T), "p_value": p,
                      "configs": [str(c) for c in M.columns],
                      "z": {str(c): float(z) for c, z in zip(M.columns, Z)}}

    (OUT / "bootstrap_errorbars.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n写入 outputs/bootstrap_errorbars.json")


if __name__ == "__main__":
    main()
