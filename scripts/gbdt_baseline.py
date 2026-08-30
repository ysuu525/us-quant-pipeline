"""LightGBM 基线（调研建议 #2）：回答"Kronos 有没有赢过一棵树"。

设计要点
- 特征：只用 CRSP 日线派生的价量特征（与 Kronos 的输入模态一致，公平对比）；
  每日做**截面秩变换**（0-1），这是调研里唯一有跨市场一致证据的处理。
- 标签：训练用"6 日前瞻收益的截面去均值"（调研建议：把大盘噪声从目标里删掉）；
  **评估用官方 execution-return 标签**（labels.parquet），与 Kronos 完全同口径。
- 折：与 Kronos 相同的 7 折近代窗口，训练窗同起止；池 = universe（同 B 臂）。
- 隔离：跑在 .venv-gbdt，不碰训练 venv（预注册环境冻结）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
OUT = Path(r"F:\quant\us-quant-pipeline\outputs")

FOLDS = [
    ("fold36", "2017-07-03", "2020-06-22", "2020-07-01", "2020-12-31"),
    ("fold37", "2018-01-02", "2020-12-22", "2021-01-04", "2021-06-30"),
    ("fold38", "2018-07-02", "2021-06-22", "2021-07-01", "2021-12-31"),
    ("fold39", "2019-01-02", "2021-12-22", "2022-01-03", "2022-06-30"),
    ("fold40", "2019-07-01", "2022-06-22", "2022-07-01", "2022-12-30"),
    ("fold41", "2020-01-02", "2022-12-21", "2023-01-03", "2023-06-30"),
    ("fold42", "2020-07-01", "2023-06-22", "2023-07-03", "2023-12-29"),
]
HORIZON = 6


def build_features(lo: str, hi: str) -> pd.DataFrame:
    cols = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
            "DlyVol", "DlyPrcVol", "DlyRet", "DlyCap"]
    df = pd.read_parquet(P / "panel_raw.parquet", columns=cols)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df[(df["DlyCalDt"] >= lo) & (df["DlyCalDt"] <= hi)]
    uni = pd.read_parquet(P / "universe.parquet", columns=["PERMNO", "DlyCalDt", "in_universe"])
    uni["DlyCalDt"] = pd.to_datetime(uni["DlyCalDt"])
    df = df.merge(uni[uni["in_universe"]][["PERMNO", "DlyCalDt"]], on=["PERMNO", "DlyCalDt"])
    del uni
    df = df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)
    g = df.groupby("PERMNO", sort=False)

    feat = {}
    for w in (1, 2, 3, 5, 10, 20, 60):
        feat[f"ret{w}"] = g["DlyClose"].transform(lambda s, w=w: s / s.shift(w) - 1.0)
    for w in (5, 10, 20, 60):
        feat[f"vol{w}"] = g["DlyRet"].transform(lambda s, w=w: s.rolling(w, min_periods=w // 2).std())
        feat[f"ma{w}"] = g["DlyClose"].transform(
            lambda s, w=w: s / s.rolling(w, min_periods=w // 2).mean() - 1.0)
        feat[f"vratio{w}"] = g["DlyVol"].transform(
            lambda s, w=w: s / s.rolling(w, min_periods=w // 2).mean().replace(0, np.nan))
    for w in (20, 60):
        hi_ = g["DlyHigh"].transform(lambda s, w=w: s.rolling(w, min_periods=w // 2).max())
        lo_ = g["DlyLow"].transform(lambda s, w=w: s.rolling(w, min_periods=w // 2).min())
        feat[f"pos{w}"] = (df["DlyClose"] - lo_) / (hi_ - lo_).replace(0, np.nan)
    feat["hl"] = (df["DlyHigh"] - df["DlyLow"]) / df["DlyClose"].abs().replace(0, np.nan)
    feat["co"] = df["DlyClose"] / df["DlyOpen"].abs().replace(0, np.nan) - 1.0
    feat["turn"] = df["DlyPrcVol"] / df["DlyCap"].replace(0, np.nan)
    feat["logcap"] = np.log(df["DlyCap"].clip(lower=1))
    feat["logadv"] = np.log(
        g["DlyPrcVol"].transform(lambda s: s.rolling(20, min_periods=10).mean()).clip(lower=1))
    # 前瞻 6 日收益（训练标签用，非官方 execution-return）
    fwd = g["DlyClose"].transform(lambda s: s.shift(-HORIZON) / s - 1.0)

    X = pd.DataFrame(feat)
    X["PERMNO"] = df["PERMNO"].to_numpy()
    X["date"] = df["DlyCalDt"].to_numpy()
    X["y_raw"] = fwd.to_numpy()
    del df, feat, g
    return X


def xsec_rank(X: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    """逐日截面秩变换到 [0,1]；标签逐日去均值（剔除大盘成分）。"""
    out = X.copy()
    grp = out.groupby("date", sort=False)
    for c in fcols:
        out[c] = grp[c].rank(pct=True)
    out["y"] = grp["y_raw"].transform(lambda s: s - s.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(OUT / "gbdt_baseline.json"))
    args = ap.parse_args()

    print("构建特征（2017-01 .. 2024-01）...", flush=True)
    X = build_features("2017-01-01", "2024-01-05")
    fcols = [c for c in X.columns if c not in ("PERMNO", "date", "y_raw")]
    X = xsec_rank(X, fcols)
    X = X.replace([np.inf, -np.inf], np.nan)
    print(f"特征表 {X.shape}，特征数 {len(fcols)}", flush=True)

    results = {}
    for name, ts, te, vs, ve in FOLDS:
        tr = X[(X["date"] >= ts) & (X["date"] <= te)].dropna(subset=["y"])
        va = X[(X["date"] >= vs) & (X["date"] <= ve)]
        if not len(tr) or not len(va):
            print(f"{name}: 数据不足，跳过"); continue
        m = lgb.train(
            dict(objective="regression", metric="l2", learning_rate=0.03,
                 num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbosity=-1, num_threads=8, seed=0),
            lgb.Dataset(tr[fcols].to_numpy(np.float32), label=tr["y"].to_numpy(np.float32)),
            num_boost_round=300)
        pred = m.predict(va[fcols].to_numpy(np.float32))
        sc = pd.DataFrame({"PERMNO": va["PERMNO"].to_numpy(),
                           "signal_date": va["date"].to_numpy(), "score": pred})
        # 用官方 execution-return 标签评估，与 Kronos 同口径
        lab = pd.read_parquet(
            OUT / f"{name}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{name}" / "labels.parquet",
            columns=["PERMNO", "signal_date", "label"])
        lab["signal_date"] = pd.to_datetime(lab["signal_date"])
        mm = sc.merge(lab, on=["PERMNO", "signal_date"]).dropna()
        ics = [np.corrcoef(g["score"].rank(), g["label"].rank())[0, 1]
               for _, g in mm.groupby("signal_date") if len(g) >= 50]
        ic = pd.Series(ics)
        v = ic.to_numpy()
        n = len(v)
        e = v - v.mean()
        var = float(e @ e) / n
        for L in range(1, 6):
            w = 1.0 - L / 6.0
            var += 2.0 * w * float(e[L:] @ e[:-L]) / n
        t = v.mean() / np.sqrt(var / n)
        results[name] = {"rank_ic": float(v.mean()), "t": float(t), "n_days": n,
                         "n_obs": int(len(mm))}
        print(f"{name}: RankIC={v.mean():+.5f} t={t:+.2f} ({n} 天, {len(mm):,} 观测)", flush=True)
        m.free_dataset()

    pooled = float(np.mean([r["rank_ic"] for r in results.values()]))
    results["_pooled_mean"] = pooled
    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n七折均值 RankIC = {pooled:+.5f}   （Kronos lb90 同口径 = +0.02067）")


if __name__ == "__main__":
    main()
