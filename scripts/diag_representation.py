"""表示诊断：抽一次表示缓存，然后快速试多种池化/头/正则组合。

动机：首次"冻结主干+MLP头"在 fold40 只拿到 RankIC +0.0075（生成式基线 +0.0265），
训练损失平稳下降而内层 RankIC 纯噪声 → 疑似 (a) 取表示位置不当、(b) 头过拟合、
(c) 内层窗过短不足以早停。本脚本分离这三种可能。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.labels import compute_labels  # noqa: E402
from kronos_ft.models import load_pretrained  # noqa: E402
from kronos_ft.represent import extract_representations  # noqa: E402
from kronos_ft.windows import build_scoring_index, filter_index_by_universe  # noqa: E402


def log(m):
    print(m, flush=True)


def cache_block(P, cal, adj, uni, lookback, s, e, tok, mdl, pool, tag, cdir, predict=6):
    f_emb, f_idx = cdir / f"{tag}_{pool}.npy", cdir / f"{tag}.parquet"
    if f_emb.exists() and f_idx.exists():
        log(f"  复用缓存 {tag}_{pool}")
        return pd.read_parquet(f_idx), np.load(f_emb)
    sidx = build_scoring_index(adj, cal, lookback)
    sidx = sidx[(sidx["anchor"] >= s) & (sidx["anchor"] <= e)]
    sidx = filter_index_by_universe(sidx, uni).reset_index(drop=True)
    need = set(sidx["PERMNO"])
    ix, E = extract_representations(tok, mdl, adj[adj["PERMNO"].isin(need)], sidx, cal,
                                    lookback, batch_size=256, amp="bf16", pool=pool)
    if not f_idx.exists():
        lo, hi = adj["DlyCalDt"].min(), adj["DlyCalDt"].max()
        raw = pd.read_parquet(P / "panel_raw.parquet",
                              columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose",
                                       "DlyRet", "DlyDelFlg", "DlyCap", "DlyPrcVol"])
        raw = raw[(raw["DlyCalDt"] >= lo) & (raw["DlyCalDt"] <= hi)
                  & raw["PERMNO"].isin(need)]
        dist = pd.read_parquet(P / "distributions.parquet",
                               columns=["permno", "disexdt", "disdivamt"])
        cash = dist[dist["disdivamt"].fillna(0) > 0].rename(
            columns={"permno": "PERMNO", "disexdt": "ex_date", "disdivamt": "div_cash"})
        lab = compute_labels(raw, ix.rename(columns={"signal_date": "signal_date"}), cal,
                             cash_dividends=cash, predict_window=predict)
        m = ix.merge(lab[["PERMNO", "signal_date", "status", "label"]],
                     on=["PERMNO", "signal_date"], how="left")
        m.to_parquet(f_idx, index=False)
    np.save(f_emb, E)
    return pd.read_parquet(f_idx), E


def prep(m, E):
    keep = ((m["status"] == "ok") & m["label"].notna()).to_numpy()
    m2 = m[keep].reset_index(drop=True)
    y = m2.groupby("signal_date")["label"].rank(pct=True).to_numpy(np.float32)
    d = m2["signal_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    return E[keep], d, y


def daily_ic(pred, y, day):
    out = []
    for u in np.unique(day):
        k = day == u
        if k.sum() < 50:
            continue
        a = np.argsort(np.argsort(pred[k])).astype(float)
        b = np.argsort(np.argsort(y[k])).astype(float)
        out.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(out)), len(out)


def ridge_probe(Etr, ytr, Eva, alpha):
    """闭式岭回归线性探针：只有 513 个参数，几乎不可能过拟合。"""
    X = Etr.astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd
    y = ytr.astype(np.float64) - ytr.mean()
    A = X.T @ X + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ y)
    return ((Eva.astype(np.float64) - mu) / sd) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--backbone-dir", default="outputs/zeroshot_base")
    ap.add_argument("--cache-dir", default="outputs/repr_cache_fold40")
    ap.add_argument("--lookback", type=int, default=90)
    args = ap.parse_args()

    P = Path(args.processed)
    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    ts, te = pd.Timestamp("2019-07-01"), pd.Timestamp("2022-06-22")
    vs, ve = pd.Timestamp("2022-07-01"), pd.Timestamp("2022-12-30")

    cal = TradingCalendar.from_market_index(
        pd.read_parquet(P / "market_index.parquet"), "caldt")
    lo = cal.shift(cal.snap_forward(ts), -(args.lookback + 30))
    hi = cal.dates[min(cal.index_of(cal.snap_back(ve)) + 8, len(cal) - 1)]
    adj = pd.read_parquet(P / "panel_kronos_adj.parquet")
    adj = adj[(adj["DlyCalDt"] >= lo) & (adj["DlyCalDt"] <= hi)]
    uni = pd.read_parquet(P / "universe.parquet",
                          columns=["PERMNO", "DlyCalDt", "in_universe"])
    tok, mdl = load_pretrained(f"{args.backbone_dir}/tokenizer_final",
                               f"{args.backbone_dir}/predictor_final")

    for pool in ("last", "mean"):
        log(f"\n===== 池化方式 = {pool} =====")
        m_tr, E_tr = cache_block(P, cal, adj, uni, args.lookback, ts, te, tok, mdl,
                                 pool, "train", cdir)
        m_va, E_va = cache_block(P, cal, adj, uni, args.lookback, vs, ve, tok, mdl,
                                 pool, "val", cdir)
        Xtr, dtr, ytr = prep(m_tr, E_tr)
        Xva, dva, yva = prep(m_va, E_va)
        log(f"  样本 训练 {len(Xtr):,} / 验证 {len(Xva):,}")
        for alpha in (1e3, 1e4, 1e5, 1e6):
            p = ridge_probe(Xtr, ytr, Xva, alpha)
            ic, nd = daily_ic(p, yva, dva)
            log(f"  线性探针 alpha={alpha:>8.0e}: 验证窗 RankIC {ic:+.5f} ({nd} 天)")


if __name__ == "__main__":
    main()
