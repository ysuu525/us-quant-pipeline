"""七折岭回归线性探针：冻结表示里到底有没有横截面信息？

背景：首次"冻结主干 + MLP 头"在 fold40 只得 RankIC +0.0075（生成式基线 +0.0265），
训练损失平稳下降而内层 RankIC 纯噪声 → 疑似头过拟合（6.6 万参数 vs 约 750 个
有效独立日）。线性探针只有 513 个参数、闭式解、无早停、无超参搜索，
**几乎不可能过拟合**，因此能干净地回答"表示本身有没有信号"。

若探针在七折上普遍为正且量级接近生成式基线 → 表示可用，问题在头的容量/正则；
若普遍接近零 → 取表示的位置或池化方式不对，需要换 tap 点，而不是调头。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.calendar import TradingCalendar  # noqa: E402
from crsp_pipeline.labels import compute_labels  # noqa: E402
from kronos_ft.models import load_pretrained  # noqa: E402
from kronos_ft.represent import extract_representations  # noqa: E402
from kronos_ft.windows import build_scoring_index, filter_index_by_universe  # noqa: E402

FOLDS = [
    ("fold36", "2017-07-03", "2020-06-22", "2020-07-01", "2020-12-31"),
    ("fold37", "2018-01-02", "2020-12-22", "2021-01-04", "2021-06-30"),
    ("fold38", "2018-07-02", "2021-06-22", "2021-07-01", "2021-12-31"),
    ("fold39", "2019-01-02", "2021-12-22", "2022-01-03", "2022-06-30"),
    ("fold40", "2019-07-01", "2022-06-22", "2022-07-01", "2022-12-30"),
    ("fold41", "2020-01-02", "2022-12-21", "2023-01-03", "2023-06-30"),
    ("fold42", "2020-07-01", "2023-06-22", "2023-07-03", "2023-12-29"),
]


def log(m):
    print(m, flush=True)


def block(P, cal, adj, uni, lookback, s, e, tok, mdl, pool, cache, predict=6):
    f_emb, f_idx = cache.with_suffix(f".{pool}.npy"), cache.with_suffix(".parquet")
    if f_emb.exists() and f_idx.exists():
        return pd.read_parquet(f_idx), np.load(f_emb)
    sidx = build_scoring_index(adj, cal, lookback)
    sidx = sidx[(sidx["anchor"] >= pd.Timestamp(s)) & (sidx["anchor"] <= pd.Timestamp(e))]
    sidx = filter_index_by_universe(sidx, uni).reset_index(drop=True)
    need = set(sidx["PERMNO"])
    ix, E = extract_representations(tok, mdl, adj[adj["PERMNO"].isin(need)], sidx, cal,
                                    lookback, batch_size=256, amp="bf16", pool=pool)
    np.save(f_emb, E)
    if not f_idx.exists():
        raw = pd.read_parquet(P / "panel_raw.parquet",
                              columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose",
                                       "DlyRet", "DlyDelFlg", "DlyCap", "DlyPrcVol"])
        raw = raw[(raw["DlyCalDt"] >= adj["DlyCalDt"].min())
                  & (raw["DlyCalDt"] <= adj["DlyCalDt"].max())
                  & raw["PERMNO"].isin(need)]
        dist = pd.read_parquet(P / "distributions.parquet",
                               columns=["permno", "disexdt", "disdivamt"])
        cash = dist[dist["disdivamt"].fillna(0) > 0].rename(
            columns={"permno": "PERMNO", "disexdt": "ex_date", "disdivamt": "div_cash"})
        lab = compute_labels(raw, ix, cal, cash_dividends=cash, predict_window=predict)
        ix.merge(lab[["PERMNO", "signal_date", "status", "label"]],
                 on=["PERMNO", "signal_date"], how="left").to_parquet(f_idx, index=False)
    return pd.read_parquet(f_idx), E


def prep(m, E):
    keep = ((m["status"] == "ok") & m["label"].notna()).to_numpy()
    m2 = m[keep].reset_index(drop=True)
    y = m2.groupby("signal_date")["label"].rank(pct=True).to_numpy(np.float64)
    d = m2["signal_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    return E[keep].astype(np.float64), d, y


def ridge(Xtr, ytr, Xva, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = (Xtr - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (ytr - ytr.mean()))
    return ((Xva - mu) / sd) @ w


def daily_ic(pred, y, day):
    out = []
    for u in np.unique(day):
        k = day == u
        if k.sum() < 50:
            continue
        out.append(np.corrcoef(np.argsort(np.argsort(pred[k])).astype(float),
                               np.argsort(np.argsort(y[k])).astype(float))[0, 1])
    v = np.asarray(out, dtype=float)
    n = len(v)
    e = v - v.mean()
    var = float(e @ e) / n
    for L in range(1, 6):
        var += 2.0 * (1 - L / 6.0) * float(e[L:] @ e[:-L]) / n
    return float(v.mean()), float(v.mean() / np.sqrt(var / n)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--backbone-dir", default="outputs/zeroshot_base")
    ap.add_argument("--cache-root", default="outputs/repr_cache")
    ap.add_argument("--pool", default="mean", choices=["last", "mean"])
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--alphas", default="1e3,1e4,1e5,1e6")
    ap.add_argument("--out-json", default="outputs/ridge_probe.json")
    args = ap.parse_args()

    P = Path(args.processed)
    croot = Path(args.cache_root)
    croot.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    cal = TradingCalendar.from_market_index(
        pd.read_parquet(P / "market_index.parquet"), "caldt")
    tok, mdl = load_pretrained(f"{args.backbone_dir}/tokenizer_final",
                               f"{args.backbone_dir}/predictor_final")
    uni_all = pd.read_parquet(P / "universe.parquet",
                              columns=["PERMNO", "DlyCalDt", "in_universe"])

    results = {}
    for name, ts, te, vs, ve in FOLDS:
        log(f"\n===== {name}（pool={args.pool}）=====")
        lo = cal.shift(cal.snap_forward(pd.Timestamp(ts)), -(args.lookback + 30))
        hi = cal.dates[min(cal.index_of(cal.snap_back(pd.Timestamp(ve))) + 8, len(cal) - 1)]
        adj = pd.read_parquet(P / "panel_kronos_adj.parquet")
        adj = adj[(adj["DlyCalDt"] >= lo) & (adj["DlyCalDt"] <= hi)]
        m_tr, E_tr = block(P, cal, adj, uni_all, args.lookback, ts, te, tok, mdl,
                           args.pool, croot / f"{name}_train")
        m_va, E_va = block(P, cal, adj, uni_all, args.lookback, vs, ve, tok, mdl,
                           args.pool, croot / f"{name}_val")
        del adj
        Xtr, _, ytr = prep(m_tr, E_tr)
        Xva, dva, yva = prep(m_va, E_va)
        log(f"  样本 训练 {len(Xtr):,} / 验证 {len(Xva):,}")
        best = None
        for a in alphas:
            p = ridge(Xtr, ytr, Xva, a)
            ic, t, nd = daily_ic(p, yva, dva)
            log(f"  alpha={a:>8.0e}: RankIC {ic:+.5f}  t {t:+.2f}  ({nd} 天)")
            if best is None or ic > best[1]:
                best = (a, ic, t, nd)
        results[name] = {"alpha": best[0], "rank_ic": best[1], "t": best[2],
                         "n_days": best[3], "pool": args.pool}
        del Xtr, Xva, E_tr, E_va, m_tr, m_va

    vals = [r["rank_ic"] for r in results.values()]
    results["_summary"] = {"mean_rank_ic": float(np.mean(vals)),
                           "n_positive": int(sum(v > 0 for v in vals)),
                           "n_folds": len(vals), "pool": args.pool}
    Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    log(f"\n七折均值 RankIC = {np.mean(vals):+.5f}  正折数 {sum(v > 0 for v in vals)}/{len(vals)}")
    log("对照：生成式微调 lb90 七折 +0.02067；生成式零样本 三折 +0.02240")


if __name__ == "__main__":
    main()
