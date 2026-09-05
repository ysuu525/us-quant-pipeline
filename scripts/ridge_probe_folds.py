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
    """保持表示为 float16（原样），float64 转换交给 RidgeFit 分块做。

    此前直接 astype(float64) 会把 110 万 x 512 的训练表示膨胀到 4.6GB，
    加上标准化后的副本峰值超过 10GB —— 首次七折跑批即在 fold40 处被内存打断。
    """
    keep = ((m["status"] == "ok") & m["label"].notna()).to_numpy()
    m2 = m[keep].reset_index(drop=True)
    y = m2.groupby("signal_date")["label"].rank(pct=True).to_numpy(np.float64)
    d = m2["signal_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    return E[keep], d, y


class RidgeFit:
    """分块预计算 X'X 与 X'y，之后每个 alpha 只需一次 512x512 求解。

    两处优化：
    - X'X 与 alpha 无关且占全部算力的 99%，原实现对每个 alpha 重算一遍（14 次冗余）；
    - 分块累加避免把 float64 的标准化矩阵整体materialize（峰值从 >10GB 降到约 0.4GB）。
    """

    CHUNK = 20_000

    def __init__(self, X, y, rows=None):
        """rows: 行索引（可选）。用它而不是 X[mask] 取子集——布尔取子集会整体
        复制（110 万 x 512 的 float16 就是 1.1GB），在提交限制紧张时直接炸。"""
        idx = np.arange(len(X)) if rows is None else np.asarray(rows)
        d = X.shape[1]
        n = len(idx)
        self.mu = np.zeros(d)
        acc2 = np.zeros(d)
        for i in range(0, n, self.CHUNK):
            blk = X[idx[i:i + self.CHUNK]].astype(np.float64)
            self.mu += blk.sum(0)
            acc2 += (blk * blk).sum(0)
            del blk
        self.mu /= n
        self.sd = np.sqrt(np.maximum(acc2 / n - self.mu ** 2, 0)) + 1e-8
        self.G = np.zeros((d, d))
        self.b = np.zeros(d)
        ym = y[idx].mean()
        for i in range(0, n, self.CHUNK):
            j = idx[i:i + self.CHUNK]
            Z = (X[j].astype(np.float64) - self.mu) / self.sd
            self.G += Z.T @ Z
            self.b += Z.T @ (y[j] - ym)
            del Z
        self.I = np.eye(d)

    def predict(self, Xva, alpha, rows=None):
        w = np.linalg.solve(self.G + alpha * self.I, self.b)
        idx = np.arange(len(Xva)) if rows is None else np.asarray(rows)
        out = np.empty(len(idx))
        for i in range(0, len(idx), self.CHUNK):
            j = idx[i:i + self.CHUNK]
            out[i:i + self.CHUNK] = (
                (Xva[j].astype(np.float64) - self.mu) / self.sd) @ w
        return out


def ridge(Xtr, ytr, Xva, alpha):
    return RidgeFit(Xtr, ytr).predict(Xva, alpha)


def daily_ic_series(pred, y, day):
    """逐日 RankIC 序列。与 daily_ic 完全同一套算法（同一份代码），
    只是把中间量交出来，供 --daily-ic-dir 落盘。默认不调用，不改任何读数。"""
    days, out = [], []
    for u in np.unique(day):
        k = day == u
        if k.sum() < 50:
            continue
        days.append(u)
        out.append(np.corrcoef(np.argsort(np.argsort(pred[k])).astype(float),
                               np.argsort(np.argsort(y[k])).astype(float))[0, 1])
    return np.asarray(days, dtype=np.int64), np.asarray(out, dtype=float)


def daily_ic(pred, y, day):
    _, v = daily_ic_series(pred, y, day)
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
    ap.add_argument("--alphas", default="1e6,1e7,3e7,1e8,3e8,1e9,1e10")
    ap.add_argument("--out-json", default="outputs/ridge_probe.json")
    ap.add_argument("--folds", default="", help="逗号分隔的折名；留空=全部。"
                    "用于把已缓存的折先跑出来，未缓存的等 GPU 空出来再补")
    ap.add_argument("--daily-ic-dir", default="",
                    help="可选：把**内层选定 alpha** 的外层逐日 RankIC 序列落盘到该"
                         "目录（每折一个 parquet）。留空=不落盘，输出与既有产物逐位"
                         "一致。仅供实验 8 汇总表算 NW(5) t 用，不参与选参。")
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

    want = {x.strip() for x in args.folds.split(",") if x.strip()}
    results = {}
    if Path(args.out_json).exists():          # 增量累积，便于分批补折
        results = {k: v for k, v in
                   json.loads(Path(args.out_json).read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
    for name, ts, te, vs, ve in FOLDS:
        if want and name not in want:
            continue
        log(f"\n===== {name}（pool={args.pool}）=====")
        lo = cal.shift(cal.snap_forward(pd.Timestamp(ts)), -(args.lookback + 30))
        hi = cal.dates[min(cal.index_of(cal.snap_back(pd.Timestamp(ve))) + 8, len(cal) - 1)]
        # 只在缓存缺失时才读面板：整表读入约 2.4GB，缓存命中时纯属浪费，
        # 且与并行的 GPU 任务叠加会触发 OOM（首轮七折跑批即因此在 fold40 中断）。
        cached = all((croot / f"{name}_{part}").with_suffix(sfx).exists()
                     for part in ("train", "val")
                     for sfx in (f".{args.pool}.npy", ".parquet"))
        if cached:
            adj = None
            log("  表示缓存命中，跳过面板读取")
        else:
            adj = pd.read_parquet(
                P / "panel_kronos_adj.parquet",
                filters=[("DlyCalDt", ">=", lo), ("DlyCalDt", "<=", hi)])
        m_tr, E_tr = block(P, cal, adj, uni_all, args.lookback, ts, te, tok, mdl,
                           args.pool, croot / f"{name}_train")
        m_va, E_va = block(P, cal, adj, uni_all, args.lookback, vs, ve, tok, mdl,
                           args.pool, croot / f"{name}_val")
        del adj
        Xtr, dtr, ytr = prep(m_tr, E_tr)
        Xva, dva, yva = prep(m_va, E_va)
        log(f"  样本 训练 {len(Xtr):,} / 验证 {len(Xva):,}")

        # alpha 必须在**内层**验证上选：训练窗尾部 inner_months 个月，
        # 与内层训练按 predict+1 日 purge。在外层验证窗上选 alpha 等于在考卷上
        # 调参，会系统性虚高（fold40 诊断即犯此错，此处修正）。
        anchors = m_tr.loc[((m_tr["status"] == "ok") & m_tr["label"].notna()).to_numpy(),
                           "signal_date"].reset_index(drop=True)
        inner_start = cal.snap_forward(anchors.max() - pd.DateOffset(months=6))
        inner_cut = cal.shift(inner_start, -7)
        d_in = (anchors >= inner_start).to_numpy()
        d_out = (anchors <= inner_cut).to_numpy()
        log(f"  内层选参窗 [{inner_start.date()}..{anchors.max().date()}]，"
            f"内层训练截至 {inner_cut.date()}")

        r_out, r_in = np.flatnonzero(d_out), np.flatnonzero(d_in)
        fit_in = RidgeFit(Xtr, ytr, rows=r_out)   # 内层选参用
        fit_out = RidgeFit(Xtr, ytr)              # 全训练窗，出外层读数
        picked, best_inner = None, -np.inf
        for a in alphas:
            ic_in, _, _ = daily_ic(fit_in.predict(Xtr, a, rows=r_in),
                                   ytr[r_in], dtr[r_in])
            ic_out, t_out, nd = daily_ic(fit_out.predict(Xva, a), yva, dva)
            mark = ""
            if ic_in > best_inner:
                best_inner, picked = ic_in, (a, ic_out, t_out, nd)
                mark = "  << 内层最优"
            log(f"  alpha={a:>8.0e}: 内层 {ic_in:+.5f} | 外层 {ic_out:+.5f} "
                f"(t {t_out:+.2f}){mark}")
        oracle = max(daily_ic(fit_out.predict(Xva, a), yva, dva)[0] for a in alphas)
        results[name] = {"alpha": picked[0], "rank_ic": picked[1], "t": picked[2],
                         "n_days": picked[3], "pool": args.pool,
                         "inner_rank_ic": float(best_inner),
                         "oracle_rank_ic": float(oracle)}
        log(f"  → 选定 alpha={picked[0]:.0e}：外层 RankIC {picked[1]:+.5f} "
            f"(t {picked[2]:+.2f})　[事后最优上界 {oracle:+.5f}]")
        if args.daily_ic_dir:
            ddir = Path(args.daily_ic_dir)
            ddir.mkdir(parents=True, exist_ok=True)
            dd, vv = daily_ic_series(fit_out.predict(Xva, picked[0]), yva, dva)
            pd.DataFrame({"signal_date": pd.to_datetime(dd, unit="D"),
                          "rank_ic": vv}).to_parquet(ddir / f"{name}.parquet",
                                                     index=False)
            log(f"  逐日 RankIC 序列已落盘 {ddir / (name + '.parquet')}（{len(vv)} 天）")
        del Xtr, Xva, E_tr, E_va, m_tr, m_va

    vals = [r["rank_ic"] for r in results.values()]
    orc = [r["oracle_rank_ic"] for r in results.values()]
    results["_summary"] = {"mean_rank_ic": float(np.mean(vals)),
                           "mean_oracle_rank_ic": float(np.mean(orc)),
                           "n_positive": int(sum(v > 0 for v in vals)),
                           "n_folds": len(vals), "pool": args.pool}
    Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    log(f"\n七折均值 RankIC = {np.mean(vals):+.5f}  正折数 {sum(v > 0 for v in vals)}/{len(vals)}")
    log("对照：生成式微调 lb90 七折 +0.02067；生成式零样本 三折 +0.02240")


if __name__ == "__main__":
    main()
