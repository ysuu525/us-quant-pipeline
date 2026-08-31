"""便宜版横截面上下文检验：给手工读出的分数加极少参数的截面上下文。

动机（ledger 2026-08-31 困境 F.1）：本项目比较过的所有方案都是**单资产模型**
（线性探针也只看一只股票的表示，只是损失是横截面的），"横截面模型更适合排序"
这一理论预期从未被检验。但从零训练的横截面 Transformer 在本项目的样本量下
（约 750 个有效独立日、信噪比 0.02）大概率过拟合——本项目实测"从我们数据估的
有效自由度"与表现单调负相关。

故先做代价极低的版本：**保留生成式的手工读出（零自由参数），只在其之上叠加
少量截面上下文特征，用 3–8 个参数的岭回归组合。** 若连这个都无增益，则百万
参数的横截面模型更无希望；若有明显增益，则"看到截面"确有价值，再投入大模型
才有依据。

截面上下文特征（每个都是"这只股票相对当天其他股票"的量，单资产模型看不到）：
  1. rel_industry  同行业内的分数相对位置（行业 = SIC 二位，缺失则用市值十分组）
  2. rel_size      同市值分组内的分数相对位置
  3. rel_adv       同流动性分组内的分数相对位置
  4. score_z       分数的当日截面 z-score（相对全市场的强弱，含尾部信息）
  5. disp_x_score  当日截面离散度 × 分数秩（离散度是收益的强解释变量）
  6. mkt_x_score   近期大盘状态 × 分数秩（regime 交互）
  7. rev5_rel      个股近 5 日收益相对当日截面中位数（短期反转的截面位置）

判读（先于结果写死）：以"仅用原始分数秩"为基线，
  加入上下文后七折均值提升 ≥ +0.003（约 +15%）→ 截面上下文有价值，值得投入大模型
  提升 +0.001 ~ +0.003                        → 边际，不足以支撑换架构
  提升 < +0.001 或为负                        → 假设证伪，横截面路线关闭
超参（岭 alpha）在**内层窗**选，不得在外层验证窗上挑。
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

from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402

P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
OUT = REPO_ROOT / "outputs"
FOLDS = ["fold36", "fold37", "fold38", "fold39", "fold40", "fold41", "fold42"]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def log(m):
    print(m, flush=True)


def daily_ic(pred, y, day):
    out = []
    for u in np.unique(day):
        k = day == u
        if k.sum() < 50:
            continue
        out.append(np.corrcoef(np.argsort(np.argsort(pred[k])).astype(float),
                               np.argsort(np.argsort(y[k])).astype(float))[0, 1])
    v = np.asarray(out, float)
    r = newey_west_tstat(pd.Series(v), 5)
    return float(r["mean"]), float(r["t"]), len(v)


def build_features(fold: str, ctx: pd.DataFrame) -> pd.DataFrame:
    d = OUT / f"{fold}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{fold}"
    sc = pd.read_parquet(d / "scores.parquet", columns=["PERMNO", "signal_date", "score"])
    lb = pd.read_parquet(d / "labels.parquet",
                         columns=["PERMNO", "signal_date", "status", "label"])
    m = sc.merge(lb, on=["PERMNO", "signal_date"])
    m = m[(m["status"] == "ok") & m["score"].notna() & m["label"].notna()]
    m["signal_date"] = pd.to_datetime(m["signal_date"])
    m = m.merge(ctx, on=["PERMNO", "signal_date"], how="left")
    del sc, lb

    g = m.groupby("signal_date")
    n = g["score"].transform("size")
    m["s_rank"] = g["score"].rank(pct=True) - 0.5                      # 基线特征
    m["score_z"] = (m["score"] - g["score"].transform("mean")) / (
        g["score"].transform("std") + 1e-12)
    m["score_z"] = m["score_z"].clip(-4, 4)
    # 组内相对位置：组 = 行业 / 市值十分组 / 流动性十分组
    for col, name in (("sic2", "rel_industry"), ("cap_g", "rel_size"), ("adv_g", "rel_adv")):
        grp = m.groupby(["signal_date", col])["score"]
        cnt = grp.transform("size")
        m[name] = np.where(cnt >= 5, grp.rank(pct=True) - 0.5, 0.0)
    # 截面离散度必须用**过去**的收益，不能用 label（label 是未来 6 日收益，
    # 用它构造特征是前视）。此处用当日 rev5（近 5 日收益）的截面标准差。
    disp = g["rev5"].transform("std")
    m["disp_x"] = ((disp - disp.mean()) / (disp.std() + 1e-12)).fillna(0.0) * m["s_rank"]
    m["mkt_x"] = m["mkt20"].fillna(0.0) * m["s_rank"]
    rev = m["rev5"].fillna(0.0)
    m["rev5_rel"] = (rev - g["rev5"].transform("median").fillna(0.0)).clip(-0.5, 0.5)
    m["n_names"] = n
    return m


def load_context(lo, hi):
    """行业码、市值/流动性分组、近 5 日个股收益、近 20 日大盘状态。

    **逐折调用、只读该折窗口**：整机提交上限常年吃紧（曾多次 OOM），
    一次性载入全部 3.5 年的面板会在 GPU 任务并行时炸掉。
    """
    raw = pd.read_parquet(
        P / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyClose", "DlyCap", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(lo) - pd.Timedelta(days=20)),
                 ("DlyCalDt", "<=", pd.Timestamp(hi))])
    raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"])
    raw = raw.sort_values(["PERMNO", "DlyCalDt"])
    raw["rev5"] = raw.groupby("PERMNO")["DlyClose"].transform(lambda s: s / s.shift(5) - 1.0)
    raw["cap_g"] = raw.groupby("DlyCalDt")["DlyCap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    raw["adv_g"] = raw.groupby("DlyCalDt")["DlyPrcVol"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
    ctx = raw[["PERMNO", "DlyCalDt", "rev5", "cap_g", "adv_g"]].rename(
        columns={"DlyCalDt": "signal_date"})
    del raw

    # 行业：security_info_history 是**带生效区间**的历史表，必须按日期匹配，
    # 用最新值会引入前视（公司改行业码、并购重组都会改变归属）。
    sic_path = P / "security_info_history.parquet"
    if sic_path.exists():
        try:
            si = pd.read_parquet(
                sic_path, columns=["permno", "secinfostartdt", "secinfoenddt", "siccd"])
            si = si.dropna(subset=["siccd"]).rename(columns={"permno": "PERMNO"})
            si["secinfostartdt"] = pd.to_datetime(
                si["secinfostartdt"]).astype("datetime64[ns]")
            si["secinfoenddt"] = pd.to_datetime(si["secinfoenddt"]).fillna(
                pd.Timestamp("2100-01-01")).astype("datetime64[ns]")
            si["sic2"] = si["siccd"].astype(str).str.zfill(4).str[:2]
            si = si.sort_values(["PERMNO", "secinfostartdt"])
            ctx["signal_date"] = ctx["signal_date"].astype("datetime64[ns]")
            ctx = ctx.sort_values(["signal_date", "PERMNO"])
            si = si.sort_values(["secinfostartdt", "PERMNO"])
            ctx = pd.merge_asof(
                ctx, si[["PERMNO", "secinfostartdt", "secinfoenddt", "sic2"]],
                left_on="signal_date", right_on="secinfostartdt", by="PERMNO",
                direction="backward")
            bad = ctx["secinfoenddt"].notna() & (ctx["signal_date"] > ctx["secinfoenddt"])
            ctx.loc[bad, "sic2"] = np.nan          # 区间已失效 → 视为缺失
            ctx = ctx.drop(columns=["secinfostartdt", "secinfoenddt"])
            log(f"  行业码：point-in-time 匹配 security_info_history，"
                f"覆盖 {ctx['sic2'].notna().mean():.1%}，{ctx['sic2'].nunique()} 个二位 SIC")
        except Exception as exc:
            log(f"  [warn] 读行业码失败：{exc}")
    if "sic2" not in ctx.columns:
        log("  未取到 SIC，rel_industry 回退为市值分组（与 rel_size 高度重合）")
        ctx["sic2"] = ctx["cap_g"]
    ctx["sic2"] = ctx["sic2"].fillna("NA").astype(str)

    idx = pd.read_parquet(P / "market_index.parquet", columns=["caldt", "vwretd"])
    idx["caldt"] = pd.to_datetime(idx["caldt"])
    idx = idx.sort_values("caldt")
    idx["mkt20"] = idx["vwretd"].rolling(20, min_periods=10).mean() * 100
    idx["mkt20"] = ((idx["mkt20"] - idx["mkt20"].mean()) / (idx["mkt20"].std() + 1e-12))
    ctx = ctx.merge(idx[["caldt", "mkt20"]].rename(columns={"caldt": "signal_date"}),
                    on="signal_date", how="left")
    return ctx


def ridge(Xtr, ytr, Xva, a):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + a * len(Z) * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
    return ((Xva - mu) / sd) @ w


CTX_COLS = ["s_rank", "score_z", "rel_industry", "rel_size", "rel_adv",
            "disp_x", "mkt_x", "rev5_rel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(OUT / "xsec_context_probe.json"))
    args = ap.parse_args()

    FOLD_WIN = {
        "fold36": ("2020-07-01", "2020-12-31"), "fold37": ("2021-01-04", "2021-06-30"),
        "fold38": ("2021-07-01", "2021-12-31"), "fold39": ("2022-01-03", "2022-06-30"),
        "fold40": ("2022-07-01", "2022-12-30"), "fold41": ("2023-01-03", "2023-06-30"),
        "fold42": ("2023-07-03", "2023-12-29"),
    }
    base_all, full_all, per_fold = [], [], {}
    cache = {}
    for fold in FOLDS:
        lo, hi = FOLD_WIN[fold]
        log(f"[{fold}] 读取该折窗口的截面上下文 [{lo}..{hi}] ...")
        ctx = load_context(lo, hi)
        m = build_features(fold, ctx).sort_values("signal_date").reset_index(drop=True)
        cache[fold] = {
            "X": m[CTX_COLS].fillna(0.0).to_numpy(np.float64),
            "y": m.groupby("signal_date")["label"].rank(pct=True).to_numpy(np.float64),
            "d": m["signal_date"].to_numpy().astype("datetime64[D]").astype(np.int64),
            "s": m["s_rank"].to_numpy(np.float64),
        }
        del ctx, m

    # 跨折滚动：第 k 折用前 k-1 折训练、整窗评估；alpha 用"留最后一训练折"选
    for k in range(2, len(FOLDS)):
        te = FOLDS[k]
        tr_folds = FOLDS[:k]
        Xtr = np.concatenate([cache[f]["X"] for f in tr_folds[:-1]])
        ytr = np.concatenate([cache[f]["y"] for f in tr_folds[:-1]])
        hold = tr_folds[-1]
        best_a, best_in = None, -np.inf
        for a in ALPHAS:
            ic_in, _, _ = daily_ic(ridge(Xtr, ytr, cache[hold]["X"], a),
                                   cache[hold]["y"], cache[hold]["d"])
            if ic_in > best_in:
                best_in, best_a = ic_in, a
        XA = np.concatenate([cache[f]["X"] for f in tr_folds])
        yA = np.concatenate([cache[f]["y"] for f in tr_folds])
        c = cache[te]
        full_ic, full_t, nd = daily_ic(ridge(XA, yA, c["X"], best_a), c["y"], c["d"])
        base_ic, base_t, _ = daily_ic(c["s"], c["y"], c["d"])
        per_fold[te] = {"base": float(base_ic), "with_ctx": float(full_ic),
                        "alpha": float(best_a), "n_days": nd,
                        "t_base": float(base_t), "t_ctx": float(full_t),
                        "train_folds": tr_folds}
        base_all.append(base_ic)
        full_all.append(full_ic)
        log(f"  {te}（训练 {tr_folds[0]}..{tr_folds[-1]}，选参留出 {hold}）: "
            f"仅分数 {base_ic:+.5f} → 加上下文 {full_ic:+.5f}  "
            f"（差 {full_ic-base_ic:+.5f}, alpha={best_a:g}, {nd} 天）")

    b, f = float(np.mean(base_all)), float(np.mean(full_all))
    log(f"\n七折均值：仅分数 {b:+.5f} → 加截面上下文 {f:+.5f}   提升 {f-b:+.5f}")
    n_pos = sum(1 for a, c in zip(base_all, full_all) if c > a)
    gate = n_pos >= int(np.ceil(0.75 * len(base_all)))   # 补：折数同向门槛
    verdict = ("截面上下文有价值，值得投入大模型" if (f - b >= 0.003 and gate) else
               "均值过线但折数不同向（离群折驱动），不构成证据" if f - b >= 0.003 else
               "边际，不足以支撑换架构" if f - b >= 0.001 else
               "假设证伪，横截面路线关闭")
    log(f"**判据（先于结果写死）→ {verdict}**")
    log(f"  提升在 {n_pos}/{len(base_all)} 折为正"
        f"（判据要求 ≥{int(np.ceil(0.75*len(base_all)))}）")
    Path(args.out_json).write_text(json.dumps(
        {"per_fold": per_fold, "mean_base": b, "mean_with_ctx": f,
         "delta": f - b, "verdict": verdict}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    log(f"写入 {args.out_json}")


if __name__ == "__main__":
    main()
