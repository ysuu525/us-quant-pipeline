"""通宵 CPU 分析三件套（不占 GPU，与 run_overnight.ps1 并行）。

A. 七折合成：lb90 + lb200 等权秩合成。此前只在探针三折上测过（较最优臂 +11%），
   而三折结论在本项目已两次被七折推翻，必须补齐。
B. E0 微观结构检验（调研报告排第一、一直未跑）：把建仓推迟 j 个交易日
   （j=0,1,2,3,5,10），看毛价差与 RankIC 掉多少。若 j=1 就损失 >40%，说明
   收益主要是买卖价差回弹，整条净价差讨论从一开始就建立在虚假毛收益上。
C. E8a 市值中性化（路径① 的否决项之一）：把分数对 log 市值做逐日截面回归取残差，
   重算 RankIC 与十分位价差。保留 <40% 即否决"这是真选股"的解读。

口径：近代 7 折、lb90/lb200 的 amp 统一分数、universe 内、等权。
B/C 用开盘到开盘的简单收益（可平移），**仅作组内相对比较**，
不与官方 execution-return 的绝对水平混用。
"""
from __future__ import annotations

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


def log(m):
    print(m, flush=True)


def load_arm(fold, lb):
    d = OUT / f"{fold}_lb{lb}_s0_poolB_universe" / f"eval_amp_lb{lb}_{fold}"
    s = pd.read_parquet(d / "scores.parquet", columns=["PERMNO", "signal_date", "score"])
    s["signal_date"] = pd.to_datetime(s["signal_date"])
    return s


def load_labels(fold):
    d = OUT / f"{fold}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{fold}"
    l = pd.read_parquet(d / "labels.parquet",
                        columns=["PERMNO", "signal_date", "status", "label"])
    l["signal_date"] = pd.to_datetime(l["signal_date"])
    return l[l["status"] == "ok"].drop(columns=["status"])


def rank_ic_series(df, score_col="score"):
    out = []
    for day, g in df.groupby("signal_date"):
        if len(g) < 50:
            continue
        out.append((day, np.corrcoef(g[score_col].rank(), g["label"].rank())[0, 1]))
    return pd.Series(dict(out)).sort_index()


def decile_gross(df, score_col="score"):
    out = []
    for day, g in df.groupby("signal_date"):
        n = len(g)
        if n < 50:
            continue
        k = max(1, n // 10)
        gs = g.sort_values(score_col)
        out.append((day, gs["label"].tail(k).mean() - gs["label"].head(k).mean()))
    return pd.Series(dict(out)).sort_index()


# ---------------------------------------------------------------- A 七折合成
def part_a():
    log("\n" + "=" * 62 + "\nA. 七折合成（lb90 + lb200 等权秩）\n" + "=" * 62)
    res = {}
    pooled = {k: [] for k in ("lb90", "lb200", "ens")}
    for fold in FOLDS:
        a, b, lab = load_arm(fold, 90), load_arm(fold, 200), load_labels(fold)
        m = a.merge(b, on=["PERMNO", "signal_date"], suffixes=("_a", "_b")).merge(
            lab, on=["PERMNO", "signal_date"]).dropna()
        per = {}
        for day, g in m.groupby("signal_date"):
            n = len(g)
            if n < 50:
                continue
            ra, rb = g["score_a"].rank() / n, g["score_b"].rank() / n
            yr = g["label"].rank()
            per.setdefault("lb90", []).append(np.corrcoef(ra, yr)[0, 1])
            per.setdefault("lb200", []).append(np.corrcoef(rb, yr)[0, 1])
            per.setdefault("ens", []).append(
                np.corrcoef((ra + rb).rank(), yr)[0, 1])
        for k in pooled:
            pooled[k] += per[k]
        res[fold] = {k: float(np.mean(v)) for k, v in per.items()}
        log(f"  {fold}: lb90 {res[fold]['lb90']:+.5f}  lb200 {res[fold]['lb200']:+.5f}  "
            f"合成 {res[fold]['ens']:+.5f}")
        del a, b, lab, m
    log("\n  -- 七折合并（NW lag=5）--")
    summary = {}
    base = None
    for k, nm in (("lb90", "lb90"), ("lb200", "lb200"), ("ens", "50/50 合成")):
        r = newey_west_tstat(pd.Series(pooled[k]), 5)
        summary[k] = {"rank_ic": r["mean"], "t": r["t"]}
        if k == "lb90":
            base = r["mean"]
        delta = "" if k == "lb90" else f"  (相对 lb90 {100 * (r['mean'] / base - 1):+.1f}%)"
        log(f"  {nm:>10}: RankIC {r['mean']:+.5f}  t {r['t']:.2f}{delta}")
    n_ens_best = sum(1 for f in FOLDS
                     if res[f]["ens"] >= max(res[f]["lb90"], res[f]["lb200"]))
    log(f"  合成在 {n_ens_best}/7 折为最优")
    log("  [对照] 探针三折曾测：lb90 +0.01500 / lb200 +0.02118 / 合成 +0.02358")
    return {"by_fold": res, "pooled": summary, "ens_best_folds": n_ens_best}


# ------------------------------------------------------- B E0 延迟建仓曲线
def part_b():
    log("\n" + "=" * 62 + "\nB. E0 微观结构检验：延迟建仓 j 日\n" + "=" * 62)
    raw = pd.read_parquet(P / "panel_raw.parquet",
                          columns=["PERMNO", "DlyCalDt", "DlyOpen"])
    raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"])
    raw = raw[(raw["DlyCalDt"] >= "2020-06-01") & (raw["DlyCalDt"] <= "2024-03-01")]
    raw["op"] = raw["DlyOpen"].abs()
    raw = raw[raw["op"] > 0].sort_values(["PERMNO", "DlyCalDt"])
    # 每只股票的开盘价序列 + 日期→行号
    px = {pn: g["op"].to_numpy(np.float64) for pn, g in raw.groupby("PERMNO")}
    ix = {pn: {d: i for i, d in enumerate(g["DlyCalDt"])}
          for pn, g in raw.groupby("PERMNO")}
    del raw

    res = {}
    for j in (0, 1, 2, 3, 5, 10):
        ics, sps = [], []
        for fold in FOLDS:
            sc, lab = load_arm(fold, 90), load_labels(fold)
            m = sc.merge(lab[["PERMNO", "signal_date"]], on=["PERMNO", "signal_date"])
            fwd = np.full(len(m), np.nan)
            pn_arr = m["PERMNO"].to_numpy()
            dt_arr = m["signal_date"].to_numpy()
            for i in range(len(m)):
                pn = pn_arr[i]
                pos = ix.get(pn, {}).get(pd.Timestamp(dt_arr[i]))
                if pos is None:
                    continue
                a, b = pos + 1 + j, pos + 6 + j
                arr = px[pn]
                if b < len(arr) and arr[a] > 0:
                    fwd[i] = arr[b] / arr[a] - 1.0
            m = m.assign(label=fwd).dropna(subset=["label", "score"])
            ics.append(rank_ic_series(m))
            sps.append(decile_gross(m))
            del sc, lab
        ic = pd.concat(ics).sort_index()
        sp = pd.concat(sps).sort_index()
        ric, rsp = newey_west_tstat(ic, 5), newey_west_tstat(sp, 5)
        res[j] = {"rank_ic": ric["mean"], "ic_t": ric["t"],
                  "gross_bp": rsp["mean"] * 1e4, "gross_t": rsp["t"]}
        log(f"  跳 {j:>2} 日: RankIC {ric['mean']:+.5f} (t {ric['t']:+.2f})   "
            f"毛价差 {rsp['mean'] * 1e4:+7.2f}bp (t {rsp['t']:+.2f})")
    b0 = res[0]["gross_bp"]
    if b0:
        log(f"\n  判读：跳 1 日保留 {100 * res[1]['gross_bp'] / b0:.0f}% 毛价差"
            f"（<60% 触发否决，≥75% 为绿）")
    return res


# -------------------------------------------------- C E8a 市值中性化
def part_c():
    log("\n" + "=" * 62 + "\nC. E8a 市值中性化\n" + "=" * 62)
    cap = pd.read_parquet(P / "panel_raw.parquet",
                          columns=["PERMNO", "DlyCalDt", "DlyCap"])
    cap["DlyCalDt"] = pd.to_datetime(cap["DlyCalDt"])
    cap = cap[(cap["DlyCalDt"] >= "2020-06-01") & (cap["DlyCalDt"] <= "2024-01-05")]
    cap = cap.rename(columns={"DlyCalDt": "signal_date"})
    cap["logcap"] = np.log(cap["DlyCap"].clip(lower=1.0))

    raw_ic, neu_ic, raw_sp, neu_sp = [], [], [], []
    for fold in FOLDS:
        sc, lab = load_arm(fold, 90), load_labels(fold)
        m = sc.merge(lab, on=["PERMNO", "signal_date"]).merge(
            cap[["PERMNO", "signal_date", "logcap"]], on=["PERMNO", "signal_date"],
            how="left").dropna(subset=["score", "label", "logcap"])
        # 逐日截面回归取残差
        def resid(g):
            x = g["logcap"].to_numpy()
            y = g["score"].to_numpy()
            x = (x - x.mean())
            beta = (x @ (y - y.mean())) / max(x @ x, 1e-12)
            return y - y.mean() - beta * x
        parts = []
        for day, g in m.groupby("signal_date"):
            if len(g) < 50:
                continue
            gg = g.copy()
            gg["score_neu"] = resid(g)
            parts.append(gg)
        mm = pd.concat(parts, ignore_index=True)
        raw_ic.append(rank_ic_series(mm, "score"))
        neu_ic.append(rank_ic_series(mm, "score_neu"))
        raw_sp.append(decile_gross(mm, "score"))
        neu_sp.append(decile_gross(mm, "score_neu"))
        del sc, lab, m, mm, parts
    out = {}
    for nm, a, b in (("RankIC", raw_ic, neu_ic), ("毛价差", raw_sp, neu_sp)):
        ra = newey_west_tstat(pd.concat(a).sort_index(), 5)
        rb = newey_west_tstat(pd.concat(b).sort_index(), 5)
        scale = 1e4 if nm == "毛价差" else 1.0
        keep = 100 * rb["mean"] / ra["mean"] if ra["mean"] else float("nan")
        unit = "bp" if nm == "毛价差" else ""
        log(f"  {nm}: 原始 {ra['mean'] * scale:+.4f}{unit} (t {ra['t']:+.2f}) → "
            f"中性化后 {rb['mean'] * scale:+.4f}{unit} (t {rb['t']:+.2f})  保留 {keep:.0f}%")
        out[nm] = {"raw": ra["mean"], "neutral": rb["mean"], "keep_pct": keep,
                   "raw_t": ra["t"], "neutral_t": rb["t"]}
    log("  判读：保留 ≥60% 为绿，<40% 否决（调研报告判据①c）")
    return out


if __name__ == "__main__":
    result = {}
    for name, fn in (("A_ensemble_7fold", part_a), ("B_e0_skip_days", part_b),
                     ("C_e8a_cap_neutral", part_c)):
        try:
            result[name] = fn()
        except Exception as exc:  # 单项失败不拖垮整批
            log(f"!! {name} 失败: {exc}")
            result[name] = {"error": str(exc)}
    (OUT / "analysis_overnight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/analysis_overnight.json")
