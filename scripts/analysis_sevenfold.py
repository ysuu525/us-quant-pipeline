"""七折口径的两项补齐（纯 CPU，内存按提交上限设计）。

D. **当前最佳设计的实盘读数**（七折 + 流动性过滤 + 省着换 + t+1 开盘建仓）：
   日频净值、年化、波动、夏普、最大回撤、最长水下、哑火统计。
   此前只在探针三折 / 全池上算过，而 §4.1 停机规则与交接文档都需要七折口径。

E. **E9 张成回归扩到七折**：策略日收益对 JKP 13 主题 + 市场回归，看 alpha 与
   t(alpha)。三折时 t(alpha) 全部 <2（功效不足），881 天应能给出更有判别力的读数。

内存约束：本机提交上限 55.9GB 常年吃紧（曾多次 OOM）。故一律列裁剪 + 日期
下推过滤读取，不整表载入。
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
J = Path(r"F:\quant\external\jkp")
OUT = REPO_ROOT / "outputs"
FOLDS = ["fold36", "fold37", "fold38", "fold39", "fold40", "fold41", "fold42"]
LO, HI = "2020-06-01", "2024-01-05"
EXIT_PCT, NT, TOPN, COST_BP = 0.30, 6, 500, 8.0


def log(m):
    print(m, flush=True)


def nw_ols(y, X, lags=5):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (A + A.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return b, np.where(se > 0, b / se, np.nan)


def load_prices():
    """列裁剪 + 日期下推：整表 49.8M 行，此处只取窗口内的 5 列。"""
    df = pd.read_parquet(
        P / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(LO)), ("DlyCalDt", "<=", pd.Timestamp(HI))])
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df.dropna(subset=["DlyRet"]).sort_values(["PERMNO", "DlyCalDt"])
    df["oc"] = np.where(df["DlyOpen"].abs() > 0, df["DlyClose"] / df["DlyOpen"].abs() - 1.0,
                        np.nan)
    df["adv20"] = df.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    ret = {d: dict(zip(g["PERMNO"], g["DlyRet"])) for d, g in df.groupby("DlyCalDt")}
    oc = {d: dict(zip(g["PERMNO"], g["oc"])) for d, g in df.groupby("DlyCalDt")}
    adv = {d: dict(zip(g["PERMNO"], g["adv20"])) for d, g in df.groupby("DlyCalDt")}
    del df
    return ret, oc, adv


def daily_returns(ret, oc, adv):
    """六档错位、进入前10%、跌出前30%才卖、t+1 开盘建仓、top-N 流动性过滤。"""
    rows = []
    for fold in FOLDS:
        d = OUT / f"{fold}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{fold}"
        sc = pd.read_parquet(d / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"]).dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        by_day = {day: dict(zip(g["PERMNO"], g["score"]))
                  for day, g in sc.groupby("signal_date") if len(g) >= 50}
        del sc
        days = sorted(by_day)
        book = {"L": [None] * NT, "S": [None] * NT}
        for i, day in enumerate(days):
            s = by_day[day]
            a = adv.get(day, {})
            elig = [p for p in s if p in a and np.isfinite(a[p])]
            if len(elig) > TOPN:
                elig = sorted(elig, key=lambda p: -a[p])[:TOPN]
            elig = set(elig)
            s = {p: v for p, v in s.items() if p in elig}
            n = len(s)
            if n < 50:
                continue
            pct = (pd.Series(s).rank() / n).to_dict()
            k = max(1, n // 10)
            order = sorted(pct, key=lambda p: -pct[p])
            j = i % NT
            turn, fresh = 0.0, {"L": set(), "S": set()}
            for side, seq, cond in (("L", order, lambda p: pct.get(p, 0.) >= 1 - EXIT_PCT),
                                    ("S", order[::-1], lambda p: pct.get(p, 1.) <= EXIT_PCT)):
                prev = book[side][j]
                if prev is None:
                    nb = list(seq[:k])
                    fresh[side] = set(nb)
                else:
                    keep = [p for p in prev if p in pct and cond(p)][:k]
                    held = set(keep)
                    add = [p for p in seq if p not in held][:k - len(keep)]
                    nb, fresh[side] = keep + add, set(add)
                    turn += (k - len(keep)) / k / 2.0
                book[side][j] = nb
            if i + 1 >= len(days) or i < NT:
                continue
            nd = days[i + 1]
            rm, om = ret.get(nd, {}), oc.get(nd, {})
            if not rm:
                continue
            cost = 2.0 * (COST_BP / 1e4) * turn / NT

            def leg(side):
                vals = []
                for t in range(NT):
                    nm = book[side][t]
                    if not nm:
                        continue
                    rs = [(om.get(p) if (t == j and p in fresh[side]) else rm.get(p))
                          for p in nm]
                    rs = [v for v in rs if v is not None and np.isfinite(v)]
                    if rs:
                        vals.append(np.mean(rs))
                return float(np.mean(vals)) if vals else 0.0

            rl, rs_ = leg("L"), leg("S")
            bench = float(np.mean([rm[p] for p in pct if p in rm])) if pct else 0.0
            rows.append((nd, rl - bench - cost, rl - rs_ - cost))
    return pd.DataFrame(rows, columns=["date", "long", "ls"]).set_index("date").sort_index()


def stats(r):
    eq = (1 + r).cumprod()
    n = len(r)
    ann = eq.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = eq / eq.cummax() - 1
    under = (dd < -1e-9)
    longest = cur = 0
    for v in under:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    m = r.groupby(r.index.to_period("M")).sum()
    neg = (m <= 0)
    lm = c = 0
    for v in neg:
        c = c + 1 if v else 0
        lm = max(lm, c)
    return {"ann": float(ann), "vol": float(vol),
            "sharpe": float(ann / vol) if vol else np.nan,
            "maxdd": float(dd.min()), "longest_underwater_days": int(longest),
            "months_neg_share": float(neg.mean()), "months_total": int(len(m)),
            "longest_neg_months": int(lm), "n_days": n}


def main():
    log("读取价格面板（列裁剪 + 日期下推）...")
    ret, oc, adv = load_prices()
    log("构造七折日频组合收益（top500 / 省着换 / t+1 开盘）...")
    r = daily_returns(ret, oc, adv)
    log(f"  组合日收益 {len(r)} 天  [{r.index.min().date()} .. {r.index.max().date()}]")

    res = {}
    log("\n=== D. 当前最佳设计的七折实盘读数（成本 8bp 单边）===")
    for col, nm in (("long", "纯多超额"), ("ls", "多空")):
        s = stats(r[col])
        res[nm] = s
        log(f"  {nm}: 年化 {s['ann']*100:+.2f}%  波动 {s['vol']*100:.2f}%  "
            f"夏普 {s['sharpe']:.2f}  最大回撤 {s['maxdd']*100:.2f}%")
        log(f"      最长水下 {s['longest_underwater_days']} 天  "
            f"月度非正 {s['months_neg_share']:.0%}（{s['months_total']} 个月）  "
            f"最长连续哑火 {s['longest_neg_months']} 个月")

    log("\n=== E. E9 张成回归（JKP 13 主题 + 市场，七折 881 天）===")
    th = pd.read_csv(J / "usa_all_themes_daily_vw_cap.csv", usecols=["name", "date", "ret"])
    th["date"] = pd.to_datetime(th["date"])
    th = th[(th["date"] >= r.index.min()) & (th["date"] <= r.index.max())]
    TH = th.pivot_table(index="date", columns="name", values="ret")
    mk = pd.read_csv(J / "usa_mkt_daily_vw_cap.csv")
    mk["date"] = pd.to_datetime(mk["date"])
    mk = mk.set_index("date")
    mcol = [c for c in mk.columns if c.lower() in ("ret", "mkt", "mktrf")]
    if mcol:
        TH["market"] = mk[mcol[0]]
    TH = TH.dropna(axis=1, how="all")
    for col, nm in (("long", "纯多超额"), ("ls", "多空")):
        df = pd.concat([r[col].rename("y"), TH], axis=1, sort=True).dropna()
        y = df["y"].to_numpy()
        F = df.drop(columns=["y"])
        X = np.column_stack([np.ones(len(df)), F.to_numpy()])
        b, t = nw_ols(y, X, 5)
        raw_ann = y.mean() * 252 * 100
        alpha_ann = b[0] * 252 * 100
        cors = F.apply(lambda c: np.corrcoef(c, y)[0, 1]).sort_values(key=abs, ascending=False)
        log(f"  {nm}（{len(df)} 天）: 原始年化 {raw_ann:+.2f}% → "
            f"alpha {alpha_ann:+.2f}%（保留 {100*alpha_ann/raw_ann:.0f}%）  "
            f"**t(alpha) = {t[0]:.2f}**")
        log("      最相关因子: " + "  ".join(f"{k}={v:+.3f}" for k, v in cors.head(4).items()))
        res[f"spanning_{nm}"] = {"raw_ann": raw_ann, "alpha_ann": alpha_ann,
                                 "t_alpha": float(t[0]), "n_days": int(len(df))}

    (OUT / "analysis_sevenfold.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n写入 outputs/analysis_sevenfold.json")


if __name__ == "__main__":
    main()
