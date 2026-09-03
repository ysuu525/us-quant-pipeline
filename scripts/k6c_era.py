"""K6c：K6b 的跨年代对照（判据见 ledger 的 K6c pre-registration 条，先于运行落笔）。

相对 K6b 的三处改动：① 同时跑折01-04(2003-04) 与折36-42(2020H2-23) 两个年代；
② **策略腿改用毛收益（成本=0）**——2003 年真实成本未知，而它正是要排除的混淆项；
③ 逐年代报告 8 个控制因子自身的毛收益，看符号是否随年代翻转。

原 K6b 说明（保留）：
K6b：K6 + 短期动量控制（判据见 ledger 的 K6b pre-registration 条，先于运行落笔）。

相对 K6 的两处改动：① 控制集补 mom_skip(t-24..t-5) 与 mom_12_1(t-251..t-21)；
② 一致性门槛改为「固定合并载荷后的逐折残差 > 0 的折数 >= 5/7」。

背景
----
E9 已做过「策略 vs JKP 13 主题 + 市场」的张成回归，纯多超额 alpha 保留 89%。
但 **JKP 13 主题是月/年频、基本面导向的因子**，结构上张不成一个半衰期 2 天、
持有 6 天的价量信号——用它做控制集，89% 的保留率很可能是低功效检验的产物，
不足以支撑「这不是已知效应的翻版」这一主张。

K6 换用**同频同族**的控制集：1 日反转、5 日反转、Amihud 非流动性、成交量冲击、
特质波动、相对价差代理。若策略被这组因子张成，则它是已知短期价量效应的一个
实现而非新信号——按 de Groot(2012)/Nagel(2012) 的年代衰减先验，该族在 2009 年后
净收益基本消失，这与「折 01-04（2003-04）亏钱、折 36-42（2020H2-23，高 VIX 期）
赚钱」的观察是同一个故事。

判据（**先于任何结果写下**，2026-08-31）
--------------------------------------
主判据 = S1 规格下纯多超额的 **alpha 保留率**（alpha_ann / raw_ann）：

  * 保留 < 50%        → 判「已知短期价量效应的翻版」。触发重定位：撤回「新信号」
                        主张，改按已知效应族的年代衰减先验重估是否值得上线；
                        在此判读下不应动用确认集（折 05-35）。
  * 50% ≤ 保留 < 75%  → 判「部分重叠」。可继续，但报告与确认集判读必须以
                        **增量 alpha** 而非总收益为准，且须披露控制集构成。
  * 保留 ≥ 75%        → 判「非翻版」，「新信号」主张成立（与 JKP 口径的 89% 一致）。

一致性门槛（CLAUDE.md §一.3 强制，判据必须同时含量级与折数同向）：

  * 逐折单独回归，**alpha > 0 的折数 ≥ 5/7** 方可采信上面的量级判读；
  * < 5/7 → 一律降级为「不稳定，不可判定」，无论均值保留率多高。

t(alpha) 报告但 **不作判据**：832 天下夏普 0.74 的 t 天花板约 1.5（困境 B），
本检验没有功效去证伪 alpha=0，它只能回答「保留了多少」。

三个规格（**全部预先指定，全部报告，不得事后择优**）：

  S1 = 6 个核心控制因子 + 市场          （主判据依此）
  S2 = S1 + 1/价格                       （相对价差的第二代理）
  S3 = S1 + JKP 13 主题 + 市场           （最严口径，参考）

口径上的刻意从严
----------------
* 控制因子用**毛收益（成本 = 0）**，策略用**净收益（8bp）**——让控制集尽可能强、
  检验尽可能难。反过来做会人为抬高保留率。
* 控制因子走**与策略完全相同的组合管道**（top500 / 六档错位 / 进前 10% / 跌出
  前 30% 才卖 / t+1 开盘建仓），只把 score 换成因子值。这样控制掉「变现构造」这个
  共同因素，回归出的 alpha 才是「我的排序相对已知排序的增量」。
* 控制因子取**多空价差腿**而非纯多腿：该管道在分数取负时精确反号（进出场条件、
  档位顺序、成本项全部对称），故**因子方向不构成自由度**，alpha 对方向翻转不变。
  纯多腿没有这个性质，会引入「按哪个方向赚钱来定符号」的选择偏差。
* 控制因子每天的候选池 = 该折 scores.parquet 里当天有分的名字，与策略同池同日。

无前视核查（CLAUDE.md §一.1）
-----------------------------
本脚本 **不读取任何 label 文件**，全文无 `label` 变量。六个因子只用 signal_date
当天收盘及之前的量价：rev1/rev5 用到 t 日收益；amihud/hlrange 用 t-19..t；
volshock 用 t 日成交量与 t-20..t-1 的均量（均量已 shift(1) 排除当日）；
idiovol 用 t-59..t 的 CAPM 残差。收益侧（ret/oc）只在 t+1 上被使用，属结果不属特征。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

P = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
J = Path(r"F:\quant\external\jkp")
OUT = REPO_ROOT / "outputs"
ERAS = {
    "2003-04": (["fold01", "fold02", "fold03", "fold04"], "2002-01-01", "2005-01-05",
                {"fold01": "eval_poolB_universe",
                 "fold02": "eval_poolB_universe_fold02",
                 "fold03": "eval_poolB_universe_fold03",
                 "fold04": "eval_poolB_universe_fold04"}),
    "2020H2-23": ([f"fold{f}" for f in range(36, 43)], "2019-01-01", "2024-01-05",
                  {f"fold{f}": f"eval_amp_lb90_fold{f}" for f in range(36, 43)}),
}
FOLDS, LO, HI = None, None, None
EXIT_PCT, NT, TOPN, COST_BP = 0.30, 6, 500, 8.0

CORE = ["rev1", "rev5", "mom_skip", "mom_12_1",
        "amihud", "volshock", "idiovol", "hlrange"]
ALL_FAC = CORE + ["invprice"]
SPECS = {"S1": CORE, "S2": CORE + ["invprice"], "S3": CORE}  # S3 另加 JKP 主题


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


def load_panel(LO, HI):
    """列裁剪 + 日期下推。整表 4987 万行，此处只取窗口内的 9 列。"""
    df = pd.read_parquet(
        P / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
                 "DlyVol", "DlyPrcVol", "DlyRet"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(LO)), ("DlyCalDt", "<=", pd.Timestamp(HI))])
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = (df.dropna(subset=["DlyRet"])
            .sort_values(["PERMNO", "DlyCalDt"])
            .reset_index(drop=True))
    log(f"  面板 {len(df):,} 行 / {df['PERMNO'].nunique():,} 只 / "
        f"{df['DlyCalDt'].nunique()} 天")
    return df


def build_factors(df):
    """六(七)个控制因子，全部只用 signal_date 当日收盘及之前的信息。"""
    mk = pd.read_parquet(P / "market_index.parquet", columns=["caldt", "vwretd"])
    mk["caldt"] = pd.to_datetime(mk["caldt"])
    df = df.merge(mk.rename(columns={"caldt": "DlyCalDt"}), on="DlyCalDt", how="left")
    df = df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)
    gp = df["PERMNO"]

    def roll(s, w, mp):
        return (s.groupby(gp, sort=False).rolling(w, min_periods=mp)
                 .mean().reset_index(level=0, drop=True))

    df["oc"] = np.where(df["DlyOpen"].abs() > 0,
                        df["DlyClose"] / df["DlyOpen"].abs() - 1.0, np.nan)
    df["adv20"] = roll(df["DlyPrcVol"], 20, 10).groupby(gp, sort=False).shift(1)

    # --- 1 日 / 5 日反转（符号取「买输家」；多空腿对符号不敏感，见 docstring）
    df["rev1"] = -df["DlyRet"]
    lr = np.log1p(df["DlyRet"].clip(lower=-0.99))
    s5 = (lr.groupby(gp, sort=False).rolling(5, min_periods=5)
            .sum().reset_index(level=0, drop=True))
    df["rev5"] = -np.expm1(s5)

    # --- 短期动量（跳过最近一周，t-24..t-5）与经典 12-1 动量（t-251..t-21）
    #     两者都只用 signal_date 当日收盘及之前的数据；shift 保证跳窗
    def cumret(w, skip):
        c = (lr.groupby(gp, sort=False).rolling(w, min_periods=max(5, w // 2))
               .sum().reset_index(level=0, drop=True))
        return np.expm1(c.groupby(gp, sort=False).shift(skip))
    df["mom_skip"] = cumret(20, 5)
    df["mom_12_1"] = cumret(231, 21)

    # --- Amihud 非流动性：20 日均 |ret| / 美元成交额
    del lr
    illiq = (df["DlyRet"].abs() / df["DlyPrcVol"].where(df["DlyPrcVol"] > 0)) * 1e6
    df["amihud"] = roll(illiq, 20, 10)

    # --- 成交量冲击：log(当日量 / 前 20 日均量)；均量 shift(1) 排除当日
    v = df["DlyVol"].where(df["DlyVol"] > 0)
    base = roll(v, 20, 10).groupby(gp, sort=False).shift(1)
    df["volshock"] = np.log(v / base.where(base > 0))

    # --- 特质波动：60 日 CAPM 残差标准差（闭式，不做逐窗回归）
    r, m = df["DlyRet"], df["vwretd"]
    mr, mm = roll(r, 60, 40), roll(m, 60, 40)
    vr = roll(r * r, 60, 40) - mr ** 2
    vm = roll(m * m, 60, 40) - mm ** 2
    cv = roll(r * m, 60, 40) - mr * mm
    beta = cv / vm.where(vm > 0)
    df["idiovol"] = np.sqrt(np.maximum(vr - beta ** 2 * vm, 0.0))
    del mr, mm, vr, vm, cv, beta, r, m, s5, illiq, v, base

    # --- 相对价差代理：20 日均 (High-Low)/Close（CIZ 日线无 bid/ask，见限制）
    hl = ((df["DlyHigh"] - df["DlyLow"]) / df["DlyClose"].abs()).replace(
        [np.inf, -np.inf], np.nan)
    df["hlrange"] = roll(hl, 20, 10)

    # --- 1/价格（仅 S2）
    df["invprice"] = 1.0 / df["DlyClose"].abs().where(df["DlyClose"].abs() > 0)

    return df[["PERMNO", "DlyCalDt", "DlyRet", "oc", "adv20"] + ALL_FAC]


def run_pipeline(by_day, days, ret, oc, adv, cost_bp):
    """六档错位 / 进前 10% / 跌出前 30% 才卖 / t+1 开盘建仓 / top-N 流动性过滤。
    与 scripts/analysis_sevenfold.py 的 daily_returns() 构造逐行一致。"""
    book = {"L": [None] * NT, "S": [None] * NT}
    rows = []
    for i, day in enumerate(days):
        s = by_day.get(day)
        if not s:
            continue
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
        cost = 2.0 * (cost_bp / 1e4) * turn / NT

        def leg(side):
            vals = []
            for t in range(NT):
                nm = book[side][t]
                if not nm:
                    continue
                rs = [(om.get(p) if (t == j and p in fresh[side]) else rm.get(p))
                      for p in nm]
                rs = [x for x in rs if x is not None and np.isfinite(x)]
                if rs:
                    vals.append(np.mean(rs))
            return float(np.mean(vals)) if vals else 0.0

        rl, rs_ = leg("L"), leg("S")
        bench = float(np.mean([rm[p] for p in pct if p in rm])) if pct else 0.0
        rows.append((nd, rl - bench - cost, rl - rs_ - cost))
    return pd.DataFrame(rows, columns=["date", "long", "ls"]).set_index("date")


def run_era(era, folds, lo, hi, evdirs):
    log(f"\n{'='*74}\n### 年代 {era}  ({len(folds)} 折)\n{'='*74}")
    df = build_factors(load_panel(lo, hi))
    ret = {d: dict(zip(g["PERMNO"], g["DlyRet"])) for d, g in df.groupby("DlyCalDt")}
    oc = {d: dict(zip(g["PERMNO"], g["oc"])) for d, g in df.groupby("DlyCalDt")}
    adv = {d: dict(zip(g["PERMNO"], g["adv20"])) for d, g in df.groupby("DlyCalDt")}
    frames = {d: g for d, g in df.groupby("DlyCalDt")}
    del df
    strat, ctrl = [], {c: [] for c in ALL_FAC}
    for fold in folds:
        d = OUT / f"{fold}_lb90_s0_poolB_universe" / evdirs[fold]
        sc = pd.read_parquet(d / "scores.parquet",
                             columns=["PERMNO", "signal_date", "score"]).dropna()
        sc["signal_date"] = pd.to_datetime(sc["signal_date"])
        by_day = {day: dict(zip(g.PERMNO, g.score))
                  for day, g in sc.groupby("signal_date") if len(g) >= 50}
        del sc
        days = sorted(by_day)
        r = run_pipeline(by_day, days, ret, oc, adv, 0.0)   # 策略腿也用毛（成本=0）
        r["fold"] = fold
        strat.append(r)
        for c in ALL_FAC:
            fb = {}
            for day in days:
                f = frames.get(day)
                if f is None:
                    continue
                pn, vv = f["PERMNO"].to_numpy(), f[c].to_numpy()
                keep = by_day[day]
                fb[day] = {p: x for p, x in zip(pn, vv) if np.isfinite(x) and p in keep}
            ctrl[c].append(run_pipeline(fb, days, ret, oc, adv, 0.0)["ls"].rename(c))
        log(f"  {fold}: {len(r)} 天")
    S = pd.concat(strat).sort_index()
    C = pd.concat([pd.concat(ctrl[c]).sort_index() for c in ALL_FAC], axis=1)
    return S, C


def main():
    mk = pd.read_csv(J / "usa_mkt_daily_vw_cap.csv")
    mk["date"] = pd.to_datetime(mk["date"]); mk = mk.set_index("date")
    mcol = [c for c in mk.columns if c.lower() in ("ret", "mkt", "mktrf")][0]
    th = pd.read_csv(J / "usa_all_themes_daily_vw_cap.csv", usecols=["name", "date", "ret"])
    th["date"] = pd.to_datetime(th["date"])
    TH = th.pivot_table(index="date", columns="name", values="ret").dropna(axis=1, how="all")

    res = {"criteria": "见 ledger 的 K6c pre-registration 条", "eras": {}}
    keep = {}
    for era, (folds, lo, hi, ev) in ERAS.items():
        S, C = run_era(era, folds, lo, hi, ev)
        keep[era] = (S, C, folds)
        raw = S["long"].mean() * 252 * 100
        log(f"\n  策略毛纯多超额 = {raw:+.2f}%/年  ({len(S)} 天)")
        log("  控制因子自身（毛收益、多空价差腿）年化：")
        fs = {}
        for c in ALL_FAC:
            a = float(C[c].dropna().mean() * 252 * 100)
            fs[c] = a
            log(f"    {c:10s} {a:+7.2f}%")
        res["eras"][era] = {"strategy_gross_ann": float(raw), "n_days": int(len(S)),
                            "factor_gross_ann": fs}

    log(f"\n{'='*74}\n### 张成回归（两个年代同口径：策略毛 ~ 控制毛多空腿 + 市场）\n{'='*74}")
    for era, (S, C, folds) in keep.items():
        y_all = S["long"]
        res["eras"][era]["specs"] = {}
        for spec, cols, extra in (("S1b", CORE, []), ("S2b", CORE + ["invprice"], []),
                                  ("S3b", CORE, [TH])):
            F = pd.concat([C[cols], mk[mcol].rename("market")] + extra, axis=1, sort=True)
            d = pd.concat([y_all.rename("y"), F], axis=1, sort=True).dropna()
            if len(d) < 100:
                continue
            y = d["y"].to_numpy(); Fm = d.drop(columns=["y"])
            X = np.column_stack([np.ones(len(d)), Fm.to_numpy()])
            b, t = nw_ols(y, X, 5)
            raw = y.mean() * 252 * 100; al = b[0] * 252 * 100
            resid = y - Fm.to_numpy() @ b[1:]
            pos, pf = 0, {}
            for fold in folds:
                m = d.index.isin(S.index[S["fold"] == fold])
                if m.sum() < 40:
                    continue
                mu = float(resid[m].mean() * 252 * 100); pf[fold] = mu; pos += int(mu > 0)
            log(f"  [{era} · {spec}] {len(d)} 天 / {Fm.shape[1]} 回归元")
            log(f"     毛原始 {raw:+.2f}%  ->  alpha {al:+.2f}%  （保留 {100*al/raw if raw else float('nan'):.0f}%）"
                f"  t={t[0]:.2f}   逐折 alpha>0 = {pos}/{len(pf)}")
            log("     " + "  ".join(f"{k[-2:]}:{v:+.1f}" for k, v in pf.items()))
            bt = sorted(zip(Fm.columns, b[1:], t[1:]), key=lambda x: -abs(x[2]))
            log("     载荷: " + "  ".join(f"{k}={v:+.2f}(t{s:+.1f})" for k, v, s in bt[:5]))
            res["eras"][era]["specs"][spec] = {
                "raw_gross_ann": float(raw), "alpha_ann": float(al),
                "retention_pct": float(100*al/raw) if raw else None,
                "t_alpha": float(t[0]), "folds_pos": int(pos), "n_folds": len(pf),
                "per_fold_alpha": pf}

    log(f"\n{'='*74}\n### 辅判据：策略正载荷的四个因子是否随年代翻转符号\n{'='*74}")
    A, B = res["eras"]["2003-04"]["factor_gross_ann"], res["eras"]["2020H2-23"]["factor_gross_ann"]
    log(f"  {'因子':<12}{'2003-04':>12}{'2020H2-23':>12}   翻转?")
    flips = 0
    for c in ["mom_skip", "amihud", "idiovol", "hlrange"]:
        f = A[c] > 0 and B[c] < 0
        flips += f
        log(f"  {c:<12}{A[c]:>+11.2f}%{B[c]:>+11.2f}%   {'是' if f else '否'}")
    log(f"\n  ** 四个正载荷因子中，2003-04 转为赚钱的有 {flips}/4 **")
    res["sign_flips_of_4"] = int(flips)
    (OUT / "k6c_era.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    log("\n写入 outputs/k6c_era.json")


if __name__ == "__main__":
    main()
