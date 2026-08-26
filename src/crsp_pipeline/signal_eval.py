"""信号层评估（规范 §7.5）——横截面信号是否存在（全广度）。

- 对验证期内全部通过 §2 过滤的股票计算，不受仓位数约束；
- RankIC（原始为主，winsorized 附加 §9）；十分位组合 top−bottom 价差
  （等权，用 §4 execution-return 标签，扣 30bp 档成本）；
- IC 序列均值做 Newey-West t 检验；
- 中性化诊断：score 对市场 beta、60 日已实现波动率做横截面回归，报告残差
  RankIC。残差 IC 相对原始 IC 大幅衰减 → 按失败处理；
- 通过标准（冻结前写死，只许消耗验证期）：原始与残差 RankIC 验证期均值 > 0
  且 t > 2；十分位价差扣成本后 > 0。

统计检验只发生在信号层；执行层（execution_sim）只做可行性确认，不做显著性
声明。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- RankIC


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman 相关（平均秩处理并列），任一侧零方差返回 NaN。"""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    ra = df.iloc[:, 0].rank()
    rb = df.iloc[:, 1].rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_rank_ic(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "signal_date",
) -> pd.Series:
    """逐日横截面 RankIC。标签用真实值，不截尾（§9）。"""
    return df.groupby(pd.to_datetime(df[date_col])).apply(
        lambda g: _spearman(g[score_col], g[label_col])
    )


def winsorized_rank_ic(
    df: pd.DataFrame,
    limits: tuple[float, float] = (0.01, 0.99),
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "signal_date",
) -> pd.Series:
    """winsorized RankIC，仅作附加稳健性指标（§9）：标签按横截面分位数截边
    （尾部形成并列 → 平均秩），score 不动。"""

    def _one(g: pd.DataFrame) -> float:
        lo, hi = g[label_col].quantile(list(limits))
        return _spearman(g[score_col], g[label_col].clip(lo, hi))

    return df.groupby(pd.to_datetime(df[date_col])).apply(_one)


# ---------------------------------------------------------------- Newey-West


def newey_west_tstat(x: pd.Series, lags: int) -> dict:
    """IC 序列均值的 Newey-West t 检验（Bartlett 核）。

    lags 由调用方按重叠视野显式给定（6 日标签、日频信号 → 至少 5），
    不用自动带宽——冻结前写死。
    """
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 2:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "n": n}
    m = v.mean()
    e = v - m
    gamma0 = float(e @ e) / n
    var = gamma0
    for j in range(1, min(lags, n - 1) + 1):
        w = 1.0 - j / (lags + 1.0)
        var += 2.0 * w * float(e[j:] @ e[:-j]) / n
    se = np.sqrt(var / n)
    return {"mean": m, "se": se, "t": m / se if se > 0 else np.nan, "n": n}


# ---------------------------------------------------------------- 十分位价差


def decile_spread(
    df: pd.DataFrame,
    n_groups: int = 10,
    oneway_cost_bp: float = 30.0,
    cost_legs: float = 4.0,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "signal_date",
) -> pd.DataFrame:
    """逐期十分位 top−bottom 等权价差（标签 = §4 execution-return）。

    成本约定（冻结）：扣 30bp 档单边成本 × cost_legs=4（多头买+卖、空头
    卖+买，每期名单全换的悲观上界）。net = gross − oneway_cost_bp/1e4 × 4。
    名字数 < n_groups 的期返回 NaN。
    """
    cost = oneway_cost_bp / 1e4 * cost_legs

    def _one(g: pd.DataFrame) -> float:
        g = g.dropna(subset=[score_col, label_col])
        if len(g) < n_groups:
            return np.nan
        grp = pd.qcut(g[score_col].rank(method="first"), n_groups, labels=False)
        return g.loc[grp == n_groups - 1, label_col].mean() - g.loc[grp == 0, label_col].mean()

    gross = df.groupby(pd.to_datetime(df[date_col])).apply(_one).rename("gross")
    out = gross.to_frame()
    out["net"] = out["gross"] - cost
    return out


# ---------------------------------------------------------------- 中性化诊断


def neutralize_scores(
    df: pd.DataFrame,
    exposure_cols: tuple[str, ...] = ("beta", "vol"),
    score_col: str = "score",
    date_col: str = "signal_date",
) -> pd.Series:
    """score 对暴露（市场 beta、60 日已实现波动率）逐期横截面 OLS，返回残差
    （index 与 df 对齐）。暴露缺失的行残差为 NaN（不参与回归，也不冒充中性）。"""
    resid = pd.Series(np.nan, index=df.index)
    dates = pd.to_datetime(df[date_col])
    for _, g in df.groupby(dates):
        sub = g.dropna(subset=[score_col, *exposure_cols])
        if len(sub) <= len(exposure_cols) + 1:
            continue
        X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy(float) for c in exposure_cols])
        y = sub[score_col].to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid.loc[sub.index] = y - X @ coef
    return resid


# ---------------------------------------------------------------- 汇总与通过标准


def signal_layer_report(
    df: pd.DataFrame,
    nw_lags: int = 5,
    n_groups: int = 10,
    oneway_cost_bp: float = 30.0,
    attenuation_alarm: float = 0.5,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "signal_date",
) -> dict:
    """信号层完整报告 + 冻结的通过标准（§7.5）。

    输入：验证期全部 §2 过滤后观测，列 (signal_date, PERMNO, score, label,
    beta, vol)。label 来自 §4 标签引擎 status=='ok' 的行。

    通过标准（写死）：原始与残差 RankIC 均值 > 0 且 NW t > 2；十分位价差
    扣成本后均值 > 0。另报告残差/原始 IC 衰减比，低于 attenuation_alarm
    记警报（信号主要是波动率/beta 排序 → 按失败处理的依据）。
    """
    ic_raw = daily_rank_ic(df, score_col, label_col, date_col)
    ic_wins = winsorized_rank_ic(df, score_col=score_col, label_col=label_col, date_col=date_col)

    d2 = df.copy()
    d2["_resid"] = neutralize_scores(d2, score_col=score_col, date_col=date_col)
    ic_resid = daily_rank_ic(d2, "_resid", label_col, date_col)

    spread = decile_spread(df, n_groups=n_groups, oneway_cost_bp=oneway_cost_bp,
                           score_col=score_col, label_col=label_col, date_col=date_col)

    nw_raw = newey_west_tstat(ic_raw, nw_lags)
    nw_resid = newey_west_tstat(ic_resid, nw_lags)
    nw_spread_net = newey_west_tstat(spread["net"], nw_lags)

    atten = (nw_resid["mean"] / nw_raw["mean"]) if nw_raw["mean"] not in (0, np.nan) else np.nan

    passes = bool(
        nw_raw["mean"] > 0 and nw_raw["t"] > 2
        and nw_resid["mean"] > 0 and nw_resid["t"] > 2
        and np.nanmean(spread["net"]) > 0
    )
    return {
        "ic_raw": ic_raw, "ic_winsorized": ic_wins, "ic_residual": ic_resid,
        "spread": spread,
        "nw_raw": nw_raw, "nw_resid": nw_resid, "nw_spread_net": nw_spread_net,
        "attenuation_ratio": atten,
        "attenuation_alarm": bool(np.isnan(atten) or atten < attenuation_alarm),
        "passes": passes,
    }
