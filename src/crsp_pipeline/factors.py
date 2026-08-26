"""自建归因因子（规范 §1，v1.3：弃用 Fama-French 外部表，全链单一 CRSP 数据源）。

- 市场因子：``VWRETD``（评估口径统一为对 VWRETD 的超额，不引入无风险利率）；
- 规模因子代理：``DlyCap`` 十分位构造小减大组合；
- 动量因子：12-1 月收益构造赢减输组合；
- 价值因子 CRSP 无法自建，随 v2 基本面再加。

构造纪律：universe 过滤与 §2 同一套，**禁止另起口径**——本模块的因子构造
函数只接收调用方已用 ``universe.selection_panel`` 过滤后的行（in_universe
为 True 的观测），自身不再做任何过滤。分组特征一律用上一交易日的值
（禁止用 t 日价格既定分组又算 t 日收益）。

自建因子仅服务验证期归因诊断，非核心评估指标（§7.5 通过标准不依赖它）；
与学术 FF 口径不可直接对比，此局限记入文档。

另含中性化诊断所需的个股暴露（§7.5）：60 日已实现波动率、市场 beta。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import TradingCalendar


def market_factor(index_df: pd.DataFrame, ret_col: str = "vwretd",
                  date_col: str = "caldt") -> pd.Series:
    """VWRETD 日收益序列（index=日期）。"""
    s = index_df.set_index(pd.to_datetime(index_df[date_col]))[ret_col].astype(float)
    return s.sort_index()


# ---------------------------------------------------------------- 个股暴露

def rolling_realized_vol(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    window: int = 60,
    min_obs: int = 40,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    ret_col: str = "DlyRet",
) -> pd.DataFrame:
    """60 日已实现波动率（§7.5 中性化诊断用），日收益滚动标准差（不年化——
    只用于横截面回归，量纲不影响残差）。交易日历上滚动：缺行 = 无观测。"""
    return _rolling_per_permno(
        panel, calendar, permno_col, date_col, ret_col,
        lambda s: s.rolling(window, min_periods=min_obs).std(),
        out_col="vol",
    )


def rolling_beta(
    panel: pd.DataFrame,
    market: pd.Series,
    calendar: TradingCalendar,
    window: int = 252,
    min_obs: int = 126,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    ret_col: str = "DlyRet",
) -> pd.DataFrame:
    """市场 beta（对 VWRETD），滚动 cov/var。窗口/最小观测数为诊断层参数，
    冻结前在验证期内定死，不做逐股调优。"""

    def _beta(s: pd.Series) -> pd.Series:
        m = market.reindex(s.index)
        cov = s.rolling(window, min_periods=min_obs).cov(m)
        var = m.rolling(window, min_periods=min_obs).var()
        return cov / var

    return _rolling_per_permno(panel, calendar, permno_col, date_col, ret_col,
                               _beta, out_col="beta")


def _rolling_per_permno(panel, calendar, permno_col, date_col, ret_col, fn, out_col):
    df = panel[[permno_col, date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    frames = []
    for pn, g in df.groupby(permno_col):
        g = g.sort_values(date_col)
        sessions = calendar.sessions(g[date_col].iloc[0], g[date_col].iloc[-1])
        s = g.set_index(date_col)[ret_col].reindex(sessions)
        out = fn(s)
        frames.append(pd.DataFrame({
            permno_col: pn,
            date_col: g[date_col].to_numpy(),
            out_col: out.loc[g[date_col]].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 组合因子

def long_short_factor(
    universe_panel: pd.DataFrame,
    char_col: str,
    long_leg: str,
    n_groups: int = 10,
    lag_sessions: int = 1,
    min_names_per_leg: int = 5,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    ret_col: str = "DlyRet",
) -> pd.Series:
    """通用等权多空组合日收益：按上一观测日的特征分 n_groups 组，
    long_leg='low' 取最小组减最大组（规模：小减大），'high' 反之（动量：赢减输）。

    universe_panel 必须是已过 §2 过滤的观测（本函数不过滤）。特征用该股
    上一行观测（lag_sessions 期前）——分组信息严格先于收益。任一腿有效
    名字数 < min_names_per_leg 的日期返回 NaN。
    """
    if long_leg not in ("low", "high"):
        raise ValueError("long_leg must be 'low' or 'high'")
    df = universe_panel[[permno_col, date_col, char_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([permno_col, date_col])
    df["_char_lag"] = df.groupby(permno_col)[char_col].shift(lag_sessions)

    out = {}
    for t, g in df.groupby(date_col):
        g = g.dropna(subset=["_char_lag", ret_col])
        if len(g) < n_groups:
            out[t] = np.nan
            continue
        grp = pd.qcut(g["_char_lag"].rank(method="first"), n_groups,
                      labels=False, duplicates="drop")
        lo = g.loc[grp == 0, ret_col]
        hi = g.loc[grp == grp.max(), ret_col]
        if len(lo) < min_names_per_leg or len(hi) < min_names_per_leg:
            out[t] = np.nan
            continue
        spread = lo.mean() - hi.mean() if long_leg == "low" else hi.mean() - lo.mean()
        out[t] = spread
    return pd.Series(out).sort_index()


def size_factor(universe_panel: pd.DataFrame, cap_col: str = "DlyCap",
                **kw) -> pd.Series:
    """规模因子代理：DlyCap 十分位，小减大（等权）。"""
    return long_short_factor(universe_panel, char_col=cap_col, long_leg="low", **kw)


def momentum_char(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    formation_sessions: int = 252,
    skip_sessions: int = 21,
    min_obs: int = 126,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    ret_col: str = "DlyRet",
) -> pd.DataFrame:
    """12-1 动量特征：t 日的动量 = [t−formation+1, t−skip] 区间累计收益
    （log1p 求和后还原），跳过最近 skip_sessions 个交易日。窗口内有效收益
    观测 < min_obs 的记 NaN。交易日历上滚动，缺行 = 无观测。"""
    span = formation_sessions - skip_sessions

    def _mom(s: pd.Series) -> pd.Series:
        lg = np.log1p(s)
        cum = lg.rolling(span, min_periods=min_obs).sum().shift(skip_sessions)
        return np.expm1(cum)

    return _rolling_per_permno(panel, calendar, permno_col, date_col, ret_col,
                               _mom, out_col="mom")


def momentum_factor(universe_panel_with_mom: pd.DataFrame, **kw) -> pd.Series:
    """动量因子：12-1 特征十分位，赢减输（等权）。输入需已 merge 好
    momentum_char 的 ``mom`` 列且已过 §2 过滤。"""
    return long_short_factor(universe_panel_with_mom, char_col="mom",
                             long_leg="high", **kw)
