"""清洗规则（规范 §9，CIZ 语义）。

- 无负价格逻辑（那是 legacy SIZ）；报价中点由 ``DlyPrcFlg='BA'`` 标识，
  单独统计与处理；
- 停牌/缺失不删行、不插值、不压缩时间：个股在开市日缺失有效 OHLC 时该日
  留空；lookback 含缺口的训练样本**整体排除**（防止相隔数日的蜡烛被当作
  相邻），按年份 × 交易所报告排除率；
- 收益面板不受此规则影响（labels.py 自己处理缺失并报告）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import TradingCalendar

OHLC_COLS = ("DlyOpen", "DlyHigh", "DlyLow", "DlyClose")


def valid_ohlc_mask(panel: pd.DataFrame, require_volume: bool = True) -> pd.Series:
    """行级有效性：OHLC 全部非空且 >0（volume 允许为 0，但须非空）。"""
    m = pd.Series(True, index=panel.index)
    for c in OHLC_COLS:
        m &= panel[c].notna() & (panel[c] > 0)
    if require_volume and "DlyVol" in panel.columns:
        m &= panel["DlyVol"].notna()
    return m


def ba_flag_stats(
    panel: pd.DataFrame,
    date_col: str = "DlyCalDt",
    exch_col: str | None = None,
    flag_col: str = "DlyPrcFlg",
) -> pd.DataFrame:
    """``DlyPrcFlg='BA'``（买卖报价中点，非成交价）占比，按年（× 交易所）统计。"""
    df = panel.copy()
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df["is_ba"] = df[flag_col].astype(str).str.upper().eq("BA")
    keys = ["year"] + ([exch_col] if exch_col else [])
    g = df.groupby(keys)
    return pd.DataFrame({"n": g.size(), "n_ba": g["is_ba"].sum(), "ba_share": g["is_ba"].mean()}).reset_index()


def lookback_usable_mask(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    lookback: int,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    require_volume: bool = True,
) -> pd.DataFrame:
    """训练样本的 lookback 侧缺口排除（§9）。

    信号日 t 的样本可用，当且仅当 [t-lookback+1, t] 这 lookback 个**交易日**
    每天都存在该股的行且 OHLC(V) 有效——即窗口内连续无缺口。预测区间侧的
    缺口由标签引擎判定（status != ok 即排除），两者取交集得到最终训练样本。

    返回 (PERMNO, date, usable) 长表，仅含面板中实际存在的行。
    """
    df = panel[[permno_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_valid"] = valid_ohlc_mask(panel, require_volume=require_volume).to_numpy()

    frames = []
    for pn, g in df.groupby(permno_col):
        g = g.sort_values(date_col)
        sessions = calendar.sessions(g[date_col].iloc[0], g[date_col].iloc[-1])
        v = (
            g.set_index(date_col)["_valid"]
            .reindex(sessions)          # 缺行 -> NaN -> 无效观测（不压缩时间）
            .astype(float)
            .fillna(0.0)
        )
        ok = v.rolling(lookback, min_periods=lookback).sum() == lookback
        frames.append(
            pd.DataFrame({
                permno_col: pn,
                date_col: g[date_col].to_numpy(),
                "usable": ok.loc[g[date_col]].to_numpy(),
            })
        )
    return pd.concat(frames, ignore_index=True)


def exclusion_report(
    usable: pd.DataFrame,
    panel: pd.DataFrame,
    exch_col: str | None = "PrimaryExch",
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
) -> pd.DataFrame:
    """按年份 × 交易所报告排除率（§9）。exch_col 不在面板中则只按年。"""
    df = usable.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if exch_col and exch_col in panel.columns:
        df = df.merge(
            panel[[permno_col, date_col, exch_col]].assign(
                **{date_col: pd.to_datetime(panel[date_col])}
            ),
            on=[permno_col, date_col], how="left",
        )
        keys = [df[date_col].dt.year.rename("year"), df[exch_col]]
    else:
        keys = [df[date_col].dt.year.rename("year")]
    g = df.groupby(keys)
    return pd.DataFrame({
        "n_obs": g.size(),
        "n_excluded": g["usable"].apply(lambda s: int((~s).sum())),
        "exclusion_rate": g["usable"].apply(lambda s: float((~s).mean())),
    }).reset_index()
