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


def quality_ok_mask(
    panel: pd.DataFrame,
    permno_col: str = "PERMNO",
    max_abs_ret: float = 0.5,
    stagnation_run: int = 5,
) -> pd.Series:
    """Kronos 论文 Appendix B 式质量过滤的行级近似（训练池消融 C 臂，2026-08-27）。

    行不合格（False）当：DlyVol == 0（illiquidity）；相邻行收盘价变动
    |close/prev − 1| > max_abs_ret（结构跳变；输入为复权后面板时拆股跳变
    已被消除，此处滤到的是数据异常与极端行情——后者也被滤是本过滤的已知
    代价）；或处于连续 ≥ stagnation_run 个相同 DlyClose 的停滞段内。

    要求 panel 已按 (PERMNO, 日期) 排序（snapshot.load_daily 的输出即是）。
    缺失值不在此判定（交给 valid_ohlc_mask）。窗口级排除由
    windows.build_window_index 的 extra_valid 参数完成。
    """
    close = panel["DlyClose"]
    pn = panel[permno_col]
    same_stock = pn.eq(pn.shift())

    ok = pd.Series(True, index=panel.index)
    if "DlyVol" in panel.columns:
        ok &= panel["DlyVol"].fillna(0) > 0

    ret = (close / close.shift() - 1).where(same_stock)
    ok &= ret.abs().fillna(0) <= max_abs_ret

    new_run = ~(close.eq(close.shift()) & same_stock)
    run_id = new_run.cumsum()
    run_size = run_id.groupby(run_id).transform("size")
    ok &= ~((run_size >= stagnation_run) & close.notna())
    return ok


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
