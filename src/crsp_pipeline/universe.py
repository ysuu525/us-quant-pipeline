"""Universe 筛选 —— 选股面板（规范 §2 / §3）。

静态 CIZ 普通股筛选（冻结，写死）：

    ShareType='NS' AND SecurityType='EQTY' AND SecuritySubType='COM'
    AND USIncFlg='Y' AND IssuerType IN ('ACOR','CORP')
    AND PrimaryExch IN ('N','A','Q') AND ConditionalType='RW'
    AND TradingStatusFlg='A'

流动性条件（全部按 t 日信息滚动，禁止回填）：

- 有效 DlyClose ≥ $5；
- ADV20_t = mean(DlyPrcVol[t-19:t])，窗口内 ≥15 个有效观测，ADV20_t ≥ $5m；
- 上市 ≥ 120 个交易日（非自然日）；
- DlyCap 排名前 1500（按 t 日值，在通过静态筛选且 DlyCap 有效的股票中排名）。

本模块只回答「t 日能否入选」（选股面板）；持有期收益一律走 labels.py
的未过滤全量面板（§3 双面板分离）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import TradingCalendar

# 冻结的静态筛选条件（§2）。键 = crsp.stksecurityinfohist 列名（小写）。
STATIC_FILTER: dict[str, tuple[str, ...]] = {
    "sharetype": ("NS",),
    "securitytype": ("EQTY",),
    "securitysubtype": ("COM",),
    "usincflg": ("Y",),
    "issuertype": ("ACOR", "CORP"),
    "primaryexch": ("N", "A", "Q"),
    "conditionaltype": ("RW",),
    "tradingstatusflg": ("A",),
}


def static_eligible_intervals(
    info_hist: pd.DataFrame,
    permno_col: str = "permno",
    beg_col: str = "secinfostartdt",
    end_col: str = "secinfoenddt",
) -> pd.DataFrame:
    """从证券属性历史表筛出满足静态条件的 (permno, 起, 止) 区间。

    end 列可能是 '9999-12-31' 这类远期哨兵或空（至今有效）——下载器原样保存
    了字符串；这里统一解析：解析失败或缺失 → 置为 pd.Timestamp.max 语义
    （用 2262-04-11 之前的安全上界表示「至今」）。
    """
    df = info_hist.copy()
    df.columns = [c.lower() for c in df.columns]
    mask = pd.Series(True, index=df.index)
    for col, allowed in STATIC_FILTER.items():
        if col not in df.columns:
            raise KeyError(f"info_hist missing required column: {col}")
        mask &= df[col].isin(allowed)
    out = df.loc[mask, [permno_col, beg_col, end_col]].copy()
    out[beg_col] = pd.to_datetime(out[beg_col])
    far_future = pd.Timestamp("2200-01-01")
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce").fillna(far_future)
    out.loc[out[end_col] > far_future, end_col] = far_future
    return out.rename(columns={permno_col: "PERMNO", beg_col: "beg", end_col: "end"})


def static_eligible_mask(
    panel: pd.DataFrame,
    intervals: pd.DataFrame,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
) -> pd.Series:
    """面板每行 (PERMNO, t) 是否落在某个合格区间内。"""
    key = panel[[permno_col, date_col]].copy()
    key[date_col] = pd.to_datetime(key[date_col])
    key["_row"] = np.arange(len(key))
    merged = key.merge(intervals, on=permno_col, how="left")
    hit = (merged[date_col] >= merged["beg"]) & (merged[date_col] <= merged["end"])
    ok_rows = merged.loc[hit, "_row"].unique()
    mask = pd.Series(False, index=panel.index)
    mask.iloc[ok_rows] = True
    return mask


def liquidity_flags(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    min_price: float = 5.0,
    adv_window: int = 20,
    adv_min_valid_obs: int = 15,
    adv_min_dollar: float = 5e6,
    min_listed_sessions: int = 120,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    first_trade_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """逐条流动性条件（除市值排名外），返回与 panel 等长的布尔列。

    ADV20 与上市天数都在**交易日历**上滚动：每只股票 reindex 到其首末日
    之间的全部交易日，缺失日算无观测（§9 停牌不压缩时间），再取 20 日窗口。
    全部只用 t 日及以前的信息，禁止回填。

    first_trade_dates : {PERMNO -> 首个交易日}。默认用该 PERMNO 在面板中的
        首行日期近似「上市日」；接入真实数据后应改传
        stksecurityinfohist 的证券起始日，此参数即为该改造的接口。
    """
    df = panel[[permno_col, date_col, "DlyClose", "DlyPrcVol"]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    price_ok_all, adv_ok_all, age_ok_all, idx_all = [], [], [], []

    for pn, g in df.groupby(permno_col):
        g = g.sort_values(date_col)
        sessions = calendar.sessions(g[date_col].iloc[0], g[date_col].iloc[-1])
        s = g.set_index(date_col).reindex(sessions)

        price_ok = s["DlyClose"].notna() & (s["DlyClose"] >= min_price)

        adv = s["DlyPrcVol"].rolling(adv_window, min_periods=adv_min_valid_obs).mean()
        adv_ok = adv.notna() & (adv >= adv_min_dollar)

        if first_trade_dates is not None and pn in first_trade_dates.index:
            first_dt = calendar.snap_forward(first_trade_dates.loc[pn])
        else:
            first_dt = sessions[0]
        # 上市 ≥ 120 个交易日：t 日（含）距首个交易日（含）满 min_listed_sessions 个交易日
        age = np.arange(len(sessions)) + 1 + calendar.sessions_between(first_dt, sessions[0])
        age_ok = pd.Series(age >= min_listed_sessions, index=sessions)

        # 映射回原面板行（只在有行的日期上取值）
        pos = g[date_col]
        price_ok_all.append(price_ok.loc[pos].to_numpy())
        adv_ok_all.append(adv_ok.loc[pos].to_numpy())
        age_ok_all.append(age_ok.loc[pos].to_numpy())
        idx_all.append(g.index.to_numpy())

    idx = np.concatenate(idx_all)
    out = pd.DataFrame(
        {
            "price_ok": np.concatenate(price_ok_all),
            "adv_ok": np.concatenate(adv_ok_all),
            "age_ok": np.concatenate(age_ok_all),
        },
        index=idx,
    ).reindex(panel.index, fill_value=False)
    return out


def selection_panel(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    intervals: pd.DataFrame | None = None,
    top_n_by_cap: int = 1500,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    cap_col: str = "DlyCap",
    **liquidity_kwargs,
) -> pd.DataFrame:
    """组合静态 + 流动性 + 市值排名，返回 panel + 各条件列 + 最终 in_universe。

    市值排名（前 top_n）在「通过静态筛选与其余流动性条件、且 DlyCap 有效」
    的股票中按 t 日值排；并列取排名靠前者（method='first' 按面板顺序，
    确定性由输入排序保证）。
    """
    out = panel.copy()
    if intervals is not None:
        out["static_ok"] = static_eligible_mask(panel, intervals, permno_col, date_col)
    else:
        out["static_ok"] = True

    flags = liquidity_flags(panel, calendar, permno_col=permno_col,
                            date_col=date_col, **liquidity_kwargs)
    out = pd.concat([out, flags], axis=1)

    pre = out["static_ok"] & out["price_ok"] & out["adv_ok"] & out["age_ok"]
    cap = out[cap_col].where(pre & out[cap_col].notna())
    rank = cap.groupby(pd.to_datetime(out[date_col])).rank(ascending=False, method="first")
    out["cap_ok"] = rank <= top_n_by_cap

    out["in_universe"] = pre & out["cap_ok"]
    return out
