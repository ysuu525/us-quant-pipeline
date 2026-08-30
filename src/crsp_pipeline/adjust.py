"""复权模块（规范 §5）——独立模块，不与标签引擎混用。

统一路径：从 CIZ distribution / corporate-action **事件**构造以固定基准日
（anchor）为锚的累计复权因子；禁止逐日裸乘 ``DlyFacPrc``。

- Kronos 蜡烛复权只处理拆股与股票股利；现金股息、分拆、配股不写入 OHLC。
  CIZ 事件码规则已于 2026-08-26 用真实快照核对并冻结（见
  ``split_events_from_distributions``）：``distype='FRS'``（disdetailtype ∈
  {STKSPL 拆股, STKDIV 股票股利}），factor = 1 + disfacshr。
- ``DlyFacPrc`` 的语义已验证为**当期事件因子**（AAPL 2020-08-31、NVDA
  2024-06-10 双路对比均判 event，§5 验证条款）；管线仍走事件累计路径，
  ``DlyFacPrc`` 只用于交叉审计（scripts/prepare_data.py 的 audit）。

因子约定
--------
事件表每行一个事件：(ex_date, factor)，factor = 新股数/旧股数（2:1 拆股
factor=2，5% 股票股利 factor=1.05）。ex-date 当日起价格按新股本计。

以 anchor 为基准的累计因子：

    cumfactor(t) = ∏ factor(e)   对所有 t < ex_date(e) ≤ anchor 的事件 e

调整到 anchor 口径：price_adj(t) = price_raw(t) / cumfactor(t)，
volume_adj(t) = volume_raw(t) × cumfactor(t)。anchor 之后的事件不参与
（锚定固定基准日，因子不随「最新日」漂移，可复现）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 冻结（2026-08-26，真实数据核对）：写入 OHLC 的复权事件 = FRS（拆股+股票股利）。
# 全表 FRS 行 disfacpr == disfacshr；反向拆股 disfacshr<0（factor∈(0,1)）。
# 分拆（SP/SEC*）等价格因子事件刻意排除（§5：不写入 OHLC）。
SPLIT_DISTYPE = "FRS"


def split_events_from_distributions(dist: pd.DataFrame) -> pd.DataFrame:
    """CIZ distributions（数据库小写列名）→ 复权事件表 (PERMNO, ex_date, factor)。

    factor = 新股数/旧股数 = 1 + disfacshr（CIZ 沿用 legacy 约定
    facshr = (新−旧)/旧）。"""
    ev = dist[dist["distype"] == SPLIT_DISTYPE].copy()
    ev["ex_date"] = pd.to_datetime(ev["disexdt"])
    ev["factor"] = 1.0 + ev["disfacshr"].astype(float)
    if (ev["factor"] <= 0).any():
        bad = ev.loc[ev["factor"] <= 0, ["permno", "disexdt", "disfacshr"]]
        raise ValueError(f"非正复权因子（disfacshr ≤ -1）:\n{bad.head()}")
    out = ev.rename(columns={"permno": "PERMNO"})[["PERMNO", "ex_date", "factor"]]
    return out.sort_values(["PERMNO", "ex_date"], ignore_index=True)


def adjust_panel(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    anchor: pd.Timestamp | str,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
) -> pd.DataFrame:
    """全量面板 → §5 复权后的训练蜡烛面板（8 列）。

    price_adj = price / cumfactor；vol_adj = vol × cumfactor；
    amt（DlyPrcVol）不动——price×vol 对拆股不变。无事件的股票 cumfactor=1。
    events 为 ``split_events_from_distributions`` 的输出（或同构表）。"""
    out = panel[[permno_col, date_col, "DlyOpen", "DlyHigh", "DlyLow",
                 "DlyClose", "DlyVol", "DlyPrcVol"]].copy()
    ev_by_pn = {pn: g for pn, g in events.groupby(permno_col)}
    cf = np.ones(len(out))
    dates_all = out[date_col].to_numpy()
    for pn, idx in out.groupby(permno_col).indices.items():
        ev = ev_by_pn.get(pn)
        if ev is None:
            continue
        cf[idx] = event_cumfactor(ev, pd.DatetimeIndex(dates_all[idx]), anchor).to_numpy()
    for c in ("DlyOpen", "DlyHigh", "DlyLow", "DlyClose"):
        out[c] = out[c].to_numpy() / cf
    out["DlyVol"] = out["DlyVol"].to_numpy() * cf
    return out


def event_cumfactor(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    anchor: pd.Timestamp | str,
    ex_date_col: str = "ex_date",
    factor_col: str = "factor",
) -> pd.Series:
    """对给定日期序列计算以 anchor 为锚的累计复权因子（单只证券）。"""
    anchor = pd.Timestamp(anchor)
    dates = pd.DatetimeIndex(dates)
    ev = events.copy()
    if len(ev) == 0:
        return pd.Series(1.0, index=dates)
    ev[ex_date_col] = pd.to_datetime(ev[ex_date_col])
    ev = ev[ev[ex_date_col] <= anchor].sort_values(ex_date_col)
    if (ev[factor_col] <= 0).any():
        raise ValueError("event factor must be positive")

    # 对每个 t：乘上所有 t < ex_date ≤ anchor 的事件因子。
    # 用从后往前的累乘：suffix[i] = ∏_{j≥i} factor_j
    f = ev[factor_col].to_numpy(dtype=float)
    suffix = np.concatenate([np.cumprod(f[::-1])[::-1], [1.0]])
    # 每个 t 找到第一个 ex_date > t 的事件位置
    pos = np.searchsorted(ev[ex_date_col].to_numpy(), dates.to_numpy(), side="right")
    return pd.Series(suffix[pos], index=dates)


def adjust_ohlcv(
    sec: pd.DataFrame,
    events: pd.DataFrame,
    anchor: pd.Timestamp | str,
    price_cols: tuple[str, ...] = ("DlyOpen", "DlyHigh", "DlyLow", "DlyClose"),
    volume_col: str | None = "DlyVol",
) -> pd.DataFrame:
    """把单只证券的 OHLCV（index 为日期）调整到 anchor 口径。

    输入 events 必须已由调用方筛成「拆股 + 股票股利」事件（§5：现金股息、
    分拆、配股不写入 OHLC）。返回副本，原 df 不动。
    """
    cf = event_cumfactor(events, sec.index, anchor)
    out = sec.copy()
    for c in price_cols:
        if c in out.columns:
            out[c] = out[c] / cf
    if volume_col and volume_col in out.columns:
        out[volume_col] = out[volume_col] * cf
    return out


def dual_path_report(
    sec: pd.DataFrame,
    events: pd.DataFrame,
    anchor: pd.Timestamp | str,
    facprc_col: str = "DlyFacPrc",
    rtol: float = 1e-6,
) -> dict:
    """双路构造对比（§5 验证）：事件累计因子 vs 两种 DlyFacPrc 语义假设。

    语义 A（当期事件因子）：DlyFacPrc 仅在事件日非 1/非空，累计因子需自行
        从后向前累乘；
    语义 B（累计因子）：DlyFacPrc 本身已是某种锚定的累计因子，可直接用
        （允许整体差一个常数比例，因为锚可能不同）。

    返回各路径序列与匹配结论，供在 AAPL/NVDA 拆股日跑完后锁定语义。
    """
    cf_event = event_cumfactor(events, sec.index, anchor)

    fac = sec[facprc_col].astype(float)
    # 语义 A：把非空且 ≠1 的值当作当日事件因子，构造事件表走同一条累计路径
    ev_a = pd.DataFrame({
        "ex_date": sec.index[fac.notna() & (fac != 1.0)],
        "factor": fac[fac.notna() & (fac != 1.0)].to_numpy(),
    })
    cf_a = event_cumfactor(ev_a, sec.index, anchor)

    # 语义 B：直接用（与事件路径比对时允许统一缩放：除以 anchor 之前最后
    # 一个有效值使锚一致）
    fac_filled = fac.ffill().bfill()
    base = fac_filled.loc[fac_filled.index <= pd.Timestamp(anchor)]
    scale = base.iloc[-1] if len(base) else 1.0
    cf_b = fac_filled / scale if scale not in (0, np.nan) else fac_filled

    def _match(x: pd.Series) -> bool:
        both = pd.concat([cf_event.rename("e"), x.rename("x")], axis=1).dropna()
        if len(both) == 0:
            return False
        return bool(np.allclose(both["e"], both["x"], rtol=rtol))

    report = {
        "cumfactor_event": cf_event,
        "cumfactor_facprc_as_event": cf_a,
        "cumfactor_facprc_as_cumulative": cf_b,
        "matches_event_semantics": _match(cf_a),
        "matches_cumulative_semantics": _match(cf_b),
    }
    report["conclusion"] = (
        "event" if report["matches_event_semantics"] and not report["matches_cumulative_semantics"]
        else "cumulative" if report["matches_cumulative_semantics"] and not report["matches_event_semantics"]
        else "ambiguous" if report["matches_event_semantics"] and report["matches_cumulative_semantics"]
        else "neither"
    )
    return report
