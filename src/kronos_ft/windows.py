"""窗口索引构造与内层验证切分。

**训练窗口 = lookback + predict 行**（连续交易日、全部 OHLCV 有效）。
官方 QlibDataset 用 lookback + predict + 1 行（多一个 next-token 目标），
但那会让训练样本实际消耗 predict+1=7 个未来交易日，与 §7 purge 的 6 日
视野差一天。本项目刻意少取 1 行：未来消耗恰为 6 个交易日，purge 断言
保持 §7 原文；代价是每样本少一个移位目标，对训练无结构影响。

lookback / 预测区间含缺口的样本整体排除（§9），复用 cleaning 的
「日历 reindex + 滚动计数」口径。
"""

from __future__ import annotations

import pandas as pd

from crsp_pipeline.calendar import TradingCalendar
from crsp_pipeline.cleaning import valid_ohlc_mask
from crsp_pipeline.splits import assert_purged


def build_window_index(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    lookback: int,
    predict: int,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    require_volume: bool = True,
) -> pd.DataFrame:
    """训练窗口索引：返回 (PERMNO, anchor, start, end)。

    anchor = 信号日 t（lookback 段最后一行）；窗口行 = [t−lookback+1, t+predict]
    共 lookback+predict 个**连续交易日**，全部行存在且 OHLC(V) 有效。
    """
    win = lookback + predict
    df = panel[[permno_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_valid"] = valid_ohlc_mask(panel, require_volume=require_volume).to_numpy()

    frames = []
    for pn, g in df.groupby(permno_col):
        g = g.sort_values(date_col)
        sessions = calendar.sessions(g[date_col].iloc[0], g[date_col].iloc[-1])
        if len(sessions) < win:
            continue
        v = (
            g.set_index(date_col)["_valid"]
            .reindex(sessions)
            .astype(float)
            .fillna(0.0)
        )
        full = v.rolling(win, min_periods=win).sum() == win  # index = 窗口末行 e = t+predict
        ends = v.index[full]
        if len(ends) == 0:
            continue
        pos = {d: i for i, d in enumerate(sessions)}
        rows = pd.DataFrame({
            permno_col: pn,
            "end": ends,
            "anchor": [sessions[pos[e] - predict] for e in ends],
            "start": [sessions[pos[e] - win + 1] for e in ends],
        })
        frames.append(rows[[permno_col, "anchor", "start", "end"]])
    if not frames:
        return pd.DataFrame(columns=[permno_col, "anchor", "start", "end"])
    return pd.concat(frames, ignore_index=True)


def filter_anchors(index: pd.DataFrame, start, end) -> pd.DataFrame:
    """取 anchor（信号日）落在 [start, end] 的样本。训练折用
    [fold.train_start, fold.train_end]——train_end 已按 §7 purge 回退，
    窗口未来行最多到 shift(val_start, −1)，不触碰验证窗。"""
    m = (index["anchor"] >= pd.Timestamp(start)) & (index["anchor"] <= pd.Timestamp(end))
    return index[m].reset_index(drop=True)


def inner_split(
    index: pd.DataFrame,
    calendar: TradingCalendar,
    inner_months: int = 6,
    horizon: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """内层验证切分（预注册 §1.1）：训练窗尾部 inner_months 个月做内层验证，
    内外之间按 horizon 交易日 purge。构造后跑 assert_purged，构造即验证。"""
    if len(index) == 0:
        return index, index
    inner_start = calendar.snap_forward(
        index["anchor"].max() - pd.DateOffset(months=inner_months)
    )
    train_end = calendar.shift(inner_start, -(horizon + 1))
    tr = index[index["anchor"] <= train_end].reset_index(drop=True)
    iv = index[index["anchor"] >= inner_start].reset_index(drop=True)
    if len(tr) and len(iv):
        assert_purged(tr["anchor"], iv["anchor"], calendar, horizon)
    return tr, iv


def build_scoring_index(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    lookback: int,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    require_volume: bool = True,
) -> pd.DataFrame:
    """推理（打分）窗口索引：只要求 lookback 侧 [t−lookback+1, t] 连续有效
    （未来未知）。返回 (PERMNO, anchor, start)。等价于 predict=0 的训练索引。"""
    idx = build_window_index(panel, calendar, lookback, 0,
                             permno_col, date_col, require_volume)
    if len(idx) == 0:
        return pd.DataFrame(columns=[permno_col, "anchor", "start"])
    return idx.rename(columns={})[[permno_col, "anchor", "start"]]
