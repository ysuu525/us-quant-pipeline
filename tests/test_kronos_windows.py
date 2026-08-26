"""窗口索引（kronos_ft.windows）：缺口排除、anchor 过滤、内层 purge 边界。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline.calendar import TradingCalendar
from crsp_pipeline.splits import PurgeViolation, assert_purged
from kronos_ft.windows import (
    build_scoring_index,
    build_window_index,
    filter_anchors,
    inner_split,
)


def make_panel(dates, permno=1):
    n = len(dates)
    return pd.DataFrame({
        "PERMNO": permno, "DlyCalDt": dates,
        "DlyOpen": 10.0, "DlyHigh": 11.0, "DlyLow": 9.0, "DlyClose": 10.5,
        "DlyVol": 1000.0, "DlyPrcVol": 10500.0 * np.ones(n),
    })


def test_window_counts_and_boundaries(cal):
    n, L, P = 50, 10, 6
    p = make_panel(cal.dates[:n])
    idx = build_window_index(p, cal, L, P)
    # 连续 50 日 → 50 − 16 + 1 = 35 个窗口
    assert len(idx) == 35
    first = idx.iloc[0]
    assert first["start"] == cal.dates[0]
    assert first["anchor"] == cal.dates[L - 1]      # lookback 段最后一行
    assert first["end"] == cal.dates[L + P - 1]     # 未来消耗恰 6 个交易日
    # anchor 与 end 恰差 predict 个交易日（§7 purge 视野一致性）
    assert cal.sessions_between(first["anchor"], first["end"]) == P


def test_gap_excludes_windows(cal):
    L, P = 5, 3
    positions = [i for i in range(30) if i != 12]  # 位置 12 缺行
    p = make_panel(cal.dates[positions])
    idx = build_window_index(p, cal, L, P)
    # 触及位置 12 的窗口（end ∈ [12, 19]）全被排除
    banned = set(cal.dates[12:20])
    assert not (set(idx["end"]) & banned)
    # 缺口之后恢复
    assert cal.dates[20] in set(idx["end"])


def test_invalid_ohlc_breaks_window(cal):
    L, P = 5, 3
    p = make_panel(cal.dates[:30])
    p.loc[12, "DlyClose"] = np.nan
    idx = build_window_index(p, cal, L, P)
    assert cal.dates[12] not in set(idx["end"]) and cal.dates[13] not in set(idx["end"])


def test_filter_and_inner_split_purged(cal):
    L, P = 10, 6
    p = make_panel(cal.dates[:400])
    idx = build_window_index(p, cal, L, P)
    idx = filter_anchors(idx, cal.dates[0], cal.dates[380])
    tr, iv = inner_split(idx, cal, inner_months=2, horizon=P)
    assert len(tr) and len(iv)
    # purge 恰好：max(train anchor)+6 交易日 < min(inner anchor)
    assert_purged(tr["anchor"], iv["anchor"], cal, P)
    gap = cal.sessions_between(tr["anchor"].max(), iv["anchor"].min())
    assert gap == P + 1  # 不多不少，off-by-one 卡死
    with pytest.raises(PurgeViolation):
        assert_purged(tr["anchor"], [cal.shift(tr["anchor"].max(), P)], cal, P)


def test_scoring_index_lookback_only(cal):
    L = 10
    p = make_panel(cal.dates[:20])
    idx = build_scoring_index(p, cal, L)
    # 只要 lookback 侧连续：20 − 10 + 1 = 11 个 anchor，最后一个是末日
    assert len(idx) == 11
    assert idx["anchor"].iloc[-1] == cal.dates[19]
