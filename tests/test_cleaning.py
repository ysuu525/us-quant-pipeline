"""清洗规则（§9）：BA 统计、lookback 缺口整体排除、排除率报告。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline import cleaning as C


def make_rows(cal, permno, positions, valid=True, exch="N"):
    d = cal.dates[list(positions)]
    n = len(d)
    v = 10.0 if valid else np.nan
    return pd.DataFrame({
        "PERMNO": permno, "DlyCalDt": d,
        "DlyOpen": v, "DlyHigh": v, "DlyLow": v, "DlyClose": v, "DlyVol": 100.0,
        "DlyPrcFlg": "", "PrimaryExch": exch,
    })


def test_valid_ohlc_mask(cal):
    p = make_rows(cal, 1, range(3))
    p.loc[1, "DlyHigh"] = np.nan
    p.loc[2, "DlyClose"] = 0.0  # 非正价格无效
    m = C.valid_ohlc_mask(p)
    assert list(m) == [True, False, False]


def test_ba_flag_stats(cal):
    p = make_rows(cal, 1, range(4))
    p.loc[[1, 2], "DlyPrcFlg"] = "BA"
    s = C.ba_flag_stats(p)
    assert s.iloc[0]["n"] == 4 and s.iloc[0]["n_ba"] == 2
    assert s.iloc[0]["ba_share"] == pytest.approx(0.5)


def test_lookback_gap_excludes_whole_sample(cal):
    # 30 个交易日，位置 9 缺行（停牌，不压缩时间）。lookback=5：
    # 窗口触及缺口的样本（位置 10..13）整体排除；位置 14 起窗口重新连续。
    positions = [i for i in range(30) if i != 9]
    p = make_rows(cal, 1, positions)
    u = C.lookback_usable_mask(p, cal, lookback=5).set_index("DlyCalDt")["usable"]

    # 头部：不足 5 日历史的样本不可用（位置 0..3），位置 4 起可用
    assert not u.loc[cal.dates[3]]
    assert u.loc[cal.dates[4]] and u.loc[cal.dates[8]]
    # 缺口后 4 个样本不可用（它们的 5 日窗口含位置 9）
    for pos in (10, 11, 12, 13):
        assert not u.loc[cal.dates[pos]]
    assert u.loc[cal.dates[14]]


def test_invalid_ohlc_breaks_window(cal):
    # 行存在但 OHLC 无效，效果等同缺行
    p = make_rows(cal, 1, range(20))
    p.loc[10, "DlyClose"] = np.nan
    u = C.lookback_usable_mask(p, cal, lookback=5).set_index("DlyCalDt")["usable"]
    assert not u.loc[cal.dates[12]]
    assert u.loc[cal.dates[15]]


def test_exclusion_report_by_year_exchange(cal):
    a = make_rows(cal, 1, range(10), exch="N")
    b = make_rows(cal, 2, range(10), exch="Q")
    p = pd.concat([a, b], ignore_index=True)
    u = C.lookback_usable_mask(p, cal, lookback=5)
    rep = C.exclusion_report(u, p)
    # 每只股票前 4 个样本被排除：排除率 4/10
    assert set(rep["PrimaryExch"]) == {"N", "Q"}
    assert rep["exclusion_rate"].tolist() == pytest.approx([0.4, 0.4])
