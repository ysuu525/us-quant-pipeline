"""Universe 筛选（§2）：价格/ADV/上市天数按 t 日滚动、市值排名、静态区间。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline import universe as U


def make_panel(cal, permno, n, start=0, close=10.0, vol=1e7, cap=100.0,
               closes=None, vols=None, dates=None):
    d = dates if dates is not None else cal.dates[start:start + n]
    n = len(d)
    return pd.DataFrame({
        "PERMNO": permno,
        "DlyCalDt": d,
        "DlyClose": closes if closes is not None else [close] * n,
        "DlyPrcVol": vols if vols is not None else [vol] * n,
        "DlyCap": cap,
    })


def test_price_and_age_flags_roll_on_t_day_info(cal):
    n = 200
    closes = [4.0] * 50 + [10.0] * (n - 50)  # 前 50 日低于 $5，之后达标
    p = make_panel(cal, 1, n, closes=closes)
    f = U.liquidity_flags(p, cal)

    # 价格条件严格按 t 日：第 50 天（0 起）起才通过，禁止回填
    assert not f["price_ok"].iloc[49] and f["price_ok"].iloc[50]
    # 上市 ≥120 个交易日：第 120 个交易日（下标 119）起
    assert not f["age_ok"].iloc[118] and f["age_ok"].iloc[119]
    # ADV20 ≥15 个有效观测：第 15 个观测（下标 14）起有值
    assert not f["adv_ok"].iloc[13] and f["adv_ok"].iloc[14]


def test_adv_needs_15_valid_obs_in_window(cal):
    # 前 5 日 DlyPrcVol 缺失：窗口内有效观测到下标 18 只有 14 个 → 不通过
    vols = [np.nan] * 5 + [1e7] * 15
    p = make_panel(cal, 1, 20, vols=vols)
    f = U.liquidity_flags(p, cal)
    assert not f["adv_ok"].iloc[18]
    assert f["adv_ok"].iloc[19]  # 15 个有效观测，均值 1e7 ≥ 5e6


def test_missing_rows_count_as_no_observation(cal):
    # 行情行缺失的交易日 = 无观测（§9 不压缩时间）：中间停牌 10 日后，
    # 尽管自身每行 vol 都有效，20 日窗口内有效观测不足 → adv 不通过
    dates = list(cal.dates[:5]) + list(cal.dates[15:25])
    p = make_panel(cal, 1, None, dates=pd.DatetimeIndex(dates))
    f = U.liquidity_flags(p, cal)
    # 最后一行：日历位置 24，窗口 [5..24] 内只有位置 15..24 共 10 个有效观测
    assert not f["adv_ok"].iloc[-1]


def test_cap_rank_top_n_excludes_failed_stocks(cal):
    n = 30
    a = make_panel(cal, 1, n, cap=200.0)
    b = make_panel(cal, 2, n, cap=100.0)
    c = make_panel(cal, 3, n, close=4.0, cap=1000.0)  # 价格不达标，市值最大
    panel = pd.concat([a, b, c], ignore_index=True)

    out = U.selection_panel(panel, cal, intervals=None, top_n_by_cap=1,
                            min_listed_sessions=1)
    t = cal.dates[20]  # ADV 预热期之后
    day = out[out.DlyCalDt == t].set_index("PERMNO")
    # C 价格不达标，不得占用排名名额；名额归市值第二大的 A
    assert bool(day.loc[1, "in_universe"])
    assert not bool(day.loc[2, "in_universe"])
    assert not bool(day.loc[3, "in_universe"])


def test_static_intervals_and_mask(cal):
    base = {
        "sharetype": "NS", "securitytype": "EQTY", "securitysubtype": "COM",
        "usincflg": "Y", "issuertype": "CORP", "primaryexch": "N",
        "conditionaltype": "RW", "tradingstatusflg": "A",
    }
    rows = [
        # permno 1：上半年合格；下半年 sharetype 变化 → 不合格
        dict(base, permno=1, secinfostartdt="2020-01-01", secinfoenddt="2020-06-30"),
        dict(base, permno=1, sharetype="AD", secinfostartdt="2020-07-01", secinfoenddt="9999-12-31"),
        # permno 2：场外交易所 → 不合格
        dict(base, permno=2, primaryexch="O", secinfostartdt="2020-01-01", secinfoenddt="9999-12-31"),
        # permno 3：合格且「至今有效」（end 为远期哨兵字符串）
        dict(base, permno=3, secinfostartdt="2020-01-01", secinfoenddt="9999-12-31"),
    ]
    info = pd.DataFrame(rows)
    iv = U.static_eligible_intervals(info)
    assert set(iv["PERMNO"]) == {1, 3}

    panel = pd.DataFrame({
        "PERMNO": [1, 1, 2, 3],
        "DlyCalDt": pd.to_datetime(["2020-02-03", "2020-08-03", "2020-02-03", "2021-06-01"]),
    })
    m = U.static_eligible_mask(panel, iv)
    assert list(m) == [True, False, False, True]
