"""复权（§5）：事件累计因子、anchor 锚定、DlyFacPrc 双路语义判定。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline import adjust as A

DATES = pd.bdate_range("2020-01-06", periods=10)  # d0..d9


def test_single_split_cumfactor():
    # d4 当日 2:1 拆股：d0..d3 因子 2，d4 起 1（anchor = 最后一日）
    ev = pd.DataFrame({"ex_date": [DATES[4]], "factor": [2.0]})
    cf = A.event_cumfactor(ev, DATES, anchor=DATES[-1])
    assert list(cf.iloc[:4]) == [2.0] * 4
    assert list(cf.iloc[4:]) == [1.0] * 6


def test_two_events_compound():
    ev = pd.DataFrame({"ex_date": [DATES[3], DATES[7]], "factor": [2.0, 1.05]})
    cf = A.event_cumfactor(ev, DATES, anchor=DATES[-1])
    assert cf.iloc[0] == pytest.approx(2.0 * 1.05)
    assert cf.iloc[5] == pytest.approx(1.05)
    assert cf.iloc[8] == pytest.approx(1.0)


def test_anchor_is_fixed_not_latest():
    # anchor 在第二个事件之前 → 第二个事件不参与（因子不随「最新日」漂移）
    ev = pd.DataFrame({"ex_date": [DATES[3], DATES[7]], "factor": [2.0, 1.05]})
    cf = A.event_cumfactor(ev, DATES, anchor=DATES[5])
    assert cf.iloc[0] == pytest.approx(2.0)
    assert cf.iloc[5] == pytest.approx(1.0)
    assert cf.iloc[9] == pytest.approx(1.0)  # anchor 之后也不引入新事件


def test_adjust_ohlcv_prices_div_volume_mul():
    sec = pd.DataFrame({
        "DlyOpen": 100.0, "DlyHigh": 110.0, "DlyLow": 90.0, "DlyClose": 105.0,
        "DlyVol": 1000.0,
    }, index=DATES)
    ev = pd.DataFrame({"ex_date": [DATES[4]], "factor": [2.0]})
    out = A.adjust_ohlcv(sec, ev, anchor=DATES[-1])
    assert out["DlyClose"].iloc[0] == pytest.approx(52.5)
    assert out["DlyVol"].iloc[0] == pytest.approx(2000.0)
    assert out["DlyClose"].iloc[5] == pytest.approx(105.0)
    # 原 df 不动
    assert sec["DlyClose"].iloc[0] == 105.0 and sec["DlyVol"].iloc[0] == 1000.0


def test_dual_path_detects_event_semantics():
    # DlyFacPrc 只在事件日 = 2，其余为 1 → 「当期事件因子」语义
    fac = pd.Series(1.0, index=DATES)
    fac.iloc[4] = 2.0
    sec = pd.DataFrame({"DlyFacPrc": fac}, index=DATES)
    ev = pd.DataFrame({"ex_date": [DATES[4]], "factor": [2.0]})
    rep = A.dual_path_report(sec, ev, anchor=DATES[-1])
    assert rep["conclusion"] == "event"


def test_dual_path_detects_cumulative_semantics():
    # DlyFacPrc 本身已是累计因子（前 4 日 2.0，之后 1.0）→ 「累计」语义
    fac = pd.Series([2.0] * 4 + [1.0] * 6, index=DATES)
    sec = pd.DataFrame({"DlyFacPrc": fac}, index=DATES)
    ev = pd.DataFrame({"ex_date": [DATES[4]], "factor": [2.0]})
    rep = A.dual_path_report(sec, ev, anchor=DATES[-1])
    assert rep["conclusion"] == "cumulative"


def test_no_events_identity():
    cf = A.event_cumfactor(pd.DataFrame(columns=["ex_date", "factor"]), DATES, DATES[-1])
    assert (cf == 1.0).all()


def test_negative_factor_rejected():
    ev = pd.DataFrame({"ex_date": [DATES[2]], "factor": [-2.0]})
    with pytest.raises(ValueError):
        A.event_cumfactor(ev, DATES, DATES[-1])
