import pandas as pd
import pytest

from crsp_pipeline.calendar import CalendarError, TradingCalendar


@pytest.fixture
def cal5():
    # 2020-01-06(一) 起连续 10 个工作日
    return TradingCalendar(pd.bdate_range("2020-01-06", periods=10))


def test_shift_skips_weekend(cal5):
    # 周五 2020-01-10 的下一个交易日是周一 2020-01-13
    assert cal5.shift("2020-01-10", 1) == pd.Timestamp("2020-01-13")
    assert cal5.shift("2020-01-06", 6) == pd.Timestamp("2020-01-14")


def test_shift_out_of_range_raises(cal5):
    with pytest.raises(CalendarError):
        cal5.shift("2020-01-17", 1)
    with pytest.raises(CalendarError):
        cal5.shift("2020-01-06", -1)


def test_non_session_raises(cal5):
    with pytest.raises(CalendarError):
        cal5.shift("2020-01-11", 1)  # 周六


def test_sessions_between(cal5):
    assert cal5.sessions_between("2020-01-06", "2020-01-14") == 6
    assert cal5.sessions_between("2020-01-14", "2020-01-06") == -6


def test_sessions_and_snap(cal5):
    s = cal5.sessions("2020-01-08", "2020-01-13")
    assert list(s) == list(pd.to_datetime(["2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"]))
    assert cal5.snap_forward("2020-01-11") == pd.Timestamp("2020-01-13")
    assert cal5.snap_back("2020-01-11") == pd.Timestamp("2020-01-10")
