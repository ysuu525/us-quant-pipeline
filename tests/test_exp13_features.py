"""exp13 构造层：§4.1 SUE、§4.2 rdq_hat / ea_prox、§3.4 时点、折号白名单。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp13_compustat_dev_diag", REPO / "scripts" / "exp13_compustat_dev_diag.py"
)
EXP13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP13)

CAL = pd.DatetimeIndex(pd.bdate_range("2018-01-01", "2023-12-29"))


def quarterly(gvkey: str, quarters, values, rdqs=None) -> pd.DataFrame:
    rows = []
    for index, ((year, qtr), value) in enumerate(zip(quarters, values)):
        month = qtr * 3
        datadate = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        rdq = None if rdqs is None else rdqs[index]
        rows.append({
            "gvkey": gvkey, "datadate": datadate,
            "rdq": pd.NaT if rdq is None else pd.Timestamp(rdq),
            "fyearq": float(year), "fqtr": float(qtr), "curcdq": "USD",
            "dedup_group_rows": 1, "epspxq": value, "ajexq": 1.0,
        })
    return pd.DataFrame(rows)


ALL_Q = [(y, q) for y in range(2018, 2023) for q in (1, 2, 3, 4)]


# --------------------------------------------------------------------------
# §4.1 SUE
# --------------------------------------------------------------------------
def test_seasonal_difference_aligns_on_fiscal_quarter_not_row_position():
    """删掉 2019Q2 后，2020Q2 的季节差必须是缺失，而不是错配到 2019Q1。"""
    quarters = [q for q in ALL_Q if q != (2019, 2)]
    values = [float(index) for index in range(len(quarters))]
    frame = quarterly("A", quarters, values)
    out, _ = EXP13.build_quarter_features(frame)
    out = out.set_index("fidx")
    idx_2020q2 = 2020 * 4 + 1
    idx_2020q3 = 2020 * 4 + 2
    assert np.isnan(out.loc[idx_2020q2, "seasonal_diff"])
    # 相邻的 2020Q3 有完整的 q-4（2019Q3），必须算得出来
    e_2020q3 = out.loc[idx_2020q3, "E"]
    e_2019q3 = out.loc[2019 * 4 + 2, "E"]
    assert out.loc[idx_2020q3, "seasonal_diff"] == pytest.approx(e_2020q3 - e_2019q3)


def test_sue_is_missing_when_fewer_than_six_seasonal_differences():
    """§4.1：σ 窗口内非缺 < 6 个置缺。"""
    quarters = ALL_Q[:12]                       # 12 季 -> 只有 8 个季节差
    frame = quarterly("A", quarters, [float(i) * 1.7 % 3 for i in range(12)])
    out, _ = EXP13.build_quarter_features(frame)
    out = out.set_index("fidx")
    # 第 5 个季节差（2020Q1）之前不足 6 个观测 -> σ、SUE 均缺
    assert np.isnan(out.loc[2020 * 4 + 0, "sigma"])
    assert np.isnan(out.loc[2020 * 4 + 0, "sue"])
    # 到第 6 个季节差（2020Q2）刚好 6 个 -> 可算
    assert np.isfinite(out.loc[2020 * 4 + 1, "sigma"])
    assert np.isfinite(out.loc[2020 * 4 + 1, "sue"])


def test_sue_is_missing_when_sigma_is_zero():
    """§4.1：σ = 0 置缺（每年恒定增长 -> 季节差全等）。"""
    values = [10.0 + 4.0 * (year - 2018) for year, _ in ALL_Q]
    frame = quarterly("A", ALL_Q, values)
    out, _ = EXP13.build_quarter_features(frame)
    out = out.set_index("fidx")
    late = out.loc[2022 * 4 + 3]
    assert late["sigma"] == 0.0
    assert np.isnan(late["sue"])


def test_sue_uses_ajex_adjusted_eps():
    frame = quarterly("A", ALL_Q, [1.0] * len(ALL_Q))
    frame.loc[frame.index[-1], "ajexq"] = 2.0
    out, _ = EXP13.build_quarter_features(frame)
    assert out["E"].iloc[-1] == pytest.approx(0.5)
    assert out["E"].iloc[-2] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# §4.2 rdq_hat 只用上一财年
# --------------------------------------------------------------------------
def test_rdq_hat_comes_from_previous_fiscal_year_same_quarter_only():
    rdqs = [pd.Timestamp(year=year, month=qtr * 3, day=1) + pd.Timedelta(days=40)
            for year, qtr in ALL_Q]
    frame = quarterly("A", ALL_Q, [float(i) for i in range(len(ALL_Q))], rdqs=rdqs)
    out, _ = EXP13.build_quarter_features(frame)
    out = out.set_index("fidx")
    target = 2021 * 4 + 2                       # 2021Q3
    source = 2020 * 4 + 2                       # 2020Q3
    assert out.loc[target, "rdq_source"] == out.loc[source, "rdq"]
    assert out.loc[target, "rdq_hat"] == out.loc[source, "rdq"] + pd.DateOffset(years=1)
    # 第一财年没有上一年 -> 置缺
    assert pd.isna(out.loc[2018 * 4 + 0, "rdq_hat"])


# --------------------------------------------------------------------------
# §3.4 时点
# --------------------------------------------------------------------------
def _permno_quarters(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("datadate", "rdq", "rdq_hat", "rdq_source"):
        if column not in frame:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("curcdq",):
        if column not in frame:
            frame[column] = "USD"
    if "dedup_group_rows" not in frame:
        frame["dedup_group_rows"] = 1
    return frame.sort_values(["PERMNO", "rdq"], kind="mergesort").reset_index(drop=True)


def _keys(permno, days):
    return pd.DataFrame({"PERMNO": [permno] * len(days),
                         "signal_date": pd.to_datetime(list(days))})


def test_quarter_is_invisible_on_rdq_day_and_visible_next_trading_day():
    rdq = pd.Timestamp("2021-02-10")            # 周三
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2020-12-31", "rdq": rdq, "sue": 0.7,
    }])
    days = [pd.Timestamp("2021-02-09"), rdq, pd.Timestamp("2021-02-11")]
    out, _ = EXP13.assign_point_in_time(_keys(1, days), quarters, CAL)
    got = out.set_index("signal_date")["sue"]
    assert np.isnan(got.loc[pd.Timestamp("2021-02-09")])     # 公告前
    assert np.isnan(got.loc[rdq])                            # 公告当日不可用
    assert got.loc[pd.Timestamp("2021-02-11")] == pytest.approx(0.7)


def test_rdq_on_weekend_becomes_available_on_the_following_monday():
    rdq = pd.Timestamp("2021-02-13")            # 周六
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2020-12-31", "rdq": rdq, "sue": 0.4,
    }])
    monday = pd.Timestamp("2021-02-15")
    out, _ = EXP13.assign_point_in_time(_keys(1, [pd.Timestamp("2021-02-12"), monday]),
                                       quarters, CAL)
    got = out.set_index("signal_date")["sue"]
    assert np.isnan(got.loc[pd.Timestamp("2021-02-12")])
    assert got.loc[monday] == pytest.approx(0.4)


def test_quarter_older_than_180_calendar_days_is_voided():
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2021-01-01", "rdq": "2021-02-01", "sue": 1.5,
    }])
    fresh = pd.Timestamp("2021-01-01") + pd.Timedelta(days=EXP13.STALENESS_DAYS)
    stale = fresh + pd.Timedelta(days=1)
    # 取日历上最近的交易日以避免落在周末
    fresh = CAL[CAL.get_indexer([fresh], method="bfill")[0]]
    stale = CAL[CAL.get_indexer([stale], method="bfill")[0]]
    out, stats = EXP13.assign_point_in_time(_keys(1, [fresh, stale]), quarters, CAL)
    got = out.set_index("signal_date")
    age_fresh = (fresh - pd.Timestamp("2021-01-01")).days
    if age_fresh <= EXP13.STALENESS_DAYS:
        assert got.loc[fresh, "sue"] == pytest.approx(1.5)
    assert np.isnan(got.loc[stale, "sue"])
    assert np.isnan(got.loc[stale, "quarter_age_days"])
    assert stats["rows_stale_beyond_180d"] >= 1


def test_quarter_without_rdq_is_never_usable():
    quarters = _permno_quarters([
        {"PERMNO": 1, "datadate": "2021-03-31", "rdq": pd.NaT, "sue": 9.9},
        {"PERMNO": 1, "datadate": "2020-12-31", "rdq": "2021-02-01", "sue": 0.2},
    ])
    day = pd.Timestamp("2021-06-01")
    out, _ = EXP13.assign_point_in_time(_keys(1, [day]), quarters, CAL)
    # rdq 缺的那一季不可用；落到上一季（且仍在 180 日内）
    assert out["sue"].iloc[0] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# §4.2 ea_prox
# --------------------------------------------------------------------------
def test_ea_prox_counts_trading_days_and_caps_at_63():
    hat_near = pd.Timestamp("2021-03-10")
    quarters = _permno_quarters([
        {"PERMNO": 1, "datadate": "2020-12-31", "rdq": "2021-02-01", "sue": np.nan,
         "rdq_hat": hat_near, "rdq_source": "2020-03-10"},
        {"PERMNO": 1, "datadate": "2021-03-31", "rdq": "2021-05-01", "sue": np.nan,
         "rdq_hat": "2021-12-01", "rdq_source": "2020-12-01"},
    ])
    day = pd.Timestamp("2021-03-03")
    far_day = pd.Timestamp("2021-06-01")
    out, _ = EXP13.assign_point_in_time(_keys(1, [day, far_day]), quarters, CAL)
    got = out.set_index("signal_date")["ea_prox"]
    expected = CAL.get_loc(hat_near) - CAL.get_loc(day)
    assert got.loc[day] == pytest.approx(-min(expected, EXP13.EA_CAP_TRADING_DAYS))
    assert got.loc[far_day] == pytest.approx(-EXP13.EA_CAP_TRADING_DAYS)


def test_ea_prox_is_zero_on_the_predicted_day_itself():
    hat = pd.Timestamp("2021-03-10")
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2020-12-31", "rdq": "2021-02-01", "sue": np.nan,
        "rdq_hat": hat, "rdq_source": "2020-03-10",
    }])
    out, _ = EXP13.assign_point_in_time(_keys(1, [hat]), quarters, CAL)
    assert out["ea_prox"].iloc[0] == pytest.approx(0.0)


def test_ea_prox_is_missing_without_any_future_predicted_date():
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2020-12-31", "rdq": "2021-02-01", "sue": np.nan,
        "rdq_hat": pd.NaT, "rdq_source": pd.NaT,
    }])
    out, _ = EXP13.assign_point_in_time(_keys(1, [pd.Timestamp("2021-03-03")]),
                                       quarters, CAL)
    assert np.isnan(out["ea_prox"].iloc[0])


def test_ea_prox_rejects_a_predicted_date_whose_source_is_not_yet_announced():
    """前视闸门：rdq_hat 必须由信号日之前已公告的 rdq 推出。"""
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2020-12-31", "rdq": "2021-02-01", "sue": np.nan,
        "rdq_hat": "2021-03-10", "rdq_source": "2021-03-05",   # 源公告还在未来
    }])
    day = pd.Timestamp("2021-03-03")
    out, stats = EXP13.assign_point_in_time(_keys(1, [day]), quarters, CAL)
    assert np.isnan(out["ea_prox"].iloc[0])
    assert stats["ea_prox_rows_voided_source_not_yet_announced"] == 1


def test_ea_real_uses_actual_announcement_dates():
    quarters = _permno_quarters([{
        "PERMNO": 1, "datadate": "2021-03-31", "rdq": "2021-05-05", "sue": np.nan,
    }])
    day = pd.Timestamp("2021-05-03")
    out, _ = EXP13.assign_point_in_time(_keys(1, [day]), quarters, CAL)
    expected = CAL.get_loc(pd.Timestamp("2021-05-05")) - CAL.get_loc(day)
    assert out["ea_real"].iloc[0] == pytest.approx(-expected)


# --------------------------------------------------------------------------
# 折号白名单（任务书 §2.1）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [5, 20, 35, 43, 44, 45, 0, -1])
def test_fold_whitelist_rejects_unconsumed_folds(bad):
    with pytest.raises(EXP13.FoldWhitelistError):
        EXP13.assert_folds([36, bad])


def test_fold_whitelist_accepts_exactly_the_seven_dev_folds():
    assert EXP13.assert_folds(range(36, 43)) == tuple(range(36, 43))
    assert EXP13.ALLOWED_FOLDS_EXP13 == frozenset(range(36, 43))
    with pytest.raises(EXP13.FoldWhitelistError):
        EXP13.assert_folds([])


def test_frozen_constants_match_the_task_document():
    assert EXP13.STALENESS_DAYS == 180
    assert EXP13.EA_CAP_TRADING_DAYS == 63
    assert EXP13.SUE_LOOKBACK_QUARTERS == 8
    assert EXP13.SUE_MIN_OBS == 6
    assert EXP13.SUE_SEASONAL_LAG == 4
    assert EXP13.HOLDING_DAYS == 6
    assert EXP13.NW_LAGS == 5
    assert EXP13.K6B.NT == 6 and EXP13.K6B.TOPN == 500
    assert EXP13.SPEC_COLUMNS["S-TH-ind-SUE"] == EXP13.BASE_CONTROLS + ("sue",)
    assert EXP13.SPEC_COLUMNS["S-TH-ind-EA"] == EXP13.BASE_CONTROLS + ("ea_prox",)
    assert EXP13.SPEC_COLUMNS["S-TH-ind-SUE-EA"] == EXP13.BASE_CONTROLS + ("sue", "ea_prox")
    assert EXP13.SPEC_COLUMNS["S-TH-ind-EAreal"] == EXP13.BASE_CONTROLS + ("ea_real",)
    assert EXP13.SPEC_COLUMNS["S-TH-ind-repro"] == EXP13.BASE_CONTROLS
