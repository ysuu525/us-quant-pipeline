"""标签引擎 golden case（§4 五段复合、退市接管、unfillable、插补）。

信号日 t = 2020-03-02（周一），持有期（工作日历）：
t+1=03-03, t+2=03-04, t+3=03-05, t+4=03-06, t+5=03-09, t+6=03-10。
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_sec
from crsp_pipeline import labels as L

T = pd.Timestamp("2020-03-02")
H = pd.to_datetime(["2020-03-03", "2020-03-04", "2020-03-05", "2020-03-06", "2020-03-09", "2020-03-10"])


def normal_sec():
    # t 行本身给不给都行；这里给全 t..t+6
    dates = [T] + list(H)
    opens = [99.0, 100.0, np.nan, np.nan, np.nan, np.nan, 111.0]
    closes = [99.5, 102.0, np.nan, np.nan, np.nan, 110.0, 112.0]
    rets = [0.0, 0.01, 0.01, 0.02, -0.01, 0.005, 0.009]
    return make_sec(dates, opens, closes, rets)


def test_normal_path_golden(cal):
    r = L.compute_label(normal_sec(), T, cal)
    assert r.status == L.STATUS_OK
    expected = (102.0 / 100.0) * 1.01 * 1.02 * 0.99 * 1.005 * (111.0 / 110.0) - 1.0
    assert r.label == pytest.approx(expected, rel=1e-12)
    assert not r.delist_takeover and not r.imputed
    assert r.entry_open == 100.0
    assert r.exit_dividend == 0.0
    # 段分解一致
    assert r.segments["day1"] == pytest.approx(0.02)
    assert r.segments["mid"] == pytest.approx([0.01, 0.02, -0.01, 0.005])
    assert r.segments["exit"] == pytest.approx(111.0 / 110.0 - 1.0)


def test_exit_exdate_dividend(cal):
    # t+6 为 ex-date：退出段 = (open(t6)+div)/close(t5) − 1（v1.2 新增）
    div = {H[5]: 0.5}
    r = L.compute_label(normal_sec(), T, cal, dividends=div)
    expected = (102.0 / 100.0) * 1.01 * 1.02 * 0.99 * 1.005 * ((111.0 + 0.5) / 110.0) - 1.0
    assert r.label == pytest.approx(expected, rel=1e-12)
    assert r.exit_dividend == 0.5


def test_day1_excludes_t1_dividend(cal):
    # 建仓在 ex 之后：t+1 的股息不进标签 —— t+1 有股息时结果与无股息完全相同
    r0 = L.compute_label(normal_sec(), T, cal)
    r1 = L.compute_label(normal_sec(), T, cal, dividends={H[0]: 1.0})
    assert r1.label == pytest.approx(r0.label, rel=1e-15)


def test_delist_mid_hold_takeover(cal):
    # t+3 = 2020-03-05 为退市终值记录，DlyRet 已含退市收益；此后无行
    dates = [H[0], H[1], H[2]]
    sec = make_sec(dates,
                   opens=[100.0, np.nan, np.nan],
                   closes=[102.0, np.nan, np.nan],
                   rets=[0.01, 0.01, -0.40],
                   delflg=["", "", "Y"])
    r = L.compute_label(sec, T, cal)
    assert r.status == L.STATUS_OK
    assert r.delist_takeover and r.delist_date == H[2]
    assert r.label == pytest.approx(1.02 * 1.01 * 0.60 - 1.0, rel=1e-12)
    assert not r.imputed


def test_delist_missing_ret_shumway_imputation(cal):
    sec = make_sec([H[0], H[1], H[2]],
                   opens=[100.0, np.nan, np.nan],
                   closes=[102.0, np.nan, np.nan],
                   rets=[0.01, 0.01, np.nan],
                   delflg=["", "", "Y"],
                   DelReasonType=["", "", "PERF"])
    is_perf = lambda row: row.get("DelReasonType") == "PERF"

    # 无插补档位 → invalid
    r0 = L.compute_label(sec, T, cal)
    assert r0.status == L.STATUS_INVALID and r0.reason == L.REASON_MISSING_DELIST_RET

    # 业绩类 + 插补 −30% → 用插补值，imputed=True
    r1 = L.compute_label(sec, T, cal, delist_imputation=-0.30, is_performance_delist=is_perf)
    assert r1.status == L.STATUS_OK and r1.imputed
    assert r1.label == pytest.approx(1.02 * 1.01 * 0.70 - 1.0, rel=1e-12)

    # 插补只发生在业绩类退市码上（§10 不变量）：非业绩类 → 仍 invalid
    r2 = L.compute_label(sec, T, cal, delist_imputation=-0.30,
                         is_performance_delist=lambda row: False)
    assert r2.status == L.STATUS_INVALID and r2.reason == L.REASON_MISSING_DELIST_RET


def test_delist_on_exit_day(cal):
    # 退市发生在 t+6：复合该记录 DlyRet，不再要求 t+6 open（§4 第 5 条）
    dates = [H[0], H[1], H[2], H[3], H[4], H[5]]
    sec = make_sec(dates,
                   opens=[100.0] + [np.nan] * 5,
                   closes=[102.0, np.nan, np.nan, np.nan, 110.0, np.nan],
                   rets=[0.01, 0.01, 0.02, -0.01, 0.005, -0.20],
                   delflg=["", "", "", "", "", "Y"])
    r = L.compute_label(sec, T, cal)
    assert r.status == L.STATUS_OK and r.delist_takeover and r.delist_date == H[5]
    expected = 1.02 * 1.01 * 1.02 * 0.99 * 1.005 * 0.80 - 1.0
    assert r.label == pytest.approx(expected, rel=1e-12)


def test_unfillable_halted_vs_delisted(cal):
    # 停牌：t+1 行存在但 open 无效，之后还有记录 → halted
    sec_halt = make_sec([H[0], H[1]], opens=[np.nan, 100.0], closes=[np.nan, 101.0], rets=[np.nan, 0.01])
    r = L.compute_label(sec_halt, T, cal)
    assert r.status == L.STATUS_UNFILLABLE and r.reason == L.REASON_HALTED

    # 退市：最后记录（退市终值行）在 t 当日，t+1 已无行 → delisted
    sec_del = make_sec([T], opens=[np.nan], closes=[np.nan], rets=[-0.5], delflg=["Y"])
    r = L.compute_label(sec_del, T, cal)
    assert r.status == L.STATUS_UNFILLABLE and r.reason == L.REASON_DELISTED

    # 记录早已结束（无退市行残留）也归 delisted
    sec_gone = make_sec(["2020-02-03"], opens=[10.0], closes=[10.0], rets=[0.0])
    r = L.compute_label(sec_gone, T, cal)
    assert r.status == L.STATUS_UNFILLABLE and r.reason == L.REASON_DELISTED


def test_missing_mid_ret_invalid_not_filled(cal):
    # t+3 整行缺失（长停牌）且非退市 → INVALID，禁止静默填补
    dates = [H[0], H[1], H[3], H[4], H[5]]  # 缺 H[2]
    sec = make_sec(dates,
                   opens=[100.0, np.nan, np.nan, np.nan, 111.0],
                   closes=[102.0, np.nan, np.nan, 110.0, 112.0],
                   rets=[0.01, 0.01, -0.01, 0.005, 0.009])
    r = L.compute_label(sec, T, cal)
    assert r.status == L.STATUS_INVALID and r.reason == L.REASON_MISSING_MID_RET
    assert np.isnan(r.label)


def test_missing_exit_open_invalid(cal):
    sec = normal_sec()
    sec.loc[H[5], "DlyOpen"] = np.nan
    r = L.compute_label(sec, T, cal)
    assert r.status == L.STATUS_INVALID and r.reason == L.REASON_MISSING_EXIT_PRICE


def test_calendar_end_invalid(cal):
    t_late = cal.dates[-3]  # t+6 超出日历
    r = L.compute_label(normal_sec(), t_late, cal)
    assert r.status == L.STATUS_INVALID and r.reason == L.REASON_CALENDAR_END


def test_batch_and_report(cal):
    sec = normal_sec().reset_index(names="DlyCalDt")
    sec["PERMNO"] = 10001
    gone = make_sec(["2020-02-03"], opens=[10.0], closes=[10.0], rets=[0.0]).reset_index(names="DlyCalDt")
    gone["PERMNO"] = 10002
    panel = pd.concat([sec, gone], ignore_index=True)

    obs = pd.DataFrame({"PERMNO": [10001, 10002], "signal_date": [T, T]})
    divs = pd.DataFrame({"PERMNO": [10001], "ex_date": [H[5]], "div_cash": [0.5]})

    out = L.compute_labels(panel, obs, cal, cash_dividends=divs)
    assert len(out) == 2
    ok = out[out.PERMNO == 10001].iloc[0]
    expected = (102.0 / 100.0) * 1.01 * 1.02 * 0.99 * 1.005 * ((111.0 + 0.5) / 110.0) - 1.0
    assert ok.status == L.STATUS_OK and ok.label == pytest.approx(expected, rel=1e-12)
    bad = out[out.PERMNO == 10002].iloc[0]
    assert bad.status == L.STATUS_UNFILLABLE and bad.reason == L.REASON_DELISTED

    rep = L.unfillable_report(out)
    assert len(rep) == 1
    assert rep.iloc[0]["share"] == pytest.approx(0.5)
