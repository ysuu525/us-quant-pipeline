"""exp15 执行器：折号白名单、`rdq_hat'` 的 +364 / 顺延 / 时点纪律、事件窗、无未来收益变量。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "exp15_announcer_tilt_dev_estimate",
    REPO / "scripts" / "exp15_announcer_tilt_dev_estimate.py")
EXP15 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EXP15)

_SPEC13 = importlib.util.spec_from_file_location(
    "exp15_test_exp13", REPO / "scripts" / "exp13_compustat_dev_diag.py")
EXP13 = importlib.util.module_from_spec(_SPEC13)
_SPEC13.loader.exec_module(EXP13)

_SPEC14 = importlib.util.spec_from_file_location(
    "exp15_test_exp14", REPO / "scripts" / "exp14_rule_e_dev_estimate.py")
EXP14 = importlib.util.module_from_spec(_SPEC14)
_SPEC14.loader.exec_module(EXP14)

#: 合成交易日历（工作日，含周末缺口，用来测「顺延」）
CALENDAR = pd.DatetimeIndex(pd.bdate_range("2019-01-01", periods=1600))

FUTURE_RETURN_COLUMN = "label"


# --------------------------------------------------------------------------
# §2.1 折号白名单
# --------------------------------------------------------------------------
def test_fold_whitelist_accepts_only_36_to_42():
    assert EXP15.assert_folds(range(36, 43)) == tuple(range(36, 43))
    assert EXP15.ALLOWED_FOLDS_EXP15 == frozenset(range(36, 43))
    for bad in ([5], [35], [44], [45], [36, 44], [1, 2, 3, 4], list(range(5, 36))):
        with pytest.raises(EXP15.FoldWhitelistError):
            EXP15.assert_folds(bad)
    with pytest.raises(EXP15.FoldWhitelistError):
        EXP15.assert_folds([])


def test_scores_path_guard_rejects_sealed_folds():
    from signals.kronos_adapter import ALLOWED_FOLDS, SEALED_FOLDS, scores_path

    assert EXP15.ALLOWED_FOLDS_EXP15 <= ALLOWED_FOLDS
    assert not (EXP15.ALLOWED_FOLDS_EXP15 & SEALED_FOLDS)
    for fold in (5, 20, 35, 44, 45):
        with pytest.raises(Exception):
            scores_path(fold, "ft")


def test_frozen_constants_match_the_task_doc():
    assert EXP15.HAT_OFFSET_DAYS == 364 and EXP15.EXP14_HAT_OFFSET_DAYS == 365
    assert EXP15.HOLDING_DAYS == 6 and EXP15.SLACK_DAYS == 1
    assert EXP15.EVENT_WINDOW == (-1, +1)
    assert EXP15.W_EVENT_MAIN == 0.10
    assert EXP15.W_EVENT_SENSITIVITY == (0.05, 0.20)
    assert EXP15.CALENDARS == ("rdq_hat364", "rdq_real")
    assert EXP15.EX_POST_CALENDARS == frozenset({"rdq_real"})
    assert EXP15.TOPN == 500 and EXP15.EXIT_PCT == 0.30 and EXP15.MIN_NAMES == 50
    assert EXP15.DRIFT_HORIZONS == (1, 3, 5)
    assert EXP15.TASK_DOC_SHA256 == (
        "6f961db688861e9bc4a7914e0305da12d67771542293bb2e2f1516e6d8564484")


# --------------------------------------------------------------------------
# §3 的 +364 / 顺延
# --------------------------------------------------------------------------
def test_roll_to_trading_day_is_identity_on_trading_days():
    sample = CALENDAR[[0, 5, 100, 500]]
    got = EXP15.roll_to_trading_day(CALENDAR, pd.Series(sample))
    assert list(pd.DatetimeIndex(got)) == list(sample)


def test_roll_to_trading_day_moves_weekends_forward_never_backward():
    friday = pd.Timestamp("2021-01-08")
    assert friday in set(CALENDAR)
    saturday, sunday = friday + pd.Timedelta(days=1), friday + pd.Timedelta(days=2)
    monday = friday + pd.Timedelta(days=3)
    got = EXP15.roll_to_trading_day(CALENDAR, pd.Series([saturday, sunday]))
    assert [pd.Timestamp(x) for x in got] == [monday, monday]
    # 顺延后一定 >= 原日期，且一定落在日历上
    probe = pd.Series(pd.date_range("2021-01-01", periods=60, freq="D"))
    rolled = pd.DatetimeIndex(EXP15.roll_to_trading_day(CALENDAR, probe))
    assert (rolled >= pd.DatetimeIndex(probe)).all()
    assert set(rolled) <= set(CALENDAR)


def _quarterly_frame(rdq_dates) -> pd.DataFrame:
    """一个 gvkey、12 个连续财季；`rdq` 由调用方给定（照 exp14 测试的合成表形状）。"""
    rows = []
    for i, day in enumerate(rdq_dates):
        rows.append({"gvkey": "001004", "fyearq": 2019 + i // 4, "fqtr": i % 4 + 1,
                     "datadate": pd.Timestamp(day) - pd.Timedelta(days=40),
                     "rdq": pd.Timestamp(day), "fyr": 12, "cusip": "000000000",
                     "curcdq": "USD", "epspxq": 1.0 + 0.1 * i, "ajexq": 1.0,
                     "dedup_group_rows": 1})
    return pd.DataFrame(rows)


def _permno_quarters(rdq_dates) -> pd.DataFrame:
    quarters, _ = EXP13.build_quarter_features(_quarterly_frame(rdq_dates))
    link = pd.DataFrame({"gvkey": ["001004"], "permno": [10001]})
    return EXP13.broadcast_to_permno(quarters, link)


#: 每季固定在星期三公告（+364 天 = 同星期几；+365 天会平移一天）
RDQ_DATES = [pd.Timestamp("2019-02-06") + pd.Timedelta(days=91 * i) for i in range(12)]


def test_hat364_is_exactly_the_source_plus_364_days_then_rolled():
    pq = EXP15.with_hat364(_permno_quarters(RDQ_DATES), CALENDAR)
    has = pq["rdq_source"].notna()
    assert has.sum() == 8                       # 前 4 个财季没有上一年同季
    raw = pq.loc[has, "rdq_source"] + pd.Timedelta(days=364)
    assert (pq.loc[has, "rdq_hat364_raw"] == raw).all()
    # +364 天 = 52 周 ⇒ 星期几不变
    assert (pq.loc[has, "rdq_hat364_raw"].dt.dayofweek
            == pq.loc[has, "rdq_source"].dt.dayofweek).all()
    rolled = pd.DatetimeIndex(pq.loc[has, "rdq_hat364"])
    assert (rolled >= pd.DatetimeIndex(pq.loc[has, "rdq_hat364_raw"])).all()
    assert set(rolled) <= set(CALENDAR)
    # 对照：exp13/exp14 的 +365（DateOffset(years=1)）会平移星期几
    assert (pq.loc[has, "rdq_hat"] == pq.loc[has, "rdq_source"]
            + pd.DateOffset(years=1)).all()


def test_hat364_never_uses_an_announcement_not_yet_public():
    """§3 的时点纪律：`assign_point_in_time` 把 `rdq_source >= signal_date` 的格置空。"""
    pq = EXP15.with_hat364(_permno_quarters(RDQ_DATES), CALENDAR)
    pq_hat = pq.copy()
    pq_hat["rdq_hat"] = pq["rdq_hat364"]
    probes = CALENDAR[(CALENDAR >= "2019-06-03") & (CALENDAR <= "2021-06-30")][::5]
    keys = pd.DataFrame({"PERMNO": [10001] * len(probes), "signal_date": probes})
    matched, _ = EXP13.assign_point_in_time(keys, pq_hat, CALENDAR)
    lookup = pq.dropna(subset=["rdq_hat364"]).set_index("rdq_hat364")["rdq_source"]
    picked = matched.dropna(subset=["rdq_hat"])
    assert len(picked) > 0
    for _, row in picked.iterrows():
        source = lookup.loc[row["rdq_hat"]]
        source = source.iloc[0] if isinstance(source, pd.Series) else source
        assert source < row["signal_date"]              # 严格早于信号日
        assert row["rdq_hat"] >= row["signal_date"]     # 只取未来的预测日


def test_alignment_accuracy_compares_both_rules_on_the_same_sample():
    events = pd.DataFrame({
        "rdq": pd.to_datetime(["2021-02-04", "2021-02-05", "2021-02-08"]),
        "rdq_hat": pd.to_datetime(["2021-02-05", "2021-02-05", "2021-02-06"]),
        "rdq_hat364": pd.to_datetime(["2021-02-04", "2021-02-04", "2021-02-08"]),
        "rdq_hat364_raw": pd.to_datetime(["2021-02-04", "2021-02-04", "2021-02-06"]),
    })
    got = EXP15.alignment_accuracy(events, CALENDAR)
    assert got["n_events"] == 3
    # +365 的 err = −1 / 0 / +2 ⇒ |err|<=0 命中 1 个、<=1 命中 2 个、<=2 命中 3 个
    assert got["plus365_exp14"]["abs_err_cum_share"]["0"] == pytest.approx(1 / 3)
    assert got["plus365_exp14"]["abs_err_cum_share"]["1"] == pytest.approx(2 / 3)
    assert got["plus365_exp14"]["abs_err_cum_share"]["2"] == pytest.approx(1.0)
    # +364 顺延后的 err = 0 / +1 / 0 ⇒ |err|<=0 命中 2 个
    assert got["plus364_rolled"]["abs_err_cum_share"]["0"] == pytest.approx(2 / 3)
    assert got["plus364_rolled"]["abs_err_cum_share"]["1"] == pytest.approx(1.0)
    # 2021-02-06 是周六 ⇒ 未顺延规则有 1/3 落在非交易日；顺延后为 0
    assert got["plus364_raw"]["non_trading_day_share"] == pytest.approx(1 / 3)
    assert got["plus364_rolled"]["non_trading_day_share"] == pytest.approx(0.0)
    assert got["rdq_real_non_trading_day_share"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# §4.2 事件窗与筛选
# --------------------------------------------------------------------------
def test_build_event_windows_uses_adjacent_trading_days():
    events = pd.DataFrame({"PERMNO": [1, 2],
                           "event_date": [CALENDAR[100], CALENDAR[500]]})
    got = EXP15.build_event_windows(events, CALENDAR, event_col="event_date")
    assert list(got["d_entry"]) == [CALENDAR[99], CALENDAR[499]]
    assert list(got["d_event"]) == [CALENDAR[100], CALENDAR[500]]
    assert list(got["d_exit"]) == [CALENDAR[101], CALENDAR[501]]


def test_build_event_windows_drops_non_trading_and_boundary_events():
    saturday = pd.Timestamp("2021-01-09")
    assert saturday not in set(CALENDAR)
    events = pd.DataFrame({"PERMNO": [1, 2, 3, 4],
                           "event_date": [saturday, CALENDAR[0], CALENDAR[-1],
                                          CALENDAR[300]]})
    got = EXP15.build_event_windows(events, CALENDAR, event_col="event_date")
    assert list(got["PERMNO"]) == [4]


def test_screen_events_keeps_only_pool_members_and_applies_the_point_in_time_gate():
    entry, event, exit_ = CALENDAR[200], CALENDAR[201], CALENDAR[202]
    signal_day = CALENDAR[199]
    pool = {entry: frozenset({10, 11})}
    signal_of = {entry: signal_day}
    events = pd.DataFrame({
        "PERMNO": [10, 12, 11, 10],
        "rdq_source": [signal_day - pd.Timedelta(days=1), signal_day, signal_day,
                       signal_day + pd.Timedelta(days=1)],
        "d_entry": [entry, entry, entry, CALENDAR[900]],
        "d_event": [event, event, event, CALENDAR[901]],
        "d_exit": [exit_, exit_, exit_, CALENDAR[902]]})
    kept, stats = EXP15.screen_events_to_pool(
        events, pool, signal_of, require_source_before_signal=True)
    assert stats["n_in"] == 4
    assert stats["n_entry_day_in_fold_index"] == 3      # 第 4 行进场日不在折内
    assert stats["n_in_pool"] == 2                      # PERMNO 12 不在池里
    assert stats["n_voided_by_pit"] == 1                # rdq_source == 信号日 ⇒ 不可用
    assert list(kept["PERMNO"]) == [10]
    kept2, stats2 = EXP15.screen_events_to_pool(
        events, pool, signal_of, require_source_before_signal=False)
    assert stats2["n_voided_by_pit"] == 0
    assert sorted(kept2["PERMNO"]) == [10, 11]


# --------------------------------------------------------------------------
# 诊断 D 与统计
# --------------------------------------------------------------------------
def test_drift_diagnostics_computes_r0_and_relative_cumulative_returns():
    days = list(CALENDAR[100:110])
    ret = {day: {10: 0.02, 11: -0.01, 12: 0.0} for day in days}
    bench = {day: 0.005 for day in days}
    pool = {day: frozenset({10, 11, 12}) for day in days}
    events = pd.DataFrame({"PERMNO": [10], "d_entry": [days[0]], "d_event": [days[1]],
                           "d_exit": [days[2]], "fold": [36]})
    got = EXP15.drift_diagnostics(events, ret, bench, pool, CALENDAR)
    assert got.loc[0, "r0"] == pytest.approx(0.02)
    assert got.loc[0, "cr1"] == pytest.approx(1.02 - 1.005)
    assert got.loc[0, "cr3"] == pytest.approx(1.02 ** 3 - 1.005 ** 3)
    assert got.loc[0, "cr5"] == pytest.approx(1.02 ** 5 - 1.005 ** 5)
    # 池内截面百分位：0.02 是三只里最高 ⇒ 1.0
    assert got.loc[0, "r0_pool_pct"] == pytest.approx(1.0)


def test_drift_diagnostics_drops_horizons_that_run_off_the_index():
    days = list(CALENDAR[100:104])
    ret = {day: {10: 0.01} for day in CALENDAR[100:110]}
    bench = {day: 0.0 for day in days}          # 只有 4 天有基准
    pool = {day: frozenset({10}) for day in days}
    events = pd.DataFrame({"PERMNO": [10], "d_entry": [days[0]], "d_event": [days[1]],
                           "d_exit": [days[2]], "fold": [36]})
    got = EXP15.drift_diagnostics(events, ret, bench, pool, CALENDAR)
    assert "cr1" in got.columns and np.isfinite(got.loc[0, "cr1"])
    assert "cr5" not in got.columns or not np.isfinite(got.loc[0, "cr5"])


def test_group_series_stats_uses_the_daily_cross_sectional_mean_as_the_unit():
    rng = np.random.default_rng(4)
    days = list(CALENDAR[:120])
    rows = []
    for fold, chunk in enumerate(np.array_split(np.arange(len(days)), 7), start=36):
        for i in chunk:
            for _ in range(3):
                rows.append({"d_event": days[i], "fold": fold,
                             "value": float(rng.normal(0.001, 0.01))})
    frame = pd.DataFrame(rows)
    got = EXP15.group_series_stats(frame, "value", EXP14)
    assert got["n_events"] == len(frame)
    assert got["n_days"] == len(days)            # 推断单位 = 逐日截面均值，不是逐事件
    assert got["folds_total"] == 7
    manual = frame.groupby(["fold", "d_event"])["value"].mean()
    assert got["mean_bp"] == pytest.approx(float(manual.mean() * 1e4))


def test_quintile_bins_split_a_day_into_five_groups():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    got = EXP15.quintile_bins(values)
    assert list(got) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert EXP15.quintile_bins(pd.Series([1.0, 2.0])).isna().all()


def test_extrapolate_se_uses_the_ledger_square_root_rule():
    assert EXP15.FOLDS31_TRADING_DAYS == 3900
    got = EXP15.extrapolate_se_31folds(1.0, 839)
    assert got == pytest.approx(np.sqrt(839 / 3900))
    assert not np.isfinite(EXP15.extrapolate_se_31folds(float("nan"), 839))
    assert not np.isfinite(EXP15.extrapolate_se_31folds(1.0, 0))


def test_config_names_cover_the_eight_cells_plus_baseline():
    names = [EXP15.BASELINE_CONFIG]
    for calendar_name in EXP15.CALENDARS:
        names.append(EXP15._b1_name(calendar_name))
        for w in (EXP15.W_EVENT_MAIN, *EXP15.W_EVENT_SENSITIVITY):
            names.append(EXP15._b2_name(calendar_name, w))
    assert len(names) == 9                       # E0 + B1×2 + B2×2×3
    assert EXP15._b2_name("rdq_hat364", 0.10) == "B2|rdq_hat364|w10"
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------
# 前视自查：全脚本无未来收益变量
# --------------------------------------------------------------------------
@pytest.mark.parametrize("relative", [
    "src/portfolio/construction_announcer.py",
    "scripts/exp15_smoke.py",
    "tests/test_exp15_construction_announcer.py",
])
def test_exp15_sources_never_mention_the_future_return_variable(relative):
    text = (REPO / relative).read_text(encoding="utf-8")
    assert FUTURE_RETURN_COLUMN not in text


def test_executor_code_outside_the_transcribed_task_doc_is_clean():
    """模块 docstring 是任务书 §2–§5 的逐字抄录（任务书 §0 要求），其中 §2.3 这条
    禁令本身就写着那个列名——那是禁令原文，不是数据引用。本测试把模块 docstring
    整段剥掉之后，断言执行器代码区 0 命中。"""
    import ast

    path = REPO / "scripts" / "exp15_announcer_tilt_dev_estimate.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc = ast.get_docstring(tree, clean=False) or ""
    assert FUTURE_RETURN_COLUMN in doc, "任务书 §2.3 原文应当在 docstring 里"
    body = source.replace(doc, "", 1)
    assert FUTURE_RETURN_COLUMN not in body
