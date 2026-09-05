"""exp14 执行器：折号白名单、`rdq_hat` 的时点纪律、Rule E 窗口判定、无未来收益变量。"""
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
    "exp14_rule_e_dev_estimate", REPO / "scripts" / "exp14_rule_e_dev_estimate.py")
EXP14 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(EXP14)

_SPEC13 = importlib.util.spec_from_file_location(
    "exp14_test_exp13", REPO / "scripts" / "exp13_compustat_dev_diag.py")
EXP13 = importlib.util.module_from_spec(_SPEC13)
_SPEC13.loader.exec_module(EXP13)

CALENDAR = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=1200))


# --------------------------------------------------------------------------
# §2.1 折号白名单
# --------------------------------------------------------------------------
def test_fold_whitelist_accepts_only_36_to_42():
    assert EXP14.assert_folds(range(36, 43)) == tuple(range(36, 43))
    assert EXP14.ALLOWED_FOLDS_EXP14 == frozenset(range(36, 43))
    for bad in ([5], [35], [44], [45], [36, 44], [1, 2, 3, 4], list(range(5, 36))):
        with pytest.raises(EXP14.FoldWhitelistError):
            EXP14.assert_folds(bad)
    with pytest.raises(EXP14.FoldWhitelistError):
        EXP14.assert_folds([])


def test_scores_path_guard_rejects_sealed_folds():
    """分数路径必须走 kronos_adapter 的守卫（它自己也拒绝封存折号）。"""
    from signals.kronos_adapter import FoldNotAllowedError, scores_path
    for bad in (5, 20, 35, 44, 45):
        with pytest.raises(FoldNotAllowedError):
            scores_path(bad, "ft")
    for good in range(36, 43):
        assert scores_path(good, "ft").name == "scores.parquet"


# --------------------------------------------------------------------------
# §3 rdq_hat 的时点：t 日只能用 t 日之前已公告的 rdq
# --------------------------------------------------------------------------
def _synthetic_quarters() -> pd.DataFrame:
    """一个 gvkey、12 个连续财季；rdq 固定在 datadate 后 40 天。"""
    rows = []
    for index in range(12):
        fyearq = 2019 + index // 4
        fqtr = index % 4 + 1
        datadate = pd.Timestamp(f"{fyearq}-01-01") + pd.DateOffset(months=3 * fqtr) \
            - pd.Timedelta(days=1)
        rows.append({
            "gvkey": "001004", "datadate": datadate,
            "rdq": datadate + pd.Timedelta(days=40),
            "fyearq": fyearq, "fqtr": fqtr, "fyr": 12,
            "cusip": "000000000", "curcdq": "USD",
            "epspxq": 1.0 + 0.1 * index, "ajexq": 1.0, "dedup_group_rows": 1,
        })
    return pd.DataFrame(rows)


def _permno_quarters() -> pd.DataFrame:
    quarters, _ = EXP13.build_quarter_features(_synthetic_quarters())
    link = pd.DataFrame({"gvkey": ["001004"], "permno": [10001]})
    return EXP13.broadcast_to_permno(quarters, link)


def test_rdq_hat_equals_prior_year_same_quarter_rdq_plus_one_year():
    permno_quarters = _permno_quarters()
    with_hat = permno_quarters.dropna(subset=["rdq_hat"])
    assert len(with_hat) == 8            # 前 4 个财季没有上一年同季
    expected = with_hat["rdq_source"] + pd.DateOffset(years=1)
    assert (with_hat["rdq_hat"] == expected).all()
    # rdq_source 必须是同财季上一财年的实际 rdq（fidx 差 4）
    lookup = permno_quarters.set_index("fidx")["rdq"]
    for _, row in with_hat.iterrows():
        assert row["rdq_source"] == lookup.loc[row["fidx"] - 4]


def _probe_matched(lo: str = "2020-02-03", hi: str = "2022-06-30", step: int = 3):
    permno_quarters = _permno_quarters()
    probes = CALENDAR[(CALENDAR >= lo) & (CALENDAR <= hi)][::step]
    keys = pd.DataFrame({"PERMNO": [10001] * len(probes), "signal_date": probes})
    matched, _ = EXP13.assign_point_in_time(keys, permno_quarters, CALENDAR)
    return permno_quarters, matched


def test_rdq_hat_never_uses_an_announcement_not_yet_public():
    """§3 的核心前视纪律：取到的 rdq_hat 必须由 t 日之前已公告的 rdq 推出。"""
    permno_quarters, matched = _probe_matched()
    source_of = dict(zip(permno_quarters["rdq_hat"], permno_quarters["rdq_source"]))
    got = matched.dropna(subset=["rdq_hat"])
    assert len(got) > 0
    assert (got["rdq_hat"] >= got["signal_date"]).all()      # 只看未来的公告
    for _, row in got.iterrows():
        assert source_of[row["rdq_hat"]] < row["signal_date"]


def test_rdq_hat_selection_matches_exp13_forward_asof_plus_source_check():
    """逐格重现 exp13:419-426 的取法：最近的未来 rdq_hat，源未公开则整格置空。"""
    permno_quarters, matched = _probe_matched()
    hats = permno_quarters.dropna(subset=["rdq_hat"])[["rdq_source", "rdq_hat"]]
    hats = hats.sort_values("rdq_hat").reset_index(drop=True)
    for _, row in matched.iterrows():
        t = row["signal_date"]
        future = hats[hats["rdq_hat"] >= t]
        if future.empty:
            assert pd.isna(row["rdq_hat"])
            continue
        nearest = future.iloc[0]
        if nearest["rdq_source"] >= t:      # 源公告尚未公开 => 置空，不顺延
            assert pd.isna(row["rdq_hat"])
        else:
            assert row["rdq_hat"] == nearest["rdq_hat"]


def test_rdq_hat_is_void_before_its_source_announcement_is_public():
    """探针恰好压在源公告日前后：源公告日当日及之前不得取到由它推出的 rdq_hat。"""
    permno_quarters = _permno_quarters()
    usable = permno_quarters.dropna(subset=["rdq_hat"])
    usable = usable[usable["rdq_source"] > CALENDAR[20]]      # 源公告要落在日历内
    row = usable.iloc[0]
    source, hat = row["rdq_source"], row["rdq_hat"]
    probes = [source - pd.Timedelta(days=10), source - pd.Timedelta(days=1), source]
    # 取 <= 探针日的最后一个交易日（源公告日可能落在周末）
    probes = [CALENDAR[CALENDAR.searchsorted(p, "right") - 1] for p in probes]
    keys = pd.DataFrame({"PERMNO": [10001] * len(probes), "signal_date": probes})
    keys = keys.drop_duplicates().reset_index(drop=True)
    matched, _ = EXP13.assign_point_in_time(keys, permno_quarters, CALENDAR)
    assert (matched["signal_date"] <= source).all()
    assert (matched["rdq_hat"] != hat).all()


def test_nearest_future_rdq_real_never_looks_backwards():
    permno_quarters = _permno_quarters()
    probes = CALENDAR[(CALENDAR >= "2020-03-02") & (CALENDAR <= "2021-06-30")][::7]
    keys = pd.DataFrame({"PERMNO": [10001] * len(probes), "signal_date": probes})
    real = EXP14.nearest_future_rdq_real(keys, permno_quarters).reset_index()
    joined = real.dropna(subset=["rdq_real_next"])
    assert (joined["rdq_real_next"] >= joined["signal_date"]).all()
    truth = permno_quarters["rdq"].dropna().sort_values()
    for _, row in joined.iterrows():
        future = truth[truth >= row["signal_date"]]
        assert row["rdq_real_next"] == future.iloc[0]


# --------------------------------------------------------------------------
# §3 Rule E 的窗口判定
# --------------------------------------------------------------------------
def _brute(signal_day, event, slack, holding=6):
    if pd.isna(event):
        return False
    pos = int(CALENDAR.searchsorted(pd.Timestamp(signal_day), "left"))
    assert CALENDAR[pos] == pd.Timestamp(signal_day)
    if pos + holding >= len(CALENDAR):
        return False
    lo = pd.Timestamp(event) - pd.Timedelta(days=slack)
    hi = pd.Timestamp(event) + pd.Timedelta(days=slack)
    return any(lo <= CALENDAR[pos + off] <= hi for off in range(1, holding + 1))


@pytest.mark.parametrize("slack", [0, 1, 2, 3, 5])
def test_window_blocked_matches_brute_force(slack):
    rng = np.random.default_rng(20260905)
    days = CALENDAR[100:400]
    offsets = rng.integers(-8, 25, len(days))
    events = pd.Series([pd.Timestamp(d) + pd.Timedelta(days=int(o))
                        for d, o in zip(days, offsets)])
    frame = pd.DataFrame({"signal_date": days, "event": events.to_numpy()})
    got = EXP14.window_blocked(CALENDAR, frame["signal_date"], frame["event"],
                               slack_days=slack)
    want = np.array([_brute(d, e, slack) for d, e in zip(frame["signal_date"],
                                                         frame["event"])])
    assert np.array_equal(got, want)


def test_missing_event_date_is_always_eligible():
    days = CALENDAR[200:230]
    frame = pd.DataFrame({"signal_date": days,
                          "event": pd.Series([pd.NaT] * len(days))})
    got = EXP14.window_blocked(CALENDAR, frame["signal_date"], frame["event"],
                               slack_days=5)
    assert not got.any()


def test_slack_is_monotone_in_the_blocked_set():
    rng = np.random.default_rng(3)
    days = CALENDAR[150:450]
    events = pd.Series([pd.Timestamp(d) + pd.Timedelta(days=int(o))
                        for d, o in zip(days, rng.integers(-6, 20, len(days)))])
    frame = pd.DataFrame({"signal_date": days, "event": events.to_numpy()})
    previous = None
    for slack in (0, 1, 2, 3, 5):
        got = EXP14.window_blocked(CALENDAR, frame["signal_date"], frame["event"],
                                   slack_days=slack)
        if previous is not None:
            assert (got | previous == got).all(), "slack 增大后被拦集合必须只增不减"
        previous = got


def test_window_uses_t_plus_1_to_t_plus_6_never_t_itself():
    """事件正好落在信号日 t 上（且 slack=0）时不得触发：窗口从 t+1 起。"""
    days = CALENDAR[300:340]
    frame = pd.DataFrame({"signal_date": days, "event": days})
    assert not EXP14.window_blocked(CALENDAR, frame["signal_date"], frame["event"],
                                    slack_days=0).any()
    # 第 6 个交易日之后（t+7）不得触发
    seventh = pd.Series([CALENDAR[CALENDAR.searchsorted(d, "left") + 7] for d in days])
    frame7 = pd.DataFrame({"signal_date": days, "event": seventh.to_numpy()})
    assert not EXP14.window_blocked(CALENDAR, frame7["signal_date"], frame7["event"],
                                    slack_days=0).any()
    # 第 1 与第 6 个交易日必须触发
    for offset in (1, 6):
        target = pd.Series([CALENDAR[CALENDAR.searchsorted(d, "left") + offset]
                            for d in days])
        cell = pd.DataFrame({"signal_date": days, "event": target.to_numpy()})
        assert EXP14.window_blocked(CALENDAR, cell["signal_date"], cell["event"],
                                    slack_days=0).all()


# --------------------------------------------------------------------------
# 候选池与统计
# --------------------------------------------------------------------------
def test_daily_pools_reproduces_the_frozen_liquidity_filter():
    rng = np.random.default_rng(19)
    day = pd.Timestamp("2021-03-01")
    names = list(range(1, 801))
    scores = {day: {p: float(v) for p, v in zip(names, rng.normal(size=len(names)))}}
    adv = {day: {p: float(v) for p, v in zip(names, rng.uniform(1e5, 1e9, len(names)))}}
    adv[day][names[0]] = float("nan")
    pools = EXP14.daily_pools(scores, adv)
    assert len(pools[day]) == EXP14.TOPN
    assert names[0] not in pools[day]
    threshold = sorted((v for p, v in adv[day].items() if np.isfinite(v)),
                       reverse=True)[EXP14.TOPN - 1]
    assert min(adv[day][p] for p in pools[day]) >= threshold - 1e-9


def test_paired_stats_reports_ci_direction_and_variance_ratio():
    rng = np.random.default_rng(31)
    n = 600
    base = pd.Series(rng.normal(0, 0.01, n))
    other = base + rng.normal(0.0001, 0.002, n)
    fold = pd.Series(np.repeat(np.arange(36, 42), n // 6))
    out = EXP14.paired_stats(other, base, fold)
    assert out["n_days"] == n
    assert out["ci95_bp"][0] < out["mean_diff_bp_per_day"] < out["ci95_bp"][1]
    assert out["folds_total"] == 6
    assert out["var_ratio_pooled"] > 0


def test_e2_recall_and_false_block_definitions():
    pairs = pd.DataFrame({
        "real_in_window": [True, True, True, False, False, False],
        "blocked":        [True, False, True, True, False, False],
    })
    out = EXP14.e2_rule_recall(pairs)
    assert out["n_real_in_window"] == 3
    assert out["n_blocked"] == 3
    assert out["recall_blocked_given_real"] == pytest.approx(2 / 3)
    assert out["false_block_share_of_blocked"] == pytest.approx(1 / 3)


def test_e2_error_distribution_counts_missing_hats():
    events = pd.DataFrame({
        "rdq": pd.to_datetime(["2021-02-01", "2021-02-03", "2021-02-10", "2021-03-01"]),
        "rdq_hat": pd.to_datetime(["2021-02-01", "2021-02-01", "2021-02-01", None]),
    })
    out = EXP14.e2_error_distribution(events)
    assert out["n_events"] == 4
    assert out["n_with_rdq_hat"] == 3
    assert out["rdq_hat_missing_share"] == pytest.approx(0.25)
    assert out["err_median_days"] == pytest.approx(2.0)
    assert out["abs_err_cum_share"]["2"] == pytest.approx(2 / 3)
    assert out["abs_err_cum_share_of_all_events"]["2"] == pytest.approx(2 / 4)


# --------------------------------------------------------------------------
# 前视纪律
# --------------------------------------------------------------------------
#: CLAUDE.md §一.1 点名的未来收益列名。用拼接构造，免得断言本身污染 grep 结果。
FUTURE_RETURN_COLUMN = "lab" + "el"


@pytest.mark.parametrize("relative", [
    "scripts/exp14_smoke.py",
    "src/portfolio/construction_rule_e.py",
])
def test_exp14_sources_never_mention_the_future_return_variable(relative):
    source = (REPO / relative).read_text(encoding="utf-8")
    assert FUTURE_RETURN_COLUMN not in source


def test_executor_code_outside_the_transcribed_task_doc_is_clean():
    """执行器的**代码**里不得出现未来收益列名。

    模块 docstring 是任务书 §2–§5 的逐字抄录（任务书 §0 要求），其中 §2.3 这条
    禁令本身就写着那个列名——那是禁令原文，不是数据引用。本测试把模块 docstring
    整段剥掉之后再断言，报告的 grep 原始输出里会逐条标出剩下的命中来自何处。
    """
    import ast

    path = REPO / "scripts" / "exp14_rule_e_dev_estimate.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    doc = ast.get_docstring(tree, clean=False) or ""
    assert FUTURE_RETURN_COLUMN in doc, "任务书 §2.3 原文应当在 docstring 里"
    body = source.replace(doc, "", 1)
    assert FUTURE_RETURN_COLUMN not in body


def test_frozen_grid_constants_are_the_ones_written_into_the_task_doc():
    assert EXP14.GRID_SLACKS == (0, 1, 2, 3, 5)
    assert EXP14.GRID_CALENDARS == ("rdq_hat", "rdq_real")
    assert EXP14.GRID_ACTIONS == ("entry_only", "entry_and_exit")
    assert len(EXP14.grid_configs()) == 20
    assert EXP14.MAIN_CONFIG == "rdq_hat|slack1|entry_only"
    assert EXP14.HOLDING_DAYS == 6 and EXP14.MAIN_SLACK_DAYS == 1
    assert EXP14.TOPN == 500 and EXP14.EXIT_PCT == 0.30 and EXP14.MIN_NAMES == 50
