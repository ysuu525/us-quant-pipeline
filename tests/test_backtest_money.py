"""`backtest.money` 薄包装的冒烟测试。

这一层**不含任何新统计量**，所以测试只核两件事：
1. 包装没有偷偷改口径（与被包装函数直接调用的结果完全一致）；
2. 拼表 / 过滤 / 计数这些管道动作是对的。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.money import daily_rank_ic, fold_sign_count, nw_summary, run_frozen_money
from crsp_pipeline.signal_eval import newey_west_tstat
from portfolio.construction import frozen_long_only_returns, scores_frame_to_by_day


def _long(day_to_scores: dict, col: str) -> pd.DataFrame:
    rows = []
    for d, mapping in day_to_scores.items():
        for p, v in mapping.items():
            rows.append((pd.Timestamp(d), np.int64(p), float(v)))
    return pd.DataFrame(rows, columns=["signal_date", "PERMNO", col])


# ---------------------------------------------------------------- daily_rank_ic

def test_daily_rank_ic_known_cases():
    scores = _long({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
                    "2021-01-05": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}}, "score")
    labels = _long({"2021-01-04": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
                    "2021-01-05": {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}}, "label")
    ic = daily_rank_ic(scores, labels)
    assert ic.loc[pd.Timestamp("2021-01-04")] == pytest.approx(1.0)
    assert ic.loc[pd.Timestamp("2021-01-05")] == pytest.approx(-1.0)


def test_daily_rank_ic_inner_join_drops_unmatched():
    scores = _long({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0},
                    "2021-01-05": {1: 1.0, 2: 2.0, 3: 3.0}}, "score")
    labels = _long({"2021-01-04": {1: 0.1, 2: 0.2, 3: 0.3}}, "label")
    ic = daily_rank_ic(scores, labels)
    assert list(ic.index) == [pd.Timestamp("2021-01-04")]


def test_daily_rank_ic_accepts_single_frame():
    df = _long({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0}}, "score")
    df["label"] = [0.3, 0.2, 0.1]
    ic = daily_rank_ic(df)
    assert ic.iloc[0] == pytest.approx(-1.0)


def test_daily_rank_ic_universe_mask_mapping():
    """逐日允许名单——合成实验里就是 `combine.slow_filter` 的可持有集。"""
    scores = _long({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}}, "score")
    labels = _long({"2021-01-04": {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}}, "label")
    full = daily_rank_ic(scores, labels)
    assert full.iloc[0] == pytest.approx(-1.0)
    # 只留 3 个名字，label 顺序在子集里仍然完全相反 → 还是 -1
    mask = {pd.Timestamp("2021-01-04"): {1, 2, 3}}
    sub = daily_rank_ic(scores, labels, universe_mask=mask)
    assert sub.iloc[0] == pytest.approx(-1.0)
    # 只留 2 个名字 → 共同名字 < 3，_spearman 约定返回 NaN
    mask2 = {pd.Timestamp("2021-01-04"): {1, 2}}
    assert np.isnan(daily_rank_ic(scores, labels, universe_mask=mask2).iloc[0])


def test_daily_rank_ic_universe_mask_boolean_length_check():
    scores = _long({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0}}, "score")
    labels = _long({"2021-01-04": {1: 0.3, 2: 0.2, 3: 0.1}}, "label")
    with pytest.raises(ValueError):
        daily_rank_ic(scores, labels, universe_mask=[True, False])


def test_daily_rank_ic_rejects_bad_labels():
    scores = _long({"2021-01-04": {1: 1.0, 2: 2.0, 3: 3.0}}, "score")
    with pytest.raises(ValueError):
        daily_rank_ic(scores, scores.rename(columns={"score": "not_label"}))


# ---------------------------------------------------------------- nw_summary

def test_nw_summary_is_pure_passthrough():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0.001, 0.01, 300))
    got = nw_summary(x, lags=5)
    want = newey_west_tstat(x, 5)
    assert set(got) == {"mean", "se", "t", "n"}
    for key in want:
        assert got[key] == want[key] or (np.isnan(got[key]) and np.isnan(want[key]))


def test_nw_summary_default_lag_is_five():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(0.0, 0.01, 200))
    assert nw_summary(x) == nw_summary(x, lags=5)


# ---------------------------------------------------------------- fold_sign_count

def test_fold_sign_count():
    d = {"fold36": pd.Series([1.0, 2.0]),          # +
         "fold37": pd.Series([-1.0, -2.0]),        # -
         "fold38": pd.Series([1.0, -0.5]),         # + (均值 +0.25)
         "fold39": pd.Series([np.nan, np.nan]),    # 无数据 → 不计
         "fold40": pd.Series([], dtype=float)}     # 空 → 不计
    assert fold_sign_count(d) == (2, 3)


def test_fold_sign_count_zero_mean_is_not_positive():
    assert fold_sign_count({"a": pd.Series([1.0, -1.0])}) == (0, 1)


def test_fold_sign_count_empty():
    assert fold_sign_count({}) == (0, 0)


# ---------------------------------------------------------------- run_frozen_money

def _tiny_fold(seed=5):
    rng = np.random.default_rng(seed)
    universe = np.arange(1, 301, dtype=np.int64)
    days = pd.bdate_range("2021-01-04", periods=30)
    parts, ret, oc, adv = [], {}, {}, {}
    for d in days:
        m = int(rng.integers(120, 251))
        names = rng.choice(universe, size=m, replace=False)
        parts.append(pd.DataFrame({"PERMNO": names, "signal_date": d,
                                   "score": np.round(rng.normal(size=m), 2)}))
        ret[d] = dict(zip(universe, rng.normal(0.0, 0.02, len(universe))))
        oc[d] = dict(zip(universe, rng.normal(0.0, 0.015, len(universe))))
        adv[d] = dict(zip(universe, np.round(rng.lognormal(15.0, 1.0, len(universe)), -3)))
    return pd.concat(parts, ignore_index=True), ret, oc, adv


def test_run_frozen_money_matches_two_step_call():
    sc, ret, oc, adv = _tiny_fold()
    one = run_frozen_money(sc, ret, oc, adv)
    two = frozen_long_only_returns(scores_frame_to_by_day(sc), ret, oc, adv)
    assert list(one.columns) == ["r", "turn", "n_names"]
    assert list(one.index) == list(two.index)
    assert np.array_equal(one["r"].to_numpy(), two["r"].to_numpy())
    assert len(one) > 10


def test_run_frozen_money_forwards_kwargs():
    sc, ret, oc, adv = _tiny_fold()
    a = run_frozen_money(sc, ret, oc, adv, cost_bp=8.0)
    b = run_frozen_money(sc, ret, oc, adv, cost_bp=22.0)
    # 成本只在换手非零的日子上有差；差额 = 2*(22-8)/1e4*turn/6
    delta = (a["r"] - b["r"]).to_numpy()
    expect = 2.0 * ((22.0 - 8.0) / 1e4) * a["turn"].to_numpy() / 6
    assert np.allclose(delta, expect)
    assert (a["turn"] > 0).any(), "合成数据应当有换手，否则这条没测到东西"
