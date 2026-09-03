"""实验 2 的加权构造只用合成数据测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.construction import (
    _weight_turnover,
    frozen_long_only_returns,
    frozen_long_only_returns_weighted,
)


def _synthetic(seed: int):
    rng = np.random.default_rng(seed)
    names = np.arange(10001, 10081, dtype=np.int64)
    score_days = pd.bdate_range("2022-01-03", periods=24)
    price_days = pd.bdate_range("2022-01-03", periods=25)
    scores = {
        day: dict(zip(names, np.round(rng.normal(size=len(names)), 2)))
        for day in score_days
    }
    ret = {
        day: dict(zip(names, rng.normal(0.0, 0.02, size=len(names))))
        for day in price_days
    }
    oc = {
        day: dict(zip(names, rng.normal(0.0, 0.015, size=len(names))))
        for day in price_days
    }
    adv = {
        day: dict(zip(names, np.round(rng.lognormal(15.0, 1.0, len(names)), -3)))
        for day in price_days
    }
    return scores, ret, oc, adv


@pytest.mark.parametrize("seed", [20260904, 7, 424242])
def test_equal_weighting_is_bitwise_identical(seed):
    scores, ret, oc, adv = _synthetic(seed)
    expected = frozen_long_only_returns(
        scores, ret, oc, adv, topn=500, cost_bp=8.0, exit_pct=0.30, nt=5
    )
    actual = frozen_long_only_returns_weighted(
        scores,
        ret,
        oc,
        adv,
        topn=500,
        cost_bp=8.0,
        exit_pct=0.30,
        nt=5,
        weighting="equal",
    )
    assert list(actual.index) == list(expected.index)
    assert list(actual.columns) == list(expected.columns)
    assert np.array_equal(actual.to_numpy(), expected.to_numpy())
    assert np.array_equal(actual["turn"].to_numpy(), expected["turn"].to_numpy())


def test_equal_weight_turnover_reduces_to_original_definition():
    old_names = (1, 2, 3, 4, 5)
    new_names = (1, 2, 3, 6, 7)
    old = {p: 1.0 / len(old_names) for p in old_names}
    new = {p: 1.0 / len(new_names) for p in new_names}
    assert _weight_turnover(old, new) == pytest.approx(2 / 5)


def test_rank_weighting_runs_and_turnover_is_bounded():
    scores, ret, oc, adv = _synthetic(20260904)
    out = frozen_long_only_returns_weighted(
        scores, ret, oc, adv, cost_bp=0.0, nt=5, weighting="rank"
    )
    assert list(out.columns) == ["r", "turn", "n_names"]
    assert len(out) > 10
    assert np.isfinite(out["r"]).all()
    assert out["turn"].between(0.0, 1.0).all()


def test_unknown_weighting_is_rejected():
    scores, ret, oc, adv = _synthetic(20260904)
    with pytest.raises(ValueError, match="weighting"):
        frozen_long_only_returns_weighted(
            scores, ret, oc, adv, weighting="score"
        )
