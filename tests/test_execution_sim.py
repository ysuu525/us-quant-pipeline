"""执行层模拟（§7.5/§8）：t+1 成交、缓冲区替换、强制卖出、退市转现金、噪声带。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline.costs import CHANNELS, FeeSchedule, RegulatoryFees
from crsp_pipeline.execution_sim import (
    excess_total_return,
    max_drawdown,
    selection_noise_band,
    simulate_portfolio,
)

FREE = CHANNELS["alpaca"]
FIXED = CHANNELS["moomoo_au_whole"]
NO_REG = RegulatoryFees(0.0, 0.0, 0.0)


def _dates(n):
    return pd.bdate_range("2020-01-06", periods=n)


def _scores(date, d):
    return pd.DataFrame({
        "signal_date": date, "PERMNO": list(d), "score": list(d.values()),
    })


def _returns(dates, cols):
    return pd.DataFrame(cols, index=dates)


def test_buy_and_hold_next_day_execution():
    dates = _dates(20)
    rets = _returns(dates, {1: [0.01] * 20, 2: [0.0] * 20})
    sc = _scores(dates[0], {1: 0.9, 2: 0.1})
    r = simulate_portfolio(sc, rets, [dates[0]], n_positions=1, schedule=FREE,
                           slippage_bp=0, regulatory=NO_REG)
    # 信号日 d0 净值仍是现金；d1 起建仓，吃 d1..d19 共 19 天收益
    assert r.nav.iloc[0] == pytest.approx(3000.0)
    assert r.nav.iloc[-1] == pytest.approx(3000.0 * 1.01 ** 19, rel=1e-12)
    assert len(r.orders) == 1 and r.orders.iloc[0]["side"] == "buy"


def test_fixed_fee_reduces_position():
    dates = _dates(5)
    rets = _returns(dates, {1: [0.0] * 5})
    sc = _scores(dates[0], {1: 1.0})
    r = simulate_portfolio(sc, rets, [dates[0]], n_positions=1, schedule=FIXED,
                           slippage_bp=0, regulatory=NO_REG)
    assert r.nav.iloc[-1] == pytest.approx(3000.0 - 0.99)
    assert r.cost_totals["commission"] == pytest.approx(0.99)


def test_buffer_keeps_holding_and_threshold_gates_replacement():
    dates = _dates(15)
    rets = _returns(dates, {1: [0.0] * 15, 2: [0.0] * 15, 3: [0.0] * 15})
    sc = pd.concat([
        _scores(dates[0], {1: 0.9, 2: 0.5, 3: 0.1}),
        _scores(dates[7], {2: 0.9, 1: 0.5, 3: 0.1}),  # 持仓 1 跌到第 2 名
    ], ignore_index=True)

    # buffer_rank=2：名次 2 仍在缓冲区 → 不换
    r1 = simulate_portfolio(sc, rets, [dates[0], dates[7]], n_positions=1,
                            schedule=FREE, buffer_rank=2, slippage_bp=0, regulatory=NO_REG)
    assert len(r1.orders) == 1

    # buffer_rank=1 + 阈值 0.6：改善 0.4 不够 → 不换
    r2 = simulate_portfolio(sc, rets, [dates[0], dates[7]], n_positions=1,
                            schedule=FREE, buffer_rank=1,
                            score_improvement_threshold=0.6,
                            slippage_bp=0, regulatory=NO_REG)
    assert len(r2.orders) == 1

    # buffer_rank=1 + 阈值 0.1：改善 0.4 触发 → 卖 1 买 2（换一只 = 两单）
    r3 = simulate_portfolio(sc, rets, [dates[0], dates[7]], n_positions=1,
                            schedule=FREE, buffer_rank=1,
                            score_improvement_threshold=0.1,
                            slippage_bp=0, regulatory=NO_REG)
    assert len(r3.orders) == 3
    sides = r3.orders["side"].tolist()
    assert sides.count("sell") == 1 and sides.count("buy") == 2
    assert set(r3.final_positions) == {2}


def test_forced_sell_when_out_of_universe():
    dates = _dates(15)
    rets = _returns(dates, {1: [0.0] * 15, 2: [0.0] * 15})
    sc = pd.concat([
        _scores(dates[0], {1: 0.9, 2: 0.1}),
        _scores(dates[7], {2: 0.5}),  # 1 出 universe
    ], ignore_index=True)
    r = simulate_portfolio(sc, rets, [dates[0], dates[7]], n_positions=1,
                           schedule=FREE, slippage_bp=0, regulatory=NO_REG)
    assert set(r.final_positions) == {2}


def test_delisted_position_converts_to_cash():
    dates = _dates(12)
    a = [0.01] * 4 + [-0.5] + [np.nan] * 7  # 下标 4 为退市终值（DlyRet 已含退市收益）
    rets = _returns(dates, {1: a, 2: [0.0] * 12})
    sc = _scores(dates[0], {1: 0.9, 2: 0.1})
    r = simulate_portfolio(sc, rets, [dates[0]], n_positions=1, schedule=FREE,
                           slippage_bp=0, regulatory=NO_REG)
    # d1 建仓 → 吃下标 1..3 三天 +1%，再吃下标 4 的退市收益 −50%
    expected = 3000.0 * (1.01 ** 3) * 0.5
    assert r.nav.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert r.final_positions == {}  # 已转现金
    assert (r.nav.iloc[6:] == r.nav.iloc[-1]).all()  # 退市后净值不再变动


def test_max_drawdown_and_excess():
    nav = pd.Series([1.0, 1.2, 0.6, 0.9])
    assert max_drawdown(nav) == pytest.approx(0.5)
    nav2 = pd.Series([1.0, 2.0], index=_dates(2))
    mkt = pd.Series([0.0, 0.0], index=_dates(2))
    assert excess_total_return(nav2, mkt) == pytest.approx(1.0)


def test_noise_band_shapes_and_reproducibility():
    dates = _dates(30)
    rng = np.random.default_rng(3)
    cols = {i: rng.normal(0.001, 0.02, 30) for i in range(1, 7)}
    rets = _returns(dates, cols)
    sc = pd.concat([
        _scores(dates[0], {i: 7 - i for i in range(1, 7)}),
        _scores(dates[10], {i: i for i in range(1, 7)}),
        _scores(dates[20], {i: 7 - i for i in range(1, 7)}),
    ], ignore_index=True)
    kw = dict(daily_returns=rets, rebalance_dates=[dates[0], dates[10], dates[20]],
              n_positions=2, schedule=FREE, slippage_bp=0, regulatory=NO_REG)

    band = selection_noise_band(sc, n_draws=15, rank_jitter=3.0, seed=42, **kw)
    assert band["paths"].shape == (30, 15)
    assert list(band["band"].columns) == [0.05, 0.5, 0.95]
    assert 0.0 <= band["actual_final_percentile"] <= 1.0
    assert isinstance(band["median_positive"], bool)

    band2 = selection_noise_band(sc, n_draws=15, rank_jitter=3.0, seed=42, **kw)
    pd.testing.assert_frame_equal(band["paths"], band2["paths"])
