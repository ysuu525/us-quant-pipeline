"""信号层评估（§7.5）：RankIC、Newey-West 手算校验、十分位价差、中性化。"""

import numpy as np
import pandas as pd
import pytest

from crsp_pipeline import signal_eval as S


def _cs(date, scores, labels, **extra):
    n = len(scores)
    return pd.DataFrame({
        "signal_date": pd.Timestamp(date), "PERMNO": range(1, n + 1),
        "score": scores, "label": labels, **extra,
    })


def test_rank_ic_monotone():
    df = _cs("2020-01-06", np.arange(10.0), np.arange(10.0) * 0.01)
    assert S.daily_rank_ic(df).iloc[0] == pytest.approx(1.0)
    df2 = _cs("2020-01-06", np.arange(10.0), -np.arange(10.0))
    assert S.daily_rank_ic(df2).iloc[0] == pytest.approx(-1.0)


def test_rank_ic_zero_variance_nan():
    df = _cs("2020-01-06", np.zeros(10), np.arange(10.0))
    assert np.isnan(S.daily_rank_ic(df).iloc[0])


def test_newey_west_hand_computed():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    # lags=0 退化为经典 t（总体方差 ddof=0）：se = sqrt(1.25/4)
    r0 = S.newey_west_tstat(x, lags=0)
    assert r0["mean"] == pytest.approx(2.5)
    assert r0["se"] == pytest.approx(np.sqrt(1.25 / 4))
    assert r0["t"] == pytest.approx(2.5 / np.sqrt(1.25 / 4))
    # lags=1：var = γ0 + 2·(1/2)·γ1 = 1.25 + 0.3125 = 1.5625 → se=0.625, t=4
    r1 = S.newey_west_tstat(x, lags=1)
    assert r1["se"] == pytest.approx(0.625)
    assert r1["t"] == pytest.approx(4.0)


def test_decile_spread_golden():
    # 20 名，label = score/100：top 组均值 0.195，bottom 组 0.015 → gross 0.18
    sc = np.arange(1.0, 21.0)
    df = _cs("2020-01-06", sc, sc / 100.0)
    out = S.decile_spread(df, n_groups=10, oneway_cost_bp=30.0)
    assert out["gross"].iloc[0] == pytest.approx(0.18)
    assert out["net"].iloc[0] == pytest.approx(0.18 - 4 * 30e-4)


def test_neutralize_removes_exposure_component():
    rng = np.random.default_rng(7)
    beta = np.linspace(0.5, 1.5, 30)
    df = _cs("2020-01-06", 2.0 * beta + 3.0, rng.normal(size=30),
             beta=beta, vol=rng.uniform(0.1, 0.5, 30))
    resid = S.neutralize_scores(df)
    assert np.allclose(resid.to_numpy(), 0.0, atol=1e-10)


def test_signal_layer_report_pass_and_alarm():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-06", periods=40)

    # 场景 A：score 与 label 强相关且与暴露无关 → 通过
    frames_a, frames_b = [], []
    for d in dates:
        idio = rng.normal(size=60)
        beta = rng.normal(1.0, 0.3, 60)
        vol = rng.uniform(0.1, 0.6, 60)
        label_a = 0.05 * idio + 0.01 * rng.normal(size=60)
        frames_a.append(_cs(d, idio, label_a, beta=beta, vol=vol))
        # 场景 B：score 完全是 beta 排序 → 残差 IC 崩掉，报警且不通过
        label_b = 0.05 * beta + 0.01 * rng.normal(size=60)
        frames_b.append(_cs(d, beta, label_b, beta=beta, vol=vol))

    rep_a = S.signal_layer_report(pd.concat(frames_a, ignore_index=True), nw_lags=5)
    assert rep_a["passes"]
    assert not rep_a["attenuation_alarm"]
    assert rep_a["nw_raw"]["t"] > 2 and rep_a["nw_resid"]["t"] > 2

    rep_b = S.signal_layer_report(pd.concat(frames_b, ignore_index=True), nw_lags=5)
    assert not rep_b["passes"]
    assert rep_b["attenuation_alarm"]


def test_winsorized_ic_close_to_raw_without_outliers():
    sc = np.arange(1.0, 51.0)
    df = _cs("2020-01-06", sc, sc + 0.1)
    raw = S.daily_rank_ic(df).iloc[0]
    wins = S.winsorized_rank_ic(df).iloc[0]
    assert wins == pytest.approx(raw, abs=0.02)
