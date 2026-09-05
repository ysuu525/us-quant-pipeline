"""exp13 读数层：NW(5) 抄件与 K6b 原件逐位一致；readout 不输出任何结论字符串。"""
from __future__ import annotations

import importlib.util
import inspect
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


def test_nw_ols_copy_matches_k6b_original_bit_for_bit():
    """任务书 §2.2 只允许 import 或逐字抄；抄件必须与原件给出相同的 b/t。"""
    rng = np.random.default_rng(7)
    y = rng.normal(size=400)
    design = np.column_stack([np.ones(400), rng.normal(size=(400, 3))])
    beta_ref, t_ref = EXP13.K6B.nw_ols(y, design, 5)
    beta, tstat, se = EXP13.nw_ols_se(y, design, 5)
    assert np.array_equal(beta, beta_ref)
    assert np.array_equal(np.nan_to_num(tstat), np.nan_to_num(t_ref))
    assert np.all(se > 0)
    assert np.allclose(tstat, beta / se)


def test_nw_mean_ci_recovers_a_known_mean():
    series = pd.Series(np.full(300, 0.002))
    out = EXP13.nw_mean_ci(series)
    assert out["mean"] == pytest.approx(0.002)
    assert out["n"] == 300


def _fake_panel(n_days: int = 300):
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2020-07-01", periods=n_days)
    controls = pd.DataFrame(rng.normal(0, 0.01, (n_days, 2)),
                            index=index, columns=["c1", "c2"])
    market = pd.Series(rng.normal(0, 0.01, n_days), index=index, name="market")
    strategy = pd.DataFrame(
        {"long": 0.0004 + 0.3 * controls["c1"] + rng.normal(0, 0.005, n_days),
         "ls": rng.normal(0, 0.005, n_days)},
        index=index,
    )
    strategy["fold"] = np.where(np.arange(n_days) < 150, "fold36", "fold37")
    return strategy, controls, market


def test_readout_reports_estimates_and_ci_without_any_verdict_string():
    strategy, controls, market = _fake_panel()
    out = EXP13.readout("spec", strategy, controls, market, (36, 37))
    assert set(["alpha_ann_pct", "alpha_ann_pct_ci95", "retention_pct",
                "nw5_t_alpha", "folds_alpha_positive", "betas"]) <= set(out)
    low, high = out["alpha_ann_pct_ci95"]
    assert low < out["alpha_ann_pct"] < high
    for payload in out["betas"].values():
        assert payload["ci95"][0] < payload["coefficient"] < payload["ci95"][1]
    # §5：无门槛、无 PASS/FAIL、无「有价值 / 无价值」措辞
    banned = ["mechanical_conclusion", "conclusion", "thresholds", "pass", "fail"]
    assert not [key for key in out if key in banned]
    text = str(out)
    for word in ("PASS", "FAIL", "有价值", "无价值", "通过", "未通过", "翻版"):
        assert word not in text


def test_readout_regression_design_matches_exp11():
    """回归元顺序、dropna、y 的取法必须与 exp11:439-502 相同。"""
    strategy, controls, market = _fake_panel()
    mine = EXP13.readout("spec", strategy, controls, market, (36, 37))
    theirs = EXP13.EXP11.mechanical_readout("spec", strategy, controls, market,
                                            folds=(36, 37))
    assert mine["n_days"] == theirs["n_days"]
    assert mine["regressors"] == theirs["regressors"]
    assert mine["alpha_ann_pct"] == pytest.approx(theirs["alpha_ann_pct"])
    assert mine["retention_pct"] == pytest.approx(theirs["retention_pct"])
    assert mine["nw5_t_alpha"] == pytest.approx(theirs["nw5_t_alpha"])
    assert mine["folds_alpha_positive"] == theirs["folds_alpha_positive"]
    assert (mine["per_fold_fixed_loading_residual_alpha_ann_pct"]
            == pytest.approx(theirs["per_fold_fixed_loading_residual_alpha_ann_pct"]))


def test_upstream_pipelines_are_only_imported_never_edited():
    """任务书 §2.2：exp11 / k6b 只 import。此处断言 exp13 没有改它们的冻结常量。"""
    assert EXP13.EXP11.FOLDS == tuple(range(36, 43))
    assert EXP13.EXP11.SPEC_COLUMNS["S-TH-ind"] == EXP13.BASE_CONTROLS
    assert EXP13.K6B.EXIT_PCT == 0.30 and EXP13.K6B.COST_BP == 8.0
    assert EXP13.ALL_CONTROL_COLUMNS[:len(EXP13.BASE_CONTROLS)] == EXP13.BASE_CONTROLS


def test_missing_factor_names_leave_the_daily_pool_exactly_as_exp11_does():
    """照抄 exp11:336-343 的 _by_day：缺失名字剔出该因子当日候选池，<50 名整天不进。"""
    source = inspect.getsource(EXP13.EXP11._by_day)
    assert "dropna()" in source and "minimum_names" in source
    frame = pd.DataFrame({
        "signal_date": [pd.Timestamp("2021-01-04")] * 60 + [pd.Timestamp("2021-01-05")] * 60,
        "PERMNO": list(range(60)) * 2,
        "value": [1.0] * 60 + [np.nan] * 20 + [2.0] * 40,
    })
    got = EXP13.EXP11._by_day(frame, "value")
    assert pd.Timestamp("2021-01-04") in got and len(got[pd.Timestamp("2021-01-04")]) == 60
    # 第二天只剩 40 个有效名字 (<50) -> 整天不进该因子
    assert pd.Timestamp("2021-01-05") not in got


def test_top500_universe_follows_k6b_liquidity_filter():
    rng = np.random.default_rng(3)
    n = 900
    frame = pd.DataFrame({
        "signal_date": [pd.Timestamp("2021-01-04")] * n,
        "PERMNO": np.arange(n),
        "adv20": rng.normal(1e6, 1e5, n),
    })
    frame.loc[0:9, "adv20"] = np.nan
    universe = EXP13.top500_universe(frame)
    assert len(universe) == EXP13.K6B.TOPN
    assert universe["adv20"].notna().all()
    assert universe["adv20"].min() >= frame["adv20"].nlargest(
        EXP13.K6B.TOPN).min() - 1e-9
