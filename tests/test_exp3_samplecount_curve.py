"""实验 3 汇总器的合成数据测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals import kronos_adapter as ka

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp3_samplecount_curve", REPO / "scripts" / "exp3_samplecount_curve.py"
)
EXP3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP3)


def _frame(seed: int = 7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=15)
    names = np.arange(10001, 10101, dtype=np.int64)
    rows = []
    labels = []
    adv = {}
    for day in days:
        y = rng.normal(size=len(names))
        score = y + rng.normal(scale=0.5, size=len(names))
        rows.extend(zip(names, [day] * len(names), score))
        labels.extend(zip(names, [day] * len(names), y, ["ok"] * len(names)))
        adv[day] = dict(zip(names, rng.lognormal(15, 1, size=len(names))))
    return (
        pd.DataFrame(rows, columns=["PERMNO", "signal_date", "score"]),
        pd.DataFrame(labels, columns=["PERMNO", "signal_date", "label", "status"]),
        adv,
    )


def test_experiment_paths_are_explicitly_whitelisted(tmp_path):
    got = ka.scores_path(36, "ft", root=tmp_path, experiment_tag="e1_sc20")
    assert got == (
        tmp_path / "fold36_lb90_s0_poolB_universe" / "eval_e1_sc20" / "scores.parquet"
    )
    with pytest.raises(ValueError):
        ka.scores_path(36, "ft", root=tmp_path, experiment_tag="e1_sc30")
    with pytest.raises(ValueError):
        ka.scores_path(37, "ft", root=tmp_path, experiment_tag="e1_sc40")
    with pytest.raises(ValueError):
        ka.scores_path(36, "zs", root=tmp_path, experiment_tag="e1_sc20")


def test_daily_metrics_and_reliability_on_synthetic_data(monkeypatch):
    scores, labels, adv = _frame()
    metrics = EXP3.daily_metrics(scores, labels, adv)
    assert set(metrics) == {
        "ic_full", "ic_top500", "decile_spread_full", "decile_spread_top500"
    }
    assert all(len(series) == 15 for series in metrics.values())
    assert metrics["ic_full"].mean() > 0.5
    reliability = EXP3.reliability_daily(scores, scores.copy())
    assert np.allclose(reliability.to_numpy(), 1.0)


def test_bootstrap_summary_and_svg_are_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(EXP3, "BOOT_DRAWS", 100)
    series = {
        36: pd.Series(np.linspace(-0.1, 0.2, 20)),
        39: pd.Series(np.linspace(0.0, 0.3, 20)),
        42: pd.Series(np.linspace(0.1, 0.4, 20)),
    }
    a = EXP3.summarize(series)
    b = EXP3.summarize(series)
    assert a == b
    assert a["n_days"] == 60
    assert a["folds_positive"] == 3

    curve = {
        str(sc): {
            key: {"mean": float(sc) / 100, "stationary_bootstrap_ci95": [0.0, 1.0]}
            for key in (
                "ic_full", "ic_top500", "decile_spread_full", "decile_spread_top500"
            )
        }
        for sc in EXP3.SAMPLE_COUNTS
    }
    report = {
        "curves": {"matched_folds_36_39_42": curve},
        "reliability": {
            "sc5_vs_sc20_all7": {"mean": 0.8, "stationary_bootstrap_ci95": [0.7, 0.9]},
            "sc20_vs_sc40_three": {"mean": 0.9, "stationary_bootstrap_ci95": [0.85, 0.95]},
        },
    }
    out = tmp_path / "curve.svg"
    EXP3.write_svg(report, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "sample-count curve" in text
