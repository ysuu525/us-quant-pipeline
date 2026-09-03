"""Synthetic and text-only tests for experiment 6."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp6_deflated_sharpe", REPO / "scripts" / "exp6_deflated_sharpe.py"
)
EXP6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXP6
SPEC.loader.exec_module(EXP6)


def _ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.md"
    path.write_text(
        "# ledger\n"
        "- 2026-08-26T18:03:34+00:00 | eval | t=2.0\n"
        "- 2026-08-27 | ablation-read | 纯多 8bp 夏普0.50 t=1.2\n"
        "- 2026-08-28 | DECISION | frozen\n"
        "- 2026-08-29 | eval | t=-3.0\n"
        "- 2026-08-30 | ablation-read | 纯多 8bp 夏普 0.75 no statistic\n",
        encoding="utf-8",
    )
    return path


def test_ledger_snapshot_counts_dates_distribution_and_bytes(tmp_path):
    path = _ledger(tmp_path)
    raw, lines = EXP6.read_ledger_once(path)
    got = EXP6.ledger_snapshot(raw, lines)
    assert got["sha256"] == hashlib.sha256(raw).hexdigest()
    assert got["bytes"] == len(raw)
    assert got["N_low"] == 2
    assert got["N_high"] == 4
    assert got["selected_counts"]["eval"] == {
        "count": 2,
        "first_date": "2026-08-26T18:03:34+00:00",
        "last_date": "2026-08-29",
    }
    assert got["type_distribution"] == {"DECISION": 1, "ablation-read": 2, "eval": 2}


def test_empirical_sharpe_variance_requires_five_entries():
    lines = [
        EXP6.LedgerLine(i, "2026-01-01", "ablation-read", f"纯多 8bp 夏普 {x}")
        for i, x in enumerate([0.1, 0.2, 0.4, 0.7, 1.0], 1)
    ]
    got = EXP6.parse_same_family_sharpes(lines)
    expected = np.var(np.asarray([0.1, 0.2, 0.4, 0.7, 1.0]) / np.sqrt(252), ddof=1)
    assert got["method"] == "ledger_empirical_sample_variance"
    assert got["daily_sharpe_variance"] == expected
    assert [x["line"] for x in got["entries"]] == [1, 2, 3, 4, 5]


def test_dsr_formula_and_maximum_n_are_monotone():
    moments = {
        "T": 800,
        "SR_daily": 0.08,
        "skewness_bias_corrected": 0.0,
        "kurtosis_nonexcess_bias_corrected": 3.0,
    }
    variance = 0.0001
    one = EXP6.deflated_sharpe_probability(moments, variance, 1)
    ten = EXP6.deflated_sharpe_probability(moments, variance, 10)
    assert one["SR0_daily"] == 0.0
    assert ten["SR0_daily"] > 0.0
    assert ten["DSR"] < one["DSR"]
    maximum = EXP6.maximum_n_at_dsr(moments, variance)
    assert isinstance(maximum, int) and maximum >= 1
    assert EXP6.deflated_sharpe_probability(moments, variance, maximum)["DSR"] >= 0.95
    assert EXP6.deflated_sharpe_probability(moments, variance, maximum + 1)["DSR"] < 0.95


def test_holm_and_bhy_match_hand_calculation():
    raw = [0.01, 0.03, 0.20]
    got = EXP6.adjusted_p_values(raw)
    assert np.allclose(got["bonferroni"], [0.03, 0.09, 0.60])
    assert np.allclose(got["holm"], [0.03, 0.06, 0.20])
    # BHY: c(3)=11/6; raw sorted corrections .055, .0825, .3667.
    assert np.allclose(got["bhy"], [0.055, 0.0825, 11 / 30])


def test_t_parser_does_not_mistake_execution_lag_for_a_statistic():
    text = "t+1 开盘；NW t +1.32；t(alpha)=2.11；t=-0.4"
    assert [float(x) for x in EXP6.T_RE.findall(text)] == [1.32, 2.11, -0.4]


def test_haircut_and_prediction_fallback():
    family = [0.001, 0.02, 1.0, 1.0]
    got = EXP6.haircut_for_family(3.0, 0.02, family)
    assert got["N"] == 4
    assert set(got["methods"]) == {"bonferroni", "holm", "bhy"}
    assert all(0 <= x["haircut"] <= 1 for x in got["methods"].values())

    same_side = {
        "methods": {
            "bonferroni": {"adjusted_effect": 0.010},
            "holm": {"adjusted_effect": 0.011},
            "bhy": {"adjusted_effect": 0.012},
        }
    }
    pred = EXP6.prediction_interval(same_side)
    assert pred["fallback_triggered"] is True
    assert math.isclose(pred["lower"], 0.010 - 0.0027)
    assert math.isclose(pred["upper"], EXP6.PREDICTION_CENTER + 0.0027)


def test_return_moments_use_measured_daily_values():
    daily = pd.Series([0.01, -0.005, 0.004, 0.002, -0.001, 0.007])
    got = EXP6.return_moments(daily)
    expected = daily.mean() / daily.std(ddof=1)
    assert got["T"] == 6
    assert math.isclose(got["SR_daily"], expected)
    assert math.isclose(got["SR_annualized"], expected * math.sqrt(252))
