"""Unit tests for Experiment 12's frozen cost(AUM) mechanics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import exp12_cost_aum_curve as exp12  # noqa: E402


def _audit() -> dict:
    arms = {}
    for arm, factor in (("FT", 1.0), ("ZS", 2.0)):
        by_aum = {}
        for aum in exp12.AUM_GRID:
            scale = aum / 1_000_000.0
            by_aum[f"{aum:.0f}"] = {
                "participation_of_adv20": {
                    "p50": 1.0e-5 * scale * factor,
                    "p90": 2.0e-5 * scale * factor,
                    "p99": 3.0e-5 * scale * factor,
                },
                "daily_traded_usd": 100_000.0 * scale * factor,
            }
        arms[arm] = {"by_aum": by_aum}
    return {"arms": arms}


def test_fixed_impact_formulas() -> None:
    participation, sigma = 0.01, 0.02
    assert exp12.square_root_impact_bp(participation, sigma, 1.0) == pytest.approx(20.0)
    assert exp12.athl_temporary_impact_bp(participation, sigma) == pytest.approx(
        1.0e4 * 0.142 * 0.02 * participation ** 0.6
    )
    with pytest.raises(ValueError):
        exp12.square_root_impact_bp(-0.01, sigma, 1.0)


def test_development_return_window_ends_before_2024() -> None:
    assert exp12.DEV_RETURN_START == pd.Timestamp("2020-07-01")
    assert exp12.DEV_RETURN_END == pd.Timestamp("2023-12-29")
    assert exp12.DEV_RETURN_END < pd.Timestamp("2024-01-01")


def test_grid_and_nt6_to_nt5_mechanical_scaling() -> None:
    sigma = 0.02
    rows = exp12.participation_rows(_audit(), exp12.NT6_TO_NT5, sigma)
    assert len(rows) == 2 * 4 * 3
    first = rows[0]
    assert first["arm"] == "FT"
    assert first["aum_usd"] == 1_000_000.0
    assert first["participation_percentile"] == "p50"
    assert first["participation_decimal"] == pytest.approx(1.2e-5)
    assert set(first["square_root_impact_bp"]) == {"Y=0.5", "Y=1", "Y=2"}


def test_k13_input_prefers_nt5_and_declares_nt6_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = Path("X:/synthetic/outputs")

    def install(available: set[str]) -> None:
        monkeypatch.setattr(Path, "exists", lambda self: self.name in available)
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda self, encoding=None: json.dumps(
                {"frozen_construction": {"NT": 5 if "nt5" in self.stem else 6}}
            ),
        )

    install({"k13_order_audit_nt5.json", "k13_order_audit.json"})
    _, path, scale, status = exp12.select_k13_input(out_dir)
    assert path.name == "k13_order_audit_nt5.json"
    assert scale == 1.0
    assert "measured" in status

    install({"k13_order_audit.json"})
    _, path, scale, status = exp12.select_k13_input(out_dir)
    assert path.name == "k13_order_audit.json"
    assert scale == pytest.approx(6.0 / 5.0)
    assert "approximation" in status


def test_crossing_is_analytic_not_a_scan() -> None:
    assert exp12.crossing_aum(1_000_000.0, 2.0, 4.0, 0.5) == pytest.approx(4_000_000.0)
    assert exp12.crossing_aum(1_000_000.0, 2.0, 8.0, 0.5) == pytest.approx(16_000_000.0)


def test_sigma_is_cross_sectional_median_of_name_volatility() -> None:
    # Name sigmas are 1 and 3, so the cross-sectional median is 2.
    stats = pd.DataFrame(
        {"count": [2, 2], "sum": [0.0, 0.0], "sum_sq": [1.0, 9.0]},
        index=[1, 2],
    )
    sigma, meta = exp12._sigma_from_moments(stats)
    assert sigma == pytest.approx(2.0)
    assert meta["n_names_with_two_returns"] == 2


def test_svg_has_frozen_curves_and_limitations() -> None:
    sigma = 0.02
    audit = _audit()
    result = {
        "impact_table": exp12.participation_rows(audit, 1.0, sigma),
        "crossings": exp12.compute_crossings(audit, 1.0, sigma),
    }
    svg = exp12._svg(result)
    assert svg.startswith("<svg")
    assert "MODEL VALUES, NOT MEASUREMENTS" in svg
    assert "ATHL beta=0.6" in svg
    assert "FT 11.2bp" in svg
    assert "ZS 6.3bp" in svg
