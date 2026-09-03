"""Experiment 5 tests for the pre-decided NT=5 mechanical K13 rerun."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import k13_order_audit as k13  # noqa: E402


def test_cli_defaults_and_custom_output() -> None:
    default = k13.parse_args([])
    assert default.nt == k13.DEFAULT_NT == 6
    assert default.out == k13.OUT / "k13_order_audit.json"

    custom_path = REPO / "outputs" / "audit.json"
    custom = k13.parse_args(["--nt", "5", "--out", str(custom_path)])
    assert custom.nt == 5
    assert custom.out == custom_path

    with pytest.raises(SystemExit):
        k13.parse_args(["--nt", "0"])


def test_custom_output_keeps_legacy_and_nt5_orders_separate() -> None:
    default = k13.OUT / "k13_order_audit.json"
    nt5 = k13.OUT / "k13_order_audit_nt5.json"
    assert k13._orders_dest(default, "FT") == k13.OUT / "k13_orders_FT.parquet"
    assert k13._orders_dest(nt5, "FT") == k13.OUT / "k13_orders_FT_nt5.parquet"


def test_orders_use_requested_nt_without_search(monkeypatch: pytest.MonkeyPatch) -> None:
    names = np.arange(10001, 10101)
    days = pd.date_range("2023-01-02", periods=8, freq="B")
    rows = []
    for i, day in enumerate(days):
        scores = np.roll(np.arange(100, dtype=float), i * 10)
        rows.extend(
            {"PERMNO": int(p), "signal_date": day, "score": float(s)}
            for p, s in zip(names, scores)
        )
    score_frame = pd.DataFrame(rows)
    adv = {d: {int(p): float(j + 1) for j, p in enumerate(names)} for d in days}
    px = {d: {int(p): 10.0 for p in names} for d in days}

    monkeypatch.setattr(k13, "FOLDS", ["fold36"])
    monkeypatch.setitem(k13.ARM_PATHS, "FT", {"fold36": Path("unused")})
    monkeypatch.setattr(k13, "load_scores", lambda path: score_frame.copy())

    orders = k13.orders_for_arm("FT", adv, px, nt=5)
    assert not orders.empty
    assert np.allclose(orders["w_frac"].unique(), [1.0 / (10 * 5)])
    assert orders["trade_date"].min() == days[6]
