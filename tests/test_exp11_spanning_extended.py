"""实验 11 扩充张成控制的纯函数与门禁测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp11_spanning_extended", REPO / "scripts" / "exp11_spanning_extended.py"
)
EXP11 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP11)


def test_fixed_blocks_and_specs_cannot_reach_unopened_folds():
    assert EXP11.FOLDS == tuple(range(36, 43))
    assert sorted(fold for block in EXP11.BLOCKS for fold in block["folds"]) == list(EXP11.FOLDS)
    for block in EXP11.BLOCKS:
        lo, hi = pd.Timestamp(block["lo"]), pd.Timestamp(block["hi"])
        assert hi <= lo + pd.DateOffset(years=2)
    assert EXP11.SPEC_COLUMNS == {
        "S-T": EXP11.CORE + ("turnover", "turnover_x_rev1", "turnover_x_rev5"),
        "S-H": EXP11.CORE + ("hi52",),
        "S-TH-ind": EXP11.CORE + EXP11.EXTENDED_COLUMNS,
    }


def test_raw_loader_uses_exact_columns_and_date_pushdown(monkeypatch, tmp_path):
    captured = {}

    def fake_read(path, *, columns, filters):
        captured.update(path=path, columns=columns, filters=filters)
        return pd.DataFrame(
            {
                "PERMNO": [1], "DlyCalDt": [pd.Timestamp("2020-01-02")],
                "DlyOpen": [10.0], "DlyHigh": [11.0], "DlyLow": [9.0],
                "DlyClose": [10.5], "DlyVol": [100.0], "DlyPrcVol": [1050.0],
                "DlyRet": [0.01], "DlyCap": [10000.0],
            }
        )

    monkeypatch.setattr(EXP11, "assert_readable", lambda path: None)
    monkeypatch.setattr(EXP11.pd, "read_parquet", fake_read)
    got = EXP11.load_raw_block(tmp_path, "2020-01-01", "2020-12-31")
    assert len(got) == 1
    assert captured["columns"] == list(EXP11.RAW_COLUMNS)
    assert captured["filters"] == [
        ("DlyCalDt", ">=", pd.Timestamp("2020-01-01")),
        ("DlyCalDt", "<=", pd.Timestamp("2020-12-31")),
    ]


def test_turnover_and_hi52_use_only_current_and_past():
    days = pd.bdate_range("2020-01-01", periods=260)
    raw = pd.DataFrame(
        {
            "PERMNO": 1,
            "DlyCalDt": days,
            "DlyPrcVol": np.arange(1.0, 261.0),
            "DlyCap": 10.0,
        }
    )
    adjusted = pd.DataFrame(
        {"PERMNO": 1, "DlyCalDt": days, "DlyClose": np.arange(1.0, 261.0)}
    )
    turnover_before = EXP11.turnover_history(raw)
    hi52_before = EXP11.hi52_history(adjusted)
    raw.loc[259, "DlyPrcVol"] = 1e9
    adjusted.loc[259, "DlyClose"] = 1e9
    turnover_after = EXP11.turnover_history(raw)
    hi52_after = EXP11.hi52_history(adjusted)
    assert turnover_before.loc[250, "turnover_raw"] == pytest.approx(
        turnover_after.loc[250, "turnover_raw"]
    )
    assert hi52_before.loc[251, "hi52"] == pytest.approx(hi52_after.loc[251, "hi52"])
    assert hi52_before["hi52"].iloc[:251].isna().all()
    assert hi52_before.loc[251, "hi52"] == pytest.approx(1.0)


def test_point_in_time_sic2_honours_effective_end_date():
    keys = pd.DataFrame(
        {
            "PERMNO": [1, 1, 1],
            "signal_date": pd.to_datetime(["2020-03-01", "2020-08-01", "2021-02-01"]),
        }
    )
    history = pd.DataFrame(
        {
            # Match the nullable integer dtype returned by the real parquet.
            "PERMNO": pd.Series([1, 1], dtype="Int64"),
            "secinfostartdt": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "secinfoenddt": pd.to_datetime(["2020-06-30", "2021-12-31"]),
            "sic2": ["20", "35"],
        }
    )
    got = EXP11.point_in_time_sic2(keys, history)
    assert got["sic2"].iloc[0] == "20"
    assert pd.isna(got["sic2"].iloc[1])
    assert got["sic2"].iloc[2] == "35"


def test_candidate_ranks_interactions_and_industry_demean():
    day = pd.Timestamp("2022-01-03")
    frame = pd.DataFrame(
        {
            "PERMNO": [1, 2, 3, 4], "signal_date": day,
            "score": [1.0, 3.0, 2.0, 6.0], "turnover_raw": [10.0, 20.0, 30.0, 40.0],
            "rev1": [4.0, 3.0, 2.0, 1.0], "rev5": [1.0, 2.0, 3.0, 4.0],
            "hi52": [0.7, 0.8, 0.9, 1.0], "sic2": ["10", "10", "20", "20"],
        }
    )
    for column in EXP11.CORE:
        if column not in frame:
            frame[column] = np.arange(4, dtype=float)
    extended = EXP11.add_candidate_extended_columns(frame)
    assert extended["turnover"].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert extended["turnover_x_rev1"].iloc[0] < 0
    assert extended["turnover_x_rev5"].iloc[0] > 0
    neutral = EXP11.industry_demean(
        extended, ("score",) + EXP11.CORE + EXP11.EXTENDED_COLUMNS
    )
    for column in ("score",) + EXP11.CORE + EXP11.EXTENDED_COLUMNS:
        means = neutral.groupby(["signal_date", "sic2"])[f"ind_{column}"].mean()
        assert np.allclose(means, 0.0, atol=1e-12, equal_nan=True)


def test_memory_gate_stops_before_large_read(monkeypatch):
    monkeypatch.setattr(EXP11, "committed_memory_gb", lambda: 52.8)
    with pytest.raises(EXP11.MemoryGateError, match="52.80 GB"):
        EXP11.enforce_memory_gate()


def test_mechanical_readout_requires_both_thresholds(monkeypatch):
    dates = pd.bdate_range("2020-07-01", periods=350)
    fold_names = np.repeat([f"fold{i}" for i in range(36, 43)], 50)
    strategy = pd.DataFrame(
        {"long": np.linspace(0.0001, 0.001, 350), "fold": fold_names}, index=dates
    )
    controls = pd.DataFrame({"control": np.linspace(-0.001, 0.001, 350)}, index=dates)
    market = pd.Series(np.sin(np.arange(350)) * 0.001, index=dates, name="market")

    def fake_nw(y, design, lags):
        beta = np.zeros(design.shape[1])
        beta[0] = y.mean() * 0.8
        return beta, np.ones(design.shape[1])

    monkeypatch.setattr(EXP11.K6B, "nw_ols", fake_nw)
    result = EXP11.mechanical_readout("S-T", strategy, controls, market)
    assert result["retention_pct"] == pytest.approx(80.0)
    assert result["folds_alpha_positive"] == 7
    assert result["mechanical_conclusion"].startswith("非翻版")
