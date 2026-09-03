from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "exp4_strata_residual_ic.py"
    spec = importlib.util.spec_from_file_location("exp4_strata_residual_ic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_residualization_removes_linear_factor_component() -> None:
    mod = _load_module()
    rng = np.random.default_rng(20260904)
    n = 80
    frame = pd.DataFrame({name: rng.normal(size=n) for name in mod.CORE})
    frame["label"] = rng.normal(size=n)
    frame["score"] = (
        2.0 + sum((i + 1) * frame[name] for i, name in enumerate(mod.CORE))
        + 0.1 * frame["label"]
    )
    out = mod.residualize_one_day(frame)
    x = np.column_stack([np.ones(n), out.loc[:, mod.CORE].to_numpy()])
    assert np.max(np.abs(x.T @ out["residual_score"].to_numpy())) < 1e-8


def test_summarize_ic_reports_ratio_and_trigger() -> None:
    mod = _load_module()
    rng = np.random.default_rng(7)
    rows = []
    for day in pd.date_range("2020-01-01", periods=20):
        label = rng.normal(size=40)
        score = label + rng.normal(scale=0.2, size=40)
        residual = rng.normal(size=40)
        for i in range(40):
            rows.append((day, i, score[i], residual[i], label[i]))
    frame = pd.DataFrame(
        rows, columns=["signal_date", "PERMNO", "score", "residual_score", "label"])
    result = mod.summarize_ic(frame)
    assert result["n_days"] == 20
    assert result["original_ic"]["mean"] > 0.9
    assert result["residual_to_original_ratio"] < 0.5
    assert result["disclosure_triggered"] is True
