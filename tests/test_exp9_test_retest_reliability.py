"""Synthetic-only tests for experiment 9; no real scores, outcomes or GPU are touched."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp9_test_retest_reliability", REPO / "scripts" / "exp9_test_retest_reliability.py"
)
EXP9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXP9
SPEC.loader.exec_module(EXP9)


def _synthetic(seed: int = 9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=15)
    names = np.arange(10_000, 10_060, dtype=np.int64)
    scores = {}
    labels = {}
    adv = {}
    for fold_index, fold in enumerate(EXP9.FOLDS):
        outcomes = []
        latent_by_day = {}
        for day in dates + pd.offsets.DateOffset(months=fold_index * 6):
            latent = rng.normal(size=len(names))
            future = latent + rng.normal(scale=1.5, size=len(names))
            latent_by_day[day] = latent
            outcomes.extend(zip(names, [day] * len(names), future, ["ok"] * len(names)))
            adv[day] = dict(zip(names, rng.lognormal(15, 1, len(names))))
        labels[fold] = pd.DataFrame(
            outcomes, columns=["PERMNO", "signal_date", "label", "status"]
        )
        for sample_count in EXP9.SAMPLE_COUNTS:
            noise_scale = math.sqrt(5 / sample_count)
            for repeat in EXP9.REPEATS:
                rows = []
                for day, latent in latent_by_day.items():
                    score = latent + rng.normal(scale=noise_scale, size=len(names))
                    rows.extend(zip(names, [day] * len(names), score))
                scores[(fold, sample_count, repeat)] = pd.DataFrame(
                    rows, columns=["PERMNO", "signal_date", "score"]
                )
    return scores, labels, adv


def test_path_map_reuses_all_twelve_r1_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(EXP9, "REPO", tmp_path)
    reused = EXP9.reuse_map()
    new = EXP9.new_gpu_cells()
    assert len(reused) == 12
    assert len(new) == 12
    assert len({row["logical_cell"] for row in reused + new}) == 24
    assert "eval_amp_lb90_fold36" in reused[0]["path"]
    assert any("eval_e1_sc40" in row["path"] for row in reused)
    assert all("_r2" in row["tag"] for row in new)


def test_invalid_logical_cells_fail_before_path_read():
    with pytest.raises(ValueError):
        EXP9.score_file(37, 5, 1)
    with pytest.raises(ValueError):
        EXP9.score_file(36, 30, 1)
    with pytest.raises(ValueError):
        EXP9.score_file(36, 5, 3)


def test_reliability_uses_exact_name_intersection_and_reports_drops():
    day = pd.Timestamp("2022-01-03")
    left = pd.DataFrame({
        "PERMNO": [1, 2, 3, 4], "signal_date": [day] * 4, "score": [1, 2, 3, 4]
    })
    right = pd.DataFrame({
        "PERMNO": [2, 3, 4, 5], "signal_date": [day] * 4, "score": [2, 3, 4, 5]
    })
    daily, coverage = EXP9.reliability_daily(left, right)
    assert daily.iloc[0] == 1.0
    assert coverage["intersection_nonmissing_rows"] == 3
    assert coverage["left_only_keys"] == 1
    assert coverage["right_only_keys"] == 1


def test_duplicate_score_keys_fail_closed():
    frame = pd.DataFrame({
        "PERMNO": [1, 1], "signal_date": pd.to_datetime(["2022-01-03"] * 2),
        "score": [0.1, 0.2],
    })
    with pytest.raises(ValueError, match="duplicate"):
        EXP9.reliability_daily(frame, frame)


def test_spearman_brown_formula():
    measured = {5: 0.75, 10: 0.80, 20: 0.90, 40: 0.94}
    got = EXP9.spearman_brown(measured)
    assert got["signal_variance_over_sampling_noise_variance"] == 3.0
    assert math.isclose(got["sample_count_multiple_needed_for_reliability_0.95"], 19 / 3)
    assert math.isclose(
        got["prediction_vs_measurement"]["20"]["predicted_reliability"],
        4 * 0.75 / (1 + 3 * 0.75),
    )


def test_full_synthetic_analysis_keeps_auxiliary_repeats_separate(monkeypatch):
    monkeypatch.setattr(EXP9, "BOOT_DRAWS", 100)
    monkeypatch.setattr(EXP9, "QUEUE_DONE", Path("does-not-exist"))
    scores, labels, adv = _synthetic()
    report = EXP9.analyse(scores, labels, adv)
    assert set(report["reliability"]) == {"5", "10", "20", "40"}
    assert report["reliability"]["5"]["pooled"]["n_days"] == 45
    assert report["reliability"]["40"]["pooled"]["mean"] > report["reliability"]["5"]["pooled"]["mean"]
    assert report["meta"]["new_gpu_calls"] == 12
    assert set(report["reliability_monotonicity"]["successive_differences"]) == {
        "sc5_to_sc10", "sc10_to_sc20", "sc20_to_sc40"
    }
    r1 = report["auxiliary_metrics"]["5"]["r1"]["ic_full"]["pooled"]["mean"]
    r2 = report["auxiliary_metrics"]["5"]["r2"]["ic_full"]["pooled"]["mean"]
    assert r1 != r2
    assert "r1" in report["auxiliary_metrics"]["5"]
    assert "r2" in report["auxiliary_metrics"]["5"]
    comparison = report["reliability_vs_ic_se"]["5"]["36"]
    assert "ic_r1_over_reliability_se_ratio_full" in comparison
    assert "ic_r2_over_reliability_se_ratio_full" in comparison


def test_svg_is_plain_deterministic_text(tmp_path, monkeypatch):
    monkeypatch.setattr(EXP9, "BOOT_DRAWS", 50)
    monkeypatch.setattr(EXP9, "QUEUE_DONE", Path("does-not-exist"))
    scores, labels, adv = _synthetic(17)
    report = EXP9.analyse(scores, labels, adv)
    out = tmp_path / "exp9.svg"
    EXP9.write_svg(report, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "test-retest reliability" in text
    assert "never averaged" in text
