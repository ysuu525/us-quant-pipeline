"""实验 7 配对逐日 IC 分析的合成数据测试。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp7_kronos_tree_rho", REPO / "scripts" / "exp7_kronos_tree_rho.py"
)
EXP7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP7)


def _daily(values: np.ndarray, start: str = "2022-01-03") -> pd.DataFrame:
    return pd.DataFrame(
        {"signal_date": pd.bdate_range(start, periods=len(values)), "rank_ic": values}
    )


def test_kronos_daily_ic_matches_e4_full_pool_definition():
    days = pd.bdate_range("2022-01-03", periods=2)
    names = np.arange(100, 112)
    scores, labels = [], []
    for day in days:
        for i, permno in enumerate(names):
            scores.append((permno, day, float(i)))
            labels.append((permno, day, float(i), "ok"))
    labels[0] = (*labels[0][:3], "bad")
    got = EXP7.kronos_daily_ic(
        pd.DataFrame(scores, columns=["PERMNO", "signal_date", "score"]),
        pd.DataFrame(labels, columns=["PERMNO", "signal_date", "label", "status"]),
    )
    assert len(got) == 2
    assert np.allclose(got["rank_ic"], 1.0)
    assert got["n_obs"].tolist() == [11, 12]


def test_pair_stats_uses_daily_pairing_and_reports_drops():
    rng = np.random.default_rng(20260904)
    base = rng.normal(size=80)
    kronos = _daily(base + rng.normal(scale=0.2, size=80))
    tree = _daily(base + rng.normal(scale=0.4, size=80)).iloc[1:].reset_index(drop=True)
    stats, paired = EXP7.pair_stats(kronos, tree)
    assert len(paired) == 79
    assert stats["n_paired_days"] == 79
    assert stats["kronos_days_dropped"] == 1
    assert stats["tree_days_dropped"] == 0
    assert stats["pearson_rho"] > 0.8
    assert stats["spearman_rho"] > 0.8
    assert stats["paired_difference"]["mde80"] == pytest.approx(
        EXP7.MDE80_MULTIPLIER * stats["paired_difference"]["nw5_se"]
    )
    assert stats["independence_comparison"]["paired_nw5_se"] < (
        stats["independence_comparison"]["independent_nw5_se"]
    )


def test_preflight_fails_closed_when_daily_files_are_missing(tmp_path):
    with pytest.raises(EXP7.ArtifactPreflightError, match="需先跑 scripts/gbdt_baseline.py"):
        EXP7.preflight_tree_artifacts(tmp_path)


def test_preflight_checks_registered_sha256(tmp_path, monkeypatch):
    monkeypatch.setattr(EXP7, "MODELS", ("xgboost",))
    monkeypatch.setattr(EXP7, "FOLDS", (36,))
    fold_dir = tmp_path / "xgboost" / "fold36"
    fold_dir.mkdir(parents=True)
    daily = fold_dir / "daily_ic_ensemble.parquet"
    pd.DataFrame(
        {"signal_date": pd.bdate_range("2022-01-03", periods=3), "rank_ic": [0.1, 0.2, 0.3]}
    ).to_parquet(daily, index=False)
    actual = hashlib.sha256(daily.read_bytes()).hexdigest()
    summary = {
        "model": "xgboost",
        "fold": "fold36",
        "ensemble_daily_ic_sha256": actual,
    }
    (fold_dir / "fold_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    checks = EXP7.preflight_tree_artifacts(tmp_path)
    assert checks == [
        {
            "model": "xgboost",
            "fold": 36,
            "daily_ic_path": str(daily),
            "fold_summary_path": str(fold_dir / "fold_summary.json"),
            "sha256": actual,
            "status": "matched",
        }
    ]
    summary["ensemble_daily_ic_sha256"] = "0" * 64
    (fold_dir / "fold_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(EXP7.ArtifactPreflightError, match="sha256 mismatch"):
        EXP7.preflight_tree_artifacts(tmp_path)


def test_hierarchy_advice_uses_both_frozen_sesoi_candidates():
    low = EXP7.hierarchy_advice(0.006)
    assert "A 层" in low["suggestion_only_not_decision"]
    high = EXP7.hierarchy_advice(0.009)
    assert "B 层" in high["suggestion_only_not_decision"]
    assert "在本样本量下该问题不可回答" in high["suggestion_only_not_decision"]
    assert high["candidate_sesoi"]["half_current_gap"]["mde80_le_sesoi"] is False
    assert high["candidate_sesoi"]["discounted_current_gap"]["mde80_le_sesoi"] is True
