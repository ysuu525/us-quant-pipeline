from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gbdt_baseline.py"
SPEC = importlib.util.spec_from_file_location("gbdt_baseline", SCRIPT)
assert SPEC and SPEC.loader
gbdt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gbdt
SPEC.loader.exec_module(gbdt)


def test_frozen_grids_scan_eight_structural_combinations() -> None:
    config, _ = gbdt._read_config(Path(__file__).resolve().parents[1] / "configs" / "gbdt_strong_v2.json")
    assert len(gbdt._grid(config, "lightgbm")) == 8
    assert len(gbdt._grid(config, "xgboost")) == 8
    assert len(gbdt._grid(config, "catboost")) == 8


def test_choose_trial_uses_frozen_tie_break() -> None:
    trials = [
        {"params": {"num_leaves": 127, "num_boost_round": 300}, "inner_rank_ic": 0.01},
        {"params": {"num_leaves": 31, "num_boost_round": 300}, "inner_rank_ic": 0.01},
        {"params": {"num_leaves": 31, "num_boost_round": 700}, "inner_rank_ic": 0.01},
    ]
    assert gbdt._choose_trial("lightgbm", trials) == trials[1]


def test_inner_split_purges_labels_that_cross_inner_boundary() -> None:
    dates = pd.bdate_range("2020-01-01", periods=140)
    frame = pd.DataFrame(
        {
            "PERMNO": np.tile([1, 2], len(dates)),
            "date": np.repeat(dates, 2),
            "y": 1.0,
            "label_end_date": np.repeat(dates + pd.offsets.BDay(6), 2),
        }
    )
    config = {"inner_validation": {"tail_trading_days": 20, "tune_max_stocks_per_day": 2}}
    train, valid, boundary = gbdt._inner_split(frame, config)
    assert train["label_end_date"].max() < boundary
    assert valid["date"].min() == boundary


def test_jkp_state_matrix_is_not_cross_sectionally_ranked() -> None:
    frame = pd.DataFrame(
        {
            "PERMNO": [1, 2],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            **{column: [0.25, 0.75] for column in gbdt.STOCK_FEATURES},
        }
    )
    state = pd.DataFrame(
        {"jkp_a_cum5": [0.123], "jkp_b_cum20": [-0.04]},
        index=pd.to_datetime(["2020-01-02"]),
    )
    config = {"max_rss_gb": 8.0, "compute": {"min_available_ram_gb_before_fold": 0.0}}
    matrix = gbdt._build_matrix(frame, state, config)
    np.testing.assert_allclose(matrix[:, -2:], [[0.123, -0.04], [0.123, -0.04]])


def test_nw_mean_matches_plain_daily_mean() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.04])
    stats = gbdt._nw_stats(values)
    assert stats["mean"] == np.mean(values)
    assert stats["n"] == 4


def test_v2_uses_a_physical_cutoff_jkp_snapshot() -> None:
    config, _ = gbdt._read_config(Path(__file__).resolve().parents[1] / "configs" / "gbdt_strong_v2.json")
    features = config["features"]
    assert "jkp_source" not in features
    assert features["jkp_state_snapshot"].endswith("through_2023-12-29.parquet")
    assert len(features["jkp_state_snapshot_sha256"]) == 64


def test_score_resume_fails_closed_on_stale_sidecar(tmp_path: Path) -> None:
    score = tmp_path / "scores.parquet"
    score.write_bytes(b"frozen-score")
    expected = {"score_schema_version": 2, "model": "lightgbm"}
    sidecar = gbdt._score_sidecar(score)
    sidecar.write_text(
        json.dumps(
            {
                **expected,
                "model": "catboost",
                "score_sha256": gbdt._sha256(score),
            }
        ),
        encoding="utf-8",
    )
    try:
        gbdt._score_complete(score, expected)
    except RuntimeError as error:
        assert "Stale score provenance" in str(error)
    else:
        raise AssertionError("stale score sidecar was accepted")


def test_pre_registered_decision_boundaries() -> None:
    assert gbdt._decision(0.009999) == "Kronos clearly wins"
    assert gbdt._decision(0.010000) == "Kronos leads, but not decisively"
    assert gbdt._decision(0.020000) == "Kronos leads, but not decisively"
    assert gbdt._decision(0.020001) == "The tree model matches or beats Kronos"


def test_tuning_resume_rejects_cross_fold_artifact() -> None:
    config, config_sha = gbdt._read_config(
        Path(__file__).resolve().parents[1] / "configs" / "gbdt_strong_v2.json"
    )
    provenance = {"frozen": True}
    trials = []
    for index, base in enumerate(gbdt._grid(config, "lightgbm")):
        for rounds in config["round_checkpoints"]:
            trials.append(
                {
                    "params": {**base, "num_boost_round": rounds},
                    "inner_rank_ic": float(index) / 1000.0,
                }
            )
    value = {
        "config_sha256": config_sha,
        "provenance": provenance,
        "model": "lightgbm",
        "fold": "fold37",
        "trials": trials,
        "best": gbdt._choose_trial("lightgbm", trials),
    }
    try:
        gbdt._validate_tuning_artifact(
            value, "lightgbm", gbdt.FOLDS[0], config, config_sha, provenance
        )
    except RuntimeError as error:
        assert "Stale tuning field fold" in str(error)
    else:
        raise AssertionError("cross-fold tuning artifact was accepted")
