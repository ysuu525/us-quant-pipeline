"""Synthetic-only tests for experiment 9; no real scores, outcomes or GPU are touched."""
from __future__ import annotations

import importlib.util
import json
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
    assert set(report["spearman_brown"]["by_fold"]) == {"36", "39", "42"}
    for sc in ("5", "10", "20", "40"):
        for fold in ("36", "39", "42"):
            assert sc in report["spearman_brown"]["by_fold"][fold]["by_sample_count"]
        summary = report["reliability"][sc]
        assert summary["inverse_variance"]["status"] == "ok"
        assert summary["fold_cluster_bootstrap"]["n_clusters"] == 3
        assert "nw5_ci95" in summary["pooled"]
        for repeat in ("r1", "r2"):
            for pool in ("full", "top500"):
                assert report["auxiliary_metrics"][sc][repeat][f"decile_spread_{pool}"][
                    "pooled"]["unit"] == EXP9.SPREAD_UNIT
    assert report["meta"]["holding_period"]["NT"] == 5
    json.dumps(EXP9.json_safe(report), allow_nan=False)


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


def test_year_gaps_never_enter_hac_and_gap_length_does_not_change_results(monkeypatch):
    monkeypatch.setattr(EXP9, "BOOT_DRAWS", 200)
    rng = np.random.default_rng(27)
    parts = {f: pd.Series(rng.normal(loc=i, scale=i + 1, size=30),
                         index=pd.bdate_range(f"{2020 + i * 2}-07-01", periods=30))
             for i, f in enumerate(EXP9.FOLDS)}
    original_nw = EXP9.newey_west_tstat
    original_boot = EXP9.stationary_means
    calls = []

    def single_fold_only(values, lag):
        assert len(values) == 30  # concatenated 90-day HAC must fail this test
        assert (values.index.max() - values.index.min()).days < 50
        calls.append(len(values))
        return original_nw(values, lag)

    monkeypatch.setattr(EXP9, "newey_west_tstat", single_fold_only)

    def single_fold_boot(data, generator):
        assert len(data) == 30  # no stationary block may see concatenated folds
        return original_boot(data, generator)

    monkeypatch.setattr(EXP9, "stationary_means", single_fold_boot)
    report = EXP9.pooled_with_folds(parts)
    farther = {f: v.set_axis(v.index + pd.DateOffset(years=i * 5))
               for i, (f, v) in enumerate(parts.items())}
    assert report == EXP9.pooled_with_folds(farther)
    assert len(calls) == 6
    assert report["pooled"]["n_days"] == 90
    assert math.isclose(report["pooled"]["mean"], np.mean([s.mean() for s in parts.values()]))
    assert not math.isclose(report["pooled"]["mean"], report["inverse_variance"]["mean"])


def test_inverse_variance_and_day_weighted_se_against_hand_calculation(monkeypatch):
    monkeypatch.setattr(EXP9, "BOOT_DRAWS", 50)
    # For [-1,0,1], Bartlett NW(5) variance of mean = 2/27, by hand.
    parts = {
        36: pd.Series([-1., 0., 1.], index=pd.bdate_range("2020-07-01", periods=3)),
        39: pd.Series([1., 3., 5.], index=pd.bdate_range("2022-01-03", periods=3)),
    }
    got = EXP9.pooled_with_folds(parts)
    assert math.isclose(got["inverse_variance"]["mean"], 0.6)
    assert math.isclose(got["inverse_variance"]["nw5_se"]**2, 8 / 135)
    assert math.isclose(got["pooled"]["mean"], 1.5)
    assert math.isclose(got["pooled"]["nw5_se"]**2, 5 / 54)
    assert np.allclose(got["inverse_variance"]["nw5_ci95"],
                       [0.6 - EXP9.Z95 * math.sqrt(8 / 135),
                        0.6 + EXP9.Z95 * math.sqrt(8 / 135)])


def test_cluster_resamples_folds_not_just_blocks_and_zero_variance_is_explicit(monkeypatch):
    monkeypatch.setattr(EXP9, "BOOT_DRAWS", 10_000)
    parts = {f: pd.Series(np.full(12, value), index=pd.bdate_range(f"{2020+i*2}-01-03", periods=12))
             for i, (f, value) in enumerate(zip(EXP9.FOLDS, (-1., 0., 1.)))}
    got = EXP9.pooled_with_folds(parts)
    # Within-fold bootstrap is degenerate, but fold-cluster bootstrap must move.
    assert got["pooled"]["stationary_bootstrap_ci95"] == [0., 0.]
    assert got["fold_cluster_bootstrap"]["ci95"] == [-1., 1.]
    assert got["fold_cluster_bootstrap"]["bootstrap_se"] > 0
    assert got["inverse_variance"]["mean"] is None
    assert "zero" in got["inverse_variance"]["status"]
    json.dumps(EXP9.json_safe(got), allow_nan=False)


def test_pooling_rejects_duplicate_dates_and_insufficient_fold_days(monkeypatch):
    monkeypatch.setattr(EXP9, "BOOT_DRAWS", 10)
    dates = pd.bdate_range("2020-07-01", periods=3)
    with pytest.raises(ValueError, match="disjoint"):
        EXP9.pooled_with_folds({36: pd.Series([1, 2, 3], index=dates),
                               39: pd.Series([4, 5, 6], index=dates)})
    with pytest.raises(ValueError, match="two valid days"):
        EXP9.pooled_with_folds({36: pd.Series([1.]), 39: pd.Series([2., 3.])})


def test_paired_se_drops_unmatched_extreme_day():
    dates = pd.bdate_range("2022-01-03", periods=4)
    r = pd.Series([-1., 0., 1., 1000.], index=dates)
    ic = pd.Series([-2., 0., 2.], index=dates[:3])
    got = EXP9.paired_se_comparison(r, ic)
    assert got["n_paired_days"] == 3
    assert got["n_reliability_days"] == 4
    assert math.isclose(got["ic_over_reliability_se_ratio"], 2.)
    assert math.isclose(got["log10_se_ratio"], math.log10(2.))


@pytest.mark.parametrize("r", [None, float("nan"), -0.1, 0., 1.])
def test_sb_boundary_does_not_abort_other_tables(r):
    got = EXP9.spearman_brown({5: r, 10: 0.8, 20: 0.9, 40: 0.95})
    assert got["prediction_vs_measurement"]["10"]["predicted_reliability"] is None
    assert got["by_sample_count"]["10"]["status"] == "ok"
    json.dumps(EXP9.json_safe(got), allow_nan=False)


def test_all_sample_count_sb_decompositions():
    got = EXP9.spearman_brown({5: 0.5, 10: 0.75, 20: 0.9, 40: 0.95})
    assert math.isclose(got["by_sample_count"]["10"][
        "signal_variance_over_sampling_noise_variance"], 3.)
    assert math.isclose(got["by_sample_count"]["40"][
        "sample_count_multiple_needed_for_reliability_0.95"], 1.)


def test_runtime_accounts_for_missing_history_without_charging_it_to_new_run(tmp_path, monkeypatch):
    monkeypatch.setattr(EXP9, "REPO", tmp_path)
    for repeat in EXP9.REPEATS:
        for fold in EXP9.FOLDS:
            for sc in EXP9.SAMPLE_COUNTS:
                path = EXP9.score_file(fold, sc, repeat).parent / "metrics.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "scoring_config": {"amp": "bf16", "batch_size": 128,
                                       "sample_count": sc, "lookback": 90, "predict": 6},
                    "val_window": list(EXP9.VAL_WINDOWS[fold]),
                }
                if repeat == 2 or sc != 5:
                    metadata["runtime"] = {"device": "cuda", "scoring_seconds": 60.}
                path.write_text(json.dumps(metadata), encoding="utf-8")
    log_dir = tmp_path / "outputs" / "exp9_queue_logs"
    log_dir.mkdir()
    for name in ("sc5_fold36_r2.a1.log", "sc5_fold36_r2.a2.log", "sc5_fold36_r2.a2.err.log"):
        (log_dir / name).write_text("synthetic", encoding="utf-8")
    got = EXP9.collect_gpu_runtime()
    assert got["new_r2"]["total_seconds"] == 720.
    assert got["reused_r1"]["total_seconds"] is None
    assert got["reused_r1"]["verified_subtotal_seconds"] == 540.
    assert got["reused_r1"]["unverified_cells"] == 3
    assert got["failed_attempt_seconds"] is None
    assert got["failed_attempts_lower_bound"] == 1
    assert len(got["observed_attempt_stdout_logs"]) == 2


def test_review_gate_precedes_every_data_or_memory_read(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["exp9_test_retest_reliability.py"])

    def forbidden(*args, **kwargs):
        pytest.fail("unreviewed entry point touched data or memory")

    monkeypatch.setattr(EXP9, "committed_memory_gb", forbidden)
    monkeypatch.setattr(EXP9, "_load_compare_arms_money", forbidden)
    monkeypatch.setattr(EXP9, "load_scores", forbidden)
    monkeypatch.setattr(EXP9, "collect_gpu_runtime", forbidden)
    with pytest.raises(SystemExit) as exc:
        EXP9.main()
    assert exc.value.code == 2
