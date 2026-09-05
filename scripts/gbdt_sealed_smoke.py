"""End-to-end smoke for the tree baseline's compute-only mode, on synthetic data.

Lives in ``scripts/`` rather than ``tests/`` because it needs xgboost, which only
exists in ``.venv-gbdt`` — and ``.venv-gbdt`` has no pytest.  ``tests/test_gbdt_sealed.py``
imports :func:`run` so the same path is exercised from either interpreter.

    .venv-gbdt\\Scripts\\python.exe scripts\\gbdt_sealed_smoke.py

It drives the real ``run_fold(..., sealed=True)``: inner selection, three seed
fits, the frozen mean ensemble, sentinel plus manifest, directory audit.  The
label loader and every metric entry point are replaced by land mines first, so
the run fails loudly if the compute-only path ever reaches them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 直接按路径加载，绕开 crsp_pipeline 包的 __init__（.venv-gbdt 只装了树栈）
_sealed = _load("crsp_pipeline_sealed_for_smoke", ROOT / "src" / "crsp_pipeline" / "sealed.py")
FORBIDDEN_FILES = _sealed.FORBIDDEN_FILES
SealedReadError = _sealed.SealedReadError
assert_readable = _sealed.assert_readable
audit_dir = _sealed.audit_dir

gbdt = _load("gbdt_baseline_for_smoke", ROOT / "scripts" / "gbdt_baseline.py")

SEEDS = [11, 29, 47]
BANNED_NAMES = set(FORBIDDEN_FILES) | {"fold_summary.json", "scores_ensemble.parquet"}
BANNED_PATTERNS = ("daily_ic",)


def _synthetic_cache(tmp_path: Path, dates: pd.DatetimeIndex, permnos: np.ndarray) -> Path:
    cache = tmp_path / "cache"
    (cache / "base").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for year in sorted({int(d.year) for d in dates}):
        rows = dates[dates.year == year]
        frame = pd.DataFrame({
            "PERMNO": np.tile(permnos, len(rows)),
            "date": np.repeat(rows.to_numpy(), len(permnos)),
        })
        for column in gbdt.STOCK_FEATURES:
            frame[column] = rng.random(len(frame)).astype(np.float32)
        frame["y"] = (frame["ret1"] * 0.1 + rng.normal(0, 0.02, len(frame))).astype(np.float32)
        frame["label_end_date"] = frame["date"] + pd.Timedelta(days=8)
        frame = frame[["PERMNO", "date", *gbdt.STOCK_FEATURES, "y", "label_end_date"]]
        frame.to_parquet(cache / "base" / f"stock_features_{year}.parquet", index=False)
    (cache / "base" / "manifest.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    state = pd.DataFrame(
        {f"jkp_f{i}_cum5": rng.normal(0, 0.01, len(dates)).astype(np.float32) for i in range(4)}
    )
    state.insert(0, "date", dates)
    state.to_parquet(cache / "jkp_state.parquet", index=False)
    (cache / "jkp_manifest.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    return cache


def _synthetic_config(snapshot: Path) -> dict:
    return {
        "max_rss_gb": 64.0,
        "num_threads": 2,
        "tuning_seed": 20260831,
        "seeds": SEEDS,
        "round_checkpoints": [5],
        "models": ["xgboost"],
        "grids": {"xgboost": {"learning_rate": [0.05], "max_depth": [3],
                              "min_child_weight": [10]}},
        "inner_validation": {"tail_trading_days": 20, "tune_max_stocks_per_day": 40,
                             "selection_metric": "mean daily cross-sectional RankIC"},
        "features": {"jkp_state_snapshot": str(snapshot),
                     "jkp_state_snapshot_sha256": "0" * 64},
        "compute": {"min_available_ram_gb_before_fold": 0.0},
    }


def run(tmp_path: Path) -> Path:
    """Run one compute-only fold on synthetic data; return its sealed directory."""
    dates = pd.DatetimeIndex(pd.bdate_range("2004-01-05", periods=180))
    permnos = np.arange(1001, 1081, dtype=np.int64)
    cache = _synthetic_cache(tmp_path, dates, permnos)
    config = _synthetic_config(cache / "jkp_state.parquet")
    fold = gbdt.Fold("fold05", str(dates[0].date()), str(dates[149].date()),
                     str(dates[150].date()), str(dates[-1].date()))
    out_dir = tmp_path / "out"

    def _boom(*_args, **_kwargs):
        raise AssertionError("the compute-only path reached a label or metric entry point")

    original = (gbdt._load_labels, gbdt._daily_ic, gbdt._fold_result, gbdt._label_path)
    gbdt._load_labels = _boom
    gbdt._daily_ic = _boom
    gbdt._fold_result = _boom
    gbdt._label_path = _boom
    try:
        gbdt.run_fold(fold, ["xgboost"], out_dir, cache, config, "deadbeef" * 8,
                      sealed=True, config_path=Path("configs/synthetic.json"))
    finally:
        (gbdt._load_labels, gbdt._daily_ic, gbdt._fold_result, gbdt._label_path) = original
    return out_dir / "xgboost" / "sealed" / fold.name


def check(directory: Path) -> None:
    report = audit_dir(directory, extra_allowed=gbdt._sealed_allowed(SEEDS))
    assert report["clean"], report
    names = {p.name for p in directory.iterdir()}
    assert not names & BANNED_NAMES, sorted(names & BANNED_NAMES)
    assert not [n for n in names for pat in BANNED_PATTERNS if pat in n], sorted(names)

    scores = pd.read_parquet(directory / "scores.parquet")
    assert list(scores.columns) == ["PERMNO", "signal_date", "score"]
    assert len(scores) and not scores[["PERMNO", "signal_date"]].duplicated().any()
    stacked = np.column_stack([
        pd.read_parquet(directory / f"scores_seed{seed}.parquet")["score"].to_numpy()
        for seed in SEEDS
    ])
    np.testing.assert_allclose(scores["score"].to_numpy(), stacked.mean(axis=1), rtol=1e-9)

    manifest = json.loads((directory / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))
    for key in ("snapshot_id", "code_sha256", "config_sha256", "jkp_snapshot_sha256",
                "scores_sha256", "val_window", "seeds", "model", "sealed_utc"):
        assert key in manifest, key
    assert manifest["scores_sha256"] == gbdt._sha256(directory / "scores.parquet")

    try:
        assert_readable(directory / "scores.parquet")
    except SealedReadError:
        pass
    else:
        raise AssertionError("the read guard did not cover the sealed directory")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = run(Path(tmp))
        check(directory)
        print(f"[smoke] compute-only end-to-end OK: "
              f"{sorted(p.name for p in directory.iterdir())}", flush=True)


if __name__ == "__main__":
    main()
