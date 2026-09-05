"""Nested, multi-seed CPU tree baselines with lagged JKP regime features.

The frozen experiment lives in ``configs/gbdt_strong_v1.json``.  This pipeline
builds stock features before applying the point-in-time universe filter, tunes
inside each outer training window, and evaluates on the same frozen official
labels as Kronos.  All artifacts are resumable.

Compute-only mode (``--sealed``)
-------------------------------
The 2026-09-05 authorisation extends the 2026-09-01 compute-only grant to the
tree baseline on the untouched folds: train the frozen XGBoost recipe and emit
scores, **compute no metric of any kind**.  In that mode this script

* takes its fold table from ``--folds-json`` (mechanically produced by
  ``scripts/emit_folds.py``; the seven development folds stay the default),
* never loads ``labels.parquet`` and never calls the daily-IC path,
* keeps its per-year feature cache — which carries the training target column
  ``y`` — inside a directory that itself carries the ``SEALED`` sentinel,
* writes each fold's scores plus sentinel and manifest through
  ``crsp_pipeline.sealed.write_seal`` and checks the directory with
  ``audit_dir``.

The hyperparameter grid, the seeds, the inner selection rule and the round
checkpoints are untouched; only the window and the JKP snapshot are widened.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_sealed_module() -> Any:
    """Load ``crsp_pipeline/sealed.py`` by path.

    ``.venv-gbdt`` deliberately holds only the tree stack, so importing the
    ``crsp_pipeline`` package (which pulls in yaml and friends) is not an option
    here.  ``sealed.py`` itself is pure standard library.
    """
    spec = importlib.util.spec_from_file_location(
        "crsp_pipeline_sealed_direct", ROOT / "src" / "crsp_pipeline" / "sealed.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot locate src/crsp_pipeline/sealed.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SEALED = _load_sealed_module()
audit_dir = _SEALED.audit_dir
write_seal = _SEALED.write_seal
PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
DEFAULT_CONFIG = ROOT / "configs" / "gbdt_strong_v2.json"
DEFAULT_OUT = ROOT / "outputs" / "gbdt_strong_jkp_v2"
HORIZON = 6
PIPELINE_BUILD_ID = "gbdt-strong-v2-20260831"
BASE_CACHE_BUILD_ID = "stock-features-v2-adjusted-exact-calendar"
CACHE_SCHEMA_VERSION = 3
SCORE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str


FOLDS = (
    Fold("fold36", "2017-07-03", "2020-06-22", "2020-07-01", "2020-12-31"),
    Fold("fold37", "2018-01-02", "2020-12-22", "2021-01-04", "2021-06-30"),
    Fold("fold38", "2018-07-02", "2021-06-22", "2021-07-01", "2021-12-31"),
    Fold("fold39", "2019-01-02", "2021-12-22", "2022-01-03", "2022-06-30"),
    Fold("fold40", "2019-07-01", "2022-06-22", "2022-07-01", "2022-12-30"),
    Fold("fold41", "2020-01-02", "2022-12-21", "2023-01-03", "2023-06-30"),
    Fold("fold42", "2020-07-01", "2023-06-22", "2023-07-03", "2023-12-29"),
)

# The active table defaults to the seven frozen development folds and is only
# replaced by ``--folds-json``, whose contents must come from
# ``scripts/emit_folds.py`` (hand-written windows are forbidden).
ACTIVE_FOLDS: tuple[Fold, ...] = FOLDS

SEALED_DIR_NAME = "sealed"
SEALED_LOG_NAME = "sealed_run.log"
# The frozen per-fold artifacts a compute-only run is allowed to leave behind on
# top of what ``crsp_pipeline.sealed`` already permits.  ``tuning.json`` records
# the inner selection; because its inner-split RankIC values fall inside the
# untouched range for the later folds it stays under the sentinel and is never
# printed.
SEALED_EXTRA_BASE = {"tuning.json"}


def _sealed_allowed(seeds: Iterable[int]) -> set[str]:
    return SEALED_EXTRA_BASE | {
        f"scores_seed{int(seed)}.{suffix}"
        for seed in seeds
        for suffix in ("parquet", "json")
    }


def _load_folds_json(path: Path) -> tuple[Fold, ...]:
    """Fold windows emitted by ``scripts/emit_folds.py`` — never hand-written."""
    # utf-8-sig: PowerShell's Set-Content -Encoding utf8 leaves a BOM on new files
    records = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    folds = tuple(
        Fold(str(item["n"]), str(item["ts"]), str(item["te"]), str(item["vs"]), str(item["ve"]))
        for item in records
    )
    if not folds:
        raise ValueError(f"Empty fold table: {path}")
    names = [fold.name for fold in folds]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate fold names in {path}")
    return folds

STOCK_FEATURES = (
    *(f"ret{w}" for w in (1, 2, 3, 5, 10, 20, 60)),
    *(f"{kind}{w}" for w in (5, 10, 20, 60) for kind in ("vol", "ma", "vratio")),
    "pos20", "pos60", "hl", "co", "turn", "logcap", "logadv",
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_status_gb() -> tuple[float, float, float]:
    if os.name != "nt":
        return math.nan, math.nan, math.inf
    size_t = ctypes.c_size_t

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", size_t), ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t), ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t), ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t), ("PeakPagefileUsage", size_t),
        ]

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()
    get_process_memory = kernel32.K32GetProcessMemoryInfo
    get_process_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory.restype = ctypes.c_int
    ok_process = get_process_memory(
        handle, ctypes.byref(counters), ctypes.sizeof(counters)
    )
    memory = MemoryStatusEx()
    memory.dwLength = ctypes.sizeof(memory)
    ok_system = kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    gib = float(1024**3)
    return (
        counters.WorkingSetSize / gib if ok_process else math.nan,
        counters.PeakWorkingSetSize / gib if ok_process else math.nan,
        memory.ullAvailPhys / gib if ok_system else math.nan,
    )


def _check_memory(config: dict[str, Any], stage: str, require_available: bool = False) -> None:
    rss, peak, available = _memory_status_gb()
    print(
        f"[memory] {stage}: rss={rss:.2f}GB peak={peak:.2f}GB available={available:.2f}GB",
        flush=True,
    )
    cap = float(config["max_rss_gb"])
    observed_peak = max(rss, peak)
    if np.isfinite(observed_peak) and observed_peak > cap:
        raise MemoryError(
            f"Peak RSS {observed_peak:.2f}GB exceeded the frozen {cap:.2f}GB cap"
        )
    if require_available:
        minimum = float(config["compute"]["min_available_ram_gb_before_fold"])
        if np.isfinite(available) and available < minimum:
            raise MemoryError(
                f"Only {available:.2f}GB RAM is available before {stage}; need {minimum:.2f}GB. "
                "Resume after the other training process releases memory."
            )


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config["sealed_raw_data_cutoff"] != "2023-12-29":
        raise ValueError("The raw-data cutoff must remain 2023-12-29")
    return config, hashlib.sha256(raw).hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("lightgbm", "xgboost", "catboost", "pandas", "numpy")
    }


def _check_runtime_versions(config: dict[str, Any]) -> None:
    actual = _runtime_versions()
    expected = config["package_versions"]
    if actual != expected:
        raise RuntimeError(f"Frozen package versions differ: expected={expected}, actual={actual}")


def _freeze_preregister(config_path: Path, out_dir: Path, name: str = "preregister.json") -> None:
    target = out_dir / name
    source = config_path.read_bytes()
    if target.exists() and target.read_bytes() != source:
        raise RuntimeError(f"Frozen preregistration differs from {config_path}: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path, target)


def _date_filters(column: str, lo: pd.Timestamp, hi: pd.Timestamp) -> list[tuple[str, str, Any]]:
    return [(column, ">=", lo.to_pydatetime()), (column, "<=", hi.to_pydatetime())]


def _rolling(grouped: Any, window: int, operation: str, min_periods: int) -> pd.Series:
    return getattr(grouped.rolling(window, min_periods=min_periods), operation)().reset_index(
        level=0, drop=True
    )


def _build_base_year(year: int, target_lo: pd.Timestamp, target_hi: pd.Timestamp) -> pd.DataFrame:
    """Build one target year with bounded buffers so native peak memory stays below 8GB."""
    read_lo = target_lo - timedelta(days=150)
    cutoff = pd.Timestamp("2023-12-29")
    read_hi = min(target_hi + timedelta(days=14), cutoff)
    filters = _date_filters("DlyCalDt", read_lo, read_hi)
    adjusted = pd.read_parquet(
        PROCESSED / "panel_kronos_adj.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
                 "DlyVol", "DlyPrcVol"],
        filters=filters,
    )
    raw = pd.read_parquet(
        PROCESSED / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyRet", "DlyCap"],
        filters=filters,
    )
    df = adjusted.merge(raw, on=["PERMNO", "DlyCalDt"], validate="one_to_one")
    del adjusted, raw
    df = df.sort_values(["PERMNO", "DlyCalDt"], kind="stable").reset_index(drop=True)
    all_dates = np.sort(df["DlyCalDt"].dropna().unique())
    date_to_pos = pd.Series(np.arange(len(all_dates), dtype=np.int32), index=all_dates)
    df["_date_pos"] = df["DlyCalDt"].map(date_to_pos).astype(np.int32)
    grouped = df.groupby("PERMNO", sort=False, observed=True)

    feat: dict[str, pd.Series | np.ndarray] = {}
    close = df["DlyClose"]
    for window in (1, 2, 3, 5, 10, 20, 60):
        feat[f"ret{window}"] = (close / grouped["DlyClose"].shift(window) - 1.0).astype(np.float32)
    for window in (5, 10, 20, 60):
        minimum = max(2, window // 2)
        feat[f"vol{window}"] = _rolling(grouped["DlyRet"], window, "std", minimum).astype(np.float32)
        mean_close = _rolling(grouped["DlyClose"], window, "mean", minimum)
        mean_volume = _rolling(grouped["DlyVol"], window, "mean", minimum).replace(0, np.nan)
        feat[f"ma{window}"] = (close / mean_close - 1.0).astype(np.float32)
        feat[f"vratio{window}"] = (df["DlyVol"] / mean_volume).astype(np.float32)
    for window in (20, 60):
        high = _rolling(grouped["DlyHigh"], window, "max", window // 2)
        low = _rolling(grouped["DlyLow"], window, "min", window // 2)
        feat[f"pos{window}"] = ((close - low) / (high - low).replace(0, np.nan)).astype(np.float32)
    feat["hl"] = ((df["DlyHigh"] - df["DlyLow"]) / close.abs().replace(0, np.nan)).astype(np.float32)
    feat["co"] = (close / df["DlyOpen"].abs().replace(0, np.nan) - 1.0).astype(np.float32)
    feat["turn"] = (df["DlyPrcVol"] / df["DlyCap"].replace(0, np.nan)).astype(np.float32)
    feat["logcap"] = np.log(df["DlyCap"].clip(lower=1)).astype(np.float32)
    feat["logadv"] = np.log(_rolling(grouped["DlyPrcVol"], 20, "mean", 10).clip(lower=1)).astype(np.float32)

    # Exact six market sessions on the unfiltered stock history.  A missing
    # session invalidates the label rather than silently extending its horizon.
    current_pos = df["_date_pos"].to_numpy(dtype=np.int32, copy=False)
    future_gross = np.ones(len(df), dtype=np.float64)
    label_valid = np.ones(len(df), dtype=bool)
    for step in range(1, HORIZON + 1):
        shifted_ret = grouped["DlyRet"].shift(-step).to_numpy(dtype=np.float64, copy=False)
        shifted_pos = grouped["_date_pos"].shift(-step).to_numpy(dtype=np.float64, copy=False)
        valid_step = np.isfinite(shifted_ret) & (shifted_ret >= -1.0) & (shifted_pos == current_pos + step)
        label_valid &= valid_step
        future_gross[valid_step] *= 1.0 + shifted_ret[valid_step]
    y_raw = np.full(len(df), np.nan, dtype=np.float32)
    y_raw[label_valid] = (future_gross[label_valid] - 1.0).astype(np.float32)
    label_end = np.full(len(df), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid_end = label_valid & (current_pos + HORIZON < len(all_dates))
    label_end[valid_end] = all_dates[current_pos[valid_end] + HORIZON]

    features = pd.DataFrame(feat)
    features.insert(0, "date", df["DlyCalDt"].to_numpy())
    features.insert(0, "PERMNO", df["PERMNO"].to_numpy(dtype=np.int64, copy=False))
    features["y_raw"] = y_raw
    features["label_end_date"] = label_end
    del feat, future_gross, label_valid, y_raw, label_end, grouped, df
    gc.collect()

    features = features[(features["date"] >= target_lo) & (features["date"] <= target_hi)]
    universe = pd.read_parquet(
        PROCESSED / "universe.parquet",
        columns=["PERMNO", "DlyCalDt", "in_universe"],
        filters=_date_filters("DlyCalDt", target_lo, target_hi),
    )
    universe = universe.loc[universe["in_universe"], ["PERMNO", "DlyCalDt"]].rename(
        columns={"DlyCalDt": "date"}
    )
    features = features.merge(universe, on=["PERMNO", "date"], validate="one_to_one")
    del universe
    by_date = features.groupby("date", sort=False, observed=True)
    for column in STOCK_FEATURES:
        features[column] = by_date[column].rank(pct=True).astype(np.float32)
    features["y"] = (features["y_raw"] - by_date["y_raw"].transform("mean")).astype(np.float32)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features[["PERMNO", "date", *STOCK_FEATURES, "y", "label_end_date"]]
    features = features.sort_values(["date", "PERMNO"], kind="stable").reset_index(drop=True)
    if features.duplicated(["PERMNO", "date"]).any():
        raise AssertionError(f"Duplicate anchors in base feature year {year}")
    return features


def _feature_builder_sha256() -> str:
    source = inspect.getsource(_build_base_year).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _prepare_base_cache(cache_dir: Path, config: dict[str, Any], config_sha: str) -> None:
    base_dir = cache_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_dir / "manifest.json"
    first_signal = pd.Timestamp(min(f.train_start for f in ACTIVE_FOLDS))
    cutoff = pd.Timestamp(config["sealed_raw_data_cutoff"])
    # Never build years no active fold can reach.  For the seven development
    # folds the last validation date is the cutoff itself, so this is a no-op
    # there; for an earlier fold table it avoids touching later raw data.
    last_needed = min(cutoff, pd.Timestamp(max(f.val_end for f in ACTIVE_FOLDS)))
    years = list(range(first_signal.year, last_needed.year + 1))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_header = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "base_cache_build_id": BASE_CACHE_BUILD_ID,
            "config_sha256": config_sha,
            "sealed_raw_data_cutoff": config["sealed_raw_data_cutoff"],
            "stock_features": list(STOCK_FEATURES),
            "feature_builder_sha256": _feature_builder_sha256(),
        }
        for key, value in expected_header.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Stale base-cache manifest field {key}: {manifest_path}")
        files = manifest.get("files", {})
        for year in years:
            path = base_dir / f"stock_features_{year}.parquet"
            record = files.get(path.name)
            if not path.exists() or not record:
                raise RuntimeError(f"Incomplete base cache: {path}")
            if path.stat().st_size != int(record["size_bytes"]) or _sha256(path) != record["sha256"]:
                raise RuntimeError(f"Base-cache fingerprint mismatch: {path}")
        print(f"[prepare] verified fail-closed base cache: {manifest_path}", flush=True)
        return
    existing = list(base_dir.glob("stock_features_*.parquet"))
    if existing:
        raise RuntimeError(
            f"Unsealed partial base cache exists without a valid manifest: {existing[0]}"
        )
    file_records: dict[str, Any] = {}
    for year in years:
        path = base_dir / f"stock_features_{year}.parquet"
        target_lo = max(first_signal, pd.Timestamp(f"{year}-01-01"))
        target_hi = min(last_needed, pd.Timestamp(f"{year}-12-31"))
        print(f"[prepare] stock features {year}: {target_lo.date()}..{target_hi.date()}", flush=True)
        frame = _build_base_year(year, target_lo, target_hi)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
        file_records[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
            "rows": int(len(frame)),
        }
        print(f"[prepare] wrote {path.name}: {len(frame):,} rows", flush=True)
        del frame
        gc.collect()
        _check_memory(config, f"base cache {year}")
    _json_dump(
        manifest_path,
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "base_cache_build_id": BASE_CACHE_BUILD_ID,
            "config_sha256": config_sha,
            "sealed_raw_data_cutoff": config["sealed_raw_data_cutoff"],
            "stock_features": list(STOCK_FEATURES),
            "feature_builder_sha256": _feature_builder_sha256(),
            "construction": "year chunks; unfiltered history first, universe anchors second",
            "files": file_records,
        },
    )


def _prepare_jkp_cache(cache_dir: Path, config: dict[str, Any], config_sha: str) -> None:
    path = cache_dir / "jkp_state.parquet"
    manifest_path = cache_dir / "jkp_manifest.json"
    spec = config["features"]
    snapshot = Path(spec["jkp_state_snapshot"])
    expected_sha = spec["jkp_state_snapshot_sha256"]
    cutoff = pd.Timestamp(config["sealed_raw_data_cutoff"])
    if not snapshot.exists() or _sha256(snapshot) != expected_sha:
        raise RuntimeError(f"Physical cutoff JKP snapshot fingerprint mismatch: {snapshot}")
    if path.exists() or manifest_path.exists():
        if not (path.exists() and manifest_path.exists()):
            raise RuntimeError("Incomplete JKP cache/manifest pair")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "config_sha256": config_sha,
            "snapshot_sha256": expected_sha,
            "cache_sha256": _sha256(path),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Stale JKP-cache manifest field {key}: {manifest_path}")
        return
    state = pd.read_parquet(snapshot)
    state["date"] = pd.to_datetime(state["date"])
    expected_features = int(spec["jkp_series"]) * len(spec["jkp_windows"])
    if state.shape[1] - 1 != expected_features:
        raise AssertionError(f"Expected {expected_features} JKP state columns, got {state.shape[1] - 1}")
    required_state = state[state["date"] >= pd.Timestamp(min(f.train_start for f in ACTIVE_FOLDS))]
    if state["date"].duplicated().any() or not np.isfinite(
        required_state.drop(columns="date").to_numpy()
    ).all():
        raise AssertionError("Duplicate dates or non-finite values in required JKP state dates")
    if state["date"].max() > cutoff:
        raise AssertionError("JKP cache crossed the raw-data cutoff")
    tmp = path.with_suffix(".parquet.tmp")
    shutil.copyfile(snapshot, tmp)
    tmp.replace(path)
    cache_sha = _sha256(path)
    _json_dump(
        manifest_path,
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "config_sha256": config_sha,
            "snapshot": str(snapshot),
            "snapshot_sha256": expected_sha,
            "cache_sha256": cache_sha,
            "retained_start": str(state["date"].min().date()),
            "retained_end": str(state["date"].max().date()),
            "feature_count": int(state.shape[1] - 1),
            "lag_observations": spec["jkp_lag_observations"],
            "windows": spec["jkp_windows"],
        },
    )
    print(f"[prepare] wrote {path.name}: {state.shape}", flush=True)
    del state
    gc.collect()


def prepare_caches(cache_dir: Path, config: dict[str, Any], config_sha: str) -> None:
    _check_memory(config, "cache preparation", require_available=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_base_cache(cache_dir, config, config_sha)
    _prepare_jkp_cache(cache_dir, config, config_sha)
    _check_memory(config, "cache preparation complete")


def _provenance(cache_dir: Path, config_sha: str) -> dict[str, Any]:
    base_manifest = cache_dir / "base" / "manifest.json"
    jkp_manifest = cache_dir / "jkp_manifest.json"
    return {
        "pipeline_build_id": PIPELINE_BUILD_ID,
        "pipeline_script_sha256": _sha256(Path(__file__)),
        "config_sha256": config_sha,
        "base_manifest_sha256": _sha256(base_manifest),
        "jkp_manifest_sha256": _sha256(jkp_manifest),
        "package_versions": _runtime_versions(),
    }


def _score_sidecar(score_path: Path) -> Path:
    return score_path.with_suffix(".json")


def _score_complete(score_path: Path, expected: dict[str, Any]) -> bool:
    sidecar = _score_sidecar(score_path)
    if not score_path.exists() and not sidecar.exists():
        return False
    if not score_path.exists() or not sidecar.exists():
        raise RuntimeError(f"Incomplete score/sidecar pair: {score_path}")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"Stale score provenance field {key}: {score_path}")
    if record.get("score_sha256") != _sha256(score_path):
        raise RuntimeError(f"Score fingerprint mismatch: {score_path}")
    return True


def _load_base(cache_dir: Path, lo: str, hi: str) -> pd.DataFrame:
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
    frames: list[pd.DataFrame] = []
    for year in range(lo_ts.year, hi_ts.year + 1):
        year_lo = max(lo_ts, pd.Timestamp(f"{year}-01-01"))
        year_hi = min(hi_ts, pd.Timestamp(f"{year}-12-31"))
        frames.append(
            pd.read_parquet(
                cache_dir / "base" / f"stock_features_{year}.parquet",
                filters=_date_filters("date", year_lo, year_hi),
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result["label_end_date"] = pd.to_datetime(result["label_end_date"])
    return result.sort_values(["date", "PERMNO"], kind="stable").reset_index(drop=True)


def _load_state(cache_dir: Path) -> pd.DataFrame:
    state = pd.read_parquet(cache_dir / "jkp_state.parquet")
    state["date"] = pd.to_datetime(state["date"])
    return state.set_index("date").sort_index()


def _systematic_sample_by_date(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    positions: list[np.ndarray] = []
    for locs in frame.groupby("date", sort=True, observed=True).indices.values():
        locs = np.asarray(locs, dtype=np.int64)
        if len(locs) > maximum:
            locs = locs[np.linspace(0, len(locs) - 1, maximum, dtype=np.int64)]
        positions.append(locs)
    chosen = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    return frame.iloc[chosen].sort_values(["date", "PERMNO"], kind="stable").reset_index(drop=True)


def _build_matrix(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    config: dict[str, Any],
    chunk_rows: int = 25_000,
) -> np.ndarray:
    dates = pd.DatetimeIndex(frame["date"])
    codes = state.index.get_indexer(dates)
    if (codes < 0).any():
        raise AssertionError(f"JKP state missing dates: {list(dates[codes < 0].unique()[:5])}")
    n_stock, n_state = len(STOCK_FEATURES), state.shape[1]
    matrix = np.empty((len(frame), n_stock + n_state), dtype=np.float32)
    matrix[:, :n_stock] = frame[list(STOCK_FEATURES)].to_numpy(dtype=np.float32, copy=False)
    values = state.to_numpy(dtype=np.float32, copy=False)
    for start in range(0, len(frame), chunk_rows):
        stop = min(start + chunk_rows, len(frame))
        matrix[start:stop, n_stock:] = values[codes[start:stop]]
    _check_memory(config, f"matrix {len(frame):,}x{matrix.shape[1]}")
    return matrix


def _mean_daily_rank_ic(score: np.ndarray, label: np.ndarray, dates: Iterable[Any]) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "score": score, "label": label})
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True, observed=True):
        group = group.dropna(subset=["score", "label"])
        if len(group) < 50:
            continue
        x = group["score"].rank().to_numpy(dtype=np.float64)
        y = group["label"].rank().to_numpy(dtype=np.float64)
        if x.std() and y.std():
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else math.nan


def _daily_ic(scores: pd.DataFrame, labels: pd.DataFrame, fold_name: str) -> tuple[pd.DataFrame, int]:
    if scores.duplicated(["PERMNO", "signal_date"]).any():
        raise AssertionError(f"Duplicate score keys in {fold_name}")
    if labels.duplicated(["PERMNO", "signal_date"]).any():
        raise AssertionError(f"Duplicate official-label keys in {fold_name}")
    valid_labels = labels.dropna(subset=["label"])
    merged = scores.merge(
        valid_labels[["PERMNO", "signal_date", "label"]],
        on=["PERMNO", "signal_date"], how="inner", validate="one_to_one",
    ).dropna(subset=["score", "label"])
    if len(merged) != len(valid_labels):
        raise AssertionError(
            f"Official-label coverage mismatch in {fold_name}: {len(merged):,}/{len(valid_labels):,}"
        )
    rows: list[dict[str, Any]] = []
    for date, group in merged.groupby("signal_date", sort=True, observed=True):
        if len(group) < 50:
            continue
        x = group["score"].rank().to_numpy(dtype=np.float64)
        y = group["label"].rank().to_numpy(dtype=np.float64)
        if x.std() and y.std():
            rows.append({
                "fold": fold_name, "signal_date": date,
                "rank_ic": float(np.corrcoef(x, y)[0, 1]), "n_obs": int(len(group)),
            })
    return pd.DataFrame(rows), int(len(merged))


def _nw_stats(values: Iterable[float], lag: int = 5) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    n = len(vector)
    if not n:
        return {"mean": math.nan, "se": math.nan, "t": math.nan, "n": 0, "lag": lag}
    residual = vector - vector.mean()
    variance = float(residual @ residual) / n
    for step in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - step / (lag + 1.0)
        variance += 2.0 * weight * float(residual[step:] @ residual[:-step]) / n
    se = math.sqrt(max(variance, 0.0) / n)
    return {
        "mean": float(vector.mean()), "se": float(se),
        "t": float(vector.mean() / se) if se > 0 else math.nan,
        "n": int(n), "lag": int(lag),
    }


def _grid(config: dict[str, Any], model: str) -> list[dict[str, Any]]:
    spec = config["grids"][model]
    keys = list(spec)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(spec[k] for k in keys))]


def _complexity(model: str, params: dict[str, Any]) -> float:
    if model == "lightgbm":
        return float(params["num_leaves"])
    return float(params["max_depth"] if model == "xgboost" else params["depth"])


def _choose_trial(model: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [trial for trial in trials if np.isfinite(trial["inner_rank_ic"])]
    if not valid:
        raise RuntimeError(f"Every {model} tuning trial returned non-finite RankIC")
    return min(
        valid,
        key=lambda trial: (
            -trial["inner_rank_ic"],
            trial["params"]["num_boost_round"],
            _complexity(model, trial["params"]),
        ),
    )


def _validate_tuning_artifact(
    value: dict[str, Any],
    model: str,
    fold: Fold,
    config: dict[str, Any],
    config_sha: str,
    provenance: dict[str, Any],
) -> None:
    expected_header = {
        "config_sha256": config_sha,
        "provenance": provenance,
        "model": model,
        "fold": fold.name,
    }
    for key, expected in expected_header.items():
        if value.get(key) != expected:
            raise RuntimeError(f"Stale tuning field {key} for {model}/{fold.name}")
    trials = value.get("trials")
    if not isinstance(trials, list):
        raise RuntimeError(f"Missing tuning trials for {model}/{fold.name}")
    allowed_params = [
        {**base, "num_boost_round": int(rounds)}
        for base in _grid(config, model)
        for rounds in config["round_checkpoints"]
    ]
    canonical_allowed = {json.dumps(params, sort_keys=True) for params in allowed_params}
    canonical_seen = [json.dumps(trial.get("params"), sort_keys=True) for trial in trials]
    if len(canonical_seen) != len(canonical_allowed) or set(canonical_seen) != canonical_allowed:
        raise RuntimeError(f"Tuning grid/checkpoints are incomplete or duplicated for {model}/{fold.name}")
    if value.get("best") != _choose_trial(model, trials):
        raise RuntimeError(f"Recorded best trial is not the frozen selection for {model}/{fold.name}")


def _lgb_params(config: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "objective": "regression", "metric": "l2",
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "feature_pre_filter": False, "force_col_wise": True,
        "device_type": "cpu", "num_threads": int(config["num_threads"]), "verbosity": -1,
        "seed": int(seed), "bagging_seed": int(seed + 101),
        "feature_fraction_seed": int(seed + 211), "data_random_seed": int(seed + 307),
    }


def _xgb_params(config: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "eta": float(params["learning_rate"]), "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
        "tree_method": "hist", "device": "cpu", "max_bin": 255,
        "nthread": int(config["num_threads"]), "seed": int(seed), "verbosity": 0,
    }


def _cat_params(config: dict[str, Any], params: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "loss_function": "RMSE", "learning_rate": float(params["learning_rate"]),
        "depth": int(params["depth"]), "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 0.8,
        "task_type": "CPU", "thread_count": int(config["num_threads"]),
        "random_seed": int(seed), "allow_writing_files": False, "verbose": False,
    }


def _tune_model(
    model: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_dates: Iterable[Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoints = [int(value) for value in config["round_checkpoints"]]
    max_rounds, seed = max(checkpoints), int(config["tuning_seed"])
    trials: list[dict[str, Any]] = []
    if model == "lightgbm":
        import lightgbm as lgb

        train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
        for base in _grid(config, model):
            booster = lgb.train(_lgb_params(config, base, seed), train_set, num_boost_round=max_rounds)
            for rounds in checkpoints:
                pred = booster.predict(x_valid, num_iteration=rounds)
                trials.append({
                    "params": {**base, "num_boost_round": rounds},
                    "inner_rank_ic": _mean_daily_rank_ic(pred, y_valid, valid_dates),
                })
            booster.free_dataset()
            del booster
            gc.collect()
    elif model == "xgboost":
        import xgboost as xgb

        dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=255)
        dvalid = xgb.QuantileDMatrix(x_valid, ref=dtrain, max_bin=255)
        for base in _grid(config, model):
            booster = xgb.train(_xgb_params(config, base, seed), dtrain, num_boost_round=max_rounds)
            for rounds in checkpoints:
                pred = booster.predict(dvalid, iteration_range=(0, rounds))
                trials.append({
                    "params": {**base, "num_boost_round": rounds},
                    "inner_rank_ic": _mean_daily_rank_ic(pred, y_valid, valid_dates),
                })
            del booster
            gc.collect()
        del dtrain, dvalid
    elif model == "catboost":
        from catboost import CatBoostRegressor, Pool

        train_pool = Pool(x_train, label=y_train)
        for base in _grid(config, model):
            estimator = CatBoostRegressor(iterations=max_rounds, **_cat_params(config, base, seed))
            estimator.fit(train_pool)
            for rounds in checkpoints:
                pred = estimator.predict(x_valid, ntree_end=rounds)
                trials.append({
                    "params": {**base, "num_boost_round": rounds},
                    "inner_rank_ic": _mean_daily_rank_ic(pred, y_valid, valid_dates),
                })
            del estimator
            gc.collect()
        del train_pool
    else:
        raise ValueError(model)
    return trials, _choose_trial(model, trials)


def _fit_predict(
    model: str,
    params: dict[str, Any],
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    rounds = int(params["num_boost_round"])
    base = {key: value for key, value in params.items() if key != "num_boost_round"}
    if model == "lightgbm":
        import lightgbm as lgb

        train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
        booster = lgb.train(_lgb_params(config, base, seed), train_set, num_boost_round=rounds)
        prediction = booster.predict(x_valid, num_iteration=rounds)
        booster.free_dataset()
        del booster, train_set
    elif model == "xgboost":
        import xgboost as xgb

        dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=255)
        dvalid = xgb.QuantileDMatrix(x_valid, ref=dtrain, max_bin=255)
        booster = xgb.train(_xgb_params(config, base, seed), dtrain, num_boost_round=rounds)
        prediction = booster.predict(dvalid, iteration_range=(0, rounds))
        del booster, dtrain, dvalid
    elif model == "catboost":
        from catboost import CatBoostRegressor

        estimator = CatBoostRegressor(iterations=rounds, **_cat_params(config, base, seed))
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_valid)
        del estimator
    else:
        raise ValueError(model)
    gc.collect()
    return np.asarray(prediction, dtype=np.float64)


def _label_path(fold: Fold) -> Path:
    return (
        ROOT / "outputs" / f"{fold.name}_lb90_s0_poolB_universe"
        / f"eval_amp_lb90_{fold.name}" / "labels.parquet"
    )


def _load_labels(fold: Fold) -> tuple[pd.DataFrame, str]:
    path = _label_path(fold)
    labels = pd.read_parquet(path, columns=["PERMNO", "signal_date", "label"])
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    if labels["signal_date"].min() < pd.Timestamp(fold.val_start) or labels["signal_date"].max() > pd.Timestamp(fold.val_end):
        raise AssertionError(f"Official label dates exceed {fold.name}'s signal-date window")
    return labels, _sha256(path)


def _inner_split(
    outer_train: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.sort(outer_train.loc[outer_train["y"].notna(), "date"].unique())
    tail = int(config["inner_validation"]["tail_trading_days"])
    if len(dates) <= tail:
        raise AssertionError("Not enough dates for the frozen inner split")
    inner_start = pd.Timestamp(dates[-tail])
    inner_train = outer_train[
        outer_train["y"].notna()
        & (outer_train["date"] < inner_start)
        & (outer_train["label_end_date"] < inner_start)
    ]
    inner_valid = outer_train[outer_train["y"].notna() & (outer_train["date"] >= inner_start)]
    inner_train = _systematic_sample_by_date(
        inner_train, int(config["inner_validation"]["tune_max_stocks_per_day"])
    )
    return inner_train, inner_valid, inner_start


def _check_packages(models: Iterable[str]) -> None:
    names = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}
    missing = [model for model in models if importlib.util.find_spec(names[model]) is None]
    if missing:
        raise RuntimeError(f"Missing packages in .venv-gbdt: {', '.join(missing)}")


def _smoke(cache_dir: Path, config: dict[str, Any]) -> None:
    fold = ACTIVE_FOLDS[0]
    base = _load_base(cache_dir, fold.train_start, fold.train_end)
    inner_train, inner_valid, inner_start = _inner_split(base, config)
    inner_train = _systematic_sample_by_date(inner_train.tail(80_000), 64)
    inner_valid = _systematic_sample_by_date(inner_valid, 64)
    state = _load_state(cache_dir)
    x_train, x_valid = _build_matrix(inner_train, state, config), _build_matrix(inner_valid, state, config)
    smoke_params = {
        "lightgbm": {
            "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 250, "num_boost_round": 5,
        },
        "xgboost": {
            "learning_rate": 0.05, "max_depth": 4,
            "min_child_weight": 50, "num_boost_round": 5,
        },
        "catboost": {
            "learning_rate": 0.05, "depth": 6,
            "l2_leaf_reg": 3.0, "num_boost_round": 5,
        },
    }
    y_train = inner_train["y"].to_numpy(np.float32)
    y_valid = inner_valid["y"].to_numpy(np.float32)
    for model, params in smoke_params.items():
        prediction = _fit_predict(
            model, params, int(config["tuning_seed"]), x_train, y_train, x_valid, config
        )
        rank_ic = _mean_daily_rank_ic(prediction, y_valid, inner_valid["date"])
        print(
            f"[smoke] model={model} inner_start={inner_start.date()} "
            f"train={len(inner_train):,} valid={len(inner_valid):,} "
            f"features={x_train.shape[1]} RankIC={rank_ic:+.5f}",
            flush=True,
        )
    del x_train, x_valid, base, inner_train, inner_valid, state, y_train, y_valid
    gc.collect()
    _check_memory(config, "smoke complete")


def _fold_result(
    model: str,
    fold: Fold,
    model_dir: Path,
    labels: pd.DataFrame,
    label_sha: str,
    seeds: list[int],
    best: dict[str, Any],
    config_sha: str,
    provenance: dict[str, Any],
) -> None:
    seed_scores: list[pd.DataFrame] = []
    seed_stats: dict[str, Any] = {}
    seed_daily_hashes: dict[str, str] = {}
    for seed in seeds:
        scores = pd.read_parquet(model_dir / f"scores_seed{seed}.parquet")
        scores["signal_date"] = pd.to_datetime(scores["signal_date"])
        daily, n_obs = _daily_ic(scores, labels, fold.name)
        daily_path = model_dir / f"daily_ic_seed{seed}.parquet"
        daily.to_parquet(daily_path, index=False)
        seed_daily_hashes[str(seed)] = _sha256(daily_path)
        seed_stats[str(seed)] = {**_nw_stats(daily["rank_ic"]), "n_obs": n_obs}
        seed_scores.append(scores.rename(columns={"score": f"score_{seed}"}))
    ensemble = seed_scores[0]
    for scores in seed_scores[1:]:
        ensemble = ensemble.merge(
            scores, on=["PERMNO", "signal_date"], how="inner", validate="one_to_one"
        )
    ensemble["score"] = ensemble[[f"score_{seed}" for seed in seeds]].mean(axis=1)
    ensemble = ensemble[["PERMNO", "signal_date", "score"]]
    ensemble.to_parquet(model_dir / "scores_ensemble.parquet", index=False)
    daily, n_obs = _daily_ic(ensemble, labels, fold.name)
    ensemble_daily_path = model_dir / "daily_ic_ensemble.parquet"
    daily.to_parquet(ensemble_daily_path, index=False)
    result = {
        "config_sha256": config_sha, "provenance": provenance,
        "model": model, "fold": fold.name,
        "best_inner_trial": best, "seeds": seeds,
        "official_label_path": str(_label_path(fold)),
        "official_label_sha256": label_sha,
        "ensemble": {**_nw_stats(daily["rank_ic"]), "n_obs": n_obs},
        "per_seed": seed_stats,
        "seed_daily_ic_sha256": seed_daily_hashes,
        "ensemble_daily_ic_sha256": _sha256(ensemble_daily_path),
    }
    _json_dump(model_dir / "fold_summary.json", result)
    print(
        f"[{model}] {fold.name}: ensemble RankIC={result['ensemble']['mean']:+.5f} "
        f"t={result['ensemble']['t']:+.2f} days={result['ensemble']['n']}",
        flush=True,
    )


def _append_ledger_line(fold: Fold, model: str, model_dir: Path, seeds: list[int]) -> None:
    """Append the one compute-only row the authorisation allows.

    Append-only, no metric of any kind: tag, model, validation window, row count
    and seeds.  The row count is read back from the manifest so this never opens
    the score file.  Idempotent by tag, because a crash between publishing and
    logging must not produce a second row on resume.
    """
    ledger = ROOT / "experiments" / "ledger.md"
    tag = f"sealed_xgb_{fold.name}" if model == "xgboost" else f"sealed_{model}_{fold.name}"
    if ledger.exists() and f"tag={tag} " in ledger.read_text(encoding="utf-8"):
        return
    manifest = json.loads((model_dir / "SEALED_MANIFEST.json").read_text(encoding="utf-8"))
    stamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    seed_text = ",".join(str(seed) for seed in seeds)
    line = (
        f"- {stamp} | sealed-compute | tag={tag} model={model} "
        f"val=[{fold.val_start}..{fold.val_end}] rows={manifest['n_rows']} "
        f"days={manifest['n_signal_dates']} seeds={seed_text} "
        "（封存打分：未生成 labels、未计算任何指标；计算授权 != 读取授权）"
    )
    with open(ledger, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _fold_dir(out_dir: Path, model: str, fold: Fold, sealed: bool) -> Path:
    """Compute-only runs live in their own subtree so nothing can be confused
    with a development-fold artifact."""
    if sealed:
        return out_dir / model / SEALED_DIR_NAME / fold.name
    return out_dir / model / fold.name


def _publish_seal(
    model_dir: Path,
    model: str,
    fold: Fold,
    seeds: list[int],
    config: dict[str, Any],
    config_sha: str,
    config_path: Path,
    provenance: dict[str, Any],
    best_params: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Average the seed scores, write the sentinel + manifest, audit the directory.

    Nothing here touches a label or produces a statistic: the ensemble is the
    frozen arithmetic mean of the three seed predictions, exactly the
    ``scores_ensemble`` recipe used on the development folds.
    """
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        scores = pd.read_parquet(model_dir / f"scores_seed{seed}.parquet")
        scores["signal_date"] = pd.to_datetime(scores["signal_date"])
        frames.append(scores.rename(columns={"score": f"score_{seed}"}))
    ensemble = frames[0]
    for frame in frames[1:]:
        ensemble = ensemble.merge(
            frame, on=["PERMNO", "signal_date"], how="inner", validate="one_to_one"
        )
    if len(ensemble) != len(frames[0]):
        raise AssertionError(f"Seed score keys disagree for {fold.name}")
    ensemble["score"] = ensemble[[f"score_{seed}" for seed in seeds]].mean(axis=1)
    ensemble = ensemble[["PERMNO", "signal_date", "score"]].sort_values(
        ["signal_date", "PERMNO"], kind="stable"
    ).reset_index(drop=True)
    score_path = model_dir / "scores.parquet"
    tmp = score_path.with_suffix(".parquet.tmp")
    ensemble.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(score_path)

    manifest = {
        "snapshot_id": str(PROCESSED.name),
        "processed_root": str(PROCESSED),
        "model": model,
        "fold": fold.name,
        "train_window": [fold.train_start, fold.train_end],
        "val_window": [fold.val_start, fold.val_end],
        "seeds": seeds,
        "config_sha256": config_sha,
        "config_path": str(config_path),
        "code_sha256": {
            "gbdt_baseline.py": _sha256(Path(__file__)),
            "config": config_sha,
        },
        "jkp_snapshot": config["features"]["jkp_state_snapshot"],
        "jkp_snapshot_sha256": config["features"]["jkp_state_snapshot_sha256"],
        "scores_sha256": _sha256(score_path),
        "seed_scores_sha256": {
            str(seed): _sha256(model_dir / f"scores_seed{seed}.parquet") for seed in seeds
        },
        "best_params": best_params,
        "provenance": provenance,
        "n_rows": int(len(ensemble)),
        "n_signal_dates": int(ensemble["signal_date"].nunique()),
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "scoring_config": {
            "model": model,
            "device": "CPU only",
            "num_threads": int(config["num_threads"]),
            "seeds": seeds,
            "ensemble": "arithmetic mean of the three seed predictions",
            "inner_validation": config["inner_validation"],
            "round_checkpoints": config["round_checkpoints"],
            "grid": config["grids"][model],
        },
        "authorisation": (
            "2026-09-05 用户授权（H3 处置取第一条）：树基线在折 05–35 上做"
            "『计算专用』的训练与打分，封存模式。只跑 XGBoost 主口径；超参网格 / "
            "seeds / 内层选参规则一字不改，只扩窗口与 JKP 快照。训练目标标签只在"
            "内存中消费，不得以可读形式落盘；不得计算、打印、登记任何封存窗口的 "
            "IC / 收益 / 分层统计。计算授权 != 读取授权。"
        ),
        "no_metrics": (
            "compute-only: no label was loaded for the validation window and no "
            "IC, return or stratified statistic was computed, printed or logged"
        ),
        "inner_tuning_artifact": (
            "tuning.json stays inside this directory; its inner-split RankIC "
            "values are training-window quantities of the frozen selection rule "
            "and are never printed or exported"
        ),
    }
    write_seal(model_dir, manifest)
    report = audit_dir(model_dir, extra_allowed=_sealed_allowed(seeds))
    if not report["clean"]:
        raise RuntimeError(f"Compute-only directory failed its audit: {report}")
    return report


def run_fold(
    fold: Fold,
    models: list[str],
    out_dir: Path,
    cache_dir: Path,
    config: dict[str, Any],
    config_sha: str,
    sealed: bool = False,
    config_path: Path = DEFAULT_CONFIG,
    append_ledger: bool = False,
) -> None:
    started = time.perf_counter()
    _check_memory(config, f"{fold.name} start", require_available=True)
    provenance = _provenance(cache_dir, config_sha)
    base = _load_base(cache_dir, fold.train_start, fold.val_end)
    outer_train = base[
        (base["date"] >= pd.Timestamp(fold.train_start))
        & (base["date"] <= pd.Timestamp(fold.train_end))
        & base["y"].notna()
        & (base["label_end_date"] < pd.Timestamp(fold.val_start))
    ].copy()
    outer_valid = base[
        (base["date"] >= pd.Timestamp(fold.val_start))
        & (base["date"] <= pd.Timestamp(fold.val_end))
    ].copy()
    del base
    if outer_train["label_end_date"].max() >= pd.Timestamp(fold.val_start):
        raise AssertionError(f"Outer-label purge failed for {fold.name}")
    state = _load_state(cache_dir)
    seeds = [int(value) for value in config["seeds"]]

    tuning: dict[str, dict[str, Any]] = {}
    missing_tuning: list[str] = []
    for model in models:
        path = _fold_dir(out_dir, model, fold, sealed) / "tuning.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            _validate_tuning_artifact(
                value, model, fold, config, config_sha, provenance
            )
            tuning[model] = value
        else:
            missing_tuning.append(model)
    if missing_tuning:
        inner_train, inner_valid, inner_start = _inner_split(outer_train, config)
        print(
            f"[{fold.name}] inner split {inner_start.date()}: "
            f"train(sampled)={len(inner_train):,}, valid={len(inner_valid):,}",
            flush=True,
        )
        x_inner_train = _build_matrix(inner_train, state, config)
        x_inner_valid = _build_matrix(inner_valid, state, config)
        y_inner_train = inner_train["y"].to_numpy(dtype=np.float32, copy=False)
        y_inner_valid = inner_valid["y"].to_numpy(dtype=np.float32, copy=False)
        for model in missing_tuning:
            print(f"[{model}] {fold.name}: nested tuning", flush=True)
            trials, best = _tune_model(
                model, x_inner_train, y_inner_train, x_inner_valid, y_inner_valid,
                inner_valid["date"], config,
            )
            value = {
                "config_sha256": config_sha, "model": model, "fold": fold.name,
                "provenance": provenance,
                "inner_start": str(inner_start.date()),
                "n_inner_train_sampled": int(len(inner_train)),
                "n_inner_valid": int(len(inner_valid)), "trials": trials, "best": best,
            }
            _check_memory(config, f"{model} {fold.name} tuning before publish")
            _json_dump(_fold_dir(out_dir, model, fold, sealed) / "tuning.json", value)
            tuning[model] = value
            if sealed:
                # The inner split of a late fold sits inside the untouched
                # range, so its selection score is never surfaced.
                print(f"[{model}] {fold.name}: inner selection published", flush=True)
            else:
                print(
                    f"[{model}] {fold.name}: best inner RankIC={best['inner_rank_ic']:+.5f} "
                    f"params={best['params']}", flush=True,
                )
            _check_memory(config, f"{model} {fold.name} tuning")
        del inner_train, inner_valid, x_inner_train, x_inner_valid
        gc.collect()

    missing_scores: list[tuple[str, int]] = []
    for model in models:
        best_params = tuning[model]["best"]["params"]
        for seed in seeds:
            score_path = _fold_dir(out_dir, model, fold, sealed) / f"scores_seed{seed}.parquet"
            expected_score = {
                "score_schema_version": SCORE_SCHEMA_VERSION,
                "provenance": provenance,
                "model": model,
                "fold": fold.name,
                "seed": seed,
                "best_params": best_params,
            }
            if not _score_complete(score_path, expected_score):
                missing_scores.append((model, seed))
    if missing_scores:
        x_train = _build_matrix(outer_train, state, config)
        x_valid = _build_matrix(outer_valid, state, config)
        y_train = outer_train["y"].to_numpy(dtype=np.float32, copy=True)
        score_keys = outer_valid[["PERMNO", "date"]].rename(
            columns={"date": "signal_date"}
        ).copy()
        del outer_train, outer_valid
        gc.collect()
        for model in models:
            best = tuning[model]["best"]
            for seed in seeds:
                score_path = _fold_dir(out_dir, model, fold, sealed) / f"scores_seed{seed}.parquet"
                if score_path.exists():
                    continue
                print(f"[{model}] {fold.name}: final fit seed={seed} params={best['params']}", flush=True)
                prediction = _fit_predict(
                    model, best["params"], seed, x_train, y_train, x_valid, config
                )
                _check_memory(config, f"{model} {fold.name} seed {seed} before publish")
                scores = score_keys.copy()
                scores["score"] = prediction
                score_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = score_path.with_suffix(".parquet.tmp")
                scores.to_parquet(tmp, index=False, compression="zstd")
                tmp.replace(score_path)
                score_record = {
                    "score_schema_version": SCORE_SCHEMA_VERSION,
                    "provenance": provenance,
                    "model": model,
                    "fold": fold.name,
                    "seed": seed,
                    "best_params": best["params"],
                    "n_rows": int(len(scores)),
                    "signal_date_min": str(pd.Timestamp(scores["signal_date"].min()).date()),
                    "signal_date_max": str(pd.Timestamp(scores["signal_date"].max()).date()),
                    "score_sha256": _sha256(score_path),
                }
                _json_dump(_score_sidecar(score_path), score_record)
                del prediction, scores
                gc.collect()
                _check_memory(config, f"{model} {fold.name} seed {seed}")
        del x_train, x_valid, y_train, score_keys
        gc.collect()
    else:
        del outer_train, outer_valid
        gc.collect()

    if sealed:
        del state
        gc.collect()
        elapsed = time.perf_counter() - started
        for model in models:
            model_dir = _fold_dir(out_dir, model, fold, sealed)
            report = _publish_seal(
                model_dir, model, fold, seeds, config,
                config_sha, config_path, provenance, tuning[model]["best"]["params"], elapsed,
            )
            print(
                f"[{model}] {fold.name}: compute-only scores published "
                f"({report['dir']}); audit clean={report['clean']}",
                flush=True,
            )
            if append_ledger:
                _append_ledger_line(fold, model, model_dir, seeds)
        _check_memory(config, f"{fold.name} complete")
        print(f"[{fold.name}] elapsed {elapsed / 60.0:.1f} min", flush=True)
        return

    labels, label_sha = _load_labels(fold)
    for model in models:
        _fold_result(
            model, fold, _fold_dir(out_dir, model, fold, sealed), labels, label_sha,
            seeds, tuning[model]["best"], config_sha, provenance,
        )
    del state, labels
    gc.collect()
    _check_memory(config, f"{fold.name} complete")


def _decision(value: float) -> str:
    if value < 0.010:
        return "Kronos clearly wins"
    if value <= 0.020:
        return "Kronos leads, but not decisively"
    return "The tree model matches or beats Kronos"


def summarize(out_dir: Path, config: dict[str, Any], config_sha: str) -> dict[str, Any]:
    model_results: dict[str, Any] = {}
    seeds = [int(value) for value in config["seeds"]]
    provenance = _provenance(out_dir / "cache", config_sha)
    for model in config["models"]:
        fold_summaries: dict[str, Any] = {}
        ensemble_daily: list[pd.DataFrame] = []
        seed_daily: dict[int, list[pd.DataFrame]] = {seed: [] for seed in seeds}
        for fold in ACTIVE_FOLDS:
            model_dir = out_dir / model / fold.name
            summary_path = model_dir / "fold_summary.json"
            if not summary_path.exists():
                break
            fold_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if fold_summary.get("provenance") != provenance:
                raise RuntimeError(f"Stale fold summary provenance: {summary_path}")
            if fold_summary.get("official_label_sha256") != _sha256(_label_path(fold)):
                raise RuntimeError(f"Official-label fingerprint changed for {fold.name}")
            fold_summaries[fold.name] = fold_summary
            ensemble_path = model_dir / "daily_ic_ensemble.parquet"
            if _sha256(ensemble_path) != fold_summary["ensemble_daily_ic_sha256"]:
                raise RuntimeError(f"Ensemble daily-IC fingerprint mismatch: {ensemble_path}")
            ensemble_daily.append(pd.read_parquet(ensemble_path))
            for seed in seeds:
                score_path = model_dir / f"scores_seed{seed}.parquet"
                _score_complete(
                    score_path,
                    {
                        "score_schema_version": SCORE_SCHEMA_VERSION,
                        "provenance": provenance,
                        "model": model,
                        "fold": fold.name,
                        "seed": seed,
                        "best_params": fold_summary["best_inner_trial"]["params"],
                    },
                )
                daily_path = model_dir / f"daily_ic_seed{seed}.parquet"
                if _sha256(daily_path) != fold_summary["seed_daily_ic_sha256"][str(seed)]:
                    raise RuntimeError(f"Seed daily-IC fingerprint mismatch: {daily_path}")
                seed_daily[seed].append(pd.read_parquet(daily_path))
        if len(fold_summaries) != len(ACTIVE_FOLDS):
            continue
        all_daily = pd.concat(ensemble_daily, ignore_index=True).sort_values("signal_date")
        expected = int(config["evaluation"]["expected_daily_ic_count"])
        if len(all_daily) != expected or all_daily["signal_date"].nunique() != expected:
            raise AssertionError(
                f"{model}: expected {expected} daily ICs, got {len(all_daily)} rows / "
                f"{all_daily['signal_date'].nunique()} dates"
            )
        fold_means = {
            fold: float(value["ensemble"]["mean"]) for fold, value in fold_summaries.items()
        }
        seed_means = {
            str(seed): float(pd.concat(frames, ignore_index=True)["rank_ic"].mean())
            for seed, frames in seed_daily.items()
        }
        seed_values = np.asarray(list(seed_means.values()), dtype=np.float64)
        primary = _nw_stats(all_daily["rank_ic"])
        model_results[model] = {
            "primary_881_day": primary,
            "simple_seven_fold_mean": float(np.mean(list(fold_means.values()))),
            "positive_folds": int(sum(value > 0 for value in fold_means.values())),
            "fold_means": fold_means, "seed_881_day_means": seed_means,
            "seed_mean": float(seed_values.mean()),
            "seed_std": float(seed_values.std(ddof=1)), "decision": _decision(primary["mean"]),
        }
    result: dict[str, Any] = {
        "config_sha256": config_sha,
        "kronos_reference_rank_ic": config["evaluation"]["kronos_reference_rank_ic"],
        "models": model_results,
    }
    if len(model_results) == len(config["models"]):
        winner = max(model_results, key=lambda name: model_results[name]["primary_881_day"]["mean"])
        result.update({
            "best_tree_model": winner,
            "best_tree_rank_ic": model_results[winner]["primary_881_day"]["mean"],
            "overall_decision": _decision(model_results[winner]["primary_881_day"]["mean"]),
        })
    _json_dump(out_dir / "summary.json", result)
    return result


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _seal_cache(cache_dir: Path, config: dict[str, Any], config_sha: str) -> None:
    """The per-year feature cache carries the training target ``y``.

    The 2026-09-05 authorisation allows the target to be consumed as a training
    objective but forbids leaving it in readable form, so the cache directory
    gets its own sentinel and the read guard covers everything beneath it.
    """
    write_seal(
        cache_dir,
        {
            "artifact": "per-year stock feature cache plus the shared JKP state table",
            "why_sealed": (
                "stock_features_<year>.parquet carries the column y, the "
                "cross-sectionally demeaned six-session forward return used as the "
                "training target; it is an input to training only and must not be "
                "read, plotted or aggregated"
            ),
            "config_sha256": config_sha,
            "jkp_snapshot": config["features"]["jkp_state_snapshot"],
            "jkp_snapshot_sha256": config["features"]["jkp_state_snapshot_sha256"],
            "years": sorted(
                int(path.stem.rsplit("_", 1)[1])
                for path in (cache_dir / "base").glob("stock_features_*.parquet")
            ),
        },
    )


def main() -> None:
    global ACTIVE_FOLDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--models", default="lightgbm,xgboost,catboost")
    parser.add_argument("--folds", default=None)
    parser.add_argument(
        "--folds-json", type=Path, default=None,
        help="fold table produced by scripts/emit_folds.py; replaces the seven "
             "frozen development folds",
    )
    parser.add_argument(
        "--sealed", action="store_true",
        help="compute-only mode: train and score, load no label, compute no metric",
    )
    parser.add_argument(
        "--append-ledger", action="store_true",
        help="append one metric-free sealed-compute row per published fold",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.folds_json is not None:
        ACTIVE_FOLDS = _load_folds_json(args.folds_json)
    if args.sealed and args.folds_json is None:
        raise ValueError("--sealed needs --folds-json from scripts/emit_folds.py")
    if args.sealed and args.summarize_only:
        raise ValueError("--summarize-only computes metrics and is refused in compute-only mode")

    config, config_sha = _read_config(args.config)
    _check_runtime_versions(config)
    out_dir = args.out_dir.resolve()
    cache_dir = (args.cache_dir.resolve() if args.cache_dir else out_dir / "cache")
    _freeze_preregister(
        args.config, out_dir,
        "preregister_sealed.json" if args.sealed else "preregister.json",
    )
    if args.summarize_only:
        print(json.dumps(summarize(out_dir, config, config_sha), indent=2, ensure_ascii=False))
        return
    prepare_caches(cache_dir, config, config_sha)
    if args.sealed:
        _seal_cache(cache_dir, config, config_sha)
    if args.prepare_only:
        return
    if args.smoke:
        _check_packages(config["models"])
        _smoke(cache_dir, config)
        return

    models = _parse_csv(args.models)
    unknown_models = sorted(set(models) - set(config["models"]))
    if unknown_models:
        raise ValueError(f"Models not in frozen config: {unknown_models}")
    fold_names = set(
        _parse_csv(args.folds) if args.folds
        else [fold.name for fold in ACTIVE_FOLDS]
    )
    selected_folds = [fold for fold in ACTIVE_FOLDS if fold.name in fold_names]
    if len(selected_folds) != len(fold_names):
        known = {fold.name for fold in ACTIVE_FOLDS}
        raise ValueError(f"Unknown folds: {sorted(fold_names - known)}")
    _check_packages(models)
    seeds_from_config = [int(value) for value in config["seeds"]]
    for fold in selected_folds:
        if args.sealed:
            done = all(
                (_fold_dir(out_dir, model, fold, True) / "SEALED_MANIFEST.json").exists()
                for model in models
            )
            if done:
                print(f"[{fold.name}] already published, skipping", flush=True)
                if args.append_ledger:
                    for model in models:
                        _append_ledger_line(
                            fold, model, _fold_dir(out_dir, model, fold, True), seeds_from_config
                        )
                continue
            run_fold(
                fold, models, out_dir, cache_dir, config, config_sha,
                sealed=True, config_path=args.config, append_ledger=args.append_ledger,
            )
            continue
        run_fold(fold, models, out_dir, cache_dir, config, config_sha)
        summarize(out_dir, config, config_sha)
    if args.sealed:
        print("[done] compute-only run finished; reading these scores needs a separate grant", flush=True)
        return
    print(json.dumps(summarize(out_dir, config, config_sha), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
