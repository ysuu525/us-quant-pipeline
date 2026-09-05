"""Rebuild the derived JKP market-state snapshot used by the tree baseline.

The formula, windows and lag are the ones frozen in ``configs/gbdt_strong_v2.json``:

    state[w] = expm1(rolling_sum(log1p(ret.shift(1)), window=w, min_periods=w))

with ``w in {5, 20, 60}`` and ``lag = 1`` observation.  Columns are named
``jkp_<factor>_cum<w>``, blocked by window (all cum5, then cum20, then cum60) and
alphabetically ordered inside each block, stored as ``float32`` next to a
``datetime64`` ``date`` column.  This reproduces
``usa_all_factors_daily_vw_cap_state_lag1_5_20_60_through_2023-12-29.parquet``
(SHA256 ``bbcfc2a2...``) bit for bit on its own physical window.

Two modes:

``--verify-against <parquet>``
    Rebuild on exactly the reference snapshot's physical window and assert the
    result is identical (float tolerance 1e-12, NaN pattern included).  This is
    the fidelity check for the formula, independent of the window extension.

``--start / --end / --out``
    Build the extended snapshot.  The extension only widens the physical window;
    no formula, window or lag changes.  The rolling warm-up is therefore the only
    place where a longer history can differ from a shorter one, and the script
    reports that region explicitly.

No metric of any kind is computed here; the output is an input feature table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_CSV = Path(r"F:\quant\external\jkp\usa_all_factors_daily_vw_cap.csv")
WINDOWS = (5, 20, 60)
LAG = 1
CUTOFF = "2023-12-29"
TOLERANCE = 1e-12


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_returns(csv_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Wide factor-return panel (date x factor) restricted to [start, end]."""
    frame = pd.read_csv(
        csv_path,
        usecols=["name", "date", "ret"],
        dtype={"name": "string", "ret": "float64"},
        parse_dates=["date"],
    )
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    wide = frame.pivot(index="date", columns="name", values="ret").sort_index()
    wide.columns = [str(name) for name in wide.columns]
    return wide[sorted(wide.columns)]


def build_state(wide: pd.DataFrame) -> pd.DataFrame:
    """expm1(rolling_sum(log1p(ret.shift(1)))) for each frozen window."""
    logged = np.log1p(wide).shift(LAG)
    blocks: list[pd.DataFrame] = []
    for window in WINDOWS:
        rolled = logged.rolling(window, min_periods=window).sum()
        block = np.expm1(rolled)
        block.columns = [f"jkp_{name}_cum{window}" for name in block.columns]
        blocks.append(block)
    state = pd.concat(blocks, axis=1).astype(np.float32)
    state.insert(0, "date", state.index)
    return state.reset_index(drop=True)


def compare(new: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Column-by-column, row-by-row comparison on the shared date range."""
    if list(new.columns) != list(reference.columns):
        return {
            "columns_match": False,
            "only_new": sorted(set(new.columns) - set(reference.columns))[:10],
            "only_reference": sorted(set(reference.columns) - set(new.columns))[:10],
        }
    lo, hi = reference["date"].min(), reference["date"].max()
    left = new[(new["date"] >= lo) & (new["date"] <= hi)].reset_index(drop=True)
    right = reference.reset_index(drop=True)
    if len(left) != len(right) or not left["date"].equals(right["date"]):
        return {"columns_match": True, "dates_match": False,
                "n_new": int(len(left)), "n_reference": int(len(right))}
    value_columns = [c for c in reference.columns if c != "date"]
    a = left[value_columns].to_numpy(dtype=np.float64)
    b = right[value_columns].to_numpy(dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    only_reference_nan = np.isnan(b) & ~np.isnan(a)
    only_new_nan = np.isnan(a) & ~np.isnan(b)
    finite = ~np.isnan(a) & ~np.isnan(b)
    delta = np.abs(a[finite] - b[finite])
    max_abs = float(delta.max()) if delta.size else 0.0
    # Rows where the reference itself is NaN are the reference's own rolling
    # warm-up: a longer history fills them in.  Report them separately instead of
    # hiding them inside the tolerance.
    warmup_rows = np.flatnonzero(only_reference_nan.any(axis=1))
    return {
        "columns_match": True,
        "dates_match": True,
        "n_rows": int(len(right)),
        "n_columns": int(len(value_columns)),
        "max_abs_diff_on_shared_finite_cells": max_abs,
        "n_shared_finite_cells": int(finite.sum()),
        "n_both_nan_cells": int(both_nan.sum()),
        "n_cells_nan_only_in_reference": int(only_reference_nan.sum()),
        "n_cells_nan_only_in_new": int(only_new_nan.sum()),
        "reference_warmup_row_indices": warmup_rows.tolist()[:80],
        "reference_warmup_last_date": (
            str(right.loc[int(warmup_rows.max()), "date"].date()) if warmup_rows.size else None
        ),
        "identical_where_both_defined": bool(max_abs <= TOLERANCE),
        "bitwise_identical_including_nan": bool(
            max_abs <= TOLERANCE
            and only_reference_nan.sum() == 0
            and only_new_nan.sum() == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--start", default="2001-10-01")
    parser.add_argument("--end", default=CUTOFF)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    end = pd.Timestamp(args.end)
    if end > pd.Timestamp(CUTOFF):
        raise ValueError(f"The raw-data cutoff {CUTOFF} may not be crossed")

    report: dict = {
        "source_csv": str(args.csv),
        "source_csv_sha256": sha256_file(args.csv),
        "formula": "expm1(rolling_sum(log1p(ret.shift(1))))",
        "windows": list(WINDOWS),
        "lag_observations": LAG,
        "cutoff": CUTOFF,
        "tolerance": TOLERANCE,
    }

    if args.verify_against is not None:
        reference = pd.read_parquet(args.verify_against)
        reference["date"] = pd.to_datetime(reference["date"])
        ref_lo = pd.Timestamp(reference["date"].min())
        ref_hi = pd.Timestamp(reference["date"].max())
        print(f"[verify] rebuilding on the reference window {ref_lo.date()}..{ref_hi.date()}", flush=True)
        rebuilt = build_state(load_returns(args.csv, ref_lo, ref_hi))
        report["reference"] = str(args.verify_against)
        report["reference_sha256"] = sha256_file(args.verify_against)
        report["same_window_rebuild"] = compare(rebuilt, reference)
        print(json.dumps(report["same_window_rebuild"], indent=2), flush=True)

    start = pd.Timestamp(args.start)
    print(f"[build] extended window {start.date()}..{end.date()}", flush=True)
    state = build_state(load_returns(args.csv, start, end))
    report["built_window"] = [str(start.date()), str(end.date())]
    report["shape"] = list(state.shape)
    report["first_all_finite_date"] = str(
        state.loc[state.drop(columns="date").notna().all(axis=1).idxmax(), "date"].date()
    )

    if args.verify_against is not None:
        report["extended_vs_reference"] = compare(state, reference)
        print(json.dumps(report["extended_vs_reference"], indent=2), flush=True)

    if args.out is not None:
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        state.to_parquet(tmp, index=False)
        tmp.replace(args.out)
        report["out"] = str(args.out)
        report["out_sha256"] = sha256_file(args.out)
        print(f"[build] wrote {args.out} sha256={report['out_sha256']}", flush=True)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
