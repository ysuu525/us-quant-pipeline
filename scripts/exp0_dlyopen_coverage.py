"""实验 0：CRSP DlyOpen 缺失率与口径核查。

判据：诊断，无阈值。用途限制：只报数，不改任何口径。

只读取原始 CRSP 面板与已消耗开发折 36--42 的既有 FT 标签。原始面板按
两年一块做日期下推和列裁剪；块间只携带每只证券最后 20 条观测，以保持
lagged ADV20 与前收口径连续，不保存历史块的其余行。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402
from signals.kronos_adapter import labels_path  # noqa: E402

PANEL = Path(
    r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z\panel_raw.parquet"
)
START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2025-12-31")
FOLDS = (36, 37, 38, 39, 40, 41, 42)
COLUMNS = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyPrcVol"]
METRIC_COUNTS = (
    "n",
    "open_missing",
    "open_negative",
    "open_zero",
    "open_nonpositive",
    "open_eq_abs_prev_close",
    "open_jump_gt_20pct",
)


def two_year_blocks(start: pd.Timestamp, end: pd.Timestamp):
    block_start = start
    while block_start <= end:
        block_end = min(
            pd.Timestamp(year=block_start.year + 1, month=12, day=31), end
        )
        yield block_start, block_end
        block_start = block_end + pd.Timedelta(days=1)


def add_counts(store: dict, frame: pd.DataFrame) -> None:
    """把一个股票日集合的逐年计数累加进 ``store``。"""
    if frame.empty:
        return
    values = frame.assign(
        year=frame["DlyCalDt"].dt.year,
        n=1,
        open_missing=frame["DlyOpen"].isna().astype("int64"),
        open_negative=frame["DlyOpen"].lt(0).astype("int64"),
        open_zero=frame["DlyOpen"].eq(0).astype("int64"),
        open_nonpositive=frame["DlyOpen"].le(0).astype("int64"),
        open_eq_abs_prev_close=(
            frame["DlyOpen"].abs().eq(frame["prev_close"].abs())
            & frame["DlyOpen"].notna()
            & frame["prev_close"].notna()
        ).astype("int64"),
        open_jump_gt_20pct=(
            frame["DlyOpen"].abs().div(frame["prev_close"].abs()).sub(1).abs().gt(0.20)
            & frame["DlyOpen"].notna()
            & frame["prev_close"].abs().gt(0)
        ).astype("int64"),
    )
    counts = values.groupby("year", sort=True)[list(METRIC_COUNTS)].sum()
    for year, row in counts.iterrows():
        for metric in METRIC_COUNTS:
            store[int(year)][metric] += int(row[metric])


def finalize_counts(store: dict) -> list[dict]:
    rows = []
    for year in sorted(store):
        counts = store[year]
        n = counts["n"]
        row = {"year": year, "N": n}
        for metric in METRIC_COUNTS[1:]:
            row[f"{metric}_n"] = counts[metric]
            row[f"{metric}_pct"] = 100.0 * counts[metric] / n if n else None
        rows.append(row)
    return rows


def analyze_panel(panel_path: Path) -> tuple[list[dict], list[dict]]:
    assert_readable(panel_path)
    all_counts = defaultdict(lambda: defaultdict(int))
    top_counts = defaultdict(lambda: defaultdict(int))
    carry = pd.DataFrame(columns=COLUMNS)

    for block_start, block_end in two_year_blocks(START, END):
        current = pd.read_parquet(
            panel_path,
            columns=COLUMNS,
            filters=[
                ("DlyCalDt", ">=", block_start),
                ("DlyCalDt", "<=", block_end),
            ],
        )
        current["DlyCalDt"] = pd.to_datetime(current["DlyCalDt"])
        current["_current"] = True
        if carry.empty:
            work = current
        else:
            prior = carry.copy()
            prior["_current"] = False
            work = pd.concat([prior, current], ignore_index=True)
        work = work.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort")
        grouped = work.groupby("PERMNO", sort=False)
        work["prev_close"] = grouped["DlyClose"].shift(1)
        work["adv20"] = grouped["DlyPrcVol"].transform(
            lambda s: s.rolling(20, min_periods=10).mean().shift(1)
        )
        block = work.loc[work["_current"]].copy()

        add_counts(all_counts, block)
        eligible = block[np.isfinite(block["adv20"])].sort_values(
            ["DlyCalDt", "adv20"], ascending=[True, False], kind="mergesort"
        )
        top500 = eligible.groupby("DlyCalDt", sort=False).head(500)
        add_counts(top_counts, top500)

        carry = (
            work.groupby("PERMNO", sort=False, group_keys=False)
            .tail(20)[COLUMNS]
            .copy()
        )
        print(
            f"block {block_start.date()}..{block_end.date()}: "
            f"all={len(block):,} top500={len(top500):,}",
            flush=True,
        )

    return finalize_counts(top_counts), finalize_counts(all_counts)


def label_status_readout() -> list[dict]:
    rows = []
    for fold in FOLDS:
        path = labels_path(fold, "ft")
        assert_readable(path)
        status = pd.read_parquet(path, columns=["status"])["status"]
        counts = status.value_counts(dropna=False)
        n = int(len(status))
        rows.append(
            {
                "fold": fold,
                "N": n,
                "ok_n": int(counts.get("ok", 0)),
                "ok_pct": 100.0 * int(counts.get("ok", 0)) / n if n else None,
                "unfillable_n": int(counts.get("unfillable", 0)),
                "unfillable_pct": (
                    100.0 * int(counts.get("unfillable", 0)) / n if n else None
                ),
                "invalid_n": int(counts.get("invalid", 0)),
                "invalid_pct": (
                    100.0 * int(counts.get("invalid", 0)) / n if n else None
                ),
                "other_n": int(
                    n
                    - counts.get("ok", 0)
                    - counts.get("unfillable", 0)
                    - counts.get("invalid", 0)
                ),
            }
        )
    return rows


def print_annual(title: str, rows: list[dict]) -> None:
    columns = [
        "year",
        "N",
        "open_missing_pct",
        "open_negative_pct",
        "open_zero_pct",
        "open_nonpositive_pct",
        "open_eq_abs_prev_close_pct",
        "open_jump_gt_20pct_pct",
    ]
    table = pd.DataFrame(rows)[columns]
    print(f"\n=== {title} ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel", type=Path, default=PANEL, help="原始 panel_raw.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "exp0_dlyopen_coverage.json"
    )
    args = parser.parse_args()

    top500, full_pool = analyze_panel(args.panel)
    status = label_status_readout()
    result = {
        "meta": {
            "experiment": 0,
            "date_range": [str(START.date()), str(END.date())],
            "topn": 500,
            "adv20": "rolling(20,min_periods=10).mean().shift(1)",
            "annual_rate_denominator": "all stock-days in the reported group",
        },
        "top500_annual": top500,
        "full_pool_annual": full_pool,
        "fold_status": status,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_annual("top500 annual", top500)
    print_annual("full pool annual", full_pool)
    print("\n=== fold 36--42 FT label status ===")
    print(pd.DataFrame(status).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    flagged = [
        row["year"]
        for row in top500
        if row["open_missing_pct"] > 2.0 or row["open_nonpositive_pct"] > 2.0
    ]
    print(f"\nFLAG top500 missing or nonpositive > 2%: {flagged}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
