"""实验 9（E1 改版）：无标签 test–retest 信度。

诊断 + 估计交付。不据此改封存口径（封存队列 sc=5 已完成，不追溯）；
不选 sc——sc=20 已由 CLAUDE.md §二「零参数且理论单调」例外条款裁定，本实验不重开该决定。
预期方向：r(sc) 随 sc 单调不降。若观察到非单调，只报，不解释成信号性质
（更可能是打分批次效应）。
IC 曲线只报、不作任何结论。
不得把两次打分的均值当成一次「sc 翻倍」的打分去和别的 sc 档比 IC
（会与 sc 档混淆）。

固定口径为 FT / lb90 / predict=6 / amp=bf16 / batch_size=128，折号只含已消耗的
36、39、42。逻辑 r1 复用：sc=5 读取既有 ``eval_amp_lb90_foldXX``；sc=10/20/40
读取实验 3 的 ``eval_e1_scXX``。逻辑 r2 才读取新建的 ``eval_e9_scXX_r2``。
辅助 IC 对 r1、r2 分开计算与报告，从不平均两次分数。
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402
from signals.kronos_adapter import labels_path, scores_path  # noqa: E402

FOLDS = (36, 39, 42)
SAMPLE_COUNTS = (5, 10, 20, 40)
REPEATS = (1, 2)
VAL_WINDOWS = {
    36: ("2020-07-01", "2020-12-31"),
    39: ("2022-01-03", "2022-06-30"),
    42: ("2023-07-03", "2023-12-29"),
}
NW_LAG = 5
TOPN = 500
MIN_NAMES = 50
BOOT_BLOCK = 10
BOOT_DRAWS = 10_000
BOOT_SEED = 20260904
OUT_JSON = REPO / "outputs" / "exp9_test_retest_reliability.json"
OUT_SVG = REPO / "outputs" / "exp9_test_retest_reliability.svg"
QUEUE_DONE = REPO / "outputs" / "exp9_queue_logs" / "QUEUE.DONE"


def _load_compare_arms_money():
    spec = importlib.util.spec_from_file_location(
        "compare_arms_money_exp9", REPO / "scripts" / "compare_arms_money.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def score_file(fold: int, sample_count: int, repeat: int) -> Path:
    """Resolve the logical 24-cell matrix, reusing 12 bit-identical r1 cells."""
    if fold not in FOLDS or sample_count not in SAMPLE_COUNTS or repeat not in REPEATS:
        raise ValueError(f"invalid experiment-9 cell: fold={fold}, sc={sample_count}, r={repeat}")
    if repeat == 1 and sample_count == 5:
        return scores_path(fold, "ft")
    if repeat == 1:
        return scores_path(fold, "ft", experiment_tag=f"e1_sc{sample_count}")
    return scores_path(fold, "ft", experiment_tag=f"e9_sc{sample_count}_r2")


def reuse_map() -> list[dict]:
    rows = []
    for sample_count in SAMPLE_COUNTS:
        for fold in FOLDS:
            source = "existing eval_amp" if sample_count == 5 else "experiment 3 e1"
            rows.append({
                "logical_cell": f"fold{fold}_sc{sample_count}_r1",
                "source": source,
                "path": str(score_file(fold, sample_count, 1)),
            })
    return rows


def new_gpu_cells() -> list[dict]:
    return [
        {
            "logical_cell": f"fold{fold}_sc{sample_count}_r2",
            "tag": f"e9_sc{sample_count}_r2",
            "path": str(score_file(fold, sample_count, 2)),
        }
        for sample_count in SAMPLE_COUNTS for fold in FOLDS
    ]


def committed_memory_gb() -> float | None:
    if sys.platform != "win32":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.ullTotalPageFile - status.ullAvailPageFile) / 2**30


def stationary_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    restart = 1.0 / BOOT_BLOCK
    out = np.empty(n, dtype=np.int64)
    current = int(rng.integers(n))
    for i in range(n):
        out[i] = current
        current = int(rng.integers(n)) if rng.random() < restart else (current + 1) % n
    return out


def stationary_mean_ci(values: pd.Series) -> list[float]:
    data = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(data) < 2:
        return [math.nan, math.nan]
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty(BOOT_DRAWS, dtype=float)
    for draw in range(BOOT_DRAWS):
        means[draw] = data[stationary_indices(len(data), rng)].mean()
    return [float(x) for x in np.percentile(means, (2.5, 97.5))]


def spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return math.nan
    a, b = left[valid].rank(), right[valid].rank()
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(a.corr(b))


def _check_score_keys(frame: pd.DataFrame, name: str) -> None:
    required = {"PERMNO", "signal_date", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name}: missing columns {sorted(required - set(frame.columns))}")
    if frame.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError(f"{name}: duplicate (PERMNO, signal_date) keys")


def reliability_daily(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Daily Spearman on the exact date/name intersection; no labels are touched."""
    _check_score_keys(left, "left")
    _check_score_keys(right, "right")
    l = left[["PERMNO", "signal_date", "score"]].rename(columns={"score": "left"})
    r = right[["PERMNO", "signal_date", "score"]].rename(columns={"score": "right"})
    merged = l.merge(r, on=["PERMNO", "signal_date"], how="inner").dropna()
    daily = merged.groupby("signal_date", sort=True).apply(
        lambda group: spearman(group["left"], group["right"]), include_groups=False
    )
    left_keys = pd.MultiIndex.from_frame(l[["PERMNO", "signal_date"]].drop_duplicates())
    right_keys = pd.MultiIndex.from_frame(r[["PERMNO", "signal_date"]].drop_duplicates())
    coverage = {
        "left_rows": len(l), "right_rows": len(r), "intersection_nonmissing_rows": len(merged),
        "left_only_keys": int(len(left_keys.difference(right_keys))),
        "right_only_keys": int(len(right_keys.difference(left_keys))),
        "n_days": int(daily.notna().sum()),
    }
    return daily, coverage


def _one_day_spread(group: pd.DataFrame) -> float:
    clean = group.dropna(subset=["score", "label"])
    if len(clean) < 10:
        return math.nan
    bins = pd.qcut(clean["score"].rank(method="first"), 10, labels=False)
    return float(clean.loc[bins == 9, "label"].mean() - clean.loc[bins == 0, "label"].mean())


def _top500(frame: pd.DataFrame, adv: dict) -> pd.DataFrame:
    parts = []
    for day, group in frame.groupby("signal_date", sort=True):
        values = adv.get(day, {})
        valid = group[group["PERMNO"].map(lambda p: p in values and np.isfinite(values[p]))].copy()
        if len(valid) > TOPN:
            valid["_adv20"] = valid["PERMNO"].map(values)
            valid = valid.sort_values("_adv20", ascending=False, kind="mergesort").head(TOPN)
        if len(valid) >= MIN_NAMES:
            parts.append(valid.drop(columns="_adv20", errors="ignore"))
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0].copy()


def daily_metrics(scores: pd.DataFrame, outcomes: pd.DataFrame, adv: dict) -> dict[str, pd.Series]:
    """Auxiliary outcomes only; each repeat is evaluated separately, never score-averaged."""
    _check_score_keys(scores, "scores")
    merged = scores[["PERMNO", "signal_date", "score"]].dropna().merge(
        outcomes[["PERMNO", "signal_date", "label", "status"]],
        on=["PERMNO", "signal_date"], how="inner",
    )
    merged = merged[(merged["status"] == "ok") & merged["label"].notna()]
    top = _top500(merged, adv)

    def rank_ic(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("signal_date", sort=True).apply(
            lambda group: spearman(group["score"], group["label"]), include_groups=False
        )

    return {
        "ic_full": rank_ic(merged),
        "ic_top500": rank_ic(top),
        "decile_spread_full": merged.groupby("signal_date", sort=True).apply(
            _one_day_spread, include_groups=False
        ),
        "decile_spread_top500": top.groupby("signal_date", sort=True).apply(
            _one_day_spread, include_groups=False
        ),
    }


def series_summary(series: pd.Series, *, scale: float = 1.0, unit: str = "correlation") -> dict:
    values = pd.Series(series).dropna().sort_index()
    nw = newey_west_tstat(values, NW_LAG)
    ci = stationary_mean_ci(values)
    return {
        "mean": float(nw["mean"] * scale), "nw5_se": float(nw["se"] * scale),
        "nw5_t": float(nw["t"]), "stationary_bootstrap_ci95": [x * scale for x in ci],
        "n_days": int(nw["n"]), "unit": unit,
    }


def pooled_with_folds(parts: dict[int, pd.Series], *, scale: float = 1.0,
                      unit: str = "correlation") -> dict:
    return {
        "pooled": series_summary(pd.concat([parts[f] for f in sorted(parts)]), scale=scale, unit=unit),
        "by_fold": {str(f): series_summary(parts[f], scale=scale, unit=unit) for f in sorted(parts)},
    }


def spearman_brown(measured: dict[int, float]) -> dict:
    base = float(measured[5])
    if not 0 < base < 1:
        raise ValueError(f"sc=5 reliability must be in (0,1), got {base}")
    ratio = base / (1.0 - base)
    k95 = 0.95 * (1.0 - base) / (0.05 * base)
    rows = {}
    for sample_count in SAMPLE_COUNTS:
        multiple = sample_count / 5.0
        predicted = multiple * base / (1.0 + (multiple - 1.0) * base)
        rows[str(sample_count)] = {
            "sample_count_multiple_vs_sc5": multiple,
            "predicted_reliability": predicted,
            "measured_reliability": float(measured[sample_count]),
            "measured_minus_predicted": float(measured[sample_count] - predicted),
        }
    return {
        "single_score_reliability_sc5": base,
        "signal_variance_over_sampling_noise_variance": ratio,
        "sample_count_multiple_needed_for_reliability_0.95": k95,
        "prediction_vs_measurement": rows,
        "all_higher_sc_below_prediction": all(
            rows[str(sc)]["measured_minus_predicted"] < 0 for sc in SAMPLE_COUNTS[1:]
        ),
        "interpretation_limit": "shortfall is reported as possible fixed batch noise; no signal interpretation",
    }


def validate_metrics(path: Path, fold: int, sample_count: int) -> None:
    metrics_path = path.parent / "metrics.json"
    assert_readable(metrics_path)
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metadata.get("scoring_config", {})
    expected = {"amp": "bf16", "batch_size": 128, "sample_count": sample_count,
                "lookback": 90, "predict": 6}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"configuration mismatch: {metrics_path}: {config} != {expected}")
    if metadata.get("val_window") != list(VAL_WINDOWS[fold]):
        raise RuntimeError(f"validation-window mismatch: {metrics_path}")


def load_scores(fold: int, sample_count: int, repeat: int) -> pd.DataFrame:
    path = score_file(fold, sample_count, repeat)
    assert_readable(path)
    validate_metrics(path, fold, sample_count)
    frame = pd.read_parquet(path, columns=["PERMNO", "signal_date", "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    _check_score_keys(frame, str(path))
    return frame


def _line_points(values: list[tuple[float, float]], x0: float, y0: float,
                 width: float, height: float, ymin: float, ymax: float) -> str:
    xmin, xmax = min(x for x, _ in values), max(x for x, _ in values)
    coords = []
    for x, y in values:
        px = x0 + width * (x - xmin) / (xmax - xmin)
        py = y0 + height * (1.0 - (y - ymin) / (ymax - ymin))
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def write_svg(report: dict, path: Path) -> None:
    reliability = [
        (float(sc), report["reliability"][str(sc)]["pooled"]["mean"])
        for sc in SAMPLE_COUNTS
    ]
    prediction = [
        (float(sc), report["spearman_brown"]["prediction_vs_measurement"][str(sc)]
         ["predicted_reliability"])
        for sc in SAMPLE_COUNTS
    ]
    auxiliary = {
        repeat: [
            (float(sc), report["auxiliary_metrics"][str(sc)][repeat]["ic_full"]["pooled"]["mean"])
            for sc in SAMPLE_COUNTS
        ] for repeat in ("r1", "r2")
    }
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="690" viewBox="0 0 1000 690">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}.axis{stroke:#6b7280}.line{fill:none;stroke-width:3}.dot{stroke:white;stroke-width:1.5}</style>',
        '<text x="45" y="38" font-size="24" font-weight="600">Experiment 9: test-retest reliability</text>',
        '<text x="45" y="62" font-size="13">FT · folds 36/39/42 · diagnostic + estimation only</text>',
    ]
    panels = [
        ("Daily score-rank reliability", {"measured": reliability, "Spearman-Brown": prediction},
         (0.0, 1.0), ("#2563eb", "#9ca3af")),
        ("Auxiliary full-pool RankIC (repeats kept separate)", auxiliary, None,
         ("#059669", "#dc2626")),
    ]
    for panel_index, (title, series, fixed_range, colors) in enumerate(panels):
        x0, y0, width, height = 85.0, 115.0 + 275.0 * panel_index, 820.0, 190.0
        chunks.append(f'<text x="{x0}" y="{y0 - 18}" font-size="16" font-weight="600">{escape(title)}</text>')
        chunks.append(f'<line class="axis" x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}"/>')
        chunks.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}"/>')
        all_y = [y for values in series.values() for _, y in values]
        if fixed_range:
            ymin, ymax = fixed_range
        else:
            ymin, ymax = min(all_y), max(all_y)
            pad = max((ymax - ymin) * 0.1, 0.001)
            ymin, ymax = ymin - pad, ymax + pad
        for index, (name, values) in enumerate(series.items()):
            points = _line_points(values, x0, y0, width, height, ymin, ymax)
            dash = ' stroke-dasharray="8 6"' if name == "Spearman-Brown" else ""
            chunks.append(f'<polyline class="line" stroke="{colors[index]}"{dash} points="{points}"/>')
            for (sc, value), coordinate in zip(values, points.split()):
                px, py = coordinate.split(",")
                chunks.append(f'<circle class="dot" fill="{colors[index]}" cx="{px}" cy="{py}" r="5"/>')
                chunks.append(f'<text x="{float(px)+6:.2f}" y="{float(py)-7:.2f}" font-size="11">{value:.4f}</text>')
            chunks.append(f'<text x="{x0+650}" y="{y0+20+index*20}" font-size="12">{escape(name)}</text>')
        for sc in SAMPLE_COUNTS:
            px = x0 + width * (sc - 5) / 35
            chunks.append(f'<text x="{px-10:.2f}" y="{y0+height+22}" font-size="11">{sc}</text>')
    chunks.extend([
        '<text x="85" y="655" font-size="12">IC panels are auxiliary only; r1/r2 scores are never averaged.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(chunks), encoding="utf-8")


def analyse(score_frames: dict[tuple[int, int, int], pd.DataFrame],
            labels: dict[int, pd.DataFrame], adv: dict) -> dict:
    reliability_parts: dict[int, dict[int, pd.Series]] = {}
    coverage = {}
    metric_parts: dict[tuple[int, int, int], dict[str, pd.Series]] = {}
    for sample_count in SAMPLE_COUNTS:
        reliability_parts[sample_count] = {}
        coverage[str(sample_count)] = {}
        for fold in FOLDS:
            r, cell_coverage = reliability_daily(
                score_frames[(fold, sample_count, 1)], score_frames[(fold, sample_count, 2)]
            )
            reliability_parts[sample_count][fold] = r
            coverage[str(sample_count)][str(fold)] = cell_coverage
            for repeat in REPEATS:
                metric_parts[(fold, sample_count, repeat)] = daily_metrics(
                    score_frames[(fold, sample_count, repeat)], labels[fold], adv
                )

    reliability = {
        str(sc): pooled_with_folds(reliability_parts[sc]) for sc in SAMPLE_COUNTS
    }
    measured = {sc: reliability[str(sc)]["pooled"]["mean"] for sc in SAMPLE_COUNTS}
    sb = spearman_brown(measured)
    reliability_differences = {
        f"sc{left}_to_sc{right}": measured[right] - measured[left]
        for left, right in zip(SAMPLE_COUNTS[:-1], SAMPLE_COUNTS[1:])
    }
    auxiliary = {}
    metric_names = ("ic_full", "ic_top500", "decile_spread_full", "decile_spread_top500")
    for sample_count in SAMPLE_COUNTS:
        auxiliary[str(sample_count)] = {}
        for repeat in REPEATS:
            auxiliary[str(sample_count)][f"r{repeat}"] = {}
            for metric in metric_names:
                parts = {f: metric_parts[(f, sample_count, repeat)][metric] for f in FOLDS}
                is_spread = metric.startswith("decile_spread")
                auxiliary[str(sample_count)][f"r{repeat}"][metric] = pooled_with_folds(
                    parts, scale=1e4 if is_spread else 1.0,
                    unit="bp/day" if is_spread else "correlation",
                )

    se_comparison = {}
    for sample_count in SAMPLE_COUNTS:
        se_comparison[str(sample_count)] = {}
        for fold in FOLDS:
            r_se = reliability[str(sample_count)]["by_fold"][str(fold)]["nw5_se"]
            row = {"reliability_nw5_se": r_se}
            for pool in ("full", "top500"):
                for repeat in REPEATS:
                    ic_se = auxiliary[str(sample_count)][f"r{repeat}"][f"ic_{pool}"]["by_fold"][str(fold)]["nw5_se"]
                    row[f"ic_r{repeat}_{pool}_nw5_se"] = ic_se
                    row[f"ic_r{repeat}_over_reliability_se_ratio_{pool}"] = (
                        ic_se / r_se if r_se > 0 else math.nan
                    )
            se_comparison[str(sample_count)][str(fold)] = row

    queue = None
    if QUEUE_DONE.exists():
        queue = json.loads(QUEUE_DONE.read_text(encoding="utf-8-sig"))
    return {
        "meta": {
            "experiment": 9, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "configuration": {"arm": "ft", "folds": list(FOLDS), "lookback": 90,
                              "predict": 6, "amp": "bf16", "batch_size": 128,
                              "sample_counts": list(SAMPLE_COUNTS)},
            "bootstrap": {"kind": "circular stationary", "mean_block": BOOT_BLOCK,
                          "draws": BOOT_DRAWS, "seed": BOOT_SEED},
            "logical_cells": 24, "new_gpu_calls": 12,
            "reuse": reuse_map(), "new_cells": new_gpu_cells(),
            "restriction": "diagnostic and estimation only; IC is auxiliary; repeat scores never averaged",
            "gpu_queue": queue,
        },
        "reliability": reliability,
        "reliability_monotonicity": {
            "successive_differences": reliability_differences,
            "nondecreasing": all(value >= 0 for value in reliability_differences.values()),
            "restriction": "nonmonotonicity is reported only; do not interpret as a signal property",
        },
        "prior_sc5_crosscheck": {
            "prior_rank_reliability": 0.75,
            "measured": measured[5],
            "difference": measured[5] - 0.75,
            "restriction": "descriptive crosscheck only; no post-hoc discrepancy threshold",
        },
        "same_name_set_coverage": coverage,
        "spearman_brown": sb,
        "reliability_vs_ic_se": se_comparison,
        "auxiliary_metrics": auxiliary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-svg", type=Path, default=OUT_SVG)
    parser.add_argument("--max-committed-gb", type=float, default=40.0)
    args = parser.parse_args()
    memory = committed_memory_gb()
    if memory is not None and memory > args.max_committed_gb:
        print(f"REFUSE analysis: committed memory {memory:.2f}GB exceeds {args.max_committed_gb:.2f}GB",
              file=sys.stderr)
        return 75

    cam = _load_compare_arms_money()
    assert_readable(cam.P / "panel_raw.parquet")
    _, _, adv = cam.load_prices()
    labels = {}
    scores = {}
    for fold in FOLDS:
        outcome_path = labels_path(fold, "ft")
        assert_readable(outcome_path)
        frame = pd.read_parquet(
            outcome_path, columns=["PERMNO", "signal_date", "label", "status"]
        )
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        labels[fold] = frame
        for sample_count in SAMPLE_COUNTS:
            for repeat in REPEATS:
                scores[(fold, sample_count, repeat)] = load_scores(fold, sample_count, repeat)
                print(f"loaded fold{fold} sc={sample_count} r{repeat}", flush=True)

    report = analyse(scores, labels, adv)
    report["meta"]["committed_memory_gb_before_analysis"] = memory
    report["meta"]["committed_memory_limit_gb"] = args.max_committed_gb
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(report, args.out_svg)
    print("=== pooled test-retest reliability ===")
    for sample_count in SAMPLE_COUNTS:
        item = report["reliability"][str(sample_count)]["pooled"]
        sb = report["spearman_brown"]["prediction_vs_measurement"][str(sample_count)]
        print(f"sc={sample_count:2d} r={item['mean']:.6f} SE={item['nw5_se']:.6f} "
              f"CI=[{item['stationary_bootstrap_ci95'][0]:.6f},"
              f"{item['stationary_bootstrap_ci95'][1]:.6f}] SB={sb['predicted_reliability']:.6f}")
    print(f"wrote {args.out_json}\nwrote {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
