"""实验 3（E1）：Kronos 采样数曲线。

判据：诊断；预期 IC 随 sc 单调不降、信度单调上升。用途限制：不据此改封存口径；不选 sc（sc=20 已由理论条款定）。

固定口径：FT / lb90 / predict=6 / amp=bf16 / batch_size=128。只读已消耗的
开发折 36--42；现有 sc=5 与实验重打分路径全部经
``signals.kronos_adapter.scores_path`` 生成，标签统一读既有 FT 路径。
三折曲线在同一组折 36/39/42 上比较 sc=5/10/20/40；七折曲线只并列
sc=5/20。stationary bootstrap 固定平均块长 10、10000 次、seed 20260904。
"""
from __future__ import annotations

import argparse
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

ALL_FOLDS = (36, 37, 38, 39, 40, 41, 42)
CURVE_FOLDS = (36, 39, 42)
SAMPLE_COUNTS = (5, 10, 20, 40)
NW_LAG = 5
TOPN = 500
MIN_NAMES = 50
BOOT_BLOCK = 10
BOOT_DRAWS = 10_000
BOOT_SEED = 20260904
OUT_JSON = REPO / "outputs" / "exp3_samplecount_curve.json"
OUT_SVG = REPO / "outputs" / "exp3_samplecount_curve.svg"


def _load_compare_arms_money():
    spec = importlib.util.spec_from_file_location(
        "compare_arms_money", REPO / "scripts" / "compare_arms_money.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_file(fold: int, sample_count: int) -> Path:
    if sample_count == 5:
        return scores_path(fold, "ft")
    return scores_path(
        fold, "ft", experiment_tag=f"e1_sc{sample_count}"
    )


def stationary_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    restart = 1.0 / BOOT_BLOCK
    idx = np.empty(n, dtype=np.int64)
    current = int(rng.integers(n))
    for i in range(n):
        idx[i] = current
        current = (
            int(rng.integers(n))
            if rng.random() < restart
            else (current + 1) % n
        )
    return idx


def stationary_mean_ci(values: pd.Series) -> list[float]:
    v = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(v) < 2:
        return [math.nan, math.nan]
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(BOOT_DRAWS, dtype=float)
    for i in range(BOOT_DRAWS):
        draws[i] = v[stationary_indices(len(v), rng)].mean()
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 3:
        return math.nan
    return float(a[ok].rank().corr(b[ok].rank()))


def _one_day_spread(group: pd.DataFrame) -> float:
    g = group.dropna(subset=["score", "label"])
    if len(g) < 10:
        return math.nan
    bins = pd.qcut(g["score"].rank(method="first"), 10, labels=False)
    return float(g.loc[bins == 9, "label"].mean() - g.loc[bins == 0, "label"].mean())


def _top500(frame: pd.DataFrame, adv: dict) -> pd.DataFrame:
    parts = []
    for day, group in frame.groupby("signal_date", sort=True):
        values = adv.get(day, {})
        g = group[group["PERMNO"].map(
            lambda p: p in values and np.isfinite(values[p])
        )].copy()
        if len(g) > TOPN:
            g["_adv20"] = g["PERMNO"].map(values)
            g = g.sort_values("_adv20", ascending=False, kind="mergesort").head(TOPN)
        if len(g) >= MIN_NAMES:
            parts.append(g.drop(columns=["_adv20"], errors="ignore"))
    if not parts:
        return frame.iloc[:0].copy()
    return pd.concat(parts, ignore_index=True)


def daily_metrics(scores: pd.DataFrame, labels: pd.DataFrame, adv: dict) -> dict[str, pd.Series]:
    """标签只作为 IC / 价差结果变量，不进入任何特征。"""
    merged = (
        scores[["PERMNO", "signal_date", "score"]]
        .dropna()
        .merge(
            labels[["PERMNO", "signal_date", "label", "status"]],
            on=["PERMNO", "signal_date"],
            how="inner",
        )
    )
    merged = merged[(merged["status"] == "ok") & merged["label"].notna()]
    top = _top500(merged, adv)

    def ic_series(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("signal_date", sort=True).apply(
            lambda g: spearman(g["score"], g["label"]),
            include_groups=False,
        )

    return {
        "ic_full": ic_series(merged),
        "ic_top500": ic_series(top),
        "decile_spread_full": merged.groupby("signal_date", sort=True).apply(
            _one_day_spread, include_groups=False
        ),
        "decile_spread_top500": top.groupby("signal_date", sort=True).apply(
            _one_day_spread, include_groups=False
        ),
    }


def reliability_daily(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    merged = (
        left[["PERMNO", "signal_date", "score"]]
        .rename(columns={"score": "left"})
        .merge(
            right[["PERMNO", "signal_date", "score"]].rename(
                columns={"score": "right"}
            ),
            on=["PERMNO", "signal_date"],
            how="inner",
        )
        .dropna()
    )
    return merged.groupby("signal_date", sort=True).apply(
        lambda g: spearman(g["left"], g["right"]),
        include_groups=False,
    )


def summarize(parts: dict[int, pd.Series], *, basis_points: bool = False) -> dict:
    pooled = pd.concat([parts[f] for f in sorted(parts)]).sort_index()
    nw = newey_west_tstat(pooled, NW_LAG)
    ci = stationary_mean_ci(pooled)
    scale = 1e4 if basis_points else 1.0
    per_fold = {str(f): float(parts[f].mean() * scale) for f in sorted(parts)}
    return {
        "mean": float(nw["mean"] * scale),
        "nw5_se": float(nw["se"] * scale),
        "nw5_t": float(nw["t"]),
        "stationary_bootstrap_ci95": [float(x * scale) for x in ci],
        "unit": "bp/day" if basis_points else "correlation",
        "n_days": int(nw["n"]),
        "folds_positive": int(sum(v > 0 for v in per_fold.values())),
        "per_fold_mean": per_fold,
    }


def _axis_points(values: list[tuple[float, float]], x0: float, y0: float,
                 width: float, height: float, ymin: float, ymax: float) -> str:
    xs = [x for x, _ in values]
    xmin, xmax = min(xs), max(xs)
    coords = []
    for x, y in values:
        px = x0 + width * (x - xmin) / (xmax - xmin if xmax != xmin else 1.0)
        py = y0 + height * (1.0 - (y - ymin) / (ymax - ymin))
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def write_svg(report: dict, path: Path) -> None:
    panels = [
        ("Matched folds 36/39/42: RankIC", "ic_full", "ic_top500"),
        ("Matched folds 36/39/42: decile spread (bp/day)",
         "decile_spread_full", "decile_spread_top500"),
    ]
    colors = ("#2563eb", "#dc2626")
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="720" viewBox="0 0 1000 720">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827}.axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:3}.dot{stroke:white;stroke-width:1.5}</style>',
        '<text x="40" y="38" font-size="24" font-weight="600">Experiment 3: sample-count curve</text>',
        '<text x="40" y="64" font-size="13">FT · lb90 · predict6 · bf16 · batch128 · diagnostic only</text>',
    ]
    matched = report["curves"]["matched_folds_36_39_42"]
    for panel_i, (title, key_a, key_b) in enumerate(panels):
        x0, y0, width, height = 80.0, 110.0 + panel_i * 270.0, 840.0, 190.0
        chunks.append(f'<text x="{x0}" y="{y0 - 18}" font-size="16" font-weight="600">{escape(title)}</text>')
        chunks.append(f'<line class="axis" x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}"/>')
        chunks.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}"/>')
        series = {
            key: [(float(sc), float(matched[str(sc)][key]["mean"]))
                  for sc in SAMPLE_COUNTS]
            for key in (key_a, key_b)
        }
        all_y = [y for values in series.values() for _, y in values]
        ymin, ymax = min(all_y), max(all_y)
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5
        pad = 0.08 * (ymax - ymin)
        ymin, ymax = ymin - pad, ymax + pad
        for j, key in enumerate((key_a, key_b)):
            values = series[key]
            points = _axis_points(values, x0, y0, width, height, ymin, ymax)
            chunks.append(f'<polyline class="line" stroke="{colors[j]}" points="{points}"/>')
            coords = points.split()
            for (sc, val), coord in zip(values, coords):
                px, py = coord.split(",")
                chunks.append(f'<circle class="dot" fill="{colors[j]}" cx="{px}" cy="{py}" r="5"/>')
                chunks.append(f'<text x="{float(px) + 7:.2f}" y="{float(py) - 7:.2f}" font-size="11">sc{int(sc)} {val:+.4f}</text>')
            chunks.append(f'<text x="{x0 + 610}" y="{y0 + 20 + j * 20}" font-size="12" fill="{colors[j]}">{escape(key)}</text>')
        for sc in SAMPLE_COUNTS:
            px = x0 + width * (sc - min(SAMPLE_COUNTS)) / (max(SAMPLE_COUNTS) - min(SAMPLE_COUNTS))
            chunks.append(f'<text x="{px - 10:.2f}" y="{y0 + height + 22}" font-size="11">{sc}</text>')
    rel = report["reliability"]
    chunks.extend([
        '<text x="80" y="642" font-size="16" font-weight="600">Daily score-rank reliability</text>',
        f'<text x="80" y="672" font-size="14">sc5 vs sc20 (7 folds): {rel["sc5_vs_sc20_all7"]["mean"]:.4f}  CI [{rel["sc5_vs_sc20_all7"]["stationary_bootstrap_ci95"][0]:.4f}, {rel["sc5_vs_sc20_all7"]["stationary_bootstrap_ci95"][1]:.4f}]</text>',
        f'<text x="520" y="672" font-size="14">sc20 vs sc40 (3 folds): {rel["sc20_vs_sc40_three"]["mean"]:.4f}  CI [{rel["sc20_vs_sc40_three"]["stationary_bootstrap_ci95"][0]:.4f}, {rel["sc20_vs_sc40_three"]["stationary_bootstrap_ci95"][1]:.4f}]</text>',
        '</svg>',
    ])
    path.write_text("\n".join(chunks), encoding="utf-8")


def load_scores(fold: int, sample_count: int) -> pd.DataFrame:
    path = score_file(fold, sample_count)
    assert_readable(path)
    frame = pd.read_parquet(path, columns=["PERMNO", "signal_date", "score"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-svg", type=Path, default=OUT_SVG)
    args = parser.parse_args()

    cam = _load_compare_arms_money()
    assert_readable(cam.P / "panel_raw.parquet")
    _, _, adv = cam.load_prices()

    scores: dict[tuple[int, int], pd.DataFrame] = {}
    metric_parts: dict[tuple[int, int], dict[str, pd.Series]] = {}
    for fold in ALL_FOLDS:
        counts = SAMPLE_COUNTS if fold in CURVE_FOLDS else (5, 20)
        label_path = labels_path(fold, "ft")
        assert_readable(label_path)
        labels = pd.read_parquet(
            label_path, columns=["PERMNO", "signal_date", "label", "status"]
        )
        labels["signal_date"] = pd.to_datetime(labels["signal_date"])
        for sample_count in counts:
            sc = load_scores(fold, sample_count)
            scores[(fold, sample_count)] = sc
            metric_parts[(fold, sample_count)] = daily_metrics(sc, labels, adv)
            print(f"fold{fold} sc={sample_count}: metrics ready", flush=True)

    def curve(folds: tuple[int, ...], counts: tuple[int, ...]) -> dict:
        result = {}
        for sample_count in counts:
            node = {}
            for metric in (
                "ic_full", "ic_top500", "decile_spread_full", "decile_spread_top500"
            ):
                parts = {f: metric_parts[(f, sample_count)][metric] for f in folds}
                node[metric] = summarize(
                    parts, basis_points=metric.startswith("decile_spread")
                )
            result[str(sample_count)] = node
        return result

    rel_5_20 = {
        f: reliability_daily(scores[(f, 5)], scores[(f, 20)]) for f in ALL_FOLDS
    }
    rel_20_40 = {
        f: reliability_daily(scores[(f, 20)], scores[(f, 40)]) for f in CURVE_FOLDS
    }
    report = {
        "meta": {
            "experiment": 3,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "configuration": {
                "arm": "ft", "lookback": 90, "predict": 6,
                "amp": "bf16", "batch_size": 128,
            },
            "all_folds": list(ALL_FOLDS),
            "matched_curve_folds": list(CURVE_FOLDS),
            "sample_counts": list(SAMPLE_COUNTS),
            "bootstrap": {
                "kind": "circular stationary", "mean_block": BOOT_BLOCK,
                "draws": BOOT_DRAWS, "seed": BOOT_SEED,
            },
            "use_restriction": "diagnostic only; do not change the prior sc=20 decision or historical scoring",
        },
        "curves": {
            "all_folds_36_42": curve(ALL_FOLDS, (5, 20)),
            "matched_folds_36_39_42": curve(CURVE_FOLDS, SAMPLE_COUNTS),
        },
        "reliability": {
            "sc5_vs_sc20_all7": summarize(rel_5_20),
            "sc20_vs_sc40_three": summarize(rel_20_40),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_svg(report, args.out_svg)

    print("\n=== matched folds 36/39/42 ===")
    for sc in SAMPLE_COUNTS:
        node = report["curves"]["matched_folds_36_39_42"][str(sc)]
        print(
            f"sc={sc:2d} fullIC={node['ic_full']['mean']:+.6f} "
            f"top500IC={node['ic_top500']['mean']:+.6f} "
            f"top500 spread={node['decile_spread_top500']['mean']:+.3f}bp",
            flush=True,
        )
    print("=== reliability ===")
    for key, value in report["reliability"].items():
        print(
            f"{key}: mean={value['mean']:.6f} "
            f"CI=[{value['stationary_bootstrap_ci95'][0]:.6f},"
            f"{value['stationary_bootstrap_ci95'][1]:.6f}]",
            flush=True,
        )
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
