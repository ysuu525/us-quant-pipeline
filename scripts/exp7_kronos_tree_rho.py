"""实验 7：Kronos 与树基线逐日 IC 的配对相关与 H3 的 MDE80。

判据 / 用途限制（先写死）

估计交付，不做任何判定。不得据此说 H3 通过或失败，也不得据此改基线配置或超参。
唯一用途：给 H3 定层级（A 层做推断 / B 层按先验），以及把 MDE80 写进协议 v4。
推断单位 = 逐日配对（CLAUDE.md §二功效关：折数只作一致性指标，不作标准误的基本单位）。
只在折 36–42 上做；树侧只读已落盘产物，绝不重训。

Kronos 的全池逐日 RankIC 逐字遵循
``scripts/e4_metric_alignment_power.py::daily_metrics`` 的全池分支：分数与
``status == "ok"`` 的 FT 标签按名字/日期内连接、删除缺失、截面内平均秩后算
Pearson 相关（即 Spearman），且少于 10 个名字的日期记为缺失。ZS 也统一读取
FT 标签。树侧只接受 ``daily_ic_ensemble.parquet``，并在读取前与同目录
``fold_summary.json`` 的 ``ensemble_daily_ic_sha256`` 逐字节核哈希；任何文件
缺失或哈希不符都 fail closed，绝不用折级汇总替代逐日推断。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402
from signals.kronos_adapter import labels_path, scores_path  # noqa: E402

FOLDS: tuple[int, ...] = tuple(range(36, 43))
MODELS: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")
ARMS: tuple[str, ...] = ("ft", "zs")
NW_LAG = 5
MDE80_MULTIPLIER = 2.8016
CI95_MULTIPLIER = 1.96
SESOI_HALF_CURRENT_GAP = 0.0072
SESOI_DISCOUNTED_GAP = 0.0108
DEFAULT_TREE_ROOT = REPO / "outputs" / "gbdt_strong_jkp_v2"
DEFAULT_OUTPUTS_ROOT = REPO / "outputs"
DEFAULT_OUT = REPO / "outputs" / "exp7_kronos_tree_rho.json"


class ArtifactPreflightError(RuntimeError):
    """树侧逐日产物缺失、登记不完整或哈希不匹配。"""


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def preflight_tree_artifacts(tree_root: Path) -> list[dict[str, Any]]:
    """一次性核完 3 树 × 7 折；通过前不读取任何逐日数值。"""
    checks: list[dict[str, Any]] = []
    problems: list[str] = []
    for model in MODELS:
        for fold in FOLDS:
            fold_name = f"fold{fold}"
            fold_dir = tree_root / model / fold_name
            summary_path = fold_dir / "fold_summary.json"
            daily_path = fold_dir / "daily_ic_ensemble.parquet"
            if not summary_path.is_file():
                problems.append(f"missing summary: {summary_path}")
                continue
            assert_readable(summary_path)
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                problems.append(f"invalid summary {summary_path}: {exc}")
                continue
            expected = summary.get("ensemble_daily_ic_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                problems.append(f"missing/invalid ensemble daily sha256: {summary_path}")
                continue
            if summary.get("model") != model or summary.get("fold") != fold_name:
                problems.append(
                    f"summary identity mismatch: {summary_path} "
                    f"model={summary.get('model')!r} fold={summary.get('fold')!r}"
                )
                continue
            if not daily_path.is_file():
                problems.append(f"missing daily IC: {daily_path}")
                continue
            assert_readable(daily_path)
            actual = sha256_file(daily_path)
            if actual != expected:
                problems.append(
                    f"daily IC sha256 mismatch: {daily_path} expected={expected} actual={actual}"
                )
                continue
            checks.append(
                {
                    "model": model,
                    "fold": fold,
                    "daily_ic_path": str(daily_path),
                    "fold_summary_path": str(summary_path),
                    "sha256": actual,
                    "status": "matched",
                }
            )
    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise ArtifactPreflightError(
            "树侧逐日 IC 前置检查失败；需先跑 scripts/gbdt_baseline.py "
            "重新生成逐日 IC 产物。禁止用 fold_summary.json 的折级数字凑配对。\n"
            f"{detail}"
        )
    expected_count = len(MODELS) * len(FOLDS)
    if len(checks) != expected_count:
        raise ArtifactPreflightError(
            f"树侧逐日 IC 前置检查内部计数错误：{len(checks)}/{expected_count}"
        )
    return checks


def _spearman_like_e4(left: pd.Series, right: pd.Series) -> float:
    """与 E4 `_spearman` 相同：平均秩后算 Pearson；n < 10 返回 NaN。"""
    if len(left) < 10:
        return math.nan
    x = left.rank().to_numpy(dtype=np.float64)
    y = right.rank().to_numpy(dtype=np.float64)
    # pandas 3 may expose a read-only ndarray; E4 also creates centered arrays
    # rather than mutating the rank result in place.
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denominator) if denominator > 0 else math.nan


def kronos_daily_ic(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """复现 E4 `daily_metrics` 的 `ic_full` 分支。"""
    score_columns = ["PERMNO", "signal_date", "score"]
    label_columns = ["PERMNO", "signal_date", "label", "status"]
    if not set(score_columns).issubset(scores.columns):
        raise ValueError(f"Kronos scores 缺列：{sorted(set(score_columns) - set(scores.columns))}")
    if not set(label_columns).issubset(labels.columns):
        raise ValueError(f"FT labels 缺列：{sorted(set(label_columns) - set(labels.columns))}")
    scores = scores[score_columns].copy()
    labels = labels[label_columns].copy()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    if scores.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError("Kronos scores 存在重复 (PERMNO, signal_date)")
    if labels.duplicated(["PERMNO", "signal_date"]).any():
        raise ValueError("FT labels 存在重复 (PERMNO, signal_date)")
    merged = scores.merge(
        labels.loc[labels["status"] == "ok", ["PERMNO", "signal_date", "label"]],
        on=["PERMNO", "signal_date"],
        how="inner",
        validate="one_to_one",
    ).dropna()
    rows = [
        {
            "signal_date": day,
            "rank_ic": _spearman_like_e4(group["score"], group["label"]),
            "n_obs": int(len(group)),
        }
        for day, group in merged.groupby("signal_date", sort=True)
    ]
    return pd.DataFrame(rows, columns=["signal_date", "rank_ic", "n_obs"])


def load_kronos_daily(fold: int, arm: str, outputs_root: Path) -> pd.DataFrame:
    """只经 adapter 解析路径；ZS 也从 FT 目录读取标签。"""
    score_file = scores_path(fold, arm, root=outputs_root)
    label_file = labels_path(fold, "ft", root=outputs_root)
    assert_readable(score_file)
    assert_readable(label_file)
    scores = pd.read_parquet(score_file, columns=["PERMNO", "signal_date", "score"])
    labels = pd.read_parquet(
        label_file, columns=["PERMNO", "signal_date", "label", "status"]
    )
    return kronos_daily_ic(scores, labels)


def load_tree_daily(tree_root: Path, model: str, fold: int) -> pd.DataFrame:
    path = tree_root / model / f"fold{fold}" / "daily_ic_ensemble.parquet"
    assert_readable(path)
    daily = pd.read_parquet(path, columns=["signal_date", "rank_ic"])
    daily["signal_date"] = pd.to_datetime(daily["signal_date"])
    if daily.duplicated("signal_date").any():
        raise ValueError(f"树逐日 IC 存在重复日期：{model}/fold{fold}")
    return daily.sort_values("signal_date", kind="mergesort").reset_index(drop=True)


def _finite_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def nw_se(values: Iterable[float]) -> float:
    return float(newey_west_tstat(pd.Series(list(values), dtype=float), NW_LAG)["se"])


def pair_stats(kronos: pd.DataFrame, tree: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """日期内连接，并以逐日配对差作为唯一推断单位。"""
    left = kronos[["signal_date", "rank_ic"]].rename(columns={"rank_ic": "kronos_ic"})
    right = tree[["signal_date", "rank_ic"]].rename(columns={"rank_ic": "tree_ic"})
    if left.duplicated("signal_date").any() or right.duplicated("signal_date").any():
        raise ValueError("逐日 IC 日期键必须唯一")
    joined = left.merge(right, on="signal_date", how="inner", validate="one_to_one")
    joined = joined.dropna(subset=["kronos_ic", "tree_ic"]).sort_values("signal_date")
    joined["difference"] = joined["kronos_ic"] - joined["tree_ic"]
    left_dates, right_dates = set(left["signal_date"]), set(right["signal_date"])
    paired_dates = set(joined["signal_date"])
    nw = newey_west_tstat(joined["difference"], NW_LAG)
    independent_se = math.sqrt(nw_se(joined["kronos_ic"]) ** 2 + nw_se(joined["tree_ic"]) ** 2)
    paired_se = float(nw["se"])
    if independent_se > 0:
        se_reduction = 1.0 - paired_se / independent_se
        variance_reduction = 1.0 - (paired_se / independent_se) ** 2
    else:
        se_reduction = variance_reduction = math.nan
    mean = float(nw["mean"])
    stats = {
        "n_paired_days": int(len(joined)),
        "n_kronos_days": int(len(left)),
        "n_tree_days": int(len(right)),
        "kronos_days_dropped": int(len(left_dates - paired_dates)),
        "tree_days_dropped": int(len(right_dates - paired_dates)),
        "drop_reason": (
            "date-set mismatch or non-finite IC removed before pairing"
            if len(left_dates - paired_dates) or len(right_dates - paired_dates)
            else "none"
        ),
        "pearson_rho": _finite_corr(joined["kronos_ic"], joined["tree_ic"], "pearson"),
        "spearman_rho": _finite_corr(joined["kronos_ic"], joined["tree_ic"], "spearman"),
        "paired_difference": {
            "mean": mean,
            "nw5_se": paired_se,
            "t": float(nw["t"]),
            "ci95": [mean - CI95_MULTIPLIER * paired_se, mean + CI95_MULTIPLIER * paired_se],
            "mde80": MDE80_MULTIPLIER * paired_se,
        },
        "independence_comparison": {
            "independent_nw5_se": independent_se,
            "paired_nw5_se": paired_se,
            "se_reduction_fraction": se_reduction,
            "variance_reduction_fraction": variance_reduction,
        },
    }
    return stats, joined


def hierarchy_advice(mde80: float) -> dict[str, Any]:
    comparisons = {
        "half_current_gap": {
            "sesoi": SESOI_HALF_CURRENT_GAP,
            "mde80_le_sesoi": bool(mde80 <= SESOI_HALF_CURRENT_GAP),
        },
        "discounted_current_gap": {
            "sesoi": SESOI_DISCOUNTED_GAP,
            "mde80_le_sesoi": bool(mde80 <= SESOI_DISCOUNTED_GAP),
        },
    }
    if all(item["mde80_le_sesoi"] for item in comparisons.values()):
        suggestion = "建议 H3 归 A 层做推断（最终层级仍由用户裁定）。"
    else:
        suggestion = (
            "建议 H3 降为 B 层 / 估计交付（最终层级仍由用户裁定）；"
            "在本样本量下该问题不可回答，按预先规则取冻结树基线。"
        )
    return {
        "mde80": float(mde80),
        "candidate_sesoi": comparisons,
        "suggestion_only_not_decision": suggestion,
    }


def build_report(
    kronos_daily: dict[str, dict[int, pd.DataFrame]],
    tree_daily: dict[str, dict[int, pd.DataFrame]],
    artifact_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    hierarchy: dict[str, Any] = {}
    for model in MODELS:
        results[model] = {}
        for arm in ARMS:
            fold_stats: dict[str, Any] = {}
            pooled_parts: list[pd.DataFrame] = []
            for fold in FOLDS:
                stats, paired = pair_stats(kronos_daily[arm][fold], tree_daily[model][fold])
                fold_stats[str(fold)] = stats
                pooled_parts.append(paired.assign(fold=fold))
            pooled = pd.concat(pooled_parts, ignore_index=True).sort_values(
                ["signal_date", "fold"], kind="mergesort"
            )
            combined, _ = pair_stats(
                pooled[["signal_date", "kronos_ic"]].rename(
                    columns={"kronos_ic": "rank_ic"}
                ),
                pooled[["signal_date", "tree_ic"]].rename(columns={"tree_ic": "rank_ic"}),
            )
            results[model][arm] = {
                "primary": bool(model == "xgboost"),
                "by_fold": fold_stats,
                "combined": combined,
            }
            if model == "xgboost":
                hierarchy[arm] = hierarchy_advice(combined["paired_difference"]["mde80"])
    return {
        "meta": {
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "folds": list(FOLDS),
            "models": list(MODELS),
            "arms": list(ARMS),
            "primary_tree": "xgboost",
            "nw_lag": NW_LAG,
            "mde80_multiplier": MDE80_MULTIPLIER,
            "inference_unit": "paired daily full-universe RankIC",
            "use_restriction": (
                "estimate only; do not judge H3 or change baseline/configuration; "
                "only informs proposed H3 tier and protocol-v4 MDE80"
            ),
        },
        "tree_artifact_checks": artifact_checks,
        "results": results,
        "h3_tier_advice_xgboost": hierarchy,
    }


def run(tree_root: Path, outputs_root: Path) -> dict[str, Any]:
    checks = preflight_tree_artifacts(tree_root)
    kronos_daily = {
        arm: {fold: load_kronos_daily(fold, arm, outputs_root) for fold in FOLDS}
        for arm in ARMS
    }
    tree_daily = {
        model: {fold: load_tree_daily(tree_root, model, fold) for fold in FOLDS}
        for model in MODELS
    }
    return build_report(kronos_daily, tree_daily, checks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-root", type=Path, default=DEFAULT_TREE_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args.tree_root.resolve(), args.outputs_root.resolve())
    except ArtifactPreflightError as exc:
        print(f"[exp7] STOP: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[exp7] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
