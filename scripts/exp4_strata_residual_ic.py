"""实验 4（E8）：STRATA 式分数残差化后的 IC。

判据：诊断。预期残差 IC 低于原 IC；若残差 IC / 原 IC < 0.5，论文须披露
「信号大部分可由价量风格解释」。用途限制：不改信号、不改构造。

八个价量风格因子直接 import ``scripts/k6b_spanning.py`` 的 ``CORE`` 与
``build_factors``，因子定义及无前视约束以该文件为唯一来源。本脚本只读已消耗的
开发折 36--42；标签只作为 IC 结果变量，统一从 FT 评估目录读取。每天先取
ADV20 最大的 500 个可用名字，再在完整因子样本上做含截距的截面 OLS。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from crsp_pipeline.signal_eval import daily_rank_ic, newey_west_tstat  # noqa: E402
from k6b_spanning import CORE, build_factors  # noqa: E402
from signals.kronos_adapter import load_kronos_scores, load_labels  # noqa: E402

PROCESSED = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")
OUT = REPO / "outputs" / "exp4_strata_residual_ic.json"
FOLDS = tuple(range(36, 43))
TOPN = 500
NW_LAGS = 5
PANEL_COLUMNS = [
    "PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
    "DlyVol", "DlyPrcVol", "DlyRet",
]


def log(message: str) -> None:
    print(message, flush=True)


def _factor_start(first_signal_date: pd.Timestamp) -> pd.Timestamp:
    """取首个信号日前 270 个交易日，覆盖 K6b 最长 252 日因子窗口。"""
    market = pd.read_parquet(PROCESSED / "market_index.parquet", columns=["caldt"])
    dates = pd.DatetimeIndex(pd.to_datetime(market["caldt"]).sort_values().unique())
    pos = int(dates.searchsorted(first_signal_date, side="left"))
    return pd.Timestamp(dates[max(0, pos - 270)])


def load_fold_factors(first_signal_date: pd.Timestamp,
                      last_signal_date: pd.Timestamp) -> pd.DataFrame:
    """按折日期下推读取价格面板，避免把 4,987 万行整表载入内存。"""
    lo = _factor_start(first_signal_date)
    raw = pd.read_parquet(
        PROCESSED / "panel_raw.parquet",
        columns=PANEL_COLUMNS,
        filters=[("DlyCalDt", ">=", lo), ("DlyCalDt", "<=", last_signal_date)],
    )
    raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"])
    raw = (raw.dropna(subset=["DlyRet"])
           .sort_values(["PERMNO", "DlyCalDt"])
           .reset_index(drop=True))
    factors = build_factors(raw)
    return factors[
        factors["DlyCalDt"].between(first_signal_date, last_signal_date)
    ].copy()


def residualize_one_day(frame: pd.DataFrame,
                        factor_cols: tuple[str, ...] = tuple(CORE)) -> pd.DataFrame:
    """在一天的完整因子样本上做 ``score ~ 1 + factors`` 并返回残差。"""
    needed = ["score", "label", *factor_cols]
    work = frame.dropna(subset=needed).copy()
    work = work[np.isfinite(work[needed]).all(axis=1)]
    if len(work) <= len(factor_cols) + 1:
        return work.assign(residual_score=np.nan)
    x = np.column_stack([
        np.ones(len(work), dtype=float),
        work.loc[:, factor_cols].to_numpy(dtype=float),
    ])
    y = work["score"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    work["residual_score"] = y - x @ coef
    return work


def prepare_fold_frame(fold: int) -> pd.DataFrame:
    scores = load_kronos_scores([fold], "ft")
    labels = load_labels([fold], arm="ft", ok_only=True)
    first, last = scores["signal_date"].min(), scores["signal_date"].max()
    factors = load_fold_factors(first, last).rename(columns={"DlyCalDt": "signal_date"})
    merged = (scores.merge(
        labels[["PERMNO", "signal_date", "label"]],
        on=["PERMNO", "signal_date"], how="inner")
        .merge(
            factors[["PERMNO", "signal_date", "adv20", *CORE]],
            on=["PERMNO", "signal_date"], how="inner"))

    days: list[pd.DataFrame] = []
    for _, group in merged.groupby("signal_date", sort=True):
        eligible = group.dropna(subset=["adv20"])
        eligible = eligible[np.isfinite(eligible["adv20"])]
        if len(eligible) > TOPN:
            eligible = eligible.nlargest(TOPN, "adv20")
        days.append(residualize_one_day(eligible))
    if not days:
        return pd.DataFrame(columns=[*merged.columns, "residual_score"])
    return pd.concat(days, ignore_index=True)


def summarize_ic(frame: pd.DataFrame) -> dict:
    raw_ic = daily_rank_ic(frame, score_col="score", label_col="label")
    residual_ic = daily_rank_ic(
        frame, score_col="residual_score", label_col="label")
    raw_stats = newey_west_tstat(raw_ic, NW_LAGS)
    residual_stats = newey_west_tstat(residual_ic, NW_LAGS)
    raw_mean = float(raw_stats["mean"])
    residual_mean = float(residual_stats["mean"])
    ratio = residual_mean / raw_mean if raw_mean != 0 else np.nan
    return {
        "n_obs": int(len(frame)),
        "n_days": int(raw_ic.notna().sum()),
        "original_ic": raw_stats,
        "residual_ic": residual_stats,
        "residual_to_original_ratio": float(ratio),
        "disclosure_triggered": bool(np.isfinite(ratio) and ratio < 0.5),
        "daily_original_ic": {
            str(pd.Timestamp(k).date()): float(v) for k, v in raw_ic.dropna().items()
        },
        "daily_residual_ic": {
            str(pd.Timestamp(k).date()): float(v) for k, v in residual_ic.dropna().items()
        },
    }


def main() -> None:
    per_fold: dict[str, dict] = {}
    pooled_frames: list[pd.DataFrame] = []
    for fold in FOLDS:
        log(f"fold{fold}: 日期下推读取、构造 K6b 八因子并残差化...")
        frame = prepare_fold_frame(fold)
        pooled_frames.append(frame)
        per_fold[str(fold)] = summarize_ic(frame)
        r = per_fold[str(fold)]
        log(
            f"  days={r['n_days']} original={r['original_ic']['mean']:+.6f} "
            f"residual={r['residual_ic']['mean']:+.6f} "
            f"ratio={r['residual_to_original_ratio']:.3f}"
        )
        del frame

    pooled = summarize_ic(pd.concat(pooled_frames, ignore_index=True))
    result = {
        "experiment": "exp4_strata_residual_ic",
        "criteria": (
            "诊断；预期残差 IC 低于原 IC；若残差 IC / 原 IC < 0.5，"
            "论文须披露信号大部分可由价量风格解释。"
        ),
        "use_restriction": "不改信号、不改构造。",
        "folds": list(FOLDS),
        "universe": "daily top500 by lagged ADV20, complete 8-factor cases",
        "factor_source": "scripts/k6b_spanning.py::CORE, build_factors",
        "factor_columns": list(CORE),
        "per_fold": per_fold,
        "pooled": pooled,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\nfold  days  original IC (t)    residual IC (t)    ratio")
    for fold, row in per_fold.items():
        log(
            f"{fold:>4}  {row['n_days']:>4}  "
            f"{row['original_ic']['mean']:+.6f} ({row['original_ic']['t']:+.2f})  "
            f"{row['residual_ic']['mean']:+.6f} ({row['residual_ic']['t']:+.2f})  "
            f"{row['residual_to_original_ratio']:.3f}"
        )
    log(
        f"ALL   {pooled['n_days']:>4}  "
        f"{pooled['original_ic']['mean']:+.6f} ({pooled['original_ic']['t']:+.2f})  "
        f"{pooled['residual_ic']['mean']:+.6f} ({pooled['residual_ic']['t']:+.2f})  "
        f"{pooled['residual_to_original_ratio']:.3f}"
    )
    log(f"写入 {OUT}")


if __name__ == "__main__":
    main()
