"""K9x: exact K6b-characteristic holdings-tilt audit.

Pre-registered in experiments/ledger.md before execution.  This script reuses
the exact point-in-time feature builder from k6b_spanning.py and does not read
future labels or unopened folds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from k6b_spanning import build_factors, load_panel


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
FOLDS = [f"fold{i}" for i in range(36, 43)]
FEATURES = ("invprice", "amihud", "idiovol", "hlrange")
TOPN, NT, EXIT_PCT = 500, 6, 0.30
NW_LAG, N_BOOT, MEAN_BLOCK, SEED = 5, 10_000, 10, 20260901


def nw_mean(values: np.ndarray, lags: int = NW_LAG) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mean = float(x.mean())
    residual = x - mean
    total = float(residual @ residual)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        total += 2.0 * weight * float(residual[lag:] @ residual[:-lag])
    se = math.sqrt(max(total, 0.0)) / n
    return mean, se, mean / se if se > 0 else math.nan


def stationary_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    restart = 1.0 / MEAN_BLOCK
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(n))
    for j in range(n):
        idx[j] = cur
        cur = int(rng.integers(n)) if rng.random() < restart else (cur + 1) % n
    return idx


def bootstrap_mean(frame: pd.DataFrame, column: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    groups = [g[column].dropna().to_numpy() for _, g in frame.groupby("fold")]
    draws = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        sample = [values[stationary_idx(len(values), rng)] for values in groups]
        draws[b] = float(np.concatenate(sample).mean())
    return draws


def build_daily() -> pd.DataFrame:
    print("构造与 K6b 完全相同的四个特征...", flush=True)
    panel = build_factors(load_panel())
    frames = {day: g for day, g in panel.groupby("DlyCalDt")}
    del panel

    rows = []
    for fold in FOLDS:
        score_dir = OUT / f"{fold}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{fold}"
        scores = pd.read_parquet(
            score_dir / "scores.parquet", columns=["PERMNO", "signal_date", "score"]
        ).dropna()
        scores["signal_date"] = pd.to_datetime(scores["signal_date"])
        by_day = {
            day: dict(zip(g.PERMNO, g.score))
            for day, g in scores.groupby("signal_date") if len(g) >= 50
        }
        del scores
        days = sorted(by_day)
        book: list[list[int] | None] = [None] * NT
        usable = 0

        for day_index, day in enumerate(days):
            day_frame = frames.get(day)
            if day_frame is None:
                continue
            raw_scores = by_day[day]
            adv = dict(zip(day_frame.PERMNO, day_frame.adv20))
            eligible = [
                p for p in raw_scores if p in adv and np.isfinite(adv[p])
            ]
            if len(eligible) > TOPN:
                eligible = sorted(eligible, key=lambda p: -adv[p])[:TOPN]
            eligible_set = set(eligible)
            current_scores = {p: v for p, v in raw_scores.items() if p in eligible_set}
            if len(current_scores) < 50:
                continue

            n = len(current_scores)
            score_pct = (pd.Series(current_scores).rank() / n).to_dict()
            order = sorted(score_pct, key=lambda p: -score_pct[p])
            k = max(1, n // 10)
            sleeve = day_index % NT
            previous = book[sleeve]
            if previous is None:
                selected = list(order[:k])
            else:
                kept = [
                    p for p in previous
                    if p in score_pct and score_pct[p] >= 1.0 - EXIT_PCT
                ][:k]
                kept_set = set(kept)
                added = [p for p in order if p not in kept_set][:k - len(kept)]
                selected = kept + added
            book[sleeve] = selected
            if day_index < NT or any(names is None for names in book):
                continue

            held = [p for names in book if names is not None for p in names]
            pool = day_frame[day_frame.PERMNO.isin(eligible_set)]
            for feature in FEATURES:
                valid = pool[["PERMNO", feature]].dropna()
                if len(valid) < 50:
                    continue
                ranks = valid[feature].rank(method="average", pct=True)
                rank_by_name = dict(zip(valid.PERMNO, ranks))
                pool_mean = float(ranks.mean())
                formation_values = [rank_by_name[p] for p in selected if p in rank_by_name]
                held_values = [rank_by_name[p] for p in held if p in rank_by_name]
                if not formation_values or not held_values:
                    continue
                rows.append({
                    "date": day,
                    "fold": fold,
                    "feature": feature,
                    "pool_n": int(len(valid)),
                    "pool_rank_mean": pool_mean,
                    "formation_rank_mean": float(np.mean(formation_values)),
                    "formation_tilt": float(np.mean(formation_values) - pool_mean),
                    "held_rank_mean": float(np.mean(held_values)),
                    "held_tilt": float(np.mean(held_values) - pool_mean),
                    "formation_coverage": len(formation_values) / len(selected),
                    "held_coverage": len(held_values) / len(held),
                })
            usable += 1
        print(f"  {fold}: {usable} 天", flush=True)
    return pd.DataFrame(rows).sort_values(["date", "feature"]).reset_index(drop=True)


def summarize(daily: pd.DataFrame) -> dict:
    result = {
        "feature_definitions": "exact reuse of k6b_spanning.build_factors",
        "folds": FOLDS,
        "features": {},
    }
    for feature in FEATURES:
        part = daily[daily.feature == feature]
        entry = {
            "n_days": int(part.date.nunique()),
            "mean_pool_n": float(part.pool_n.mean()),
            "mean_formation_coverage": float(part.formation_coverage.mean()),
            "mean_held_coverage": float(part.held_coverage.mean()),
        }
        for column in ("formation_tilt", "held_tilt"):
            mean, se, t_stat = nw_mean(part[column].to_numpy())
            draws = bootstrap_mean(part, column)
            fold_means = part.groupby("fold")[column].mean()
            entry[column] = {
                "mean": mean,
                "nw5_se": se,
                "nw5_t": t_stat,
                "bootstrap_ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
                "folds_positive": int((fold_means > 0).sum()),
                "folds_negative": int((fold_means < 0).sum()),
                "per_fold": {k: float(v) for k, v in fold_means.items()},
            }
        result["features"][feature] = entry
    return result


def main() -> None:
    daily = build_daily()
    result = summarize(daily)
    daily.to_parquet(OUT / "holdings_tilt_exact_k6b_daily.parquet", index=False)
    (OUT / "holdings_tilt_exact_k6b.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n特征               形成袖套倾斜 [95%CI]          六袖套持仓倾斜 [95%CI]", flush=True)
    for feature in FEATURES:
        f = result["features"][feature]
        a, b = f["formation_tilt"], f["held_tilt"]
        print(
            f"{feature:<18} {a['mean']:+.4f} [{a['bootstrap_ci95'][0]:+.4f},"
            f"{a['bootstrap_ci95'][1]:+.4f}]   {b['mean']:+.4f} "
            f"[{b['bootstrap_ci95'][0]:+.4f},{b['bootstrap_ci95'][1]:+.4f}]",
            flush=True,
        )
    print("写入 outputs/holdings_tilt_exact_k6b.json 与 daily.parquet", flush=True)


if __name__ == "__main__":
    main()
