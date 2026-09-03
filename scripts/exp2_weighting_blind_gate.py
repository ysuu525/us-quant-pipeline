"""实验 2：分数加权 vs 等权，盲功效关先行。

判据（先写死）：

配对量 d_t = r_rank,t − r_equal,t（FT，折 36–42，NT=5，cost=0，逐日毛超额差）。
SESOI = 0.50 bp/日（= NT=5 毛 12.68%/年 的 10% ÷ 252）。
盲阶段只算 SE_NW5(d)、MDE80 = 2.8016 × SE、两构造的年单边换手比。均值必须遮住（写入 outputs/exp2_weighting.blind.json 并在控制台打印 ***）。
MDE80 > SESOI → 结论「在本样本量下该问题不可回答，按先验取等权（上游默认 1/N）」，不揭盲，实验结束。
MDE80 ≤ SESOI → 用 --unblind 揭盲：报 mean、NW t、stationary bootstrap 95% CI（块长 10、10000 次、seed 20260904）、7 折中 d 均值 > 0 的折数、换手比。通过判据：t ≥ 1.96 ∧ ≥ 5/7 折同向 ∧ 换手增幅 ≤ 20% → 记「设计修订候选，待用户裁定」；否则「不采用」。
用途限制：无论结果如何，执行者不得改动任何冻结构造；只交回。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402
from crsp_pipeline.signal_eval import newey_west_tstat  # noqa: E402
from portfolio.construction import (  # noqa: E402
    frozen_long_only_returns_weighted,
    scores_frame_to_by_day,
)
from signals.kronos_adapter import scores_path  # noqa: E402

FOLDS = (36, 37, 38, 39, 40, 41, 42)
ARM = "ft"
NT = 5
TOPN = 500
EXIT_PCT = 0.30
MIN_NAMES = 50
NW_LAG = 5
SESOI = 0.50 / 1e4
MDE80_MULTIPLIER = 2.8016
BOOT_BLOCK = 10
BOOT_DRAWS = 10_000
BOOT_SEED = 20260904
BLIND_OUT = REPO / "outputs" / "exp2_weighting.blind.json"
UNBLIND_OUT = REPO / "outputs" / "exp2_weighting.unblind.json"


def _load_compare_arms_money():
    spec = importlib.util.spec_from_file_location(
        "compare_arms_money", REPO / "scripts" / "compare_arms_money.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stationary_indices(
    n: int, rng: np.random.Generator, mean_block: int = BOOT_BLOCK
) -> np.ndarray:
    restart = 1.0 / mean_block
    indices = np.empty(n, dtype=np.int64)
    current = int(rng.integers(n))
    for i in range(n):
        indices[i] = current
        current = (
            int(rng.integers(n))
            if rng.random() < restart
            else (current + 1) % n
        )
    return indices


def stationary_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(BOOT_DRAWS, dtype=float)
    for i in range(BOOT_DRAWS):
        draws[i] = values[stationary_indices(len(values), rng)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def load_paired_returns() -> pd.DataFrame:
    cam = _load_compare_arms_money()
    assert_readable(cam.P / "panel_raw.parquet")
    print("加载价格/成交额（复用 compare_arms_money.load_prices）...", flush=True)
    ret, oc, adv = cam.load_prices()
    parts = []
    for fold in FOLDS:
        path = scores_path(fold, ARM)
        assert_readable(path)
        scores = pd.read_parquet(
            path, columns=["PERMNO", "signal_date", "score"]
        )
        by_day = scores_frame_to_by_day(scores, min_names=MIN_NAMES)
        equal = frozen_long_only_returns_weighted(
            by_day,
            ret,
            oc,
            adv,
            topn=TOPN,
            cost_bp=0.0,
            exit_pct=EXIT_PCT,
            nt=NT,
            min_names=MIN_NAMES,
            weighting="equal",
        )
        rank = frozen_long_only_returns_weighted(
            by_day,
            ret,
            oc,
            adv,
            topn=TOPN,
            cost_bp=0.0,
            exit_pct=EXIT_PCT,
            nt=NT,
            min_names=MIN_NAMES,
            weighting="rank",
        )
        if not equal.index.equals(rank.index):
            raise AssertionError(f"fold{fold}: 两构造的成交日不一致")
        paired = pd.DataFrame(
            {
                "r_equal": equal["r"],
                "r_rank": rank["r"],
                "turn_equal": equal["turn"],
                "turn_rank": rank["turn"],
                "fold": fold,
            }
        )
        paired["d"] = paired["r_rank"] - paired["r_equal"]
        parts.append(paired)
        print(f"  fold{fold}: days={len(paired)}", flush=True)
    return pd.concat(parts).sort_index()


def annual_oneway_turnover(turn: pd.Series) -> float:
    return float(2.0 * 252.0 * turn.mean() / NT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unblind", action="store_true")
    parser.add_argument("--blind-out", type=Path, default=BLIND_OUT)
    parser.add_argument("--unblind-out", type=Path, default=UNBLIND_OUT)
    args = parser.parse_args()

    paired = load_paired_returns()
    nw = newey_west_tstat(paired["d"], NW_LAG)
    se = float(nw["se"])
    mde80 = MDE80_MULTIPLIER * se
    equal_turn = annual_oneway_turnover(paired["turn_equal"])
    rank_turn = annual_oneway_turnover(paired["turn_rank"])
    turn_ratio = rank_turn / equal_turn
    gate_pass = mde80 <= SESOI
    blind_conclusion = (
        "功效关通过，可用 --unblind 揭盲"
        if gate_pass
        else "在本样本量下该问题不可回答，按先验取等权（上游默认 1/N）"
    )
    blind = {
        "meta": {
            "experiment": 2,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "arm": ARM,
            "folds": list(FOLDS),
            "construction": {
                "nt": NT,
                "topn": TOPN,
                "exit_pct": EXIT_PCT,
                "cost_bp": 0.0,
            },
            "nw_lag": NW_LAG,
        },
        "paired_difference": {
            "definition": "r_rank - r_equal",
            "n_days": int(len(paired)),
            "mean_daily_return": "***",
            "nw_t": "***",
            "nw5_se": se,
            "nw5_se_bp_per_day": se * 1e4,
            "mde80": mde80,
            "mde80_bp_per_day": mde80 * 1e4,
            "sesoi": SESOI,
            "sesoi_bp_per_day": SESOI * 1e4,
        },
        "turnover": {
            "equal_annual_oneway": equal_turn,
            "rank_annual_oneway": rank_turn,
            "rank_over_equal": turn_ratio,
            "rank_increase_pct": (turn_ratio - 1.0) * 100.0,
        },
        "gate": {
            "mde80_le_sesoi": bool(gate_pass),
            "unblind_allowed": bool(gate_pass),
            "conclusion": blind_conclusion,
        },
    }
    args.blind_out.parent.mkdir(parents=True, exist_ok=True)
    args.blind_out.write_text(
        json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== BLIND GATE ===")
    print("mean(d): ***")
    print("NW t(d): ***")
    print(f"SE_NW5(d): {se * 1e4:.6f} bp/day")
    print(f"MDE80: {mde80 * 1e4:.6f} bp/day")
    print(f"SESOI: {SESOI * 1e4:.6f} bp/day")
    print(
        f"annual one-way turnover: equal={equal_turn:.6f}x "
        f"rank={rank_turn:.6f}x ratio={turn_ratio:.6f}"
    )
    print(blind_conclusion)
    print(f"wrote {args.blind_out}")

    if not args.unblind or not gate_pass:
        return

    ci_lo, ci_hi = stationary_mean_ci(paired["d"].to_numpy(dtype=float))
    per_fold = paired.groupby("fold", sort=True)["d"].mean()
    folds_positive = int((per_fold > 0).sum())
    turnover_ok = (turn_ratio - 1.0) <= 0.20
    pass_all = bool(nw["t"] >= 1.96 and folds_positive >= 5 and turnover_ok)
    conclusion = "设计修订候选，待用户裁定" if pass_all else "不采用"
    unblind = {
        "meta": blind["meta"],
        "blind_gate": blind,
        "unblinded": {
            "mean": float(nw["mean"]),
            "mean_bp_per_day": float(nw["mean"] * 1e4),
            "nw_t": float(nw["t"]),
            "stationary_bootstrap_ci95": [ci_lo, ci_hi],
            "stationary_bootstrap_ci95_bp_per_day": [ci_lo * 1e4, ci_hi * 1e4],
            "bootstrap": {
                "mean_block": BOOT_BLOCK,
                "draws": BOOT_DRAWS,
                "seed": BOOT_SEED,
            },
            "per_fold_mean": {str(k): float(v) for k, v in per_fold.items()},
            "folds_positive": folds_positive,
            "turnover_increase_le_20pct": bool(turnover_ok),
            "criteria_pass": pass_all,
            "conclusion": conclusion,
        },
    }
    args.unblind_out.write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== UNBLINDED ===")
    print(f"mean(d): {nw['mean'] * 1e4:+.6f} bp/day")
    print(f"NW t(d): {nw['t']:+.6f}")
    print(f"stationary bootstrap 95% CI: [{ci_lo * 1e4:+.6f}, {ci_hi * 1e4:+.6f}] bp/day")
    print(f"positive folds: {folds_positive}/7")
    print(f"turnover ratio: {turn_ratio:.6f}")
    print(conclusion)
    print(f"wrote {args.unblind_out}")


if __name__ == "__main__":
    main()
