"""K11: decide whether zero-shot deserves a second confirmation slot.

The decision rule was appended to experiments/ledger.md before this script ran.
Only already-consumed folds 36--42 are read.  No ADV-bin or year breakdown is
produced: the scientific endpoint is the single preregistered delta-ADV scalar.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from k9_cost_alpha_frontier import (
    COST_GRID_BP,
    EXIT_PCT,
    FOLDS,
    NT,
    OUT,
    TOPN,
    load_market,
    nw_mean,
    rebalanced_return,
    weight_turnover,
    weighted_return,
)


N_BOOT, BOOT_SEED, BLOCK_MEAN = 10_000, 20260901, 10
ARM_PATHS = {
    "FT": {
        fold: OUT / f"{fold}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{fold}"
        for fold in FOLDS
    },
    "ZS": {
        fold: OUT / "zeroshot_base" / f"eval_zeroshot_{fold}"
        for fold in FOLDS
    },
}


def _guard(p):
    """封存守卫：拒绝读取带 SEALED 哨兵的目录（2026-09-02 计算授权 != 读取授权）。"""
    import sys as _s
    from pathlib import Path as _P
    _r = _P(__file__).resolve().parents[1] / "src"
    if str(_r) not in _s.path:
        _s.path.insert(0, str(_r))
    from crsp_pipeline.sealed import assert_readable
    assert_readable(p)
    return p


def stationary_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    j = int(rng.integers(n))
    for i in range(n):
        idx[i] = j
        j = int(rng.integers(n)) if rng.random() < 1.0 / BLOCK_MEAN else (j + 1) % n
    return idx


def load_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_parquet(
        _guard(path) / "scores.parquet", columns=["PERMNO", "signal_date", "score"]
    ).dropna()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    return scores


def portfolio_arm(arm: str, ret: dict, oc: dict, adv: dict) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        scores = load_scores(ARM_PATHS[arm][fold])
        by_day = {
            day: dict(zip(group.PERMNO, group.score))
            for day, group in scores.groupby("signal_date") if len(group) >= 50
        }
        days = sorted(by_day)
        selection_book: list[list[int] | None] = [None] * NT
        path_book: list[dict[int, float] | None] = [None] * NT
        for i, day in enumerate(days):
            raw = by_day[day]
            day_adv = adv.get(day, {})
            eligible = [p for p in raw if p in day_adv and np.isfinite(day_adv[p])]
            if len(eligible) > TOPN:
                eligible = sorted(eligible, key=lambda p: -day_adv[p])[:TOPN]
            eligible_set = set(eligible)
            current = {p: raw[p] for p in raw if p in eligible_set}
            if len(current) < 50:
                continue
            pct = (pd.Series(current).rank() / len(current)).to_dict()
            order = sorted(pct, key=lambda p: -pct[p])
            k = max(1, len(current) // 10)
            sleeve = i % NT
            previous_selection = selection_book[sleeve]
            if previous_selection is None:
                selected = list(order[:k])
            else:
                kept = [
                    p for p in previous_selection
                    if p in pct and pct[p] >= 1.0 - EXIT_PCT
                ][:k]
                held = set(kept)
                selected = kept + [p for p in order if p not in held][:k - len(kept)]
            selection_book[sleeve] = selected
            new = {p: 1.0 / k for p in selected}
            old = path_book[sleeve]
            turnover, _ = weight_turnover(old, new)

            next_day = days[i + 1] if i + 1 < len(days) else None
            can_score = next_day is not None and i >= NT
            if can_score:
                full_ret = ret.get(next_day, {})
                intraday_ret = oc.get(next_day, {})
                benchmark_values = [
                    full_ret[p] for p in pct if p in full_ret and np.isfinite(full_ret[p])
                ]
                sleeve_returns = []
                for slot in range(NT):
                    old_slot = path_book[slot]
                    if old_slot is None:
                        continue
                    value = (
                        rebalanced_return(old_slot, new, full_ret, intraday_ret)
                        if slot == sleeve else weighted_return(old_slot, full_ret)
                    )
                    if np.isfinite(value):
                        sleeve_returns.append(value)
                if sleeve_returns and benchmark_values:
                    rows.append({
                        "date": next_day,
                        "fold": fold,
                        "arm": arm,
                        "gross": float(np.mean(sleeve_returns) - np.mean(benchmark_values)),
                        "turnover": turnover,
                    })
            path_book[sleeve] = new
    return pd.DataFrame(rows)


def scientific_arm(arm: str, adv: dict) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        path = ARM_PATHS[arm][fold]
        scores = load_scores(path)
        labels = pd.read_parquet(
            path / "labels.parquet",
            columns=["PERMNO", "signal_date", "label", "status"],
        )
        labels["signal_date"] = pd.to_datetime(labels["signal_date"])
        merged = scores.merge(labels, on=["PERMNO", "signal_date"])
        merged = merged[(merged.status == "ok") & merged.label.notna()]
        for day, group in merged.groupby("signal_date"):
            day_adv = adv.get(day, {})
            group = group.assign(_adv=[day_adv.get(p, np.nan) for p in group.PERMNO])
            group = group[np.isfinite(group._adv)]
            if len(group) < 100:
                continue
            ic_full = group.score.rank().corr(group.label.rank())
            top = group.nlargest(min(TOPN, len(group)), "_adv")
            ic_top = top.score.rank().corr(top.label.rank()) if len(top) >= 50 else np.nan
            if np.isfinite(ic_full) and np.isfinite(ic_top):
                rows.append({
                    "date": day,
                    "fold": fold,
                    "arm": arm,
                    "ic_full": float(ic_full),
                    "ic_top500": float(ic_top),
                    "delta_adv": float(ic_top - ic_full),
                })
    return pd.DataFrame(rows)


def summarize_arm(portfolio: pd.DataFrame, science: pd.DataFrame) -> dict:
    gross = portfolio.gross.to_numpy()
    mu, se, t_value = nw_mean(gross)
    delta_mu, delta_se, delta_t = nw_mean(science.delta_adv.to_numpy())
    top_mu, top_se, top_t = nw_mean(science.ic_top500.to_numpy())
    full_mu, full_se, full_t = nw_mean(science.ic_full.to_numpy())
    turn = float(portfolio.turnover.mean())
    annual_gross = float(mu * 252 * 100)
    breakeven = float(mu / turn * 1e4) if turn > 0 else math.nan
    costs = {}
    for cost in COST_GRID_BP:
        net = portfolio.gross - cost / 1e4 * portfolio.turnover
        costs[str(cost)] = float(net.mean() * 252 * 100)
    eligible = {
        "gross_ann_ge_5": annual_gross >= 5.0,
        "breakeven_ge_11": breakeven >= 11.0,
        "delta_adv_ge_half_modern": delta_mu >= 0.0026,
        "top500_ic_positive": top_mu > 0,
    }
    return {
        "n_money_days": int(len(portfolio)),
        "gross_mean_daily_bp": float(mu * 1e4),
        "gross_nw_se_daily_bp": float(se * 1e4),
        "gross_nw_t": float(t_value),
        "gross_annual_arithmetic_pct": annual_gross,
        "active_vol_annual_pct": float(portfolio.gross.std(ddof=1) * np.sqrt(252) * 100),
        "mean_one_way_traded_fraction": turn,
        "breakeven_one_way_bp": breakeven,
        "net_annual_arithmetic_pct_by_cost": costs,
        "n_science_days": int(len(science)),
        "ic_full": {"mean": float(full_mu), "nw_se": float(full_se), "nw_t": float(full_t)},
        "ic_top500": {"mean": float(top_mu), "nw_se": float(top_se), "nw_t": float(top_t)},
        "delta_adv": {"mean": float(delta_mu), "nw_se": float(delta_se), "nw_t": float(delta_t)},
        "gross_positive_folds": int((portfolio.groupby("fold").gross.mean() > 0).sum()),
        "delta_adv_positive_folds": int((science.groupby("fold").delta_adv.mean() > 0).sum()),
        "eligibility_components": eligible,
        "eligible_for_second_slot": bool(all(eligible.values())),
    }


def paired_bootstrap(frame: pd.DataFrame, column: str) -> dict:
    groups = [group[column].to_numpy() for _, group in frame.groupby("fold")]
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draws[b] = np.concatenate([
            values[stationary_idx(len(values), rng)] for values in groups
        ]).mean()
    return {
        "mean": float(frame[column].mean()),
        "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "p_gt0": float(np.mean(draws > 0)),
    }


def main() -> None:
    print("读取冻结窗口价格与滞后 ADV...", flush=True)
    ret, oc, adv = load_market()
    money_parts, science_parts = [], []
    for arm in ARM_PATHS:
        print(f"构造 {arm} 的冻结钱管道与单标量 ΔADV...", flush=True)
        money_parts.append(portfolio_arm(arm, ret, oc, adv))
        science_parts.append(scientific_arm(arm, adv))
    money = pd.concat(money_parts, ignore_index=True)
    science = pd.concat(science_parts, ignore_index=True)
    by_arm = {
        arm: summarize_arm(money[money.arm == arm], science[science.arm == arm])
        for arm in ARM_PATHS
    }

    money_pair = (
        money.pivot(index=["fold", "date"], columns="arm", values="gross")
        .dropna().reset_index()
    )
    money_pair["zs_minus_ft"] = money_pair.ZS - money_pair.FT
    science_pair = (
        science.pivot(index=["fold", "date"], columns="arm", values="delta_adv")
        .dropna().reset_index()
    )
    science_pair["zs_minus_ft"] = science_pair.ZS - science_pair.FT
    gross_difference = paired_bootstrap(money_pair, "zs_minus_ft")
    gross_difference = {
        "mean": gross_difference["mean"] * 1e4,
        "ci95": [value * 1e4 for value in gross_difference["ci95"]],
        "p_gt0": gross_difference["p_gt0"],
    }
    dominated = (
        by_arm["ZS"]["gross_annual_arithmetic_pct"] <= by_arm["FT"]["gross_annual_arithmetic_pct"]
        and by_arm["ZS"]["delta_adv"]["mean"] <= by_arm["FT"]["delta_adv"]["mean"]
    )
    result = {
        "design": {
            "folds": FOLDS,
            "candidate_thresholds": {
                "gross_annual_pct": 5.0,
                "breakeven_one_way_bp": 11.0,
                "delta_adv": 0.0026,
                "ic_top500_sign": "positive",
            },
            "cost_grid_bp_one_way": list(COST_GRID_BP),
        },
        "by_arm": by_arm,
        "paired": {
            "n_money_days": int(len(money_pair)),
            "gross_return_correlation": float(money_pair[["FT", "ZS"]].corr().iloc[0, 1]),
            "gross_zs_minus_ft_daily_bp": gross_difference,
            "n_science_days": int(len(science_pair)),
            "delta_adv_correlation": float(science_pair[["FT", "ZS"]].corr().iloc[0, 1]),
            "delta_adv_zs_minus_ft": paired_bootstrap(science_pair, "zs_minus_ft"),
        },
        "zs_weakly_dominated_on_both_primary_effects": bool(dominated),
        "decision": (
            "ZS不应占第二确认配置"
            if (not by_arm["ZS"]["eligible_for_second_slot"] or dominated)
            else "ZS有资格进入k=2功效复算，尚未自动入选"
        ),
    }
    money.merge(science, on=["date", "fold", "arm"], how="outer").to_parquet(
        OUT / "k11_zs_ft_candidate_daily.parquet", index=False
    )
    (OUT / "k11_zs_ft_candidate_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nK11 候选资格审计", flush=True)
    for arm, row in by_arm.items():
        print(
            f"  {arm}: gross={row['gross_annual_arithmetic_pct']:+.2f}%  "
            f"BE={row['breakeven_one_way_bp']:.2f}bp  "
            f"top500 IC={row['ic_top500']['mean']:+.5f}  "
            f"ΔADV={row['delta_adv']['mean']:+.5f}  "
            f"eligible={row['eligible_for_second_slot']}",
            flush=True,
        )
    print(
        f"  gross corr={result['paired']['gross_return_correlation']:+.3f}; "
        f"{result['decision']}", flush=True
    )


if __name__ == "__main__":
    main()
