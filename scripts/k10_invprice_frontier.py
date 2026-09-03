"""K10: long-only inverse-price exposure frontier plus K6b loading retest.

Design and interpretation were appended to experiments/ledger.md before run.
Only consumed folds 36--42 are used.  ``--power-only`` hides constrained means.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from k6b_spanning import ALL_FAC, J, build_factors, load_panel, nw_ols
from k9_cost_alpha_frontier import (
    COST_GRID_BP,
    EPS,
    FOLDS,
    NT,
    NW_LAG,
    Q_LEVELS,
    TOPN,
    EXIT_PCT,
    adv_quintile,
    audit_base,
    geometric_annual,
    nw_mean,
    primary_frame,
    project_weights,
    rebalanced_return,
    stationary_idx,
    summarize,
    weight_turnover,
    weighted_return,
)


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
N_BOOT, BOOT_SEED = 10_000, 20260901
KEY_FACTORS = ("invprice", "amihud", "idiovol", "hlrange")


def load_exact_panel() -> dict[pd.Timestamp, pd.DataFrame]:
    print("构造 K6b 同口径 invprice 与收益面板...", flush=True)
    panel = build_factors(load_panel())
    keep = ["PERMNO", "DlyCalDt", "DlyRet", "oc", "adv20", "invprice"]
    frames = {day: g[keep] for day, g in panel.groupby("DlyCalDt")}
    del panel
    return frames


def run_frontier() -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    frames = load_exact_panel()
    rows = []
    trade_bins = {q: np.zeros(6, dtype=float) for q in Q_LEVELS}

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
        selection_book: list[list[int] | None] = [None] * NT
        path_books: dict[float, list[dict[int, float] | None]] = {
            q: [None] * NT for q in Q_LEVELS
        }
        fold_rows = 0

        for day_index, day in enumerate(days):
            day_frame = frames.get(day)
            if day_frame is None:
                continue
            raw_scores = by_day[day]
            adv = dict(zip(day_frame.PERMNO, day_frame.adv20))
            invprice = dict(zip(day_frame.PERMNO, day_frame.invprice))
            eligible = [
                p for p in raw_scores
                if p in adv and np.isfinite(adv[p])
            ]
            if len(eligible) > TOPN:
                eligible = sorted(eligible, key=lambda p: -adv[p])[:TOPN]
            eligible = [p for p in eligible if p in invprice and np.isfinite(invprice[p])]
            eligible_set = set(eligible)
            current_scores = {p: v for p, v in raw_scores.items() if p in eligible_set}
            n = len(current_scores)
            if n < 50:
                continue

            score_pct = (pd.Series(current_scores).rank() / n).to_dict()
            order = sorted(score_pct, key=lambda p: -score_pct[p])
            k = max(1, n // 10)
            sleeve = day_index % NT
            previous_selection = selection_book[sleeve]
            if previous_selection is None:
                selected = list(order[:k])
            else:
                kept = [
                    p for p in previous_selection
                    if p in score_pct and score_pct[p] >= 1.0 - EXIT_PCT
                ][:k]
                kept_set = set(kept)
                added = [p for p in order if p not in kept_set][:k - len(kept)]
                selected = kept + added
            selection_book[sleeve] = selected

            names = list(current_scores)
            loc = {p: j for j, p in enumerate(names)}
            adv_values = np.array([adv[p] for p in names], dtype=float)
            feature_values = np.array([invprice[p] for p in names], dtype=float)
            feature_rank = pd.Series(feature_values).rank(method="average", pct=True).to_numpy()
            x = feature_rank - feature_rank.mean()
            w0 = np.zeros(n, dtype=float)
            for p in selected:
                w0[loc[p]] = 1.0 / k
            base_exposure = float(x @ w0)
            cap = 1.0 / k
            sorted_pool_adv = np.sort(adv_values)

            new_books = {}
            day_turnover = {}
            day_exposure = {}
            day_eff_n = {}
            day_npos = {}
            for q in Q_LEVELS:
                weights = (w0.copy() if q == 1.0 else
                           project_weights(w0, x, q * base_exposure, cap))
                new = {p: float(weights[j]) for j, p in enumerate(names) if weights[j] > EPS}
                old = path_books[q][sleeve]
                turnover, deltas = weight_turnover(old, new)
                day_turnover[q] = turnover
                for p, amount in deltas.items():
                    trade_bins[q][adv_quintile(p, adv, sorted_pool_adv)] += amount / NT
                new_books[q] = new
                day_exposure[q] = float(x @ weights)
                day_eff_n[q] = float(1.0 / np.sum(weights ** 2))
                day_npos[q] = int(np.sum(weights > EPS))

            next_day = days[day_index + 1] if day_index + 1 < len(days) else None
            next_frame = frames.get(next_day) if next_day is not None else None
            can_score = next_frame is not None and day_index >= NT
            if can_score:
                full_ret = dict(zip(next_frame.PERMNO, next_frame.DlyRet))
                intraday_ret = dict(zip(next_frame.PERMNO, next_frame.oc))
                benchmark_values = [
                    full_ret[p] for p in score_pct
                    if p in full_ret and np.isfinite(full_ret[p])
                ]
                benchmark = float(np.mean(benchmark_values)) if benchmark_values else math.nan
                for q in Q_LEVELS:
                    sleeve_returns = []
                    for slot in range(NT):
                        old = path_books[q][slot]
                        if slot == sleeve:
                            value = (rebalanced_return(old, new_books[q], full_ret, intraday_ret)
                                     if old is not None else math.nan)
                        else:
                            value = weighted_return(old, full_ret) if old is not None else math.nan
                        if np.isfinite(value):
                            sleeve_returns.append(value)
                    if sleeve_returns and np.isfinite(benchmark):
                        rows.append((
                            next_day, fold, q, float(np.mean(sleeve_returns) - benchmark),
                            day_turnover[q], base_exposure, day_exposure[q],
                            day_exposure[q] - q * base_exposure,
                            day_eff_n[q], day_npos[q],
                        ))
                fold_rows += 1
            for q in Q_LEVELS:
                path_books[q][sleeve] = new_books[q]
        print(f"  {fold}: {fold_rows} 天", flush=True)

    daily = pd.DataFrame(rows, columns=[
        "date", "fold", "q", "gross", "turnover", "base_exposure",
        "exposure", "target_error", "effective_n", "n_positions",
    ]).sort_values(["date", "q"]).reset_index(drop=True)
    return daily, trade_bins


def load_s2_factors() -> pd.DataFrame:
    controls = pd.read_parquet(OUT / "k6b_control_daily.parquet")
    market = pd.read_csv(J / "usa_mkt_daily_vw_cap.csv")
    market["date"] = pd.to_datetime(market["date"])
    market = market.set_index("date")
    market_col = [c for c in market.columns if c.lower() in ("ret", "mkt", "mktrf")][0]
    return pd.concat([controls[ALL_FAC], market[market_col].rename("market")], axis=1)


def fit_loadings(y: pd.Series, factors: pd.DataFrame) -> dict:
    joined = pd.concat([y.rename("y"), factors], axis=1).dropna()
    names = list(joined.columns[1:])
    values = joined.y.to_numpy()
    X = np.column_stack([np.ones(len(joined)), joined[names].to_numpy()])
    beta, t_stat = nw_ols(values, X, NW_LAG)
    return {
        "n_days": int(len(joined)),
        "alpha_ann_pct": float(beta[0] * 252 * 100),
        "t_alpha": float(t_stat[0]),
        "betas": {
            name: {"b": float(beta[j + 1]), "t": float(t_stat[j + 1])}
            for j, name in enumerate(names)
        },
        "correlations": {
            name: float(np.corrcoef(values, joined[name].to_numpy())[0, 1])
            for name in KEY_FACTORS
        },
    }


def bootstrap_invprice_beta(daily: pd.DataFrame, factors: pd.DataFrame) -> dict:
    gross = daily.pivot(index=["fold", "date"], columns="q", values="gross")
    turn = daily.pivot(index=["fold", "date"], columns="q", values="turnover")
    y100 = (gross[1.0] - 8e-4 * turn[1.0]).rename("y100")
    y50 = (gross[0.5] - 8e-4 * turn[0.5]).rename("y50")
    frame = pd.concat([
        y100.reset_index().set_index("date"),
        y50.reset_index().set_index("date")[["y50"]],
        factors,
    ], axis=1).dropna()
    factor_names = list(factors.columns)
    inv_col = factor_names.index("invprice") + 1
    groups = []
    for _, group in frame.groupby("fold"):
        X = np.column_stack([np.ones(len(group)), group[factor_names].to_numpy()])
        Y = group[["y100", "y50"]].to_numpy()
        groups.append((X, Y))

    rng = np.random.default_rng(BOOT_SEED)
    beta100 = np.empty(N_BOOT)
    beta50 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        xs, ys = [], []
        for X, Y in groups:
            idx = stationary_idx(len(X), rng)
            xs.append(X[idx])
            ys.append(Y[idx])
        Xb, Yb = np.concatenate(xs), np.concatenate(ys)
        coefficients = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ Yb)
        beta100[b], beta50[b] = coefficients[inv_col, 0], coefficients[inv_col, 1]
    difference = beta50 - beta100
    return {
        "beta100_ci95": [float(x) for x in np.percentile(beta100, [2.5, 97.5])],
        "beta50_ci95": [float(x) for x in np.percentile(beta50, [2.5, 97.5])],
        "difference_beta50_minus_beta100_ci95": [
            float(x) for x in np.percentile(difference, [2.5, 97.5])
        ],
        "p_beta50_less_than_beta100": float(np.mean(difference < 0)),
    }


def full_summary(daily: pd.DataFrame, trade_bins: dict[float, np.ndarray]) -> dict:
    result = summarize(daily, trade_bins)
    result["exposure_feature"] = "invprice exact K6b definition"
    factors = load_s2_factors()
    loadings = {}
    for q in Q_LEVELS:
        part = daily[daily.q == q].set_index("date").sort_index()
        net8 = part.gross - 8e-4 * part.turnover
        loadings[str(q)] = fit_loadings(net8, factors)
    beta_base = loadings["1.0"]["betas"]["invprice"]["b"]
    beta_half = loadings["0.5"]["betas"]["invprice"]["b"]
    loading_bootstrap = bootstrap_invprice_beta(daily, factors)
    beta_retention = abs(beta_half) / abs(beta_base) if abs(beta_base) > EPS else math.nan
    if beta_retention <= 0.60 and loading_bootstrap["difference_beta50_minus_beta100_ci95"][1] < 0:
        interpretation = "持仓约束能明显传导到收益载荷"
    elif beta_retention >= 0.80:
        interpretation = "持仓暴露减半，但收益载荷基本未消除"
    else:
        interpretation = "收益载荷仅部分下降或不可判读"
    result["k6b_s2_loading_retest"] = {
        "by_q": loadings,
        "invprice_beta_abs_retention_q50": beta_retention,
        "bootstrap": loading_bootstrap,
        "interpretation": interpretation,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-only", action="store_true")
    args = parser.parse_args()
    daily, trade_bins = run_frontier()
    primary = primary_frame(daily)
    _, z_se, _ = nw_mean(primary.z70.to_numpy())
    mde80 = (1.959963984540054 + 0.8416212335729143) * z_se
    base_mean = float(primary.g100.mean())
    margin = 0.30 * base_mean
    if args.power_only:
        power = {
            "n_days": int(len(primary)),
            "base_gross_mean_daily_bp_already_known": base_mean * 1e4,
            "z70_nw5_se_daily_bp_blinded_mean": z_se * 1e4,
            "mde80_daily_bp": mde80 * 1e4,
            "sesoi_30pct_base_daily_bp": margin * 1e4,
            "mde_le_sesoi": bool(mde80 <= margin),
        }
        (OUT / "k10_invprice_power_gate.json").write_text(
            json.dumps(power, indent=2), encoding="utf-8"
        )
        print(f"盲功效关 n={len(primary)}", flush=True)
        print(f"  基座毛均值 {base_mean*1e4:.3f} bp/日", flush=True)
        print(f"  Z70 NW5 SE {z_se*1e4:.3f} bp/日", flush=True)
        print(f"  MDE80 {mde80*1e4:.3f} vs SESOI {margin*1e4:.3f} bp/日", flush=True)
        print(f"  功效关 {'通过' if mde80 <= margin else '不通过'}；约束收益方向保持隐藏", flush=True)
        return

    result = full_summary(daily, trade_bins)
    result["base_pipeline_audit"] = audit_base(result)
    daily.to_parquet(OUT / "k10_invprice_daily.parquet", index=False)
    (OUT / "k10_invprice_frontier.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print("\nK10 1/价格纯多前沿", flush=True)
    print("q     毛年化%  保留率  日均成交%  invprice beta(t)", flush=True)
    for q in Q_LEVELS:
        f = result["frontier"][str(q)]
        b = result["k6b_s2_loading_retest"]["by_q"][str(q)]["betas"]["invprice"]
        print(
            f"{q:>4.2f} {f['gross_annual_arithmetic_pct']:>+8.2f} "
            f"{f['gross_retention_vs_base']:>7.1%} "
            f"{f['mean_one_way_traded_fraction']*100:>9.2f} "
            f"{b['b']:>+9.4f}({b['t']:+.2f})",
            flush=True,
        )
    p = result["primary"]
    l = result["k6b_s2_loading_retest"]
    print(
        f"主终点 R50={p['gross_retention_r50']:.3f}, Z70={p['z70_mean_daily_bp']:+.3f}bp/日, "
        f"CI={p['z70_bootstrap_ci95_daily_bp']}, folds+={p['folds_z70_positive']}/7",
        flush=True,
    )
    print(
        f"invprice beta 绝对保留率={l['invprice_beta_abs_retention_q50']:.3f}；"
        f"{l['interpretation']}", flush=True
    )
    print("写入 outputs/k10_invprice_frontier.json 与 daily.parquet", flush=True)


if __name__ == "__main__":
    main()
