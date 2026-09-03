"""K9: ADV exposure--alpha--cost frontier.

The design and interpretation rules are pre-registered in experiments/ledger.md
before this script is run.  This script never reads future-return labels.  It
uses only frozen lb90 scores, lagged ADV, and next-day realized returns.

Two-stage use
-------------
1. ``--power-only`` prints only the already-known base gross mean, the NW(5)
   standard error of Z70 = gross(q=.50) - .70*gross(q=1), and MDE80.  It hides
   the constrained portfolio's mean, sign, fold results, and confidence interval.
2. The full run is allowed only after that MDE is recorded in the ledger.

Portfolio construction
----------------------
For each staggered sleeve, w0 is the frozen equal-weight top-decile portfolio.
For q in {1,.75,.50,.25,0}, solve independently

    min ||w-w0||^2
    s.t. sum(w)=1, 0<=w_i<=1/k, w'x=q*(w0'x),

where x is centered percentile rank of lagged ADV within the day's top500.
The box-constrained equality QP is solved through its two KKT multipliers; no
return or label enters the solver.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
PANEL = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")

FOLDS = [f"fold{i}" for i in range(36, 43)]
LO, HI = "2020-06-01", "2024-01-05"
TOPN, NT, EXIT_PCT = 500, 6, 0.30
Q_LEVELS = (1.0, 0.75, 0.50, 0.25, 0.0)
COST_GRID_BP = (2, 4, 8, 12, 16, 22)
NW_LAG = 5
N_BOOT, MEAN_BLOCK, BOOT_SEED = 10_000, 10, 20260831
EPS = 1e-12


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


def log(msg: str) -> None:
    print(msg, flush=True)


def nw_mean(x: np.ndarray, lags: int = NW_LAG) -> tuple[float, float, float]:
    """Newey-West mean, standard error, and t statistic."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mu = float(x.mean())
    e = x - mu
    long_var_sum = float(e @ e)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_var_sum += 2.0 * weight * float(e[lag:] @ e[:-lag])
    se = math.sqrt(max(long_var_sum, 0.0)) / n
    return mu, se, (mu / se if se > 0 else math.nan)


def stationary_idx(n: int, rng: np.random.Generator) -> np.ndarray:
    """Circular stationary-bootstrap indices with mean block length 10."""
    restart = 1.0 / MEAN_BLOCK
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(n))
    for j in range(n):
        idx[j] = cur
        cur = int(rng.integers(n)) if rng.random() < restart else (cur + 1) % n
    return idx


def _weights_at_beta(w0: np.ndarray, x: np.ndarray, cap: float,
                     beta: float) -> np.ndarray:
    """KKT weights for a fixed exposure multiplier beta; solve budget dual."""
    z = w0 - beta * x
    lo = float(z.min() - cap)
    hi = float(z.max())
    for _ in range(70):
        alpha = (lo + hi) / 2.0
        total = float(np.clip(z - alpha, 0.0, cap).sum())
        if total > 1.0:
            lo = alpha
        else:
            hi = alpha
    w = np.clip(z - (lo + hi) / 2.0, 0.0, cap)
    w /= w.sum()
    return w


def project_weights(w0: np.ndarray, x: np.ndarray, target: float,
                    cap: float) -> np.ndarray:
    """Minimum-L2 projection onto budget, box, and exact exposure constraints."""
    if abs(float(x @ w0) - target) <= 2e-13:
        return w0.copy()

    span = 1e-4
    for _ in range(40):
        w_lo = _weights_at_beta(w0, x, cap, -span)
        w_hi = _weights_at_beta(w0, x, cap, +span)
        e_lo, e_hi = float(x @ w_lo), float(x @ w_hi)
        if e_lo + 1e-12 >= target >= e_hi - 1e-12:
            break
        span *= 2.0
    else:
        raise RuntimeError(f"Could not bracket target exposure {target:+.8f}")

    lo, hi = -span, span
    for _ in range(80):
        mid = (lo + hi) / 2.0
        w = _weights_at_beta(w0, x, cap, mid)
        exposure = float(x @ w)
        if exposure > target:
            lo = mid
        else:
            hi = mid
    w = _weights_at_beta(w0, x, cap, (lo + hi) / 2.0)
    if abs(w.sum() - 1.0) > 2e-10 or abs(float(x @ w) - target) > 2e-9:
        raise RuntimeError("Exposure projection failed numerical checks")
    if w.min() < -2e-12 or w.max() > cap + 2e-10:
        raise RuntimeError("Exposure projection violated box constraints")
    return w


def load_market() -> tuple[dict, dict, dict]:
    """Column-pruned and date-filtered price load; ADV is strictly shifted one day."""
    df = pd.read_parquet(
        PANEL / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(LO)),
                 ("DlyCalDt", "<=", pd.Timestamp(HI))],
    )
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df.dropna(subset=["DlyRet"]).sort_values(["PERMNO", "DlyCalDt"])
    df["oc"] = np.where(
        df["DlyOpen"].abs() > 0,
        df["DlyClose"] / df["DlyOpen"].abs() - 1.0,
        np.nan,
    )
    df["adv20"] = df.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1)
    )
    ret = {d: dict(zip(g.PERMNO, g.DlyRet)) for d, g in df.groupby("DlyCalDt")}
    oc = {d: dict(zip(g.PERMNO, g.oc)) for d, g in df.groupby("DlyCalDt")}
    adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in df.groupby("DlyCalDt")}
    del df
    return ret, oc, adv


def weighted_return(weights: dict[int, float], returns: dict[int, float]) -> float:
    num = den = 0.0
    for permno, weight in weights.items():
        value = returns.get(permno)
        if value is not None and np.isfinite(value):
            num += weight * value
            den += weight
    return num / den if den > 0 else math.nan


def rebalanced_return(old: dict[int, float], new: dict[int, float],
                      close_to_close: dict[int, float],
                      open_to_close: dict[int, float]) -> float:
    """Generalize the frozen fresh-name convention to fractional reweighting.

    Weight already present in the sleeve receives close-to-close return; weight
    added at next-day open receives open-to-close return.  This reduces exactly
    to compare_arms_money.py for q=1 equal-weight books.
    """
    num = den = 0.0
    for permno, new_weight in new.items():
        retained = min(old.get(permno, 0.0), new_weight)
        added = new_weight - retained
        full_ret = close_to_close.get(permno)
        if retained > EPS and full_ret is not None and np.isfinite(full_ret):
            num += retained * full_ret
            den += retained
        intraday_ret = open_to_close.get(permno)
        if added > EPS and intraday_ret is not None and np.isfinite(intraday_ret):
            num += added * intraday_ret
            den += added
    return num / den if den > 0 else math.nan


def weight_turnover(old: dict[int, float] | None,
                    new: dict[int, float]) -> tuple[float, dict[int, float]]:
    if old is None:
        return 0.0, {}
    names = set(old) | set(new)
    delta = {p: abs(new.get(p, 0.0) - old.get(p, 0.0)) for p in names}
    delta = {p: v for p, v in delta.items() if v > EPS}
    return float(sum(delta.values())) / NT, delta


def adv_quintile(permno: int, adv_today: dict[int, float],
                 sorted_pool_adv: np.ndarray) -> int:
    value = adv_today.get(permno)
    if value is None or not np.isfinite(value):
        return 5  # separate missing bucket after Q1..Q5
    pct = np.searchsorted(sorted_pool_adv, value, side="right") / len(sorted_pool_adv)
    return min(4, max(0, int(math.ceil(5.0 * pct) - 1)))


def run_frontier() -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    log("读取现代窗口价格与 shift(1) ADV（列裁剪、日期下推）...")
    ret, oc, adv = load_market()
    rows: list[tuple] = []
    trade_bins = {q: np.zeros(6, dtype=float) for q in Q_LEVELS}

    for fold in FOLDS:
        score_dir = OUT / f"{fold}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{fold}"
        scores = pd.read_parquet(
            _guard(score_dir) / "scores.parquet", columns=["PERMNO", "signal_date", "score"]
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
            raw_scores = by_day[day]
            adv_today = adv.get(day, {})
            eligible = [
                p for p in raw_scores
                if p in adv_today and np.isfinite(adv_today[p])
            ]
            if len(eligible) > TOPN:
                eligible = sorted(eligible, key=lambda p: -adv_today[p])[:TOPN]
            eligible_set = set(eligible)
            current_scores = {p: v for p, v in raw_scores.items() if p in eligible_set}
            n = len(current_scores)
            if n < 50:
                continue

            score_pct = (pd.Series(current_scores).rank() / n).to_dict()
            score_order = sorted(score_pct, key=lambda p: -score_pct[p])
            k = max(1, n // 10)
            sleeve = day_index % NT
            previous_selection = selection_book[sleeve]
            if previous_selection is None:
                selected = list(score_order[:k])
            else:
                kept = [
                    p for p in previous_selection
                    if p in score_pct and score_pct[p] >= 1.0 - EXIT_PCT
                ][:k]
                kept_set = set(kept)
                added = [p for p in score_order if p not in kept_set][:k - len(kept)]
                selected = kept + added
            selection_book[sleeve] = selected

            names = list(current_scores)
            loc = {p: j for j, p in enumerate(names)}
            adv_values = np.array([adv_today[p] for p in names], dtype=float)
            adv_rank = pd.Series(adv_values).rank(method="average", pct=True).to_numpy()
            x = adv_rank - adv_rank.mean()
            w0 = np.zeros(n, dtype=float)
            for p in selected:
                w0[loc[p]] = 1.0 / k
            base_exposure = float(x @ w0)
            cap = 1.0 / k
            sorted_pool_adv = np.sort(adv_values)

            new_books: dict[float, dict[int, float]] = {}
            day_turnover: dict[float, float] = {}
            day_exposure: dict[float, float] = {}
            day_eff_n: dict[float, float] = {}
            day_npos: dict[float, int] = {}
            for q in Q_LEVELS:
                weights = (w0.copy() if q == 1.0 else
                           project_weights(w0, x, q * base_exposure, cap))
                new = {p: float(weights[j]) for j, p in enumerate(names) if weights[j] > EPS}
                old = path_books[q][sleeve]
                turnover, deltas = weight_turnover(old, new)
                day_turnover[q] = turnover
                for p, amount in deltas.items():
                    trade_bins[q][adv_quintile(p, adv_today, sorted_pool_adv)] += amount / NT
                new_books[q] = new
                day_exposure[q] = float(x @ weights)
                day_eff_n[q] = float(1.0 / np.sum(weights ** 2))
                day_npos[q] = int(np.sum(weights > EPS))

            next_day = days[day_index + 1] if day_index + 1 < len(days) else None
            can_score = next_day is not None and day_index >= NT and bool(ret.get(next_day, {}))
            if can_score:
                full_ret = ret[next_day]
                intraday_ret = oc.get(next_day, {})
                benchmark_values = [
                    full_ret[p] for p in score_pct
                    if p in full_ret and np.isfinite(full_ret[p])
                ]
                benchmark = float(np.mean(benchmark_values)) if benchmark_values else math.nan
                for q in Q_LEVELS:
                    sleeve_returns = []
                    for slot in range(NT):
                        if slot == sleeve:
                            old = path_books[q][slot]
                            if old is None:
                                value = math.nan
                            else:
                                value = rebalanced_return(
                                    old, new_books[q], full_ret, intraday_ret
                                )
                        else:
                            old = path_books[q][slot]
                            value = weighted_return(old, full_ret) if old is not None else math.nan
                        if np.isfinite(value):
                            sleeve_returns.append(value)
                    if sleeve_returns and np.isfinite(benchmark):
                        gross = float(np.mean(sleeve_returns) - benchmark)
                        target_error = day_exposure[q] - q * base_exposure
                        rows.append((
                            next_day, fold, q, gross, day_turnover[q],
                            base_exposure, day_exposure[q], target_error,
                            day_eff_n[q], day_npos[q],
                        ))
                fold_rows += 1

            for q in Q_LEVELS:
                path_books[q][sleeve] = new_books[q]

        log(f"  {fold}: {fold_rows} 个可评估交易日")

    daily = pd.DataFrame(rows, columns=[
        "date", "fold", "q", "gross", "turnover", "base_exposure",
        "exposure", "target_error", "effective_n", "n_positions",
    ]).sort_values(["date", "q"]).reset_index(drop=True)
    return daily, trade_bins


def primary_frame(daily: pd.DataFrame) -> pd.DataFrame:
    gross = daily.pivot(index=["fold", "date"], columns="q", values="gross")
    required = [1.0, 0.5]
    if any(q not in gross.columns for q in required):
        raise RuntimeError("Missing q=1 or q=.5 daily returns")
    out = gross[required].rename(columns={1.0: "g100", 0.5: "g50"}).dropna()
    out["z70"] = out.g50 - 0.70 * out.g100
    return out.reset_index().sort_values("date")


def bootstrap_primary(primary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOT_SEED)
    groups = [
        g[["g50", "g100", "z70"]].to_numpy()
        for _, g in primary.groupby("fold", sort=True)
    ]
    draw_z = np.empty(N_BOOT, dtype=float)
    draw_ratio = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        sampled = []
        for values in groups:
            sampled.append(values[stationary_idx(len(values), rng)])
        sample = np.concatenate(sampled, axis=0)
        draw_z[b] = float(sample[:, 2].mean())
        denominator = float(sample[:, 1].mean())
        draw_ratio[b] = (float(sample[:, 0].mean()) / denominator
                         if abs(denominator) > EPS else math.nan)
    return draw_z, draw_ratio[np.isfinite(draw_ratio)]


def geometric_annual(daily_return: np.ndarray) -> float:
    values = np.asarray(daily_return, dtype=float)
    return float(np.prod(1.0 + values) ** (252.0 / len(values)) - 1.0)


def summarize(daily: pd.DataFrame, trade_bins: dict[float, np.ndarray]) -> dict:
    result: dict = {
        "design": {
            "folds": FOLDS,
            "q_levels": list(Q_LEVELS),
            "cost_grid_bp_one_way": list(COST_GRID_BP),
            "nw_lag": NW_LAG,
            "bootstrap_reps": N_BOOT,
            "bootstrap_mean_block": MEAN_BLOCK,
        },
        "frontier": {},
    }
    means: dict[float, tuple[float, float]] = {}
    for q in Q_LEVELS:
        part = daily[daily.q == q].sort_values("date")
        gross_mean = float(part.gross.mean())
        turn_mean = float(part.turnover.mean())
        means[q] = (gross_mean, turn_mean)
        costs = {}
        for cost_bp in COST_GRID_BP:
            net = part.gross.to_numpy() - (cost_bp / 1e4) * part.turnover.to_numpy()
            costs[str(cost_bp)] = {
                "annual_arithmetic_pct": float(net.mean() * 252 * 100),
                "annual_geometric_pct": geometric_annual(net) * 100,
            }
        bins = trade_bins[q]
        bin_total = float(bins.sum())
        result["frontier"][str(q)] = {
            "n_days": int(len(part)),
            "gross_mean_daily_bp": gross_mean * 1e4,
            "gross_annual_arithmetic_pct": gross_mean * 252 * 100,
            "gross_annual_geometric_pct": geometric_annual(part.gross.to_numpy()) * 100,
            "gross_retention_vs_base": None,
            "mean_one_way_traded_fraction": turn_mean,
            "uniform_cost_breakeven_bp": (
                gross_mean / turn_mean * 1e4 if turn_mean > EPS else math.nan
            ),
            "mean_exposure_ratio": float(np.mean(
                part.exposure.to_numpy() / part.base_exposure.to_numpy()
            )),
            "max_abs_target_error": float(part.target_error.abs().max()),
            "mean_effective_n": float(part.effective_n.mean()),
            "mean_positive_positions": float(part.n_positions.mean()),
            "trade_adv_quintile_share": {
                f"Q{i + 1}": (float(bins[i] / bin_total) if bin_total > 0 else math.nan)
                for i in range(5)
            },
            "trade_adv_missing_share": (
                float(bins[5] / bin_total) if bin_total > 0 else math.nan
            ),
            "net_by_uniform_cost": costs,
        }

    base_gross, base_turn = means[1.0]
    for q in Q_LEVELS:
        result["frontier"][str(q)]["gross_retention_vs_base"] = (
            means[q][0] / base_gross if abs(base_gross) > EPS else math.nan
        )

    boundaries = {}
    for q in Q_LEVELS:
        if q == 1.0:
            continue
        gross_q, turn_q = means[q]
        boundaries[str(q)] = {
            str(base_cost): (
                (gross_q - base_gross + (base_cost / 1e4) * base_turn)
                / turn_q * 1e4
                if turn_q > EPS else math.nan
            )
            for base_cost in COST_GRID_BP
        }
    result["indifference_cost_bp_for_constrained_portfolio"] = boundaries

    primary = primary_frame(daily)
    z_mean, z_se, z_t = nw_mean(primary.z70.to_numpy())
    mde80 = (1.959963984540054 + 0.8416212335729143) * z_se
    draws_z, draws_ratio = bootstrap_primary(primary)
    z_ci = np.percentile(draws_z, [2.5, 97.5])
    ratio = float(primary.g50.mean() / primary.g100.mean())
    ratio_ci = np.percentile(draws_ratio, [2.5, 97.5])
    fold_z = primary.groupby("fold").z70.mean()
    folds_pos = int((fold_z > 0).sum())
    folds_neg = int((fold_z < 0).sum())
    margin = 0.30 * float(primary.g100.mean())
    power_pass = bool(mde80 <= margin)
    if z_ci[0] > 0 and folds_pos >= 5:
        verdict = "半数 ADV 暴露可删除，仍保留至少 70% 毛 alpha"
    elif z_ci[1] < 0 and folds_neg >= 5:
        verdict = "达不到 70% 保留线，ADV 暴露是重要载体"
    else:
        verdict = "不可判读"
    if not power_pass:
        verdict += "（MDE>SESOI，按预注册仅作探索性诊断）"

    result["primary"] = {
        "n_days": int(len(primary)),
        "gross_retention_r50": ratio,
        "gross_retention_r50_bootstrap_ci95": [float(x) for x in ratio_ci],
        "z70_mean_daily_bp": z_mean * 1e4,
        "z70_nw_se_daily_bp": z_se * 1e4,
        "z70_nw_t": z_t,
        "z70_bootstrap_ci95_daily_bp": [float(x * 1e4) for x in z_ci],
        "mde80_daily_bp": mde80 * 1e4,
        "sesoi_budget_daily_bp": margin * 1e4,
        "power_gate_pass": power_pass,
        "folds_z70_positive": folds_pos,
        "folds_z70_negative": folds_neg,
        "per_fold_z70_daily_bp": {k: float(v * 1e4) for k, v in fold_z.items()},
        "verdict": verdict,
    }
    return result


def audit_base(summary: dict) -> dict:
    """q=1 must reproduce the existing frozen lb90 money pipeline at 8bp."""
    old_path = OUT / "compare_arms_money.json"
    if not old_path.exists():
        return {"checked": False, "reason": "compare_arms_money.json missing"}
    old = json.loads(old_path.read_text(encoding="utf-8"))
    expected = float(old["lb90"]["ann"] * 100)
    actual = float(summary["frontier"]["1.0"]["net_by_uniform_cost"]["8"]
                   ["annual_geometric_pct"])
    difference = actual - expected
    if abs(difference) > 1e-8:
        raise RuntimeError(
            f"q=1 baseline failed frozen-pipeline audit: {actual} vs {expected} pct"
        )
    return {
        "checked": True,
        "existing_net8_annual_geometric_pct": expected,
        "k9_net8_annual_geometric_pct": actual,
        "difference_pct_point": difference,
    }


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
            "alpha_two_sided": 0.05,
            "target_power": 0.80,
        }
        (OUT / "k9_power_gate.json").write_text(
            json.dumps(power, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"盲均值功效关：{len(primary)} 天")
        log(f"  已知基座毛均值       {base_mean * 1e4:.3f} bp/日")
        log(f"  Z70 NW(5) SE        {z_se * 1e4:.3f} bp/日")
        log(f"  MDE80               {mde80 * 1e4:.3f} bp/日")
        log(f"  SESOI(基座毛的30%)  {margin * 1e4:.3f} bp/日")
        log(f"  功效关               {'通过' if mde80 <= margin else '不通过'}")
        log("  已隐藏：g50、Z70 均值/方向、折读数、CI、成本前沿")
        return

    summary = summarize(daily, trade_bins)
    summary["base_pipeline_audit"] = audit_base(summary)
    daily.to_parquet(OUT / "k9_cost_alpha_daily.parquet", index=False)
    (OUT / "k9_cost_alpha_frontier.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    log("\nK9 ADV 暴露—alpha—成本前沿")
    log("q     毛年化%   保留率   日均成交%   BE成本bp   有效持仓数")
    for q in Q_LEVELS:
        row = summary["frontier"][str(q)]
        log(
            f"{q:>4.2f}  {row['gross_annual_arithmetic_pct']:>+8.2f}  "
            f"{row['gross_retention_vs_base']:>7.1%}  "
            f"{row['mean_one_way_traded_fraction'] * 100:>9.2f}  "
            f"{row['uniform_cost_breakeven_bp']:>9.2f}  "
            f"{row['mean_effective_n']:>10.1f}"
        )
    p = summary["primary"]
    log("\n主终点：半暴露的 70% 毛收益非劣性")
    log(
        f"  R50={p['gross_retention_r50']:.3f}  "
        f"bootstrap 95%CI [{p['gross_retention_r50_bootstrap_ci95'][0]:.3f}, "
        f"{p['gross_retention_r50_bootstrap_ci95'][1]:.3f}]"
    )
    log(
        f"  Z70={p['z70_mean_daily_bp']:+.3f} bp/日  "
        f"NW t={p['z70_nw_t']:+.2f}  "
        f"CI [{p['z70_bootstrap_ci95_daily_bp'][0]:+.3f}, "
        f"{p['z70_bootstrap_ci95_daily_bp'][1]:+.3f}]"
    )
    log(
        f"  折同向 {p['folds_z70_positive']}/7  MDE80 {p['mde80_daily_bp']:.3f} "
        f"vs SESOI {p['sesoi_budget_daily_bp']:.3f} bp/日"
    )
    log(f"  **判读：{p['verdict']}**")
    log("\n写入 outputs/k9_cost_alpha_frontier.json 与 k9_cost_alpha_daily.parquet")


if __name__ == "__main__":
    main()
