"""K14: two-arm collision structure and pilot day-cluster sizing.

Why
---
Alpaca rejects simultaneously-open opposite-side market orders on the same
symbol (HTTP 403; paper accounts enforce it too).  K13 found 2,270 same-day
same-name opposite-side collisions between FT and ZS, so "submit both arms in
parallel from one account" is not an available design.  This script prices the
alternatives and sizes the pilot in trading days.

Inputs: outputs/k13_orders_{FT,ZS}.parquet only (already-consumed folds 36--42).
No return, no label, no P&L.

Pre-registered scope (written before the run)
---------------------------------------------
MAY be used to choose: the collision-handling scheme, the pilot's duration in
trading days, and the per-arm fill target.
MAY NOT be used to choose holding period, exit band, universe, or price floor.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from k9_cost_alpha_frontier import OUT

SE_TARGETS_BP = (0.5, 1.0, 2.0)
SIGMA_FILL_BP = (3.0, 5.0, 10.0, 20.0)
RHO_WITHIN_DAY = (0.0, 0.2, 0.5)
MAX_DAYS = 250


def log(m: str) -> None:
    print(m, flush=True)


def describe(d: pd.DataFrame, mask: np.ndarray, label: str) -> dict:
    sub = d[mask]
    if not len(sub):
        return {"label": label, "n": 0}
    return {
        "label": label, "n": int(len(sub)),
        "px_p25": float(np.nanpercentile(sub.close_px, 25)),
        "px_p50": float(np.nanpercentile(sub.close_px, 50)),
        "px_p75": float(np.nanpercentile(sub.close_px, 75)),
        "adv_pct_p50": float(np.nanpercentile(sub.adv_pct_in_pool, 50)),
        "adv_pct_p10": float(np.nanpercentile(sub.adv_pct_in_pool, 10)),
        "inv_px_mean": float(np.nanmean(1.0 / sub.close_px)),
    }


def days_needed(sigma_bp: float, rho: float, n_per_day: float,
                target_bp: float) -> int | None:
    """Smallest D with SE = sigma*sqrt((1+(n-1)rho)/(D*n)) <= target."""
    inflate = 1.0 + (n_per_day - 1.0) * rho
    for D in range(1, MAX_DAYS + 1):
        if sigma_bp * np.sqrt(inflate / (D * n_per_day)) <= target_bp:
            return D
    return None


def main() -> None:
    ft = pd.read_parquet(OUT / "k13_orders_FT.parquet")
    zs = pd.read_parquet(OUT / "k13_orders_ZS.parquet")
    for d in (ft, zs):
        d["key"] = list(zip(d.trade_date, d.PERMNO))

    log("=== 1. collision structure (FT vs ZS, same day + same name) ===")
    ft_side = dict(zip(ft.key, ft.side))
    zs_side = dict(zip(zs.key, zs.side))
    both = set(ft_side) & set(zs_side)
    opposite = {k for k in both if ft_side[k] != zs_side[k]}
    same = both - opposite

    ft["collide"] = ft.key.isin(opposite).to_numpy()
    zs["collide"] = zs.key.isin(opposite).to_numpy()

    per_day = pd.Series([k[0] for k in opposite]).value_counts()
    all_days = sorted(set(ft.trade_date) | set(zs.trade_date))
    coll_days = per_day.reindex(all_days).fillna(0.0)

    collisions = {
        "n_opposite_side": len(opposite),
        "n_same_side": len(same),
        "ft_fills": int(len(ft)), "zs_fills": int(len(zs)),
        "share_of_ft_fills": float(ft.collide.mean()),
        "share_of_zs_fills": float(zs.collide.mean()),
        "share_of_ft_notional": float(ft.loc[ft.collide, "w_frac"].sum() / ft.w_frac.sum()),
        "share_of_zs_notional": float(zs.loc[zs.collide, "w_frac"].sum() / zs.w_frac.sum()),
        "n_trading_days": len(all_days),
        "days_with_at_least_one": int((coll_days > 0).sum()),
        "share_of_days_affected": float((coll_days > 0).mean()),
        "per_day_mean": float(coll_days.mean()),
        "per_day_p50": float(coll_days.median()),
        "per_day_p90": float(coll_days.quantile(0.90)),
        "per_day_max": float(coll_days.max()),
        "profile_ft_colliding": describe(ft, ft.collide.to_numpy(), "FT colliding"),
        "profile_ft_clean": describe(ft, ~ft.collide.to_numpy(), "FT non-colliding"),
        "profile_zs_colliding": describe(zs, zs.collide.to_numpy(), "ZS colliding"),
        "profile_zs_clean": describe(zs, ~zs.collide.to_numpy(), "ZS non-colliding"),
    }
    for k, v in collisions.items():
        if not isinstance(v, dict):
            log(f"  {k}: {v}")
    log("  colliding-vs-clean profile (is dropping them a biased sample?):")
    for key in ("profile_ft_colliding", "profile_ft_clean",
                "profile_zs_colliding", "profile_zs_clean"):
        p = collisions[key]
        log(f"    {p['label']:<20} n={p['n']:>6}  px p50=${p['px_p50']:>7.1f}  "
            f"advpct p50={p['adv_pct_p50']:.3f} p10={p['adv_pct_p10']:.3f}  "
            f"mean(1/px)={p['inv_px_mean']:.5f}")

    log("\n=== 2. candidate schemes ===")
    # (iii) drop: parallel submission, skip colliding fills
    ft_clean_per_day = ft[~ft.collide].groupby("trade_date").size()
    zs_clean_per_day = zs[~zs.collide].groupby("trade_date").size()
    # (i) net: one combined order per name per day across both arms
    net_rows = []
    for arm, d in (("FT", ft), ("ZS", zs)):
        s = d[["trade_date", "PERMNO", "side", "w_frac"]].copy()
        s["signed"] = np.where(s.side == "BUY", s.w_frac, -s.w_frac)
        net_rows.append(s)
    net = pd.concat(net_rows).groupby(["trade_date", "PERMNO"], as_index=False).signed.sum()
    gross_notional = ft.w_frac.sum() + zs.w_frac.sum()
    net_notional = net.signed.abs().sum()
    net_per_day = net[net.signed.abs() > 1e-12].groupby("trade_date").size()

    schemes = {
        "parallel_two_accounts": {
            "feasible_if": "Alpaca confirms the 403 block is per-account, not per-entity",
            "orders_per_day_FT": float(ft.groupby("trade_date").size().mean()),
            "orders_per_day_ZS": float(zs.groupby("trade_date").size().mean()),
            "fills_lost": 0.0,
            "calendar_multiplier": 1.0,
        },
        "drop_collisions": {
            "orders_per_day_FT": float(ft_clean_per_day.mean()),
            "orders_per_day_ZS": float(zs_clean_per_day.mean()),
            "fills_lost_FT": collisions["share_of_ft_fills"],
            "fills_lost_ZS": collisions["share_of_zs_fills"],
            "calendar_multiplier": 1.0,
        },
        "net_into_one_book": {
            "orders_per_day": float(net_per_day.mean()),
            "gross_notional_units": float(gross_notional),
            "net_notional_units": float(net_notional),
            "notional_cancelled": float(1.0 - net_notional / gross_notional),
            "calendar_multiplier": 1.0,
        },
        "staggered_arms": {
            "orders_per_day_FT": float(ft.groupby("trade_date").size().mean()),
            "orders_per_day_ZS": float(zs.groupby("trade_date").size().mean()),
            "fills_lost": 0.0,
            "calendar_multiplier": 2.0,
        },
    }
    for name, blk in schemes.items():
        log(f"  {name}:")
        for k, v in blk.items():
            log(f"      {k}: {v}")

    log("\n=== 3. pilot sizing: trading days needed for a day-clustered SE(C) ===")
    sizing = {}
    for arm, n_day in (("FT", float(ft.groupby('trade_date').size().mean())),
                       ("ZS", float(zs.groupby('trade_date').size().mean()))):
        sizing[arm] = {"fills_per_day": n_day, "grid": {}}
        log(f"  --- {arm}  (n={n_day:.1f} fills/day) ---")
        for sig in SIGMA_FILL_BP:
            for rho in RHO_WITHIN_DAY:
                row = {}
                for tgt in SE_TARGETS_BP:
                    row[f"SE<={tgt}bp"] = days_needed(sig, rho, n_day, tgt)
                sizing[arm]["grid"][f"sigma{sig}_rho{rho}"] = row
                log(f"    sigma_fill={sig:>4.0f}bp rho={rho:.1f} -> days for "
                    + "  ".join(f"SE<={t}bp: {row[f'SE<={t}bp']}" for t in SE_TARGETS_BP))

    result = {"collisions": collisions, "schemes": schemes, "sizing": sizing,
              "sizing_model": "SE = sigma*sqrt((1+(n-1)*rho)/(D*n)); "
                              "rho = within-day correlation across fills"}
    dest = OUT / "k14_collision_and_sizing.json"
    dest.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str),
                    encoding="utf-8")
    log(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
