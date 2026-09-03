"""K13: mechanical order audit for the two-arm MOO cost pilot.

Purpose
-------
Size the real-money cost pilot (AUM, duration, target fill count, per-arm vs
shared orders).  Nothing here is an alpha read.

What this script reads
----------------------
Frozen scores for folds 36--42 (already-consumed development folds), lagged
ADV20, and signal-date closing prices.  It NEVER reads ``labels.parquet``,
``DlyRet``, or any forward return.  No P&L, no IC, no fold-level performance
number is produced.

Pre-registered scope (written before the run)
---------------------------------------------
The outputs MAY be used to choose the pilot's AUM, its duration, its target
fill count per arm, and whether the two arms can share fills.
The outputs MAY NOT be used to choose holding period NT, exit band EXIT_PCT,
universe TOPN, or any other element of the frozen strategy: selecting those on
already-consumed folds would be a new development search.  The frozen values
NT=6, EXIT_PCT=0.30, TOPN=500 are imported, not searched.

Cost benchmark this pilot will measure
--------------------------------------
``src/crsp_pipeline/labels.py`` builds every position at the t+1 raw ``DlyOpen``.
The cost that enters ``net = gross - d * C`` is therefore, per one-way fill,

    C = sign * (fill_price - CRSP DlyOpen) / DlyOpen + commissions and fees

and nothing else.  Decision-close -> open drift is NOT part of C: the backtest
pays it too.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from k9_cost_alpha_frontier import EPS, EXIT_PCT, FOLDS, HI, LO, NT, OUT, PANEL, TOPN

ARM_PATHS = {
    "FT": {f: OUT / f"{f}_lb90_s0_poolB_universe" / f"eval_amp_lb90_{f}" for f in FOLDS},
    "ZS": {f: OUT / "zeroshot_base" / f"eval_zeroshot_{f}" for f in FOLDS},
}
AUM_GRID = (1e6, 3e6, 1e7, 3e7)
FILL_TARGETS = (200, 500, 1000)
PCTS = (10, 25, 50, 75, 90, 99)


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


def load_market() -> tuple[dict, dict]:
    """Column-pruned, date-pushdown load.  ADV20 is strictly shifted one day."""
    df = pd.read_parquet(
        PANEL / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyClose", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", pd.Timestamp(LO)),
                 ("DlyCalDt", "<=", pd.Timestamp(HI))],
    )
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df.sort_values(["PERMNO", "DlyCalDt"])
    df["adv20"] = df.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1)
    )
    df["px"] = df["DlyClose"].abs()
    adv = {d: dict(zip(g.PERMNO, g.adv20)) for d, g in df.groupby("DlyCalDt")}
    px = {d: dict(zip(g.PERMNO, g.px)) for d, g in df.groupby("DlyCalDt")}
    del df
    return adv, px


def load_scores(path: Path) -> pd.DataFrame:
    s = pd.read_parquet(_guard(path) / "scores.parquet",
                        columns=["PERMNO", "signal_date", "score"]).dropna()
    s["signal_date"] = pd.to_datetime(s["signal_date"])
    return s


def orders_for_arm(arm: str, adv: dict, px: dict) -> pd.DataFrame:
    """Replay the frozen sleeve construction and emit one row per one-way fill."""
    rows: list[dict] = []
    for fold in FOLDS:
        scores = load_scores(ARM_PATHS[arm][fold])
        by_day = {d: dict(zip(g.PERMNO, g.score))
                  for d, g in scores.groupby("signal_date") if len(g) >= 50}
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
            prev_sel = selection_book[sleeve]
            if prev_sel is None:
                selected = list(order[:k])
            else:
                kept = [p for p in prev_sel if p in pct and pct[p] >= 1.0 - EXIT_PCT][:k]
                held = set(kept)
                selected = kept + [p for p in order if p not in held][:k - len(kept)]
            selection_book[sleeve] = selected
            new = {p: 1.0 / k for p in selected}
            old = path_book[sleeve]

            trade_day = days[i + 1] if i + 1 < len(days) else None
            # old is None only during the NT-day warm-up build; excluded so the
            # audit describes the steady state a pilot would actually observe.
            if old is not None and trade_day is not None:
                pool_adv = np.sort(np.asarray(
                    [day_adv[p] for p in eligible_set if np.isfinite(day_adv[p])],
                    dtype=float))
                names = set(old) | set(new)
                for p in names:
                    d_w = new.get(p, 0.0) - old.get(p, 0.0)
                    if abs(d_w) <= EPS:
                        continue
                    a = day_adv.get(p, np.nan)
                    price = px.get(day, {}).get(p, np.nan)
                    apct = (float(np.searchsorted(pool_adv, a, side="right"))
                            / len(pool_adv)) if np.isfinite(a) and len(pool_adv) else np.nan
                    rows.append({
                        "arm": arm, "fold": fold,
                        "signal_date": day, "trade_date": trade_day,
                        "PERMNO": int(p),
                        "side": "BUY" if d_w > 0 else "SELL",
                        # book-level notional fraction of AUM; only one sleeve
                        # rebalances per day, so no cross-sleeve netting exists
                        "w_frac": abs(d_w) / NT,
                        "adv20": float(a) if np.isfinite(a) else np.nan,
                        "close_px": float(price) if np.isfinite(price) else np.nan,
                        "adv_pct_in_pool": apct,
                    })
            path_book[sleeve] = new
    return pd.DataFrame(rows)


def pct_dict(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {f"p{q}": float(np.percentile(x, q)) for q in PCTS} | {
        "mean": float(x.mean()), "max": float(x.max()), "n": int(len(x))}


def summarize(df: pd.DataFrame) -> dict:
    per_day = df.groupby("trade_date").size()
    out = {
        "n_fills": int(len(df)),
        "n_trade_days": int(df.trade_date.nunique()),
        "n_distinct_names": int(df.PERMNO.nunique()),
        "fills_per_day": pct_dict(per_day.to_numpy()),
        "buy_share": float((df.side == "BUY").mean()),
        "one_way_book_fraction_per_day": pct_dict(
            df.groupby("trade_date").w_frac.sum().to_numpy()),
        "fill_w_frac": pct_dict(df.w_frac.to_numpy()),
        "close_px": pct_dict(df.close_px.to_numpy()),
        "adv_pct_in_pool": pct_dict(df.adv_pct_in_pool.to_numpy()),
        "days_to_reach_fills": {
            str(t): (math.ceil(t / float(per_day.mean())) if per_day.mean() > 0 else None)
            for t in FILL_TARGETS},
        "by_aum": {},
    }
    for aum in AUM_GRID:
        notional = df.w_frac.to_numpy() * aum
        part = notional / df.adv20.to_numpy()
        shares = notional / df.close_px.to_numpy()
        out["by_aum"][f"{aum:.0f}"] = {
            "notional_usd": pct_dict(notional),
            "participation_of_adv20": pct_dict(part),
            "shares_per_fill": pct_dict(shares),
            "daily_traded_usd": float(np.mean(
                df.groupby("trade_date").w_frac.sum().to_numpy()) * aum),
            "frac_fills_under_2k_usd": float(np.mean(notional < 2000.0)),
            "frac_fills_over_1pct_adv": float(np.nanmean(part > 0.01)),
        }
    return out


def main() -> None:
    log("[1/4] loading lagged ADV20 and signal-date closes (pruned + pushdown)...")
    adv, px = load_market()

    frames = {}
    for arm in ("FT", "ZS"):
        log(f"[2/4] replaying frozen sleeve construction for {arm}...")
        frames[arm] = orders_for_arm(arm, adv, px)
        log(f"      {arm}: {len(frames[arm])} one-way fills, "
            f"{frames[arm].trade_date.nunique()} trade days")

    log("[3/4] cross-arm overlap...")
    ft, zs = frames["FT"], frames["ZS"]
    key = ["trade_date", "PERMNO", "side"]
    ft_k = set(map(tuple, ft[key].to_numpy()))
    zs_k = set(map(tuple, zs[key].to_numpy()))
    shared = ft_k & zs_k
    # opposite-side same-name same-day collisions would have to be netted or
    # sequenced by hand in a shared pilot account
    ft_ns = set(map(tuple, ft[["trade_date", "PERMNO"]].to_numpy()))
    zs_ns = set(map(tuple, zs[["trade_date", "PERMNO"]].to_numpy()))
    overlap = {
        "ft_fills": len(ft_k), "zs_fills": len(zs_k),
        "shared_same_name_same_side_same_day": len(shared),
        "shared_share_of_ft": len(shared) / max(len(ft_k), 1),
        "shared_share_of_zs": len(shared) / max(len(zs_k), 1),
        "union_distinct_orders": len(ft_k | zs_k),
        "same_name_same_day_either_side": len(ft_ns & zs_ns),
        "opposite_side_collisions": len(ft_ns & zs_ns) - len(shared),
    }

    result = {
        "frozen_construction": {
            "NT": NT, "TOPN": TOPN, "EXIT_PCT": EXIT_PCT,
            "folds": FOLDS, "window": [LO, HI],
            "execution_convention": "score at t close, fill at t+1 raw DlyOpen",
            "warmup_excluded": "first NT rebalances per sleeve (old is None)",
        },
        "cost_benchmark": (
            "C = sign * (fill - CRSP DlyOpen)/DlyOpen + commissions/fees; "
            "close->open drift is not part of C"),
        "arms": {a: summarize(frames[a]) for a in ("FT", "ZS")},
        "cross_arm": overlap,
    }

    dest = OUT / "k13_order_audit.json"
    dest.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for arm in ("FT", "ZS"):
        frames[arm].to_parquet(OUT / f"k13_orders_{arm}.parquet", index=False)
    log(f"[4/4] wrote {dest}")

    for arm in ("FT", "ZS"):
        s = result["arms"][arm]
        log(f"\n=== {arm} ===")
        log(f"  fills={s['n_fills']}  trade_days={s['n_trade_days']}  "
            f"distinct_names={s['n_distinct_names']}  buy_share={s['buy_share']:.3f}")
        fp = s["fills_per_day"]
        log(f"  fills/day: mean={fp['mean']:.1f} p10={fp['p10']:.0f} "
            f"p50={fp['p50']:.0f} p90={fp['p90']:.0f} max={fp['max']:.0f}")
        ow = s["one_way_book_fraction_per_day"]
        log(f"  one-way book fraction/day: mean={ow['mean']:.5f} p50={ow['p50']:.5f}")
        log(f"  days to 200/500/1000 fills: {s['days_to_reach_fills']}")
        cp = s["close_px"]
        log(f"  traded-name close px: p10={cp['p10']:.1f} p50={cp['p50']:.1f} "
            f"p90={cp['p90']:.1f}")
        ap = s["adv_pct_in_pool"]
        log(f"  ADV pct-rank in top500 pool: p10={ap['p10']:.3f} p50={ap['p50']:.3f} "
            f"p90={ap['p90']:.3f}")
        for aum, blk in s["by_aum"].items():
            n, p = blk["notional_usd"], blk["participation_of_adv20"]
            log(f"  AUM ${float(aum)/1e6:.0f}M: notional/fill p10=${n['p10']:,.0f} "
                f"p50=${n['p50']:,.0f} p90=${n['p90']:,.0f} | "
                f"participation p50={p['p50']*1e4:.2f}bp p90={p['p90']*1e4:.2f}bp "
                f"p99={p['p99']*1e4:.2f}bp | under$2k={blk['frac_fills_under_2k_usd']:.3f}")
    log("\n=== cross-arm ===")
    for k2, v in overlap.items():
        log(f"  {k2}: {v}")


if __name__ == "__main__":
    main()
