"""K15: two market-side structural series over the full 2000-2026 panel.

A. sigma_cs (cross-sectional dispersion of the 6-day forward return) by year.
   The existing -0.396pp/yr trend was estimated on 2020H2--2023 only, and that
   window STARTS at the post-COVID dispersion spike.  Whether it is a secular
   trend or a reversion to normal has never been checked.  It multiplies gross
   directly (bridge formula: gross_decile_spread ~ 3.51 * IC * sigma_cs), so it
   sets the "honest forward gross" that C_stop was solved from.

B. mom_skip(t-24..t-5) long-short payoff BY LIQUIDITY TERCILE, by year.
   K6b located the strategy on the short-horizon-momentum side.  A.1's proximate
   mechanism is a migration of the signal along the liquidity spectrum (illiquid
   end in 2003-04, liquid end today); the remote cause is unexplained.  If the
   underlying anomaly's own liquidity profile flipped on the same timeline, the
   remote cause closes and A.1 becomes a known market-structure fact rather than
   a defect of this model.

What this reads
---------------
CRSP panel columns PERMNO / DlyCalDt / DlyRet / DlyPrcVol ONLY.
**No model scores, no strategy readings, no fold-level performance numbers.**
Both series are properties of the market, not of this project's split; running
them does not consume folds 05--35.

Disclosure: knowing sigma_cs for 2005--2020 does narrow the prior on what the
confirmation set can show (gross = 3.51 * IC * sigma_cs), but reveals nothing
about IC, and the design is frozen by rule before the folds open.  Recorded here
so the ledger entry can state it.

Memory: CLAUDE.md section 7 forbids reading panel_raw whole (49.87M rows).
Processed in 2-year chunks with warm-up/cool-down overlap; only daily aggregates
are retained.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from k9_cost_alpha_frontier import OUT, PANEL

START, END = 2000, 2026
CHUNK_YEARS = 2
WARMUP_DAYS, COOLDOWN_DAYS = 45, 12
POOLS = (500, 1000, 2000)
FWD_H = 6            # matches the 6-day label horizon
N_TERCILES = 3
MIN_NAMES = 100


def log(m: str) -> None:
    print(m, flush=True)


def load_chunk(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(
        PANEL / "panel_raw.parquet",
        columns=["PERMNO", "DlyCalDt", "DlyRet", "DlyPrcVol"],
        filters=[("DlyCalDt", ">=", lo), ("DlyCalDt", "<=", hi)],
    )
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    return (df.dropna(subset=["DlyRet"])
              .sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True))


def build(df: pd.DataFrame) -> pd.DataFrame:
    gp = df["PERMNO"]

    def roll_mean(s, w, mp):
        return (s.groupby(gp, sort=False).rolling(w, min_periods=mp)
                 .mean().reset_index(level=0, drop=True))

    # lagged 20-day dollar ADV -- known at t close
    df["adv20"] = roll_mean(df["DlyPrcVol"], 20, 10).groupby(gp, sort=False).shift(1)

    lr = np.log1p(df["DlyRet"].clip(lower=-0.99))
    # mom_skip(t-24..t-5): 20-day sum shifted 5, exactly as in k6b_spanning
    c20 = (lr.groupby(gp, sort=False).rolling(20, min_periods=10)
             .sum().reset_index(level=0, drop=True))
    df["mom_skip"] = np.expm1(c20.groupby(gp, sort=False).shift(5))

    # forward 6-day compounded return (label proxy) and next-day return
    fwd = (lr.groupby(gp, sort=False)
             .apply(lambda s: s.iloc[::-1].rolling(FWD_H, min_periods=FWD_H)
                    .sum().iloc[::-1])
             .reset_index(level=0, drop=True))
    df["fwd6"] = np.expm1(fwd.groupby(gp, sort=False).shift(-1))
    df["fwd1"] = df.groupby(gp, sort=False)["DlyRet"].shift(-1)
    return df


def daily_rows(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> list[dict]:
    out = []
    core = df[(df.DlyCalDt >= lo) & (df.DlyCalDt <= hi)]
    for dt, g in core.groupby("DlyCalDt", sort=True):
        g = g[np.isfinite(g.adv20)]
        if len(g) < MIN_NAMES:
            continue
        g = g.sort_values("adv20", ascending=False)
        row: dict = {"date": dt, "n_eligible": int(len(g))}
        for n in POOLS:
            pool = g.head(n)
            lab = pool.fwd6.to_numpy(dtype=float)
            lab = lab[np.isfinite(lab)]
            row[f"sigma_cs_top{n}"] = float(lab.std(ddof=1)) if len(lab) >= MIN_NAMES else np.nan

        # --- B: mom_skip long-short within liquidity terciles of top500
        pool = g.head(500)
        pool = pool[np.isfinite(pool.mom_skip) & np.isfinite(pool.fwd1)]
        if len(pool) >= MIN_NAMES:
            # tercile 0 = most liquid (pool already sorted by adv20 desc)
            edges = np.array_split(np.arange(len(pool)), N_TERCILES)
            for ti, idx in enumerate(edges):
                sub = pool.iloc[idx]
                if len(sub) < 30:
                    continue
                q = sub.mom_skip.quantile([0.2, 0.8])
                hi_leg = sub.fwd1[sub.mom_skip >= q.loc[0.8]].mean()
                lo_leg = sub.fwd1[sub.mom_skip <= q.loc[0.2]].mean()
                row[f"momskip_ls_t{ti}"] = float(hi_leg - lo_leg)
                row[f"adv_med_t{ti}"] = float(sub.adv20.median())
        out.append(row)
    return out


def main() -> None:
    rows: list[dict] = []
    for y0 in range(START, END + 1, CHUNK_YEARS):
        y1 = min(y0 + CHUNK_YEARS - 1, END)
        lo, hi = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1}-12-31")
        pad_lo = lo - pd.Timedelta(days=int(WARMUP_DAYS * 1.6))
        pad_hi = hi + pd.Timedelta(days=int(COOLDOWN_DAYS * 1.6))
        log(f"[{y0}-{y1}] loading {pad_lo.date()}..{pad_hi.date()} ...")
        df = load_chunk(pad_lo, pad_hi)
        if not len(df):
            continue
        df = build(df)
        got = daily_rows(df, lo, hi)
        rows.extend(got)
        log(f"          {len(df):,} rows -> {len(got)} trading days")
        del df

    panel = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    panel["year"] = panel.date.dt.year
    panel.to_parquet(OUT / "k15_market_structure_daily.parquet", index=False)

    log("\n=== A. sigma_cs by year (annualised %, cross-sectional std of 6d fwd) ===")
    log("year   n_days  top500   top1000  top2000   momskip_LS liquid/mid/illiq (bp/day)")
    yearly = {}
    for y, g in panel.groupby("year"):
        s5 = g[f"sigma_cs_top500"].mean() * 100
        s10 = g["sigma_cs_top1000"].mean() * 100
        s20 = g["sigma_cs_top2000"].mean() * 100
        ls = [g.get(f"momskip_ls_t{t}", pd.Series(dtype=float)).mean() * 1e4
              for t in range(N_TERCILES)]
        yearly[int(y)] = {"n_days": int(len(g)), "sigma_top500_pct": float(s5),
                          "sigma_top1000_pct": float(s10), "sigma_top2000_pct": float(s20),
                          "momskip_ls_bp": [float(x) for x in ls]}
        log(f"{y}   {len(g):>5}   {s5:6.3f}   {s10:7.3f}  {s20:7.3f}    "
            + "  ".join(f"{x:+7.2f}" for x in ls))

    def trend(sub: pd.DataFrame, col: str) -> tuple[float, float]:
        x = (sub.date - sub.date.min()).dt.days.to_numpy() / 365.25
        y = sub[col].to_numpy() * 100
        ok = np.isfinite(y)
        if ok.sum() < 100:
            return float("nan"), float("nan")
        b = np.polyfit(x[ok], y[ok], 1)
        return float(b[0]), float(y[ok].mean())

    log("\n=== A2. sigma_cs(top500) trend, pp/year, by window ===")
    windows = {
        "full 2000-2026": (2000, 2026),
        "pre-COVID 2000-2019": (2000, 2019),
        "modern window used for -0.396 (2020H2-2023)": (2020, 2023),
        "ex-COVID 2000-2019 + 2022-2026": None,
        "post-COVID 2022-2026": (2022, 2026),
    }
    trends = {}
    for name, w in windows.items():
        if w is None:
            sub = panel[(panel.year <= 2019) | (panel.year >= 2022)]
        else:
            sub = panel[(panel.year >= w[0]) & (panel.year <= w[1])]
            if name.startswith("modern"):
                sub = sub[sub.date >= "2020-06-01"]
        slope, mean = trend(sub, "sigma_cs_top500")
        rel = slope / mean * 100 if mean == mean and mean else float("nan")
        trends[name] = {"slope_pp_per_yr": slope, "mean_pct": mean,
                        "rel_pct_per_yr": rel, "n_days": int(len(sub))}
        log(f"  {name:<44} slope {slope:+7.3f} pp/yr  mean {mean:6.3f}%  "
            f"rel {rel:+6.2f}%/yr  n={len(sub)}")

    log("\n=== A3. where does today sit vs history? ===")
    lr_mean = panel[panel.year <= 2019].sigma_cs_top500.mean() * 100
    recent = panel[panel.year >= 2024].sigma_cs_top500.mean() * 100
    modern = panel[(panel.date >= "2020-06-01") & (panel.date <= "2023-12-31")] \
        .sigma_cs_top500.mean() * 100
    log(f"  2000-2019 mean = {lr_mean:.3f}%   2020H2-2023 mean = {modern:.3f}%   "
        f"2024+ mean = {recent:.3f}%")
    log(f"  2024+ vs pre-2020 long-run mean: {(recent/lr_mean-1)*100:+.1f}%")

    log("\n=== B. mom_skip liquidity migration: liquid tercile minus illiquid ===")
    panel["ls_diff"] = panel.get("momskip_ls_t0") - panel.get("momskip_ls_t2")
    for y, g in panel.groupby("year"):
        d = g.ls_diff.mean() * 1e4
        log(f"  {y}: liquid - illiquid = {d:+7.2f} bp/day")
    x = (panel.date - panel.date.min()).dt.days.to_numpy() / 365.25
    y = panel.ls_diff.to_numpy() * 1e4
    ok = np.isfinite(y)
    b = np.polyfit(x[ok], y[ok], 1)
    log(f"  trend of (liquid - illiquid): {b[0]:+.4f} bp/day per year, n={ok.sum()}")

    result = {
        "definitions": {
            "sigma_cs": "cross-sectional std of 6-day forward compounded return "
                        "within top-N by lagged ADV20; matches bootstrap_errorbars "
                        "sigma except the pool is ADV-ranked rather than the scoring pool",
            "mom_skip": "expm1(20-day log-return sum shifted 5) == k6b_spanning",
            "ls": "top-quintile minus bottom-quintile mom_skip, next-day return, "
                  "within each liquidity tercile of top500",
            "tercile_0": "most liquid", "tercile_2": "least liquid",
        },
        "yearly": yearly, "sigma_trends": trends,
        "sigma_levels_pct": {"pre2020_mean": lr_mean, "modern_2020H2_2023_mean": modern,
                             "recent_2024plus_mean": recent},
        "momskip_liquid_minus_illiquid_trend_bp_per_day_per_yr": float(b[0]),
    }
    dest = OUT / "k15_market_structure_eras.json"
    dest.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str),
                    encoding="utf-8")
    log(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
