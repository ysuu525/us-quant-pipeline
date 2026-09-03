"""Experiment 12: literature-fixed cost(AUM) projection on development data.

Criterion and use restriction (frozen before reading the first result)
-----------------------------------------------------------------------
This is an estimation deliverable, not a decision rule.  It MUST NOT be used
to choose AUM: target deployment AUM is a blocked field that only the real
deployment mandate can supply.  No impact parameter is fit to project data;
the AUM grid, participation quantiles, coefficients, exponents, and threshold
lines below are fixed in the 2026-09-04 experiment instruction.

Every value on the cost curve is a MODEL VALUE, NOT A MEASUREMENT.  A pilot-
measured C is friction in the participation-rate-near-zero limit and is not the
same quantity as the impact segment projected here.  ``fill - DlyOpen`` cannot
see the strategy's own effect on the opening print: MOO orders jointly help
determine that very print.

Inputs and scope
----------------
Only the already-consumed folds 36--42 enter, through K13 order-audit files
and the corresponding raw daily-return distribution.  No score, label, IC,
P&L, or sealed-fold artifact is read.  ``panel_raw.parquet`` is column-pruned,
date-pushdown filtered, and processed in fixed two-year chunks.

Models (fixed, not scanned)
---------------------------
Square-root projection:

    impact_bp = 1e4 * Y * sigma_d * sqrt(q / ADV),  Y in {0.5, 1, 2}.

The order-unity normalization and square-root form are the experiment's fixed
reading of Grinold & Kahn (1999, *Active Portfolio Management*, 2nd ed.,
ch. 16) and Torre (1997, *BARRA Market Impact Model Handbook*, ch. 7).

ATHL shape comparison:

    temporary_bp = 1e4 * eta * sigma_d * ((q / ADV) / T)^beta,
    eta=0.142, beta=3/5, T=1 day.

These are the published temporary-impact estimates in Almgren, Thum,
Hauptmann & Li (2005), *Direct Estimation of Equity Market Impact*, eq. (8)
and sec. 4.3.  T=1 is an explicitly fixed daily-volume normalization because
K13 has daily ADV but no MOO execution-duration estimate.  Consequently this
line is a shape/sensitivity comparison, not an opening-auction calibration;
the paper's permanent component is deliberately not added.  ATHL sec. 2.1
explicitly excludes market-on-open/close orders because their execution
profiles violate the model assumption, an additional external-validity limit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
PANEL = Path(r"F:\quant\processed\crsp_ciz_2026-08-24_20260825T130601Z")

AUM_GRID = (1_000_000.0, 3_000_000.0, 10_000_000.0, 30_000_000.0)
PARTICIPATION_PCTS = ("p50", "p90", "p99")
Y_VALUES = (0.5, 1.0, 2.0)
ATHL_ETA = 0.142
ATHL_BETA = 0.6
ATHL_T_DAYS = 1.0
NT6_TO_NT5 = 6.0 / 5.0
CURRENT_THRESHOLDS_BP = {"FT": 11.2, "ZS": 6.3}
HISTORICAL_NT6_THRESHOLDS_BP = {"FT": 10.4, "ZS": 4.2}
BREAKEVEN_NT5_BP = {"FT": 22.9, "ZS": 16.4}
FEE_BAND_BP = (3.0, 3.4)
BROKER_REVIEW_DAILY_USD = 200_000.0
OUTPUT_JSON = OUT / "exp12_cost_aum_curve.json"
OUTPUT_SVG = OUT / "exp12_cost_aum_curve.svg"
DEV_RETURN_START = pd.Timestamp("2020-07-01")
DEV_RETURN_END = pd.Timestamp("2023-12-29")

SOURCES = {
    "square_root": {
        "formula": "impact = Y * daily volatility * sqrt(order / ADV)",
        "coefficient_policy": (
            "Y=1.0 is the frozen order-unity center; Y=0.5 and 2.0 are the "
            "only sensitivity values. No project-data calibration."
        ),
        "references": [
            "Grinold & Kahn (1999), Active Portfolio Management, 2nd ed., ch. 16",
            "Torre (1997), BARRA Market Impact Model Handbook, ch. 7",
        ],
    },
    "athl": {
        "formula": "K/sigma = eta * abs(X/(V*T))**beta, where K=J-I/2",
        "eta": ATHL_ETA,
        "beta": ATHL_BETA,
        "T_days": ATHL_T_DAYS,
        "T_assumption": (
            "Fixed to one day so K13 q/ADV is the normalized rate. K13 does not "
            "identify MOO execution duration; this is shape sensitivity only."
        ),
        "reference": (
            "Almgren, Thum, Hauptmann & Li (2005), Direct Estimation of Equity "
            "Market Impact, eq. (8), secs. 4.2-4.3: beta=0.600 +/- 0.038; "
            "eta=0.142 +/- 0.0062."
        ),
        "sample_scope_warning": (
            "Section 2.1 excludes market-on-open and market-on-close orders "
            "because their strongly nonlinear profiles violate the model assumption."
        ),
        "primary_source_url": "https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf",
    },
}


def square_root_impact_bp(participation: float, sigma_d: float, y: float) -> float:
    """Literature-fixed square-root projection in basis points."""
    if participation < 0 or sigma_d < 0 or y < 0:
        raise ValueError("participation, sigma_d, and y must be non-negative")
    return 1.0e4 * y * sigma_d * math.sqrt(participation)


def athl_temporary_impact_bp(
    participation: float,
    sigma_d: float,
    *,
    eta: float = ATHL_ETA,
    beta: float = ATHL_BETA,
    t_days: float = ATHL_T_DAYS,
) -> float:
    """ATHL temporary component K=J-I/2 in bp, not total realized impact."""
    if participation < 0 or sigma_d < 0 or eta < 0 or t_days <= 0:
        raise ValueError("invalid ATHL input")
    return 1.0e4 * eta * sigma_d * (participation / t_days) ** beta


def select_k13_input(out_dir: Path = OUT) -> tuple[dict, Path, float, str]:
    """Prefer measured NT=5; otherwise return the authorized NT=6 x 6/5 proxy."""
    nt5 = out_dir / "k13_order_audit_nt5.json"
    nt6 = out_dir / "k13_order_audit.json"
    if nt5.exists():
        path, expected_nt, scale, status = nt5, 5, 1.0, "NT=5 measured K13 table"
    elif nt6.exists():
        path, expected_nt, scale, status = (
            nt6,
            6,
            NT6_TO_NT5,
            "NT=6 K13 table mechanically scaled by 6/5 (approximation)",
        )
    else:
        raise FileNotFoundError("neither NT=5 nor NT=6 K13 audit JSON exists")
    audit = json.loads(path.read_text(encoding="utf-8"))
    actual_nt = int(audit["frozen_construction"]["NT"])
    if actual_nt != expected_nt:
        raise ValueError(f"{path} declares NT={actual_nt}, expected {expected_nt}")
    return audit, path, scale, status


def select_order_inputs(out_dir: Path = OUT) -> tuple[list[Path], str]:
    """Use matching NT=5 orders when available, otherwise the NT=6 name proxy."""
    nt5 = [out_dir / f"k13_orders_{arm}_nt5.parquet" for arm in ("FT", "ZS")]
    nt6 = [out_dir / f"k13_orders_{arm}.parquet" for arm in ("FT", "ZS")]
    if all(p.exists() for p in nt5):
        return nt5, "NT=5 traded-name union"
    if all(p.exists() for p in nt6):
        return nt6, "NT=6 traded-name union proxy"
    raise FileNotFoundError("complete FT/ZS K13 order parquet pair not found")


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start.normalize()
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=2) - pd.Timedelta(days=1), end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _sigma_from_moments(stats: pd.DataFrame) -> tuple[float, dict]:
    valid = stats.loc[stats["count"] >= 2].copy()
    numerator = valid["sum_sq"] - valid["sum"] ** 2 / valid["count"]
    valid["sigma"] = np.sqrt(np.maximum(numerator, 0.0) / (valid["count"] - 1.0))
    values = valid["sigma"].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("no traded name has at least two finite daily returns")
    return float(values.median()), {
        "definition": (
            "cross-sectional median of per-name sample standard deviations of "
            "daily DlyRet over the K13 development window"
        ),
        "n_names_with_two_returns": int(len(values)),
        "per_name_sigma_p10": float(values.quantile(0.10)),
        "per_name_sigma_p50": float(values.quantile(0.50)),
        "per_name_sigma_p90": float(values.quantile(0.90)),
    }


def load_sigma_d(
    panel_path: Path,
    order_paths: list[Path],
    start: str,
    end: str,
) -> tuple[float, dict]:
    """Load only traded-name returns using date pushdown and fixed 2-year chunks."""
    names: set[int] = set()
    for path in order_paths:
        frame = pd.read_parquet(path, columns=["PERMNO"])
        names.update(frame["PERMNO"].dropna().astype(np.int64).tolist())
        del frame
    if not names:
        raise ValueError("K13 order files contain no traded names")

    pieces: list[pd.DataFrame] = []
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    for chunk_lo, chunk_hi in _date_chunks(lo, hi):
        frame = pd.read_parquet(
            panel_path,
            columns=["PERMNO", "DlyCalDt", "DlyRet"],
            filters=[
                ("DlyCalDt", ">=", chunk_lo),
                ("DlyCalDt", "<=", chunk_hi),
            ],
        )
        frame = frame.loc[frame["PERMNO"].isin(names), ["PERMNO", "DlyRet"]]
        frame["DlyRet"] = pd.to_numeric(frame["DlyRet"], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["DlyRet"])
        if not frame.empty:
            frame["ret_sq"] = frame["DlyRet"] ** 2
            grouped = frame.groupby("PERMNO", sort=False).agg(
                count=("DlyRet", "count"),
                sum=("DlyRet", "sum"),
                sum_sq=("ret_sq", "sum"),
            )
            pieces.append(grouped)
        del frame
    if not pieces:
        raise ValueError("no finite DlyRet observations found for K13 traded names")
    stats = pd.concat(pieces).groupby(level=0).sum()
    sigma, meta = _sigma_from_moments(stats)
    meta.update({
        "n_order_names": int(len(names)),
        "window": [str(lo.date()), str(hi.date())],
        "panel_columns": ["PERMNO", "DlyCalDt", "DlyRet"],
        "date_pushdown": True,
        "chunking": "fixed non-overlapping intervals of at most two years",
    })
    return sigma, meta


def participation_rows(audit: dict, scale: float, sigma_d: float) -> list[dict]:
    """Build the exact arm x AUM x percentile grid; no interpolation or scan."""
    rows: list[dict] = []
    for arm in ("FT", "ZS"):
        for aum in AUM_GRID:
            block = audit["arms"][arm]["by_aum"][f"{aum:.0f}"]
            for pct in PARTICIPATION_PCTS:
                participation = float(block["participation_of_adv20"][pct]) * scale
                row = {
                    "arm": arm,
                    "aum_usd": aum,
                    "participation_percentile": pct,
                    "participation_decimal": participation,
                    "participation_bp_of_adv": participation * 1.0e4,
                    "square_root_impact_bp": {
                        f"Y={y:g}": square_root_impact_bp(participation, sigma_d, y)
                        for y in Y_VALUES
                    },
                    "athl_temporary_impact_bp": athl_temporary_impact_bp(
                        participation, sigma_d
                    ),
                }
                rows.append(row)
    return rows


def crossing_aum(base_aum: float, base_impact_bp: float, threshold_bp: float, exponent: float) -> float:
    """Analytic AUM crossing when participation is proportional to AUM."""
    if min(base_aum, base_impact_bp, threshold_bp, exponent) <= 0:
        raise ValueError("crossing inputs must be positive")
    return base_aum * (threshold_bp / base_impact_bp) ** (1.0 / exponent)


def compute_crossings(audit: dict, scale: float, sigma_d: float) -> dict:
    result: dict[str, dict] = {}
    for arm in ("FT", "ZS"):
        base = audit["arms"][arm]["by_aum"]["1000000"]
        p90 = float(base["participation_of_adv20"]["p90"]) * scale
        sqrt_base = square_root_impact_bp(p90, sigma_d, 1.0)
        athl_base = athl_temporary_impact_bp(p90, sigma_d)
        threshold = CURRENT_THRESHOLDS_BP[arm]
        daily_at_1m = float(base["daily_traded_usd"]) * scale
        result[arm] = {
            "basis": "p90 participation, primary square-root Y=1",
            "deployment_threshold_bp": threshold,
            "square_root_threshold_aum_usd": crossing_aum(
                1_000_000.0, sqrt_base, threshold, 0.5
            ),
            "athl_shape_threshold_aum_usd": crossing_aum(
                1_000_000.0, athl_base, threshold, ATHL_BETA
            ),
            "broker_200k_daily_review_aum_usd": (
                BROKER_REVIEW_DAILY_USD / daily_at_1m * 1_000_000.0
            ),
            "daily_traded_usd_at_10m": daily_at_1m * 10.0,
        }
    return result


def _svg(result: dict) -> str:
    """Render a dependency-free two-panel SVG using p90 participation curves."""
    width, height = 1120, 540
    margin_x, top, panel_w, plot_h, gap = 76, 70, 455, 385, 65
    rows = result["impact_table"]
    curve_rows = [r for r in rows if r["participation_percentile"] == "p90"]
    impacts = []
    for row in curve_rows:
        impacts.extend(row["square_root_impact_bp"].values())
        impacts.append(row["athl_temporary_impact_bp"])
    y_max = max(14.0, math.ceil(max(impacts + [11.2, 6.3, 3.4]) / 5.0) * 5.0)

    def sx(aum: float, panel_left: float) -> float:
        lo, hi = math.log10(AUM_GRID[0]), math.log10(AUM_GRID[-1])
        return panel_left + (math.log10(aum) - lo) / (hi - lo) * panel_w

    def sy(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    colors = {"Y=0.5": "#2563eb", "Y=1": "#059669", "Y=2": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.axis{stroke:#536078;stroke-width:1}.grid{stroke:#d9dee8;stroke-width:1}.small{font-size:11px}.label{font-size:13px}.title{font-size:18px;font-weight:600}.panel{font-size:15px;font-weight:600}</style>',
        '<text x="560" y="28" text-anchor="middle" class="title">Experiment 12: modeled impact vs AUM (p90 participation)</text>',
        '<text x="560" y="48" text-anchor="middle" class="small">MODEL VALUES, NOT MEASUREMENTS · log AUM axis</text>',
    ]
    for panel_idx, arm in enumerate(("FT", "ZS")):
        left = margin_x + panel_idx * (panel_w + gap)
        right = left + panel_w
        bottom = top + plot_h
        parts.append(f'<text x="{(left + right) / 2:.1f}" y="64" text-anchor="middle" class="panel">{arm}</text>')
        fee_top, fee_bottom = sy(FEE_BAND_BP[1]), sy(FEE_BAND_BP[0])
        parts.append(f'<rect x="{left}" y="{fee_top:.2f}" width="{panel_w}" height="{fee_bottom-fee_top:.2f}" fill="#fbbf24" opacity="0.20"/>')
        for y_tick in np.linspace(0.0, y_max, 6):
            y = sy(float(y_tick))
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>')
            if panel_idx == 0:
                parts.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" class="small">{y_tick:.0f}</text>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
        for aum in AUM_GRID:
            x = sx(aum, left)
            parts.append(f'<line x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom+5}" class="axis"/>')
            parts.append(f'<text x="{x:.2f}" y="{bottom+20}" text-anchor="middle" class="small">${aum/1e6:g}M</text>')
        arm_rows = sorted((r for r in curve_rows if r["arm"] == arm), key=lambda r: r["aum_usd"])
        for label, color in colors.items():
            pts = " ".join(
                f'{sx(r["aum_usd"], left):.2f},{sy(r["square_root_impact_bp"][label]):.2f}'
                for r in arm_rows
            )
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        athl_pts = " ".join(
            f'{sx(r["aum_usd"], left):.2f},{sy(r["athl_temporary_impact_bp"]):.2f}'
            for r in arm_rows
        )
        parts.append(f'<polyline points="{athl_pts}" fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="7 4"/>')
        for threshold_arm, threshold in CURRENT_THRESHOLDS_BP.items():
            y = sy(threshold)
            dash = "4 3" if threshold_arm == arm else "2 5"
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#111827" stroke-width="1.2" stroke-dasharray="{dash}"/>')
            parts.append(f'<text x="{right-3}" y="{y-3:.2f}" text-anchor="end" class="small">{threshold_arm} {threshold:g}bp</text>')
        review_aum = result["crossings"][arm]["broker_200k_daily_review_aum_usd"]
        x_review = sx(max(AUM_GRID[0], min(AUM_GRID[-1], review_aum)), left)
        parts.append(f'<line x1="{x_review:.2f}" y1="{top}" x2="{x_review:.2f}" y2="{bottom}" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x_review+4:.2f}" y="{top+14}" class="small">$200k/day @ ${review_aum/1e6:.2f}M</text>')
    parts.extend([
        '<text x="18" y="270" text-anchor="middle" class="label" transform="rotate(-90 18 270)">modeled impact (bp)</text>',
        '<text x="560" y="520" text-anchor="middle" class="label">AUM per arm</text>',
        '<line x1="78" y1="492" x2="102" y2="492" stroke="#2563eb" stroke-width="2.5"/><text x="108" y="496" class="small">Y=0.5</text>',
        '<line x1="170" y1="492" x2="194" y2="492" stroke="#059669" stroke-width="2.5"/><text x="200" y="496" class="small">Y=1</text>',
        '<line x1="254" y1="492" x2="278" y2="492" stroke="#dc2626" stroke-width="2.5"/><text x="284" y="496" class="small">Y=2</text>',
        '<line x1="338" y1="492" x2="362" y2="492" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="7 4"/><text x="368" y="496" class="small">ATHL beta=0.6</text>',
        '<rect x="490" y="485" width="24" height="10" fill="#fbbf24" opacity="0.25"/><text x="520" y="496" class="small">fee room 3.0-3.4bp</text>',
        '</svg>',
    ])
    return "\n".join(parts)


def main() -> None:
    audit, audit_path, participation_scale, source_status = select_k13_input()
    order_paths, order_status = select_order_inputs()
    audit_start, audit_end = audit["frozen_construction"]["window"]
    # K13 keeps a wider price/ADV buffer (currently through 2024-01-05).  The
    # volatility input is explicitly a fold36--42 development-window statistic;
    # cap it before any panel read so no 2024 row can enter this experiment.
    start = max(pd.Timestamp(audit_start), DEV_RETURN_START)
    end = min(pd.Timestamp(audit_end), DEV_RETURN_END)
    sigma_d, sigma_meta = load_sigma_d(
        PANEL / "panel_raw.parquet", order_paths,
        str(start.date()), str(end.date())
    )
    sigma_meta["order_name_source"] = order_status
    sigma_meta["k13_audit_buffer_window"] = [audit_start, audit_end]
    sigma_meta["development_window_cap"] = [
        str(DEV_RETURN_START.date()), str(DEV_RETURN_END.date())]
    rows = participation_rows(audit, participation_scale, sigma_d)
    crossings = compute_crossings(audit, participation_scale, sigma_d)
    result = {
        "experiment": "exp12_cost_aum_curve",
        "criterion": "estimation delivery; no decision threshold",
        "use_restriction": "must not be used to choose AUM",
        "model_status": "all curve values are model values, not measurements",
        "pilot_limitation": (
            "Pilot C is near-zero-participation friction, not this modeled impact; "
            "fill-DlyOpen cannot measure own MOO impact because the order helps set "
            "the opening print."
        ),
        "k13_source": {
            "path": str(audit_path),
            "status": source_status,
            "participation_multiplier": participation_scale,
        },
        "sigma_d": sigma_d,
        "sigma_d_metadata": sigma_meta,
        "parameters": {
            "aum_grid_usd": list(AUM_GRID),
            "participation_percentiles": list(PARTICIPATION_PCTS),
            "square_root_Y": list(Y_VALUES),
            "athl_eta": ATHL_ETA,
            "athl_beta": ATHL_BETA,
            "athl_T_days": ATHL_T_DAYS,
            "broker_review_daily_usd": BROKER_REVIEW_DAILY_USD,
        },
        "threshold_fork": {
            "current_nt5_bp": CURRENT_THRESHOLDS_BP,
            "current_nt5_breakeven_bp": BREAKEVEN_NT5_BP,
            "current_formula": "C <= BE_dev * 0.75 - 6bp",
            "current_source": "ledger 2026-09-03 ablation-read",
            "historical_nt6_bp": HISTORICAL_NT6_THRESHOLDS_BP,
            "historical_source": "ledger:415",
            "policy": "use current NT=5 thresholds; NT=6 pair shown as historical only",
        },
        "broker_context": {
            "alpaca_all_in_share_of_C_stop": "15-25%",
            "room_for_fill_minus_DlyOpen_bp": list(FEE_BAND_BP),
            "non_retail_review_hint_daily_usd": BROKER_REVIEW_DAILY_USD,
            "historical_nt6_daily_traded_usd_at_10m": {
                "FT": 1_900_000.0,
                "ZS": 1_400_000.0,
            },
            "historical_note": (
                "The 1.90M/1.40M figures are HANDOFF 11.4 NT=6 context; "
                "crossings use the selected NT=5 table or declared 6/5 proxy."
            ),
        },
        "fee_room_bp": list(FEE_BAND_BP),
        "sources": SOURCES,
        "impact_table": rows,
        "crossings": crossings,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_SVG.write_text(_svg(result), encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_SVG}")
    print(f"sigma_d={sigma_d:.8f}; {source_status}; {order_status}")


if __name__ == "__main__":
    main()
