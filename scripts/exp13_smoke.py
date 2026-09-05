"""exp13 的合成数据冒烟：不碰任何真实数据，只验证全链路能跑通并自洽。

`scripts/exp13_compustat_dev_diag.py --smoke` 调这里。合成盘只用来证明
「读 -> 构造 -> 时点 -> 张成 -> D3」这条链路无异常、无缺列、无维度错，
**任何数值都不具备任何解释含义**。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

N_DAYS = 700
N_NAMES = 120
FOLD_SLICES = {36: (400, 520), 37: (520, 640)}
SEED = 20260905


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-02", periods=N_DAYS)


def build_synthetic(root: Path) -> dict[str, object]:
    """在 root 下写出一份自洽的合成快照 + 分数树 + Compustat 派生表。"""
    rng = np.random.default_rng(SEED)
    days = _calendar()
    permnos = np.arange(10001, 10001 + N_NAMES, dtype="int64")
    n_days, n_names = len(days), len(permnos)

    market = rng.normal(0.0004, 0.010, n_days)
    beta = rng.uniform(0.5, 1.5, n_names)
    idio = rng.normal(0.0, 0.018, (n_days, n_names))
    returns = market[:, None] * beta[None, :] + idio
    close = 20.0 * np.exp(np.cumsum(np.log1p(returns), axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.004, (n_days, n_names)))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, (n_days, n_names))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, (n_days, n_names))))
    volume = rng.lognormal(12.0, 0.6, (n_days, n_names))
    shares = rng.lognormal(17.0, 0.5, n_names)

    grid_day = np.repeat(days.to_numpy(), n_names)
    grid_permno = np.tile(permnos, n_days)
    panel = pd.DataFrame({
        "PERMNO": grid_permno,
        "DlyCalDt": grid_day,
        "DlyOpen": open_.reshape(-1),
        "DlyHigh": high.reshape(-1),
        "DlyLow": low.reshape(-1),
        "DlyClose": close.reshape(-1),
        "DlyVol": volume.reshape(-1),
        "DlyPrcVol": (volume * close).reshape(-1),
        "DlyRet": returns.reshape(-1),
        "DlyCap": (close * shares[None, :]).reshape(-1),
    })
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(processed / "panel_raw.parquet", index=False)
    panel[["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
           "DlyVol", "DlyPrcVol"]].to_parquet(
        processed / "panel_kronos_adj.parquet", index=False)
    pd.DataFrame({"caldt": days, "vwretd": market, "ewretd": market}).to_parquet(
        processed / "market_index.parquet", index=False)

    cusips = [f"{i:08d}0"[:9] for i in range(1, n_names + 1)]
    pd.DataFrame({
        "permno": permnos,
        "secinfostartdt": [days[0] - pd.Timedelta(days=400)] * n_names,
        "secinfoenddt": [days[-1] + pd.Timedelta(days=400)] * n_names,
        "siccd": rng.integers(2000, 4000, n_names),
        "hdrcusip9": cusips,
    }).to_parquet(processed / "security_info_history.parquet", index=False)

    jkp = root / "jkp"
    jkp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": days, "ret": market}).to_csv(
        jkp / "usa_mkt_daily_vw_cap.csv", index=False)

    scores_root = root / "scores"
    fold_windows: dict[int, tuple[str, str]] = {}
    for fold, (lo, hi) in FOLD_SLICES.items():
        fold_days = days[lo:hi]
        fold_dir = scores_root / f"fold{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "PERMNO": np.tile(permnos, len(fold_days)),
            "signal_date": np.repeat(fold_days.to_numpy(), n_names),
            "score": rng.normal(0.0, 1.0, len(fold_days) * n_names),
        }).to_parquet(fold_dir / "scores.parquet", index=False)
        fold_windows[fold] = (str(fold_days[0].date()), str(fold_days[-1].date()))

    # --- Compustat 派生表（合成）：每个 permno 一个 gvkey，2014Q1..2024Q4
    quarter_ends = pd.date_range("2014-03-31", "2024-12-31", freq="QE")
    rows = []
    for index, permno in enumerate(permnos):
        gvkey = f"{index + 1:06d}"
        base = rng.normal(0.5, 0.2)
        # 公告日按公司错开，否则合成盘上每天要么全是财报日要么全不是，
        # D3 的「当日 ea 组均值 − 非 ea 组均值」会整列为 NaN，走不到那条路径。
        rdq_offset = 20 + (index % 45)
        for q_index, datadate in enumerate(quarter_ends):
            eps = base + 0.02 * q_index + rng.normal(0.0, 0.08)
            drop_rdq = (index % 17 == 0) and (q_index % 9 == 0)
            rows.append({
                "gvkey": gvkey,
                "datadate": datadate,
                "rdq": pd.NaT if drop_rdq else datadate + pd.Timedelta(days=rdq_offset),
                "fyearq": datadate.year,
                "fqtr": (datadate.month - 1) // 3 + 1,
                "fyr": 12,
                "datacqtr": f"{datadate.year}Q{(datadate.month - 1) // 3 + 1}",
                "datafqtr": f"{datadate.year}Q{(datadate.month - 1) // 3 + 1}",
                "cusip": cusips[index],
                "curcdq": "CAD" if index % 40 == 0 else "USD",
                "epspxq": np.nan if (index % 11 == 0 and q_index % 7 == 0) else eps,
                "ajexq": 1.0,
                "dedup_group_rows": 2 if (index % 7 == 0 and q_index % 5 == 0) else 1,
            })
    fundq = pd.DataFrame(rows)
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    fundq.to_parquet(derived / "fundq_slim.parquet", index=False)
    pd.DataFrame({
        "gvkey": [f"{i + 1:06d}" for i in range(n_names)],
        "permno": permnos,
        "cusip9": cusips,
    }).to_parquet(derived / "gvkey_permno_link.parquet", index=False)

    blocks = ({"lo": str(days[0].date()), "hi": str(days[-1].date()),
               "folds": tuple(FOLD_SLICES)},)
    return {"processed": processed, "jkp": jkp, "derived": derived,
            "scores_root": scores_root, "fold_windows": fold_windows,
            "blocks": blocks, "folds": tuple(FOLD_SLICES)}


def run_smoke(out_dir: Path) -> int:
    import exp13_compustat_dev_diag as exp13

    with tempfile.TemporaryDirectory(prefix="exp13_smoke_") as tmp:
        root = Path(tmp)
        parts = build_synthetic(root)
        report = exp13.run(
            parts["processed"], parts["jkp"], parts["derived"], root / "outputs",
            folds=parts["folds"], fold_windows=parts["fold_windows"],
            blocks=parts["blocks"], scores_root=parts["scores_root"],
            memory_limit_gb=1e9, allow_alt_processed=True,
        )
    checks = {
        "specs_present": sorted(report["D2_spanning"]) == sorted(exp13.SPEC_COLUMNS),
        "alpha_finite": all(np.isfinite(r["alpha_ann_pct"])
                            for r in report["D2_spanning"].values()),
        "beta_ci_present": all("ci95" in b for r in report["D2_spanning"].values()
                               for b in r["betas"].values()),
        "sue_coverage_positive": all(c["sue_non_missing_share"] > 0
                                     for c in report["D1_coverage"]),
        "ea_coverage_positive": all(c["ea_prox_non_missing_share"] > 0
                                    for c in report["D1_coverage"]),
        "dedup_touch_reported": all("dedup_touched_stock_quarter_cells" in c
                                    for c in report["D1_coverage"]),
        "d3_windows_positive": report["D3_earnings_day_variance"]["pooled"]["windows_complete"] > 0,
        "d3_ea_share_in_range": 0.0 < report["D3_earnings_day_variance"]["pooled"][
            "window_share_with_ea_day"] < 1.0,
        "d3_daily_diff_estimated": report["D3_earnings_day_variance"]["pooled"][
            "daily_mean_return_diff"]["n"] > 0,
        "dedup_touch_positive": sum(c["dedup_touched_stock_quarter_cells"]
                                    for c in report["D1_coverage"]) > 0,
        "sue_spec_uses_sue": "sue" in report["D2_spanning"]["S-TH-ind-SUE"]["regressors"],
        "repro_spec_no_extra": not (
            set(exp13.EXTRA_FACTORS)
            & set(report["D2_spanning"]["S-TH-ind-repro"]["regressors"])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_report.json").write_text(
        json.dumps({"checks": checks,
                    "D1_coverage": report["D1_coverage"],
                    "D2_spanning": {k: {kk: vv for kk, vv in v.items() if kk != "betas"}
                                    for k, v in report["D2_spanning"].items()},
                    "D3_pooled": report["D3_earnings_day_variance"]["pooled"]},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    failed = [name for name, ok in checks.items() if not ok]
    print(f"[exp13 smoke] wrote {out_dir / 'smoke_report.json'}")
    return 1 if failed else 0
