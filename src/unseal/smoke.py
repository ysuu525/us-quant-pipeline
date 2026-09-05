"""合成假数据工作区（v4 §7 第 3 条：**先用合成假数据跑通全链路，再碰真数据**）。

价格路径来自 ``src/kronos_ft/train.py::make_smoke_panel``（任务书指定的合成器），
只把日期轴换成从 2000-01-03 起的工作日——这样
``crsp_pipeline.splits.walk_forward_folds`` 在合成日历上产出的折号与真实快照
**一一对应**（fold05 → 2005H1 … fold21 → 2013H1），于是 2013-01-01 的年代切点
在冒烟里也真的把折分成两段，G4 的代码路径才跑得到。

合成分数是**5 日反转 + 噪声**，只用信号日收盘及之前的价格（`CLAUDE.md` §一.1
无前视自查）。**冒烟读数没有任何含义**，不得记入登记簿、不得与真实读数并列。

产出的目录结构与真实工作区同构：

* ``<root>/processed/``：market_index / panel_raw / panel_kronos_adj / universe /
  distributions / security_info_history
* ``<root>/outputs/``：封存评估目录（scores.parquet + SEALED_MANIFEST.json +
  SEALED 哨兵），口径字段与真实封存清单逐项相同
* ``<root>/jkp/usa_mkt_daily_vw_cap.csv``：H1b 张成回归的市场腿
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from crsp_pipeline.sealed import sha256_file, write_seal

from . import config as C
from . import paths as P
from .folds import fold_windows, load_calendar

__all__ = ["build_workspace", "DEFAULT_SMOKE_FOLDS"]

#: 两早两晚，跨 2013-01-01 的年代切点，够 G4 跑出两段。
DEFAULT_SMOKE_FOLDS: tuple[int, ...] = (5, 6, 21, 22)
CALENDAR_START = "2000-01-03"
#: 一直造到 2026-01-02，使机械滚动规则也能产出 fold44–45（干净窗），
#: 让 ``--clean-window`` 的独立交付路径在冒烟里也真的跑一遍。
CALENDAR_END = "2026-01-02"
SNAPSHOT_ID = "smoke_synthetic_panel"


def _base_panel(n_permno: int, dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """调用冻结的合成器，再把日期轴换成需要的那一段。"""
    from kronos_ft.train import make_smoke_panel        # 延迟导入（torch 较重）

    panel, _ = make_smoke_panel(n_permno=n_permno, n_sessions=len(dates), seed=seed)
    panel = panel.reset_index(drop=True)
    panel["DlyCalDt"] = np.tile(np.asarray(dates, dtype="datetime64[ns]"), n_permno)
    panel = panel.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort").reset_index(drop=True)
    grp = panel.groupby("PERMNO", sort=False)["DlyClose"]
    panel["DlyRet"] = grp.pct_change()
    panel["DlyCap"] = panel["DlyClose"] * 1.0e7
    panel["DlyDelFlg"] = ""
    return panel


def _scores_for_window(panel: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp,
                       rng: np.random.Generator) -> pd.DataFrame:
    """5 日反转 + 噪声。**只用 signal_date 当日收盘及之前的价格。**"""
    d = panel[["PERMNO", "DlyCalDt", "DlyClose"]].copy()
    d = d.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort")
    past5 = d.groupby("PERMNO", sort=False)["DlyClose"].shift(5)
    d["score"] = -(d["DlyClose"] / past5 - 1.0)
    d = d[(d["DlyCalDt"] >= lo) & (d["DlyCalDt"] <= hi)].dropna(subset=["score"])
    d["score"] = d["score"] + rng.normal(0.0, 0.02, len(d))
    return (d.rename(columns={"DlyCalDt": "signal_date"})[
        ["PERMNO", "signal_date", "score"]].reset_index(drop=True))


def build_workspace(root: Path, *, n_permno: int = 150,
                    folds: Sequence[int] = DEFAULT_SMOKE_FOLDS,
                    seed: int = 20260905) -> dict:
    """造出一个与真实工作区同构的合成工作区，返回路径与机械生成的折窗口。"""
    root = Path(root)
    processed = root / "processed"
    outputs = root / "outputs"
    jkp = root / "jkp"
    for d in (processed, outputs, jkp):
        d.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(CALENDAR_START, CALENDAR_END)
    panel = _base_panel(n_permno, dates, seed)

    # --- market_index（交易日历的来源）
    mkt = (panel.groupby("DlyCalDt")["DlyRet"].mean().rename("vwretd")
           .reset_index().rename(columns={"DlyCalDt": "caldt"}))
    mkt["vwretd"] = mkt["vwretd"].fillna(0.0)
    mkt.to_parquet(processed / "market_index.parquet", index=False)

    # --- panel_raw / panel_kronos_adj（合成数据无拆股，两者同源）
    raw_cols = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
                "DlyVol", "DlyPrcVol", "DlyRet", "DlyCap", "DlyDelFlg"]
    panel[raw_cols].to_parquet(processed / "panel_raw.parquet", index=False)
    panel[["PERMNO", "DlyCalDt", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
           "DlyVol", "DlyPrcVol"]].to_parquet(
        processed / "panel_kronos_adj.parquet", index=False)

    # --- universe（全在池内）
    uni = panel[["PERMNO", "DlyCalDt"]].copy()
    uni["in_universe"] = True
    uni.to_parquet(processed / "universe.parquet", index=False)

    # --- distributions（少量现金股息，走一遍退出段的除息分支）
    rng = np.random.default_rng(seed)
    ex_dates = pd.DatetimeIndex(rng.choice(dates, size=min(200, len(dates)), replace=False))
    dist = pd.DataFrame({
        "permno": rng.integers(1, n_permno + 1, len(ex_dates)),
        "disexdt": ex_dates,
        "disdivamt": np.round(rng.uniform(0.05, 0.4, len(ex_dates)), 3),
    })
    dist.to_parquet(processed / "distributions.parquet", index=False)

    # --- security_info_history（H1b 的 point-in-time SIC2）
    sic = pd.DataFrame({
        "permno": pd.Series(np.arange(1, n_permno + 1), dtype="Int64"),
        "secinfostartdt": pd.Timestamp("1990-01-01"),
        "secinfoenddt": pd.Timestamp("2100-01-01"),
        "siccd": (2000 + (np.arange(n_permno) % 8) * 500).astype(int),
    })
    sic.to_parquet(processed / "security_info_history.parquet", index=False)

    # --- JKP 市场腿
    pd.DataFrame({"date": mkt["caldt"], "ret": mkt["vwretd"]}).to_csv(
        jkp / "usa_mkt_daily_vw_cap.csv", index=False)

    # --- 折窗口：机械生成，**不手写**
    cal = load_calendar(processed)
    windows = fold_windows(cal, folds)

    # --- 封存评估目录（scores + 清单 + 哨兵），口径字段与真实清单逐项相同
    for f, w in windows.items():
        for arm in C.ARMS:
            d = P.sealed_eval_dir(f, arm, outputs)
            d.mkdir(parents=True, exist_ok=True)
            sc = _scores_for_window(panel, w.val_start, w.val_end,
                                    np.random.default_rng(seed + 100 * f
                                                          + (0 if arm == "ft" else 1)))
            sc.to_parquet(d / "scores.parquet", index=False)
            write_seal(d, {
                "fold_tag": f"sealed_{arm}_fold{f:02d}",
                "snapshot_id": SNAPSHOT_ID,
                "val_window": [str(w.val_start.date()), str(w.val_end.date())],
                "config": dict(C.REQUIRED_SCORING_CONFIG) | {
                    "device": "cpu", "pool": "B (universe anchor)",
                    "arm": f"sealed_{arm}_fold{f:02d}"},
                "code_sha256": {"smoke": "synthetic"},
                "scores_sha256": sha256_file(d / "scores.parquet"),
                "scores_rows": int(len(sc)),
                "limit_obs": 0,
                "score_nan_share": 0.0,
                "extrapolated_share": 0.0,
            })
            (d / "sealed_run.log").write_text("smoke synthetic\n", encoding="utf-8")

    # --- 树基线（XGBoost）的假封存目录：清单字段与真目录同构
    #     （config_sha256 / jkp_snapshot_sha256 / scores_sha256 / val_window）。
    for f, w in windows.items():
        d = Path(outputs) / C.H3_TREE_DIR_TEMPLATE.format(f=f)
        d.mkdir(parents=True, exist_ok=True)
        sc = _scores_for_window(panel, w.val_start, w.val_end,
                                np.random.default_rng(seed + 900 + f))
        sc.to_parquet(d / "scores.parquet", index=False)
        write_seal(d, {
            "snapshot_id": SNAPSHOT_ID,
            "model": "xgboost",
            "fold": f"fold{f:02d}",
            "val_window": [str(w.val_start.date()), str(w.val_end.date())],
            "seeds": [11, 29, 47],
            # 跨折必须唯一，否则 unseal.h3.verify_tree_folds 会中止整个运行
            "config_sha256": "0" * 63 + "1",
            "jkp_snapshot_sha256": "0" * 63 + "2",
            "code_sha256": {"smoke": "synthetic"},
            "scores_sha256": sha256_file(d / "scores.parquet"),
        })
        (d / "sealed_run.log").write_text("smoke synthetic tree\n", encoding="utf-8")

    meta = {
        "root": str(root), "processed": str(processed), "outputs": str(outputs),
        "jkp": str(jkp), "n_permno": n_permno, "folds": list(folds),
        "calendar": [CALENDAR_START, CALENDAR_END],
        "warning": "合成读数没有任何含义；不得记入登记簿、不得与真实读数并列比较。",
    }
    (root / "smoke_workspace.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"processed": processed, "outputs": outputs, "jkp": jkp,
            "cal": cal, "windows": windows, "meta": meta}
