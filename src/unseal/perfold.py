"""G1：逐折的标签补算 + 指标补算 + 逐日序列落盘。

**最关键的一步**——封存目录里只有 ``scores.parquet``（2026-09-02「计算授权 !=
读取授权」），标签、IC、ΔADV、钱管道全部要在解封时现算。

复用与口径（一处都不许自己另写）
--------------------------------
* 标签：整段复用 ``scripts/evaluate_fold.py:281-309`` 的做法——``panel_raw``
  列裁剪 + 日期下推、``distributions`` 现金股息（``disdivamt > 0``）、
  ``crsp_pipeline.labels.compute_labels(predict_window=6)``、DlyCap/ADV20 合并。
* ``metrics.json`` 的键沿 ``evaluate_fold.py:313-334``。**注意**
  ``ic_by_adv_quintile`` 的 ADV20 是**当日**口径（``rolling(20, min_periods=15)``），
  与 H2 的**滞后**口径不同，**只作诊断、不进 H2**。
* H2 用的滞后 ADV20 口径来自 ``scripts/k9_cost_alpha_frontier.py:160-161``：
  ``rolling(20, min_periods=10).mean().shift(1)``。
* ΔADV 的逐日构造逐行照 ``scripts/k11_zs_ft_candidate_audit.py:144-174``
  （当日 ADV 有限的名字 ≥ 100 才算；top500 需 ≥ 50 个名字）。
* 钱管道：``src/portfolio/construction.frozen_long_only_returns``（经
  ``src/backtest/money.run_frozen_money``），**NT=5**、top500、进前 10%、
  跌出前 30% 才卖、t 收盘出分 → t+1 开盘成交（`CLAUDE.md` §一.4）；
  ``cost_bp=0`` 出毛收益，成本网格在汇总层用 ``turn`` 列换算。

产物（**绝不写进封存目录**）
----------------------------
``<out>/<arm>/fold<NN>/``：``labels.parquet``、``metrics.json``、
``daily_ic.parquet``、``daily_money.parquet``、``scores.parquet``（分数副本，
供 H1b / H6 取用，使它们不必知道封存目录）、``verify.json``（口径核对事实）。

无前视自查（`CLAUDE.md` §一.1）
-------------------------------
``label`` 只出现在两处：``compute_labels`` 的产物，以及 RankIC / 十分位价差的
**结果侧**。任何当日截面统计量都不用 ``label`` 构造。滞后 ADV20 已 ``shift(1)``。

内存（`CLAUDE.md` §七）
-----------------------
``panel_raw`` **按折切片读**（验证窗前 60 个交易日、后 predict+2 个交易日），
列裁剪到 8 列并按 PERMNO 过滤；单折峰值控制在 4 GB 内，读完即释放。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.money import run_frozen_money
from crsp_pipeline.calendar import TradingCalendar
from crsp_pipeline.labels import compute_labels
from crsp_pipeline.sealed import assert_readable
from crsp_pipeline.signal_eval import (
    daily_rank_ic,
    decile_spread,
    newey_west_tstat,
    winsorized_rank_ic,
)

from . import config as C
from . import paths as P
from .folds import FoldWindow

__all__ = ["run_fold", "fold_out_dir"]

RAW_COLUMNS = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet",
               "DlyDelFlg", "DlyCap", "DlyPrcVol"]


def fold_out_dir(out_root: Path, arm: str, fold: int) -> Path:
    return Path(out_root) / arm / f"fold{fold:02d}"


def _stratified_ic(df: pd.DataFrame, strat_col: str, n_groups: int = 5,
                   nw_lags: int = C.NW_LAG) -> dict:
    """照 ``evaluate_fold.py:127-140`` 原样：按 strat_col 逐日五分位分层。"""
    d = df.dropna(subset=[strat_col, "score", "label"]).copy()
    d["_q"] = d.groupby("signal_date")[strat_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=False)
        if len(s) >= n_groups else pd.Series(np.nan, index=s.index)
    )
    out = {}
    for q in range(n_groups):
        sub = d[d["_q"] == q]
        ic = daily_rank_ic(sub) if len(sub) else pd.Series(dtype=float)
        out[f"q{q + 1}"] = newey_west_tstat(ic, nw_lags)
    return out


def _slice_bounds(cal: TradingCalendar, w: FoldWindow) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = cal.shift(cal.snap_forward(w.val_start), -C.PANEL_BACK_SESSIONS)
    end_idx = min(cal.index_of(cal.snap_back(w.val_end)) + C.PREDICT_WINDOW + 2,
                  len(cal) - 1)
    return lo, cal.dates[end_idx]


def _cash_dividends(processed: Path, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    dist = pd.read_parquet(processed / "distributions.parquet",
                           columns=["permno", "disexdt", "disdivamt"])
    dist["disexdt"] = pd.to_datetime(dist["disexdt"])
    cash = dist[(dist["disdivamt"].fillna(0) > 0)
                & (dist["disexdt"] >= lo) & (dist["disexdt"] <= hi)]
    return cash.rename(columns={"permno": "PERMNO", "disexdt": "ex_date",
                                "disdivamt": "div_cash"})


def _delta_adv_daily(merged: pd.DataFrame) -> pd.DataFrame:
    """逐行照 ``k11_zs_ft_candidate_audit.py:156-173`` 的 ΔADV 构造。

    ``merged`` 需含 ``signal_date / score / label / adv_lag``（滞后 ADV20）。
    """
    rows = []
    for day, group in merged.groupby("signal_date", sort=True):
        group = group[np.isfinite(group["adv_lag"].to_numpy(dtype=float))]
        if len(group) < 100:
            continue
        ic_full = group["score"].rank().corr(group["label"].rank())
        top = group.nlargest(min(C.TOPN, len(group)), "adv_lag")
        ic_top = (top["score"].rank().corr(top["label"].rank())
                  if len(top) >= C.MIN_NAMES else np.nan)
        if np.isfinite(ic_full) and np.isfinite(ic_top):
            rows.append((day, float(ic_full), float(ic_top),
                         float(ic_top - ic_full), int(len(group)), int(len(top))))
    return pd.DataFrame(rows, columns=["signal_date", "ic_full_adv", "ic_top500",
                                       "delta_adv", "n_adv", "n_top"])


def run_fold(processed: Path, outputs_root: Path, window: FoldWindow, arm: str,
             out_root: Path, *, cal: TradingCalendar,
             check_scores_hash: bool = True) -> dict:
    """跑一折一臂：口径核对 → 标签 → 指标 → 逐日 IC/ΔADV → 逐日毛收益。"""
    processed, outputs_root = Path(processed), Path(outputs_root)
    dst = fold_out_dir(out_root, arm, window.fold)
    dst.mkdir(parents=True, exist_ok=True)

    # ---- G0 逐折口径核对（失配即抛，调用方中止整个运行）
    verify = P.verify_fold(window.fold, arm, outputs_root,
                           (window.val_start, window.val_end),
                           check_scores_hash=check_scores_hash)
    (dst / "verify.json").write_text(
        json.dumps(verify, indent=2, ensure_ascii=False), encoding="utf-8")

    scores_path = P.sealed_scores_path(window.fold, arm, outputs_root)
    assert_readable(scores_path, unseal=True)      # 解封授权后的显式放行
    scores = pd.read_parquet(scores_path, columns=["PERMNO", "signal_date", "score"])
    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    scores = scores.dropna(subset=["score"])
    # 分数副本：下游（H1b / H6）只读这棵未封存的输出树，不必知道封存目录。
    scores.to_parquet(dst / "scores.parquet", index=False)

    lo, hi = _slice_bounds(cal, window)
    date_filters = [("DlyCalDt", ">=", lo), ("DlyCalDt", "<=", hi)]
    raw = pd.read_parquet(processed / "panel_raw.parquet",
                          columns=RAW_COLUMNS, filters=date_filters)
    raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"])
    need_pn = set(scores["PERMNO"].unique())
    raw = raw[raw["PERMNO"].isin(need_pn)].reset_index(drop=True)

    # ---- §4 标签（未复权全量面板；`label` 只作目标或结果）
    cash = _cash_dividends(processed, lo, hi)
    labels = compute_labels(raw, scores[["PERMNO", "signal_date"]], cal,
                            cash_dividends=cash, predict_window=C.PREDICT_WINDOW)
    labels.to_parquet(dst / "labels.parquet", index=False)
    ok_share = float((labels["status"] == "ok").mean()) if len(labels) else float("nan")

    df = scores.merge(labels[["PERMNO", "signal_date", "status", "label"]],
                      on=["PERMNO", "signal_date"])
    df = df[(df["status"] == "ok") & df["score"].notna()]

    # ---- 分层诊断变量：t 日市值；ADV20 当日口径（**只作诊断，不进 H2**）
    cap = raw[["PERMNO", "DlyCalDt", "DlyCap"]].rename(columns={"DlyCalDt": "signal_date"})
    df = df.merge(cap, on=["PERMNO", "signal_date"], how="left")
    adv_diag = raw.sort_values(["PERMNO", "DlyCalDt"]).copy()
    adv_diag["ADV20"] = adv_diag.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=15).mean())
    df = df.merge(adv_diag[["PERMNO", "DlyCalDt", "ADV20"]].rename(
        columns={"DlyCalDt": "signal_date"}), on=["PERMNO", "signal_date"], how="left")
    del adv_diag

    # ---- 钱管道与 H2 都用的**滞后** ADV20（k9:160-161 口径）
    px = raw.dropna(subset=["DlyRet"]).sort_values(["PERMNO", "DlyCalDt"]).copy()
    px["oc"] = np.where(px["DlyOpen"].abs() > 0,
                        px["DlyClose"] / px["DlyOpen"].abs() - 1.0, np.nan)
    px["adv_lag"] = px.groupby("PERMNO")["DlyPrcVol"].transform(
        lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    ret = {d: dict(zip(g.PERMNO, g.DlyRet)) for d, g in px.groupby("DlyCalDt")}
    oc = {d: dict(zip(g.PERMNO, g.oc)) for d, g in px.groupby("DlyCalDt")}
    adv = {d: dict(zip(g.PERMNO, g.adv_lag)) for d, g in px.groupby("DlyCalDt")}

    df = df.merge(px[["PERMNO", "DlyCalDt", "adv_lag"]].rename(
        columns={"DlyCalDt": "signal_date"}), on=["PERMNO", "signal_date"], how="left")
    del raw, px

    # ---- 逐日 RankIC（H1 口径 = evaluate_fold 的全池，不带 ADV 条件）
    ic_h1 = daily_rank_ic(df).rename("ic_full_h1")
    ic_h1.index.name = "signal_date"
    delta = _delta_adv_daily(df[["signal_date", "score", "label", "adv_lag"]])
    n_all = df.groupby("signal_date").size().rename("n_all")
    daily_ic = (ic_h1.to_frame().join(n_all)
                .reset_index()
                .merge(delta, on="signal_date", how="outer")
                .sort_values("signal_date").reset_index(drop=True))
    daily_ic.to_parquet(dst / "daily_ic.parquet", index=False)

    # ---- 冻结构造下的逐日毛主动收益（NT=5，cost_bp=0 → 毛）
    money = run_frozen_money(scores, ret, oc, adv, topn=C.TOPN, cost_bp=0.0,
                             exit_pct=C.EXIT_PCT, nt=C.NT, min_names=C.MIN_NAMES)
    money = money.rename(columns={"r": "gross"}).reset_index()
    money.to_parquet(dst / "daily_money.parquet", index=False)
    del ret, oc, adv

    nw = newey_west_tstat(ic_h1, C.NW_LAG)
    sp = decile_spread(df)["net"] if len(df) else pd.Series(dtype=float)
    metrics = {
        "tag": f"unseal_{arm}_fold{window.fold:02d}",
        "val_window": [str(window.val_start.date()), str(window.val_end.date())],
        # `CLAUDE.md` §八：口径必须可机器核对；此处原样抄封存清单的 config
        "scoring_config": verify["scoring_config"],
        "n_obs": int(len(df)),
        "n_days": int(ic_h1.notna().sum()),
        "label_ok_share": ok_share,
        "nw_raw": nw,
        "nw_winsorized": newey_west_tstat(winsorized_rank_ic(df), C.NW_LAG),
        "decile_spread_net_mean": (float(np.nanmean(sp)) if sp.notna().any() else None),
        "ic_by_cap_quintile": _stratified_ic(df, "DlyCap"),
        # 当日 ADV 口径，**只作诊断、不进 H2**
        "ic_by_adv_quintile": _stratified_ic(df, "ADV20"),
        "n_money_days": int(len(money)),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dst / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    return {
        "fold": window.fold, "arm": arm, "dir": str(dst),
        "n_obs": metrics["n_obs"], "n_days": metrics["n_days"],
        "n_money_days": metrics["n_money_days"],
        "n_delta_days": int(delta["delta_adv"].notna().sum()),
        "label_ok_share": ok_share,
        "verify": verify,
    }
