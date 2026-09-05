"""G2 / G3 / G4 / G7：31 折合并统计（H1、H2、H2-era、E 端）。

判据（先于结果落笔，正本见 :mod:`unseal.config` 的 docstring）
--------------------------------------------------------------
* **H1**：区间估计，不做二元 PASS。点估计 + NW(5) 95% CI + 正折数（/31）+
  实现收缩比与主预测区间的相交关系。H4 读取闸门（CI 下界 > 0）**只记录、
  不触发任何动作**。
* **H2**：三门 —— CI 下界 > 0（z = 2.2414）∧ 正折数 ≥ 18/31 ∧
  ``mean RankIC(top500) > 0``。**没有量级门**，不得照抄 `k11:193` 的
  ``delta_adv_ge_half_modern``。
* **H2-era**：切点写死 2013-01-01（按 val_start），只切一次；估计交付、无门槛。
  交互项 = 晚期 − 早期，两段为互斥日集，故 ``SE = sqrt(SE_早² + SE_晚²)``。
* **E**：估计交付。毛年化 + NW(5) CI + 正折数、单边换手、``BE ± 95% CI``、
  成本网格净年化；部署线只报不判（``C`` 未实测前不判 go）。

用途限制
--------
本模块**不计算任何 FT−ZS 差值**：所有函数一次只吃一个臂的数据，返回的字典也
只描述该臂。臂间比较已由 `CLAUDE.md` §二定为 B 层（正交臂淘汰门槛 1.10–1.51×
信号本身，永不触发），在确认集上做它只会消耗数据。

逐折量只作为**符号计数**输出（``folds_positive`` / ``folds_total``），
**不输出逐折数值**——报告里不得出现逐折 IC 表（v4 §3）。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from crsp_pipeline.signal_eval import newey_west_tstat

from . import config as C
from .folds import FoldWindow
from .perfold import fold_out_dir

__all__ = [
    "nw_interval", "load_daily", "h1_summary", "h2_summary",
    "h2_era_summary", "e_summary",
]

_ANNUAL = 252.0


def nw_interval(x, *, lags: int = C.NW_LAG, z: float = C.Z95) -> dict:
    """NW 均值 + 对称 CI + MDE80。``x`` 为逐日观测（推断单位 = 日，不是折）。"""
    s = pd.Series(x, dtype=float).dropna()
    r = newey_west_tstat(s, lags)
    mean, se = float(r["mean"]), float(r["se"])
    ok = np.isfinite(se) and se > 0
    return {
        "mean": mean,
        "nw_se": se if np.isfinite(se) else None,
        "nw_t": float(r["t"]) if np.isfinite(r["t"]) else None,
        "n_days": int(r["n"]),
        "nw_lag": lags,
        "z": z,
        "ci_low": (mean - z * se) if ok else None,
        "ci_high": (mean + z * se) if ok else None,
        "mde80": (C.POWER_MULT_80 * se) if ok else None,
    }


def _sign_counts(frame: pd.DataFrame, column: str) -> tuple[int, int]:
    """(该列折均值 > 0 的折数, 有效折数)。**只回计数，不回逐折数值。**"""
    pos = total = 0
    for _, g in frame.groupby("fold", sort=True):
        v = pd.Series(g[column], dtype=float).dropna()
        if v.empty:
            continue
        total += 1
        if float(v.mean()) > 0:
            pos += 1
    return pos, total


def load_daily(out_root: Path, arm: str, folds: Iterable[int], kind: str) -> pd.DataFrame:
    """把逐折的 ``daily_ic`` / ``daily_money`` 拼成一张逐日表并贴 ``fold`` 列。

    折窗口互不重叠（滚动 6 个月 / 验 6 个月），故日期不会跨折重复。
    """
    parts = []
    for f in folds:
        p = fold_out_dir(out_root, arm, f) / f"{kind}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"缺 {p}；G1 未跑完不得进入汇总")
        d = pd.read_parquet(p)
        d["fold"] = int(f)
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    date_col = "signal_date" if "signal_date" in out.columns else "date"
    return out.sort_values([date_col, "fold"]).reset_index(drop=True)


# ------------------------------------------------------------------ H1


def _intersects(a: Sequence[float | None], b: Sequence[float]) -> bool | None:
    if a[0] is None or a[1] is None:
        return None
    return bool(a[0] <= b[1] and b[0] <= a[1])


def h1_summary(daily_ic: pd.DataFrame, arm: str) -> dict:
    """G2：全池逐日 RankIC 的区间估计 + 收缩比 + 与预注册区间的相交关系。"""
    stat = nw_interval(daily_ic["ic_full_h1"], lags=C.NW_LAG, z=C.Z95)
    pos, total = _sign_counts(daily_ic, "ic_full_h1")
    base = C.H1_DEV_BASE[arm]
    main = C.H1_MAIN_PREDICTION[arm]
    sens = C.H1_SENSITIVITY_PREDICTION
    ci = (stat["ci_low"], stat["ci_high"])
    ratio_ci = (None if ci[0] is None else ci[0] / base,
                None if ci[1] is None else ci[1] / base)
    return {
        "endpoint": "H1",
        "arm": arm,
        "form": "区间估计（不做二元 PASS）",
        "rank_ic": stat,
        "folds_positive": pos,
        "folds_total": total,
        "dev_base": base,
        "realized_shrinkage_ratio": stat["mean"] / base if base else None,
        "realized_shrinkage_ratio_ci95": list(ratio_ci),
        "main_prediction_interval": list(main),
        "main_prediction_intersects_ci95": _intersects(ci, main),
        "sensitivity_prediction_interval": list(sens),
        "sensitivity_prediction_intersects_ci95": _intersects(ci, sens),
        # v4 §2.3：H4 读取闸门是一条**规则**（不是检验），只记录，不触发动作。
        "h4_read_gate_ci_low_gt_0": (None if ci[0] is None else bool(ci[0] > 0)),
        "h4_status": "空置（信号#2 未过准入，ledger:477）——闸门不触发任何动作",
        "caveat": (
            "不相交只否定『选择偏差估计正确』这一元主张，不构成对信号的否定"
            "（v4 §2.3.1）。FT 基数为逐日合并均值、ZS 基数为七折均值，"
            "两者口径不同这一事实随区间一起披露（v4 §2.3.2 限定 1）。"
        ),
    }


# ------------------------------------------------------------------ H2


def h2_summary(daily_ic: pd.DataFrame, arm: str) -> dict:
    """G3：ΔADV 的三门判读（无量级门），z = 2.2414（k=2 配置族 Bonferroni）。"""
    d = daily_ic.dropna(subset=["delta_adv"])
    stat = nw_interval(d["delta_adv"], lags=C.NW_LAG, z=C.Z_BONFERRONI_K2)
    top = nw_interval(d["ic_top500"], lags=C.NW_LAG, z=C.Z_BONFERRONI_K2)
    full = nw_interval(d["ic_full_adv"], lags=C.NW_LAG, z=C.Z_BONFERRONI_K2)
    pos, total = _sign_counts(d, "delta_adv")
    gate_ci = None if stat["ci_low"] is None else bool(stat["ci_low"] > 0)
    gate_folds = bool(pos >= C.H2_POSITIVE_FOLDS_GATE)
    gate_top = bool(top["mean"] > 0)
    gates = {
        "ci_low_gt_0": gate_ci,
        f"folds_positive_ge_{C.H2_POSITIVE_FOLDS_GATE}_of_{C.H2_TOTAL_FOLDS}": gate_folds,
        "mean_ic_top500_gt_0": gate_top,
    }
    all_pass = (gate_ci is True) and gate_folds and gate_top
    return {
        "endpoint": "H2",
        "arm": arm,
        "form": "符号 + CI + 折数（**无量级门**，v3 的 ≥+0.0026 已删）",
        "delta_adv": stat,
        "ic_top500": top,
        "ic_full_pool": full,
        "folds_positive": pos,
        "folds_total": total,
        "gates": gates,
        "all_gates_pass": all_pass,
        "power_at_true_0026": C.H2_POWER_AT_0026,
        "reading": (
            "现代型流动性放大的机制证据成立" if all_pass
            else f"三门未全过；{C.UNANSWERABLE}——真值 0.0026 处功效在本协议的 k=2 口径下仅 "
                 f"FT {C.H2_POWER_AT_0026['k2']['ft']:.1%} / "
                 f"ZS {C.H2_POWER_AT_0026['k2']['zs']:.1%}"
                 "（ledger:508；单臂 z=1.96 口径 45–52%，ledger:412），"
                 "**不得**写成「机制不存在」"
        ),
    }


# ------------------------------------------------------------------ H2-era


def h2_era_summary(daily_ic: pd.DataFrame, windows: dict[int, FoldWindow],
                   arm: str) -> dict:
    """G4：一次年代二分（切点写死 2013-01-01，按 val_start）。**无分年循环。**"""
    era_by_fold = {
        f: ("early" if w.val_start < pd.Timestamp(C.ERA_CUT) else "late")
        for f, w in windows.items()
    }
    d = daily_ic.dropna(subset=["delta_adv"]).copy()
    d["era"] = d["fold"].map(era_by_fold)

    segments: dict[str, dict] = {}
    for era, label in (("early", C.ERA_EARLY), ("late", C.ERA_LATE)):
        part = d[d["era"] == era]
        pos, total = _sign_counts(part, "delta_adv")
        segments[era] = {
            "label": label,
            "n_folds": total,
            "delta_adv": nw_interval(part["delta_adv"], z=C.Z95),
            "ic_top500": nw_interval(part["ic_top500"], z=C.Z95),
            "ic_full_pool": nw_interval(part["ic_full_adv"], z=C.Z95),
            "folds_delta_adv_positive": pos,
            # v4 §2.5.1 已算定的单段 MDE80（原样引用，本次不重算）
            "mde80_disclosed": {
                m: C.H2_ERA_MDE80_SEGMENT[m][arm][era]
                for m in ("delta_adv", "ic_top500")},
        }

    def _interaction(metric: str) -> dict:
        e, l = segments["early"][metric], segments["late"][metric]
        se_e, se_l = e["nw_se"], l["nw_se"]
        out = {"metric": metric, "z": C.Z95,
               "mde80_disclosed": C.H2_ERA_MDE80_INTERACTION[metric][arm]}
        if se_e and se_l:
            diff = l["mean"] - e["mean"]
            se = math.sqrt(se_e ** 2 + se_l ** 2)
            out |= {"mean": diff, "nw_se": se,
                    "ci_low": diff - C.Z95 * se, "ci_high": diff + C.Z95 * se,
                    "mde80_realised": C.POWER_MULT_80 * se,
                    "covers_zero": bool((diff - C.Z95 * se) <= 0 <= (diff + C.Z95 * se))}
        else:
            out |= {"mean": None, "nw_se": None, "ci_low": None, "ci_high": None,
                    "mde80_realised": None, "covers_zero": None}
        return out

    inter = _interaction("delta_adv")
    # 折号对账：31 折全跑时两段必须恰是 fold05–20（16）与 fold21–35（15）（v4 §2.5.1）
    got_early = tuple(sorted(f for f, e in era_by_fold.items() if e == "early"))
    got_late = tuple(sorted(f for f, e in era_by_fold.items() if e == "late"))
    full_run = tuple(sorted(windows)) == C.CONFIRM_FOLDS
    if full_run and (got_early != C.ERA_EARLY_FOLDS or got_late != C.ERA_LATE_FOLDS):
        raise RuntimeError(
            f"年代归段与 v4 §2.5.1 不符：早期 {got_early}、晚期 {got_late}；"
            f"应为 {C.ERA_EARLY_FOLDS} / {C.ERA_LATE_FOLDS}")

    return {
        "endpoint": "H2-era",
        "arm": arm,
        "form": "估计交付、无门槛、只切一次（A 层）",
        "cut": C.ERA_CUT,
        "cut_rule": "按 val_start 机械二分，不扫切点、不试其它分法",
        "expected_early_folds": len(C.ERA_EARLY_FOLDS),
        "expected_late_folds": len(C.ERA_LATE_FOLDS),
        "folds_match_protocol": bool(not full_run or
                                     (got_early == C.ERA_EARLY_FOLDS
                                      and got_late == C.ERA_LATE_FOLDS)),
        "boundary_note": C.ERA_BOUNDARY_NOTE,
        "segments": segments,
        "interaction_late_minus_early_delta_adv": inter,
        "interaction_late_minus_early_ic_top500": _interaction("ic_top500"),
        "reading": (
            f"年代差异{C.UNANSWERABLE}" if inter["covers_zero"] in (True, None)
            else "交互项区间不含 0"
        ),
        "caveat": (
            "早期 16 折 / 晚期 15 折，MDE 较 31 折放大约 √2 倍；MDE80 原样引用 v4 §2.5.1，"
            "只作披露。31 折按折间独立抽样，而 FT 臂相邻折训练窗重叠约 83%，"
            "故真实 SE 被低估、MDE 偏乐观。"
            "交互区间覆盖 0 时不得写「两个年代相同」（措辞强制）。"
        ),
    }


# ------------------------------------------------------------------ E 端


def e_summary(daily_money: pd.DataFrame, arm: str) -> dict:
    """G7：毛年化 + NW(5) CI + 正折数、单边换手、BE ± 95% CI、成本网格、部署线。"""
    g = daily_money["gross"].to_numpy(dtype=float)
    t = daily_money["turn"].to_numpy(dtype=float)
    stat = nw_interval(daily_money["gross"], lags=C.NW_LAG, z=C.Z95)
    pos, total = _sign_counts(daily_money, "gross")

    drag_daily = 2.0 * (1.0 / 1e4) * t / C.NT          # 每 bp 单边成本的日拖累
    drag_ann = float(np.nanmean(drag_daily) * _ANNUAL * 100)
    gross_ann = float(np.nanmean(g) * _ANNUAL * 100)
    se_ann = None if stat["nw_se"] is None else float(stat["nw_se"] * _ANNUAL * 100)
    be = gross_ann / drag_ann if drag_ann > 0 else None
    be_se = (se_ann / drag_ann) if (se_ann is not None and drag_ann > 0) else None

    net = {}
    for bp in C.COST_GRID_BP:
        n = g - bp * drag_daily
        net[str(bp)] = float(np.nanmean(n) * _ANNUAL * 100)

    be_dev = C.BE_DEV_BP[arm]

    def _line(h: float) -> tuple[float, float]:
        disc = be_dev * (1.0 - h)
        return disc, disc - C.DEPLOY_RESERVE_BP

    disc_main, c_main = _line(C.DEPLOY_HAIRCUT_MAIN)
    disc_loose, c_loose = _line(C.DEPLOY_HAIRCUT_RANGE[0])   # h 最小 → 线最松
    disc_tight, c_tight = _line(C.DEPLOY_HAIRCUT_RANGE[1])
    disc_legacy, c_legacy = _line(C.DEPLOY_HAIRCUT_LEGACY)
    deploy = {
        "rule": "go <=> C <= BE_dev * (1 - h) - 6bp（是规则，不是检验；v4 §2.2）",
        "be_dev_bp": be_dev,
        "reserve_bp": C.DEPLOY_RESERVE_BP,
        "h_main": C.DEPLOY_HAIRCUT_MAIN,
        "be_disc_bp_main": disc_main,
        "c_max_bp_main": c_main,
        "h_sensitivity": list(C.DEPLOY_HAIRCUT_RANGE),
        "c_max_bp_sensitivity": [c_tight, c_loose],
        "legacy_line": {
            "h": C.DEPLOY_HAIRCUT_LEGACY,
            "be_disc_bp": disc_legacy,
            "c_max_bp": c_legacy,
            "status": "旧线（×0.75），v4 生效后作废，只作对照披露",
            "tightening_vs_legacy_pct": (
                100.0 * (1.0 - c_main / c_legacy) if c_legacy else None),
        },
        "c_stop_bp": C.C_STOP_BP,
        "c_stop_note": (
            "只作披露、不门控部署：实测成本高于 4bp 时，15.5 年历史无法在统计上证明"
            "净超额 > 0"),
        "deploy_line_below_c_stop": bool(c_main < C.C_STOP_BP),
        "verdict": "C 未实测，只报线、不判 go",
    }
    return {
        "endpoint": "E",
        "arm": arm,
        "form": "估计交付（无 5% 阈值、无二元 PASS）",
        "n_days": int(len(daily_money)),
        "gross_daily": stat,
        "gross_annual_pct": gross_ann,
        "gross_annual_se_pct": se_ann,
        "gross_annual_ci95_pct": [
            None if se_ann is None else gross_ann - C.Z95 * se_ann,
            None if se_ann is None else gross_ann + C.Z95 * se_ann,
        ],
        "folds_positive": pos,
        "folds_total": total,
        "mean_sleeve_turn": float(np.nanmean(t)),
        "oneway_trades_per_year": float(2.0 * _ANNUAL * np.nanmean(t) / C.NT),
        "drag_per_bp_annual_pct": drag_ann,
        "breakeven_oneway_bp": be,
        "breakeven_oneway_bp_se": be_se,
        "breakeven_oneway_bp_ci95": [
            None if be_se is None else be - C.Z95 * be_se,
            None if be_se is None else be + C.Z95 * be_se,
        ],
        "net_annual_pct_by_cost_bp": net,
        "deployment_line": deploy,
        "caveat": (
            "不得说「达到 / 未达到经济目标」——5% 阈值已作废（修订 1）；"
            "不得把「BE 的 CI 覆盖实测 C」说成「策略无效」。"
        ),
    }
