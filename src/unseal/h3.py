"""H3：Kronos 与冻结树基线的**配对 ΔIC**（B 层 / 估计交付）。

判据与地位（先于结果落笔）
--------------------------
* **层级 = B 层**（v4 §2.6）：`[gpt-exp 7]` 实测两主臂的 MDE80 在 7 折口径
  与 31 折外推口径下**均高于**两个候选 SESOI（0.0072 / 0.0108）。故：

  > **在本样本量下该问题不可回答，按预先规则取冻结树基线。**
  > **不得据 H3 判通过 / 失败，不得据 H3 改任何配置。**

* **H3 已退出 family**，链缩为 H1 → H4。**交付形式**：折 05–35 上报配对 ΔIC 的
  点估计、NW(5) 95% CI、正折数（/31），**不产生 PASS/FAIL**。
* **主口径 = XGBoost × FT**（v4.1 附录连带变更；v4 §2.6 原定 XGBoost × ZS，
  ZS 臂已按附录抛弃）。必须随读数披露：FT 组的 ρ 为负（−0.208），
  **配对反而使 SE 增加 9.1% / 方差增加 19.1%**（`[gpt-exp 7]`），
  MDE80 由 ZS 口径的 0.0259 升到 **0.0329**（31 折外推约 **0.0156**）——
  **换臂使功效更差，这一点必须披露，不得只报点估计。**
* LightGBM / CatBoost **不跑**（2026-09-05 授权 (i) 只覆盖 XGBoost），须披露。

口径核对（`CLAUDE.md` §八）
---------------------------
树基线清单的字段与 Kronos 不同，核 ``scores_sha256`` + ``config_sha256`` +
``jkp_snapshot_sha256`` + ``val_window``（2026-09-05 授权行 (iii)），
并**跨折核对 ``config_sha256`` 与 ``jkp_snapshot_sha256`` 唯一**——
两者一旦逐折漂移，说明配置或 JKP 快照在队列中途被改过，中止整个运行。

配对结构
--------
两侧在**同一观测集**上算逐日全池 RankIC：``status == "ok"`` ∧ Kronos 分数非缺
∧ 树分数非缺。配对差 ``ΔIC = Kronos − 树``，**推断单位是逐日配对观测**
（`CLAUDE.md` §二：有配对就用逐日配对，折数只作一致性指标）。

**这不是臂间比较**：v4 §3 第 4 项明列 H3 的配对 ΔIC 为释放项；
被禁止的是 ``FT − ZS``，与本模块无关。
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from crsp_pipeline.sealed import assert_readable
from crsp_pipeline.signal_eval import _spearman

from . import config as C
from . import paths as P
from .aggregate import _sign_counts, nw_interval
from .folds import FoldWindow, excluded_dates_for
from .perfold import fold_out_dir

__all__ = ["verify_tree_folds", "daily_paired_ic", "h3_summary", "run_h3"]


def verify_tree_folds(outputs_root: Path, windows: Sequence[FoldWindow]) -> list[dict]:
    """逐折核对树基线封存清单 + 跨折核对配置与 JKP 快照哈希唯一。"""
    rows = [P.verify_tree_fold(w.fold, outputs_root, (w.val_start, w.val_end))
            for w in windows]
    for key in ("config_sha256", "jkp_snapshot_sha256"):
        distinct = {r[key] for r in rows}
        if len(distinct) > 1:
            raise P.SealedConfigMismatch(
                f"树基线的 {key} 逐折不一致（{sorted(distinct)}）——配置或 JKP 快照"
                f"在队列中途被改过，中止整个运行。")
    return rows


def daily_paired_ic(kronos_scores: pd.DataFrame, tree_scores: pd.DataFrame,
                    labels: pd.DataFrame) -> pd.DataFrame:
    """同一观测集上的逐日全池 RankIC 与配对差。

    ``label`` 只作结果，不进任何特征（`CLAUDE.md` §一.1）。
    """
    lab = labels[labels["status"] == "ok"][["PERMNO", "signal_date", "label"]]
    df = (kronos_scores.rename(columns={"score": "score_kronos"})
          .merge(tree_scores.rename(columns={"score": "score_tree"}),
                 on=["PERMNO", "signal_date"], how="inner")
          .merge(lab, on=["PERMNO", "signal_date"], how="inner"))
    df = df.dropna(subset=["score_kronos", "score_tree", "label"])
    rows = []
    for day, g in df.groupby("signal_date", sort=True):
        if len(g) < 3:
            continue
        ic_k = _spearman(g["score_kronos"], g["label"])
        ic_t = _spearman(g["score_tree"], g["label"])
        if np.isfinite(ic_k) and np.isfinite(ic_t):
            rows.append((day, float(ic_k), float(ic_t), float(ic_k - ic_t), int(len(g))))
    return pd.DataFrame(rows, columns=["signal_date", "ic_kronos", "ic_tree",
                                       "delta_ic", "n_obs"])


def h3_summary(daily: pd.DataFrame) -> dict:
    """配对 ΔIC 的估计交付（不产生 PASS/FAIL）。"""
    stat = nw_interval(daily["delta_ic"], lags=C.NW_LAG, z=C.Z95)
    pos, total = _sign_counts(daily, "delta_ic")
    return {
        "endpoint": "H3",
        "arm": C.H3_ARM,
        "baseline": "frozen XGBoost (configs/gbdt_strong_v2_sealed.json)",
        "form": "配对 ΔIC 的估计交付（B 层，不产生 PASS/FAIL）",
        "delta_ic": stat,
        "ic_kronos": nw_interval(daily["ic_kronos"], lags=C.NW_LAG, z=C.Z95),
        "ic_tree": nw_interval(daily["ic_tree"], lags=C.NW_LAG, z=C.Z95),
        "folds_positive": pos,
        "folds_total": total,
        "n_paired_days": int(len(daily)),
        "mde80_dev7_disclosed": C.H3_MDE80_DEV7,
        "mde80_31fold_extrapolated": C.H3_MDE80_31FOLD,
        "sesoi_candidates": list(C.H3_SESOI_CANDIDATES),
        "reading": (
            f"{C.UNANSWERABLE}，按预先规则取冻结树基线。"
            "不得据此判通过 / 失败，不得据此改任何配置。"),
        "caveat": (
            "主口径按 v4.1 附录由 XGBoost×ZS 改为 XGBoost×FT：FT 组 ρ 为负（−0.208），"
            "配对反而使 SE 增加 9.1%、方差增加 19.1%，MDE80 由 0.0259 升到 0.0329"
            "（31 折外推约 0.0156）——**换臂使功效更差，须与点估计一并披露**。"
            "LightGBM / CatBoost 未跑（2026-09-05 授权只覆盖 XGBoost）。"),
    }


def run_h3(outputs_root: Path, out_root: Path, windows: Sequence[FoldWindow],
           *, arm: str = C.H3_ARM) -> dict:
    """驱动：核对 → 逐折配对 → 合并估计。Kronos 侧读解封输出树，树侧读封存目录。"""
    outputs_root, out_root = Path(outputs_root), Path(out_root)
    ordered = sorted(windows, key=lambda w: w.fold)
    verified = verify_tree_folds(outputs_root, ordered)

    parts = []
    n_tree_excluded = 0
    for w in ordered:
        kdir = fold_out_dir(out_root, arm, w.fold)
        kronos = pd.read_parquet(kdir / "scores.parquet",
                                 columns=["PERMNO", "signal_date", "score"])
        labels = pd.read_parquet(kdir / "labels.parquet",
                                 columns=["PERMNO", "signal_date", "status", "label"])
        tpath = P.tree_sealed_dir(w.fold, outputs_root) / "scores.parquet"
        assert_readable(tpath, unseal=True)
        tree = pd.read_parquet(tpath, columns=["PERMNO", "signal_date", "score"])
        for frame in (kronos, labels, tree):
            frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        # v4.1 附录 §5：树基线一侧显式同样剔除（Kronos 侧在 perfold 已剔除；
        # 这里不靠 inner join 顺带生效，而是明写，免得日后改 join 方式失效）。
        drop = excluded_dates_for(w.fold)
        if drop:
            n_tree_excluded += int(tree["signal_date"].isin(drop).sum())
            tree = tree.loc[~tree["signal_date"].isin(drop)]
        d = daily_paired_ic(kronos, tree, labels)
        d["fold"] = w.fold
        parts.append(d)

    daily = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=["signal_date", "ic_kronos", "ic_tree",
                                        "delta_ic", "n_obs", "fold"]))
    out_dir = out_root / "h3_tree_paired"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "daily_paired_ic.parquet", index=False)
    summary = h3_summary(daily)
    summary["caliber_check"] = verified
    summary["n_tree_rows_excluded"] = n_tree_excluded
    return summary
