"""按钱读数的薄包装层。**不引入任何新统计量。**

CLAUDE.md §五：主指标是冻结组合构造下的净夏普，RankIC 已降级为诊断。
本模块只是把已有的两样东西摆在一个门口，方便合成实验调用：

- 统计量全部复用 `crsp_pipeline.signal_eval`（`daily_rank_ic`、
  `newey_west_tstat`）——**没有第二套实现**，避免口径分叉；
- 组合构造复用 `portfolio.construction`——那份与
  `scripts/compare_arms_money.py::arm_returns` 逐位一致。

唯一新增的函数 `fold_sign_count` 也不是统计量，是 CLAUDE.md §一.3 要求的
**一致性门槛**的计数器：任何判据必须同时含「量级」与「折数同向」两项。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from crsp_pipeline.signal_eval import daily_rank_ic as _daily_rank_ic
from crsp_pipeline.signal_eval import newey_west_tstat as _newey_west_tstat
from portfolio.construction import frozen_long_only_returns, scores_frame_to_by_day

__all__ = ["daily_rank_ic", "nw_summary", "fold_sign_count", "run_frozen_money"]


def daily_rank_ic(
    scores: pd.DataFrame,
    labels: pd.DataFrame | pd.Series | None = None,
    *,
    universe_mask: Mapping | Iterable | pd.Series | None = None,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "signal_date",
) -> pd.Series:
    """逐日横截面 RankIC。**诊断指标，不作决策依据**（CLAUDE.md §五）。

    直接调用 `crsp_pipeline.signal_eval.daily_rank_ic(df, score_col, label_col,
    date_col)`，本函数只负责把两张长表拼成它要的那一张 df。

    参数
    ----
    scores : `DataFrame[signal_date, PERMNO, score]`。若已含 `label_col`，
        `labels` 可以留空。
    labels : `DataFrame[signal_date, PERMNO, label]`，按 (signal_date, PERMNO)
        **inner** 合并。**label 只能作目标或结果，绝不能回流进特征**
        （CLAUDE.md §一.1）。
    universe_mask : 三选一——
        - `None`：不过滤；
        - `Mapping[Timestamp, set[PERMNO]]`：逐日允许名单（合成实验里就是
          `combine.slow_filter` 的可持有集）；
        - 与合并后行数等长的布尔序列。
    """
    df = scores.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if labels is not None:
        lab = labels
        if isinstance(lab, pd.Series):
            lab = lab.rename(label_col).reset_index()   # 需要 (signal_date, PERMNO) MultiIndex
        lab = lab.copy()
        missing = [c for c in (date_col, "PERMNO", label_col) if c not in lab.columns]
        if missing:
            raise ValueError(f"labels 缺列 {missing}；需要 {date_col} / PERMNO / {label_col}")
        lab[date_col] = pd.to_datetime(lab[date_col])
        df = df.merge(lab[[date_col, "PERMNO", label_col]], on=[date_col, "PERMNO"], how="inner")

    if universe_mask is not None:
        if isinstance(universe_mask, Mapping):
            mask = np.fromiter(
                (p in universe_mask.get(d, ())
                 for d, p in zip(df[date_col], df["PERMNO"])),
                dtype=bool, count=len(df))
        else:
            mask = np.asarray(universe_mask, dtype=bool)
            if len(mask) != len(df):
                raise ValueError(
                    f"universe_mask 长度 {len(mask)} 与合并后行数 {len(df)} 不符")
        df = df.loc[mask]

    return _daily_rank_ic(df, score_col=score_col, label_col=label_col, date_col=date_col)


def nw_summary(x: pd.Series, lags: int = 5) -> dict:
    """Newey-West 均值检验，直通 `signal_eval.newey_west_tstat`。

    `lags=5` 是冻结值：6 日标签 + 日频信号，重叠视野 5（见该函数 docstring，
    不用自动带宽）。返回 `{"mean", "se", "t", "n"}`，原样透传，不加工。

    提醒（CLAUDE.md §二）：t 值只在 A 层（MDE ≤ SESOI）才有推断意义；
    臂间比较是 B 层，报了 t 也不得据以淘汰。
    """
    return _newey_west_tstat(x, lags)


def fold_sign_count(series_by_fold: Mapping[str, pd.Series]) -> tuple[int, int]:
    """(均值 > 0 的折数, 有效折数)。CLAUDE.md §一.3 的一致性门槛计数器。

    「有效折」= 该折至少有一个有限观测。全 NaN / 空序列的折不计入分母，也不
    计入分子——它不是「不同向」，它是没数据。

    判据只有量级门槛、没有一致性门槛是本项目 2026-08-31 犯过的错：横截面探针
    在 **2/7 折为正**的情况下仍被均值拉过门槛。用法：
    `pos, total = fold_sign_count(...)`，再按预注册的
    `pos >= max(3, ceil(0.75 * total))` 判。
    """
    pos = total = 0
    for _, v in series_by_fold.items():
        arr = pd.Series(v).dropna().to_numpy(dtype=float)
        if arr.size == 0:
            continue
        total += 1
        if arr.mean() > 0:
            pos += 1
    return pos, total


def run_frozen_money(
    scores_df: pd.DataFrame,
    ret_by_day: dict,
    oc_by_day: dict,
    adv_by_day: dict,
    *,
    topn: int = 500,
    cost_bp: float = 8.0,
    exit_pct: float = 0.30,
    nt: int = 6,
    min_names: int = 50,
) -> pd.DataFrame:
    """长表分数 → 冻结构造的日收益。`scores_frame_to_by_day` + 构造，一步到位。

    `scores_df` 是**一折**的 `DataFrame[signal_date, PERMNO, score]`。多折请
    自己循环并贴 `fold` 列——原脚本 `arm_returns` 的折循环有意留在调用方，
    因为 `book` 必须逐折重置。

    返回 `DataFrame(index=date, columns=["r", "turn", "n_names"])`。
    """
    by_day = scores_frame_to_by_day(scores_df, min_names=min_names)
    return frozen_long_only_returns(
        by_day, ret_by_day, oc_by_day, adv_by_day,
        topn=topn, cost_bp=cost_bp, exit_pct=exit_pct, nt=nt, min_names=min_names)
