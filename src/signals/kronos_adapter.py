"""读取 Kronos（信号 #1）已有的开发折分数，供信号 #2 做相关性与合成读数。

**只读、只碰已消耗的开发折。** 允许的折号硬编码为 :data:`ALLOWED_FOLDS`
= {1, 2, 3, 4, 36, …, 42}——这 11 折已被数十次决策读数消耗（CLAUDE.md §四），
其读数是方向性证据；折 05–35 与 2024-07 起的窗口**不得经此模块读取**，
传入即 ``ValueError``，不提供任何绕过开关。

每次打开文件前先过 ``crsp_pipeline.sealed.assert_readable``：路径若落在带哨兵的
目录之下会直接抛错。守卫在**每个文件**上调用，不做一次性检查——目录布局将来
若变，逐文件调用仍然拦得住。

路径约定（抄自 ``scripts/k8_ensemble.py``，两处必须一致）
--------------------------------------------------------
折 36–42（近代，微调 lb90 / 零样本）::

    ft: outputs/fold{f}_lb90_s0_poolB_universe/eval_amp_lb90_fold{f}/scores.parquet
    zs: outputs/zeroshot_base/eval_zeroshot_fold{f}/scores.parquet

折 01–04（早期，目录名不规则，逐折写死）::

    ft: outputs/fold0{k}_lb90_s0_poolB_universe/eval_poolB_universe[_fold0{k}]/...
    zs: outputs/zeroshot_base/eval_zs_fold0{k}/scores.parquet

标签的位置
----------
``labels.parquet`` 只在 **FT 评估目录**里（``scripts/k8_ensemble.py`` 也是从那儿读的）。
零样本与微调是同日、同名字集打的分，共用这一套标签，所以
:func:`load_labels` 默认 ``arm="ft"``——信号 #2 也必须用它，才能保证
「与 Kronos 完全同一套标签口径」。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # 允许脚本直接 import 本模块
    sys.path.insert(0, str(REPO / "src"))

from crsp_pipeline.sealed import assert_readable  # noqa: E402

DEFAULT_ROOT = REPO / "outputs"

Arm = Literal["ft", "zs"]
ARMS: tuple[str, ...] = ("ft", "zs")

#: 已消耗的开发折，唯一允许经本模块读取的折号
EARLY_FOLDS: tuple[int, ...] = (1, 2, 3, 4)
MODERN_FOLDS: tuple[int, ...] = tuple(range(36, 43))
ALLOWED_FOLDS: frozenset[int] = frozenset(EARLY_FOLDS + MODERN_FOLDS)

# 折 01–04 的不规则目录名（与 k8_ensemble.EARLY 逐字一致）
_EARLY_FT_DIRS: dict[int, str] = {
    1: "fold01_lb90_s0_poolB_universe/eval_poolB_universe",
    2: "fold02_lb90_s0_poolB_universe/eval_poolB_universe_fold02",
    3: "fold03_lb90_s0_poolB_universe/eval_poolB_universe_fold03",
    4: "fold04_lb90_s0_poolB_universe/eval_poolB_universe_fold04",
}
_EARLY_ZS_DIRS: dict[int, str] = {f: f"zeroshot_base/eval_zs_fold{f:02d}"
                                  for f in EARLY_FOLDS}

SCORE_COLUMNS = ["PERMNO", "signal_date", "score"]
LABEL_COLUMNS = ["PERMNO", "signal_date", "label", "status"]
STATUS_OK = "ok"


class FoldNotAllowedError(ValueError):
    """折号不在已消耗的开发折集合内。"""


def _check_fold(fold_id: int) -> int:
    try:
        f = int(fold_id)
    except (TypeError, ValueError):
        raise FoldNotAllowedError(f"折号必须是整数，收到 {fold_id!r}") from None
    if f not in ALLOWED_FOLDS:
        raise FoldNotAllowedError(
            f"折 {f:02d} 不在允许集合 {sorted(ALLOWED_FOLDS)} 内。"
            f"折 05–35 与 2024-07 起的窗口未被消耗，读取须用户另行授权"
            f"（CLAUDE.md §四）。")
    return f


def _check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm 必须是 {ARMS} 之一，收到 {arm!r}")
    return arm


def eval_dir(fold_id: int, arm: Arm, root: Path | str = DEFAULT_ROOT) -> Path:
    """该折该臂的评估目录（不检查是否存在）。"""
    f, a = _check_fold(fold_id), _check_arm(arm)
    root = Path(root)
    if a == "ft":
        if f in _EARLY_FT_DIRS:
            return root / _EARLY_FT_DIRS[f]
        return root / f"fold{f:02d}_lb90_s0_poolB_universe" / f"eval_amp_lb90_fold{f:02d}"
    if f in _EARLY_ZS_DIRS:
        return root / _EARLY_ZS_DIRS[f]
    return root / "zeroshot_base" / f"eval_zeroshot_fold{f:02d}"


def scores_path(fold_id: int, arm: Arm, root: Path | str = DEFAULT_ROOT) -> Path:
    return eval_dir(fold_id, arm, root) / "scores.parquet"


def labels_path(fold_id: int, arm: Arm = "ft",
                root: Path | str = DEFAULT_ROOT) -> Path:
    return eval_dir(fold_id, arm, root) / "labels.parquet"


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    assert_readable(path)          # 逐文件守卫，不做一次性检查
    df = pd.read_parquet(path, columns=columns)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def load_kronos_scores(fold_ids: Iterable[int], arm: Arm,
                       root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """读若干折的 Kronos 分数，纵向拼接。

    Returns
    -------
    DataFrame
        列恰为 ``signal_date, PERMNO, score``，按 ``(signal_date, PERMNO)`` 排序。

    Raises
    ------
    FoldNotAllowedError
        任一折号不在 :data:`ALLOWED_FOLDS` 内（检查在读任何文件**之前**完成）。
    """
    a = _check_arm(arm)
    folds = [_check_fold(f) for f in fold_ids]   # 先全部校验，再动磁盘
    frames = [_read(scores_path(f, a, root), SCORE_COLUMNS) for f in folds]
    if not frames:
        return pd.DataFrame({"signal_date": pd.Series(dtype="datetime64[ns]"),
                             "PERMNO": pd.Series(dtype="int64"),
                             "score": pd.Series(dtype="float64")})
    out = pd.concat(frames, ignore_index=True)
    return (out[["signal_date", "PERMNO", "score"]]
            .sort_values(["signal_date", "PERMNO"], kind="mergesort")
            .reset_index(drop=True))


def load_labels(fold_ids: Iterable[int], root: Path | str = DEFAULT_ROOT,
                arm: Arm = "ft", ok_only: bool = True) -> pd.DataFrame:
    """读开发折已有的 6 日执行收益标签（**不重算**，保证与 Kronos 同一口径）。

    ``label`` 在本项目里只能作为**目标或结果**，绝不可进入任何特征
    （CLAUDE.md §一.1）。本函数的调用方只应把它送进 IC 计算。
    """
    a = _check_arm(arm)
    folds = [_check_fold(f) for f in fold_ids]
    frames = [_read(labels_path(f, a, root), LABEL_COLUMNS) for f in folds]
    if not frames:
        return pd.DataFrame({"signal_date": pd.Series(dtype="datetime64[ns]"),
                             "PERMNO": pd.Series(dtype="int64"),
                             "label": pd.Series(dtype="float64"),
                             "status": pd.Series(dtype="object")})
    out = pd.concat(frames, ignore_index=True)
    if ok_only:
        out = out[out["status"] == STATUS_OK]
    return (out[["signal_date", "PERMNO", "label", "status"]]
            .sort_values(["signal_date", "PERMNO"], kind="mergesort")
            .reset_index(drop=True))
