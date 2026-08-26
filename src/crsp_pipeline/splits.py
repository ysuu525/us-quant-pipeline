"""Walk-forward 切分与测试纪律（规范 §7）。

- 训 3 年 → 验 6 个月 → 滚动；
- Purge：标签视野 6 个交易日，断言
  ``max(train_label_end) < min(val_signal_date)``，
  其中 ``train_label_end = train_signal_date + 6 个交易日``——**显式按交易
  日历加 6**，防日期索引实现差异的 off-by-one；
- 封存 OOS：最后 ≥2 年。设计冻结前禁止查看：切分器永不产出进入 OOS 的
  验证窗口；OOS 只能通过 ``sealed_oos_window(unseal=True)`` 显式解封取得。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .calendar import TradingCalendar


class PurgeViolation(AssertionError):
    pass


class SealedOOSError(RuntimeError):
    pass


@dataclass(frozen=True)
class Fold:
    """一折 walk-forward。所有端点都是**信号日**（含端点）的交易日。"""
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # 已按 purge 回退：shift(train_end, horizon) < val_start
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def assert_purged(
    train_signal_dates,
    val_signal_dates,
    calendar: TradingCalendar,
    label_horizon: int = 6,
) -> None:
    """§7 代码断言：max(train_signal + horizon 交易日) < min(val_signal)。"""
    tr = pd.DatetimeIndex(pd.to_datetime(list(train_signal_dates)))
    va = pd.DatetimeIndex(pd.to_datetime(list(val_signal_dates)))
    if len(tr) == 0 or len(va) == 0:
        return
    train_label_end = calendar.shift(tr.max(), label_horizon)
    if not train_label_end < va.min():
        raise PurgeViolation(
            f"purge violated: train_label_end={train_label_end.date()} "
            f">= min(val_signal_date)={va.min().date()}"
        )


def walk_forward_folds(
    calendar: TradingCalendar,
    start,
    end,
    train_years: int = 3,
    val_months: int = 6,
    step_months: int = 6,
    label_horizon: int = 6,
    oos_start=None,
) -> list[Fold]:
    """生成 walk-forward 折。

    - 训练窗口名义长度 train_years 年（自然年边界，起点滚动 step_months）；
    - train_end 从 val_start 按交易日历回退 label_horizon+1 日，使
      shift(train_end, horizon) = shift(val_start, -1) < val_start，恰好满足
      purge 且不多不少（off-by-one 由测试盯死）；
    - oos_start 给定时，任何 val_end 触及 oos_start（含）之后的折被丢弃——
      封存期不进入任何可见窗口。

    每折构造后都跑一遍 assert_purged（用窗口端点日），构造即验证。
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    limit = pd.Timestamp(oos_start) if oos_start is not None else None

    folds: list[Fold] = []
    k = 0
    while True:
        train_start_nom = start + pd.DateOffset(months=step_months * k)
        val_start_nom = train_start_nom + pd.DateOffset(years=train_years)
        val_end_nom = val_start_nom + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)
        if val_end_nom > end:
            break
        if limit is not None and val_end_nom >= limit:
            break

        train_start = calendar.snap_forward(train_start_nom)
        val_start = calendar.snap_forward(val_start_nom)
        val_end = calendar.snap_back(val_end_nom)
        train_end = calendar.shift(val_start, -(label_horizon + 1))

        fold = Fold(train_start, train_end, val_start, val_end)
        assert_purged(
            calendar.sessions(fold.train_start, fold.train_end),
            calendar.sessions(fold.val_start, fold.val_end),
            calendar,
            label_horizon,
        )
        folds.append(fold)
        k += 1

    return folds


def sealed_oos_window(
    calendar: TradingCalendar,
    oos_start,
    end,
    unseal: bool = False,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """封存 OOS 窗口。设计冻结前禁止查看：必须显式 unseal=True 才返回。

    解封是一次性、不可逆的决定（§7），调用方应在实验日志里记录解封时间。
    """
    if not unseal:
        raise SealedOOSError(
            "OOS window is sealed (§7). Pass unseal=True only after the design "
            "freeze; this choice is one-way."
        )
    return calendar.snap_forward(oos_start), calendar.snap_back(end)
