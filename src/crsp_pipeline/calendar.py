"""交易日历（规范 §7 / §9）。

所有「t+N 日」一律指交易日偏移，禁止自然日算术。日历从市场指数表
（daily_market_indexes，市场休市日不生成行）的日期列构建；测试用合成日期序列。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class CalendarError(ValueError):
    pass


class TradingCalendar:
    def __init__(self, dates: Iterable):
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique().sort_values()
        if len(idx) == 0:
            raise CalendarError("empty trading calendar")
        self._dates = idx
        # 值 -> 位置，用于 O(1) 严格查找
        self._pos = pd.Series(np.arange(len(idx)), index=idx)

    @classmethod
    def from_market_index(cls, df: pd.DataFrame, date_col: str = "caldt") -> "TradingCalendar":
        return cls(df[date_col])

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def __len__(self) -> int:
        return len(self._dates)

    def __contains__(self, d) -> bool:
        return pd.Timestamp(d) in self._pos.index

    def index_of(self, d) -> int:
        """d 必须是交易日，否则报错（防止拿自然日混进来悄悄错位）。"""
        ts = pd.Timestamp(d)
        try:
            return int(self._pos.loc[ts])
        except KeyError:
            raise CalendarError(f"{ts.date()} is not a trading session") from None

    def shift(self, d, n: int) -> pd.Timestamp:
        """交易日偏移：shift(t, 6) = t 之后第 6 个交易日。越界报错，不静默截断。"""
        i = self.index_of(d) + n
        if i < 0 or i >= len(self._dates):
            raise CalendarError(
                f"shift({pd.Timestamp(d).date()}, {n}) is outside the calendar range"
            )
        return self._dates[i]

    def sessions_between(self, a, b) -> int:
        """a 到 b 的交易日数（有向：b 在 a 后为正）。两端都必须是交易日。"""
        return self.index_of(b) - self.index_of(a)

    def sessions(self, start, end) -> pd.DatetimeIndex:
        """[start, end] 闭区间内的全部交易日。端点不要求是交易日。"""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self._dates[(self._dates >= s) & (self._dates <= e)]

    def snap_forward(self, d) -> pd.Timestamp:
        """d 当天或之后最近的交易日。"""
        i = int(self._dates.searchsorted(pd.Timestamp(d), side="left"))
        if i >= len(self._dates):
            raise CalendarError(f"no session on/after {pd.Timestamp(d).date()}")
        return self._dates[i]

    def snap_back(self, d) -> pd.Timestamp:
        """d 当天或之前最近的交易日。"""
        i = int(self._dates.searchsorted(pd.Timestamp(d), side="right")) - 1
        if i < 0:
            raise CalendarError(f"no session on/before {pd.Timestamp(d).date()}")
        return self._dates[i]
